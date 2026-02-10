//! Optimized HoloTensor to Safetensors Converter
//!
//! High-performance conversion with:
//! - **Batched layer mode**: Reconstruct all layer tensors together, write one file per layer
//! - **Memory-safe**: Explicit CUDA sync and cleanup between tensors (fixes OOM)
//! - **CPU fallback**: Large MLP tensors (>1GB) reconstructed on CPU to avoid VRAM exhaustion
//! - Resume capability (skip already converted)
//! - Async I/O pipeline (write while reconstructing next)
//! - GPU reconstruction with haagenti-cuda zero-copy pipeline
//!
//! Run with:
//! ```bash
//! cargo run --example holo_to_safetensors --release --features cuda -- \
//!     --input /tmp/llama405b-holo \
//!     --output /tmp/llama405b-safetensors \
//!     --batch-layers
//! ```
//!
//! Options:
//! - `--input`: Path to HoloTensor directory
//! - `--output`: Path to output safetensors directory
//! - `--quality`: Reconstruction quality (0.0-1.0, default 0.95)
//! - `--batch-layers`: Batch tensors by layer (faster, recommended)
//! - `--cpu-large`: Force CPU reconstruction for tensors >1GB (default: true)
//! - `--force`: Force re-conversion even if output exists

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use crossbeam::channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use abaddon::cuda_inference::streams::device_synchronize;
use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::TensorProvider;

/// Single tensor ready for writing
struct TensorToWrite {
    name: String,
    tensor: Tensor,
    output_path: PathBuf,
}

/// Batched layer ready for writing (multiple tensors in one file)
struct LayerToWrite {
    layer_name: String,
    tensors: Vec<(String, Tensor)>,
    output_path: PathBuf,
}

/// Conversion statistics
#[derive(Default)]
struct ConversionStats {
    converted: AtomicUsize,
    skipped_exists: AtomicUsize,
    skipped_small: AtomicUsize,
    failed: AtomicUsize,
    total_bytes: AtomicU64,
    reconstruct_ms: AtomicU64,
    write_ms: AtomicU64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut input_dir = PathBuf::from("/tmp/llama405b-holo");
    let mut output_dir = PathBuf::from("/tmp/llama405b-safetensors");
    let mut quality = 0.95f32;
    let mut force = false;
    let mut batch_layers = false; // Batch all tensors per layer into one file
    let mut cpu_large = true; // CPU fallback for large tensors (>1GB) - fixes OOM
    let large_tensor_threshold: u64 = 1024 * 1024 * 1024; // 1GB

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            },
            "--output" | "-o" => {
                output_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            },
            "--quality" | "-q" => {
                quality = args[i + 1].parse()?;
                i += 2;
            },
            "--batch-layers" | "-b" => {
                batch_layers = true;
                i += 1;
            },
            "--cpu-large" => {
                cpu_large = true;
                i += 1;
            },
            "--no-cpu-large" => {
                cpu_large = false;
                i += 1;
            },
            "--force" | "-f" => {
                force = true;
                i += 1;
            },
            "--help" | "-h" => {
                print_help();
                return Ok(());
            },
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                return Ok(());
            },
        }
    }

    println!("=== Optimized HoloTensor to Safetensors Converter ===\n");
    println!("Input:       {}", input_dir.display());
    println!("Output:      {}", output_dir.display());
    println!("Quality:     {:.0}%", quality * 100.0);
    println!(
        "Mode:        {}",
        if batch_layers {
            "BATCHED (one file per layer)"
        } else {
            "Individual tensors"
        }
    );
    println!(
        "CPU large:   {} (tensors >1GB use CPU to avoid OOM)",
        cpu_large
    );
    println!("Resume:      {}", !force);
    println!();

    // Validate input directory
    if !input_dir.exists() {
        eprintln!(
            "Error: Input directory does not exist: {}",
            input_dir.display()
        );
        return Ok(());
    }

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Find all .hct files
    let hct_files = find_hct_files(&input_dir)?;
    println!("Found {} HoloTensor files", hct_files.len());

    if hct_files.is_empty() {
        println!("No .hct files found in {}", input_dir.display());
        return Ok(());
    }

    // Branch based on mode
    if batch_layers {
        return run_batched_mode(
            &input_dir,
            &output_dir,
            quality,
            force,
            cpu_large,
            large_tensor_threshold,
            &hct_files,
        );
    }

    // Individual tensor mode (legacy)
    let (large_files, small_files) = categorize_files(&hct_files, true, 1.0);

    // Check which files need conversion (resume capability)
    let files_to_convert: Vec<_> = if force {
        large_files.clone()
    } else {
        large_files
            .iter()
            .filter(|(path, _)| {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let output_path = output_dir.join(format!("{}.safetensors", name));
                !output_path.exists()
            })
            .cloned()
            .collect()
    };

    let already_done = large_files.len() - files_to_convert.len();

    println!("\nConversion plan:");
    println!("  Large tensors (will convert): {}", files_to_convert.len());
    println!("  Already converted (skip):     {}", already_done);
    println!("  Small tensors (skip):         {}", small_files.len());
    println!();

    if files_to_convert.is_empty() {
        println!("Nothing to convert! All large tensors already exist.");
        println!("Use --force to re-convert.");
        return Ok(());
    }

    // Setup device
    // CRITICAL: With --cpu-large, use CPU device to avoid VRAM OOM
    // CPU reconstruction is slower but memory-safe for large tensors
    let has_cuda = candle_core::utils::cuda_is_available();
    let device = if cpu_large || !has_cuda {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    println!("Device: {:?}", device);
    println!("CUDA available: {} (using CPU for --cpu-large)", has_cuda);

    // Create tiered loader
    let dtype = if has_cuda { DType::F16 } else { DType::F32 };
    let config = TieredConfig {
        vram_budget: if has_cuda { 20 * 1024 * 1024 * 1024 } else { 0 },
        ram_budget: 60 * 1024 * 1024 * 1024,
        min_quality: quality,
        target_quality: quality,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = Arc::new(TieredHoloLoader::new(
        &input_dir,
        config,
        device.clone(),
        dtype,
    )?);
    println!("TieredHoloLoader created");
    println!(
        "GPU acceleration: {}\n",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    // Stats
    let stats = Arc::new(ConversionStats::default());
    stats.skipped_exists.store(already_done, Ordering::Relaxed);
    stats
        .skipped_small
        .store(small_files.len(), Ordering::Relaxed);

    // Create async I/O pipeline
    // Channel: reconstruction threads -> writer thread
    // Channel size = 15 to handle 12 parallel reconstructions + some buffer
    let (tx, rx): (Sender<TensorToWrite>, Receiver<TensorToWrite>) = bounded(15);

    // Start writer thread (async I/O)
    let stats_writer = Arc::clone(&stats);
    let writer_handle = thread::spawn(move || {
        writer_thread(rx, stats_writer);
    });

    // Sort files by size (largest first for better GPU utilization)
    let mut files_sorted = files_to_convert.clone();
    files_sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let total = files_sorted.len();
    let start = Instant::now();

    println!("--- Converting {} tensors (largest first) ---\n", total);
    println!(
        "(PARALLEL mode - {} threads, CPU reconstruction avoids CUDA context issues)\n",
        rayon::current_num_threads()
    );

    // Process tensors in PARALLEL (safe with CPU reconstruction)
    // Ultra-conservative concurrency: 4 threads = 16GB max working set
    rayon::ThreadPoolBuilder::new()
        .num_threads(4) // 4 parallel reconstructions (4 * 4GB = 16GB max)
        .build()
        .unwrap()
        .install(|| {
            let processed = AtomicUsize::new(0);
            let stats_recon = Arc::clone(&stats);

            files_sorted
                .par_iter()
                .for_each(|(hct_path, _estimated_size)| {
                    let tensor_name = hct_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let output_path = output_dir.join(format!("{}.safetensors", tensor_name));

                    let idx = processed.fetch_add(1, Ordering::Relaxed);

                    // Reconstruct tensor
                    let recon_start = Instant::now();
                    match loader.get(&tensor_name, &device, dtype) {
                        Ok(tensor) => {
                            let recon_ms = recon_start.elapsed().as_millis() as u64;
                            stats_recon
                                .reconstruct_ms
                                .fetch_add(recon_ms, Ordering::Relaxed);

                            let size_mb = tensor.elem_count() as f64 * dtype.size_in_bytes() as f64
                                / (1024.0 * 1024.0);

                            // Print progress
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = (idx + 1) as f64 / elapsed;
                            let remaining = (total - idx - 1) as f64 / rate;

                            println!(
                                "[{}/{}] {} ({:.1} MB, {:.1}s recon) ETA: {:.0}m",
                                idx + 1,
                                total,
                                tensor_name,
                                size_mb,
                                recon_ms as f64 / 1000.0,
                                remaining / 60.0
                            );

                            // CRITICAL: Move to CPU here (main thread has CUDA context)
                            // Writer thread cannot access CUDA tensors (thread-local context)
                            let tensor_cpu = match tensor.to_device(&Device::Cpu) {
                                Ok(t) => t,
                                Err(e) => {
                                    eprintln!(
                                        "[{}/{}] {} CPU transfer FAILED: {}",
                                        idx + 1,
                                        total,
                                        tensor_name,
                                        e
                                    );
                                    stats_recon.failed.fetch_add(1, Ordering::Relaxed);
                                    return; // Exit this iteration
                                },
                            };

                            // Send to writer thread
                            if tx
                                .send(TensorToWrite {
                                    name: tensor_name,
                                    tensor: tensor_cpu,
                                    output_path,
                                })
                                .is_err()
                            {
                                eprintln!("Writer channel closed");
                            }
                        },
                        Err(e) => {
                            eprintln!("[{}/{}] {} FAILED: {}", idx + 1, total, tensor_name, e);
                            stats_recon.failed.fetch_add(1, Ordering::Relaxed);
                        },
                    }
                }); // end par_iter
        }); // end thread pool install

    // Close channel and wait for writer
    drop(tx);
    writer_handle.join().ok();

    let elapsed = start.elapsed();

    // Print summary
    println!("\n{}", "=".repeat(60));
    println!("CONVERSION COMPLETE");
    println!("{}", "=".repeat(60));
    println!(
        "Converted:        {} tensors",
        stats.converted.load(Ordering::Relaxed)
    );
    println!(
        "Skipped (exists): {} tensors",
        stats.skipped_exists.load(Ordering::Relaxed)
    );
    println!(
        "Skipped (small):  {} tensors",
        stats.skipped_small.load(Ordering::Relaxed)
    );
    println!(
        "Failed:           {} tensors",
        stats.failed.load(Ordering::Relaxed)
    );
    println!();
    println!(
        "Total time:       {:.1} minutes",
        elapsed.as_secs_f64() / 60.0
    );
    println!(
        "Total written:    {:.2} GB",
        stats.total_bytes.load(Ordering::Relaxed) as f64 / 1e9
    );
    println!();
    println!("Timing breakdown:");
    println!(
        "  Reconstruction: {:.1} s",
        stats.reconstruct_ms.load(Ordering::Relaxed) as f64 / 1000.0
    );
    println!(
        "  Writing:        {:.1} s",
        stats.write_ms.load(Ordering::Relaxed) as f64 / 1000.0
    );
    println!();
    println!(
        "Throughput: {:.1} tensors/min",
        stats.converted.load(Ordering::Relaxed) as f64 / (elapsed.as_secs_f64() / 60.0)
    );
    println!();
    println!("Output: {}", output_dir.display());

    // Print loader stats
    let loader_stats = loader.stats();
    println!("\nReconstruction stats:");
    println!(
        "  GPU reconstructions: {}",
        loader_stats.gpu_reconstructions
    );
    println!(
        "  CPU reconstructions: {}",
        loader_stats.cpu_reconstructions
    );
    if loader_stats.gpu_reconstructions > 0 {
        println!(
            "  Avg GPU time: {:.1} ms",
            loader_stats.gpu_time_ms as f64 / loader_stats.gpu_reconstructions as f64
        );
    }

    Ok(())
}

/// Batched layer mode: group tensors by layer, write one file per layer
///
/// Memory-safe implementation:
/// - Creates separate GPU and CPU loaders
/// - Uses CPU for large tensors (>threshold) to avoid OOM
/// - Explicit CUDA sync after each tensor reconstruction
/// - Writes and drops tensors immediately to free VRAM
fn run_batched_mode(
    input_dir: &Path,
    output_dir: &Path,
    quality: f32,
    force: bool,
    cpu_large: bool,
    large_tensor_threshold: u64,
    hct_files: &[PathBuf],
) -> Result<()> {
    // Group files by layer
    let mut layers: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let mut non_layer_files: Vec<PathBuf> = Vec::new();

    for path in hct_files {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        if let Some(layer_name) = extract_layer_name(name) {
            layers.entry(layer_name).or_default().push(path.clone());
        } else {
            // Non-layer files (embed_tokens, lm_head, model_norm)
            non_layer_files.push(path.clone());
        }
    }

    let total_layers = layers.len();
    let total_non_layer = non_layer_files.len();

    println!("\nBatched conversion plan:");
    println!(
        "  Layers:      {} (each with {} tensors avg)",
        total_layers,
        if total_layers > 0 {
            hct_files.len().saturating_sub(total_non_layer) / total_layers
        } else {
            0
        }
    );
    println!("  Non-layer:   {} (embed, lm_head, etc.)", total_non_layer);

    // Check which layers need conversion
    let mut layers_to_convert: Vec<(String, Vec<PathBuf>)> = Vec::new();
    let mut skipped = 0;

    for (layer_name, files) in layers.iter() {
        let output_path = output_dir.join(format!("{}.safetensors", layer_name));
        if !force && output_path.exists() {
            skipped += 1;
        } else {
            layers_to_convert.push((layer_name.clone(), files.clone()));
        }
    }

    // Also check non-layer files
    let non_layer_to_convert: Vec<_> = if force {
        non_layer_files.clone()
    } else {
        non_layer_files
            .iter()
            .filter(|path| {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let output_path = output_dir.join(format!("{}.safetensors", name));
                !output_path.exists()
            })
            .cloned()
            .collect()
    };

    let non_layer_skipped = non_layer_files.len() - non_layer_to_convert.len();

    println!(
        "  To convert:  {} layers + {} non-layer files",
        layers_to_convert.len(),
        non_layer_to_convert.len()
    );
    println!(
        "  Skipped:     {} layers + {} non-layer files (already exist)",
        skipped, non_layer_skipped
    );
    println!();

    if layers_to_convert.is_empty() && non_layer_to_convert.is_empty() {
        println!("Nothing to convert! Use --force to re-convert.");
        return Ok(());
    }

    // Setup devices - BOTH GPU and CPU loaders for fallback
    let has_cuda = candle_core::utils::cuda_is_available();
    let gpu_device = if has_cuda {
        Some(Device::new_cuda(0)?)
    } else {
        None
    };
    let cpu_device = Device::Cpu;

    println!("CUDA available: {}", has_cuda);
    if cpu_large && has_cuda {
        println!(
            "CPU fallback:   enabled for tensors >{}MB",
            large_tensor_threshold / (1024 * 1024)
        );
    }

    // Create GPU loader (for smaller tensors)
    let dtype_gpu = DType::F16;
    let dtype_cpu = DType::F32;

    let gpu_config = TieredConfig {
        vram_budget: if has_cuda { 16 * 1024 * 1024 * 1024 } else { 0 }, // 16GB VRAM budget
        ram_budget: 32 * 1024 * 1024 * 1024,
        min_quality: quality,
        target_quality: quality,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let cpu_config = TieredConfig {
        vram_budget: 0, // Force CPU
        ram_budget: 60 * 1024 * 1024 * 1024,
        min_quality: quality,
        target_quality: quality,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let gpu_loader = if has_cuda {
        Some(Arc::new(TieredHoloLoader::new(
            input_dir,
            gpu_config,
            gpu_device.clone().unwrap(),
            dtype_gpu,
        )?))
    } else {
        None
    };

    let cpu_loader = Arc::new(TieredHoloLoader::new(
        input_dir,
        cpu_config,
        cpu_device.clone(),
        dtype_cpu,
    )?);

    println!(
        "GPU acceleration: {}",
        gpu_loader
            .as_ref()
            .map_or("disabled", |l| if l.is_gpu_enabled() {
                "enabled"
            } else {
                "disabled"
            })
    );
    println!();

    let start = Instant::now();
    let mut converted_layers = 0;
    let mut converted_tensors = 0;
    let mut failed = 0;
    let mut total_bytes: u64 = 0;
    let mut gpu_recons = 0;
    let mut cpu_recons = 0;

    // Sort layers by number (layer_0, layer_1, ...)
    layers_to_convert.sort_by(|a, b| {
        let num_a: usize =
            a.0.split('_')
                .last()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        let num_b: usize =
            b.0.split('_')
                .last()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        num_a.cmp(&num_b)
    });

    let total = layers_to_convert.len() + non_layer_to_convert.len();

    println!("--- Converting {} items ---\n", total);

    // Helper: decide which loader to use based on tensor size
    // HYBRID MODE: Use GPU for small tensors, CPU for large ones to avoid VRAM OOM
    // MLP weights are 1.7GB uncompressed but only 35MB in HCT (50:1 ratio)
    let should_use_cpu = |path: &Path, tensor_name: &str| -> bool {
        if !has_cuda {
            return true; // No GPU available
        }

        // ALWAYS use CPU for known large tensors (regardless of cpu_large flag)
        // MLP projection weights: ~1.7GB each (3 per layer = 5GB)
        if tensor_name.contains("mlp")
            && (tensor_name.contains("down_proj")
                || tensor_name.contains("gate_proj")
                || tensor_name.contains("up_proj"))
        {
            return true;
        }

        // Attention Q/K/V/O projections: ~500MB-1GB each
        if tensor_name.contains("self_attn") && tensor_name.contains("_proj_weight") {
            // Use CPU for these to be safe
            return true;
        }

        // Check file size for other tensors
        let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // HCT files compress ~50:1, so 10MB HCT = ~500MB uncompressed
        // Use CPU for anything that will be >500MB uncompressed
        let estimated_size = file_size * 50;
        let cpu_threshold = 500 * 1024 * 1024; // 500MB

        estimated_size > cpu_threshold
    };

    // Convert non-layer files first (embed_tokens, lm_head are large)
    for (idx, path) in non_layer_to_convert.iter().enumerate() {
        let tensor_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let output_path = output_dir.join(format!("{}.safetensors", tensor_name));

        let use_cpu = should_use_cpu(path, tensor_name);
        let recon_start = Instant::now();

        let result = if use_cpu {
            cpu_recons += 1;
            cpu_loader.get(tensor_name, &cpu_device, dtype_cpu)
        } else {
            gpu_recons += 1;
            gpu_loader
                .as_ref()
                .unwrap()
                .get(tensor_name, gpu_device.as_ref().unwrap(), dtype_gpu)
        };

        match result {
            Ok(tensor) => {
                let recon_ms = recon_start.elapsed().as_millis();
                let size_mb = tensor.elem_count() as f64 * tensor.dtype().size_in_bytes() as f64
                    / (1024.0 * 1024.0);

                // Write single tensor
                match save_safetensor_fast(&tensor, tensor_name, &output_path) {
                    Ok(bytes) => {
                        total_bytes += bytes as u64;
                        converted_tensors += 1;

                        let elapsed = start.elapsed().as_secs_f64();
                        let rate = (idx + 1) as f64 / elapsed;
                        let remaining = (total - idx - 1) as f64 / rate;

                        let device_tag = if use_cpu { "CPU" } else { "GPU" };
                        println!(
                            "[{}/{}] {} ({:.1} MB, {:.1}s, {}) ETA: {:.0}m",
                            idx + 1,
                            total,
                            tensor_name,
                            size_mb,
                            recon_ms as f64 / 1000.0,
                            device_tag,
                            remaining / 60.0
                        );
                    },
                    Err(e) => {
                        eprintln!(
                            "[{}/{}] {} WRITE FAILED: {}",
                            idx + 1,
                            total,
                            tensor_name,
                            e
                        );
                        failed += 1;
                    },
                }

                // CRITICAL: Explicit cleanup to free VRAM
                drop(tensor);
                #[cfg(feature = "cuda")]
                if has_cuda && !use_cpu {
                    let _ = device_synchronize();
                }
            },
            Err(e) => {
                eprintln!("[{}/{}] {} FAILED: {}", idx + 1, total, tensor_name, e);
                failed += 1;
            },
        }
    }

    let non_layer_offset = non_layer_to_convert.len();

    // Convert layers (batched) - reconstruct one tensor at a time, write immediately
    for (idx, (layer_name, tensor_paths)) in layers_to_convert.iter().enumerate() {
        let output_path = output_dir.join(format!("{}.safetensors", layer_name));
        let layer_start = Instant::now();

        // Reconstruct tensors ONE AT A TIME and immediately move to CPU
        // This prevents VRAM from filling up with multiple large tensors
        let mut layer_tensors: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();
        let mut layer_failed = 0;
        let mut layer_size: u64 = 0;

        for path in tensor_paths {
            let tensor_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let use_cpu = should_use_cpu(path, tensor_name);

            let result = if use_cpu {
                cpu_recons += 1;
                cpu_loader.get(tensor_name, &cpu_device, dtype_cpu)
            } else {
                gpu_recons += 1;
                gpu_loader.as_ref().unwrap().get(
                    tensor_name,
                    gpu_device.as_ref().unwrap(),
                    dtype_gpu,
                )
            };

            match result {
                Ok(tensor) => {
                    // IMMEDIATELY convert to CPU bytes to free VRAM
                    let tensor_cpu = tensor.to_device(&Device::Cpu)?;

                    // Extract data and metadata
                    let (data, dtype_str) = match tensor_cpu.dtype() {
                        DType::F16 => {
                            let data: Vec<half::f16> = tensor_cpu.flatten_all()?.to_vec1()?;
                            (bytemuck::cast_slice(&data).to_vec(), "F16")
                        },
                        DType::F32 => {
                            let data: Vec<f32> = tensor_cpu.flatten_all()?.to_vec1()?;
                            (bytemuck::cast_slice(&data).to_vec(), "F32")
                        },
                        DType::BF16 => {
                            let data: Vec<half::bf16> = tensor_cpu.flatten_all()?.to_vec1()?;
                            (bytemuck::cast_slice(&data).to_vec(), "BF16")
                        },
                        _ => {
                            let t32 = tensor_cpu.to_dtype(DType::F32)?;
                            let data: Vec<f32> = t32.flatten_all()?.to_vec1()?;
                            (bytemuck::cast_slice(&data).to_vec(), "F32")
                        },
                    };

                    let shape: Vec<usize> = tensor_cpu.dims().to_vec();
                    layer_size += data.len() as u64;
                    layer_tensors.push((
                        tensor_name.to_string(),
                        data,
                        dtype_str.to_string(),
                        shape,
                    ));

                    // CRITICAL: Explicit cleanup to free VRAM immediately
                    drop(tensor_cpu);
                    drop(tensor);

                    #[cfg(feature = "cuda")]
                    if has_cuda {
                        // Force CUDA to release memory NOW
                        let _ = device_synchronize();
                        // Small sleep to ensure cleanup completes
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                },
                Err(e) => {
                    eprintln!("  {} FAILED: {}", tensor_name, e);
                    layer_failed += 1;
                },
            }
        }

        let recon_ms = layer_start.elapsed().as_millis();
        let layer_tensors_count = layer_tensors.len();

        if layer_tensors.is_empty() {
            eprintln!(
                "[{}/{}] {} - all tensors failed!",
                non_layer_offset + idx + 1,
                total,
                layer_name
            );
            failed += 1;
            continue;
        }

        // Report if any tensors failed
        if layer_failed > 0 {
            eprintln!(
                "  WARNING: {} tensors failed in {}",
                layer_failed, layer_name
            );
        }

        // Write batched layer file from pre-extracted bytes
        match save_layer_safetensor_from_bytes(&layer_tensors, &output_path) {
            Ok(bytes) => {
                total_bytes += bytes as u64;
                converted_layers += 1;
                converted_tensors += layer_tensors_count;

                let size_mb = bytes as f64 / (1024.0 * 1024.0);
                let elapsed = start.elapsed().as_secs_f64();
                let current = non_layer_offset + idx + 1;
                let rate = current as f64 / elapsed;
                let remaining = (total - current) as f64 / rate;

                println!(
                    "[{}/{}] {} ({} tensors, {:.1} MB, {:.1}s) ETA: {:.0}m",
                    current,
                    total,
                    layer_name,
                    layer_tensors_count,
                    size_mb,
                    recon_ms as f64 / 1000.0,
                    remaining / 60.0
                );
            },
            Err(e) => {
                eprintln!(
                    "[{}/{}] {} WRITE FAILED: {}",
                    non_layer_offset + idx + 1,
                    total,
                    layer_name,
                    e
                );
                failed += 1;
            },
        }

        // Clear layer tensors (free RAM)
        drop(layer_tensors);
    }

    let elapsed = start.elapsed();

    println!("\n{}", "=".repeat(60));
    println!("BATCHED CONVERSION COMPLETE");
    println!("{}", "=".repeat(60));
    println!("Layers converted: {}", converted_layers);
    println!("Tensors total:    {}", converted_tensors);
    println!("Failed:           {}", failed);
    println!();
    println!("Reconstruction breakdown:");
    println!("  GPU:            {} tensors", gpu_recons);
    println!("  CPU (large):    {} tensors", cpu_recons);
    println!();
    println!(
        "Total time:       {:.1} minutes",
        elapsed.as_secs_f64() / 60.0
    );
    println!("Total written:    {:.2} GB", total_bytes as f64 / 1e9);
    println!(
        "Throughput:       {:.1} layers/min",
        converted_layers as f64 / (elapsed.as_secs_f64() / 60.0)
    );
    println!();
    println!("Output: {}", output_dir.display());

    Ok(())
}

/// Save layer from pre-extracted byte data (avoids holding tensors in memory)
fn save_layer_safetensor_from_bytes(
    tensors: &[(String, Vec<u8>, String, Vec<usize>)],
    path: &Path,
) -> Result<usize> {
    // Build header
    let mut header_parts: Vec<String> = Vec::new();
    let mut offset = 0usize;

    for (name, data, dtype, shape) in tensors {
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let end_offset = offset + data.len();

        header_parts.push(format!(
            r#""{}": {{"dtype": "{}", "shape": [{}], "data_offsets": [{}, {}]}}"#,
            name,
            dtype,
            shape_str.join(", "),
            offset,
            end_offset
        ));

        offset = end_offset;
    }

    let header = format!("{{{}}}", header_parts.join(", "));

    // Write file
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16MB buffer

    // Header length
    let header_len = header.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;

    // Header
    writer.write_all(header.as_bytes())?;

    // All tensor data
    for (_, data, _, _) in tensors {
        writer.write_all(data)?;
    }

    writer.flush()?;

    Ok(8 + header.len() + offset)
}

/// Extract layer name from tensor name (e.g., "model_layers_42_mlp_down_proj_weight" -> "layer_42")
fn extract_layer_name(name: &str) -> Option<String> {
    if name.starts_with("model_layers_") {
        // Parse layer number
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() >= 3 {
            if let Ok(layer_num) = parts[2].parse::<usize>() {
                return Some(format!("layer_{}", layer_num));
            }
        }
    }
    None
}

/// Save multiple tensors to a single safetensors file
fn save_layer_safetensor(tensors: &[(String, Tensor)], path: &Path) -> Result<usize> {
    let mut tensor_data: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();

    for (name, tensor) in tensors {
        let tensor = tensor.to_device(&Device::Cpu)?;

        let data = match tensor.dtype() {
            DType::F16 => {
                let data: Vec<half::f16> = tensor.flatten_all()?.to_vec1()?;
                bytemuck::cast_slice(&data).to_vec()
            },
            DType::F32 => {
                let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
                bytemuck::cast_slice(&data).to_vec()
            },
            DType::BF16 => {
                let data: Vec<half::bf16> = tensor.flatten_all()?.to_vec1()?;
                bytemuck::cast_slice(&data).to_vec()
            },
            _ => {
                let tensor = tensor.to_dtype(DType::F32)?;
                let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
                bytemuck::cast_slice(&data).to_vec()
            },
        };

        let dtype_str = match tensor.dtype() {
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::BF16 => "BF16",
            _ => "F32",
        };

        let shape: Vec<usize> = tensor.dims().to_vec();
        tensor_data.push((name.clone(), data, dtype_str.to_string(), shape));
    }

    // Build header
    let mut header_parts: Vec<String> = Vec::new();
    let mut offset = 0usize;

    for (name, data, dtype, shape) in &tensor_data {
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let end_offset = offset + data.len();

        header_parts.push(format!(
            r#""{}": {{"dtype": "{}", "shape": [{}], "data_offsets": [{}, {}]}}"#,
            name,
            dtype,
            shape_str.join(", "),
            offset,
            end_offset
        ));

        offset = end_offset;
    }

    let header = format!("{{{}}}", header_parts.join(", "));

    // Write file
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

    // Header length
    let header_len = header.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;

    // Header
    writer.write_all(header.as_bytes())?;

    // All tensor data
    for (_, data, _, _) in &tensor_data {
        writer.write_all(data)?;
    }

    writer.flush()?;

    Ok(8 + header.len() + offset)
}

/// Writer thread - handles async I/O while reconstruction continues
fn writer_thread(rx: Receiver<TensorToWrite>, stats: Arc<ConversionStats>) {
    while let Ok(item) = rx.recv() {
        let write_start = Instant::now();

        match save_safetensor_fast(&item.tensor, &item.name, &item.output_path) {
            Ok(bytes) => {
                let write_ms = write_start.elapsed().as_millis() as u64;
                stats.write_ms.fetch_add(write_ms, Ordering::Relaxed);
                stats.total_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
                stats.converted.fetch_add(1, Ordering::Relaxed);
            },
            Err(e) => {
                eprintln!("Write failed for {}: {}", item.name, e);
                stats.failed.fetch_add(1, Ordering::Relaxed);
            },
        }
    }
}

/// Fast safetensor writing with buffered I/O
/// ASSUMES tensor is already on CPU (moved by main thread)
fn save_safetensor_fast(tensor: &Tensor, name: &str, path: &Path) -> Result<usize> {
    // Tensor should already be on CPU (moved in main thread)
    // Do NOT call to_device here - writer thread has no CUDA context

    // Get raw data
    let data = match tensor.dtype() {
        DType::F16 => {
            let data: Vec<half::f16> = tensor.flatten_all()?.to_vec1()?;
            bytemuck::cast_slice(&data).to_vec()
        },
        DType::F32 => {
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            bytemuck::cast_slice(&data).to_vec()
        },
        DType::BF16 => {
            let data: Vec<half::bf16> = tensor.flatten_all()?.to_vec1()?;
            bytemuck::cast_slice(&data).to_vec()
        },
        _ => {
            let tensor = tensor.to_dtype(DType::F32)?;
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            bytemuck::cast_slice(&data).to_vec()
        },
    };

    let dtype_str = match tensor.dtype() {
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::BF16 => "BF16",
        _ => "F32",
    };

    let shape: Vec<usize> = tensor.dims().to_vec();
    let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();

    let header = format!(
        r#"{{"{}": {{"dtype": "{}", "shape": [{}], "data_offsets": [0, {}]}}}}"#,
        name,
        dtype_str,
        shape_str.join(", "),
        data.len()
    );

    // Use buffered writer for better I/O performance
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer

    // Write header length (8 bytes, little endian)
    let header_len = header.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;

    // Write header
    writer.write_all(header.as_bytes())?;

    // Write data
    writer.write_all(&data)?;
    writer.flush()?;

    Ok(8 + header.len() + data.len())
}

fn find_hct_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "hct") {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

/// Categorize files into large (need conversion) and small (skip)
fn categorize_files(
    files: &[PathBuf],
    skip_small: bool,
    threshold_mb: f64,
) -> (Vec<(PathBuf, u64)>, Vec<PathBuf>) {
    let threshold_bytes = (threshold_mb * 1024.0 * 1024.0) as u64;

    let mut large = Vec::new();
    let mut small = Vec::new();

    for path in files {
        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Skip small tensors if enabled
        let is_small = skip_small && (size < threshold_bytes || is_small_tensor_name(name));

        if is_small {
            small.push(path.clone());
        } else {
            large.push((path.clone(), size));
        }
    }

    (large, small)
}

/// Check if tensor name indicates a small tensor (norms, scales, biases)
fn is_small_tensor_name(name: &str) -> bool {
    name.contains("layernorm")
        || name.contains("_norm")
        || name.contains("_scale")
        || name.contains("input_scale")
        || name.contains("_bias")
        || name.ends_with("bias")
        || name.contains("rotary")
        || name == "model_norm_weight"
}

fn print_help() {
    println!("Optimized HoloTensor to Safetensors Converter");
    println!();
    println!("Usage: holo_to_safetensors [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -i, --input <DIR>       Input HoloTensor directory");
    println!("  -o, --output <DIR>      Output safetensors directory");
    println!("  -q, --quality <FLOAT>   Reconstruction quality (0.0-1.0)");
    println!("  -b, --batch-layers      Batch tensors by layer (RECOMMENDED)");
    println!("  --cpu-large             Use CPU for tensors >1GB (default: enabled)");
    println!("  --no-cpu-large          Disable CPU fallback (may cause OOM)");
    println!("  -f, --force             Force re-conversion even if exists");
    println!("  -h, --help              Show this help");
    println!();
    println!("Modes:");
    println!("  Default:      Individual tensor files (legacy)");
    println!("  --batch-layers: One file per layer (~25 tensors each)");
    println!();
    println!("Memory-safe features:");
    println!("  - CPU fallback: Large tensors (MLP >1GB) use CPU to avoid VRAM OOM");
    println!("  - Explicit cleanup: CUDA sync after each tensor frees VRAM immediately");
    println!("  - Streaming write: Tensors converted to bytes before batching");
    println!();
    println!("Other features:");
    println!("  - Resume: skips already-converted layers/tensors");
    println!("  - Batched: 126 layer files instead of 884 tensor files");
    println!("  - Efficient I/O: fewer files, larger sequential writes");
}
