//! Quality verification for real HCT models.
//!
//! This example loads HCT files and verifies reconstruction quality.

use std::path::Path;
use std::time::Instant;

use abaddon::{load_hct_directory_sequential, TensorProvider, TieredConfig, TieredHoloLoader};
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/qwen2.5-32b-int4-v2".to_string());

    println!("=== HoloTensor Quality Verification ===\n");
    println!("Model directory: {}", model_dir);

    let path = Path::new(&model_dir);
    if !path.exists() {
        anyhow::bail!("Directory not found: {}", model_dir);
    }

    // Count files
    let hct_files: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "hct"))
        .collect();

    println!("HCT files found: {}\n", hct_files.len());

    // Test 1: Sequential loading
    println!("--- Test 1: Sequential Loading ---");
    let start = Instant::now();
    let tensors = load_hct_directory_sequential(path, &Device::Cpu, DType::F32)?;
    let seq_time = start.elapsed();
    println!("Loaded {} tensors in {:?}", tensors.len(), seq_time);

    // Calculate total size
    let total_elements: usize = tensors.values().map(|t| t.elem_count()).sum();
    let total_bytes = total_elements * 4; // F32
    println!(
        "Total elements: {} ({:.2} GB)",
        total_elements,
        total_bytes as f64 / 1e9
    );

    // Sample some tensor stats
    println!("\nSample tensor statistics:");
    for (name, tensor) in tensors.iter().take(5) {
        let dims = tensor.dims();
        let mean = tensor.mean_all()?.to_scalar::<f32>()?;
        let var = tensor.var(0)?.mean_all()?.to_scalar::<f32>()?;
        println!("  {}: {:?} mean={:.4} var={:.4}", name, dims, mean, var);
    }

    // Test 2: Tiered loading
    println!("\n--- Test 2: Tiered Loading ---");
    let config = TieredConfig {
        vram_budget: 8 * 1024 * 1024 * 1024, // 8GB
        ram_budget: 32 * 1024 * 1024 * 1024, // 32GB
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let start = Instant::now();
    let loader = TieredHoloLoader::new(path, config, Device::Cpu, DType::F32)?;
    let init_time = start.elapsed();
    println!("Loader initialized in {:?}", init_time);

    let tensor_names = loader.tensor_names();
    println!("Tensor count: {}", tensor_names.len());

    // Load a few key tensors and check quality
    let key_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ];

    println!("\nKey tensor loading:");
    for name in key_tensors {
        let start = Instant::now();
        match loader.get(name, &Device::Cpu, DType::F32) {
            Ok(tensor) => {
                let load_time = start.elapsed();
                let dims = tensor.dims();
                let mean = tensor.mean_all()?.to_scalar::<f32>()?;
                println!(
                    "  {} {:?} loaded in {:?} (mean={:.4})",
                    name, dims, load_time, mean
                );
            },
            Err(e) => {
                println!("  {} FAILED: {}", name, e);
            },
        }
    }

    // Test 3: Compare sequential vs tiered for same tensor
    println!("\n--- Test 3: Quality Comparison ---");
    if let (Some(seq_embed), Ok(tiered_embed)) = (
        tensors.get("model.embed_tokens.weight"),
        loader.get("model.embed_tokens.weight", &Device::Cpu, DType::F32),
    ) {
        // Check if they match
        let seq_mean = seq_embed.mean_all()?.to_scalar::<f32>()?;
        let tiered_mean = tiered_embed.mean_all()?.to_scalar::<f32>()?;

        println!("Embedding comparison:");
        println!("  Sequential mean: {:.6}", seq_mean);
        println!("  Tiered mean:     {:.6}", tiered_mean);
        println!("  Difference:      {:.6}", (seq_mean - tiered_mean).abs());

        // Compute element-wise difference
        let diff = (seq_embed - &tiered_embed)?.abs()?;
        let max_diff = diff.max(0)?.max(0)?.to_scalar::<f32>()?;
        let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

        println!("  Max difference:  {:.6}", max_diff);
        println!("  Mean difference: {:.6}", mean_diff);

        // Quality metric (1 - relative error)
        let seq_norm = seq_embed.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let diff_norm = diff.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let quality = 1.0 - (diff_norm / seq_norm);

        println!("\n  QUALITY: {:.2}%", quality * 100.0);

        if quality >= 0.95 {
            println!("  ✓ PASS: Quality >= 95%");
        } else if quality >= 0.70 {
            println!("  ~ MARGINAL: Quality between 70-95%");
        } else {
            println!("  ✗ FAIL: Quality < 70%");
        }
    }

    let stats = loader.stats();
    println!("\n--- Final Stats ---");
    println!("  Tensors loaded: {}", stats.tensors_loaded);
    println!("  VRAM tensors: {}", stats.vram_tensors);
    println!("  RAM tensors: {}", stats.ram_tensors);

    println!("\n=== Verification Complete ===");

    Ok(())
}
