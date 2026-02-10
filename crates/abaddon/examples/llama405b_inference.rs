//! Test 405B HoloTensor inference with LazyLlama.
//!
//! This validates end-to-end inference with the 405B model using lazy loading.
//!
//! Run with:
//! ```bash
//! CARGO_INCREMENTAL=0 cargo run --example llama405b_inference --release --features cuda
//! ```

use std::sync::Arc;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use abaddon::models::{LazyLlama, LlamaConfig};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    println!("=== Llama 405B Inference Test ===\n");

    let hct_dir = std::path::Path::new("/tmp/llama405b-holo");

    if !hct_dir.exists() {
        println!("405B HCT directory not found: {}", hct_dir.display());
        return Ok(());
    }

    // Check CUDA availability
    let has_cuda = candle_core::utils::cuda_is_available();
    println!("CUDA available: {}", has_cuda);

    let device = if has_cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}\n", device);

    // Use F16 for GPU, F32 for CPU
    let dtype = if has_cuda { DType::F16 } else { DType::F32 };

    // Configure for 405B with memory constraints
    let config = TieredConfig {
        vram_budget: if has_cuda { 20 * 1024 * 1024 * 1024 } else { 0 },
        ram_budget: 60 * 1024 * 1024 * 1024, // 60GB RAM budget
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    println!("Memory configuration:");
    println!(
        "  VRAM budget: {} GB",
        config.vram_budget / (1024 * 1024 * 1024)
    );
    println!(
        "  RAM budget: {} GB",
        config.ram_budget / (1024 * 1024 * 1024)
    );
    println!();

    // Create the tiered loader
    println!("--- Creating TieredHoloLoader ---");
    let start = Instant::now();

    // Check for pre-converted safetensors directory (fast path)
    let safetensors_dir = std::path::Path::new("/tmp/llama405b-safetensors");
    let mut loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;

    if safetensors_dir.exists() {
        println!("Found safetensors directory: {}", safetensors_dir.display());
        loader = loader.with_safetensors_dir(safetensors_dir);
        println!("Safetensors fast-load: ENABLED (100x faster loading!)");
    } else {
        println!(
            "Safetensors directory not found: {}",
            safetensors_dir.display()
        );
        println!("  Run 'holo_to_safetensors' to pre-convert for faster loading");
    }

    let loader = Arc::new(loader);
    println!("TieredHoloLoader created in {:?}", start.elapsed());
    println!(
        "GPU acceleration: {}",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled (CPU fallback)"
        }
    );

    // Create LazyVarBuilder
    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Llama 405B config
    let model_config = LlamaConfig {
        hidden_size: 16384,
        intermediate_size: 53248,
        vocab_size: 128256,
        num_hidden_layers: 126,
        num_attention_heads: 128,
        num_key_value_heads: Some(8), // GQA with 8 KV heads
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_position_embeddings: 131072, // 128K context for Llama 405B
        tie_word_embeddings: false,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        rope_scaling: None,
    };

    // First test direct loading to verify the loader works
    println!("\n--- Testing Direct Tensor Loading ---");

    let test_tensors = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];

    for name in &test_tensors {
        let start = Instant::now();
        match lazy_vb.get(name) {
            Ok(tensor) => {
                let elapsed = start.elapsed();
                let size_mb = tensor.elem_count() as f64 * 4.0 / (1024.0 * 1024.0);
                println!(
                    "  ✓ {}: {:?} ({:.2} MB, {:.2?})",
                    name,
                    tensor.dims(),
                    size_mb,
                    elapsed
                );
            },
            Err(e) => {
                println!("  ✗ {}: {}", name, e);
            },
        }
    }

    println!("\n--- Creating LazyLlama Model ---");
    println!(
        "Config: {} layers, {} hidden, {} heads, {} KV heads",
        model_config.num_hidden_layers,
        model_config.hidden_size,
        model_config.num_attention_heads,
        model_config
            .num_key_value_heads
            .unwrap_or(model_config.num_attention_heads)
    );

    // Calculate max layers for available RAM
    // Each layer is ~3.7GB in FP8, but we're loading as F16/F32
    let bytes_per_element = match dtype {
        DType::F16 | DType::BF16 => 2,
        DType::F32 => 4,
        _ => 4,
    };
    let layer_params: u64 = 4 * 16384 * 16384 + 3 * 16384 * 53248;
    let layer_bytes = layer_params * bytes_per_element;
    let ram_budget = 60 * 1024 * 1024 * 1024u64;
    let max_layers = (ram_budget / layer_bytes) as usize;

    println!("Memory estimate:");
    println!(
        "  Per-layer size ({}): {:.2} GB",
        if bytes_per_element == 2 { "F16" } else { "F32" },
        layer_bytes as f64 / 1e9
    );
    println!(
        "  Max layers in {}GB RAM: {}",
        ram_budget / (1024 * 1024 * 1024),
        max_layers
    );

    // Limit loaded layers to what fits in RAM
    let loaded_layers = max_layers.min(20); // Start with up to 20 layers

    let start = Instant::now();
    let model_result = LazyLlama::load(model_config.clone(), lazy_vb, loaded_layers);

    match model_result {
        Ok(mut model) => {
            println!("LazyLlama created in {:?}", start.elapsed());

            // Get initial stats
            let stats = model.stats();
            println!("\nInitial model stats:");
            println!(
                "  Layers loaded: {}/{}",
                stats.loaded_layers, stats.total_layers
            );
            println!("  Max loaded layers: {}", stats.max_loaded_layers);
            println!("  Prefetch depth: {}", stats.prefetch_depth);
            println!("  Layer loads: {}", stats.layer_loads);
            println!("  Layer evictions: {}", stats.layer_evictions);

            // Warmup - prefetch initial layers
            println!("\n--- Warming up (prefetching initial layers) ---");
            let warmup_start = Instant::now();
            let warmed = model.warmup();
            println!(
                "Warmed up {} layers in {:?}",
                warmed,
                warmup_start.elapsed()
            );

            // Test forward passes to measure cache effectiveness
            println!("\n--- Testing Forward Passes (Cache Effectiveness) ---");

            // Create a simple input: [1, 4] with token IDs
            let input_ids = Tensor::new(&[1u32, 2, 3, 4], &device)?.unsqueeze(0)?; // [1, 4]

            println!("Input shape: {:?}", input_ids.dims());

            // FIRST FORWARD PASS - must reconstruct all tensors (slow)
            println!("\n[Pass 1] First forward pass (reconstruction required)...");
            let start1 = Instant::now();
            match model.forward(&input_ids, 0) {
                Ok(logits) => {
                    let elapsed1 = start1.elapsed();
                    println!("✓ Pass 1 succeeded!");
                    println!("  Output shape: {:?}", logits.dims());
                    println!("  Time: {:?}", elapsed1);

                    let stats = model.stats();
                    println!(
                        "  Layer loads: {}, evictions: {}",
                        stats.layer_loads, stats.layer_evictions
                    );

                    // SECOND FORWARD PASS - should use cached tensors (fast)
                    println!("\n[Pass 2] Second forward pass (should use cache)...");
                    let start2 = Instant::now();
                    match model.forward(&input_ids, 4) {
                        Ok(logits2) => {
                            let elapsed2 = start2.elapsed();
                            println!("✓ Pass 2 succeeded!");
                            println!("  Output shape: {:?}", logits2.dims());
                            println!("  Time: {:?}", elapsed2);

                            let stats2 = model.stats();
                            println!(
                                "  Layer loads: {}, evictions: {}",
                                stats2.layer_loads, stats2.layer_evictions
                            );

                            // Calculate speedup
                            let speedup = elapsed1.as_secs_f64() / elapsed2.as_secs_f64();
                            println!("\n=== CACHE EFFECTIVENESS ===");
                            println!("  Pass 1 (reconstruct): {:.2}s", elapsed1.as_secs_f64());
                            println!("  Pass 2 (cached):      {:.2}s", elapsed2.as_secs_f64());
                            println!("  Speedup:              {:.1}x", speedup);

                            if speedup > 10.0 {
                                println!("  ✓ Cache is working! Subsequent tokens will be fast.");
                            } else if speedup > 2.0 {
                                println!("  ~ Partial cache hit. Some tensors being reloaded.");
                            } else {
                                println!("  ✗ No speedup - cache may be evicting tensors.");
                            }
                        },
                        Err(e) => {
                            println!("✗ Pass 2 failed: {}", e);
                        },
                    }
                },
                Err(e) => {
                    println!("✗ Pass 1 failed: {}", e);
                    println!("\nThis may be expected for 405B on limited hardware.");
                    println!("The model requires progressive loading of 126 layers.");
                },
            }
        },
        Err(e) => {
            println!("Failed to create LazyLlama: {}", e);
            println!("\nThis may be due to memory constraints or missing tensors.");
        },
    }

    // Print tiered loader stats
    println!("\n--- Tiered Loader Statistics ---");
    let tiered_stats = loader.stats();
    println!("  Tensors loaded: {}", tiered_stats.tensors_loaded);

    // Safetensor fast-loads (if any)
    if tiered_stats.safetensor_loads > 0 {
        println!(
            "  Safetensor loads: {} ({} ms total)",
            tiered_stats.safetensor_loads, tiered_stats.safetensor_time_ms
        );
        println!(
            "  Avg safetensor time: {:.1} ms/tensor",
            tiered_stats.safetensor_time_ms as f64 / tiered_stats.safetensor_loads as f64
        );
    }

    // HoloTensor reconstructions
    println!(
        "  GPU reconstructions: {} ({} ms total)",
        tiered_stats.gpu_reconstructions, tiered_stats.gpu_time_ms
    );
    println!(
        "  CPU reconstructions: {} ({} ms total)",
        tiered_stats.cpu_reconstructions, tiered_stats.cpu_time_ms
    );
    if tiered_stats.gpu_reconstructions > 0 {
        println!(
            "  Avg GPU time: {:.1} ms/tensor",
            tiered_stats.gpu_time_ms as f64 / tiered_stats.gpu_reconstructions as f64
        );
    }
    if tiered_stats.cpu_reconstructions > 0 {
        println!(
            "  Avg CPU time: {:.1} ms/tensor",
            tiered_stats.cpu_time_ms as f64 / tiered_stats.cpu_reconstructions as f64
        );
    }
    if tiered_stats.gpu_reconstructions > 0 && tiered_stats.cpu_reconstructions > 0 {
        let gpu_avg = tiered_stats.gpu_time_ms as f64 / tiered_stats.gpu_reconstructions as f64;
        let cpu_avg = tiered_stats.cpu_time_ms as f64 / tiered_stats.cpu_reconstructions as f64;
        println!("  GPU speedup: {:.1}x", cpu_avg / gpu_avg);
    }

    // Show speedup from safetensors vs reconstruction
    if tiered_stats.safetensor_loads > 0 && tiered_stats.gpu_reconstructions > 0 {
        let st_avg = tiered_stats.safetensor_time_ms as f64 / tiered_stats.safetensor_loads as f64;
        let gpu_avg = tiered_stats.gpu_time_ms as f64 / tiered_stats.gpu_reconstructions as f64;
        println!("  Safetensor speedup vs GPU: {:.0}x", gpu_avg / st_avg);
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
