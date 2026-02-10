//! Quick test of 405B progressive loading (no speculative decoding)
use std::sync::Arc;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use anyhow::Result;
use candle_core::{DType, Device};

fn main() -> Result<()> {
    println!("=== 405B Progressive Loading Test ===");

    let hct_dir = std::path::Path::new(
        "/home/crook/.cache/infernum/models/hct/meta-llama--Llama-405B-HoloTensor",
    );

    if !hct_dir.exists() {
        println!("ERROR: 405B model not found");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("Device: CUDA:0");
    println!("DType: BF16\n");

    // Full 24GB VRAM budget (no draft model competing)
    let config = TieredConfig {
        vram_budget: 22 * 1024 * 1024 * 1024, // 22GB
        ram_budget: 60 * 1024 * 1024 * 1024,  // 60GB
        min_quality: 0.7,                     // Progressive: 16/32 fragments
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    println!("Configuration:");
    println!(
        "  VRAM budget: {} GB",
        config.vram_budget / (1024 * 1024 * 1024)
    );
    println!(
        "  min_quality: {} (progressive loading!)",
        config.min_quality
    );
    println!();

    // Create loader
    let start = Instant::now();
    let loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);
    println!("Loader created in {:?}", start.elapsed());
    println!(
        "GPU acceleration: {}\n",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    // Create provider
    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Test loading a few key tensors
    println!("=== Testing Progressive Tensor Loading ===");

    let test_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight", // Layer 0 attention
        "model.layers.0.mlp.gate_proj.weight",    // Layer 0 MLP (biggest!)
    ];

    for name in &test_tensors {
        let start = Instant::now();
        match lazy_vb.get(name) {
            Ok(tensor) => {
                let elapsed = start.elapsed();
                let size_mb = tensor.elem_count() as f64 * 2.0 / (1024.0 * 1024.0); // bf16 = 2 bytes
                println!("✓ {}", name);
                println!("  Shape: {:?}", tensor.dims());
                println!("  Size: {:.1} MB", size_mb);
                println!("  Time: {:?}\n", elapsed);
            },
            Err(e) => {
                println!("✗ {}: {}\n", name, e);
            },
        }
    }

    // Print loader stats
    let stats = loader.stats();
    println!("=== Loader Statistics ===");
    println!("  Tensors loaded: {}", stats.tensors_loaded);
    println!(
        "  GPU reconstructions: {} ({} ms)",
        stats.gpu_reconstructions, stats.gpu_time_ms
    );
    println!(
        "  CPU reconstructions: {} ({} ms)",
        stats.cpu_reconstructions, stats.cpu_time_ms
    );

    if stats.gpu_reconstructions > 0 {
        println!(
            "  Avg GPU time: {:.1} ms/tensor",
            stats.gpu_time_ms as f64 / stats.gpu_reconstructions as f64
        );
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
