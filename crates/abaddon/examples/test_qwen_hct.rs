//! Quick test for HCT loading with converted Qwen 0.5B model.
//!
//! Run with:
//! ```bash
//! cargo run --example test_qwen_hct --release
//! ```

use std::sync::Arc;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use anyhow::Result;
use candle_core::{DType, Device};

fn main() -> Result<()> {
    println!("=== Qwen 0.5B HCT Loading Test ===\n");

    // Try LZ4 version first (Zstd has known interop issues)
    let hct_dir = std::path::Path::new("/tmp/qwen_0.5b_lz4");

    if !hct_dir.exists() {
        println!("HCT directory not found: {}", hct_dir.display());
        println!("Run safetensors_to_hct converter first.");
        return Ok(());
    }

    let device = Device::Cpu;
    let dtype = DType::BF16; // Match the original model dtype

    println!("HCT directory: {}", hct_dir.display());

    // Count HCT files
    let hct_count = std::fs::read_dir(hct_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "hct")
                .unwrap_or(false)
        })
        .count();
    println!("HCT files found: {}\n", hct_count);

    // Create loader
    println!("--- Creating TieredHoloLoader ---");
    let start = Instant::now();

    let config = TieredConfig {
        vram_budget: 0,
        ram_budget: 4 * 1024 * 1024 * 1024, // 4GB
        min_quality: 1.0,
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;
    println!("Loader created in {:?}", start.elapsed());

    // List tensors
    let tensor_names = loader.tensor_names();
    println!("Tensors discovered: {}", tensor_names.len());

    // Show some tensor names
    println!("\nSample tensors:");
    for name in tensor_names.iter().take(5) {
        println!("  - {}", name);
    }

    // Load a few tensors
    println!("\n--- Loading Sample Tensors ---");
    let provider: Arc<dyn TensorProvider> = Arc::new(loader);

    let test_tensors = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];

    for name in &test_tensors {
        let start = Instant::now();
        match provider.get(name, &device, dtype) {
            Ok(tensor) => {
                println!(
                    "  {} -> {:?} loaded in {:?}",
                    name,
                    tensor.dims(),
                    start.elapsed()
                );
            },
            Err(e) => {
                println!("  {} -> ERROR: {}", name, e);
            },
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
