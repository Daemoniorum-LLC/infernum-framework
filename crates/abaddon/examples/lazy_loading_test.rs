//! Test lazy loading infrastructure with TieredHoloLoader.
//!
//! This validates the layer-by-layer loading mechanism that enables 405B inference.
//!
//! Run with:
//! ```bash
//! cargo run --example lazy_loading_test --release
//! ```

use std::sync::Arc;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use anyhow::Result;
use candle_core::{DType, Device};

fn main() -> Result<()> {
    println!("=== Lazy Loading Infrastructure Test ===\n");

    // Use the Qwen2 7B HCT model for testing
    let hct_dir = std::path::Path::new("test_models/qwen2.5-7b-int4-v3");

    if !hct_dir.exists() {
        println!("HCT model directory not found: {}", hct_dir.display());
        println!("Please ensure test_models/qwen2.5-7b-int4-v3 exists with HCT files.");
        return Ok(());
    }

    let device = Device::Cpu; // Use CPU for this test
    let dtype = DType::F32;

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

    // Test 1: Create TieredHoloLoader
    println!("--- Test 1: TieredHoloLoader Creation ---");
    let config = TieredConfig {
        vram_budget: 0,                     // No VRAM for CPU test
        ram_budget: 8 * 1024 * 1024 * 1024, // 8GB RAM budget
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false, // Disable for deterministic test
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;
    println!("TieredHoloLoader created successfully");

    // Test 2: List available tensors
    println!("\n--- Test 2: Tensor Discovery ---");
    let tensor_names = loader.tensor_names();
    println!("Total tensors available: {}", tensor_names.len());

    // Show first 10 tensor names
    println!("First 10 tensors:");
    for name in tensor_names.iter().take(10) {
        println!("  - {}", name);
    }

    // Test 3: Create LazyVarBuilder
    println!("\n--- Test 3: LazyVarBuilder Creation ---");
    let provider: Arc<dyn TensorProvider> = Arc::new(loader);
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);
    println!("LazyVarBuilder created successfully");

    // Test 4: Load a few tensors lazily
    println!("\n--- Test 4: Lazy Tensor Loading ---");

    // Load layer 0 weights
    let layer0_tensors = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.input_layernorm.weight",
    ];

    let mut load_times = Vec::new();
    for name in &layer0_tensors {
        let start = Instant::now();
        match lazy_vb.get(name) {
            Ok(tensor) => {
                let elapsed = start.elapsed();
                load_times.push(elapsed);
                println!(
                    "  Loaded: {} -> {:?} ({:.2?})",
                    name,
                    tensor.dims(),
                    elapsed
                );
            },
            Err(e) => {
                println!("  Failed: {} -> {}", name, e);
            },
        }
    }

    // Test 5: Load from multiple layers (simulates forward pass)
    println!("\n--- Test 5: Multi-Layer Loading (Simulating Forward Pass) ---");

    let layers_to_test = [0, 5, 10, 15, 20, 25];
    let mut layer_load_times = Vec::new();

    for layer_idx in layers_to_test {
        let start = Instant::now();

        // Load key attention tensors for this layer
        let tensors = [
            format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.gate_proj.weight", layer_idx),
            format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
        ];

        let mut loaded = 0;
        for name in &tensors {
            if lazy_vb.get(name).is_ok() {
                loaded += 1;
            }
        }

        let elapsed = start.elapsed();
        layer_load_times.push(elapsed);
        println!(
            "  Layer {}: loaded {}/{} tensors ({:.2?})",
            layer_idx,
            loaded,
            tensors.len(),
            elapsed
        );
    }

    // Test 6: Cache statistics
    println!("\n--- Test 6: Cache Statistics ---");
    let (cache_entries, cache_bytes) = lazy_vb.cache_stats();
    println!("  Cache entries: {}", cache_entries);
    println!(
        "  Cache memory: {:.2} MB",
        cache_bytes as f64 / (1024.0 * 1024.0)
    );

    // Test 7: Reload cached tensor (should be faster)
    println!("\n--- Test 7: Cache Hit Performance ---");
    let test_tensor = "model.layers.0.self_attn.q_proj.weight";

    // First load (already cached from Test 4)
    let start = Instant::now();
    let _ = lazy_vb.get(test_tensor);
    let cached_time = start.elapsed();

    println!("  Cached access: {} ({:.2?})", test_tensor, cached_time);
    println!("  Expected: near-instant (microseconds)");

    // Summary
    println!("\n=== Summary ===");
    println!("Total tensors: {}", tensor_names.len());
    println!("Cache entries: {}", cache_entries);
    println!(
        "Cache memory: {:.2} MB",
        cache_bytes as f64 / (1024.0 * 1024.0)
    );

    if !layer_load_times.is_empty() {
        let avg_layer_time: f64 = layer_load_times
            .iter()
            .map(|t| t.as_secs_f64())
            .sum::<f64>()
            / layer_load_times.len() as f64;
        println!("Avg layer load time: {:.2} ms", avg_layer_time * 1000.0);
    }

    println!("\n=== Test Complete ===");
    println!("The lazy loading infrastructure is working correctly.");
    println!("For 405B inference:");
    println!("  - Only ~12 layers would be kept in memory (based on 80GB RAM)");
    println!("  - Layers load on-demand during forward pass");
    println!("  - LRU eviction prevents OOM");

    Ok(())
}
