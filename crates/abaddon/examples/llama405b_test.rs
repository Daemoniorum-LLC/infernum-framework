//! Test 405B HoloTensor loading with recovery for corrupted files.
//!
//! This validates the progressive loading infrastructure with the actual 405B model.
//!
//! Run with:
//! ```bash
//! CARGO_INCREMENTAL=0 cargo run --example llama405b_test --release
//! ```

use std::sync::Arc;
use std::time::Instant;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use anyhow::Result;
use candle_core::{DType, Device};

fn main() -> Result<()> {
    println!("=== Llama 405B HoloTensor Loading Test ===\n");

    let hct_dir = std::path::Path::new("/tmp/llama405b-holo");

    if !hct_dir.exists() {
        println!("405B HCT directory not found: {}", hct_dir.display());
        println!("Expected at: /tmp/llama405b-holo");
        return Ok(());
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("HCT directory: {}", hct_dir.display());

    // Count and analyze HCT files
    let mut total_files = 0;
    let mut small_files = 0; // < 500 bytes (likely truncated)
    let mut medium_files = 0; // < 100KB
    let mut large_files = 0; // >= 100KB

    for entry in std::fs::read_dir(hct_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "hct") {
            total_files += 1;
            let size = entry.metadata()?.len();
            if size < 500 {
                small_files += 1;
            } else if size < 100 * 1024 {
                medium_files += 1;
            } else {
                large_files += 1;
            }
        }
    }

    println!("HCT file analysis:");
    println!("  Total files: {}", total_files);
    println!("  Truncated (<500 bytes): {} ⚠️", small_files);
    println!("  Small (<100KB): {}", medium_files);
    println!("  Large (>=100KB): {} ✓", large_files);
    println!();

    // Test 1: Create TieredHoloLoader
    println!("--- Test 1: TieredHoloLoader Creation ---");
    let config = TieredConfig {
        vram_budget: 0,                     // No VRAM for CPU test
        ram_budget: 8 * 1024 * 1024 * 1024, // 8GB RAM budget
        min_quality: 0.7,
        target_quality: 0.95,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(hct_dir, config, device.clone(), dtype)?;
    println!("TieredHoloLoader created successfully");

    // Test 2: List available tensors
    println!("\n--- Test 2: Tensor Discovery ---");
    let tensor_names = loader.tensor_names();
    println!("Total tensors available: {}", tensor_names.len());

    // Categorize tensors
    let mut weight_tensors = 0;
    let mut scale_tensors = 0;
    let mut layernorm_tensors = 0;
    let mut embed_tensors = 0;

    for name in &tensor_names {
        if name.contains("weight") && !name.contains("scale") {
            weight_tensors += 1;
        } else if name.contains("scale") {
            scale_tensors += 1;
        } else if name.contains("layernorm") || name.contains("norm") {
            layernorm_tensors += 1;
        } else if name.contains("embed") || name.contains("lm_head") {
            embed_tensors += 1;
        }
    }

    println!("Tensor categories:");
    println!("  Weight tensors: {}", weight_tensors);
    println!("  Scale tensors: {}", scale_tensors);
    println!("  LayerNorm tensors: {}", layernorm_tensors);
    println!("  Embed/LM-head tensors: {}", embed_tensors);

    // Test 3: Create LazyVarBuilder
    println!("\n--- Test 3: LazyVarBuilder Creation ---");
    let provider: Arc<dyn TensorProvider> = Arc::new(loader);
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);
    println!("LazyVarBuilder created successfully");

    // Test 4: Try loading valid tensors (large weights)
    println!("\n--- Test 4: Loading Valid Weight Tensors ---");

    let valid_tensors = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.embed_tokens.weight",
    ];

    for name in &valid_tensors {
        let start = Instant::now();
        match lazy_vb.get(name) {
            Ok(tensor) => {
                let elapsed = start.elapsed();
                let size_mb = tensor.elem_count() as f64 * 4.0 / (1024.0 * 1024.0);
                println!(
                    "  ✓ Loaded: {} -> {:?} ({:.2} MB, {:.2?})",
                    name,
                    tensor.dims(),
                    size_mb,
                    elapsed
                );
            },
            Err(e) => {
                println!("  ✗ Failed: {} -> {}", name, e);
            },
        }
    }

    // Test 5: Try loading potentially corrupted tensors (scales)
    println!("\n--- Test 5: Testing Truncated Scale Tensors ---");

    let scale_tensors_to_test = [
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.k_proj.weight_scale",
        "model.layers.1.self_attn.k_proj.k_scale",
        "model.layers.11.self_attn.v_proj.v_scale",
    ];

    let mut corrupted_count = 0;
    for name in &scale_tensors_to_test {
        let start = Instant::now();
        match lazy_vb.get(name) {
            Ok(tensor) => {
                let elapsed = start.elapsed();
                println!(
                    "  ✓ Loaded: {} -> {:?} ({:.2?})",
                    name,
                    tensor.dims(),
                    elapsed
                );
            },
            Err(e) => {
                corrupted_count += 1;
                println!("  ⚠️ Corrupted: {} -> {}", name, e);
            },
        }
    }

    if corrupted_count > 0 {
        println!(
            "\n  Found {} corrupted tensors - need recovery logic!",
            corrupted_count
        );
    }

    // Test 6: Memory usage estimate for 405B
    println!("\n--- Test 6: 405B Memory Analysis ---");

    // Llama 405B has 126 layers
    let num_layers: u64 = 126;
    let hidden_size: u64 = 16384;
    let intermediate_size: u64 = 53248; // 405B uses ~3.25x hidden for FFN

    // Per-layer memory (approximate) - use u64 to avoid overflow
    let attn_size: u64 = 4 * hidden_size * hidden_size; // q, k, v, o projections
    let mlp_size: u64 = 3 * hidden_size * intermediate_size; // gate, up, down
    let layer_params = attn_size + mlp_size;
    let layer_size_f32 = layer_params as f64 * 4.0; // F32 = 4 bytes

    println!("405B Model Analysis:");
    println!("  Layers: {}", num_layers);
    println!("  Hidden size: {}", hidden_size);
    println!("  Intermediate size: {}", intermediate_size);
    println!("  Per-layer params: {:.0}M", layer_params as f64 / 1e6);
    println!("  Per-layer size (F32): {:.2} GB", layer_size_f32 / 1e9);
    println!(
        "  Total model size (F32): {:.2} GB",
        layer_size_f32 * num_layers as f64 / 1e9
    );

    // With FP8 quantization (what the NVIDIA model uses)
    let layer_size_fp8 = layer_size_f32 / 4.0; // FP8 = 1/4 of F32
    println!("  Per-layer size (FP8): {:.2} GB", layer_size_fp8 / 1e9);
    println!(
        "  Total model size (FP8): {:.2} GB",
        layer_size_fp8 * num_layers as f64 / 1e9
    );

    // How many layers fit in RAM
    let ram_budget = 76.0 * 1e9; // 76GB usable
    let layers_in_ram = (ram_budget / layer_size_fp8).floor() as usize;
    println!("\n  With 80GB RAM budget (76GB usable):");
    println!("    Max layers in RAM (FP8): {}", layers_in_ram);
    println!(
        "    This is {:.1}% of the model",
        layers_in_ram as f64 / num_layers as f64 * 100.0
    );

    // Cache stats
    println!("\n--- Test 7: Cache Statistics ---");
    let (cache_entries, cache_bytes) = lazy_vb.cache_stats();
    println!("  Cache entries: {}", cache_entries);
    println!("  Cache memory: {:.2} GB", cache_bytes as f64 / 1e9);

    // Summary
    println!("\n=== Summary ===");
    println!("Total tensors discovered: {}", tensor_names.len());
    println!("Truncated files needing recovery: {}", small_files);
    println!("Valid weight files: {}", large_files);

    println!("\n=== Next Steps ===");
    if corrupted_count > 0 || small_files > 0 {
        println!(
            "1. Add recovery logic for {} truncated scale tensors",
            small_files
        );
        println!("   - Initialize scales to 1.0 (neutral)");
        println!("   - Initialize biases to 0.0");
        println!("2. Test end-to-end inference with recovered tensors");
    }

    Ok(())
}
