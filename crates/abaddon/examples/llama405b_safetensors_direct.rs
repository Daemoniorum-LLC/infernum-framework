//! Test 405B inference loading directly from safetensors (no HCT conversion needed).
//!
//! This validates direct safetensor loading for the 405B model.
//!
//! Run with:
//! ```bash
//! cargo run --example llama405b_safetensors_direct --release -p abaddon
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

fn main() -> Result<()> {
    println!("=== Llama 405B Direct Safetensors Test ===\n");

    let safetensors_dir = Path::new("/tmp/llama405b-safetensors");

    if !safetensors_dir.exists() {
        println!(
            "Safetensors directory not found: {}",
            safetensors_dir.display()
        );
        println!("Expected at: /tmp/llama405b-safetensors");
        return Ok(());
    }

    // Count safetensors files
    let st_files: Vec<_> = std::fs::read_dir(safetensors_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    println!("Safetensors files found: {}", st_files.len());

    // Analyze layer coverage
    let mut max_layer = 0;
    for entry in &st_files {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(layer_str) = name_str.strip_prefix("model_layers_") {
            if let Some(layer_num) = layer_str.split('_').next() {
                if let Ok(num) = layer_num.parse::<usize>() {
                    max_layer = max_layer.max(num);
                }
            }
        }
    }
    println!(
        "Layer coverage: 0-{} ({} of 126 layers)",
        max_layer,
        max_layer + 1
    );
    println!();

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Test loading a few tensors directly
    println!("--- Testing Direct Tensor Loading ---");

    let test_files = [
        "model_embed_tokens_weight.safetensors",
        "model_layers_0_self_attn_q_proj_weight.safetensors",
        "model_layers_0_input_layernorm_weight.safetensors",
    ];

    let mut total_params: u64 = 0;
    let mut total_time = std::time::Duration::ZERO;

    for filename in &test_files {
        let path = safetensors_dir.join(filename);
        if !path.exists() {
            println!("  Skip (not found): {}", filename);
            continue;
        }

        let start = Instant::now();

        // Load the safetensor file
        let data = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&data)?;

        // Get the tensor (safetensors files have one tensor named after the weight)
        let tensor_name = filename
            .strip_suffix(".safetensors")
            .unwrap_or(filename)
            .replace('_', ".");

        // Try to find the tensor - might be stored with different naming
        let tensor_names: Vec<_> = st.names().into_iter().collect();

        let elapsed = start.elapsed();
        total_time += elapsed;

        if let Some(first_name) = tensor_names.first() {
            let tensor_view = st.tensor(first_name)?;
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let params: u64 = shape.iter().map(|&x| x as u64).product();
            total_params += params;

            let size_mb = params as f64 * 4.0 / (1024.0 * 1024.0);
            println!(
                "  {} -> {:?} ({:.1}MB, {:.2?})",
                filename, shape, size_mb, elapsed
            );
        } else {
            println!("  {} -> no tensors found", filename);
        }
    }

    println!();
    println!("Loaded {} params in {:?}", total_params, total_time);
    println!(
        "Throughput: {:.1} MB/s",
        (total_params as f64 * 4.0 / 1e6) / total_time.as_secs_f64()
    );

    // Memory estimate for full model
    println!("\n--- 405B Memory Analysis ---");

    // Count all parameters
    let mut total_model_params: u64 = 0;
    let mut layer_counts: HashMap<usize, usize> = HashMap::new();

    for entry in &st_files {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Extract layer number
        if let Some(layer_str) = name_str.strip_prefix("model_layers_") {
            if let Some(layer_num) = layer_str.split('_').next() {
                if let Ok(num) = layer_num.parse::<usize>() {
                    *layer_counts.entry(num).or_insert(0) += 1;
                }
            }
        }

        // Quick size check without full deserialization
        let size = entry.metadata()?.len();
        // Rough estimate: safetensors overhead is ~200 bytes
        total_model_params += (size.saturating_sub(200)) / 4; // F32 = 4 bytes
    }

    println!("Available tensors by layer:");
    let mut layers: Vec<_> = layer_counts.keys().collect();
    layers.sort();

    // Show first few and last few layers
    if layers.len() > 10 {
        for &layer in layers.iter().take(3) {
            println!("  Layer {}: {} tensors", layer, layer_counts[layer]);
        }
        println!("  ...");
        for &layer in layers.iter().rev().take(3).collect::<Vec<_>>().iter().rev() {
            println!("  Layer {}: {} tensors", layer, layer_counts[layer]);
        }
    } else {
        for &layer in &layers {
            println!("  Layer {}: {} tensors", layer, layer_counts[layer]);
        }
    }

    println!();
    println!(
        "Estimated loaded params: {:.1}B",
        total_model_params as f64 / 1e9
    );
    println!(
        "Estimated size (F32): {:.1} GB",
        total_model_params as f64 * 4.0 / 1e9
    );
    println!(
        "Estimated size (F16): {:.1} GB",
        total_model_params as f64 * 2.0 / 1e9
    );

    // Full 405B stats
    println!("\nFull 405B model:");
    println!("  Total layers: 126");
    println!("  Total params: ~405B");
    println!("  Size (FP8): ~405 GB");
    println!("  Size (F16): ~810 GB");
    println!(
        "  Available: {} layers ({:.1}%)",
        max_layer + 1,
        (max_layer + 1) as f64 / 126.0 * 100.0
    );

    println!("\n=== Test Complete ===");
    Ok(())
}
