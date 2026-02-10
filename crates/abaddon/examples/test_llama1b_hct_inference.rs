//! Test HCT inference with Llama 3.2-1B model.
//!
//! This test verifies that HCT weights can be loaded and used for inference.
//!
//! Run with:
//! ```bash
//! cargo run --example test_llama1b_hct_inference -p abaddon --release
//! ```

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Llama 3.2-1B HCT Inference Test ===\n");

    // Paths
    let hct_dir = Path::new("/tmp/llama1b-lossless");

    // Check if HCT files exist
    if !hct_dir.exists() {
        println!("HCT directory not found: {}", hct_dir.display());
        println!("Please run the convert_to_holo converter first.");
        return Ok(());
    }

    // Count HCT files
    let hct_count = std::fs::read_dir(hct_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "hct"))
        .count();
    println!("HCT files found: {}", hct_count);

    // Use CPU for testing - F32 required for CPU matmul
    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Device: CPU");
    println!("DType: F32 (CPU doesn't support BF16 matmul)");
    println!("Model: Llama 3.2-1B\n");

    // Llama 3.2-1B config
    let config = LlamaConfig {
        hidden_size: 2048,
        intermediate_size: 8192,
        vocab_size: 128256,
        num_hidden_layers: 16,
        num_attention_heads: 32,
        num_key_value_heads: Some(8), // GQA with 8 KV heads
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_position_embeddings: 131072,
        tie_word_embeddings: true,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        rope_scaling: None,
    };

    println!("--- Model Config ---");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!("  Attention heads: {}", config.num_attention_heads);
    println!("  KV heads: {:?}", config.num_key_value_heads);
    println!("  Intermediate size: {}", config.intermediate_size);
    println!("  Vocab size: {}", config.vocab_size);
    println!();

    // Load HCT weights
    println!("--- Loading HCT Weights ---");
    let start = Instant::now();

    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;

    let load_time = start.elapsed();
    println!(
        "  Loaded {} tensors in {:.2}s",
        tensors.len(),
        load_time.as_secs_f64()
    );

    // Show some tensor shapes and statistics
    println!("\n  Sample tensor shapes and stats:");
    for (name, tensor) in tensors.iter().take(5) {
        let flat = tensor.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;
        let non_zero = vals.iter().filter(|&&x| x.abs() > 1e-10).count();
        let sum: f32 = vals.iter().sum();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("    {} -> {:?}", name, tensor.dims());
        println!(
            "      non-zero: {}/{}, sum: {:.4}, range: [{:.4}, {:.4}]",
            non_zero,
            vals.len(),
            sum,
            min,
            max
        );
    }
    println!();

    // Create VarBuilder
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);

    // Build model
    println!("--- Building Llama Model ---");
    let start = Instant::now();

    let mut model = Llama::load(config.clone(), vb)?;

    let build_time = start.elapsed();
    println!("  Model built in {:.2}s\n", build_time.as_secs_f64());

    // Run inference
    println!("--- Running Inference ---");

    // Test prompt: BOS token + "Hello"
    // Llama 3.2: BOS=128000
    let test_tokens = vec![128000u32, 9906u32]; // BOS + approximate "Hello" token

    let input_ids = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?;
    println!("  Input shape: {:?}", input_ids.dims());
    println!("  Input tokens: {:?}", test_tokens);

    let start = Instant::now();

    // Run forward pass
    let logits = model.forward(&input_ids, 0)?;

    let inference_time = start.elapsed();

    println!("  Output shape: {:?}", logits.dims());
    println!("  Forward pass: {:.3}s", inference_time.as_secs_f64());

    // Validate logits
    println!("\n--- Validating Output ---");

    // Convert to f32 for analysis
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_data: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    let finite_count = logits_data.iter().filter(|x| x.is_finite()).count();
    let total = logits_data.len();
    let finite_ratio = finite_count as f64 / total as f64 * 100.0;

    println!("  Total values: {}", total);
    println!("  Finite values: {} ({:.1}%)", finite_count, finite_ratio);

    if finite_ratio > 99.0 {
        let valid: Vec<f32> = logits_data
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
        let min = valid.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  Mean: {:.4}", mean);
        println!("  Range: [{:.4}, {:.4}]", min, max);

        // Get top predicted token
        let last_logits = logits_f32.i((0, logits.dim(1)? - 1, ..))?;
        let top_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
        println!("  Top predicted token: {}", top_token);

        // Get top 5 tokens
        println!("\n  Top 5 predicted tokens:");
        let logits_vec: Vec<f32> = last_logits.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, (token, score)) in indexed.iter().take(5).enumerate() {
            println!("    {}. Token {} (score: {:.4})", i + 1, token, score);
        }
    }

    // Summary
    println!("\n=== Summary ===");
    println!("  HCT weights loaded successfully");
    println!("  Model built without errors");
    println!(
        "  Forward pass completed in {:.3}s",
        inference_time.as_secs_f64()
    );

    if finite_ratio > 99.0 {
        println!("  Logits are valid (>99% finite)");
        println!("\n  HCT inference test PASSED!");
    } else {
        println!(
            "  Logits contain too many NaN/Inf values ({:.1}% finite)",
            finite_ratio
        );
        println!("\n  HCT inference test FAILED");
    }

    Ok(())
}
