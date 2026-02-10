//! Test INT4 inference with fixed dequantization.
//!
//! This test verifies that the INT4 HCT weights can be loaded and used
//! for actual inference with the Qwen2 model.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};

fn main() -> anyhow::Result<()> {
    println!("=== INT4 Inference Test ===\n");

    let model_dir = Path::new(
        "/home/crook/dev2/workspace/nyx/infernum/infernum-complete/test_models/qwen2.5-7b-int4-v3",
    );
    let config_path = model_dir.join("config.json");

    // Check if model exists
    if !model_dir.exists() {
        println!("Model directory not found: {}", model_dir.display());
        println!("Please ensure the INT4 model is available.");
        return Ok(());
    }

    // Use CPU for testing (avoids CUDA memory issues)
    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Device: CPU");
    println!("DType: F32");
    println!("Model: {}\n", model_dir.display());

    // Load model config
    println!("Loading config...");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Qwen2Config = serde_json::from_str(&config_str)?;

    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_hidden_layers);
    println!("  Attention heads: {}", config.num_attention_heads);
    println!("  Vocab size: {}", config.vocab_size);
    println!();

    // Load HCT weights
    println!("Loading INT4 weights (this may take a while on CPU)...");
    let start = Instant::now();

    let tensors = load_hct_directory_sequential(model_dir, &device, dtype)?;

    let load_time = start.elapsed();
    println!(
        "  Loaded {} tensors in {:.2}s\n",
        tensors.len(),
        load_time.as_secs_f64()
    );

    // Create VarBuilder
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);

    // Load model
    println!("Building Qwen2 model...");
    let start = Instant::now();

    let mut model = Qwen2::load(config.clone(), vb)?;

    let build_time = start.elapsed();
    println!("  Model built in {:.2}s\n", build_time.as_secs_f64());

    // Create a simple test input (token IDs)
    println!("Running inference test...");

    // Test prompt: "Hello" -> token ID (using a simple placeholder since we don't have tokenizer)
    // For Qwen2, typical BOS is 151643 and "Hello" might be around 22557
    let test_tokens = vec![151643u32, 9707]; // BOS + "Hello" approximation

    let input_ids = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?; // [1, seq_len]

    println!("  Input shape: {:?}", input_ids.dims());

    let start = Instant::now();

    // Run forward pass
    let logits = model.forward(&input_ids, 0)?;

    let inference_time = start.elapsed();

    println!("  Output shape: {:?}", logits.dims());
    println!("  Forward pass: {:.3}s", inference_time.as_secs_f64());

    // Check logits are valid
    let logits_data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let finite_count = logits_data.iter().filter(|x| x.is_finite()).count();
    let total = logits_data.len();
    let finite_ratio = finite_count as f32 / total as f32 * 100.0;

    println!("\n  Logits statistics:");
    println!("    Total values: {}", total);
    println!("    Finite values: {} ({:.1}%)", finite_count, finite_ratio);

    if finite_ratio > 99.0 {
        let valid: Vec<f32> = logits_data
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let mean: f32 = valid.iter().sum::<f32>() / valid.len() as f32;
        let min = valid.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("    Mean: {:.4}", mean);
        println!("    Range: [{:.4}, {:.4}]", min, max);

        // Get top predicted token
        let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
        let top_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
        println!("    Top predicted token: {}", top_token);
    }

    // Summary
    println!("\n=== Summary ===");
    println!("✓ INT4 weights loaded successfully");
    println!("✓ Model built without errors");
    println!("✓ Forward pass completed");

    if finite_ratio > 99.0 {
        println!("✓ Logits are valid (>99% finite)");
        println!("\n🎉 INT4 inference test PASSED!");
    } else {
        println!("✗ Logits contain too many NaN/Inf values");
        println!("\n❌ INT4 inference test FAILED - check weight dequantization");
    }

    Ok(())
}
