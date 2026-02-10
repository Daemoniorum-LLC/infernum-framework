//! Quick test for Qwen 7B HCT lossless.
use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::qwen2::{Qwen2, Qwen2Config};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use std::time::Instant;

fn tensor_stats(t: &Tensor, name: &str) -> Result<()> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?;
    let vals: Vec<f32> = flat.to_vec1()?;
    let non_zero = vals.iter().filter(|&&x| x.abs() > 1e-10).count();
    let total = vals.len();
    let sum: f32 = vals.iter().sum();
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  {} -> {:?}", name, t.dims());
    println!(
        "    non-zero: {}/{} ({:.1}%), sum: {:.4}, range: [{:.4}, {:.4}]",
        non_zero,
        total,
        100.0 * non_zero as f64 / total as f64,
        sum,
        min,
        max
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Qwen 7B HCT Lossless Test ===\n");

    let hct_dir = Path::new("/tmp/qwen7b-hct");
    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("Loading 339 tensors from HCT...");
    let start = Instant::now();
    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;
    println!(
        "Loaded {} tensors in {:.2}s\n",
        tensors.len(),
        start.elapsed().as_secs_f64()
    );

    // Check key tensors
    println!("--- Key Tensor Stats ---");
    for key in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ] {
        if let Some(t) = tensors.get(key) {
            tensor_stats(t, key)?;
        } else {
            println!("  {} -> NOT FOUND", key);
        }
    }

    // Build model
    println!("\n--- Building Model ---");
    let config = Qwen2Config {
        hidden_size: 3584,
        intermediate_size: 18944,
        vocab_size: 152064,
        num_hidden_layers: 28,
        num_attention_heads: 28,
        num_key_value_heads: Some(4),
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
        sliding_window: None,
        use_sliding_window: false,
        bos_token_id: Some(151643),
        eos_token_id: Some(151645),
    };

    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let start = Instant::now();
    let mut model = Qwen2::load(config.clone(), vb)?;
    println!("Model built in {:.2}s", start.elapsed().as_secs_f64());

    // Forward pass
    println!("\n--- Forward Pass ---");
    let input_ids = Tensor::new(&[151643u32, 9707u32], &device)?.unsqueeze(0)?;
    println!("Input shape: {:?}", input_ids.dims());

    let start = Instant::now();
    let logits = model.forward(&input_ids, 0)?;
    println!("Forward time: {:.2}s", start.elapsed().as_secs_f64());
    println!("Output shape: {:?}", logits.dims());

    // Validate
    let flat = logits.flatten_all()?;
    let vals: Vec<f32> = flat.to_vec1()?;
    let finite = vals.iter().filter(|x| x.is_finite()).count();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n--- Output Stats ---");
    println!("  Total: {}", vals.len());
    println!(
        "  Finite: {} ({:.1}%)",
        finite,
        100.0 * finite as f64 / vals.len() as f64
    );
    println!("  Mean: {:.4}", mean);
    println!("  Range: [{:.4}, {:.4}]", min, max);

    // Top tokens
    let last_logits = logits.i((0, 1, ..))?;
    let probs: Vec<f32> = last_logits.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Top 5 tokens:");
    for (i, (token, score)) in indexed.iter().take(5).enumerate() {
        println!("    {}. Token {} (score: {:.4})", i + 1, token, score);
    }

    if finite as f64 / vals.len() as f64 > 0.99 {
        println!("\n=== TEST PASSED ===");
    } else {
        println!("\n=== TEST FAILED ===");
    }
    Ok(())
}
