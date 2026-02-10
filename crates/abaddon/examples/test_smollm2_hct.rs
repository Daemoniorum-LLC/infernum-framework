//! Quick SmolLM2 HCT inference test
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== SmolLM2-135M HCT Inference Test ===\n");

    let hct_dir = Path::new("/tmp/smollm2-hct-test");

    let hct_count = std::fs::read_dir(hct_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "hct"))
        .count();
    println!("HCT files found: {}", hct_count);

    let device = Device::Cpu;
    let dtype = DType::F32;

    // SmolLM2-135M config
    let config = LlamaConfig {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: Some(3),
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        max_position_embeddings: 8192,
        tie_word_embeddings: true,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        rope_scaling: None,
    };

    println!("Loading HCT weights...");
    let start = Instant::now();
    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;
    println!(
        "Loaded {} tensors in {:.2}s\n",
        tensors.len(),
        start.elapsed().as_secs_f64()
    );

    // Check embedding tensor
    if let Some(embed) = tensors.get("model.embed_tokens.weight") {
        println!("Embedding tensor:");
        println!("  Shape: {:?}", embed.dims());
        let flat = embed.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;
        let abs_sum: f64 = vals.iter().map(|x| x.abs() as f64).sum();
        println!("  First 5: {:?}", &vals[..5]);
        println!("  Abs sum: {:.2}", abs_sum);
        println!("  Expected: ~2802168.45");

        let diff_percent = ((abs_sum - 2802168.45).abs() / 2802168.45) * 100.0;
        if diff_percent < 1.0 {
            println!("  ✓ Embedding values match (diff: {:.4}%)", diff_percent);
        } else {
            println!("  ✗ Embedding values MISMATCH (diff: {:.4}%)", diff_percent);
        }
    }

    println!("\nBuilding model...");
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let mut model = Llama::load(config.clone(), vb)?;

    // Run a simple forward pass
    println!("\nRunning forward pass...");
    let input_ids = Tensor::new(&[1u32, 5, 10, 15], &device)?.unsqueeze(0)?;
    let start_pos = 0;

    let logits = model.forward(&input_ids, start_pos)?;
    println!("Output shape: {:?}", logits.dims());

    // Get next token prediction
    let seq_len = logits.dims()[1];
    let last_logits = logits.i((.., seq_len - 1, ..))?;
    let probs = candle_nn::ops::softmax(&last_logits, 1)?;
    let top_probs: Vec<f32> = probs.flatten_all()?.to_vec1()?;

    // Find top 5 tokens
    let mut indexed: Vec<(usize, f32)> =
        top_probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 predicted tokens:");
    for (token_id, prob) in indexed.iter().take(5) {
        println!("  Token {}: {:.4}%", token_id, prob * 100.0);
    }

    // Check if output looks reasonable
    let top_prob = indexed[0].1;
    let entropy: f64 = top_probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -(p as f64) * (p as f64).ln())
        .sum();
    println!("\nTop probability: {:.4}%", top_prob * 100.0);
    println!("Output entropy: {:.2} nats", entropy);

    if top_prob > 0.005 && entropy < 8.0 {
        println!("\n✓ Output looks reasonable (not random)");
    } else {
        println!("\n✗ Output may be garbage (high entropy or flat distribution)");
    }

    Ok(())
}
