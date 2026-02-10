//! Test HCT quality at different DCT retention levels
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn run_test(hct_dir: &Path, retention: &str) -> Result<(Vec<u32>, f32)> {
    println!("\n=== Testing {}% retention ===", retention);
    println!("Path: {}", hct_dir.display());

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

    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;
    println!("  Loaded {} tensors", tensors.len());

    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let mut model = Llama::load(config, vb)?;

    // Test prompt tokens (BOS + "Hello")
    let test_tokens = vec![1u32, 15496u32];
    let input_ids = Tensor::new(&test_tokens[..], &device)?.unsqueeze(0)?;

    let logits = model.forward(&input_ids, 0)?;

    // Get last position logits
    let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
    let logits_vec: Vec<f32> = last_logits.to_vec1()?;

    // Get top 5 predictions
    let mut indexed: Vec<(u32, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_tokens: Vec<u32> = indexed.iter().take(5).map(|(t, _)| *t).collect();
    let top_score = indexed[0].1;

    println!("  Top 5 predictions:");
    for (i, (token, score)) in indexed.iter().take(5).enumerate() {
        println!("    {}. Token {} (score: {:.4})", i + 1, token, score);
    }

    Ok((top_tokens, top_score))
}

fn main() -> Result<()> {
    println!("=== HCT Retention Quality Comparison ===");

    let results_20 = run_test(Path::new("/tmp/smollm2-spectral-20"), "20")?;
    let results_50 = run_test(Path::new("/tmp/smollm2-spectral-50"), "50")?;
    let results_80 = run_test(Path::new("/tmp/smollm2-spectral-80"), "80")?;

    println!("\n=== Summary ===");
    println!("Top-1 predictions:");
    println!(
        "  20% retention: Token {} (score: {:.4})",
        results_20.0[0], results_20.1
    );
    println!(
        "  50% retention: Token {} (score: {:.4})",
        results_50.0[0], results_50.1
    );
    println!(
        "  80% retention: Token {} (score: {:.4})",
        results_80.0[0], results_80.1
    );

    // Check agreement
    let agree_20_50 = results_20.0[0] == results_50.0[0];
    let agree_50_80 = results_50.0[0] == results_80.0[0];
    let agree_20_80 = results_20.0[0] == results_80.0[0];

    println!("\nTop-1 agreement:");
    println!("  20% vs 50%: {}", if agree_20_50 { "✓" } else { "✗" });
    println!("  50% vs 80%: {}", if agree_50_80 { "✓" } else { "✗" });
    println!("  20% vs 80%: {}", if agree_20_80 { "✓" } else { "✗" });

    Ok(())
}
