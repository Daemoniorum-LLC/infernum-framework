//! Check embed_tokens quality specifically - embeddings are critical

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;
use anyhow::Result;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn main() -> Result<()> {
    let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
    let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-45pct");

    let device = Device::Cpu;

    // Load HCT tensors
    println!("Loading HCT tensors...");
    let hct_tensors = load_hct_directory_sequential(hct_dir, &device, DType::F32)?;

    // Load safetensors
    println!("Loading safetensors...");
    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    // Check embed_tokens
    println!("\n=== Checking embed_tokens ===");
    let embed_name = "model.embed_tokens.weight";

    if let Some(hct_embed) = hct_tensors.get(embed_name) {
        let st_tensor = st.tensor(embed_name)?;
        let shape: Vec<usize> = st_tensor.shape().to_vec();
        let data = st_tensor.data();

        let original: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect();

        let hct_flat = hct_embed.flatten_all()?;
        let hct_values: Vec<f32> = hct_flat.to_vec1()?;

        println!("Shape: {:?}", shape);
        println!("Original elements: {}", original.len());
        println!("HCT elements: {}", hct_values.len());

        // Overall similarity
        let overall_sim = cosine_similarity(&original, &hct_values);
        println!("\nOverall cosine similarity: {:.6}", overall_sim);

        // Per-token similarity (each row is a token embedding)
        let vocab_size = shape[0];
        let embed_dim = shape[1];

        println!("\nChecking per-token embedding quality...");
        let mut per_token_sims: Vec<f32> = Vec::with_capacity(vocab_size);
        let mut bad_tokens = Vec::new();

        for token_id in 0..vocab_size {
            let start = token_id * embed_dim;
            let end = start + embed_dim;

            let orig_row = &original[start..end];
            let hct_row = &hct_values[start..end];

            let sim = cosine_similarity(orig_row, hct_row);
            per_token_sims.push(sim);

            if sim < 0.90 {
                bad_tokens.push((token_id, sim));
            }
        }

        // Statistics
        per_token_sims.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_sim = per_token_sims[0];
        let p5_sim = per_token_sims[per_token_sims.len() * 5 / 100];
        let median_sim = per_token_sims[per_token_sims.len() / 2];
        let mean_sim: f32 = per_token_sims.iter().sum::<f32>() / per_token_sims.len() as f32;

        println!("Per-token embedding statistics:");
        println!("  Min similarity: {:.6}", min_sim);
        println!("  5th percentile: {:.6}", p5_sim);
        println!("  Median: {:.6}", median_sim);
        println!("  Mean: {:.6}", mean_sim);
        println!(
            "  Tokens with <90% similarity: {}/{}",
            bad_tokens.len(),
            vocab_size
        );

        // Show worst tokens
        if !bad_tokens.is_empty() {
            println!("\nWorst 10 token embeddings:");
            bad_tokens.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (token_id, sim) in bad_tokens.iter().take(10) {
                println!("  Token {}: {:.6}", token_id, sim);
            }
        }

        // Check specific tokens
        let test_tokens = [128000u32, 9906u32, 11u32, 358u32]; // BOS + "Hello, I"
        println!("\nTest tokens quality:");
        for &token_id in &test_tokens {
            if (token_id as usize) < vocab_size {
                let start = token_id as usize * embed_dim;
                let end = start + embed_dim;
                let orig_row = &original[start..end];
                let hct_row = &hct_values[start..end];
                let sim = cosine_similarity(orig_row, hct_row);
                println!("  Token {} (BOS/Hello/etc): {:.6}", token_id, sim);
            }
        }
    } else {
        println!("embed_tokens NOT FOUND in HCT!");
    }

    Ok(())
}
