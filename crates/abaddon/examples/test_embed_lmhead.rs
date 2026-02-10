//! Test just embed_tokens and lm_head without the transformer layers

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
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

    // Load tensors
    println!("Loading tensors...");
    let hct_tensors = load_hct_directory_sequential(hct_dir, &device, DType::F32)?;

    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;

    // Get embed_tokens
    let embed_name = "model.embed_tokens.weight";

    let st_embed_tensor = st.tensor(embed_name)?;
    let st_embed_shape: Vec<usize> = st_embed_tensor.shape().to_vec();
    let st_embed_data = st_embed_tensor.data();
    let st_embed_floats: Vec<f32> = st_embed_data
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    let orig_embed = Tensor::from_vec(st_embed_floats, st_embed_shape.as_slice(), &device)?;

    let hct_embed = hct_tensors.get(embed_name).expect("embed_tokens in HCT");

    println!("Original embed shape: {:?}", orig_embed.dims());
    println!("HCT embed shape: {:?}", hct_embed.dims());

    // Test 1: Embedding lookup
    println!("\n=== Test 1: Embedding Lookup ===");
    let test_tokens = Tensor::new(&[128000u32, 9906u32, 11u32, 358u32], &device)?;

    // Lookup embeddings manually
    let orig_lookups = orig_embed.index_select(&test_tokens, 0)?;
    let hct_lookups = hct_embed.index_select(&test_tokens, 0)?;

    let orig_flat: Vec<f32> = orig_lookups.flatten_all()?.to_vec1()?;
    let hct_flat: Vec<f32> = hct_lookups.flatten_all()?.to_vec1()?;

    println!(
        "Embedding lookup cosine similarity: {:.6}",
        cosine_similarity(&orig_flat, &hct_flat)
    );
    println!(
        "Original embedding stats: mean={:.6}, std={:.6}",
        orig_flat.iter().sum::<f32>() / orig_flat.len() as f32,
        (orig_flat.iter().map(|x| x.powi(2)).sum::<f32>() / orig_flat.len() as f32).sqrt()
    );
    println!(
        "HCT embedding stats: mean={:.6}, std={:.6}",
        hct_flat.iter().sum::<f32>() / hct_flat.len() as f32,
        (hct_flat.iter().map(|x| x.powi(2)).sum::<f32>() / hct_flat.len() as f32).sqrt()
    );

    // Test 2: lm_head projection (embed @ embed.T)
    // For tied embeddings, lm_head(x) = x @ embed.T
    println!("\n=== Test 2: LM Head Projection ===");

    // Use the first token's embedding as a hidden state
    let hidden = orig_lookups.i((0, ..))?; // [2048]
    let hidden_expanded = hidden.unsqueeze(0)?; // [1, 2048]

    // Create lm_head from embed_tokens
    let orig_lmhead = Linear::new(orig_embed.clone(), None);
    let hct_lmhead = Linear::new(hct_embed.clone(), None);

    let orig_logits = orig_lmhead.forward(&hidden_expanded)?;
    let hct_logits = hct_lmhead.forward(&hidden_expanded)?;

    let orig_logits_flat: Vec<f32> = orig_logits.flatten_all()?.to_vec1()?;
    let hct_logits_flat: Vec<f32> = hct_logits.flatten_all()?.to_vec1()?;

    println!("LM Head output shape: {:?}", orig_logits.dims());
    println!(
        "Logits cosine similarity (orig hidden, orig/hct lmhead): {:.6}",
        cosine_similarity(&orig_logits_flat, &hct_logits_flat)
    );

    // Test 3: Using HCT hidden with HCT lmhead
    println!("\n=== Test 3: HCT Hidden + HCT LM Head ===");
    let hct_hidden = hct_lookups.i((0, ..))?;
    let hct_hidden_expanded = hct_hidden.unsqueeze(0)?;

    let orig_logits2 = orig_lmhead.forward(&hct_hidden_expanded)?;
    let hct_logits2 = hct_lmhead.forward(&hct_hidden_expanded)?;

    let orig_logits2_flat: Vec<f32> = orig_logits2.flatten_all()?.to_vec1()?;
    let hct_logits2_flat: Vec<f32> = hct_logits2.flatten_all()?.to_vec1()?;

    println!("Using HCT hidden:");
    println!(
        "  orig_lmhead(hct_hidden) stats: mean={:.4}, min={:.4}, max={:.4}",
        orig_logits2_flat.iter().sum::<f32>() / orig_logits2_flat.len() as f32,
        orig_logits2_flat
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        orig_logits2_flat
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  hct_lmhead(hct_hidden) stats: mean={:.4}, min={:.4}, max={:.4}",
        hct_logits2_flat.iter().sum::<f32>() / hct_logits2_flat.len() as f32,
        hct_logits2_flat
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        hct_logits2_flat
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "Cosine similarity (hct_hidden): {:.6}",
        cosine_similarity(&orig_logits2_flat, &hct_logits2_flat)
    );

    // Test 4: Multiple random hidden states
    println!("\n=== Test 4: Random Hidden States ===");
    // Create random hidden states to simulate transformer output
    let random_hidden = Tensor::randn(0.0f32, 0.02f32, &[1, 2048], &device)?;

    let orig_rand_logits = orig_lmhead.forward(&random_hidden)?;
    let hct_rand_logits = hct_lmhead.forward(&random_hidden)?;

    let orig_rand_flat: Vec<f32> = orig_rand_logits.flatten_all()?.to_vec1()?;
    let hct_rand_flat: Vec<f32> = hct_rand_logits.flatten_all()?.to_vec1()?;

    println!(
        "Random hidden -> logits cosine similarity: {:.6}",
        cosine_similarity(&orig_rand_flat, &hct_rand_flat)
    );

    // Test 5: Check if the issue compounds through the transformer
    println!("\n=== Analysis ===");
    println!("If embedding lookup and lm_head individually work well,");
    println!("but full inference doesn't, the issue is in the transformer layers.");

    Ok(())
}
