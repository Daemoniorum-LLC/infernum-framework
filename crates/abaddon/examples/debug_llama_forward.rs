//! Debug Llama forward pass step-by-step.
//!
//! This test traces each stage of the forward pass to identify where zeros appear.
//!
//! Run with:
//! ```bash
//! cargo run --example debug_llama_forward -p abaddon --release
//! ```

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;

use abaddon::hct_sequential::load_hct_directory_sequential;
use anyhow::Result;

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
        "    non-zero: {}/{} ({:.1}%), sum: {:.6}, range: [{:.6}, {:.6}]",
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
    println!("=== Debug Llama Forward Pass ===\n");

    let hct_dir = Path::new("/tmp/llama1b-lossless");

    if !hct_dir.exists() {
        println!("HCT directory not found: {}", hct_dir.display());
        return Ok(());
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("--- Loading tensors ---");
    let tensors = load_hct_directory_sequential(hct_dir, &device, dtype)?;
    println!("Loaded {} tensors\n", tensors.len());

    // Check if key tensors exist
    println!("--- Checking key tensors ---");
    let embed_key = "model.embed_tokens.weight";
    let norm_key = "model.norm.weight";
    let layer0_norm = "model.layers.0.input_layernorm.weight";

    for key in [embed_key, norm_key, layer0_norm] {
        if let Some(t) = tensors.get(key) {
            tensor_stats(t, key)?;
        } else {
            println!("  {} -> NOT FOUND!", key);
            // List similar keys
            let similar: Vec<_> = tensors
                .keys()
                .filter(|k| k.contains("embed") || k.contains("norm"))
                .take(5)
                .collect();
            println!("    Similar keys: {:?}", similar);
        }
    }
    println!();

    // Manual embedding lookup test
    println!("--- Manual Embedding Test ---");
    if let Some(embed_weight) = tensors.get(embed_key) {
        println!("Embedding weight shape: {:?}", embed_weight.dims());

        // Test token 128000 (BOS)
        let token_id = 128000usize;
        println!("Looking up token {}", token_id);

        let embedding_row = embed_weight.i(token_id)?;
        tensor_stats(&embedding_row, &format!("embed[{}]", token_id))?;

        // Test token 9906 (approximate "Hello")
        let token_id = 9906usize;
        println!("Looking up token {}", token_id);

        let embedding_row = embed_weight.i(token_id)?;
        tensor_stats(&embedding_row, &format!("embed[{}]", token_id))?;

        // Test that candle_nn::Embedding works
        println!("\nTesting candle_nn::Embedding...");
        let embedding = candle_nn::Embedding::new(embed_weight.clone(), embed_weight.dim(1)?);

        let input_ids = Tensor::new(&[128000u32, 9906u32], &device)?.unsqueeze(0)?;
        println!("Input IDs shape: {:?}", input_ids.dims());

        let embedded = embedding.forward(&input_ids)?;
        tensor_stats(&embedded, "Embedded output")?;
    }
    println!();

    // Test layer 0 components
    println!("--- Layer 0 Components ---");
    for suffix in [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "mlp.gate_proj.weight",
    ] {
        let key = format!("model.layers.0.{}", suffix);
        if let Some(t) = tensors.get(&key) {
            tensor_stats(t, &key)?;
        } else {
            println!("  {} -> NOT FOUND", key);
        }
    }
    println!();

    // Now test a full VarBuilder load
    println!("--- VarBuilder Test ---");
    let vb = VarBuilder::from_tensors(tensors.clone(), dtype, &device);

    // Try to get embed_tokens via VarBuilder path
    println!("Trying vb.pp(\"model.embed_tokens\").get(\"weight\")...");
    match vb
        .pp("model.embed_tokens")
        .get((128256usize, 2048usize), "weight")
    {
        Ok(w) => {
            tensor_stats(&w, "embed_tokens.weight via VarBuilder")?;
        },
        Err(e) => {
            println!("  Error: {}", e);
            // Try to find what keys are available
            println!(
                "  Available keys with 'embed': {:?}",
                tensors
                    .keys()
                    .filter(|k| k.contains("embed"))
                    .collect::<Vec<_>>()
            );
        },
    }

    // Test loading embedding via candle_nn::embedding
    println!("\nTrying candle_nn::embedding(vocab_size, hidden_size, vb)...");
    match candle_nn::embedding(128256, 2048, vb.pp("model.embed_tokens")) {
        Ok(embed) => {
            let weights = embed.embeddings();
            tensor_stats(weights, "Loaded embedding weights")?;

            // Test forward
            let input_ids = Tensor::new(&[128000u32, 9906u32], &device)?.unsqueeze(0)?;
            let embedded = embed.forward(&input_ids)?;
            tensor_stats(&embedded, "Forward result")?;
        },
        Err(e) => {
            println!("  Error: {}", e);
        },
    }

    println!("\n=== Done ===");
    Ok(())
}
