//! Test generic KV cache integration with Llama model.
//!
//! This validates that the model-agnostic cache abstraction works correctly
//! with real inference across Standard, Quantized, and CUDA Quantized caches.
//!
//! Run with:
//! ```bash
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --example llama_cache_test --features cuda --release
//! ```

use std::time::Instant;

use abaddon::attention_cache::{CacheType, QuantizationGranularity};
use abaddon::models::Llama;
use abaddon::Tokenizer;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

fn main() -> Result<()> {
    println!("=== Llama Model-Agnostic Cache Integration Test ===\n");

    // Check CUDA availability
    let has_cuda = candle_core::utils::cuda_is_available();
    println!("CUDA available: {}", has_cuda);

    let device = if has_cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}\n", device);

    // Use a small Llama model for testing
    let model_id = "HuggingFaceTB/SmolLM2-135M-Instruct";
    println!("Loading model: {model_id}");

    // Load tokenizer
    let tokenizer_start = Instant::now();
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    println!("Tokenizer loaded in {:?}", tokenizer_start.elapsed());

    // Load model config
    let config_path = repo.get("config.json")?;
    let config: abaddon::models::LlamaConfig =
        serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    println!(
        "Config: hidden_size={}, layers={}, heads={}, kv_heads={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_kv_heads()
    );

    // Load model weights
    let weights_start = Instant::now();
    let weights = repo.get("model.safetensors")?;
    let dtype = if has_cuda { DType::BF16 } else { DType::F32 };
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
    println!("Weights loaded in {:?}\n", weights_start.elapsed());

    // Test prompts
    let prompts = vec![
        "The capital of France is",
        "In machine learning,",
        "The fibonacci sequence",
    ];

    // Define cache types to test
    let cache_types: Vec<(&str, CacheType)> = {
        let mut types = vec![
            ("Standard", CacheType::Standard),
            (
                "Quantized (PerToken)",
                CacheType::Quantized(QuantizationGranularity::PerToken),
            ),
        ];
        #[cfg(feature = "cuda")]
        if has_cuda {
            types.push(("CUDA Quantized", CacheType::CudaQuantized { device_id: 0 }));
        }
        types
    };

    println!("=== Testing {} cache types ===\n", cache_types.len());

    // Store results for comparison
    let mut all_results: Vec<(&str, Vec<String>, Vec<std::time::Duration>)> = Vec::new();

    for (cache_name, cache_type) in &cache_types {
        println!("--- Cache Type: {} ---", cache_name);

        let mut texts = Vec::new();
        let mut times = Vec::new();

        for prompt in &prompts {
            // Load model with this cache type
            let mut model =
                Llama::load_with_cache_type(config.clone(), vb.clone(), cache_type.clone())?;

            // Tokenize
            let token_ids = tokenizer.encode(prompt, true)?;
            let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

            // Generate tokens
            let gen_start = Instant::now();
            let mut current_tokens = token_ids.clone();
            let mut logits = model.forward(&input_tensor, 0)?;

            // Generate 15 tokens
            for _ in 0..15 {
                let next_token = sample_token(&logits)?;
                current_tokens.push(next_token);
                let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                logits = model.forward(&next_input, current_tokens.len() - 1)?;
            }
            let gen_time = gen_start.elapsed();

            // Decode
            let text = tokenizer.decode(&current_tokens, true)?;
            println!(
                "  \"{prompt}\" -> \"{}\" ({:.2?})",
                &text[prompt.len()..].trim(),
                gen_time
            );

            // Report cache stats
            println!(
                "    Cache: seq_len={}, memory={:.2} KB",
                model.cache_seq_len(),
                model.cache_memory_bytes() as f64 / 1024.0
            );

            texts.push(text);
            times.push(gen_time);
        }

        all_results.push((cache_name, texts, times));
        println!();
    }

    // Compare results
    println!("=== Results Comparison ===\n");

    if all_results.len() >= 2 {
        let (base_name, base_texts, base_times) = &all_results[0];
        println!("Baseline: {}", base_name);

        for (name, texts, times) in &all_results[1..] {
            println!("\nComparing {} vs {}:", name, base_name);

            // Check if outputs match
            let mut match_count = 0;
            for (i, (base_text, other_text)) in base_texts.iter().zip(texts.iter()).enumerate() {
                let matches = base_text == other_text;
                if matches {
                    match_count += 1;
                }
                println!(
                    "  Prompt {}: {}",
                    i + 1,
                    if matches { "MATCH ✓" } else { "DIFFER ✗" }
                );
                if !matches {
                    println!("    Base:  \"{}\"", base_text);
                    println!("    Other: \"{}\"", other_text);
                }
            }

            // Timing comparison
            let base_total: f64 = base_times.iter().map(|t| t.as_secs_f64()).sum();
            let other_total: f64 = times.iter().map(|t| t.as_secs_f64()).sum();
            let speedup = base_total / other_total;

            println!(
                "  Outputs: {}/{} match, Speedup: {:.2}x",
                match_count,
                base_texts.len(),
                speedup
            );
        }
    }

    // Longer sequence benchmark
    println!("\n=== Longer Sequence Benchmark (50 tokens) ===\n");

    let long_prompt = "Explain how attention works in neural networks:";
    let token_ids = tokenizer.encode(long_prompt, true)?;
    println!(
        "Prompt: \"{}\" ({} input tokens)",
        long_prompt,
        token_ids.len()
    );

    for (cache_name, cache_type) in &cache_types {
        let mut model =
            Llama::load_with_cache_type(config.clone(), vb.clone(), cache_type.clone())?;
        let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

        let start = Instant::now();
        let mut current_tokens = token_ids.clone();
        let mut logits = model.forward(&input_tensor, 0)?;

        for _ in 0..50 {
            let next_token = sample_token(&logits)?;
            current_tokens.push(next_token);
            let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            logits = model.forward(&next_input, current_tokens.len() - 1)?;
        }
        let elapsed = start.elapsed();

        let tok_per_sec = 50.0 / elapsed.as_secs_f64();
        let cache_kb = model.cache_memory_bytes() as f64 / 1024.0;

        println!(
            "{:20}: {:.2?} ({:.1} tok/s), cache: {:.1} KB",
            cache_name, elapsed, tok_per_sec, cache_kb
        );
    }

    println!("\n=== Test Complete ===");
    Ok(())
}

/// Simple argmax sampling - takes last token logits
fn sample_token(logits: &Tensor) -> Result<u32> {
    let dims = logits.dims();
    let logits = match dims.len() {
        3 => {
            let seq_len = dims[1];
            logits.i((0, seq_len - 1, ..))?
        },
        2 => logits.i((dims[0] - 1, ..))?,
        1 => logits.clone(),
        _ => anyhow::bail!("Unexpected logits shape: {:?}", dims),
    };
    let logits = logits.to_dtype(DType::F32)?;
    let logits: Vec<f32> = logits.to_vec1()?;
    let token = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0);
    Ok(token)
}
