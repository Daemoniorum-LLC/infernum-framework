//! Test CUDA quantized KV cache integration with actual Qwen2 model.
//!
//! Run with:
//! ```bash
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --example cuda_kv_cache_model_test --features cuda --release
//! ```

use std::time::Instant;

use abaddon::Tokenizer;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

fn main() -> Result<()> {
    println!("=== CUDA Quantized KV Cache Model Integration Test ===\n");

    // Check CUDA availability
    if !candle_core::utils::cuda_is_available() {
        println!("CUDA not available, skipping test");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);
    println!("Device dtype: BF16");

    // Use a small model for testing
    let model_id = "Qwen/Qwen2.5-0.5B-Instruct";
    println!("\nLoading model: {model_id}");

    // Load tokenizer
    let tokenizer_start = Instant::now();
    let tokenizer_path = hf_hub::api::sync::Api::new()?
        .model(model_id.to_string())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    println!("Tokenizer loaded in {:?}", tokenizer_start.elapsed());

    // Load model config
    let config_path = hf_hub::api::sync::Api::new()?
        .model(model_id.to_string())
        .get("config.json")?;
    let config: abaddon::models::Qwen2Config =
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
    let weights = hf_hub::api::sync::Api::new()?
        .model(model_id.to_string())
        .get("model.safetensors")?;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::BF16, &device)?
    };
    println!("Weights loaded in {:?}", weights_start.elapsed());

    // Test prompts
    let prompts = vec![
        "The capital of France is",
        "In machine learning, a neural network",
        "The fibonacci sequence starts with",
    ];

    // Compare Standard vs CUDA Quantized cache
    println!("\n--- Comparing Standard vs CUDA Quantized Cache ---\n");

    for prompt in &prompts {
        println!("Prompt: \"{prompt}\"");

        // Tokenize
        let token_ids = tokenizer.encode(prompt, true)?;
        println!("  Tokens: {:?}", token_ids);

        // === Standard cache ===
        let mut model_standard = abaddon::models::Qwen2::load(config.clone(), vb.clone())?;
        let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

        let std_start = Instant::now();
        let mut current_tokens = token_ids.clone();
        let mut logits = model_standard.forward(&input_tensor, 0)?;

        // Generate 10 tokens
        let mut std_gen_tokens = Vec::new();
        for _i in 0..10 {
            let next_token = sample_token(&logits)?;
            std_gen_tokens.push(next_token);
            current_tokens.push(next_token);
            let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            logits = model_standard.forward(&next_input, current_tokens.len() - 1)?;
        }
        let std_time = std_start.elapsed();
        let std_text = tokenizer.decode(&current_tokens, true)?;
        println!("  Standard: \"{std_text}\" ({:.2?})", std_time);
        if prompt == &prompts[0] {
            println!("  Standard token IDs: {:?}", std_gen_tokens);
        }
        drop(model_standard);

        // === CUDA Quantized cache ===
        #[cfg(feature = "cuda")]
        {
            let mut model_cuda = abaddon::models::Qwen2::load_with_cuda_quantized_cache(
                config.clone(),
                vb.clone(),
                0,
            )?;
            let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

            let cuda_start = Instant::now();
            let mut current_tokens = token_ids.clone();
            let mut logits = model_cuda.forward(&input_tensor, 0)?;

            // Compare first token logits with standard (for first prompt only)
            if prompt == &prompts[0] {
                // Re-run standard to get logits for comparison
                let mut model_std_cmp = abaddon::models::Qwen2::load(config.clone(), vb.clone())?;
                let std_logits = model_std_cmp.forward(&input_tensor, 0)?;

                // Get logits for last position (where we sample from)
                let std_dims = std_logits.dims();
                let vocab_size = std_dims[2]; // [batch, seq_len, vocab_size]
                let std_last_logits = std_logits
                    .i((0, std_dims[1] - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let cuda_dims = logits.dims();
                let cuda_last_logits = logits
                    .i((0, cuda_dims[1] - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;

                let mut max_diff = 0.0f32;
                let mut sum_diff = 0.0f32;
                for (s, c) in std_last_logits.iter().zip(cuda_last_logits.iter()) {
                    let diff = (s - c).abs();
                    sum_diff += diff;
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                let mean_diff = sum_diff / std_last_logits.len() as f32;
                println!(
                    "  Prefill logits comparison: max_diff={:.4}, mean_diff={:.4}",
                    max_diff, mean_diff
                );

                // Check logits for tokens 13 (.) and 15 (0)
                println!(
                    "  Token 13 '.' logit: std={:.4}, cuda={:.4}, diff={:.4}",
                    std_last_logits[13],
                    cuda_last_logits[13],
                    (std_last_logits[13] - cuda_last_logits[13]).abs()
                );
                println!(
                    "  Token 15 '0' logit: std={:.4}, cuda={:.4}, diff={:.4}",
                    std_last_logits[15],
                    cuda_last_logits[15],
                    (std_last_logits[15] - cuda_last_logits[15]).abs()
                );
                println!(
                    "  Token 220 ' ' logit: std={:.4}, cuda={:.4}",
                    std_last_logits[220], cuda_last_logits[220]
                );

                // Find top 5 tokens in each
                let mut std_top: Vec<(usize, f32)> =
                    std_last_logits.iter().cloned().enumerate().collect();
                std_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut cuda_top: Vec<(usize, f32)> =
                    cuda_last_logits.iter().cloned().enumerate().collect();
                cuda_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("  Std top 5: {:?}", &std_top[..5]);
                println!("  CUDA top 5: {:?}", &cuda_top[..5]);

                drop(model_std_cmp);
            }

            // Generate 10 tokens
            let mut gen_tokens = Vec::new();
            for _i in 0..10 {
                let next_token = sample_token(&logits)?;
                gen_tokens.push(next_token);
                current_tokens.push(next_token);
                let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                logits = model_cuda.forward(&next_input, current_tokens.len() - 1)?;
            }
            let cuda_time = cuda_start.elapsed();
            let cuda_text = tokenizer.decode(&current_tokens, true)?;
            println!("  CUDA INT8: \"{cuda_text}\" ({:.2?})", cuda_time);
            if prompt == &prompts[0] {
                println!("  Generated token IDs: {:?}", gen_tokens);
            }

            // Compare speedup
            let speedup = std_time.as_secs_f64() / cuda_time.as_secs_f64();
            println!("  Speedup: {:.2}x\n", speedup);
        }
    }

    // Benchmark longer sequence
    println!("\n--- Benchmark: Longer Sequence Generation ---\n");

    let long_prompt = "Write a detailed explanation of how transformers work in machine learning. Start with the attention mechanism.";
    let token_ids = tokenizer.encode(long_prompt, true)?;
    println!("Prompt: \"{long_prompt}\"");
    println!(
        "Tokens: {} input tokens, generating 50 tokens",
        token_ids.len()
    );

    // Standard cache benchmark
    let mut model_standard = abaddon::models::Qwen2::load(config.clone(), vb.clone())?;
    let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    let std_start = Instant::now();
    let mut current_tokens = token_ids.clone();
    let mut logits = model_standard.forward(&input_tensor, 0)?;

    for _ in 0..50 {
        let next_token = sample_token(&logits)?;
        current_tokens.push(next_token);
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        logits = model_standard.forward(&next_input, current_tokens.len() - 1)?;
    }
    let std_time = std_start.elapsed();
    println!(
        "Standard cache: {:.2?} ({:.1} tok/s)",
        std_time,
        50.0 / std_time.as_secs_f64()
    );
    drop(model_standard);

    // CUDA cache benchmark
    #[cfg(feature = "cuda")]
    {
        let mut model_cuda =
            abaddon::models::Qwen2::load_with_cuda_quantized_cache(config.clone(), vb.clone(), 0)?;
        let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

        let cuda_start = Instant::now();
        let mut current_tokens = token_ids.clone();
        let mut logits = model_cuda.forward(&input_tensor, 0)?;

        for _ in 0..50 {
            let next_token = sample_token(&logits)?;
            current_tokens.push(next_token);
            let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            logits = model_cuda.forward(&next_input, current_tokens.len() - 1)?;
        }
        let cuda_time = cuda_start.elapsed();
        println!(
            "CUDA INT8 cache: {:.2?} ({:.1} tok/s)",
            cuda_time,
            50.0 / cuda_time.as_secs_f64()
        );

        let speedup = std_time.as_secs_f64() / cuda_time.as_secs_f64();
        println!("\nCUDA Speedup: {:.2}x", speedup);
    }

    println!("\n=== Test Complete ===");
    Ok(())
}

/// Simple argmax sampling - takes last token logits
fn sample_token(logits: &Tensor) -> Result<u32> {
    // Logits shape: [batch, seq_len, vocab_size]
    // Take the last position
    let dims = logits.dims();
    let logits = match dims.len() {
        3 => {
            let seq_len = dims[1];
            logits.i((0, seq_len - 1, ..))? // Get last token for first batch
        },
        2 => logits.i((dims[0] - 1, ..))?, // [seq_len, vocab_size]
        1 => logits.clone(),
        _ => anyhow::bail!("Unexpected logits shape: {:?}", dims),
    };
    let logits = logits.to_dtype(candle_core::DType::F32)?;
    let logits: Vec<f32> = logits.to_vec1()?;
    let token = logits
        .iter()
        .enumerate()
        .max_by(|a: &(usize, &f32), b: &(usize, &f32)| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0);
    Ok(token)
}
