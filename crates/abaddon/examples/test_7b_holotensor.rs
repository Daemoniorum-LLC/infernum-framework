//! Quick test for 7B HoloTensor reconversion quality.
//!
//! Usage:
//!   cargo run --release --example test_7b_holotensor --features cuda

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::{LazyVarBuilder, TensorProvider};
use abaddon::models::lazy_qwen2::LazyQwen2;
use abaddon::models::qwen2::Qwen2Config;

fn main() -> Result<()> {
    println!("========================================================================");
    println!("  Test: Qwen2.5-7B HoloTensor v2 (Reconverted)");
    println!("========================================================================\n");

    let hct_dir =
        PathBuf::from("/home/crook/.cache/infernum/models/hct/Qwen--Qwen2.5-7B-HoloTensor-v2");

    if !hct_dir.exists() {
        println!(
            "ERROR: 7B HoloTensor model not found at: {}",
            hct_dir.display()
        );
        return Ok(());
    }

    if !candle_core::utils::cuda_is_available() {
        println!("ERROR: CUDA required for this test");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    println!("Device: CUDA:0");
    println!("DType: BF16\n");

    // Load tokenizer
    let api = Api::new()?;
    let model_id = "Qwen/Qwen2.5-7B-Instruct";
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Create TieredHoloLoader
    println!("=== Loading Qwen2.5-7B HoloTensor ===");
    let loader_start = Instant::now();

    let config = TieredConfig {
        vram_budget: 20 * 1024 * 1024 * 1024, // 20GB
        ram_budget: 16 * 1024 * 1024 * 1024,  // 16GB RAM cache
        min_quality: 1.0,
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(&hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);
    println!("TieredHoloLoader created in {:?}", loader_start.elapsed());

    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Qwen2.5-7B config
    let model_config = Qwen2Config {
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

    // Keep all 28 layers loaded (no eviction)
    let max_loaded_layers = 28;
    let mut model = LazyQwen2::load(model_config.clone(), lazy_vb, max_loaded_layers)?;
    println!(
        "LazyQwen2 shell created with max_loaded_layers: {}",
        max_loaded_layers
    );

    // Warmup all layers
    println!("\nWarming up all 28 layers...");
    let warmup_start = Instant::now();
    let layers_loaded = model.warmup();
    println!(
        "Warmed up {} layers in {:?}",
        layers_loaded,
        warmup_start.elapsed()
    );

    // Test generation
    println!("\n=== Starting Generation ===");

    let prompt = "The future of artificial intelligence will";
    let max_tokens = 30;
    let eos_token = model_config.eos_token_id.unwrap_or(151645) as u32;

    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    println!("Prompt: \"{}\"", prompt);
    println!("Prompt tokens: {:?}", prompt_tokens);
    println!("Max new tokens: {}", max_tokens);

    // Prefill
    let input = Tensor::new(prompt_tokens.clone(), &device)?.unsqueeze(0)?;
    println!("\nPrefilling...");
    let prefill_start = Instant::now();
    let logits = model.forward(&input, 0)?;
    println!("Prefill done in {:?}", prefill_start.elapsed());

    // Check logits
    let last_logits = logits
        .i((0, logits.dims()[1] - 1, ..))?
        .to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = last_logits.to_vec1()?;

    println!("\nLogits stats:");
    println!("  Mean: {:.4}", last_logits.mean_all()?.to_scalar::<f32>()?);
    println!(
        "  Max: {:.4}",
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  Min: {:.4}",
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min)
    );

    // Top 5 predictions
    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 next token predictions:");
    for (i, (token_id, score)) in indexed.iter().take(5).enumerate() {
        let decoded = tokenizer
            .decode(&[*token_id as u32], false)
            .unwrap_or_else(|_| "[decode error]".to_string());
        println!(
            "  {}. token {} \"{}\": {:.4}",
            i + 1,
            token_id,
            decoded,
            score
        );
    }

    // First token
    let first_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
    let first_decoded = tokenizer
        .decode(&[first_token], false)
        .unwrap_or_else(|_| "[decode error]".to_string());
    println!(
        "\nFirst generated token: {} \"{}\"",
        first_token, first_decoded
    );

    // Generate remaining tokens
    println!("\nGenerating {} tokens...\n", max_tokens);
    let gen_start = Instant::now();

    let mut generated_tokens = vec![first_token];
    let mut pos = prompt_tokens.len();

    for i in 1..max_tokens {
        let last_token = *generated_tokens.last().unwrap();
        if last_token == eos_token {
            break;
        }

        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let next_logits = logits.i((0, 0, ..))?.to_dtype(DType::F32)?;
        let next_token = next_logits.argmax(0)?.to_scalar::<u32>()?;
        generated_tokens.push(next_token);
        pos += 1;

        if (i + 1) % 10 == 0 {
            let decoded = tokenizer
                .decode(&generated_tokens, false)
                .unwrap_or_else(|_| "[decode error]".to_string());
            println!("  [{}] {}", i + 1, decoded);
        }
    }

    let gen_elapsed = gen_start.elapsed();

    // Final output
    let generated_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    println!("\n========================================================================");
    println!("GENERATION RESULTS");
    println!("========================================================================");

    println!("\nPerformance:");
    println!(
        "  Generated: {} tokens in {:.2}s",
        generated_tokens.len(),
        gen_elapsed.as_secs_f64()
    );
    println!(
        "  Speed: {:.1} tokens/sec",
        generated_tokens.len() as f64 / gen_elapsed.as_secs_f64()
    );

    let final_stats = model.stats();
    println!("\nModel Stats:");
    println!(
        "  Loaded layers: {}/{}",
        final_stats.loaded_layers, final_stats.total_layers
    );
    println!("  Layer loads: {}", final_stats.layer_loads);
    println!("  Layer evictions: {}", final_stats.layer_evictions);

    println!("\n========================================================================");
    println!("Generated Text:");
    println!("------------------------------------------------------------------------");
    println!("{}{}", prompt, generated_text);
    println!("========================================================================");

    // Quality assessment
    let has_garbage = generated_text
        .chars()
        .any(|c| !c.is_ascii() && !c.is_alphanumeric())
        || generated_text.contains("▁")
        || generated_tokens.iter().any(|&t| t > 150000);

    println!("\nQuality Assessment:");
    if has_garbage {
        println!("  STATUS: GARBAGE OUTPUT DETECTED");
        println!("  The HoloTensor reconversion may have quality issues.");
    } else {
        println!("  STATUS: OUTPUT APPEARS COHERENT");
        println!("  The HoloTensor reconversion is working correctly.");
    }

    Ok(())
}
