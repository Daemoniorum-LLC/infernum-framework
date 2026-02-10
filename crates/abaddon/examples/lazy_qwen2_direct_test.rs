//! Direct test of LazyQwen2 14B without speculative decoding.
//!
//! This tests whether the target model (LazyQwen2 with HoloTensor)
//! can generate coherent text on its own.
//!
//! Usage:
//!   cargo run --release --example lazy_qwen2_direct_test --features cuda

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
    println!("  Direct Test: LazyQwen2 14B HoloTensor (No Speculative Decoding)");
    println!("========================================================================\n");

    let hct_dir =
        PathBuf::from("/home/crook/.cache/infernum/models/hct/Qwen--Qwen2.5-14B-HoloTensor");

    if !hct_dir.exists() {
        println!(
            "ERROR: 14B HoloTensor model not found at: {}",
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

    // Load tokenizer from HuggingFace
    let api = Api::new()?;
    let model_id = "Qwen/Qwen2.5-0.5B-Instruct"; // Same tokenizer as 14B
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Create TieredHoloLoader
    println!("=== Loading LazyQwen2 14B HoloTensor ===");
    let loader_start = Instant::now();

    let config = TieredConfig {
        vram_budget: 22 * 1024 * 1024 * 1024, // 22GB for 14B
        ram_budget: 32 * 1024 * 1024 * 1024,  // 32GB RAM cache
        min_quality: 1.0,
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    let loader = TieredHoloLoader::new(&hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);
    println!("TieredHoloLoader created in {:?}", loader_start.elapsed());
    println!(
        "GPU acceleration: {}\n",
        if loader.is_gpu_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );

    let provider: Arc<dyn TensorProvider> = Arc::clone(&loader) as Arc<dyn TensorProvider>;
    let lazy_vb = LazyVarBuilder::new(Arc::clone(&provider), device.clone(), dtype);

    // Qwen2.5-14B config
    let model_config = Qwen2Config {
        hidden_size: 5120,
        intermediate_size: 13824,
        vocab_size: 152064,
        num_hidden_layers: 48,
        num_attention_heads: 40,
        num_key_value_heads: Some(8),
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        max_position_embeddings: 32768,
        tie_word_embeddings: false,
        sliding_window: None,
        use_sliding_window: false,
        bos_token_id: Some(151643),
        eos_token_id: Some(151645),
    };

    // Keep 24 layers in VRAM
    let max_loaded_layers = 24;
    let mut model = LazyQwen2::load(model_config.clone(), lazy_vb, max_loaded_layers)?;
    println!("LazyQwen2 shell created");

    // Warmup - prefetch initial layers
    println!("\nWarming up model...");
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
    println!(
        "Prompt tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );
    println!("Max new tokens: {}", max_tokens);
    println!();

    // Create input tensor for prefill
    let input = Tensor::new(prompt_tokens.clone(), &device)?.unsqueeze(0)?;
    println!("Input shape: {:?}", input.dims());

    // Prefill
    println!("Prefilling...");
    let prefill_start = Instant::now();
    let logits = model.forward(&input, 0)?;
    println!("Prefill done in {:?}", prefill_start.elapsed());
    println!("Output logits shape: {:?}", logits.dims());

    // Check logits for NaN/Inf
    let last_logits = logits
        .i((0, logits.dims()[1] - 1, ..))?
        .to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = last_logits.to_vec1()?;
    let nan_count = logits_vec.iter().filter(|v| v.is_nan()).count();
    let inf_count = logits_vec.iter().filter(|v| v.is_infinite()).count();

    println!("\nLogits stats:");
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);
    println!("  Mean: {:.4}", last_logits.mean_all()?.to_scalar::<f32>()?);
    println!(
        "  Max: {:.4}",
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  Min: {:.4}",
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min)
    );

    // Sample first token
    let first_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
    let first_decoded = tokenizer
        .decode(&[first_token], false)
        .unwrap_or_else(|_| "[decode error]".to_string());
    println!(
        "\nFirst generated token: {} \"{}\"",
        first_token, first_decoded
    );

    // Generate more tokens
    println!("\nGenerating {} tokens...\n", max_tokens);
    let gen_start = Instant::now();

    let mut generated_tokens = vec![first_token];
    let mut pos = prompt_tokens.len();

    for i in 1..max_tokens {
        let last_token = *generated_tokens.last().unwrap();

        // Check for EOS
        if last_token == eos_token {
            println!("Hit EOS token at position {}", i);
            break;
        }

        // Forward single token
        let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;

        // Get logits for last position
        let next_logits = logits.i((0, 0, ..))?.to_dtype(DType::F32)?;

        // Greedy sampling
        let next_token = next_logits.argmax(0)?.to_scalar::<u32>()?;
        generated_tokens.push(next_token);
        pos += 1;

        // Print progress every 5 tokens
        if (i + 1) % 5 == 0 {
            let decoded = tokenizer
                .decode(&generated_tokens, false)
                .unwrap_or_else(|_| "[decode error]".to_string());
            println!("  [{}] {}", i + 1, decoded);
        }
    }

    let gen_elapsed = gen_start.elapsed();
    let tokens_per_sec = generated_tokens.len() as f64 / gen_elapsed.as_secs_f64();

    // Decode final output
    let generated_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    // Print results
    println!("\n========================================================================");
    println!("GENERATION RESULTS");
    println!("========================================================================");

    println!("\nPerformance:");
    println!(
        "  Generated: {} tokens in {:.2}s",
        generated_tokens.len(),
        gen_elapsed.as_secs_f64()
    );
    println!("  Speed: {:.1} tokens/sec", tokens_per_sec);

    let stats = model.stats();
    println!("\nModel Stats:");
    println!(
        "  Loaded layers: {}/{}",
        stats.loaded_layers, stats.total_layers
    );
    println!("  Layer loads: {}", stats.layer_loads);
    println!("  Layer evictions: {}", stats.layer_evictions);

    println!("\n========================================================================");
    println!("Generated Text:");
    println!("------------------------------------------------------------------------");
    println!("{}{}", prompt, generated_text);
    println!("========================================================================");

    let loader_stats = loader.stats();
    println!("\nTiered Loader Stats:");
    println!("  Tensors loaded: {}", loader_stats.tensors_loaded);
    println!(
        "  GPU reconstructions: {} ({} ms)",
        loader_stats.gpu_reconstructions, loader_stats.gpu_time_ms
    );
    println!(
        "  CPU fallbacks: {} ({} ms)",
        loader_stats.cpu_reconstructions, loader_stats.cpu_time_ms
    );

    println!("\n=== Complete ===");
    Ok(())
}
