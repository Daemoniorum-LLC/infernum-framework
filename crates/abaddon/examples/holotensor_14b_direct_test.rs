//! Direct test of Qwen2.5-14B HoloTensor model without lazy loading.
//!
//! This test verifies the HoloTensor model works correctly by:
//! 1. Loading a subset of layers (to fit in memory)
//! 2. Running a forward pass through embedding + first layer
//! 3. Checking output for NaN/Inf and reasonable values
//!
//! Usage:
//!   cargo run --release --example holotensor_14b_direct_test

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use abaddon::holotensor::tiered_loading::{TieredConfig, TieredHoloLoader};
use abaddon::lazy_varbuilder::TensorProvider;

fn main() -> Result<()> {
    println!("========================================================================");
    println!("  Direct Test: Qwen2.5-14B HoloTensor (No Lazy Loading)");
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

    let device = Device::Cpu; // Use CPU for this diagnostic to avoid GPU memory issues
    let dtype = DType::F32;

    println!("Device: CPU");
    println!("DType: F32\n");

    // Load tokenizer from HuggingFace
    let api = Api::new()?;
    let draft_model_id = "Qwen/Qwen2.5-0.5B-Instruct"; // Same tokenizer
    let draft_repo = api.repo(Repo::new(draft_model_id.to_string(), RepoType::Model));
    let tokenizer_path = draft_repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Create TieredHoloLoader for loading tensors
    let config = TieredConfig {
        vram_budget: 0,                      // CPU only
        ram_budget: 32 * 1024 * 1024 * 1024, // 32GB
        min_quality: 1.0,                    // Full quality
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    println!("=== Creating TieredHoloLoader ===");
    let start = Instant::now();
    let loader = TieredHoloLoader::new(&hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);
    println!("Loader created in {:?}\n", start.elapsed());

    // Test 1: Load embedding and do a simple lookup
    println!("=== Test 1: Embedding Lookup ===");
    let start = Instant::now();

    let embed_weight = loader.get("model.embed_tokens.weight", &device, dtype)?;
    println!("Embedding shape: {:?}", embed_weight.dims());
    println!("Embedding loaded in {:?}", start.elapsed());

    // Tokenize a simple prompt
    let prompt = "Hello";
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Prompt: \"{}\"", prompt);
    println!("Token IDs: {:?}", input_ids);

    // Look up embeddings for the input tokens
    let input_tensor = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?;
    let embedded = embed_weight.index_select(&input_tensor.flatten_all()?, 0)?;
    let embedded = embedded.reshape((1, input_ids.len(), embed_weight.dims()[1]))?;

    println!("Embedded shape: {:?}", embedded.dims());

    // Check for NaN/Inf in embeddings
    let embed_vec: Vec<f32> = embedded.flatten_all()?.to_vec1()?;
    let nan_count = embed_vec.iter().filter(|v| v.is_nan()).count();
    let inf_count = embed_vec.iter().filter(|v| v.is_infinite()).count();
    println!("Embedding NaN count: {}", nan_count);
    println!("Embedding Inf count: {}", inf_count);
    println!(
        "First 5 embedding values: {:?}",
        &embed_vec[..5.min(embed_vec.len())]
    );

    let embed_mean = embedded.mean_all()?.to_scalar::<f32>()?;
    let embed_var = embedded.var(2)?.mean_all()?.to_scalar::<f32>()?;
    println!("Embedding mean: {:.6}, var: {:.6}", embed_mean, embed_var);

    // Test 2: Load RMSNorm and apply to embeddings
    println!("\n=== Test 2: Layer 0 Input LayerNorm ===");
    let start = Instant::now();

    let input_norm_weight = loader.get("model.layers.0.input_layernorm.weight", &device, dtype)?;
    println!("LayerNorm weight shape: {:?}", input_norm_weight.dims());
    println!("LayerNorm loaded in {:?}", start.elapsed());

    // Apply RMSNorm manually
    // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    let hidden_size = input_norm_weight.dims()[0];
    let eps = 1e-6f32;

    let sq = embedded.sqr()?;
    let mean_sq = sq.mean_keepdim(2)?;
    let eps_tensor = Tensor::new(&[eps], &device)?.broadcast_as(mean_sq.dims())?;
    let norm_factor = (mean_sq + eps_tensor)?.sqrt()?.recip()?;
    let normalized: Tensor = embedded.broadcast_mul(&norm_factor)?;
    let normalized =
        normalized.broadcast_mul(&input_norm_weight.reshape((1, 1, hidden_size))?)?;

    println!("Normalized shape: {:?}", normalized.dims());

    let norm_vec: Vec<f32> = normalized.flatten_all()?.to_vec1()?;
    let norm_nan = norm_vec.iter().filter(|v: &&f32| v.is_nan()).count();
    let norm_inf = norm_vec.iter().filter(|v: &&f32| v.is_infinite()).count();
    println!("Normalized NaN count: {}", norm_nan);
    println!("Normalized Inf count: {}", norm_inf);
    println!(
        "Normalized first 5: {:?}",
        &norm_vec[..5.min(norm_vec.len())]
    );

    // Test 3: Check Q projection weight shapes and values
    println!("\n=== Test 3: Q/K/V Projection Weights ===");

    let q_proj = loader.get("model.layers.0.self_attn.q_proj.weight", &device, dtype)?;
    let k_proj = loader.get("model.layers.0.self_attn.k_proj.weight", &device, dtype)?;
    let v_proj = loader.get("model.layers.0.self_attn.v_proj.weight", &device, dtype)?;

    println!("Q proj shape: {:?}", q_proj.dims());
    println!("K proj shape: {:?}", k_proj.dims());
    println!("V proj shape: {:?}", v_proj.dims());

    // Check value distributions
    for (name, tensor) in [("Q", &q_proj), ("K", &k_proj), ("V", &v_proj)] {
        let vec: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let mean = tensor.mean_all()?.to_scalar::<f32>()?;
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let nan_count = vec.iter().filter(|v| v.is_nan()).count();
        println!(
            "  {} proj: mean={:.6}, min={:.4}, max={:.4}, nan={}",
            name, mean, min, max, nan_count
        );
    }

    // Test 4: Simple Q projection
    println!("\n=== Test 4: Q Projection Forward ===");

    // Expected dimensions for Qwen2.5-14B:
    // hidden_size = 5120
    // num_attention_heads = 40
    // num_key_value_heads = 8 (GQA)
    // head_dim = 5120 / 40 = 128

    let hidden_size = 5120usize;
    let _num_heads = 40usize;
    let _num_kv_heads = 8usize;
    let _head_dim = 128usize;

    // Q projection: [batch, seq, hidden] @ [out_features, in_features]^T = [batch, seq, out_features]
    // Weight is [out_features, in_features] = [5120, 5120] for Q
    let seq_len = normalized.dims()[1];
    let x_2d = normalized.reshape((seq_len, hidden_size))?; // [seq, hidden]

    // matmul: [seq, hidden] @ [hidden, out]^T -> need [seq, hidden] @ [hidden, out]
    let q_out = x_2d.matmul(&q_proj.t()?)?;
    println!("Q output shape: {:?}", q_out.dims());

    let q_out_vec: Vec<f32> = q_out.flatten_all()?.to_vec1()?;
    let q_nan = q_out_vec.iter().filter(|v: &&f32| v.is_nan()).count();
    let q_inf = q_out_vec.iter().filter(|v: &&f32| v.is_infinite()).count();
    let q_mean = q_out.mean_all()?.to_scalar::<f32>()?;
    let q_max = q_out_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "Q output: mean={:.6}, max={:.4}, nan={}, inf={}",
        q_mean, q_max, q_nan, q_inf
    );
    println!(
        "Q output first 5: {:?}",
        &q_out_vec[..5.min(q_out_vec.len())]
    );

    // Test 5: LM Head projection
    println!("\n=== Test 5: LM Head ===");

    let lm_head = loader.get("lm_head.weight", &device, dtype)?;
    println!("LM head shape: {:?}", lm_head.dims()); // Should be [vocab_size, hidden_size]

    // Do a simple projection to get logits (using the Q output as a proxy for hidden states)
    // This is NOT how the real model works, but tests the weight values
    let fake_hidden = q_out.i(0..1)?; // Take first token's Q output as fake hidden state
    let fake_hidden = fake_hidden.reshape((1, 5120))?;

    let logits = fake_hidden.matmul(&lm_head.t()?)?; // [1, vocab_size]
    println!("Logits shape: {:?}", logits.dims());

    let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let logits_nan = logits_vec.iter().filter(|v: &&f32| v.is_nan()).count();
    let logits_inf = logits_vec.iter().filter(|v: &&f32| v.is_infinite()).count();

    println!("Logits: nan={}, inf={}", logits_nan, logits_inf);
    println!(
        "Logits min: {:.4}",
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min)
    );
    println!(
        "Logits max: {:.4}",
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("Logits mean: {:.6}", logits.mean_all()?.to_scalar::<f32>()?);

    // Get top 5 token predictions
    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 predictions (from fake forward pass):");
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

    // Summary
    println!("\n========================================================================");
    println!("DIAGNOSTIC SUMMARY");
    println!("========================================================================");

    let all_ok = nan_count == 0
        && inf_count == 0
        && norm_nan == 0
        && norm_inf == 0
        && q_nan == 0
        && q_inf == 0
        && logits_nan == 0
        && logits_inf == 0;

    if all_ok {
        println!("✓ All tensor operations completed without NaN/Inf");
        println!("✓ Individual tensor values look reasonable");
        println!("\nIf speculative decoding produces garbage, the issue is likely in:");
        println!("  1. LazyQwen2 layer loading/caching");
        println!("  2. Position encoding (RoPE)");
        println!("  3. Attention mask handling");
        println!("  4. Speculative verification logic");
    } else {
        println!("✗ Found NaN/Inf values in tensor operations");
        println!("  This indicates a problem with HoloTensor conversion");
    }

    let stats = loader.stats();
    println!("\nLoader stats:");
    println!("  Tensors loaded: {}", stats.tensors_loaded);
    println!(
        "  CPU reconstructions: {} ({} ms)",
        stats.cpu_reconstructions, stats.cpu_time_ms
    );

    println!("\n=== Test Complete ===");
    Ok(())
}
