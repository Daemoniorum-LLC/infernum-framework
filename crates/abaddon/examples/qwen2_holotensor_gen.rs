//! Test Qwen2.5-14B HoloTensor with standard (non-lazy) Qwen2 model.
//!
//! This test uses the regular Qwen2 model instead of LazyQwen2 to isolate
//! whether the issue is with lazy loading or the HoloTensor data.
//!
//! Since loading all 48 layers would take too long on CPU, we:
//! 1. Use CUDA for GPU acceleration
//! 2. Load only a few layers for testing (won't give meaningful text)
//! 3. Check that outputs don't have NaN/Inf
//!
//! Usage:
//!   cargo run --release --example qwen2_holotensor_gen --features cuda

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
    println!("  Qwen2.5-14B HoloTensor Full Forward Test (Single Layer)");
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

    // Use CUDA if available
    let device = if candle_core::utils::cuda_is_available() {
        println!("CUDA available, using GPU");
        Device::new_cuda(0)?
    } else {
        println!("CUDA not available, using CPU (will be slow)");
        Device::Cpu
    };
    // Use F32 on CPU (BF16 matmul not supported), BF16 on GPU
    let dtype = if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::BF16
    };

    println!("Device: {:?}", device);
    println!("DType: {:?}\n", dtype);

    // Load tokenizer
    let api = Api::new()?;
    let model_id = "Qwen/Qwen2.5-14B-Instruct";
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Create loader with 100% quality
    let config = TieredConfig {
        vram_budget: 24 * 1024 * 1024 * 1024, // 24GB
        ram_budget: 32 * 1024 * 1024 * 1024,
        min_quality: 1.0,
        target_quality: 1.0,
        enable_background_streaming: false,
        background_streams: 0,
    };

    println!("=== Creating Loader ===");
    let loader = TieredHoloLoader::new(&hct_dir, config, device.clone(), dtype)?;
    let loader = Arc::new(loader);

    // Qwen2.5-14B config
    let hidden_size = 5120usize;
    let num_heads = 40usize;
    let num_kv_heads = 8usize;
    let head_dim = hidden_size / num_heads; // 128
    let intermediate_size = 13824usize;
    let _vocab_size = 152064usize;
    let rope_theta = 1000000.0f32;

    println!("Model config:");
    println!("  hidden_size: {}", hidden_size);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  intermediate_size: {}", intermediate_size);
    println!("  rope_theta: {}", rope_theta);

    // Input
    let prompt = "The capital of France is";
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("\nPrompt: \"{}\"", prompt);
    println!("Token IDs: {:?}", input_ids);
    let seq_len = input_ids.len();

    // Load embeddings
    println!("\n=== Loading Weights ===");
    let start = Instant::now();

    let embed_weight = loader.get("model.embed_tokens.weight", &device, dtype)?;
    println!(
        "Embedding loaded: {:?} in {:?}",
        embed_weight.dims(),
        start.elapsed()
    );

    // Embed input tokens
    let input_tensor = Tensor::from_vec(input_ids.clone(), (input_ids.len(),), &device)?;
    let mut hidden = embed_weight.index_select(&input_tensor, 0)?;
    hidden = hidden.reshape((1, seq_len, hidden_size))?;
    println!("Input hidden: {:?}", hidden.dims());

    // Process through layer 0 only (to keep this fast)
    println!("\n=== Processing Layer 0 ===");

    // 1. Input LayerNorm
    let input_norm_w = loader.get("model.layers.0.input_layernorm.weight", &device, dtype)?;
    hidden = rms_norm(&hidden, &input_norm_w, 1e-6)?;
    println!("After RMSNorm: {:?}", hidden.dims());
    check_tensor("RMSNorm output", &hidden)?;

    // 2. Q/K/V projections
    let q_proj = loader.get("model.layers.0.self_attn.q_proj.weight", &device, dtype)?;
    let k_proj = loader.get("model.layers.0.self_attn.k_proj.weight", &device, dtype)?;
    let v_proj = loader.get("model.layers.0.self_attn.v_proj.weight", &device, dtype)?;

    let q_bias = loader.get("model.layers.0.self_attn.q_proj.bias", &device, dtype)?;
    let k_bias = loader.get("model.layers.0.self_attn.k_proj.bias", &device, dtype)?;
    let v_bias = loader.get("model.layers.0.self_attn.v_proj.bias", &device, dtype)?;

    println!("Q/K/V weights loaded");
    println!("  Q proj: {:?}", q_proj.dims());
    println!("  K proj: {:?}", k_proj.dims());
    println!("  V proj: {:?}", v_proj.dims());

    // Linear projections: [batch, seq, hidden] @ [out, in]^T + bias
    let hidden_2d = hidden.reshape((seq_len, hidden_size))?;

    let q = hidden_2d.matmul(&q_proj.t()?)?.broadcast_add(&q_bias)?;
    let k = hidden_2d.matmul(&k_proj.t()?)?.broadcast_add(&k_bias)?;
    let v = hidden_2d.matmul(&v_proj.t()?)?.broadcast_add(&v_bias)?;

    println!("After Q/K/V projection:");
    println!("  Q: {:?}", q.dims());
    println!("  K: {:?}", k.dims());
    println!("  V: {:?}", v.dims());

    check_tensor("Q projection", &q)?;
    check_tensor("K projection", &k)?;
    check_tensor("V projection", &v)?;

    // 3. Reshape for multi-head attention
    // Q: [seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
    // K/V: [seq, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq, head_dim]
    let q = q
        .reshape((1, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?;
    let k = k
        .reshape((1, seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
    let v = v
        .reshape((1, seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;

    println!("After reshape:");
    println!("  Q: {:?}", q.dims()); // [1, 40, seq, 128]
    println!("  K: {:?}", k.dims()); // [1, 8, seq, 128]
    println!("  V: {:?}", v.dims()); // [1, 8, seq, 128]

    // 4. Apply RoPE
    let (q, k) = apply_rope(&q, &k, seq_len, head_dim, rope_theta, 0, &device, dtype)?;
    println!("After RoPE:");
    check_tensor("Q+RoPE", &q)?;
    check_tensor("K+RoPE", &k)?;

    // 5. GQA: Repeat K/V to match num_heads
    let n_rep = num_heads / num_kv_heads; // 5
    let k = repeat_kv(&k, n_rep)?;
    let v = repeat_kv(&v, n_rep)?;
    println!("After KV repeat: K {:?}, V {:?}", k.dims(), v.dims());

    // 6. Attention
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale)?;

    // Causal mask
    let mask = get_causal_mask(seq_len, &device, dtype)?;
    let attn_weights = attn_weights.broadcast_add(&mask)?;

    println!("Attention weights: {:?}", attn_weights.dims());
    check_tensor("Attention weights (pre-softmax)", &attn_weights)?;

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    check_tensor("Attention weights (post-softmax)", &attn_weights)?;

    // Attention output
    let attn_output = attn_weights.matmul(&v)?;
    println!("Attention output: {:?}", attn_output.dims());
    check_tensor("Attention output", &attn_output)?;

    // 7. O projection
    let o_proj = loader.get("model.layers.0.self_attn.o_proj.weight", &device, dtype)?;
    let attn_output = attn_output
        .transpose(1, 2)?
        .reshape((seq_len, hidden_size))?;
    let attn_output = attn_output.matmul(&o_proj.t()?)?;
    println!("O projection output: {:?}", attn_output.dims());
    check_tensor("O projection", &attn_output)?;

    // Skip residual and MLP for this test - just do lm_head

    // 8. LM Head (simplified - just to test logit generation)
    let final_hidden = attn_output.i(seq_len - 1..seq_len)?; // Last token
    let lm_head = loader.get("lm_head.weight", &device, dtype)?;
    let logits = final_hidden.matmul(&lm_head.t()?)?;

    println!("\n=== Final Output ===");
    println!("Logits shape: {:?}", logits.dims());
    check_tensor("Logits", &logits)?;

    // Convert to f32 for argmax/softmax
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    println!(
        "Logits range: [{:.4}, {:.4}]",
        logits_vec.iter().cloned().fold(f32::INFINITY, f32::min),
        logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Top 5 predictions
    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 5 predictions:");
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

    // Note: This is only ONE layer so output won't be meaningful
    println!("\n(Note: Only processed 1 of 48 layers, output is NOT expected to be meaningful)");
    println!("(This test verifies that HoloTensor weights work in a real forward pass)");

    let stats = loader.stats();
    println!("\n=== Stats ===");
    println!("Tensors loaded: {}", stats.tensors_loaded);
    println!(
        "GPU reconstructions: {} ({} ms)",
        stats.gpu_reconstructions, stats.gpu_time_ms
    );
    println!(
        "CPU reconstructions: {} ({} ms)",
        stats.cpu_reconstructions, stats.cpu_time_ms
    );

    Ok(())
}

/// Check tensor for NaN/Inf and print stats
/// Note: -inf is allowed (expected in attention masks)
fn check_tensor(name: &str, tensor: &Tensor) -> Result<()> {
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let vec: Vec<f32> = tensor_f32.flatten_all()?.to_vec1()?;
    let nan_count = vec.iter().filter(|v| v.is_nan()).count();
    let pos_inf_count = vec.iter().filter(|v| **v == f32::INFINITY).count();
    let neg_inf_count = vec.iter().filter(|v| **v == f32::NEG_INFINITY).count();

    // -inf is expected in attention weights (causal mask), only +inf and NaN are errors
    if nan_count > 0 || pos_inf_count > 0 {
        println!(
            "  ✗ {}: NaN={}, +Inf={}, -Inf={}",
            name, nan_count, pos_inf_count, neg_inf_count
        );
        anyhow::bail!("{} contains NaN or +Inf", name);
    } else {
        let finite_vec: Vec<f32> = vec.iter().filter(|v| v.is_finite()).cloned().collect();
        let mean = if finite_vec.is_empty() {
            0.0
        } else {
            finite_vec.iter().sum::<f32>() / finite_vec.len() as f32
        };
        let max = finite_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = finite_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        if neg_inf_count > 0 {
            println!(
                "  ✓ {}: mean={:.6}, range=[{:.4}, {:.4}], -inf={} (expected)",
                name, mean, min, max, neg_inf_count
            );
        } else {
            println!(
                "  ✓ {}: mean={:.6}, range=[{:.4}, {:.4}]",
                name, mean, min, max
            );
        }
    }
    Ok(())
}

/// RMS normalization
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden_size = weight.dims()[0];
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(2)?;
    let eps_t = mean_sq.ones_like()? * eps;
    let norm_factor = (mean_sq + eps_t)?.sqrt()?.recip()?;
    let normalized = x.broadcast_mul(&norm_factor)?;
    let weight_3d = weight.reshape((1, 1, hidden_size))?;
    Ok(normalized.broadcast_mul(&weight_3d)?)
}

/// Apply RoPE to Q and K
fn apply_rope(
    q: &Tensor, // [batch, heads, seq, head_dim]
    k: &Tensor,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    pos_offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // Generate frequencies
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?.to_dtype(dtype)?;

    // Position indices
    let positions: Vec<f32> = (pos_offset..pos_offset + seq_len)
        .map(|p| p as f32)
        .collect();
    let positions = Tensor::from_vec(positions, (seq_len,), device)?.to_dtype(dtype)?;

    // Compute angles: [seq_len, half_dim]
    let angles = positions
        .reshape((seq_len, 1))?
        .broadcast_mul(&inv_freq.reshape((1, half_dim))?)?;

    // Compute sin and cos
    let cos = angles.cos()?;
    let sin = angles.sin()?;

    // Reshape for broadcasting: [1, 1, seq, half_dim]
    let cos = cos.reshape((1, 1, seq_len, half_dim))?;
    let sin = sin.reshape((1, 1, seq_len, half_dim))?;

    // Apply rotation
    let q_rot = apply_rotary_emb(q, &cos, &sin)?;
    let k_rot = apply_rotary_emb(k, &cos, &sin)?;

    Ok((q_rot, k_rot))
}

/// Apply rotary embedding to a single tensor
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half_d = d / 2;

    // Split into first and second half
    let x1 = x.narrow(3, 0, half_d)?;
    let x2 = x.narrow(3, half_d, half_d)?;

    // Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
    let y1 = x1
        .broadcast_mul(cos)?
        .broadcast_sub(&x2.broadcast_mul(sin)?)?;
    let y2 = x1
        .broadcast_mul(sin)?
        .broadcast_add(&x2.broadcast_mul(cos)?)?;

    // Concatenate back
    Tensor::cat(&[&y1, &y2], 3).map_err(Into::into)
}

/// Repeat K/V for GQA
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }

    let (b, kv_heads, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, s, d))?
        .reshape((b, kv_heads * n_rep, s, d))
        .map_err(Into::into)
}

/// Create causal attention mask
fn get_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = neg_inf;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?;
    Ok(mask.to_dtype(dtype)?)
}
