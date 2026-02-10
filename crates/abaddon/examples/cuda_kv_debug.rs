//! Debug test for CUDA quantized KV cache.
//!
//! Run with:
//! ```bash
//! LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo run --example cuda_kv_debug --features cuda --release
//! ```

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    cuda_main()
}

#[cfg(feature = "cuda")]
fn cuda_main() -> anyhow::Result<()> {
    use std::time::Instant;

    use abaddon::kv_cache_quant_cuda::cuda::{CudaQuantizedKvCache, Int8AttentionContext};
    use candle_core::{DType, Device, IndexOp, Tensor};

    println!("=== CUDA Quantized KV Cache Debug Test ===\n");

    // Check CUDA availability
    if !candle_core::utils::cuda_is_available() {
        println!("CUDA not available, skipping test");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("Using device: {:?}", device);

    // Qwen2.5-0.5B dimensions
    let batch = 1;
    let num_heads = 14; // Q heads (from config)
    let num_kv_heads = 2; // K/V heads (GQA: 7 Q heads per KV head)
    let seq_len = 5; // Like a short prompt
    let head_dim = 64; // 896 / 14 = 64

    println!("\nTest dimensions:");
    println!(
        "  batch={}, num_heads={}, num_kv_heads={}, seq_len={}, head_dim={}",
        batch, num_heads, num_kv_heads, seq_len, head_dim
    );
    println!(
        "  GQA ratio: {} Q heads per KV head",
        num_heads / num_kv_heads
    );

    // Create known test data
    // Q: [batch, num_heads, seq_len, head_dim]
    let q_data: Vec<f32> = (0..batch * num_heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.1) - 1.0)
        .collect();
    let q = Tensor::from_vec(
        q_data.clone(),
        (batch, num_heads, seq_len, head_dim),
        &device,
    )?
    .to_dtype(DType::BF16)?;

    // K: [batch, num_kv_heads, seq_len, head_dim]
    let k_data: Vec<f32> = (0..batch * num_kv_heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.05) - 0.5)
        .collect();
    let k = Tensor::from_vec(
        k_data.clone(),
        (batch, num_kv_heads, seq_len, head_dim),
        &device,
    )?
    .to_dtype(DType::BF16)?;

    // V: [batch, num_kv_heads, seq_len, head_dim]
    let v_data: Vec<f32> = (0..batch * num_kv_heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.02))
        .collect();
    let v = Tensor::from_vec(
        v_data.clone(),
        (batch, num_kv_heads, seq_len, head_dim),
        &device,
    )?
    .to_dtype(DType::BF16)?;

    println!(
        "\nQ sample (first head, first token): {:?}",
        q.i((0, 0, 0, ..))?.to_dtype(DType::F32)?.to_vec1::<f32>()?
    );
    println!(
        "K sample (first kv_head, first token): {:?}",
        k.i((0, 0, 0, ..))?.to_dtype(DType::F32)?.to_vec1::<f32>()?
    );
    println!(
        "V sample (first kv_head, first token): {:?}",
        v.i((0, 0, 0, ..))?.to_dtype(DType::F32)?.to_vec1::<f32>()?
    );

    // ========== Test 1: Standard attention (ground truth) ==========
    println!("\n--- Test 1: Standard BF16 Attention (Ground Truth) ---");

    // Repeat K/V for GQA
    let k_repeated = repeat_kv(&k, num_heads / num_kv_heads)?;
    let v_repeated = repeat_kv(&v, num_heads / num_kv_heads)?;

    println!("K repeated shape: {:?}", k_repeated.dims());

    // Q @ K^T
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_scores = q.matmul(&k_repeated.transpose(2, 3)?)?;
    let attn_scores = (attn_scores * scale)?;

    println!(
        "Attention scores (head 0, token 0): {:?}",
        attn_scores
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // Apply causal mask (for fair comparison with CUDA path)
    let attn_scores = apply_causal_mask(&attn_scores, 0)?;
    println!(
        "Attention scores after causal mask (head 0, token 0): {:?}",
        attn_scores
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
    println!(
        "Attention weights (head 0, token 0): {:?}",
        attn_weights
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // attn @ V
    let output_std = attn_weights.matmul(&v_repeated)?;
    println!(
        "Output (head 0, token 0): {:?}",
        output_std
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // ========== Test 2: CUDA Quantized Attention ==========
    println!("\n--- Test 2: CUDA INT8 Quantized Attention ---");

    let mut cuda_cache = CudaQuantizedKvCache::new(num_kv_heads, head_dim, 0)?;

    // Append K/V to cache
    cuda_cache.append(&k, &v)?;
    println!("Cache seq_len after append: {}", cuda_cache.seq_len());

    // Compute attention
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let output_cuda = cuda_cache.forward_attention(&q, num_heads, attn_scale)?;

    println!("CUDA output shape: {:?}", output_cuda.dims());
    println!(
        "CUDA output (head 0, token 0): {:?}",
        output_cuda
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // ========== Test 3: Compare outputs ==========
    println!("\n--- Test 3: Compare Standard vs CUDA ---");

    let std_flat = output_std
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cuda_flat = output_cuda
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut num_large_diff = 0;

    for (i, (s, c)) in std_flat.iter().zip(cuda_flat.iter()).enumerate() {
        let diff: f32 = (s - c).abs();
        sum_diff += diff;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.1 {
            num_large_diff += 1;
            if num_large_diff <= 5 {
                println!(
                    "  Large diff at {}: std={:.4}, cuda={:.4}, diff={:.4}",
                    i, s, c, diff
                );
            }
        }
    }

    let mean_diff = sum_diff / std_flat.len() as f32;
    println!("\nDifference statistics:");
    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);
    println!(
        "  Large diffs (>0.1): {}/{}",
        num_large_diff,
        std_flat.len()
    );

    // ========== Test 4: Test with single token decode ==========
    println!("\n--- Test 4: Single Token Decode ---");

    // Create new Q for decode (single token)
    let q_decode_data: Vec<f32> = (0..batch * num_heads * 1 * head_dim)
        .map(|i| (i as f32 * 0.1) + 0.5)
        .collect();
    let q_decode = Tensor::from_vec(q_decode_data, (batch, num_heads, 1, head_dim), &device)?
        .to_dtype(DType::BF16)?;

    // Standard decode
    let attn_scores_decode = q_decode.matmul(&k_repeated.transpose(2, 3)?)?;
    let attn_scores_decode = (attn_scores_decode * scale)?;
    let attn_weights_decode = candle_nn::ops::softmax_last_dim(&attn_scores_decode)?;
    let output_std_decode = attn_weights_decode.matmul(&v_repeated)?;

    println!(
        "Standard decode output (head 0): {:?}",
        output_std_decode
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // CUDA decode (cache already has K/V)
    let output_cuda_decode = cuda_cache.forward_attention(&q_decode, num_heads, attn_scale)?;

    println!(
        "CUDA decode output (head 0): {:?}",
        output_cuda_decode
            .i((0, 0, 0, ..))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );

    // ========== Test 5: Check intermediate values ==========
    println!("\n--- Test 5: Debug Intermediate Values ---");

    // Create a fresh context to test kernels directly
    let mut attn_ctx = Int8AttentionContext::new(0)?;
    attn_ctx.load_kernels()?;

    // Manually quantize K for inspection
    let k_f32 = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    println!("\nK values (first 16): {:?}", &k_f32[..16.min(k_f32.len())]);

    // Check quantization range
    let k_min = k_f32.iter().cloned().fold(f32::INFINITY, f32::min);
    let k_max = k_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("K range: [{:.4}, {:.4}]", k_min, k_max);

    // Manual quantization check
    let k_abs_max = k_min.abs().max(k_max.abs());
    let k_scale_expected = k_abs_max / 127.0;
    println!("Expected scale for uniform quant: {:.6}", k_scale_expected);

    // ========== Test 6: Multiple decode steps (like real generation) ==========
    println!("\n--- Test 6: Multi-step Generation Simulation ---");

    // Fresh caches
    let mut std_k_cache: Option<Tensor> = None;
    let mut std_v_cache: Option<Tensor> = None;
    let mut cuda_cache2 = CudaQuantizedKvCache::new(num_kv_heads, head_dim, 0)?;

    // Prefill with 5 tokens
    let prefill_q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, 5, head_dim), &device)?
        .to_dtype(DType::BF16)?;
    let prefill_k = Tensor::randn(0.0f32, 1.0, (batch, num_kv_heads, 5, head_dim), &device)?
        .to_dtype(DType::BF16)?;
    let prefill_v = Tensor::randn(0.0f32, 1.0, (batch, num_kv_heads, 5, head_dim), &device)?
        .to_dtype(DType::BF16)?;

    // Standard: store in cache
    std_k_cache = Some(prefill_k.clone());
    std_v_cache = Some(prefill_v.clone());

    // CUDA: append to cache
    cuda_cache2.append(&prefill_k, &prefill_v)?;

    println!(
        "After prefill: std_cache_len=5, cuda_cache_len={}",
        cuda_cache2.seq_len()
    );

    // Compute prefill attention (with causal mask for fair comparison)
    let k_rep = repeat_kv(&prefill_k, num_heads / num_kv_heads)?;
    let v_rep = repeat_kv(&prefill_v, num_heads / num_kv_heads)?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn = prefill_q.matmul(&k_rep.transpose(2, 3)?)?;
    let attn = (attn * scale)?;
    let attn = apply_causal_mask(&attn, 0)?; // Apply causal mask for prefill (cache_offset=0)
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    let std_prefill_out = attn.matmul(&v_rep)?;

    let cuda_prefill_out = cuda_cache2.forward_attention(&prefill_q, num_heads, scale as f32)?;

    let std_prefill_f32 = std_prefill_out.to_dtype(DType::F32)?;
    let cuda_prefill_f32 = cuda_prefill_out.to_dtype(DType::F32)?;
    let prefill_diff = std_prefill_f32
        .sub(&cuda_prefill_f32)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("Prefill max diff (with causal mask): {:.6}", prefill_diff);

    // Now do 10 decode steps
    for step in 0..10 {
        // New single-token Q, K, V for this decode step
        let step_q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, 1, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let step_k = Tensor::randn(0.0f32, 1.0, (batch, num_kv_heads, 1, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let step_v = Tensor::randn(0.0f32, 1.0, (batch, num_kv_heads, 1, head_dim), &device)?
            .to_dtype(DType::BF16)?;

        // Standard: append to cache
        let full_k = Tensor::cat(&[std_k_cache.as_ref().unwrap(), &step_k], 2)?;
        let full_v = Tensor::cat(&[std_v_cache.as_ref().unwrap(), &step_v], 2)?;
        std_k_cache = Some(full_k.clone());
        std_v_cache = Some(full_v.clone());

        // CUDA: append to cache
        cuda_cache2.append(&step_k, &step_v)?;

        // Compute attention
        let k_rep = repeat_kv(&full_k, num_heads / num_kv_heads)?;
        let v_rep = repeat_kv(&full_v, num_heads / num_kv_heads)?;
        let attn = step_q.matmul(&k_rep.transpose(2, 3)?)?;
        let attn = (attn * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let std_out = attn.matmul(&v_rep)?;

        let cuda_out = cuda_cache2.forward_attention(&step_q, num_heads, scale as f32)?;

        let std_out_f32 = std_out.to_dtype(DType::F32)?;
        let cuda_out_f32 = cuda_out.to_dtype(DType::F32)?;
        let diff = std_out_f32
            .sub(&cuda_out_f32)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let mean_std = std_out_f32.abs()?.mean_all()?.to_scalar::<f32>()?;
        let mean_cuda = cuda_out_f32.abs()?.mean_all()?.to_scalar::<f32>()?;

        println!(
            "Step {}: cache_len={}, max_diff={:.6}, std_mean={:.4}, cuda_mean={:.4}",
            step,
            cuda_cache2.seq_len(),
            diff,
            mean_std,
            mean_cuda
        );

        if diff > 1.0 {
            println!("  WARNING: Large difference detected!");
            // Print some sample values
            let std_sample = std_out
                .i((0, 0, 0, ..8))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let cuda_sample = cuda_out
                .i((0, 0, 0, ..8))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            println!("  std[:8]:  {:?}", std_sample);
            println!("  cuda[:8]: {:?}", cuda_sample);
        }
    }

    println!("\n=== Debug Test Complete ===");
    Ok(())
}

/// Repeat KV heads for GQA
#[cfg(feature = "cuda")]
fn repeat_kv(x: &candle_core::Tensor, n_rep: usize) -> anyhow::Result<candle_core::Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
    Ok(x)
}

/// Apply causal mask to attention scores.
/// Scores shape: [batch, num_heads, q_len, kv_len]
/// cache_offset: number of KV positions that existed before the current Q tokens
#[cfg(feature = "cuda")]
fn apply_causal_mask(
    scores: &candle_core::Tensor,
    cache_offset: usize,
) -> anyhow::Result<candle_core::Tensor> {
    let (batch, num_heads, q_len, kv_len) = scores.dims4()?;
    let device = scores.device();
    let dtype = scores.dtype();

    // Create causal mask: mask[q_pos, kv_pos] = -inf if kv_pos > cache_offset + q_pos
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for q_pos in 0..q_len {
        let max_visible_kv = cache_offset + q_pos;
        for kv_pos in (max_visible_kv + 1)..kv_len {
            mask_data[q_pos * kv_len + kv_pos] = f32::NEG_INFINITY;
        }
    }

    let mask = candle_core::Tensor::from_vec(mask_data, (1, 1, q_len, kv_len), device)?
        .to_dtype(dtype)?
        .broadcast_as((batch, num_heads, q_len, kv_len))?;

    Ok((scores + mask)?)
}
