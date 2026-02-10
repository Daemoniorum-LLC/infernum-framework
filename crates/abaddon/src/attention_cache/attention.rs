//! Generic attention computation that works with any KV cache.
//!
//! This module provides cache-agnostic attention functions that automatically
//! dispatch to the appropriate implementation based on cache capabilities.
//!
//! ## Attention Modes
//!
//! - **Standard**: Traditional Q @ K^T @ V computation (O(N²) memory)
//! - **Flash**: Memory-efficient tiled attention (O(N) memory)
//!
//! Flash Attention is automatically used for sequences > 2048 tokens,
//! or can be forced via `attention_with_cache_mode`.

use super::KvCache;
use crate::flash_attention::{AttentionVariant, FlashAttention, FlashAttentionConfig};
use candle_core::{Result as CandleResult, Tensor, D};

/// Threshold for automatic Flash Attention selection (tokens).
#[allow(dead_code)]
const FLASH_ATTENTION_THRESHOLD: usize = 2048;

/// Compute scaled dot-product attention with a KV cache.
///
/// This function automatically handles both standard and fused attention paths:
/// - For caches that support fused attention (e.g., CUDA quantized), it uses
///   the cache's internal `forward_attention()` method.
/// - For standard caches with seq_len > 2048, it uses Flash Attention.
/// - For standard caches with shorter sequences, it uses standard attention.
///
/// # Arguments
/// * `q` - Query tensor, shape: `[batch, num_heads, q_len, head_dim]`
/// * `k` - New key tensor to append, shape: `[batch, num_kv_heads, seq_len, head_dim]`
/// * `v` - New value tensor to append, shape: `[batch, num_kv_heads, seq_len, head_dim]`
/// * `cache` - KV cache (any implementation)
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads
/// * `mask` - Optional attention mask (additive, typically causal)
///
/// # Returns
/// Attention output, shape: `[batch, num_heads, q_len, head_dim]`
pub fn attention_with_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &mut dyn KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    mask: Option<&Tensor>,
) -> CandleResult<Tensor> {
    // Append new K/V to cache
    cache.append(k, v)?;

    // Dispatch based on cache capabilities
    if cache.supports_fused_attention() {
        // Use fused attention (e.g., CUDA quantized)
        let head_dim = q.dims()[3];
        let scale = 1.0 / (head_dim as f32).sqrt();
        cache.forward_attention(q, num_heads, scale)
    } else {
        // Get full KV from cache
        let (k_full, v_full) = cache
            .get_kv()?
            .ok_or_else(|| candle_core::Error::Msg("Cache returned empty K/V".to_string()))?;

        // Repeat KV for GQA if needed
        let k_full = repeat_kv(k_full, num_heads / num_kv_heads)?;
        let v_full = repeat_kv(v_full, num_heads / num_kv_heads)?;

        // Auto-select attention mode based on sequence length
        let seq_len = k_full.dims()[2];
        let mode = AttentionVariant::recommended_for_seq_len(seq_len);

        match mode {
            AttentionVariant::Flash => {
                // Use memory-efficient Flash Attention for long sequences
                attention_flash(q, &k_full, &v_full, mask)
            },
            _ => {
                // Use standard attention for short sequences
                attention_standard(q, &k_full, &v_full, mask)
            },
        }
    }
}

/// Compute attention with explicit mode selection.
///
/// Use this when you want to force a specific attention algorithm.
pub fn attention_with_cache_mode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cache: &mut dyn KvCache,
    num_heads: usize,
    num_kv_heads: usize,
    mask: Option<&Tensor>,
    mode: AttentionVariant,
) -> CandleResult<Tensor> {
    // Append new K/V to cache
    cache.append(k, v)?;

    // Dispatch based on cache capabilities
    if cache.supports_fused_attention() {
        let head_dim = q.dims()[3];
        let scale = 1.0 / (head_dim as f32).sqrt();
        cache.forward_attention(q, num_heads, scale)
    } else {
        let (k_full, v_full) = cache
            .get_kv()?
            .ok_or_else(|| candle_core::Error::Msg("Cache returned empty K/V".to_string()))?;

        let k_full = repeat_kv(k_full, num_heads / num_kv_heads)?;
        let v_full = repeat_kv(v_full, num_heads / num_kv_heads)?;

        match mode {
            AttentionVariant::Flash => attention_flash(q, &k_full, &v_full, mask),
            _ => attention_standard(q, &k_full, &v_full, mask),
        }
    }
}

/// Standard scaled dot-product attention.
///
/// Computes: softmax(Q @ K^T / sqrt(d)) @ V
///
/// Memory: O(batch * heads * seq^2) for attention matrix
fn attention_standard(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> CandleResult<Tensor> {
    let head_dim = q.dims()[3];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q @ K^T
    let attn_scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let attn_scores = (attn_scores * (scale as f64))?;

    // Apply mask if provided
    let attn_scores = match mask {
        Some(m) => attn_scores.broadcast_add(m)?,
        None => attn_scores,
    };

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

    // Attention @ V
    attn_weights.matmul(v)
}

/// Flash Attention (memory-efficient tiled attention).
///
/// Uses online softmax algorithm to avoid materializing the full attention matrix.
///
/// Memory: O(batch * heads * seq * head_dim) - no O(seq^2) term
fn attention_flash(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> CandleResult<Tensor> {
    let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
    flash_attn.forward(q, k, v, mask, Some(true))
}

/// Repeat KV heads for Grouped Query Attention (GQA).
///
/// If `n_rep == 1`, returns the input unchanged.
/// Otherwise, expands the heads dimension by repeating each head `n_rep` times.
///
/// # Arguments
/// * `x` - Input tensor, shape: `[batch, num_kv_heads, seq_len, head_dim]`
/// * `n_rep` - Number of times to repeat each head
///
/// # Returns
/// Repeated tensor, shape: `[batch, num_kv_heads * n_rep, seq_len, head_dim]`
pub fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }

    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
}

/// Create a causal attention mask.
///
/// Creates an additive mask where positions that should be masked have
/// large negative values (effectively -inf after softmax).
///
/// # Arguments
/// * `seq_len` - Sequence length
/// * `dtype` - Data type for the mask
/// * `device` - Device for the mask
///
/// # Returns
/// Causal mask tensor, shape: `[1, 1, seq_len, seq_len]`
pub fn create_causal_mask(
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();

    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

/// Configuration for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA).
    pub num_kv_heads: usize,
    /// Dimension of each head.
    pub head_dim: usize,
    /// Whether to use causal masking.
    pub causal: bool,
}

impl AttentionConfig {
    /// Create a new attention configuration.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            causal: true,
        }
    }

    /// Set whether to use causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Get the attention scale factor.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Get the GQA repetition factor.
    pub fn gqa_rep(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention_cache::StandardCache;
    use candle_core::{DType, Device};

    #[test]
    fn test_repeat_kv_no_repeat() -> CandleResult<()> {
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        let y = repeat_kv(x.clone(), 1)?;
        assert_eq!(x.dims(), y.dims());
        Ok(())
    }

    #[test]
    fn test_repeat_kv_gqa() -> CandleResult<()> {
        let device = Device::Cpu;
        let x = Tensor::zeros((1, 2, 10, 64), DType::F32, &device)?;
        let y = repeat_kv(x, 4)?;
        assert_eq!(y.dims(), &[1, 8, 10, 64]);
        Ok(())
    }

    #[test]
    fn test_causal_mask() -> CandleResult<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, DType::F32, &device)?;

        assert_eq!(mask.dims(), &[1, 1, 4, 4]);

        let values: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Check diagonal and below are 0
        assert_eq!(values[0], 0.0); // (0,0)
        assert_eq!(values[4], 0.0); // (1,0)
        assert_eq!(values[5], 0.0); // (1,1)

        // Check above diagonal is -inf
        assert!(values[1].is_infinite() && values[1] < 0.0); // (0,1)
        assert!(values[2].is_infinite() && values[2] < 0.0); // (0,2)

        Ok(())
    }

    #[test]
    fn test_attention_with_standard_cache() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 8;
        let num_kv_heads = 2;
        let seq_len = 5;
        let head_dim = 64;

        let mut cache = StandardCache::new();

        let q = Tensor::randn(0.0f32, 0.1, (batch, num_heads, seq_len, head_dim), &device)?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch, num_kv_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch, num_kv_heads, seq_len, head_dim),
            &device,
        )?;

        let output = attention_with_cache(&q, &k, &v, &mut cache, num_heads, num_kv_heads, None)?;

        assert_eq!(output.dims(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(cache.seq_len(), seq_len);

        Ok(())
    }

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(32, 8, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.gqa_rep(), 4);
        assert!((config.scale() - 0.0884).abs() < 0.001);
    }
}
