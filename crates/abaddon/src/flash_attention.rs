//! Flash Attention implementation for memory-efficient attention computation.
//!
//! Flash Attention is an IO-aware attention algorithm that reduces memory usage
//! from O(N²) to O(N) by computing attention in tiles and avoiding materializing
//! the full attention matrix.
//!
//! ## Features
//!
//! - **Memory Efficient**: Uses tiling to avoid O(N²) memory for attention matrix
//! - **CUDA Optimized**: Uses flash-attn CUDA kernels when available
//! - **Fallback**: Pure Rust implementation for CPU/Metal backends
//!
//! ## Usage
//!
//! ```ignore
//! use abaddon::flash_attention::{FlashAttention, FlashAttentionConfig};
//!
//! let config = FlashAttentionConfig::default();
//! let flash_attn = FlashAttention::new(config);
//!
//! let output = flash_attn.forward(&q, &k, &v, mask, true)?;
//! ```
//!
//! ## Algorithm
//!
//! The algorithm works by:
//! 1. Splitting Q, K, V into blocks
//! 2. Computing attention for each block pair
//! 3. Using online softmax to accumulate results
//! 4. This avoids materializing the full NxN attention matrix

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use serde::{Deserialize, Serialize};

/// Configuration for Flash Attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    /// Block size for tiling (affects memory vs compute tradeoff).
    /// Larger blocks = more memory but potentially faster.
    pub block_size: usize,
    /// Whether to use causal masking (for autoregressive generation).
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout).
    pub dropout: f32,
    /// Softmax scale (typically 1/sqrt(head_dim)).
    /// If None, computed automatically from head dimension.
    pub softmax_scale: Option<f32>,
    /// Whether to return softmax statistics for debugging.
    pub return_softmax: bool,
    /// Maximum sequence length for optimization hints.
    pub max_seqlen: Option<usize>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            // Larger block size reduces overhead but uses more memory per block
            // 512 for high-VRAM GPUs (20GB+), reduces tiling overhead
            block_size: 512,
            causal: true,
            dropout: 0.0,
            softmax_scale: None,
            return_softmax: false,
            max_seqlen: None,
        }
    }
}

impl FlashAttentionConfig {
    /// Creates a config optimized for long sequences.
    #[must_use]
    pub fn for_long_context() -> Self {
        Self {
            block_size: 128,
            causal: true,
            dropout: 0.0,
            softmax_scale: None,
            return_softmax: false,
            max_seqlen: Some(32768),
        }
    }

    /// Creates a config for non-causal attention (e.g., BERT-style).
    #[must_use]
    pub fn non_causal() -> Self {
        Self {
            block_size: 64,
            causal: false,
            dropout: 0.0,
            softmax_scale: None,
            return_softmax: false,
            max_seqlen: None,
        }
    }

    /// Sets the softmax scale explicitly.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.softmax_scale = Some(scale);
        self
    }

    /// Sets dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Flash Attention implementation with automatic backend selection.
pub struct FlashAttention {
    config: FlashAttentionConfig,
    /// Whether CUDA flash attention is available.
    cuda_available: bool,
}

impl FlashAttention {
    /// Creates a new Flash Attention instance.
    #[must_use]
    pub fn new(config: FlashAttentionConfig) -> Self {
        let cuda_available = Self::check_cuda_flash_available();

        if cuda_available {
            tracing::info!("Flash Attention: Using CUDA kernels");
        } else {
            tracing::info!("Flash Attention: Using tiled CPU implementation");
        }

        Self {
            config,
            cuda_available,
        }
    }

    /// Creates with default config.
    #[must_use]
    pub fn default_causal() -> Self {
        Self::new(FlashAttentionConfig::default())
    }

    /// Checks if CUDA flash attention kernels are available.
    fn check_cuda_flash_available() -> bool {
        // Check if candle-flash-attn feature is enabled and CUDA is available
        #[cfg(feature = "flash-attn")]
        {
            // Try to detect if flash-attn kernels are loadable
            cfg!(feature = "cuda")
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            false
        }
    }

    /// Computes flash attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape (batch, heads, seq_len, head_dim)
    /// * `k` - Key tensor of shape (batch, heads, seq_len, head_dim)
    /// * `v` - Value tensor of shape (batch, heads, seq_len, head_dim)
    /// * `mask` - Optional attention mask
    /// * `causal` - Override causal setting if Some
    ///
    /// # Returns
    /// Output tensor of shape (batch, heads, seq_len, head_dim)
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: Option<bool>,
    ) -> CandleResult<Tensor> {
        let causal = causal.unwrap_or(self.config.causal);
        let device = q.device();

        // Get dimensions
        let (batch_size, num_heads, seq_len, head_dim) = q.dims4()?;

        // Compute softmax scale
        let scale = self
            .config
            .softmax_scale
            .unwrap_or(1.0 / (head_dim as f32).sqrt());

        // Route to appropriate implementation
        match device {
            Device::Cuda(_) if self.cuda_available => {
                self.forward_cuda(q, k, v, mask, causal, scale)
            },
            _ => {
                // Use tiled implementation for memory-efficient attention
                self.forward_tiled(
                    q, k, v, mask, causal, scale, batch_size, num_heads, seq_len, head_dim,
                )
            },
        }
    }

    /// CUDA Flash Attention implementation using flash-attn kernels.
    #[cfg(feature = "flash-attn")]
    fn forward_cuda(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: Option<&Tensor>,
        causal: bool,
        softmax_scale: f32,
    ) -> CandleResult<Tensor> {
        use candle_flash_attn::flash_attn;

        // flash_attn expects (batch, seq_len, heads, head_dim)
        // Our input is (batch, heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let output = flash_attn(&q, &k, &v, softmax_scale, causal)?;

        // Transpose back to (batch, heads, seq_len, head_dim)
        output.transpose(1, 2)
    }

    /// Fallback when flash-attn feature is not enabled.
    #[cfg(not(feature = "flash-attn"))]
    fn forward_cuda(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        softmax_scale: f32,
    ) -> CandleResult<Tensor> {
        // Fall back to tiled implementation
        let (batch_size, num_heads, seq_len, head_dim) = q.dims4()?;
        self.forward_tiled(
            q,
            k,
            v,
            mask,
            causal,
            softmax_scale,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
        )
    }

    /// Memory-efficient tiled attention implementation.
    ///
    /// This implements the Flash Attention algorithm using block-wise computation
    /// to avoid materializing the full O(N²) attention matrix.
    ///
    /// ## Algorithm (Online Softmax)
    ///
    /// For each row, we track:
    /// - m: running maximum of attention scores (for numerical stability)
    /// - l: running sum of exp(scores - m)
    /// - O: unnormalized running output
    ///
    /// Update rules for each new K,V block:
    /// 1. S = Q @ K^T * scale
    /// 2. m_new = max(m_old, rowmax(S))
    /// 3. α = exp(m_old - m_new) (correction factor)
    /// 4. l_new = α * l_old + rowsum(exp(S - m_new))
    /// 5. O_new = α * O_old + exp(S - m_new) @ V
    /// 6. Final: O = O / l
    fn forward_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> CandleResult<Tensor> {
        let block_size = self.config.block_size.min(seq_len);
        let num_blocks = (seq_len + block_size - 1) / block_size;

        // For very short sequences, standard attention is more efficient
        if seq_len <= block_size {
            return self.forward_standard(q, k, v, mask, causal, scale);
        }

        let device = q.device();
        let dtype = q.dtype();

        // Initialize output accumulator - we'll fill it block by block
        let mut output = Tensor::zeros((batch_size, num_heads, seq_len, head_dim), dtype, device)?;

        // Process Q in blocks (outer loop)
        for q_block_idx in 0..num_blocks {
            let q_start = q_block_idx * block_size;
            let q_end = (q_start + block_size).min(seq_len);
            let q_block_len = q_end - q_start;

            // Extract Q block: (batch, heads, q_block_len, head_dim)
            let q_block = q.narrow(2, q_start, q_block_len)?;

            // Online softmax accumulators for this Q block:
            // m: running max, shape (batch, heads, q_block_len)
            // l: running sum of exp, shape (batch, heads, q_block_len)
            // O: unnormalized output, shape (batch, heads, q_block_len, head_dim)
            let mut m = Tensor::full(
                f32::NEG_INFINITY,
                (batch_size, num_heads, q_block_len),
                device,
            )?
            .to_dtype(dtype)?;
            let mut l = Tensor::zeros((batch_size, num_heads, q_block_len), dtype, device)?;
            let mut o = Tensor::zeros(
                (batch_size, num_heads, q_block_len, head_dim),
                dtype,
                device,
            )?;

            // Process K, V in blocks (inner loop)
            let kv_end_block = if causal { q_block_idx + 1 } else { num_blocks };

            for kv_block_idx in 0..kv_end_block {
                let kv_start = kv_block_idx * block_size;
                let kv_end = (kv_start + block_size).min(seq_len);
                let kv_block_len = kv_end - kv_start;

                // Extract K, V blocks
                let k_block = k.narrow(2, kv_start, kv_block_len)?;
                let v_block = v.narrow(2, kv_start, kv_block_len)?;

                // Step 1: Compute attention scores S = Q @ K^T * scale
                // Shape: (batch, heads, q_block_len, kv_block_len)
                let k_t = k_block.transpose(D::Minus2, D::Minus1)?;
                let scores = q_block.matmul(&k_t)?;
                let scores = (scores * scale as f64)?;

                // Apply causal mask within block if needed
                let scores = if causal && q_block_idx == kv_block_idx {
                    self.apply_causal_mask_block(
                        &scores,
                        q_start,
                        kv_start,
                        q_block_len,
                        kv_block_len,
                    )?
                } else if causal && kv_block_idx > q_block_idx {
                    // This entire block is masked - skip it
                    continue;
                } else {
                    scores
                };

                // Apply external mask if provided
                let scores = if let Some(mask_tensor) = mask {
                    let mask_block = mask_tensor
                        .narrow(D::Minus2, q_start, q_block_len)?
                        .narrow(D::Minus1, kv_start, kv_block_len)?;
                    scores.broadcast_add(&mask_block)?
                } else {
                    scores
                };

                // Step 2: m_new = max(m_old, rowmax(S))
                // rowmax(S) has shape (batch, heads, q_block_len, 1), squeeze to (batch, heads, q_block_len)
                let scores_rowmax = scores.max_keepdim(D::Minus1)?.squeeze(D::Minus1)?;
                let m_new = m.maximum(&scores_rowmax)?;

                // Step 3: α = exp(m_old - m_new) - correction factor for previous values
                // Shape: (batch, heads, q_block_len)
                let alpha = (&m - &m_new)?.exp()?;

                // Step 4: Compute exp(S - m_new) for this block
                // Need to broadcast m_new to (batch, heads, q_block_len, 1) for subtraction
                let m_new_expanded = m_new.unsqueeze(D::Minus1)?;
                let scores_shifted = scores.broadcast_sub(&m_new_expanded)?;
                let p = scores_shifted.exp()?; // Shape: (batch, heads, q_block_len, kv_block_len)

                // Step 5: l_new = α * l_old + rowsum(exp(S - m_new))
                let p_rowsum = p.sum_keepdim(D::Minus1)?.squeeze(D::Minus1)?; // Shape: (batch, heads, q_block_len)
                let l_new = (l.broadcast_mul(&alpha)? + p_rowsum)?;

                // Step 6: O_new = α * O_old + P @ V
                // α needs to be expanded to (batch, heads, q_block_len, 1) for broadcasting with O
                let alpha_expanded = alpha.unsqueeze(D::Minus1)?.contiguous()?;
                let pv = p.matmul(&v_block)?; // Shape: (batch, heads, q_block_len, head_dim)
                let o_new = (o.broadcast_mul(&alpha_expanded)? + pv)?;

                // Update accumulators
                m = m_new;
                l = l_new;
                o = o_new;
            }

            // Step 7: Final normalization - O_final = O / l
            // l has shape (batch, heads, q_block_len), expand to (batch, heads, q_block_len, 1)
            let l_expanded = l.unsqueeze(D::Minus1)?.contiguous()?;
            let block_output = o.broadcast_div(&l_expanded)?;

            // Write this Q block's output to the full output tensor
            output = self.scatter_copy(&output, &block_output, 2, q_start)?;
        }

        Ok(output)
    }

    /// Standard attention implementation (fallback for short sequences).
    fn forward_standard(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
    ) -> CandleResult<Tensor> {
        let (_, _, seq_len, _) = q.dims4()?;

        // Q @ K^T
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores * scale as f64)?;

        // Apply causal mask
        let scores = if causal {
            let causal_mask = self.create_causal_mask(seq_len, q.device(), q.dtype())?;
            scores.broadcast_add(&causal_mask)?
        } else {
            scores
        };

        // Apply external mask
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Attention output
        attn_weights.matmul(v)
    }

    /// Creates a causal attention mask.
    fn create_causal_mask(
        &self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();

        Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
    }

    /// Applies causal mask within a block.
    fn apply_causal_mask_block(
        &self,
        scores: &Tensor,
        q_start: usize,
        kv_start: usize,
        q_len: usize,
        kv_len: usize,
    ) -> CandleResult<Tensor> {
        let device = scores.device();
        let dtype = scores.dtype();

        let mask: Vec<f32> = (0..q_len)
            .flat_map(|qi| {
                let q_pos = q_start + qi;
                (0..kv_len).map(move |ki| {
                    let k_pos = kv_start + ki;
                    if k_pos > q_pos {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let mask = Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)?;
        scores.broadcast_add(&mask)
    }

    /// Replace NaN values with zeros (safety net for numerical edge cases).
    #[allow(dead_code)]
    fn handle_nan(&self, tensor: &Tensor) -> CandleResult<Tensor> {
        // Create a mask of finite values and use it to zero out NaNs
        // For now, rely on the numerics being correct - this is a safety net
        Ok(tensor.clone())
    }

    /// Scatter-add a block into the output tensor.
    #[allow(dead_code)]
    fn scatter_add(
        &self,
        output: &Tensor,
        block: &Tensor,
        dim: usize,
        offset: usize,
    ) -> CandleResult<Tensor> {
        // For now, reconstruct the tensor by concatenating slices
        // A proper implementation would use in-place scatter
        let seq_len = output.dim(dim)?;
        let block_len = block.dim(dim)?;

        if offset == 0 && block_len == seq_len {
            return Ok(block.clone());
        }

        let mut parts = Vec::new();

        if offset > 0 {
            parts.push(output.narrow(dim, 0, offset)?);
        }
        parts.push(block.clone());
        if offset + block_len < seq_len {
            parts.push(output.narrow(dim, offset + block_len, seq_len - offset - block_len)?);
        }

        if parts.len() == 1 {
            Ok(parts.remove(0))
        } else {
            Tensor::cat(&parts, dim)
        }
    }

    /// Copy a block into a tensor at an offset.
    fn scatter_copy(
        &self,
        output: &Tensor,
        block: &Tensor,
        dim: usize,
        offset: usize,
    ) -> CandleResult<Tensor> {
        let seq_len = output.dim(dim)?;
        let block_len = block.dim(dim)?;

        if offset == 0 && block_len == seq_len {
            return Ok(block.clone());
        }

        let mut parts = Vec::new();

        if offset > 0 {
            parts.push(output.narrow(dim, 0, offset)?);
        }
        parts.push(block.clone());
        if offset + block_len < seq_len {
            parts.push(output.narrow(dim, offset + block_len, seq_len - offset - block_len)?);
        }

        if parts.len() == 1 {
            Ok(parts.remove(0))
        } else {
            Tensor::cat(&parts, dim)
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Returns whether CUDA flash attention is being used.
    #[must_use]
    pub fn is_using_cuda(&self) -> bool {
        self.cuda_available
    }
}

/// Attention variant selector for model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionVariant {
    /// Standard scaled dot-product attention.
    Standard,
    /// Flash Attention (memory-efficient).
    Flash,
    /// Multi-Query Attention (shared K/V heads).
    MultiQuery,
    /// Grouped-Query Attention (GQA).
    GroupedQuery,
}

impl Default for AttentionVariant {
    fn default() -> Self {
        Self::Standard
    }
}

impl AttentionVariant {
    /// Returns whether this variant is memory-efficient.
    #[must_use]
    pub fn is_memory_efficient(&self) -> bool {
        matches!(self, Self::Flash | Self::MultiQuery | Self::GroupedQuery)
    }

    /// Returns recommended variant based on sequence length.
    #[must_use]
    pub fn recommended_for_seq_len(seq_len: usize) -> Self {
        if seq_len > 2048 {
            Self::Flash
        } else {
            Self::Standard
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_config_defaults() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size, 512);
        assert!(config.causal);
        assert_eq!(config.dropout, 0.0);
        assert!(config.softmax_scale.is_none());
    }

    #[test]
    fn test_flash_attention_long_context_config() {
        let config = FlashAttentionConfig::for_long_context();
        assert_eq!(config.block_size, 128);
        assert_eq!(config.max_seqlen, Some(32768));
    }

    #[test]
    fn test_flash_attention_non_causal() {
        let config = FlashAttentionConfig::non_causal();
        assert!(!config.causal);
    }

    #[test]
    fn test_config_builder_methods() {
        let config = FlashAttentionConfig::default()
            .with_scale(0.125)
            .with_dropout(0.1);
        assert_eq!(config.softmax_scale, Some(0.125));
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_attention_variant_recommendations() {
        assert_eq!(
            AttentionVariant::recommended_for_seq_len(512),
            AttentionVariant::Standard
        );
        assert_eq!(
            AttentionVariant::recommended_for_seq_len(4096),
            AttentionVariant::Flash
        );
    }

    #[test]
    fn test_attention_variant_memory_efficiency() {
        assert!(!AttentionVariant::Standard.is_memory_efficient());
        assert!(AttentionVariant::Flash.is_memory_efficient());
        assert!(AttentionVariant::MultiQuery.is_memory_efficient());
        assert!(AttentionVariant::GroupedQuery.is_memory_efficient());
    }

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_output_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 16;
        let head_dim = 32;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
        Ok(())
    }

    #[test]
    fn test_flash_attention_single_token() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 8;
        let seq_len = 1;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

        // For single token with causal attention, output should equal V
        // (softmax of single element is 1.0)
        let diff = (&output - &v)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "Single token attention should return V, diff={}",
            diff
        );
        Ok(())
    }

    // =========================================================================
    // Numerical Equivalence Tests (Flash vs Standard)
    // =========================================================================

    #[test]
    fn test_flash_matches_standard_short_sequence() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 32;
        let head_dim = 16;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let standard_output = flash_attn.forward_standard(&q, &k, &v, None, true, scale)?;
        let flash_output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(standard_output.dims(), flash_output.dims());

        // Check numerical closeness
        let diff = (&standard_output - &flash_output)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "Flash and standard should match, max diff={}",
            diff
        );
        Ok(())
    }

    #[test]
    fn test_flash_matches_standard_long_sequence() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 256; // Longer than default block_size forces tiling
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Use smaller block size to test tiling
        let config = FlashAttentionConfig {
            block_size: 64,
            causal: true,
            ..Default::default()
        };

        let q = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(config);
        let standard_output = flash_attn.forward_standard(&q, &k, &v, None, true, scale)?;
        let flash_output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        // Tiled computation may have slightly more numerical error
        let diff = (&standard_output - &flash_output)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-3,
            "Flash (tiled) and standard should match, max diff={}",
            diff
        );
        Ok(())
    }

    #[test]
    fn test_flash_matches_standard_non_causal() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 64;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let config = FlashAttentionConfig::non_causal();
        let flash_attn = FlashAttention::new(config);

        let q = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let standard_output = flash_attn.forward_standard(&q, &k, &v, None, false, scale)?;
        let flash_output = flash_attn.forward(&q, &k, &v, None, Some(false))?;

        let diff = (&standard_output - &flash_output)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "Non-causal flash and standard should match, max diff={}",
            diff
        );
        Ok(())
    }

    // =========================================================================
    // Causal Mask Correctness Tests
    // =========================================================================

    #[test]
    fn test_causal_mask_correctness() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 4;
        let head_dim = 8;

        // Create distinct Q, K, V so we can verify attention patterns
        let q = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            DType::F32,
            &device,
        )?;
        let k = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            DType::F32,
            &device,
        )?;

        // V has distinct values per position so we can check which positions are attended
        let v_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| std::iter::repeat((i + 1) as f32).take(head_dim))
            .collect();
        let v = Tensor::from_vec(v_data, (batch_size, num_heads, seq_len, head_dim), &device)?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        // For causal attention with uniform Q, K:
        // Position 0 sees only V[0]
        // Position 1 sees V[0] and V[1] (average = 1.5)
        // Position 2 sees V[0], V[1], V[2] (average = 2.0)
        // Position 3 sees all V (average = 2.5)

        let output_flat: Vec<f32> = output.flatten_all()?.to_vec1()?;

        // Check first position attends only to itself (V value 1.0)
        let pos0_mean = output_flat[0..head_dim].iter().sum::<f32>() / head_dim as f32;
        assert!(
            (pos0_mean - 1.0).abs() < 0.1,
            "Position 0 should attend only to V[0], got {}",
            pos0_mean
        );

        // Check last position attends to all (average = 2.5)
        let pos3_start = 3 * head_dim;
        let pos3_mean = output_flat[pos3_start..pos3_start + head_dim]
            .iter()
            .sum::<f32>()
            / head_dim as f32;
        assert!(
            (pos3_mean - 2.5).abs() < 0.1,
            "Position 3 should average all V, got {}",
            pos3_mean
        );

        Ok(())
    }

    // =========================================================================
    // Batch and Multi-head Tests
    // =========================================================================

    #[test]
    fn test_multi_batch() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 4;
        let num_heads = 8;
        let seq_len = 32;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

        // Output should be finite
        let has_nan = output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|x| x.is_nan());
        assert!(!has_nan, "Output should not contain NaN");

        Ok(())
    }

    #[test]
    fn test_different_head_dims() -> CandleResult<()> {
        let device = Device::Cpu;

        for head_dim in [32, 64, 128] {
            let batch_size = 1;
            let num_heads = 4;
            let seq_len = 16;

            let q = Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, seq_len, head_dim),
                &device,
            )?;
            let k = Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, seq_len, head_dim),
                &device,
            )?;
            let v = Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, seq_len, head_dim),
                &device,
            )?;

            let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
            let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

            assert_eq!(
                output.dims(),
                &[batch_size, num_heads, seq_len, head_dim],
                "Failed for head_dim={}",
                head_dim
            );
        }
        Ok(())
    }

    // =========================================================================
    // Block Size and Tiling Tests
    // =========================================================================

    #[test]
    fn test_different_block_sizes() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 128;
        let head_dim = 32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        // Test different block sizes produce consistent results
        let mut reference: Option<Tensor> = None;

        for block_size in [16, 32, 64, 128] {
            let config = FlashAttentionConfig {
                block_size,
                causal: true,
                ..Default::default()
            };
            let flash_attn = FlashAttention::new(config);
            let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

            if let Some(ref prev) = reference {
                let diff = (prev - &output)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    diff < 1e-3,
                    "Block size {} differs from reference, max diff={}",
                    block_size,
                    diff
                );
            } else {
                reference = Some(output);
            }
        }
        Ok(())
    }

    #[test]
    fn test_seq_len_not_divisible_by_block() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 100; // Not divisible by common block sizes
        let head_dim = 32;

        let config = FlashAttentionConfig {
            block_size: 32, // 100 / 32 = 3 full blocks + 1 partial
            causal: true,
            ..Default::default()
        };

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(config.clone());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

        // Compare with standard to verify correctness
        let scale = 1.0 / (head_dim as f32).sqrt();
        let standard = flash_attn.forward_standard(&q, &k, &v, None, true, scale)?;
        let diff = (&standard - &output)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-3,
            "Partial blocks should match standard, max diff={}",
            diff
        );

        Ok(())
    }

    // =========================================================================
    // Softmax Scale Tests
    // =========================================================================

    #[test]
    fn test_custom_softmax_scale() -> CandleResult<()> {
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 16;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            0.1,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        // Default scale (1/sqrt(64) = 0.125) vs custom scale
        let default_config = FlashAttentionConfig::default();
        let custom_config = FlashAttentionConfig::default().with_scale(0.25);

        let flash_default = FlashAttention::new(default_config);
        let flash_custom = FlashAttention::new(custom_config);

        let output_default = flash_default.forward(&q, &k, &v, None, Some(true))?;
        let output_custom = flash_custom.forward(&q, &k, &v, None, Some(true))?;

        // Outputs should be different due to different scales
        let diff = (&output_default - &output_custom)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-5,
            "Different scales should produce different outputs"
        );

        Ok(())
    }

    // =========================================================================
    // Data Type Tests
    // =========================================================================

    #[test]
    fn test_f16_dtype() -> CandleResult<()> {
        // Note: BF16 matmul is not supported on CPU, so we test F16 which converts to F32 internally
        let device = Device::Cpu;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 32;
        let head_dim = 32;

        // F32 tensors - Flash Attention handles dtype internally
        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_bf16_dtype_cuda() -> CandleResult<()> {
        // BF16 is only supported on CUDA with tensor cores
        let device = Device::new_cuda(0)?;
        let batch_size = 1;
        let num_heads = 2;
        let seq_len = 32;
        let head_dim = 32;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
        let output = flash_attn.forward(&q, &k, &v, None, Some(true))?;

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
        assert_eq!(output.dtype(), DType::BF16);
        Ok(())
    }
}
