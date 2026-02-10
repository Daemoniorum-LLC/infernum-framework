//! CUDA-accelerated INT8 quantized KV cache with fused attention.
//!
//! This cache uses CUDA kernels for:
//! - INT8 quantization with per-token BF16 scales
//! - Fused Q@K^T attention with on-the-fly dequantization
//! - Fused attention@V with on-the-fly dequantization
//!
//! Unlike standard/quantized caches, this cache computes attention internally
//! using fused kernels, avoiding the need to materialize full K/V tensors.

use super::KvCache;
use crate::kv_cache_quant_cuda::cuda::CudaQuantizedKvCache as InnerCache;
use candle_core::{Result as CandleResult, Tensor};

/// CUDA-accelerated INT8 quantized KV cache.
///
/// Provides ~2x memory reduction with fused attention kernels that
/// avoid materializing full K/V tensors during attention computation.
///
/// # Requirements
/// - CUDA-capable GPU
/// - The `cuda` feature must be enabled
///
/// # Example
/// ```ignore
/// let mut cache = CudaQuantizedCache::new(8, 64, 0)?;
/// cache.append(&k, &v)?;
///
/// // Use fused attention instead of get_kv()
/// if cache.supports_fused_attention() {
///     let output = cache.forward_attention(&q, num_heads, scale)?;
/// }
/// ```
pub struct CudaQuantizedCache {
    /// Inner cache implementation
    inner: InnerCache,
    /// Number of KV heads (for reference)
    num_kv_heads: usize,
    /// Head dimension (for reference)
    head_dim: usize,
}

impl CudaQuantizedCache {
    /// Create a new CUDA quantized cache.
    ///
    /// # Arguments
    /// * `num_kv_heads` - Number of key/value heads (may differ from Q heads for GQA)
    /// * `head_dim` - Dimension of each attention head
    /// * `device_id` - CUDA device ID (typically 0)
    ///
    /// # Errors
    /// Returns an error if CUDA initialization fails or kernels cannot be compiled.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        device_id: usize,
    ) -> Result<Self, crate::kv_cache_quant_cuda::cuda::Int8AttentionError> {
        let inner = InnerCache::new(num_kv_heads, head_dim, device_id)?;
        Ok(Self {
            inner,
            num_kv_heads,
            head_dim,
        })
    }

    /// Get the number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl KvCache for CudaQuantizedCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()> {
        self.inner
            .append(k, v)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA cache append failed: {e}")))
    }

    fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    fn clear(&mut self) {
        self.inner.clear()
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn get_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        // CUDA quantized cache doesn't support getting raw K/V
        // It uses fused attention instead
        Err(candle_core::Error::Msg(
            "CudaQuantizedCache uses fused attention. Use forward_attention() instead of get_kv()."
                .to_string(),
        ))
    }

    fn supports_fused_attention(&self) -> bool {
        true
    }

    fn forward_attention(
        &mut self,
        q: &Tensor,
        num_heads: usize,
        scale: f32,
    ) -> CandleResult<Tensor> {
        self.inner
            .forward_attention(q, num_heads, scale)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA attention failed: {e}")))
    }
}

impl std::fmt::Debug for CudaQuantizedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaQuantizedCache")
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("seq_len", &self.inner.seq_len())
            .field("memory_bytes", &self.inner.memory_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cuda_cache_creation() {
        // Skip if CUDA not available
        if !candle_core::utils::cuda_is_available() {
            println!("CUDA not available, skipping test");
            return;
        }

        let cache = CudaQuantizedCache::new(2, 64, 0);
        assert!(
            cache.is_ok(),
            "Failed to create CUDA cache: {:?}",
            cache.err()
        );

        let cache = cache.unwrap();
        assert_eq!(cache.num_kv_heads(), 2);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.supports_fused_attention());
    }

    #[test]
    fn test_cuda_cache_append() -> CandleResult<()> {
        if !candle_core::utils::cuda_is_available() {
            println!("CUDA not available, skipping test");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let mut cache = CudaQuantizedCache::new(2, 64, 0)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 64), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 64), &device)?.to_dtype(DType::BF16)?;

        cache.append(&k, &v)?;
        assert_eq!(cache.seq_len(), 5);

        // Append more
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 64), &device)?.to_dtype(DType::BF16)?;
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 64), &device)?.to_dtype(DType::BF16)?;

        cache.append(&k2, &v2)?;
        assert_eq!(cache.seq_len(), 8);

        Ok(())
    }

    #[test]
    fn test_cuda_cache_fused_attention() -> CandleResult<()> {
        if !candle_core::utils::cuda_is_available() {
            println!("CUDA not available, skipping test");
            return Ok(());
        }

        let device = Device::new_cuda(0)?;
        let num_kv_heads = 2;
        let num_heads = 8; // GQA: 4 Q heads per KV head
        let head_dim = 64;

        let mut cache = CudaQuantizedCache::new(num_kv_heads, head_dim, 0)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        // Add K/V
        let k = Tensor::randn(0.0f32, 1.0, (1, num_kv_heads, 5, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, num_kv_heads, 5, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        cache.append(&k, &v)?;

        // Query
        let q = Tensor::randn(0.0f32, 1.0, (1, num_heads, 1, head_dim), &device)?
            .to_dtype(DType::BF16)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let output = cache.forward_attention(&q, num_heads, scale)?;

        assert_eq!(output.dims(), &[1, num_heads, 1, head_dim]);

        Ok(())
    }
}
