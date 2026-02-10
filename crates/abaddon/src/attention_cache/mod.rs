//! Model-agnostic attention cache implementations.
//!
//! This module provides a unified interface for different KV cache strategies
//! that can be used with any transformer model (Llama, Qwen2, Mistral, etc.).
//!
//! ## Cache Types
//!
//! - [`StandardCache`]: Full-precision BF16/FP16 cache
//! - [`QuantizedCache`]: INT8 quantized cache with CPU dequantization
//! - `CudaQuantizedCache`: INT8 cache with fused CUDA attention kernels (requires `cuda` feature)
//!
//! ## Usage
//!
//! All caches implement the [`KvCache`] trait, allowing models to be cache-agnostic:
//!
//! ```ignore
//! use abaddon::attention_cache::{KvCache, CacheType, CacheConfig, attention_with_cache};
//!
//! // Create a cache (works with any model)
//! let config = CacheConfig::new(8, 64, device, DType::BF16);
//! let cache_type = CacheType::CudaQuantized { device_id: 0 };
//! let mut cache = cache_type.create(&config)?;
//!
//! // Use with generic attention function
//! let output = attention_with_cache(&q, &k, &v, cache.as_mut(), num_heads, num_kv_heads, None)?;
//! ```
//!
//! ## Adding Cache Support to a New Model
//!
//! To add cache support to a new model, use the generic attention function:
//!
//! ```ignore
//! // In your model's attention layer:
//! let output = attention_with_cache(
//!     &q, &k, &v,
//!     &mut self.cache,
//!     self.num_heads,
//!     self.num_kv_heads,
//!     causal_mask.as_ref(),
//! )?;
//! ```

mod attention;
#[cfg(feature = "cuda")]
mod cuda_quantized;
mod quantized;
mod standard;

pub use attention::{
    attention_with_cache, attention_with_cache_mode, create_causal_mask, repeat_kv, AttentionConfig,
};
#[cfg(feature = "cuda")]
pub use cuda_quantized::CudaQuantizedCache;
pub use quantized::{QuantizationGranularity, QuantizedCache};
pub use standard::StandardCache;

// Re-export attention variants for explicit mode selection
pub use crate::flash_attention::AttentionVariant;

use candle_core::{DType, Device, Result as CandleResult, Tensor};

/// Trait for KV cache implementations.
///
/// This trait provides a unified interface for different cache strategies,
/// allowing models to be agnostic about how K/V states are stored and retrieved.
///
/// # Cache Modes
///
/// Caches operate in one of two modes:
///
/// 1. **Standard mode** (`supports_fused_attention() == false`):
///    - Cache stores K/V tensors
///    - Model retrieves them via `get_kv()` and computes attention itself
///    - Used by: `StandardCache`, `QuantizedCache`
///
/// 2. **Fused mode** (`supports_fused_attention() == true`):
///    - Cache computes attention internally via `forward_attention()`
///    - More efficient as it avoids materializing full K/V tensors
///    - Used by: `CudaQuantizedCache`
pub trait KvCache: Send {
    /// Append new K/V tensors to the cache.
    ///
    /// # Arguments
    /// * `k` - Key tensor, shape: `[batch, num_kv_heads, seq_len, head_dim]`
    /// * `v` - Value tensor, shape: `[batch, num_kv_heads, seq_len, head_dim]`
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()>;

    /// Get the current sequence length in the cache.
    fn seq_len(&self) -> usize;

    /// Clear all cached K/V states.
    fn clear(&mut self);

    /// Get memory usage in bytes.
    fn memory_bytes(&self) -> usize;

    /// Get the full K/V tensors for standard attention computation.
    ///
    /// Returns `None` if the cache is empty.
    /// For fused-attention caches, this may trigger dequantization.
    fn get_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>>;

    /// Whether this cache supports fused attention (computing attention internally).
    ///
    /// If true, use `forward_attention()` instead of `get_kv()` + manual attention.
    fn supports_fused_attention(&self) -> bool {
        false
    }

    /// Compute attention using fused kernels (for caches that support it).
    ///
    /// # Arguments
    /// * `q` - Query tensor, shape: `[batch, num_heads, q_len, head_dim]`
    /// * `num_heads` - Number of attention heads (for GQA expansion)
    /// * `scale` - Attention scale factor (typically `1/sqrt(head_dim)`)
    ///
    /// # Returns
    /// Attention output, shape: `[batch, num_heads, q_len, head_dim]`
    ///
    /// # Panics
    /// Default implementation panics. Only call if `supports_fused_attention()` is true.
    fn forward_attention(
        &mut self,
        _q: &Tensor,
        _num_heads: usize,
        _scale: f32,
    ) -> CandleResult<Tensor> {
        panic!("forward_attention called on cache that doesn't support fused attention")
    }
}

/// Configuration for creating KV caches.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of KV heads (for GQA, this may differ from num_attention_heads).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Device to store cache on.
    pub device: Device,
    /// Data type for cache storage (for standard cache).
    pub dtype: DType,
}

impl KvCacheConfig {
    /// Create a new cache configuration.
    pub fn new(num_kv_heads: usize, head_dim: usize, device: Device, dtype: DType) -> Self {
        Self {
            num_kv_heads,
            head_dim,
            device,
            dtype,
        }
    }
}

/// Factory for creating KV caches.
///
/// This provides a convenient way to create caches without knowing the concrete type.
#[derive(Debug, Clone)]
pub enum CacheType {
    /// Standard full-precision cache.
    Standard,
    /// INT8 quantized cache with per-token scales.
    Quantized(QuantizationGranularity),
    /// CUDA-accelerated INT8 cache with fused attention.
    #[cfg(feature = "cuda")]
    CudaQuantized {
        /// CUDA device ID.
        device_id: usize,
    },
}

impl Default for CacheType {
    fn default() -> Self {
        Self::Standard
    }
}

impl CacheType {
    /// Create a cache instance from this configuration.
    pub fn create(&self, config: &KvCacheConfig) -> CandleResult<Box<dyn KvCache>> {
        match self {
            Self::Standard => Ok(Box::new(StandardCache::new())),
            Self::Quantized(granularity) => Ok(Box::new(QuantizedCache::new(
                config.num_kv_heads,
                config.head_dim,
                *granularity,
            ))),
            #[cfg(feature = "cuda")]
            Self::CudaQuantized { device_id } => {
                let cache =
                    CudaQuantizedCache::new(config.num_kv_heads, config.head_dim, *device_id)
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("CUDA cache init failed: {e}"))
                        })?;
                Ok(Box::new(cache))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config() {
        let config = KvCacheConfig::new(2, 64, Device::Cpu, DType::BF16);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_cache_type_default() {
        let cache_type = CacheType::default();
        assert!(matches!(cache_type, CacheType::Standard));
    }
}
