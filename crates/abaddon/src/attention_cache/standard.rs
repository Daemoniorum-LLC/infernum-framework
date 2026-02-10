//! Standard full-precision KV cache.
//!
//! This cache stores K/V tensors in their original precision (BF16/FP16/FP32).
//! It's the simplest and most accurate cache, but uses the most memory.

use super::KvCache;
use candle_core::{Result as CandleResult, Tensor};

/// Standard full-precision KV cache.
///
/// Stores K/V tensors by concatenating new values to existing cache.
/// No quantization or compression is applied.
///
/// # Memory Usage
/// For a model with:
/// - 32 layers, 8 KV heads, 128 head_dim, BF16 precision
/// - 4096 sequence length
///
/// Memory = 32 * 2 * 8 * 4096 * 128 * 2 bytes = 512 MB per cache
#[derive(Debug, Default)]
pub struct StandardCache {
    /// Cached K tensor: [batch, num_kv_heads, seq_len, head_dim]
    k: Option<Tensor>,
    /// Cached V tensor: [batch, num_kv_heads, seq_len, head_dim]
    v: Option<Tensor>,
}

impl StandardCache {
    /// Create a new empty standard cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Create a cache pre-populated with K/V tensors.
    pub fn with_kv(k: Tensor, v: Tensor) -> Self {
        Self {
            k: Some(k),
            v: Some(v),
        }
    }
}

impl KvCache for StandardCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()> {
        let (new_k, new_v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                // Concatenate along sequence dimension (dim 2)
                let k_cat = Tensor::cat(&[prev_k, k], 2)?;
                let v_cat = Tensor::cat(&[prev_v, v], 2)?;
                (k_cat, v_cat)
            },
            _ => (k.clone(), v.clone()),
        };

        self.k = Some(new_k);
        self.v = Some(new_v);
        Ok(())
    }

    fn seq_len(&self) -> usize {
        self.k
            .as_ref()
            .map(|k| k.dims()[2]) // [batch, heads, seq_len, head_dim]
            .unwrap_or(0)
    }

    fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    fn memory_bytes(&self) -> usize {
        let k_bytes = self
            .k
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0);
        let v_bytes = self
            .v
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0);
        k_bytes + v_bytes
    }

    fn get_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Ok(Some((k.clone(), v.clone()))),
            _ => Ok(None),
        }
    }

    fn supports_fused_attention(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_standard_cache_empty() {
        let cache = StandardCache::new();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.memory_bytes(), 0);
        assert!(cache.get_kv().unwrap().is_none());
    }

    #[test]
    fn test_standard_cache_append() -> CandleResult<()> {
        let mut cache = StandardCache::new();
        let device = Device::Cpu;

        // First append: [1, 2, 3, 64]
        let k1 = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        cache.append(&k1, &v1)?;

        assert_eq!(cache.seq_len(), 3);

        // Second append: [1, 2, 2, 64]
        let k2 = Tensor::zeros((1, 2, 2, 64), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 2, 2, 64), DType::F32, &device)?;
        cache.append(&k2, &v2)?;

        assert_eq!(cache.seq_len(), 5);

        // Verify dimensions
        let (k, v) = cache.get_kv()?.unwrap();
        assert_eq!(k.dims(), &[1, 2, 5, 64]);
        assert_eq!(v.dims(), &[1, 2, 5, 64]);

        Ok(())
    }

    #[test]
    fn test_standard_cache_clear() -> CandleResult<()> {
        let mut cache = StandardCache::new();
        let device = Device::Cpu;

        let k = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        cache.append(&k, &v)?;

        assert_eq!(cache.seq_len(), 3);

        cache.clear();

        assert_eq!(cache.seq_len(), 0);
        assert!(cache.get_kv()?.is_none());

        Ok(())
    }

    #[test]
    fn test_standard_cache_memory() -> CandleResult<()> {
        let mut cache = StandardCache::new();
        let device = Device::Cpu;

        // [1, 2, 3, 64] * 2 tensors * 4 bytes (F32) = 1536 * 2 = 3072
        let k = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        let v = Tensor::zeros((1, 2, 3, 64), DType::F32, &device)?;
        cache.append(&k, &v)?;

        assert_eq!(cache.memory_bytes(), 1 * 2 * 3 * 64 * 4 * 2);

        Ok(())
    }
}
