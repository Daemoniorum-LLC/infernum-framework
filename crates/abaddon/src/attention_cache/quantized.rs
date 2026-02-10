//! INT8 quantized KV cache with per-token scales.
//!
//! This cache quantizes K/V tensors to INT8 (stored as U8 with offset 128),
//! providing ~2x memory reduction compared to BF16/FP16.
//!
//! ## Quantization Strategy
//! - Symmetric per-token quantization
//! - Scale = max(|x|) / 127
//! - Quantized = round(x / scale) + 128 (offset to U8 range)
//! - Dequantized = (quantized - 128) * scale

use super::KvCache;
use candle_core::{DType, Result as CandleResult, Tensor, D};

/// Quantization granularity for KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationGranularity {
    /// One scale per token (per head, per sequence position).
    /// Best accuracy, moderate overhead.
    #[default]
    PerToken,
    /// One scale per head (shared across sequence positions).
    /// Lower accuracy, minimal overhead.
    PerHead,
    /// One scale per tensor.
    /// Lowest accuracy, no overhead.
    PerTensor,
}

/// INT8 quantized KV cache.
///
/// Stores K/V tensors as INT8 with per-token (or other granularity) scales.
/// Provides ~2x memory reduction with minimal accuracy loss.
///
/// # Example
/// ```ignore
/// let mut cache = QuantizedCache::new(8, 64, QuantizationGranularity::PerToken);
/// cache.append(&k, &v)?;
/// let (k_full, v_full) = cache.get_kv()?.unwrap();
/// ```
#[derive(Debug)]
pub struct QuantizedCache {
    /// Quantized keys as U8: [batch, num_kv_heads, seq_len, head_dim]
    k_quantized: Option<Tensor>,
    /// Quantized values as U8: [batch, num_kv_heads, seq_len, head_dim]
    v_quantized: Option<Tensor>,
    /// Key scales (shape depends on granularity)
    k_scales: Option<Tensor>,
    /// Value scales (shape depends on granularity)
    v_scales: Option<Tensor>,
    /// Number of KV heads (used when GPU attention is wired up)
    #[allow(dead_code)]
    num_kv_heads: usize,
    /// Head dimension (used when GPU attention is wired up)
    #[allow(dead_code)]
    head_dim: usize,
    /// Quantization granularity
    granularity: QuantizationGranularity,
    /// Output dtype for dequantization (inferred from input)
    dtype: Option<DType>,
}

impl QuantizedCache {
    /// Create a new empty quantized cache.
    pub fn new(num_kv_heads: usize, head_dim: usize, granularity: QuantizationGranularity) -> Self {
        Self {
            k_quantized: None,
            v_quantized: None,
            k_scales: None,
            v_scales: None,
            num_kv_heads,
            head_dim,
            granularity,
            dtype: None,
        }
    }

    /// Quantize a tensor to U8 with scales.
    /// Input shape: [batch, num_heads, seq_len, head_dim]
    fn quantize(&self, tensor: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let dtype = tensor.dtype();
        let device = tensor.device();

        // Compute scales based on granularity
        let abs_tensor = tensor.abs()?;

        let max_vals = match self.granularity {
            QuantizationGranularity::PerToken => {
                // [batch, heads, seq, head_dim] -> [batch, heads, seq, 1]
                abs_tensor.max_keepdim(D::Minus1)?
            },
            QuantizationGranularity::PerHead => {
                // [batch, heads, seq, head_dim] -> [batch, heads, 1, 1]
                abs_tensor.max_keepdim(D::Minus1)?.max_keepdim(D::Minus2)?
            },
            QuantizationGranularity::PerTensor => {
                // [batch, heads, seq, head_dim] -> [1, 1, 1, 1]
                let max_val = abs_tensor.max_all()?;
                max_val.reshape((1, 1, 1, 1))?
            },
        };

        // Avoid division by zero
        let eps = Tensor::new(&[1e-8f32], device)?
            .broadcast_as(max_vals.shape())?
            .to_dtype(dtype)?;
        let max_vals = max_vals.maximum(&eps)?;

        // Scale = max / 127
        let scale = (&max_vals / 127.0)?;

        // Quantize: round(x / scale) + 128
        let scaled = tensor.broadcast_div(&scale)?;
        let offset = (scaled + 128.0)?;
        let clamped = offset.clamp(0.0, 255.0)?;
        let quantized = clamped.round()?.to_dtype(DType::U8)?;

        Ok((quantized, scale))
    }

    /// Dequantize U8 tensor back to original dtype.
    fn dequantize(
        &self,
        quantized: &Tensor,
        scales: &Tensor,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        // (quantized - 128) * scale
        let float_vals = quantized.to_dtype(dtype)?;
        let unoffset = (float_vals - 128.0)?;
        unoffset.broadcast_mul(scales)
    }
}

impl KvCache for QuantizedCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()> {
        // Remember dtype from first input
        if self.dtype.is_none() {
            self.dtype = Some(k.dtype());
        }

        // Quantize new K and V
        let (k_quant, k_scale) = self.quantize(k)?;
        let (v_quant, v_scale) = self.quantize(v)?;

        // Concatenate with existing cache
        let (k_quantized, k_scales) = match (&self.k_quantized, &self.k_scales) {
            (Some(prev_k), Some(prev_scale)) => {
                let k = Tensor::cat(&[prev_k, &k_quant], 2)?;
                let scale = Tensor::cat(&[prev_scale, &k_scale], 2)?;
                (k, scale)
            },
            _ => (k_quant, k_scale),
        };

        let (v_quantized, v_scales) = match (&self.v_quantized, &self.v_scales) {
            (Some(prev_v), Some(prev_scale)) => {
                let v = Tensor::cat(&[prev_v, &v_quant], 2)?;
                let scale = Tensor::cat(&[prev_scale, &v_scale], 2)?;
                (v, scale)
            },
            _ => (v_quant, v_scale),
        };

        self.k_quantized = Some(k_quantized);
        self.k_scales = Some(k_scales);
        self.v_quantized = Some(v_quantized);
        self.v_scales = Some(v_scales);

        Ok(())
    }

    fn seq_len(&self) -> usize {
        self.k_quantized.as_ref().map(|t| t.dims()[2]).unwrap_or(0)
    }

    fn clear(&mut self) {
        self.k_quantized = None;
        self.v_quantized = None;
        self.k_scales = None;
        self.v_scales = None;
        self.dtype = None;
    }

    fn memory_bytes(&self) -> usize {
        let k_mem = self
            .k_quantized
            .as_ref()
            .map(|t| t.elem_count())
            .unwrap_or(0);
        let v_mem = self
            .v_quantized
            .as_ref()
            .map(|t| t.elem_count())
            .unwrap_or(0);
        let k_scale_mem = self
            .k_scales
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0);
        let v_scale_mem = self
            .v_scales
            .as_ref()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .unwrap_or(0);

        // INT8 = 1 byte per element
        k_mem + v_mem + k_scale_mem + v_scale_mem
    }

    fn get_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        let dtype = self.dtype.unwrap_or(DType::BF16);

        match (
            &self.k_quantized,
            &self.k_scales,
            &self.v_quantized,
            &self.v_scales,
        ) {
            (Some(k_q), Some(k_s), Some(v_q), Some(v_s)) => {
                let k = self.dequantize(k_q, k_s, dtype)?;
                let v = self.dequantize(v_q, v_s, dtype)?;
                Ok(Some((k, v)))
            },
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
    use candle_core::Device;

    #[test]
    fn test_quantized_cache_empty() {
        let cache = QuantizedCache::new(2, 64, QuantizationGranularity::PerToken);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.memory_bytes(), 0);
        assert!(cache.get_kv().unwrap().is_none());
    }

    #[test]
    fn test_quantized_cache_append() -> CandleResult<()> {
        let mut cache = QuantizedCache::new(4, 64, QuantizationGranularity::PerToken);
        let device = Device::Cpu;

        // Create test tensors
        let k1 = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        let v1 = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        cache.append(&k1, &v1)?;

        assert_eq!(cache.seq_len(), 10);

        // Append more
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        cache.append(&k2, &v2)?;

        assert_eq!(cache.seq_len(), 15);

        // Get dequantized
        let (k, v) = cache.get_kv()?.unwrap();
        assert_eq!(k.dims(), &[1, 4, 15, 64]);
        assert_eq!(v.dims(), &[1, 4, 15, 64]);

        Ok(())
    }

    #[test]
    fn test_quantization_accuracy() -> CandleResult<()> {
        let cache = QuantizedCache::new(2, 8, QuantizationGranularity::PerToken);
        let device = Device::Cpu;

        // Create tensor with known values
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let tensor = Tensor::from_vec(data, (1, 2, 8, 8), &device)?;

        // Quantize and dequantize
        let (quantized, scales) = cache.quantize(&tensor)?;
        let recovered = cache.dequantize(&quantized, &scales, DType::F32)?;

        // Check accuracy
        let orig: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let recv: Vec<f32> = recovered.flatten_all()?.to_vec1()?;

        for (o, r) in orig.iter().zip(recv.iter()) {
            let error = (o - r).abs();
            assert!(
                error < 0.02,
                "Error too large: {} vs {} (error: {})",
                o,
                r,
                error
            );
        }

        Ok(())
    }

    #[test]
    fn test_memory_savings() -> CandleResult<()> {
        let mut cache = QuantizedCache::new(8, 128, QuantizationGranularity::PerToken);
        let device = Device::Cpu;

        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 100, 128), &device)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 100, 128), &device)?.to_dtype(DType::BF16)?;
        cache.append(&k, &v)?;

        let quant_mem = cache.memory_bytes();
        let full_mem = 2 * 1 * 8 * 100 * 128 * 2; // K + V, BF16

        let savings = full_mem as f32 / quant_mem as f32;
        println!(
            "Memory: quantized={}, full={}, savings={:.2}x",
            quant_mem, full_mem, savings
        );

        assert!(savings > 1.5, "Expected at least 1.5x savings");

        Ok(())
    }
}
