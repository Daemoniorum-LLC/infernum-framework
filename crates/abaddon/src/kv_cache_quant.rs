//! INT8 KV Cache Quantization for memory-efficient inference.
//!
//! This module provides symmetric per-token INT8 quantization for KV cache,
//! reducing memory usage by 2x compared to BF16/FP16 storage.
//!
//! ## Benefits
//! - 2x memory reduction for KV cache
//! - Longer context windows with same VRAM
//! - Reduced memory bandwidth (faster attention)
//!
//! ## Quantization Strategy
//! - Symmetric per-token quantization
//! - Scale = max(|x|) / 127
//! - Quantized = round(x / scale)
//! - Dequantized = quantized * scale
//!
//! ## Usage
//! ```ignore
//! let mut cache = QuantizedKvCache::new(num_kv_heads, head_dim, device)?;
//! cache.append(&k, &v)?;
//! let (k_full, v_full) = cache.get_dequantized()?;
//! ```

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};

/// Quantized KV cache using INT8 storage with per-token scales.
#[derive(Debug)]
pub struct QuantizedKvCache {
    /// Quantized keys: (batch, num_kv_heads, seq_len, head_dim) as I8
    k_quantized: Option<Tensor>,
    /// Quantized values: (batch, num_kv_heads, seq_len, head_dim) as I8
    v_quantized: Option<Tensor>,
    /// Key scales per token: (batch, num_kv_heads, seq_len, 1)
    k_scales: Option<Tensor>,
    /// Value scales per token: (batch, num_kv_heads, seq_len, 1)
    v_scales: Option<Tensor>,
    /// Number of KV heads
    #[allow(dead_code)]
    num_kv_heads: usize,
    /// Head dimension
    #[allow(dead_code)]
    head_dim: usize,
    /// Device
    #[allow(dead_code)]
    device: Device,
    /// Output dtype for dequantization
    dtype: DType,
}

impl QuantizedKvCache {
    /// Create a new empty quantized KV cache.
    pub fn new(num_kv_heads: usize, head_dim: usize, device: &Device, dtype: DType) -> Self {
        Self {
            k_quantized: None,
            v_quantized: None,
            k_scales: None,
            v_scales: None,
            num_kv_heads,
            head_dim,
            device: device.clone(),
            dtype,
        }
    }

    /// Quantize a tensor to UINT8 with per-token scales.
    /// Uses symmetric quantization with offset 128 (U8 range 0-255 maps to -128..127).
    /// Input shape: (batch, num_heads, seq_len, head_dim)
    /// Returns: (quantized U8 tensor, scales tensor)
    fn quantize(tensor: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // Compute per-token max absolute value
        // Shape: (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, 1)
        let abs_tensor = tensor.abs()?;
        let max_vals = abs_tensor.max_keepdim(D::Minus1)?;

        // Avoid division by zero
        let eps = Tensor::new(&[1e-8f32], tensor.device())?
            .broadcast_as(max_vals.shape())?
            .to_dtype(tensor.dtype())?;
        let max_vals = max_vals.maximum(&eps)?;

        // Scale = max / 127 (symmetric quantization)
        let scale = (&max_vals / 127.0)?;

        // Quantize: round(x / scale) + 128 (offset to U8 range)
        let scaled = tensor.broadcast_div(&scale)?;
        let offset = (scaled + 128.0)?;

        // Clamp to U8 range and convert
        let clamped = offset.clamp(0.0, 255.0)?;
        let quantized = clamped.round()?.to_dtype(DType::U8)?;

        Ok((quantized, scale))
    }

    /// Dequantize UINT8 tensor back to original dtype.
    fn dequantize(quantized: &Tensor, scales: &Tensor, dtype: DType) -> CandleResult<Tensor> {
        // Convert U8 to float, subtract offset, multiply by scales
        let float_vals = quantized.to_dtype(dtype)?;
        let unoffset = (float_vals - 128.0)?;
        unoffset.broadcast_mul(scales)
    }

    /// Append new K and V tensors to the cache.
    /// Input shape: (batch, num_kv_heads, new_seq_len, head_dim)
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()> {
        // Quantize new K and V
        let (k_quant, k_scale) = Self::quantize(k)?;
        let (v_quant, v_scale) = Self::quantize(v)?;

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

    /// Get dequantized K and V tensors for attention computation.
    /// Returns: (K, V) both with shape (batch, num_kv_heads, full_seq_len, head_dim)
    pub fn get_dequantized(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        match (
            &self.k_quantized,
            &self.k_scales,
            &self.v_quantized,
            &self.v_scales,
        ) {
            (Some(k_q), Some(k_s), Some(v_q), Some(v_s)) => {
                let k = Self::dequantize(k_q, k_s, self.dtype)?;
                let v = Self::dequantize(v_q, v_s, self.dtype)?;
                Ok(Some((k, v)))
            },
            _ => Ok(None),
        }
    }

    /// Get the current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        self.k_quantized.as_ref().map(|t| t.dims()[2]).unwrap_or(0)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.k_quantized = None;
        self.v_quantized = None;
        self.k_scales = None;
        self.v_scales = None;
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
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
            .map(|t| t.elem_count() * 2) // BF16 = 2 bytes
            .unwrap_or(0);
        let v_scale_mem = self
            .v_scales
            .as_ref()
            .map(|t| t.elem_count() * 2)
            .unwrap_or(0);

        // INT8 = 1 byte per element
        k_mem + v_mem + k_scale_mem + v_scale_mem
    }

    /// Calculate memory savings compared to full precision cache.
    pub fn memory_savings_ratio(&self) -> f32 {
        // Full precision: 2 bytes per element (BF16)
        // Quantized: 1 byte + scale overhead
        // Scale overhead: 1 scale per token per head (negligible for head_dim >> 1)
        let full_precision_bytes = self
            .k_quantized
            .as_ref()
            .map(|t| t.elem_count() * 2 * 2) // K + V, 2 bytes each
            .unwrap_or(0);

        if full_precision_bytes == 0 {
            return 1.0;
        }

        full_precision_bytes as f32 / self.memory_bytes() as f32
    }
}

/// Configuration for KV cache quantization.
#[derive(Debug, Clone, Copy)]
pub struct KvCacheQuantConfig {
    /// Enable INT8 quantization for KV cache.
    pub enabled: bool,
    /// Minimum sequence length before quantization kicks in.
    /// Short sequences may not benefit from quantization overhead.
    pub min_seq_len: usize,
}

impl Default for KvCacheQuantConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_seq_len: 64, // Only quantize for sequences >= 64 tokens
        }
    }
}

impl KvCacheQuantConfig {
    /// Create a new config with quantization enabled.
    pub fn enabled() -> Self {
        Self::default()
    }

    /// Create a config with quantization disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            min_seq_len: 0,
        }
    }

    /// Set minimum sequence length for quantization.
    pub fn with_min_seq_len(mut self, len: usize) -> Self {
        self.min_seq_len = len;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a test tensor with known values
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let tensor = Tensor::from_vec(data.clone(), (1, 2, 8, 8), &device)?;

        // Quantize and dequantize
        let (quantized, scales) = QuantizedKvCache::quantize(&tensor)?;
        let recovered = QuantizedKvCache::dequantize(&quantized, &scales, dtype)?;

        // Check shape preserved
        assert_eq!(tensor.dims(), recovered.dims());

        // Check values are close (within quantization error)
        let orig: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let recv: Vec<f32> = recovered.flatten_all()?.to_vec1()?;

        for (o, r) in orig.iter().zip(recv.iter()) {
            let error = (o - r).abs();
            assert!(
                error < 0.02,
                "Quantization error too large: {} vs {} (error: {})",
                o,
                r,
                error
            );
        }

        Ok(())
    }

    #[test]
    fn test_cache_append() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut cache = QuantizedKvCache::new(4, 64, &device, dtype);

        // Append first batch
        let k1 = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        let v1 = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        cache.append(&k1, &v1)?;
        assert_eq!(cache.seq_len(), 10);

        // Append second batch
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        cache.append(&k2, &v2)?;
        assert_eq!(cache.seq_len(), 15);

        // Get dequantized
        let (k, v) = cache.get_dequantized()?.unwrap();
        assert_eq!(k.dims(), &[1, 4, 15, 64]);
        assert_eq!(v.dims(), &[1, 4, 15, 64]);

        Ok(())
    }

    #[test]
    fn test_memory_savings() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::BF16;

        let mut cache = QuantizedKvCache::new(8, 128, &device, dtype);

        // Simulate 1000 tokens
        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 1000, 128), &device)?.to_dtype(dtype)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 1000, 128), &device)?.to_dtype(dtype)?;
        cache.append(&k, &v)?;

        let savings = cache.memory_savings_ratio();
        println!("Memory savings ratio: {:.2}x", savings);

        // Should be close to 2x (INT8 vs BF16, minus scale overhead)
        assert!(
            savings > 1.5,
            "Expected at least 1.5x savings, got {:.2}x",
            savings
        );

        Ok(())
    }
}
