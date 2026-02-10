//! INT4 and INT8 weight quantization for model compression.
//!
//! This module provides quantization algorithms to convert FP16/FP32 weights
//! to INT4/INT8 format. The quantized weights are compatible with the
//! dequantization kernels in the `gpu_dequant` module.
//!
//! ## Quantization Formats
//!
//! ### INT4 Symmetric
//! - Range: [-8, 7] mapped to [-max_abs, max_abs]
//! - No zero point (zero_point = 0)
//! - Formula: `quantized = round(value / scale)`
//! - Dequantize: `value = quantized * scale`
//!
//! ### INT4 Asymmetric (GPTQ/AWQ compatible)
//! - Range: [0, 15] mapped to [min_val, max_val]
//! - Uses zero point for asymmetric distribution
//! - Formula: `quantized = round((value - zero_point * scale) / scale) + zero_point`
//! - Dequantize: `value = (quantized - zero_point) * scale`
//!
//! ## Block Quantization
//!
//! Weights are quantized in blocks (default: 128 values per block).
//! Each block has its own scale (and optionally zero point).
//! This preserves accuracy better than per-tensor quantization.
//!
//! ## Usage
//!
//! ```ignore
//! use abaddon::quantize::{Quantizer, QuantizeConfig, QuantizeFormat};
//!
//! let config = QuantizeConfig {
//!     format: QuantizeFormat::Int4Symmetric,
//!     block_size: 128,
//! };
//!
//! let quantizer = Quantizer::new(config);
//! let result = quantizer.quantize_tensor(&weights)?;
//! ```

use candle_core::{DType, Device, Tensor};
use thiserror::Error;

/// Default block size for quantization (matches dequantization).
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// INT4 range for symmetric quantization: [-8, 7]
const INT4_SYMMETRIC_MIN: i8 = -8;
const INT4_SYMMETRIC_MAX: i8 = 7;

/// INT4 range for asymmetric quantization: [0, 15]
const INT4_ASYMMETRIC_MIN: u8 = 0;
const INT4_ASYMMETRIC_MAX: u8 = 15;

/// Quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeFormat {
    /// INT4 symmetric quantization (simpler, faster).
    /// Range: [-8, 7], zero_point = 0
    Int4Symmetric,

    /// INT4 asymmetric quantization (GPTQ/AWQ compatible).
    /// Range: [0, 15], uses per-block zero points
    Int4Asymmetric,

    /// INT8 symmetric quantization.
    /// Range: [-128, 127], zero_point = 0
    Int8Symmetric,

    /// INT8 asymmetric quantization.
    /// Range: [-128, 127] with per-block zero points
    Int8Asymmetric,
}

impl QuantizeFormat {
    /// Returns the number of bits per value.
    pub fn bits(&self) -> usize {
        match self {
            Self::Int4Symmetric | Self::Int4Asymmetric => 4,
            Self::Int8Symmetric | Self::Int8Asymmetric => 8,
        }
    }

    /// Returns whether this format uses zero points.
    pub fn uses_zero_point(&self) -> bool {
        matches!(self, Self::Int4Asymmetric | Self::Int8Asymmetric)
    }
}

/// Configuration for quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Quantization format.
    pub format: QuantizeFormat,

    /// Number of values per quantization block.
    /// Each block has its own scale (and optionally zero point).
    pub block_size: usize,

    /// Whether to use activation-aware scaling (AWQ-style).
    /// If true, scales are computed to minimize activation error.
    pub activation_aware: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            format: QuantizeFormat::Int4Symmetric,
            block_size: DEFAULT_BLOCK_SIZE,
            activation_aware: false,
        }
    }
}

impl QuantizeConfig {
    /// Creates a config for INT4 symmetric quantization.
    pub fn int4_symmetric() -> Self {
        Self {
            format: QuantizeFormat::Int4Symmetric,
            ..Default::default()
        }
    }

    /// Creates a config for INT4 asymmetric quantization (GPTQ compatible).
    pub fn int4_asymmetric() -> Self {
        Self {
            format: QuantizeFormat::Int4Asymmetric,
            ..Default::default()
        }
    }

    /// Creates a config for INT8 symmetric quantization.
    pub fn int8_symmetric() -> Self {
        Self {
            format: QuantizeFormat::Int8Symmetric,
            ..Default::default()
        }
    }

    /// Sets the block size.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
}

/// Result of quantizing a tensor.
#[derive(Debug)]
pub struct QuantizedTensor {
    /// Quantized data.
    /// - INT4: packed 2 values per byte (low nibble first)
    /// - INT8: one value per byte
    pub data: Vec<u8>,

    /// Per-block scale factors (F16).
    pub scales: Vec<half::f16>,

    /// Per-block zero points (only for asymmetric formats).
    pub zero_points: Option<Vec<i8>>,

    /// Original tensor shape.
    pub shape: Vec<usize>,

    /// Total number of values.
    pub num_values: usize,

    /// Quantization format used.
    pub format: QuantizeFormat,

    /// Block size used.
    pub block_size: usize,

    /// Quantization statistics.
    pub stats: QuantizeStats,
}

/// Statistics from quantization.
#[derive(Debug, Clone, Default)]
pub struct QuantizeStats {
    /// Mean absolute error vs original.
    pub mean_abs_error: f32,

    /// Max absolute error vs original.
    pub max_abs_error: f32,

    /// Root mean square error.
    pub rmse: f32,

    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,

    /// Compression ratio achieved.
    pub compression_ratio: f32,
}

/// Errors from quantization operations.
#[derive(Debug, Error)]
pub enum QuantizeError {
    /// Tensor has wrong dtype.
    #[error("Unsupported dtype: expected F16 or F32, got {dtype:?}")]
    UnsupportedDtype {
        /// The unsupported data type.
        dtype: DType,
    },

    /// Tensor is empty.
    #[error("Cannot quantize empty tensor")]
    EmptyTensor,

    /// Block size is invalid.
    #[error("Invalid block size: {block_size} (must be > 0)")]
    InvalidBlockSize {
        /// The invalid block size.
        block_size: usize,
    },

    /// Candle tensor operation failed.
    #[error("Tensor operation failed: {0}")]
    TensorOp(#[from] candle_core::Error),

    /// Missing zero points for asymmetric quantization.
    #[error("Asymmetric quantization requires zero points")]
    MissingZeroPoints,
}

/// Weight quantizer.
pub struct Quantizer {
    config: QuantizeConfig,
}

impl Quantizer {
    /// Creates a new quantizer with the given configuration.
    pub fn new(config: QuantizeConfig) -> Self {
        Self { config }
    }

    /// Creates a quantizer for INT4 symmetric quantization.
    pub fn int4_symmetric() -> Self {
        Self::new(QuantizeConfig::int4_symmetric())
    }

    /// Creates a quantizer for INT4 asymmetric quantization.
    pub fn int4_asymmetric() -> Self {
        Self::new(QuantizeConfig::int4_asymmetric())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &QuantizeConfig {
        &self.config
    }

    /// Quantizes a tensor.
    ///
    /// The tensor is flattened and quantized in blocks.
    pub fn quantize_tensor(&self, tensor: &Tensor) -> Result<QuantizedTensor, QuantizeError> {
        // Validate input
        if tensor.elem_count() == 0 {
            return Err(QuantizeError::EmptyTensor);
        }
        if self.config.block_size == 0 {
            return Err(QuantizeError::InvalidBlockSize {
                block_size: self.config.block_size,
            });
        }

        // Convert to F32 for processing
        let tensor_f32 = match tensor.dtype() {
            DType::F32 => tensor.clone(),
            DType::F16 | DType::BF16 => tensor.to_dtype(DType::F32)?,
            dtype => return Err(QuantizeError::UnsupportedDtype { dtype }),
        };

        // Flatten to 1D
        let flat = tensor_f32.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;
        let num_values = values.len();

        // Quantize based on format
        let (data, scales, zero_points, stats) = match self.config.format {
            QuantizeFormat::Int4Symmetric => self.quantize_int4_symmetric(&values),
            QuantizeFormat::Int4Asymmetric => self.quantize_int4_asymmetric(&values),
            QuantizeFormat::Int8Symmetric => self.quantize_int8_symmetric(&values),
            QuantizeFormat::Int8Asymmetric => self.quantize_int8_asymmetric(&values),
        };

        Ok(QuantizedTensor {
            data,
            scales,
            zero_points,
            shape: tensor.dims().to_vec(),
            num_values,
            format: self.config.format,
            block_size: self.config.block_size,
            stats,
        })
    }

    /// Quantizes values using INT4 symmetric quantization.
    fn quantize_int4_symmetric(
        &self,
        values: &[f32],
    ) -> (Vec<u8>, Vec<half::f16>, Option<Vec<i8>>, QuantizeStats) {
        let block_size = self.config.block_size;
        let num_blocks = (values.len() + block_size - 1) / block_size;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized = Vec::with_capacity(values.len());
        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;
        let mut total_sq_error = 0.0f64;
        let mut total_sq_signal = 0.0f64;

        // Process each block
        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            // Find max absolute value in block
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            // Compute scale: max_abs / 7 (symmetric range is [-8, 7], using 7 for positive max)
            let scale = if max_abs > 1e-10 {
                max_abs / (INT4_SYMMETRIC_MAX as f32)
            } else {
                1.0 // Avoid division by zero for all-zero blocks
            };
            scales.push(half::f16::from_f32(scale));

            // Quantize block values
            for &val in block {
                // Quantize: round(val / scale), clamp to [-8, 7]
                let q = (val / scale).round();
                let q_clamped = q.clamp(INT4_SYMMETRIC_MIN as f32, INT4_SYMMETRIC_MAX as f32) as i8;
                quantized.push(q_clamped);

                // Track error
                let dequant = (q_clamped as f32) * scale;
                let error = (val - dequant).abs();
                total_error += error as f64;
                max_error = max_error.max(error);
                total_sq_error += (error * error) as f64;
                total_sq_signal += (val * val) as f64;
            }
        }

        // Pack INT4 values (2 per byte, low nibble first)
        // Convert signed [-8,7] to unsigned [0,15] for packing
        let packed = pack_int4_signed(&quantized);

        // Compute stats
        let n = values.len() as f64;
        let stats = QuantizeStats {
            mean_abs_error: (total_error / n) as f32,
            max_abs_error: max_error,
            rmse: (total_sq_error / n).sqrt() as f32,
            snr_db: if total_sq_error > 1e-10 {
                (10.0 * (total_sq_signal / total_sq_error).log10()) as f32
            } else {
                f32::INFINITY
            },
            compression_ratio: compute_compression_ratio(
                values.len(),
                packed.len(),
                scales.len(),
                None,
            ),
        };

        (packed, scales, None, stats)
    }

    /// Quantizes values using INT4 asymmetric quantization (GPTQ/AWQ compatible).
    fn quantize_int4_asymmetric(
        &self,
        values: &[f32],
    ) -> (Vec<u8>, Vec<half::f16>, Option<Vec<i8>>, QuantizeStats) {
        let block_size = self.config.block_size;
        let num_blocks = (values.len() + block_size - 1) / block_size;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);
        let mut quantized = Vec::with_capacity(values.len());
        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;
        let mut total_sq_error = 0.0f64;
        let mut total_sq_signal = 0.0f64;

        // Process each block
        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            // Find min and max in block
            let (min_val, max_val) = block
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });

            // Compute scale and zero point
            // scale = (max - min) / 15
            // zero_point = round(-min / scale)
            let range = max_val - min_val;
            let scale = if range > 1e-10 {
                range / (INT4_ASYMMETRIC_MAX as f32)
            } else {
                1.0
            };

            let zp = if scale > 1e-10 {
                (-min_val / scale).round().clamp(0.0, 15.0) as i8
            } else {
                0
            };

            scales.push(half::f16::from_f32(scale));
            zero_points.push(zp);

            // Quantize block values
            for &val in block {
                // Quantize: round(val / scale) + zero_point, clamp to [0, 15]
                let q = ((val / scale) + (zp as f32)).round();
                let q_clamped =
                    q.clamp(INT4_ASYMMETRIC_MIN as f32, INT4_ASYMMETRIC_MAX as f32) as u8;
                quantized.push(q_clamped as i8); // Store as i8 for consistency

                // Track error
                // Dequantize: (q - zp) * scale
                let dequant = ((q_clamped as i8 - zp) as f32) * scale;
                let error = (val - dequant).abs();
                total_error += error as f64;
                max_error = max_error.max(error);
                total_sq_error += (error * error) as f64;
                total_sq_signal += (val * val) as f64;
            }
        }

        // Pack INT4 values (2 per byte, low nibble first)
        // Values are already in [0, 15] range
        let packed = pack_int4_unsigned(&quantized);

        // Compute stats
        let n = values.len() as f64;
        let stats = QuantizeStats {
            mean_abs_error: (total_error / n) as f32,
            max_abs_error: max_error,
            rmse: (total_sq_error / n).sqrt() as f32,
            snr_db: if total_sq_error > 1e-10 {
                (10.0 * (total_sq_signal / total_sq_error).log10()) as f32
            } else {
                f32::INFINITY
            },
            compression_ratio: compute_compression_ratio(
                values.len(),
                packed.len(),
                scales.len(),
                Some(zero_points.len()),
            ),
        };

        (packed, scales, Some(zero_points), stats)
    }

    /// Quantizes values using INT8 symmetric quantization.
    fn quantize_int8_symmetric(
        &self,
        values: &[f32],
    ) -> (Vec<u8>, Vec<half::f16>, Option<Vec<i8>>, QuantizeStats) {
        let block_size = self.config.block_size;
        let num_blocks = (values.len() + block_size - 1) / block_size;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized = Vec::with_capacity(values.len());
        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;
        let mut total_sq_error = 0.0f64;
        let mut total_sq_signal = 0.0f64;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-10 {
                max_abs / 127.0
            } else {
                1.0
            };
            scales.push(half::f16::from_f32(scale));

            for &val in block {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                quantized.push(q as u8);

                let dequant = (q as f32) * scale;
                let error = (val - dequant).abs();
                total_error += error as f64;
                max_error = max_error.max(error);
                total_sq_error += (error * error) as f64;
                total_sq_signal += (val * val) as f64;
            }
        }

        let n = values.len() as f64;
        let stats = QuantizeStats {
            mean_abs_error: (total_error / n) as f32,
            max_abs_error: max_error,
            rmse: (total_sq_error / n).sqrt() as f32,
            snr_db: if total_sq_error > 1e-10 {
                (10.0 * (total_sq_signal / total_sq_error).log10()) as f32
            } else {
                f32::INFINITY
            },
            compression_ratio: (values.len() * 4) as f32
                / (quantized.len() + scales.len() * 2) as f32,
        };

        (quantized, scales, None, stats)
    }

    /// Quantizes values using INT8 asymmetric quantization.
    fn quantize_int8_asymmetric(
        &self,
        values: &[f32],
    ) -> (Vec<u8>, Vec<half::f16>, Option<Vec<i8>>, QuantizeStats) {
        let block_size = self.config.block_size;
        let num_blocks = (values.len() + block_size - 1) / block_size;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);
        let mut quantized = Vec::with_capacity(values.len());
        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;
        let mut total_sq_error = 0.0f64;
        let mut total_sq_signal = 0.0f64;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            let (min_val, max_val) = block
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });

            let range = max_val - min_val;
            let scale = if range > 1e-10 { range / 255.0 } else { 1.0 };
            let zp = if scale > 1e-10 {
                (-min_val / scale).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };

            scales.push(half::f16::from_f32(scale));
            zero_points.push(zp);

            for &val in block {
                let q = ((val / scale) + (zp as f32)).round().clamp(-128.0, 127.0) as i8;
                quantized.push(q as u8);

                let dequant = ((q - zp) as f32) * scale;
                let error = (val - dequant).abs();
                total_error += error as f64;
                max_error = max_error.max(error);
                total_sq_error += (error * error) as f64;
                total_sq_signal += (val * val) as f64;
            }
        }

        let n = values.len() as f64;
        let stats = QuantizeStats {
            mean_abs_error: (total_error / n) as f32,
            max_abs_error: max_error,
            rmse: (total_sq_error / n).sqrt() as f32,
            snr_db: if total_sq_error > 1e-10 {
                (10.0 * (total_sq_signal / total_sq_error).log10()) as f32
            } else {
                f32::INFINITY
            },
            compression_ratio: (values.len() * 4) as f32
                / (quantized.len() + scales.len() * 2 + zero_points.len()) as f32,
        };

        (quantized, scales, Some(zero_points), stats)
    }

    /// Dequantizes a previously quantized tensor back to F32.
    ///
    /// This is useful for verification and comparison.
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Tensor, QuantizeError> {
        let values = match quantized.format {
            QuantizeFormat::Int4Symmetric => self.dequantize_int4_symmetric(quantized),
            QuantizeFormat::Int4Asymmetric => self.dequantize_int4_asymmetric(quantized)?,
            QuantizeFormat::Int8Symmetric => self.dequantize_int8_symmetric(quantized),
            QuantizeFormat::Int8Asymmetric => self.dequantize_int8_asymmetric(quantized)?,
        };

        Tensor::from_vec(values, quantized.shape.as_slice(), &Device::Cpu)
            .map_err(QuantizeError::TensorOp)
    }

    fn dequantize_int4_symmetric(&self, quantized: &QuantizedTensor) -> Vec<f32> {
        let unpacked = unpack_int4_signed(&quantized.data, quantized.num_values);
        let mut values = Vec::with_capacity(quantized.num_values);

        for (i, &q) in unpacked.iter().enumerate() {
            let block_idx = i / quantized.block_size;
            let scale = quantized.scales[block_idx].to_f32();
            values.push((q as f32) * scale);
        }

        values
    }

    fn dequantize_int4_asymmetric(
        &self,
        quantized: &QuantizedTensor,
    ) -> Result<Vec<f32>, QuantizeError> {
        let unpacked = unpack_int4_unsigned(&quantized.data, quantized.num_values);
        let zero_points = quantized
            .zero_points
            .as_ref()
            .ok_or(QuantizeError::MissingZeroPoints)?;
        let mut values = Vec::with_capacity(quantized.num_values);

        for (i, &q) in unpacked.iter().enumerate() {
            let block_idx = i / quantized.block_size;
            let scale = quantized.scales[block_idx].to_f32();
            let zp = zero_points[block_idx];
            values.push(((q as i8 - zp) as f32) * scale);
        }

        Ok(values)
    }

    fn dequantize_int8_symmetric(&self, quantized: &QuantizedTensor) -> Vec<f32> {
        let mut values = Vec::with_capacity(quantized.num_values);

        for (i, &q) in quantized.data.iter().enumerate() {
            let block_idx = i / quantized.block_size;
            let scale = quantized.scales[block_idx].to_f32();
            values.push((q as i8 as f32) * scale);
        }

        values
    }

    fn dequantize_int8_asymmetric(
        &self,
        quantized: &QuantizedTensor,
    ) -> Result<Vec<f32>, QuantizeError> {
        let zero_points = quantized
            .zero_points
            .as_ref()
            .ok_or(QuantizeError::MissingZeroPoints)?;
        let mut values = Vec::with_capacity(quantized.num_values);

        for (i, &q) in quantized.data.iter().enumerate() {
            let block_idx = i / quantized.block_size;
            let scale = quantized.scales[block_idx].to_f32();
            let zp = zero_points[block_idx];
            values.push(((q as i8 - zp) as f32) * scale);
        }

        Ok(values)
    }
}

/// Packs signed INT4 values ([-8, 7]) into bytes.
/// Two values per byte, low nibble first.
/// Values are converted from signed to unsigned [0, 15] for packing.
fn pack_int4_signed(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);

    for chunk in values.chunks(2) {
        // Convert signed [-8, 7] to unsigned [0, 15] by adding 8
        let low = ((chunk[0] + 8) as u8) & 0x0F;
        let high = if chunk.len() > 1 {
            ((chunk[1] + 8) as u8) & 0x0F
        } else {
            0
        };
        packed.push(low | (high << 4));
    }

    packed
}

/// Packs unsigned INT4 values ([0, 15]) into bytes.
/// Two values per byte, low nibble first.
fn pack_int4_unsigned(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);

    for chunk in values.chunks(2) {
        let low = (chunk[0] as u8) & 0x0F;
        let high = if chunk.len() > 1 {
            (chunk[1] as u8) & 0x0F
        } else {
            0
        };
        packed.push(low | (high << 4));
    }

    packed
}

/// Unpacks bytes to signed INT4 values ([-8, 7]).
fn unpack_int4_signed(packed: &[u8], num_values: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(num_values);

    for &byte in packed.iter() {
        // Low nibble
        let low = (byte & 0x0F) as i8 - 8; // Convert [0, 15] back to [-8, 7]
        values.push(low);
        if values.len() >= num_values {
            break;
        }

        // High nibble
        let high = ((byte >> 4) & 0x0F) as i8 - 8;
        values.push(high);
        if values.len() >= num_values {
            break;
        }
    }

    values.truncate(num_values);
    values
}

/// Unpacks bytes to unsigned INT4 values ([0, 15]).
fn unpack_int4_unsigned(packed: &[u8], num_values: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(num_values);

    for &byte in packed {
        // Low nibble
        values.push(byte & 0x0F);
        if values.len() >= num_values {
            break;
        }

        // High nibble
        values.push((byte >> 4) & 0x0F);
        if values.len() >= num_values {
            break;
        }
    }

    values.truncate(num_values);
    values
}

/// Computes compression ratio.
fn compute_compression_ratio(
    num_values: usize,
    packed_bytes: usize,
    num_scales: usize,
    num_zero_points: Option<usize>,
) -> f32 {
    // Original: FP32 = 4 bytes per value
    let original_bytes = num_values * 4;

    // Quantized: packed data + scales (2 bytes each) + optional zero points (1 byte each)
    let quantized_bytes = packed_bytes + num_scales * 2 + num_zero_points.unwrap_or(0);

    original_bytes as f32 / quantized_bytes as f32
}

/// Quantizes a model's safetensors file and saves in quantized format.
///
/// This is the high-level API for model quantization.
#[derive(Debug)]
pub struct ModelQuantizer {
    config: QuantizeConfig,
}

impl ModelQuantizer {
    /// Creates a new model quantizer.
    pub fn new(config: QuantizeConfig) -> Self {
        Self { config }
    }

    /// Quantizes a tensor and returns metadata for saving.
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor, QuantizeError> {
        let quantizer = Quantizer::new(self.config.clone());
        quantizer.quantize_tensor(tensor)
    }
}

// =============================================================================
// Runtime Quantization
// =============================================================================

/// Configuration for runtime quantization during model loading.
#[derive(Debug, Clone)]
pub struct RuntimeQuantConfig {
    /// Quantization format to use.
    pub format: QuantizeFormat,
    /// Block size for quantization.
    pub block_size: usize,
    /// Minimum tensor size (in elements) to quantize.
    /// Smaller tensors are kept in original precision.
    pub min_tensor_size: usize,
    /// Weight name patterns to exclude from quantization.
    /// Typically includes layer norms, embeddings, lm_head.
    pub exclude_patterns: Vec<String>,
}

impl Default for RuntimeQuantConfig {
    fn default() -> Self {
        Self {
            format: QuantizeFormat::Int8Symmetric,
            block_size: DEFAULT_BLOCK_SIZE,
            min_tensor_size: 1024, // Don't quantize small tensors
            exclude_patterns: vec![
                "norm".to_string(),
                "embed".to_string(),
                "lm_head".to_string(),
            ],
        }
    }
}

impl RuntimeQuantConfig {
    /// Creates INT4 symmetric runtime quantization config.
    pub fn int4_symmetric() -> Self {
        Self {
            format: QuantizeFormat::Int4Symmetric,
            ..Default::default()
        }
    }

    /// Creates INT8 symmetric runtime quantization config.
    pub fn int8_symmetric() -> Self {
        Self {
            format: QuantizeFormat::Int8Symmetric,
            ..Default::default()
        }
    }

    /// Adds a pattern to exclude from quantization.
    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_string());
        self
    }

    /// Sets minimum tensor size for quantization.
    pub fn with_min_size(mut self, size: usize) -> Self {
        self.min_tensor_size = size;
        self
    }

    /// Checks if a tensor name should be excluded from quantization.
    pub fn should_exclude(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        self.exclude_patterns
            .iter()
            .any(|p| name_lower.contains(&p.to_lowercase()))
    }
}

/// A runtime-quantized weight that stores data in compressed format
/// and dequantizes on access.
#[derive(Debug)]
pub struct RuntimeQuantizedWeight {
    /// Quantized tensor data.
    quantized: QuantizedTensor,
    /// Original device for dequantization.
    device: Device,
    /// Target dtype for dequantization.
    target_dtype: DType,
}

impl RuntimeQuantizedWeight {
    /// Creates a new runtime-quantized weight from a tensor.
    pub fn from_tensor(
        tensor: &Tensor,
        config: &RuntimeQuantConfig,
    ) -> Result<Self, QuantizeError> {
        let quantizer = Quantizer::new(QuantizeConfig {
            format: config.format,
            block_size: config.block_size,
            activation_aware: false,
        });

        let quantized = quantizer.quantize_tensor(tensor)?;

        Ok(Self {
            quantized,
            device: tensor.device().clone(),
            target_dtype: tensor.dtype(),
        })
    }

    /// Dequantizes and returns the weight tensor.
    pub fn dequantize(&self) -> Result<Tensor, QuantizeError> {
        let quantizer = Quantizer::new(QuantizeConfig {
            format: self.quantized.format,
            block_size: self.quantized.block_size,
            activation_aware: false,
        });

        let tensor = quantizer.dequantize(&self.quantized)?;

        // Move to original device and dtype
        tensor
            .to_device(&self.device)
            .and_then(|t| t.to_dtype(self.target_dtype))
            .map_err(QuantizeError::TensorOp)
    }

    /// Returns memory usage of the quantized representation in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.quantized.data.len()
            + self.quantized.scales.len() * 2
            + self.quantized.zero_points.as_ref().map_or(0, |zp| zp.len())
    }

    /// Returns the original tensor size in bytes (for comparison).
    pub fn original_bytes(&self) -> usize {
        let dtype_size = match self.target_dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 4,
        };
        self.quantized.num_values * dtype_size
    }

    /// Returns the compression ratio achieved.
    pub fn compression_ratio(&self) -> f32 {
        self.original_bytes() as f32 / self.memory_bytes() as f32
    }

    /// Returns quantization statistics.
    pub fn stats(&self) -> &QuantizeStats {
        &self.quantized.stats
    }

    /// Returns the original tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.quantized.shape
    }
}

/// Stores model weights in quantized format for memory-efficient inference.
///
/// Weights are quantized on insertion and dequantized on access.
/// This trades compute for memory, allowing larger models to fit in VRAM.
#[derive(Debug)]
pub struct RuntimeQuantizedStore {
    /// Configuration for quantization.
    config: RuntimeQuantConfig,
    /// Quantized weights by name.
    weights: std::collections::HashMap<String, RuntimeQuantizedWeight>,
    /// Non-quantized weights (excluded patterns, small tensors).
    passthrough: std::collections::HashMap<String, Tensor>,
    /// Total memory saved in bytes.
    memory_saved: usize,
}

impl RuntimeQuantizedStore {
    /// Creates a new runtime-quantized weight store.
    pub fn new(config: RuntimeQuantConfig) -> Self {
        Self {
            config,
            weights: std::collections::HashMap::new(),
            passthrough: std::collections::HashMap::new(),
            memory_saved: 0,
        }
    }

    /// Inserts a weight tensor, quantizing it if appropriate.
    pub fn insert(&mut self, name: String, tensor: Tensor) -> Result<(), QuantizeError> {
        // Check if this tensor should be quantized
        let should_quantize = tensor.elem_count() >= self.config.min_tensor_size
            && !self.config.should_exclude(&name)
            && matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16);

        if should_quantize {
            let original_size = tensor.elem_count()
                * match tensor.dtype() {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    _ => 4,
                };

            let quantized = RuntimeQuantizedWeight::from_tensor(&tensor, &self.config)?;
            let quantized_size = quantized.memory_bytes();

            self.memory_saved += original_size.saturating_sub(quantized_size);
            self.weights.insert(name, quantized);
        } else {
            self.passthrough.insert(name, tensor);
        }

        Ok(())
    }

    /// Gets a weight tensor by name, dequantizing if necessary.
    pub fn get(&self, name: &str) -> Result<Option<Tensor>, QuantizeError> {
        // Check passthrough first
        if let Some(tensor) = self.passthrough.get(name) {
            return Ok(Some(tensor.clone()));
        }

        // Check quantized weights
        if let Some(weight) = self.weights.get(name) {
            return Ok(Some(weight.dequantize()?));
        }

        Ok(None)
    }

    /// Returns the total memory saved by quantization in bytes.
    pub fn memory_saved(&self) -> usize {
        self.memory_saved
    }

    /// Returns the number of quantized weights.
    pub fn num_quantized(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of passthrough (non-quantized) weights.
    pub fn num_passthrough(&self) -> usize {
        self.passthrough.len()
    }

    /// Returns total number of weights.
    pub fn len(&self) -> usize {
        self.weights.len() + self.passthrough.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the configuration.
    pub fn config(&self) -> &RuntimeQuantConfig {
        &self.config
    }

    /// Returns an iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.weights.keys().chain(self.passthrough.keys())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_symmetric_quantize() {
        let quantizer = Quantizer::int4_symmetric();

        // Create a simple tensor
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let tensor = Tensor::from_vec(values.clone(), &[256], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        assert_eq!(quantized.num_values, 256);
        assert_eq!(quantized.data.len(), 128); // 2 values per byte
        assert_eq!(quantized.scales.len(), 2); // 256 / 128 = 2 blocks
        assert!(quantized.zero_points.is_none());
        assert!(quantized.stats.rmse < 0.1);
    }

    #[test]
    fn test_int4_asymmetric_quantize() {
        let quantizer = Quantizer::int4_asymmetric();

        let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let tensor = Tensor::from_vec(values, &[256], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        assert_eq!(quantized.num_values, 256);
        assert_eq!(quantized.data.len(), 128);
        assert_eq!(quantized.scales.len(), 2);
        assert!(quantized.zero_points.is_some());
        assert_eq!(quantized.zero_points.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_int8_symmetric_quantize() {
        let config = QuantizeConfig::int8_symmetric();
        let quantizer = Quantizer::new(config);

        let values: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[1024], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        assert_eq!(quantized.num_values, 1024);
        assert_eq!(quantized.data.len(), 1024); // 1 byte per value
        assert_eq!(quantized.scales.len(), 8); // 1024 / 128 = 8 blocks
    }

    #[test]
    fn test_pack_unpack_int4_signed() {
        let values: Vec<i8> = vec![-8, -4, 0, 4, 7, -1, 2, 3];
        let packed = pack_int4_signed(&values);
        let unpacked = unpack_int4_signed(&packed, values.len());

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_int4_unsigned() {
        let values: Vec<i8> = vec![0, 1, 2, 3, 8, 15, 7, 10];
        let packed = pack_int4_unsigned(&values);
        let unpacked = unpack_int4_unsigned(&packed, values.len());

        let values_u8: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        assert_eq!(values_u8, unpacked);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let quantizer = Quantizer::int4_symmetric();

        let original: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let tensor = Tensor::from_vec(original.clone(), &[512], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        let dequantized = quantizer.dequantize(&quantized).unwrap();

        let recovered: Vec<f32> = dequantized.to_vec1().unwrap();

        // Check that error is bounded
        // INT4 symmetric: 16 levels over range ~5.1 → step ~0.34 → max error ~0.17
        // Allow 0.20 to account for edge cases at quantization boundaries
        let max_error = original
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 0.20,
            "Max error {} too large for INT4",
            max_error
        );
    }

    #[test]
    fn test_compression_ratio() {
        let quantizer = Quantizer::int4_symmetric();

        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[1024], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        // INT4: ~8x compression (4 bytes -> 0.5 bytes, minus scale overhead)
        assert!(
            quantized.stats.compression_ratio > 6.0,
            "Compression ratio {} too low",
            quantized.stats.compression_ratio
        );
    }

    #[test]
    fn test_snr_calculation() {
        let quantizer = Quantizer::int4_symmetric();

        // Create a tensor with known signal
        let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
        let tensor = Tensor::from_vec(values, &[256], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        // SNR should be positive for reasonable quantization
        assert!(
            quantized.stats.snr_db > 20.0,
            "SNR {} dB too low",
            quantized.stats.snr_db
        );
    }

    #[test]
    fn test_empty_tensor_error() {
        let quantizer = Quantizer::int4_symmetric();
        let tensor = Tensor::from_vec(Vec::<f32>::new(), &[0], &Device::Cpu).unwrap();

        let result = quantizer.quantize_tensor(&tensor);
        assert!(matches!(result, Err(QuantizeError::EmptyTensor)));
    }

    #[test]
    fn test_2d_tensor_shape_preserved() {
        let quantizer = Quantizer::int4_symmetric();

        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[32, 32], &Device::Cpu).unwrap();

        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert_eq!(quantized.shape, vec![32, 32]);

        let dequantized = quantizer.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.dims(), &[32, 32]);
    }

    // =========================================================================
    // Runtime Quantization Tests
    // =========================================================================

    #[test]
    fn test_runtime_quant_config_default() {
        let config = RuntimeQuantConfig::default();

        assert_eq!(config.format, QuantizeFormat::Int8Symmetric);
        assert_eq!(config.block_size, DEFAULT_BLOCK_SIZE);
        assert_eq!(config.min_tensor_size, 1024);
        assert!(config.exclude_patterns.contains(&"norm".to_string()));
        assert!(config.exclude_patterns.contains(&"embed".to_string()));
        assert!(config.exclude_patterns.contains(&"lm_head".to_string()));
    }

    #[test]
    fn test_runtime_quant_config_int4() {
        let config = RuntimeQuantConfig::int4_symmetric();

        assert_eq!(config.format, QuantizeFormat::Int4Symmetric);
    }

    #[test]
    fn test_runtime_quant_config_int8() {
        let config = RuntimeQuantConfig::int8_symmetric();

        assert_eq!(config.format, QuantizeFormat::Int8Symmetric);
    }

    #[test]
    fn test_runtime_quant_config_exclude() {
        let config = RuntimeQuantConfig::default().exclude("rotary");

        assert!(config.exclude_patterns.contains(&"rotary".to_string()));
    }

    #[test]
    fn test_runtime_quant_config_should_exclude() {
        let config = RuntimeQuantConfig::default();

        // Should exclude norm layers
        assert!(config.should_exclude("model.layers.0.input_layernorm.weight"));
        assert!(config.should_exclude("model.norm.weight"));

        // Should exclude embeddings
        assert!(config.should_exclude("model.embed_tokens.weight"));

        // Should exclude lm_head
        assert!(config.should_exclude("lm_head.weight"));

        // Should NOT exclude attention/mlp weights
        assert!(!config.should_exclude("model.layers.0.self_attn.q_proj.weight"));
        assert!(!config.should_exclude("model.layers.0.mlp.up_proj.weight"));
    }

    #[test]
    fn test_runtime_quant_config_case_insensitive() {
        let config = RuntimeQuantConfig::default();

        // Should match case-insensitively
        assert!(config.should_exclude("Model.NORM.weight"));
        assert!(config.should_exclude("EMBED_TOKENS.weight"));
    }

    #[test]
    fn test_runtime_quantized_weight_from_tensor() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[2048], &Device::Cpu).unwrap();

        let quantized = RuntimeQuantizedWeight::from_tensor(&tensor, &config).unwrap();

        assert_eq!(quantized.shape(), &[2048]);
        assert!(quantized.compression_ratio() > 1.0);
    }

    #[test]
    fn test_runtime_quantized_weight_dequantize() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values.clone(), &[2048], &Device::Cpu).unwrap();

        let quantized = RuntimeQuantizedWeight::from_tensor(&tensor, &config).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        assert_eq!(dequantized.dims(), &[2048]);

        // Check error is bounded
        let recovered: Vec<f32> = dequantized.to_vec1().unwrap();
        let max_error = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // INT8 should have smaller error than INT4
        assert!(
            max_error < 0.01,
            "Max error {} too large for INT8",
            max_error
        );
    }

    #[test]
    fn test_runtime_quantized_weight_memory_savings() {
        let config = RuntimeQuantConfig::int4_symmetric();
        let values: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[4096], &Device::Cpu).unwrap();

        let quantized = RuntimeQuantizedWeight::from_tensor(&tensor, &config).unwrap();

        // INT4 should achieve ~6-8x compression for large tensors
        let ratio = quantized.compression_ratio();
        assert!(ratio > 5.0, "Compression ratio {} too low for INT4", ratio);
    }

    #[test]
    fn test_runtime_quantized_store_new() {
        let config = RuntimeQuantConfig::default();
        let store = RuntimeQuantizedStore::new(config);

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.num_quantized(), 0);
        assert_eq!(store.num_passthrough(), 0);
    }

    #[test]
    fn test_runtime_quantized_store_insert_quantized() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[2048], &Device::Cpu).unwrap();

        store
            .insert("model.layers.0.self_attn.q_proj.weight".to_string(), tensor)
            .unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.num_quantized(), 1);
        assert_eq!(store.num_passthrough(), 0);
        assert!(store.memory_saved() > 0);
    }

    #[test]
    fn test_runtime_quantized_store_insert_excluded() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[2048], &Device::Cpu).unwrap();

        // Insert a norm layer (should be excluded)
        store
            .insert("model.layers.0.input_layernorm.weight".to_string(), tensor)
            .unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.num_quantized(), 0);
        assert_eq!(store.num_passthrough(), 1);
        assert_eq!(store.memory_saved(), 0);
    }

    #[test]
    fn test_runtime_quantized_store_insert_small_tensor() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        // Small tensor (below min_tensor_size)
        let values: Vec<f32> = (0..512).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[512], &Device::Cpu).unwrap();

        store.insert("small_weight".to_string(), tensor).unwrap();

        assert_eq!(store.num_quantized(), 0);
        assert_eq!(store.num_passthrough(), 1);
    }

    #[test]
    fn test_runtime_quantized_store_get() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        let values: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values.clone(), &[2048], &Device::Cpu).unwrap();

        store.insert("weight".to_string(), tensor).unwrap();

        let retrieved = store.get("weight").unwrap().unwrap();
        assert_eq!(retrieved.dims(), &[2048]);

        // Check values are approximately correct
        let recovered: Vec<f32> = retrieved.to_vec1().unwrap();
        let max_error = values
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.01);
    }

    #[test]
    fn test_runtime_quantized_store_get_passthrough() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        let values: Vec<f32> = (0..2048).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values.clone(), &[2048], &Device::Cpu).unwrap();

        // Insert as excluded (norm layer)
        store
            .insert("model.norm.weight".to_string(), tensor)
            .unwrap();

        let retrieved = store.get("model.norm.weight").unwrap().unwrap();

        // Should be exact match for passthrough
        let recovered: Vec<f32> = retrieved.to_vec1().unwrap();
        assert_eq!(values, recovered);
    }

    #[test]
    fn test_runtime_quantized_store_get_not_found() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let store = RuntimeQuantizedStore::new(config);

        let result = store.get("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_runtime_quantized_store_multiple_weights() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        // Insert multiple weights
        for i in 0..10 {
            let values: Vec<f32> = (0..2048)
                .map(|j| (j as f32 + i as f32 * 100.0) * 0.001)
                .collect();
            let tensor = Tensor::from_vec(values, &[2048], &Device::Cpu).unwrap();
            store
                .insert(format!("model.layers.{}.weight", i), tensor)
                .unwrap();
        }

        assert_eq!(store.len(), 10);
        assert_eq!(store.num_quantized(), 10);
        assert!(store.memory_saved() > 0);
    }

    #[test]
    fn test_runtime_quantized_store_names() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        // Insert quantized and passthrough weights
        let values: Vec<f32> = (0..2048).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values.clone(), &[2048], &Device::Cpu).unwrap();
        store
            .insert("q_proj.weight".to_string(), tensor.clone())
            .unwrap();
        store
            .insert("model.norm.weight".to_string(), tensor)
            .unwrap();

        let names: Vec<&String> = store.names().collect();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_runtime_quantized_int4_vs_int8() {
        let values: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.001).collect();
        let tensor = Tensor::from_vec(values.clone(), &[4096], &Device::Cpu).unwrap();

        let int4_config = RuntimeQuantConfig::int4_symmetric();
        let int8_config = RuntimeQuantConfig::int8_symmetric();

        let int4_weight = RuntimeQuantizedWeight::from_tensor(&tensor, &int4_config).unwrap();
        let int8_weight = RuntimeQuantizedWeight::from_tensor(&tensor, &int8_config).unwrap();

        // INT4 should have better compression
        assert!(
            int4_weight.compression_ratio() > int8_weight.compression_ratio(),
            "INT4 ratio {} should be higher than INT8 ratio {}",
            int4_weight.compression_ratio(),
            int8_weight.compression_ratio()
        );

        // INT8 should have lower error
        let int4_deq: Vec<f32> = int4_weight.dequantize().unwrap().to_vec1().unwrap();
        let int8_deq: Vec<f32> = int8_weight.dequantize().unwrap().to_vec1().unwrap();

        let int4_error: f32 = values
            .iter()
            .zip(int4_deq.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / values.len() as f32;

        let int8_error: f32 = values
            .iter()
            .zip(int8_deq.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / values.len() as f32;

        assert!(
            int8_error < int4_error,
            "INT8 error {} should be lower than INT4 error {}",
            int8_error,
            int4_error
        );
    }

    #[test]
    fn test_runtime_quantized_weight_2d_shape() {
        let config = RuntimeQuantConfig::int8_symmetric();
        let values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
        let tensor = Tensor::from_vec(values, &[64, 64], &Device::Cpu).unwrap();

        let quantized = RuntimeQuantizedWeight::from_tensor(&tensor, &config).unwrap();
        assert_eq!(quantized.shape(), &[64, 64]);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[64, 64]);
    }

    #[test]
    fn test_runtime_quantized_store_memory_tracking() {
        let config = RuntimeQuantConfig::int4_symmetric();
        let mut store = RuntimeQuantizedStore::new(config);

        // Insert several large weights
        for i in 0..5 {
            let values: Vec<f32> = (0..8192)
                .map(|j| (j as f32 + i as f32 * 1000.0) * 0.0001)
                .collect();
            let tensor = Tensor::from_vec(values, &[8192], &Device::Cpu).unwrap();
            store.insert(format!("layer.{}.weight", i), tensor).unwrap();
        }

        // Should save significant memory with INT4
        // 5 * 8192 * 4 bytes = 163,840 bytes original
        // INT4 should compress to ~1/6th = ~27,000 bytes
        // Memory saved should be ~136,000 bytes
        assert!(
            store.memory_saved() > 100_000,
            "Memory saved {} too low",
            store.memory_saved()
        );
    }
}
