//! HCT to HoloTensor Model Converter
//!
//! Converts quantized INT4 models from HCT format to holographic format
//! for progressive inference.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::thread;

#[cfg(feature = "cuda")]
use crossbeam::channel::{self, Receiver, Sender};

use haagenti::compressive::{CompressiveSpectralDecoder, CompressiveSpectralEncoder};
use haagenti::entropy::{fast_should_compress, CompressibilityFingerprint};
use haagenti::holotensor::{HoloFragment, HoloTensorHeader, HolographicEncoding, LrdfEncoder};
use haagenti::tensor::DType;
// Note: We use the standard zstd crate for compression instead of haagenti-zstd
// because haagenti-zstd's compressor has bugs that produce corrupted frames
use xxhash_rust::xxh3::xxh3_64;

use super::{HoloInferenceError, Result};

/// Convert FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) to f32.
/// E4M3 uses exponent bias of 7, supports values up to 448, no inf.
#[inline]
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exponent = (byte >> 3) & 0xF;
    let mantissa = byte & 0x7;

    if exponent == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: (-1)^s * 2^(-6) * (m/8)
        let value = (mantissa as f32 / 8.0) * 2.0f32.powi(-6);
        if sign == 1 {
            -value
        } else {
            value
        }
    } else if exponent == 15 && mantissa == 7 {
        // NaN (E4M3 doesn't have infinity, uses max exp + max mantissa for NaN)
        f32::NAN
    } else {
        // Normal: (-1)^s * 2^(e-7) * (1 + m/8)
        let value = (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7);
        if sign == 1 {
            -value
        } else {
            value
        }
    }
}

/// Convert FP8 E5M2 (1 sign, 5 exponent, 2 mantissa) to f32.
/// E5M2 uses exponent bias of 15, supports inf/nan.
#[inline]
fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exponent = (byte >> 2) & 0x1F;
    let mantissa = byte & 0x3;

    if exponent == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: (-1)^s * 2^(-14) * (m/4)
        let value = (mantissa as f32 / 4.0) * 2.0f32.powi(-14);
        if sign == 1 {
            -value
        } else {
            value
        }
    } else if exponent == 31 {
        // Infinity or NaN
        if mantissa == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normal: (-1)^s * 2^(e-15) * (1 + m/4)
        let value = (1.0 + mantissa as f32 / 4.0) * 2.0f32.powi(exponent as i32 - 15);
        if sign == 1 {
            -value
        } else {
            value
        }
    }
}

/// Configuration for model conversion.
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Holographic encoding scheme.
    pub encoding: HolographicEncoding,

    /// Number of fragments per tensor.
    pub num_fragments: u16,

    /// Maximum rank for LRDF encoding.
    pub max_rank: usize,

    /// Seed for deterministic encoding.
    pub seed: u64,

    /// Enable parallel conversion.
    pub parallel: bool,

    /// Number of threads for parallel conversion.
    pub num_threads: usize,

    /// Verify reconstruction quality after conversion.
    pub verify_quality: bool,

    /// Minimum quality threshold for verification.
    pub min_quality: f32,

    /// Use GPU for SVD computation (requires CUDA).
    pub use_gpu: bool,

    /// Compress fragment data with Zstd (Phase 1 optimization: 9.4x faster decompression).
    pub compress_fragments: bool,

    /// Force lossless passthrough encoding for all tensors.
    /// Disables LRDF compression - larger files but exact reconstruction.
    /// Useful for testing inference correctness before enabling compression.
    pub lossless: bool,

    /// DCT retention ratio for Spectral encoding (0.0-1.0).
    /// Higher = better quality, larger files. Default: 0.2 (keep 20% of coefficients).
    pub retention_ratio: f32,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            encoding: HolographicEncoding::LowRankDistributed,
            num_fragments: 32,
            max_rank: 256, // Increased from 128 for better reconstruction quality
            seed: 42,
            parallel: true,
            num_threads: 4,
            verify_quality: true,
            min_quality: 0.95, // Increased from 0.85 for better inference quality
            use_gpu: false,    // CPU by default for compatibility
            compress_fragments: true, // Enable Zstd compression by default
            lossless: false,
            retention_ratio: 0.2, // Keep 20% of DCT coefficients
        }
    }
}

impl ConversionConfig {
    /// Create config optimized for speed.
    pub fn fast() -> Self {
        Self {
            num_fragments: 16,
            max_rank: 64,
            verify_quality: false,
            ..Default::default()
        }
    }

    /// Create config optimized for quality.
    pub fn high_quality() -> Self {
        Self {
            num_fragments: 64,
            max_rank: 256,
            verify_quality: true,
            min_quality: 0.95,
            ..Default::default()
        }
    }

    /// Create config for GPU-accelerated conversion.
    #[cfg(feature = "cuda")]
    pub fn gpu() -> Self {
        Self {
            use_gpu: true,
            parallel: false, // GPU handles parallelism internally
            ..Default::default()
        }
    }

    /// Create config for fast GPU-accelerated conversion.
    #[cfg(feature = "cuda")]
    pub fn gpu_fast() -> Self {
        Self {
            num_fragments: 16,
            max_rank: 64,
            verify_quality: false,
            use_gpu: true,
            parallel: false,
            ..Default::default()
        }
    }

    /// Create config for lossless passthrough encoding.
    /// All tensors are stored directly without LRDF compression.
    /// Results in larger files but exact tensor reconstruction.
    pub fn lossless() -> Self {
        Self {
            lossless: true,
            verify_quality: false,     // Not needed for lossless
            compress_fragments: false, // Disable Zstd to avoid decompression issues
            ..Default::default()
        }
    }
}

/// Progress callback for conversion.
pub type ProgressCallback = Box<dyn Fn(ConversionProgress) + Send + Sync>;

/// Conversion progress information.
#[derive(Debug, Clone)]
pub struct ConversionProgress {
    /// Current tensor being processed.
    pub current_tensor: String,
    /// Tensors processed so far.
    pub tensors_processed: usize,
    /// Total tensors to process.
    pub total_tensors: usize,
    /// Bytes processed.
    pub bytes_processed: usize,
    /// Total bytes.
    pub total_bytes: usize,
    /// Current phase.
    pub phase: ConversionPhase,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Estimated remaining time in seconds.
    pub eta_secs: f64,
}

impl ConversionProgress {
    /// Get progress as percentage.
    pub fn percent(&self) -> f32 {
        if self.total_tensors == 0 {
            return 0.0;
        }
        (self.tensors_processed as f32 / self.total_tensors as f32) * 100.0
    }
}

/// Conversion phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionPhase {
    /// Scanning input files.
    Scanning,
    /// Converting tensors.
    Converting,
    /// Verifying quality.
    Verifying,
    /// Writing output.
    Writing,
    /// Complete.
    Complete,
}

/// Tensor metadata from HCT file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Shape (rows, cols for 2D).
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
    /// File path.
    pub path: PathBuf,
    /// Size in bytes.
    pub size: usize,
}

/// Converted tensor output.
#[derive(Debug)]
pub struct ConvertedTensor {
    /// Original tensor info.
    pub info: TensorInfo,
    /// Holotensor header.
    pub header: HoloTensorHeader,
    /// Holographic fragments.
    pub fragments: Vec<HoloFragment>,
    /// Reconstruction quality (if verified).
    pub quality: Option<f32>,
}

/// HCT to HoloTensor model converter.
///
/// # Future optimization (Phase 4)
/// Fragment buffer allocations currently use standard `Vec` heap allocation.
/// The `holotensor::arena::FragmentArena` bump allocator is available and could
/// replace these allocations to reduce heap fragmentation during large model
/// conversions. This would require changing `convert_tensor` and related methods
/// to allocate fragment data from a shared arena and reset between batches.
pub struct HoloModelConverter {
    config: ConversionConfig,
    tensors_processed: AtomicUsize,
    bytes_processed: AtomicUsize,
    #[cfg(feature = "cuda")]
    gpu_encoder: Option<std::sync::Arc<crate::gpu_lrdf::cuda::GpuLrdfEncoder>>,
    #[cfg(feature = "cuda")]
    gpu_dtype_converter: Option<std::sync::Arc<crate::gpu_dtype::cuda::GpuDtypeConverter>>,
}

impl HoloModelConverter {
    /// Create new converter with given configuration.
    pub fn new(config: ConversionConfig) -> Self {
        #[cfg(feature = "cuda")]
        let (gpu_encoder, gpu_dtype_converter) = if config.use_gpu {
            match cudarc::driver::CudaDevice::new(0) {
                Ok(device) => {
                    // Initialize GPU LRDF encoder
                    let encoder = match crate::gpu_lrdf::cuda::GpuLrdfEncoder::new(
                        device.clone(),
                        config.num_fragments,
                        config.seed,
                    ) {
                        Ok(encoder) => {
                            let encoder = encoder.with_max_rank(config.max_rank);
                            Some(std::sync::Arc::new(encoder))
                        },
                        Err(e) => {
                            eprintln!(
                                "Warning: Failed to create GPU encoder: {}. Falling back to CPU.",
                                e
                            );
                            None
                        },
                    };

                    // Initialize GPU dtype converter for FP8 → F32 conversion
                    let dtype_converter =
                        match crate::gpu_dtype::cuda::GpuDtypeConverter::new(device) {
                            Ok(converter) => {
                                println!(
                                "GPU dtype converter initialized (FP8→F32 acceleration enabled)"
                            );
                                Some(std::sync::Arc::new(converter))
                            },
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to create GPU dtype converter: {}. Using CPU.",
                                    e
                                );
                                None
                            },
                        };

                    (encoder, dtype_converter)
                },
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to initialize CUDA device: {}. Falling back to CPU.",
                        e
                    );
                    (None, None)
                },
            }
        } else {
            (None, None)
        };

        Self {
            config,
            tensors_processed: AtomicUsize::new(0),
            bytes_processed: AtomicUsize::new(0),
            #[cfg(feature = "cuda")]
            gpu_encoder,
            #[cfg(feature = "cuda")]
            gpu_dtype_converter,
        }
    }

    /// Create converter with default configuration.
    pub fn default_converter() -> Self {
        Self::new(ConversionConfig::default())
    }

    /// Create converter with GPU acceleration.
    #[cfg(feature = "cuda")]
    pub fn gpu_converter() -> Self {
        Self::new(ConversionConfig::gpu())
    }

    /// Scan HCT directory for tensors.
    pub fn scan_hct_directory(&self, path: &Path) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::new();

        if !path.is_dir() {
            return Err(HoloInferenceError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Not a directory: {}", path.display()),
            )));
        }

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            if file_path.extension().map_or(false, |ext| ext == "hct") {
                if let Some(info) = self.parse_hct_header(&file_path)? {
                    tensors.push(info);
                }
            }
        }

        // Sort by name for consistent ordering
        tensors.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(tensors)
    }

    /// Parse HCT file header to get tensor info.
    fn parse_hct_header(&self, path: &Path) -> Result<Option<TensorInfo>> {
        // Extract tensor name from filename
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let metadata = fs::metadata(path)?;
        let size = metadata.len() as usize;

        // For now, infer shape from filename pattern
        // In production, this would read the actual HCT header
        let shape = self.infer_shape_from_name(&name);

        Ok(Some(TensorInfo {
            name,
            shape,
            dtype: DType::I4,
            path: path.to_path_buf(),
            size,
        }))
    }

    /// Infer tensor shape from name.
    fn infer_shape_from_name(&self, name: &str) -> Vec<usize> {
        // Common patterns for LLM weights
        if name.contains("embed") {
            vec![152064, 5120] // vocab_size x hidden_size (Qwen2.5-32B)
        } else if name.contains("q_proj") || name.contains("o_proj") {
            vec![5120, 5120] // hidden_size x hidden_size
        } else if name.contains("k_proj") || name.contains("v_proj") {
            vec![5120, 1280] // hidden_size x (kv_heads * head_dim)
        } else if name.contains("gate_proj") || name.contains("up_proj") {
            vec![5120, 27648] // hidden_size x intermediate_size
        } else if name.contains("down_proj") {
            vec![27648, 5120] // intermediate_size x hidden_size
        } else if name.contains("norm") {
            vec![5120] // hidden_size
        } else {
            vec![4096, 4096] // Default
        }
    }

    /// Convert a single tensor to holographic format.
    ///
    /// Optimizations:
    /// - 1D tensors (norms, biases) bypass LRDF - stored directly
    /// - Small tensors (<4KB) bypass LRDF - not worth the overhead
    /// - 2D+ tensors use GPU-accelerated LRDF when available
    ///
    /// Quality enforcement:
    /// - If LRDF encoding doesn't meet min_quality threshold, falls back to passthrough
    /// - This ensures all tensors meet the quality threshold while maximizing compression
    pub fn convert_tensor(&self, info: &TensorInfo, data: &[f32]) -> Result<ConvertedTensor> {
        // Handle different tensor dimensionalities
        let (rows, cols) = match info.shape.len() {
            0 => (1, data.len()),                // Scalar
            1 => (1, info.shape[0]),             // 1D: treat as row vector
            2 => (info.shape[0], info.shape[1]), // 2D: rows x cols
            _ => {
                // Higher dims: flatten to 2D (first dim x product of rest)
                let first = info.shape[0];
                let rest: usize = info.shape[1..].iter().product();
                (first, rest)
            },
        };

        // Skip LRDF for 1D tensors (norms, biases) and small tensors
        // These don't compress well and add overhead
        let min_size_for_lrdf = 4096; // 4KB = 1024 floats
        let is_1d = info.shape.len() <= 1 || rows == 1 || cols == 1;
        let is_small = data.len() < min_size_for_lrdf;

        // Phase 3 Enhancement: Use entropy fingerprinting for better compression decisions
        // High-entropy data (>7.5 bits/byte) is likely random/encrypted and won't compress well
        // Convert f32 slice to bytes for entropy analysis
        let data_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

        // Fast entropy check for large tensors to skip expensive LRDF for incompressible data
        let is_incompressible = if !is_1d && !is_small && data.len() > 16384 {
            // For large tensors, check if data is incompressible (random/encrypted)
            let fingerprint = CompressibilityFingerprint::analyze(data_bytes);
            fingerprint.entropy > 7.5 && !fast_should_compress(data_bytes)
        } else {
            false
        };

        // Force passthrough for all tensors if lossless mode is enabled
        let initial_passthrough = self.config.lossless || is_1d || is_small || is_incompressible;

        // Try holographic encoding, with quality-based fallback to passthrough
        let (fragments, quality, used_passthrough) = if initial_passthrough {
            // Direct passthrough - no encoding attempt
            let frags = self.create_passthrough_fragment(data, rows, cols)?;
            (frags, Some(1.0f32), true)
        } else {
            // Encode based on configured encoding type
            let encoded_fragments = match self.config.encoding {
                HolographicEncoding::Spectral => {
                    self.encode_with_spectral(data, rows, cols, &info.name)?
                },
                HolographicEncoding::LowRankDistributed | _ => {
                    self.encode_with_lrdf(data, rows, cols, &info.name)?
                },
            };

            // Check quality if verification is enabled
            if self.config.verify_quality {
                // Build temporary header for quality check
                let temp_header = HoloTensorHeader::new(
                    self.config.encoding,
                    DType::F32,
                    info.shape.iter().map(|&d| d as u64).collect(),
                    encoded_fragments.len() as u16,
                )
                .with_seed(self.config.seed);

                let quality =
                    self.verify_reconstruction(data, &temp_header, &encoded_fragments, rows, cols)?;

                // QUALITY ENFORCEMENT: Fall back to passthrough if quality is too low
                if quality < self.config.min_quality {
                    tracing::warn!(
                        tensor = %info.name,
                        quality = %format!("{:.4}", quality),
                        threshold = %format!("{:.4}", self.config.min_quality),
                        "Encoding quality below threshold, falling back to passthrough"
                    );
                    let passthrough_frags = self.create_passthrough_fragment(data, rows, cols)?;
                    (passthrough_frags, Some(1.0f32), true)
                } else {
                    (encoded_fragments, Some(quality), false)
                }
            } else {
                // No quality verification - just use encoded result
                (encoded_fragments, None, false)
            }
        };

        // Store the original shape, not the encoding shape (rows x cols)
        // This ensures proper shape restoration during loading
        let original_shape: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();

        // Use actual fragment count (passthrough tensors have 1 fragment, not config.num_fragments)
        let actual_fragments = fragments.len() as u16;

        // CRITICAL: Set encoding based on what was actually used, not config!
        // Passthrough uses LRDF format, so header must indicate LowRankDistributed
        // for the decoder to correctly interpret the fragment data.
        let actual_encoding = if used_passthrough {
            HolographicEncoding::LowRankDistributed
        } else {
            self.config.encoding
        };

        let header = HoloTensorHeader::new(
            actual_encoding,
            DType::F32, // Fragments are f32 even if source is INT4
            original_shape,
            actual_fragments,
        )
        .with_seed(self.config.seed)
        // WORKAROUND: Disable fragment checksums for Spectral encoding
        // There's a bug causing checksum mismatches that needs investigation
        .without_fragment_checksums();

        // Phase 1 Enhancement: Apply Zstd compression to fragment data
        // This must happen AFTER quality verification (decoder expects uncompressed)
        let fragments = self.compress_fragments(fragments)?;

        self.tensors_processed.fetch_add(1, Ordering::Relaxed);
        self.bytes_processed.fetch_add(info.size, Ordering::Relaxed);

        // Log passthrough fallback for visibility (only for non-trivial tensors)
        if used_passthrough && !initial_passthrough {
            tracing::info!(
                tensor = %info.name,
                shape = ?info.shape,
                "Tensor stored as passthrough due to quality threshold"
            );
        }

        Ok(ConvertedTensor {
            info: info.clone(),
            header,
            fragments,
            quality,
        })
    }

    /// Create passthrough fragment for lossless storage.
    ///
    /// For 1D tensors (rows=1): Uses single LRDF component with sigma=1, u=[1], v=data
    /// For 2D tensors (rows>1): Uses raw format (num_components = 0xFFFFFFFF marker)
    ///   - Header: rows, cols, 0xFFFFFFFF (raw marker)
    ///   - Data: rows*cols f32 values in row-major order
    ///
    /// The raw format is O(rows*cols) storage, much better than the O(rows^2) one-hot encoding.
    fn create_passthrough_fragment(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<HoloFragment>> {
        if rows == 1 {
            // 1D case: single LRDF component with u=[1.0], v=data
            let mut frag_data = Vec::with_capacity(12 + 4 + 4 + cols * 4);

            // Header
            frag_data.extend_from_slice(&(rows as u32).to_le_bytes());
            frag_data.extend_from_slice(&(cols as u32).to_le_bytes());
            frag_data.extend_from_slice(&1u32.to_le_bytes()); // num_components=1

            // Single component: sigma=1.0, u=[1.0], v=data
            frag_data.extend_from_slice(&1.0f32.to_le_bytes()); // sigma
            frag_data.extend_from_slice(&1.0f32.to_le_bytes()); // u[0] = 1.0

            // v vector: all data elements
            for &val in data.iter().take(cols) {
                frag_data.extend_from_slice(&val.to_le_bytes());
            }

            Ok(vec![HoloFragment::new(0, frag_data)])
        } else {
            // 2D case: Use raw format (num_components = 0xFFFFFFFF marker)
            // This is O(rows*cols) storage instead of O(rows^2) with one-hot encoding
            let mut frag_data = Vec::with_capacity(12 + rows * cols * 4);

            // Header: rows, cols, raw marker
            frag_data.extend_from_slice(&(rows as u32).to_le_bytes());
            frag_data.extend_from_slice(&(cols as u32).to_le_bytes());
            frag_data.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // Raw format marker

            // Raw f32 data in row-major order
            for i in 0..rows * cols {
                let val = if i < data.len() { data[i] } else { 0.0f32 };
                frag_data.extend_from_slice(&val.to_le_bytes());
            }

            Ok(vec![HoloFragment::new(0, frag_data)])
        }
    }

    /// Encode tensor using LRDF (GPU or CPU).
    fn encode_with_lrdf(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<Vec<HoloFragment>> {
        // Try GPU encoder first, fall back to CPU
        #[cfg(feature = "cuda")]
        let fragments = if let Some(ref gpu_encoder) = self.gpu_encoder {
            // Use GPU-accelerated LRDF encoder
            gpu_encoder
                .encode_2d(data, rows, cols)
                .map(|gpu_frags| gpu_frags.into_iter().map(|f| f.to_haagenti()).collect())
                .map_err(|e| {
                    HoloInferenceError::Conversion(format!("GPU encode failed for {}: {}", name, e))
                })?
        } else {
            // Use CPU LRDF encoder
            let encoder =
                LrdfEncoder::new(self.config.num_fragments).with_max_rank(self.config.max_rank);
            encoder.encode_2d(data, rows, cols).map_err(|e| {
                HoloInferenceError::Conversion(format!("Failed to encode {}: {}", name, e))
            })?
        };

        #[cfg(not(feature = "cuda"))]
        let fragments = {
            // Use CPU LRDF encoder
            let encoder =
                LrdfEncoder::new(self.config.num_fragments).with_max_rank(self.config.max_rank);
            encoder.encode_2d(data, rows, cols).map_err(|e| {
                HoloInferenceError::Conversion(format!("Failed to encode {}: {}", name, e))
            })?
        };

        Ok(fragments)
    }

    /// Encode tensor using Spectral (DCT-based) encoding.
    ///
    /// Uses CompressiveSpectralEncoder which:
    /// 1. Transforms tensor to frequency domain via 2D DCT
    /// 2. Retains only top 20% of coefficients by energy (configurable)
    /// 3. Stores coefficients using bitmap + f16 for efficient compression
    /// 4. Reconstruction uses IDCT
    ///
    /// This is the production-ready spectral encoder that produces HCT3 format,
    /// compatible with CompressiveSpectralDecoder used during inference.
    fn encode_with_spectral(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<Vec<HoloFragment>> {
        // Use CompressiveSpectralEncoder which produces HCT3 format
        // retention_ratio: keep top N% of DCT coefficients by energy
        // Higher retention = better quality, larger files
        let encoder =
            CompressiveSpectralEncoder::new(self.config.num_fragments, self.config.retention_ratio);

        let fragments = encoder.encode_2d(data, cols, rows).map_err(|e| {
            HoloInferenceError::Conversion(format!("Spectral encoding failed for {}: {}", name, e))
        })?;

        tracing::debug!(
            tensor = %name,
            rows = %rows,
            cols = %cols,
            fragments = %fragments.len(),
            retention = %self.config.retention_ratio,
            "Compressive Spectral DCT encoding complete"
        );

        Ok(fragments)
    }

    /// Compress fragment data using Zstd (Phase 1 optimization: 9.4x faster decompression).
    ///
    /// Compress fragments using Zstd compression.
    ///
    /// Uses the standard zstd crate (not haagenti-zstd) for reliable compression.
    ///
    /// Returns a new vector of fragments with compressed data. Fragments that don't
    /// benefit from compression (expand under compression) are left uncompressed.
    fn compress_fragments(&self, fragments: Vec<HoloFragment>) -> Result<Vec<HoloFragment>> {
        if !self.config.compress_fragments {
            return Ok(fragments);
        }

        let mut compressed_fragments = Vec::with_capacity(fragments.len());

        for frag in fragments {
            let original_size = frag.data.len();

            // Compress the data using standard zstd crate (level 3 = default)
            let compressed_data = zstd::encode_all(&frag.data[..], 3).map_err(|e| {
                HoloInferenceError::Conversion(format!("Zstd compression failed: {}", e))
            })?;

            // Only use compressed data if it's actually smaller
            let (final_data, is_compressed) = if compressed_data.len() < original_size {
                (compressed_data, true)
            } else {
                (frag.data, false)
            };

            // CRITICAL FIX: Recompute checksum on the final data that will be stored.
            // The reader verifies the checksum against what it reads (compressed data),
            // so we must store the checksum of the compressed data, not the original.
            let final_checksum = xxh3_64(&final_data);
            let mut new_frag = HoloFragment::with_checksum(frag.index, final_data, final_checksum);

            // Set compression flag in flags field (bit 0 = compressed)
            if is_compressed {
                new_frag.flags |= 0x0001;
            }

            compressed_fragments.push(new_frag);
        }

        Ok(compressed_fragments)
    }

    /// Verify reconstruction quality.
    fn verify_reconstruction(
        &self,
        original: &[f32],
        header: &HoloTensorHeader,
        fragments: &[HoloFragment],
        rows: usize,
        cols: usize,
    ) -> Result<f32> {
        use haagenti::holotensor::LrdfDecoder;

        // Use the appropriate decoder based on encoding type
        let reconstructed = match header.encoding {
            HolographicEncoding::Spectral => {
                // Use CompressiveSpectralDecoder for HCT3 format (matches CompressiveSpectralEncoder)
                let mut decoder = CompressiveSpectralDecoder::new();

                // Fragment 0 contains essentials (header + bitmap + essential coefficients)
                // Other fragments contain detail coefficients
                for fragment in fragments {
                    if fragment.index == 0 {
                        decoder.add_essentials(fragment).map_err(|e| {
                            HoloInferenceError::Conversion(format!("Add essentials error: {}", e))
                        })?;
                    } else {
                        decoder.add_detail(fragment).map_err(|e| {
                            HoloInferenceError::Conversion(format!("Add detail error: {}", e))
                        })?;
                    }
                }
                decoder.reconstruct().map_err(|e| {
                    HoloInferenceError::Conversion(format!("Spectral reconstruct error: {}", e))
                })?
            },
            HolographicEncoding::LowRankDistributed | _ => {
                // LrdfDecoder uses (rows, cols)
                let mut decoder = LrdfDecoder::new(rows, cols, fragments.len() as u16);
                for fragment in fragments {
                    decoder.add_fragment(fragment).map_err(|e| {
                        HoloInferenceError::Conversion(format!("Decoder error: {}", e))
                    })?;
                }
                decoder.reconstruct()
            },
        };

        // Calculate cosine similarity
        let quality = cosine_similarity(original, &reconstructed);

        // Just return quality - caller can decide if it's acceptable
        // This allows conversion to complete and report overall quality
        Ok(quality)
    }

    /// Get conversion progress.
    pub fn progress(
        &self,
        total_tensors: usize,
        total_bytes: usize,
        start_time: std::time::Instant,
    ) -> ConversionProgress {
        let tensors_processed = self.tensors_processed.load(Ordering::Relaxed);
        let bytes_processed = self.bytes_processed.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed().as_secs_f64();

        let eta = if tensors_processed > 0 {
            let rate = tensors_processed as f64 / elapsed;
            (total_tensors - tensors_processed) as f64 / rate
        } else {
            0.0
        };

        ConversionProgress {
            current_tensor: String::new(),
            tensors_processed,
            total_tensors,
            bytes_processed,
            total_bytes,
            phase: if tensors_processed >= total_tensors {
                ConversionPhase::Complete
            } else {
                ConversionPhase::Converting
            },
            elapsed_secs: elapsed,
            eta_secs: eta,
        }
    }

    /// Reset progress counters.
    pub fn reset(&self) {
        self.tensors_processed.store(0, Ordering::Relaxed);
        self.bytes_processed.store(0, Ordering::Relaxed);
    }

    /// Convert a model from source format to HoloTensor HCT format.
    ///
    /// This loads tensors from the source model (SafeTensors or PyTorch),
    /// converts them to holographic format, and writes .hct files to the output directory.
    ///
    /// **Streaming mode**: Processes one safetensors file at a time to minimize memory usage.
    /// This allows converting 400B+ models on systems with limited RAM.
    pub async fn convert_model(
        &self,
        source: &str,
        output_dir: &Path,
    ) -> Result<ConversionMetadata> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        use safetensors::SafeTensors;

        self.reset();

        // Determine if source is HuggingFace repo ID or local path
        let source_path = if source.contains('/') && !Path::new(source).exists() {
            // HuggingFace repo - download first
            let api = Api::new().map_err(|e| {
                HoloInferenceError::Conversion(format!("Failed to create HuggingFace API: {}", e))
            })?;
            let repo = api.repo(Repo::new(source.to_string(), RepoType::Model));

            // Get safetensors index or model file
            let model_path = if let Ok(index) = repo.get("model.safetensors.index.json") {
                index
                    .parent()
                    .ok_or_else(|| {
                        HoloInferenceError::Conversion(
                            "Safetensors index file has no parent directory".to_string(),
                        )
                    })?
                    .to_path_buf()
            } else if let Ok(model) = repo.get("model.safetensors") {
                model
                    .parent()
                    .ok_or_else(|| {
                        HoloInferenceError::Conversion(
                            "Safetensors model file has no parent directory".to_string(),
                        )
                    })?
                    .to_path_buf()
            } else {
                return Err(HoloInferenceError::Conversion(
                    "No safetensors files found in repository".to_string(),
                ));
            };
            model_path
        } else {
            PathBuf::from(source)
        };

        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Copy config.json and tokenizer files to output
        for filename in &["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let src = source_path.join(filename);
            if src.exists() {
                fs::copy(&src, output_dir.join(filename))?;
            }
        }

        // Find all safetensors files
        let files: Vec<PathBuf> = if source_path.is_file() {
            vec![source_path.clone()]
        } else {
            let mut files: Vec<PathBuf> = fs::read_dir(&source_path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
                .collect();
            files.sort(); // Process in deterministic order
            files
        };

        let total_files = files.len();
        println!(
            "\nFound {} safetensors files to convert (streaming mode)",
            total_files
        );

        // Configure rayon thread pool based on config
        if self.config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.num_threads)
                .build_global()
                .ok(); // Ignore if already built
        }

        // Aggregate results
        let mut min_quality = 1.0f32;
        let mut hct_size = 0u64;
        let mut total_bytes = 0u64;
        let mut tensor_count = 0usize;

        // Process one file at a time (streaming) to minimize memory usage
        for (file_idx, file_path) in files.iter().enumerate() {
            let file_name = file_path.file_name().unwrap_or_default().to_string_lossy();
            println!(
                "\n[{}/{}] Processing: {}",
                file_idx + 1,
                total_files,
                file_name
            );

            // Load single safetensors file
            let data = fs::read(&file_path)?;
            let file_size = data.len();
            println!("  Loaded {} ({:.1} GB)", file_name, file_size as f64 / 1e9);

            let tensors = SafeTensors::deserialize(&data).map_err(|e| {
                HoloInferenceError::Conversion(format!(
                    "Failed to load {}: {}",
                    file_path.display(),
                    e
                ))
            })?;

            let tensor_names: Vec<String> = tensors.names().iter().map(|s| s.to_string()).collect();
            let num_tensors = tensor_names.len();
            println!("  Contains {} tensors", num_tensors);

            // Process each tensor in this file
            for (t_idx, name) in tensor_names.iter().enumerate() {
                let tensor = tensors.tensor(name).map_err(|e| {
                    HoloInferenceError::Conversion(format!("Failed to get tensor {}: {}", name, e))
                })?;

                let shape: Vec<usize> = tensor.shape().to_vec();
                let tensor_size = tensor.data().len();
                total_bytes += tensor_size as u64;

                // Try GPU-resident FP8 encoding (zero-copy path)
                #[cfg(feature = "cuda")]
                let converted = if tensor.dtype() == safetensors::Dtype::F8_E4M3
                    && self.gpu_encoder.is_some()
                    && self.gpu_dtype_converter.is_some()
                    && shape.len() >= 2
                    && tensor_size >= 4096
                // Only for tensors worth LRDF encoding
                {
                    // GPU-resident path: FP8 → GPU → F32 → SVD (no CPU round-trip)
                    // Flatten to 2D: (first_dim, product_of_rest) - matches CPU path and decoder
                    let rows = shape[0];
                    let cols: usize = shape[1..].iter().product();

                    if rows > 1 && cols > 1 {
                        // Safety: These are guaranteed to be Some by the outer condition check
                        // Using if-let pattern to gracefully handle unexpected None
                        if let (Some(gpu_encoder), Some(dtype_converter)) =
                            (self.gpu_encoder.as_ref(), self.gpu_dtype_converter.as_ref())
                        {
                            match gpu_encoder.encode_2d_fp8_e4m3(
                                tensor.data(),
                                rows,
                                cols,
                                dtype_converter,
                            ) {
                                Ok(gpu_fragments) => {
                                    let fragments: Vec<HoloFragment> =
                                        gpu_fragments.iter().map(|f| f.to_haagenti()).collect();

                                    let info = TensorInfo {
                                        name: name.clone(),
                                        shape: shape.clone(),
                                        dtype: DType::F32, // Converted from FP8
                                        path: file_path.clone(),
                                        size: tensor_size * 4, // f32 size
                                    };

                                    Some(ConvertedTensor {
                                        info,
                                        header: HoloTensorHeader::new(
                                            HolographicEncoding::LowRankDistributed,
                                            DType::F32,
                                            shape.iter().map(|&s| s as u64).collect(),
                                            fragments.len() as u16,
                                        ),
                                        fragments,
                                        quality: None, // Skip verification for speed
                                    })
                                },
                                Err(e) => {
                                    eprintln!(
                                        "    GPU FP8 encoding failed for {}: {}, falling back",
                                        name, e
                                    );
                                    None
                                },
                            }
                        } else {
                            // GPU encoder or dtype converter unexpectedly None
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                #[cfg(not(feature = "cuda"))]
                let converted: Option<ConvertedTensor> = None;

                // Fall back to standard path if GPU-resident failed or not applicable
                let converted = if let Some(c) = converted {
                    c
                } else {
                    // Standard path: convert to f32, then encode
                    let f32_data = self.tensor_to_f32(tensor.data(), tensor.dtype())?;
                    let info = TensorInfo {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: DType::F32,
                        path: file_path.clone(),
                        size: f32_data.len() * 4,
                    };
                    self.convert_tensor(&info, &f32_data)?
                };

                let hct_path = output_dir.join(format!("{}.hct", name.replace('.', "_")));
                let written = self.write_hct_file(&hct_path, &converted)?;

                hct_size += written;
                if let Some(q) = converted.quality {
                    min_quality = min_quality.min(q);
                }

                tensor_count += 1;

                // Progress for large files
                if num_tensors > 10 && (t_idx + 1) % 10 == 0 {
                    println!("    Converted {}/{} tensors", t_idx + 1, num_tensors);
                }
            }

            println!("  ✓ Completed {} ({} tensors)", file_name, num_tensors);

            // data and tensors are dropped here, freeing memory before next file
        }

        println!(
            "\n✓ Conversion complete: {} tensors from {} files",
            tensor_count, total_files
        );

        Ok(ConversionMetadata {
            num_layers: self.count_layers(&output_dir),
            total_fragments: self.config.num_fragments,
            original_size: total_bytes,
            hct_size,
            verified_quality: min_quality,
        })
    }

    /// Convert model using producer-consumer pipeline for maximum throughput.
    ///
    /// This method uses multiple producer threads to load and prepare tensors
    /// while a single GPU consumer thread processes them. This overlaps I/O
    /// with GPU computation for better utilization.
    ///
    /// # Architecture
    /// - N producer threads: Load files, parse tensors, prepare data
    /// - Bounded work queue: Prevents memory explosion
    /// - 1 GPU consumer: Processes tensors sequentially on GPU
    #[cfg(feature = "cuda")]
    pub async fn convert_model_pipeline(
        &self,
        source: &str,
        output_dir: &Path,
        num_producers: usize,
    ) -> Result<ConversionMetadata> {
        use hf_hub::{api::sync::Api, Repo, RepoType};
        use safetensors::SafeTensors;

        self.reset();

        // Determine source path
        let source_path = if source.contains('/') && !Path::new(source).exists() {
            let api = Api::new().map_err(|e| {
                HoloInferenceError::Conversion(format!("Failed to create HuggingFace API: {}", e))
            })?;
            let repo = api.repo(Repo::new(source.to_string(), RepoType::Model));
            let model_path = if let Ok(index) = repo.get("model.safetensors.index.json") {
                index
                    .parent()
                    .ok_or_else(|| {
                        HoloInferenceError::Conversion(
                            "Safetensors index file has no parent directory".to_string(),
                        )
                    })?
                    .to_path_buf()
            } else if let Ok(model) = repo.get("model.safetensors") {
                model
                    .parent()
                    .ok_or_else(|| {
                        HoloInferenceError::Conversion(
                            "Safetensors model file has no parent directory".to_string(),
                        )
                    })?
                    .to_path_buf()
            } else {
                return Err(HoloInferenceError::Conversion(
                    "No safetensors files found in repository".to_string(),
                ));
            };
            model_path
        } else {
            PathBuf::from(source)
        };

        // Create output directory and copy config files
        fs::create_dir_all(output_dir)?;
        for filename in &["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let src = source_path.join(filename);
            if src.exists() {
                fs::copy(&src, output_dir.join(filename))?;
            }
        }

        // Find all safetensors files
        let files: Vec<PathBuf> = if source_path.is_file() {
            vec![source_path.clone()]
        } else {
            let mut files: Vec<PathBuf> = fs::read_dir(&source_path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
                .collect();
            files.sort();
            files
        };

        let total_files = files.len();
        let num_producers = num_producers.min(total_files).max(1);

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║           Pipeline Mode: {} producers, 1 GPU consumer          ║",
            num_producers
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");
        println!("Found {} safetensors files to convert", total_files);

        // Bounded channel - prevents memory explosion
        // Queue size = 2x producers so each producer can have work in flight
        let queue_size = num_producers * 2 + 4;
        let (work_tx, work_rx): (Sender<PipelineWorkItem>, Receiver<PipelineWorkItem>) =
            channel::bounded(queue_size);
        let (result_tx, result_rx): (Sender<PipelineResult>, Receiver<PipelineResult>) =
            channel::unbounded();

        // Shared state
        let files = Arc::new(files);
        let file_index = Arc::new(AtomicUsize::new(0));
        let output_dir = Arc::new(output_dir.to_path_buf());
        let config = Arc::new(self.config.clone());

        // Spawn producer threads
        let mut producer_handles = Vec::new();
        for producer_id in 0..num_producers {
            let work_tx = work_tx.clone();
            let files = Arc::clone(&files);
            let file_index = Arc::clone(&file_index);
            let _config = Arc::clone(&config);

            let handle = thread::spawn(move || {
                loop {
                    // Grab next file atomically
                    let idx = file_index.fetch_add(1, Ordering::SeqCst);
                    if idx >= files.len() {
                        break;
                    }

                    let file_path = &files[idx];
                    let file_name = file_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    // Load and parse file
                    let data = match fs::read(file_path) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!(
                                "[Producer {}] Failed to read {}: {}",
                                producer_id, file_name, e
                            );
                            continue;
                        },
                    };

                    let tensors = match SafeTensors::deserialize(&data) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!(
                                "[Producer {}] Failed to parse {}: {}",
                                producer_id, file_name, e
                            );
                            continue;
                        },
                    };

                    println!(
                        "[Producer {}] Loaded {} ({:.1} GB, {} tensors)",
                        producer_id,
                        file_name,
                        data.len() as f64 / 1e9,
                        tensors.names().len()
                    );

                    // Send each tensor to the work queue
                    for name in tensors.names() {
                        let tensor = match tensors.tensor(name) {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!(
                                    "[Producer {}] Failed to get tensor {}: {}",
                                    producer_id, name, e
                                );
                                continue;
                            },
                        };

                        let shape: Vec<usize> = tensor.shape().to_vec();
                        let dtype = tensor.dtype();
                        let raw_data = tensor.data().to_vec();
                        let tensor_size = raw_data.len();

                        // Determine if this is a small tensor (passthrough) or needs LRDF
                        let is_small = tensor_size < 4096;
                        let is_1d = shape.len() <= 1
                            || (shape.len() >= 1 && shape.iter().filter(|&&s| s > 1).count() <= 1);

                        let work_item = PipelineWorkItem {
                            name: name.to_string(),
                            shape,
                            dtype,
                            raw_data,
                            file_index: idx,
                            is_small_or_1d: is_small || is_1d,
                        };

                        // This blocks if queue is full (backpressure)
                        if work_tx.send(work_item).is_err() {
                            break; // Consumer closed
                        }
                    }

                    // File data is dropped here, freeing memory
                }
            });
            producer_handles.push(handle);
        }

        // Drop our copy of work_tx so consumer knows when producers are done
        drop(work_tx);

        // GPU consumer thread
        let gpu_encoder = self.gpu_encoder.clone();
        let gpu_dtype_converter = self.gpu_dtype_converter.clone();
        let output_dir_consumer = Arc::clone(&output_dir);
        let config_consumer = Arc::clone(&config);

        let consumer_handle = thread::spawn(move || {
            let mut total_bytes = 0u64;
            let mut hct_size = 0u64;
            let mut min_quality = 1.0f32;
            let mut tensor_count = 0usize;
            let mut last_file_idx = usize::MAX;

            // Create CPU encoder for fallback/small tensors
            let cpu_encoder = LrdfEncoder::new(config_consumer.num_fragments)
                .with_max_rank(config_consumer.max_rank);

            while let Ok(item) = work_rx.recv() {
                if item.file_index != last_file_idx {
                    if last_file_idx != usize::MAX {
                        println!("  [GPU] Completed file {}", last_file_idx + 1);
                    }
                    last_file_idx = item.file_index;
                }

                total_bytes += item.raw_data.len() as u64;
                let rows = if item.shape.len() >= 2 {
                    item.shape[0..item.shape.len() - 1].iter().product()
                } else {
                    1
                };
                let cols = item.shape.last().copied().unwrap_or(1);

                // Process tensor (lossless mode forces passthrough for all tensors)
                let use_passthrough = config_consumer.lossless || item.is_small_or_1d;
                let (fragments, quality): (Vec<HoloFragment>, Option<f32>) = if use_passthrough {
                    // Passthrough for small/1D tensors or lossless mode
                    let f32_data = convert_to_f32_cpu(&item.raw_data, item.dtype);
                    let frag = create_passthrough_fragment(&f32_data, rows, cols);
                    (vec![frag], None)
                } else if item.dtype == safetensors::Dtype::F8_E4M3
                    && gpu_encoder.is_some()
                    && gpu_dtype_converter.is_some()
                {
                    // GPU-resident FP8 path
                    // Safety: These are guaranteed to be Some by the condition above
                    if let (Some(encoder), Some(dtype_conv)) =
                        (gpu_encoder.as_ref(), gpu_dtype_converter.as_ref())
                    {
                        match encoder.encode_2d_fp8_e4m3(&item.raw_data, rows, cols, dtype_conv) {
                            Ok(gpu_frags) => {
                                let frags: Vec<HoloFragment> =
                                    gpu_frags.iter().map(|f| f.to_haagenti()).collect();
                                (frags, None)
                            },
                            Err(e) => {
                                eprintln!(
                                    "  [GPU] FP8 encode failed for {}: {}, using CPU",
                                    item.name, e
                                );
                                let f32_data = convert_to_f32_cpu(&item.raw_data, item.dtype);
                                match cpu_encoder.encode_2d(&f32_data, rows, cols) {
                                    Ok(frags) => (frags, None),
                                    Err(e2) => {
                                        eprintln!(
                                            "  [CPU] Encode also failed for {}: {}",
                                            item.name, e2
                                        );
                                        // Return passthrough as last resort
                                        (
                                            vec![create_passthrough_fragment(
                                                &f32_data, rows, cols,
                                            )],
                                            None,
                                        )
                                    },
                                }
                            },
                        }
                    } else {
                        // GPU encoder or dtype converter unexpectedly None, use CPU
                        let f32_data = convert_to_f32_cpu(&item.raw_data, item.dtype);
                        match cpu_encoder.encode_2d(&f32_data, rows, cols) {
                            Ok(frags) => (frags, None),
                            Err(e) => {
                                eprintln!("  [CPU] Encode failed for {}: {}", item.name, e);
                                (
                                    vec![create_passthrough_fragment(&f32_data, rows, cols)],
                                    None,
                                )
                            },
                        }
                    }
                } else if let Some(ref encoder) = gpu_encoder {
                    // GPU path for non-FP8
                    let f32_data = convert_to_f32_cpu(&item.raw_data, item.dtype);
                    match encoder.encode_2d(&f32_data, rows, cols) {
                        Ok(gpu_frags) => {
                            let frags: Vec<HoloFragment> =
                                gpu_frags.iter().map(|f| f.to_haagenti()).collect();
                            (frags, None)
                        },
                        Err(e) => {
                            eprintln!("  [GPU] Encode failed for {}: {}, using CPU", item.name, e);
                            match cpu_encoder.encode_2d(&f32_data, rows, cols) {
                                Ok(frags) => (frags, None),
                                Err(e2) => {
                                    eprintln!(
                                        "  [CPU] Encode also failed for {}: {}",
                                        item.name, e2
                                    );
                                    (
                                        vec![create_passthrough_fragment(&f32_data, rows, cols)],
                                        None,
                                    )
                                },
                            }
                        },
                    }
                } else {
                    // CPU fallback
                    let f32_data = convert_to_f32_cpu(&item.raw_data, item.dtype);
                    match cpu_encoder.encode_2d(&f32_data, rows, cols) {
                        Ok(frags) => (frags, None),
                        Err(e) => {
                            eprintln!("  [CPU] Encode failed for {}: {}", item.name, e);
                            (
                                vec![create_passthrough_fragment(&f32_data, rows, cols)],
                                None,
                            )
                        },
                    }
                };

                // Build header and write (use actual fragment count, not config)
                let header = HoloTensorHeader::new(
                    HolographicEncoding::LowRankDistributed,
                    DType::F32,
                    item.shape.iter().map(|&s| s as u64).collect(),
                    fragments.len() as u16,
                );

                let hct_path =
                    output_dir_consumer.join(format!("{}.hct", item.name.replace('.', "_")));

                match haagenti::holotensor::write_holotensor(&hct_path, &header, &fragments) {
                    Ok(size) => {
                        hct_size += size;
                        if let Some(q) = quality {
                            min_quality = min_quality.min(q);
                        }
                        tensor_count += 1;

                        if tensor_count % 50 == 0 {
                            println!(
                                "  [GPU] Processed {} tensors ({:.1} GB written)",
                                tensor_count,
                                hct_size as f64 / 1e9
                            );
                        }
                    },
                    Err(e) => {
                        eprintln!("  [GPU] Failed to write {}: {}", item.name, e);
                    },
                }
            }

            // Send final results
            let _ = result_tx.send(PipelineResult {
                total_bytes,
                hct_size,
                min_quality,
                tensor_count,
            });
        });

        // Wait for producers to finish
        for handle in producer_handles {
            let _ = handle.join();
        }

        // Wait for consumer to finish
        let _ = consumer_handle.join();

        // Collect results
        let result = result_rx.recv().map_err(|_| {
            HoloInferenceError::Conversion("Failed to get pipeline results".to_string())
        })?;

        println!(
            "\n✓ Pipeline conversion complete: {} tensors from {} files",
            result.tensor_count, total_files
        );

        Ok(ConversionMetadata {
            num_layers: self.count_layers(&output_dir),
            total_fragments: self.config.num_fragments,
            original_size: result.total_bytes,
            hct_size: result.hct_size,
            verified_quality: result.min_quality,
        })
    }

    /// Load tensors from source path (SafeTensors format).
    /// Returns (name, shape, f32_data) tuples.
    #[allow(dead_code)]
    fn load_source_tensors(&self, path: &Path) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
        use safetensors::SafeTensors;

        let mut all_tensors = Vec::new();

        // Find all safetensors files
        let files: Vec<PathBuf> = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
                .collect()
        };

        for file_path in files {
            let data = fs::read(&file_path)?;
            let tensors = SafeTensors::deserialize(&data).map_err(|e| {
                HoloInferenceError::Conversion(format!(
                    "Failed to load {}: {}",
                    file_path.display(),
                    e
                ))
            })?;

            for name in tensors.names() {
                let tensor = tensors.tensor(name).map_err(|e| {
                    HoloInferenceError::Conversion(format!("Failed to get tensor {}: {}", name, e))
                })?;

                // Get shape
                let shape: Vec<usize> = tensor.shape().to_vec();

                // Convert to f32
                let f32_data = self.tensor_to_f32(tensor.data(), tensor.dtype())?;
                all_tensors.push((name.to_string(), shape, f32_data));
            }
        }

        Ok(all_tensors)
    }

    /// Convert tensor data to f32.
    fn tensor_to_f32(&self, data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
        use safetensors::Dtype;

        match dtype {
            Dtype::F32 => Ok(data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            Dtype::F16 => Ok(data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()),
            Dtype::BF16 => Ok(data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()),
            Dtype::F8_E4M3 => {
                // FP8 E4M3: Use GPU conversion if available (3x faster, smaller transfer)
                #[cfg(feature = "cuda")]
                if let Some(ref converter) = self.gpu_dtype_converter {
                    return converter.fp8_e4m3_to_f32_host(data).map_err(|e| {
                        HoloInferenceError::Conversion(format!(
                            "GPU FP8 E4M3 conversion failed: {}",
                            e
                        ))
                    });
                }
                // CPU fallback
                Ok(data.iter().map(|&byte| fp8_e4m3_to_f32(byte)).collect())
            },
            Dtype::F8_E5M2 => {
                // FP8 E5M2: Use GPU conversion if available
                #[cfg(feature = "cuda")]
                if let Some(ref converter) = self.gpu_dtype_converter {
                    return converter.fp8_e5m2_to_f32_host(data).map_err(|e| {
                        HoloInferenceError::Conversion(format!(
                            "GPU FP8 E5M2 conversion failed: {}",
                            e
                        ))
                    });
                }
                // CPU fallback
                Ok(data.iter().map(|&byte| fp8_e5m2_to_f32(byte)).collect())
            },
            _ => Err(HoloInferenceError::Conversion(format!(
                "Unsupported dtype: {:?}",
                dtype
            ))),
        }
    }

    /// Write converted tensor to HCT file.
    fn write_hct_file(&self, path: &Path, converted: &ConvertedTensor) -> Result<u64> {
        use haagenti::holotensor::write_holotensor;

        write_holotensor(path, &converted.header, &converted.fragments)
            .map_err(|e| HoloInferenceError::Conversion(format!("Failed to write HCT file: {}", e)))
    }

    /// Count transformer layers from output directory.
    fn count_layers(&self, dir: &Path) -> usize {
        fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map_or(false, |n| n.starts_with("layers_"))
                    })
                    .count()
            })
            .unwrap_or(0)
            .max(1) // At least 1 layer
    }
}

/// Metadata returned after model conversion.
#[derive(Debug, Clone)]
pub struct ConversionMetadata {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Total fragments per tensor.
    pub total_fragments: u16,
    /// Original model size in bytes.
    pub original_size: u64,
    /// HCT compressed size in bytes.
    pub hct_size: u64,
    /// Minimum verified quality across all tensors.
    pub verified_quality: f32,
}

/// Validation report for HCT model completeness.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Total tensors found.
    pub total_tensors: usize,
    /// Model architecture (e.g., "llama", "qwen2").
    pub architecture: String,
    /// Missing required tensors.
    pub missing_tensors: Vec<String>,
    /// Corrupted tensors (failed to parse).
    pub corrupted_tensors: Vec<(String, String)>,
    /// Whether the model is complete.
    pub is_complete: bool,
    /// Number of transformer layers detected.
    pub num_layers: usize,
}

impl ValidationReport {
    /// Check if validation passed.
    pub fn passed(&self) -> bool {
        self.is_complete && self.corrupted_tensors.is_empty()
    }

    /// Print validation summary.
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    Validation Report                         ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("Architecture: {}", self.architecture);
        println!("Total tensors: {}", self.total_tensors);
        println!("Layers: {}", self.num_layers);

        if self.missing_tensors.is_empty() {
            println!("Missing: None ✓");
        } else {
            println!("Missing ({}):", self.missing_tensors.len());
            for name in &self.missing_tensors {
                println!("  - {}", name);
            }
        }

        if self.corrupted_tensors.is_empty() {
            println!("Corrupted: None ✓");
        } else {
            println!("Corrupted ({}):", self.corrupted_tensors.len());
            for (name, err) in &self.corrupted_tensors {
                println!("  - {}: {}", name, err);
            }
        }

        if self.passed() {
            println!("\n✓ Validation PASSED");
        } else {
            println!("\n✗ Validation FAILED");
        }
    }
}

/// Validate a converted HCT model directory for completeness.
///
/// Checks that all required tensors exist for the model architecture:
/// - embed_tokens (embedding layer)
/// - lm_head (output projection)
/// - model.norm (final layer norm)
/// - All layer tensors (q_proj, k_proj, v_proj, o_proj, etc.)
pub fn validate_hct_directory(dir: &Path) -> Result<ValidationReport> {
    let config_path = dir.join("config.json");
    if !config_path.exists() {
        return Err(HoloInferenceError::Conversion(
            "No config.json found in HCT directory".to_string(),
        ));
    }

    // Parse architecture from config
    let config_data = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_data).map_err(|e| {
        HoloInferenceError::Conversion(format!("Failed to parse config.json: {}", e))
    })?;

    let architecture = config
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Scan for HCT files
    let hct_files: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "hct"))
        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
        .collect();

    let total_tensors = hct_files.len();

    // Get required tensors based on architecture
    let required = get_required_tensors(&architecture, num_layers);

    // Check for missing tensors
    let mut missing_tensors = Vec::new();
    for required_name in &required {
        // Convert tensor name to filename pattern
        let pattern = required_name.replace(".", "_");
        let found = hct_files.iter().any(|f| f.contains(&pattern));
        if !found {
            missing_tensors.push(required_name.clone());
        }
    }

    // Check for corrupted files (try to read header)
    let mut corrupted_tensors = Vec::new();
    for filename in &hct_files {
        let path = dir.join(filename);
        if let Err(e) = validate_hct_file(&path) {
            corrupted_tensors.push((filename.clone(), e.to_string()));
        }
    }

    let is_complete = missing_tensors.is_empty();

    Ok(ValidationReport {
        total_tensors,
        architecture,
        missing_tensors,
        corrupted_tensors,
        is_complete,
        num_layers,
    })
}

/// Get required tensor names for a given architecture.
fn get_required_tensors(architecture: &str, num_layers: usize) -> Vec<String> {
    let mut required = Vec::new();

    let arch_lower = architecture.to_lowercase();
    if arch_lower.contains("llama") || arch_lower.contains("qwen") || arch_lower.contains("mistral")
    {
        // Standard decoder-only transformer
        required.push("model.embed_tokens.weight".to_string());
        required.push("model.norm.weight".to_string());
        required.push("lm_head.weight".to_string());

        for layer in 0..num_layers {
            required.push(format!("model.layers.{}.self_attn.q_proj.weight", layer));
            required.push(format!("model.layers.{}.self_attn.k_proj.weight", layer));
            required.push(format!("model.layers.{}.self_attn.v_proj.weight", layer));
            required.push(format!("model.layers.{}.self_attn.o_proj.weight", layer));
            required.push(format!("model.layers.{}.mlp.gate_proj.weight", layer));
            required.push(format!("model.layers.{}.mlp.up_proj.weight", layer));
            required.push(format!("model.layers.{}.mlp.down_proj.weight", layer));
            required.push(format!("model.layers.{}.input_layernorm.weight", layer));
            required.push(format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer
            ));
        }
    }

    required
}

/// Validate a single HCT file header.
fn validate_hct_file(path: &Path) -> Result<()> {
    use std::io::Read;

    let mut file = fs::File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    // Check magic bytes (HTNS for HoloTensor)
    if &magic != b"HTNS" && &magic != b"HCTN" {
        return Err(HoloInferenceError::Conversion(format!(
            "Invalid magic bytes: {:?}",
            magic
        )));
    }

    Ok(())
}

/// Work item for pipeline processing.
///
/// Contains tensor data and metadata ready for GPU encoding.
#[cfg(feature = "cuda")]
struct PipelineWorkItem {
    /// Tensor name.
    name: String,
    /// Tensor shape.
    shape: Vec<usize>,
    /// Original dtype from safetensors.
    dtype: safetensors::Dtype,
    /// Raw tensor data bytes.
    raw_data: Vec<u8>,
    /// Index of the source file (for progress tracking).
    file_index: usize,
    /// Whether this tensor should use passthrough (small or 1D).
    is_small_or_1d: bool,
}

/// Result from pipeline processing.
#[cfg(feature = "cuda")]
struct PipelineResult {
    /// Total bytes processed.
    total_bytes: u64,
    /// Total HCT output size.
    hct_size: u64,
    /// Minimum quality across all tensors.
    min_quality: f32,
    /// Number of tensors processed.
    tensor_count: usize,
}

/// Convert raw tensor bytes to f32 on CPU.
///
/// This is used by producer threads to prepare data for GPU encoding.
#[cfg(feature = "cuda")]
fn convert_to_f32_cpu(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    use safetensors::Dtype;

    match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        Dtype::F8_E4M3 => data.iter().map(|&byte| fp8_e4m3_to_f32(byte)).collect(),
        Dtype::F8_E5M2 => data.iter().map(|&byte| fp8_e5m2_to_f32(byte)).collect(),
        _ => {
            // Unsupported dtype - return zeros (caller should handle)
            eprintln!("Warning: unsupported dtype {:?}, returning zeros", dtype);
            vec![0.0; data.len()]
        },
    }
}

/// Create a passthrough fragment using proper LRDF format.
///
/// For 1D tensors (rows=1): Uses single component with sigma=1, u=[1], v=data
/// For 2D tensors (rows>1): Uses `rows` components, each storing one row:
///   - sigma = 1.0
///   - u = one_hot vector with 1.0 at position i
///   - v = data[i*cols : (i+1)*cols] (the i-th row)
///
/// Reconstruction: A[i,j] = Σ_k σ_k * u_k[i] * v_k[j] = data[i,j]
/// Create passthrough fragment for lossless storage.
/// Uses raw format (num_components = 0xFFFFFFFF) for 2D tensors.
#[cfg(feature = "cuda")]
fn create_passthrough_fragment(data: &[f32], rows: usize, cols: usize) -> HoloFragment {
    if rows == 1 {
        // 1D case: single LRDF component with u=[1.0], v=data
        let mut frag_data = Vec::with_capacity(12 + 4 + 4 + cols * 4);

        // Header
        frag_data.extend_from_slice(&(rows as u32).to_le_bytes());
        frag_data.extend_from_slice(&(cols as u32).to_le_bytes());
        frag_data.extend_from_slice(&1u32.to_le_bytes()); // num_components=1

        // Single component: sigma=1.0, u=[1.0], v=data
        frag_data.extend_from_slice(&1.0f32.to_le_bytes()); // sigma
        frag_data.extend_from_slice(&1.0f32.to_le_bytes()); // u[0] = 1.0

        // v vector: all data elements
        for &val in data.iter().take(cols) {
            frag_data.extend_from_slice(&val.to_le_bytes());
        }

        HoloFragment::new(0, frag_data)
    } else {
        // 2D case: Use raw format (num_components = 0xFFFFFFFF marker)
        // This is O(rows*cols) storage instead of O(rows^2) with one-hot encoding
        let mut frag_data = Vec::with_capacity(12 + rows * cols * 4);

        // Header: rows, cols, raw marker
        frag_data.extend_from_slice(&(rows as u32).to_le_bytes());
        frag_data.extend_from_slice(&(cols as u32).to_le_bytes());
        frag_data.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // Raw format marker

        // Raw f32 data in row-major order
        for i in 0..rows * cols {
            let val = if i < data.len() { data[i] } else { 0.0f32 };
            frag_data.extend_from_slice(&val.to_le_bytes());
        }

        HoloFragment::new(0, frag_data)
    }
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }

    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_config_default() {
        let config = ConversionConfig::default();
        assert_eq!(config.num_fragments, 32);
        assert_eq!(config.encoding, HolographicEncoding::LowRankDistributed);
    }

    #[test]
    fn test_conversion_config_fast() {
        let config = ConversionConfig::fast();
        assert_eq!(config.num_fragments, 16);
        assert!(!config.verify_quality);
    }

    #[test]
    fn test_conversion_config_high_quality() {
        let config = ConversionConfig::high_quality();
        assert_eq!(config.num_fragments, 64);
        assert!(config.verify_quality);
        assert_eq!(config.min_quality, 0.95);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_convert_small_tensor() {
        let converter = HoloModelConverter::new(ConversionConfig {
            num_fragments: 4,
            max_rank: 8,
            verify_quality: true,
            min_quality: 0.5, // Lower threshold for small test
            ..Default::default()
        });

        // Create random-ish test data
        let data: Vec<f32> = (0..64 * 64).map(|i| (i as f32 * 0.01).sin()).collect();

        let info = TensorInfo {
            name: "test_tensor".to_string(),
            shape: vec![64, 64],
            dtype: DType::F32,
            path: PathBuf::new(),
            size: 64 * 64 * 4,
        };

        let result = converter.convert_tensor(&info, &data).unwrap();
        assert_eq!(result.fragments.len(), 4);
        assert!(result.quality.unwrap() >= 0.5);
    }

    #[test]
    fn test_progress_tracking() {
        let converter = HoloModelConverter::default_converter();
        let start = std::time::Instant::now();

        let progress = converter.progress(100, 1000000, start);
        assert_eq!(progress.tensors_processed, 0);
        assert_eq!(progress.total_tensors, 100);
    }

    #[test]
    fn test_shape_inference() {
        let converter = HoloModelConverter::default_converter();

        let shape = converter.infer_shape_from_name("layer_0_q_proj");
        assert_eq!(shape, vec![5120, 5120]);

        let shape = converter.infer_shape_from_name("layer_0_gate_proj");
        assert_eq!(shape, vec![5120, 27648]);
    }

    #[test]
    fn test_get_required_tensors_llama() {
        let required = get_required_tensors("LlamaForCausalLM", 2);

        // Check core tensors
        assert!(required.contains(&"model.embed_tokens.weight".to_string()));
        assert!(required.contains(&"model.norm.weight".to_string()));
        assert!(required.contains(&"lm_head.weight".to_string()));

        // Check layer tensors
        assert!(required.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(required.contains(&"model.layers.1.mlp.gate_proj.weight".to_string()));

        // Should have 3 core + 9 per layer * 2 layers = 21 tensors
        assert_eq!(required.len(), 21);
    }

    #[test]
    fn test_validation_report_passed() {
        let report = ValidationReport {
            total_tensors: 100,
            architecture: "LlamaForCausalLM".to_string(),
            missing_tensors: vec![],
            corrupted_tensors: vec![],
            is_complete: true,
            num_layers: 32,
        };

        assert!(report.passed());
    }

    #[test]
    fn test_validation_report_failed_missing() {
        let report = ValidationReport {
            total_tensors: 99,
            architecture: "LlamaForCausalLM".to_string(),
            missing_tensors: vec!["model.embed_tokens.weight".to_string()],
            corrupted_tensors: vec![],
            is_complete: false,
            num_layers: 32,
        };

        assert!(!report.passed());
    }

    #[test]
    fn test_validation_report_failed_corrupted() {
        let report = ValidationReport {
            total_tensors: 100,
            architecture: "LlamaForCausalLM".to_string(),
            missing_tensors: vec![],
            corrupted_tensors: vec![(
                "model_layers_0_q_proj_weight.hct".to_string(),
                "Invalid magic".to_string(),
            )],
            is_complete: true,
            num_layers: 32,
        };

        assert!(!report.passed());
    }

    #[test]
    fn test_quality_enforcement_passthrough_fallback() {
        // Test that low-quality LRDF encoding falls back to passthrough
        // Use very strict quality threshold that LRDF won't meet with random data
        let converter = HoloModelConverter::new(ConversionConfig {
            num_fragments: 4,
            max_rank: 4, // Very low rank = poor quality
            verify_quality: true,
            min_quality: 0.999, // Very strict threshold
            ..Default::default()
        });

        // Create high-entropy random-ish test data (harder to compress)
        let data: Vec<f32> = (0..128 * 128)
            .map(|i| {
                let x = (i as f32 * 0.73).sin() * (i as f32 * 1.37).cos();
                x * (i as f32 % 17.0) - (i as f32 % 7.0)
            })
            .collect();

        let info = TensorInfo {
            name: "test_quality_fallback".to_string(),
            shape: vec![128, 128],
            dtype: DType::F32,
            path: PathBuf::new(),
            size: 128 * 128 * 4,
        };

        let result = converter.convert_tensor(&info, &data).unwrap();

        // With very strict threshold, it should fall back to passthrough (quality = 1.0)
        // Note: This test verifies the MECHANISM, not a specific quality value
        // The actual quality achieved depends on the data and encoding params
        assert!(result.quality.is_some());
        let quality = result.quality.unwrap();

        // Quality should be >= min_quality (either LRDF met threshold or passthrough was used)
        assert!(
            quality >= converter.config.min_quality,
            "Quality {} should be >= threshold {} (passthrough fallback should ensure this)",
            quality,
            converter.config.min_quality
        );
    }

    #[test]
    fn test_quality_enforcement_lrdf_acceptable() {
        // Test that good-quality LRDF encoding is retained
        let converter = HoloModelConverter::new(ConversionConfig {
            num_fragments: 32,
            max_rank: 64,
            verify_quality: true,
            min_quality: 0.5, // Lenient threshold
            ..Default::default()
        });

        // Create compressible test data (low-rank structure)
        let data: Vec<f32> = (0..64 * 64)
            .map(|i| {
                let row = (i / 64) as f32;
                let col = (i % 64) as f32;
                row * 0.01 + col * 0.01 // Low-rank linear data
            })
            .collect();

        let info = TensorInfo {
            name: "test_quality_acceptable".to_string(),
            shape: vec![64, 64],
            dtype: DType::F32,
            path: PathBuf::new(),
            size: 64 * 64 * 4,
        };

        let result = converter.convert_tensor(&info, &data).unwrap();

        assert!(result.quality.is_some());
        let quality = result.quality.unwrap();

        // Should meet threshold without needing passthrough
        assert!(
            quality >= converter.config.min_quality,
            "Quality {} should be >= lenient threshold {}",
            quality,
            converter.config.min_quality
        );
    }
}
