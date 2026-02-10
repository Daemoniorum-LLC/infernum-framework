//! HCT (Haagenti Compressed Tensor) format loader.
//!
//! This module provides support for loading compressed model weights stored
//! in the HCT format. The format enables efficient storage and loading of
//! quantized LLM weights with block-level compression.
//!
//! ## Format Benefits
//!
//! - **5-8x compression** on INT4/INT8 quantized weights
//! - **Block-level random access** for parallel decompression
//! - **Memory efficient** loading via streaming decompression
//! - **LZ4 or Zstd** compression algorithms

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};

use haagenti::compressive::CompressiveSpectralDecoder;
use haagenti::holotensor::{
    HoloFragment, HoloTensorDecoder, HoloTensorHeader, HoloTensorReader, HOLO_MAGIC,
};
use haagenti::tensor::{
    CompressionAlgorithm, DType as HctDType, HctHeader, HctReader, FLAG_HOLOGRAPHIC,
};
use haagenti::{Lz4Decompressor, ZstdDecompressor};
use std::io::{Read, Seek, SeekFrom};

/// Magic bytes for V3 spectral format (DCT with bitmap + f16 coefficients)
const V3_SPECTRAL_MAGIC: [u8; 4] = [0x33, 0x54, 0x43, 0x48]; // "3TCH" (0x48435433 in little-endian)

/// Metadata about an HCT file.
#[derive(Debug, Clone)]
pub struct HctMetadata {
    /// Tensor name (from filename).
    pub name: String,
    /// Original uncompressed size in bytes.
    pub original_size: u64,
    /// Compressed size in bytes.
    pub compressed_size: u64,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Data type of the tensor.
    pub dtype: HctDType,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// Compression algorithm used.
    pub algorithm: CompressionAlgorithm,
    /// Number of compressed blocks.
    pub num_blocks: u32,
    /// Header flags.
    pub flags: u16,
    /// Whether this file contains holographic encoded data.
    pub is_holographic: bool,
}

impl HctMetadata {
    /// Creates metadata from an HCT header.
    pub fn from_header(name: impl Into<String>, header: &HctHeader) -> Self {
        let is_holographic = header.flags & FLAG_HOLOGRAPHIC != 0;
        Self {
            name: name.into(),
            original_size: header.original_size,
            compressed_size: header.compressed_size,
            compression_ratio: header.original_size as f64 / header.compressed_size as f64,
            dtype: header.dtype,
            shape: header.shape.clone(),
            algorithm: header.algorithm,
            num_blocks: header.num_blocks,
            flags: header.flags,
            is_holographic,
        }
    }

    /// Creates metadata from a HoloTensor header.
    pub fn from_holo_header(name: impl Into<String>, header: &HoloTensorHeader) -> Self {
        // Use shape from header
        let shape = header.shape.clone();

        // Use original_size from header
        let original_size = header.original_size;
        // Estimate compressed size from fragment count
        let compressed_size = header.total_fragments as u64 * 256; // rough estimate

        Self {
            name: name.into(),
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size.max(1) as f64,
            dtype: header.dtype, // HoloTensor uses the same DType as HCT
            shape,
            algorithm: header.compression, // Use the actual compression from header
            num_blocks: header.total_fragments as u32,
            flags: FLAG_HOLOGRAPHIC, // Mark as holographic
            is_holographic: true,
        }
    }

    /// Returns true if this file contains holographic encoded data.
    pub fn is_holographic(&self) -> bool {
        self.is_holographic
    }
}

/// File format detected from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFileFormat {
    /// Standard HCT format (magic: HCTN)
    Hct,
    /// HoloTensor format (magic: HTNS)
    HoloTensor,
    /// V3 Spectral format (magic: 3TCH / 0x48435433)
    /// Uses DCT compression with bitmap + f16 coefficients
    CompressiveV3,
}

/// HCT file loader for compressed model weights.
///
/// Supports both standard HCT format (HCTN magic) and HoloTensor format (HTNS magic).
pub struct HctLoader {
    /// Path to the HCT file.
    path: std::path::PathBuf,
    /// Cached metadata.
    metadata: HctMetadata,
    /// Detected file format.
    format: TensorFileFormat,
    /// Cached HoloTensor header (only for HTNS format).
    #[allow(dead_code)]
    holo_header: Option<HoloTensorHeader>,
}

impl HctLoader {
    /// Opens an HCT or HoloTensor file for reading.
    ///
    /// Automatically detects the format by reading the magic bytes:
    /// - HCTN: Standard HCT format (block-compressed tensors)
    /// - HTNS: HoloTensor format (holographic compressed tensors)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or has an invalid format.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, HctError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| HctError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

        // Read magic bytes to detect format
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| HctError::Io {
            path: path.to_path_buf(),
            message: format!("Failed to read magic bytes: {}", e),
        })?;

        // Seek back to start
        file.seek(SeekFrom::Start(0)).map_err(|e| HctError::Io {
            path: path.to_path_buf(),
            message: format!("Failed to seek: {}", e),
        })?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        if magic == HOLO_MAGIC {
            // HoloTensor format (HTNS)
            let reader = BufReader::new(file);
            let holo_reader = HoloTensorReader::new(reader).map_err(|e| HctError::Format {
                message: format!("Failed to parse HoloTensor header: {}", e),
            })?;

            let header = holo_reader.header();
            let metadata = HctMetadata::from_holo_header(name, header);

            Ok(Self {
                path: path.to_path_buf(),
                metadata,
                format: TensorFileFormat::HoloTensor,
                holo_header: Some(header.clone()),
            })
        } else if magic == V3_SPECTRAL_MAGIC {
            // V3 Spectral format (3TCH / 0x48435433)
            // Read header: [magic:4][width:4][height:4][retain_count:4][essential_count:4][detail_per_frag:4]
            let mut header_buf = [0u8; 24];
            header_buf[0..4].copy_from_slice(&magic);
            file.read_exact(&mut header_buf[4..])
                .map_err(|e| HctError::Io {
                    path: path.to_path_buf(),
                    message: format!("Failed to read V3 header: {}", e),
                })?;

            let width =
                u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]])
                    as u64;
            let height =
                u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]])
                    as u64;

            // Get file size for compression stats
            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let original_size = width * height * 4; // f32 = 4 bytes

            let metadata = HctMetadata {
                name,
                original_size,
                compressed_size: file_size,
                compression_ratio: original_size as f64 / file_size.max(1) as f64,
                dtype: HctDType::F32,
                shape: vec![height, width], // [rows, cols] = [height, width]
                algorithm: CompressionAlgorithm::Zstd, // V3 typically uses zstd compression on top
                num_blocks: 1,              // V3 is single-block spectral
                flags: 0,
                is_holographic: false, // V3 uses spectral/DCT, not holographic/LRDF
            };

            Ok(Self {
                path: path.to_path_buf(),
                metadata,
                format: TensorFileFormat::CompressiveV3,
                holo_header: None,
            })
        } else {
            // Standard HCT format (HCTN) or other
            let reader = BufReader::new(file);
            let hct_reader = HctReader::new(reader).map_err(|e| HctError::Format {
                message: format!("Failed to parse HCT header: {}", e),
            })?;

            let metadata = HctMetadata::from_header(name, hct_reader.header());

            Ok(Self {
                path: path.to_path_buf(),
                metadata,
                format: TensorFileFormat::Hct,
                holo_header: None,
            })
        }
    }

    /// Returns the detected file format.
    pub fn format(&self) -> TensorFileFormat {
        self.format
    }

    /// Returns the metadata for this HCT file.
    pub fn metadata(&self) -> &HctMetadata {
        &self.metadata
    }

    /// Returns the path to this HCT file.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Decompresses all blocks and returns the raw bytes.
    ///
    /// For HCT format, this decompresses using LZ4 or Zstd.
    /// For HoloTensor format, this reads all fragments and reconstructs the tensor.
    /// For V3 Spectral format, this uses DCT reconstruction.
    ///
    /// # Errors
    ///
    /// Returns an error if decompression fails.
    pub fn decompress_all(&self) -> Result<Vec<u8>, HctError> {
        match self.format {
            TensorFileFormat::Hct => self.decompress_hct(),
            TensorFileFormat::HoloTensor => self.reconstruct_holotensor(),
            TensorFileFormat::CompressiveV3 => self.decompress_v3_spectral(),
        }
    }

    /// Decompresses HCT format using block decompression.
    fn decompress_hct(&self) -> Result<Vec<u8>, HctError> {
        let file = File::open(&self.path).map_err(|e| HctError::Io {
            path: self.path.clone(),
            message: e.to_string(),
        })?;

        let reader = BufReader::new(file);
        let mut hct_reader = HctReader::new(reader).map_err(|e| HctError::Format {
            message: format!("Failed to parse HCT: {}", e),
        })?;

        let data = match self.metadata.algorithm {
            CompressionAlgorithm::Lz4 => {
                let decompressor = Lz4Decompressor::new();
                hct_reader
                    .decompress_all(&decompressor)
                    .map_err(|e| HctError::Decompress {
                        message: format!("LZ4 decompression failed: {}", e),
                    })?
            },
            CompressionAlgorithm::Zstd => {
                let decompressor = ZstdDecompressor::new();
                hct_reader
                    .decompress_all(&decompressor)
                    .map_err(|e| HctError::Decompress {
                        message: format!("Zstd decompression failed: {}", e),
                    })?
            },
        };

        Ok(data)
    }

    /// Reconstructs HoloTensor format by reading all fragments.
    fn reconstruct_holotensor(&self) -> Result<Vec<u8>, HctError> {
        let file = File::open(&self.path).map_err(|e| HctError::Io {
            path: self.path.clone(),
            message: e.to_string(),
        })?;

        let reader = BufReader::new(file);
        let mut holo_reader = HoloTensorReader::new(reader).map_err(|e| HctError::Format {
            message: format!("Failed to parse HoloTensor: {}", e),
        })?;

        let (header, fragments) = holo_reader.read_all().map_err(|e| HctError::Holographic {
            message: format!("Failed to read fragments: {}", e),
        })?;

        // Check if this is HCT3 format (CompressiveSpectralEncoder output)
        // HCT3 magic: 0x48435433 ("HCT3" in little-endian)
        const HCT3_MAGIC: u32 = 0x48435433;
        let is_hct3 = if let Some(frag0) = fragments.iter().find(|f| f.index == 0) {
            if frag0.data.len() >= 4 {
                let magic = u32::from_le_bytes([
                    frag0.data[0],
                    frag0.data[1],
                    frag0.data[2],
                    frag0.data[3],
                ]);
                magic == HCT3_MAGIC
            } else {
                false
            }
        } else {
            false
        };

        let f32_data = if is_hct3
            && matches!(
                header.encoding,
                haagenti::holotensor::HolographicEncoding::Spectral
            ) {
            // Use CompressiveSpectralDecoder for HCT3 format
            let mut decoder = CompressiveSpectralDecoder::new();

            // Find and add fragment 0 (essentials) first
            if let Some(frag0) = fragments.iter().find(|f| f.index == 0) {
                decoder
                    .add_essentials(frag0)
                    .map_err(|e| HctError::Holographic {
                        message: format!("Failed to add HCT3 essentials: {}", e),
                    })?;
            } else {
                return Err(HctError::Holographic {
                    message: "HCT3 format missing fragment 0 (essentials)".to_string(),
                });
            }

            // Add detail fragments (1..N)
            for fragment in fragments.iter().filter(|f| f.index > 0) {
                decoder
                    .add_detail(fragment)
                    .map_err(|e| HctError::Holographic {
                        message: format!(
                            "Failed to add HCT3 detail fragment {}: {}",
                            fragment.index, e
                        ),
                    })?;
            }

            decoder.reconstruct().map_err(|e| HctError::Holographic {
                message: format!("Failed to reconstruct HCT3 tensor: {}", e),
            })?
        } else {
            // Use standard HoloTensorDecoder for legacy formats (SV01/SV02/SV03, RPH, LRDF)
            let mut decoder = HoloTensorDecoder::new(header.clone());

            for fragment in fragments {
                decoder
                    .add_fragment(fragment)
                    .map_err(|e| HctError::Holographic {
                        message: format!("Failed to add fragment: {}", e),
                    })?;
            }

            decoder.reconstruct().map_err(|e| HctError::Holographic {
                message: format!("Failed to reconstruct tensor: {}", e),
            })?
        };

        // Convert f32 to bytes
        let bytes: Vec<u8> = f32_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        Ok(bytes)
    }

    /// Decompresses V3 spectral format using DCT reconstruction.
    ///
    /// V3 format uses 2D DCT (Discrete Cosine Transform) compression with:
    /// - Bitmap indices for sparse coefficient storage
    /// - f16 coefficients for compact storage
    /// - Zstd compression on the bitmap+coefficient data (after 20-byte header)
    fn decompress_v3_spectral(&self) -> Result<Vec<u8>, HctError> {
        // Read entire file
        let mut file = File::open(&self.path).map_err(|e| HctError::Io {
            path: self.path.clone(),
            message: e.to_string(),
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| HctError::Io {
            path: self.path.clone(),
            message: format!("Failed to read file: {}", e),
        })?;

        // V3 files have format: [20-byte header][zstd compressed bitmap+coefficients]
        // Header: [magic:4][width:4][height:4][retain_count:4][essential_count:4]
        if data.len() < 24 {
            return Err(HctError::Decompress {
                message: "V3 file too short".to_string(),
            });
        }

        // V3 header is 20 bytes, zstd starts at byte 20 OR 21 depending on variant
        // Check both positions for zstd magic (0x28 0xB5 0x2F 0xFD)
        let zstd_offset = if data.len() >= 24
            && data[20] == 0x28
            && data[21] == 0xB5
            && data[22] == 0x2F
            && data[23] == 0xFD
        {
            Some(20)
        } else if data.len() >= 25
            && data[21] == 0x28
            && data[22] == 0xB5
            && data[23] == 0x2F
            && data[24] == 0xFD
        {
            Some(21)
        } else {
            None
        };

        let decompressed_data = if let Some(offset) = zstd_offset {
            // Decompress only the zstd portion
            let zstd_data = &data[offset..];
            let decompressed_payload =
                zstd::decode_all(zstd_data).map_err(|e| HctError::Decompress {
                    message: format!("Zstd decompression failed: {}", e),
                })?;

            // Check if decompressed data starts with V3 magic (self-contained format)
            if decompressed_payload.len() >= 4
                && decompressed_payload[0] == 0x33
                && decompressed_payload[1] == 0x54
                && decompressed_payload[2] == 0x43
                && decompressed_payload[3] == 0x48
            {
                decompressed_payload
            } else {
                // Reconstruct: [header (first 20 bytes)][padding to 24 bytes][decompressed bitmap+coefficients]
                let mut full_data = data[0..20].to_vec();
                // Append 4 more bytes for detail_per_frag (set to 0)
                full_data.extend_from_slice(&[0u8; 4]);
                full_data.extend(decompressed_payload);
                full_data
            }
        } else {
            // No embedded zstd, check if entire file is zstd wrapped
            let is_zstd = data.len() >= 4
                && data[0] == 0x28
                && data[1] == 0xB5
                && data[2] == 0x2F
                && data[3] == 0xFD;

            if is_zstd {
                zstd::decode_all(&data[..]).map_err(|e| HctError::Decompress {
                    message: format!("Zstd decompression failed: {}", e),
                })?
            } else {
                data
            }
        };

        // Create V3 decoder
        let mut decoder = CompressiveSpectralDecoder::new();

        // The entire file (after decompression) is the fragment 0 essentials data
        let fragment0 = HoloFragment {
            index: 0,
            flags: 0,
            checksum: 0,
            data: decompressed_data,
        };

        decoder
            .add_essentials(&fragment0)
            .map_err(|e| HctError::Decompress {
                message: format!("V3 essentials parsing failed: {}", e),
            })?;

        // Reconstruct tensor via IDCT
        let f32_data = decoder.reconstruct().map_err(|e| HctError::Decompress {
            message: format!("V3 IDCT reconstruction failed: {}", e),
        })?;

        // Convert f32 to bytes
        let bytes: Vec<u8> = f32_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        Ok(bytes)
    }

    /// Decompresses and converts to a Candle tensor.
    ///
    /// # Arguments
    ///
    /// * `device` - Target device for the tensor
    /// * `target_dtype` - Optional dtype conversion (if None, uses native dtype)
    ///
    /// # Errors
    ///
    /// Returns an error if decompression or tensor creation fails.
    pub fn to_tensor(
        &self,
        device: &Device,
        target_dtype: Option<DType>,
    ) -> Result<Tensor, HctError> {
        let data = self.decompress_all()?;

        let shape: Vec<usize> = self.metadata.shape.iter().map(|&d| d as usize).collect();

        // Handle quantized types with dequantization (only for standard HCT format)
        if self.format == TensorFileFormat::Hct
            && matches!(self.metadata.dtype, HctDType::I4 | HctDType::I8)
        {
            return self.dequantize_to_tensor(&data, &shape, device, target_dtype);
        }

        // CRITICAL: HoloTensor and CompressiveV3 formats ALWAYS output F32 bytes from reconstruction
        // (both reconstruct_holotensor() and decompress_v3_spectral() convert f32_data to bytes)
        // So we must always interpret their output as F32, regardless of metadata.dtype
        let native_dtype = match self.format {
            TensorFileFormat::HoloTensor | TensorFileFormat::CompressiveV3 => {
                // Reconstruction always outputs F32 data converted to bytes
                DType::F32
            },
            TensorFileFormat::Hct => {
                // Standard HCT stores data in native dtype
                match self.metadata.dtype {
                    HctDType::F32 => DType::F32,
                    HctDType::F16 => DType::F16,
                    HctDType::BF16 => DType::BF16,
                    HctDType::I8 | HctDType::I4 => unreachable!(), // Handled above
                }
            },
        };

        // Create tensor from raw bytes
        let tensor = match native_dtype {
            DType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape.as_slice(), device)
            },
            DType::F16 => {
                let halfs: Vec<f16> = data
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Tensor::from_vec(halfs, shape.as_slice(), device)
            },
            DType::BF16 => {
                let bfloats: Vec<bf16> = data
                    .chunks_exact(2)
                    .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Tensor::from_vec(bfloats, shape.as_slice(), device)
            },
            _ => {
                // Should not reach here for supported types
                return Err(HctError::UnsupportedDtype {
                    dtype: format!("{:?}", self.metadata.dtype),
                });
            },
        }
        .map_err(|e| HctError::Tensor {
            message: format!("Failed to create tensor: {}", e),
        })?;

        // Convert to target dtype if specified
        if let Some(target) = target_dtype {
            if target != native_dtype {
                return tensor.to_dtype(target).map_err(|e| HctError::Tensor {
                    message: format!("Failed to convert dtype: {}", e),
                });
            }
        }

        Ok(tensor)
    }

    /// Dequantizes INT4/INT8 data to a tensor.
    ///
    /// The data layout is:
    /// - FP16 scale factors (one per block of 128 elements)
    /// - Packed quantized values (4-bit or 8-bit)
    fn dequantize_to_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        device: &Device,
        target_dtype: Option<DType>,
    ) -> Result<Tensor, HctError> {
        // Block size must match the quantizer's DEFAULT_BLOCK_SIZE (128).
        // See GPU-CODEC-PIPELINE-SPEC.md §2 and DD-1.
        const BLOCK_SIZE: usize = crate::gpu_dequant::INT4_BLOCK_SIZE;

        let num_elements: usize = shape.iter().product();

        match self.metadata.dtype {
            HctDType::I4 => {
                // INT4 format: [all FP16 scales][all packed INT4 data]
                // Layout: scales first (2 bytes per block), then packed nibbles
                const Q4_BLOCK_SIZE: usize = crate::gpu_dequant::INT4_BLOCK_SIZE;

                let num_blocks = (num_elements + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
                let scales_bytes = num_blocks * 2; // FP16 scales
                let packed_bytes = (num_elements + 1) / 2; // 2 nibbles per byte

                let expected_bytes = scales_bytes + packed_bytes;

                if data.len() < expected_bytes {
                    return Err(HctError::Tensor {
                        message: format!(
                            "INT4 data too short: {} bytes, expected {} ({} scales + {} packed)",
                            data.len(),
                            expected_bytes,
                            scales_bytes,
                            packed_bytes
                        ),
                    });
                }

                // Parse all FP16 scales first
                let scales_data = &data[..scales_bytes];
                let packed_data = &data[scales_bytes..];

                let scales: Vec<f32> = scales_data
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect();

                // Dequantize INT4 to F32
                // Formula: value = (nibble - 8) * scale
                // - nibble is in range [0, 15], centered at 8
                // - (nibble - 8) gives range [-8, +7]
                // - scale is per-block FP16, computed as max_abs/7.0 during quantization
                let mut values = Vec::with_capacity(num_elements);

                for i in 0..num_elements {
                    let byte_idx = i / 2;
                    let nibble_idx = i % 2;

                    let packed_byte = packed_data[byte_idx];
                    let nibble = if nibble_idx == 0 {
                        packed_byte & 0x0F
                    } else {
                        (packed_byte >> 4) & 0x0F
                    };

                    // INT4 uses centered interpretation: nibble - 8 gives range -8 to +7
                    let centered_val = (nibble as i32) - 8;

                    let block_idx = i / Q4_BLOCK_SIZE;
                    let scale = scales.get(block_idx).copied().unwrap_or(1.0);
                    values.push((centered_val as f32) * scale);
                }

                let tensor =
                    Tensor::from_vec(values, shape, device).map_err(|e| HctError::Tensor {
                        message: format!("Failed to create tensor from Q4_0 data: {}", e),
                    })?;

                // Convert to target dtype if needed
                let target = target_dtype.unwrap_or(DType::F32);
                if target != DType::F32 {
                    tensor.to_dtype(target).map_err(|e| HctError::Tensor {
                        message: format!("Failed to convert to {:?}: {}", target, e),
                    })
                } else {
                    Ok(tensor)
                }
            },
            HctDType::I8 => {
                // INT8: 1 value per byte, symmetric quantization
                // Data layout: [FP16 scales][INT8 values]
                let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let scales_bytes = num_blocks * 2; // FP16 scales

                if data.len() < num_elements + scales_bytes {
                    return Err(HctError::Tensor {
                        message: format!(
                            "INT8 data too short: {} bytes, expected {} + {} = {}",
                            data.len(),
                            scales_bytes,
                            num_elements,
                            num_elements + scales_bytes
                        ),
                    });
                }

                // Layout: scales first, then quantized data
                let scales_data = &data[..scales_bytes];
                let quant_data = &data[scales_bytes..scales_bytes + num_elements];

                // Parse FP16 scales
                let scales: Vec<f32> = scales_data
                    .chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect();

                // Dequantize INT8 to F32
                let mut values = Vec::with_capacity(num_elements);
                for (i, &q) in quant_data.iter().enumerate() {
                    let signed_val = q as i8;
                    let block_idx = i / BLOCK_SIZE;
                    let scale = scales.get(block_idx).copied().unwrap_or(1.0);
                    values.push((signed_val as f32) * scale);
                }

                let tensor =
                    Tensor::from_vec(values, shape, device).map_err(|e| HctError::Tensor {
                        message: format!("Failed to create tensor from INT8 data: {}", e),
                    })?;

                // Convert to target dtype if needed
                let target = target_dtype.unwrap_or(DType::F32);
                if target != DType::F32 {
                    tensor.to_dtype(target).map_err(|e| HctError::Tensor {
                        message: format!("Failed to convert to {:?}: {}", target, e),
                    })
                } else {
                    Ok(tensor)
                }
            },
            _ => Err(HctError::UnsupportedDtype {
                dtype: format!("{:?}", self.metadata.dtype),
            }),
        }
    }

    /// Returns true if this HCT file contains holographic encoded data.
    ///
    /// Holographic files use `FLAG_HOLOGRAPHIC` and require progressive
    /// reconstruction instead of standard block decompression.
    pub fn is_holographic(&self) -> bool {
        self.metadata.is_holographic
    }
}

/// Load multiple HCT files from a directory into a tensor map.
///
/// This function scans for `.hct` files and loads them in parallel.
///
/// # Arguments
///
/// * `dir` - Directory containing HCT files
/// * `device` - Target device for tensors
/// * `dtype` - Target dtype for all tensors
///
/// # Returns
///
/// A map of tensor names to tensors.
pub fn load_hct_directory(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, HctError> {
    use rayon::prelude::*;

    let dir = dir.as_ref();

    // Find all .hct files
    let hct_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    tracing::info!(
        count = hct_files.len(),
        directory = %dir.display(),
        "Loading HCT weight files"
    );

    // Load in parallel using rayon
    let results: Vec<Result<(String, Tensor), HctError>> = hct_files
        .par_iter()
        .map(|path| {
            let loader = HctLoader::from_file(path)?;
            // Convert filename back to tensor name (replace _ with .)
            // The converter replaces . with _ for filesystem compatibility
            let filename = loader.metadata().name.clone();
            let tensor_name = filename_to_tensor_name(&filename);
            let tensor = loader.to_tensor(device, Some(dtype))?;
            Ok((tensor_name, tensor))
        })
        .collect();

    // Collect results
    let mut tensors = HashMap::new();
    for result in results {
        let (name, tensor) = result?;
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

/// Load HCT files with GPU-accelerated decompression when available.
///
/// This function attempts to use GPU decompression for LZ4-compressed files
/// when CUDA is available and the device is a CUDA device. Falls back to
/// CPU decompression otherwise.
///
/// # Arguments
///
/// * `dir` - Directory containing HCT files
/// * `device` - Target device for tensors (should be CUDA for GPU acceleration)
/// * `dtype` - Target dtype for all tensors
///
/// # Returns
///
/// A map of tensor names to tensors, with decompression performed on GPU
/// when possible.
#[cfg(feature = "cuda")]
pub fn load_hct_directory_gpu(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, HctError> {
    use crate::gpu_lz4::GpuLz4Context;

    let dir = dir.as_ref();

    // Try to create GPU context
    let gpu_ctx = match device {
        Device::Cuda(_cuda_dev) => {
            // Try to get device ordinal
            match GpuLz4Context::new(0) {
                Ok(mut ctx) => {
                    if let Err(e) = ctx.load_kernel() {
                        tracing::warn!(
                            error = %e,
                            "Failed to load GPU LZ4 kernel, falling back to CPU"
                        );
                        None
                    } else {
                        Some(ctx)
                    }
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create GPU LZ4 context, falling back to CPU"
                    );
                    None
                },
            }
        },
        _ => None,
    };

    // Find all .hct files
    let hct_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    // Count LZ4 files for GPU acceleration
    let lz4_count = hct_files
        .iter()
        .filter(|path| {
            HctLoader::from_file(path)
                .map(|l| matches!(l.metadata().algorithm, CompressionAlgorithm::Lz4))
                .unwrap_or(false)
        })
        .count();

    tracing::info!(
        count = hct_files.len(),
        lz4_count = lz4_count,
        gpu_enabled = gpu_ctx.is_some(),
        directory = %dir.display(),
        "Loading HCT weight files"
    );

    let mut tensors = HashMap::new();

    // If we have GPU context and LZ4 files, try GPU decompression
    if let Some(ref ctx) = gpu_ctx {
        // Collect LZ4 blocks for batch GPU decompression
        let mut lz4_loaders: Vec<(String, HctLoader)> = Vec::new();
        let mut other_loaders: Vec<(String, HctLoader)> = Vec::new();

        for path in &hct_files {
            let loader = HctLoader::from_file(path)?;
            let name = loader.metadata().name.clone();

            if matches!(loader.metadata().algorithm, CompressionAlgorithm::Lz4) {
                lz4_loaders.push((name, loader));
            } else {
                other_loaders.push((name, loader));
            }
        }

        // Process LZ4 files with GPU
        for (name, loader) in lz4_loaders {
            match gpu_decompress_tensor(ctx, &loader, device, dtype) {
                Ok(tensor) => {
                    tensors.insert(name, tensor);
                },
                Err(e) => {
                    tracing::warn!(
                        name = %name,
                        error = %e,
                        "GPU decompression failed, falling back to CPU"
                    );
                    // Fallback to CPU
                    let tensor = loader.to_tensor(device, Some(dtype))?;
                    tensors.insert(name, tensor);
                },
            }
        }

        // Process non-LZ4 files with CPU (Zstd not yet GPU-accelerated)
        for (name, loader) in other_loaders {
            let tensor = loader.to_tensor(device, Some(dtype))?;
            tensors.insert(name, tensor);
        }
    } else {
        // No GPU context, use CPU for all files
        return load_hct_directory(dir, device, dtype);
    }

    Ok(tensors)
}

/// GPU decompression helper for a single tensor.
#[cfg(feature = "cuda")]
fn gpu_decompress_tensor(
    ctx: &crate::gpu_lz4::GpuLz4Context,
    loader: &HctLoader,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, HctError> {
    // Read the HCT file to extract compressed blocks
    let path = loader.path();
    let file = File::open(path).map_err(|e| HctError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let reader = BufReader::new(file);
    let mut hct_reader = HctReader::new(reader).map_err(|e| HctError::Format {
        message: format!("Failed to parse HCT: {}", e),
    })?;

    // Get block information
    let header = hct_reader.header();
    let num_blocks = header.num_blocks as usize;
    let block_size = header.block_size as usize;
    let original_size = header.original_size as usize;

    // Read compressed blocks
    let blocks: Vec<(Vec<u8>, usize)> = (0..num_blocks)
        .map(|i| {
            let uncompressed_size = if i == num_blocks - 1 {
                // Last block may be smaller
                original_size - (i * block_size)
            } else {
                block_size
            };

            // Read compressed block data
            let compressed = hct_reader.read_block(i).map_err(|e| HctError::Decompress {
                message: format!("Failed to read block {}: {}", i, e),
            })?;

            Ok((compressed, uncompressed_size))
        })
        .collect::<Result<Vec<_>, HctError>>()?;

    // Decompress using GPU
    let shape: Vec<usize> = loader
        .metadata()
        .shape
        .iter()
        .map(|&d| d as usize)
        .collect();

    ctx.decompress_to_tensor(&blocks, &shape, dtype, device)
        .map_err(|e| HctError::Decompress {
            message: format!("GPU decompression failed: {}", e),
        })
}

/// Result of progressive loading with quality information.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct ProgressiveLoadResult {
    /// Loaded tensors (name -> tensor).
    pub tensors: HashMap<String, Tensor>,
    /// Quality achieved for holographic tensors (name -> quality).
    pub qualities: HashMap<String, f32>,
    /// Number of holographic files processed.
    pub holographic_count: usize,
    /// Number of standard files processed.
    pub standard_count: usize,
}

/// Load HCT files with progressive holographic reconstruction.
///
/// This function handles both standard HCT files and holographic HCT files:
/// - Standard files: Decompressed using GPU LZ4 when available
/// - Holographic files: Progressively reconstructed to target quality
///
/// # Arguments
///
/// * `dir` - Directory containing HCT files
/// * `device` - Target device for tensors
/// * `dtype` - Target dtype for all tensors
/// * `min_quality` - Minimum quality for holographic reconstruction (0.0-1.0)
///
/// # Returns
///
/// Progressive load result with tensors and quality information.
#[cfg(feature = "cuda")]
pub fn load_hct_directory_gpu_progressive(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
    min_quality: f32,
) -> Result<ProgressiveLoadResult, HctError> {
    use crate::gpu_holo::StreamingHoloContext;
    use crate::gpu_lz4::GpuLz4Context;

    let dir = dir.as_ref();

    // Find all .hct files
    let hct_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    // Categorize files
    let mut standard_loaders: Vec<(String, HctLoader)> = Vec::new();
    let mut holographic_loaders: Vec<(String, HctLoader)> = Vec::new();

    for path in &hct_files {
        let loader = HctLoader::from_file(path)?;
        let name = loader.metadata().name.clone();

        if loader.is_holographic() {
            holographic_loaders.push((name, loader));
        } else {
            standard_loaders.push((name, loader));
        }
    }

    tracing::info!(
        total = hct_files.len(),
        standard = standard_loaders.len(),
        holographic = holographic_loaders.len(),
        min_quality = %min_quality,
        directory = %dir.display(),
        "Loading HCT files with progressive holographic support"
    );

    let mut tensors = HashMap::new();
    let mut qualities = HashMap::new();

    // Try to create GPU contexts
    let lz4_ctx = match device {
        Device::Cuda(_) => match GpuLz4Context::new(0) {
            Ok(mut ctx) => {
                if ctx.load_kernel().is_ok() {
                    Some(ctx)
                } else {
                    None
                }
            },
            Err(_) => None,
        },
        _ => None,
    };

    let holo_ctx = match device {
        Device::Cuda(_) => match StreamingHoloContext::new(0, 4) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to create holographic context, holographic files will fail"
                );
                None
            },
        },
        _ => None,
    };

    // Process standard files
    for (name, loader) in standard_loaders {
        let tensor = if let Some(ref ctx) = lz4_ctx {
            if matches!(loader.metadata().algorithm, CompressionAlgorithm::Lz4) {
                match gpu_decompress_tensor(ctx, &loader, device, dtype) {
                    Ok(t) => t,
                    Err(_) => loader.to_tensor(device, Some(dtype))?,
                }
            } else {
                loader.to_tensor(device, Some(dtype))?
            }
        } else {
            loader.to_tensor(device, Some(dtype))?
        };
        tensors.insert(name, tensor);
    }

    // Process holographic files
    if !holographic_loaders.is_empty() {
        if holo_ctx.is_none() {
            return Err(HctError::Holographic {
                message: "Holographic files present but GPU holographic context unavailable"
                    .to_string(),
            });
        }

        let holo_ctx = holo_ctx.as_ref().expect("checked above");

        for (name, loader) in &holographic_loaders {
            match load_holographic_tensor(holo_ctx, loader, device, dtype, min_quality) {
                Ok((tensor, quality)) => {
                    tensors.insert(name.clone(), tensor);
                    qualities.insert(name.clone(), quality);
                },
                Err(e) => {
                    return Err(HctError::Holographic {
                        message: format!("Failed to load holographic tensor '{}': {}", name, e),
                    });
                },
            }
        }
    }

    Ok(ProgressiveLoadResult {
        tensors,
        qualities,
        holographic_count: holographic_loaders.len(),
        standard_count: hct_files.len() - holographic_loaders.len(),
    })
}

/// Load a single holographic tensor with streaming reconstruction.
#[cfg(feature = "cuda")]
fn load_holographic_tensor(
    ctx: &crate::gpu_holo::StreamingHoloContext,
    loader: &HctLoader,
    device: &Device,
    dtype: DType,
    min_quality: f32,
) -> Result<(Tensor, f32), HctError> {
    use haagenti::holotensor::HoloTensorReader;

    let path = loader.path();
    let file = File::open(path).map_err(|e| HctError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let reader = BufReader::new(file);

    // Use HoloTensorReader to properly parse the HoloTensor format
    let mut holo_reader = HoloTensorReader::new(reader).map_err(|e| HctError::Format {
        message: format!("Failed to parse HoloTensor: {}", e),
    })?;

    // Read fragments up to target quality, or all if needed for minimum quality
    let (fragments, _predicted_quality) =
        holo_reader
            .read_to_quality(min_quality)
            .map_err(|e| HctError::Holographic {
                message: format!("Failed to read fragments: {}", e),
            })?;

    let holo_header = holo_reader.header().clone();

    // Reconstruct using streaming context
    let reconstructed = ctx
        .reconstruct_streaming(&holo_header, fragments.iter(), min_quality)
        .map_err(|e| HctError::Holographic {
            message: format!("Holographic reconstruction failed: {}", e),
        })?;

    // Calculate achieved quality
    let quality = holo_header
        .quality_curve
        .predict(fragments.len() as u16, holo_header.total_fragments);

    // Copy to host and create tensor
    let host_data =
        ctx.context()
            .copy_to_host(&reconstructed)
            .map_err(|e| HctError::Holographic {
                message: format!("Failed to copy reconstructed data to host: {}", e),
            })?;

    // Create tensor
    let shape: Vec<usize> = loader
        .metadata()
        .shape
        .iter()
        .map(|&d| d as usize)
        .collect();
    let tensor =
        Tensor::from_vec(host_data, shape.as_slice(), device).map_err(|e| HctError::Tensor {
            message: format!("Failed to create tensor from reconstructed data: {}", e),
        })?;

    // Convert dtype if needed
    let tensor = if dtype != DType::F32 {
        tensor.to_dtype(dtype).map_err(|e| HctError::Tensor {
            message: format!("Failed to convert dtype: {}", e),
        })?
    } else {
        tensor
    };

    Ok((tensor, quality))
}

/// CPU-only version of progressive loading (for non-CUDA builds).
#[cfg(not(feature = "cuda"))]
pub fn load_hct_directory_gpu_progressive(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
    _min_quality: f32,
) -> Result<HashMap<String, Tensor>, HctError> {
    // Check for holographic files and fail if found
    let dir = dir.as_ref();
    let hct_files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    for path in &hct_files {
        let loader = HctLoader::from_file(path)?;
        if loader.is_holographic() {
            return Err(HctError::Holographic {
                message: "Holographic files require CUDA support".to_string(),
            });
        }
    }

    // No holographic files, use standard loading
    load_hct_directory(dir, device, dtype)
}

/// CPU-only version of GPU loading (for non-CUDA builds).
#[cfg(not(feature = "cuda"))]
pub fn load_hct_directory_gpu(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, HctError> {
    // No GPU available, delegate to CPU implementation
    load_hct_directory(dir, device, dtype)
}

/// Convert a filename (with underscores) back to tensor name (with dots).
///
/// The HoloTensor converter replaces `.` with `_` for filesystem compatibility.
/// This function reverses that transformation, preserving underscores within
/// component names while restoring dots as hierarchy separators.
///
/// Examples:
/// - `model_embed_tokens_weight` -> `model.embed_tokens.weight`
/// - `model_layers_0_self_attn_q_proj_weight` -> `model.layers.0.self_attn.q_proj.weight`
pub fn filename_to_tensor_name(filename: &str) -> String {
    // Known patterns that should be kept together with underscores
    // These are common transformer component suffixes (reserved for future use)
    let _compound_suffixes = ["_tokens", "_proj", "_attn", "_layernorm", "_norm", "_mlp"];

    // Build regex-like state machine
    let mut result = String::new();
    let parts: Vec<&str> = filename.split('_').collect();
    let mut i = 0;

    while i < parts.len() {
        let part = parts[i];

        // Handle "model" prefix
        if i == 0 && part == "model" {
            result.push_str("model");
            i += 1;
            continue;
        }

        // Handle "layers" followed by number
        if part == "layers" && i + 1 < parts.len() {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("layers");
            if let Ok(_) = parts[i + 1].parse::<usize>() {
                result.push('.');
                result.push_str(parts[i + 1]);
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        // Handle "embed" followed by "tokens"
        if part == "embed" && i + 1 < parts.len() && parts[i + 1] == "tokens" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("embed_tokens");
            i += 2;
            continue;
        }

        // Handle "self" followed by "attn"
        if part == "self" && i + 1 < parts.len() && parts[i + 1] == "attn" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("self_attn");
            i += 2;
            continue;
        }

        // Handle "input_layernorm"
        if part == "input" && i + 1 < parts.len() && parts[i + 1] == "layernorm" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("input_layernorm");
            i += 2;
            continue;
        }

        // Handle "post_attention_layernorm"
        if part == "post"
            && i + 2 < parts.len()
            && parts[i + 1] == "attention"
            && parts[i + 2] == "layernorm"
        {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("post_attention_layernorm");
            i += 3;
            continue;
        }

        // Handle projection names like "q_proj", "k_proj", "v_proj", "o_proj", etc.
        if i + 1 < parts.len() && parts[i + 1] == "proj" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str(part);
            result.push_str("_proj");
            i += 2;
            continue;
        }

        // Handle MLP components: "up", "down", "gate" followed by "proj"
        // (already covered by the above rule)

        // Handle final "weight" or "bias"
        if part == "weight" || part == "bias" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str(part);
            i += 1;
            continue;
        }

        // Handle "lm" followed by "head"
        if part == "lm" && i + 1 < parts.len() && parts[i + 1] == "head" {
            if !result.is_empty() {
                result.push('.');
            }
            result.push_str("lm_head");
            i += 2;
            continue;
        }

        // Default: add as separate component
        if !result.is_empty() {
            result.push('.');
        }
        result.push_str(part);
        i += 1;
    }

    result
}

/// Errors from HCT loading operations.
#[derive(Debug, thiserror::Error)]
pub enum HctError {
    /// IO error reading file.
    #[error("IO error reading {path}: {message}")]
    Io {
        /// File path.
        path: std::path::PathBuf,
        /// Error message.
        message: String,
    },

    /// Invalid HCT format.
    #[error("Invalid HCT format: {message}")]
    Format {
        /// Error message.
        message: String,
    },

    /// Decompression error.
    #[error("Decompression failed: {message}")]
    Decompress {
        /// Error message.
        message: String,
    },

    /// Tensor creation error.
    #[error("Tensor error: {message}")]
    Tensor {
        /// Error message.
        message: String,
    },

    /// Unsupported dtype.
    #[error("Unsupported dtype for tensor creation: {dtype}")]
    UnsupportedDtype {
        /// The unsupported dtype.
        dtype: String,
    },

    /// Holographic tensor error.
    #[error("Holographic tensor error: {message}")]
    Holographic {
        /// Error message.
        message: String,
    },
}

impl From<HctError> for infernum_core::Error {
    fn from(err: HctError) -> Self {
        infernum_core::Error::ModelLoad {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hct_metadata_from_header() {
        let header = HctHeader {
            algorithm: CompressionAlgorithm::Zstd,
            dtype: HctDType::I8,
            flags: 0,
            original_size: 1024 * 1024,
            compressed_size: 256 * 1024,
            block_size: 16 * 1024,
            num_blocks: 64,
            shape: vec![1024, 1024],
        };

        let metadata = HctMetadata::from_header("test_tensor", &header);

        assert_eq!(metadata.name, "test_tensor");
        assert_eq!(metadata.original_size, 1024 * 1024);
        assert_eq!(metadata.compressed_size, 256 * 1024);
        assert!((metadata.compression_ratio - 4.0).abs() < 0.01);
        assert_eq!(metadata.shape, vec![1024, 1024]);
    }

    // ==================== INT4 Dequantization Tests ====================

    /// Helper to create INT4 quantized data in HCT format.
    /// Layout: [FP16 scales][packed INT4 nibbles]
    fn create_int4_data(values: &[f32], block_size: usize) -> Vec<u8> {
        let num_elements = values.len();
        let num_blocks = (num_elements + block_size - 1) / block_size;

        // Compute per-block scales
        let mut scales = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(num_elements);
            let block_values = &values[start..end];

            let max_abs = block_values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 1e-10 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Quantize values to nibbles
        let mut nibbles = Vec::with_capacity(num_elements);
        for (i, &value) in values.iter().enumerate() {
            let block_idx = i / block_size;
            let scale = scales[block_idx];
            let quantized = ((value / scale).round() as i32 + 8).clamp(0, 15) as u8;
            nibbles.push(quantized);
        }

        // Build output: [FP16 scales][packed nibbles]
        let mut data = Vec::new();

        // Write scales as FP16
        for scale in &scales {
            let f16_val = f16::from_f32(*scale);
            data.extend_from_slice(&f16_val.to_le_bytes());
        }

        // Pack nibbles (2 per byte, low nibble first)
        let packed_size = (num_elements + 1) / 2;
        for i in 0..packed_size {
            let low = nibbles.get(i * 2).copied().unwrap_or(0);
            let high = nibbles.get(i * 2 + 1).copied().unwrap_or(0);
            data.push(low | (high << 4));
        }

        data
    }

    /// Helper to dequantize INT4 data (mirrors HctLoader logic).
    fn dequantize_int4(data: &[u8], num_elements: usize, block_size: usize) -> Vec<f32> {
        let num_blocks = (num_elements + block_size - 1) / block_size;
        let scales_bytes = num_blocks * 2;

        // Parse scales
        let scales: Vec<f32> = data[..scales_bytes]
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
            .collect();

        let packed_data = &data[scales_bytes..];

        // Dequantize
        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                packed_data[byte_idx] & 0x0F
            } else {
                (packed_data[byte_idx] >> 4) & 0x0F
            };

            let centered = (nibble as i32) - 8;
            let block_idx = i / block_size;
            let scale = scales.get(block_idx).copied().unwrap_or(1.0);
            values.push((centered as f32) * scale);
        }

        values
    }

    #[test]
    fn test_int4_dequant_zeros() {
        // All zeros should quantize to nibble=8, dequantize to 0
        let values = vec![0.0f32; 64];
        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 64, 32);

        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-6, "Element {} should be ~0, got {}", i, v);
        }
    }

    #[test]
    fn test_int4_dequant_positive_values() {
        // Test positive values
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 32, 32);

        // Check that values are approximately preserved
        for (i, (&orig, &dequant)) in values.iter().zip(result.iter()).enumerate() {
            let error = (orig - dequant).abs();
            let max_error = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 7.0;
            assert!(
                error <= max_error + 1e-6,
                "Element {}: orig={}, dequant={}, error={} > max_error={}",
                i,
                orig,
                dequant,
                error,
                max_error
            );
        }
    }

    #[test]
    fn test_int4_dequant_negative_values() {
        // Test negative values
        let values: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.1).collect();
        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 32, 32);

        // Check that negative values are preserved
        for (i, (&orig, &dequant)) in values.iter().zip(result.iter()).enumerate() {
            let error = (orig - dequant).abs();
            let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let max_error = max_abs / 7.0;
            assert!(
                error <= max_error + 1e-6,
                "Element {}: orig={}, dequant={}, error={}",
                i,
                orig,
                dequant,
                error
            );
        }
    }

    #[test]
    fn test_int4_dequant_mixed_values() {
        // Test mixed positive/negative (typical weight distribution)
        let values: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.01).collect();
        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 64, 32);

        // Verify mean is approximately preserved
        let orig_mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let dequant_mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(
            (orig_mean - dequant_mean).abs() < 0.01,
            "Mean not preserved: orig={}, dequant={}",
            orig_mean,
            dequant_mean
        );
    }

    #[test]
    fn test_int4_dequant_layernorm_values() {
        // LayerNorm weights are typically around 0.2-0.4 with small variation
        // This tests the case that was previously buggy
        let values: Vec<f32> = (0..32).map(|i| 0.28 + (i as f32 - 16.0) * 0.005).collect();

        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 32, 32);

        // Verify mean is close to original
        let orig_mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let dequant_mean: f32 = result.iter().sum::<f32>() / result.len() as f32;

        assert!(
            (orig_mean - dequant_mean).abs() < 0.05,
            "LayerNorm mean not preserved: orig={:.4}, dequant={:.4}",
            orig_mean,
            dequant_mean
        );

        // Verify no value is exactly 1.0 (the previous bug returned all ones)
        let all_ones = result.iter().all(|&v| (v - 1.0).abs() < 1e-6);
        assert!(!all_ones, "LayerNorm values should NOT all be 1.0");
    }

    #[test]
    fn test_int4_dequant_extreme_nibbles() {
        // Test minimum nibble (0 -> -8 * scale)
        // Test maximum nibble (15 -> +7 * scale)
        let scale = 0.1f32;
        let min_val = -8.0 * scale; // nibble 0
        let max_val = 7.0 * scale; // nibble 15

        let values = vec![min_val, max_val];
        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 2, 32);

        assert!(
            (result[0] - min_val).abs() < 0.02,
            "Min value: expected {}, got {}",
            min_val,
            result[0]
        );
        assert!(
            (result[1] - max_val).abs() < 0.02,
            "Max value: expected {}, got {}",
            max_val,
            result[1]
        );
    }

    #[test]
    fn test_int4_dequant_multi_block() {
        // Test with multiple blocks having different scales
        let mut values = Vec::new();
        // Block 0: small values (scale ~0.01)
        values.extend((0..32).map(|i| (i as f32 - 16.0) * 0.001));
        // Block 1: larger values (scale ~0.1)
        values.extend((0..32).map(|i| (i as f32 - 16.0) * 0.01));

        let data = create_int4_data(&values, 32);
        let result = dequantize_int4(&data, 64, 32);

        // Check that both blocks are reasonably dequantized
        let block0_orig_std: f32 = {
            let mean = values[..32].iter().sum::<f32>() / 32.0;
            (values[..32].iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 32.0).sqrt()
        };
        let block0_dequant_std: f32 = {
            let mean = result[..32].iter().sum::<f32>() / 32.0;
            (result[..32].iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 32.0).sqrt()
        };

        // Std should be roughly preserved (within 50% due to quantization noise)
        assert!(
            block0_dequant_std > block0_orig_std * 0.5
                && block0_dequant_std < block0_orig_std * 1.5,
            "Block 0 std not preserved: orig={}, dequant={}",
            block0_orig_std,
            block0_dequant_std
        );
    }

    #[test]
    fn test_int4_nibble_packing() {
        // Verify nibble packing: low nibble first, high nibble second
        let values = vec![0.7, -0.8]; // Should map to nibbles ~15 and ~0
        let scale = 0.1f32;

        // Manually create data with known nibbles
        let mut data = Vec::new();
        // Scale as FP16
        data.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        // Packed nibbles: low=15 (value 7*scale), high=0 (value -8*scale)
        data.push(15 | (0 << 4)); // 0x0F

        let result = dequantize_int4(&data, 2, 32);

        assert!(
            (result[0] - 7.0 * scale).abs() < 0.01,
            "First nibble (low): expected {}, got {}",
            7.0 * scale,
            result[0]
        );
        assert!(
            (result[1] - (-8.0 * scale)).abs() < 0.01,
            "Second nibble (high): expected {}, got {}",
            -8.0 * scale,
            result[1]
        );
    }

    #[test]
    fn test_filename_to_tensor_name_basic() {
        assert_eq!(
            filename_to_tensor_name("model_embed_tokens_weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            filename_to_tensor_name("model_layers_0_self_attn_q_proj_weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            filename_to_tensor_name("model_norm_weight"),
            "model.norm.weight"
        );
        assert_eq!(filename_to_tensor_name("lm_head_weight"), "lm_head.weight");
    }

    #[test]
    fn test_filename_to_tensor_name_layernorm() {
        assert_eq!(
            filename_to_tensor_name("model_layers_0_input_layernorm_weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            filename_to_tensor_name("model_layers_0_post_attention_layernorm_weight"),
            "model.layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_filename_to_tensor_name_mlp() {
        assert_eq!(
            filename_to_tensor_name("model_layers_0_mlp_gate_proj_weight"),
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            filename_to_tensor_name("model_layers_0_mlp_up_proj_weight"),
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            filename_to_tensor_name("model_layers_0_mlp_down_proj_weight"),
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    // ==================== TDD Phase 1: Constant Integrity ====================
    // GPU-CODEC-PIPELINE-TDD.md §1

    #[test]
    fn test_int4_block_size_matches_quantizer() {
        // DD-1: The GPU dequant block size must match the CPU quantizer default.
        // Both write/read INT4 data with per-block scales; if they disagree on
        // block_size, the scale layout is misinterpreted.
        use crate::gpu_dequant::INT4_BLOCK_SIZE;
        use crate::quantize::DEFAULT_BLOCK_SIZE;

        assert_eq!(
            INT4_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
            "GPU INT4_BLOCK_SIZE ({}) must equal CPU DEFAULT_BLOCK_SIZE ({})",
            INT4_BLOCK_SIZE, DEFAULT_BLOCK_SIZE
        );
    }

    #[test]
    fn test_hct_int4_dequant_block_size_128() {
        // DD-1 regression: data created with block_size=128 must be read
        // with block_size=128. Using 32 misinterprets the scale layout.
        use crate::quantize::DEFAULT_BLOCK_SIZE;

        let values: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
        let data = create_int4_data(&values, DEFAULT_BLOCK_SIZE);

        // Dequantize with the correct block_size (128)
        let result = dequantize_int4(&data, 256, DEFAULT_BLOCK_SIZE);

        // Verify values are approximately preserved
        for (i, (&orig, &deq)) in values.iter().zip(result.iter()).enumerate() {
            let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let max_error = max_abs / 7.0;
            assert!(
                (orig - deq).abs() <= max_error + 0.01,
                "Block_size=128 dequant error at index {}: orig={}, got={}",
                i,
                orig,
                deq
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_hct_int4_wrong_block_size_panics() {
        // DD-1: Reading block_size=128 data with block_size=32 panics because
        // it expects 8 scale entries (256/32) but only 2 exist (256/128).
        // The reader overruns the packed data buffer.
        use crate::quantize::DEFAULT_BLOCK_SIZE;

        let values: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
        let data = create_int4_data(&values, DEFAULT_BLOCK_SIZE);

        // This panics: reads 16 bytes as scales (expects 8 blocks) but only
        // 4 bytes of scales exist. The packed data offset is wrong, causing OOB.
        let _ = dequantize_int4(&data, 256, 32);
    }

    // ============ Phase 1: Cross-Validation (HCT vs quantize.rs) ============

    /// Cross-validate: data quantized by quantize.rs (Int4Symmetric) must
    /// produce identical dequantized output when read through the HCT path.
    ///
    /// This tests trust boundary §4 (Quantization Math) from GPU-CODEC-PIPELINE-TDD.md.
    #[test]
    fn test_hct_dequant_matches_quantizer_int4_symmetric() {
        use crate::quantize::{QuantizeConfig, QuantizeFormat, Quantizer, DEFAULT_BLOCK_SIZE};
        use candle_core::{DType, Device, Tensor};

        // 1. Create known input data (2 blocks of 128 = 256 elements)
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
        let tensor = Tensor::from_vec(input.clone(), &[256], &Device::Cpu).unwrap();

        // 2. Quantize with quantize.rs
        let quantizer = Quantizer::int4_symmetric();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();
        assert_eq!(quantized.block_size, DEFAULT_BLOCK_SIZE);

        // 3. Dequantize with quantize.rs (ground truth)
        let reference = quantizer.dequantize(&quantized).unwrap();
        let ref_values: Vec<f32> = reference.to_vec1().unwrap();

        // 4. Convert QuantizedTensor to HCT byte layout: [FP16 scales][packed data]
        let mut hct_bytes = Vec::new();
        for scale in &quantized.scales {
            hct_bytes.extend_from_slice(&scale.to_le_bytes());
        }
        hct_bytes.extend_from_slice(&quantized.data);

        // 5. Dequantize with HCT path
        let hct_values = dequantize_int4(&hct_bytes, 256, DEFAULT_BLOCK_SIZE);

        // 6. Assert identical (both use FP16 scales, so no precision gap)
        assert_eq!(ref_values.len(), hct_values.len(), "length mismatch");
        for (i, (r, h)) in ref_values.iter().zip(hct_values.iter()).enumerate() {
            assert!(
                (r - h).abs() < 1e-7,
                "Element {}: quantize.rs={} vs hct.rs={} (diff={})",
                i,
                r,
                h,
                (r - h).abs()
            );
        }
    }

    /// Cross-validate with multiple block sizes worth of data (stress test).
    /// Uses 1280 elements = 10 blocks of 128.
    #[test]
    fn test_hct_dequant_matches_quantizer_multi_block() {
        use crate::quantize::{Quantizer, DEFAULT_BLOCK_SIZE};
        use candle_core::{Device, Tensor};

        let n = DEFAULT_BLOCK_SIZE * 10; // 1280 elements
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let block = i / DEFAULT_BLOCK_SIZE;
                let pos = i % DEFAULT_BLOCK_SIZE;
                // Different scale per block so each block has a unique scale factor
                ((pos as f32) - 64.0) * 0.001 * ((block + 1) as f32)
            })
            .collect();

        let tensor = Tensor::from_vec(input.clone(), &[n], &Device::Cpu).unwrap();
        let quantizer = Quantizer::int4_symmetric();
        let quantized = quantizer.quantize_tensor(&tensor).unwrap();

        // Ground truth from quantize.rs
        let reference = quantizer.dequantize(&quantized).unwrap();
        let ref_values: Vec<f32> = reference.to_vec1().unwrap();

        // HCT path
        let mut hct_bytes = Vec::new();
        for scale in &quantized.scales {
            hct_bytes.extend_from_slice(&scale.to_le_bytes());
        }
        hct_bytes.extend_from_slice(&quantized.data);
        let hct_values = dequantize_int4(&hct_bytes, n, DEFAULT_BLOCK_SIZE);

        assert_eq!(ref_values.len(), hct_values.len());
        let max_diff: f32 = ref_values
            .iter()
            .zip(hct_values.iter())
            .map(|(r, h)| (r - h).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-7,
            "Max diff between quantize.rs and hct.rs dequant: {} (should be 0)",
            max_diff
        );
    }

    /// Cross-validate HCT INT4 format: data created by create_int4_data() and
    /// read back by dequantize_int4() must agree with quantize.rs on nibble layout.
    /// This validates trust boundary §2 (Data Format Correctness).
    #[test]
    fn test_hct_nibble_format_matches_quantizer_packing() {
        use crate::quantize::DEFAULT_BLOCK_SIZE;

        // Create data with hct.rs helper and quantize.rs, compare packed bytes
        let values: Vec<f32> = (0..DEFAULT_BLOCK_SIZE)
            .map(|i| ((i as f32) - 64.0) * 0.01)
            .collect();

        // HCT path: create_int4_data does its own quantization
        let hct_data = create_int4_data(&values, DEFAULT_BLOCK_SIZE);

        // Parse HCT bytes: [2 bytes scale][packed nibbles]
        let num_blocks = 1; // exactly one block
        let hct_scale_bytes = &hct_data[..num_blocks * 2];
        let hct_packed = &hct_data[num_blocks * 2..];

        // Verify packed nibble low-first convention: for each byte,
        // element 2*i is in low nibble, element 2*i+1 is in high nibble
        for byte_idx in 0..hct_packed.len() {
            let byte = hct_packed[byte_idx];
            let low = byte & 0x0F;
            let high = (byte >> 4) & 0x0F;
            // Both should be in valid nibble range [0, 15]
            assert!(
                low <= 15,
                "Low nibble at byte {} out of range: {}",
                byte_idx,
                low
            );
            assert!(
                high <= 15,
                "High nibble at byte {} out of range: {}",
                byte_idx,
                high
            );
        }

        // Round-trip: create → dequant should not panic and produce sane values
        let result = dequantize_int4(&hct_data, DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE);
        assert_eq!(result.len(), DEFAULT_BLOCK_SIZE);
        // Original values are in [-0.64, 0.63], dequantized should be in similar range
        for (i, v) in result.iter().enumerate() {
            assert!(
                v.abs() < 2.0,
                "Element {} out of expected range: {} (original: {})",
                i,
                v,
                values[i]
            );
        }
    }
}
