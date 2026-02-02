//! HCT V3 Safetensors Loader
//!
//! Loads model weights from HCT V3 format safetensors files.
//! HCT V3 uses spectral compression (DCT) with zstd, achieving ~10x compression
//! on neural network weights while preserving model quality.
//!
//! ## Format
//!
//! HCT V3 files are standard safetensors with:
//! - `dtype: "hct_v3"` for compressed tensors
//! - Data is zstd-compressed spectral fragments
//!
//! ## Usage
//!
//! ```ignore
//! use abaddon::hct_v3::HctV3Loader;
//!
//! let loader = HctV3Loader::open("/path/to/model.safetensors")?;
//! let tensor = loader.load_tensor("model.layers.0.self_attn.q_proj.weight", &device)?;
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{Device, Tensor};
use haagenti::compressive::CompressiveSpectralDecoder;
use haagenti::holotensor::HoloFragment;
use parking_lot::RwLock;

/// Error type for HCT V3 loading.
#[derive(Debug, thiserror::Error)]
pub enum HctV3Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tensor not found: {name}")]
    TensorNotFound { name: String },

    #[error("Invalid HCT V3 format: {message}")]
    InvalidFormat { message: String },

    #[error("Decompression error: {message}")]
    Decompression { message: String },

    #[error("Tensor creation error: {0}")]
    Tensor(#[from] candle_core::Error),
}

/// Tensor metadata from safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Data type (should be "hct_v3" for compressed).
    pub dtype: String,
    /// Start offset in data section.
    pub start_offset: usize,
    /// End offset in data section.
    pub end_offset: usize,
}

impl TensorInfo {
    /// Returns true if this tensor is HCT V3 compressed.
    pub fn is_hct_v3(&self) -> bool {
        self.dtype == "hct_v3"
    }

    /// Returns the compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.end_offset - self.start_offset
    }

    /// Returns the original size in bytes (assuming f32).
    pub fn original_size(&self) -> usize {
        self.shape.iter().product::<usize>() * 4
    }

    /// Returns the compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        self.original_size() as f64 / self.compressed_size() as f64
    }
}

/// HCT V3 Safetensors loader.
///
/// Provides efficient loading and decompression of HCT V3 compressed tensors.
pub struct HctV3Loader {
    /// Path to the safetensors file.
    path: PathBuf,
    /// Header size in bytes.
    header_size: usize,
    /// Tensor metadata.
    tensors: HashMap<String, TensorInfo>,
    /// Cached decompressed tensors.
    cache: Arc<RwLock<HashMap<String, Tensor>>>,
}

impl HctV3Loader {
    /// Opens an HCT V3 safetensors file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or parsed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HctV3Error> {
        let path = path.as_ref().to_path_buf();
        let mut file = BufReader::new(File::open(&path)?);

        // Read header size (8 bytes, little-endian)
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        // Read and parse header JSON
        let mut header_json = vec![0u8; header_size];
        file.read_exact(&mut header_json)?;
        let header_str = String::from_utf8_lossy(&header_json);
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str)?;

        // Parse tensor metadata
        let mut tensors = HashMap::new();
        for (name, info) in header {
            if name == "__metadata__" {
                continue;
            }

            let dtype = info
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            let data_offsets = info.get("data_offsets").and_then(|v| v.as_array());
            let (start_offset, end_offset) = match data_offsets {
                Some(arr) if arr.len() >= 2 => {
                    let start = arr[0].as_u64().unwrap_or(0) as usize;
                    let end = arr[1].as_u64().unwrap_or(0) as usize;
                    (start, end)
                }
                _ => (0, 0),
            };

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    dtype,
                    start_offset,
                    end_offset,
                },
            );
        }

        Ok(Self {
            path,
            header_size,
            tensors,
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Returns the path to the safetensors file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of tensors in the file.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Returns all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Returns tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Returns iterator over all tensor info.
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.values()
    }

    /// Returns total compressed size in bytes.
    pub fn total_compressed_size(&self) -> usize {
        self.tensors.values().map(|t| t.compressed_size()).sum()
    }

    /// Returns total original size in bytes.
    pub fn total_original_size(&self) -> usize {
        self.tensors.values().map(|t| t.original_size()).sum()
    }

    /// Returns average compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        self.total_original_size() as f64 / self.total_compressed_size() as f64
    }

    /// Loads and decompresses a single tensor.
    ///
    /// Uses caching to avoid repeated decompression.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found or decompression fails.
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor, HctV3Error> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(tensor) = cache.get(name) {
                return Ok(tensor.clone());
            }
        }

        // Get tensor info
        let info = self.tensors.get(name).ok_or_else(|| HctV3Error::TensorNotFound {
            name: name.to_string(),
        })?;

        // Decompress
        let data = self.decompress_tensor(info)?;

        // Create tensor
        let tensor = Tensor::from_vec(data, info.shape.as_slice(), device)?;

        // Cache the result
        {
            let mut cache = self.cache.write();
            cache.insert(name.to_string(), tensor.clone());
        }

        Ok(tensor)
    }

    /// Loads a tensor without caching (for memory-constrained scenarios).
    pub fn load_tensor_uncached(&self, name: &str, device: &Device) -> Result<Tensor, HctV3Error> {
        let info = self.tensors.get(name).ok_or_else(|| HctV3Error::TensorNotFound {
            name: name.to_string(),
        })?;

        let data = self.decompress_tensor(info)?;
        Ok(Tensor::from_vec(data, info.shape.as_slice(), device)?)
    }

    /// Decompresses a tensor to f32 data.
    fn decompress_tensor(&self, info: &TensorInfo) -> Result<Vec<f32>, HctV3Error> {
        if !info.is_hct_v3() {
            return Err(HctV3Error::InvalidFormat {
                message: format!("Expected dtype 'hct_v3', got '{}'", info.dtype),
            });
        }

        // Read compressed data from file
        let mut file = BufReader::new(File::open(&self.path)?);
        let data_start = 8 + self.header_size as u64;
        file.seek(SeekFrom::Start(data_start + info.start_offset as u64))?;

        let mut zstd_data = vec![0u8; info.compressed_size()];
        file.read_exact(&mut zstd_data)?;

        // Step 1: Decompress zstd (using standard zstd crate, matches turbo pipeline)
        let fragment_data = zstd::decode_all(&zstd_data[..]).map_err(|e| HctV3Error::Decompression {
            message: format!("zstd decompression failed: {}", e),
        })?;

        // Step 2: Parse fragments
        // Format: [num_fragments: u16] then per fragment: [index: u16][flags: u16][checksum: u64][data_len: u32][data]
        if fragment_data.len() < 2 {
            return Err(HctV3Error::InvalidFormat {
                message: "Fragment data too short".to_string(),
            });
        }

        let num_fragments = u16::from_le_bytes([fragment_data[0], fragment_data[1]]) as usize;
        let mut offset = 2;
        let mut fragments = Vec::with_capacity(num_fragments);

        for _ in 0..num_fragments {
            if offset + 16 > fragment_data.len() {
                return Err(HctV3Error::InvalidFormat {
                    message: "Truncated fragment header".to_string(),
                });
            }

            let index = u16::from_le_bytes([fragment_data[offset], fragment_data[offset + 1]]);
            offset += 2;
            let flags = u16::from_le_bytes([fragment_data[offset], fragment_data[offset + 1]]);
            offset += 2;
            let checksum = u64::from_le_bytes([
                fragment_data[offset],
                fragment_data[offset + 1],
                fragment_data[offset + 2],
                fragment_data[offset + 3],
                fragment_data[offset + 4],
                fragment_data[offset + 5],
                fragment_data[offset + 6],
                fragment_data[offset + 7],
            ]);
            offset += 8;
            let data_len = u32::from_le_bytes([
                fragment_data[offset],
                fragment_data[offset + 1],
                fragment_data[offset + 2],
                fragment_data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + data_len > fragment_data.len() {
                return Err(HctV3Error::InvalidFormat {
                    message: "Truncated fragment data".to_string(),
                });
            }

            let frag_data = fragment_data[offset..offset + data_len].to_vec();
            offset += data_len;

            fragments.push(HoloFragment {
                index,
                flags,
                checksum,
                data: frag_data,
            });
        }

        // Step 3: Decode using CompressiveSpectralDecoder
        let mut decoder = CompressiveSpectralDecoder::new();

        if fragments.is_empty() {
            return Err(HctV3Error::InvalidFormat {
                message: "No fragments to decode".to_string(),
            });
        }

        // Add essentials fragment (index 0)
        decoder.add_essentials(&fragments[0]).map_err(|e| HctV3Error::Decompression {
            message: format!("Failed to add essentials: {}", e),
        })?;

        // Add detail fragments
        for frag in &fragments[1..] {
            decoder.add_detail(frag).map_err(|e| HctV3Error::Decompression {
                message: format!("Failed to add detail fragment: {}", e),
            })?;
        }

        // Reconstruct
        let decoded = decoder.reconstruct().map_err(|e| HctV3Error::Decompression {
            message: format!("Reconstruction failed: {}", e),
        })?;

        // Verify size
        let expected_size: usize = info.shape.iter().product();
        if decoded.len() != expected_size {
            return Err(HctV3Error::InvalidFormat {
                message: format!(
                    "Size mismatch: decoded {} elements, expected {}",
                    decoded.len(),
                    expected_size
                ),
            });
        }

        Ok(decoded)
    }

    /// Loads all tensors into a map.
    ///
    /// This is useful for building a complete model from compressed weights.
    pub fn load_all(&self, device: &Device) -> Result<HashMap<String, Tensor>, HctV3Error> {
        let mut result = HashMap::new();
        for name in self.tensor_names() {
            let tensor = self.load_tensor(name, device)?;
            result.insert(name.to_string(), tensor);
        }
        Ok(result)
    }

    /// Loads tensors matching a prefix.
    ///
    /// Useful for loading a single layer at a time.
    pub fn load_prefix(
        &self,
        prefix: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>, HctV3Error> {
        let mut result = HashMap::new();
        for (name, _) in &self.tensors {
            if name.starts_with(prefix) {
                let tensor = self.load_tensor(name, device)?;
                result.insert(name.clone(), tensor);
            }
        }
        Ok(result)
    }

    /// Clears the tensor cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let cached_count = cache.len();
        let cached_bytes: usize = cache
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum();
        (cached_count, cached_bytes)
    }
}

impl std::fmt::Debug for HctV3Loader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HctV3Loader")
            .field("path", &self.path)
            .field("num_tensors", &self.tensors.len())
            .field("compression_ratio", &format!("{:.1}x", self.compression_ratio()))
            .finish()
    }
}

/// Convenience function to load a single tensor from an HCT V3 file.
pub fn load_hct_v3_tensor(
    path: impl AsRef<Path>,
    tensor_name: &str,
    device: &Device,
) -> Result<Tensor, HctV3Error> {
    let loader = HctV3Loader::open(path)?;
    loader.load_tensor(tensor_name, device)
}

/// Convenience function to load all tensors from an HCT V3 file.
pub fn load_hct_v3_all(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<HashMap<String, Tensor>, HctV3Error> {
    let loader = HctV3Loader::open(path)?;
    loader.load_all(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info_compression_ratio() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![1024, 1024],
            dtype: "hct_v3".to_string(),
            start_offset: 0,
            end_offset: 400_000, // ~10x compression
        };

        assert!(info.is_hct_v3());
        assert_eq!(info.original_size(), 1024 * 1024 * 4);
        assert_eq!(info.compressed_size(), 400_000);
        assert!((info.compression_ratio() - 10.49).abs() < 0.1);
    }
}
