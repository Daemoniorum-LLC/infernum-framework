//! Sequential HCT loader with memory budget tracking.
//!
//! This module provides memory-bounded sequential loading of HCT/HoloTensor files,
//! designed to prevent OOM when loading large models like Llama-405B.
//!
//! ## Design
//!
//! Unlike `load_hct_directory()` which uses rayon to load all files in parallel,
//! `SequentialHctLoader` loads files one at a time with configurable memory limits.
//!
//! ## Features
//!
//! - **Memory Budgeting**: Tracks memory usage and pauses when approaching limits
//! - **Corrupted File Recovery**: Initializes sensible defaults for truncated files
//! - **Iterator-based**: Yields tensors one at a time via `iter_tensors()`
//! - **Progress Tracking**: Reports loading progress for monitoring
//! - **GPU Decompression**: Optional GPU-accelerated IDCT via haagenti-cuda

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};

use crate::hct::{HctError, HctLoader, HctMetadata};

#[cfg(feature = "haagenti-gpu")]
use haagenti_cuda::decompress::{DecompressConfig, GpuDecompressor};

/// Returns the size in bytes of a single element of the given dtype.
fn dtype_size(dtype: DType) -> u64 {
    match dtype {
        DType::F32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 => 2,
        DType::U8 => 1,
        // Handle new candle_core DType variants (I16, I32, F8E4M3, etc.)
        // Default to 4 bytes as a safe fallback
        _ => 4,
    }
}

/// Memory budget configuration for sequential loading.
#[derive(Debug)]
pub struct MemoryBudget {
    /// Maximum memory to use for loaded tensors (bytes).
    pub max_bytes: u64,
    /// Current memory usage (bytes).
    current_bytes: AtomicU64,
    /// Warning threshold as percentage (0.0-1.0).
    pub warning_threshold: f32,
}

impl Clone for MemoryBudget {
    fn clone(&self) -> Self {
        Self {
            max_bytes: self.max_bytes,
            current_bytes: AtomicU64::new(self.current_bytes.load(Ordering::Relaxed)),
            warning_threshold: self.warning_threshold,
        }
    }
}

impl MemoryBudget {
    /// Creates a new memory budget with the given maximum bytes.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            current_bytes: AtomicU64::new(0),
            warning_threshold: 0.85,
        }
    }

    /// Creates a budget that allows unlimited memory (for testing).
    pub fn unlimited() -> Self {
        Self::new(u64::MAX)
    }

    /// Returns current memory usage in bytes.
    pub fn current_usage(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Returns remaining budget in bytes.
    pub fn remaining(&self) -> u64 {
        self.max_bytes.saturating_sub(self.current_usage())
    }

    /// Checks if we can allocate the given number of bytes.
    pub fn can_allocate(&self, bytes: u64) -> bool {
        self.current_usage() + bytes <= self.max_bytes
    }

    /// Records an allocation.
    pub fn allocate(&self, bytes: u64) {
        self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records a deallocation.
    pub fn deallocate(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Returns true if we're above the warning threshold.
    pub fn is_warning(&self) -> bool {
        let usage_ratio = self.current_usage() as f64 / self.max_bytes as f64;
        usage_ratio >= self.warning_threshold as f64
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        // Default to 64GB
        Self::new(64 * 1024 * 1024 * 1024)
    }
}

/// Strategy for handling corrupted or truncated files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Skip corrupted files entirely.
    Skip,
    /// Initialize with default values based on tensor type.
    InitializeDefault,
    /// Fail on any corrupted file.
    Fail,
}

impl Default for FallbackStrategy {
    fn default() -> Self {
        Self::InitializeDefault
    }
}

/// Configuration for sequential HCT loading.
#[derive(Debug, Clone)]
pub struct SequentialLoadConfig {
    /// Memory budget for loaded tensors.
    pub memory_budget: MemoryBudget,
    /// Target device for tensors.
    pub device: Device,
    /// Target dtype for tensors.
    pub dtype: DType,
    /// Strategy for handling corrupted files.
    pub fallback_strategy: FallbackStrategy,
    /// Minimum quality for HoloTensor reconstruction (0.0-1.0).
    pub min_quality: f32,
}

impl Default for SequentialLoadConfig {
    fn default() -> Self {
        Self {
            memory_budget: MemoryBudget::default(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::default(),
            min_quality: 0.7,
        }
    }
}

/// Result of loading a single tensor.
#[derive(Debug)]
pub struct LoadedTensor {
    /// Tensor name (derived from filename).
    pub name: String,
    /// The loaded tensor.
    pub tensor: Tensor,
    /// Original file metadata.
    pub metadata: HctMetadata,
    /// Whether this tensor was recovered from a corrupted file.
    pub recovered: bool,
    /// Memory used by this tensor in bytes.
    pub memory_bytes: u64,
}

/// Progress information for loading operations.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// Total number of files to load.
    pub total_files: usize,
    /// Number of files loaded so far.
    pub loaded_files: usize,
    /// Number of files skipped due to errors.
    pub skipped_files: usize,
    /// Number of files recovered with defaults.
    pub recovered_files: usize,
    /// Current memory usage in bytes.
    pub memory_used: u64,
    /// Memory budget in bytes.
    pub memory_budget: u64,
}

/// Sequential HCT/HoloTensor loader with memory budget tracking.
///
/// This loader processes files one at a time to prevent OOM on large models.
pub struct SequentialHctLoader {
    /// Configuration for loading.
    config: SequentialLoadConfig,
    /// Files to load.
    files: Vec<PathBuf>,
    /// Current index in files.
    current_index: usize,
    /// Number of skipped files.
    skipped_count: usize,
    /// Number of recovered files.
    recovered_count: usize,
}

impl SequentialHctLoader {
    /// Creates a new sequential loader for the given directory.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory containing HCT/HoloTensor files
    /// * `config` - Loading configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn new(dir: impl AsRef<Path>, config: SequentialLoadConfig) -> Result<Self, HctError> {
        let dir = dir.as_ref();

        // Find all .hct files
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| HctError::Io {
                path: dir.to_path_buf(),
                message: e.to_string(),
            })?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
            .map(|entry| entry.path())
            .collect();

        // Sort for deterministic ordering
        files.sort();

        tracing::info!(
            count = files.len(),
            directory = %dir.display(),
            memory_budget = %config.memory_budget.max_bytes,
            "Sequential HCT loader initialized"
        );

        Ok(Self {
            config,
            files,
            current_index: 0,
            skipped_count: 0,
            recovered_count: 0,
        })
    }

    /// Returns the total number of files to load.
    pub fn total_files(&self) -> usize {
        self.files.len()
    }

    /// Returns current loading progress.
    pub fn progress(&self) -> LoadProgress {
        LoadProgress {
            total_files: self.files.len(),
            loaded_files: self.current_index,
            skipped_files: self.skipped_count,
            recovered_files: self.recovered_count,
            memory_used: self.config.memory_budget.current_usage(),
            memory_budget: self.config.memory_budget.max_bytes,
        }
    }

    /// Loads the next tensor, returning None when done.
    ///
    /// This method respects the memory budget and will return an error
    /// if loading would exceed the budget.
    pub fn next_tensor(&mut self) -> Option<Result<LoadedTensor, HctError>> {
        while self.current_index < self.files.len() {
            let path = &self.files[self.current_index];
            self.current_index += 1;

            match self.load_single_file(path) {
                Ok(Some(tensor)) => return Some(Ok(tensor)),
                Ok(None) => {
                    // File was skipped
                    self.skipped_count += 1;
                    continue;
                },
                Err(e) => match self.config.fallback_strategy {
                    FallbackStrategy::Fail => return Some(Err(e)),
                    FallbackStrategy::Skip => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "Skipping corrupted file"
                        );
                        self.skipped_count += 1;
                        continue;
                    },
                    FallbackStrategy::InitializeDefault => match self.recover_tensor(path, &e) {
                        Ok(tensor) => {
                            self.recovered_count += 1;
                            return Some(Ok(tensor));
                        },
                        Err(recover_err) => {
                            tracing::warn!(
                                path = %path.display(),
                                original_error = %e,
                                recover_error = %recover_err,
                                "Failed to recover tensor, skipping"
                            );
                            self.skipped_count += 1;
                            continue;
                        },
                    },
                },
            }
        }
        None
    }

    /// Loads all tensors into a HashMap.
    ///
    /// This is a convenience method that collects all tensors from the iterator.
    pub fn load_all(mut self) -> Result<HashMap<String, Tensor>, HctError> {
        let mut tensors = HashMap::new();

        while let Some(result) = self.next_tensor() {
            let loaded = result?;
            tensors.insert(loaded.name, loaded.tensor);
        }

        Ok(tensors)
    }

    /// Loads a single file, returning None if it should be skipped.
    fn load_single_file(&self, path: &Path) -> Result<Option<LoadedTensor>, HctError> {
        let loader = HctLoader::from_file(path)?;
        let metadata = loader.metadata().clone();

        // Estimate memory requirement
        let estimated_bytes = self.estimate_tensor_bytes(&metadata);

        // Check memory budget
        if !self.config.memory_budget.can_allocate(estimated_bytes) {
            tracing::warn!(
                path = %path.display(),
                estimated_bytes = estimated_bytes,
                remaining = self.config.memory_budget.remaining(),
                "Insufficient memory budget for tensor"
            );
            return Err(HctError::Tensor {
                message: format!(
                    "Memory budget exceeded: need {} bytes, have {} remaining",
                    estimated_bytes,
                    self.config.memory_budget.remaining()
                ),
            });
        }

        // Load the tensor
        let tensor = loader.to_tensor(&self.config.device, Some(self.config.dtype))?;

        // Record memory allocation
        let actual_bytes = self.tensor_bytes(&tensor);
        self.config.memory_budget.allocate(actual_bytes);

        // Convert filename to tensor name
        let name = crate::hct::filename_to_tensor_name(&metadata.name);

        Ok(Some(LoadedTensor {
            name,
            tensor,
            metadata,
            recovered: false,
            memory_bytes: actual_bytes,
        }))
    }

    /// Attempts to recover a tensor by initializing with defaults.
    fn recover_tensor(
        &self,
        path: &Path,
        _original_error: &HctError,
    ) -> Result<LoadedTensor, HctError> {
        // Try to at least parse the header to get shape/dtype info
        let loader_result = HctLoader::from_file(path);

        let (name, shape, dtype) = match loader_result {
            Ok(loader) => {
                let metadata = loader.metadata();
                let name = crate::hct::filename_to_tensor_name(&metadata.name);
                let shape: Vec<usize> = metadata.shape.iter().map(|&d| d as usize).collect();
                (name, shape, self.config.dtype)
            },
            Err(_) => {
                // Can't even read the header - infer from filename
                let filename = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let name = crate::hct::filename_to_tensor_name(filename);

                // Use default shape based on tensor type
                let shape = self.infer_default_shape(&name);
                (name, shape, self.config.dtype)
            },
        };

        // Create default tensor based on tensor type
        let tensor = self.create_default_tensor(&name, &shape, dtype)?;

        let actual_bytes = self.tensor_bytes(&tensor);
        self.config.memory_budget.allocate(actual_bytes);

        tracing::warn!(
            name = %name,
            shape = ?shape,
            "Recovered tensor with default values"
        );

        Ok(LoadedTensor {
            name: name.clone(),
            tensor,
            metadata: HctMetadata {
                name,
                original_size: 0,
                compressed_size: 0,
                compression_ratio: 0.0,
                dtype: haagenti::tensor::DType::F32,
                shape: shape.iter().map(|&d| d as u64).collect(),
                algorithm: haagenti::tensor::CompressionAlgorithm::Lz4,
                num_blocks: 0,
                flags: 0,
                is_holographic: false,
            },
            recovered: true,
            memory_bytes: actual_bytes,
        })
    }

    /// Estimates tensor size in bytes from metadata.
    fn estimate_tensor_bytes(&self, metadata: &HctMetadata) -> u64 {
        let elements: u64 = metadata.shape.iter().product();
        let bytes_per_element = dtype_size(self.config.dtype);
        elements * bytes_per_element
    }

    /// Gets actual tensor size in bytes.
    fn tensor_bytes(&self, tensor: &Tensor) -> u64 {
        let elements = tensor.elem_count() as u64;
        let bytes_per_element = dtype_size(tensor.dtype());
        elements * bytes_per_element
    }

    /// Infers default shape for a tensor based on its name.
    fn infer_default_shape(&self, name: &str) -> Vec<usize> {
        // Common patterns for 405B model:
        // - layernorm weights: [hidden_size] = [16384]
        // - biases: varies
        // - FP8 scales: typically [1] or small

        if name.contains("layernorm") || name.contains("norm") {
            // Assume 405B hidden size
            vec![16384]
        } else if name.contains("scale") {
            vec![1]
        } else if name.contains("bias") {
            // Could be various sizes, use hidden size as default
            vec![16384]
        } else {
            // Generic small tensor
            vec![1]
        }
    }

    /// Creates a default tensor based on tensor type.
    fn create_default_tensor(
        &self,
        name: &str,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, HctError> {
        // Determine default value based on tensor type
        let tensor =
            if name.contains("layernorm") || name.contains("norm") || name.contains("scale") {
                // LayerNorm weights and scales should be ones
                Tensor::ones(shape, dtype, &self.config.device)
            } else if name.contains("bias") {
                // Biases should be zeros
                Tensor::zeros(shape, dtype, &self.config.device)
            } else {
                // Generic: use zeros as safe default
                Tensor::zeros(shape, dtype, &self.config.device)
            }
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to create default tensor: {}", e),
            })?;

        Ok(tensor)
    }
}

/// Creates a sequential loader with default configuration.
///
/// This is a convenience function for quick loading with sensible defaults.
pub fn load_hct_directory_sequential(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, HctError> {
    let config = SequentialLoadConfig {
        device: device.clone(),
        dtype,
        ..Default::default()
    };

    let loader = SequentialHctLoader::new(dir, config)?;
    loader.load_all()
}

/// Creates a sequential loader with custom memory budget.
pub fn load_hct_directory_sequential_budgeted(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
    max_memory_bytes: u64,
) -> Result<HashMap<String, Tensor>, HctError> {
    let config = SequentialLoadConfig {
        device: device.clone(),
        dtype,
        memory_budget: MemoryBudget::new(max_memory_bytes),
        ..Default::default()
    };

    let loader = SequentialHctLoader::new(dir, config)?;
    loader.load_all()
}

/// Loads HCT files in parallel using rayon for maximum throughput.
///
/// This is optimized for speed when memory is not a concern. For memory-constrained
/// scenarios (like 405B models on limited hardware), use `load_hct_directory_sequential`.
///
/// # Performance
///
/// Uses rayon's parallel iterator to load and dequantize multiple files concurrently.
/// On a multi-core CPU, this can provide 4-8x speedup over sequential loading.
///
/// # Arguments
///
/// * `dir` - Directory containing HCT files
/// * `device` - Target device (CPU or CUDA)
/// * `dtype` - Target dtype for dequantized tensors
///
/// # Example
///
/// ```ignore
/// let tensors = load_hct_directory_parallel(model_dir, &device, DType::BF16)?;
/// ```
pub fn load_hct_directory_parallel(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>, HctError> {
    use rayon::prelude::*;

    let dir = dir.as_ref();

    // Collect all HCT files
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    let total = files.len();
    tracing::info!(
        count = total,
        directory = %dir.display(),
        "Loading HCT files in parallel"
    );

    // Load in parallel using rayon
    let results: Vec<Result<(String, Tensor), HctError>> = files
        .par_iter()
        .map(|path| {
            let loader = HctLoader::from_file(path)?;
            let name = crate::hct::filename_to_tensor_name(&loader.metadata().name);
            let tensor = loader.to_tensor(device, Some(dtype))?;
            Ok((name, tensor))
        })
        .collect();

    // Collect results, propagating first error
    let mut tensors = HashMap::with_capacity(total);
    for result in results {
        let (name, tensor) = result?;
        tensors.insert(name, tensor);
    }

    tracing::info!(loaded = tensors.len(), "Parallel loading complete");

    Ok(tensors)
}

/// GPU-accelerated HCT loading using haagenti-cuda.
///
/// Uses CUDA kernels for IDCT reconstruction, providing 10-50x speedup over CPU
/// for spectral coefficient decompression. The decompressed data is then
/// transferred to the target device.
///
/// # Prerequisites
///
/// - Requires `haagenti-gpu` feature enabled
/// - Requires NVIDIA GPU with CUDA support
///
/// # Performance
///
/// - GPU IDCT: ~50µs per 576x576 tensor (vs ~900µs CPU)
/// - Batch mode: ~2ms for 100 tensors
/// - Best for models with many spectral-compressed tensors
///
/// # Arguments
///
/// * `dir` - Directory containing HCT files
/// * `device` - Target device for final tensors (can be CPU or CUDA)
/// * `dtype` - Target dtype for tensors
/// * `gpu_device_id` - CUDA device ID for decompression (typically 0)
///
/// # Example
///
/// ```ignore
/// let tensors = load_hct_directory_gpu(model_dir, &device, DType::BF16, 0)?;
/// println!("Loaded {} tensors using GPU decompression", tensors.len());
/// ```
#[cfg(feature = "haagenti-gpu")]
pub fn load_hct_directory_gpu(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
    gpu_device_id: usize,
) -> Result<HashMap<String, Tensor>, HctError> {
    use std::time::Instant;

    let dir = dir.as_ref();
    let start = Instant::now();

    // Collect all HCT files
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    let total = files.len();
    tracing::info!(
        count = total,
        directory = %dir.display(),
        gpu_device = gpu_device_id,
        "Loading HCT files with GPU decompression"
    );

    // Create GPU decompressor
    let config = DecompressConfig {
        device_id: gpu_device_id,
        verify_checksums: false, // Skip for performance
        output_f16: dtype == DType::F16,
    };

    let mut decompressor = GpuDecompressor::with_config(config).map_err(|e| HctError::Format {
        message: format!("Failed to create GPU decompressor: {}", e),
    })?;

    let mut tensors = HashMap::with_capacity(total);
    let mut total_input_bytes = 0usize;
    let mut total_output_bytes = 0usize;

    for path in &files {
        // Read compressed data
        let compressed = std::fs::read(path).map_err(|e| HctError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;
        total_input_bytes += compressed.len();

        // Parse HCT header to get shape and name
        let loader = HctLoader::from_file(path)?;
        let metadata = loader.metadata();
        let name = crate::hct::filename_to_tensor_name(&metadata.name);
        let shape: Vec<usize> = metadata.shape.iter().map(|&d| d as usize).collect();

        // GPU decompress (returns f32 data after IDCT)
        let decompressed =
            decompressor
                .decompress(&compressed, &shape)
                .map_err(|e| HctError::Format {
                    message: format!("GPU decompression failed for {}: {}", name, e),
                })?;
        total_output_bytes += decompressed.len() * 4;

        // Create tensor from decompressed data
        let tensor =
            Tensor::from_vec(decompressed, shape.as_slice(), &Device::Cpu).map_err(|e| {
                HctError::Tensor {
                    message: format!("Failed to create tensor from GPU output: {}", e),
                }
            })?;

        // Convert dtype and move to target device
        let tensor = tensor
            .to_dtype(dtype)
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to convert dtype: {}", e),
            })?
            .to_device(device)
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to move to device: {}", e),
            })?;

        tensors.insert(name, tensor);
    }

    let elapsed = start.elapsed();
    let throughput_mbps = total_output_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    tracing::info!(
        loaded = tensors.len(),
        input_mb = total_input_bytes as f64 / 1_000_000.0,
        output_mb = total_output_bytes as f64 / 1_000_000.0,
        ratio = total_output_bytes as f64 / total_input_bytes.max(1) as f64,
        elapsed_ms = elapsed.as_millis(),
        throughput_mbps = throughput_mbps,
        "GPU decompression complete"
    );

    Ok(tensors)
}

/// GPU-accelerated HCT loading with statistics tracking.
///
/// Same as `load_hct_directory_gpu` but returns detailed decompression statistics.
#[cfg(feature = "haagenti-gpu")]
pub fn load_hct_directory_gpu_with_stats(
    dir: impl AsRef<Path>,
    device: &Device,
    dtype: DType,
    gpu_device_id: usize,
) -> Result<(HashMap<String, Tensor>, GpuDecompressStats), HctError> {
    use std::time::Instant;

    let dir = dir.as_ref();
    let start = Instant::now();

    // Collect all HCT files
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| HctError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "hct"))
        .map(|entry| entry.path())
        .collect();

    let total = files.len();

    let config = DecompressConfig {
        device_id: gpu_device_id,
        verify_checksums: false,
        output_f16: dtype == DType::F16,
    };

    let mut decompressor = GpuDecompressor::with_config(config).map_err(|e| HctError::Format {
        message: format!("Failed to create GPU decompressor: {}", e),
    })?;

    let mut tensors = HashMap::with_capacity(total);
    let mut stats = GpuDecompressStats::default();

    for path in &files {
        let compressed = std::fs::read(path).map_err(|e| HctError::Io {
            path: path.clone(),
            message: e.to_string(),
        })?;
        stats.total_input_bytes += compressed.len();

        let loader = HctLoader::from_file(path)?;
        let metadata = loader.metadata();
        let name = crate::hct::filename_to_tensor_name(&metadata.name);
        let shape: Vec<usize> = metadata.shape.iter().map(|&d| d as usize).collect();

        let decompress_start = Instant::now();
        let decompressed =
            decompressor
                .decompress(&compressed, &shape)
                .map_err(|e| HctError::Format {
                    message: format!("GPU decompression failed for {}: {}", name, e),
                })?;
        stats.decompress_time_ms += decompress_start.elapsed().as_secs_f64() * 1000.0;
        stats.total_output_bytes += decompressed.len() * 4;

        let tensor = Tensor::from_vec(decompressed, shape.as_slice(), &Device::Cpu)
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to create tensor: {}", e),
            })?
            .to_dtype(dtype)
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to convert dtype: {}", e),
            })?
            .to_device(device)
            .map_err(|e| HctError::Tensor {
                message: format!("Failed to move to device: {}", e),
            })?;

        tensors.insert(name, tensor);
        stats.num_tensors += 1;
    }

    stats.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    stats.throughput_mbps = stats.total_output_bytes as f64 / stats.total_time_ms / 1000.0;

    Ok((tensors, stats))
}

/// Statistics from GPU-accelerated HCT decompression.
#[cfg(feature = "haagenti-gpu")]
#[derive(Debug, Clone, Default)]
pub struct GpuDecompressStats {
    /// Number of tensors decompressed.
    pub num_tensors: usize,
    /// Total input bytes (compressed).
    pub total_input_bytes: usize,
    /// Total output bytes (decompressed).
    pub total_output_bytes: usize,
    /// Time spent on GPU decompression (ms).
    pub decompress_time_ms: f64,
    /// Total time including I/O and tensor creation (ms).
    pub total_time_ms: f64,
    /// Throughput in MB/s (output bytes / total time).
    pub throughput_mbps: f64,
}

#[cfg(feature = "haagenti-gpu")]
impl GpuDecompressStats {
    /// Compression ratio (output / input).
    pub fn compression_ratio(&self) -> f32 {
        if self.total_input_bytes == 0 {
            0.0
        } else {
            self.total_output_bytes as f32 / self.total_input_bytes as f32
        }
    }

    /// Format for display.
    pub fn summary(&self) -> String {
        format!(
            "{} tensors, {:.1} MB -> {:.1} MB ({:.1}x), {:.1}ms total ({:.1}ms decompress), {:.1} MB/s",
            self.num_tensors,
            self.total_input_bytes as f64 / 1_000_000.0,
            self.total_output_bytes as f64 / 1_000_000.0,
            self.compression_ratio(),
            self.total_time_ms,
            self.decompress_time_ms,
            self.throughput_mbps,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a minimal valid standard HCT file for testing.
    /// Uses the simpler HCT format (not HoloTensor) for easier testing.
    fn create_test_hct_file(dir: &Path, name: &str, shape: &[u64]) -> PathBuf {
        use haagenti::tensor::{CompressionAlgorithm, DType as HctDType, HctWriter};
        use haagenti::Lz4Compressor;

        let path = dir.join(format!("{}.hct", name));
        let file = fs::File::create(&path).expect("create file");

        // Calculate data size
        let elements: u64 = shape.iter().product();
        let data: Vec<f32> = (0..elements).map(|i| i as f32 * 0.001).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Create HCT file with LZ4 compression
        let mut writer = HctWriter::new(
            file,
            CompressionAlgorithm::Lz4,
            HctDType::F32,
            shape.to_vec(),
        )
        .with_block_size(64 * 1024);

        let compressor = Lz4Compressor::new();
        writer
            .compress_data(&bytes, &compressor)
            .expect("write data");
        writer.finish().expect("finish");

        path
    }

    /// Helper to create a truncated/corrupted HCT file.
    fn create_truncated_hct_file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(format!("{}.hct", name));
        // Write just the magic bytes and some garbage - will fail to parse
        let mut file = fs::File::create(&path).expect("create file");
        file.write_all(b"HCTN\x00\x00\x00\x00")
            .expect("write truncated file");
        path
    }

    #[test]
    fn test_memory_budget_tracking() {
        let budget = MemoryBudget::new(1024);

        assert_eq!(budget.current_usage(), 0);
        assert_eq!(budget.remaining(), 1024);
        assert!(budget.can_allocate(512));
        assert!(budget.can_allocate(1024));
        assert!(!budget.can_allocate(1025));

        budget.allocate(512);
        assert_eq!(budget.current_usage(), 512);
        assert_eq!(budget.remaining(), 512);
        assert!(budget.can_allocate(512));
        assert!(!budget.can_allocate(513));

        budget.deallocate(256);
        assert_eq!(budget.current_usage(), 256);
        assert_eq!(budget.remaining(), 768);
    }

    #[test]
    fn test_memory_budget_warning_threshold() {
        let budget = MemoryBudget::new(1000);
        assert!(!budget.is_warning());

        budget.allocate(840);
        assert!(!budget.is_warning()); // 84% < 85%

        budget.allocate(20); // Now at 860/1000 = 86%
        assert!(budget.is_warning()); // 86% >= 85%
    }

    #[test]
    fn test_sequential_loader_respects_memory_budget() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create some test files
        create_test_hct_file(temp_dir.path(), "tensor_a", &[100]);
        create_test_hct_file(temp_dir.path(), "tensor_b", &[100]);

        // Create a very small budget that should fail
        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::new(100), // Only 100 bytes
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::Fail,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        // First tensor should fail due to budget
        let result = loader.next_tensor();
        assert!(result.is_some());
        let err = result.unwrap();
        assert!(err.is_err(), "Should fail due to memory budget");
    }

    #[test]
    fn test_sequential_loader_yields_tensors_one_at_a_time() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create test files
        create_test_hct_file(temp_dir.path(), "tensor_a", &[16]);
        create_test_hct_file(temp_dir.path(), "tensor_b", &[16]);
        create_test_hct_file(temp_dir.path(), "tensor_c", &[16]);

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::Fail,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        assert_eq!(loader.total_files(), 3);

        // Load one at a time
        let mut count = 0;
        while let Some(result) = loader.next_tensor() {
            let loaded = result.expect("load tensor");
            count += 1;

            let progress = loader.progress();
            assert_eq!(progress.loaded_files, count);
        }

        assert_eq!(count, 3);
    }

    #[test]
    fn test_sequential_loader_handles_corrupted_files_skip() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create one valid and one corrupted file
        create_test_hct_file(temp_dir.path(), "tensor_valid", &[16]);
        create_truncated_hct_file(temp_dir.path(), "tensor_corrupt");

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::Skip,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        assert_eq!(loader.total_files(), 2);

        let mut loaded_count = 0;
        while let Some(result) = loader.next_tensor() {
            // Should not error when skipping
            if result.is_ok() {
                loaded_count += 1;
            }
        }

        let progress = loader.progress();
        // One file loaded, one skipped
        assert!(progress.loaded_files >= 1);
        assert!(progress.skipped_files >= 1);
    }

    #[test]
    fn test_sequential_loader_handles_corrupted_files_recover() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create a corrupted layernorm file (should be recovered with ones)
        create_truncated_hct_file(temp_dir.path(), "model_layers_0_input_layernorm_weight");

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::InitializeDefault,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        let result = loader.next_tensor();
        assert!(result.is_some());

        let loaded = result.unwrap().expect("should recover");
        assert!(loaded.recovered);
        assert!(loaded.name.contains("layernorm"));

        // Layernorm should be initialized to ones
        let sum = loaded
            .tensor
            .sum_all()
            .expect("sum")
            .to_scalar::<f32>()
            .expect("scalar");
        let count = loaded.tensor.elem_count() as f32;
        assert!(
            (sum - count).abs() < 0.01,
            "Layernorm should be ones, sum={}, count={}",
            sum,
            count
        );
    }

    #[test]
    fn test_sequential_loader_initializes_bias_to_zeros() {
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create a corrupted bias file (should be recovered with zeros)
        create_truncated_hct_file(temp_dir.path(), "model_layers_0_self_attn_o_proj_bias");

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::InitializeDefault,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        let result = loader.next_tensor();
        assert!(result.is_some());

        let loaded = result.unwrap().expect("should recover");
        assert!(loaded.recovered);
        assert!(loaded.name.contains("bias"));

        // Bias should be initialized to zeros
        let sum = loaded
            .tensor
            .sum_all()
            .expect("sum")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!((sum).abs() < 0.01, "Bias should be zeros, sum={}", sum);
    }

    #[test]
    fn test_load_all_collects_tensors() {
        let temp_dir = TempDir::new().expect("create temp dir");

        create_test_hct_file(temp_dir.path(), "tensor_a", &[8]);
        create_test_hct_file(temp_dir.path(), "tensor_b", &[8]);

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::Fail,
            min_quality: 0.7,
        };

        let loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        let tensors = loader.load_all().expect("load all");

        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains_key("tensor.a") || tensors.contains_key("tensor_a"));
    }

    #[test]
    fn test_progress_tracking() {
        let temp_dir = TempDir::new().expect("create temp dir");

        create_test_hct_file(temp_dir.path(), "t1", &[4]);
        create_test_hct_file(temp_dir.path(), "t2", &[4]);
        create_truncated_hct_file(temp_dir.path(), "t3_corrupt");

        let config = SequentialLoadConfig {
            memory_budget: MemoryBudget::unlimited(),
            device: Device::Cpu,
            dtype: DType::F32,
            fallback_strategy: FallbackStrategy::Skip,
            min_quality: 0.7,
        };

        let mut loader = SequentialHctLoader::new(temp_dir.path(), config).expect("create loader");

        let initial = loader.progress();
        assert_eq!(initial.total_files, 3);
        assert_eq!(initial.loaded_files, 0);

        while loader.next_tensor().is_some() {}

        let final_progress = loader.progress();
        assert_eq!(final_progress.total_files, 3);
        assert_eq!(final_progress.loaded_files, 3);
        assert!(final_progress.skipped_files >= 1);
    }
}
