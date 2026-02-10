//! Error types for tiered memory management.

use std::path::PathBuf;

/// Errors from tiered weight loading.
#[derive(Debug, thiserror::Error)]
pub enum TieredError {
    /// VRAM allocation failed.
    #[error("VRAM allocation failed: {message}")]
    VramAllocation {
        /// Error message.
        message: String,
        /// Requested size in bytes.
        requested: u64,
        /// Available size in bytes.
        available: u64,
    },

    /// RAM allocation failed.
    #[error("RAM allocation failed: {message}")]
    RamAllocation {
        /// Error message.
        message: String,
        /// Requested size in bytes.
        requested: u64,
        /// Available size in bytes.
        available: u64,
    },

    /// NVMe cache error.
    #[error("NVMe cache error: {message}")]
    NvmeCache {
        /// Error message.
        message: String,
        /// Path involved (if any).
        path: Option<PathBuf>,
    },

    /// HCT decompression failed.
    #[error("HCT decompression failed for {tensor}: {message}")]
    Decompression {
        /// Tensor name.
        tensor: String,
        /// Error message.
        message: String,
    },

    /// Layer not found.
    #[error("layer {0} not found (model has {1} layers)")]
    LayerNotFound(usize, usize),

    /// Tensor not found.
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),

    /// Shape mismatch.
    #[error("shape mismatch: expected {expected}, got {got}")]
    Shape {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },

    /// CUDA error.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Prefetch thread panicked.
    #[error("prefetch thread panicked: {0}")]
    PrefetchPanic(String),

    /// Channel send error.
    #[error("channel send failed: {0}")]
    ChannelSend(String),

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Model load error.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Weight store error (delegation from existing error type).
    #[error("weight store error: {0}")]
    WeightStore(String),
}

impl TieredError {
    /// Create a VRAM allocation error.
    pub fn vram_alloc(message: impl Into<String>, requested: u64, available: u64) -> Self {
        Self::VramAllocation {
            message: message.into(),
            requested,
            available,
        }
    }

    /// Create a RAM allocation error.
    pub fn ram_alloc(message: impl Into<String>, requested: u64, available: u64) -> Self {
        Self::RamAllocation {
            message: message.into(),
            requested,
            available,
        }
    }

    /// Create an NVMe cache error.
    pub fn nvme(message: impl Into<String>) -> Self {
        Self::NvmeCache {
            message: message.into(),
            path: None,
        }
    }

    /// Create an NVMe cache error with path.
    pub fn nvme_path(message: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::NvmeCache {
            message: message.into(),
            path: Some(path.into()),
        }
    }

    /// Create a decompression error.
    pub fn decompress(tensor: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Decompression {
            tensor: tensor.into(),
            message: message.into(),
        }
    }
}

impl From<cudarc::driver::DriverError> for TieredError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e.to_string())
    }
}

impl From<super::super::InferenceError> for TieredError {
    fn from(e: super::super::InferenceError) -> Self {
        Self::WeightStore(e.to_string())
    }
}
