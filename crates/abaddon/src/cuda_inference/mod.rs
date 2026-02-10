//! Custom CUDA inference engine for GPU-resident model execution.
//!
//! This module provides a high-performance inference engine that keeps all data
//! GPU-resident from loading through generation, bypassing Candle's CPU tensor
//! requirement.
//!
//! ## Architecture
//!
//! ```text
//! HCT Files → WeightStore → ComputeEngine → Generator
//!                ↓              ↓
//!           GpuTensor      KvCache
//! ```
//!
//! ## Supported Model Architectures
//!
//! - Llama / Llama 2 / Llama 3
//! - CodeLlama
//! - Mistral / Mixtral
//! - Qwen / Qwen2
//! - Phi-2 / Phi-3
//!
//! ## Example
//!
//! ```rust,ignore
//! use abaddon::cuda_inference::{WeightStore, Generator, ModelArch};
//!
//! // Load quantized model
//! let weights = WeightStore::load_hct("/path/to/model", ModelArch::Llama)?;
//!
//! // Create generator
//! let mut generator = Generator::new(weights, tokenizer)?;
//!
//! // Generate
//! for token in generator.generate("Hello, ", &params)? {
//!     print!("{}", token);
//! }
//! ```

#[cfg(feature = "cuda")]
pub mod tensor;

#[cfg(feature = "cuda")]
pub mod arch;

#[cfg(feature = "cuda")]
pub mod weight_store;

#[cfg(feature = "cuda")]
pub mod cublas;

#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub mod kv_cache;

#[cfg(feature = "cuda")]
pub mod compute;

#[cfg(feature = "cuda")]
pub mod generate;

#[cfg(feature = "cuda")]
pub mod streams;

#[cfg(feature = "cuda")]
pub mod batch;

#[cfg(feature = "cuda")]
pub mod speculative;

#[cfg(feature = "cuda")]
pub mod lazy_layers;

#[cfg(feature = "cuda")]
pub mod lazy_weight_store;

#[cfg(feature = "cuda")]
pub mod lazy_generate;

#[cfg(feature = "cuda")]
pub mod tiered;

#[cfg(feature = "cuda")]
pub mod tiered_generate;

// Re-exports
#[cfg(feature = "cuda")]
pub use tensor::GpuTensor;

#[cfg(feature = "cuda")]
pub use arch::{ModelArch, ModelConfig};

#[cfg(feature = "cuda")]
pub use weight_store::WeightStore;

#[cfg(feature = "cuda")]
pub use compute::ComputeEngine;

#[cfg(feature = "cuda")]
pub use generate::{Generator, SamplingParams};

#[cfg(feature = "cuda")]
pub use streams::StreamManager;

#[cfg(feature = "cuda")]
pub use batch::{BatchScheduler, BatchStats, Request, RequestState};

#[cfg(feature = "cuda")]
pub use speculative::{SpeculativeConfig, SpeculativeDecoder, VerificationResult};

#[cfg(feature = "cuda")]
pub use lazy_layers::{HoloLayerLoader, LayerLoader, LazyLayerStats, LazyLayerStore};

#[cfg(feature = "cuda")]
pub use lazy_weight_store::{LazyWeightConfig, LazyWeightStore};

#[cfg(feature = "cuda")]
pub use lazy_generate::LazyGenerator;

#[cfg(feature = "cuda")]
pub use tiered::{
    create_loader, EagerLoader, LoadingStrategy, ProgressiveLoader, TieredConfig, TieredError,
    TieredStats, TieredWeightStore, WeightLoader,
};

#[cfg(feature = "cuda")]
pub use tiered_generate::TieredGenerator;

/// Errors from CUDA inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// CUDA device initialization failed.
    #[error("CUDA device error: {0}")]
    Device(String),

    /// Kernel execution failed.
    #[error("Kernel execution error: {0}")]
    Kernel(String),

    /// Memory allocation failed.
    #[error("Memory allocation error: {0}")]
    Memory(String),

    /// cuBLAS operation failed.
    #[error("cuBLAS error: {0}")]
    CuBlas(String),

    /// Invalid tensor shape.
    #[error("Shape mismatch: expected {expected}, got {got}")]
    Shape {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        got: String,
    },

    /// Model loading error.
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Unsupported model architecture.
    #[error("Unsupported architecture: {0}")]
    UnsupportedArch(String),

    /// Generation error.
    #[error("Generation error: {0}")]
    Generation(String),

    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

#[cfg(all(test, feature = "cuda"))]
mod tests;
