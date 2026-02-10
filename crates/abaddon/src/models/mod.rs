//! Model architecture implementations.
//!
//! Supports multiple LLM architectures:
//! - Llama (Llama 2, Llama 3, Llama 3.1, Llama 3.2, CodeLlama)
//! - LazyLlama (layer-lazy variant for 405B+ models)
//! - Qwen2 (Qwen2, Qwen2.5, Qwen2.5-Coder)

pub mod lazy_llama;
pub mod lazy_qwen2;
pub mod llama;
#[allow(dead_code)]
mod quantized_llama;
pub mod qwen2;

pub use lazy_llama::{LazyLlama, LazyLoadError, LazyStats};
pub use lazy_qwen2::LazyQwen2;
pub use llama::{Llama, LlamaConfig};
pub use qwen2::{CacheType, Qwen2, Qwen2Config};

use candle_core::{Result as CandleResult, Tensor};

/// Loaded model variant - wraps different model implementations.
///
/// This allows the engine to work with multiple model types through a unified interface.
/// Named `ModelKind` to avoid conflict with `infernum_core::ModelArchitecture`.
pub enum ModelKind {
    /// Llama-family models (Llama, CodeLlama, Mistral, etc.)
    Llama(Llama),
    /// Lazy Llama for 405B+ models (layer-by-layer loading)
    LazyLlama(LazyLlama),
    /// Qwen2-family models (Qwen2, Qwen2.5, Qwen2.5-Coder)
    Qwen2(Qwen2),
    /// Lazy Qwen2 for 14B+ models (layer-by-layer loading)
    LazyQwen2(LazyQwen2),
}

impl ModelKind {
    /// Forward pass for the model.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.forward(input_ids, start_pos),
            Self::LazyLlama(model) => model
                .forward(input_ids, start_pos)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
            Self::Qwen2(model) => model.forward(input_ids, start_pos),
            Self::LazyQwen2(model) => model
                .forward(input_ids, start_pos)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
        }
    }

    /// Clears the KV cache.
    pub fn clear_cache(&mut self) {
        match self {
            Self::Llama(model) => model.clear_cache(),
            Self::LazyLlama(model) => model.clear_cache(),
            Self::Qwen2(model) => model.clear_cache(),
            Self::LazyQwen2(model) => model.clear_cache(),
        }
    }

    /// Forward pass for embedding extraction.
    pub fn forward_embedding(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.forward_embedding(input_ids),
            Self::LazyLlama(_model) => {
                // LazyLlama doesn't support embedding extraction yet
                Err(candle_core::Error::Msg(
                    "Embedding extraction not supported for LazyLlama".to_string(),
                ))
            },
            Self::Qwen2(model) => model.forward_embedding(input_ids),
            Self::LazyQwen2(_model) => {
                // LazyQwen2 doesn't support embedding extraction yet
                Err(candle_core::Error::Msg(
                    "Embedding extraction not supported for LazyQwen2".to_string(),
                ))
            },
        }
    }

    /// Extract embeddings by mean pooling.
    pub fn extract_embeddings(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.extract_embeddings(input_ids),
            Self::LazyLlama(_model) => {
                // LazyLlama doesn't support embedding extraction yet
                Err(candle_core::Error::Msg(
                    "Embedding extraction not supported for LazyLlama".to_string(),
                ))
            },
            Self::Qwen2(model) => model.extract_embeddings(input_ids),
            Self::LazyQwen2(_model) => {
                // LazyQwen2 doesn't support embedding extraction yet
                Err(candle_core::Error::Msg(
                    "Embedding extraction not supported for LazyQwen2".to_string(),
                ))
            },
        }
    }
}

/// Supported model architectures for detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureType {
    /// Llama architecture family
    Llama,
    /// Qwen2 architecture family
    Qwen2,
    /// Unknown/unsupported architecture
    Unknown,
}

impl ArchitectureType {
    /// Detects the architecture type from a model type string or architecture name.
    ///
    /// # Arguments
    /// * `model_type` - The model_type from config.json (e.g., "llama", "qwen2")
    /// * `architectures` - Optional list of architecture names (e.g., ["LlamaForCausalLM"])
    pub fn detect(model_type: Option<&str>, architectures: Option<&[String]>) -> Self {
        // First check model_type
        if let Some(mt) = model_type {
            let mt_lower = mt.to_lowercase();
            if mt_lower.contains("llama") || mt_lower.contains("mistral") {
                return Self::Llama;
            }
            if mt_lower.contains("qwen2") || mt_lower == "qwen2" {
                return Self::Qwen2;
            }
        }

        // Then check architectures list
        if let Some(archs) = architectures {
            for arch in archs {
                let arch_lower = arch.to_lowercase();
                if arch_lower.contains("llama") || arch_lower.contains("mistral") {
                    return Self::Llama;
                }
                if arch_lower.contains("qwen2") {
                    return Self::Qwen2;
                }
            }
        }

        Self::Unknown
    }

    /// Returns a human-readable name for this architecture.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama",
            Self::Qwen2 => "Qwen2",
            Self::Unknown => "Unknown",
        }
    }
}
