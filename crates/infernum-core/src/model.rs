//! Model metadata and architecture types.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::types::{ModelId, QuantizationType};

/// Source location for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    /// HuggingFace Hub model.
    HuggingFace {
        /// Repository ID (e.g., "meta-llama/Llama-3.2-3B-Instruct").
        repo_id: String,
        /// Optional revision (branch, tag, or commit).
        revision: Option<String>,
    },
    /// Local filesystem path.
    LocalPath {
        /// Path to the model directory or file.
        path: PathBuf,
    },
    /// S3 bucket.
    S3 {
        /// Bucket name.
        bucket: String,
        /// Object key.
        key: String,
        /// Optional region.
        region: Option<String>,
    },
    /// GGUF file format.
    Gguf {
        /// Path to the GGUF file.
        path: PathBuf,
    },
}

impl ModelSource {
    /// Creates a HuggingFace source.
    #[must_use]
    pub fn huggingface(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            revision: None,
        }
    }

    /// Creates a HuggingFace source with a specific revision.
    #[must_use]
    pub fn huggingface_rev(repo_id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo_id: repo_id.into(),
            revision: Some(revision.into()),
        }
    }

    /// Creates a local path source.
    #[must_use]
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::LocalPath { path: path.into() }
    }

    /// Creates a GGUF source.
    #[must_use]
    pub fn gguf(path: impl Into<PathBuf>) -> Self {
        Self::Gguf { path: path.into() }
    }
}

/// Llama model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlamaVersion {
    /// Llama 2.
    V2,
    /// Llama 3.
    V3,
    /// Llama 3.1.
    V3_1,
    /// Llama 3.2.
    V3_2,
}

/// Mistral model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MistralVariant {
    /// Mistral 7B.
    Mistral7B,
    /// Mistral Nemo.
    Nemo,
    /// Mistral Large.
    Large,
}

/// Qwen model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QwenVersion {
    /// Qwen 2.
    V2,
    /// Qwen 2.5.
    V2_5,
}

/// Phi model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiVersion {
    /// Phi 3.
    V3,
    /// Phi 3.5.
    V3_5,
    /// Phi 4.
    V4,
}

/// Gemma model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GemmaVersion {
    /// Gemma 1.
    V1,
    /// Gemma 2.
    V2,
}

/// Supported model architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelArchitecture {
    // === Decoder-only (Causal LM) ===
    /// Llama family models.
    Llama {
        /// Model version.
        version: LlamaVersion,
    },
    /// Mistral family models.
    Mistral {
        /// Model variant.
        variant: MistralVariant,
    },
    /// Mixtral MoE models.
    Mixtral {
        /// Number of experts.
        num_experts: u8,
    },
    /// Qwen family models.
    Qwen {
        /// Model version.
        version: QwenVersion,
    },
    /// Phi family models.
    Phi {
        /// Model version.
        version: PhiVersion,
    },
    /// Gemma family models.
    Gemma {
        /// Model version.
        version: GemmaVersion,
    },
    /// DeepSeek models.
    DeepSeek {
        /// Model version.
        version: u8,
    },

    // === Encoder-only (Embeddings) ===
    /// BERT-based models.
    Bert,
    /// Nomic Embed models.
    NomicEmbed,
    /// Jina Embed models.
    JinaEmbed,

    // === Vision-Language ===
    /// LLaVA-Next models.
    LlavaNext,
    /// Qwen2-VL models.
    Qwen2VL,
    /// Pixtral models.
    Pixtral,

    // === Code-specialized ===
    /// CodeLlama models.
    CodeLlama,
    /// StarCoder 2 models.
    StarCoder2,
    /// DeepSeek Coder models.
    DeepSeekCoder {
        /// Model version.
        version: u8,
    },
}

impl ModelArchitecture {
    /// Returns `true` if this architecture supports vision input.
    #[must_use]
    pub fn supports_vision(&self) -> bool {
        matches!(
            self,
            Self::LlavaNext | Self::Qwen2VL | Self::Pixtral
        )
    }

    /// Returns `true` if this is an embedding model.
    #[must_use]
    pub fn is_embedding_model(&self) -> bool {
        matches!(self, Self::Bert | Self::NomicEmbed | Self::JinaEmbed)
    }

    /// Returns `true` if this is specialized for code.
    #[must_use]
    pub fn is_code_specialized(&self) -> bool {
        matches!(
            self,
            Self::CodeLlama | Self::StarCoder2 | Self::DeepSeekCoder { .. }
        )
    }
}

/// Model metadata and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier.
    pub id: ModelId,
    /// Model architecture.
    pub architecture: ModelArchitecture,
    /// Model source location.
    pub source: ModelSource,
    /// Maximum context length in tokens.
    pub context_length: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Number of layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: Option<u32>,
    /// Quantization applied to the model.
    pub quantization: Option<QuantizationType>,
    /// Model size in bytes.
    pub size_bytes: Option<u64>,
    /// Human-readable description.
    pub description: Option<String>,
}

impl ModelMetadata {
    /// Creates a new `ModelMetadata` builder.
    #[must_use]
    pub fn builder(id: impl Into<ModelId>, architecture: ModelArchitecture) -> ModelMetadataBuilder {
        ModelMetadataBuilder::new(id, architecture)
    }
}

/// Builder for `ModelMetadata`.
#[derive(Debug)]
pub struct ModelMetadataBuilder {
    id: ModelId,
    architecture: ModelArchitecture,
    source: Option<ModelSource>,
    context_length: u32,
    vocab_size: u32,
    hidden_size: u32,
    num_layers: u32,
    num_attention_heads: u32,
    num_kv_heads: Option<u32>,
    quantization: Option<QuantizationType>,
    size_bytes: Option<u64>,
    description: Option<String>,
}

impl ModelMetadataBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(id: impl Into<ModelId>, architecture: ModelArchitecture) -> Self {
        Self {
            id: id.into(),
            architecture,
            source: None,
            context_length: 4096,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: None,
            quantization: None,
            size_bytes: None,
            description: None,
        }
    }

    /// Sets the model source.
    #[must_use]
    pub fn source(mut self, source: ModelSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Sets the context length.
    #[must_use]
    pub fn context_length(mut self, length: u32) -> Self {
        self.context_length = length;
        self
    }

    /// Sets the vocabulary size.
    #[must_use]
    pub fn vocab_size(mut self, size: u32) -> Self {
        self.vocab_size = size;
        self
    }

    /// Sets the hidden size.
    #[must_use]
    pub fn hidden_size(mut self, size: u32) -> Self {
        self.hidden_size = size;
        self
    }

    /// Sets the number of layers.
    #[must_use]
    pub fn num_layers(mut self, layers: u32) -> Self {
        self.num_layers = layers;
        self
    }

    /// Sets the number of attention heads.
    #[must_use]
    pub fn num_attention_heads(mut self, heads: u32) -> Self {
        self.num_attention_heads = heads;
        self
    }

    /// Sets the number of KV heads.
    #[must_use]
    pub fn num_kv_heads(mut self, heads: u32) -> Self {
        self.num_kv_heads = Some(heads);
        self
    }

    /// Sets the quantization type.
    #[must_use]
    pub fn quantization(mut self, quant: QuantizationType) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Sets the model size in bytes.
    #[must_use]
    pub fn size_bytes(mut self, size: u64) -> Self {
        self.size_bytes = Some(size);
        self
    }

    /// Sets the description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builds the `ModelMetadata`.
    ///
    /// # Panics
    ///
    /// Panics if source is not set.
    #[must_use]
    pub fn build(self) -> ModelMetadata {
        ModelMetadata {
            id: self.id,
            architecture: self.architecture,
            source: self.source.expect("source must be set"),
            context_length: self.context_length,
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            quantization: self.quantization,
            size_bytes: self.size_bytes,
            description: self.description,
        }
    }
}
