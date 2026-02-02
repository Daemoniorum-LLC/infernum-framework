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
    /// HoloTensor compressed model (HCT format).
    ///
    /// HoloTensor models use holographic compression to enable 70B+ models
    /// on 24GB VRAM via progressive quality reconstruction.
    HoloTensor {
        /// Path to the HCT model directory.
        path: PathBuf,
        /// Minimum quality threshold (0.0-1.0). Start generating at this quality.
        min_quality: f32,
        /// Target quality threshold (0.0-1.0). Improve to this during generation.
        target_quality: f32,
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

    /// Creates a HoloTensor source with default quality settings.
    ///
    /// Default: min_quality=0.7, target_quality=0.95
    #[must_use]
    pub fn holotensor(path: impl Into<PathBuf>) -> Self {
        Self::HoloTensor {
            path: path.into(),
            min_quality: 0.7,
            target_quality: 0.95,
        }
    }

    /// Creates a HoloTensor source with custom quality settings.
    #[must_use]
    pub fn holotensor_with_quality(
        path: impl Into<PathBuf>,
        min_quality: f32,
        target_quality: f32,
    ) -> Self {
        Self::HoloTensor {
            path: path.into(),
            min_quality: min_quality.clamp(0.5, 1.0),
            target_quality: target_quality.clamp(min_quality, 1.0),
        }
    }

    /// Returns `true` if this is a HoloTensor source.
    #[must_use]
    pub fn is_holotensor(&self) -> bool {
        matches!(self, Self::HoloTensor { .. })
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
        matches!(self, Self::LlavaNext | Self::Qwen2VL | Self::Pixtral)
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
    pub fn builder(
        id: impl Into<ModelId>,
        architecture: ModelArchitecture,
    ) -> ModelMetadataBuilder {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // ModelSource tests
    // ==========================================================================

    #[test]
    fn test_model_source_huggingface() {
        let source = ModelSource::huggingface("meta-llama/Llama-3.2-3B");

        match source {
            ModelSource::HuggingFace { repo_id, revision } => {
                assert_eq!(repo_id, "meta-llama/Llama-3.2-3B");
                assert!(revision.is_none());
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn test_model_source_huggingface_rev() {
        let source = ModelSource::huggingface_rev("microsoft/phi-3", "v1.0");

        match source {
            ModelSource::HuggingFace { repo_id, revision } => {
                assert_eq!(repo_id, "microsoft/phi-3");
                assert_eq!(revision, Some("v1.0".to_string()));
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn test_model_source_local() {
        let source = ModelSource::local("/models/llama");

        match source {
            ModelSource::LocalPath { path } => {
                assert_eq!(path, PathBuf::from("/models/llama"));
            }
            _ => panic!("Expected LocalPath source"),
        }
    }

    #[test]
    fn test_model_source_gguf() {
        let source = ModelSource::gguf("/models/model.gguf");

        match source {
            ModelSource::Gguf { path } => {
                assert_eq!(path, PathBuf::from("/models/model.gguf"));
            }
            _ => panic!("Expected Gguf source"),
        }
    }

    #[test]
    fn test_model_source_s3() {
        let source = ModelSource::S3 {
            bucket: "my-bucket".to_string(),
            key: "models/llama.bin".to_string(),
            region: Some("us-west-2".to_string()),
        };

        match source {
            ModelSource::S3 { bucket, key, region } => {
                assert_eq!(bucket, "my-bucket");
                assert_eq!(key, "models/llama.bin");
                assert_eq!(region, Some("us-west-2".to_string()));
            }
            _ => panic!("Expected S3 source"),
        }
    }

    #[test]
    fn test_model_source_serialization() {
        let source = ModelSource::huggingface("test/model");
        let json = serde_json::to_string(&source).expect("serialize");
        assert!(json.contains("hugging_face"));
        assert!(json.contains("test/model"));

        let parsed: ModelSource = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            ModelSource::HuggingFace { repo_id, .. } => {
                assert_eq!(repo_id, "test/model");
            }
            _ => panic!("Wrong variant"),
        }
    }

    // ==========================================================================
    // ModelArchitecture tests
    // ==========================================================================

    #[test]
    fn test_architecture_supports_vision() {
        assert!(ModelArchitecture::LlavaNext.supports_vision());
        assert!(ModelArchitecture::Qwen2VL.supports_vision());
        assert!(ModelArchitecture::Pixtral.supports_vision());

        assert!(!ModelArchitecture::Llama {
            version: LlamaVersion::V3_2
        }
        .supports_vision());
        assert!(!ModelArchitecture::Bert.supports_vision());
        assert!(!ModelArchitecture::CodeLlama.supports_vision());
    }

    #[test]
    fn test_architecture_is_embedding_model() {
        assert!(ModelArchitecture::Bert.is_embedding_model());
        assert!(ModelArchitecture::NomicEmbed.is_embedding_model());
        assert!(ModelArchitecture::JinaEmbed.is_embedding_model());

        assert!(!ModelArchitecture::Llama {
            version: LlamaVersion::V3
        }
        .is_embedding_model());
        assert!(!ModelArchitecture::LlavaNext.is_embedding_model());
    }

    #[test]
    fn test_architecture_is_code_specialized() {
        assert!(ModelArchitecture::CodeLlama.is_code_specialized());
        assert!(ModelArchitecture::StarCoder2.is_code_specialized());
        assert!(ModelArchitecture::DeepSeekCoder { version: 2 }.is_code_specialized());

        assert!(!ModelArchitecture::Llama {
            version: LlamaVersion::V2
        }
        .is_code_specialized());
        assert!(!ModelArchitecture::Bert.is_code_specialized());
    }

    #[test]
    fn test_llama_versions() {
        let v2 = LlamaVersion::V2;
        let v3 = LlamaVersion::V3;
        let v3_1 = LlamaVersion::V3_1;
        let v3_2 = LlamaVersion::V3_2;

        assert_ne!(v2, v3);
        assert_eq!(v3_1.clone(), v3_1);
        assert_eq!(v3_2.clone(), v3_2);
    }

    #[test]
    fn test_mistral_variants() {
        assert_eq!(MistralVariant::Mistral7B, MistralVariant::Mistral7B);
        assert_ne!(MistralVariant::Mistral7B, MistralVariant::Nemo);
        assert_ne!(MistralVariant::Nemo, MistralVariant::Large);
    }

    #[test]
    fn test_architecture_serialization() {
        let arch = ModelArchitecture::Llama {
            version: LlamaVersion::V3_2,
        };
        let json = serde_json::to_string(&arch).expect("serialize");
        assert!(json.contains("llama"));

        let parsed: ModelArchitecture = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            ModelArchitecture::Llama { version } => {
                assert_eq!(version, LlamaVersion::V3_2);
            }
            _ => panic!("Wrong variant"),
        }
    }

    // ==========================================================================
    // ModelMetadataBuilder tests
    // ==========================================================================

    #[test]
    fn test_model_metadata_builder_basic() {
        let metadata = ModelMetadata::builder(
            "test-model",
            ModelArchitecture::Llama {
                version: LlamaVersion::V3_2,
            },
        )
        .source(ModelSource::huggingface("test/model"))
        .build();

        assert_eq!(metadata.id.0, "test-model");
        assert_eq!(metadata.context_length, 4096); // default
        assert_eq!(metadata.vocab_size, 32000); // default
    }

    #[test]
    fn test_model_metadata_builder_full() {
        let metadata = ModelMetadata::builder(
            "custom-model",
            ModelArchitecture::Mistral {
                variant: MistralVariant::Nemo,
            },
        )
        .source(ModelSource::local("/models/custom"))
        .context_length(131072)
        .vocab_size(128256)
        .hidden_size(8192)
        .num_layers(80)
        .num_attention_heads(64)
        .num_kv_heads(8)
        .quantization(QuantizationType::GgufQ4KM)
        .size_bytes(4_000_000_000)
        .description("Custom fine-tuned model")
        .build();

        assert_eq!(metadata.id.0, "custom-model");
        assert_eq!(metadata.context_length, 131072);
        assert_eq!(metadata.vocab_size, 128256);
        assert_eq!(metadata.hidden_size, 8192);
        assert_eq!(metadata.num_layers, 80);
        assert_eq!(metadata.num_attention_heads, 64);
        assert_eq!(metadata.num_kv_heads, Some(8));
        assert_eq!(metadata.quantization, Some(QuantizationType::GgufQ4KM));
        assert_eq!(metadata.size_bytes, Some(4_000_000_000));
        assert_eq!(
            metadata.description,
            Some("Custom fine-tuned model".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "source must be set")]
    fn test_model_metadata_builder_no_source_panics() {
        let _metadata = ModelMetadata::builder(
            "test",
            ModelArchitecture::Bert,
        )
        .build();
    }

    #[test]
    fn test_model_metadata_serialization() {
        let metadata = ModelMetadata::builder(
            "serialized-model",
            ModelArchitecture::Phi {
                version: PhiVersion::V4,
            },
        )
        .source(ModelSource::huggingface("microsoft/phi-4"))
        .context_length(8192)
        .build();

        let json = serde_json::to_string(&metadata).expect("serialize");
        assert!(json.contains("serialized-model"));
        assert!(json.contains("phi"));

        let parsed: ModelMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id.0, "serialized-model");
        assert_eq!(parsed.context_length, 8192);
    }

    // ==========================================================================
    // Version enum tests
    // ==========================================================================

    #[test]
    fn test_qwen_versions() {
        assert_eq!(QwenVersion::V2, QwenVersion::V2);
        assert_ne!(QwenVersion::V2, QwenVersion::V2_5);
    }

    #[test]
    fn test_phi_versions() {
        assert_eq!(PhiVersion::V3, PhiVersion::V3);
        assert_ne!(PhiVersion::V3, PhiVersion::V3_5);
        assert_ne!(PhiVersion::V3_5, PhiVersion::V4);
    }

    #[test]
    fn test_gemma_versions() {
        assert_eq!(GemmaVersion::V1, GemmaVersion::V1);
        assert_ne!(GemmaVersion::V1, GemmaVersion::V2);
    }

    #[test]
    fn test_architecture_debug() {
        let arch = ModelArchitecture::Mixtral { num_experts: 8 };
        let debug_str = format!("{:?}", arch);
        assert!(debug_str.contains("Mixtral"));
        assert!(debug_str.contains("8"));
    }

    #[test]
    fn test_architecture_clone() {
        let arch = ModelArchitecture::DeepSeek { version: 2 };
        let cloned = arch.clone();
        match (arch, cloned) {
            (
                ModelArchitecture::DeepSeek { version: v1 },
                ModelArchitecture::DeepSeek { version: v2 },
            ) => {
                assert_eq!(v1, v2);
            }
            _ => panic!("Clone failed"),
        }
    }
}
