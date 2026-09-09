//! BERT-family embedding models (BERT, NomicBERT, Jina BERT).
//!
//! Wraps candle-transformers' BERT implementations to provide embedding
//! extraction through the same `ModelKind` interface the rest of the engine
//! uses. Two inner variants cover the two distinct attention schemes:
//!
//! - **Standard BERT** (`candle_transformers::models::bert`) — absolute
//!   position embeddings, needs `token_type_ids`. Covers `bert-base-uncased`,
//!   `bge-base-en-v1.5`, `all-MiniLM-L6-v2`, etc.
//!
//! - **Jina/Nomic BERT** (`candle_transformers::models::jina_bert`) — ALiBi
//!   positional bias, no `token_type_ids`. Covers `nomic-embed-text-v1.5`,
//!   `jina-embeddings-v2-*`, etc.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};

use candle_transformers::models::bert as ct_bert;
use candle_transformers::models::jina_bert as ct_jina;

/// Which inner BERT implementation to use.
enum BertInner {
    Standard(ct_bert::BertModel),
    Jina(ct_jina::BertModel),
}

/// Unified BERT embedding model.
pub struct Bert {
    inner: BertInner,
    device: Device,
}

/// Configuration parsed from a BERT model's `config.json`.
///
/// A superset of the fields both candle BERT variants need, with serde
/// defaults so that missing keys fall back to standard BERT values.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BertConfig {
    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Hidden layer dimension.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Number of transformer layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// FFN intermediate dimension.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Token-type vocabulary size.
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    /// Layer-norm epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Padding token ID.
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
    /// Position embedding strategy (`"absolute"`, `"alibi"`, `"rotary"`).
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    /// HuggingFace `model_type` field (e.g. `"bert"`, `"nomic_bert"`).
    #[serde(default)]
    pub model_type: Option<String>,
    /// Rotary embedding base frequency (NomicBERT).
    #[serde(default)]
    pub rotary_emb_base: Option<f64>,
}

fn default_vocab_size() -> usize { 30522 }
fn default_hidden_size() -> usize { 768 }
fn default_num_hidden_layers() -> usize { 12 }
fn default_num_attention_heads() -> usize { 12 }
fn default_intermediate_size() -> usize { 3072 }
fn default_max_position_embeddings() -> usize { 512 }
fn default_type_vocab_size() -> usize { 2 }
fn default_layer_norm_eps() -> f64 { 1e-12 }
fn default_pad_token_id() -> usize { 0 }

impl BertConfig {
    /// Returns `true` when the config indicates a Jina/Nomic-style BERT
    /// (ALiBi or rotary positions rather than absolute learned embeddings).
    fn is_jina_style(&self) -> bool {
        if let Some(ref pet) = self.position_embedding_type {
            let lower = pet.to_lowercase();
            if lower == "alibi" || lower == "rotary" {
                return true;
            }
        }
        if let Some(ref mt) = self.model_type {
            let lower = mt.to_lowercase();
            if lower.contains("nomic") || lower.contains("jina") {
                return true;
            }
        }
        false
    }

    fn to_standard_config(&self) -> ct_bert::Config {
        ct_bert::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            hidden_act: ct_bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: self.layer_norm_eps,
            pad_token_id: self.pad_token_id,
            position_embedding_type: ct_bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: self.model_type.clone(),
        }
    }

    fn to_jina_config(&self) -> ct_jina::Config {
        let pet = match self.position_embedding_type.as_deref() {
            Some("alibi") => ct_jina::PositionEmbeddingType::Alibi,
            _ => ct_jina::PositionEmbeddingType::Absolute,
        };
        ct_jina::Config::new(
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            candle_nn::Activation::Gelu,
            self.max_position_embeddings,
            self.type_vocab_size,
            0.02, // initializer_range
            self.layer_norm_eps,
            self.pad_token_id,
            pet,
        )
    }
}

impl Bert {
    /// Loads a BERT model, automatically selecting the standard or Jina/Nomic
    /// variant based on `config.position_embedding_type` and `model_type`.
    pub fn load(config: BertConfig, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();

        let inner = if config.is_jina_style() {
            let jina_config = config.to_jina_config();
            let model = ct_jina::BertModel::new(vb, &jina_config)?;
            BertInner::Jina(model)
        } else {
            let std_config = config.to_standard_config();
            let model = ct_bert::BertModel::load(vb, &std_config)?;
            BertInner::Standard(model)
        };

        Ok(Self { inner, device })
    }

    /// Forward pass returning sequence hidden states.
    ///
    /// For standard BERT, synthesizes zero `token_type_ids` (single-sentence
    /// embedding). For Jina/Nomic BERT, delegates directly.
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            BertInner::Standard(model) => {
                let token_type_ids = input_ids.zeros_like()?;
                model.forward(input_ids, &token_type_ids, None)
            },
            BertInner::Jina(model) => model.forward(input_ids),
        }
    }

    /// Forward pass for embedding extraction — same as `forward` for BERT
    /// (there is no separate lm_head to skip).
    pub fn forward_embedding(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward(input_ids)
    }

    /// Mean-pool over the sequence dimension to produce a single embedding
    /// vector per input in the batch. Always returns F32.
    pub fn extract_embeddings(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.forward(input_ids)?;
        let pooled = hidden_states.mean(1)?;
        pooled.to_dtype(candle_core::DType::F32)
    }

    /// No-op: BERT has no KV cache.
    pub fn clear_cache(&mut self) {}

    /// Returns the device this model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
