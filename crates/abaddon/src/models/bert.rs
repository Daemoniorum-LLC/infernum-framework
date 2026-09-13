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

fn default_vocab_size() -> usize {
    30522
}
fn default_hidden_size() -> usize {
    768
}
fn default_num_hidden_layers() -> usize {
    12
}
fn default_num_attention_heads() -> usize {
    12
}
fn default_intermediate_size() -> usize {
    3072
}
fn default_max_position_embeddings() -> usize {
    512
}
fn default_type_vocab_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f64 {
    1e-12
}
fn default_pad_token_id() -> usize {
    0
}

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `Bert` is a thin wrapper: the transformer maths belongs to
// `candle_transformers` and is tested upstream. What this module actually owns
// — and what was previously untested — is the config layer: the serde
// defaults, the `is_jina_style` routing decision, and the two mappings onto
// the candle configs. Getting that routing wrong loads a *different
// architecture* against the same weights, which produces plausible-looking
// garbage rather than an error, so it is worth pinning precisely.
#[cfg(test)]
mod tests {
    use super::*;

    fn config_from(json: serde_json::Value) -> BertConfig {
        serde_json::from_value(json).expect("config deserializes")
    }

    /// An empty `config.json` must fall back to `bert-base-uncased` geometry.
    #[test]
    fn config_defaults_match_bert_base() {
        let cfg: BertConfig = serde_json::from_str("{}").expect("empty config deserializes");

        assert_eq!(cfg.vocab_size, 30522);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.max_position_embeddings, 512);
        assert_eq!(cfg.type_vocab_size, 2);
        assert_eq!(cfg.layer_norm_eps, 1e-12);
        assert_eq!(cfg.pad_token_id, 0);
        assert_eq!(cfg.position_embedding_type, None);
        assert_eq!(cfg.model_type, None);
        assert_eq!(cfg.rotary_emb_base, None);

        // No positional hints at all: this is a plain BERT.
        assert!(!cfg.is_jina_style());
    }

    #[test]
    fn jina_style_detected_from_position_embedding_type() {
        for pet in ["alibi", "ALiBi", "rotary", "ROTARY"] {
            let cfg = config_from(serde_json::json!({ "position_embedding_type": pet }));
            assert!(
                cfg.is_jina_style(),
                "position_embedding_type {pet:?} should select the Jina variant",
            );
        }
    }

    #[test]
    fn jina_style_detected_from_model_type() {
        for mt in ["nomic_bert", "NomicBert", "jina_bert", "JinaBertModel"] {
            let cfg = config_from(serde_json::json!({ "model_type": mt }));
            assert!(
                cfg.is_jina_style(),
                "model_type {mt:?} should select the Jina variant",
            );
        }
    }

    /// Standard BERT checkpoints — including ones that spell out
    /// `"absolute"` — must stay on the standard path.
    #[test]
    fn standard_bert_is_not_jina_style() {
        for json in [
            serde_json::json!({ "model_type": "bert" }),
            serde_json::json!({ "model_type": "bert", "position_embedding_type": "absolute" }),
            serde_json::json!({ "model_type": "roberta" }),
            serde_json::json!({}),
        ] {
            let cfg = config_from(json.clone());
            assert!(
                !cfg.is_jina_style(),
                "{json} should stay on the standard BERT path",
            );
        }
    }

    #[test]
    fn standard_config_mapping_carries_fields_through() {
        let cfg = config_from(serde_json::json!({
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 77,
            "type_vocab_size": 3,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 7,
            "model_type": "bert",
        }));
        let out = cfg.to_standard_config();

        assert_eq!(out.vocab_size, 1000);
        assert_eq!(out.hidden_size, 64);
        assert_eq!(out.num_hidden_layers, 2);
        assert_eq!(out.num_attention_heads, 4);
        assert_eq!(out.intermediate_size, 128);
        assert_eq!(out.max_position_embeddings, 77);
        assert_eq!(out.type_vocab_size, 3);
        assert_eq!(out.layer_norm_eps, 1e-5);
        assert_eq!(out.pad_token_id, 7);
        assert_eq!(out.model_type.as_deref(), Some("bert"));
    }

    #[test]
    fn jina_config_mapping_carries_fields_through() {
        let cfg = config_from(serde_json::json!({
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 77,
            "type_vocab_size": 3,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 7,
            "position_embedding_type": "alibi",
        }));
        let out = cfg.to_jina_config();

        assert_eq!(out.vocab_size, 1000);
        assert_eq!(out.hidden_size, 64);
        assert_eq!(out.num_hidden_layers, 2);
        assert_eq!(out.num_attention_heads, 4);
        assert_eq!(out.intermediate_size, 128);
        assert_eq!(out.max_position_embeddings, 77);
        assert_eq!(out.type_vocab_size, 3);
        assert_eq!(out.layer_norm_eps, 1e-5);
        assert_eq!(out.pad_token_id, 7);
        assert!(matches!(
            out.position_embedding_type,
            ct_jina::PositionEmbeddingType::Alibi
        ));
    }

    /// Documents a sharp edge rather than endorsing it: `"rotary"` is enough
    /// to route *away* from standard BERT, but `candle_transformers`' Jina
    /// model has no rotary mode, so the mapping silently lands on `Absolute`.
    ///
    /// Rotary checkpoints are meant to reach `NomicBert` (see
    /// `ArchitectureType::detect`), never this wrapper. If a config ever does
    /// arrive here with `"rotary"`, it gets absolute position embeddings and
    /// no warning — so this assertion exists to make that behaviour visible
    /// and to fail loudly if someone later wires rotary through here.
    #[test]
    fn rotary_position_type_falls_back_to_absolute_in_jina_mapping() {
        let cfg = config_from(serde_json::json!({ "position_embedding_type": "rotary" }));

        assert!(cfg.is_jina_style(), "rotary routes off the standard path");
        assert!(
            matches!(
                cfg.to_jina_config().position_embedding_type,
                ct_jina::PositionEmbeddingType::Absolute
            ),
            "candle's jina_bert has no rotary mode; the mapping degrades to Absolute",
        );
    }
}
