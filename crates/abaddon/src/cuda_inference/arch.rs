//! Model architecture definitions and configurations.
//!
//! Supports multiple transformer architectures with a unified interface.

use std::path::Path;

use super::InferenceError;

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// Llama 1/2/3 and CodeLlama.
    Llama,
    /// Mistral and Mixtral (MoE).
    Mistral,
    /// Qwen and Qwen2.
    Qwen,
    /// Microsoft Phi-2/3.
    Phi,
    /// Google Gemma.
    Gemma,
    /// Falcon.
    Falcon,
    /// GPT-NeoX / Pythia.
    GptNeoX,
    /// StarCoder.
    StarCoder,
}

impl ModelArch {
    /// Detect architecture from config.json or model files.
    pub fn detect(model_dir: &Path) -> Result<Self, InferenceError> {
        let config_path = model_dir.join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;

            return Self::from_config_json(&config_str);
        }

        // Try to detect from file naming patterns
        let dir_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if dir_name.contains("llama") || dir_name.contains("codellama") {
            Ok(ModelArch::Llama)
        } else if dir_name.contains("mistral") || dir_name.contains("mixtral") {
            Ok(ModelArch::Mistral)
        } else if dir_name.contains("qwen") {
            Ok(ModelArch::Qwen)
        } else if dir_name.contains("phi") {
            Ok(ModelArch::Phi)
        } else if dir_name.contains("gemma") {
            Ok(ModelArch::Gemma)
        } else {
            // Default to Llama-style (most common)
            Ok(ModelArch::Llama)
        }
    }

    /// Parse architecture from HuggingFace config.json.
    pub fn from_config_json(config: &str) -> Result<Self, InferenceError> {
        let json: serde_json::Value = serde_json::from_str(config)
            .map_err(|e| InferenceError::ModelLoad(format!("Invalid config.json: {}", e)))?;

        // Check architectures field
        if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
            for arch in archs {
                if let Some(arch_str) = arch.as_str() {
                    let arch_lower = arch_str.to_lowercase();

                    if arch_lower.contains("llama") {
                        return Ok(ModelArch::Llama);
                    } else if arch_lower.contains("mistral") || arch_lower.contains("mixtral") {
                        return Ok(ModelArch::Mistral);
                    } else if arch_lower.contains("qwen") {
                        return Ok(ModelArch::Qwen);
                    } else if arch_lower.contains("phi") {
                        return Ok(ModelArch::Phi);
                    } else if arch_lower.contains("gemma") {
                        return Ok(ModelArch::Gemma);
                    } else if arch_lower.contains("falcon") {
                        return Ok(ModelArch::Falcon);
                    } else if arch_lower.contains("neox") || arch_lower.contains("pythia") {
                        return Ok(ModelArch::GptNeoX);
                    } else if arch_lower.contains("starcoder") {
                        return Ok(ModelArch::StarCoder);
                    }
                }
            }
        }

        // Check model_type field
        if let Some(model_type) = json.get("model_type").and_then(|v| v.as_str()) {
            let model_lower = model_type.to_lowercase();

            if model_lower == "llama" {
                return Ok(ModelArch::Llama);
            } else if model_lower == "mistral" {
                return Ok(ModelArch::Mistral);
            } else if model_lower == "qwen" || model_lower == "qwen2" {
                return Ok(ModelArch::Qwen);
            } else if model_lower == "phi" || model_lower == "phi3" {
                return Ok(ModelArch::Phi);
            } else if model_lower == "gemma" || model_lower == "gemma2" {
                return Ok(ModelArch::Gemma);
            }
        }

        // Default to Llama
        Ok(ModelArch::Llama)
    }

    /// Get default weight name mappings for this architecture.
    pub fn weight_map(&self) -> WeightNameMap {
        match self {
            ModelArch::Llama | ModelArch::Mistral | ModelArch::Qwen => WeightNameMap::llama_style(),
            ModelArch::Phi => WeightNameMap::phi_style(),
            ModelArch::Gemma => WeightNameMap::gemma_style(),
            ModelArch::Falcon => WeightNameMap::falcon_style(),
            ModelArch::GptNeoX | ModelArch::StarCoder => WeightNameMap::gpt_neox_style(),
        }
    }
}

/// Model configuration parameters.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model architecture.
    pub arch: ModelArch,

    /// Hidden dimension size.
    pub hidden_size: usize,

    /// Intermediate (MLP) dimension size.
    pub intermediate_size: usize,

    /// Number of transformer layers.
    pub num_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,

    /// Dimension per attention head.
    pub head_dim: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// RMS norm epsilon.
    pub rms_norm_eps: f32,

    /// RoPE theta (base frequency).
    pub rope_theta: f32,

    /// RoPE scaling factor (for extended context).
    pub rope_scaling: Option<RopeScaling>,

    /// Whether to use bias in attention.
    pub attention_bias: bool,

    /// Whether to use bias in MLP.
    pub mlp_bias: bool,

    /// Activation function for MLP.
    pub hidden_act: Activation,

    /// Whether embeddings are tied to LM head.
    pub tie_word_embeddings: bool,

    /// Sliding window attention size (Mistral/Mixtral).
    /// If Some, attention is limited to this window size.
    pub sliding_window: Option<usize>,

    /// BOS token ID.
    pub bos_token_id: u32,

    /// EOS token ID.
    pub eos_token_id: u32,

    /// Padding token ID.
    pub pad_token_id: Option<u32>,
}

impl ModelConfig {
    /// Load config from HuggingFace config.json.
    pub fn from_json(config: &str, arch: ModelArch) -> Result<Self, InferenceError> {
        let json: serde_json::Value = serde_json::from_str(config)
            .map_err(|e| InferenceError::ModelLoad(format!("Invalid config: {}", e)))?;

        let get_usize = |key: &str, default: usize| -> usize {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_f32 = |key: &str, default: f32| -> f32 {
            json.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default)
        };

        let get_bool = |key: &str, default: bool| -> bool {
            json.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
        };

        let hidden_size = get_usize("hidden_size", 4096);
        let num_attention_heads = get_usize("num_attention_heads", 32);

        // Handle different naming conventions
        let num_kv_heads = json
            .get("num_key_value_heads")
            .or_else(|| json.get("num_kv_heads"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let head_dim = json
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let hidden_act = json
            .get("hidden_act")
            .and_then(|v| v.as_str())
            .map(Activation::from_str)
            .unwrap_or(Activation::SiLU);

        let rope_scaling = json.get("rope_scaling").and_then(|v| {
            let scale_type = v.get("type").and_then(|t| t.as_str())?;
            let factor = v.get("factor").and_then(|f| f.as_f64())? as f32;
            Some(RopeScaling::from_type(scale_type, factor))
        });

        // Sliding window attention (Mistral/Mixtral feature)
        let sliding_window = json
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Ok(Self {
            arch,
            hidden_size,
            intermediate_size: get_usize("intermediate_size", hidden_size * 4),
            num_layers: get_usize("num_hidden_layers", 32),
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size: get_usize("vocab_size", 32000),
            max_seq_len: get_usize("max_position_embeddings", 4096),
            rms_norm_eps: get_f32("rms_norm_eps", 1e-6),
            rope_theta: get_f32("rope_theta", 10000.0),
            rope_scaling,
            attention_bias: get_bool("attention_bias", false),
            mlp_bias: get_bool("mlp_bias", false),
            hidden_act,
            tie_word_embeddings: get_bool("tie_word_embeddings", false),
            sliding_window,
            bos_token_id: get_usize("bos_token_id", 1) as u32,
            eos_token_id: get_usize("eos_token_id", 2) as u32,
            pad_token_id: json
                .get("pad_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        })
    }

    /// Create config with default values for an architecture.
    pub fn default_for_arch(arch: ModelArch, hidden_size: usize, num_layers: usize) -> Self {
        let num_attention_heads = match hidden_size {
            h if h <= 1024 => 8,
            h if h <= 2048 => 16,
            h if h <= 4096 => 32,
            _ => 64,
        };

        // Mistral uses sliding window attention by default
        let sliding_window = match arch {
            ModelArch::Mistral => Some(4096),
            _ => None,
        };

        Self {
            arch,
            hidden_size,
            intermediate_size: hidden_size * 4,
            num_layers,
            num_attention_heads,
            num_kv_heads: num_attention_heads,
            head_dim: hidden_size / num_attention_heads,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: false,
            sliding_window,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        }
    }

    /// Total number of parameters (approximate).
    pub fn num_params(&self) -> usize {
        let embed_params = self.vocab_size * self.hidden_size;

        let attn_params = self.num_layers * (self.hidden_size * self.hidden_size * 4); // Q, K, V, O

        let mlp_params = self.num_layers * (self.hidden_size * self.intermediate_size * 3); // gate, up, down

        let norm_params = self.num_layers * self.hidden_size * 2 + self.hidden_size;

        let lm_head = if self.tie_word_embeddings {
            0
        } else {
            self.vocab_size * self.hidden_size
        };

        embed_params + attn_params + mlp_params + norm_params + lm_head
    }
}

/// RoPE scaling configuration.
#[derive(Debug, Clone, Copy)]
pub enum RopeScaling {
    /// Linear scaling.
    Linear(f32),
    /// Dynamic NTK scaling.
    Dynamic(f32),
    /// YaRN scaling.
    Yarn {
        /// Scaling factor.
        factor: f32,
        /// Original maximum sequence length before scaling.
        original_max_len: usize,
    },
}

impl RopeScaling {
    fn from_type(scale_type: &str, factor: f32) -> Self {
        match scale_type.to_lowercase().as_str() {
            "linear" => RopeScaling::Linear(factor),
            "dynamic" => RopeScaling::Dynamic(factor),
            "yarn" => RopeScaling::Yarn {
                factor,
                original_max_len: 4096,
            },
            _ => RopeScaling::Linear(factor),
        }
    }
}

/// Activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Sigmoid Linear Unit (Llama, Mistral).
    SiLU,
    /// Gaussian Error Linear Unit.
    GELU,
    /// Approximate GELU (GPT-2 style).
    GELUApprox,
    /// ReLU.
    ReLU,
    /// Squared ReLU (Phi).
    ReLU2,
}

impl Activation {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "silu" | "swish" => Activation::SiLU,
            "gelu" => Activation::GELU,
            "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Activation::GELUApprox,
            "relu" => Activation::ReLU,
            "relu2" | "squared_relu" => Activation::ReLU2,
            _ => Activation::SiLU,
        }
    }
}

/// Mapping of weight names from HuggingFace format to our internal names.
#[derive(Debug, Clone)]
pub struct WeightNameMap {
    /// Pattern mappings: (hf_pattern, internal_name).
    mappings: Vec<(String, String)>,
}

impl WeightNameMap {
    /// Create Llama-style weight mappings.
    pub fn llama_style() -> Self {
        Self {
            mappings: vec![
                // Embeddings
                ("model.embed_tokens.weight".into(), "embed_tokens".into()),
                ("lm_head.weight".into(), "lm_head".into()),
                ("model.norm.weight".into(), "final_norm".into()),
                // Per-layer patterns (use {layer} as placeholder)
                (
                    "model.layers.{layer}.input_layernorm.weight".into(),
                    "layers.{layer}.input_norm".into(),
                ),
                (
                    "model.layers.{layer}.post_attention_layernorm.weight".into(),
                    "layers.{layer}.post_attn_norm".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.q_proj.weight".into(),
                    "layers.{layer}.q_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.k_proj.weight".into(),
                    "layers.{layer}.k_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.v_proj.weight".into(),
                    "layers.{layer}.v_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.o_proj.weight".into(),
                    "layers.{layer}.o_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.gate_proj.weight".into(),
                    "layers.{layer}.gate_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.up_proj.weight".into(),
                    "layers.{layer}.up_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.down_proj.weight".into(),
                    "layers.{layer}.down_proj".into(),
                ),
            ],
        }
    }

    /// Create Phi-style weight mappings.
    pub fn phi_style() -> Self {
        Self {
            mappings: vec![
                ("model.embed_tokens.weight".into(), "embed_tokens".into()),
                ("lm_head.weight".into(), "lm_head".into()),
                ("model.final_layernorm.weight".into(), "final_norm".into()),
                (
                    "model.layers.{layer}.input_layernorm.weight".into(),
                    "layers.{layer}.input_norm".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.q_proj.weight".into(),
                    "layers.{layer}.q_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.k_proj.weight".into(),
                    "layers.{layer}.k_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.v_proj.weight".into(),
                    "layers.{layer}.v_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.dense.weight".into(),
                    "layers.{layer}.o_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.fc1.weight".into(),
                    "layers.{layer}.gate_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.fc2.weight".into(),
                    "layers.{layer}.down_proj".into(),
                ),
            ],
        }
    }

    /// Create Gemma-style weight mappings.
    pub fn gemma_style() -> Self {
        Self {
            mappings: vec![
                ("model.embed_tokens.weight".into(), "embed_tokens".into()),
                ("model.norm.weight".into(), "final_norm".into()),
                (
                    "model.layers.{layer}.input_layernorm.weight".into(),
                    "layers.{layer}.input_norm".into(),
                ),
                (
                    "model.layers.{layer}.post_attention_layernorm.weight".into(),
                    "layers.{layer}.post_attn_norm".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.q_proj.weight".into(),
                    "layers.{layer}.q_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.k_proj.weight".into(),
                    "layers.{layer}.k_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.v_proj.weight".into(),
                    "layers.{layer}.v_proj".into(),
                ),
                (
                    "model.layers.{layer}.self_attn.o_proj.weight".into(),
                    "layers.{layer}.o_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.gate_proj.weight".into(),
                    "layers.{layer}.gate_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.up_proj.weight".into(),
                    "layers.{layer}.up_proj".into(),
                ),
                (
                    "model.layers.{layer}.mlp.down_proj.weight".into(),
                    "layers.{layer}.down_proj".into(),
                ),
            ],
        }
    }

    /// Create Falcon-style weight mappings.
    pub fn falcon_style() -> Self {
        Self {
            mappings: vec![
                (
                    "transformer.word_embeddings.weight".into(),
                    "embed_tokens".into(),
                ),
                ("lm_head.weight".into(), "lm_head".into()),
                ("transformer.ln_f.weight".into(), "final_norm".into()),
                (
                    "transformer.h.{layer}.input_layernorm.weight".into(),
                    "layers.{layer}.input_norm".into(),
                ),
                (
                    "transformer.h.{layer}.self_attention.query_key_value.weight".into(),
                    "layers.{layer}.qkv_proj".into(),
                ),
                (
                    "transformer.h.{layer}.self_attention.dense.weight".into(),
                    "layers.{layer}.o_proj".into(),
                ),
                (
                    "transformer.h.{layer}.mlp.dense_h_to_4h.weight".into(),
                    "layers.{layer}.gate_proj".into(),
                ),
                (
                    "transformer.h.{layer}.mlp.dense_4h_to_h.weight".into(),
                    "layers.{layer}.down_proj".into(),
                ),
            ],
        }
    }

    /// Create GPT-NeoX style weight mappings.
    pub fn gpt_neox_style() -> Self {
        Self {
            mappings: vec![
                ("gpt_neox.embed_in.weight".into(), "embed_tokens".into()),
                ("embed_out.weight".into(), "lm_head".into()),
                (
                    "gpt_neox.final_layer_norm.weight".into(),
                    "final_norm".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.input_layernorm.weight".into(),
                    "layers.{layer}.input_norm".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.post_attention_layernorm.weight".into(),
                    "layers.{layer}.post_attn_norm".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.attention.query_key_value.weight".into(),
                    "layers.{layer}.qkv_proj".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.attention.dense.weight".into(),
                    "layers.{layer}.o_proj".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight".into(),
                    "layers.{layer}.gate_proj".into(),
                ),
                (
                    "gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight".into(),
                    "layers.{layer}.down_proj".into(),
                ),
            ],
        }
    }

    /// Map a HuggingFace weight name to internal name.
    pub fn map_name(&self, hf_name: &str) -> Option<String> {
        // Try to extract layer number if present
        let layer_num = extract_layer_number(hf_name);

        for (pattern, internal) in &self.mappings {
            if let Some(layer) = layer_num {
                let expanded_pattern = pattern.replace("{layer}", &layer.to_string());
                if hf_name == expanded_pattern {
                    return Some(internal.replace("{layer}", &layer.to_string()));
                }
            } else if !pattern.contains("{layer}") && hf_name == pattern {
                return Some(internal.clone());
            }
        }

        None
    }
}

/// Extract layer number from weight name like "model.layers.5.self_attn.q_proj.weight".
fn extract_layer_number(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" || *part == "h" {
            if let Some(num_str) = parts.get(i + 1) {
                if let Ok(num) = num_str.parse::<usize>() {
                    return Some(num);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_name_mapping() {
        let map = WeightNameMap::llama_style();

        assert_eq!(
            map.map_name("model.embed_tokens.weight"),
            Some("embed_tokens".to_string())
        );

        assert_eq!(
            map.map_name("model.layers.5.self_attn.q_proj.weight"),
            Some("layers.5.q_proj".to_string())
        );

        assert_eq!(
            map.map_name("model.layers.31.mlp.down_proj.weight"),
            Some("layers.31.down_proj".to_string())
        );
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            extract_layer_number("model.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
        assert_eq!(
            extract_layer_number("transformer.h.12.mlp.weight"),
            Some(12)
        );
        assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
    }
}
