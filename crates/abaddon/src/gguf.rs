//! GGUF file format loading support.
//!
//! GGUF (GPT-Generated Unified Format) is a binary format for storing quantized
//! LLM models. This module provides loading and inference support for GGUF files.

use std::path::Path;

use candle_core::quantized::gguf_file;
use infernum_core::Result;

/// GGUF file reader and parser.
pub struct GgufLoader {
    content: gguf_file::Content,
    metadata: GgufMetadata,
}

/// Extracted metadata from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// Model architecture (e.g., "llama", "mistral", "qwen2").
    pub architecture: String,
    /// Model name if available.
    pub name: Option<String>,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Intermediate size (FFN dimension).
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub context_length: usize,
    /// RoPE theta value.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// Quantization type.
    pub quantization_type: String,
    /// BOS token ID.
    pub bos_token_id: Option<u32>,
    /// EOS token ID.
    pub eos_token_id: Option<u32>,
    /// Pad token ID.
    pub pad_token_id: Option<u32>,
}

impl GgufLoader {
    /// Loads a GGUF file from the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!(path = %path.display(), "Loading GGUF file");

        let file = std::fs::File::open(path).map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Failed to open GGUF file: {}", e),
        })?;

        let content =
            gguf_file::Content::read(&mut std::io::BufReader::new(file)).map_err(|e| {
                infernum_core::Error::ModelLoad {
                    message: format!("Failed to parse GGUF file: {}", e),
                }
            })?;

        let metadata = Self::extract_metadata(&content)?;

        tracing::info!(
            architecture = %metadata.architecture,
            layers = metadata.num_layers,
            vocab_size = metadata.vocab_size,
            quantization = %metadata.quantization_type,
            "Loaded GGUF metadata"
        );

        Ok(Self { content, metadata })
    }

    /// Extracts metadata from the GGUF content.
    fn extract_metadata(content: &gguf_file::Content) -> Result<GgufMetadata> {
        let metadata = &content.metadata;

        // Helper to get string metadata
        let get_str = |key: &str| -> Option<String> {
            metadata.get(key).and_then(|v| {
                if let gguf_file::Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
        };

        // Helper to get u32 metadata
        let get_u32 = |key: &str| -> Option<u32> {
            metadata.get(key).and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n),
                gguf_file::Value::I32(n) => Some(*n as u32),
                gguf_file::Value::U64(n) => Some(*n as u32),
                gguf_file::Value::I64(n) => Some(*n as u32),
                _ => None,
            })
        };

        // Helper to get usize metadata
        let get_usize = |key: &str| -> Option<usize> {
            metadata.get(key).and_then(|v| match v {
                gguf_file::Value::U32(n) => Some(*n as usize),
                gguf_file::Value::I32(n) => Some(*n as usize),
                gguf_file::Value::U64(n) => Some(*n as usize),
                gguf_file::Value::I64(n) => Some(*n as usize),
                _ => None,
            })
        };

        // Helper to get f64 metadata
        let get_f64 = |key: &str| -> Option<f64> {
            metadata.get(key).and_then(|v| match v {
                gguf_file::Value::F32(n) => Some(*n as f64),
                gguf_file::Value::F64(n) => Some(*n),
                _ => None,
            })
        };

        // Extract architecture (required)
        let architecture =
            get_str("general.architecture").ok_or_else(|| infernum_core::Error::ModelLoad {
                message: "Missing general.architecture in GGUF metadata".to_string(),
            })?;

        let arch_prefix = format!("{}.", architecture);

        // Extract model parameters
        let num_attention_heads = get_usize(&format!("{arch_prefix}attention.head_count"))
            .or_else(|| get_usize(&format!("{arch_prefix}head_count")))
            .unwrap_or(32);

        let num_kv_heads = get_usize(&format!("{arch_prefix}attention.head_count_kv"))
            .or_else(|| get_usize(&format!("{arch_prefix}head_count_kv")))
            .unwrap_or(num_attention_heads);

        let num_layers = get_usize(&format!("{arch_prefix}block_count"))
            .or_else(|| get_usize(&format!("{arch_prefix}layer_count")))
            .unwrap_or(32);

        let hidden_size = get_usize(&format!("{arch_prefix}embedding_length"))
            .or_else(|| get_usize(&format!("{arch_prefix}hidden_size")))
            .unwrap_or(4096);

        let intermediate_size = get_usize(&format!("{arch_prefix}feed_forward_length"))
            .or_else(|| get_usize(&format!("{arch_prefix}intermediate_size")))
            .unwrap_or(hidden_size * 4);

        let vocab_size = get_usize(&format!("{arch_prefix}vocab_size"))
            .or_else(|| get_usize("tokenizer.ggml.vocab_size"))
            .unwrap_or(32000);

        let context_length = get_usize(&format!("{arch_prefix}context_length"))
            .or_else(|| get_usize(&format!("{arch_prefix}max_position_embeddings")))
            .unwrap_or(4096);

        let rope_theta = get_f64(&format!("{arch_prefix}rope.freq_base"))
            .or_else(|| get_f64(&format!("{arch_prefix}rope_theta")))
            .unwrap_or(10000.0);

        let rms_norm_eps = get_f64(&format!("{arch_prefix}attention.layer_norm_rms_epsilon"))
            .or_else(|| get_f64(&format!("{arch_prefix}rms_norm_eps")))
            .unwrap_or(1e-5);

        // Extract token IDs
        let bos_token_id = get_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = get_u32("tokenizer.ggml.eos_token_id");
        let pad_token_id = get_u32("tokenizer.ggml.padding_token_id");

        // Determine quantization type from tensor info
        let quantization_type = Self::detect_quantization_type(content);

        Ok(GgufMetadata {
            architecture,
            name: get_str("general.name"),
            num_attention_heads,
            num_kv_heads,
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            context_length,
            rope_theta,
            rms_norm_eps,
            quantization_type,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Detects the quantization type from tensor info.
    fn detect_quantization_type(content: &gguf_file::Content) -> String {
        // Look at the first weight tensor to determine quantization
        for (name, info) in content.tensor_infos.iter() {
            if name.contains("weight") {
                return format!("{:?}", info.ggml_dtype).to_lowercase();
            }
        }
        "unknown".to_string()
    }

    /// Returns the extracted metadata.
    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Returns the raw GGUF content.
    #[must_use]
    pub fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    /// Checks if a tensor exists in the GGUF file.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// Lists all tensor names in the GGUF file.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.content
            .tensor_infos
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Extracts vocabulary tokens if available.
    #[must_use]
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        self.content
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| {
                if let gguf_file::Value::Array(arr) = v {
                    let tokens: Vec<String> = arr
                        .iter()
                        .filter_map(|item| {
                            if let gguf_file::Value::String(s) = item {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !tokens.is_empty() {
                        Some(tokens)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    /// Returns the number of tensors in the GGUF file.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.content.tensor_infos.len()
    }
}

/// Configuration for a quantized model loaded from GGUF.
#[derive(Debug, Clone)]
pub struct QuantizedModelConfig {
    /// Model architecture.
    pub architecture: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length.
    pub context_length: usize,
    /// RoPE theta.
    pub rope_theta: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// BOS token ID.
    pub bos_token_id: Option<u32>,
    /// EOS token ID.
    pub eos_token_id: Option<u32>,
}

impl From<&GgufMetadata> for QuantizedModelConfig {
    fn from(metadata: &GgufMetadata) -> Self {
        Self {
            architecture: metadata.architecture.clone(),
            num_layers: metadata.num_layers,
            hidden_size: metadata.hidden_size,
            num_attention_heads: metadata.num_attention_heads,
            num_kv_heads: metadata.num_kv_heads,
            vocab_size: metadata.vocab_size,
            context_length: metadata.context_length,
            rope_theta: metadata.rope_theta,
            rms_norm_eps: metadata.rms_norm_eps,
            bos_token_id: metadata.bos_token_id,
            eos_token_id: metadata.eos_token_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_config_from_metadata() {
        let metadata = GgufMetadata {
            architecture: "llama".to_string(),
            name: Some("test".to_string()),
            num_attention_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            context_length: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            quantization_type: "q4_k_m".to_string(),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: None,
        };

        let config = QuantizedModelConfig::from(&metadata);
        assert_eq!(config.architecture, "llama");
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_kv_heads, 8);
    }
}
