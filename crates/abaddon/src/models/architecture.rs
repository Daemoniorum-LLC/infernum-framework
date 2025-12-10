//! Model architecture detection and configuration.
//!
//! Detects the model architecture from config.json and provides
//! architecture-specific configurations for proper model loading.

use serde::Deserialize;
use std::path::Path;

use infernum_core::Result;

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Llama 2, Llama 3, Llama 3.1, Llama 3.2
    Llama,
    /// Qwen2 series (based on Llama but with modifications)
    Qwen2,
    /// Mistral series
    Mistral,
    /// Unknown/unsupported architecture
    Unknown,
}

/// Minimal config structure for architecture detection.
#[derive(Debug, Deserialize)]
struct ArchitectureDetectionConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Option<Vec<String>>,
}

impl Architecture {
    /// Detects the model architecture from a config.json file.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file cannot be read or parsed.
    pub fn detect_from_config(config_path: impl AsRef<Path>) -> Result<Self> {
        let config_str = std::fs::read_to_string(config_path).map_err(|e| {
            infernum_core::Error::Internal {
                message: format!("Failed to read config.json: {}", e),
            }
        })?;

        let config: ArchitectureDetectionConfig =
            serde_json::from_str(&config_str).map_err(|e| {
                infernum_core::Error::Internal {
                    message: format!("Failed to parse config.json: {}", e),
                }
            })?;

        Ok(Self::from_config(&config))
    }

    /// Determines architecture from config fields.
    fn from_config(config: &ArchitectureDetectionConfig) -> Self {
        // First try model_type field
        if let Some(model_type) = &config.model_type {
            match model_type.as_str() {
                "llama" => return Self::Llama,
                "qwen2" => return Self::Qwen2,
                "mistral" => return Self::Mistral,
                _ => {}
            }
        }

        // Fall back to architectures field
        if let Some(architectures) = &config.architectures {
            for arch in architectures {
                if arch.contains("Llama") {
                    return Self::Llama;
                } else if arch.contains("Qwen2") {
                    return Self::Qwen2;
                } else if arch.contains("Mistral") {
                    return Self::Mistral;
                }
            }
        }

        Self::Unknown
    }

    /// Returns a human-readable name for the architecture.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Llama => "Llama",
            Self::Qwen2 => "Qwen2",
            Self::Mistral => "Mistral",
            Self::Unknown => "Unknown",
        }
    }

    /// Returns whether this architecture is compatible with the Llama implementation.
    ///
    /// Qwen2 and Mistral are Llama-based architectures that can use the same
    /// core implementation with different configuration parameters.
    #[must_use]
    pub fn is_llama_compatible(&self) -> bool {
        matches!(self, Self::Llama | Self::Qwen2 | Self::Mistral)
    }

    /// Returns architecture-specific notes or warnings.
    #[must_use]
    pub fn notes(&self) -> Option<&str> {
        match self {
            Self::Qwen2 => Some(
                "Qwen2 uses very large RoPE theta (1M vs 10K) and smaller RMS norm eps (1e-6 vs 1e-5)",
            ),
            Self::Mistral => Some(
                "Mistral may use sliding window attention in some layers",
            ),
            Self::Unknown => Some(
                "Unknown architecture - may not work correctly with current implementation",
            ),
            Self::Llama => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama() {
        let config = ArchitectureDetectionConfig {
            model_type: Some("llama".to_string()),
            architectures: None,
        };
        assert_eq!(Architecture::from_config(&config), Architecture::Llama);
    }

    #[test]
    fn test_detect_qwen2() {
        let config = ArchitectureDetectionConfig {
            model_type: Some("qwen2".to_string()),
            architectures: Some(vec!["Qwen2ForCausalLM".to_string()]),
        };
        assert_eq!(Architecture::from_config(&config), Architecture::Qwen2);
    }

    #[test]
    fn test_detect_from_architectures() {
        let config = ArchitectureDetectionConfig {
            model_type: None,
            architectures: Some(vec!["MistralForCausalLM".to_string()]),
        };
        assert_eq!(Architecture::from_config(&config), Architecture::Mistral);
    }

    #[test]
    fn test_llama_compatible() {
        assert!(Architecture::Llama.is_llama_compatible());
        assert!(Architecture::Qwen2.is_llama_compatible());
        assert!(Architecture::Mistral.is_llama_compatible());
        assert!(!Architecture::Unknown.is_llama_compatible());
    }
}
