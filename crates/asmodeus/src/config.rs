//! Training configuration types.

use serde::{Deserialize, Serialize};

/// Configuration for LoRA adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of adaptation matrices.
    pub r: u32,
    /// Scaling factor (alpha).
    pub alpha: f32,
    /// Dropout probability.
    pub dropout: f32,
    /// Target modules to adapt.
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: u32,
    /// Number of epochs.
    pub num_epochs: u32,
    /// Warmup steps.
    pub warmup_steps: u32,
    /// Weight decay.
    pub weight_decay: f64,
    /// Gradient accumulation steps.
    pub gradient_accumulation_steps: u32,
    /// Maximum gradient norm.
    pub max_grad_norm: f64,
    /// LoRA configuration.
    pub lora: Option<LoraConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            warmup_steps: 100,
            weight_decay: 0.01,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
            lora: Some(LoraConfig::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // LoraConfig tests
    // ==========================================================================

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.r, 8);
        assert!((config.alpha - 16.0).abs() < 0.01);
        assert!((config.dropout - 0.05).abs() < 0.01);
        assert_eq!(config.target_modules.len(), 2);
        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));
    }

    #[test]
    fn test_lora_config_custom() {
        let config = LoraConfig {
            r: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        };

        assert_eq!(config.r, 16);
        assert!((config.alpha - 32.0).abs() < 0.01);
        assert!((config.dropout - 0.1).abs() < 0.01);
        assert_eq!(config.target_modules.len(), 4);
    }

    #[test]
    fn test_lora_config_serialization() {
        let config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec!["attn".to_string()],
        };

        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("\"r\":4"));
        assert!(json.contains("attn"));

        let parsed: LoraConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.r, 4);
        assert!((parsed.alpha - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_lora_config_clone() {
        let config = LoraConfig::default();
        let cloned = config.clone();

        assert_eq!(cloned.r, config.r);
        assert!((cloned.alpha - config.alpha).abs() < 0.01);
        assert_eq!(cloned.target_modules, config.target_modules);
    }

    #[test]
    fn test_lora_config_debug() {
        let config = LoraConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LoraConfig"));
        assert!(debug_str.contains("q_proj"));
    }

    // ==========================================================================
    // TrainingConfig tests
    // ==========================================================================

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!((config.learning_rate - 2e-4).abs() < 1e-6);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.warmup_steps, 100);
        assert!((config.weight_decay - 0.01).abs() < 0.001);
        assert_eq!(config.gradient_accumulation_steps, 4);
        assert!((config.max_grad_norm - 1.0).abs() < 0.01);
        assert!(config.lora.is_some());
    }

    #[test]
    fn test_training_config_custom() {
        let config = TrainingConfig {
            learning_rate: 1e-5,
            batch_size: 8,
            num_epochs: 10,
            warmup_steps: 500,
            weight_decay: 0.0,
            gradient_accumulation_steps: 1,
            max_grad_norm: 0.5,
            lora: None,
        };

        assert!((config.learning_rate - 1e-5).abs() < 1e-7);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.num_epochs, 10);
        assert!(config.lora.is_none());
    }

    #[test]
    fn test_training_config_serialization() {
        let config = TrainingConfig::default();

        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains("learning_rate"));
        assert!(json.contains("batch_size"));
        assert!(json.contains("lora"));

        let parsed: TrainingConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.batch_size, config.batch_size);
        assert_eq!(parsed.num_epochs, config.num_epochs);
    }

    #[test]
    fn test_training_config_without_lora() {
        let config = TrainingConfig {
            lora: None,
            ..Default::default()
        };

        assert!(config.lora.is_none());

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: TrainingConfig = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.lora.is_none());
    }

    #[test]
    fn test_training_config_clone() {
        let config = TrainingConfig::default();
        let cloned = config.clone();

        assert!((cloned.learning_rate - config.learning_rate).abs() < 1e-7);
        assert_eq!(cloned.batch_size, config.batch_size);
        assert!(cloned.lora.is_some());
    }

    #[test]
    fn test_training_config_debug() {
        let config = TrainingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("TrainingConfig"));
        assert!(debug_str.contains("learning_rate"));
    }

    // ==========================================================================
    // Integration tests
    // ==========================================================================

    #[test]
    fn test_training_config_with_custom_lora() {
        let lora = LoraConfig {
            r: 32,
            alpha: 64.0,
            dropout: 0.1,
            target_modules: vec!["all".to_string()],
        };

        let config = TrainingConfig {
            lora: Some(lora),
            ..Default::default()
        };

        assert!(config.lora.is_some());
        let lora_config = config.lora.as_ref().unwrap();
        assert_eq!(lora_config.r, 32);
    }

    #[test]
    fn test_effective_batch_size() {
        let config = TrainingConfig {
            batch_size: 8,
            gradient_accumulation_steps: 4,
            ..Default::default()
        };

        // Effective batch size = batch_size * gradient_accumulation_steps
        let effective_batch = config.batch_size * config.gradient_accumulation_steps;
        assert_eq!(effective_batch, 32);
    }
}
