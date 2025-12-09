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
