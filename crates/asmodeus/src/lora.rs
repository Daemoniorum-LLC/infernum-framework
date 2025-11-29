//! LoRA implementation.

use crate::config::LoraConfig;

/// A LoRA adapter layer.
pub struct LoraLayer {
    /// Configuration.
    pub config: LoraConfig,
    /// Layer name.
    pub name: String,
}

impl LoraLayer {
    /// Creates a new LoRA layer.
    #[must_use]
    pub fn new(name: impl Into<String>, config: LoraConfig) -> Self {
        Self {
            config,
            name: name.into(),
        }
    }

    /// Returns the number of trainable parameters.
    #[must_use]
    pub fn num_parameters(&self, in_features: u32, out_features: u32) -> u64 {
        // LoRA adds two matrices: A (in_features x r) and B (r x out_features)
        let r = self.config.r as u64;
        (in_features as u64 * r) + (r * out_features as u64)
    }
}

/// Collection of LoRA adapters for a model.
pub struct LoraModel {
    /// Adapter layers.
    pub layers: Vec<LoraLayer>,
    /// Base model identifier.
    pub base_model: String,
}

impl LoraModel {
    /// Creates a new LoRA model.
    #[must_use]
    pub fn new(base_model: impl Into<String>) -> Self {
        Self {
            layers: Vec::new(),
            base_model: base_model.into(),
        }
    }

    /// Adds a LoRA layer.
    pub fn add_layer(&mut self, layer: LoraLayer) {
        self.layers.push(layer);
    }

    /// Returns total number of trainable parameters.
    #[must_use]
    pub fn total_parameters(&self) -> u64 {
        // Placeholder - would need actual layer dimensions
        self.layers.len() as u64 * 1_000_000
    }
}
