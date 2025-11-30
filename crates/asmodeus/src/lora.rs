//! LoRA (Low-Rank Adaptation) implementation.
//!
//! LoRA freezes the pretrained model weights and injects trainable rank
//! decomposition matrices into each layer of the Transformer architecture.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor, Result as CandleResult, safetensors as candle_safetensors};
use serde::{Deserialize, Serialize};

use crate::config::LoraConfig;

/// A LoRA adapter layer that modifies a linear layer.
///
/// For a pretrained weight matrix W, LoRA represents the update as:
/// W' = W + BA where B is (out_features x r) and A is (r x in_features)
#[derive(Debug)]
pub struct LoraLayer {
    /// Configuration.
    pub config: LoraConfig,
    /// Layer name/path in the model.
    pub name: String,
    /// Input dimension.
    pub in_features: usize,
    /// Output dimension.
    pub out_features: usize,
    /// Low-rank matrix A (r x in_features).
    pub lora_a: Option<Tensor>,
    /// Low-rank matrix B (out_features x r).
    pub lora_b: Option<Tensor>,
    /// Scaling factor: alpha / r.
    scaling: f32,
}

impl LoraLayer {
    /// Creates a new LoRA layer.
    pub fn new(
        name: impl Into<String>,
        config: LoraConfig,
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let r = config.r as usize;
        let scaling = config.alpha / config.r as f32;

        // Initialize A with random normal (scaled), B with zeros
        // This ensures the LoRA update is initially zero (BA = 0)
        let lora_a = Tensor::randn(0.0f32, 1.0f32, (r, in_features), device)?;
        let lora_a = (lora_a * (1.0 / (r as f64).sqrt()))?;

        let lora_b = Tensor::zeros((out_features, r), DType::F32, device)?;

        Ok(Self {
            config,
            name: name.into(),
            in_features,
            out_features,
            lora_a: Some(lora_a),
            lora_b: Some(lora_b),
            scaling,
        })
    }

    /// Creates a LoRA layer without initializing tensors (for loading).
    pub fn empty(name: impl Into<String>, config: LoraConfig) -> Self {
        let scaling = config.alpha / config.r as f32;
        Self {
            config,
            name: name.into(),
            in_features: 0,
            out_features: 0,
            lora_a: None,
            lora_b: None,
            scaling,
        }
    }

    /// Returns the number of trainable parameters.
    pub fn num_parameters(&self) -> u64 {
        let r = self.config.r as u64;
        let in_f = self.in_features as u64;
        let out_f = self.out_features as u64;
        (in_f * r) + (r * out_f)
    }

    /// Applies the LoRA adaptation to an input tensor.
    ///
    /// output = input @ W + input @ A^T @ B^T * scaling
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (lora_a, lora_b) = match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(candle_core::Error::Msg("LoRA tensors not initialized".into())),
        };

        // x @ A^T gives (batch, r)
        let hidden = x.matmul(&lora_a.t()?)?;

        // Apply dropout during training (TODO: add training mode flag)
        // hidden = dropout(hidden, self.config.dropout)

        // hidden @ B^T gives (batch, out_features)
        let lora_out = hidden.matmul(&lora_b.t()?)?;

        // Scale the output
        lora_out.affine(self.scaling as f64, 0.0)
    }

    /// Merges the LoRA weights into the original weight matrix.
    pub fn merge(&self, original_weight: &Tensor) -> CandleResult<Tensor> {
        let delta = self.get_delta()?;
        original_weight.add(&delta)
    }

    /// Computes the weight delta: B @ A * scaling.
    pub fn get_delta(&self) -> CandleResult<Tensor> {
        let (lora_a, lora_b) = match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(candle_core::Error::Msg("LoRA tensors not initialized".into())),
        };

        // B @ A gives (out_features x in_features)
        let delta = lora_b.matmul(lora_a)?;
        delta.affine(self.scaling as f64, 0.0)
    }
}

/// Collection of LoRA adapters for a model.
#[derive(Debug)]
pub struct LoraModel {
    /// Adapter layers by name.
    pub layers: HashMap<String, LoraLayer>,
    /// Base model identifier.
    pub base_model: String,
    /// Global LoRA configuration.
    pub config: LoraConfig,
}

impl LoraModel {
    /// Creates a new LoRA model.
    pub fn new(base_model: impl Into<String>, config: LoraConfig) -> Self {
        Self {
            layers: HashMap::new(),
            base_model: base_model.into(),
            config,
        }
    }

    /// Adds a LoRA layer for a specific module.
    pub fn add_layer(&mut self, layer: LoraLayer) {
        self.layers.insert(layer.name.clone(), layer);
    }

    /// Gets a LoRA layer by name.
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.get(name)
    }

    /// Returns total number of trainable parameters.
    pub fn total_parameters(&self) -> u64 {
        self.layers.values().map(|l| l.num_parameters()).sum()
    }

    /// Saves the LoRA adapters to a directory.
    pub fn save(&self, path: &PathBuf) -> infernum_core::Result<()> {
        std::fs::create_dir_all(path)?;

        // Save config
        let config_path = path.join("lora_config.json");
        let config_json = serde_json::to_string_pretty(&LoraModelConfig {
            base_model: self.base_model.clone(),
            config: self.config.clone(),
            layers: self.layers.keys().cloned().collect(),
        })?;
        std::fs::write(config_path, config_json)?;

        // Save tensors
        let tensors_path = path.join("lora_weights.safetensors");
        let mut tensors = HashMap::new();

        for (name, layer) in &self.layers {
            if let (Some(a), Some(b)) = (&layer.lora_a, &layer.lora_b) {
                tensors.insert(format!("{}.lora_a", name), a.clone());
                tensors.insert(format!("{}.lora_b", name), b.clone());
            }
        }

        if !tensors.is_empty() {
            candle_safetensors::save(&tensors, &tensors_path)
                .map_err(|e| infernum_core::Error::Internal { message: e.to_string() })?;
        }

        tracing::info!(path = ?path, parameters = self.total_parameters(), "Saved LoRA model");
        Ok(())
    }

    /// Loads LoRA adapters from a directory.
    pub fn load(path: &PathBuf, device: &Device) -> infernum_core::Result<Self> {
        // Load config
        let config_path = path.join("lora_config.json");
        let config_json = std::fs::read_to_string(config_path)?;
        let model_config: LoraModelConfig = serde_json::from_str(&config_json)?;

        // Load tensors
        let tensors_path = path.join("lora_weights.safetensors");
        let tensors = if tensors_path.exists() {
            candle_safetensors::load(&tensors_path, device)
                .map_err(|e| infernum_core::Error::Internal { message: e.to_string() })?
        } else {
            HashMap::new()
        };

        let mut model = Self::new(model_config.base_model, model_config.config.clone());

        for layer_name in model_config.layers {
            let lora_a = tensors.get(&format!("{}.lora_a", layer_name)).cloned();
            let lora_b = tensors.get(&format!("{}.lora_b", layer_name)).cloned();

            let in_features = lora_a.as_ref().map(|a| a.dim(1).unwrap_or(0)).unwrap_or(0);
            let out_features = lora_b.as_ref().map(|b| b.dim(0).unwrap_or(0)).unwrap_or(0);

            let mut layer = LoraLayer::empty(layer_name, model_config.config.clone());
            layer.lora_a = lora_a;
            layer.lora_b = lora_b;
            layer.in_features = in_features;
            layer.out_features = out_features;

            model.add_layer(layer);
        }

        tracing::info!(path = ?path, parameters = model.total_parameters(), "Loaded LoRA model");
        Ok(model)
    }
}

/// Serializable LoRA model configuration.
#[derive(Debug, Serialize, Deserialize)]
struct LoraModelConfig {
    base_model: String,
    config: LoraConfig,
    layers: Vec<String>,
}

/// Identifies target modules in a model for LoRA adaptation.
pub fn find_target_modules(
    model_type: &str,
    target_modules: &[String],
) -> Vec<String> {
    // Common patterns for different architectures
    let mut modules = Vec::new();

    for target in target_modules {
        match model_type {
            "llama" | "mistral" => {
                // LLaMA/Mistral architecture
                for i in 0..32 {
                    modules.push(format!("model.layers.{}.self_attn.{}", i, target));
                }
            }
            "gpt2" | "gpt-neo" => {
                // GPT-2 style architecture
                for i in 0..12 {
                    modules.push(format!("h.{}.attn.{}", i, target));
                }
            }
            _ => {
                // Generic: just use the target name
                modules.push(target.clone());
            }
        }
    }

    modules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_parameters() {
        let config = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string()],
        };

        let layer = LoraLayer::new(
            "test",
            config,
            768,  // in_features
            768,  // out_features
            &Device::Cpu,
        ).unwrap();

        // A: 8 x 768 = 6144
        // B: 768 x 8 = 6144
        // Total: 12288
        assert_eq!(layer.num_parameters(), 12288);
    }

    #[test]
    fn test_lora_scaling() {
        let config = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![],
        };

        let layer = LoraLayer::new("test", config, 64, 64, &Device::Cpu).unwrap();

        // scaling = alpha / r = 16 / 8 = 2
        assert!((layer.scaling - 2.0).abs() < 1e-6);
    }
}
