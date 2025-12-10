//! Qwen2 model architecture wrapper using Candle's native implementation.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;

/// Qwen2 model wrapper providing a unified interface.
pub struct Qwen2 {
    model: qwen2::ModelForCausalLM,
    device: Device,
    dtype: DType,
}

impl Qwen2 {
    /// Loads a Qwen2 model from weights.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails.
    pub fn load(config: qwen2::Config, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let model = qwen2::ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            device,
            dtype,
        })
    }

    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs tensor of shape [batch_size, seq_len]
    /// * `start_pos` - Starting position in the sequence (for KV cache)
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        // ModelForCausalLM's forward signature: forward(input_ids, seqlen_offset)
        // It applies the lm_head internally and returns logits
        self.model.forward(input_ids, start_pos)
    }

    /// Clears the KV cache (if applicable).
    pub fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    /// Returns the device the model is on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the dtype of the model.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Qwen2 configuration re-export for convenience.
pub use qwen2::Config as Qwen2Config;
