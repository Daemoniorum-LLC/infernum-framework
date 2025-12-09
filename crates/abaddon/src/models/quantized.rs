//! Quantized model support using GGUF files.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_llama;
use std::path::Path;

/// Quantized Llama model wrapper.
pub struct QuantizedLlama {
    model: quantized_llama::ModelWeights,
    device: Device,
    tokenizer: tokenizers::Tokenizer,
}

impl QuantizedLlama {
    /// Loads a quantized Llama model from a GGUF file.
    pub fn from_gguf(
        path: impl AsRef<Path>,
        tokenizer: tokenizers::Tokenizer,
        device: &Device,
    ) -> CandleResult<Self> {
        let path = path.as_ref();
        tracing::info!(?path, "Loading quantized GGUF model");

        let mut file = std::fs::File::open(path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        let model = quantized_llama::ModelWeights::from_gguf(content, &mut file, device)?;

        Ok(Self {
            model,
            device: device.clone(),
            tokenizer,
        })
    }

    /// Generates text from the given prompt.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> CandleResult<String> {
        // Tokenize input
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization error: {}", e)))?;
        let token_ids = tokens.get_ids();

        let mut generated_tokens = Vec::new();
        let mut index_pos = 0;

        // Process prompt tokens
        for &token_id in token_ids {
            let input = Tensor::new(&[token_id], &self.device)?;
            let _logits = self.model.forward(&input, index_pos)?;
            index_pos += 1;
        }

        // Generate new tokens
        let mut next_token = *token_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?;
            let logits = self.model.forward(&input, index_pos)?;

            // Sample from logits
            let logits = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
            let logits = if temperature > 0.0 {
                (&logits / temperature)?
            } else {
                logits
            };

            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            let next_token_tensor = probs.argmax(candle_core::D::Minus1)?;
            next_token = next_token_tensor.to_scalar::<u32>()?;

            generated_tokens.push(next_token);
            index_pos += 1;

            // Check for EOS token (common EOS tokens)
            if next_token == 2 || next_token == 128001 || next_token == 128009 {
                break;
            }
        }

        // Decode generated tokens
        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| candle_core::Error::Msg(format!("Decoding error: {}", e)))?;

        Ok(text)
    }

    /// Returns a reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
