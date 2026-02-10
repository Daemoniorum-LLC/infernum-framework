//! Speculative decoding for accelerated inference.
//!
//! Speculative decoding uses a smaller "draft" model to generate candidate tokens,
//! which are then verified by the main model. This can significantly speed up
//! inference for models with similar vocabularies.
//!
//! ## Algorithm
//!
//! 1. Draft model generates K speculative tokens
//! 2. Main model evaluates all K+1 positions in one forward pass
//! 3. For each position, compare draft and main model probabilities
//! 4. Accept tokens where draft matches main model (with some tolerance)
//! 5. Resample from adjusted distribution if rejection occurs
//!
//! ## Benefits
//!
//! - Speedup of 2-3x for well-matched draft models
//! - Same output distribution as standard decoding (when using rejection sampling)
//! - Particularly effective for greedy/low-temperature sampling

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use parking_lot::Mutex;

use crate::config::SpeculativeConfig;
use crate::models::ModelKind;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use infernum_core::{Result, SamplingParams};

/// Statistics from speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated.
    pub total_tokens: u64,
    /// Tokens accepted from draft model.
    pub accepted_tokens: u64,
    /// Tokens rejected and resampled.
    pub rejected_tokens: u64,
    /// Number of speculation rounds.
    pub rounds: u64,
}

impl SpeculativeStats {
    /// Returns the acceptance rate (0.0 - 1.0).
    #[must_use]
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.total_tokens as f32
        }
    }

    /// Returns average tokens accepted per round.
    #[must_use]
    pub fn avg_tokens_per_round(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.rounds as f32
        }
    }
}

/// Speculative decoder for accelerated token generation.
pub struct SpeculativeDecoder {
    /// Draft model (smaller, faster).
    draft_model: Arc<Mutex<ModelKind>>,
    /// Draft model tokenizer.
    draft_tokenizer: Arc<Tokenizer>,
    /// Configuration.
    config: SpeculativeConfig,
    /// Computation device.
    device: Device,
    /// Data type for computations.
    dtype: DType,
    /// Accumulated statistics.
    stats: Mutex<SpeculativeStats>,
}

impl SpeculativeDecoder {
    /// Creates a new speculative decoder with the given draft model.
    #[must_use]
    pub fn new(
        draft_model: ModelKind,
        draft_tokenizer: Tokenizer,
        config: SpeculativeConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        Self {
            draft_model: Arc::new(Mutex::new(draft_model)),
            draft_tokenizer: Arc::new(draft_tokenizer),
            config,
            device,
            dtype,
            stats: Mutex::new(SpeculativeStats::default()),
        }
    }

    /// Returns the speculative decoding configuration.
    #[must_use]
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Returns the computation device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the data type used for computations.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns current statistics.
    #[must_use]
    pub fn stats(&self) -> SpeculativeStats {
        self.stats.lock().clone()
    }

    /// Resets statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = SpeculativeStats::default();
    }

    /// Generates tokens using speculative decoding.
    ///
    /// # Arguments
    ///
    /// * `main_model` - The main model for verification
    /// * `prompt_tokens` - Initial prompt token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `sampling_params` - Sampling parameters
    /// * `eos_token` - End of sequence token ID
    ///
    /// # Returns
    ///
    /// A tuple of (generated_tokens, generated_text).
    pub fn generate(
        &self,
        main_model: &mut ModelKind,
        prompt_tokens: &[u32],
        max_tokens: u32,
        sampling_params: &SamplingParams,
        eos_token: u32,
    ) -> Result<(Vec<u32>, Vec<String>)> {
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let mut current_pos = prompt_tokens.len();

        // Create sampler for verification
        let mut sampler = Sampler::new(sampling_params.clone());

        // Clear KV caches
        main_model.clear_cache();
        self.draft_model.lock().clear_cache();

        // Prefill both models with prompt
        let input_ids = Tensor::new(prompt_tokens, &self.device)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        // Prefill main model
        let _ = main_model
            .forward(&input_ids, 0)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        // Prefill draft model
        {
            let mut draft = self.draft_model.lock();
            let _ = draft
                .forward(&input_ids, 0)
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;
        }

        // Main generation loop
        let num_spec_tokens = self.config.num_speculative_tokens as usize;

        while generated_tokens.len() < max_tokens as usize {
            self.stats.lock().rounds += 1;

            // Step 1: Generate draft tokens
            let draft_tokens = self.generate_draft_tokens(
                prompt_tokens,
                &generated_tokens,
                num_spec_tokens,
                sampling_params,
                eos_token,
            )?;

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify with main model
            let (accepted, next_token) = self.verify_draft_tokens(
                main_model,
                prompt_tokens,
                &generated_tokens,
                &draft_tokens,
                &mut sampler,
                current_pos,
            )?;

            // Update statistics
            {
                let mut stats = self.stats.lock();
                stats.total_tokens += draft_tokens.len() as u64;
                stats.accepted_tokens += accepted as u64;
                stats.rejected_tokens += (draft_tokens.len() - accepted) as u64;
            }

            // Add accepted tokens
            for token in &draft_tokens[..accepted] {
                if *token == eos_token {
                    return Ok((generated_tokens, generated_text));
                }
                generated_tokens.push(*token);
                let token_text = self.draft_tokenizer.decode_token(*token)?;
                generated_text.push(token_text);
            }

            // Add the next token from main model
            if let Some(token) = next_token {
                if token == eos_token {
                    return Ok((generated_tokens, generated_text));
                }
                generated_tokens.push(token);
                let token_text = self.draft_tokenizer.decode_token(token)?;
                generated_text.push(token_text);
            }

            current_pos = prompt_tokens.len() + generated_tokens.len();

            // Check stop sequences
            let full_text: String = generated_text.join("");
            if sampler.is_stop_token(&full_text) {
                break;
            }
        }

        Ok((generated_tokens, generated_text))
    }

    /// Generates draft tokens using the draft model.
    fn generate_draft_tokens(
        &self,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
        num_tokens: usize,
        sampling_params: &SamplingParams,
        eos_token: u32,
    ) -> Result<Vec<u32>> {
        // Create a new sampler for draft generation
        let mut sampler = Sampler::new(sampling_params.clone());
        let mut draft = self.draft_model.lock();
        let mut draft_tokens = Vec::with_capacity(num_tokens);

        // Build full context
        let mut context: Vec<u32> = prompt_tokens.to_vec();
        context.extend_from_slice(generated_tokens);

        let base_pos = context.len();

        for i in 0..num_tokens {
            // Create input for next token
            let input = if i == 0 && generated_tokens.is_empty() {
                // First token after prefill - just need last token
                let last_token = *context.last().unwrap_or(&0);
                Tensor::new(&[last_token], &self.device)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            } else if !draft_tokens.is_empty() {
                // Use last draft token (safe: checked is_empty above)
                let last_token = *draft_tokens.last().expect("checked non-empty");
                Tensor::new(&[last_token], &self.device)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            } else {
                // Use last context token
                let last_token = *context.last().unwrap_or(&0);
                Tensor::new(&[last_token], &self.device)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            };

            // Forward pass
            let logits = draft
                .forward(&input, base_pos + i - 1)
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            let last_logits = logits
                .i((0, 0, ..))
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                .to_dtype(DType::F32)
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            let logits_vec: Vec<f32> = last_logits
                .to_vec1()
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            // Sample next token
            let next_token = sampler.sample(&logits_vec);

            // Stop if EOS
            if next_token == eos_token {
                draft_tokens.push(next_token);
                break;
            }

            draft_tokens.push(next_token);
        }

        Ok(draft_tokens)
    }

    /// Verifies draft tokens with the main model.
    ///
    /// Returns (num_accepted, optional_next_token).
    fn verify_draft_tokens(
        &self,
        main_model: &mut ModelKind,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
        draft_tokens: &[u32],
        sampler: &mut Sampler,
        base_pos: usize,
    ) -> Result<(usize, Option<u32>)> {
        if draft_tokens.is_empty() {
            return Ok((0, None));
        }

        // Create input tensor with all draft tokens for parallel verification
        let input_tokens: Vec<u32> = if generated_tokens.is_empty() {
            // First iteration - include last prompt token + draft tokens
            let last_prompt = *prompt_tokens.last().unwrap_or(&0);
            std::iter::once(last_prompt)
                .chain(draft_tokens.iter().copied())
                .collect()
        } else {
            // Subsequent iterations - include last generated + draft tokens
            let last_gen = *generated_tokens.last().unwrap_or(&0);
            std::iter::once(last_gen)
                .chain(draft_tokens.iter().copied())
                .collect()
        };

        let input_ids = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        // Forward pass to get logits for all positions
        let logits = main_model
            .forward(&input_ids, base_pos - 1)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        // Verify each draft token
        let mut accepted = 0;
        let threshold = self.config.acceptance_threshold;

        for (i, draft_token) in draft_tokens.iter().enumerate() {
            // Get logits for position i (which predicts token i)
            let pos_logits = logits
                .i((0, i, ..))
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                .to_dtype(DType::F32)
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            let logits_vec: Vec<f32> = pos_logits
                .to_vec1()
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            // Compute softmax probabilities
            let probs = softmax(&logits_vec);

            // Get main model's probability for the draft token
            let main_prob = probs.get(*draft_token as usize).copied().unwrap_or(0.0);

            // Simple acceptance: accept if probability is above threshold
            // For more accurate rejection sampling, compare with draft model's probability
            if main_prob >= threshold * get_max_prob(&probs) {
                accepted += 1;
            } else {
                // Reject - sample from main model's distribution for this position
                let next_token = sampler.sample(&logits_vec);
                return Ok((accepted, Some(next_token)));
            }
        }

        // All tokens accepted - sample next token from the last position
        let last_pos_logits = logits
            .i((0, draft_tokens.len(), ..))
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        let logits_vec: Vec<f32> = last_pos_logits
            .to_vec1()
            .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

        let next_token = sampler.sample(&logits_vec);
        Ok((accepted, Some(next_token)))
    }
}

/// Computes softmax of a logits vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect()
}

/// Gets the maximum probability from a probability distribution.
fn get_max_prob(probs: &[f32]) -> f32 {
    probs.iter().fold(0.0_f32, |a, &b| a.max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.rejected_tokens, 0);
        assert_eq!(stats.rounds, 0);
    }

    #[test]
    fn test_speculative_stats_acceptance_rate() {
        let mut stats = SpeculativeStats::default();
        stats.total_tokens = 100;
        stats.accepted_tokens = 75;
        assert!((stats.acceptance_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_acceptance_rate_zero() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_stats_avg_tokens_per_round() {
        let mut stats = SpeculativeStats::default();
        stats.accepted_tokens = 50;
        stats.rounds = 10;
        assert!((stats.avg_tokens_per_round() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_speculative_stats_avg_tokens_zero_rounds() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.avg_tokens_per_round(), 0.0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be approximately 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Probabilities should be in ascending order
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_softmax_all_same() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        // All probabilities should be equal
        for prob in &probs {
            assert!((*prob - 1.0 / 3.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_get_max_prob() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        assert!((get_max_prob(&probs) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_get_max_prob_empty() {
        let probs: Vec<f32> = vec![];
        assert_eq!(get_max_prob(&probs), 0.0);
    }
}
