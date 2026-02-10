//! Speculative Decoding for Large Model Inference
//!
//! Uses a small draft model (e.g., Llama 3.2 1B/8B) to generate candidate tokens,
//! which are then verified by a large target model in a single forward pass.
//!
//! ## Performance Expectations
//!
//! With a well-matched draft model (80%+ acceptance rate):
//! - 3-4 tokens accepted per verification round
//! - Effective 3-4x speedup in tokens/second
//!
//! ## Usage
//!
//! ```rust,ignore
//! let spec = Speculative405B::new(
//!     draft_model,      // Small model (1B-8B) loaded in VRAM
//!     target_model,     // Large model (LazyLlama or LazyQwen2) with layer streaming
//!     config,
//! );
//!
//! let tokens = spec.generate(&prompt_tokens, max_tokens)?;
//! ```

use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use parking_lot::Mutex;

/// Configuration for 405B speculative decoding.
#[derive(Debug, Clone)]
pub struct Speculative405BConfig {
    /// Number of draft tokens per speculation round.
    pub num_draft_tokens: usize,
    /// Acceptance threshold (0.0-1.0). Higher = stricter verification.
    pub acceptance_threshold: f32,
    /// Temperature for draft sampling.
    pub draft_temperature: f32,
    /// Temperature for target verification.
    pub target_temperature: f32,
    /// Whether to use greedy decoding for drafts.
    pub greedy_draft: bool,
}

impl Default for Speculative405BConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 5,
            acceptance_threshold: 0.1, // Low threshold for 405B (high quality)
            draft_temperature: 0.7,
            target_temperature: 0.7,
            greedy_draft: true, // Greedy is faster and works well with matched models
        }
    }
}

impl Speculative405BConfig {
    /// Configuration optimized for maximum throughput.
    pub fn fast() -> Self {
        Self {
            num_draft_tokens: 8,        // More aggressive speculation
            acceptance_threshold: 0.05, // Lower threshold
            greedy_draft: true,
            ..Default::default()
        }
    }

    /// Configuration optimized for quality (higher acceptance rate).
    pub fn quality() -> Self {
        Self {
            num_draft_tokens: 4,       // Fewer drafts
            acceptance_threshold: 0.2, // Higher threshold
            greedy_draft: true,
            ..Default::default()
        }
    }
}

/// Statistics from speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct Speculative405BStats {
    /// Total speculation rounds.
    pub rounds: u64,
    /// Total draft tokens generated.
    pub draft_tokens: u64,
    /// Tokens accepted from draft.
    pub accepted_tokens: u64,
    /// Tokens rejected (resampled from 405B).
    pub rejected_tokens: u64,
    /// Total 405B forward passes (verification).
    pub target_forward_passes: u64,
    /// Total draft forward passes.
    pub draft_forward_passes: u64,
    /// Time spent in draft generation (ms).
    pub draft_time_ms: u64,
    /// Time spent in 405B verification (ms).
    pub verify_time_ms: u64,
}

impl Speculative405BStats {
    /// Acceptance rate (0.0-1.0).
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.draft_tokens as f32
        }
    }

    /// Average tokens accepted per round.
    pub fn tokens_per_round(&self) -> f32 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.accepted_tokens + self.rounds) as f32 / self.rounds as f32
        }
    }

    /// Effective speedup vs standard decoding.
    /// This estimates how many 405B forward passes were saved.
    pub fn speedup(&self) -> f32 {
        if self.target_forward_passes == 0 {
            1.0
        } else {
            // Standard would need 1 forward per token
            // We used target_forward_passes for (accepted + rejected + rounds) tokens
            let total_tokens = self.accepted_tokens + self.rounds;
            total_tokens as f32 / self.target_forward_passes as f32
        }
    }
}

/// Trait for draft models compatible with speculative decoding.
///
/// Note: `Sync` is not required since draft models are wrapped in `Mutex`.
pub trait DraftModel: Send {
    /// Forward pass returning logits.
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor>;

    /// Clear KV cache.
    fn clear_cache(&mut self);

    /// Get the device.
    fn device(&self) -> &Device;

    /// Get the dtype.
    fn dtype(&self) -> DType;
}

/// Trait for target models compatible with speculative decoding.
///
/// This is implemented by lazy-loading models like `LazyLlama` and `LazyQwen2`.
/// Note: `Sync` is not required since target models are wrapped in `Mutex`.
pub trait TargetModel: Send {
    /// Forward pass returning logits.
    ///
    /// May return an error if layer loading fails.
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor>;

    /// Clear KV cache.
    fn clear_cache(&mut self);
}

/// Speculative decoder for large model inference.
pub struct Speculative405B<D: DraftModel, T: TargetModel = crate::models::lazy_llama::LazyLlama> {
    /// Small draft model (fits entirely in VRAM).
    draft: Arc<Mutex<D>>,
    /// Large target model (layer streaming).
    target: Arc<Mutex<T>>,
    /// Configuration.
    config: Speculative405BConfig,
    /// Device for tensor operations.
    device: Device,
    /// Data type.
    #[allow(dead_code)]
    dtype: DType,
    /// Accumulated statistics.
    stats: Mutex<Speculative405BStats>,
}

impl<D: DraftModel, T: TargetModel> Speculative405B<D, T> {
    /// Creates a new speculative decoder.
    pub fn new(draft: D, target: T, config: Speculative405BConfig) -> Self {
        let device = draft.device().clone();
        let dtype = draft.dtype();

        Self {
            draft: Arc::new(Mutex::new(draft)),
            target: Arc::new(Mutex::new(target)),
            config,
            device,
            dtype,
            stats: Mutex::new(Speculative405BStats::default()),
        }
    }

    /// Returns current statistics.
    pub fn stats(&self) -> Speculative405BStats {
        self.stats.lock().clone()
    }

    /// Resets statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = Speculative405BStats::default();
    }

    /// Generates tokens using speculative decoding.
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` - Initial prompt token IDs
    /// * `max_tokens` - Maximum tokens to generate
    /// * `eos_token` - End of sequence token ID
    ///
    /// # Returns
    ///
    /// Generated token IDs.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>> {
        let mut generated = Vec::with_capacity(max_tokens);
        let num_draft = self.config.num_draft_tokens;

        // Clear caches
        {
            self.draft.lock().clear_cache();
            self.target.lock().clear_cache();
        }

        // Prefill both models with prompt
        println!("  Prefilling draft model...");
        let prefill_start = Instant::now();
        {
            let input = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
            let mut draft = self.draft.lock();
            let _ = draft.forward(&input, 0)?;
        }
        println!("  Draft prefill: {:?}", prefill_start.elapsed());

        println!("  Prefilling 405B model...");
        let prefill_start = Instant::now();
        {
            let input = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
            let mut target = self.target.lock();
            let _ = target.forward(&input, 0)?;
        }
        println!("  405B prefill: {:?}", prefill_start.elapsed());

        let mut current_pos = prompt_tokens.len();

        // Main generation loop
        while generated.len() < max_tokens {
            self.stats.lock().rounds += 1;

            // Step 1: Generate draft tokens
            let draft_start = Instant::now();
            let draft_tokens =
                self.generate_draft_tokens(prompt_tokens, &generated, num_draft, eos_token)?;
            let draft_elapsed = draft_start.elapsed().as_millis() as u64;

            {
                let mut stats = self.stats.lock();
                stats.draft_time_ms += draft_elapsed;
                stats.draft_tokens += draft_tokens.len() as u64;
                stats.draft_forward_passes += draft_tokens.len() as u64;
            }

            if draft_tokens.is_empty() {
                break;
            }

            // Step 2: Verify with 405B in single forward pass
            let verify_start = Instant::now();
            let (accepted, next_token) =
                self.verify_draft_tokens(prompt_tokens, &generated, &draft_tokens, current_pos)?;
            let verify_elapsed = verify_start.elapsed().as_millis() as u64;

            {
                let mut stats = self.stats.lock();
                stats.verify_time_ms += verify_elapsed;
                stats.target_forward_passes += 1;
                stats.accepted_tokens += accepted as u64;
                stats.rejected_tokens += (draft_tokens.len() - accepted) as u64;
            }

            // Add accepted tokens
            for token in &draft_tokens[..accepted] {
                if *token == eos_token {
                    return Ok(generated);
                }
                generated.push(*token);
            }

            // Add the resampled/next token
            if let Some(token) = next_token {
                if token == eos_token {
                    return Ok(generated);
                }
                generated.push(token);
            }

            current_pos = prompt_tokens.len() + generated.len();

            // Progress update
            if self.stats.lock().rounds % 10 == 0 {
                let stats = self.stats.lock();
                println!(
                    "  [Round {}] {} tokens, {:.1}% accepted, {:.1} tok/round",
                    stats.rounds,
                    generated.len(),
                    stats.acceptance_rate() * 100.0,
                    stats.tokens_per_round()
                );
            }
        }

        Ok(generated)
    }

    /// Generate draft tokens using the small model.
    fn generate_draft_tokens(
        &self,
        prompt_tokens: &[u32],
        generated: &[u32],
        num_tokens: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>> {
        let mut draft = self.draft.lock();
        let mut draft_tokens = Vec::with_capacity(num_tokens);

        // Build context
        let context: Vec<u32> = prompt_tokens
            .iter()
            .chain(generated.iter())
            .copied()
            .collect();

        let base_pos = context.len();

        for i in 0..num_tokens {
            // Determine input token
            let input_token = if i == 0 {
                *context.last().unwrap_or(&0)
            } else {
                draft_tokens[i - 1]
            };

            // Forward pass
            let input = Tensor::new(&[input_token], &self.device)?.unsqueeze(0)?;
            let logits = draft.forward(&input, base_pos + i - 1)?;

            // Get last token logits
            let last_logits = logits.i((0, 0, ..))?.to_dtype(DType::F32)?;

            // Sample (greedy for speed)
            let next_token = if self.config.greedy_draft {
                last_logits.argmax(0)?.to_scalar::<u32>()?
            } else {
                // Temperature sampling
                let scaled = (&last_logits / self.config.draft_temperature as f64)?;
                let probs = candle_nn::ops::softmax(&scaled, 0)?;
                sample_from_probs(&probs)?
            };

            if next_token == eos_token {
                draft_tokens.push(next_token);
                break;
            }

            draft_tokens.push(next_token);
        }

        Ok(draft_tokens)
    }

    /// Verify draft tokens with 405B in single forward pass.
    fn verify_draft_tokens(
        &self,
        prompt_tokens: &[u32],
        generated: &[u32],
        draft_tokens: &[u32],
        base_pos: usize,
    ) -> anyhow::Result<(usize, Option<u32>)> {
        if draft_tokens.is_empty() {
            return Ok((0, None));
        }

        let mut target = self.target.lock();

        // Build input: last context token + all draft tokens
        let last_context = if generated.is_empty() {
            *prompt_tokens.last().unwrap_or(&0)
        } else {
            *generated.last().unwrap_or(&0)
        };

        let input_tokens: Vec<u32> = std::iter::once(last_context)
            .chain(draft_tokens.iter().copied())
            .collect();

        let input = Tensor::new(input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Single forward pass to verify all positions
        let logits = target.forward(&input, base_pos - 1)?;

        // Verify each draft token
        let threshold = self.config.acceptance_threshold;
        let mut accepted = 0;

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let pos_logits = logits.i((0, i, ..))?.to_dtype(DType::F32)?;

            if self.config.greedy_draft {
                // Greedy verification: accept if argmax matches
                let target_token = pos_logits.argmax(0)?.to_scalar::<u32>()?;

                if target_token == draft_token {
                    accepted += 1;
                } else {
                    // Reject - return target's choice
                    return Ok((accepted, Some(target_token)));
                }
            } else {
                // Probability-based verification
                let probs = candle_nn::ops::softmax(&pos_logits, 0)?;
                let probs_vec: Vec<f32> = probs.to_vec1()?;
                let draft_prob = probs_vec.get(draft_token as usize).copied().unwrap_or(0.0);
                let max_prob = probs_vec.iter().fold(0.0_f32, |a, &b| a.max(b));

                if draft_prob >= threshold * max_prob {
                    accepted += 1;
                } else {
                    // Reject - sample from target distribution
                    let next_token = sample_from_probs(&probs)?;
                    return Ok((accepted, Some(next_token)));
                }
            }
        }

        // All accepted - get next token from last position
        let next_pos_logits = logits
            .i((0, draft_tokens.len(), ..))?
            .to_dtype(DType::F32)?;
        let next_token = if self.config.greedy_draft {
            next_pos_logits.argmax(0)?.to_scalar::<u32>()?
        } else {
            let probs = candle_nn::ops::softmax(&next_pos_logits, 0)?;
            sample_from_probs(&probs)?
        };

        Ok((accepted, Some(next_token)))
    }
}

/// Sample a token from probability distribution.
fn sample_from_probs(probs: &Tensor) -> anyhow::Result<u32> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let r: f32 = fastrand::f32();
    let mut cumsum = 0.0;

    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return Ok(i as u32);
        }
    }

    // Fallback to last token
    Ok((probs_vec.len() - 1) as u32)
}

// ============================================================================
// DraftModel implementations for small models (fit entirely in VRAM)
// ============================================================================

impl DraftModel for crate::models::qwen2::Qwen2 {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        crate::models::qwen2::Qwen2::forward(self, input_ids, pos)
    }

    fn clear_cache(&mut self) {
        crate::models::qwen2::Qwen2::clear_cache(self);
    }

    fn device(&self) -> &Device {
        crate::models::qwen2::Qwen2::device(self)
    }

    fn dtype(&self) -> DType {
        crate::models::qwen2::Qwen2::dtype(self)
    }
}

impl DraftModel for crate::models::llama::Llama {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        crate::models::llama::Llama::forward(self, input_ids, pos)
    }

    fn clear_cache(&mut self) {
        crate::models::llama::Llama::clear_cache(self);
    }

    fn device(&self) -> &Device {
        crate::models::llama::Llama::device(self)
    }

    fn dtype(&self) -> DType {
        crate::models::llama::Llama::dtype(self)
    }
}

// ============================================================================
// TargetModel implementations for lazy-loading models
// ============================================================================

impl TargetModel for crate::models::lazy_llama::LazyLlama {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        crate::models::lazy_llama::LazyLlama::forward(self, input_ids, pos)
            .map_err(|e| anyhow::anyhow!("LazyLlama forward error: {}", e))
    }

    fn clear_cache(&mut self) {
        crate::models::lazy_llama::LazyLlama::clear_cache(self);
    }
}

impl TargetModel for crate::models::lazy_qwen2::LazyQwen2 {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> anyhow::Result<Tensor> {
        crate::models::lazy_qwen2::LazyQwen2::forward(self, input_ids, pos)
            .map_err(|e| anyhow::anyhow!("LazyQwen2 forward error: {}", e))
    }

    fn clear_cache(&mut self) {
        crate::models::lazy_qwen2::LazyQwen2::clear_cache(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Speculative405BConfig::default();
        assert_eq!(config.num_draft_tokens, 5);
        assert!(config.greedy_draft);
    }

    #[test]
    fn test_config_fast() {
        let config = Speculative405BConfig::fast();
        assert_eq!(config.num_draft_tokens, 8);
        assert!(config.acceptance_threshold < 0.1);
    }

    #[test]
    fn test_stats_acceptance_rate() {
        let mut stats = Speculative405BStats::default();
        stats.draft_tokens = 100;
        stats.accepted_tokens = 80;
        assert!((stats.acceptance_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_stats_speedup() {
        let mut stats = Speculative405BStats::default();
        stats.rounds = 25;
        stats.accepted_tokens = 75;
        stats.target_forward_passes = 25;
        // 100 tokens (75 + 25) with 25 forward passes = 4x speedup
        assert!((stats.speedup() - 4.0).abs() < 0.001);
    }
}
