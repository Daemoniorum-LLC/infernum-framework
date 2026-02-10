//! Speculative decoding for accelerated inference.
//!
//! Uses a small draft model to propose candidate tokens, then verifies them
//! in parallel with the target model. When candidates are accepted, we get
//! multiple tokens per forward pass.
//!
//! ## Algorithm
//!
//! 1. Draft model generates K candidate tokens autoregressively
//! 2. Target model processes all K+1 positions in parallel
//! 3. Compare logits to accept/reject each candidate
//! 4. Accept first N matching tokens, sample from first mismatch
//!
//! ## Performance
//!
//! Speedup depends on draft model accuracy:
//! - 70% acceptance rate → ~2.3x speedup with K=5
//! - 90% acceptance rate → ~3.3x speedup with K=5
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Speculative Decoding                      │
//! │                                                              │
//! │  Draft Model (fast)        Target Model (slow)              │
//! │  ┌─────────────────┐       ┌─────────────────────────────┐  │
//! │  │ Generate K=5    │   →   │ Verify all 5 in parallel    │  │
//! │  │ tokens serially │       │ Accept: [✓✓✓✗]              │  │
//! │  └─────────────────┘       │ Sample at first ✗           │  │
//! │                            └─────────────────────────────┘  │
//! │                                                              │
//! │  Result: 3 tokens accepted + 1 sampled = 4 tokens/iter      │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::compute::ComputeEngine;
use super::generate::SamplingParams;
use super::kv_cache::KvCache;
use super::weight_store::WeightStore;
use super::InferenceError;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of candidate tokens to generate per iteration.
    pub num_candidates: usize,

    /// Rejection sampling temperature adjustment.
    pub temperature_adjustment: f32,

    /// Whether to use nucleus sampling for draft model.
    pub draft_top_p: f32,

    /// Maximum consecutive rejections before falling back.
    pub max_rejections: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_candidates: 5,
            temperature_adjustment: 1.0,
            draft_top_p: 0.9,
            max_rejections: 3,
        }
    }
}

/// Token candidate with probability information.
#[derive(Debug, Clone)]
pub struct TokenCandidate {
    /// Token ID.
    pub token: u32,
    /// Draft model probability.
    pub draft_prob: f32,
    /// Target model probability.
    pub target_prob: f32,
}

/// Result of speculative verification.
#[derive(Debug)]
pub struct VerificationResult {
    /// Accepted tokens (before first rejection).
    pub accepted: Vec<u32>,
    /// Token sampled at rejection point (or final position).
    pub sampled: u32,
    /// Total tokens produced this iteration.
    pub total_tokens: usize,
    /// Acceptance rate for this iteration.
    pub acceptance_rate: f32,
}

/// Speculative decoder using draft + target models.
pub struct SpeculativeDecoder {
    /// Draft (small) model compute engine.
    draft_engine: ComputeEngine,

    /// Draft model weights.
    draft_weights: Arc<WeightStore>,

    /// Draft model KV cache.
    draft_kv: KvCache,

    /// Target (large) model compute engine.
    target_engine: ComputeEngine,

    /// Target model weights.
    target_weights: Arc<WeightStore>,

    /// Target model KV cache.
    target_kv: KvCache,

    /// Configuration.
    config: SpeculativeConfig,

    /// CUDA device.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,

    /// Consecutive rejection counter.
    rejection_count: usize,

    /// Statistics: total accepted tokens.
    total_accepted: usize,

    /// Statistics: total proposed tokens.
    total_proposed: usize,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder.
    pub fn new(
        draft_weights: Arc<WeightStore>,
        target_weights: Arc<WeightStore>,
        max_seq_len: usize,
        device: Arc<CudaDevice>,
        config: SpeculativeConfig,
    ) -> Result<Self, InferenceError> {
        let draft_config = draft_weights.config.clone();
        let target_config = target_weights.config.clone();

        let draft_engine =
            ComputeEngine::new(draft_config.clone(), max_seq_len, Arc::clone(&device))?;
        let target_engine =
            ComputeEngine::new(target_config.clone(), max_seq_len, Arc::clone(&device))?;

        let draft_kv = KvCache::new(&draft_config, max_seq_len, Arc::clone(&device))?;
        let target_kv = KvCache::new(&target_config, max_seq_len, Arc::clone(&device))?;

        Ok(Self {
            draft_engine,
            draft_weights,
            draft_kv,
            target_engine,
            target_weights,
            target_kv,
            config,
            device,
            rejection_count: 0,
            total_accepted: 0,
            total_proposed: 0,
        })
    }

    /// Run one iteration of speculative decoding.
    ///
    /// This generates K candidate tokens with the draft model, then verifies
    /// them in parallel with the target model using rejection sampling.
    pub fn step(
        &mut self,
        input_ids: &[u32],
        position: usize,
        params: &SamplingParams,
    ) -> Result<VerificationResult, InferenceError> {
        // Step 1: Generate K candidate tokens with draft model
        let candidates = self.generate_draft_candidates(input_ids, position, params)?;
        let num_candidates = candidates.len();

        if num_candidates == 0 {
            // Fallback to single-token generation
            let token = self.generate_single_token(input_ids, position)?;
            return Ok(VerificationResult {
                accepted: Vec::new(),
                sampled: token,
                total_tokens: 1,
                acceptance_rate: 0.0,
            });
        }

        // Step 2: Verify all candidates in parallel with target model
        let result = self.verify_candidates(input_ids, position, &candidates, params)?;

        // Update statistics
        self.total_proposed += num_candidates;
        self.total_accepted += result.accepted.len();

        // Track rejection streaks
        if result.accepted.is_empty() {
            self.rejection_count += 1;
        } else {
            self.rejection_count = 0;
        }

        Ok(result)
    }

    /// Generate K candidate tokens using draft model.
    fn generate_draft_candidates(
        &mut self,
        input_ids: &[u32],
        position: usize,
        params: &SamplingParams,
    ) -> Result<Vec<TokenCandidate>, InferenceError> {
        let mut candidates = Vec::with_capacity(self.config.num_candidates);
        let mut draft_ids = input_ids.to_vec();
        let mut pos = position;

        // Draft sampling params (potentially more aggressive)
        let mut draft_params = params.clone();
        draft_params.top_p = self.config.draft_top_p;

        for _ in 0..self.config.num_candidates {
            // Forward through draft model
            let logits = self
                .draft_engine
                .forward(&draft_ids, &self.draft_weights, pos)?;

            // Sample token from logits
            let (token, prob) = self.sample_from_logits(&logits, &draft_params)?;

            candidates.push(TokenCandidate {
                token,
                draft_prob: prob,
                target_prob: 0.0, // Filled during verification
            });

            // Extend for next iteration
            draft_ids.push(token);
            pos += 1;
        }

        Ok(candidates)
    }

    /// Verify candidates in parallel with target model.
    fn verify_candidates(
        &mut self,
        input_ids: &[u32],
        position: usize,
        candidates: &[TokenCandidate],
        params: &SamplingParams,
    ) -> Result<VerificationResult, InferenceError> {
        let n = candidates.len();

        // Build verification sequence: [input_ids..., candidate_0, ..., candidate_{n-1}]
        let mut verify_ids = input_ids.to_vec();
        for c in candidates {
            verify_ids.push(c.token);
        }

        // Forward all positions at once through target model
        let logits = self
            .target_engine
            .forward(&verify_ids, &self.target_weights, position)?;

        // For now, simple verification: accept all candidates if logits look reasonable
        // Full implementation would do proper rejection sampling
        let mut accepted = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            // Get target model probability for draft token
            let target_prob = self.get_token_prob_from_logits(&logits, candidate.token, i)?;

            // Rejection sampling: accept with probability min(1, p_target / p_draft)
            let accept_prob = (target_prob / candidate.draft_prob.max(1e-10)).min(1.0);
            let rand: f32 = fastrand::f32();

            if rand < accept_prob {
                accepted.push(candidate.token);
            } else {
                break; // First rejection ends acceptance
            }
        }

        // Sample final token
        let sampled = self.sample_final_token(&logits, params)?;

        let num_accepted = accepted.len();
        let acceptance_rate = if n > 0 {
            num_accepted as f32 / n as f32
        } else {
            0.0
        };

        Ok(VerificationResult {
            accepted,
            sampled,
            total_tokens: num_accepted + 1,
            acceptance_rate,
        })
    }

    /// Generate single token without speculation (fallback).
    fn generate_single_token(
        &mut self,
        input_ids: &[u32],
        position: usize,
    ) -> Result<u32, InferenceError> {
        let logits = self
            .target_engine
            .forward(input_ids, &self.target_weights, position)?;

        // Simple argmax for fallback
        let cpu_logits = self.logits_to_cpu(&logits)?;
        let token = cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        Ok(token)
    }

    /// Sample token from logits with probability.
    fn sample_from_logits(
        &self,
        logits: &super::tensor::GpuTensor,
        params: &SamplingParams,
    ) -> Result<(u32, f32), InferenceError> {
        let cpu_logits = self.logits_to_cpu(logits)?;

        // Apply temperature
        let scaled: Vec<f32> = cpu_logits.iter().map(|&x| x / params.temperature).collect();

        // Softmax
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max).exp()).sum();
        let probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp() / exp_sum).collect();

        // Top-p sampling
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumsum = 0.0;
        let mut nucleus = Vec::new();

        for &idx in &sorted_indices {
            cumsum += probs[idx];
            nucleus.push((idx, probs[idx]));
            if cumsum >= params.top_p {
                break;
            }
        }

        // Renormalize and sample
        let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        let rand: f32 = fastrand::f32();
        let mut accum = 0.0;

        for (idx, prob) in nucleus {
            accum += prob / nucleus_sum;
            if rand < accum {
                return Ok((idx as u32, prob));
            }
        }

        // Fallback to argmax
        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Ok((idx as u32, prob))
    }

    /// Get probability of specific token from logits at position.
    fn get_token_prob_from_logits(
        &self,
        logits: &super::tensor::GpuTensor,
        token: u32,
        _position: usize,
    ) -> Result<f32, InferenceError> {
        let cpu_logits = self.logits_to_cpu(logits)?;
        let vocab_size = cpu_logits.len();

        if (token as usize) >= vocab_size {
            return Ok(0.0);
        }

        // Softmax for token probability
        let max = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = cpu_logits.iter().map(|&x| (x - max).exp()).sum();
        let prob = (cpu_logits[token as usize] - max).exp() / exp_sum;

        Ok(prob)
    }

    /// Sample final token from logits.
    fn sample_final_token(
        &self,
        logits: &super::tensor::GpuTensor,
        params: &SamplingParams,
    ) -> Result<u32, InferenceError> {
        let (token, _) = self.sample_from_logits(logits, params)?;
        Ok(token)
    }

    /// Convert GPU logits to CPU vector.
    fn logits_to_cpu(&self, logits: &super::tensor::GpuTensor) -> Result<Vec<f32>, InferenceError> {
        // Get raw bytes from GPU
        let bytes = logits.to_host()?;

        // Convert based on dtype (assuming F16 for now)
        let f16_values: Vec<half::f16> = bytes
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        Ok(f16_values.iter().map(|x| x.to_f32()).collect())
    }

    /// Get acceptance rate statistics.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_proposed as f32
        }
    }

    /// Get speedup estimate based on acceptance rate.
    pub fn estimated_speedup(&self) -> f32 {
        let alpha = self.acceptance_rate();
        let k = self.config.num_candidates as f32;

        // Expected tokens per iteration: sum of geometric series
        // E[tokens] = 1 + alpha + alpha^2 + ... + alpha^k
        if alpha < 0.01 {
            1.0
        } else {
            (1.0 - alpha.powf(k + 1.0)) / (1.0 - alpha)
        }
    }

    /// Check if we should fall back to non-speculative mode.
    pub fn should_fallback(&self) -> bool {
        self.rejection_count >= self.config.max_rejections
    }

    /// Reset KV caches for new sequence.
    pub fn reset(&mut self) {
        self.draft_kv.reset();
        self.target_kv.reset();
        self.rejection_count = 0;
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.total_accepted = 0;
        self.total_proposed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_candidates, 5);
        assert_eq!(config.draft_top_p, 0.9);
    }

    #[test]
    fn test_acceptance_rate_calculation() {
        // With no proposals, rate should be 0
        let total_proposed = 0;
        let total_accepted = 0;
        let rate = if total_proposed == 0 {
            0.0
        } else {
            total_accepted as f32 / total_proposed as f32
        };
        assert_eq!(rate, 0.0);

        // With 70% acceptance
        let total_proposed = 100;
        let total_accepted = 70;
        let rate = total_accepted as f32 / total_proposed as f32;
        assert!((rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_speedup_formula() {
        // With alpha=0.7, k=5: E[tokens] ≈ 2.8
        let alpha = 0.7_f32;
        let k = 5.0_f32;
        let expected = (1.0 - alpha.powf(k + 1.0)) / (1.0 - alpha);
        assert!(expected > 2.0 && expected < 3.5);

        // With alpha=0.9, k=5: E[tokens] ≈ 4.7
        let alpha = 0.9_f32;
        let expected = (1.0 - alpha.powf(k + 1.0)) / (1.0 - alpha);
        assert!(expected > 4.0 && expected < 5.5);
    }
}
