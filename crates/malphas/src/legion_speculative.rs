//! Legion-enhanced speculative decoding.
//!
//! This module integrates Legion's multi-agent speculative decoding with
//! the inference pipeline. Multiple draft agents generate candidates
//! from different frequency bands, which are ranked via wave interference
//! in the Legion field before verification by the oracle model.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   LegionSpeculativeDecoder                       │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │              Draft Generation (Parallel)                  │  │
//! │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │  │
//! │  │  │Strategic│ │Tactical │ │Operative│ │Reflective│        │  │
//! │  │  │ Draft   │ │ Draft   │ │ Draft   │ │ Draft   │        │  │
//! │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │  │
//! │  │       │           │           │           │              │  │
//! │  │       └───────────┴─────┬─────┴───────────┘              │  │
//! │  │                         ▼                                │  │
//! │  │  ┌─────────────────────────────────────────────────┐    │  │
//! │  │  │              Legion Field (∿)                   │    │  │
//! │  │  │  Wave interference → Coherence ranking          │    │  │
//! │  │  └─────────────────────────────────────────────────┘    │  │
//! │  │                         │                                │  │
//! │  │                         ▼                                │  │
//! │  │  ┌─────────────────────────────────────────────────┐    │  │
//! │  │  │              Oracle Verification               │    │  │
//! │  │  │  Main model verifies top-ranked paths          │    │  │
//! │  │  └─────────────────────────────────────────────────┘    │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Benefits
//!
//! - Multiple perspectives on generation (frequency band specialization)
//! - Coherence-based path ranking identifies most likely sequences
//! - Parallel draft generation across bands
//! - Graceful degradation: any single agent can still produce output

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use parking_lot::Mutex;

use abaddon::models::ModelKind;
use abaddon::sampler::Sampler;
use abaddon::speculative::SpeculativeStats;
use abaddon::tokenizer::Tokenizer;
use infernum_core::{Result, SamplingParams};
use legion::{
    DraftSequence, FrequencyBand, LegionPattern, RankedPath, SpeculativeLegion,
    SpeculativeLegionConfig, TokenId,
};

/// Configuration for Legion-enhanced speculative decoding.
#[derive(Debug, Clone)]
pub struct LegionSpeculativeConfig {
    /// Number of speculative tokens to generate per round.
    pub num_speculative_tokens: u32,
    /// Acceptance threshold for verification (0.0 - 1.0).
    pub acceptance_threshold: f32,
    /// Number of draft agents (frequency bands).
    pub draft_agent_count: usize,
    /// Coherence threshold for path selection (0.0 - 1.0).
    pub coherence_threshold: f32,
    /// Maximum paths to verify with oracle.
    pub max_verify_paths: usize,
    /// Enable adaptive lookahead based on acceptance rate.
    pub adaptive_lookahead: bool,
    /// Target acceptance rate for adaptive lookahead.
    pub target_acceptance_rate: f32,
}

impl Default for LegionSpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            acceptance_threshold: 0.9,
            draft_agent_count: 4,
            coherence_threshold: 0.7,
            max_verify_paths: 3,
            adaptive_lookahead: true,
            target_acceptance_rate: 0.75,
        }
    }
}

impl LegionSpeculativeConfig {
    /// Creates a new config with specified number of agents.
    #[must_use]
    pub fn new(num_speculative_tokens: u32, draft_agent_count: usize) -> Self {
        Self {
            num_speculative_tokens,
            draft_agent_count,
            ..Default::default()
        }
    }

    /// Sets the number of draft agents.
    #[must_use]
    pub fn with_draft_agent_count(mut self, count: usize) -> Self {
        self.draft_agent_count = count;
        self
    }

    /// Sets the coherence threshold.
    #[must_use]
    pub fn with_coherence_threshold(mut self, threshold: f32) -> Self {
        self.coherence_threshold = threshold;
        self
    }

    /// Sets the acceptance threshold.
    #[must_use]
    pub fn with_acceptance_threshold(mut self, threshold: f32) -> Self {
        self.acceptance_threshold = threshold;
        self
    }
}

/// Legion-enhanced statistics extending base speculative stats.
#[derive(Debug, Clone, Default)]
pub struct LegionSpeculativeStats {
    /// Base speculative decoding stats.
    pub base: SpeculativeStats,
    /// Total drafts generated across all agents.
    pub total_drafts: u64,
    /// Average coherence of selected paths.
    pub avg_coherence: f32,
    /// Number of paths that exceeded coherence threshold.
    pub coherent_paths: u64,
    /// Paths verified by oracle.
    pub paths_verified: u64,
    /// Draft stats per frequency band.
    pub band_stats: Vec<BandStats>,
}

/// Statistics for a single frequency band.
#[derive(Debug, Clone)]
pub struct BandStats {
    /// Frequency band identifier.
    pub band: FrequencyBand,
    /// Tokens contributed from this band.
    pub tokens_contributed: u64,
    /// Times this band's path was selected.
    pub times_selected: u64,
}

impl Default for BandStats {
    fn default() -> Self {
        Self {
            band: FrequencyBand::Strategic, // Default band
            tokens_contributed: 0,
            times_selected: 0,
        }
    }
}

/// Legion-enhanced speculative decoder.
///
/// Uses multiple draft agents across frequency bands, ranks paths via
/// wave interference, and verifies with the oracle model.
pub struct LegionSpeculativeDecoder {
    /// Draft model (smaller, faster).
    draft_model: Arc<Mutex<ModelKind>>,
    /// Draft model tokenizer.
    draft_tokenizer: Arc<Tokenizer>,
    /// Legion speculative coordinator.
    legion: SpeculativeLegion,
    /// Configuration.
    config: LegionSpeculativeConfig,
    /// Computation device.
    device: Device,
    /// Data type for computations.
    dtype: DType,
    /// Accumulated statistics.
    stats: Mutex<LegionSpeculativeStats>,
    /// Current lookahead (adaptive).
    current_lookahead: Mutex<usize>,
}

impl LegionSpeculativeDecoder {
    /// Creates a new Legion speculative decoder.
    pub fn new(
        draft_model: ModelKind,
        draft_tokenizer: Tokenizer,
        config: LegionSpeculativeConfig,
        device: Device,
        dtype: DType,
    ) -> Self {
        let legion_config = SpeculativeLegionConfig {
            draft_agent_count: config.draft_agent_count,
            lookahead: config.num_speculative_tokens as usize,
            coherence_threshold: config.coherence_threshold,
            max_verify_paths: config.max_verify_paths,
            adaptive_lookahead: config.adaptive_lookahead,
            target_acceptance_rate: config.target_acceptance_rate,
            ..Default::default()
        };

        let legion = SpeculativeLegion::new(legion_config);
        let initial_lookahead = config.num_speculative_tokens as usize;

        Self {
            draft_model: Arc::new(Mutex::new(draft_model)),
            draft_tokenizer: Arc::new(draft_tokenizer),
            legion,
            config,
            device,
            dtype,
            stats: Mutex::new(LegionSpeculativeStats::default()),
            current_lookahead: Mutex::new(initial_lookahead),
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &LegionSpeculativeConfig {
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
    pub fn stats(&self) -> LegionSpeculativeStats {
        self.stats.lock().clone()
    }

    /// Returns base speculative stats (for compatibility).
    #[must_use]
    pub fn base_stats(&self) -> SpeculativeStats {
        self.stats.lock().base.clone()
    }

    /// Resets statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = LegionSpeculativeStats::default();
    }

    /// Returns the current adaptive lookahead.
    #[must_use]
    pub fn current_lookahead(&self) -> usize {
        *self.current_lookahead.lock()
    }

    /// Generates tokens using Legion-enhanced speculative decoding.
    ///
    /// # Arguments
    ///
    /// * `main_model` - The main/oracle model for verification
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
        while generated_tokens.len() < max_tokens as usize {
            self.stats.lock().base.rounds += 1;

            // Get current lookahead
            let lookahead = *self.current_lookahead.lock();

            // Step 1: Generate draft sequences from multiple bands
            let context: Vec<TokenId> = prompt_tokens
                .iter()
                .chain(generated_tokens.iter())
                .map(|&t| t as TokenId)
                .collect();

            let draft_sequences =
                self.generate_draft_sequences(&context, lookahead, sampling_params, eos_token)?;

            if draft_sequences.is_empty() {
                break;
            }

            // Update draft stats
            {
                let mut stats = self.stats.lock();
                stats.total_drafts += draft_sequences.len() as u64;
            }

            // Step 2: Rank paths via Legion field interference
            let ranked_paths = self.legion.rank_paths(draft_sequences);

            // Filter by coherence threshold
            let coherent_paths: Vec<_> = ranked_paths
                .iter()
                .filter(|p| p.coherence >= self.config.coherence_threshold)
                .take(self.config.max_verify_paths)
                .collect();

            {
                let mut stats = self.stats.lock();
                stats.coherent_paths += coherent_paths.len() as u64;
                if !coherent_paths.is_empty() {
                    let avg_coh: f32 = coherent_paths.iter().map(|p| p.coherence).sum::<f32>()
                        / coherent_paths.len() as f32;
                    // Running average
                    let n = stats.base.rounds as f32;
                    stats.avg_coherence = (stats.avg_coherence * (n - 1.0) + avg_coh) / n;
                }
            }

            // Step 3: Verify top paths with oracle
            let (accepted, next_token, selected_band) = self.verify_ranked_paths(
                main_model,
                prompt_tokens,
                &generated_tokens,
                &ranked_paths,
                &mut sampler,
                current_pos,
            )?;

            // Update statistics
            let tokens_in_round = if ranked_paths.is_empty() {
                0
            } else {
                ranked_paths[0].draft.tokens.len()
            };

            {
                let mut stats = self.stats.lock();
                stats.base.total_tokens += tokens_in_round as u64;
                stats.base.accepted_tokens += accepted as u64;
                stats.base.rejected_tokens += (tokens_in_round.saturating_sub(accepted)) as u64;
                stats.paths_verified += ranked_paths.len().min(self.config.max_verify_paths) as u64;

                // Track band-specific stats
                if let Some(band) = selected_band {
                    let band_stat = stats.band_stats.iter_mut().find(|b| b.band == band);
                    if let Some(stat) = band_stat {
                        stat.tokens_contributed += accepted as u64;
                        stat.times_selected += 1;
                    } else {
                        stats.band_stats.push(BandStats {
                            band,
                            tokens_contributed: accepted as u64,
                            times_selected: 1,
                        });
                    }
                }
            }

            // Add accepted tokens from the best path
            if !ranked_paths.is_empty() {
                let best_path = &ranked_paths[0];
                for i in 0..accepted {
                    if let Some(&token) = best_path.draft.tokens.get(i) {
                        let token = token as u32;
                        if token == eos_token {
                            return Ok((generated_tokens, generated_text));
                        }
                        generated_tokens.push(token);
                        let token_text = self.draft_tokenizer.decode_token(token)?;
                        generated_text.push(token_text);
                    }
                }
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

            // Adaptive lookahead adjustment
            if self.config.adaptive_lookahead {
                self.adjust_lookahead(accepted, tokens_in_round);
            }

            // Check stop sequences
            let full_text: String = generated_text.join("");
            if sampler.is_stop_token(&full_text) {
                break;
            }
        }

        Ok((generated_tokens, generated_text))
    }

    /// Generates draft sequences from multiple frequency bands.
    fn generate_draft_sequences(
        &self,
        context: &[TokenId],
        lookahead: usize,
        sampling_params: &SamplingParams,
        eos_token: u32,
    ) -> Result<Vec<DraftSequence>> {
        let bands = self.get_active_bands();
        let mut sequences = Vec::with_capacity(bands.len());

        let mut draft = self.draft_model.lock();

        for band in bands {
            // Each band generates with slightly different temperature
            // to explore different parts of the probability space
            let band_params = self.adjust_params_for_band(&band, sampling_params);
            let mut sampler = Sampler::new(band_params);

            let mut tokens = Vec::with_capacity(lookahead);
            let mut log_probs = Vec::with_capacity(lookahead);
            let base_pos = context.len();

            for i in 0..lookahead {
                let input_token = if i == 0 {
                    context.last().copied().unwrap_or(0) as u32
                } else {
                    tokens.last().copied().unwrap_or(0) as u32
                };

                let input = Tensor::new(&[input_token], &self.device)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

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

                let next_token = sampler.sample(&logits_vec);

                // Compute log probability
                let probs = softmax(&logits_vec);
                let log_prob = probs
                    .get(next_token as usize)
                    .map(|p| p.ln())
                    .unwrap_or(f32::NEG_INFINITY);

                tokens.push(next_token as TokenId);
                log_probs.push(log_prob);

                if next_token == eos_token {
                    break;
                }
            }

            // Create pattern from token sequence for interference
            let pattern = self.tokens_to_pattern(&tokens);

            sequences.push(DraftSequence {
                agent_id: format!("draft-{:?}", band),
                band: band.clone(),
                tokens,
                log_probs,
                pattern,
                timestamp: std::time::Instant::now(),
            });
        }

        // Clear draft model cache after all bands
        draft.clear_cache();

        Ok(sequences)
    }

    /// Verifies ranked paths with the oracle model.
    ///
    /// Returns (num_accepted, optional_next_token, selected_band).
    fn verify_ranked_paths(
        &self,
        main_model: &mut ModelKind,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
        ranked_paths: &[RankedPath],
        sampler: &mut Sampler,
        base_pos: usize,
    ) -> Result<(usize, Option<u32>, Option<FrequencyBand>)> {
        if ranked_paths.is_empty() {
            return Ok((0, None, None));
        }

        // Verify paths in order of coherence ranking
        for path in ranked_paths.iter().take(self.config.max_verify_paths) {
            let draft_tokens: Vec<u32> = path.draft.tokens.iter().map(|&t| t as u32).collect();

            if draft_tokens.is_empty() {
                continue;
            }

            // Create input tensor with all draft tokens for parallel verification
            let input_tokens: Vec<u32> = if generated_tokens.is_empty() {
                let last_prompt = *prompt_tokens.last().unwrap_or(&0);
                std::iter::once(last_prompt)
                    .chain(draft_tokens.iter().copied())
                    .collect()
            } else {
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
                let pos_logits = logits
                    .i((0, i, ..))
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                    .to_dtype(DType::F32)
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

                let logits_vec: Vec<f32> = pos_logits
                    .to_vec1()
                    .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

                let probs = softmax(&logits_vec);
                let main_prob = probs.get(*draft_token as usize).copied().unwrap_or(0.0);

                // Accept with coherence-adjusted threshold
                // Higher coherence paths get more lenient thresholds
                let adjusted_threshold = threshold * (1.0 - path.coherence * 0.2);

                if main_prob >= adjusted_threshold * get_max_prob(&probs) {
                    accepted += 1;
                } else {
                    // Reject - sample from main model's distribution
                    let next_token = sampler.sample(&logits_vec);
                    return Ok((accepted, Some(next_token), Some(path.draft.band.clone())));
                }
            }

            // All tokens accepted - sample next token
            let last_pos_logits = logits
                .i((0, draft_tokens.len(), ..))
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?
                .to_dtype(DType::F32)
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            let logits_vec: Vec<f32> = last_pos_logits
                .to_vec1()
                .map_err(|e| infernum_core::Error::internal(e.to_string()))?;

            let next_token = sampler.sample(&logits_vec);
            return Ok((accepted, Some(next_token), Some(path.draft.band.clone())));
        }

        Ok((0, None, None))
    }

    /// Returns the active frequency bands based on config.
    fn get_active_bands(&self) -> Vec<FrequencyBand> {
        let all_bands = [
            FrequencyBand::Strategic,
            FrequencyBand::Tactical,
            FrequencyBand::Operational,
            FrequencyBand::Reflective,
        ];

        all_bands
            .iter()
            .take(self.config.draft_agent_count)
            .cloned()
            .collect()
    }

    /// Adjusts sampling parameters based on frequency band.
    fn adjust_params_for_band(
        &self,
        band: &FrequencyBand,
        base_params: &SamplingParams,
    ) -> SamplingParams {
        let mut params = base_params.clone();

        // Each band explores different temperature ranges
        let temp_adjustment = match band {
            FrequencyBand::Anima => 0.0,        // DC component - unchanged
            FrequencyBand::Strategic => -0.1,   // Lower temp for planning
            FrequencyBand::Tactical => 0.0,     // Standard temp
            FrequencyBand::Operational => 0.05, // Slightly higher for diversity
            FrequencyBand::Verification => 0.1, // More exploration
            FrequencyBand::Reflective => 0.15,  // Most exploration
        };

        params.temperature = (params.temperature + temp_adjustment).max(0.1);
        params
    }

    /// Converts token sequence to Legion pattern for interference.
    fn tokens_to_pattern(&self, tokens: &[TokenId]) -> LegionPattern {
        // Create spectral coefficients from token sequence
        // Use token IDs to create a unique frequency signature
        const PATTERN_SIZE: usize = 32;

        let mut coefficients = vec![0.0f32; PATTERN_SIZE * PATTERN_SIZE];

        for (i, &token) in tokens.iter().enumerate() {
            // Map token to spectral coefficient positions
            let freq_x = (token as usize % PATTERN_SIZE) as f32;
            let freq_y = (token as usize / PATTERN_SIZE % PATTERN_SIZE) as f32;

            // Add contribution with position-based phase
            let phase = (i as f32) * std::f32::consts::PI / tokens.len() as f32;
            let magnitude = 1.0 / (i + 1) as f32; // Decay with position

            let idx = (freq_y as usize * PATTERN_SIZE + freq_x as usize) % coefficients.len();
            coefficients[idx] += magnitude * phase.cos();
        }

        LegionPattern::from_coefficients(coefficients, PATTERN_SIZE, PATTERN_SIZE)
    }

    /// Adjusts lookahead based on acceptance rate.
    fn adjust_lookahead(&self, accepted: usize, total: usize) {
        if total == 0 {
            return;
        }

        let acceptance_rate = accepted as f32 / total as f32;
        let target = self.config.target_acceptance_rate;
        let mut lookahead = self.current_lookahead.lock();

        if acceptance_rate > target + 0.1 {
            // High acceptance - increase lookahead
            *lookahead = (*lookahead + 1).min(self.config.num_speculative_tokens as usize * 2);
        } else if acceptance_rate < target - 0.1 {
            // Low acceptance - decrease lookahead
            *lookahead = (*lookahead - 1).max(2);
        }
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
    fn test_legion_speculative_config_default() {
        let config = LegionSpeculativeConfig::default();
        assert_eq!(config.draft_agent_count, 4);
        assert!((config.coherence_threshold - 0.7).abs() < 0.001);
        assert_eq!(config.max_verify_paths, 3);
        assert!(config.adaptive_lookahead);
    }

    #[test]
    fn test_legion_speculative_config_builder() {
        let config = LegionSpeculativeConfig::new(5, 4)
            .with_draft_agent_count(6)
            .with_coherence_threshold(0.8);

        assert_eq!(config.draft_agent_count, 6);
        assert!((config.coherence_threshold - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_legion_speculative_stats_default() {
        let stats = LegionSpeculativeStats::default();
        assert_eq!(stats.total_drafts, 0);
        assert_eq!(stats.coherent_paths, 0);
        assert_eq!(stats.paths_verified, 0);
        assert_eq!(stats.avg_coherence, 0.0);
    }

    #[test]
    fn test_band_stats_default() {
        let stats = BandStats::default();
        assert_eq!(stats.tokens_contributed, 0);
        assert_eq!(stats.times_selected, 0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_get_max_prob() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        assert!((get_max_prob(&probs) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_get_active_bands_count() {
        // Simulating get_active_bands logic
        let all_bands = [
            FrequencyBand::Strategic,
            FrequencyBand::Tactical,
            FrequencyBand::Operational,
            FrequencyBand::Reflective,
        ];

        let count = 2;
        let active: Vec<_> = all_bands.iter().take(count).cloned().collect();
        assert_eq!(active.len(), 2);
        assert!(matches!(active[0], FrequencyBand::Strategic));
        assert!(matches!(active[1], FrequencyBand::Tactical));
    }
}
