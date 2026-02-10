//! Tiered memory token generation for efficient large model inference.
//!
//! Like `Generator`, but uses `TieredWeightStore` for 3-tier memory management,
//! enabling efficient inference of models larger than available VRAM.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;

use super::arch::ModelConfig;
use super::compute::ComputeEngine;
use super::generate::{GenerationStats, SamplingParams, TokenCallback};
use super::kernels::sampling::{RepetitionPenalty, SamplingKernel};
use super::tiered::{TieredStats, TieredWeightStore};
use super::InferenceError;

/// Token generator with tiered memory management.
///
/// Uses the 3-tier memory hierarchy (VRAM ← RAM ← NVMe) for efficient
/// inference of models larger than available VRAM.
pub struct TieredGenerator {
    /// Model weights with tiered storage.
    weights: TieredWeightStore,

    /// Compute engine.
    engine: ComputeEngine,

    /// Sampling kernel.
    sampler: SamplingKernel,

    /// CUDA device.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,

    /// Model configuration.
    config: ModelConfig,

    /// Vocabulary size.
    vocab_size: usize,

    /// Current position in generation.
    position: usize,

    /// RNG state for sampling.
    rng_state: u64,

    /// Prefetch depth for layer loading.
    prefetch_depth: usize,
}

impl TieredGenerator {
    /// Create a new tiered generator.
    ///
    /// # Arguments
    ///
    /// * `weights` - TieredWeightStore with multi-tier memory management
    /// * `config` - Model configuration
    /// * `max_seq_len` - Maximum sequence length for generation
    /// * `prefetch_depth` - How many layers ahead to prefetch (2-4 recommended)
    pub fn new(
        weights: TieredWeightStore,
        config: ModelConfig,
        max_seq_len: usize,
        prefetch_depth: usize,
    ) -> Result<Self, InferenceError> {
        let device = weights.device().clone();
        let vocab_size = config.vocab_size;

        let engine = ComputeEngine::new(config.clone(), max_seq_len, device.clone())?;
        let sampler = SamplingKernel::new(device.clone())?;

        Ok(Self {
            weights,
            engine,
            sampler,
            device,
            config,
            vocab_size,
            position: 0,
            rng_state: 42,
            prefetch_depth,
        })
    }

    /// Reset generator state for new generation.
    pub fn reset(&mut self) {
        self.engine.reset_cache();
        self.position = 0;
    }

    /// Get tiered storage statistics.
    pub fn tier_stats(&self) -> &TieredStats {
        self.weights.stats()
    }

    /// Get tier distribution summary.
    pub fn tier_summary(&self) -> super::tiered::store::TierSummary {
        self.weights.tier_summary()
    }

    /// Generate tokens from input IDs.
    ///
    /// Returns generated token IDs (not including input).
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        params: Option<&SamplingParams>,
    ) -> Result<Vec<u32>, InferenceError> {
        let params = params.cloned().unwrap_or_default();
        let mut output_ids = Vec::new();

        self.reset();
        self.rng_state = params.seed;

        // Prefill phase (uses tiered loading)
        let _hidden =
            self.engine
                .prefill_tiered(input_ids, &mut self.weights, self.prefetch_depth)?;
        self.position = input_ids.len();

        // Set up repetition penalty
        let mut rep_penalty =
            RepetitionPenalty::new(params.repetition_penalty, params.repetition_context);
        for &id in input_ids {
            rep_penalty.add_token(id);
        }

        // Decode phase
        for _ in 0..params.max_tokens {
            // Get last token's logits
            let logits = self.engine.get_logits()?;

            // Sample next token
            let next_token = self.sample_token(logits, &params, &mut rep_penalty)?;

            // Check for stop token
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            output_ids.push(next_token);
            rep_penalty.add_token(next_token);

            // Decode next token (uses tiered loading)
            let _hidden =
                self.engine
                    .decode_tiered(next_token, &mut self.weights, self.prefetch_depth)?;
            self.position += 1;
        }

        Ok(output_ids)
    }

    /// Generate tokens with streaming callback.
    ///
    /// The callback is called for each generated token. Return `false` to stop.
    pub fn generate_stream(
        &mut self,
        input_ids: &[u32],
        params: Option<&SamplingParams>,
        mut callback: TokenCallback,
    ) -> Result<GenerationStats, InferenceError> {
        let params = params.cloned().unwrap_or_default();
        let start_time = Instant::now();

        self.reset();
        self.rng_state = params.seed;

        let mut stats = GenerationStats::default();

        // Prefill phase (tiered loading)
        let prefill_start = Instant::now();
        let _hidden =
            self.engine
                .prefill_tiered(input_ids, &mut self.weights, self.prefetch_depth)?;
        self.position = input_ids.len();
        stats.prefill_time_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        stats.prefill_tokens = input_ids.len();

        // Set up repetition penalty
        let mut rep_penalty =
            RepetitionPenalty::new(params.repetition_penalty, params.repetition_context);
        for &id in input_ids {
            rep_penalty.add_token(id);
        }

        // Decode phase
        for _ in 0..params.max_tokens {
            let forward_start = Instant::now();
            let logits = self.engine.get_logits()?;

            // Sample next token
            let sample_start = Instant::now();
            let next_token = self.sample_token(logits, &params, &mut rep_penalty)?;
            stats.sampling_time_ms += sample_start.elapsed().as_secs_f64() * 1000.0;

            // Check for stop token
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            stats.tokens_generated += 1;

            // Callback
            if !callback(next_token) {
                break;
            }

            rep_penalty.add_token(next_token);

            // Decode next token (tiered loading)
            let _hidden =
                self.engine
                    .decode_tiered(next_token, &mut self.weights, self.prefetch_depth)?;
            stats.forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;
            self.position += 1;
        }

        stats.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        stats.tokens_per_second = stats.tokens_generated as f64 / (stats.total_time_ms / 1000.0);

        Ok(stats)
    }

    /// Sample a token from logits (adapted from LazyGenerator).
    fn sample_token(
        &mut self,
        mut logits: super::tensor::GpuTensor,
        params: &SamplingParams,
        rep_penalty: &mut RepetitionPenalty,
    ) -> Result<u32, InferenceError> {
        // Apply repetition penalty (on CPU for now)
        if params.repetition_penalty != 1.0 && !rep_penalty.context.is_empty() {
            // Copy logits to CPU
            let mut host_logits = vec![0u8; logits.numel() * 2];
            logits.copy_to_host(&mut host_logits)?;

            // Convert to f32
            let mut logits_f32: Vec<f32> = host_logits
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();

            // Apply penalty
            rep_penalty.apply(&mut logits_f32);

            // Convert back to f16 and upload
            let logits_f16: Vec<u8> = logits_f32
                .iter()
                .flat_map(|&f| half::f16::from_f32(f).to_le_bytes())
                .collect();
            logits.copy_from_host(&logits_f16)?;
        }

        // Sample
        if params.temperature == 0.0 || params.temperature < 0.01 {
            // Greedy sampling
            self.sampler.sample_greedy(&logits)
        } else {
            // Update RNG state
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);

            self.sampler.sample(
                &mut logits,
                params.temperature,
                params.top_k,
                params.top_p,
                self.rng_state,
            )
        }
    }

    /// Get the current generation position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get mutable access to weights (for tier management).
    pub fn weights_mut(&mut self) -> &mut TieredWeightStore {
        &mut self.weights
    }

    /// Set prefetch depth.
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth;
    }
}
