//! Lazy-loading token generation for memory-efficient inference.
//!
//! Like `Generator`, but uses `LazyWeightStore` for on-demand layer loading,
//! enabling inference of models larger than available VRAM.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;

use super::compute::ComputeEngine;
use super::generate::{GenerationStats, SamplingParams, TokenCallback};
use super::kernels::sampling::{RepetitionPenalty, SamplingKernel};
use super::lazy_weight_store::LazyWeightStore;
use super::InferenceError;

/// Token generator with lazy layer loading.
///
/// Loads transformer layers on-demand during inference, enabling
/// models larger than available VRAM on consumer GPUs.
pub struct LazyGenerator {
    /// Model weights with lazy loading.
    weights: LazyWeightStore,

    /// Compute engine.
    engine: ComputeEngine,

    /// Sampling kernel.
    sampler: SamplingKernel,

    /// CUDA device.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,

    /// Vocabulary size.
    vocab_size: usize,

    /// Current position in generation.
    position: usize,

    /// RNG state for sampling.
    rng_state: u64,
}

impl LazyGenerator {
    /// Create a new lazy generator.
    ///
    /// # Arguments
    ///
    /// * `weights` - Lazy weight store with on-demand loading
    /// * `max_seq_len` - Maximum sequence length for generation
    pub fn new(weights: LazyWeightStore, max_seq_len: usize) -> Result<Self, InferenceError> {
        let device = weights.device().clone();
        let config = weights.config.clone();
        let vocab_size = config.vocab_size;

        let engine = ComputeEngine::new(config, max_seq_len, device.clone())?;
        let sampler = SamplingKernel::new(device.clone())?;

        Ok(Self {
            weights,
            engine,
            sampler,
            device,
            vocab_size,
            position: 0,
            rng_state: 42,
        })
    }

    /// Reset generator state for new generation.
    pub fn reset(&mut self) {
        self.engine.reset_cache();
        self.position = 0;
    }

    /// Get lazy loading statistics.
    pub fn layer_stats(&self) -> super::lazy_layers::LazyLayerStats {
        self.weights.stats()
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

        // Prefill phase (uses lazy loading)
        let _hidden = self.engine.prefill_lazy(input_ids, &mut self.weights)?;
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

            // Decode next token (uses lazy loading)
            let _hidden = self.engine.decode_lazy(next_token, &mut self.weights)?;
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

        // Prefill phase (lazy loading)
        let prefill_start = Instant::now();
        let _hidden = self.engine.prefill_lazy(input_ids, &mut self.weights)?;
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
            let next_token = self.sample_token(logits, &params, &mut rep_penalty)?;

            let decode_time = forward_start.elapsed().as_secs_f64() * 1000.0;
            stats.forward_time_ms += decode_time;

            // Check for stop token
            if params.stop_tokens.contains(&next_token) {
                break;
            }

            stats.tokens_generated += 1;

            // Call user callback
            if !callback(next_token) {
                break;
            }

            rep_penalty.add_token(next_token);

            // Decode next token (lazy loading)
            let _hidden = self.engine.decode_lazy(next_token, &mut self.weights)?;
            self.position += 1;
        }

        stats.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        if stats.forward_time_ms > 0.0 {
            stats.tokens_per_second =
                stats.tokens_generated as f64 / (stats.forward_time_ms / 1000.0);
        }

        // Log lazy loading stats
        let layer_stats = self.weights.stats();
        tracing::info!(
            "Lazy loading stats: {} loads, {} evictions, {:.1}% hit rate",
            layer_stats.total_loads,
            layer_stats.total_evictions,
            layer_stats.hit_rate * 100.0
        );

        Ok(stats)
    }

    /// Sample a token from logits (adapted from Generator).
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
}

impl std::fmt::Debug for LazyGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.weights.stats();
        f.debug_struct("LazyGenerator")
            .field("vocab_size", &self.vocab_size)
            .field("position", &self.position)
            .field("layers_loaded", &stats.layers_loaded)
            .field("total_layers", &self.weights.num_layers())
            .field("vram_used_mb", &(stats.vram_used / (1024 * 1024)))
            .field("hit_rate", &format!("{:.1}%", stats.hit_rate * 100.0))
            .finish()
    }
}
