//! Token generation and sampling.
//!
//! Provides streaming token generation with temperature, top-k, and top-p sampling.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaDevice;

use super::compute::ComputeEngine;
use super::kernels::sampling::{RepetitionPenalty, SamplingKernel};
use super::weight_store::WeightStore;
use super::InferenceError;

/// Sampling parameters for generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for sampling (1.0 = no change, >1.0 = more random).
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold (1.0 = disabled).
    pub top_p: f32,

    /// Top-k sampling (0 = disabled).
    pub top_k: usize,

    /// Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition).
    pub repetition_penalty: f32,

    /// Maximum context length for repetition penalty.
    pub repetition_context: usize,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Stop token IDs.
    pub stop_tokens: Vec<u32>,

    /// Random seed for sampling.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            repetition_context: 64,
            max_tokens: 256,
            stop_tokens: vec![],
            seed: 42,
        }
    }
}

impl SamplingParams {
    /// Create greedy sampling parameters (temperature=0, argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
            max_tokens: 256,
            stop_tokens: vec![],
            seed: 0,
        }
    }

    /// Create creative sampling parameters.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.2,
            repetition_context: 128,
            max_tokens: 512,
            stop_tokens: vec![],
            seed: 42,
        }
    }

    /// Create precise/deterministic sampling parameters.
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            repetition_context: 64,
            max_tokens: 256,
            stop_tokens: vec![],
            seed: 42,
        }
    }
}

/// Generation statistics.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Total tokens generated.
    pub tokens_generated: usize,

    /// Time spent in forward passes (milliseconds).
    pub forward_time_ms: f64,

    /// Time spent in sampling (milliseconds).
    pub sampling_time_ms: f64,

    /// Total generation time (milliseconds).
    pub total_time_ms: f64,

    /// Tokens per second.
    pub tokens_per_second: f64,

    /// Prefill time (milliseconds).
    pub prefill_time_ms: f64,

    /// Prefill tokens.
    pub prefill_tokens: usize,
}

/// Callback for streaming token output.
pub type TokenCallback = Box<dyn FnMut(u32) -> bool + Send>;

/// Token generator with streaming output.
pub struct Generator {
    /// Model weights.
    weights: WeightStore,

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

impl Generator {
    /// Create a new generator.
    pub fn new(weights: WeightStore, max_seq_len: usize) -> Result<Self, InferenceError> {
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

        // Prefill phase
        let _hidden = self.engine.prefill(input_ids, &self.weights)?;
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

            // Decode next token
            let _hidden = self.engine.decode(next_token, &self.weights)?;
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

        // Prefill phase
        let prefill_start = Instant::now();
        let _hidden = self.engine.prefill(input_ids, &self.weights)?;
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
            // Forward pass timing
            let forward_start = Instant::now();
            let logits = self.engine.get_logits()?;
            stats.forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

            // Sampling timing
            let sample_start = Instant::now();
            let next_token = self.sample_token(logits, &params, &mut rep_penalty)?;
            stats.sampling_time_ms += sample_start.elapsed().as_secs_f64() * 1000.0;

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

            // Decode next token
            let forward_start = Instant::now();
            let _hidden = self.engine.decode(next_token, &self.weights)?;
            stats.forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;
            self.position += 1;
        }

        stats.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        if stats.total_time_ms > 0.0 {
            stats.tokens_per_second =
                stats.tokens_generated as f64 / (stats.total_time_ms / 1000.0);
        }

        Ok(stats)
    }

    /// Sample a single token from logits.
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

    /// Get reference to weights.
    pub fn weights(&self) -> &WeightStore {
        &self.weights
    }

    /// Get reference to compute engine.
    pub fn engine(&self) -> &ComputeEngine {
        &self.engine
    }

    /// Get mutable reference to compute engine.
    pub fn engine_mut(&mut self) -> &mut ComputeEngine {
        &mut self.engine
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get current position.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// Builder for Generator with custom settings.
pub struct GeneratorBuilder {
    max_seq_len: usize,
    device_id: usize,
}

impl Default for GeneratorBuilder {
    fn default() -> Self {
        Self {
            max_seq_len: 4096,
            device_id: 0,
        }
    }
}

impl GeneratorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Set CUDA device ID.
    pub fn device_id(mut self, id: usize) -> Self {
        self.device_id = id;
        self
    }

    /// Build generator from HCT model path.
    pub fn build(
        self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<Generator, InferenceError> {
        let weights = WeightStore::load_hct(model_path, None, self.device_id)?;
        Generator::new(weights, self.max_seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 0.7).abs() < 0.01);
        assert!((params.top_p - 0.9).abs() < 0.01);
        assert_eq!(params.top_k, 40);
        assert!((params.repetition_penalty - 1.1).abs() < 0.01);
    }

    #[test]
    fn test_sampling_params_greedy() {
        let params = SamplingParams::greedy();
        assert!(params.temperature < 0.01);
        assert_eq!(params.top_k, 0);
        assert!((params.repetition_penalty - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sampling_params_creative() {
        let params = SamplingParams::creative();
        assert!(params.temperature > 0.8);
        assert!(params.top_p > 0.9);
    }

    #[test]
    fn test_generation_stats_default() {
        let stats = GenerationStats::default();
        assert_eq!(stats.tokens_generated, 0);
        assert!(stats.total_time_ms < 0.01);
    }
}
