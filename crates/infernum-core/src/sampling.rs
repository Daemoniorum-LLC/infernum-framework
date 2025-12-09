//! Sampling parameters for text generation.

use serde::{Deserialize, Serialize};

/// Parameters controlling text generation sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    /// Default: 1.0
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    /// Default: 1.0
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled).
    /// Default: 0
    #[serde(default)]
    pub top_k: u32,

    /// Minimum probability for min-p sampling (0.0 = disabled).
    /// Default: 0.0
    #[serde(default)]
    pub min_p: f32,

    /// Repetition penalty (1.0 = no penalty).
    /// Default: 1.0
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Presence penalty (-2.0 to 2.0).
    /// Default: 0.0
    #[serde(default)]
    pub presence_penalty: f32,

    /// Frequency penalty (-2.0 to 2.0).
    /// Default: 0.0
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Stop sequences that halt generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Maximum number of tokens to generate.
    /// Default: 256
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Random seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_max_tokens() -> u32 {
    256
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: Vec::new(),
            max_tokens: 256,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Creates greedy sampling parameters (temperature = 0).
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Creates balanced sampling parameters.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            ..Default::default()
        }
    }

    /// Creates creative sampling parameters.
    #[must_use]
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Sets the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the top-p value.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets the top-k value.
    #[must_use]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Adds a stop sequence.
    #[must_use]
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Sets the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the sampling parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err("top_p must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err("min_p must be between 0.0 and 1.0".to_string());
        }
        if self.repetition_penalty < 0.0 {
            return Err("repetition_penalty must be non-negative".to_string());
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err("presence_penalty must be between -2.0 and 2.0".to_string());
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err("frequency_penalty must be between -2.0 and 2.0".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
        Ok(())
    }
}
