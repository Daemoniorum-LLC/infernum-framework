//! Token sampling implementations.

use infernum_core::SamplingParams;

/// Token sampler for text generation.
pub struct Sampler {
    params: SamplingParams,
    rng: Option<fastrand::Rng>,
}

impl Sampler {
    /// Creates a new sampler with the given parameters.
    #[must_use]
    pub fn new(params: SamplingParams) -> Self {
        let rng = params.seed.map(|seed| {
            let mut rng = fastrand::Rng::new();
            rng.seed(seed);
            rng
        });

        Self { params, rng }
    }

    /// Adds a token to the repetition penalty context (placeholder for future use).
    #[allow(unused_variables)]
    pub fn add_token(&mut self, token: u32) {
        // TODO: Implement repetition penalty tracking
    }

    /// Samples a token from the logits.
    #[must_use]
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.params.temperature == 0.0 {
            // Greedy sampling
            return self.argmax(logits);
        }

        // Apply temperature
        let scaled: Vec<f32> = logits
            .iter()
            .map(|&l| l / self.params.temperature)
            .collect();

        // Apply top-k
        let filtered = if self.params.top_k > 0 {
            self.top_k_filter(&scaled, self.params.top_k as usize)
        } else {
            scaled
        };

        // Apply top-p
        let filtered = if self.params.top_p < 1.0 {
            self.top_p_filter(&filtered, self.params.top_p)
        } else {
            filtered
        };

        // Apply min-p
        let filtered = if self.params.min_p > 0.0 {
            self.min_p_filter(&filtered, self.params.min_p)
        } else {
            filtered
        };

        // Sample from distribution
        self.categorical_sample(&filtered)
    }

    /// Returns the index of the maximum value.
    fn argmax(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Applies top-k filtering.
    fn top_k_filter(&self, logits: &[f32], k: usize) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = vec![f32::NEG_INFINITY; logits.len()];
        for (i, v) in indexed.into_iter().take(k) {
            result[i] = v;
        }
        result
    }

    /// Applies top-p (nucleus) filtering.
    fn top_p_filter(&self, logits: &[f32], p: f32) -> Vec<f32> {
        let probs = self.softmax(logits);
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut result = vec![f32::NEG_INFINITY; logits.len()];

        for (i, prob) in indexed {
            if cumsum < p {
                result[i] = logits[i];
                cumsum += prob;
            }
        }

        result
    }

    /// Applies min-p filtering.
    fn min_p_filter(&self, logits: &[f32], min_p: f32) -> Vec<f32> {
        let probs = self.softmax(logits);
        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_prob * min_p;

        logits
            .iter()
            .zip(probs.iter())
            .map(|(&l, &p)| if p >= threshold { l } else { f32::NEG_INFINITY })
            .collect()
    }

    /// Computes softmax probabilities.
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|&e| e / sum).collect()
    }

    /// Samples from a categorical distribution.
    fn categorical_sample(&mut self, logits: &[f32]) -> u32 {
        let probs = self.softmax(logits);
        let r = self.random_f32();

        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }

        (probs.len() - 1) as u32
    }

    /// Generates a random f32 in [0, 1).
    fn random_f32(&mut self) -> f32 {
        if let Some(rng) = &mut self.rng {
            rng.f32()
        } else {
            fastrand::f32()
        }
    }

    /// Returns the sampling parameters.
    #[must_use]
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Checks if a token matches any stop sequence.
    #[must_use]
    pub fn is_stop_token(&self, text: &str) -> bool {
        self.params
            .stop_sequences
            .iter()
            .any(|stop| text.contains(stop))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Greedy Sampling Tests ===

    #[test]
    fn test_greedy_sampling() {
        let params = SamplingParams::greedy();
        let mut sampler = Sampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 0.5];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_greedy_sampling_first_element() {
        let params = SamplingParams::greedy();
        let mut sampler = Sampler::new(params);

        let logits = vec![10.0, 5.0, 2.0, 0.5];
        assert_eq!(sampler.sample(&logits), 0);
    }

    #[test]
    fn test_greedy_sampling_last_element() {
        let params = SamplingParams::greedy();
        let mut sampler = Sampler::new(params);

        let logits = vec![1.0, 2.0, 3.0, 10.0];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_greedy_sampling_negative() {
        let params = SamplingParams::greedy();
        let mut sampler = Sampler::new(params);

        let logits = vec![-1.0, -5.0, -2.0, -0.5];
        assert_eq!(sampler.sample(&logits), 3); // -0.5 is highest
    }

    // === Deterministic Sampling Tests ===

    #[test]
    fn test_deterministic_with_seed() {
        let params = SamplingParams::balanced().with_seed(42);
        let mut sampler1 = Sampler::new(params.clone());
        let mut sampler2 = Sampler::new(params);

        let logits = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(sampler1.sample(&logits), sampler2.sample(&logits));
    }

    #[test]
    fn test_deterministic_multiple_samples() {
        let params = SamplingParams::balanced().with_seed(12345);
        let mut sampler1 = Sampler::new(params.clone());
        let mut sampler2 = Sampler::new(params);

        let logits = vec![1.0, 1.0, 1.0, 1.0];

        // Multiple samples should match
        for _ in 0..5 {
            assert_eq!(sampler1.sample(&logits), sampler2.sample(&logits));
        }
    }

    // === Argmax Tests ===

    #[test]
    fn test_argmax_simple() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        assert_eq!(sampler.argmax(&[1.0, 5.0, 2.0]), 1);
    }

    #[test]
    fn test_argmax_first() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        assert_eq!(sampler.argmax(&[10.0, 5.0, 2.0]), 0);
    }

    #[test]
    fn test_argmax_last() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        assert_eq!(sampler.argmax(&[1.0, 2.0, 10.0]), 2);
    }

    #[test]
    fn test_argmax_empty() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        assert_eq!(sampler.argmax(&[]), 0);
    }

    #[test]
    fn test_argmax_single() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        assert_eq!(sampler.argmax(&[42.0]), 0);
    }

    // === Top-K Filter Tests ===

    #[test]
    fn test_top_k_filter() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 4.0];
        let filtered = sampler.top_k_filter(&logits, 2);

        // Only top 2 should be kept (indices 1 and 3)
        assert!(filtered[1].is_finite());
        assert!(filtered[3].is_finite());
        assert!(filtered[0].is_infinite() && filtered[0] < 0.0);
        assert!(filtered[2].is_infinite() && filtered[2] < 0.0);
    }

    #[test]
    fn test_top_k_filter_k_equals_length() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![1.0, 2.0, 3.0];
        let filtered = sampler.top_k_filter(&logits, 3);

        // All should be kept
        assert!(filtered[0].is_finite());
        assert!(filtered[1].is_finite());
        assert!(filtered[2].is_finite());
    }

    #[test]
    fn test_top_k_filter_k_greater_than_length() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![1.0, 2.0];
        let filtered = sampler.top_k_filter(&logits, 10);

        // All should be kept
        assert!(filtered[0].is_finite());
        assert!(filtered[1].is_finite());
    }

    // === Softmax Tests ===

    #[test]
    fn test_softmax_uniform() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let probs = sampler.softmax(&logits);

        // Should be uniform distribution
        for p in probs {
            assert!((p - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = sampler.softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_order_preserved() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        let logits = vec![1.0, 3.0, 2.0];
        let probs = sampler.softmax(&logits);

        // Higher logit -> higher probability
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    // === Stop Token Tests ===

    #[test]
    fn test_is_stop_token_match() {
        let params = SamplingParams::default().with_stop("<|end|>");
        let sampler = Sampler::new(params);

        assert!(sampler.is_stop_token("Some text<|end|>"));
    }

    #[test]
    fn test_is_stop_token_no_match() {
        let params = SamplingParams::default().with_stop("<|end|>");
        let sampler = Sampler::new(params);

        assert!(!sampler.is_stop_token("Some text without stop"));
    }

    #[test]
    fn test_is_stop_token_multiple() {
        let params = SamplingParams::default()
            .with_stop("<|end|>")
            .with_stop("STOP");
        let sampler = Sampler::new(params);

        assert!(sampler.is_stop_token("Text STOP here"));
        assert!(sampler.is_stop_token("Text<|end|>here"));
        assert!(!sampler.is_stop_token("No stop sequence"));
    }

    #[test]
    fn test_is_stop_token_empty_sequences() {
        let params = SamplingParams::default();
        let sampler = Sampler::new(params);

        assert!(!sampler.is_stop_token("Any text"));
    }

    // === Params Accessor Tests ===

    #[test]
    fn test_params_accessor() {
        let params = SamplingParams::default().with_max_tokens(100);
        let sampler = Sampler::new(params);

        assert_eq!(sampler.params().max_tokens, 100);
    }

    #[test]
    fn test_params_temperature() {
        let params = SamplingParams::default().with_temperature(0.7);
        let sampler = Sampler::new(params);

        assert!((sampler.params().temperature - 0.7).abs() < 1e-5);
    }

    // === Temperature Sampling Tests ===

    #[test]
    fn test_temperature_zero_is_greedy() {
        let params = SamplingParams::default().with_temperature(0.0);
        let mut sampler = Sampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 0.5];
        // Temperature 0 should always pick the max
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_high_temperature_more_random() {
        // With very high temperature, distribution should be more uniform
        // Hard to test randomness, but we can verify it doesn't crash
        let params = SamplingParams::default()
            .with_temperature(2.0)
            .with_seed(42);
        let mut sampler = Sampler::new(params);

        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let _ = sampler.sample(&logits);
    }

    // === Min-P Filter Tests ===

    #[test]
    fn test_min_p_filter() {
        let params = SamplingParams::greedy();
        let sampler = Sampler::new(params);

        // Create logits where one is clearly dominant
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let filtered = sampler.min_p_filter(&logits, 0.1);

        // The dominant one should be kept
        assert!(filtered[0].is_finite());
    }

    // === Sampler Creation Tests ===

    #[test]
    fn test_sampler_without_seed() {
        let params = SamplingParams::default();
        let sampler = Sampler::new(params);

        // Should have no seeded RNG
        assert!(sampler.rng.is_none());
    }

    #[test]
    fn test_sampler_with_seed() {
        let params = SamplingParams::default().with_seed(42);
        let sampler = Sampler::new(params);

        // Should have seeded RNG
        assert!(sampler.rng.is_some());
    }
}
