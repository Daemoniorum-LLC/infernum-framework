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

    #[test]
    fn test_greedy_sampling() {
        let params = SamplingParams::greedy();
        let mut sampler = Sampler::new(params);

        let logits = vec![1.0, 5.0, 2.0, 0.5];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let params = SamplingParams::balanced().with_seed(42);
        let mut sampler1 = Sampler::new(params.clone());
        let mut sampler2 = Sampler::new(params);

        let logits = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(sampler1.sample(&logits), sampler2.sample(&logits));
    }
}
