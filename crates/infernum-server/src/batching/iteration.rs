//! Token-level iteration for continuous batching.
//!
//! This module provides the token-by-token iteration logic that enables
//! continuous batching, allowing new requests to be added mid-generation.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::batch::{ActiveBatch, FinishReason, SequenceState};

/// Configuration for iteration behavior.
#[derive(Debug, Clone)]
pub struct IterationConfig {
    /// Maximum tokens to process per iteration.
    pub max_tokens_per_step: usize,

    /// Whether to allow preemption during iteration.
    pub allow_preemption: bool,

    /// Minimum tokens before checking for new requests.
    pub tokens_between_checks: usize,

    /// Whether to collect detailed metrics.
    pub collect_metrics: bool,

    /// Stop tokens that end generation.
    pub stop_tokens: Vec<u32>,
}

impl Default for IterationConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_step: 1,
            allow_preemption: true,
            tokens_between_checks: 1,
            collect_metrics: true,
            stop_tokens: vec![2], // Common EOS token
        }
    }
}

impl IterationConfig {
    /// Creates a new iteration config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method for max tokens per step.
    pub fn with_max_tokens_per_step(mut self, n: usize) -> Self {
        self.max_tokens_per_step = n;
        self
    }

    /// Builder method for preemption setting.
    pub fn with_preemption(mut self, allow: bool) -> Self {
        self.allow_preemption = allow;
        self
    }

    /// Builder method for stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }
}

/// Result of a single iteration step.
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Tokens generated in this step.
    pub tokens_generated: usize,

    /// Sequences that finished.
    pub sequences_finished: usize,

    /// Whether the batch is complete.
    pub batch_complete: bool,

    /// Whether preemption was requested.
    pub preemption_requested: bool,

    /// Time taken for this iteration.
    pub duration: Duration,

    /// Output tokens per sequence (indexed by position).
    pub outputs: Vec<Option<u32>>,
}

impl IterationResult {
    /// Creates an empty iteration result.
    pub fn empty() -> Self {
        Self {
            tokens_generated: 0,
            sequences_finished: 0,
            batch_complete: false,
            preemption_requested: false,
            duration: Duration::ZERO,
            outputs: Vec::new(),
        }
    }

    /// Creates a result indicating batch completion.
    pub fn complete(duration: Duration) -> Self {
        Self {
            tokens_generated: 0,
            sequences_finished: 0,
            batch_complete: true,
            preemption_requested: false,
            duration,
            outputs: Vec::new(),
        }
    }

    /// Creates a result indicating preemption.
    pub fn preempted(duration: Duration) -> Self {
        Self {
            tokens_generated: 0,
            sequences_finished: 0,
            batch_complete: false,
            preemption_requested: true,
            duration,
            outputs: Vec::new(),
        }
    }
}

/// A step in the iteration process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationStep {
    /// Prepare input tensors.
    PrepareInputs,
    /// Execute model forward pass.
    Forward,
    /// Sample next tokens.
    Sample,
    /// Update sequences.
    UpdateSequences,
    /// Check for completion.
    CheckCompletion,
    /// Process outputs.
    ProcessOutputs,
}

impl fmt::Display for IterationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrepareInputs => write!(f, "prepare_inputs"),
            Self::Forward => write!(f, "forward"),
            Self::Sample => write!(f, "sample"),
            Self::UpdateSequences => write!(f, "update_sequences"),
            Self::CheckCompletion => write!(f, "check_completion"),
            Self::ProcessOutputs => write!(f, "process_outputs"),
        }
    }
}

/// Iterator over tokens in a batch.
#[derive(Debug)]
pub struct TokenIterator {
    /// Configuration.
    config: IterationConfig,

    /// Current iteration number.
    iteration: u64,

    /// Tokens generated so far.
    tokens_generated: u64,

    /// Whether preemption is requested.
    preemption_requested: bool,

    /// When iteration started.
    started_at: Instant,

    /// Metrics collector.
    metrics: IterationMetrics,
}

impl TokenIterator {
    /// Creates a new token iterator.
    pub fn new(config: IterationConfig) -> Self {
        Self {
            config,
            iteration: 0,
            tokens_generated: 0,
            preemption_requested: false,
            started_at: Instant::now(),
            metrics: IterationMetrics::new(),
        }
    }

    /// Returns the current iteration number.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Returns total tokens generated.
    pub fn tokens_generated(&self) -> u64 {
        self.tokens_generated
    }

    /// Returns elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Requests preemption at the next opportunity.
    pub fn request_preemption(&mut self) {
        self.preemption_requested = true;
    }

    /// Returns true if preemption is requested.
    pub fn is_preemption_requested(&self) -> bool {
        self.preemption_requested
    }

    /// Returns the metrics.
    pub fn metrics(&self) -> &IterationMetrics {
        &self.metrics
    }

    /// Performs a single iteration step on the batch.
    pub fn step(&mut self, batch: &mut ActiveBatch) -> IterationResult {
        let step_start = Instant::now();
        self.iteration += 1;

        // Check for preemption
        if self.config.allow_preemption && self.preemption_requested {
            self.metrics.record_preemption();
            return IterationResult::preempted(step_start.elapsed());
        }

        // Check if batch is already complete
        if batch.all_finished() {
            return IterationResult::complete(step_start.elapsed());
        }

        let mut tokens_generated = 0;
        let mut sequences_finished = 0;
        let mut outputs = Vec::new();

        // Process each sequence group
        batch.for_each_group_mut(|group| {
            for seq in &mut group.sequences {
                if seq.state != SequenceState::Running && seq.state != SequenceState::Waiting {
                    outputs.push(None);
                    continue;
                }

                // Mark as running
                if seq.state == SequenceState::Waiting {
                    seq.state = SequenceState::Running;
                }

                // Simulate token generation (in real impl, this would call model)
                let next_token = self.simulate_next_token(seq.tokens.last().copied());

                // Check stop conditions
                if self.config.stop_tokens.contains(&next_token) {
                    seq.finish(FinishReason::Stop);
                    sequences_finished += 1;
                    outputs.push(Some(next_token));
                    continue;
                }

                // Check length limit
                if seq.is_at_max_length() {
                    seq.finish(FinishReason::Length);
                    sequences_finished += 1;
                    outputs.push(None);
                    continue;
                }

                // Append token
                seq.append_token(next_token, Some(-0.5)); // Simulated logprob
                tokens_generated += 1;
                outputs.push(Some(next_token));
            }
        });

        self.tokens_generated += tokens_generated as u64;
        batch.record_tokens(tokens_generated as u64);

        let duration = step_start.elapsed();
        self.metrics.record_step(duration, tokens_generated);

        IterationResult {
            tokens_generated,
            sequences_finished,
            batch_complete: batch.all_finished(),
            preemption_requested: false,
            duration,
            outputs,
        }
    }

    /// Simulates generating a next token (placeholder for real model).
    fn simulate_next_token(&self, _last_token: Option<u32>) -> u32 {
        // In real implementation, this would be replaced by actual model inference
        // For now, return a token based on iteration to simulate generation
        ((self.iteration % 1000) + 100) as u32
    }

    /// Runs until batch completion or preemption.
    pub fn run_to_completion(&mut self, batch: &mut ActiveBatch) -> IterationResult {
        let start = Instant::now();
        let mut total_tokens = 0;
        let mut total_finished = 0;

        loop {
            let result = self.step(batch);
            total_tokens += result.tokens_generated;
            total_finished += result.sequences_finished;

            if result.batch_complete || result.preemption_requested {
                return IterationResult {
                    tokens_generated: total_tokens,
                    sequences_finished: total_finished,
                    batch_complete: result.batch_complete,
                    preemption_requested: result.preemption_requested,
                    duration: start.elapsed(),
                    outputs: result.outputs,
                };
            }
        }
    }

    /// Runs for a specified number of tokens.
    pub fn run_for_tokens(
        &mut self,
        batch: &mut ActiveBatch,
        max_tokens: usize,
    ) -> IterationResult {
        let start = Instant::now();
        let mut total_tokens = 0;
        let mut total_finished = 0;
        let mut last_outputs = Vec::new();

        while total_tokens < max_tokens {
            let result = self.step(batch);
            total_tokens += result.tokens_generated;
            total_finished += result.sequences_finished;
            last_outputs = result.outputs;

            if result.batch_complete || result.preemption_requested {
                return IterationResult {
                    tokens_generated: total_tokens,
                    sequences_finished: total_finished,
                    batch_complete: result.batch_complete,
                    preemption_requested: result.preemption_requested,
                    duration: start.elapsed(),
                    outputs: last_outputs,
                };
            }
        }

        IterationResult {
            tokens_generated: total_tokens,
            sequences_finished: total_finished,
            batch_complete: false,
            preemption_requested: false,
            duration: start.elapsed(),
            outputs: last_outputs,
        }
    }

    /// Runs for a specified duration.
    pub fn run_for_duration(
        &mut self,
        batch: &mut ActiveBatch,
        max_duration: Duration,
    ) -> IterationResult {
        let start = Instant::now();
        let mut total_tokens = 0;
        let mut total_finished = 0;
        let mut last_outputs = Vec::new();

        while start.elapsed() < max_duration {
            let result = self.step(batch);
            total_tokens += result.tokens_generated;
            total_finished += result.sequences_finished;
            last_outputs = result.outputs;

            if result.batch_complete || result.preemption_requested {
                return IterationResult {
                    tokens_generated: total_tokens,
                    sequences_finished: total_finished,
                    batch_complete: result.batch_complete,
                    preemption_requested: result.preemption_requested,
                    duration: start.elapsed(),
                    outputs: last_outputs,
                };
            }
        }

        IterationResult {
            tokens_generated: total_tokens,
            sequences_finished: total_finished,
            batch_complete: false,
            preemption_requested: false,
            duration: start.elapsed(),
            outputs: last_outputs,
        }
    }

    /// Resets the iterator for a new batch.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.tokens_generated = 0;
        self.preemption_requested = false;
        self.started_at = Instant::now();
    }
}

/// Metrics for iteration performance.
#[derive(Debug)]
pub struct IterationMetrics {
    /// Total iterations performed.
    iterations: AtomicU64,

    /// Total tokens generated.
    tokens_generated: AtomicU64,

    /// Total time in nanoseconds.
    total_time_ns: AtomicU64,

    /// Preemption count.
    preemptions: AtomicU64,

    /// Step count by type (simplified: just total steps).
    total_steps: AtomicU64,
}

impl IterationMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self {
            iterations: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            preemptions: AtomicU64::new(0),
            total_steps: AtomicU64::new(0),
        }
    }

    /// Records a step.
    pub fn record_step(&self, duration: Duration, tokens: usize) {
        self.iterations.fetch_add(1, Ordering::Relaxed);
        self.tokens_generated
            .fetch_add(tokens as u64, Ordering::Relaxed);
        self.total_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.total_steps.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a preemption.
    pub fn record_preemption(&self) {
        self.preemptions.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns total iterations.
    pub fn iterations(&self) -> u64 {
        self.iterations.load(Ordering::Relaxed)
    }

    /// Returns total tokens generated.
    pub fn tokens_generated(&self) -> u64 {
        self.tokens_generated.load(Ordering::Relaxed)
    }

    /// Returns total time.
    pub fn total_time(&self) -> Duration {
        Duration::from_nanos(self.total_time_ns.load(Ordering::Relaxed))
    }

    /// Returns preemption count.
    pub fn preemptions(&self) -> u64 {
        self.preemptions.load(Ordering::Relaxed)
    }

    /// Returns tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        let tokens = self.tokens_generated() as f64;
        let secs = self.total_time().as_secs_f64();
        if secs > 0.0 {
            tokens / secs
        } else {
            0.0
        }
    }

    /// Returns average time per token.
    pub fn avg_time_per_token(&self) -> Duration {
        let tokens = self.tokens_generated();
        if tokens > 0 {
            self.total_time() / tokens as u32
        } else {
            Duration::ZERO
        }
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(
            "# HELP infernum_batch_iterations_total Total batch iterations\n",
        );
        output.push_str("# TYPE infernum_batch_iterations_total counter\n");
        output.push_str(&format!(
            "infernum_batch_iterations_total {}\n",
            self.iterations()
        ));

        output.push_str(
            "# HELP infernum_batch_tokens_generated_total Total tokens generated\n",
        );
        output.push_str("# TYPE infernum_batch_tokens_generated_total counter\n");
        output.push_str(&format!(
            "infernum_batch_tokens_generated_total {}\n",
            self.tokens_generated()
        ));

        output.push_str(
            "# HELP infernum_batch_preemptions_total Total batch preemptions\n",
        );
        output.push_str("# TYPE infernum_batch_preemptions_total counter\n");
        output.push_str(&format!(
            "infernum_batch_preemptions_total {}\n",
            self.preemptions()
        ));

        output.push_str(
            "# HELP infernum_batch_tokens_per_second Current tokens per second\n",
        );
        output.push_str("# TYPE infernum_batch_tokens_per_second gauge\n");
        output.push_str(&format!(
            "infernum_batch_tokens_per_second {:.2}\n",
            self.tokens_per_second()
        ));

        output
    }
}

impl Default for IterationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::batch::{BatchId, SamplingParams, Sequence, SequenceGroup, SequenceId};
    use super::*;

    fn create_test_batch() -> ActiveBatch {
        let batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 10);
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());
        let _ = batch.try_add(group);

        batch
    }

    #[test]
    fn test_iteration_config_default() {
        let config = IterationConfig::default();

        assert_eq!(config.max_tokens_per_step, 1);
        assert!(config.allow_preemption);
        assert!(config.collect_metrics);
    }

    #[test]
    fn test_iteration_config_builder() {
        let config = IterationConfig::new()
            .with_max_tokens_per_step(4)
            .with_preemption(false)
            .with_stop_tokens(vec![0, 2]);

        assert_eq!(config.max_tokens_per_step, 4);
        assert!(!config.allow_preemption);
        assert_eq!(config.stop_tokens, vec![0, 2]);
    }

    #[test]
    fn test_iteration_result_empty() {
        let result = IterationResult::empty();

        assert_eq!(result.tokens_generated, 0);
        assert_eq!(result.sequences_finished, 0);
        assert!(!result.batch_complete);
        assert!(!result.preemption_requested);
    }

    #[test]
    fn test_iteration_result_complete() {
        let result = IterationResult::complete(Duration::from_millis(100));

        assert!(result.batch_complete);
        assert_eq!(result.duration.as_millis(), 100);
    }

    #[test]
    fn test_iteration_result_preempted() {
        let result = IterationResult::preempted(Duration::from_millis(50));

        assert!(result.preemption_requested);
        assert!(!result.batch_complete);
    }

    #[test]
    fn test_iteration_step_display() {
        assert_eq!(IterationStep::PrepareInputs.to_string(), "prepare_inputs");
        assert_eq!(IterationStep::Forward.to_string(), "forward");
        assert_eq!(IterationStep::Sample.to_string(), "sample");
    }

    #[test]
    fn test_token_iterator_new() {
        let config = IterationConfig::default();
        let iterator = TokenIterator::new(config);

        assert_eq!(iterator.iteration(), 0);
        assert_eq!(iterator.tokens_generated(), 0);
        assert!(!iterator.is_preemption_requested());
    }

    #[test]
    fn test_token_iterator_step() {
        let config = IterationConfig::default();
        let mut iterator = TokenIterator::new(config);
        let mut batch = create_test_batch();

        let result = iterator.step(&mut batch);

        assert_eq!(iterator.iteration(), 1);
        assert!(result.tokens_generated > 0 || result.sequences_finished > 0);
    }

    #[test]
    fn test_token_iterator_preemption() {
        let config = IterationConfig::default();
        let mut iterator = TokenIterator::new(config);
        let mut batch = create_test_batch();

        iterator.request_preemption();
        assert!(iterator.is_preemption_requested());

        let result = iterator.step(&mut batch);
        assert!(result.preemption_requested);
    }

    #[test]
    fn test_token_iterator_run_for_tokens() {
        let config = IterationConfig::new().with_stop_tokens(vec![]); // No stop tokens
        let mut iterator = TokenIterator::new(config);
        let mut batch = create_test_batch();

        let result = iterator.run_for_tokens(&mut batch, 5);

        assert!(result.tokens_generated >= 5 || result.batch_complete);
    }

    #[test]
    fn test_token_iterator_run_for_duration() {
        let config = IterationConfig::new().with_stop_tokens(vec![]);
        let mut iterator = TokenIterator::new(config);
        let mut batch = create_test_batch();

        let result = iterator.run_for_duration(&mut batch, Duration::from_millis(10));

        assert!(result.tokens_generated > 0 || result.batch_complete);
    }

    #[test]
    fn test_token_iterator_reset() {
        let config = IterationConfig::default();
        let mut iterator = TokenIterator::new(config);
        let mut batch = create_test_batch();

        let _ = iterator.step(&mut batch);
        assert!(iterator.iteration() > 0);

        iterator.reset();
        assert_eq!(iterator.iteration(), 0);
        assert_eq!(iterator.tokens_generated(), 0);
    }

    #[test]
    fn test_iteration_metrics_new() {
        let metrics = IterationMetrics::new();

        assert_eq!(metrics.iterations(), 0);
        assert_eq!(metrics.tokens_generated(), 0);
        assert_eq!(metrics.preemptions(), 0);
    }

    #[test]
    fn test_iteration_metrics_record() {
        let metrics = IterationMetrics::new();

        metrics.record_step(Duration::from_millis(10), 5);
        metrics.record_step(Duration::from_millis(15), 3);

        assert_eq!(metrics.iterations(), 2);
        assert_eq!(metrics.tokens_generated(), 8);
        assert_eq!(metrics.total_time().as_millis(), 25);
    }

    #[test]
    fn test_iteration_metrics_preemption() {
        let metrics = IterationMetrics::new();

        metrics.record_preemption();
        metrics.record_preemption();

        assert_eq!(metrics.preemptions(), 2);
    }

    #[test]
    fn test_iteration_metrics_tokens_per_second() {
        let metrics = IterationMetrics::new();

        metrics.record_step(Duration::from_secs(1), 100);

        // Should be approximately 100 tokens/sec
        let tps = metrics.tokens_per_second();
        assert!(tps > 90.0 && tps < 110.0);
    }

    #[test]
    fn test_iteration_metrics_prometheus() {
        let metrics = IterationMetrics::new();
        metrics.record_step(Duration::from_millis(10), 5);

        let output = metrics.prometheus();

        assert!(output.contains("infernum_batch_iterations_total 1"));
        assert!(output.contains("infernum_batch_tokens_generated_total 5"));
        assert!(output.contains("infernum_batch_preemptions_total 0"));
    }

    #[test]
    fn test_batch_completion() {
        // Create a batch with a sequence that will finish quickly
        let mut batch = ActiveBatch::new(BatchId::new(1), 32, 4096);
        let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 2); // Only 2 tokens to generate
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());
        let _ = batch.try_add(group);

        let config = IterationConfig::new().with_stop_tokens(vec![]); // No stop tokens
        let mut iterator = TokenIterator::new(config);

        let result = iterator.run_to_completion(&mut batch);

        assert!(result.batch_complete);
    }
}
