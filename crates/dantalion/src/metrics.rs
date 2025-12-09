//! Metrics collection for inference performance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::TelemetryConfig;

/// Collector for inference metrics.
pub struct MetricsCollector {
    inference: InferenceMetrics,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    #[must_use]
    pub fn new(_config: &TelemetryConfig) -> Self {
        Self {
            inference: InferenceMetrics::default(),
        }
    }

    /// Returns the inference metrics.
    #[must_use]
    pub fn inference(&self) -> &InferenceMetrics {
        &self.inference
    }
}

/// Metrics specific to LLM inference.
#[derive(Default)]
pub struct InferenceMetrics {
    /// Total requests processed.
    pub total_requests: AtomicU64,
    /// Total tokens generated.
    pub total_tokens_generated: AtomicU64,
    /// Total prompt tokens processed.
    pub total_prompt_tokens: AtomicU64,
    /// Total errors.
    pub total_errors: AtomicU64,
}

impl InferenceMetrics {
    /// Records a completed request.
    pub fn record_request(&self, prompt_tokens: u32, completion_tokens: u32) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(u64::from(prompt_tokens), Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(u64::from(completion_tokens), Ordering::Relaxed);
    }

    /// Records an error.
    pub fn record_error(&self) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the total number of requests.
    #[must_use]
    pub fn requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Returns the total tokens generated.
    #[must_use]
    pub fn tokens_generated(&self) -> u64 {
        self.total_tokens_generated.load(Ordering::Relaxed)
    }
}

/// Timer for measuring operation duration.
pub struct Timer {
    start: Instant,
    label: &'static str,
}

impl Timer {
    /// Starts a new timer.
    #[must_use]
    pub fn start(label: &'static str) -> Self {
        Self {
            start: Instant::now(),
            label,
        }
    }

    /// Returns the elapsed duration.
    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Stops the timer and logs the duration.
    pub fn stop(self) {
        let elapsed = self.elapsed_ms();
        tracing::debug!(label = self.label, elapsed_ms = elapsed, "Timer stopped");
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // Optionally log on drop
    }
}
