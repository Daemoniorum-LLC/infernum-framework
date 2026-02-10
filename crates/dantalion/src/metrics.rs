//! Metrics collection for inference performance.
//!
//! Provides Prometheus-compatible metrics for monitoring LLM inference workloads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use crate::TelemetryConfig;

/// Bucket boundaries for latency histograms (in seconds).
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Bucket boundaries for token count histograms.
const TOKEN_BUCKETS: &[f64] = &[
    1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
];

/// Collector for inference metrics.
pub struct MetricsCollector {
    inference: InferenceMetrics,
    prometheus: PrometheusRegistry,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    #[must_use]
    pub fn new(_config: &TelemetryConfig) -> Self {
        Self {
            inference: InferenceMetrics::default(),
            prometheus: PrometheusRegistry::new(),
        }
    }

    /// Returns the inference metrics.
    #[must_use]
    pub fn inference(&self) -> &InferenceMetrics {
        &self.inference
    }

    /// Returns the Prometheus registry.
    #[must_use]
    pub fn prometheus(&self) -> &PrometheusRegistry {
        &self.prometheus
    }

    /// Records a completed chat request with timing.
    pub fn record_chat_request(
        &self,
        prompt_tokens: u32,
        completion_tokens: u32,
        latency_secs: f64,
        model: &str,
    ) {
        self.inference
            .record_request(prompt_tokens, completion_tokens);
        self.prometheus.record_request("chat", model);
        self.prometheus.record_latency("chat", model, latency_secs);
        self.prometheus
            .record_tokens("prompt", model, prompt_tokens);
        self.prometheus
            .record_tokens("completion", model, completion_tokens);
    }

    /// Records a completed completion request with timing.
    pub fn record_completion_request(
        &self,
        prompt_tokens: u32,
        completion_tokens: u32,
        latency_secs: f64,
        model: &str,
    ) {
        self.inference
            .record_request(prompt_tokens, completion_tokens);
        self.prometheus.record_request("completion", model);
        self.prometheus
            .record_latency("completion", model, latency_secs);
        self.prometheus
            .record_tokens("prompt", model, prompt_tokens);
        self.prometheus
            .record_tokens("completion", model, completion_tokens);
    }

    /// Records a completed embedding request with timing.
    pub fn record_embedding_request(
        &self,
        tokens: u32,
        latency_secs: f64,
        model: &str,
        batch_size: usize,
    ) {
        self.inference.record_request(tokens, 0);
        self.prometheus.record_request("embedding", model);
        self.prometheus
            .record_latency("embedding", model, latency_secs);
        self.prometheus.record_tokens("embedding", model, tokens);
        self.prometheus.record_embedding_batch(model, batch_size);
    }

    /// Records an error.
    pub fn record_error(&self, endpoint: &str, model: &str, error_type: &str) {
        self.inference.record_error();
        self.prometheus.record_error(endpoint, model, error_type);
    }

    /// Renders metrics in Prometheus text exposition format.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        self.prometheus.render()
    }
}

/// Prometheus-compatible metric registry.
pub struct PrometheusRegistry {
    /// Request counters by endpoint and model.
    requests: RwLock<HashMap<(String, String), u64>>,
    /// Error counters by endpoint, model, and error type.
    errors: RwLock<HashMap<(String, String, String), u64>>,
    /// Latency histogram data.
    latency_histograms: RwLock<HashMap<(String, String), HistogramData>>,
    /// Token count histograms.
    token_histograms: RwLock<HashMap<(String, String), HistogramData>>,
    /// Embedding batch size histograms.
    embedding_batches: RwLock<HashMap<String, HistogramData>>,
    /// Active requests gauge.
    active_requests: AtomicU64,
    /// Model loaded gauge (1 = loaded, 0 = not loaded).
    model_loaded: AtomicU64,
}

impl PrometheusRegistry {
    /// Creates a new registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            errors: RwLock::new(HashMap::new()),
            latency_histograms: RwLock::new(HashMap::new()),
            token_histograms: RwLock::new(HashMap::new()),
            embedding_batches: RwLock::new(HashMap::new()),
            active_requests: AtomicU64::new(0),
            model_loaded: AtomicU64::new(0),
        }
    }

    /// Increments active request count.
    pub fn inc_active_requests(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements active request count.
    pub fn dec_active_requests(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Sets model loaded state.
    pub fn set_model_loaded(&self, loaded: bool) {
        self.model_loaded
            .store(if loaded { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Records a request.
    fn record_request(&self, endpoint: &str, model: &str) {
        let key = (endpoint.to_string(), model.to_string());
        let mut requests = self.requests.write();
        *requests.entry(key).or_insert(0) += 1;
    }

    /// Records an error.
    fn record_error(&self, endpoint: &str, model: &str, error_type: &str) {
        let key = (
            endpoint.to_string(),
            model.to_string(),
            error_type.to_string(),
        );
        let mut errors = self.errors.write();
        *errors.entry(key).or_insert(0) += 1;
    }

    /// Records latency.
    fn record_latency(&self, endpoint: &str, model: &str, latency_secs: f64) {
        let key = (endpoint.to_string(), model.to_string());
        let mut histograms = self.latency_histograms.write();
        let histogram = histograms
            .entry(key)
            .or_insert_with(|| HistogramData::new(LATENCY_BUCKETS));
        histogram.observe(latency_secs);
    }

    /// Records token counts.
    fn record_tokens(&self, token_type: &str, model: &str, count: u32) {
        let key = (token_type.to_string(), model.to_string());
        let mut histograms = self.token_histograms.write();
        let histogram = histograms
            .entry(key)
            .or_insert_with(|| HistogramData::new(TOKEN_BUCKETS));
        histogram.observe(f64::from(count));
    }

    /// Records embedding batch size.
    fn record_embedding_batch(&self, model: &str, batch_size: usize) {
        let mut batches = self.embedding_batches.write();
        let histogram = batches
            .entry(model.to_string())
            .or_insert_with(|| HistogramData::new(&[1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 256.0]));
        histogram.observe(batch_size as f64);
    }

    /// Renders all metrics in Prometheus text format.
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = String::new();

        // Requests total
        output.push_str("# HELP infernum_requests_total Total number of inference requests.\n");
        output.push_str("# TYPE infernum_requests_total counter\n");
        for ((endpoint, model), count) in self.requests.read().iter() {
            output.push_str(&format!(
                "infernum_requests_total{{endpoint=\"{}\",model=\"{}\"}} {}\n",
                endpoint, model, count
            ));
        }

        // Errors total
        output.push_str("# HELP infernum_errors_total Total number of errors.\n");
        output.push_str("# TYPE infernum_errors_total counter\n");
        for ((endpoint, model, error_type), count) in self.errors.read().iter() {
            output.push_str(&format!(
                "infernum_errors_total{{endpoint=\"{}\",model=\"{}\",error_type=\"{}\"}} {}\n",
                endpoint, model, error_type, count
            ));
        }

        // Active requests gauge
        output.push_str(
            "# HELP infernum_active_requests Number of requests currently being processed.\n",
        );
        output.push_str("# TYPE infernum_active_requests gauge\n");
        output.push_str(&format!(
            "infernum_active_requests {}\n",
            self.active_requests.load(Ordering::Relaxed)
        ));

        // Model loaded gauge
        output.push_str(
            "# HELP infernum_model_loaded Whether a model is currently loaded (1) or not (0).\n",
        );
        output.push_str("# TYPE infernum_model_loaded gauge\n");
        output.push_str(&format!(
            "infernum_model_loaded {}\n",
            self.model_loaded.load(Ordering::Relaxed)
        ));

        // Latency histograms
        output.push_str("# HELP infernum_request_duration_seconds Request latency in seconds.\n");
        output.push_str("# TYPE infernum_request_duration_seconds histogram\n");
        for ((endpoint, model), histogram) in self.latency_histograms.read().iter() {
            output.push_str(&histogram.render_prometheus(
                "infernum_request_duration_seconds",
                &[("endpoint", endpoint), ("model", model)],
            ));
        }

        // Token histograms
        output.push_str("# HELP infernum_tokens Token counts per request.\n");
        output.push_str("# TYPE infernum_tokens histogram\n");
        for ((token_type, model), histogram) in self.token_histograms.read().iter() {
            output.push_str(
                &histogram.render_prometheus(
                    "infernum_tokens",
                    &[("type", token_type), ("model", model)],
                ),
            );
        }

        // Embedding batch sizes
        output.push_str(
            "# HELP infernum_embedding_batch_size Number of inputs per embedding request.\n",
        );
        output.push_str("# TYPE infernum_embedding_batch_size histogram\n");
        for (model, histogram) in self.embedding_batches.read().iter() {
            output.push_str(
                &histogram.render_prometheus("infernum_embedding_batch_size", &[("model", model)]),
            );
        }

        output
    }
}

impl Default for PrometheusRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram data for a single metric.
#[derive(Clone)]
struct HistogramData {
    /// Bucket boundaries.
    buckets: Vec<f64>,
    /// Count per bucket.
    bucket_counts: Vec<u64>,
    /// Total count of observations.
    count: u64,
    /// Sum of all observations.
    sum: f64,
}

impl HistogramData {
    /// Creates a new histogram with the given bucket boundaries.
    fn new(buckets: &[f64]) -> Self {
        Self {
            buckets: buckets.to_vec(),
            bucket_counts: vec![0; buckets.len()],
            count: 0,
            sum: 0.0,
        }
    }

    /// Records an observation.
    fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;

        // Find the bucket this observation belongs to and increment it
        // Each bucket counts observations in range (prev_boundary, boundary]
        for (i, &boundary) in self.buckets.iter().enumerate() {
            if value <= boundary {
                self.bucket_counts[i] += 1;
                return; // Only increment one bucket
            }
        }
        // If value exceeds all buckets, it only contributes to +Inf (handled by count)
    }

    /// Renders the histogram in Prometheus format.
    fn render_prometheus(&self, name: &str, labels: &[(&str, &str)]) -> String {
        let mut output = String::new();
        let label_str = labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect::<Vec<_>>()
            .join(",");

        // Cumulative bucket counts
        let mut cumulative = 0u64;
        for (i, &boundary) in self.buckets.iter().enumerate() {
            cumulative += self.bucket_counts[i];
            let bucket_labels = if label_str.is_empty() {
                format!("le=\"{}\"", boundary)
            } else {
                format!("{},le=\"{}\"", label_str, boundary)
            };
            output.push_str(&format!(
                "{}_bucket{{{}}} {}\n",
                name, bucket_labels, cumulative
            ));
        }

        // +Inf bucket
        let inf_labels = if label_str.is_empty() {
            "le=\"+Inf\"".to_string()
        } else {
            format!("{},le=\"+Inf\"", label_str)
        };
        output.push_str(&format!(
            "{}_bucket{{{}}} {}\n",
            name, inf_labels, self.count
        ));

        // Sum and count
        let sum_labels = if label_str.is_empty() {
            String::new()
        } else {
            format!("{{{}}}", label_str)
        };
        output.push_str(&format!("{}_sum{} {}\n", name, sum_labels, self.sum));
        output.push_str(&format!("{}_count{} {}\n", name, sum_labels, self.count));

        output
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

    /// Returns the total prompt tokens.
    #[must_use]
    pub fn prompt_tokens(&self) -> u64 {
        self.total_prompt_tokens.load(Ordering::Relaxed)
    }

    /// Returns the total errors.
    #[must_use]
    pub fn errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
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

    /// Returns the elapsed duration in seconds.
    #[must_use]
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Returns the elapsed duration in milliseconds.
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

/// Guard that decrements active requests when dropped.
pub struct ActiveRequestGuard<'a> {
    registry: &'a PrometheusRegistry,
}

impl<'a> ActiveRequestGuard<'a> {
    /// Creates a new guard that increments active requests.
    pub fn new(registry: &'a PrometheusRegistry) -> Self {
        registry.inc_active_requests();
        Self { registry }
    }
}

impl<'a> Drop for ActiveRequestGuard<'a> {
    fn drop(&mut self) {
        self.registry.dec_active_requests();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_observe() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);

        histogram.observe(0.5); // goes to bucket 0 (<=1.0)
        histogram.observe(3.0); // goes to bucket 1 (<=5.0)
        histogram.observe(7.0); // goes to bucket 2 (<=10.0)
        histogram.observe(15.0); // exceeds all buckets (only in +Inf)

        assert_eq!(histogram.count, 4);
        assert!((histogram.sum - 25.5).abs() < 0.001);
        // Per-bucket counts (non-cumulative)
        assert_eq!(histogram.bucket_counts[0], 1); // observations <= 1.0
        assert_eq!(histogram.bucket_counts[1], 1); // observations (1.0, 5.0]
        assert_eq!(histogram.bucket_counts[2], 1); // observations (5.0, 10.0]
                                                   // Note: 15.0 exceeds all buckets, so it's only counted in +Inf (handled by count)
    }

    #[test]
    fn test_histogram_render_prometheus() {
        let mut histogram = HistogramData::new(&[1.0, 5.0]);
        histogram.observe(0.5);
        histogram.observe(3.0);

        let output = histogram.render_prometheus("test_metric", &[("label", "value")]);

        assert!(output.contains("test_metric_bucket{label=\"value\",le=\"1\"} 1"));
        assert!(output.contains("test_metric_bucket{label=\"value\",le=\"5\"} 2"));
        assert!(output.contains("test_metric_bucket{label=\"value\",le=\"+Inf\"} 2"));
        assert!(output.contains("test_metric_sum{label=\"value\"} 3.5"));
        assert!(output.contains("test_metric_count{label=\"value\"} 2"));
    }

    #[test]
    fn test_prometheus_registry_requests() {
        let registry = PrometheusRegistry::new();

        registry.record_request("chat", "llama-3");
        registry.record_request("chat", "llama-3");
        registry.record_request("completion", "llama-3");

        let requests = registry.requests.read();
        assert_eq!(
            *requests
                .get(&("chat".to_string(), "llama-3".to_string()))
                .unwrap(),
            2
        );
        assert_eq!(
            *requests
                .get(&("completion".to_string(), "llama-3".to_string()))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_prometheus_registry_errors() {
        let registry = PrometheusRegistry::new();

        registry.record_error("chat", "llama-3", "timeout");
        registry.record_error("chat", "llama-3", "timeout");
        registry.record_error("chat", "llama-3", "internal");

        let errors = registry.errors.read();
        assert_eq!(
            *errors
                .get(&(
                    "chat".to_string(),
                    "llama-3".to_string(),
                    "timeout".to_string()
                ))
                .unwrap(),
            2
        );
        assert_eq!(
            *errors
                .get(&(
                    "chat".to_string(),
                    "llama-3".to_string(),
                    "internal".to_string()
                ))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_prometheus_registry_active_requests() {
        let registry = PrometheusRegistry::new();

        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);

        registry.inc_active_requests();
        registry.inc_active_requests();
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 2);

        registry.dec_active_requests();
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_prometheus_registry_model_loaded() {
        let registry = PrometheusRegistry::new();

        assert_eq!(registry.model_loaded.load(Ordering::Relaxed), 0);

        registry.set_model_loaded(true);
        assert_eq!(registry.model_loaded.load(Ordering::Relaxed), 1);

        registry.set_model_loaded(false);
        assert_eq!(registry.model_loaded.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_prometheus_registry_render() {
        let registry = PrometheusRegistry::new();

        registry.record_request("chat", "test-model");
        registry.record_latency("chat", "test-model", 0.5);
        registry.record_error("chat", "test-model", "timeout");
        registry.set_model_loaded(true);

        let output = registry.render();

        assert!(output.contains("# HELP infernum_requests_total"));
        assert!(output.contains("# TYPE infernum_requests_total counter"));
        assert!(
            output.contains("infernum_requests_total{endpoint=\"chat\",model=\"test-model\"} 1")
        );

        assert!(output.contains("# HELP infernum_errors_total"));
        assert!(output.contains("infernum_errors_total{endpoint=\"chat\",model=\"test-model\",error_type=\"timeout\"} 1"));

        assert!(output.contains("# HELP infernum_active_requests"));
        assert!(output.contains("infernum_active_requests 0"));

        assert!(output.contains("# HELP infernum_model_loaded"));
        assert!(output.contains("infernum_model_loaded 1"));

        assert!(output.contains("# HELP infernum_request_duration_seconds"));
        assert!(output.contains("infernum_request_duration_seconds_bucket"));
    }

    #[test]
    fn test_metrics_collector_chat_request() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_chat_request(100, 50, 0.5, "test-model");

        assert_eq!(collector.inference().requests(), 1);
        assert_eq!(collector.inference().prompt_tokens(), 100);
        assert_eq!(collector.inference().tokens_generated(), 50);
    }

    #[test]
    fn test_metrics_collector_error() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_error("chat", "test-model", "timeout");

        assert_eq!(collector.inference().errors(), 1);
    }

    #[test]
    fn test_active_request_guard() {
        let registry = PrometheusRegistry::new();

        {
            let _guard = ActiveRequestGuard::new(&registry);
            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);

            {
                let _guard2 = ActiveRequestGuard::new(&registry);
                assert_eq!(registry.active_requests.load(Ordering::Relaxed), 2);
            }

            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
        }

        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed_ms = timer.elapsed_ms();
        let elapsed_secs = timer.elapsed_secs();

        assert!(elapsed_ms >= 10.0);
        assert!(elapsed_secs >= 0.01);
    }

    #[test]
    fn test_inference_metrics_default() {
        let metrics = InferenceMetrics::default();
        assert_eq!(metrics.requests(), 0);
        assert_eq!(metrics.tokens_generated(), 0);
        assert_eq!(metrics.prompt_tokens(), 0);
        assert_eq!(metrics.errors(), 0);
    }

    // === Additional HistogramData Tests ===

    #[test]
    fn test_histogram_new() {
        let histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        assert_eq!(histogram.buckets, vec![1.0, 5.0, 10.0]);
        assert_eq!(histogram.bucket_counts, vec![0, 0, 0]);
        assert_eq!(histogram.count, 0);
        assert!((histogram.sum - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_empty_buckets() {
        let histogram = HistogramData::new(&[]);
        assert!(histogram.buckets.is_empty());
        assert!(histogram.bucket_counts.is_empty());
    }

    #[test]
    fn test_histogram_observe_at_boundary() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(1.0); // exactly at first boundary
        histogram.observe(5.0); // exactly at second boundary
        histogram.observe(10.0); // exactly at third boundary

        assert_eq!(histogram.count, 3);
        assert_eq!(histogram.bucket_counts[0], 1);
        assert_eq!(histogram.bucket_counts[1], 1);
        assert_eq!(histogram.bucket_counts[2], 1);
    }

    #[test]
    fn test_histogram_observe_negative_value() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(-1.0); // negative value should go to first bucket

        assert_eq!(histogram.count, 1);
        assert_eq!(histogram.bucket_counts[0], 1);
        assert!((histogram.sum - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_observe_zero() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(0.0);

        assert_eq!(histogram.count, 1);
        assert_eq!(histogram.bucket_counts[0], 1);
    }

    #[test]
    fn test_histogram_observe_large_value() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(1000.0); // exceeds all buckets

        assert_eq!(histogram.count, 1);
        // All bucket_counts should remain 0
        assert_eq!(histogram.bucket_counts[0], 0);
        assert_eq!(histogram.bucket_counts[1], 0);
        assert_eq!(histogram.bucket_counts[2], 0);
    }

    #[test]
    fn test_histogram_multiple_observations_same_bucket() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(0.1);
        histogram.observe(0.2);
        histogram.observe(0.5);
        histogram.observe(0.9);

        assert_eq!(histogram.count, 4);
        assert_eq!(histogram.bucket_counts[0], 4);
        assert!((histogram.sum - 1.7).abs() < 0.001);
    }

    #[test]
    fn test_histogram_clone() {
        let mut histogram = HistogramData::new(&[1.0, 5.0]);
        histogram.observe(0.5);
        histogram.observe(3.0);

        let cloned = histogram.clone();
        assert_eq!(cloned.count, histogram.count);
        assert!((cloned.sum - histogram.sum).abs() < f64::EPSILON);
        assert_eq!(cloned.bucket_counts, histogram.bucket_counts);
    }

    #[test]
    fn test_histogram_render_prometheus_no_labels() {
        let mut histogram = HistogramData::new(&[1.0, 5.0]);
        histogram.observe(0.5);

        let output = histogram.render_prometheus("test_metric", &[]);

        assert!(output.contains("test_metric_bucket{le=\"1\"} 1"));
        assert!(output.contains("test_metric_bucket{le=\"5\"} 1"));
        assert!(output.contains("test_metric_bucket{le=\"+Inf\"} 1"));
        assert!(output.contains("test_metric_sum 0.5"));
        assert!(output.contains("test_metric_count 1"));
    }

    #[test]
    fn test_histogram_render_prometheus_multiple_labels() {
        let mut histogram = HistogramData::new(&[1.0]);
        histogram.observe(0.5);

        let output = histogram.render_prometheus("test", &[("a", "1"), ("b", "2")]);

        assert!(output.contains("test_bucket{a=\"1\",b=\"2\",le=\"1\"} 1"));
        assert!(output.contains("test_sum{a=\"1\",b=\"2\"} 0.5"));
    }

    #[test]
    fn test_histogram_cumulative_counts() {
        let mut histogram = HistogramData::new(&[1.0, 5.0, 10.0]);
        histogram.observe(0.5); // bucket 0
        histogram.observe(0.7); // bucket 0
        histogram.observe(3.0); // bucket 1
        histogram.observe(7.0); // bucket 2

        let output = histogram.render_prometheus("test", &[]);

        // Cumulative counts: bucket 0 = 2, bucket 1 = 2+1=3, bucket 2 = 3+1=4
        assert!(output.contains("test_bucket{le=\"1\"} 2"));
        assert!(output.contains("test_bucket{le=\"5\"} 3"));
        assert!(output.contains("test_bucket{le=\"10\"} 4"));
        assert!(output.contains("test_bucket{le=\"+Inf\"} 4"));
    }

    // === Additional PrometheusRegistry Tests ===

    #[test]
    fn test_prometheus_registry_default() {
        let registry = PrometheusRegistry::default();
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);
        assert_eq!(registry.model_loaded.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_prometheus_registry_latency() {
        let registry = PrometheusRegistry::new();
        registry.record_latency("chat", "model", 0.5);
        registry.record_latency("chat", "model", 1.5);

        let histograms = registry.latency_histograms.read();
        let histogram = histograms
            .get(&("chat".to_string(), "model".to_string()))
            .unwrap();
        assert_eq!(histogram.count, 2);
        assert!((histogram.sum - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_prometheus_registry_tokens() {
        let registry = PrometheusRegistry::new();
        registry.record_tokens("prompt", "model", 100);
        registry.record_tokens("prompt", "model", 200);
        registry.record_tokens("completion", "model", 50);

        let histograms = registry.token_histograms.read();
        let prompt = histograms
            .get(&("prompt".to_string(), "model".to_string()))
            .unwrap();
        assert_eq!(prompt.count, 2);

        let completion = histograms
            .get(&("completion".to_string(), "model".to_string()))
            .unwrap();
        assert_eq!(completion.count, 1);
    }

    #[test]
    fn test_prometheus_registry_embedding_batch() {
        let registry = PrometheusRegistry::new();
        registry.record_embedding_batch("model", 10);
        registry.record_embedding_batch("model", 50);

        let batches = registry.embedding_batches.read();
        let histogram = batches.get("model").unwrap();
        assert_eq!(histogram.count, 2);
    }

    #[test]
    fn test_prometheus_registry_multiple_models() {
        let registry = PrometheusRegistry::new();
        registry.record_request("chat", "llama-3");
        registry.record_request("chat", "mistral");
        registry.record_request("chat", "llama-3");

        let requests = registry.requests.read();
        assert_eq!(
            *requests
                .get(&("chat".to_string(), "llama-3".to_string()))
                .unwrap(),
            2
        );
        assert_eq!(
            *requests
                .get(&("chat".to_string(), "mistral".to_string()))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_prometheus_registry_render_empty() {
        let registry = PrometheusRegistry::new();
        let output = registry.render();

        assert!(output.contains("# HELP infernum_requests_total"));
        assert!(output.contains("# HELP infernum_errors_total"));
        assert!(output.contains("infernum_active_requests 0"));
        assert!(output.contains("infernum_model_loaded 0"));
    }

    #[test]
    fn test_prometheus_registry_render_with_token_histograms() {
        let registry = PrometheusRegistry::new();
        registry.record_tokens("prompt", "test-model", 100);

        let output = registry.render();
        assert!(output.contains("# HELP infernum_tokens"));
        assert!(output.contains("# TYPE infernum_tokens histogram"));
    }

    #[test]
    fn test_prometheus_registry_render_with_embedding_batch() {
        let registry = PrometheusRegistry::new();
        registry.record_embedding_batch("test-model", 25);

        let output = registry.render();
        assert!(output.contains("# HELP infernum_embedding_batch_size"));
        assert!(output.contains("# TYPE infernum_embedding_batch_size histogram"));
    }

    // === Additional MetricsCollector Tests ===

    #[test]
    fn test_metrics_collector_new() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        assert_eq!(collector.inference().requests(), 0);
        assert_eq!(collector.inference().errors(), 0);
    }

    #[test]
    fn test_metrics_collector_inference_accessor() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        let inference = collector.inference();
        assert_eq!(inference.requests(), 0);
    }

    #[test]
    fn test_metrics_collector_prometheus_accessor() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        let prometheus = collector.prometheus();
        assert_eq!(prometheus.active_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_collector_completion_request() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_completion_request(200, 100, 0.8, "test-model");

        assert_eq!(collector.inference().requests(), 1);
        assert_eq!(collector.inference().prompt_tokens(), 200);
        assert_eq!(collector.inference().tokens_generated(), 100);
    }

    #[test]
    fn test_metrics_collector_embedding_request() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_embedding_request(500, 0.2, "embedding-model", 10);

        assert_eq!(collector.inference().requests(), 1);
        assert_eq!(collector.inference().prompt_tokens(), 500);
        assert_eq!(collector.inference().tokens_generated(), 0); // embeddings don't generate tokens
    }

    #[test]
    fn test_metrics_collector_multiple_requests() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_chat_request(100, 50, 0.5, "model-a");
        collector.record_completion_request(200, 100, 0.8, "model-b");
        collector.record_embedding_request(300, 0.2, "model-c", 5);

        assert_eq!(collector.inference().requests(), 3);
        assert_eq!(collector.inference().prompt_tokens(), 600);
        assert_eq!(collector.inference().tokens_generated(), 150);
    }

    #[test]
    fn test_metrics_collector_render_prometheus() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_chat_request(100, 50, 0.5, "test-model");
        let output = collector.render_prometheus();

        assert!(output.contains("infernum_requests_total"));
        assert!(output.contains("test-model"));
    }

    #[test]
    fn test_metrics_collector_error_multiple() {
        let config = TelemetryConfig::default();
        let collector = MetricsCollector::new(&config);

        collector.record_error("chat", "model", "timeout");
        collector.record_error("chat", "model", "internal");
        collector.record_error("completion", "model", "timeout");

        assert_eq!(collector.inference().errors(), 3);
    }

    // === Additional InferenceMetrics Tests ===

    #[test]
    fn test_inference_metrics_record_request() {
        let metrics = InferenceMetrics::default();
        metrics.record_request(100, 50);

        assert_eq!(metrics.requests(), 1);
        assert_eq!(metrics.prompt_tokens(), 100);
        assert_eq!(metrics.tokens_generated(), 50);
    }

    #[test]
    fn test_inference_metrics_record_multiple_requests() {
        let metrics = InferenceMetrics::default();
        metrics.record_request(100, 50);
        metrics.record_request(200, 100);
        metrics.record_request(150, 75);

        assert_eq!(metrics.requests(), 3);
        assert_eq!(metrics.prompt_tokens(), 450);
        assert_eq!(metrics.tokens_generated(), 225);
    }

    #[test]
    fn test_inference_metrics_record_error_multiple() {
        let metrics = InferenceMetrics::default();
        metrics.record_error();
        metrics.record_error();
        metrics.record_error();

        assert_eq!(metrics.errors(), 3);
    }

    #[test]
    fn test_inference_metrics_zero_tokens() {
        let metrics = InferenceMetrics::default();
        metrics.record_request(0, 0);

        assert_eq!(metrics.requests(), 1);
        assert_eq!(metrics.prompt_tokens(), 0);
        assert_eq!(metrics.tokens_generated(), 0);
    }

    #[test]
    fn test_inference_metrics_large_values() {
        let metrics = InferenceMetrics::default();
        metrics.record_request(u32::MAX, u32::MAX);

        assert_eq!(metrics.requests(), 1);
        assert_eq!(metrics.prompt_tokens(), u64::from(u32::MAX));
        assert_eq!(metrics.tokens_generated(), u64::from(u32::MAX));
    }

    // === Additional Timer Tests ===

    #[test]
    fn test_timer_start_label() {
        let timer = Timer::start("operation");
        assert_eq!(timer.label, "operation");
    }

    #[test]
    fn test_timer_elapsed_initially_small() {
        let timer = Timer::start("test");
        let elapsed = timer.elapsed_secs();
        // Should be very small immediately after creation
        assert!(elapsed < 0.1);
    }

    #[test]
    fn test_timer_elapsed_ms_initially_small() {
        let timer = Timer::start("test");
        let elapsed = timer.elapsed_ms();
        // Should be very small immediately after creation
        assert!(elapsed < 100.0);
    }

    #[test]
    fn test_timer_stop() {
        let timer = Timer::start("test");
        std::thread::sleep(std::time::Duration::from_millis(5));
        timer.stop(); // Should not panic, logs debug message
    }

    #[test]
    fn test_timer_drop() {
        let _timer = Timer::start("test");
        // Timer drops here, should not panic
    }

    #[test]
    fn test_timer_multiple() {
        let timer1 = Timer::start("op1");
        let timer2 = Timer::start("op2");

        std::thread::sleep(std::time::Duration::from_millis(5));

        let elapsed1 = timer1.elapsed_ms();
        let elapsed2 = timer2.elapsed_ms();

        // Both should have elapsed time
        assert!(elapsed1 >= 5.0);
        assert!(elapsed2 >= 5.0);
    }

    // === Additional ActiveRequestGuard Tests ===

    #[test]
    fn test_active_request_guard_new() {
        let registry = PrometheusRegistry::new();
        let _guard = ActiveRequestGuard::new(&registry);

        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_active_request_guard_drop() {
        let registry = PrometheusRegistry::new();
        {
            let _guard = ActiveRequestGuard::new(&registry);
            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
        }
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_active_request_guard_multiple_sequential() {
        let registry = PrometheusRegistry::new();

        {
            let _g1 = ActiveRequestGuard::new(&registry);
            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
        }
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);

        {
            let _g2 = ActiveRequestGuard::new(&registry);
            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
        }
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_active_request_guard_deeply_nested() {
        let registry = PrometheusRegistry::new();

        {
            let _g1 = ActiveRequestGuard::new(&registry);
            {
                let _g2 = ActiveRequestGuard::new(&registry);
                {
                    let _g3 = ActiveRequestGuard::new(&registry);
                    {
                        let _g4 = ActiveRequestGuard::new(&registry);
                        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 4);
                    }
                    assert_eq!(registry.active_requests.load(Ordering::Relaxed), 3);
                }
                assert_eq!(registry.active_requests.load(Ordering::Relaxed), 2);
            }
            assert_eq!(registry.active_requests.load(Ordering::Relaxed), 1);
        }
        assert_eq!(registry.active_requests.load(Ordering::Relaxed), 0);
    }

    // === Constant Value Tests ===

    #[test]
    fn test_latency_buckets_values() {
        assert_eq!(LATENCY_BUCKETS.len(), 13);
        assert!((LATENCY_BUCKETS[0] - 0.005).abs() < f64::EPSILON);
        assert!((LATENCY_BUCKETS[LATENCY_BUCKETS.len() - 1] - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_buckets_values() {
        assert_eq!(TOKEN_BUCKETS.len(), 10);
        assert!((TOKEN_BUCKETS[0] - 1.0).abs() < f64::EPSILON);
        assert!((TOKEN_BUCKETS[TOKEN_BUCKETS.len() - 1] - 8000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_buckets_sorted() {
        for i in 1..LATENCY_BUCKETS.len() {
            assert!(LATENCY_BUCKETS[i] > LATENCY_BUCKETS[i - 1]);
        }
    }

    #[test]
    fn test_token_buckets_sorted() {
        for i in 1..TOKEN_BUCKETS.len() {
            assert!(TOKEN_BUCKETS[i] > TOKEN_BUCKETS[i - 1]);
        }
    }
}
