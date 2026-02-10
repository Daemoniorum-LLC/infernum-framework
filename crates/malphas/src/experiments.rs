//! A/B testing and experimentation framework for model comparison.
//!
//! This module enables running controlled experiments to compare model
//! performance, quality, and cost across different configurations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rand::Rng;

/// Experiment identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExperimentId(pub String);

impl From<&str> for ExperimentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Variant in an experiment.
#[derive(Debug, Clone)]
pub struct Variant {
    /// Variant name (e.g., "control", "treatment_a").
    pub name: String,
    /// Model ID to use for this variant.
    pub model_id: String,
    /// Traffic allocation (0.0-1.0).
    pub allocation: f64,
    /// Whether this is the control variant.
    pub is_control: bool,
}

impl Variant {
    /// Creates a control variant.
    pub fn control(model_id: impl Into<String>) -> Self {
        Self {
            name: "control".to_string(),
            model_id: model_id.into(),
            allocation: 0.5,
            is_control: true,
        }
    }

    /// Creates a treatment variant.
    pub fn treatment(name: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model_id: model_id.into(),
            allocation: 0.5,
            is_control: false,
        }
    }

    /// Sets the traffic allocation.
    pub fn with_allocation(mut self, allocation: f64) -> Self {
        self.allocation = allocation.clamp(0.0, 1.0);
        self
    }
}

/// Metrics collected for each variant.
#[derive(Debug, Default)]
pub struct VariantMetrics {
    /// Number of requests.
    pub request_count: AtomicU64,
    /// Number of successful requests.
    pub success_count: AtomicU64,
    /// Number of failed requests.
    pub failure_count: AtomicU64,
    /// Total latency in milliseconds.
    pub total_latency_ms: AtomicU64,
    /// Total input tokens.
    pub total_input_tokens: AtomicU64,
    /// Total output tokens.
    pub total_output_tokens: AtomicU64,
    /// Quality scores (if applicable).
    quality_scores: RwLock<Vec<f64>>,
    /// User feedback scores.
    feedback_scores: RwLock<Vec<f64>>,
}

impl VariantMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a successful request.
    pub fn record_success(&self, latency_ms: u64, input_tokens: u32, output_tokens: u32) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
        self.total_input_tokens
            .fetch_add(input_tokens as u64, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(output_tokens as u64, Ordering::Relaxed);
    }

    /// Records a failed request.
    pub fn record_failure(&self, latency_ms: u64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
    }

    /// Records a quality score.
    pub fn record_quality(&self, score: f64) {
        self.quality_scores.write().push(score);
    }

    /// Records user feedback.
    pub fn record_feedback(&self, score: f64) {
        self.feedback_scores.write().push(score);
    }

    /// Returns average latency in milliseconds.
    pub fn average_latency_ms(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.total_latency_ms.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Returns success rate (0.0-1.0).
    pub fn success_rate(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.success_count.load(Ordering::Relaxed) as f64 / count as f64
    }

    /// Returns average quality score.
    pub fn average_quality(&self) -> Option<f64> {
        let scores = self.quality_scores.read();
        if scores.is_empty() {
            return None;
        }
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Returns average feedback score.
    pub fn average_feedback(&self) -> Option<f64> {
        let scores = self.feedback_scores.read();
        if scores.is_empty() {
            return None;
        }
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Returns average tokens per request.
    pub fn average_tokens(&self) -> (f64, f64) {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            return (0.0, 0.0);
        }
        let input = self.total_input_tokens.load(Ordering::Relaxed) as f64 / count as f64;
        let output = self.total_output_tokens.load(Ordering::Relaxed) as f64 / count as f64;
        (input, output)
    }
}

/// Experiment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Experiment is being set up.
    Draft,
    /// Experiment is actively running.
    Running,
    /// Experiment is paused.
    Paused,
    /// Experiment has concluded.
    Completed,
    /// Experiment was cancelled.
    Cancelled,
}

/// An A/B test experiment.
pub struct Experiment {
    /// Experiment identifier.
    pub id: ExperimentId,
    /// Human-readable name.
    pub name: String,
    /// Description of what's being tested.
    pub description: String,
    /// Variants in the experiment.
    variants: Vec<Variant>,
    /// Metrics per variant.
    metrics: HashMap<String, Arc<VariantMetrics>>,
    /// Current status.
    status: RwLock<ExperimentStatus>,
    /// Start time.
    started_at: RwLock<Option<Instant>>,
    /// End time.
    ended_at: RwLock<Option<Instant>>,
    /// Minimum samples per variant before concluding.
    pub min_samples: u64,
    /// Maximum duration.
    pub max_duration: Option<Duration>,
    /// Statistical significance threshold (p-value).
    pub significance_threshold: f64,
}

impl Experiment {
    /// Creates a new experiment.
    pub fn new(id: impl Into<ExperimentId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            variants: Vec::new(),
            metrics: HashMap::new(),
            status: RwLock::new(ExperimentStatus::Draft),
            started_at: RwLock::new(None),
            ended_at: RwLock::new(None),
            min_samples: 100,
            max_duration: None,
            significance_threshold: 0.05,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Adds a variant.
    pub fn with_variant(mut self, variant: Variant) -> Self {
        let name = variant.name.clone();
        self.variants.push(variant);
        self.metrics.insert(name, Arc::new(VariantMetrics::new()));
        self
    }

    /// Sets minimum samples.
    pub fn with_min_samples(mut self, samples: u64) -> Self {
        self.min_samples = samples;
        self
    }

    /// Sets maximum duration.
    pub fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Starts the experiment.
    pub fn start(&self) {
        *self.status.write() = ExperimentStatus::Running;
        *self.started_at.write() = Some(Instant::now());
    }

    /// Pauses the experiment.
    pub fn pause(&self) {
        *self.status.write() = ExperimentStatus::Paused;
    }

    /// Resumes the experiment.
    pub fn resume(&self) {
        *self.status.write() = ExperimentStatus::Running;
    }

    /// Completes the experiment.
    pub fn complete(&self) {
        *self.status.write() = ExperimentStatus::Completed;
        *self.ended_at.write() = Some(Instant::now());
    }

    /// Cancels the experiment.
    pub fn cancel(&self) {
        *self.status.write() = ExperimentStatus::Cancelled;
        *self.ended_at.write() = Some(Instant::now());
    }

    /// Returns the current status.
    pub fn status(&self) -> ExperimentStatus {
        *self.status.read()
    }

    /// Selects a variant for a request.
    pub fn select_variant(&self) -> Option<&Variant> {
        if *self.status.read() != ExperimentStatus::Running {
            return None;
        }

        let mut rng = rand::thread_rng();
        let roll: f64 = rng.gen();

        let mut cumulative = 0.0;
        for variant in &self.variants {
            cumulative += variant.allocation;
            if roll < cumulative {
                return Some(variant);
            }
        }

        self.variants.last()
    }

    /// Returns metrics for a variant.
    pub fn metrics(&self, variant_name: &str) -> Option<Arc<VariantMetrics>> {
        self.metrics.get(variant_name).cloned()
    }

    /// Returns all variants.
    pub fn variants(&self) -> &[Variant] {
        &self.variants
    }

    /// Checks if experiment should auto-complete.
    pub fn should_complete(&self) -> bool {
        // Check minimum samples
        let all_have_min_samples = self
            .metrics
            .values()
            .all(|m| m.request_count.load(Ordering::Relaxed) >= self.min_samples);

        if !all_have_min_samples {
            return false;
        }

        // Check max duration
        if let Some(max_dur) = self.max_duration {
            if let Some(started) = *self.started_at.read() {
                if started.elapsed() >= max_dur {
                    return true;
                }
            }
        }

        // Check statistical significance
        self.is_significant()
    }

    /// Checks if results are statistically significant.
    pub fn is_significant(&self) -> bool {
        // Simple significance check based on success rate difference
        // In production, would use proper statistical tests (chi-squared, t-test)

        let control = self
            .variants
            .iter()
            .find(|v| v.is_control)
            .and_then(|v| self.metrics.get(&v.name));
        let treatment = self
            .variants
            .iter()
            .find(|v| !v.is_control)
            .and_then(|v| self.metrics.get(&v.name));

        if let (Some(ctrl), Some(treat)) = (control, treatment) {
            let ctrl_rate = ctrl.success_rate();
            let treat_rate = treat.success_rate();
            let ctrl_n = ctrl.request_count.load(Ordering::Relaxed) as f64;
            let treat_n = treat.request_count.load(Ordering::Relaxed) as f64;

            if ctrl_n < 30.0 || treat_n < 30.0 {
                return false;
            }

            // Simple z-test approximation
            let pooled_rate = (ctrl_rate * ctrl_n + treat_rate * treat_n) / (ctrl_n + treat_n);
            let se = (pooled_rate * (1.0 - pooled_rate) * (1.0 / ctrl_n + 1.0 / treat_n)).sqrt();

            if se == 0.0 {
                return false;
            }

            let z = (treat_rate - ctrl_rate).abs() / se;

            // z > 1.96 corresponds to p < 0.05
            z > 1.96
        } else {
            false
        }
    }

    /// Generates a summary report.
    pub fn summary(&self) -> ExperimentSummary {
        let variant_summaries: Vec<_> = self
            .variants
            .iter()
            .map(|v| {
                let metrics = self.metrics.get(&v.name);
                VariantSummary {
                    name: v.name.clone(),
                    model_id: v.model_id.clone(),
                    is_control: v.is_control,
                    request_count: metrics
                        .map(|m| m.request_count.load(Ordering::Relaxed))
                        .unwrap_or(0),
                    success_rate: metrics.map(|m| m.success_rate()).unwrap_or(0.0),
                    avg_latency_ms: metrics.map(|m| m.average_latency_ms()).unwrap_or(0.0),
                    avg_quality: metrics.and_then(|m| m.average_quality()),
                }
            })
            .collect();

        let winner = self.determine_winner();

        ExperimentSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            status: *self.status.read(),
            variants: variant_summaries,
            is_significant: self.is_significant(),
            winner,
            duration: self.started_at.read().map(|s| s.elapsed()),
        }
    }

    /// Determines the winning variant.
    fn determine_winner(&self) -> Option<String> {
        if !self.is_significant() {
            return None;
        }

        self.variants
            .iter()
            .filter_map(|v| {
                let metrics = self.metrics.get(&v.name)?;
                let score = metrics.success_rate() * 100.0 - metrics.average_latency_ms() * 0.01;
                Some((v.name.clone(), score))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name)
    }
}

/// Summary of a variant's performance.
#[derive(Debug, Clone)]
pub struct VariantSummary {
    /// Variant name.
    pub name: String,
    /// Model ID.
    pub model_id: String,
    /// Whether this is control.
    pub is_control: bool,
    /// Total requests.
    pub request_count: u64,
    /// Success rate.
    pub success_rate: f64,
    /// Average latency.
    pub avg_latency_ms: f64,
    /// Average quality score.
    pub avg_quality: Option<f64>,
}

/// Summary of an experiment.
#[derive(Debug, Clone)]
pub struct ExperimentSummary {
    /// Experiment ID.
    pub id: ExperimentId,
    /// Experiment name.
    pub name: String,
    /// Current status.
    pub status: ExperimentStatus,
    /// Variant summaries.
    pub variants: Vec<VariantSummary>,
    /// Whether results are significant.
    pub is_significant: bool,
    /// Winner (if determined).
    pub winner: Option<String>,
    /// Duration.
    pub duration: Option<Duration>,
}

/// Manager for running experiments.
pub struct ExperimentManager {
    /// Active experiments.
    experiments: RwLock<HashMap<ExperimentId, Arc<Experiment>>>,
    /// Completed experiments (for history).
    completed: RwLock<Vec<ExperimentSummary>>,
    /// Maximum completed experiments to retain.
    max_history: usize,
}

impl ExperimentManager {
    /// Creates a new experiment manager.
    pub fn new() -> Self {
        Self {
            experiments: RwLock::new(HashMap::new()),
            completed: RwLock::new(Vec::new()),
            max_history: 100,
        }
    }

    /// Registers an experiment.
    pub fn register(&self, experiment: Experiment) -> Arc<Experiment> {
        let exp = Arc::new(experiment);
        self.experiments
            .write()
            .insert(exp.id.clone(), Arc::clone(&exp));
        exp
    }

    /// Gets an experiment by ID.
    pub fn get(&self, id: &ExperimentId) -> Option<Arc<Experiment>> {
        self.experiments.read().get(id).cloned()
    }

    /// Lists active experiments.
    pub fn active(&self) -> Vec<Arc<Experiment>> {
        self.experiments
            .read()
            .values()
            .filter(|e| e.status() == ExperimentStatus::Running)
            .cloned()
            .collect()
    }

    /// Gets an experiment for a model (if one is running).
    pub fn get_for_model(&self, model_id: &str) -> Option<(Arc<Experiment>, Variant)> {
        for exp in self.experiments.read().values() {
            if exp.status() != ExperimentStatus::Running {
                continue;
            }

            for variant in exp.variants() {
                if variant.model_id == model_id {
                    if let Some(selected) = exp.select_variant() {
                        return Some((Arc::clone(exp), selected.clone()));
                    }
                }
            }
        }
        None
    }

    /// Checks and auto-completes experiments.
    pub fn check_completions(&self) {
        let mut to_complete = Vec::new();

        for (id, exp) in self.experiments.read().iter() {
            if exp.status() == ExperimentStatus::Running && exp.should_complete() {
                to_complete.push(id.clone());
            }
        }

        for id in to_complete {
            if let Some(exp) = self.experiments.read().get(&id) {
                exp.complete();
                let summary = exp.summary();

                let mut completed = self.completed.write();
                completed.push(summary);
                if completed.len() > self.max_history {
                    completed.remove(0);
                }
            }
        }
    }

    /// Returns experiment history.
    pub fn history(&self) -> Vec<ExperimentSummary> {
        self.completed.read().clone()
    }

    /// Removes an experiment.
    pub fn remove(&self, id: &ExperimentId) {
        self.experiments.write().remove(id);
    }
}

impl Default for ExperimentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_creation() {
        let control = Variant::control("gpt-4");
        assert!(control.is_control);
        assert_eq!(control.allocation, 0.5);

        let treatment = Variant::treatment("faster", "gpt-4-turbo").with_allocation(0.3);
        assert!(!treatment.is_control);
        assert_eq!(treatment.allocation, 0.3);
    }

    #[test]
    fn test_experiment_creation() {
        let exp = Experiment::new("test-001", "Model Speed Test")
            .with_description("Compare GPT-4 vs GPT-4-Turbo latency")
            .with_variant(Variant::control("gpt-4"))
            .with_variant(Variant::treatment("turbo", "gpt-4-turbo"))
            .with_min_samples(50);

        assert_eq!(exp.variants().len(), 2);
        assert_eq!(exp.status(), ExperimentStatus::Draft);
    }

    #[test]
    fn test_variant_selection() {
        let exp = Experiment::new("test-002", "Test")
            .with_variant(Variant::control("a").with_allocation(0.5))
            .with_variant(Variant::treatment("b", "b").with_allocation(0.5));

        exp.start();

        // Should select a variant when running
        let variant = exp.select_variant();
        assert!(variant.is_some());

        exp.pause();

        // Should not select when paused
        let variant = exp.select_variant();
        assert!(variant.is_none());
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = VariantMetrics::new();

        metrics.record_success(100, 500, 200);
        metrics.record_success(150, 600, 250);
        metrics.record_failure(50);

        assert_eq!(metrics.request_count.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.success_count.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.failure_count.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.average_latency_ms(), 100.0); // (100+150+50)/3

        let rate = metrics.success_rate();
        assert!((rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_experiment_manager() {
        let manager = ExperimentManager::new();

        let exp = Experiment::new("exp-001", "Test Experiment")
            .with_variant(Variant::control("model-a"))
            .with_variant(Variant::treatment("b", "model-b"));

        let registered = manager.register(exp);
        registered.start();

        let active = manager.active();
        assert_eq!(active.len(), 1);

        let fetched = manager.get(&"exp-001".into());
        assert!(fetched.is_some());
    }
}
