//! Model health monitoring and failure detection.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::Notify;

use crate::registry::ModelRegistry;

/// Health status of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Model is healthy and accepting requests.
    Healthy,
    /// Model is degraded but still functional.
    Degraded,
    /// Model is unhealthy and should not receive requests.
    Unhealthy,
    /// Model health is unknown (never checked).
    Unknown,
}

impl HealthStatus {
    /// Returns true if requests can be routed to this model.
    #[must_use]
    pub fn is_routable(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

/// Configuration for health monitoring.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Interval between health checks.
    pub check_interval: Duration,
    /// Timeout for health check probe.
    pub check_timeout: Duration,
    /// Number of consecutive failures before marking unhealthy.
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy.
    pub recovery_threshold: u32,
    /// Latency threshold for degraded status (P99 in ms).
    pub degraded_latency_ms: f64,
    /// Error rate threshold for degraded status (0.0-1.0).
    pub degraded_error_rate: f64,
    /// Whether to enable automatic health checks.
    pub auto_check: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_threshold: 2,
            degraded_latency_ms: 5000.0,
            degraded_error_rate: 0.05,
            auto_check: true,
        }
    }
}

impl HealthConfig {
    /// Creates a config for production use with stricter thresholds.
    #[must_use]
    pub fn production() -> Self {
        Self {
            check_interval: Duration::from_secs(10),
            check_timeout: Duration::from_secs(5),
            failure_threshold: 2,
            recovery_threshold: 3,
            degraded_latency_ms: 2000.0,
            degraded_error_rate: 0.01,
            auto_check: true,
        }
    }

    /// Creates a config for development with relaxed thresholds.
    #[must_use]
    pub fn development() -> Self {
        Self {
            check_interval: Duration::from_secs(60),
            check_timeout: Duration::from_secs(30),
            failure_threshold: 5,
            recovery_threshold: 1,
            degraded_latency_ms: 10000.0,
            degraded_error_rate: 0.1,
            auto_check: false,
        }
    }

    /// Sets the check interval.
    #[must_use]
    pub fn with_check_interval(mut self, interval: Duration) -> Self {
        self.check_interval = interval;
        self
    }

    /// Sets the failure threshold.
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }
}

/// Health state for a single model.
#[derive(Debug)]
pub struct ModelHealthState {
    /// Current health status.
    status: RwLock<HealthStatus>,
    /// Consecutive failure count.
    consecutive_failures: AtomicU32,
    /// Consecutive success count.
    consecutive_successes: AtomicU32,
    /// Total error count.
    total_errors: AtomicU64,
    /// Total request count.
    total_requests: AtomicU64,
    /// Last successful check timestamp.
    last_success: RwLock<Option<Instant>>,
    /// Last failure timestamp.
    last_failure: RwLock<Option<Instant>>,
    /// Last error message.
    last_error: RwLock<Option<String>>,
}

impl ModelHealthState {
    /// Creates a new health state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            status: RwLock::new(HealthStatus::Unknown),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            total_errors: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            last_success: RwLock::new(None),
            last_failure: RwLock::new(None),
            last_error: RwLock::new(None),
        }
    }

    /// Returns the current status.
    #[must_use]
    pub fn status(&self) -> HealthStatus {
        *self.status.read()
    }

    /// Returns the error rate (0.0-1.0).
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let errors = self.total_errors.load(Ordering::Relaxed);
        errors as f64 / total as f64
    }

    /// Returns the consecutive failure count.
    #[must_use]
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Returns the last error message.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.last_error.read().clone()
    }

    /// Records a successful request/check.
    pub fn record_success(&self, config: &HealthConfig) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_success.write() = Some(Instant::now());

        // Check if we should transition to healthy
        let mut status = self.status.write();
        if *status != HealthStatus::Healthy && successes >= config.recovery_threshold {
            tracing::info!(
                old_status = ?*status,
                successes = successes,
                "Model recovered to healthy status"
            );
            *status = HealthStatus::Healthy;
        }
    }

    /// Records a failed request/check.
    pub fn record_failure(&self, error: impl Into<String>, config: &HealthConfig) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure.write() = Some(Instant::now());
        *self.last_error.write() = Some(error.into());

        // Check if we should transition to unhealthy
        let mut status = self.status.write();
        if failures >= config.failure_threshold {
            if *status != HealthStatus::Unhealthy {
                tracing::warn!(
                    old_status = ?*status,
                    failures = failures,
                    "Model marked unhealthy"
                );
            }
            *status = HealthStatus::Unhealthy;
        } else if *status == HealthStatus::Healthy {
            tracing::info!(
                failures = failures,
                threshold = config.failure_threshold,
                "Model degraded due to failures"
            );
            *status = HealthStatus::Degraded;
        }
    }

    /// Updates status based on latency.
    pub fn check_latency(&self, p99_latency_ms: f64, config: &HealthConfig) {
        let mut status = self.status.write();
        if p99_latency_ms > config.degraded_latency_ms {
            if *status == HealthStatus::Healthy {
                tracing::info!(
                    p99_latency_ms = p99_latency_ms,
                    threshold = config.degraded_latency_ms,
                    "Model degraded due to high latency"
                );
                *status = HealthStatus::Degraded;
            }
        }
    }

    /// Updates status based on error rate.
    pub fn check_error_rate(&self, config: &HealthConfig) {
        let error_rate = self.error_rate();
        let mut status = self.status.write();
        if error_rate > config.degraded_error_rate {
            if *status == HealthStatus::Healthy {
                tracing::info!(
                    error_rate = error_rate,
                    threshold = config.degraded_error_rate,
                    "Model degraded due to high error rate"
                );
                *status = HealthStatus::Degraded;
            }
        }
    }
}

impl Default for ModelHealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Health monitor for all registered models.
pub struct HealthMonitor {
    /// Health state for each model.
    states: dashmap::DashMap<String, Arc<ModelHealthState>>,
    /// Configuration.
    config: HealthConfig,
    /// Whether monitoring is running.
    running: AtomicBool,
    /// Notify to stop monitoring.
    stop_notify: Notify,
}

impl HealthMonitor {
    /// Creates a new health monitor.
    #[must_use]
    pub fn new(config: HealthConfig) -> Self {
        Self {
            states: dashmap::DashMap::new(),
            config,
            running: AtomicBool::new(false),
            stop_notify: Notify::new(),
        }
    }

    /// Creates with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(HealthConfig::default())
    }

    /// Returns the health state for a model.
    #[must_use]
    pub fn get_state(&self, model_id: &str) -> Arc<ModelHealthState> {
        self.states
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(ModelHealthState::new()))
            .clone()
    }

    /// Returns the health status for a model.
    #[must_use]
    pub fn status(&self, model_id: &str) -> HealthStatus {
        self.states
            .get(model_id)
            .map_or(HealthStatus::Unknown, |s| s.status())
    }

    /// Records a successful request for a model.
    pub fn record_success(&self, model_id: &str) {
        let state = self.get_state(model_id);
        state.record_success(&self.config);
    }

    /// Records a failed request for a model.
    pub fn record_failure(&self, model_id: &str, error: impl Into<String>) {
        let state = self.get_state(model_id);
        state.record_failure(error, &self.config);
    }

    /// Starts background health monitoring.
    pub fn start_monitoring(self: Arc<Self>, registry: Arc<ModelRegistry>) {
        if self.running.swap(true, Ordering::SeqCst) {
            tracing::warn!("Health monitoring already running");
            return;
        }

        let monitor = Arc::clone(&self);
        tokio::spawn(async move {
            tracing::info!(
                interval_secs = monitor.config.check_interval.as_secs(),
                "Starting health monitor"
            );

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(monitor.config.check_interval) => {
                        monitor.check_all_models(&registry).await;
                    }
                    _ = monitor.stop_notify.notified() => {
                        tracing::info!("Health monitor stopped");
                        break;
                    }
                }
            }

            monitor.running.store(false, Ordering::SeqCst);
        });
    }

    /// Stops background health monitoring.
    pub fn stop_monitoring(&self) {
        if self.running.load(Ordering::SeqCst) {
            self.stop_notify.notify_one();
        }
    }

    /// Performs health checks on all registered models.
    async fn check_all_models(&self, registry: &ModelRegistry) {
        let models = registry.all();

        for model in models {
            let model_id = model.id.0.as_str();
            let state = self.get_state(model_id);

            // Check latency
            let p99 = model.latency_stats.p99_latency_ms();
            state.check_latency(p99, &self.config);

            // Check error rate
            state.check_error_rate(&self.config);

            // Update model's healthy flag based on our state
            let is_healthy = state.status().is_routable();
            model.healthy.store(is_healthy, Ordering::SeqCst);
            *model.last_health_check.write() = Some(Instant::now());

            tracing::debug!(
                model_id = model_id,
                status = ?state.status(),
                p99_ms = p99,
                error_rate = state.error_rate(),
                "Health check completed"
            );
        }
    }

    /// Returns a summary of all model health states.
    #[must_use]
    pub fn summary(&self) -> HealthSummary {
        let mut healthy = 0;
        let mut degraded = 0;
        let mut unhealthy = 0;
        let mut unknown = 0;

        for entry in self.states.iter() {
            match entry.value().status() {
                HealthStatus::Healthy => healthy += 1,
                HealthStatus::Degraded => degraded += 1,
                HealthStatus::Unhealthy => unhealthy += 1,
                HealthStatus::Unknown => unknown += 1,
            }
        }

        HealthSummary {
            healthy,
            degraded,
            unhealthy,
            unknown,
            total: healthy + degraded + unhealthy + unknown,
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &HealthConfig {
        &self.config
    }
}

/// Summary of model health states.
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Number of healthy models.
    pub healthy: usize,
    /// Number of degraded models.
    pub degraded: usize,
    /// Number of unhealthy models.
    pub unhealthy: usize,
    /// Number of models with unknown health.
    pub unknown: usize,
    /// Total number of models.
    pub total: usize,
}

impl HealthSummary {
    /// Returns the percentage of healthy models.
    #[must_use]
    pub fn healthy_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.healthy as f64 / self.total as f64) * 100.0
    }

    /// Returns the percentage of routable models (healthy + degraded).
    #[must_use]
    pub fn routable_percentage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        ((self.healthy + self.degraded) as f64 / self.total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_routable() {
        assert!(HealthStatus::Healthy.is_routable());
        assert!(HealthStatus::Degraded.is_routable());
        assert!(!HealthStatus::Unhealthy.is_routable());
        assert!(!HealthStatus::Unknown.is_routable());
    }

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.recovery_threshold, 2);
        assert!(config.auto_check);
    }

    #[test]
    fn test_health_config_production() {
        let config = HealthConfig::production();
        assert_eq!(config.failure_threshold, 2);
        assert_eq!(config.check_interval.as_secs(), 10);
    }

    #[test]
    fn test_model_health_state_success() {
        let config = HealthConfig::default();
        let state = ModelHealthState::new();

        // Record successes until healthy
        for _ in 0..config.recovery_threshold {
            state.record_success(&config);
        }

        assert_eq!(state.status(), HealthStatus::Healthy);
        assert_eq!(state.consecutive_failures(), 0);
    }

    #[test]
    fn test_model_health_state_failure() {
        let config = HealthConfig::default();
        let state = ModelHealthState::new();

        // First make it healthy
        state.record_success(&config);
        state.record_success(&config);
        assert_eq!(state.status(), HealthStatus::Healthy);

        // Record failures
        for i in 0..config.failure_threshold {
            state.record_failure(format!("Error {}", i), &config);
        }

        assert_eq!(state.status(), HealthStatus::Unhealthy);
        assert!(state.last_error().is_some());
    }

    #[test]
    fn test_model_health_state_recovery() {
        let config = HealthConfig::default();
        let state = ModelHealthState::new();

        // Make unhealthy
        for i in 0..config.failure_threshold {
            state.record_failure(format!("Error {}", i), &config);
        }
        assert_eq!(state.status(), HealthStatus::Unhealthy);

        // Recover
        for _ in 0..config.recovery_threshold {
            state.record_success(&config);
        }
        assert_eq!(state.status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_model_health_state_error_rate() {
        let state = ModelHealthState::new();
        let config = HealthConfig::default();

        // Record mixed results
        for _ in 0..90 {
            state.record_success(&config);
        }
        for i in 0..10 {
            state.record_failure(format!("Error {}", i), &config);
        }

        let error_rate = state.error_rate();
        assert!((error_rate - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_health_monitor_get_state() {
        let monitor = HealthMonitor::default_config();

        let state1 = monitor.get_state("model-1");
        let state2 = monitor.get_state("model-1");

        // Should return the same state
        assert!(Arc::ptr_eq(&state1, &state2));
    }

    #[test]
    fn test_health_monitor_record() {
        let monitor = HealthMonitor::default_config();

        monitor.record_success("model-1");
        monitor.record_success("model-1");
        assert_eq!(monitor.status("model-1"), HealthStatus::Healthy);

        monitor.record_failure("model-2", "Connection error");
        monitor.record_failure("model-2", "Connection error");
        monitor.record_failure("model-2", "Connection error");
        assert_eq!(monitor.status("model-2"), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_summary() {
        let monitor = HealthMonitor::new(HealthConfig {
            failure_threshold: 1,
            recovery_threshold: 1,
            ..Default::default()
        });

        monitor.record_success("model-1");
        monitor.record_success("model-2");
        monitor.record_failure("model-3", "Error");

        let summary = monitor.summary();
        assert_eq!(summary.healthy, 2);
        assert_eq!(summary.unhealthy, 1);
        assert_eq!(summary.total, 3);
        assert!((summary.healthy_percentage() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_health_latency_degradation() {
        let config = HealthConfig {
            degraded_latency_ms: 100.0,
            recovery_threshold: 1,
            ..Default::default()
        };
        let state = ModelHealthState::new();

        // Make healthy first
        state.record_success(&config);
        assert_eq!(state.status(), HealthStatus::Healthy);

        // High latency should degrade
        state.check_latency(200.0, &config);
        assert_eq!(state.status(), HealthStatus::Degraded);
    }
}
