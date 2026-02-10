//! Circuit breaker for inference backend resilience.
//!
//! The circuit breaker pattern prevents cascading failures by temporarily
//! blocking requests to a failing service, allowing it time to recover.
//!
//! # States
//!
//! | State | Behavior |
//! |-------|----------|
//! | Closed | Normal operation, requests pass through |
//! | Open | All requests rejected immediately |
//! | HalfOpen | Limited requests allowed to test recovery |
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
//! use std::time::Duration;
//!
//! let config = CircuitBreakerConfig {
//!     failure_threshold: 5,
//!     reset_timeout: Duration::from_secs(30),
//!     half_open_requests: 3,
//!     ..Default::default()
//! };
//!
//! let breaker = CircuitBreaker::new(config);
//!
//! // Check if request is allowed
//! if breaker.allow_request() {
//!     match do_inference().await {
//!         Ok(_) => breaker.record_success(),
//!         Err(_) => breaker.record_failure(),
//!     }
//! }
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    /// Normal operation - requests pass through.
    Closed = 0,
    /// Circuit is open - requests are rejected.
    Open = 1,
    /// Testing recovery - limited requests allowed.
    HalfOpen = 2,
}

impl CircuitState {
    /// Returns the string name of the state.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

impl From<u8> for CircuitState {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Closed,
            1 => Self::Open,
            2 => Self::HalfOpen,
            _ => Self::Closed,
        }
    }
}

/// Configuration for the circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,

    /// Time to wait before attempting recovery (transition to half-open).
    pub reset_timeout: Duration,

    /// Number of successful requests required in half-open state to close circuit.
    pub half_open_requests: u32,

    /// Optional name for metrics and logging.
    pub name: String,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            half_open_requests: 3,
            name: "inference".to_string(),
        }
    }
}

impl CircuitBreakerConfig {
    /// Creates a new configuration with the given failure threshold.
    #[must_use]
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            ..Default::default()
        }
    }

    /// Creates a strict configuration (opens quickly, recovers slowly).
    #[must_use]
    pub fn strict() -> Self {
        Self {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(60),
            half_open_requests: 5,
            name: "inference".to_string(),
        }
    }

    /// Creates a lenient configuration (tolerates more failures).
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            failure_threshold: 10,
            reset_timeout: Duration::from_secs(15),
            half_open_requests: 2,
            name: "inference".to_string(),
        }
    }

    /// Sets the name for this circuit breaker.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// Thread-safe circuit breaker for protecting inference calls.
///
/// Uses atomic operations for lock-free state management, suitable for
/// high-throughput scenarios.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state (0=Closed, 1=Open, 2=HalfOpen).
    state: AtomicU8,

    /// Consecutive failure count.
    failure_count: AtomicU32,

    /// Consecutive success count in half-open state.
    half_open_successes: AtomicU32,

    /// Timestamp when circuit opened (milliseconds since UNIX epoch).
    opened_at: AtomicU64,

    /// Reference instant for relative time calculations.
    start_instant: Instant,

    /// Configuration.
    config: CircuitBreakerConfig,

    // Metrics counters
    total_requests: AtomicU64,
    rejected_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    state_transitions: AtomicU64,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker with the given configuration.
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU32::new(0),
            half_open_successes: AtomicU32::new(0),
            opened_at: AtomicU64::new(0),
            start_instant: Instant::now(),
            config,
            total_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            state_transitions: AtomicU64::new(0),
        }
    }

    /// Creates a circuit breaker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Returns the current state of the circuit breaker.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        CircuitState::from(self.state.load(Ordering::Acquire))
    }

    /// Returns the name of this circuit breaker.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns the current failure count.
    #[must_use]
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Checks if a request should be allowed.
    ///
    /// Returns `true` if the request can proceed, `false` if it should be rejected.
    #[must_use]
    pub fn allow_request(&self) -> bool {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let current_state = self.state();

        match current_state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout has elapsed
                if self.should_attempt_reset() {
                    // Transition to half-open
                    self.transition_to(CircuitState::HalfOpen);
                    true
                } else {
                    self.rejected_requests.fetch_add(1, Ordering::Relaxed);
                    false
                }
            },
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                true
            },
        }
    }

    /// Records a successful request.
    ///
    /// In half-open state, may transition to closed after enough successes.
    pub fn record_success(&self) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);

        let current_state = self.state();

        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            },
            CircuitState::HalfOpen => {
                let successes = self.half_open_successes.fetch_add(1, Ordering::Relaxed) + 1;

                if successes >= self.config.half_open_requests {
                    // Enough successes, close the circuit
                    self.transition_to(CircuitState::Closed);
                }
            },
            CircuitState::Open => {
                // Should not happen, but handle gracefully
            },
        }
    }

    /// Records a failed request.
    ///
    /// May transition to open state after reaching failure threshold.
    pub fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);

        let current_state = self.state();

        match current_state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

                if failures >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open);
                }
            },
            CircuitState::HalfOpen => {
                // Any failure in half-open immediately reopens the circuit
                self.transition_to(CircuitState::Open);
            },
            CircuitState::Open => {
                // Already open, nothing to do
            },
        }
    }

    /// Manually resets the circuit breaker to closed state.
    pub fn reset(&self) {
        self.transition_to(CircuitState::Closed);
        self.failure_count.store(0, Ordering::Relaxed);
        self.half_open_successes.store(0, Ordering::Relaxed);
    }

    /// Returns metrics for this circuit breaker.
    #[must_use]
    pub fn metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            name: self.config.name.clone(),
            state: self.state(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            state_transitions: self.state_transitions.load(Ordering::Relaxed),
        }
    }

    /// Renders Prometheus-format metrics.
    #[must_use]
    pub fn render_prometheus_metrics(&self) -> String {
        let metrics = self.metrics();
        let name = &metrics.name;

        format!(
            r#"# HELP infernum_circuit_breaker_state Current circuit breaker state (0=closed, 1=open, 2=half_open)
# TYPE infernum_circuit_breaker_state gauge
infernum_circuit_breaker_state{{name="{name}"}} {}

# HELP infernum_circuit_breaker_failures Current consecutive failure count
# TYPE infernum_circuit_breaker_failures gauge
infernum_circuit_breaker_failures{{name="{name}"}} {}

# HELP infernum_circuit_breaker_requests_total Total requests through circuit breaker
# TYPE infernum_circuit_breaker_requests_total counter
infernum_circuit_breaker_requests_total{{name="{name}"}} {}

# HELP infernum_circuit_breaker_rejected_total Requests rejected due to open circuit
# TYPE infernum_circuit_breaker_rejected_total counter
infernum_circuit_breaker_rejected_total{{name="{name}"}} {}

# HELP infernum_circuit_breaker_transitions_total State transitions
# TYPE infernum_circuit_breaker_transitions_total counter
infernum_circuit_breaker_transitions_total{{name="{name}"}} {}
"#,
            metrics.state as u8,
            metrics.failure_count,
            metrics.total_requests,
            metrics.rejected_requests,
            metrics.state_transitions,
        )
    }

    // Internal helper methods

    fn should_attempt_reset(&self) -> bool {
        let opened_at = self.opened_at.load(Ordering::Acquire);
        if opened_at == 0 {
            return false;
        }

        let now = self.start_instant.elapsed().as_millis() as u64;
        let elapsed = now.saturating_sub(opened_at);
        elapsed >= self.config.reset_timeout.as_millis() as u64
    }

    fn transition_to(&self, new_state: CircuitState) {
        let old_state = self.state.swap(new_state as u8, Ordering::AcqRel);

        if old_state != new_state as u8 {
            self.state_transitions.fetch_add(1, Ordering::Relaxed);

            match new_state {
                CircuitState::Open => {
                    // Record when we opened (use max(1, ...) to ensure non-zero)
                    let now = self.start_instant.elapsed().as_millis() as u64;
                    self.opened_at.store(now.max(1), Ordering::Release);
                },
                CircuitState::Closed => {
                    // Reset counters
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.half_open_successes.store(0, Ordering::Relaxed);
                    self.opened_at.store(0, Ordering::Relaxed);
                },
                CircuitState::HalfOpen => {
                    // Reset half-open success counter
                    self.half_open_successes.store(0, Ordering::Relaxed);
                },
            }

            tracing::info!(
                circuit_breaker = %self.config.name,
                old_state = CircuitState::from(old_state).as_str(),
                new_state = new_state.as_str(),
                "Circuit breaker state transition"
            );
        }
    }
}

/// Metrics snapshot for a circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    /// Name of the circuit breaker.
    pub name: String,
    /// Current state.
    pub state: CircuitState,
    /// Current consecutive failure count.
    pub failure_count: u32,
    /// Total requests processed.
    pub total_requests: u64,
    /// Requests rejected due to open circuit.
    pub rejected_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Number of state transitions.
    pub state_transitions: u64,
}

/// Error returned when circuit is open.
#[derive(Debug, Clone)]
pub struct CircuitOpenError {
    /// Name of the circuit breaker.
    pub circuit_name: String,
    /// How long until reset will be attempted.
    pub retry_after: Option<Duration>,
}

impl std::fmt::Display for CircuitOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "circuit breaker '{}' is open", self.circuit_name)?;
        if let Some(retry_after) = self.retry_after {
            write!(f, ", retry after {:?}", retry_after)?;
        }
        Ok(())
    }
}

impl std::error::Error for CircuitOpenError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_circuit_state_default() {
        let breaker = CircuitBreaker::with_defaults();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_allows_requests_when_closed() {
        let breaker = CircuitBreaker::with_defaults();
        assert!(breaker.allow_request());
        assert!(breaker.allow_request());
        assert!(breaker.allow_request());
    }

    #[test]
    fn test_circuit_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(30),
            half_open_requests: 2,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Record failures up to threshold
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_rejects_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_secs(60), // Long timeout
            half_open_requests: 1,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Requests should be rejected
        assert!(!breaker.allow_request());
        assert!(!breaker.allow_request());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(30),
            half_open_requests: 2,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.failure_count(), 2);

        // Success resets the counter
        breaker.record_success();
        assert_eq!(breaker.failure_count(), 0);
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_transitions_to_closed_on_success() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(50), // Short timeout
            half_open_requests: 2,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for reset timeout (generous margin)
        thread::sleep(Duration::from_millis(100));

        // Next request should transition to half-open
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Successes in half-open
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_returns_to_open_on_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(50),
            half_open_requests: 3,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Wait for reset timeout (generous margin)
        thread::sleep(Duration::from_millis(100));

        // Transition to half-open
        assert!(breaker.allow_request());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Any failure in half-open reopens
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
    }

    #[test]
    fn test_manual_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_secs(60),
            half_open_requests: 1,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        // Manual reset
        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert_eq!(breaker.failure_count(), 0);
    }

    #[test]
    fn test_metrics() {
        let breaker = CircuitBreaker::with_defaults();

        let _ = breaker.allow_request();
        breaker.record_success();
        let _ = breaker.allow_request();
        breaker.record_failure();

        let metrics = breaker.metrics();
        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[test]
    fn test_prometheus_metrics_output() {
        let breaker = CircuitBreaker::with_defaults();
        let output = breaker.render_prometheus_metrics();

        assert!(output.contains("infernum_circuit_breaker_state"));
        assert!(output.contains("infernum_circuit_breaker_failures"));
        assert!(output.contains("infernum_circuit_breaker_requests_total"));
    }

    #[test]
    fn test_config_presets() {
        let strict = CircuitBreakerConfig::strict();
        assert_eq!(strict.failure_threshold, 3);
        assert_eq!(strict.reset_timeout, Duration::from_secs(60));

        let lenient = CircuitBreakerConfig::lenient();
        assert_eq!(lenient.failure_threshold, 10);
        assert_eq!(lenient.reset_timeout, Duration::from_secs(15));
    }

    #[test]
    fn test_config_with_name() {
        let config = CircuitBreakerConfig::default().with_name("gpu-inference");
        assert_eq!(config.name, "gpu-inference");
    }

    #[test]
    fn test_circuit_state_as_str() {
        assert_eq!(CircuitState::Closed.as_str(), "closed");
        assert_eq!(CircuitState::Open.as_str(), "open");
        assert_eq!(CircuitState::HalfOpen.as_str(), "half_open");
    }

    #[test]
    fn test_state_transitions_counted() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(50),
            half_open_requests: 1,
            name: "test".to_string(),
        };
        let breaker = CircuitBreaker::new(config);

        // Closed -> Open
        breaker.record_failure();
        assert_eq!(breaker.metrics().state_transitions, 1);

        // Wait for reset (generous margin)
        thread::sleep(Duration::from_millis(100));

        // Open -> HalfOpen
        let _ = breaker.allow_request();
        assert_eq!(breaker.metrics().state_transitions, 2);

        // HalfOpen -> Closed
        breaker.record_success();
        assert_eq!(breaker.metrics().state_transitions, 3);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let breaker = Arc::new(CircuitBreaker::with_defaults());
        let mut handles = vec![];

        // Spawn multiple threads doing requests
        for _ in 0..10 {
            let b = Arc::clone(&breaker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    if b.allow_request() {
                        if rand_bool() {
                            b.record_success();
                        } else {
                            b.record_failure();
                        }
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Should not panic and metrics should be consistent
        let metrics = breaker.metrics();
        assert!(metrics.total_requests > 0);
    }

    // Simple pseudo-random for testing
    fn rand_bool() -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        thread::current().id().hash(&mut hasher);
        hasher.finish() % 2 == 0
    }
}
