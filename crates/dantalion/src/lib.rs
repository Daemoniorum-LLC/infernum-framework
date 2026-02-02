//! # Dantalion
//!
//! *"The Duke reveals all secrets"*
//!
//! Dantalion provides comprehensive observability for the Infernum ecosystem,
//! including distributed tracing, metrics collection, and structured logging.
//!
//! ## Features
//!
//! - **OpenTelemetry Integration**: Native OTLP export support
//! - **LLM-Specific Metrics**: Token throughput, latency percentiles, cost tracking
//! - **Structured Logging**: JSON-formatted logs with trace correlation
//! - **Prometheus Export**: Metrics endpoint for scraping

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod logging;
pub mod metrics;
pub mod research;
pub mod tracing_config;

use std::sync::Arc;

use parking_lot::RwLock;

pub use logging::init_logging;
pub use metrics::{ActiveRequestGuard, InferenceMetrics, MetricsCollector, PrometheusRegistry, Timer};
pub use research::{
    ConsoleListener, EventListener, EventType, GlobalStats, JsonFileListener, ModelFamily,
    ResearchEvent, ResearchTracker, SessionId, SessionStats,
};
pub use tracing_config::{
    create_tracer, init_tracing, AgentSpan, InferenceSpan, LLMSpan, LLMSpanBuilder,
    RetrievalSpan, ToolSpan, TracingConfig, TracingGuard,
};

/// Global telemetry state.
static TELEMETRY: RwLock<Option<Arc<Telemetry>>> = RwLock::new(None);

/// Central telemetry manager.
pub struct Telemetry {
    /// Inference metrics collector.
    pub metrics: MetricsCollector,
}

impl Telemetry {
    /// Initializes global telemetry.
    pub fn init(config: TelemetryConfig) -> Arc<Self> {
        let telemetry = Arc::new(Self {
            metrics: MetricsCollector::new(&config),
        });

        *TELEMETRY.write() = Some(Arc::clone(&telemetry));
        telemetry
    }

    /// Returns the global telemetry instance.
    #[must_use]
    pub fn global() -> Option<Arc<Self>> {
        TELEMETRY.read().clone()
    }
}

/// Configuration for telemetry.
#[derive(Debug, Clone, Default)]
pub struct TelemetryConfig {
    /// Service name for tracing.
    pub service_name: String,
    /// OTLP endpoint for traces.
    pub otlp_endpoint: Option<String>,
    /// Enable Prometheus metrics endpoint.
    pub prometheus_enabled: bool,
    /// Prometheus listen address.
    pub prometheus_addr: Option<String>,
    /// Log level.
    pub log_level: String,
    /// Enable JSON logging.
    pub json_logs: bool,
}

impl TelemetryConfig {
    /// Creates a new telemetry configuration.
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            otlp_endpoint: None,
            prometheus_enabled: false,
            prometheus_addr: None,
            log_level: "info".to_string(),
            json_logs: false,
        }
    }

    /// Sets the OTLP endpoint.
    #[must_use]
    pub fn with_otlp(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Enables Prometheus metrics.
    #[must_use]
    pub fn with_prometheus(mut self, addr: impl Into<String>) -> Self {
        self.prometheus_enabled = true;
        self.prometheus_addr = Some(addr.into());
        self
    }

    /// Sets the log level.
    #[must_use]
    pub fn with_log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = level.into();
        self
    }

    /// Enables JSON logging.
    #[must_use]
    pub fn with_json_logs(mut self) -> Self {
        self.json_logs = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // TelemetryConfig tests
    // ==========================================================================

    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert!(config.service_name.is_empty());
        assert!(config.otlp_endpoint.is_none());
        assert!(!config.prometheus_enabled);
        assert!(config.prometheus_addr.is_none());
        assert!(config.log_level.is_empty());
        assert!(!config.json_logs);
    }

    #[test]
    fn test_telemetry_config_new() {
        let config = TelemetryConfig::new("infernum");
        assert_eq!(config.service_name, "infernum");
        assert!(config.otlp_endpoint.is_none());
        assert!(!config.prometheus_enabled);
        assert_eq!(config.log_level, "info");
        assert!(!config.json_logs);
    }

    #[test]
    fn test_telemetry_config_with_otlp() {
        let config = TelemetryConfig::new("test")
            .with_otlp("http://localhost:4317");

        assert_eq!(config.otlp_endpoint, Some("http://localhost:4317".to_string()));
    }

    #[test]
    fn test_telemetry_config_with_prometheus() {
        let config = TelemetryConfig::new("test")
            .with_prometheus("0.0.0.0:9090");

        assert!(config.prometheus_enabled);
        assert_eq!(config.prometheus_addr, Some("0.0.0.0:9090".to_string()));
    }

    #[test]
    fn test_telemetry_config_with_log_level() {
        let config = TelemetryConfig::new("test")
            .with_log_level("debug");

        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn test_telemetry_config_with_json_logs() {
        let config = TelemetryConfig::new("test")
            .with_json_logs();

        assert!(config.json_logs);
    }

    #[test]
    fn test_telemetry_config_builder_chain() {
        let config = TelemetryConfig::new("my-service")
            .with_otlp("http://jaeger:4317")
            .with_prometheus("0.0.0.0:8080")
            .with_log_level("trace")
            .with_json_logs();

        assert_eq!(config.service_name, "my-service");
        assert_eq!(config.otlp_endpoint, Some("http://jaeger:4317".to_string()));
        assert!(config.prometheus_enabled);
        assert_eq!(config.prometheus_addr, Some("0.0.0.0:8080".to_string()));
        assert_eq!(config.log_level, "trace");
        assert!(config.json_logs);
    }

    #[test]
    fn test_telemetry_config_clone() {
        let config = TelemetryConfig::new("clone-test")
            .with_otlp("http://localhost:4317")
            .with_json_logs();

        let cloned = config.clone();
        assert_eq!(cloned.service_name, config.service_name);
        assert_eq!(cloned.otlp_endpoint, config.otlp_endpoint);
        assert_eq!(cloned.json_logs, config.json_logs);
    }

    #[test]
    fn test_telemetry_config_debug() {
        let config = TelemetryConfig::new("debug-test");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("TelemetryConfig"));
        assert!(debug_str.contains("debug-test"));
    }

    // ==========================================================================
    // Telemetry tests
    // ==========================================================================

    #[test]
    fn test_telemetry_init() {
        let config = TelemetryConfig::new("test-service");
        let telemetry = Telemetry::init(config);

        // Verify the telemetry instance was created
        assert_eq!(telemetry.metrics.inference().requests(), 0);

        // Verify global access works
        let global = Telemetry::global();
        assert!(global.is_some());
    }

    #[test]
    fn test_telemetry_metrics_access() {
        let config = TelemetryConfig::new("metrics-test");
        let telemetry = Telemetry::init(config);

        // Record some metrics
        telemetry.metrics.inference().record_request(100, 50);
        telemetry.metrics.inference().record_request(200, 100);
        telemetry.metrics.record_error("chat", "test-model", "timeout");

        assert_eq!(telemetry.metrics.inference().requests(), 2);
        assert_eq!(telemetry.metrics.inference().prompt_tokens(), 300);
        assert_eq!(telemetry.metrics.inference().tokens_generated(), 150);
        assert_eq!(telemetry.metrics.inference().errors(), 1);
    }

    #[test]
    fn test_telemetry_prometheus_render() {
        let config = TelemetryConfig::new("prometheus-test")
            .with_prometheus("0.0.0.0:9090");
        let telemetry = Telemetry::init(config);

        telemetry.metrics.record_chat_request(50, 25, 0.1, "test-model");

        let output = telemetry.metrics.render_prometheus();
        assert!(output.contains("infernum_requests_total"));
        assert!(output.contains("infernum_active_requests"));
        assert!(output.contains("infernum_model_loaded"));
    }
}
