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

pub mod logging;
pub mod metrics;
pub mod tracing_config;

use std::sync::Arc;

use parking_lot::RwLock;

pub use logging::init_logging;
pub use metrics::{InferenceMetrics, MetricsCollector};

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
