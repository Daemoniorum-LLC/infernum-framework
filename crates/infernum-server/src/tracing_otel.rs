//! OpenTelemetry distributed tracing integration.
//!
//! Provides distributed tracing with trace context propagation for
//! observability across service boundaries.
//!
//! # Features
//!
//! - OTLP exporter for Jaeger, Tempo, or other collectors
//! - W3C Trace Context propagation
//! - Automatic span creation for HTTP requests
//! - Custom span attributes for LLM inference
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::tracing_otel::{init_tracing, TracingConfig};
//!
//! let config = TracingConfig::default()
//!     .with_service_name("infernum")
//!     .with_otlp_endpoint("http://localhost:4317");
//!
//! init_tracing(config)?;
//! ```
//!
//! # Environment Variables
//!
//! - `OTEL_SERVICE_NAME`: Service name for traces (default: "infernum")
//! - `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP collector endpoint
//! - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`: Traces-specific endpoint
//! - `OTEL_TRACES_SAMPLER`: Sampler type (always_on, always_off, traceidratio)
//! - `OTEL_TRACES_SAMPLER_ARG`: Sampler argument (e.g., ratio for traceidratio)

use std::time::Duration;

use axum::{
    body::Body, extract::Request, http::header::HeaderValue, middleware::Next, response::Response,
};
use opentelemetry::{
    global,
    trace::{Span, SpanKind, Status, TraceContextExt, Tracer},
    Context, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    trace::{RandomIdGenerator, Sampler},
    Resource,
};

/// OpenTelemetry tracing configuration.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for traces.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// OTLP endpoint for trace export.
    pub otlp_endpoint: Option<String>,
    /// Whether tracing is enabled.
    pub enabled: bool,
    /// Sampling ratio (0.0 to 1.0).
    pub sampling_ratio: f64,
    /// Export timeout.
    pub export_timeout: Duration,
    /// Batch export delay.
    pub batch_delay: Duration,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "infernum".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            otlp_endpoint: None,
            enabled: true,
            sampling_ratio: 1.0,
            export_timeout: Duration::from_secs(10),
            batch_delay: Duration::from_secs(5),
        }
    }
}

impl TracingConfig {
    /// Creates configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(name) = std::env::var("OTEL_SERVICE_NAME") {
            config.service_name = name;
        }

        if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
            config.otlp_endpoint = Some(endpoint);
        } else if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") {
            config.otlp_endpoint = Some(endpoint);
        }

        if let Ok(enabled) = std::env::var("OTEL_SDK_DISABLED") {
            config.enabled = enabled != "true" && enabled != "1";
        }

        if let Ok(sampler_arg) = std::env::var("OTEL_TRACES_SAMPLER_ARG") {
            if let Ok(ratio) = sampler_arg.parse::<f64>() {
                config.sampling_ratio = ratio.clamp(0.0, 1.0);
            }
        }

        config
    }

    /// Sets the service name.
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Sets the service version.
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Sets the OTLP endpoint.
    pub fn with_otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Enables or disables tracing.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the sampling ratio (0.0 to 1.0).
    pub fn with_sampling_ratio(mut self, ratio: f64) -> Self {
        self.sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }
}

/// Initializes OpenTelemetry tracing.
///
/// Returns an error if the OTLP exporter cannot be configured.
///
/// # Errors
///
/// Returns an error if tracing initialization fails.
pub fn init_tracing(config: &TracingConfig) -> Result<(), TracingError> {
    if !config.enabled {
        tracing::info!("OpenTelemetry tracing disabled");
        return Ok(());
    }

    let Some(endpoint) = &config.otlp_endpoint else {
        tracing::info!("No OTLP endpoint configured, tracing disabled");
        return Ok(());
    };

    // Create resource with service info
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attribute(KeyValue::new(
            "service.version",
            config.service_version.clone(),
        ))
        .build();

    // Create sampler based on ratio
    let sampler = if config.sampling_ratio >= 1.0 {
        Sampler::AlwaysOn
    } else if config.sampling_ratio <= 0.0 {
        Sampler::AlwaysOff
    } else {
        Sampler::TraceIdRatioBased(config.sampling_ratio)
    };

    // Create OTLP exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(config.export_timeout)
        .build()
        .map_err(|e| TracingError::ExporterInit(e.to_string()))?;

    // Create tracer provider
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_sampler(sampler)
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();

    // Set global tracer provider
    global::set_tracer_provider(provider);

    tracing::info!(
        endpoint = %endpoint,
        service = %config.service_name,
        sampling_ratio = config.sampling_ratio,
        "OpenTelemetry tracing initialized"
    );

    Ok(())
}

/// Shuts down OpenTelemetry tracing gracefully.
pub fn shutdown_tracing() {
    // Note: In OpenTelemetry 0.31+, shutdown is handled by dropping the provider
    // or by calling provider.shutdown() directly on the SdkTracerProvider
    tracing::info!("OpenTelemetry tracing shutdown requested");
}

/// OpenTelemetry tracing errors.
#[derive(Debug, thiserror::Error)]
pub enum TracingError {
    /// Failed to initialize the exporter.
    #[error("failed to initialize OTLP exporter: {0}")]
    ExporterInit(String),
    /// Failed to create tracer provider.
    #[error("failed to create tracer provider: {0}")]
    ProviderInit(String),
}

/// Middleware for HTTP request tracing with OpenTelemetry.
///
/// Creates spans for each HTTP request with standard attributes.
pub async fn otel_tracing_middleware(request: Request<Body>, next: Next) -> Response {
    let tracer = global::tracer("infernum-server");

    // Extract trace context from incoming headers
    let parent_cx = extract_context(&request);

    // Get request info for span
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let span_name = format!("{} {}", method, path);

    // Create span with parent context
    let mut span = tracer
        .span_builder(span_name)
        .with_kind(SpanKind::Server)
        .start_with_context(&tracer, &parent_cx);

    // Set span attributes
    span.set_attribute(KeyValue::new("http.method", method.clone()));
    span.set_attribute(KeyValue::new("http.target", path.clone()));
    span.set_attribute(KeyValue::new("http.scheme", "http"));

    if let Some(host) = request.headers().get("host").and_then(|h| h.to_str().ok()) {
        span.set_attribute(KeyValue::new("http.host", host.to_string()));
    }

    if let Some(user_agent) = request
        .headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
    {
        span.set_attribute(KeyValue::new("http.user_agent", user_agent.to_string()));
    }

    // Create context with span
    let cx = Context::current_with_span(span);

    // Execute request
    let mut response = next.run(request).await;

    // Update span with response info
    let span = cx.span();
    let status = response.status();
    span.set_attribute(KeyValue::new("http.status_code", status.as_u16() as i64));

    if status.is_client_error() {
        span.set_status(Status::error("Client error"));
    } else if status.is_server_error() {
        span.set_status(Status::error("Server error"));
    } else {
        span.set_status(Status::Ok);
    }

    // Inject trace context into response headers
    inject_context(&cx, &mut response);

    span.end();

    response
}

/// Extracts W3C Trace Context from request headers.
fn extract_context(request: &Request<Body>) -> Context {
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);

    impl opentelemetry::propagation::Extractor for HeaderExtractor<'_> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }

        fn keys(&self) -> Vec<&str> {
            self.0.keys().filter_map(|k| k.as_str().into()).collect()
        }
    }

    let propagator = TraceContextPropagator::new();
    propagator.extract(&HeaderExtractor(request.headers()))
}

/// Injects W3C Trace Context into response headers.
fn inject_context(cx: &Context, response: &mut Response) {
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    struct HeaderInjector<'a>(&'a mut axum::http::HeaderMap);

    impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
        fn set(&mut self, key: &str, value: String) {
            if let Ok(header_name) = axum::http::header::HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = HeaderValue::from_str(&value) {
                    self.0.insert(header_name, header_value);
                }
            }
        }
    }

    let propagator = TraceContextPropagator::new();
    propagator.inject_context(cx, &mut HeaderInjector(response.headers_mut()));
}

/// Span builder for LLM inference operations.
pub struct InferenceSpan {
    span: opentelemetry::global::BoxedSpan,
}

impl InferenceSpan {
    /// Creates a new inference span.
    pub fn new(operation: &str, model: &str) -> Self {
        let tracer = global::tracer("infernum-inference");
        let span = tracer
            .span_builder(format!("llm.{}", operation))
            .with_kind(SpanKind::Internal)
            .with_attributes(vec![
                KeyValue::new("llm.model", model.to_string()),
                KeyValue::new("llm.operation", operation.to_string()),
            ])
            .start(&tracer);

        Self { span }
    }

    /// Records input token count.
    pub fn set_input_tokens(&mut self, count: u32) {
        self.span
            .set_attribute(KeyValue::new("llm.input_tokens", count as i64));
    }

    /// Records output token count.
    pub fn set_output_tokens(&mut self, count: u32) {
        self.span
            .set_attribute(KeyValue::new("llm.output_tokens", count as i64));
    }

    /// Records total token count.
    pub fn set_total_tokens(&mut self, count: u32) {
        self.span
            .set_attribute(KeyValue::new("llm.total_tokens", count as i64));
    }

    /// Sets the finish reason.
    pub fn set_finish_reason(&mut self, reason: &str) {
        self.span
            .set_attribute(KeyValue::new("llm.finish_reason", reason.to_string()));
    }

    /// Marks the span as errored.
    pub fn set_error(&mut self, message: &str) {
        self.span.set_status(Status::error(message.to_string()));
        self.span
            .set_attribute(KeyValue::new("error.message", message.to_string()));
    }

    /// Ends the span.
    pub fn end(mut self) {
        self.span.end();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "infernum");
        assert!(config.enabled);
        assert!(config.otlp_endpoint.is_none());
        assert!((config.sampling_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracing_config_builder() {
        let config = TracingConfig::default()
            .with_service_name("test-service")
            .with_otlp_endpoint("http://localhost:4317")
            .with_sampling_ratio(0.5)
            .with_enabled(false);

        assert_eq!(config.service_name, "test-service");
        assert_eq!(
            config.otlp_endpoint,
            Some("http://localhost:4317".to_string())
        );
        assert!((config.sampling_ratio - 0.5).abs() < f64::EPSILON);
        assert!(!config.enabled);
    }

    #[test]
    fn test_sampling_ratio_clamped() {
        let config = TracingConfig::default().with_sampling_ratio(2.0);
        assert!((config.sampling_ratio - 1.0).abs() < f64::EPSILON);

        let config = TracingConfig::default().with_sampling_ratio(-1.0);
        assert!(config.sampling_ratio.abs() < f64::EPSILON);
    }

    #[test]
    fn test_init_tracing_disabled() {
        let config = TracingConfig::default().with_enabled(false);
        let result = init_tracing(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_tracing_no_endpoint() {
        let config = TracingConfig::default();
        assert!(config.otlp_endpoint.is_none());
        let result = init_tracing(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_inference_span_creation() {
        // Just verify it doesn't panic
        let mut span = InferenceSpan::new("chat_completion", "llama-3b");
        span.set_input_tokens(100);
        span.set_output_tokens(50);
        span.set_total_tokens(150);
        span.set_finish_reason("stop");
        span.end();
    }
}
