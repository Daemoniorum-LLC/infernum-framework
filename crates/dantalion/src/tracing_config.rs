//! Distributed tracing configuration.

use std::time::Duration;

use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::{
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
    Resource,
};

use crate::TelemetryConfig;

/// Guard for the tracing provider that shuts down on drop.
pub struct TracingGuard {
    provider: Option<SdkTracerProvider>,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            if let Err(e) = provider.shutdown() {
                tracing::warn!("Error shutting down tracer provider: {:?}", e);
            }
        }
    }
}

/// Initializes distributed tracing with OpenTelemetry.
///
/// # Errors
///
/// Returns an error if tracing cannot be initialized.
pub fn init_tracing(
    config: &TelemetryConfig,
) -> Result<TracingGuard, Box<dyn std::error::Error + Send + Sync>> {
    let provider = if let Some(endpoint) = &config.otlp_endpoint {
        tracing::info!(endpoint = %endpoint, "Initializing OTLP tracing");

        #[cfg(feature = "otlp")]
        {
            use opentelemetry_otlp::{SpanExporter, WithExportConfig};

            // Build OTLP exporter with tonic
            let exporter = SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .with_timeout(Duration::from_secs(10))
                .build()?;

            // Build resource with service info
            let resource = Resource::builder()
                .with_service_name(config.service_name.clone())
                .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
                .build();

            // Build the provider
            let provider = SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_sampler(Sampler::AlwaysOn)
                .with_id_generator(RandomIdGenerator::default())
                .with_resource(resource)
                .build();

            // Set global tracer provider
            let _ = global::set_tracer_provider(provider.clone());

            tracing::info!(
                service = %config.service_name,
                endpoint = %endpoint,
                "OTLP tracing initialized"
            );

            Some(provider)
        }

        #[cfg(not(feature = "otlp"))]
        {
            tracing::warn!("OTLP feature not enabled, tracing will be local only");
            None
        }
    } else {
        tracing::debug!("No OTLP endpoint configured, using local tracing only");
        None
    };

    Ok(TracingGuard { provider })
}

/// Creates a tracer for a specific component.
#[must_use]
pub fn create_tracer(component: &str) -> opentelemetry::global::BoxedTracer {
    global::tracer(component.to_string())
}

/// A span for LLM-specific tracing.
pub struct LLMSpan {
    /// Model identifier.
    pub model_id: String,
    /// Input token count.
    pub input_tokens: u32,
    /// Output token count.
    pub output_tokens: u32,
    /// Temperature used.
    pub temperature: f32,
    /// Time to first token in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Total generation time in milliseconds.
    pub total_time_ms: Option<f64>,
    /// Tokens per second.
    pub tokens_per_second: Option<f64>,
}

impl LLMSpan {
    /// Creates a new LLM span.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            input_tokens: 0,
            output_tokens: 0,
            temperature: 1.0,
            ttft_ms: None,
            total_time_ms: None,
            tokens_per_second: None,
        }
    }

    /// Records token counts.
    pub fn record_tokens(&mut self, input: u32, output: u32) {
        self.input_tokens = input;
        self.output_tokens = output;
    }

    /// Records timing metrics.
    pub fn record_timing(&mut self, ttft_ms: f64, total_time_ms: f64) {
        self.ttft_ms = Some(ttft_ms);
        self.total_time_ms = Some(total_time_ms);

        if total_time_ms > 0.0 && self.output_tokens > 0 {
            self.tokens_per_second = Some((self.output_tokens as f64 / total_time_ms) * 1000.0);
        }
    }

    /// Converts to OpenTelemetry attributes.
    #[must_use]
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("llm.model_id", self.model_id.clone()),
            KeyValue::new("llm.input_tokens", self.input_tokens as i64),
            KeyValue::new("llm.output_tokens", self.output_tokens as i64),
            KeyValue::new("llm.temperature", self.temperature as f64),
        ];

        if let Some(ttft) = self.ttft_ms {
            attrs.push(KeyValue::new("llm.ttft_ms", ttft));
        }

        if let Some(total) = self.total_time_ms {
            attrs.push(KeyValue::new("llm.total_time_ms", total));
        }

        if let Some(tps) = self.tokens_per_second {
            attrs.push(KeyValue::new("llm.tokens_per_second", tps));
        }

        attrs
    }
}

/// Builder for LLM spans with fluent API.
pub struct LLMSpanBuilder {
    span: LLMSpan,
}

impl LLMSpanBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            span: LLMSpan::new(model_id),
        }
    }

    /// Sets input tokens.
    #[must_use]
    pub fn input_tokens(mut self, tokens: u32) -> Self {
        self.span.input_tokens = tokens;
        self
    }

    /// Sets output tokens.
    #[must_use]
    pub fn output_tokens(mut self, tokens: u32) -> Self {
        self.span.output_tokens = tokens;
        self
    }

    /// Sets temperature.
    #[must_use]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.span.temperature = temp;
        self
    }

    /// Sets time to first token.
    #[must_use]
    pub fn ttft_ms(mut self, ttft: f64) -> Self {
        self.span.ttft_ms = Some(ttft);
        self
    }

    /// Sets total time.
    #[must_use]
    pub fn total_time_ms(mut self, total: f64) -> Self {
        self.span.total_time_ms = Some(total);
        self
    }

    /// Builds the span.
    #[must_use]
    pub fn build(mut self) -> LLMSpan {
        // Calculate tokens per second if we have the data
        if let (Some(total), tokens) = (self.span.total_time_ms, self.span.output_tokens) {
            if total > 0.0 && tokens > 0 {
                self.span.tokens_per_second = Some((tokens as f64 / total) * 1000.0);
            }
        }
        self.span
    }
}

/// Configuration for distributed tracing.
#[derive(Debug, Clone, Default)]
pub struct TracingConfig {
    /// Whether tracing is enabled.
    pub enabled: bool,
    /// OTLP endpoint URL.
    pub otlp_endpoint: Option<String>,
    /// Service name for traces.
    pub service_name: String,
    /// Sampling ratio (0.0 - 1.0).
    pub sampling_ratio: f64,
    /// Whether to propagate trace context.
    pub propagate_context: bool,
}

impl TracingConfig {
    /// Creates a new tracing configuration.
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            enabled: true,
            otlp_endpoint: None,
            service_name: service_name.into(),
            sampling_ratio: 1.0,
            propagate_context: true,
        }
    }

    /// Sets the OTLP endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Sets the sampling ratio.
    #[must_use]
    pub fn with_sampling_ratio(mut self, ratio: f64) -> Self {
        self.sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Disables tracing.
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_span_builder() {
        let span = LLMSpanBuilder::new("test-model")
            .input_tokens(100)
            .output_tokens(50)
            .temperature(0.7)
            .ttft_ms(25.0)
            .total_time_ms(500.0)
            .build();

        assert_eq!(span.model_id, "test-model");
        assert_eq!(span.input_tokens, 100);
        assert_eq!(span.output_tokens, 50);
        assert_eq!(span.temperature, 0.7);
        assert!(span.tokens_per_second.is_some());

        // 50 tokens / 500ms * 1000 = 100 tps
        let tps = span.tokens_per_second.unwrap();
        assert!((tps - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_llm_span_attributes() {
        let span = LLMSpan::new("gpt-4");
        let attrs = span.to_attributes();

        assert!(attrs.iter().any(|kv| kv.key.as_str() == "llm.model_id"));
    }

    #[test]
    fn test_tracing_config() {
        let config = TracingConfig::new("infernum")
            .with_endpoint("http://localhost:4317")
            .with_sampling_ratio(0.5);

        assert!(config.enabled);
        assert_eq!(
            config.otlp_endpoint,
            Some("http://localhost:4317".to_string())
        );
        assert_eq!(config.sampling_ratio, 0.5);
    }
}
