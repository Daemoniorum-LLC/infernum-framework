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

    /// Configures for Jaeger with OTLP (modern Jaeger accepts OTLP natively).
    ///
    /// Default Jaeger OTLP endpoint is `http://localhost:4317`.
    #[must_use]
    pub fn with_jaeger(mut self, host: impl Into<String>, port: u16) -> Self {
        self.otlp_endpoint = Some(format!("http://{}:{}", host.into(), port));
        self
    }

    /// Configures for Jaeger with default OTLP port (4317).
    #[must_use]
    pub fn with_jaeger_default(self) -> Self {
        self.with_jaeger("localhost", 4317)
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

/// A span for inference operations.
#[derive(Debug, Clone)]
pub struct InferenceSpan {
    /// Model identifier.
    pub model_id: String,
    /// Request ID.
    pub request_id: String,
    /// Input token count.
    pub input_tokens: u32,
    /// Output token count.
    pub output_tokens: u32,
    /// Temperature used.
    pub temperature: f32,
    /// Top-p (nucleus) sampling.
    pub top_p: f32,
    /// Time to first token in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Total generation time in milliseconds.
    pub total_time_ms: Option<f64>,
    /// Batch size (for batched inference).
    pub batch_size: Option<u32>,
    /// Whether streaming was used.
    pub streaming: bool,
    /// Cache hit for prompt tokens.
    pub cache_hit_tokens: Option<u32>,
}

impl InferenceSpan {
    /// Creates a new inference span.
    #[must_use]
    pub fn new(model_id: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            request_id: request_id.into(),
            input_tokens: 0,
            output_tokens: 0,
            temperature: 1.0,
            top_p: 1.0,
            ttft_ms: None,
            total_time_ms: None,
            batch_size: None,
            streaming: false,
            cache_hit_tokens: None,
        }
    }

    /// Converts to OpenTelemetry attributes.
    #[must_use]
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("llm.model_id", self.model_id.clone()),
            KeyValue::new("llm.request_id", self.request_id.clone()),
            KeyValue::new("llm.input_tokens", self.input_tokens as i64),
            KeyValue::new("llm.output_tokens", self.output_tokens as i64),
            KeyValue::new("llm.temperature", self.temperature as f64),
            KeyValue::new("llm.top_p", self.top_p as f64),
            KeyValue::new("llm.streaming", self.streaming),
        ];

        if let Some(ttft) = self.ttft_ms {
            attrs.push(KeyValue::new("llm.ttft_ms", ttft));
        }
        if let Some(total) = self.total_time_ms {
            attrs.push(KeyValue::new("llm.total_time_ms", total));
            if self.output_tokens > 0 && total > 0.0 {
                attrs.push(KeyValue::new(
                    "llm.tokens_per_second",
                    (self.output_tokens as f64 / total) * 1000.0,
                ));
            }
        }
        if let Some(batch) = self.batch_size {
            attrs.push(KeyValue::new("llm.batch_size", batch as i64));
        }
        if let Some(cache_hit) = self.cache_hit_tokens {
            attrs.push(KeyValue::new("llm.cache_hit_tokens", cache_hit as i64));
        }

        attrs
    }
}

/// A span for RAG retrieval operations.
#[derive(Debug, Clone)]
pub struct RetrievalSpan {
    /// Query text or identifier.
    pub query_id: String,
    /// Number of documents retrieved.
    pub num_retrieved: u32,
    /// Number of documents after reranking.
    pub num_reranked: Option<u32>,
    /// Top similarity score.
    pub top_score: Option<f32>,
    /// Retrieval latency in milliseconds.
    pub retrieval_time_ms: f64,
    /// Embedding time in milliseconds.
    pub embedding_time_ms: Option<f64>,
    /// Whether hybrid search was used.
    pub hybrid_search: bool,
    /// Collection/index name.
    pub collection: Option<String>,
}

impl RetrievalSpan {
    /// Creates a new retrieval span.
    #[must_use]
    pub fn new(query_id: impl Into<String>) -> Self {
        Self {
            query_id: query_id.into(),
            num_retrieved: 0,
            num_reranked: None,
            top_score: None,
            retrieval_time_ms: 0.0,
            embedding_time_ms: None,
            hybrid_search: false,
            collection: None,
        }
    }

    /// Converts to OpenTelemetry attributes.
    #[must_use]
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("rag.query_id", self.query_id.clone()),
            KeyValue::new("rag.num_retrieved", self.num_retrieved as i64),
            KeyValue::new("rag.retrieval_time_ms", self.retrieval_time_ms),
            KeyValue::new("rag.hybrid_search", self.hybrid_search),
        ];

        if let Some(reranked) = self.num_reranked {
            attrs.push(KeyValue::new("rag.num_reranked", reranked as i64));
        }
        if let Some(score) = self.top_score {
            attrs.push(KeyValue::new("rag.top_score", score as f64));
        }
        if let Some(embed_time) = self.embedding_time_ms {
            attrs.push(KeyValue::new("rag.embedding_time_ms", embed_time));
        }
        if let Some(ref collection) = self.collection {
            attrs.push(KeyValue::new("rag.collection", collection.clone()));
        }

        attrs
    }
}

/// A span for tool execution in agents.
#[derive(Debug, Clone)]
pub struct ToolSpan {
    /// Tool name.
    pub tool_name: String,
    /// Agent ID.
    pub agent_id: String,
    /// Whether the tool succeeded.
    pub success: bool,
    /// Execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Error message if failed.
    pub error: Option<String>,
    /// Tool parameters (sanitized).
    pub params_summary: Option<String>,
    /// Risk level of the tool.
    pub risk_level: Option<String>,
}

impl ToolSpan {
    /// Creates a new tool span.
    #[must_use]
    pub fn new(tool_name: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            agent_id: agent_id.into(),
            success: false,
            execution_time_ms: 0.0,
            error: None,
            params_summary: None,
            risk_level: None,
        }
    }

    /// Marks as successful.
    pub fn mark_success(&mut self, execution_time_ms: f64) {
        self.success = true;
        self.execution_time_ms = execution_time_ms;
    }

    /// Marks as failed.
    pub fn mark_failure(&mut self, error: impl Into<String>, execution_time_ms: f64) {
        self.success = false;
        self.error = Some(error.into());
        self.execution_time_ms = execution_time_ms;
    }

    /// Converts to OpenTelemetry attributes.
    #[must_use]
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("agent.tool_name", self.tool_name.clone()),
            KeyValue::new("agent.agent_id", self.agent_id.clone()),
            KeyValue::new("agent.tool_success", self.success),
            KeyValue::new("agent.execution_time_ms", self.execution_time_ms),
        ];

        if let Some(ref error) = self.error {
            attrs.push(KeyValue::new("agent.tool_error", error.clone()));
        }
        if let Some(ref params) = self.params_summary {
            attrs.push(KeyValue::new("agent.params_summary", params.clone()));
        }
        if let Some(ref risk) = self.risk_level {
            attrs.push(KeyValue::new("agent.risk_level", risk.clone()));
        }

        attrs
    }
}

/// A span for agent operations.
#[derive(Debug, Clone)]
pub struct AgentSpan {
    /// Agent ID.
    pub agent_id: String,
    /// Agent name/persona.
    pub agent_name: String,
    /// Objective or task.
    pub objective: String,
    /// Number of steps executed.
    pub steps_executed: u32,
    /// Number of tool calls made.
    pub tool_calls: u32,
    /// Total tokens used (input + output).
    pub total_tokens: u32,
    /// Total execution time in milliseconds.
    pub total_time_ms: f64,
    /// Whether the task completed successfully.
    pub success: bool,
    /// Planning strategy used.
    pub planning_strategy: Option<String>,
}

impl AgentSpan {
    /// Creates a new agent span.
    #[must_use]
    pub fn new(
        agent_id: impl Into<String>,
        agent_name: impl Into<String>,
        objective: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            agent_name: agent_name.into(),
            objective: objective.into(),
            steps_executed: 0,
            tool_calls: 0,
            total_tokens: 0,
            total_time_ms: 0.0,
            success: false,
            planning_strategy: None,
        }
    }

    /// Converts to OpenTelemetry attributes.
    #[must_use]
    pub fn to_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("agent.id", self.agent_id.clone()),
            KeyValue::new("agent.name", self.agent_name.clone()),
            KeyValue::new("agent.objective", self.objective.clone()),
            KeyValue::new("agent.steps_executed", self.steps_executed as i64),
            KeyValue::new("agent.tool_calls", self.tool_calls as i64),
            KeyValue::new("agent.total_tokens", self.total_tokens as i64),
            KeyValue::new("agent.total_time_ms", self.total_time_ms),
            KeyValue::new("agent.success", self.success),
        ];

        if let Some(ref strategy) = self.planning_strategy {
            attrs.push(KeyValue::new("agent.planning_strategy", strategy.clone()));
        }

        attrs
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

    #[test]
    fn test_tracing_config_jaeger() {
        let config = TracingConfig::new("infernum").with_jaeger("jaeger.local", 4317);

        assert_eq!(
            config.otlp_endpoint,
            Some("http://jaeger.local:4317".to_string())
        );
    }

    #[test]
    fn test_tracing_config_jaeger_default() {
        let config = TracingConfig::new("infernum").with_jaeger_default();

        assert_eq!(
            config.otlp_endpoint,
            Some("http://localhost:4317".to_string())
        );
    }

    #[test]
    fn test_inference_span() {
        let mut span = InferenceSpan::new("llama-3.2", "req-123");
        span.input_tokens = 100;
        span.output_tokens = 50;
        span.temperature = 0.8;
        span.streaming = true;
        span.ttft_ms = Some(150.0);
        span.total_time_ms = Some(500.0);

        let attrs = span.to_attributes();
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "llm.model_id"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "llm.request_id"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "llm.streaming"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "llm.tokens_per_second"));
    }

    #[test]
    fn test_retrieval_span() {
        let mut span = RetrievalSpan::new("query-456");
        span.num_retrieved = 10;
        span.num_reranked = Some(5);
        span.top_score = Some(0.92);
        span.retrieval_time_ms = 45.0;
        span.collection = Some("documents".to_string());

        let attrs = span.to_attributes();
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "rag.query_id"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "rag.num_retrieved"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "rag.top_score"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "rag.collection"));
    }

    #[test]
    fn test_tool_span_success() {
        let mut span = ToolSpan::new("calculator", "agent-789");
        span.mark_success(25.0);
        span.risk_level = Some("low".to_string());

        let attrs = span.to_attributes();
        assert!(span.success);
        assert_eq!(span.execution_time_ms, 25.0);
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.tool_name"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.risk_level"));
    }

    #[test]
    fn test_tool_span_failure() {
        let mut span = ToolSpan::new("web_search", "agent-789");
        span.mark_failure("Connection timeout", 5000.0);

        assert!(!span.success);
        assert_eq!(span.error, Some("Connection timeout".to_string()));
        assert_eq!(span.execution_time_ms, 5000.0);
    }

    #[test]
    fn test_agent_span() {
        let mut span = AgentSpan::new("agent-001", "research-assistant", "Find latest papers");
        span.steps_executed = 5;
        span.tool_calls = 3;
        span.total_tokens = 2500;
        span.total_time_ms = 12000.0;
        span.success = true;
        span.planning_strategy = Some("ReAct".to_string());

        let attrs = span.to_attributes();
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.id"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.name"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.objective"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.planning_strategy"));
        assert!(attrs.iter().any(|kv| kv.key.as_str() == "agent.success"));
    }
}
