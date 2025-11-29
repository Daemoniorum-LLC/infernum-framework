//! Distributed tracing configuration.

use crate::TelemetryConfig;

/// Initializes distributed tracing.
///
/// # Errors
///
/// Returns an error if tracing cannot be initialized.
pub fn init_tracing(config: &TelemetryConfig) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(endpoint) = &config.otlp_endpoint {
        tracing::info!(endpoint = %endpoint, "Initializing OTLP tracing");
        // TODO: Implement actual OTLP setup with opentelemetry crates
    }

    Ok(())
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
        }
    }

    /// Records token counts.
    pub fn record_tokens(&mut self, input: u32, output: u32) {
        self.input_tokens = input;
        self.output_tokens = output;
    }
}
