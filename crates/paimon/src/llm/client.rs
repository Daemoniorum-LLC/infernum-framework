//! Core LLM client trait and types.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Request failed.
    #[error("Request failed: {0}")]
    Request(String),

    /// Response parsing failed.
    #[error("Failed to parse response: {0}")]
    Parse(String),

    /// Provider not available.
    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),

    /// Rate limited.
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        /// Milliseconds to wait before retrying.
        retry_after_ms: u64,
    },

    /// Timeout.
    #[error("Request timed out after {0}ms")]
    Timeout(u64),

    /// Context too long.
    #[error("Context too long: {tokens} tokens exceeds limit of {limit}")]
    ContextTooLong {
        /// Number of tokens in the request.
        tokens: usize,
        /// Maximum allowed tokens.
        limit: usize,
    },

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type for LLM operations.
pub type Result<T> = std::result::Result<T, LlmError>;

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions).
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: MessageRole,
    /// Content of the message.
    pub content: String,
}

impl Message {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// Options for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,

    /// Temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,

    /// Timeout in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: Some(4096),
            temperature: Some(0.7),
            top_p: None,
            stop: Vec::new(),
            timeout_ms: Some(60000),
        }
    }
}

/// Request for text generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Conversation messages.
    pub messages: Vec<Message>,

    /// Model to use (provider-specific).
    pub model: Option<String>,

    /// Generation options.
    pub options: GenerateOptions,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            model: None,
            options: GenerateOptions::default(),
        }
    }
}

impl GenerateRequest {
    /// Creates a new request with the given messages.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// Creates a simple request with a single user message.
    pub fn simple(content: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(content)],
            ..Default::default()
        }
    }

    /// Sets the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.options.max_tokens = Some(max_tokens);
        self
    }

    /// Sets temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.options.temperature = Some(temperature);
        self
    }

    /// Adds a system message at the beginning.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.messages.insert(0, Message::system(system));
        self
    }
}

/// Response from text generation.
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    /// Generated content.
    pub content: String,

    /// Model used.
    pub model: String,

    /// Usage statistics.
    pub usage: Option<Usage>,

    /// Finish reason.
    pub finish_reason: Option<FinishReason>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens.
    pub prompt_tokens: usize,
    /// Output tokens.
    pub completion_tokens: usize,
    /// Total tokens.
    pub total_tokens: usize,
}

/// Reason generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Completed normally.
    Stop,
    /// Hit max tokens.
    Length,
    /// Content filtered.
    ContentFilter,
    /// Tool call requested.
    ToolCalls,
}

/// A chunk from streaming generation.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Delta content.
    pub delta: String,
    /// Whether this is the final chunk.
    pub is_final: bool,
    /// Finish reason (only on final chunk).
    pub finish_reason: Option<FinishReason>,
}

/// Stream of generation chunks.
pub type GenerateStream = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>;

/// Trait for LLM clients.
///
/// Implementations provide access to language model inference,
/// whether local or cloud-based.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Returns the provider name.
    fn provider_name(&self) -> &str;

    /// Returns the default model for this provider.
    fn default_model(&self) -> &str;

    /// Generates a response for the given request.
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Streams a response for the given request.
    async fn stream(&self, request: GenerateRequest) -> Result<GenerateStream>;

    /// Checks if the client is available.
    async fn is_available(&self) -> bool {
        true
    }

    /// Returns supported models.
    fn supported_models(&self) -> Vec<String> {
        vec![self.default_model().to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let system = Message::system("You are helpful");
        assert_eq!(system.role, MessageRole::System);
        assert_eq!(system.content, "You are helpful");

        let user = Message::user("Hello");
        assert_eq!(user.role, MessageRole::User);

        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role, MessageRole::Assistant);
    }

    #[test]
    fn test_generate_request_builder() {
        let request = GenerateRequest::simple("What is 2+2?")
            .with_system("You are a math tutor")
            .with_model("gpt-4")
            .with_max_tokens(100)
            .with_temperature(0.5);

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.model, Some("gpt-4".to_string()));
        assert_eq!(request.options.max_tokens, Some(100));
        assert_eq!(request.options.temperature, Some(0.5));
    }

    #[test]
    fn test_generate_options_default() {
        let options = GenerateOptions::default();
        assert_eq!(options.max_tokens, Some(4096));
        assert_eq!(options.temperature, Some(0.7));
        assert!(options.stop.is_empty());
    }
}
