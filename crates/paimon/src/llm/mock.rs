//! Mock LLM client for testing.
//!
//! Provides a configurable mock client that can return
//! predefined responses or generate responses based on rules.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use parking_lot::Mutex;

use super::client::{
    FinishReason, GenerateRequest, GenerateResponse, GenerateStream, LlmClient, LlmError, Result,
    StreamChunk, Usage,
};

/// A mock LLM client for testing.
///
/// Supports:
/// - Predefined responses (FIFO queue)
/// - Response generation based on input patterns
/// - Simulated errors
/// - Streaming simulation
///
/// # Example
///
/// ```rust,ignore
/// let client = MockLlmClient::new()
///     .with_response("Hello! How can I help?")
///     .with_response("The answer is 42.");
///
/// // First call returns "Hello! How can I help?"
/// let response = client.generate(request).await?;
///
/// // Second call returns "The answer is 42."
/// let response = client.generate(request).await?;
/// ```
pub struct MockLlmClient {
    /// Provider name.
    name: String,

    /// Default model.
    model: String,

    /// Queue of responses to return.
    responses: Arc<Mutex<VecDeque<MockResponse>>>,

    /// Whether to simulate unavailability.
    unavailable: bool,

    /// Simulated latency in milliseconds.
    latency_ms: u64,
}

/// A mock response configuration.
#[derive(Debug, Clone)]
pub enum MockResponse {
    /// A simple text response.
    Text(String),

    /// A response with full metadata.
    Full(GenerateResponse),

    /// An error response.
    Error(String),

    /// Echo the input back.
    Echo,

    /// JSON response (for structured output testing).
    Json(serde_json::Value),
}

impl MockLlmClient {
    /// Creates a new mock client.
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            model: "mock-model".to_string(),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            unavailable: false,
            latency_ms: 0,
        }
    }

    /// Sets the provider name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Adds a text response to the queue.
    pub fn with_response(self, text: impl Into<String>) -> Self {
        self.responses
            .lock()
            .push_back(MockResponse::Text(text.into()));
        self
    }

    /// Adds a full response to the queue.
    pub fn with_full_response(self, response: GenerateResponse) -> Self {
        self.responses
            .lock()
            .push_back(MockResponse::Full(response));
        self
    }

    /// Adds an error response to the queue.
    pub fn with_error(self, error: impl Into<String>) -> Self {
        self.responses
            .lock()
            .push_back(MockResponse::Error(error.into()));
        self
    }

    /// Adds an echo response (returns the input).
    pub fn with_echo(self) -> Self {
        self.responses.lock().push_back(MockResponse::Echo);
        self
    }

    /// Adds a JSON response.
    pub fn with_json(self, json: serde_json::Value) -> Self {
        self.responses.lock().push_back(MockResponse::Json(json));
        self
    }

    /// Simulates unavailability.
    pub fn unavailable(mut self) -> Self {
        self.unavailable = true;
        self
    }

    /// Adds simulated latency.
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Clears all queued responses.
    pub fn clear(&self) {
        self.responses.lock().clear();
    }

    /// Returns the number of remaining responses.
    pub fn remaining_responses(&self) -> usize {
        self.responses.lock().len()
    }

    /// Gets the next response, or generates a default.
    fn next_response(&self, request: &GenerateRequest) -> MockResponse {
        self.responses.lock().pop_front().unwrap_or_else(|| {
            // Default: echo the last user message
            if let Some(last) = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == super::client::MessageRole::User)
            {
                MockResponse::Text(format!("Mock response to: {}", last.content))
            } else {
                MockResponse::Text("Mock response".to_string())
            }
        })
    }
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    fn provider_name(&self) -> &str {
        &self.name
    }

    fn default_model(&self) -> &str {
        &self.model
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.latency_ms)).await;
        }

        // Check availability
        if self.unavailable {
            return Err(LlmError::ProviderUnavailable(self.name.clone()));
        }

        let mock_response = self.next_response(&request);

        match mock_response {
            MockResponse::Text(text) => Ok(GenerateResponse {
                content: text,
                model: request.model.unwrap_or_else(|| self.model.clone()),
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                }),
                finish_reason: Some(FinishReason::Stop),
            }),

            MockResponse::Full(response) => Ok(response),

            MockResponse::Error(msg) => Err(LlmError::Request(msg)),

            MockResponse::Echo => {
                let content = request
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == super::client::MessageRole::User)
                    .map(|m| m.content.clone())
                    .unwrap_or_else(|| "Echo: no input".to_string());

                Ok(GenerateResponse {
                    content,
                    model: request.model.unwrap_or_else(|| self.model.clone()),
                    usage: None,
                    finish_reason: Some(FinishReason::Stop),
                })
            },

            MockResponse::Json(value) => Ok(GenerateResponse {
                content: serde_json::to_string(&value).unwrap_or_default(),
                model: request.model.unwrap_or_else(|| self.model.clone()),
                usage: Some(Usage {
                    prompt_tokens: 15,
                    completion_tokens: 25,
                    total_tokens: 40,
                }),
                finish_reason: Some(FinishReason::Stop),
            }),
        }
    }

    async fn stream(&self, request: GenerateRequest) -> Result<GenerateStream> {
        // Check availability
        if self.unavailable {
            return Err(LlmError::ProviderUnavailable(self.name.clone()));
        }

        // Generate full response first
        let response = self.generate(request).await?;

        // Split into chunks (word by word)
        let words: Vec<String> = response
            .content
            .split_whitespace()
            .map(|w| format!("{} ", w))
            .collect();

        let chunks: Vec<Result<StreamChunk>> = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let is_last = i == response.content.split_whitespace().count() - 1;
                Ok(StreamChunk {
                    delta: word,
                    is_final: is_last,
                    finish_reason: if is_last {
                        Some(FinishReason::Stop)
                    } else {
                        None
                    },
                })
            })
            .collect();

        Ok(Box::pin(stream::iter(chunks)))
    }

    async fn is_available(&self) -> bool {
        !self.unavailable
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            self.model.clone(),
            "mock-small".to_string(),
            "mock-large".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_mock_client_basic() {
        let client = MockLlmClient::new().with_response("Hello, world!");

        let response = client
            .generate(GenerateRequest::simple("Hi"))
            .await
            .expect("generate");

        assert_eq!(response.content, "Hello, world!");
        assert_eq!(response.model, "mock-model");
        assert!(response.usage.is_some());
    }

    #[tokio::test]
    async fn test_mock_client_queue() {
        let client = MockLlmClient::new()
            .with_response("First")
            .with_response("Second")
            .with_response("Third");

        assert_eq!(client.remaining_responses(), 3);

        let r1 = client
            .generate(GenerateRequest::simple("1"))
            .await
            .expect("1");
        assert_eq!(r1.content, "First");
        assert_eq!(client.remaining_responses(), 2);

        let r2 = client
            .generate(GenerateRequest::simple("2"))
            .await
            .expect("2");
        assert_eq!(r2.content, "Second");

        let r3 = client
            .generate(GenerateRequest::simple("3"))
            .await
            .expect("3");
        assert_eq!(r3.content, "Third");

        // Queue is empty, should generate default response
        let r4 = client
            .generate(GenerateRequest::simple("Hello"))
            .await
            .expect("4");
        assert!(r4.content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_mock_client_echo() {
        let client = MockLlmClient::new().with_echo();

        let response = client
            .generate(GenerateRequest::simple("Test message"))
            .await
            .expect("generate");

        assert_eq!(response.content, "Test message");
    }

    #[tokio::test]
    async fn test_mock_client_error() {
        let client = MockLlmClient::new().with_error("Simulated error");

        let result = client.generate(GenerateRequest::simple("Hi")).await;

        assert!(result.is_err());
        if let Err(LlmError::Request(msg)) = result {
            assert_eq!(msg, "Simulated error");
        } else {
            panic!("Expected Request error");
        }
    }

    #[tokio::test]
    async fn test_mock_client_unavailable() {
        let client = MockLlmClient::new().unavailable();

        assert!(!client.is_available().await);

        let result = client.generate(GenerateRequest::simple("Hi")).await;
        assert!(matches!(result, Err(LlmError::ProviderUnavailable(_))));
    }

    #[tokio::test]
    async fn test_mock_client_json() {
        let client = MockLlmClient::new().with_json(serde_json::json!({
            "action": "execute_tool",
            "tool": "calculator"
        }));

        let response = client
            .generate(GenerateRequest::simple("Calculate"))
            .await
            .expect("generate");

        let parsed: serde_json::Value =
            serde_json::from_str(&response.content).expect("parse json");
        assert_eq!(parsed["action"], "execute_tool");
    }

    #[tokio::test]
    async fn test_mock_client_streaming() {
        let client = MockLlmClient::new().with_response("Hello world test");

        let mut stream = client
            .stream(GenerateRequest::simple("Hi"))
            .await
            .expect("stream");

        let mut collected = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("chunk");
            collected.push_str(&chunk.delta);
        }

        assert_eq!(collected.trim(), "Hello world test");
    }

    #[tokio::test]
    async fn test_mock_client_provider_info() {
        let client = MockLlmClient::new()
            .with_name("test-provider")
            .with_model("test-model-v1");

        assert_eq!(client.provider_name(), "test-provider");
        assert_eq!(client.default_model(), "test-model-v1");
        assert!(client
            .supported_models()
            .contains(&"test-model-v1".to_string()));
    }
}
