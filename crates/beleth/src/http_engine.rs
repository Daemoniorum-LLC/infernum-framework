//! HTTP client wrapper for remote Infernum server.
//!
//! This is a simplified client for testing purposes that bypasses
//! the full InferenceEngine trait complexity.

use std::sync::atomic::{AtomicBool, Ordering};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Simplified HTTP client for Infernum server.
///
/// Designed for wellbeing monitoring tests where we need real inference
/// but don't need the full InferenceEngine trait implementation.
pub struct HttpEngine {
    client: Client,
    base_url: String,
    model_id: String,
    ready: AtomicBool,
}

/// Model info from /v1/models endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Chat completion response.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

/// Simple message for the chat interface.
#[derive(Debug, Clone)]
pub struct SimpleMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

impl SimpleMessage {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Result type for HTTP engine operations.
pub type Result<T> = std::result::Result<T, HttpEngineError>;

/// Errors from the HTTP engine.
#[derive(Debug, thiserror::Error)]
pub enum HttpEngineError {
    /// Connection failed.
    #[error("Connection failed: {0}")]
    Connection(String),

    /// Request failed.
    #[error("Request failed: {0}")]
    Request(String),

    /// Server returned an error.
    #[error("Server error {status}: {message}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Error message from the server.
        message: String,
    },

    /// No model loaded.
    #[error("No model loaded on server")]
    NoModel,
}

impl HttpEngine {
    /// Creates a new HTTP engine client.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            model_id: String::new(),
            ready: AtomicBool::new(false),
        }
    }

    /// Connects to the server and discovers the loaded model.
    pub async fn connect(&mut self) -> Result<()> {
        // Check health
        let health_url = format!("{}/health", self.base_url);
        let resp = self
            .client
            .get(&health_url)
            .send()
            .await
            .map_err(|e| HttpEngineError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(HttpEngineError::Connection(
                "Server not healthy".to_string(),
            ));
        }

        info!("Connected to Infernum at {}", self.base_url);

        // Discover model
        let models_url = format!("{}/v1/models", self.base_url);
        let resp: ModelsResponse = self
            .client
            .get(&models_url)
            .send()
            .await
            .map_err(|e| HttpEngineError::Request(e.to_string()))?
            .json()
            .await
            .map_err(|e| HttpEngineError::Request(format!("Failed to parse: {}", e)))?;

        if let Some(model) = resp.data.first() {
            self.model_id = model.id.clone();
            self.ready.store(true, Ordering::SeqCst);
            info!("Model available: {}", model.id);
            Ok(())
        } else {
            Err(HttpEngineError::NoModel)
        }
    }

    /// Returns the loaded model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns whether the engine is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Generates a response from the given messages.
    pub async fn chat(
        &self,
        messages: Vec<SimpleMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        if !self.is_ready() {
            return Err(HttpEngineError::NoModel);
        }

        let url = format!("{}/v1/chat/completions", self.base_url);

        let request = ChatRequest {
            model: self.model_id.clone(),
            messages: messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect(),
            max_tokens,
            temperature,
            stream: false,
        };

        debug!("Sending chat request to {}", url);

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| HttpEngineError::Request(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let message = resp.text().await.unwrap_or_default();
            return Err(HttpEngineError::Server { status, message });
        }

        let response: ChatResponse = resp
            .json()
            .await
            .map_err(|e| HttpEngineError::Request(format!("Failed to parse: {}", e)))?;

        let content = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        debug!("Received response: {} chars", content.len());

        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_engine_creation() {
        let engine = HttpEngine::new("http://localhost:8081");
        assert!(!engine.is_ready());
        assert!(engine.model_id().is_empty());
    }

    #[test]
    fn test_simple_message() {
        let sys = SimpleMessage::system("You are helpful");
        assert_eq!(sys.role, "system");

        let user = SimpleMessage::user("Hello");
        assert_eq!(user.role, "user");

        let asst = SimpleMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
    }
}
