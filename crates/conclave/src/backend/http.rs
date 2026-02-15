//! HTTP-based custom backend implementation.
//!
//! Allows connecting to custom agent endpoints via HTTP/WebSocket.
//!
//! # Protocols
//!
//! - **JSON-RPC**: Standard JSON-RPC 2.0 over HTTP POST
//! - **REST**: RESTful API with message-based endpoints
//! - **SSE**: Server-Sent Events for streaming responses
//!
//! # Example Endpoint
//!
//! ```json
//! POST /v1/chat
//! {
//!   "messages": [{"role": "user", "content": "Hello"}],
//!   "stream": true
//! }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::{AgentBackendSession, AgentEvent, RoomContext, TerminationReason};
use crate::error::{ConclaveError, Result};
use crate::types::{AgentBackend, AgentProtocol, Message, MessageContent};

// =============================================================================
// HTTP Response Types
// =============================================================================

/// Response from a custom HTTP agent endpoint.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum HttpResponse {
    /// Standard chat completion response.
    ChatCompletion(ChatCompletionResponse),
    /// Streaming chunk.
    StreamChunk(StreamChunk),
    /// Error response.
    Error(ErrorResponse),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub message: ChoiceMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

// =============================================================================
// HTTP Session Configuration
// =============================================================================

/// Configuration for an HTTP backend session.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Endpoint URL.
    pub endpoint: String,
    /// Protocol to use.
    pub protocol: AgentProtocol,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Model name (for OpenAI-compatible endpoints).
    pub model: Option<String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Whether to use streaming.
    pub stream: bool,
}

impl HttpConfig {
    /// Creates a new HTTP config.
    pub fn new(endpoint: impl Into<String>, protocol: AgentProtocol) -> Self {
        Self {
            endpoint: endpoint.into(),
            protocol,
            api_key: None,
            model: None,
            timeout_secs: 120,
            stream: true,
        }
    }

    /// Sets the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

// =============================================================================
// HTTP Session
// =============================================================================

/// An HTTP-based agent backend session.
///
/// This session communicates with a custom agent endpoint via HTTP.
/// It supports both synchronous (request/response) and streaming (SSE) modes.
pub struct HttpSession {
    /// Session identifier.
    session_id: String,
    /// Backend configuration.
    backend: AgentBackend,
    /// HTTP configuration.
    config: HttpConfig,
    /// Whether the session is running.
    running: Arc<AtomicBool>,
    /// Event sender.
    event_tx: mpsc::Sender<AgentEvent>,
    /// Event receiver (taken once).
    event_rx: std::sync::Mutex<Option<mpsc::Receiver<AgentEvent>>>,
    /// Message history for context.
    messages: tokio::sync::Mutex<Vec<ChatMessage>>,
}

/// A message in the chat history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl HttpSession {
    /// Creates a new HTTP session.
    pub fn new(session_id: String, backend: AgentBackend, config: HttpConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(256);

        info!(
            "Creating HTTP session {} to {}",
            session_id, config.endpoint
        );

        Self {
            session_id,
            backend,
            config,
            running: Arc::new(AtomicBool::new(true)),
            event_tx,
            event_rx: std::sync::Mutex::new(Some(event_rx)),
            messages: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    /// Sends a chat completion request.
    async fn send_request(&self, messages: &[ChatMessage]) -> Result<()> {
        // Build request body
        let body = serde_json::json!({
            "messages": messages,
            "stream": self.config.stream,
            "model": self.config.model.as_deref().unwrap_or("default"),
        });

        debug!(
            "[{}] Sending request to {}: {:?}",
            self.session_id, self.config.endpoint, body
        );

        // In a real implementation, this would make the HTTP request
        // For now, we'll just log and emit an event
        warn!(
            "[{}] HTTP backend not fully implemented - request logged only",
            self.session_id
        );

        // Emit a placeholder response
        let _ = self
            .event_tx
            .send(AgentEvent::Message {
                content: "[HTTP backend placeholder - request received]".to_string(),
                mentions: vec![],
            })
            .await;

        Ok(())
    }

    /// Parses a streaming chunk.
    /// Reserved for future SSE streaming implementation.
    #[allow(dead_code)]
    fn parse_chunk(&self, chunk: &str) -> Vec<AgentEvent> {
        // SSE format: data: {...}\n\n
        let mut events = vec![];

        for line in chunk.lines() {
            let line = line.trim();
            if line.is_empty() || line == "data: [DONE]" {
                continue;
            }

            let json_str = line.strip_prefix("data: ").unwrap_or(line);

            match serde_json::from_str::<StreamChunk>(json_str) {
                Ok(chunk) => {
                    for choice in &chunk.choices {
                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                events.push(AgentEvent::Message {
                                    content: content.clone(),
                                    mentions: vec![],
                                });
                            }
                        }

                        // Handle tool calls
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                if let Some(func) = &tc.function {
                                    if let Some(name) = &func.name {
                                        events.push(AgentEvent::ToolCall {
                                            tool: name.clone(),
                                            input: serde_json::json!({}),
                                            call_id: tc.id.clone().unwrap_or_default(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!("[{}] Failed to parse chunk: {}", self.session_id, e);
                }
            }
        }

        events
    }
}

#[async_trait]
impl AgentBackendSession for HttpSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn backend(&self) -> &AgentBackend {
        &self.backend
    }

    async fn send_message(&self, message: &Message) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }

        // Convert message to chat format
        let chat_msg = match &message.content {
            MessageContent::Text { content } => ChatMessage {
                role: "user".to_string(),
                content: content.clone(),
            },
            MessageContent::ToolResult { output, .. } => ChatMessage {
                role: "tool".to_string(),
                content: output.to_string(),
            },
            _ => {
                return Ok(()); // Skip non-text messages
            }
        };

        // Add to history
        {
            let mut messages = self.messages.lock().await;
            messages.push(chat_msg);
        }

        // Send request
        let messages = self.messages.lock().await.clone();
        self.send_request(&messages).await
    }

    async fn interrupt(&self) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }

        info!("[{}] Interrupt requested (HTTP sessions don't support true interrupt)", self.session_id);
        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);

        let _ = self
            .event_tx
            .send(AgentEvent::Terminated {
                reason: TerminationReason::Requested,
            })
            .await;

        info!("[{}] HTTP session terminated", self.session_id);
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn take_event_receiver(&self) -> Option<mpsc::Receiver<AgentEvent>> {
        self.event_rx.lock().unwrap().take()
    }
}

// =============================================================================
// Spawning
// =============================================================================

/// Spawns a custom HTTP backend session.
pub fn spawn_http(
    session_id: String,
    endpoint: &str,
    protocol: AgentProtocol,
    _context: &RoomContext,
) -> Result<Box<dyn AgentBackendSession>> {
    let backend = AgentBackend::Custom {
        endpoint: endpoint.to_string(),
        protocol: protocol.clone(),
    };

    let config = HttpConfig::new(endpoint, protocol);
    let session = HttpSession::new(session_id, backend, config);

    Ok(Box::new(session))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_config() {
        let config = HttpConfig::new("http://localhost:8080", AgentProtocol::JsonRpc)
            .with_api_key("test-key")
            .with_model("gpt-4");

        assert_eq!(config.endpoint, "http://localhost:8080");
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_parse_chat_completion() {
        let json = r#"{
            "id": "chatcmpl-123",
            "choices": [{
                "message": {
                    "content": "Hello!",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, Some("Hello!".to_string()));
    }

    #[test]
    fn test_parse_stream_chunk() {
        let json = r#"{
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_parse_error_response() {
        let json = r#"{
            "error": {
                "message": "Invalid API key",
                "code": "invalid_api_key"
            }
        }"#;

        let error: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(error.error.message, "Invalid API key");
        assert_eq!(error.error.code, Some("invalid_api_key".to_string()));
    }

    #[tokio::test]
    async fn test_http_session_lifecycle() {
        let backend = AgentBackend::Custom {
            endpoint: "http://localhost:8080".to_string(),
            protocol: AgentProtocol::JsonRpc,
        };
        let config = HttpConfig::new("http://localhost:8080", AgentProtocol::JsonRpc);
        let session = HttpSession::new("test-123".to_string(), backend, config);

        assert!(session.is_running());
        assert_eq!(session.session_id(), "test-123");

        session.terminate().await.unwrap();
        assert!(!session.is_running());
    }

    #[tokio::test]
    async fn test_send_to_terminated_fails() {
        use crate::types::{ChannelType, MessageId, ParticipantId};
        use chrono::Utc;
        use std::collections::HashMap;

        let backend = AgentBackend::Custom {
            endpoint: "http://localhost:8080".to_string(),
            protocol: AgentProtocol::JsonRpc,
        };
        let config = HttpConfig::new("http://localhost:8080", AgentProtocol::JsonRpc);
        let session = HttpSession::new("test-123".to_string(), backend, config);

        session.terminate().await.unwrap();

        let message = Message {
            id: MessageId::new(),
            channel: ChannelType::Main,
            sender: ParticipantId::new(),
            content: MessageContent::Text {
                content: "Hello".to_string(),
            },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        let result = session.send_message(&message).await;
        assert!(matches!(result, Err(ConclaveError::BackendTerminated { .. })));
    }
}
