//! Infernum local LLM backend implementation.
//!
//! Spawns and manages an Infernum CLI process for local model inference.
//!
//! # Output Format
//!
//! Infernum with `--output-format json-stream` produces newline-delimited JSON:
//! ```json
//! {"type":"token","content":"Hello"}
//! {"type":"tool_use","name":"bash","input":{"command":"ls"}}
//! {"type":"tool_result","id":"...","output":"..."}
//! {"type":"complete","finish_reason":"stop","usage":{"tokens":123}}
//! ```

use std::sync::Arc;

use serde::Deserialize;
use tracing::debug;

use super::process::{OutputParser, ProcessSession};
use super::{AgentBackendSession, AgentEvent, RoomContext};
use crate::error::Result;
use crate::types::{AgentBackend, Message, MessageContent};

// =============================================================================
// Infernum Output Types
// =============================================================================

/// A message from Infernum's JSON stream output.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InfernumOutput {
    /// Streaming token.
    Token(TokenMessage),

    /// Complete assistant message.
    Assistant(AssistantMessage),

    /// Tool use request.
    ToolUse(ToolUseMessage),

    /// Tool result.
    ToolResult(ToolResultMessage),

    /// Generation complete.
    Complete(CompleteMessage),

    /// Error message.
    Error(ErrorMessage),

    /// Model loading status.
    Status(StatusMessage),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenMessage {
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantMessage {
    pub content: String,
    #[serde(default)]
    pub role: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolUseMessage {
    pub name: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolResultMessage {
    pub id: String,
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
    /// Tool name (may not always be present in streaming output).
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompleteMessage {
    pub finish_reason: String,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageInfo {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorMessage {
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StatusMessage {
    pub status: String,
    #[serde(default)]
    pub progress: Option<f32>,
}

// =============================================================================
// Infernum Parser
// =============================================================================

/// Parser for Infernum's JSON stream output format.
pub struct InfernumParser {
    /// Buffer for accumulating streamed tokens into complete messages.
    token_buffer: std::sync::Mutex<String>,
}

impl InfernumParser {
    /// Creates a new Infernum parser.
    pub fn new() -> Self {
        Self {
            token_buffer: std::sync::Mutex::new(String::new()),
        }
    }
}

impl Default for InfernumParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for InfernumParser {
    fn parse_line(&self, line: &str) -> Result<Vec<AgentEvent>> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(vec![]);
        }

        // Try to parse as JSON
        let output: InfernumOutput = match serde_json::from_str(trimmed) {
            Ok(o) => o,
            Err(e) => {
                debug!("Non-JSON line from Infernum: {} ({})", trimmed, e);
                // Treat as plain text
                return Ok(vec![AgentEvent::Message {
                    content: trimmed.to_string(),
                    mentions: vec![],
                }]);
            }
        };

        // Convert to AgentEvents
        let events = match output {
            InfernumOutput::Token(msg) => {
                // Accumulate tokens into buffer
                let mut buffer = self.token_buffer.lock().unwrap();
                buffer.push_str(&msg.content);
                // Don't emit events for individual tokens - wait for complete
                vec![]
            }

            InfernumOutput::Assistant(msg) => {
                // Clear token buffer and emit full message
                let mut buffer = self.token_buffer.lock().unwrap();
                buffer.clear();

                vec![AgentEvent::Message {
                    content: msg.content,
                    mentions: vec![],
                }]
            }

            InfernumOutput::ToolUse(msg) => {
                vec![AgentEvent::ToolCall {
                    tool: msg.name,
                    input: msg.input,
                    call_id: msg.id.unwrap_or_default(),
                }]
            }

            InfernumOutput::ToolResult(msg) => {
                vec![AgentEvent::ToolResult {
                    tool: msg.name.unwrap_or_else(|| "unknown".to_string()),
                    call_id: msg.id,
                    output: msg.output,
                    success: !msg.is_error,
                    duration_ms: 0,
                }]
            }

            InfernumOutput::Complete(_msg) => {
                // Flush any remaining tokens as a message
                let mut buffer = self.token_buffer.lock().unwrap();
                if buffer.is_empty() {
                    vec![]
                } else {
                    let content = buffer.clone();
                    buffer.clear();
                    vec![AgentEvent::Message {
                        content,
                        mentions: vec![],
                    }]
                }
            }

            InfernumOutput::Error(msg) => {
                vec![AgentEvent::Error {
                    message: msg.message,
                }]
            }

            InfernumOutput::Status(msg) => {
                // Status updates are informational
                debug!("Infernum status: {} ({:?})", msg.status, msg.progress);
                vec![]
            }
        };

        Ok(events)
    }

    fn format_message(&self, message: &Message) -> Result<String> {
        // Infernum accepts JSON messages on stdin
        match &message.content {
            MessageContent::Text { content } => {
                let msg = serde_json::json!({
                    "role": "user",
                    "content": content
                });
                Ok(format!("{}\n", serde_json::to_string(&msg)?))
            }
            MessageContent::ToolCall { tool, input, .. } => {
                let msg = serde_json::json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "name": tool,
                        "input": input
                    }]
                });
                Ok(format!("{}\n", serde_json::to_string(&msg)?))
            }
            MessageContent::ToolResult {
                tool,
                output,
                call_id,
                success,
            } => {
                let msg = serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool,
                    "content": output,
                    "is_error": !success
                });
                Ok(format!("{}\n", serde_json::to_string(&msg)?))
            }
            MessageContent::System { event } => {
                let msg = serde_json::json!({
                    "role": "system",
                    "content": format!("{:?}", event)
                });
                Ok(format!("{}\n", serde_json::to_string(&msg)?))
            }
        }
    }
}

// =============================================================================
// Infernum Session Configuration
// =============================================================================

/// Configuration for spawning an Infernum session.
#[derive(Debug, Clone)]
pub struct InfernumConfig {
    /// Model identifier (e.g., "qwen-7b", "llama-3-8b").
    pub model: String,
    /// Working directory.
    pub working_dir: std::path::PathBuf,
    /// System prompt.
    pub system_prompt: Option<String>,
    /// Maximum context length.
    pub max_context: Option<u32>,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Whether to enable tool use.
    pub tools_enabled: bool,
    /// Custom model path (if not using HuggingFace).
    pub model_path: Option<std::path::PathBuf>,
}

impl InfernumConfig {
    /// Creates a new config with defaults.
    pub fn new(model: impl Into<String>, working_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            model: model.into(),
            working_dir: working_dir.into(),
            system_prompt: None,
            max_context: None,
            temperature: None,
            tools_enabled: true,
            model_path: None,
        }
    }

    /// Builds CLI arguments for Infernum.
    pub fn build_args(&self) -> Vec<String> {
        let mut args = vec![
            "chat".to_string(),
            "--model".to_string(),
            self.model.clone(),
            "--output-format".to_string(),
            "json-stream".to_string(),
        ];

        if let Some(ref prompt) = self.system_prompt {
            args.extend(["--system".to_string(), prompt.clone()]);
        }

        if let Some(ctx) = self.max_context {
            args.extend(["--max-context".to_string(), ctx.to_string()]);
        }

        if let Some(temp) = self.temperature {
            args.extend(["--temperature".to_string(), temp.to_string()]);
        }

        if self.tools_enabled {
            args.push("--tools".to_string());
        }

        if let Some(ref path) = self.model_path {
            args.extend(["--model-path".to_string(), path.display().to_string()]);
        }

        args
    }
}

// =============================================================================
// Infernum Session Spawning
// =============================================================================

/// Spawns an Infernum session.
pub async fn spawn_infernum(
    session_id: String,
    config: InfernumConfig,
    _context: &RoomContext,
) -> Result<Box<dyn AgentBackendSession>> {
    let args = config.build_args();
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Build the backend config for tracking
    let backend = AgentBackend::Infernum {
        model: config.model.clone(),
        inference_backend: crate::types::InferenceBackend::Cuda,
        tools: vec![],
    };

    let parser = Arc::new(InfernumParser::new());

    let session = ProcessSession::spawn(
        session_id,
        backend,
        "infernum", // Infernum CLI command
        &args_refs,
        &config.working_dir,
        vec![], // No special env vars
        parser,
    )
    .await?;

    Ok(Box::new(session))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_token() {
        let parser = InfernumParser::new();
        let line = r#"{"type":"token","content":"Hello"}"#;

        let events = parser.parse_line(line).unwrap();
        // Tokens are buffered, no event emitted
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_assistant_message() {
        let parser = InfernumParser::new();
        let line = r#"{"type":"assistant","content":"Hello, world!","role":"assistant"}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::Message { content, .. } => {
                assert_eq!(content, "Hello, world!");
            }
            _ => panic!("Expected Message event"),
        }
    }

    #[test]
    fn test_parse_tool_use() {
        let parser = InfernumParser::new();
        let line = r#"{"type":"tool_use","name":"bash","input":{"command":"ls"},"id":"tool123"}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::ToolCall {
                tool,
                input,
                call_id,
            } => {
                assert_eq!(tool, "bash");
                assert_eq!(call_id, "tool123");
                assert!(input.get("command").is_some());
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_complete_flushes_buffer() {
        let parser = InfernumParser::new();

        // First, add some tokens
        parser.parse_line(r#"{"type":"token","content":"Hello "}"#).unwrap();
        parser.parse_line(r#"{"type":"token","content":"world"}"#).unwrap();

        // Complete should flush the buffer
        let events = parser
            .parse_line(r#"{"type":"complete","finish_reason":"stop"}"#)
            .unwrap();

        assert_eq!(events.len(), 1);
        match &events[0] {
            AgentEvent::Message { content, .. } => {
                assert_eq!(content, "Hello world");
            }
            _ => panic!("Expected Message event"),
        }
    }

    #[test]
    fn test_parse_error() {
        let parser = InfernumParser::new();
        let line = r#"{"type":"error","message":"Out of memory"}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::Error { message } => {
                assert_eq!(message, "Out of memory");
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_status() {
        let parser = InfernumParser::new();
        let line = r#"{"type":"status","status":"loading","progress":0.5}"#;

        // Status messages don't emit events
        let events = parser.parse_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_config_build_args() {
        let config = InfernumConfig::new("qwen-7b", "/tmp");

        let args = config.build_args();
        assert!(args.contains(&"chat".to_string()));
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"qwen-7b".to_string()));
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"json-stream".to_string()));
        assert!(args.contains(&"--tools".to_string()));
    }

    #[test]
    fn test_config_with_options() {
        let mut config = InfernumConfig::new("llama-3-8b", "/tmp");
        config.system_prompt = Some("You are helpful.".to_string());
        config.max_context = Some(4096);
        config.temperature = Some(0.7);
        config.tools_enabled = false;

        let args = config.build_args();
        assert!(args.contains(&"--system".to_string()));
        assert!(args.contains(&"You are helpful.".to_string()));
        assert!(args.contains(&"--max-context".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"--temperature".to_string()));
        assert!(args.contains(&"0.7".to_string()));
        assert!(!args.contains(&"--tools".to_string()));
    }

    #[test]
    fn test_format_text_message() {
        use crate::types::{ChannelType, MessageId, ParticipantId};
        use chrono::Utc;
        use std::collections::HashMap;

        let parser = InfernumParser::new();

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

        let formatted = parser.format_message(&message).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(formatted.trim()).unwrap();

        assert_eq!(parsed["role"], "user");
        assert_eq!(parsed["content"], "Hello");
    }
}
