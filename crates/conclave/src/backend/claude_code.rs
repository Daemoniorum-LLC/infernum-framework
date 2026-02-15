//! Claude Code backend implementation.
//!
//! Spawns and manages a Claude Code CLI process for agent interactions.
//!
//! # Output Format
//!
//! Claude Code with `--output-format stream-json` produces newline-delimited JSON:
//! ```json
//! {"type":"assistant","message":{"content":"Hello"},"session_id":"..."}
//! {"type":"tool_use","name":"Read","input":{...}}
//! {"type":"tool_result","tool_use_id":"...","output":"..."}
//! {"type":"result","result":"...","cost_usd":0.01}
//! ```

use std::sync::Arc;

use serde::Deserialize;
use tracing::debug;

use super::process::{OutputParser, ProcessSession};
use super::{AgentBackendSession, AgentEvent, RoomContext};
use crate::error::Result;
use crate::types::{AgentBackend, ClaudeTier, Message, MessageContent};

// =============================================================================
// Claude Code Output Types
// =============================================================================

/// A message from Claude Code's stream-json output.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeCodeOutput {
    /// System information message.
    System(SystemMessage),

    /// Assistant is starting to respond.
    AssistantStart,

    /// Assistant text content.
    Assistant(AssistantMessage),

    /// Tool use request.
    ToolUse(ToolUseMessage),

    /// Tool result.
    ToolResult(ToolResultMessage),

    /// Final result.
    Result(ResultMessage),

    /// Error message.
    Error(ErrorMessage),
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemMessage {
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantMessage {
    pub message: AssistantContent,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantContent {
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolUseMessage {
    pub name: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub tool_use_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolResultMessage {
    pub tool_use_id: String,
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
    /// Tool name (may not always be present in streaming output).
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResultMessage {
    pub result: String,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub num_turns: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorMessage {
    pub error: String,
}

// =============================================================================
// Claude Code Parser
// =============================================================================

/// Parser for Claude Code's stream-json output format.
pub struct ClaudeCodeParser;

impl OutputParser for ClaudeCodeParser {
    fn parse_line(&self, line: &str) -> Result<Vec<AgentEvent>> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(vec![]);
        }

        // Try to parse as JSON
        let output: ClaudeCodeOutput = match serde_json::from_str(trimmed) {
            Ok(o) => o,
            Err(e) => {
                // Not valid JSON - might be raw text output
                debug!("Non-JSON line from Claude Code: {} ({})", trimmed, e);
                // Treat as plain text message
                return Ok(vec![AgentEvent::Message {
                    content: trimmed.to_string(),
                    mentions: vec![],
                }]);
            }
        };

        // Convert to AgentEvents
        let events = match output {
            ClaudeCodeOutput::System(msg) => {
                // System messages are informational
                vec![AgentEvent::Message {
                    content: format!("[system] {}", msg.message),
                    mentions: vec![],
                }]
            }

            ClaudeCodeOutput::AssistantStart => {
                // Assistant starting - no event needed
                vec![]
            }

            ClaudeCodeOutput::Assistant(msg) => {
                vec![AgentEvent::Message {
                    content: msg.message.content,
                    mentions: vec![], // TODO: Parse @mentions from content
                }]
            }

            ClaudeCodeOutput::ToolUse(msg) => {
                vec![AgentEvent::ToolCall {
                    tool: msg.name,
                    input: msg.input,
                    call_id: msg.tool_use_id.unwrap_or_default(),
                }]
            }

            ClaudeCodeOutput::ToolResult(msg) => {
                vec![AgentEvent::ToolResult {
                    tool: msg.name.unwrap_or_else(|| "unknown".to_string()),
                    call_id: msg.tool_use_id,
                    output: msg.output,
                    success: !msg.is_error,
                    duration_ms: 0, // Not provided by Claude Code
                }]
            }

            ClaudeCodeOutput::Result(msg) => {
                // Final result - could emit stats or completion
                vec![AgentEvent::Message {
                    content: msg.result,
                    mentions: vec![],
                }]
            }

            ClaudeCodeOutput::Error(msg) => {
                vec![AgentEvent::Error {
                    message: msg.error,
                }]
            }
        };

        Ok(events)
    }

    fn format_message(&self, message: &Message) -> Result<String> {
        // Claude Code reads from stdin as plain text prompts
        match &message.content {
            MessageContent::Text { content } => Ok(format!("{}\n", content)),
            MessageContent::ToolCall { tool, input, .. } => {
                // Shouldn't send tool calls to Claude Code, but format anyway
                Ok(format!("[Tool request: {}] {}\n", tool, input))
            }
            MessageContent::ToolResult { output, .. } => {
                // Tool results might be sent as context
                Ok(format!("{}\n", output))
            }
            MessageContent::System { event } => Ok(format!("[System] {:?}\n", event)),
        }
    }
}

// =============================================================================
// Claude Code Session Builder
// =============================================================================

/// Builds Claude Code CLI arguments based on configuration.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Claude tier (model selection).
    pub tier: ClaudeTier,
    /// Working directory.
    pub working_dir: std::path::PathBuf,
    /// System prompt additions.
    pub system_prompt: Option<String>,
    /// Allowed tools (empty = all).
    pub allowed_tools: Vec<String>,
    /// Disallowed tools.
    pub disallowed_tools: Vec<String>,
    /// Maximum turns before stopping.
    pub max_turns: Option<u32>,
    /// Whether to resume a session.
    pub resume_session: Option<String>,
}

impl ClaudeCodeConfig {
    /// Creates a new config with defaults.
    pub fn new(tier: ClaudeTier, working_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            tier,
            working_dir: working_dir.into(),
            system_prompt: None,
            allowed_tools: vec![],
            disallowed_tools: vec![],
            max_turns: None,
            resume_session: None,
        }
    }

    /// Builds CLI arguments.
    pub fn build_args(&self) -> Vec<String> {
        // --verbose is required when using stream-json output format
        let mut args = vec![
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
        ];

        // Model selection based on tier
        let model = match self.tier {
            ClaudeTier::Opus => "claude-opus-4-5-20251101",
            ClaudeTier::Sonnet => "claude-sonnet-4-20250514",
            ClaudeTier::Haiku => "claude-haiku-3-5-20250630",
        };
        args.extend(["--model".to_string(), model.to_string()]);

        // System prompt
        if let Some(ref prompt) = self.system_prompt {
            args.extend([
                "--append-system-prompt".to_string(),
                prompt.clone(),
            ]);
        }

        // Allowed tools
        for tool in &self.allowed_tools {
            args.extend(["--allowedTools".to_string(), tool.clone()]);
        }

        // Disallowed tools
        for tool in &self.disallowed_tools {
            args.extend(["--disallowedTools".to_string(), tool.clone()]);
        }

        // Max turns
        if let Some(turns) = self.max_turns {
            args.extend(["--max-turns".to_string(), turns.to_string()]);
        }

        // Resume session
        if let Some(ref session) = self.resume_session {
            args.extend(["--resume".to_string(), session.clone()]);
        }

        // Print mode for non-interactive
        args.push("--print".to_string());

        args
    }
}

// =============================================================================
// Claude Code Session Spawning
// =============================================================================

/// Spawns a Claude Code session.
pub async fn spawn_claude_code(
    session_id: String,
    config: ClaudeCodeConfig,
    context: &RoomContext,
) -> Result<Box<dyn AgentBackendSession>> {
    let args = config.build_args();
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Build persona prompt from context
    let mut system_additions = vec![];

    if let Some(ref persona) = context.persona_prompt {
        system_additions.push(persona.clone());
    }

    // Add room context
    system_additions.push(format!(
        "You are in room '{}' with {} other participants.",
        context.room_name,
        context.participants.len()
    ));

    let parser = Arc::new(ClaudeCodeParser);

    let session = ProcessSession::spawn(
        session_id,
        AgentBackend::ClaudeCode {
            tier: config.tier,
            allowed_tools: config.allowed_tools.clone(),
        },
        "claude", // Claude Code CLI command
        &args_refs,
        &config.working_dir,
        vec![], // No special env vars needed
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
    fn test_parse_assistant_message() {
        let parser = ClaudeCodeParser;
        let line = r#"{"type":"assistant","message":{"content":"Hello, world!"}}"#;

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
        let parser = ClaudeCodeParser;
        let line = r#"{"type":"tool_use","name":"Read","input":{"file_path":"/test.txt"},"tool_use_id":"abc123"}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::ToolCall {
                tool,
                input,
                call_id,
            } => {
                assert_eq!(tool, "Read");
                assert_eq!(call_id, "abc123");
                assert!(input.get("file_path").is_some());
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_tool_result() {
        let parser = ClaudeCodeParser;
        let line =
            r#"{"type":"tool_result","tool_use_id":"abc123","output":"File contents here","is_error":false}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::ToolResult {
                call_id,
                output,
                success,
                ..
            } => {
                assert_eq!(call_id, "abc123");
                assert_eq!(output, "File contents here");
                assert!(*success);
            }
            _ => panic!("Expected ToolResult event"),
        }
    }

    #[test]
    fn test_parse_error() {
        let parser = ClaudeCodeParser;
        let line = r#"{"type":"error","error":"Something went wrong"}"#;

        let events = parser.parse_line(line).unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::Error { message } => {
                assert_eq!(message, "Something went wrong");
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_empty_line() {
        let parser = ClaudeCodeParser;
        let events = parser.parse_line("").unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_parse_non_json() {
        let parser = ClaudeCodeParser;
        // Non-JSON is treated as plain text
        let events = parser.parse_line("Some plain text output").unwrap();
        assert_eq!(events.len(), 1);

        match &events[0] {
            AgentEvent::Message { content, .. } => {
                assert_eq!(content, "Some plain text output");
            }
            _ => panic!("Expected Message event"),
        }
    }

    #[test]
    fn test_config_build_args() {
        let config = ClaudeCodeConfig::new(ClaudeTier::Opus, "/tmp");

        let args = config.build_args();
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--verbose".to_string())); // Required for stream-json
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"claude-opus-4-5-20251101".to_string()));
        assert!(args.contains(&"--print".to_string()));
    }

    #[test]
    fn test_config_with_tools() {
        let mut config = ClaudeCodeConfig::new(ClaudeTier::Sonnet, "/tmp");
        config.allowed_tools = vec!["Read".to_string(), "Write".to_string()];
        config.max_turns = Some(10);

        let args = config.build_args();
        assert!(args.contains(&"--allowedTools".to_string()));
        assert!(args.contains(&"Read".to_string()));
        assert!(args.contains(&"Write".to_string()));
        assert!(args.contains(&"--max-turns".to_string()));
        assert!(args.contains(&"10".to_string()));
    }

    #[test]
    fn test_tier_to_model() {
        let opus = ClaudeCodeConfig::new(ClaudeTier::Opus, "/tmp");
        let sonnet = ClaudeCodeConfig::new(ClaudeTier::Sonnet, "/tmp");
        let haiku = ClaudeCodeConfig::new(ClaudeTier::Haiku, "/tmp");

        assert!(opus.build_args().contains(&"claude-opus-4-5-20251101".to_string()));
        assert!(sonnet.build_args().contains(&"claude-sonnet-4-20250514".to_string()));
        assert!(haiku.build_args().contains(&"claude-haiku-3-5-20250630".to_string()));
    }
}
