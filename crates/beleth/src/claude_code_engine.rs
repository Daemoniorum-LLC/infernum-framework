//! Claude Code inference engine.
//!
//! Wraps the Claude Code CLI to provide an `InferenceEngine` implementation,
//! allowing the supervisor and other components to use Claude Code as a backend.
//!
//! # Usage
//!
//! ```ignore
//! use beleth::ClaudeCodeEngine;
//!
//! let engine = ClaudeCodeEngine::new(ClaudeTier::Opus, "/path/to/project");
//! let response = engine.generate(request).await?;
//! ```

use std::path::PathBuf;
use std::process::Stdio;

use async_trait::async_trait;
use infernum_core::{
    model::{LlamaVersion, ModelArchitecture, ModelMetadata, ModelSource},
    request::{EmbedRequest, GenerateRequest, PromptInput},
    response::{Choice, EmbedResponse, GenerateResponse},
    streaming::{StreamChunkBuilder, TokenStream},
    types::{FinishReason, Message, ModelId, RequestId, Role, Usage},
    Result,
};
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, warn};

use abaddon::InferenceEngine;

// =============================================================================
// Claude Code Tier
// =============================================================================

/// Claude model tier for Claude Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeTier {
    /// Claude Opus 4.5 - most capable.
    Opus,
    /// Claude Sonnet 4 - balanced.
    Sonnet,
    /// Claude Haiku 3.5 - fastest.
    Haiku,
}

impl ClaudeTier {
    /// Returns the model ID for this tier.
    pub fn model_id(&self) -> &'static str {
        match self {
            ClaudeTier::Opus => "claude-opus-4-5-20251101",
            ClaudeTier::Sonnet => "claude-sonnet-4-5-20250929",
            ClaudeTier::Haiku => "claude-haiku-4-5-20251001",
        }
    }
}

// =============================================================================
// Claude Code Output Types
// =============================================================================

/// Parsed output from Claude Code's stream-json format.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClaudeOutput {
    System(SystemOutput),
    Assistant(AssistantOutput),
    ToolUse(ToolUseOutput),
    ToolResult(ToolResultOutput),
    Result(ResultOutput),
    Error(ErrorOutput),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
struct SystemOutput {
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AssistantOutput {
    message: AssistantMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct AssistantMessage {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolUseOutput {
    name: String,
    input: serde_json::Value,
    #[serde(default)]
    tool_use_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolResultOutput {
    tool_use_id: String,
    output: String,
    #[serde(default)]
    is_error: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ResultOutput {
    result: String,
    #[serde(default)]
    cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ErrorOutput {
    error: String,
}

// =============================================================================
// Claude Code Engine
// =============================================================================

/// An inference engine that uses Claude Code CLI.
///
/// This allows the supervisor to use Claude Code as a backend for agent execution.
/// Each `generate` call spawns a new Claude Code process with the prompt.
pub struct ClaudeCodeEngine {
    /// Claude tier (model selection).
    tier: ClaudeTier,
    /// Working directory for Claude Code.
    working_dir: PathBuf,
    /// System prompt to prepend.
    system_prompt: Option<String>,
    /// Allowed tools.
    allowed_tools: Vec<String>,
    /// Max turns per invocation.
    max_turns: Option<u32>,
    /// Model metadata (cached).
    metadata: ModelMetadata,
}

impl ClaudeCodeEngine {
    /// Creates a new Claude Code engine.
    pub fn new(tier: ClaudeTier, working_dir: impl Into<PathBuf>) -> Self {
        let working_dir = working_dir.into();
        // Use a placeholder architecture since Claude is a remote API
        let metadata = ModelMetadata::builder(
            tier.model_id(),
            ModelArchitecture::Llama { version: LlamaVersion::V3_2 }, // placeholder
        )
        .source(ModelSource::huggingface("anthropic/claude"))
        .context_length(200_000)
        .build();

        Self {
            tier,
            working_dir,
            system_prompt: None,
            allowed_tools: vec![],
            max_turns: None,
            metadata,
        }
    }

    /// Sets the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets allowed tools.
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }

    /// Sets max turns.
    pub fn with_max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Builds CLI arguments for Claude Code.
    fn build_args(&self, prompt: &str) -> Vec<String> {
        let mut args = vec![
            "-p".to_string(),
            prompt.to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
            "--model".to_string(),
            self.tier.model_id().to_string(),
        ];

        if let Some(ref sys) = self.system_prompt {
            args.extend([
                "--append-system-prompt".to_string(),
                sys.clone(),
            ]);
        }

        for tool in &self.allowed_tools {
            args.extend(["--allowedTools".to_string(), tool.clone()]);
        }

        if let Some(turns) = self.max_turns {
            args.extend(["--max-turns".to_string(), turns.to_string()]);
        }

        args
    }

    /// Extracts the final text response from Claude Code output.
    fn extract_response(lines: &[String]) -> String {
        let mut final_result = String::new();
        let mut assistant_texts = Vec::new();

        for line in lines {
            if let Ok(output) = serde_json::from_str::<ClaudeOutput>(line) {
                match output {
                    ClaudeOutput::Result(r) => {
                        final_result = r.result;
                    }
                    ClaudeOutput::Assistant(a) => {
                        for block in a.message.content {
                            if let ContentBlock::Text { text } = block {
                                assistant_texts.push(text);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Prefer the final result, fall back to concatenated assistant texts
        if !final_result.is_empty() {
            final_result
        } else if !assistant_texts.is_empty() {
            assistant_texts.join("\n")
        } else {
            String::new()
        }
    }
}

#[async_trait]
impl InferenceEngine for ClaudeCodeEngine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // Convert prompt input to a single prompt string
        let prompt = match &request.prompt {
            PromptInput::Text(text) => text.clone(),
            PromptInput::Messages(messages) => {
                // Extract user messages
                messages
                    .iter()
                    .filter(|m| m.role == Role::User)
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            PromptInput::Tokens(_) => {
                return Err(infernum_core::Error::internal(
                    "Claude Code does not support token input",
                ));
            }
        };

        if prompt.is_empty() {
            return Err(infernum_core::Error::internal("No prompt provided"));
        }

        let args = self.build_args(&prompt);
        debug!("Running claude with args: {:?}", args);

        // Spawn Claude Code
        let mut child = Command::new("claude")
            .args(&args)
            .current_dir(&self.working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| infernum_core::Error::internal(format!("Failed to spawn claude: {}", e)))?;

        let stdout = child.stdout.take().ok_or_else(|| {
            infernum_core::Error::internal("Failed to capture stdout")
        })?;

        let reader = BufReader::new(stdout);
        let mut lines_reader = reader.lines();
        let mut output_lines = Vec::new();

        // Read all output
        while let Ok(Some(line)) = lines_reader.next_line().await {
            debug!("Claude output: {}", &line[..line.len().min(200)]);
            output_lines.push(line);
        }

        // Wait for process to complete
        let status = child.wait().await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to wait for claude: {}", e))
        })?;

        if !status.success() {
            warn!("Claude Code exited with status: {:?}", status);
        }

        // Extract response text
        let response_text = Self::extract_response(&output_lines);

        if response_text.is_empty() {
            return Err(infernum_core::Error::internal("No response from Claude Code"));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Ok(GenerateResponse {
            request_id: request.request_id,
            created: now,
            model: ModelId::new(self.tier.model_id()),
            choices: vec![Choice {
                index: 0,
                text: response_text.clone(),
                message: Some(Message {
                    role: Role::Assistant,
                    content: response_text,
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }),
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 0, // Not available from CLI
                completion_tokens: 0,
                total_tokens: 0,
            },
            time_to_first_token_ms: None,
            total_time_ms: None,
        })
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<TokenStream> {
        // For now, just call generate and wrap the result
        // A proper implementation would stream the output
        let response = self.generate(request).await?;
        let text = response
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        // Create a single chunk with the full response
        let chunk = StreamChunkBuilder::new(response.request_id, response.model)
            .text(0, text)
            .build();

        Ok(TokenStream::once(chunk))
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
        Err(infernum_core::Error::internal("Embeddings not supported by Claude Code"))
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        // Check if claude binary exists
        std::process::Command::new("which")
            .arg("claude")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_model_id() {
        assert_eq!(ClaudeTier::Opus.model_id(), "claude-opus-4-5-20251101");
        assert_eq!(ClaudeTier::Sonnet.model_id(), "claude-sonnet-4-5-20250929");
        assert_eq!(ClaudeTier::Haiku.model_id(), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_build_args() {
        let engine = ClaudeCodeEngine::new(ClaudeTier::Opus, "/tmp");
        let args = engine.build_args("Hello");

        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--verbose".to_string()));
    }

    #[test]
    fn test_build_args_with_options() {
        let engine = ClaudeCodeEngine::new(ClaudeTier::Haiku, "/tmp")
            .with_system_prompt("Be helpful")
            .with_allowed_tools(vec!["Read".to_string()])
            .with_max_turns(5);

        let args = engine.build_args("Test");

        assert!(args.contains(&"--append-system-prompt".to_string()));
        assert!(args.contains(&"Be helpful".to_string()));
        assert!(args.contains(&"--allowedTools".to_string()));
        assert!(args.contains(&"Read".to_string()));
        assert!(args.contains(&"--max-turns".to_string()));
        assert!(args.contains(&"5".to_string()));
    }

    #[test]
    fn test_extract_response_from_result() {
        let lines = vec![
            r#"{"type":"system","session_id":"test"}"#.to_string(),
            r#"{"type":"result","result":"Hello from Claude","cost_usd":0.01}"#.to_string(),
        ];

        let response = ClaudeCodeEngine::extract_response(&lines);
        assert_eq!(response, "Hello from Claude");
    }

    #[test]
    fn test_extract_response_from_assistant() {
        let lines = vec![
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Test response"}]}}"#.to_string(),
        ];

        let response = ClaudeCodeEngine::extract_response(&lines);
        assert_eq!(response, "Test response");
    }

    #[test]
    fn test_is_ready() {
        let engine = ClaudeCodeEngine::new(ClaudeTier::Opus, "/tmp");
        // This should return true if claude is installed
        let ready = engine.is_ready();
        println!("Claude Code available: {}", ready);
    }
}
