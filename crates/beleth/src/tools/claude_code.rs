//! Claude Code delegation tool.
//!
//! Delegates complex subtasks to the local Claude Code CLI (`claude`).
//! This allows the agent to call for expert help when it encounters
//! problems beyond its local model's capability — code review,
//! architectural reasoning, debugging, refactoring, etc.
//!
//! Requires `claude` to be installed and available on `$PATH`.

use std::time::Duration;

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;
use tokio::process::Command;

use super::{optional_str_param, optional_u64_param, require_str_param};
use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};

/// Default timeout for Claude Code invocations (5 minutes).
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Maximum timeout (15 minutes).
const MAX_TIMEOUT_SECS: u64 = 900;

/// Delegates tasks to the local Claude Code CLI.
pub struct ClaudeCodeTool {
    /// Path to the claude binary. Defaults to "claude" (found on $PATH).
    cli_path: String,
    /// Default model to request (passed via --model). None = use CLI default.
    model: Option<String>,
}

impl Default for ClaudeCodeTool {
    fn default() -> Self {
        Self {
            cli_path: "claude".to_string(),
            model: None,
        }
    }
}

impl ClaudeCodeTool {
    /// Creates a Claude Code tool with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the path to the `claude` binary.
    #[must_use]
    pub fn with_cli_path(mut self, path: impl Into<String>) -> Self {
        self.cli_path = path.into();
        self
    }

    /// Sets the model to use (e.g., "sonnet", "opus").
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[async_trait]
impl Tool for ClaudeCodeTool {
    fn name(&self) -> &str {
        "claude_code"
    }

    fn description(&self) -> &str {
        "Delegates a complex task to Claude Code, a capable AI coding assistant \
         running locally. Use this when you need help with tasks that require deep \
         reasoning, code review, architectural decisions, multi-file refactoring, \
         or when you're stuck on a problem. Claude Code has full access to the \
         project files and can read, write, and execute code."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear description of what you need help with. \
                                    Be specific about files, functions, or areas of code."
                },
                "context": {
                    "type": "string",
                    "description": "Additional context: error messages, relevant code \
                                    snippets, constraints, or background information"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 300, max: 900)"
                }
            },
            "required": ["task"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Dangerous
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let task = require_str_param(&params, "task")?;
        let context = optional_str_param(&params, "context");
        let timeout_secs = optional_u64_param(&params, "timeout_secs")
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        // Build the prompt
        let mut prompt = task.to_string();
        if let Some(ctx_text) = context {
            prompt.push_str("\n\n## Context\n\n");
            prompt.push_str(ctx_text);
        }

        // Build the claude command
        let mut cmd = Command::new(&self.cli_path);
        // -p = print mode (non-interactive, single prompt, stdout output)
        cmd.arg("-p").arg(&prompt);
        // --output-format json gives structured output with result + metadata
        cmd.arg("--output-format").arg("json");

        if let Some(ref model) = self.model {
            cmd.arg("--model").arg(model);
        }

        // Set working directory if available
        if let Some(wd) = ctx.get_state("working_dir").and_then(Value::as_str) {
            cmd.current_dir(wd);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to launch claude CLI (is it installed and on $PATH?): {e}"
                )));
            },
        };

        let timeout = Duration::from_secs(timeout_secs);
        let output_result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match output_result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if !output.status.success() {
                    let mut err_msg = format!(
                        "claude CLI exited with code {}",
                        output.status.code().unwrap_or(-1)
                    );
                    if !stderr.is_empty() {
                        err_msg.push_str(": ");
                        err_msg.push_str(&stderr);
                    }
                    return Ok(ToolResult::error(err_msg));
                }

                // Try to parse JSON output
                if let Ok(json_output) = serde_json::from_str::<Value>(&stdout) {
                    // Extract the result text from Claude Code's JSON output
                    let result_text = json_output["result"]
                        .as_str()
                        .unwrap_or_else(|| {
                            // Fall back to raw stdout if no "result" field
                            stdout.as_ref()
                        })
                        .to_string();

                    if result_text.is_empty() {
                        return Ok(ToolResult::error("Claude Code returned empty response"));
                    }

                    // Truncate very large responses
                    let mut output_text = result_text;
                    if output_text.len() > 100_000 {
                        output_text.truncate(100_000);
                        output_text.push_str("\n... [output truncated at 100KB]");
                    }

                    let mut data = serde_json::json!({});
                    if let Some(model) = json_output["model"].as_str() {
                        data["model"] = Value::String(model.to_string());
                    }
                    if let Some(input) = json_output["input_tokens"].as_u64() {
                        data["input_tokens"] = Value::Number(input.into());
                    }
                    if let Some(output) = json_output["output_tokens"].as_u64() {
                        data["output_tokens"] = Value::Number(output.into());
                    }
                    if let Some(cost) = json_output["cost_usd"].as_f64() {
                        data["cost_usd"] = serde_json::json!(cost);
                    }

                    Ok(ToolResult::success(output_text).with_data(data))
                } else {
                    // Non-JSON output — just return raw stdout
                    let mut output_text = stdout.to_string();
                    if output_text.is_empty() {
                        return Ok(ToolResult::error("Claude Code returned empty response"));
                    }
                    if output_text.len() > 100_000 {
                        output_text.truncate(100_000);
                        output_text.push_str("\n... [output truncated at 100KB]");
                    }
                    Ok(ToolResult::success(output_text))
                }
            },
            Ok(Err(e)) => Ok(ToolResult::error(format!(
                "Claude Code execution failed: {e}"
            ))),
            Err(_) => Ok(ToolResult::error(format!(
                "Claude Code timed out after {timeout_secs} seconds"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_code_tool_defaults() {
        let tool = ClaudeCodeTool::new();
        assert_eq!(tool.name(), "claude_code");
        assert_eq!(tool.risk_level(), RiskLevel::Dangerous);
        assert_eq!(tool.cli_path, "claude");
        assert!(tool.model.is_none());
    }

    #[test]
    fn test_claude_code_tool_with_model() {
        let tool = ClaudeCodeTool::new().with_model("opus");
        assert_eq!(tool.model.as_deref(), Some("opus"));
    }

    #[test]
    fn test_claude_code_tool_with_cli_path() {
        let tool = ClaudeCodeTool::new().with_cli_path("/usr/local/bin/claude");
        assert_eq!(tool.cli_path, "/usr/local/bin/claude");
    }

    #[test]
    fn test_claude_code_parameters_schema() {
        let tool = ClaudeCodeTool::new();
        let schema = tool.parameters_schema();
        assert_eq!(schema["required"], serde_json::json!(["task"]));
        assert!(schema["properties"]["task"].is_object());
        assert!(schema["properties"]["context"].is_object());
        assert!(schema["properties"]["timeout_secs"].is_object());
    }
}
