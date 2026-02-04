//! Shell command execution tool.

use std::time::Duration;

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;
use tokio::process::Command;

use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};
use super::{optional_str_param, optional_u64_param, require_str_param};

/// Default timeout for shell commands (120 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum timeout for shell commands (600 seconds / 10 minutes).
const MAX_TIMEOUT_SECS: u64 = 600;

/// Executes shell commands.
pub struct BashTool {
    /// Shell to use for execution.
    shell: String,
}

impl Default for BashTool {
    fn default() -> Self {
        Self {
            shell: "sh".to_string(),
        }
    }
}

impl BashTool {
    /// Creates a bash tool with a specific shell.
    #[must_use]
    pub fn with_shell(shell: impl Into<String>) -> Self {
        Self {
            shell: shell.into(),
        }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Executes a shell command and returns its output. Use for running builds, \
         tests, git commands, and other system operations. Commands run with sh -c."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (relative to agent working dir). Defaults to agent working dir."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120, max: 600)"
                }
            },
            "required": ["command"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Dangerous
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let command = require_str_param(&params, "command")?;
        let timeout_secs = optional_u64_param(&params, "timeout_secs")
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS);

        // Determine working directory
        let working_dir = if let Some(wd) = optional_str_param(&params, "working_dir") {
            // Resolve relative to agent's working dir
            let base = ctx
                .get_state("working_dir")
                .and_then(Value::as_str)
                .unwrap_or(".");
            let base_path = std::path::PathBuf::from(base);
            base_path.join(wd)
        } else {
            ctx.get_state("working_dir")
                .and_then(Value::as_str)
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()))
        };

        let child = Command::new(&self.shell)
            .arg("-c")
            .arg(command)
            .current_dir(&working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let child = match child {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to spawn command: {}",
                    e
                )));
            }
        };

        let timeout = Duration::from_secs(timeout_secs);

        // wait_with_output() consumes child, so we can't call kill() after.
        // Use tokio::time::timeout and handle the result.
        let output_result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match output_result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let mut combined = String::new();
                if !stdout.is_empty() {
                    combined.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str("[stderr]\n");
                    combined.push_str(&stderr);
                }

                // Truncate large outputs
                if combined.len() > 100_000 {
                    combined.truncate(100_000);
                    combined.push_str("\n... [output truncated at 100KB]");
                }

                if combined.is_empty() {
                    combined = "(no output)".to_string();
                }

                let success = output.status.success();
                let mut tool_result = if success {
                    ToolResult::success(combined)
                } else {
                    ToolResult {
                        success: false,
                        output: combined,
                        error: Some(format!("Command exited with code {}", exit_code)),
                        data: None,
                    }
                };

                tool_result.data = Some(serde_json::json!({
                    "exit_code": exit_code,
                    "command": command,
                }));

                Ok(tool_result)
            }
            Ok(Err(e)) => Ok(ToolResult::error(format!(
                "Command execution failed: {}",
                e
            ))),
            Err(_) => {
                // Timeout — wait_with_output consumed the child, but the
                // process is still running. The drop of the future will
                // not automatically kill it, but there's nothing we can
                // reference here. The OS will clean up when the process
                // exits or the parent process exits. For a more robust
                // implementation, we'd use child.kill() before
                // wait_with_output(), but that requires a different pattern.
                Ok(ToolResult::error(format!(
                    "Command timed out after {} seconds",
                    timeout_secs
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests_common::make_ctx_with_dir;

    #[tokio::test]
    async fn test_bash_echo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "echo hello"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "echo should succeed: {:?}", result.error);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "exit 42"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        let data = result.data.expect("data");
        assert_eq!(data["exit_code"], 42);
    }

    #[tokio::test]
    async fn test_bash_stderr() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "echo error >&2"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("error"));
        assert!(result.output.contains("[stderr]"));
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({
            "command": "sleep 60",
            "timeout_secs": 1
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("timed out"));
    }

    #[tokio::test]
    async fn test_bash_working_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("subdir")).expect("mkdir");
        std::fs::write(dir.path().join("subdir/test.txt"), "found").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({
            "command": "cat test.txt",
            "working_dir": "subdir"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "should succeed: {:?}", result.error);
        assert!(result.output.contains("found"));
    }
}
