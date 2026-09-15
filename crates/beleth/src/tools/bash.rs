//! Shell command execution tool.

use std::time::Duration;

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;
use tokio::process::Command;

use super::{optional_str_param, optional_u64_param, require_str_param};
use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};

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
        // The contract is stated because the model has to be able to rely on
        // it: without the last sentence, a non-zero status is ambiguous
        // between "the command reported something" and "the tool broke", and
        // a model that guesses the second stops investigating. See
        // infernum-framework#20.
        "Executes a shell command and returns its output. Use for running builds, \
         tests, git commands, and other system operations. Commands run with sh -c. \
         Output includes stdout and stderr; if the command exits non-zero, its \
         status is appended as [exit status: N]. A non-zero exit is a result, \
         not a tool failure — read it and carry on."
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
                return Ok(ToolResult::error(format!("Failed to spawn command: {}", e)));
            },
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

                // The command ran. That makes this a successful tool call
                // whatever the command thought of its own work.
                //
                // `ToolResult::success` answers "did the tool do its job?".
                // Conflating it with "did the command exit zero?" told the
                // model its shell was broken every time a test failed, a
                // `grep` matched nothing, or a `false` did exactly what it
                // says on the tin — and the model believed it, because that
                // is a reasonable reading of what it was told. See
                // infernum-framework#20.
                //
                // The exit status is not swallowed, though. It goes in the
                // `output` text as well as `data`, because `output` is the
                // field that reliably reaches the model's context: the
                // agentic loop renders a failed result as a bare error string
                // and never serialises `data` at all.
                if !output.status.success() {
                    combined.push_str(&match output.status.code() {
                        Some(code) => format!("\n[exit status: {code}]"),
                        // No code means a signal killed it. Reporting the
                        // `-1` placeholder as if it were an exit status would
                        // be inventing a number the process never returned.
                        None => "\n[terminated by signal]".to_string(),
                    });
                }

                Ok(ToolResult::success(combined).with_data(serde_json::json!({
                    "exit_code": exit_code,
                    "command": command,
                })))
            },
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
            },
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

    /// CHANGED DELIBERATELY for infernum-framework#20 — this test used to
    /// assert `!result.success` for `exit 42`, which encoded the defect
    /// rather than guarding against it.
    ///
    /// `success` on a [`ToolResult`] answers "did the tool work?". A command
    /// that runs to completion and returns 42 is a tool that worked and a
    /// command that returned 42. Reporting it as a tool failure told the
    /// model its shell was broken, and the model believed it — see
    /// `exit-code-discipline` in `toolcall-eval`'s Phase B, 0/3.
    ///
    /// If a future change makes this assert `!result.success` again, that is
    /// this bug coming back. The exit code is still asserted below: not
    /// treating a non-zero exit as a failure must not mean hiding it.
    #[tokio::test]
    async fn a_non_zero_exit_is_a_result_not_a_tool_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "exit 42"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(
            result.success,
            "the command ran; `success` reports whether the TOOL worked, not \
             whether the command exited zero: {:?}",
            result.error
        );
        assert!(
            result.error.is_none(),
            "a non-zero exit is not a tool error: {:?}",
            result.error
        );

        let data = result.data.expect("data");
        assert_eq!(data["exit_code"], 42, "the exit code must still be carried");
    }

    /// The exit status has to be in `output`, because that is the field that
    /// reliably reaches the model's context. `data.exit_code` alone was not
    /// enough: the agentic loop renders a failed result as a bare
    /// `"Error: ..."` string and never serialises `data` at all.
    #[tokio::test]
    async fn the_exit_status_appears_in_the_text_the_model_reads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "exit 42"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(
            result.output.contains("42"),
            "the model reads `output`; an exit code only in `data` is invisible \
             to it: {:?}",
            result.output
        );
    }

    /// Both streams survive alongside a non-zero exit. The failing case is a
    /// compiler or test runner: its diagnostics go to stderr *and* it exits
    /// non-zero, which is exactly when the model most needs to read them.
    #[tokio::test]
    async fn output_of_a_failing_command_is_not_discarded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({
            "command": "echo to-stdout; echo to-stderr >&2; exit 3"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("to-stdout"), "{:?}", result.output);
        assert!(result.output.contains("to-stderr"), "{:?}", result.output);
        assert!(result.output.contains('3'), "{:?}", result.output);
    }

    /// A successful command stays clean — no status noise on the hot path.
    #[tokio::test]
    async fn a_zero_exit_carries_no_status_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        let params = serde_json::json!({"command": "echo hello"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(
            !result.output.contains("exit status"),
            "exit 0 is the overwhelming majority of calls; marking every one \
             of them is pure token cost: {:?}",
            result.output
        );
        assert_eq!(result.data.expect("data")["exit_code"], 0);
    }

    // === The other direction: a tool that genuinely could not run ===
    //
    // `success: false` has to keep meaning something. These pin the cases
    // where the tool really did fail, so widening "ran and exited non-zero"
    // into success cannot quietly swallow them too.

    #[tokio::test]
    async fn a_command_that_cannot_be_spawned_is_a_tool_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = BashTool::default();

        // A working_dir that does not exist: the shell is never started, so
        // there is no exit status to report and nothing ran.
        let params = serde_json::json!({
            "command": "echo hello",
            "working_dir": "no/such/directory"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(
            !result.success,
            "nothing ran, so this is a tool failure, not a command result"
        );
        assert!(result.error.is_some());
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
