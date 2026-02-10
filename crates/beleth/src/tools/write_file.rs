//! File writing tool.

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;

use super::{require_str_param, validate_path};
use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};

/// Creates or overwrites a file.
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Creates or overwrites a file with the given content. \
         Creates parent directories if they don't exist."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write to (relative to working directory)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Write
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = require_str_param(&params, "path")?;
        let content = require_str_param(&params, "content")?;

        let resolved = validate_path(path_str, ctx)?;

        // Create parent directories if needed
        if let Some(parent) = resolved.parent() {
            if !parent.exists() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return Ok(ToolResult::error(format!(
                        "Failed to create directories for '{}': {}",
                        path_str, e
                    )));
                }
            }
        }

        let byte_count = content.len();
        match tokio::fs::write(&resolved, content).await {
            Ok(()) => {
                let msg = format!("Wrote {} bytes to '{}'", byte_count, path_str);
                Ok(ToolResult::success(msg).with_data(serde_json::json!({
                    "path": path_str,
                    "bytes_written": byte_count,
                })))
            },
            Err(e) => Ok(ToolResult::error(format!(
                "Failed to write '{}': {}",
                path_str, e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests_common::make_ctx_with_dir;

    #[tokio::test]
    async fn test_write_file_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = WriteFileTool;

        let params = serde_json::json!({
            "path": "output.txt",
            "content": "Hello, Infernum!"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "write should succeed: {:?}", result.error);
        assert!(result.output.contains("16 bytes"));

        let written = std::fs::read_to_string(dir.path().join("output.txt")).expect("read back");
        assert_eq!(written, "Hello, Infernum!");
    }

    #[tokio::test]
    async fn test_write_file_creates_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = WriteFileTool;

        let params = serde_json::json!({
            "path": "deep/nested/dir/file.txt",
            "content": "nested content"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "write should succeed: {:?}", result.error);
        let written =
            std::fs::read_to_string(dir.path().join("deep/nested/dir/file.txt")).expect("read");
        assert_eq!(written, "nested content");
    }

    #[tokio::test]
    async fn test_write_file_overwrites() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("existing.txt"), "old content").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = WriteFileTool;

        let params = serde_json::json!({
            "path": "existing.txt",
            "content": "new content"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        let written = std::fs::read_to_string(dir.path().join("existing.txt")).expect("read");
        assert_eq!(written, "new content");
    }
}
