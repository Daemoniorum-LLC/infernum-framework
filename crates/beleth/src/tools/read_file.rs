//! File reading tool.

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;

use super::{optional_u64_param, require_str_param, validate_path};
use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};

/// Reads file contents with optional line range.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the contents of a file. Returns lines with line numbers. \
         Use offset and limit to read specific sections of large files."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to working directory)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Defaults to 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Defaults to 2000."
                }
            },
            "required": ["path"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::ReadOnly
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = require_str_param(&params, "path")?;
        let offset = optional_u64_param(&params, "offset").unwrap_or(1).max(1) as usize;
        let limit = optional_u64_param(&params, "limit").unwrap_or(2000) as usize;

        let resolved = validate_path(path_str, ctx)?;

        if resolved.is_dir() {
            return Ok(ToolResult::error(format!(
                "'{}' is a directory, not a file. Use list_files to explore directories.",
                path_str
            )));
        }

        let content = match tokio::fs::read(&resolved).await {
            Ok(bytes) => bytes,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to read '{}': {}",
                    path_str, e
                )));
            },
        };

        // Binary detection: if >10% of first 8KB are null bytes, treat as binary
        let sample = &content[..content.len().min(8192)];
        let null_count = sample.iter().filter(|&&b| b == 0).count();
        if sample.len() > 32 && null_count * 10 > sample.len() {
            return Ok(ToolResult::error(format!(
                "'{}' appears to be a binary file ({} bytes)",
                path_str,
                content.len()
            )));
        }

        let text = String::from_utf8_lossy(&content);
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();

        // Apply offset (1-based) and limit
        let start = (offset - 1).min(total_lines);
        let end = (start + limit).min(total_lines);
        let selected = &lines[start..end];

        let mut output = String::new();
        for (i, line) in selected.iter().enumerate() {
            let line_num = start + i + 1;
            output.push_str(&format!("{:>6}\t{}\n", line_num, line));
        }

        if output.is_empty() {
            output = "(empty file)\n".to_string();
        }

        let mut result = ToolResult::success(output);
        result.data = Some(serde_json::json!({
            "path": path_str,
            "total_lines": total_lines,
            "lines_shown": end - start,
            "offset": start + 1,
        }));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests_common::make_ctx_with_dir;

    #[tokio::test]
    async fn test_read_file_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ReadFileTool;
        let params = serde_json::json!({"path": "hello.txt"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("line one"));
        assert!(result.output.contains("line three"));
        assert!(result.output.contains("     1\t"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_path = dir.path().join("numbers.txt");
        let content: String = (1..=100).map(|i| format!("line {}\n", i)).collect();
        std::fs::write(&file_path, &content).expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ReadFileTool;
        let params = serde_json::json!({"path": "numbers.txt", "offset": 50, "limit": 5});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("line 50"));
        assert!(result.output.contains("line 54"));
        assert!(!result.output.contains("line 49"));
        assert!(!result.output.contains("line 55"));

        let data = result.data.expect("data");
        assert_eq!(data["lines_shown"], 5);
        assert_eq!(data["total_lines"], 100);
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = ReadFileTool;
        let params = serde_json::json!({"path": "nonexistent.txt"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to read"));
    }

    #[tokio::test]
    async fn test_read_file_directory_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("subdir")).expect("mkdir");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ReadFileTool;
        let params = serde_json::json!({"path": "subdir"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("is a directory"));
    }

    #[tokio::test]
    async fn test_read_file_missing_param() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = ReadFileTool;
        let params = serde_json::json!({});
        let result = tool.execute(params, &ctx).await;

        assert!(result.is_err());
    }
}
