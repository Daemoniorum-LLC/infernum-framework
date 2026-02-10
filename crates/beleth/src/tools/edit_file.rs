//! File editing tool (search-and-replace).

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;

use super::{optional_bool_param, require_str_param, validate_path};
use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};

/// Performs search-and-replace edits within a file.
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Performs exact string replacement in a file. Finds old_string and replaces it \
         with new_string. Fails if old_string is not found or matches multiple times \
         (unless replace_all is true)."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative to working directory)"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false, fails if not unique)"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::Write
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = require_str_param(&params, "path")?;
        let old_string = require_str_param(&params, "old_string")?;
        let new_string = require_str_param(&params, "new_string")?;
        let replace_all = optional_bool_param(&params, "replace_all").unwrap_or(false);

        if old_string == new_string {
            return Ok(ToolResult::error(
                "old_string and new_string are identical — no change needed",
            ));
        }

        let resolved = validate_path(path_str, ctx)?;

        let content = match tokio::fs::read_to_string(&resolved).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to read '{}': {}",
                    path_str, e
                )));
            },
        };

        let match_count = content.matches(old_string).count();

        if match_count == 0 {
            return Ok(ToolResult::error(format!(
                "old_string not found in '{}'. Verify the exact text including whitespace.",
                path_str
            )));
        }

        if match_count > 1 && !replace_all {
            return Ok(ToolResult::error(format!(
                "old_string matches {} times in '{}'. Use replace_all: true to replace all, \
                 or provide more context to make it unique.",
                match_count, path_str
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        match tokio::fs::write(&resolved, &new_content).await {
            Ok(()) => {
                let msg = format!(
                    "Replaced {} occurrence{} in '{}'",
                    match_count,
                    if match_count == 1 { "" } else { "s" },
                    path_str
                );
                Ok(ToolResult::success(msg).with_data(serde_json::json!({
                    "path": path_str,
                    "replacements": match_count,
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
    async fn test_edit_file_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("code.rs"),
            "fn hello() {\n    println!(\"hello\");\n}\n",
        )
        .expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = EditFileTool;
        let params = serde_json::json!({
            "path": "code.rs",
            "old_string": "println!(\"hello\")",
            "new_string": "println!(\"world\")"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "edit should succeed: {:?}", result.error);
        let content = std::fs::read_to_string(dir.path().join("code.rs")).expect("read");
        assert!(content.contains("println!(\"world\")"));
        assert!(!content.contains("println!(\"hello\")"));
    }

    #[tokio::test]
    async fn test_edit_file_not_unique() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("dup.txt"), "foo bar foo baz foo").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = EditFileTool;
        let params = serde_json::json!({
            "path": "dup.txt",
            "old_string": "foo",
            "new_string": "qux"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("3 times"));
    }

    #[tokio::test]
    async fn test_edit_file_replace_all() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("dup.txt"), "foo bar foo baz foo").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = EditFileTool;
        let params = serde_json::json!({
            "path": "dup.txt",
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": true
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        let content = std::fs::read_to_string(dir.path().join("dup.txt")).expect("read");
        assert_eq!(content, "qux bar qux baz qux");
    }

    #[tokio::test]
    async fn test_edit_file_not_found_in_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("file.txt"), "some content").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = EditFileTool;
        let params = serde_json::json!({
            "path": "file.txt",
            "old_string": "nonexistent text",
            "new_string": "replacement"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_file_identical_strings() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("file.txt"), "content").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = EditFileTool;
        let params = serde_json::json!({
            "path": "file.txt",
            "old_string": "same",
            "new_string": "same"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("identical"));
    }
}
