//! Directory listing and glob matching tool.

use async_trait::async_trait;
use infernum_core::Result;
use serde_json::Value;

use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};
use super::{optional_str_param, validate_path};

/// Maximum number of entries to return.
const MAX_ENTRIES: usize = 1000;

/// Lists directory contents or matches files by glob pattern.
pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "Lists directory contents or finds files matching a glob pattern. \
         Without a pattern, lists the directory. With a pattern like '**/*.rs', \
         finds all matching files recursively."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (relative to working directory). Defaults to '.'."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., '**/*.rs', 'src/**/*.toml')"
                }
            }
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::ReadOnly
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let path_str = optional_str_param(&params, "path").unwrap_or(".");
        let pattern = optional_str_param(&params, "pattern");

        let resolved = validate_path(path_str, ctx)?;

        if let Some(glob_pattern) = pattern {
            // Glob mode: find files matching pattern
            let search_pattern = resolved.join(glob_pattern);
            let pattern_str = search_pattern.to_string_lossy().to_string();

            let entries = match glob::glob(&pattern_str) {
                Ok(paths) => paths,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Invalid glob pattern '{}': {}",
                        glob_pattern, e
                    )));
                }
            };

            let mut results = Vec::new();
            for entry in entries {
                if results.len() >= MAX_ENTRIES {
                    break;
                }
                if let Ok(path) = entry {
                    // Show path relative to the search directory
                    if let Ok(relative) = path.strip_prefix(&resolved) {
                        results.push(relative.display().to_string());
                    } else {
                        results.push(path.display().to_string());
                    }
                }
            }

            let count = results.len();
            let mut output = results.join("\n");
            if count >= MAX_ENTRIES {
                output.push_str(&format!("\n... (truncated at {} entries)", MAX_ENTRIES));
            }
            if output.is_empty() {
                output = format!("No files matching '{}' in '{}'", glob_pattern, path_str);
            }

            Ok(ToolResult::success(output).with_data(serde_json::json!({
                "count": count,
                "pattern": glob_pattern,
                "path": path_str,
            })))
        } else {
            // Directory listing mode
            if !resolved.is_dir() {
                return Ok(ToolResult::error(format!(
                    "'{}' is not a directory",
                    path_str
                )));
            }

            let mut entries = Vec::new();
            let mut read_dir = match tokio::fs::read_dir(&resolved).await {
                Ok(rd) => rd,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Failed to read directory '{}': {}",
                        path_str, e
                    )));
                }
            };

            while let Ok(Some(entry)) = read_dir.next_entry().await {
                if entries.len() >= MAX_ENTRIES {
                    break;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                let file_type = entry.file_type().await.ok();
                let suffix = match file_type {
                    Some(ft) if ft.is_dir() => "/",
                    Some(ft) if ft.is_symlink() => "@",
                    _ => "",
                };
                entries.push(format!("{}{}", name, suffix));
            }

            entries.sort();
            let count = entries.len();
            let mut output = entries.join("\n");
            if count >= MAX_ENTRIES {
                output.push_str(&format!("\n... (truncated at {} entries)", MAX_ENTRIES));
            }
            if output.is_empty() {
                output = format!("'{}' is empty", path_str);
            }

            Ok(ToolResult::success(output).with_data(serde_json::json!({
                "count": count,
                "path": path_str,
            })))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests_common::make_ctx_with_dir;

    #[tokio::test]
    async fn test_list_files_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.txt"), "").expect("write");
        std::fs::write(dir.path().join("b.rs"), "").expect("write");
        std::fs::create_dir_all(dir.path().join("subdir")).expect("mkdir");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ListFilesTool;
        let params = serde_json::json!({"path": "."});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("a.txt"));
        assert!(result.output.contains("b.rs"));
        assert!(result.output.contains("subdir/"));
    }

    #[tokio::test]
    async fn test_list_files_glob_pattern() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join("src/main.rs"), "").expect("write");
        std::fs::write(dir.path().join("src/lib.rs"), "").expect("write");
        std::fs::write(dir.path().join("README.md"), "").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ListFilesTool;
        let params = serde_json::json!({"path": ".", "pattern": "**/*.rs"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success, "glob should succeed: {:?}", result.error);
        assert!(result.output.contains("main.rs"));
        assert!(result.output.contains("lib.rs"));
        assert!(!result.output.contains("README.md"));
    }

    #[tokio::test]
    async fn test_list_files_not_a_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("file.txt"), "content").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ListFilesTool;
        let params = serde_json::json!({"path": "file.txt"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not a directory"));
    }

    #[tokio::test]
    async fn test_list_files_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("empty")).expect("mkdir");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = ListFilesTool;
        let params = serde_json::json!({"path": "empty"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("is empty"));
    }
}
