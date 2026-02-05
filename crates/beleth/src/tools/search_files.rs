//! Regex search across files.

use std::path::Path;

use async_trait::async_trait;
use infernum_core::Result;
use regex::Regex;
use serde_json::Value;

use crate::tool::{RiskLevel, Tool, ToolContext, ToolResult};
use super::{optional_str_param, optional_u64_param, require_str_param, validate_path};

/// Default maximum number of matching lines to return.
const DEFAULT_MAX_RESULTS: u64 = 100;

/// Searches file contents using regex patterns.
pub struct SearchFilesTool;

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Searches file contents using regex patterns. Returns matching lines with \
         file path, line number, and content. Use file_glob to filter by file type."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (relative to working directory). Defaults to '.'."
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.rs', '*.toml')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return. Default: 100."
                }
            },
            "required": ["pattern"]
        })
    }

    fn risk_level(&self) -> RiskLevel {
        RiskLevel::ReadOnly
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolResult> {
        let pattern_str = require_str_param(&params, "pattern")?;
        let path_str = optional_str_param(&params, "path").unwrap_or(".");
        let file_glob = optional_str_param(&params, "file_glob");
        let max_results = optional_u64_param(&params, "max_results")
            .unwrap_or(DEFAULT_MAX_RESULTS) as usize;

        let regex = match Regex::new(pattern_str) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Invalid regex pattern '{}': {}",
                    pattern_str, e
                )));
            }
        };

        let resolved = validate_path(path_str, ctx)?;

        // Collect files to search
        let files = if let Some(glob_pat) = file_glob {
            let search_pattern = resolved.join(format!("**/{}", glob_pat));
            let pattern_string = search_pattern.to_string_lossy().to_string();
            match glob::glob(&pattern_string) {
                Ok(paths) => paths
                    .filter_map(std::result::Result::ok)
                    .filter(|p| p.is_file())
                    .collect::<Vec<_>>(),
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Invalid file_glob '{}': {}",
                        glob_pat, e
                    )));
                }
            }
        } else {
            collect_files_recursive(&resolved, 10)
        };

        let mut matches = Vec::new();
        let mut files_searched = 0usize;
        let mut files_with_matches = 0usize;

        for file_path in &files {
            if matches.len() >= max_results {
                break;
            }
            files_searched += 1;

            // Skip binary files
            let content = match std::fs::read(file_path) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };

            let sample = &content[..content.len().min(512)];
            let null_count = sample.iter().filter(|&&b| b == 0).count();
            if sample.len() > 32 && null_count * 10 > sample.len() {
                continue;
            }

            let text = match std::str::from_utf8(&content) {
                Ok(t) => t,
                Err(_) => continue,
            };

            let relative = file_path
                .strip_prefix(&resolved)
                .unwrap_or(file_path)
                .display()
                .to_string();

            let mut file_had_match = false;
            for (line_num, line) in text.lines().enumerate() {
                if matches.len() >= max_results {
                    break;
                }
                if regex.is_match(line) {
                    matches.push(format!("{}:{}:{}", relative, line_num + 1, line));
                    file_had_match = true;
                }
            }
            if file_had_match {
                files_with_matches += 1;
            }
        }

        let total_matches = matches.len();
        let mut output = matches.join("\n");

        if total_matches >= max_results {
            output.push_str(&format!(
                "\n... (truncated at {} results, searched {} files)",
                max_results, files_searched
            ));
        }

        if output.is_empty() {
            output = format!(
                "No matches for '{}' in {} files",
                pattern_str, files_searched
            );
        }

        Ok(ToolResult::success(output).with_data(serde_json::json!({
            "pattern": pattern_str,
            "matches": total_matches,
            "files_searched": files_searched,
            "files_with_matches": files_with_matches,
        })))
    }
}

/// Recursively collect files up to a given depth.
fn collect_files_recursive(dir: &Path, max_depth: usize) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    collect_files_inner(dir, max_depth, 0, &mut files);
    files
}

fn collect_files_inner(
    dir: &Path,
    max_depth: usize,
    current_depth: usize,
    files: &mut Vec<std::path::PathBuf>,
) {
    if current_depth > max_depth {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            // Skip hidden directories and common large directories
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with('.') && name != "target" && name != "node_modules" {
                collect_files_inner(&path, max_depth, current_depth + 1, files);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tests_common::make_ctx_with_dir;

    #[tokio::test]
    async fn test_search_files_basic() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.rs"), "fn main() {\n    hello();\n}\n").expect("write");
        std::fs::write(dir.path().join("b.rs"), "fn helper() {\n    world();\n}\n").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = SearchFilesTool;
        let params = serde_json::json!({"pattern": "fn \\w+"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("fn main"));
        assert!(result.output.contains("fn helper"));

        let data = result.data.expect("data");
        assert_eq!(data["matches"], 2);
        assert_eq!(data["files_with_matches"], 2);
    }

    #[tokio::test]
    async fn test_search_files_with_glob() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("code.rs"), "let x = 42;\n").expect("write");
        std::fs::write(dir.path().join("notes.txt"), "let x = 42;\n").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = SearchFilesTool;
        let params = serde_json::json!({
            "pattern": "let x",
            "file_glob": "*.rs"
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("code.rs"));
        assert!(!result.output.contains("notes.txt"));
    }

    #[tokio::test]
    async fn test_search_files_no_matches() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("file.txt"), "hello world\n").expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = SearchFilesTool;
        let params = serde_json::json!({"pattern": "nonexistent_pattern_xyz"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        assert!(result.output.contains("No matches"));
    }

    #[tokio::test]
    async fn test_search_files_invalid_regex() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ctx = make_ctx_with_dir(dir.path());
        let tool = SearchFilesTool;
        let params = serde_json::json!({"pattern": "[invalid"});
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Invalid regex"));
    }

    #[tokio::test]
    async fn test_search_files_max_results() {
        let dir = tempfile::tempdir().expect("tempdir");
        let content: String = (1..=50).map(|i| format!("match line {}\n", i)).collect();
        std::fs::write(dir.path().join("many.txt"), &content).expect("write");

        let ctx = make_ctx_with_dir(dir.path());
        let tool = SearchFilesTool;
        let params = serde_json::json!({
            "pattern": "match line",
            "max_results": 5
        });
        let result = tool.execute(params, &ctx).await.expect("execute");

        assert!(result.success);
        let data = result.data.expect("data");
        assert_eq!(data["matches"], 5);
        assert!(result.output.contains("truncated"));
    }
}
