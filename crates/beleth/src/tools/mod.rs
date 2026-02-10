//! Code-relevant tools for agent-driven software engineering.
//!
//! This module provides the tools an agent needs to read, write, search,
//! and execute code — the foundation for self-modifying capability.

mod bash;
mod claude_code;
mod edit_file;
mod list_files;
mod read_file;
mod search_files;
mod write_file;

pub use bash::BashTool;
pub use claude_code::ClaudeCodeTool;
pub use edit_file::EditFileTool;
pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use search_files::SearchFilesTool;
pub use write_file::WriteFileTool;

use std::path::{Path, PathBuf};

use infernum_core::Result;
use serde_json::Value;

use crate::tool::ToolContext;

/// Resolves and validates a path against the working directory boundary.
///
/// All file tools must call this before operating on any path. It:
/// 1. Resolves relative paths against the working directory
/// 2. Canonicalizes to resolve symlinks and `..` components
/// 3. Verifies the result is under the working directory
///
/// # Errors
///
/// Returns an error if:
/// - No `working_dir` is set in the tool context state
/// - The resolved path escapes the working directory boundary
/// - The path cannot be canonicalized (parent doesn't exist)
pub fn validate_path(path: &str, ctx: &ToolContext) -> Result<PathBuf> {
    let working_dir = ctx
        .get_state("working_dir")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            infernum_core::Error::internal(
                "No working_dir set in tool context. The agent runner must set \
                 ToolContext.state[\"working_dir\"] before executing file tools.",
            )
        })?;

    let working_dir = PathBuf::from(working_dir);
    let resolved = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        working_dir.join(path)
    };

    // For read operations, canonicalize the full path (must exist).
    // For write operations, canonicalize the parent (must exist) and append the filename.
    let canonical = if resolved.exists() {
        resolved.canonicalize().map_err(|e| {
            infernum_core::Error::internal(format!("Failed to resolve path '{}': {}", path, e))
        })?
    } else {
        // Parent must exist for write operations
        let parent = resolved.parent().ok_or_else(|| {
            infernum_core::Error::internal(format!("Invalid path '{}': no parent directory", path))
        })?;
        if parent.exists() {
            let canonical_parent = parent.canonicalize().map_err(|e| {
                infernum_core::Error::internal(format!(
                    "Failed to resolve parent of '{}': {}",
                    path, e
                ))
            })?;
            let file_name = resolved.file_name().ok_or_else(|| {
                infernum_core::Error::internal(format!("Invalid path '{}': no filename", path))
            })?;
            canonical_parent.join(file_name)
        } else {
            // For nested creates, resolve as far as possible
            resolved.clone()
        }
    };

    let canonical_working = working_dir.canonicalize().unwrap_or(working_dir);
    if !canonical.starts_with(&canonical_working) {
        return Err(infernum_core::Error::internal(format!(
            "Path '{}' escapes working directory boundary '{}'",
            path,
            canonical_working.display()
        )));
    }

    Ok(canonical)
}

/// Extracts a required string parameter from JSON params.
pub(crate) fn require_str_param<'a>(params: &'a Value, key: &str) -> Result<&'a str> {
    params.get(key).and_then(Value::as_str).ok_or_else(|| {
        infernum_core::Error::internal(format!("Missing required parameter: '{}'", key))
    })
}

/// Extracts an optional string parameter from JSON params.
pub(crate) fn optional_str_param<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(Value::as_str)
}

/// Extracts an optional u64 parameter from JSON params.
pub(crate) fn optional_u64_param(params: &Value, key: &str) -> Option<u64> {
    params.get(key).and_then(Value::as_u64)
}

/// Extracts an optional bool parameter from JSON params.
pub(crate) fn optional_bool_param(params: &Value, key: &str) -> Option<bool> {
    params.get(key).and_then(Value::as_bool)
}

#[cfg(test)]
pub(crate) mod tests_common {
    use std::collections::HashMap;
    use std::path::Path;

    use serde_json::Value;

    use crate::tool::{TaskComplexity, ToolContext, ToolTimeoutConfig};

    /// Creates a ToolContext with working_dir set to the given path.
    pub fn make_ctx_with_dir(dir: &Path) -> ToolContext {
        let mut state = HashMap::new();
        state.insert(
            "working_dir".to_string(),
            Value::String(dir.to_string_lossy().to_string()),
        );
        ToolContext {
            agent_id: "test".to_string(),
            messages: Vec::new(),
            state,
            timeout_config: ToolTimeoutConfig::default(),
            task_complexity: TaskComplexity::Moderate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use crate::tool::{TaskComplexity, ToolTimeoutConfig};

    fn make_ctx(working_dir: &str) -> ToolContext {
        let mut state = HashMap::new();
        state.insert(
            "working_dir".to_string(),
            Value::String(working_dir.to_string()),
        );
        ToolContext {
            agent_id: "test".to_string(),
            messages: Vec::new(),
            state,
            timeout_config: ToolTimeoutConfig::default(),
            task_complexity: TaskComplexity::Moderate,
        }
    }

    #[test]
    fn test_validate_path_relative() {
        let dir = std::env::temp_dir();
        let ctx = make_ctx(dir.to_str().expect("temp dir is utf-8"));

        // A file in the temp dir should resolve fine
        let result = validate_path("somefile.txt", &ctx);
        assert!(result.is_ok());
        let resolved = result.expect("should resolve");
        assert!(resolved.starts_with(&dir));
    }

    #[test]
    fn test_validate_path_escape_rejected() {
        let dir = std::env::temp_dir().join("beleth_test_boundary");
        std::fs::create_dir_all(&dir).ok();
        let ctx = make_ctx(dir.to_str().expect("temp dir is utf-8"));

        let result = validate_path("../../etc/passwd", &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("escapes working directory"),
            "Expected escape error, got: {}",
            err
        );

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_path_no_working_dir() {
        let ctx = ToolContext::new("test");
        let result = validate_path("file.txt", &ctx);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No working_dir set"));
    }

    #[test]
    fn test_require_str_param() {
        let params = serde_json::json!({"name": "hello", "count": 42});
        assert_eq!(require_str_param(&params, "name").expect("ok"), "hello");
        assert!(require_str_param(&params, "missing").is_err());
        assert!(require_str_param(&params, "count").is_err()); // not a string
    }
}
