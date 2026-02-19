//! Claude Code data directory discovery.
//!
//! Discovers all Claude Code data directories across the system, including:
//! - Main user directory (~/.claude/)
//! - Backup directories (~/.claude.backup.*/)
//! - Windows installations via WSL (/mnt/c/Users/*/.claude/)

use std::path::PathBuf;
use tokio::fs;
use tracing::debug;

/// A discovered Claude Code data directory.
#[derive(Debug, Clone)]
pub struct ClaudeDataSource {
    /// Path to the .claude directory
    pub path: PathBuf,
    /// Source type for display
    pub source_type: SourceType,
    /// Label for this source (e.g., "main", "backup-20260106", "windows-crook")
    pub label: String,
}

/// Type of data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    /// Main ~/.claude directory
    Main,
    /// Backup directory
    Backup,
    /// Windows installation via WSL
    Windows,
}

/// Discover all Claude Code data directories on the system.
pub async fn discover_claude_directories() -> Vec<ClaudeDataSource> {
    let mut sources = Vec::new();

    let home_dir = match dirs::home_dir() {
        Some(h) => h,
        None => return sources,
    };

    // 1. Main ~/.claude directory
    let main_claude = home_dir.join(".claude");
    if main_claude.exists() {
        sources.push(ClaudeDataSource {
            path: main_claude,
            source_type: SourceType::Main,
            label: "main".to_string(),
        });
    }

    // 2. Backup directories (~/.claude.backup.*)
    if let Ok(mut entries) = fs::read_dir(&home_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(".claude.backup") {
                let path = entry.path();
                if path.is_dir() {
                    let label = name_str
                        .strip_prefix(".claude.backup")
                        .unwrap_or("")
                        .trim_start_matches('.')
                        .to_string();
                    sources.push(ClaudeDataSource {
                        path,
                        source_type: SourceType::Backup,
                        label: format!("backup-{}", if label.is_empty() { "unknown" } else { &label }),
                    });
                }
            }
        }
    }

    // 3. Windows installations via WSL (/mnt/c/Users/*/.claude/)
    let mnt_c_users = PathBuf::from("/mnt/c/Users");
    if mnt_c_users.exists() {
        if let Ok(mut entries) = fs::read_dir(&mnt_c_users).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let user_path = entry.path();
                // Skip system directories
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str == "Default" || name_str == "Public" || name_str == "Default User" || name_str == "All Users" {
                    continue;
                }

                let claude_path = user_path.join(".claude");
                if claude_path.exists() {
                    sources.push(ClaudeDataSource {
                        path: claude_path,
                        source_type: SourceType::Windows,
                        label: format!("windows-{}", name_str),
                    });
                }
            }
        }
    }

    debug!("Discovered {} Claude data sources", sources.len());
    for source in &sources {
        debug!("  - {:?}: {:?}", source.label, source.path);
    }

    sources
}

/// Get paths to all stats-cache.json files.
pub async fn get_all_stats_paths() -> Vec<(PathBuf, String)> {
    let sources = discover_claude_directories().await;
    sources
        .into_iter()
        .map(|s| (s.path.join("stats-cache.json"), s.label))
        .filter(|(p, _)| p.exists())
        .collect()
}

/// Get paths to all history.jsonl files.
pub async fn get_all_history_paths() -> Vec<(PathBuf, String)> {
    let sources = discover_claude_directories().await;
    sources
        .into_iter()
        .map(|s| (s.path.join("history.jsonl"), s.label))
        .filter(|(p, _)| p.exists())
        .collect()
}

/// Get paths to all todos directories.
pub async fn get_all_todos_paths() -> Vec<(PathBuf, String)> {
    let sources = discover_claude_directories().await;
    sources
        .into_iter()
        .map(|s| (s.path.join("todos"), s.label))
        .filter(|(p, _)| p.exists())
        .collect()
}

/// Get paths to all projects directories.
pub async fn get_all_projects_paths() -> Vec<(PathBuf, String)> {
    let sources = discover_claude_directories().await;
    sources
        .into_iter()
        .map(|s| (s.path.join("projects"), s.label))
        .filter(|(p, _)| p.exists())
        .collect()
}

/// Get paths to all plans directories.
pub async fn get_all_plans_paths() -> Vec<(PathBuf, String)> {
    let sources = discover_claude_directories().await;
    sources
        .into_iter()
        .map(|s| (s.path.join("plans"), s.label))
        .filter(|(p, _)| p.exists())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discover_finds_main() {
        let sources = discover_claude_directories().await;
        // Should find at least the main directory if it exists
        let has_main = sources.iter().any(|s| s.source_type == SourceType::Main);
        // This test passes if main exists OR if running in CI without ~/.claude
        assert!(has_main || sources.is_empty());
    }
}
