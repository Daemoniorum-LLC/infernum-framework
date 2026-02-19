//! Claude Code command history API.
//!
//! Provides endpoints to search and browse Claude Code command history.
//!
//! ## Endpoints
//!
//! - `GET /api/claude/history` - List command history with optional filtering
//! - `GET /api/claude/history/search` - Search prompts by keyword
//! - `GET /api/claude/history/templates` - Get frequently used prompt patterns

use axum::{
    extract::Query,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;
use tracing::{debug, info, warn};

use crate::claude_discovery;

// =============================================================================
// API Query Parameters
// =============================================================================

/// Query parameters for history listing.
#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    /// Maximum number of entries to return (default: 100)
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Filter by project path
    pub project: Option<String>,
    /// Filter by search keyword
    pub search: Option<String>,
}

fn default_limit() -> usize {
    100
}

// =============================================================================
// API Response Types
// =============================================================================

/// History listing response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryResponse {
    /// List of history entries
    pub entries: Vec<HistoryEntry>,
    /// Total count of entries (before limit)
    pub total_count: usize,
}

/// A single history entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The prompt/command text
    pub display: String,
    /// Unix timestamp (milliseconds)
    pub timestamp: u64,
    /// Formatted timestamp for display
    pub timestamp_formatted: String,
    /// Project path where this was run
    pub project: String,
    /// Session ID
    pub session_id: String,
    /// Whether this prompt was pasted (vs typed)
    pub has_pasted_content: bool,
}

/// Prompt templates response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplates {
    /// Most frequent prompt patterns (prefix-based)
    pub frequent_patterns: Vec<PatternFrequency>,
    /// Recent unique prompts
    pub recent_unique: Vec<String>,
}

/// Frequency of a prompt pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFrequency {
    /// The pattern (first few words)
    pub pattern: String,
    /// Number of times used
    pub count: usize,
    /// When last used (formatted)
    pub last_used: String,
}

// =============================================================================
// Raw JSON Types (for parsing history.jsonl)
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawHistoryEntry {
    display: Option<String>,
    pasted_contents: Option<serde_json::Value>,
    timestamp: Option<u64>,
    project: Option<String>,
    session_id: Option<String>,
}

// =============================================================================
// Router
// =============================================================================

/// Create the Claude history router.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/", get(get_history))
        .route("/search", get(search_history))
        .route("/templates", get(get_templates))
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /api/claude/history
/// Returns command history with optional filtering.
pub async fn get_history(Query(params): Query<HistoryQuery>) -> Json<HistoryResponse> {
    let entries = load_history().await;

    // Apply filters
    let filtered: Vec<HistoryEntry> = entries
        .into_iter()
        .filter(|e| {
            // Filter by project
            if let Some(ref proj) = params.project {
                if !e.project.contains(proj) {
                    return false;
                }
            }
            // Filter by search
            if let Some(ref search) = params.search {
                if !e.display.to_lowercase().contains(&search.to_lowercase()) {
                    return false;
                }
            }
            true
        })
        .collect();

    let total_count = filtered.len();
    let entries: Vec<HistoryEntry> = filtered.into_iter().take(params.limit).collect();

    Json(HistoryResponse {
        entries,
        total_count,
    })
}

/// GET /api/claude/history/search
/// Search prompts by keyword.
pub async fn search_history(Query(params): Query<HistoryQuery>) -> Json<HistoryResponse> {
    // Same as get_history but search is required
    get_history(Query(params)).await
}

/// GET /api/claude/history/templates
/// Get frequently used prompt patterns.
pub async fn get_templates() -> Json<PromptTemplates> {
    let entries = load_history().await;

    // Extract patterns (first 3-5 words)
    let mut pattern_counts: HashMap<String, (usize, u64)> = HashMap::new();
    let mut unique_prompts: Vec<String> = Vec::new();

    for entry in entries.iter() {
        // Extract pattern (first few words)
        let pattern = extract_pattern(&entry.display);
        if !pattern.is_empty() {
            let (count, last_ts) = pattern_counts
                .entry(pattern.clone())
                .or_insert((0, entry.timestamp));
            *count += 1;
            if entry.timestamp > *last_ts {
                *last_ts = entry.timestamp;
            }
        }

        // Track unique prompts
        if !unique_prompts.iter().any(|p| p == &entry.display) {
            unique_prompts.push(entry.display.clone());
        }
    }

    // Sort patterns by count
    let mut frequent_patterns: Vec<PatternFrequency> = pattern_counts
        .into_iter()
        .filter(|(_, (count, _))| *count >= 2) // At least 2 uses
        .map(|(pattern, (count, last_ts))| PatternFrequency {
            pattern,
            count,
            last_used: format_timestamp(last_ts),
        })
        .collect();

    frequent_patterns.sort_by(|a, b| b.count.cmp(&a.count));
    frequent_patterns.truncate(20); // Top 20 patterns

    // Get recent unique prompts
    let recent_unique: Vec<String> = unique_prompts.into_iter().rev().take(10).collect();

    Json(PromptTemplates {
        frequent_patterns,
        recent_unique,
    })
}

// =============================================================================
// Data Loading
// =============================================================================

/// Load history from all discovered history.jsonl files across all data sources.
async fn load_history() -> Vec<HistoryEntry> {
    let history_paths = claude_discovery::get_all_history_paths().await;
    info!("Aggregating history from {} sources", history_paths.len());

    let mut entries = Vec::new();

    for (history_path, source_label) in history_paths {
        debug!("Loading history from source '{}': {:?}", source_label, history_path);

        let content = match fs::read_to_string(&history_path).await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read history file {:?}: {}", history_path, e);
                continue;
            }
        };

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(raw) = serde_json::from_str::<RawHistoryEntry>(line) {
                let display = raw.display.unwrap_or_default();
                let timestamp = raw.timestamp.unwrap_or(0);
                let mut project = raw.project.unwrap_or_default();
                let mut session_id = raw.session_id.unwrap_or_default();
                let has_pasted_content = raw
                    .pasted_contents
                    .map(|p| !p.is_null() && p.as_object().map(|o| !o.is_empty()).unwrap_or(false))
                    .unwrap_or(false);

                // Tag with source for multi-source disambiguation
                if source_label != "main" {
                    session_id = format!("[{}] {}", source_label, session_id);
                    if !project.is_empty() {
                        project = format!("[{}] {}", source_label, project);
                    }
                }

                entries.push(HistoryEntry {
                    display,
                    timestamp,
                    timestamp_formatted: format_timestamp(timestamp),
                    project,
                    session_id,
                    has_pasted_content,
                });
            }
        }
    }

    info!("Aggregated {} history entries from all sources", entries.len());

    // Sort by timestamp descending (most recent first)
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    entries
}

/// Extract a pattern from a prompt (first 3-5 words, normalized).
fn extract_pattern(prompt: &str) -> String {
    let words: Vec<&str> = prompt.split_whitespace().take(5).collect();

    // If it starts with a command-like pattern, use fewer words
    let pattern = if words.first().map(|w| w.starts_with('/')).unwrap_or(false) {
        words.into_iter().take(2).collect::<Vec<_>>().join(" ")
    } else if words.len() <= 3 {
        words.join(" ")
    } else {
        // Use first 3 words for longer prompts
        words.into_iter().take(3).collect::<Vec<_>>().join(" ")
    };

    // Normalize: lowercase, remove special chars except /
    pattern
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '/')
        .collect::<String>()
        .to_lowercase()
}

/// Format a Unix timestamp (milliseconds) as a readable string.
fn format_timestamp(timestamp: u64) -> String {
    if timestamp == 0 {
        return "Unknown".to_string();
    }

    // Convert milliseconds to seconds
    let secs = timestamp / 1000;

    // Calculate date/time from epoch
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let remaining = remaining % 3600;
    let minutes = remaining / 60;

    let mut year = 1970;
    let mut remaining_days = days as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let (month, day) = days_to_month_day(remaining_days as u32, is_leap_year(year));

    let month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_name = month_names[(month - 1) as usize];

    format!("{} {} {:02}:{:02}", month_name, day, hours, minutes)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_to_month_day(mut day: u32, leap: bool) -> (u32, u32) {
    let days_in_months: [u32; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for (i, &days) in days_in_months.iter().enumerate() {
        if day < days {
            return ((i + 1) as u32, day + 1);
        }
        day -= days;
    }

    (12, 31)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_pattern_slash_command() {
        let pattern = extract_pattern("/commit -m \"test message\"");
        assert!(pattern.starts_with("/commit"));
    }

    #[test]
    fn test_extract_pattern_normal() {
        let pattern = extract_pattern("Fix the bug in the authentication module");
        assert_eq!(pattern, "fix the bug");
    }

    #[test]
    fn test_extract_pattern_short() {
        let pattern = extract_pattern("Help me");
        assert_eq!(pattern, "help me");
    }

    #[test]
    fn test_format_timestamp() {
        // Jan 1, 2024 12:00:00 UTC in milliseconds
        let ts = 1704110400000u64;
        let formatted = format_timestamp(ts);
        assert!(formatted.contains("Jan"));
    }

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "Unknown");
    }
}
