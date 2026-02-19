//! Claude Code todo list aggregation API.
//!
//! Provides endpoints to aggregate and browse todos across all Claude Code sessions.
//!
//! ## Endpoints
//!
//! - `GET /api/claude/todos` - Aggregated todos from all sessions
//! - `GET /api/claude/todos/:session_id` - Todos for a specific session

use axum::{extract::Path, routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;
use tracing::{debug, info, warn};

use crate::claude_discovery;

// =============================================================================
// API Response Types
// =============================================================================

/// Aggregated todos response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedTodos {
    /// Todos grouped by status
    pub by_status: TodosByStatus,
    /// Todos grouped by session
    pub by_session: Vec<SessionTodos>,
    /// Count summaries
    pub counts: TodoCounts,
}

/// Todos organized by status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodosByStatus {
    pub in_progress: Vec<TodoWithSession>,
    pub pending: Vec<TodoWithSession>,
    pub completed: Vec<TodoWithSession>,
}

/// A todo item with session context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoWithSession {
    /// Todo content (what to do)
    pub content: String,
    /// Todo status: "pending", "in_progress", "completed"
    pub status: String,
    /// Active form (present continuous)
    pub active_form: String,
    /// Session ID this todo belongs to
    pub session_id: String,
    /// Project path for the session
    pub project_path: Option<String>,
}

/// Todos for a single session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTodos {
    /// Session ID
    pub session_id: String,
    /// Project path
    pub project_path: Option<String>,
    /// List of todos
    pub todos: Vec<Todo>,
    /// Last modification time (file mtime)
    pub last_updated: Option<String>,
}

/// A single todo item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: String,
    pub active_form: String,
}

/// Count summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoCounts {
    pub total: usize,
    pub in_progress: usize,
    pub pending: usize,
    pub completed: usize,
}

// =============================================================================
// Raw JSON Types (for parsing todo files)
// =============================================================================

#[derive(Debug, Deserialize)]
struct RawTodo {
    content: String,
    status: String,
    #[serde(rename = "activeForm")]
    active_form: Option<String>,
}

// =============================================================================
// Router
// =============================================================================

/// Create the Claude todos router.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/", get(get_all_todos))
        .route("/{session_id}", get(get_session_todos))
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /api/claude/todos
/// Returns aggregated todos from all sessions.
pub async fn get_all_todos() -> Json<AggregatedTodos> {
    let todos = load_all_todos().await;
    Json(todos)
}

/// GET /api/claude/todos/:session_id
/// Returns todos for a specific session.
pub async fn get_session_todos(Path(session_id): Path<String>) -> Json<Option<SessionTodos>> {
    let all_todos = load_all_todos().await;

    let session = all_todos
        .by_session
        .into_iter()
        .find(|s| s.session_id == session_id || s.session_id.starts_with(&session_id));

    Json(session)
}

// =============================================================================
// Data Loading
// =============================================================================

/// Load and aggregate todos from all session files across all data sources.
async fn load_all_todos() -> AggregatedTodos {
    let todos_paths = claude_discovery::get_all_todos_paths().await;
    info!("Aggregating todos from {} sources", todos_paths.len());

    let mut by_session = Vec::new();
    let mut in_progress = Vec::new();
    let mut pending = Vec::new();
    let mut completed = Vec::new();

    for (todos_dir, source_label) in todos_paths {
        debug!("Loading todos from source '{}': {:?}", source_label, todos_dir);

        // Read all .json files in todos directory
        let mut entries = match fs::read_dir(&todos_dir).await {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to read todos directory {:?}: {}", todos_dir, e);
                continue;
            }
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Some(mut session_todos) = parse_todo_file(&path).await {
                    // Tag with source for multi-source disambiguation
                    if source_label != "main" {
                        session_todos.session_id = format!("[{}] {}", source_label, session_todos.session_id);
                    }

                    // Add to by-status lists
                    for todo in &session_todos.todos {
                        let todo_with_session = TodoWithSession {
                            content: todo.content.clone(),
                            status: todo.status.clone(),
                            active_form: todo.active_form.clone(),
                            session_id: session_todos.session_id.clone(),
                            project_path: session_todos.project_path.clone(),
                        };

                        match todo.status.as_str() {
                            "in_progress" => in_progress.push(todo_with_session),
                            "pending" => pending.push(todo_with_session),
                            "completed" => completed.push(todo_with_session),
                            _ => pending.push(todo_with_session),
                        }
                    }

                    by_session.push(session_todos);
                }
            }
        }
    }

    // Sort by_session by last_updated descending
    by_session.sort_by(|a, b| {
        b.last_updated
            .as_ref()
            .cmp(&a.last_updated.as_ref())
    });

    let counts = TodoCounts {
        total: in_progress.len() + pending.len() + completed.len(),
        in_progress: in_progress.len(),
        pending: pending.len(),
        completed: completed.len(),
    };

    info!(
        "Aggregated {} todos ({} in-progress, {} pending, {} completed)",
        counts.total, counts.in_progress, counts.pending, counts.completed
    );

    AggregatedTodos {
        by_status: TodosByStatus {
            in_progress,
            pending,
            completed,
        },
        by_session,
        counts,
    }
}

/// Parse a todo file and extract session info.
async fn parse_todo_file(path: &std::path::PathBuf) -> Option<SessionTodos> {
    let content = fs::read_to_string(path).await.ok()?;
    let metadata = fs::metadata(path).await.ok()?;

    // Parse filename to extract session ID
    // Format: {session-id}-agent-{agent-id}.json or just {session-id}.json
    let filename = path.file_stem()?.to_str()?;
    let session_id = extract_session_id(filename);

    // Try to determine project path from session ID
    // This would require cross-referencing with session files
    let project_path = None;

    // Parse the JSON array of todos
    let raw_todos: Vec<RawTodo> = serde_json::from_str(&content).ok()?;

    let todos: Vec<Todo> = raw_todos
        .into_iter()
        .map(|raw| Todo {
            content: raw.content,
            status: raw.status,
            active_form: raw.active_form.unwrap_or_default(),
        })
        .collect();

    let last_updated = metadata
        .modified()
        .ok()
        .map(|t| format_system_time(t));

    Some(SessionTodos {
        session_id,
        project_path,
        todos,
        last_updated,
    })
}

/// Extract session ID from filename.
fn extract_session_id(filename: &str) -> String {
    // Handle format: {session-id}-agent-{agent-id}
    if let Some(idx) = filename.find("-agent-") {
        return filename[..idx].to_string();
    }
    filename.to_string()
}

/// Return empty aggregated todos.
fn empty_aggregated_todos() -> AggregatedTodos {
    AggregatedTodos {
        by_status: TodosByStatus {
            in_progress: Vec::new(),
            pending: Vec::new(),
            completed: Vec::new(),
        },
        by_session: Vec::new(),
        counts: TodoCounts {
            total: 0,
            in_progress: 0,
            pending: 0,
            completed: 0,
        },
    }
}

/// Format a SystemTime as an ISO 8601 string.
fn format_system_time(time: std::time::SystemTime) -> String {
    let duration = time
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO);
    let secs = duration.as_secs();

    // Calculate date/time from epoch
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let remaining = remaining % 3600;
    let minutes = remaining / 60;
    let seconds = remaining % 60;

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

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
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
    fn test_extract_session_id_with_agent() {
        let filename = "abc123-agent-456";
        assert_eq!(extract_session_id(filename), "abc123");
    }

    #[test]
    fn test_extract_session_id_simple() {
        let filename = "abc123";
        assert_eq!(extract_session_id(filename), "abc123");
    }

    #[test]
    fn test_empty_counts() {
        let todos = empty_aggregated_todos();
        assert_eq!(todos.counts.total, 0);
        assert_eq!(todos.counts.in_progress, 0);
        assert_eq!(todos.counts.pending, 0);
        assert_eq!(todos.counts.completed, 0);
    }
}
