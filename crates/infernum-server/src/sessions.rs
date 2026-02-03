//! Agent Sessions HTTP endpoints for monitoring active agent runs.
//!
//! Provides a central dashboard view of all agent activity with real-time event streaming.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::error_response::{api_error, ErrorCode};

// =============================================================================
// Data Types
// =============================================================================

/// Agent session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session is actively processing.
    Running,
    /// Waiting for tool approval.
    AwaitingApproval,
    /// Completed successfully.
    Completed,
    /// Failed with error.
    Failed,
    /// Cancelled by user.
    Cancelled,
}

/// Event counts by type within a session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCounts {
    /// Number of thought events.
    pub thoughts: u32,
    /// Number of tool call events.
    pub tool_calls: u32,
    /// Number of tool result events.
    pub tool_results: u32,
    /// Number of error events.
    pub errors: u32,
}

/// Agent session representing a running agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSession {
    /// Unique session identifier.
    pub id: String,
    /// Human-readable name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The objective/task being executed.
    pub objective: String,
    /// Current session status.
    pub status: SessionStatus,
    /// Tools enabled for this session.
    pub tools: Vec<String>,
    /// Working directory (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    /// Session creation timestamp (Unix ms).
    pub created_at: u64,
    /// Last activity timestamp (Unix ms).
    pub updated_at: u64,
    /// Current iteration number.
    pub iteration: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Event count by type.
    pub event_counts: EventCounts,
    /// Client identifier (for multi-tenant scenarios).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl AgentSession {
    /// Creates a new session with the given ID and objective.
    pub fn new(id: String, objective: String) -> Self {
        let now = now_ms();
        Self {
            id,
            name: None,
            objective,
            status: SessionStatus::Running,
            tools: Vec::new(),
            working_dir: None,
            created_at: now,
            updated_at: now,
            iteration: 0,
            max_iterations: 10,
            event_counts: EventCounts::default(),
            client_id: None,
        }
    }

    /// Sets the session name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the tools.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets the max iterations.
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }
}

/// Agent event types for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEventData {
    /// Agent thinking/reasoning.
    Thought {
        /// The thought content.
        content: String,
    },
    /// Tool invocation.
    ToolCall {
        /// Tool call identifier.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: serde_json::Value,
    },
    /// Tool execution result.
    ToolResult {
        /// Tool call identifier.
        id: String,
        /// Tool output value.
        output: serde_json::Value,
        /// Whether the tool executed successfully.
        success: bool,
    },
    /// Error occurred.
    Error {
        /// Error message.
        message: String,
    },
}

/// Session events broadcast to subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SessionEvent {
    /// New session started.
    SessionStarted {
        /// The newly created session.
        session: AgentSession,
    },
    /// Session state updated.
    SessionUpdated {
        /// Session identifier.
        session_id: String,
        /// Current session status.
        status: SessionStatus,
        /// Current iteration number.
        iteration: u32,
        /// Updated event counts.
        event_counts: EventCounts,
    },
    /// Agent event occurred within session.
    AgentEvent {
        /// Session identifier.
        session_id: String,
        /// The agent event data.
        #[serde(flatten)]
        data: AgentEventData,
    },
    /// Session ended.
    SessionEnded {
        /// Session identifier.
        session_id: String,
        /// Final session status.
        status: SessionStatus,
        /// Total session duration in milliseconds.
        duration_ms: u64,
        /// Final answer produced by the agent, if any.
        final_answer: Option<String>,
    },
}

// =============================================================================
// Session Registry
// =============================================================================

/// Registry for tracking active agent sessions.
pub struct SessionRegistry {
    /// Active sessions (id -> session).
    sessions: RwLock<HashMap<String, AgentSession>>,
    /// Broadcast channel for session events.
    event_tx: broadcast::Sender<SessionEvent>,
}

impl Default for SessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionRegistry {
    /// Creates a new session registry.
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        Self {
            sessions: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Registers a new session.
    pub async fn register_session(&self, session: AgentSession) {
        let id = session.id.clone();
        self.sessions.write().await.insert(id, session.clone());
        let _ = self.event_tx.send(SessionEvent::SessionStarted { session });
    }

    /// Emits an event for a session.
    pub async fn emit_event(&self, session_id: &str, event: AgentEventData) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            // Update event counts
            match &event {
                AgentEventData::Thought { .. } => session.event_counts.thoughts += 1,
                AgentEventData::ToolCall { .. } => session.event_counts.tool_calls += 1,
                AgentEventData::ToolResult { .. } => session.event_counts.tool_results += 1,
                AgentEventData::Error { .. } => session.event_counts.errors += 1,
            }
            session.updated_at = now_ms();

            // Broadcast update
            let _ = self.event_tx.send(SessionEvent::SessionUpdated {
                session_id: session_id.to_string(),
                status: session.status,
                iteration: session.iteration,
                event_counts: session.event_counts.clone(),
            });
        }

        // Broadcast the actual event
        let _ = self.event_tx.send(SessionEvent::AgentEvent {
            session_id: session_id.to_string(),
            data: event,
        });
    }

    /// Updates session iteration.
    pub async fn set_iteration(&self, session_id: &str, iteration: u32) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.iteration = iteration;
            session.updated_at = now_ms();
        }
    }

    /// Updates session status.
    pub async fn set_status(&self, session_id: &str, status: SessionStatus) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.status = status;
            session.updated_at = now_ms();
        }
    }

    /// Ends a session.
    pub async fn end_session(
        &self,
        session_id: &str,
        status: SessionStatus,
        final_answer: Option<String>,
    ) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.status = status;
            session.updated_at = now_ms();
            let duration_ms = session.updated_at.saturating_sub(session.created_at);

            let _ = self.event_tx.send(SessionEvent::SessionEnded {
                session_id: session_id.to_string(),
                status,
                duration_ms,
                final_answer,
            });
        }
    }

    /// Gets a session by ID.
    pub async fn get_session(&self, session_id: &str) -> Option<AgentSession> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// Lists all active sessions.
    pub async fn list_sessions(&self) -> Vec<AgentSession> {
        self.sessions.read().await.values().cloned().collect()
    }

    /// Subscribes to all session events.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.event_tx.subscribe()
    }

    /// Cancels a running session.
    pub async fn cancel_session(&self, session_id: &str) -> Result<(), &'static str> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            if session.status != SessionStatus::Running
                && session.status != SessionStatus::AwaitingApproval
            {
                return Err("Session is not running");
            }
            session.status = SessionStatus::Cancelled;
            session.updated_at = now_ms();
            let duration_ms = session.updated_at.saturating_sub(session.created_at);

            let _ = self.event_tx.send(SessionEvent::SessionEnded {
                session_id: session_id.to_string(),
                status: SessionStatus::Cancelled,
                duration_ms,
                final_answer: None,
            });
            Ok(())
        } else {
            Err("Session not found")
        }
    }

    /// Generates a new session ID.
    pub fn generate_id() -> String {
        format!("sess_{}", uuid::Uuid::new_v4().simple().to_string()[..12].to_string())
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// Query parameters for listing sessions.
#[derive(Debug, Deserialize)]
pub struct ListSessionsQuery {
    /// Filter by status.
    pub status: Option<String>,
    /// Maximum results.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Pagination offset.
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

/// Response for listing sessions.
#[derive(Debug, Serialize)]
pub struct ListSessionsResponse {
    /// Sessions matching the query.
    pub sessions: Vec<AgentSession>,
    /// Total number of sessions.
    pub total: usize,
}

/// Response for getting a single session.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetSessionResponse {
    /// The session.
    pub session: AgentSession,
    /// Recent events (placeholder - would need event history).
    pub recent_events: Vec<serde_json::Value>,
}

/// Response for cancel operation.
#[derive(Debug, Serialize)]
pub struct CancelResponse {
    /// Result status.
    pub status: String,
    /// Session ID.
    pub session_id: String,
}

/// Query parameters for SSE streams.
#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    /// Filter by session IDs (comma-separated).
    pub session_ids: Option<String>,
    /// Filter by event types (comma-separated).
    pub event_types: Option<String>,
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /api/agent/sessions - List all active sessions.
pub async fn list_sessions(
    State(registry): State<Arc<SessionRegistry>>,
    Query(query): Query<ListSessionsQuery>,
) -> impl IntoResponse {
    let mut sessions = registry.list_sessions().await;

    // Filter by status if provided
    if let Some(status_str) = &query.status {
        let filter_status = match status_str.as_str() {
            "running" => Some(SessionStatus::Running),
            "awaiting_approval" => Some(SessionStatus::AwaitingApproval),
            "completed" => Some(SessionStatus::Completed),
            "failed" => Some(SessionStatus::Failed),
            "cancelled" => Some(SessionStatus::Cancelled),
            _ => None,
        };
        if let Some(status) = filter_status {
            sessions.retain(|s| s.status == status);
        }
    }

    let total = sessions.len();

    // Apply pagination
    let sessions: Vec<_> = sessions
        .into_iter()
        .skip(query.offset)
        .take(query.limit)
        .collect();

    Json(ListSessionsResponse { sessions, total })
}

/// GET /api/agent/sessions/{session_id} - Get session details.
pub async fn get_session(
    State(registry): State<Arc<SessionRegistry>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match registry.get_session(&session_id).await {
        Some(session) => Json(GetSessionResponse {
            session,
            recent_events: Vec::new(), // TODO: implement event history
        })
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(api_error(ErrorCode::NotFound, "Session not found")),
        )
            .into_response(),
    }
}

/// GET /api/agent/sessions/stream - SSE stream of all session events.
pub async fn sessions_stream(
    State(registry): State<Arc<SessionRegistry>>,
    Query(query): Query<StreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let rx = registry.subscribe();
    let stream = BroadcastStream::new(rx);

    // Parse filter parameters
    let session_filter: Option<Vec<String>> = query
        .session_ids
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
    let event_filter: Option<Vec<String>> = query
        .event_types
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

    // Convert broadcast stream events to SSE events
    let sse_stream = stream
        .filter_map(move |result| {
            let session_filter = session_filter.clone();
            let event_filter = event_filter.clone();

            match result {
                Ok(event) => {
                    let session_id = match &event {
                        SessionEvent::SessionStarted { session } => session.id.clone(),
                        SessionEvent::SessionUpdated { session_id, .. } => session_id.clone(),
                        SessionEvent::AgentEvent { session_id, .. } => session_id.clone(),
                        SessionEvent::SessionEnded { session_id, .. } => session_id.clone(),
                    };

                    let event_type = match &event {
                        SessionEvent::SessionStarted { .. } => "session_started",
                        SessionEvent::SessionUpdated { .. } => "session_updated",
                        SessionEvent::AgentEvent { .. } => "agent_event",
                        SessionEvent::SessionEnded { .. } => "session_ended",
                    };

                    // Apply session filter
                    let session_ok = session_filter
                        .as_ref()
                        .map(|f| f.contains(&session_id))
                        .unwrap_or(true);

                    // Apply event type filter
                    let event_ok = event_filter
                        .as_ref()
                        .map(|f| f.iter().any(|t| t == event_type))
                        .unwrap_or(true);

                    if session_ok && event_ok {
                        serde_json::to_string(&event).ok().map(|data| {
                            Ok::<_, std::convert::Infallible>(
                                Event::default().event(event_type).data(data),
                            )
                        })
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        });

    Sse::new(sse_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

/// GET /api/agent/sessions/{session_id}/stream - SSE stream for a single session.
pub async fn session_stream(
    State(registry): State<Arc<SessionRegistry>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    // Check if session exists
    if registry.get_session(&session_id).await.is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(api_error(ErrorCode::NotFound, "Session not found")),
        )
            .into_response();
    }

    let rx = registry.subscribe();
    let stream = BroadcastStream::new(rx);
    let filter_id = session_id.clone();

    let sse_stream = stream.filter_map(move |result| {
        let filter_id = filter_id.clone();

        match result {
            Ok(event) => {
                let event_session_id = match &event {
                    SessionEvent::SessionStarted { session } => session.id.clone(),
                    SessionEvent::SessionUpdated { session_id, .. } => session_id.clone(),
                    SessionEvent::AgentEvent { session_id, .. } => session_id.clone(),
                    SessionEvent::SessionEnded { session_id, .. } => session_id.clone(),
                };

                if event_session_id != filter_id {
                    None
                } else {
                    let event_type = match &event {
                        SessionEvent::SessionStarted { .. } => "session_started",
                        SessionEvent::SessionUpdated { .. } => "session_updated",
                        SessionEvent::AgentEvent { .. } => "agent_event",
                        SessionEvent::SessionEnded { .. } => "session_ended",
                    };

                    serde_json::to_string(&event).ok().map(|data| {
                        Ok::<_, std::convert::Infallible>(
                            Event::default().event(event_type).data(data),
                        )
                    })
                }
            }
            Err(_) => None,
        }
    });

    Sse::new(sse_stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response()
}

/// POST /api/agent/sessions/{session_id}/cancel - Cancel a running session.
pub async fn cancel_session(
    State(registry): State<Arc<SessionRegistry>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match registry.cancel_session(&session_id).await {
        Ok(()) => Json(CancelResponse {
            status: "cancelled".to_string(),
            session_id,
        })
        .into_response(),
        Err("Session not found") => (
            StatusCode::NOT_FOUND,
            Json(api_error(ErrorCode::NotFound, "Session not found")),
        )
            .into_response(),
        Err("Session is not running") => (
            StatusCode::BAD_REQUEST,
            Json(api_error(
                ErrorCode::InvalidRequest,
                "Session is not running",
            )),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(api_error(ErrorCode::InternalError, e)),
        )
            .into_response(),
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Returns current time in milliseconds since Unix epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_status_serialization() {
        assert_eq!(
            serde_json::to_string(&SessionStatus::Running).unwrap(),
            "\"running\""
        );
        assert_eq!(
            serde_json::to_string(&SessionStatus::AwaitingApproval).unwrap(),
            "\"awaiting_approval\""
        );
    }

    #[test]
    fn test_agent_session_new() {
        let session = AgentSession::new("sess_test".to_string(), "Test objective".to_string());
        assert_eq!(session.id, "sess_test");
        assert_eq!(session.objective, "Test objective");
        assert_eq!(session.status, SessionStatus::Running);
        assert_eq!(session.iteration, 0);
    }

    #[test]
    fn test_agent_session_builder() {
        let session = AgentSession::new("sess_test".to_string(), "Test".to_string())
            .with_name("Test Session")
            .with_tools(vec!["file_read".to_string(), "grep".to_string()])
            .with_max_iterations(20);

        assert_eq!(session.name, Some("Test Session".to_string()));
        assert_eq!(session.tools.len(), 2);
        assert_eq!(session.max_iterations, 20);
    }

    #[test]
    fn test_event_counts_default() {
        let counts = EventCounts::default();
        assert_eq!(counts.thoughts, 0);
        assert_eq!(counts.tool_calls, 0);
        assert_eq!(counts.tool_results, 0);
        assert_eq!(counts.errors, 0);
    }

    #[test]
    fn test_session_registry_generate_id() {
        let id1 = SessionRegistry::generate_id();
        let id2 = SessionRegistry::generate_id();
        assert!(id1.starts_with("sess_"));
        assert!(id2.starts_with("sess_"));
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn test_session_registry_register() {
        let registry = SessionRegistry::new();
        let session =
            AgentSession::new("sess_test".to_string(), "Test objective".to_string());

        registry.register_session(session.clone()).await;

        let retrieved = registry.get_session("sess_test").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().objective, "Test objective");
    }

    #[tokio::test]
    async fn test_session_registry_emit_event() {
        let registry = SessionRegistry::new();
        let session = AgentSession::new("sess_test".to_string(), "Test".to_string());
        registry.register_session(session).await;

        registry
            .emit_event(
                "sess_test",
                AgentEventData::Thought {
                    content: "Thinking...".to_string(),
                },
            )
            .await;

        let session = registry.get_session("sess_test").await.unwrap();
        assert_eq!(session.event_counts.thoughts, 1);
    }

    #[tokio::test]
    async fn test_session_registry_cancel() {
        let registry = SessionRegistry::new();
        let session = AgentSession::new("sess_test".to_string(), "Test".to_string());
        registry.register_session(session).await;

        let result = registry.cancel_session("sess_test").await;
        assert!(result.is_ok());

        let session = registry.get_session("sess_test").await.unwrap();
        assert_eq!(session.status, SessionStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_session_registry_cancel_not_found() {
        let registry = SessionRegistry::new();
        let result = registry.cancel_session("nonexistent").await;
        assert_eq!(result, Err("Session not found"));
    }

    #[tokio::test]
    async fn test_session_registry_list() {
        let registry = SessionRegistry::new();
        registry
            .register_session(AgentSession::new("sess_1".to_string(), "Task 1".to_string()))
            .await;
        registry
            .register_session(AgentSession::new("sess_2".to_string(), "Task 2".to_string()))
            .await;

        let sessions = registry.list_sessions().await;
        assert_eq!(sessions.len(), 2);
    }
}
