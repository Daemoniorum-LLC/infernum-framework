//! Claude Code session and plan file discovery API.
//!
//! Provides endpoints to browse Claude Code sessions stored on the local machine,
//! plan files with inferred status, and direct Claude Code session interaction.
//!
//! ## Discovery Endpoints
//!
//! - `GET /api/claude/sessions` - List all Claude Code sessions
//! - `GET /api/claude/sessions/*path` - Get session transcript
//! - `GET /api/claude/plans` - List plan files with inferred status
//! - `GET /api/claude/plans/:name` - Get plan content
//!
//! ## Direct Session Endpoints
//!
//! - `POST /api/claude/start` - Start a new Claude Code session
//! - `GET /ws/claude/{session_id}` - WebSocket for Claude Code interaction

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// A Claude Code session discovered on the filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeSession {
    /// Session ID (filename without extension)
    pub session_id: String,
    /// Project path this session was run in
    pub project_path: String,
    /// When the session started (first message timestamp)
    pub started_at: Option<String>,
    /// When the session was last updated (last message timestamp)
    pub last_updated: Option<String>,
    /// Number of messages in the session
    pub message_count: usize,
    /// Summary extracted from session (if available)
    pub summary: Option<String>,
}

/// A message from a session transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptMessage {
    /// Message UUID
    pub uuid: String,
    /// Role: "user" or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Parent message UUID (for threading)
    pub parent_uuid: Option<String>,
}

/// Full session transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTranscript {
    /// Session ID
    pub session_id: String,
    /// Project path
    pub project_path: String,
    /// All messages in the session
    pub messages: Vec<TranscriptMessage>,
}

/// A plan file discovered on the filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanFile {
    /// Plan filename
    pub name: String,
    /// Full path to the plan file
    pub path: String,
    /// Inferred status
    pub status: PlanStatus,
    /// Title extracted from content (first H1)
    pub title: Option<String>,
    /// File creation time (if available)
    pub created_at: Option<String>,
    /// Last modification time
    pub modified_at: Option<String>,
}

/// Inferred status of a plan file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanStatus {
    /// Plan has been completed (explicit markers or past tense)
    Completed,
    /// Plan is actively being worked on (recent modifications)
    InProgress,
    /// Plan is old with no recent activity
    Abandoned,
    /// Cannot determine status
    Unknown,
}

// =============================================================================
// Direct Session Types
// =============================================================================

/// Configuration for starting a new Claude Code session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSessionRequest {
    /// Working directory for the session
    pub working_dir: String,
    /// Model tier to use (opus, sonnet, haiku)
    #[serde(default = "default_model")]
    pub model: String,
    /// Allowed tools (empty = all allowed)
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Disallowed tools
    #[serde(default)]
    pub disallowed_tools: Vec<String>,
    /// Maximum turns (0 = unlimited)
    #[serde(default)]
    pub max_turns: u32,
    /// Resume an existing session by ID
    pub resume_session: Option<String>,
}

fn default_model() -> String {
    "sonnet".to_string()
}

/// Response from starting a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSessionResponse {
    /// Unique session identifier
    pub session_id: String,
    /// WebSocket URL for connecting to the session
    pub ws_url: String,
}

/// A message in the session history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Message role (user or assistant)
    pub role: String,
    /// Message content (raw stream-json lines for assistant)
    pub content: String,
    /// Timestamp
    pub timestamp: String,
}

/// An active Claude Code session (persisted between turns).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSession {
    /// Our internal session ID
    pub internal_id: String,
    /// Claude's session ID (extracted from result events)
    pub claude_session_id: Option<String>,
    /// Working directory for the session
    pub working_dir: String,
    /// Model tier (opus, sonnet, haiku)
    pub model: String,
    /// Session configuration
    pub config: StartSessionRequest,
    /// When the session was started
    pub started_at: String,
    /// Last activity timestamp
    pub last_updated: String,
    /// Conversation history
    pub messages: Vec<SessionMessage>,
    /// Whether the session is archived
    #[serde(default)]
    pub is_archived: bool,
}

/// Shared state for direct Claude Code sessions.
pub struct ClaudeSessionState {
    /// Active sessions by ID (multi-turn capable)
    sessions: RwLock<HashMap<String, ActiveSession>>,
    /// Persistence directory path
    persist_dir: PathBuf,
}

impl ClaudeSessionState {
    /// Create a new session state and load persisted sessions.
    pub fn new() -> Self {
        let persist_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".claude/observer-sessions");

        let state = Self {
            sessions: RwLock::new(HashMap::new()),
            persist_dir,
        };

        state
    }

    /// Load all persisted sessions from disk.
    pub async fn load_from_disk(&self) -> std::io::Result<usize> {
        // Create directory if it doesn't exist
        if !self.persist_dir.exists() {
            fs::create_dir_all(&self.persist_dir).await?;
            return Ok(0);
        }

        let mut count = 0;
        let mut entries = fs::read_dir(&self.persist_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                match fs::read_to_string(&path).await {
                    Ok(content) => {
                        match serde_json::from_str::<ActiveSession>(&content) {
                            Ok(session) => {
                                // Skip archived sessions from main list
                                if !session.is_archived {
                                    self.sessions
                                        .write()
                                        .await
                                        .insert(session.internal_id.clone(), session);
                                    count += 1;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse session file {:?}: {}", path, e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read session file {:?}: {}", path, e);
                    }
                }
            }
        }

        info!("Loaded {} sessions from disk", count);
        Ok(count)
    }

    /// Save a session to disk.
    async fn persist_session(&self, session: &ActiveSession) -> std::io::Result<()> {
        // Ensure directory exists
        if !self.persist_dir.exists() {
            fs::create_dir_all(&self.persist_dir).await?;
        }

        let path = self.persist_dir.join(format!("{}.json", session.internal_id));
        let content = serde_json::to_string_pretty(session)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(&path, content).await?;
        debug!("Persisted session {} to disk", session.internal_id);
        Ok(())
    }

    /// Delete a session file from disk.
    async fn delete_session_file(&self, session_id: &str) -> std::io::Result<()> {
        let path = self.persist_dir.join(format!("{}.json", session_id));
        if path.exists() {
            fs::remove_file(&path).await?;
            debug!("Deleted session file for {}", session_id);
        }
        Ok(())
    }

    /// Get a session by ID.
    pub async fn get_session(&self, session_id: &str) -> Option<ActiveSession> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// Insert or update a session (and persist to disk).
    pub async fn upsert_session(&self, session: ActiveSession) {
        let session_clone = session.clone();
        self.sessions
            .write()
            .await
            .insert(session.internal_id.clone(), session);

        // Persist to disk asynchronously
        if let Err(e) = self.persist_session(&session_clone).await {
            error!("Failed to persist session {}: {}", session_clone.internal_id, e);
        }
    }

    /// Update Claude session ID after first turn.
    pub async fn set_claude_session_id(&self, internal_id: &str, claude_id: String) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(internal_id) {
            session.claude_session_id = Some(claude_id);
            session.last_updated = format_system_time(SystemTime::now());

            // Persist to disk
            let session_clone = session.clone();
            drop(sessions); // Release lock before async I/O
            if let Err(e) = self.persist_session(&session_clone).await {
                error!("Failed to persist session {}: {}", internal_id, e);
            }
        }
    }

    /// Add a message to session history.
    pub async fn add_message(&self, internal_id: &str, message: SessionMessage) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(internal_id) {
            session.messages.push(message);
            session.last_updated = format_system_time(SystemTime::now());

            // Persist to disk
            let session_clone = session.clone();
            drop(sessions); // Release lock before async I/O
            if let Err(e) = self.persist_session(&session_clone).await {
                error!("Failed to persist session {}: {}", internal_id, e);
            }
        }
    }

    /// Archive a session.
    pub async fn archive_session(&self, session_id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.is_archived = true;
            session.last_updated = format_system_time(SystemTime::now());

            // Persist to disk and remove from active list
            let session_clone = session.clone();
            sessions.remove(session_id);
            drop(sessions);

            if let Err(e) = self.persist_session(&session_clone).await {
                error!("Failed to persist archived session {}: {}", session_id, e);
                return false;
            }
            info!("Archived session {}", session_id);
            true
        } else {
            false
        }
    }

    /// Remove a session (from memory and disk).
    pub async fn remove_session(&self, session_id: &str) -> bool {
        let removed = self.sessions.write().await.remove(session_id).is_some();
        if removed {
            if let Err(e) = self.delete_session_file(session_id).await {
                error!("Failed to delete session file {}: {}", session_id, e);
            }
            info!("Deleted session {}", session_id);
        }
        removed
    }

    /// List all active sessions (non-archived).
    pub async fn list_sessions(&self) -> Vec<ActiveSession> {
        self.sessions.read().await.values().cloned().collect()
    }

    /// List archived sessions (from disk).
    pub async fn list_archived_sessions(&self) -> Vec<ActiveSession> {
        let mut archived = Vec::new();

        if !self.persist_dir.exists() {
            return archived;
        }

        if let Ok(mut entries) = fs::read_dir(&self.persist_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Ok(content) = fs::read_to_string(&path).await {
                        if let Ok(session) = serde_json::from_str::<ActiveSession>(&content) {
                            if session.is_archived {
                                archived.push(session);
                            }
                        }
                    }
                }
            }
        }

        archived
    }
}

impl Default for ClaudeSessionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create the Claude API router.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/sessions", get(list_sessions))
        .route("/sessions/{*path}", get(get_session))
        .route("/plans", get(list_plans))
        .route("/plans/{name}", get(get_plan))
}

/// Create the Claude API router with state for direct sessions.
pub fn router_with_state(state: Arc<ClaudeSessionState>) -> Router {
    Router::new()
        .route("/sessions", get(list_sessions))
        // Note: specific routes MUST come before /sessions/{*path} to avoid wildcard matching
        .route("/sessions/active", get(list_active_sessions))
        .route("/sessions/archived", get(list_archived_sessions))
        .route("/sessions/{session_id}/transcript", get(get_active_session_transcript))
        .route("/sessions/{session_id}/archive", post(archive_session))
        .route("/sessions/{session_id}", delete(delete_session))
        .route("/sessions/{*path}", get(get_session))
        .route("/plans", get(list_plans))
        .route("/plans/{name}", get(get_plan))
        .route("/start", post(start_session))
        .with_state(state)
}

/// Create the WebSocket router for Claude Code sessions.
pub fn ws_router(state: Arc<ClaudeSessionState>) -> Router {
    Router::new()
        .route("/{session_id}", get(claude_ws_handler))
        .with_state(state)
}

// =============================================================================
// Direct Session Endpoints
// =============================================================================

/// List active Observer sessions.
pub async fn list_active_sessions(
    State(state): State<Arc<ClaudeSessionState>>,
) -> Json<Vec<ActiveSession>> {
    Json(state.list_sessions().await)
}

/// List archived sessions.
pub async fn list_archived_sessions(
    State(state): State<Arc<ClaudeSessionState>>,
) -> Json<Vec<ActiveSession>> {
    Json(state.list_archived_sessions().await)
}

/// Get transcript for an active Observer session.
///
/// Converts the session's message history to the standard transcript format.
pub async fn get_active_session_transcript(
    State(state): State<Arc<ClaudeSessionState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state.get_session(&session_id).await {
        Some(session) => {
            // Convert SessionMessage to TranscriptMessage format
            let messages: Vec<TranscriptMessage> = session
                .messages
                .iter()
                .enumerate()
                .map(|(i, m)| TranscriptMessage {
                    uuid: format!("{}-{}", session_id, i),
                    role: m.role.clone(),
                    content: m.content.clone(),
                    timestamp: m.timestamp.clone(),
                    parent_uuid: if i > 0 {
                        Some(format!("{}-{}", session_id, i - 1))
                    } else {
                        None
                    },
                })
                .collect();

            Json(SessionTranscript {
                session_id: session.internal_id,
                project_path: session.working_dir,
                messages,
            })
            .into_response()
        }
        None => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        )
            .into_response(),
    }
}

/// Archive a session.
pub async fn archive_session(
    State(state): State<Arc<ClaudeSessionState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    if state.archive_session(&session_id).await {
        Json(serde_json::json!({"success": true, "session_id": session_id})).into_response()
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        )
            .into_response()
    }
}

/// Delete a session.
pub async fn delete_session(
    State(state): State<Arc<ClaudeSessionState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    if state.remove_session(&session_id).await {
        Json(serde_json::json!({"success": true, "session_id": session_id})).into_response()
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        )
            .into_response()
    }
}

/// Start a new Claude Code session.
///
/// Creates a session record. The actual Claude process is spawned per-message
/// via the WebSocket connection, enabling multi-turn conversations.
pub async fn start_session(
    State(state): State<Arc<ClaudeSessionState>>,
    Json(request): Json<StartSessionRequest>,
) -> impl IntoResponse {
    // Generate internal session ID
    let session_id = uuid::Uuid::new_v4().to_string();
    let now = format_system_time(SystemTime::now());

    // Create session record (no process yet - spawned per message)
    let session = ActiveSession {
        internal_id: session_id.clone(),
        claude_session_id: request.resume_session.clone(),
        working_dir: request.working_dir.clone(),
        model: request.model.clone(),
        config: request,
        started_at: now.clone(),
        last_updated: now,
        messages: Vec::new(),
        is_archived: false,
    };

    state.upsert_session(session).await;

    info!("Created Claude Code session: {}", session_id);

    let ws_url = format!("/ws/claude/{}", session_id);

    Json(StartSessionResponse { session_id, ws_url }).into_response()
}

/// WebSocket handler for Claude Code session interaction.
pub async fn claude_ws_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    State(state): State<Arc<ClaudeSessionState>>,
) -> impl IntoResponse {
    // Check if session exists
    if state.get_session(&session_id).await.is_none() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            "Session not found",
        )
            .into_response();
    }

    ws.on_upgrade(move |socket| handle_claude_ws(socket, session_id, state))
}

/// Handle WebSocket connection for a Claude Code session.
///
/// Multi-turn capable: spawns a new Claude process for each user message.
async fn handle_claude_ws(
    socket: WebSocket,
    session_id: String,
    state: Arc<ClaudeSessionState>,
) {
    let (ws_sender, mut ws_receiver) = socket.split();
    let ws_sender = Arc::new(Mutex::new(ws_sender));

    info!("WebSocket connected for session {}", session_id);

    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let text_str = text.to_string();
                info!("Received message for session {}: {}", session_id, &text_str[..text_str.len().min(100)]);

                // Parse the message
                let user_message = match serde_json::from_str::<serde_json::Value>(&text_str) {
                    Ok(parsed) => {
                        parsed.get("message")
                            .and_then(|m| m.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or(text_str)
                    }
                    Err(_) => text_str,
                };

                // Get current session state
                let session = match state.get_session(&session_id).await {
                    Some(s) => s,
                    None => {
                        let _ = ws_sender.lock().await
                            .send(Message::Text(
                                serde_json::json!({"type": "error", "error": "Session not found"}).to_string().into(),
                            ))
                            .await;
                        break;
                    }
                };

                // Add user message to history
                state.add_message(&session_id, SessionMessage {
                    role: "user".to_string(),
                    content: user_message.clone(),
                    timestamp: format_system_time(SystemTime::now()),
                }).await;

                // Spawn Claude process for this message
                let claude_output = spawn_claude_for_message(
                    &session,
                    &user_message,
                    ws_sender.clone(),
                    state.clone(),
                    session_id.clone(),
                ).await;

                // Store assistant response in history
                if let Some(output) = claude_output {
                    state.add_message(&session_id, SessionMessage {
                        role: "assistant".to_string(),
                        content: output,
                        timestamp: format_system_time(SystemTime::now()),
                    }).await;
                }

                // Send ready indicator for next message
                let _ = ws_sender.lock().await
                    .send(Message::Text(
                        serde_json::json!({"type": "ready"}).to_string().into(),
                    ))
                    .await;
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket closed for session {}", session_id);
                break;
            }
            Err(e) => {
                warn!("WebSocket error for session {}: {}", session_id, e);
                break;
            }
            _ => {}
        }
    }

    info!("WebSocket session {} ended (session preserved)", session_id);
    // Note: Session is NOT removed - it persists for resume
}

/// Spawn a Claude process for a single message and stream output to WebSocket.
///
/// Returns the raw output for history storage.
async fn spawn_claude_for_message(
    session: &ActiveSession,
    message: &str,
    ws_sender: Arc<Mutex<futures::stream::SplitSink<WebSocket, Message>>>,
    state: Arc<ClaudeSessionState>,
    session_id: String,
) -> Option<String> {
    // Build Claude command
    let mut cmd = Command::new("claude");
    cmd.arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .current_dir(&session.working_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Add model flag
    match session.model.to_lowercase().as_str() {
        "opus" => {
            cmd.arg("--model").arg("opus");
        }
        "haiku" => {
            cmd.arg("--model").arg("haiku");
        }
        _ => {
            // Default to sonnet
        }
    }

    // Add allowed tools
    for tool in &session.config.allowed_tools {
        cmd.arg("--allowedTools").arg(tool);
    }

    // Add disallowed tools
    for tool in &session.config.disallowed_tools {
        cmd.arg("--disallowedTools").arg(tool);
    }

    // Add max turns if specified
    if session.config.max_turns > 0 {
        cmd.arg("--max-turns").arg(session.config.max_turns.to_string());
    }

    // Resume session if we have a Claude session ID (after first turn)
    if let Some(ref claude_id) = session.claude_session_id {
        cmd.arg("--resume").arg(claude_id);
        info!("Resuming Claude session: {}", claude_id);
    }

    // Add the message as positional argument
    cmd.arg(message);

    info!("Spawning Claude for session {}", session_id);

    // Spawn the process
    let mut process = match cmd.spawn() {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to spawn Claude process: {}", e);
            let _ = ws_sender.lock().await
                .send(Message::Text(
                    serde_json::json!({"type": "error", "error": format!("Failed to start Claude: {}", e)}).to_string().into(),
                ))
                .await;
            return None;
        }
    };

    let stdout = match process.stdout.take() {
        Some(s) => s,
        None => {
            error!("No stdout from Claude process");
            return None;
        }
    };

    // Stream stdout to WebSocket and collect for history
    let mut reader = BufReader::new(stdout).lines();
    let mut output_lines = Vec::new();

    while let Ok(Some(line)) = reader.next_line().await {
        debug!("Claude output: {}", &line[..line.len().min(100)]);
        output_lines.push(line.clone());

        // Extract session ID from result event
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
            if parsed.get("type").and_then(|t| t.as_str()) == Some("result") {
                if let Some(sid) = parsed.get("session_id").and_then(|s| s.as_str()) {
                    info!("Extracted Claude session ID: {}", sid);
                    state.set_claude_session_id(&session_id, sid.to_string()).await;
                }
            }
        }

        // Forward to WebSocket
        if ws_sender.lock().await.send(Message::Text(line.into())).await.is_err() {
            warn!("Failed to send to WebSocket");
            break;
        }
    }

    // Wait for process to complete
    let _ = process.wait().await;

    Some(output_lines.join("\n"))
}

/// List all Claude Code sessions.
///
/// Scans `~/.claude/projects/` for session files (*.jsonl).
pub async fn list_sessions() -> Json<Vec<ClaudeSession>> {
    let sessions = discover_sessions().await;
    Json(sessions)
}

/// Get a specific session transcript.
///
/// Path should be the relative path under `~/.claude/projects/`.
pub async fn get_session(Path(path): Path<String>) -> impl IntoResponse {
    let claude_dir = match dirs::home_dir() {
        Some(h) => h.join(".claude/projects"),
        None => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Cannot determine home directory"})),
            )
                .into_response();
        }
    };

    // Sanitize path to prevent directory traversal
    let safe_path = path.replace("..", "");
    let session_path = claude_dir.join(&safe_path);

    if !session_path.exists() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        )
            .into_response();
    }

    match parse_session_file(&session_path).await {
        Ok(transcript) => Json(transcript).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to parse session: {}", e)})),
        )
            .into_response(),
    }
}

/// List all plan files with inferred status.
///
/// Scans `~/.claude/plans/` for markdown files.
pub async fn list_plans() -> Json<Vec<PlanFile>> {
    let plans = discover_plans().await;
    Json(plans)
}

/// Get a specific plan's content.
pub async fn get_plan(Path(name): Path<String>) -> impl IntoResponse {
    let plans_dir = match dirs::home_dir() {
        Some(h) => h.join(".claude/plans"),
        None => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Cannot determine home directory"})),
            )
                .into_response();
        }
    };

    // Sanitize name to prevent directory traversal
    let safe_name = name.replace("..", "").replace('/', "");
    let plan_path = plans_dir.join(&safe_name);

    // Add .md extension if not present
    let plan_path = if plan_path.extension().map(|e| e == "md").unwrap_or(false) {
        plan_path
    } else {
        plan_path.with_extension("md")
    };

    if !plan_path.exists() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Plan not found"})),
        )
            .into_response();
    }

    match fs::read_to_string(&plan_path).await {
        Ok(content) => content.into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Failed to read plan: {}", e)})),
        )
            .into_response(),
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Discover all Claude Code sessions.
async fn discover_sessions() -> Vec<ClaudeSession> {
    let claude_dir = match dirs::home_dir() {
        Some(h) => h.join(".claude/projects"),
        None => {
            warn!("Cannot determine home directory for session discovery");
            return Vec::new();
        }
    };

    if !claude_dir.exists() {
        debug!("Claude projects directory does not exist: {:?}", claude_dir);
        return Vec::new();
    }

    let mut sessions = Vec::new();

    // Walk the projects directory recursively
    if let Ok(entries) = walkdir(&claude_dir).await {
        for entry in entries {
            if entry.extension().map(|e| e == "jsonl").unwrap_or(false) {
                if let Some(session) = parse_session_metadata(&entry, &claude_dir).await {
                    sessions.push(session);
                }
            }
        }
    }

    // Sort by last_updated descending
    sessions.sort_by(|a, b| {
        b.last_updated
            .as_ref()
            .cmp(&a.last_updated.as_ref())
    });

    sessions
}

/// Recursively walk a directory and return all file paths.
async fn walkdir(dir: &PathBuf) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![dir.clone()];

    while let Some(current) = stack.pop() {
        let mut entries = fs::read_dir(&current).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }

    Ok(files)
}

/// Parse session metadata from a JSONL file without reading the entire file.
async fn parse_session_metadata(path: &PathBuf, base_dir: &PathBuf) -> Option<ClaudeSession> {
    let content = fs::read_to_string(path).await.ok()?;
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return None;
    }

    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let project_path = path
        .parent()
        .and_then(|p| p.strip_prefix(base_dir).ok())
        .and_then(|p| p.to_str())
        .unwrap_or("")
        .to_string();

    let mut started_at = None;
    let mut last_updated = None;
    let mut message_count = 0;
    let mut summary = None;

    // Parse first line for start time
    if let Ok(first) = serde_json::from_str::<serde_json::Value>(lines[0]) {
        started_at = first
            .get("timestamp")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
    }

    // Parse last line for end time
    if let Ok(last) = serde_json::from_str::<serde_json::Value>(lines.last().unwrap_or(&"")) {
        last_updated = last
            .get("timestamp")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
    }

    // Count messages (lines with "type": "user" or "type": "assistant")
    for line in &lines {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg.get("type").and_then(|t| t.as_str()).map(|t| t == "user" || t == "assistant").unwrap_or(false) {
                message_count += 1;
            }
            // Look for summary in conversation summary messages
            if msg.get("type").and_then(|t| t.as_str()) == Some("summary") {
                summary = msg
                    .get("summary")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());
            }
        }
    }

    Some(ClaudeSession {
        session_id,
        project_path,
        started_at,
        last_updated,
        message_count,
        summary,
    })
}

/// Parse a full session file into a transcript.
async fn parse_session_file(path: &PathBuf) -> std::io::Result<SessionTranscript> {
    let content = fs::read_to_string(path).await?;
    let lines: Vec<&str> = content.lines().collect();

    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let project_path = path
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or("")
        .to_string();

    let mut messages = Vec::new();

    for line in lines {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");

            if msg_type == "user" || msg_type == "assistant" {
                let uuid = msg
                    .get("uuid")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string();

                let content = extract_message_content(&msg);

                let timestamp = msg
                    .get("timestamp")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                let parent_uuid = msg
                    .get("parentUuid")
                    .and_then(|p| p.as_str())
                    .map(|s| s.to_string());

                messages.push(TranscriptMessage {
                    uuid,
                    role: msg_type.to_string(),
                    content,
                    timestamp,
                    parent_uuid,
                });
            }
        }
    }

    Ok(SessionTranscript {
        session_id,
        project_path,
        messages,
    })
}

/// Extract message content from a JSONL message object.
///
/// For assistant messages with tool calls, formats them as stream-json lines
/// so the frontend can render them with RichRoomMessage.
fn extract_message_content(msg: &serde_json::Value) -> String {
    // Try "message.content" first (assistant messages)
    if let Some(content) = msg
        .get("message")
        .and_then(|m| m.get("content"))
    {
        if let Some(arr) = content.as_array() {
            // Content is an array of content blocks - format as stream-json
            let mut lines = Vec::new();

            for block in arr {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match block_type {
                    "text" => {
                        // Text block - create assistant event
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            let event = serde_json::json!({
                                "type": "assistant",
                                "message": { "content": text }
                            });
                            lines.push(event.to_string());
                        }
                    }
                    "tool_use" => {
                        // Tool use block - create tool_use event
                        let event = serde_json::json!({
                            "type": "tool_use",
                            "name": block.get("name"),
                            "input": block.get("input"),
                            "tool_use_id": block.get("id")
                        });
                        lines.push(event.to_string());
                    }
                    "tool_result" => {
                        // Tool result block - create tool_result event
                        let output = block.get("content")
                            .and_then(|c| c.as_str())
                            .or_else(|| block.get("output").and_then(|o| o.as_str()))
                            .unwrap_or("");
                        let event = serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id"),
                            "output": output,
                            "is_error": block.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false)
                        });
                        lines.push(event.to_string());
                    }
                    _ => {
                        // Unknown block type - include as raw JSON
                        lines.push(block.to_string());
                    }
                }
            }

            return lines.join("\n");
        } else if let Some(s) = content.as_str() {
            return s.to_string();
        }
    }

    // Try "content" directly (user messages - can be string or tool_result)
    if let Some(content) = msg.get("content") {
        if let Some(s) = content.as_str() {
            return s.to_string();
        }
        // Could be a tool_result array in user message
        if let Some(arr) = content.as_array() {
            let mut lines = Vec::new();
            for block in arr {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if block_type == "tool_result" {
                    let output = block.get("content")
                        .and_then(|c| c.as_str())
                        .or_else(|| block.get("output").and_then(|o| o.as_str()))
                        .unwrap_or("");
                    let event = serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id"),
                        "output": output,
                        "is_error": block.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false)
                    });
                    lines.push(event.to_string());
                }
            }
            if !lines.is_empty() {
                return lines.join("\n");
            }
        }
    }

    String::new()
}

/// Discover all plan files.
async fn discover_plans() -> Vec<PlanFile> {
    let plans_dir = match dirs::home_dir() {
        Some(h) => h.join(".claude/plans"),
        None => {
            warn!("Cannot determine home directory for plan discovery");
            return Vec::new();
        }
    };

    if !plans_dir.exists() {
        debug!("Claude plans directory does not exist: {:?}", plans_dir);
        return Vec::new();
    }

    let mut plans = Vec::new();

    if let Ok(mut entries) = fs::read_dir(&plans_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().map(|e| e == "md").unwrap_or(false) {
                if let Some(plan) = parse_plan_file(&path).await {
                    plans.push(plan);
                }
            }
        }
    }

    // Sort by modified_at descending
    plans.sort_by(|a, b| {
        b.modified_at
            .as_ref()
            .cmp(&a.modified_at.as_ref())
    });

    plans
}

/// Parse a plan file and infer its status.
async fn parse_plan_file(path: &PathBuf) -> Option<PlanFile> {
    let content = fs::read_to_string(path).await.ok()?;
    let metadata = fs::metadata(path).await.ok()?;

    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let full_path = path.to_string_lossy().to_string();

    let modified = metadata.modified().ok();
    let created = metadata.created().ok();

    let modified_at = modified.map(|t| format_system_time(t));
    let created_at = created.map(|t| format_system_time(t));

    let title = extract_plan_title(&content);
    let status = infer_plan_status(&content, modified.unwrap_or(SystemTime::UNIX_EPOCH));

    Some(PlanFile {
        name,
        path: full_path,
        status,
        title,
        created_at,
        modified_at,
    })
}

/// Extract the title from a plan file (first H1 heading).
fn extract_plan_title(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# ") {
            return Some(trimmed[2..].trim().to_string());
        }
    }
    None
}

/// Infer the status of a plan based on content and modification time.
fn infer_plan_status(content: &str, modified: SystemTime) -> PlanStatus {
    let now = SystemTime::now();
    let age = now.duration_since(modified).unwrap_or(Duration::ZERO);
    let content_lower = content.to_lowercase();

    // Check explicit markers
    if content.contains("## Status: Completed")
        || content.contains("[COMPLETED]")
        || content.contains("# DONE")
        || content.contains("## DONE")
    {
        return PlanStatus::Completed;
    }

    // Check for past-tense completion indicators
    let completion_phrases = [
        "has been implemented",
        "was implemented",
        "implementation complete",
        "implementation is complete",
        "merged in pr",
        "shipped in",
        "this has been completed",
        "all tasks completed",
        "plan completed",
    ];

    if completion_phrases
        .iter()
        .any(|p| content_lower.contains(p))
    {
        return PlanStatus::Completed;
    }

    // Check for in-progress indicators
    let in_progress_phrases = [
        "## implementation order",
        "## next steps",
        "## todo",
        "- [ ]", // Unchecked checkbox
        "## phase",
    ];

    let has_in_progress = in_progress_phrases
        .iter()
        .any(|p| content_lower.contains(p));

    // Check age - over 30 days without in-progress indicators = likely abandoned
    let thirty_days = Duration::from_secs(30 * 24 * 60 * 60);
    if age > thirty_days && !has_in_progress {
        return PlanStatus::Abandoned;
    }

    // If it has in-progress indicators, it's in progress
    if has_in_progress {
        return PlanStatus::InProgress;
    }

    // Recent modification = in progress
    let seven_days = Duration::from_secs(7 * 24 * 60 * 60);
    if age < seven_days {
        return PlanStatus::InProgress;
    }

    // Default to unknown for older files without clear indicators
    if age > thirty_days {
        PlanStatus::Abandoned
    } else {
        PlanStatus::Unknown
    }
}

/// Format a SystemTime as an ISO 8601 string.
fn format_system_time(time: SystemTime) -> String {
    let duration = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let secs = duration.as_secs();

    // Simple ISO 8601 formatting without external dependencies
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let remaining = remaining % 3600;
    let minutes = remaining / 60;
    let seconds = remaining % 60;

    // Calculate year/month/day from days since epoch (1970-01-01)
    // This is a simplified calculation
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

    (12, 31) // Fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_plan_status_completed_explicit() {
        let content = "# My Plan\n\n## Status: Completed\n\nThis was done.";
        let status = infer_plan_status(content, SystemTime::now());
        assert_eq!(status, PlanStatus::Completed);
    }

    #[test]
    fn test_infer_plan_status_completed_phrase() {
        let content = "# My Plan\n\nThe feature has been implemented.";
        let status = infer_plan_status(content, SystemTime::now());
        assert_eq!(status, PlanStatus::Completed);
    }

    #[test]
    fn test_infer_plan_status_in_progress() {
        let content = "# My Plan\n\n## Implementation Order\n\n1. Do thing\n2. Do other thing";
        let status = infer_plan_status(content, SystemTime::now());
        assert_eq!(status, PlanStatus::InProgress);
    }

    #[test]
    fn test_infer_plan_status_abandoned() {
        let content = "# Old Plan\n\nSome content without clear status.";
        let old_time = SystemTime::now() - Duration::from_secs(60 * 24 * 60 * 60); // 60 days ago
        let status = infer_plan_status(content, old_time);
        assert_eq!(status, PlanStatus::Abandoned);
    }

    #[test]
    fn test_extract_plan_title() {
        let content = "# Session & Plan Management Feature\n\n## Goal\n\nDo stuff.";
        let title = extract_plan_title(content);
        assert_eq!(title, Some("Session & Plan Management Feature".to_string()));
    }

    #[test]
    fn test_extract_plan_title_none() {
        let content = "No heading here\n\nJust content.";
        let title = extract_plan_title(content);
        assert_eq!(title, None);
    }
}
