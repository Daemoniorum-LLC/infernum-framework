//! WebSocket endpoint for Conclave room observation.
//!
//! Provides real-time room event streaming to web clients.
//!
//! # Endpoints
//!
//! - `GET /ws/room/{room_id}` - WebSocket connection for room events
//! - `GET /api/rooms` - List active rooms
//! - `POST /api/rooms` - Create a new room
//! - `POST /api/rooms/{room_id}/message` - Send a message
//! - `POST /api/rooms/{room_id}/spawn` - Spawn an agent
//! - `GET /api/repos` - List local git repositories
//! - `GET /api/repos/branches` - List branches for a repository

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    response::Response,
    Json,
};
use std::process::Command;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use conclave::{
    AgentConfig, ClaudeTier, CreateRoomRequest, PersistenceConfig, PersistenceStore, RoomEvent,
    RoomId, RoomRegistry, UserId,
};

// =============================================================================
// State
// =============================================================================

/// Shared state for room endpoints.
pub struct RoomState {
    /// The room registry.
    pub registry: Arc<RoomRegistry>,
    /// Persistence store for durable storage.
    pub persistence: Option<Arc<PersistenceStore>>,
}

impl RoomState {
    /// Creates a new room state with a default registry (no persistence).
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RoomRegistry::with_defaults()),
            persistence: None,
        }
    }

    /// Creates room state with persistence enabled.
    pub async fn with_persistence() -> Result<Self, conclave::ConclaveError> {
        let store = PersistenceStore::with_defaults();
        let registry = RoomRegistry::with_persistence(store).await?;
        let persistence = PersistenceStore::with_defaults();

        Ok(Self {
            registry: Arc::new(registry),
            persistence: Some(Arc::new(persistence)),
        })
    }

    /// Creates room state with a custom persistence config.
    pub async fn with_persistence_config(
        config: PersistenceConfig,
    ) -> Result<Self, conclave::ConclaveError> {
        let store = PersistenceStore::new(config.clone());
        let registry = RoomRegistry::with_persistence(store).await?;
        let persistence = PersistenceStore::new(config);

        Ok(Self {
            registry: Arc::new(registry),
            persistence: Some(Arc::new(persistence)),
        })
    }

    /// Creates room state with an existing registry.
    pub fn with_registry(registry: Arc<RoomRegistry>) -> Self {
        Self {
            registry,
            persistence: None,
        }
    }

    /// Persists current state to disk (if persistence is enabled).
    pub async fn persist(&self) -> Result<(), conclave::ConclaveError> {
        if let Some(ref store) = self.persistence {
            self.registry.persist(store).await?;
        }
        Ok(())
    }
}

impl Default for RoomState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// API Types
// =============================================================================

/// Request to create a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRoomApiRequest {
    /// Room name.
    pub name: String,
    /// Labels for filtering/organization.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Working directory for agents.
    pub working_dir: String,
}

/// Response for room creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRoomResponse {
    /// Room ID.
    pub room_id: String,
    /// Room name.
    pub name: String,
}

/// Request to send a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendMessageRequest {
    /// Message content.
    pub content: String,
}

/// Response for message sending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendMessageResponse {
    /// Message ID.
    pub message_id: String,
}

/// A message in the room history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMessageDto {
    /// Message ID.
    pub message_id: String,
    /// Sender participant ID.
    pub sender_id: String,
    /// Sender display name (if available).
    pub sender_name: Option<String>,
    /// Whether sender is an agent.
    pub is_agent: bool,
    /// Message content.
    pub content: String,
    /// Timestamp (RFC3339 format).
    pub timestamp: String,
}

/// A participant in the room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomParticipantDto {
    /// Participant ID.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Whether this is an agent.
    pub is_agent: bool,
}

/// Response for getting room messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMessagesResponse {
    /// Room ID.
    pub room_id: String,
    /// Messages in chronological order.
    pub messages: Vec<RoomMessageDto>,
    /// Current participants in the room.
    pub participants: Vec<RoomParticipantDto>,
}

/// Request to spawn an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnAgentRequest {
    /// Agent type (claude-opus, claude-sonnet, claude-haiku, or infernum:<model>).
    pub agent_type: String,
    /// Display name for the agent.
    pub display_name: Option<String>,
    /// Grimoire persona code to apply.
    pub persona: Option<String>,
}

/// Response for agent spawning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnAgentResponse {
    /// Participant ID.
    pub participant_id: String,
    /// Display name.
    pub display_name: String,
}

/// Response for room restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreRoomResponse {
    /// Number of agents that were restored.
    pub restored_count: u32,
    /// Total number of disconnected agents found.
    pub total_disconnected: usize,
    /// Any errors encountered during restoration.
    pub errors: Vec<String>,
}

/// Information about a disconnected agent for the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisconnectedAgentInfo {
    /// Agent participant ID.
    pub participant_id: String,
    /// Agent display name.
    pub display_name: String,
    /// Backend type.
    pub backend_type: String,
}

/// Room summary for listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSummary {
    /// Room ID.
    pub id: String,
    /// Room name.
    pub name: String,
    /// Labels for filtering/organization.
    pub labels: Vec<String>,
    /// Number of participants.
    pub participant_count: usize,
    /// Whether the room is archived.
    pub archived: bool,
}

/// Available agent type for spawning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTypeInfo {
    /// Agent type ID (e.g., "claude-opus", "infernum:model").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Category (claude, infernum).
    pub category: String,
    /// Whether this agent is available (model loaded, API key set, etc.).
    pub available: bool,
    /// Description of the agent.
    pub description: String,
}

/// Available persona for agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaInfo {
    /// Persona ID/code.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Brief description.
    pub description: Option<String>,
}

/// WebSocket event sent to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsRoomEvent {
    /// Connected to room.
    Connected {
        room_id: String,
        room_name: String,
    },
    /// Room was created.
    RoomCreated {
        room_id: String,
        name: String,
    },
    /// Participant joined.
    ParticipantJoined {
        room_id: String,
        participant_id: String,
        display_name: String,
        is_agent: bool,
    },
    /// Participant left.
    ParticipantLeft {
        room_id: String,
        participant_id: String,
    },
    /// Message sent.
    MessageSent {
        room_id: String,
        message_id: String,
        sender_id: String,
        content: String,
        timestamp: String,
    },
    /// Attention changed.
    AttentionChanged {
        room_id: String,
        participant_id: String,
        new_state: String,
    },
    /// Room archived.
    RoomArchived {
        room_id: String,
    },
    /// Error.
    Error {
        message: String,
    },
}

/// API error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
}

/// Repository information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoInfo {
    /// Path to the repository.
    pub path: String,
    /// Repository name (directory name).
    pub name: String,
    /// Remote URL (if available).
    pub remote: Option<String>,
    /// Whether this is a local repo (vs. cloned from remote).
    pub is_local: bool,
}

/// Query parameters for listing repositories.
#[derive(Debug, Clone, Deserialize)]
pub struct ListReposQuery {
    /// Base directory to search for repos (defaults to home directory).
    pub base_path: Option<String>,
    /// Maximum depth to search (defaults to 3).
    pub max_depth: Option<usize>,
}

/// Branch information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    /// Branch name.
    pub name: String,
    /// Whether this is the current branch.
    pub is_current: bool,
    /// Whether this is a remote branch.
    pub is_remote: bool,
}

/// Query parameters for listing branches.
#[derive(Debug, Clone, Deserialize)]
pub struct ListBranchesQuery {
    /// Path to the repository.
    pub path: String,
}

/// Directory entry for filesystem browsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    /// Entry name.
    pub name: String,
    /// Full path.
    pub path: String,
    /// Whether this is a directory.
    pub is_dir: bool,
    /// Whether this contains a git repository.
    pub is_git_repo: bool,
}

/// Query parameters for browsing directories.
#[derive(Debug, Clone, Deserialize)]
pub struct BrowseDirQuery {
    /// Directory path to browse (defaults to home directory).
    pub path: Option<String>,
}

// =============================================================================
// WebSocket Handler
// =============================================================================

/// Handles WebSocket upgrade for room observation.
pub async fn room_ws_handler(
    ws: WebSocketUpgrade,
    Path(room_id): Path<String>,
    State(state): State<Arc<RoomState>>,
) -> Response {
    info!("WebSocket connection request for room: {}", room_id);

    ws.on_upgrade(move |socket| handle_room_socket(socket, room_id, state))
}

/// Handles the WebSocket connection for a room.
async fn handle_room_socket(socket: WebSocket, room_id: String, state: Arc<RoomState>) {
    let (mut sender, mut receiver) = socket.split();

    // Resolve room ID
    let rid = match resolve_room_id(&room_id, &state.registry).await {
        Ok(rid) => rid,
        Err(e) => {
            let error = WsRoomEvent::Error { message: e };
            let json = serde_json::to_string(&error).unwrap_or_default();
            let _ = sender.send(Message::Text(json.into())).await;
            return;
        }
    };

    // Get room info
    let room = match state.registry.get_room(rid).await {
        Some(r) => r,
        None => {
            let error = WsRoomEvent::Error {
                message: "Room not found".to_string(),
            };
            let json = serde_json::to_string(&error).unwrap_or_default();
            let _ = sender.send(Message::Text(json.into())).await;
            return;
        }
    };

    // Send connected event
    let connected = WsRoomEvent::Connected {
        room_id: rid.0.to_string(),
        room_name: room.name.clone(),
    };
    let json = serde_json::to_string(&connected).unwrap_or_default();
    if sender.send(Message::Text(json.into())).await.is_err() {
        return;
    }

    info!("WebSocket connected for room: {} ({})", room.name, rid.0);

    // Subscribe to events
    let mut events_rx = state.registry.subscribe();

    // Event loop
    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) => {
                        debug!("WebSocket client closed connection");
                        break;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        if sender.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Text(text))) => {
                        // Could handle client commands here
                        debug!("Received text: {}", text);
                    }
                    Some(Err(e)) => {
                        warn!("WebSocket error: {}", e);
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }

            // Forward room events
            event = events_rx.recv() => {
                match event {
                    Ok(room_event) => {
                        debug!("Received room event: {:?}", room_event);
                        if let Some(ws_event) = convert_room_event(&room_event, rid) {
                            let json = serde_json::to_string(&ws_event).unwrap_or_default();
                            debug!("Forwarding to WebSocket: {}", json);
                            if sender.send(Message::Text(json.into())).await.is_err() {
                                warn!("Failed to send event to WebSocket");
                                break;
                            }
                        } else {
                            debug!("Event filtered (different room or unhandled type)");
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Lagged {} events", n);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("Event channel closed");
                        break;
                    }
                }
            }
        }
    }

    info!("WebSocket disconnected for room: {}", rid.0);
}

/// Converts a RoomEvent to a WebSocket event (if it matches the subscribed room).
fn convert_room_event(event: &RoomEvent, subscribed_room: RoomId) -> Option<WsRoomEvent> {
    match event {
        RoomEvent::RoomCreated { room_id, name, .. } if *room_id == subscribed_room => {
            Some(WsRoomEvent::RoomCreated {
                room_id: room_id.0.to_string(),
                name: name.clone(),
            })
        }
        RoomEvent::ParticipantJoined {
            room_id,
            participant,
        } if *room_id == subscribed_room => Some(WsRoomEvent::ParticipantJoined {
            room_id: room_id.0.to_string(),
            participant_id: participant.id.0.to_string(),
            display_name: participant.display_name.clone(),
            is_agent: participant.kind.is_agent(),
        }),
        RoomEvent::ParticipantLeft {
            room_id,
            participant_id,
            ..
        } if *room_id == subscribed_room => Some(WsRoomEvent::ParticipantLeft {
            room_id: room_id.0.to_string(),
            participant_id: participant_id.0.to_string(),
        }),
        RoomEvent::MessageSent { room_id, message } if *room_id == subscribed_room => {
            let content = match &message.content {
                conclave::MessageContent::Text { content } => content.clone(),
                conclave::MessageContent::ToolCall { tool, input, .. } => {
                    format!("[Tool: {}] {}", tool, input)
                }
                conclave::MessageContent::ToolResult { output, .. } => {
                    let output_str = output.to_string();
                    let truncated = if output_str.len() > 200 {
                        format!("{}...", &output_str[..200])
                    } else {
                        output_str
                    };
                    format!("[Result] {}", truncated)
                }
                conclave::MessageContent::System { event } => format!("[System] {:?}", event),
            };

            Some(WsRoomEvent::MessageSent {
                room_id: room_id.0.to_string(),
                message_id: message.id.0.to_string(),
                sender_id: message.sender.0.to_string(),
                content,
                timestamp: message.timestamp.to_rfc3339(),
            })
        }
        RoomEvent::AttentionChanged {
            room_id,
            participant_id,
            new_state,
        } if *room_id == subscribed_room => Some(WsRoomEvent::AttentionChanged {
            room_id: room_id.0.to_string(),
            participant_id: participant_id.0.to_string(),
            new_state: format!("{:?}", new_state),
        }),
        RoomEvent::RoomArchived { room_id, .. } if *room_id == subscribed_room => {
            Some(WsRoomEvent::RoomArchived {
                room_id: room_id.0.to_string(),
            })
        }
        _ => None,
    }
}

// =============================================================================
// REST API Handlers
// =============================================================================

/// Lists all active rooms.
pub async fn list_rooms(State(state): State<Arc<RoomState>>) -> Json<Vec<RoomSummary>> {
    let rooms = state.registry.list_active_rooms().await;
    let summaries: Vec<RoomSummary> = rooms
        .into_iter()
        .map(|r| RoomSummary {
            id: r.id.0.to_string(),
            name: r.name,
            labels: r.labels,
            participant_count: r.participants.len(),
            archived: r.archived,
        })
        .collect();

    Json(summaries)
}

/// Creates a new room.
pub async fn create_room(
    State(state): State<Arc<RoomState>>,
    Json(req): Json<CreateRoomApiRequest>,
) -> Result<Json<CreateRoomResponse>, Json<ApiError>> {
    let working_dir = PathBuf::from(&req.working_dir);

    if !working_dir.exists() {
        return Err(Json(ApiError {
            error: format!("Working directory does not exist: {}", req.working_dir),
        }));
    }

    let user_id = UserId("web-user".to_string());
    let create_req = CreateRoomRequest::new(&req.name, working_dir, user_id)
        .with_labels(req.labels);

    match state.registry.create_room(create_req).await {
        Ok(room_id) => {
            // Persist after creating room
            if let Err(e) = state.persist().await {
                warn!("Failed to persist after room creation: {}", e);
            }
            Ok(Json(CreateRoomResponse {
                room_id: room_id.0.to_string(),
                name: req.name,
            }))
        }
        Err(e) => Err(Json(ApiError {
            error: e.to_string(),
        })),
    }
}

/// Sends a message to a room.
pub async fn send_message(
    State(state): State<Arc<RoomState>>,
    Path(room_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> Result<Json<SendMessageResponse>, Json<ApiError>> {
    let rid = resolve_room_id(&room_id, &state.registry)
        .await
        .map_err(|e| Json(ApiError { error: e }))?;

    // Get a human participant
    let room = state.registry.get_room(rid).await.ok_or_else(|| {
        Json(ApiError {
            error: "Room not found".to_string(),
        })
    })?;

    let sender = room
        .participants
        .iter()
        .find(|p| p.kind.is_human())
        .ok_or_else(|| {
            Json(ApiError {
                error: "No human participant in room".to_string(),
            })
        })?;

    // Use send_and_route to automatically route messages to agents
    debug!(
        "Sending message to room {} from sender {}",
        rid.0, sender.id.0
    );
    match state
        .registry
        .send_and_route(rid, sender.id, req.content.clone())
        .await
    {
        Ok(message_id) => {
            info!(
                "Message {} sent to room {} (content: {})",
                message_id.0, rid.0, req.content
            );
            // Persist after sending message
            if let Err(e) = state.persist().await {
                warn!("Failed to persist after message send: {}", e);
            }
            Ok(Json(SendMessageResponse {
                message_id: message_id.0.to_string(),
            }))
        }
        Err(e) => {
            warn!("Failed to send message to room {}: {}", rid.0, e);
            Err(Json(ApiError {
                error: e.to_string(),
            }))
        }
    }
}

/// Gets the message history for a room.
pub async fn get_messages(
    State(state): State<Arc<RoomState>>,
    Path(room_id): Path<String>,
) -> Result<Json<GetMessagesResponse>, Json<ApiError>> {
    let rid = resolve_room_id(&room_id, &state.registry)
        .await
        .map_err(|e| Json(ApiError { error: e }))?;

    // Get room to look up participant names
    let room = state.registry.get_room(rid).await.ok_or_else(|| {
        Json(ApiError {
            error: "Room not found".to_string(),
        })
    })?;

    // Build a map of participant ID to (name, is_agent)
    let participant_info: std::collections::HashMap<String, (String, bool)> = room
        .participants
        .iter()
        .chain(room.alumni.iter())
        .map(|p| (p.id.0.to_string(), (p.display_name.clone(), p.kind.is_agent())))
        .collect();

    // Get messages from main channel
    let messages = state
        .registry
        .get_main_channel_messages(rid)
        .await
        .map_err(|e| Json(ApiError { error: e.to_string() }))?;

    // Convert to DTOs
    let message_dtos: Vec<RoomMessageDto> = messages
        .into_iter()
        .map(|msg| {
            let sender_id = msg.sender.0.to_string();
            let (sender_name, is_agent) = participant_info
                .get(&sender_id)
                .map(|(name, is_agent)| (Some(name.clone()), *is_agent))
                .unwrap_or_else(|| (None, false));

            let content = match &msg.content {
                conclave::MessageContent::Text { content } => content.clone(),
                conclave::MessageContent::ToolCall { tool, input, .. } => {
                    format!("[Tool: {}] {}", tool, input)
                }
                conclave::MessageContent::ToolResult { output, .. } => {
                    let output_str = output.to_string();
                    if output_str.len() > 500 {
                        format!("[Result] {}...", &output_str[..500])
                    } else {
                        format!("[Result] {}", output_str)
                    }
                }
                conclave::MessageContent::System { event } => format!("[System] {:?}", event),
            };

            RoomMessageDto {
                message_id: msg.id.0.to_string(),
                sender_id,
                sender_name,
                is_agent,
                content,
                timestamp: msg.timestamp.to_rfc3339(),
            }
        })
        .collect();

    // Convert participants to DTOs
    let participant_dtos: Vec<RoomParticipantDto> = room
        .participants
        .iter()
        .map(|p| RoomParticipantDto {
            id: p.id.0.to_string(),
            display_name: p.display_name.clone(),
            is_agent: p.kind.is_agent(),
        })
        .collect();

    debug!(
        "Returning {} messages and {} participants for room {}",
        message_dtos.len(),
        participant_dtos.len(),
        rid.0
    );

    Ok(Json(GetMessagesResponse {
        room_id: rid.0.to_string(),
        messages: message_dtos,
        participants: participant_dtos,
    }))
}

/// Spawns an agent in a room.
pub async fn spawn_agent(
    State(state): State<Arc<RoomState>>,
    Path(room_id): Path<String>,
    Json(req): Json<SpawnAgentRequest>,
) -> Result<Json<SpawnAgentResponse>, Json<ApiError>> {
    let rid = resolve_room_id(&room_id, &state.registry)
        .await
        .map_err(|e| Json(ApiError { error: e }))?;

    // Parse agent config
    let mut config = parse_agent_type(&req.agent_type).map_err(|e| Json(ApiError { error: e }))?;

    if let Some(name) = &req.display_name {
        config.display_name = Some(name.clone());
    }

    if let Some(persona) = &req.persona {
        config.persona = Some(persona.clone());
    }

    // Get a human participant as spawner
    let room = state.registry.get_room(rid).await.ok_or_else(|| {
        Json(ApiError {
            error: "Room not found".to_string(),
        })
    })?;

    let spawner = room
        .participants
        .iter()
        .find(|p| p.kind.is_human())
        .ok_or_else(|| {
            Json(ApiError {
                error: "No human participant in room".to_string(),
            })
        })?;

    let display_name = config
        .display_name
        .clone()
        .unwrap_or_else(|| req.agent_type.clone());

    // Use spawn_agent_with_events to start the event processor for message routing
    match state
        .registry
        .spawn_agent_with_events(rid, config, spawner.id)
        .await
    {
        Ok((participant_id, _handle)) => {
            // Persist after spawning agent
            if let Err(e) = state.persist().await {
                warn!("Failed to persist after agent spawn: {}", e);
            }
            // The event processor handle runs in the background
            // It will be cleaned up when the agent terminates
            Ok(Json(SpawnAgentResponse {
                participant_id: participant_id.0.to_string(),
                display_name,
            }))
        }
        Err(e) => Err(Json(ApiError {
            error: e.to_string(),
        })),
    }
}

/// Gets list of disconnected agents in a room.
pub async fn get_disconnected_agents(
    State(state): State<Arc<RoomState>>,
    Path(room_id): Path<String>,
) -> Result<Json<Vec<DisconnectedAgentInfo>>, Json<ApiError>> {
    let rid = resolve_room_id(&room_id, &state.registry)
        .await
        .map_err(|e| Json(ApiError { error: e }))?;

    let disconnected = state.registry.get_disconnected_agents(rid).await.map_err(|e| {
        Json(ApiError {
            error: e.to_string(),
        })
    })?;

    let info: Vec<DisconnectedAgentInfo> = disconnected
        .into_iter()
        .map(|agent| DisconnectedAgentInfo {
            participant_id: agent.agent_id.0.to_string(),
            display_name: agent.display_name,
            backend_type: agent.backend_type,
        })
        .collect();

    Ok(Json(info))
}

/// Restores all disconnected agents in a room.
///
/// This re-spawns agent backend processes with conversation context injected.
pub async fn restore_room(
    State(state): State<Arc<RoomState>>,
    Path(room_id): Path<String>,
) -> Result<Json<RestoreRoomResponse>, Json<ApiError>> {
    let rid = resolve_room_id(&room_id, &state.registry)
        .await
        .map_err(|e| Json(ApiError { error: e }))?;

    // Get disconnected count before restore
    let disconnected = state.registry.get_disconnected_agents(rid).await.map_err(|e| {
        Json(ApiError {
            error: e.to_string(),
        })
    })?;
    let total_disconnected = disconnected.len();

    if total_disconnected == 0 {
        return Ok(Json(RestoreRoomResponse {
            restored_count: 0,
            total_disconnected: 0,
            errors: vec![],
        }));
    }

    // Get a human participant to act as recoverer
    let room = state.registry.get_room(rid).await.ok_or_else(|| {
        Json(ApiError {
            error: "Room not found".to_string(),
        })
    })?;

    let recoverer = room
        .participants
        .iter()
        .find(|p| p.kind.is_human())
        .ok_or_else(|| {
            Json(ApiError {
                error: "No human participant in room to perform recovery".to_string(),
            })
        })?;

    // Restore all agents
    let restored_count = state
        .registry
        .recover_all_agents(rid, recoverer.id)
        .await
        .map_err(|e| {
            Json(ApiError {
                error: e.to_string(),
            })
        })?;

    // Check for remaining disconnected (indicates errors)
    let remaining = state
        .registry
        .get_disconnected_agents(rid)
        .await
        .unwrap_or_default();
    let errors: Vec<String> = remaining
        .iter()
        .map(|agent| format!("Failed to restore agent: {}", agent.display_name))
        .collect();

    // Persist state
    if let Err(e) = state.persist().await {
        warn!("Failed to persist after room restore: {}", e);
    }

    info!(
        "Restored {}/{} agents in room {}",
        restored_count, total_disconnected, rid
    );

    Ok(Json(RestoreRoomResponse {
        restored_count,
        total_disconnected,
        errors,
    }))
}

// =============================================================================
// Agent Types & Personas API Handlers
// =============================================================================

/// Lists available agent types for spawning.
pub async fn list_agent_types() -> Json<Vec<AgentTypeInfo>> {
    let types = vec![
        AgentTypeInfo {
            id: "claude-opus".to_string(),
            name: "Claude Opus".to_string(),
            category: "claude".to_string(),
            available: true, // Assume available if Claude Code is installed
            description: "Most capable Claude model. Best for complex reasoning and coding."
                .to_string(),
        },
        AgentTypeInfo {
            id: "claude-sonnet".to_string(),
            name: "Claude Sonnet".to_string(),
            category: "claude".to_string(),
            available: true,
            description: "Balanced performance and speed. Good for most tasks.".to_string(),
        },
        AgentTypeInfo {
            id: "claude-haiku".to_string(),
            name: "Claude Haiku".to_string(),
            category: "claude".to_string(),
            available: true,
            description: "Fastest Claude model. Good for simple tasks and quick iterations."
                .to_string(),
        },
        AgentTypeInfo {
            id: "infernum".to_string(),
            name: "Infernum (Local LLM)".to_string(),
            category: "infernum".to_string(),
            available: true, // Model availability checked at spawn time
            description: "Local LLM inference using the loaded model.".to_string(),
        },
    ];

    Json(types)
}

/// Lists available personas from the grimoire.
pub async fn list_personas() -> Json<Vec<PersonaInfo>> {
    use grimoire_loader::GrimoireLoader;

    let loader = GrimoireLoader::new();
    let personas = match loader.list().await {
        Ok(ids) => {
            let mut results = Vec::new();
            for id in ids {
                if let Ok(persona) = loader.load(&id).await {
                    results.push(PersonaInfo {
                        id: id.clone(),
                        name: persona.name,
                        description: Some(persona.system_prompt.chars().take(100).collect::<String>() + "..."),
                    });
                } else {
                    results.push(PersonaInfo {
                        id: id.clone(),
                        name: id,
                        description: None,
                    });
                }
            }
            results
        }
        Err(_) => vec![],
    };

    Json(personas)
}

// =============================================================================
// Repository API Handlers
// =============================================================================

/// Lists git repositories in a base directory.
pub async fn list_repos(
    Query(query): Query<ListReposQuery>,
) -> Result<Json<Vec<RepoInfo>>, Json<ApiError>> {
    let base_path = query.base_path.unwrap_or_else(|| {
        std::env::var("HOME").unwrap_or_else(|_| "/home".to_string())
    });
    let max_depth = query.max_depth.unwrap_or(3);

    let base = PathBuf::from(&base_path);
    if !base.exists() {
        return Err(Json(ApiError {
            error: format!("Base path does not exist: {}", base_path),
        }));
    }

    let repos = find_git_repos(&base, max_depth);
    Ok(Json(repos))
}

/// Lists branches for a repository.
pub async fn list_branches(
    Query(query): Query<ListBranchesQuery>,
) -> Result<Json<Vec<BranchInfo>>, Json<ApiError>> {
    let repo_path = PathBuf::from(&query.path);

    if !repo_path.exists() {
        return Err(Json(ApiError {
            error: format!("Repository path does not exist: {}", query.path),
        }));
    }

    let git_dir = repo_path.join(".git");
    if !git_dir.exists() {
        return Err(Json(ApiError {
            error: format!("Not a git repository: {}", query.path),
        }));
    }

    let branches = get_repo_branches(&repo_path);
    Ok(Json(branches))
}

/// Browse directory response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowseDirResponse {
    /// Current directory path.
    pub path: String,
    /// Parent directory path (if not root).
    pub parent: Option<String>,
    /// Directory entries.
    pub entries: Vec<DirEntry>,
}

/// Browses a directory and returns its contents.
pub async fn browse_dir(
    Query(query): Query<BrowseDirQuery>,
) -> Result<Json<BrowseDirResponse>, Json<ApiError>> {
    let path = query.path.unwrap_or_else(|| {
        std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
    });

    let dir_path = PathBuf::from(&path);
    if !dir_path.exists() {
        return Err(Json(ApiError {
            error: format!("Path does not exist: {}", path),
        }));
    }

    if !dir_path.is_dir() {
        return Err(Json(ApiError {
            error: format!("Path is not a directory: {}", path),
        }));
    }

    // Get canonical path
    let canonical = dir_path.canonicalize().unwrap_or(dir_path.clone());
    let canonical_str = canonical.to_string_lossy().to_string();

    // Get parent
    let parent = canonical.parent().map(|p| p.to_string_lossy().to_string());

    // Read directory entries
    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(&canonical) {
        for entry in read_dir.flatten() {
            let entry_path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            // Skip hidden files/directories
            if name.starts_with('.') {
                continue;
            }

            let is_dir = entry_path.is_dir();
            let is_git_repo = is_dir && entry_path.join(".git").exists();

            entries.push(DirEntry {
                name,
                path: entry_path.to_string_lossy().to_string(),
                is_dir,
                is_git_repo,
            });
        }
    }

    // Sort: directories first, then by name
    entries.sort_by(|a, b| {
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });

    Ok(Json(BrowseDirResponse {
        path: canonical_str,
        parent,
        entries,
    }))
}

// =============================================================================
// Helpers
// =============================================================================

/// Finds git repositories in a directory tree.
fn find_git_repos(base: &PathBuf, max_depth: usize) -> Vec<RepoInfo> {
    let mut repos = Vec::new();
    find_git_repos_recursive(base, 0, max_depth, &mut repos);
    repos
}

fn find_git_repos_recursive(
    dir: &PathBuf,
    current_depth: usize,
    max_depth: usize,
    repos: &mut Vec<RepoInfo>,
) {
    if current_depth > max_depth {
        return;
    }

    // Check if this directory is a git repo
    let git_dir = dir.join(".git");
    if git_dir.exists() {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let remote = get_repo_remote(dir);
        let is_local = remote.is_none();

        repos.push(RepoInfo {
            path: dir.to_string_lossy().to_string(),
            name,
            remote,
            is_local,
        });

        // Don't recurse into git repos
        return;
    }

    // Recurse into subdirectories
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories and common non-repo directories
                let name = path.file_name().map(|n| n.to_string_lossy().to_string());
                if let Some(n) = name {
                    if n.starts_with('.') || n == "node_modules" || n == "target" || n == "vendor" {
                        continue;
                    }
                }
                find_git_repos_recursive(&path, current_depth + 1, max_depth, repos);
            }
        }
    }
}

/// Gets the remote URL for a git repository.
fn get_repo_remote(repo_path: &PathBuf) -> Option<String> {
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(repo_path)
        .output()
        .ok()?;

    if output.status.success() {
        let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !url.is_empty() {
            return Some(url);
        }
    }
    None
}

/// Gets branches for a git repository.
fn get_repo_branches(repo_path: &PathBuf) -> Vec<BranchInfo> {
    let mut branches = Vec::new();

    // Get current branch
    let current_branch = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(repo_path)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    // Get local branches
    if let Ok(output) = Command::new("git")
        .args(["branch", "--format=%(refname:short)"])
        .current_dir(repo_path)
        .output()
    {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let name = line.trim().to_string();
                if !name.is_empty() {
                    branches.push(BranchInfo {
                        name: name.clone(),
                        is_current: name == current_branch,
                        is_remote: false,
                    });
                }
            }
        }
    }

    // Get remote branches
    if let Ok(output) = Command::new("git")
        .args(["branch", "-r", "--format=%(refname:short)"])
        .current_dir(repo_path)
        .output()
    {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let name = line.trim().to_string();
                // Skip HEAD references and empty lines
                if !name.is_empty() && !name.contains("HEAD") {
                    branches.push(BranchInfo {
                        name,
                        is_current: false,
                        is_remote: true,
                    });
                }
            }
        }
    }

    branches
}

/// Resolves a room ID prefix to a full RoomId.
async fn resolve_room_id(prefix: &str, registry: &RoomRegistry) -> Result<RoomId, String> {
    let rooms = registry.list_active_rooms().await;
    let matches: Vec<_> = rooms
        .iter()
        .filter(|r| r.id.0.to_string().starts_with(prefix))
        .collect();

    match matches.len() {
        0 => Err(format!("No room matches prefix: {}", prefix)),
        1 => Ok(matches[0].id),
        _ => Err(format!(
            "Ambiguous: {} rooms match prefix '{}'. Be more specific.",
            matches.len(),
            prefix
        )),
    }
}

/// Parses agent type string into AgentConfig.
fn parse_agent_type(agent_type: &str) -> Result<AgentConfig, String> {
    match agent_type.to_lowercase().as_str() {
        "claude-opus" => Ok(AgentConfig::claude_code(ClaudeTier::Opus)),
        "claude-sonnet" => Ok(AgentConfig::claude_code(ClaudeTier::Sonnet)),
        "claude-haiku" => Ok(AgentConfig::claude_code(ClaudeTier::Haiku)),
        s if s.starts_with("infernum:") => {
            let model = s.strip_prefix("infernum:").unwrap();
            Ok(AgentConfig::infernum(model))
        }
        _ => Err(format!(
            "Unknown agent type: {}. Use claude-opus, claude-sonnet, claude-haiku, or infernum:<model>",
            agent_type
        )),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_agent_type_claude() {
        let config = parse_agent_type("claude-opus").unwrap();
        assert!(matches!(
            config.backend,
            conclave::AgentBackend::ClaudeCode { .. }
        ));

        let config = parse_agent_type("claude-sonnet").unwrap();
        assert!(matches!(
            config.backend,
            conclave::AgentBackend::ClaudeCode { .. }
        ));

        let config = parse_agent_type("claude-haiku").unwrap();
        assert!(matches!(
            config.backend,
            conclave::AgentBackend::ClaudeCode { .. }
        ));
    }

    #[test]
    fn test_parse_agent_type_infernum() {
        let config = parse_agent_type("infernum:llama-7b").unwrap();
        assert!(matches!(
            config.backend,
            conclave::AgentBackend::Infernum { .. }
        ));
    }

    #[test]
    fn test_parse_agent_type_invalid() {
        let result = parse_agent_type("unknown-agent");
        assert!(result.is_err());
    }
}
