//! Room daemon server.
//!
//! Hosts the RoomRegistry and accepts commands over Unix socket.
//!
//! Implements CONCLAVE-CLI-SPEC.md §4.1 Daemon Lifecycle.

use std::path::PathBuf;
use std::sync::Arc;

use color_eyre::eyre::{eyre, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use conclave::{
    AgentConfig, ClaudeTier, CreateRoomRequest, Room, RoomEvent, RoomId, RoomRegistry, UserId,
};

// =============================================================================
// Daemon Protocol
// =============================================================================

/// Request from CLI to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonRequest {
    /// Create a new room.
    CreateRoom { name: String, working_dir: PathBuf },
    /// List all active rooms.
    ListRooms,
    /// Get room details.
    GetRoom { room_id: String },
    /// Spawn an agent in a room.
    SpawnAgent {
        room_id: String,
        agent_type: String,
        display_name: Option<String>,
    },
    /// Send a message to a room.
    SendMessage {
        room_id: String,
        content: String,
    },
    /// Subscribe to room events.
    Subscribe { room_id: String },
    /// Archive a room.
    ArchiveRoom { room_id: String },
    /// Shutdown the daemon.
    Shutdown,
    /// Ping for health check.
    Ping,
}

/// Response from daemon to CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonResponse {
    /// Room created.
    RoomCreated { room_id: String },
    /// List of rooms.
    Rooms { rooms: Vec<RoomSummary> },
    /// Room details.
    Room { room: RoomSnapshot },
    /// Agent spawned.
    AgentSpawned { participant_id: String },
    /// Message sent.
    MessageSent { message_id: String },
    /// Room event (for subscriptions).
    Event { event: String },
    /// Room archived.
    RoomArchived,
    /// Pong response.
    Pong,
    /// Shutting down.
    ShuttingDown,
    /// Error.
    Error { message: String },
}

/// Summary of a room for list view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSummary {
    pub id: String,
    pub name: String,
    pub participant_count: usize,
    pub message_count: usize,
    pub archived: bool,
}

/// Snapshot of room for detailed view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSnapshot {
    pub id: String,
    pub name: String,
    pub working_dir: String,
    pub participants: Vec<ParticipantInfo>,
    pub archived: bool,
    pub created_at: String,
}

/// Info about a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantInfo {
    pub id: String,
    pub display_name: String,
    pub is_agent: bool,
    pub agent_type: Option<String>,
}

// =============================================================================
// Daemon Server
// =============================================================================

/// Returns the socket path for the daemon.
pub fn socket_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("infernum")
        .join("room.sock")
}

/// Checks if the daemon is running by attempting to connect.
pub async fn is_daemon_running() -> bool {
    UnixStream::connect(socket_path()).await.is_ok()
}

/// Runs the room daemon (blocking).
pub async fn run_daemon() -> Result<()> {
    let socket = socket_path();

    // Ensure parent directory exists
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Remove stale socket if it exists
    if socket.exists() {
        if is_daemon_running().await {
            return Err(eyre!(
                "Room daemon already running at {}",
                socket.display()
            ));
        }
        std::fs::remove_file(&socket)?;
    }

    // Create registry
    let registry = Arc::new(RoomRegistry::with_defaults());

    // Bind socket
    let listener = UnixListener::bind(&socket)?;

    // Set permissions (0o600 - owner only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    }

    info!("Room daemon listening on {}", socket.display());
    println!("Room daemon started: {}", socket.display());

    // Accept connections
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_clone = shutdown.clone();

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, _)) => {
                        let registry = registry.clone();
                        let shutdown = shutdown.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, registry, shutdown).await {
                                warn!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Accept error: {}", e);
                    }
                }
            }
            _ = shutdown_clone.notified() => {
                info!("Shutdown signal received");
                break;
            }
        }
    }

    // Cleanup
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }

    info!("Room daemon stopped");
    Ok(())
}

/// Handles a single client connection.
async fn handle_connection(
    stream: UnixStream,
    registry: Arc<RoomRegistry>,
    shutdown: Arc<tokio::sync::Notify>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Track if this client is subscribing
    let mut subscription: Option<broadcast::Receiver<RoomEvent>> = None;
    let mut subscribed_room: Option<RoomId> = None;

    loop {
        line.clear();

        tokio::select! {
            // Read from client
            result = reader.read_line(&mut line) => {
                match result {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        debug!("Received: {}", trimmed);

                        let request: DaemonRequest = match serde_json::from_str(trimmed) {
                            Ok(req) => req,
                            Err(e) => {
                                let resp = DaemonResponse::Error {
                                    message: format!("Invalid request: {}", e),
                                };
                                send_response(&mut writer, &resp).await?;
                                continue;
                            }
                        };

                        let response = handle_request(
                            request,
                            &registry,
                            &shutdown,
                            &mut subscription,
                            &mut subscribed_room,
                        )
                        .await;

                        send_response(&mut writer, &response).await?;

                        // Check for shutdown
                        if matches!(response, DaemonResponse::ShuttingDown) {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Read error: {}", e);
                        break;
                    }
                }
            }

            // Forward events for subscriptions
            event = async {
                if let Some(ref mut rx) = subscription {
                    rx.recv().await.ok()
                } else {
                    std::future::pending::<Option<RoomEvent>>().await
                }
            } => {
                if let Some(event) = event {
                    // Only forward events for subscribed room
                    if let Some(rid) = subscribed_room {
                        if event_matches_room(&event, rid) {
                            let event_json = serde_json::to_string(&format!("{:?}", event))
                                .unwrap_or_else(|_| "{}".to_string());
                            let resp = DaemonResponse::Event { event: event_json };
                            send_response(&mut writer, &resp).await?;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Sends a response to the client.
async fn send_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: &DaemonResponse,
) -> Result<()> {
    let json = serde_json::to_string(response)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

/// Checks if an event belongs to a room.
fn event_matches_room(event: &RoomEvent, room_id: RoomId) -> bool {
    match event {
        RoomEvent::RoomCreated { room_id: rid, .. } => *rid == room_id,
        RoomEvent::ParticipantJoined { room_id: rid, .. } => *rid == room_id,
        RoomEvent::ParticipantLeft { room_id: rid, .. } => *rid == room_id,
        RoomEvent::MessageSent { room_id: rid, .. } => *rid == room_id,
        RoomEvent::AttentionChanged { room_id: rid, .. } => *rid == room_id,
        RoomEvent::RoomArchived { room_id: rid, .. } => *rid == room_id,
        RoomEvent::RoomForked {
            source_room_id,
            new_room_id,
            ..
        } => *source_room_id == room_id || *new_room_id == room_id,
    }
}

/// Handles a single request.
async fn handle_request(
    request: DaemonRequest,
    registry: &Arc<RoomRegistry>,
    shutdown: &Arc<tokio::sync::Notify>,
    subscription: &mut Option<broadcast::Receiver<RoomEvent>>,
    subscribed_room: &mut Option<RoomId>,
) -> DaemonResponse {
    match request {
        DaemonRequest::Ping => DaemonResponse::Pong,

        DaemonRequest::Shutdown => {
            shutdown.notify_one();
            DaemonResponse::ShuttingDown
        }

        DaemonRequest::CreateRoom { name, working_dir } => {
            let user_id = UserId("cli-user".to_string());
            let request = CreateRoomRequest::new(&name, working_dir, user_id);

            match registry.create_room(request).await {
                Ok(room_id) => DaemonResponse::RoomCreated {
                    room_id: room_id.to_string(),
                },
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }

        DaemonRequest::ListRooms => {
            let rooms = registry.list_active_rooms().await;
            let summaries: Vec<RoomSummary> = rooms
                .into_iter()
                .map(|r| RoomSummary {
                    id: r.id.to_string(),
                    name: r.name.clone(),
                    participant_count: r.participants.len(),
                    message_count: 0, // Would need to fetch from messages
                    archived: r.archived,
                })
                .collect();

            DaemonResponse::Rooms { rooms: summaries }
        }

        DaemonRequest::GetRoom { room_id } => {
            let rid = match resolve_room_id(&room_id, registry).await {
                Ok(rid) => rid,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            match registry.get_room(rid).await {
                Some(room) => DaemonResponse::Room {
                    room: room_to_snapshot(&room),
                },
                None => DaemonResponse::Error {
                    message: "Room not found".to_string(),
                },
            }
        }

        DaemonRequest::SpawnAgent {
            room_id,
            agent_type,
            display_name,
        } => {
            let rid = match resolve_room_id(&room_id, registry).await {
                Ok(rid) => rid,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            let config = match parse_agent_type(&agent_type, display_name) {
                Ok(c) => c,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            // Get a human participant from the room to act as spawner
            let room = match registry.get_room(rid).await {
                Some(r) => r,
                None => {
                    return DaemonResponse::Error {
                        message: "Room not found".to_string(),
                    }
                }
            };

            let spawner = room
                .participants
                .iter()
                .find(|p| p.kind.is_human())
                .map(|p| p.id);

            let spawner = match spawner {
                Some(id) => id,
                None => {
                    return DaemonResponse::Error {
                        message: "No human participant in room to spawn agent".to_string(),
                    }
                }
            };

            // Spawn the agent
            match registry.spawn_agent(rid, config, spawner).await {
                Ok(participant_id) => DaemonResponse::AgentSpawned {
                    participant_id: participant_id.to_string(),
                },
                Err(e) => DaemonResponse::Error {
                    message: format!("Failed to spawn agent: {}", e),
                },
            }
        }

        DaemonRequest::SendMessage { room_id, content } => {
            let rid = match resolve_room_id(&room_id, registry).await {
                Ok(rid) => rid,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            // Get a human participant from the room to send as
            let room = match registry.get_room(rid).await {
                Some(r) => r,
                None => {
                    return DaemonResponse::Error {
                        message: "Room not found".to_string(),
                    }
                }
            };

            let sender = room
                .participants
                .iter()
                .find(|p| p.kind.is_human())
                .map(|p| p.id);

            let sender = match sender {
                Some(id) => id,
                None => {
                    return DaemonResponse::Error {
                        message: "No human participant in room to send message".to_string(),
                    }
                }
            };

            // Send the message
            match registry.send_message(rid, sender, content).await {
                Ok(message_id) => DaemonResponse::MessageSent {
                    message_id: message_id.0.to_string(),
                },
                Err(e) => DaemonResponse::Error {
                    message: format!("Failed to send message: {}", e),
                },
            }
        }

        DaemonRequest::Subscribe { room_id } => {
            let rid = match resolve_room_id(&room_id, registry).await {
                Ok(rid) => rid,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            *subscription = Some(registry.subscribe());
            *subscribed_room = Some(rid);

            DaemonResponse::Room {
                room: match registry.get_room(rid).await {
                    Some(r) => room_to_snapshot(&r),
                    None => {
                        return DaemonResponse::Error {
                            message: "Room not found".to_string(),
                        }
                    }
                },
            }
        }

        DaemonRequest::ArchiveRoom { room_id } => {
            let rid = match resolve_room_id(&room_id, registry).await {
                Ok(rid) => rid,
                Err(e) => return DaemonResponse::Error { message: e },
            };

            // Get a participant ID from the room
            let room = match registry.get_room(rid).await {
                Some(r) => r,
                None => {
                    return DaemonResponse::Error {
                        message: "Room not found".to_string(),
                    }
                }
            };

            let archiver = room
                .participants
                .first()
                .map(|p| p.id)
                .unwrap_or_else(conclave::ParticipantId::new);

            match registry.archive_room(rid, archiver).await {
                Ok(()) => DaemonResponse::RoomArchived,
                Err(e) => DaemonResponse::Error {
                    message: e.to_string(),
                },
            }
        }
    }
}

/// Resolves a room ID prefix to a full RoomId.
async fn resolve_room_id(prefix: &str, registry: &RoomRegistry) -> std::result::Result<RoomId, String> {
    let rooms = registry.list_active_rooms().await;
    let matches: Vec<_> = rooms
        .iter()
        .filter(|r| r.id.to_string().starts_with(prefix))
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
fn parse_agent_type(
    agent_type: &str,
    display_name: Option<String>,
) -> std::result::Result<AgentConfig, String> {
    let mut config = match agent_type.to_lowercase().as_str() {
        "claude-opus" => AgentConfig::claude_code(ClaudeTier::Opus),
        "claude-sonnet" => AgentConfig::claude_code(ClaudeTier::Sonnet),
        "claude-haiku" => AgentConfig::claude_code(ClaudeTier::Haiku),
        s if s.starts_with("infernum:") => {
            let model = s.strip_prefix("infernum:").unwrap();
            AgentConfig::infernum(model)
        }
        _ => {
            return Err(format!(
                "Unknown agent type: {}. Use claude-opus, claude-sonnet, claude-haiku, or infernum:<model>",
                agent_type
            ))
        }
    };

    if let Some(name) = display_name {
        config.display_name = Some(name);
    }

    Ok(config)
}

/// Converts a Room to a RoomSnapshot.
fn room_to_snapshot(room: &Room) -> RoomSnapshot {
    RoomSnapshot {
        id: room.id.to_string(),
        name: room.name.clone(),
        working_dir: room.working_dir.display().to_string(),
        participants: room
            .participants
            .iter()
            .map(|p| ParticipantInfo {
                id: p.id.to_string(),
                display_name: p.display_name.clone(),
                is_agent: p.kind.is_agent(),
                agent_type: match &p.kind {
                    conclave::ParticipantKind::Agent { backend, .. } => {
                        Some(format!("{:?}", backend))
                    }
                    _ => None,
                },
            })
            .collect(),
        archived: room.archived,
        created_at: room.created_at.to_rfc3339(),
    }
}
