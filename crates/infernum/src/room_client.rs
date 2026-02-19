//! Room daemon client.
//!
//! Connects to the room daemon over Unix socket to execute commands.
//!
//! Implements CONCLAVE-CLI-SPEC.md §3.1 Daemon Protocol.

use std::path::PathBuf;

use color_eyre::eyre::{eyre, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use crate::room_daemon::{
    socket_path, DaemonRequest, DaemonResponse, ParticipantInfo, RoomSnapshot, RoomSummary,
};

// =============================================================================
// Daemon Client
// =============================================================================

/// Client for communicating with the room daemon.
pub struct DaemonClient {
    stream: UnixStream,
}

impl DaemonClient {
    /// Connects to the room daemon.
    pub async fn connect() -> Result<Self> {
        let socket = socket_path();

        let stream = UnixStream::connect(&socket).await.map_err(|e| {
            eyre!(
                "Failed to connect to room daemon at {}: {}\n\
                 Is the daemon running? Start with: infernum room daemon",
                socket.display(),
                e
            )
        })?;

        Ok(Self { stream })
    }

    /// Sends a request and receives a response.
    async fn request(&mut self, req: DaemonRequest) -> Result<DaemonResponse> {
        let json = serde_json::to_string(&req)?;

        self.stream.write_all(json.as_bytes()).await?;
        self.stream.write_all(b"\n").await?;
        self.stream.flush().await?;

        let mut reader = BufReader::new(&mut self.stream);
        let mut line = String::new();
        reader.read_line(&mut line).await?;

        let response: DaemonResponse = serde_json::from_str(line.trim())?;
        Ok(response)
    }

    /// Pings the daemon.
    pub async fn ping(&mut self) -> Result<()> {
        match self.request(DaemonRequest::Ping).await? {
            DaemonResponse::Pong => Ok(()),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Creates a new room.
    pub async fn create_room(&mut self, name: String, working_dir: PathBuf) -> Result<String> {
        match self
            .request(DaemonRequest::CreateRoom { name, working_dir })
            .await?
        {
            DaemonResponse::RoomCreated { room_id } => Ok(room_id),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Lists all active rooms.
    pub async fn list_rooms(&mut self) -> Result<Vec<RoomSummary>> {
        match self.request(DaemonRequest::ListRooms).await? {
            DaemonResponse::Rooms { rooms } => Ok(rooms),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Gets room details.
    pub async fn get_room(&mut self, room_id: String) -> Result<RoomSnapshot> {
        match self.request(DaemonRequest::GetRoom { room_id }).await? {
            DaemonResponse::Room { room } => Ok(room),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Spawns an agent in a room.
    pub async fn spawn_agent(
        &mut self,
        room_id: String,
        agent_type: String,
        display_name: Option<String>,
    ) -> Result<String> {
        match self
            .request(DaemonRequest::SpawnAgent {
                room_id,
                agent_type,
                display_name,
            })
            .await?
        {
            DaemonResponse::AgentSpawned { participant_id } => Ok(participant_id),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Sends a message to a room.
    pub async fn send_message(&mut self, room_id: String, content: String) -> Result<String> {
        match self
            .request(DaemonRequest::SendMessage { room_id, content })
            .await?
        {
            DaemonResponse::MessageSent { message_id } => Ok(message_id),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Subscribes to room events.
    pub async fn subscribe(&mut self, room_id: String) -> Result<RoomSnapshot> {
        match self.request(DaemonRequest::Subscribe { room_id }).await? {
            DaemonResponse::Room { room } => Ok(room),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Archives a room.
    pub async fn archive_room(&mut self, room_id: String) -> Result<()> {
        match self.request(DaemonRequest::ArchiveRoom { room_id }).await? {
            DaemonResponse::RoomArchived => Ok(()),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Requests daemon shutdown.
    pub async fn shutdown(&mut self) -> Result<()> {
        match self.request(DaemonRequest::Shutdown).await? {
            DaemonResponse::ShuttingDown => Ok(()),
            DaemonResponse::Error { message } => Err(eyre!("{}", message)),
            _ => Err(eyre!("Unexpected response")),
        }
    }

    /// Reads the next event (for subscribers).
    pub async fn read_event(&mut self) -> Result<Option<String>> {
        let mut reader = BufReader::new(&mut self.stream);
        let mut line = String::new();

        match reader.read_line(&mut line).await {
            Ok(0) => Ok(None), // EOF
            Ok(_) => {
                let response: DaemonResponse = serde_json::from_str(line.trim())?;
                match response {
                    DaemonResponse::Event { event } => Ok(Some(event)),
                    DaemonResponse::Error { message } => Err(eyre!("{}", message)),
                    _ => Ok(None),
                }
            }
            Err(e) => Err(e.into()),
        }
    }
}

// =============================================================================
// Formatting Helpers
// =============================================================================

/// Formats a room summary for display.
pub fn format_room_summary(room: &RoomSummary) -> String {
    let status = if room.archived { " [archived]" } else { "" };
    format!(
        "{} - {} ({} participants){}",
        short_id(&room.id),
        room.name,
        room.participant_count,
        status
    )
}

/// Formats room details for display.
pub fn format_room_details(room: &RoomSnapshot) -> String {
    let mut lines = Vec::new();

    lines.push(format!("Room: {} [{}]", room.name, short_id(&room.id)));
    lines.push(format!("Working Dir: {}", room.working_dir));
    lines.push(format!("Created: {}", room.created_at));
    lines.push(format!(
        "Status: {}",
        if room.archived { "Archived" } else { "Active" }
    ));
    lines.push(String::new());
    lines.push(format!("Participants ({}):", room.participants.len()));

    for p in &room.participants {
        let kind = if p.is_agent {
            format!(" ({})", p.agent_type.as_deref().unwrap_or("agent"))
        } else {
            " (human)".to_string()
        };
        lines.push(format!("  {} {}{}", short_id(&p.id), p.display_name, kind));
    }

    lines.join("\n")
}

/// Formats a participant for display.
pub fn format_participant(p: &ParticipantInfo) -> String {
    let kind = if p.is_agent {
        p.agent_type.as_deref().unwrap_or("agent")
    } else {
        "human"
    };
    format!("{} ({}) [{}]", p.display_name, kind, short_id(&p.id))
}

/// Returns short version of an ID (first 8 chars).
fn short_id(id: &str) -> String {
    if id.len() > 8 {
        id[..8].to_string()
    } else {
        id.to_string()
    }
}
