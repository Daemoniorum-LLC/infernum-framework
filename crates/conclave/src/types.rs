//! Core type definitions for the Conclave collaboration system.
//!
//! These types are derived from AGENT-COLLABORATION-SPEC.md §2.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// =============================================================================
// Identifiers
// =============================================================================

/// Unique room identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RoomId(pub Uuid);

impl RoomId {
    /// Generates a new random RoomId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RoomId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RoomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "room_{}", &self.0.simple().to_string()[..12])
    }
}

/// Unique participant identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ParticipantId(pub Uuid);

impl ParticipantId {
    /// Generates a new random ParticipantId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ParticipantId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "part_{}", &self.0.simple().to_string()[..12])
    }
}

/// Unique message identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageId(pub Uuid);

impl MessageId {
    /// Generates a new random MessageId.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// User identifier (for humans).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub String);

// =============================================================================
// Participant Types
// =============================================================================

/// A participant in a room (human or agent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Unique participant identifier.
    pub id: ParticipantId,

    /// Display name (chosen name for agents, user name for humans).
    pub display_name: String,

    /// Type of participant (Human or Agent).
    pub kind: ParticipantKind,

    /// Applied persona code (if any).
    pub persona: Option<String>,

    /// Current attention state.
    pub attention: AttentionState,

    /// When this participant joined the room.
    pub joined_at: DateTime<Utc>,

    /// Last activity timestamp.
    pub last_active: DateTime<Utc>,

    /// Number of messages sent by this participant.
    pub message_count: u32,

    /// Number of tool calls made (agents only).
    pub tool_calls: u32,
}

/// Type of participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParticipantKind {
    /// Human participant.
    Human {
        /// User identifier.
        user_id: UserId,
    },
    /// Agent participant.
    Agent {
        /// Backend powering this agent.
        backend: AgentBackend,
        /// Session identifier from the backend.
        session_id: String,
        /// Who spawned this agent.
        spawned_by: ParticipantId,
    },
}

impl ParticipantKind {
    /// Returns true if this is an agent.
    pub fn is_agent(&self) -> bool {
        matches!(self, ParticipantKind::Agent { .. })
    }

    /// Returns true if this is a human.
    pub fn is_human(&self) -> bool {
        matches!(self, ParticipantKind::Human { .. })
    }
}

/// Backend powering an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum AgentBackend {
    /// Infernum native agent (local LLM).
    Infernum {
        /// Model identifier.
        model: String,
        /// Inference backend (cuda, metal, cpu).
        inference_backend: InferenceBackend,
        /// Enabled tools.
        tools: Vec<String>,
    },
    /// Claude Code CLI agent.
    ClaudeCode {
        /// Claude tier (Opus, Sonnet, Haiku).
        tier: ClaudeTier,
        /// Allowed tools for auto-approval.
        allowed_tools: Vec<String>,
    },
    /// OpenAI Codex agent.
    Codex {
        /// API key reference (not the actual key).
        api_key_ref: String,
        /// Model name.
        model: String,
    },
    /// Cursor AI agent.
    Cursor {
        /// Workspace path.
        workspace: PathBuf,
        /// Enabled features.
        features: Vec<CursorFeature>,
    },
    /// Custom agent implementation.
    Custom {
        /// Endpoint URL.
        endpoint: String,
        /// Communication protocol.
        protocol: AgentProtocol,
    },
}

/// Inference backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// CPU fallback.
    Cpu,
}

/// Claude model tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaudeTier {
    /// Claude Opus (most capable).
    Opus,
    /// Claude Sonnet (balanced).
    Sonnet,
    /// Claude Haiku (fast).
    Haiku,
}

/// Cursor feature flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CursorFeature {
    /// Code completion.
    Autocomplete,
    /// Chat interface.
    Chat,
    /// Composer mode.
    Composer,
}

/// Agent communication protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentProtocol {
    /// JSON-RPC over HTTP.
    JsonRpc,
    /// gRPC.
    Grpc,
    /// WebSocket.
    WebSocket,
}

// =============================================================================
// Attention States
// =============================================================================

/// Attention state for a participant.
///
/// Agents control their own attention state, similar to how humans
/// set their status in Slack. Humans can override with pause/urgent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AttentionState {
    /// Available and responsive to messages.
    Available,

    /// Focused on a specific task.
    Focused {
        /// What the participant is working on.
        task: String,
        /// When focus started.
        started: DateTime<Utc>,
        /// Estimated completion time.
        eta: Option<chrono::Duration>,
        /// Whether this can be interrupted.
        interruptible: bool,
    },

    /// Do not disturb - suppress non-urgent messages.
    DoNotDisturb {
        /// Reason for DND.
        reason: String,
        /// When DND expires.
        until: Option<DateTime<Utc>>,
    },

    /// Away (not actively participating).
    Away {
        /// When the participant went away.
        since: DateTime<Utc>,
        /// Optional reason.
        reason: Option<String>,
    },
}

impl Default for AttentionState {
    fn default() -> Self {
        Self::Available
    }
}

impl AttentionState {
    /// Returns true if the participant should receive messages.
    pub fn should_receive_messages(&self, urgent: bool) -> bool {
        match self {
            AttentionState::Available => true,
            AttentionState::Focused { interruptible, .. } => urgent || *interruptible,
            AttentionState::DoNotDisturb { .. } => urgent,
            AttentionState::Away { .. } => false,
        }
    }
}

// =============================================================================
// Channel Types
// =============================================================================

/// Channel type within a room.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "channel_type", rename_all = "snake_case")]
pub enum ChannelType {
    /// Main room channel (coordinated turn-taking).
    Main,

    /// Agent reasoning channel (private to one agent).
    AgentReasoning {
        /// The agent this channel belongs to.
        agent_id: ParticipantId,
    },

    /// Direct message between participants.
    DirectMessage {
        /// Participants in this DM.
        participants: Vec<ParticipantId>,
    },

    /// Thread attached to a message.
    Thread {
        /// Parent message ID.
        parent_message: MessageId,
    },
}

/// A message in a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Unique message identifier.
    pub id: MessageId,

    /// Channel this message belongs to.
    pub channel: ChannelType,

    /// Who sent this message.
    pub sender: ParticipantId,

    /// Message content.
    pub content: MessageContent,

    /// When the message was sent.
    pub timestamp: DateTime<Utc>,

    /// Metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Message content variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    /// Plain text message.
    Text { content: String },

    /// Tool call (agent action).
    ToolCall {
        tool: String,
        input: serde_json::Value,
        call_id: String,
    },

    /// Tool result.
    ToolResult {
        tool: String,
        output: serde_json::Value,
        call_id: String,
        success: bool,
    },

    /// System event.
    System { event: SystemEvent },
}

/// System events in the main channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SystemEvent {
    /// Participant joined the room.
    ParticipantJoined {
        participant_id: ParticipantId,
        display_name: String,
    },

    /// Participant left the room.
    ParticipantLeft {
        participant_id: ParticipantId,
        reason: LeaveReason,
    },

    /// Participant changed attention state.
    AttentionChanged {
        participant_id: ParticipantId,
        new_state: AttentionState,
    },

    /// Room was archived.
    RoomArchived {
        archived_by: ParticipantId,
        message_count: u64,
    },

    /// Room was forked.
    RoomForked {
        new_room_id: RoomId,
        forked_by: ParticipantId,
    },
}

/// Reason a participant left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaveReason {
    /// Voluntarily left.
    Left,
    /// Kicked by moderator.
    Kicked,
    /// Backend terminated.
    Terminated,
    /// Room archived.
    Archived,
}

// =============================================================================
// Room Configuration
// =============================================================================

/// Invite policy for a room.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvitePolicy {
    /// Anyone can invite, announce to room.
    #[default]
    Announce,
    /// Require approval from existing participants.
    RequireApproval,
    /// Only the room owner can invite.
    OwnerOnly,
}

/// Turn-taking mode for the main channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinatorMode {
    /// Participants volunteer for turns.
    #[default]
    Volunteer,
    /// Round-robin turn order.
    RoundRobin,
    /// Priority-based (urgency scores).
    Priority,
}

/// Coordinator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Turn-taking mode.
    pub mode: CoordinatorMode,

    /// Turn timeout in seconds.
    pub turn_timeout_secs: u32,

    /// Maximum queue depth.
    pub max_queue_depth: u32,

    /// Whether to allow parallel tool execution.
    pub allow_parallel_tools: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            mode: CoordinatorMode::Volunteer,
            turn_timeout_secs: 300, // 5 minutes
            max_queue_depth: 10,
            allow_parallel_tools: true,
        }
    }
}

/// Project reference for a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProjectRef {
    /// Git repository.
    GitRepo {
        /// Remote URL.
        remote: String,
        /// Default branch.
        branch: String,
    },
    /// Local directory.
    LocalDir {
        /// Absolute path.
        path: PathBuf,
    },
}

// =============================================================================
// Room Definition
// =============================================================================

/// A collaborative room where humans and agents work together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    /// Unique room identifier.
    pub id: RoomId,

    /// Room name/title.
    pub name: String,

    /// Working directory for file operations.
    pub working_dir: PathBuf,

    /// Project reference (git repo, local dir).
    pub project: Option<ProjectRef>,

    /// Active participants.
    pub participants: Vec<Participant>,

    /// Former participants (left, kicked, terminated).
    pub alumni: Vec<Participant>,

    /// Invite policy.
    pub invite_policy: InvitePolicy,

    /// Coordinator configuration.
    pub coordinator_config: CoordinatorConfig,

    /// Whether this room is archived.
    pub archived: bool,

    /// If this room was forked, the source room.
    pub fork_of: Option<RoomId>,

    /// When the room was created.
    pub created_at: DateTime<Utc>,

    /// Last activity timestamp.
    pub updated_at: DateTime<Utc>,
}

impl Room {
    /// Returns the number of active participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Returns the number of active agents.
    pub fn agent_count(&self) -> usize {
        self.participants.iter().filter(|p| p.kind.is_agent()).count()
    }

    /// Returns the number of active humans.
    pub fn human_count(&self) -> usize {
        self.participants.iter().filter(|p| p.kind.is_human()).count()
    }

    /// Finds a participant by ID.
    pub fn find_participant(&self, id: ParticipantId) -> Option<&Participant> {
        self.participants.iter().find(|p| p.id == id)
    }

    /// Finds a participant by ID (mutable).
    pub fn find_participant_mut(&mut self, id: ParticipantId) -> Option<&mut Participant> {
        self.participants.iter_mut().find(|p| p.id == id)
    }
}

// =============================================================================
// Request Types
// =============================================================================

/// Request to create a new room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRoomRequest {
    /// Room name/title.
    pub name: String,

    /// Working directory for file operations.
    pub working_dir: PathBuf,

    /// Project reference.
    pub project: Option<ProjectRef>,

    /// Initial agents to spawn.
    pub initial_agents: Vec<AgentConfig>,

    /// Invite policy.
    pub invite_policy: Option<InvitePolicy>,

    /// Coordinator configuration.
    pub coordinator_config: Option<CoordinatorConfig>,

    /// Who is creating this room.
    pub creator: UserId,
}

impl CreateRoomRequest {
    /// Creates a new request with the given name.
    pub fn new(name: impl Into<String>, working_dir: PathBuf, creator: UserId) -> Self {
        Self {
            name: name.into(),
            working_dir,
            project: None,
            initial_agents: Vec::new(),
            invite_policy: None,
            coordinator_config: None,
            creator,
        }
    }

    /// Adds an initial agent.
    pub fn with_agent(mut self, config: AgentConfig) -> Self {
        self.initial_agents.push(config);
        self
    }

    /// Sets the project reference.
    pub fn with_project(mut self, project: ProjectRef) -> Self {
        self.project = Some(project);
        self
    }

    /// Sets the invite policy.
    pub fn with_invite_policy(mut self, policy: InvitePolicy) -> Self {
        self.invite_policy = Some(policy);
        self
    }
}

/// Configuration for spawning an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Display name for the agent.
    pub display_name: Option<String>,

    /// Backend configuration.
    pub backend: AgentBackend,

    /// Persona code from Grimoire.
    pub persona: Option<String>,
}

impl AgentConfig {
    /// Creates a Claude Code agent config.
    pub fn claude_code(tier: ClaudeTier) -> Self {
        Self {
            display_name: Some(format!("Claude {:?}", tier)),
            backend: AgentBackend::ClaudeCode {
                tier,
                allowed_tools: vec!["Read".to_string(), "Glob".to_string(), "Grep".to_string()],
            },
            persona: None,
        }
    }

    /// Creates an Infernum native agent config.
    pub fn infernum(model: impl Into<String>) -> Self {
        Self {
            display_name: Some("Infernum".to_string()),
            backend: AgentBackend::Infernum {
                model: model.into(),
                inference_backend: InferenceBackend::Cuda,
                tools: vec!["file_read".to_string(), "glob".to_string(), "grep".to_string()],
            },
            persona: None,
        }
    }

    /// Sets the display name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Sets the persona code.
    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_id_uniqueness() {
        let id1 = RoomId::new();
        let id2 = RoomId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_room_id_display() {
        let id = RoomId::new();
        let display = format!("{}", id);
        assert!(display.starts_with("room_"));
        assert_eq!(display.len(), 17); // "room_" + 12 chars
    }

    #[test]
    fn test_participant_id_display() {
        let id = ParticipantId::new();
        let display = format!("{}", id);
        assert!(display.starts_with("part_"));
    }

    #[test]
    fn test_attention_state_available() {
        let state = AttentionState::Available;
        assert!(state.should_receive_messages(false));
        assert!(state.should_receive_messages(true));
    }

    #[test]
    fn test_attention_state_focused_interruptible() {
        let state = AttentionState::Focused {
            task: "Working".to_string(),
            started: Utc::now(),
            eta: None,
            interruptible: true,
        };
        assert!(state.should_receive_messages(false));
        assert!(state.should_receive_messages(true));
    }

    #[test]
    fn test_attention_state_focused_not_interruptible() {
        let state = AttentionState::Focused {
            task: "Deep work".to_string(),
            started: Utc::now(),
            eta: None,
            interruptible: false,
        };
        assert!(!state.should_receive_messages(false));
        assert!(state.should_receive_messages(true)); // urgent still goes through
    }

    #[test]
    fn test_attention_state_dnd() {
        let state = AttentionState::DoNotDisturb {
            reason: "Compiling".to_string(),
            until: None,
        };
        assert!(!state.should_receive_messages(false));
        assert!(state.should_receive_messages(true)); // urgent still goes through
    }

    #[test]
    fn test_attention_state_away() {
        let state = AttentionState::Away {
            since: Utc::now(),
            reason: None,
        };
        assert!(!state.should_receive_messages(false));
        assert!(!state.should_receive_messages(true)); // even urgent doesn't go to away
    }

    #[test]
    fn test_participant_kind_is_agent() {
        let agent = ParticipantKind::Agent {
            backend: AgentBackend::ClaudeCode {
                tier: ClaudeTier::Opus,
                allowed_tools: vec![],
            },
            session_id: "sess_123".to_string(),
            spawned_by: ParticipantId::new(),
        };
        assert!(agent.is_agent());
        assert!(!agent.is_human());
    }

    #[test]
    fn test_participant_kind_is_human() {
        let human = ParticipantKind::Human {
            user_id: UserId("user123".to_string()),
        };
        assert!(human.is_human());
        assert!(!human.is_agent());
    }

    #[test]
    fn test_agent_config_claude_code() {
        let config = AgentConfig::claude_code(ClaudeTier::Opus);
        assert!(matches!(config.backend, AgentBackend::ClaudeCode { tier: ClaudeTier::Opus, .. }));
    }

    #[test]
    fn test_agent_config_infernum() {
        let config = AgentConfig::infernum("qwen-7b");
        assert!(matches!(config.backend, AgentBackend::Infernum { model, .. } if model == "qwen-7b"));
    }

    #[test]
    fn test_create_room_request_builder() {
        let request = CreateRoomRequest::new(
            "Fix auth bug",
            PathBuf::from("/home/user/project"),
            UserId("user123".to_string()),
        )
        .with_agent(AgentConfig::claude_code(ClaudeTier::Opus))
        .with_agent(AgentConfig::infernum("qwen-7b"))
        .with_invite_policy(InvitePolicy::Announce);

        assert_eq!(request.name, "Fix auth bug");
        assert_eq!(request.initial_agents.len(), 2);
        assert_eq!(request.invite_policy, Some(InvitePolicy::Announce));
    }
}
