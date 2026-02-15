//! Agent backend implementations.
//!
//! Implements AGENT-COLLABORATION-SPEC.md §2.3 Agent Backend Types and §4 Agent Capabilities.
//!
//! Each backend (Infernum, Claude Code, Codex, Cursor, Custom) implements the
//! `AgentBackendSession` trait, providing a unified interface for spawning,
//! communicating with, and terminating agent processes.
//!
//! # Module Structure
//!
//! - `process` - Common infrastructure for process-based backends
//! - `claude_code` - Claude Code CLI integration

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{ConclaveError, Result};
use crate::types::{AgentBackend, AgentConfig, AttentionState, Message, ParticipantId, RoomId};

// Submodules
pub mod claude_code;
pub mod http;
pub mod infernum;
pub mod process;

// Re-exports
pub use claude_code::{ClaudeCodeConfig, ClaudeCodeParser};
pub use http::{HttpConfig, HttpSession};
pub use infernum::{InfernumConfig, InfernumParser};
pub use process::{OutputParser, PlainTextParser, ProcessSession};

// =============================================================================
// Agent Events
// =============================================================================

/// Events produced by an agent backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent sent a text message.
    Message {
        content: String,
        #[serde(default)]
        mentions: Vec<ParticipantId>,
    },

    /// Agent made a tool call.
    ToolCall {
        tool: String,
        input: serde_json::Value,
        call_id: String,
    },

    /// Tool call completed.
    ToolResult {
        /// Which tool was called.
        tool: String,
        /// The tool call ID this result corresponds to.
        call_id: String,
        /// Tool output.
        output: String,
        /// Whether the tool succeeded.
        success: bool,
        /// Execution time in milliseconds.
        duration_ms: u32,
    },

    /// Agent is thinking (internal reasoning).
    Thinking { content: String },

    /// Agent changed attention state.
    AttentionChanged { new_state: AttentionState },

    /// Agent requested a speaking turn.
    TurnRequested {
        reason: Option<String>,
        priority: TurnPriority,
    },

    /// Agent yielded speaking turn.
    TurnYielded,

    /// Agent wants to invite another agent.
    InviteRequested {
        config: AgentConfig,
        reason: String,
    },

    /// Agent process terminated.
    Terminated { reason: TerminationReason },

    /// Agent encountered an error.
    Error { message: String },
}

/// Priority level for turn requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnPriority {
    /// Normal priority.
    Normal,
    /// High priority (has something important).
    High,
    /// Urgent (needs immediate attention).
    Urgent,
    /// Yielding (finishing up, will yield soon).
    Yielding,
}

impl Default for TurnPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Reason for agent termination.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationReason {
    /// Normal completion.
    Completed,
    /// User/human requested termination.
    Requested,
    /// Room was archived.
    RoomArchived,
    /// Backend process crashed.
    Crashed { error: String },
    /// Timeout.
    Timeout,
}

// =============================================================================
// Room Context
// =============================================================================

/// Context provided to an agent when joining a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomContext {
    /// Room identifier.
    pub room_id: RoomId,
    /// Room name.
    pub room_name: String,
    /// Working directory for file operations.
    pub working_dir: PathBuf,
    /// Recent messages from main channel.
    pub recent_messages: Vec<Message>,
    /// List of participants with their attention states.
    pub participants: Vec<ParticipantSummary>,
    /// Optional system prompt addition for persona.
    pub persona_prompt: Option<String>,
}

/// Summary of a participant for context injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantSummary {
    /// Participant ID.
    pub id: ParticipantId,
    /// Display name.
    pub display_name: String,
    /// Whether this is an agent.
    pub is_agent: bool,
    /// Current attention state.
    pub attention: AttentionState,
}

// =============================================================================
// Agent Backend Session Trait
// =============================================================================

/// A running agent backend session.
///
/// Each backend (Infernum, Claude Code, Codex, Cursor, Custom) implements this
/// trait to provide a unified interface for the room system.
#[async_trait]
pub trait AgentBackendSession: Send + Sync {
    /// Returns the session identifier.
    fn session_id(&self) -> &str;

    /// Returns the backend configuration.
    fn backend(&self) -> &AgentBackend;

    /// Sends a message to the agent.
    ///
    /// This delivers a message from the room to the agent, which may trigger
    /// the agent to respond with events.
    async fn send_message(&self, message: &Message) -> Result<()>;

    /// Sends an interrupt signal to the agent.
    ///
    /// Used for pause commands and urgent messages. The agent should:
    /// 1. Complete current atomic operation if safe
    /// 2. Save state
    /// 3. Check for the interrupt reason
    async fn interrupt(&self) -> Result<()>;

    /// Terminates the agent session.
    ///
    /// This should gracefully shut down the backend process.
    async fn terminate(&self) -> Result<()>;

    /// Returns true if the session is still running.
    fn is_running(&self) -> bool;

    /// Takes the event receiver for this session.
    ///
    /// Events from the agent backend are received through this channel.
    /// This can only be called once - subsequent calls return `None`.
    /// The receiver is moved out to allow the caller to own it.
    fn take_event_receiver(&self) -> Option<mpsc::Receiver<AgentEvent>>;
}

// =============================================================================
// Agent Backend Factory
// =============================================================================

/// Factory for creating agent backend sessions.
pub struct AgentBackendFactory {
    /// Working directory for spawning processes.
    #[allow(dead_code)]
    working_dir: PathBuf,
}

impl AgentBackendFactory {
    /// Creates a new factory.
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }

    /// Spawns an agent backend session.
    ///
    /// This creates a new backend session based on the configuration,
    /// injects the room context, and returns a session handle.
    pub async fn spawn(
        &self,
        config: &AgentConfig,
        context: RoomContext,
        spawned_by: ParticipantId,
    ) -> Result<Box<dyn AgentBackendSession>> {
        match &config.backend {
            AgentBackend::Infernum { model, .. } => {
                self.spawn_infernum(config, context, spawned_by, model)
                    .await
            }
            AgentBackend::ClaudeCode { tier, .. } => {
                self.spawn_claude_code(config, context, spawned_by, *tier)
                    .await
            }
            AgentBackend::Codex { model, .. } => {
                self.spawn_codex(config, context, spawned_by, model).await
            }
            AgentBackend::Cursor { workspace, .. } => {
                self.spawn_cursor(config, context, spawned_by, workspace)
                    .await
            }
            AgentBackend::Custom { endpoint, .. } => {
                self.spawn_custom(config, context, spawned_by, endpoint)
                    .await
            }
        }
    }

    // -------------------------------------------------------------------------
    // Backend-Specific Spawning
    // -------------------------------------------------------------------------

    async fn spawn_infernum(
        &self,
        config: &AgentConfig,
        context: RoomContext,
        _spawned_by: ParticipantId,
        model: &str,
    ) -> Result<Box<dyn AgentBackendSession>> {
        let session_id = format!("infernum-{}", &uuid::Uuid::new_v4().to_string()[..8]);

        // Check if infernum binary exists
        if !Self::binary_exists("infernum") {
            // Fall back to stub session in test/dev environments
            tracing::warn!("Infernum binary not found, using stub session");
            return Ok(Box::new(StubSession::new(session_id, config.backend.clone())));
        }

        // Build Infernum configuration
        let mut inf_config = infernum::InfernumConfig::new(model, &context.working_dir);

        // Apply persona prompt if provided
        if let Some(ref persona) = context.persona_prompt {
            inf_config.system_prompt = Some(persona.clone());
        }

        // Spawn the real Infernum process
        infernum::spawn_infernum(session_id, inf_config, &context).await
    }

    async fn spawn_claude_code(
        &self,
        config: &AgentConfig,
        context: RoomContext,
        _spawned_by: ParticipantId,
        tier: crate::types::ClaudeTier,
    ) -> Result<Box<dyn AgentBackendSession>> {
        let session_id = format!("claude-{}", &uuid::Uuid::new_v4().to_string()[..8]);

        // Check if claude binary exists
        if !Self::binary_exists("claude") {
            // Fall back to stub session in test/dev environments
            tracing::warn!("Claude Code binary not found, using stub session");
            return Ok(Box::new(StubSession::new(session_id, config.backend.clone())));
        }

        // Build Claude Code configuration
        let mut claude_config = claude_code::ClaudeCodeConfig::new(tier, &context.working_dir);

        // Apply persona prompt if provided
        if let Some(ref persona) = context.persona_prompt {
            claude_config.system_prompt = Some(persona.clone());
        }

        // Extract allowed tools from backend config
        if let AgentBackend::ClaudeCode { allowed_tools, .. } = &config.backend {
            claude_config.allowed_tools = allowed_tools.clone();
        }

        // Spawn the real Claude Code process
        claude_code::spawn_claude_code(session_id, claude_config, &context).await
    }

    async fn spawn_codex(
        &self,
        config: &AgentConfig,
        _context: RoomContext,
        _spawned_by: ParticipantId,
        _model: &str,
    ) -> Result<Box<dyn AgentBackendSession>> {
        // TODO: Implement Codex API client
        Ok(Box::new(StubSession::new(
            format!("codex-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            config.backend.clone(),
        )))
    }

    async fn spawn_cursor(
        &self,
        config: &AgentConfig,
        _context: RoomContext,
        _spawned_by: ParticipantId,
        _workspace: &PathBuf,
    ) -> Result<Box<dyn AgentBackendSession>> {
        // TODO: Implement Cursor integration
        Ok(Box::new(StubSession::new(
            format!("cursor-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            config.backend.clone(),
        )))
    }

    async fn spawn_custom(
        &self,
        _config: &AgentConfig,
        context: RoomContext,
        _spawned_by: ParticipantId,
        endpoint: &str,
    ) -> Result<Box<dyn AgentBackendSession>> {
        let session_id = format!("custom-{}", &uuid::Uuid::new_v4().to_string()[..8]);

        // Extract protocol from config or default to JsonRpc
        let protocol = crate::types::AgentProtocol::JsonRpc;

        // Spawn HTTP session
        http::spawn_http(session_id, endpoint, protocol, &context)
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Checks if a binary exists in PATH.
    fn binary_exists(name: &str) -> bool {
        std::process::Command::new("which")
            .arg(name)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

// =============================================================================
// Stub Session (for testing and placeholder)
// =============================================================================

/// A stub session for testing purposes.
///
/// This session doesn't actually run a backend process, but implements
/// the trait interface for testing and as a placeholder during development.
pub struct StubSession {
    session_id: String,
    backend: AgentBackend,
    running: std::sync::atomic::AtomicBool,
    event_tx: mpsc::Sender<AgentEvent>,
    event_rx: std::sync::Mutex<Option<mpsc::Receiver<AgentEvent>>>,
}

impl StubSession {
    /// Creates a new stub session.
    pub fn new(session_id: String, backend: AgentBackend) -> Self {
        let (event_tx, event_rx) = mpsc::channel(100);
        Self {
            session_id,
            backend,
            running: std::sync::atomic::AtomicBool::new(true),
            event_tx,
            event_rx: std::sync::Mutex::new(Some(event_rx)),
        }
    }
}

#[async_trait]
impl AgentBackendSession for StubSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn backend(&self) -> &AgentBackend {
        &self.backend
    }

    async fn send_message(&self, _message: &Message) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }
        // Stub: just acknowledge receipt
        Ok(())
    }

    async fn interrupt(&self) -> Result<()> {
        if !self.is_running() {
            return Err(ConclaveError::BackendTerminated {
                session_id: self.session_id.clone(),
            });
        }
        // Stub: just acknowledge interrupt
        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // Send termination event
        let _ = self
            .event_tx
            .send(AgentEvent::Terminated {
                reason: TerminationReason::Requested,
            })
            .await;

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn take_event_receiver(&self) -> Option<mpsc::Receiver<AgentEvent>> {
        self.event_rx.lock().unwrap().take()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::types::ClaudeTier;

    fn test_working_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn test_context() -> RoomContext {
        RoomContext {
            room_id: RoomId::new(),
            room_name: "Test Room".to_string(),
            working_dir: test_working_dir(),
            recent_messages: Vec::new(),
            participants: Vec::new(),
            persona_prompt: None,
        }
    }

    // -------------------------------------------------------------------------
    // Phase 6.1: Backend Trait Tests
    // -------------------------------------------------------------------------

    /// spec_backend_trait_implemented - All backends implement AgentBackend
    #[tokio::test]
    async fn test_infernum_backend_spawns() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig::infernum("qwen-7b");
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await;
        assert!(session.is_ok());

        let session = session.unwrap();
        assert!(session.is_running());
        assert!(session.session_id().starts_with("infernum-"));
    }

    /// spec_claude_spawn
    #[tokio::test]
    async fn test_claude_code_backend_spawns() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig::claude_code(ClaudeTier::Opus);
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await;
        assert!(session.is_ok());

        let session = session.unwrap();
        assert!(session.is_running());
        assert!(session.session_id().starts_with("claude-"));
    }

    /// Codex backend spawns
    #[tokio::test]
    async fn test_codex_backend_spawns() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig {
            display_name: Some("Codex".to_string()),
            backend: AgentBackend::Codex {
                api_key_ref: "test_key".to_string(),
                model: "code-davinci-002".to_string(),
            },
            persona: None,
        };
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await;
        assert!(session.is_ok());
        assert!(session.unwrap().session_id().starts_with("codex-"));
    }

    /// Cursor backend spawns
    #[tokio::test]
    async fn test_cursor_backend_spawns() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig {
            display_name: Some("Cursor".to_string()),
            backend: AgentBackend::Cursor {
                workspace: test_working_dir(),
                features: vec![],
            },
            persona: None,
        };
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await;
        assert!(session.is_ok());
        assert!(session.unwrap().session_id().starts_with("cursor-"));
    }

    /// Custom backend spawns
    #[tokio::test]
    async fn test_custom_backend_spawns() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig {
            display_name: Some("Custom".to_string()),
            backend: AgentBackend::Custom {
                endpoint: "http://localhost:8080".to_string(),
                protocol: crate::types::AgentProtocol::JsonRpc,
            },
            persona: None,
        };
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await;
        assert!(session.is_ok());
        assert!(session.unwrap().session_id().starts_with("custom-"));
    }

    // -------------------------------------------------------------------------
    // Session Lifecycle Tests
    // -------------------------------------------------------------------------

    /// Session terminates correctly
    #[tokio::test]
    async fn test_session_terminate() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig::infernum("qwen-7b");
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await.unwrap();
        assert!(session.is_running());

        session.terminate().await.unwrap();
        assert!(!session.is_running());
    }

    /// Sending to terminated session fails
    #[tokio::test]
    async fn test_send_to_terminated_session_fails() {
        let factory = AgentBackendFactory::new(test_working_dir());
        let config = AgentConfig::infernum("qwen-7b");
        let context = test_context();
        let spawned_by = ParticipantId::new();

        let session = factory.spawn(&config, context, spawned_by).await.unwrap();
        session.terminate().await.unwrap();

        let message = Message {
            id: crate::types::MessageId::new(),
            channel: crate::types::ChannelType::Main,
            sender: ParticipantId::new(),
            content: crate::types::MessageContent::Text {
                content: "Hello".to_string(),
            },
            timestamp: Utc::now(),
            metadata: std::collections::HashMap::new(),
        };

        let result = session.send_message(&message).await;
        assert!(matches!(result, Err(ConclaveError::BackendTerminated { .. })));
    }

    // -------------------------------------------------------------------------
    // Turn Priority Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_turn_priority_default() {
        let priority = TurnPriority::default();
        assert_eq!(priority, TurnPriority::Normal);
    }

    // -------------------------------------------------------------------------
    // Room Context Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_room_context_serialization() {
        let context = test_context();
        let json = serde_json::to_string(&context).unwrap();
        let parsed: RoomContext = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.room_name, "Test Room");
    }
}
