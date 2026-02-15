//! Conclave: Room-based agent collaboration for Infernum.
//!
//! This crate implements the Agent Collaboration specification, providing:
//! - **Rooms**: Persistent collaborative spaces where humans and agents work together
//! - **Participants**: Unified abstraction for human and agent participants
//! - **Channels**: Main (coordinated), DMs (free-form), AgentReasoning, Threads
//! - **Attention**: Agent-controlled focus states with human override
//! - **Coordination**: Turn-taking for the main channel to prevent cacophony
//! - **Backends**: Pluggable agent backends (Infernum, Claude Code, Codex, Cursor, Custom)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                           ROOM                                   │
//! │  ┌─────────────────────────────────────────────────────────────┐│
//! │  │ Main Channel (coordinated)                                  ││
//! │  │   [Human] [Claude] [Infernum] → turn-based messages         ││
//! │  └─────────────────────────────────────────────────────────────┘│
//! │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐│
//! │  │ @claude DM   │ │ @inf DM      │ │ AgentReasoning channels  ││
//! │  │ (free-form)  │ │ (free-form)  │ │ (private per agent)      ││
//! │  └──────────────┘ └──────────────┘ └──────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use conclave::{Room, CreateRoomRequest, AgentConfig, AgentBackend};
//!
//! // Create a room with initial agents
//! let request = CreateRoomRequest::new("Fix auth bug")
//!     .with_working_dir("/home/user/project")
//!     .with_agent(AgentConfig::claude_code(ClaudeTier::Opus))
//!     .with_agent(AgentConfig::infernum("qwen-7b"));
//!
//! let room = room_service.create_room(request).await?;
//!
//! // Send a message to the room
//! room_service.send_message(room.id, "What's the bug?").await?;
//! ```

pub mod attention;
pub mod backend;
pub mod channel;
pub mod coordinator;
pub mod error;
pub mod event_loop;
pub mod participant;
pub mod persistence;
pub mod recovery;
pub mod room;
pub mod routing;
pub mod types;

pub use attention::{start_decay_timer, AttentionConfig, AttentionDecayHandle, AttentionManager};
pub use backend::{AgentBackendFactory, AgentBackendSession, AgentEvent, RoomContext, TurnPriority};
pub use coordinator::{CoordinatorState, TurnPosition, TurnRequest};
pub use error::{ConclaveError, Result};
pub use event_loop::{start_event_loop, EventLoopHandle, SessionEventProcessor};
pub use persistence::{PersistenceConfig, PersistenceStore, RegistryIndex, RoomSnapshot};
pub use recovery::{DisconnectedAgent, RecoveryStatus};
pub use room::{RoomEvent, RoomRegistry, RoomRegistryConfig};
pub use types::*;
