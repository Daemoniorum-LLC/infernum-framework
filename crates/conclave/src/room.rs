//! Room lifecycle management.
//!
//! Implements AGENT-COLLABORATION-SPEC.md §3.1 Room Lifecycle.

use std::collections::HashMap;
use std::path::Path;

use chrono::Utc;
use tokio::sync::{broadcast, RwLock};
use tracing::info;

use crate::error::{ConclaveError, Result};
use crate::types::*;

// =============================================================================
// Room Events
// =============================================================================

/// Events emitted by the room system.
#[derive(Debug, Clone)]
pub enum RoomEvent {
    /// Room was created.
    RoomCreated {
        room_id: RoomId,
        name: String,
        creator: ParticipantId,
    },
    /// Participant joined.
    ParticipantJoined {
        room_id: RoomId,
        participant: Participant,
    },
    /// Participant left.
    ParticipantLeft {
        room_id: RoomId,
        participant_id: ParticipantId,
        reason: LeaveReason,
    },
    /// Message sent.
    MessageSent {
        room_id: RoomId,
        message: Message,
    },
    /// Attention changed.
    AttentionChanged {
        room_id: RoomId,
        participant_id: ParticipantId,
        new_state: AttentionState,
    },
    /// Room archived.
    RoomArchived {
        room_id: RoomId,
        archived_by: ParticipantId,
        message_count: u64,
    },
    /// Room forked.
    RoomForked {
        source_room_id: RoomId,
        new_room_id: RoomId,
        forked_by: ParticipantId,
    },
}

// =============================================================================
// Room Registry
// =============================================================================

/// Configuration for the room registry.
#[derive(Debug, Clone)]
pub struct RoomRegistryConfig {
    /// Maximum concurrent rooms.
    pub max_rooms: usize,
    /// Default turn timeout in seconds.
    pub default_turn_timeout_secs: u32,
}

impl Default for RoomRegistryConfig {
    fn default() -> Self {
        Self {
            max_rooms: 100,
            default_turn_timeout_secs: 300,
        }
    }
}

/// Registry managing all active rooms.
pub struct RoomRegistry {
    /// Configuration.
    pub(crate) config: RoomRegistryConfig,
    /// Active rooms.
    pub(crate) rooms: RwLock<HashMap<RoomId, Room>>,
    /// Message history per room.
    pub(crate) messages: RwLock<HashMap<RoomId, Vec<Message>>>,
    /// Active agent sessions (participant_id -> session).
    pub(crate) sessions: RwLock<HashMap<ParticipantId, Box<dyn crate::backend::AgentBackendSession>>>,
    /// Turn coordinator state per room.
    pub(crate) coordinator_states: RwLock<HashMap<RoomId, crate::coordinator::CoordinatorState>>,
    /// Event broadcast channel.
    pub(crate) event_tx: broadcast::Sender<RoomEvent>,
}

impl RoomRegistry {
    /// Creates a new room registry.
    pub fn new(config: RoomRegistryConfig) -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        Self {
            config,
            rooms: RwLock::new(HashMap::new()),
            messages: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            coordinator_states: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Creates a new room with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RoomRegistryConfig::default())
    }

    /// Subscribes to room events.
    pub fn subscribe(&self) -> broadcast::Receiver<RoomEvent> {
        self.event_tx.subscribe()
    }

    // =========================================================================
    // Room Lifecycle
    // =========================================================================

    /// Creates a new room.
    ///
    /// # Preconditions
    /// - `request.working_dir` must exist
    /// - Active rooms < max_rooms
    ///
    /// # Postconditions
    /// - Room exists in registry with `archived = false`
    /// - Creator is added as participant
    /// - RoomCreated event emitted
    pub async fn create_room(&self, request: CreateRoomRequest) -> Result<RoomId> {
        // Validate working directory
        if !Path::new(&request.working_dir).exists() {
            return Err(ConclaveError::WorkingDirNotFound(request.working_dir));
        }

        // Check room limit
        let rooms = self.rooms.read().await;
        if rooms.len() >= self.config.max_rooms {
            return Err(ConclaveError::MaxRoomsReached {
                max: self.config.max_rooms,
            });
        }
        drop(rooms);

        // Generate room ID
        let room_id = RoomId::new();
        let now = Utc::now();

        // Create creator participant
        let creator_id = ParticipantId::new();
        let creator = Participant {
            id: creator_id,
            display_name: request.creator.0.clone(),
            kind: ParticipantKind::Human {
                user_id: request.creator.clone(),
            },
            persona: None,
            attention: AttentionState::Available,
            joined_at: now,
            last_active: now,
            message_count: 0,
            tool_calls: 0,
        };

        // Create room
        let room = Room {
            id: room_id,
            name: request.name.clone(),
            labels: request.labels,
            working_dir: request.working_dir,
            project: request.project,
            participants: vec![creator.clone()],
            alumni: Vec::new(),
            invite_policy: request.invite_policy.unwrap_or_default(),
            coordinator_config: request.coordinator_config.unwrap_or_default(),
            archived: false,
            fork_of: None,
            created_at: now,
            updated_at: now,
        };

        // Insert room
        self.rooms.write().await.insert(room_id, room);
        self.messages.write().await.insert(room_id, Vec::new());

        // Emit event
        let _ = self.event_tx.send(RoomEvent::RoomCreated {
            room_id,
            name: request.name,
            creator: creator_id,
        });

        info!("Created room {} with creator {}", room_id, creator_id);

        // TODO: Spawn initial agents (Phase 2)
        // for agent_config in request.initial_agents {
        //     self.spawn_agent(room_id, agent_config, creator_id).await?;
        // }

        Ok(room_id)
    }

    /// Archives a room.
    ///
    /// # Preconditions
    /// - Room exists and is not already archived
    /// - Requester is in the room
    ///
    /// # Postconditions
    /// - Room is archived
    /// - All agents terminated
    /// - RoomArchived event emitted
    pub async fn archive_room(
        &self,
        room_id: RoomId,
        archived_by: ParticipantId,
    ) -> Result<()> {
        let mut rooms = self.rooms.write().await;
        let room = rooms
            .get_mut(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        if room.archived {
            return Err(ConclaveError::RoomArchived(room_id));
        }

        // Verify requester is in room
        if room.find_participant(archived_by).is_none() {
            return Err(ConclaveError::NotInRoom(archived_by, room_id));
        }

        // Get message count
        let messages = self.messages.read().await;
        let message_count = messages.get(&room_id).map(|m| m.len() as u64).unwrap_or(0);
        drop(messages);

        // Move all agents to alumni
        let agents: Vec<_> = room
            .participants
            .iter()
            .filter(|p| p.kind.is_agent())
            .cloned()
            .collect();

        for agent in &agents {
            room.alumni.push(agent.clone());
        }
        room.participants.retain(|p| p.kind.is_human());

        // Mark as archived
        room.archived = true;
        room.updated_at = Utc::now();

        // Emit event
        let _ = self.event_tx.send(RoomEvent::RoomArchived {
            room_id,
            archived_by,
            message_count,
        });

        info!("Archived room {} by {}", room_id, archived_by);

        // TODO: Actually terminate agent backends (Phase 2)

        Ok(())
    }

    /// Forks a room, creating a new room with context from the original.
    ///
    /// # Preconditions
    /// - Source room exists
    /// - Requester is in the source room
    ///
    /// # Postconditions
    /// - New room created with fork_of = source_room_id
    /// - Forker is added as participant
    /// - RoomForked event emitted
    pub async fn fork_room(
        &self,
        source_room_id: RoomId,
        new_name: String,
        forked_by: ParticipantId,
        forker_user_id: UserId,
    ) -> Result<RoomId> {
        let rooms = self.rooms.read().await;
        let source = rooms
            .get(&source_room_id)
            .ok_or(ConclaveError::RoomNotFound(source_room_id))?;

        // Verify forker is in source room
        if source.find_participant(forked_by).is_none() {
            return Err(ConclaveError::NotInRoom(forked_by, source_room_id));
        }

        // Check room limit
        if rooms.len() >= self.config.max_rooms {
            return Err(ConclaveError::MaxRoomsReached {
                max: self.config.max_rooms,
            });
        }

        // Copy relevant data from source
        let working_dir = source.working_dir.clone();
        let project = source.project.clone();
        let labels = source.labels.clone();
        let invite_policy = source.invite_policy;
        let coordinator_config = source.coordinator_config.clone();
        drop(rooms);

        // Create new room
        let new_room_id = RoomId::new();
        let now = Utc::now();

        let forker = Participant {
            id: forked_by,
            display_name: forker_user_id.0.clone(),
            kind: ParticipantKind::Human {
                user_id: forker_user_id,
            },
            persona: None,
            attention: AttentionState::Available,
            joined_at: now,
            last_active: now,
            message_count: 0,
            tool_calls: 0,
        };

        let new_room = Room {
            id: new_room_id,
            name: new_name,
            labels,
            working_dir,
            project,
            participants: vec![forker],
            alumni: Vec::new(),
            invite_policy,
            coordinator_config,
            archived: false,
            fork_of: Some(source_room_id),
            created_at: now,
            updated_at: now,
        };

        // Insert new room
        self.rooms.write().await.insert(new_room_id, new_room);
        self.messages.write().await.insert(new_room_id, Vec::new());

        // Emit event
        let _ = self.event_tx.send(RoomEvent::RoomForked {
            source_room_id,
            new_room_id,
            forked_by,
        });

        info!(
            "Forked room {} to {} by {}",
            source_room_id, new_room_id, forked_by
        );

        Ok(new_room_id)
    }

    // =========================================================================
    // Room Queries
    // =========================================================================

    /// Gets a room by ID.
    pub async fn get_room(&self, room_id: RoomId) -> Option<Room> {
        self.rooms.read().await.get(&room_id).cloned()
    }

    /// Lists all active (non-archived) rooms.
    pub async fn list_active_rooms(&self) -> Vec<Room> {
        self.rooms
            .read()
            .await
            .values()
            .filter(|r| !r.archived)
            .cloned()
            .collect()
    }

    /// Lists rooms a participant is in.
    pub async fn list_rooms_for_participant(&self, participant_id: ParticipantId) -> Vec<Room> {
        self.rooms
            .read()
            .await
            .values()
            .filter(|r| r.find_participant(participant_id).is_some())
            .cloned()
            .collect()
    }

    /// Gets message history for a room.
    pub async fn get_messages(&self, room_id: RoomId) -> Result<Vec<Message>> {
        let rooms = self.rooms.read().await;
        if !rooms.contains_key(&room_id) {
            return Err(ConclaveError::RoomNotFound(room_id));
        }
        drop(rooms);

        let messages = self.messages.read().await;
        Ok(messages.get(&room_id).cloned().unwrap_or_default())
    }

    /// Returns the number of active rooms.
    pub async fn room_count(&self) -> usize {
        self.rooms.read().await.len()
    }

}

impl Default for RoomRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokio;

    fn test_working_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn test_user() -> UserId {
        UserId("test_user".to_string())
    }

    // -------------------------------------------------------------------------
    // Phase 1.1: Room Lifecycle Tests
    // -------------------------------------------------------------------------

    /// spec_create_room_succeeds
    #[tokio::test]
    async fn test_create_room_succeeds() {
        let registry = RoomRegistry::with_defaults();

        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );

        let result = registry.create_room(request).await;
        assert!(result.is_ok());

        let room_id = result.unwrap();
        let room = registry.get_room(room_id).await;
        assert!(room.is_some());

        let room = room.unwrap();
        assert_eq!(room.name, "Test Room");
        assert!(!room.archived);
        assert_eq!(room.participants.len(), 1); // creator
    }

    /// spec_create_room_fails_missing_dir
    #[tokio::test]
    async fn test_create_room_fails_missing_dir() {
        let registry = RoomRegistry::with_defaults();

        let request = CreateRoomRequest::new(
            "Test Room",
            PathBuf::from("/nonexistent/path/that/does/not/exist"),
            test_user(),
        );

        let result = registry.create_room(request).await;
        assert!(matches!(result, Err(ConclaveError::WorkingDirNotFound(_))));
    }

    /// spec_create_room_spawns_agents (placeholder - agents not implemented yet)
    #[tokio::test]
    async fn test_create_room_with_initial_agents_placeholder() {
        let registry = RoomRegistry::with_defaults();

        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        )
        .with_agent(AgentConfig::claude_code(ClaudeTier::Opus))
        .with_agent(AgentConfig::infernum("qwen-7b"));

        let result = registry.create_room(request).await;
        assert!(result.is_ok());

        // TODO: When agent spawning is implemented, verify:
        // assert_eq!(room.participants.len(), 3); // creator + 2 agents
    }

    /// spec_archive_terminates_agents
    #[tokio::test]
    async fn test_archive_room() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        let room_id = registry.create_room(request).await.unwrap();

        // Get creator participant ID
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        // Archive room
        let result = registry.archive_room(room_id, creator_id).await;
        assert!(result.is_ok());

        // Verify archived
        let room = registry.get_room(room_id).await.unwrap();
        assert!(room.archived);
    }

    /// spec_archived_room_immutable
    #[tokio::test]
    async fn test_archived_room_cannot_be_archived_again() {
        let registry = RoomRegistry::with_defaults();

        // Create and archive room
        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        registry.archive_room(room_id, creator_id).await.unwrap();

        // Try to archive again
        let result = registry.archive_room(room_id, creator_id).await;
        assert!(matches!(result, Err(ConclaveError::RoomArchived(_))));
    }

    /// spec_fork_room_creates_linked_room
    #[tokio::test]
    async fn test_fork_room() {
        let registry = RoomRegistry::with_defaults();

        // Create source room
        let request = CreateRoomRequest::new(
            "Source Room",
            test_working_dir(),
            test_user(),
        );
        let source_id = registry.create_room(request).await.unwrap();
        let source = registry.get_room(source_id).await.unwrap();
        let creator_id = source.participants[0].id;

        // Fork room
        let new_id = registry
            .fork_room(source_id, "Forked Room".to_string(), creator_id, test_user())
            .await
            .unwrap();

        // Verify fork
        let forked = registry.get_room(new_id).await.unwrap();
        assert_eq!(forked.name, "Forked Room");
        assert_eq!(forked.fork_of, Some(source_id));
        assert_eq!(forked.working_dir, source.working_dir);
    }

    /// property_room_ids_unique
    #[tokio::test]
    async fn test_room_ids_unique() {
        let registry = RoomRegistry::with_defaults();

        let mut ids = Vec::new();
        for i in 0..10 {
            let request = CreateRoomRequest::new(
                format!("Room {}", i),
                test_working_dir(),
                test_user(),
            );
            let id = registry.create_room(request).await.unwrap();
            assert!(!ids.contains(&id), "Duplicate room ID found");
            ids.push(id);
        }
    }

    // -------------------------------------------------------------------------
    // Phase 1.2: Room Types Tests
    // -------------------------------------------------------------------------

    /// spec_room_id_is_uuid
    #[test]
    fn test_room_id_is_uuid() {
        let id = RoomId::new();
        // Verify it's a valid UUID v4 by checking format
        let uuid = id.0;
        assert_eq!(uuid.get_version_num(), 4);
    }

    /// spec_room_project_ref
    #[tokio::test]
    async fn test_room_preserves_project_ref() {
        let registry = RoomRegistry::with_defaults();

        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        )
        .with_project(ProjectRef::GitRepo {
            remote: "https://github.com/test/repo".to_string(),
            branch: "main".to_string(),
        });

        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();

        assert!(matches!(
            room.project,
            Some(ProjectRef::GitRepo { ref remote, .. }) if remote == "https://github.com/test/repo"
        ));
    }

    /// spec_default_policies
    #[tokio::test]
    async fn test_default_policies_applied() {
        let registry = RoomRegistry::with_defaults();

        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        // No explicit policies set

        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();

        assert_eq!(room.invite_policy, InvitePolicy::Announce);
        assert_eq!(room.coordinator_config.mode, CoordinatorMode::Volunteer);
    }

    // -------------------------------------------------------------------------
    // Query Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_active_rooms() {
        let registry = RoomRegistry::with_defaults();

        // Create 3 rooms
        for i in 0..3 {
            let request = CreateRoomRequest::new(
                format!("Room {}", i),
                test_working_dir(),
                test_user(),
            );
            registry.create_room(request).await.unwrap();
        }

        let active = registry.list_active_rooms().await;
        assert_eq!(active.len(), 3);
    }

    #[tokio::test]
    async fn test_list_active_rooms_excludes_archived() {
        let registry = RoomRegistry::with_defaults();

        // Create 2 rooms
        let request1 = CreateRoomRequest::new("Room 1", test_working_dir(), test_user());
        let id1 = registry.create_room(request1).await.unwrap();

        let request2 = CreateRoomRequest::new("Room 2", test_working_dir(), test_user());
        registry.create_room(request2).await.unwrap();

        // Archive first room
        let room = registry.get_room(id1).await.unwrap();
        let creator_id = room.participants[0].id;
        registry.archive_room(id1, creator_id).await.unwrap();

        let active = registry.list_active_rooms().await;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "Room 2");
    }

    #[tokio::test]
    async fn test_room_count() {
        let registry = RoomRegistry::with_defaults();
        assert_eq!(registry.room_count().await, 0);

        let request = CreateRoomRequest::new("Room", test_working_dir(), test_user());
        registry.create_room(request).await.unwrap();
        assert_eq!(registry.room_count().await, 1);
    }

    // -------------------------------------------------------------------------
    // Max Rooms Test
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_max_rooms_enforced() {
        let config = RoomRegistryConfig {
            max_rooms: 2,
            ..Default::default()
        };
        let registry = RoomRegistry::new(config);

        // Create 2 rooms (at limit)
        for i in 0..2 {
            let request = CreateRoomRequest::new(
                format!("Room {}", i),
                test_working_dir(),
                test_user(),
            );
            registry.create_room(request).await.unwrap();
        }

        // Try to create a 3rd room
        let request = CreateRoomRequest::new("Room 3", test_working_dir(), test_user());
        let result = registry.create_room(request).await;
        assert!(matches!(result, Err(ConclaveError::MaxRoomsReached { max: 2 })));
    }
}
