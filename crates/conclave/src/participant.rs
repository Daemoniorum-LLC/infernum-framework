//! Participant management for rooms.
//!
//! Implements AGENT-COLLABORATION-SPEC.md §3.2 Participant Lifecycle.

use chrono::Utc;
use tracing::info;

use crate::backend::{AgentBackendFactory, AgentBackendSession, ParticipantSummary, RoomContext};
use crate::error::{ConclaveError, Result};
use crate::room::{RoomEvent, RoomRegistry};
use crate::types::{
    AgentConfig, AttentionState, ChannelType, LeaveReason, Participant, ParticipantId,
    ParticipantKind, RoomId, UserId,
};

impl RoomRegistry {
    // =========================================================================
    // Participant Management
    // =========================================================================

    /// Adds a human participant to a room.
    ///
    /// # Preconditions
    /// - Room exists and is not archived
    /// - User is not already in the room
    ///
    /// # Postconditions
    /// - Participant added to room
    /// - ParticipantJoined event emitted
    pub async fn join_room(
        &self,
        room_id: RoomId,
        user_id: UserId,
        display_name: String,
    ) -> Result<ParticipantId> {
        let mut rooms = self.rooms.write().await;
        let room = rooms
            .get_mut(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        if room.archived {
            return Err(ConclaveError::RoomArchived(room_id));
        }

        // Check if user already in room
        let already_in = room.participants.iter().any(|p| {
            matches!(&p.kind, ParticipantKind::Human { user_id: uid } if *uid == user_id)
        });
        if already_in {
            return Err(ConclaveError::NotAuthorized(
                "User already in room".to_string(),
            ));
        }

        // Create participant
        let participant_id = ParticipantId::new();
        let now = Utc::now();
        let participant = Participant {
            id: participant_id,
            display_name: display_name.clone(),
            kind: ParticipantKind::Human { user_id },
            persona: None,
            attention: AttentionState::Available,
            joined_at: now,
            last_active: now,
            message_count: 0,
            tool_calls: 0,
        };

        room.participants.push(participant.clone());
        room.updated_at = Utc::now();

        // Emit event
        let _ = self.event_tx.send(RoomEvent::ParticipantJoined {
            room_id,
            participant: participant.clone(),
        });

        info!(
            "Participant {} joined room {}",
            participant_id, room_id
        );

        Ok(participant_id)
    }

    /// Removes a participant from a room.
    ///
    /// # Preconditions
    /// - Room exists
    /// - Participant is in the room
    ///
    /// # Postconditions
    /// - Participant moved to alumni
    /// - ParticipantLeft event emitted
    pub async fn leave_room(
        &self,
        room_id: RoomId,
        participant_id: ParticipantId,
        reason: LeaveReason,
    ) -> Result<()> {
        let mut rooms = self.rooms.write().await;
        let room = rooms
            .get_mut(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        // Find and remove participant
        let idx = room
            .participants
            .iter()
            .position(|p| p.id == participant_id)
            .ok_or(ConclaveError::NotInRoom(participant_id, room_id))?;

        let participant = room.participants.remove(idx);
        room.alumni.push(participant);
        room.updated_at = Utc::now();

        // Emit event
        let _ = self.event_tx.send(RoomEvent::ParticipantLeft {
            room_id,
            participant_id,
            reason,
        });

        info!(
            "Participant {} left room {} ({:?})",
            participant_id, room_id, reason
        );

        Ok(())
    }

    /// Updates a participant's attention state.
    ///
    /// # Preconditions
    /// - Room exists
    /// - Participant is in the room
    ///
    /// # Postconditions
    /// - Attention state updated
    /// - AttentionChanged event emitted
    pub async fn set_attention(
        &self,
        room_id: RoomId,
        participant_id: ParticipantId,
        new_state: AttentionState,
    ) -> Result<()> {
        let mut rooms = self.rooms.write().await;
        let room = rooms
            .get_mut(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        let participant = room
            .find_participant_mut(participant_id)
            .ok_or(ConclaveError::NotInRoom(participant_id, room_id))?;

        participant.attention = new_state.clone();
        room.updated_at = Utc::now();

        // Emit event
        let _ = self.event_tx.send(RoomEvent::AttentionChanged {
            room_id,
            participant_id,
            new_state,
        });

        Ok(())
    }

    /// Gets a participant from a room.
    pub async fn get_participant(
        &self,
        room_id: RoomId,
        participant_id: ParticipantId,
    ) -> Result<Participant> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        room.find_participant(participant_id)
            .cloned()
            .ok_or(ConclaveError::NotInRoom(participant_id, room_id))
    }

    /// Lists all participants in a room.
    pub async fn list_participants(&self, room_id: RoomId) -> Result<Vec<Participant>> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        Ok(room.participants.clone())
    }

    // =========================================================================
    // Agent Spawning
    // =========================================================================

    /// Spawns an agent in a room.
    ///
    /// # Preconditions
    /// - Room exists and is not archived
    /// - spawned_by is a participant in the room
    ///
    /// # Postconditions
    /// - Agent backend session running
    /// - Agent added as participant
    /// - ParticipantJoined event emitted
    pub async fn spawn_agent(
        &self,
        room_id: RoomId,
        config: AgentConfig,
        spawned_by: ParticipantId,
    ) -> Result<ParticipantId> {
        // Get room info and validate
        let (working_dir, participants_summary, recent_messages) = {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            if room.archived {
                return Err(ConclaveError::RoomArchived(room_id));
            }

            // Verify spawner is in room
            if room.find_participant(spawned_by).is_none() {
                return Err(ConclaveError::NotInRoom(spawned_by, room_id));
            }

            // Build participant summary
            let summary: Vec<ParticipantSummary> = room
                .participants
                .iter()
                .map(|p| ParticipantSummary {
                    id: p.id,
                    display_name: p.display_name.clone(),
                    is_agent: p.kind.is_agent(),
                    attention: p.attention.clone(),
                })
                .collect();

            (room.working_dir.clone(), summary, room.name.clone())
        };

        // Get recent messages from main channel
        let messages = self.messages.read().await;
        let room_messages = messages.get(&room_id).cloned().unwrap_or_default();
        let main_messages: Vec<_> = room_messages
            .into_iter()
            .filter(|m| matches!(m.channel, ChannelType::Main))
            .take(50) // Limit context
            .collect();
        drop(messages);

        // Build room context for the agent
        let context = RoomContext {
            room_id,
            room_name: recent_messages,
            working_dir: working_dir.clone(),
            recent_messages: main_messages,
            participants: participants_summary,
            persona_prompt: config.persona.clone(), // TODO: Load from Grimoire
        };

        // Spawn backend
        let factory = AgentBackendFactory::new(working_dir);
        let session = factory
            .spawn(&config, context, spawned_by)
            .await
            .map_err(|e| ConclaveError::SpawnFailed(e.to_string()))?;

        let session_id = session.session_id().to_string();

        // Create participant
        let participant_id = ParticipantId::new();
        let now = Utc::now();
        let display_name = config
            .display_name
            .clone()
            .unwrap_or_else(|| format!("Agent-{}", &participant_id.0.to_string()[..8]));

        let participant = Participant {
            id: participant_id,
            display_name: display_name.clone(),
            kind: ParticipantKind::Agent {
                backend: config.backend.clone(),
                session_id,
                spawned_by,
            },
            persona: config.persona,
            attention: AttentionState::Available,
            joined_at: now,
            last_active: now,
            message_count: 0,
            tool_calls: 0,
        };

        // Add participant to room
        {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
                room.participants.push(participant.clone());
                room.updated_at = now;
            }
        }

        // Store session
        self.sessions.write().await.insert(participant_id, session);

        // Emit event
        let _ = self.event_tx.send(RoomEvent::ParticipantJoined {
            room_id,
            participant: participant.clone(),
        });

        info!(
            "Spawned agent {} ({}) in room {}",
            participant_id, display_name, room_id
        );

        Ok(participant_id)
    }

    /// Terminates an agent session.
    ///
    /// This terminates the backend process and moves the agent to alumni.
    pub async fn terminate_agent(
        &self,
        room_id: RoomId,
        participant_id: ParticipantId,
        reason: LeaveReason,
    ) -> Result<()> {
        // Get and terminate session
        let session = self.sessions.write().await.remove(&participant_id);
        if let Some(session) = session {
            let _ = session.terminate().await;
        }

        // Move to alumni via leave_room
        self.leave_room(room_id, participant_id, reason).await
    }

    /// Checks if an agent session exists for the given participant.
    pub async fn has_agent_session(&self, participant_id: ParticipantId) -> bool {
        self.sessions.read().await.contains_key(&participant_id)
    }

    /// Applies a function to an agent session if it exists.
    ///
    /// This is the preferred way to access agent sessions as it properly
    /// manages the lock lifetime.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let is_running = registry
    ///     .with_agent_session(agent_id, |session| session.is_running())
    ///     .await
    ///     .unwrap_or(false);
    /// ```
    pub async fn with_agent_session<F, R>(
        &self,
        participant_id: ParticipantId,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&Box<dyn AgentBackendSession>) -> R,
    {
        let sessions = self.sessions.read().await;
        sessions.get(&participant_id).map(f)
    }

    /// Applies an async function to an agent session if it exists.
    ///
    /// Use this when you need to perform async operations on the session.
    /// Note: The lock is held for the duration of the async operation.
    pub async fn with_agent_session_async<F, Fut, R>(
        &self,
        participant_id: ParticipantId,
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&Box<dyn AgentBackendSession>) -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(&participant_id) {
            Some(f(session).await)
        } else {
            None
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CreateRoomRequest;
    use std::path::PathBuf;

    fn test_working_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn test_user() -> UserId {
        UserId("test_user".to_string())
    }

    fn other_user() -> UserId {
        UserId("other_user".to_string())
    }

    // -------------------------------------------------------------------------
    // Phase 2.1: Human Participant Tests
    // -------------------------------------------------------------------------

    /// spec_human_joins_room
    #[tokio::test]
    async fn test_human_joins_room() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Join as another user
        let participant_id = registry
            .join_room(room_id, other_user(), "Other User".to_string())
            .await
            .unwrap();

        // Verify
        let room = registry.get_room(room_id).await.unwrap();
        assert_eq!(room.participants.len(), 2); // creator + new user
        assert!(room.find_participant(participant_id).is_some());
    }

    /// spec_join_announces
    #[tokio::test]
    async fn test_join_emits_event() {
        let registry = RoomRegistry::with_defaults();
        let mut rx = registry.subscribe();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Drain creation event
        let _ = rx.recv().await;

        // Join
        registry
            .join_room(room_id, other_user(), "Other User".to_string())
            .await
            .unwrap();

        // Check event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, RoomEvent::ParticipantJoined { .. }));
    }

    /// spec_human_leaves_room
    #[tokio::test]
    async fn test_human_leaves_room() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Join as another user
        let participant_id = registry
            .join_room(room_id, other_user(), "Other User".to_string())
            .await
            .unwrap();

        // Leave
        registry
            .leave_room(room_id, participant_id, LeaveReason::Left)
            .await
            .unwrap();

        // Verify
        let room = registry.get_room(room_id).await.unwrap();
        assert_eq!(room.participants.len(), 1); // only creator remains
        assert_eq!(room.alumni.len(), 1); // leaver in alumni
    }

    /// spec_leave_emits_event
    #[tokio::test]
    async fn test_leave_emits_event() {
        let registry = RoomRegistry::with_defaults();
        let mut rx = registry.subscribe();

        // Create room and join
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let participant_id = registry
            .join_room(room_id, other_user(), "Other User".to_string())
            .await
            .unwrap();

        // Drain events
        let _ = rx.recv().await; // created
        let _ = rx.recv().await; // joined

        // Leave
        registry
            .leave_room(room_id, participant_id, LeaveReason::Left)
            .await
            .unwrap();

        // Check event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, RoomEvent::ParticipantLeft { reason: LeaveReason::Left, .. }));
    }

    /// Cannot join same room twice
    #[tokio::test]
    async fn test_cannot_join_twice() {
        let registry = RoomRegistry::with_defaults();

        // Create room (creator is test_user)
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Try to join as same user
        let result = registry
            .join_room(room_id, test_user(), "Test User".to_string())
            .await;
        assert!(matches!(result, Err(ConclaveError::NotAuthorized(_))));
    }

    /// Cannot join archived room
    #[tokio::test]
    async fn test_cannot_join_archived_room() {
        let registry = RoomRegistry::with_defaults();

        // Create and archive room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;
        registry.archive_room(room_id, creator_id).await.unwrap();

        // Try to join
        let result = registry
            .join_room(room_id, other_user(), "Other User".to_string())
            .await;
        assert!(matches!(result, Err(ConclaveError::RoomArchived(_))));
    }

    // -------------------------------------------------------------------------
    // Phase 2.2: Attention Tests
    // -------------------------------------------------------------------------

    /// spec_set_attention
    #[tokio::test]
    async fn test_set_attention() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant_id = room.participants[0].id;

        // Set attention to focused
        let new_state = AttentionState::Focused {
            task: "Working on fix".to_string(),
            started: Utc::now(),
            eta: None,
            interruptible: true,
        };
        registry
            .set_attention(room_id, participant_id, new_state)
            .await
            .unwrap();

        // Verify
        let participant = registry
            .get_participant(room_id, participant_id)
            .await
            .unwrap();
        assert!(matches!(participant.attention, AttentionState::Focused { .. }));
    }

    /// spec_attention_changed_event
    #[tokio::test]
    async fn test_attention_changed_event() {
        let registry = RoomRegistry::with_defaults();
        let mut rx = registry.subscribe();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant_id = room.participants[0].id;

        // Drain creation event
        let _ = rx.recv().await;

        // Set attention
        registry
            .set_attention(room_id, participant_id, AttentionState::DoNotDisturb {
                reason: "Testing".to_string(),
                until: None,
            })
            .await
            .unwrap();

        // Check event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, RoomEvent::AttentionChanged { .. }));
    }

    /// Cannot set attention for non-participant
    #[tokio::test]
    async fn test_cannot_set_attention_non_participant() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Try to set attention for random participant
        let fake_id = ParticipantId::new();
        let result = registry
            .set_attention(room_id, fake_id, AttentionState::Available)
            .await;
        assert!(matches!(result, Err(ConclaveError::NotInRoom(_, _))));
    }

    // -------------------------------------------------------------------------
    // Phase 4: Agent Spawning Tests (TDD Phase 6 - Backend Integration)
    // -------------------------------------------------------------------------

    /// spec_agent_spawn_backend
    #[tokio::test]
    async fn test_spawn_agent_succeeds() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::claude_code(crate::types::ClaudeTier::Opus);
        let agent_id = registry
            .spawn_agent(room_id, config, creator_id)
            .await
            .unwrap();

        // Verify agent is in room
        let room = registry.get_room(room_id).await.unwrap();
        assert_eq!(room.participants.len(), 2); // creator + agent
        let agent = room.find_participant(agent_id).unwrap();
        assert!(agent.kind.is_agent());
    }

    /// spec_agent_spawn_emits_event
    #[tokio::test]
    async fn test_spawn_agent_emits_event() {
        let registry = RoomRegistry::with_defaults();
        let mut rx = registry.subscribe();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        // Drain creation event
        let _ = rx.recv().await;

        // Spawn agent
        let config = AgentConfig::infernum("qwen-7b");
        registry
            .spawn_agent(room_id, config, creator_id)
            .await
            .unwrap();

        // Check event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, RoomEvent::ParticipantJoined { participant, .. } if participant.kind.is_agent()));
    }

    /// Cannot spawn agent in archived room
    #[tokio::test]
    async fn test_cannot_spawn_in_archived_room() {
        let registry = RoomRegistry::with_defaults();

        // Create and archive room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;
        registry.archive_room(room_id, creator_id).await.unwrap();

        // Try to spawn agent
        let config = AgentConfig::claude_code(crate::types::ClaudeTier::Opus);
        let result = registry.spawn_agent(room_id, config, creator_id).await;
        assert!(matches!(result, Err(ConclaveError::RoomArchived(_))));
    }

    /// Cannot spawn agent if not in room
    #[tokio::test]
    async fn test_cannot_spawn_if_not_in_room() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Try to spawn as non-participant
        let fake_id = ParticipantId::new();
        let config = AgentConfig::infernum("qwen-7b");
        let result = registry.spawn_agent(room_id, config, fake_id).await;
        assert!(matches!(result, Err(ConclaveError::NotInRoom(_, _))));
    }

    /// Spawned agent has correct session
    #[tokio::test]
    async fn test_spawned_agent_has_session() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::claude_code(crate::types::ClaudeTier::Opus);
        let agent_id = registry
            .spawn_agent(room_id, config, creator_id)
            .await
            .unwrap();

        // Verify session exists
        let sessions = registry.sessions.read().await;
        assert!(sessions.contains_key(&agent_id));
        let session = sessions.get(&agent_id).unwrap();
        assert!(session.is_running());
        assert!(session.session_id().starts_with("claude-"));
    }

    /// spec_agent_termination
    #[tokio::test]
    async fn test_terminate_agent() {
        let registry = RoomRegistry::with_defaults();

        // Create room and spawn agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        let config = AgentConfig::infernum("qwen-7b");
        let agent_id = registry
            .spawn_agent(room_id, config, creator_id)
            .await
            .unwrap();

        // Terminate agent
        registry
            .terminate_agent(room_id, agent_id, LeaveReason::Terminated)
            .await
            .unwrap();

        // Verify agent is in alumni
        let room = registry.get_room(room_id).await.unwrap();
        assert_eq!(room.participants.len(), 1); // only creator
        assert_eq!(room.alumni.len(), 1);

        // Verify session removed
        let sessions = registry.sessions.read().await;
        assert!(!sessions.contains_key(&agent_id));
    }

    /// Multiple agents can be spawned in a room
    #[tokio::test]
    async fn test_spawn_multiple_agents() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator_id = room.participants[0].id;

        // Spawn multiple agents
        let config1 = AgentConfig::claude_code(crate::types::ClaudeTier::Opus);
        let config2 = AgentConfig::infernum("qwen-7b");

        registry.spawn_agent(room_id, config1, creator_id).await.unwrap();
        registry.spawn_agent(room_id, config2, creator_id).await.unwrap();

        // Verify
        let room = registry.get_room(room_id).await.unwrap();
        assert_eq!(room.participants.len(), 3); // creator + 2 agents
        assert_eq!(room.agent_count(), 2);
    }
}
