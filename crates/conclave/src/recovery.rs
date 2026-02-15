//! Session recovery for agent backends after restart.
//!
//! When the system restarts, room state is restored from persistence
//! but agent backend sessions (processes) are NOT automatically restored.
//! This module provides tools for detecting and re-establishing sessions.
//!
//! # Recovery Strategies
//!
//! 1. **Manual Recovery**: User explicitly re-spawns agents they want
//! 2. **Auto Recovery**: System automatically re-spawns agents on room load
//! 3. **Lazy Recovery**: Re-spawn agents on first interaction
//!
//! # Example
//!
//! ```ignore
//! // Load registry from persistence
//! let registry = RoomRegistry::with_persistence(store).await?;
//!
//! // Check for disconnected agents
//! let disconnected = registry.get_disconnected_agents(room_id).await?;
//!
//! // Manually recover a specific agent
//! registry.recover_agent(room_id, agent_id, human_id).await?;
//!
//! // Or recover all agents in a room
//! registry.recover_all_agents(room_id, human_id).await?;
//! ```

use tracing::{info, warn};

use crate::backend::AgentBackendFactory;
use crate::error::{ConclaveError, Result};
use crate::room::RoomRegistry;
use crate::types::{ChannelType, ParticipantId, ParticipantKind, RoomId};

// =============================================================================
// Recovery Status
// =============================================================================

/// Status of an agent's session after recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStatus {
    /// Session is active and running.
    Connected,
    /// Session needs to be re-established.
    Disconnected,
    /// Session was terminated (agent left room).
    Terminated,
    /// Recovery failed.
    Failed(String),
}

/// Information about a disconnected agent.
#[derive(Debug, Clone)]
pub struct DisconnectedAgent {
    /// The agent's participant ID.
    pub agent_id: ParticipantId,
    /// The agent's display name.
    pub display_name: String,
    /// Who originally spawned this agent.
    pub spawned_by: ParticipantId,
    /// The backend type (for display).
    pub backend_type: String,
}

// =============================================================================
// RoomRegistry Recovery Extensions
// =============================================================================

impl RoomRegistry {
    /// Returns a list of agents in a room that need session recovery.
    ///
    /// After loading from persistence, agent participants exist in the room
    /// but their backend sessions are not running. This method identifies
    /// which agents need their sessions re-established.
    pub async fn get_disconnected_agents(
        &self,
        room_id: RoomId,
    ) -> Result<Vec<DisconnectedAgent>> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        let sessions = self.sessions.read().await;
        let mut disconnected = Vec::new();

        for participant in &room.participants {
            if let ParticipantKind::Agent {
                backend,
                spawned_by,
                ..
            } = &participant.kind
            {
                // Check if session exists
                if !sessions.contains_key(&participant.id) {
                    let backend_type = match backend {
                        crate::types::AgentBackend::Infernum { .. } => "Infernum",
                        crate::types::AgentBackend::ClaudeCode { .. } => "ClaudeCode",
                        crate::types::AgentBackend::Codex { .. } => "Codex",
                        crate::types::AgentBackend::Cursor { .. } => "Cursor",
                        crate::types::AgentBackend::Custom { .. } => "Custom",
                    };
                    disconnected.push(DisconnectedAgent {
                        agent_id: participant.id,
                        display_name: participant.display_name.clone(),
                        spawned_by: *spawned_by,
                        backend_type: backend_type.to_string(),
                    });
                }
            }
        }

        Ok(disconnected)
    }

    /// Checks if an agent has an active session.
    pub async fn agent_is_connected(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
    ) -> Result<bool> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        // Verify agent is in room
        let participant = room
            .find_participant(agent_id)
            .ok_or(ConclaveError::NotInRoom(agent_id, room_id))?;

        // Verify it's an agent
        if !participant.kind.is_agent() {
            return Err(ConclaveError::NotAuthorized(
                "Participant is not an agent".to_string(),
            ));
        }

        // Check session
        let sessions = self.sessions.read().await;
        Ok(sessions.contains_key(&agent_id))
    }

    /// Re-establishes a session for a disconnected agent.
    ///
    /// This re-spawns the backend process and reconnects it to the room.
    /// The agent keeps its existing participant ID and history.
    pub async fn recover_agent(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        recovered_by: ParticipantId,
    ) -> Result<()> {
        // Get room info
        let (working_dir, backend, room_name) = {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            // Verify recoverer is in room
            if room.find_participant(recovered_by).is_none() {
                return Err(ConclaveError::NotInRoom(recovered_by, room_id));
            }

            // Get agent participant
            let participant = room
                .find_participant(agent_id)
                .ok_or(ConclaveError::NotInRoom(agent_id, room_id))?;

            // Get backend config
            let backend = match &participant.kind {
                ParticipantKind::Agent { backend, .. } => backend.clone(),
                ParticipantKind::Human { .. } => {
                    return Err(ConclaveError::NotAuthorized(
                        "Cannot recover a human participant".to_string(),
                    ));
                }
            };

            (
                room.working_dir.clone(),
                backend,
                room.name.clone(),
            )
        };

        // Check if already connected
        {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(&agent_id) {
                warn!(
                    "Agent {} already has active session, skipping recovery",
                    agent_id
                );
                return Ok(());
            }
        }

        // Build participant summary
        let participants_summary = {
            let rooms = self.rooms.read().await;
            let room = rooms.get(&room_id).unwrap();
            room.participants
                .iter()
                .map(|p| crate::backend::ParticipantSummary {
                    id: p.id,
                    display_name: p.display_name.clone(),
                    is_agent: p.kind.is_agent(),
                    attention: p.attention.clone(),
                })
                .collect()
        };

        // Get recent messages from main channel
        let messages = self.messages.read().await;
        let room_messages = messages.get(&room_id).cloned().unwrap_or_default();
        let main_messages: Vec<_> = room_messages
            .into_iter()
            .filter(|m| matches!(m.channel, ChannelType::Main))
            .take(50)
            .collect();
        drop(messages);

        // Build context
        let context = crate::backend::RoomContext {
            room_id,
            room_name,
            working_dir: working_dir.clone(),
            recent_messages: main_messages,
            participants: participants_summary,
            persona_prompt: None,
        };

        // Create config from backend
        let config = crate::types::AgentConfig {
            display_name: None,
            backend: backend.clone(),
            persona: None,
        };

        // Spawn new session
        let factory = AgentBackendFactory::new(working_dir);
        let session = factory
            .spawn(&config, context, recovered_by)
            .await
            .map_err(|e| ConclaveError::SpawnFailed(e.to_string()))?;

        let session_id = session.session_id().to_string();

        // Update participant with new session ID
        {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
                if let Some(participant) = room.find_participant_mut(agent_id) {
                    if let ParticipantKind::Agent {
                        session_id: ref mut sid,
                        ..
                    } = participant.kind
                    {
                        *sid = session_id.clone();
                    }
                }
            }
        }

        // Store session
        self.sessions.write().await.insert(agent_id, session);

        info!(
            "Recovered agent {} with session {} in room {}",
            agent_id, session_id, room_id
        );

        Ok(())
    }

    /// Recovers all disconnected agents in a room.
    ///
    /// Returns the number of agents successfully recovered.
    pub async fn recover_all_agents(
        &self,
        room_id: RoomId,
        recovered_by: ParticipantId,
    ) -> Result<u32> {
        let disconnected = self.get_disconnected_agents(room_id).await?;
        let mut recovered = 0;

        for agent in disconnected {
            match self.recover_agent(room_id, agent.agent_id, recovered_by).await {
                Ok(()) => {
                    recovered += 1;
                }
                Err(e) => {
                    warn!(
                        "Failed to recover agent {}: {}",
                        agent.agent_id, e
                    );
                }
            }
        }

        info!(
            "Recovered {}/{} agents in room {}",
            recovered,
            self.get_disconnected_agents(room_id).await?.len() + recovered as usize,
            room_id
        );

        Ok(recovered)
    }

    /// Returns the recovery status of all agents in a room.
    pub async fn get_agent_recovery_status(
        &self,
        room_id: RoomId,
    ) -> Result<Vec<(ParticipantId, RecoveryStatus)>> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        let sessions = self.sessions.read().await;
        let mut statuses = Vec::new();

        for participant in &room.participants {
            if participant.kind.is_agent() {
                let status = if sessions.contains_key(&participant.id) {
                    RecoveryStatus::Connected
                } else {
                    RecoveryStatus::Disconnected
                };
                statuses.push((participant.id, status));
            }
        }

        // Check alumni for terminated
        for alumnus in &room.alumni {
            if alumnus.kind.is_agent() {
                statuses.push((alumnus.id, RecoveryStatus::Terminated));
            }
        }

        Ok(statuses)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentConfig, CreateRoomRequest, UserId};
    use std::path::PathBuf;

    fn test_working_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn test_user() -> UserId {
        UserId("test_user".to_string())
    }

    #[tokio::test]
    async fn test_get_disconnected_agents_empty() {
        let registry = RoomRegistry::with_defaults();

        // Create room without agents
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        let disconnected = registry.get_disconnected_agents(room_id).await.unwrap();
        assert!(disconnected.is_empty());
    }

    #[tokio::test]
    async fn test_agent_is_connected_after_spawn() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Should be connected
        let connected = registry.agent_is_connected(room_id, agent_id).await.unwrap();
        assert!(connected);

        // No disconnected agents
        let disconnected = registry.get_disconnected_agents(room_id).await.unwrap();
        assert!(disconnected.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_status() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Check status
        let statuses = registry.get_agent_recovery_status(room_id).await.unwrap();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].0, agent_id);
        assert_eq!(statuses[0].1, RecoveryStatus::Connected);
    }

    #[tokio::test]
    async fn test_recovery_status_after_termination() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn and terminate agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();
        registry
            .terminate_agent(room_id, agent_id, crate::types::LeaveReason::Terminated)
            .await
            .unwrap();

        // Check status
        let statuses = registry.get_agent_recovery_status(room_id).await.unwrap();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].0, agent_id);
        assert_eq!(statuses[0].1, RecoveryStatus::Terminated);
    }

    #[tokio::test]
    async fn test_simulate_disconnected_agent() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Verify connected
        assert!(registry.agent_is_connected(room_id, agent_id).await.unwrap());

        // Simulate disconnect by removing session (like after restart)
        registry.sessions.write().await.remove(&agent_id);

        // Now should be disconnected
        assert!(!registry.agent_is_connected(room_id, agent_id).await.unwrap());

        let disconnected = registry.get_disconnected_agents(room_id).await.unwrap();
        assert_eq!(disconnected.len(), 1);
        assert_eq!(disconnected[0].agent_id, agent_id);
    }

    #[tokio::test]
    async fn test_recover_disconnected_agent() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Simulate disconnect
        registry.sessions.write().await.remove(&agent_id);
        assert!(!registry.agent_is_connected(room_id, agent_id).await.unwrap());

        // Recover
        registry.recover_agent(room_id, agent_id, human_id).await.unwrap();

        // Should be connected again
        assert!(registry.agent_is_connected(room_id, agent_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_recover_all_agents() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn multiple agents
        let config1 = AgentConfig::infernum("test1");
        let config2 = AgentConfig::infernum("test2");
        let agent1 = registry.spawn_agent(room_id, config1, human_id).await.unwrap();
        let agent2 = registry.spawn_agent(room_id, config2, human_id).await.unwrap();

        // Simulate disconnect
        registry.sessions.write().await.remove(&agent1);
        registry.sessions.write().await.remove(&agent2);

        // Recover all
        let recovered = registry.recover_all_agents(room_id, human_id).await.unwrap();
        assert_eq!(recovered, 2);

        // Both should be connected
        assert!(registry.agent_is_connected(room_id, agent1).await.unwrap());
        assert!(registry.agent_is_connected(room_id, agent2).await.unwrap());
    }

    #[tokio::test]
    async fn test_recover_already_connected_no_op() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Try to recover (should be no-op)
        registry.recover_agent(room_id, agent_id, human_id).await.unwrap();

        // Still connected
        assert!(registry.agent_is_connected(room_id, agent_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_cannot_recover_human() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Try to recover human (should fail)
        let result = registry.recover_agent(room_id, human_id, human_id).await;
        assert!(matches!(result, Err(ConclaveError::NotAuthorized(_))));
    }
}
