//! Turn coordination for the main channel.
//!
//! Implements AGENT-COLLABORATION-SPEC.md §2.6 Coordinator Types and §3.5 Turn Coordination.
//!
//! The coordinator manages turn-taking in the Main channel to prevent cacophony.
//! Agents request turns, and the coordinator manages a priority queue.

use std::collections::VecDeque;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::backend::TurnPriority;
use crate::error::{ConclaveError, Result};
use crate::room::RoomRegistry;
use crate::types::{CoordinatorMode, ParticipantId, RoomId};

// =============================================================================
// Coordinator Types
// =============================================================================

/// State of the turn coordinator for a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorState {
    /// Current speaker (if any).
    pub current_speaker: Option<ParticipantId>,
    /// Queue of pending turn requests.
    pub queue: VecDeque<TurnRequest>,
    /// When the current speaker got their turn.
    pub turn_started: Option<DateTime<Utc>>,
    /// Last state transition timestamp.
    pub last_transition: DateTime<Utc>,
}

impl Default for CoordinatorState {
    fn default() -> Self {
        Self {
            current_speaker: None,
            queue: VecDeque::new(),
            turn_started: None,
            last_transition: Utc::now(),
        }
    }
}

impl CoordinatorState {
    /// Creates a new coordinator state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if there's a current speaker.
    pub fn has_speaker(&self) -> bool {
        self.current_speaker.is_some()
    }

    /// Returns how long the current speaker has had the turn.
    pub fn turn_duration(&self) -> Option<Duration> {
        self.turn_started.map(|started| Utc::now() - started)
    }
}

/// A request to speak in the main channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRequest {
    /// Who is requesting the turn.
    pub participant: ParticipantId,
    /// When the request was made.
    pub requested_at: DateTime<Utc>,
    /// Optional reason for wanting to speak.
    pub reason: Option<String>,
    /// Priority of this request.
    pub priority: TurnPriority,
    /// Whether this participant gets a priority boost (e.g., human).
    pub priority_boost: bool,
}

impl TurnRequest {
    /// Creates a new turn request.
    pub fn new(participant: ParticipantId, priority: TurnPriority) -> Self {
        Self {
            participant,
            requested_at: Utc::now(),
            reason: None,
            priority,
            priority_boost: false,
        }
    }

    /// Sets the reason for the request.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Sets the priority boost flag.
    pub fn with_boost(mut self, boost: bool) -> Self {
        self.priority_boost = boost;
        self
    }

    /// Returns the effective priority for sorting.
    /// Priority boost puts the request at the front.
    pub fn effective_priority(&self) -> u8 {
        let base = match self.priority {
            TurnPriority::Urgent => 3,
            TurnPriority::High => 2,
            TurnPriority::Normal => 1,
            TurnPriority::Yielding => 0,
        };
        if self.priority_boost {
            base + 10 // Boost puts it ahead of everything
        } else {
            base
        }
    }
}

/// Result of requesting a speaking turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnPosition {
    /// Turn was granted immediately.
    Speaking,
    /// Request was queued.
    Queued {
        /// Position in the queue (0-indexed).
        position: usize,
    },
}

/// Event emitted when turn state changes.
#[derive(Debug, Clone)]
pub enum TurnEvent {
    /// A participant got the speaking turn.
    TurnGranted {
        room_id: RoomId,
        participant: ParticipantId,
    },
    /// A participant yielded their turn.
    TurnYielded {
        room_id: RoomId,
        participant: ParticipantId,
        next_speaker: Option<ParticipantId>,
    },
    /// A participant was auto-yielded due to timeout.
    TurnTimedOut {
        room_id: RoomId,
        participant: ParticipantId,
    },
    /// A participant joined the queue.
    TurnQueued {
        room_id: RoomId,
        participant: ParticipantId,
        position: usize,
    },
}

// =============================================================================
// RoomRegistry Coordinator Implementation
// =============================================================================

impl RoomRegistry {
    // =========================================================================
    // Turn Coordination
    // =========================================================================

    /// Requests a speaking turn in the main channel.
    ///
    /// # Preconditions
    /// - Room exists and is not archived
    /// - Participant is in the room
    ///
    /// # Postconditions
    /// - If no current speaker: participant becomes speaker
    /// - Else: participant added to queue based on priority
    pub async fn request_turn(
        &self,
        room_id: RoomId,
        participant: ParticipantId,
        priority: TurnPriority,
        reason: Option<String>,
    ) -> Result<TurnPosition> {
        // First, validate room and participant
        let (is_human, max_queue_depth, mode) = {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            if room.archived {
                return Err(ConclaveError::RoomArchived(room_id));
            }

            let is_human = room
                .find_participant(participant)
                .ok_or(ConclaveError::NotInRoom(participant, room_id))?
                .kind
                .is_human();

            (
                is_human,
                room.coordinator_config.max_queue_depth,
                room.coordinator_config.mode,
            )
        };

        // Now update coordinator state
        let mut states = self.coordinator_states.write().await;
        let state = states.entry(room_id).or_insert_with(CoordinatorState::new);

        // Check if coordinated mode
        if mode == CoordinatorMode::Volunteer
            || mode == CoordinatorMode::RoundRobin
            || mode == CoordinatorMode::Priority
        {
            // Check if already speaking
            if state.current_speaker == Some(participant) {
                return Ok(TurnPosition::Speaking);
            }

            // Check if already in queue
            if state.queue.iter().any(|r| r.participant == participant) {
                let pos = state
                    .queue
                    .iter()
                    .position(|r| r.participant == participant)
                    .unwrap();
                return Ok(TurnPosition::Queued { position: pos });
            }

            // If no current speaker, grant immediately
            if state.current_speaker.is_none() {
                state.current_speaker = Some(participant);
                state.turn_started = Some(Utc::now());
                state.last_transition = Utc::now();

                info!(
                    "Turn granted to {} in room {} (no current speaker)",
                    participant, room_id
                );

                return Ok(TurnPosition::Speaking);
            }

            // Check queue limit
            if state.queue.len() >= max_queue_depth as usize {
                return Err(ConclaveError::QueueFull { max: max_queue_depth });
            }

            // Create turn request
            let mut request = TurnRequest::new(participant, priority);
            if let Some(r) = reason {
                request = request.with_reason(r);
            }
            // Human participants get priority boost
            request = request.with_boost(is_human);

            // Insert into queue based on priority
            let position = Self::insert_by_priority_static(&mut state.queue, request);

            info!(
                "Turn queued for {} in room {} at position {}",
                participant, room_id, position
            );

            Ok(TurnPosition::Queued { position })
        } else {
            // FreeForm mode - no coordination needed, everyone can speak
            Ok(TurnPosition::Speaking)
        }
    }

    /// Yields the current speaking turn.
    ///
    /// # Preconditions
    /// - Participant is the current speaker
    ///
    /// # Postconditions
    /// - Next participant in queue becomes speaker (if any)
    /// - Or current_speaker becomes None
    pub async fn yield_turn(
        &self,
        room_id: RoomId,
        participant: ParticipantId,
    ) -> Result<Option<ParticipantId>> {
        // Validate room and participant
        {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            // Verify participant is in room
            if room.find_participant(participant).is_none() {
                return Err(ConclaveError::NotInRoom(participant, room_id));
            }
        }

        // Update coordinator state directly
        let mut states = self.coordinator_states.write().await;
        let state = states.entry(room_id).or_insert_with(CoordinatorState::new);

        // Verify this participant is the current speaker
        if state.current_speaker != Some(participant) {
            return Err(ConclaveError::NotYourTurn);
        }

        // Yield the turn
        state.current_speaker = None;
        state.turn_started = None;
        state.last_transition = Utc::now();

        // Get next speaker from queue
        let next_speaker = if let Some(next) = state.queue.pop_front() {
            state.current_speaker = Some(next.participant);
            state.turn_started = Some(Utc::now());

            info!(
                "Turn passed from {} to {} in room {}",
                participant, next.participant, room_id
            );

            Some(next.participant)
        } else {
            info!(
                "Turn yielded by {} in room {}, no next speaker",
                participant, room_id
            );
            None
        };

        Ok(next_speaker)
    }

    /// Gets the current coordinator state for a room.
    pub async fn get_coordinator_state(&self, room_id: RoomId) -> Result<CoordinatorState> {
        let states = self.coordinator_states.read().await;
        Ok(states.get(&room_id).cloned().unwrap_or_default())
    }

    /// Gets the current speaker for a room.
    pub async fn get_current_speaker(&self, room_id: RoomId) -> Result<Option<ParticipantId>> {
        let state = self.get_coordinator_state(room_id).await?;
        Ok(state.current_speaker)
    }

    /// Gets the turn queue for a room.
    pub async fn get_turn_queue(&self, room_id: RoomId) -> Result<Vec<TurnRequest>> {
        let state = self.get_coordinator_state(room_id).await?;
        Ok(state.queue.into_iter().collect())
    }

    /// Checks if a participant can send to main channel.
    ///
    /// Returns Ok(()) if allowed, Err(NotYourTurn) if coordination is active
    /// and participant doesn't have the turn.
    pub async fn check_can_send_main(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
    ) -> Result<()> {
        let rooms = self.rooms.read().await;
        let room = rooms
            .get(&room_id)
            .ok_or(ConclaveError::RoomNotFound(room_id))?;

        // FreeForm allows everyone
        if room.coordinator_config.mode == CoordinatorMode::Volunteer {
            // In Volunteer mode, check if sender has the turn
            let state = self.coordinator_states.read().await;
            if let Some(coord_state) = state.get(&room_id) {
                // If there's a current speaker and it's not us, we can't send
                if let Some(speaker) = coord_state.current_speaker {
                    if speaker != sender {
                        return Err(ConclaveError::NotYourTurn);
                    }
                }
                // If no speaker, anyone can send (they become speaker implicitly)
            }
        }
        // Other modes or no coordinator state = allowed

        Ok(())
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Inserts a turn request into the queue by priority.
    /// Returns the position where it was inserted.
    fn insert_by_priority_static(
        queue: &mut VecDeque<TurnRequest>,
        request: TurnRequest,
    ) -> usize {
        let effective = request.effective_priority();

        // Find insertion point - after all requests with higher or equal priority
        let pos = queue
            .iter()
            .position(|r| r.effective_priority() < effective)
            .unwrap_or(queue.len());

        queue.insert(pos, request);
        pos
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CreateRoomRequest, UserId};
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
    // Phase 5.1: Turn Request Tests
    // -------------------------------------------------------------------------

    /// spec_first_speaker_immediate
    #[tokio::test]
    async fn test_first_speaker_gets_immediate_turn() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant = room.participants[0].id;

        // Request turn (should be granted immediately since no current speaker)
        let result = registry
            .request_turn(room_id, participant, TurnPriority::Normal, None)
            .await
            .unwrap();

        assert_eq!(result, TurnPosition::Speaking);

        // Verify coordinator state
        let state = registry.get_coordinator_state(room_id).await.unwrap();
        assert_eq!(state.current_speaker, Some(participant));
    }

    /// spec_subsequent_queued
    #[tokio::test]
    async fn test_subsequent_requests_are_queued() {
        let registry = RoomRegistry::with_defaults();

        // Create room with two participants
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant1 = room.participants[0].id;

        // Add second participant
        let participant2 = registry
            .join_room(room_id, other_user(), "Other".to_string())
            .await
            .unwrap();

        // First request gets turn
        registry
            .request_turn(room_id, participant1, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Second request gets queued
        let result = registry
            .request_turn(room_id, participant2, TurnPriority::Normal, None)
            .await
            .unwrap();

        assert_eq!(result, TurnPosition::Queued { position: 0 });

        // Verify queue
        let queue = registry.get_turn_queue(room_id).await.unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].participant, participant2);
    }

    /// spec_priority_queue
    #[tokio::test]
    async fn test_priority_affects_queue_position() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let speaker = room.participants[0].id;

        // Add more participants
        let agent1 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("model1"),
                speaker,
            )
            .await
            .unwrap();
        let agent2 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("model2"),
                speaker,
            )
            .await
            .unwrap();
        let agent3 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("model3"),
                speaker,
            )
            .await
            .unwrap();

        // Speaker takes turn
        registry
            .request_turn(room_id, speaker, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Queue normal priority requests
        registry
            .request_turn(room_id, agent1, TurnPriority::Normal, None)
            .await
            .unwrap();
        registry
            .request_turn(room_id, agent2, TurnPriority::Normal, None)
            .await
            .unwrap();

        // High priority request should jump ahead
        let result = registry
            .request_turn(room_id, agent3, TurnPriority::High, None)
            .await
            .unwrap();

        assert_eq!(result, TurnPosition::Queued { position: 0 });

        // Verify queue order
        let queue = registry.get_turn_queue(room_id).await.unwrap();
        assert_eq!(queue[0].participant, agent3);
        assert_eq!(queue[0].priority, TurnPriority::High);
    }

    /// spec_human_priority_boost
    #[tokio::test]
    async fn test_human_gets_priority_boost() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        // Add agents
        let agent1 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("model1"),
                human,
            )
            .await
            .unwrap();
        let agent2 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("model2"),
                human,
            )
            .await
            .unwrap();

        // Agent1 takes turn
        registry
            .request_turn(room_id, agent1, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Agent2 queues
        registry
            .request_turn(room_id, agent2, TurnPriority::High, None)
            .await
            .unwrap();

        // Add another human
        let human2 = registry
            .join_room(room_id, other_user(), "Human 2".to_string())
            .await
            .unwrap();

        // Human should get priority boost and jump to front
        let result = registry
            .request_turn(room_id, human2, TurnPriority::Normal, None)
            .await
            .unwrap();

        assert_eq!(result, TurnPosition::Queued { position: 0 });

        // Verify queue - human at front despite normal priority
        let queue = registry.get_turn_queue(room_id).await.unwrap();
        assert_eq!(queue[0].participant, human2);
        assert!(queue[0].priority_boost);
    }

    // -------------------------------------------------------------------------
    // Phase 5.2: Turn Yield Tests
    // -------------------------------------------------------------------------

    /// spec_yield_passes_turn
    #[tokio::test]
    async fn test_yield_passes_turn_to_next() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant1 = room.participants[0].id;

        // Add second participant
        let participant2 = registry
            .join_room(room_id, other_user(), "Other".to_string())
            .await
            .unwrap();

        // First gets turn, second queues
        registry
            .request_turn(room_id, participant1, TurnPriority::Normal, None)
            .await
            .unwrap();
        registry
            .request_turn(room_id, participant2, TurnPriority::Normal, None)
            .await
            .unwrap();

        // First yields
        let next = registry.yield_turn(room_id, participant1).await.unwrap();

        assert_eq!(next, Some(participant2));

        // Verify state
        let state = registry.get_coordinator_state(room_id).await.unwrap();
        assert_eq!(state.current_speaker, Some(participant2));
        assert!(state.queue.is_empty());
    }

    /// spec_yield_empty_queue
    #[tokio::test]
    async fn test_yield_with_empty_queue_clears_speaker() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant = room.participants[0].id;

        // Take turn
        registry
            .request_turn(room_id, participant, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Yield with no queue
        let next = registry.yield_turn(room_id, participant).await.unwrap();

        assert_eq!(next, None);

        // Verify no current speaker
        let state = registry.get_coordinator_state(room_id).await.unwrap();
        assert!(state.current_speaker.is_none());
    }

    /// Cannot yield if not current speaker
    #[tokio::test]
    async fn test_cannot_yield_if_not_speaker() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant1 = room.participants[0].id;

        // Add second participant
        let participant2 = registry
            .join_room(room_id, other_user(), "Other".to_string())
            .await
            .unwrap();

        // First takes turn
        registry
            .request_turn(room_id, participant1, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Second tries to yield (should fail)
        let result = registry.yield_turn(room_id, participant2).await;
        assert!(matches!(result, Err(ConclaveError::NotYourTurn)));
    }

    // -------------------------------------------------------------------------
    // Phase 5.3: Coordinator Mode Tests
    // -------------------------------------------------------------------------

    /// Duplicate request returns current position
    #[tokio::test]
    async fn test_duplicate_request_returns_position() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let participant1 = room.participants[0].id;
        let participant2 = registry
            .join_room(room_id, other_user(), "Other".to_string())
            .await
            .unwrap();

        // First takes turn
        registry
            .request_turn(room_id, participant1, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Second queues
        registry
            .request_turn(room_id, participant2, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Second requests again - should return same position
        let result = registry
            .request_turn(room_id, participant2, TurnPriority::Normal, None)
            .await
            .unwrap();

        assert_eq!(result, TurnPosition::Queued { position: 0 });
    }

    /// Queue full error
    #[tokio::test]
    async fn test_queue_full_error() {
        let registry = RoomRegistry::with_defaults();

        // Create room with small queue
        let mut request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let mut config = crate::types::CoordinatorConfig::default();
        config.max_queue_depth = 2;
        request.coordinator_config = Some(config);

        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator = room.participants[0].id;

        // Spawn agents
        let agent1 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("m1"),
                creator,
            )
            .await
            .unwrap();
        let agent2 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("m2"),
                creator,
            )
            .await
            .unwrap();
        let agent3 = registry
            .spawn_agent(
                room_id,
                crate::types::AgentConfig::infernum("m3"),
                creator,
            )
            .await
            .unwrap();

        // Creator takes turn
        registry
            .request_turn(room_id, creator, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Fill queue
        registry
            .request_turn(room_id, agent1, TurnPriority::Normal, None)
            .await
            .unwrap();
        registry
            .request_turn(room_id, agent2, TurnPriority::Normal, None)
            .await
            .unwrap();

        // Queue full
        let result = registry
            .request_turn(room_id, agent3, TurnPriority::Normal, None)
            .await;
        assert!(matches!(result, Err(ConclaveError::QueueFull { max: 2 })));
    }

    // -------------------------------------------------------------------------
    // TurnRequest Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_turn_request_effective_priority() {
        let participant = ParticipantId::new();

        let normal = TurnRequest::new(participant, TurnPriority::Normal);
        let high = TurnRequest::new(participant, TurnPriority::High);
        let urgent = TurnRequest::new(participant, TurnPriority::Urgent);
        let boosted = TurnRequest::new(participant, TurnPriority::Normal).with_boost(true);

        assert!(urgent.effective_priority() > high.effective_priority());
        assert!(high.effective_priority() > normal.effective_priority());
        assert!(boosted.effective_priority() > urgent.effective_priority());
    }

    #[test]
    fn test_coordinator_state_default() {
        let state = CoordinatorState::default();
        assert!(state.current_speaker.is_none());
        assert!(state.queue.is_empty());
    }
}
