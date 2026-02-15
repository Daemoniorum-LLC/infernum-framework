//! Attention management for agents.
//!
//! This module automates attention state transitions:
//!
//! - **Auto-escalation**: When an agent receives a direct @mention,
//!   escalate to Focused
//! - **Auto-decay**: Focused agents decay to Available after inactivity
//! - **Human override**: Humans can override agent attention states
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐                    ┌──────────────┐
//! │   Message    │  @mentions agent   │  Attention   │
//! │   Routing    │ ─────────────────► │  Escalator   │
//! └──────────────┘                    └──────────────┘
//!                                            │
//!                                            │ escalate to Focused
//!                                            ▼
//! ┌──────────────┐                    ┌──────────────┐
//! │   Decay      │  timeout elapsed   │    Agent     │
//! │   Timer      │ ─────────────────► │   State      │
//! └──────────────┘                    └──────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tracing::{debug, info};

use crate::room::RoomRegistry;
use crate::types::{AttentionState, ParticipantId, RoomId};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for attention management.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Duration after which Focused state decays to Available.
    /// Default: 5 minutes.
    pub focus_decay_timeout: Duration,

    /// Whether to auto-escalate on @mentions.
    /// Default: true.
    pub auto_escalate_on_mention: bool,

    /// How often to check for decayed focus states.
    /// Default: 30 seconds.
    pub decay_check_interval: Duration,

    /// Duration for human override to persist before agent can change.
    /// Default: 10 minutes.
    pub human_override_duration: Duration,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            focus_decay_timeout: Duration::from_secs(5 * 60),
            auto_escalate_on_mention: true,
            decay_check_interval: Duration::from_secs(30),
            human_override_duration: Duration::from_secs(10 * 60),
        }
    }
}

// =============================================================================
// Attention Manager
// =============================================================================

/// Manages attention states for agents in rooms.
///
/// Tracks focus times and human overrides to enable automatic
/// attention transitions.
pub struct AttentionManager {
    /// Configuration.
    config: AttentionConfig,

    /// Room registry reference.
    registry: Arc<RoomRegistry>,

    /// Tracks when agents entered Focused state.
    /// Key: (RoomId, ParticipantId), Value: when focus started.
    focus_times: RwLock<HashMap<(RoomId, ParticipantId), DateTime<Utc>>>,

    /// Tracks human overrides.
    /// Key: (RoomId, ParticipantId), Value: when override was set.
    human_overrides: RwLock<HashMap<(RoomId, ParticipantId), DateTime<Utc>>>,
}

impl AttentionManager {
    /// Creates a new attention manager.
    pub fn new(registry: Arc<RoomRegistry>, config: AttentionConfig) -> Self {
        Self {
            config,
            registry,
            focus_times: RwLock::new(HashMap::new()),
            human_overrides: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a new attention manager with default configuration.
    pub fn with_defaults(registry: Arc<RoomRegistry>) -> Self {
        Self::new(registry, AttentionConfig::default())
    }

    /// Escalates an agent to Focused state when mentioned.
    ///
    /// Called by message routing when an agent is directly @mentioned.
    pub async fn on_mention(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        task_context: Option<String>,
    ) {
        if !self.config.auto_escalate_on_mention {
            return;
        }

        // Check if there's an active human override
        if self.has_active_human_override(room_id, agent_id).await {
            debug!(
                "Agent {} has human override, skipping auto-escalation",
                agent_id
            );
            return;
        }

        let now = Utc::now();
        let new_state = AttentionState::Focused {
            task: task_context.unwrap_or_else(|| "Responding to mention".to_string()),
            started: now,
            eta: None,
            interruptible: true,
        };

        // Set the attention state
        if let Err(e) = self.registry.set_attention(room_id, agent_id, new_state).await {
            debug!("Failed to escalate attention for agent {}: {}", agent_id, e);
            return;
        }

        // Track when focus started
        self.focus_times
            .write()
            .await
            .insert((room_id, agent_id), now);

        info!(
            "Auto-escalated agent {} to Focused in room {}",
            agent_id, room_id
        );
    }

    /// Records a human override of an agent's attention state.
    ///
    /// While a human override is active, the agent cannot change
    /// its own attention state.
    pub async fn record_human_override(&self, room_id: RoomId, agent_id: ParticipantId) {
        self.human_overrides
            .write()
            .await
            .insert((room_id, agent_id), Utc::now());

        info!(
            "Recorded human override for agent {} in room {}",
            agent_id, room_id
        );
    }

    /// Checks if an agent can change its own attention state.
    ///
    /// Returns false if there's an active human override.
    pub async fn agent_can_change_attention(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
    ) -> bool {
        !self.has_active_human_override(room_id, agent_id).await
    }

    /// Checks if there's an active human override for an agent.
    async fn has_active_human_override(&self, room_id: RoomId, agent_id: ParticipantId) -> bool {
        let overrides = self.human_overrides.read().await;
        if let Some(override_time) = overrides.get(&(room_id, agent_id)) {
            let now = Utc::now();
            let elapsed = (now - *override_time).to_std().unwrap_or(Duration::ZERO);
            elapsed < self.config.human_override_duration
        } else {
            false
        }
    }

    /// Updates focus time when agent activity is detected.
    ///
    /// Called when agent sends a message or makes a tool call to
    /// reset the decay timer.
    pub async fn record_activity(&self, room_id: RoomId, agent_id: ParticipantId) {
        let mut focus_times = self.focus_times.write().await;
        if focus_times.contains_key(&(room_id, agent_id)) {
            // Agent is focused, update the time to reset decay
            focus_times.insert((room_id, agent_id), Utc::now());
        }
    }

    /// Runs a single decay check cycle.
    ///
    /// This is called by the decay timer and can also be called
    /// manually for testing.
    pub async fn check_and_decay(&self) {
        let now = Utc::now();
        let decay_threshold = self.config.focus_decay_timeout;

        // Find agents that should decay
        let to_decay: Vec<_> = {
            let focus_times = self.focus_times.read().await;
            focus_times
                .iter()
                .filter(|(_, start_time)| {
                    let elapsed = (now - **start_time).to_std().unwrap_or(Duration::ZERO);
                    elapsed >= decay_threshold
                })
                .map(|((room_id, agent_id), _)| (*room_id, *agent_id))
                .collect()
        };

        // Decay each agent
        for (room_id, agent_id) in to_decay {
            // Remove from focus tracking
            self.focus_times
                .write()
                .await
                .remove(&(room_id, agent_id));

            // Set to Available
            let new_state = AttentionState::Available;
            if let Err(e) = self.registry.set_attention(room_id, agent_id, new_state).await {
                debug!(
                    "Failed to decay attention for agent {} in room {}: {}",
                    agent_id, room_id, e
                );
                continue;
            }

            info!(
                "Decayed agent {} from Focused to Available in room {}",
                agent_id, room_id
            );
        }
    }

    /// Clears focus tracking for an agent.
    ///
    /// Called when agent terminates or leaves room.
    pub async fn clear_agent(&self, room_id: RoomId, agent_id: ParticipantId) {
        self.focus_times
            .write()
            .await
            .remove(&(room_id, agent_id));
        self.human_overrides
            .write()
            .await
            .remove(&(room_id, agent_id));
    }

    /// Clears all tracking for a room.
    ///
    /// Called when room is archived.
    pub async fn clear_room(&self, room_id: RoomId) {
        self.focus_times
            .write()
            .await
            .retain(|(rid, _), _| *rid != room_id);
        self.human_overrides
            .write()
            .await
            .retain(|(rid, _), _| *rid != room_id);
    }
}

// =============================================================================
// Decay Timer
// =============================================================================

/// Handle to a running attention decay timer.
pub struct AttentionDecayHandle {
    task: JoinHandle<()>,
}

impl AttentionDecayHandle {
    /// Stops the decay timer.
    pub fn stop(self) {
        self.task.abort();
    }

    /// Checks if the timer is still running.
    pub fn is_running(&self) -> bool {
        !self.task.is_finished()
    }
}

/// Starts the attention decay timer.
///
/// This runs in the background and periodically checks for
/// focused agents that should decay to Available.
pub fn start_decay_timer(manager: Arc<AttentionManager>) -> AttentionDecayHandle {
    let interval = manager.config.decay_check_interval;

    let task = tokio::spawn(async move {
        let mut timer = tokio::time::interval(interval);

        loop {
            timer.tick().await;
            manager.check_and_decay().await;
        }
    });

    AttentionDecayHandle { task }
}

// =============================================================================
// RoomRegistry Integration
// =============================================================================

impl RoomRegistry {
    /// Sets an agent's attention state with human override tracking.
    ///
    /// Use this method when a human explicitly sets an agent's attention
    /// to ensure the agent cannot immediately override it.
    pub async fn set_attention_with_override(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        new_state: AttentionState,
        attention_manager: &AttentionManager,
    ) -> crate::error::Result<()> {
        // Record the override
        attention_manager.record_human_override(room_id, agent_id).await;

        // Set the attention state
        self.set_attention(room_id, agent_id, new_state).await
    }

    /// Sets an agent's attention from agent self-report.
    ///
    /// This checks if the agent is allowed to change its attention
    /// (no active human override) before applying.
    pub async fn set_attention_from_agent(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        new_state: AttentionState,
        attention_manager: &AttentionManager,
    ) -> crate::error::Result<()> {
        // Check if agent can change
        if !attention_manager.agent_can_change_attention(room_id, agent_id).await {
            debug!(
                "Agent {} cannot change attention due to human override",
                agent_id
            );
            return Ok(());
        }

        // Set the attention state
        self.set_attention(room_id, agent_id, new_state).await
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

    #[tokio::test]
    async fn test_attention_config_defaults() {
        let config = AttentionConfig::default();
        assert_eq!(config.focus_decay_timeout, Duration::from_secs(5 * 60));
        assert!(config.auto_escalate_on_mention);
    }

    #[tokio::test]
    async fn test_on_mention_escalates_to_focused() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let manager = AttentionManager::with_defaults(Arc::clone(&registry));

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        // Spawn agent
        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Verify agent starts as Available
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Available));

        // Trigger mention
        manager.on_mention(room_id, agent_id, Some("Fix the bug".to_string())).await;

        // Verify agent is now Focused
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Focused { task, .. } if task == "Fix the bug"));
    }

    #[tokio::test]
    async fn test_human_override_blocks_agent_change() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let config = AttentionConfig {
            human_override_duration: Duration::from_secs(60),
            ..Default::default()
        };
        let manager = AttentionManager::new(Arc::clone(&registry), config);

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Human sets agent to DND
        registry.set_attention_with_override(
            room_id,
            agent_id,
            AttentionState::DoNotDisturb {
                reason: "On break".to_string(),
                until: None,
            },
            &manager,
        ).await.unwrap();

        // Agent cannot change to Available
        assert!(!manager.agent_can_change_attention(room_id, agent_id).await);
    }

    #[tokio::test]
    async fn test_focus_decay() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let config = AttentionConfig {
            focus_decay_timeout: Duration::from_millis(50),
            ..Default::default()
        };
        let manager = AttentionManager::new(Arc::clone(&registry), config);

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Set agent to focused via mention
        manager.on_mention(room_id, agent_id, None).await;

        // Verify focused
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Focused { .. }));

        // Wait for decay
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Run decay check
        manager.check_and_decay().await;

        // Verify decayed to Available
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Available));
    }

    #[tokio::test]
    async fn test_activity_resets_decay_timer() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let config = AttentionConfig {
            focus_decay_timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let manager = AttentionManager::new(Arc::clone(&registry), config);

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Set agent to focused
        manager.on_mention(room_id, agent_id, None).await;

        // Wait partway
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Record activity (resets timer)
        manager.record_activity(room_id, agent_id).await;

        // Wait more (but not full timeout from new activity)
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Run decay check - should NOT decay yet
        manager.check_and_decay().await;

        // Still focused
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Focused { .. }));

        // Wait for full timeout
        tokio::time::sleep(Duration::from_millis(50)).await;
        manager.check_and_decay().await;

        // Now decayed
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Available));
    }

    #[tokio::test]
    async fn test_clear_agent_removes_tracking() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let manager = AttentionManager::with_defaults(Arc::clone(&registry));

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Set up tracking
        manager.on_mention(room_id, agent_id, None).await;
        manager.record_human_override(room_id, agent_id).await;

        // Clear agent
        manager.clear_agent(room_id, agent_id).await;

        // Tracking should be gone
        assert!(manager.focus_times.read().await.is_empty());
        assert!(manager.human_overrides.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_decay_timer_runs() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let config = AttentionConfig {
            decay_check_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let manager = Arc::new(AttentionManager::new(Arc::clone(&registry), config));

        let handle = start_decay_timer(Arc::clone(&manager));

        // Timer should be running
        assert!(handle.is_running());

        // Stop it
        handle.stop();

        // Give it time to stop
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    #[tokio::test]
    async fn test_mention_skipped_when_disabled() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let config = AttentionConfig {
            auto_escalate_on_mention: false,
            ..Default::default()
        };
        let manager = AttentionManager::new(Arc::clone(&registry), config);

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human_id = room.participants[0].id;

        let config = crate::types::AgentConfig::infernum("test");
        let agent_id = registry.spawn_agent(room_id, config, human_id).await.unwrap();

        // Trigger mention
        manager.on_mention(room_id, agent_id, None).await;

        // Agent should still be Available (not escalated)
        let agent = registry.get_participant(room_id, agent_id).await.unwrap();
        assert!(matches!(agent.attention, AttentionState::Available));
    }
}
