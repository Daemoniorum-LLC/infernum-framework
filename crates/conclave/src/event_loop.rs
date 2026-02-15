//! Event loop for processing agent backend events.
//!
//! This module provides the background event processing that connects
//! agent backends to the room system. It:
//!
//! - Monitors all active backend sessions for events
//! - Routes events through `handle_agent_event`
//! - Manages session lifecycle (termination cleanup)
//!
//! # Architecture
//!
//! ```text
//! ┌────────────┐     ┌────────────┐     ┌────────────┐
//! │ Claude Code│     │  Infernum  │     │   Custom   │
//! │  Session   │     │  Session   │     │  Session   │
//! └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
//!       │                  │                  │
//!       │ AgentEvent       │ AgentEvent       │ AgentEvent
//!       ▼                  ▼                  ▼
//! ┌─────────────────────────────────────────────────────┐
//! │                    Event Loop                        │
//! │              (monitors all sessions)                 │
//! └─────────────────────────┬───────────────────────────┘
//!                           │
//!                           │ handle_agent_event()
//!                           ▼
//! ┌─────────────────────────────────────────────────────┐
//! │                   RoomRegistry                       │
//! │        (messages, turns, attention, etc.)           │
//! └─────────────────────────┬───────────────────────────┘
//!                           │
//!                           │ RoomEvent
//!                           ▼
//!                      [Subscribers]
//! ```

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

use crate::backend::AgentEvent;
use crate::room::RoomRegistry;
use crate::types::{ParticipantId, RoomId};

// =============================================================================
// Event Loop Handle
// =============================================================================

/// Handle to a running event loop.
///
/// The event loop runs in the background, processing events from all
/// agent backend sessions. Drop this handle to stop the loop.
pub struct EventLoopHandle {
    /// Task handle for the main event loop.
    task: JoinHandle<()>,
    /// Shutdown signal sender.
    shutdown_tx: mpsc::Sender<()>,
}

impl EventLoopHandle {
    /// Signals the event loop to shut down gracefully.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(()).await;
        let _ = self.task.await;
    }

    /// Checks if the event loop is still running.
    pub fn is_running(&self) -> bool {
        !self.task.is_finished()
    }
}

// =============================================================================
// Event Loop
// =============================================================================

/// Starts the global event loop for a room registry.
///
/// This spawns a background task that can handle system-wide events.
/// Individual agent session events are handled by `SessionEventProcessor`
/// which is started automatically when using `spawn_agent_with_events`.
///
/// # Event Processing Architecture
///
/// - **Per-session processing**: Use `spawn_agent_with_events()` which
///   automatically starts a `SessionEventProcessor` for each agent.
///   This is the recommended approach for most use cases.
///
/// - **Global event loop**: This loop handles system-wide concerns like
///   monitoring session health, cleanup of dead sessions, etc.
///
/// # Arguments
///
/// * `registry` - The room registry to process events for
///
/// # Returns
///
/// An `EventLoopHandle` that can be used to shut down the loop.
pub fn start_event_loop(registry: Arc<RoomRegistry>) -> EventLoopHandle {
    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    let task = tokio::spawn(async move {
        info!("Global event loop started");

        let mut health_check_interval = tokio::time::interval(std::time::Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Event loop received shutdown signal");
                    break;
                }

                _ = health_check_interval.tick() => {
                    // Periodic health check for all active sessions
                    let sessions = registry.sessions.read().await;
                    let active_count = sessions.len();
                    let running_count = sessions.values()
                        .filter(|s| s.is_running())
                        .count();

                    if active_count > 0 {
                        debug!(
                            "Session health: {}/{} sessions running",
                            running_count, active_count
                        );
                    }
                }
            }
        }

        info!("Global event loop stopped");
    });

    EventLoopHandle { task, shutdown_tx }
}

// =============================================================================
// Session Event Processor
// =============================================================================

/// Processes events from a single backend session.
///
/// This is spawned as a background task for each active session.
pub struct SessionEventProcessor {
    registry: Arc<RoomRegistry>,
    room_id: RoomId,
    agent_id: ParticipantId,
    event_rx: mpsc::Receiver<AgentEvent>,
}

impl SessionEventProcessor {
    /// Creates a new session event processor.
    pub fn new(
        registry: Arc<RoomRegistry>,
        room_id: RoomId,
        agent_id: ParticipantId,
        event_rx: mpsc::Receiver<AgentEvent>,
    ) -> Self {
        Self {
            registry,
            room_id,
            agent_id,
            event_rx,
        }
    }

    /// Starts processing events in the background.
    ///
    /// Returns a handle to the spawned task.
    pub fn start(mut self) -> JoinHandle<()> {
        tokio::spawn(async move {
            info!(
                "Session event processor started for agent {} in room {}",
                self.agent_id, self.room_id
            );

            while let Some(event) = self.event_rx.recv().await {
                debug!(
                    "Received event from agent {}: {:?}",
                    self.agent_id,
                    event_summary(&event)
                );

                // Check for termination
                let is_terminal = matches!(event, AgentEvent::Terminated { .. });

                // Process the event
                if let Err(e) = self
                    .registry
                    .handle_agent_event(self.room_id, self.agent_id, event)
                    .await
                {
                    error!(
                        "Failed to handle event from agent {}: {}",
                        self.agent_id, e
                    );
                }

                // Stop processing if session terminated
                if is_terminal {
                    info!(
                        "Agent {} terminated, stopping event processor",
                        self.agent_id
                    );
                    break;
                }
            }

            info!(
                "Session event processor stopped for agent {}",
                self.agent_id
            );
        })
    }
}

/// Returns a summary of an event for logging.
fn event_summary(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::Message { .. } => "Message",
        AgentEvent::ToolCall { .. } => "ToolCall",
        AgentEvent::ToolResult { .. } => "ToolResult",
        AgentEvent::Thinking { .. } => "Thinking",
        AgentEvent::AttentionChanged { .. } => "AttentionChanged",
        AgentEvent::TurnRequested { .. } => "TurnRequested",
        AgentEvent::TurnYielded => "TurnYielded",
        AgentEvent::InviteRequested { .. } => "InviteRequested",
        AgentEvent::Terminated { .. } => "Terminated",
        AgentEvent::Error { .. } => "Error",
    }
}

// =============================================================================
// Room Registry Extensions
// =============================================================================

impl RoomRegistry {
    /// Starts an event processor for a spawned agent session.
    ///
    /// Call this after spawning an agent to begin processing its events.
    /// The processor runs in the background until the session terminates.
    pub fn start_session_processor(
        self: &Arc<Self>,
        room_id: RoomId,
        agent_id: ParticipantId,
        event_rx: mpsc::Receiver<AgentEvent>,
    ) -> JoinHandle<()> {
        let processor = SessionEventProcessor::new(
            Arc::clone(self),
            room_id,
            agent_id,
            event_rx,
        );
        processor.start()
    }

    /// Spawns an agent and starts its event processor.
    ///
    /// This is the preferred method for spawning agents as it ensures
    /// events are properly processed. The event processor runs in the
    /// background until the session terminates.
    ///
    /// # Returns
    ///
    /// A tuple of (agent_id, processor_handle). The handle can be used
    /// to wait for the processor to finish or check if it's still running.
    pub async fn spawn_agent_with_events(
        self: &Arc<Self>,
        room_id: RoomId,
        config: crate::types::AgentConfig,
        spawned_by: ParticipantId,
    ) -> crate::error::Result<(ParticipantId, JoinHandle<()>)> {
        // Spawn the agent using the existing method
        let agent_id = self.spawn_agent(room_id, config, spawned_by).await?;

        // Take the event receiver from the session
        let event_rx = {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(&agent_id) {
                session.take_event_receiver()
            } else {
                None
            }
        };

        // Start the event processor
        let handle = if let Some(rx) = event_rx {
            self.start_session_processor(room_id, agent_id, rx)
        } else {
            // Session doesn't have an event receiver, spawn a no-op task
            tokio::spawn(async {})
        };

        Ok((agent_id, handle))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TerminationReason;

    #[tokio::test]
    async fn test_event_loop_start_stop() {
        let registry = Arc::new(RoomRegistry::with_defaults());
        let handle = start_event_loop(registry);

        assert!(handle.is_running());

        handle.shutdown().await;
        // After shutdown, task should be finished
    }

    #[tokio::test]
    async fn test_session_processor_handles_events() {
        use crate::types::{AgentConfig, CreateRoomRequest, UserId};
        use std::path::PathBuf;

        let registry = Arc::new(RoomRegistry::with_defaults());

        // Create room with agent
        let request = CreateRoomRequest::new(
            "Test Room",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            UserId("test".to_string()),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        let agent_id = registry
            .spawn_agent(room_id, AgentConfig::infernum("test"), human)
            .await
            .unwrap();

        // Create event channel
        let (tx, rx) = mpsc::channel(16);

        // Start processor
        let handle = registry.start_session_processor(room_id, agent_id, rx);

        // Send some events
        tx.send(AgentEvent::Message {
            content: "Hello from test".to_string(),
            mentions: vec![],
        })
        .await
        .unwrap();

        // Give processor time to handle
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Verify message was created
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert!(messages.iter().any(|m| {
            if let crate::types::MessageContent::Text { content } = &m.content {
                content == "Hello from test"
            } else {
                false
            }
        }));

        // Send termination
        tx.send(AgentEvent::Terminated {
            reason: TerminationReason::Completed,
        })
        .await
        .unwrap();

        // Processor should stop
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(handle.is_finished());
    }

    #[tokio::test]
    async fn test_processor_stops_on_channel_close() {
        use crate::types::{AgentConfig, CreateRoomRequest, UserId};
        use std::path::PathBuf;

        let registry = Arc::new(RoomRegistry::with_defaults());

        let request = CreateRoomRequest::new(
            "Test Room",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            UserId("test".to_string()),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        let agent_id = registry
            .spawn_agent(room_id, AgentConfig::infernum("test"), human)
            .await
            .unwrap();

        let (tx, rx) = mpsc::channel(16);
        let handle = registry.start_session_processor(room_id, agent_id, rx);

        // Drop sender to close channel
        drop(tx);

        // Processor should stop gracefully
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(handle.is_finished());
    }

    #[test]
    fn test_event_summary() {
        assert_eq!(
            event_summary(&AgentEvent::Message {
                content: "".to_string(),
                mentions: vec![]
            }),
            "Message"
        );
        assert_eq!(
            event_summary(&AgentEvent::Terminated {
                reason: TerminationReason::Completed
            }),
            "Terminated"
        );
    }
}
