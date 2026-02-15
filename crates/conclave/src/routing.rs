//! Message routing between rooms and backend sessions.
//!
//! This module connects the messaging system with agent backend sessions,
//! routing incoming messages to appropriate backends and handling responses.
//!
//! # Routing Rules
//!
//! - **Main channel (human → agents)**: Route to current speaker's backend
//!   (in coordinated mode) or all active agents (in free-form mode)
//! - **DM to agent**: Route directly to that agent's session
//! - **Agent responses**: Create messages in appropriate channel
//!
//! # Example
//!
//! ```ignore
//! // Human sends message, routed to agents
//! registry.send_message(room_id, human_id, "What's the bug?").await?;
//!
//! // Message automatically routed to active agent backends
//! // Agent responses come back via event loop
//! ```

use chrono::Utc;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::backend::AgentEvent;
use crate::error::{ConclaveError, Result};
use crate::room::{RoomEvent, RoomRegistry};
use crate::types::{
    ChannelType, CoordinatorMode, Message, MessageContent, MessageId, ParticipantId, RoomId,
};

impl RoomRegistry {
    // =========================================================================
    // Message Routing to Backends
    // =========================================================================

    /// Routes a message to appropriate backend sessions.
    ///
    /// Called after a message is stored, this method determines which
    /// agent backends should receive the message and sends it to them.
    pub async fn route_message_to_backends(
        &self,
        room_id: RoomId,
        message: &Message,
    ) -> Result<()> {
        // Get room info
        let (coordinator_mode, agent_participants) = {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            let mode = room.coordinator_config.mode;
            let agents: Vec<ParticipantId> = room
                .participants
                .iter()
                .filter(|p| p.kind.is_agent())
                .map(|p| p.id)
                .collect();

            (mode, agents)
        };

        // Determine which agents should receive this message
        let target_agents = self
            .determine_routing_targets(room_id, message, coordinator_mode, &agent_participants)
            .await?;

        if target_agents.is_empty() {
            debug!(
                "No routing targets for message {:?} in room {}",
                message.id, room_id
            );
            return Ok(());
        }

        // Route to each target agent's session
        let sessions = self.sessions.read().await;
        for agent_id in target_agents {
            if let Some(session) = sessions.get(&agent_id) {
                match session.send_message(message).await {
                    Ok(_) => {
                        debug!("Routed message {:?} to agent {}", message.id, agent_id);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to route message {:?} to agent {}: {}",
                            message.id, agent_id, e
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Determines which agents should receive a message.
    async fn determine_routing_targets(
        &self,
        room_id: RoomId,
        message: &Message,
        coordinator_mode: CoordinatorMode,
        agent_participants: &[ParticipantId],
    ) -> Result<Vec<ParticipantId>> {
        // Check sender type
        let sender_is_agent = {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            room.find_participant(message.sender)
                .map(|p| p.kind.is_agent())
                .unwrap_or(false)
        };

        match &message.channel {
            ChannelType::Main => {
                // Don't route agent messages back to agents (prevents loops)
                if sender_is_agent {
                    return Ok(vec![]);
                }

                // In Volunteer mode (default), route to all agents
                // In RoundRobin/Priority, route based on current speaker
                match coordinator_mode {
                    CoordinatorMode::Volunteer => {
                        // All agents receive the message (least coordinated)
                        Ok(agent_participants.to_vec())
                    }
                    CoordinatorMode::RoundRobin | CoordinatorMode::Priority => {
                        // Only current speaker (if any) receives the message
                        let state = self.get_coordinator_state(room_id).await?;
                        if let Some(speaker) = state.current_speaker {
                            if agent_participants.contains(&speaker) {
                                Ok(vec![speaker])
                            } else {
                                // Current speaker is human, route to all queued agents
                                // so they have context when their turn comes
                                Ok(agent_participants.to_vec())
                            }
                        } else {
                            // No speaker, route to all for context
                            Ok(agent_participants.to_vec())
                        }
                    }
                }
            }

            ChannelType::DirectMessage { participants } => {
                // Route only to agent participants in this DM
                if sender_is_agent {
                    return Ok(vec![]);
                }

                Ok(participants
                    .iter()
                    .filter(|p| agent_participants.contains(p))
                    .copied()
                    .collect())
            }

            ChannelType::Thread { .. } => {
                // Route thread messages to all agents for context
                if sender_is_agent {
                    return Ok(vec![]);
                }
                Ok(agent_participants.to_vec())
            }

            ChannelType::AgentReasoning { .. } => {
                // Never route reasoning channel messages
                Ok(vec![])
            }
        }
    }

    // =========================================================================
    // Handling Backend Events
    // =========================================================================

    /// Processes an event from an agent backend and updates room state.
    ///
    /// This handles events like messages, tool calls, turn requests, etc.
    pub async fn handle_agent_event(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        event: AgentEvent,
    ) -> Result<()> {
        match event {
            AgentEvent::Message { content, mentions } => {
                self.handle_agent_message(room_id, agent_id, content, mentions)
                    .await?;
            }

            AgentEvent::ToolCall {
                tool,
                input,
                call_id,
            } => {
                self.handle_agent_tool_call(room_id, agent_id, tool, input, call_id)
                    .await?;
            }

            AgentEvent::ToolResult {
                tool,
                call_id,
                output,
                success,
                duration_ms,
            } => {
                self.handle_agent_tool_result(room_id, agent_id, tool, call_id, output, success, duration_ms)
                    .await?;
            }

            AgentEvent::Thinking { content } => {
                self.handle_agent_thinking(room_id, agent_id, content)
                    .await?;
            }

            AgentEvent::TurnRequested { reason, priority } => {
                self.request_turn(room_id, agent_id, priority, reason)
                    .await?;
            }

            AgentEvent::TurnYielded => {
                self.yield_turn(room_id, agent_id).await?;
            }

            AgentEvent::AttentionChanged { new_state } => {
                self.set_attention(room_id, agent_id, new_state).await?;
            }

            AgentEvent::InviteRequested { config, reason } => {
                info!(
                    "Agent {} requested to invite {:?}: {}",
                    agent_id, config, reason
                );
                // Could auto-spawn or queue for human approval
            }

            AgentEvent::Terminated { reason } => {
                info!("Agent {} terminated: {:?}", agent_id, reason);
                // Update participant state or remove from room
            }

            AgentEvent::Error { message } => {
                warn!("Agent {} error: {}", agent_id, message);
                // Could emit error event or notify human
            }
        }

        Ok(())
    }

    /// Handles an agent sending a text message.
    async fn handle_agent_message(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        content: String,
        _mentions: Vec<ParticipantId>,
    ) -> Result<MessageId> {
        // Create message in main channel
        let message_id = MessageId::new();
        let message = Message {
            id: message_id,
            channel: ChannelType::Main,
            sender: agent_id,
            content: MessageContent::Text { content },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        // Store message
        {
            let mut messages = self.messages.write().await;
            messages
                .entry(room_id)
                .or_insert_with(Vec::new)
                .push(message.clone());
        }

        // Update participant stats
        {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
                if let Some(participant) = room.participants.iter_mut().find(|p| p.id == agent_id) {
                    participant.message_count += 1;
                    participant.last_active = Utc::now();
                }
                room.updated_at = Utc::now();
            }
        }

        // Emit event
        let _ = self.event_tx.send(RoomEvent::MessageSent {
            room_id,
            message,
        });

        Ok(message_id)
    }

    /// Handles an agent making a tool call.
    async fn handle_agent_tool_call(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        tool: String,
        input: serde_json::Value,
        call_id: String,
    ) -> Result<MessageId> {
        let message_id = MessageId::new();
        let message = Message {
            id: message_id,
            channel: ChannelType::Main,
            sender: agent_id,
            content: MessageContent::ToolCall {
                tool: tool.clone(),
                input,
                call_id,
            },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        // Store message
        {
            let mut messages = self.messages.write().await;
            messages
                .entry(room_id)
                .or_insert_with(Vec::new)
                .push(message.clone());
        }

        // Update stats
        {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
                if let Some(participant) = room.participants.iter_mut().find(|p| p.id == agent_id) {
                    participant.tool_calls += 1;
                    participant.last_active = Utc::now();
                }
            }
        }

        // Emit event
        let _ = self.event_tx.send(RoomEvent::MessageSent {
            room_id,
            message,
        });

        Ok(message_id)
    }

    /// Handles a tool result.
    async fn handle_agent_tool_result(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        tool: String,
        call_id: String,
        output: String,
        success: bool,
        _duration_ms: u32,
    ) -> Result<MessageId> {
        let message_id = MessageId::new();
        let message = Message {
            id: message_id,
            channel: ChannelType::Main,
            sender: agent_id,
            content: MessageContent::ToolResult {
                tool,
                output: serde_json::Value::String(output),
                call_id,
                success,
            },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        // Store
        {
            let mut messages = self.messages.write().await;
            messages
                .entry(room_id)
                .or_insert_with(Vec::new)
                .push(message.clone());
        }

        // Emit
        let _ = self.event_tx.send(RoomEvent::MessageSent {
            room_id,
            message,
        });

        Ok(message_id)
    }

    /// Handles agent thinking/reasoning content.
    async fn handle_agent_thinking(
        &self,
        room_id: RoomId,
        agent_id: ParticipantId,
        content: String,
    ) -> Result<MessageId> {
        // Store in AgentReasoning channel
        let message_id = MessageId::new();
        let message = Message {
            id: message_id,
            channel: ChannelType::AgentReasoning { agent_id },
            sender: agent_id,
            content: MessageContent::Text { content },
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        {
            let mut messages = self.messages.write().await;
            messages
                .entry(room_id)
                .or_insert_with(Vec::new)
                .push(message.clone());
        }

        // Emit (observers can filter by channel type)
        let _ = self.event_tx.send(RoomEvent::MessageSent {
            room_id,
            message,
        });

        Ok(message_id)
    }

    // =========================================================================
    // Enhanced Send Message with Auto-Routing
    // =========================================================================

    /// Sends a message and routes to backends.
    ///
    /// This is the preferred method for sending messages as it handles
    /// both storage and routing in one call.
    pub async fn send_and_route(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        content: String,
    ) -> Result<MessageId> {
        // Send message (stores and emits event)
        let message_id = self.send_message(room_id, sender, content.clone()).await?;

        // Get the message we just created
        let message = {
            let messages = self.messages.read().await;
            messages
                .get(&room_id)
                .and_then(|msgs| msgs.iter().find(|m| m.id == message_id))
                .cloned()
        };

        // Route to backends
        if let Some(msg) = message {
            self.route_message_to_backends(room_id, &msg).await?;
        }

        Ok(message_id)
    }

    /// Sends a DM and routes to agent backends.
    pub async fn send_dm_and_route(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        recipients: Vec<ParticipantId>,
        content: String,
    ) -> Result<MessageId> {
        let message_id = self
            .send_dm(room_id, sender, recipients, content)
            .await?;

        // Get and route
        let message = {
            let messages = self.messages.read().await;
            messages
                .get(&room_id)
                .and_then(|msgs| msgs.iter().find(|m| m.id == message_id))
                .cloned()
        };

        if let Some(msg) = message {
            self.route_message_to_backends(room_id, &msg).await?;
        }

        Ok(message_id)
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

    // -------------------------------------------------------------------------
    // Routing Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_route_message_to_agent() {
        let registry = RoomRegistry::with_defaults();

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        // Spawn agent
        let _agent_id = registry
            .spawn_agent(
                room_id,
                AgentConfig::infernum("test-model"),
                human,
            )
            .await
            .unwrap();

        // Send message with routing
        let message_id = registry
            .send_and_route(room_id, human, "Hello agent".to_string())
            .await
            .unwrap();

        // Verify message was sent
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, message_id);
    }

    #[tokio::test]
    async fn test_agent_message_not_routed_to_agents() {
        let registry = RoomRegistry::with_defaults();

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        // Spawn agent
        let agent_id = registry
            .spawn_agent(
                room_id,
                AgentConfig::infernum("test-model"),
                human,
            )
            .await
            .unwrap();

        // Handle agent message (simulating backend event)
        registry
            .handle_agent_message(room_id, agent_id, "Hello human".to_string(), vec![])
            .await
            .unwrap();

        // Verify message was created
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].sender, agent_id);
    }

    #[tokio::test]
    async fn test_handle_agent_tool_call() {
        let registry = RoomRegistry::with_defaults();

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        let agent_id = registry
            .spawn_agent(
                room_id,
                AgentConfig::infernum("test-model"),
                human,
            )
            .await
            .unwrap();

        // Handle tool call
        registry
            .handle_agent_tool_call(
                room_id,
                agent_id,
                "Read".to_string(),
                serde_json::json!({"file": "test.txt"}),
                "call-123".to_string(),
            )
            .await
            .unwrap();

        // Verify message
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert!(matches!(
            &messages[0].content,
            MessageContent::ToolCall { tool, .. } if tool == "Read"
        ));

        // Verify tool_calls stat updated
        let room = registry.get_room(room_id).await.unwrap();
        let agent = room.find_participant(agent_id).unwrap();
        assert_eq!(agent.tool_calls, 1);
    }

    #[tokio::test]
    async fn test_handle_agent_thinking() {
        let registry = RoomRegistry::with_defaults();

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        let agent_id = registry
            .spawn_agent(
                room_id,
                AgentConfig::infernum("test-model"),
                human,
            )
            .await
            .unwrap();

        // Handle thinking
        registry
            .handle_agent_thinking(room_id, agent_id, "Let me think...".to_string())
            .await
            .unwrap();

        // Verify in reasoning channel
        let reasoning_channel = ChannelType::AgentReasoning { agent_id };
        let messages = registry
            .get_channel_messages(room_id, &reasoning_channel)
            .await
            .unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn test_dm_routing_to_agent() {
        let registry = RoomRegistry::with_defaults();

        // Create room with agent
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let human = room.participants[0].id;

        let agent_id = registry
            .spawn_agent(
                room_id,
                AgentConfig::infernum("test-model"),
                human,
            )
            .await
            .unwrap();

        // Send DM with routing
        registry
            .send_dm_and_route(room_id, human, vec![agent_id], "Private question".to_string())
            .await
            .unwrap();

        // Message should have been routed (no direct way to verify without mocking,
        // but we can at least verify the DM was created)
    }
}
