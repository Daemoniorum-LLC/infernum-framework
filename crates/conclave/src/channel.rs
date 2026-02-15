//! Channel and messaging infrastructure.
//!
//! Implements AGENT-COLLABORATION-SPEC.md §3.3 Messaging.

use chrono::Utc;
use std::collections::HashMap;

use crate::error::{ConclaveError, Result};
use crate::room::{RoomEvent, RoomRegistry};
use crate::types::{
    ChannelType, Message, MessageContent, MessageId, ParticipantId, RoomId,
};

impl RoomRegistry {
    // =========================================================================
    // Messaging
    // =========================================================================

    /// Sends a text message to the main channel.
    ///
    /// # Preconditions
    /// - Room exists and is not archived
    /// - Sender is a participant in the room
    ///
    /// # Postconditions
    /// - Message added to room history
    /// - MessageSent event emitted
    pub async fn send_message(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        content: String,
    ) -> Result<MessageId> {
        self.send_message_to_channel(
            room_id,
            sender,
            ChannelType::Main,
            MessageContent::Text { content },
        )
        .await
    }

    /// Sends a direct message to specific participants.
    pub async fn send_dm(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        recipients: Vec<ParticipantId>,
        content: String,
    ) -> Result<MessageId> {
        // Include sender in DM participants
        let mut participants = recipients;
        if !participants.contains(&sender) {
            participants.push(sender);
        }
        // Sort participants for consistent channel identity
        // DM([a, b]) should be the same channel as DM([b, a])
        participants.sort();

        self.send_message_to_channel(
            room_id,
            sender,
            ChannelType::DirectMessage { participants },
            MessageContent::Text { content },
        )
        .await
    }

    /// Sends a message to a thread.
    pub async fn send_thread_message(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        parent_message: MessageId,
        content: String,
    ) -> Result<MessageId> {
        self.send_message_to_channel(
            room_id,
            sender,
            ChannelType::Thread { parent_message },
            MessageContent::Text { content },
        )
        .await
    }

    /// Internal: sends a message to any channel type.
    async fn send_message_to_channel(
        &self,
        room_id: RoomId,
        sender: ParticipantId,
        channel: ChannelType,
        content: MessageContent,
    ) -> Result<MessageId> {
        // Verify room exists and is active
        {
            let rooms = self.rooms.read().await;
            let room = rooms
                .get(&room_id)
                .ok_or(ConclaveError::RoomNotFound(room_id))?;

            if room.archived {
                return Err(ConclaveError::RoomArchived(room_id));
            }

            // Verify sender is in room
            if room.find_participant(sender).is_none() {
                return Err(ConclaveError::NotInRoom(sender, room_id));
            }
        }

        // Create message
        let message_id = MessageId::new();
        let message = Message {
            id: message_id,
            channel,
            sender,
            content,
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

        // Update room timestamp
        {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
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

    /// Gets messages for a specific channel type.
    pub async fn get_channel_messages(
        &self,
        room_id: RoomId,
        channel: &ChannelType,
    ) -> Result<Vec<Message>> {
        let rooms = self.rooms.read().await;
        if !rooms.contains_key(&room_id) {
            return Err(ConclaveError::RoomNotFound(room_id));
        }
        drop(rooms);

        let messages = self.messages.read().await;
        let room_messages = messages.get(&room_id).cloned().unwrap_or_default();

        Ok(room_messages
            .into_iter()
            .filter(|m| &m.channel == channel)
            .collect())
    }

    /// Gets main channel messages.
    pub async fn get_main_channel_messages(&self, room_id: RoomId) -> Result<Vec<Message>> {
        self.get_channel_messages(room_id, &ChannelType::Main).await
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
    // Phase 3: Channel Tests
    // -------------------------------------------------------------------------

    /// spec_send_message_to_main
    #[tokio::test]
    async fn test_send_message_to_main_channel() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let sender = room.participants[0].id;

        // Send message
        let message_id = registry
            .send_message(room_id, sender, "Hello world".to_string())
            .await
            .unwrap();

        // Verify
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, message_id);
        assert!(matches!(&messages[0].content, MessageContent::Text { content } if content == "Hello world"));
    }

    /// spec_send_dm
    #[tokio::test]
    async fn test_send_dm() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator = room.participants[0].id;

        // Add another user
        let other_id = registry
            .join_room(room_id, other_user(), "Other".to_string())
            .await
            .unwrap();

        // Send DM
        registry
            .send_dm(room_id, creator, vec![other_id], "Private message".to_string())
            .await
            .unwrap();

        // Verify DM exists - participants must be sorted for consistent matching
        let mut dm_participants = vec![other_id, creator];
        dm_participants.sort();
        let dm_channel = ChannelType::DirectMessage {
            participants: dm_participants,
        };
        let messages = registry.get_channel_messages(room_id, &dm_channel).await.unwrap();
        assert!(!messages.is_empty(), "DM should contain the sent message");
    }

    /// spec_send_thread_message
    #[tokio::test]
    async fn test_send_thread_message() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let sender = room.participants[0].id;

        // Send main message
        let parent_id = registry
            .send_message(room_id, sender, "Parent message".to_string())
            .await
            .unwrap();

        // Send thread reply
        registry
            .send_thread_message(room_id, sender, parent_id, "Thread reply".to_string())
            .await
            .unwrap();

        // Verify
        let thread_channel = ChannelType::Thread { parent_message: parent_id };
        let messages = registry.get_channel_messages(room_id, &thread_channel).await.unwrap();
        assert_eq!(messages.len(), 1);
    }

    /// spec_message_sent_event
    #[tokio::test]
    async fn test_message_sent_event() {
        let registry = RoomRegistry::with_defaults();
        let mut rx = registry.subscribe();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let sender = room.participants[0].id;

        // Drain creation event
        let _ = rx.recv().await;

        // Send message
        registry
            .send_message(room_id, sender, "Hello".to_string())
            .await
            .unwrap();

        // Check event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, RoomEvent::MessageSent { .. }));
    }

    /// Cannot send message to archived room
    #[tokio::test]
    async fn test_cannot_send_to_archived_room() {
        let registry = RoomRegistry::with_defaults();

        // Create and archive room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let creator = room.participants[0].id;
        registry.archive_room(room_id, creator).await.unwrap();

        // Try to send message
        let result = registry
            .send_message(room_id, creator, "Hello".to_string())
            .await;
        assert!(matches!(result, Err(ConclaveError::RoomArchived(_))));
    }

    /// Cannot send message as non-participant
    #[tokio::test]
    async fn test_cannot_send_as_non_participant() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();

        // Try to send as random participant
        let fake_sender = ParticipantId::new();
        let result = registry
            .send_message(room_id, fake_sender, "Hello".to_string())
            .await;
        assert!(matches!(result, Err(ConclaveError::NotInRoom(_, _))));
    }

    /// Messages preserve order
    #[tokio::test]
    async fn test_message_ordering() {
        let registry = RoomRegistry::with_defaults();

        // Create room
        let request = CreateRoomRequest::new("Test Room", test_working_dir(), test_user());
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();
        let sender = room.participants[0].id;

        // Send multiple messages
        for i in 0..5 {
            registry
                .send_message(room_id, sender, format!("Message {}", i))
                .await
                .unwrap();
        }

        // Verify order
        let messages = registry.get_main_channel_messages(room_id).await.unwrap();
        assert_eq!(messages.len(), 5);
        for (i, msg) in messages.iter().enumerate() {
            if let MessageContent::Text { content } = &msg.content {
                assert_eq!(content, &format!("Message {}", i));
            }
        }
    }
}
