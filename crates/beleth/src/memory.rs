//! Agent memory systems.

use infernum_core::Message;

/// Agent memory containing conversation history.
pub struct AgentMemory {
    /// Working memory (current conversation).
    messages: Vec<Message>,
    /// Maximum messages before summarization.
    max_messages: usize,
}

impl AgentMemory {
    /// Creates a new agent memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 50,
        }
    }

    /// Creates with a custom max messages limit.
    #[must_use]
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
        }
    }

    /// Adds a message to memory.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);

        // TODO: Implement summarization when exceeding max
        if self.messages.len() > self.max_messages {
            // For now, just remove oldest non-system message
            if let Some(idx) = self.messages.iter().position(|m| {
                !matches!(m.role, infernum_core::Role::System)
            }) {
                self.messages.remove(idx);
            }
        }
    }

    /// Returns all messages.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clears non-system messages.
    pub fn clear(&mut self) {
        self.messages.retain(|m| matches!(m.role, infernum_core::Role::System));
    }

    /// Returns the number of messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns true if memory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new()
    }
}
