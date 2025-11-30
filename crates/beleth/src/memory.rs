//! Agent memory systems.

use std::sync::Arc;

use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};
use abaddon::{Engine, InferenceEngine};

/// Memory summarization strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum SummarizationStrategy {
    /// Drop oldest messages (no summarization).
    #[default]
    DropOldest,
    /// Summarize oldest messages into a single message.
    Summarize,
    /// Use a sliding window with overlap.
    SlidingWindow {
        /// Number of messages to keep from the window.
        keep_recent: usize,
    },
}

/// A summary of previous conversation.
#[derive(Debug, Clone)]
pub struct ConversationSummary {
    /// The summary text.
    pub text: String,
    /// Number of messages summarized.
    pub message_count: usize,
    /// Timestamp of summarization.
    pub created_at: u64,
}

/// Agent memory containing conversation history.
pub struct AgentMemory {
    /// Working memory (current conversation).
    messages: Vec<Message>,
    /// Maximum messages before summarization.
    max_messages: usize,
    /// Summarization strategy.
    strategy: SummarizationStrategy,
    /// Previous summaries.
    summaries: Vec<ConversationSummary>,
    /// Optional engine for LLM-based summarization.
    engine: Option<Arc<Engine>>,
    /// Number of messages to summarize at once.
    summarize_batch_size: usize,
}

impl AgentMemory {
    /// Creates a new agent memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 50,
            strategy: SummarizationStrategy::default(),
            summaries: Vec::new(),
            engine: None,
            summarize_batch_size: 10,
        }
    }

    /// Creates with a custom max messages limit.
    #[must_use]
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
            strategy: SummarizationStrategy::default(),
            summaries: Vec::new(),
            engine: None,
            summarize_batch_size: 10,
        }
    }

    /// Sets the summarization strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SummarizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the engine for LLM-based summarization.
    #[must_use]
    pub fn with_engine(mut self, engine: Arc<Engine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Sets the batch size for summarization.
    #[must_use]
    pub fn with_summarize_batch_size(mut self, size: usize) -> Self {
        self.summarize_batch_size = size;
        self
    }

    /// Adds a message to memory.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);

        // Check if we need to manage memory
        if self.messages.len() > self.max_messages {
            self.manage_memory_sync();
        }
    }

    /// Adds a message and performs async summarization if needed.
    pub async fn add_message_async(&mut self, message: Message) -> Result<()> {
        self.messages.push(message);

        if self.messages.len() > self.max_messages {
            self.manage_memory().await?;
        }

        Ok(())
    }

    /// Synchronous memory management (fallback).
    fn manage_memory_sync(&mut self) {
        match self.strategy {
            SummarizationStrategy::DropOldest => {
                self.drop_oldest_messages();
            }
            SummarizationStrategy::Summarize => {
                // Fall back to drop oldest if no engine available
                if self.engine.is_none() {
                    self.drop_oldest_messages();
                } else {
                    // Will be handled by async version
                    self.drop_oldest_messages();
                }
            }
            SummarizationStrategy::SlidingWindow { keep_recent } => {
                self.apply_sliding_window(keep_recent);
            }
        }
    }

    /// Async memory management with LLM summarization.
    async fn manage_memory(&mut self) -> Result<()> {
        match self.strategy {
            SummarizationStrategy::DropOldest => {
                self.drop_oldest_messages();
            }
            SummarizationStrategy::Summarize => {
                if let Some(engine) = &self.engine {
                    self.summarize_messages(engine.clone()).await?;
                } else {
                    self.drop_oldest_messages();
                }
            }
            SummarizationStrategy::SlidingWindow { keep_recent } => {
                self.apply_sliding_window(keep_recent);
            }
        }
        Ok(())
    }

    /// Drops oldest non-system messages.
    fn drop_oldest_messages(&mut self) {
        // Keep system messages and recent messages
        let to_remove = self.messages.len().saturating_sub(self.max_messages);

        if to_remove > 0 {
            // Find and remove oldest non-system messages
            let mut removed = 0;
            self.messages.retain(|m| {
                if removed >= to_remove {
                    return true;
                }
                if matches!(m.role, Role::System) {
                    return true;
                }
                removed += 1;
                false
            });
        }
    }

    /// Applies a sliding window strategy.
    fn apply_sliding_window(&mut self, keep_recent: usize) {
        if self.messages.len() <= keep_recent {
            return;
        }

        // Separate system messages
        let system_messages: Vec<_> = self.messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .cloned()
            .collect();

        // Get recent non-system messages
        let non_system: Vec<_> = self.messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .cloned()
            .collect();

        let keep_count = keep_recent.min(non_system.len());
        let recent: Vec<_> = non_system
            .into_iter()
            .rev()
            .take(keep_count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Rebuild messages
        self.messages = system_messages;
        self.messages.extend(recent);
    }

    /// Summarizes older messages using LLM.
    async fn summarize_messages(&mut self, engine: Arc<Engine>) -> Result<()> {
        // Find non-system messages to summarize
        let non_system_indices: Vec<_> = self.messages
            .iter()
            .enumerate()
            .filter(|(_, m)| !matches!(m.role, Role::System))
            .map(|(i, _)| i)
            .collect();

        if non_system_indices.len() <= self.summarize_batch_size {
            return Ok(());
        }

        // Take oldest batch for summarization
        let summarize_count = self.summarize_batch_size.min(
            non_system_indices.len().saturating_sub(self.max_messages / 2)
        );

        if summarize_count == 0 {
            return Ok(());
        }

        let indices_to_summarize: Vec<_> = non_system_indices
            .iter()
            .take(summarize_count)
            .copied()
            .collect();

        // Extract messages to summarize
        let messages_to_summarize: Vec<_> = indices_to_summarize
            .iter()
            .filter_map(|&i| self.messages.get(i).cloned())
            .collect();

        // Generate summary using LLM
        let summary_text = self.generate_summary(&engine, &messages_to_summarize).await?;

        // Store the summary
        let summary = ConversationSummary {
            text: summary_text.clone(),
            message_count: messages_to_summarize.len(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.summaries.push(summary);

        // Remove summarized messages (in reverse order to preserve indices)
        for &idx in indices_to_summarize.iter().rev() {
            if idx < self.messages.len() {
                self.messages.remove(idx);
            }
        }

        // Insert summary as a system message after the original system prompt
        let insert_pos = self.messages
            .iter()
            .position(|m| !matches!(m.role, Role::System))
            .unwrap_or(self.messages.len());

        self.messages.insert(insert_pos, Message {
            role: Role::System,
            content: format!("[Previous conversation summary: {}]", summary_text),
            name: Some("memory_summary".to_string()),
            tool_call_id: None,
        });

        tracing::debug!(
            summarized_count = messages_to_summarize.len(),
            remaining_messages = self.messages.len(),
            "Summarized conversation history"
        );

        Ok(())
    }

    /// Generates a summary of messages using the LLM.
    async fn generate_summary(&self, engine: &Engine, messages: &[Message]) -> Result<String> {
        // Format messages for summarization
        let conversation = messages
            .iter()
            .map(|m| format!("{}: {}", role_to_str(&m.role), m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Summarize the following conversation in a concise paragraph.
Focus on:
- Key information exchanged
- Decisions made
- Important context for continuing the conversation

Conversation:
{}

Summary:"#,
            conversation
        );

        let request = GenerateRequest::new(prompt)
            .with_sampling(SamplingParams::default()
                .with_max_tokens(256)
                .with_temperature(0.3));

        let response = engine.generate(request).await?;
        let summary = response.choices.first()
            .map(|c| c.text.trim().to_string())
            .unwrap_or_else(|| "Previous conversation context.".to_string());

        Ok(summary)
    }

    /// Returns all messages including summary context.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns all messages without summary messages.
    #[must_use]
    pub fn messages_without_summaries(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.name.as_deref() != Some("memory_summary"))
            .collect()
    }

    /// Returns the conversation summaries.
    #[must_use]
    pub fn summaries(&self) -> &[ConversationSummary] {
        &self.summaries
    }

    /// Clears non-system messages.
    pub fn clear(&mut self) {
        self.messages.retain(|m| matches!(m.role, Role::System));
        self.summaries.clear();
    }

    /// Clears all messages including system messages.
    pub fn clear_all(&mut self) {
        self.messages.clear();
        self.summaries.clear();
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

    /// Returns the total number of messages ever processed (including summarized).
    #[must_use]
    pub fn total_processed(&self) -> usize {
        let summarized: usize = self.summaries.iter().map(|s| s.message_count).sum();
        self.messages.len() + summarized
    }

    /// Gets the context window size estimate (for token budgeting).
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        // Rough estimate: ~4 chars per token
        self.messages
            .iter()
            .map(|m| m.content.len() / 4)
            .sum()
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a role to string for display.
fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_basic() {
        let mut memory = AgentMemory::new();
        assert!(memory.is_empty());

        memory.add_message(Message::user("Hello"));
        assert_eq!(memory.len(), 1);

        memory.add_message(Message::assistant("Hi there!"));
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_memory_drop_oldest() {
        let mut memory = AgentMemory::with_max_messages(3);

        memory.add_message(Message::user("Message 1"));
        memory.add_message(Message::assistant("Response 1"));
        memory.add_message(Message::user("Message 2"));
        assert_eq!(memory.len(), 3);

        // Adding a 4th message should trigger drop
        memory.add_message(Message::assistant("Response 2"));
        assert_eq!(memory.len(), 3);

        // The oldest message should be dropped
        assert_eq!(memory.messages()[0].content, "Response 1");
    }

    #[test]
    fn test_memory_preserves_system() {
        let mut memory = AgentMemory::with_max_messages(2);

        memory.add_message(Message {
            role: Role::System,
            content: "System prompt".to_string(),
            name: None,
            tool_call_id: None,
        });
        memory.add_message(Message::user("User 1"));
        memory.add_message(Message::user("User 2"));
        memory.add_message(Message::user("User 3"));

        // System message should be preserved
        assert!(memory.messages().iter().any(|m| m.content == "System prompt"));
    }

    #[test]
    fn test_sliding_window() {
        let mut memory = AgentMemory::with_max_messages(5)
            .with_strategy(SummarizationStrategy::SlidingWindow { keep_recent: 2 });

        for i in 0..10 {
            memory.add_message(Message::user(format!("Message {}", i)));
        }

        // Should keep only the most recent 2 non-system messages
        let non_system: Vec<_> = memory.messages()
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .collect();

        assert!(non_system.len() <= 2);
    }

    #[test]
    fn test_clear() {
        let mut memory = AgentMemory::new();

        memory.add_message(Message {
            role: Role::System,
            content: "System".to_string(),
            name: None,
            tool_call_id: None,
        });
        memory.add_message(Message::user("User"));
        memory.add_message(Message::assistant("Assistant"));

        memory.clear();

        // Only system message should remain
        assert_eq!(memory.len(), 1);
        assert!(matches!(memory.messages()[0].role, Role::System));
    }

    #[test]
    fn test_estimated_tokens() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("Hello world")); // ~3 tokens
        memory.add_message(Message::assistant("Hi there friend")); // ~4 tokens

        let estimate = memory.estimated_tokens();
        assert!(estimate > 0);
        assert!(estimate < 20); // Should be a reasonable small number
    }
}
