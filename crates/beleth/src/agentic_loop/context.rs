//! Context window management for the agentic loop.
//!
//! Manages the conversation context, applying compression strategies when
//! the token budget gets tight. Key insight from the spec: when compressing,
//! preserve the *shape* of exploration, not just the final path.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §2.3.

use infernum_core::Message;

use super::types::*;

// ---------------------------------------------------------------------------
// Context window manager
// ---------------------------------------------------------------------------

/// Manages the context window for an agentic loop session.
///
/// Tracks messages, token estimates, and applies compression strategies
/// to keep the context within the token budget.
#[derive(Debug)]
pub struct ContextWindowManager {
    /// Current messages in the context.
    messages: Vec<Message>,
    /// Current loop state snapshot for injection.
    state_snapshot: LoopStateSnapshot,
    /// Estimated total tokens in the current context.
    estimated_tokens: u32,
    /// Maximum tokens allowed in the context window.
    max_context_tokens: u32,
    /// Compression events applied so far.
    compressions: Vec<CompressionEvent>,
    /// Original token count before any compression.
    original_tokens: u32,
}

impl ContextWindowManager {
    /// Creates a new context window manager.
    pub fn new(max_context_tokens: u32) -> Self {
        Self {
            messages: Vec::new(),
            state_snapshot: LoopStateSnapshot {
                iteration: 0,
                max_iterations: 0,
                token_budget_remaining: max_context_tokens,
                tools_available: Vec::new(),
                context_pressure: 0.0,
            },
            estimated_tokens: 0,
            max_context_tokens,
            compressions: Vec::new(),
            original_tokens: 0,
        }
    }

    /// Returns the current messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns the current estimated token count.
    pub fn estimated_tokens(&self) -> u32 {
        self.estimated_tokens
    }

    /// Returns the context pressure (0.0–1.0).
    pub fn pressure(&self) -> f32 {
        if self.max_context_tokens == 0 {
            return 1.0;
        }
        (self.estimated_tokens as f32 / self.max_context_tokens as f32).clamp(0.0, 1.0)
    }

    /// Returns whether the context is under high pressure (>80% full).
    pub fn is_under_pressure(&self) -> bool {
        self.pressure() > 0.8
    }

    /// Returns all compression events.
    pub fn compressions(&self) -> &[CompressionEvent] {
        &self.compressions
    }

    /// Updates the loop state snapshot.
    pub fn update_state(&mut self, snapshot: LoopStateSnapshot) {
        self.state_snapshot = snapshot;
    }

    /// Sets the initial messages (system prompt + user objective).
    pub fn set_initial_messages(&mut self, messages: Vec<Message>) {
        self.estimated_tokens = estimate_tokens_for_messages(&messages);
        self.original_tokens = self.estimated_tokens;
        self.messages = messages;
    }

    /// Adds a message to the context.
    pub fn push_message(&mut self, message: Message) {
        let tokens = estimate_tokens(&message.content);
        self.estimated_tokens = self.estimated_tokens.saturating_add(tokens);
        self.original_tokens = self.original_tokens.saturating_add(tokens);
        self.messages.push(message);
    }

    /// Attempts to compress the context using the given strategy.
    ///
    /// Returns the number of tokens saved, or 0 if compression was not applicable.
    pub fn compress(&mut self, strategy: &CompressionStrategy, iteration: u32) -> u32 {
        let before = self.estimated_tokens;

        match strategy {
            CompressionStrategy::SummarizeOldResults { keep_recent } => {
                self.summarize_old_results(*keep_recent);
            },
            CompressionStrategy::PruneDeadEnds => {
                self.prune_dead_ends();
            },
            CompressionStrategy::CollapseExploration { summary_tokens } => {
                self.collapse_exploration(*summary_tokens);
            },
            CompressionStrategy::AgentDirected => {
                // Agent-directed compression requires LLM call — not handled here.
                // The executor should call this separately.
            },
        }

        let after = self.estimated_tokens;
        let saved = before.saturating_sub(after);

        if saved > 0 {
            self.compressions.push(CompressionEvent {
                strategy: strategy.clone(),
                tokens_saved: saved,
                at_iteration: iteration,
            });
        }

        saved
    }

    /// Builds a `ContextWindow` snapshot for serialization.
    pub fn snapshot(&self) -> ContextWindow {
        let context_messages: Vec<ContextMessage> = self
            .messages
            .iter()
            .map(|m| ContextMessage {
                role: format!("{:?}", m.role).to_lowercase(),
                content: m.content.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();

        ContextWindow {
            messages: context_messages,
            system_state: self.state_snapshot.clone(),
            original_token_count: self.original_tokens,
            current_token_count: self.estimated_tokens,
            compressions_applied: self.compressions.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Compression strategies
    // -----------------------------------------------------------------------

    /// Summarize old tool results, keeping the most recent ones verbatim.
    fn summarize_old_results(&mut self, keep_recent: u32) {
        let tool_indices: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.tool_call_id.is_some())
            .map(|(i, _)| i)
            .collect();

        if tool_indices.len() <= keep_recent as usize {
            return; // Nothing to summarize
        }

        let to_summarize = tool_indices.len() - keep_recent as usize;
        let mut summarized_count = 0;

        for &idx in tool_indices.iter().take(to_summarize) {
            let msg = &self.messages[idx];
            let original_len = msg.content.len();

            // Create a truncated summary
            let summary = if original_len > 200 {
                format!(
                    "[Summarized] {}... ({} chars truncated)",
                    &msg.content[..200.min(msg.content.len())],
                    original_len - 200
                )
            } else {
                continue; // Already short enough
            };

            let old_tokens = estimate_tokens(&self.messages[idx].content);
            self.messages[idx].content = summary;
            let new_tokens = estimate_tokens(&self.messages[idx].content);
            self.estimated_tokens = self
                .estimated_tokens
                .saturating_sub(old_tokens)
                .saturating_add(new_tokens);
            summarized_count += 1;
        }

        if summarized_count > 0 {
            tracing::debug!(summarized_count, "Summarized old tool results");
        }
    }

    /// Remove tool result messages that contributed nothing (empty or error).
    fn prune_dead_ends(&mut self) {
        let original_len = self.messages.len();

        // Find tool results that were errors or empty — but preserve the
        // *most recent* cycle so the model knows what just happened.
        let last_assistant_idx = self
            .messages
            .iter()
            .rposition(|m| matches!(m.role, infernum_core::Role::Assistant));

        let cutoff = last_assistant_idx.unwrap_or(0);

        let mut indices_to_remove = Vec::new();
        for (i, msg) in self.messages.iter().enumerate() {
            if i >= cutoff {
                break;
            }
            if msg.tool_call_id.is_some() && is_dead_end_result(&msg.content) {
                indices_to_remove.push(i);
            }
        }

        // Remove in reverse order to preserve indices
        let mut tokens_freed = 0u32;
        for &idx in indices_to_remove.iter().rev() {
            tokens_freed =
                tokens_freed.saturating_add(estimate_tokens(&self.messages[idx].content));
            self.messages.remove(idx);
        }

        self.estimated_tokens = self.estimated_tokens.saturating_sub(tokens_freed);

        if self.messages.len() < original_len {
            tracing::debug!(
                pruned = original_len - self.messages.len(),
                tokens_freed,
                "Pruned dead-end tool results"
            );
        }
    }

    /// Collapse exploration branches into summaries.
    fn collapse_exploration(&mut self, _summary_tokens: u32) {
        // Identify sequences of assistant+tool messages that form exploration branches.
        // For now, use a simple heuristic: if there are more than 6 tool result messages
        // before the last 4, collapse the older ones into a single summary message.

        let tool_result_count = self
            .messages
            .iter()
            .filter(|m| m.tool_call_id.is_some())
            .count();

        if tool_result_count <= 6 {
            return;
        }

        // Find old tool result messages (all except the last 4)
        let tool_indices: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.tool_call_id.is_some())
            .map(|(i, _)| i)
            .collect();

        let keep_count = 4.min(tool_indices.len());
        let to_collapse = &tool_indices[..tool_indices.len() - keep_count];

        if to_collapse.is_empty() {
            return;
        }

        // Build a summary of collapsed results
        let mut summary_parts = Vec::new();
        let mut tokens_freed = 0u32;

        for &idx in to_collapse {
            let msg = &self.messages[idx];
            let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
            let snippet = if msg.content.len() > 80 {
                format!("{}...", &msg.content[..80])
            } else {
                msg.content.clone()
            };
            summary_parts.push(format!("- {tool_id}: {snippet}"));
            tokens_freed = tokens_freed.saturating_add(estimate_tokens(&msg.content));
        }

        let summary = format!(
            "[Exploration summary — {} earlier tool results collapsed]\n{}",
            to_collapse.len(),
            summary_parts.join("\n")
        );

        // Remove old messages and insert summary
        for &idx in to_collapse.iter().rev() {
            self.messages.remove(idx);
        }

        let summary_tokens = estimate_tokens(&summary);
        tokens_freed = tokens_freed.saturating_sub(summary_tokens);
        self.estimated_tokens = self.estimated_tokens.saturating_sub(tokens_freed);

        // Insert summary after the system message
        let insert_pos = 1.min(self.messages.len());
        self.messages.insert(insert_pos, Message::system(summary));

        tracing::debug!(
            collapsed = to_collapse.len(),
            tokens_freed,
            "Collapsed exploration branches"
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough token estimate: ~4 characters per token (English text heuristic).
fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32 / 4).max(1)
}

/// Estimate tokens for a list of messages.
fn estimate_tokens_for_messages(messages: &[Message]) -> u32 {
    messages
        .iter()
        .map(|m| estimate_tokens(&m.content) + 4) // +4 for role tokens and separators
        .sum()
}

/// Returns true if a tool result looks like a dead-end (error or empty).
fn is_dead_end_result(content: &str) -> bool {
    let lower = content.to_lowercase();
    lower.starts_with("error:")
        || lower.starts_with("no results")
        || lower.starts_with("not found")
        || lower.is_empty()
        || lower == "null"
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_core::Message;

    #[test]
    fn test_new_context_manager() {
        let mgr = ContextWindowManager::new(4096);
        assert_eq!(mgr.estimated_tokens(), 0);
        assert_eq!(mgr.pressure(), 0.0);
        assert!(!mgr.is_under_pressure());
    }

    #[test]
    fn test_push_message_updates_tokens() {
        let mut mgr = ContextWindowManager::new(4096);
        mgr.push_message(Message::user("Hello, world!"));
        assert!(mgr.estimated_tokens() > 0);
    }

    #[test]
    fn test_pressure_calculation() {
        let mut mgr = ContextWindowManager::new(100);
        // Add ~90 tokens worth of content (360 chars ≈ 90 tokens)
        mgr.push_message(Message::user(&"x".repeat(360)));
        assert!(mgr.pressure() > 0.8);
        assert!(mgr.is_under_pressure());
    }

    #[test]
    fn test_set_initial_messages() {
        let mut mgr = ContextWindowManager::new(4096);
        mgr.set_initial_messages(vec![
            Message::system("You are a helper."),
            Message::user("Do something."),
        ]);
        assert_eq!(mgr.messages().len(), 2);
        assert!(mgr.estimated_tokens() > 0);
    }

    #[test]
    fn test_summarize_old_results_keeps_recent() {
        let mut mgr = ContextWindowManager::new(10000);
        mgr.push_message(Message::system("sys"));

        // Add 5 tool results with long content
        for i in 0..5 {
            let mut msg = Message::tool_result(format!("call_{i}"), &"x".repeat(500));
            msg.tool_call_id = Some(format!("call_{i}"));
            mgr.push_message(msg);
        }

        let saved = mgr.compress(
            &CompressionStrategy::SummarizeOldResults { keep_recent: 2 },
            1,
        );

        // Should have summarized 3 of 5 results
        assert!(saved > 0);
        // Recent 2 should still have full content
        let tool_msgs: Vec<_> = mgr
            .messages()
            .iter()
            .filter(|m| m.tool_call_id.is_some())
            .collect();
        assert_eq!(tool_msgs.len(), 5); // All still present, just summarized
                                        // Last 2 should be full length
        assert!(tool_msgs[3].content.len() >= 500);
        assert!(tool_msgs[4].content.len() >= 500);
    }

    #[test]
    fn test_prune_dead_ends() {
        let mut mgr = ContextWindowManager::new(10000);
        mgr.push_message(Message::system("sys"));

        // Add a successful tool result
        let mut good = Message::tool_result("call_good", "Found 42 results");
        good.tool_call_id = Some("call_good".to_string());
        mgr.push_message(good);

        // Add a dead-end tool result
        let mut bad = Message::tool_result("call_bad", "Error: file not found");
        bad.tool_call_id = Some("call_bad".to_string());
        mgr.push_message(bad);

        // Add a recent assistant message (protects results after it)
        mgr.push_message(Message::assistant("Let me try something else."));

        let saved = mgr.compress(&CompressionStrategy::PruneDeadEnds, 1);
        assert!(saved > 0);
        // The dead-end result should be removed
        assert_eq!(
            mgr.messages()
                .iter()
                .filter(|m| m.tool_call_id.is_some())
                .count(),
            1 // only the good one remains
        );
    }

    #[test]
    fn test_snapshot() {
        let mut mgr = ContextWindowManager::new(4096);
        mgr.set_initial_messages(vec![Message::system("sys"), Message::user("do")]);
        let snap = mgr.snapshot();
        assert_eq!(snap.messages.len(), 2);
        assert_eq!(snap.original_token_count, snap.current_token_count);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 1); // min 1
        assert_eq!(estimate_tokens("abcd"), 1); // 4 chars = 1 token
        assert_eq!(estimate_tokens(&"a".repeat(100)), 25); // 100/4
    }

    #[test]
    fn test_is_dead_end_result() {
        assert!(is_dead_end_result("Error: something went wrong"));
        assert!(is_dead_end_result("No results found"));
        assert!(is_dead_end_result("Not found"));
        assert!(is_dead_end_result(""));
        assert!(!is_dead_end_result("Found 42 items"));
    }
}
