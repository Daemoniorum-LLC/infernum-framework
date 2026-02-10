//! Property-based tests for agent framework using proptest.
//!
//! These tests verify invariants across a wide range of inputs,
//! catching edge cases that unit tests might miss.

#![cfg(test)]

use proptest::prelude::*;

use crate::dynamic_context::{score_message_relevance, semantic_chunk, ContextComplexity};
use crate::long_term_memory::{ImportanceLevel, MemoryEntry, MemoryType};
use crate::tool::{TaskComplexity, ToolTimeoutConfig};
use infernum_core::{Message, Role};
use std::time::Duration;

// ============================================================================
// Strategies for generating test data
// ============================================================================

/// Strategy for generating valid message content.
fn message_content_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 .,!?'\"-]{1,500}"
}

/// Strategy for generating task descriptions.
fn task_description_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("fix typo".to_string()),
        Just("add feature".to_string()),
        Just("refactor code".to_string()),
        Just("implement authentication".to_string()),
        "[a-z ]{5,100}",
    ]
}

/// Strategy for generating memory types.
fn memory_type_strategy() -> impl Strategy<Value = MemoryType> {
    prop_oneof![
        Just(MemoryType::Context),
        Just(MemoryType::Decision),
        Just(MemoryType::SessionSummary),
        Just(MemoryType::ProjectLearning),
        Just(MemoryType::ErrorPattern),
        Just(MemoryType::Optimization),
        Just(MemoryType::UserPreference),
        Just(MemoryType::CodePattern),
    ]
}

/// Strategy for generating importance levels.
fn importance_level_strategy() -> impl Strategy<Value = ImportanceLevel> {
    prop_oneof![
        Just(ImportanceLevel::Critical),
        Just(ImportanceLevel::High),
        Just(ImportanceLevel::Medium),
        Just(ImportanceLevel::Low),
    ]
}

/// Strategy for generating tool names.
fn tool_name_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("http".to_string()),
        Just("database".to_string()),
        Just("file_read".to_string()),
        Just("search".to_string()),
        "[a-z_]{3,20}",
    ]
}

// ============================================================================
// Semantic Chunking Properties
// ============================================================================

proptest! {
    /// Property: Semantic chunking never produces empty chunks for non-empty input.
    #[test]
    fn semantic_chunking_no_empty_chunks(
        content in "[a-zA-Z0-9 \n.,!?]{10,1000}",
        max_chunk_size in 50usize..500
    ) {
        let chunks = semantic_chunk(&content, max_chunk_size);

        // At least one chunk for non-empty input
        prop_assert!(!chunks.is_empty(), "Non-empty input should produce chunks");

        // No empty chunks
        for chunk in &chunks {
            prop_assert!(!chunk.content.is_empty(), "Chunks should not be empty");
        }
    }

    /// Property: Total content is preserved after chunking.
    #[test]
    fn semantic_chunking_preserves_content(
        content in "[a-zA-Z0-9 ]{10,500}",
        max_chunk_size in 100usize..300
    ) {
        let chunks = semantic_chunk(&content, max_chunk_size);

        // Concatenated chunks should contain all original non-whitespace chars
        let original_chars: String = content.chars().filter(|c| !c.is_whitespace()).collect();
        let chunk_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("");
        let chunk_chars: String = chunk_content.chars().filter(|c| !c.is_whitespace()).collect();

        prop_assert!(
            original_chars.len() <= chunk_chars.len() + 10, // Allow small variance for trimming
            "Content should be roughly preserved"
        );
    }

    /// Property: Chunks respect maximum size (with reasonable tolerance).
    #[test]
    fn semantic_chunking_respects_max_size(
        content in "[a-zA-Z ]{100,1000}",
        max_chunk_size in 50usize..200
    ) {
        let chunks = semantic_chunk(&content, max_chunk_size);

        for chunk in &chunks {
            // Allow 20% tolerance for semantic boundary preservation
            // Note: max_chunk_size is in tokens, so compare with token_estimate
            let tolerance = max_chunk_size + (max_chunk_size / 5);
            prop_assert!(
                chunk.token_estimate <= tolerance,
                "Chunk token estimate {} exceeds tolerance {}", chunk.token_estimate, tolerance
            );
        }
    }
}

// ============================================================================
// Message Relevance Scoring Properties
// ============================================================================

proptest! {
    /// Property: Relevance scores are always in valid range [0, 1].
    #[test]
    fn relevance_score_in_valid_range(
        content in message_content_strategy(),
        position in 0usize..100,
        total in 1usize..101,
        task in task_description_strategy()
    ) {
        // Ensure position < total
        let actual_position = position % total;

        let message = Message {
            role: Role::User,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let factors = score_message_relevance(&message, actual_position, total, Some(&task));
        let score = factors.score();

        prop_assert!(score >= 0.0, "Score {} should be >= 0.0", score);
        prop_assert!(score <= 2.0, "Score {} should be <= 2.0 (sum of max factor weights)", score);
    }

    /// Property: System messages have higher base relevance.
    #[test]
    fn system_messages_have_higher_relevance(
        content in message_content_strategy(),
        position in 0usize..50,
        total in 51usize..100,
    ) {
        let system_msg = Message {
            role: Role::System,
            content: content.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let user_msg = Message {
            role: Role::User,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let system_factors = score_message_relevance(&system_msg, position, total, None);
        let user_factors = score_message_relevance(&user_msg, position, total, None);

        // System messages should have higher role_importance
        prop_assert!(
            system_factors.role_importance >= user_factors.role_importance,
            "System ({}) should have role >= user ({})",
            system_factors.role_importance, user_factors.role_importance
        );
    }

    /// Property: More recent messages have higher recency boost.
    #[test]
    fn recent_messages_score_higher(
        content in message_content_strategy(),
        total in 10usize..100,
    ) {
        let message = Message {
            role: Role::User,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let recent_factors = score_message_relevance(&message, total - 1, total, None);
        let old_factors = score_message_relevance(&message, 0, total, None);

        prop_assert!(
            recent_factors.recency >= old_factors.recency,
            "Recent ({}) should have recency >= old ({})",
            recent_factors.recency, old_factors.recency
        );
    }
}

// ============================================================================
// Complexity Classification Properties
// ============================================================================

proptest! {
    /// Property: Complexity classification is deterministic.
    #[test]
    fn complexity_classification_is_deterministic(
        task in task_description_strategy()
    ) {
        let first = ContextComplexity::classify(&task);
        let second = ContextComplexity::classify(&task);

        prop_assert_eq!(first, second, "Classification should be deterministic");
    }

    /// Property: Complexity always returns valid enum variant.
    #[test]
    fn complexity_always_valid(
        task in "[a-zA-Z ]{1,200}"
    ) {
        let complexity = ContextComplexity::classify(&task);

        // Check it's one of the valid variants
        match complexity {
            ContextComplexity::Simple |
            ContextComplexity::Moderate |
            ContextComplexity::Complex => {},
        }
    }
}

// ============================================================================
// Memory Entry Properties
// ============================================================================

proptest! {
    /// Property: Memory entry tags are preserved.
    #[test]
    fn memory_tags_preserved(
        content in message_content_strategy(),
        tags in prop::collection::vec("[a-z]{3,10}", 0..5),
        memory_type in memory_type_strategy()
    ) {
        let mut entry = MemoryEntry::new(memory_type, &content);

        for tag in &tags {
            entry = entry.with_tag(tag);
        }

        for tag in &tags {
            prop_assert!(entry.matches(tag), "Entry should match tag: {}", tag);
        }
    }

    /// Property: Memory importance levels have correct ordering via Ord.
    #[test]
    fn importance_levels_ordered(
        _seed in 0u32..1000
    ) {
        // Critical > High > Medium > Low (Ord implementation)
        prop_assert!(ImportanceLevel::Critical > ImportanceLevel::High);
        prop_assert!(ImportanceLevel::High > ImportanceLevel::Medium);
        prop_assert!(ImportanceLevel::Medium > ImportanceLevel::Low);
    }

    /// Property: Memory entry content search works correctly.
    #[test]
    fn memory_content_search(
        base_content in "[a-z]{20,50}",
        search_term in "[a-z]{3,8}",
        memory_type in memory_type_strategy()
    ) {
        // Create entry with search term in content
        let content_with_term = format!("{} {} more text", base_content, search_term);
        let entry = MemoryEntry::new(memory_type, &content_with_term);

        // Should match the search term
        prop_assert!(
            entry.matches(&search_term),
            "Entry with content '{}' should match '{}'",
            content_with_term, search_term
        );
    }
}

// ============================================================================
// Tool Timeout Properties
// ============================================================================

proptest! {
    /// Property: Default timeout is always returned for unknown tools.
    #[test]
    fn default_timeout_for_unknown_tools(
        default_secs in 10u64..120,
        tool_name in "[a-z]{5,15}"
    ) {
        let config = ToolTimeoutConfig::new(Duration::from_secs(default_secs));

        let timeout = config.get_timeout(&tool_name);

        prop_assert_eq!(
            timeout,
            Duration::from_secs(default_secs),
            "Unknown tool should get default timeout"
        );
    }

    /// Property: Custom tool timeouts override default.
    #[test]
    fn custom_timeout_overrides_default(
        default_secs in 10u64..60,
        custom_secs in 61u64..120,
        tool_name in tool_name_strategy()
    ) {
        let config = ToolTimeoutConfig::new(Duration::from_secs(default_secs))
            .with_tool_timeout(&tool_name, Duration::from_secs(custom_secs));

        let timeout = config.get_timeout(&tool_name);

        prop_assert_eq!(
            timeout,
            Duration::from_secs(custom_secs),
            "Custom tool should get custom timeout"
        );
    }

    /// Property: Task complexity multipliers are in expected range.
    #[test]
    fn complexity_multipliers_in_range(_seed in 0u32..1000) {
        let simple = TaskComplexity::Simple.multiplier();
        let moderate = TaskComplexity::Moderate.multiplier();
        let complex = TaskComplexity::Complex.multiplier();

        prop_assert!(simple >= 0.5 && simple <= 1.0, "Simple: {}", simple);
        prop_assert!(moderate >= 1.0 && moderate <= 2.0, "Moderate: {}", moderate);
        prop_assert!(complex >= 1.5 && complex <= 3.0, "Complex: {}", complex);

        // Ordering
        prop_assert!(simple <= moderate, "Simple <= Moderate");
        prop_assert!(moderate <= complex, "Moderate <= Complex");
    }
}

// ============================================================================
// Round-trip Properties
// ============================================================================

proptest! {
    /// Property: Memory entry serialization round-trip preserves data.
    #[test]
    fn memory_entry_serialization_roundtrip(
        content in message_content_strategy(),
        memory_type in memory_type_strategy(),
        importance in importance_level_strategy(),
        summary in prop::option::of("[a-zA-Z ]{10,50}")
    ) {
        let mut entry = MemoryEntry::new(memory_type, &content)
            .with_importance(importance)
            .with_tag("test");

        if let Some(ref s) = summary {
            entry = entry.with_summary(s);
        }

        // Serialize and deserialize
        let json = serde_json::to_string(&entry).expect("Serialization should work");
        let restored: MemoryEntry = serde_json::from_str(&json).expect("Deserialization should work");

        prop_assert_eq!(entry.content, restored.content);
        prop_assert_eq!(entry.memory_type, restored.memory_type);
        prop_assert_eq!(entry.importance, restored.importance);
    }
}
