//! Dynamic context management with semantic chunking and adaptive budgeting.
//!
//! Inspired by Leviathan/Persona's `DynamicContextManager` that provides:
//! - Intelligent context window optimization
//! - Message relevance scoring
//! - Semantic chunking for large content
//! - Task complexity classification
//! - Adaptive token budgeting

use infernum_core::{Message, Role};
use serde::{Deserialize, Serialize};

// ============================================================================
// Task Complexity Classification
// ============================================================================

/// Task complexity level for adaptive context sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextComplexity {
    /// Simple tasks: quick responses, minimal context needed.
    Simple,
    /// Moderate tasks: standard context requirements.
    Moderate,
    /// Complex tasks: full context window needed.
    Complex,
}

impl ContextComplexity {
    /// Returns the input token budget for this complexity.
    #[must_use]
    pub fn input_budget(&self) -> usize {
        match self {
            Self::Simple => 20_000,
            Self::Moderate => 60_000,
            Self::Complex => 100_000,
        }
    }

    /// Returns the output token budget for this complexity.
    #[must_use]
    pub fn output_budget(&self) -> usize {
        match self {
            Self::Simple => 2_000,
            Self::Moderate => 4_000,
            Self::Complex => 8_000,
        }
    }

    /// Classifies complexity based on task description.
    #[must_use]
    pub fn classify(task: &str) -> Self {
        let task_lower = task.to_lowercase();

        // Complex indicators
        let complex_keywords = [
            "refactor", "redesign", "architect", "migrate", "implement feature",
            "full system", "end-to-end", "comprehensive", "multi-step",
        ];

        // Simple indicators
        let simple_keywords = [
            "fix typo", "rename", "add comment", "simple", "quick", "small",
            "one line", "minor", "trivial",
        ];

        if complex_keywords.iter().any(|k| task_lower.contains(k)) {
            return Self::Complex;
        }

        if simple_keywords.iter().any(|k| task_lower.contains(k)) {
            return Self::Simple;
        }

        Self::Moderate
    }
}

// ============================================================================
// Message Relevance Scoring
// ============================================================================

/// Factors contributing to message relevance.
#[derive(Debug, Clone, Default)]
pub struct RelevanceFactors {
    /// Recency score (0.0 - 1.0).
    pub recency: f32,
    /// Role importance (system > assistant > user).
    pub role_importance: f32,
    /// Contains tool calls.
    pub has_tool_calls: f32,
    /// Contains error indicators.
    pub has_errors: f32,
    /// Contains code blocks.
    pub has_code: f32,
    /// Semantic similarity to current task.
    pub semantic_similarity: f32,
}

impl RelevanceFactors {
    /// Calculates the overall relevance score.
    #[must_use]
    pub fn score(&self) -> f32 {
        // Weighted combination
        self.recency * 0.25
            + self.role_importance * 0.20
            + self.has_tool_calls * 0.15
            + self.has_errors * 0.15
            + self.has_code * 0.10
            + self.semantic_similarity * 0.15
    }
}

/// Scores a message's relevance to the current context.
pub fn score_message_relevance(
    message: &Message,
    index: usize,
    total_messages: usize,
    current_task: Option<&str>,
) -> RelevanceFactors {
    let mut factors = RelevanceFactors::default();

    // Recency: more recent = higher score
    factors.recency = if total_messages > 0 {
        index as f32 / total_messages as f32
    } else {
        1.0
    };

    // Role importance
    factors.role_importance = match message.role {
        Role::System => 1.0,
        Role::Assistant => 0.8,
        Role::User => 0.7,
        Role::Tool => 0.6,
    };

    let content_lower = message.content.to_lowercase();

    // Tool call detection
    if content_lower.contains("tool") || content_lower.contains("function") {
        factors.has_tool_calls = 0.8;
    }

    // Error detection
    if content_lower.contains("error")
        || content_lower.contains("failed")
        || content_lower.contains("exception")
    {
        factors.has_errors = 1.0;
    }

    // Code detection
    if message.content.contains("```") || message.content.contains("fn ")
        || message.content.contains("function") || message.content.contains("class ")
    {
        factors.has_code = 0.7;
    }

    // Semantic similarity (simple keyword matching)
    if let Some(task) = current_task {
        let task_words: Vec<&str> = task.split_whitespace().collect();
        let matches = task_words
            .iter()
            .filter(|w| content_lower.contains(&w.to_lowercase()))
            .count();
        factors.semantic_similarity = (matches as f32 / task_words.len().max(1) as f32).min(1.0);
    }

    factors
}

// ============================================================================
// Semantic Chunking
// ============================================================================

/// A semantic chunk of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    /// The chunk content.
    pub content: String,
    /// Start position in original content.
    pub start: usize,
    /// End position in original content.
    pub end: usize,
    /// Chunk type.
    pub chunk_type: ChunkType,
    /// Estimated token count.
    pub token_estimate: usize,
}

/// Type of semantic chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkType {
    /// Code block.
    Code,
    /// Prose/text paragraph.
    Prose,
    /// List items.
    List,
    /// Header/title.
    Header,
    /// JSON/structured data.
    Structured,
}

/// Chunks content into semantic units.
pub fn semantic_chunk(content: &str, max_chunk_tokens: usize) -> Vec<SemanticChunk> {
    let mut chunks = Vec::new();
    let mut current_pos = 0;

    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Check for code block
        if line.starts_with("```") {
            let start = current_pos;
            let mut code_content = String::new();
            code_content.push_str(line);
            code_content.push('\n');
            i += 1;

            while i < lines.len() && !lines[i].starts_with("```") {
                code_content.push_str(lines[i]);
                code_content.push('\n');
                i += 1;
            }

            if i < lines.len() {
                code_content.push_str(lines[i]);
                i += 1;
            }

            let end = start + code_content.len();
            chunks.push(SemanticChunk {
                token_estimate: estimate_tokens(&code_content),
                content: code_content,
                start,
                end,
                chunk_type: ChunkType::Code,
            });
            current_pos = end;
            continue;
        }

        // Check for header
        if line.starts_with('#') || line.starts_with("==") || line.starts_with("--") {
            let end = current_pos + line.len() + 1;
            chunks.push(SemanticChunk {
                token_estimate: estimate_tokens(line),
                content: line.to_string(),
                start: current_pos,
                end,
                chunk_type: ChunkType::Header,
            });
            current_pos = end;
            i += 1;
            continue;
        }

        // Check for list item
        if line.starts_with("- ") || line.starts_with("* ") || line.starts_with("1.") {
            let start = current_pos;
            let mut list_content = String::new();

            while i < lines.len() {
                let l = lines[i];
                if l.starts_with("- ") || l.starts_with("* ") || l.starts_with(char::is_numeric) {
                    list_content.push_str(l);
                    list_content.push('\n');
                    i += 1;
                } else if l.trim().is_empty() || l.starts_with("  ") {
                    list_content.push_str(l);
                    list_content.push('\n');
                    i += 1;
                } else {
                    break;
                }
            }

            let end = start + list_content.len();
            chunks.push(SemanticChunk {
                token_estimate: estimate_tokens(&list_content),
                content: list_content,
                start,
                end,
                chunk_type: ChunkType::List,
            });
            current_pos = end;
            continue;
        }

        // Check for JSON/structured data
        if line.trim().starts_with('{') || line.trim().starts_with('[') {
            let start = current_pos;
            let mut struct_content = String::new();
            let mut brace_count = 0;

            while i < lines.len() {
                let l = lines[i];
                brace_count += l.matches('{').count() as i32;
                brace_count += l.matches('[').count() as i32;
                brace_count -= l.matches('}').count() as i32;
                brace_count -= l.matches(']').count() as i32;

                struct_content.push_str(l);
                struct_content.push('\n');
                i += 1;

                if brace_count <= 0 {
                    break;
                }
            }

            let end = start + struct_content.len();
            chunks.push(SemanticChunk {
                token_estimate: estimate_tokens(&struct_content),
                content: struct_content,
                start,
                end,
                chunk_type: ChunkType::Structured,
            });
            current_pos = end;
            continue;
        }

        // Default: prose paragraph
        let start = current_pos;
        let mut prose_content = String::new();

        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty() {
                prose_content.push('\n');
                i += 1;
                break;
            }
            if l.starts_with("```") || l.starts_with('#') || l.starts_with("- ") {
                break;
            }
            prose_content.push_str(l);
            prose_content.push('\n');
            i += 1;
        }

        if !prose_content.trim().is_empty() {
            let end = start + prose_content.len();
            chunks.push(SemanticChunk {
                token_estimate: estimate_tokens(&prose_content),
                content: prose_content,
                start,
                end,
                chunk_type: ChunkType::Prose,
            });
            current_pos = end;
        } else {
            // The while loop already incremented i for empty lines,
            // so only update current_pos here
            current_pos += 1;
        }
    }

    // Split chunks that exceed max size
    let mut final_chunks = Vec::new();
    for chunk in chunks {
        if chunk.token_estimate > max_chunk_tokens {
            final_chunks.extend(split_chunk(&chunk, max_chunk_tokens));
        } else {
            final_chunks.push(chunk);
        }
    }

    final_chunks
}

fn split_chunk(chunk: &SemanticChunk, max_tokens: usize) -> Vec<SemanticChunk> {
    let mut result = Vec::new();
    let words: Vec<&str> = chunk.content.split_whitespace().collect();
    let mut current = String::new();
    let mut current_start = chunk.start;

    for word in words {
        let test = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };

        if estimate_tokens(&test) > max_tokens && !current.is_empty() {
            let end = current_start + current.len();
            result.push(SemanticChunk {
                token_estimate: estimate_tokens(&current),
                content: current.clone(),
                start: current_start,
                end,
                chunk_type: chunk.chunk_type,
            });
            current_start = end;
            current = word.to_string();
        } else {
            current = test;
        }
    }

    if !current.is_empty() {
        result.push(SemanticChunk {
            token_estimate: estimate_tokens(&current),
            content: current.clone(),
            start: current_start,
            end: current_start + current.len(),
            chunk_type: chunk.chunk_type,
        });
    }

    result
}

/// Estimates token count for text (rough approximation).
fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: 1 token ≈ 4 characters for English text
    // Code tends to have more tokens per character
    let char_count = text.len();
    (char_count + 3) / 4
}

// ============================================================================
// Dynamic Context Manager
// ============================================================================

/// Configuration for dynamic context management.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum input tokens.
    pub max_input_tokens: usize,
    /// Maximum output tokens.
    pub max_output_tokens: usize,
    /// Minimum relevance score to keep a message.
    pub min_relevance: f32,
    /// Overlap tokens when truncating.
    pub overlap_tokens: usize,
    /// Maximum chunk size in tokens.
    pub max_chunk_tokens: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_input_tokens: 60_000,
            max_output_tokens: 4_000,
            min_relevance: 0.3,
            overlap_tokens: 100,
            max_chunk_tokens: 2_000,
        }
    }
}

impl ContextConfig {
    /// Creates config for a given complexity level.
    #[must_use]
    pub fn for_complexity(complexity: ContextComplexity) -> Self {
        Self {
            max_input_tokens: complexity.input_budget(),
            max_output_tokens: complexity.output_budget(),
            ..Default::default()
        }
    }
}

/// Dynamic context manager.
pub struct DynamicContextManager {
    config: ContextConfig,
    complexity: ContextComplexity,
    current_task: Option<String>,
}

impl DynamicContextManager {
    /// Creates a new context manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            complexity: ContextComplexity::Moderate,
            current_task: None,
        }
    }

    /// Sets the configuration.
    #[must_use]
    pub fn with_config(mut self, config: ContextConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the task complexity.
    #[must_use]
    pub fn with_complexity(mut self, complexity: ContextComplexity) -> Self {
        self.complexity = complexity;
        self.config = ContextConfig::for_complexity(complexity);
        self
    }

    /// Sets the current task.
    #[must_use]
    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        let task = task.into();
        self.complexity = ContextComplexity::classify(&task);
        self.config = ContextConfig::for_complexity(self.complexity);
        self.current_task = Some(task);
        self
    }

    /// Optimizes messages to fit within the context budget.
    pub fn optimize(&self, messages: &[Message]) -> Vec<Message> {
        let mut scored: Vec<(usize, &Message, f32)> = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let factors = score_message_relevance(
                    msg,
                    i,
                    messages.len(),
                    self.current_task.as_deref(),
                );
                (i, msg, factors.score())
            })
            .collect();

        // Sort by relevance (descending)
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate total tokens and filter
        let mut total_tokens = 0;
        let mut selected: Vec<(usize, Message)> = Vec::new();

        for (idx, msg, score) in scored {
            if score < self.config.min_relevance {
                continue;
            }

            let msg_tokens = estimate_tokens(&msg.content);
            if total_tokens + msg_tokens > self.config.max_input_tokens {
                // Try to truncate the message to fit remaining budget
                if let Some(truncated) = self.truncate_message(msg, self.config.max_input_tokens - total_tokens) {
                    selected.push((idx, truncated));
                }
                break;
            }

            selected.push((idx, msg.clone()));
            total_tokens += msg_tokens;
        }

        // Sort back to original order (preserve chronology)
        selected.sort_by_key(|(idx, _)| *idx);
        selected.into_iter().map(|(_, msg)| msg).collect()
    }

    /// Truncates a message to fit within a token budget.
    fn truncate_message(&self, message: &Message, max_tokens: usize) -> Option<Message> {
        if max_tokens < 50 {
            return None;
        }

        let chunks = semantic_chunk(&message.content, self.config.max_chunk_tokens);

        let mut content = String::new();
        let mut total_tokens = 0;

        for chunk in chunks {
            if total_tokens + chunk.token_estimate > max_tokens {
                break;
            }
            content.push_str(&chunk.content);
            total_tokens += chunk.token_estimate;
        }

        if content.is_empty() {
            return None;
        }

        content.push_str("\n... [truncated]");

        Some(Message {
            role: message.role.clone(),
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        })
    }

    /// Returns the current context budget.
    #[must_use]
    pub fn budget(&self) -> (usize, usize) {
        (self.config.max_input_tokens, self.config.max_output_tokens)
    }

    /// Returns the current complexity.
    #[must_use]
    pub fn complexity(&self) -> ContextComplexity {
        self.complexity
    }
}

impl Default for DynamicContextManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_complexity_budgets() {
        assert_eq!(ContextComplexity::Simple.input_budget(), 20_000);
        assert_eq!(ContextComplexity::Moderate.input_budget(), 60_000);
        assert_eq!(ContextComplexity::Complex.input_budget(), 100_000);

        assert_eq!(ContextComplexity::Simple.output_budget(), 2_000);
        assert_eq!(ContextComplexity::Complex.output_budget(), 8_000);
    }

    #[test]
    fn test_context_complexity_classification() {
        // "fix typo" matches the simple keyword, not "fix a typo"
        assert_eq!(ContextComplexity::classify("fix typo in readme"), ContextComplexity::Simple);
        assert_eq!(ContextComplexity::classify("rename variable"), ContextComplexity::Simple);
        assert_eq!(ContextComplexity::classify("refactor the authentication system"), ContextComplexity::Complex);
        assert_eq!(ContextComplexity::classify("implement feature X"), ContextComplexity::Complex);
        assert_eq!(ContextComplexity::classify("add a function"), ContextComplexity::Moderate);
    }

    #[test]
    fn test_relevance_factors_score() {
        let factors = RelevanceFactors {
            recency: 1.0,
            role_importance: 1.0,
            has_tool_calls: 0.0,
            has_errors: 0.0,
            has_code: 0.0,
            semantic_similarity: 0.0,
        };

        let score = factors.score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_score_message_relevance() {
        let message = Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let factors = score_message_relevance(&message, 0, 10, None);
        assert_eq!(factors.role_importance, 1.0); // System role is highest
    }

    #[test]
    fn test_score_message_with_errors() {
        let message = Message {
            role: Role::Assistant,
            content: "An error occurred while processing.".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let factors = score_message_relevance(&message, 5, 10, None);
        assert_eq!(factors.has_errors, 1.0);
    }

    #[test]
    fn test_score_message_with_code() {
        let message = Message {
            role: Role::Assistant,
            content: "```rust\nfn main() {}\n```".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let factors = score_message_relevance(&message, 5, 10, None);
        assert!(factors.has_code > 0.0);
    }

    #[test]
    fn test_semantic_chunk_code_blocks() {
        let content = "Some text\n```rust\nfn main() {}\n```\nMore text";
        let chunks = semantic_chunk(content, 1000);

        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Code));
        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Prose));
    }

    #[test]
    fn test_semantic_chunk_headers() {
        let content = "# Header 1\n\nSome content\n\n## Header 2\n\nMore content";
        let chunks = semantic_chunk(content, 1000);

        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Header));
    }

    #[test]
    fn test_semantic_chunk_lists() {
        let content = "- Item 1\n- Item 2\n- Item 3";
        let chunks = semantic_chunk(content, 1000);

        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::List));
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "Hello, world!";
        let tokens = estimate_tokens(text);
        assert!(tokens > 0);
        assert!(tokens < text.len()); // Should be less than char count
    }

    #[test]
    fn test_dynamic_context_manager_with_task() {
        let manager = DynamicContextManager::new()
            .with_task("refactor the entire codebase");

        assert_eq!(manager.complexity(), ContextComplexity::Complex);
        let (input, _output) = manager.budget();
        assert_eq!(input, 100_000);
    }

    #[test]
    fn test_dynamic_context_manager_optimize() {
        let manager = DynamicContextManager::new()
            .with_complexity(ContextComplexity::Simple);

        let messages = vec![
            Message { role: Role::System, content: "System prompt".to_string(), name: None, tool_calls: None, tool_call_id: None },
            Message { role: Role::User, content: "User message".to_string(), name: None, tool_calls: None, tool_call_id: None },
            Message { role: Role::Assistant, content: "Response".to_string(), name: None, tool_calls: None, tool_call_id: None },
        ];

        let optimized = manager.optimize(&messages);
        assert!(!optimized.is_empty());
    }

    #[test]
    fn test_context_config_for_complexity() {
        let simple = ContextConfig::for_complexity(ContextComplexity::Simple);
        assert_eq!(simple.max_input_tokens, 20_000);

        let complex = ContextConfig::for_complexity(ContextComplexity::Complex);
        assert_eq!(complex.max_input_tokens, 100_000);
    }

    #[test]
    fn test_chunk_type_variants() {
        assert_eq!(ChunkType::Code, ChunkType::Code);
        assert_ne!(ChunkType::Code, ChunkType::Prose);
    }
}
