//! OpenAI-compatible API types.

use serde::{Deserialize, Serialize};

/// Chat completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use.
    pub model: String,
    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Temperature for sampling.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Number of completions to generate.
    #[serde(default)]
    pub n: Option<u32>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Stop sequences.
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role (system, user, assistant).
    pub role: String,
    /// Message content.
    pub content: String,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage.
    pub usage: Usage,
}

/// A completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,
    /// Generated message.
    pub message: ChatMessage,
    /// Finish reason.
    pub finish_reason: String,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}
