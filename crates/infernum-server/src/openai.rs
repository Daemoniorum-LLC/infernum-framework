//! OpenAI-compatible API types.
//!
//! These types mirror the OpenAI API specification for drop-in compatibility.

use serde::{Deserialize, Serialize};

// === Chat Completions ===

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use.
    pub model: String,
    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Temperature for sampling (0.0 - 2.0).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling.
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
    /// Presence penalty (-2.0 to 2.0).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (-2.0 to 2.0).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// User identifier for abuse monitoring.
    #[serde(default)]
    pub user: Option<String>,
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role (system, user, assistant, tool).
    pub role: String,
    /// Message content.
    pub content: String,
    /// Optional name for the sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type ("chat.completion").
    pub object: String,
    /// Creation timestamp (Unix epoch).
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics.
    pub usage: Usage,
}

/// A chat completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,
    /// Generated message.
    pub message: ChatMessage,
    /// Finish reason (stop, length, tool_calls, content_filter).
    pub finish_reason: String,
}

/// Streaming chat completion chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    /// Response ID.
    pub id: String,
    /// Object type ("chat.completion.chunk").
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Streaming choices.
    pub choices: Vec<ChatChunkChoice>,
}

/// A streaming chat choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    /// Choice index.
    pub index: u32,
    /// Incremental content.
    pub delta: ChatDelta,
    /// Finish reason (only present on final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Incremental chat content.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatDelta {
    /// Role (only on first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// === Text Completions ===

/// Text completion request (OpenAI-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model to use.
    pub model: String,
    /// The prompt to complete.
    pub prompt: String,
    /// Temperature for sampling.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Number of completions.
    #[serde(default)]
    pub n: Option<u32>,
    /// Whether to stream.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Stop sequences.
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Maximum tokens.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Include log probabilities.
    #[serde(default)]
    pub logprobs: Option<u32>,
    /// Echo the prompt.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Suffix to append.
    #[serde(default)]
    pub suffix: Option<String>,
}

/// Text completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type ("text_completion").
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token usage.
    pub usage: Usage,
}

/// A text completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    /// Generated text.
    pub text: String,
    /// Choice index.
    pub index: u32,
    /// Finish reason.
    pub finish_reason: String,
    /// Log probabilities (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

/// Log probability information.
#[derive(Debug, Clone, Serialize)]
pub struct LogProbs {
    /// Token strings.
    pub tokens: Vec<String>,
    /// Token log probabilities.
    pub token_logprobs: Vec<f32>,
    /// Top log probabilities.
    pub top_logprobs: Vec<std::collections::HashMap<String, f32>>,
    /// Text offsets.
    pub text_offset: Vec<u32>,
}

// === Embeddings ===

/// Embedding request (OpenAI-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    /// Model to use.
    pub model: String,
    /// Input text(s) to embed.
    pub input: EmbeddingInput,
    /// Encoding format (float or base64).
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Dimensions to truncate to.
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Embedding input - single string or array.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text input.
    Single(String),
    /// Multiple text inputs.
    Multiple(Vec<String>),
}

/// Embedding response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    /// Object type ("list").
    pub object: String,
    /// Embedding data.
    pub data: Vec<EmbeddingData>,
    /// Model used.
    pub model: String,
    /// Usage statistics.
    pub usage: EmbeddingUsage,
}

/// A single embedding result.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    /// Object type ("embedding").
    pub object: String,
    /// Index in the input array.
    pub index: u32,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

/// Embedding usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    /// Prompt tokens used.
    pub prompt_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
}

// === Models ===

/// Models list response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    /// Object type ("list").
    pub object: String,
    /// Available models.
    pub data: Vec<ModelObject>,
}

/// Model information.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    /// Model ID.
    pub id: String,
    /// Object type ("model").
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Owner (e.g., "openai", "infernum").
    pub owned_by: String,
}

// === Common ===

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

impl Usage {
    /// Creates new usage statistics.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_deserialization() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn test_chat_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1677652288,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage::new(10, 5),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("chatcmpl-123"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_embedding_input_variants() {
        // Single input
        let json = r#"{"model": "text-embedding-3-small", "input": "Hello"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        matches!(req.input, EmbeddingInput::Single(_));

        // Multiple inputs
        let json = r#"{"model": "text-embedding-3-small", "input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        matches!(req.input, EmbeddingInput::Multiple(_));
    }

    #[test]
    fn test_usage() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }
}
