//! Infernum API wire types.
//!
//! This module defines the request/response types for Infernum's HTTP API.
//!
//! # Supported Endpoints
//!
//! | Endpoint | Request Type | Response Type |
//! |----------|--------------|---------------|
//! | `POST /v1/chat/completions` | [`ChatCompletionRequest`] | [`ChatCompletionResponse`] |
//! | `POST /v1/completions` | [`CompletionRequest`] | [`CompletionResponse`] |
//! | `POST /v1/embeddings` | [`EmbeddingRequest`] | [`EmbeddingResponse`] |
//! | `GET /v1/models` | - | [`ModelsResponse`] |
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::{ChatCompletionRequest, ChatMessage};
//!
//! // Parse an incoming request
//! let json = r#"{
//!     "model": "meta-llama/Llama-3.2-3B-Instruct",
//!     "messages": [
//!         {"role": "system", "content": "You are a helpful assistant."},
//!         {"role": "user", "content": "Hello!"}
//!     ],
//!     "temperature": 0.7
//! }"#;
//!
//! let request: ChatCompletionRequest = serde_json::from_str(json)?;
//! ```
//!
//! # Streaming
//!
//! For streaming responses, use [`ChatCompletionChunk`] which contains incremental
//! content in [`ChatDelta`]. The server sends these as Server-Sent Events (SSE)
//! with the `data: ` prefix.
//!
//! # Wire Format Notes
//!
//! - Optional fields use `Option<T>` for proper deserialization
//! - Response IDs use the format `inf-chat-{uuid}` or `inf-cmpl-{uuid}`
//! - The `finish_reason` field uses lowercase values: `stop`, `length`, `tool_calls`

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::structured::ResponseFormat;

// === Chat Completions ===

/// Chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
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

    /// Whether to return log probabilities.
    #[serde(default)]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return (0-20).
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    /// A list of tools the model may call.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,

    /// Controls which tool is called by the model.
    /// Can be "none", "auto", "required", or a specific tool.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to enable parallel function calling.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Response format specification for structured outputs.
    /// Can be "text", "json_object", or a JSON schema specification.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatMessage {
    /// Role (system, user, assistant, tool).
    pub role: String,
    /// Message content.
    pub content: String,
    /// Optional name for the sender.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls made by the assistant (only present for assistant messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID (only present for tool response messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// === Function Calling / Tool Use ===

/// A tool that can be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Tool {
    /// The type of the tool. Currently, only "function" is supported.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
}

/// A function definition for tool use.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FunctionDefinition {
    /// The name of the function to call.
    pub name: String,
    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the function accepts, described as a JSON Schema object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Whether the function should be called in strict mode (exact schema matching).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A tool call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only "function" is supported.
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function that was called.
    pub function: FunctionCall,
}

/// A function call with arguments.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to call the function with, as a JSON string.
    pub arguments: String,
}

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A string option: "none", "auto", or "required".
    String(String),
    /// A specific tool to call.
    Tool(ToolChoiceFunction),
}

/// A specific tool choice.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolChoiceFunction {
    /// The type of the tool. Currently, only "function" is supported.
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The function to call.
    pub function: ToolChoiceFunctionName,
}

/// The name of a function to force.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolChoiceFunctionName {
    /// The name of the function to call.
    pub name: String,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,
    /// Generated message.
    pub message: ChatMessage,
    /// Finish reason (stop, length, tool_calls, content_filter).
    pub finish_reason: String,
    /// Log probability information (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// Log probability information for chat completions.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ChatLogProbs {
    /// Log probabilities for each token in the response.
    pub content: Vec<TokenLogProb>,
}

/// Log probability information for a single token.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TokenLogProb {
    /// The token string.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// UTF-8 byte offsets for the token (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Top alternative tokens with their log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogProb>>,
}

/// A top alternative token with its log probability.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TopLogProb {
    /// The token string.
    pub token: String,
    /// The log probability of this token.
    pub logprob: f32,
    /// UTF-8 byte offsets for the token (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, Default, ToSchema)]
pub struct ChatDelta {
    /// Role (only on first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// === Text Completions ===

/// Text completion request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
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
    /// Presence penalty (-2.0 to 2.0).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (-2.0 to 2.0).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

/// Text completion response.
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
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

/// Embedding request.
#[derive(Debug, Clone, Deserialize, ToSchema)]
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
#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text input.
    Single(String),
    /// Multiple text inputs.
    Multiple(Vec<String>),
}

/// Embedding response.
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EmbeddingData {
    /// Object type ("embedding").
    pub object: String,
    /// Index in the input array.
    pub index: u32,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

/// Embedding usage statistics.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EmbeddingUsage {
    /// Prompt tokens used.
    pub prompt_tokens: u32,
    /// Total tokens used.
    pub total_tokens: u32,
}

// === Models ===

/// Models list response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelsResponse {
    /// Object type ("list").
    pub object: String,
    /// Available models.
    pub data: Vec<ModelObject>,
}

/// Model information.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelObject {
    /// Model ID.
    pub id: String,
    /// Object type ("model").
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Owner (e.g., "infernum", "huggingface").
    pub owned_by: String,
}

// === Common ===

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize, ToSchema)]
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
            id: "inf-chat-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1677652288,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_string(),
                logprobs: None,
            }],
            usage: Usage::new(10, 5),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("inf-chat-123"));
        assert!(json.contains("Hello!"));
        // logprobs and tool_calls should be omitted when None
        assert!(!json.contains("logprobs"));
        assert!(!json.contains("tool_calls"));
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

    #[test]
    fn test_chat_request_with_logprobs() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "logprobs": true,
            "top_logprobs": 5
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.logprobs, Some(true));
        assert_eq!(req.top_logprobs, Some(5));
    }

    #[test]
    fn test_chat_request_logprobs_defaults() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.logprobs, None);
        assert_eq!(req.top_logprobs, None);
    }

    #[test]
    fn test_logprobs_serialization() {
        let logprobs = ChatLogProbs {
            content: vec![
                TokenLogProb {
                    token: "Hello".to_string(),
                    logprob: -0.5,
                    bytes: Some(vec![72, 101, 108, 108, 111]),
                    top_logprobs: Some(vec![
                        TopLogProb {
                            token: "Hello".to_string(),
                            logprob: -0.5,
                            bytes: None,
                        },
                        TopLogProb {
                            token: "Hi".to_string(),
                            logprob: -1.2,
                            bytes: None,
                        },
                    ]),
                },
            ],
        };

        let json = serde_json::to_string(&logprobs).unwrap();
        assert!(json.contains("\"token\":\"Hello\""));
        assert!(json.contains("\"logprob\":-0.5"));
        assert!(json.contains("\"top_logprobs\""));
    }

    #[test]
    fn test_chat_response_with_logprobs() {
        let response = ChatCompletionResponse {
            id: "inf-chat-456".to_string(),
            object: "chat.completion".to_string(),
            created: 1677652288,
            model: "gpt-4".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hi!".to_string(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_string(),
                logprobs: Some(ChatLogProbs {
                    content: vec![TokenLogProb {
                        token: "Hi".to_string(),
                        logprob: -0.3,
                        bytes: None,
                        top_logprobs: None,
                    }],
                }),
            }],
            usage: Usage::new(5, 2),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"logprobs\""));
        assert!(json.contains("\"token\":\"Hi\""));
        assert!(json.contains("\"logprob\":-0.3"));
    }

    #[test]
    fn test_token_logprob_minimal() {
        let token_logprob = TokenLogProb {
            token: "test".to_string(),
            logprob: -1.0,
            bytes: None,
            top_logprobs: None,
        };

        let json = serde_json::to_string(&token_logprob).unwrap();
        assert!(json.contains("\"token\":\"test\""));
        assert!(json.contains("\"logprob\":-1"));
        // bytes and top_logprobs should be omitted when None
        assert!(!json.contains("\"bytes\""));
        assert!(!json.contains("\"top_logprobs\""));
    }

    // === Function Calling Tests ===

    #[test]
    fn test_tool_serialization() {
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
                strict: None,
            },
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("\"description\":\"Get the current weather\""));
        assert!(json.contains("\"location\""));
    }

    #[test]
    fn test_tool_deserialization() {
        let json = r#"{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }"#;

        let tool: Tool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(
            tool.function.description,
            Some("Get current weather".to_string())
        );
    }

    #[test]
    fn test_tool_call_serialization() {
        let tool_call = ToolCall {
            id: "call_abc123".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "San Francisco"}"#.to_string(),
            },
        };

        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains("\"id\":\"call_abc123\""));
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
        assert!(json.contains("San Francisco"));
    }

    #[test]
    fn test_tool_call_deserialization() {
        let json = r#"{
            "id": "call_xyz789",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": "{\"expression\": \"2+2\"}"
            }
        }"#;

        let tool_call: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.id, "call_xyz789");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "calculate");
    }

    #[test]
    fn test_chat_request_with_tools() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_tool_choice_string_variant() {
        let choice = ToolChoice::String("auto".to_string());
        let json = serde_json::to_string(&choice).unwrap();
        assert_eq!(json, "\"auto\"");

        let parsed: ToolChoice = serde_json::from_str("\"none\"").unwrap();
        match parsed {
            ToolChoice::String(s) => assert_eq!(s, "none"),
            _ => panic!("Expected String variant"),
        }
    }

    #[test]
    fn test_tool_choice_function_variant() {
        let choice = ToolChoice::Tool(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName {
                name: "get_weather".to_string(),
            },
        });

        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn test_message_with_tool_calls() {
        let message = ChatMessage {
            role: "assistant".to_string(),
            content: "".to_string(),
            name: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location": "NYC"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"tool_calls\""));
        assert!(json.contains("\"call_123\""));
    }

    #[test]
    fn test_tool_response_message() {
        let message = ChatMessage {
            role: "tool".to_string(),
            content: "72°F, sunny".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("\"role\":\"tool\""));
        assert!(json.contains("\"tool_call_id\":\"call_123\""));
        assert!(json.contains("72°F"));
    }

    #[test]
    fn test_function_definition_minimal() {
        let func = FunctionDefinition {
            name: "simple_func".to_string(),
            description: None,
            parameters: None,
            strict: None,
        };

        let json = serde_json::to_string(&func).unwrap();
        assert!(json.contains("\"name\":\"simple_func\""));
        // Optional fields should be omitted
        assert!(!json.contains("\"description\""));
        assert!(!json.contains("\"parameters\""));
        assert!(!json.contains("\"strict\""));
    }

    #[test]
    fn test_function_definition_with_strict() {
        let func = FunctionDefinition {
            name: "strict_func".to_string(),
            description: Some("A strict function".to_string()),
            parameters: Some(serde_json::json!({"type": "object"})),
            strict: Some(true),
        };

        let json = serde_json::to_string(&func).unwrap();
        assert!(json.contains("\"strict\":true"));
    }

    // === Response Format Tests ===

    #[test]
    fn test_chat_request_with_response_format_text() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "response_format": {"type": "text"}
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.response_format.is_some());
        let format = req.response_format.unwrap();
        assert!(matches!(format, ResponseFormat::Text));
    }

    #[test]
    fn test_chat_request_with_response_format_json_object() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Give me JSON"}],
            "response_format": {"type": "json_object"}
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.response_format.is_some());
        let format = req.response_format.unwrap();
        assert!(matches!(format, ResponseFormat::JsonObject));
        assert!(format.requires_json());
    }

    #[test]
    fn test_chat_request_with_response_format_json_schema() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Give me a person object"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "description": "A person object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"]
                    },
                    "strict": true
                }
            }
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.response_format.is_some());
        let format = req.response_format.unwrap();
        assert!(format.requires_json());
        assert!(format.is_strict());

        let schema = format.schema();
        assert!(schema.is_some());
        let schema = schema.unwrap();
        assert_eq!(schema.name, "person");
        assert_eq!(schema.description, Some("A person object".to_string()));
    }

    #[test]
    fn test_chat_request_without_response_format() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.response_format.is_none());
    }
}
