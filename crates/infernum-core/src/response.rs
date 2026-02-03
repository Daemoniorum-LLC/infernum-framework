//! Response types for inference operations.

use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, Message, ModelId, RequestId, Usage};

/// Response from text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Request identifier.
    pub request_id: RequestId,

    /// Unix timestamp of creation.
    pub created: i64,

    /// Model used for generation.
    pub model: ModelId,

    /// Generated completions.
    pub choices: Vec<Choice>,

    /// Token usage statistics.
    pub usage: Usage,

    /// Time to first token in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,

    /// Total generation time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,
}

/// A single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice.
    pub index: u32,

    /// Generated text.
    pub text: String,

    /// Full message with role, content, and tool_calls (chat mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,

    /// Reason generation stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

/// Log probability information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Token strings.
    pub tokens: Vec<String>,

    /// Log probabilities for each token.
    pub token_logprobs: Vec<f32>,

    /// Top log probabilities at each position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<Vec<TopLogProb>>>,
}

/// Top log probability entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    /// Token string.
    pub token: String,

    /// Log probability.
    pub logprob: f32,
}

/// Information about a single generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token ID.
    pub id: u32,

    /// Token text.
    pub text: String,

    /// Log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,

    /// Whether this is a special token.
    #[serde(default)]
    pub special: bool,
}

/// Response from embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// Request identifier.
    pub request_id: RequestId,

    /// Model used for embeddings.
    pub model: ModelId,

    /// Generated embeddings.
    pub data: Vec<Embedding>,

    /// Token usage statistics.
    pub usage: Usage,
}

/// A single embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Index of this embedding.
    pub index: u32,

    /// Embedding vector.
    pub embedding: EmbeddingData,
}

/// Embedding data in different formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    /// Float vector.
    Float(Vec<f32>),

    /// Base64 encoded binary.
    Base64(String),
}

impl EmbeddingData {
    /// Returns the embedding as a float vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is base64 encoded and decoding fails.
    pub fn as_floats(&self) -> Result<Vec<f32>, &'static str> {
        match self {
            Self::Float(v) => Ok(v.clone()),
            Self::Base64(_) => Err("base64 decoding not implemented"),
        }
    }

    /// Returns the dimensionality of the embedding.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Float(v) => v.len(),
            Self::Base64(s) => {
                // Each f32 is 4 bytes, base64 encoding is ~4/3 ratio
                (s.len() * 3) / 16
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choice_creation() {
        let choice = Choice {
            index: 0,
            text: "Hello world".to_string(),
            message: None,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        assert_eq!(choice.index, 0);
        assert_eq!(choice.text, "Hello world");
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_logprobs() {
        let logprobs = LogProbs {
            tokens: vec!["Hello".to_string(), " ".to_string(), "world".to_string()],
            token_logprobs: vec![-0.5, -0.1, -0.3],
            top_logprobs: None,
        };

        assert_eq!(logprobs.tokens.len(), 3);
        assert_eq!(logprobs.token_logprobs.len(), 3);
    }

    #[test]
    fn test_top_logprob() {
        let top = TopLogProb {
            token: "hello".to_string(),
            logprob: -0.5,
        };

        assert_eq!(top.token, "hello");
        assert_eq!(top.logprob, -0.5);
    }

    #[test]
    fn test_token_info() {
        let info = TokenInfo {
            id: 1234,
            text: "token".to_string(),
            logprob: Some(-0.2),
            special: false,
        };

        assert_eq!(info.id, 1234);
        assert_eq!(info.text, "token");
        assert_eq!(info.logprob, Some(-0.2));
        assert!(!info.special);
    }

    #[test]
    fn test_embedding_data_float() {
        let data = EmbeddingData::Float(vec![0.1, 0.2, 0.3, 0.4]);

        let floats = data.as_floats().unwrap();
        assert_eq!(floats, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(data.dimensions(), 4);
    }

    #[test]
    fn test_embedding_data_base64() {
        let data = EmbeddingData::Base64("AAAAAAAAAAAAAAAA".to_string());

        assert!(data.as_floats().is_err());
        // 16 chars * 3 / 16 = 3
        assert_eq!(data.dimensions(), 3);
    }

    #[test]
    fn test_embedding() {
        let embedding = Embedding {
            index: 0,
            embedding: EmbeddingData::Float(vec![0.1, 0.2]),
        };

        assert_eq!(embedding.index, 0);
        assert_eq!(embedding.embedding.dimensions(), 2);
    }

    #[test]
    fn test_generate_response_serialization() {
        let response = GenerateResponse {
            request_id: RequestId::new(),
            created: 1706832000,
            model: ModelId::from("test-model"),
            choices: vec![Choice {
                index: 0,
                text: "Generated text".to_string(),
                message: None,
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            time_to_first_token_ms: Some(50.0),
            total_time_ms: Some(200.0),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: GenerateResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model.to_string(), "test-model");
        assert_eq!(deserialized.choices.len(), 1);
        assert_eq!(deserialized.usage.total_tokens, 15);
        assert_eq!(deserialized.created, 1706832000);
    }

    #[test]
    fn test_embed_response_serialization() {
        let response = EmbedResponse {
            request_id: RequestId::new(),
            model: ModelId::from("embedding-model"),
            data: vec![Embedding {
                index: 0,
                embedding: EmbeddingData::Float(vec![0.1, 0.2, 0.3]),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 0,
                total_tokens: 5,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: EmbedResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.data.len(), 1);
        assert_eq!(deserialized.data[0].embedding.dimensions(), 3);
    }

    #[test]
    fn test_choice_without_finish_reason() {
        let choice = Choice {
            index: 0,
            text: "partial...".to_string(),
            message: None,
            finish_reason: None,
            logprobs: None,
        };

        // Should serialize without finish_reason field
        let json = serde_json::to_string(&choice).unwrap();
        assert!(!json.contains("finish_reason"));
    }

    #[test]
    fn test_finish_reasons() {
        let stop = Choice {
            index: 0,
            text: "done".to_string(),
            message: None,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        let length = Choice {
            index: 1,
            text: "truncated".to_string(),
            message: None,
            finish_reason: Some(FinishReason::Length),
            logprobs: None,
        };

        assert_eq!(stop.finish_reason, Some(FinishReason::Stop));
        assert_eq!(length.finish_reason, Some(FinishReason::Length));
    }

    // =========================================================================
    // Choice with message (chat mode)
    // =========================================================================

    #[test]
    fn test_choice_with_message() {
        use crate::types::{Role, ToolCall};

        let msg = Message::assistant_with_tool_calls(
            "I'll read that file.",
            vec![ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "src/main.rs"}),
            }],
        );

        let choice = Choice {
            index: 0,
            text: "I'll read that file.".to_string(),
            message: Some(msg),
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        };

        assert!(choice.message.is_some());
        let m = choice.message.as_ref().unwrap();
        assert_eq!(m.role, Role::Assistant);
        assert!(m.tool_calls.is_some());
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn test_choice_message_omitted_when_none() {
        let choice = Choice {
            index: 0,
            text: "text".to_string(),
            message: None,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        let json = serde_json::to_string(&choice).unwrap();
        assert!(!json.contains("message"));
    }

    #[test]
    fn test_choice_message_serialization_roundtrip() {
        let choice = Choice {
            index: 0,
            text: "Done.".to_string(),
            message: Some(Message::assistant("Done.")),
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"message\""));
        assert!(json.contains("\"role\":\"assistant\""));

        let parsed: Choice = serde_json::from_str(&json).unwrap();
        assert!(parsed.message.is_some());
        assert_eq!(parsed.message.unwrap().content, "Done.");
    }

    // =========================================================================
    // GenerateResponse.created
    // =========================================================================

    #[test]
    fn test_generate_response_created_timestamp() {
        let response = GenerateResponse {
            request_id: RequestId::new(),
            created: 1706832000,
            model: ModelId::from("test"),
            choices: vec![],
            usage: Usage::default(),
            time_to_first_token_ms: None,
            total_time_ms: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"created\":1706832000"));
    }

    #[test]
    fn test_generate_response_spec_example() {
        // Verify the spec response format from INFERNUM-API-SPEC.md §3.7
        let json = r#"{
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "created": 1706832000,
            "model": "llama-3.2-3b",
            "choices": [
                {
                    "index": 0,
                    "text": "I'll read that file for you.",
                    "message": {
                        "role": "assistant",
                        "content": "I'll read that file for you.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "name": "read_file",
                                "arguments": { "path": "src/main.rs" }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 18,
                "total_tokens": 60
            },
            "time_to_first_token_ms": 45.2,
            "total_time_ms": 320.8
        }"#;

        let response: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.created, 1706832000);
        assert_eq!(response.model.to_string(), "llama-3.2-3b");
        assert_eq!(response.choices.len(), 1);

        let choice = &response.choices[0];
        assert!(choice.message.is_some());
        let msg = choice.message.as_ref().unwrap();
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap()[0].name, "read_file");
        assert_eq!(
            msg.tool_calls.as_ref().unwrap()[0].arguments["path"],
            "src/main.rs"
        );
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.usage.total_tokens, 60);
    }
}
