//! Token counting API types and handlers.
//!
//! Provides `/v1/tokenize` endpoint for pre-flight token estimation
//! without running inference. This helps clients:
//!
//! - Validate requests won't exceed context limits
//! - Estimate costs before making requests
//! - Debug tokenization behavior
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::tokenize::{TokenizeRequest, TokenizeResponse};
//!
//! let request = TokenizeRequest {
//!     model: "llama-3b".to_string(),
//!     messages: Some(vec![...]),
//!     prompt: None,
//! };
//! ```

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::openai::ChatMessage;

/// Request to count tokens in a prompt or messages.
#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct TokenizeRequest {
    /// Model to use for tokenization.
    pub model: String,

    /// Messages to tokenize (for chat format).
    #[serde(default)]
    pub messages: Option<Vec<ChatMessage>>,

    /// Raw prompt to tokenize (for completion format).
    #[serde(default)]
    pub prompt: Option<String>,

    /// Whether to include individual token strings in response.
    #[serde(default)]
    pub return_tokens: Option<bool>,
}

/// Token counting response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TokenizeResponse {
    /// Total number of tokens.
    pub token_count: u32,

    /// Model used for tokenization.
    pub model: String,

    /// Individual tokens (if return_tokens was true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<String>>,

    /// Token IDs (if return_tokens was true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
}

impl TokenizeRequest {
    /// Validates the request has either messages or prompt.
    pub fn validate(&self) -> Result<(), TokenizeError> {
        if self.messages.is_none() && self.prompt.is_none() {
            return Err(TokenizeError::NoInput);
        }
        if self.messages.is_some() && self.prompt.is_some() {
            return Err(TokenizeError::BothInputs);
        }
        if self.model.is_empty() {
            return Err(TokenizeError::EmptyModel);
        }
        Ok(())
    }
}

/// Errors that can occur during tokenization.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TokenizeError {
    /// Neither messages nor prompt provided.
    #[error("either 'messages' or 'prompt' must be provided")]
    NoInput,

    /// Both messages and prompt provided.
    #[error("provide either 'messages' or 'prompt', not both")]
    BothInputs,

    /// Model field is empty.
    #[error("model field is required")]
    EmptyModel,

    /// Model not found or not loaded.
    #[error("model '{0}' not found")]
    ModelNotFound(String),

    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
}

/// Trait for tokenizer implementations.
pub trait Tokenizer: Send + Sync {
    /// Counts tokens in the given text.
    fn count_tokens(&self, text: &str) -> Result<u32, TokenizeError>;

    /// Tokenizes text and returns tokens with IDs.
    fn tokenize(&self, text: &str) -> Result<(Vec<String>, Vec<u32>), TokenizeError>;

    /// Formats chat messages into a prompt string.
    fn format_chat(&self, messages: &[ChatMessage]) -> String;
}

/// Simple tokenizer that estimates ~4 chars per token.
/// Used as fallback when model-specific tokenizer unavailable.
#[derive(Debug, Clone, Default)]
pub struct EstimatingTokenizer;

impl Tokenizer for EstimatingTokenizer {
    fn count_tokens(&self, text: &str) -> Result<u32, TokenizeError> {
        // Rough estimate: ~4 characters per token for English
        // This matches GPT-style tokenization approximately
        let char_count = text.chars().count();
        #[allow(clippy::cast_possible_truncation)]
        Ok((char_count as f64 / 4.0).ceil() as u32)
    }

    fn tokenize(&self, text: &str) -> Result<(Vec<String>, Vec<u32>), TokenizeError> {
        // Simple whitespace tokenization for estimation
        let tokens: Vec<String> = text.split_whitespace().map(String::from).collect();
        let ids: Vec<u32> = (0..tokens.len() as u32).collect();
        Ok((tokens, ids))
    }

    fn format_chat(&self, messages: &[ChatMessage]) -> String {
        // Simple chat format: <|role|>content
        messages
            .iter()
            .map(|m| format!("<|{}|>{}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Counts tokens for a tokenize request.
pub fn count_tokens<T: Tokenizer>(
    tokenizer: &T,
    request: &TokenizeRequest,
) -> Result<TokenizeResponse, TokenizeError> {
    request.validate()?;

    let text = if let Some(messages) = &request.messages {
        tokenizer.format_chat(messages)
    } else if let Some(prompt) = &request.prompt {
        prompt.clone()
    } else {
        return Err(TokenizeError::NoInput);
    };

    let return_tokens = request.return_tokens.unwrap_or(false);

    if return_tokens {
        let (tokens, token_ids) = tokenizer.tokenize(&text)?;
        #[allow(clippy::cast_possible_truncation)]
        let token_count = tokens.len() as u32;
        Ok(TokenizeResponse {
            token_count,
            model: request.model.clone(),
            tokens: Some(tokens),
            token_ids: Some(token_ids),
        })
    } else {
        let token_count = tokenizer.count_tokens(&text)?;
        Ok(TokenizeResponse {
            token_count,
            model: request.model.clone(),
            tokens: None,
            token_ids: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TokenizeRequest Tests
    // =========================================================================

    #[test]
    fn test_tokenize_request_with_messages() {
        let json = r#"{
            "model": "llama-3b",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;

        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "llama-3b");
        assert!(request.messages.is_some());
        assert!(request.prompt.is_none());
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tokenize_request_with_prompt() {
        let json = r#"{
            "model": "llama-3b",
            "prompt": "Hello, world!"
        }"#;

        let request: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "llama-3b");
        assert!(request.messages.is_none());
        assert!(request.prompt.is_some());
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tokenize_request_validation_no_input() {
        let request = TokenizeRequest {
            model: "llama-3b".to_string(),
            messages: None,
            prompt: None,
            return_tokens: None,
        };

        let err = request.validate().unwrap_err();
        assert!(matches!(err, TokenizeError::NoInput));
    }

    #[test]
    fn test_tokenize_request_validation_both_inputs() {
        let request = TokenizeRequest {
            model: "llama-3b".to_string(),
            messages: Some(vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]),
            prompt: Some("Hello".to_string()),
            return_tokens: None,
        };

        let err = request.validate().unwrap_err();
        assert!(matches!(err, TokenizeError::BothInputs));
    }

    #[test]
    fn test_tokenize_request_validation_empty_model() {
        let request = TokenizeRequest {
            model: String::new(),
            messages: Some(vec![]),
            prompt: None,
            return_tokens: None,
        };

        let err = request.validate().unwrap_err();
        assert!(matches!(err, TokenizeError::EmptyModel));
    }

    // =========================================================================
    // TokenizeResponse Tests
    // =========================================================================

    #[test]
    fn test_tokenize_response_serialization() {
        let response = TokenizeResponse {
            token_count: 42,
            model: "llama-3b".to_string(),
            tokens: None,
            token_ids: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"token_count\":42"));
        assert!(json.contains("\"model\":\"llama-3b\""));
        // tokens should be omitted when None
        assert!(!json.contains("tokens"));
    }

    #[test]
    fn test_tokenize_response_with_tokens() {
        let response = TokenizeResponse {
            token_count: 3,
            model: "llama-3b".to_string(),
            tokens: Some(vec!["hello".to_string(), "world".to_string()]),
            token_ids: Some(vec![1, 2]),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"tokens\""));
        assert!(json.contains("\"token_ids\""));
    }

    // =========================================================================
    // EstimatingTokenizer Tests
    // =========================================================================

    #[test]
    fn test_estimating_tokenizer_count() {
        let tokenizer = EstimatingTokenizer;

        // ~4 chars per token
        assert_eq!(tokenizer.count_tokens("Hello").unwrap(), 2); // 5 chars -> 2 tokens
        assert_eq!(tokenizer.count_tokens("Hi").unwrap(), 1); // 2 chars -> 1 token
        assert_eq!(tokenizer.count_tokens("Hello, world!").unwrap(), 4); // 13 chars -> 4 tokens
    }

    #[test]
    fn test_estimating_tokenizer_format_chat() {
        let tokenizer = EstimatingTokenizer;
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi!".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let formatted = tokenizer.format_chat(&messages);
        assert!(formatted.contains("<|system|>You are helpful."));
        assert!(formatted.contains("<|user|>Hi!"));
    }

    #[test]
    fn test_estimating_tokenizer_tokenize() {
        let tokenizer = EstimatingTokenizer;
        let (tokens, ids) = tokenizer.tokenize("hello world test").unwrap();

        assert_eq!(tokens.len(), 3);
        assert_eq!(ids.len(), 3);
        assert_eq!(tokens[0], "hello");
        assert_eq!(tokens[1], "world");
        assert_eq!(tokens[2], "test");
    }

    // =========================================================================
    // count_tokens Function Tests
    // =========================================================================

    #[test]
    fn test_count_tokens_with_prompt() {
        let tokenizer = EstimatingTokenizer;
        let request = TokenizeRequest {
            model: "test-model".to_string(),
            messages: None,
            prompt: Some("Hello, world!".to_string()),
            return_tokens: None,
        };

        let response = count_tokens(&tokenizer, &request).unwrap();
        assert_eq!(response.model, "test-model");
        assert!(response.token_count > 0);
        assert!(response.tokens.is_none());
    }

    #[test]
    fn test_count_tokens_with_messages() {
        let tokenizer = EstimatingTokenizer;
        let request = TokenizeRequest {
            model: "test-model".to_string(),
            messages: Some(vec![ChatMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]),
            prompt: None,
            return_tokens: None,
        };

        let response = count_tokens(&tokenizer, &request).unwrap();
        assert!(response.token_count > 0);
    }

    #[test]
    fn test_count_tokens_with_return_tokens() {
        let tokenizer = EstimatingTokenizer;
        let request = TokenizeRequest {
            model: "test-model".to_string(),
            messages: None,
            prompt: Some("hello world".to_string()),
            return_tokens: Some(true),
        };

        let response = count_tokens(&tokenizer, &request).unwrap();
        assert!(response.tokens.is_some());
        assert!(response.token_ids.is_some());
        assert_eq!(response.tokens.as_ref().unwrap().len(), 2);
    }
}
