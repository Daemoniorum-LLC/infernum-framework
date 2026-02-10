//! Request validation for Infernum API endpoints.
//!
//! This module provides validation functions and error types for incoming
//! API requests, ensuring inputs are within acceptable bounds before processing.
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::validation::{validate_chat_request, RequestValidationError};
//!
//! let result = validate_chat_request(&request, &limits);
//! if let Err(e) = result {
//!     return e.to_api_error(&request_id).into_response();
//! }
//! ```

use std::fmt;

use axum::http::StatusCode;

use crate::api_types::{ChatCompletionRequest, CompletionRequest, EmbeddingRequest};
use crate::error_response::{ApiError, ErrorCode, ErrorSubcode};
use crate::server::ValidationLimits;

/// Validation error for API requests.
///
/// Each variant contains the specific details of what failed validation,
/// enabling precise error messages and programmatic handling.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestValidationError {
    /// Messages array is empty (at least one message required).
    EmptyMessages,

    /// Too many messages in the request.
    TooManyMessages {
        /// Number of messages in request.
        count: usize,
        /// Maximum allowed messages.
        limit: usize,
    },

    /// A single message content exceeds the maximum length.
    MessageTooLong {
        /// Index of the message (0-based).
        index: usize,
        /// Length of the message content.
        length: usize,
        /// Maximum allowed length.
        limit: usize,
    },

    /// Temperature value out of valid range.
    InvalidTemperature {
        /// The invalid temperature value.
        value: f32,
    },

    /// Top-p value out of valid range.
    InvalidTopP {
        /// The invalid top_p value.
        value: f32,
    },

    /// max_tokens is zero or exceeds the limit.
    InvalidMaxTokens {
        /// The invalid value (0 means zero was provided).
        value: u32,
        /// Maximum allowed value.
        limit: u32,
    },

    /// Prompt is empty.
    EmptyPrompt,

    /// Prompt exceeds maximum length.
    PromptTooLong {
        /// Length of the prompt.
        length: usize,
        /// Maximum allowed length.
        limit: usize,
    },

    /// Too many embedding inputs.
    TooManyEmbeddingInputs {
        /// Number of inputs provided.
        count: usize,
        /// Maximum allowed inputs.
        limit: usize,
    },

    /// Model ID is invalid or malformed.
    InvalidModelId {
        /// The invalid model ID.
        model: String,
        /// Reason why it's invalid.
        reason: String,
    },
}

impl RequestValidationError {
    /// Returns the HTTP status code for this validation error.
    ///
    /// All validation errors return 400 Bad Request.
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        StatusCode::BAD_REQUEST
    }

    /// Returns the error code for API responses.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::EmptyMessages | Self::TooManyMessages { .. } | Self::MessageTooLong { .. } => {
                ErrorCode::InvalidMessages
            },
            Self::InvalidTemperature { .. } => ErrorCode::InvalidTemperature,
            Self::InvalidTopP { .. } => ErrorCode::InvalidTemperature, // Reuse same code for range errors
            Self::InvalidMaxTokens { .. } => ErrorCode::InvalidMaxTokens,
            Self::EmptyPrompt => ErrorCode::EmptyPrompt,
            Self::PromptTooLong { .. } => ErrorCode::PromptTooLong,
            Self::TooManyEmbeddingInputs { .. } => ErrorCode::InvalidEmbeddingInput,
            Self::InvalidModelId { .. } => ErrorCode::InvalidModel,
        }
    }

    /// Returns the error subcode for more granular handling.
    #[must_use]
    pub fn error_subcode(&self) -> Option<ErrorSubcode> {
        match self {
            Self::TooManyMessages { .. } | Self::TooManyEmbeddingInputs { .. } => {
                Some(ErrorSubcode::TooManyItems)
            },
            Self::MessageTooLong { .. } => Some(ErrorSubcode::SingleMessageTooLong),
            Self::InvalidMaxTokens { value, .. } if *value == 0 => Some(ErrorSubcode::BelowMinimum),
            Self::InvalidMaxTokens { .. } => Some(ErrorSubcode::AboveMaximum),
            Self::InvalidTemperature { value } if *value < 0.0 => Some(ErrorSubcode::BelowMinimum),
            Self::InvalidTemperature { .. } => Some(ErrorSubcode::AboveMaximum),
            Self::InvalidTopP { value } if *value < 0.0 => Some(ErrorSubcode::BelowMinimum),
            Self::InvalidTopP { .. } => Some(ErrorSubcode::AboveMaximum),
            Self::PromptTooLong { .. } => Some(ErrorSubcode::InputTooLong),
            Self::InvalidModelId { .. } => Some(ErrorSubcode::MalformedIdentifier),
            Self::EmptyMessages | Self::EmptyPrompt => Some(ErrorSubcode::MissingRequired),
        }
    }

    /// Returns the parameter name that caused the error.
    #[must_use]
    pub const fn param(&self) -> &'static str {
        match self {
            Self::EmptyMessages | Self::TooManyMessages { .. } => "messages",
            Self::MessageTooLong { .. } => "messages[].content",
            Self::InvalidTemperature { .. } => "temperature",
            Self::InvalidTopP { .. } => "top_p",
            Self::InvalidMaxTokens { .. } => "max_tokens",
            Self::EmptyPrompt | Self::PromptTooLong { .. } => "prompt",
            Self::TooManyEmbeddingInputs { .. } => "input",
            Self::InvalidModelId { .. } => "model",
        }
    }

    /// Converts to an API error response.
    #[must_use]
    pub fn to_api_error(&self, request_id: &str) -> ApiError {
        let mut builder = ApiError::new(self.error_code(), request_id)
            .message(self.to_string())
            .param(self.param());

        if let Some(subcode) = self.error_subcode() {
            builder = builder.subcode(subcode);
        }

        // Add limit/actual for applicable errors
        match self {
            Self::TooManyMessages { count, limit }
            | Self::TooManyEmbeddingInputs { count, limit } => {
                builder = builder.limit_exceeded(*limit as u64, *count as u64);
            },
            Self::MessageTooLong { length, limit, .. } | Self::PromptTooLong { length, limit } => {
                builder = builder.limit_exceeded(*limit as u64, *length as u64);
            },
            Self::InvalidMaxTokens { value, limit } => {
                builder = builder.limit_exceeded(*limit as u64, *value as u64);
            },
            _ => {},
        }

        builder.build()
    }
}

impl fmt::Display for RequestValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMessages => write!(f, "messages array cannot be empty"),
            Self::TooManyMessages { count, limit } => {
                write!(
                    f,
                    "too many messages: {} provided, maximum is {}",
                    count, limit
                )
            },
            Self::MessageTooLong {
                index,
                length,
                limit,
            } => {
                write!(
                    f,
                    "message at index {} is too long: {} characters, maximum is {}",
                    index, length, limit
                )
            },
            Self::InvalidTemperature { value } => {
                write!(
                    f,
                    "invalid temperature {}: must be between 0.0 and 2.0",
                    value
                )
            },
            Self::InvalidTopP { value } => {
                write!(f, "invalid top_p {}: must be between 0.0 and 1.0", value)
            },
            Self::InvalidMaxTokens { value, limit } => {
                if *value == 0 {
                    write!(f, "max_tokens must be at least 1")
                } else {
                    write!(
                        f,
                        "max_tokens {} exceeds maximum allowed value of {}",
                        value, limit
                    )
                }
            },
            Self::EmptyPrompt => write!(f, "prompt cannot be empty"),
            Self::PromptTooLong { length, limit } => {
                write!(
                    f,
                    "prompt too long: {} characters, maximum is {}",
                    length, limit
                )
            },
            Self::TooManyEmbeddingInputs { count, limit } => {
                write!(
                    f,
                    "too many embedding inputs: {} provided, maximum is {}",
                    count, limit
                )
            },
            Self::InvalidModelId { model, reason } => {
                write!(f, "invalid model '{}': {}", model, reason)
            },
        }
    }
}

impl std::error::Error for RequestValidationError {}

// === Validation Functions ===

/// Validates a chat completion request against the configured limits.
///
/// # Errors
///
/// Returns `RequestValidationError` if any field fails validation.
pub fn validate_chat_request(
    req: &ChatCompletionRequest,
    limits: &ValidationLimits,
) -> Result<(), RequestValidationError> {
    // Validate message count
    if req.messages.is_empty() {
        return Err(RequestValidationError::EmptyMessages);
    }
    if req.messages.len() > limits.max_messages {
        return Err(RequestValidationError::TooManyMessages {
            count: req.messages.len(),
            limit: limits.max_messages,
        });
    }

    // Validate message content lengths
    for (index, msg) in req.messages.iter().enumerate() {
        if msg.content.len() > limits.max_message_length {
            return Err(RequestValidationError::MessageTooLong {
                index,
                length: msg.content.len(),
                limit: limits.max_message_length,
            });
        }
    }

    // Validate temperature
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(RequestValidationError::InvalidTemperature { value: temp });
        }
    }

    // Validate top_p
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(RequestValidationError::InvalidTopP { value: top_p });
        }
    }

    // Validate max_tokens
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 || max_tokens > limits.max_max_tokens {
            return Err(RequestValidationError::InvalidMaxTokens {
                value: max_tokens,
                limit: limits.max_max_tokens,
            });
        }
    }

    Ok(())
}

/// Validates a text completion request.
///
/// # Errors
///
/// Returns `RequestValidationError` if any field fails validation.
pub fn validate_completion_request(
    req: &CompletionRequest,
    limits: &ValidationLimits,
) -> Result<(), RequestValidationError> {
    // Validate prompt
    if req.prompt.is_empty() {
        return Err(RequestValidationError::EmptyPrompt);
    }
    if req.prompt.len() > limits.max_prompt_length {
        return Err(RequestValidationError::PromptTooLong {
            length: req.prompt.len(),
            limit: limits.max_prompt_length,
        });
    }

    // Validate temperature
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(RequestValidationError::InvalidTemperature { value: temp });
        }
    }

    // Validate top_p
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(RequestValidationError::InvalidTopP { value: top_p });
        }
    }

    // Validate max_tokens
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 || max_tokens > limits.max_max_tokens {
            return Err(RequestValidationError::InvalidMaxTokens {
                value: max_tokens,
                limit: limits.max_max_tokens,
            });
        }
    }

    Ok(())
}

/// Validates an embedding request.
///
/// # Errors
///
/// Returns `RequestValidationError` if any field fails validation.
pub fn validate_embedding_request(
    req: &EmbeddingRequest,
    limits: &ValidationLimits,
) -> Result<(), RequestValidationError> {
    let input_count = match &req.input {
        crate::api_types::EmbeddingInput::Single(_) => 1,
        crate::api_types::EmbeddingInput::Multiple(inputs) => inputs.len(),
    };

    if input_count > limits.max_embedding_inputs {
        return Err(RequestValidationError::TooManyEmbeddingInputs {
            count: input_count,
            limit: limits.max_embedding_inputs,
        });
    }

    Ok(())
}

/// Validates a model ID format.
///
/// Model IDs should:
/// - Not be empty
/// - Not contain path traversal sequences
/// - Be reasonably short (< 256 chars)
///
/// # Errors
///
/// Returns `RequestValidationError::InvalidModelId` if validation fails.
pub fn validate_model_id(model: &str) -> Result<(), RequestValidationError> {
    if model.is_empty() {
        return Err(RequestValidationError::InvalidModelId {
            model: model.to_string(),
            reason: "model ID cannot be empty".to_string(),
        });
    }

    if model.len() > 256 {
        return Err(RequestValidationError::InvalidModelId {
            model: model.to_string(),
            reason: "model ID too long (max 256 characters)".to_string(),
        });
    }

    // Check for path traversal
    if model.contains("..") || model.contains("//") {
        return Err(RequestValidationError::InvalidModelId {
            model: model.to_string(),
            reason: "model ID contains invalid path sequences".to_string(),
        });
    }

    // Check for null bytes
    if model.contains('\0') {
        return Err(RequestValidationError::InvalidModelId {
            model: model.to_string(),
            reason: "model ID contains invalid characters".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_types::{ChatMessage, EmbeddingInput};

    fn default_limits() -> ValidationLimits {
        ValidationLimits::default()
    }

    // === RequestValidationError tests (Task 2.4) ===

    #[test]
    fn test_empty_messages_error() {
        let err = RequestValidationError::EmptyMessages;
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_code(), ErrorCode::InvalidMessages);
        assert_eq!(err.param(), "messages");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_too_many_messages_error() {
        let err = RequestValidationError::TooManyMessages {
            count: 300,
            limit: 256,
        };
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_code(), ErrorCode::InvalidMessages);
        assert_eq!(err.error_subcode(), Some(ErrorSubcode::TooManyItems));
        assert!(err.to_string().contains("300"));
        assert!(err.to_string().contains("256"));
    }

    #[test]
    fn test_message_too_long_error() {
        let err = RequestValidationError::MessageTooLong {
            index: 2,
            length: 150_000,
            limit: 100_000,
        };
        assert_eq!(
            err.error_subcode(),
            Some(ErrorSubcode::SingleMessageTooLong)
        );
        assert!(err.to_string().contains("index 2"));
        assert!(err.to_string().contains("150000"));
    }

    #[test]
    fn test_invalid_temperature_error() {
        let err = RequestValidationError::InvalidTemperature { value: 3.5 };
        assert_eq!(err.error_code(), ErrorCode::InvalidTemperature);
        assert!(err.to_string().contains("3.5"));
        assert!(err.to_string().contains("0.0"));
        assert!(err.to_string().contains("2.0"));
    }

    #[test]
    fn test_invalid_top_p_error() {
        let err = RequestValidationError::InvalidTopP { value: 1.5 };
        assert!(err.to_string().contains("1.5"));
        assert!(err.to_string().contains("0.0"));
        assert!(err.to_string().contains("1.0"));
    }

    #[test]
    fn test_invalid_max_tokens_zero() {
        let err = RequestValidationError::InvalidMaxTokens {
            value: 0,
            limit: 32768,
        };
        assert_eq!(err.error_subcode(), Some(ErrorSubcode::BelowMinimum));
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn test_invalid_max_tokens_exceeded() {
        let err = RequestValidationError::InvalidMaxTokens {
            value: 50000,
            limit: 32768,
        };
        assert_eq!(err.error_subcode(), Some(ErrorSubcode::AboveMaximum));
        assert!(err.to_string().contains("50000"));
        assert!(err.to_string().contains("32768"));
    }

    #[test]
    fn test_empty_prompt_error() {
        let err = RequestValidationError::EmptyPrompt;
        assert_eq!(err.error_code(), ErrorCode::EmptyPrompt);
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_prompt_too_long_error() {
        let err = RequestValidationError::PromptTooLong {
            length: 600_000,
            limit: 500_000,
        };
        assert_eq!(err.error_subcode(), Some(ErrorSubcode::InputTooLong));
        assert!(err.to_string().contains("600000"));
    }

    #[test]
    fn test_invalid_model_id_error() {
        let err = RequestValidationError::InvalidModelId {
            model: "../../../etc/passwd".to_string(),
            reason: "path traversal".to_string(),
        };
        assert_eq!(err.error_code(), ErrorCode::InvalidModel);
        assert!(err.to_string().contains("../../../etc/passwd"));
    }

    #[test]
    fn test_to_api_error_includes_limit_actual() {
        let err = RequestValidationError::TooManyMessages {
            count: 300,
            limit: 256,
        };
        let api_err = err.to_api_error("req-123");

        assert_eq!(api_err.error.limit, Some(256));
        assert_eq!(api_err.error.actual, Some(300));
        assert_eq!(api_err.error.param, Some("messages".to_string()));
    }

    // === Validation function tests ===

    #[test]
    fn test_validate_chat_request_empty_messages() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            n: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
        };
        let result = validate_chat_request(&req, &default_limits());
        assert!(matches!(result, Err(RequestValidationError::EmptyMessages)));
    }

    #[test]
    fn test_validate_chat_request_too_many_messages() {
        let messages: Vec<ChatMessage> = (0..300)
            .map(|i| ChatMessage {
                role: "user".to_string(),
                content: format!("message {}", i),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            n: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
        };
        let result = validate_chat_request(&req, &default_limits());
        assert!(matches!(
            result,
            Err(RequestValidationError::TooManyMessages { count: 300, .. })
        ));
    }

    #[test]
    fn test_validate_chat_request_invalid_temperature() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(5.0),
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            n: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
        };
        let result = validate_chat_request(&req, &default_limits());
        assert!(matches!(
            result,
            Err(RequestValidationError::InvalidTemperature { value }) if (value - 5.0).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_validate_chat_request_valid() {
        let req = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(100),
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            n: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
        };
        let result = validate_chat_request(&req, &default_limits());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_id_empty() {
        let result = validate_model_id("");
        assert!(matches!(
            result,
            Err(RequestValidationError::InvalidModelId { .. })
        ));
    }

    #[test]
    fn test_validate_model_id_path_traversal() {
        let result = validate_model_id("../../../etc/passwd");
        assert!(matches!(
            result,
            Err(RequestValidationError::InvalidModelId { .. })
        ));
    }

    #[test]
    fn test_validate_model_id_null_byte() {
        let result = validate_model_id("model\0name");
        assert!(matches!(
            result,
            Err(RequestValidationError::InvalidModelId { .. })
        ));
    }

    #[test]
    fn test_validate_model_id_valid() {
        let result = validate_model_id("meta-llama/Llama-3.2-3B-Instruct");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_model_id_valid_with_special_chars() {
        // These should be valid: hyphens, underscores, forward slashes
        assert!(validate_model_id("my-model_v1/latest").is_ok());
        assert!(validate_model_id("org/model:tag").is_ok());
    }
}
