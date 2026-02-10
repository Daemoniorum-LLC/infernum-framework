//! Standardized error response types for the Infernum API.
//!
//! This module provides a comprehensive error handling system including:
//!
//! - Machine-readable error codes for programmatic handling
//! - Human-readable messages with recovery hints
//! - Request ID tracking for debugging
//! - Retry information for transient errors
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::error_response::{ApiError, ErrorCode};
//!
//! let error = ApiError::new(ErrorCode::ModelNotLoaded, "req-12345")
//!     .with_message("No model is currently loaded")
//!     .build();
//! ```

use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::Serialize;
use utoipa::ToSchema;

/// Standardized API error response wrapper.
#[derive(Debug, Serialize, ToSchema)]
pub struct ApiError {
    /// The error details.
    pub error: ErrorDetail,
}

/// Detailed error information.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    /// Human-readable error message (safe for end users).
    pub message: String,

    /// Error type category for programmatic handling.
    #[serde(rename = "type")]
    pub error_type: ErrorType,

    /// Machine-readable error code.
    pub code: ErrorCode,

    /// More specific error subcode for granular handling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subcode: Option<ErrorSubcode>,

    /// Request ID for correlation and debugging.
    pub request_id: String,

    /// ISO 8601 timestamp of when the error occurred.
    pub timestamp: DateTime<Utc>,

    /// Actionable recovery hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,

    /// Parameter that caused the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,

    /// The limit that was exceeded (for validation errors).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,

    /// The actual value that exceeded the limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<u64>,

    /// Documentation link for this error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_url: Option<String>,

    /// Retry information for transient errors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryInfo>,
}

/// Information about retry behavior for transient errors.
#[derive(Debug, Serialize, ToSchema)]
pub struct RetryInfo {
    /// Whether the request can be retried.
    pub retryable: bool,
    /// Suggested retry delay in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_seconds: Option<u64>,
}

/// Error type categories for the Infernum API.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    /// Invalid request parameters or format.
    InvalidRequestError,
    /// Authentication failure (missing or invalid API key).
    AuthenticationError,
    /// Authorization failure (valid key but insufficient permissions).
    PermissionDeniedError,
    /// Requested resource not found.
    NotFoundError,
    /// Rate limit exceeded.
    RateLimitError,
    /// Internal server error.
    ServerError,
    /// Service temporarily unavailable.
    ServiceUnavailableError,
    /// Request timeout.
    TimeoutError,
}

/// Specific error subcodes for granular error handling.
///
/// Subcodes provide more specific information about what caused an error,
/// enabling more precise programmatic handling and error recovery.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorSubcode {
    // === Context Length Subcodes ===
    /// Input (prompt or messages) exceeds context window.
    InputTooLong,
    /// Output would exceed remaining context.
    OutputTooLong,
    /// Combined input + output exceeds context.
    TotalTooLong,

    // === Validation Subcodes ===
    /// Value is below minimum.
    BelowMinimum,
    /// Value is above maximum.
    AboveMaximum,
    /// Value is out of valid range.
    OutOfRange,
    /// Required field is missing.
    MissingRequired,
    /// Field has invalid format.
    InvalidFormat,
    /// Field type is incorrect.
    InvalidType,

    // === Message Subcodes ===
    /// Individual message is too long.
    SingleMessageTooLong,
    /// Too many messages in array.
    TooManyItems,
    /// Invalid message role.
    InvalidRole,
    /// Missing message content.
    MissingContent,

    // === Model Subcodes ===
    /// Model identifier is malformed.
    MalformedIdentifier,
    /// Model architecture not supported.
    UnsupportedArchitecture,
    /// Model file is corrupted.
    CorruptedModel,
    /// Model is still loading.
    ModelLoading,

    // === Rate Limit Subcodes ===
    /// Per-minute limit exceeded.
    MinuteLimit,
    /// Per-hour limit exceeded.
    HourLimit,
    /// Per-day limit exceeded.
    DayLimit,
    /// Concurrent request limit exceeded.
    ConcurrentLimit,

    // === Authentication Subcodes ===
    /// API key format is invalid.
    MalformedKey,
    /// API key not found in database.
    KeyNotFound,
    /// API key has been revoked.
    KeyRevoked,

    // === Content Filter Subcodes ===
    /// Content violates safety policy.
    SafetyViolation,
    /// Content contains disallowed topics.
    DisallowedTopic,
    /// Output was truncated due to filter.
    FilterTruncated,
}

impl ErrorSubcode {
    /// Returns a human-readable description of this subcode.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::InputTooLong => "Input exceeds context window limit",
            Self::OutputTooLong => "Requested output exceeds remaining context",
            Self::TotalTooLong => "Combined input and output exceed context limit",
            Self::BelowMinimum => "Value is below the minimum allowed",
            Self::AboveMaximum => "Value is above the maximum allowed",
            Self::OutOfRange => "Value is outside the valid range",
            Self::MissingRequired => "A required field is missing",
            Self::InvalidFormat => "Field value has an invalid format",
            Self::InvalidType => "Field value has an incorrect type",
            Self::SingleMessageTooLong => "A single message exceeds the length limit",
            Self::TooManyItems => "Too many items in the array",
            Self::InvalidRole => "Message role is not recognized",
            Self::MissingContent => "Message is missing required content",
            Self::MalformedIdentifier => "Model identifier format is invalid",
            Self::UnsupportedArchitecture => "Model architecture is not supported",
            Self::CorruptedModel => "Model file appears to be corrupted",
            Self::ModelLoading => "Model is currently loading",
            Self::MinuteLimit => "Per-minute rate limit exceeded",
            Self::HourLimit => "Per-hour rate limit exceeded",
            Self::DayLimit => "Per-day rate limit exceeded",
            Self::ConcurrentLimit => "Concurrent request limit exceeded",
            Self::MalformedKey => "API key format is invalid",
            Self::KeyNotFound => "API key was not found",
            Self::KeyRevoked => "API key has been revoked",
            Self::SafetyViolation => "Content violates safety policies",
            Self::DisallowedTopic => "Content contains disallowed topics",
            Self::FilterTruncated => "Output was truncated by content filter",
        }
    }
}

/// Specific error codes for programmatic handling.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // === Request Validation Errors ===
    /// Invalid model identifier.
    InvalidModel,
    /// Invalid messages array format.
    InvalidMessages,
    /// Temperature parameter out of range.
    InvalidTemperature,
    /// max_tokens parameter invalid.
    InvalidMaxTokens,
    /// Individual message too long.
    MessageTooLong,
    /// Too many messages in the conversation.
    TooManyMessages,
    /// Prompt is empty.
    EmptyPrompt,
    /// Prompt exceeds maximum length.
    PromptTooLong,

    // === Authentication Errors ===
    /// API key is missing or invalid.
    InvalidApiKey,
    /// API key has expired.
    ExpiredApiKey,
    /// API key lacks required scope.
    InsufficientScope,

    // === Resource Errors ===
    /// No model is loaded.
    ModelNotLoaded,
    /// Requested model not found.
    ModelNotFound,
    /// Context length exceeded.
    ContextLengthExceeded,
    /// Content was filtered.
    ContentFiltered,

    // === System Errors ===
    /// Internal server error.
    InternalError,
    /// Server is overloaded.
    ServiceOverloaded,
    /// Request timed out.
    Timeout,
    /// Rate limit exceeded.
    RateLimited,

    // === Embedding Specific ===
    /// Failed to extract embedding from model.
    EmbeddingExtractionFailed,
    /// Invalid embedding input format.
    InvalidEmbeddingInput,

    // === General API Errors ===
    /// Generic invalid request (bad input).
    InvalidRequest,
    /// Resource not found.
    NotFound,
    /// Service is unavailable (e.g., not initialized).
    ServiceUnavailable,
}

impl ErrorCode {
    /// Returns the appropriate HTTP status code for this error.
    #[must_use]
    #[inline]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            // 400 Bad Request
            Self::InvalidModel
            | Self::InvalidMessages
            | Self::InvalidTemperature
            | Self::InvalidMaxTokens
            | Self::MessageTooLong
            | Self::TooManyMessages
            | Self::EmptyPrompt
            | Self::PromptTooLong
            | Self::InvalidEmbeddingInput
            | Self::InvalidRequest => StatusCode::BAD_REQUEST,

            // 401 Unauthorized
            Self::InvalidApiKey | Self::ExpiredApiKey => StatusCode::UNAUTHORIZED,

            // 403 Forbidden
            Self::InsufficientScope => StatusCode::FORBIDDEN,

            // 404 Not Found
            Self::ModelNotFound | Self::NotFound => StatusCode::NOT_FOUND,

            // 408 Request Timeout
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,

            // 429 Too Many Requests
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,

            // 500 Internal Server Error
            Self::InternalError
            | Self::ContextLengthExceeded
            | Self::ContentFiltered
            | Self::EmbeddingExtractionFailed => StatusCode::INTERNAL_SERVER_ERROR,

            // 503 Service Unavailable
            Self::ModelNotLoaded | Self::ServiceOverloaded | Self::ServiceUnavailable => {
                StatusCode::SERVICE_UNAVAILABLE
            },
        }
    }

    /// Returns an actionable recovery hint for this error, if applicable.
    #[must_use]
    #[inline]
    pub const fn hint(&self) -> Option<&'static str> {
        match self {
            Self::InvalidTemperature => Some("Temperature must be between 0.0 and 2.0"),
            Self::InvalidMaxTokens => Some("max_tokens must be between 1 and the model's maximum"),
            Self::MessageTooLong => Some("Reduce message length or split into multiple messages"),
            Self::TooManyMessages => {
                Some("Reduce conversation history or summarize earlier messages")
            },
            Self::ContextLengthExceeded => {
                Some("Reduce prompt length or use a model with larger context")
            },
            Self::ModelNotLoaded => {
                Some("Load a model via POST /api/models/load or restart with --model")
            },
            Self::RateLimited => {
                Some("Reduce request frequency or contact support for higher limits")
            },
            Self::ServiceOverloaded => {
                Some("Server is at capacity. Retry with exponential backoff")
            },
            Self::InsufficientScope => {
                Some("API key lacks required scope. Use a key with appropriate permissions")
            },
            Self::ExpiredApiKey => Some("API key has expired. Generate a new key"),
            Self::EmbeddingExtractionFailed => {
                Some("Embedding extraction failed. This may indicate a model compatibility issue")
            },
            Self::EmptyPrompt => Some("Provide a non-empty prompt or messages array"),
            Self::PromptTooLong => Some("Reduce prompt length to within model limits"),
            Self::InvalidModel => Some("Check the model identifier format and availability"),
            _ => None,
        }
    }

    /// Returns the error type category for this error code.
    #[must_use]
    #[inline]
    pub const fn error_type(&self) -> ErrorType {
        match self {
            Self::InvalidModel
            | Self::InvalidMessages
            | Self::InvalidTemperature
            | Self::InvalidMaxTokens
            | Self::MessageTooLong
            | Self::TooManyMessages
            | Self::EmptyPrompt
            | Self::PromptTooLong
            | Self::InvalidEmbeddingInput
            | Self::InvalidRequest => ErrorType::InvalidRequestError,

            Self::InvalidApiKey | Self::ExpiredApiKey => ErrorType::AuthenticationError,

            Self::InsufficientScope => ErrorType::PermissionDeniedError,

            Self::ModelNotFound | Self::NotFound => ErrorType::NotFoundError,

            Self::RateLimited => ErrorType::RateLimitError,

            Self::Timeout => ErrorType::TimeoutError,

            Self::ModelNotLoaded | Self::ServiceOverloaded | Self::ServiceUnavailable => {
                ErrorType::ServiceUnavailableError
            },

            Self::InternalError
            | Self::ContextLengthExceeded
            | Self::ContentFiltered
            | Self::EmbeddingExtractionFailed => ErrorType::ServerError,
        }
    }

    /// Returns whether this error is retryable.
    #[must_use]
    #[inline]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited | Self::ServiceOverloaded | Self::Timeout
        )
    }

    /// Returns the default message for this error code.
    #[must_use]
    #[inline]
    pub const fn default_message(&self) -> &'static str {
        match self {
            Self::InvalidModel => "Invalid model identifier",
            Self::InvalidMessages => "Invalid messages array",
            Self::InvalidTemperature => "Invalid temperature value",
            Self::InvalidMaxTokens => "Invalid max_tokens value",
            Self::MessageTooLong => "Message exceeds maximum length",
            Self::TooManyMessages => "Too many messages in request",
            Self::EmptyPrompt => "Prompt cannot be empty",
            Self::PromptTooLong => "Prompt exceeds maximum length",
            Self::InvalidApiKey => "Invalid API key",
            Self::ExpiredApiKey => "API key has expired",
            Self::InsufficientScope => "Insufficient permissions for this operation",
            Self::ModelNotLoaded => "No model is currently loaded",
            Self::ModelNotFound => "Model not found",
            Self::ContextLengthExceeded => "Context length exceeded",
            Self::ContentFiltered => "Content was filtered due to policy violation",
            Self::InternalError => "An internal error occurred",
            Self::ServiceOverloaded => "Server is currently overloaded",
            Self::Timeout => "Request timed out",
            Self::RateLimited => "Rate limit exceeded",
            Self::EmbeddingExtractionFailed => "Failed to extract embedding from model output",
            Self::InvalidEmbeddingInput => "Invalid embedding input",
            Self::InvalidRequest => "Invalid request",
            Self::NotFound => "Resource not found",
            Self::ServiceUnavailable => "Service is unavailable",
        }
    }
}

/// Builder for constructing API errors with fluent interface.
pub struct ApiErrorBuilder {
    code: ErrorCode,
    subcode: Option<ErrorSubcode>,
    message: Option<String>,
    request_id: String,
    param: Option<String>,
    limit: Option<u64>,
    actual: Option<u64>,
    retry_after: Option<u64>,
    include_doc_url: bool,
}

impl ApiErrorBuilder {
    /// Creates a new error builder with the given code and request ID.
    #[must_use]
    pub fn new(code: ErrorCode, request_id: impl Into<String>) -> Self {
        Self {
            code,
            subcode: None,
            message: None,
            request_id: request_id.into(),
            param: None,
            limit: None,
            actual: None,
            retry_after: None,
            include_doc_url: true,
        }
    }

    /// Override the default error message.
    #[must_use]
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Specify a more detailed error subcode.
    #[must_use]
    pub fn subcode(mut self, subcode: ErrorSubcode) -> Self {
        self.subcode = Some(subcode);
        self
    }

    /// Specify the parameter that caused the error.
    #[must_use]
    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// Specify the limit that was exceeded.
    #[must_use]
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Specify the actual value that exceeded the limit.
    #[must_use]
    pub fn actual(mut self, actual: u64) -> Self {
        self.actual = Some(actual);
        self
    }

    /// Specify both limit and actual value for validation errors.
    #[must_use]
    pub fn limit_exceeded(mut self, limit: u64, actual: u64) -> Self {
        self.limit = Some(limit);
        self.actual = Some(actual);
        self
    }

    /// Specify retry delay for retryable errors.
    #[must_use]
    pub fn retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// Disable including the documentation URL.
    #[must_use]
    pub fn without_doc_url(mut self) -> Self {
        self.include_doc_url = false;
        self
    }

    /// Build the error response.
    #[must_use]
    pub fn build(self) -> ApiError {
        let message = self
            .message
            .unwrap_or_else(|| self.code.default_message().to_string());

        let retryable = self.code.is_retryable();

        let retry = if retryable || self.retry_after.is_some() {
            Some(RetryInfo {
                retryable,
                after_seconds: self.retry_after,
            })
        } else {
            None
        };

        ApiError {
            error: ErrorDetail {
                message,
                error_type: self.code.error_type(),
                code: self.code,
                subcode: self.subcode,
                request_id: self.request_id,
                timestamp: Utc::now(),
                hint: self.code.hint().map(String::from),
                param: self.param,
                limit: self.limit,
                actual: self.actual,
                doc_url: if self.include_doc_url {
                    Some("https://infernum.dev/docs/errors".to_string())
                } else {
                    None
                },
                retry,
            },
        }
    }
}

impl ApiError {
    /// Creates a new error builder with the given code and request ID.
    #[must_use]
    pub fn new(code: ErrorCode, request_id: impl Into<String>) -> ApiErrorBuilder {
        ApiErrorBuilder::new(code, request_id)
    }

    /// Convenience method to create an error with default message.
    #[must_use]
    pub fn from_code(code: ErrorCode, request_id: impl Into<String>) -> Self {
        ApiErrorBuilder::new(code, request_id).build()
    }

    /// Convenience method to create an error with custom message.
    #[must_use]
    pub fn with_message(
        code: ErrorCode,
        request_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        ApiErrorBuilder::new(code, request_id)
            .message(message)
            .build()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.error.code.status_code();
        let mut headers = HeaderMap::new();

        // Add request ID header for correlation
        if let Ok(val) = HeaderValue::from_str(&self.error.request_id) {
            headers.insert("x-request-id", val);
        }

        // Add Retry-After header for rate limits and overload
        if let Some(retry) = &self.error.retry {
            if let Some(after) = retry.after_seconds {
                if let Ok(val) = HeaderValue::from_str(&after.to_string()) {
                    headers.insert("retry-after", val);
                }
            }
        }

        (status, headers, Json(self)).into_response()
    }
}

/// Convenience function for creating error responses.
#[inline]
#[must_use]
pub fn api_error(code: ErrorCode, request_id: &str) -> ApiError {
    ApiError::from_code(code, request_id)
}

/// Convenience function for creating error responses with custom message.
#[inline]
#[must_use]
pub fn api_error_with_message(code: ErrorCode, request_id: &str, message: &str) -> ApiError {
    ApiError::with_message(code, request_id, message)
}

/// Sanitizes internal errors for external display.
///
/// Removes sensitive information like file paths and memory addresses.
#[must_use]
pub fn sanitize_error(error: &str) -> String {
    // Remove file paths (Unix and Windows)
    let mut sanitized = error.to_string();

    // Remove Unix paths
    let unix_path_pattern =
        regex::Regex::new(r"(/[a-zA-Z0-9_./-]+)+").expect("valid regex pattern");
    sanitized = unix_path_pattern
        .replace_all(&sanitized, "[path]")
        .to_string();

    // Remove Windows paths
    let windows_path_pattern =
        regex::Regex::new(r"[A-Za-z]:\\[^\s]+").expect("valid regex pattern");
    sanitized = windows_path_pattern
        .replace_all(&sanitized, "[path]")
        .to_string();

    // Remove memory addresses
    let addr_pattern = regex::Regex::new(r"0x[0-9a-fA-F]+").expect("valid regex pattern");
    sanitized = addr_pattern.replace_all(&sanitized, "[addr]").to_string();

    // Truncate very long messages
    if sanitized.len() > 200 {
        format!("{}...", &sanitized[..200])
    } else {
        sanitized
    }
}

/// Handles internal errors by logging the full error and returning a sanitized response.
#[must_use]
pub fn handle_internal_error(error: &dyn std::error::Error, request_id: &str) -> ApiError {
    // Log full error for debugging
    tracing::error!(
        request_id = %request_id,
        error = %error,
        "Internal error occurred"
    );

    // Return sanitized error to client
    api_error_with_message(
        ErrorCode::InternalError,
        request_id,
        &sanitize_error(&error.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_status_codes() {
        assert_eq!(
            ErrorCode::InvalidModel.status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ErrorCode::InvalidApiKey.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ErrorCode::InsufficientScope.status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            ErrorCode::ModelNotFound.status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ErrorCode::Timeout.status_code(),
            StatusCode::REQUEST_TIMEOUT
        );
        assert_eq!(
            ErrorCode::RateLimited.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ErrorCode::ModelNotLoaded.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            ErrorCode::InternalError.status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_code_hints() {
        assert!(ErrorCode::InvalidTemperature.hint().is_some());
        assert!(ErrorCode::ModelNotLoaded.hint().is_some());
        assert!(ErrorCode::RateLimited.hint().is_some());
        // InternalError has no specific hint
        assert!(ErrorCode::InternalError.hint().is_none());
    }

    #[test]
    fn test_error_code_retryable() {
        assert!(ErrorCode::RateLimited.is_retryable());
        assert!(ErrorCode::ServiceOverloaded.is_retryable());
        assert!(ErrorCode::Timeout.is_retryable());
        assert!(!ErrorCode::InvalidModel.is_retryable());
        assert!(!ErrorCode::InternalError.is_retryable());
    }

    #[test]
    fn test_api_error_builder() {
        let error = ApiError::new(ErrorCode::InvalidTemperature, "test-123")
            .message("Custom temperature error")
            .param("temperature")
            .build();

        assert_eq!(error.error.code, ErrorCode::InvalidTemperature);
        assert_eq!(error.error.request_id, "test-123");
        assert_eq!(error.error.message, "Custom temperature error");
        assert_eq!(error.error.param, Some("temperature".to_string()));
        assert!(error.error.hint.is_some());
    }

    #[test]
    fn test_api_error_convenience() {
        let error = api_error(ErrorCode::ModelNotLoaded, "req-456");
        assert_eq!(error.error.code, ErrorCode::ModelNotLoaded);
        assert_eq!(error.error.message, "No model is currently loaded");
    }

    #[test]
    fn test_sanitize_error() {
        // Test path removal
        let error = "Error at /home/user/project/src/main.rs:42";
        let sanitized = sanitize_error(error);
        assert!(!sanitized.contains("/home"));
        assert!(sanitized.contains("[path]"));

        // Test address removal
        let error = "Segfault at 0xDEADBEEF";
        let sanitized = sanitize_error(error);
        assert!(!sanitized.contains("0xDEADBEEF"));
        assert!(sanitized.contains("[addr]"));

        // Test truncation
        let long_error = "x".repeat(300);
        let sanitized = sanitize_error(&long_error);
        assert!(sanitized.len() <= 203); // 200 + "..."
        assert!(sanitized.ends_with("..."));
    }

    #[test]
    fn test_error_types() {
        assert_eq!(
            ErrorCode::InvalidModel.error_type(),
            ErrorType::InvalidRequestError
        );
        assert_eq!(
            ErrorCode::InvalidApiKey.error_type(),
            ErrorType::AuthenticationError
        );
        assert_eq!(
            ErrorCode::InsufficientScope.error_type(),
            ErrorType::PermissionDeniedError
        );
        assert_eq!(
            ErrorCode::ModelNotLoaded.error_type(),
            ErrorType::ServiceUnavailableError
        );
    }

    #[test]
    fn test_retry_info_for_retryable_errors() {
        let error = api_error(ErrorCode::RateLimited, "req-789");
        assert!(error.error.retry.is_some());
        let retry = error.error.retry.unwrap();
        assert!(retry.retryable);
    }

    #[test]
    fn test_retry_after_seconds() {
        let error = ApiError::new(ErrorCode::RateLimited, "req-xyz")
            .retry_after(60)
            .build();

        let retry = error.error.retry.unwrap();
        assert_eq!(retry.after_seconds, Some(60));
    }

    #[test]
    fn test_error_subcode() {
        let error = ApiError::new(ErrorCode::ContextLengthExceeded, "req-sub-1")
            .subcode(ErrorSubcode::InputTooLong)
            .build();

        assert_eq!(error.error.subcode, Some(ErrorSubcode::InputTooLong));
    }

    #[test]
    fn test_error_subcode_description() {
        assert_eq!(
            ErrorSubcode::InputTooLong.description(),
            "Input exceeds context window limit"
        );
        assert_eq!(
            ErrorSubcode::AboveMaximum.description(),
            "Value is above the maximum allowed"
        );
        assert_eq!(
            ErrorSubcode::MinuteLimit.description(),
            "Per-minute rate limit exceeded"
        );
    }

    #[test]
    fn test_error_limit_and_actual() {
        let error = ApiError::new(ErrorCode::ContextLengthExceeded, "req-lim-1")
            .subcode(ErrorSubcode::InputTooLong)
            .param("messages")
            .limit(8192)
            .actual(12000)
            .build();

        assert_eq!(error.error.limit, Some(8192));
        assert_eq!(error.error.actual, Some(12000));
        assert_eq!(error.error.param, Some("messages".to_string()));
    }

    #[test]
    fn test_error_limit_exceeded_convenience() {
        let error = ApiError::new(ErrorCode::TooManyMessages, "req-lim-2")
            .subcode(ErrorSubcode::TooManyItems)
            .limit_exceeded(256, 500)
            .build();

        assert_eq!(error.error.limit, Some(256));
        assert_eq!(error.error.actual, Some(500));
    }

    #[test]
    fn test_error_serialization_with_subcode() {
        let error = ApiError::new(ErrorCode::ContextLengthExceeded, "req-ser-1")
            .subcode(ErrorSubcode::InputTooLong)
            .param("messages")
            .limit(8192)
            .actual(12000)
            .build();

        let json = serde_json::to_string(&error).expect("should serialize");
        assert!(json.contains("\"subcode\":\"input_too_long\""));
        assert!(json.contains("\"limit\":8192"));
        assert!(json.contains("\"actual\":12000"));
    }

    #[test]
    fn test_error_serialization_without_subcode() {
        let error = api_error(ErrorCode::InvalidModel, "req-ser-2");
        let json = serde_json::to_string(&error).expect("should serialize");
        // subcode should not be present if None
        assert!(!json.contains("subcode"));
    }

    #[test]
    fn test_all_error_subcodes_have_descriptions() {
        let subcodes = [
            ErrorSubcode::InputTooLong,
            ErrorSubcode::OutputTooLong,
            ErrorSubcode::TotalTooLong,
            ErrorSubcode::BelowMinimum,
            ErrorSubcode::AboveMaximum,
            ErrorSubcode::OutOfRange,
            ErrorSubcode::MissingRequired,
            ErrorSubcode::InvalidFormat,
            ErrorSubcode::InvalidType,
            ErrorSubcode::SingleMessageTooLong,
            ErrorSubcode::TooManyItems,
            ErrorSubcode::InvalidRole,
            ErrorSubcode::MissingContent,
            ErrorSubcode::MalformedIdentifier,
            ErrorSubcode::UnsupportedArchitecture,
            ErrorSubcode::CorruptedModel,
            ErrorSubcode::ModelLoading,
            ErrorSubcode::MinuteLimit,
            ErrorSubcode::HourLimit,
            ErrorSubcode::DayLimit,
            ErrorSubcode::ConcurrentLimit,
            ErrorSubcode::MalformedKey,
            ErrorSubcode::KeyNotFound,
            ErrorSubcode::KeyRevoked,
            ErrorSubcode::SafetyViolation,
            ErrorSubcode::DisallowedTopic,
            ErrorSubcode::FilterTruncated,
        ];

        for subcode in subcodes {
            let desc = subcode.description();
            assert!(
                !desc.is_empty(),
                "Subcode {:?} should have description",
                subcode
            );
        }
    }

    #[test]
    fn test_rate_limit_with_subcode() {
        let error = ApiError::new(ErrorCode::RateLimited, "req-rate-1")
            .subcode(ErrorSubcode::MinuteLimit)
            .retry_after(60)
            .build();

        assert_eq!(error.error.code, ErrorCode::RateLimited);
        assert_eq!(error.error.subcode, Some(ErrorSubcode::MinuteLimit));
        assert!(error.error.retry.is_some());
    }
}
