//! Unified server error type for Infernum.
//!
//! This module provides a single error type that wraps all subsystem errors,
//! enabling consistent error handling throughout the server crate.
//!
//! # Error Hierarchy
//!
//! ```text
//! ServerError (this module)
//!   ├── Config      -> ConfigError
//!   ├── Queue       -> QueueError
//!   ├── Tls         -> TlsError
//!   ├── Grpc        -> GrpcError
//!   ├── Inference   -> infernum_core::Error
//!   ├── Auth        -> Authentication failures
//!   ├── RateLimit   -> Rate limit exceeded
//!   ├── Timeout     -> Request timeout
//!   └── Internal    -> Catch-all for unexpected errors
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::ServerError;
//!
//! fn handle_request() -> Result<Response, ServerError> {
//!     let config = Config::load().map_err(ServerError::Config)?;
//!     Ok(Response::ok())
//! }
//! ```

use std::fmt;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::config_error::ConfigError;
use crate::error_response::{ApiError, ErrorCode};
use crate::grpc::GrpcError;
use crate::queue::QueueError;
use crate::tls::TlsError;

/// Unified error type for the Infernum server.
///
/// This enum consolidates all subsystem errors into a single type, making
/// error handling consistent across the server. Each variant can be converted
/// to an appropriate `ApiError` for HTTP responses.
#[derive(Debug)]
pub enum ServerError {
    /// Configuration error (invalid config, missing files, etc.).
    Config(ConfigError),

    /// Request queue error (overflow, shutdown, etc.).
    Queue(QueueError),

    /// TLS configuration error.
    Tls(TlsError),

    /// gRPC transport error.
    Grpc(GrpcError),

    /// Inference engine error.
    Inference(infernum_core::Error),

    /// Authentication failure.
    Auth {
        /// Description of the auth failure.
        message: String,
    },

    /// Rate limit exceeded.
    RateLimit {
        /// Number of seconds until rate limit resets.
        retry_after_secs: u64,
    },

    /// Request timeout.
    Timeout {
        /// The operation that timed out.
        operation: String,
    },

    /// Internal server error (unexpected conditions).
    Internal {
        /// Error description.
        message: String,
    },
}

impl ServerError {
    /// Creates an authentication error.
    #[must_use]
    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth {
            message: message.into(),
        }
    }

    /// Creates a rate limit error.
    #[must_use]
    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::RateLimit { retry_after_secs }
    }

    /// Creates a timeout error.
    #[must_use]
    pub fn timeout(operation: impl Into<String>) -> Self {
        Self::Timeout {
            operation: operation.into(),
        }
    }

    /// Creates an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Returns the appropriate HTTP status code for this error.
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Queue(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Tls(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Grpc(_) => StatusCode::BAD_GATEWAY,
            Self::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Auth { .. } => StatusCode::UNAUTHORIZED,
            Self::RateLimit { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::Timeout { .. } => StatusCode::REQUEST_TIMEOUT,
            Self::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Returns the error code for API responses.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::Config(_) => ErrorCode::InternalError,
            Self::Queue(_) => ErrorCode::ServiceOverloaded,
            Self::Tls(_) => ErrorCode::InternalError,
            Self::Grpc(_) => ErrorCode::InternalError,
            Self::Inference(_) => ErrorCode::InternalError,
            Self::Auth { .. } => ErrorCode::InvalidApiKey,
            Self::RateLimit { .. } => ErrorCode::RateLimited,
            Self::Timeout { .. } => ErrorCode::Timeout,
            Self::Internal { .. } => ErrorCode::InternalError,
        }
    }

    /// Returns whether this error is retryable.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Queue(_) | Self::RateLimit { .. } | Self::Timeout { .. }
        )
    }

    /// Converts to an API error response.
    #[must_use]
    pub fn to_api_error(&self, request_id: &str) -> ApiError {
        match self {
            Self::RateLimit { retry_after_secs } => ApiError::new(self.error_code(), request_id)
                .message(self.to_string())
                .retry_after(*retry_after_secs)
                .build(),
            _ => ApiError::with_message(self.error_code(), request_id, self.to_string()),
        }
    }
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "Configuration error: {e}"),
            Self::Queue(e) => write!(f, "Queue error: {e}"),
            Self::Tls(e) => write!(f, "TLS error: {e}"),
            Self::Grpc(e) => write!(f, "gRPC error: {e}"),
            Self::Inference(e) => write!(f, "Inference error: {e}"),
            Self::Auth { message } => write!(f, "Authentication failed: {message}"),
            Self::RateLimit { retry_after_secs } => {
                write!(f, "Rate limit exceeded, retry after {retry_after_secs}s")
            }
            Self::Timeout { operation } => write!(f, "Request timed out: {operation}"),
            Self::Internal { message } => write!(f, "Internal error: {message}"),
        }
    }
}

impl std::error::Error for ServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(e) => Some(e),
            Self::Queue(e) => Some(e),
            Self::Tls(e) => Some(e),
            Self::Grpc(e) => Some(e),
            Self::Inference(e) => Some(e),
            Self::Auth { .. }
            | Self::RateLimit { .. }
            | Self::Timeout { .. }
            | Self::Internal { .. } => None,
        }
    }
}

// === From implementations for subsystem errors ===

impl From<ConfigError> for ServerError {
    fn from(err: ConfigError) -> Self {
        Self::Config(err)
    }
}

impl From<QueueError> for ServerError {
    fn from(err: QueueError) -> Self {
        Self::Queue(err)
    }
}

impl From<TlsError> for ServerError {
    fn from(err: TlsError) -> Self {
        Self::Tls(err)
    }
}

impl From<GrpcError> for ServerError {
    fn from(err: GrpcError) -> Self {
        Self::Grpc(err)
    }
}

impl From<infernum_core::Error> for ServerError {
    fn from(err: infernum_core::Error) -> Self {
        Self::Inference(err)
    }
}

// === IntoResponse for direct use in handlers ===

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        // Log the error for debugging
        tracing::error!(error = %self, "Server error occurred");

        // Convert to API error with a placeholder request ID
        // In practice, handlers should use to_api_error with the real request ID
        self.to_api_error("unknown").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Task 2.1: Tests for ServerError enum (TDD) ===

    #[test]
    fn test_server_error_auth_creation() {
        let err = ServerError::auth("Invalid API key");
        assert!(matches!(err, ServerError::Auth { .. }));
        assert_eq!(err.status_code(), StatusCode::UNAUTHORIZED);
        assert_eq!(err.error_code(), ErrorCode::InvalidApiKey);
        assert!(err.to_string().contains("Invalid API key"));
    }

    #[test]
    fn test_server_error_rate_limit_creation() {
        let err = ServerError::rate_limited(60);
        assert!(matches!(err, ServerError::RateLimit { retry_after_secs: 60 }));
        assert_eq!(err.status_code(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.error_code(), ErrorCode::RateLimited);
        assert!(err.to_string().contains("60"));
    }

    #[test]
    fn test_server_error_timeout_creation() {
        let err = ServerError::timeout("chat completion");
        assert!(matches!(err, ServerError::Timeout { .. }));
        assert_eq!(err.status_code(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(err.error_code(), ErrorCode::Timeout);
        assert!(err.to_string().contains("chat completion"));
    }

    #[test]
    fn test_server_error_internal_creation() {
        let err = ServerError::internal("Unexpected null pointer");
        assert!(matches!(err, ServerError::Internal { .. }));
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error_code(), ErrorCode::InternalError);
        assert!(err.to_string().contains("Unexpected null pointer"));
    }

    #[test]
    fn test_server_error_retryable() {
        // Retryable errors
        assert!(ServerError::rate_limited(60).is_retryable());
        assert!(ServerError::timeout("op").is_retryable());

        // Non-retryable errors
        assert!(!ServerError::auth("bad key").is_retryable());
        assert!(!ServerError::internal("oops").is_retryable());
    }

    #[test]
    fn test_server_error_to_api_error() {
        let err = ServerError::auth("Bad token");
        let api_err = err.to_api_error("req-123");

        assert_eq!(api_err.error.code, ErrorCode::InvalidApiKey);
        assert_eq!(api_err.error.request_id, "req-123");
        assert!(api_err.error.message.contains("Bad token"));
    }

    #[test]
    fn test_server_error_rate_limit_includes_retry_after() {
        let err = ServerError::rate_limited(120);
        let api_err = err.to_api_error("req-456");

        assert_eq!(api_err.error.code, ErrorCode::RateLimited);
        assert!(api_err.error.retry.is_some());
        let retry = api_err.error.retry.as_ref().unwrap();
        assert_eq!(retry.after_seconds, Some(120));
    }

    #[test]
    fn test_server_error_status_codes() {
        // Map each variant to expected status code
        assert_eq!(
            ServerError::auth("x").status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            ServerError::rate_limited(1).status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ServerError::timeout("x").status_code(),
            StatusCode::REQUEST_TIMEOUT
        );
        assert_eq!(
            ServerError::internal("x").status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_server_error_display_format() {
        let auth_err = ServerError::auth("token expired");
        assert_eq!(
            auth_err.to_string(),
            "Authentication failed: token expired"
        );

        let rate_err = ServerError::rate_limited(30);
        assert_eq!(
            rate_err.to_string(),
            "Rate limit exceeded, retry after 30s"
        );

        let timeout_err = ServerError::timeout("embedding");
        assert_eq!(timeout_err.to_string(), "Request timed out: embedding");

        let internal_err = ServerError::internal("disk full");
        assert_eq!(internal_err.to_string(), "Internal error: disk full");
    }

    #[test]
    fn test_server_error_from_config_error() {
        let config_err = ConfigError::out_of_range("port", 0, 1, 65535);
        let server_err: ServerError = config_err.into();

        assert!(matches!(server_err, ServerError::Config(_)));
        assert_eq!(server_err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Note: More From tests would require constructing other error types
    // which may have complex construction. These are tested at integration level.
}
