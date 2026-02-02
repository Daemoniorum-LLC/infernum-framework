//! Request handlers for the Infernum API.
//!
//! This module provides handler utilities and documentation for the API endpoints.
//! The actual handler implementations are in [`crate::server`] alongside the routing.
//!
//! # Endpoint Overview
//!
//! ## Health & Status
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/health` | GET | Server health check (always returns ok) |
//! | `/ready` | GET | Readiness check (returns model status) |
//! | `/metrics` | GET | Prometheus metrics |
//!
//! ## OpenAI-Compatible Endpoints
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/v1/chat/completions` | POST | Chat completion (supports streaming) |
//! | `/v1/completions` | POST | Text completion |
//! | `/v1/embeddings` | POST | Generate embeddings |
//! | `/v1/models` | GET | List available models |
//!
//! ## Model Management
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/api/models/load` | POST | Load a model |
//! | `/api/models/unload` | POST | Unload the current model |
//!
//! # Authentication
//!
//! Endpoints (except `/health`, `/ready`, `/metrics`) require authentication
//! when auth is enabled. Pass the API key in the `Authorization` header:
//!
//! ```text
//! Authorization: Bearer sk-inf-your-api-key
//! ```
//!
//! # Error Responses
//!
//! All errors follow the OpenAI error format:
//!
//! ```json
//! {
//!   "error": {
//!     "message": "Description of what went wrong",
//!     "type": "invalid_request_error",
//!     "code": "invalid_messages",
//!     "param": "messages"
//!   }
//! }
//! ```
//!
//! # Streaming
//!
//! Chat and completion endpoints support streaming via `"stream": true`.
//! Streaming responses use Server-Sent Events (SSE) format.

use std::future::Future;
use std::pin::Pin;

use axum::response::Response;

/// Type alias for async handler results.
pub type HandlerResult = Pin<Box<dyn Future<Output = Response> + Send>>;

/// Handler configuration options.
#[derive(Debug, Clone, Default)]
pub struct HandlerConfig {
    /// Enable detailed error messages (development mode).
    pub debug_errors: bool,
    /// Include timing information in responses.
    pub include_timing: bool,
    /// Maximum response time before timeout warning.
    pub slow_request_threshold_ms: u64,
}

impl HandlerConfig {
    /// Creates a development configuration.
    pub fn development() -> Self {
        Self {
            debug_errors: true,
            include_timing: true,
            slow_request_threshold_ms: 1000,
        }
    }

    /// Creates a production configuration.
    pub fn production() -> Self {
        Self {
            debug_errors: false,
            include_timing: false,
            slow_request_threshold_ms: 5000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_config_development() {
        let config = HandlerConfig::development();
        assert!(config.debug_errors);
        assert!(config.include_timing);
        assert_eq!(config.slow_request_threshold_ms, 1000);
    }

    #[test]
    fn test_handler_config_production() {
        let config = HandlerConfig::production();
        assert!(!config.debug_errors);
        assert!(!config.include_timing);
        assert_eq!(config.slow_request_threshold_ms, 5000);
    }

    #[test]
    fn test_handler_config_default() {
        let config = HandlerConfig::default();
        assert!(!config.debug_errors);
        assert!(!config.include_timing);
    }
}
