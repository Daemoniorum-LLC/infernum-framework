//! Standardized API response types.
//!
//! Provides consistent response formats for all endpoints, ensuring
//! predictable API behavior for clients.
//!
//! # Response Format
//!
//! All successful responses include:
//! - Relevant data fields
//! - Timestamp in ISO 8601 format
//! - Request metadata where applicable
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::responses::HealthResponse;
//!
//! let response = HealthResponse::new("1.0.0", 3600);
//! assert_eq!(response.status, "ok");
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Health check response.
///
/// Returned by the `/health` endpoint to indicate server status.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HealthResponse {
    /// Health status (always "ok" if endpoint is reachable).
    pub status: &'static str,
    /// Server version from Cargo.toml.
    pub version: String,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
    /// Current timestamp.
    pub timestamp: DateTime<Utc>,
}

impl HealthResponse {
    /// Creates a new health response.
    pub fn new(version: &str, uptime_seconds: u64) -> Self {
        Self {
            status: "ok",
            version: version.to_string(),
            uptime_seconds,
            timestamp: Utc::now(),
        }
    }

    /// Creates a health response using the crate version.
    pub fn from_uptime(uptime_seconds: u64) -> Self {
        Self::new(env!("CARGO_PKG_VERSION"), uptime_seconds)
    }
}

/// Readiness check response.
///
/// Returned by the `/ready` endpoint to indicate whether the server
/// is ready to accept inference requests.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ReadyResponse {
    /// Whether the server is ready (model is loaded).
    pub ready: bool,
    /// Information about the loaded model, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelInfo>,
    /// Current timestamp.
    pub timestamp: DateTime<Utc>,
}

impl ReadyResponse {
    /// Creates a not-ready response.
    pub fn not_ready() -> Self {
        Self {
            ready: false,
            model: None,
            timestamp: Utc::now(),
        }
    }

    /// Creates a ready response with model information.
    pub fn ready_with_model(model: ModelInfo) -> Self {
        Self {
            ready: true,
            model: Some(model),
            timestamp: Utc::now(),
        }
    }
}

/// Information about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// When the model was loaded.
    pub loaded_at: DateTime<Utc>,
    /// Model memory usage in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    /// Model context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

impl ModelInfo {
    /// Creates a new model info.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            loaded_at: Utc::now(),
            memory_bytes: None,
            context_length: None,
        }
    }

    /// Sets the memory usage.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Sets the context length.
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }
}

/// Metrics response.
///
/// Returned by the `/metrics` endpoint in JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    /// Queue depth (requests waiting).
    pub queue_depth: u64,
    /// Queue capacity.
    pub queue_capacity: usize,
    /// Maximum concurrent requests allowed.
    pub concurrent_requests_limit: usize,
    /// Currently active requests.
    pub active_requests: u64,
    /// Total requests served.
    pub total_requests_served: u64,
    /// Total failed requests.
    pub failed_requests: u64,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
    /// Current timestamp.
    pub timestamp: DateTime<Utc>,
}

/// Response metadata.
///
/// Included in responses that need request tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMeta {
    /// Request ID for tracing.
    pub request_id: String,
    /// Timestamp of the response.
    pub timestamp: DateTime<Utc>,
    /// Processing time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,
}

impl ResponseMeta {
    /// Creates new response metadata.
    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            timestamp: Utc::now(),
            processing_time_ms: None,
        }
    }

    /// Adds processing time.
    pub fn with_processing_time(mut self, ms: u64) -> Self {
        self.processing_time_ms = Some(ms);
        self
    }
}

/// Generic API response wrapper.
///
/// Wraps any response type with success indicator and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Whether the request succeeded.
    pub success: bool,
    /// Response data.
    pub data: T,
    /// Response metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResponseMeta>,
}

impl<T> ApiResponse<T> {
    /// Creates a successful response.
    pub fn ok(data: T) -> Self {
        Self {
            success: true,
            data,
            meta: None,
        }
    }

    /// Adds metadata to the response.
    pub fn with_meta(mut self, meta: ResponseMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Adds request ID to the response.
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.meta = Some(ResponseMeta::new(request_id));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response() {
        let response = HealthResponse::new("1.0.0", 3600);
        assert_eq!(response.status, "ok");
        assert_eq!(response.version, "1.0.0");
        assert_eq!(response.uptime_seconds, 3600);
    }

    #[test]
    fn test_ready_response_not_ready() {
        let response = ReadyResponse::not_ready();
        assert!(!response.ready);
        assert!(response.model.is_none());
    }

    #[test]
    fn test_ready_response_with_model() {
        let model = ModelInfo::new("test-model")
            .with_memory(1024 * 1024 * 1024)
            .with_context_length(8192);
        let response = ReadyResponse::ready_with_model(model);
        assert!(response.ready);
        assert!(response.model.is_some());
        let m = response.model.unwrap();
        assert_eq!(m.id, "test-model");
        assert_eq!(m.memory_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(m.context_length, Some(8192));
    }

    #[test]
    fn test_api_response_wrapper() {
        let inner = HealthResponse::new("1.0.0", 100);
        let response = ApiResponse::ok(inner)
            .with_request_id("req-123");
        assert!(response.success);
        assert!(response.meta.is_some());
        assert_eq!(response.meta.unwrap().request_id, "req-123");
    }

    #[test]
    fn test_response_meta() {
        let meta = ResponseMeta::new("req-456")
            .with_processing_time(42);
        assert_eq!(meta.request_id, "req-456");
        assert_eq!(meta.processing_time_ms, Some(42));
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo::new("llama-3b");
        assert_eq!(info.id, "llama-3b");
        assert!(info.memory_bytes.is_none());
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse::new("1.0.0", 100);
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"version\":\"1.0.0\""));
    }
}
