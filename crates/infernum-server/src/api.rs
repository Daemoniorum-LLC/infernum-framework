//! Infernum-native API types.
//!
//! This module defines the HTTP wire format for the Infernum API.
//! Core request/response types live in `infernum-core` and are re-exported here.
//! Server-specific types (models listing, error format) are defined locally.
//!
//! See `docs/specs/INFERNUM-API-SPEC.md` for the full specification.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ---------------------------------------------------------------------------
// Re-exports from infernum-core (the canonical wire types)
// ---------------------------------------------------------------------------

/// Generation request. Accepts text, chat messages, or pre-tokenized input.
/// See INFERNUM-API-SPEC.md §3.1.
pub use infernum_core::{GenerateRequest, SamplingParams};

/// Generation response with choices, usage, and timing.
/// See INFERNUM-API-SPEC.md §3.7.
pub use infernum_core::response::{Choice, GenerateResponse};

/// Embedding request and response.
/// See INFERNUM-API-SPEC.md §5.
pub use infernum_core::request::{EmbedInput, EmbedRequest, EncodingFormat};
pub use infernum_core::response::{EmbedResponse, Embedding, EmbeddingData};

/// Streaming types.
/// See INFERNUM-API-SPEC.md §4.
pub use infernum_core::streaming::{StreamChoice, StreamChunk, StreamDelta, TokenStream};

/// Message and role types.
/// See INFERNUM-API-SPEC.md §3.2.
pub use infernum_core::types::{Message, Role};

/// Tool types.
/// See INFERNUM-API-SPEC.md §3.3–3.5.
pub use infernum_core::types::{ToolCall, ToolControl, ToolControlMode, ToolDefinition};

/// Common types.
pub use infernum_core::types::{FinishReason, ModelId, RequestId, Usage};

/// Prompt input variants.
pub use infernum_core::request::PromptInput;

// ---------------------------------------------------------------------------
// Server-specific types: Models endpoint (§6)
// ---------------------------------------------------------------------------

/// Response for `GET /v1/models`.
///
/// Lists all models currently loaded in the server.
///
/// ```json
/// {
///   "models": [
///     {
///       "id": "llama-3.2-3b",
///       "architecture": "llama",
///       "context_length": 8192,
///       "quantization": "gguf_q4_k_m",
///       "owned_by": "infernum"
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// List of available models.
    pub models: Vec<ModelListEntry>,
}

/// Information about a single model in the models list.
///
/// Flat, descriptive structure—no `object: "model"` ceremony.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListEntry {
    /// Model identifier (e.g. `"llama-3.2-3b"`).
    pub id: String,

    /// Model architecture (e.g. `"llama"`, `"qwen2"`, `"phi"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,

    /// Maximum context length in tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    /// Quantization format (e.g. `"gguf_q4_k_m"`, `"f16"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,

    /// Owner identifier.
    #[serde(default = "default_owned_by")]
    pub owned_by: String,
}

fn default_owned_by() -> String {
    "infernum".to_string()
}

impl ModelListEntry {
    /// Creates a new model entry with just an ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            architecture: None,
            context_length: None,
            quantization: None,
            owned_by: "infernum".to_string(),
        }
    }

    /// Sets the architecture.
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Sets the context length.
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Sets the quantization format.
    pub fn with_quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }

    /// Sets the owner.
    pub fn with_owned_by(mut self, owner: impl Into<String>) -> Self {
        self.owned_by = owner.into();
        self
    }
}

impl From<&infernum_core::ModelMetadata> for ModelListEntry {
    fn from(meta: &infernum_core::ModelMetadata) -> Self {
        // Extract architecture family name from the tagged enum
        let architecture = serde_json::to_value(&meta.architecture)
            .ok()
            .and_then(|v| v.get("type").and_then(|t| t.as_str().map(String::from)));

        // Format quantization as a string
        let quantization = meta.quantization.map(|q| {
            serde_json::to_value(q)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_else(|| format!("{q:?}"))
        });

        Self {
            id: meta.id.to_string(),
            architecture,
            context_length: Some(meta.context_length),
            quantization,
            owned_by: "infernum".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Server-specific types: Error format (§8)
// ---------------------------------------------------------------------------

/// Standard error response format.
///
/// ```json
/// {
///   "error": {
///     "code": "invalid_request",
///     "message": "Temperature must be between 0.0 and 2.0",
///     "details": { "field": "sampling.temperature" }
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// The error object.
    pub error: ErrorBody,
}

/// Body of an error response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorBody {
    /// Machine-readable error code.
    pub code: ErrorCode,

    /// Human-readable error message.
    pub message: String,

    /// Additional error details (field-specific info, constraints, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Machine-readable error codes.
///
/// See INFERNUM-API-SPEC.md §8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Malformed or invalid request (400).
    InvalidRequest,
    /// Requested model not loaded (404).
    ModelNotFound,
    /// Input exceeds model context length (400).
    ContextOverflow,
    /// Too many requests (429).
    RateLimited,
    /// Model is processing other requests (503).
    ModelBusy,
    /// Unexpected server error (500).
    InternalError,
}

impl ErrorCode {
    /// Returns the HTTP status code for this error.
    pub fn status_code(self) -> u16 {
        match self {
            Self::InvalidRequest | Self::ContextOverflow => 400,
            Self::ModelNotFound => 404,
            Self::RateLimited => 429,
            Self::ModelBusy => 503,
            Self::InternalError => 500,
        }
    }
}

impl ErrorResponse {
    /// Creates an error response.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                code,
                message: message.into(),
                details: None,
            },
        }
    }

    /// Creates an error response with details.
    pub fn with_details(
        code: ErrorCode,
        message: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        Self {
            error: ErrorBody {
                code,
                message: message.into(),
                details: Some(details),
            },
        }
    }

    /// Convenience: invalid request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidRequest, message)
    }

    /// Convenience: model not found error.
    pub fn model_not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ModelNotFound, message)
    }

    /// Convenience: context overflow error.
    pub fn context_overflow(tokens: u64, limit: u64) -> Self {
        Self::with_details(
            ErrorCode::ContextOverflow,
            format!("Input ({tokens} tokens) exceeds context length ({limit})"),
            serde_json::json!({
                "tokens": tokens,
                "limit": limit,
            }),
        )
    }

    /// Convenience: rate limited error.
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::RateLimited, message)
    }

    /// Convenience: model busy error.
    pub fn model_busy() -> Self {
        Self::new(
            ErrorCode::ModelBusy,
            "Model is currently processing other requests",
        )
    }

    /// Convenience: internal error.
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InternalError, message)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelsResponse tests --

    #[test]
    fn test_models_response_serialization() {
        let response = ModelsResponse {
            models: vec![ModelListEntry::new("llama-3.2-3b")
                .with_architecture("llama")
                .with_context_length(8192)
                .with_quantization("gguf_q4_k_m")],
        };

        let json = serde_json::to_value(&response).expect("serialize");
        assert_eq!(json["models"][0]["id"], "llama-3.2-3b");
        assert_eq!(json["models"][0]["architecture"], "llama");
        assert_eq!(json["models"][0]["context_length"], 8192);
        assert_eq!(json["models"][0]["quantization"], "gguf_q4_k_m");
        assert_eq!(json["models"][0]["owned_by"], "infernum");
    }

    #[test]
    fn test_models_response_spec_example() {
        // Exact JSON from INFERNUM-API-SPEC.md §6.1
        let json = r#"{
            "models": [
                {
                    "id": "llama-3.2-3b",
                    "architecture": "llama",
                    "context_length": 8192,
                    "quantization": "gguf_q4_k_m",
                    "owned_by": "infernum"
                }
            ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.models.len(), 1);
        assert_eq!(parsed.models[0].id, "llama-3.2-3b");
        assert_eq!(parsed.models[0].architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn test_model_list_entry_minimal() {
        let entry = ModelListEntry::new("test-model");
        let json = serde_json::to_value(&entry).expect("serialize");

        // Optional fields should be omitted
        assert_eq!(json["id"], "test-model");
        assert_eq!(json["owned_by"], "infernum");
        assert!(json.get("architecture").is_none());
        assert!(json.get("context_length").is_none());
        assert!(json.get("quantization").is_none());
    }

    #[test]
    fn test_model_list_entry_roundtrip() {
        let entry = ModelListEntry::new("qwen-2.5-7b")
            .with_architecture("qwen2")
            .with_context_length(32768)
            .with_quantization("f16")
            .with_owned_by("custom");

        let json = serde_json::to_string(&entry).expect("serialize");
        let parsed: ModelListEntry = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.id, "qwen-2.5-7b");
        assert_eq!(parsed.architecture.as_deref(), Some("qwen2"));
        assert_eq!(parsed.context_length, Some(32768));
        assert_eq!(parsed.quantization.as_deref(), Some("f16"));
        assert_eq!(parsed.owned_by, "custom");
    }

    // -- ErrorResponse tests --

    #[test]
    fn test_error_response_serialization() {
        let err = ErrorResponse::invalid_request("Temperature must be between 0.0 and 2.0");
        let json = serde_json::to_value(&err).expect("serialize");

        assert_eq!(json["error"]["code"], "invalid_request");
        assert_eq!(
            json["error"]["message"],
            "Temperature must be between 0.0 and 2.0"
        );
        assert!(json["error"].get("details").is_none());
    }

    #[test]
    fn test_error_response_spec_example() {
        // Exact JSON from INFERNUM-API-SPEC.md §8
        let json = r#"{
            "error": {
                "code": "invalid_request",
                "message": "Temperature must be between 0.0 and 2.0",
                "details": {
                    "field": "sampling.temperature",
                    "value": 3.0,
                    "constraint": "0.0 <= temperature <= 2.0"
                }
            }
        }"#;

        let parsed: ErrorResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(parsed.error.code, ErrorCode::InvalidRequest);
        assert!(parsed.error.details.is_some());
        let details = parsed.error.details.as_ref().expect("details");
        assert_eq!(details["field"], "sampling.temperature");
    }

    #[test]
    fn test_error_context_overflow() {
        let err = ErrorResponse::context_overflow(20000, 16384);
        let json = serde_json::to_value(&err).expect("serialize");

        assert_eq!(json["error"]["code"], "context_overflow");
        let details = &json["error"]["details"];
        assert_eq!(details["tokens"], 20000);
        assert_eq!(details["limit"], 16384);
    }

    #[test]
    fn test_error_code_status_codes() {
        assert_eq!(ErrorCode::InvalidRequest.status_code(), 400);
        assert_eq!(ErrorCode::ModelNotFound.status_code(), 404);
        assert_eq!(ErrorCode::ContextOverflow.status_code(), 400);
        assert_eq!(ErrorCode::RateLimited.status_code(), 429);
        assert_eq!(ErrorCode::ModelBusy.status_code(), 503);
        assert_eq!(ErrorCode::InternalError.status_code(), 500);
    }

    #[test]
    fn test_error_code_roundtrip() {
        for code in [
            ErrorCode::InvalidRequest,
            ErrorCode::ModelNotFound,
            ErrorCode::ContextOverflow,
            ErrorCode::RateLimited,
            ErrorCode::ModelBusy,
            ErrorCode::InternalError,
        ] {
            let json = serde_json::to_value(code).expect("serialize");
            let parsed: ErrorCode = serde_json::from_value(json).expect("deserialize");
            assert_eq!(parsed, code);
        }
    }

    // -- Re-export availability tests --

    #[test]
    fn test_core_types_accessible() {
        // Verify core types are re-exported and usable
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");

        let tool = ToolDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file".to_string()),
            parameters: None,
            strict: None,
        };
        assert_eq!(tool.name, "read_file");
    }

    #[test]
    fn test_generate_request_accessible() {
        let req = GenerateRequest::new("Hello, world!")
            .with_sampling(SamplingParams::default().with_max_tokens(100));

        match &req.prompt {
            PromptInput::Text(t) => assert_eq!(t, "Hello, world!"),
            _ => panic!("Expected text prompt"),
        }
    }

    #[test]
    fn test_embed_request_accessible() {
        let req = EmbedRequest {
            request_id: RequestId::new(),
            model: Some(ModelId::from("nomic-embed")),
            input: EmbedInput::Single("test".to_string()),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
        };

        assert!(req.model.is_some());
    }
}
