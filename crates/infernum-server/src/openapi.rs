//! OpenAPI specification and Swagger UI for the Infernum API.
//!
//! This module provides auto-generated OpenAPI 3.1 documentation using utoipa.
//!
//! # Endpoints
//!
//! - `/api-docs/openapi.json` - Raw OpenAPI specification
//! - `/swagger-ui/` - Interactive Swagger UI
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::openapi::ApiDoc;
//! use utoipa::OpenApi;
//!
//! let spec = ApiDoc::openapi();
//! println!("{}", spec.to_json().unwrap());
//! ```

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::api::{
    ErrorBody, ErrorCode as ApiErrorCode, ErrorResponse, ModelListEntry,
    ModelsResponse as ApiModelsResponse,
};
use crate::error_response::{ApiError, ErrorDetail};
use crate::responses::{HealthResponse, ModelInfo, ReadyResponse};
use crate::tokenize::TokenizeResponse;

/// OpenAPI documentation for the Infernum API.
///
/// Documents the Infernum-native API. See `INFERNUM-API-SPEC.md` for the
/// complete wire-format specification.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Infernum API",
        version = "2.0.0",
        description = "Infernum-native local LLM inference server.\n\n\
            Infernum provides high-performance inference with native tool calling, \
            structured outputs, and agentic capabilities.\n\n\
            See INFERNUM-API-SPEC.md for the complete wire format specification.",
        license(name = "MIT", url = "https://opensource.org/licenses/MIT"),
        contact(name = "Infernum", url = "https://github.com/daemoniorum/infernum")
    ),
    servers(
        (url = "http://localhost:8080", description = "Local development server"),
        (url = "http://localhost:8081", description = "Docker container"),
    ),
    paths(
        health,
        ready,
        list_models,
        tokenize,
        generate,
        embed,
    ),
    components(
        schemas(
            HealthResponse,
            ReadyResponse,
            ModelInfo,
            ApiModelsResponse,
            ModelListEntry,
            ErrorResponse,
            ErrorBody,
            ApiErrorCode,
            TokenizeResponse,
            ApiError,
            ErrorDetail,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "Health", description = "Health and readiness endpoints"),
        (name = "Models", description = "Model listing and management"),
        (name = "Generate", description = "Text and chat generation endpoint"),
        (name = "Embed", description = "Embedding generation endpoint"),
        (name = "Tokenize", description = "Token counting endpoints"),
    )
)]
pub struct ApiDoc;

/// Security scheme modifier for API key authentication.
struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("Authorization"))),
            );
        }
    }
}

/// Health check endpoint.
#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse),
    )
)]
pub async fn health() {}

/// Readiness check endpoint.
#[utoipa::path(
    get,
    path = "/ready",
    tag = "Health",
    responses(
        (status = 200, description = "Readiness status", body = ReadyResponse),
    )
)]
pub async fn ready() {}

/// List available models.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "Models",
    responses(
        (status = 200, description = "List of available models", body = ApiModelsResponse),
        (status = 401, description = "Unauthorized", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn list_models() {}

/// Count tokens in a prompt or messages.
///
/// This endpoint allows you to count tokens without running inference,
/// useful for pre-flight validation and cost estimation.
#[utoipa::path(
    post,
    path = "/v1/tokenize",
    tag = "Tokenize",
    request_body = String,
    responses(
        (status = 200, description = "Token count result", body = TokenizeResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn tokenize() {}

/// Generate text or chat completion.
///
/// Unified endpoint that accepts both text prompts (`prompt: "string"`) and
/// chat messages (`prompt: [{role, content}]`). Supports streaming via SSE,
/// native tool calling, and agentic mode.
///
/// See INFERNUM-API-SPEC.md §3 for the full request/response schema.
#[utoipa::path(
    post,
    path = "/v1/generate",
    tag = "Generate",
    request_body(content = String, description = "GenerateRequest JSON. See INFERNUM-API-SPEC.md §3.1.",
        example = json!({
            "model": "llama-3.2-3b",
            "prompt": [{"role": "user", "content": "Hello!"}],
            "sampling": {"temperature": 0.7, "max_tokens": 4096},
            "stream": false
        })
    ),
    responses(
        (status = 200, description = "Successful generation", content_type = "application/json",
            body = String,
            example = json!({
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "model": "llama-3.2-3b",
                "choices": [{"index": 0, "text": "Hello! How can I help?", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
            })
        ),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limited", body = ApiError),
        (status = 503, description = "Model not loaded", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn generate() {}

/// Generate embeddings.
///
/// Accepts single or batch text inputs and returns vector embeddings.
///
/// See INFERNUM-API-SPEC.md §5 for the full request/response schema.
#[utoipa::path(
    post,
    path = "/v1/embed",
    tag = "Embed",
    request_body(content = String, description = "EmbedRequest JSON. See INFERNUM-API-SPEC.md §5.1.",
        example = json!({
            "model": "nomic-embed-text",
            "input": "The quick brown fox"
        })
    ),
    responses(
        (status = 200, description = "Successful embedding", content_type = "application/json",
            body = String,
            example = json!({
                "model": "nomic-embed-text",
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })
        ),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limited", body = ApiError),
        (status = 503, description = "Model not loaded", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn embed() {}

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn test_openapi_spec_generates() {
        let spec = ApiDoc::openapi();
        assert_eq!(spec.info.title, "Infernum API");
        assert_eq!(spec.info.version, "2.0.0");
    }

    #[test]
    fn test_openapi_spec_has_paths() {
        let spec = ApiDoc::openapi();
        assert!(spec.paths.paths.contains_key("/health"));
        assert!(spec.paths.paths.contains_key("/ready"));
        assert!(spec.paths.paths.contains_key("/v1/models"));
        assert!(spec.paths.paths.contains_key("/v1/tokenize"));
        assert!(spec.paths.paths.contains_key("/v1/generate"));
        assert!(spec.paths.paths.contains_key("/v1/embed"));
    }

    #[test]
    fn test_openapi_spec_no_legacy_paths() {
        let spec = ApiDoc::openapi();
        assert!(!spec.paths.paths.contains_key("/v1/chat/completions"));
        assert!(!spec.paths.paths.contains_key("/v1/completions"));
        assert!(!spec.paths.paths.contains_key("/v1/embeddings"));
    }

    #[test]
    fn test_openapi_spec_has_components() {
        let spec = ApiDoc::openapi();
        let components = spec.components.expect("components should exist");
        let schemas = components.schemas;
        assert!(schemas.contains_key("HealthResponse"));
        assert!(schemas.contains_key("ModelsResponse"));
        assert!(schemas.contains_key("ApiError"));
    }

    #[test]
    fn test_openapi_spec_to_json() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("should serialize to JSON");
        assert!(json.contains("Infernum API"));
        assert!(json.contains("/v1/generate"));
        assert!(json.contains("/v1/embed"));
    }
}
