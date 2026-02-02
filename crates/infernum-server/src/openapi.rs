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

use crate::error_response::{ApiError, ErrorDetail};
use crate::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatLogProbs, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, FunctionCall, FunctionDefinition, ModelObject,
    ModelsResponse, TokenLogProb, Tool, ToolCall, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, TopLogProb, Usage,
};
use crate::responses::{HealthResponse, ModelInfo, ReadyResponse};
use crate::tokenize::{TokenizeRequest, TokenizeResponse};

/// OpenAPI documentation for the Infernum API.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Infernum API",
        version = "1.0.0",
        description = "OpenAI-compatible local LLM inference server.\n\n\
            Infernum provides high-performance inference endpoints that are fully compatible \
            with the OpenAI API, allowing you to use existing OpenAI SDKs and tools.",
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
        chat_completions,
        completions,
        embeddings,
    ),
    components(
        schemas(
            HealthResponse,
            ReadyResponse,
            ModelInfo,
            ModelsResponse,
            ModelObject,
            ChatCompletionRequest,
            ChatCompletionResponse,
            ChatMessage,
            ChatChoice,
            ChatLogProbs,
            TokenLogProb,
            TopLogProb,
            Tool,
            FunctionDefinition,
            ToolCall,
            FunctionCall,
            ToolChoice,
            ToolChoiceFunction,
            ToolChoiceFunctionName,
            CompletionRequest,
            CompletionResponse,
            CompletionChoice,
            EmbeddingRequest,
            EmbeddingInput,
            EmbeddingResponse,
            EmbeddingData,
            Usage,
            TokenizeRequest,
            TokenizeResponse,
            ApiError,
            ErrorDetail,
        )
    ),
    modifiers(&SecurityAddon),
    tags(
        (name = "Health", description = "Health and readiness endpoints"),
        (name = "Models", description = "Model listing and management"),
        (name = "Chat", description = "Chat completion endpoints"),
        (name = "Completions", description = "Text completion endpoints"),
        (name = "Embeddings", description = "Embedding generation endpoints"),
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
        (status = 200, description = "List of available models", body = ModelsResponse),
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
    request_body = TokenizeRequest,
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

/// Create a chat completion.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "Chat",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "Successful completion", body = ChatCompletionResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limited", body = ApiError),
        (status = 503, description = "Model not loaded", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn chat_completions() {}

/// Create a text completion.
#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "Completions",
    request_body = CompletionRequest,
    responses(
        (status = 200, description = "Successful completion", body = CompletionResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limited", body = ApiError),
        (status = 503, description = "Model not loaded", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn completions() {}

/// Create embeddings.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "Embeddings",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Successful embedding", body = EmbeddingResponse),
        (status = 400, description = "Invalid request", body = ApiError),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 429, description = "Rate limited", body = ApiError),
        (status = 503, description = "Model not loaded", body = ApiError),
    ),
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn embeddings() {}

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn test_openapi_spec_generates() {
        let spec = ApiDoc::openapi();
        assert_eq!(spec.info.title, "Infernum API");
        assert_eq!(spec.info.version, "1.0.0");
    }

    #[test]
    fn test_openapi_spec_has_paths() {
        let spec = ApiDoc::openapi();
        assert!(spec.paths.paths.contains_key("/health"));
        assert!(spec.paths.paths.contains_key("/ready"));
        assert!(spec.paths.paths.contains_key("/v1/models"));
        assert!(spec.paths.paths.contains_key("/v1/tokenize"));
        assert!(spec.paths.paths.contains_key("/v1/chat/completions"));
        assert!(spec.paths.paths.contains_key("/v1/completions"));
        assert!(spec.paths.paths.contains_key("/v1/embeddings"));
    }

    #[test]
    fn test_openapi_spec_has_components() {
        let spec = ApiDoc::openapi();
        let components = spec.components.expect("components should exist");
        let schemas = components.schemas;
        assert!(schemas.contains_key("HealthResponse"));
        assert!(schemas.contains_key("ChatCompletionRequest"));
        assert!(schemas.contains_key("ApiError"));
    }

    #[test]
    fn test_openapi_spec_to_json() {
        let spec = ApiDoc::openapi();
        let json = spec.to_json().expect("should serialize to JSON");
        assert!(json.contains("Infernum API"));
        assert!(json.contains("/v1/chat/completions"));
    }
}
