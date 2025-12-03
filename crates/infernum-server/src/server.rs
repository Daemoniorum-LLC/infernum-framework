//! HTTP server implementation with OpenAI-compatible API endpoints.
//!
//! Provides a production-ready server that interfaces with the Abaddon inference engine
//! for text generation, chat completions, and embeddings.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use abaddon::{Engine, EngineConfig, InferenceEngine};
use infernum_core::{GenerateRequest, Result, SamplingParams};

use crate::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, ModelObject, ModelsResponse, Usage,
};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address.
    pub addr: SocketAddr,
    /// Enable CORS.
    pub cors: bool,
    /// Model to load (optional - server can start without a model).
    pub model: Option<String>,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:8080".parse().unwrap(),
            cors: true,
            model: None,
            max_concurrent_requests: 64,
        }
    }
}

impl ServerConfig {
    /// Creates a new server config builder.
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::default()
    }
}

/// Builder for ServerConfig.
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    addr: Option<SocketAddr>,
    cors: Option<bool>,
    model: Option<String>,
    max_concurrent_requests: Option<usize>,
}

impl ServerConfigBuilder {
    /// Sets the listen address.
    pub fn addr(mut self, addr: SocketAddr) -> Self {
        self.addr = Some(addr);
        self
    }

    /// Sets whether CORS is enabled.
    pub fn cors(mut self, enabled: bool) -> Self {
        self.cors = Some(enabled);
        self
    }

    /// Sets the model to load.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the maximum concurrent requests.
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = Some(max);
        self
    }

    /// Builds the server config.
    pub fn build(self) -> ServerConfig {
        ServerConfig {
            addr: self.addr.unwrap_or_else(|| "0.0.0.0:8080".parse().unwrap()),
            cors: self.cors.unwrap_or(true),
            model: self.model,
            max_concurrent_requests: self.max_concurrent_requests.unwrap_or(64),
        }
    }
}

/// Shared application state.
pub struct AppState {
    /// The inference engine (None if no model is loaded).
    pub engine: RwLock<Option<Arc<Engine>>>,
    /// Server configuration.
    pub config: ServerConfig,
    /// Server start time.
    pub start_time: Instant,
}

impl AppState {
    /// Creates new app state with the given config.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            engine: RwLock::new(None),
            config,
            start_time: Instant::now(),
        }
    }

    /// Creates new app state with a pre-loaded engine.
    pub fn with_engine(config: ServerConfig, engine: Engine) -> Self {
        Self {
            engine: RwLock::new(Some(Arc::new(engine))),
            config,
            start_time: Instant::now(),
        }
    }
}

/// The HTTP server.
pub struct Server {
    config: ServerConfig,
    state: Arc<AppState>,
}

impl Server {
    /// Creates a new server with the given configuration.
    pub fn new(config: ServerConfig) -> Self {
        let state = Arc::new(AppState::new(config.clone()));
        Self { config, state }
    }

    /// Creates a new server with a pre-loaded engine.
    pub fn with_engine(config: ServerConfig, engine: Engine) -> Self {
        let state = Arc::new(AppState::with_engine(config.clone(), engine));
        Self { config, state }
    }

    /// Creates the router.
    fn router(&self) -> Router {
        let mut router = Router::new()
            // Health endpoints
            .route("/health", get(health))
            .route("/ready", get(ready))
            // OpenAI-compatible API endpoints
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            // NOTE: /v1/embeddings disabled until embedding models are supported
            // .route("/v1/embeddings", post(embeddings))
            // Internal management endpoints
            .route("/api/models/load", post(load_model))
            .route("/api/models/unload", post(unload_model))
            .route("/api/status", get(server_status))
            .with_state(self.state.clone());

        // Add middleware
        router = router.layer(TraceLayer::new_for_http());

        if self.config.cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }

    /// Loads a model into the server.
    pub async fn load_model(&self, model_source: &str) -> Result<()> {
        tracing::info!(model = %model_source, "Loading model");

        let engine_config = EngineConfig::builder()
            .model(model_source)
            .build()
            .map_err(|e| infernum_core::Error::Internal { message: e })?;

        let engine = Engine::new(engine_config).await?;
        let mut engine_guard = self.state.engine.write().await;
        *engine_guard = Some(Arc::new(engine));

        tracing::info!(model = %model_source, "Model loaded successfully");
        Ok(())
    }

    /// Runs the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot start.
    pub async fn run(self) -> Result<()> {
        // Load model if specified
        if let Some(model) = &self.config.model {
            self.load_model(model).await?;
            tracing::info!(model = %model, "Model loaded and ready for inference");
        } else {
            tracing::warn!("=======================================================");
            tracing::warn!("  SERVER STARTED WITHOUT A MODEL");
            tracing::warn!("  All inference requests will fail until a model is loaded.");
            tracing::warn!("  ");
            tracing::warn!("  To load a model, either:");
            tracing::warn!("    1. Restart with: infernum serve --model <model>");
            tracing::warn!("    2. POST to /api/models/load with {{\"model\": \"<model>\"}}");
            tracing::warn!("=======================================================");
        }

        let router = self.router();

        tracing::info!(addr = %self.config.addr, "Starting Infernum server");
        eprintln!(
            "\n\x1b[32m✓\x1b[0m Server listening on http://{}",
            self.config.addr
        );
        eprintln!("  Press Ctrl+C to stop\n");

        let listener = tokio::net::TcpListener::bind(self.config.addr)
            .await
            .map_err(infernum_core::Error::Io)?;

        // Set up graceful shutdown
        let shutdown_signal = async {
            let ctrl_c = async {
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to install Ctrl+C handler");
            };

            #[cfg(unix)]
            let terminate = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Failed to install signal handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let terminate = std::future::pending::<()>();

            tokio::select! {
                () = ctrl_c => {
                    eprintln!("\n\x1b[33m⚡\x1b[0m Received Ctrl+C, shutting down gracefully...");
                },
                () = terminate => {
                    eprintln!("\n\x1b[33m⚡\x1b[0m Received SIGTERM, shutting down gracefully...");
                },
            }
        };

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal)
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?;

        tracing::info!("Server shutdown complete");
        eprintln!("\x1b[32m✓\x1b[0m Server stopped");

        Ok(())
    }
}

// === Error Response ===

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: Option<String>,
}

impl ErrorResponse {
    fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
        }
    }

    #[allow(dead_code)] // Reserved for future use with specific error codes
    fn with_code(mut self, code: impl Into<String>) -> Self {
        self.error.code = Some(code.into());
        self
    }
}

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    let body = Json(ErrorResponse::new(message, error_type));
    (status, body).into_response()
}

// === Health Endpoints ===

async fn health() -> &'static str {
    "OK"
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let engine = state.engine.read().await;
    if engine.is_some() {
        (StatusCode::OK, "Ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "No model loaded").into_response()
    }
}

#[derive(Debug, Serialize)]
struct ServerStatus {
    status: String,
    uptime_seconds: u64,
    model_loaded: bool,
    model_id: Option<String>,
}

async fn server_status(State(state): State<Arc<AppState>>) -> Json<ServerStatus> {
    let engine = state.engine.read().await;
    let model_id = engine.as_ref().map(|e| e.model_info().id.to_string());

    Json(ServerStatus {
        status: "running".to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        model_loaded: engine.is_some(),
        model_id,
    })
}

// === Model Management ===

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model: String,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    tracing::info!(model = %req.model, "Loading model via API");

    let engine_config = match EngineConfig::builder().model(&req.model).build() {
        Ok(config) => config,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid model configuration: {}", e),
                "invalid_request_error",
            );
        },
    };

    let engine = match Engine::new(engine_config).await {
        Ok(engine) => engine,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to load model: {}", e),
                "model_load_error",
            );
        },
    };

    let mut engine_guard = state.engine.write().await;
    *engine_guard = Some(Arc::new(engine));

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "loaded", "model": req.model})),
    )
        .into_response()
}

async fn unload_model(State(state): State<Arc<AppState>>) -> Response {
    let mut engine_guard = state.engine.write().await;
    *engine_guard = None;
    tracing::info!("Model unloaded");
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "unloaded"})),
    )
        .into_response()
}

// === OpenAI-Compatible Endpoints ===

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let engine = state.engine.read().await;

    let models = match engine.as_ref() {
        Some(engine) => {
            let info = engine.model_info();
            vec![ModelObject {
                id: info.id.to_string(),
                object: "model".to_string(),
                created: chrono::Utc::now().timestamp(),
                owned_by: "infernum".to_string(),
            }]
        },
        None => vec![],
    };

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    tracing::debug!(request_id = %request_id, model = %req.model, "Chat completion request");

    // Get engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "No model loaded",
                "model_not_loaded",
            );
        },
    };
    drop(engine_guard); // Release lock early

    // Check for streaming
    let stream = req.stream.unwrap_or(false);

    // Build messages into prompt
    let messages: Vec<infernum_core::Message> = req
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => infernum_core::Role::System,
                "user" => infernum_core::Role::User,
                "assistant" => infernum_core::Role::Assistant,
                _ => infernum_core::Role::User,
            };
            infernum_core::Message {
                role,
                content: m.content.clone(),
                name: None,
                tool_call_id: None,
            }
        })
        .collect();

    // Build sampling params
    let mut sampling = SamplingParams::default();
    if let Some(temp) = req.temperature {
        sampling = sampling.with_temperature(temp);
    }
    if let Some(top_p) = req.top_p {
        sampling = sampling.with_top_p(top_p);
    }
    if let Some(max_tokens) = req.max_tokens {
        sampling = sampling.with_max_tokens(max_tokens);
    }
    if let Some(stop) = &req.stop {
        for s in stop {
            sampling = sampling.with_stop(s.clone());
        }
    }

    // Create inference request
    let gen_request = GenerateRequest::new(infernum_core::request::PromptInput::Messages(messages))
        .with_sampling(sampling);

    if stream {
        // Streaming response
        match engine.generate_stream(gen_request).await {
            Ok(token_stream) => {
                let model_name = engine.model_info().id.to_string();
                let sse_stream = token_stream.map(move |chunk_result| {
                    match chunk_result {
                        Ok(chunk) => {
                            let data = serde_json::json!({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": chrono::Utc::now().timestamp(),
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": chunk.choices.first().map(|c| c.delta.content.as_deref().unwrap_or("")).unwrap_or("")
                                    },
                                    "finish_reason": chunk.choices.first().and_then(|c| c.finish_reason.as_ref().map(|r| format!("{:?}", r).to_lowercase()))
                                }]
                            });
                            Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(serde_json::to_string(&data).unwrap()))
                        }
                        Err(e) => {
                            let data = serde_json::json!({
                                "error": {
                                    "message": e.to_string(),
                                    "type": "server_error"
                                }
                            });
                            Ok(axum::response::sse::Event::default().data(serde_json::to_string(&data).unwrap()))
                        }
                    }
                });

                Sse::new(sse_stream)
                    .keep_alive(axum::response::sse::KeepAlive::default())
                    .into_response()
            },
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "generation_error",
            ),
        }
    } else {
        // Non-streaming response
        match engine.generate(gen_request).await {
            Ok(response) => {
                let choice = response.choices.first();
                let content = choice.map(|c| c.text.clone()).unwrap_or_default();
                let finish_reason = choice
                    .and_then(|c| c.finish_reason.as_ref())
                    .map(|r| format!("{:?}", r).to_lowercase())
                    .unwrap_or_else(|| "stop".to_string());

                let chat_response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: engine.model_info().id.to_string(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content,
                            name: None,
                        },
                        finish_reason,
                    }],
                    usage: Usage {
                        prompt_tokens: response.usage.prompt_tokens,
                        completion_tokens: response.usage.completion_tokens,
                        total_tokens: response.usage.total_tokens,
                    },
                };

                tracing::debug!(
                    request_id = %chat_response.id,
                    prompt_tokens = response.usage.prompt_tokens,
                    completion_tokens = response.usage.completion_tokens,
                    latency_ms = start.elapsed().as_millis() as u64,
                    "Chat completion finished"
                );

                Json(chat_response).into_response()
            },
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "generation_error",
            ),
        }
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    tracing::debug!(request_id = %request_id, model = %req.model, "Completion request");

    // Get engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "No model loaded",
                "model_not_loaded",
            );
        },
    };
    drop(engine_guard);

    // Build sampling params
    let mut sampling = SamplingParams::default();
    if let Some(temp) = req.temperature {
        sampling = sampling.with_temperature(temp);
    }
    if let Some(top_p) = req.top_p {
        sampling = sampling.with_top_p(top_p);
    }
    if let Some(max_tokens) = req.max_tokens {
        sampling = sampling.with_max_tokens(max_tokens);
    }
    if let Some(stop) = &req.stop {
        for s in stop {
            sampling = sampling.with_stop(s.clone());
        }
    }

    // Create inference request
    let gen_request = GenerateRequest::new(infernum_core::request::PromptInput::Text(req.prompt))
        .with_sampling(sampling);

    match engine.generate(gen_request).await {
        Ok(response) => {
            let choice = response.choices.first();
            let text = choice.map(|c| c.text.clone()).unwrap_or_default();
            let finish_reason = choice
                .and_then(|c| c.finish_reason.as_ref())
                .map(|r| format!("{:?}", r).to_lowercase())
                .unwrap_or_else(|| "stop".to_string());

            let completion_response = CompletionResponse {
                id: request_id.clone(),
                object: "text_completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: engine.model_info().id.to_string(),
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason,
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens: response.usage.prompt_tokens,
                    completion_tokens: response.usage.completion_tokens,
                    total_tokens: response.usage.total_tokens,
                },
            };

            tracing::debug!(
                request_id = %request_id,
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                latency_ms = start.elapsed().as_millis() as u64,
                "Completion finished"
            );

            Json(completion_response).into_response()
        },
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "generation_error",
        ),
    }
}

// TODO: Re-enable when embedding models are supported
#[allow(dead_code)]
async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let request_id = format!("emb-{}", uuid::Uuid::new_v4());

    tracing::debug!(request_id = %request_id, model = %req.model, "Embedding request");

    // Get engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "No model loaded",
                "model_not_loaded",
            );
        },
    };
    drop(engine_guard);

    // Get input texts
    let texts: Vec<String> = match &req.input {
        EmbeddingInput::Single(s) => vec![s.clone()],
        EmbeddingInput::Multiple(v) => v.clone(),
    };

    // Generate embeddings for each input
    let mut embeddings = Vec::new();
    let mut total_tokens = 0u32;

    for (idx, text) in texts.iter().enumerate() {
        let embed_request = infernum_core::EmbedRequest::new(text.clone());

        match engine.embed(embed_request).await {
            Ok(response) => {
                // Extract embedding vector from the response
                let embedding_vec = response
                    .data
                    .first()
                    .and_then(|e| e.embedding.as_floats().ok())
                    .unwrap_or_default();

                embeddings.push(EmbeddingData {
                    object: "embedding".to_string(),
                    index: idx as u32,
                    embedding: embedding_vec,
                });
                total_tokens += response.usage.total_tokens;
            },
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "embedding_error",
                );
            },
        }
    }

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: embeddings,
        model: engine.model_info().id.to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Json(response).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_builder() {
        let config = ServerConfig::builder()
            .addr("127.0.0.1:3000".parse().unwrap())
            .cors(false)
            .model("test-model")
            .max_concurrent_requests(32)
            .build();

        assert_eq!(config.addr, "127.0.0.1:3000".parse().unwrap());
        assert!(!config.cors);
        assert_eq!(config.model, Some("test-model".to_string()));
        assert_eq!(config.max_concurrent_requests, 32);
    }

    #[test]
    fn test_error_response() {
        let err = ErrorResponse::new("Test error", "test_error").with_code("TEST_CODE");

        assert_eq!(err.error.message, "Test error");
        assert_eq!(err.error.error_type, "test_error");
        assert_eq!(err.error.code, Some("TEST_CODE".to_string()));
    }
}
