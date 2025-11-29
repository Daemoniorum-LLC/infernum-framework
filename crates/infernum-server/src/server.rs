//! HTTP server implementation.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};
use infernum_core::Result;
use tower_http::cors::CorsLayer;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address.
    pub addr: SocketAddr,
    /// Enable CORS.
    pub cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:8080".parse().unwrap(),
            cors: true,
        }
    }
}

/// The HTTP server.
pub struct Server {
    config: ServerConfig,
}

impl Server {
    /// Creates a new server with the given configuration.
    #[must_use]
    pub fn new(config: ServerConfig) -> Self {
        Self { config }
    }

    /// Creates the router.
    fn router(&self) -> Router {
        let mut router = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", axum::routing::post(chat_completions))
            .route("/v1/completions", axum::routing::post(completions))
            .route("/v1/embeddings", axum::routing::post(embeddings));

        if self.config.cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }

    /// Runs the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot start.
    pub async fn run(self) -> Result<()> {
        let router = self.router();

        tracing::info!(addr = %self.config.addr, "Starting Infernum server");

        let listener = tokio::net::TcpListener::bind(self.config.addr)
            .await
            .map_err(|e| infernum_core::Error::Io(e))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?;

        Ok(())
    }
}

// === Handlers ===

async fn health() -> &'static str {
    "OK"
}

async fn list_models() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "object": "list",
        "data": []
    }))
}

async fn chat_completions(
    axum::Json(body): axum::Json<serde_json::Value>,
) -> axum::Json<serde_json::Value> {
    // TODO: Implement actual chat completions
    axum::Json(serde_json::json!({
        "id": "chatcmpl-placeholder",
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": "infernum",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! Infernum server is running."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }))
}

async fn completions(
    axum::Json(body): axum::Json<serde_json::Value>,
) -> axum::Json<serde_json::Value> {
    // TODO: Implement actual completions
    axum::Json(serde_json::json!({
        "id": "cmpl-placeholder",
        "object": "text_completion",
        "created": chrono::Utc::now().timestamp(),
        "model": "infernum",
        "choices": [{
            "text": "Hello from Infernum!",
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10
        }
    }))
}

async fn embeddings(
    axum::Json(body): axum::Json<serde_json::Value>,
) -> axum::Json<serde_json::Value> {
    // TODO: Implement actual embeddings
    axum::Json(serde_json::json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": vec![0.0f32; 768]
        }],
        "model": "infernum",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }))
}
