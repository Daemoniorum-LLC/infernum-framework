//! HTTP server implementation for the Infernum API.
//!
//! Provides a production-ready server that interfaces with the Abaddon inference engine
//! for text generation, chat completions, and embeddings.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use axum::middleware;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::observability::request_id_middleware;

use abaddon::{Engine, EngineConfig, InferenceEngine};
use dantalion::{MetricsCollector, TelemetryConfig};
use infernum_core::{GenerateRequest, ModelSource, Result, SamplingParams};

use crate::agentic::run_agent;
use crate::error_response::{api_error, ApiError, ErrorCode};
use crate::rag::{RagState, rag_health, list_documents, document_count, index_document, delete_document, search};
use crate::sessions::{
    SessionRegistry, cancel_session, get_session, list_sessions, session_stream, sessions_stream,
};
use crate::model_cache::{
    ModelCacheState, list_cached_models, delete_cached_model, convert_model, download_model,
    find_model_path, is_holotensor_model,
};
use crate::speculative_engine::{SpeculativeEngine, SpeculativeEngineBuilder, SpeculativeEngineConfig, SpeculativeEngineError};
use crate::validation::validate_chat_request;
use crate::batching::{BatchConfig, BatchScheduler};
use crate::request_batcher::{BatcherConfig, BatcherHandle, RequestBatcher};
use crate::api_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, ModelObject, ModelsResponse, ToolChoice, Usage,
};
use crate::tool_use::{
    format_tools_for_prompt, get_forced_tool, process_model_output, should_include_tools,
    validate_tool_exists, ModelFamily,
    // Agent-centric streaming tool detection (Phase 3)
    StreamingToolDetector, SseEvent,
};

/// Default server address.
const DEFAULT_ADDR: ([u8; 4], u16) = ([0, 0, 0, 0], 8080);

/// Timeout configuration for different request types.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Timeout for chat completion requests (default: 120s).
    pub chat_completion: Duration,
    /// Timeout for text completion requests (default: 120s).
    pub completion: Duration,
    /// Timeout for embedding requests (default: 30s).
    pub embedding: Duration,
    /// Timeout for model load operations (default: 600s).
    pub model_load: Duration,
    /// Global request timeout for other endpoints (default: 30s).
    pub default: Duration,
    /// Graceful shutdown timeout (default: 30s).
    pub shutdown: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            chat_completion: Duration::from_secs(120),
            completion: Duration::from_secs(120),
            embedding: Duration::from_secs(30),
            model_load: Duration::from_secs(600),
            default: Duration::from_secs(30),
            shutdown: Duration::from_secs(30),
        }
    }
}

/// Request queue configuration for backpressure control.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum concurrent inference requests.
    pub max_concurrent_requests: usize,
    /// Maximum requests waiting in queue before rejecting.
    pub max_queue_size: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 64,
            max_queue_size: 256,
        }
    }
}

/// Request validation limits.
#[derive(Debug, Clone)]
pub struct ValidationLimits {
    /// Maximum allowed messages in a single chat request.
    pub max_messages: usize,
    /// Maximum allowed content length per message (characters).
    pub max_message_length: usize,
    /// Maximum allowed max_tokens value.
    pub max_max_tokens: u32,
    /// Maximum allowed prompt length (characters).
    pub max_prompt_length: usize,
    /// Maximum embedding inputs per request.
    pub max_embedding_inputs: usize,
    /// Maximum request body size in bytes.
    pub max_body_size: usize,
}

/// Default maximum body size: 10MB
const DEFAULT_MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_messages: 256,
            max_message_length: 100_000,
            max_max_tokens: 32_768,
            max_prompt_length: 500_000,
            max_embedding_inputs: 256,
            max_body_size: DEFAULT_MAX_BODY_SIZE,
        }
    }
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address.
    pub addr: SocketAddr,
    /// Enable CORS.
    pub cors: bool,
    /// Model to load (optional - server can start without a model).
    pub model: Option<String>,
    /// Draft model for speculative decoding (smaller model, fits in VRAM).
    pub draft_model: Option<String>,
    /// Number of draft tokens per speculative round (default: 5).
    pub speculative_tokens: u32,
    /// Request validation limits.
    pub validation_limits: ValidationLimits,
    /// Timeout configuration.
    pub timeouts: TimeoutConfig,
    /// Request queue configuration.
    pub queue: QueueConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(DEFAULT_ADDR),
            cors: true,
            model: None,
            draft_model: None,
            speculative_tokens: 5,
            validation_limits: ValidationLimits::default(),
            timeouts: TimeoutConfig::default(),
            queue: QueueConfig::default(),
        }
    }
}

impl ServerConfig {
    /// Creates a new server config builder.
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::default()
    }

    /// Returns whether speculative decoding is enabled.
    pub fn is_speculative_enabled(&self) -> bool {
        self.draft_model.is_some() && self.model.is_some()
    }
}

/// Builder for ServerConfig.
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    addr: Option<SocketAddr>,
    cors: Option<bool>,
    model: Option<String>,
    draft_model: Option<String>,
    speculative_tokens: Option<u32>,
    validation_limits: Option<ValidationLimits>,
    timeouts: Option<TimeoutConfig>,
    queue: Option<QueueConfig>,
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

    /// Sets the draft model for speculative decoding.
    pub fn draft_model(mut self, model: impl Into<String>) -> Self {
        self.draft_model = Some(model.into());
        self
    }

    /// Sets the number of speculative tokens per round (1-16).
    pub fn speculative_tokens(mut self, tokens: u32) -> Self {
        self.speculative_tokens = Some(tokens.clamp(1, 16));
        self
    }

    /// Sets the maximum concurrent requests.
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        let mut queue = self.queue.unwrap_or_default();
        queue.max_concurrent_requests = max;
        self.queue = Some(queue);
        self
    }

    /// Sets the maximum queue size for pending requests.
    pub fn max_queue_size(mut self, max: usize) -> Self {
        let mut queue = self.queue.unwrap_or_default();
        queue.max_queue_size = max;
        self.queue = Some(queue);
        self
    }

    /// Sets the validation limits.
    pub fn validation_limits(mut self, limits: ValidationLimits) -> Self {
        self.validation_limits = Some(limits);
        self
    }

    /// Sets the timeout configuration.
    pub fn timeouts(mut self, timeouts: TimeoutConfig) -> Self {
        self.timeouts = Some(timeouts);
        self
    }

    /// Sets the queue configuration.
    pub fn queue(mut self, queue: QueueConfig) -> Self {
        self.queue = Some(queue);
        self
    }

    /// Builds the server config.
    pub fn build(self) -> ServerConfig {
        ServerConfig {
            addr: self.addr.unwrap_or_else(|| SocketAddr::from(DEFAULT_ADDR)),
            cors: self.cors.unwrap_or(true),
            model: self.model,
            draft_model: self.draft_model,
            speculative_tokens: self.speculative_tokens.unwrap_or(5),
            validation_limits: self.validation_limits.unwrap_or_default(),
            timeouts: self.timeouts.unwrap_or_default(),
            queue: self.queue.unwrap_or_default(),
        }
    }
}

/// Shared application state.
///
/// # Lock Ordering
///
/// To prevent deadlocks, locks must be acquired in this order:
/// 1. `engine` (RwLock) - acquired by handlers for inference operations
/// 2. `auth_config` (in auth module) - acquired for key validation
/// 3. `rate_limit_entries` (in security module) - acquired for rate limit checks
///
/// All locks use `tokio::sync::RwLock` for async-safe access. Handlers should
/// hold locks for the minimum duration necessary and avoid nested lock acquisition.
pub struct AppState {
    /// The inference engine (None if no model is loaded).
    ///
    /// Use `read()` for inference operations, `write()` only for model load/unload.
    pub engine: RwLock<Option<Arc<Engine>>>,
    /// Server configuration (read-only after initialization).
    pub config: ServerConfig,
    /// Server start time for uptime calculations.
    pub start_time: Instant,
    /// Metrics collector for observability.
    pub metrics: MetricsCollector,
    /// Request queue semaphore for backpressure control.
    pub request_semaphore: Arc<Semaphore>,
    /// Current queue depth (waiting requests).
    pub queue_depth: AtomicUsize,
    /// Active request count.
    pub active_requests: AtomicUsize,
    /// Whether the server is shutting down.
    pub is_shutting_down: AtomicBool,
    /// Total requests processed (for health checks).
    pub total_requests: AtomicU64,
    /// Failed requests count (for health checks).
    pub failed_requests: AtomicU64,
    /// RAG state for Stolas integration.
    pub rag: Arc<RwLock<RagState>>,
    /// Agent session registry for monitoring.
    pub sessions: Arc<SessionRegistry>,
    /// Model cache state for resolving cached models.
    pub model_cache: ModelCacheState,
    /// Speculative decoding engine (enabled when both draft and target models are configured).
    pub speculative_engine: RwLock<Option<Arc<SpeculativeEngine>>>,
    /// Batch scheduler for continuous batching.
    batch_scheduler: Arc<BatchScheduler>,
    /// Request batcher for parallel inference (initialized when engine is loaded).
    pub batcher: RwLock<Option<BatcherHandle>>,
}

impl AppState {
    /// Creates new app state with the given config.
    pub fn new(config: ServerConfig) -> Self {
        let telemetry_config = TelemetryConfig::new("infernum-server");
        let semaphore = Arc::new(Semaphore::new(config.queue.max_concurrent_requests));

        // Configure batch scheduler from server's queue config
        let batch_config = BatchConfig::new()
            .with_max_batch_size(config.queue.max_concurrent_requests)
            .with_max_queue_size(config.queue.max_queue_size);

        Self {
            engine: RwLock::new(None),
            request_semaphore: semaphore,
            queue_depth: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
            is_shutting_down: AtomicBool::new(false),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            config,
            start_time: Instant::now(),
            metrics: MetricsCollector::new(&telemetry_config),
            rag: Arc::new(RwLock::new(RagState::new())),
            sessions: Arc::new(SessionRegistry::new()),
            model_cache: ModelCacheState::new(),
            speculative_engine: RwLock::new(None),
            batch_scheduler: Arc::new(BatchScheduler::new(batch_config)),
            batcher: RwLock::new(None),
        }
    }

    /// Creates new app state with a pre-loaded engine.
    pub fn with_engine(config: ServerConfig, engine: Engine) -> Self {
        let telemetry_config = TelemetryConfig::new("infernum-server");
        let metrics = MetricsCollector::new(&telemetry_config);
        metrics.prometheus().set_model_loaded(true);
        let semaphore = Arc::new(Semaphore::new(config.queue.max_concurrent_requests));

        // Configure batch scheduler from server's queue config
        let batch_config = BatchConfig::new()
            .with_max_batch_size(config.queue.max_concurrent_requests)
            .with_max_queue_size(config.queue.max_queue_size);

        // Create engine Arc and start the request batcher
        let engine_arc = Arc::new(engine);
        let batcher_config = BatcherConfig {
            max_batch_size: config.queue.max_concurrent_requests.min(8),
            max_queue_size: config.queue.max_queue_size,
            ..BatcherConfig::default()
        };
        let batcher = RequestBatcher::new(batcher_config);
        let batcher_handle = batcher.start(engine_arc.clone());

        Self {
            engine: RwLock::new(Some(engine_arc)),
            request_semaphore: semaphore,
            queue_depth: AtomicUsize::new(0),
            active_requests: AtomicUsize::new(0),
            is_shutting_down: AtomicBool::new(false),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            config,
            start_time: Instant::now(),
            metrics,
            rag: Arc::new(RwLock::new(RagState::new())),
            sessions: Arc::new(SessionRegistry::new()),
            model_cache: ModelCacheState::new(),
            speculative_engine: RwLock::new(None),
            batch_scheduler: Arc::new(BatchScheduler::new(batch_config)),
            batcher: RwLock::new(Some(batcher_handle)),
        }
    }

    /// Checks if the server is healthy and ready to accept requests.
    pub fn is_healthy(&self) -> bool {
        !self.is_shutting_down.load(Ordering::Relaxed)
    }

    /// Gets the current queue depth.
    pub fn get_queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Gets the current active request count.
    pub fn get_active_requests(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Calculates the error rate over total requests.
    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let failed = self.failed_requests.load(Ordering::Relaxed);
        failed as f64 / total as f64
    }

    /// Gets the batch scheduler for continuous batching.
    pub fn batch_scheduler(&self) -> &BatchScheduler {
        &self.batch_scheduler
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
    ///
    /// Returns the fully-configured Axum router with all endpoints and middleware.
    /// Useful for testing with a custom server harness.
    pub fn router(&self) -> Router {
        let default_timeout = self.config.timeouts.default;

        // RAG routes with their own state (Arc<RwLock<RagState>>)
        let rag_router = Router::new()
            .route("/health", get(rag_health))
            .route("/documents", get(list_documents))
            .route("/documents", post(index_document))
            .route("/documents/count", get(document_count))
            .route("/documents/{id}", axum::routing::delete(delete_document))
            .route("/search", post(search))
            .with_state(self.state.rag.clone());

        // Agent sessions routes with their own state (Arc<SessionRegistry>)
        let sessions_router = Router::new()
            .route("/", get(list_sessions))
            .route("/stream", get(sessions_stream))
            .route("/{session_id}", get(get_session))
            .route("/{session_id}/stream", get(session_stream))
            .route("/{session_id}/cancel", post(cancel_session))
            .with_state(self.state.sessions.clone());

        // Model cache management routes
        let cache_router = Router::new()
            .route("/models", get(list_cached_models))
            .route("/models/delete", post(delete_cached_model))
            .route("/models/convert", post(convert_model))
            .route("/models/download", post(download_model))
            .with_state(ModelCacheState::new());

        let mut router = Router::new()
            // Health endpoints (no timeout - must always respond quickly)
            .route("/health", get(health))
            .route("/health/deep", get(deep_health))
            .route("/ready", get(ready))
            // Metrics endpoint (Prometheus format)
            .route("/metrics", get(prometheus_metrics))
            // Inference API endpoints
            .route("/v1/models", get(list_models))
            .route("/v1/tokenize", post(tokenize))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            // Internal management endpoints
            .route("/api/models/load", post(load_model))
            .route("/api/models/unload", post(unload_model))
            .route("/api/status", get(server_status))
            // Speculative decoding monitoring
            .route("/api/speculative/stats", get(speculative_stats))
            // Continuous batching monitoring
            .route("/v1/batching/stats", get(batching_stats))
            // RAG endpoints (Stolas integration)
            .nest("/api/rag", rag_router)
            // Agent session endpoints (Beleth monitoring)
            .nest("/api/agent/sessions", sessions_router)
            // Agentic loop endpoint (Beleth executor)
            .route("/api/agent/run", post(run_agent))
            // Model cache management endpoints
            .nest("/api/cache", cache_router)
            .with_state(self.state.clone());

        // Add middleware (order matters - last added is first executed)
        router = router
            // Request body size limit (prevents DoS with oversized payloads)
            .layer(DefaultBodyLimit::max(self.config.validation_limits.max_body_size))
            // Global request timeout (returns 408 Request Timeout on expiry)
            .layer(TimeoutLayer::with_status_code(axum::http::StatusCode::REQUEST_TIMEOUT, default_timeout))
            // Request tracing
            .layer(TraceLayer::new_for_http())
            // Request ID middleware (generates/propagates x-request-id)
            .layer(middleware::from_fn(request_id_middleware));

        if self.config.cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }

    /// Loads a model into the server.
    ///
    /// Supports HoloTensor models with the `holo://` URL scheme:
    /// `holo:///path/to/hct?min=0.7&target=0.95`
    pub async fn load_model(&self, model_source: &str) -> Result<()> {
        tracing::info!(model = %model_source, "Loading model");

        let engine_config = if model_source.starts_with("holo://") {
            // Parse HoloTensor URL: holo://path?min=0.7&target=0.95
            let url_part = model_source.strip_prefix("holo://").unwrap_or(model_source);
            let (path, params) = url_part.split_once('?').unwrap_or((url_part, ""));

            let mut min_quality = 0.7f32;
            let mut target_quality = 0.95f32;

            for param in params.split('&') {
                if let Some((key, value)) = param.split_once('=') {
                    match key {
                        "min" => min_quality = value.parse().unwrap_or(0.7),
                        "target" => target_quality = value.parse().unwrap_or(0.95),
                        _ => {}
                    }
                }
            }

            tracing::info!(
                path = %path,
                min_quality = %min_quality,
                target_quality = %target_quality,
                "Loading HoloTensor model with progressive quality"
            );

            use abaddon::HoloTensorConfig;
            use infernum_core::ModelSource;

            EngineConfig::builder()
                .model_source(ModelSource::holotensor_with_quality(path, min_quality, target_quality))
                .holotensor(HoloTensorConfig {
                    min_quality,
                    target_quality,
                    ..HoloTensorConfig::for_rtx_4500()
                })
                .build()
                .map_err(|e| infernum_core::Error::Internal { message: e })?
        } else {
            EngineConfig::builder()
                .model(model_source)
                .build()
                .map_err(|e| infernum_core::Error::Internal { message: e })?
        };

        let engine = Engine::new(engine_config).await?;
        let engine_arc = Arc::new(engine);

        // Start the request batcher for parallel inference
        let batcher_config = BatcherConfig {
            max_batch_size: self.config.queue.max_concurrent_requests.min(8),
            max_queue_size: self.config.queue.max_queue_size,
            ..BatcherConfig::default()
        };
        let batcher = RequestBatcher::new(batcher_config);
        let batcher_handle = batcher.start(engine_arc.clone());

        // Update both engine and batcher
        {
            let mut engine_guard = self.state.engine.write().await;
            *engine_guard = Some(engine_arc);
        }
        {
            let mut batcher_guard = self.state.batcher.write().await;
            *batcher_guard = Some(batcher_handle);
        }

        tracing::info!(model = %model_source, "Model loaded successfully with batching enabled");
        Ok(())
    }

    /// Runs the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot start.
    pub async fn run(self) -> Result<()> {
        // Check if speculative decoding is enabled FIRST
        // When enabled, ONLY load the speculative engine to avoid loading the large model twice
        let speculative_mode = self.config.is_speculative_enabled();

        if speculative_mode {
            let draft_model = self.config.draft_model.as_ref().expect("draft model checked");
            let target_model = self.config.model.as_ref().expect("model checked");

            // Parse HoloTensor URL params if present: holo://path?min=0.7&target=0.95
            let (model_path, min_quality, target_quality) = if target_model.starts_with("holo://") {
                let url_part = target_model.strip_prefix("holo://").unwrap_or(target_model);
                let (path, params) = url_part.split_once('?').unwrap_or((url_part, ""));

                let mut min_q = 0.7f32;
                let mut target_q = 0.95f32;

                for param in params.split('&') {
                    if let Some((key, value)) = param.split_once('=') {
                        match key {
                            "min" => min_q = value.parse().unwrap_or(0.7),
                            "target" => target_q = value.parse().unwrap_or(0.95),
                            _ => {}
                        }
                    }
                }

                (path.to_string(), min_q, target_q)
            } else {
                (target_model.clone(), 0.7, 0.95)
            };

            // Get cache directory from environment
            let cache_dir = std::env::var("INFERNUM_CACHE_DIR").ok();

            tracing::info!(
                draft = %draft_model,
                target = %model_path,
                tokens = self.config.speculative_tokens,
                min_quality = min_quality,
                target_quality = target_quality,
                cache_dir = ?cache_dir,
                "Initializing speculative decoding engine (model loaded once)"
            );

            // Build speculative engine - this is the ONLY place the target model is loaded
            // VRAM budget for 70B on 24GB with 1B draft model:
            // - Draft model: ~2GB
            // - Embeddings: ~2GB
            // - 5 layers @ ~1.75GB: ~8.75GB
            // - KV cache: ~0.5GB
            // - Activations/buffers: ~2GB
            // - Safety margin: ~8GB
            let mut builder = SpeculativeEngineBuilder::new()
                .draft_model(draft_model)
                .target_model(&model_path)
                .num_draft_tokens(self.config.speculative_tokens as usize)
                .min_quality(min_quality)
                .target_quality(target_quality)
                .vram_budget(20 * 1024 * 1024 * 1024)  // 20GB VRAM (leave 4GB headroom)
                .ram_budget(64 * 1024 * 1024 * 1024)   // 64GB RAM
                .max_loaded_layers(5);                 // 5 layers for 70B on 24GB with draft model

            if let Some(dir) = cache_dir {
                builder = builder.cache_dir(dir);
            }

            match builder.build().await {
                Ok(engine) => {
                    let mut spec_guard = self.state.speculative_engine.write().await;
                    *spec_guard = Some(Arc::new(engine));
                    tracing::info!(
                        model = %model_path,
                        "Speculative decoding engine ready (draft + target models loaded)"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Failed to initialize speculative engine"
                    );
                    // Fall back to loading just the main engine
                    tracing::warn!("Falling back to standard inference (no speculative decoding)");
                    if let Some(model) = &self.config.model {
                        self.load_model(model).await?;
                        tracing::info!(model = %model, "Model loaded for standard inference");
                    }
                }
            }
        } else if let Some(model) = &self.config.model {
            // Standard mode (no speculative decoding) - load main engine
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

        // Clone state for shutdown handling
        let state = self.state.clone();
        let shutdown_timeout = self.config.timeouts.shutdown;

        // Set up graceful shutdown with proper error handling (no panics)
        let shutdown_signal = async move {
            let ctrl_c = async {
                match tokio::signal::ctrl_c().await {
                    Ok(()) => Some("Ctrl+C"),
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to listen for Ctrl+C signal");
                        None
                    }
                }
            };

            #[cfg(unix)]
            let terminate = async {
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(mut signal) => {
                        signal.recv().await;
                        Some("SIGTERM")
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to listen for SIGTERM signal");
                        None
                    }
                }
            };

            #[cfg(not(unix))]
            let terminate = std::future::pending::<Option<&str>>();

            let signal_name = tokio::select! {
                name = ctrl_c => name,
                name = terminate => name,
            };

            if let Some(name) = signal_name {
                eprintln!("\n\x1b[33m⚡\x1b[0m Received {}, shutting down gracefully...", name);
                tracing::info!(signal = name, "Initiating graceful shutdown");
            }

            // Mark server as shutting down
            state.is_shutting_down.store(true, Ordering::SeqCst);

            // Wait for active requests to complete (with timeout)
            let drain_start = Instant::now();
            while state.get_active_requests() > 0 {
                if drain_start.elapsed() > shutdown_timeout {
                    let remaining = state.get_active_requests();
                    tracing::warn!(
                        remaining_requests = remaining,
                        "Shutdown timeout reached, forcing shutdown"
                    );
                    eprintln!(
                        "\x1b[33m⚠\x1b[0m Shutdown timeout, {} requests still active",
                        remaining
                    );
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
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

/// Creates an API error response with proper error code and request ID.
#[inline]
fn typed_error(code: ErrorCode, request_id: &str) -> Response {
    api_error(code, request_id).into_response()
}

/// Creates an API error response with custom message.
#[inline]
fn typed_error_with_message(code: ErrorCode, request_id: &str, message: &str) -> Response {
    ApiError::with_message(code, request_id, message).into_response()
}

// === Health Endpoints ===

/// Health check response structure.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    timestamp: chrono::DateTime<chrono::Utc>,
    uptime_seconds: u64,
}

/// Readiness check response structure.
#[derive(Debug, Serialize)]
struct ReadyResponse {
    ready: bool,
    model: Option<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Simple health check - always responds quickly with JSON.
async fn health(State(state): State<Arc<AppState>>) -> Response {
    let uptime = state.start_time.elapsed().as_secs();

    if state.is_shutting_down.load(Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "shutting_down",
                timestamp: chrono::Utc::now(),
                uptime_seconds: uptime,
            }),
        ).into_response();
    }

    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            timestamp: chrono::Utc::now(),
            uptime_seconds: uptime,
        }),
    ).into_response()
}

/// Deep health check response.
#[derive(Debug, Serialize)]
struct DeepHealthResponse {
    status: &'static str,
    checks: HealthChecks,
    #[serde(skip_serializing_if = "Option::is_none")]
    degraded_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct HealthChecks {
    server: CheckStatus,
    model: CheckStatus,
    queue: CheckStatus,
    error_rate: CheckStatus,
}

#[derive(Debug, Serialize)]
struct CheckStatus {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

impl CheckStatus {
    fn healthy() -> Self {
        Self { status: "healthy", message: None }
    }

    fn unhealthy(message: impl Into<String>) -> Self {
        Self { status: "unhealthy", message: Some(message.into()) }
    }

    fn degraded(message: impl Into<String>) -> Self {
        Self { status: "degraded", message: Some(message.into()) }
    }
}

/// Deep health check - verifies all components.
async fn deep_health(State(state): State<Arc<AppState>>) -> Response {
    let mut overall_status = "healthy";
    let mut degraded_reason = None;

    // Check server status
    let server_check = if state.is_shutting_down.load(Ordering::Relaxed) {
        overall_status = "unhealthy";
        CheckStatus::unhealthy("Server is shutting down")
    } else {
        CheckStatus::healthy()
    };

    // Check model status
    let engine = state.engine.read().await;
    let model_check = if engine.is_some() {
        CheckStatus::healthy()
    } else {
        // Model not loaded is degraded, not unhealthy
        if overall_status == "healthy" {
            overall_status = "degraded";
            degraded_reason = Some("No model loaded".to_string());
        }
        CheckStatus::degraded("No model loaded")
    };
    drop(engine);

    // Check queue health
    let queue_depth = state.get_queue_depth();
    let max_queue = state.config.queue.max_queue_size;
    let queue_utilization = queue_depth as f64 / max_queue as f64;
    let queue_check = if queue_utilization > 0.9 {
        if overall_status == "healthy" {
            overall_status = "degraded";
            degraded_reason = Some("Request queue near capacity".to_string());
        }
        CheckStatus::degraded(format!("Queue at {}% capacity", (queue_utilization * 100.0) as u32))
    } else if queue_utilization > 0.7 {
        CheckStatus::degraded(format!("Queue at {}% capacity", (queue_utilization * 100.0) as u32))
    } else {
        CheckStatus::healthy()
    };

    // Check error rate
    let error_rate = state.error_rate();
    let error_check = if error_rate > 0.5 {
        if overall_status == "healthy" {
            overall_status = "degraded";
            degraded_reason = Some("High error rate".to_string());
        }
        CheckStatus::degraded(format!("Error rate: {:.1}%", error_rate * 100.0))
    } else if error_rate > 0.1 {
        CheckStatus::degraded(format!("Error rate: {:.1}%", error_rate * 100.0))
    } else {
        CheckStatus::healthy()
    };

    let response = DeepHealthResponse {
        status: overall_status,
        checks: HealthChecks {
            server: server_check,
            model: model_check,
            queue: queue_check,
            error_rate: error_check,
        },
        degraded_reason,
    };

    let status_code = match overall_status {
        "healthy" => StatusCode::OK,
        "degraded" => StatusCode::OK, // Still operational
        _ => StatusCode::SERVICE_UNAVAILABLE,
    };

    (status_code, Json(response)).into_response()
}

/// Prometheus metrics endpoint.
async fn prometheus_metrics(State(state): State<Arc<AppState>>) -> Response {
    let mut metrics = state.metrics.render_prometheus();

    // Add queue metrics
    metrics.push_str("\n# HELP infernum_queue_depth Current number of requests waiting in queue.\n");
    metrics.push_str("# TYPE infernum_queue_depth gauge\n");
    metrics.push_str(&format!(
        "infernum_queue_depth {}\n",
        state.queue_depth.load(Ordering::Relaxed)
    ));

    metrics.push_str("# HELP infernum_queue_capacity Maximum queue capacity.\n");
    metrics.push_str("# TYPE infernum_queue_capacity gauge\n");
    metrics.push_str(&format!(
        "infernum_queue_capacity {}\n",
        state.config.queue.max_queue_size
    ));

    metrics.push_str("# HELP infernum_concurrent_requests_limit Maximum concurrent requests allowed.\n");
    metrics.push_str("# TYPE infernum_concurrent_requests_limit gauge\n");
    metrics.push_str(&format!(
        "infernum_concurrent_requests_limit {}\n",
        state.config.queue.max_concurrent_requests
    ));

    metrics.push_str("# HELP infernum_active_requests_total Current number of requests being processed.\n");
    metrics.push_str("# TYPE infernum_active_requests_total gauge\n");
    metrics.push_str(&format!(
        "infernum_active_requests_total {}\n",
        state.active_requests.load(Ordering::Relaxed)
    ));

    metrics.push_str("# HELP infernum_total_requests_served Total requests served since startup.\n");
    metrics.push_str("# TYPE infernum_total_requests_served counter\n");
    metrics.push_str(&format!(
        "infernum_total_requests_served {}\n",
        state.total_requests.load(Ordering::Relaxed)
    ));

    metrics.push_str("# HELP infernum_failed_requests_total Total failed requests since startup.\n");
    metrics.push_str("# TYPE infernum_failed_requests_total counter\n");
    metrics.push_str(&format!(
        "infernum_failed_requests_total {}\n",
        state.failed_requests.load(Ordering::Relaxed)
    ));

    metrics.push_str("# HELP infernum_uptime_seconds Server uptime in seconds.\n");
    metrics.push_str("# TYPE infernum_uptime_seconds gauge\n");
    metrics.push_str(&format!(
        "infernum_uptime_seconds {}\n",
        state.start_time.elapsed().as_secs()
    ));

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics,
    )
        .into_response()
}

/// Readiness check - verifies model is loaded and ready.
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    // Check if shutting down
    if state.is_shutting_down.load(Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                ready: false,
                model: None,
                timestamp: chrono::Utc::now(),
            }),
        ).into_response();
    }

    let engine = state.engine.read().await;
    if let Some(ref eng) = *engine {
        (
            StatusCode::OK,
            Json(ReadyResponse {
                ready: true,
                model: Some(eng.model_info().id.to_string()),
                timestamp: chrono::Utc::now(),
            }),
        ).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyResponse {
                ready: false,
                model: None,
                timestamp: chrono::Utc::now(),
            }),
        ).into_response()
    }
}

#[derive(Debug, Serialize)]
struct ServerStatus {
    version: &'static str,
    status: String,
    uptime_seconds: u64,
    model_loaded: bool,
    model_id: Option<String>,
    active_requests: usize,
    queue_depth: usize,
    total_requests: u64,
    error_rate: f64,
    is_shutting_down: bool,
}

async fn server_status(State(state): State<Arc<AppState>>) -> Json<ServerStatus> {
    let engine = state.engine.read().await;
    let model_id = engine.as_ref().map(|e| e.model_info().id.to_string());

    Json(ServerStatus {
        version: env!("CARGO_PKG_VERSION"),
        status: if state.is_shutting_down.load(Ordering::Relaxed) {
            "shutting_down".to_string()
        } else {
            "running".to_string()
        },
        uptime_seconds: state.start_time.elapsed().as_secs(),
        model_loaded: engine.is_some(),
        model_id,
        active_requests: state.get_active_requests(),
        queue_depth: state.get_queue_depth(),
        total_requests: state.total_requests.load(Ordering::Relaxed),
        error_rate: state.error_rate(),
        is_shutting_down: state.is_shutting_down.load(Ordering::Relaxed),
    })
}

// === Speculative Decoding Stats ===

#[derive(Debug, Serialize)]
struct SpeculativeStatsResponse {
    enabled: bool,
    rounds: u64,
    draft_tokens: u64,
    accepted_tokens: u64,
    rejected_tokens: u64,
    acceptance_rate: f32,
    tokens_per_round: f32,
    speedup: f32,
    target_forward_passes: u64,
    draft_forward_passes: u64,
    draft_time_ms: u64,
    verify_time_ms: u64,
}

async fn speculative_stats(State(state): State<Arc<AppState>>) -> Json<SpeculativeStatsResponse> {
    let guard = state.speculative_engine.read().await;

    if let Some(spec_engine) = guard.as_ref() {
        let stats = spec_engine.stats();
        Json(SpeculativeStatsResponse {
            enabled: true,
            rounds: stats.rounds,
            draft_tokens: stats.draft_tokens,
            accepted_tokens: stats.accepted_tokens,
            rejected_tokens: stats.rejected_tokens,
            acceptance_rate: stats.acceptance_rate(),
            tokens_per_round: stats.tokens_per_round(),
            speedup: stats.speedup(),
            target_forward_passes: stats.target_forward_passes,
            draft_forward_passes: stats.draft_forward_passes,
            draft_time_ms: stats.draft_time_ms,
            verify_time_ms: stats.verify_time_ms,
        })
    } else {
        Json(SpeculativeStatsResponse {
            enabled: false,
            rounds: 0,
            draft_tokens: 0,
            accepted_tokens: 0,
            rejected_tokens: 0,
            acceptance_rate: 0.0,
            tokens_per_round: 0.0,
            speedup: 0.0,
            target_forward_passes: 0,
            draft_forward_passes: 0,
            draft_time_ms: 0,
            verify_time_ms: 0,
        })
    }
}

// === Batching Statistics ===

/// Response for batching statistics endpoint.
#[derive(Debug, Serialize)]
struct BatchingStatsResponse {
    /// Number of pending requests in the queue.
    pending_requests: usize,
    /// Size of the currently active batch (0 if no active batch).
    active_batch_size: usize,
    /// Total number of batches formed.
    total_batches_formed: u64,
    /// Total requests submitted to the scheduler.
    total_requests_submitted: u64,
}

/// Returns batching statistics for monitoring continuous batching.
async fn batching_stats(State(state): State<Arc<AppState>>) -> Json<BatchingStatsResponse> {
    let scheduler = state.batch_scheduler();
    let metrics = scheduler.metrics();

    Json(BatchingStatsResponse {
        pending_requests: scheduler.pending_count(),
        active_batch_size: scheduler.current_batch_size(),
        total_batches_formed: metrics.batches_formed(),
        total_requests_submitted: metrics.requests_submitted(),
    })
}

// === Model Management ===

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model: String,
}

/// Validates model identifier for security.
/// Prevents path traversal and invalid characters.
#[inline]
fn validate_model_id(model: &str) -> std::result::Result<(), (StatusCode, &'static str)> {
    // Check for empty
    if model.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "model identifier cannot be empty"));
    }

    // Check length
    if model.len() > 256 {
        return Err((StatusCode::BAD_REQUEST, "model identifier too long"));
    }

    // Check for path traversal attempts
    if model.contains("..") {
        return Err((StatusCode::BAD_REQUEST, "model identifier contains invalid characters"));
    }

    // Check for absolute paths (security concern)
    if model.starts_with('/') || model.starts_with('\\') {
        return Err((StatusCode::BAD_REQUEST, "model identifier cannot be an absolute path"));
    }

    // Check for shell metacharacters
    const FORBIDDEN_CHARS: &[char] = &['$', '`', ';', '|', '&', '>', '<', '\n', '\r', '\0'];
    if model.chars().any(|c| FORBIDDEN_CHARS.contains(&c)) {
        return Err((StatusCode::BAD_REQUEST, "model identifier contains invalid characters"));
    }

    Ok(())
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let request_id = format!("load-{}", uuid::Uuid::new_v4());

    // Validate model identifier
    if let Err((_, message)) = validate_model_id(&req.model) {
        return typed_error_with_message(ErrorCode::InvalidModel, &request_id, message);
    }

    tracing::info!(request_id = %request_id, model = %req.model, "Loading model via API");

    // Resolve model source - check cache first for local/HCT models
    let model_source = if let Some(cached_path) = find_model_path(&state.model_cache, &req.model) {
        if is_holotensor_model(&cached_path) {
            // Use HoloTensor source for .hct models
            tracing::info!(
                path = %cached_path.display(),
                "Resolved cached HoloTensor model"
            );
            ModelSource::holotensor_with_quality(
                cached_path.display().to_string(),
                0.7,  // min_quality
                0.95, // target_quality
            )
        } else {
            // Use local path for regular cached models
            tracing::info!(
                path = %cached_path.display(),
                "Resolved cached model path"
            );
            ModelSource::local(cached_path.display().to_string())
        }
    } else {
        // Not in cache - use as HuggingFace model ID
        ModelSource::huggingface(&req.model)
    };

    let engine_config = match EngineConfig::builder().model_source(model_source).build() {
        Ok(config) => config,
        Err(e) => {
            return typed_error_with_message(
                ErrorCode::InvalidModel,
                &request_id,
                &format!("Invalid model configuration: {}", e),
            );
        },
    };

    let engine = match Engine::new(engine_config).await {
        Ok(engine) => engine,
        Err(e) => {
            return typed_error_with_message(
                ErrorCode::InternalError,
                &request_id,
                &format!("Failed to load model: {}", e),
            );
        },
    };

    let mut engine_guard = state.engine.write().await;
    *engine_guard = Some(Arc::new(engine));

    // Update metrics
    state.metrics.prometheus().set_model_loaded(true);

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

    // Update metrics
    state.metrics.prometheus().set_model_loaded(false);

    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "unloaded"})),
    )
        .into_response()
}

// === API Endpoints ===

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

/// Token counting endpoint.
///
/// Counts tokens in a prompt or messages without running inference.
/// Useful for pre-flight validation and cost estimation.
async fn tokenize(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<crate::tokenize::TokenizeRequest>,
) -> Response {
    use crate::tokenize::{count_tokens, EstimatingTokenizer, TokenizeError};

    // Validate request
    if let Err(e) = req.validate() {
        let error_code = match e {
            TokenizeError::NoInput | TokenizeError::BothInputs => ErrorCode::InvalidMessages,
            TokenizeError::EmptyModel => ErrorCode::InvalidModel,
            _ => ErrorCode::InternalError,
        };
        return api_error(error_code, "tokenize-error").into_response();
    }

    // Use the estimating tokenizer (can be swapped for model-specific later)
    let tokenizer = EstimatingTokenizer;

    match count_tokens(&tokenizer, &req) {
        Ok(response) => Json(response).into_response(),
        Err(e) => {
            let error_code = match e {
                TokenizeError::ModelNotFound(_) => ErrorCode::ModelNotFound,
                TokenizeError::TokenizationFailed(_) => ErrorCode::InternalError,
                _ => ErrorCode::InternalError,
            };
            api_error(error_code, "tokenize-error").into_response()
        }
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = format!("inf-chat-{}", uuid::Uuid::new_v4());
    let error_request_id = request_id.clone(); // Clone for error handling

    tracing::debug!(request_id = %request_id, model = %req.model, "Chat completion request");

    // Check if server is shutting down
    if state.is_shutting_down.load(Ordering::Relaxed) {
        return typed_error_with_message(
            ErrorCode::ServiceOverloaded,
            &request_id,
            "Server is shutting down",
        );
    }

    // Check queue capacity before waiting
    let current_queue = state.queue_depth.load(Ordering::Relaxed);
    if current_queue >= state.config.queue.max_queue_size {
        state.failed_requests.fetch_add(1, Ordering::Relaxed);
        return ApiError::new(ErrorCode::ServiceOverloaded, &request_id)
            .message("Server overloaded, please retry later")
            .retry_after(5)
            .build()
            .into_response();
    }

    // Increment queue depth while waiting for permit
    state.queue_depth.fetch_add(1, Ordering::Relaxed);

    // Acquire semaphore permit for backpressure
    let permit = match state.request_semaphore.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            state.queue_depth.fetch_sub(1, Ordering::Relaxed);
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
            return typed_error_with_message(
                ErrorCode::ServiceOverloaded,
                &request_id,
                "Server is shutting down",
            );
        }
    };

    // Decrement queue, increment active
    state.queue_depth.fetch_sub(1, Ordering::Relaxed);
    state.active_requests.fetch_add(1, Ordering::Relaxed);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Validate request using typed validation
    if let Err(validation_err) = validate_chat_request(&req, &state.config.validation_limits) {
        state.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        return validation_err.to_api_error(&request_id).into_response();
    }

    // Validate tool_choice: if a specific tool is forced, it must exist in tools list
    if let Some(forced_tool) = get_forced_tool(req.tool_choice.as_ref()) {
        let tools = req.tools.as_deref().unwrap_or(&[]);
        if !validate_tool_exists(forced_tool, tools) {
            state.active_requests.fetch_sub(1, Ordering::Relaxed);
            drop(permit);
            return ApiError::new(ErrorCode::InvalidRequest, &request_id)
                .message(&format!("Tool '{}' not found in tools list", forced_tool))
                .param("tool_choice")
                .build()
                .into_response();
        }
    }

    // Validate tool_choice: "required" must have at least one tool defined
    if let Some(ToolChoice::String(ref choice)) = req.tool_choice {
        if choice == "required" && req.tools.as_ref().map_or(true, |t| t.is_empty()) {
            state.active_requests.fetch_sub(1, Ordering::Relaxed);
            drop(permit);
            return ApiError::new(ErrorCode::InvalidRequest, &request_id)
                .message("tool_choice is 'required' but no tools are defined")
                .param("tool_choice")
                .build()
                .into_response();
        }
    }

    // Check for available inference engines (speculative or standard)
    // In speculative mode, main engine is intentionally not loaded to avoid OOM
    let speculative_guard = state.speculative_engine.read().await;
    let has_speculative = speculative_guard.is_some();
    drop(speculative_guard);

    let engine_guard = state.engine.read().await;
    let engine = engine_guard.as_ref().map(Arc::clone);
    drop(engine_guard);

    // Require at least one engine to be available
    if engine.is_none() && !has_speculative {
        state.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        return typed_error(ErrorCode::ModelNotLoaded, &request_id);
    }

    // Check for streaming
    let stream = req.stream.unwrap_or(false);

    // Determine model family for tool formatting
    let model_family = ModelFamily::from_model_name(&req.model);

    // Format tools for prompt injection (if tools are present and should be included)
    let tools_prompt = if should_include_tools(req.tool_choice.as_ref()) {
        req.tools
            .as_ref()
            .map(|tools| format_tools_for_prompt(tools, model_family))
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Build messages into prompt, injecting tools into system message
    let messages: Vec<infernum_core::Message> = req
        .messages
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let role = match m.role.as_str() {
                "system" => infernum_core::Role::System,
                "user" => infernum_core::Role::User,
                "assistant" => infernum_core::Role::Assistant,
                "tool" => infernum_core::Role::User, // Tool results sent as user messages
                _ => infernum_core::Role::User,
            };

            // Inject tools into the first system message, or create one if needed.
            // For multi-turn tool conversations, reconstruct model-native format
            // per TOOL-CALLING-SPEC §5.5.
            let content = if role == infernum_core::Role::System && idx == 0 && !tools_prompt.is_empty() {
                format!("{}{}", m.content, tools_prompt)
            } else if m.role == "tool" {
                // §5.5: Format tool result in model-native format
                match model_family {
                    ModelFamily::Qwen => {
                        format!("<tool_response>\n{}\n</tool_response>", m.content)
                    }
                    _ => {
                        if let Some(ref tool_call_id) = m.tool_call_id {
                            format!("[Tool Result for {}]: {}", tool_call_id, m.content)
                        } else {
                            format!("[Tool Result]: {}", m.content)
                        }
                    }
                }
            } else if m.role == "assistant" && m.tool_calls.is_some() {
                // §5.5: Reconstruct native tool_call tags in assistant message content
                match model_family {
                    ModelFamily::Qwen => {
                        use std::fmt::Write;
                        let mut content = m.content.clone();
                        if let Some(ref calls) = m.tool_calls {
                            for tc in calls {
                                let _ = write!(
                                    content,
                                    "\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
                                    tc.function.name, tc.function.arguments
                                );
                            }
                        }
                        content
                    }
                    _ => m.content.clone(),
                }
            } else {
                m.content.clone()
            };

            infernum_core::Message {
                role,
                content,
                name: None,
                tool_calls: None,
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect();

    // If no system message exists but we have tools, prepend a system message with tools
    let messages = if !tools_prompt.is_empty()
        && !messages.iter().any(|m| m.role == infernum_core::Role::System)
    {
        let mut new_messages = vec![infernum_core::Message {
            role: infernum_core::Role::System,
            content: format!("You are a helpful assistant.{}", tools_prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        new_messages.extend(messages);
        new_messages
    } else {
        messages
    };

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
    if let Some(presence_penalty) = req.presence_penalty {
        sampling = sampling.with_presence_penalty(presence_penalty);
    }
    if let Some(frequency_penalty) = req.frequency_penalty {
        sampling = sampling.with_frequency_penalty(frequency_penalty);
    }
    // Apply a default repetition penalty if no penalty specified to prevent loops
    // Presence/frequency penalties map to repetition penalty when neither is set
    if req.presence_penalty.is_none() && req.frequency_penalty.is_none() {
        sampling = sampling.with_repetition_penalty(1.1);
    }

    // Create inference request
    let gen_request = GenerateRequest::new(infernum_core::request::PromptInput::Messages(messages))
        .with_sampling(sampling);

    let response = if stream {
        // Streaming response - Note: permit is kept alive via state clone
        // Speculative decoding doesn't support streaming (batches tokens), require main engine
        let engine = match engine {
            Some(e) => e,
            None => {
                state.active_requests.fetch_sub(1, Ordering::Relaxed);
                drop(permit);
                return typed_error_with_message(
                    ErrorCode::InvalidRequest,
                    &request_id,
                    "Streaming not supported in speculative decoding mode. Use stream=false.",
                );
            }
        };
        let state_clone = state.clone();
        match engine.generate_stream(gen_request).await {
            Ok(token_stream) => {
                let model_name = engine.model_info().id.to_string();

                // Agent-centric streaming format (INFERNUM-SPEC.md §10)
                // Complete, typed events - no partial JSON accumulation required by agents
                let model_family = ModelFamily::from_model_name(&model_name);
                let detector = std::sync::Arc::new(parking_lot::Mutex::new(
                    StreamingToolDetector::new(model_family)
                ));

                let sse_stream = token_stream.flat_map(move |chunk_result| {
                    let events: Vec<std::result::Result<axum::response::sse::Event, std::convert::Infallible>> = match chunk_result {
                        Ok(chunk) => {
                            // Extract content from chunk
                            let content = chunk.choices.first()
                                .and_then(|c| c.delta.content.as_deref())
                                .unwrap_or("");
                            let finish_reason = chunk.choices.first()
                                .and_then(|c| c.finish_reason.as_ref());

                            let mut result_events = Vec::new();

                            // Process through detector
                            if !content.is_empty() {
                                let detection_events = detector.lock().process_chunk(content);
                                for event in detection_events {
                                    if let Some(sse_event) = Option::<SseEvent>::from(event) {
                                        let json_str = sse_event.to_json();
                                        result_events.push(Ok(axum::response::sse::Event::default().data(json_str)));
                                    }
                                }
                            }

                            // Handle stream end - flush buffer and emit done event
                            if finish_reason.is_some() {
                                // Flush any remaining buffer
                                let remaining = detector.lock().finish();
                                for event in remaining {
                                    if let Some(sse_event) = Option::<SseEvent>::from(event) {
                                        let json_str = sse_event.to_json();
                                        result_events.push(Ok(axum::response::sse::Event::default().data(json_str)));
                                    }
                                }

                                // Emit done event
                                let reason = match finish_reason {
                                    Some(r) => format!("{:?}", r).to_lowercase(),
                                    None => "stop".to_string(),
                                };
                                let done_event = if let Some(u) = &chunk.usage {
                                    SseEvent::done_with_usage(&reason, u.prompt_tokens, u.completion_tokens)
                                } else {
                                    SseEvent::done(&reason)
                                };
                                let json_str = done_event.to_json();
                                result_events.push(Ok(axum::response::sse::Event::default().data(json_str)));
                            }

                            result_events
                        }
                        Err(e) => {
                            state_clone.failed_requests.fetch_add(1, Ordering::Relaxed);
                            let error_event = SseEvent::error("server_error", e.to_string());
                            let json_str = error_event.to_json();
                            vec![Ok(axum::response::sse::Event::default().data(json_str))]
                        }
                    };
                    futures::stream::iter(events)
                });

                Sse::new(sse_stream)
                    .keep_alive(axum::response::sse::KeepAlive::default())
                    .into_response()
            },
            Err(e) => {
                state.failed_requests.fetch_add(1, Ordering::Relaxed);
                typed_error_with_message(
                    ErrorCode::InternalError,
                    &error_request_id,
                    &crate::error_response::sanitize_error(&e.to_string()),
                )
            },
        }
    } else {
        // Non-streaming response
        // Try speculative decoding first if available (3-4x faster for large models)
        let speculative_guard = state.speculative_engine.read().await;
        let use_speculative = speculative_guard.is_some();

        if use_speculative {
            // Use speculative decoding - clone the Arc to release the lock
            let spec_engine = Arc::clone(speculative_guard.as_ref().expect("speculative engine checked"));
            drop(speculative_guard);  // Release read lock

            let max_tokens = req.max_tokens.unwrap_or(256) as usize;

            // Build prompt from messages (simple concatenation for now)
            let prompt: String = req.messages.iter().map(|m| {
                match m.role.as_str() {
                    "system" => format!("<|system|>\n{}\n", m.content),
                    "user" => format!("<|user|>\n{}\n", m.content),
                    "assistant" => format!("<|assistant|>\n{}\n", m.content),
                    _ => m.content.clone(),
                }
            }).collect::<String>() + "<|assistant|>\n";

            match spec_engine.generate(&prompt, max_tokens) {
                Ok(raw_content) => {
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let stats = spec_engine.stats();

                    tracing::info!(
                        request_id = %request_id,
                        latency_ms = latency_ms,
                        acceptance_rate = %format!("{:.1}%", stats.acceptance_rate() * 100.0),
                        tokens_per_round = %format!("{:.1}", stats.tokens_per_round()),
                        speculative = true,
                        "Speculative chat completion finished"
                    );

                    // Process output for tool calls if tools were provided
                    let has_tools = req.tools.as_ref().map_or(false, |t| !t.is_empty());
                    let (content, tool_calls, finish_reason) = if has_tools {
                        let result = process_model_output(&raw_content, model_family);
                        (
                            result.content.unwrap_or_default(),
                            if result.tool_calls.is_empty() {
                                None
                            } else {
                                Some(result.tool_calls)
                            },
                            result.finish_reason,
                        )
                    } else {
                        (raw_content, None, "stop".to_string())
                    };

                    // Estimate token counts (not exact, but reasonable)
                    let prompt_tokens = (prompt.len() / 4) as u32;
                    let completion_tokens = (content.len() / 4) as u32;

                    let chat_response = ChatCompletionResponse {
                        id: request_id,
                        object: "chat.completion".to_string(),
                        created: chrono::Utc::now().timestamp(),
                        model: spec_engine.model_id().to_string(),
                        choices: vec![ChatChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content,
                                name: None,
                                tool_calls,
                                tool_call_id: None,
                            },
                            finish_reason,
                            logprobs: None,
                        }],
                        usage: Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        },
                    };

                    state.active_requests.fetch_sub(1, Ordering::Relaxed);
                    drop(permit);

                    return Json(chat_response).into_response();
                }
                Err(e) => {
                    // In speculative-only mode, there's no fallback engine
                    if engine.is_none() {
                        state.active_requests.fetch_sub(1, Ordering::Relaxed);
                        state.failed_requests.fetch_add(1, Ordering::Relaxed);
                        drop(permit);
                        return typed_error_with_message(
                            ErrorCode::InternalError,
                            &error_request_id,
                            &format!("Speculative decoding failed: {}", e),
                        );
                    }
                    tracing::warn!(
                        error = %e,
                        "Speculative decoding failed, falling back to regular generation"
                    );
                    // Fall through to regular generation
                }
            }
        } else {
            drop(speculative_guard);
        }

        // Regular generation (fallback or when speculative not available)
        // Requires main engine to be loaded
        let engine = match engine {
            Some(e) => e,
            None => {
                // This shouldn't happen - we checked earlier
                state.active_requests.fetch_sub(1, Ordering::Relaxed);
                drop(permit);
                return typed_error(ErrorCode::ModelNotLoaded, &request_id);
            }
        };

        // Try to use batcher for parallel inference, fall back to direct generation
        let batcher_handle = state.batcher.read().await.clone();
        let result = match batcher_handle {
            Some(batcher) => {
                // Submit to batcher for batched processing
                match batcher.submit(gen_request).await {
                    Ok(rx) => rx.await.unwrap_or_else(|_| Err(infernum_core::Error::internal("Batcher channel closed"))),
                    Err(e) => Err(e),
                }
            }
            None => {
                // No batcher, use direct generation
                engine.generate(gen_request).await
            }
        };

        match result {
            Ok(response) => {
                let choice = response.choices.first();
                let raw_content = choice.map(|c| c.text.clone()).unwrap_or_default();

                // Process output for tool calls if tools were provided
                let has_tools = req.tools.as_ref().map_or(false, |t| !t.is_empty());
                let (content, tool_calls, finish_reason) = if has_tools {
                    let result = process_model_output(&raw_content, model_family);
                    (
                        result.content.unwrap_or_default(),
                        if result.tool_calls.is_empty() {
                            None
                        } else {
                            Some(result.tool_calls)
                        },
                        result.finish_reason,
                    )
                } else {
                    let finish_reason = choice
                        .and_then(|c| c.finish_reason.as_ref())
                        .map(|r| format!("{:?}", r).to_lowercase())
                        .unwrap_or_else(|| "stop".to_string());
                    (raw_content, None, finish_reason)
                };

                // For tool_choice: "required", verify at least one tool was called
                if let Some(ToolChoice::String(ref tc)) = req.tool_choice {
                    if tc == "required" && tool_calls.is_none() {
                        // Model didn't call any tools despite "required" - log warning
                        // but don't fail the request (model behavior, not server error)
                        tracing::warn!(
                            request_id = %request_id,
                            "tool_choice was 'required' but model did not call any tools"
                        );
                    }
                }

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
                            tool_calls,
                            tool_call_id: None,
                        },
                        finish_reason,
                        logprobs: None,
                    }],
                    usage: Usage {
                        prompt_tokens: response.usage.prompt_tokens,
                        completion_tokens: response.usage.completion_tokens,
                        total_tokens: response.usage.total_tokens,
                    },
                };

                let latency_secs = start.elapsed().as_secs_f64();
                tracing::debug!(
                    request_id = %chat_response.id,
                    prompt_tokens = response.usage.prompt_tokens,
                    completion_tokens = response.usage.completion_tokens,
                    latency_ms = start.elapsed().as_millis() as u64,
                    "Chat completion finished"
                );

                // Record metrics
                state.metrics.record_chat_request(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    latency_secs,
                    &chat_response.model,
                );

                Json(chat_response).into_response()
            },
            Err(e) => {
                state.failed_requests.fetch_add(1, Ordering::Relaxed);
                let model_id = engine.model_info().id.to_string();
                state.metrics.record_error("chat", &model_id, "generation_error");
                typed_error_with_message(
                    ErrorCode::InternalError,
                    &error_request_id,
                    &crate::error_response::sanitize_error(&e.to_string()),
                )
            },
        }
    };

    // Clean up request tracking
    state.active_requests.fetch_sub(1, Ordering::Relaxed);
    drop(permit);

    response
}

/// Validates completion request parameters.
#[inline]
fn validate_completion_request(
    req: &CompletionRequest,
    limits: &ValidationLimits,
) -> std::result::Result<(), (StatusCode, &'static str)> {
    // Validate prompt length
    if req.prompt.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt cannot be empty"));
    }
    if req.prompt.len() > limits.max_prompt_length {
        return Err((StatusCode::BAD_REQUEST, "prompt exceeds maximum length"));
    }

    // Validate temperature
    if let Some(temp) = req.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err((StatusCode::BAD_REQUEST, "temperature must be between 0.0 and 2.0"));
        }
    }

    // Validate top_p
    if let Some(top_p) = req.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err((StatusCode::BAD_REQUEST, "top_p must be between 0.0 and 1.0"));
        }
    }

    // Validate max_tokens
    if let Some(max_tokens) = req.max_tokens {
        if max_tokens == 0 || max_tokens > limits.max_max_tokens {
            return Err((StatusCode::BAD_REQUEST, "max_tokens exceeds maximum allowed value"));
        }
    }

    Ok(())
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = format!("inf-cmpl-{}", uuid::Uuid::new_v4());
    let error_request_id = request_id.clone();

    tracing::debug!(request_id = %request_id, model = %req.model, "Completion request");

    // Check if server is shutting down
    if state.is_shutting_down.load(Ordering::Relaxed) {
        return typed_error_with_message(
            ErrorCode::ServiceOverloaded,
            &request_id,
            "Server is shutting down",
        );
    }

    // Check queue capacity
    let current_queue = state.queue_depth.load(Ordering::Relaxed);
    if current_queue >= state.config.queue.max_queue_size {
        state.failed_requests.fetch_add(1, Ordering::Relaxed);
        return ApiError::new(ErrorCode::ServiceOverloaded, &request_id)
            .message("Server overloaded, please retry later")
            .retry_after(5)
            .build()
            .into_response();
    }

    // Increment queue depth
    state.queue_depth.fetch_add(1, Ordering::Relaxed);

    // Acquire semaphore permit
    let permit = match state.request_semaphore.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            state.queue_depth.fetch_sub(1, Ordering::Relaxed);
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
            return typed_error_with_message(
                ErrorCode::ServiceOverloaded,
                &request_id,
                "Server is shutting down",
            );
        }
    };

    state.queue_depth.fetch_sub(1, Ordering::Relaxed);
    state.active_requests.fetch_add(1, Ordering::Relaxed);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Validate request
    if let Err((_, message)) = validate_completion_request(&req, &state.config.validation_limits) {
        state.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        let code = match message {
            m if m.contains("empty") => ErrorCode::EmptyPrompt,
            m if m.contains("exceeds maximum") => ErrorCode::PromptTooLong,
            m if m.contains("temperature") => ErrorCode::InvalidTemperature,
            m if m.contains("max_tokens") => ErrorCode::InvalidMaxTokens,
            _ => ErrorCode::InvalidMessages,
        };
        return typed_error_with_message(code, &request_id, message);
    }

    // Check for available inference engines (speculative or standard)
    // In speculative mode, main engine is intentionally not loaded to avoid OOM
    let speculative_guard = state.speculative_engine.read().await;
    let has_speculative = speculative_guard.is_some();
    drop(speculative_guard);

    let engine_guard = state.engine.read().await;
    let engine = engine_guard.as_ref().map(Arc::clone);
    drop(engine_guard);

    // If neither engine is available, return error
    if !has_speculative && engine.is_none() {
        state.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        return typed_error(ErrorCode::ModelNotLoaded, &error_request_id);
    }

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
    if let Some(presence_penalty) = req.presence_penalty {
        sampling = sampling.with_presence_penalty(presence_penalty);
    }
    if let Some(frequency_penalty) = req.frequency_penalty {
        sampling = sampling.with_frequency_penalty(frequency_penalty);
    }
    // Apply a default repetition penalty if no penalty specified to prevent loops
    if req.presence_penalty.is_none() && req.frequency_penalty.is_none() {
        sampling = sampling.with_repetition_penalty(1.1);
    }

    // Try speculative decoding first if available (3-4x faster for large models)
    let speculative_guard = state.speculative_engine.read().await;
    let use_speculative = speculative_guard.is_some();

    let response = if use_speculative {
        // Use speculative decoding - clone the Arc to release the lock
        let spec_engine = Arc::clone(speculative_guard.as_ref().expect("speculative engine checked"));
        drop(speculative_guard);  // Release read lock

        let max_tokens = req.max_tokens.unwrap_or(256) as usize;

        // Use speculative engine for generation
        match spec_engine.generate(&req.prompt, max_tokens) {
            Ok(generated_text) => {
                let stats = spec_engine.stats();
                let prompt_tokens: u32 = (req.prompt.split_whitespace().count() as u32 * 4) / 3; // rough estimate
                let completion_tokens: u32 = (generated_text.split_whitespace().count() as u32 * 4) / 3;

                let completion_response = CompletionResponse {
                    id: request_id.clone(),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: spec_engine.model_id().to_string(),
                    choices: vec![CompletionChoice {
                        text: generated_text,
                        index: 0,
                        finish_reason: "stop".to_string(),
                        logprobs: None,
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };

                let latency_secs = start.elapsed().as_secs_f64();
                tracing::debug!(
                    request_id = %request_id,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms = start.elapsed().as_millis() as u64,
                    speculative_rounds = stats.rounds,
                    accepted_tokens = stats.accepted_tokens,
                    "Speculative completion finished"
                );

                // Record metrics
                state.metrics.record_completion_request(
                    prompt_tokens,
                    completion_tokens,
                    latency_secs,
                    &completion_response.model,
                );

                Json(completion_response).into_response()
            },
            Err(e) => {
                state.failed_requests.fetch_add(1, Ordering::Relaxed);
                let model_id = spec_engine.model_id();
                state.metrics.record_error("completion", model_id, "speculative_error");
                let error_msg = format!("{}", e);
                typed_error_with_message(
                    ErrorCode::InternalError,
                    &error_request_id,
                    &crate::error_response::sanitize_error(&error_msg),
                )
            },
        }
    } else {
        drop(speculative_guard);

        // Standard generation (fallback or when speculative not available)
        let engine = engine.expect("engine checked earlier");

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

                let latency_secs = start.elapsed().as_secs_f64();
                tracing::debug!(
                    request_id = %request_id,
                    prompt_tokens = response.usage.prompt_tokens,
                    completion_tokens = response.usage.completion_tokens,
                    latency_ms = start.elapsed().as_millis() as u64,
                    "Completion finished"
                );

                // Record metrics
                state.metrics.record_completion_request(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    latency_secs,
                    &completion_response.model,
                );

                Json(completion_response).into_response()
            },
            Err(e) => {
                state.failed_requests.fetch_add(1, Ordering::Relaxed);
                let model_id = engine.model_info().id.to_string();
                state.metrics.record_error("completion", &model_id, "generation_error");
                typed_error_with_message(
                    ErrorCode::InternalError,
                    &error_request_id,
                    &crate::error_response::sanitize_error(&e.to_string()),
                )
            },
        }
    };

    state.active_requests.fetch_sub(1, Ordering::Relaxed);
    drop(permit);

    response
}

/// Validates embedding request parameters.
#[inline]
fn validate_embedding_request(
    req: &EmbeddingRequest,
    limits: &ValidationLimits,
) -> std::result::Result<(), (StatusCode, &'static str)> {
    match &req.input {
        EmbeddingInput::Single(s) => {
            if s.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "input cannot be empty"));
            }
            if s.len() > limits.max_prompt_length {
                return Err((StatusCode::BAD_REQUEST, "input exceeds maximum length"));
            }
        }
        EmbeddingInput::Multiple(inputs) => {
            if inputs.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "input array cannot be empty"));
            }
            if inputs.len() > limits.max_embedding_inputs {
                return Err((StatusCode::BAD_REQUEST, "too many inputs in request"));
            }
            for input in inputs {
                if input.is_empty() {
                    return Err((StatusCode::BAD_REQUEST, "input cannot be empty"));
                }
                if input.len() > limits.max_prompt_length {
                    return Err((StatusCode::BAD_REQUEST, "input exceeds maximum length"));
                }
            }
        }
    }
    Ok(())
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = format!("emb-{}", uuid::Uuid::new_v4());
    let error_request_id = request_id.clone();

    tracing::debug!(request_id = %request_id, model = %req.model, "Embedding request");

    // Check if server is shutting down
    if state.is_shutting_down.load(Ordering::Relaxed) {
        return typed_error_with_message(
            ErrorCode::ServiceOverloaded,
            &request_id,
            "Server is shutting down",
        );
    }

    // Check queue capacity
    let current_queue = state.queue_depth.load(Ordering::Relaxed);
    if current_queue >= state.config.queue.max_queue_size {
        state.failed_requests.fetch_add(1, Ordering::Relaxed);
        return ApiError::new(ErrorCode::ServiceOverloaded, &request_id)
            .message("Server overloaded, please retry later")
            .retry_after(5)
            .build()
            .into_response();
    }

    // Increment queue depth
    state.queue_depth.fetch_add(1, Ordering::Relaxed);

    // Acquire semaphore permit
    let permit = match state.request_semaphore.acquire().await {
        Ok(permit) => permit,
        Err(_) => {
            state.queue_depth.fetch_sub(1, Ordering::Relaxed);
            state.failed_requests.fetch_add(1, Ordering::Relaxed);
            return typed_error_with_message(
                ErrorCode::ServiceOverloaded,
                &request_id,
                "Server is shutting down",
            );
        }
    };

    state.queue_depth.fetch_sub(1, Ordering::Relaxed);
    state.active_requests.fetch_add(1, Ordering::Relaxed);
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Validate request
    if let Err((_, message)) = validate_embedding_request(&req, &state.config.validation_limits) {
        state.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        let code = match message {
            m if m.contains("empty") => ErrorCode::InvalidEmbeddingInput,
            m if m.contains("exceeds maximum") => ErrorCode::InvalidEmbeddingInput,
            m if m.contains("too many inputs") => ErrorCode::InvalidEmbeddingInput,
            _ => ErrorCode::InvalidEmbeddingInput,
        };
        return typed_error_with_message(code, &request_id, message);
    }

    // Get engine
    let engine_guard = state.engine.read().await;
    let engine = match engine_guard.as_ref() {
        Some(engine) => Arc::clone(engine),
        None => {
            state.active_requests.fetch_sub(1, Ordering::Relaxed);
            drop(permit);
            return typed_error(ErrorCode::ModelNotLoaded, &error_request_id);
        },
    };
    drop(engine_guard);

    // Get input texts (borrow to avoid cloning the full collection)
    let texts: Vec<&str> = match &req.input {
        EmbeddingInput::Single(s) => vec![s.as_str()],
        EmbeddingInput::Multiple(v) => v.iter().map(String::as_str).collect(),
    };

    // Pre-allocate embeddings vector with known size
    let mut embeddings = Vec::with_capacity(texts.len());
    let mut total_tokens = 0u32;

    for (idx, text) in texts.iter().enumerate() {
        let embed_request = infernum_core::EmbedRequest::new((*text).to_string());

        match engine.embed(embed_request).await {
            Ok(response) => {
                // E3 FIX: Explicit error handling instead of silent failure with unwrap_or_default()
                let embedding_data = match response.data.first() {
                    Some(data) => data,
                    None => {
                        state.failed_requests.fetch_add(1, Ordering::Relaxed);
                        let model_id = engine.model_info().id.to_string();
                        state.metrics.record_error("embedding", &model_id, "embedding_extraction_failed");
                        state.active_requests.fetch_sub(1, Ordering::Relaxed);
                        drop(permit);
                        return typed_error_with_message(
                            ErrorCode::EmbeddingExtractionFailed,
                            &error_request_id,
                            "No embedding data returned from model",
                        );
                    }
                };

                let embedding_vec = match embedding_data.embedding.as_floats() {
                    Ok(floats) => floats,
                    Err(e) => {
                        tracing::error!(request_id = %error_request_id, error = %e, "Failed to extract embedding floats");
                        state.failed_requests.fetch_add(1, Ordering::Relaxed);
                        let model_id = engine.model_info().id.to_string();
                        state.metrics.record_error("embedding", &model_id, "embedding_extraction_failed");
                        state.active_requests.fetch_sub(1, Ordering::Relaxed);
                        drop(permit);
                        return typed_error_with_message(
                            ErrorCode::EmbeddingExtractionFailed,
                            &error_request_id,
                            "Failed to convert embedding to float array",
                        );
                    }
                };

                embeddings.push(EmbeddingData {
                    object: "embedding".to_string(),
                    index: idx as u32,
                    embedding: embedding_vec,
                });
                total_tokens += response.usage.total_tokens;
            },
            Err(e) => {
                state.failed_requests.fetch_add(1, Ordering::Relaxed);
                let model_id = engine.model_info().id.to_string();
                state.metrics.record_error("embedding", &model_id, "embedding_error");
                state.active_requests.fetch_sub(1, Ordering::Relaxed);
                drop(permit);
                return typed_error_with_message(
                    ErrorCode::InternalError,
                    &error_request_id,
                    &crate::error_response::sanitize_error(&e.to_string()),
                );
            },
        }
    }

    let batch_size = texts.len();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: embeddings,
        model: engine.model_info().id.to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    // Record metrics
    let latency_secs = start.elapsed().as_secs_f64();
    state.metrics.record_embedding_request(
        total_tokens,
        latency_secs,
        &response.model,
        batch_size,
    );

    state.active_requests.fetch_sub(1, Ordering::Relaxed);
    drop(permit);

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

        assert_eq!(config.addr, "127.0.0.1:3000".parse::<std::net::SocketAddr>().unwrap());
        assert!(!config.cors);
        assert_eq!(config.model, Some("test-model".to_string()));
        assert_eq!(config.queue.max_concurrent_requests, 32);
        // Default validation limits should be applied
        assert_eq!(config.validation_limits.max_messages, 256);
    }

    #[test]
    fn test_validation_limits_customization() {
        let custom_limits = ValidationLimits {
            max_messages: 100,
            max_message_length: 50_000,
            max_max_tokens: 16_384,
            max_prompt_length: 250_000,
            max_embedding_inputs: 128,
            max_body_size: 5 * 1024 * 1024, // 5MB
        };

        let config = ServerConfig::builder()
            .validation_limits(custom_limits)
            .build();

        assert_eq!(config.validation_limits.max_messages, 100);
        assert_eq!(config.validation_limits.max_message_length, 50_000);
        assert_eq!(config.validation_limits.max_max_tokens, 16_384);
        assert_eq!(config.validation_limits.max_prompt_length, 250_000);
        assert_eq!(config.validation_limits.max_embedding_inputs, 128);
        assert_eq!(config.validation_limits.max_body_size, 5 * 1024 * 1024);
    }

    #[test]
    fn test_error_response() {
        use crate::error_response::{ApiError, ErrorCode, ErrorType};

        let err = ApiError::new(ErrorCode::InvalidModel, "test-123")
            .message("Test error")
            .build();

        assert_eq!(err.error.message, "Test error");
        assert_eq!(err.error.error_type, ErrorType::InvalidRequestError);
        assert_eq!(err.error.code, ErrorCode::InvalidModel);
        assert_eq!(err.error.request_id, "test-123");
    }

    // === Model ID Validation Tests ===

    #[test]
    fn test_validate_model_id_valid() {
        assert!(validate_model_id("meta-llama/Llama-3.2-3B-Instruct").is_ok());
        assert!(validate_model_id("gpt-4").is_ok());
        assert!(validate_model_id("my_model_v1").is_ok());
        assert!(validate_model_id("model-with-dashes").is_ok());
        assert!(validate_model_id("Org/Model-Name").is_ok());
    }

    #[test]
    fn test_validate_model_id_empty() {
        let result = validate_model_id("");
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_validate_model_id_too_long() {
        let long_id = "a".repeat(257);
        let result = validate_model_id(&long_id);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("too long"));
    }

    #[test]
    fn test_validate_model_id_path_traversal() {
        // Test path traversal attempts
        assert!(validate_model_id("../etc/passwd").is_err());
        assert!(validate_model_id("model/../secret").is_err());
        assert!(validate_model_id("..").is_err());
        assert!(validate_model_id("foo/..").is_err());
    }

    #[test]
    fn test_validate_model_id_absolute_path() {
        assert!(validate_model_id("/etc/passwd").is_err());
        assert!(validate_model_id("/home/user/model").is_err());
        assert!(validate_model_id("\\Windows\\System32").is_err());
    }

    #[test]
    fn test_validate_model_id_shell_metacharacters() {
        // Test shell injection prevention
        assert!(validate_model_id("model; rm -rf /").is_err());
        assert!(validate_model_id("model | cat /etc/passwd").is_err());
        assert!(validate_model_id("model & echo pwned").is_err());
        assert!(validate_model_id("model$(whoami)").is_err());
        assert!(validate_model_id("model`id`").is_err());
        assert!(validate_model_id("model > /tmp/out").is_err());
        assert!(validate_model_id("model < /etc/shadow").is_err());
        assert!(validate_model_id("model\necho evil").is_err());
        assert!(validate_model_id("model\r\necho evil").is_err());
        assert!(validate_model_id("model\0evil").is_err());
    }

    // === Chat Request Validation Tests ===

    fn make_chat_message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn make_chat_request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }

    #[test]
    fn test_validate_chat_request_valid() {
        let limits = ValidationLimits::default();
        let req = make_chat_request(vec![
            make_chat_message("user", "Hello, world!"),
        ]);
        assert!(validate_chat_request(&req, &limits).is_ok());
    }

    #[test]
    fn test_validate_chat_request_empty_messages() {
        use crate::validation::RequestValidationError;
        let limits = ValidationLimits::default();
        let req = make_chat_request(vec![]);
        let result = validate_chat_request(&req, &limits);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, RequestValidationError::EmptyMessages));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_validate_chat_request_too_many_messages() {
        use crate::validation::RequestValidationError;
        let limits = ValidationLimits {
            max_messages: 5,
            ..Default::default()
        };
        let messages: Vec<_> = (0..10)
            .map(|i| make_chat_message("user", &format!("Message {}", i)))
            .collect();
        let req = make_chat_request(messages);
        let result = validate_chat_request(&req, &limits);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, RequestValidationError::TooManyMessages { count: 10, limit: 5 }));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_validate_chat_request_message_too_long() {
        use crate::validation::RequestValidationError;
        let limits = ValidationLimits {
            max_message_length: 100,
            ..Default::default()
        };
        let long_content = "x".repeat(150);
        let req = make_chat_request(vec![make_chat_message("user", &long_content)]);
        let result = validate_chat_request(&req, &limits);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, RequestValidationError::MessageTooLong { index: 0, length: 150, limit: 100 }));
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_validate_chat_request_temperature() {
        let limits = ValidationLimits::default();

        // Valid temperatures
        let mut req = make_chat_request(vec![make_chat_message("user", "test")]);
        req.temperature = Some(0.0);
        assert!(validate_chat_request(&req, &limits).is_ok());

        req.temperature = Some(1.0);
        assert!(validate_chat_request(&req, &limits).is_ok());

        req.temperature = Some(2.0);
        assert!(validate_chat_request(&req, &limits).is_ok());

        // Invalid temperatures
        req.temperature = Some(-0.1);
        assert!(validate_chat_request(&req, &limits).is_err());

        req.temperature = Some(2.1);
        assert!(validate_chat_request(&req, &limits).is_err());
    }

    #[test]
    fn test_validate_chat_request_top_p() {
        let limits = ValidationLimits::default();
        let mut req = make_chat_request(vec![make_chat_message("user", "test")]);

        // Valid top_p
        req.top_p = Some(0.0);
        assert!(validate_chat_request(&req, &limits).is_ok());

        req.top_p = Some(0.5);
        assert!(validate_chat_request(&req, &limits).is_ok());

        req.top_p = Some(1.0);
        assert!(validate_chat_request(&req, &limits).is_ok());

        // Invalid top_p
        req.top_p = Some(-0.1);
        assert!(validate_chat_request(&req, &limits).is_err());

        req.top_p = Some(1.1);
        assert!(validate_chat_request(&req, &limits).is_err());
    }

    #[test]
    fn test_validate_chat_request_max_tokens() {
        let limits = ValidationLimits {
            max_max_tokens: 1000,
            ..Default::default()
        };
        let mut req = make_chat_request(vec![make_chat_message("user", "test")]);

        // Valid max_tokens
        req.max_tokens = Some(100);
        assert!(validate_chat_request(&req, &limits).is_ok());

        req.max_tokens = Some(1000);
        assert!(validate_chat_request(&req, &limits).is_ok());

        // Invalid max_tokens
        req.max_tokens = Some(0);
        assert!(validate_chat_request(&req, &limits).is_err());

        req.max_tokens = Some(1001);
        assert!(validate_chat_request(&req, &limits).is_err());
    }

    // === Completion Request Validation Tests ===

    fn make_completion_request(prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: "test-model".to_string(),
            prompt: prompt.to_string(),
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            logprobs: None,
            echo: None,
            suffix: None,
            presence_penalty: None,
            frequency_penalty: None,
        }
    }

    #[test]
    fn test_validate_completion_request_valid() {
        let limits = ValidationLimits::default();
        let req = make_completion_request("Hello, world!");
        assert!(validate_completion_request(&req, &limits).is_ok());
    }

    #[test]
    fn test_validate_completion_request_empty_prompt() {
        let limits = ValidationLimits::default();
        let req = make_completion_request("");
        let result = validate_completion_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_validate_completion_request_prompt_too_long() {
        let limits = ValidationLimits {
            max_prompt_length: 100,
            ..Default::default()
        };
        let long_prompt = "x".repeat(150);
        let req = make_completion_request(&long_prompt);
        let result = validate_completion_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("maximum length"));
    }

    #[test]
    fn test_validate_completion_request_temperature() {
        let limits = ValidationLimits::default();

        let mut req = make_completion_request("test");
        req.temperature = Some(1.5);
        assert!(validate_completion_request(&req, &limits).is_ok());

        req.temperature = Some(-0.5);
        assert!(validate_completion_request(&req, &limits).is_err());

        req.temperature = Some(2.5);
        assert!(validate_completion_request(&req, &limits).is_err());
    }

    #[test]
    fn test_validate_completion_request_top_p() {
        let limits = ValidationLimits::default();

        let mut req = make_completion_request("test");
        req.top_p = Some(0.9);
        assert!(validate_completion_request(&req, &limits).is_ok());

        req.top_p = Some(-0.1);
        assert!(validate_completion_request(&req, &limits).is_err());

        req.top_p = Some(1.5);
        assert!(validate_completion_request(&req, &limits).is_err());
    }

    #[test]
    fn test_validate_completion_request_max_tokens() {
        let limits = ValidationLimits {
            max_max_tokens: 500,
            ..Default::default()
        };

        let mut req = make_completion_request("test");
        req.max_tokens = Some(250);
        assert!(validate_completion_request(&req, &limits).is_ok());

        req.max_tokens = Some(0);
        assert!(validate_completion_request(&req, &limits).is_err());

        req.max_tokens = Some(501);
        assert!(validate_completion_request(&req, &limits).is_err());
    }

    // === Embedding Request Validation Tests ===

    fn make_embedding_request_single(input: &str) -> EmbeddingRequest {
        EmbeddingRequest {
            model: "test-model".to_string(),
            input: EmbeddingInput::Single(input.to_string()),
            encoding_format: None,
            dimensions: None,
        }
    }

    fn make_embedding_request_multiple(inputs: Vec<&str>) -> EmbeddingRequest {
        EmbeddingRequest {
            model: "test-model".to_string(),
            input: EmbeddingInput::Multiple(inputs.into_iter().map(String::from).collect()),
            encoding_format: None,
            dimensions: None,
        }
    }

    #[test]
    fn test_validate_embedding_request_valid_single() {
        let limits = ValidationLimits::default();
        let req = make_embedding_request_single("Hello, world!");
        assert!(validate_embedding_request(&req, &limits).is_ok());
    }

    #[test]
    fn test_validate_embedding_request_valid_multiple() {
        let limits = ValidationLimits::default();
        let req = make_embedding_request_multiple(vec!["Hello", "World", "Test"]);
        assert!(validate_embedding_request(&req, &limits).is_ok());
    }

    #[test]
    fn test_validate_embedding_request_empty_single() {
        let limits = ValidationLimits::default();
        let req = make_embedding_request_single("");
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_validate_embedding_request_empty_multiple() {
        let limits = ValidationLimits::default();
        let req = make_embedding_request_multiple(vec![]);
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_validate_embedding_request_empty_item_in_multiple() {
        let limits = ValidationLimits::default();
        let req = make_embedding_request_multiple(vec!["Hello", "", "World"]);
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_validate_embedding_request_single_too_long() {
        let limits = ValidationLimits {
            max_prompt_length: 50,
            ..Default::default()
        };
        let long_input = "x".repeat(100);
        let req = make_embedding_request_single(&long_input);
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("maximum length"));
    }

    #[test]
    fn test_validate_embedding_request_too_many_inputs() {
        let limits = ValidationLimits {
            max_embedding_inputs: 3,
            ..Default::default()
        };
        let req = make_embedding_request_multiple(vec!["a", "b", "c", "d", "e"]);
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("too many"));
    }

    #[test]
    fn test_validate_embedding_request_item_too_long_in_multiple() {
        let limits = ValidationLimits {
            max_prompt_length: 50,
            ..Default::default()
        };
        let long_input = "x".repeat(100);
        let req = make_embedding_request_multiple(vec!["short", &long_input]);
        let result = validate_embedding_request(&req, &limits);
        assert!(result.is_err());
        let (status, msg) = result.err().unwrap();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("maximum length"));
    }

    // === Default Values Tests ===

    #[test]
    fn test_validation_limits_default() {
        let limits = ValidationLimits::default();
        assert_eq!(limits.max_messages, 256);
        assert_eq!(limits.max_message_length, 100_000);
        assert_eq!(limits.max_max_tokens, 32_768);
        assert_eq!(limits.max_prompt_length, 500_000);
        assert_eq!(limits.max_embedding_inputs, 256);
        assert_eq!(limits.max_body_size, 10 * 1024 * 1024); // 10MB default
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.addr, SocketAddr::from(DEFAULT_ADDR));
        assert!(config.cors);
        assert!(config.model.is_none());
        assert_eq!(config.queue.max_concurrent_requests, 64);
    }

    // === Batch Scheduler Unit Tests ===

    #[test]
    fn test_app_state_has_batch_scheduler() {
        let config = ServerConfig::default();
        let state = AppState::new(config);

        // AppState should have a batch scheduler
        let scheduler = state.batch_scheduler();
        assert!(scheduler.pending_count() == 0, "New scheduler should be empty");
    }

    #[test]
    fn test_batch_scheduler_config_from_server_config() {
        let config = ServerConfig::builder()
            .max_concurrent_requests(32)
            .max_queue_size(128)
            .build();
        let state = AppState::new(config);

        let scheduler = state.batch_scheduler();
        // Scheduler config should match server's queue config
        assert_eq!(scheduler.config().max_queue_size, 128);
    }

    #[test]
    fn test_batch_scheduler_inherits_queue_limits() {
        // Verify the scheduler respects the server's queue configuration
        let config = ServerConfig::builder()
            .max_queue_size(500)
            .build();
        let state = AppState::new(config);

        let scheduler = state.batch_scheduler();
        let batch_config = scheduler.config();

        // The scheduler's queue size should match server config
        assert_eq!(batch_config.max_queue_size, 500);
    }

    // === Integration Tests ===
    // Note: These tests require tokio runtime compatibility fixes with axum-test.
    // Run with: cargo test -p infernum-server --features integration-tests -- integration

    #[cfg(feature = "integration-tests")]
    mod integration {
        use super::*;
        use axum::http::StatusCode;
        use axum_test::TestServer;

        /// Creates a test server without an engine (for health checks, etc.)
        async fn create_test_server() -> TestServer {
            let config = ServerConfig::default();
            let server = Server::new(config);
            let router = server.router();
            TestServer::new(router).expect("Failed to create test server")
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_health_endpoint() {
            let server = create_test_server().await;
            let response = server.get("/health").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            let body: serde_json::Value = response.json();
            assert_eq!(body["status"], "healthy");
        }

        #[tokio::test]
        async fn test_ready_endpoint_without_model() {
            let server = create_test_server().await;
            let response = server.get("/ready").await;

            // Should return 503 when no model is loaded
            assert_eq!(response.status_code(), StatusCode::SERVICE_UNAVAILABLE);

            let body: serde_json::Value = response.json();
            assert_eq!(body["status"], "not_ready");
        }

        #[tokio::test]
        async fn test_models_endpoint_without_model() {
            let server = create_test_server().await;
            let response = server.get("/v1/models").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            let body: serde_json::Value = response.json();
            assert_eq!(body["object"], "list");
            assert!(body["data"].as_array().expect("data array").is_empty());
        }

        #[tokio::test]
        async fn test_status_endpoint() {
            let server = create_test_server().await;
            let response = server.get("/api/status").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            let body: serde_json::Value = response.json();
            assert_eq!(body["model_loaded"], false);
            assert!(body["uptime_seconds"].as_f64().is_some());
            // Day 18.5: Status should include version
            assert!(body["version"].is_string(), "status should include version");
            assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
        }

        #[tokio::test]
        async fn test_chat_completions_without_model() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ]
            });

            let response = server
                .post("/v1/chat/completions")
                .json(&request)
                .await;

            // Should fail because no model is loaded
            assert_eq!(response.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        }

        #[tokio::test]
        async fn test_completions_without_model() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "prompt": "Hello!"
            });

            let response = server
                .post("/v1/completions")
                .json(&request)
                .await;

            // Should fail because no model is loaded
            assert_eq!(response.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        }

        #[tokio::test]
        async fn test_embeddings_without_model() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "input": "Hello, world!"
            });

            let response = server
                .post("/v1/embeddings")
                .json(&request)
                .await;

            // Should fail because no model is loaded
            assert_eq!(response.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        }

        #[tokio::test]
        async fn test_chat_validation_empty_messages() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "messages": []
            });

            let response = server
                .post("/v1/chat/completions")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

            let body: serde_json::Value = response.json();
            assert!(body["error"]["message"].as_str()
                .expect("error message")
                .contains("at least one message"));
        }

        #[tokio::test]
        async fn test_chat_validation_invalid_temperature() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 3.0
            });

            let response = server
                .post("/v1/chat/completions")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

            let body: serde_json::Value = response.json();
            assert!(body["error"]["message"].as_str()
                .expect("error message")
                .contains("temperature"));
        }

        #[tokio::test]
        async fn test_completion_validation_empty_prompt() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "prompt": ""
            });

            let response = server
                .post("/v1/completions")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

            let body: serde_json::Value = response.json();
            assert!(body["error"]["message"].as_str()
                .expect("error message")
                .contains("empty"));
        }

        #[tokio::test]
        async fn test_embedding_validation_empty_input() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "test-model",
                "input": ""
            });

            let response = server
                .post("/v1/embeddings")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);
        }

        #[tokio::test]
        async fn test_model_id_validation_path_traversal() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "../etc/passwd"
            });

            let response = server
                .post("/api/models/load")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

            let body: serde_json::Value = response.json();
            assert!(body["error"]["message"].as_str()
                .expect("error message")
                .contains("path traversal"));
        }

        #[tokio::test]
        async fn test_model_id_validation_shell_injection() {
            let server = create_test_server().await;

            let request = serde_json::json!({
                "model": "model; rm -rf /"
            });

            let response = server
                .post("/api/models/load")
                .json(&request)
                .await;

            assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

            let body: serde_json::Value = response.json();
            assert!(body["error"]["message"].as_str()
                .expect("error message")
                .contains("shell"));
        }

        #[tokio::test]
        async fn test_metrics_endpoint() {
            let server = create_test_server().await;
            let response = server.get("/metrics").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            let body = response.text();
            // Day 16.1: Verify Prometheus text exposition format
            // Check for required metric types
            assert!(body.contains("infernum_queue_depth"), "should have queue_depth gauge");
            assert!(body.contains("infernum_active_requests_total"), "should have active_requests gauge");
            assert!(body.contains("infernum_total_requests_served"), "should have total_requests counter");
            assert!(body.contains("infernum_uptime_seconds"), "should have uptime gauge");
            // Check for proper Prometheus format markers
            assert!(body.contains("# HELP"), "should have HELP comments");
            assert!(body.contains("# TYPE"), "should have TYPE declarations");
        }

        #[tokio::test]
        async fn test_metrics_prometheus_format() {
            let server = create_test_server().await;
            let response = server.get("/metrics").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            // Verify Content-Type header for Prometheus
            let content_type = response.header("content-type");
            assert!(content_type.contains("text/plain"), "should have text/plain content type");

            let body = response.text();
            // Verify each line is valid Prometheus format:
            // Either empty, a comment (#), or a metric (name{labels} value)
            for line in body.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let valid = trimmed.starts_with('#')  // Comment line
                    || trimmed.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_'); // Metric line
                assert!(valid, "invalid Prometheus line: {}", trimmed);
            }
        }

        #[tokio::test]
        async fn test_unknown_endpoint_404() {
            let server = create_test_server().await;
            let response = server.get("/unknown/path").await;

            assert_eq!(response.status_code(), StatusCode::NOT_FOUND);
        }

        #[tokio::test]
        async fn test_batch_metrics_endpoint() {
            let server = create_test_server().await;
            let response = server.get("/v1/batching/stats").await;

            assert_eq!(response.status_code(), StatusCode::OK);

            let body: serde_json::Value = response.json();
            assert!(body["pending_requests"].is_number());
            assert!(body["active_batch_size"].is_number());
            assert!(body["total_batches_formed"].is_number());
        }
    }
}
