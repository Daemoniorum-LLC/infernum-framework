//! # Infernum Server
//!
//! HTTP API server for local LLM inference.
//!
//! ## Features
//!
//! - **Chat Completions**: `/v1/chat/completions` with streaming support
//! - **Text Completions**: `/v1/completions` for raw text generation
//! - **Embeddings**: `/v1/embeddings` for vector generation
//! - **Model Management**: Load/unload models at runtime via API
//! - **Health Checks**: `/health` and `/ready` endpoints for orchestration
//! - **Prometheus Metrics**: `/metrics` endpoint for observability
//! - **Authentication**: API key authentication with scopes and permission levels
//! - **Rate Limiting**: Per-IP and per-API-key rate limiting
//! - **Security Headers**: CSP, X-Frame-Options, HSTS, and more
//! - **Request Validation**: Input validation with configurable limits
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use infernum_server::{Server, ServerConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ServerConfig::builder()
//!         .addr("0.0.0.0:8080".parse()?)
//!         .model("meta-llama/Llama-3.2-3B-Instruct")
//!         .build();
//!
//!     let server = Server::new(config);
//!     server.run().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`auth`]: Authentication and scope-based authorization
//! - [`api_types`]: API request/response wire types
//! - [`security`]: Rate limiting, security headers, and CORS
//! - [`server`]: HTTP server configuration and handlers
//!
//! ## Authentication
//!
//! The server supports API key authentication with scope-based access control:
//!
//! ```rust,ignore
//! use infernum_server::auth::{AuthConfig, ApiKey, Scope};
//!
//! let config = AuthConfig::enabled()
//!     .add_key(ApiKey::admin("sk-adm-admin123"))
//!     .add_key(ApiKey::with_scopes("sk-inf-user456", vec![Scope::Inference]));
//! ```
//!
//! API key format: `sk-{scope}-{random}` where scope is `inf`, `adm`, or `met`.
//!
//! ## Request Validation
//!
//! Configure request limits to prevent abuse:
//!
//! ```rust,ignore
//! use infernum_server::{ServerConfig, ValidationLimits};
//!
//! let limits = ValidationLimits {
//!     max_messages: 256,
//!     max_message_length: 100_000,
//!     max_max_tokens: 32_768,
//!     max_body_size: 10 * 1024 * 1024, // 10MB
//!     ..Default::default()
//! };
//!
//! let config = ServerConfig::builder()
//!     .validation_limits(limits)
//!     .build();
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod admin;
pub mod agent_identity;
pub mod agentic;
pub mod api;
pub mod api_types;
pub mod audit;
pub mod audit_client;
pub mod auth;
pub mod batching;
pub mod cache;
pub mod cancellation;
pub mod circuit_breaker;
pub mod config;
pub mod config_error;
pub mod config_reload;
pub mod dedup;
pub mod error_response;
pub mod gpu_metrics;
pub mod grpc;
pub mod handlers;
pub mod model_cache;
pub mod observability;
pub mod openapi;
pub mod priority;
pub mod queue;
pub mod rag;
pub mod request_batcher;
pub mod responses;
pub mod security;
pub mod server;
pub mod server_error;
pub mod sessions;
pub mod speculative;
pub mod speculative_engine;
pub mod structured;
pub mod timeout;
pub mod tls;
pub mod tokenize;
pub mod tool_use;
pub mod tracing_otel;
pub mod validation;
pub mod vision;
pub mod websocket;
pub mod wellbeing_intervention;

pub use admin::{
    AdminError, AdminModelInfo, LoadModelRequest, LoadModelResponse, ModelLoadOptions,
    ModelRegistry, ModelStatus, ModelsStatusResponse, UnloadModelRequest, UnloadModelResponse,
    WarmupModelRequest, WarmupModelResponse,
};
pub use agent_identity::{
    AgentIdentity, AgentIdentityError, AgentIdentityExport, EncryptedIdentity,
    EncryptedIdentityStore,
};
pub use agentic::{
    inspect_agent, override_subtask, run_agent, run_supervisor, AgentInspectionResponse,
    AgenticRunError, AgenticRunRequest, SubtaskAction, SubtaskOverrideRequest,
    SubtaskOverrideResponse, SubtaskSpec, SuperviseRequest,
};
pub use api_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatLogProbs, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, FunctionCall, FunctionDefinition, ModelObject,
    ModelsResponse, TokenLogProb, Tool, ToolCall, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, TopLogProb, Usage,
};
pub use audit::{AuditConfig, AuditEvent, AuditEventType, AuditLogger};
pub use audit_client::{
    generate_encryption_keypair,
    policies as encryption_policies,
    AuditClient,
    AuditClientConfig,
    AuditClientError,
    // HoloCrypt re-exports
    EncryptedEvent,
    EncryptedEventBuilder,
    EncryptionPolicy,
    EventOpeningKey,
    EventSealingKey,
    FieldVisibility,
    SubmitEventRequest,
    SubmitEventResponse,
};
pub use auth::{required_scope_for_path, ApiKey, AuthConfig, AuthState, Permission, Scope};
pub use batching::{
    ActiveBatch, BatchConfig, BatchEntry, BatchError, BatchId, BatchPriority, BatchScheduler,
    BatchState, BatchStats, FinishReason, IterationConfig, IterationMetrics, IterationResult,
    IterationStep, PendingRequest, PreemptionPolicy, RequestState, SamplingParams,
    SchedulerMetrics, SchedulerState, SchedulerStats, SchedulingPolicy, Sequence, SequenceGroup,
    SequenceId, SequenceState, TokenIterator,
};
pub use cache::{
    CacheConfig, CacheKey, CacheMetrics, CacheResult, CachedResponse, ResponseCache, CACHE_HEADER,
};
pub use cancellation::{
    CancellationError, CancellationMetrics, CancellationReason, CancellationToken,
    RequestCancellation,
};
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerMetrics, CircuitOpenError, CircuitState,
};
pub use config::{Config, ConfigBuilder};
pub use config_error::ConfigError;
pub use config_reload::{ConfigChange, ConfigWatcher, ReloadResult, ReloadableConfig};
pub use dedup::{
    ComputeHandle, DeduplicatedResult, DeduplicatorConfig, DeduplicatorMetrics,
    RequestDeduplicator, RequestHash,
};
pub use error_response::{
    api_error, api_error_with_message, handle_internal_error, sanitize_error, ApiError,
    ApiErrorBuilder, ErrorCode, ErrorDetail, ErrorSubcode, ErrorType, RetryInfo,
};
pub use gpu_metrics::{GpuInfo, GpuMetrics, GpuMetricsProvider, MockGpuMetrics, NoGpuMetrics};
pub use grpc::{
    ChatChoice as GrpcChatChoice, ChatChoiceDelta, ChatCompletionChunk,
    ChatCompletionRequest as GrpcChatCompletionRequest,
    ChatCompletionResponse as GrpcChatCompletionResponse, ChatMessage as GrpcChatMessage,
    ChatMessageDelta, CompletionChoice as GrpcCompletionChoice, CompletionChoiceDelta,
    CompletionChunk, CompletionRequest as GrpcCompletionRequest,
    CompletionResponse as GrpcCompletionResponse, ComponentHealth, EmbedRequest, EmbedResponse,
    Embedding, GrpcConfig, GrpcError, GrpcMetrics, GrpcPriority, HealthCheckRequest,
    HealthCheckResponse, InfernumService, ListModelsRequest, ListModelsResponse,
    MockInfernumService, Model as GrpcModel, Role, Usage as GrpcUsage,
};
pub use model_cache::{
    convert_model, delete_cached_model, download_model, list_cached_models, CacheSource,
    CachedModel, CachedModelsResponse, ConvertModelMetadata, ConvertModelRequest,
    ConvertModelResponse, DeleteCachedModelRequest, DeleteCachedModelResponse,
    DownloadModelRequest, DownloadProgress, ModelCacheState,
};
pub use observability::{ObservabilityState, RequestId};
pub use openapi::ApiDoc;
pub use priority::{RequestPriority, PRIORITY_HEADER};
pub use queue::{
    PeekResult, QueueConfig, QueueError, QueueMetrics, QueueState, QueueStats, QueuedRequest,
    RequestQueue, NUM_PRIORITY_LEVELS,
};
pub use rag::{
    delete_document, document_count, index_document, list_documents, rag_health, search,
    DeleteResponse, DocumentCountResponse, DocumentListResponse, DocumentMeta,
    IndexDocumentRequest, RagHealthResponse, RagState, SearchRequest, SearchResponse,
    SearchResultItem,
};
pub use request_batcher::{BatcherConfig, BatcherHandle, BatcherStats, RequestBatcher};
pub use responses::{
    ApiResponse, HealthResponse, MetricsResponse, ModelInfo, ReadyResponse, ResponseMeta,
};
pub use security::{CorsConfig, RateLimitConfig, RateLimiter, SecurityHeadersConfig};
pub use server::{AppState, Server, ServerConfig, ServerConfigBuilder, ValidationLimits};
pub use server_error::ServerError;
pub use sessions::{
    cancel_session, get_session, list_sessions, session_stream, sessions_stream, AgentEventData,
    AgentSession, CancelResponse, EventCounts, GetSessionResponse, ListSessionsResponse,
    SessionEvent, SessionRegistry, SessionStatus,
};
pub use speculative::{
    DraftToken, SpeculativeConfig, SpeculativeError, SpeculativeMetrics, SpeculativeMode,
    SpeculativeParams, SpeculativeRequest, SpeculativeScheduler, SpeculativeState,
    SpeculativeStats, VerificationResult, SPECULATIVE_DRAFT_HEADER, SPECULATIVE_HEADER,
    SPECULATIVE_TOKENS_HEADER,
};
pub use speculative_engine::{
    SpeculativeEngine, SpeculativeEngineBuilder, SpeculativeEngineConfig, SpeculativeEngineError,
};
pub use structured::{
    validate_json, validate_json_string, JsonSchema, ResponseFormat, SchemaRegistry,
    ValidationError, ValidationResult,
};
pub use timeout::{RequestTimeout, TimeoutConfig, TimeoutMetrics, TIMEOUT_HEADER};
pub use tokenize::{
    count_tokens, EstimatingTokenizer, TokenizeError, TokenizeRequest, TokenizeResponse, Tokenizer,
};
pub use tool_use::{
    // Phase 3: Streaming detection (low-level)
    buffer_might_contain_tool_start,
    definitely_not_tool_call,
    // Core functions
    detect_tool_calls,
    // Phase 3: Parallel tool calls
    enforce_parallel_tool_calls,
    // Phase 3: Deep JSON parsing
    extract_json_object,
    extract_text_content,
    format_tools_for_prompt,
    get_forced_tool,
    process_model_output,
    process_model_output_with_options,
    process_model_output_with_validation,
    should_include_tools,
    try_extract_complete_tool_call,
    // Phase 3: Unknown tool validation
    validate_detected_calls,
    // Phase 3: Strict mode validation
    validate_tool_arguments,
    validate_tool_exists,
    DetectedCallsValidation,
    // Core types
    DetectedToolCall,
    ModelFamily,
    ProcessingOptions,
    // Phase 3: Agent-centric SSE events
    SseEvent,
    SseUsage,
    StreamingExtractResult,
    // Phase 3: Streaming detector (high-level stateful)
    StreamingToolDetector,
    ToolDetectionEvent,
    ToolProcessingResult,
    ToolValidationResult,
};
pub use tracing_otel::{
    init_tracing, otel_tracing_middleware, shutdown_tracing, InferenceSpan, TracingConfig,
    TracingError,
};
pub use validation::{
    validate_chat_request, validate_completion_request, validate_embedding_request,
    validate_model_id, RequestValidationError,
};
pub use vision::{
    is_supported_media_type, ContentPart, ImageBase64, ImageDetail, ImageUrl, MessageContent,
    VisionConfig, VisionError, VisionMetrics, SUPPORTED_MEDIA_TYPES,
};
pub use websocket::{
    ClientMessage, CloseReason, ConnectionInfo, ConnectionManager, ConnectionState, ServerMessage,
    UsageInfo, WsConfig, WsError, WsMetrics,
};
pub use wellbeing_intervention::{
    create_intervention_controller, InterventionConfig, InterventionController, InterventionError,
    InterventionMetrics, SharedInterventionController, WellbeingState,
};
