//! gRPC API for Infernum.
//!
//! This module provides a gRPC interface for high-performance internal services,
//! complementing the REST API for external clients.
//!
//! # Architecture
//!
//! The gRPC service reuses types from the REST API where possible, providing
//! a consistent interface across both protocols.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    InfernumGrpcServer                       │
//! │  ┌──────────────────────────────────────────────────────┐  │
//! │  │  Services                                            │  │
//! │  │  ┌────────────────┐  ┌─────────────────────────────┐ │  │
//! │  │  │ InfernumService│  │    ModelService (Admin)     │ │  │
//! │  │  │  - ChatComplete│  │    - LoadModel              │ │  │
//! │  │  │  - Complete    │  │    - UnloadModel            │ │  │
//! │  │  │  - Embed       │  │    - GetStatus              │ │  │
//! │  │  │  - ListModels  │  │                             │ │  │
//! │  │  └────────────────┘  └─────────────────────────────┘ │  │
//! │  └──────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::grpc::{GrpcServer, GrpcConfig};
//!
//! let config = GrpcConfig::default();
//! let server = GrpcServer::new(config);
//! server.serve("[::1]:50051".parse()?).await?;
//! ```

use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::Status;

use crate::batching::BatchPriority;

/// Configuration for the gRPC server.
#[derive(Debug, Clone)]
pub struct GrpcConfig {
    /// Maximum message size in bytes.
    pub max_message_size: usize,

    /// Connection timeout.
    pub connection_timeout: Duration,

    /// Request timeout.
    pub request_timeout: Duration,

    /// Enable reflection service.
    pub enable_reflection: bool,

    /// Enable health check service.
    pub enable_health: bool,

    /// Maximum concurrent streams per connection.
    pub max_concurrent_streams: u32,

    /// Initial connection window size.
    pub initial_connection_window_size: u32,

    /// Initial stream window size.
    pub initial_stream_window_size: u32,

    /// Keep-alive interval.
    pub keepalive_interval: Duration,

    /// Keep-alive timeout.
    pub keepalive_timeout: Duration,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            max_message_size: 16 * 1024 * 1024, // 16MB
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(300),
            enable_reflection: true,
            enable_health: true,
            max_concurrent_streams: 200,
            initial_connection_window_size: 1024 * 1024, // 1MB
            initial_stream_window_size: 512 * 1024,      // 512KB
            keepalive_interval: Duration::from_secs(60),
            keepalive_timeout: Duration::from_secs(20),
        }
    }
}

impl GrpcConfig {
    /// Creates a new gRPC configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method for max message size.
    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    /// Builder method for request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Builder method for reflection.
    pub fn with_reflection(mut self, enabled: bool) -> Self {
        self.enable_reflection = enabled;
        self
    }

    /// Builder method for health check.
    pub fn with_health(mut self, enabled: bool) -> Self {
        self.enable_health = enabled;
        self
    }

    /// Builder method for max concurrent streams.
    pub fn with_max_concurrent_streams(mut self, max: u32) -> Self {
        self.max_concurrent_streams = max;
        self
    }
}

// ============================================================================
// Request/Response Types (matching proto definitions)
// ============================================================================

/// Priority level for gRPC requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum GrpcPriority {
    /// Unspecified priority.
    Unspecified = 0,
    /// Background priority.
    Background = 1,
    /// Normal priority.
    #[default]
    Normal = 2,
    /// High priority.
    High = 3,
    /// Critical priority.
    Critical = 4,
}

impl From<GrpcPriority> for BatchPriority {
    fn from(p: GrpcPriority) -> Self {
        match p {
            GrpcPriority::Unspecified | GrpcPriority::Normal => BatchPriority::Normal,
            GrpcPriority::Background => BatchPriority::Background,
            GrpcPriority::High => BatchPriority::High,
            GrpcPriority::Critical => BatchPriority::Critical,
        }
    }
}

impl From<i32> for GrpcPriority {
    fn from(v: i32) -> Self {
        match v {
            1 => Self::Background,
            2 => Self::Normal,
            3 => Self::High,
            4 => Self::Critical,
            _ => Self::Unspecified,
        }
    }
}

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Role {
    /// Unspecified role.
    #[default]
    Unspecified = 0,
    /// System message.
    System = 1,
    /// User message.
    User = 2,
    /// Assistant message.
    Assistant = 3,
    /// Tool response.
    Tool = 4,
}

impl From<i32> for Role {
    fn from(v: i32) -> Self {
        match v {
            1 => Self::System,
            2 => Self::User,
            3 => Self::Assistant,
            4 => Self::Tool,
            _ => Self::Unspecified,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unspecified => write!(f, "unspecified"),
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// Token usage information.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Number of prompt tokens.
    pub prompt_tokens: i32,
    /// Number of completion tokens.
    pub completion_tokens: i32,
    /// Total tokens.
    pub total_tokens: i32,
}

impl Usage {
    /// Creates new usage info.
    pub fn new(prompt: i32, completion: i32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// A message in the chat conversation.
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    /// Role of the message.
    pub role: Role,
    /// Content of the message.
    pub content: String,
    /// Optional name.
    pub name: Option<String>,
    /// Tool call ID (for tool responses).
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Creates an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            ..Default::default()
        }
    }
}

/// Chat completion request.
#[derive(Debug, Clone, Default)]
pub struct ChatCompletionRequest {
    /// Model ID.
    pub model: String,
    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Top-p sampling.
    pub top_p: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<i32>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Whether to return logprobs.
    pub logprobs: Option<bool>,
    /// Number of top logprobs to return.
    pub top_logprobs: Option<i32>,
    /// Number of completions to generate.
    pub n: Option<i32>,
    /// Random seed.
    pub seed: Option<i64>,
    /// Request priority.
    pub priority: GrpcPriority,
    /// Request ID for tracing.
    pub request_id: Option<String>,
}

/// Chat completion choice.
#[derive(Debug, Clone, Default)]
pub struct ChatChoice {
    /// Index of the choice.
    pub index: i32,
    /// The generated message.
    pub message: ChatMessage,
    /// Finish reason.
    pub finish_reason: String,
}

/// Chat completion response.
#[derive(Debug, Clone, Default)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage.
    pub usage: Usage,
}

/// Streaming chat completion chunk.
#[derive(Debug, Clone, Default)]
pub struct ChatCompletionChunk {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Delta choices.
    pub choices: Vec<ChatChoiceDelta>,
}

/// Delta for streaming chat.
#[derive(Debug, Clone, Default)]
pub struct ChatChoiceDelta {
    /// Index of the choice.
    pub index: i32,
    /// Delta content.
    pub delta: ChatMessageDelta,
    /// Finish reason (if done).
    pub finish_reason: Option<String>,
}

/// Message delta for streaming.
#[derive(Debug, Clone, Default)]
pub struct ChatMessageDelta {
    /// Role (only in first chunk).
    pub role: Option<Role>,
    /// Content delta.
    pub content: Option<String>,
}

/// Text completion request.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    /// Model ID.
    pub model: String,
    /// The prompt.
    pub prompt: String,
    /// Temperature for sampling.
    pub temperature: Option<f32>,
    /// Top-p sampling.
    pub top_p: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<i32>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Request priority.
    pub priority: GrpcPriority,
    /// Request ID for tracing.
    pub request_id: Option<String>,
}

/// Text completion choice.
#[derive(Debug, Clone, Default)]
pub struct CompletionChoice {
    /// Index of the choice.
    pub index: i32,
    /// Generated text.
    pub text: String,
    /// Finish reason.
    pub finish_reason: String,
}

/// Text completion response.
#[derive(Debug, Clone, Default)]
pub struct CompletionResponse {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token usage.
    pub usage: Usage,
}

/// Streaming completion chunk.
#[derive(Debug, Clone, Default)]
pub struct CompletionChunk {
    /// Response ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Model used.
    pub model: String,
    /// Delta choices.
    pub choices: Vec<CompletionChoiceDelta>,
}

/// Delta for streaming completion.
#[derive(Debug, Clone, Default)]
pub struct CompletionChoiceDelta {
    /// Index of the choice.
    pub index: i32,
    /// Text delta.
    pub text: String,
    /// Finish reason (if done).
    pub finish_reason: Option<String>,
}

/// Embedding request.
#[derive(Debug, Clone, Default)]
pub struct EmbedRequest {
    /// Model ID.
    pub model: String,
    /// Input texts.
    pub input: Vec<String>,
    /// Encoding format.
    pub encoding_format: Option<String>,
    /// Dimensions.
    pub dimensions: Option<i32>,
    /// Request priority.
    pub priority: GrpcPriority,
    /// Request ID.
    pub request_id: Option<String>,
}

/// Single embedding.
#[derive(Debug, Clone, Default)]
pub struct Embedding {
    /// Object type.
    pub object: String,
    /// Index.
    pub index: i32,
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

/// Embedding response.
#[derive(Debug, Clone, Default)]
pub struct EmbedResponse {
    /// Object type.
    pub object: String,
    /// Embeddings.
    pub data: Vec<Embedding>,
    /// Model used.
    pub model: String,
    /// Token usage.
    pub usage: Usage,
}

/// List models request.
#[derive(Debug, Clone, Default)]
pub struct ListModelsRequest {}

/// Model information.
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// Model ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Creation timestamp.
    pub created: i64,
    /// Owner.
    pub owned_by: String,
    /// Context length.
    pub context_length: Option<i32>,
}

/// List models response.
#[derive(Debug, Clone, Default)]
pub struct ListModelsResponse {
    /// Object type.
    pub object: String,
    /// Models.
    pub data: Vec<Model>,
}

/// Health check request.
#[derive(Debug, Clone, Default)]
pub struct HealthCheckRequest {}

/// Component health.
#[derive(Debug, Clone, Default)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Status.
    pub status: String,
    /// Optional message.
    pub message: Option<String>,
}

/// Health check response.
#[derive(Debug, Clone, Default)]
pub struct HealthCheckResponse {
    /// Overall status.
    pub status: String,
    /// Version.
    pub version: String,
    /// Uptime in seconds.
    pub uptime_seconds: i64,
    /// Component health.
    pub components: Vec<ComponentHealth>,
}

// ============================================================================
// Service Error
// ============================================================================

/// Error from gRPC service.
#[derive(Debug, Clone)]
pub enum GrpcError {
    /// Invalid request.
    InvalidRequest(String),
    /// Model not found.
    ModelNotFound(String),
    /// Internal error.
    Internal(String),
    /// Unavailable.
    Unavailable(String),
    /// Resource exhausted.
    ResourceExhausted(String),
    /// Deadline exceeded.
    DeadlineExceeded(String),
    /// Unauthenticated.
    Unauthenticated(String),
    /// Permission denied.
    PermissionDenied(String),
}

impl fmt::Display for GrpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
            Self::Unavailable(msg) => write!(f, "Unavailable: {}", msg),
            Self::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            Self::DeadlineExceeded(msg) => write!(f, "Deadline exceeded: {}", msg),
            Self::Unauthenticated(msg) => write!(f, "Unauthenticated: {}", msg),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
        }
    }
}

impl std::error::Error for GrpcError {}

impl From<GrpcError> for Status {
    fn from(err: GrpcError) -> Self {
        match err {
            GrpcError::InvalidRequest(msg) => Status::invalid_argument(msg),
            GrpcError::ModelNotFound(msg) => Status::not_found(msg),
            GrpcError::Internal(msg) => Status::internal(msg),
            GrpcError::Unavailable(msg) => Status::unavailable(msg),
            GrpcError::ResourceExhausted(msg) => Status::resource_exhausted(msg),
            GrpcError::DeadlineExceeded(msg) => Status::deadline_exceeded(msg),
            GrpcError::Unauthenticated(msg) => Status::unauthenticated(msg),
            GrpcError::PermissionDenied(msg) => Status::permission_denied(msg),
        }
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Metrics for the gRPC service.
#[derive(Debug)]
pub struct GrpcMetrics {
    /// Total requests.
    requests_total: AtomicU64,
    /// Successful requests.
    requests_success: AtomicU64,
    /// Failed requests.
    requests_failed: AtomicU64,
    /// Active streams.
    active_streams: AtomicU64,
    /// Total response time in nanoseconds.
    response_time_ns: AtomicU64,
    /// Started at.
    started_at: Instant,
}

impl GrpcMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            active_streams: AtomicU64::new(0),
            response_time_ns: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    /// Records a request start.
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a successful request.
    pub fn record_success(&self, duration: Duration) {
        self.requests_success.fetch_add(1, Ordering::Relaxed);
        self.response_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Records a failed request.
    pub fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments active streams.
    pub fn stream_start(&self) {
        self.active_streams.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements active streams.
    pub fn stream_end(&self) {
        self.active_streams.fetch_sub(1, Ordering::Relaxed);
    }

    /// Returns total requests.
    pub fn requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    /// Returns successful requests.
    pub fn requests_success(&self) -> u64 {
        self.requests_success.load(Ordering::Relaxed)
    }

    /// Returns failed requests.
    pub fn requests_failed(&self) -> u64 {
        self.requests_failed.load(Ordering::Relaxed)
    }

    /// Returns active streams.
    pub fn active_streams(&self) -> u64 {
        self.active_streams.load(Ordering::Relaxed)
    }

    /// Returns uptime.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns average response time.
    pub fn avg_response_time(&self) -> Duration {
        let success = self.requests_success();
        let total_ns = self.response_time_ns.load(Ordering::Relaxed);
        if success > 0 {
            Duration::from_nanos(total_ns / success)
        } else {
            Duration::ZERO
        }
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_grpc_requests_total Total gRPC requests\n");
        output.push_str("# TYPE infernum_grpc_requests_total counter\n");
        output.push_str(&format!(
            "infernum_grpc_requests_total {}\n",
            self.requests_total()
        ));

        output.push_str("# HELP infernum_grpc_requests_success Successful gRPC requests\n");
        output.push_str("# TYPE infernum_grpc_requests_success counter\n");
        output.push_str(&format!(
            "infernum_grpc_requests_success {}\n",
            self.requests_success()
        ));

        output.push_str("# HELP infernum_grpc_requests_failed Failed gRPC requests\n");
        output.push_str("# TYPE infernum_grpc_requests_failed counter\n");
        output.push_str(&format!(
            "infernum_grpc_requests_failed {}\n",
            self.requests_failed()
        ));

        output.push_str("# HELP infernum_grpc_active_streams Active gRPC streams\n");
        output.push_str("# TYPE infernum_grpc_active_streams gauge\n");
        output.push_str(&format!(
            "infernum_grpc_active_streams {}\n",
            self.active_streams()
        ));

        output.push_str("# HELP infernum_grpc_avg_response_seconds Average response time\n");
        output.push_str("# TYPE infernum_grpc_avg_response_seconds gauge\n");
        output.push_str(&format!(
            "infernum_grpc_avg_response_seconds {:.6}\n",
            self.avg_response_time().as_secs_f64()
        ));

        output
    }
}

impl Default for GrpcMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Service Trait
// ============================================================================

/// The Infernum service trait.
#[tonic::async_trait]
pub trait InfernumService: Send + Sync + 'static {
    /// Chat completion.
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GrpcError>;

    /// Streaming chat completion.
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, Status>> + Send>>, GrpcError>;

    /// Text completion.
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, GrpcError>;

    /// Streaming text completion.
    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionChunk, Status>> + Send>>, GrpcError>;

    /// Generate embeddings.
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, GrpcError>;

    /// List available models.
    async fn list_models(&self) -> Result<ListModelsResponse, GrpcError>;

    /// Health check.
    async fn health_check(&self) -> Result<HealthCheckResponse, GrpcError>;
}

// ============================================================================
// Mock Service Implementation (for testing)
// ============================================================================

/// Mock implementation of the Infernum service for testing.
#[derive(Debug, Default)]
pub struct MockInfernumService {
    /// Metrics.
    pub metrics: Arc<GrpcMetrics>,
}

impl MockInfernumService {
    /// Creates a new mock service.
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(GrpcMetrics::new()),
        }
    }
}

#[tonic::async_trait]
impl InfernumService for MockInfernumService {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GrpcError> {
        let start = Instant::now();
        self.metrics.record_request();

        // Simulate response
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::assistant("This is a mock response."),
                finish_reason: "stop".to_string(),
            }],
            usage: Usage::new(10, 5),
        };

        self.metrics.record_success(start.elapsed());
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, Status>> + Send>>, GrpcError>
    {
        self.metrics.record_request();
        self.metrics.stream_start();

        let (tx, rx) = mpsc::channel(10);
        let model = request.model.clone();
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();

        tokio::spawn(async move {
            // First chunk with role
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatMessageDelta {
                        role: Some(Role::Assistant),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            let _ = tx.send(Ok(chunk)).await;

            // Content chunks
            for word in ["This", " is", " a", " mock", " streaming", " response", "."] {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatChoiceDelta {
                        index: 0,
                        delta: ChatMessageDelta {
                            role: None,
                            content: Some(word.to_string()),
                        },
                        finish_reason: None,
                    }],
                };
                let _ = tx.send(Ok(chunk)).await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            // Final chunk
            let chunk = ChatCompletionChunk {
                id,
                object: "chat.completion.chunk".to_string(),
                created,
                model,
                choices: vec![ChatChoiceDelta {
                    index: 0,
                    delta: ChatMessageDelta::default(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let _ = tx.send(Ok(chunk)).await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, GrpcError> {
        let start = Instant::now();
        self.metrics.record_request();

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "This is a mock completion.".to_string(),
                finish_reason: "stop".to_string(),
            }],
            usage: Usage::new(10, 5),
        };

        self.metrics.record_success(start.elapsed());
        Ok(response)
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionChunk, Status>> + Send>>, GrpcError> {
        self.metrics.record_request();
        self.metrics.stream_start();

        let (tx, rx) = mpsc::channel(10);
        let model = request.model.clone();
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();

        tokio::spawn(async move {
            for word in ["This", " is", " a", " mock", " completion", "."] {
                let chunk = CompletionChunk {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![CompletionChoiceDelta {
                        index: 0,
                        text: word.to_string(),
                        finish_reason: None,
                    }],
                };
                let _ = tx.send(Ok(chunk)).await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            // Final chunk
            let chunk = CompletionChunk {
                id,
                object: "text_completion".to_string(),
                created,
                model,
                choices: vec![CompletionChoiceDelta {
                    index: 0,
                    text: String::new(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let _ = tx.send(Ok(chunk)).await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, GrpcError> {
        let start = Instant::now();
        self.metrics.record_request();

        let dims = request.dimensions.unwrap_or(1536) as usize;
        let embeddings: Vec<Embedding> = request
            .input
            .iter()
            .enumerate()
            .map(|(i, _)| Embedding {
                object: "embedding".to_string(),
                index: i as i32,
                embedding: vec![0.0; dims],
            })
            .collect();

        let response = EmbedResponse {
            object: "list".to_string(),
            data: embeddings,
            model: request.model.clone(),
            usage: Usage::new(request.input.len() as i32 * 5, 0),
        };

        self.metrics.record_success(start.elapsed());
        Ok(response)
    }

    async fn list_models(&self) -> Result<ListModelsResponse, GrpcError> {
        self.metrics.record_request();

        Ok(ListModelsResponse {
            object: "list".to_string(),
            data: vec![
                Model {
                    id: "llama-3-8b".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "meta".to_string(),
                    context_length: Some(8192),
                },
                Model {
                    id: "qwen-2.5-32b".to_string(),
                    object: "model".to_string(),
                    created: 1700000000,
                    owned_by: "alibaba".to_string(),
                    context_length: Some(32768),
                },
            ],
        })
    }

    async fn health_check(&self) -> Result<HealthCheckResponse, GrpcError> {
        Ok(HealthCheckResponse {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.metrics.uptime().as_secs() as i64,
            components: vec![
                ComponentHealth {
                    name: "inference".to_string(),
                    status: "healthy".to_string(),
                    message: None,
                },
                ComponentHealth {
                    name: "models".to_string(),
                    status: "healthy".to_string(),
                    message: Some("2 models loaded".to_string()),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_config_default() {
        let config = GrpcConfig::default();

        assert_eq!(config.max_message_size, 16 * 1024 * 1024);
        assert!(config.enable_reflection);
        assert!(config.enable_health);
    }

    #[test]
    fn test_grpc_config_builder() {
        let config = GrpcConfig::new()
            .with_max_message_size(8 * 1024 * 1024)
            .with_reflection(false)
            .with_max_concurrent_streams(100);

        assert_eq!(config.max_message_size, 8 * 1024 * 1024);
        assert!(!config.enable_reflection);
        assert_eq!(config.max_concurrent_streams, 100);
    }

    #[test]
    fn test_grpc_priority_conversion() {
        assert_eq!(GrpcPriority::from(0), GrpcPriority::Unspecified);
        assert_eq!(GrpcPriority::from(1), GrpcPriority::Background);
        assert_eq!(GrpcPriority::from(2), GrpcPriority::Normal);
        assert_eq!(GrpcPriority::from(3), GrpcPriority::High);
        assert_eq!(GrpcPriority::from(4), GrpcPriority::Critical);
    }

    #[test]
    fn test_grpc_priority_to_batch_priority() {
        assert_eq!(BatchPriority::from(GrpcPriority::Background), BatchPriority::Background);
        assert_eq!(BatchPriority::from(GrpcPriority::Normal), BatchPriority::Normal);
        assert_eq!(BatchPriority::from(GrpcPriority::High), BatchPriority::High);
        assert_eq!(BatchPriority::from(GrpcPriority::Critical), BatchPriority::Critical);
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    #[test]
    fn test_usage_new() {
        let usage = Usage::new(10, 5);

        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_chat_message_constructors() {
        let system = ChatMessage::system("You are helpful.");
        assert_eq!(system.role, Role::System);
        assert_eq!(system.content, "You are helpful.");

        let user = ChatMessage::user("Hello!");
        assert_eq!(user.role, Role::User);

        let assistant = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant.role, Role::Assistant);
    }

    #[test]
    fn test_grpc_error_display() {
        let err = GrpcError::InvalidRequest("bad input".to_string());
        assert!(err.to_string().contains("Invalid request"));

        let err = GrpcError::ModelNotFound("gpt-5".to_string());
        assert!(err.to_string().contains("Model not found"));

        let err = GrpcError::Internal("oops".to_string());
        assert!(err.to_string().contains("Internal"));
    }

    #[test]
    fn test_grpc_error_to_status() {
        let err = GrpcError::InvalidRequest("test".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);

        let err = GrpcError::ModelNotFound("test".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::NotFound);

        let err = GrpcError::Unauthenticated("test".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_grpc_metrics_new() {
        let metrics = GrpcMetrics::new();

        assert_eq!(metrics.requests_total(), 0);
        assert_eq!(metrics.requests_success(), 0);
        assert_eq!(metrics.requests_failed(), 0);
        assert_eq!(metrics.active_streams(), 0);
    }

    #[test]
    fn test_grpc_metrics_record() {
        let metrics = GrpcMetrics::new();

        metrics.record_request();
        metrics.record_request();
        metrics.record_success(Duration::from_millis(100));
        metrics.record_failure();

        assert_eq!(metrics.requests_total(), 2);
        assert_eq!(metrics.requests_success(), 1);
        assert_eq!(metrics.requests_failed(), 1);
    }

    #[test]
    fn test_grpc_metrics_streams() {
        let metrics = GrpcMetrics::new();

        metrics.stream_start();
        metrics.stream_start();
        assert_eq!(metrics.active_streams(), 2);

        metrics.stream_end();
        assert_eq!(metrics.active_streams(), 1);
    }

    #[test]
    fn test_grpc_metrics_prometheus() {
        let metrics = GrpcMetrics::new();
        metrics.record_request();
        metrics.record_success(Duration::from_millis(10));

        let output = metrics.prometheus();

        assert!(output.contains("infernum_grpc_requests_total 1"));
        assert!(output.contains("infernum_grpc_requests_success 1"));
        assert!(output.contains("infernum_grpc_active_streams"));
    }

    #[tokio::test]
    async fn test_mock_service_chat_completion() {
        let service = MockInfernumService::new();

        let request = ChatCompletionRequest {
            model: "llama-3-8b".to_string(),
            messages: vec![ChatMessage::user("Hello!")],
            ..Default::default()
        };

        let response = service.chat_completion(request).await.unwrap();

        assert!(!response.id.is_empty());
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[tokio::test]
    async fn test_mock_service_completion() {
        let service = MockInfernumService::new();

        let request = CompletionRequest {
            model: "llama-3-8b".to_string(),
            prompt: "Hello".to_string(),
            ..Default::default()
        };

        let response = service.completion(request).await.unwrap();

        assert!(!response.id.is_empty());
        assert_eq!(response.object, "text_completion");
        assert_eq!(response.choices.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_service_embed() {
        let service = MockInfernumService::new();

        let request = EmbedRequest {
            model: "llama-3-8b".to_string(),
            input: vec!["Hello".to_string(), "World".to_string()],
            dimensions: Some(768),
            ..Default::default()
        };

        let response = service.embed(request).await.unwrap();

        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].embedding.len(), 768);
    }

    #[tokio::test]
    async fn test_mock_service_list_models() {
        let service = MockInfernumService::new();

        let response = service.list_models().await.unwrap();

        assert_eq!(response.object, "list");
        assert!(!response.data.is_empty());
    }

    #[tokio::test]
    async fn test_mock_service_health_check() {
        let service = MockInfernumService::new();

        let response = service.health_check().await.unwrap();

        assert_eq!(response.status, "healthy");
        assert!(!response.version.is_empty());
        assert!(!response.components.is_empty());
    }
}
