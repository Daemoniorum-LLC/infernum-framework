//! WebSocket streaming support for low-latency inference.
//!
//! This module provides WebSocket-based streaming as an alternative to SSE,
//! offering lower latency and bidirectional communication.
//!
//! # Features
//!
//! - **Low-Latency Streaming**: Faster than SSE for token-by-token output
//! - **Bidirectional**: Supports sending messages while receiving responses
//! - **Reconnection**: Built-in support for connection recovery
//! - **Message Framing**: JSON-based message protocol
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::websocket::{WebSocketHandler, WsConfig};
//!
//! let config = WsConfig::default();
//! let handler = WebSocketHandler::new(config);
//!
//! // In axum router:
//! Router::new()
//!     .route("/ws/chat", get(handler.handle))
//! ```
//!
//! # Protocol
//!
//! The WebSocket protocol uses JSON messages:
//!
//! ## Client -> Server
//! ```json
//! {
//!   "type": "chat",
//!   "payload": { /* ChatCompletionRequest */ }
//! }
//! ```
//!
//! ## Server -> Client
//! ```json
//! {
//!   "type": "delta",
//!   "payload": { "content": "Hello" }
//! }
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::openai::ChatCompletionRequest;

/// WebSocket connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is being established.
    Connecting,
    /// Connection is open and ready.
    Open,
    /// Connection is closing gracefully.
    Closing,
    /// Connection is closed.
    Closed,
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connecting => write!(f, "connecting"),
            Self::Open => write!(f, "open"),
            Self::Closing => write!(f, "closing"),
            Self::Closed => write!(f, "closed"),
        }
    }
}

/// Client-to-server WebSocket message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Chat completion request.
    Chat {
        /// The request payload.
        payload: ChatCompletionRequest,
        /// Optional request ID for correlation.
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
    },

    /// Ping message for keepalive.
    Ping {
        /// Optional timestamp for latency measurement.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },

    /// Cancel an in-flight request.
    Cancel {
        /// ID of the request to cancel.
        request_id: String,
    },
}

/// Server-to-client WebSocket message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Connection established acknowledgment.
    Connected {
        /// Server-assigned connection ID.
        connection_id: String,
        /// Server timestamp.
        timestamp: u64,
    },

    /// Streaming delta (incremental content).
    Delta {
        /// The request ID this delta belongs to.
        request_id: String,
        /// Index of the choice.
        index: u32,
        /// Incremental content.
        content: Option<String>,
        /// Role (only on first delta).
        role: Option<String>,
    },

    /// Completion done.
    Done {
        /// The request ID that completed.
        request_id: String,
        /// Finish reason.
        finish_reason: String,
        /// Token usage statistics.
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<UsageInfo>,
    },

    /// Error response.
    Error {
        /// The request ID (if applicable).
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        /// Error code.
        code: String,
        /// Error message.
        message: String,
    },

    /// Pong response to ping.
    Pong {
        /// Echo of the client timestamp.
        client_timestamp: Option<u64>,
        /// Server timestamp.
        server_timestamp: u64,
    },

    /// Request was cancelled.
    Cancelled {
        /// The request ID that was cancelled.
        request_id: String,
    },
}

/// Token usage information for WebSocket responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

impl UsageInfo {
    /// Creates new usage info.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

impl From<crate::openai::Usage> for UsageInfo {
    fn from(usage: crate::openai::Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

/// WebSocket configuration.
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// Maximum message size in bytes.
    pub max_message_size: usize,

    /// Maximum frame size in bytes.
    pub max_frame_size: usize,

    /// Ping interval for keepalive.
    pub ping_interval: Duration,

    /// Pong timeout (disconnect if no pong received).
    pub pong_timeout: Duration,

    /// Maximum concurrent requests per connection.
    pub max_concurrent_requests: usize,

    /// Enable compression.
    pub compression: bool,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            max_message_size: 16 * 1024 * 1024, // 16MB
            max_frame_size: 64 * 1024,          // 64KB
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(10),
            max_concurrent_requests: 10,
            compression: true,
        }
    }
}

impl WsConfig {
    /// Creates a new WebSocket config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set max message size.
    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    /// Builder method to set ping interval.
    pub fn with_ping_interval(mut self, interval: Duration) -> Self {
        self.ping_interval = interval;
        self
    }

    /// Builder method to set max concurrent requests.
    pub fn with_max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = max;
        self
    }

    /// Builder method to enable/disable compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }
}

/// WebSocket close reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    /// Normal closure.
    Normal,
    /// Server is going away.
    GoingAway,
    /// Protocol error.
    ProtocolError,
    /// Unsupported data.
    UnsupportedData,
    /// Invalid frame payload.
    InvalidPayload,
    /// Policy violation.
    PolicyViolation,
    /// Message too big.
    MessageTooBig,
    /// Server error.
    InternalError,
    /// Service restart.
    ServiceRestart,
    /// Try again later.
    TryAgainLater,
}

impl CloseReason {
    /// Gets the WebSocket close code.
    pub fn code(&self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::ProtocolError => 1002,
            Self::UnsupportedData => 1003,
            Self::InvalidPayload => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::InternalError => 1011,
            Self::ServiceRestart => 1012,
            Self::TryAgainLater => 1013,
        }
    }

    /// Gets the close reason description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Normal => "Normal closure",
            Self::GoingAway => "Server going away",
            Self::ProtocolError => "Protocol error",
            Self::UnsupportedData => "Unsupported data",
            Self::InvalidPayload => "Invalid payload",
            Self::PolicyViolation => "Policy violation",
            Self::MessageTooBig => "Message too big",
            Self::InternalError => "Internal server error",
            Self::ServiceRestart => "Service restart",
            Self::TryAgainLater => "Try again later",
        }
    }

    /// Creates a close reason from a WebSocket close code.
    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1000 => Some(Self::Normal),
            1001 => Some(Self::GoingAway),
            1002 => Some(Self::ProtocolError),
            1003 => Some(Self::UnsupportedData),
            1007 => Some(Self::InvalidPayload),
            1008 => Some(Self::PolicyViolation),
            1009 => Some(Self::MessageTooBig),
            1011 => Some(Self::InternalError),
            1012 => Some(Self::ServiceRestart),
            1013 => Some(Self::TryAgainLater),
            _ => None,
        }
    }
}

impl fmt::Display for CloseReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.description(), self.code())
    }
}

/// WebSocket connection info.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Connection ID.
    pub id: String,

    /// Connection state.
    pub state: ConnectionState,

    /// When the connection was established.
    pub connected_at: Instant,

    /// Client IP address (if available).
    pub client_ip: Option<String>,

    /// Number of messages received.
    pub messages_received: u64,

    /// Number of messages sent.
    pub messages_sent: u64,

    /// Number of active requests.
    pub active_requests: u64,
}

impl ConnectionInfo {
    /// Creates new connection info.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state: ConnectionState::Connecting,
            connected_at: Instant::now(),
            client_ip: None,
            messages_received: 0,
            messages_sent: 0,
            active_requests: 0,
        }
    }

    /// Builder method to set client IP.
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }

    /// Returns the connection duration.
    pub fn duration(&self) -> Duration {
        self.connected_at.elapsed()
    }

    /// Returns true if the connection is open.
    pub fn is_open(&self) -> bool {
        self.state == ConnectionState::Open
    }
}

/// WebSocket metrics.
#[derive(Debug)]
pub struct WsMetrics {
    /// Total connections opened.
    connections_opened: AtomicU64,

    /// Total connections closed.
    connections_closed: AtomicU64,

    /// Current active connections.
    active_connections: AtomicU64,

    /// Total messages received.
    messages_received: AtomicU64,

    /// Total messages sent.
    messages_sent: AtomicU64,

    /// Total errors.
    errors: AtomicU64,

    /// Total requests processed.
    requests_processed: AtomicU64,
}

impl WsMetrics {
    /// Creates a new metrics instance.
    pub fn new() -> Self {
        Self {
            connections_opened: AtomicU64::new(0),
            connections_closed: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            requests_processed: AtomicU64::new(0),
        }
    }

    /// Records a connection opened.
    pub fn record_connection_opened(&self) {
        self.connections_opened.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a connection closed.
    pub fn record_connection_closed(&self) {
        self.connections_closed.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Records a message received.
    pub fn record_message_received(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a message sent.
    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an error.
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a request processed.
    pub fn record_request_processed(&self) {
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Gets the total connections opened.
    pub fn connections_opened(&self) -> u64 {
        self.connections_opened.load(Ordering::Relaxed)
    }

    /// Gets the total connections closed.
    pub fn connections_closed(&self) -> u64 {
        self.connections_closed.load(Ordering::Relaxed)
    }

    /// Gets the current active connections.
    pub fn active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Gets the total messages received.
    pub fn messages_received(&self) -> u64 {
        self.messages_received.load(Ordering::Relaxed)
    }

    /// Gets the total messages sent.
    pub fn messages_sent(&self) -> u64 {
        self.messages_sent.load(Ordering::Relaxed)
    }

    /// Gets the total errors.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Gets the total requests processed.
    pub fn requests_processed(&self) -> u64 {
        self.requests_processed.load(Ordering::Relaxed)
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_ws_connections_opened_total Total WebSocket connections opened\n");
        output.push_str("# TYPE infernum_ws_connections_opened_total counter\n");
        output.push_str(&format!(
            "infernum_ws_connections_opened_total {}\n",
            self.connections_opened()
        ));

        output.push_str("# HELP infernum_ws_connections_closed_total Total WebSocket connections closed\n");
        output.push_str("# TYPE infernum_ws_connections_closed_total counter\n");
        output.push_str(&format!(
            "infernum_ws_connections_closed_total {}\n",
            self.connections_closed()
        ));

        output.push_str("# HELP infernum_ws_connections_active Current active WebSocket connections\n");
        output.push_str("# TYPE infernum_ws_connections_active gauge\n");
        output.push_str(&format!(
            "infernum_ws_connections_active {}\n",
            self.active_connections()
        ));

        output.push_str("# HELP infernum_ws_messages_received_total Total WebSocket messages received\n");
        output.push_str("# TYPE infernum_ws_messages_received_total counter\n");
        output.push_str(&format!(
            "infernum_ws_messages_received_total {}\n",
            self.messages_received()
        ));

        output.push_str("# HELP infernum_ws_messages_sent_total Total WebSocket messages sent\n");
        output.push_str("# TYPE infernum_ws_messages_sent_total counter\n");
        output.push_str(&format!(
            "infernum_ws_messages_sent_total {}\n",
            self.messages_sent()
        ));

        output.push_str("# HELP infernum_ws_errors_total Total WebSocket errors\n");
        output.push_str("# TYPE infernum_ws_errors_total counter\n");
        output.push_str(&format!(
            "infernum_ws_errors_total {}\n",
            self.errors()
        ));

        output.push_str("# HELP infernum_ws_requests_processed_total Total WebSocket requests processed\n");
        output.push_str("# TYPE infernum_ws_requests_processed_total counter\n");
        output.push_str(&format!(
            "infernum_ws_requests_processed_total {}\n",
            self.requests_processed()
        ));

        output
    }
}

impl Default for WsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket connection manager for tracking all connections.
#[derive(Debug)]
pub struct ConnectionManager {
    /// Configuration.
    config: WsConfig,

    /// Metrics.
    metrics: Arc<WsMetrics>,

    /// Shutdown broadcast channel.
    shutdown_tx: broadcast::Sender<()>,
}

impl ConnectionManager {
    /// Creates a new connection manager.
    pub fn new(config: WsConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            config,
            metrics: Arc::new(WsMetrics::new()),
            shutdown_tx,
        }
    }

    /// Gets the configuration.
    pub fn config(&self) -> &WsConfig {
        &self.config
    }

    /// Gets the metrics.
    pub fn metrics(&self) -> Arc<WsMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Subscribes to shutdown notifications.
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Broadcasts shutdown to all connections.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Generates a new connection ID.
    pub fn generate_connection_id(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new(WsConfig::default())
    }
}

/// Error type for WebSocket operations.
#[derive(Debug)]
pub enum WsError {
    /// Connection closed.
    ConnectionClosed(CloseReason),

    /// Message parse error.
    ParseError(String),

    /// Request limit exceeded.
    TooManyRequests,

    /// Message too large.
    MessageTooLarge,

    /// Request not found.
    RequestNotFound(String),

    /// Internal error.
    Internal(String),
}

impl fmt::Display for WsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionClosed(reason) => write!(f, "Connection closed: {}", reason),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::TooManyRequests => write!(f, "Too many concurrent requests"),
            Self::MessageTooLarge => write!(f, "Message too large"),
            Self::RequestNotFound(id) => write!(f, "Request not found: {}", id),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for WsError {}

impl WsError {
    /// Converts the error to a server message.
    pub fn to_server_message(&self, request_id: Option<String>) -> ServerMessage {
        let (code, message) = match self {
            Self::ConnectionClosed(reason) => ("connection_closed".to_string(), reason.to_string()),
            Self::ParseError(msg) => ("parse_error".to_string(), msg.clone()),
            Self::TooManyRequests => ("too_many_requests".to_string(), "Too many concurrent requests".to_string()),
            Self::MessageTooLarge => ("message_too_large".to_string(), "Message exceeds size limit".to_string()),
            Self::RequestNotFound(id) => ("request_not_found".to_string(), format!("Request not found: {}", id)),
            Self::Internal(msg) => ("internal_error".to_string(), msg.clone()),
        };

        ServerMessage::Error {
            request_id,
            code,
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_state_display() {
        assert_eq!(ConnectionState::Connecting.to_string(), "connecting");
        assert_eq!(ConnectionState::Open.to_string(), "open");
        assert_eq!(ConnectionState::Closing.to_string(), "closing");
        assert_eq!(ConnectionState::Closed.to_string(), "closed");
    }

    #[test]
    fn test_client_message_chat_serialization() {
        let msg = ClientMessage::Chat {
            payload: make_test_request(),
            request_id: Some("req-123".to_string()),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"chat\""));
        assert!(json.contains("\"request_id\":\"req-123\""));
    }

    #[test]
    fn test_client_message_ping() {
        let msg = ClientMessage::Ping {
            timestamp: Some(1234567890),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"ping\""));
        assert!(json.contains("\"timestamp\":1234567890"));
    }

    #[test]
    fn test_client_message_cancel() {
        let msg = ClientMessage::Cancel {
            request_id: "req-456".to_string(),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"cancel\""));
        assert!(json.contains("\"request_id\":\"req-456\""));
    }

    #[test]
    fn test_server_message_connected() {
        let msg = ServerMessage::Connected {
            connection_id: "conn-123".to_string(),
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"connected\""));
        assert!(json.contains("\"connection_id\":\"conn-123\""));
    }

    #[test]
    fn test_server_message_delta() {
        let msg = ServerMessage::Delta {
            request_id: "req-123".to_string(),
            index: 0,
            content: Some("Hello".to_string()),
            role: Some("assistant".to_string()),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"delta\""));
        assert!(json.contains("\"content\":\"Hello\""));
        assert!(json.contains("\"role\":\"assistant\""));
    }

    #[test]
    fn test_server_message_done() {
        let msg = ServerMessage::Done {
            request_id: "req-123".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(UsageInfo::new(10, 20)),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"done\""));
        assert!(json.contains("\"finish_reason\":\"stop\""));
        assert!(json.contains("\"total_tokens\":30"));
    }

    #[test]
    fn test_server_message_error() {
        let msg = ServerMessage::Error {
            request_id: Some("req-123".to_string()),
            code: "rate_limit".to_string(),
            message: "Too many requests".to_string(),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"code\":\"rate_limit\""));
    }

    #[test]
    fn test_server_message_pong() {
        let msg = ServerMessage::Pong {
            client_timestamp: Some(1000),
            server_timestamp: 2000,
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"pong\""));
        assert!(json.contains("\"client_timestamp\":1000"));
        assert!(json.contains("\"server_timestamp\":2000"));
    }

    #[test]
    fn test_usage_info_new() {
        let usage = UsageInfo::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_usage_info_from_openai() {
        let openai_usage = crate::openai::Usage::new(200, 100);
        let usage: UsageInfo = openai_usage.into();

        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 100);
        assert_eq!(usage.total_tokens, 300);
    }

    #[test]
    fn test_ws_config_default() {
        let config = WsConfig::default();

        assert_eq!(config.max_message_size, 16 * 1024 * 1024);
        assert_eq!(config.max_frame_size, 64 * 1024);
        assert_eq!(config.ping_interval, Duration::from_secs(30));
        assert!(config.compression);
    }

    #[test]
    fn test_ws_config_builder() {
        let config = WsConfig::new()
            .with_max_message_size(1024)
            .with_ping_interval(Duration::from_secs(60))
            .with_max_concurrent_requests(5)
            .with_compression(false);

        assert_eq!(config.max_message_size, 1024);
        assert_eq!(config.ping_interval, Duration::from_secs(60));
        assert_eq!(config.max_concurrent_requests, 5);
        assert!(!config.compression);
    }

    #[test]
    fn test_close_reason_codes() {
        assert_eq!(CloseReason::Normal.code(), 1000);
        assert_eq!(CloseReason::GoingAway.code(), 1001);
        assert_eq!(CloseReason::ProtocolError.code(), 1002);
        assert_eq!(CloseReason::MessageTooBig.code(), 1009);
        assert_eq!(CloseReason::InternalError.code(), 1011);
    }

    #[test]
    fn test_close_reason_from_code() {
        assert_eq!(CloseReason::from_code(1000), Some(CloseReason::Normal));
        assert_eq!(CloseReason::from_code(1001), Some(CloseReason::GoingAway));
        assert_eq!(CloseReason::from_code(9999), None);
    }

    #[test]
    fn test_close_reason_display() {
        let reason = CloseReason::Normal;
        assert!(reason.to_string().contains("1000"));
        assert!(reason.to_string().contains("Normal"));
    }

    #[test]
    fn test_connection_info_new() {
        let info = ConnectionInfo::new("conn-123");

        assert_eq!(info.id, "conn-123");
        assert_eq!(info.state, ConnectionState::Connecting);
        assert_eq!(info.messages_received, 0);
        assert_eq!(info.messages_sent, 0);
        assert!(info.client_ip.is_none());
    }

    #[test]
    fn test_connection_info_with_client_ip() {
        let info = ConnectionInfo::new("conn-456")
            .with_client_ip("192.168.1.1");

        assert_eq!(info.client_ip, Some("192.168.1.1".to_string()));
    }

    #[test]
    fn test_connection_info_is_open() {
        let mut info = ConnectionInfo::new("conn-789");
        assert!(!info.is_open());

        info.state = ConnectionState::Open;
        assert!(info.is_open());
    }

    #[test]
    fn test_ws_metrics_new() {
        let metrics = WsMetrics::new();

        assert_eq!(metrics.connections_opened(), 0);
        assert_eq!(metrics.connections_closed(), 0);
        assert_eq!(metrics.active_connections(), 0);
        assert_eq!(metrics.messages_received(), 0);
        assert_eq!(metrics.messages_sent(), 0);
        assert_eq!(metrics.errors(), 0);
        assert_eq!(metrics.requests_processed(), 0);
    }

    #[test]
    fn test_ws_metrics_record_connection() {
        let metrics = WsMetrics::new();

        metrics.record_connection_opened();
        assert_eq!(metrics.connections_opened(), 1);
        assert_eq!(metrics.active_connections(), 1);

        metrics.record_connection_opened();
        assert_eq!(metrics.connections_opened(), 2);
        assert_eq!(metrics.active_connections(), 2);

        metrics.record_connection_closed();
        assert_eq!(metrics.connections_closed(), 1);
        assert_eq!(metrics.active_connections(), 1);
    }

    #[test]
    fn test_ws_metrics_record_messages() {
        let metrics = WsMetrics::new();

        metrics.record_message_received();
        metrics.record_message_received();
        assert_eq!(metrics.messages_received(), 2);

        metrics.record_message_sent();
        assert_eq!(metrics.messages_sent(), 1);
    }

    #[test]
    fn test_ws_metrics_record_errors() {
        let metrics = WsMetrics::new();

        metrics.record_error();
        metrics.record_error();
        assert_eq!(metrics.errors(), 2);
    }

    #[test]
    fn test_ws_metrics_prometheus() {
        let metrics = WsMetrics::new();
        metrics.record_connection_opened();
        metrics.record_message_received();
        metrics.record_message_sent();

        let output = metrics.prometheus();

        assert!(output.contains("infernum_ws_connections_opened_total 1"));
        assert!(output.contains("infernum_ws_connections_active 1"));
        assert!(output.contains("infernum_ws_messages_received_total 1"));
        assert!(output.contains("infernum_ws_messages_sent_total 1"));
    }

    #[test]
    fn test_connection_manager_new() {
        let config = WsConfig::default();
        let manager = ConnectionManager::new(config.clone());

        assert_eq!(manager.config().max_message_size, config.max_message_size);
        assert_eq!(manager.metrics().active_connections(), 0);
    }

    #[test]
    fn test_connection_manager_generate_id() {
        let manager = ConnectionManager::default();

        let id1 = manager.generate_connection_id();
        let id2 = manager.generate_connection_id();

        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 36); // UUID format
    }

    #[test]
    fn test_connection_manager_shutdown_subscribe() {
        let manager = ConnectionManager::default();
        let _rx = manager.subscribe_shutdown();

        // Should be able to subscribe multiple times
        let _rx2 = manager.subscribe_shutdown();
    }

    #[test]
    fn test_ws_error_display() {
        let err = WsError::ConnectionClosed(CloseReason::Normal);
        assert!(err.to_string().contains("Connection closed"));

        let err = WsError::ParseError("Invalid JSON".to_string());
        assert!(err.to_string().contains("Parse error"));

        let err = WsError::TooManyRequests;
        assert!(err.to_string().contains("Too many"));

        let err = WsError::MessageTooLarge;
        assert!(err.to_string().contains("too large"));

        let err = WsError::RequestNotFound("req-123".to_string());
        assert!(err.to_string().contains("req-123"));

        let err = WsError::Internal("Oops".to_string());
        assert!(err.to_string().contains("Internal"));
    }

    #[test]
    fn test_ws_error_to_server_message() {
        let err = WsError::TooManyRequests;
        let msg = err.to_server_message(Some("req-123".to_string()));

        match msg {
            ServerMessage::Error { request_id, code, message } => {
                assert_eq!(request_id, Some("req-123".to_string()));
                assert_eq!(code, "too_many_requests");
                assert!(message.contains("Too many"));
            }
            _ => panic!("Expected Error message"),
        }
    }

    #[test]
    fn test_ws_error_to_server_message_no_request_id() {
        let err = WsError::ParseError("Bad JSON".to_string());
        let msg = err.to_server_message(None);

        match msg {
            ServerMessage::Error { request_id, code, .. } => {
                assert!(request_id.is_none());
                assert_eq!(code, "parse_error");
            }
            _ => panic!("Expected Error message"),
        }
    }

    #[test]
    fn test_client_message_deserialization_chat() {
        let json = r#"{
            "type": "chat",
            "payload": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            "request_id": "req-001"
        }"#;

        let msg: ClientMessage = serde_json::from_str(json).expect("deserialize");
        match msg {
            ClientMessage::Chat { payload, request_id } => {
                assert_eq!(payload.model, "test-model");
                assert_eq!(request_id, Some("req-001".to_string()));
            }
            _ => panic!("Expected Chat message"),
        }
    }

    #[test]
    fn test_client_message_deserialization_ping() {
        let json = r#"{"type": "ping", "timestamp": 12345}"#;

        let msg: ClientMessage = serde_json::from_str(json).expect("deserialize");
        match msg {
            ClientMessage::Ping { timestamp } => {
                assert_eq!(timestamp, Some(12345));
            }
            _ => panic!("Expected Ping message"),
        }
    }

    #[test]
    fn test_client_message_deserialization_cancel() {
        let json = r#"{"type": "cancel", "request_id": "req-999"}"#;

        let msg: ClientMessage = serde_json::from_str(json).expect("deserialize");
        match msg {
            ClientMessage::Cancel { request_id } => {
                assert_eq!(request_id, "req-999");
            }
            _ => panic!("Expected Cancel message"),
        }
    }

    #[test]
    fn test_server_message_cancelled() {
        let msg = ServerMessage::Cancelled {
            request_id: "req-123".to_string(),
        };

        let json = serde_json::to_string(&msg).expect("serialize");
        assert!(json.contains("\"type\":\"cancelled\""));
        assert!(json.contains("\"request_id\":\"req-123\""));
    }

    // Helper to create a test request
    fn make_test_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![crate::openai::ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
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
}
