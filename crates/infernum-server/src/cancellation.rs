//! Request cancellation handling for graceful client disconnection.
//!
//! This module provides cancellation token support for long-running inference
//! requests. When a client disconnects, the cancellation token is triggered,
//! allowing handlers to abort work early and free resources.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::cancellation::{CancellationToken, RequestCancellation};
//!
//! let cancellation = RequestCancellation::new();
//! let token = cancellation.token();
//!
//! // In handler:
//! loop {
//!     if token.is_cancelled() {
//!         return Err(CancellationError::ClientDisconnected);
//!     }
//!     // Do work...
//! }
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Error returned when a request is cancelled.
#[derive(Debug, Clone)]
pub struct CancellationError {
    /// Reason for cancellation.
    pub reason: CancellationReason,
    /// How long the request ran before cancellation.
    pub elapsed: Duration,
}

impl std::fmt::Display for CancellationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "request cancelled: {} (after {:?})",
            self.reason, self.elapsed
        )
    }
}

impl std::error::Error for CancellationError {}

/// Reason for request cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationReason {
    /// Client disconnected before response completed.
    ClientDisconnected,
    /// Request timed out.
    Timeout,
    /// Server is shutting down.
    ServerShutdown,
    /// Manually cancelled by admin or system.
    Manual,
}

impl std::fmt::Display for CancellationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClientDisconnected => write!(f, "client disconnected"),
            Self::Timeout => write!(f, "timeout"),
            Self::ServerShutdown => write!(f, "server shutdown"),
            Self::Manual => write!(f, "manual cancellation"),
        }
    }
}

/// Token that can be checked for cancellation.
#[derive(Clone)]
pub struct CancellationToken {
    inner: Arc<CancellationState>,
}

impl CancellationToken {
    /// Creates a new cancellation token.
    fn new(inner: Arc<CancellationState>) -> Self {
        Self { inner }
    }

    /// Returns true if cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Returns the cancellation reason, if cancelled.
    #[must_use]
    pub fn reason(&self) -> Option<CancellationReason> {
        if self.is_cancelled() {
            self.inner.reason()
        } else {
            None
        }
    }

    /// Returns how long since the request started.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.inner.started_at.elapsed()
    }

    /// Creates a CancellationError if cancelled.
    #[must_use]
    pub fn error(&self) -> Option<CancellationError> {
        self.reason().map(|reason| CancellationError {
            reason,
            elapsed: self.elapsed(),
        })
    }

    /// Waits for cancellation (async).
    ///
    /// Returns when the token is cancelled. This is useful for
    /// `tokio::select!` patterns.
    pub async fn cancelled(&self) {
        // Poll until cancelled
        while !self.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Creates a child token that is cancelled when either this token
    /// or the child is cancelled.
    #[must_use]
    pub fn child_token(&self) -> CancellationToken {
        // For simplicity, return a clone (shares same state)
        // A full implementation would create a linked hierarchy
        self.clone()
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("is_cancelled", &self.is_cancelled())
            .field("reason", &self.reason())
            .finish()
    }
}

/// Internal cancellation state.
struct CancellationState {
    cancelled: AtomicBool,
    reason_code: AtomicU64,
    started_at: Instant,
}

impl CancellationState {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            reason_code: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    fn cancel(&self, reason: CancellationReason) {
        self.reason_code.store(reason.to_code(), Ordering::Release);
        self.cancelled.store(true, Ordering::Release);
    }

    fn reason(&self) -> Option<CancellationReason> {
        let code = self.reason_code.load(Ordering::Acquire);
        CancellationReason::from_code(code)
    }
}

impl CancellationReason {
    fn to_code(self) -> u64 {
        match self {
            Self::ClientDisconnected => 1,
            Self::Timeout => 2,
            Self::ServerShutdown => 3,
            Self::Manual => 4,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        match code {
            1 => Some(Self::ClientDisconnected),
            2 => Some(Self::Timeout),
            3 => Some(Self::ServerShutdown),
            4 => Some(Self::Manual),
            _ => None,
        }
    }
}

/// Request cancellation controller.
///
/// Owns the ability to cancel a request and provides tokens for checking.
#[derive(Clone)]
pub struct RequestCancellation {
    state: Arc<CancellationState>,
}

impl Default for RequestCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestCancellation {
    /// Creates a new request cancellation controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancellationState::new()),
        }
    }

    /// Returns a token for checking cancellation.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        CancellationToken::new(Arc::clone(&self.state))
    }

    /// Cancels the request with the given reason.
    pub fn cancel(&self, reason: CancellationReason) {
        self.state.cancel(reason);
    }

    /// Returns true if cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Returns how long since the request started.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.state.started_at.elapsed()
    }
}

impl std::fmt::Debug for RequestCancellation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestCancellation")
            .field("is_cancelled", &self.is_cancelled())
            .field("elapsed", &self.elapsed())
            .finish()
    }
}

/// Metrics for cancelled requests.
#[derive(Debug, Default)]
pub struct CancellationMetrics {
    /// Total cancelled requests.
    client_disconnected: AtomicU64,
    timeout: AtomicU64,
    server_shutdown: AtomicU64,
    manual: AtomicU64,
}

impl CancellationMetrics {
    /// Creates new cancellation metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a cancellation.
    pub fn record(&self, reason: CancellationReason) {
        match reason {
            CancellationReason::ClientDisconnected => {
                self.client_disconnected.fetch_add(1, Ordering::Relaxed);
            }
            CancellationReason::Timeout => {
                self.timeout.fetch_add(1, Ordering::Relaxed);
            }
            CancellationReason::ServerShutdown => {
                self.server_shutdown.fetch_add(1, Ordering::Relaxed);
            }
            CancellationReason::Manual => {
                self.manual.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the count of client disconnected cancellations.
    #[must_use]
    pub fn client_disconnected(&self) -> u64 {
        self.client_disconnected.load(Ordering::Relaxed)
    }

    /// Returns the count of timeout cancellations.
    #[must_use]
    pub fn timeout(&self) -> u64 {
        self.timeout.load(Ordering::Relaxed)
    }

    /// Returns the count of server shutdown cancellations.
    #[must_use]
    pub fn server_shutdown(&self) -> u64 {
        self.server_shutdown.load(Ordering::Relaxed)
    }

    /// Returns the count of manual cancellations.
    #[must_use]
    pub fn manual(&self) -> u64 {
        self.manual.load(Ordering::Relaxed)
    }

    /// Returns total cancellations.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.client_disconnected()
            + self.timeout()
            + self.server_shutdown()
            + self.manual()
    }

    /// Renders Prometheus-format metrics.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        format!(
            r#"# HELP infernum_requests_cancelled_total Total cancelled requests by reason
# TYPE infernum_requests_cancelled_total counter
infernum_requests_cancelled_total{{reason="client_disconnected"}} {}
infernum_requests_cancelled_total{{reason="timeout"}} {}
infernum_requests_cancelled_total{{reason="server_shutdown"}} {}
infernum_requests_cancelled_total{{reason="manual"}} {}
"#,
            self.client_disconnected(),
            self.timeout(),
            self.server_shutdown(),
            self.manual(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token_initial_state() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        assert!(!token.is_cancelled());
        assert!(token.reason().is_none());
        assert!(token.error().is_none());
    }

    #[test]
    fn test_cancellation_token_cancelled() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        cancellation.cancel(CancellationReason::ClientDisconnected);

        assert!(token.is_cancelled());
        assert_eq!(token.reason(), Some(CancellationReason::ClientDisconnected));
    }

    #[test]
    fn test_cancellation_token_error() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        cancellation.cancel(CancellationReason::Timeout);

        let error = token.error().expect("should have error");
        assert_eq!(error.reason, CancellationReason::Timeout);
    }

    #[test]
    fn test_cancellation_token_elapsed() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        std::thread::sleep(Duration::from_millis(10));

        assert!(token.elapsed() >= Duration::from_millis(10));
    }

    #[test]
    fn test_multiple_tokens_share_state() {
        let cancellation = RequestCancellation::new();
        let token1 = cancellation.token();
        let token2 = cancellation.token();

        assert!(!token1.is_cancelled());
        assert!(!token2.is_cancelled());

        cancellation.cancel(CancellationReason::Manual);

        assert!(token1.is_cancelled());
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_cancellation_reason_display() {
        assert_eq!(
            CancellationReason::ClientDisconnected.to_string(),
            "client disconnected"
        );
        assert_eq!(CancellationReason::Timeout.to_string(), "timeout");
        assert_eq!(
            CancellationReason::ServerShutdown.to_string(),
            "server shutdown"
        );
        assert_eq!(CancellationReason::Manual.to_string(), "manual cancellation");
    }

    #[test]
    fn test_cancellation_error_display() {
        let error = CancellationError {
            reason: CancellationReason::Timeout,
            elapsed: Duration::from_secs(30),
        };
        let display = error.to_string();
        assert!(display.contains("timeout"));
        assert!(display.contains("30"));
    }

    #[test]
    fn test_request_cancellation_default() {
        let cancellation = RequestCancellation::default();
        assert!(!cancellation.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_debug() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        let debug = format!("{:?}", token);
        assert!(debug.contains("CancellationToken"));
        assert!(debug.contains("is_cancelled"));
    }

    #[test]
    fn test_request_cancellation_debug() {
        let cancellation = RequestCancellation::new();
        let debug = format!("{:?}", cancellation);
        assert!(debug.contains("RequestCancellation"));
        assert!(debug.contains("is_cancelled"));
    }

    #[test]
    fn test_child_token() {
        let cancellation = RequestCancellation::new();
        let parent = cancellation.token();
        let child = parent.child_token();

        assert!(!parent.is_cancelled());
        assert!(!child.is_cancelled());

        cancellation.cancel(CancellationReason::ServerShutdown);

        assert!(parent.is_cancelled());
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_cancellation_reason_code_roundtrip() {
        let reasons = [
            CancellationReason::ClientDisconnected,
            CancellationReason::Timeout,
            CancellationReason::ServerShutdown,
            CancellationReason::Manual,
        ];

        for reason in reasons {
            let code = reason.to_code();
            let recovered = CancellationReason::from_code(code);
            assert_eq!(recovered, Some(reason));
        }
    }

    #[test]
    fn test_cancellation_reason_from_invalid_code() {
        assert!(CancellationReason::from_code(0).is_none());
        assert!(CancellationReason::from_code(100).is_none());
    }

    #[test]
    fn test_cancellation_metrics_new() {
        let metrics = CancellationMetrics::new();
        assert_eq!(metrics.total(), 0);
    }

    #[test]
    fn test_cancellation_metrics_record() {
        let metrics = CancellationMetrics::new();

        metrics.record(CancellationReason::ClientDisconnected);
        metrics.record(CancellationReason::ClientDisconnected);
        metrics.record(CancellationReason::Timeout);
        metrics.record(CancellationReason::Manual);

        assert_eq!(metrics.client_disconnected(), 2);
        assert_eq!(metrics.timeout(), 1);
        assert_eq!(metrics.server_shutdown(), 0);
        assert_eq!(metrics.manual(), 1);
        assert_eq!(metrics.total(), 4);
    }

    #[test]
    fn test_cancellation_metrics_prometheus() {
        let metrics = CancellationMetrics::new();
        metrics.record(CancellationReason::ClientDisconnected);
        metrics.record(CancellationReason::Timeout);

        let output = metrics.render_prometheus();

        assert!(output.contains("infernum_requests_cancelled_total"));
        assert!(output.contains("client_disconnected"));
        assert!(output.contains("timeout"));
    }

    #[tokio::test]
    async fn test_cancellation_token_cancelled_async() {
        let cancellation = RequestCancellation::new();
        let token = cancellation.token();

        // Cancel in background
        let cancel_handle = {
            let c = cancellation.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                c.cancel(CancellationReason::Manual);
            })
        };

        // Wait for cancellation
        let start = Instant::now();
        token.cancelled().await;
        let elapsed = start.elapsed();

        assert!(token.is_cancelled());
        assert!(elapsed >= Duration::from_millis(50));

        let _ = cancel_handle.await;
    }
}
