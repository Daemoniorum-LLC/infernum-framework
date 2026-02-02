//! Observability middleware for HTTP metrics and request tracing.
//!
//! This module provides middleware for:
//! - Request ID generation and propagation
//! - HTTP request/response metrics
//! - Response timing headers
//!
//! # Request IDs
//!
//! Every request gets a unique ID that's:
//! - Extracted from incoming `x-request-id` header (if present)
//! - Generated if not present
//! - Included in all log entries
//! - Returned in the `x-request-id` response header
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::observability::{request_id_layer, ObservabilityState};
//!
//! let state = ObservabilityState::new();
//! let app = Router::new()
//!     .route("/api/test", get(handler))
//!     .layer(request_id_layer(state));
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Request, State},
    http::header::HeaderValue,
    middleware::Next,
    response::Response,
};
use parking_lot::RwLock;
use std::collections::HashMap;

/// Request ID stored in request extensions.
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

impl RequestId {
    /// Creates a new request ID.
    pub fn new() -> Self {
        Self(format!("req-{}", uuid::Uuid::new_v4().as_simple()))
    }

    /// Creates a request ID from an existing string.
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Returns the request ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Observability state for metrics tracking.
#[derive(Clone)]
pub struct ObservabilityState {
    inner: Arc<ObservabilityInner>,
}

struct ObservabilityInner {
    /// HTTP request counts by method, path, and status.
    http_requests: RwLock<HashMap<(String, String, u16), u64>>,
    /// HTTP request latencies by method and path.
    http_latencies: RwLock<HashMap<(String, String), LatencyStats>>,
    /// Total requests served.
    total_requests: AtomicU64,
    /// Total 5xx errors.
    server_errors: AtomicU64,
    /// Total 4xx errors.
    client_errors: AtomicU64,
}

/// Latency statistics for a specific endpoint.
#[derive(Clone, Default)]
struct LatencyStats {
    count: u64,
    sum_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl LatencyStats {
    fn record(&mut self, latency_ms: f64) {
        self.count += 1;
        self.sum_ms += latency_ms;
        if self.count == 1 {
            self.min_ms = latency_ms;
            self.max_ms = latency_ms;
        } else {
            if latency_ms < self.min_ms {
                self.min_ms = latency_ms;
            }
            if latency_ms > self.max_ms {
                self.max_ms = latency_ms;
            }
        }
    }

    #[allow(dead_code)]
    fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_ms / self.count as f64
        }
    }
}

impl Default for ObservabilityState {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservabilityState {
    /// Creates a new observability state.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ObservabilityInner {
                http_requests: RwLock::new(HashMap::new()),
                http_latencies: RwLock::new(HashMap::new()),
                total_requests: AtomicU64::new(0),
                server_errors: AtomicU64::new(0),
                client_errors: AtomicU64::new(0),
            }),
        }
    }

    /// Records an HTTP request.
    fn record_request(&self, method: &str, path: &str, status: u16, latency_ms: f64) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);

        // Track error counts
        if status >= 500 {
            self.inner.server_errors.fetch_add(1, Ordering::Relaxed);
        } else if status >= 400 {
            self.inner.client_errors.fetch_add(1, Ordering::Relaxed);
        }

        // Record request count
        let key = (method.to_string(), normalize_path(path), status);
        {
            let mut requests = self.inner.http_requests.write();
            *requests.entry(key).or_insert(0) += 1;
        }

        // Record latency
        let latency_key = (method.to_string(), normalize_path(path));
        {
            let mut latencies = self.inner.http_latencies.write();
            latencies.entry(latency_key).or_default().record(latency_ms);
        }
    }

    /// Returns total request count.
    pub fn total_requests(&self) -> u64 {
        self.inner.total_requests.load(Ordering::Relaxed)
    }

    /// Returns server error count (5xx).
    pub fn server_errors(&self) -> u64 {
        self.inner.server_errors.load(Ordering::Relaxed)
    }

    /// Returns client error count (4xx).
    pub fn client_errors(&self) -> u64 {
        self.inner.client_errors.load(Ordering::Relaxed)
    }

    /// Renders HTTP metrics in Prometheus format.
    pub fn render_http_metrics(&self) -> String {
        let mut output = String::new();

        // HTTP request totals
        output.push_str("# HELP infernum_http_requests_total Total HTTP requests by method, path, and status.\n");
        output.push_str("# TYPE infernum_http_requests_total counter\n");
        for ((method, path, status), count) in self.inner.http_requests.read().iter() {
            output.push_str(&format!(
                "infernum_http_requests_total{{method=\"{}\",path=\"{}\",status=\"{}\"}} {}\n",
                method, path, status, count
            ));
        }

        // HTTP latency summary
        output.push_str("# HELP infernum_http_duration_ms HTTP request duration in milliseconds.\n");
        output.push_str("# TYPE infernum_http_duration_ms summary\n");
        for ((method, path), stats) in self.inner.http_latencies.read().iter() {
            output.push_str(&format!(
                "infernum_http_duration_ms_sum{{method=\"{}\",path=\"{}\"}} {:.2}\n",
                method, path, stats.sum_ms
            ));
            output.push_str(&format!(
                "infernum_http_duration_ms_count{{method=\"{}\",path=\"{}\"}} {}\n",
                method, path, stats.count
            ));
        }

        // Error totals
        output.push_str("# HELP infernum_http_errors_total Total HTTP errors.\n");
        output.push_str("# TYPE infernum_http_errors_total counter\n");
        output.push_str(&format!(
            "infernum_http_errors_total{{type=\"server\"}} {}\n",
            self.server_errors()
        ));
        output.push_str(&format!(
            "infernum_http_errors_total{{type=\"client\"}} {}\n",
            self.client_errors()
        ));

        output
    }
}

/// Normalizes a path for metrics grouping.
///
/// Replaces dynamic segments (UUIDs, numbers) with placeholders to prevent
/// high-cardinality metrics.
fn normalize_path(path: &str) -> String {
    let mut result = path.to_string();

    // Replace UUIDs with :id placeholder
    if let Ok(re) = regex::Regex::new(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}") {
        result = re.replace_all(&result, "/:id").to_string();
    }

    // Replace numeric segments with :num placeholder
    if let Ok(re) = regex::Regex::new(r"/\d+") {
        result = re.replace_all(&result, "/:num").to_string();
    }

    result
}

/// Request ID middleware.
///
/// Extracts or generates a request ID and includes it in the response.
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    // Extract existing request ID or generate new one
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| RequestId::from_string(s.to_string()))
        .unwrap_or_else(RequestId::new);

    // Store in extensions for handlers
    request.extensions_mut().insert(request_id.clone());

    let mut response = next.run(request).await;

    // Always return request ID in response
    if let Ok(header_value) = HeaderValue::from_str(request_id.as_str()) {
        response.headers_mut().insert("x-request-id", header_value);
    }

    response
}

/// HTTP metrics middleware.
///
/// Records request metrics and adds timing headers.
pub async fn http_metrics_middleware(
    State(state): State<ObservabilityState>,
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let mut response = next.run(request).await;

    let duration = start.elapsed();
    let duration_ms = duration.as_secs_f64() * 1000.0;
    let status = response.status().as_u16();

    // Record metrics
    state.record_request(&method, &path, status, duration_ms);

    // Add timing header
    if let Ok(header_value) = HeaderValue::from_str(&format!("{:.2}", duration_ms)) {
        response.headers_mut().insert("x-response-time-ms", header_value);
    }

    response
}

/// Combined observability middleware.
///
/// Handles both request ID and metrics in a single middleware layer.
pub async fn observability_middleware(
    State(state): State<ObservabilityState>,
    mut request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    // Extract or generate request ID
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| RequestId::from_string(s.to_string()))
        .unwrap_or_else(RequestId::new);

    // Store in extensions for handlers
    request.extensions_mut().insert(request_id.clone());

    // Log request start
    tracing::debug!(
        request_id = %request_id,
        method = %method,
        path = %path,
        "Request started"
    );

    let mut response = next.run(request).await;

    let duration = start.elapsed();
    let duration_ms = duration.as_secs_f64() * 1000.0;
    let status = response.status().as_u16();

    // Record metrics
    state.record_request(&method, &path, status, duration_ms);

    // Log request completion
    tracing::debug!(
        request_id = %request_id,
        method = %method,
        path = %path,
        status = status,
        duration_ms = format!("{:.2}", duration_ms),
        "Request completed"
    );

    // Add response headers
    if let Ok(header_value) = HeaderValue::from_str(request_id.as_str()) {
        response.headers_mut().insert("x-request-id", header_value);
    }
    if let Ok(header_value) = HeaderValue::from_str(&format!("{:.2}", duration_ms)) {
        response.headers_mut().insert("x-response-time-ms", header_value);
    }

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_new() {
        let id = RequestId::new();
        assert!(id.as_str().starts_with("req-"));
        assert_eq!(id.as_str().len(), 36); // "req-" + 32 hex chars
    }

    #[test]
    fn test_request_id_from_string() {
        let id = RequestId::from_string("custom-123".to_string());
        assert_eq!(id.as_str(), "custom-123");
    }

    #[test]
    fn test_observability_state_record() {
        let state = ObservabilityState::new();

        state.record_request("GET", "/health", 200, 5.0);
        state.record_request("POST", "/v1/chat/completions", 200, 150.0);
        state.record_request("POST", "/v1/chat/completions", 500, 50.0);
        state.record_request("GET", "/v1/models", 401, 2.0);

        assert_eq!(state.total_requests(), 4);
        assert_eq!(state.server_errors(), 1);
        assert_eq!(state.client_errors(), 1);
    }

    #[test]
    fn test_latency_stats() {
        let mut stats = LatencyStats::default();

        stats.record(10.0);
        stats.record(20.0);
        stats.record(30.0);

        assert_eq!(stats.count, 3);
        assert!((stats.sum_ms - 60.0).abs() < 0.001);
        assert!((stats.min_ms - 10.0).abs() < 0.001);
        assert!((stats.max_ms - 30.0).abs() < 0.001);
        assert!((stats.avg_ms() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(
            normalize_path("/users/123/posts"),
            "/users/:num/posts"
        );
        assert_eq!(
            normalize_path("/items/550e8400-e29b-41d4-a716-446655440000"),
            "/items/:id"
        );
        assert_eq!(
            normalize_path("/v1/chat/completions"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn test_render_http_metrics() {
        let state = ObservabilityState::new();

        state.record_request("GET", "/health", 200, 5.0);
        state.record_request("POST", "/v1/chat/completions", 200, 150.0);

        let output = state.render_http_metrics();

        assert!(output.contains("infernum_http_requests_total"));
        assert!(output.contains("method=\"GET\""));
        assert!(output.contains("path=\"/health\""));
        assert!(output.contains("status=\"200\""));
        assert!(output.contains("infernum_http_duration_ms"));
    }
}
