//! Security middleware for the inference server.
//!
//! This module provides security features for the HTTP server including:
//!
//! - **Rate Limiting**: Protect against abuse with configurable request limits
//! - **Security Headers**: CSP, X-Frame-Options, HSTS, and other security headers
//! - **CORS**: Cross-Origin Resource Sharing configuration
//!
//! # Rate Limiting
//!
//! Rate limiting can be configured per-IP or per-API-key:
//!
//! ```rust,ignore
//! use infernum_server::security::{RateLimiter, RateLimitConfig};
//! use std::time::Duration;
//!
//! // Create a rate limiter with custom limits
//! let config = RateLimitConfig::new(100, Duration::from_secs(60));
//! let limiter = RateLimiter::new(config);
//!
//! // Check if a request is allowed
//! match limiter.check("client-key").await {
//!     RateLimitResult::Allowed { remaining, .. } => {
//!         println!("{} requests remaining", remaining);
//!     }
//!     RateLimitResult::Exceeded { retry_after } => {
//!         println!("Rate limited, retry after {:?}", retry_after);
//!     }
//! }
//! ```
//!
//! # Security Headers
//!
//! Apply security headers using middleware:
//!
//! ```rust,ignore
//! use infernum_server::security::SecurityHeadersConfig;
//!
//! // For API-only servers (no browser content)
//! let config = SecurityHeadersConfig::api_only();
//!
//! // For servers with HTTPS
//! let config = SecurityHeadersConfig::default().with_https();
//! ```
//!
//! # CORS Configuration
//!
//! Configure CORS for browser-based API access:
//!
//! ```rust,ignore
//! use infernum_server::security::CorsConfig;
//!
//! // Permissive (allow any origin)
//! let cors = CorsConfig::permissive();
//!
//! // Restrictive (specific origins only)
//! let cors = CorsConfig::restrictive(vec![
//!     "https://app.example.com".to_string(),
//! ]);
//!
//! // From environment variable
//! // export INFERNUM_CORS_ORIGINS=https://a.com,https://b.com
//! let cors = CorsConfig::from_env();
//! ```

use axum::{
    extract::{ConnectInfo, Request},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

// ============================================================================
// Rate Limit Metrics (Sprint 6 Day 20.7)
// ============================================================================

/// Metrics for rate limiting events.
///
/// Tracks rate limit hits for Prometheus monitoring.
#[derive(Debug, Default)]
pub struct RateLimitMetrics {
    /// Total rate limit hits.
    pub hits_total: AtomicU64,
    /// Rate limit hits by IP address.
    pub hits_by_ip: AtomicU64,
    /// Rate limit hits by API key.
    pub hits_by_api_key: AtomicU64,
}

impl RateLimitMetrics {
    /// Creates a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a rate limit hit.
    pub fn record_hit(&self, by_api_key: bool) {
        self.hits_total.fetch_add(1, Ordering::Relaxed);
        if by_api_key {
            self.hits_by_api_key.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hits_by_ip.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the total number of rate limit hits.
    pub fn total_hits(&self) -> u64 {
        self.hits_total.load(Ordering::Relaxed)
    }

    /// Renders metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_rate_limit_hits_total Total rate limit hits.\n");
        output.push_str("# TYPE infernum_rate_limit_hits_total counter\n");
        output.push_str(&format!(
            "infernum_rate_limit_hits_total{{identifier=\"ip\"}} {}\n",
            self.hits_by_ip.load(Ordering::Relaxed)
        ));
        output.push_str(&format!(
            "infernum_rate_limit_hits_total{{identifier=\"api_key\"}} {}\n",
            self.hits_by_api_key.load(Ordering::Relaxed)
        ));

        output
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Rate limit configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitConfig {
    /// Maximum requests per window.
    pub max_requests: u32,
    /// Time window duration.
    pub window: Duration,
    /// Whether rate limiting is enabled.
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Creates a disabled rate limit config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Creates a rate limit config with custom limits.
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            enabled: true,
        }
    }

    /// High-throughput config (1000 requests/minute).
    pub fn high_throughput() -> Self {
        Self {
            max_requests: 1000,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }

    /// Strict config (10 requests/minute).
    pub fn strict() -> Self {
        Self {
            max_requests: 10,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }

    /// Creates a rate limit config from environment variables.
    ///
    /// Reads:
    /// - `INFERNUM_RATE_LIMIT_ENABLED`: Whether to enable rate limiting (default: true)
    /// - `INFERNUM_RATE_LIMIT_REQUESTS`: Max requests per window (default: 100)
    /// - `INFERNUM_RATE_LIMIT_WINDOW_SECS`: Window duration in seconds (default: 60)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(enabled) = std::env::var("INFERNUM_RATE_LIMIT_ENABLED") {
            config.enabled = enabled.to_lowercase() == "true" || enabled == "1";
        }

        if let Ok(max) = std::env::var("INFERNUM_RATE_LIMIT_REQUESTS") {
            if let Ok(n) = max.parse::<u32>() {
                config.max_requests = n;
            }
        }

        if let Ok(window) = std::env::var("INFERNUM_RATE_LIMIT_WINDOW_SECS") {
            if let Ok(secs) = window.parse::<u64>() {
                config.window = Duration::from_secs(secs);
            }
        }

        config
    }
}

/// Rate limit entry for tracking requests.
#[derive(Debug, Clone)]
struct RateLimitEntry {
    count: u32,
    window_start: Instant,
}

/// Rate limiter state.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    config: RateLimitConfig,
    entries: Arc<RwLock<HashMap<String, RateLimitEntry>>>,
    metrics: Arc<RateLimitMetrics>,
}

impl RateLimiter {
    /// Creates a new rate limiter with the given config.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RateLimitMetrics::new()),
        }
    }

    /// Returns a reference to the metrics.
    pub fn metrics(&self) -> &RateLimitMetrics {
        &self.metrics
    }

    /// Checks if a request is allowed for the given key (IP or API key).
    pub async fn check(&self, key: &str) -> RateLimitResult {
        if !self.config.enabled {
            return RateLimitResult::Allowed {
                remaining: u32::MAX,
                reset: Duration::from_secs(0),
            };
        }

        let mut entries = self.entries.write().await;
        let now = Instant::now();

        let entry = entries.entry(key.to_string()).or_insert(RateLimitEntry {
            count: 0,
            window_start: now,
        });

        // Check if window has expired
        if now.duration_since(entry.window_start) >= self.config.window {
            entry.count = 0;
            entry.window_start = now;
        }

        // Check if limit exceeded
        if entry.count >= self.config.max_requests {
            let reset = self.config.window - now.duration_since(entry.window_start);
            return RateLimitResult::Exceeded { retry_after: reset };
        }

        // Increment counter
        entry.count += 1;
        let remaining = self.config.max_requests - entry.count;
        let reset = self.config.window - now.duration_since(entry.window_start);

        RateLimitResult::Allowed { remaining, reset }
    }

    /// Cleans up expired entries to prevent memory growth.
    pub async fn cleanup(&self) {
        let mut entries = self.entries.write().await;
        let now = Instant::now();

        entries.retain(|_, entry| now.duration_since(entry.window_start) < self.config.window * 2);
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

/// Result of a rate limit check.
#[derive(Debug)]
pub enum RateLimitResult {
    /// Request is allowed.
    Allowed {
        /// Remaining requests in window.
        remaining: u32,
        /// Time until window reset.
        reset: Duration,
    },
    /// Rate limit exceeded.
    Exceeded {
        /// Time until rate limit resets.
        retry_after: Duration,
    },
}

/// Rate limit error response.
#[derive(Debug, Serialize)]
struct RateLimitError {
    error: RateLimitErrorDetail,
}

#[derive(Debug, Serialize)]
struct RateLimitErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    retry_after_seconds: u64,
}

/// Extracts the client identifier for rate limiting.
fn get_rate_limit_key(request: &Request) -> String {
    // Try to get API key first (for per-key limits)
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return format!("key:{}", token.trim());
            }
        }
    }

    // Fall back to IP address
    if let Some(forwarded) = request.headers().get("x-forwarded-for") {
        if let Ok(ip_str) = forwarded.to_str() {
            if let Some(first_ip) = ip_str.split(',').next() {
                return format!("ip:{}", first_ip.trim());
            }
        }
    }

    // Try to get from connection info
    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return format!("ip:{}", addr.ip());
    }

    // Default fallback
    "ip:unknown".to_string()
}

/// Rate limiting middleware.
pub async fn rate_limit_middleware(
    rate_limiter: RateLimiter,
    request: Request,
    next: Next,
) -> Response {
    let key = get_rate_limit_key(&request);
    let is_api_key = key.starts_with("sk-");

    match rate_limiter.check(&key).await {
        RateLimitResult::Allowed { remaining, reset } => {
            let mut response = next.run(request).await;

            // Add rate limit headers
            let headers = response.headers_mut();
            headers.insert(
                "x-ratelimit-remaining",
                HeaderValue::from_str(&remaining.to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );
            headers.insert(
                "x-ratelimit-reset",
                HeaderValue::from_str(&reset.as_secs().to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("0")),
            );

            response
        },
        RateLimitResult::Exceeded { retry_after } => {
            // Record rate limit hit in metrics
            rate_limiter.metrics().record_hit(is_api_key);

            let error = RateLimitError {
                error: RateLimitErrorDetail {
                    message: "Rate limit exceeded. Please slow down your requests.".to_string(),
                    error_type: "rate_limit_error".to_string(),
                    retry_after_seconds: retry_after.as_secs(),
                },
            };

            let mut response = (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response();
            response.headers_mut().insert(
                "retry-after",
                HeaderValue::from_str(&retry_after.as_secs().to_string())
                    .unwrap_or_else(|_| HeaderValue::from_static("60")),
            );
            response
        },
    }
}

// ============================================================================
// Security Headers
// ============================================================================

/// Security headers configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SecurityHeadersConfig {
    /// Content Security Policy.
    pub content_security_policy: Option<String>,
    /// X-Frame-Options.
    pub x_frame_options: Option<String>,
    /// X-Content-Type-Options.
    pub x_content_type_options: bool,
    /// X-XSS-Protection.
    pub x_xss_protection: bool,
    /// Strict-Transport-Security.
    pub strict_transport_security: Option<String>,
    /// Referrer-Policy.
    pub referrer_policy: Option<String>,
}

impl Default for SecurityHeadersConfig {
    fn default() -> Self {
        Self {
            content_security_policy: Some("default-src 'self'".to_string()),
            x_frame_options: Some("DENY".to_string()),
            x_content_type_options: true,
            x_xss_protection: true,
            strict_transport_security: None, // Only set when using HTTPS
            referrer_policy: Some("strict-origin-when-cross-origin".to_string()),
        }
    }
}

impl SecurityHeadersConfig {
    /// Creates a config for API-only servers (no browser content).
    pub fn api_only() -> Self {
        Self {
            content_security_policy: None,
            x_frame_options: Some("DENY".to_string()),
            x_content_type_options: true,
            x_xss_protection: false,
            strict_transport_security: None,
            referrer_policy: None,
        }
    }

    /// Creates a config with HTTPS enabled.
    pub fn with_https(mut self) -> Self {
        self.strict_transport_security = Some("max-age=31536000; includeSubDomains".to_string());
        self
    }
}

/// Security headers middleware.
pub async fn security_headers_middleware(
    config: SecurityHeadersConfig,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    if let Some(csp) = &config.content_security_policy {
        if let Ok(value) = HeaderValue::from_str(csp) {
            headers.insert("content-security-policy", value);
        }
    }

    if let Some(xfo) = &config.x_frame_options {
        if let Ok(value) = HeaderValue::from_str(xfo) {
            headers.insert("x-frame-options", value);
        }
    }

    if config.x_content_type_options {
        headers.insert(
            "x-content-type-options",
            HeaderValue::from_static("nosniff"),
        );
    }

    if config.x_xss_protection {
        headers.insert(
            "x-xss-protection",
            HeaderValue::from_static("1; mode=block"),
        );
    }

    if let Some(hsts) = &config.strict_transport_security {
        if let Ok(value) = HeaderValue::from_str(hsts) {
            headers.insert("strict-transport-security", value);
        }
    }

    if let Some(rp) = &config.referrer_policy {
        if let Ok(value) = HeaderValue::from_str(rp) {
            headers.insert("referrer-policy", value);
        }
    }

    response
}

// ============================================================================
// CORS Configuration
// ============================================================================

/// CORS configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CorsConfig {
    /// Allowed origins (None = any).
    pub allowed_origins: Option<Vec<String>>,
    /// Allowed methods.
    pub allowed_methods: Vec<Method>,
    /// Allowed headers.
    pub allowed_headers: Vec<String>,
    /// Whether credentials are allowed.
    pub allow_credentials: bool,
    /// Max age for preflight cache.
    pub max_age: Duration,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: None, // Any origin
            allowed_methods: vec![
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::OPTIONS,
            ],
            allowed_headers: vec![
                "content-type".to_string(),
                "authorization".to_string(),
                "x-api-key".to_string(),
            ],
            allow_credentials: false,
            max_age: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl CorsConfig {
    /// Creates a permissive CORS config (any origin).
    pub fn permissive() -> Self {
        Self::default()
    }

    /// Creates a restrictive CORS config with specific origins.
    pub fn restrictive(origins: Vec<String>) -> Self {
        Self {
            allowed_origins: Some(origins),
            allow_credentials: true,
            ..Default::default()
        }
    }

    /// Creates a CORS config from environment variable.
    ///
    /// Reads `INFERNUM_CORS_ORIGINS` as comma-separated origins.
    pub fn from_env() -> Self {
        if let Ok(origins_str) = std::env::var("INFERNUM_CORS_ORIGINS") {
            let origins: Vec<String> = origins_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if !origins.is_empty() {
                return Self::restrictive(origins);
            }
        }

        Self::permissive()
    }

    /// Converts to tower-http CorsLayer.
    pub fn into_layer(self) -> CorsLayer {
        let mut layer = CorsLayer::new()
            .allow_methods(self.allowed_methods)
            .max_age(self.max_age);

        // Set origins
        if let Some(origins) = self.allowed_origins {
            let origins: Vec<HeaderValue> = origins
                .iter()
                .filter_map(|o| HeaderValue::from_str(o).ok())
                .collect();
            layer = layer.allow_origin(origins);
        } else {
            layer = layer.allow_origin(Any);
        }

        // Set headers
        let headers: Vec<header::HeaderName> = self
            .allowed_headers
            .iter()
            .filter_map(|h| header::HeaderName::try_from(h.as_str()).ok())
            .collect();
        layer = layer.allow_headers(headers);

        // Set credentials
        if self.allow_credentials {
            layer = layer.allow_credentials(true);
        }

        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Rate Limit Tests ===

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_requests, 100);
        assert_eq!(config.window, Duration::from_secs(60));
        assert!(config.enabled);
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_rate_limit_config_high_throughput() {
        let config = RateLimitConfig::high_throughput();
        assert_eq!(config.max_requests, 1000);
        assert!(config.enabled);
    }

    #[test]
    fn test_rate_limit_config_strict() {
        let config = RateLimitConfig::strict();
        assert_eq!(config.max_requests, 10);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_rate_limiter_allows_requests() {
        let limiter = RateLimiter::new(RateLimitConfig::new(5, Duration::from_secs(60)));

        for _ in 0..5 {
            match limiter.check("test-key").await {
                RateLimitResult::Allowed { .. } => {},
                RateLimitResult::Exceeded { .. } => panic!("Should not exceed limit"),
            }
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_excess() {
        let limiter = RateLimiter::new(RateLimitConfig::new(2, Duration::from_secs(60)));

        // First two should succeed
        assert!(matches!(
            limiter.check("test-key").await,
            RateLimitResult::Allowed { .. }
        ));
        assert!(matches!(
            limiter.check("test-key").await,
            RateLimitResult::Allowed { .. }
        ));

        // Third should be blocked
        assert!(matches!(
            limiter.check("test-key").await,
            RateLimitResult::Exceeded { .. }
        ));
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let limiter = RateLimiter::new(RateLimitConfig::disabled());

        // Should always allow when disabled
        for _ in 0..1000 {
            match limiter.check("test-key").await {
                RateLimitResult::Allowed { remaining, .. } => {
                    assert_eq!(remaining, u32::MAX);
                },
                RateLimitResult::Exceeded { .. } => panic!("Should not exceed when disabled"),
            }
        }
    }

    // === Security Headers Tests ===

    #[test]
    fn test_security_headers_default() {
        let config = SecurityHeadersConfig::default();
        assert!(config.content_security_policy.is_some());
        assert!(config.x_frame_options.is_some());
        assert!(config.x_content_type_options);
        assert!(config.x_xss_protection);
    }

    #[test]
    fn test_security_headers_api_only() {
        let config = SecurityHeadersConfig::api_only();
        assert!(config.content_security_policy.is_none());
        assert!(!config.x_xss_protection);
    }

    #[test]
    fn test_security_headers_with_https() {
        let config = SecurityHeadersConfig::default().with_https();
        assert!(config.strict_transport_security.is_some());
    }

    // === CORS Tests ===

    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.allowed_origins.is_none());
        assert!(!config.allow_credentials);
    }

    #[test]
    fn test_cors_config_permissive() {
        let config = CorsConfig::permissive();
        assert!(config.allowed_origins.is_none());
    }

    #[test]
    fn test_cors_config_restrictive() {
        let config = CorsConfig::restrictive(vec![
            "https://example.com".to_string(),
            "https://app.example.com".to_string(),
        ]);
        assert!(config.allowed_origins.is_some());
        assert_eq!(config.allowed_origins.as_ref().unwrap().len(), 2);
        assert!(config.allow_credentials);
    }

    #[test]
    fn test_cors_into_layer() {
        let config = CorsConfig::default();
        let _layer = config.into_layer();
        // Just verify it compiles and doesn't panic
    }

    // ========================================================================
    // Sprint 6 Day 20.7: Rate Limit Metrics Tests
    // ========================================================================

    #[test]
    fn test_rate_limit_metrics_new() {
        let metrics = RateLimitMetrics::new();
        assert_eq!(metrics.total_hits(), 0);
    }

    #[test]
    fn test_rate_limit_metrics_record_hits() {
        let metrics = RateLimitMetrics::new();

        metrics.record_hit(false); // IP-based
        metrics.record_hit(false); // IP-based
        metrics.record_hit(true); // API key-based

        assert_eq!(metrics.total_hits(), 3);
        assert_eq!(metrics.hits_by_ip.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.hits_by_api_key.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_rate_limit_metrics_prometheus_format() {
        let metrics = RateLimitMetrics::new();
        metrics.record_hit(false);
        metrics.record_hit(true);
        metrics.record_hit(true);

        let output = metrics.render_prometheus();

        assert!(output.contains("# HELP infernum_rate_limit_hits_total"));
        assert!(output.contains("# TYPE infernum_rate_limit_hits_total counter"));
        assert!(output.contains("infernum_rate_limit_hits_total{identifier=\"ip\"} 1"));
        assert!(output.contains("infernum_rate_limit_hits_total{identifier=\"api_key\"} 2"));
    }

    #[test]
    fn test_rate_limiter_has_metrics() {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        assert_eq!(limiter.metrics().total_hits(), 0);
    }
}
