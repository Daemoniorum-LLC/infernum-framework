//! Per-request timeout configuration via HTTP headers.
//!
//! This module provides support for clients to specify custom timeout durations
//! using the `X-Request-Timeout` header. This allows different requests to have
//! different timeout values based on their expected processing time.
//!
//! # Header Format
//!
//! The `X-Request-Timeout` header accepts a timeout value in seconds:
//!
//! ```text
//! X-Request-Timeout: 30
//! ```
//!
//! Values can include decimals for sub-second precision:
//!
//! ```text
//! X-Request-Timeout: 2.5
//! ```
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::timeout::{RequestTimeout, TimeoutConfig};
//! use axum::http::HeaderMap;
//!
//! let config = TimeoutConfig::default();
//! let headers = HeaderMap::new();
//!
//! let timeout = RequestTimeout::from_headers(&headers, &config);
//! println!("Effective timeout: {:?}", timeout.effective());
//! ```
//!
//! # Timeout Precedence
//!
//! 1. Header value (if valid and within bounds)
//! 2. Default timeout from configuration
//!
//! Header values are clamped to configured min/max bounds for safety.

use std::fmt;
use std::time::Duration;

use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

/// The header name for request timeout.
pub const TIMEOUT_HEADER: &str = "x-request-timeout";

/// Default minimum timeout (1 second).
pub const DEFAULT_MIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Default maximum timeout (10 minutes).
pub const DEFAULT_MAX_TIMEOUT: Duration = Duration::from_secs(600);

/// Default timeout when not specified (2 minutes).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Configuration for request timeouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Default timeout when not specified in header.
    pub default_timeout: Duration,

    /// Minimum allowed timeout (values below are clamped up).
    pub min_timeout: Duration,

    /// Maximum allowed timeout (values above are clamped down).
    pub max_timeout: Duration,

    /// Whether to honor client-specified timeouts.
    pub allow_client_timeout: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: DEFAULT_TIMEOUT,
            min_timeout: DEFAULT_MIN_TIMEOUT,
            max_timeout: DEFAULT_MAX_TIMEOUT,
            allow_client_timeout: true,
        }
    }
}

impl TimeoutConfig {
    /// Creates a configuration that ignores client timeouts.
    pub fn server_only() -> Self {
        Self {
            allow_client_timeout: false,
            ..Default::default()
        }
    }

    /// Creates a configuration with a custom default timeout.
    pub fn with_default(default: Duration) -> Self {
        Self {
            default_timeout: default,
            ..Default::default()
        }
    }

    /// Builder method to set the default timeout.
    pub fn default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Builder method to set the minimum timeout.
    pub fn min_timeout(mut self, timeout: Duration) -> Self {
        self.min_timeout = timeout;
        self
    }

    /// Builder method to set the maximum timeout.
    pub fn max_timeout(mut self, timeout: Duration) -> Self {
        self.max_timeout = timeout;
        self
    }

    /// Builder method to allow/disallow client timeouts.
    pub fn allow_client(mut self, allow: bool) -> Self {
        self.allow_client_timeout = allow;
        self
    }

    /// Clamps a timeout value to the configured bounds.
    pub fn clamp(&self, timeout: Duration) -> Duration {
        timeout.clamp(self.min_timeout, self.max_timeout)
    }

    /// Returns true if a timeout value is within bounds.
    pub fn is_within_bounds(&self, timeout: Duration) -> bool {
        timeout >= self.min_timeout && timeout <= self.max_timeout
    }
}

/// Parsed request timeout information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestTimeout {
    /// The effective timeout to use (after clamping).
    effective: Duration,

    /// The original value from header (if any).
    from_header: Option<Duration>,

    /// Whether the value was clamped.
    was_clamped: bool,
}

impl RequestTimeout {
    /// Creates a request timeout from HTTP headers.
    pub fn from_headers(headers: &HeaderMap, config: &TimeoutConfig) -> Self {
        if !config.allow_client_timeout {
            return Self {
                effective: config.default_timeout,
                from_header: None,
                was_clamped: false,
            };
        }

        let from_header = headers
            .get(TIMEOUT_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| parse_timeout(s));

        match from_header {
            Some(timeout) => {
                let clamped = config.clamp(timeout);
                Self {
                    effective: clamped,
                    from_header: Some(timeout),
                    was_clamped: clamped != timeout,
                }
            },
            None => Self {
                effective: config.default_timeout,
                from_header: None,
                was_clamped: false,
            },
        }
    }

    /// Creates a request timeout with a specific duration.
    pub fn new(timeout: Duration) -> Self {
        Self {
            effective: timeout,
            from_header: Some(timeout),
            was_clamped: false,
        }
    }

    /// Creates a request timeout using the default from config.
    pub fn default_from(config: &TimeoutConfig) -> Self {
        Self {
            effective: config.default_timeout,
            from_header: None,
            was_clamped: false,
        }
    }

    /// Returns the effective timeout to use.
    pub fn effective(&self) -> Duration {
        self.effective
    }

    /// Returns the original header value if present.
    pub fn from_header(&self) -> Option<Duration> {
        self.from_header
    }

    /// Returns true if the timeout was specified via header.
    pub fn is_custom(&self) -> bool {
        self.from_header.is_some()
    }

    /// Returns true if the value was clamped to fit bounds.
    pub fn was_clamped(&self) -> bool {
        self.was_clamped
    }

    /// Returns the timeout in seconds.
    pub fn as_secs(&self) -> u64 {
        self.effective.as_secs()
    }

    /// Returns the timeout in milliseconds.
    pub fn as_millis(&self) -> u128 {
        self.effective.as_millis()
    }
}

impl Default for RequestTimeout {
    fn default() -> Self {
        Self {
            effective: DEFAULT_TIMEOUT,
            from_header: None,
            was_clamped: false,
        }
    }
}

impl fmt::Display for RequestTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.effective.as_secs_f64();
        if self.was_clamped {
            write!(f, "{:.1}s (clamped)", secs)
        } else if self.from_header.is_some() {
            write!(f, "{:.1}s (custom)", secs)
        } else {
            write!(f, "{:.1}s (default)", secs)
        }
    }
}

/// Parses a timeout value from a header string.
///
/// Accepts:
/// - Integers: "30" -> 30 seconds
/// - Decimals: "2.5" -> 2.5 seconds
/// - With suffix: "30s" or "30sec" -> 30 seconds
///
/// Returns `None` for invalid values.
fn parse_timeout(value: &str) -> Option<Duration> {
    let trimmed = value.trim();

    // Remove optional suffix
    let numeric = trimmed
        .strip_suffix("sec")
        .or_else(|| trimmed.strip_suffix('s'))
        .unwrap_or(trimmed)
        .trim();

    // Parse as float
    let secs: f64 = numeric.parse().ok()?;

    // Reject negative or NaN
    if secs < 0.0 || secs.is_nan() {
        return None;
    }

    // Reject infinity
    if secs.is_infinite() {
        return None;
    }

    Some(Duration::from_secs_f64(secs))
}

/// Metrics for timeout tracking.
#[derive(Debug, Default)]
pub struct TimeoutMetrics {
    /// Number of requests with custom timeouts.
    pub custom_timeouts: std::sync::atomic::AtomicU64,

    /// Number of requests with clamped timeouts.
    pub clamped_timeouts: std::sync::atomic::AtomicU64,

    /// Number of requests that timed out.
    pub timeouts: std::sync::atomic::AtomicU64,
}

impl TimeoutMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a request timeout being used.
    pub fn record(&self, timeout: &RequestTimeout) {
        use std::sync::atomic::Ordering;

        if timeout.is_custom() {
            self.custom_timeouts.fetch_add(1, Ordering::Relaxed);
        }
        if timeout.was_clamped() {
            self.clamped_timeouts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a timeout occurrence.
    pub fn record_timeout(&self) {
        use std::sync::atomic::Ordering;
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Renders metrics in Prometheus format.
    pub fn render_prometheus(&self) -> String {
        use std::sync::atomic::Ordering;

        let mut output = String::with_capacity(512);

        output.push_str("# HELP infernum_custom_timeouts_total Requests with custom timeouts\n");
        output.push_str("# TYPE infernum_custom_timeouts_total counter\n");
        output.push_str(&format!(
            "infernum_custom_timeouts_total {}\n",
            self.custom_timeouts.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_clamped_timeouts_total Requests with clamped timeouts\n");
        output.push_str("# TYPE infernum_clamped_timeouts_total counter\n");
        output.push_str(&format!(
            "infernum_clamped_timeouts_total {}\n",
            self.clamped_timeouts.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_request_timeouts_total Request timeout occurrences\n");
        output.push_str("# TYPE infernum_request_timeouts_total counter\n");
        output.push_str(&format!(
            "infernum_request_timeouts_total {}\n",
            self.timeouts.load(Ordering::Relaxed)
        ));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_timeout_config_default() {
        let config = TimeoutConfig::default();

        assert_eq!(config.default_timeout, DEFAULT_TIMEOUT);
        assert_eq!(config.min_timeout, DEFAULT_MIN_TIMEOUT);
        assert_eq!(config.max_timeout, DEFAULT_MAX_TIMEOUT);
        assert!(config.allow_client_timeout);
    }

    #[test]
    fn test_timeout_config_server_only() {
        let config = TimeoutConfig::server_only();
        assert!(!config.allow_client_timeout);
    }

    #[test]
    fn test_timeout_config_with_default() {
        let config = TimeoutConfig::with_default(Duration::from_secs(60));
        assert_eq!(config.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_timeout_config_builder() {
        let config = TimeoutConfig::default()
            .default_timeout(Duration::from_secs(30))
            .min_timeout(Duration::from_secs(5))
            .max_timeout(Duration::from_secs(300))
            .allow_client(false);

        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert_eq!(config.min_timeout, Duration::from_secs(5));
        assert_eq!(config.max_timeout, Duration::from_secs(300));
        assert!(!config.allow_client_timeout);
    }

    #[test]
    fn test_timeout_config_clamp() {
        let config = TimeoutConfig::default()
            .min_timeout(Duration::from_secs(5))
            .max_timeout(Duration::from_secs(60));

        // Below min
        assert_eq!(config.clamp(Duration::from_secs(1)), Duration::from_secs(5));

        // Above max
        assert_eq!(
            config.clamp(Duration::from_secs(100)),
            Duration::from_secs(60)
        );

        // Within bounds
        assert_eq!(
            config.clamp(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn test_timeout_config_is_within_bounds() {
        let config = TimeoutConfig::default()
            .min_timeout(Duration::from_secs(5))
            .max_timeout(Duration::from_secs(60));

        assert!(!config.is_within_bounds(Duration::from_secs(1)));
        assert!(config.is_within_bounds(Duration::from_secs(5)));
        assert!(config.is_within_bounds(Duration::from_secs(30)));
        assert!(config.is_within_bounds(Duration::from_secs(60)));
        assert!(!config.is_within_bounds(Duration::from_secs(61)));
    }

    #[test]
    fn test_parse_timeout_integer() {
        assert_eq!(parse_timeout("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_timeout("  30  "), Some(Duration::from_secs(30)));
        assert_eq!(parse_timeout("0"), Some(Duration::ZERO));
    }

    #[test]
    fn test_parse_timeout_decimal() {
        let timeout = parse_timeout("2.5").expect("should parse");
        assert!((timeout.as_secs_f64() - 2.5).abs() < 0.001);

        let timeout = parse_timeout("0.5").expect("should parse");
        assert!((timeout.as_secs_f64() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_timeout_with_suffix() {
        assert_eq!(parse_timeout("30s"), Some(Duration::from_secs(30)));
        assert_eq!(parse_timeout("30sec"), Some(Duration::from_secs(30)));
        assert_eq!(parse_timeout("  30s  "), Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_timeout_invalid() {
        assert!(parse_timeout("").is_none());
        assert!(parse_timeout("abc").is_none());
        assert!(parse_timeout("-5").is_none());
        assert!(parse_timeout("inf").is_none());
        assert!(parse_timeout("NaN").is_none());
    }

    #[test]
    fn test_request_timeout_from_headers_no_header() {
        let headers = HeaderMap::new();
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        assert_eq!(timeout.effective(), DEFAULT_TIMEOUT);
        assert!(timeout.from_header().is_none());
        assert!(!timeout.is_custom());
        assert!(!timeout.was_clamped());
    }

    #[test]
    fn test_request_timeout_from_headers_with_header() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "30".parse().expect("valid header"));
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        assert_eq!(timeout.effective(), Duration::from_secs(30));
        assert_eq!(timeout.from_header(), Some(Duration::from_secs(30)));
        assert!(timeout.is_custom());
        assert!(!timeout.was_clamped());
    }

    #[test]
    fn test_request_timeout_from_headers_clamped_high() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "1000".parse().expect("valid header")); // Above max
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        assert_eq!(timeout.effective(), DEFAULT_MAX_TIMEOUT);
        assert_eq!(timeout.from_header(), Some(Duration::from_secs(1000)));
        assert!(timeout.is_custom());
        assert!(timeout.was_clamped());
    }

    #[test]
    fn test_request_timeout_from_headers_clamped_low() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "0.1".parse().expect("valid header")); // Below min
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        assert_eq!(timeout.effective(), DEFAULT_MIN_TIMEOUT);
        assert!(timeout.was_clamped());
    }

    #[test]
    fn test_request_timeout_from_headers_client_disabled() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "30".parse().expect("valid header"));
        let config = TimeoutConfig::server_only();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        assert_eq!(timeout.effective(), DEFAULT_TIMEOUT);
        assert!(timeout.from_header().is_none());
        assert!(!timeout.is_custom());
    }

    #[test]
    fn test_request_timeout_new() {
        let timeout = RequestTimeout::new(Duration::from_secs(45));

        assert_eq!(timeout.effective(), Duration::from_secs(45));
        assert_eq!(timeout.from_header(), Some(Duration::from_secs(45)));
        assert!(timeout.is_custom());
    }

    #[test]
    fn test_request_timeout_default_from() {
        let config = TimeoutConfig::with_default(Duration::from_secs(60));
        let timeout = RequestTimeout::default_from(&config);

        assert_eq!(timeout.effective(), Duration::from_secs(60));
        assert!(!timeout.is_custom());
    }

    #[test]
    fn test_request_timeout_default() {
        let timeout = RequestTimeout::default();
        assert_eq!(timeout.effective(), DEFAULT_TIMEOUT);
    }

    #[test]
    fn test_request_timeout_as_secs() {
        let timeout = RequestTimeout::new(Duration::from_secs(30));
        assert_eq!(timeout.as_secs(), 30);
    }

    #[test]
    fn test_request_timeout_as_millis() {
        let timeout = RequestTimeout::new(Duration::from_millis(500));
        assert_eq!(timeout.as_millis(), 500);
    }

    #[test]
    fn test_request_timeout_display_default() {
        let timeout = RequestTimeout::default();
        assert!(timeout.to_string().contains("default"));
    }

    #[test]
    fn test_request_timeout_display_custom() {
        let timeout = RequestTimeout::new(Duration::from_secs(30));
        assert!(timeout.to_string().contains("custom"));
    }

    #[test]
    fn test_request_timeout_display_clamped() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "0.1".parse().expect("valid header"));
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);
        assert!(timeout.to_string().contains("clamped"));
    }

    #[test]
    fn test_timeout_metrics_new() {
        let metrics = TimeoutMetrics::new();

        assert_eq!(metrics.custom_timeouts.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.clamped_timeouts.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.timeouts.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_timeout_metrics_record_custom() {
        let metrics = TimeoutMetrics::new();
        let timeout = RequestTimeout::new(Duration::from_secs(30));

        metrics.record(&timeout);

        assert_eq!(metrics.custom_timeouts.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.clamped_timeouts.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_timeout_metrics_record_clamped() {
        let metrics = TimeoutMetrics::new();

        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "0.1".parse().expect("valid header"));
        let config = TimeoutConfig::default();
        let timeout = RequestTimeout::from_headers(&headers, &config);

        metrics.record(&timeout);

        assert_eq!(metrics.custom_timeouts.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.clamped_timeouts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_timeout_metrics_record_timeout() {
        let metrics = TimeoutMetrics::new();

        metrics.record_timeout();
        metrics.record_timeout();

        assert_eq!(metrics.timeouts.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_timeout_metrics_prometheus() {
        let metrics = TimeoutMetrics::new();
        metrics.custom_timeouts.store(10, Ordering::Relaxed);
        metrics.clamped_timeouts.store(2, Ordering::Relaxed);
        metrics.timeouts.store(5, Ordering::Relaxed);

        let output = metrics.render_prometheus();

        assert!(output.contains("infernum_custom_timeouts_total 10"));
        assert!(output.contains("infernum_clamped_timeouts_total 2"));
        assert!(output.contains("infernum_request_timeouts_total 5"));
    }

    #[test]
    fn test_timeout_header_constant() {
        assert_eq!(TIMEOUT_HEADER, "x-request-timeout");
    }

    #[test]
    fn test_parse_timeout_large_value() {
        // Should parse but will be clamped by config
        let timeout = parse_timeout("999999").expect("should parse");
        assert_eq!(timeout, Duration::from_secs(999999));
    }

    #[test]
    fn test_request_timeout_equality() {
        let t1 = RequestTimeout::new(Duration::from_secs(30));
        let t2 = RequestTimeout::new(Duration::from_secs(30));
        let t3 = RequestTimeout::new(Duration::from_secs(60));

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_request_timeout_invalid_header() {
        let mut headers = HeaderMap::new();
        headers.insert(TIMEOUT_HEADER, "invalid".parse().expect("valid header"));
        let config = TimeoutConfig::default();

        let timeout = RequestTimeout::from_headers(&headers, &config);

        // Should fall back to default
        assert_eq!(timeout.effective(), DEFAULT_TIMEOUT);
        assert!(!timeout.is_custom());
    }
}
