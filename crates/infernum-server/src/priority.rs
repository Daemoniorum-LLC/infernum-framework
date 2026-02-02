//! Request priority support for queue ordering.
//!
//! This module provides priority-based request ordering through the `X-Priority`
//! header. Higher priority requests are processed before lower priority ones
//! when request queuing is enabled.
//!
//! # Priority Levels
//!
//! | Level | Name | Use Case |
//! |-------|------|----------|
//! | 1 | High | Interactive user requests |
//! | 2 | Normal | Standard API requests (default) |
//! | 3 | Low | Batch processing |
//! | 4 | Background | Non-urgent background tasks |
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::priority::RequestPriority;
//!
//! // Parse from header value
//! let priority = RequestPriority::from_header_value("high");
//! assert_eq!(priority, RequestPriority::High);
//!
//! // Parse from numeric value
//! let priority = RequestPriority::from_header_value("1");
//! assert_eq!(priority, RequestPriority::High);
//! ```

use std::fmt;
use std::str::FromStr;

use axum::http::HeaderMap;

/// The header name for request priority.
pub const PRIORITY_HEADER: &str = "x-priority";

/// Request priority levels for queue ordering.
///
/// Lower numeric values indicate higher priority. The default priority is `Normal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RequestPriority {
    /// Highest priority (1). For interactive user requests requiring fast response.
    High = 1,

    /// Normal priority (2). Default for standard API requests.
    #[default]
    Normal = 2,

    /// Low priority (3). For batch processing and non-urgent tasks.
    Low = 3,

    /// Background priority (4). For non-urgent background processing.
    Background = 4,
}

impl RequestPriority {
    /// Returns the numeric priority value (1-4).
    #[must_use]
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Creates a priority from a header value.
    ///
    /// Accepts both string names ("high", "normal", "low", "background") and
    /// numeric values ("1", "2", "3", "4"). Case-insensitive.
    ///
    /// Returns `Normal` for invalid or missing values.
    #[must_use]
    pub fn from_header_value(value: &str) -> Self {
        let normalized = value.trim().to_lowercase();

        // Try parsing as numeric first
        if let Ok(num) = normalized.parse::<u8>() {
            return match num {
                1 => Self::High,
                2 => Self::Normal,
                3 => Self::Low,
                4 => Self::Background,
                _ => Self::Normal,
            };
        }

        // Parse as string name
        match normalized.as_str() {
            "high" | "critical" | "urgent" => Self::High,
            "normal" | "default" | "standard" => Self::Normal,
            "low" | "batch" => Self::Low,
            "background" | "bg" | "idle" => Self::Background,
            _ => Self::Normal,
        }
    }

    /// Extracts request priority from HTTP headers.
    ///
    /// Reads the `X-Priority` header and parses it. Returns `Normal` if the
    /// header is missing or invalid.
    #[must_use]
    pub fn from_headers(headers: &HeaderMap) -> Self {
        headers
            .get(PRIORITY_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(Self::from_header_value)
            .unwrap_or_default()
    }

    /// Returns true if this priority should be processed before `other`.
    #[must_use]
    pub fn is_higher_than(&self, other: &Self) -> bool {
        self.as_u8() < other.as_u8()
    }

    /// Returns the string name of this priority level.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
            Self::Background => "background",
        }
    }
}

impl fmt::Display for RequestPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for RequestPriority {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_header_value(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_default() {
        assert_eq!(RequestPriority::default(), RequestPriority::Normal);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(RequestPriority::High < RequestPriority::Normal);
        assert!(RequestPriority::Normal < RequestPriority::Low);
        assert!(RequestPriority::Low < RequestPriority::Background);
    }

    #[test]
    fn test_priority_from_numeric() {
        assert_eq!(RequestPriority::from_header_value("1"), RequestPriority::High);
        assert_eq!(RequestPriority::from_header_value("2"), RequestPriority::Normal);
        assert_eq!(RequestPriority::from_header_value("3"), RequestPriority::Low);
        assert_eq!(RequestPriority::from_header_value("4"), RequestPriority::Background);
        assert_eq!(RequestPriority::from_header_value("5"), RequestPriority::Normal); // Invalid
        assert_eq!(RequestPriority::from_header_value("0"), RequestPriority::Normal); // Invalid
    }

    #[test]
    fn test_priority_from_string() {
        assert_eq!(RequestPriority::from_header_value("high"), RequestPriority::High);
        assert_eq!(RequestPriority::from_header_value("HIGH"), RequestPriority::High);
        assert_eq!(RequestPriority::from_header_value("critical"), RequestPriority::High);
        assert_eq!(RequestPriority::from_header_value("urgent"), RequestPriority::High);

        assert_eq!(RequestPriority::from_header_value("normal"), RequestPriority::Normal);
        assert_eq!(RequestPriority::from_header_value("default"), RequestPriority::Normal);
        assert_eq!(RequestPriority::from_header_value("standard"), RequestPriority::Normal);

        assert_eq!(RequestPriority::from_header_value("low"), RequestPriority::Low);
        assert_eq!(RequestPriority::from_header_value("batch"), RequestPriority::Low);

        assert_eq!(RequestPriority::from_header_value("background"), RequestPriority::Background);
        assert_eq!(RequestPriority::from_header_value("bg"), RequestPriority::Background);
        assert_eq!(RequestPriority::from_header_value("idle"), RequestPriority::Background);
    }

    #[test]
    fn test_priority_from_invalid() {
        assert_eq!(RequestPriority::from_header_value("invalid"), RequestPriority::Normal);
        assert_eq!(RequestPriority::from_header_value(""), RequestPriority::Normal);
        assert_eq!(RequestPriority::from_header_value("   "), RequestPriority::Normal);
    }

    #[test]
    fn test_priority_as_u8() {
        assert_eq!(RequestPriority::High.as_u8(), 1);
        assert_eq!(RequestPriority::Normal.as_u8(), 2);
        assert_eq!(RequestPriority::Low.as_u8(), 3);
        assert_eq!(RequestPriority::Background.as_u8(), 4);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(RequestPriority::High.to_string(), "high");
        assert_eq!(RequestPriority::Normal.to_string(), "normal");
        assert_eq!(RequestPriority::Low.to_string(), "low");
        assert_eq!(RequestPriority::Background.to_string(), "background");
    }

    #[test]
    fn test_priority_is_higher_than() {
        assert!(RequestPriority::High.is_higher_than(&RequestPriority::Normal));
        assert!(RequestPriority::High.is_higher_than(&RequestPriority::Low));
        assert!(RequestPriority::Normal.is_higher_than(&RequestPriority::Low));
        assert!(!RequestPriority::Low.is_higher_than(&RequestPriority::Normal));
        assert!(!RequestPriority::Normal.is_higher_than(&RequestPriority::High));
    }

    #[test]
    fn test_priority_from_str() {
        let priority: RequestPriority = "high".parse().expect("parse should succeed");
        assert_eq!(priority, RequestPriority::High);

        let priority: RequestPriority = "3".parse().expect("parse should succeed");
        assert_eq!(priority, RequestPriority::Low);
    }

    #[test]
    fn test_priority_whitespace_handling() {
        assert_eq!(RequestPriority::from_header_value("  high  "), RequestPriority::High);
        assert_eq!(RequestPriority::from_header_value(" 1 "), RequestPriority::High);
    }

    #[test]
    fn test_priority_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(PRIORITY_HEADER, "high".parse().expect("valid header value"));
        assert_eq!(RequestPriority::from_headers(&headers), RequestPriority::High);

        headers.insert(PRIORITY_HEADER, "3".parse().expect("valid header value"));
        assert_eq!(RequestPriority::from_headers(&headers), RequestPriority::Low);

        let empty_headers = HeaderMap::new();
        assert_eq!(RequestPriority::from_headers(&empty_headers), RequestPriority::Normal);
    }
}
