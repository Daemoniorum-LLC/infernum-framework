//! Configuration error types with detailed error messages.
//!
//! Provides structured errors for configuration validation failures,
//! ensuring operators receive clear, actionable error messages.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::config_error::ConfigError;
//!
//! // Report an invalid value
//! let err = ConfigError::invalid_value("INFERNUM_PORT", "integer 1-65535", "abc");
//! assert!(err.to_string().contains("INFERNUM_PORT"));
//! ```

use thiserror::Error;

/// Configuration errors with detailed context.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// An environment variable or config value is invalid.
    #[error("Invalid value for {var}: expected {expected}, got '{value}'")]
    InvalidValue {
        /// The configuration variable name.
        var: String,
        /// Description of expected format.
        expected: String,
        /// The actual invalid value.
        value: String,
    },

    /// A required configuration is missing.
    #[error("Missing required configuration: {var}")]
    MissingRequired {
        /// The missing variable name.
        var: String,
    },

    /// A value is outside the allowed range.
    #[error("Value out of range for {var}: {value} (must be {min} to {max})")]
    OutOfRange {
        /// The configuration variable name.
        var: String,
        /// The actual value.
        value: String,
        /// Minimum allowed value.
        min: String,
        /// Maximum allowed value.
        max: String,
    },

    /// A value has invalid format.
    #[error("Invalid format for {var}: {message}")]
    InvalidFormat {
        /// The configuration variable name.
        var: String,
        /// Description of the format error.
        message: String,
    },

    /// Configuration file not found.
    #[error("Configuration file not found: {path}")]
    FileNotFound {
        /// The path that was not found.
        path: String,
    },

    /// Failed to parse configuration file.
    #[error("Failed to parse config file: {message}")]
    ParseError {
        /// Description of the parse error.
        message: String,
    },

    /// Multiple configuration settings conflict.
    #[error("Conflicting configuration: {message}")]
    Conflict {
        /// Description of the conflict.
        message: String,
    },

    /// Multiple configuration errors occurred.
    #[error("Multiple configuration errors:\n{}", .0.iter().map(|e| format!("  - {}", e)).collect::<Vec<_>>().join("\n"))]
    Multiple(Vec<ConfigError>),
}

impl ConfigError {
    /// Creates an invalid value error.
    pub fn invalid_value(var: &str, expected: &str, value: &str) -> Self {
        Self::InvalidValue {
            var: var.to_string(),
            expected: expected.to_string(),
            value: value.to_string(),
        }
    }

    /// Creates a missing required error.
    pub fn missing_required(var: &str) -> Self {
        Self::MissingRequired {
            var: var.to_string(),
        }
    }

    /// Creates an out of range error.
    pub fn out_of_range<T: std::fmt::Display>(var: &str, value: T, min: T, max: T) -> Self {
        Self::OutOfRange {
            var: var.to_string(),
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
        }
    }

    /// Creates an invalid format error.
    pub fn invalid_format(var: &str, message: &str) -> Self {
        Self::InvalidFormat {
            var: var.to_string(),
            message: message.to_string(),
        }
    }

    /// Creates a conflict error.
    pub fn conflict(message: &str) -> Self {
        Self::Conflict {
            message: message.to_string(),
        }
    }

    /// Creates a file not found error.
    pub fn file_not_found(path: &str) -> Self {
        Self::FileNotFound {
            path: path.to_string(),
        }
    }

    /// Creates a parse error.
    pub fn parse_error(message: &str) -> Self {
        Self::ParseError {
            message: message.to_string(),
        }
    }

    /// Combines multiple errors into a single error.
    pub fn multiple(errors: Vec<ConfigError>) -> Self {
        match errors.len() {
            0 => Self::ParseError {
                message: "No errors provided".to_string(),
            },
            1 => {
                // SAFETY: We just checked that len() == 1, so next() will always return Some
                errors
                    .into_iter()
                    .next()
                    .expect("vector has exactly one element")
            },
            _ => Self::Multiple(errors),
        }
    }

    /// Returns whether this error indicates a retryable condition.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::FileNotFound { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_value_error() {
        let err = ConfigError::invalid_value("INFERNUM_PORT", "integer 1-65535", "abc");
        assert_eq!(
            err.to_string(),
            "Invalid value for INFERNUM_PORT: expected integer 1-65535, got 'abc'"
        );
    }

    #[test]
    fn test_out_of_range_error() {
        let err = ConfigError::out_of_range("INFERNUM_PORT", 0, 1, 65535);
        assert_eq!(
            err.to_string(),
            "Value out of range for INFERNUM_PORT: 0 (must be 1 to 65535)"
        );
    }

    #[test]
    fn test_missing_required_error() {
        let err = ConfigError::missing_required("INFERNUM_API_KEY");
        assert_eq!(
            err.to_string(),
            "Missing required configuration: INFERNUM_API_KEY"
        );
    }

    #[test]
    fn test_conflict_error() {
        let err = ConfigError::conflict("TLS enabled but using port 80");
        assert_eq!(
            err.to_string(),
            "Conflicting configuration: TLS enabled but using port 80"
        );
    }

    #[test]
    fn test_multiple_errors() {
        let errors = vec![
            ConfigError::invalid_value("INFERNUM_PORT", "integer", "abc"),
            ConfigError::out_of_range("INFERNUM_MAX_CONCURRENT", 0, 1, 10000),
        ];
        let err = ConfigError::multiple(errors);
        let msg = err.to_string();
        assert!(msg.contains("Multiple configuration errors"));
        assert!(msg.contains("INFERNUM_PORT"));
        assert!(msg.contains("INFERNUM_MAX_CONCURRENT"));
    }

    #[test]
    fn test_single_error_from_multiple() {
        let errors = vec![ConfigError::missing_required("FOO")];
        let err = ConfigError::multiple(errors);
        // Single error should not wrap in Multiple
        assert!(matches!(err, ConfigError::MissingRequired { .. }));
    }

    #[test]
    fn test_file_not_found() {
        let err = ConfigError::file_not_found("/etc/infernum/config.toml");
        assert!(err.to_string().contains("/etc/infernum/config.toml"));
        assert!(err.is_retryable());
    }

    #[test]
    fn test_parse_error() {
        let err = ConfigError::parse_error("unexpected token at line 5");
        assert!(err.to_string().contains("unexpected token"));
        assert!(!err.is_retryable());
    }
}
