//! Error types for the Infernum ecosystem.

use std::time::Duration;
use thiserror::Error;

/// Result type alias using [`enum@Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for the Infernum ecosystem.
#[derive(Error, Debug)]
pub enum Error {
    /// Model was not found in the registry.
    #[error("Model not found: {model_id}")]
    ModelNotFound {
        /// The requested model identifier.
        model_id: String,
    },

    /// Model architecture is not supported.
    #[error("Unsupported model architecture: {architecture}")]
    UnsupportedArchitecture {
        /// The unsupported architecture name.
        architecture: String,
    },

    /// Out of memory during inference.
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Bytes requested.
        requested: usize,
        /// Bytes available.
        available: usize,
    },

    /// Context length exceeded for the model.
    #[error("Context length exceeded: {current} tokens > {max} max tokens")]
    ContextLengthExceeded {
        /// Current token count.
        current: u32,
        /// Maximum allowed tokens.
        max: u32,
    },

    /// Invalid configuration provided.
    #[error("Invalid configuration: {message}")]
    InvalidConfig {
        /// Description of the configuration error.
        message: String,
    },

    /// Backend-specific error.
    #[error("Backend error: {message}")]
    Backend {
        /// Backend name (cuda, metal, cpu, etc.).
        backend: String,
        /// Error message.
        message: String,
    },

    /// Operation timed out.
    #[error("Operation timed out after {duration:?}")]
    Timeout {
        /// Duration before timeout.
        duration: Duration,
    },

    /// Rate limited by the system.
    #[error("Rate limited: retry after {retry_after:?}")]
    RateLimited {
        /// Duration to wait before retrying.
        retry_after: Duration,
    },

    /// Tokenization error.
    #[error("Tokenization error: {message}")]
    Tokenization {
        /// Error message.
        message: String,
    },

    /// Model loading error.
    #[error("Failed to load model: {message}")]
    ModelLoad {
        /// Error message.
        message: String,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Internal error (unexpected state).
    #[error("Internal error: {message}")]
    Internal {
        /// Error message.
        message: String,
    },
}

impl Error {
    /// Returns `true` if this error is retryable.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Timeout { .. } | Self::RateLimited { .. })
    }

    /// Returns `true` if this error is due to resource exhaustion.
    #[must_use]
    pub fn is_resource_exhaustion(&self) -> bool {
        matches!(
            self,
            Self::OutOfMemory { .. } | Self::ContextLengthExceeded { .. }
        )
    }

    /// Creates an internal error with the given message.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Creates a backend error with the given backend name and message.
    #[must_use]
    pub fn backend(backend: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Backend {
            backend: backend.into(),
            message: message.into(),
        }
    }

    /// Creates a model load error.
    #[must_use]
    pub fn model_load(message: impl Into<String>) -> Self {
        Self::ModelLoad {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_not_found_error() {
        let error = Error::ModelNotFound {
            model_id: "llama-7b".to_string(),
        };

        let msg = error.to_string();
        assert!(msg.contains("Model not found"));
        assert!(msg.contains("llama-7b"));
        assert!(!error.is_retryable());
        assert!(!error.is_resource_exhaustion());
    }

    #[test]
    fn test_unsupported_architecture_error() {
        let error = Error::UnsupportedArchitecture {
            architecture: "mamba".to_string(),
        };

        let msg = error.to_string();
        assert!(msg.contains("Unsupported model architecture"));
        assert!(msg.contains("mamba"));
    }

    #[test]
    fn test_out_of_memory_error() {
        let error = Error::OutOfMemory {
            requested: 16_000_000_000,
            available: 8_000_000_000,
        };

        let msg = error.to_string();
        assert!(msg.contains("Out of memory"));
        assert!(msg.contains("16000000000"));
        assert!(msg.contains("8000000000"));

        assert!(!error.is_retryable());
        assert!(error.is_resource_exhaustion());
    }

    #[test]
    fn test_context_length_exceeded_error() {
        let error = Error::ContextLengthExceeded {
            current: 32000,
            max: 8192,
        };

        let msg = error.to_string();
        assert!(msg.contains("Context length exceeded"));
        assert!(msg.contains("32000"));
        assert!(msg.contains("8192"));

        assert!(!error.is_retryable());
        assert!(error.is_resource_exhaustion());
    }

    #[test]
    fn test_invalid_config_error() {
        let error = Error::InvalidConfig {
            message: "temperature must be positive".to_string(),
        };

        let msg = error.to_string();
        assert!(msg.contains("Invalid configuration"));
        assert!(msg.contains("temperature must be positive"));
    }

    #[test]
    fn test_backend_error() {
        let error = Error::backend("cuda", "CUDA out of memory");

        let msg = error.to_string();
        assert!(msg.contains("Backend error"));
        assert!(msg.contains("CUDA out of memory"));

        match error {
            Error::Backend { backend, message } => {
                assert_eq!(backend, "cuda");
                assert_eq!(message, "CUDA out of memory");
            },
            _ => panic!("Expected Backend error"),
        }
    }

    #[test]
    fn test_timeout_error() {
        let error = Error::Timeout {
            duration: Duration::from_secs(30),
        };

        let msg = error.to_string();
        assert!(msg.contains("timed out"));
        assert!(msg.contains("30"));

        assert!(error.is_retryable());
        assert!(!error.is_resource_exhaustion());
    }

    #[test]
    fn test_rate_limited_error() {
        let error = Error::RateLimited {
            retry_after: Duration::from_secs(60),
        };

        let msg = error.to_string();
        assert!(msg.contains("Rate limited"));
        assert!(msg.contains("60"));

        assert!(error.is_retryable());
        assert!(!error.is_resource_exhaustion());
    }

    #[test]
    fn test_tokenization_error() {
        let error = Error::Tokenization {
            message: "unknown token".to_string(),
        };

        let msg = error.to_string();
        assert!(msg.contains("Tokenization error"));
        assert!(msg.contains("unknown token"));
    }

    #[test]
    fn test_model_load_error() {
        let error = Error::model_load("corrupted weights file");

        let msg = error.to_string();
        assert!(msg.contains("Failed to load model"));
        assert!(msg.contains("corrupted weights file"));
    }

    #[test]
    fn test_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: Error = io_error.into();

        let msg = error.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_serialization_error() {
        let json_str = "invalid json {";
        let json_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let error: Error = json_error.into();

        let msg = error.to_string();
        assert!(msg.contains("Serialization error"));
    }

    #[test]
    fn test_internal_error() {
        let error = Error::internal("unexpected state");

        let msg = error.to_string();
        assert!(msg.contains("Internal error"));
        assert!(msg.contains("unexpected state"));
    }

    #[test]
    fn test_is_retryable() {
        // Retryable errors
        assert!(Error::Timeout {
            duration: Duration::from_secs(1)
        }
        .is_retryable());
        assert!(Error::RateLimited {
            retry_after: Duration::from_secs(1)
        }
        .is_retryable());

        // Non-retryable errors
        assert!(!Error::ModelNotFound {
            model_id: "x".to_string()
        }
        .is_retryable());
        assert!(!Error::OutOfMemory {
            requested: 1,
            available: 0
        }
        .is_retryable());
        assert!(!Error::internal("error").is_retryable());
    }

    #[test]
    fn test_is_resource_exhaustion() {
        // Resource exhaustion errors
        assert!(Error::OutOfMemory {
            requested: 100,
            available: 50
        }
        .is_resource_exhaustion());
        assert!(Error::ContextLengthExceeded {
            current: 100,
            max: 50
        }
        .is_resource_exhaustion());

        // Non-resource exhaustion errors
        assert!(!Error::Timeout {
            duration: Duration::from_secs(1)
        }
        .is_resource_exhaustion());
        assert!(!Error::internal("error").is_resource_exhaustion());
    }

    #[test]
    fn test_error_debug() {
        let error = Error::internal("test");
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("Internal"));
        assert!(debug_str.contains("test"));
    }
}
