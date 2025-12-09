//! Error types for the Infernum ecosystem.

use std::time::Duration;
use thiserror::Error;

/// Result type alias using [`Error`].
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
