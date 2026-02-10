//! Error types for the persistence layer.

use thiserror::Error;

/// Errors that can occur during persistence operations.
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// Database connection failed.
    #[error("Database connection failed: {0}")]
    Connection(String),

    /// SQL execution failed.
    #[error("SQL error: {0}")]
    Sql(#[from] rusqlite::Error),

    /// Transaction failed.
    #[error("Transaction failed: {0}")]
    Transaction(String),

    /// Migration failed.
    #[error("Migration failed: {0}")]
    Migration(String),

    /// Record not found.
    #[error("Not found: {entity} with id '{id}'")]
    NotFound {
        /// Entity type (dataset, experiment, etc.).
        entity: String,
        /// Entity ID.
        id: String,
    },

    /// Constraint violation.
    #[error("Constraint violation: {0}")]
    Constraint(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Database is locked.
    #[error("Database is locked, another operation in progress")]
    Locked,

    /// Invalid data format.
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Result type for persistence operations.
pub type Result<T> = std::result::Result<T, PersistenceError>;

impl PersistenceError {
    /// Creates a not found error.
    pub fn not_found(entity: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            entity: entity.into(),
            id: id.into(),
        }
    }

    /// Creates a constraint error.
    pub fn constraint(msg: impl Into<String>) -> Self {
        Self::Constraint(msg.into())
    }

    /// Creates a migration error.
    pub fn migration(msg: impl Into<String>) -> Self {
        Self::Migration(msg.into())
    }

    /// Returns true if this is a not found error.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }

    /// Returns true if this is a constraint error.
    pub fn is_constraint(&self) -> bool {
        matches!(self, Self::Constraint(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_error() {
        let err = PersistenceError::not_found("dataset", "abc123");
        assert!(err.is_not_found());
        assert!(err.to_string().contains("dataset"));
        assert!(err.to_string().contains("abc123"));
    }

    #[test]
    fn test_constraint_error() {
        let err = PersistenceError::constraint("unique name required");
        assert!(err.is_constraint());
        assert!(err.to_string().contains("unique name required"));
    }

    #[test]
    fn test_error_from_rusqlite() {
        let sqlite_err = rusqlite::Error::InvalidQuery;
        let err: PersistenceError = sqlite_err.into();
        assert!(matches!(err, PersistenceError::Sql(_)));
    }
}
