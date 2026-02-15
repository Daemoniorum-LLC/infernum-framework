//! Error types for the Conclave collaboration system.

use crate::{ParticipantId, RoomId};
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for Conclave operations.
pub type Result<T> = std::result::Result<T, ConclaveError>;

/// Errors that can occur in the Conclave system.
#[derive(Debug, Error)]
pub enum ConclaveError {
    // =========================================================================
    // Room Errors
    // =========================================================================
    /// Room not found.
    #[error("Room not found: {0}")]
    RoomNotFound(RoomId),

    /// Room is archived and cannot be modified.
    #[error("Room is archived: {0}")]
    RoomArchived(RoomId),

    /// Working directory does not exist.
    #[error("Working directory not found: {0}")]
    WorkingDirNotFound(PathBuf),

    /// Maximum rooms reached.
    #[error("Maximum rooms reached: {max}")]
    MaxRoomsReached { max: usize },

    // =========================================================================
    // Participant Errors
    // =========================================================================
    /// Participant not found.
    #[error("Participant not found: {0}")]
    ParticipantNotFound(ParticipantId),

    /// Participant is not in the specified room.
    #[error("Participant {0} is not in room {1}")]
    NotInRoom(ParticipantId, RoomId),

    /// Participant already exists.
    #[error("Participant already in room: {0}")]
    AlreadyInRoom(ParticipantId),

    /// Not authorized to perform action.
    #[error("Not authorized: {0}")]
    NotAuthorized(String),

    // =========================================================================
    // Backend Errors
    // =========================================================================
    /// Failed to spawn agent backend.
    #[error("Failed to spawn agent: {0}")]
    SpawnFailed(String),

    /// Failed to spawn a specific backend process.
    #[error("Failed to spawn {backend}: {reason}")]
    BackendSpawnFailed { backend: String, reason: String },

    /// Backend terminated unexpectedly.
    #[error("Backend terminated: {session_id}")]
    BackendTerminated { session_id: String },

    /// Failed to communicate with backend.
    #[error("Backend communication failed for {session_id}: {reason}")]
    BackendCommunicationFailed { session_id: String, reason: String },

    /// Persona not found in Grimoire.
    #[error("Persona not found: {0}")]
    PersonaNotFound(String),

    /// Skill not found in Grimoire.
    #[error("Skill not found: {0}")]
    SkillNotFound(String),

    // =========================================================================
    // Channel Errors
    // =========================================================================
    /// Invalid channel operation.
    #[error("Invalid channel operation: {0}")]
    InvalidChannel(String),

    /// Not this participant's turn.
    #[error("Not your turn in main channel")]
    NotYourTurn,

    /// Turn queue is full.
    #[error("Turn queue is full (max: {max})")]
    QueueFull { max: u32 },

    // =========================================================================
    // System Errors
    // =========================================================================
    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl ConclaveError {
    /// Returns true if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ConclaveError::NotYourTurn
                | ConclaveError::QueueFull { .. }
                | ConclaveError::NotAuthorized(_)
        )
    }

    /// Returns true if this error indicates the room cannot be used.
    pub fn is_room_unusable(&self) -> bool {
        matches!(
            self,
            ConclaveError::RoomNotFound(_)
                | ConclaveError::RoomArchived(_)
                | ConclaveError::WorkingDirNotFound(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_error_display() {
        let err = ConclaveError::RoomNotFound(RoomId::new());
        assert!(err.to_string().contains("Room not found"));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(ConclaveError::NotYourTurn.is_recoverable());
        assert!(ConclaveError::QueueFull { max: 10 }.is_recoverable());
        assert!(!ConclaveError::RoomNotFound(RoomId::new()).is_recoverable());
    }

    #[test]
    fn test_error_is_room_unusable() {
        assert!(ConclaveError::RoomArchived(RoomId::new()).is_room_unusable());
        assert!(ConclaveError::WorkingDirNotFound(PathBuf::from("/tmp")).is_room_unusable());
        assert!(!ConclaveError::NotYourTurn.is_room_unusable());
    }
}
