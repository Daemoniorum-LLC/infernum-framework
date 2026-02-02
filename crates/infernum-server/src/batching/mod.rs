//! Continuous batching for high-throughput LLM inference.
//!
//! This module implements dynamic batching similar to vLLM, enabling 2-5x
//! throughput improvements through intelligent request scheduling.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     BatchScheduler                          │
//! │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
//! │  │ PendingQueue │ → │ BatchFormer  │ → │ ActiveBatch  │    │
//! │  └──────────────┘   └──────────────┘   └──────────────┘    │
//! │         ↑                                     │             │
//! │    New Requests                          Token Iteration    │
//! │                                               ↓             │
//! │                                    ┌──────────────────┐    │
//! │                                    │  KV Cache Mgmt   │    │
//! │                                    └──────────────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Dynamic Batching**: Requests are grouped based on sequence length
//! - **Preemption**: Low-priority requests yield to high-priority ones
//! - **Memory Pressure Handling**: Graceful degradation under memory pressure
//! - **Token-level Iteration**: Process one token at a time across batch
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::batching::{BatchScheduler, BatchConfig};
//!
//! let config = BatchConfig::default();
//! let scheduler = BatchScheduler::new(config);
//!
//! // Submit requests
//! let handle = scheduler.submit(request).await?;
//!
//! // Wait for completion
//! let response = handle.await?;
//! ```

mod batch;
mod iteration;
mod scheduler;

pub use batch::{
    ActiveBatch, BatchEntry, BatchId, BatchState, BatchStats, FinishReason, PendingRequest,
    RequestState, SamplingParams, Sequence, SequenceGroup, SequenceId, SequenceState,
};
pub use iteration::{
    IterationConfig, IterationMetrics, IterationResult, IterationStep, TokenIterator,
};
pub use scheduler::{
    BatchConfig, BatchScheduler, PreemptionPolicy, SchedulerMetrics, SchedulerState,
    SchedulerStats, SchedulingPolicy,
};

use std::fmt;
use std::time::Duration;

/// Priority level for batched requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BatchPriority {
    /// Background tasks, lowest priority.
    Background = 0,
    /// Normal user requests.
    Normal = 1,
    /// High priority requests.
    High = 2,
    /// Critical requests, highest priority.
    Critical = 3,
}

impl Default for BatchPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl fmt::Display for BatchPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Background => write!(f, "background"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

impl BatchPriority {
    /// Creates from a numeric value (1-4).
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Self::Background,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical,
        }
    }

    /// Converts to numeric value.
    pub fn as_level(&self) -> u8 {
        *self as u8
    }
}

/// Error type for batching operations.
#[derive(Debug, Clone)]
pub enum BatchError {
    /// Queue is full.
    QueueFull {
        /// Current queue size.
        current: usize,
        /// Maximum queue size.
        max: usize,
    },

    /// Request timed out waiting in queue.
    QueueTimeout {
        /// Time spent waiting.
        waited: Duration,
        /// Maximum wait time.
        max: Duration,
    },

    /// Batch was preempted.
    Preempted {
        /// Reason for preemption.
        reason: String,
    },

    /// Memory pressure caused rejection.
    MemoryPressure {
        /// Available memory.
        available: u64,
        /// Required memory.
        required: u64,
    },

    /// Sequence too long.
    SequenceTooLong {
        /// Actual length.
        actual: usize,
        /// Maximum length.
        max: usize,
    },

    /// Scheduler is shutting down.
    ShuttingDown,

    /// Internal error.
    Internal(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { current, max } => {
                write!(f, "Queue full: {}/{}", current, max)
            }
            Self::QueueTimeout { waited, max } => {
                write!(f, "Queue timeout: {:?} > {:?}", waited, max)
            }
            Self::Preempted { reason } => {
                write!(f, "Request preempted: {}", reason)
            }
            Self::MemoryPressure { available, required } => {
                write!(
                    f,
                    "Memory pressure: {} available, {} required",
                    available, required
                )
            }
            Self::SequenceTooLong { actual, max } => {
                write!(f, "Sequence too long: {} > {}", actual, max)
            }
            Self::ShuttingDown => write!(f, "Scheduler is shutting down"),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for BatchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_priority_default() {
        assert_eq!(BatchPriority::default(), BatchPriority::Normal);
    }

    #[test]
    fn test_batch_priority_ordering() {
        assert!(BatchPriority::Background < BatchPriority::Normal);
        assert!(BatchPriority::Normal < BatchPriority::High);
        assert!(BatchPriority::High < BatchPriority::Critical);
    }

    #[test]
    fn test_batch_priority_from_level() {
        assert_eq!(BatchPriority::from_level(0), BatchPriority::Background);
        assert_eq!(BatchPriority::from_level(1), BatchPriority::Normal);
        assert_eq!(BatchPriority::from_level(2), BatchPriority::High);
        assert_eq!(BatchPriority::from_level(3), BatchPriority::Critical);
        assert_eq!(BatchPriority::from_level(99), BatchPriority::Critical);
    }

    #[test]
    fn test_batch_priority_as_level() {
        assert_eq!(BatchPriority::Background.as_level(), 0);
        assert_eq!(BatchPriority::Normal.as_level(), 1);
        assert_eq!(BatchPriority::High.as_level(), 2);
        assert_eq!(BatchPriority::Critical.as_level(), 3);
    }

    #[test]
    fn test_batch_priority_display() {
        assert_eq!(BatchPriority::Background.to_string(), "background");
        assert_eq!(BatchPriority::Normal.to_string(), "normal");
        assert_eq!(BatchPriority::High.to_string(), "high");
        assert_eq!(BatchPriority::Critical.to_string(), "critical");
    }

    #[test]
    fn test_batch_error_display() {
        let err = BatchError::QueueFull {
            current: 100,
            max: 100,
        };
        assert!(err.to_string().contains("Queue full"));

        let err = BatchError::QueueTimeout {
            waited: Duration::from_secs(30),
            max: Duration::from_secs(10),
        };
        assert!(err.to_string().contains("timeout"));

        let err = BatchError::Preempted {
            reason: "high priority".to_string(),
        };
        assert!(err.to_string().contains("preempted"));

        let err = BatchError::MemoryPressure {
            available: 1000,
            required: 2000,
        };
        assert!(err.to_string().contains("Memory pressure"));

        let err = BatchError::SequenceTooLong {
            actual: 5000,
            max: 4096,
        };
        assert!(err.to_string().contains("too long"));

        let err = BatchError::ShuttingDown;
        assert!(err.to_string().contains("shutting down"));

        let err = BatchError::Internal("oops".to_string());
        assert!(err.to_string().contains("Internal"));
    }
}
