//! Allocation types and requests.
//!
//! Represents GPU memory allocations and requests for workloads.

use std::time::Instant;
use serde::{Deserialize, Serialize};
use crate::priority::{Priority, WorkloadType};

/// A GPU memory allocation.
#[derive(Debug, Clone)]
pub struct Allocation {
    /// Unique allocation ID.
    pub id: String,
    /// Type of workload.
    pub workload_type: WorkloadType,
    /// Priority level.
    pub priority: Priority,
    /// Memory allocated in bytes.
    pub memory_allocated: u64,
    /// Quality target (0.0 - 1.0).
    pub quality_target: f32,
    /// When this allocation was created.
    pub created_at: Instant,
}

impl Allocation {
    /// Returns how long this allocation has been active.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns whether this is an LLM workload.
    pub fn is_llm(&self) -> bool {
        matches!(self.workload_type, WorkloadType::LlmInference)
    }

    /// Returns whether this is a diffusion workload.
    pub fn is_diffusion(&self) -> bool {
        matches!(
            self.workload_type,
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration
        )
    }

    /// Returns memory in megabytes.
    pub fn memory_mb(&self) -> u64 {
        self.memory_allocated / (1024 * 1024)
    }
}

/// A request for GPU allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRequest {
    /// Type of workload.
    pub workload_type: WorkloadType,
    /// Priority level.
    pub priority: Priority,
    /// Memory required in bytes.
    pub memory_required: u64,
    /// Minimum acceptable quality.
    pub min_quality: Option<f32>,
    /// Whether to wait for resources or fail immediately.
    pub wait_for_resources: bool,
    /// Maximum time to wait if waiting.
    pub timeout_ms: Option<u64>,
    /// Metadata for tracking.
    pub metadata: Option<String>,
}

impl AllocationRequest {
    /// Creates a new request for LLM inference.
    pub fn llm(memory_required: u64) -> Self {
        Self {
            workload_type: WorkloadType::LlmInference,
            priority: Priority::Normal,
            memory_required,
            min_quality: None,
            wait_for_resources: true,
            timeout_ms: Some(30_000),
            metadata: None,
        }
    }

    /// Creates a new request for image generation.
    pub fn image(memory_required: u64) -> Self {
        Self {
            workload_type: WorkloadType::ImageGeneration,
            priority: Priority::Normal,
            memory_required,
            min_quality: None,
            wait_for_resources: true,
            timeout_ms: Some(60_000),
            metadata: None,
        }
    }

    /// Creates a new request for video generation.
    pub fn video(memory_required: u64) -> Self {
        Self {
            workload_type: WorkloadType::VideoGeneration,
            priority: Priority::Normal,
            memory_required,
            min_quality: None,
            wait_for_resources: true,
            timeout_ms: Some(120_000),
            metadata: None,
        }
    }

    /// Sets priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets minimum quality.
    pub fn with_min_quality(mut self, min_quality: f32) -> Self {
        self.min_quality = Some(min_quality.clamp(0.0, 1.0));
        self
    }

    /// Sets to fail immediately if resources unavailable.
    pub fn no_wait(mut self) -> Self {
        self.wait_for_resources = false;
        self.timeout_ms = None;
        self
    }

    /// Sets timeout.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets metadata.
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }

    /// Returns effective minimum quality.
    pub fn effective_min_quality(&self) -> f32 {
        self.min_quality.unwrap_or_else(|| self.workload_type.min_quality())
    }
}

/// Result of an allocation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationResult {
    /// Allocation succeeded.
    Success {
        /// Allocated quality target.
        quality: f32,
        /// Memory allocated in bytes.
        memory: u64,
    },
    /// Insufficient memory.
    InsufficientMemory {
        /// Memory requested.
        requested: u64,
        /// Memory available.
        available: u64,
    },
    /// Quality requirements cannot be met.
    InsufficientQuality {
        /// Minimum quality requested.
        requested: f32,
        /// Maximum quality achievable.
        achievable: f32,
    },
    /// Timed out waiting.
    Timeout {
        /// How long we waited.
        waited_ms: u64,
    },
    /// Request was preempted by higher priority.
    Preempted,
}

impl AllocationResult {
    /// Returns whether allocation succeeded.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Returns quality if successful.
    pub fn quality(&self) -> Option<f32> {
        match self {
            Self::Success { quality, .. } => Some(*quality),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_age() {
        let alloc = Allocation {
            id: "test".to_string(),
            workload_type: WorkloadType::LlmInference,
            priority: Priority::Normal,
            memory_allocated: 1024,
            quality_target: 1.0,
            created_at: Instant::now(),
        };

        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(alloc.age().as_millis() >= 10);
    }

    #[test]
    fn test_request_builder() {
        let req = AllocationRequest::llm(1024 * 1024 * 1024)
            .with_priority(Priority::High)
            .with_min_quality(0.8)
            .with_metadata("test inference");

        assert!(matches!(req.workload_type, WorkloadType::LlmInference));
        assert!(matches!(req.priority, Priority::High));
        assert_eq!(req.min_quality, Some(0.8));
        assert!(req.wait_for_resources);
    }

    #[test]
    fn test_effective_min_quality() {
        let req = AllocationRequest::llm(1024);
        // Should use workload default
        assert!((req.effective_min_quality() - 0.4).abs() < 0.001);

        let req_with_min = req.with_min_quality(0.6);
        assert!((req_with_min.effective_min_quality() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_allocation_result() {
        let success = AllocationResult::Success {
            quality: 0.9,
            memory: 1024,
        };
        assert!(success.is_success());
        assert_eq!(success.quality(), Some(0.9));

        let failure = AllocationResult::InsufficientMemory {
            requested: 1000,
            available: 500,
        };
        assert!(!failure.is_success());
        assert_eq!(failure.quality(), None);
    }
}
