//! Priority and workload type definitions.
//!
//! Defines the priority levels and workload categories for GPU scheduling.

use serde::{Deserialize, Serialize};

/// Priority levels for workload scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Background tasks - yield to everything.
    Background = 0,
    /// Low priority - nice to have.
    Low = 1,
    /// Normal priority - default.
    Normal = 2,
    /// High priority - user-facing work.
    High = 3,
    /// Critical - must complete, preempts others.
    Critical = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl Priority {
    /// Returns whether this priority can preempt another.
    pub fn can_preempt(self, other: Self) -> bool {
        self as u8 > other as u8 + 1
    }

    /// Returns the quality multiplier for this priority.
    ///
    /// Higher priority workloads get better quality allocations.
    pub fn quality_multiplier(self) -> f32 {
        match self {
            Self::Background => 0.6,
            Self::Low => 0.8,
            Self::Normal => 1.0,
            Self::High => 1.1,
            Self::Critical => 1.2,
        }
    }

    /// Returns the timeout multiplier for this priority.
    ///
    /// Higher priority workloads get more time.
    pub fn timeout_multiplier(self) -> f32 {
        match self {
            Self::Background => 0.5,
            Self::Low => 0.75,
            Self::Normal => 1.0,
            Self::High => 1.5,
            Self::Critical => 2.0,
        }
    }
}

/// Types of workloads the Arbiter manages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadType {
    /// LLM inference (Infernum).
    LlmInference,
    /// Image generation (Dantalion diffusion).
    ImageGeneration,
    /// Video generation (Dantalion video diffusion).
    VideoGeneration,
}

impl WorkloadType {
    /// Returns the minimum quality threshold for this workload.
    ///
    /// Below this, results are too degraded to be useful.
    pub fn min_quality(self) -> f32 {
        match self {
            Self::LlmInference => 0.4,     // LLM needs more precision
            Self::ImageGeneration => 0.3,  // Images more tolerant
            Self::VideoGeneration => 0.25, // Video very tolerant at high timesteps
        }
    }

    /// Returns the target quality for this workload at full resources.
    pub fn target_quality(self) -> f32 {
        match self {
            Self::LlmInference => 1.0,
            Self::ImageGeneration => 1.0,
            Self::VideoGeneration => 1.0,
        }
    }

    /// Returns typical memory usage estimate in bytes.
    pub fn typical_memory_mb(self) -> u64 {
        match self {
            Self::LlmInference => 8 * 1024,     // 8GB typical for inference
            Self::ImageGeneration => 6 * 1024,  // 6GB for SDXL
            Self::VideoGeneration => 12 * 1024, // 12GB for video
        }
    }

    /// Returns whether this workload type is streaming (continuous output).
    pub fn is_streaming(self) -> bool {
        match self {
            Self::LlmInference => true,     // Token by token
            Self::ImageGeneration => false, // Single output
            Self::VideoGeneration => true,  // Frame by frame
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Low > Priority::Background);
    }

    #[test]
    fn test_preemption() {
        assert!(Priority::Critical.can_preempt(Priority::Normal));
        assert!(Priority::Critical.can_preempt(Priority::Low));
        assert!(!Priority::High.can_preempt(Priority::Normal));
        assert!(!Priority::Normal.can_preempt(Priority::Normal));
    }

    #[test]
    fn test_quality_multipliers() {
        assert!(Priority::Critical.quality_multiplier() > Priority::Normal.quality_multiplier());
        assert!(Priority::Normal.quality_multiplier() > Priority::Background.quality_multiplier());
    }

    #[test]
    fn test_workload_min_quality() {
        assert!(
            WorkloadType::LlmInference.min_quality() > WorkloadType::VideoGeneration.min_quality()
        );
    }
}
