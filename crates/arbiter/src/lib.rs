//! # Arbiter - Unified GPU Coordination
//!
//! *"The judge allocates resources justly"*
//!
//! Arbiter coordinates GPU resources between Infernum (LLM inference)
//! and Dantalion (diffusion/image generation), enabling simultaneous
//! multimodal workloads on a single GPU.
//!
//! ## Core Principles
//!
//! 1. **Quality-Aware Scheduling**: Both systems can run at reduced quality
//!    when sharing GPU, with quality improving as resources become available.
//!
//! 2. **Priority-Based Arbitration**: User-facing workloads get priority,
//!    background improvement yields when needed.
//!
//! 3. **Unified Fragment Cache**: HoloTensor fragments are cached across
//!    both systems, avoiding redundant loading.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         ARBITER                                 │
//! │  Monitors GPU memory, coordinates quality targets, routes work  │
//! └──────────────────────┬──────────────────────────────────────────┘
//!                        │
//!          ┌─────────────┴─────────────┐
//!          │                           │
//!          ▼                           ▼
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │     INFERNUM        │     │     DANTALION       │
//! │  (LLM Inference)    │     │  (Diffusion)        │
//! │                     │     │                     │
//! │  Quality: 40-100%   │     │  Quality: 30-100%   │
//! │  via HoloTensor     │     │  via ProgressiveLoad│
//! └─────────────────────┘     └─────────────────────┘
//!          │                           │
//!          └─────────────┬─────────────┘
//!                        │
//!                        ▼
//!          ┌─────────────────────────────┐
//!          │    UNIFIED FRAGMENT CACHE   │
//!          │  VRAM ← RAM ← NVMe ← CDN    │
//!          └─────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use arbiter::{Arbiter, ArbiterConfig, WorkloadType, Priority};
//!
//! let arbiter = Arbiter::new(ArbiterConfig::auto_detect())?;
//!
//! // Request LLM inference at high priority
//! let llm_allocation = arbiter.request_allocation(
//!     WorkloadType::LlmInference,
//!     Priority::UserFacing,
//! ).await?;
//!
//! // LLM gets 70% quality, Dantalion drops to 40%
//! let diffusion_allocation = arbiter.request_allocation(
//!     WorkloadType::ImageGeneration,
//!     Priority::Background,
//! ).await?;
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod allocation;
pub mod cache;
pub mod coordinator;
pub mod gpu;
pub mod memory;
pub mod priority;
pub mod quality;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use thiserror::Error;

pub use allocation::{Allocation, AllocationRequest, AllocationResult};
pub use cache::{CacheConfig, CacheStats, CacheTier, FragmentCache};
pub use coordinator::{Coordinator, CoordinatorConfig};
pub use gpu::{DetectionMethod, GpuDetectionResult, GpuDetector, GpuInfo, GpuVendor};
pub use memory::{GpuMemoryTracker, MemoryPressure, MemoryStats};
pub use priority::{Priority, WorkloadType};
pub use quality::{QualityAllocation, QualityBudget, QualityPolicy};

// ==================== Error Types ====================

/// Errors from Arbiter operations.
#[derive(Debug, Error)]
pub enum ArbiterError {
    /// Insufficient GPU memory for allocation.
    #[error("Insufficient GPU memory: requested {requested}MB, available {available}MB")]
    InsufficientMemory {
        /// Requested memory in MB.
        requested: u64,
        /// Available memory in MB.
        available: u64,
    },

    /// Quality target cannot be achieved.
    #[error("Cannot achieve minimum quality {minimum}: max available {available}")]
    InsufficientQuality {
        /// Minimum quality required.
        minimum: f32,
        /// Maximum quality available.
        available: f32,
    },

    /// Workload was preempted.
    #[error("Workload preempted by higher priority task")]
    Preempted,

    /// Timeout waiting for resources.
    #[error("Timeout waiting for resources after {0:?}")]
    Timeout(Duration),

    /// Arbiter is shutting down.
    #[error("Arbiter is shutting down")]
    ShuttingDown,

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for Arbiter operations.
pub type Result<T> = std::result::Result<T, ArbiterError>;

// ==================== Arbiter Configuration ====================

/// Configuration for the Arbiter.
#[derive(Debug, Clone)]
pub struct ArbiterConfig {
    /// Total VRAM budget in bytes.
    pub vram_budget: u64,
    /// RAM budget for fragment caching.
    pub ram_budget: u64,
    /// Minimum quality for LLM inference.
    pub llm_min_quality: f32,
    /// Minimum quality for diffusion.
    pub diffusion_min_quality: f32,
    /// Enable adaptive quality balancing.
    pub adaptive_quality: bool,
    /// Memory pressure threshold to trigger quality reduction.
    pub pressure_threshold: f32,
    /// Timeout for allocation requests.
    pub allocation_timeout: Duration,
}

impl Default for ArbiterConfig {
    fn default() -> Self {
        Self {
            vram_budget: 20 * 1024 * 1024 * 1024, // 20GB
            ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
            llm_min_quality: 0.4,
            diffusion_min_quality: 0.3,
            adaptive_quality: true,
            pressure_threshold: 0.85,
            allocation_timeout: Duration::from_secs(30),
        }
    }
}

impl ArbiterConfig {
    /// Auto-detects GPU memory and creates appropriate config.
    ///
    /// Uses `GpuDetector` to query nvidia-smi, rocm-smi, or system info.
    /// Falls back to 8GB default if detection fails.
    pub fn auto_detect() -> Self {
        let detector = gpu::GpuDetector::new();
        let default_vram = 8 * 1024 * 1024 * 1024; // 8GB fallback

        let result = detector.detect_or_default(default_vram);

        // Use detected VRAM, reserving 10% for system overhead
        let usable_vram = (result.total_vram_bytes as f64 * 0.9) as u64;

        // Estimate RAM as 4x VRAM for fragment caching
        let ram_budget = result.total_vram_bytes * 4;

        Self {
            vram_budget: usable_vram,
            ram_budget,
            ..Default::default()
        }
    }

    /// Creates config for a specific VRAM size in GB.
    pub fn for_vram_gb(vram_gb: u64) -> Self {
        Self {
            vram_budget: vram_gb * 1024 * 1024 * 1024,
            ..Default::default()
        }
    }

    /// Creates config from detected GPU info.
    pub fn from_detection(result: &gpu::GpuDetectionResult) -> Self {
        let usable_vram = (result.total_vram_bytes as f64 * 0.9) as u64;
        let ram_budget = result.total_vram_bytes * 4;

        Self {
            vram_budget: usable_vram,
            ram_budget,
            ..Default::default()
        }
    }
}

// ==================== Arbiter State ====================

/// Current state of the Arbiter.
#[derive(Debug, Clone)]
pub struct ArbiterState {
    /// Active LLM workloads.
    pub active_llm_workloads: usize,
    /// Active diffusion workloads.
    pub active_diffusion_workloads: usize,
    /// Current LLM quality target.
    pub llm_quality: f32,
    /// Current diffusion quality target.
    pub diffusion_quality: f32,
    /// GPU memory pressure (0.0 - 1.0).
    pub memory_pressure: f32,
    /// VRAM used in bytes.
    pub vram_used: u64,
    /// VRAM available in bytes.
    pub vram_available: u64,
}

// ==================== Arbiter Statistics ====================

/// Statistics for the Arbiter.
#[derive(Debug, Clone, Default)]
pub struct ArbiterStats {
    /// Total allocations made.
    pub total_allocations: u64,
    /// Allocations that succeeded.
    pub successful_allocations: u64,
    /// Allocations that failed.
    pub failed_allocations: u64,
    /// Times quality was reduced due to pressure.
    pub quality_reductions: u64,
    /// Workloads preempted.
    pub preemptions: u64,
    /// Average memory pressure.
    pub avg_memory_pressure: f64,
    /// Average LLM quality achieved.
    pub avg_llm_quality: f64,
    /// Average diffusion quality achieved.
    pub avg_diffusion_quality: f64,
}

// ==================== The Arbiter ====================

/// The main GPU arbiter coordinating Infernum and Dantalion.
pub struct Arbiter {
    config: ArbiterConfig,
    memory_tracker: Arc<GpuMemoryTracker>,
    fragment_cache: Arc<FragmentCache>,
    coordinator: Arc<Coordinator>,
    state: RwLock<ArbiterState>,
    stats: RwLock<ArbiterStats>,
    started_at: Instant,
}

impl Arbiter {
    /// Creates a new Arbiter with the given configuration.
    pub fn new(config: ArbiterConfig) -> Result<Self> {
        let memory_tracker = Arc::new(GpuMemoryTracker::new(config.vram_budget));
        let fragment_cache = Arc::new(FragmentCache::new(CacheConfig {
            vram_capacity: config.vram_budget / 2, // Reserve half for weights
            ram_capacity: config.ram_budget,
        }));
        let coordinator = Arc::new(Coordinator::new(CoordinatorConfig {
            llm_min_quality: config.llm_min_quality,
            diffusion_min_quality: config.diffusion_min_quality,
            adaptive: config.adaptive_quality,
            policy: quality::QualityPolicy::Adaptive,
        }));

        let state = ArbiterState {
            active_llm_workloads: 0,
            active_diffusion_workloads: 0,
            llm_quality: 1.0,
            diffusion_quality: 1.0,
            memory_pressure: 0.0,
            vram_used: 0,
            vram_available: config.vram_budget,
        };

        Ok(Self {
            config,
            memory_tracker,
            fragment_cache,
            coordinator,
            state: RwLock::new(state),
            stats: RwLock::new(ArbiterStats::default()),
            started_at: Instant::now(),
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ArbiterConfig {
        &self.config
    }

    /// Returns current state.
    pub fn state(&self) -> ArbiterState {
        self.state.read().clone()
    }

    /// Returns statistics.
    pub fn stats(&self) -> ArbiterStats {
        self.stats.read().clone()
    }

    /// Returns the memory tracker.
    pub fn memory_tracker(&self) -> &Arc<GpuMemoryTracker> {
        &self.memory_tracker
    }

    /// Returns the fragment cache.
    pub fn fragment_cache(&self) -> &Arc<FragmentCache> {
        &self.fragment_cache
    }

    /// Returns the coordinator.
    pub fn coordinator(&self) -> &Arc<Coordinator> {
        &self.coordinator
    }

    /// Returns uptime.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Requests a GPU allocation for a workload.
    pub fn request_allocation(
        &self,
        workload_type: WorkloadType,
        priority: Priority,
        memory_required: u64,
    ) -> Result<Allocation> {
        let mut stats = self.stats.write();
        stats.total_allocations += 1;

        // Check memory availability
        let available = self.memory_tracker.available();
        if memory_required > available {
            stats.failed_allocations += 1;
            return Err(ArbiterError::InsufficientMemory {
                requested: memory_required / (1024 * 1024),
                available: available / (1024 * 1024),
            });
        }

        // Track allocation in memory tracker first
        self.memory_tracker.allocate(memory_required);

        // Calculate quality allocation based on current pressure (after allocation)
        let pressure = self.memory_tracker.pressure();
        let quality = self
            .coordinator
            .calculate_quality(workload_type, priority, pressure);

        // Update state
        let mut state = self.state.write();
        match workload_type {
            WorkloadType::LlmInference => {
                state.active_llm_workloads += 1;
                state.llm_quality = quality;
            },
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => {
                state.active_diffusion_workloads += 1;
                state.diffusion_quality = quality;
            },
        }
        state.vram_used += memory_required;
        state.vram_available = self.config.vram_budget.saturating_sub(state.vram_used);
        state.memory_pressure = pressure;

        stats.successful_allocations += 1;

        Ok(Allocation {
            id: format!("{:?}-{}", workload_type, stats.total_allocations),
            workload_type,
            priority,
            memory_allocated: memory_required,
            quality_target: quality,
            created_at: Instant::now(),
        })
    }

    /// Releases an allocation.
    pub fn release_allocation(&self, allocation: &Allocation) {
        self.memory_tracker.deallocate(allocation.memory_allocated);

        let mut state = self.state.write();
        match allocation.workload_type {
            WorkloadType::LlmInference => {
                state.active_llm_workloads = state.active_llm_workloads.saturating_sub(1);
            },
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => {
                state.active_diffusion_workloads =
                    state.active_diffusion_workloads.saturating_sub(1);
            },
        }
        state.vram_used = state.vram_used.saturating_sub(allocation.memory_allocated);
        state.vram_available = self.config.vram_budget.saturating_sub(state.vram_used);
        state.memory_pressure = self.memory_tracker.pressure();

        // Rebalance quality if workloads reduced
        if state.active_llm_workloads == 0 {
            state.diffusion_quality = 1.0;
        }
        if state.active_diffusion_workloads == 0 {
            state.llm_quality = 1.0;
        }
    }

    /// Gets recommended quality for a workload type.
    pub fn recommended_quality(&self, workload_type: WorkloadType) -> f32 {
        let state = self.state.read();
        match workload_type {
            WorkloadType::LlmInference => state.llm_quality,
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => {
                state.diffusion_quality
            },
        }
    }
}

// ==================== Integration Tests ====================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_arbiter_creation() {
        let config = ArbiterConfig::for_vram_gb(24);
        let arbiter = Arbiter::new(config);

        assert!(arbiter.is_ok());
        let arbiter = arbiter.expect("Failed to create arbiter");
        assert_eq!(arbiter.config().vram_budget, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_allocation_request() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        let allocation = arbiter.request_allocation(
            WorkloadType::LlmInference,
            Priority::High,
            1024 * 1024 * 1024, // 1GB
        );

        assert!(allocation.is_ok());
        let allocation = allocation.expect("Failed to allocate");
        assert_eq!(allocation.workload_type, WorkloadType::LlmInference);
        assert!(allocation.quality_target > 0.0);
    }

    #[test]
    fn test_allocation_release() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        let allocation = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                2 * 1024 * 1024 * 1024, // 2GB
            )
            .expect("Failed to allocate");

        let state_before = arbiter.state();
        assert_eq!(state_before.active_diffusion_workloads, 1);

        arbiter.release_allocation(&allocation);

        let state_after = arbiter.state();
        assert_eq!(state_after.active_diffusion_workloads, 0);
    }

    #[test]
    fn test_insufficient_memory() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(8)).expect("Failed");

        let result = arbiter.request_allocation(
            WorkloadType::LlmInference,
            Priority::High,
            100 * 1024 * 1024 * 1024, // 100GB - more than available
        );

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ArbiterError::InsufficientMemory { .. }
        ));
    }

    #[test]
    fn test_quality_balancing() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        // First LLM allocation gets full quality
        let llm1 = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                10 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        // Adding diffusion should reduce both qualities
        let _diff = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                8 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        let state = arbiter.state();
        // Both should be running
        assert_eq!(state.active_llm_workloads, 1);
        assert_eq!(state.active_diffusion_workloads, 1);
        // High priority LLM should have better quality
        assert!(state.llm_quality >= state.diffusion_quality);
    }
}
