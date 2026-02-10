//! Arbiter integration for GPU resource management.
//!
//! Enables the inference engine to coordinate GPU memory with other workloads
//! (like image generation) through the Arbiter resource manager.
//!
//! ## Features
//!
//! - Automatic workload registration with Arbiter
//! - Dynamic quality adjustment based on memory pressure
//! - Graceful degradation when sharing GPU with other workloads
//!
//! ## Example
//!
//! ```ignore
//! use abaddon::{Engine, EngineConfig, ArbiterCoordinator};
//! use infernum_arbiter::{Arbiter, ArbiterConfig};
//! use std::sync::Arc;
//!
//! let arbiter = Arc::new(Arbiter::new(ArbiterConfig::auto_detect())?);
//! let mut coordinator = ArbiterCoordinator::new(arbiter, "llm-engine");
//!
//! // Register with estimated memory requirements
//! let quality = coordinator.register(8_000_000_000, 4_000_000_000)?;
//! println!("Allocated with quality: {}", quality);
//!
//! // Check pressure before inference
//! let pressure = coordinator.memory_pressure();
//! if pressure > 0.8 {
//!     // Reduce batch size or context length
//! }
//! ```

use std::sync::Arc;

use infernum_arbiter::{Allocation, Arbiter, MemoryPressure, Priority, WorkloadType};

/// Wrapper for Arbiter integration with the inference engine.
pub struct ArbiterCoordinator {
    /// The Arbiter instance.
    arbiter: Arc<Arbiter>,
    /// Current allocation for this workload.
    allocation: Option<Allocation>,
    /// Workload identifier.
    workload_id: String,
    /// Estimated model memory footprint in bytes.
    model_memory_bytes: u64,
}

impl ArbiterCoordinator {
    /// Creates a new coordinator with the given Arbiter.
    #[must_use]
    pub fn new(arbiter: Arc<Arbiter>, workload_id: impl Into<String>) -> Self {
        Self {
            arbiter,
            allocation: None,
            workload_id: workload_id.into(),
            model_memory_bytes: 0,
        }
    }

    /// Registers the inference workload with Arbiter.
    ///
    /// # Arguments
    ///
    /// * `model_memory_bytes` - Estimated memory for model weights
    /// * `kv_cache_bytes` - Estimated memory for KV cache
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    pub fn register(
        &mut self,
        model_memory_bytes: u64,
        kv_cache_bytes: u64,
    ) -> Result<f32, ArbiterCoordinatorError> {
        self.model_memory_bytes = model_memory_bytes;
        let total_memory = model_memory_bytes + kv_cache_bytes;

        let allocation = self
            .arbiter
            .request_allocation(WorkloadType::LlmInference, Priority::Normal, total_memory)
            .map_err(|e| ArbiterCoordinatorError::AllocationFailed(e.to_string()))?;

        let quality_factor = allocation.quality_target;

        tracing::info!(
            workload_id = %self.workload_id,
            allocated_mb = allocation.memory_allocated / (1024 * 1024),
            quality_factor = %quality_factor,
            "Registered inference workload with Arbiter"
        );

        self.allocation = Some(allocation);
        Ok(quality_factor)
    }

    /// Registers with high priority (for user-facing requests).
    pub fn register_high_priority(
        &mut self,
        model_memory_bytes: u64,
        kv_cache_bytes: u64,
    ) -> Result<f32, ArbiterCoordinatorError> {
        self.model_memory_bytes = model_memory_bytes;
        let total_memory = model_memory_bytes + kv_cache_bytes;

        let allocation = self
            .arbiter
            .request_allocation(WorkloadType::LlmInference, Priority::High, total_memory)
            .map_err(|e| ArbiterCoordinatorError::AllocationFailed(e.to_string()))?;

        let quality_factor = allocation.quality_target;

        tracing::info!(
            workload_id = %self.workload_id,
            allocated_mb = allocation.memory_allocated / (1024 * 1024),
            quality_factor = %quality_factor,
            priority = "high",
            "Registered high-priority inference workload with Arbiter"
        );

        self.allocation = Some(allocation);
        Ok(quality_factor)
    }

    /// Updates the allocation based on current memory pressure.
    ///
    /// Should be called periodically or before generation to adjust quality.
    /// Returns the new quality factor if adjustment was made.
    pub fn update_allocation(&mut self) -> Option<f32> {
        let current = self.allocation.as_ref()?;
        let pressure = self.memory_pressure();
        let current_quality = current.quality_target;

        // Check if we need to adjust based on pressure
        let pressure_level = self.pressure_level();
        let needs_adjustment = match pressure_level {
            MemoryPressure::Critical => current_quality > 0.5,
            MemoryPressure::High => current_quality > 0.7,
            _ => false,
        };

        if needs_adjustment {
            // Release current allocation
            self.arbiter.release_allocation(current);

            // Request reduced allocation
            let reduced_kv = self.calculate_reduced_kv_cache(pressure_level);
            let reduced_memory = self.model_memory_bytes + reduced_kv;

            if let Ok(new_allocation) = self.arbiter.request_allocation(
                WorkloadType::LlmInference,
                Priority::Normal,
                reduced_memory,
            ) {
                let quality = new_allocation.quality_target;

                tracing::info!(
                    workload_id = %self.workload_id,
                    pressure = %pressure,
                    new_quality_factor = %quality,
                    "Adjusted allocation due to memory pressure"
                );

                self.allocation = Some(new_allocation);
                return Some(quality);
            }
        }

        Some(current_quality)
    }

    /// Returns the current quality factor (0.0 - 1.0).
    #[must_use]
    pub fn quality_factor(&self) -> f32 {
        self.allocation
            .as_ref()
            .map(|a| a.quality_target)
            .unwrap_or(1.0)
    }

    /// Returns current memory pressure as a float (0.0 - 1.0).
    #[must_use]
    pub fn memory_pressure(&self) -> f32 {
        self.arbiter.memory_tracker().pressure()
    }

    /// Returns current memory pressure level as enum.
    #[must_use]
    pub fn pressure_level(&self) -> MemoryPressure {
        let pressure = self.memory_pressure();
        if pressure >= 0.95 {
            MemoryPressure::Critical
        } else if pressure >= 0.85 {
            MemoryPressure::High
        } else if pressure >= 0.7 {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Low
        }
    }

    /// Returns whether this workload is registered.
    #[must_use]
    pub fn is_registered(&self) -> bool {
        self.allocation.is_some()
    }

    /// Returns the current allocation if any.
    #[must_use]
    pub fn allocation(&self) -> Option<&Allocation> {
        self.allocation.as_ref()
    }

    /// Returns the allocated memory in bytes.
    #[must_use]
    pub fn allocated_memory(&self) -> u64 {
        self.allocation
            .as_ref()
            .map(|a| a.memory_allocated)
            .unwrap_or(0)
    }

    /// Releases the allocation.
    pub fn release(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            self.arbiter.release_allocation(&allocation);
            tracing::info!(
                workload_id = %self.workload_id,
                "Released inference workload allocation"
            );
        }
    }

    /// Calculates reduced KV cache size based on memory pressure.
    fn calculate_reduced_kv_cache(&self, pressure: MemoryPressure) -> u64 {
        let base_kv = self
            .allocation
            .as_ref()
            .map(|a| a.memory_allocated.saturating_sub(self.model_memory_bytes))
            .unwrap_or(0);

        match pressure {
            MemoryPressure::Critical => base_kv / 4, // 25% of original
            MemoryPressure::High => base_kv / 2,     // 50% of original
            MemoryPressure::Moderate => base_kv * 3 / 4, // 75% of original
            MemoryPressure::Low => base_kv,          // Full KV cache
        }
    }
}

impl Drop for ArbiterCoordinator {
    fn drop(&mut self) {
        self.release();
    }
}

/// Errors from Arbiter coordination.
#[derive(Debug, Clone)]
pub enum ArbiterCoordinatorError {
    /// Allocation request failed.
    AllocationFailed(String),
    /// Arbiter not available.
    NotAvailable,
    /// Insufficient memory.
    InsufficientMemory {
        /// Requested memory in bytes.
        requested: u64,
        /// Available memory in bytes.
        available: u64,
    },
}

impl std::fmt::Display for ArbiterCoordinatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed(msg) => write!(f, "Allocation failed: {}", msg),
            Self::NotAvailable => write!(f, "Arbiter not available"),
            Self::InsufficientMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Insufficient memory: requested {} bytes, {} available",
                    requested, available
                )
            },
        }
    }
}

impl std::error::Error for ArbiterCoordinatorError {}

/// Quality level based on Arbiter allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    /// Minimal quality for memory-constrained scenarios.
    Minimal,
    /// Reduced quality with smaller KV cache.
    Reduced,
    /// Full quality with maximum KV cache.
    Full,
}

impl QualityLevel {
    /// Creates quality level from a quality factor.
    #[must_use]
    pub fn from_factor(factor: f32) -> Self {
        if factor >= 0.9 {
            Self::Full
        } else if factor >= 0.6 {
            Self::Reduced
        } else {
            Self::Minimal
        }
    }

    /// Returns the maximum context length multiplier for this quality level.
    #[must_use]
    pub fn context_multiplier(self) -> f32 {
        match self {
            Self::Full => 1.0,
            Self::Reduced => 0.5,
            Self::Minimal => 0.25,
        }
    }

    /// Returns whether this quality level supports full context.
    #[must_use]
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_arbiter::ArbiterConfig;

    #[test]
    fn test_coordinator_creation() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let coordinator = ArbiterCoordinator::new(arbiter, "test-inference");

        assert!(!coordinator.is_registered());
        assert_eq!(coordinator.quality_factor(), 1.0);
    }

    #[test]
    fn test_coordinator_register() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test-inference");

        let model_size = 8 * 1024 * 1024 * 1024; // 8GB
        let kv_cache = 4 * 1024 * 1024 * 1024; // 4GB

        let result = coordinator.register(model_size, kv_cache);
        assert!(result.is_ok());
        assert!(coordinator.is_registered());
        assert!(coordinator.quality_factor() > 0.0);
    }

    #[test]
    fn test_coordinator_register_high_priority() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test-inference");

        let model_size = 4 * 1024 * 1024 * 1024;
        let kv_cache = 2 * 1024 * 1024 * 1024;

        let result = coordinator.register_high_priority(model_size, kv_cache);
        assert!(result.is_ok());
        assert!(coordinator.is_registered());
    }

    #[test]
    fn test_coordinator_release() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test-inference");

        let model_size = 4 * 1024 * 1024 * 1024;
        let kv_cache = 2 * 1024 * 1024 * 1024;

        coordinator
            .register(model_size, kv_cache)
            .expect("Failed to register");
        assert!(coordinator.is_registered());

        coordinator.release();
        assert!(!coordinator.is_registered());
    }

    #[test]
    fn test_allocated_memory() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test");

        // Before registration
        assert_eq!(coordinator.allocated_memory(), 0);

        let model_size = 4 * 1024 * 1024 * 1024;
        let kv_cache = 2 * 1024 * 1024 * 1024;
        coordinator.register(model_size, kv_cache).expect("Failed");

        // After registration
        assert_eq!(coordinator.allocated_memory(), model_size + kv_cache);
    }

    #[test]
    fn test_quality_level_from_factor() {
        assert_eq!(QualityLevel::from_factor(1.0), QualityLevel::Full);
        assert_eq!(QualityLevel::from_factor(0.95), QualityLevel::Full);
        assert_eq!(QualityLevel::from_factor(0.8), QualityLevel::Reduced);
        assert_eq!(QualityLevel::from_factor(0.5), QualityLevel::Minimal);
        assert_eq!(QualityLevel::from_factor(0.3), QualityLevel::Minimal);
    }

    #[test]
    fn test_quality_level_context_multiplier() {
        assert!((QualityLevel::Full.context_multiplier() - 1.0).abs() < 0.001);
        assert!((QualityLevel::Reduced.context_multiplier() - 0.5).abs() < 0.001);
        assert!((QualityLevel::Minimal.context_multiplier() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_quality_level_is_full() {
        assert!(QualityLevel::Full.is_full());
        assert!(!QualityLevel::Reduced.is_full());
        assert!(!QualityLevel::Minimal.is_full());
    }

    #[test]
    fn test_coordinator_error_display() {
        let err = ArbiterCoordinatorError::AllocationFailed("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = ArbiterCoordinatorError::NotAvailable;
        assert!(format!("{}", err).contains("not available"));

        let err = ArbiterCoordinatorError::InsufficientMemory {
            requested: 1000,
            available: 500,
        };
        assert!(format!("{}", err).contains("1000"));
        assert!(format!("{}", err).contains("500"));
    }

    #[test]
    fn test_memory_pressure_query() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let coordinator = ArbiterCoordinator::new(arbiter, "test");

        // Fresh arbiter should have low pressure (close to 0)
        let pressure = coordinator.memory_pressure();
        assert!(pressure < 0.1);
    }

    #[test]
    fn test_pressure_level() {
        let arbiter = Arc::new(
            Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed to create arbiter"),
        );
        let coordinator = ArbiterCoordinator::new(arbiter, "test");

        // Fresh arbiter should have low pressure level
        let level = coordinator.pressure_level();
        assert!(matches!(level, MemoryPressure::Low));
    }

    #[test]
    fn test_quality_level_ordering() {
        assert!(QualityLevel::Full > QualityLevel::Reduced);
        assert!(QualityLevel::Reduced > QualityLevel::Minimal);
    }

    #[test]
    fn test_coordinator_workload_id() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed"));
        let coordinator = ArbiterCoordinator::new(arbiter, "my-workload");

        assert_eq!(coordinator.workload_id, "my-workload");
    }

    #[test]
    fn test_allocation_not_present_initially() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed"));
        let coordinator = ArbiterCoordinator::new(arbiter, "test");

        assert!(coordinator.allocation().is_none());
    }

    #[test]
    fn test_update_allocation_when_not_registered() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed"));
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test");

        assert!(coordinator.update_allocation().is_none());
    }

    #[test]
    fn test_update_allocation_returns_quality() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed"));
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test");

        let model_size = 4 * 1024 * 1024 * 1024;
        let kv_cache = 2 * 1024 * 1024 * 1024;
        coordinator.register(model_size, kv_cache).expect("Failed");

        // Should return current quality when no adjustment needed
        let quality = coordinator.update_allocation();
        assert!(quality.is_some());
        assert!(quality.expect("Has quality") > 0.0);
    }

    #[test]
    fn test_insufficient_memory_allocation() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(8)).expect("Failed"));
        let mut coordinator = ArbiterCoordinator::new(arbiter, "test");

        // Try to allocate more than available
        let huge_model = 100 * 1024 * 1024 * 1024; // 100GB
        let result = coordinator.register(huge_model, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_drop_releases_allocation() {
        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed"));

        // Scope to trigger drop
        {
            let mut coordinator = ArbiterCoordinator::new(Arc::clone(&arbiter), "test");
            coordinator
                .register(4 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
                .expect("Failed");
            // Drop happens here
        }

        // After drop, arbiter should have low pressure again
        let state = arbiter.state();
        assert_eq!(state.active_llm_workloads, 0);
    }
}
