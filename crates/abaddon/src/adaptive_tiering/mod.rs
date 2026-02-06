//! Adaptive Memory Tiering System
//!
//! Intelligent memory allocation that maximizes inference quality within hardware
//! constraints, regardless of model size.
//!
//! Unlike fixed layer-swapping (designed for 405B+ models), this system:
//! - Analyzes the gap between model size and available memory
//! - Scores tensor importance based on inference impact
//! - Selects optimal precision per tensor (BF16, INT8, INT4)
//! - Places tensors in optimal memory tier (VRAM, RAM, NVMe)
//! - Minimizes or eliminates runtime swapping when possible
//!
//! # Example
//!
//! ```ignore
//! use abaddon::adaptive_tiering::{
//!     AdaptiveTieringConfig, AllocationPlanner, ModelProfile,
//! };
//!
//! let config = AdaptiveTieringConfig::default();
//! let planner = AllocationPlanner::new(config);
//! let profile = ModelProfile::from_directory(model_dir)?;
//! let plan = planner.plan(&profile)?;
//!
//! println!("VRAM usage: {} GB", plan.vram_usage / (1024 * 1024 * 1024));
//! println!("Swap count: {}", plan.swap_count);
//! ```

mod config;
mod importance;
mod loader;
mod planner;
mod types;

pub use config::AdaptiveTieringConfig;
pub use importance::{ImportanceFactors, ImportanceScorer};
pub use loader::{AdaptiveLoader, AdaptiveLoaderError, AdaptiveLoaderStats, EagerTensorProvider};
pub use planner::{AllocationPlanner, PlannerError};
pub use types::{
    AllocationPlan, ArchitectureType, LoadingBackend, MemoryTier, ModelProfile, ProfileError,
    ReconstructionPath, ReconstructionSummary, TensorAllocation, TensorInfo, TensorPrecision,
    TensorType,
};

#[cfg(test)]
mod tests;
