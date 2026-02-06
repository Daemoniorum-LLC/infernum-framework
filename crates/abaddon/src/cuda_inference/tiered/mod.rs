//! Tiered memory management for CUDA inference.
//!
//! Implements a 3-tier memory hierarchy (VRAM ← RAM ← NVMe) for efficient
//! inference of models larger than available VRAM.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ VRAM (Hot)  - GPU memory, ~0.1ms latency, 24GB typical      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ RAM (Warm)  - Pinned CPU memory, ~1ms latency, 64-128GB     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ NVMe (Cold) - Disk cache, ~10ms latency, 1TB+               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use abaddon::cuda_inference::tiered::{TieredWeightStore, TieredConfig};
//!
//! let config = TieredConfig::for_24gb_gpu();
//! let weights = TieredWeightStore::load(model_dir, config)?;
//!
//! // Layers are loaded on-demand from the appropriate tier
//! let layer = weights.get_layer(0)?;
//! ```

mod config;
mod error;
mod loader;
mod lru;
mod nvme_cache;
mod prefetch;
mod ram_cache;
mod stats;
pub mod store;
mod vram_cache;

pub use config::{HardwareConfig, LoadingStrategy, ProgressiveConfig, TieredConfig};
pub use error::TieredError;
pub use loader::{create_loader, EagerLoader, ProgressiveLoader, WeightLoader};
pub use prefetch::{PrefetchController, PrefetchScheduler};
pub use stats::TieredStats;
pub use store::TieredWeightStore;

// Re-export cache types for testing
#[cfg(test)]
pub use nvme_cache::NvmeCache;
#[cfg(test)]
pub use ram_cache::RamCache;
#[cfg(test)]
pub use vram_cache::VramCache;

// Property-based tests for memory invariants
#[cfg(test)]
mod property_tests;
