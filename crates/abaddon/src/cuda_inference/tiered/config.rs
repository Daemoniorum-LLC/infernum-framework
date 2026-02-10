//! Configuration for tiered memory management.

use std::path::PathBuf;

use crate::adaptive_tiering::{AllocationPlan, TensorPrecision};

/// Bytes in a gigabyte.
const GB: u64 = 1024 * 1024 * 1024;

/// Configuration for tiered weight loading.
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Hardware configuration.
    pub hardware: HardwareConfig,

    /// Progressive loading configuration (when NVMe tier is needed).
    pub progressive: ProgressiveConfig,

    /// CUDA device ID.
    pub device_id: usize,

    /// Maximum sequence length for KV cache sizing.
    pub max_seq_len: usize,

    /// Whether to use eager loading when possible.
    pub prefer_eager: bool,
}

impl TieredConfig {
    /// Create configuration for a 24GB GPU with default RAM.
    pub fn for_24gb_gpu() -> Self {
        Self {
            hardware: HardwareConfig {
                vram_budget: 22 * GB, // Leave 2GB headroom
                ram_budget: 60 * GB,
                nvme_cache_size: 0, // Unlimited
                use_pinned_memory: true,
            },
            progressive: ProgressiveConfig::default(),
            device_id: 0,
            max_seq_len: 4096,
            prefer_eager: true,
        }
    }

    /// Create configuration with explicit budgets.
    pub fn with_budgets(vram_gb: f64, ram_gb: f64) -> Self {
        Self {
            hardware: HardwareConfig {
                vram_budget: (vram_gb * GB as f64) as u64,
                ram_budget: (ram_gb * GB as f64) as u64,
                nvme_cache_size: 0,
                use_pinned_memory: true,
            },
            progressive: ProgressiveConfig::default(),
            device_id: 0,
            max_seq_len: 4096,
            prefer_eager: true,
        }
    }

    /// Select loading strategy based on allocation plan.
    pub fn select_strategy(&self, plan: &AllocationPlan) -> LoadingStrategy {
        LoadingStrategy::select(plan, self.prefer_eager)
    }
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self::for_24gb_gpu()
    }
}

/// Hardware resource configuration.
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Available VRAM budget in bytes.
    /// Should be total VRAM minus headroom for KV cache and working buffers.
    pub vram_budget: u64,

    /// Available RAM budget in bytes.
    /// Should be total RAM minus system usage.
    pub ram_budget: u64,

    /// Maximum NVMe cache size in bytes (0 = unlimited).
    pub nvme_cache_size: u64,

    /// Whether to use CUDA pinned memory for RAM tensors.
    /// Pinned memory enables faster GPU uploads (~12GB/s vs ~6GB/s).
    pub use_pinned_memory: bool,
}

impl HardwareConfig {
    /// Total available memory across all tiers.
    pub fn total_budget(&self) -> u64 {
        self.vram_budget + self.ram_budget + self.nvme_cache_size
    }

    /// Create config with auto-detected resources.
    ///
    /// Note: This is a placeholder. Real implementation would query
    /// CUDA and system APIs for actual available memory.
    pub fn auto_detect() -> Self {
        Self {
            vram_budget: 22 * GB, // Conservative 24GB GPU estimate
            ram_budget: 60 * GB,  // Conservative 64GB RAM estimate
            nvme_cache_size: 0,   // Unlimited
            use_pinned_memory: true,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Configuration for progressive (NVMe-backed) loading.
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Number of layers to prefetch ahead of current position.
    pub prefetch_depth: usize,

    /// Directory for NVMe tensor cache.
    pub cache_dir: PathBuf,

    /// Maximum cache size in bytes (0 = unlimited).
    pub max_cache_size: u64,

    /// Whether to cache decompressed tensors on NVMe.
    /// If false, always decompress from HCT files.
    pub enable_caching: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            prefetch_depth: 2,
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("infernum")
                .join("tensor_cache"),
            max_cache_size: 0, // Unlimited
            enable_caching: true,
        }
    }
}

/// Loading strategy determined by allocation plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingStrategy {
    /// Fast path: decompress all tensors upfront to VRAM + RAM.
    /// No NVMe tier needed. Expected: 2-10+ tk/s.
    Eager,

    /// Medium path: eager load with aggressive quantization.
    /// Used when model almost fits with some precision reduction.
    EagerQuantized {
        /// Precision for VRAM tensors.
        vram_precision: TensorPrecision,
        /// Precision for RAM tensors.
        ram_precision: TensorPrecision,
    },

    /// Slow path: stream from NVMe with prefetching.
    /// Used for 405B+ models. Expected: 0.1-1 tk/s.
    Progressive,
}

impl LoadingStrategy {
    /// Select strategy based on allocation plan.
    pub fn select(plan: &AllocationPlan, prefer_eager: bool) -> Self {
        // If model fits entirely in VRAM + RAM, use eager loading
        if plan.nvme_usage == 0 {
            return LoadingStrategy::Eager;
        }

        // If not eager-preferring, go straight to progressive
        if !prefer_eager {
            return LoadingStrategy::Progressive;
        }

        // Check if aggressive quantization would eliminate NVMe usage
        // This is a heuristic - INT4 gives ~4x compression
        let potential_vram_with_int4 = plan.vram_usage / 4;
        let potential_ram_with_int8 = plan.ram_usage / 2;
        let total_quantized = potential_vram_with_int4 + potential_ram_with_int8;

        if total_quantized < plan.nvme_usage {
            // Quantization would help significantly
            LoadingStrategy::EagerQuantized {
                vram_precision: TensorPrecision::INT4,
                ram_precision: TensorPrecision::INT8,
            }
        } else {
            // NVMe tier is unavoidable
            LoadingStrategy::Progressive
        }
    }

    /// Whether this strategy requires NVMe tier.
    pub fn needs_nvme(&self) -> bool {
        matches!(self, LoadingStrategy::Progressive)
    }

    /// Whether this strategy uses quantization.
    pub fn uses_quantization(&self) -> bool {
        matches!(self, LoadingStrategy::EagerQuantized { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_tiering::{MemoryTier, TensorAllocation};
    use std::collections::HashMap;

    fn make_plan(vram: u64, ram: u64, nvme: u64) -> AllocationPlan {
        AllocationPlan {
            allocations: HashMap::new(),
            vram_usage: vram,
            ram_usage: ram,
            nvme_usage: nvme,
            swap_count: if nvme > 0 { 10 } else { 0 },
            quality_score: 0.95,
        }
    }

    #[test]
    fn test_strategy_eager_when_fits() {
        let plan = make_plan(20 * GB, 40 * GB, 0);
        let strategy = LoadingStrategy::select(&plan, true);
        assert_eq!(strategy, LoadingStrategy::Eager);
    }

    #[test]
    fn test_strategy_progressive_when_nvme_needed() {
        let plan = make_plan(20 * GB, 60 * GB, 100 * GB);
        let strategy = LoadingStrategy::select(&plan, true);
        assert_eq!(strategy, LoadingStrategy::Progressive);
    }

    #[test]
    fn test_config_defaults() {
        let config = TieredConfig::default();
        assert_eq!(config.hardware.vram_budget, 22 * GB);
        assert!(config.hardware.use_pinned_memory);
    }
}
