//! System memory detection and monitoring.
//!
//! Provides cross-platform detection of system RAM and available memory
//! for fragment placement decisions.

use std::sync::OnceLock;
use sysinfo::System;

/// Cached system info for performance.
static SYSTEM_INFO: OnceLock<SystemMemoryInfo> = OnceLock::new();

/// System memory information.
#[derive(Debug, Clone, Copy)]
pub struct SystemMemoryInfo {
    /// Total physical RAM in bytes.
    pub total_bytes: u64,

    /// Available RAM in bytes (at detection time).
    pub available_bytes: u64,

    /// Used RAM in bytes.
    pub used_bytes: u64,

    /// Swap total in bytes.
    pub swap_total_bytes: u64,

    /// Swap used in bytes.
    pub swap_used_bytes: u64,
}

impl SystemMemoryInfo {
    /// Detects current system memory.
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_memory();

        Self {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
            used_bytes: sys.used_memory(),
            swap_total_bytes: sys.total_swap(),
            swap_used_bytes: sys.used_swap(),
        }
    }

    /// Gets cached system memory info (faster after first call).
    pub fn cached() -> Self {
        *SYSTEM_INFO.get_or_init(Self::detect)
    }

    /// Refreshes and returns updated memory info.
    pub fn refresh() -> Self {
        let info = Self::detect();
        // Note: OnceLock doesn't support updating, so cached() may be stale
        // Use detect() for real-time info
        info
    }

    /// Returns total RAM in gigabytes.
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns available RAM in gigabytes.
    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns memory utilization (0.0 - 1.0).
    pub fn utilization(&self) -> f32 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.total_bytes as f32
    }

    /// Returns a safe RAM budget for inference (reserves memory for system).
    ///
    /// Uses 70% of available memory to leave headroom for OS and other apps.
    pub fn safe_budget_bytes(&self) -> u64 {
        (self.available_bytes as f64 * 0.7) as u64
    }

    /// Returns a safe RAM budget for inference in GB.
    pub fn safe_budget_gb(&self) -> f64 {
        self.safe_budget_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Checks if there's enough memory for a given operation.
    pub fn has_available(&self, bytes: u64) -> bool {
        self.available_bytes >= bytes
    }

    /// Returns memory pressure level.
    pub fn pressure(&self) -> MemoryPressure {
        let util = self.utilization();
        if util < 0.5 {
            MemoryPressure::Low
        } else if util < 0.75 {
            MemoryPressure::Moderate
        } else if util < 0.9 {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        }
    }
}

impl Default for SystemMemoryInfo {
    fn default() -> Self {
        Self::detect()
    }
}

/// Memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// Plenty of memory available (< 50%).
    Low,
    /// Moderate usage (50-75%).
    Moderate,
    /// High usage (75-90%).
    High,
    /// Critical usage (> 90%).
    Critical,
}

impl MemoryPressure {
    /// Returns whether new allocations should be cautious.
    pub fn should_be_cautious(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }

    /// Returns whether we should avoid large allocations.
    pub fn should_avoid_large_allocs(self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Returns a quality factor based on memory pressure.
    pub fn quality_factor(self) -> f32 {
        match self {
            Self::Low => 1.0,
            Self::Moderate => 0.9,
            Self::High => 0.7,
            Self::Critical => 0.5,
        }
    }
}

/// Recommended memory configuration based on system detection.
#[derive(Debug, Clone)]
pub struct RecommendedConfig {
    /// Recommended VRAM budget in bytes.
    pub vram_bytes: u64,

    /// Recommended RAM budget in bytes.
    pub ram_bytes: u64,

    /// Whether NVMe caching is recommended.
    pub use_nvme_cache: bool,

    /// Explanation of the recommendation.
    pub explanation: String,
}

impl RecommendedConfig {
    /// Generates recommended config based on system detection.
    ///
    /// If `gpu_vram_bytes` is provided, uses that for VRAM budget.
    /// Otherwise estimates based on common GPU configurations.
    pub fn detect(gpu_vram_bytes: Option<u64>) -> Self {
        let mem = SystemMemoryInfo::detect();
        let ram_budget = mem.safe_budget_bytes();

        // Use provided GPU VRAM or estimate
        let vram = gpu_vram_bytes.unwrap_or_else(|| {
            // Heuristic: assume GPU VRAM is roughly 25% of system RAM
            // (e.g., 64GB RAM often paired with 16GB VRAM)
            let estimated = (mem.total_bytes as f64 * 0.25) as u64;
            // Clamp to reasonable range (4GB - 48GB)
            estimated.clamp(4 * 1024 * 1024 * 1024, 48 * 1024 * 1024 * 1024)
        });

        // Recommend NVMe cache if RAM is limited relative to typical model sizes
        let use_nvme = ram_budget < 32 * 1024 * 1024 * 1024;

        let explanation = format!(
            "System: {:.1}GB RAM ({:.1}GB available), estimated {:.1}GB VRAM. \
             RAM budget: {:.1}GB (70% of available). {}",
            mem.total_gb(),
            mem.available_gb(),
            vram as f64 / (1024.0 * 1024.0 * 1024.0),
            ram_budget as f64 / (1024.0 * 1024.0 * 1024.0),
            if use_nvme {
                "NVMe cache recommended for large models."
            } else {
                "Sufficient RAM for most models."
            }
        );

        Self {
            vram_bytes: vram,
            ram_bytes: ram_budget,
            use_nvme_cache: use_nvme,
            explanation,
        }
    }

    /// Creates config with explicit GPU VRAM.
    pub fn with_gpu(gpu_vram_bytes: u64) -> Self {
        Self::detect(Some(gpu_vram_bytes))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_memory_detection() {
        let mem = SystemMemoryInfo::detect();

        // Should detect some memory
        assert!(mem.total_bytes > 0, "Total memory should be > 0");
        assert!(mem.available_bytes > 0, "Available memory should be > 0");
        assert!(mem.available_bytes <= mem.total_bytes);
    }

    #[test]
    fn test_memory_gb_conversion() {
        let mem = SystemMemoryInfo {
            total_bytes: 64 * 1024 * 1024 * 1024,     // 64 GB
            available_bytes: 48 * 1024 * 1024 * 1024, // 48 GB
            used_bytes: 16 * 1024 * 1024 * 1024,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };

        assert!((mem.total_gb() - 64.0).abs() < 0.01);
        assert!((mem.available_gb() - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_utilization() {
        let mem = SystemMemoryInfo {
            total_bytes: 100,
            available_bytes: 40,
            used_bytes: 60,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };

        assert!((mem.utilization() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_memory_utilization_zero_total() {
        let mem = SystemMemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
            used_bytes: 0,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };

        assert_eq!(mem.utilization(), 0.0);
    }

    #[test]
    fn test_safe_budget() {
        let mem = SystemMemoryInfo {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 48 * 1024 * 1024 * 1024,
            used_bytes: 16 * 1024 * 1024 * 1024,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };

        // 70% of 48GB = 33.6GB
        let budget = mem.safe_budget_bytes();
        assert!(budget > 30 * 1024 * 1024 * 1024);
        assert!(budget < 40 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_pressure_levels() {
        assert_eq!(
            SystemMemoryInfo {
                total_bytes: 100,
                used_bytes: 30,
                available_bytes: 70,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            }
            .pressure(),
            MemoryPressure::Low
        );

        assert_eq!(
            SystemMemoryInfo {
                total_bytes: 100,
                used_bytes: 60,
                available_bytes: 40,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            }
            .pressure(),
            MemoryPressure::Moderate
        );

        assert_eq!(
            SystemMemoryInfo {
                total_bytes: 100,
                used_bytes: 85,
                available_bytes: 15,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            }
            .pressure(),
            MemoryPressure::High
        );

        assert_eq!(
            SystemMemoryInfo {
                total_bytes: 100,
                used_bytes: 95,
                available_bytes: 5,
                swap_total_bytes: 0,
                swap_used_bytes: 0,
            }
            .pressure(),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn test_has_available() {
        let mem = SystemMemoryInfo {
            total_bytes: 64 * 1024 * 1024 * 1024,
            available_bytes: 32 * 1024 * 1024 * 1024,
            used_bytes: 32 * 1024 * 1024 * 1024,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
        };

        assert!(mem.has_available(16 * 1024 * 1024 * 1024)); // 16GB
        assert!(mem.has_available(32 * 1024 * 1024 * 1024)); // 32GB (exact)
        assert!(!mem.has_available(64 * 1024 * 1024 * 1024)); // 64GB (too much)
    }

    #[test]
    fn test_recommended_config() {
        let config = RecommendedConfig::detect(Some(24 * 1024 * 1024 * 1024));

        // Should use provided VRAM
        assert_eq!(config.vram_bytes, 24 * 1024 * 1024 * 1024);

        // RAM budget should be reasonable
        assert!(config.ram_bytes > 0);

        // Explanation should be non-empty
        assert!(!config.explanation.is_empty());
    }

    #[test]
    fn test_recommended_config_no_gpu() {
        let config = RecommendedConfig::detect(None);

        // Should estimate VRAM
        assert!(config.vram_bytes >= 4 * 1024 * 1024 * 1024); // At least 4GB
        assert!(config.vram_bytes <= 48 * 1024 * 1024 * 1024); // At most 48GB
    }

    #[test]
    fn test_cached_memory_info() {
        let cached1 = SystemMemoryInfo::cached();
        let cached2 = SystemMemoryInfo::cached();

        // Cached values should be the same
        assert_eq!(cached1.total_bytes, cached2.total_bytes);
    }

    #[test]
    fn test_pressure_quality_factor() {
        assert_eq!(MemoryPressure::Low.quality_factor(), 1.0);
        assert_eq!(MemoryPressure::Moderate.quality_factor(), 0.9);
        assert_eq!(MemoryPressure::High.quality_factor(), 0.7);
        assert_eq!(MemoryPressure::Critical.quality_factor(), 0.5);
    }

    #[test]
    fn test_pressure_should_be_cautious() {
        assert!(!MemoryPressure::Low.should_be_cautious());
        assert!(!MemoryPressure::Moderate.should_be_cautious());
        assert!(MemoryPressure::High.should_be_cautious());
        assert!(MemoryPressure::Critical.should_be_cautious());
    }
}
