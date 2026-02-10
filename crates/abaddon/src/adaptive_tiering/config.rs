//! Configuration for the adaptive memory tiering system.

/// Configuration for adaptive memory tiering.
///
/// Controls how tensors are allocated across memory tiers (VRAM, RAM, NVMe)
/// and what precision levels are acceptable.
#[derive(Debug, Clone)]
pub struct AdaptiveTieringConfig {
    /// VRAM budget in bytes.
    ///
    /// Set to 0 for auto-detection (total VRAM - 2GB headroom).
    pub vram_budget: u64,

    /// RAM budget in bytes.
    ///
    /// Set to 0 for auto-detection (total RAM - 4GB headroom).
    pub ram_budget: u64,

    /// Minimum quality target [0.0, 1.0].
    ///
    /// Higher values prefer BF16, lower values allow more aggressive quantization.
    /// Default: 0.95
    pub quality_target: f32,

    /// Maximum acceptable tensor swap latency in milliseconds.
    ///
    /// If estimated swap latency exceeds this, prefer lower precision in VRAM
    /// over swapping from RAM/NVMe.
    /// Default: 100ms
    pub max_swap_latency_ms: u32,

    /// Enable runtime adaptation based on access patterns.
    ///
    /// When enabled, the system monitors tensor access during inference
    /// and dynamically adjusts placement.
    /// Default: true
    pub enable_adaptation: bool,

    /// Number of layers to prefetch ahead during inference.
    ///
    /// Default: 2
    pub prefetch_depth: usize,

    /// Enable mixed precision in VRAM.
    ///
    /// When true, different tensors can use different precisions in VRAM.
    /// When false, all VRAM tensors use the same (highest possible) precision.
    /// Default: true
    pub enable_mixed_precision: bool,

    /// Importance threshold below which tensors can be placed in RAM.
    ///
    /// Tensors with importance >= this threshold are prioritized for VRAM.
    /// Default: 0.7
    pub vram_importance_threshold: f32,

    /// Minimum VRAM headroom to reserve for KV cache growth (bytes).
    ///
    /// Default: 2GB
    pub kv_cache_headroom: u64,
}

impl Default for AdaptiveTieringConfig {
    fn default() -> Self {
        Self {
            vram_budget: 0, // Auto-detect
            ram_budget: 0,  // Auto-detect
            quality_target: 0.95,
            max_swap_latency_ms: 100,
            enable_adaptation: true,
            prefetch_depth: 2,
            enable_mixed_precision: true,
            vram_importance_threshold: 0.7,
            kv_cache_headroom: 2 * 1024 * 1024 * 1024, // 2GB
        }
    }
}

impl AdaptiveTieringConfig {
    /// Creates a config optimized for maximum quality (prefers BF16, less quantization).
    pub fn high_quality() -> Self {
        Self {
            quality_target: 0.99,
            enable_mixed_precision: false,
            ..Default::default()
        }
    }

    /// Creates a config optimized for fitting larger models (aggressive quantization).
    pub fn memory_optimized() -> Self {
        Self {
            quality_target: 0.90,
            enable_mixed_precision: true,
            vram_importance_threshold: 0.5,
            ..Default::default()
        }
    }

    /// Creates a config with explicit memory budgets.
    pub fn with_budgets(vram_gb: f64, ram_gb: f64) -> Self {
        Self {
            vram_budget: (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            ram_budget: (ram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            ..Default::default()
        }
    }

    /// Returns effective VRAM budget, auto-detecting if not set.
    ///
    /// # Arguments
    /// * `detected_vram` - Total VRAM detected on the system (bytes)
    pub fn effective_vram_budget(&self, detected_vram: u64) -> u64 {
        if self.vram_budget > 0 {
            self.vram_budget
        } else {
            // Auto-detect: total VRAM minus headroom for KV cache and OS
            detected_vram.saturating_sub(self.kv_cache_headroom)
        }
    }

    /// Returns effective RAM budget, auto-detecting if not set.
    ///
    /// # Arguments
    /// * `detected_ram` - Total RAM detected on the system (bytes)
    pub fn effective_ram_budget(&self, detected_ram: u64) -> u64 {
        if self.ram_budget > 0 {
            self.ram_budget
        } else {
            // Auto-detect: total RAM minus 4GB for OS/applications
            let headroom = 4 * 1024 * 1024 * 1024; // 4GB
            detected_ram.saturating_sub(headroom)
        }
    }

    /// Validates the configuration and returns any issues.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.quality_target < 0.0 || self.quality_target > 1.0 {
            return Err(ConfigError::InvalidQualityTarget(self.quality_target));
        }
        if self.vram_importance_threshold < 0.0 || self.vram_importance_threshold > 1.0 {
            return Err(ConfigError::InvalidImportanceThreshold(
                self.vram_importance_threshold,
            ));
        }
        Ok(())
    }
}

/// Configuration validation errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    /// Quality target must be in [0.0, 1.0].
    #[error("quality_target must be in [0.0, 1.0], got {0}")]
    InvalidQualityTarget(f32),

    /// Importance threshold must be in [0.0, 1.0].
    #[error("vram_importance_threshold must be in [0.0, 1.0], got {0}")]
    InvalidImportanceThreshold(f32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AdaptiveTieringConfig::default();
        assert!((config.quality_target - 0.95).abs() < f32::EPSILON);
        assert!(config.enable_mixed_precision);
        assert!(config.enable_adaptation);
    }

    #[test]
    fn test_effective_vram_budget_auto() {
        let config = AdaptiveTieringConfig::default();
        let detected_vram = 24 * 1024 * 1024 * 1024u64; // 24GB
        let effective = config.effective_vram_budget(detected_vram);
        // Should be 24GB - 2GB headroom = 22GB
        assert_eq!(effective, 22 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_effective_vram_budget_explicit() {
        let config = AdaptiveTieringConfig::with_budgets(20.0, 60.0);
        let detected_vram = 24 * 1024 * 1024 * 1024u64;
        let effective = config.effective_vram_budget(detected_vram);
        // Should use explicit value
        assert_eq!(effective, 20 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_validation() {
        let mut config = AdaptiveTieringConfig::default();
        assert!(config.validate().is_ok());

        config.quality_target = 1.5;
        assert!(config.validate().is_err());

        config.quality_target = 0.95;
        config.vram_importance_threshold = -0.1;
        assert!(config.validate().is_err());
    }
}
