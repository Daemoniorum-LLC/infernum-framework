//! Coordinator for quality targets between workloads.
//!
//! The coordinator dynamically adjusts quality targets based on
//! memory pressure and workload priorities.

use serde::{Deserialize, Serialize};
use crate::priority::{Priority, WorkloadType};
use crate::quality::{QualityCalculator, QualityPolicy};

/// Configuration for the coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Minimum quality for LLM inference.
    pub llm_min_quality: f32,
    /// Minimum quality for diffusion.
    pub diffusion_min_quality: f32,
    /// Enable adaptive quality balancing.
    pub adaptive: bool,
    /// Policy for quality allocation.
    pub policy: QualityPolicy,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            llm_min_quality: 0.4,
            diffusion_min_quality: 0.3,
            adaptive: true,
            policy: QualityPolicy::Adaptive,
        }
    }
}

impl CoordinatorConfig {
    /// Creates LLM-first configuration.
    pub fn llm_first() -> Self {
        Self {
            llm_min_quality: 0.6,
            diffusion_min_quality: 0.25,
            adaptive: true,
            policy: QualityPolicy::LlmFirst,
        }
    }

    /// Creates diffusion-first configuration.
    pub fn diffusion_first() -> Self {
        Self {
            llm_min_quality: 0.35,
            diffusion_min_quality: 0.5,
            adaptive: true,
            policy: QualityPolicy::DiffusionFirst,
        }
    }

    /// Creates balanced configuration.
    pub fn balanced() -> Self {
        Self {
            llm_min_quality: 0.4,
            diffusion_min_quality: 0.35,
            adaptive: true,
            policy: QualityPolicy::Balanced,
        }
    }
}

/// The quality coordinator.
pub struct Coordinator {
    config: CoordinatorConfig,
    calculator: QualityCalculator,
}

impl Coordinator {
    /// Creates a new coordinator.
    pub fn new(config: CoordinatorConfig) -> Self {
        let calculator = QualityCalculator::new(
            config.llm_min_quality,
            config.diffusion_min_quality,
        );

        Self { config, calculator }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Calculates quality for a workload given current pressure.
    pub fn calculate_quality(
        &self,
        workload: WorkloadType,
        priority: Priority,
        pressure: f32,
    ) -> f32 {
        if !self.config.adaptive {
            // Fixed quality mode
            return match workload {
                WorkloadType::LlmInference => 1.0,
                WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => 1.0,
            };
        }

        // Apply policy-based adjustments
        let base_quality = self.calculator.calculate(workload, priority, pressure);

        match self.config.policy {
            QualityPolicy::Fixed => 1.0,
            QualityPolicy::Adaptive => base_quality,
            QualityPolicy::LlmFirst => {
                match workload {
                    WorkloadType::LlmInference => base_quality.max(0.7),
                    _ => base_quality * 0.9,
                }
            }
            QualityPolicy::DiffusionFirst => {
                match workload {
                    WorkloadType::LlmInference => base_quality * 0.9,
                    _ => base_quality.max(0.7),
                }
            }
            QualityPolicy::Balanced => {
                // Compress range to 0.5-0.95
                0.5 + base_quality * 0.45
            }
        }
    }

    /// Returns minimum quality for a workload type.
    pub fn min_quality(&self, workload: WorkloadType) -> f32 {
        match workload {
            WorkloadType::LlmInference => self.config.llm_min_quality,
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => {
                self.config.diffusion_min_quality
            }
        }
    }

    /// Suggests quality adjustments when both systems are active.
    pub fn suggest_rebalance(
        &self,
        llm_active: bool,
        diffusion_active: bool,
        pressure: f32,
    ) -> QualitySuggestion {
        if !llm_active && !diffusion_active {
            return QualitySuggestion::full();
        }

        if !llm_active {
            return QualitySuggestion {
                llm_quality: 1.0,
                diffusion_quality: 1.0,
            };
        }

        if !diffusion_active {
            return QualitySuggestion {
                llm_quality: 1.0,
                diffusion_quality: 1.0,
            };
        }

        // Both active - calculate based on pressure and policy
        let pressure_factor = 1.0 - pressure * 0.3;

        match self.config.policy {
            QualityPolicy::LlmFirst => QualitySuggestion {
                llm_quality: (0.9 * pressure_factor).max(self.config.llm_min_quality),
                diffusion_quality: (0.7 * pressure_factor).max(self.config.diffusion_min_quality),
            },
            QualityPolicy::DiffusionFirst => QualitySuggestion {
                llm_quality: (0.7 * pressure_factor).max(self.config.llm_min_quality),
                diffusion_quality: (0.9 * pressure_factor).max(self.config.diffusion_min_quality),
            },
            QualityPolicy::Balanced | QualityPolicy::Adaptive => QualitySuggestion {
                llm_quality: (0.8 * pressure_factor).max(self.config.llm_min_quality),
                diffusion_quality: (0.75 * pressure_factor).max(self.config.diffusion_min_quality),
            },
            QualityPolicy::Fixed => QualitySuggestion::full(),
        }
    }
}

/// Suggested quality levels for rebalancing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualitySuggestion {
    /// Suggested LLM quality.
    pub llm_quality: f32,
    /// Suggested diffusion quality.
    pub diffusion_quality: f32,
}

impl QualitySuggestion {
    /// Full quality for all workloads.
    pub fn full() -> Self {
        Self {
            llm_quality: 1.0,
            diffusion_quality: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_default() {
        let coord = Coordinator::new(CoordinatorConfig::default());

        // No pressure should give high quality
        let q = coord.calculate_quality(WorkloadType::LlmInference, Priority::Normal, 0.0);
        assert!(q > 0.9);
    }

    #[test]
    fn test_coordinator_under_pressure() {
        let coord = Coordinator::new(CoordinatorConfig::default());

        let q_low = coord.calculate_quality(WorkloadType::LlmInference, Priority::Normal, 0.0);
        let q_high = coord.calculate_quality(WorkloadType::LlmInference, Priority::Normal, 0.9);

        assert!(q_low > q_high);
        assert!(q_high >= coord.config().llm_min_quality);
    }

    #[test]
    fn test_llm_first_policy() {
        let coord = Coordinator::new(CoordinatorConfig::llm_first());

        let llm_q = coord.calculate_quality(WorkloadType::LlmInference, Priority::Normal, 0.5);
        let img_q = coord.calculate_quality(WorkloadType::ImageGeneration, Priority::Normal, 0.5);

        assert!(llm_q >= img_q);
    }

    #[test]
    fn test_diffusion_first_policy() {
        let coord = Coordinator::new(CoordinatorConfig::diffusion_first());

        let llm_q = coord.calculate_quality(WorkloadType::LlmInference, Priority::Normal, 0.5);
        let img_q = coord.calculate_quality(WorkloadType::ImageGeneration, Priority::Normal, 0.5);

        assert!(img_q >= llm_q);
    }

    #[test]
    fn test_rebalance_suggestions() {
        let coord = Coordinator::new(CoordinatorConfig::balanced());

        // Only LLM active
        let s1 = coord.suggest_rebalance(true, false, 0.5);
        assert_eq!(s1.llm_quality, 1.0);
        assert_eq!(s1.diffusion_quality, 1.0);

        // Both active
        let s2 = coord.suggest_rebalance(true, true, 0.5);
        assert!(s2.llm_quality < 1.0);
        assert!(s2.diffusion_quality < 1.0);
    }

    #[test]
    fn test_non_adaptive_mode() {
        let mut config = CoordinatorConfig::default();
        config.adaptive = false;

        let coord = Coordinator::new(config);

        // Should always return full quality
        let q = coord.calculate_quality(WorkloadType::LlmInference, Priority::Low, 0.9);
        assert_eq!(q, 1.0);
    }
}
