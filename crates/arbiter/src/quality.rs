//! Quality budget and allocation for workloads.
//!
//! Manages quality targets based on available resources and priorities.

use serde::{Deserialize, Serialize};
use crate::priority::{Priority, WorkloadType};
use crate::memory::MemoryPressure;

/// Policy for quality allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPolicy {
    /// Fixed quality regardless of pressure.
    Fixed,
    /// Adaptive quality based on pressure.
    Adaptive,
    /// Prioritize LLM quality over diffusion.
    LlmFirst,
    /// Prioritize diffusion quality over LLM.
    DiffusionFirst,
    /// Equal quality for all workloads.
    Balanced,
}

impl Default for QualityPolicy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Quality allocation for a specific workload.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QualityAllocation {
    /// Target quality (0.0 - 1.0).
    pub target: f32,
    /// Minimum acceptable quality.
    pub minimum: f32,
    /// Whether quality can be dynamically adjusted.
    pub adjustable: bool,
}

impl Default for QualityAllocation {
    fn default() -> Self {
        Self {
            target: 1.0,
            minimum: 0.4,
            adjustable: true,
        }
    }
}

impl QualityAllocation {
    /// Creates allocation with specific target.
    pub fn with_target(target: f32) -> Self {
        Self {
            target: target.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Sets minimum quality.
    pub fn with_minimum(mut self, minimum: f32) -> Self {
        self.minimum = minimum.clamp(0.0, self.target);
        self
    }

    /// Sets whether quality is adjustable.
    pub fn fixed(mut self) -> Self {
        self.adjustable = false;
        self
    }

    /// Returns whether the allocation meets minimum requirements.
    pub fn is_viable(&self) -> bool {
        self.target >= self.minimum
    }

    /// Reduces quality by a factor, respecting minimum.
    pub fn reduce(&self, factor: f32) -> Self {
        if !self.adjustable {
            return *self;
        }
        Self {
            target: (self.target * factor).max(self.minimum),
            minimum: self.minimum,
            adjustable: self.adjustable,
        }
    }
}

/// Budget for quality across workload types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBudget {
    /// LLM inference quality.
    pub llm: QualityAllocation,
    /// Image generation quality.
    pub image: QualityAllocation,
    /// Video generation quality.
    pub video: QualityAllocation,
    /// Current policy.
    pub policy: QualityPolicy,
}

impl Default for QualityBudget {
    fn default() -> Self {
        Self {
            llm: QualityAllocation {
                target: 1.0,
                minimum: 0.4,
                adjustable: true,
            },
            image: QualityAllocation {
                target: 1.0,
                minimum: 0.3,
                adjustable: true,
            },
            video: QualityAllocation {
                target: 1.0,
                minimum: 0.25,
                adjustable: true,
            },
            policy: QualityPolicy::Adaptive,
        }
    }
}

impl QualityBudget {
    /// Gets quality allocation for a workload type.
    pub fn for_workload(&self, workload: WorkloadType) -> QualityAllocation {
        match workload {
            WorkloadType::LlmInference => self.llm,
            WorkloadType::ImageGeneration => self.image,
            WorkloadType::VideoGeneration => self.video,
        }
    }

    /// Applies pressure to the budget, reducing qualities.
    pub fn apply_pressure(&self, pressure: MemoryPressure) -> Self {
        let factor = pressure.quality_factor();
        Self {
            llm: self.llm.reduce(factor),
            image: self.image.reduce(factor),
            video: self.video.reduce(factor),
            policy: self.policy,
        }
    }

    /// Creates budget with LLM-first policy.
    pub fn llm_first() -> Self {
        Self {
            llm: QualityAllocation {
                target: 1.0,
                minimum: 0.6,
                adjustable: true,
            },
            image: QualityAllocation {
                target: 0.8,
                minimum: 0.3,
                adjustable: true,
            },
            video: QualityAllocation {
                target: 0.7,
                minimum: 0.25,
                adjustable: true,
            },
            policy: QualityPolicy::LlmFirst,
        }
    }

    /// Creates budget with diffusion-first policy.
    pub fn diffusion_first() -> Self {
        Self {
            llm: QualityAllocation {
                target: 0.8,
                minimum: 0.4,
                adjustable: true,
            },
            image: QualityAllocation {
                target: 1.0,
                minimum: 0.5,
                adjustable: true,
            },
            video: QualityAllocation {
                target: 1.0,
                minimum: 0.4,
                adjustable: true,
            },
            policy: QualityPolicy::DiffusionFirst,
        }
    }

    /// Creates balanced budget.
    pub fn balanced() -> Self {
        Self {
            llm: QualityAllocation {
                target: 0.85,
                minimum: 0.4,
                adjustable: true,
            },
            image: QualityAllocation {
                target: 0.85,
                minimum: 0.3,
                adjustable: true,
            },
            video: QualityAllocation {
                target: 0.85,
                minimum: 0.25,
                adjustable: true,
            },
            policy: QualityPolicy::Balanced,
        }
    }
}

/// Calculates quality allocation based on context.
pub struct QualityCalculator {
    /// Base budget.
    budget: QualityBudget,
    /// LLM minimum quality.
    llm_min: f32,
    /// Diffusion minimum quality.
    diffusion_min: f32,
}

impl QualityCalculator {
    /// Creates a new calculator with minimums.
    pub fn new(llm_min: f32, diffusion_min: f32) -> Self {
        Self {
            budget: QualityBudget::default(),
            llm_min,
            diffusion_min,
        }
    }

    /// Calculates quality for a workload given current pressure.
    pub fn calculate(
        &self,
        workload: WorkloadType,
        priority: Priority,
        pressure: f32,
    ) -> f32 {
        let base = self.budget.for_workload(workload);
        let pressure_factor = 1.0 - (pressure * 0.5); // 50% max reduction
        let priority_factor = priority.quality_multiplier();

        let min_quality = match workload {
            WorkloadType::LlmInference => self.llm_min,
            WorkloadType::ImageGeneration | WorkloadType::VideoGeneration => self.diffusion_min,
        };

        (base.target * pressure_factor * priority_factor)
            .max(min_quality)
            .min(1.0) // Clamp to maximum 1.0
    }

    /// Updates the budget.
    pub fn set_budget(&mut self, budget: QualityBudget) {
        self.budget = budget;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_allocation_reduce() {
        let alloc = QualityAllocation {
            target: 1.0,
            minimum: 0.4,
            adjustable: true,
        };

        let reduced = alloc.reduce(0.5);
        assert!((reduced.target - 0.5).abs() < 0.001);

        // Should not go below minimum
        let heavily_reduced = alloc.reduce(0.1);
        assert!((heavily_reduced.target - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_quality_allocation_fixed() {
        let alloc = QualityAllocation::with_target(0.8).fixed();
        let reduced = alloc.reduce(0.5);

        // Fixed allocation should not change
        assert!((reduced.target - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_budget_pressure() {
        let budget = QualityBudget::default();
        let under_pressure = budget.apply_pressure(MemoryPressure::High);

        assert!(under_pressure.llm.target < budget.llm.target);
        assert!(under_pressure.image.target < budget.image.target);
    }

    #[test]
    fn test_calculator() {
        let calc = QualityCalculator::new(0.4, 0.3);

        // No pressure, normal priority
        let q1 = calc.calculate(WorkloadType::LlmInference, Priority::Normal, 0.0);
        assert!((q1 - 1.0).abs() < 0.001);

        // High pressure should reduce quality
        let q2 = calc.calculate(WorkloadType::LlmInference, Priority::Normal, 0.8);
        assert!(q2 < q1);

        // High priority should increase quality
        let q3 = calc.calculate(WorkloadType::LlmInference, Priority::High, 0.8);
        assert!(q3 > q2);
    }

    #[test]
    fn test_workload_minimums() {
        let calc = QualityCalculator::new(0.4, 0.3);

        // Even at maximum pressure, should not go below minimum
        let q = calc.calculate(WorkloadType::LlmInference, Priority::Background, 1.0);
        assert!(q >= 0.4);
    }
}
