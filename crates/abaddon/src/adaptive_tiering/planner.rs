//! Allocation planner for adaptive memory tiering.
//!
//! Uses a greedy algorithm to place tensors across memory tiers (VRAM, RAM, NVMe)
//! while respecting memory budgets and maximizing inference quality.
//!
//! # Algorithm Overview
//!
//! 1. Score all tensors by importance
//! 2. Sort by importance (highest first)
//! 3. For each tensor, try to place in VRAM with best precision
//! 4. If VRAM full, try lower precision or fall back to RAM/NVMe
//! 5. Track total usage and swap count

use super::config::AdaptiveTieringConfig;
use super::importance::ImportanceScorer;
use super::types::{
    AllocationPlan, MemoryTier, ModelProfile, TensorAllocation, TensorInfo, TensorPrecision,
    TensorType,
};

/// Errors from the allocation planner.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PlannerError {
    /// Model doesn't fit in any configuration.
    #[error("model ({model_size_gb:.1}GB) cannot fit: VRAM budget {vram_gb:.1}GB, RAM budget {ram_gb:.1}GB")]
    ModelTooLarge {
        /// Model size in gigabytes.
        model_size_gb: f64,
        /// Available VRAM budget in gigabytes.
        vram_gb: f64,
        /// Available RAM budget in gigabytes.
        ram_gb: f64,
    },

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// No tensors to allocate.
    #[error("no tensors to allocate")]
    EmptyModel,
}

/// Adaptive allocation planner.
///
/// Plans tensor placement across memory tiers to maximize inference quality
/// within hardware constraints.
pub struct AllocationPlanner {
    config: AdaptiveTieringConfig,
}

impl AllocationPlanner {
    /// Creates a new allocation planner with the given configuration.
    pub fn new(config: AdaptiveTieringConfig) -> Self {
        Self { config }
    }

    /// Plans allocation for a model profile.
    ///
    /// Returns an allocation plan mapping each tensor to a memory tier and precision.
    pub fn plan(&self, profile: &ModelProfile) -> Result<AllocationPlan, PlannerError> {
        if profile.tensors.is_empty() {
            return Err(PlannerError::EmptyModel);
        }

        // Use auto-detected budgets if not explicitly set
        // For now, use configured values; auto-detection happens at integration layer
        let vram_budget = if self.config.vram_budget > 0 {
            self.config.vram_budget
        } else {
            // Default to 22GB for auto-detect (24GB - 2GB headroom)
            22 * 1024 * 1024 * 1024
        };

        let ram_budget = if self.config.ram_budget > 0 {
            self.config.ram_budget
        } else {
            // Default to 60GB for auto-detect
            60 * 1024 * 1024 * 1024
        };

        // Create importance scorer
        let max_tensor_size = profile
            .tensors
            .iter()
            .map(|t| t.size_bytes)
            .max()
            .unwrap_or(1);
        let scorer = ImportanceScorer::new(profile.num_layers, max_tensor_size);

        // Score and sort tensors by importance (highest first)
        let mut scored_tensors: Vec<_> = profile
            .tensors
            .iter()
            .map(|t| (t, scorer.score(t)))
            .collect();
        scored_tensors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy allocation
        let mut plan = AllocationPlan::new();
        let mut vram_used = 0u64;
        let mut ram_used = 0u64;
        let mut nvme_used = 0u64;
        let mut total_quality_weighted = 0.0f64;
        let mut total_importance = 0.0f64;

        for (tensor, importance) in scored_tensors {
            let allocation = self.allocate_tensor(
                tensor,
                importance,
                vram_budget,
                ram_budget,
                vram_used,
                ram_used,
            );

            // Update usage counters
            match allocation.tier {
                MemoryTier::Vram => vram_used += allocation.storage_size,
                MemoryTier::Ram => ram_used += allocation.storage_size,
                MemoryTier::Nvme => nvme_used += allocation.storage_size,
            }

            // Track quality contribution
            total_quality_weighted +=
                importance as f64 * allocation.precision.quality_factor() as f64;
            total_importance += importance as f64;

            plan.allocations.insert(tensor.name.clone(), allocation);
        }

        // Compute swap count (tensors not in VRAM)
        plan.swap_count = plan
            .allocations
            .values()
            .filter(|a| a.tier != MemoryTier::Vram)
            .count();

        plan.vram_usage = vram_used;
        plan.ram_usage = ram_used;
        plan.nvme_usage = nvme_used;

        // Weighted quality score
        plan.quality_score = if total_importance > 0.0 {
            (total_quality_weighted / total_importance) as f32
        } else {
            1.0
        };

        Ok(plan)
    }

    /// Allocates a single tensor using greedy strategy.
    fn allocate_tensor(
        &self,
        tensor: &TensorInfo,
        importance: f32,
        vram_budget: u64,
        ram_budget: u64,
        vram_used: u64,
        ram_used: u64,
    ) -> TensorAllocation {
        // Critical tensors (embeddings, lm_head) must go to VRAM at BF16 if possible
        let is_critical = matches!(
            tensor.tensor_type,
            TensorType::Embedding | TensorType::LmHead
        );

        // Determine precisions to try based on config and tensor criticality
        let precisions = if self.config.enable_mixed_precision {
            if is_critical {
                // Critical tensors: prefer BF16, only fall back if necessary
                &[TensorPrecision::BF16, TensorPrecision::INT8][..]
            } else if importance >= self.config.vram_importance_threshold {
                // High importance: try all precisions, prefer higher quality
                TensorPrecision::all_by_quality()
            } else {
                // Lower importance: can use lower precision more readily
                &[
                    TensorPrecision::INT8,
                    TensorPrecision::INT4,
                    TensorPrecision::BF16,
                ][..]
            }
        } else {
            // No mixed precision: only BF16
            &[TensorPrecision::BF16][..]
        };

        // Try to fit in VRAM at various precisions
        for &precision in precisions {
            // Skip if precision quality is below target for high-importance tensors
            if importance > 0.8 && precision.quality_factor() < self.config.quality_target {
                continue;
            }

            let size = precision.storage_size(tensor.size_bytes);
            if vram_used + size <= vram_budget {
                return TensorAllocation {
                    tier: MemoryTier::Vram,
                    precision,
                    priority: importance,
                    prefetch: false, // No prefetch needed for VRAM-resident
                    storage_size: size,
                };
            }
        }

        // VRAM full - try RAM (BF16 only, no point quantizing for RAM)
        if ram_used + tensor.size_bytes <= ram_budget {
            return TensorAllocation {
                tier: MemoryTier::Ram,
                precision: TensorPrecision::BF16,
                priority: importance,
                prefetch: importance > 0.6, // Prefetch moderately important tensors
                storage_size: tensor.size_bytes,
            };
        }

        // Fall back to NVMe
        TensorAllocation {
            tier: MemoryTier::Nvme,
            precision: TensorPrecision::BF16,
            priority: importance,
            prefetch: importance > 0.8, // Only prefetch high importance from NVMe
            storage_size: tensor.size_bytes,
        }
    }

    /// Replans allocation with updated constraints (e.g., KV cache growth).
    ///
    /// # Arguments
    /// * `current` - Current allocation plan
    /// * `new_vram_budget` - New VRAM budget after KV cache growth
    pub fn replan_for_vram_pressure(
        &self,
        current: &AllocationPlan,
        new_vram_budget: u64,
    ) -> AllocationPlan {
        let mut new_plan = current.clone();

        if current.vram_usage <= new_vram_budget {
            return new_plan; // No changes needed
        }

        let to_evict = current.vram_usage - new_vram_budget;
        let mut evicted = 0u64;

        // Sort VRAM tensors by priority (lowest first for eviction)
        let mut vram_tensors: Vec<_> = new_plan
            .allocations
            .iter()
            .filter(|(_, a)| a.tier == MemoryTier::Vram)
            .map(|(name, a)| (name.clone(), a.priority, a.storage_size))
            .collect();

        vram_tensors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (name, _priority, size) in vram_tensors {
            if evicted >= to_evict {
                break;
            }

            // Demote to RAM
            if let Some(alloc) = new_plan.allocations.get_mut(&name) {
                alloc.tier = MemoryTier::Ram;
                alloc.prefetch = true;
                new_plan.vram_usage -= size;
                new_plan.ram_usage += alloc.storage_size;
                new_plan.swap_count += 1;
                evicted += size;
            }
        }

        new_plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_14b_profile() -> ModelProfile {
        // Simulate a 14B model with 48 layers
        let mut tensors = Vec::new();

        // Embeddings (~500MB)
        tensors.push(TensorInfo::from_name(
            "model.embed_tokens.weight",
            500_000_000,
        ));

        // lm_head (~500MB)
        tensors.push(TensorInfo::from_name("lm_head.weight", 500_000_000));

        // 48 layers, each with attention + mlp (~600MB per layer)
        for i in 0..48 {
            // Attention weights (~150MB)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                40_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                10_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                10_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                40_000_000,
            ));

            // MLP weights (~400MB)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                150_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                150_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                150_000_000,
            ));

            // Layer norms (~10MB)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.input_layernorm.weight"),
                5_000_000,
            ));
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                5_000_000,
            ));
        }

        // Final norm
        tensors.push(TensorInfo::from_name("model.norm.weight", 5_000_000));

        ModelProfile::new(tensors)
    }

    #[test]
    fn test_14b_fits_with_mixed_precision() {
        let profile = make_14b_profile();
        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("should plan successfully");

        // With mixed precision on 22GB VRAM, 14B model should fit entirely
        // (29GB BF16 -> ~20GB with INT8 for MLPs)
        println!("VRAM usage: {:.2} GB", plan.vram_usage as f64 / 1e9);
        println!("RAM usage: {:.2} GB", plan.ram_usage as f64 / 1e9);
        println!("Swap count: {}", plan.swap_count);
        println!("Quality score: {:.3}", plan.quality_score);

        // Key assertion: should minimize or eliminate swapping
        assert!(
            plan.swap_count < 50,
            "14B model should mostly fit in VRAM with mixed precision, got {} swaps",
            plan.swap_count
        );
    }

    #[test]
    fn test_embeddings_always_in_vram() {
        let profile = make_14b_profile();
        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("should plan successfully");

        let embed = plan.allocations.get("model.embed_tokens.weight").unwrap();
        let lm_head = plan.allocations.get("lm_head.weight").unwrap();

        assert_eq!(embed.tier, MemoryTier::Vram, "embeddings should be in VRAM");
        assert_eq!(lm_head.tier, MemoryTier::Vram, "lm_head should be in VRAM");
    }

    #[test]
    fn test_vram_budget_respected() {
        let profile = make_14b_profile();
        let vram_gb = 22.0;
        let config = AdaptiveTieringConfig::with_budgets(vram_gb, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("should plan successfully");

        let vram_budget_bytes = (vram_gb * 1e9) as u64;
        assert!(
            plan.vram_usage <= vram_budget_bytes,
            "VRAM usage {} exceeds budget {}",
            plan.vram_usage,
            vram_budget_bytes
        );
    }

    #[test]
    fn test_edge_layers_prioritized() {
        let profile = make_14b_profile();
        // Very limited VRAM to force prioritization
        let config = AdaptiveTieringConfig::with_budgets(5.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("should plan successfully");

        // Layer 0 attention should be in VRAM before middle layers
        let layer_0_q = plan
            .allocations
            .get("model.layers.0.self_attn.q_proj.weight");
        let layer_24_q = plan
            .allocations
            .get("model.layers.24.self_attn.q_proj.weight");

        if let (Some(l0), Some(l24)) = (layer_0_q, layer_24_q) {
            // Layer 0 should be in higher-priority tier or same tier with higher priority
            assert!(
                l0.tier.priority() >= l24.tier.priority(),
                "layer 0 should be in equal or better tier than middle layer"
            );
        }
    }

    #[test]
    fn test_replan_for_vram_pressure() {
        let profile = make_14b_profile();
        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let initial_plan = planner.plan(&profile).expect("should plan successfully");
        let initial_vram = initial_plan.vram_usage;

        // Simulate KV cache growth reducing available VRAM
        let new_budget = initial_vram / 2;
        let new_plan = planner.replan_for_vram_pressure(&initial_plan, new_budget);

        assert!(
            new_plan.vram_usage <= new_budget,
            "replanned VRAM {} should be within new budget {}",
            new_plan.vram_usage,
            new_budget
        );
        assert!(
            new_plan.swap_count > initial_plan.swap_count,
            "should have more swaps after VRAM pressure"
        );
    }

    #[test]
    fn test_quality_score_reasonable() {
        let profile = make_14b_profile();
        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("should plan successfully");

        // Quality score should be above 0.9 for a 14B model on 22GB VRAM
        assert!(
            plan.quality_score >= 0.90,
            "quality score {} should be >= 0.90 for 14B on 22GB",
            plan.quality_score
        );
    }

    #[test]
    fn test_empty_model_error() {
        let profile = ModelProfile::new(vec![]);
        let config = AdaptiveTieringConfig::default();
        let planner = AllocationPlanner::new(config);

        let result = planner.plan(&profile);
        assert!(matches!(result, Err(PlannerError::EmptyModel)));
    }
}
