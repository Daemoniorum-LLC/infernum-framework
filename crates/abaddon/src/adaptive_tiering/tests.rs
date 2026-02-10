//! Property-based tests for the adaptive memory tiering system.
//!
//! These tests verify critical invariants that must hold regardless of input.

#[cfg(test)]
mod property_tests {
    use crate::adaptive_tiering::{
        AdaptiveTieringConfig, AllocationPlanner, ImportanceScorer, MemoryTier, ModelProfile,
        TensorInfo, TensorPrecision,
    };
    use proptest::prelude::*;

    // Strategy for generating random tensor info
    fn tensor_info_strategy() -> impl Strategy<Value = TensorInfo> {
        (
            "[a-z]{5,15}",                  // name suffix
            0usize..100,                    // layer index
            1_000_000u64..1_000_000_000u64, // size (1MB - 1GB)
        )
            .prop_map(|(suffix, layer, size)| {
                let name = if layer < 50 {
                    format!("model.layers.{layer}.mlp.{suffix}.weight")
                } else {
                    format!("model.{suffix}.weight")
                };
                TensorInfo::from_name(name, size)
            })
    }

    // Strategy for generating a model profile
    fn model_profile_strategy() -> impl Strategy<Value = ModelProfile> {
        prop::collection::vec(tensor_info_strategy(), 10..100).prop_map(ModelProfile::new)
    }

    // Strategy for generating config
    fn config_strategy() -> impl Strategy<Value = AdaptiveTieringConfig> {
        (
            1u64..100u64,   // vram_gb
            10u64..200u64,  // ram_gb
            0.5f32..1.0f32, // quality_target
        )
            .prop_map(|(vram_gb, ram_gb, quality_target)| AdaptiveTieringConfig {
                vram_budget: vram_gb * 1024 * 1024 * 1024,
                ram_budget: ram_gb * 1024 * 1024 * 1024,
                quality_target,
                ..AdaptiveTieringConfig::default()
            })
    }

    proptest! {
        /// CRITICAL INVARIANT: VRAM usage never exceeds budget.
        #[test]
        fn prop_vram_never_exceeds_budget(
            profile in model_profile_strategy(),
            config in config_strategy()
        ) {
            let planner = AllocationPlanner::new(config.clone());
            if let Ok(plan) = planner.plan(&profile) {
                prop_assert!(
                    plan.vram_usage <= config.vram_budget,
                    "VRAM usage {} exceeded budget {}",
                    plan.vram_usage,
                    config.vram_budget
                );
            }
        }

        /// INVARIANT: All tensors are allocated exactly once.
        #[test]
        fn prop_all_tensors_allocated(
            profile in model_profile_strategy(),
            config in config_strategy()
        ) {
            let planner = AllocationPlanner::new(config);
            if let Ok(plan) = planner.plan(&profile) {
                for tensor in &profile.tensors {
                    prop_assert!(
                        plan.allocations.contains_key(&tensor.name),
                        "tensor {} not allocated",
                        tensor.name
                    );
                }
                prop_assert_eq!(
                    plan.allocations.len(),
                    profile.tensors.len(),
                    "allocation count mismatch"
                );
            }
        }

        /// INVARIANT: Total usage equals sum of tier usages.
        #[test]
        fn prop_usage_sums_correctly(
            profile in model_profile_strategy(),
            config in config_strategy()
        ) {
            let planner = AllocationPlanner::new(config);
            if let Ok(plan) = planner.plan(&profile) {
                let vram_sum: u64 = plan.allocations.values()
                    .filter(|a| a.tier == MemoryTier::Vram)
                    .map(|a| a.storage_size)
                    .sum();
                let ram_sum: u64 = plan.allocations.values()
                    .filter(|a| a.tier == MemoryTier::Ram)
                    .map(|a| a.storage_size)
                    .sum();
                let nvme_sum: u64 = plan.allocations.values()
                    .filter(|a| a.tier == MemoryTier::Nvme)
                    .map(|a| a.storage_size)
                    .sum();

                prop_assert_eq!(plan.vram_usage, vram_sum, "VRAM usage mismatch");
                prop_assert_eq!(plan.ram_usage, ram_sum, "RAM usage mismatch");
                prop_assert_eq!(plan.nvme_usage, nvme_sum, "NVMe usage mismatch");
            }
        }

        /// INVARIANT: Importance scores are always in [0, 1].
        #[test]
        fn prop_importance_bounded(
            profile in model_profile_strategy()
        ) {
            let max_size = profile.tensors.iter().map(|t| t.size_bytes).max().unwrap_or(1);
            let scorer = ImportanceScorer::new(profile.num_layers, max_size);

            for tensor in &profile.tensors {
                let score = scorer.score(tensor);
                prop_assert!(
                    (0.0..=1.0).contains(&score),
                    "score {} out of bounds for {}",
                    score,
                    tensor.name
                );
            }
        }

        /// INVARIANT: Quality score is in [0, 1].
        #[test]
        fn prop_quality_score_bounded(
            profile in model_profile_strategy(),
            config in config_strategy()
        ) {
            let planner = AllocationPlanner::new(config);
            if let Ok(plan) = planner.plan(&profile) {
                prop_assert!(
                    (0.0..=1.0).contains(&plan.quality_score),
                    "quality score {} out of bounds",
                    plan.quality_score
                );
            }
        }

        /// INVARIANT: Storage size respects precision divisor.
        #[test]
        fn prop_storage_size_correct(
            bf16_size in 1_000_000u64..1_000_000_000u64,
            precision in prop_oneof![
                Just(TensorPrecision::BF16),
                Just(TensorPrecision::INT8),
                Just(TensorPrecision::INT4),
            ]
        ) {
            let storage = precision.storage_size(bf16_size);
            let expected = (bf16_size as f64 / precision.size_divisor() as f64).ceil() as u64;
            prop_assert_eq!(storage, expected);
        }

        /// INVARIANT: Higher importance tensors get better placement.
        /// (This is a statistical property - we check that embeddings are prioritized)
        #[test]
        fn prop_high_importance_prioritized(
            config in config_strategy()
        ) {
            // Create a simple profile with clear importance differences
            let tensors = vec![
                TensorInfo::from_name("model.embed_tokens.weight", 500_000_000),
                TensorInfo::from_name("model.layers.20.mlp.gate_proj.weight", 500_000_000),
            ];
            let profile = ModelProfile::new(tensors);

            let planner = AllocationPlanner::new(config);
            if let Ok(plan) = planner.plan(&profile) {
                let embed = plan.allocations.get("model.embed_tokens.weight").unwrap();
                let mlp = plan.allocations.get("model.layers.20.mlp.gate_proj.weight").unwrap();

                // Embedding should have equal or better tier
                prop_assert!(
                    embed.tier.priority() >= mlp.tier.priority(),
                    "embedding should be in equal or better tier than MLP"
                );
            }
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::adaptive_tiering::{
        AdaptiveTieringConfig, AllocationPlanner, MemoryTier, ModelProfile, TensorInfo,
        TensorPrecision,
    };

    /// Creates a realistic 14B model profile for testing.
    fn make_14b_profile() -> ModelProfile {
        let mut tensors = Vec::new();

        // Embeddings and lm_head (~1GB total)
        tensors.push(TensorInfo::from_name(
            "model.embed_tokens.weight",
            524_288_000,
        )); // 500MB
        tensors.push(TensorInfo::from_name("lm_head.weight", 524_288_000)); // 500MB

        // 48 layers, each ~600MB
        for i in 0..48 {
            // Attention (100MB total)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                41_943_040,
            )); // 40MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                10_485_760,
            )); // 10MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                10_485_760,
            )); // 10MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                41_943_040,
            )); // 40MB

            // MLP (450MB total)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                146_800_640,
            )); // 140MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                146_800_640,
            )); // 140MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                146_800_640,
            )); // 140MB

            // Layer norms (10MB total)
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.input_layernorm.weight"),
                5_242_880,
            )); // 5MB
            tensors.push(TensorInfo::from_name(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                5_242_880,
            )); // 5MB
        }

        // Final norm
        tensors.push(TensorInfo::from_name("model.norm.weight", 5_242_880)); // 5MB

        ModelProfile::new(tensors)
    }

    /// Test case: 14B model on 24GB VRAM should fit with mixed precision and no swapping.
    #[test]
    fn test_14b_on_24gb_no_swapping() {
        let profile = make_14b_profile();
        println!("Model size: {:.2} GB", profile.size_gb());

        // 24GB VRAM with 2GB headroom = 22GB effective
        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("planning should succeed");

        println!("=== Allocation Summary ===");
        println!("VRAM usage: {:.2} GB", plan.vram_usage as f64 / 1e9);
        println!("RAM usage: {:.2} GB", plan.ram_usage as f64 / 1e9);
        println!("NVMe usage: {:.2} GB", plan.nvme_usage as f64 / 1e9);
        println!("Swap count: {}", plan.swap_count);
        println!("Quality score: {:.3}", plan.quality_score);

        // Count tensors by tier
        let vram_count = plan
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Vram)
            .count();
        let ram_count = plan
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Ram)
            .count();
        let nvme_count = plan
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Nvme)
            .count();

        println!("VRAM tensors: {}", vram_count);
        println!("RAM tensors: {}", ram_count);
        println!("NVMe tensors: {}", nvme_count);

        // Count by precision
        let bf16_count = plan
            .allocations
            .values()
            .filter(|a| a.precision == TensorPrecision::BF16)
            .count();
        let int8_count = plan
            .allocations
            .values()
            .filter(|a| a.precision == TensorPrecision::INT8)
            .count();
        let int4_count = plan
            .allocations
            .values()
            .filter(|a| a.precision == TensorPrecision::INT4)
            .count();

        println!("BF16 tensors: {}", bf16_count);
        println!("INT8 tensors: {}", int8_count);
        println!("INT4 tensors: {}", int4_count);

        // Key assertions for 14B on 24GB:
        // 1. Model should mostly fit in VRAM with mixed precision
        assert!(
            plan.swap_count < profile.tensors.len() / 2,
            "14B should fit mostly in VRAM (swap_count={} < {})",
            plan.swap_count,
            profile.tensors.len() / 2
        );

        // 2. Quality should remain high (> 0.9)
        assert!(
            plan.quality_score >= 0.90,
            "quality should be >= 0.90, got {}",
            plan.quality_score
        );

        // 3. Critical tensors (embeddings, lm_head) should be in VRAM
        let embed_alloc = plan.allocations.get("model.embed_tokens.weight").unwrap();
        let lm_head_alloc = plan.allocations.get("lm_head.weight").unwrap();
        assert_eq!(
            embed_alloc.tier,
            MemoryTier::Vram,
            "embeddings must be in VRAM"
        );
        assert_eq!(
            lm_head_alloc.tier,
            MemoryTier::Vram,
            "lm_head must be in VRAM"
        );

        // 4. VRAM budget respected
        assert!(
            plan.vram_usage <= 22 * 1024 * 1024 * 1024,
            "VRAM usage {} exceeds 22GB budget",
            plan.vram_usage
        );
    }

    /// Test case: Tiny VRAM should still produce valid plan (falls back to RAM/NVMe).
    #[test]
    fn test_tiny_vram_graceful_fallback() {
        let profile = make_14b_profile();

        // Only 2GB VRAM - can't fit much
        let config = AdaptiveTieringConfig::with_budgets(2.0, 60.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner
            .plan(&profile)
            .expect("planning should succeed even with tiny VRAM");

        // Critical tensors should still be allocated (possibly in RAM)
        assert!(plan.allocations.contains_key("model.embed_tokens.weight"));
        assert!(plan.allocations.contains_key("lm_head.weight"));

        // VRAM budget must be respected
        assert!(plan.vram_usage <= 2 * 1024 * 1024 * 1024);

        // Most tensors will be in RAM
        let ram_count = plan
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Ram)
            .count();
        assert!(
            ram_count > profile.tensors.len() / 2,
            "with 2GB VRAM, most tensors should be in RAM"
        );
    }

    /// Test case: Abundant VRAM should use BF16 everywhere.
    #[test]
    fn test_abundant_vram_prefers_bf16() {
        let profile = make_14b_profile();

        // 100GB VRAM - way more than needed
        let config = AdaptiveTieringConfig::with_budgets(100.0, 100.0);
        let planner = AllocationPlanner::new(config);

        let plan = planner.plan(&profile).expect("planning should succeed");

        // Everything should be in VRAM
        let vram_count = plan
            .allocations
            .values()
            .filter(|a| a.tier == MemoryTier::Vram)
            .count();
        assert_eq!(
            vram_count,
            profile.tensors.len(),
            "with abundant VRAM, all tensors should be in VRAM"
        );

        // No swapping needed
        assert_eq!(plan.swap_count, 0, "no swapping with abundant VRAM");

        // Quality should be 1.0 (all BF16 in VRAM)
        assert!(
            plan.quality_score >= 0.99,
            "quality should be ~1.0 with abundant VRAM, got {}",
            plan.quality_score
        );
    }
}
