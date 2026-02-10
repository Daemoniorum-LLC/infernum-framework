# Adaptive Memory Tiering TDD Roadmap

**Version:** 0.1.0
**Status:** Test Specification
**Date:** 2026-02-05
**Spec Reference:** ADAPTIVE-MEMORY-TIERING-SPEC.md v0.1.0

---

## Philosophy

Tests are crystallized understanding, not coverage theater.

Each test in this roadmap exists because it captures something we *must* know is true. If a test doesn't teach us something about the system's behavior, it doesn't belong here.

We test at **trust boundaries** — the edges where assumptions meet reality:
- Memory budgets are hard limits (OOM is catastrophic)
- Tensor importance determines inference quality
- Precision choices trade quality for capacity
- Runtime swaps must respect allocation decisions

We prefer **property tests** over example tests where the property is the point. "VRAM allocation never exceeds budget" is a property. "14B model uses 23.5GB VRAM" is an example.

---

## 1. Importance Scoring

**Trust Boundary:** Importance scores determine which tensors stay in fast memory. Incorrect scoring degrades either quality (important tensors quantized) or speed (unimportant tensors in VRAM).

### 1.1 Ordering Properties

```rust
// Property: Embeddings and lm_head always have maximum importance
#[proptest]
fn test_token_tensors_have_max_importance(
    model: ArbitraryModelProfile,
) {
    let scorer = ImportanceScorer::new();

    let embed_score = scorer.score("embed_tokens", &model);
    let lm_head_score = scorer.score("lm_head", &model);

    // Must be maximum (these are used every single token)
    prop_assert_eq!(embed_score, 1.0);
    prop_assert_eq!(lm_head_score, 1.0);
}

// Property: Edge layers have higher importance than middle layers
#[proptest]
fn test_edge_layers_more_important_than_middle(
    num_layers in 12usize..128,
    tensor_type: TensorType,
) {
    let scorer = ImportanceScorer::new();
    let middle = num_layers / 2;

    let edge_score = scorer.score_layer(0, num_layers, tensor_type);
    let middle_score = scorer.score_layer(middle, num_layers, tensor_type);
    let last_score = scorer.score_layer(num_layers - 1, num_layers, tensor_type);

    prop_assert!(edge_score > middle_score);
    prop_assert!(last_score > middle_score);
}

// Property: Attention weights more important than MLP weights (same layer)
#[proptest]
fn test_attention_more_important_than_mlp(
    layer_idx in 0usize..128,
    num_layers in 12usize..128,
) {
    prop_assume!(layer_idx < num_layers);
    let scorer = ImportanceScorer::new();

    let attn_score = scorer.score(&format!("layers.{}.self_attn.q_proj", layer_idx), num_layers);
    let mlp_score = scorer.score(&format!("layers.{}.mlp.gate_proj", layer_idx), num_layers);

    prop_assert!(attn_score > mlp_score);
}

// Property: LayerNorm more important than MLP (affects all activations)
#[proptest]
fn test_layernorm_more_important_than_mlp(
    layer_idx in 0usize..128,
    num_layers in 12usize..128,
) {
    prop_assume!(layer_idx < num_layers);
    let scorer = ImportanceScorer::new();

    let norm_score = scorer.score(&format!("layers.{}.input_layernorm", layer_idx), num_layers);
    let mlp_score = scorer.score(&format!("layers.{}.mlp.down_proj", layer_idx), num_layers);

    prop_assert!(norm_score > mlp_score);
}
```

### 1.2 Score Bounds

```rust
// Property: All importance scores are in [0, 1]
#[proptest]
fn test_importance_scores_bounded(
    tensor_name: ArbitraryTensorName,
    model: ArbitraryModelProfile,
) {
    let scorer = ImportanceScorer::new();
    let score = scorer.score(&tensor_name, &model);

    prop_assert!(score >= 0.0);
    prop_assert!(score <= 1.0);
}

// Property: No tensor has zero importance (all tensors matter somewhat)
#[proptest]
fn test_no_zero_importance(
    tensor_name: ArbitraryValidTensorName,  // Excludes malformed names
    model: ArbitraryModelProfile,
) {
    let scorer = ImportanceScorer::new();
    let score = scorer.score(&tensor_name, &model);

    prop_assert!(score > 0.0, "Tensor {} had zero importance", tensor_name);
}
```

### 1.3 QualityCurve Integration

```rust
#[test]
fn test_quality_curve_increases_importance() {
    let scorer = ImportanceScorer::new();

    // Tensor with steep quality curve (sensitive to precision)
    let steep_curve = QualityCurve::new(0.9, 0.1);  // 90% quality at 25% fragments
    let flat_curve = QualityCurve::new(0.5, 0.3);   // 50% quality at 25% fragments

    let steep_score = scorer.score_with_curve("layers.10.self_attn.q_proj", Some(&steep_curve));
    let flat_score = scorer.score_with_curve("layers.10.self_attn.q_proj", Some(&flat_curve));

    // Steep curve = more sensitive = higher importance
    assert!(steep_score > flat_score);
}

// Property: QualityCurve can only increase importance, never decrease
#[proptest]
fn test_quality_curve_never_decreases_importance(
    tensor_name: ArbitraryTensorName,
    curve: ArbitraryQualityCurve,
) {
    let scorer = ImportanceScorer::new();

    let base_score = scorer.score_without_curve(&tensor_name);
    let curve_score = scorer.score_with_curve(&tensor_name, Some(&curve));

    prop_assert!(curve_score >= base_score);
}
```

---

## 2. Allocation Planning

**Trust Boundary:** Allocation decisions determine memory layout. Exceeding budgets causes OOM. Under-utilizing budgets wastes performance.

### 2.1 Budget Constraints (CRITICAL)

```rust
// Property: VRAM allocation NEVER exceeds budget
#[proptest]
fn test_vram_allocation_respects_budget(
    model: ArbitraryModelProfile,
    vram_budget: u64,
    ram_budget: u64,
) {
    let planner = AllocationPlanner::new(AdaptiveTieringConfig {
        vram_budget,
        ram_budget,
        ..Default::default()
    });

    let plan = planner.plan(&model).unwrap();

    prop_assert!(
        plan.vram_usage <= vram_budget,
        "VRAM usage {} exceeds budget {}",
        plan.vram_usage,
        vram_budget
    );
}

// Property: RAM allocation NEVER exceeds budget
#[proptest]
fn test_ram_allocation_respects_budget(
    model: ArbitraryModelProfile,
    vram_budget: u64,
    ram_budget: u64,
) {
    let planner = AllocationPlanner::new(AdaptiveTieringConfig {
        vram_budget,
        ram_budget,
        ..Default::default()
    });

    let plan = planner.plan(&model).unwrap();

    prop_assert!(
        plan.ram_usage <= ram_budget,
        "RAM usage {} exceeds budget {}",
        plan.ram_usage,
        ram_budget
    );
}

// Property: Every tensor is allocated exactly once
#[proptest]
fn test_all_tensors_allocated(
    model: ArbitraryModelProfile,
    config: ArbitraryConfig,
) {
    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();

    for tensor in &model.tensors {
        prop_assert!(
            plan.allocations.contains_key(&tensor.name),
            "Tensor {} not allocated",
            tensor.name
        );
    }

    // No extra allocations
    prop_assert_eq!(plan.allocations.len(), model.tensors.len());
}
```

### 2.2 Priority Ordering

```rust
// Property: Higher importance tensors get better placement
#[proptest]
fn test_higher_importance_gets_better_tier(
    model: ArbitraryModelProfile,
    config: ArbitraryConfig,
) {
    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();

    let scorer = ImportanceScorer::new();

    for (name_a, alloc_a) in &plan.allocations {
        for (name_b, alloc_b) in &plan.allocations {
            let importance_a = scorer.score(name_a, &model);
            let importance_b = scorer.score(name_b, &model);

            // If A is more important AND in a worse tier, something is wrong
            // (unless budget forced it)
            if importance_a > importance_b + 0.1 {  // Margin for ties
                let tier_order = |t: &MemoryTier| match t {
                    MemoryTier::Vram => 0,
                    MemoryTier::Ram => 1,
                    MemoryTier::Nvme => 2,
                };

                // A should be in same or better tier than B
                prop_assert!(
                    tier_order(&alloc_a.tier) <= tier_order(&alloc_b.tier),
                    "{} (importance {}) in worse tier than {} (importance {})",
                    name_a, importance_a, name_b, importance_b
                );
            }
        }
    }
}

// Property: VRAM tensors have higher average importance than RAM tensors
#[proptest]
fn test_vram_tensors_more_important_than_ram(
    model: ArbitraryModelProfile,
    config: ArbitraryConfig,
) {
    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();
    let scorer = ImportanceScorer::new();

    let vram_importance: f32 = plan.allocations.iter()
        .filter(|(_, a)| a.tier == MemoryTier::Vram)
        .map(|(n, _)| scorer.score(n, &model))
        .sum::<f32>();

    let ram_importance: f32 = plan.allocations.iter()
        .filter(|(_, a)| a.tier == MemoryTier::Ram)
        .map(|(n, _)| scorer.score(n, &model))
        .sum::<f32>();

    let vram_count = plan.allocations.values().filter(|a| a.tier == MemoryTier::Vram).count();
    let ram_count = plan.allocations.values().filter(|a| a.tier == MemoryTier::Ram).count();

    if vram_count > 0 && ram_count > 0 {
        let vram_avg = vram_importance / vram_count as f32;
        let ram_avg = ram_importance / ram_count as f32;
        prop_assert!(vram_avg >= ram_avg);
    }
}
```

### 2.3 Precision Selection

```rust
// Property: Higher quality target = less quantization
#[proptest]
fn test_higher_quality_target_less_quantization(
    model: ArbitraryModelProfile,
    vram_budget: u64,
) {
    let high_quality_config = AdaptiveTieringConfig {
        vram_budget,
        quality_target: 0.99,
        ..Default::default()
    };

    let low_quality_config = AdaptiveTieringConfig {
        vram_budget,
        quality_target: 0.80,
        ..Default::default()
    };

    let planner_high = AllocationPlanner::new(high_quality_config);
    let planner_low = AllocationPlanner::new(low_quality_config);

    let plan_high = planner_high.plan(&model).unwrap();
    let plan_low = planner_low.plan(&model).unwrap();

    let count_quantized = |plan: &AllocationPlan| {
        plan.allocations.values()
            .filter(|a| a.precision != TensorPrecision::BF16)
            .count()
    };

    prop_assert!(count_quantized(&plan_high) <= count_quantized(&plan_low));
}

// Property: Quantization only applied when VRAM budget requires it
#[test]
fn test_no_unnecessary_quantization() {
    // Model that fits in VRAM without quantization
    let small_model = ModelProfile::from_params(1_000_000_000);  // 1B = 2GB BF16
    let large_vram = AdaptiveTieringConfig {
        vram_budget: 10 * GB,
        ..Default::default()
    };

    let planner = AllocationPlanner::new(large_vram);
    let plan = planner.plan(&small_model).unwrap();

    // Everything should be BF16 since it fits
    for (name, alloc) in &plan.allocations {
        assert_eq!(
            alloc.precision, TensorPrecision::BF16,
            "Tensor {} unnecessarily quantized",
            name
        );
    }
}

// Property: Important tensors quantized last
#[proptest]
fn test_important_tensors_quantized_last(
    model: ArbitraryModelProfile,
    vram_budget: u64,  // Constrained to force some quantization
) {
    prop_assume!(vram_budget < model.total_size_bf16() && vram_budget > model.total_size_bf16() / 4);

    let planner = AllocationPlanner::new(AdaptiveTieringConfig {
        vram_budget,
        ..Default::default()
    });
    let plan = planner.plan(&model).unwrap();
    let scorer = ImportanceScorer::new();

    let quantized: Vec<_> = plan.allocations.iter()
        .filter(|(_, a)| a.precision != TensorPrecision::BF16 && a.tier == MemoryTier::Vram)
        .collect();

    let bf16: Vec<_> = plan.allocations.iter()
        .filter(|(_, a)| a.precision == TensorPrecision::BF16 && a.tier == MemoryTier::Vram)
        .collect();

    if !quantized.is_empty() && !bf16.is_empty() {
        let max_quantized_importance = quantized.iter()
            .map(|(n, _)| scorer.score(n, &model))
            .fold(0.0f32, |a, b| a.max(b));

        let min_bf16_importance = bf16.iter()
            .map(|(n, _)| scorer.score(n, &model))
            .fold(1.0f32, |a, b| a.min(b));

        prop_assert!(
            max_quantized_importance <= min_bf16_importance + 0.1,
            "Quantized high-importance tensor while keeping low-importance in BF16"
        );
    }
}
```

### 2.4 Specific Allocation Scenarios

```rust
#[test]
fn test_14b_model_fits_with_mixed_precision() {
    // Qwen2.5-14B: 14.7B params = 29.4GB BF16
    let model = ModelProfile::qwen2_14b();
    let config = AdaptiveTieringConfig {
        vram_budget: 24 * GB,
        ram_budget: 64 * GB,
        ..Default::default()
    };

    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();

    // Should fit entirely in VRAM with mixed precision
    assert_eq!(plan.swap_count, 0, "14B should fit without swapping");
    assert!(plan.vram_usage <= 24 * GB);
    assert!(plan.ram_usage == 0, "Everything should be in VRAM");

    // Verify embeddings/lm_head are BF16 in VRAM
    assert_eq!(plan.allocations["embed_tokens"].tier, MemoryTier::Vram);
    assert_eq!(plan.allocations["embed_tokens"].precision, TensorPrecision::BF16);
    assert_eq!(plan.allocations["lm_head"].tier, MemoryTier::Vram);
    assert_eq!(plan.allocations["lm_head"].precision, TensorPrecision::BF16);
}

#[test]
fn test_70b_model_uses_all_tiers() {
    // Llama-70B: ~140GB BF16
    let model = ModelProfile::llama_70b();
    let config = AdaptiveTieringConfig {
        vram_budget: 24 * GB,
        ram_budget: 64 * GB,
        ..Default::default()
    };

    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();

    // Should use all tiers
    let vram_count = plan.allocations.values().filter(|a| a.tier == MemoryTier::Vram).count();
    let ram_count = plan.allocations.values().filter(|a| a.tier == MemoryTier::Ram).count();
    let nvme_count = plan.allocations.values().filter(|a| a.tier == MemoryTier::Nvme).count();

    assert!(vram_count > 0, "Should have tensors in VRAM");
    assert!(ram_count > 0, "Should have tensors in RAM");
    // NVMe may or may not be used depending on exact allocation

    // Embeddings must be in VRAM (used every token)
    assert_eq!(plan.allocations["embed_tokens"].tier, MemoryTier::Vram);
}

#[test]
fn test_tiny_model_all_bf16_vram() {
    // SmolLM-135M: ~270MB BF16
    let model = ModelProfile::smollm_135m();
    let config = AdaptiveTieringConfig::default();  // Any reasonable VRAM

    let planner = AllocationPlanner::new(config);
    let plan = planner.plan(&model).unwrap();

    // Everything should be BF16 in VRAM
    for (name, alloc) in &plan.allocations {
        assert_eq!(alloc.tier, MemoryTier::Vram, "{} not in VRAM", name);
        assert_eq!(alloc.precision, TensorPrecision::BF16, "{} not BF16", name);
    }
    assert_eq!(plan.swap_count, 0);
}
```

---

## 3. Precision Conversion

**Trust Boundary:** Quantization affects model quality. Incorrect conversion corrupts weights.

### 3.1 Conversion Correctness

```rust
// Property: BF16 → INT8 → BF16 roundtrip has bounded error
#[proptest]
fn test_int8_quantization_bounded_error(
    tensor: ArbitraryBF16Tensor,
) {
    let quantized = quantize_int8(&tensor);
    let dequantized = dequantize_int8(&quantized);

    let max_error = tensor.iter().zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    // INT8 error bound: ~0.4% of range per element
    let tensor_range = tensor.max() - tensor.min();
    let expected_max_error = tensor_range / 256.0 * 2.0;  // 2x for safety margin

    prop_assert!(
        max_error <= expected_max_error,
        "INT8 quantization error {} exceeds expected {}",
        max_error,
        expected_max_error
    );
}

// Property: BF16 → INT4 → BF16 roundtrip has bounded (larger) error
#[proptest]
fn test_int4_quantization_bounded_error(
    tensor: ArbitraryBF16Tensor,
) {
    let quantized = quantize_int4(&tensor);
    let dequantized = dequantize_int4(&quantized);

    let mse: f32 = tensor.iter().zip(dequantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / tensor.len() as f32;

    // INT4 allows ~6% RMSE degradation
    let tensor_variance: f32 = tensor.iter()
        .map(|x| (x - tensor.mean()).powi(2))
        .sum::<f32>() / tensor.len() as f32;

    prop_assert!(mse <= tensor_variance * 0.06);
}

// Property: Size reduction matches precision
#[proptest]
fn test_quantized_size_correct(
    tensor_size: u64,
    precision: TensorPrecision,
) {
    let expected_size = match precision {
        TensorPrecision::BF16 => tensor_size,
        TensorPrecision::FP8 | TensorPrecision::INT8 => tensor_size / 2,
        TensorPrecision::INT4 => tensor_size / 4,
    };

    let actual_size = compute_quantized_size(tensor_size, precision);
    prop_assert_eq!(actual_size, expected_size);
}
```

### 3.2 GPU Dequantization

```rust
#[test]
fn test_gpu_dequant_matches_cpu() {
    let tensor = create_test_tensor(1024, 1024);
    let quantized = quantize_int8(&tensor);

    let cpu_result = dequantize_int8_cpu(&quantized);
    let gpu_result = dequantize_int8_gpu(&quantized);

    // GPU and CPU should produce identical results
    assert_tensors_equal(&cpu_result, &gpu_result, 1e-6);
}

// Property: GPU dequantization is faster than CPU for large tensors
#[proptest]
fn test_gpu_dequant_faster_for_large_tensors(
    rows in 256usize..4096,
    cols in 256usize..4096,
) {
    let tensor = create_test_tensor(rows, cols);
    let quantized = quantize_int8(&tensor);

    let cpu_time = benchmark(|| dequantize_int8_cpu(&quantized));
    let gpu_time = benchmark(|| dequantize_int8_gpu(&quantized));

    // GPU should be faster for tensors > 64KB
    if rows * cols > 65536 {
        prop_assert!(gpu_time < cpu_time);
    }
}
```

---

## 4. Adaptive Loader

**Trust Boundary:** The loader is the runtime enforcement of allocation decisions. Violations here cause either OOM (budget exceeded) or slow inference (wrong tier).

### 4.1 Tier Enforcement

```rust
// Property: Tensors are loaded from their assigned tier
#[proptest]
fn test_tensors_loaded_from_assigned_tier(
    plan: ArbitraryAllocationPlan,
    tensor_name: String,
) {
    prop_assume!(plan.allocations.contains_key(&tensor_name));

    let loader = AdaptiveLoader::new(plan.clone());
    let access_log = AccessLog::new();
    loader.set_access_log(&access_log);

    let _ = loader.get(&tensor_name, &Device::Cuda(0), DType::BF16);

    let expected_tier = plan.allocations[&tensor_name].tier;
    let actual_source = access_log.last_source(&tensor_name);

    prop_assert_eq!(actual_source, expected_tier);
}

// Property: VRAM tensors load without disk/RAM access
#[proptest]
fn test_vram_tensors_no_io(
    plan: ArbitraryAllocationPlan,
) {
    let vram_tensors: Vec<_> = plan.allocations.iter()
        .filter(|(_, a)| a.tier == MemoryTier::Vram)
        .map(|(n, _)| n.clone())
        .collect();

    let loader = AdaptiveLoader::new(plan);
    let io_counter = IoCounter::new();
    loader.set_io_counter(&io_counter);

    for name in &vram_tensors {
        let _ = loader.get(name, &Device::Cuda(0), DType::BF16);
    }

    prop_assert_eq!(io_counter.disk_reads(), 0);
    prop_assert_eq!(io_counter.ram_to_gpu_transfers(), 0);
}
```

### 4.2 Precision Handling

```rust
// Property: Returned tensor has correct dtype regardless of storage precision
#[proptest]
fn test_returned_dtype_correct(
    plan: ArbitraryAllocationPlan,
    tensor_name: String,
    requested_dtype: DType,
) {
    prop_assume!(plan.allocations.contains_key(&tensor_name));

    let loader = AdaptiveLoader::new(plan);
    let tensor = loader.get(&tensor_name, &Device::Cuda(0), requested_dtype).unwrap();

    prop_assert_eq!(tensor.dtype(), requested_dtype);
}

// Property: INT8 tensors are dequantized before return
#[test]
fn test_int8_tensors_dequantized() {
    let mut plan = AllocationPlan::new();
    plan.allocations.insert("test_tensor".to_string(), TensorAllocation {
        tier: MemoryTier::Vram,
        precision: TensorPrecision::INT8,
        priority: 0.5,
        prefetch: false,
    });

    let loader = AdaptiveLoader::new(plan);
    let tensor = loader.get("test_tensor", &Device::Cuda(0), DType::BF16).unwrap();

    // Should be BF16, not raw INT8
    assert_eq!(tensor.dtype(), DType::BF16);

    // Values should be in normal weight range, not INT8 range
    let max_val = tensor.max().unwrap().to_scalar::<f32>().unwrap();
    assert!(max_val < 10.0, "Values seem to be raw INT8, not dequantized");
}
```

### 4.3 Cache Behavior

```rust
// Property: Repeated access doesn't reload from slow tier
#[proptest]
fn test_cache_prevents_reload(
    plan: ArbitraryAllocationPlan,
    tensor_name: String,
) {
    prop_assume!(plan.allocations.contains_key(&tensor_name));

    let loader = AdaptiveLoader::new(plan);
    let io_counter = IoCounter::new();
    loader.set_io_counter(&io_counter);

    // First access
    let _ = loader.get(&tensor_name, &Device::Cuda(0), DType::BF16);
    let first_io = io_counter.total_io();

    // Second access
    let _ = loader.get(&tensor_name, &Device::Cuda(0), DType::BF16);
    let second_io = io_counter.total_io();

    // No additional I/O for cached tensor
    prop_assert_eq!(first_io, second_io);
}

// Property: LRU eviction preserves high-priority tensors
#[test]
fn test_lru_preserves_high_priority() {
    let mut plan = AllocationPlan::new();

    // High priority tensor
    plan.allocations.insert("important".to_string(), TensorAllocation {
        tier: MemoryTier::Vram,
        precision: TensorPrecision::BF16,
        priority: 0.95,
        prefetch: false,
    });

    // Low priority tensors
    for i in 0..100 {
        plan.allocations.insert(format!("filler_{}", i), TensorAllocation {
            tier: MemoryTier::Vram,
            precision: TensorPrecision::BF16,
            priority: 0.1,
            prefetch: false,
        });
    }

    let loader = AdaptiveLoader::with_cache_limit(plan, 10);  // Only cache 10 tensors

    // Load important tensor
    let _ = loader.get("important", &Device::Cuda(0), DType::BF16);

    // Load many low-priority tensors to trigger eviction
    for i in 0..100 {
        let _ = loader.get(&format!("filler_{}", i), &Device::Cuda(0), DType::BF16);
    }

    // Important tensor should still be cached
    assert!(loader.is_cached("important"));
}
```

---

## 5. Runtime Adaptation

**Trust Boundary:** Dynamic reallocation can improve performance but must not violate budget constraints or cause inconsistency.

### 5.1 Adaptation Safety

```rust
// Property: Adaptation never exceeds memory budgets
#[proptest]
fn test_adaptation_respects_budgets(
    initial_plan: ArbitraryAllocationPlan,
    access_pattern: Vec<String>,
) {
    let config = initial_plan.config.clone();
    let mut loader = AdaptiveLoader::new(initial_plan);

    // Simulate access pattern
    for name in &access_pattern {
        if loader.contains(name) {
            let _ = loader.get(name, &Device::Cuda(0), DType::BF16);
        }
    }

    // Trigger adaptation
    loader.adapt();

    let stats = loader.memory_stats();

    prop_assert!(stats.vram_used <= config.vram_budget);
    prop_assert!(stats.ram_used <= config.ram_budget);
}

// Property: Adaptation is idempotent
#[proptest]
fn test_adaptation_idempotent(
    plan: ArbitraryAllocationPlan,
    access_pattern: Vec<String>,
) {
    let mut loader = AdaptiveLoader::new(plan);

    for name in &access_pattern {
        if loader.contains(name) {
            let _ = loader.get(name, &Device::Cuda(0), DType::BF16);
        }
    }

    loader.adapt();
    let state_after_first = loader.allocation_state();

    loader.adapt();
    let state_after_second = loader.allocation_state();

    // No access pattern change = no allocation change
    prop_assert_eq!(state_after_first, state_after_second);
}
```

### 5.2 Promotion/Demotion

```rust
// Property: Frequently accessed cold tensors get promoted
#[test]
fn test_hot_tensor_promotion() {
    let mut plan = AllocationPlan::new();

    // Start in RAM
    plan.allocations.insert("hot_tensor".to_string(), TensorAllocation {
        tier: MemoryTier::Ram,
        precision: TensorPrecision::BF16,
        priority: 0.3,
        prefetch: false,
    });

    let mut loader = AdaptiveLoader::new(plan);
    loader.enable_adaptation(true);

    // Access many times
    for _ in 0..100 {
        let _ = loader.get("hot_tensor", &Device::Cuda(0), DType::BF16);
    }

    loader.adapt();

    // Should be promoted to VRAM
    let new_tier = loader.current_tier("hot_tensor");
    assert_eq!(new_tier, MemoryTier::Vram);
}

// Property: Unused VRAM tensors get demoted under pressure
#[test]
fn test_cold_tensor_demotion() {
    let mut plan = AllocationPlan::new();

    // Cold tensor in VRAM
    plan.allocations.insert("cold_tensor".to_string(), TensorAllocation {
        tier: MemoryTier::Vram,
        precision: TensorPrecision::BF16,
        priority: 0.2,
        prefetch: false,
    });

    // Hot tensors that need space
    for i in 0..10 {
        plan.allocations.insert(format!("hot_{}", i), TensorAllocation {
            tier: MemoryTier::Vram,
            precision: TensorPrecision::BF16,
            priority: 0.8,
            prefetch: false,
        });
    }

    let mut loader = AdaptiveLoader::with_vram_limit(plan, 5);  // Only 5 fit
    loader.enable_adaptation(true);

    // Access only hot tensors
    for _ in 0..50 {
        for i in 0..10 {
            let _ = loader.get(&format!("hot_{}", i), &Device::Cuda(0), DType::BF16);
        }
    }

    loader.adapt();

    // Cold tensor should be demoted
    let cold_tier = loader.current_tier("cold_tensor");
    assert!(cold_tier != MemoryTier::Vram);
}
```

### 5.3 KV Cache Pressure

```rust
// Property: KV cache growth triggers demotion of low-priority VRAM tensors
#[test]
fn test_kv_cache_pressure_handling() {
    let plan = create_full_vram_plan();
    let mut loader = AdaptiveLoader::new(plan);

    let initial_vram = loader.memory_stats().vram_used;

    // Simulate KV cache growth
    loader.notify_kv_cache_size(5 * GB);

    let after_vram = loader.memory_stats().vram_used;

    // VRAM usage should decrease to make room
    assert!(after_vram < initial_vram);

    // Low priority tensors should be evicted first
    let evicted = loader.evicted_tensors();
    let high_priority_evicted = evicted.iter()
        .any(|t| loader.priority(t) > 0.8);

    assert!(!high_priority_evicted, "High priority tensor evicted for KV cache");
}
```

---

## 6. Prefetching

**Trust Boundary:** Prefetching should improve latency without wasting bandwidth or causing memory pressure.

### 6.1 Prefetch Correctness

```rust
// Property: Prefetched tensors are available without blocking
#[test]
fn test_prefetch_eliminates_blocking() {
    let plan = create_plan_with_ram_tensors();
    let loader = AdaptiveLoader::new(plan);

    // Start prefetch
    loader.prefetch("layer_5_tensors");

    // Wait for prefetch
    std::thread::sleep(Duration::from_millis(100));

    // Access should be instant (no I/O)
    let start = Instant::now();
    let _ = loader.get("layers.5.self_attn.q_proj", &Device::Cuda(0), DType::BF16);
    let elapsed = start.elapsed();

    // Should be much faster than cold load
    assert!(elapsed < Duration::from_millis(10));
}

// Property: Prefetch doesn't exceed memory budget
#[proptest]
fn test_prefetch_respects_budget(
    plan: ArbitraryAllocationPlan,
    prefetch_requests: Vec<String>,
) {
    let config = plan.config.clone();
    let loader = AdaptiveLoader::new(plan);

    for name in prefetch_requests {
        loader.prefetch(&name);
    }

    // Wait for prefetches
    std::thread::sleep(Duration::from_millis(500));

    let stats = loader.memory_stats();
    prop_assert!(stats.vram_used <= config.vram_budget);
}
```

### 6.2 Layer-based Prefetching

```rust
#[test]
fn test_layer_prefetch_during_forward() {
    let plan = create_48_layer_plan_with_swapping();
    let loader = AdaptiveLoader::new(plan);
    loader.set_prefetch_depth(2);

    let mut total_wait_time = Duration::ZERO;

    for layer in 0..48 {
        // Notify current layer (triggers prefetch of layer+1, layer+2)
        loader.notify_layer_start(layer);

        // Get current layer tensors
        let start = Instant::now();
        for tensor in layer_tensors(layer) {
            let _ = loader.get(&tensor, &Device::Cuda(0), DType::BF16);
        }
        total_wait_time += start.elapsed();
    }

    // With good prefetching, total wait time should be low
    // (prefetch overlaps with compute)
    assert!(total_wait_time < Duration::from_secs(5));
}
```

---

## 7. Integration Tests

### 7.1 End-to-End Scenarios

```rust
#[test]
fn test_14b_inference_no_swapping() {
    let model = load_qwen2_14b_hct();
    let config = AdaptiveTieringConfig {
        vram_budget: 24 * GB,
        enable_mixed_precision: true,
        ..Default::default()
    };

    let engine = InferenceEngine::with_adaptive_tiering(model, config);

    // Generate tokens
    let start = Instant::now();
    let output = engine.generate("Hello, world!", 100);
    let elapsed = start.elapsed();

    // Should be fast (no layer swapping)
    let tokens_per_sec = 100.0 / elapsed.as_secs_f64();
    assert!(tokens_per_sec > 5.0, "Expected >5 tk/s, got {}", tokens_per_sec);

    // Verify no swaps occurred
    let stats = engine.loader_stats();
    assert_eq!(stats.swap_count, 0);
}

#[test]
fn test_70b_inference_with_managed_swapping() {
    let model = load_llama_70b_hct();
    let config = AdaptiveTieringConfig {
        vram_budget: 24 * GB,
        ram_budget: 64 * GB,
        ..Default::default()
    };

    let engine = InferenceEngine::with_adaptive_tiering(model, config);

    let output = engine.generate("Explain quantum computing", 50);

    // Should complete (not OOM)
    assert!(output.len() > 0);

    // Swaps should happen but be managed
    let stats = engine.loader_stats();
    assert!(stats.swap_count > 0);  // 70B requires swapping on 24GB
    assert!(stats.average_swap_latency < Duration::from_millis(200));
}
```

### 7.2 Comparison Benchmarks

```rust
#[test]
fn test_adaptive_faster_than_fixed_layer_swap() {
    let model = load_qwen2_14b_hct();

    // Adaptive tiering
    let adaptive_config = AdaptiveTieringConfig::default();
    let adaptive_engine = InferenceEngine::with_adaptive_tiering(model.clone(), adaptive_config);

    // Fixed layer swapping (old behavior)
    let fixed_engine = InferenceEngine::with_fixed_layer_swap(model.clone(), 15);

    let prompt = "Explain the theory of relativity";
    let tokens = 100;

    let adaptive_time = benchmark(|| adaptive_engine.generate(prompt, tokens));
    let fixed_time = benchmark(|| fixed_engine.generate(prompt, tokens));

    // Adaptive should be significantly faster for 14B on 24GB
    assert!(
        adaptive_time < fixed_time / 3,
        "Adaptive ({:?}) should be >3x faster than fixed ({:?})",
        adaptive_time,
        fixed_time
    );
}
```

---

## 8. Chaos/Fuzz Tests

### 8.1 Memory Pressure

```rust
// Fuzz: Random allocation under varying memory pressure
#[proptest]
fn fuzz_allocation_under_pressure(
    model: ArbitraryModelProfile,
    vram_budget in 1_u64..100 * GB,
    ram_budget in 1_u64..500 * GB,
    kv_cache_sizes: Vec<u64>,
) {
    let config = AdaptiveTieringConfig {
        vram_budget,
        ram_budget,
        ..Default::default()
    };

    let planner = AllocationPlanner::new(config.clone());
    let plan = planner.plan(&model);

    // Should either succeed with valid plan or fail gracefully
    match plan {
        Ok(p) => {
            prop_assert!(p.vram_usage <= vram_budget);
            prop_assert!(p.ram_usage <= ram_budget);
        }
        Err(e) => {
            // Only acceptable error: model doesn't fit anywhere
            prop_assert!(matches!(e, AllocationError::InsufficientMemory { .. }));
        }
    }
}
```

### 8.2 Concurrent Access

```rust
// Fuzz: Concurrent tensor access shouldn't cause data races
#[proptest]
fn fuzz_concurrent_access(
    plan: ArbitraryAllocationPlan,
    access_sequences: Vec<Vec<String>>,
) {
    let loader = Arc::new(AdaptiveLoader::new(plan));

    let handles: Vec<_> = access_sequences
        .into_iter()
        .map(|sequence| {
            let loader = Arc::clone(&loader);
            std::thread::spawn(move || {
                for name in sequence {
                    if loader.contains(&name) {
                        let _ = loader.get(&name, &Device::Cuda(0), DType::BF16);
                    }
                }
            })
        })
        .collect();

    // All threads should complete without panic
    for handle in handles {
        prop_assert!(handle.join().is_ok());
    }

    // Memory invariants should hold
    let stats = loader.memory_stats();
    prop_assert!(stats.vram_used <= loader.config().vram_budget);
}
```

---

## 9. Test Utilities

### 9.1 Arbitrary Generators

```rust
impl Arbitrary for ArbitraryModelProfile {
    fn arbitrary(g: &mut Gen) -> Self {
        let num_layers = g.gen_range(6..128);
        let hidden_size = *g.choose(&[768, 1024, 2048, 4096, 5120, 8192]).unwrap();
        let intermediate_size = hidden_size * 4;

        ModelProfile::generate(num_layers, hidden_size, intermediate_size)
    }
}

impl Arbitrary for ArbitraryAllocationPlan {
    fn arbitrary(g: &mut Gen) -> Self {
        let model = ArbitraryModelProfile::arbitrary(g);
        let vram_budget = g.gen_range(8 * GB..48 * GB);
        let ram_budget = g.gen_range(32 * GB..256 * GB);

        let config = AdaptiveTieringConfig {
            vram_budget,
            ram_budget,
            ..Default::default()
        };

        let planner = AllocationPlanner::new(config);
        planner.plan(&model).unwrap_or_else(|_| AllocationPlan::empty())
    }
}
```

### 9.2 Test Fixtures

```rust
fn create_test_model_14b() -> ModelProfile {
    ModelProfile {
        num_layers: 48,
        hidden_size: 5120,
        intermediate_size: 13824,
        vocab_size: 152064,
        ..Default::default()
    }
}

fn create_test_model_70b() -> ModelProfile {
    ModelProfile {
        num_layers: 80,
        hidden_size: 8192,
        intermediate_size: 28672,
        vocab_size: 128256,
        ..Default::default()
    }
}

const GB: u64 = 1024 * 1024 * 1024;
```

---

## 10. Test Execution Order

Tests should be run in dependency order:

1. **Importance Scoring** (§1) — Foundation, no dependencies
2. **Precision Conversion** (§3) — Independent of allocation
3. **Allocation Planning** (§2) — Depends on scoring
4. **Adaptive Loader** (§4) — Depends on planning
5. **Runtime Adaptation** (§5) — Depends on loader
6. **Prefetching** (§6) — Depends on loader
7. **Integration Tests** (§7) — Depends on all above
8. **Chaos/Fuzz Tests** (§8) — Final validation

```bash
# Run in order
cargo test --package abaddon importance_scoring -- --test-threads=1
cargo test --package abaddon precision_conversion -- --test-threads=1
cargo test --package abaddon allocation_planning -- --test-threads=1
cargo test --package abaddon adaptive_loader -- --test-threads=1
cargo test --package abaddon runtime_adaptation -- --test-threads=1
cargo test --package abaddon prefetching -- --test-threads=1
cargo test --package abaddon integration -- --test-threads=1
cargo test --package abaddon chaos_fuzz -- --test-threads=1

# Or all at once (parallel safe)
cargo test --package abaddon
```
