# CUDA Adaptive Tiering TDD Roadmap

**Version:** 0.1.0
**Status:** Test Specification
**Date:** 2026-02-06
**Spec Reference:** CUDA-ADAPTIVE-TIERING-SPEC.md v0.1.0

---

## Philosophy

Tests are crystallized understanding, not coverage theater.

Each test exists because it captures something we **must** know is true:
- Memory budgets are **hard limits** (OOM is catastrophic)
- Tier transitions preserve tensor integrity
- Prefetch improves latency without breaking invariants
- Eviction respects priority ordering

---

## 1. Memory Budget Invariants (CRITICAL)

**Trust Boundary:** Memory limits cannot be exceeded. OOM crashes the system.

### 1.1 VRAM Budget

```rust
/// CRITICAL: VRAM usage must NEVER exceed budget.
#[proptest]
fn prop_vram_never_exceeds_budget(
    profile in model_profile_strategy(),
    config in config_strategy(),
    access_pattern in access_pattern_strategy(),
) {
    let mut store = TieredWeightStore::load_test(&profile, &config)?;

    for layer_idx in access_pattern {
        let _ = store.get_layer(layer_idx);

        prop_assert!(
            store.vram_usage() <= config.hardware.vram_budget,
            "VRAM {} exceeded budget {} after accessing layer {}",
            store.vram_usage(),
            config.hardware.vram_budget,
            layer_idx
        );
    }
}

/// CRITICAL: VRAM budget respected during KV cache growth.
#[proptest]
fn prop_vram_budget_with_kv_pressure(
    profile in model_profile_strategy(),
    config in config_strategy(),
    kv_growth_sequence in vec(1_000_000u64..100_000_000u64, 1..20),
) {
    let mut store = TieredWeightStore::load_test(&profile, &config)?;

    for kv_growth in kv_growth_sequence {
        store.evict_for_kv_cache(kv_growth)?;

        prop_assert!(
            store.vram_usage() + kv_growth <= config.hardware.vram_budget,
            "VRAM + KV cache would exceed budget"
        );
    }
}
```

### 1.2 RAM Budget

```rust
/// RAM usage must not exceed budget.
#[proptest]
fn prop_ram_never_exceeds_budget(
    profile in model_profile_strategy(),
    config in config_strategy(),
) {
    let store = TieredWeightStore::load_test(&profile, &config)?;

    prop_assert!(
        store.ram_usage() <= config.hardware.ram_budget,
        "RAM {} exceeded budget {}",
        store.ram_usage(),
        config.hardware.ram_budget
    );
}
```

### 1.3 Total Memory Accounting

```rust
/// Total usage equals sum of tier usages.
#[proptest]
fn prop_memory_accounting_correct(
    profile in model_profile_strategy(),
    config in config_strategy(),
) {
    let store = TieredWeightStore::load_test(&profile, &config)?;

    let vram = store.vram_usage();
    let ram = store.ram_usage();
    let nvme = store.nvme_usage();
    let total = store.total_usage();

    prop_assert_eq!(total, vram + ram + nvme);
}
```

---

## 2. Tier Transition Integrity

**Trust Boundary:** Tensors must be identical after tier transitions.

### 2.1 VRAM ↔ RAM Transitions

```rust
/// Tensor data preserved through VRAM→RAM eviction.
#[test]
fn test_vram_to_ram_preserves_data() {
    let store = create_test_store();

    // Load layer to VRAM
    let layer_vram = store.get_layer(0)?.clone_data();

    // Force eviction to RAM
    store.force_evict_layer(0);
    assert!(!store.vram_cache.contains(0));
    assert!(store.ram_cache.contains(0));

    // Reload to VRAM
    let layer_reloaded = store.get_layer(0)?;

    assert_tensors_equal(&layer_vram, &layer_reloaded);
}

/// Tensor data preserved through RAM→VRAM upload.
#[test]
fn test_ram_to_vram_preserves_data() {
    let store = create_test_store_with_ram_layers();

    // Layer starts in RAM
    assert!(store.ram_cache.contains(5));
    let expected = store.ram_cache.get(5)?.clone_data();

    // Access promotes to VRAM
    let layer = store.get_layer(5)?;

    assert_tensors_equal(&expected, &layer);
}
```

### 2.2 RAM ↔ NVMe Transitions

```rust
/// Tensor data preserved through RAM→NVMe eviction.
#[test]
fn test_ram_to_nvme_preserves_data() {
    let store = create_test_store_progressive();

    // Load to RAM
    store.load_layer_to_ram(10)?;
    let expected = store.ram_cache.get(10)?.clone_data();

    // Evict to NVMe
    store.evict_ram_layer(10)?;

    // Reload from NVMe
    store.load_layer_to_ram(10)?;
    let reloaded = store.ram_cache.get(10)?;

    assert_tensors_equal(&expected, &reloaded);
}

/// NVMe cache produces same result as direct HCT load.
#[proptest]
fn prop_nvme_cache_matches_hct(
    layer_idx in 0usize..80,
) {
    let store = create_test_store_progressive();

    // Load from HCT (bypass cache)
    let from_hct = store.nvme_cache.load_from_hct(layer_idx)?;

    // Load from cache (may decompress and cache)
    let from_cache = store.nvme_cache.load_layer(layer_idx)?;

    assert_tensors_equal(&from_hct, &from_cache);
}
```

### 2.3 Full Round-Trip

```rust
/// Tensor survives full tier round-trip: VRAM → RAM → NVMe → RAM → VRAM
#[test]
fn test_full_tier_round_trip() {
    let store = create_test_store_progressive();

    // Start in VRAM
    let original = store.get_layer(0)?.clone_data();

    // VRAM → RAM
    store.force_evict_to_ram(0);

    // RAM → NVMe
    store.force_evict_to_nvme(0);

    // NVMe → RAM
    store.load_layer_to_ram(0)?;

    // RAM → VRAM
    let final_layer = store.get_layer(0)?;

    assert_tensors_equal(&original, &final_layer);
}
```

---

## 3. Eviction Ordering

**Trust Boundary:** Eviction must respect priority to maintain inference quality.

### 3.1 Priority-Based Eviction

```rust
/// Low-priority layers evicted before high-priority layers.
#[test]
fn test_eviction_respects_priority() {
    let plan = create_plan_with_priorities(vec![
        (0, 1.0),  // High priority (edge layer)
        (1, 0.9),
        (40, 0.5), // Low priority (middle layer)
        (41, 0.5),
        (79, 1.0), // High priority (edge layer)
    ]);

    let store = create_store_with_plan(plan);

    // Load all layers
    for i in [0, 1, 40, 41, 79] {
        store.get_layer(i)?;
    }

    // Evict enough for one layer
    let layer_size = store.layer_size();
    store.evict_for_kv_cache(layer_size)?;

    // Middle layers should be evicted first
    assert!(!store.vram_cache.contains(40) || !store.vram_cache.contains(41));
    assert!(store.vram_cache.contains(0)); // Edge layer still present
    assert!(store.vram_cache.contains(79)); // Edge layer still present
}

/// Shared weights (embed, lm_head) are never evicted.
#[proptest]
fn prop_shared_weights_never_evicted(
    eviction_requests in vec(1_000_000u64..1_000_000_000u64, 1..100),
) {
    let store = create_test_store();

    for bytes in eviction_requests {
        store.evict_for_kv_cache(bytes)?;

        prop_assert!(store.has_embed_tokens(), "embed_tokens was evicted");
        prop_assert!(store.has_final_norm(), "final_norm was evicted");
        prop_assert!(store.has_lm_head(), "lm_head was evicted");
    }
}
```

### 3.2 LRU Within Priority

```rust
/// Among same-priority layers, evict least recently used.
#[test]
fn test_lru_within_priority() {
    let plan = create_plan_with_same_priority(0.5); // All middle layers
    let store = create_store_with_plan(plan);

    // Access in order: 10, 11, 12, then 10 again
    store.get_layer(10)?;
    store.get_layer(11)?;
    store.get_layer(12)?;
    store.get_layer(10)?; // 10 is now most recent

    // Evict one layer
    store.evict_for_kv_cache(store.layer_size())?;

    // Layer 11 should be evicted (oldest among same priority)
    assert!(!store.vram_cache.contains(11));
    assert!(store.vram_cache.contains(10)); // Recently accessed
    assert!(store.vram_cache.contains(12)); // More recent than 11
}
```

---

## 4. Loading Strategy Selection

**Trust Boundary:** Strategy selection determines performance characteristics.

### 4.1 Eager When Fits

```rust
/// Eager strategy selected when model fits in VRAM + RAM.
#[test]
fn test_eager_when_fits() {
    let profile = make_14b_profile(); // ~29GB
    let config = HardwareConfig {
        vram_budget: 22 * GB,
        ram_budget: 60 * GB,
        ..Default::default()
    };

    let plan = AllocationPlanner::new(config.into()).plan(&profile)?;
    let strategy = select_strategy(&plan, &config);

    assert!(matches!(strategy, LoadingStrategy::Eager { .. }));
    assert_eq!(plan.nvme_usage, 0);
}

/// Progressive strategy selected when NVMe needed.
#[test]
fn test_progressive_when_nvme_needed() {
    let profile = make_70b_profile(); // ~140GB
    let config = HardwareConfig {
        vram_budget: 22 * GB,
        ram_budget: 60 * GB,
        ..Default::default()
    };

    let plan = AllocationPlanner::new(config.into()).plan(&profile)?;
    let strategy = select_strategy(&plan, &config);

    assert!(matches!(strategy, LoadingStrategy::Progressive { .. }));
    assert!(plan.nvme_usage > 0);
}
```

### 4.2 Strategy Produces Valid Store

```rust
/// Any selected strategy produces a functional store.
#[proptest]
fn prop_strategy_produces_valid_store(
    profile in model_profile_strategy(),
    config in hardware_config_strategy(),
) {
    let plan = AllocationPlanner::new(config.clone().into()).plan(&profile)?;
    let strategy = select_strategy(&plan, &config);

    let store = match strategy {
        LoadingStrategy::Eager { .. } =>
            TieredWeightStore::load_eager_test(&profile, &plan)?,
        LoadingStrategy::Progressive { .. } =>
            TieredWeightStore::load_progressive_test(&profile, &plan)?,
        LoadingStrategy::EagerQuantized { .. } =>
            TieredWeightStore::load_quantized_test(&profile, &plan)?,
    };

    // Verify all layers accessible
    for i in 0..profile.num_layers {
        let layer = store.get_layer(i);
        prop_assert!(layer.is_ok(), "layer {} not accessible", i);
    }
}
```

---

## 5. Prefetching

**Trust Boundary:** Prefetch improves performance without breaking correctness.

### 5.1 Prefetch Does Not Block

```rust
/// Prefetch requests return immediately (non-blocking).
#[test]
fn test_prefetch_non_blocking() {
    let store = create_test_store_progressive();

    let start = Instant::now();
    store.prefetch_layers(&[10, 11, 12, 13, 14]);
    let elapsed = start.elapsed();

    // Prefetch should return in <1ms (just sends message)
    assert!(elapsed < Duration::from_millis(1));
}

/// Prefetch followed by access is faster than cold access.
#[test]
fn test_prefetch_reduces_latency() {
    let store = create_test_store_progressive();

    // Cold access (no prefetch)
    store.clear_caches();
    let cold_start = Instant::now();
    store.get_layer(20)?;
    let cold_latency = cold_start.elapsed();

    // Clear and prefetch
    store.clear_caches();
    store.prefetch_layers(&[21]);
    std::thread::sleep(Duration::from_millis(500)); // Wait for prefetch

    // Warm access
    let warm_start = Instant::now();
    store.get_layer(21)?;
    let warm_latency = warm_start.elapsed();

    // Warm should be significantly faster
    assert!(warm_latency < cold_latency / 2);
}
```

### 5.2 Prefetch Correctness

```rust
/// Prefetched layer is identical to directly loaded layer.
#[proptest]
fn prop_prefetch_same_as_direct(
    layer_idx in 0usize..80,
) {
    let store1 = create_test_store_progressive();
    let store2 = create_test_store_progressive();

    // Direct load
    let direct = store1.get_layer(layer_idx)?.clone_data();

    // Prefetch then load
    store2.prefetch_layers(&[layer_idx]);
    std::thread::sleep(Duration::from_millis(500));
    let prefetched = store2.get_layer(layer_idx)?;

    assert_tensors_equal(&direct, &prefetched);
}
```

---

## 6. NVMe Cache

**Trust Boundary:** Cache must be transparent - cached vs uncached should be identical.

### 6.1 Cache Transparency

```rust
/// Cached load identical to uncached load.
#[proptest]
fn prop_cache_transparent(
    layer_idx in 0usize..80,
) {
    let store = create_test_store_progressive();

    // First load (populates cache)
    let first = store.nvme_cache.load_layer(layer_idx)?.clone_data();

    // Second load (from cache)
    let second = store.nvme_cache.load_layer(layer_idx)?;

    assert_tensors_equal(&first, &second);
}

/// Cache survives store recreation.
#[test]
fn test_cache_persistence() {
    let cache_dir = tempdir()?;

    // First session: load and cache
    {
        let store = create_test_store_with_cache_dir(&cache_dir);
        store.nvme_cache.load_layer(0)?;
        assert!(cache_dir.path().join("layer_000_bf16.safetensor").exists());
    }

    // Second session: load from cache
    {
        let store = create_test_store_with_cache_dir(&cache_dir);
        let layer = store.nvme_cache.load_layer(0)?;
        assert!(layer.is_valid());
    }
}
```

### 6.2 Cache Size Limits

```rust
/// Cache respects size limit.
#[proptest]
fn prop_cache_respects_limit(
    max_size in 1_000_000_000u64..10_000_000_000u64,
    layer_loads in vec(0usize..80, 10..50),
) {
    let store = create_test_store_with_cache_limit(max_size);

    for layer_idx in layer_loads {
        store.nvme_cache.load_layer(layer_idx)?;

        prop_assert!(
            store.nvme_cache.usage() <= max_size,
            "cache {} exceeded limit {}",
            store.nvme_cache.usage(),
            max_size
        );
    }
}

/// Cache evicts LRU entries when full.
#[test]
fn test_cache_lru_eviction() {
    let layer_size = 1 * GB;
    let max_cache = 3 * GB; // Room for 3 layers

    let store = create_test_store_with_cache_limit(max_cache);

    // Load 4 layers (exceeds cache)
    for i in 0..4 {
        store.nvme_cache.load_layer(i)?;
    }

    // First layer should be evicted
    assert!(!store.nvme_cache.is_cached(0));
    assert!(store.nvme_cache.is_cached(3)); // Most recent
}
```

---

## 7. Concurrent Access

**Trust Boundary:** Concurrent operations must not corrupt state.

### 7.1 Thread Safety

```rust
/// Concurrent layer access is safe.
#[test]
fn test_concurrent_layer_access() {
    let store = Arc::new(create_test_store());
    let handles: Vec<_> = (0..10)
        .map(|t| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    let layer_idx = (t * 7 + rand::random::<usize>()) % 80;
                    let _ = store.get_layer(layer_idx);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Verify store is still valid
    assert!(store.vram_usage() <= store.vram_budget());
}

/// Concurrent prefetch and access is safe.
#[test]
fn test_concurrent_prefetch_and_access() {
    let store = Arc::new(create_test_store_progressive());

    // Prefetch thread
    let store_prefetch = Arc::clone(&store);
    let prefetch_handle = std::thread::spawn(move || {
        for _ in 0..100 {
            let layers: Vec<_> = (0..5).map(|_| rand::random::<usize>() % 80).collect();
            store_prefetch.prefetch_layers(&layers);
            std::thread::sleep(Duration::from_millis(10));
        }
    });

    // Access thread
    let store_access = Arc::clone(&store);
    let access_handle = std::thread::spawn(move || {
        for _ in 0..100 {
            let layer_idx = rand::random::<usize>() % 80;
            let _ = store_access.get_layer(layer_idx);
        }
    });

    prefetch_handle.join().expect("prefetch thread panicked");
    access_handle.join().expect("access thread panicked");
}
```

---

## 8. Integration Tests

### 8.1 Forward Pass

```rust
/// Complete forward pass with tiered store.
#[test]
#[ignore] // Requires GPU
fn test_forward_pass_tiered() {
    let store = create_test_store_with_real_model();
    let engine = ComputeEngine::new(store.model_config())?;
    let kv_cache = KvCache::new(store.model_config(), 2048)?;

    let input_ids = create_test_input(5); // 5 tokens

    let logits = engine.forward_tiered(&input_ids, &mut store, &mut kv_cache)?;

    assert_eq!(logits.shape(), &[5, store.vocab_size()]);
    assert!(!logits.has_nan());
}

/// Generation produces valid tokens.
#[test]
#[ignore] // Requires model files
fn test_generation_tiered() {
    let config = TieredConfig::for_testing();
    let mut generator = TieredGenerator::new("/path/to/model", config)?;

    let output = generator.generate("Hello", 10)?;

    assert!(!output.is_empty());
    assert!(output.chars().all(|c| !c.is_control() || c == '\n'));
}
```

### 8.2 Performance Benchmarks

```rust
/// Measure throughput for different configurations.
#[bench]
fn bench_throughput_14b_eager(b: &mut Bencher) {
    let generator = create_14b_eager_generator();

    b.iter(|| {
        generator.generate("Test prompt", 100)
    });

    // Report tokens/second
    let stats = generator.stats();
    println!("Throughput: {:.2} tk/s", stats.tokens_per_second);
}

#[bench]
fn bench_throughput_70b_progressive(b: &mut Bencher) {
    let generator = create_70b_progressive_generator();

    b.iter(|| {
        generator.generate("Test prompt", 10)
    });

    let stats = generator.stats();
    println!("Throughput: {:.2} tk/s", stats.tokens_per_second);
    println!("VRAM hits: {}", stats.vram_hits);
    println!("RAM hits: {}", stats.ram_hits);
    println!("NVMe hits: {}", stats.nvme_hits);
}
```

---

## 9. Regression Tests

Tests for specific bugs that have been fixed.

```rust
/// Regression: KV cache growth should not corrupt layer weights.
#[test]
fn regression_kv_growth_no_corruption() {
    let store = create_test_store();

    // Load layer and save reference
    let layer_before = store.get_layer(0)?.clone_data();

    // Simulate KV cache growth with eviction
    for _ in 0..10 {
        store.evict_for_kv_cache(100 * MB)?;
        store.restore_for_generation()?;
    }

    // Layer should be identical
    let layer_after = store.get_layer(0)?;
    assert_tensors_equal(&layer_before, &layer_after);
}

/// Regression: Prefetch thread should not panic on shutdown.
#[test]
fn regression_clean_shutdown() {
    let store = create_test_store_progressive();

    // Start some prefetches
    store.prefetch_layers(&[0, 1, 2, 3, 4]);

    // Immediate shutdown
    drop(store); // Should not panic
}
```

---

## 10. Test Utilities

```rust
/// Test helper: create tensors and compare.
fn assert_tensors_equal(a: &TensorData, b: &TensorData) {
    assert_eq!(a.shape(), b.shape(), "shapes differ");
    assert_eq!(a.dtype(), b.dtype(), "dtypes differ");

    let a_data = a.to_vec_f32();
    let b_data = b.to_vec_f32();

    for (i, (av, bv)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < 1e-5,
            "tensors differ at index {}: {} vs {}",
            i, av, bv
        );
    }
}

/// Test helper: create model profile for 14B model.
fn make_14b_profile() -> ModelProfile {
    let mut tensors = Vec::new();
    // ... 48 layers, ~600MB each
    ModelProfile::new(tensors)
}

/// Test helper: create model profile for 70B model.
fn make_70b_profile() -> ModelProfile {
    let mut tensors = Vec::new();
    // ... 80 layers, ~1.75GB each
    ModelProfile::new(tensors)
}

/// Proptest strategy for model profiles.
fn model_profile_strategy() -> impl Strategy<Value = ModelProfile> {
    (12usize..128, 100_000_000u64..10_000_000_000u64)
        .prop_map(|(num_layers, layer_size)| {
            make_profile_with_layers(num_layers, layer_size)
        })
}

/// Proptest strategy for hardware configs.
fn hardware_config_strategy() -> impl Strategy<Value = HardwareConfig> {
    (8u64..48, 16u64..256, 0u64..1000)
        .prop_map(|(vram_gb, ram_gb, nvme_gb)| HardwareConfig {
            vram_budget: vram_gb * GB,
            ram_budget: ram_gb * GB,
            nvme_cache_size: nvme_gb * GB,
            use_pinned_memory: true,
        })
}
```

---

## 11. Implementation Order

Tests should be implemented in this order, with each phase building on the previous:

### Phase 1: Memory Invariants
1. `prop_vram_never_exceeds_budget`
2. `prop_ram_never_exceeds_budget`
3. `prop_memory_accounting_correct`

### Phase 2: Tier Transitions
4. `test_vram_to_ram_preserves_data`
5. `test_ram_to_vram_preserves_data`
6. `test_full_tier_round_trip`

### Phase 3: Eviction
7. `test_eviction_respects_priority`
8. `prop_shared_weights_never_evicted`
9. `test_lru_within_priority`

### Phase 4: Loading Strategies
10. `test_eager_when_fits`
11. `test_progressive_when_nvme_needed`
12. `prop_strategy_produces_valid_store`

### Phase 5: Prefetch & Cache
13. `test_prefetch_non_blocking`
14. `test_prefetch_reduces_latency`
15. `prop_cache_transparent`
16. `prop_cache_respects_limit`

### Phase 6: Concurrency
17. `test_concurrent_layer_access`
18. `test_concurrent_prefetch_and_access`

### Phase 7: Integration
19. `test_forward_pass_tiered`
20. `test_generation_tiered`
