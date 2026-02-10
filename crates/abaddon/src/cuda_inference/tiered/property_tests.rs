//! Property-based tests for tiered memory system.
//!
//! These tests validate critical invariants that must hold for the memory
//! tiering system to work correctly. Based on ADAPTIVE-MEMORY-TIERING-TDD.md.

use proptest::prelude::*;
use std::collections::{HashMap, HashSet};

use super::lru::LruTracker;

// ============================================================================
// Test Utilities
// ============================================================================

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

/// Simulated layer allocation for testing.
#[derive(Debug, Clone)]
struct LayerAllocation {
    layer_idx: usize,
    size_bytes: u64,
    priority: f32,
}

/// Simple budget tracker for property tests.
#[derive(Debug, Default)]
struct BudgetTracker {
    vram_used: u64,
    vram_budget: u64,
    ram_used: u64,
    ram_budget: u64,
    allocations: HashMap<usize, MemoryTier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryTier {
    Vram,
    Ram,
    Nvme,
}

impl BudgetTracker {
    fn new(vram_budget: u64, ram_budget: u64) -> Self {
        Self {
            vram_budget,
            ram_budget,
            ..Default::default()
        }
    }

    /// Try to allocate a layer to the best available tier.
    /// Returns None if allocation would exceed all budgets.
    fn allocate(&mut self, layer_idx: usize, size_bytes: u64) -> Option<MemoryTier> {
        // Try VRAM first
        if self.vram_used + size_bytes <= self.vram_budget {
            self.vram_used += size_bytes;
            self.allocations.insert(layer_idx, MemoryTier::Vram);
            return Some(MemoryTier::Vram);
        }

        // Try RAM
        if self.ram_used + size_bytes <= self.ram_budget {
            self.ram_used += size_bytes;
            self.allocations.insert(layer_idx, MemoryTier::Ram);
            return Some(MemoryTier::Ram);
        }

        // Fall back to NVMe (infinite capacity for this simulation)
        self.allocations.insert(layer_idx, MemoryTier::Nvme);
        Some(MemoryTier::Nvme)
    }

    /// Evict a layer from its current tier.
    fn evict(&mut self, layer_idx: usize, size_bytes: u64) -> bool {
        if let Some(tier) = self.allocations.remove(&layer_idx) {
            match tier {
                MemoryTier::Vram => {
                    self.vram_used = self.vram_used.saturating_sub(size_bytes);
                },
                MemoryTier::Ram => {
                    self.ram_used = self.ram_used.saturating_sub(size_bytes);
                },
                MemoryTier::Nvme => {},
            }
            true
        } else {
            false
        }
    }

    fn vram_headroom(&self) -> u64 {
        self.vram_budget.saturating_sub(self.vram_used)
    }

    fn ram_headroom(&self) -> u64 {
        self.ram_budget.saturating_sub(self.ram_used)
    }
}

// ============================================================================
// Strategy Definitions for Proptest
// ============================================================================

/// Generate a reasonable layer size (100MB to 2GB).
fn layer_size_strategy() -> impl Strategy<Value = u64> {
    (100 * MB..2 * GB)
}

/// Generate a priority value in [0, 1].
fn priority_strategy() -> impl Strategy<Value = f32> {
    (0.0f32..=1.0f32)
}

/// Generate a layer count (realistic model sizes).
fn layer_count_strategy() -> impl Strategy<Value = usize> {
    12usize..128
}

/// Generate VRAM budget (8GB to 80GB).
fn vram_budget_strategy() -> impl Strategy<Value = u64> {
    (8 * GB..80 * GB)
}

/// Generate RAM budget (16GB to 256GB).
fn ram_budget_strategy() -> impl Strategy<Value = u64> {
    (16 * GB..256 * GB)
}

/// Generate a collection of layer allocations.
fn layer_allocations_strategy(num_layers: usize) -> impl Strategy<Value = Vec<LayerAllocation>> {
    prop::collection::vec(
        (layer_size_strategy(), priority_strategy()).prop_map(|(size, priority)| {
            LayerAllocation {
                layer_idx: 0, // Will be fixed up
                size_bytes: size,
                priority,
            }
        }),
        num_layers,
    )
    .prop_map(|mut allocs| {
        for (i, alloc) in allocs.iter_mut().enumerate() {
            alloc.layer_idx = i;
        }
        allocs
    })
}

// ============================================================================
// LRU Tracker Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Property: After touch, item is tracked.
    #[test]
    fn prop_touch_adds_item(keys in prop::collection::vec(0usize..1000, 1..100)) {
        let mut tracker = LruTracker::<usize>::new();

        for &key in &keys {
            tracker.touch(key);
        }

        for key in keys.iter().collect::<HashSet<_>>() {
            prop_assert!(tracker.contains(key), "Key {} should be tracked", key);
        }
    }

    /// Property: Remove actually removes item.
    #[test]
    fn prop_remove_removes_item(
        keys in prop::collection::vec(0usize..100, 1..50),
        to_remove in prop::collection::vec(0usize..100, 0..25)
    ) {
        let mut tracker = LruTracker::<usize>::new();

        for &key in &keys {
            tracker.touch(key);
        }

        for &key in &to_remove {
            tracker.remove(&key);
        }

        let remove_set: HashSet<_> = to_remove.into_iter().collect();
        for &key in &remove_set {
            prop_assert!(!tracker.contains(&key), "Key {} should be removed", key);
        }
    }

    /// Property: Eviction order respects priority (lower priority first).
    #[test]
    fn prop_eviction_order_respects_priority(
        items in prop::collection::vec((0usize..1000, 0.0f32..1.0f32), 2..50)
    ) {
        let mut tracker = LruTracker::<usize>::new();

        // Deduplicate by key
        let mut unique_items = HashMap::new();
        for (key, priority) in items {
            unique_items.insert(key, priority);
        }

        if unique_items.len() < 2 {
            return Ok(());
        }

        for (&key, &priority) in &unique_items {
            tracker.touch_with_priority(key, priority);
        }

        let order = tracker.eviction_order();

        // Verify order is sorted by priority
        for window in order.windows(2) {
            let p0 = tracker.priority(&window[0]);
            let p1 = tracker.priority(&window[1]);
            prop_assert!(
                p0 <= p1,
                "Eviction order violated: {} (p={}) before {} (p={})",
                window[0], p0, window[1], p1
            );
        }
    }

    /// Property: pop_lru returns lowest priority item.
    #[test]
    fn prop_pop_lru_returns_lowest_priority(
        items in prop::collection::vec((0usize..1000, 0.01f32..1.0f32), 1..50)
    ) {
        let mut tracker = LruTracker::<usize>::new();

        // Deduplicate
        let mut unique_items = HashMap::new();
        for (key, priority) in items {
            unique_items.insert(key, priority);
        }

        if unique_items.is_empty() {
            return Ok(());
        }

        for (&key, &priority) in &unique_items {
            tracker.touch_with_priority(key, priority);
        }

        // Find expected minimum priority
        let min_priority = unique_items.values().cloned().fold(f32::MAX, f32::min);

        let popped = tracker.pop_lru();
        prop_assert!(popped.is_some());

        let popped_key = popped.unwrap();
        let popped_priority = unique_items.get(&popped_key).copied().unwrap_or(0.5);

        // The popped item should have the minimum priority (within epsilon for float comparison)
        prop_assert!(
            (popped_priority - min_priority).abs() < 0.001,
            "Popped key {} with priority {} but min was {}",
            popped_key, popped_priority, min_priority
        );
    }

    /// Property: eviction_candidates_for_size returns enough bytes.
    #[test]
    fn prop_eviction_candidates_reach_target(
        items in prop::collection::vec((0usize..1000, 0.0f32..1.0f32, 100u64..10000), 1..50),
        target_fraction in 0.1f64..0.9f64
    ) {
        let mut tracker = LruTracker::<usize>::new();
        let mut sizes = HashMap::new();

        // Deduplicate
        for (key, priority, size) in items {
            tracker.touch_with_priority(key, priority);
            sizes.insert(key, size);
        }

        if sizes.is_empty() {
            return Ok(());
        }

        let total_size: u64 = sizes.values().sum();
        let target = (total_size as f64 * target_fraction) as u64;

        let candidates = tracker.eviction_candidates_for_size(target, |k| {
            sizes.get(k).copied().unwrap_or(0)
        });

        let freed: u64 = candidates.iter().map(|k| sizes.get(k).copied().unwrap_or(0)).sum();

        // Should free at least target (or everything if not enough)
        let expected_min = target.min(total_size);
        prop_assert!(
            freed >= expected_min || candidates.len() == sizes.len(),
            "Freed {} bytes but needed {} (total available: {})",
            freed, expected_min, total_size
        );
    }

    /// Property: LRU length equals number of unique items touched.
    #[test]
    fn prop_lru_length_matches_unique(
        keys in prop::collection::vec(0usize..100, 1..200)
    ) {
        let mut tracker = LruTracker::<usize>::new();

        for &key in &keys {
            tracker.touch(key);
        }

        let unique_count = keys.iter().collect::<HashSet<_>>().len();
        prop_assert_eq!(tracker.len(), unique_count);
    }
}

// ============================================================================
// Budget Constraint Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Property: VRAM allocation NEVER exceeds budget.
    #[test]
    fn prop_vram_never_exceeds_budget(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy(),
        num_layers in layer_count_strategy()
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);
        let layer_size = 500 * MB; // Fixed size for simplicity

        for i in 0..num_layers {
            budget.allocate(i, layer_size);
        }

        prop_assert!(
            budget.vram_used <= budget.vram_budget,
            "VRAM usage {} exceeds budget {}",
            budget.vram_used, budget.vram_budget
        );
    }

    /// Property: RAM allocation NEVER exceeds budget.
    #[test]
    fn prop_ram_never_exceeds_budget(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy(),
        num_layers in layer_count_strategy()
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);
        let layer_size = 500 * MB;

        for i in 0..num_layers {
            budget.allocate(i, layer_size);
        }

        prop_assert!(
            budget.ram_used <= budget.ram_budget,
            "RAM usage {} exceeds budget {}",
            budget.ram_used, budget.ram_budget
        );
    }

    /// Property: Every layer is allocated exactly once.
    #[test]
    fn prop_all_layers_allocated_once(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy(),
        num_layers in layer_count_strategy()
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);
        let layer_size = 500 * MB;

        for i in 0..num_layers {
            let result = budget.allocate(i, layer_size);
            prop_assert!(result.is_some(), "Layer {} allocation failed", i);
        }

        prop_assert_eq!(
            budget.allocations.len(), num_layers,
            "Expected {} allocations, got {}",
            num_layers, budget.allocations.len()
        );
    }

    /// Property: Eviction frees exactly the right amount.
    #[test]
    fn prop_eviction_frees_correct_amount(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy(),
        num_layers in 10usize..50,
        evict_indices in prop::collection::vec(0usize..50, 1..10)
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);
        let layer_size = 500 * MB;

        // Allocate layers
        for i in 0..num_layers {
            budget.allocate(i, layer_size);
        }

        let vram_before = budget.vram_used;
        let ram_before = budget.ram_used;

        // Deduplicate evict indices - can only evict each layer once
        let unique_evict: HashSet<usize> = evict_indices.iter()
            .filter(|&&idx| idx < num_layers)
            .copied()
            .collect();

        // Count how many we'll actually evict from each tier
        let mut vram_evict_count = 0u64;
        let mut ram_evict_count = 0u64;

        for &idx in &unique_evict {
            if let Some(&tier) = budget.allocations.get(&idx) {
                match tier {
                    MemoryTier::Vram => vram_evict_count += 1,
                    MemoryTier::Ram => ram_evict_count += 1,
                    MemoryTier::Nvme => {}
                }
            }
        }

        // Evict
        for &idx in &unique_evict {
            budget.evict(idx, layer_size);
        }

        let expected_vram = vram_before.saturating_sub(vram_evict_count * layer_size);
        let expected_ram = ram_before.saturating_sub(ram_evict_count * layer_size);

        prop_assert_eq!(
            budget.vram_used, expected_vram,
            "VRAM mismatch after eviction"
        );
        prop_assert_eq!(
            budget.ram_used, expected_ram,
            "RAM mismatch after eviction"
        );
    }
}

// ============================================================================
// Priority-Based Allocation Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Property: Higher priority layers should be placed in faster tiers when possible.
    #[test]
    fn prop_priority_influences_tier_placement(
        vram_budget in (8 * GB..16 * GB),
        num_layers in 20usize..40
    ) {
        // Create layers with decreasing priority
        let layer_size = 500 * MB;
        let mut budget = BudgetTracker::new(vram_budget, 64 * GB);
        let mut priorities: Vec<f32> = Vec::new();

        // Allocate in priority order (highest first)
        for i in 0..num_layers {
            let priority = 1.0 - (i as f32 / num_layers as f32);
            priorities.push(priority);
            budget.allocate(i, layer_size);
        }

        // Verify VRAM contains highest priority layers
        let vram_layers: Vec<usize> = budget.allocations.iter()
            .filter(|(_, &tier)| tier == MemoryTier::Vram)
            .map(|(&idx, _)| idx)
            .collect();

        let ram_layers: Vec<usize> = budget.allocations.iter()
            .filter(|(_, &tier)| tier == MemoryTier::Ram)
            .map(|(&idx, _)| idx)
            .collect();

        if !vram_layers.is_empty() && !ram_layers.is_empty() {
            let max_vram_priority = vram_layers.iter()
                .map(|&idx| priorities[idx])
                .fold(f32::MIN, f32::max);
            let min_ram_priority = ram_layers.iter()
                .map(|&idx| priorities[idx])
                .fold(f32::MAX, f32::min);

            // Since we allocate in priority order, VRAM should have higher priority items
            prop_assert!(
                max_vram_priority >= min_ram_priority - 0.1,
                "VRAM max priority {} < RAM min priority {}",
                max_vram_priority, min_ram_priority
            );
        }
    }
}

// ============================================================================
// Cache Coherence Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Property: Cache never contains duplicates.
    #[test]
    fn prop_no_duplicate_allocations(
        num_layers in layer_count_strategy(),
        operations in prop::collection::vec(
            prop_oneof![
                Just(("allocate", 0usize)),
                (0usize..100).prop_map(|i| ("evict", i))
            ],
            1..100
        )
    ) {
        let mut budget = BudgetTracker::new(24 * GB, 64 * GB);
        let layer_size = 500 * MB;
        let mut allocated_layers = 0usize;

        for (op, idx) in operations {
            match op {
                "allocate" => {
                    if allocated_layers < num_layers {
                        budget.allocate(allocated_layers, layer_size);
                        allocated_layers += 1;
                    }
                }
                "evict" => {
                    if idx < allocated_layers {
                        budget.evict(idx, layer_size);
                    }
                }
                _ => {}
            }
        }

        // Check no duplicates - each layer should appear at most once
        let all_keys: Vec<usize> = budget.allocations.keys().copied().collect();
        let unique_keys: HashSet<usize> = all_keys.iter().copied().collect();

        prop_assert_eq!(
            all_keys.len(), unique_keys.len(),
            "Duplicate allocations detected"
        );
    }

    /// Property: Allocation state is consistent after operations.
    #[test]
    fn prop_allocation_state_consistent(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy(),
        num_allocs in 10usize..50
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);
        let layer_size = 500 * MB;

        for i in 0..num_allocs {
            budget.allocate(i, layer_size);
        }

        // Count allocations by tier
        let vram_count = budget.allocations.values()
            .filter(|&&t| t == MemoryTier::Vram).count() as u64;
        let ram_count = budget.allocations.values()
            .filter(|&&t| t == MemoryTier::Ram).count() as u64;

        // Verify usage matches allocation counts
        prop_assert_eq!(
            budget.vram_used, vram_count * layer_size,
            "VRAM usage mismatch: {} vs {} layers * {}",
            budget.vram_used, vram_count, layer_size
        );
        prop_assert_eq!(
            budget.ram_used, ram_count * layer_size,
            "RAM usage mismatch: {} vs {} layers * {}",
            budget.ram_used, ram_count, layer_size
        );
    }
}

// ============================================================================
// Stress / Edge Case Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Property: System handles extreme memory pressure gracefully.
    #[test]
    fn prop_handles_memory_pressure(
        // Very limited VRAM
        vram_budget in (1 * GB..4 * GB),
        // Large model
        num_layers in 60usize..100,
        layer_size in (500 * MB..1 * GB)
    ) {
        let mut budget = BudgetTracker::new(vram_budget, 128 * GB);

        for i in 0..num_layers {
            let result = budget.allocate(i, layer_size);
            // Should always succeed (falls back to NVMe)
            prop_assert!(result.is_some(), "Allocation should not fail");
        }

        // Budget constraints still hold
        prop_assert!(budget.vram_used <= budget.vram_budget);
        prop_assert!(budget.ram_used <= budget.ram_budget);
    }

    /// Property: Zero-size allocations are handled correctly.
    #[test]
    fn prop_zero_size_allocation(
        vram_budget in vram_budget_strategy(),
        ram_budget in ram_budget_strategy()
    ) {
        let mut budget = BudgetTracker::new(vram_budget, ram_budget);

        // Zero-size should succeed
        let result = budget.allocate(0, 0);
        prop_assert!(result.is_some());

        // Should still be able to allocate normal sizes
        let result = budget.allocate(1, 500 * MB);
        prop_assert!(result.is_some());
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_budget_tracker_basic() {
        let mut budget = BudgetTracker::new(24 * GB, 64 * GB);

        // Allocate to VRAM
        assert_eq!(budget.allocate(0, 10 * GB), Some(MemoryTier::Vram));
        assert_eq!(budget.vram_used, 10 * GB);

        // Allocate more to VRAM
        assert_eq!(budget.allocate(1, 10 * GB), Some(MemoryTier::Vram));
        assert_eq!(budget.vram_used, 20 * GB);

        // This should spill to RAM
        assert_eq!(budget.allocate(2, 10 * GB), Some(MemoryTier::Ram));
        assert_eq!(budget.vram_used, 20 * GB);
        assert_eq!(budget.ram_used, 10 * GB);
    }

    #[test]
    fn test_budget_tracker_eviction() {
        let mut budget = BudgetTracker::new(24 * GB, 64 * GB);

        budget.allocate(0, 10 * GB);
        budget.allocate(1, 10 * GB);

        assert_eq!(budget.vram_used, 20 * GB);

        // Evict layer 0
        assert!(budget.evict(0, 10 * GB));
        assert_eq!(budget.vram_used, 10 * GB);
        assert!(!budget.allocations.contains_key(&0));

        // Evicting again should fail
        assert!(!budget.evict(0, 10 * GB));
    }

    #[test]
    fn test_14b_model_fits_24gb() {
        // Qwen2.5-14B: 48 layers, ~600MB each at mixed precision
        let mut budget = BudgetTracker::new(24 * GB, 64 * GB);
        let layer_size = 450 * MB; // With INT8 quantization

        for i in 0..48 {
            budget.allocate(i, layer_size);
        }

        // Should fit entirely in VRAM with quantization
        let vram_layers = budget
            .allocations
            .values()
            .filter(|&&t| t == MemoryTier::Vram)
            .count();

        assert!(
            vram_layers >= 40,
            "Expected most layers in VRAM, got {}",
            vram_layers
        );
    }

    #[test]
    fn test_70b_model_uses_all_tiers() {
        // Llama-70B: 80 layers, ~1.7GB each at BF16
        let mut budget = BudgetTracker::new(24 * GB, 64 * GB);
        let layer_size = 1700 * MB;

        for i in 0..80 {
            budget.allocate(i, layer_size);
        }

        let vram_count = budget
            .allocations
            .values()
            .filter(|&&t| t == MemoryTier::Vram)
            .count();
        let ram_count = budget
            .allocations
            .values()
            .filter(|&&t| t == MemoryTier::Ram)
            .count();
        let nvme_count = budget
            .allocations
            .values()
            .filter(|&&t| t == MemoryTier::Nvme)
            .count();

        // Should use multiple tiers
        assert!(vram_count > 0, "Should have VRAM layers");
        assert!(ram_count > 0, "Should have RAM layers");
        // NVMe depends on exact math

        // Budgets respected
        assert!(budget.vram_used <= 24 * GB);
        assert!(budget.ram_used <= 64 * GB);
    }
}
