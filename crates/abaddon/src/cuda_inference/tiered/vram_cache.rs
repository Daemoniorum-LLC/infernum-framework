//! VRAM cache for hot tensors.
//!
//! Manages GPU-resident layer weights with LRU eviction when VRAM pressure
//! increases (e.g., KV cache growth during long sequences).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::error::TieredError;
use super::lru::LruTracker;
use super::stats::TieredStats;
use crate::adaptive_tiering::AllocationPlan;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::weight_store::{LayerWeights, RMSNormWeights};

/// Shared weights that are always VRAM-resident.
///
/// These tensors are accessed every forward pass and should never be evicted.
pub struct SharedWeights {
    /// Token embeddings [vocab_size, hidden_size].
    pub embed_tokens: GpuTensor,

    /// Final RMS normalization.
    pub final_norm: RMSNormWeights,

    /// LM head projection, or None if tied to embeddings.
    pub lm_head: Option<GpuTensor>,

    /// Total size in bytes.
    size_bytes: usize,
}

impl SharedWeights {
    /// Create shared weights container.
    pub fn new(
        embed_tokens: GpuTensor,
        final_norm: RMSNormWeights,
        lm_head: Option<GpuTensor>,
    ) -> Self {
        let size_bytes = embed_tokens.size_bytes()
            + final_norm.weight.size_bytes()
            + lm_head.as_ref().map(|t| t.size_bytes()).unwrap_or(0);

        Self {
            embed_tokens,
            final_norm,
            lm_head,
            size_bytes,
        }
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

/// VRAM cache for layer weights.
///
/// Manages GPU memory for transformer layer weights with:
/// - LRU eviction based on priority and access time
/// - Never evicts shared weights (embed, lm_head, final_norm)
/// - Tracks memory usage against budget
pub struct VramCache {
    /// CUDA device for memory operations.
    device: Arc<CudaDevice>,

    /// Cached layer weights (layer_idx -> weights).
    layers: HashMap<usize, LayerWeights>,

    /// Shared weights (always resident).
    shared: Option<SharedWeights>,

    /// LRU tracking for eviction ordering.
    lru: LruTracker<usize>,

    /// Current VRAM usage in bytes (layers only, not shared).
    layer_usage: usize,

    /// VRAM budget for layers (excludes shared weights).
    layer_budget: u64,

    /// Size per layer in bytes (for uniform models).
    layer_size: Option<usize>,

    /// Reference to stats tracker.
    stats: Arc<TieredStats>,
}

impl VramCache {
    /// Create a new VRAM cache.
    ///
    /// # Arguments
    /// * `device` - CUDA device for memory operations
    /// * `budget` - Total VRAM budget in bytes (including shared weights)
    /// * `stats` - Statistics tracker
    pub fn new(device: Arc<CudaDevice>, budget: u64, stats: Arc<TieredStats>) -> Self {
        Self {
            device,
            layers: HashMap::new(),
            shared: None,
            lru: LruTracker::new(),
            layer_usage: 0,
            layer_budget: budget,
            layer_size: None,
            stats,
        }
    }

    /// Set shared weights (embed, final_norm, lm_head).
    ///
    /// This reduces the layer budget by the shared weights size.
    pub fn set_shared(&mut self, shared: SharedWeights) {
        // Reduce layer budget by shared weights size
        let shared_size = shared.size_bytes() as u64;
        self.layer_budget = self.layer_budget.saturating_sub(shared_size);

        tracing::info!(
            shared_mb = shared_size / (1024 * 1024),
            layer_budget_mb = self.layer_budget / (1024 * 1024),
            "Set shared weights, adjusted layer budget"
        );

        self.shared = Some(shared);
    }

    /// Get shared weights reference.
    pub fn shared(&self) -> Option<&SharedWeights> {
        self.shared.as_ref()
    }

    /// Get embed_tokens tensor.
    pub fn embed_tokens(&self) -> Option<&GpuTensor> {
        self.shared.as_ref().map(|s| &s.embed_tokens)
    }

    /// Get final_norm weights.
    pub fn final_norm(&self) -> Option<&RMSNormWeights> {
        self.shared.as_ref().map(|s| &s.final_norm)
    }

    /// Get lm_head tensor (or None if tied to embeddings).
    pub fn lm_head(&self) -> Option<&GpuTensor> {
        self.shared.as_ref().and_then(|s| s.lm_head.as_ref())
    }

    /// Check if a layer is cached.
    pub fn contains(&self, layer_idx: usize) -> bool {
        self.layers.contains_key(&layer_idx)
    }

    /// Get a cached layer without updating LRU.
    pub fn peek(&self, layer_idx: usize) -> Option<&LayerWeights> {
        self.layers.get(&layer_idx)
    }

    /// Get a cached layer, updating LRU tracking.
    pub fn get(&mut self, layer_idx: usize) -> Option<&LayerWeights> {
        if self.layers.contains_key(&layer_idx) {
            self.lru.touch(layer_idx);
            self.stats.record_vram_hit();
            self.layers.get(&layer_idx)
        } else {
            None
        }
    }

    /// Insert a layer into the cache.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `layer` - Layer weights to cache
    /// * `priority` - Eviction priority (higher = less likely to evict)
    ///
    /// # Returns
    /// Error if the layer doesn't fit within budget.
    pub fn insert(
        &mut self,
        layer_idx: usize,
        layer: LayerWeights,
        priority: f32,
    ) -> Result<(), TieredError> {
        let size = layer.size_bytes();

        // Check if we have room
        if self.layer_usage as u64 + size as u64 > self.layer_budget {
            return Err(TieredError::vram_alloc(
                format!("layer {} doesn't fit", layer_idx),
                size as u64,
                self.layer_budget - self.layer_usage as u64,
            ));
        }

        // Remove old entry if exists
        if let Some(old) = self.layers.remove(&layer_idx) {
            self.layer_usage -= old.size_bytes();
            self.lru.remove(&layer_idx);
        }

        // Insert new entry
        self.layers.insert(layer_idx, layer);
        self.lru.touch_with_priority(layer_idx, priority);
        self.layer_usage += size;

        // Update layer size estimate
        if self.layer_size.is_none() {
            self.layer_size = Some(size);
        }

        Ok(())
    }

    /// Remove a layer from the cache.
    ///
    /// Returns the removed layer, or None if not cached.
    pub fn remove(&mut self, layer_idx: usize) -> Option<LayerWeights> {
        if let Some(layer) = self.layers.remove(&layer_idx) {
            self.layer_usage -= layer.size_bytes();
            self.lru.remove(&layer_idx);
            Some(layer)
        } else {
            None
        }
    }

    /// Get current layer usage in bytes.
    pub fn layer_usage(&self) -> usize {
        self.layer_usage
    }

    /// Get total usage including shared weights.
    pub fn total_usage(&self) -> usize {
        self.layer_usage + self.shared.as_ref().map(|s| s.size_bytes()).unwrap_or(0)
    }

    /// Get available space for layers.
    pub fn available(&self) -> u64 {
        self.layer_budget.saturating_sub(self.layer_usage as u64)
    }

    /// Get number of cached layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer budget.
    pub fn layer_budget(&self) -> u64 {
        self.layer_budget
    }

    /// Check if there's room for a layer of given size.
    pub fn has_room_for(&self, size: usize) -> bool {
        self.layer_usage as u64 + size as u64 <= self.layer_budget
    }

    /// Evict layers to free at least `bytes_needed`.
    ///
    /// Evicts lowest-priority layers first, respecting the allocation plan.
    /// Returns the layers that were evicted (for demotion to RAM).
    ///
    /// # Arguments
    /// * `bytes_needed` - Minimum bytes to free
    /// * `plan` - Allocation plan for priority lookup
    pub fn evict_for_space(
        &mut self,
        bytes_needed: u64,
        plan: &AllocationPlan,
    ) -> Vec<(usize, LayerWeights)> {
        let mut evicted = Vec::new();
        let mut freed = 0u64;

        // Get eviction candidates
        let layer_sizes: HashMap<usize, usize> = self
            .layers
            .iter()
            .map(|(&idx, layer)| (idx, layer.size_bytes()))
            .collect();

        // Update priorities from plan
        for (&idx, _) in &self.layers {
            let priority = plan.get_layer_priority(idx).unwrap_or(0.5);
            self.lru.touch_with_priority(idx, priority);
        }

        // Get eviction order
        let candidates = self.lru.eviction_candidates_for_size(bytes_needed, |idx| {
            layer_sizes.get(idx).copied().unwrap_or(0) as u64
        });

        for layer_idx in candidates {
            if freed >= bytes_needed {
                break;
            }

            if let Some(layer) = self.layers.remove(&layer_idx) {
                let size = layer.size_bytes();
                self.layer_usage -= size;
                self.lru.remove(&layer_idx);

                self.stats.record_vram_eviction(size as u64);

                tracing::debug!(
                    layer_idx,
                    freed_mb = size / (1024 * 1024),
                    "Evicted layer from VRAM"
                );

                freed += size as u64;
                evicted.push((layer_idx, layer));
            }
        }

        evicted
    }

    /// Evict a specific layer.
    ///
    /// Returns the evicted layer for demotion to RAM.
    pub fn evict(&mut self, layer_idx: usize) -> Option<LayerWeights> {
        if let Some(layer) = self.layers.remove(&layer_idx) {
            let size = layer.size_bytes();
            self.layer_usage -= size;
            self.lru.remove(&layer_idx);
            self.stats.record_vram_eviction(size as u64);

            tracing::debug!(layer_idx, freed_mb = size / (1024 * 1024), "Evicted layer");

            Some(layer)
        } else {
            None
        }
    }

    /// Clear all cached layers (but not shared weights).
    pub fn clear_layers(&mut self) {
        self.layers.clear();
        self.lru.clear();
        self.layer_usage = 0;
    }

    /// Get CUDA device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get iterator over cached layer indices.
    pub fn layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }

    /// Estimate how many layers can fit in remaining space.
    pub fn estimated_capacity(&self) -> usize {
        if let Some(layer_size) = self.layer_size {
            (self.available() as usize) / layer_size
        } else {
            0
        }
    }
}

impl std::fmt::Debug for VramCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramCache")
            .field("num_layers", &self.layers.len())
            .field("layer_usage_mb", &(self.layer_usage / (1024 * 1024)))
            .field("layer_budget_mb", &(self.layer_budget / (1024 * 1024)))
            .field("has_shared", &self.shared.is_some())
            .finish()
    }
}

// Extension trait for AllocationPlan to get layer priority
trait AllocationPlanExt {
    fn get_layer_priority(&self, layer_idx: usize) -> Option<f32>;
}

impl AllocationPlanExt for AllocationPlan {
    fn get_layer_priority(&self, layer_idx: usize) -> Option<f32> {
        // Look up any tensor from this layer to get its priority
        let prefix = format!("model.layers.{layer_idx}.");
        self.allocations
            .iter()
            .find(|(name, _)| name.starts_with(&prefix))
            .map(|(_, alloc)| alloc.priority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require GPU. These are basic unit tests.

    #[test]
    fn test_vram_cache_capacity() {
        // Would need mock GPU device for full testing
        // This tests the logic without actual GPU allocation
    }
}
