//! RAM cache for warm tensors using pinned memory.
//!
//! Stores layer weights in CPU memory for fast upload to GPU.
//! Uses CUDA pinned (page-locked) memory when available for ~2x faster transfers.

use std::collections::HashMap;

use super::error::TieredError;
use super::lru::LruTracker;
use super::stats::TieredStats;
use std::sync::Arc;

/// CPU-side layer weights stored in RAM.
///
/// Weights are stored as contiguous f16 buffers for efficient GPU upload.
/// Can use pinned (page-locked) memory for faster PCIe transfers.
#[derive(Debug)]
pub struct CpuLayerWeights {
    /// Weight data as contiguous bytes (f16 format).
    pub data: Vec<u8>,

    /// Offsets and sizes for each weight tensor within data.
    pub layout: LayerLayout,

    /// Total size in bytes.
    pub size_bytes: usize,

    /// Whether this buffer uses pinned memory.
    pub is_pinned: bool,
}

/// Layout information for layer weights within a contiguous buffer.
#[derive(Debug, Clone)]
pub struct LayerLayout {
    /// Q projection offset and shape.
    pub q_proj: TensorLayout,
    /// K projection offset and shape.
    pub k_proj: TensorLayout,
    /// V projection offset and shape.
    pub v_proj: TensorLayout,
    /// O projection offset and shape.
    pub o_proj: TensorLayout,
    /// Gate projection offset and shape.
    pub gate_proj: TensorLayout,
    /// Up projection offset and shape.
    pub up_proj: TensorLayout,
    /// Down projection offset and shape.
    pub down_proj: TensorLayout,
    /// Input layernorm offset and shape.
    pub input_norm: TensorLayout,
    /// Post-attention layernorm offset and shape.
    pub post_attn_norm: TensorLayout,
}

/// Layout for a single tensor within the buffer.
#[derive(Debug, Clone, Copy)]
pub struct TensorLayout {
    /// Byte offset from start of buffer.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Shape dimensions.
    pub shape: [usize; 2],
}

impl TensorLayout {
    /// Create a new tensor layout.
    pub fn new(offset: usize, shape: [usize; 2]) -> Self {
        // Assuming f16 (2 bytes per element)
        let size = shape[0] * shape[1] * 2;
        Self {
            offset,
            size,
            shape,
        }
    }

    /// Create layout for a 1D tensor (e.g., layernorm).
    pub fn new_1d(offset: usize, size: usize) -> Self {
        Self {
            offset,
            size: size * 2, // f16
            shape: [size, 1],
        }
    }
}

impl CpuLayerWeights {
    /// Get a slice of the weight data for a specific tensor.
    pub fn tensor_data(&self, layout: &TensorLayout) -> &[u8] {
        &self.data[layout.offset..layout.offset + layout.size]
    }

    /// Get Q projection data.
    pub fn q_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.q_proj)
    }

    /// Get K projection data.
    pub fn k_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.k_proj)
    }

    /// Get V projection data.
    pub fn v_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.v_proj)
    }

    /// Get O projection data.
    pub fn o_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.o_proj)
    }

    /// Get gate projection data.
    pub fn gate_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.gate_proj)
    }

    /// Get up projection data.
    pub fn up_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.up_proj)
    }

    /// Get down projection data.
    pub fn down_proj_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.down_proj)
    }

    /// Get input norm data.
    pub fn input_norm_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.input_norm)
    }

    /// Get post-attention norm data.
    pub fn post_attn_norm_data(&self) -> &[u8] {
        self.tensor_data(&self.layout.post_attn_norm)
    }
}

/// RAM cache for layer weights.
///
/// Stores layers in CPU memory with optional pinned memory for fast GPU transfers.
/// Uses LRU eviction when capacity is exceeded.
pub struct RamCache {
    /// Cached layers (layer_idx -> weights).
    layers: HashMap<usize, CpuLayerWeights>,

    /// LRU tracking for eviction.
    lru: LruTracker<usize>,

    /// Current usage in bytes.
    usage: usize,

    /// Budget in bytes.
    budget: u64,

    /// Whether to use pinned memory.
    use_pinned: bool,

    /// Statistics tracker.
    stats: Arc<TieredStats>,
}

impl RamCache {
    /// Create a new RAM cache.
    ///
    /// # Arguments
    /// * `budget` - Maximum RAM usage in bytes
    /// * `use_pinned` - Whether to allocate pinned (page-locked) memory
    /// * `stats` - Statistics tracker
    pub fn new(budget: u64, use_pinned: bool, stats: Arc<TieredStats>) -> Self {
        Self {
            layers: HashMap::new(),
            lru: LruTracker::new(),
            usage: 0,
            budget,
            use_pinned,
            stats,
        }
    }

    /// Check if a layer is cached.
    pub fn contains(&self, layer_idx: usize) -> bool {
        self.layers.contains_key(&layer_idx)
    }

    /// Get a cached layer without updating LRU.
    pub fn peek(&self, layer_idx: usize) -> Option<&CpuLayerWeights> {
        self.layers.get(&layer_idx)
    }

    /// Get a cached layer, updating LRU tracking.
    pub fn get(&mut self, layer_idx: usize) -> Option<&CpuLayerWeights> {
        if self.layers.contains_key(&layer_idx) {
            self.lru.touch(layer_idx);
            self.stats.record_ram_hit();
            self.layers.get(&layer_idx)
        } else {
            None
        }
    }

    /// Insert a layer into the cache.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `weights` - CPU layer weights
    /// * `priority` - Eviction priority (higher = less likely to evict)
    pub fn insert(
        &mut self,
        layer_idx: usize,
        weights: CpuLayerWeights,
        priority: f32,
    ) -> Result<(), TieredError> {
        let size = weights.size_bytes;

        // Check budget
        if self.usage + size > self.budget as usize {
            return Err(TieredError::ram_alloc(
                format!("layer {} doesn't fit", layer_idx),
                size as u64,
                (self.budget as usize - self.usage) as u64,
            ));
        }

        // Remove old entry if exists
        if let Some(old) = self.layers.remove(&layer_idx) {
            self.usage -= old.size_bytes;
            self.lru.remove(&layer_idx);
        }

        // Insert new entry
        self.layers.insert(layer_idx, weights);
        self.lru.touch_with_priority(layer_idx, priority);
        self.usage += size;

        Ok(())
    }

    /// Remove a layer from the cache.
    pub fn remove(&mut self, layer_idx: usize) -> Option<CpuLayerWeights> {
        if let Some(layer) = self.layers.remove(&layer_idx) {
            self.usage -= layer.size_bytes;
            self.lru.remove(&layer_idx);
            Some(layer)
        } else {
            None
        }
    }

    /// Get current usage in bytes.
    pub fn usage(&self) -> usize {
        self.usage
    }

    /// Get available space in bytes.
    pub fn available(&self) -> usize {
        (self.budget as usize).saturating_sub(self.usage)
    }

    /// Get number of cached layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Check if there's room for a layer of given size.
    pub fn has_room_for(&self, size: usize) -> bool {
        self.usage + size <= self.budget as usize
    }

    /// Evict layers to free at least `bytes_needed`.
    ///
    /// Returns the number of bytes freed.
    pub fn evict_for_space(&mut self, bytes_needed: u64) -> u64 {
        let mut freed = 0u64;

        // Get sizes for eviction calculation
        let sizes: HashMap<usize, usize> = self
            .layers
            .iter()
            .map(|(&idx, layer)| (idx, layer.size_bytes))
            .collect();

        // Get eviction order
        let candidates = self.lru.eviction_candidates_for_size(bytes_needed, |idx| {
            sizes.get(idx).copied().unwrap_or(0) as u64
        });

        for layer_idx in candidates {
            if freed >= bytes_needed {
                break;
            }

            if let Some(layer) = self.layers.remove(&layer_idx) {
                freed += layer.size_bytes as u64;
                self.usage -= layer.size_bytes;
                self.lru.remove(&layer_idx);
                self.stats.record_ram_eviction(layer.size_bytes as u64);

                tracing::debug!(
                    layer_idx,
                    freed_mb = layer.size_bytes / (1024 * 1024),
                    "Evicted layer from RAM"
                );
            }
        }

        freed
    }

    /// Evict the least recently used layer.
    pub fn evict_lru(&mut self) -> Option<(usize, CpuLayerWeights)> {
        if let Some(layer_idx) = self.lru.pop_lru() {
            if let Some(layer) = self.layers.remove(&layer_idx) {
                self.usage -= layer.size_bytes;
                self.stats.record_ram_eviction(layer.size_bytes as u64);
                return Some((layer_idx, layer));
            }
        }
        None
    }

    /// Clear all cached layers.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.lru.clear();
        self.usage = 0;
    }

    /// Get budget.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Check if using pinned memory.
    pub fn uses_pinned_memory(&self) -> bool {
        self.use_pinned
    }

    /// Get iterator over cached layer indices.
    pub fn layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.keys().copied()
    }
}

impl std::fmt::Debug for RamCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RamCache")
            .field("num_layers", &self.layers.len())
            .field("usage_mb", &(self.usage / (1024 * 1024)))
            .field("budget_mb", &(self.budget / (1024 * 1024)))
            .field("use_pinned", &self.use_pinned)
            .finish()
    }
}

/// Allocate pinned memory buffer.
///
/// Pinned memory is page-locked and can be transferred to GPU ~2x faster.
/// Falls back to regular allocation if pinning fails.
#[allow(dead_code)]
pub fn allocate_pinned(size: usize) -> Result<Vec<u8>, TieredError> {
    // For now, just use regular allocation
    // Full implementation would use cuMemAllocHost or similar
    Ok(vec![0u8; size])
}

/// Free pinned memory buffer.
///
/// Must be used for memory allocated with `allocate_pinned`.
#[allow(dead_code)]
pub fn free_pinned(_data: Vec<u8>) {
    // For now, just let it drop
    // Full implementation would use cuMemFreeHost
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> RamCache {
        RamCache::new(1024 * 1024 * 1024, false, Arc::new(TieredStats::new()))
    }

    fn create_dummy_weights(size: usize) -> CpuLayerWeights {
        let layout = LayerLayout {
            q_proj: TensorLayout::new(0, [4096, 4096]),
            k_proj: TensorLayout::new(0, [1024, 4096]),
            v_proj: TensorLayout::new(0, [1024, 4096]),
            o_proj: TensorLayout::new(0, [4096, 4096]),
            gate_proj: TensorLayout::new(0, [11008, 4096]),
            up_proj: TensorLayout::new(0, [11008, 4096]),
            down_proj: TensorLayout::new(0, [4096, 11008]),
            input_norm: TensorLayout::new_1d(0, 4096),
            post_attn_norm: TensorLayout::new_1d(0, 4096),
        };

        CpuLayerWeights {
            data: vec![0u8; size],
            layout,
            size_bytes: size,
            is_pinned: false,
        }
    }

    #[test]
    fn test_ram_cache_basic() {
        let mut cache = create_test_cache();

        let weights = create_dummy_weights(1024);
        cache.insert(0, weights, 0.5).unwrap();

        assert!(cache.contains(0));
        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.usage(), 1024);
    }

    #[test]
    fn test_ram_cache_eviction() {
        let mut cache = RamCache::new(2048, false, Arc::new(TieredStats::new()));

        // Insert two layers
        cache.insert(0, create_dummy_weights(1024), 0.5).unwrap();
        cache.insert(1, create_dummy_weights(1024), 0.3).unwrap(); // Lower priority

        // Evict for space
        let freed = cache.evict_for_space(1024);
        assert_eq!(freed, 1024);

        // Lower priority layer should be evicted
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
    }

    #[test]
    fn test_ram_cache_budget() {
        let mut cache = RamCache::new(1024, false, Arc::new(TieredStats::new()));

        // First insert succeeds
        cache.insert(0, create_dummy_weights(512), 0.5).unwrap();

        // Second insert would exceed budget
        let result = cache.insert(1, create_dummy_weights(1024), 0.5);
        assert!(result.is_err());
    }
}
