//! KV-Cache management for efficient autoregressive generation.

use std::collections::HashMap;

use infernum_core::RequestId;

/// Configuration for the KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of layers.
    pub num_layers: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Maximum sequence length.
    pub max_seq_len: u32,
    /// Block size for paged attention.
    pub block_size: u32,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            block_size: 16,
        }
    }
}

/// A block of KV cache memory.
#[derive(Debug)]
pub struct CacheBlock {
    /// Block ID.
    pub id: u32,
    /// Reference count.
    pub ref_count: u32,
    /// Whether the block is allocated.
    pub allocated: bool,
}

/// KV-Cache manager with paged attention support.
#[derive(Debug)]
pub struct KVCache {
    config: KVCacheConfig,
    /// Pool of available blocks.
    free_blocks: Vec<u32>,
    /// Mapping from sequence to allocated blocks.
    sequence_blocks: HashMap<RequestId, Vec<u32>>,
    /// Total number of blocks.
    num_blocks: u32,
}

impl KVCache {
    /// Creates a new KV cache with the given configuration.
    #[must_use]
    pub fn new(config: KVCacheConfig) -> Self {
        let num_blocks = config.max_seq_len / config.block_size;
        let free_blocks = (0..num_blocks).collect();

        Self {
            config,
            free_blocks,
            sequence_blocks: HashMap::new(),
            num_blocks,
        }
    }

    /// Allocates blocks for a new sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if there are not enough free blocks.
    pub fn allocate(&mut self, request_id: RequestId, num_tokens: u32) -> Result<(), String> {
        let blocks_needed = (num_tokens + self.config.block_size - 1) / self.config.block_size;

        if blocks_needed as usize > self.free_blocks.len() {
            return Err(format!(
                "Not enough free blocks: need {}, have {}",
                blocks_needed,
                self.free_blocks.len()
            ));
        }

        let allocated: Vec<u32> = self.free_blocks.drain(..blocks_needed as usize).collect();

        self.sequence_blocks.insert(request_id, allocated);
        Ok(())
    }

    /// Extends the allocation for an existing sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence doesn't exist or there are not enough free blocks.
    pub fn extend(&mut self, request_id: &RequestId, additional_tokens: u32) -> Result<(), String> {
        let blocks = self
            .sequence_blocks
            .get_mut(request_id)
            .ok_or_else(|| format!("Sequence {} not found", request_id))?;

        let current_capacity = blocks.len() as u32 * self.config.block_size;
        let current_tokens = current_capacity; // Simplified
        let total_tokens = current_tokens + additional_tokens;
        let total_blocks_needed =
            (total_tokens + self.config.block_size - 1) / self.config.block_size;
        let additional_blocks = total_blocks_needed.saturating_sub(blocks.len() as u32);

        if additional_blocks as usize > self.free_blocks.len() {
            return Err(format!(
                "Not enough free blocks: need {}, have {}",
                additional_blocks,
                self.free_blocks.len()
            ));
        }

        let new_blocks: Vec<u32> = self
            .free_blocks
            .drain(..additional_blocks as usize)
            .collect();
        blocks.extend(new_blocks);

        Ok(())
    }

    /// Frees the blocks allocated to a sequence.
    pub fn free(&mut self, request_id: &RequestId) {
        if let Some(blocks) = self.sequence_blocks.remove(request_id) {
            self.free_blocks.extend(blocks);
        }
    }

    /// Returns the number of free blocks.
    #[must_use]
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Returns the total number of blocks.
    #[must_use]
    pub fn total_block_count(&self) -> u32 {
        self.num_blocks
    }

    /// Returns the utilization as a fraction (0.0-1.0).
    #[must_use]
    pub fn utilization(&self) -> f32 {
        let used = self.num_blocks as usize - self.free_blocks.len();
        used as f32 / self.num_blocks as f32
    }

    /// Returns the blocks allocated to a sequence.
    #[must_use]
    pub fn get_blocks(&self, request_id: &RequestId) -> Option<&[u32]> {
        self.sequence_blocks.get(request_id).map(Vec::as_slice)
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === KVCacheConfig Tests ===

    #[test]
    fn test_config_default() {
        let config = KVCacheConfig::default();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_config_clone() {
        let config = KVCacheConfig::default();
        let cloned = config.clone();
        assert_eq!(config.num_layers, cloned.num_layers);
        assert_eq!(config.block_size, cloned.block_size);
    }

    // === KVCache Creation Tests ===

    #[test]
    fn test_kv_cache_new() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let cache = KVCache::new(config);

        assert_eq!(cache.total_block_count(), 16); // 256 / 16
        assert_eq!(cache.free_block_count(), 16);
    }

    #[test]
    fn test_kv_cache_initial_utilization() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let cache = KVCache::new(config);

        assert!((cache.utilization() - 0.0).abs() < f32::EPSILON);
    }

    // === Allocation Tests ===

    #[test]
    fn test_kv_cache_allocation() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 32).unwrap();

        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 2);

        cache.free(&request_id);
        assert!(cache.get_blocks(&request_id).is_none());
    }

    #[test]
    fn test_allocate_single_block() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 10).unwrap(); // Less than block_size

        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 1);
    }

    #[test]
    fn test_allocate_exact_block_boundary() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 16).unwrap(); // Exactly one block

        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 1);
    }

    #[test]
    fn test_allocate_across_block_boundary() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 17).unwrap(); // Just over one block

        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 2);
    }

    #[test]
    fn test_allocate_insufficient_blocks() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 32, // Only 2 blocks total
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        let result = cache.allocate(request_id, 100); // Needs more than 2 blocks

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not enough free blocks"));
    }

    #[test]
    fn test_allocate_multiple_sequences() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let req1 = RequestId::new();
        let req2 = RequestId::new();

        cache.allocate(req1.clone(), 16).unwrap();
        cache.allocate(req2.clone(), 16).unwrap();

        assert!(cache.get_blocks(&req1).is_some());
        assert!(cache.get_blocks(&req2).is_some());
        assert_eq!(cache.free_block_count(), 14);
    }

    // === Extend Tests ===

    #[test]
    fn test_extend_existing_sequence() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 16).unwrap();
        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 1);

        cache.extend(&request_id, 32).unwrap();
        assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 3);
    }

    #[test]
    fn test_extend_nonexistent_sequence() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        let result = cache.extend(&request_id, 16);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_extend_insufficient_blocks() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 64, // Only 4 blocks
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 48).unwrap(); // Uses 3 blocks

        let result = cache.extend(&request_id, 48); // Needs 3 more, only 1 left
        assert!(result.is_err());
    }

    // === Free Tests ===

    #[test]
    fn test_free_returns_blocks() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let initial_free = cache.free_block_count();

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 32).unwrap();

        assert_eq!(cache.free_block_count(), initial_free - 2);

        cache.free(&request_id);
        assert_eq!(cache.free_block_count(), initial_free);
    }

    #[test]
    fn test_free_nonexistent_sequence() {
        let config = KVCacheConfig::default();
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.free(&request_id); // Should not panic
    }

    // === Utilization Tests ===

    #[test]
    fn test_utilization_after_allocation() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 64, // 4 blocks
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id, 32).unwrap(); // 2 blocks

        assert!((cache.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_full() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 32, // 2 blocks
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id, 32).unwrap(); // 2 blocks

        assert!((cache.utilization() - 1.0).abs() < 0.01);
    }

    // === Get Blocks Tests ===

    #[test]
    fn test_get_blocks_returns_none_for_unknown() {
        let config = KVCacheConfig::default();
        let cache = KVCache::new(config);

        let request_id = RequestId::new();
        assert!(cache.get_blocks(&request_id).is_none());
    }

    #[test]
    fn test_get_blocks_returns_correct_count() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 48).unwrap(); // 3 blocks

        let blocks = cache.get_blocks(&request_id).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    // === Config Accessor Tests ===

    #[test]
    fn test_config_accessor() {
        let config = KVCacheConfig {
            block_size: 32,
            max_seq_len: 512,
            num_layers: 24,
            ..Default::default()
        };
        let cache = KVCache::new(config);

        assert_eq!(cache.config().block_size, 32);
        assert_eq!(cache.config().max_seq_len, 512);
        assert_eq!(cache.config().num_layers, 24);
    }

    // === Block Count Tests ===

    #[test]
    fn test_total_block_count() {
        let config = KVCacheConfig {
            block_size: 32,
            max_seq_len: 128,
            ..Default::default()
        };
        let cache = KVCache::new(config);

        assert_eq!(cache.total_block_count(), 4); // 128 / 32
    }

    #[test]
    fn test_free_block_count_decreases() {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 256,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        let initial = cache.free_block_count();
        let request_id = RequestId::new();
        cache.allocate(request_id, 32).unwrap();

        assert_eq!(cache.free_block_count(), initial - 2);
    }
}
