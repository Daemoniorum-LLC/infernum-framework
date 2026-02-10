//! Hybrid VRAM/RAM Memory Manager for Holotensor Inference
//!
//! Manages fragments across memory tiers with async streaming and LRU eviction.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use super::{HoloInferenceError, Result};

/// Memory tier for fragment storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// GPU VRAM (hot - used for active inference).
    Vram,
    /// System RAM (warm - can be streamed to VRAM).
    Ram,
    /// Disk storage (cold - needs loading to RAM first).
    Disk,
}

impl MemoryTier {
    /// Get tier priority (higher = faster access).
    pub fn priority(&self) -> u8 {
        match self {
            MemoryTier::Vram => 3,
            MemoryTier::Ram => 2,
            MemoryTier::Disk => 1,
        }
    }
}

/// Location of a fragment in memory.
#[derive(Debug, Clone)]
pub enum FragmentLocation {
    /// Fragment in VRAM with GPU pointer.
    Vram {
        /// GPU memory offset.
        offset: usize,
        /// Size in bytes.
        size: usize,
        /// CUDA device ID.
        device_id: usize,
    },
    /// Fragment in RAM (possibly pinned for DMA).
    Ram {
        /// RAM pointer (as usize for thread safety).
        ptr: usize,
        /// Size in bytes.
        size: usize,
        /// Whether memory is pinned for GPU DMA.
        pinned: bool,
        /// NUMA node (-1 if unknown).
        numa_node: i32,
    },
    /// Fragment on disk.
    Disk {
        /// File path.
        path: String,
        /// Offset in file.
        offset: u64,
        /// Size in bytes.
        size: usize,
    },
}

impl FragmentLocation {
    /// Get memory tier for this location.
    pub fn tier(&self) -> MemoryTier {
        match self {
            FragmentLocation::Vram { .. } => MemoryTier::Vram,
            FragmentLocation::Ram { .. } => MemoryTier::Ram,
            FragmentLocation::Disk { .. } => MemoryTier::Disk,
        }
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        match self {
            FragmentLocation::Vram { size, .. } => *size,
            FragmentLocation::Ram { size, .. } => *size,
            FragmentLocation::Disk { size, .. } => *size,
        }
    }
}

/// Unique identifier for a fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentId {
    /// Layer index.
    pub layer: usize,
    /// Weight type (q_proj=0, k_proj=1, etc.).
    pub weight_type: u8,
    /// Fragment index within tensor.
    pub fragment_index: u16,
}

impl FragmentId {
    /// Create new fragment ID.
    pub fn new(layer: usize, weight_type: u8, fragment_index: u16) -> Self {
        Self {
            layer,
            weight_type,
            fragment_index,
        }
    }
}

/// Configuration for memory manager.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum VRAM budget in bytes.
    pub vram_budget: usize,
    /// Maximum RAM budget in bytes.
    pub ram_budget: usize,
    /// Preferred NUMA node (-1 for auto).
    pub numa_node: i32,
    /// Use pinned memory for RAM allocations.
    pub use_pinned_memory: bool,
    /// LRU eviction threshold (0.0-1.0).
    pub eviction_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            vram_budget: 20 * 1024 * 1024 * 1024, // 20GB
            ram_budget: 64 * 1024 * 1024 * 1024,  // 64GB
            numa_node: -1,
            use_pinned_memory: true,
            eviction_threshold: 0.9,
        }
    }
}

/// LRU entry with access timestamp.
#[derive(Debug, Clone)]
struct LruEntry {
    fragment_id: FragmentId,
    last_access: u64,
    size: usize,
}

/// Hybrid VRAM/RAM memory manager.
///
/// Manages fragment placement across memory tiers with:
/// - LRU eviction when VRAM is full
/// - Async streaming from RAM to VRAM
/// - NUMA-aware RAM allocation
/// - Pinned memory for efficient DMA
pub struct HoloMemoryManager {
    config: MemoryConfig,

    /// Fragment locations.
    locations: RwLock<HashMap<FragmentId, FragmentLocation>>,

    /// LRU tracking for VRAM.
    vram_lru: Mutex<Vec<LruEntry>>,

    /// Current VRAM usage.
    vram_used: AtomicUsize,

    /// Current RAM usage.
    ram_used: AtomicUsize,

    /// Access counter for LRU timestamps.
    access_counter: AtomicU64,

    /// Statistics.
    stats: RwLock<MemoryStats>,
}

/// Memory usage statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Fragments in VRAM.
    pub vram_fragments: usize,
    /// Fragments in RAM.
    pub ram_fragments: usize,
    /// Fragments on disk.
    pub disk_fragments: usize,
    /// Total evictions from VRAM.
    pub evictions: usize,
    /// Total promotions to VRAM.
    pub promotions: usize,
    /// Bytes transferred RAM→VRAM.
    pub bytes_promoted: usize,
    /// Bytes transferred VRAM→RAM.
    pub bytes_evicted: usize,
}

impl HoloMemoryManager {
    /// Create new memory manager with given configuration.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            locations: RwLock::new(HashMap::new()),
            vram_lru: Mutex::new(Vec::new()),
            vram_used: AtomicUsize::new(0),
            ram_used: AtomicUsize::new(0),
            access_counter: AtomicU64::new(0),
            stats: RwLock::new(MemoryStats::default()),
        }
    }

    /// Get current VRAM usage.
    pub fn vram_used(&self) -> usize {
        self.vram_used.load(Ordering::Relaxed)
    }

    /// Get current RAM usage.
    pub fn ram_used(&self) -> usize {
        self.ram_used.load(Ordering::Relaxed)
    }

    /// Get available VRAM.
    pub fn vram_available(&self) -> usize {
        self.config.vram_budget.saturating_sub(self.vram_used())
    }

    /// Get available RAM.
    pub fn ram_available(&self) -> usize {
        self.config.ram_budget.saturating_sub(self.ram_used())
    }

    /// Get fragment location.
    pub fn get_location(&self, id: &FragmentId) -> Option<FragmentLocation> {
        let locations = self.locations.read().ok()?;
        locations.get(id).cloned()
    }

    /// Get memory tier for fragment.
    pub fn get_tier(&self, id: &FragmentId) -> Option<MemoryTier> {
        self.get_location(id).map(|loc| loc.tier())
    }

    /// Check if fragment is in VRAM.
    pub fn is_in_vram(&self, id: &FragmentId) -> bool {
        self.get_tier(id) == Some(MemoryTier::Vram)
    }

    /// Register fragment location.
    pub fn register_fragment(&self, id: FragmentId, location: FragmentLocation) -> Result<()> {
        let size = location.size();
        let tier = location.tier();

        // Update usage counters
        match tier {
            MemoryTier::Vram => {
                self.vram_used.fetch_add(size, Ordering::Relaxed);
                self.add_to_lru(id, size);
            },
            MemoryTier::Ram => {
                self.ram_used.fetch_add(size, Ordering::Relaxed);
            },
            MemoryTier::Disk => {},
        }

        // Store location
        let mut locations =
            self.locations
                .write()
                .map_err(|e| HoloInferenceError::MemoryAlloc {
                    tier,
                    message: format!("lock poisoned: {}", e),
                })?;
        locations.insert(id, location);

        // Update stats
        let mut stats = self
            .stats
            .write()
            .map_err(|_| HoloInferenceError::MemoryAlloc {
                tier,
                message: "stats lock poisoned".to_string(),
            })?;
        match tier {
            MemoryTier::Vram => stats.vram_fragments += 1,
            MemoryTier::Ram => stats.ram_fragments += 1,
            MemoryTier::Disk => stats.disk_fragments += 1,
        }

        Ok(())
    }

    /// Update fragment access (for LRU tracking).
    pub fn touch(&self, id: &FragmentId) {
        let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut lru) = self.vram_lru.lock() {
            if let Some(entry) = lru.iter_mut().find(|e| e.fragment_id == *id) {
                entry.last_access = timestamp;
            }
        }
    }

    /// Check if we need to evict from VRAM.
    pub fn needs_eviction(&self) -> bool {
        let used = self.vram_used() as f32;
        let budget = self.config.vram_budget as f32;
        used / budget >= self.config.eviction_threshold
    }

    /// Get fragments to evict to make room for given size.
    pub fn get_eviction_candidates(&self, needed_size: usize) -> Vec<FragmentId> {
        let mut candidates = Vec::new();
        let mut freed = 0usize;

        if let Ok(lru) = self.vram_lru.lock() {
            // Sort by last access (oldest first)
            let mut sorted: Vec<_> = lru.iter().cloned().collect();
            sorted.sort_by_key(|e| e.last_access);

            for entry in sorted {
                if freed >= needed_size {
                    break;
                }
                candidates.push(entry.fragment_id);
                freed += entry.size;
            }
        }

        candidates
    }

    /// Evict fragment from VRAM to RAM.
    ///
    /// Returns the new RAM location.
    pub fn evict_to_ram(&self, id: &FragmentId) -> Result<FragmentLocation> {
        let old_location =
            self.get_location(id)
                .ok_or_else(|| HoloInferenceError::FragmentNotFound {
                    layer: id.layer,
                    fragment_index: id.fragment_index,
                })?;

        let size = old_location.size();

        // Check RAM availability
        if self.ram_available() < size {
            return Err(HoloInferenceError::InsufficientMemory {
                tier: MemoryTier::Ram,
                required: size,
                available: self.ram_available(),
            });
        }

        // Create RAM location (actual allocation would happen in CUDA code)
        let new_location = FragmentLocation::Ram {
            ptr: 0, // Placeholder - actual allocation happens externally
            size,
            pinned: self.config.use_pinned_memory,
            numa_node: self.config.numa_node,
        };

        // Update counters
        self.vram_used.fetch_sub(size, Ordering::Relaxed);
        self.ram_used.fetch_add(size, Ordering::Relaxed);

        // Remove from LRU
        self.remove_from_lru(id);

        // Update location
        if let Ok(mut locations) = self.locations.write() {
            locations.insert(*id, new_location.clone());
        }

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.vram_fragments = stats.vram_fragments.saturating_sub(1);
            stats.ram_fragments += 1;
            stats.evictions += 1;
            stats.bytes_evicted += size;
        }

        Ok(new_location)
    }

    /// Promote fragment from RAM to VRAM.
    ///
    /// May trigger eviction of other fragments.
    pub fn promote_to_vram(&self, id: &FragmentId) -> Result<FragmentLocation> {
        let old_location =
            self.get_location(id)
                .ok_or_else(|| HoloInferenceError::FragmentNotFound {
                    layer: id.layer,
                    fragment_index: id.fragment_index,
                })?;

        let size = old_location.size();

        // Check VRAM availability, evict if needed
        while self.vram_available() < size {
            let candidates = self.get_eviction_candidates(size - self.vram_available());
            if candidates.is_empty() {
                return Err(HoloInferenceError::InsufficientMemory {
                    tier: MemoryTier::Vram,
                    required: size,
                    available: self.vram_available(),
                });
            }

            for candidate in candidates {
                self.evict_to_ram(&candidate)?;
            }
        }

        // Create VRAM location
        let new_location = FragmentLocation::Vram {
            offset: 0, // Placeholder
            size,
            device_id: 0,
        };

        // Update counters
        self.ram_used.fetch_sub(size, Ordering::Relaxed);
        self.vram_used.fetch_add(size, Ordering::Relaxed);

        // Add to LRU
        self.add_to_lru(*id, size);

        // Update location
        if let Ok(mut locations) = self.locations.write() {
            locations.insert(*id, new_location.clone());
        }

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.ram_fragments = stats.ram_fragments.saturating_sub(1);
            stats.vram_fragments += 1;
            stats.promotions += 1;
            stats.bytes_promoted += size;
        }

        Ok(new_location)
    }

    /// Add fragment to LRU tracking.
    fn add_to_lru(&self, id: FragmentId, size: usize) {
        let timestamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut lru) = self.vram_lru.lock() {
            lru.push(LruEntry {
                fragment_id: id,
                last_access: timestamp,
                size,
            });
        }
    }

    /// Remove fragment from LRU tracking.
    fn remove_from_lru(&self, id: &FragmentId) {
        if let Ok(mut lru) = self.vram_lru.lock() {
            lru.retain(|e| e.fragment_id != *id);
        }
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Calculate optimal fragment distribution for given model size.
    ///
    /// Returns (vram_fragments, ram_fragments) counts.
    pub fn optimal_distribution(
        &self,
        total_fragments: usize,
        fragment_size: usize,
        _min_quality: f32,
    ) -> (usize, usize) {
        // Calculate how many fragments can fit in VRAM
        let max_vram_frags = self.config.vram_budget / fragment_size;

        // VRAM gets the hot fragments (up to budget)
        let vram_frags = max_vram_frags.min(total_fragments);

        // RAM gets the rest
        let ram_frags = total_fragments.saturating_sub(vram_frags);

        (vram_frags, ram_frags)
    }

    /// Get fragments sorted by priority for a layer.
    ///
    /// Higher priority fragments should be in VRAM.
    pub fn get_priority_order(&self, layer: usize, num_fragments: u16) -> Vec<FragmentId> {
        // For LRDF: lower fragment indices have higher singular values
        // So fragment 0 is most important, fragment N-1 is least
        (0..num_fragments)
            .map(|i| FragmentId::new(layer, 0, i))
            .collect()
    }

    /// Clear all allocations.
    pub fn clear(&self) {
        if let Ok(mut locations) = self.locations.write() {
            locations.clear();
        }
        if let Ok(mut lru) = self.vram_lru.lock() {
            lru.clear();
        }
        self.vram_used.store(0, Ordering::Relaxed);
        self.ram_used.store(0, Ordering::Relaxed);
        if let Ok(mut stats) = self.stats.write() {
            *stats = MemoryStats::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert_eq!(config.vram_budget, 20 * 1024 * 1024 * 1024);
        assert_eq!(config.ram_budget, 64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_fragment_registration() {
        let manager = HoloMemoryManager::new(MemoryConfig::default());

        let id = FragmentId::new(0, 0, 0);
        let location = FragmentLocation::Vram {
            offset: 0,
            size: 1024,
            device_id: 0,
        };

        manager.register_fragment(id, location).unwrap();

        assert!(manager.is_in_vram(&id));
        assert_eq!(manager.vram_used(), 1024);
    }

    #[test]
    fn test_lru_tracking() {
        let manager = HoloMemoryManager::new(MemoryConfig::default());

        // Register multiple fragments
        for i in 0..5 {
            let id = FragmentId::new(0, 0, i);
            let location = FragmentLocation::Vram {
                offset: i as usize * 1024,
                size: 1024,
                device_id: 0,
            };
            manager.register_fragment(id, location).unwrap();
        }

        // Touch fragment 2 (make it recently used)
        manager.touch(&FragmentId::new(0, 0, 2));

        // Get eviction candidates - should not include fragment 2
        let candidates = manager.get_eviction_candidates(1024);
        assert!(!candidates.contains(&FragmentId::new(0, 0, 2)));
    }

    #[test]
    fn test_optimal_distribution() {
        let config = MemoryConfig {
            vram_budget: 10 * 1024 * 1024, // 10MB
            ram_budget: 100 * 1024 * 1024, // 100MB
            ..Default::default()
        };
        let manager = HoloMemoryManager::new(config);

        // 32 fragments of 1MB each = 32MB total
        let (vram, ram) = manager.optimal_distribution(32, 1024 * 1024, 0.7);

        assert_eq!(vram, 10); // 10MB budget / 1MB per fragment
        assert_eq!(ram, 22); // Remaining 22 fragments
    }

    #[test]
    fn test_tier_priority() {
        assert!(MemoryTier::Vram.priority() > MemoryTier::Ram.priority());
        assert!(MemoryTier::Ram.priority() > MemoryTier::Disk.priority());
    }
}
