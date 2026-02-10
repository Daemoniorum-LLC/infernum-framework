//! Fragment cache for HoloTensor weights.
//!
//! Unified cache shared between Infernum and Dantalion for efficient
//! fragment reuse across workloads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// VRAM cache capacity in bytes.
    pub vram_capacity: u64,
    /// RAM cache capacity in bytes.
    pub ram_capacity: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            vram_capacity: 10 * 1024 * 1024 * 1024, // 10GB
            ram_capacity: 32 * 1024 * 1024 * 1024,  // 32GB
        }
    }
}

/// Cache tier for fragments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTier {
    /// GPU VRAM - fastest.
    Vram,
    /// System RAM - fast.
    Ram,
    /// Not cached.
    None,
}

/// Statistics for the fragment cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// VRAM bytes used.
    pub vram_used: u64,
    /// RAM bytes used.
    pub ram_used: u64,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Evictions from VRAM.
    pub vram_evictions: u64,
    /// Evictions from RAM.
    pub ram_evictions: u64,
    /// Total fragments cached.
    pub fragments_cached: u64,
}

impl CacheStats {
    /// Returns cache hit rate (0.0 - 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Returns VRAM utilization.
    pub fn vram_utilization(&self, capacity: u64) -> f64 {
        if capacity == 0 {
            return 0.0;
        }
        self.vram_used as f64 / capacity as f64
    }

    /// Returns RAM utilization.
    pub fn ram_utilization(&self, capacity: u64) -> f64 {
        if capacity == 0 {
            return 0.0;
        }
        self.ram_used as f64 / capacity as f64
    }
}

/// A cached fragment entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Fragment identifier (stored for diagnostics; keyed in HashMap).
    _fragment_id: String,
    /// Size in bytes.
    size: u64,
    /// Current tier.
    tier: CacheTier,
    /// Last access time.
    last_access: Instant,
    /// Access count.
    access_count: u64,
    /// Which systems use this fragment.
    users: FragmentUsers,
}

/// Which systems use a fragment.
#[derive(Debug, Clone, Copy, Default)]
struct FragmentUsers {
    infernum: bool,
    dantalion: bool,
}

impl FragmentUsers {
    fn count(&self) -> u32 {
        self.infernum as u32 + self.dantalion as u32
    }
}

/// The unified fragment cache.
pub struct FragmentCache {
    config: CacheConfig,
    entries: RwLock<HashMap<String, CacheEntry>>,
    vram_used: AtomicU64,
    ram_used: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    vram_evictions: AtomicU64,
    ram_evictions: AtomicU64,
}

impl FragmentCache {
    /// Creates a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            vram_used: AtomicU64::new(0),
            ram_used: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            vram_evictions: AtomicU64::new(0),
            ram_evictions: AtomicU64::new(0),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Checks if a fragment is cached.
    pub fn contains(&self, fragment_id: &str) -> bool {
        self.entries.read().contains_key(fragment_id)
    }

    /// Gets the tier for a fragment.
    pub fn get_tier(&self, fragment_id: &str) -> CacheTier {
        self.entries
            .read()
            .get(fragment_id)
            .map(|e| e.tier)
            .unwrap_or(CacheTier::None)
    }

    /// Records a cache access, returning the tier.
    pub fn access(&self, fragment_id: &str) -> CacheTier {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(fragment_id) {
            entry.last_access = Instant::now();
            entry.access_count += 1;
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.tier
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            CacheTier::None
        }
    }

    /// Inserts a fragment into the cache.
    pub fn insert(
        &self,
        fragment_id: impl Into<String>,
        size: u64,
        tier: CacheTier,
        for_infernum: bool,
    ) {
        let fragment_id = fragment_id.into();

        // Evict if necessary
        self.ensure_capacity(size, tier);

        let entry = CacheEntry {
            _fragment_id: fragment_id.clone(),
            size,
            tier,
            last_access: Instant::now(),
            access_count: 1,
            users: FragmentUsers {
                infernum: for_infernum,
                dantalion: !for_infernum,
            },
        };

        // Update usage tracking
        match tier {
            CacheTier::Vram => {
                self.vram_used.fetch_add(size, Ordering::Relaxed);
            },
            CacheTier::Ram => {
                self.ram_used.fetch_add(size, Ordering::Relaxed);
            },
            CacheTier::None => {},
        }

        self.entries.write().insert(fragment_id, entry);
    }

    /// Removes a fragment from the cache.
    pub fn remove(&self, fragment_id: &str) {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.remove(fragment_id) {
            match entry.tier {
                CacheTier::Vram => {
                    self.vram_used.fetch_sub(entry.size, Ordering::Relaxed);
                },
                CacheTier::Ram => {
                    self.ram_used.fetch_sub(entry.size, Ordering::Relaxed);
                },
                CacheTier::None => {},
            }
        }
    }

    /// Promotes a fragment to a higher tier.
    pub fn promote(&self, fragment_id: &str, to_tier: CacheTier) {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(fragment_id) {
            let from_tier = entry.tier;
            if to_tier == from_tier {
                return;
            }

            // Update usage
            match from_tier {
                CacheTier::Vram => {
                    self.vram_used.fetch_sub(entry.size, Ordering::Relaxed);
                },
                CacheTier::Ram => {
                    self.ram_used.fetch_sub(entry.size, Ordering::Relaxed);
                },
                CacheTier::None => {},
            }

            match to_tier {
                CacheTier::Vram => {
                    self.vram_used.fetch_add(entry.size, Ordering::Relaxed);
                },
                CacheTier::Ram => {
                    self.ram_used.fetch_add(entry.size, Ordering::Relaxed);
                },
                CacheTier::None => {},
            }

            entry.tier = to_tier;
        }
    }

    /// Demotes a fragment to a lower tier.
    pub fn demote(&self, fragment_id: &str, to_tier: CacheTier) {
        self.promote(fragment_id, to_tier);
    }

    /// Marks a fragment as used by both systems (shared).
    pub fn mark_shared(&self, fragment_id: &str) {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(fragment_id) {
            entry.users.infernum = true;
            entry.users.dantalion = true;
        }
    }

    /// Returns current statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            vram_used: self.vram_used.load(Ordering::Relaxed),
            ram_used: self.ram_used.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            vram_evictions: self.vram_evictions.load(Ordering::Relaxed),
            ram_evictions: self.ram_evictions.load(Ordering::Relaxed),
            fragments_cached: self.entries.read().len() as u64,
        }
    }

    /// Returns VRAM used.
    pub fn vram_used(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    /// Returns RAM used.
    pub fn ram_used(&self) -> u64 {
        self.ram_used.load(Ordering::Relaxed)
    }

    /// Clears all cached fragments.
    pub fn clear(&self) {
        self.entries.write().clear();
        self.vram_used.store(0, Ordering::Relaxed);
        self.ram_used.store(0, Ordering::Relaxed);
    }

    /// Ensures capacity for a new entry, evicting if necessary.
    fn ensure_capacity(&self, size: u64, tier: CacheTier) {
        let (capacity, used) = match tier {
            CacheTier::Vram => (
                self.config.vram_capacity,
                self.vram_used.load(Ordering::Relaxed),
            ),
            CacheTier::Ram => (
                self.config.ram_capacity,
                self.ram_used.load(Ordering::Relaxed),
            ),
            CacheTier::None => return,
        };

        if used + size <= capacity {
            return;
        }

        // Need to evict - use LRU
        let needed = used + size - capacity;
        self.evict_lru(tier, needed);
    }

    /// Evicts least recently used entries.
    fn evict_lru(&self, tier: CacheTier, needed: u64) {
        let mut entries = self.entries.write();
        let mut candidates: Vec<_> = entries
            .iter()
            .filter(|(_, e)| e.tier == tier)
            .map(|(id, e)| (id.clone(), e.last_access, e.size, e.users.count()))
            .collect();

        // Sort by: shared count (evict non-shared first), then access time
        candidates.sort_by(|a, b| a.3.cmp(&b.3).then(a.1.cmp(&b.1)));

        let mut freed = 0u64;
        for (id, _, _size, _) in candidates {
            if freed >= needed {
                break;
            }

            if let Some(entry) = entries.remove(&id) {
                freed += entry.size;
                match tier {
                    CacheTier::Vram => {
                        self.vram_used.fetch_sub(entry.size, Ordering::Relaxed);
                        self.vram_evictions.fetch_add(1, Ordering::Relaxed);
                    },
                    CacheTier::Ram => {
                        self.ram_used.fetch_sub(entry.size, Ordering::Relaxed);
                        self.ram_evictions.fetch_add(1, Ordering::Relaxed);
                    },
                    CacheTier::None => {},
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_access() {
        let cache = FragmentCache::new(CacheConfig {
            vram_capacity: 1000,
            ram_capacity: 1000,
        });

        cache.insert("frag1", 100, CacheTier::Vram, true);
        assert!(cache.contains("frag1"));
        assert_eq!(cache.get_tier("frag1"), CacheTier::Vram);

        let tier = cache.access("frag1");
        assert_eq!(tier, CacheTier::Vram);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.vram_used, 100);
    }

    #[test]
    fn test_cache_miss() {
        let cache = FragmentCache::new(CacheConfig::default());

        let tier = cache.access("nonexistent");
        assert_eq!(tier, CacheTier::None);

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = FragmentCache::new(CacheConfig {
            vram_capacity: 200,
            ram_capacity: 1000,
        });

        cache.insert("frag1", 100, CacheTier::Vram, true);
        cache.insert("frag2", 100, CacheTier::Vram, true);

        // This should trigger eviction
        cache.insert("frag3", 100, CacheTier::Vram, true);

        let stats = cache.stats();
        assert!(stats.vram_evictions >= 1);
        assert!(stats.vram_used <= 200);
    }

    #[test]
    fn test_cache_promote_demote() {
        let cache = FragmentCache::new(CacheConfig {
            vram_capacity: 1000,
            ram_capacity: 1000,
        });

        cache.insert("frag1", 100, CacheTier::Ram, true);
        assert_eq!(cache.ram_used(), 100);
        assert_eq!(cache.vram_used(), 0);

        cache.promote("frag1", CacheTier::Vram);
        assert_eq!(cache.ram_used(), 0);
        assert_eq!(cache.vram_used(), 100);
        assert_eq!(cache.get_tier("frag1"), CacheTier::Vram);
    }

    #[test]
    fn test_hit_rate() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };

        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }
}
