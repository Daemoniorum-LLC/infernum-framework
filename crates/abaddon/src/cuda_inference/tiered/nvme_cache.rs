//! NVMe cache for cold tensors.
//!
//! Provides fast disk-based storage for model weights that don't fit in VRAM or RAM.
//! Uses memory-mapped I/O with direct I/O hints for optimal throughput.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::error::TieredError;
use super::stats::TieredStats;

/// Metadata for a cached layer on NVMe.
#[derive(Debug, Clone)]
pub struct NvmeCacheEntry {
    /// Path to the cached file.
    pub path: PathBuf,

    /// Size in bytes.
    pub size_bytes: u64,

    /// Layer index.
    pub layer_idx: usize,

    /// Whether the data is compressed.
    pub compressed: bool,
}

/// NVMe cache for cold layer weights.
///
/// Stores layer weights on fast NVMe storage for models that don't fit in RAM.
/// Uses memory-mapped files or direct I/O for maximum throughput.
pub struct NvmeCache {
    /// Cache directory.
    cache_dir: PathBuf,

    /// Cached layers (layer_idx -> entry).
    entries: HashMap<usize, NvmeCacheEntry>,

    /// Current usage in bytes.
    usage: u64,

    /// Budget in bytes.
    budget: u64,

    /// Statistics tracker.
    stats: Arc<TieredStats>,
}

impl NvmeCache {
    /// Create a new NVMe cache.
    ///
    /// # Arguments
    /// * `cache_dir` - Directory for cached files
    /// * `budget` - Maximum disk usage in bytes
    /// * `stats` - Statistics tracker
    pub fn new(
        cache_dir: impl AsRef<Path>,
        budget: u64,
        stats: Arc<TieredStats>,
    ) -> Result<Self, TieredError> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir).map_err(|e| {
            TieredError::nvme_path(format!("failed to create cache dir: {}", e), &cache_dir)
        })?;

        Ok(Self {
            cache_dir,
            entries: HashMap::new(),
            usage: 0,
            budget,
            stats,
        })
    }

    /// Check if a layer is cached.
    pub fn contains(&self, layer_idx: usize) -> bool {
        self.entries.contains_key(&layer_idx)
    }

    /// Get cache entry metadata.
    pub fn get_entry(&self, layer_idx: usize) -> Option<&NvmeCacheEntry> {
        self.entries.get(&layer_idx)
    }

    /// Read a layer from the cache.
    ///
    /// Returns the raw bytes. Caller is responsible for decompression if needed.
    pub fn read_layer(&self, layer_idx: usize) -> Result<Vec<u8>, TieredError> {
        let entry = self
            .entries
            .get(&layer_idx)
            .ok_or_else(|| TieredError::nvme(format!("layer {} not in NVMe cache", layer_idx)))?;

        let start = std::time::Instant::now();

        let mut file = File::open(&entry.path)
            .map_err(|e| TieredError::nvme_path(format!("failed to open: {}", e), &entry.path))?;

        let mut data = Vec::with_capacity(entry.size_bytes as usize);
        file.read_to_end(&mut data)
            .map_err(|e| TieredError::nvme_path(format!("failed to read: {}", e), &entry.path))?;

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_nvme_load(elapsed_ns);
        self.stats.record_nvme_hit();

        tracing::debug!(
            layer_idx,
            size_mb = entry.size_bytes / (1024 * 1024),
            elapsed_ms = elapsed_ns / 1_000_000,
            "Read layer from NVMe"
        );

        Ok(data)
    }

    /// Write a layer to the cache.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `data` - Raw bytes to cache
    /// * `compressed` - Whether the data is already compressed
    pub fn write_layer(
        &mut self,
        layer_idx: usize,
        data: &[u8],
        compressed: bool,
    ) -> Result<(), TieredError> {
        let size = data.len() as u64;

        // Check budget
        if self.usage + size > self.budget {
            return Err(TieredError::nvme(format!(
                "layer {} ({} MB) exceeds NVMe budget ({} MB available)",
                layer_idx,
                size / (1024 * 1024),
                (self.budget - self.usage) / (1024 * 1024)
            )));
        }

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&layer_idx) {
            self.usage -= old.size_bytes;
            let _ = fs::remove_file(&old.path);
        }

        // Create cache file
        let filename = if compressed {
            format!("layer_{:04}.hct", layer_idx)
        } else {
            format!("layer_{:04}.bin", layer_idx)
        };
        let path = self.cache_dir.join(filename);

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| TieredError::nvme_path(format!("failed to create: {}", e), &path))?;

        file.write_all(data)
            .map_err(|e| TieredError::nvme_path(format!("failed to write: {}", e), &path))?;

        file.sync_all()
            .map_err(|e| TieredError::nvme_path(format!("failed to sync: {}", e), &path))?;

        // Add entry
        self.entries.insert(
            layer_idx,
            NvmeCacheEntry {
                path,
                size_bytes: size,
                layer_idx,
                compressed,
            },
        );
        self.usage += size;

        tracing::debug!(
            layer_idx,
            size_mb = size / (1024 * 1024),
            compressed,
            "Wrote layer to NVMe cache"
        );

        Ok(())
    }

    /// Remove a layer from the cache.
    pub fn remove(&mut self, layer_idx: usize) -> Result<(), TieredError> {
        if let Some(entry) = self.entries.remove(&layer_idx) {
            self.usage -= entry.size_bytes;
            fs::remove_file(&entry.path).map_err(|e| {
                TieredError::nvme_path(format!("failed to remove: {}", e), &entry.path)
            })?;
        }
        Ok(())
    }

    /// Get current usage in bytes.
    pub fn usage(&self) -> u64 {
        self.usage
    }

    /// Get available space in bytes.
    pub fn available(&self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }

    /// Get number of cached layers.
    pub fn num_layers(&self) -> usize {
        self.entries.len()
    }

    /// Get budget in bytes.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Check if there's room for a layer of given size.
    pub fn has_room_for(&self, size: u64) -> bool {
        self.usage + size <= self.budget
    }

    /// Get cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get iterator over cached layer indices.
    pub fn layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.keys().copied()
    }

    /// Clear all cached files.
    pub fn clear(&mut self) -> Result<(), TieredError> {
        for entry in self.entries.values() {
            let _ = fs::remove_file(&entry.path);
        }
        self.entries.clear();
        self.usage = 0;
        Ok(())
    }

    /// Scan cache directory for existing cached layers.
    ///
    /// Useful for resuming from a previous session.
    pub fn scan_existing(&mut self) -> Result<usize, TieredError> {
        let mut found = 0;

        let entries = fs::read_dir(&self.cache_dir).map_err(|e| {
            TieredError::nvme_path(format!("failed to read cache dir: {}", e), &self.cache_dir)
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TieredError::nvme_path(format!("failed to read entry: {}", e), &self.cache_dir)
            })?;

            let path = entry.path();
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Parse layer_NNNN.bin or layer_NNNN.hct
            if let Some(rest) = filename.strip_prefix("layer_") {
                let (num_str, compressed) = if let Some(n) = rest.strip_suffix(".bin") {
                    (n, false)
                } else if let Some(n) = rest.strip_suffix(".hct") {
                    (n, true)
                } else {
                    continue;
                };

                if let Ok(layer_idx) = num_str.parse::<usize>() {
                    let metadata = fs::metadata(&path).map_err(|e| {
                        TieredError::nvme_path(format!("failed to stat: {}", e), &path)
                    })?;

                    let size = metadata.len();
                    self.entries.insert(
                        layer_idx,
                        NvmeCacheEntry {
                            path: path.clone(),
                            size_bytes: size,
                            layer_idx,
                            compressed,
                        },
                    );
                    self.usage += size;
                    found += 1;

                    tracing::debug!(
                        layer_idx,
                        size_mb = size / (1024 * 1024),
                        "Found cached layer"
                    );
                }
            }
        }

        tracing::info!(
            found,
            total_mb = self.usage / (1024 * 1024),
            "Scanned NVMe cache directory"
        );

        Ok(found)
    }
}

impl std::fmt::Debug for NvmeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvmeCache")
            .field("cache_dir", &self.cache_dir)
            .field("num_layers", &self.entries.len())
            .field("usage_mb", &(self.usage / (1024 * 1024)))
            .field("budget_mb", &(self.budget / (1024 * 1024)))
            .finish()
    }
}

/// Memory-mapped layer reader for zero-copy access.
///
/// Uses mmap for large sequential reads, avoiding kernel buffer copies.
#[cfg(feature = "mmap")]
pub struct MmapReader {
    /// Memory-mapped file.
    mmap: memmap2::Mmap,

    /// Entry metadata.
    entry: NvmeCacheEntry,
}

#[cfg(feature = "mmap")]
impl MmapReader {
    /// Create a new memory-mapped reader.
    pub fn new(entry: &NvmeCacheEntry) -> Result<Self, TieredError> {
        let file = File::open(&entry.path)
            .map_err(|e| TieredError::nvme_path(format!("failed to open: {}", e), &entry.path))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                TieredError::nvme_path(format!("failed to mmap: {}", e), &entry.path)
            })?
        };

        Ok(Self {
            mmap,
            entry: entry.clone(),
        })
    }

    /// Get the mapped data.
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get entry metadata.
    pub fn entry(&self) -> &NvmeCacheEntry {
        &self.entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_cache() -> (NvmeCache, TempDir) {
        let tmp = TempDir::new().unwrap();
        let cache = NvmeCache::new(
            tmp.path(),
            1024 * 1024 * 1024, // 1GB
            Arc::new(TieredStats::new()),
        )
        .unwrap();
        (cache, tmp)
    }

    #[test]
    fn test_nvme_cache_basic() {
        let (mut cache, _tmp) = create_test_cache();

        let data = vec![0u8; 1024];
        cache.write_layer(0, &data, false).unwrap();

        assert!(cache.contains(0));
        assert_eq!(cache.num_layers(), 1);
        assert_eq!(cache.usage(), 1024);

        let read_data = cache.read_layer(0).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_nvme_cache_remove() {
        let (mut cache, _tmp) = create_test_cache();

        let data = vec![0u8; 1024];
        cache.write_layer(0, &data, false).unwrap();
        assert!(cache.contains(0));

        cache.remove(0).unwrap();
        assert!(!cache.contains(0));
        assert_eq!(cache.usage(), 0);
    }

    #[test]
    fn test_nvme_cache_budget() {
        let tmp = TempDir::new().unwrap();
        let mut cache = NvmeCache::new(
            tmp.path(),
            1024, // 1KB budget
            Arc::new(TieredStats::new()),
        )
        .unwrap();

        let data = vec![0u8; 2048]; // 2KB - exceeds budget
        let result = cache.write_layer(0, &data, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_nvme_cache_scan() {
        let tmp = TempDir::new().unwrap();

        // Create some cache files manually
        let path1 = tmp.path().join("layer_0000.bin");
        let path2 = tmp.path().join("layer_0005.hct");
        fs::write(&path1, vec![0u8; 1024]).unwrap();
        fs::write(&path2, vec![0u8; 2048]).unwrap();

        let mut cache =
            NvmeCache::new(tmp.path(), 1024 * 1024 * 1024, Arc::new(TieredStats::new())).unwrap();

        let found = cache.scan_existing().unwrap();
        assert_eq!(found, 2);
        assert!(cache.contains(0));
        assert!(cache.contains(5));
        assert_eq!(cache.usage(), 1024 + 2048);

        // Check compression flags
        assert!(!cache.get_entry(0).unwrap().compressed);
        assert!(cache.get_entry(5).unwrap().compressed);
    }
}
