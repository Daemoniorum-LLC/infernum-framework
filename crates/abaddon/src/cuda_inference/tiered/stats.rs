//! Statistics tracking for tiered memory management.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Statistics for tiered weight store operations.
#[derive(Debug)]
pub struct TieredStats {
    /// Tensor accesses served from VRAM cache.
    pub vram_hits: AtomicU64,

    /// Tensor accesses served from RAM cache.
    pub ram_hits: AtomicU64,

    /// Tensor accesses requiring NVMe load.
    pub nvme_hits: AtomicU64,

    /// Total layers loaded from any source.
    pub layers_loaded: AtomicU64,

    /// Total bytes uploaded to VRAM.
    pub bytes_uploaded: AtomicU64,

    /// Total bytes downloaded from VRAM.
    pub bytes_downloaded: AtomicU64,

    /// Total time spent loading from NVMe (nanoseconds).
    pub nvme_load_time_ns: AtomicU64,

    /// Total time spent uploading to VRAM (nanoseconds).
    pub vram_upload_time_ns: AtomicU64,

    /// Number of VRAM evictions.
    pub vram_evictions: AtomicU64,

    /// Number of RAM evictions.
    pub ram_evictions: AtomicU64,

    /// Bytes evicted from VRAM.
    pub vram_bytes_evicted: AtomicU64,

    /// Bytes evicted from RAM.
    pub ram_bytes_evicted: AtomicU64,

    /// Number of prefetch requests issued.
    pub prefetch_requests: AtomicU64,

    /// Number of prefetch requests that completed before access.
    pub prefetch_hits: AtomicU64,

    /// Start time for throughput calculation.
    start_time: Instant,
}

impl TieredStats {
    /// Create new statistics tracker.
    pub fn new() -> Self {
        Self {
            vram_hits: AtomicU64::new(0),
            ram_hits: AtomicU64::new(0),
            nvme_hits: AtomicU64::new(0),
            layers_loaded: AtomicU64::new(0),
            bytes_uploaded: AtomicU64::new(0),
            bytes_downloaded: AtomicU64::new(0),
            nvme_load_time_ns: AtomicU64::new(0),
            vram_upload_time_ns: AtomicU64::new(0),
            vram_evictions: AtomicU64::new(0),
            ram_evictions: AtomicU64::new(0),
            vram_bytes_evicted: AtomicU64::new(0),
            ram_bytes_evicted: AtomicU64::new(0),
            prefetch_requests: AtomicU64::new(0),
            prefetch_hits: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a VRAM cache hit.
    pub fn record_vram_hit(&self) {
        self.vram_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a RAM cache hit.
    pub fn record_ram_hit(&self) {
        self.ram_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an NVMe load.
    pub fn record_nvme_hit(&self) {
        self.nvme_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a layer load.
    pub fn record_layer_loaded(&self) {
        self.layers_loaded.fetch_add(1, Ordering::Relaxed);
    }

    /// Record bytes uploaded to VRAM.
    pub fn record_vram_upload(&self, bytes: u64, duration_ns: u64) {
        self.bytes_uploaded.fetch_add(bytes, Ordering::Relaxed);
        self.vram_upload_time_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record bytes downloaded from VRAM.
    pub fn record_vram_download(&self, bytes: u64) {
        self.bytes_downloaded.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record NVMe load time.
    pub fn record_nvme_load(&self, duration_ns: u64) {
        self.nvme_load_time_ns
            .fetch_add(duration_ns, Ordering::Relaxed);
    }

    /// Record VRAM eviction.
    pub fn record_vram_eviction(&self, bytes: u64) {
        self.vram_evictions.fetch_add(1, Ordering::Relaxed);
        self.vram_bytes_evicted.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record RAM eviction.
    pub fn record_ram_eviction(&self, bytes: u64) {
        self.ram_evictions.fetch_add(1, Ordering::Relaxed);
        self.ram_bytes_evicted.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a prefetch request.
    pub fn record_prefetch_request(&self) {
        self.prefetch_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prefetch hit (data was ready when accessed).
    pub fn record_prefetch_hit(&self) {
        self.prefetch_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total cache accesses.
    pub fn total_accesses(&self) -> u64 {
        self.vram_hits.load(Ordering::Relaxed)
            + self.ram_hits.load(Ordering::Relaxed)
            + self.nvme_hits.load(Ordering::Relaxed)
    }

    /// Get VRAM hit rate.
    pub fn vram_hit_rate(&self) -> f64 {
        let total = self.total_accesses();
        if total == 0 {
            return 0.0;
        }
        self.vram_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get RAM hit rate (VRAM misses that hit RAM).
    pub fn ram_hit_rate(&self) -> f64 {
        let non_vram =
            self.ram_hits.load(Ordering::Relaxed) + self.nvme_hits.load(Ordering::Relaxed);
        if non_vram == 0 {
            return 0.0;
        }
        self.ram_hits.load(Ordering::Relaxed) as f64 / non_vram as f64
    }

    /// Get prefetch effectiveness (hits / requests).
    pub fn prefetch_effectiveness(&self) -> f64 {
        let requests = self.prefetch_requests.load(Ordering::Relaxed);
        if requests == 0 {
            return 0.0;
        }
        self.prefetch_hits.load(Ordering::Relaxed) as f64 / requests as f64
    }

    /// Get average NVMe load time in milliseconds.
    pub fn avg_nvme_load_ms(&self) -> f64 {
        let hits = self.nvme_hits.load(Ordering::Relaxed);
        if hits == 0 {
            return 0.0;
        }
        let total_ns = self.nvme_load_time_ns.load(Ordering::Relaxed);
        (total_ns as f64 / hits as f64) / 1_000_000.0
    }

    /// Get average VRAM upload time in milliseconds.
    pub fn avg_vram_upload_ms(&self) -> f64 {
        let uploads = self.layers_loaded.load(Ordering::Relaxed);
        if uploads == 0 {
            return 0.0;
        }
        let total_ns = self.vram_upload_time_ns.load(Ordering::Relaxed);
        (total_ns as f64 / uploads as f64) / 1_000_000.0
    }

    /// Get elapsed time since creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get upload bandwidth in GB/s.
    pub fn upload_bandwidth_gbps(&self) -> f64 {
        let total_ns = self.vram_upload_time_ns.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        let bytes = self.bytes_uploaded.load(Ordering::Relaxed);
        let seconds = total_ns as f64 / 1_000_000_000.0;
        (bytes as f64 / seconds) / (1024.0 * 1024.0 * 1024.0)
    }

    /// Create a snapshot of current statistics.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            vram_hits: self.vram_hits.load(Ordering::Relaxed),
            ram_hits: self.ram_hits.load(Ordering::Relaxed),
            nvme_hits: self.nvme_hits.load(Ordering::Relaxed),
            layers_loaded: self.layers_loaded.load(Ordering::Relaxed),
            bytes_uploaded: self.bytes_uploaded.load(Ordering::Relaxed),
            bytes_downloaded: self.bytes_downloaded.load(Ordering::Relaxed),
            vram_evictions: self.vram_evictions.load(Ordering::Relaxed),
            ram_evictions: self.ram_evictions.load(Ordering::Relaxed),
            prefetch_requests: self.prefetch_requests.load(Ordering::Relaxed),
            prefetch_hits: self.prefetch_hits.load(Ordering::Relaxed),
            elapsed_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }
}

impl Default for TieredStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of statistics at a point in time.
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    /// VRAM cache hits.
    pub vram_hits: u64,
    /// RAM cache hits.
    pub ram_hits: u64,
    /// NVMe loads.
    pub nvme_hits: u64,
    /// Total layers loaded.
    pub layers_loaded: u64,
    /// Bytes uploaded to VRAM.
    pub bytes_uploaded: u64,
    /// Bytes downloaded from VRAM.
    pub bytes_downloaded: u64,
    /// VRAM evictions.
    pub vram_evictions: u64,
    /// RAM evictions.
    pub ram_evictions: u64,
    /// Prefetch requests.
    pub prefetch_requests: u64,
    /// Prefetch hits.
    pub prefetch_hits: u64,
    /// Elapsed seconds.
    pub elapsed_secs: f64,
}

impl std::fmt::Display for StatsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.vram_hits + self.ram_hits + self.nvme_hits;
        let vram_rate = if total > 0 {
            self.vram_hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        let ram_rate = if total > 0 {
            self.ram_hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        write!(
            f,
            "TieredStats {{ hits: VRAM={} ({:.1}%), RAM={} ({:.1}%), NVMe={}, \
             evictions: VRAM={}, RAM={}, uploaded: {:.2}GB, elapsed: {:.1}s }}",
            self.vram_hits,
            vram_rate,
            self.ram_hits,
            ram_rate,
            self.nvme_hits,
            self.vram_evictions,
            self.ram_evictions,
            self.bytes_uploaded as f64 / (1024.0 * 1024.0 * 1024.0),
            self.elapsed_secs
        )
    }
}
