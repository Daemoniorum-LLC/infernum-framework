//! GPU memory tracking and pressure monitoring.
//!
//! Tracks VRAM usage across all workloads and signals pressure levels.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Memory pressure levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Plenty of memory available (< 50%).
    Low,
    /// Moderate usage (50-75%).
    Moderate,
    /// High usage (75-90%).
    High,
    /// Critical usage (> 90%).
    Critical,
}

impl MemoryPressure {
    /// Creates pressure level from utilization fraction (0.0 - 1.0).
    pub fn from_utilization(utilization: f32) -> Self {
        if utilization < 0.5 {
            Self::Low
        } else if utilization < 0.75 {
            Self::Moderate
        } else if utilization < 0.9 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Returns the quality reduction factor for this pressure level.
    pub fn quality_factor(self) -> f32 {
        match self {
            Self::Low => 1.0,
            Self::Moderate => 0.9,
            Self::High => 0.7,
            Self::Critical => 0.5,
        }
    }

    /// Returns whether new allocations should be blocked.
    pub fn should_block_new(self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Returns whether background work should pause.
    pub fn should_pause_background(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

/// Statistics about memory usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total allocations made.
    pub total_allocations: u64,
    /// Current allocations.
    pub current_allocations: u64,
    /// Total bytes allocated.
    pub total_allocated: u64,
    /// Total bytes deallocated.
    pub total_deallocated: u64,
    /// Peak usage in bytes.
    pub peak_usage: u64,
    /// Times pressure went critical.
    pub critical_pressure_count: u64,
}

/// Tracks GPU memory usage.
pub struct GpuMemoryTracker {
    /// Total capacity in bytes.
    capacity: u64,
    /// Current used bytes.
    used: AtomicU64,
    /// Peak used bytes.
    peak: AtomicU64,
    /// Total allocations.
    total_allocations: AtomicU64,
    /// Current active allocations.
    current_allocations: AtomicU64,
    /// Total allocated ever.
    total_allocated: AtomicU64,
    /// Total deallocated ever.
    total_deallocated: AtomicU64,
    /// Critical pressure events.
    critical_events: AtomicU64,
}

impl GpuMemoryTracker {
    /// Creates a new tracker with the given capacity.
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            total_allocations: AtomicU64::new(0),
            current_allocations: AtomicU64::new(0),
            total_allocated: AtomicU64::new(0),
            total_deallocated: AtomicU64::new(0),
            critical_events: AtomicU64::new(0),
        }
    }

    /// Returns total capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns current usage in bytes.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Returns available bytes.
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.used())
    }

    /// Returns utilization as fraction (0.0 - 1.0).
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            return 1.0;
        }
        self.used() as f32 / self.capacity as f32
    }

    /// Returns current pressure level.
    pub fn pressure(&self) -> f32 {
        self.utilization()
    }

    /// Returns current pressure as enum.
    pub fn pressure_level(&self) -> MemoryPressure {
        MemoryPressure::from_utilization(self.utilization())
    }

    /// Allocates memory, returning true if successful.
    pub fn try_allocate(&self, bytes: u64) -> bool {
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let new = current + bytes;
            if new > self.capacity {
                return false;
            }
            match self
                .used
                .compare_exchange_weak(current, new, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.total_allocations.fetch_add(1, Ordering::Relaxed);
                    self.current_allocations.fetch_add(1, Ordering::Relaxed);
                    self.total_allocated.fetch_add(bytes, Ordering::Relaxed);

                    // Update peak
                    let mut peak = self.peak.load(Ordering::Relaxed);
                    while new > peak {
                        match self.peak.compare_exchange_weak(
                            peak,
                            new,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(p) => peak = p,
                        }
                    }

                    // Track critical events
                    if MemoryPressure::from_utilization(new as f32 / self.capacity as f32)
                        == MemoryPressure::Critical
                    {
                        self.critical_events.fetch_add(1, Ordering::Relaxed);
                    }

                    return true;
                },
                Err(c) => current = c,
            }
        }
    }

    /// Allocates memory unconditionally.
    pub fn allocate(&self, bytes: u64) {
        self.used.fetch_add(bytes, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.current_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(bytes, Ordering::Relaxed);

        // Update peak
        let new = self.used.load(Ordering::Relaxed);
        let mut peak = self.peak.load(Ordering::Relaxed);
        while new > peak {
            match self
                .peak
                .compare_exchange_weak(peak, new, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Deallocates memory.
    pub fn deallocate(&self, bytes: u64) {
        self.used
            .fetch_sub(bytes.min(self.used()), Ordering::Relaxed);
        self.current_allocations.fetch_sub(1, Ordering::Relaxed);
        self.total_deallocated.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Returns current statistics.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            current_allocations: self.current_allocations.load(Ordering::Relaxed),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_deallocated: self.total_deallocated.load(Ordering::Relaxed),
            peak_usage: self.peak.load(Ordering::Relaxed),
            critical_pressure_count: self.critical_events.load(Ordering::Relaxed),
        }
    }

    /// Resets the tracker (for testing).
    #[cfg(test)]
    pub fn reset(&self) {
        self.used.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.total_allocations.store(0, Ordering::Relaxed);
        self.current_allocations.store(0, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_deallocated.store(0, Ordering::Relaxed);
        self.critical_events.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_levels() {
        assert_eq!(MemoryPressure::from_utilization(0.3), MemoryPressure::Low);
        assert_eq!(
            MemoryPressure::from_utilization(0.6),
            MemoryPressure::Moderate
        );
        assert_eq!(MemoryPressure::from_utilization(0.8), MemoryPressure::High);
        assert_eq!(
            MemoryPressure::from_utilization(0.95),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn test_tracker_allocation() {
        let tracker = GpuMemoryTracker::new(1000);

        assert!(tracker.try_allocate(500));
        assert_eq!(tracker.used(), 500);
        assert_eq!(tracker.available(), 500);

        assert!(tracker.try_allocate(400));
        assert_eq!(tracker.used(), 900);

        // Should fail - not enough space
        assert!(!tracker.try_allocate(200));
        assert_eq!(tracker.used(), 900);
    }

    #[test]
    fn test_tracker_deallocation() {
        let tracker = GpuMemoryTracker::new(1000);

        tracker.allocate(500);
        tracker.deallocate(300);

        assert_eq!(tracker.used(), 200);
        assert_eq!(tracker.available(), 800);
    }

    #[test]
    fn test_utilization() {
        let tracker = GpuMemoryTracker::new(1000);

        tracker.allocate(750);
        assert!((tracker.utilization() - 0.75).abs() < 0.001);
        assert_eq!(tracker.pressure_level(), MemoryPressure::High);
    }

    #[test]
    fn test_peak_tracking() {
        let tracker = GpuMemoryTracker::new(1000);

        tracker.allocate(800);
        tracker.deallocate(500);
        tracker.allocate(200);

        let stats = tracker.stats();
        assert_eq!(stats.peak_usage, 800);
    }
}
