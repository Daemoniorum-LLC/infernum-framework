//! Performance optimization utilities.
//!
//! This module provides utilities for memory management, object pooling,
//! and performance monitoring.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Memory-efficient string buffer with pre-allocation.
///
/// Reduces allocations for frequently constructed strings by maintaining
/// a pool of pre-allocated buffers.
///
/// # Example
///
/// ```
/// use infernum_core::perf::StringPool;
///
/// let pool = StringPool::new(16, 4096);
/// let mut buffer = pool.acquire();
/// buffer.push_str("Hello, world!");
/// pool.release(buffer);
/// ```
pub struct StringPool {
    pool: Mutex<VecDeque<String>>,
    buffer_capacity: usize,
    max_pool_size: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl StringPool {
    /// Creates a new string pool.
    ///
    /// # Arguments
    ///
    /// * `max_pool_size` - Maximum number of buffers to keep in the pool
    /// * `buffer_capacity` - Initial capacity for each buffer
    pub fn new(max_pool_size: usize, buffer_capacity: usize) -> Self {
        Self {
            pool: Mutex::new(VecDeque::with_capacity(max_pool_size)),
            buffer_capacity,
            max_pool_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Acquires a string buffer from the pool.
    ///
    /// Returns a pooled buffer if available, otherwise creates a new one.
    pub fn acquire(&self) -> String {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(mut buffer) = pool.pop_front() {
                buffer.clear();
                self.hits.fetch_add(1, Ordering::Relaxed);
                return buffer;
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        String::with_capacity(self.buffer_capacity)
    }

    /// Releases a string buffer back to the pool.
    ///
    /// The buffer is only returned to the pool if it hasn't grown too large
    /// and the pool isn't full.
    pub fn release(&self, buffer: String) {
        // Don't pool oversized buffers
        if buffer.capacity() > self.buffer_capacity * 4 {
            return;
        }

        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_pool_size {
                pool.push_back(buffer);
            }
        }
    }

    /// Returns pool statistics.
    pub fn stats(&self) -> PoolStats {
        let pool_size = self.pool.lock().map_or(0, |p| p.len());
        PoolStats {
            pool_size,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Current number of buffers in the pool.
    pub pool_size: usize,
    /// Number of successful acquisitions from the pool.
    pub hits: u64,
    /// Number of new allocations (pool empty).
    pub misses: u64,
}

impl PoolStats {
    /// Returns the hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Generic object pool for reusable objects.
///
/// Useful for reducing allocations of frequently created/destroyed objects
/// like request buffers, response builders, etc.
pub struct ObjectPool<T: Default + Send> {
    pool: Mutex<Vec<T>>,
    max_size: usize,
    creates: AtomicU64,
    reuses: AtomicU64,
}

impl<T: Default + Send> ObjectPool<T> {
    /// Creates a new object pool.
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(max_size)),
            max_size,
            creates: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
        }
    }

    /// Pre-allocates objects in the pool.
    pub fn prefill(&self, count: usize) {
        if let Ok(mut pool) = self.pool.lock() {
            let to_add = count.min(self.max_size - pool.len());
            for _ in 0..to_add {
                pool.push(T::default());
            }
        }
    }

    /// Acquires an object from the pool.
    pub fn acquire(&self) -> T {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(obj) = pool.pop() {
                self.reuses.fetch_add(1, Ordering::Relaxed);
                return obj;
            }
        }
        self.creates.fetch_add(1, Ordering::Relaxed);
        T::default()
    }

    /// Releases an object back to the pool.
    pub fn release(&self, obj: T) {
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_size {
                pool.push(obj);
            }
        }
    }

    /// Returns the current pool size.
    pub fn size(&self) -> usize {
        self.pool.lock().map_or(0, |p| p.len())
    }

    /// Returns the reuse rate (0.0 to 1.0).
    pub fn reuse_rate(&self) -> f64 {
        let creates = self.creates.load(Ordering::Relaxed);
        let reuses = self.reuses.load(Ordering::Relaxed);
        let total = creates + reuses;
        if total == 0 {
            0.0
        } else {
            reuses as f64 / total as f64
        }
    }
}

/// Tracks memory usage for monitoring.
#[derive(Debug, Default)]
pub struct MemoryTracker {
    allocated: AtomicUsize,
    peak: AtomicUsize,
    allocation_count: AtomicU64,
}

impl MemoryTracker {
    /// Creates a new memory tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an allocation.
    pub fn record_alloc(&self, size: usize) {
        let current = self.allocated.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Update peak if necessary
        let mut peak = self.peak.load(Ordering::Relaxed);
        while current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Records a deallocation.
    pub fn record_dealloc(&self, size: usize) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
    }

    /// Returns current allocated bytes.
    pub fn current(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Returns peak allocated bytes.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Returns total allocation count.
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count.load(Ordering::Relaxed)
    }
}

/// Pre-allocates a vector with known size.
///
/// This is a convenience function that documents the intent to avoid
/// reallocations during the vector's lifetime.
#[inline]
pub fn preallocated_vec<T>(capacity: usize) -> Vec<T> {
    Vec::with_capacity(capacity)
}

/// Extends a vector, pre-allocating if the additional capacity is known.
///
/// # Example
///
/// ```
/// use infernum_core::perf::extend_preallocated;
///
/// let mut vec = Vec::new();
/// let items = vec![1, 2, 3, 4, 5];
/// extend_preallocated(&mut vec, items.into_iter());
/// ```
#[inline]
pub fn extend_preallocated<T, I: ExactSizeIterator<Item = T>>(vec: &mut Vec<T>, iter: I) {
    vec.reserve(iter.len());
    vec.extend(iter);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_pool() {
        let pool = StringPool::new(4, 256);

        // First acquire creates new buffer
        let buf1 = pool.acquire();
        assert!(buf1.capacity() >= 256);

        // Release and acquire again
        pool.release(buf1);
        let buf2 = pool.acquire();
        assert!(buf2.capacity() >= 256);

        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_object_pool() {
        let pool: ObjectPool<Vec<u8>> = ObjectPool::new(4);

        // Pre-fill
        pool.prefill(2);
        assert_eq!(pool.size(), 2);

        // Acquire from pool
        let obj = pool.acquire();
        assert_eq!(pool.size(), 1);

        // Release back
        pool.release(obj);
        assert_eq!(pool.size(), 2);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();

        tracker.record_alloc(100);
        assert_eq!(tracker.current(), 100);
        assert_eq!(tracker.peak(), 100);

        tracker.record_alloc(50);
        assert_eq!(tracker.current(), 150);
        assert_eq!(tracker.peak(), 150);

        tracker.record_dealloc(100);
        assert_eq!(tracker.current(), 50);
        assert_eq!(tracker.peak(), 150); // Peak unchanged
    }
}
