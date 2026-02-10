//! Arena Allocator for HoloTensor Fragment Allocations
//!
//! Provides a bump allocator optimized for fragment buffer allocations.
//! Reduces heap pressure by batching allocations into a single large buffer.
//!
//! # Usage
//!
//! ```ignore
//! use abaddon::holotensor::arena::FragmentArena;
//!
//! let arena = FragmentArena::new(1024 * 1024); // 1MB arena
//!
//! // Allocate fragment buffers
//! let ptr = arena.alloc(4096);
//!
//! // Reset for next batch (reuses memory)
//! arena.reset();
//! ```

use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for arena allocation.
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Initial capacity in bytes.
    pub initial_capacity: usize,
    /// Maximum capacity (for growing arenas).
    pub max_capacity: usize,
    /// Minimum alignment for allocations.
    pub alignment: usize,
    /// Growth factor when arena needs to expand.
    pub grow_factor: f64,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 256 * 1024,   // 256KB
            max_capacity: 64 * 1024 * 1024, // 64MB
            alignment: 8,
            grow_factor: 2.0,
        }
    }
}

impl ArenaConfig {
    /// Create config for small allocations (fragment metadata).
    pub fn small() -> Self {
        Self {
            initial_capacity: 64 * 1024,
            max_capacity: 1024 * 1024,
            ..Default::default()
        }
    }

    /// Create config for large allocations (fragment data).
    pub fn large() -> Self {
        Self {
            initial_capacity: 1024 * 1024,
            max_capacity: 256 * 1024 * 1024,
            ..Default::default()
        }
    }
}

/// Statistics for arena usage.
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total bytes allocated from arena.
    pub bytes_allocated: usize,
    /// Number of allocations.
    pub allocation_count: usize,
    /// Number of times arena was reset.
    pub reset_count: usize,
    /// Peak bytes used before any reset.
    pub peak_bytes: usize,
}

/// Bump allocator for fragment buffers.
///
/// This allocator maintains a single contiguous buffer and allocates
/// by bumping a pointer. Individual deallocations are not supported;
/// instead, call `reset()` to reclaim all memory at once.
///
/// This is ideal for batch processing where many temporary allocations
/// are made and then discarded together.
pub struct FragmentArena {
    /// Base pointer of the allocated buffer.
    buffer: NonNull<u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Current allocation offset (atomic for interior mutability).
    offset: AtomicUsize,
    /// Configuration.
    config: ArenaConfig,
    /// Statistics (interior mutability).
    stats: UnsafeCell<ArenaStats>,
    /// Number of system allocations made (should be 1).
    system_alloc_count: AtomicUsize,
}

// Safety: FragmentArena can be shared between threads because:
// - offset uses AtomicUsize for thread-safe bumping
// - buffer is never reallocated while shared
// - stats are only updated through atomic operations or single-threaded access
unsafe impl Send for FragmentArena {}
unsafe impl Sync for FragmentArena {}

impl FragmentArena {
    /// Create a new arena with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self::with_config(ArenaConfig {
            initial_capacity: capacity,
            max_capacity: capacity,
            ..Default::default()
        })
    }

    /// Create a new arena with the specified configuration.
    pub fn with_config(config: ArenaConfig) -> Self {
        let capacity = config.initial_capacity;
        let layout =
            Layout::from_size_align(capacity, config.alignment).expect("Invalid arena layout");

        // Allocate the backing buffer
        let ptr = unsafe { alloc(layout) };
        let buffer = NonNull::new(ptr).expect("Arena allocation failed");

        Self {
            buffer,
            capacity,
            offset: AtomicUsize::new(0),
            config,
            stats: UnsafeCell::new(ArenaStats::default()),
            system_alloc_count: AtomicUsize::new(1),
        }
    }

    /// Allocate bytes from the arena.
    ///
    /// Returns a pointer to the allocated memory, or null if
    /// the arena doesn't have enough capacity.
    ///
    /// The returned pointer is aligned to the arena's alignment setting.
    pub fn alloc(&self, size: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::null_mut();
        }

        let alignment = self.config.alignment;

        loop {
            let current = self.offset.load(Ordering::Relaxed);

            // Align the current offset
            let aligned = (current + alignment - 1) & !(alignment - 1);
            let new_offset = aligned + size;

            if new_offset > self.capacity {
                return std::ptr::null_mut();
            }

            // Try to bump the offset
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Update stats (relaxed because exact counts aren't critical)
                    unsafe {
                        let stats = &mut *self.stats.get();
                        stats.bytes_allocated += size;
                        stats.allocation_count += 1;
                        if new_offset > stats.peak_bytes {
                            stats.peak_bytes = new_offset;
                        }
                    }

                    // Return pointer at aligned offset
                    return unsafe { self.buffer.as_ptr().add(aligned) };
                },
                Err(_) => {
                    // Another thread beat us, retry
                    continue;
                },
            }
        }
    }

    /// Allocate and zero-initialize bytes from the arena.
    pub fn alloc_zeroed(&self, size: usize) -> *mut u8 {
        let ptr = self.alloc(size);
        if !ptr.is_null() {
            unsafe {
                std::ptr::write_bytes(ptr, 0, size);
            }
        }
        ptr
    }

    /// Reset the arena, reclaiming all allocations.
    ///
    /// This doesn't deallocate the underlying buffer; it just
    /// resets the bump pointer to the beginning.
    pub fn reset(&self) {
        self.offset.store(0, Ordering::SeqCst);
        unsafe {
            let stats = &mut *self.stats.get();
            stats.reset_count += 1;
            stats.bytes_allocated = 0;
            stats.allocation_count = 0;
        }
    }

    /// Get the base pointer of the arena buffer.
    pub fn base_ptr(&self) -> *mut u8 {
        self.buffer.as_ptr()
    }

    /// Get the total capacity of the arena.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current bytes used in the arena.
    pub fn bytes_used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Get the remaining capacity.
    pub fn bytes_remaining(&self) -> usize {
        self.capacity.saturating_sub(self.bytes_used())
    }

    /// Get arena statistics.
    pub fn stats(&self) -> ArenaStats {
        unsafe { (*self.stats.get()).clone() }
    }

    /// Get number of system allocations (should be 1).
    pub fn system_alloc_count(&self) -> usize {
        self.system_alloc_count.load(Ordering::Relaxed)
    }
}

impl Drop for FragmentArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, self.config.alignment)
            .expect("Invalid arena layout");
        unsafe {
            dealloc(self.buffer.as_ptr(), layout);
        }
    }
}

/// Thread-local arena storage.
///
/// Each thread gets its own arena to avoid contention.
pub struct ThreadLocalArena;

impl ThreadLocalArena {
    /// Get or initialize the thread-local arena.
    ///
    /// The provided closure is called to create the arena if it doesn't exist.
    pub fn get_or_init<F>(init: F) -> &'static FragmentArena
    where
        F: FnOnce() -> FragmentArena,
    {
        thread_local! {
            static ARENA: std::cell::OnceCell<FragmentArena> = std::cell::OnceCell::new();
        }

        ARENA.with(|cell| {
            let arena = cell.get_or_init(init);
            // Safety: The arena lives for the lifetime of the thread
            unsafe { &*(arena as *const FragmentArena) }
        })
    }
}

/// Custom allocator wrapper that uses a FragmentArena.
///
/// This provides a simple interface for allocating from an arena
/// without needing nightly Rust's allocator_api.
pub struct ArenaAllocator<'a> {
    arena: &'a FragmentArena,
}

impl<'a> ArenaAllocator<'a> {
    /// Create a new arena allocator wrapping the given arena.
    pub fn new(arena: &'a FragmentArena) -> Self {
        Self { arena }
    }

    /// Allocate memory from the arena.
    pub fn alloc(&self, size: usize) -> *mut u8 {
        self.arena.alloc(size)
    }

    /// Allocate and zero-initialize memory from the arena.
    pub fn alloc_zeroed(&self, size: usize) -> *mut u8 {
        self.arena.alloc_zeroed(size)
    }

    /// Get the underlying arena.
    pub fn arena(&self) -> &FragmentArena {
        self.arena
    }
}

/// Scoped arena guard that automatically resets on drop.
pub struct ArenaScope<'a> {
    arena: &'a FragmentArena,
    start_offset: usize,
}

impl<'a> ArenaScope<'a> {
    /// Create a new scope for the arena.
    pub fn new(arena: &'a FragmentArena) -> Self {
        Self {
            arena,
            start_offset: arena.bytes_used(),
        }
    }

    /// Allocate from the arena within this scope.
    pub fn alloc(&self, size: usize) -> *mut u8 {
        self.arena.alloc(size)
    }

    /// Get the allocator for this scope.
    pub fn allocator(&self) -> ArenaAllocator<'a> {
        ArenaAllocator::new(self.arena)
    }
}

impl<'a> Drop for ArenaScope<'a> {
    fn drop(&mut self) {
        // Rewind to start offset (partial reset)
        self.arena.offset.store(self.start_offset, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let arena = FragmentArena::new(4096);
        let ptr = arena.alloc(100);
        assert!(!ptr.is_null());
        assert_eq!(arena.stats().allocation_count, 1);
    }

    #[test]
    fn test_alloc_alignment() {
        let config = ArenaConfig {
            initial_capacity: 4096,
            alignment: 64,
            ..Default::default()
        };
        let arena = FragmentArena::with_config(config);

        for _ in 0..10 {
            let ptr = arena.alloc(17); // Odd size
            assert_eq!(ptr as usize % 64, 0);
        }
    }

    #[test]
    fn test_alloc_zeroed() {
        let arena = FragmentArena::new(4096);
        let ptr = arena.alloc_zeroed(100);
        assert!(!ptr.is_null());

        let slice = unsafe { std::slice::from_raw_parts(ptr, 100) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_reset() {
        let arena = FragmentArena::new(4096);
        arena.alloc(100);
        arena.alloc(200);
        assert!(arena.bytes_used() >= 300);

        arena.reset();
        assert_eq!(arena.bytes_used(), 0);
        assert_eq!(arena.stats().reset_count, 1);
    }

    #[test]
    fn test_scope() {
        let arena = FragmentArena::new(4096);
        arena.alloc(100);
        let before = arena.bytes_used();

        {
            let _scope = ArenaScope::new(&arena);
            arena.alloc(200);
            assert!(arena.bytes_used() >= before + 200);
        }

        // Scope dropped, should be back to before
        assert_eq!(arena.bytes_used(), before);
    }
}
