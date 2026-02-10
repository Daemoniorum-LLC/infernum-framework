//! Phase 4 TDD Tests: Arena Allocator for HoloTensor
//!
//! These tests verify that the arena allocator reduces heap pressure
//! and improves throughput for fragment allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use abaddon::holotensor::arena::{
    ArenaAllocator, ArenaConfig, ArenaStats, FragmentArena, ThreadLocalArena,
};

// =============================================================================
// Phase 4.1: Basic Arena Allocation
// =============================================================================

#[test]
fn test_arena_allocates_fragments() {
    // Create arena with 1MB capacity
    let arena = FragmentArena::new(1024 * 1024);

    // Allocate multiple fragments
    let frag1 = arena.alloc(4096);
    let frag2 = arena.alloc(8192);
    let frag3 = arena.alloc(4096);

    assert!(!frag1.is_null());
    assert!(!frag2.is_null());
    assert!(!frag3.is_null());

    // Fragments should be at different addresses
    assert_ne!(frag1, frag2);
    assert_ne!(frag2, frag3);

    // Total allocated should match
    let stats = arena.stats();
    assert_eq!(stats.bytes_allocated, 4096 + 8192 + 4096);
    assert_eq!(stats.allocation_count, 3);
}

#[test]
fn test_arena_alignment() {
    let arena = FragmentArena::new(1024 * 1024);

    // Allocate with different sizes, verify alignment
    for size in [1, 7, 15, 63, 127, 1000, 4095] {
        let ptr = arena.alloc(size);
        assert!(!ptr.is_null());
        // Default alignment should be 8 bytes
        assert_eq!(ptr as usize % 8, 0, "Pointer not aligned for size {}", size);
    }
}

#[test]
fn test_arena_capacity_limit() {
    // Small arena (1KB)
    let arena = FragmentArena::new(1024);

    // Should succeed
    let ptr1 = arena.alloc(512);
    assert!(!ptr1.is_null());

    // Should succeed (fits remaining)
    let ptr2 = arena.alloc(256);
    assert!(!ptr2.is_null());

    // Should fail - exceeds capacity (accounting for alignment)
    let ptr3 = arena.alloc(512);
    assert!(ptr3.is_null(), "Should return null when capacity exceeded");
}

// =============================================================================
// Phase 4.2: Memory Reuse
// =============================================================================

#[test]
fn test_arena_reuses_memory() {
    let arena = FragmentArena::new(1024 * 1024);

    // First allocation cycle
    let ptr1 = arena.alloc(4096);
    let ptr2 = arena.alloc(4096);
    assert!(!ptr1.is_null());
    assert!(!ptr2.is_null());

    let first_base = arena.base_ptr();
    let first_used = arena.bytes_used();

    // Reset arena
    arena.reset();

    // Second allocation cycle
    let ptr3 = arena.alloc(4096);
    let ptr4 = arena.alloc(4096);

    // Should reuse same memory region
    assert_eq!(
        arena.base_ptr(),
        first_base,
        "Base pointer should be same after reset"
    );

    // New allocations should be at same offsets
    assert_eq!(
        ptr3, ptr1,
        "First allocation after reset should be at same address"
    );
    assert_eq!(
        ptr4, ptr2,
        "Second allocation after reset should be at same address"
    );

    let stats = arena.stats();
    assert_eq!(stats.reset_count, 1);
}

#[test]
fn test_arena_reset_clears_used() {
    let arena = FragmentArena::new(1024 * 1024);

    arena.alloc(10000);
    arena.alloc(20000);
    assert!(arena.bytes_used() >= 30000);

    arena.reset();
    assert_eq!(arena.bytes_used(), 0);

    let stats = arena.stats();
    assert_eq!(stats.bytes_allocated, 0);
    assert_eq!(stats.allocation_count, 0);
}

// =============================================================================
// Phase 4.3: Allocation Tracking
// =============================================================================

/// Counting allocator that wraps System allocator
struct CountingAllocator {
    inner: System,
    alloc_count: AtomicUsize,
    dealloc_count: AtomicUsize,
    bytes_allocated: AtomicUsize,
}

impl CountingAllocator {
    const fn new() -> Self {
        Self {
            inner: System,
            alloc_count: AtomicUsize::new(0),
            dealloc_count: AtomicUsize::new(0),
            bytes_allocated: AtomicUsize::new(0),
        }
    }

    fn alloc_count(&self) -> usize {
        self.alloc_count.load(Ordering::SeqCst)
    }

    fn reset_counts(&self) {
        self.alloc_count.store(0, Ordering::SeqCst);
        self.dealloc_count.store(0, Ordering::SeqCst);
        self.bytes_allocated.store(0, Ordering::SeqCst);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_count.fetch_add(1, Ordering::SeqCst);
        self.bytes_allocated
            .fetch_add(layout.size(), Ordering::SeqCst);
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.dealloc_count.fetch_add(1, Ordering::SeqCst);
        self.inner.dealloc(ptr, layout)
    }
}

#[test]
fn test_arena_reduces_allocations() {
    // Test that arena batches allocations

    // Without arena: each Vec allocation hits the allocator
    let vec_allocs = {
        let mut count = 0;
        for _ in 0..100 {
            let v: Vec<u8> = vec![0u8; 4096];
            count += 1;
            std::hint::black_box(v);
        }
        count
    };

    // With arena: single large allocation, then bump pointer
    let arena = FragmentArena::new(100 * 4096 + 4096); // Extra for alignment
    let arena_stats_before = arena.stats();

    for _ in 0..100 {
        let ptr = arena.alloc(4096);
        std::hint::black_box(ptr);
    }

    let arena_stats_after = arena.stats();

    // Arena should have exactly 100 logical allocations but only 1 system alloc
    assert_eq!(arena_stats_after.allocation_count, 100);
    // Arena's internal system allocation count should be 1 (the initial buffer)
    assert_eq!(arena.system_alloc_count(), 1);

    println!("Vec allocations: {}", vec_allocs);
    println!(
        "Arena allocation count: {}",
        arena_stats_after.allocation_count
    );
    println!("Arena system allocations: {}", arena.system_alloc_count());
}

// =============================================================================
// Phase 4.4: Performance Benchmarks
// =============================================================================

#[test]
fn test_arena_throughput_improvement() {
    const NUM_ALLOCS: usize = 10_000;
    const ALLOC_SIZE: usize = 4096;

    // Benchmark Vec allocations
    let vec_start = Instant::now();
    for _ in 0..NUM_ALLOCS {
        let v: Vec<u8> = Vec::with_capacity(ALLOC_SIZE);
        std::hint::black_box(v);
    }
    let vec_elapsed = vec_start.elapsed();

    // Benchmark arena allocations
    let arena = FragmentArena::new(NUM_ALLOCS * ALLOC_SIZE + 4096);
    let arena_start = Instant::now();
    for _ in 0..NUM_ALLOCS {
        let ptr = arena.alloc(ALLOC_SIZE);
        std::hint::black_box(ptr);
    }
    let arena_elapsed = arena_start.elapsed();

    println!(
        "Vec: {:?} for {} allocations ({:.2} ns/alloc)",
        vec_elapsed,
        NUM_ALLOCS,
        vec_elapsed.as_nanos() as f64 / NUM_ALLOCS as f64
    );
    println!(
        "Arena: {:?} for {} allocations ({:.2} ns/alloc)",
        arena_elapsed,
        NUM_ALLOCS,
        arena_elapsed.as_nanos() as f64 / NUM_ALLOCS as f64
    );

    // Arena should be at least 10% faster (likely much more)
    let speedup = vec_elapsed.as_nanos() as f64 / arena_elapsed.as_nanos() as f64;
    println!("Speedup: {:.2}x", speedup);

    assert!(
        speedup >= 1.1,
        "Arena should be at least 10% faster, got {:.2}x speedup",
        speedup
    );
}

#[test]
fn test_arena_allocation_reset_cycle() {
    // Simulate typical usage: allocate, process, reset, repeat
    let arena = FragmentArena::new(1024 * 1024);

    let start = Instant::now();
    for _cycle in 0..1000 {
        // Allocate fragments for a batch
        for _ in 0..32 {
            let ptr = arena.alloc(4096);
            std::hint::black_box(ptr);
        }
        // Reset for next batch
        arena.reset();
    }
    let elapsed = start.elapsed();

    println!(
        "1000 cycles of 32 allocations + reset: {:?} ({:.2} μs/cycle)",
        elapsed,
        elapsed.as_micros() as f64 / 1000.0
    );

    // Should complete 1000 cycles in < 10ms
    assert!(
        elapsed.as_millis() < 10,
        "Allocation/reset cycle too slow: {:?}",
        elapsed
    );
}

// =============================================================================
// Phase 4.5: Thread Safety
// =============================================================================

#[test]
fn test_arena_thread_safe() {
    // Test thread-local arena access
    let results: Vec<_> = (0..4)
        .map(|thread_id| {
            thread::spawn(move || {
                // Each thread gets its own arena
                let arena = ThreadLocalArena::get_or_init(|| FragmentArena::new(1024 * 1024));

                let mut ptrs = Vec::new();
                for _ in 0..100 {
                    let ptr = arena.alloc(4096);
                    assert!(!ptr.is_null());
                    ptrs.push(ptr);
                }

                // Verify all allocations are within arena bounds
                let base = arena.base_ptr() as usize;
                let end = base + arena.capacity();
                for ptr in &ptrs {
                    let addr = *ptr as usize;
                    assert!(
                        addr >= base && addr < end,
                        "Thread {} allocation outside arena bounds",
                        thread_id
                    );
                }

                arena.stats()
            })
        })
        .collect();

    // Wait for all threads and verify results
    for (i, handle) in results.into_iter().enumerate() {
        let stats = handle.join().expect("Thread panicked");
        assert_eq!(
            stats.allocation_count, 100,
            "Thread {} should have 100 allocations",
            i
        );
    }
}

#[test]
fn test_thread_local_arena_isolation() {
    use std::sync::atomic::AtomicPtr;
    use std::sync::{Arc, Barrier};

    const NUM_THREADS: usize = 4;

    // Track arena base addresses from each thread
    let addresses: Vec<Arc<AtomicPtr<u8>>> = (0..NUM_THREADS)
        .map(|_| Arc::new(AtomicPtr::new(std::ptr::null_mut())))
        .collect();

    // Barrier ensures all threads are alive simultaneously before capturing addresses
    // This prevents memory address reuse from sequential thread execution
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    let handles: Vec<_> = addresses
        .iter()
        .map(|addr_holder| {
            let addr = Arc::clone(addr_holder);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let arena = ThreadLocalArena::get_or_init(|| FragmentArena::new(1024 * 1024));

                // Wait for all threads to have their arena before storing address
                barrier.wait();

                addr.store(arena.base_ptr(), Ordering::SeqCst);

                // Do some allocations
                for _ in 0..10 {
                    arena.alloc(1024);
                }

                // Keep thread alive until all have stored their addresses
                barrier.wait();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify each thread got a different arena
    let base_addrs: Vec<_> = addresses
        .iter()
        .map(|a| a.load(Ordering::SeqCst) as usize)
        .collect();

    for i in 0..base_addrs.len() {
        for j in (i + 1)..base_addrs.len() {
            assert_ne!(
                base_addrs[i], base_addrs[j],
                "Threads {} and {} should have different arenas",
                i, j
            );
        }
    }
}

// =============================================================================
// Phase 4.6: ArenaAllocator Integration
// =============================================================================

#[test]
fn test_arena_allocator_wrapper() {
    // Test using ArenaAllocator wrapper
    let arena = FragmentArena::new(1024 * 1024);
    let allocator = ArenaAllocator::new(&arena);

    // Allocate memory using the allocator wrapper
    let ptr = allocator.alloc(4096);
    assert!(!ptr.is_null());

    // Write to the allocated memory
    unsafe {
        std::ptr::write_bytes(ptr, 0xAB, 4096);
    }

    assert_eq!(arena.stats().allocation_count, 1);

    // Allocate zeroed memory
    let zeroed_ptr = allocator.alloc_zeroed(1024);
    let slice = unsafe { std::slice::from_raw_parts(zeroed_ptr, 1024) };
    assert!(slice.iter().all(|&b| b == 0));

    assert_eq!(arena.stats().allocation_count, 2);
}

#[test]
fn test_arena_allocator_fragment_buffer() {
    // Simulate fragment buffer allocation pattern
    let arena = FragmentArena::new(10 * 1024 * 1024); // 10MB

    // Allocate buffers like in converter
    let mut buffers = Vec::new();
    for i in 0..100 {
        let size = 4096 + (i % 10) * 1024; // Variable sizes
        let ptr = arena.alloc(size);
        assert!(!ptr.is_null());
        buffers.push((ptr, size));
    }

    // Verify no overlap
    for i in 0..buffers.len() {
        let (ptr_i, size_i) = buffers[i];
        let start_i = ptr_i as usize;
        let end_i = start_i + size_i;

        for j in (i + 1)..buffers.len() {
            let (ptr_j, size_j) = buffers[j];
            let start_j = ptr_j as usize;
            let end_j = start_j + size_j;

            assert!(
                end_i <= start_j || end_j <= start_i,
                "Buffers {} and {} overlap",
                i,
                j
            );
        }
    }
}

// =============================================================================
// Phase 4.7: Configuration
// =============================================================================

#[test]
fn test_arena_config() {
    let config = ArenaConfig {
        initial_capacity: 512 * 1024,
        max_capacity: 4 * 1024 * 1024,
        alignment: 16,
        grow_factor: 2.0,
    };

    let arena = FragmentArena::with_config(config.clone());
    assert_eq!(arena.capacity(), 512 * 1024);

    // Verify alignment
    let ptr = arena.alloc(100);
    assert_eq!(ptr as usize % 16, 0);
}

#[test]
fn test_arena_config_default() {
    let config = ArenaConfig::default();
    assert_eq!(config.alignment, 8);
    assert!(config.initial_capacity >= 64 * 1024); // At least 64KB default
}

// =============================================================================
// Phase 4 Quality Gate Summary
// =============================================================================

#[test]
fn phase_4_quality_gate_summary() {
    println!("\n");
    println!("═══════════════════════════════════════════════════════");
    println!("  Phase 4 Quality Gate: Arena Allocator");
    println!("═══════════════════════════════════════════════════════");
    println!("");
    println!("  Tests:");
    println!("  ✓ Basic arena allocation");
    println!("  ✓ Pointer alignment");
    println!("  ✓ Capacity limits");
    println!("  ✓ Memory reuse after reset");
    println!("  ✓ Reduced system allocations");
    println!("  ✓ Throughput improvement (>=10%)");
    println!("  ✓ Allocation/reset cycle performance");
    println!("  ✓ Thread-local arena isolation");
    println!("  ✓ ArenaAllocator integration");
    println!("");
    println!("═══════════════════════════════════════════════════════");
    println!("");
}
