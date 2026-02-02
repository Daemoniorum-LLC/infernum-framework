//! Integration tests for infernum-core.
//!
//! Tests the core types and utilities:
//! - Performance utilities (StringPool, ObjectPool, MemoryTracker)
//! - Edge deployment configuration
//! - Model cache and lightweight context

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

use infernum_core::{
    EdgeConfig, EdgeConfigError, EdgeTarget, LightweightContext, MemoryTracker, MemoryUsage,
    ModelCache, ObjectPool, PoolStats, QuantizationLevel, StringPool,
};

// ============================================================================
// StringPool Tests
// ============================================================================

#[test]
fn test_string_pool_basic_acquire_release() {
    let pool = StringPool::new(4, 256);

    // First acquire creates new buffer (miss)
    let buf1 = pool.acquire();
    assert!(buf1.capacity() >= 256);

    // Release and acquire again (hit)
    pool.release(buf1);
    let buf2 = pool.acquire();
    assert!(buf2.capacity() >= 256);

    let stats = pool.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_string_pool_max_size_limit() {
    let pool = StringPool::new(2, 128);

    // Fill the pool
    let b1 = pool.acquire();
    let b2 = pool.acquire();
    let b3 = pool.acquire();

    pool.release(b1);
    pool.release(b2);
    pool.release(b3); // This should be dropped (pool full)

    let stats = pool.stats();
    assert_eq!(stats.pool_size, 2); // Max size is 2
}

#[test]
fn test_string_pool_oversized_buffer_not_pooled() {
    let pool = StringPool::new(4, 100);

    // Create a buffer that's 5x the capacity (too large to pool)
    let mut buf = pool.acquire();
    for _ in 0..600 {
        buf.push('x');
    }
    assert!(buf.capacity() > 400); // 4x limit

    pool.release(buf);

    // Pool should be empty (oversized buffer dropped)
    let stats = pool.stats();
    assert_eq!(stats.pool_size, 0);
}

#[test]
fn test_string_pool_concurrent_access() {
    let pool = Arc::new(StringPool::new(16, 128));
    let mut handles = vec![];

    for _ in 0..8 {
        let pool = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let mut buf = pool.acquire();
                buf.push_str("test data");
                pool.release(buf);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panic");
    }

    let stats = pool.stats();
    // Total operations should be 800
    assert_eq!(stats.hits + stats.misses, 800);
    assert!(stats.hit_rate() > 0.0); // Some hits expected
}

#[test]
fn test_string_pool_hit_rate_calculation() {
    let pool = StringPool::new(4, 64);

    // All misses initially
    let b1 = pool.acquire();
    let b2 = pool.acquire();
    pool.release(b1);
    pool.release(b2);

    // Now some hits
    let _b3 = pool.acquire();
    let _b4 = pool.acquire();

    let stats = pool.stats();
    // 2 misses, 2 hits
    assert!((stats.hit_rate() - 0.5).abs() < 0.01);
}

// ============================================================================
// ObjectPool Tests
// ============================================================================

#[test]
fn test_object_pool_basic() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new(4);

    let obj1 = pool.acquire();
    assert!(obj1.is_empty()); // Default Vec is empty

    pool.release(obj1);
    assert_eq!(pool.size(), 1);
}

#[test]
fn test_object_pool_prefill() {
    let pool: ObjectPool<Vec<i32>> = ObjectPool::new(10);

    pool.prefill(5);
    assert_eq!(pool.size(), 5);

    // Prefill respects max size
    pool.prefill(10);
    assert_eq!(pool.size(), 10);
}

#[test]
fn test_object_pool_reuse_rate() {
    let pool: ObjectPool<String> = ObjectPool::new(4);

    // First acquire is a create
    let obj = pool.acquire();
    pool.release(obj);

    // Second acquire is a reuse
    let _obj2 = pool.acquire();

    // 1 create, 1 reuse = 50% reuse rate
    assert!((pool.reuse_rate() - 0.5).abs() < 0.01);
}

#[test]
fn test_object_pool_concurrent() {
    let pool: Arc<ObjectPool<Vec<u8>>> = Arc::new(ObjectPool::new(8));
    let mut handles = vec![];

    for _ in 0..4 {
        let pool = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let obj = pool.acquire();
                pool.release(obj);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panic");
    }

    // After all threads complete, reuse rate should be positive
    assert!(pool.reuse_rate() > 0.0);
}

// ============================================================================
// MemoryTracker Tests
// ============================================================================

#[test]
fn test_memory_tracker_basic() {
    let tracker = MemoryTracker::new();

    tracker.record_alloc(1024);
    assert_eq!(tracker.current(), 1024);
    assert_eq!(tracker.peak(), 1024);
    assert_eq!(tracker.allocation_count(), 1);
}

#[test]
fn test_memory_tracker_peak_tracking() {
    let tracker = MemoryTracker::new();

    tracker.record_alloc(1000);
    tracker.record_alloc(500);
    assert_eq!(tracker.current(), 1500);
    assert_eq!(tracker.peak(), 1500);

    tracker.record_dealloc(1000);
    assert_eq!(tracker.current(), 500);
    assert_eq!(tracker.peak(), 1500); // Peak unchanged

    tracker.record_alloc(2000);
    assert_eq!(tracker.current(), 2500);
    assert_eq!(tracker.peak(), 2500); // New peak
}

#[test]
fn test_memory_tracker_concurrent() {
    let tracker = Arc::new(MemoryTracker::new());
    let mut handles = vec![];

    for _ in 0..4 {
        let tracker = Arc::clone(&tracker);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                tracker.record_alloc(100);
                if i % 2 == 0 {
                    tracker.record_dealloc(50);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panic");
    }

    // 4 threads * 100 allocations
    assert_eq!(tracker.allocation_count(), 400);
}

// ============================================================================
// EdgeTarget Tests
// ============================================================================

#[test]
fn test_edge_target_properties() {
    // WASM target
    let wasm = EdgeTarget::Wasm;
    assert!(!wasm.has_gpu());
    assert_eq!(wasm.max_context_length(), 2048);
    assert!(wasm.max_model_size() <= 500 * 1024 * 1024);

    // iOS target
    let ios = EdgeTarget::Ios;
    assert!(ios.has_gpu());
    assert_eq!(ios.max_context_length(), 4096);

    // Embedded Linux
    let embedded = EdgeTarget::EmbeddedLinux;
    assert!(!embedded.has_gpu());
}

#[test]
fn test_edge_target_recommended_quantization() {
    assert_eq!(
        EdgeTarget::Wasm.recommended_quantization(),
        QuantizationLevel::Q4_0
    );
    assert_eq!(
        EdgeTarget::Ios.recommended_quantization(),
        QuantizationLevel::Q4_K_M
    );
    assert_eq!(
        EdgeTarget::LightweightDesktop.recommended_quantization(),
        QuantizationLevel::Q8_0
    );
}

// ============================================================================
// QuantizationLevel Tests
// ============================================================================

#[test]
fn test_quantization_compression_ratios() {
    // Higher compression = lower quality
    assert_eq!(QuantizationLevel::None.compression_ratio(), 1.0);
    assert!(QuantizationLevel::Q8_0.compression_ratio() > 1.0);
    assert!(QuantizationLevel::Q4_K_M.compression_ratio() > QuantizationLevel::Q8_0.compression_ratio());
    assert!(QuantizationLevel::Q2_K.compression_ratio() > QuantizationLevel::Q4_K_M.compression_ratio());
}

#[test]
fn test_quantization_quality_factors() {
    // No quantization = full quality
    assert_eq!(QuantizationLevel::None.quality_factor(), 1.0);

    // More aggressive = lower quality
    assert!(QuantizationLevel::Q8_0.quality_factor() > QuantizationLevel::Q4_K_M.quality_factor());
    assert!(QuantizationLevel::Q4_K_M.quality_factor() > QuantizationLevel::Q2_K.quality_factor());

    // All quality factors should be positive
    assert!(QuantizationLevel::Q2_K.quality_factor() > 0.0);
}

// ============================================================================
// EdgeConfig Tests
// ============================================================================

#[test]
fn test_edge_config_for_target() {
    let config = EdgeConfig::for_target(EdgeTarget::Wasm);
    assert_eq!(config.target, EdgeTarget::Wasm);
    assert_eq!(config.num_threads, 1);
    assert!(!config.use_mmap);
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_edge_config_wasm() {
    let config = EdgeConfig::wasm();
    assert_eq!(config.target, EdgeTarget::Wasm);
    assert!(!config.use_mmap); // WASM doesn't support mmap
}

#[test]
fn test_edge_config_mobile() {
    let config = EdgeConfig::mobile();
    assert!(config.use_mmap);
    assert_eq!(config.num_threads, 4);
}

#[test]
fn test_edge_config_default() {
    let config = EdgeConfig::default();
    assert_eq!(config.target, EdgeTarget::LightweightDesktop);
}

#[test]
fn test_edge_config_builder_pattern() {
    let temp = TempDir::new().expect("temp dir");
    let config = EdgeConfig::for_target(EdgeTarget::Android)
        .with_offline_cache(temp.path().to_path_buf())
        .with_max_memory(2 * 1024 * 1024 * 1024)
        .with_quantization(QuantizationLevel::Q5_K_M);

    assert!(config.offline_mode);
    assert!(config.cache_dir.is_some());
    assert_eq!(config.max_memory, 2 * 1024 * 1024 * 1024);
    assert_eq!(config.quantization, QuantizationLevel::Q5_K_M);
}

#[test]
fn test_edge_config_validation_success() {
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop);
    assert!(config.validate().is_ok());
}

#[test]
fn test_edge_config_validation_model_too_large() {
    let mut config = EdgeConfig::wasm();
    config.max_model_size = config.max_memory + 1;

    let result = config.validate();
    assert!(result.is_err());
    match result {
        Err(EdgeConfigError::ModelTooLarge { .. }) => {}
        _ => panic!("Expected ModelTooLarge error"),
    }
}

#[test]
fn test_edge_config_validation_context_too_large() {
    let mut config = EdgeConfig::wasm();
    config.max_context = EdgeTarget::Wasm.max_context_length() * 3;

    let result = config.validate();
    assert!(result.is_err());
    match result {
        Err(EdgeConfigError::ContextTooLarge { .. }) => {}
        _ => panic!("Expected ContextTooLarge error"),
    }
}

// ============================================================================
// ModelCache Tests
// ============================================================================

#[test]
fn test_model_cache_create() {
    let temp = TempDir::new().expect("temp dir");
    let cache = ModelCache::new(temp.path().to_path_buf(), 1024 * 1024 * 1024);
    assert!(cache.is_ok());

    let cache = cache.expect("cache");
    let stats = cache.stats();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.total_size, 0);
}

#[test]
fn test_model_cache_add_and_retrieve() {
    let temp = TempDir::new().expect("temp dir");
    let cache = ModelCache::new(temp.path().join("cache"), 1024 * 1024 * 1024).expect("cache");

    // Create a test file to cache
    let source_file = temp.path().join("model.bin");
    std::fs::write(&source_file, vec![0u8; 1024]).expect("write file");

    // Add to cache
    let cached_path = cache
        .add("test-model", &source_file, QuantizationLevel::Q4_K_M)
        .expect("add to cache");

    assert!(cache.is_cached("test-model"));
    assert!(cached_path.exists());

    // Retrieve
    let retrieved = cache.get_path("test-model");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.expect("path"), cached_path);
}

#[test]
fn test_model_cache_remove() {
    let temp = TempDir::new().expect("temp dir");
    let cache = ModelCache::new(temp.path().join("cache"), 1024 * 1024 * 1024).expect("cache");

    // Add a model
    let source = temp.path().join("model.bin");
    std::fs::write(&source, vec![0u8; 512]).expect("write");
    cache
        .add("remove-test", &source, QuantizationLevel::Q8_0)
        .expect("add");

    assert!(cache.is_cached("remove-test"));

    // Remove it
    let removed = cache.remove("remove-test").expect("remove");
    assert!(removed);
    assert!(!cache.is_cached("remove-test"));

    // Remove non-existent
    let removed = cache.remove("nonexistent").expect("remove");
    assert!(!removed);
}

#[test]
fn test_model_cache_stats() {
    let temp = TempDir::new().expect("temp dir");
    let max_size = 10 * 1024; // 10KB
    let cache = ModelCache::new(temp.path().join("cache"), max_size).expect("cache");

    let source = temp.path().join("model.bin");
    std::fs::write(&source, vec![0u8; 1024]).expect("write");
    cache.add("model1", &source, QuantizationLevel::Q4_0).expect("add");

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 1);
    assert!(stats.total_size > 0);
    assert_eq!(stats.max_size, max_size);
    assert!(stats.utilization > 0.0);
    assert!(stats.utilization < 1.0);
}

#[test]
fn test_model_cache_list() {
    let temp = TempDir::new().expect("temp dir");
    let cache = ModelCache::new(temp.path().join("cache"), 1024 * 1024).expect("cache");

    let source = temp.path().join("model.bin");
    std::fs::write(&source, vec![0u8; 128]).expect("write");

    cache.add("model-a", &source, QuantizationLevel::Q4_0).expect("add");
    cache.add("model-b", &source, QuantizationLevel::Q8_0).expect("add");

    let entries = cache.list();
    assert_eq!(entries.len(), 2);

    let ids: Vec<_> = entries.iter().map(|e| e.model_id.as_str()).collect();
    assert!(ids.contains(&"model-a"));
    assert!(ids.contains(&"model-b"));
}

#[test]
fn test_model_cache_clear() {
    let temp = TempDir::new().expect("temp dir");
    let cache = ModelCache::new(temp.path().join("cache"), 1024 * 1024).expect("cache");

    let source = temp.path().join("model.bin");
    std::fs::write(&source, vec![0u8; 64]).expect("write");

    cache.add("m1", &source, QuantizationLevel::Q4_0).expect("add");
    cache.add("m2", &source, QuantizationLevel::Q4_0).expect("add");

    assert_eq!(cache.stats().entry_count, 2);

    cache.clear().expect("clear");
    assert_eq!(cache.stats().entry_count, 0);
}

// ============================================================================
// LightweightContext Tests
// ============================================================================

#[test]
fn test_lightweight_context_basic() {
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop);
    let ctx = LightweightContext::new(config).expect("context");

    assert!(ctx.can_load_model(1024 * 1024 * 1024)); // 1GB model
}

#[test]
fn test_lightweight_context_memory_allocation() {
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop);
    let ctx = LightweightContext::new(config).expect("context");

    // Allocate some memory
    assert!(ctx.allocate(500 * 1024 * 1024)); // 500MB

    let usage = ctx.memory_usage();
    assert_eq!(usage.used, 500 * 1024 * 1024);
    assert!(usage.utilization > 0.0);

    // Try to check if we can load more
    assert!(ctx.can_load_model(100 * 1024 * 1024)); // 100MB more

    // Deallocate
    ctx.deallocate(500 * 1024 * 1024);
    assert_eq!(ctx.memory_usage().used, 0);
}

#[test]
fn test_lightweight_context_allocation_limit() {
    let mut config = EdgeConfig::for_target(EdgeTarget::Wasm);
    config.max_memory = 100; // Very small limit

    let ctx = LightweightContext::new(config).expect("context");

    // Should fail to allocate more than max
    assert!(!ctx.allocate(200));
    assert_eq!(ctx.memory_usage().used, 0);
}

#[test]
fn test_lightweight_context_with_cache() {
    let temp = TempDir::new().expect("temp dir");
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop)
        .with_offline_cache(temp.path().to_path_buf());

    let ctx = LightweightContext::new(config).expect("context");

    // Cache should be initialized
    assert!(ctx.cache.is_some());

    // Try to get non-existent model
    assert!(ctx.get_cached_model("nonexistent").is_none());
}

#[test]
fn test_lightweight_context_cache_model() {
    let temp = TempDir::new().expect("temp dir");
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop)
        .with_offline_cache(temp.path().join("cache"));

    let ctx = LightweightContext::new(config).expect("context");

    // Create a model file
    let model_file = temp.path().join("my_model.bin");
    std::fs::write(&model_file, vec![0u8; 256]).expect("write");

    // Cache it
    let cached = ctx.cache_model("my-model", &model_file).expect("cache");
    assert!(cached.is_some());

    // Retrieve it
    let retrieved = ctx.get_cached_model("my-model");
    assert!(retrieved.is_some());
}

#[test]
fn test_lightweight_context_without_cache() {
    let config = EdgeConfig::for_target(EdgeTarget::Wasm); // No offline cache

    let ctx = LightweightContext::new(config).expect("context");

    // Cache should not be initialized
    assert!(ctx.cache.is_none());

    // Cache operations should return None/Ok(None)
    let temp = TempDir::new().expect("temp dir");
    let model_file = temp.path().join("model.bin");
    std::fs::write(&model_file, vec![0u8; 64]).expect("write");

    let cached = ctx.cache_model("model", &model_file).expect("cache");
    assert!(cached.is_none());
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

#[test]
fn test_memory_usage_utilization() {
    let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop);
    let ctx = LightweightContext::new(config).expect("context");

    ctx.allocate(ctx.config.max_memory / 2);

    let usage = ctx.memory_usage();
    assert!((usage.utilization - 0.5).abs() < 0.01);
    assert_eq!(usage.used + usage.available, usage.total);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_edge_deployment_workflow() {
    let temp = TempDir::new().expect("temp dir");

    // Configure for edge deployment
    let config = EdgeConfig::for_target(EdgeTarget::Android)
        .with_offline_cache(temp.path().join("model_cache"))
        .with_max_memory(2 * 1024 * 1024 * 1024)
        .with_quantization(QuantizationLevel::Q4_K_S);

    // Validate config
    assert!(config.validate().is_ok());

    // Create context
    let ctx = LightweightContext::new(config).expect("context");

    // Simulate loading a model
    let model_size = 500 * 1024 * 1024; // 500MB
    assert!(ctx.can_load_model(model_size));
    assert!(ctx.allocate(model_size));

    // Check memory state
    let usage = ctx.memory_usage();
    assert_eq!(usage.used, model_size);

    // Create a mock model file and cache it
    let model_file = temp.path().join("quantized_model.bin");
    std::fs::write(&model_file, vec![0u8; 1024]).expect("write");

    let cached_path = ctx.cache_model("android-model", &model_file).expect("cache");
    assert!(cached_path.is_some());

    // Verify model is cached
    let retrieved = ctx.get_cached_model("android-model");
    assert!(retrieved.is_some());

    // Unload model
    ctx.deallocate(model_size);
    assert_eq!(ctx.memory_usage().used, 0);
}

#[test]
fn test_performance_pool_workflow() {
    // Create pools for a realistic workload
    let string_pool = Arc::new(StringPool::new(32, 4096));
    let buffer_pool: Arc<ObjectPool<Vec<u8>>> = Arc::new(ObjectPool::new(16));
    let memory = Arc::new(MemoryTracker::new());

    // Prefill buffer pool
    buffer_pool.prefill(8);

    // Simulate concurrent workload
    let mut handles = vec![];

    for _ in 0..4 {
        let sp = Arc::clone(&string_pool);
        let bp = Arc::clone(&buffer_pool);
        let mem = Arc::clone(&memory);

        handles.push(thread::spawn(move || {
            for i in 0..50 {
                // Use string pool
                let mut s = sp.acquire();
                s.push_str(&format!("request_{}", i));
                sp.release(s);

                // Use buffer pool
                let buf = bp.acquire();
                mem.record_alloc(buf.capacity());
                mem.record_dealloc(buf.capacity());
                bp.release(buf);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread panic");
    }

    // Verify pool efficiency
    let string_stats = string_pool.stats();
    assert!(string_stats.hit_rate() > 0.5); // Good reuse

    assert!(buffer_pool.reuse_rate() > 0.5);

    // Memory should be balanced
    assert_eq!(memory.current(), 0);
}
