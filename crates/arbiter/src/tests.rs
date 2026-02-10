//! Tests for the Arbiter crate.

use super::*;

mod arbiter_tests {
    use super::*;

    #[test]
    fn test_concurrent_allocations() {
        use std::sync::Arc;
        use std::thread;

        let arbiter = Arc::new(Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed"));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let arbiter = Arc::clone(&arbiter);
                thread::spawn(move || {
                    let workload = if i % 2 == 0 {
                        WorkloadType::LlmInference
                    } else {
                        WorkloadType::ImageGeneration
                    };

                    arbiter.request_allocation(
                        workload,
                        Priority::Normal,
                        1024 * 1024 * 1024, // 1GB each
                    )
                })
            })
            .collect();

        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("Thread panic"))
            .collect();

        // All allocations should succeed (24GB available, 4GB requested)
        for result in results {
            assert!(result.is_ok());
        }

        let state = arbiter.state();
        assert_eq!(
            state.active_llm_workloads + state.active_diffusion_workloads,
            4
        );
    }

    #[test]
    fn test_quality_degrades_under_pressure() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed");

        // First allocation at low pressure
        let alloc1 = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                8 * 1024 * 1024 * 1024, // 8GB - 50% pressure
            )
            .expect("Failed");

        // Second allocation at higher pressure
        let alloc2 = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                6 * 1024 * 1024 * 1024, // 6GB more - 87.5% pressure
            )
            .expect("Failed");

        // Quality should be lower for the second allocation
        assert!(alloc2.quality_target <= alloc1.quality_target);
    }

    #[test]
    fn test_release_restores_quality() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed");

        let alloc1 = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                12 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        let state_before = arbiter.state();

        arbiter.release_allocation(&alloc1);

        let state_after = arbiter.state();

        assert!(state_after.vram_available > state_before.vram_available);
        assert!(state_after.memory_pressure < state_before.memory_pressure);
    }

    #[test]
    fn test_recommended_quality() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        // No workloads - full quality
        assert_eq!(arbiter.recommended_quality(WorkloadType::LlmInference), 1.0);

        // Add LLM workload
        let _alloc = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                10 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        // Should have quality set
        let q = arbiter.recommended_quality(WorkloadType::LlmInference);
        assert!(q > 0.0 && q <= 1.0);
    }
}

mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_tracker_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(GpuMemoryTracker::new(10000));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for _ in 0..100 {
                        tracker.allocate(10);
                        tracker.deallocate(10);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panic");
        }

        // After equal allocs and deallocs, should be back to 0
        assert_eq!(tracker.used(), 0);
    }
}

mod cache_tests {
    use super::*;

    #[test]
    fn test_shared_fragments_evict_last() {
        let cache = FragmentCache::new(CacheConfig {
            vram_capacity: 300,
            ram_capacity: 1000,
        });

        // Insert non-shared fragment
        cache.insert("frag1", 100, CacheTier::Vram, true);

        // Insert shared fragment
        cache.insert("frag2", 100, CacheTier::Vram, false);
        cache.mark_shared("frag2");

        // Insert another non-shared
        cache.insert("frag3", 100, CacheTier::Vram, true);

        // Force eviction
        cache.insert("frag4", 100, CacheTier::Vram, true);

        // Shared fragment should still be cached
        assert!(cache.contains("frag2"));
    }

    #[test]
    fn test_cache_tier_tracking() {
        let cache = FragmentCache::new(CacheConfig {
            vram_capacity: 1000,
            ram_capacity: 1000,
        });

        cache.insert("vram_frag", 100, CacheTier::Vram, true);
        cache.insert("ram_frag", 200, CacheTier::Ram, true);

        assert_eq!(cache.vram_used(), 100);
        assert_eq!(cache.ram_used(), 200);

        cache.remove("vram_frag");
        assert_eq!(cache.vram_used(), 0);
        assert_eq!(cache.ram_used(), 200);
    }
}

mod coordinator_tests {
    use super::*;

    #[test]
    fn test_priority_affects_quality() {
        let coord = Coordinator::new(CoordinatorConfig::default());

        let q_low = coord.calculate_quality(WorkloadType::LlmInference, Priority::Low, 0.5);

        let q_high = coord.calculate_quality(WorkloadType::LlmInference, Priority::High, 0.5);

        assert!(q_high > q_low);
    }

    #[test]
    fn test_workload_type_minimums() {
        let coord = Coordinator::new(CoordinatorConfig::default());

        // Even at maximum pressure with lowest priority
        let llm_q = coord.calculate_quality(WorkloadType::LlmInference, Priority::Background, 1.0);

        let img_q =
            coord.calculate_quality(WorkloadType::ImageGeneration, Priority::Background, 1.0);

        assert!(llm_q >= coord.min_quality(WorkloadType::LlmInference));
        assert!(img_q >= coord.min_quality(WorkloadType::ImageGeneration));
    }
}

mod integration_scenarios {
    use super::*;

    #[test]
    fn test_simultaneous_multimodal_workload() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        // LLM inference starts first (user-facing)
        let llm_alloc = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                10 * 1024 * 1024 * 1024,
            )
            .expect("LLM allocation failed");

        // Image generation starts in background
        let img_alloc = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Background,
                6 * 1024 * 1024 * 1024,
            )
            .expect("Image allocation failed");

        let state = arbiter.state();

        // Both should be running
        assert_eq!(state.active_llm_workloads, 1);
        assert_eq!(state.active_diffusion_workloads, 1);

        // LLM should have better quality (higher priority)
        assert!(llm_alloc.quality_target >= img_alloc.quality_target);

        // Total memory should be tracked
        assert_eq!(state.vram_used, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_workload_handoff() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(16)).expect("Failed");

        // Fill up with LLM
        let llm1 = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::High,
                12 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        // Try image gen - should get reduced quality due to pressure
        let img = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                3 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        assert!(img.quality_target < 1.0);

        // Release LLM
        arbiter.release_allocation(&llm1);

        // New diffusion should get better quality
        let img2 = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                3 * 1024 * 1024 * 1024,
            )
            .expect("Failed");

        assert!(img2.quality_target > img.quality_target);
    }

    #[test]
    fn test_stats_tracking() {
        let arbiter = Arbiter::new(ArbiterConfig::for_vram_gb(24)).expect("Failed");

        // Make some allocations
        let alloc1 = arbiter
            .request_allocation(
                WorkloadType::LlmInference,
                Priority::Normal,
                1024 * 1024 * 1024,
            )
            .expect("Failed");

        let _ = arbiter
            .request_allocation(
                WorkloadType::ImageGeneration,
                Priority::Normal,
                1024 * 1024 * 1024,
            )
            .expect("Failed");

        arbiter.release_allocation(&alloc1);

        let stats = arbiter.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.successful_allocations, 2);
        assert_eq!(stats.failed_allocations, 0);
    }
}
