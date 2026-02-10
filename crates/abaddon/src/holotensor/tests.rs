//! Tests for the holotensor inference module.
//!
//! TDD: These tests define the expected behavior of the holotensor bridge.

use super::*;

// ==================== MemoryTier Tests ====================

#[test]
fn test_memory_tier_ordering() {
    // VRAM is fastest, Network is slowest
    assert!(MemoryTier::Vram < MemoryTier::Ram);
    assert!(MemoryTier::Ram < MemoryTier::Nvme);
    assert!(MemoryTier::Nvme < MemoryTier::Network);
}

#[test]
fn test_memory_tier_latency() {
    // Verify relative latency estimates
    assert!(MemoryTier::Vram.typical_latency_ns() < MemoryTier::Ram.typical_latency_ns());
    assert!(MemoryTier::Ram.typical_latency_ns() < MemoryTier::Nvme.typical_latency_ns());
    assert!(MemoryTier::Nvme.typical_latency_ns() < MemoryTier::Network.typical_latency_ns());
}

#[test]
fn test_memory_tier_bandwidth() {
    // VRAM has highest bandwidth
    assert!(MemoryTier::Vram.typical_bandwidth_gbps() > MemoryTier::Ram.typical_bandwidth_gbps());
}

// ==================== FragmentLocation Tests ====================

#[test]
fn test_fragment_location_creation() {
    let loc = FragmentLocation::new(0, MemoryTier::Vram);
    assert_eq!(loc.fragment_id(), 0);
    assert_eq!(loc.tier(), MemoryTier::Vram);
    assert!(!loc.is_loading());
}

#[test]
fn test_fragment_location_promotion() {
    let mut loc = FragmentLocation::new(0, MemoryTier::Nvme);
    assert_eq!(loc.tier(), MemoryTier::Nvme);

    loc.promote_to(MemoryTier::Ram);
    assert_eq!(loc.tier(), MemoryTier::Ram);

    loc.promote_to(MemoryTier::Vram);
    assert_eq!(loc.tier(), MemoryTier::Vram);
}

// ==================== QualityMetrics Tests ====================

#[test]
fn test_quality_metrics_default() {
    let metrics = QualityMetrics::default();
    assert_eq!(metrics.current_quality(), 0.0);
    assert_eq!(metrics.target_quality(), 1.0);
    assert_eq!(metrics.fragments_loaded(), 0);
}

#[test]
fn test_quality_metrics_update() {
    let mut metrics = QualityMetrics::default();
    metrics.record_fragment_loaded(0.3);

    assert_eq!(metrics.fragments_loaded(), 1);
    assert!((metrics.current_quality() - 0.3).abs() < 0.001);
}

#[test]
fn test_quality_metrics_gap() {
    let mut metrics = QualityMetrics::with_target(0.95);
    metrics.record_fragment_loaded(0.6);

    let gap = metrics.quality_gap();
    assert!((gap - 0.35).abs() < 0.001);
}

// ==================== MemoryConfig Tests ====================

#[test]
fn test_memory_config_builder() {
    let config = MemoryConfig::builder()
        .vram_budget_mb(20_000)
        .ram_budget_mb(64_000)
        .nvme_cache_path("/tmp/holo_cache")
        .build();

    assert_eq!(config.vram_budget_bytes(), 20_000 * 1024 * 1024);
    assert_eq!(config.ram_budget_bytes(), 64_000 * 1024 * 1024);
}

#[test]
fn test_memory_config_auto_detect() {
    // Should not panic, returns reasonable defaults
    let config = MemoryConfig::auto_detect();
    assert!(config.vram_budget_bytes() > 0);
}

// ==================== StreamPriority Tests ====================

#[test]
fn test_stream_priority_ordering() {
    assert!(StreamPriority::Critical > StreamPriority::High);
    assert!(StreamPriority::High > StreamPriority::Normal);
    assert!(StreamPriority::Normal > StreamPriority::Background);
}

// ==================== HoloInferenceConfig Tests ====================

#[test]
fn test_holo_inference_config_defaults() {
    let config = HoloInferenceConfig::default();

    // Sensible defaults
    assert!(config.initial_quality() >= 0.3);
    assert!(config.initial_quality() <= 0.6);
    assert!(config.target_quality() >= 0.9);
}

#[test]
fn test_holo_inference_config_builder() {
    let config = HoloInferenceConfig::builder()
        .initial_quality(0.5)
        .target_quality(0.99)
        .enable_background_improvement(true)
        .build();

    assert!((config.initial_quality() - 0.5).abs() < 0.001);
    assert!((config.target_quality() - 0.99).abs() < 0.001);
    assert!(config.background_improvement_enabled());
}

// ==================== ProgressiveWeightProvider Tests ====================

#[test]
fn test_progressive_weight_provider_quality_curve() {
    // Verify the quality curve matches haagenti's implementation
    use haagenti::holotensor::HolographicEncoding;

    let curve = HolographicEncoding::Spectral.default_quality_curve();

    // Fragment 0 should give ~60% quality for Spectral
    let q0 = curve.predict(1, 8);
    assert!(
        q0 >= 0.5 && q0 <= 0.7,
        "Expected ~60% from fragment 0, got {}",
        q0
    );

    // All fragments should give ~100%
    let q_all = curve.predict(8, 8);
    assert!(
        q_all >= 0.95,
        "Expected ~100% from all fragments, got {}",
        q_all
    );
}

// ==================== HoloModelMetadata Tests ====================

#[test]
fn test_holo_model_metadata() {
    let metadata = HoloModelMetadata {
        model_id: "llama-405b".to_string(),
        total_parameters: 405_000_000_000,
        total_fragments: 8,
        encoding: haagenti::holotensor::HolographicEncoding::Spectral,
        layers: 126,
        num_layers: 126,
        hidden_size: 16384,
        num_heads: 128,
        num_kv_heads: 8,
        original_size: 800_000_000_000, // 800GB
        hct_size: 100_000_000_000,      // 100GB compressed
        verified_quality: 0.98,
    };

    assert_eq!(metadata.model_id, "llama-405b");
    assert_eq!(metadata.total_fragments, 8);
}

// ==================== ConversionConfig Tests ====================

#[test]
fn test_conversion_config_defaults() {
    let config = ConversionConfig::default();

    // Default is optimized for 405B models with progressive loading
    assert_eq!(config.num_fragments, 32);
    assert!(matches!(
        config.encoding,
        haagenti::holotensor::HolographicEncoding::LowRankDistributed
    ));
}

#[test]
fn test_conversion_config_high_quality() {
    // High quality config should use LowRankDistributed encoding
    let config = ConversionConfig::high_quality();

    assert!(matches!(
        config.encoding,
        haagenti::holotensor::HolographicEncoding::LowRankDistributed
    ));
}

// ==================== Integration Tests ====================

#[cfg(test)]
mod integration {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let config = MemoryConfig::builder()
            .vram_budget_mb(8_000)
            .ram_budget_mb(32_000)
            .build();

        let manager = HoloMemoryManager::new(config);

        assert!(manager.available_vram() > 0);
        assert!(manager.available_ram() > 0);
    }

    #[test]
    fn test_stream_manager_creation() {
        let stream_manager = StreamManager::new(4); // 4 concurrent streams

        assert_eq!(stream_manager.max_concurrent_streams(), 4);
        assert_eq!(stream_manager.active_streams(), 0);
    }
}
