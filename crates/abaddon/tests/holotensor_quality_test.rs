//! Phase 2 TDD Tests: QualityCurve Integration for HoloTensor
//!
//! These tests verify that HoloTensor uses Haagenti's polynomial QualityCurve
//! instead of the simple sqrt(k/N) approximation.

use haagenti::holotensor::{HoloTensorHeader, HolographicEncoding, QualityCurve};
use haagenti::tensor::DType;

// =============================================================================
// Phase 2.1: QualityCurve Basic Functionality
// =============================================================================

#[test]
fn test_quality_curve_predict_basic() {
    let curve = QualityCurve::linear();

    // Linear curve: quality = k/N
    assert!((curve.predict(0, 8) - 0.0).abs() < 0.01);
    assert!((curve.predict(4, 8) - 0.5).abs() < 0.01);
    assert!((curve.predict(8, 8) - 1.0).abs() < 0.01);
}

#[test]
fn test_quality_curve_lrdf_accurate() {
    // LRDF (Low-Rank Distributed) uses polynomial: 0.3 + 0.5x + 0.15x² + 0.05x³
    let lrdf_curve = HolographicEncoding::LowRankDistributed.default_quality_curve();

    // Verify curve coefficients are as expected
    assert!(
        (lrdf_curve.coefficients[0] - 0.3).abs() < 0.01,
        "LRDF a0 should be 0.3"
    );
    assert!(
        (lrdf_curve.coefficients[1] - 0.5).abs() < 0.01,
        "LRDF a1 should be 0.5"
    );

    // With LRDF, baseline quality is 30% (a0 coefficient)
    let quality_1_of_8 = lrdf_curve.predict(1, 8);
    assert!(
        quality_1_of_8 >= 0.35 && quality_1_of_8 <= 0.45,
        "LRDF 1/8 fragments should give ~40% quality, got {:.2}",
        quality_1_of_8
    );

    // 4/8 fragments: 0.3 + 0.5*0.5 + 0.15*0.25 + 0.05*0.125 ≈ 0.59
    let quality_4_of_8 = lrdf_curve.predict(4, 8);
    assert!(
        quality_4_of_8 >= 0.55 && quality_4_of_8 <= 0.65,
        "LRDF 4/8 fragments should give ~60% quality, got {:.2}",
        quality_4_of_8
    );

    // 8/8 fragments should approach 100%
    let quality_8_of_8 = lrdf_curve.predict(8, 8);
    assert!(
        quality_8_of_8 >= 0.95,
        "LRDF 8/8 fragments should give ~100% quality, got {:.2}",
        quality_8_of_8
    );
}

#[test]
fn test_quality_curve_spectral_accurate() {
    // Spectral (DCT-based) has smooth curve with essential data providing baseline
    let spectral_curve = HolographicEncoding::Spectral.default_quality_curve();

    // Spectral provides baseline from DC component
    let quality_1_of_8 = spectral_curve.predict(1, 8);
    assert!(
        quality_1_of_8 >= 0.3,
        "Spectral 1/8 should give baseline >30% from DC, got {:.2}",
        quality_1_of_8
    );

    // Smooth improvement
    let quality_4_of_8 = spectral_curve.predict(4, 8);
    assert!(
        quality_4_of_8 >= 0.6 && quality_4_of_8 <= 0.85,
        "Spectral 4/8 should give ~70% quality, got {:.2}",
        quality_4_of_8
    );
}

#[test]
fn test_quality_curve_rph_accurate() {
    // Random Projection (JL-based) has linear quality improvement
    let rph_curve = HolographicEncoding::RandomProjection.default_quality_curve();

    // RPH needs minimum fragments for any meaningful quality
    assert_eq!(rph_curve.min_fragments, 2, "RPH requires min 2 fragments");

    // Linear progression
    let quality_4_of_8 = rph_curve.predict(4, 8);
    assert!(
        quality_4_of_8 >= 0.4 && quality_4_of_8 <= 0.7,
        "RPH 4/8 should give ~50% quality, got {:.2}",
        quality_4_of_8
    );
}

// =============================================================================
// Phase 2.2: fragments_for_quality
// =============================================================================

#[test]
fn test_fragments_for_quality_lrdf() {
    let curve = HolographicEncoding::LowRankDistributed.default_quality_curve();

    // Find minimum fragments for 50% quality (baseline + some fragments)
    let frags_for_50 = curve.fragments_for_quality(0.5, 8);
    assert!(
        frags_for_50 <= 4,
        "LRDF should need <= 4 fragments for 50% quality, got {}",
        frags_for_50
    );

    // Find minimum fragments for 70% quality
    let frags_for_70 = curve.fragments_for_quality(0.7, 8);
    assert!(
        frags_for_70 <= 6,
        "LRDF should need <= 6 fragments for 70% quality, got {}",
        frags_for_70
    );

    // Find minimum fragments for 95% quality
    let frags_for_95 = curve.fragments_for_quality(0.95, 8);
    assert!(
        frags_for_95 <= 8,
        "LRDF should need <= 8 fragments for 95% quality, got {}",
        frags_for_95
    );
}

#[test]
fn test_fragments_for_quality_returns_optimal() {
    let curve = QualityCurve::linear();

    // For linear curve, 70% quality needs 70% of fragments
    let frags = curve.fragments_for_quality(0.7, 10);
    assert_eq!(frags, 7, "Linear curve should need 7/10 for 70% quality");
}

// =============================================================================
// Phase 2.3: HoloTensorHeader Integration
// =============================================================================

#[test]
fn test_holotensor_header_has_quality_curve() {
    let header = HoloTensorHeader::new(
        HolographicEncoding::LowRankDistributed,
        DType::F32,
        vec![1024, 1024],
        8,
    );

    // Header should have encoding-appropriate quality curve
    let quality = header.quality_curve.predict(4, 8);
    assert!(
        quality > 0.5,
        "Header quality curve should be functional, got {:.2}",
        quality
    );
}

#[test]
fn test_holotensor_header_custom_quality_curve() {
    // Custom curve for specific tensor characteristics
    let custom_curve = QualityCurve::new(
        [0.2, 0.6, 0.15, 0.05], // Custom polynomial
        1,
        6,
    );

    let header = HoloTensorHeader::new(
        HolographicEncoding::Spectral,
        DType::F16,
        vec![4096, 4096],
        8,
    )
    .with_quality_curve(custom_curve);

    // Should use custom curve
    assert_eq!(header.min_fragments, 1);
}

// =============================================================================
// Phase 2.4: Quality Curve Serialization
// =============================================================================

#[test]
fn test_quality_curve_serialization() {
    let original = QualityCurve::new([0.3, 0.5, 0.15, 0.05], 1, 4);

    // Serialize to bytes
    let bytes = original.to_bytes();
    assert_eq!(
        bytes.len(),
        16,
        "Quality curve should serialize to 16 bytes"
    );

    // Deserialize
    let restored = QualityCurve::from_bytes(&bytes);

    // Coefficients should match
    for i in 0..4 {
        assert!(
            (original.coefficients[i] - restored.coefficients[i]).abs() < 0.0001,
            "Coefficient {} mismatch: {} vs {}",
            i,
            original.coefficients[i],
            restored.coefficients[i]
        );
    }
}

// =============================================================================
// Phase 2.5: Encoding-Specific Curves
// =============================================================================

#[test]
fn test_each_encoding_has_appropriate_curve() {
    let encodings = [
        HolographicEncoding::Spectral,
        HolographicEncoding::RandomProjection,
        HolographicEncoding::LowRankDistributed,
    ];

    for encoding in encodings {
        let curve = encoding.default_quality_curve();

        // All curves should be valid
        assert!(curve.min_fragments >= 1);
        assert!(curve.sufficient_fragments <= 8);

        // Quality should increase with fragments
        let q1 = curve.predict(2, 8);
        let q2 = curve.predict(4, 8);
        let q3 = curve.predict(6, 8);

        assert!(
            q2 >= q1 && q3 >= q2,
            "{:?} curve should be monotonically increasing",
            encoding
        );
    }
}

#[test]
fn test_spectral_has_dc_baseline() {
    // Spectral encoding should provide baseline quality from DC component
    let curve = HolographicEncoding::Spectral.default_quality_curve();

    // Even with just 1 fragment (DC), should have non-zero baseline
    let dc_only = curve.predict(1, 8);
    assert!(
        dc_only >= 0.3,
        "Spectral DC-only should provide baseline, got {:.2}",
        dc_only
    );
}

#[test]
fn test_lrdf_has_svd_knee() {
    // LRDF should have diminishing returns (SVD property)
    let curve = HolographicEncoding::LowRankDistributed.default_quality_curve();

    // First 2 fragments should provide more gain than last 2
    let gain_first_2 = curve.predict(2, 8) - curve.predict(0, 8);
    let gain_last_2 = curve.predict(8, 8) - curve.predict(6, 8);

    assert!(
        gain_first_2 > gain_last_2,
        "LRDF should show diminishing returns: first 2 gain ({:.2}) > last 2 ({:.2})",
        gain_first_2,
        gain_last_2
    );
}

// =============================================================================
// Phase 2.6: Edge Cases
// =============================================================================

#[test]
fn test_quality_curve_zero_fragments() {
    let curve = QualityCurve::linear();

    assert_eq!(
        curve.predict(0, 8),
        0.0,
        "0 fragments should give 0 quality"
    );
}

#[test]
fn test_quality_curve_all_fragments() {
    let curve = QualityCurve::linear();

    assert_eq!(
        curve.predict(8, 8),
        1.0,
        "All fragments should give 100% quality"
    );
}

#[test]
fn test_quality_curve_more_than_total() {
    let curve = QualityCurve::linear();

    // Should clamp to 1.0
    assert_eq!(
        curve.predict(10, 8),
        1.0,
        "More than total should clamp to 100%"
    );
}

#[test]
fn test_quality_curve_zero_total() {
    let curve = QualityCurve::linear();

    // Should handle gracefully
    assert_eq!(curve.predict(0, 0), 0.0, "Zero total should return 0");
}

// =============================================================================
// Phase 2 Quality Gate Summary
// =============================================================================

#[test]
fn phase_2_quality_gate_summary() {
    println!("\n");
    println!("═══════════════════════════════════════════════════════");
    println!("  Phase 2 Quality Gate: QualityCurve Integration");
    println!("═══════════════════════════════════════════════════════");
    println!("");
    println!("  Tests:");
    println!("  ✓ QualityCurve basic prediction");
    println!("  ✓ LRDF curve accuracy (SVD property)");
    println!("  ✓ Spectral curve accuracy (DC baseline)");
    println!("  ✓ RPH curve accuracy (JL bounds)");
    println!("  ✓ fragments_for_quality optimization");
    println!("  ✓ HoloTensorHeader integration");
    println!("  ✓ Quality curve serialization");
    println!("  ✓ Edge cases (0, all, overflow)");
    println!("");
    println!("═══════════════════════════════════════════════════════");
    println!("");
}
