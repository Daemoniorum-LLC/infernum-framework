# HoloTensor Optimization Roadmap

**TDD-Compliant Implementation Plan with Automated Quality Gates**

This roadmap upgrades HoloTensor to use the latest Haagenti APIs and optimizations,
following strict Test-Driven Development with automated quality gates.

## Overview

| Phase | Focus | Expected Gain | Quality Gate |
|-------|-------|---------------|--------------|
| 1 | Optimized Zstd Compression | 9.4x faster decompression | Benchmark regression test |
| 2 | QualityCurve Integration | Accurate quality scheduling | Quality prediction accuracy test |
| 3 | Entropy Fingerprinting | Skip incompressible data | Entropy detection accuracy test |
| 4 | Arena Allocator | +10-15% throughput | Memory allocation benchmark |
| 5 | GPU Decompression Pipeline | Zero-copy loading | End-to-end GPU benchmark |

---

## Phase 1: Optimized Zstd Compression

### Goal
Replace LZ4 default with Haagenti's optimized Zstd implementation (9.4x faster decompression).

### TDD Sequence

```
1. Write failing tests for Zstd compression/decompression
2. Implement ZstdCompressor integration
3. Write benchmark comparing LZ4 vs Zstd
4. Update default compression in HoloTensorHeader
5. Run quality gate
```

### Tests to Write First

```rust
// tests/holotensor_zstd_test.rs

#[test]
fn test_zstd_compression_roundtrip() {
    // Compress and decompress tensor data, verify lossless
}

#[test]
fn test_zstd_decompression_faster_than_lz4() {
    // Benchmark: Zstd decompress must be >= 2x faster than LZ4
}

#[test]
fn test_zstd_compression_ratio_acceptable() {
    // Compression ratio must be within 10% of LZ4
}

#[test]
fn test_holotensor_default_uses_zstd() {
    // New HoloTensorHeader should default to Zstd
}
```

### Quality Gate

```bash
# Phase 1 Quality Gate Script
cargo test --package abaddon --lib holotensor::tests::zstd
cargo bench --package abaddon -- zstd_vs_lz4

# PASS CRITERIA:
# - All Zstd tests pass
# - Zstd decompression >= 2x LZ4 speed
# - Compression ratio within 10% of LZ4
```

### Files to Modify

- `src/holotensor/mod.rs` - Default compression config
- `src/holotensor/converter.rs` - Use Zstd in conversion
- `src/holotensor/tiered_loading.rs` - Zstd decompression path
- `tests/holotensor_zstd_test.rs` - New test file

---

## Phase 2: QualityCurve Integration

### Goal
Replace simplistic sqrt(k/N) quality model with Haagenti's polynomial QualityCurve.

### TDD Sequence

```
1. Write failing tests for quality prediction accuracy
2. Import QualityCurve from haagenti::holotensor
3. Update QualityMetrics to use QualityCurve
4. Wire through provider and tiered_loading
5. Run quality gate
```

### Tests to Write First

```rust
// tests/holotensor_quality_test.rs

#[test]
fn test_quality_curve_lrdf_accurate() {
    // LRDF curve: 2/8 fragments should predict ~50% quality
    // 4/8 fragments should predict ~80% quality
}

#[test]
fn test_quality_curve_spectral_accurate() {
    // Spectral curve follows different coefficients
}

#[test]
fn test_quality_prediction_matches_reconstruction() {
    // Predicted quality should match actual reconstruction quality ±5%
}

#[test]
fn test_fragments_for_quality_correct() {
    // Given target quality, returns correct fragment count
}
```

### Quality Gate

```bash
# Phase 2 Quality Gate Script
cargo test --package abaddon --lib holotensor::tests::quality

# PASS CRITERIA:
# - Quality prediction error < 5% vs actual reconstruction
# - fragments_for_quality() returns optimal count
# - All encoding types have appropriate curves
```

### Files to Modify

- `src/holotensor/mod.rs` - Re-export QualityCurve
- `src/holotensor/provider.rs` - Use QualityCurve in QualityMetrics
- `src/holotensor/tiered_loading.rs` - Update quality estimation
- `tests/holotensor_quality_test.rs` - New test file

---

## Phase 3: Entropy Fingerprinting

### Goal
Add fast entropy detection (~100 cycles) to skip compression for incompressible data.

### TDD Sequence

```
1. Write failing tests for entropy detection
2. Import EntropyFingerprint from haagenti
3. Add entropy check before compression in converter
4. Add entropy-based path selection in tiered_loading
5. Run quality gate
```

### Tests to Write First

```rust
// tests/holotensor_entropy_test.rs

#[test]
fn test_entropy_detects_random_data() {
    // Random data should be flagged as incompressible
}

#[test]
fn test_entropy_detects_compressible_data() {
    // Repetitive/structured data should be flagged as compressible
}

#[test]
fn test_entropy_estimation_fast() {
    // Entropy estimation must complete in < 1ms for 1MB tensor
}

#[test]
fn test_incompressible_data_bypasses_compression() {
    // Incompressible tensors should use passthrough, not LRDF
}

#[test]
fn test_compression_decision_improves_throughput() {
    // Mixed workload with entropy detection should be faster
}
```

### Quality Gate

```bash
# Phase 3 Quality Gate Script
cargo test --package abaddon --lib holotensor::tests::entropy
cargo bench --package abaddon -- entropy_detection

# PASS CRITERIA:
# - Entropy detection accuracy > 95%
# - Entropy estimation < 1ms per MB
# - Mixed workload throughput improved by >= 10%
```

### Files to Modify

- `src/holotensor/converter.rs` - Add entropy check before encoding
- `src/holotensor/mod.rs` - Export entropy types
- `tests/holotensor_entropy_test.rs` - New test file

---

## Phase 4: Arena Allocator

### Goal
Add arena allocator for fragment allocations to reduce heap pressure.

### TDD Sequence

```
1. Write failing tests for arena allocation
2. Create ArenaAllocator wrapper for fragment data
3. Integrate into memory manager
4. Add arena pool to converter
5. Run quality gate
```

### Tests to Write First

```rust
// tests/holotensor_arena_test.rs

#[test]
fn test_arena_allocates_fragments() {
    // Allocate multiple fragments from arena
}

#[test]
fn test_arena_reuses_memory() {
    // After reset, arena reuses same memory region
}

#[test]
fn test_arena_reduces_allocations() {
    // Track allocator calls: arena should have fewer than Vec
}

#[test]
fn test_arena_throughput_improvement() {
    // Benchmark: arena allocation >= 10% faster than Vec
}

#[test]
fn test_arena_thread_safe() {
    // Multiple threads can allocate from thread-local arenas
}
```

### Quality Gate

```bash
# Phase 4 Quality Gate Script
cargo test --package abaddon --lib holotensor::tests::arena
cargo bench --package abaddon -- arena_allocation

# PASS CRITERIA:
# - All arena tests pass
# - Throughput improvement >= 10% vs standard allocation
# - No memory leaks (run under valgrind/miri)
```

### Files to Modify

- `src/holotensor/arena.rs` - New arena allocator module
- `src/holotensor/mod.rs` - Export arena types
- `src/holotensor/memory.rs` - Integrate arena
- `src/holotensor/converter.rs` - Use arena for fragments
- `tests/holotensor_arena_test.rs` - New test file

---

## Phase 5: GPU Decompression Pipeline

### Goal
Full integration with haagenti-cuda for zero-copy GPU decompression.

### TDD Sequence

```
1. Write failing tests for GPU decompression
2. Update GpuHoloContext to use latest haagenti-cuda
3. Implement zero-copy decompression path
4. Add GPU-resident tensor reconstruction
5. Run quality gate
```

### Tests to Write First

```rust
// tests/holotensor_gpu_test.rs

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_zstd_decompression() {
    // Decompress Zstd data on GPU
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_lrdf_reconstruction() {
    // Reconstruct LRDF tensor on GPU
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_zero_copy_path() {
    // Data never touches CPU RAM during decompression
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_faster_than_cpu() {
    // GPU path must be >= 5x faster than CPU for large tensors
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_fallback_to_cpu() {
    // Graceful fallback when GPU unavailable
}
```

### Quality Gate

```bash
# Phase 5 Quality Gate Script
cargo test --package abaddon --features cuda --lib holotensor::tests::gpu
cargo bench --package abaddon --features cuda -- gpu_decompression

# PASS CRITERIA:
# - All GPU tests pass
# - GPU decompression >= 5x CPU speed for 10MB+ tensors
# - Zero-copy verified (no host allocations in hot path)
# - Graceful fallback works
```

### Files to Modify

- `src/holotensor/tiered_loading.rs` - GPU decompression integration
- `src/gpu_holo.rs` - Update to latest haagenti-cuda
- `tests/holotensor_gpu_test.rs` - New test file

---

## Implementation Order

```
Phase 1 ─────► Phase 2 ─────► Phase 3 ─────► Phase 4 ─────► Phase 5
  Zstd          Quality        Entropy        Arena           GPU

Each phase:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Write failing tests                                      │
  │ 2. Implement minimum code to pass                           │
  │ 3. Refactor for clarity                                     │
  │ 4. Run quality gate                                         │
  │ 5. GATE MUST PASS before proceeding                         │
  └─────────────────────────────────────────────────────────────┘
```

---

## Quality Gate Runner

```bash
#!/bin/bash
# scripts/holotensor_quality_gates.sh

set -e

PHASE=${1:-all}

run_phase() {
    local phase=$1
    echo "═══════════════════════════════════════════════════════"
    echo "  Running Phase $phase Quality Gate"
    echo "═══════════════════════════════════════════════════════"

    case $phase in
        1)
            cargo test --package abaddon -F holotensor -- zstd
            cargo bench --package abaddon -- zstd_vs_lz4 --noplot
            ;;
        2)
            cargo test --package abaddon -F holotensor -- quality
            ;;
        3)
            cargo test --package abaddon -F holotensor -- entropy
            cargo bench --package abaddon -- entropy_detection --noplot
            ;;
        4)
            cargo test --package abaddon -F holotensor -- arena
            cargo bench --package abaddon -- arena_allocation --noplot
            ;;
        5)
            cargo test --package abaddon -F holotensor,cuda -- gpu
            cargo bench --package abaddon -F cuda -- gpu_decompression --noplot
            ;;
    esac

    echo "✓ Phase $phase Quality Gate PASSED"
}

if [ "$PHASE" = "all" ]; then
    for p in 1 2 3 4 5; do
        run_phase $p
    done
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  ALL QUALITY GATES PASSED"
    echo "═══════════════════════════════════════════════════════"
else
    run_phase $PHASE
fi
```

---

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Decompression speed | 1x (LZ4) | 9x (Zstd) | `cargo bench` |
| Quality prediction error | ~20% | <5% | Unit test |
| Incompressible detection | N/A | >95% accuracy | Unit test |
| Allocation throughput | 1x | 1.15x | `cargo bench` |
| GPU vs CPU speed | N/A | 5x+ for 10MB+ | `cargo bench` |
| Overall inference latency | baseline | -30% | E2E benchmark |

---

## Rollback Plan

Each phase is independently deployable. If a phase fails quality gates:

1. Revert commits for that phase
2. Feature-flag incomplete work
3. Continue using previous implementation
4. Investigate and retry

Feature flags:
- `holotensor-zstd` - Phase 1
- `holotensor-quality-curve` - Phase 2
- `holotensor-entropy` - Phase 3
- `holotensor-arena` - Phase 4
- `holotensor-gpu-pipeline` - Phase 5
