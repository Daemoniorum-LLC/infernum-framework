# GPU Codec Pipeline TDD Roadmap

**Version:** 0.1.0
**Status:** Test Specification
**Date:** 2026-02-03
**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md v0.1.0

---

## Philosophy

Tests are crystallized understanding, not coverage theater.

Each test in this roadmap exists because it captures something we *must* know is true
about the GPU codec pipeline. If a test doesn't protect a guarantee made by the spec,
it doesn't belong here.

We test at **trust boundaries** — the edges where assumptions meet reality:
- Packed data enters a kernel (format correctness)
- A kernel promises lossless output (compression boundary)
- A kernel promises mathematically correct output (quantization boundary)
- A fused kernel promises equivalence to sequential execution (fusion boundary)
- A feature gate promises isolation of broken code (stability boundary)

We prefer **property tests** over example tests where the property is the point.
"GPU LZ4 output is byte-identical to CPU LZ4 output" is a property.
"Decompressing [0x04, 0x40, 0x01, 0x02, 0x03, 0x04] gives [1, 2, 3, 4]" is an example.

---

## 0. Test Infrastructure

**Purpose:** Shared helpers that eliminate boilerplate and enable property testing.
These are prerequisites for all subsequent sections.

### 0.1 CUDA Test Context

Consolidate the three duplicate `cuda_available()` functions (`gpu_lz4.rs:1568`,
`gpu_dequant.rs:646`, `gpu_fused.rs:1135`) into shared infrastructure.

```rust
// crates/abaddon/src/test_helpers.rs (or inline in each module)

/// Returns true if CUDA device 0 is available for testing.
/// Used for graceful test skipping on CI without GPUs.
pub fn cuda_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}

/// Macro for CUDA-dependent tests. Skips gracefully if no GPU.
macro_rules! cuda_test {
    ($name:ident, $body:expr) => {
        #[test]
        #[cfg(feature = "cuda")]
        fn $name() {
            if !cuda_available() {
                eprintln!("Skipping {}: no CUDA device", stringify!($name));
                return;
            }
            $body
        }
    };
}
```

### 0.2 Proptest Strategies

```rust
use proptest::prelude::*;

/// Generate random INT4 quantized data with valid scale/zero-point structure.
fn arb_int4_block(block_size: usize) -> impl Strategy<Value = (Vec<u8>, Vec<half::f16>, Vec<i8>)> {
    let num_bytes = (block_size + 1) / 2;
    (
        prop::collection::vec(any::<u8>(), num_bytes),           // packed nibbles
        prop::collection::vec(0.001f32..10.0f32, 1)              // scale
            .prop_map(|v| v.into_iter().map(half::f16::from_f32).collect()),
        prop::collection::vec(-8i8..=7i8, 1),                    // zero point
    )
}

/// Generate random bytes that survive LZ4 round-trip.
/// Creates data, compresses with CPU LZ4, returns (compressed, original).
fn arb_lz4_roundtrip_data(max_size: usize) -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    prop::collection::vec(any::<u8>(), 1..max_size)
        .prop_map(|original| {
            let compressed = lz4_compress_cpu(&original);
            (compressed, original)
        })
}

/// Generate random FP8 E4M3 bytes (valid bit patterns only).
fn arb_fp8_e4m3() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..=0xFE, 1..1024)  // 0xFF is NaN, skip
}
```

### 0.3 CPU Reference Wrappers

```rust
/// CPU LZ4 decompression reference (wraps haagenti).
fn lz4_decompress_cpu(compressed: &[u8], expected_size: usize) -> Vec<u8> {
    let decompressor = haagenti::Lz4Decompressor::new();
    decompressor.decompress(compressed, expected_size).unwrap()
}

/// CPU INT4 symmetric dequantization reference (wraps quantize.rs).
fn int4_dequant_cpu(packed: &[u8], scales: &[half::f16], block_size: usize) -> Vec<f32> {
    let unpacked = crate::quantize::unpack_int4_signed(packed, packed.len() * 2);
    unpacked.iter().enumerate().map(|(i, &q)| {
        let scale = scales[i / block_size].to_f32();
        (q as f32) * scale
    }).collect()
}

/// CPU INT8 symmetric dequantization reference.
fn int8_dequant_cpu(data: &[u8], scale: half::f16) -> Vec<f32> {
    let scale_f32 = scale.to_f32();
    data.iter().map(|&b| (b as i8 as f32) * scale_f32).collect()
}
```

---

## 1. Constant Integrity

**Trust Boundary:** Shared constants define the contract between the quantizer (writer)
and the dequantizer (reader). If they disagree, data is silently misinterpreted.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §2

**Phase:** 1

### 1.1 Block Size Agreement

The quantizer writes INT4 data with `DEFAULT_BLOCK_SIZE = 128`. Every reader must
use the same value.

```rust
#[test]
fn test_int4_block_size_matches_quantizer() {
    // This test catches DD-1: hct.rs uses Q4_BLOCK_SIZE=32 but quantizer uses 128.
    use crate::gpu_dequant::INT4_BLOCK_SIZE;
    use crate::quantize::DEFAULT_BLOCK_SIZE;

    assert_eq!(
        INT4_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
        "GPU dequant block size ({}) must match CPU quantizer block size ({})",
        INT4_BLOCK_SIZE, DEFAULT_BLOCK_SIZE
    );
}
```

**Compile-time enforcement:**

```rust
// In gpu_dequant.rs or a shared constants module
const _: () = assert!(
    crate::gpu_dequant::cuda::INT4_BLOCK_SIZE == crate::quantize::DEFAULT_BLOCK_SIZE,
    "INT4_BLOCK_SIZE must equal DEFAULT_BLOCK_SIZE"
);
```

### 1.2 HCT Reader Block Size (DD-1 Regression)

```rust
#[test]
fn test_hct_int4_dequant_uses_correct_block_size() {
    // Create INT4 data with block_size=128 (quantizer default)
    let values: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let data = create_int4_data(&values, 128);

    // Dequantize with block_size=128 (correct) and block_size=32 (wrong)
    let correct = dequantize_int4(&data, 256, 128);
    let wrong = dequantize_int4(&data, 256, 32);

    // The wrong block size will read garbage scales for blocks 1-3
    // because it thinks there are 8 scale values (256/32) but only 2 exist (256/128)
    assert_ne!(correct, wrong, "Different block sizes must produce different results");

    // Verify the correct result approximates original values
    for (orig, deq) in values.iter().zip(correct.iter()) {
        let max_error = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 7.0;
        assert!(
            (orig - deq).abs() < max_error + 0.1,
            "Dequantized value {} too far from original {}", deq, orig
        );
    }
}
```

### 1.3 Cross-Validation: HCT CPU Dequant vs Quantizer Dequant

```rust
#[test]
fn test_hct_dequant_matches_quantizer_dequant() {
    // Quantize using quantize.rs
    let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let quantizer = Quantizer::new(QuantizeConfig {
        format: QuantizeFormat::Int4Symmetric,
        block_size: 128,
        activation_aware: false,
    });
    let tensor = Tensor::from_vec(original.clone(), &[256], &Device::Cpu).unwrap();
    let quantized = quantizer.quantize(&tensor).unwrap();

    // Dequantize using quantize.rs (ground truth)
    let reference = quantizer.dequantize(&quantized).unwrap();
    let ref_values: Vec<f32> = reference.to_vec1().unwrap();

    // Dequantize using hct.rs path (must agree after DD-1 fix)
    let hct_values = dequantize_int4(&quantized.data, 256, 128);

    for (i, (r, h)) in ref_values.iter().zip(hct_values.iter()).enumerate() {
        assert!(
            (r - h).abs() < 1e-6,
            "Mismatch at index {}: quantizer={}, hct={}", i, r, h
        );
    }
}
```

---

## 2. Data Format Correctness

**Trust Boundary:** Packed data formats define how bytes map to values. Misinterpreting
nibble order, scale position, or byte alignment corrupts all downstream results.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §3

**Phase:** 1

### 2.1 INT4 Nibble Packing Convention

```rust
#[test]
fn test_int4_nibble_packing_little_endian() {
    // Byte 0x53 should unpack to: low nibble=3 (value[0]), high nibble=5 (value[1])
    let packed = vec![0x53u8];
    let unpacked = unpack_int4_unsigned(&packed, 2);
    assert_eq!(unpacked[0], 3, "Low nibble must be first value");
    assert_eq!(unpacked[1], 5, "High nibble must be second value");
}

// Property: pack(unpack(data)) == data
proptest! {
    #[test]
    fn test_int4_pack_unpack_roundtrip(data in prop::collection::vec(any::<u8>(), 1..128)) {
        let num_values = data.len() * 2;
        let unpacked = unpack_int4_unsigned(&data, num_values);
        let repacked = pack_int4(&unpacked);
        prop_assert_eq!(&repacked, &data);
    }
}
```

### 2.2 Scale Layout

```rust
#[test]
fn test_int4_scale_layout_scales_before_data() {
    // HCT INT4 layout: [all FP16 scales][all packed INT4 data]
    let num_values = 256;
    let block_size = 128;
    let num_blocks = (num_values + block_size - 1) / block_size;
    let scales_bytes = num_blocks * 2;     // 2 bytes per FP16 scale
    let packed_bytes = (num_values + 1) / 2; // 2 nibbles per byte

    let expected_total = scales_bytes + packed_bytes;
    let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let data = create_int4_data(&values, block_size);

    assert_eq!(
        data.len(), expected_total,
        "INT4 data should be {} scale bytes + {} packed bytes = {} total",
        scales_bytes, packed_bytes, expected_total
    );
}
```

---

## 3. LZ4 Lossless Guarantee

**Trust Boundary:** LZ4 decompression on GPU MUST produce byte-identical output to
CPU decompression. This is the fundamental lossless guarantee. Any deviation means
model weights are corrupted.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §6.1

**Phase:** 1 (cross-validation), 3 (warp kernel fix)

### 3.1 GPU vs CPU Byte-Exact Match

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_lz4_gpu_matches_cpu_literals_only() {
    if !cuda_available() { return; }

    let original = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let compressed = lz4_compress_cpu(&original);

    // GPU decompress (K1: single block)
    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();
    let gpu_result = ctx.decompress_block(&compressed, original.len()).unwrap();
    let gpu_host = ctx.device().dtoh_sync_copy(&gpu_result).unwrap();

    // CPU decompress (haagenti reference)
    let cpu_result = lz4_decompress_cpu(&compressed, original.len());

    assert_eq!(gpu_host, cpu_result, "GPU LZ4 must match CPU LZ4 byte-for-byte");
    assert_eq!(gpu_host, original, "Both must match original data");
}

#[test]
#[cfg(feature = "cuda")]
fn test_lz4_gpu_matches_cpu_with_matches() {
    if !cuda_available() { return; }

    // Data with repeated patterns (triggers LZ4 match sequences)
    let original: Vec<u8> = (0..1024).map(|i| (i % 13) as u8).collect();
    let compressed = lz4_compress_cpu(&original);

    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();
    let gpu_result = ctx.decompress_block(&compressed, original.len()).unwrap();
    let gpu_host = ctx.device().dtoh_sync_copy(&gpu_result).unwrap();

    let cpu_result = lz4_decompress_cpu(&compressed, original.len());
    assert_eq!(gpu_host, cpu_result);
}
```

### 3.2 Property: GPU LZ4 is Byte-Exact for Arbitrary Data

```rust
// Property test: GPU LZ4 matches CPU LZ4 for all compressible inputs
proptest! {
    #[test]
    #[cfg(feature = "cuda")]
    fn test_lz4_gpu_cpu_equivalence(original in prop::collection::vec(any::<u8>(), 1..4096)) {
        if !cuda_available() { return Ok(()); }

        let compressed = lz4_compress_cpu(&original);
        let mut ctx = GpuLz4Context::new(0).unwrap();
        ctx.load_kernels().unwrap();

        let gpu_result = ctx.decompress_block(&compressed, original.len()).unwrap();
        let gpu_host = ctx.device().dtoh_sync_copy(&gpu_result).unwrap();

        prop_assert_eq!(gpu_host, original, "GPU LZ4 output must be byte-exact");
    }
}
```

### 3.3 Parallel Kernel (K2) vs Single-Block Kernel (K1)

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_lz4_parallel_matches_single_block() {
    if !cuda_available() { return; }

    let blocks: Vec<Vec<u8>> = (0..8).map(|i| {
        (0..256).map(|j| ((i * 37 + j * 13) % 256) as u8).collect()
    }).collect();

    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();

    // Decompress each block individually with K1
    let single_results: Vec<Vec<u8>> = blocks.iter().map(|block| {
        let compressed = lz4_compress_cpu(block);
        let result = ctx.decompress_block(&compressed, block.len()).unwrap();
        ctx.device().dtoh_sync_copy(&result).unwrap()
    }).collect();

    // Decompress all blocks in parallel with K2
    let compressed_blocks: Vec<Vec<u8>> = blocks.iter()
        .map(|b| lz4_compress_cpu(b))
        .collect();
    let parallel_results = ctx.decompress_blocks_parallel(&compressed_blocks).unwrap();

    for (i, (single, parallel)) in single_results.iter().zip(parallel_results.iter()).enumerate() {
        assert_eq!(single, parallel, "Block {} differs between K1 and K2", i);
    }
}
```

### 3.4 Warp Kernel (K3) Correctness (Phase 3)

```rust
#[test]
#[cfg(feature = "cuda-experimental")]
fn test_lz4_warp_matches_parallel() {
    if !cuda_available() { return; }

    let blocks: Vec<Vec<u8>> = (0..4).map(|i| {
        (0..512).map(|j| ((i * 41 + j * 7) % 256) as u8).collect()
    }).collect();

    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();

    let compressed_blocks: Vec<Vec<u8>> = blocks.iter()
        .map(|b| lz4_compress_cpu(b))
        .collect();

    let parallel = ctx.decompress_blocks_parallel(&compressed_blocks).unwrap();
    let warp = ctx.decompress_blocks_warp(&compressed_blocks).unwrap();

    for (i, (p, w)) in parallel.iter().zip(warp.iter()).enumerate() {
        assert_eq!(p, w, "Block {} differs between K2 (parallel) and K3 (warp)", i);
    }
}

// Property: warp kernel matches parallel kernel for all inputs
proptest! {
    #[test]
    #[cfg(feature = "cuda-experimental")]
    fn test_lz4_warp_equivalence(
        blocks in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 64..1024),
            1..8
        )
    ) {
        if !cuda_available() { return Ok(()); }

        let mut ctx = GpuLz4Context::new(0).unwrap();
        ctx.load_kernels().unwrap();

        let compressed: Vec<Vec<u8>> = blocks.iter()
            .map(|b| lz4_compress_cpu(b))
            .collect();

        let parallel = ctx.decompress_blocks_parallel(&compressed).unwrap();
        let warp = ctx.decompress_blocks_warp(&compressed).unwrap();

        for (i, (p, w)) in parallel.iter().zip(warp.iter()).enumerate() {
            prop_assert_eq!(p, w, "Block {} mismatch", i);
        }
    }
}
```

---

## 4. Quantization Math

**Trust Boundary:** Dequantization kernels make a mathematical promise: the output
is the correct application of `value * scale` (or `(value - zero_point) * scale`),
rounded to F16. Any deviation corrupts model weights.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §6.2, §6.3

**Phase:** 1 (cross-validation), 4 (property tests)

### 4.1 GPU INT4 Dequant vs CPU Reference

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_int4_gpu_matches_cpu_reference() {
    if !cuda_available() { return; }

    // Create known INT4 data using quantize.rs
    let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let tensor = Tensor::from_vec(original.clone(), &[256], &Device::Cpu).unwrap();
    let quantizer = Quantizer::new(QuantizeConfig {
        format: QuantizeFormat::Int4Symmetric,
        block_size: 128,
        activation_aware: false,
    });
    let quantized = quantizer.quantize(&tensor).unwrap();

    // CPU dequantize (ground truth)
    let cpu_result = quantizer.dequantize(&quantized).unwrap();
    let cpu_values: Vec<f32> = cpu_result.to_vec1().unwrap();

    // GPU dequantize
    let mut ctx = GpuDequantContext::new(0).unwrap();
    ctx.load_int4_kernel().unwrap();
    let gpu_tensor = ctx.dequantize_int4(
        &quantized.data,
        &quantized.scales,
        &quantized.zero_points.unwrap_or_default(),
        quantized.num_values,
    ).unwrap();
    let gpu_values: Vec<f32> = /* copy to host and convert F16 → F32 */;

    for (i, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
        let tolerance = quantized.scales[i / 128].to_f32() / 2.0;
        assert!(
            (cpu - gpu).abs() < tolerance + 1e-4,
            "INT4 dequant mismatch at index {}: cpu={}, gpu={}", i, cpu, gpu
        );
    }
}
```

### 4.2 GPU INT8 Dequant vs CPU Reference

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_int8_gpu_matches_cpu_reference() {
    if !cuda_available() { return; }

    let data: Vec<u8> = (0..256).map(|i| i as u8).collect(); // -128..127 as i8
    let scale = half::f16::from_f32(0.05);

    // CPU reference
    let cpu_values: Vec<f32> = data.iter()
        .map(|&b| (b as i8 as f32) * scale.to_f32())
        .collect();

    // GPU dequantize
    let mut ctx = GpuDequantContext::new(0).unwrap();
    ctx.load_int8_kernel().unwrap();
    let gpu_tensor = ctx.dequantize_int8(&data, scale, data.len()).unwrap();
    let gpu_values: Vec<f32> = /* copy to host and convert F16 → F32 */;

    for (i, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-3,
            "INT8 dequant mismatch at index {}: cpu={}, gpu={}", i, cpu, gpu
        );
    }
}
```

### 4.3 Property: Dequant is 0-ULP in F16

```rust
// Property: GPU dequant produces bit-exact F16 values matching the formula
proptest! {
    #[test]
    #[cfg(feature = "cuda")]
    fn test_int4_dequant_zero_ulp(
        nibble in 0u8..16u8,
        scale_f32 in 0.001f32..10.0f32,
        zero_point in -8i8..=7i8,
    ) {
        if !cuda_available() { return Ok(()); }

        let scale = half::f16::from_f32(scale_f32);
        let packed = vec![nibble]; // single nibble, zero-padded

        // Expected: compute in F32, round to F16
        let expected = half::f16::from_f32(
            (nibble as i8 - zero_point) as f32 * scale.to_f32()
        );

        let mut ctx = GpuDequantContext::new(0).unwrap();
        ctx.load_int4_kernel().unwrap();
        let result = ctx.dequantize_int4_block(&packed, scale, zero_point as i32, 1).unwrap();
        let gpu_value: half::f16 = /* copy to host */;

        prop_assert_eq!(
            gpu_value.to_bits(), expected.to_bits(),
            "0-ULP violation: gpu={:?} expected={:?}", gpu_value, expected
        );
    }
}
```

### 4.4 FP8 Conversion Correctness

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_fp8_e4m3_known_values() {
    if !cuda_available() { return; }

    // Known FP8 E4M3 → F32 conversions
    let test_cases: Vec<(u8, f32)> = vec![
        (0x00, 0.0),       // +0
        (0x80, -0.0),      // -0
        (0x38, 1.0),       // 1.0 in E4M3
        (0x3C, 1.5),       // 1.5 in E4M3
        (0x40, 2.0),       // 2.0
        (0x7E, 448.0),     // max positive
    ];

    let converter = GpuDtypeConverter::new(/* ... */).unwrap();
    let input: Vec<u8> = test_cases.iter().map(|(b, _)| *b).collect();
    let result = converter.fp8_e4m3_to_f32_host(&input).unwrap();

    for (i, ((_, expected), actual)) in test_cases.iter().zip(result.iter()).enumerate() {
        assert_eq!(
            *actual, *expected,
            "FP8 E4M3 conversion wrong at index {}: byte=0x{:02X}, expected={}, got={}",
            i, test_cases[i].0, expected, actual
        );
    }
}

// Property: FP8 → F32 is exact (no rounding error possible)
proptest! {
    #[test]
    #[cfg(feature = "cuda")]
    fn test_fp8_e4m3_conversion_is_exact(byte in 0u8..=0xFE) {
        if !cuda_available() { return Ok(()); }

        let converter = GpuDtypeConverter::new(/* ... */).unwrap();
        let gpu_result = converter.fp8_e4m3_to_f32_host(&[byte]).unwrap();
        let cpu_result = fp8_e4m3_to_f32_software(byte);

        prop_assert_eq!(
            gpu_result[0].to_bits(), cpu_result.to_bits(),
            "FP8 E4M3 0x{:02X}: gpu={}, cpu={}", byte, gpu_result[0], cpu_result
        );
    }
}
```

---

## 5. Fusion Equivalence

**Trust Boundary:** Fused kernels promise identical output to the sequential pipeline.
The optimization is purely in performance (fewer kernel launches, no intermediate buffer).
Any semantic difference is a bug.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §6.5

**Phase:** 4

### 5.1 Fused LZ4+INT4 == Sequential LZ4 → INT4

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_fused_lz4_int4_matches_sequential() {
    if !cuda_available() { return; }

    // Create INT4-quantized data, then LZ4-compress it
    let original: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
    let (packed_int4, scale, zero_point) = quantize_int4_block(&original);
    let compressed = lz4_compress_cpu(&packed_int4);

    // Sequential: LZ4 decompress → INT4 dequant
    let mut lz4_ctx = GpuLz4Context::new(0).unwrap();
    lz4_ctx.load_kernels().unwrap();
    let decompressed = lz4_ctx.decompress_block(&compressed, packed_int4.len()).unwrap();
    // ... then dequantize decompressed data ...

    // Fused: single kernel
    let mut fused_ctx = GpuFusedContext::new(0).unwrap();
    fused_ctx.load_lz4_int4_kernel().unwrap();
    let fused_result = fused_ctx.fused_lz4_int4_block(
        &compressed, 128, scale, zero_point
    ).unwrap();

    let sequential_host: Vec<u16> = /* copy sequential result to host */;
    let fused_host: Vec<u16> = /* copy fused result to host */;

    // Bit-identical: same F16 bit patterns
    assert_eq!(
        sequential_host, fused_host,
        "Fused kernel must produce bit-identical output to sequential pipeline"
    );
}
```

### 5.2 Fused LZ4+INT8 == Sequential LZ4 → INT8

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_fused_lz4_int8_matches_sequential() {
    if !cuda_available() { return; }

    let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let scale = half::f16::from_f32(0.03);
    let compressed = lz4_compress_cpu(&data);

    // Sequential pipeline
    // ... LZ4 decompress → INT8 dequant ...

    // Fused pipeline
    let mut fused_ctx = GpuFusedContext::new(0).unwrap();
    fused_ctx.load_lz4_int8_kernel().unwrap();
    let fused_result = fused_ctx.fused_lz4_int8_block(
        &compressed, data.len(), scale
    ).unwrap();

    // Compare bit patterns (F16 raw bits)
    // assert_eq!(sequential_bits, fused_bits);
}
```

### 5.3 Property: Fusion Preserves Bit-Exactness

```rust
proptest! {
    #[test]
    #[cfg(feature = "cuda")]
    fn test_fusion_bit_identical(
        data in prop::collection::vec(any::<u8>(), 2..512),
        scale_f32 in 0.001f32..5.0f32,
        zero_point in -8i8..=7i8,
    ) {
        if !cuda_available() { return Ok(()); }

        // Quantize data as INT4 packed
        let packed_int4 = pack_as_int4(&data);
        let compressed = lz4_compress_cpu(&packed_int4);
        let scale = half::f16::from_f32(scale_f32);

        // Sequential vs fused
        let sequential = sequential_lz4_int4(&compressed, packed_int4.len(), scale, zero_point);
        let fused = fused_lz4_int4(&compressed, packed_int4.len(), scale, zero_point);

        prop_assert_eq!(sequential, fused, "Fusion must be bit-identical to sequential");
    }
}
```

---

## 6. HoloTensor Reconstruction

**Trust Boundary:** Holographic tensor reconstruction converts compressed spectral/LRDF/RPH
coefficients back to dense tensors. Reconstruction error is bounded by the truncation
rank, but the GPU implementation must match the mathematical definition exactly.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §6.6

**Phase:** 4

**Current state:** gpu_holo.rs has 4,761 lines of code and **zero tests**. Four kernels
are stubs (return immediately). This section adds foundational coverage.

### 6.1 Spectral (IDCT) Reconstruction

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_spectral_idct_1d_rows_known_answer() {
    if !cuda_available() { return; }

    // Known IDCT: DC coefficient only → flat row
    let width = 4;
    let height = 1;
    let coeffs = vec![2.0f32, 0.0, 0.0, 0.0]; // Only DC component

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let result = ctx.spectral_idct_1d_rows(&coeffs, width, height).unwrap();
    let host: Vec<f32> = result.to_host().unwrap();

    // DC-only IDCT should produce constant value = DC / sqrt(N) * sqrt(2)
    let expected = coeffs[0] / (width as f32).sqrt();
    for (i, &val) in host.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-4,
            "IDCT DC-only: index {} expected {}, got {}", i, expected, val
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_spectral_accumulate_sparse_coefficients() {
    if !cuda_available() { return; }

    let buffer_size = 16;
    let indices = vec![0u32, 3, 7, 15]; // Sparse positions
    let values = vec![1.0f32, 2.0, 3.0, 4.0];

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let result = ctx.spectral_accumulate(&indices, &values, buffer_size).unwrap();
    let host: Vec<f32> = result.to_host().unwrap();

    // Verify sparse accumulation: only indexed positions should be non-zero
    assert!((host[0] - 1.0).abs() < 1e-6);
    assert!((host[3] - 2.0).abs() < 1e-6);
    assert!((host[7] - 3.0).abs() < 1e-6);
    assert!((host[15] - 4.0).abs() < 1e-6);

    // All other positions should be zero
    for (i, &val) in host.iter().enumerate() {
        if ![0, 3, 7, 15].contains(&i) {
            assert!((val).abs() < 1e-6, "Non-indexed position {} should be 0, got {}", i, val);
        }
    }
}
```

### 6.2 LRDF (Low-Rank) Reconstruction

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_lrdf_outer_product_rank1() {
    if !cuda_available() { return; }

    // Rank-1: σ * u * v^T
    let u = vec![1.0f32, 2.0, 3.0];  // 3 rows
    let v = vec![4.0f32, 5.0];       // 2 cols
    let sigma = 2.0f32;

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_lrdf_kernel().unwrap();

    let result = ctx.lrdf_outer_product(&u, &v, sigma, 3, 2).unwrap();
    let host: Vec<f32> = result.to_host().unwrap();

    // Expected: σ * u_i * v_j
    let expected = vec![
        2.0*1.0*4.0, 2.0*1.0*5.0,  // row 0
        2.0*2.0*4.0, 2.0*2.0*5.0,  // row 1
        2.0*3.0*4.0, 2.0*3.0*5.0,  // row 2
    ];

    for (i, (e, g)) in expected.iter().zip(host.iter()).enumerate() {
        assert!(
            (e - g).abs() < 1e-4,
            "LRDF outer product mismatch at {}: expected={}, got={}", i, e, g
        );
    }
}
```

### 6.3 RPH (Random Projection Hash) Reconstruction

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_rph_accumulate_deterministic() {
    if !cuda_available() { return; }

    let proj_dim = 8;
    let output_dim = 4;
    let seed = 42u64;
    let projection = vec![1.0f32; proj_dim]; // uniform projection

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_rph_kernel().unwrap();

    // Two calls with same seed must produce identical results
    let result1 = ctx.rph_accumulate(&projection, output_dim, seed).unwrap();
    let result2 = ctx.rph_accumulate(&projection, output_dim, seed).unwrap();

    let host1: Vec<f32> = result1.to_host().unwrap();
    let host2: Vec<f32> = result2.to_host().unwrap();

    assert_eq!(host1, host2, "RPH with same seed must be deterministic");
}
```

### 6.4 Stub Kernel Detection

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_stub_kernels_are_identified() {
    // These kernels are stubs (return immediately, produce no output).
    // Tests verify they don't crash and produce zero/empty output,
    // documenting that they are NOT functional.

    if !cuda_available() { return; }

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_all_kernels().unwrap();

    // holo_spectral_idct_1d_cols — stub
    let input = vec![1.0f32; 16];
    let result = ctx.spectral_idct_1d_cols(&input, 4, 4).unwrap();
    let host: Vec<f32> = result.to_host().unwrap();
    // Stub kernels produce unchanged/zero output
    assert!(host.iter().all(|&v| v == 0.0),
        "idct_1d_cols is a stub — should produce zeros");

    // holo_spectral_idct_2d — stub
    // holo_rph_generate_projection — stub
    // holo_lrdf_outer_product_batched — stub
}
```

### 6.5 Utility Kernel Correctness

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_holo_f32_to_f16_conversion() {
    if !cuda_available() { return; }

    let input = vec![1.0f32, 0.5, -1.0, 0.0, 100.0, -0.001];
    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_all_kernels().unwrap();

    let result = ctx.f32_to_f16(&input).unwrap();
    let host: Vec<u16> = /* copy to host */;

    for (i, (&f32_val, &f16_bits)) in input.iter().zip(host.iter()).enumerate() {
        let expected = half::f16::from_f32(f32_val);
        assert_eq!(
            f16_bits, expected.to_bits(),
            "F32→F16 conversion wrong at {}: input={}, got=0x{:04X}, expected=0x{:04X}",
            i, f32_val, f16_bits, expected.to_bits()
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_holo_scale_values() {
    if !cuda_available() { return; }

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let scale = 0.5f32;

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_all_kernels().unwrap();

    let result = ctx.scale_values(&data, scale).unwrap();
    let host: Vec<f32> = result.to_host().unwrap();

    let expected = vec![0.5, 1.0, 1.5, 2.0];
    for (i, (e, g)) in expected.iter().zip(host.iter()).enumerate() {
        assert!((e - g).abs() < 1e-6, "Scale mismatch at {}: expected={}, got={}", i, e, g);
    }
}
```

---

## 7. Feature-Gate Boundary

**Trust Boundary:** Experimental (broken) kernels must not be accessible in default
builds. The `cuda-experimental` feature flag isolates known-broken code from production
while preserving it for development.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §7, Phase 2

**Phase:** 2

### 7.1 Default Build Excludes Warp Kernel

```rust
#[test]
#[cfg(all(feature = "cuda", not(feature = "cuda-experimental")))]
fn test_warp_kernel_not_in_default_api() {
    // GpuLz4Context should NOT expose decompress_blocks_warp
    // in the default build. This test verifies the feature gate.
    //
    // If this test compiles, it means the warp kernel API is correctly
    // hidden behind cuda-experimental.

    let ctx = GpuLz4Context::new(0);
    // The method decompress_blocks_warp should not exist here.
    // We verify by ensuring only the safe methods are available.

    // This is a compile-time test — if it compiles, it passes.
    // The actual runtime assertion is that the warp PTX is not loaded.
    if let Ok(mut ctx) = ctx {
        ctx.load_kernels().unwrap();
        // Verify the warp kernel function is NOT registered
        let warp_func = ctx.device()
            .get_func("lz4_warp", "lz4_decompress_blocks_warp");
        assert!(warp_func.is_none(),
            "Warp kernel should not be loaded in default build");
    }
}

#[test]
#[cfg(feature = "cuda-experimental")]
fn test_warp_kernel_available_with_experimental() {
    if !cuda_available() { return; }

    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();

    // Warp kernel should be registered when cuda-experimental is enabled
    let warp_func = ctx.device()
        .get_func("lz4_warp", "lz4_decompress_blocks_warp");
    assert!(warp_func.is_some(),
        "Warp kernel should be available with cuda-experimental");
}
```

### 7.2 Existing Tests Unaffected

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_default_lz4_kernels_still_work() {
    // Regression: ensure feature-gating didn't break K1/K2
    if !cuda_available() { return; }

    let mut ctx = GpuLz4Context::new(0).unwrap();
    ctx.load_kernels().unwrap();

    // K1: single block decompress
    let compressed = create_literals_only_lz4();
    let result = ctx.decompress_block(&compressed.0, compressed.1).unwrap();
    assert!(result.len() > 0);

    // K2: parallel decompress
    // ... (same as existing tests, just verify they still pass)
}
```

---

## 8. Pipeline Integration Tests

**Trust Boundary:** Complete pipelines compose multiple kernels. The integration must
produce correct end-to-end results — from compressed/quantized source data to
inference-ready F16 tensors.

**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §5, §8.2

**Phase:** 4

### 8.1 Pipeline A: Standard HCT INT4

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_pipeline_a_standard_hct_int4() {
    if !cuda_available() { return; }

    // 1. Create realistic tensor data (e.g., LayerNorm weights)
    let original: Vec<f32> = (0..512).map(|i| {
        ((i as f32) / 512.0 - 0.5) * 2.0 // range [-1.0, 1.0]
    }).collect();

    // 2. Quantize with CPU quantizer (block_size=128)
    let tensor = Tensor::from_vec(original.clone(), &[512], &Device::Cpu).unwrap();
    let quantizer = Quantizer::new(QuantizeConfig {
        format: QuantizeFormat::Int4Symmetric,
        block_size: 128,
        activation_aware: false,
    });
    let quantized = quantizer.quantize(&tensor).unwrap();

    // 3. LZ4 compress the packed INT4 data
    let compressed = lz4_compress_cpu(&quantized.data);

    // 4. GPU Pipeline A: LZ4 decompress → INT4 dequant
    let mut lz4_ctx = GpuLz4Context::new(0).unwrap();
    lz4_ctx.load_kernels().unwrap();
    let decompressed = lz4_ctx.decompress_block(&compressed, quantized.data.len()).unwrap();

    let mut dequant_ctx = GpuDequantContext::new(0).unwrap();
    dequant_ctx.load_int4_kernel().unwrap();
    let gpu_result = dequant_ctx.dequantize_int4(
        /* decompressed data */, &quantized.scales, /* ... */
    ).unwrap();

    // 5. CPU reference: dequantize directly
    let cpu_result = quantizer.dequantize(&quantized).unwrap();
    let cpu_values: Vec<f32> = cpu_result.to_vec1().unwrap();

    // 6. Compare GPU vs CPU
    let gpu_values: Vec<f32> = /* copy GPU F16 to host, convert to F32 */;
    for (i, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 0.02,
            "Pipeline A mismatch at {}: cpu={}, gpu={}", i, cpu, gpu
        );
    }
}
```

### 8.2 Pipeline B: Fused HCT INT4

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_pipeline_b_fused_hct_int4() {
    if !cuda_available() { return; }

    // Same setup as Pipeline A, but use fused kernel
    // ...

    // Fused kernel result must match Pipeline A result (bit-identical)
    assert_eq!(pipeline_a_bits, pipeline_b_bits,
        "Fused pipeline must match sequential pipeline");
}
```

### 8.3 Pipeline E: FP8 Conversion

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_pipeline_e_fp8_conversion() {
    if !cuda_available() { return; }

    // Create FP8 E4M3 data representing activation values
    let f32_values: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0, -1.0, 100.0, 448.0];
    let fp8_data: Vec<u8> = f32_values.iter()
        .map(|&v| f32_to_fp8_e4m3(v))
        .collect();

    // GPU convert
    let converter = GpuDtypeConverter::new(/* ... */).unwrap();
    let gpu_result = converter.fp8_e4m3_to_f32_host(&fp8_data).unwrap();

    // Verify: FP8→F32→FP8 round-trip preserves bits
    for (i, (&original_fp8, &result_f32)) in fp8_data.iter().zip(gpu_result.iter()).enumerate() {
        let roundtrip_fp8 = f32_to_fp8_e4m3(result_f32);
        assert_eq!(original_fp8, roundtrip_fp8,
            "FP8 round-trip failed at {}: 0x{:02X} → {} → 0x{:02X}",
            i, original_fp8, result_f32, roundtrip_fp8);
    }
}
```

### 8.4 Pipeline F: HoloTensor End-to-End

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_pipeline_f_holotensor_spectral() {
    if !cuda_available() { return; }

    // 1. Create spectral coefficients (sparse)
    let width = 8;
    let height = 8;
    let indices = vec![0u32, 1, 8, 9]; // Low-frequency components
    let values = vec![10.0f32, 5.0, 3.0, 1.0];

    // 2. GPU reconstruct: accumulate → IDCT → F16
    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_all_kernels().unwrap();

    let coeffs = ctx.spectral_accumulate(&indices, &values, width * height).unwrap();
    let f32_result = ctx.spectral_idct_1d_rows(/* coeffs */, width, height).unwrap();
    let f16_result = ctx.f32_to_f16(/* f32_result */).unwrap();

    // 3. Verify: output is valid (non-NaN, non-Inf, reasonable range)
    let host: Vec<f32> = f32_result.to_host().unwrap();
    for (i, &val) in host.iter().enumerate() {
        assert!(val.is_finite(), "HoloTensor output is not finite at index {}", i);
    }
}
```

---

## 9. Test Infrastructure

### 9.1 Proptest Generators

```rust
use proptest::prelude::*;

/// Arbitrary quantized INT4 data with valid structure.
prop_compose! {
    fn arb_quantized_int4(max_values: usize)
        (num_values in 1..max_values)
        (
            num_values in Just(num_values),
            data in prop::collection::vec(any::<u8>(), (num_values + 1) / 2),
            scales in prop::collection::vec(
                (0.001f32..10.0).prop_map(half::f16::from_f32),
                (num_values + 127) / 128
            ),
        ) -> QuantizedInt4Data {
            QuantizedInt4Data { data, scales, num_values }
        }
}

/// Arbitrary compressible data (with some patterns for LZ4 to exploit).
fn arb_compressible_data(max_size: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..32u8, 1..max_size) // Limited range = more LZ4 matches
}
```

### 9.2 Test Fixtures

```rust
/// Standard test data for GPU codec pipeline tests.
struct GpuCodecTestFixture {
    lz4_ctx: GpuLz4Context,
    dequant_ctx: GpuDequantContext,
    fused_ctx: GpuFusedContext,
    holo_ctx: GpuHoloContext,
}

impl GpuCodecTestFixture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut lz4_ctx = GpuLz4Context::new(0)?;
        lz4_ctx.load_kernels()?;
        let mut dequant_ctx = GpuDequantContext::new(0)?;
        dequant_ctx.load_int4_kernel()?;
        dequant_ctx.load_int8_kernel()?;
        let mut fused_ctx = GpuFusedContext::new(0)?;
        fused_ctx.load_lz4_int4_kernel()?;
        fused_ctx.load_lz4_int8_kernel()?;
        let mut holo_ctx = GpuHoloContext::new(0)?;
        holo_ctx.load_all_kernels()?;

        Ok(Self { lz4_ctx, dequant_ctx, fused_ctx, holo_ctx })
    }
}
```

---

## 10. Implementation Order

Tests are implemented in dependency order. Each phase builds on the previous.

### Phase 1: Constant Unification

1. Write `test_int4_block_size_matches_quantizer` (§1.1) — **RED** (hct.rs has 32)
2. Fix `hct.rs:593` — `Q4_BLOCK_SIZE = 128` — **GREEN**
3. Add compile-time `const_assert!` — prevents future regression
4. Write `test_hct_int4_dequant_uses_correct_block_size` (§1.2)
5. Write `test_hct_dequant_matches_quantizer_dequant` (§1.3)
6. Write `test_int4_nibble_packing_little_endian` (§2.1)
7. Write `test_int4_scale_layout_scales_before_data` (§2.2)
8. Write LZ4 GPU vs CPU cross-validation (§3.1, §3.2)
9. Write INT4/INT8 GPU vs CPU cross-validation (§4.1, §4.2)

### Phase 2: Feature-Gate Warp Kernel

1. Write `test_warp_kernel_not_in_default_api` (§7.1) — **RED** (no feature gate)
2. Add `cuda-experimental` feature to `Cargo.toml`
3. Wrap K3 warp kernel behind `#[cfg(feature = "cuda-experimental")]`
4. Move 4 warp tests behind `#[cfg(feature = "cuda-experimental")]`
5. Write `test_warp_kernel_available_with_experimental` (§7.1) — **GREEN**
6. Write `test_default_lz4_kernels_still_work` (§7.2) — regression check

### Phase 3: Fix Warp LZ4 Kernel

1. Write `test_lz4_warp_matches_parallel` (§3.4) — **RED** (warp is broken)
2. Write property test `test_lz4_warp_equivalence` (§3.4) — **RED**
3. Root-cause the thread coordination bug in warp PTX
4. Fix warp kernel PTX — **GREEN**
5. Benchmark warp vs parallel throughput

### Phase 4: Pipeline Integration + HoloTensor

1. Write HoloTensor tests (§6.1–6.5) — fills the 0-test gap
2. Write FP8 known-answer and property tests (§4.4)
3. Write fusion equivalence tests (§5.1–5.3)
4. Write pipeline integration tests (§8.1–8.4)
5. Write property test `test_int4_dequant_zero_ulp` (§4.3)

---

## 11. Test Count Summary

| Section | Specification | Property | Boundary | Total |
|---------|:-:|:-:|:-:|:-:|
| 0. Infrastructure | — | — | — | — |
| 1. Constant Integrity | 3 | 0 | 0 | 3 |
| 2. Data Format | 1 | 1 | 1 | 3 |
| 3. LZ4 Lossless | 3 | 2 | 1 | 6 |
| 4. Quantization Math | 3 | 2 | 1 | 6 |
| 5. Fusion Equivalence | 2 | 1 | 0 | 3 |
| 6. HoloTensor | 5 | 0 | 2 | 7 |
| 7. Feature-Gate | 2 | 0 | 1 | 3 |
| 8. Pipeline Integration | 4 | 0 | 0 | 4 |
| **Total** | **23** | **6** | **6** | **35** |
