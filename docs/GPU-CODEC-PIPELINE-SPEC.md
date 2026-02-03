# GPU Codec Pipeline Specification

**Version:** 0.1.0
**Status:** Initial Audit, Phase 1 In Progress
**Date:** 2026-02-03
**Crate:** `abaddon`

---

## 1. Overview

This specification defines the GPU-accelerated codec pipeline in `abaddon` — the set of
CUDA kernels responsible for transforming compressed or quantized model data into
inference-ready tensors entirely on the GPU.

### 1.1 Purpose

The GPU codec pipeline eliminates PCIe round-trips during model loading. Instead of:

```
Disk → CPU decompress → PCIe transfer (large) → GPU
```

The pipeline enables:

```
Disk → PCIe transfer (small, compressed) → GPU decompress/dequant → GPU memory
```

This reduces load times proportionally to the compression ratio and eliminates the
CPU decompression bottleneck during model initialization.

### 1.2 Scope

**In Scope:**
- LZ4 decompression on GPU (block-parallel)
- INT4 and INT8 dequantization on GPU
- Fused decompression + dequantization (single-pass, no intermediate buffer)
- FP8 (E4M3/E5M2) to F32 dtype conversion on GPU
- HoloTensor reconstruction on GPU (Spectral IDCT, RPH, LRDF)
- Fused dequantization during GEMM (INT4 weights consumed on-the-fly)
- Canonical constant definitions and data format layouts

**Out of Scope:**
- CPU-side codecs (haagenti's LZ4, Zstd implementations)
- Model architecture (attention, FFN, embeddings)
- Non-quantized GEMM/GEMV kernels (documented only where they share infrastructure)
- KV-cache management
- Flash attention kernels

### 1.3 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GPU Codec Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │  Compressed   │     │  Quantized   │     │  HoloTensor          │    │
│  │  LZ4 Blocks   │     │  INT4/INT8   │     │  Coefficients        │    │
│  └──────┬───────┘     └──────┬───────┘     └──────────┬───────────┘    │
│         │                    │                         │                │
│         ▼                    ▼                         ▼                │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │  gpu_lz4     │     │  gpu_dequant │     │  gpu_holo            │    │
│  │  (PTX)       │     │  (PTX)       │     │  (PTX)               │    │
│  └──────┬───────┘     └──────┬───────┘     └──────────┬───────────┘    │
│         │                    │                         │                │
│         ▼                    ▼                         ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     F16 / F32 Tensors (GPU)                      │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                       │
│               ┌─────────────────┼─────────────────┐                    │
│               ▼                 ▼                   ▼                   │
│        ┌────────────┐   ┌────────────┐   ┌──────────────────┐          │
│        │  Attention  │   │  GEMM      │   │  Fused Dequant   │          │
│        │  Kernels    │   │  Kernels   │   │  + GEMM          │          │
│        └────────────┘   └────────────┘   │  (fused_gemm.rs) │          │
│                                          └──────────────────┘          │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FUSED PIPELINES (single-pass, no intermediate buffer):          │  │
│  │                                                                   │  │
│  │  gpu_fused: LZ4+INT4 → F16   |   LZ4+INT8 → F16                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FP8 CONVERSION:                                                  │  │
│  │                                                                   │  │
│  │  gpu_dtype: FP8 E4M3 → F32   |   FP8 E5M2 → F32                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Constants

These are the authoritative values. Any deviation in the codebase is a bug.

| Constant | Value | Meaning | Canonical Source |
|----------|-------|---------|------------------|
| `INT4_BLOCK_SIZE` | **128** | Values per scale factor for HCT-native INT4 | `gpu_dequant.rs:46` |
| `DEFAULT_BLOCK_SIZE` | **128** | CPU quantizer default block size | `quantize.rs:45` |
| `QUANT_BLOCK_SIZE` | **32** | Values per scale for GPTQ/AWQ fused GEMM | `fused_gemm.rs:38` |

### 2.1 Block Size Disambiguation

There are two distinct quantization block sizes in the codebase, and they are **not**
interchangeable:

**HCT-native INT4 (block_size = 128):**
- Used by: `quantize.rs` (CPU quantizer), `gpu_dequant.rs` (GPU dequant), `gpu_fused.rs`
  (fused decompress+dequant), `weight_store.rs` (inference weight loading)
- Data source: Infernum's own quantization pipeline, HCT file format
- Scale layout: one FP16 scale per 128 INT4 values

**GPTQ/AWQ runtime (group_size = 32):**
- Used by: `fused_gemm.rs` (on-the-fly dequant during GEMM)
- Data source: Externally-quantized GPTQ/AWQ model weights
- Scale layout: one FP16 scale per 32 INT4 values (or per `group_size` for AWQ)

### 2.2 Known Constant Bugs

| ID | Location | Current | Correct | Impact |
|----|----------|---------|---------|--------|
| **DD-1** | `hct.rs:593` | `Q4_BLOCK_SIZE = 32` | **128** | HCT reader misinterprets scale layout. Reads one scale per 32 values but the quantizer wrote one scale per 128 values. Produces garbage dequantized output for INT4 HCT tensors loaded via CPU path. |

---

## 3. Data Formats

### 3.1 INT4 (4-bit Integer)

```
Memory layout (packed, little-endian nibbles):
┌────────────────┐
│ Byte 0         │  bits[3:0] = value[0], bits[7:4] = value[1]
│ Byte 1         │  bits[3:0] = value[2], bits[7:4] = value[3]
│ ...            │
└────────────────┘

Scale layout (HCT-native, block_size=128):
┌────────────────────────────────────────┐
│ scale[0] (F16) │ scale[1] (F16) │ ... │  one per 128 values
└────────────────────────────────────────┘

Dequantization (symmetric):
  output[i] = (int4_value - zero_point) * scale[i / block_size]

Dequantization (symmetric, zero_point=8):
  output[i] = (int4_value - 8) * scale[i / block_size]
```

**Value range:** 0..15 unsigned, mapped to -8..+7 with zero_point=8.

### 3.2 INT8 (8-bit Integer)

```
Memory layout:
┌──────────────────────────┐
│ value[0] (i8) │ value[1] │ ...   one byte per value (signed: -128..127)
└──────────────────────────┘

Scale: single F16 value per tensor (or per channel)

Dequantization:
  output[i] = (f32)value[i] * (f32)scale
  Output stored as F16.
```

### 3.3 FP8 E4M3 (Inference-Optimized)

```
Bit layout: [sign:1][exponent:4][mantissa:3]
Range: ±448, precision ~0.125
Use case: Inference activations and weights
```

### 3.4 FP8 E5M2 (Training-Compatible)

```
Bit layout: [sign:1][exponent:5][mantissa:2]
Range: ±57344, precision ~0.25
Use case: Training gradients, larger dynamic range
```

### 3.5 LZ4 Compressed Blocks (HCT Format)

```
HCT file structure:
┌──────────────┐
│ HCT Header   │  magic, version, tensor metadata
├──────────────┤
│ Block Index   │  [offset, compressed_size, uncompressed_size] per block
├──────────────┤
│ Block 0       │  LZ4-compressed data
│ Block 1       │
│ ...           │
│ Block N       │
└──────────────┘

Per-block LZ4 frame:
  Standard LZ4 block format (token + literal/match sequences)
  No LZ4 frame header — raw block data
```

### 3.6 HoloTensor Encodings

Three encoding types for holographic tensor compression:

**Spectral (IDCT):**
- Frequency-domain coefficients stored as sparse (index, value) pairs
- Reconstruction via Inverse Discrete Cosine Transform
- Supports 1D (row-wise) and 2D reconstruction

**LRDF (Low-Rank Decomposition):**
- SVD factorization: `T ≈ Σ σ_i * u_i * v_i^T`
- Stored as (U vectors, V vectors, sigma values) per component
- Reconstruction via outer product accumulation

**RPH (Random Projection Hash):**
- Dimensionality-reduced projection with pseudo-random basis
- Reconstruction via projection accumulation with seeded PRNG
- Supports progressive refinement

---

## 4. Kernel Interface Catalog

### 4.1 LZ4 Decompression Kernels

**Source:** `crates/abaddon/src/gpu_lz4.rs`
**Technology:** PTX (inline, compiled via `cudarc::nvrtc::Ptx`)
**Target:** `sm_50+`

#### K1: `lz4_decompress_block`

Single-threaded LZ4 decompression of one block.

```
Parameters:
  input_ptr:   u64   Pointer to compressed LZ4 data (device memory)
  input_size:  u32   Size of compressed data in bytes
  output_ptr:  u64   Pointer to output buffer (device memory)
  output_size: u32   Expected uncompressed size in bytes

Launch: grid=(1,1,1), block=(1,1,1)
Status: Working
```

#### K2: `lz4_decompress_blocks_parallel`

Parallel decompression of N blocks (one thread per block).

```
Parameters:
  input_ptr:             u64   Base pointer to all compressed data
  output_ptr:            u64   Base pointer to output buffer
  offsets_in_ptr:        u64   Per-block input offsets [u64; N]
  offsets_out_ptr:       u64   Per-block output offsets [u64; N]
  compressed_sizes_ptr:  u64   Per-block compressed sizes [u32; N]
  uncompressed_sizes_ptr: u64  Per-block uncompressed sizes [u32; N]
  num_blocks:            u32   Number of blocks to decompress

Launch: grid=(num_blocks, 1, 1), block=(32, 1, 1) — one warp per LZ4 block
        Each GPU block (warp) decompresses one LZ4 block sequentially via tid=0.
        The 32-thread block is minimum warp granularity.
Status: Working
```

#### K3: `lz4_decompress_blocks_warp`

Warp-parallel decompression (32 threads cooperate per block).

```
Parameters: Same as K2
Launch: grid=(num_blocks, 1, 1), block=(32, 1, 1) — all 32 threads cooperate
Status: BROKEN — produces garbage output
```

**Bug:** Thread coordination failure in literal copy offsets. The 32 warp threads
fail to correctly partition literal copies and match operations. Output contains
near-zero garbage values instead of correct decompressed data. Four tests fail:
`test_decompress_to_f32_slice`, `test_decompress_to_f16_slice`,
`test_warp_parallel_basic`, `test_warp_parallel_matches_sequential`.

### 4.2 Dequantization Kernels

**Source:** `crates/abaddon/src/gpu_dequant.rs`
**Technology:** PTX (inline)
**Target:** `sm_50+`

#### K4: `int4_dequant_tensor`

Bulk INT4 dequantization with per-block scales and zero points.

```
Parameters:
  packed_ptr:  u64   Packed INT4 data (2 values per byte, low nibble first)
  scales_ptr:  u64   Per-block FP16 scales [f16; num_blocks]
  zp_ptr:      u64   Per-block zero points [i8; num_blocks]
  output_ptr:  u64   Output F16 buffer [f16; num_values]
  num_values:  u32   Total output values
  block_size:  u32   Values per quantization block (should be 128 for HCT)

Launch: grid=(ceil(num_values/256), 1, 1), block=(256, 1, 1)
Output: F16
Status: Working
```

#### K5: `int4_dequant_block`

Single-block INT4 dequantization (one scale/zero-point for all values).

```
Parameters:
  packed_ptr:  u64   Packed INT4 data
  output_ptr:  u64   Output F16 buffer
  scale_bits:  u32   FP16 scale passed as u16 bit pattern (lower 16 bits)
  zero_point:  s32   Integer zero point
  num_values:  u32   Number of output values

Launch: grid=(ceil(num_values/256), 1, 1), block=(256, 1, 1)
Output: F16
Status: Working
```

#### K6: `int8_dequant_tensor`

Bulk INT8 dequantization with per-tensor scale.

```
Parameters:
  input_ptr:   u64   Signed INT8 data [i8; num_values]
  output_ptr:  u64   Output F16 buffer [f16; num_values]
  scale_bits:  u32   FP16 scale passed as u16 bit pattern (lower 16 bits)
  num_values:  u32   Total number of values

Launch: grid=(ceil(num_values/256), 1, 1), block=(256, 1, 1)
Output: F16
Status: Working
```

**Note:** The `scale_bits` parameter passes an F16 value as the lower 16 bits of a u32.
The upper 16 bits are unused (waste). See DD-4.

### 4.3 Fused Decompression + Dequantization Kernels

**Source:** `crates/abaddon/src/gpu_fused.rs`
**Technology:** PTX (inline)
**Target:** `sm_50+`

These kernels combine LZ4 decompression and dequantization in a single pass,
eliminating the intermediate decompressed buffer.

#### K7: `fused_lz4_int4_block`

Single-block fused LZ4 decompression + INT4 dequantization.

```
Parameters:
  input_ptr:   u64   Compressed LZ4+INT4 data
  input_size:  u32   Size of compressed data
  output_ptr:  u64   Output F16 buffer
  num_values:  u32   Number of output F16 values
  scale_bits:  u32   FP16 scale as u16 bit pattern
  zero_point:  s32   Zero point for dequantization

Launch: grid=(1,1,1), block=(1,1,1) — single-threaded
Output: F16
Status: Working
```

#### K8: `fused_lz4_int4_blocks_parallel`

Multi-block fused LZ4+INT4 with per-block scales and zero points.

```
Parameters:
  input_ptr:       u64   Base compressed data pointer
  output_ptr:      u64   Base output pointer
  comp_offsets_ptr: u64  Per-block compressed data offsets [u64; N]
  out_offsets_ptr: u64   Per-block output offsets [u64; N]
  comp_sizes_ptr:  u64   Per-block compressed sizes [u32; N]
  out_sizes_ptr:   u64   Per-block output value counts [u32; N]
  scales_ptr:      u64   Per-block FP16 scales [u16; N]
  zp_ptr:          u64   Per-block zero points [i8; N]
  num_blocks:      u32   Number of blocks

Launch: grid=(ceil(num_blocks/256), 1, 1), block=(256, 1, 1)
Output: F16
Status: Working
```

#### K9: `fused_lz4_int8_block`

Single-block fused LZ4 decompression + INT8 dequantization.

```
Parameters:
  input_ptr:   u64   Compressed LZ4+INT8 data
  input_size:  u32   Size of compressed data
  output_ptr:  u64   Output F16 buffer
  num_values:  u32   Number of output F16 values
  scale_bits:  u32   FP16 scale as u16 bit pattern

Launch: grid=(1,1,1), block=(1,1,1) — single-threaded
Output: F16
Status: Working
```

### 4.4 FP8 Dtype Conversion Kernels

**Source:** `crates/abaddon/src/gpu_dtype.rs`
**Technology:** NVRTC (CUDA C compiled at runtime)

#### K10: `fp8_e4m3_to_f32`

```
Parameters:
  input:  const unsigned char*   FP8 E4M3 data
  output: float*                 F32 output
  n:      int                    Number of values

Output: F32
Status: Working
```

#### K11: `fp8_e5m2_to_f32`

```
Parameters:
  input:  const unsigned char*   FP8 E5M2 data
  output: float*                 F32 output
  n:      int                    Number of values

Output: F32
Status: Working
```

**Note:** Both FP8 kernels output F32. There is no direct FP8 → F16 path;
callers must cast F32 → F16 themselves. See DD-6.

### 4.5 HoloTensor Reconstruction Kernels

**Source:** `crates/abaddon/src/gpu_holo.rs`
**Technology:** PTX (inline)
**Target:** `sm_50+`

#### Spectral (IDCT) Kernels

| Kernel | Parameters | Status |
|--------|-----------|--------|
| `holo_spectral_accumulate` | `(indices, values, coeffs, mask, num_coeffs, buffer_size)` | Working |
| `holo_spectral_idct_1d_rows` | `(input, output, width, height)` | Working |
| `holo_spectral_idct_1d_cols` | `(input, output, width, height)` | **Stub** (returns immediately) |
| `holo_spectral_idct_2d` | `(input, output, width, height)` | **Stub** (returns immediately) |
| `holo_spectral_idct_f16` | `(coeffs, output, width, height)` | Working (F32 → F16 IDCT) |

#### RPH (Random Projection Hash) Kernels

| Kernel | Parameters | Status |
|--------|-----------|--------|
| `holo_rph_accumulate` | `(projection, output, proj_dim, output_dim, seed)` | Working |
| `holo_rph_finalize` | `(input, output, size, num_projections)` | Working |
| `holo_rph_generate_projection` | `(output, row, col, rows, cols, seed)` | **Stub** |

#### LRDF (Low-Rank Decomposition) Kernels

| Kernel | Parameters | Status |
|--------|-----------|--------|
| `holo_lrdf_outer_product` | `(u, v, output, sigma, rows, cols)` | Working |
| `holo_lrdf_outer_product_batched` | `(u, v, sigma, output, num_components, rows, cols)` | **Stub** |

#### Utility Kernels

| Kernel | Parameters | Status |
|--------|-----------|--------|
| `holo_fused_f32_to_f16` | `(input, output, size)` | Working |
| `holo_fused_dequant_f32` | `(input, scales, zeros, output, size, block_size)` | Working |
| `holo_scale_values` | `(data, scale, size)` | Working |
| `holo_coalesced_accumulate_v4` | `(src, dst, num_elements)` | Working (vectorized) |
| `holo_coalesced_idct_tile` | `(coeffs, output, width, height, tile_size)` | Working (shared mem) |
| `holo_coalesced_f32_to_f16_v4` | `(input, output, size)` | Working (vectorized) |

### 4.6 Fused Dequant + GEMM Kernels

**Source:** `crates/abaddon/src/cuda_inference/kernels/fused_gemm.rs`
**Technology:** NVRTC (CUDA C)
**Block size:** 32 (GPTQ/AWQ) or implicit 32 (HCT symmetric)

These kernels dequantize INT4 weights on-the-fly during matrix multiplication,
avoiding the 4x memory overhead of storing F16 weights.

#### HCT Symmetric INT4

| Kernel | Signature | Status |
|--------|----------|--------|
| `fused_int4_gemv_f16` | `(input[1,K], weights[K/2,N], scales[K/32,N], output[1,N], N, K)` | Working |
| `fused_int4_gemv_f16_v2` | Same (optimized) | Working |
| `fused_int4_gemm_f16` | `(input[M,K], weights[K/2,N], scales[K/32,N], output[M,N], M, N, K)` | Working |

#### GPTQ Format

| Kernel | Signature | Status |
|--------|----------|--------|
| `fused_gptq_gemv_f16` | `(input, weights, scales, zeros, g_idx, output, N, K, group_size)` | Working |
| `fused_gptq_gemm_f16` | `(input, weights, scales, zeros, g_idx, output, M, N, K, group_size)` | Working |

#### AWQ Format

| Kernel | Signature | Status |
|--------|----------|--------|
| `fused_awq_gemv_f16` | `(input, weights, scales, zeros, output, N, K, group_size)` | Working |

---

## 5. Pipeline Composition

Valid pipeline chains from compressed/quantized data to inference-ready tensors:

### Pipeline A: Standard HCT (INT4)

```
LZ4 Compressed → [K1/K2: LZ4 decompress] → Packed INT4 → [K4: dequant] → F16 Tensor
```
- Two kernel launches, requires intermediate buffer for decompressed INT4 data
- block_size = 128

### Pipeline B: Fused HCT (INT4)

```
LZ4+INT4 Compressed → [K7/K8: fused decompress+dequant] → F16 Tensor
```
- Single kernel launch, no intermediate buffer
- Saves GPU memory bandwidth and VRAM allocation
- block_size = 128

### Pipeline C: Standard HCT (INT8)

```
LZ4 Compressed → [K1/K2: LZ4 decompress] → INT8 → [K6: dequant] → F16 Tensor
```
- Two kernel launches
- Per-tensor scale (single FP16 value)

### Pipeline D: Fused HCT (INT8)

```
LZ4+INT8 Compressed → [K9: fused decompress+dequant] → F16 Tensor
```
- Single kernel launch, no intermediate buffer

### Pipeline E: FP8 Conversion

```
FP8 Data → [K10/K11: fp8_to_f32] → F32 Tensor → (optional F16 cast)
```
- Single kernel launch
- No direct F16 output path (see DD-6)

### Pipeline F: HoloTensor Reconstruction

```
                    ┌── Spectral: coefficients → [accumulate → IDCT] → F32
HoloTensor Data ────┤── RPH: projections → [accumulate → finalize] → F32
                    └── LRDF: (U,V,σ) → [outer_product accumulate] → F32

F32 Tensor → [holo_fused_f32_to_f16] → F16 Tensor
         └── [holo_fused_dequant_f32] → F32 (if post-reconstruction dequant needed)
```
- Multiple kernel launches depending on encoding type
- Supports progressive/streaming reconstruction

### Pipeline G: Dequant-During-GEMM

```
Packed INT4 + F16 Activations → [fused GEMV/GEMM] → F16 Output
```
- Weights never materialized as F16 — dequantized on-the-fly per tile
- Saves 4x weight memory vs pre-dequantized F16
- Three quantization formats: HCT symmetric, GPTQ, AWQ

### Pipeline Comparison

| Pipeline | Kernel Launches | Intermediate Buffers | Output | block_size |
|----------|:-:|:-:|--------|:--:|
| A (Standard INT4) | 2 | 1 (packed INT4) | F16 | 128 |
| B (Fused INT4) | 1 | 0 | F16 | 128 |
| C (Standard INT8) | 2 | 1 (INT8) | F16 | — |
| D (Fused INT8) | 1 | 0 | F16 | — |
| E (FP8) | 1 | 0 | F32 | — |
| F (HoloTensor) | 2-4 | 1 (F32 coefficients) | F16/F32 | — |
| G (Dequant-GEMM) | 1 | 0 | F16 | 32 (GPTQ/AWQ) or implicit |

---

## 6. Correctness Guarantees

### 6.1 LZ4 Decompression

**Guarantee:** Byte-exact match with CPU decompressor (haagenti).

Given identical compressed input, the GPU LZ4 kernel MUST produce identical output
bytes to `haagenti::lz4::decompress()`. This is a hard requirement — LZ4 is lossless.

**Verification:** Compare GPU output against haagenti reference for all test vectors.

### 6.2 INT4 Dequantization

**Guarantee:** Within F16 rounding of the mathematical formula.

```
expected = round_to_f16( (f32)(int4_value - zero_point) * (f32)scale )
actual   = kernel output
```

Maximum per-value error: 0 ULP in F16 (bit-exact with reference implementation).

The intermediate computation uses F32 arithmetic to avoid F16 precision loss during
multiply. The final result is rounded to F16 via `cvt.rn.f16.f32`.

### 6.3 INT8 Dequantization

Same guarantee as INT4: F32 intermediate, F16 output, 0 ULP error.

### 6.4 FP8 Conversion

**Guarantee:** IEEE-conformant conversion with no additional rounding.

FP8 → F32 is exact (F32 has strictly more precision than both E4M3 and E5M2).
The conversion performs bitwise extraction of sign, exponent, and mantissa fields,
then constructs the corresponding F32 value.

### 6.5 Fused Kernels

**Guarantee:** Bit-identical to the corresponding sequential pipeline.

`fused_lz4_int4_block(data)` MUST produce the same output as:
```
temp = lz4_decompress_block(data)
result = int4_dequant_block(temp, scale, zp)
```

This ensures that the fused optimization is purely a performance improvement with
no semantic difference.

### 6.6 HoloTensor Reconstruction

**Guarantee:** Reconstruction error bounded by truncation rank.

For spectral encoding with N coefficients out of total M:
```
||original - reconstructed|| ≤ Σ |dropped_coefficient[i]|  for i in N..M
```

This is inherent to lossy compression — the spec does not require exact reconstruction,
only that the GPU implementation matches the mathematical definition of the encoding.

---

## 7. Implementation Phases

### Phase 1: Spec + Constant Unification

- [x] Audit all kernel signatures and parameters
- [x] Document canonical constants (INT4_BLOCK_SIZE=128, QUANT_BLOCK_SIZE=32)
- [x] Identify hct.rs Q4_BLOCK_SIZE bug (DD-1)
- [x] Write initial spec (this document)
- [ ] Fix `hct.rs:593` — change `Q4_BLOCK_SIZE` from 32 to 128
- [ ] Add compile-time assertion: `const_assert!(INT4_BLOCK_SIZE == DEFAULT_BLOCK_SIZE)`

### Phase 2: Feature-Gate Broken Warp Kernel

- [ ] Gate `lz4_decompress_blocks_warp` behind `#[cfg(feature = "cuda-experimental")]`
- [ ] Remove warp kernel from default `GpuLz4Context` API
- [ ] Update tests: warp tests only run with `cuda-experimental` feature
- [ ] Document in spec: K3 status changed from "Broken" to "Experimental"

### Phase 3: Fix Warp-Parallel LZ4 Kernel

- [ ] Root-cause the thread coordination bug (literal copy offset distribution)
- [ ] Implement correct warp-cooperative LZ4 decompression
- [ ] Verify byte-exact match with K1/K2 for all test vectors
- [ ] Benchmark throughput improvement over K2

### Phase 4: Pipeline Integration Tests

- [ ] End-to-end test: quantize → compress → GPU transfer → decompress → dequant → verify
- [ ] Cross-validation: GPU pipeline vs CPU pipeline for identical inputs
- [ ] Property tests: random tensors survive round-trip within error bounds
- [ ] Benchmark all 7 pipeline configurations

---

## 8. Test Requirements

### 8.1 Per-Kernel Unit Tests

Every kernel MUST have tests that verify:
1. **Known-answer test:** Fixed input → expected output (golden vector)
2. **Boundary conditions:** Empty input, single element, maximum size
3. **Round-trip:** For codec pairs (compress/decompress, quantize/dequantize)

### 8.2 Pipeline Integration Tests

Each pipeline (A through G) MUST have at least one test that:
1. Starts from realistic model data (e.g., tensor of typical dimensions)
2. Passes through the complete pipeline
3. Verifies output against CPU reference implementation

### 8.3 Cross-Validation

GPU kernel outputs MUST match CPU reference implementations:
- GPU LZ4 vs haagenti LZ4
- GPU INT4 dequant vs `quantize.rs` dequantize
- GPU FP8 conversion vs software FP8 tables

### 8.4 CUDA Test Execution

Due to CUDA context thread-safety limitations (DD-3), GPU tests should be run with:
```bash
cargo test -p abaddon --features cuda -- --test-threads=1
```

Or with the WSL library path:
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test -p abaddon --features cuda -- --test-threads=1
```

---

## 9. Design Debt Register

| ID | Issue | Severity | Location | Resolution |
|----|-------|:--------:|----------|------------|
| DD-1 | `Q4_BLOCK_SIZE=32` in HCT reader vs quantizer's 128 | **Critical** | `hct.rs:593` | Change to 128. Phase 1. |
| DD-2 | Warp LZ4 kernel produces garbage output | **High** | `gpu_lz4.rs` K3 | Feature-gate (Phase 2), fix (Phase 3). |
| DD-3 | CUDA context not thread-safe across parallel tests | Medium | All GPU modules | Run with `--test-threads=1`. Consider per-test context isolation. |
| DD-4 | INT8 scale passed as u32 (upper 16 bits wasted) | Low | `gpu_dequant.rs` K6, `gpu_fused.rs` K9 | PTX limitation — u32 is minimum param width. Not actionable. |
| DD-5 | Fused GEMM `QUANT_BLOCK_SIZE=32` differs from `INT4_BLOCK_SIZE=128` | Low | `fused_gemm.rs:38` | By design — GPTQ/AWQ vs HCT. No fix needed. Documented in Section 2.1. |
| DD-6 | FP8 kernels output F32, no direct F16 path | Low | `gpu_dtype.rs` | Add `fp8_to_f16` kernels. Future optimization. |
| DD-7 | HoloTensor dequant accepts runtime `block_size` parameter | Info | `gpu_holo.rs:2254` | Flexible by design. Ensure callers pass correct value. |
| DD-8 | Four HoloTensor kernels are stubs (return immediately) | Medium | `gpu_holo.rs` | `idct_1d_cols`, `idct_2d`, `rph_generate_projection`, `lrdf_outer_product_batched`. Implement or remove. |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-03 | Codebase audit | Initial spec from source audit. Documented 14+ kernels across 6 modules. Identified DD-1 (critical block size bug), DD-2 (broken warp kernel), DD-8 (stub kernels). |
