# DD-8 Stub Kernel TDD Roadmap

**Version:** 0.1.0
**Status:** Test Specification
**Date:** 2026-02-03
**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md §9 (DD-8)
**Parent Roadmap:** GPU-CODEC-PIPELINE-TDD.md §6

---

## Philosophy

This roadmap is a focused extension of GPU-CODEC-PIPELINE-TDD.md. It targets Design
Debt DD-8: six HoloTensor kernels in `gpu_holo.rs` that are stubs, broken, or dead code.

The trust boundary is **mathematical correctness** — each kernel makes a promise about
what it computes, and our tests verify that promise against a known-correct reference.
The reference implementations live in the `haagenti` crate:

| Component | Reference File | Lines |
|-----------|---------------|-------|
| IDCT 1D (CPU) | `haagenti-core/src/dct.rs` | 261–277 (`idct_1d_direct`) |
| IDCT 1D (CUDA C) | `haagenti-cuda/src/dct_gpu.rs` | 200–253 |
| XORShift PRNG | `haagenti-hct/src/holotensor.rs` | 657–692 (`SeededRng`) |
| LRDF outer product | `haagenti-hct/src/holotensor.rs` | 1426–1529 |

Tolerance: **< 1e-4 absolute error** for IDCT (matching haagenti test thresholds).
Bit-exact for PRNG sequences.

---

## Scope

| # | Kernel | Current State | Action | Priority |
|---|--------|--------------|--------|----------|
| 1 | `holo_spectral_idct_1d_rows` | Placeholder (writes zeros) | **Implement** | Critical |
| 2 | `holo_spectral_idct_1d_cols` | Pure stub (`ret`) | **Implement** | Critical |
| 3 | `holo_rph_accumulate` | Broken XORShift PRNG + pointer bug | **Fix** | High |
| 4 | `holo_spectral_idct_2d` | Pure stub (`ret`), never called | **Remove** | Low |
| 5 | `holo_rph_generate_projection` | Pure stub (`ret`), never called | **Remove** | Low |
| 6 | `holo_lrdf_outer_product_batched` | Pure stub (`ret`) | **Implement** | Medium |

All code lives in `crates/abaddon/src/gpu_holo.rs`.

---

## Phase 1: Spectral IDCT — Row Kernel

**Trust Boundary:** The row IDCT kernel promises to compute the Type-III DCT
(Inverse DCT) along each row of a 2D coefficient matrix. The mathematical
definition is:

```
x[n] = sqrt(2/N) * [ X[0]/sqrt(2) + Σ_{k=1}^{N-1} X[k] · cos(π(2n+1)k / 2N) ]
```

**Current State:** `holo_spectral_idct_1d_rows` (PTX lines 1669–1720) has the loop
structure but accumulates `%sum = 0.0` without loading input or computing cosines.
It writes zeros to every output element.

**Launch Config (unchanged):** `finalize_spectral_direct()` at line 914:
- Grid: `(ceil(height/256), 1, 1)`, Block: `(256, 1, 1)`
- 1 thread per row, inner loop over `width` columns
- Shared memory: `width * 4` bytes (reserved but unused by placeholder)

**Reference:** `haagenti-cuda/src/dct_gpu.rs:200–225` (CUDA C row IDCT)

### §S1.1 Known-Answer: DC-Only Row IDCT

A DC-only input `[C, 0, 0, ..., 0]` must produce a uniform row where every
element equals `C * sqrt(2/N) * 1/sqrt(2) = C/sqrt(N)`.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_idct_1d_rows_dc_only() {
    if !cuda_available() { return; }

    let width = 8;
    let height = 1;
    // Frequency domain: DC = 4.0, all AC = 0
    let coeffs = {
        let mut c = vec![0.0f32; width * height];
        c[0] = 4.0;
        c
    };

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let d_input = ctx.device.htod_copy(coeffs.clone()).unwrap();
    let d_output: CudaSlice<f32> = ctx.device.alloc_zeros(width * height).unwrap();

    // Launch row IDCT directly
    let func = ctx.device.get_func("holo_spectral", "holo_spectral_idct_1d_rows").unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (width * 4) as u32,
    };
    unsafe { func.launch(cfg, (&d_input, &d_output, width as u32, height as u32)) }.unwrap();
    ctx.device.synchronize().unwrap();

    let host = ctx.copy_to_host(&d_output).unwrap();
    let expected = 4.0 / (width as f32).sqrt(); // DC / sqrt(N)

    for (i, &val) in host.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-4,
            "DC-only row IDCT at {}: expected {:.6}, got {:.6}", i, expected, val
        );
    }
}
```

### §S1.2 Known-Answer: Single AC Coefficient

A single AC coefficient at index `k` produces a cosine wave:
`x[n] = sqrt(2/N) * cos(π(2n+1)k / 2N)`

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_idct_1d_rows_single_ac() {
    if !cuda_available() { return; }

    let width = 8;
    let height = 1;
    let k = 1; // First AC frequency
    let amplitude = 3.0f32;

    let mut coeffs = vec![0.0f32; width];
    coeffs[k] = amplitude;

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let d_input = ctx.device.htod_copy(coeffs).unwrap();
    let d_output: CudaSlice<f32> = ctx.device.alloc_zeros(width).unwrap();

    let func = ctx.device.get_func("holo_spectral", "holo_spectral_idct_1d_rows").unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (width * 4) as u32,
    };
    unsafe { func.launch(cfg, (&d_input, &d_output, width as u32, 1u32)) }.unwrap();
    ctx.device.synchronize().unwrap();

    let host = ctx.copy_to_host(&d_output).unwrap();
    let scale = (2.0 / width as f32).sqrt();
    let pi_2n = std::f32::consts::PI / (2.0 * width as f32);

    for n in 0..width {
        let expected = amplitude * scale * ((2 * n + 1) as f32 * k as f32 * pi_2n).cos();
        assert!(
            (host[n] - expected).abs() < 1e-4,
            "AC[{}] row IDCT at {}: expected {:.6}, got {:.6}", k, n, expected, host[n]
        );
    }
}
```

### §S1.3 GPU vs CPU Cross-Validation (Proptest)

Random coefficient vectors: GPU row IDCT must match `haagenti_core::dct::idct_1d_direct()`
within tolerance.

```rust
mod idct_row_proptest {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        #[test]
        fn gpu_idct_1d_rows_matches_cpu(
            width in prop::sample::select(vec![4usize, 8, 16, 32]),
            coeffs_raw in proptest::collection::vec(-10.0f32..10.0f32, 32),
        ) {
            if !cuda_available() { return Ok(()); }

            let coeffs: Vec<f32> = coeffs_raw[..width].to_vec();

            // CPU reference (haagenti)
            let mut cpu_output = vec![0.0f32; width];
            haagenti_core::dct::idct_1d_direct(&coeffs, &mut cpu_output);

            // GPU
            let mut ctx = GpuHoloContext::new(0).unwrap();
            ctx.load_spectral_kernel().unwrap();
            // ... launch holo_spectral_idct_1d_rows ...
            let gpu_output = /* copy to host */;

            for (i, (&cpu, &gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
                prop_assert!(
                    (cpu - gpu).abs() < 1e-4,
                    "Row IDCT mismatch at {}: cpu={:.6}, gpu={:.6}", i, cpu, gpu
                );
            }
        }
    }
}
```

**Note:** If `haagenti_core` is not a dependency of abaddon, inline the CPU reference:

```rust
fn idct_1d_direct_reference(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    let scale = (2.0 / n as f32).sqrt();
    for (i, out_i) in output.iter_mut().enumerate().take(n) {
        let mut sum = input[0] / std::f32::consts::SQRT_2;
        for (k, &inp_k) in input.iter().enumerate().skip(1) {
            sum += inp_k * (std::f32::consts::PI * k as f32 * (i as f32 + 0.5) / n as f32).cos();
        }
        *out_i = sum * scale;
    }
}
```

---

## Phase 2: Spectral IDCT — Column Kernel

**Trust Boundary:** Same IDCT-III formula, applied along each column. Column `c`
reads `input[k * width + c]` for `k = 0..height-1` and writes to
`output[n * width + c]` for `n = 0..height-1`.

**Current State:** `holo_spectral_idct_1d_cols` (PTX lines 1723–1733) is a pure
stub: `mov.u32 %tmp, 0; ret;`

**Launch Config (unchanged):** `finalize_spectral_direct()` at line 941:
- Grid: `(ceil(width/256), 1, 1)`, Block: `(256, 1, 1)`
- 1 thread per column, inner loop over `height` rows
- Shared memory: `height * 4` bytes

**Reference:** `haagenti-cuda/src/dct_gpu.rs:228–253` (CUDA C column IDCT)

### §S2.1 Known-Answer: DC-Only Column IDCT

Same principle as §S1.1, but operating on columns.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_idct_1d_cols_dc_only() {
    if !cuda_available() { return; }

    let width = 4;
    let height = 4;
    // Input: DC = 4.0 in first row, zeros elsewhere
    // (After row IDCT, a DC-only frequency domain has energy only in row 0)
    let mut coeffs = vec![0.0f32; width * height];
    for c in 0..width {
        coeffs[c] = 4.0; // First row has the DC values (post-row-IDCT)
    }

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let d_input = ctx.device.htod_copy(coeffs).unwrap();
    let d_output: CudaSlice<f32> = ctx.device.alloc_zeros(width * height).unwrap();

    let func = ctx.device.get_func("holo_spectral", "holo_spectral_idct_1d_cols").unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (height * 4) as u32,
    };
    unsafe { func.launch(cfg, (&d_input, &d_output, width as u32, height as u32)) }.unwrap();
    ctx.device.synchronize().unwrap();

    let host = ctx.copy_to_host(&d_output).unwrap();
    let expected = 4.0 / (height as f32).sqrt();

    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            assert!(
                (host[idx] - expected).abs() < 1e-4,
                "DC-only col IDCT at [{},{}]: expected {:.6}, got {:.6}",
                row, col, expected, host[idx]
            );
        }
    }
}
```

### §S2.2 GPU vs CPU Cross-Validation (Proptest)

Same pattern as §S1.3, but for column IDCT. Extract columns, apply CPU reference
to each, compare.

```rust
// Same structure as §S1.3 but extracts each column, applies
// idct_1d_direct_reference, and compares with GPU column output.
```

### §S2.3 End-to-End Separable 2D IDCT

Full pipeline: `accumulate_spectral` → `finalize_spectral` → verify spatial output.
This exercises both row and col kernels in sequence.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_spectral_2d_idct_end_to_end() {
    if !cuda_available() { return; }

    // Use a known 4x4 spatial signal, compute its DCT-II coefficients offline,
    // then verify the GPU pipeline reconstructs the original within tolerance.
    let width = 4;
    let height = 4;

    // Known spatial signal
    let spatial: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 * 0.3).sin())
        .collect();

    // Compute DCT-II coefficients using CPU reference
    let mut dct_coeffs = vec![0.0f32; width * height];
    haagenti_core::dct::dct_2d(&spatial, &mut dct_coeffs, width, height);

    // Create spectral fragments from all non-zero coefficients
    let fragments = make_full_spectral_fragments(&dct_coeffs, width);

    // GPU reconstruction
    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_spectral_kernel().unwrap();

    let header = HoloTensorHeader::new(
        HolographicEncoding::Spectral,
        DType::F32,
        vec![height as u64, width as u64],
        fragments.len() as u16,
    );

    let mut acc = ctx.create_accumulator(&header).unwrap();
    for frag in &fragments {
        ctx.accumulate_spectral(frag, &mut acc).unwrap();
    }
    let result = ctx.finalize_spectral(&acc).unwrap();
    let host = ctx.copy_to_host(&result).unwrap();

    for (i, (&orig, &recon)) in spatial.iter().zip(host.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-4,
            "2D IDCT roundtrip at {}: original={:.6}, reconstructed={:.6}",
            i, orig, recon
        );
    }
}
```

### §S2.4 Update Existing Tests

Once row and col kernels are implemented:

1. `test_spectral_dc_only_produces_constant_output` — replace the "expected zeros due
   to stub" assertion (line 5176) with correct non-zero value assertions.
2. Remove §6.4 stub detection tests for `idct_1d_cols` (no longer a stub).

---

## Phase 3: RPH XORShift PRNG Fix

**Trust Boundary:** The RPH accumulation kernel generates pseudo-random weights
deterministically from a seed. The PRNG must produce the same sequence as
`haagenti::holotensor::SeededRng` for cross-crate reconstruction compatibility.

**Current State:** Two bugs in `holo_rph_accumulate` (PTX lines 1757–1826):

1. **XORShift PRNG broken** (lines 1798–1802): Uses `xor.b64 %rng_state, %rng_state, %rng_state`
   which always produces zero (XOR of a register with itself). All random weights
   are 0.0, making output all zeros regardless of input.

2. **Pointer arithmetic bug** (line 1795): `mad.wide.u32 %proj_addr, %i, 4, %proj_addr`
   modifies the base pointer in-place each iteration, causing quadratic address growth
   instead of linear. Should save base pointer and compute `base + i * 4` each iteration.

**Reference:** `haagenti-hct/src/holotensor.rs:657–692` (`SeededRng`)

```rust
// haagenti XORShift64 reference:
x ^= x << 13;
x ^= x >> 7;
x ^= x << 17;
```

### §R3.1 PRNG Sequence Correctness

Verify the GPU XORShift produces the same u64 sequence as `SeededRng::next_u64()`.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_rph_prng_matches_seeded_rng() {
    if !cuda_available() { return; }

    // Test that the GPU PRNG for a known seed produces the same
    // sequence as haagenti's SeededRng.
    //
    // The accumulate kernel doesn't expose raw PRNG output directly,
    // so we test indirectly: a uniform projection vector [1.0, 1.0, ...]
    // with the PRNG weights should produce a known sum.
    //
    // CPU reference:
    let seed = 42u64;
    let proj_dim = 8;
    let projection = vec![1.0f32; proj_dim];
    let mut rng = SeededRng::new(seed.wrapping_add(1)); // +1 matches SeededRng::new
    let mut expected_sum = 0.0f32;
    for _ in 0..proj_dim {
        let raw = rng.next_u64();
        let rand_val = /* same conversion as fixed kernel */;
        expected_sum += 1.0 * rand_val;
    }

    // GPU: accumulate with same inputs, compare single-element output
    // ...
}
```

### §R3.2 RPH Output Non-Zero After Fix

After fixing the PRNG, the existing `test_rph_deterministic_same_seed` test should
produce non-zero output. Update the assertion:

```rust
// Before (documents broken PRNG):
assert!(host1.iter().all(|v| *v == 0.0),
    "RPH output is expected to be all zeros due to broken XORShift PRNG (§6.4)");

// After (verifies fix):
assert!(!host1.iter().all(|v| *v == 0.0),
    "RPH output should be non-zero after XORShift fix");
```

### §R3.3 Different Seeds Produce Different Output

Reactivate the divergence test that was removed when the PRNG was broken:

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_rph_different_seeds_produce_different_output() {
    // ... same structure as before, but now expects non-zero divergent output ...
    let any_different = host_a.iter().zip(host_b.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(any_different,
        "Different seed_offsets must produce different RPH outputs");
}
```

### §R3.4 RPH GPU vs CPU Cross-Validation

Full accumulate + finalize cycle matches haagenti CPU RPH decode for a known tensor.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_rph_gpu_matches_cpu_reconstruction() {
    if !cuda_available() { return; }

    // Encode a known vector using haagenti's RphEncoder (CPU)
    // Decode using GPU accumulate + finalize
    // Compare outputs within tolerance

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let encoder = RphEncoder::new(/* proj_dim */ 4, /* seed */ 42, /* num_fragments */ 2);
    let fragments = encoder.encode(&data).unwrap();

    // GPU reconstruction
    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_rph_kernel().unwrap();
    // ... create accumulator, accumulate fragments, finalize ...

    // CPU reconstruction (haagenti decoder)
    let mut decoder = RphDecoder::new(/* ... */);
    // ... decode fragments ...

    // Compare (RPH is lossy, so use relaxed tolerance)
    for (i, (&cpu, &gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-3,
            "RPH cross-validation mismatch at {}: cpu={:.6}, gpu={:.6}", i, cpu, gpu
        );
    }
}
```

### §R3.5 Implementation Notes

**XORShift fix** — replace broken PTX with:

```ptx
// Correct xorshift64 (needs .reg .u64 %tmp64)
// x ^= x << 13
shl.b64 %tmp64, %rng_state, 13;
xor.b64 %rng_state, %rng_state, %tmp64;
// x ^= x >> 7
shr.u64 %tmp64, %rng_state, 7;
xor.b64 %rng_state, %rng_state, %tmp64;
// x ^= x << 17
shl.b64 %tmp64, %rng_state, 17;
xor.b64 %rng_state, %rng_state, %tmp64;
```

**Pointer fix** — save base and compute offset each iteration:

```ptx
// Before loop:
ld.param.u64 %proj_base, [projection_ptr];

// Inside loop:
mad.wide.u32 %proj_addr, %i, 4, %proj_base;  // offset from BASE, not accumulating
ld.global.f32 %proj_val, [%proj_addr];
```

---

## Phase 4: Remove Dead Stubs

**Priority: Low.** These kernels are loaded into PTX modules but never called
from Rust code. They add dead code to the compiled PTX and noise to the
kernel load list.

### §C4.1 Remove `holo_spectral_idct_2d`

- **PTX definition:** `gpu_holo.rs:1736–1746` (pure stub)
- **Loaded at:** `load_spectral_kernel()` line 386
- **Callers:** Zero — `get_func("holo_spectral", "holo_spectral_idct_2d")` never called
- **Rationale:** Separable row + col IDCT (Phase 1 + 2) achieves the same result.

Action: Remove PTX entry, remove from `load_ptx()` kernel list.

### §C4.2 Remove `holo_rph_generate_projection`

- **PTX definition:** `gpu_holo.rs:1868–1878` (pure stub)
- **Loaded at:** `load_rph_kernel()` line 411
- **Callers:** Zero — `get_func("holo_rph", "holo_rph_generate_projection")` never called
- **Rationale:** RPH flow receives pre-computed projections from fragment data.
  On-GPU projection generation was planned but never needed.

Action: Remove PTX entry, remove from `load_ptx()` kernel list.

### §C4.3 Verification

After removal, existing tests must still pass. Specifically verify:
- `load_spectral_kernel()` succeeds without the removed entries
- `load_rph_kernel()` succeeds without the removed entries
- All Phase 4 (GPU-CODEC-PIPELINE-TDD) tests still pass

---

## Phase 5: LRDF Batched Outer Product

**Trust Boundary:** The batched outer product computes the sum of multiple rank-1
matrices in a single kernel launch:

```
output[i][j] += Σ_{c=0}^{C-1} sigma[c] * u[c][i] * v[c][j]
```

The single-component `holo_lrdf_outer_product` kernel works correctly and serves
as both reference and pattern.

**Current State:** `holo_lrdf_outer_product_batched` (PTX lines 1954–1965) is a
pure stub. The Rust code calls the single kernel in a loop as a workaround
(`accumulate_lrdf` at line 1371).

**Reference:** Working single kernel at `gpu_holo.rs:1888–1951`

### §L5.1 Single Component Matches Unbatched

`batched(num_components=1)` must produce identical output to `single(sigma, u, v)`.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_lrdf_batched_single_matches_unbatched() {
    if !cuda_available() { return; }

    let rows = 4;
    let cols = 3;
    let u = vec![1.0f32, 2.0, 3.0, 4.0];
    let v = vec![5.0f32, 6.0, 7.0];
    let sigma = 2.0f32;

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_lrdf_kernel().unwrap();

    // Single kernel
    let single_result = ctx.lrdf_outer_product(&u, &v, sigma, rows, cols).unwrap();
    let single_host = ctx.copy_to_host(&single_result).unwrap();

    // Batched kernel with 1 component (same data packed into batched layout)
    let sigmas = vec![sigma];
    let batched_result = ctx.lrdf_outer_product_batched(
        &u, &v, &sigmas, rows, cols
    ).unwrap();
    let batched_host = ctx.copy_to_host(&batched_result).unwrap();

    for (i, (&s, &b)) in single_host.iter().zip(batched_host.iter()).enumerate() {
        assert_eq!(s.to_bits(), b.to_bits(),
            "Batched(1) != single at {}: single={}, batched={}", i, s, b);
    }
}
```

### §L5.2 Multi-Component Accumulation

Batched with `C` components must equal `C` sequential single calls accumulated.

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_lrdf_batched_multi_matches_sequential() {
    if !cuda_available() { return; }

    let rows = 3;
    let cols = 2;
    let num_components = 3;

    // 3 components with distinct u/v/sigma
    let u_all = vec![
        1.0f32, 0.5, 0.25,  // u0
        0.1, 0.2, 0.3,      // u1
        2.0, 1.0, 0.5,      // u2
    ];
    let v_all = vec![
        1.0f32, 2.0,    // v0
        3.0, 4.0,       // v1
        0.5, 1.5,       // v2
    ];
    let sigmas = vec![1.0f32, 2.0, 0.5];

    let mut ctx = GpuHoloContext::new(0).unwrap();
    ctx.load_lrdf_kernel().unwrap();

    // Sequential: 3 single calls accumulated into one buffer
    // ... launch single kernel 3 times, accumulating ...

    // Batched: single call with all 3 components
    let batched_result = ctx.lrdf_outer_product_batched(
        &u_all, &v_all, &sigmas, rows, cols
    ).unwrap();
    let batched_host = ctx.copy_to_host(&batched_result).unwrap();

    for (i, (&seq, &bat)) in sequential_host.iter().zip(batched_host.iter()).enumerate() {
        assert!(
            (seq - bat).abs() < 1e-5,
            "Batched vs sequential at {}: sequential={:.6}, batched={:.6}", i, seq, bat
        );
    }
}
```

### §L5.3 GPU vs CPU Proptest

```rust
mod lrdf_batched_proptest {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]
        #[test]
        fn batched_outer_product_matches_cpu(
            rows in 2usize..16,
            cols in 2usize..16,
            num_components in 1usize..4,
            values in proptest::collection::vec(-5.0f32..5.0f32, 512),
        ) {
            if !cuda_available() { return Ok(()); }

            // ... extract u, v, sigma from values for each component ...
            // ... CPU: nested loop computing sum of outer products ...
            // ... GPU: batched kernel ...
            // ... compare within 1e-4 ...
        }
    }
}
```

### §L5.4 Implementation Notes

**Memory layout:**
- `u_ptr` → `[u0[0..rows], u1[0..rows], ..., u_{C-1}[0..rows]]`
- `v_ptr` → `[v0[0..cols], v1[0..cols], ..., v_{C-1}[0..cols]]`
- `sigma_ptr` → `[s0, s1, ..., s_{C-1}]`

**PTX pattern:** Extend the working single kernel with an outer loop:

```ptx
// Each thread handles one (row, col) output element
// Loop over components:
mov.u32 %comp, 0;
COMP_LOOP:
    setp.ge.u32 %p, %comp, %num_components;
    @%p bra COMP_DONE;

    // Load sigma[comp]
    // Load u[comp * rows + row]
    // Load v[comp * cols + col]
    // Accumulate: sum += sigma * u * v

    add.u32 %comp, %comp, 1;
    bra COMP_LOOP;
COMP_DONE:
    // Store accumulated sum to output
```

---

## Implementation Order

| Phase | Kernels | Impact | Depends On |
|-------|---------|--------|------------|
| 1 | Spectral IDCT rows | Partial spectral reconstruction | — |
| 2 | Spectral IDCT cols | Full spectral reconstruction | Phase 1 |
| 3 | RPH XORShift + pointer fix | Full RPH reconstruction | — |
| 4 | Remove dead stubs | Cleanup | Phases 1–3 |
| 5 | LRDF batched outer product | Performance optimization | — |

Phases 1+2, 3, and 5 are independent and can be developed in parallel.
Phase 4 should be done after Phases 1–3 to avoid merge conflicts with the
PTX constant strings.

---

## Test Count Summary

| Phase | Specification | Property | Total |
|-------|:-:|:-:|:-:|
| 1. Spectral IDCT rows | 2 | 1 | 3 |
| 2. Spectral IDCT cols | 2 | 1 | 3 |
| 3. RPH PRNG fix | 4 | 0 | 4 |
| 4. Remove dead stubs | 1 | 0 | 1 |
| 5. LRDF batched | 2 | 1 | 3 |
| **Total** | **11** | **3** | **14** |

Plus updates to 2 existing tests (§S2.4).

---

## Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-03 | Codebase audit | Initial roadmap from DD-8 analysis. Covers 6 kernels: 3 implement, 1 fix, 2 remove. |
