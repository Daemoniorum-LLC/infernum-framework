# GPU Compression Pipeline - TDD Roadmap

## Goal
Enable running 70B+ quantized LLMs on 24GB VRAM (ADA 4500) by decompressing and dequantizing weights directly on GPU.

## TDD Approach
For each feature:
1. **Red**: Write failing tests that define the expected behavior
2. **Green**: Implement minimal code to pass tests
3. **Refactor**: Clean up while keeping tests green

---

## Phase 1: GPU Dequantization Kernel (Critical Path)

### 1.1 INT4 Dequantization
**Why**: INT4 quantization gives 8x compression; essential for 70B models on 24GB.

**Tests First** (`gpu_dequant.rs`):
```rust
#[test]
fn test_int4_dequant_basic() {
    // 8 INT4 values packed into 4 bytes → 8 F16 values
    let packed: [u8; 4] = [0x10, 0x32, 0x54, 0x76]; // values 0,1,2,3,4,5,6,7
    let scale: f16 = f16::from_f32(0.5);
    let zero_point: i8 = 0;

    let result = gpu_dequant_int4(&packed, scale, zero_point);

    assert_eq!(result, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
}

#[test]
fn test_int4_dequant_with_zero_point() { ... }

#[test]
fn test_int4_dequant_block_256() { ... }  // GPTQ block size

#[test]
fn test_int4_dequant_to_tensor() { ... }
```

**Implementation**:
- [ ] Create `gpu_dequant.rs` module
- [ ] Define `GpuDequantContext` struct
- [ ] Write PTX kernel for INT4→F16 conversion
- [ ] Implement block-level dequantization (GPTQ/AWQ compatible)

### 1.2 INT8 Dequantization
**Tests First**:
```rust
#[test]
fn test_int8_dequant_basic() {
    let values: [i8; 8] = [-128, -64, 0, 32, 64, 96, 127, 100];
    let scale: f16 = f16::from_f32(0.01);

    let result = gpu_dequant_int8(&values, scale);
    // Expected: values * scale
}

#[test]
fn test_int8_dequant_per_channel() { ... }
```

**Implementation**:
- [ ] Add INT8 kernel to `gpu_dequant.rs`
- [ ] Support per-tensor and per-channel scales

---

## Phase 2: Fused Decompress + Dequantize

### 2.1 LZ4 + INT4 Fused Kernel
**Why**: Avoid intermediate buffer; decompress directly to dequantized F16.

**Tests First**:
```rust
#[test]
fn test_fused_lz4_int4_dequant() {
    // Compress INT4 packed data with LZ4
    let int4_packed = create_int4_test_data(1024);
    let compressed = lz4_compress(&int4_packed);
    let scale = f16::from_f32(0.1);

    // Fused operation: LZ4 decompress → INT4 unpack → scale → F16
    let result = gpu_fused_lz4_int4(&compressed, scale, shape);

    // Verify against CPU reference
    let reference = cpu_lz4_decompress_then_dequant(&compressed, scale);
    assert_tensors_close(&result, &reference, 1e-3);
}

#[test]
fn test_fused_preserves_tensor_shape() { ... }

#[test]
fn test_fused_handles_large_tensors() { ... }  // 100M+ params
```

**Implementation**:
- [ ] Create `gpu_fused.rs` module
- [ ] Write fused PTX kernel (decompress → dequant in registers)
- [ ] Benchmark vs sequential operations

---

## Phase 3: Warp-Parallel LZ4 Optimization

### 3.1 Parallel Literal Copy
**Why**: Current kernel uses 1 thread; can use 32 threads (warp) for memcpy.

**Tests First**:
```rust
#[test]
fn test_warp_parallel_correctness() {
    // Same test data as single-threaded, verify identical output
    let (compressed, expected) = create_lz4_test_vectors();

    let single_result = gpu_lz4_decompress_single(&compressed);
    let warp_result = gpu_lz4_decompress_warp(&compressed);

    assert_eq!(single_result, warp_result);
}

#[test]
fn test_warp_parallel_large_literals() {
    // Block with 64KB literals - should benefit most from parallelism
    let data = vec![0xAB; 65536];
    let compressed = lz4_compress(&data);

    let result = gpu_lz4_decompress_warp(&compressed);
    assert_eq!(result, data);
}
```

**Implementation**:
- [ ] Modify PTX to use warp-level primitives (`__shfl_sync`, etc.)
- [ ] Parallel literal copy using all 32 threads
- [ ] Benchmark improvement

---

## Phase 4: Direct GPU Tensor Creation

### 4.1 Zero-Copy Candle Integration
**Why**: Current path: GPU buffer → CPU copy → Candle tensor. Wasteful.

**Tests First**:
```rust
#[test]
fn test_direct_gpu_tensor_f16() {
    let data = create_f16_test_data(1024);
    let compressed = lz4_compress(&data);

    let tensor = gpu_decompress_to_cuda_tensor(&compressed, &[32, 32], DType::F16);

    assert!(tensor.device().is_cuda());
    assert_eq!(tensor.dims(), &[32, 32]);
    // Verify data without copying to CPU
    let sum = tensor.sum_all()?.to_scalar::<f32>()?;
    assert_close!(sum, expected_sum, 1e-2);
}

#[test]
fn test_tensor_usable_in_matmul() {
    let weights = gpu_decompress_to_cuda_tensor(...);
    let input = Tensor::randn(..., &Device::Cuda(0))?;

    // Should work without any copies
    let output = input.matmul(&weights)?;
    assert!(output.device().is_cuda());
}
```

**Implementation**:
- [ ] Research Candle's `from_cuda_slice` or equivalent
- [ ] Wrap CudaSlice in Candle tensor without copy
- [ ] Update `decompress_to_tensor` to use direct path

---

## Phase 5: Streaming Pipeline

### 5.1 Async Decompression with CUDA Streams
**Why**: Overlap disk I/O with H2D transfer with decompression.

**Tests First**:
```rust
#[test]
fn test_streaming_correctness() {
    let layers = create_multi_layer_hct(10);  // 10 layers

    // Streaming should produce same result as sequential
    let streaming_result = load_hct_streaming(&layers);
    let sequential_result = load_hct_sequential(&layers);

    for (s, seq) in streaming_result.iter().zip(sequential_result.iter()) {
        assert_tensors_equal(s, seq);
    }
}

#[test]
fn test_streaming_faster_than_sequential() {
    let layers = create_multi_layer_hct(20);

    let streaming_time = time(|| load_hct_streaming(&layers));
    let sequential_time = time(|| load_hct_sequential(&layers));

    // Streaming should be at least 1.5x faster
    assert!(streaming_time < sequential_time * 0.67);
}
```

**Implementation**:
- [ ] Create CUDA stream pool
- [ ] Implement async H2D transfer
- [ ] Pipeline: read(N+1) | transfer(N) | decompress(N-1)

---

## Phase 6: HCT Format V2

### 6.1 Checksums & Validation
**Tests First**:
```rust
#[test]
fn test_hct_v2_checksum_valid() {
    let hct = create_hct_v2_with_checksum(&data);
    assert!(validate_hct_checksum(&hct).is_ok());
}

#[test]
fn test_hct_v2_checksum_detects_corruption() {
    let mut hct = create_hct_v2_with_checksum(&data);
    hct[100] ^= 0xFF;  // Flip a byte

    assert!(matches!(
        validate_hct_checksum(&hct),
        Err(HctError::ChecksumMismatch { .. })
    ));
}

#[test]
fn test_hct_v2_metadata() {
    let hct = HctV2Builder::new()
        .compression(Compression::Lz4)
        .quantization(Quantization::Int4)
        .block_size(65536)
        .build(&data)?;

    let meta = read_hct_v2_metadata(&hct)?;
    assert_eq!(meta.compression, Compression::Lz4);
    assert_eq!(meta.quantization, Quantization::Int4);
}
```

**Implementation**:
- [ ] Design HCT v2 header format
- [ ] Add XXH3 checksum support
- [ ] Add quantization metadata fields
- [ ] Backward compatibility with v1

---

## Implementation Order & Dependencies

```
Phase 1.1 (INT4 Dequant) ──┬──> Phase 2.1 (Fused LZ4+INT4)
Phase 1.2 (INT8 Dequant) ──┘           │
                                       v
Phase 3.1 (Warp LZ4) ──────────> Phase 5.1 (Streaming)
                                       │
Phase 4.1 (Direct Tensor) ─────────────┘
                                       │
Phase 6.1 (HCT v2) <───────────────────┘
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| 7B model load time | ~5s | <2s |
| 70B INT4 VRAM usage | N/A (OOM) | <22GB |
| Decompression throughput | ~900 MB/s (CPU) | >10 GB/s (GPU) |
| End-to-end tokens/sec | TBD | Baseline + 20% |

---

## Getting Started

```bash
# Run existing tests
cd /home/user/workspace/nyx/infernum/infernum-complete
RUSTUP_TOOLCHAIN=1.91-x86_64-unknown-linux-gnu cargo test gpu_lz4

# Run with CUDA (when on GPU machine)
RUSTUP_TOOLCHAIN=1.91-x86_64-unknown-linux-gnu cargo test --features cuda

# Benchmarks
RUSTUP_TOOLCHAIN=1.91-x86_64-unknown-linux-gnu cargo bench --bench weight_loading
```

## Next Step
Start with **Phase 1.1**: Write the INT4 dequantization tests, then implement the kernel.
