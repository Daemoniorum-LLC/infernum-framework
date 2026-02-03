# HCT Inference Optimization Roadmap

## Overview

This document tracks optimization opportunities for HCT (HoloTensor Compressed) model inference. Our 70B Llama model currently runs at **~7 minutes per token** on 24GB VRAM. Target: **<5 seconds per token**.

## Current Status

### What's Working
- OOM issue fixed (triple caching eliminated)
- 70B model running on 24GB VRAM with layer-by-layer loading
- LazyLlama with LRU eviction (max_loaded_layers=7)
- TieredHoloLoader managing VRAM/RAM/Disk tiers
- HCT decompression and tensor reconstruction

### Current Bottleneck Analysis (Updated 2026-01-12)

**Good news: GPU IDCT is already enabled for 2D tensors!**

| Tensor Type | Method | Time | Notes |
|-------------|--------|------|-------|
| Large 2D (28672×8192) | GPU IDCT | ~1.3s | mlp.gate_proj, up_proj, down_proj |
| Medium 2D (8192×8192) | GPU IDCT | ~0.4s | q_proj, o_proj |
| Small 2D (1024×8192) | GPU IDCT | ~0.03s | k_proj, v_proj |
| 1D (8192) | CPU IDCT | ~0.001s | layernorm weights (fallback) |

Per-layer breakdown:
| Phase | Time | Notes |
|-------|------|-------|
| 7 large tensors | ~6s | GPU IDCT |
| 2 small tensors | ~0.002s | CPU fallback for 1D |
| Forward pass | ~1s | Attention + MLP |
| **Total per layer** | **~7s** | × 80 layers = ~9.3 min/token |

**Root Cause**: The 28672×8192 tensors (MLP) dominate. These are 235M params each!

---

## Already Implemented Infrastructure

These features exist in Haagenti/Abaddon but are NOT integrated with LazyLlama:

### Haagenti (Compression Library)

| Feature | Location | Status |
|---------|----------|--------|
| GPU DCT/IDCT kernels | `haagenti-cuda` | Ready (18-45x faster) |
| FFT-based DCT (cufft) | `haagenti-cuda` | Ready (42-79x faster for >4K) |
| GPU Zstd decompression | `haagenti-cuda` | API ready, targets 4 GB/s |
| Dictionary compression | `haagenti-zstd` | Ready (+20% ratio) |
| Streaming preview | `haagenti-streaming` | Ready |
| Speculative prefetch | `haagenti-speculative` | Ready |
| Adaptive streaming | `haagenti-streaming` | Ready |

### Abaddon (Inference Engine)

| Feature | Location | Status |
|---------|----------|--------|
| ProgressiveHoloLoader | `holotensor/progressive.rs` | Implemented |
| StreamingHoloContext | `holotensor/streaming.rs` | Implemented |
| MultiGpuHoloContext | `holotensor/multi_gpu.rs` | Implemented |
| HotReloadController | `holotensor/hot_reload.rs` | Implemented |
| AdaptiveQualityController | `holotensor/adaptive.rs` | Implemented |
| GpuDctContext | `holotensor/gpu_dct.rs` | Implemented |

**Key Insight**: We planned ahead! Most optimization infrastructure is ready.

---

## Optimization Phases

### Phase 1: Wire Haagenti FFT-IDCT into GpuHoloContext (CRITICAL - Expected 40-80x speedup)

**Impact**: MLP tensors ~1.3s → ~0.03s each = **~2s/layer → ~2.7 min/token**

#### Problem Discovered

Abaddon has **two separate GPU implementations**:
1. `haagenti_cuda::GpuDctContext` - Has FFT-based O(n log n) IDCT (ready but unused!)
2. `gpu_holo::cuda::GpuHoloContext` - Custom PTX kernels with **placeholder O(n²) IDCT**

The `GpuHoloContext` kernels in `gpu_holo.rs` contain placeholder code:
```
// Simple IDCT sum (placeholder - real impl would use shared mem)
// Simplified IDCT placeholder - just copy for now
```

#### Solution: Use Haagenti's FFT-IDCT

Per PERFORMANCE-BASELINES.md, FFT-based IDCT is 42-79x faster:

| Tensor Size | Direct IDCT | FFT IDCT | Speedup |
|-------------|-------------|----------|---------|
| 4096×4096 | 134ms | 3.2ms | 42x |
| 8192×8192 | 536ms | 6.8ms | 79x |

**Integration Point**: `abaddon/src/gpu_holo.rs` → use `haagenti_cuda::GpuDctContext`

```rust
// In GpuHoloContext
use haagenti_cuda::GpuDctContext;

impl GpuHoloContext {
    // Add haagenti's FFT-capable context
    dct_ctx: GpuDctContext,

    pub fn finalize_spectral(&self, accumulator: &AccumulatorState) -> Result<CudaSlice<f32>> {
        // Instead of calling holo_spectral_idct_* PTX kernels,
        // use haagenti's FFT-based IDCT:
        let (height, width) = accumulator.shape;
        let coeffs = self.copy_to_host(&accumulator.data)?;

        // This auto-selects FFT for large dimensions
        let reconstructed = self.dct_ctx.idct_2d(&coeffs, width, height)?;

        // Copy back to GPU
        self.device.htod_sync_copy(&reconstructed)
    }
}
```

**Tasks**:
1. [x] Enable `cufft` feature in workspace (Done: `Cargo.toml` updated)
2. [x] Add `GpuDctContext` to `GpuHoloContext` struct (Done: `gpu_holo.rs` updated)
3. [x] Replace PTX IDCT calls with `dct_ctx.idct_2d()` in `finalize_spectral()` (Done)
4. [x] Wire through LRDF reconstruction path → N/A: LRDF uses outer products, not IDCT
5. [x] Convert test model with Spectral encoding (Done: llama-3.2-1b-spectral)
6. [ ] Fix GpuDctContext initialization hang (blocks FFT-IDCT)
7. [ ] Fix Spectral encoding quality issue (garbage output)
8. [ ] Benchmark: measure MLP tensor reconstruction time

**Blockers (2026-01-12)**:
- **cuFFT not available in WSL**: `/usr/lib/wsl/lib` doesn't include libcufft, blocking FFT path
- **GpuDctContext hangs**: `with_device()` hangs during MemoryPool allocation (conflict with existing CUDA context?)
- **Spectral quality issue**: Converted model produces garbage output, needs encoder/decoder investigation

**Workaround Applied**: FFT_THRESHOLD set to 1B to disable FFT path, uses direct PTX IDCT instead.

**Note**: Current 70B model uses LRDF encoding. FFT-IDCT speedup only applies to Spectral-encoded models.
To test: convert a model with `--encoding spectral` flag.

### Phase 2: Async Layer Prefetch (Expected 50%+ overlap)

**Impact**: While computing layer N, load layer N+1 in background.

Currently, layer loading is synchronous:

```rust
// Current (blocking)
let layer = self.lazy_llama.get_layer(layer_idx)?;  // Blocks until loaded
// ... forward pass ...
```

**Target Architecture**:

```rust
// With prefetch
impl LazyLlama {
    pub async fn prefetch_layer(&self, layer_idx: usize) {
        self.prefetch_tx.send(layer_idx).await;
    }

    pub fn get_layer_with_prefetch(&self, current: usize) -> &Layer {
        // Start prefetching next layer
        self.prefetch_layer(current + 1);
        // Return current (already prefetched or load now)
        self.get_layer(current)
    }
}
```

**Tasks**:
1. [ ] Add `tokio::sync::mpsc` channel for prefetch requests
2. [ ] Spawn background prefetch task in `LazyLlama::new()`
3. [ ] Integrate `haagenti_speculative::PrefetchManager` for prediction
4. [ ] Track prefetch hit rate metrics

### Phase 3: Pipelined Decompression (Expected 30%+ overlap)

**Impact**: Overlap Zstd decompress → GPU IDCT → Forward pass.

```
Current (sequential):
  [Zstd]→[IDCT]→[Forward]→[Zstd]→[IDCT]→[Forward]→...

Target (pipelined):
  [Zstd₁]→[IDCT₁]→[Forward₁]
          [Zstd₂]→[IDCT₂]→[Forward₂]
                  [Zstd₃]→[IDCT₃]→[Forward₃]
```

**Integration**: Use `StreamingHoloContext` from `holotensor/streaming.rs`:

```rust
use crate::holotensor::streaming::StreamingHoloContext;

impl TieredHoloLoader {
    pub fn with_streaming(mut self, ctx: StreamingHoloContext) -> Self {
        self.streaming_ctx = Some(ctx);
        self
    }
}
```

**Tasks**:
1. [ ] Create pipeline struct with 3 stages
2. [ ] Use CUDA streams for parallel IDCT/forward
3. [ ] Integrate `StreamingHoloContext` for fragment pipelining
4. [ ] Add metrics: stage utilization, stall time

### Phase 4: Batch IDCT Operations (Expected 3-5x speedup)

**Impact**: Process multiple tensors in single GPU kernel launch.

Currently, we decompress tensors one at a time. Batch processing amortizes kernel launch overhead (45x faster per PERFORMANCE-BASELINES.md).

```rust
// Current
for tensor_name in layer_tensors {
    let tensor = loader.load_tensor(tensor_name)?;
}

// Target (batch)
let tensors = loader.load_tensors_batch(&layer_tensor_names)?;
```

**Tasks**:
1. [ ] Add `load_tensors_batch()` to TieredHoloLoader
2. [ ] Use `GpuDctContext::batch_idct_2d()`
3. [ ] Group tensors by shape for optimal batching
4. [ ] Benchmark batch sizes (8, 16, 32 tensors)

### Phase 5: Dictionary Compression (Expected +20% ratio, faster decompress)

**Impact**: Smaller HCT files = faster disk/network I/O.

```rust
use haagenti::{ZstdDict, ZstdDictCompressor};

// Train dictionary from model weights
let dict = ZstdDict::train(&weight_samples, 64 * 1024)?;

// Use for compression/decompression
let compressor = ZstdDictCompressor::new(&dict);
```

**Tasks**:
1. [ ] Train dictionary during model conversion
2. [ ] Store dictionary in model directory
3. [ ] Load dictionary in TieredHoloLoader
4. [ ] Benchmark compression ratio improvement

---

## Performance Targets

| Metric | Current | Phase 1 (FFT) | Phase 1+2 | Full |
|--------|---------|---------------|-----------|------|
| Layer load time | ~7s | ~1.5s | ~1s | ~0.5s |
| Time per token | ~9.3 min | ~2 min | ~1.3 min | **<30s** |
| MLP tensor (28K×8K) | ~1.3s | ~0.03s | ~0.03s | ~0.03s |
| GPU utilization | ~60% | ~75% | ~85% | ~90% |
| Memory bandwidth | Good | Excellent | Excellent | Excellent |

**Note**: Current GPU IDCT is already 18x faster than CPU. FFT will add another 40-80x for large tensors.

---

## Implementation Order

```
Phase 1 (GPU IDCT)     ←── START HERE - Biggest impact
    ↓
Phase 2 (Prefetch)     ←── Quick win, uses existing infra
    ↓
Phase 3 (Pipelining)   ←── Requires Phase 1+2
    ↓
Phase 4 (Batching)     ←── Enhancement to Phase 1
    ↓
Phase 5 (Dictionary)   ←── Can be done in parallel
```

---

## Quick Start: Phase 1 Implementation (Wire Haagenti FFT-IDCT)

**Status**: GPU IDCT works but uses placeholder O(n²) kernels. Need to wire in haagenti's FFT-IDCT.

### Step 1: Verify cuFFT is available (Done!)

```bash
ls -la /usr/lib/x86_64-linux-gnu/libcufft.so*
# Output: libcufft.so.11 -> libcufft.so.11.0.1.95 (154MB)
```

### Step 2: Enable cufft feature (Done!)

```toml
# infernum/Cargo.toml line 151
haagenti-cuda = { path = "../haagenti/crates/haagenti-cuda", features = ["cufft"] }
```

### Step 3: Integrate GpuDctContext into GpuHoloContext

**File**: `abaddon/src/gpu_holo.rs`

```rust
// Add to GpuHoloContext struct
#[cfg(feature = "cufft")]
use haagenti_cuda::GpuDctContext;

pub struct GpuHoloContext {
    device: Arc<CudaDevice>,
    // ... existing fields ...

    /// Haagenti's FFT-capable DCT context
    #[cfg(feature = "cufft")]
    dct_ctx: GpuDctContext,
}

impl GpuHoloContext {
    pub fn new(device_id: usize) -> Result<Self, GpuHoloError> {
        let device = CudaDevice::new(device_id).map_err(|e| /* ... */)?;

        #[cfg(feature = "cufft")]
        let dct_ctx = GpuDctContext::new(device.clone())
            .map_err(|e| GpuHoloError::KernelLoad { message: e.to_string() })?;

        Ok(Self {
            device,
            #[cfg(feature = "cufft")]
            dct_ctx,
            // ... rest ...
        })
    }

    /// Performs 2D IDCT using haagenti's FFT-based implementation
    pub fn finalize_spectral(&self, accumulator: &AccumulatorState) -> Result<CudaSlice<f32>, GpuHoloError> {
        let (height, width) = (accumulator.rows, accumulator.cols);

        // Copy coefficients to host for haagenti processing
        let coeffs = self.copy_to_host(&accumulator.data)?;

        // Use haagenti's FFT-IDCT (auto-selects FFT for large dims)
        #[cfg(feature = "cufft")]
        let reconstructed = self.dct_ctx.idct_2d(&coeffs, width, height)
            .map_err(|e| GpuHoloError::KernelExec { message: e.to_string() })?;

        // Copy result back to GPU
        self.device.htod_sync_copy(&reconstructed)
            .map_err(|e| GpuHoloError::MemoryCopy { message: e.to_string() })
    }
}
```

### Step 4: Add feature flag to abaddon

```toml
# abaddon/Cargo.toml
[features]
cufft = ["haagenti-cuda/cufft"]
cuda = ["dep:cudarc", "cufft"]  # Auto-enable cufft with cuda
```

### Step 5: Rebuild and benchmark

```bash
CARGO_INCREMENTAL=0 cargo build --release -p infernum --features cuda

# Restart server
pkill infernum
./target/release/infernum serve --model /path/to/model --port 8081

# Test and measure reconstruction times
curl -X POST http://localhost:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hi", "max_tokens": 1}'
```

### Expected Results

Before (placeholder O(n²) IDCT):
```
GPU HoloTensor reconstruction complete tensor=mlp.gate_proj.weight shape=[28672, 8192]  # ~1.3s
```

After (haagenti FFT-IDCT):
```
GPU HoloTensor reconstruction complete tensor=mlp.gate_proj.weight shape=[28672, 8192]  # ~0.03s
```

---

## Verification Commands

```bash
# Build with GPU support
cargo build --release -p infernum --features haagenti-gpu

# Run benchmark
cargo run --release -p infernum --features haagenti-gpu -- \
    --model /home/crook/models/llama-3.1-70b-hct-holo \
    --benchmark-layers

# Compare GPU vs CPU IDCT
cargo run --release --example benchmark_idct -p haagenti-cuda
```

---

## References

- `haagenti/docs/HOLOTENSOR-DESIGN.md` - Full HoloTensor architecture (100% complete)
- `haagenti/docs/PERFORMANCE-BASELINES.md` - DCT/IDCT benchmarks
- `haagenti/docs/roadmap/HAAGENTI-IMPLEMENTATION-SPEC.md` - Integration APIs
- `haagenti/HANDOFF-HCT-INFERENCE.md` - HCT inference integration guide
- `infernum/docs/ENHANCEMENT-ROADMAP.md` - General Infernum enhancements

---

*Created: 2026-01-12*
*Updated: 2026-01-12*
*Status: Phase 1 Code Complete, Blocked by cuFFT availability and GpuDctContext hang. GPU direct IDCT working for 2D tensors.*
