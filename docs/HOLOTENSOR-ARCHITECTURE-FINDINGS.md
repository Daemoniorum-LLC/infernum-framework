# HoloTensor Architecture Findings

**Date:** 2026-01-13
**Purpose:** Comprehensive compilation of HoloTensor/HCT architecture from Infernum and Haagenti documentation

---

## Executive Summary

HoloTensor is a holographic compression system for neural network weights that enables running 405B+ parameter models on consumer hardware (24GB VRAM + 64GB RAM). The architecture uses a four-tier memory hierarchy with progressive loading and speculative decoding to achieve practical inference speeds.

---

## Memory Tier Architecture

From `abaddon/src/holotensor/mod.rs:7-12`:

```
┌─────────────────────────────────────────────────────────────┐
│                    HoloMemoryManager                        │
│  Tracks fragment locations across memory tiers              │
│  (VRAM ← RAM ← NVMe ← Network)                              │
└─────────────────────────────────────────────────────────────┘
```

### Tier Specifications

| Tier | Capacity | Bandwidth | Latency | Purpose |
|------|----------|-----------|---------|---------|
| VRAM | 24 GB | 900 GB/s (HBM3) | ~1µs | Hot layers, active inference |
| RAM | 64 GB | 50 GB/s (DDR5) | ~100ns | Warm layers, CPU cache |
| NVMe | ~1 TB | 7 GB/s (Gen4) | ~10µs | Cold layers, decompressed cache |
| Network | Unlimited | 1 GB/s (10GbE) | ~1ms | Remote storage, federation |

Source: `holotensor/mod.rs:167-175`

### Memory Configuration

```rust
MemoryConfig::builder()
    .vram_budget_mb(20_000)      // 20 GB usable VRAM
    .ram_budget_mb(64_000)       // 64 GB RAM budget
    .nvme_cache_path("/path")    // NVMe cache for decompressed tensors
    .enable_network_streaming(false)
    .build()
```

Source: `holotensor/tests.rs:85-89`, `holotensor/mod.rs:354-396`

---

## HCT Compression Format

### Compression Pipeline

```
Neural Network Weights (safetensors, ~810 GB for 405B)
    ↓
2D DCT Transform (spectral decomposition)
    ↓
Coefficient Retention (keep top 20-70% by importance)
    ↓
Fragment Serialization (bitmap + f16 coefficients)
    ↓
Zstd Compression (entropy coding)
    ↓
HCT V3 File (~40-80 GB for 405B)
```

### Compression Results

| Retention | Compression | 405B Size | Cosine Similarity | Output Quality |
|-----------|-------------|-----------|-------------------|----------------|
| 30% | 23.7x | ~17 GB | 0.889 | Garbage |
| 50% | 14.2x | ~28 GB | 0.966 | Garbage |
| 60% | 11.8x | ~34 GB | 0.982 | Degraded |
| **70%** | **10.2x** | **~40 GB** | **0.993** | **Good** |
| 80% | 8.9x | ~46 GB | 0.998 | Good |

Source: `haagenti/HANDOFF-HCT-INFERENCE.md:38-46`

**Key Insight:** Even 0.889 tensor cosine similarity produces garbage. Errors compound through transformer layers. Need ~0.993+ for usable output.

### Encoding Methods

| Method | Best For | Notes |
|--------|----------|-------|
| Spectral (DCT) | General weights | Requires FFT-IDCT for speed |
| LRDF (Low-Rank Distributed Factorization) | Attention matrices | Uses outer products, not IDCT |
| SVD | Low-rank attention (q/k/v/o_proj) | Optimal for rank 64-128 |
| Mixed Precision | All | FP16 top 20% + INT4 rest |

---

## Loading Architecture

### TieredHoloLoader

From `abaddon/src/holotensor/tiered_loading.rs`:

```rust
pub struct TieredHoloLoader {
    directory: PathBuf,              // HCT files
    safetensors_dir: Option<PathBuf>, // Fast path (NVMe cache)
    memory_manager: Arc<HoloMemoryManager>,
    stream_manager: Arc<StreamManager>,
    cpu_cache: RwLock<HashMap<String, Vec<u8>>>,
    cpu_lru_order: RwLock<VecDeque<String>>,
    // ... LRU eviction fields
}
```

### Loading Priority (line 535-544)

1. Check cache (already loaded)
2. Try safetensors directory (fast mmap, ~100ms)
3. Fall back to HoloTensor reconstruction (~100s for large tensors)

**Critical:** Safetensors fast path is **1000x faster** than HCT reconstruction.

### Safetensors Fast Path

```rust
// Enable NVMe cache for decompressed tensors
loader.with_safetensors_dir("/mnt/nvme/model-cache")
```

When enabled:
- First access: HCT → decompress → save to NVMe as safetensors
- Subsequent access: mmap from NVMe (~100ms)

Source: `tiered_loading.rs:381-407`

---

## GPU Decompression

### Current Performance

| Tensor Type | Shape | Method | Time |
|-------------|-------|--------|------|
| Large MLP | 28672×8192 | GPU IDCT | ~1.3s |
| Medium Attention | 8192×8192 | GPU IDCT | ~0.4s |
| Small KV | 1024×8192 | GPU IDCT | ~0.03s |
| 1D (layernorm) | 8192 | CPU fallback | ~0.001s |

Source: `infernum/docs/HCT-INFERENCE-OPTIMIZATION-ROADMAP.md:22-27`

### FFT-IDCT Optimization (Not Yet Wired)

| Tensor Size | Direct IDCT | FFT IDCT | Speedup |
|-------------|-------------|----------|---------|
| 4096×4096 | 134ms | 3.2ms | 42x |
| 8192×8192 | 536ms | 6.8ms | 79x |

Source: `haagenti/docs/PERFORMANCE-BASELINES.md`

**Blockers:**
- cuFFT not available in WSL2 (`/usr/lib/wsl/lib` missing libcufft)
- GpuDctContext hangs during initialization

---

## Haagenti Compression Library

### Benchmark Results

| Operation | Haagenti | Reference (libzstd) | Speedup |
|-----------|----------|---------------------|---------|
| Decompress 64KB text | 190 GB/s | 10 GB/s | **18.6x** |
| Decompress 1KB binary | 25 GB/s | 0.85 GB/s | **29.2x** |
| Compress 1KB | 1.15 GB/s | 0.4 GB/s | **2.9x** |

Source: `haagenti/BENCHMARK_REPORT.md`

### GPU Decompression Targets

| Operation | Target Throughput |
|-----------|-------------------|
| Zstd GPU decompress | 4 GB/s |
| Neural GPU decode | 1 GB/s |
| Codebook lookup | 10 GB/s |

Source: `haagenti/docs/roadmap/HAAGENTI-IMPLEMENTATION-SPEC.md:291-292`

---

## Speculative Decoding (405B)

From `abaddon/src/speculative_405b.rs`:

### Architecture

```rust
Speculative405B::new(
    draft_model,   // Small model (1B-8B) fully in VRAM
    target_model,  // Large model (405B) with layer streaming
    config,
)
```

### Performance Expectations

- Draft model: 1B-8B, generates 5-8 candidate tokens per round
- Target model: 405B, verifies all candidates in single forward pass
- Acceptance rate: 80%+ with well-matched models
- **Effective speedup: 3-4x**

Source: `speculative_405b.rs:5-11`

### Configuration

```rust
Speculative405BConfig {
    num_draft_tokens: 5,        // Candidates per round
    acceptance_threshold: 0.1,  // Low for high-quality 405B
    draft_temperature: 0.7,
    target_temperature: 0.7,
    greedy_draft: true,
}
```

---

## Optimization Roadmap

From `infernum/docs/HCT-INFERENCE-OPTIMIZATION-ROADMAP.md`:

### Current State (70B)

- Time per token: ~9 minutes
- Bottleneck: MLP tensors (28672×8192) at 1.3s each
- Per-layer: ~7s × 80 layers = 9.3 min/token

### Optimization Phases

| Phase | Optimization | Expected Impact |
|-------|--------------|-----------------|
| 1 | Wire FFT-IDCT | 40-80x speedup on MLP tensors |
| 2 | Async layer prefetch | 50%+ overlap |
| 3 | Pipelined decompression | 30%+ overlap |
| 4 | Batch IDCT operations | 3-5x kernel launch savings |
| 5 | Dictionary compression | +20% ratio, faster decompress |

### Performance Targets

| Metric | Current | After Phase 1 | Full Optimization |
|--------|---------|---------------|-------------------|
| Layer load time | ~7s | ~1.5s | ~0.5s |
| Time per token | ~9 min | ~2 min | **<30s** |
| GPU utilization | ~60% | ~75% | ~90% |

---

## Already Implemented Infrastructure

### Haagenti (Ready but Not Integrated)

| Feature | Location | Status |
|---------|----------|--------|
| GPU DCT/IDCT kernels | `haagenti-cuda` | Ready (18-45x faster) |
| FFT-based DCT (cufft) | `haagenti-cuda` | Ready (42-79x faster) |
| GPU Zstd decompression | `haagenti-cuda` | API ready |
| Dictionary compression | `haagenti-zstd` | Ready (+20% ratio) |
| Streaming preview | `haagenti-streaming` | Ready |
| Speculative prefetch | `haagenti-speculative` | Ready |

### Abaddon (Implemented)

| Feature | Location | Status |
|---------|----------|--------|
| ProgressiveHoloLoader | `holotensor/progressive.rs` | Implemented |
| StreamingHoloContext | `holotensor/streaming.rs` | Implemented |
| MultiGpuHoloContext | `holotensor/multi_gpu.rs` | Implemented |
| HotReloadController | `holotensor/hot_reload.rs` | Implemented |
| AdaptiveQualityController | `holotensor/adaptive.rs` | Implemented |

---

## Model Variants Tested

### Llama-3.2-1B

| Variant | Files | Status | Notes |
|---------|-------|--------|-------|
| `llama-3.2-1b-lrdf` | 146 HCT | **WORKING** | Coherent output, ~0.4s inference |
| `llama-3.2-1b-spectral` | 146 HCT | **CORRUPTED** | Checksum mismatch fragment 21 |
| `llama-3.2-1b-spectral-hct3-new` | 0 | Empty | No files |

### 70B Models

- HCT loading works with LRU eviction
- ~9 min/token (constant layer swapping)
- RAM cache eviction keeps within 64GB budget

---

## Key Integration Points

### Enabling NVMe Cache

```rust
let loader = TieredHoloLoader::new(config, &hct_dir)?
    .with_safetensors_dir("/mnt/nvme/model-cache");
```

### Enabling GPU Decompression

```rust
// Cargo.toml
haagenti-cuda = { features = ["cufft"] }

// Code
#[cfg(feature = "haagenti-gpu")]
let dct_ctx = GpuDctContext::with_device(device.clone())?;
```

### Enabling Speculative Decoding

```rust
let spec = Speculative405B::new(
    draft_model,  // LazyLlama 1B (fully loaded)
    target_model, // LazyLlama 405B (streaming)
    Speculative405BConfig::fast(),
);
```

---

## Open Questions

1. **Where is the 405B @ 4 tok/s calculation documented?**
   - Speculative gives 4x
   - But base speed needs to be ~1s/token for 4 tok/s effective
   - Current 70B is 9 min/token, so 405B would be ~15+ min/token base

2. **NVMe cache population strategy?**
   - First-run warmup: decompress all layers to NVMe
   - Subsequent runs: fast mmap loading
   - How long for initial warmup of 405B? (~2 min per PERFORMANCE-BASELINES.md)

3. **cuFFT availability in WSL2?**
   - Currently blocked
   - May need native Linux or Docker with CUDA

---

## References

| Document | Location | Purpose |
|----------|----------|---------|
| HCT Inference Handoff | `haagenti/HANDOFF-HCT-INFERENCE.md` | Compression pipeline details |
| Optimization Roadmap | `infernum/docs/HCT-INFERENCE-OPTIMIZATION-ROADMAP.md` | Performance targets |
| Performance Baselines | `haagenti/docs/PERFORMANCE-BASELINES.md` | Benchmark data |
| Implementation Spec | `haagenti/docs/roadmap/HAAGENTI-IMPLEMENTATION-SPEC.md` | API specifications |
| HoloTensor Design | `haagenti/docs/HOLOTENSOR-DESIGN.md` | Architecture overview |

---

## Code-Level Findings (from Haagenti and Abaddon source review)

### TieredHoloLoader Implementation (`abaddon/src/holotensor/tiered_loading.rs`)

The actual implementation shows:

```rust
pub struct TieredHoloLoader {
    config: TieredConfig,
    directory: PathBuf,
    safetensors_dir: Option<PathBuf>,  // NVMe cache path
    memory_manager: Arc<HoloMemoryManager>,
    cpu_cache: RwLock<HashMap<String, (Tensor, MemoryTier, u64)>>,
    cpu_lru_order: RwLock<VecDeque<String>>,
    cpu_cache_bytes: AtomicUsize,
    #[cfg(feature = "haagenti-gpu")]
    decompression_ctx: Option<Arc<HaagentiGpuContext>>,  // Zero-copy decompression
}
```

**Key implementation details:**
- CPU RAM (64GB) caches reconstructed tensors for fast reload (~100ms GPU transfer)
- GPU VRAM (24GB) only holds active DecoderLayer tensors
- LRU eviction with configurable budgets (`evict_from_ram_if_needed()`)
- Quality-aware placement using `QualityCurve` coefficients

### NVMe Cache Implementation

The NVMe cache is implemented via `with_safetensors_dir()`:

```rust
// Enable NVMe cache
loader.with_safetensors_dir("/mnt/nvme/model-cache")

// Loading priority (load_tensor, line 546-704):
// 1. Check cpu_cache (HashMap) → CPU→GPU transfer (~100ms)
// 2. Check safetensors_dir (NVMe) → mmap → GPU transfer (~100ms)
// 3. Fall back to HCT reconstruction → GPU IDCT (~30-100s)
```

**NVMe strategy:**
- First access: HCT → reconstruct → save to safetensors on NVMe
- Subsequent access: mmap from NVMe (1000x faster)

### Memory Tier Latency/Bandwidth (from `mod.rs:156-176`)

```rust
impl MemoryTier {
    pub fn typical_latency_ns(&self) -> u64 {
        match self {
            MemoryTier::Vram => 100,           // ~100ns (not µs as doc said)
            MemoryTier::Ram => 100_000,         // ~100µs
            MemoryTier::Nvme => 10_000_000,     // ~10ms (not µs as doc said)
            MemoryTier::Network => 50_000_000,  // ~50ms
        }
    }
}
```

**Corrected values:** NVMe is 10ms latency, not 10µs. This is per-tensor, not per-byte.

### Haagenti GPU Implementations

#### DCT GPU Kernels (`haagenti-cuda/src/dct_gpu.rs`, ~1788 lines)

- Full GPU DCT/IDCT implementation using CUDA via NVRTC (runtime compilation)
- Separable 2D DCT using 1D row/column transforms
- PTX caching for faster subsequent loads
- Default FFT threshold: 4096 (use FFT for dimensions > 4096)
- Performance: 18x speedup for 128x128, 45x for batch operations

#### FFT-based DCT (`haagenti-cuda/src/cufft_ffi.rs`, 540 lines)

Full cuFFT FFI bindings for O(n log n) DCT:

```rust
pub struct FftDctContext {
    device: Arc<CudaDevice>,
    plan_cache: HashMap<(usize, CufftType), CufftPlan>,
    fft_threshold: usize,  // Default 4096
}

// Performance from doc comments:
// | 1024x1024 | 2.1ms → 0.8ms | 2.6x speedup  |
// | 4096x4096 | 134ms → 3.2ms | 42x speedup   |
// | 8192x8192 | 536ms → 6.8ms | 79x speedup   |
```

#### Zstd GPU Decompression (`haagenti-cuda/src/zstd_gpu.rs`, ~1018 lines)

- GPU-accelerated Zstd sequence decoding
- FSE (Finite State Entropy) decoding tables on GPU
- **Critical finding:** Uses CPU fallback for actual decompression (`zstd::decode_all`)
- The GPU implementation handles sequence execution but core Zstd is still CPU

#### Zero-Copy Integration (`haagenti-cuda/src/lib.rs`)

```rust
// Architecture comment from lib.rs:
// Traditional: Disk → CPU RAM → Decompress (CPU) → GPU Transfer → Inference
// GPU Pipeline: Disk → Pinned Memory → GPU Transfer → Decompress (GPU) → Inference
```

### Haagenti Streaming/Speculative (NOT for LLM inference)

**Critical clarification:** These crates are for **image generation** (Dantalion), not LLM tensor loading:

#### haagenti-streaming (`lib.rs`)
```rust
// Real-Time Streaming for IMAGE generation:
// Timeline: t=0.0s [Start] → t=0.5s [Blob] → t=3.0s [Final]
// Preview modes: Instant, Balanced, Fast, Thumbnail
```

#### haagenti-speculative (`lib.rs`)
```rust
// Intent prediction for IMAGE generation:
// Keystrokes → Intent Predictor → Fragment Pre-Warmer
// Example: "portr..." → "portrait" → Load face attention
```

**LLM layer prefetching** is in `abaddon`, not haagenti.

### GPU Context Initialization

```rust
// From tiered_loading.rs:299-326
#[cfg(feature = "cuda")]
let gpu_context = match &device {
    Device::Cuda(_) => {
        match GpuHoloContext::new(0) {
            Ok(mut ctx) => {
                ctx.load_all_kernels()?;  // Load IDCT/LRDF kernels
                Some(RwLock::new(ctx))
            }
            Err(e) => None  // CPU fallback
        }
    }
    _ => None,
};
```

### Quality-Aware Placement

Uses `QualityCurve` from haagenti for placement decisions:

```rust
// From tiered_loading.rs:486-533
let quality_curve = QualityCurve::default();
let quality_impact = if info.is_attention {
    1.0  // Attention weights always prioritize VRAM
} else {
    info.importance * (1.0 + quality_curve.predict(min_fragments, total))
};

// Dynamic threshold based on min_quality target
let vram_threshold = 0.3 * self.config.min_quality;
```

---

## Updated Open Questions

1. **Where is the 405B @ 4 tok/s calculation documented?**
   - Still not found in documentation
   - Math: speculative 4x + FFT-IDCT 79x + NVMe cache 1000x = ???
   - Needs investigation into how these compound

2. **cuFFT availability in WSL2?**
   - cufft_ffi.rs implementation is ready
   - `#[link(name = "cufft")]` requires libcufft.so
   - Blocked on WSL2 driver paths

3. **Neural Compression (.nct)**
   - 10:1 compression using learned codebooks
   - `with_nct_dir()` enables this path
   - Requires codebooks.nct file
   - Not tested yet

---

## Summary of What's Implemented vs Documented

| Feature | Documentation | Code Status |
|---------|---------------|-------------|
| 4-tier memory | ✓ Documented | ✓ Implemented in abaddon |
| NVMe cache | ✓ Documented | ✓ Implemented via safetensors_dir |
| LRU eviction | ✓ Documented | ✓ Implemented with evict_from_ram_if_needed() |
| GPU DCT/IDCT | ✓ Documented | ✓ Implemented in haagenti-cuda |
| FFT-based DCT | ✓ Documented | ✓ Implemented in cufft_ffi.rs |
| GPU Zstd | ✓ Documented | ⚠️ CPU fallback (zstd::decode_all) |
| Streaming prefetch | ✓ Documented | ⚠️ Image generation only |
| Speculative prefetch | ✓ Documented | ⚠️ Image generation only |
| haagenti-gpu integration | ✓ Documented | ✓ Conditional compile (#[cfg(feature)]) |

---

*Compiled: 2026-01-13*
*Sources: Infernum and Haagenti documentation and source code review*
