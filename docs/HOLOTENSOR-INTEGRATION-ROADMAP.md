# HoloTensor Integration Roadmap

**Date:** 2026-01-13
**Updated:** 2026-01-13 (Phase 1, 2 & 3 COMPLETE)
**Purpose:** Implementation plan to wire together existing HoloTensor infrastructure

---

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| TieredHoloLoader | ✅ Working | `abaddon/src/holotensor/tiered_loading.rs` |
| LRU RAM eviction | ✅ Working | 64GB budget, evict_from_ram_if_needed() |
| NVMe cache API | ✅ **ENABLED** | `INFERNUM_CACHE_DIR` env var |
| GPU DCT kernels | ✅ **ENABLED** | `haagenti-cuda/src/dct_gpu.rs` |
| FFT-based DCT | ✅ **ENABLED** | `haagenti-cuda/cufft` feature |
| GPU Zstd decode | ⚠️ CPU fallback | `haagenti-cuda/src/zstd_gpu.rs` |
| LRDF encoding | ✅ Working | Llama-3.2-1B-lrdf produces coherent output |
| Spectral encoding | ❌ Corrupted | Checksum mismatch in test files |
| Speculative config | ✅ **ENABLED** | `INFERNUM_DRAFT_MODEL` env var |
| Speculative engine | ✅ **ENABLED** | `infernum-server/src/speculative_engine.rs` |
| Speculative decode | ✅ **ENABLED** | `abaddon/src/speculative_405b.rs` wired |
| Speculative stats | ✅ **ENABLED** | `/api/speculative/stats` endpoint |

**Performance After Phase 1 & 2 (1B model):**
- NVMe cache: 5-6x faster layer loads (230ms vs 1.5s)
- GPU IDCT: Working for all 2D tensors (1D falls back to CPU)
- cuFFT: Available for tensors >4096 (will auto-engage for 70B+)

---

## Phase 1: Enable NVMe Cache ✅ COMPLETE

### Goal
Reduce subsequent layer loads from ~100s to ~100ms by caching decompressed tensors on NVMe.

### Implementation (Done)

**1.1 Environment variable support added to `engine.rs`**
```rust
// Automatically enable NVMe cache via INFERNUM_CACHE_DIR env var
if let Ok(cache_dir) = std::env::var("INFERNUM_CACHE_DIR") {
    loader = loader.with_safetensors_dir(cache_path);
}
```

**1.2 Docker Compose configuration**
```yaml
# docker-compose.override.yml
environment:
  INFERNUM_CACHE_DIR: "/cache"
volumes:
  - infernum-cache:/cache
```

**1.3 On-the-fly cache population added to `tiered_loading.rs`**
- Tensors are automatically saved to safetensors format after HCT reconstruction
- First load: HCT → GPU IDCT → save to cache
- Subsequent loads: mmap from cache (~1000x faster)

### Measured Impact
| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Base load (1B) | 6.5s | 2.6s | 2.5x |
| Layer load | 1.5s | 230ms | 5-6x |
| MLP tensor (32MB) | 1s | 55-62ms | 15-18x |
| Total cache size (1B) | - | 2.4GB | - |

---

## Phase 2: Enable GPU IDCT ✅ COMPLETE

### Goal
Use haagenti-cuda GPU kernels for IDCT reconstruction instead of CPU.

### Blockers (Resolved)
1. ~~cuFFT not available in WSL2~~ → Works in Docker (nvidia/cuda image has libcufft)
2. ~~GpuDctContext initialization hangs~~ → Never actually hung, was working

### Implementation (Done)

**2.1 Enabled cufft feature in `abaddon/Cargo.toml`**
```toml
haagenti-gpu = ["cuda", "dep:haagenti-cuda", "haagenti-cuda/cufft"]
```

**2.2 Added CUDA library path in Docker startup**
```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

**2.3 GPU reconstruction working for all 2D tensors**
- 2D tensors (MLP, attention): GPU IDCT
- 1D tensors (layernorms): CPU fallback (expected)
- cuFFT auto-engages for dimensions > 4096

### Measured Impact (1B model)
| Tensor | Shape | GPU Time |
|--------|-------|----------|
| embed_tokens | 128256×2048 | ~4s |
| q_proj | 2048×2048 | ~15ms |
| gate_proj | 8192×2048 | ~320ms |

**For 70B+ models:** FFT-based DCT will engage (dims > 4096), expect ~150x faster IDCT

---

## Phase 3: Wire Speculative Decoding (Medium-term)

### Goal
Use small draft model (1B-8B) to generate candidate tokens, verify with 405B in single pass.

### Prerequisites
- Phase 1 (NVMe cache) complete - target model needs fast layer loading
- Draft model fully in VRAM (~2-4GB for 1B-8B)

### Implementation Status

**3.1 Server Configuration ✅ COMPLETE**

Environment variables added to `infernum-server/src/config.rs`:
```bash
INFERNUM_DRAFT_MODEL=/path/to/small/model    # Draft model (1B-8B)
INFERNUM_SPECULATIVE_TOKENS=5                 # Tokens per round (1-16)
```

Configuration builder methods:
```rust
Config::builder()
    .model("holo:///path/to/405b")
    .draft_model("/path/to/1b")
    .speculative_tokens(5)
    .build()
```

Helper method `config.is_speculative_enabled()` returns true when both models are set.

**3.2 AppState Integration ✅ COMPLETE**

Added `speculative_engine` field to `AppState`:
```rust
pub struct AppState {
    pub engine: RwLock<Option<Arc<Engine>>>,
    pub speculative_engine: RwLock<Option<Arc<SpeculativeEngine>>>,  // NEW
    // ... other fields
}
```

**3.3 SpeculativeEngine Module ✅ COMPLETE**

Created `speculative_engine.rs` with full implementation:
- `SpeculativeEngine` - wrapper for server integration
- `SpeculativeEngineConfig` - configuration struct with all settings
- `SpeculativeEngineBuilder` - fluent builder API
- `SpeculativeEngineError` - error types
- `load_draft_model()` - loads Qwen2/Llama draft model to VRAM
- `load_target_model()` - uses TieredHoloLoader for large model streaming
- `generate()` - string-based generation with tokenization
- `generate_tokens()` - token-based generation

**3.4 Server Initialization ✅ COMPLETE**

Server::run() initializes SpeculativeEngine when configured:
```rust
if self.config.is_speculative_enabled() {
    let engine = SpeculativeEngineBuilder::new()
        .draft_model(draft_model)
        .target_model(target_model)
        .num_draft_tokens(speculative_tokens)
        .build()
        .await?;
    *self.state.speculative_engine.write().await = Some(Arc::new(engine));
}
```

**3.5 Load Draft Model to VRAM ✅ COMPLETE**

Draft model loading implemented with auto-detection:
```rust
// Detects architecture (Qwen2/Llama) from config.json
let arch = ArchitectureType::from_config(&model_config);

// Loads full model into VRAM (not streamed)
match arch {
    ArchitectureType::Qwen2 => Qwen2::new(cfg, vb)?,
    ArchitectureType::Llama => Llama::new(cfg, vb)?,
}
```

**3.6 Wire Inference Handlers ✅ COMPLETE**

Chat completions handler now routes through speculative engine:
```rust
// Check for speculative engine
if let Some(spec_engine) = state.speculative_engine.read().await.as_ref() {
    // Use speculative decoding (3-4x faster for large models)
    match spec_engine.generate(&prompt, max_tokens) {
        Ok(content) => return speculative_response(...),
        Err(_) => {} // Fall back to regular generation
    }
}
// Regular generation fallback
engine.generate(gen_request).await
```

**3.7 Stats Endpoint ✅ COMPLETE**

Added `/api/speculative/stats` endpoint for monitoring:
```json
{
  "enabled": true,
  "rounds": 100,
  "acceptance_rate": 0.82,
  "tokens_per_round": 4.1,
  "speedup": 3.8,
  ...
}
```

### Remaining Work
1. ~~Add `SpeculativeEngine` field to `AppState`~~ ✅
2. ~~Server initialization for speculative mode~~ ✅
3. ~~Implement full model loading in `SpeculativeEngine::new()`~~ ✅
4. ~~Route inference through speculative decoder when available~~ ✅
5. ~~Add `/api/speculative/stats` endpoint for monitoring~~ ✅
6. End-to-end testing with real models (pending)

### Expected Impact
| Metric | Before | After |
|--------|--------|-------|
| Tokens per forward pass | 1 | 3-5 (with 80% acceptance) |
| Effective speedup | 1x | 3-4x |

### Example Usage (Docker)
```yaml
# docker-compose.override.yml
services:
  infernum:
    environment:
      INFERNUM_MODEL: "/models/llama-405b-holo"
      INFERNUM_DRAFT_MODEL: "/models/llama-1b"
      INFERNUM_SPECULATIVE_TOKENS: "5"
      INFERNUM_CACHE_DIR: "/cache"
    volumes:
      - /path/to/models:/models:ro
      - infernum-cache:/cache
```

---

## Phase 4: Layer Prefetching (Medium-term)

### Goal
Overlap next layer decompression with current layer computation.

### Implementation

**4.1 Async layer prefetch**
```rust
// While layer N is computing on GPU...
let prefetch_handle = tokio::spawn(async move {
    loader.load_tensor(&format!("model.layers.{}.q_proj.weight", n + 1)).await
});

// Compute layer N
let output = layer_n.forward(&input)?;

// Await prefetch before layer N+1
let next_weights = prefetch_handle.await?;
```

**4.2 Double-buffering strategy**
- Buffer A: Current layer weights (in use)
- Buffer B: Next layer weights (loading)
- Swap after each layer

### Expected Impact
| Metric | Before | After |
|--------|--------|-------|
| Layer load overlap | 0% | 50-70% |
| Effective layer time | load + compute | max(load, compute) |

---

## Phase 5: Fix Spectral Encoding (Low Priority)

### Goal
Repair corrupted spectral HCT files or regenerate them.

### Current Issue
- `llama-3.2-1b-spectral` has checksum mismatch on fragment 21
- LRDF variant works correctly

### Options
1. **Regenerate spectral files** - Use haagenti compression pipeline
2. **Stick with LRDF** - Works fine for attention-heavy models
3. **Debug checksum issue** - May be compression bug

### Not blocking - LRDF is sufficient for now.

---

## Implementation Priority

| Phase | Effort | Impact | Dependencies |
|-------|--------|--------|--------------|
| **1. NVMe Cache** | Low (config change) | High (100x faster reload) | NVMe storage |
| **2. GPU IDCT** | Medium (fix cuFFT) | High (20-50x faster IDCT) | CUDA toolkit |
| **3. Speculative** | Medium | High (3-4x effective) | Phase 1 |
| **4. Prefetch** | Medium | Medium (50% overlap) | Phase 2 |
| **5. Spectral Fix** | Low priority | Low | None |

---

## Quick Wins (Today)

1. **Add NVMe cache path to docker-compose.override.yml**
```yaml
services:
  infernum:
    volumes:
      - /mnt/nvme/infernum-cache:/cache
    environment:
      INFERNUM_CACHE_DIR: "/cache"
```

2. **Update TieredHoloLoader initialization to use cache**

3. **Test with 1B model first** - Verify NVMe cache works before 70B/405B

---

## Success Metrics

| Milestone | Target | Measurement |
|-----------|--------|-------------|
| NVMe cache working | Layer reload < 200ms | Log timing in load_tensor() |
| GPU IDCT working | MLP tensor < 50ms | Benchmark haagenti-cuda |
| Speculative working | 3+ tokens/verification | Log acceptance rate |
| 70B practical | < 30s/token | End-to-end timing |
| 405B viable | < 60s/token | End-to-end timing |

---

## Files Modified (Phase 3) ✅ COMPLETE

| File | Change |
|------|--------|
| `infernum-server/src/speculative_engine.rs` | ✅ Full implementation with model loading |
| `infernum-server/src/server.rs` | ✅ Added speculative_engine to AppState, wired to chat_completions, stats endpoint |
| `infernum-server/src/config.rs` | ✅ Added draft_model, speculative_tokens |
| `infernum-server/src/lib.rs` | ✅ Export SpeculativeEngine types |
| `infernum-server/Cargo.toml` | ✅ Added candle-core, candle-nn deps |
| `abaddon/src/speculative_405b.rs` | ✅ Added DraftModel impls for Qwen2, Llama |

## Phase 3 Implementation Summary

- **SpeculativeEngine**: Full model loading with Qwen2/Llama draft support
- **Draft model**: Auto-detects architecture, loads fully into VRAM
- **Target model**: Uses TieredHoloLoader with NVMe cache and lazy layer loading
- **Chat handler**: Routes non-streaming requests through speculative engine
- **Stats endpoint**: `/api/speculative/stats` for monitoring acceptance rates
- **Fallback**: Graceful fallback to regular generation on errors

---

*Next step: End-to-end testing with real models (70B target + 1B draft)*
