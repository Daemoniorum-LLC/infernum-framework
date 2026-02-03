# HoloTensor Inference Pipeline - Audit Report

**Date:** 2026-01-12
**Status:** Comprehensive audit of HoloTensor/Infernum system

## Executive Summary

The HoloTensor inference pipeline has partial implementation with **critical discrepancies** between model implementations. The 70B model loads in ~15s but inference is slow (~5min/token) due to a **bug in lazy_llama.rs** that clears the RAM tier cache on VRAM eviction, forcing re-decompression from HCT files.

**Root Cause:** `lazy_llama.rs` calls `clear_prefix()` on layer eviction, which clears tensors from **both RAM and VRAM**. The `lazy_qwen2.rs` implementation does NOT call `clear_prefix()`, correctly preserving RAM-cached tensors.

---

## What's Working

### HoloTensor Format (HCT V3)
- [x] HCT file format reading and parsing
- [x] Zstd decompression of fragments
- [x] LRDF (Low-Rank Distributed Factorization) encoding support
- [x] RAW format support (`num_components=0xFFFFFFFF` marker)
- [x] Inline format support (single-fragment tensors)
- [x] GPU reconstruction for LRDF (both SVD and RAW formats)
- [x] CPU fallback for 1D tensors (layernorm weights)

### Model Loading
- [x] SmolLM2-135M loads and runs inference correctly
- [x] Llama-3.1-70B loads in ~15 seconds (from 6.4GB HCT)
- [x] TieredHoloLoader implements progressive quality loading
- [x] GPU context initialization and kernel loading

### Tiered Memory System (Design)
- [x] TieredConfig with VRAM/RAM budgets defined
- [x] MemoryTier enum (Vram, Ram, Disk)
- [x] LRU eviction tracking in loaded_layers
- [x] 60GB LruCache in LazyVarBuilder (lazy_varbuilder.rs:71-77)

---

## What's Broken

### Critical: Layer Tensor Cache Eviction Bug

**Location:** `nyx/infernum/crates/abaddon/src/models/lazy_llama.rs:349-364`

```rust
fn evict_lru_layer(&mut self) {
    if let Some(layer_idx) = self.lru_order.first().copied() {
        tracing::debug!(layer = layer_idx, "Evicting decoder layer");
        self.loaded_layers.remove(&layer_idx);
        self.lru_order.remove(0);
        self.layer_evictions += 1;

        // CRITICAL BUG: This clears tensors from BOTH RAM and VRAM caches
        // When a layer is evicted from VRAM, the reconstructed tensors
        // should remain in RAM (60GB cache) for fast reload
        let prefix = format!("model.layers.{}.", layer_idx);
        let (evicted_count, evicted_bytes) = self.lazy_vb.clear_prefix(&prefix);
        // ...
    }
}
```

**Impact:** Every layer swap requires re-decompression from HCT files instead of loading from RAM cache:
- 70B model: 80 layers, 7 VRAM slots
- Each token requires ~73 layer swaps (80-7=73)
- Each swap re-decompresses ~2GB of tensors from HCT
- Result: ~5 minutes per token instead of expected ~1 second

**Compare to working implementation:** `lazy_qwen2.rs:343-363`
```rust
fn evict_lru_layer(&mut self) {
    if let Some(layer_idx) = self.lru_order.first().copied() {
        tracing::debug!(layer = layer_idx, "Evicting decoder layer (preserving KV cache)");

        // Correctly preserves KV cache without clearing tensor cache
        if let Some(mut layer) = self.loaded_layers.remove(&layer_idx) {
            let cache = layer.self_attn.take_cache();
            if cache.seq_len() > 0 {
                self.layer_caches.insert(layer_idx, cache);
            }
        }

        self.lru_order.remove(0);
        self.layer_evictions += 1;
        // NOTE: Does NOT call clear_prefix() - tensors stay in RAM cache!
    }
}
```

### Secondary Issues

1. **Llama-3.2-1B produces garbled output**
   - HCT conversion may have dtype or quantization issues
   - Need to verify conversion parameters

2. **No safetensors fast path for 70B**
   - TieredHoloLoader supports safetensors but none pre-converted
   - Would enable ~100ms loads instead of ~100s reconstructions

3. **Missing documentation**
   - No ROADMAP.md in infernum directory
   - No STATUS.md tracking what's working
   - Session knowledge gets lost between conversations

---

## File Reference

### Core Files

| File | Purpose | Status |
|------|---------|--------|
| `holotensor/tiered_loading.rs` | Progressive quality loading | Working |
| `lazy_varbuilder.rs` | On-demand tensor loading with LRU cache | Working |
| `models/lazy_llama.rs` | Lazy Llama model | **BUG: clear_prefix on evict** |
| `models/lazy_qwen2.rs` | Lazy Qwen2 model | Working correctly |
| `gpu_holo.rs` | GPU LRDF/Spectral reconstruction | Working (RAW fix applied) |
| `hct.rs` | HCT file loader | Working |

### Haagenti Files

| File | Purpose | Status |
|------|---------|--------|
| `haagenti/holotensor.rs` | CPU HoloTensor decoder | Working |
| `haagenti-cuda/` | GPU DCT/IDCT kernels | Working |

---

## Memory Architecture (Design vs Reality)

### Designed Tiering
```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Tiering                            │
├─────────────────────────────────────────────────────────────┤
│  VRAM (22GB)     │  RAM (60GB)      │  Disk (HCT files)    │
│  - Hot layers    │  - Warm layers   │  - Cold storage      │
│  - 7 layers @3GB │  - 20+ layers    │  - All 80 layers     │
│                  │  - Fast reload   │  - Slow decompress   │
└─────────────────────────────────────────────────────────────┘

Flow: Disk → RAM cache → VRAM (on layer load)
      VRAM → RAM cache (on layer evict, stays cached)
```

### Actual Behavior (lazy_llama.rs bug)
```
┌─────────────────────────────────────────────────────────────┐
│                    Actual Memory Flow                        │
├─────────────────────────────────────────────────────────────┤
│  VRAM (22GB)     │  RAM (CLEARED!)  │  Disk (HCT files)    │
│  - Hot layers    │  - Cache emptied │  - Re-decompressed   │
│  - 7 layers @3GB │    on eviction   │    every swap!       │
└─────────────────────────────────────────────────────────────┘

Flow: Disk → VRAM (on layer load)
      VRAM → DELETED (on evict, RAM cache cleared too)
      Next access: Disk → VRAM again (full re-decompress)
```

---

## Fix Required

### Immediate Fix

Remove `clear_prefix` call from `lazy_llama.rs:356-364`:

```rust
fn evict_lru_layer(&mut self) {
    if let Some(layer_idx) = self.lru_order.first().copied() {
        tracing::debug!(layer = layer_idx, "Evicting decoder layer");

        // Remove the layer from loaded_layers HashMap
        // This drops the DecoderLayer struct (VRAM tensors)
        self.loaded_layers.remove(&layer_idx);
        self.lru_order.remove(0);
        self.layer_evictions += 1;

        // REMOVED: clear_prefix call
        // The reconstructed tensors should remain in LazyVarBuilder's
        // 60GB LRU cache for fast reload when this layer is needed again.
        // Only VRAM DecoderLayer structs are dropped, not cached tensors.
    }
}
```

### Optional: Add KV Cache Preservation

Match `lazy_qwen2.rs` pattern by preserving KV cache across evictions:

```rust
/// Persisted KV caches for evicted layers (add to LazyLlama struct)
layer_caches: HashMap<usize, Box<dyn KvCache>>,

fn evict_lru_layer(&mut self) {
    if let Some(layer_idx) = self.lru_order.first().copied() {
        // Extract and preserve the KV cache before evicting
        if let Some(mut layer) = self.loaded_layers.remove(&layer_idx) {
            let cache = layer.self_attn.take_cache();
            if cache.seq_len() > 0 {
                self.layer_caches.insert(layer_idx, cache);
            }
        }
        self.lru_order.remove(0);
        self.layer_evictions += 1;
    }
}
```

---

## Roadmap

### Phase 1: Architectural Fix (COMPLETE 2026-01-12)
- [x] Identified root cause: TieredHoloLoader caches on GPU, not CPU RAM
- [x] Reverted premature fix - `clear_prefix` IS needed until architecture fixed
- [x] **Refactored TieredHoloLoader** to use CPU caching
- [x] Removed `clear_prefix` call from `lazy_llama.rs` eviction
- [ ] Add KV cache preservation to `lazy_llama.rs` (match lazy_qwen2.rs) - OPTIONAL

### Phase 1.5: RAM Cache Eviction Fix (COMPLETE 2026-01-13)
- [x] **Root cause:** OOM killer was killing Infernum during inference
- [x] **Issue:** CPU cache grew unbounded (no RAM eviction), exceeding 64GB budget
- [x] **Fix:** Added LRU-based RAM cache eviction to `tiered_loading.rs`
  - Added `cpu_lru_order: VecDeque<String>` for LRU tracking
  - Added `cpu_cache_bytes: AtomicUsize` for RAM usage tracking
  - Added `evict_from_ram_if_needed()` method with 90% budget target
  - Cache tuple changed from `(Tensor, MemoryTier)` to `(Tensor, MemoryTier, u64)`
- [x] **Result:** Container runs 15+ minutes without OOM restart
- [x] **Performance note:** 64GB RAM budget can hold ~30 layers; 70B model needs ~80 layers
  - Each token still requires reloading ~50 evicted layers from HCT
  - Expected ~9 minutes per token until more RAM available or safetensors used

### Implementation Details (2026-01-12)

**Changes made to `tiered_loading.rs`:**
1. Renamed `tensors` → `cpu_cache`, `device` → `inference_device`
2. `load_tensor()` now:
   - Checks CPU cache first, transfers to GPU on hit (~100ms)
   - After reconstruction, caches tensor on CPU, returns GPU copy
3. `clear_prefix()` now clears CPU cache (safe to call, but optional)

**Changes made to `lazy_llama.rs`:**
1. `evict_lru_layer()` no longer calls `clear_prefix()`
2. Only drops GPU DecoderLayer struct, CPU cache preserved

### Expected Performance After Fix
- Layer reload: ~100ms (CPU→GPU transfer) instead of ~30s (HCT decompress)
- 70B inference: ~1 tok/s instead of ~5 min/tok
- Memory: 24GB VRAM for active layers, 60GB RAM for cache

### Previous Test Results (2026-01-12, BEFORE FIX)
- **OOM at layer 24** after evicting all loaded layers
- Issue: TieredHoloLoader stored tensors on CUDA device, not CPU RAM
- The "RAM cache" was actually VRAM cache - tensors filled GPU memory

### Root Cause (FIXED)

The deeper issue was:

1. TieredHoloLoader created tensors on `self.device` (CUDA)
2. When cached in `self.tensors` HashMap, they consumed VRAM
3. Evicting from `loaded_layers` didn't free VRAM because tensors were still cached
4. Removing clear_prefix made OOM worse (more cached tensors)

**Fix applied:** TieredHoloLoader now:
1. Caches tensors on CPU (RAM tier)
2. Transfers to CUDA on-demand when layer requested
3. Only VRAM holds active DecoderLayer tensors

---

## Proper Architectural Fix

### Current (Broken) Flow
```
HCT File → GPU Reconstruction → GPU Tensor (cached in HashMap)
                                      ↓
                              DecoderLayer (GPU) uses same tensor
                                      ↓
                              Layer evicted → GPU tensor stays cached
                                      ↓
                              VRAM fills up → OOM
```

### Required Flow
```
HCT File → GPU Reconstruction → CPU Tensor (cached in HashMap)
                                      ↓
                              DecoderLayer request
                                      ↓
                              CPU → GPU transfer (clone to device)
                                      ↓
                              DecoderLayer (GPU) uses GPU tensor
                                      ↓
                              Layer evicted → GPU tensor freed
                                      ↓
                              CPU tensor remains cached (fast reload)
```

### Implementation Changes

**File:** `tiered_loading.rs`

1. **Add CPU device for caching:**
```rust
pub struct TieredHoloLoader {
    // ...existing fields...
    /// CPU device for caching reconstructed tensors
    cache_device: Device,  // Always Device::Cpu
    /// Target device for inference (usually CUDA)
    inference_device: Device,
}
```

2. **Modify load_tensor to cache on CPU:**
```rust
fn load_tensor_internal(&self, path: &Path, name: &str) -> Result<(Tensor, bool)> {
    // Reconstruct on GPU (fast)
    let gpu_tensor = self.reconstruct_holotensor_gpu(path)?;

    // Transfer to CPU for caching
    let cpu_tensor = gpu_tensor.to_device(&Device::Cpu)?;

    // Cache the CPU tensor
    self.cache_tensor(name, cpu_tensor.clone());

    // Return GPU tensor for immediate use
    Ok((gpu_tensor, true))
}
```

3. **Add method to get GPU tensor from cache:**
```rust
pub fn get_tensor_for_inference(&self, name: &str) -> Result<Tensor> {
    // Check cache for CPU tensor
    if let Some(cpu_tensor) = self.get_cached(name) {
        // Transfer to inference device (GPU)
        return cpu_tensor.to_device(&self.inference_device);
    }

    // Not cached, load from HCT
    self.load_tensor(name)
}
```

4. **Update TensorProvider impl:**
```rust
impl TensorProvider for TieredHoloLoader {
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError> {
        // Get from cache (CPU) and transfer to requested device
        let cpu_tensor = self.load_tensor(name)?;
        cpu_tensor.to_device(device)
    }

    fn clear_prefix(&self, prefix: &str) -> (usize, u64) {
        // Only clears CPU cache now - safe to call
        // GPU memory is freed when DecoderLayer is dropped
        self.clear_cpu_cache(prefix)
    }
}
```

### After Fix

With this change:
- `clear_prefix` can be **removed** from `lazy_llama.rs`
- CPU cache (60GB) holds reconstructed tensors
- GPU memory (24GB) only holds active layers (7 layers max)
- Layer reload: CPU→GPU transfer (~100ms) instead of HCT decompress (~30s)

### Phase 2: Verification (READY)
- [ ] Test SmolLM2-135M inference
- [ ] Test Llama-3.1-70B inference (should be ~1 tok/s with fix)
- [ ] Verify RAM cache utilization via stats

### Phase 3: Optimization
- [ ] Pre-convert 70B model to safetensors for fast path
- [ ] Investigate Llama-3.2-1B garbled output
- [ ] Profile GPU vs CPU reconstruction time
- [ ] Add KV cache preservation to lazy_llama.rs (match lazy_qwen2.rs)

### Phase 4: Documentation (COMPLETE)
- [x] Create INFERNUM_STATUS.md with current state
- [x] Document HoloTensor fixes in HOLOTENSOR_AUDIT.md
- [ ] Document HCT conversion process
- [ ] Add architecture diagrams

---

## Test Commands

```bash
# Rebuild Infernum with fix
cd /home/crook/dev2/workspace/nyx/infernum
CARGO_INCREMENTAL=0 cargo build --release -p abaddon --features cuda

# Test with 70B model
docker compose -f /home/crook/dev2/workspace/daemoniorum/docker-compose.yml up -d infernum

# Watch logs for layer eviction patterns
docker logs -f daemoniorum-infernum-1 2>&1 | grep -E "(evict|layer|cache)"
```

---

## Comprehensive Audit Findings (2026-01-12)

### Issues Fixed

1. **Critical: TieredHoloLoader CPU Caching** (`tiered_loading.rs`)
   - Root cause: Tensors cached on GPU instead of CPU
   - Fix: `cpu_cache` field, tensors cached on CPU, transferred to GPU on demand

2. **Critical: lazy_llama.rs clear_prefix** (`models/lazy_llama.rs`)
   - Root cause: Layer eviction cleared CPU cache, forcing re-decompression
   - Fix: Removed `clear_prefix()` call from `evict_lru_layer()`

3. **Secondary: HoloTensor eager loading cache** (`engine.rs`)
   - Issue: LazyVarBuilder cache not disabled in HoloTensor eager path
   - Fix: Added disabled cache config to match lazy loading path

### Code Audit Summary

| File | Status | Notes |
|------|--------|-------|
| `tiered_loading.rs` | FIXED | CPU caching implemented |
| `lazy_llama.rs` | FIXED | No longer clears cache on eviction |
| `lazy_qwen2.rs` | OK | Reference implementation, preserves KV cache |
| `lazy_varbuilder.rs` | OK | Cache disabled when using TieredHoloLoader |
| `engine.rs` | FIXED | Both HoloTensor paths disable LazyVarBuilder cache |
| `holotensor/memory.rs` | OK | Memory tier management |
| `holotensor/provider.rs` | OK | Progressive quality loading |
| `gpu_holo.rs` | OK | GPU reconstruction kernels |
| `hct.rs` | OK | HCT file loading |

### Potential Optimizations (Not Bugs)

1. **KV Cache Preservation** (`lazy_llama.rs`)
   - lazy_qwen2.rs preserves KV cache across evictions
   - lazy_llama.rs could benefit from same pattern
   - Status: OPTIONAL - would improve multi-turn conversations

2. **Emergency Cache Clear** (`lazy_varbuilder.rs`)
   - `clear_all()` only clears LazyVarBuilder's cache, not TensorProvider's
   - Could add full cache clear for extreme OOM recovery
   - Status: LOW PRIORITY - current architecture handles normal OOM

### Architecture Verification

Memory flow is now correct:
```
HCT File → GPU Reconstruction → CPU Cache → GPU on demand
                                   ↓
                            Fast reload (~100ms)
                                   ↓
                            Layer eviction
                                   ↓
                            GPU tensors freed, CPU cache preserved
```

---

## References

- `tiered_loading.rs`: TieredConfig defaults (vram: 20GB, ram: 64GB)
- `lazy_varbuilder.rs`: CacheConfig defaults (max_memory: 60GB, max_entries: 20000)
- `lazy_qwen2.rs`: Working reference implementation with KV cache preservation
