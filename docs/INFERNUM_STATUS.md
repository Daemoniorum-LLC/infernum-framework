# Infernum Status Report

**Last Updated:** 2026-01-13

## Quick Status

| Component | Status | Notes |
|-----------|--------|-------|
| HoloTensor/HCT Loading | FIXED | CPU caching + RAM eviction implemented |
| 70B Model Inference | WORKING (SLOW) | ~9 min/token due to 64GB RAM limit |
| Llama-3.2-1B (LRDF) | WORKING | Fast, coherent output (~0.4s for 30 tokens) |
| Llama-3.2-1B (Spectral) | CORRUPTED | Checksum mismatch - needs reconversion |
| SmolLM2-135M | WORKING | Test baseline |
| Lazy Layer Loading | WORKING | LazyLlama, LazyQwen2 |
| GPU Reconstruction | WORKING | LRDF, RAW format support |
| RAM Cache Eviction | WORKING | LRU eviction keeps RAM within 64GB budget |

## Recent Fixes

### Llama-3.2-1B Debugging (2026-01-13)

**Investigation:** Debugged reported "garbled output" from Llama-3.2-1B HCT models.

**Findings:**
- `llama-3.2-1b-lrdf` (146 HCT files) - **WORKS PERFECTLY**
  - Coherent output: "Paris. It is the most populous city in France..."
  - Fast inference: ~0.4s for 30 tokens
  - Clean startup, no errors

- `llama-3.2-1b-spectral` (146 HCT files) - **CORRUPTED**
  - Error: `fragment 21 checksum mismatch: expected b165417b39d1e809, got dfea452f6bd34e44`
  - Affects `embed_tokens.weight` tensor
  - Container crashes repeatedly

- `llama-3.2-1b-spectral-hct3-new` - **EMPTY** (no files)

**Action:** Updated docker-compose.override.yml to use working LRDF variant.

**Root Cause:** The spectral HCT conversion produced corrupted files (checksum validation failing). LRDF encoding works correctly.

### RAM Cache Eviction (2026-01-13)

**Problem:** Linux OOM killer was killing Infernum during inference:
- CPU cache grew unbounded without eviction
- RAM usage exceeded 64GB budget, triggering OOM kill
- Container would restart repeatedly, losing all cached tensors

**Fix:**
- Added LRU-based RAM cache eviction to `tiered_loading.rs`
- `evict_from_ram_if_needed()` maintains RAM at 90% of budget
- Added `cpu_lru_order` (VecDeque) for LRU tracking
- Added `cpu_cache_bytes` (AtomicUsize) for usage tracking

**Result:**
- Container runs 15+ minutes without OOM restart
- RAM stays at ~58GB within 64GB budget
- Inference is slow (~9 min/token) because 70B model needs ~160GB to cache all layers

### TieredHoloLoader CPU Caching (2026-01-12)

**Problem:** TieredHoloLoader cached tensors on GPU instead of CPU, causing:
- Layer eviction cleared tensors from VRAM but also cache
- Next access required full HCT decompression (~30s)
- 70B inference was ~5 minutes per token

**Fix:**
- `tiered_loading.rs`: Tensors now cached on CPU, transferred to GPU on demand
- `lazy_llama.rs`: Removed `clear_prefix()` from eviction
- `engine.rs`: Disabled LazyVarBuilder cache in HoloTensor eager path

**Expected Result:**
- Layer reload: ~100ms (CPU→GPU) instead of ~30s (HCT decompress)
- 70B inference: ~1 tok/s instead of ~5 min/tok

**Details:** See `HOLOTENSOR_AUDIT.md` for full analysis

## Architecture Overview

```
Memory Tiers:
┌─────────────────────────────────────────────────────────────┐
│  VRAM (24GB)     │  RAM (60GB)      │  Disk (HCT files)    │
│  Active layers   │  CPU tensor      │  Compressed          │
│  7 layers @3GB   │  cache           │  holographic         │
│  DecoderLayer    │  Fast reload     │  tensors             │
└─────────────────────────────────────────────────────────────┘

Flow: HCT → GPU Reconstruct → CPU Cache → GPU (on layer load)
                                   ↑
                              Fast path (~100ms)
```

## Key Files

| File | Purpose |
|------|---------|
| `holotensor/tiered_loading.rs` | Tiered memory loader with CPU caching |
| `models/lazy_llama.rs` | Lazy Llama with layer eviction |
| `models/lazy_qwen2.rs` | Lazy Qwen2 (reference implementation) |
| `lazy_varbuilder.rs` | On-demand tensor loading |
| `gpu_holo.rs` | GPU LRDF/spectral reconstruction |
| `hct.rs` | HCT file format loader |
| `engine.rs` | Model loading orchestration |

## Test Commands

```bash
# Rebuild with fix
cd /home/crook/dev2/workspace/nyx/infernum
CARGO_INCREMENTAL=0 cargo build --release -p abaddon --features cuda

# Start Infernum
docker compose -f /home/crook/dev2/workspace/daemoniorum/docker-compose.yml up -d infernum

# Test 70B inference
curl -s http://localhost:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, my name is",
    "max_tokens": 20,
    "model": "llama-70b"
  }'

# Watch layer eviction logs
docker logs -f daemoniorum-infernum-1 2>&1 | grep -E "(evict|layer|cache|CPU)"
```

## Known Issues

1. **Spectral HCT corruption** - `llama-3.2-1b-spectral` has checksum errors; use LRDF variant instead
2. **No safetensors fast path for 70B** - Would enable ~100ms loads
3. **70B inference speed** - ~9 min/token due to constant layer swapping (64GB RAM < 160GB needed)

## Documentation Index

- `HOLOTENSOR_AUDIT.md` - Detailed HoloTensor analysis and fixes
- `ROADMAP.md` - Production roadmap (stability, testing, security)
- `docs/RESULTS.md` - Rust vs Sigil comparison
- `docs/METHODOLOGY.md` - Development methodology
