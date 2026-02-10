# CUDA Adaptive Tiering Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-06
**Crate:** `abaddon`
**Dependencies:** `adaptive_tiering`, `cuda_inference`

---

## 1. Overview

This specification defines the integration of the Adaptive Memory Tiering system with the
`cuda_inference` backend, enabling efficient inference for models of any size on consumer
GPUs with limited VRAM.

### 1.1 Problem Statement

The current `cuda_inference` module has two loading modes:

1. **Eager loading** (`WeightStore`): Loads all weights to VRAM upfront
   - Fast inference (~10+ tk/s)
   - Fails if model exceeds VRAM

2. **Lazy loading** (`LazyWeightStore`): Loads layers on-demand with LRU eviction
   - Works for any model size
   - Extremely slow (~0.003 tk/s for 70B) due to sequential layer swapping

Neither mode uses the **3-tier memory hierarchy** (VRAM ← RAM ← NVMe) effectively.

### 1.2 Goal

Create a `TieredWeightStore` that:

1. Uses `AllocationPlan` from `adaptive_tiering` for intelligent placement
2. Maintains hot tensors in VRAM (GPU memory)
3. Keeps warm tensors in pinned RAM (fast PCIe transfer)
4. Stores cold tensors on NVMe (disk cache)
5. Prefetches tensors based on forward pass position
6. Achieves **10-100x speedup** over naive lazy loading

### 1.3 Design Principles

> **Memory is a hierarchy, not a binary.** VRAM vs "not VRAM" is the wrong model.
> The right model is: VRAM (0.1ms) ← RAM (1ms) ← NVMe (10ms).

> **Eager when possible, progressive when necessary.** If the model fits in VRAM+RAM,
> decompress everything upfront. Only use NVMe streaming for truly massive models.

> **Importance-aware placement.** Not all tensors are equal. Embeddings and edge
> layers matter more than middle-layer MLPs.

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CUDA Adaptive Tiering System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │  HCT Directory  │                                                        │
│  │  (model files)  │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │  ModelProfile   │────▶│ AllocationPlan- │────▶│  AllocationPlan     │   │
│  │  (tensor scan)  │     │ ner (importance │     │  (per-tensor        │   │
│  │                 │     │  scoring)       │     │   placement)        │   │
│  └─────────────────┘     └─────────────────┘     └──────────┬──────────┘   │
│                                                              │              │
│                          ┌───────────────────────────────────┘              │
│                          │                                                  │
│                          ▼                                                  │
│           ┌──────────────────────────────┐                                 │
│           │     LoadingBackend::select   │                                 │
│           │                              │                                 │
│           │  nvme_usage == 0 ? Eager     │                                 │
│           │                 : Progressive│                                 │
│           └──────────────┬───────────────┘                                 │
│                          │                                                  │
│           ┌──────────────┴──────────────┐                                  │
│           ▼                              ▼                                  │
│  ┌─────────────────┐          ┌─────────────────┐                          │
│  │  EagerLoader    │          │ ProgressiveLoad │                          │
│  │                 │          │                 │                          │
│  │  - Decompress   │          │  - NVMe cache   │                          │
│  │    all to RAM   │          │  - RAM prefetch │                          │
│  │  - Upload hot   │          │  - VRAM LRU     │                          │
│  │    to VRAM      │          │  - Background   │                          │
│  │  - No swapping  │          │    streaming    │                          │
│  └────────┬────────┘          └────────┬────────┘                          │
│           │                             │                                   │
│           └──────────────┬──────────────┘                                  │
│                          ▼                                                  │
│           ┌──────────────────────────────┐                                 │
│           │    TieredWeightStore         │                                 │
│           │                              │                                 │
│           │  get_layer(idx) -> &Layer    │                                 │
│           │  prefetch_layers([idx])      │                                 │
│           │  evict_for_kv_cache(bytes)   │                                 │
│           └──────────────────────────────┘                                 │
│                          │                                                  │
│                          ▼                                                  │
│           ┌──────────────────────────────┐                                 │
│           │    ComputeEngine             │                                 │
│           │    (forward pass)            │                                 │
│           └──────────────────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Tiers

| Tier | Latency | Bandwidth | Typical Size | Contents |
|------|---------|-----------|--------------|----------|
| **VRAM** | ~0.1ms | ~900 GB/s | 24 GB | Hot tensors, working buffers, KV cache |
| **RAM** | ~1ms | ~50 GB/s | 64-128 GB | Warm tensors in pinned memory |
| **NVMe** | ~10ms | ~7 GB/s | 1+ TB | Cold tensors, HCT cache |

### 2.3 Tensor Flow Between Tiers

```
                    ┌─────────┐
                    │  NVMe   │  Cold storage (HCT or safetensor cache)
                    │  (cold) │
                    └────┬────┘
                         │
            prefetch()   │   evict() [optional writeback]
                         ▼
                    ┌─────────┐
                    │   RAM   │  Pinned memory for fast GPU transfer
                    │  (warm) │
                    └────┬────┘
                         │
            upload()     │   evict() [demote to RAM]
                         ▼
                    ┌─────────┐
                    │  VRAM   │  GPU memory for inference
                    │  (hot)  │
                    └─────────┘
```

**Invariants:**
- Tensors can only move one tier at a time (no NVMe→VRAM direct)
- VRAM eviction demotes to RAM (not NVMe) for fast re-upload
- RAM eviction may writeback to NVMe if modified (quantized)
- Prefetch is always async and non-blocking

---

## 3. Data Structures

### 3.1 TieredWeightStore

```rust
/// GPU-resident model weights with 3-tier memory management.
///
/// Integrates with `AllocationPlan` to place tensors optimally across
/// VRAM, RAM, and NVMe based on importance and access patterns.
pub struct TieredWeightStore {
    /// Model configuration.
    config: ModelConfig,

    /// CUDA device for GPU operations.
    device: Arc<CudaDevice>,

    /// Allocation plan from adaptive_tiering.
    plan: AllocationPlan,

    /// VRAM-resident tensors (hot).
    vram_cache: VramCache,

    /// Pinned RAM tensors (warm).
    ram_cache: RamCache,

    /// NVMe tensor cache (cold).
    nvme_cache: NvmeCache,

    /// Background prefetch thread handle.
    prefetch_handle: Option<JoinHandle<()>>,

    /// Channel for prefetch requests.
    prefetch_tx: Sender<PrefetchRequest>,

    /// Statistics for monitoring.
    stats: TieredStats,
}
```

### 3.2 VramCache

```rust
/// VRAM cache for hot tensors.
///
/// Uses LRU eviction when VRAM pressure increases (e.g., KV cache growth).
/// Evicted tensors are demoted to RAM, not discarded.
pub struct VramCache {
    /// Layer weights currently in VRAM.
    /// Key: layer_idx, Value: LayerWeights on GPU
    layers: HashMap<usize, LayerWeights>,

    /// Shared weights always in VRAM.
    shared: SharedWeights,

    /// LRU tracking for eviction.
    lru: LruTracker<usize>,

    /// Current VRAM usage in bytes.
    usage: AtomicU64,

    /// VRAM budget from allocation plan.
    budget: u64,
}

/// Shared weights that are always VRAM-resident.
pub struct SharedWeights {
    /// Token embeddings [vocab_size, hidden_size].
    pub embed_tokens: GpuTensor,

    /// Final layer norm.
    pub final_norm: RMSNormWeights,

    /// LM head projection (or None if tied to embeddings).
    pub lm_head: Option<GpuTensor>,
}
```

### 3.3 RamCache

```rust
/// Pinned RAM cache for warm tensors.
///
/// Uses CUDA pinned memory for fast GPU uploads (~12 GB/s PCIe 4.0).
/// Tensors here can be uploaded to VRAM in ~100ms per layer.
pub struct RamCache {
    /// Layer weights in pinned RAM.
    /// Key: layer_idx, Value: LayerWeights on CPU (pinned)
    layers: HashMap<usize, CpuLayerWeights>,

    /// LRU tracking for eviction to NVMe.
    lru: LruTracker<usize>,

    /// Current RAM usage in bytes.
    usage: AtomicU64,

    /// RAM budget from allocation plan.
    budget: u64,

    /// Whether to use pinned memory (requires CUDA context).
    use_pinned: bool,
}

/// CPU-side layer weights in pinned memory.
pub struct CpuLayerWeights {
    /// Weight data as contiguous f16 buffer.
    data: PinnedBuffer,

    /// Offsets for each weight tensor within data.
    offsets: LayerOffsets,

    /// Total size in bytes.
    size_bytes: usize,

    /// Precision (may be quantized).
    precision: TensorPrecision,
}
```

### 3.4 NvmeCache

```rust
/// NVMe cache for cold tensors.
///
/// Stores decompressed tensors on disk for fast reload.
/// Can also load directly from HCT files if cache miss.
pub struct NvmeCache {
    /// Cache directory path.
    cache_dir: PathBuf,

    /// Original HCT directory path.
    hct_dir: PathBuf,

    /// Index of cached tensors.
    /// Key: tensor_name, Value: CacheEntry
    index: RwLock<HashMap<String, NvmeCacheEntry>>,

    /// Current cache size in bytes.
    usage: AtomicU64,

    /// Maximum cache size (0 = unlimited).
    max_size: u64,
}

/// Entry in the NVMe cache.
pub struct NvmeCacheEntry {
    /// Path to cached file (safetensor format).
    path: PathBuf,

    /// Size in bytes.
    size: u64,

    /// Whether this is a cached decompression or original HCT.
    is_cached: bool,

    /// Last access time for LRU.
    last_access: SystemTime,
}
```

### 3.5 Prefetch System

```rust
/// Request for background prefetch.
pub enum PrefetchRequest {
    /// Prefetch layer from NVMe to RAM.
    NvmeToRam { layer_idx: usize },

    /// Prefetch layer from RAM to VRAM.
    RamToVram { layer_idx: usize },

    /// Prefetch multiple layers (batch).
    Batch { layers: Vec<usize>, target: MemoryTier },

    /// Shutdown the prefetch thread.
    Shutdown,
}

/// Prefetch thread state.
struct PrefetchWorker {
    /// Receiver for prefetch requests.
    rx: Receiver<PrefetchRequest>,

    /// Reference to caches for loading.
    nvme_cache: Arc<NvmeCache>,
    ram_cache: Arc<RwLock<RamCache>>,
    vram_cache: Arc<RwLock<VramCache>>,

    /// CUDA stream for async uploads.
    stream: CudaStream,
}
```

---

## 4. Loading Strategies

### 4.1 Strategy Selection

```rust
/// Select loading strategy based on allocation plan.
pub fn select_strategy(plan: &AllocationPlan, config: &HardwareConfig) -> LoadingStrategy {
    // Calculate if model fits in VRAM + RAM
    let fits_in_memory = plan.nvme_usage == 0;

    // Calculate if model fits with aggressive quantization
    let fits_with_quant = plan.total_usage() <= config.vram_budget + config.ram_budget;

    if fits_in_memory {
        // Best case: everything in fast memory
        LoadingStrategy::Eager {
            parallel_decompress: true,
            vram_preload: true,
        }
    } else if fits_with_quant {
        // Model fits if we quantize aggressively
        LoadingStrategy::EagerQuantized {
            vram_precision: TensorPrecision::INT4,
            ram_precision: TensorPrecision::BF16,
        }
    } else {
        // Model needs NVMe tier
        LoadingStrategy::Progressive {
            prefetch_depth: 2,
            nvme_cache_size: config.nvme_cache_size,
        }
    }
}

pub enum LoadingStrategy {
    /// Decompress all tensors upfront, no NVMe needed.
    Eager {
        parallel_decompress: bool,
        vram_preload: bool,
    },

    /// Decompress all with aggressive quantization.
    EagerQuantized {
        vram_precision: TensorPrecision,
        ram_precision: TensorPrecision,
    },

    /// Stream from NVMe with prefetching.
    Progressive {
        prefetch_depth: usize,
        nvme_cache_size: u64,
    },
}
```

### 4.2 Eager Loading (Fast Path)

When model fits in VRAM + RAM:

```rust
impl TieredWeightStore {
    /// Eager load all tensors to VRAM and RAM.
    ///
    /// This is the fast path when model fits in memory.
    /// All HCT decompression happens upfront, then inference is swap-free.
    pub fn load_eager(
        hct_dir: &Path,
        plan: &AllocationPlan,
        device: Arc<CudaDevice>,
    ) -> Result<Self, TieredError> {
        // Phase 1: Parallel HCT decompression to RAM
        let cpu_tensors = decompress_all_parallel(hct_dir, &plan)?;

        // Phase 2: Upload VRAM-allocated tensors to GPU
        let mut vram_cache = VramCache::new(plan.vram_budget);
        let mut ram_cache = RamCache::new(plan.ram_budget, true /* pinned */);

        for (name, tensor) in cpu_tensors {
            let alloc = plan.allocations.get(&name).unwrap();

            match alloc.tier {
                MemoryTier::Vram => {
                    let gpu_tensor = upload_to_gpu(&tensor, &device)?;
                    vram_cache.insert(name, gpu_tensor);
                }
                MemoryTier::Ram => {
                    let pinned = pin_memory(&tensor)?;
                    ram_cache.insert(name, pinned);
                }
                MemoryTier::Nvme => {
                    // Should not happen in eager mode
                    unreachable!("Eager loading should not have NVMe allocations");
                }
            }
        }

        // Phase 3: Build layer index
        let layers = build_layer_index(&vram_cache, &ram_cache, &plan)?;

        Ok(Self {
            vram_cache,
            ram_cache,
            nvme_cache: NvmeCache::disabled(),
            // ... other fields
        })
    }
}

/// Parallel HCT decompression using rayon.
fn decompress_all_parallel(
    hct_dir: &Path,
    plan: &AllocationPlan,
) -> Result<HashMap<String, CpuTensor>, TieredError> {
    use rayon::prelude::*;

    let hct_files: Vec<_> = std::fs::read_dir(hct_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension() == Some("hct".as_ref()))
        .collect();

    hct_files
        .par_iter()
        .map(|entry| {
            let path = entry.path();
            let name = tensor_name_from_path(&path);
            let alloc = plan.allocations.get(&name);

            // Use reconstruction path from allocation
            let tensor = match alloc.map(|a| a.reconstruction_path) {
                Some(ReconstructionPath::GpuFast) => decompress_gpu(&path)?,
                Some(ReconstructionPath::CpuDirect) | None => decompress_cpu(&path)?,
                Some(ReconstructionPath::DirectLoad) => load_safetensor(&path)?,
            };

            Ok((name, tensor))
        })
        .collect()
}
```

### 4.3 Progressive Loading (NVMe Path)

When model needs NVMe tier:

```rust
impl TieredWeightStore {
    /// Progressive load with NVMe cold storage.
    ///
    /// Tensors are loaded on-demand with background prefetching.
    pub fn load_progressive(
        hct_dir: &Path,
        plan: &AllocationPlan,
        device: Arc<CudaDevice>,
        config: ProgressiveConfig,
    ) -> Result<Self, TieredError> {
        // Phase 1: Initialize caches
        let vram_cache = VramCache::new(plan.vram_budget);
        let ram_cache = RamCache::new(plan.ram_budget, true);
        let nvme_cache = NvmeCache::new(&config.cache_dir, hct_dir, config.max_cache_size)?;

        // Phase 2: Load shared weights (always in VRAM)
        let shared = load_shared_weights(hct_dir, &device)?;
        vram_cache.set_shared(shared);

        // Phase 3: Preload VRAM-allocated layers
        for (name, alloc) in &plan.allocations {
            if alloc.tier == MemoryTier::Vram && is_layer_weight(name) {
                let layer_idx = parse_layer_index(name)?;
                load_layer_to_vram(&nvme_cache, &mut vram_cache, layer_idx, &device)?;
            }
        }

        // Phase 4: Preload RAM-allocated layers
        for (name, alloc) in &plan.allocations {
            if alloc.tier == MemoryTier::Ram && is_layer_weight(name) {
                let layer_idx = parse_layer_index(name)?;
                load_layer_to_ram(&nvme_cache, &mut ram_cache, layer_idx)?;
            }
        }

        // Phase 5: Start prefetch thread
        let (tx, rx) = crossbeam_channel::unbounded();
        let prefetch_handle = spawn_prefetch_worker(rx, &nvme_cache, &ram_cache, &vram_cache);

        Ok(Self {
            vram_cache,
            ram_cache,
            nvme_cache,
            prefetch_handle: Some(prefetch_handle),
            prefetch_tx: tx,
            // ... other fields
        })
    }
}
```

---

## 5. Runtime Behavior

### 5.1 Layer Access

```rust
impl TieredWeightStore {
    /// Get layer weights for forward pass.
    ///
    /// Returns weights from fastest available tier:
    /// 1. VRAM (instant)
    /// 2. RAM (upload required, ~100ms)
    /// 3. NVMe (decompress + upload, ~500ms)
    ///
    /// Also triggers prefetch for upcoming layers.
    pub fn get_layer(&mut self, layer_idx: usize) -> Result<&LayerWeights, TieredError> {
        // Prefetch next layers in background
        self.prefetch_ahead(layer_idx);

        // Check VRAM first (hot path)
        if let Some(layer) = self.vram_cache.get(layer_idx) {
            self.stats.vram_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(layer);
        }

        // Check RAM (warm path)
        if let Some(cpu_layer) = self.ram_cache.get(layer_idx) {
            self.stats.ram_hits.fetch_add(1, Ordering::Relaxed);

            // Upload to VRAM (may trigger eviction)
            let gpu_layer = self.upload_layer(layer_idx, cpu_layer)?;
            return Ok(gpu_layer);
        }

        // Load from NVMe (cold path)
        self.stats.nvme_hits.fetch_add(1, Ordering::Relaxed);
        let layer = self.load_from_nvme(layer_idx)?;
        Ok(layer)
    }

    /// Prefetch upcoming layers based on forward pass position.
    fn prefetch_ahead(&self, current_layer: usize) {
        let prefetch_depth = self.config.prefetch_depth;

        for offset in 1..=prefetch_depth {
            let target = current_layer + offset;
            if target >= self.config.num_layers {
                break;
            }

            // Determine target tier based on allocation
            let alloc = self.plan.get_layer_allocation(target);

            if !self.vram_cache.contains(target) && alloc.tier == MemoryTier::Vram {
                // Prefetch to VRAM
                if self.ram_cache.contains(target) {
                    let _ = self.prefetch_tx.try_send(PrefetchRequest::RamToVram {
                        layer_idx: target
                    });
                } else {
                    let _ = self.prefetch_tx.try_send(PrefetchRequest::NvmeToRam {
                        layer_idx: target
                    });
                }
            } else if !self.ram_cache.contains(target) && alloc.tier <= MemoryTier::Ram {
                // Prefetch to RAM
                let _ = self.prefetch_tx.try_send(PrefetchRequest::NvmeToRam {
                    layer_idx: target
                });
            }
        }
    }
}
```

### 5.2 VRAM Eviction

```rust
impl VramCache {
    /// Evict layers to free VRAM for KV cache growth.
    ///
    /// Evicts lowest-priority layers first, demoting them to RAM.
    /// Returns the amount of VRAM freed.
    pub fn evict_for_kv_cache(
        &mut self,
        bytes_needed: u64,
        ram_cache: &mut RamCache,
        plan: &AllocationPlan,
    ) -> Result<u64, TieredError> {
        let mut freed = 0u64;

        // Get eviction candidates sorted by priority (lowest first)
        let mut candidates: Vec<_> = self.layers
            .keys()
            .map(|&idx| {
                let priority = plan.get_layer_priority(idx);
                let last_access = self.lru.last_access(idx);
                (idx, priority, last_access)
            })
            .collect();

        // Sort by priority, then by last access time
        candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        for (layer_idx, _priority, _) in candidates {
            if freed >= bytes_needed {
                break;
            }

            // Skip shared weights (never evict)
            if self.is_shared_weight(layer_idx) {
                continue;
            }

            // Evict to RAM
            if let Some(gpu_layer) = self.layers.remove(&layer_idx) {
                let size = gpu_layer.size_bytes();

                // Download to pinned RAM
                let cpu_layer = gpu_layer.to_cpu_pinned()?;
                ram_cache.insert(layer_idx, cpu_layer);

                freed += size as u64;
                self.usage.fetch_sub(size as u64, Ordering::Relaxed);
                self.lru.remove(layer_idx);

                tracing::debug!(
                    layer_idx,
                    freed_mb = size / (1024 * 1024),
                    "Evicted layer from VRAM to RAM"
                );
            }
        }

        Ok(freed)
    }
}
```

### 5.3 RAM Eviction

```rust
impl RamCache {
    /// Evict layers from RAM when capacity exceeded.
    ///
    /// Evicted layers are either:
    /// - Discarded (if clean - can reload from NVMe)
    /// - Written back to NVMe cache (if modified/quantized)
    pub fn evict_lru(
        &mut self,
        bytes_needed: u64,
        nvme_cache: &NvmeCache,
    ) -> Result<u64, TieredError> {
        let mut freed = 0u64;

        while freed < bytes_needed {
            let layer_idx = match self.lru.pop_lru() {
                Some(idx) => idx,
                None => break, // No more layers to evict
            };

            if let Some(cpu_layer) = self.layers.remove(&layer_idx) {
                let size = cpu_layer.size_bytes as u64;

                // If layer was quantized in RAM, write back to NVMe cache
                if cpu_layer.precision != TensorPrecision::BF16 {
                    nvme_cache.write_layer(layer_idx, &cpu_layer)?;
                }
                // Otherwise, discard (can reload from original HCT)

                freed += size;
                self.usage.fetch_sub(size, Ordering::Relaxed);

                tracing::debug!(
                    layer_idx,
                    freed_mb = size / (1024 * 1024),
                    "Evicted layer from RAM"
                );
            }
        }

        Ok(freed)
    }
}
```

---

## 6. NVMe Cache

### 6.1 Cache Structure

```
~/.cache/infernum/models/<model_hash>/
├── manifest.json           # Cache metadata
├── layer_000_bf16.safetensor
├── layer_001_bf16.safetensor
├── layer_002_int8.safetensor   # Quantized layer
├── ...
└── layer_079_bf16.safetensor
```

### 6.2 Cache Operations

```rust
impl NvmeCache {
    /// Load a layer from NVMe cache or original HCT.
    pub fn load_layer(&self, layer_idx: usize) -> Result<CpuLayerWeights, TieredError> {
        // Check cache first
        if let Some(entry) = self.get_cache_entry(layer_idx) {
            return self.load_from_cache(&entry);
        }

        // Cache miss - decompress from HCT
        let layer = self.decompress_from_hct(layer_idx)?;

        // Optionally cache for future use
        if self.should_cache(layer_idx) {
            self.write_to_cache(layer_idx, &layer)?;
        }

        Ok(layer)
    }

    /// Write a layer to the NVMe cache.
    pub fn write_layer(
        &self,
        layer_idx: usize,
        layer: &CpuLayerWeights,
    ) -> Result<(), TieredError> {
        let path = self.cache_path(layer_idx, layer.precision);

        // Write as safetensor for fast reload
        write_safetensor(&path, layer)?;

        // Update index
        let mut index = self.index.write().unwrap();
        index.insert(layer_idx, NvmeCacheEntry {
            path,
            size: layer.size_bytes as u64,
            is_cached: true,
            last_access: SystemTime::now(),
        });

        self.usage.fetch_add(layer.size_bytes as u64, Ordering::Relaxed);

        // Evict old entries if over limit
        self.maybe_evict_old_entries(&mut index)?;

        Ok(())
    }

    /// Evict old cache entries to stay within size limit.
    fn maybe_evict_old_entries(
        &self,
        index: &mut HashMap<usize, NvmeCacheEntry>,
    ) -> Result<(), TieredError> {
        if self.max_size == 0 {
            return Ok(()); // Unlimited
        }

        while self.usage.load(Ordering::Relaxed) > self.max_size {
            // Find oldest entry
            let oldest = index
                .iter()
                .filter(|(_, e)| e.is_cached) // Only evict cached, not original HCT
                .min_by_key(|(_, e)| e.last_access)
                .map(|(idx, _)| *idx);

            if let Some(idx) = oldest {
                if let Some(entry) = index.remove(&idx) {
                    std::fs::remove_file(&entry.path)?;
                    self.usage.fetch_sub(entry.size, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        Ok(())
    }
}
```

---

## 7. Integration with ComputeEngine

### 7.1 Modified Forward Pass

```rust
impl ComputeEngine {
    /// Forward pass with tiered weight access.
    pub fn forward_tiered(
        &mut self,
        input_ids: &GpuTensor,
        weights: &mut TieredWeightStore,
        kv_cache: &mut KvCache,
    ) -> Result<GpuTensor, InferenceError> {
        let seq_len = input_ids.shape()[0];

        // Embedding lookup (always in VRAM)
        let mut hidden = self.embedding_lookup(input_ids, weights.embed_tokens())?;

        // Process each layer
        for layer_idx in 0..weights.num_layers() {
            // Get layer weights (may trigger load/prefetch)
            let layer = weights.get_layer(layer_idx)?;

            // Check VRAM pressure from KV cache
            let kv_growth = self.estimate_kv_growth(seq_len, layer_idx);
            if kv_growth > weights.vram_headroom() {
                weights.evict_for_kv_cache(kv_growth)?;
            }

            // Transformer block
            hidden = self.transformer_block(&hidden, layer, kv_cache, layer_idx)?;
        }

        // LM head projection (always in VRAM)
        let logits = self.lm_head_projection(&hidden, weights)?;

        Ok(logits)
    }
}
```

### 7.2 Generator Integration

```rust
/// Generator using tiered weight store.
pub struct TieredGenerator {
    engine: ComputeEngine,
    weights: TieredWeightStore,
    kv_cache: KvCache,
    tokenizer: Tokenizer,
}

impl TieredGenerator {
    /// Create generator with tiered loading.
    pub fn new(
        model_dir: impl AsRef<Path>,
        config: TieredConfig,
    ) -> Result<Self, GeneratorError> {
        let model_dir = model_dir.as_ref();

        // Build model profile
        let profile = ModelProfile::from_hct_directory(model_dir)?;

        // Create allocation plan
        let planner = AllocationPlanner::new(config.tiering_config.clone());
        let plan = planner.plan(&profile)?;

        tracing::info!(
            vram_gb = plan.vram_usage as f64 / 1e9,
            ram_gb = plan.ram_usage as f64 / 1e9,
            nvme_gb = plan.nvme_usage as f64 / 1e9,
            swap_count = plan.swap_count,
            "Allocation plan computed"
        );

        // Select loading strategy
        let strategy = select_strategy(&plan, &config.hardware);

        // Load weights
        let device = CudaDevice::new(config.device_id)?;
        let weights = match strategy {
            LoadingStrategy::Eager { .. } => {
                tracing::info!("Using eager loading (model fits in VRAM+RAM)");
                TieredWeightStore::load_eager(model_dir, &plan, device)?
            }
            LoadingStrategy::Progressive { .. } => {
                tracing::info!("Using progressive loading with NVMe tier");
                TieredWeightStore::load_progressive(model_dir, &plan, device, config.progressive)?
            }
            LoadingStrategy::EagerQuantized { .. } => {
                tracing::info!("Using eager loading with quantization");
                TieredWeightStore::load_eager_quantized(model_dir, &plan, device, config.quant)?
            }
        };

        // Create compute engine and KV cache
        let model_config = weights.model_config().clone();
        let engine = ComputeEngine::new(model_config.clone(), device)?;
        let kv_cache = KvCache::new(&model_config, config.max_seq_len, device)?;

        Ok(Self { engine, weights, kv_cache, tokenizer: config.tokenizer })
    }

    /// Generate tokens.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, GeneratorError> {
        // ... generation loop using forward_tiered
    }
}
```

---

## 8. Configuration

```rust
/// Configuration for tiered weight loading.
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Adaptive tiering configuration.
    pub tiering_config: AdaptiveTieringConfig,

    /// Hardware configuration.
    pub hardware: HardwareConfig,

    /// Progressive loading configuration.
    pub progressive: ProgressiveConfig,

    /// Quantization configuration.
    pub quant: QuantConfig,

    /// CUDA device ID.
    pub device_id: usize,

    /// Maximum sequence length for KV cache.
    pub max_seq_len: usize,

    /// Tokenizer.
    pub tokenizer: Tokenizer,
}

#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Available VRAM in bytes.
    pub vram_budget: u64,

    /// Available RAM in bytes.
    pub ram_budget: u64,

    /// NVMe cache size limit (0 = unlimited).
    pub nvme_cache_size: u64,

    /// Whether to use pinned memory.
    pub use_pinned_memory: bool,
}

#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Number of layers to prefetch ahead.
    pub prefetch_depth: usize,

    /// NVMe cache directory.
    pub cache_dir: PathBuf,

    /// Maximum NVMe cache size in bytes.
    pub max_cache_size: u64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            tiering_config: AdaptiveTieringConfig::default(),
            hardware: HardwareConfig {
                vram_budget: 22 * GB,
                ram_budget: 60 * GB,
                nvme_cache_size: 0, // Unlimited
                use_pinned_memory: true,
            },
            progressive: ProgressiveConfig {
                prefetch_depth: 2,
                cache_dir: dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("/tmp"))
                    .join("infernum/models"),
                max_cache_size: 0, // Unlimited
            },
            quant: QuantConfig::default(),
            device_id: 0,
            max_seq_len: 4096,
            tokenizer: Tokenizer::default(),
        }
    }
}
```

---

## 9. Error Handling

```rust
/// Errors from tiered weight loading.
#[derive(Debug, thiserror::Error)]
pub enum TieredError {
    /// VRAM allocation failed.
    #[error("VRAM allocation failed: {0}")]
    VramAllocation(String),

    /// RAM allocation failed.
    #[error("RAM allocation failed: {0}")]
    RamAllocation(String),

    /// NVMe cache error.
    #[error("NVMe cache error: {0}")]
    NvmeCache(String),

    /// HCT decompression failed.
    #[error("HCT decompression failed for {tensor}: {error}")]
    Decompression { tensor: String, error: String },

    /// Layer not found.
    #[error("layer {0} not found")]
    LayerNotFound(usize),

    /// Tensor not found.
    #[error("tensor {0} not found")]
    TensorNotFound(String),

    /// CUDA error.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Prefetch thread panicked.
    #[error("prefetch thread panicked")]
    PrefetchPanic,
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    /// INVARIANT: VRAM usage never exceeds budget.
    #[proptest]
    fn prop_vram_never_exceeds_budget(
        model: ArbitraryModelProfile,
        config: ArbitraryConfig,
    ) {
        let store = TieredWeightStore::load_mock(&model, &config)?;

        // Access all layers in random order
        let mut rng = rand::thread_rng();
        let mut indices: Vec<_> = (0..model.num_layers).collect();
        indices.shuffle(&mut rng);

        for idx in indices {
            let _ = store.get_layer(idx)?;
            prop_assert!(store.vram_usage() <= config.hardware.vram_budget);
        }
    }

    /// INVARIANT: Evicted tensors are recoverable.
    #[test]
    fn test_evicted_tensors_recoverable() {
        let store = create_test_store();

        // Fill VRAM
        for i in 0..10 {
            store.get_layer(i)?;
        }

        // Force eviction
        store.evict_for_kv_cache(1 * GB)?;

        // Verify evicted layers can still be accessed
        for i in 0..10 {
            let layer = store.get_layer(i)?;
            assert!(layer.is_valid());
        }
    }

    /// Prefetch improves access latency.
    #[test]
    fn test_prefetch_reduces_latency() {
        let store = create_test_store_progressive();

        // Cold access (no prefetch)
        let start = Instant::now();
        store.get_layer(0)?;
        let cold_latency = start.elapsed();

        // Trigger prefetch for layer 5
        store.prefetch_layers(&[5]);
        std::thread::sleep(Duration::from_secs(1)); // Wait for prefetch

        // Warm access (after prefetch)
        let start = Instant::now();
        store.get_layer(5)?;
        let warm_latency = start.elapsed();

        assert!(warm_latency < cold_latency / 2, "prefetch should reduce latency");
    }
}
```

### 10.2 Integration Tests

```rust
/// Full pipeline test with real HCT model.
#[test]
#[ignore] // Requires model files
fn test_tiered_generation_14b() {
    let config = TieredConfig {
        hardware: HardwareConfig {
            vram_budget: 22 * GB,
            ram_budget: 60 * GB,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut generator = TieredGenerator::new(
        "/path/to/qwen2.5-14b-hct",
        config,
    )?;

    let output = generator.generate("Hello, world!", 100)?;
    assert!(!output.is_empty());

    // Verify performance
    let stats = generator.stats();
    assert!(stats.tokens_per_second > 1.0, "should achieve >1 tk/s for 14B");
}

/// Test progressive loading with NVMe tier.
#[test]
#[ignore] // Requires large model
fn test_tiered_generation_70b() {
    let config = TieredConfig {
        hardware: HardwareConfig {
            vram_budget: 22 * GB,
            ram_budget: 60 * GB,
            nvme_cache_size: 100 * GB,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut generator = TieredGenerator::new(
        "/path/to/llama-70b-hct",
        config,
    )?;

    let output = generator.generate("Hello, world!", 10)?;
    assert!(!output.is_empty());

    // Verify NVMe tier is used
    let stats = generator.stats();
    assert!(stats.nvme_hits > 0, "should use NVMe tier for 70B");
}
```

---

## 11. Performance Targets

| Model | VRAM | RAM | NVMe | Strategy | Target Throughput |
|-------|------|-----|------|----------|-------------------|
| 14B | 24GB | 0 | 0 | Eager | 10+ tk/s |
| 14B | 24GB | 64GB | 0 | Eager+RAM | 5+ tk/s |
| 70B | 24GB | 64GB | 0 | Eager+Quant | 2+ tk/s |
| 70B | 24GB | 64GB | 100GB | Progressive | 0.5+ tk/s |
| 405B | 24GB | 128GB | 500GB | Progressive | 0.1+ tk/s |

---

## 12. Open Questions

1. **Pinned memory allocation strategy**: Should we pre-allocate a pinned memory pool or allocate on-demand?

2. **Quantization on upload vs on disk**: Should INT4/INT8 quantization happen during VRAM upload or be pre-computed and stored on NVMe?

3. **Prefetch thread count**: Single prefetch thread or thread pool? NVMe→RAM and RAM→VRAM can be parallelized.

4. **KV cache integration**: Should KV cache growth trigger synchronous eviction or async background eviction?

5. **Multi-GPU support**: How should allocation plans work across multiple GPUs?

---

## 13. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `VramCache` with LRU eviction
- [ ] `RamCache` with pinned memory
- [ ] `NvmeCache` with safetensor format
- [ ] `TieredWeightStore` basic structure

### Phase 2: Loading Strategies
- [ ] Eager loading path
- [ ] Progressive loading path
- [ ] Strategy selection logic

### Phase 3: Runtime Features
- [ ] Background prefetch thread
- [ ] KV cache pressure handling
- [ ] Dynamic reallocation

### Phase 4: Integration
- [ ] `ComputeEngine` integration
- [ ] `TieredGenerator` wrapper
- [ ] CLI/API support

### Phase 5: Optimization
- [ ] Parallel HCT decompression
- [ ] Async CUDA streams
- [ ] Quantization kernels

---

## 14. References

- ADAPTIVE-MEMORY-TIERING-SPEC.md v0.2.0
- CUDA-LAZY-HOLOTENSOR-LOADING.md
- [FlexGen: High-Throughput Generative Inference](https://arxiv.org/abs/2303.06865)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
