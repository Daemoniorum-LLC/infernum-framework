# CUDA Inference Lazy HoloTensor Loading

**Status:** 🟡 **IMPLEMENTATION COMPLETE - AWAITING TESTING**
**Author:** Claude (Opus 4.5)
**Date:** 2026-02-06
**Version:** 1.0.0

---

## 1. Gap Discovery

During HoloTensor integration testing, `cuda_inference::WeightStore::load_holotensor_weights()`
successfully loads all weights but exhausts VRAM before `Generator` can be created:

```
[INFO] CUDA weights loaded: 14525 MB total, 28 layers
[ERROR] Failed to create CUDA generator: CUDA_ERROR_OUT_OF_MEMORY
```

### 1.1 Root Cause

`cuda_inference::WeightStore` was designed for traditional eager loading:

```rust
// weight_store.rs:181
pub struct WeightStore {
    pub layers: Vec<LayerWeights>,  // ALL layers loaded upfront
    // ...
}
```

For a 70B model with 28 layers:
- Weight memory: 14.5 GB
- Generator buffers (KV cache, hidden states, etc.): ~8-10 GB
- Total required: ~24 GB
- Available on 24GB GPU: 24 GB
- **Result: OUT_OF_MEMORY**

### 1.2 The HoloTensor Promise

HoloTensor's entire purpose is enabling "too big" models on consumer GPUs:

1. **Progressive Quality**: Start at 70% quality, improve during idle time
2. **Lazy Layer Loading**: Load layers on-demand during forward pass
3. **LRU Eviction**: Move cold layers VRAM → RAM when memory pressure rises

The Candle path already implements this via `LazyLlama` + `TieredHoloLoader`:

```
Memory Layout During Inference:
┌─────────────────────────────────────────────────────┐
│ ALWAYS LOADED (embedding, norm, lm_head)   ~2GB    │
├─────────────────────────────────────────────────────┤
│ LAYER WINDOW (N layers in memory)          ~N×0.5GB│
│   Layer i-1  (recently used)                       │
│   Layer i    (currently processing)                │
│   Layer i+1  (prefetched)                          │
├─────────────────────────────────────────────────────┤
│ NOT LOADED (remaining layers)                      │
└─────────────────────────────────────────────────────┘
```

### 1.3 What cuda_inference Needs

Transform `WeightStore` from eager to lazy loading:

| Current | Required |
|---------|----------|
| `Vec<LayerWeights>` | `LazyLayerStore` with on-demand loading |
| Load all at init | Load layer before `layer_forward()` |
| No eviction | LRU eviction when VRAM budget exceeded |
| Single quality | Progressive quality (optional phase 2) |

---

## 2. Solution Specification

### 2.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        WeightStore                            │
├──────────────────────────────────────────────────────────────┤
│  embed_tokens: GpuTensor           (always loaded)           │
│  final_norm: RMSNormWeights        (always loaded)           │
│  lm_head: GpuTensor                (always loaded)           │
│                                                              │
│  layer_store: LazyLayerStore {                               │
│      loaded: HashMap<usize, LayerWeights>                    │
│      loader: Arc<dyn LayerLoader>                            │
│      lru: VecDeque<usize>                                    │
│      vram_budget: u64                                        │
│      current_vram: AtomicU64                                 │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     LayerLoader Trait                         │
├──────────────────────────────────────────────────────────────┤
│  fn load_layer(&self, idx: usize) -> Result<LayerWeights>    │
│  fn layer_size(&self, idx: usize) -> u64                     │
│  fn num_layers(&self) -> usize                               │
└──────────────────────────────────────────────────────────────┘
            ▲                               ▲
            │                               │
   ┌────────┴────────┐           ┌─────────┴─────────┐
   │ HoloLayerLoader │           │ StandardHctLoader │
   └─────────────────┘           └───────────────────┘
```

### 2.2 LazyLayerStore Interface

```rust
pub struct LazyLayerStore {
    /// Currently loaded layers.
    loaded: HashMap<usize, LayerWeights>,

    /// Loader for on-demand layer loading.
    loader: Arc<dyn LayerLoader + Send + Sync>,

    /// LRU order (front = oldest, back = newest).
    lru: VecDeque<usize>,

    /// VRAM budget for layers (excludes embed/norm/lm_head).
    vram_budget: u64,

    /// Current VRAM usage by loaded layers.
    current_vram: u64,

    /// CUDA device for loading.
    device: Arc<CudaDevice>,
}

impl LazyLayerStore {
    /// Get layer, loading if necessary.
    /// Evicts LRU layers if VRAM budget exceeded.
    pub fn get_layer(&mut self, idx: usize) -> Result<&LayerWeights, InferenceError>;

    /// Prefetch layers (for async loading).
    pub fn prefetch(&mut self, indices: &[usize]) -> Result<(), InferenceError>;

    /// Evict a specific layer to free VRAM.
    pub fn evict(&mut self, idx: usize);

    /// Current memory usage statistics.
    pub fn stats(&self) -> LayerStoreStats;
}
```

### 2.3 LayerLoader Trait

```rust
pub trait LayerLoader: Send + Sync {
    /// Load a single layer's weights to GPU.
    fn load_layer(
        &self,
        idx: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<LayerWeights, InferenceError>;

    /// Estimate VRAM size for a layer.
    fn layer_vram_size(&self, idx: usize) -> u64;

    /// Total number of layers.
    fn num_layers(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;
}
```

### 2.4 HoloTensor Layer Loader

```rust
pub struct HoloLayerLoader {
    /// HoloTensor files indexed by tensor name.
    tensor_files: HashMap<String, PathBuf>,

    /// Model configuration.
    config: ModelConfig,

    /// GPU holographic reconstruction context.
    gpu_ctx: Mutex<GpuHoloContext>,

    /// Weight name mapping.
    weight_map: WeightNameMap,
}

impl LayerLoader for HoloLayerLoader {
    fn load_layer(
        &self,
        idx: usize,
        device: &Arc<CudaDevice>,
    ) -> Result<LayerWeights, InferenceError> {
        // 1. Find all tensor files for this layer
        // 2. Use GpuHoloContext for GPU reconstruction
        // 3. Build LayerWeights struct
    }
}
```

### 2.5 ComputeEngine Integration

Change from iterating `&weights.layers` to requesting layers on-demand:

```rust
// Current (eager):
for layer in &weights.layers {
    self.layer_forward(&mut hidden, layer, seq_len, start_pos)?;
}

// Required (lazy):
for layer_idx in 0..weights.num_layers() {
    let layer = weights.get_layer(layer_idx)?;  // Load on-demand
    self.layer_forward(&mut hidden, layer, seq_len, start_pos)?;
}
```

---

## 3. Implementation Plan

### Phase 1: Core Lazy Loading (MVP)

1. **Create `LazyLayerStore`** with HashMap storage and LRU eviction
2. **Implement `LayerLoader` trait** and `HoloLayerLoader`
3. **Modify `WeightStore`** to use `LazyLayerStore` instead of `Vec<LayerWeights>`
4. **Update `ComputeEngine`** to request layers via `get_layer()`
5. **Add VRAM budget configuration** with sensible defaults

### Phase 2: Performance Optimization

1. **Async prefetching**: Load layer i+1 while processing layer i
2. **CPU staging**: Keep evicted layers in RAM for faster reload
3. **Stream overlap**: Use CUDA streams for concurrent load/compute

### Phase 3: Progressive Quality (Future)

1. **Quality levels**: Load at 70% initially, improve during idle
2. **Background refinement**: Stream additional fragments between requests
3. **Adaptive quality**: Reduce quality under memory pressure

---

## 4. Memory Budget Calculation

For 24GB VRAM with a 70B model:

| Component | Size | Notes |
|-----------|------|-------|
| Embeddings | ~500 MB | Always loaded |
| LM Head | ~500 MB | Always loaded |
| Final Norm | ~16 MB | Always loaded |
| Generator Buffers | ~6 GB | KV cache, hidden states, scratch |
| **Available for Layers** | **~17 GB** | |
| Per-layer (70B) | ~520 MB | 28 layers total |
| **Max Layers Loaded** | **~32** | Can hold ALL layers! |

For a 405B model (126 layers):

| Component | Size | Notes |
|-----------|------|-------|
| Fixed overhead | ~8 GB | Embeddings, norm, buffers |
| **Available for Layers** | **~16 GB** | |
| Per-layer (405B) | ~1.2 GB | 126 layers total |
| **Max Layers Loaded** | **~13** | ~10% of model |

---

## 5. Acceptance Criteria

- [x] `LazyLayerStore` struct implemented with LRU eviction
- [x] `LayerLoader` trait and `HoloLayerLoader` implemented
- [x] `LazyWeightStore` provides lazy loading interface
- [x] `ComputeEngine` has `*_lazy()` methods for on-demand layer access
- [x] `LazyGenerator` provides full lazy inference pipeline
- [x] VRAM budget configurable (default: 80% of available)
- [ ] 70B HoloTensor model runs inference on 24GB GPU
- [x] No regression for standard HCT format (eager `WeightStore` unchanged)

---

## 6. Test Plan

### 6.1 Unit Tests

```rust
#[test]
fn test_lazy_layer_store_loads_on_demand() {
    // Layer not loaded until get_layer() called
}

#[test]
fn test_lazy_layer_store_lru_eviction() {
    // When budget exceeded, oldest layer evicted
}

#[test]
fn test_holo_layer_loader_reconstructs_layer() {
    // Single layer loaded via GPU holographic reconstruction
}
```

### 6.2 Integration Tests

```rust
#[test]
#[cfg(feature = "cuda")]
fn test_holotensor_lazy_inference_24gb() {
    // Load 70B HoloTensor model, run inference
    // Verify VRAM stays under budget
}
```

---

## 7. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-06 | Initial gap documentation |
| 1.1.0 | 2026-02-06 | Implementation complete: LazyLayerStore, HoloLayerLoader, LazyWeightStore, lazy compute methods, LazyGenerator |
