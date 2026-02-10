# Adaptive Memory Tiering Specification

**Version:** 0.2.0
**Status:** Draft
**Date:** 2026-02-05
**Crate:** `abaddon`

> **v0.2.0 Update:** Added Loading Backend Selection (§2.4) and Tensor Classification
> (§3.4) based on performance validation showing 2.17 tk/s with eager loading vs
> 0.36 tk/s with progressive TieredHoloLoader wrapper.

---

## 1. Overview

This specification defines the Adaptive Memory Tiering system in `abaddon` — an intelligent
memory allocation strategy that maximizes inference quality within hardware constraints,
regardless of model size.

### 1.1 Problem Statement

The current HoloTensor loading system makes a **binary decision**:

```
Is HCT directory? → Use 405B layer-swapping mode
```

This results in:
- A 14B model (29GB BF16) on 24GB VRAM doing **33 layer swaps per token**
- Fixed `max_loaded_layers = vram_budget / layer_size` with no intelligence
- All tensors treated equally regardless of inference impact
- ~0.6 tk/s when the same model with smarter allocation could achieve ~10+ tk/s

### 1.2 Goal

Replace fixed layer-swapping with **adaptive allocation** that:

1. Analyzes the gap between model size and available memory
2. Scores tensor importance based on inference impact
3. Selects optimal precision per tensor (BF16, INT8, INT4)
4. Places tensors in optimal memory tier (VRAM, RAM, NVMe)
5. Minimizes or eliminates runtime swapping when possible

### 1.3 Design Principle

> **The system should be model-size agnostic.** A 135M model and a 405B model
> should use the same allocation algorithm — only the inputs differ.

The key insight: **Not all weights are equally important for inference quality.**

- Attention weights have higher impact than MLP weights
- First and last layers are more critical than middle layers
- HoloTensor QualityCurve encodes per-tensor importance via singular values
- Precision sensitivity varies by tensor type and layer position

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Adaptive Memory Tiering System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │  Model Profile  │     │  Hardware       │     │  Quality Target     │   │
│  │  - Tensor sizes │     │  - VRAM budget  │     │  - Min quality      │   │
│  │  - Layer count  │     │  - RAM budget   │     │  - Latency target   │   │
│  │  - Architecture │     │  - NVMe speed   │     │  - Swap tolerance   │   │
│  └────────┬────────┘     └────────┬────────┘     └──────────┬──────────┘   │
│           │                       │                          │              │
│           └───────────────────────┼──────────────────────────┘              │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                        │
│                    │     Allocation Planner       │                        │
│                    │                              │                        │
│                    │  1. Score tensor importance  │                        │
│                    │  2. Compute precision gains  │                        │
│                    │  3. Solve placement problem  │                        │
│                    │  4. Generate allocation map  │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                         │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                        │
│                    │     Allocation Map           │                        │
│                    │                              │                        │
│                    │  tensor_name → {             │                        │
│                    │    tier: VRAM|RAM|NVMe,      │                        │
│                    │    precision: BF16|INT8|INT4 │                        │
│                    │    priority: 0.0-1.0         │                        │
│                    │  }                           │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                         │
│           ┌───────────────────────┼───────────────────────┐                │
│           ▼                       ▼                       ▼                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐      │
│  │  VRAM Resident  │   │  RAM Cached     │   │  NVMe/Disk          │      │
│  │  (hot path)     │   │  (warm path)    │   │  (cold path)        │      │
│  │                 │   │                 │   │                     │      │
│  │  - Embeddings   │   │  - Evicted      │   │  - Original HCT     │      │
│  │  - Edge layers  │   │    layers       │   │  - Safetensor cache │      │
│  │  - Attention    │   │  - Prefetch     │   │                     │      │
│  │  - Critical MLP │   │    queue        │   │                     │      │
│  └─────────────────┘   └─────────────────┘   └─────────────────────┘      │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Runtime Adaptation                                                   │ │
│  │                                                                       │ │
│  │  - Monitor access patterns during inference                          │ │
│  │  - Adjust priorities based on actual usage                           │ │
│  │  - Prefetch next layers based on forward pass position               │ │
│  │  - Rebalance on KV cache growth                                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Tiers

| Tier | Latency | Capacity | Use Case |
|------|---------|----------|----------|
| **VRAM** | ~0.1ms | 24GB (typical) | Hot tensors accessed every forward pass |
| **RAM** | ~1ms | 64-128GB | Warm tensors, can be transferred quickly |
| **NVMe** | ~10ms | 1TB+ | Cold storage, safetensor cache |

### 2.4 Loading Backend Selection

**Critical Discovery:** The loading backend has a larger impact on throughput than
allocation strategy. Wrapping `TieredHoloLoader` (progressive streaming) adds
significant overhead even when tensors are "preloaded."

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Loading Backend Decision Tree                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model fits in VRAM+RAM with mixed precision?                               │
│           │                                                                 │
│           ├── YES ──► EAGER LOADING (hct_sequential)                        │
│           │           - Decompress all tensors upfront                      │
│           │           - Place in VRAM/RAM per allocation plan               │
│           │           - No background streaming overhead                    │
│           │           - Expected: 2+ tk/s for 14B                           │
│           │                                                                 │
│           └── NO ───► PROGRESSIVE LOADING (TieredHoloLoader)                │
│                       - Stream quality fragments on demand                  │
│                       - Layer swapping for 405B+ models                     │
│                       - Background quality improvement                      │
│                       - Expected: <1 tk/s (I/O bound)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Performance Validation (Qwen2.5-14B on 24GB VRAM):**

| Loading Mode | Throughput | Notes |
|--------------|------------|-------|
| `TieredHoloLoader` (405B mode) | 0.26 tk/s | Layer swapping overhead |
| `AdaptiveLoader` wrapping `TieredHoloLoader` | 0.36 tk/s | Preload helps, but streaming overhead remains |
| `INFERNUM_HCT_EAGER=1` (eager) | 2.17 tk/s | **8x faster** - no streaming overhead |

**Implication:** Adaptive tiering should use **eager loading as the backend** for
models that fit, not wrap the progressive loader. The allocation planner decides
*where* tensors go; the loader should use the fastest path to get them there.

```rust
// CORRECT architecture:
//
//   AllocationPlanner ──► AllocationPlan (VRAM/RAM/NVMe decisions)
//          │
//          ▼
//   EagerLoader (for models that fit)
//        OR
//   TieredHoloLoader (only for 405B+ that need true layer swapping)

fn select_loading_backend(plan: &AllocationPlan, model_size: u64) -> LoadingBackend {
    // If everything fits in VRAM + RAM, use eager loading
    if plan.nvme_usage == 0 && plan.swap_count == 0 {
        LoadingBackend::Eager
    }
    // If minimal swapping needed (< 10% of tensors), still use eager with RAM cache
    else if plan.swap_count < plan.allocations.len() / 10 {
        LoadingBackend::EagerWithRamCache
    }
    // Only use progressive for truly massive models
    else {
        LoadingBackend::Progressive
    }
}
```

### 2.5 Precision Levels

| Precision | Bits | Quality Impact | Size Reduction |
|-----------|------|----------------|----------------|
| **BF16** | 16 | Baseline | 1.0x |
| **FP8** | 8 | ~1% degradation | 2.0x |
| **INT8** | 8 | ~2% degradation | 2.0x |
| **INT4** | 4 | ~5% degradation | 4.0x |

---

## 3. Tensor Importance Scoring

### 3.1 Importance Factors

Each tensor receives an importance score `I ∈ [0, 1]` computed from:

```rust
struct ImportanceFactors {
    /// Position factor: edge layers (0, 1, n-2, n-1) are more critical
    /// Range: [0.5, 1.0] where edges = 1.0, middle = 0.5
    layer_position: f32,

    /// Type factor: attention > layernorm > mlp
    /// attention: 1.0, layernorm: 0.9, mlp.gate/up: 0.6, mlp.down: 0.7
    tensor_type: f32,

    /// Access frequency: tensors used every token vs occasionally
    /// embed/lm_head: 1.0, layer weights: 0.8
    access_frequency: f32,

    /// Quality sensitivity from HoloTensor QualityCurve (if available)
    /// Higher singular value variance = more quality sensitive
    quality_sensitivity: f32,

    /// Size factor: smaller tensors are cheaper to keep in VRAM
    /// Normalized inverse of tensor size
    size_efficiency: f32,
}

fn compute_importance(factors: ImportanceFactors) -> f32 {
    // Weighted combination
    0.25 * factors.layer_position
    + 0.30 * factors.tensor_type
    + 0.20 * factors.access_frequency
    + 0.15 * factors.quality_sensitivity
    + 0.10 * factors.size_efficiency
}
```

### 3.2 Default Importance Heuristics

When HoloTensor metadata is unavailable, use architecture-based heuristics:

```rust
fn default_importance(tensor_name: &str, layer_idx: usize, num_layers: usize) -> f32 {
    let mut score = 0.5;

    // Edge layer bonus
    if layer_idx <= 2 || layer_idx >= num_layers - 3 {
        score += 0.2;
    }

    // Tensor type scoring
    if tensor_name.contains("embed") || tensor_name.contains("lm_head") {
        score = 1.0;  // Always hot
    } else if tensor_name.contains("self_attn") {
        score += 0.2;  // Attention is critical
    } else if tensor_name.contains("layernorm") || tensor_name.contains("_norm") {
        score += 0.15;  // Normalization affects all activations
    } else if tensor_name.contains("mlp") {
        score += 0.0;  // MLP is less sensitive
    }

    score.clamp(0.0, 1.0)
}
```

### 3.3 QualityCurve Integration

When loading from HoloTensor format, extract importance from the quality curve:

```rust
fn importance_from_quality_curve(curve: &QualityCurve, total_fragments: u16) -> f32 {
    // Tensors where early fragments matter more have higher importance
    // (steeper quality curve = more sensitive to precision loss)

    let q_at_25_pct = curve.predict(total_fragments / 4, total_fragments);
    let q_at_50_pct = curve.predict(total_fragments / 2, total_fragments);

    // Steep early curve = high importance
    let steepness = q_at_25_pct / q_at_50_pct.max(0.01);

    // Normalize to [0, 1]
    (steepness - 0.5).clamp(0.0, 1.0)
}
```

### 3.4 Tensor Classification for Reconstruction Path

During the warmup/profiling phase, classify tensors into reconstruction paths to
avoid runtime overhead and spurious warnings.

**Problem:** The GPU HoloTensor reconstruction kernel (`gpu_holo.rs:1788`) only
supports 2D tensors. When 1D tensors (bias vectors, layernorm weights) are passed
to GPU reconstruction, they fail and fall back to CPU with a warning. This creates
log noise and unnecessary overhead.

**Solution:** Classify tensors by shape during profile building, then route them
directly to the appropriate reconstruction path.

```rust
/// Reconstruction path for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionPath {
    /// 2D+ tensors with sufficient size benefit from GPU reconstruction.
    /// Threshold: shape.len() >= 2 && total_elements > 4096
    GpuFast,

    /// 1D tensors (bias, layernorm) or small tensors go directly to CPU.
    /// No GPU attempt, no fallback warning.
    CpuDirect,

    /// Tensors stored as raw safetensors (no HoloTensor reconstruction needed).
    DirectLoad,
}

/// Determine reconstruction path based on tensor metadata.
fn classify_reconstruction_path(
    name: &str,
    shape: &[usize],
    format: TensorFormat,
) -> ReconstructionPath {
    // Raw safetensors bypass reconstruction entirely
    if format == TensorFormat::Safetensors {
        return ReconstructionPath::DirectLoad;
    }

    // 1D tensors always use CPU (bias, layernorm, embeddings sometimes)
    if shape.len() < 2 {
        return ReconstructionPath::CpuDirect;
    }

    // Small tensors don't benefit from GPU overhead
    let total_elements: usize = shape.iter().product();
    if total_elements < 4096 {
        return ReconstructionPath::CpuDirect;
    }

    // 2D+ tensors with sufficient size use GPU
    ReconstructionPath::GpuFast
}
```

**Integration with Model Profile:**

```rust
pub struct TensorInfo {
    pub name: String,
    pub size_bytes: u64,
    pub shape: Vec<usize>,
    pub layer_index: Option<usize>,
    pub tensor_type: TensorType,
    pub importance: f32,
    pub reconstruction_path: ReconstructionPath,  // NEW: classify at profile time
}

impl TensorInfo {
    pub fn from_hct_metadata(name: &str, metadata: &HctMetadata) -> Self {
        let shape = metadata.shape.clone();
        let reconstruction_path = classify_reconstruction_path(
            name,
            &shape,
            TensorFormat::Hct,
        );

        Self {
            name: name.to_string(),
            size_bytes: metadata.original_size,
            shape,
            layer_index: parse_layer_index(name),
            tensor_type: TensorType::from_name(name),
            importance: 0.0,  // Computed later
            reconstruction_path,
        }
    }
}
```

**Expected Classification Distribution (14B Model):**

| Path | Count | Tensor Types | Notes |
|------|-------|--------------|-------|
| `GpuFast` | ~450 | q/k/v/o_proj, gate/up/down_proj | Large 2D weight matrices |
| `CpuDirect` | ~130 | bias, layernorm, small embeddings | 1D vectors, tiny tensors |
| `DirectLoad` | 0 | (only if safetensor cache exists) | Bypass reconstruction |

This classification happens once during model profiling, not per-tensor-access,
eliminating the runtime overhead of failed GPU attempts.

---

## 4. Allocation Planning

### 4.1 Problem Formulation

Given:
- `T` = set of tensors with sizes `s[t]` and importance `I[t]`
- `V` = VRAM budget (bytes)
- `R` = RAM budget (bytes)
- `P` = set of precision options with size multipliers `m[p]` and quality costs `q[p]`

Find allocation `A: T → (Tier, Precision)` that:

```
Maximize: Σ I[t] * quality(A[t].precision)
Subject to:
  Σ s[t] * m[A[t].precision] for t where A[t].tier = VRAM  ≤  V
  Σ s[t] * m[A[t].precision] for t where A[t].tier = RAM   ≤  R
  All tensors allocated to exactly one tier
```

### 4.2 Greedy Allocation Algorithm

For practical implementation, use a greedy approach:

```rust
fn plan_allocation(
    tensors: &[TensorInfo],
    vram_budget: u64,
    ram_budget: u64,
    quality_target: f32,
) -> AllocationMap {
    let mut map = AllocationMap::new();
    let mut vram_used = 0u64;
    let mut ram_used = 0u64;

    // Sort by importance (descending)
    let mut sorted: Vec<_> = tensors.iter().collect();
    sorted.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

    for tensor in sorted {
        // Try precision levels in order of quality (BF16 > INT8 > INT4)
        for precision in [Precision::BF16, Precision::INT8, Precision::INT4] {
            let size = tensor.size_bytes / precision.size_divisor();

            // Try VRAM first
            if vram_used + size <= vram_budget {
                map.insert(tensor.name.clone(), Allocation {
                    tier: Tier::Vram,
                    precision,
                    priority: tensor.importance,
                });
                vram_used += size;
                break;
            }

            // Fall back to RAM (only BF16 in RAM - no point quantizing for RAM)
            if precision == Precision::BF16 && ram_used + tensor.size_bytes <= ram_budget {
                map.insert(tensor.name.clone(), Allocation {
                    tier: Tier::Ram,
                    precision: Precision::BF16,
                    priority: tensor.importance,
                });
                ram_used += tensor.size_bytes;
                break;
            }
        }

        // If still not allocated, goes to NVMe (disk)
        if !map.contains(&tensor.name) {
            map.insert(tensor.name.clone(), Allocation {
                tier: Tier::Nvme,
                precision: Precision::BF16,
                priority: tensor.importance,
            });
        }
    }

    map
}
```

### 4.3 Allocation Scenarios

**Scenario A: 14B model on 24GB VRAM**
```
Model: 29GB BF16
VRAM: 24GB
Gap: 5GB

Optimal allocation:
├── VRAM (BF16): embeddings, lm_head, layers 0-2, 45-47, all attention (20GB)
├── VRAM (INT8): middle layer MLPs (4GB → fits remaining 4GB)
└── RAM: nothing needed

Result: Zero swapping, ~10+ tk/s expected
```

**Scenario B: 70B model on 24GB VRAM**
```
Model: 140GB BF16
VRAM: 24GB
Gap: 116GB

Optimal allocation:
├── VRAM (BF16): embeddings, lm_head, layers 0-1, 78-79 (8GB)
├── VRAM (INT4): 16 most important layer attention weights (8GB)
├── VRAM (INT8): remaining VRAM for MLP fragments (8GB)
├── RAM: 40GB of warm layers (prefetch queue)
└── NVMe: remaining cold layers

Result: Minimal swapping for 40 hot layers, swap for 40 cold layers
```

**Scenario C: 405B model on 24GB VRAM + 80GB RAM**
```
Model: 800GB BF16
Gap: Doesn't fit anywhere fully

Optimal allocation:
├── VRAM (INT4): Most critical attention (24GB = 96GB equivalent)
├── RAM (INT8): Next tier of layers (80GB = 160GB equivalent)
└── NVMe: Cold layers with aggressive quality reduction

Result: Full layer swapping but with quality-aware prioritization
```

---

## 5. Runtime Adaptation

### 5.1 Access Pattern Monitoring

Track tensor access during inference:

```rust
struct AccessStats {
    /// Number of times tensor was accessed
    access_count: u64,
    /// Total time spent waiting for this tensor
    wait_time_ns: u64,
    /// Last access timestamp
    last_access: Instant,
    /// Whether tensor was in VRAM at last access
    was_hot: bool,
}
```

### 5.2 Dynamic Reallocation

Periodically adjust allocation based on observed patterns:

```rust
fn should_promote(tensor: &str, stats: &AccessStats, current_tier: Tier) -> bool {
    match current_tier {
        Tier::Ram => {
            // Promote to VRAM if frequently accessed and causing delays
            stats.access_count > 100 && stats.wait_time_ns > 10_000_000
        }
        Tier::Nvme => {
            // Promote to RAM if accessed more than once
            stats.access_count > 1
        }
        Tier::Vram => false,  // Already optimal
    }
}

fn should_demote(tensor: &str, stats: &AccessStats, current_tier: Tier) -> bool {
    match current_tier {
        Tier::Vram => {
            // Demote if not accessed recently and VRAM pressure is high
            stats.last_access.elapsed() > Duration::from_secs(10)
        }
        Tier::Ram => {
            // Demote to NVMe if not accessed in a long time
            stats.last_access.elapsed() > Duration::from_secs(60)
        }
        Tier::Nvme => false,  // Already coldest
    }
}
```

### 5.3 Prefetching

Predict next tensor needs based on forward pass position:

```rust
fn prefetch_for_layer(current_layer: usize, num_layers: usize, prefetch_depth: usize) {
    for offset in 1..=prefetch_depth {
        let next_layer = current_layer + offset;
        if next_layer < num_layers {
            // Async prefetch from RAM/NVMe to staging buffer
            prefetch_layer_tensors(next_layer);
        }
    }
}
```

### 5.4 KV Cache Pressure

As KV cache grows during long context, dynamically free VRAM:

```rust
fn handle_kv_cache_growth(kv_size: u64, vram_headroom: u64) {
    if kv_size > vram_headroom {
        let to_evict = kv_size - vram_headroom;

        // Evict lowest-priority VRAM tensors to RAM
        let eviction_candidates = allocation_map
            .iter()
            .filter(|(_, a)| a.tier == Tier::Vram)
            .sorted_by_key(|(_, a)| OrderedFloat(a.priority));

        let mut evicted = 0;
        for (name, alloc) in eviction_candidates {
            if evicted >= to_evict { break; }
            demote_to_ram(name);
            evicted += tensor_size(name);
        }
    }
}
```

---

## 6. API Design

### 6.1 Configuration

```rust
/// Configuration for adaptive memory tiering.
#[derive(Debug, Clone)]
pub struct AdaptiveTieringConfig {
    /// VRAM budget in bytes (default: auto-detect - 2GB headroom)
    pub vram_budget: u64,

    /// RAM budget in bytes (default: auto-detect - 4GB headroom)
    pub ram_budget: u64,

    /// Minimum quality target [0.0, 1.0] (default: 0.95)
    /// Higher values prefer BF16, lower values allow more quantization
    pub quality_target: f32,

    /// Maximum acceptable layer swap latency in ms (default: 100)
    /// If swapping would exceed this, prefer lower precision in VRAM
    pub max_swap_latency_ms: u32,

    /// Enable runtime adaptation (default: true)
    pub enable_adaptation: bool,

    /// Prefetch depth (default: 2 layers)
    pub prefetch_depth: usize,

    /// Enable mixed precision in VRAM (default: true)
    /// If false, all VRAM tensors use same precision
    pub enable_mixed_precision: bool,
}

impl Default for AdaptiveTieringConfig {
    fn default() -> Self {
        Self {
            vram_budget: 0,  // Auto-detect
            ram_budget: 0,   // Auto-detect
            quality_target: 0.95,
            max_swap_latency_ms: 100,
            enable_adaptation: true,
            prefetch_depth: 2,
            enable_mixed_precision: true,
        }
    }
}
```

### 6.2 Allocation Map

```rust
/// Memory tier for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// GPU VRAM - fastest, limited capacity
    Vram,
    /// System RAM - fast, larger capacity
    Ram,
    /// NVMe/Disk - slowest, unlimited capacity
    Nvme,
}

/// Precision level for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPrecision {
    BF16,
    FP8,
    INT8,
    INT4,
}

/// Allocation decision for a single tensor.
#[derive(Debug, Clone)]
pub struct TensorAllocation {
    /// Which memory tier to place this tensor
    pub tier: MemoryTier,
    /// What precision to use
    pub precision: TensorPrecision,
    /// Priority for eviction decisions [0.0, 1.0]
    pub priority: f32,
    /// Whether this tensor should be prefetched
    pub prefetch: bool,
}

/// Complete allocation plan for a model.
#[derive(Debug, Clone)]
pub struct AllocationPlan {
    /// Per-tensor allocation decisions
    pub allocations: HashMap<String, TensorAllocation>,
    /// Expected VRAM usage
    pub vram_usage: u64,
    /// Expected RAM usage
    pub ram_usage: u64,
    /// Expected NVMe usage
    pub nvme_usage: u64,
    /// Number of tensors requiring runtime swapping
    pub swap_count: usize,
    /// Estimated quality score [0.0, 1.0]
    pub quality_score: f32,
}
```

### 6.3 Planner Interface

```rust
/// Adaptive allocation planner.
pub struct AllocationPlanner {
    config: AdaptiveTieringConfig,
}

impl AllocationPlanner {
    pub fn new(config: AdaptiveTieringConfig) -> Self;

    /// Plan allocation for a model.
    ///
    /// Analyzes model structure and hardware constraints to produce
    /// an optimal allocation plan.
    pub fn plan(&self, model: &ModelProfile) -> Result<AllocationPlan>;

    /// Replan with updated constraints (e.g., KV cache growth).
    pub fn replan(&self,
        current: &AllocationPlan,
        new_constraints: &Constraints
    ) -> Result<AllocationPlan>;
}

/// Model profile for allocation planning.
pub struct ModelProfile {
    /// Tensor metadata (name, size, dtype)
    pub tensors: Vec<TensorInfo>,
    /// Number of layers
    pub num_layers: usize,
    /// Architecture type for heuristics
    pub architecture: ArchitectureType,
    /// Optional HoloTensor quality curves
    pub quality_curves: Option<HashMap<String, QualityCurve>>,
}
```

---

## 7. Integration Points

### 7.1 Engine Integration

Replace current fixed allocation in `engine.rs` with backend-aware loading:

```rust
// BEFORE (v0.1 - wrapped TieredHoloLoader, still slow)
let planner = AllocationPlanner::new(config);
let plan = planner.plan(&profile)?;
let tiered_loader = TieredHoloLoader::new(...);
let adaptive_loader = AdaptiveLoader::new(plan, tiered_loader);  // WRONG: inherits streaming overhead

// AFTER (v0.2 - backend-aware, uses eager loading when possible)
let planner = AllocationPlanner::new(config);
let profile = ModelProfile::from_hct_directory(directory)?;
let plan = planner.plan(&profile)?;

// Select loading backend based on allocation plan
let backend = select_loading_backend(&plan);

let loaded_model = match backend {
    LoadingBackend::Eager => {
        // Fast path: decompress all tensors, place per allocation plan
        tracing::info!(
            vram_gb = plan.vram_usage / GB,
            ram_gb = plan.ram_usage / GB,
            "Using eager loading (model fits in memory)"
        );
        load_eager_with_placement(&plan, directory, device, dtype)?
    }
    LoadingBackend::EagerWithRamCache => {
        // Medium path: eager load to VRAM + RAM, LRU cache for overflow
        tracing::info!(
            vram_gb = plan.vram_usage / GB,
            ram_gb = plan.ram_usage / GB,
            swap_count = plan.swap_count,
            "Using eager loading with RAM overflow cache"
        );
        load_eager_with_ram_cache(&plan, directory, device, dtype)?
    }
    LoadingBackend::Progressive => {
        // Slow path: only for 405B+ models that truly need layer swapping
        tracing::info!(
            swap_count = plan.swap_count,
            "Using progressive loading (405B+ mode)"
        );
        load_progressive(&plan, directory, device, dtype)?
    }
};
```

### 7.2 Eager Loading with Placement

The fast path uses `hct_sequential` with placement hints:

```rust
fn load_eager_with_placement(
    plan: &AllocationPlan,
    directory: &Path,
    device: &Device,
    dtype: DType,
) -> Result<EagerLoadedModel> {
    // Phase 1: Classify tensors by reconstruction path (from profile)
    let gpu_tensors: Vec<_> = plan.allocations.iter()
        .filter(|(_, a)| a.reconstruction_path == ReconstructionPath::GpuFast)
        .collect();
    let cpu_tensors: Vec<_> = plan.allocations.iter()
        .filter(|(_, a)| a.reconstruction_path == ReconstructionPath::CpuDirect)
        .collect();

    tracing::info!(
        gpu_count = gpu_tensors.len(),
        cpu_count = cpu_tensors.len(),
        "Classified tensors for reconstruction"
    );

    // Phase 2: Decompress tensors (parallel where possible)
    let mut vram_tensors = HashMap::new();
    let mut ram_tensors = HashMap::new();

    // GPU reconstruction for large 2D tensors (parallel batch)
    for (name, alloc) in gpu_tensors {
        let tensor = decompress_gpu(directory, name, device, dtype)?;
        match alloc.tier {
            MemoryTier::Vram => { vram_tensors.insert(name.clone(), tensor); }
            MemoryTier::Ram => { ram_tensors.insert(name.clone(), tensor.to_device(&Device::Cpu)?); }
            _ => unreachable!("Eager mode doesn't use NVMe"),
        }
    }

    // CPU reconstruction for 1D/small tensors (no GPU attempt, no warning)
    for (name, alloc) in cpu_tensors {
        let tensor = decompress_cpu(directory, name, dtype)?;
        match alloc.tier {
            MemoryTier::Vram => { vram_tensors.insert(name.clone(), tensor.to_device(device)?); }
            MemoryTier::Ram => { ram_tensors.insert(name.clone(), tensor); }
            _ => unreachable!(),
        }
    }

    Ok(EagerLoadedModel {
        vram_tensors,
        ram_tensors,
        plan: plan.clone(),
    })
}
```

### 7.3 Loader Integration

The `AdaptiveLoader` wraps the eager-loaded model for `TensorProvider` compatibility:

```rust
impl TensorProvider for AdaptiveLoader {
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let alloc = self.plan.allocations.get(name)?;

        match alloc.tier {
            MemoryTier::Vram => {
                // Direct VRAM access, possibly with dequantization
                self.get_vram_tensor(name, alloc.precision, device, dtype)
            }
            MemoryTier::Ram => {
                // Transfer from RAM, update stats
                self.get_ram_tensor(name, device, dtype)
            }
            MemoryTier::Nvme => {
                // Load from disk, update stats
                self.get_nvme_tensor(name, device, dtype)
            }
        }
    }
}
```

### 7.4 Quantization Integration

Leverage existing GPU dequantization kernels:

```rust
fn get_vram_tensor(
    &self,
    name: &str,
    precision: TensorPrecision,
    device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    match precision {
        TensorPrecision::BF16 => {
            // Direct load, no conversion
            self.vram_cache.get(name)
        }
        TensorPrecision::INT8 => {
            // GPU dequantization kernel
            let quantized = self.vram_cache.get_quantized(name)?;
            gpu_dequant::int8_to_bf16(quantized, device)
        }
        TensorPrecision::INT4 => {
            // GPU dequantization kernel (fused with GEMM if possible)
            let quantized = self.vram_cache.get_quantized(name)?;
            gpu_dequant::int4_to_bf16(quantized, device)
        }
    }
}
```

---

## 8. Metrics and Observability

### 8.1 Allocation Metrics

```rust
pub struct AllocationMetrics {
    /// Total tensors in each tier
    pub tier_counts: HashMap<MemoryTier, usize>,
    /// Bytes used in each tier
    pub tier_bytes: HashMap<MemoryTier, u64>,
    /// Tensors at each precision
    pub precision_counts: HashMap<TensorPrecision, usize>,
    /// Number of runtime swaps performed
    pub swap_count: AtomicU64,
    /// Total swap latency (nanoseconds)
    pub swap_latency_ns: AtomicU64,
    /// Cache hit rate for RAM tier
    pub ram_hit_rate: AtomicU64,
    /// Prefetch effectiveness (hits / prefetches)
    pub prefetch_hit_rate: AtomicU64,
}
```

### 8.2 Logging

```rust
// At initialization
tracing::info!(
    model = %model_name,
    total_params = %total_params,
    vram_tensors = plan.tier_counts[&Vram],
    ram_tensors = plan.tier_counts[&Ram],
    nvme_tensors = plan.tier_counts[&Nvme],
    bf16_count = plan.precision_counts[&BF16],
    int8_count = plan.precision_counts[&INT8],
    int4_count = plan.precision_counts[&INT4],
    expected_swaps_per_token = plan.swap_count,
    quality_score = plan.quality_score,
    "Adaptive allocation plan computed"
);

// During inference (periodic)
tracing::debug!(
    swaps = metrics.swap_count.load(Relaxed),
    avg_swap_ms = metrics.swap_latency_ns.load(Relaxed) / 1_000_000 / swaps,
    ram_hit_rate = metrics.ram_hit_rate(),
    prefetch_effectiveness = metrics.prefetch_hit_rate(),
    "Adaptive tiering stats"
);
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```rust
#[test]
fn test_importance_scoring() {
    // Edge layers should have higher importance
    assert!(importance("layers.0.self_attn") > importance("layers.20.self_attn"));

    // Attention should have higher importance than MLP
    assert!(importance("layers.10.self_attn") > importance("layers.10.mlp"));

    // Embeddings should have maximum importance
    assert_eq!(importance("embed_tokens"), 1.0);
}

#[test]
fn test_allocation_fits_budget() {
    let plan = planner.plan(&model_14b, vram_24gb, ram_64gb);
    assert!(plan.vram_usage <= vram_24gb);
    assert!(plan.ram_usage <= ram_64gb);
}

#[test]
fn test_14b_no_swapping() {
    // 14B model should fit in 24GB VRAM with mixed precision
    let plan = planner.plan(&model_14b, vram_24gb, ram_64gb);
    assert_eq!(plan.swap_count, 0);
}
```

### 9.2 Integration Tests

```rust
#[test]
fn test_adaptive_inference_speed() {
    // Compare adaptive vs fixed allocation
    let adaptive_throughput = benchmark_adaptive(&model_14b, 100_tokens);
    let fixed_throughput = benchmark_fixed_layer_swap(&model_14b, 100_tokens);

    // Adaptive should be at least 5x faster for 14B on 24GB
    assert!(adaptive_throughput > fixed_throughput * 5.0);
}
```

### 9.3 Benchmarks

```rust
// benches/adaptive_tiering.rs
fn benchmark_allocation_planning(c: &mut Criterion) {
    c.bench_function("plan_14b", |b| {
        b.iter(|| planner.plan(&model_14b))
    });

    c.bench_function("plan_70b", |b| {
        b.iter(|| planner.plan(&model_70b))
    });
}

fn benchmark_inference_throughput(c: &mut Criterion) {
    let models = [
        ("14B_adaptive", setup_adaptive_14b()),
        ("14B_fixed", setup_fixed_14b()),
        ("70B_adaptive", setup_adaptive_70b()),
    ];

    for (name, engine) in models {
        c.bench_function(name, |b| {
            b.iter(|| engine.generate(prompt, 100))
        });
    }
}
```

---

## 10. Migration Path

### 10.1 Phase 1: Allocation Planner (Week 1-2)

1. Implement `ImportanceScorer` with heuristics
2. Implement `AllocationPlanner` with greedy algorithm
3. Add configuration and API types
4. Unit tests for scoring and planning

### 10.2 Phase 2: Adaptive Loader (Week 2-3)

1. Implement `AdaptiveLoader` replacing `TieredHoloLoader`
2. Integrate with existing quantization kernels
3. Add prefetching infrastructure
4. Integration tests with 14B model

### 10.3 Phase 3: Runtime Adaptation (Week 3-4)

1. Implement access pattern monitoring
2. Implement dynamic reallocation
3. Implement KV cache pressure handling
4. Performance benchmarks and tuning

### 10.4 Phase 4: Production Hardening (Week 4+)

1. Metrics and observability
2. Configuration documentation
3. Performance regression tests
4. Memory safety auditing

---

## 11. Open Questions

1. **Quantization during inference vs at load time?**
   - Load-time: simpler, one-time cost
   - Inference-time (fused GEMM): lower memory, but compute overhead

2. **How to handle adapter/LoRA weights?**
   - Should adapters always be in VRAM?
   - How does this interact with base model allocation?

3. **Multi-GPU allocation?**
   - How to split allocation across GPUs?
   - Tensor parallelism vs pipeline parallelism implications?

4. **Quality regression testing?**
   - How to verify quantization doesn't degrade output quality?
   - Automated quality benchmarks needed?

---

## 12. References

- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
- [FlexGen: Offloading for LLM Inference](https://arxiv.org/abs/2303.06865)
