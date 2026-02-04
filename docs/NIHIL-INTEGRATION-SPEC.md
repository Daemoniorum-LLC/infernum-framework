# Nihil Integration Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-03
**Audience:** Nihil development agent, Infernum maintainers
**Spec Reference:** GPU-CODEC-PIPELINE-SPEC.md (DD-9)

---

## 1. Purpose

This specification defines the contract between Infernum's inference engine
(abaddon) and the Nihil tensor framework. Nihil replaces Candle as the tensor
backend. The goal is 25 tk/s for 70B parameter models on a single 24GB VRAM
GPU, up from the current 1.3 tk/s bottlenecked by Candle.

This document tells the Nihil agent exactly what Infernum needs, in priority
order, so integration work can proceed without guesswork.

---

## 2. Candle Removal Scope

Candle (v0.9) is used across 5 crates and 30+ source files. The dependency
is deep but structurally contained.

### 2.1 Workspace Dependencies

```toml
# Current workspace Cargo.toml
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
candle-flash-attn = "0.9"  # cuda feature only
```

### 2.2 Crate-by-Crate Impact

| Crate | Files | Candle Types Used | Role |
|-------|:-----:|-------------------|------|
| **abaddon** | 28 | Tensor, Device, DType, VarBuilder, Module, Embedding, Linear, IndexOp, D | Inference engine (core) |
| **infernum-server** | 1 | DType, Device, VarBuilder | Speculative engine wrapper |
| **malphas** | 1 | Tensor, Device, DType, IndexOp | Legion speculative routing |
| **asmodeus** | 4 | Tensor, Device, DType, VarBuilder | LoRA, gradients, training |
| **infernum** (CLI) | 1 | Tensor, Device, DType | Safetensors conversion, CUDA probing |

**Not affected:** infernum-core, stolas, beleth, grimoire-loader, dantalion,
legion, arbiter. These crates do not import Candle.

### 2.3 Existing Abstraction Layer

Abaddon already defines backend abstraction traits in `backend.rs`:

```rust
// crates/abaddon/src/backend.rs
pub trait TensorOps { /* ... */ }
pub trait DeviceOps { /* ... */ }
pub trait ComputeBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, ...) -> Result<Tensor>;
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>;
    // ...
}
```

Additionally, `cuda_inference/` uses a custom `GpuTensor` type for direct CUDA
kernel dispatch, independent of Candle's GPU support. This path is already
Candle-free and will not require migration.

### 2.4 Key Type Mapping

| Candle Type | Nihil Equivalent | Notes |
|-------------|-----------------|-------|
| `candle_core::Tensor` | `nihil::Tensor<S, D, Dev>` | Nihil adds compile-time shapes |
| `candle_core::Device` | `nihil::Device` trait / `Cpu`, `Cuda` | Nihil uses trait-based dispatch |
| `candle_core::DType` | `nihil::DType` trait | Nihil uses trait not enum |
| `candle_nn::VarBuilder` | `nihil_io::SafetensorsFile` + manual loading | See section 3.3 |
| `candle_nn::Module` trait | Custom aspect or direct `forward()` | See section 3.4 |
| `candle_nn::Embedding` | `nihil_nn::Embedding` | Direct equivalent |
| `candle_nn::Linear` | `nihil_nn::Linear` | Direct equivalent |
| `candle_core::IndexOp` | `nihil::View` / `nihil::Slice` | Nihil uses view system |
| `candle_core::D` (dimension) | Compile-time const generics | Nihil resolves dimensions at compile time |
| `candle_core::Result` | `nihil::Result<T>` | Both are `Result<T, Error>` |

---

## 3. Nihil Contract

These are the capabilities Nihil must expose for Infernum integration,
ordered by priority.

### 3.1 Tensor Core (Priority: BLOCKING)

Abaddon uses these tensor operations in the hot path. Every model forward
pass depends on them.

**Creation:**
- `Tensor::zeros(shape, dtype, device)`
- `Tensor::ones(shape, dtype, device)`
- `Tensor::new(data: &[T], shape, device)` (from host slice)
- `Tensor::cat(tensors: &[&Tensor], dim)` (concatenation)
- `Tensor::from_raw_buffer(bytes, dtype, shape, device)` (for weight loading)

**Arithmetic (element-wise with broadcasting):**
- `add`, `sub`, `mul`, `div`
- `broadcast_add`, `broadcast_mul`, `broadcast_div`
- `sqr`, `sqrt`, `rsqrt`
- `neg`, `abs`
- `exp`, `log`

**Matrix Operations:**
- `matmul(a, b)` (the critical path for inference)
- `transpose(dim0, dim1)`

**Shape Operations:**
- `reshape(new_shape)`
- `unsqueeze(dim)`
- `squeeze(dim)`
- `narrow(dim, start, len)` (slicing along a dimension)
- `contiguous()` (ensure contiguous memory layout)
- `dims()` -> shape as slice
- `dim(idx)` -> size of specific dimension
- `elem_count()` -> total number of elements
- `dtype()` -> data type
- `device()` -> device reference

**Type Conversion:**
- `to_dtype(target_dtype)` (F32 <-> F16 <-> BF16)
- `to_device(target_device)` (CPU <-> GPU)

**Reduction:**
- `mean_keepdim(dim)` (used in RMSNorm)
- `sum(dim)`, `max(dim)`, `argmax(dim)`
- `softmax(dim)` (used in attention)

**Trigonometric (for RoPE):**
- `cos()`, `sin()`

**Comparison / Selection:**
- `where_cond(condition, true_val, false_val)` (for attention masking)

### 3.2 Device Management (Priority: BLOCKING)

```
// Required API surface
Cpu::new() -> Device
Cuda::new(device_id: usize) -> Result<Device>
device.synchronize() -> Result<()>
device.is_cuda() -> bool
device.is_cpu() -> bool
```

Infernum selects the device at engine startup and passes it through all
model constructors. The device must be `Send + Sync` for async inference
serving.

### 3.3 Weight Loading (Priority: BLOCKING)

Candle's `VarBuilder` provides hierarchical weight access:

```rust
// Current Candle pattern
let vb = VarBuilder::from_safetensors(files, dtype, device)?;
let embed = vb.pp("model.embed_tokens").get((vocab, hidden), "weight")?;
let layer_0_qkv = vb.pp("model.layers.0.self_attn").pp("q_proj").get(shape, "weight")?;
```

Nihil must provide equivalent functionality. The interface need not be
identical, but must support:

1. **Safetensors loading**: Read `.safetensors` files, access tensors by name
2. **Hierarchical path traversal**: Navigate nested weight names (dot-separated)
3. **Dtype conversion on load**: Load F16 weights as BF16, etc.
4. **Device placement on load**: Load directly to GPU memory
5. **Lazy loading**: Load individual tensors on-demand (for 70B+ models that
   exceed VRAM). Abaddon's `LazyVarBuilder` with LRU cache is the pattern.
6. **GGUF format**: Parse GGUF files, dequantize weights on access

**Suggested Nihil API:**

```
let file = nihil_io::safetensors::load("model.safetensors")?;
let weight = file.get_dyn("model.layers.0.self_attn.q_proj.weight")?;
let weight_gpu = weight.to_device(&cuda)?.to_dtype(bf16)?;
```

### 3.4 Neural Network Layers (Priority: HIGH)

Abaddon's model files construct these layers directly:

| Layer | Candle Source | Nihil Equivalent | Used By |
|-------|-------------|-----------------|---------|
| `Embedding` | `candle_nn::embedding` | `nihil_nn::Embedding` | All models (token embed) |
| `Linear` (no bias) | `candle_nn::linear_no_bias` | `nihil_nn::Linear` | Llama Q/K/V/O, MLP |
| `Linear` (with bias) | `candle_nn::linear` | `nihil_nn::Linear` | Qwen2 Q/K/V/O |
| `RmsNorm` | Custom impl | `nihil_nn::RMSNorm` | All models (layer norm) |

The `Module` trait pattern (Candle: `impl Module for T { fn forward(&self, x: &Tensor) -> Result<Tensor> }`)
is convenient but not strictly required. Nihil's models can use direct
`forward()` methods instead.

### 3.5 Attention & KV Cache (Priority: HIGH)

Abaddon defines a `KvCache` trait:

```rust
pub trait KvCache: Send {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()>;
    fn get_kv(&self) -> Result<Option<(Tensor, Tensor)>>;
    fn forward_attention(&mut self, q: &Tensor, ...) -> Result<Tensor>;
    fn clear(&mut self);
}
```

Three implementations exist: StandardCache, QuantizedCache, CudaQuantizedCache.

**Nihil equivalents exist:** `KvCache`, `MultiLayerKvCache`, paged attention.
The Nihil KV cache uses pre-allocated tensors with a sequence length cursor
rather than concatenation, which is the more efficient approach. Abaddon's
`StandardCache` concatenates on every step (`Tensor::cat`), which is a
known performance problem.

**Contract:** Nihil's KV cache must support:
- Pre-allocated fixed-size buffers (avoids reallocation)
- Per-layer cache access (`MultiLayerKvCache::get_layer(idx)`)
- Sequence length tracking
- Cache reset (for new sequences)
- Optional: Paged attention for batch serving

### 3.6 RoPE (Priority: HIGH)

Abaddon implements two RoPE variants:

1. **Standard RoPE** (Llama 1/2/3): Precomputed cos/sin, applied via
   element-wise multiply and interleave
2. **NeoX-style RoPE** (Qwen2): Different rotation formula (split-half
   instead of interleave)

Nihil's `RotaryEmbedding` supports both via `RotaryConfig`, plus dynamic
scaling (Llama 3.1), NTK scaling, and YaRN. This exceeds what Abaddon
currently needs.

**Contract:** Nihil must expose `RotaryEmbedding::forward(q, k, positions)`
that returns transformed `(q_rot, k_rot)`.

### 3.7 Flash Attention (Priority: HIGH)

Currently feature-gated behind `candle-flash-attn` (CUDA only).

**Contract:** Nihil must provide `flash_attention(q, k, v, config)` with:
- Causal masking
- Configurable scale factor
- KV cache integration (`flash_attention_kv_cache`)
- F16/BF16 precision (tensor cores)

Nihil already has a 1,241 LOC Flash Attention 2 implementation targeting
Ada Lovelace tensor cores. This should exceed Candle's flash-attn wrapper.

### 3.8 Quantization (Priority: MEDIUM)

Abaddon supports quantized inference via:
- INT4 symmetric/asymmetric (GPTQ layout)
- INT8 symmetric/asymmetric
- GGUF quantized formats (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0)
- FP8 (via custom GPU kernels in `gpu_dtype.rs`)

**Contract:** Nihil must support:
1. Dequantization of GGUF weights on load (`gguf_file.dequantize(name)`)
2. FP8 E4M3 inference (Ada tensor cores)
3. INT4/INT8 dequantization for GPTQ/AWQ models

Nihil's `nihil-quant` already defines `QuantType` with all these formats
and `QuantizedTensor::dequantize()`.

### 3.9 Model Architectures (Priority: MEDIUM)

Abaddon currently implements:
- Llama (678 LOC) - Llama 1/2/3/3.1
- Qwen2 (871 LOC) - with sliding window attention
- LazyLlama (1,188 LOC) - lazy-loading variant
- LazyQwen2 (960 LOC) - lazy-loading variant
- QuantizedLlama (465 LOC) - quantized variant

Nihil defines Llama, Mistral, Qwen, Phi, Gemma, GPT2 in `nihil-models`.

**Two migration paths exist:**

**Path A: Nihil provides models directly.**
Abaddon calls `nihil_models::Llama::new(config).forward(input_ids, ...)`.
This is cleaner but requires Nihil's models to match Abaddon's inference
patterns (start_pos tracking, KV cache management, lazy loading).

**Path B: Abaddon keeps its model code, swaps tensor types.**
Replace `candle_core::Tensor` with `nihil::Tensor` and `candle_nn::*` with
`nihil_nn::*` throughout existing model files. More mechanical but preserves
Abaddon's battle-tested inference logic.

**Recommendation:** Path B for initial migration, Path A as a future goal.
Abaddon's model code handles edge cases (lazy loading, progressive
HoloTensor reconstruction, quantized variants) that Nihil's models don't
yet account for. Swapping the tensor type is lower risk than rewriting
the model code.

### 3.10 Autograd / LoRA (Priority: LOW)

Asmodeus (fine-tuning crate) uses Candle for:
- LoRA matrix creation (`Tensor::randn`, `Tensor::zeros`)
- Gradient computation
- Weight saving via safetensors

Nihil's `nihil-autograd` provides source-to-source AD with gradient tapes.
This is more powerful than what asmodeus currently uses.

**Contract:** Not blocking for inference. Can migrate after the core path.

---

## 4. DD-9 Compliance: PTX Generation Constraints

Nihil's `nihil-cuda` crate generates PTX dynamically via `PtxBuilder`.
The following constraints from DD-9 (GPU-CODEC-PIPELINE-SPEC.md section 4.5.1)
must be enforced at the emission boundary.

### 4.1 ASCII-Only Encoding

**The CUDA JIT compiler rejects any non-ASCII byte in PTX source, including
in comments.**

This is critical for Nihil because:
- Sigil uses Unicode mathematical notation extensively (`Sigma`, `in`, `->`,
  `forall`, `infty`, `odot`, `nabla`)
- Nihil's source comments contain Sigil operators (e.g., `// forall x in batch`)
- The `PtxBuilder` must strip or sanitize all non-ASCII from generated PTX

**Required:** An ASCII sanitization pass in `PtxBuilder::build()` that:
1. Strips all non-ASCII from comment strings
2. Rejects non-ASCII in instruction operands (hard error)
3. Replaces Unicode operators with ASCII equivalents in debug info

### 4.2 No Scientific Notation in Immediates

Floating-point literals like `5.96e-08` may fail JIT compilation.

**Required:** `PtxBuilder` must emit float literals as explicit division:
```
// Bad:  mov.f32 %r, 5.96046448e-08;
// Good: mov.f32 %r, 1.0; div.full.f32 %r, %r, 16777216.0;
```

### 4.3 No Negative Float Immediates

`add.f32 %r, %r, -1.0` fails on some architectures.

**Required:** Use `sub.f32` instead of `add.f32` with negative operands.

### 4.4 No Predicated mov on All Architectures

`@%p mov.u64 %r, 1` may fail.

**Required:** Use branch-based patterns:
```ptx
@%p bra SKIP;
mov.u64 %r, 1;
SKIP:
```

### 4.5 Testing

`nihil-cuda` must include a test that verifies PTX output from `PtxBuilder`
is pure ASCII:

```
fn test_ptx_builder_ascii_only() {
    let ptx = PtxBuilder::for_ada()
        .kernel("test", &[])
        .ret()
        .build();
    assert!(ptx.bytes().all(|b| b < 128),
        "PtxBuilder output must be ASCII-only (DD-9)");
}
```

---

## 5. Migration Strategy

### Phase 1: Type Shim (Non-Breaking)

Create a `nihil-compat` crate or module that re-exports Nihil types with
Candle-compatible names:

```rust
// nihil-compat/src/lib.rs
pub use nihil::Tensor;
pub type CandleResult<T> = nihil::Result<T>;
pub use nihil::{Cpu as Device_Cpu, Cuda as Device_Cuda};
// ... etc
```

This allows incremental migration: swap imports file-by-file without
breaking the build.

### Phase 2: Core Engine Swap

Migrate `engine.rs` and `backend.rs` first. These are the entry points
that create Device and DType, and the abstraction layer that all other
code calls through.

### Phase 3: Model Files

Migrate `llama.rs`, `qwen2.rs` and their lazy variants. Replace
`candle_nn::Embedding` with `nihil_nn::Embedding`, etc.

### Phase 4: Attention & Cache

Replace KV cache implementations with Nihil's pre-allocated approach.
This is where performance gains should be most visible.

### Phase 5: Peripheral Crates

Migrate infernum-server, malphas, asmodeus, infernum CLI. These have
minimal Candle usage (1-4 files each).

### Phase 6: Remove Candle

Delete `candle-*` from workspace Cargo.toml. Remove `nihil-compat` shim.

---

## 6. Acceptance Criteria

### 6.1 Functional Parity

All existing Infernum tests must pass after migration:
- `cargo test --workspace` (excluding CUDA-only tests on CI)
- Model loading from safetensors format
- Model loading from GGUF format
- HoloTensor progressive loading
- Speculative decoding (draft + oracle)
- KV cache correctness (standard + quantized)

### 6.2 Performance Targets

| Metric | Candle (Current) | Nihil (Target) | Measurement |
|--------|:----------------:|:--------------:|-------------|
| 70B inference, 24GB VRAM | 1.3 tk/s | 25 tk/s | `infernum chat --model 70B` |
| Time to first token (TTFT) | TBD | < 2s | Cold start, single sequence |
| KV cache memory | Grows per step | Pre-allocated | Fixed VRAM footprint |
| Kernel launch overhead | Candle default | CUDA graphs | < 10us per launch |

### 6.3 DD-9 Compliance

- `PtxBuilder::build()` output passes `bytes().all(|b| b < 128)` assertion
- No Unicode in generated PTX (comments, labels, or operands)
- Float immediate sanitization active by default

### 6.4 Compatibility

- Nihil Tensor type must be `Send + Sync` (async inference serving)
- Device selection must support `CUDA_VISIBLE_DEVICES` environment variable
- Weight loading must support memory-mapped I/O (for models larger than RAM)

---

## 7. Non-Goals

The following are explicitly out of scope for the initial integration:

1. **Training**: Asmodeus/LoRA migration is deferred (section 3.10)
2. **Multi-GPU**: Nihil's `nihil-distributed` is not needed for single-GPU target
3. **Python bindings**: `pynihil` is not relevant to Infernum
4. **Compile-time shape enforcement**: Abaddon uses dynamic shapes at runtime.
   Nihil's `DynShape` must work correctly for all operations.
5. **Sigil compiler fixes**: Blocked on the Sigil compiler team, not on this spec

---

## 8. Open Questions

1. **HoloTensor output type**: GPU HoloTensor reconstruction (`gpu_holo.rs`)
   currently outputs `CudaSlice<f32>` via `cudarc`. Should this output a
   Nihil tensor directly, or should there be a conversion step?

2. **Custom CUDA kernels**: Abaddon has hand-written PTX kernels for LZ4
   decompression, dequantization, and HoloTensor reconstruction. These
   use `cudarc` directly. Should these migrate to Nihil's `PtxBuilder`,
   or remain as standalone `cudarc` calls?

3. **Error type unification**: Candle uses `candle_core::Error`. Nihil uses
   `nihil::NihilError`. Abaddon has `AbaddonError`. What's the conversion
   strategy at boundaries?

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-03 | Infernum agent | Initial spec from Candle usage audit + Nihil API review |
