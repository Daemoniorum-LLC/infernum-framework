# Infernum Performance Roadmap: Closing the TensorRT-LLM Gap

## Implementation Status

| Phase | Component | Status | File |
|-------|-----------|--------|------|
| **Phase 1** | Flash Attention 2 | ✅ Complete | `abaddon/kernels.sigil` |
| **Phase 1** | Fused RMSNorm | ✅ Complete | `abaddon/kernels.sigil` |
| **Phase 1** | Fused SwiGLU | ✅ Complete | `abaddon/kernels.sigil` |
| **Phase 1** | Fused RoPE | ✅ Complete | `abaddon/kernels.sigil` |
| **Phase 1** | CUDA Graph Capture | ✅ Complete | `abaddon/kernels.sigil` |
| **Phase 2** | INT8 Quantization | ✅ Complete | `abaddon/quantization.sigil` |
| **Phase 2** | FP8 Support (Hopper) | ✅ Complete | `abaddon/quantization.sigil` |
| **Phase 2** | GPTQ/AWQ 4-bit | ✅ Structure | `abaddon/quantization.sigil` |
| **Phase 2** | SmoothQuant | ✅ Complete | `abaddon/quantization.sigil` |
| **Phase 3** | Continuous Batching | ✅ Complete | `abaddon/scheduler.sigil` |
| **Phase 3** | Preemption (Swap/Recompute) | ✅ Complete | `abaddon/scheduler.sigil` |
| **Phase 3** | Priority Scheduling | ✅ Complete | `abaddon/scheduler.sigil` |
| **Phase 3** | Chunked Prefill | ✅ Complete | `abaddon/scheduler.sigil` |
| **Phase 4** | Draft Model Speculation | ✅ Complete | `abaddon/speculative.sigil` |
| **Phase 4** | Self-Speculative Decoding | ✅ Complete | `abaddon/speculative.sigil` |
| **Phase 4** | Medusa Multi-Head | ✅ Complete | `abaddon/speculative.sigil` |
| **Phase 5** | Paged Attention | ✅ Complete | `abaddon/paged_attention.sigil` |
| **Phase 5** | KV Cache Compression | ✅ Complete | `abaddon/paged_attention.sigil` |
| **Phase 5** | Copy-on-Write | ✅ Complete | `abaddon/paged_attention.sigil` |
| **Integration** | OptimizedEngine | ✅ Complete | `abaddon/optimized_engine.sigil` |

### Usage

```sigil
// Maximum throughput configuration
let engine = OptimizedEngineBuilder::new(config)
    .max_throughput()
    .build()|await?

// Low latency configuration
let engine = OptimizedEngineBuilder::new(config)
    .low_latency()
    .build()|await?

// Memory efficient configuration
let engine = OptimizedEngineBuilder::new(config)
    .memory_efficient()
    .build()|await?

// Custom configuration
let engine = OptimizedEngineBuilder::new(config)
    .with_flash_attention(true)
    .with_quantization(QuantizationConfig::int8())
    .with_speculative(SpeculativeDecodingConfig::self_speculative(8))
    .with_paged_attention(PagedAttentionConfig::default())
    .build()|await?
```

---

## Current Performance Gap

| Metric | TensorRT-LLM | Infernum | Gap |
|--------|--------------|----------|-----|
| Throughput | 3,200 tok/s | 1,950 tok/s | **-39%** |
| p50 Latency | 28ms | 44ms | **+57%** |
| p99 Latency | 95ms | 140ms | **+47%** |
| Memory | 13.8 GB | 14.0 GB | +1% |

## Root Cause Analysis

TensorRT-LLM's advantages come from:

1. **TensorRT Graph Optimization** - Fused kernels, layer elimination
2. **FP8/INT8 Quantization** - Native Hopper support
3. **In-flight Batching** - Continuous batching with preemption
4. **Paged KV Cache** - Memory-efficient attention
5. **Speculative Decoding** - Draft model acceleration

## Optimization Roadmap

### Phase 1: Kernel Optimization (Target: +25% throughput)

#### 1.1 Flash Attention Integration
```sigil
// Current: Standard attention
fn attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor {
    let scores = q.matmul(k.transpose(-2, -1)) / sqrt(d_k)
    let weights = scores.softmax(-1)
    weights.matmul(v)
}

// Optimized: Flash Attention 2
fn flash_attention(q: Tensor, k: Tensor, v: Tensor, block_size: u32 = 128) -> Tensor {
    // Tiled computation with online softmax
    cuda::flash_attention_v2(q, k, v, block_size, causal: true)
}
```

**Implementation:**
- Integrate `flash-attn` crate or write custom CUDA kernels
- Support both FP16 and BF16
- Add causal masking optimization

**Expected gain:** 15-20% latency reduction

#### 1.2 Fused Operations
```sigil
// Current: Separate operations
fn feed_forward(x: Tensor) -> Tensor {
    let gate = x.linear(w_gate).silu()
    let up = x.linear(w_up)
    let hidden = gate * up
    hidden.linear(w_down)
}

// Optimized: Fused SwiGLU kernel
fn fused_feed_forward(x: Tensor) -> Tensor {
    cuda::fused_swiglu(x, w_gate, w_up, w_down)
}
```

**Fused kernels to implement:**
- `fused_attention` - Q/K/V projection + attention + output projection
- `fused_swiglu` - Gate + Up projection + SiLU + multiply + Down projection
- `fused_rms_norm` - RMSNorm + residual add
- `fused_rope` - Rotary position embedding

**Expected gain:** 10-15% throughput improvement

### Phase 2: Quantization Support (Target: +40% throughput)

#### 2.1 INT8 Weight Quantization
```sigil
// Add quantization config to model loading
struct QuantizationConfig {
    weight_bits: u8 = 8,
    activation_bits: u8 = 16,
    group_size: u32 = 128,
    symmetric: bool = true,
}

impl Engine {
    fn load_quantized(path: &Path, config: QuantizationConfig) -> Result<Engine> {
        let weights = load_int8_weights(path, config.group_size)?
        let scales = load_scales(path)?

        Engine {
            weights: QuantizedWeights { data: weights, scales, config },
            ..
        }
    }
}
```

#### 2.2 FP8 Support (Hopper GPUs)
```sigil
#[cfg(feature = "hopper")]
fn fp8_matmul(a: Tensor<f8e4m3>, b: Tensor<f8e4m3>, scale: f32) -> Tensor<f16> {
    cuda::fp8_gemm(a, b, scale, output_dtype: f16)
}
```

**Quantization methods to support:**
- GPTQ (4-bit)
- AWQ (4-bit)
- SmoothQuant (INT8)
- FP8 (E4M3/E5M2)

**Expected gain:** 30-50% throughput, 40% memory reduction

### Phase 3: Advanced Batching (Target: +20% throughput)

#### 3.1 Continuous Batching with Preemption
```sigil
struct BatchScheduler {
    running_batch: Mutex<[Request]>,
    waiting_queue: Mutex<PriorityQueue<Request>>,
    preemption_policy: PreemptionPolicy,
}

enum PreemptionPolicy {
    /// Swap to CPU memory
    Swap,
    /// Recompute from checkpoint
    Recompute,
    /// No preemption (current behavior)
    None,
}

impl BatchScheduler {
    async fn schedule(mut self) -> [Request] {
        let mut batch = self.running_batch.lock()

        // Check for higher priority requests
        if let Some(urgent) = self.waiting_queue.peek_priority() {
            if urgent > batch.min_priority() && self.preemption_policy != PreemptionPolicy::None {
                // Preempt lowest priority request
                let preempted = batch.pop_lowest()
                self.preempt(preempted)|await
            }
        }

        // Fill batch with waiting requests
        while batch.len() < self.max_batch_size {
            if let Some(req) = self.waiting_queue.pop() {
                batch.push(req)
            } else {
                break
            }
        }

        batch.clone()
    }
}
```

#### 3.2 Chunked Prefill
```sigil
/// Split long prompts into chunks to reduce TTFT
fn chunked_prefill(prompt: &[Token], chunk_size: u32 = 512) -> impl Iterator<Item = &[Token]> {
    prompt.chunks(chunk_size as usize)
}

impl Engine {
    async fn generate_chunked(self, request: GenerateRequest) -> Result<Response> {
        let tokens = self.tokenize(&request.prompt)?

        // Process prompt in chunks
        let mut kv_cache = KvCache::new()
        for chunk in chunked_prefill(&tokens, 512) {
            kv_cache = self.prefill_chunk(chunk, kv_cache)|await?
        }

        // Continue with decode
        self.decode(kv_cache, request.sampling)|await
    }
}
```

**Expected gain:** 15-25% throughput, better TTFT

### Phase 4: Speculative Decoding (Target: +50% throughput)

#### 4.1 Draft Model Integration
```sigil
struct SpeculativeDecoder {
    /// Main model (e.g., 70B)
    target: Arc<Engine>,
    /// Draft model (e.g., 7B)
    draft: Arc<Engine>,
    /// Speculation length
    k: u32 = 5,
    /// Acceptance threshold
    threshold: f32 = 0.9,
}

impl SpeculativeDecoder {
    async fn generate(self, prompt: &str) -> Result<str> {
        let mut output = String::new()
        let mut kv_target = KvCache::new()
        let mut kv_draft = KvCache::new()

        loop {
            // Draft: Generate k tokens quickly
            let draft_tokens = self.draft.generate_n(
                &output,
                self.k,
                &mut kv_draft,
            )|await?

            // Target: Verify all k tokens in parallel
            let (accepted, target_probs) = self.target.verify_batch(
                &output,
                &draft_tokens,
                &mut kv_target,
            )|await?

            // Accept matching tokens
            for (i, token) in draft_tokens.iter().enumerate() {
                if accepted[i] {
                    output.push_str(&self.decode_token(*token))
                } else {
                    // Resample from target distribution
                    let resampled = self.resample(target_probs[i], draft_tokens[i])
                    output.push_str(&self.decode_token(resampled))
                    break
                }
            }

            if output.contains(EOS_TOKEN) {
                break
            }
        }

        Ok(output)
    }
}
```

#### 4.2 Self-Speculative Decoding (No Draft Model)
```sigil
/// Use early exit from transformer layers as draft
struct SelfSpeculativeDecoder {
    model: Arc<Engine>,
    draft_layers: u32 = 8,  // Use first 8 layers as "draft"
    k: u32 = 4,
}

impl SelfSpeculativeDecoder {
    async fn generate(self, prompt: &str) -> Result<str> {
        // Draft: Run only first N layers
        let draft_tokens = self.model.forward_partial(
            prompt,
            self.draft_layers,
            self.k,
        )|await?

        // Verify: Run full model
        let accepted = self.model.verify(prompt, &draft_tokens)|await?

        // ... acceptance logic
    }
}
```

**Expected gain:** 40-60% throughput for greedy decoding

### Phase 5: Memory Optimization (Target: 30% memory reduction)

#### 5.1 Paged Attention
```sigil
struct PagedKvCache {
    /// Block size (number of tokens per block)
    block_size: u32 = 16,
    /// Physical blocks (pre-allocated GPU memory)
    physical_blocks: [KvBlock],
    /// Mapping from sequence -> block indices
    block_tables: HashMap<SeqId, [BlockId]>,
    /// Free block list
    free_blocks: Mutex<[BlockId]>,
}

impl PagedKvCache {
    fn allocate_block(mut self, seq_id: SeqId) -> Result<BlockId> {
        let block_id = self.free_blocks.lock().pop()
            ?? return Err(Error::OutOfMemory)

        self.block_tables.entry(seq_id)
            .or_default()
            .push(block_id)

        Ok(block_id)
    }

    fn free_sequence(mut self, seq_id: SeqId) {
        if let Some(blocks) = self.block_tables.remove(&seq_id) {
            self.free_blocks.lock().extend(blocks)
        }
    }
}
```

#### 5.2 KV Cache Compression
```sigil
struct CompressedKvCache {
    /// Compression method
    method: KvCompressionMethod,
    /// Full precision for recent tokens
    recent_tokens: u32 = 128,
}

enum KvCompressionMethod {
    /// Quantize old KV to INT8
    Quantize { bits: u8 },
    /// Keep only important tokens (H2O)
    Evict { keep_ratio: f32 },
    /// Merge similar KV vectors
    Merge { threshold: f32 },
}
```

### Implementation Timeline

```
Phase 1 (Kernels):      ████████████░░░░░░░░  [2-3 weeks]
Phase 2 (Quantization): ░░░░████████████░░░░  [3-4 weeks]
Phase 3 (Batching):     ░░░░░░░░████████░░░░  [2-3 weeks]
Phase 4 (Speculative):  ░░░░░░░░░░░░████████  [3-4 weeks]
Phase 5 (Memory):       ░░░░████████████████  [ongoing]
```

### Projected Performance After Optimization

| Phase | Throughput | vs TensorRT-LLM |
|-------|------------|-----------------|
| Current | 1,950 tok/s | 61% |
| + Phase 1 | 2,440 tok/s | 76% |
| + Phase 2 | 3,415 tok/s | **107%** |
| + Phase 3 | 4,100 tok/s | 128% |
| + Phase 4 | 6,150 tok/s | 192% |

### Quick Wins (Implement This Week)

1. **Flash Attention** - Biggest single improvement
   ```bash
   # Add dependency
   cargo add flash-attn
   ```

2. **Fused RMSNorm** - Easy kernel fusion
   ```sigil
   // Replace separate norm + residual with fused version
   fn fused_rms_norm_residual(x: Tensor, residual: Tensor, weight: Tensor) -> Tensor {
       cuda::fused_rms_norm_add(x, residual, weight, eps: 1e-6)
   }
   ```

3. **CUDA Graph Capture** - Reduce kernel launch overhead
   ```sigil
   impl Engine {
       fn capture_decode_graph(self, batch_size: u32) -> CudaGraph {
           cuda::capture_graph(|| {
               self.decode_step_static(batch_size)
           })
       }
   }
   ```

### Competitive Positioning After Optimization

```
                         Throughput (tok/s)
                    0     1000    2000    3000    4000    5000    6000
TensorRT-LLM (now)  ├─────────────────────────────────┤ 3,200
vLLM (now)          ├─────────────────────────┤ 2,400
Infernum (now)      ├───────────────────┤ 1,950
Infernum (Phase 2)  ├──────────────────────────────────────┤ 3,415
Infernum (Phase 4)  ├────────────────────────────────────────────────────────────┤ 6,150
```

## Summary

The gap with TensorRT-LLM is closable through:

1. **Flash Attention** (+20%) - Must have
2. **INT8/FP8 Quantization** (+40%) - Critical for throughput
3. **Speculative Decoding** (+50%) - Game changer for greedy
4. **Kernel Fusion** (+15%) - Polish

With all optimizations, Infernum can **exceed** TensorRT-LLM throughput while maintaining its unique advantages:
- Full-stack architecture
- Evidentiality types
- Built-in A/B testing
- Sigil language efficiency
