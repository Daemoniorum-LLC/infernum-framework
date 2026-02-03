# Infernum Enhancement Roadmap

**Version:** 1.0
**Created:** 2024-12-24
**Methodology:** Test-Driven Development with Code Review Checkpoints

---

## Overview

This roadmap interleaves quick wins with complex tasks to maintain development momentum while building toward significant capability improvements. Each phase concludes with a mandatory code review checkpoint.

### Guiding Principles

1. **TDD First**: Write tests before implementation. No feature merges without tests.
2. **Small PRs**: Each task should be a single, reviewable PR (< 500 lines ideal).
3. **Checkpoint Reviews**: End-of-phase reviews catch spec drift early.
4. **Working Software**: Each phase produces deployable improvements.

### TDD Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. SPEC    │  Write acceptance criteria & edge cases       │
├─────────────────────────────────────────────────────────────┤
│  2. RED     │  Write failing tests that define behavior     │
├─────────────────────────────────────────────────────────────┤
│  3. GREEN   │  Implement minimum code to pass tests         │
├─────────────────────────────────────────────────────────────┤
│  4. REFACTOR│  Clean up while keeping tests green           │
├─────────────────────────────────────────────────────────────┤
│  5. REVIEW  │  PR review checking spec alignment            │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation & Safety (Week 1-2)

**Theme:** Eliminate runtime panics, establish test infrastructure, quick visibility wins.

### Quick Wins (1-2 hours each)

#### 1.1 Add `#![deny(clippy::unwrap_used)]` to infernum-framework
- [ ] **Spec**: All crates reject `.unwrap()` at compile time
- [ ] **Test**: `cargo clippy` returns zero warnings
- [ ] **Impl**: Add deny directive to each crate's `lib.rs`
- [ ] **Files**: All `src/lib.rs` files in `infernum-framework/crates/`

#### 1.2 Fix Server Address Parsing Panic
- [ ] **Spec**: Invalid bind address returns error, doesn't panic
- [ ] **Test**: `ServerConfig::builder().addr("invalid").build()` returns `Err`
- [ ] **Impl**: Replace `.parse().unwrap()` with proper error handling
- [ ] **File**: `infernum-server/src/server.rs:46,98`

#### 1.3 Fix Trainer Lock Poisoning
- [ ] **Spec**: Poisoned locks return error, don't panic
- [ ] **Test**: Simulate panic in locked section, verify graceful handling
- [ ] **Impl**: Use `lock().unwrap_or_else(|e| e.into_inner())` pattern
- [ ] **File**: `asmodeus/src/trainer.rs:263,368,378`

#### 1.4 Add Model Warm-up on Server Start
- [ ] **Spec**: First real request has same latency as subsequent requests
- [ ] **Test**: Measure TTFT for request 1 vs request 2, assert within 10%
- [ ] **Impl**: Generate single token with dummy prompt at startup
- [ ] **File**: `infernum-server/src/server.rs` (new `warm_up()` method)

### Moderate Tasks (1-2 days each)

#### 1.5 Server Integration Test Harness
- [ ] **Spec**: Can spin up test server, make requests, assert responses
- [ ] **Test**: Meta-test that the harness itself works
- [ ] **Impl**: Create `tests/integration/` with test utilities
- [ ] **Files**:
  - `infernum-server/tests/integration/mod.rs`
  - `infernum-server/tests/integration/helpers.rs`
  - `infernum-server/tests/integration/chat_test.rs`

```rust
// Example test structure
#[tokio::test]
async fn test_chat_completion_basic() {
    let server = TestServer::start().await;
    let response = server.chat("Hello").await.unwrap();
    assert!(!response.choices.is_empty());
    assert!(!response.choices[0].message.content.is_empty());
}
```

#### 1.6 Mock Inference Engine for Testing
- [ ] **Spec**: Deterministic engine that returns predictable outputs
- [ ] **Test**: Mock engine returns configured responses
- [ ] **Impl**: Implement `InferenceEngine` trait with canned responses
- [ ] **File**: `infernum-framework/crates/test-utils/src/mock_engine.rs`

```rust
pub struct MockEngine {
    responses: HashMap<String, String>,
    latency: Duration,
}

impl InferenceEngine for MockEngine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        tokio::time::sleep(self.latency).await;
        // Return deterministic response based on prompt hash
    }
}
```

### Foundation for Next Phase

#### 1.7 Stolas Test Infrastructure
- [ ] **Spec**: Can test RAG pipeline without real embeddings
- [ ] **Test**: Mock embedder returns consistent vectors
- [ ] **Impl**: Create `MockEmbedder` for deterministic testing
- [ ] **File**: `stolas/src/embedding.rs` (already has `MockEmbedder`, verify coverage)

---

### 🔍 Checkpoint Review 1

**Trigger:** All Phase 1 tasks complete
**Duration:** 2-4 hours
**Participants:** 2+ reviewers

#### Review Checklist

- [ ] **Spec Drift**: Do implementations match original specs?
- [ ] **Test Coverage**: Run `cargo tarpaulin` - target 80% on modified files
- [ ] **Panic Safety**: `cargo clippy` clean with deny directives
- [ ] **Error Messages**: Are errors actionable? Do they include context?
- [ ] **Documentation**: Are public APIs documented?

#### Acceptance Criteria
```bash
# All must pass
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps
```

---

## Phase 2: Agent Intelligence (Week 3-4)

**Theme:** Make the agent actually useful with persistence and better tools.

### Quick Wins

#### 2.1 Agent Tool: Enhanced Project Tree
- [ ] **Spec**: Shows file tree respecting `.gitignore`, with sizes
- [ ] **Test**: Tree of test directory matches expected output
- [ ] **Impl**: Add gitignore parsing, size formatting
- [ ] **File**: `beleth/src/tool.rs` (`ProjectTreeTool`)

#### 2.2 Add Common Stop Sequences
- [ ] **Spec**: Generation stops at model-specific end tokens
- [ ] **Test**: Generation with Llama prompt stops at `<|eot_id|>`
- [ ] **Impl**: Auto-detect model family, add appropriate stops
- [ ] **File**: `abaddon/src/sampler.rs`

#### 2.3 Improve Engine Error Messages
- [ ] **Spec**: Errors include model name, layer, tensor shapes
- [ ] **Test**: Trigger OOM, verify error mentions tensor dimensions
- [ ] **Impl**: Wrap Candle errors with context
- [ ] **File**: `abaddon/src/engine.rs`

### Moderate Tasks

#### 2.4 Agent Memory Persistence (SQLite)
- [ ] **Spec**: Conversations persist across agent restarts
- [ ] **Test**:
  - Create conversation, restart agent, retrieve conversation
  - Search conversations by content
  - Delete old conversations
- [ ] **Impl**: SQLite backend for `AgentMemory`
- [ ] **Files**:
  - `beleth/src/memory/persistence.rs` (new)
  - `beleth/src/memory/mod.rs` (update)

```rust
// Schema
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    summary TEXT
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT,
    content TEXT,
    created_at TIMESTAMP
);

CREATE VIRTUAL TABLE messages_fts USING fts5(content, conversation_id);
```

#### 2.5 Agent Semantic Memory via Stolas
- [ ] **Spec**: Agent can store and retrieve facts by similarity
- [ ] **Test**:
  - Store "The capital of France is Paris"
  - Query "What is France's capital?" returns stored fact
- [ ] **Impl**: Connect Beleth to Stolas vector store
- [ ] **Files**:
  - `beleth/src/memory/semantic.rs` (new)
  - Integration with Stolas `VectorStore`

#### 2.6 Conversation Summarization
- [ ] **Spec**: Long conversations get summarized to fit context
- [ ] **Test**: 50-message conversation summarizes to < 1000 tokens
- [ ] **Impl**: Use inference engine to generate summaries
- [ ] **File**: `beleth/src/memory.rs` (`SummarizationStrategy`)

### Foundation for Next Phase

#### 2.7 BM25 Index Construction
- [ ] **Spec**: Can build inverted index from documents
- [ ] **Test**: Index 100 docs, verify term frequencies correct
- [ ] **Impl**: Ensure `BM25Index` is fully tested
- [ ] **File**: `stolas/src/bm25.rs`

---

### 🔍 Checkpoint Review 2

**Focus Areas:**
- [ ] **Memory Correctness**: No data loss on restart?
- [ ] **Semantic Search Quality**: Are retrieved facts relevant?
- [ ] **Performance**: Memory operations < 100ms?
- [ ] **Schema Migrations**: Is there a migration strategy?

#### Metrics to Collect
```
- Agent task completion rate (before/after)
- Average conversation length before context overflow
- Memory retrieval latency p50/p95/p99
```

---

## Phase 3: RAG Excellence (Week 5-6)

**Theme:** Complete the hybrid search pipeline for production-quality retrieval.

### Quick Wins

#### 3.1 Tokenizer LRU Cache
- [ ] **Spec**: Recently-used tokenizers cached in memory
- [ ] **Test**: Load same tokenizer 100x, measure no disk I/O after first
- [ ] **Impl**: Add `lru` cache to `Tokenizer::from_file`
- [ ] **File**: `abaddon/src/tokenizer.rs`

#### 3.2 Chunking Strategy Presets
- [ ] **Spec**: Named presets for common use cases (code, prose, markdown)
- [ ] **Test**: Code chunker respects function boundaries
- [ ] **Impl**: Add `ChunkingStrategy::for_code()`, etc.
- [ ] **File**: `stolas/src/chunker.rs`

### Moderate Tasks

#### 3.3 Wire BM25 Hybrid Search
- [ ] **Spec**: Queries use both dense and sparse retrieval
- [ ] **Test**:
  - Exact keyword match ranks higher with BM25
  - Semantic match ranks higher with dense
  - Combined outperforms either alone
- [ ] **Impl**: Connect `BM25Index` to `HybridRagPipeline`
- [ ] **Files**:
  - `stolas/src/rag.rs` (update `HybridRagPipeline`)
  - `stolas/src/bm25.rs` (verify integration)

```rust
// Target API
let pipeline = HybridRagPipeline::builder()
    .dense_weight(0.7)
    .sparse_weight(0.3)
    .fusion_method(FusionMethod::ReciprocalRankFusion)
    .build();

let results = pipeline.search("query", k=10).await?;
```

#### 3.4 Reciprocal Rank Fusion
- [ ] **Spec**: Combine rankings from multiple retrievers
- [ ] **Test**: Known rankings produce expected fused order
- [ ] **Impl**: Implement RRF algorithm
- [ ] **File**: `stolas/src/bm25.rs` (`reciprocal_rank_fusion()`)

```rust
// RRF formula: score = Σ 1/(k + rank_i)
fn reciprocal_rank_fusion(
    rankings: &[Vec<SearchResult>],
    k: f32,  // typically 60
) -> Vec<SearchResult>
```

#### 3.5 Cross-Encoder Reranking
- [ ] **Spec**: Top-k results reranked by cross-encoder
- [ ] **Test**: Reranking improves nDCG@10 on test set
- [ ] **Impl**: Load small cross-encoder, score pairs
- [ ] **Files**:
  - `stolas/src/rerank.rs` (`CandleCrossEncoder`)
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

#### 3.6 MMR Diversity Reranking
- [ ] **Spec**: Results are relevant AND diverse
- [ ] **Test**: Top-5 results have < 0.8 pairwise similarity
- [ ] **Impl**: Implement Maximal Marginal Relevance
- [ ] **File**: `stolas/src/rerank.rs` (`MMRReranker`)

### Foundation for Next Phase

#### 3.7 Benchmark Suite for RAG
- [ ] **Spec**: Automated quality metrics for retrieval
- [ ] **Test**: Benchmark runs in CI
- [ ] **Impl**: Create eval harness with test queries
- [ ] **File**: `stolas/benches/retrieval_quality.rs`

---

### 🔍 Checkpoint Review 3

**Focus Areas:**
- [ ] **Retrieval Quality**: nDCG, MRR, Recall@k metrics
- [ ] **Latency**: End-to-end search < 200ms?
- [ ] **Index Size**: Reasonable memory footprint?
- [ ] **Edge Cases**: Empty queries, special characters, very long docs

#### Quality Gates
```
Minimum thresholds (on held-out test set):
- Recall@10 > 0.85
- MRR > 0.70
- p95 latency < 500ms
```

---

## Phase 4: Inference Speed (Week 7-8)

**Theme:** Significant performance improvements through batching and speculation.

### Quick Wins

#### 4.1 KV Cache Clear on Generation Complete
- [ ] **Spec**: GPU memory freed after each generation
- [ ] **Test**: Monitor VRAM, verify drop after generation
- [ ] **Impl**: Call `clear_and_sync()` in finally block
- [ ] **File**: `abaddon/src/engine.rs`

#### 4.2 Request Priority Headers
- [ ] **Spec**: Clients can mark requests as high/low priority
- [ ] **Test**: High priority request processed before queued low priority
- [ ] **Impl**: Parse `X-Priority` header, use in scheduler
- [ ] **File**: `infernum-server/src/handlers.rs`

### Moderate Tasks

#### 4.3 Continuous Batching Integration
- [ ] **Spec**: Multiple requests processed in single forward pass
- [ ] **Test**:
  - 4 concurrent requests complete faster than sequential
  - Throughput scales with batch size (up to memory limit)
- [ ] **Impl**: Wire `BatchScheduler` to `Engine`
- [ ] **Files**:
  - `malphas/src/lib.rs` (update `process_batch`)
  - `abaddon/src/engine.rs` (add `generate_batch`)
  - `abaddon/src/batch.rs` (batch management)

```rust
// Target: batch forward pass
impl Engine {
    pub async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerateResponse>> {
        // Pad sequences to same length
        // Single forward pass
        // Split outputs
    }
}
```

#### 4.4 Speculative Decoding
- [ ] **Spec**: Draft model proposes, main model verifies
- [ ] **Test**:
  - Acceptance rate > 70% on typical prompts
  - Speedup > 1.5x vs standard decoding
- [ ] **Impl**: Complete `SpeculativeDecoder`
- [ ] **Files**:
  - `abaddon/src/speculative.rs`
  - Add draft model loading to `Engine`

```rust
pub struct SpeculativeDecoder {
    main_model: Arc<Engine>,
    draft_model: Arc<Engine>,  // Smaller, faster model
    num_speculative_tokens: usize,  // typically 4-8
}

impl SpeculativeDecoder {
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        loop {
            // 1. Draft model generates K tokens
            let draft_tokens = self.draft_model.generate_n(K).await?;

            // 2. Main model verifies in single forward pass
            let verified = self.main_model.verify(draft_tokens).await?;

            // 3. Accept matching prefix, reject divergent suffix
            // 4. Continue from first rejection point
        }
    }
}
```

#### 4.5 Prefix Caching
- [ ] **Spec**: Common prompt prefixes share KV cache
- [ ] **Test**: Same system prompt with different user messages reuses cache
- [ ] **Impl**: Hash prompt prefix, cache KV states
- [ ] **File**: `abaddon/src/kv_cache.rs`

### Foundation for Next Phase

#### 4.6 Performance Benchmark Suite
- [ ] **Spec**: Automated throughput and latency benchmarks
- [ ] **Test**: Benchmarks run in CI, alert on regression
- [ ] **Impl**: Criterion benchmarks for hot paths
- [ ] **File**: `abaddon/benches/inference.rs`

---

### 🔍 Checkpoint Review 4

**Focus Areas:**
- [ ] **Correctness**: Batched outputs identical to sequential?
- [ ] **Speculation Accuracy**: Are we accepting enough tokens?
- [ ] **Memory Safety**: No leaks under sustained load?
- [ ] **Regression**: Baseline metrics still met?

#### Performance Gates
```
vs. baseline (single request, no optimization):
- Throughput: > 2x improvement at batch=4
- Speculative speedup: > 1.5x on conversational prompts
- Memory: < 10% overhead for batching
```

---

## Phase 5: Production Hardening (Week 9-10)

**Theme:** Security, observability, and operational readiness.

### Quick Wins

#### 5.1 Request ID Propagation
- [ ] **Spec**: All logs include request ID for tracing
- [ ] **Test**: Parse logs, verify request ID present
- [ ] **Impl**: Add request ID to tracing span
- [ ] **File**: `infernum-server/src/handlers.rs`

#### 5.2 Health Check Depth Levels
- [ ] **Spec**: `/health?depth=full` checks model, GPU, memory
- [ ] **Test**: Each depth level returns appropriate checks
- [ ] **Impl**: Parameterized health endpoint
- [ ] **File**: `infernum-server/src/handlers.rs`

### Moderate Tasks

#### 5.3 Semantic Response Caching
- [ ] **Spec**: Similar queries return cached responses
- [ ] **Test**:
  - "What is Python?" and "Tell me about Python" hit same cache
  - Cache invalidation works
- [ ] **Impl**: Embed queries, similarity threshold for cache hit
- [ ] **Files**:
  - `infernum-server/src/cache.rs` (extend with embeddings)
  - Integrate with Stolas for vector similarity

#### 5.4 Structured Output Validation
- [ ] **Spec**: JSON schema enforced on generation
- [ ] **Test**: Invalid JSON retried until valid or max attempts
- [ ] **Impl**: Post-generation validation with retry
- [ ] **File**: `infernum-server/src/structured.rs`

#### 5.5 Graceful Shutdown
- [ ] **Spec**: In-flight requests complete before shutdown
- [ ] **Test**: Send SIGTERM during request, verify completion
- [ ] **Impl**: Shutdown signal handler with drain timeout
- [ ] **File**: `infernum-server/src/server.rs`

#### 5.6 Rate Limit by Token Usage
- [ ] **Spec**: Rate limits based on tokens consumed, not just requests
- [ ] **Test**: High-token request consumes more quota
- [ ] **Impl**: Track token usage per API key
- [ ] **File**: `infernum-server/src/security.rs`

### Foundation for Next Phase

#### 5.7 Telemetry Dashboard Queries
- [ ] **Spec**: Grafana queries for key metrics
- [ ] **Test**: Queries return data in test environment
- [ ] **Impl**: Document PromQL queries for SLIs
- [ ] **File**: `docs/observability/grafana-queries.md`

---

### 🔍 Checkpoint Review 5

**Focus Areas:**
- [ ] **Security Audit**: Auth bypass attempts blocked?
- [ ] **Graceful Degradation**: What happens at 100% capacity?
- [ ] **Observability**: Can we diagnose production issues?
- [ ] **Documentation**: Runbook for common issues?

#### Operational Readiness
```
- [ ] Runbook exists for top 5 error scenarios
- [ ] Alerts configured for SLO violations
- [ ] Capacity planning documented
- [ ] Incident response process defined
```

---

## Phase 6: Advanced Capabilities (Week 11-12)

**Theme:** Next-generation features that differentiate Infernum.

### Moderate Tasks

#### 6.1 LoRA Hot-Swapping
- [ ] **Spec**: Switch LoRA adapters without reloading base model
- [ ] **Test**:
  - Load base model
  - Apply LoRA A, generate
  - Apply LoRA B, generate (no reload)
- [ ] **Impl**: Runtime merge/unmerge of LoRA weights
- [ ] **File**: `asmodeus/src/lora.rs`

#### 6.2 GGUF Quantized Inference
- [ ] **Spec**: Load and run Q4_0, Q8_0 quantized models
- [ ] **Test**: GGUF model produces coherent output
- [ ] **Impl**: Dequantization kernels for common formats
- [ ] **File**: `abaddon/src/gguf.rs`

#### 6.3 Vision Input Support
- [ ] **Spec**: Accept images in chat completions
- [ ] **Test**: Describe image returns relevant description
- [ ] **Impl**: Image encoding, vision transformer integration
- [ ] **Files**:
  - `abaddon/src/models/vision.rs` (new)
  - `infernum-server/src/vision.rs` (already exists, wire up)

#### 6.4 Tool Use via Function Calling
- [ ] **Spec**: Model can request tool execution, receive results
- [ ] **Test**: Math question triggers calculator, returns correct answer
- [ ] **Impl**: Function calling protocol, tool execution loop
- [ ] **Files**:
  - `infernum-server/src/handlers.rs` (function call handling)
  - Integration with Beleth tools

### Exploratory

#### 6.5 Self-Improvement Pipeline
- [ ] **Spec**: Agent generates tests for untested code
- [ ] **Test**: Generated tests are valid and improve coverage
- [ ] **Impl**: Connect Beleth agent to Paimon tracking
- [ ] **Files**: Multiple, orchestration layer

---

### 🔍 Final Checkpoint Review

**Comprehensive Review:**
- [ ] All phases complete with passing tests
- [ ] Documentation updated for new features
- [ ] Performance benchmarks show improvement
- [ ] Security review passed
- [ ] Deployment tested in staging environment

---

## Appendix A: Test Categories

### Unit Tests
- Pure functions, no I/O
- Mock all dependencies
- Fast (< 1ms each)

### Integration Tests
- Test component interactions
- Use mock engine for speed
- Medium speed (< 1s each)

### End-to-End Tests
- Full server with real model
- Slower, run nightly
- Smoke tests only in CI

### Property Tests
- Use `proptest` for invariant testing
- Sampling behavior, tokenization, etc.

---

## Appendix B: Code Review Checklist

```markdown
## PR Review Template

### Spec Alignment
- [ ] Implementation matches original spec
- [ ] Edge cases from spec are handled
- [ ] No scope creep

### Test Quality
- [ ] Tests are deterministic
- [ ] Tests cover happy path and errors
- [ ] Tests are readable (good names, comments)
- [ ] No flaky tests

### Code Quality
- [ ] No `.unwrap()` in library code
- [ ] Errors include context
- [ ] Public APIs documented
- [ ] No TODO without issue link

### Performance
- [ ] No obvious O(n²) or worse
- [ ] Allocations minimized in hot paths
- [ ] Async code doesn't block

### Security
- [ ] Input validated
- [ ] No path traversal
- [ ] Secrets not logged
```

---

## Appendix C: Metrics & SLIs

### Latency
- `infernum_ttft_seconds` - Time to first token
- `infernum_generation_seconds` - Total generation time
- `infernum_tokens_per_second` - Throughput

### Availability
- `infernum_requests_total{status="success|error"}`
- `infernum_health_check_status`

### Quality
- `infernum_cache_hit_ratio`
- `infernum_speculation_acceptance_rate`
- `infernum_batch_utilization`

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-24 | Initial roadmap |
