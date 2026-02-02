# Infernum Server Enhancement Roadmap

This document outlines a phased approach to implementing production-ready features for infernum-server. Each phase interleaves high-complexity features with quick wins to maintain development momentum.

## Development Methodology

### TDD Requirements
Every feature MUST follow Test-Driven Development:
1. **Red**: Write failing tests first
2. **Green**: Implement minimal code to pass
3. **Refactor**: Clean up while keeping tests green

### Documentation Requirements
- Update module-level rustdoc before merging
- Add OpenAPI annotations for new endpoints
- Update CHANGELOG.md with each feature
- Add integration tests for API changes

### Definition of Done
- [ ] Unit tests passing (>80% coverage for new code)
- [ ] Integration tests for API endpoints
- [ ] Benchmark added for performance-critical paths
- [ ] Documentation updated
- [ ] No new clippy warnings
- [ ] Reviewed and approved

---

## Phase 1: API Parity (Weeks 1-3)

**Goal**: Achieve feature parity with OpenAI API essentials.

### 1.1 Token Counting Endpoint [LOW]
**Complexity**: Low | **Priority**: High | **Est**: 2 days

Add `/v1/tokenize` endpoint for pre-flight token estimation.

```rust
// Request
POST /v1/tokenize
{ "model": "llama-3b", "messages": [...] }

// Response
{ "token_count": 150, "model": "llama-3b" }
```

**TDD Steps**:
1. Write tests for tokenize request/response types
2. Write handler tests with mock tokenizer
3. Implement types in `openai.rs`
4. Implement handler in `handlers.rs`
5. Add OpenAPI annotations

**Files**:
- `src/openai.rs` - TokenizeRequest, TokenizeResponse
- `src/handlers.rs` - tokenize_handler
- `tests/tokenize_tests.rs`

---

### 1.2 Logprobs Support [LOW]
**Complexity**: Low | **Priority**: Medium | **Est**: 1 day

Enable `logprobs` parameter in chat completions.

**TDD Steps**:
1. Add tests for logprobs in ChatCompletionRequest
2. Add tests for TopLogProb in response
3. Extend existing types
4. Update serialization tests

**Files**:
- `src/openai.rs` - Add logprobs field, TopLogProb type

---

### 1.3 Function Calling / Tool Use [HIGH]
**Complexity**: High | **Priority**: High | **Est**: 5 days

Add OpenAI-compatible function calling.

```rust
// New types needed
pub struct Tool {
    pub r#type: String,  // "function"
    pub function: FunctionDefinition,
}

pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,  // JSON Schema
}

pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}
```

**TDD Steps**:
1. Write comprehensive type tests (serialization round-trip)
2. Write request validation tests (invalid tool definitions)
3. Write response generation tests
4. Implement Tool, FunctionDefinition, ToolCall types
5. Extend ChatCompletionRequest with tools field
6. Extend ChatMessage with tool_calls field
7. Add tool_choice parameter
8. Integration tests with mock inference

**Files**:
- `src/openai.rs` - Tool types
- `src/tool_use.rs` - NEW: Tool validation, parsing
- `tests/tool_use_tests.rs`

---

### 1.4 Request Priority Header [LOW]
**Complexity**: Low | **Priority**: Medium | **Est**: 1 day

Add `X-Priority` header support for queue ordering.

```rust
pub enum RequestPriority {
    High,
    Normal,
    Low,
    Background,
}
```

**TDD Steps**:
1. Write header extraction tests
2. Write priority parsing tests
3. Implement extractor middleware
4. Add to request context

**Files**:
- `src/priority.rs` - NEW
- `src/server.rs` - Add to middleware stack

---

## Phase 2: Operational Excellence (Weeks 4-6)

**Goal**: Production-ready reliability and observability.

### 2.1 Circuit Breaker [MEDIUM]
**Complexity**: Medium | **Priority**: High | **Est**: 3 days

Implement circuit breaker for inference backend.

```rust
pub struct CircuitBreaker {
    state: AtomicU8,  // Closed, Open, HalfOpen
    failure_count: AtomicU32,
    last_failure: AtomicU64,
    config: CircuitBreakerConfig,
}

pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub reset_timeout: Duration,
    pub half_open_requests: u32,
}
```

**TDD Steps**:
1. Write state transition tests
2. Write timeout behavior tests
3. Write concurrent access tests
4. Implement CircuitBreaker
5. Add Prometheus metrics
6. Integration with inference calls

**Files**:
- `src/circuit_breaker.rs` - NEW
- `src/observability.rs` - Add circuit breaker metrics

---

### 2.2 GPU Metrics [LOW]
**Complexity**: Low | **Priority**: Medium | **Est**: 2 days

Add GPU utilization to Prometheus metrics.

```
# HELP infernum_gpu_memory_used_bytes GPU memory used
# TYPE infernum_gpu_memory_used_bytes gauge
infernum_gpu_memory_used_bytes{gpu="0"} 8589934592

# HELP infernum_gpu_utilization GPU utilization percentage
# TYPE infernum_gpu_utilization gauge
infernum_gpu_utilization{gpu="0"} 0.85
```

**TDD Steps**:
1. Write mock GPU provider tests
2. Write metrics rendering tests
3. Implement GpuMetrics trait
4. Add NVML integration (optional feature)
5. Update Prometheus output

**Files**:
- `src/gpu_metrics.rs` - NEW
- `src/observability.rs` - Integrate GPU metrics
- `Cargo.toml` - Optional nvml feature

---

### 2.3 Request Cancellation [MEDIUM]
**Complexity**: Medium | **Priority**: Medium | **Est**: 3 days

Handle client disconnection gracefully.

**TDD Steps**:
1. Write cancellation token tests
2. Write handler abort tests
3. Implement CancellationToken wrapper
4. Add to request context
5. Propagate to inference layer
6. Add cancelled request metrics

**Files**:
- `src/cancellation.rs` - NEW
- `src/handlers.rs` - Check cancellation in streaming

---

### 2.4 Structured Error Codes [LOW]
**Complexity**: Low | **Priority**: Low | **Est**: 1 day

Add machine-readable error subcodes.

```json
{
  "error": {
    "code": "context_length_exceeded",
    "subcode": "input_too_long",
    "param": "messages",
    "limit": 8192,
    "actual": 12000
  }
}
```

**TDD Steps**:
1. Extend ErrorDetail tests
2. Add subcode field
3. Update error constructors

**Files**:
- `src/error_response.rs`

---

## Phase 3: Performance (Weeks 7-10)

**Goal**: Competitive throughput and latency.

### 3.1 Response Caching [MEDIUM]
**Complexity**: Medium | **Priority**: High | **Est**: 4 days

LRU cache for deterministic requests (temperature=0).

```rust
pub struct ResponseCache {
    cache: RwLock<LruCache<CacheKey, CachedResponse>>,
    config: CacheConfig,
    metrics: CacheMetrics,
}

pub struct CacheConfig {
    pub max_entries: usize,
    pub max_memory_bytes: usize,
    pub ttl: Duration,
    pub cacheable_temp_max: f32,  // Cache when temp <= this
}
```

**TDD Steps**:
1. Write cache key generation tests
2. Write LRU eviction tests
3. Write TTL expiration tests
4. Write cache hit/miss tests
5. Implement ResponseCache
6. Add cache headers (X-Cache: HIT/MISS)
7. Add Prometheus metrics
8. Benchmark cache overhead

**Files**:
- `src/cache.rs` - NEW
- `src/handlers.rs` - Cache integration
- `benches/cache_benchmarks.rs`

---

### 3.2 Admin Model API [LOW]
**Complexity**: Low | **Priority**: Medium | **Est**: 2 days

Runtime model management endpoints.

```
POST   /admin/models/load    { "model": "...", "options": {...} }
POST   /admin/models/unload  { "model": "..." }
GET    /admin/models/status
POST   /admin/models/warmup  { "model": "..." }
```

**TDD Steps**:
1. Write admin auth tests (require admin scope)
2. Write load/unload request tests
3. Implement handlers
4. Add OpenAPI docs

**Files**:
- `src/admin.rs` - NEW
- `src/handlers.rs` - Admin routes

---

### 3.3 Request Deduplication [MEDIUM]
**Complexity**: Medium | **Priority**: Medium | **Est**: 3 days

Coalesce identical in-flight requests.

```rust
pub struct RequestDeduplicator {
    in_flight: DashMap<RequestHash, Shared<InferenceResult>>,
}
```

**TDD Steps**:
1. Write request hashing tests
2. Write concurrent request tests
3. Write result sharing tests
4. Implement deduplicator
5. Add metrics (dedupe_hits, dedupe_misses)

**Files**:
- `src/dedup.rs` - NEW

---

### 3.4 Request Timeout Improvements [LOW]
**Complexity**: Low | **Priority**: Medium | **Est**: 1 day

Per-request timeout via header.

```
X-Request-Timeout: 30
```

**TDD Steps**:
1. Write header parsing tests
2. Write timeout behavior tests
3. Implement extractor
4. Add to handler context

**Files**:
- `src/timeout.rs` - NEW

---

## Phase 4: Advanced Features (Weeks 11-16)

**Goal**: Differentiated capabilities.

### 4.1 Continuous Batching [HIGH]
**Complexity**: High | **Priority**: High | **Est**: 10 days

Dynamic batching like vLLM for 2-5x throughput.

```rust
pub struct BatchScheduler {
    pending: SegQueue<PendingRequest>,
    active_batch: RwLock<ActiveBatch>,
    config: BatchConfig,
}

pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_waiting_time: Duration,
    pub max_tokens_per_batch: usize,
}
```

**TDD Steps**:
1. Write batch formation tests
2. Write scheduling policy tests
3. Write preemption tests
4. Write memory pressure tests
5. Implement BatchScheduler
6. Implement token-level iteration
7. Integration with KV cache
8. Comprehensive benchmarks
9. Load testing

**Files**:
- `src/batching/mod.rs` - NEW
- `src/batching/scheduler.rs`
- `src/batching/batch.rs`
- `src/batching/iteration.rs`
- `benches/batching_benchmarks.rs`

---

### 4.2 Structured Outputs [MEDIUM]
**Complexity**: Medium | **Priority**: Medium | **Est**: 4 days

JSON schema enforcement for responses.

```rust
// Request
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": { "type": "object", ... }
    }
  }
}
```

**TDD Steps**:
1. Write schema validation tests
2. Write constrained decoding tests
3. Implement ResponseFormat type
4. Implement schema-guided generation
5. Add validation middleware

**Files**:
- `src/structured.rs` - NEW
- `src/openai.rs` - ResponseFormat type

---

### 4.3 WebSocket Streaming [MEDIUM]
**Complexity**: Medium | **Priority**: Low | **Est**: 4 days

Lower-latency alternative to SSE.

**TDD Steps**:
1. Write WebSocket connection tests
2. Write message format tests
3. Write reconnection tests
4. Implement WebSocket handler
5. Add connection metrics

**Files**:
- `src/websocket.rs` - NEW

---

### 4.4 Vision/Multimodal [HIGH]
**Complexity**: High | **Priority**: Low | **Est**: 8 days

Image input support for vision models.

```rust
pub enum MessageContent {
    Text(String),
    ImageUrl { url: String, detail: Option<String> },
    ImageBase64 { data: String, media_type: String },
}
```

**TDD Steps**:
1. Write content type parsing tests
2. Write image validation tests
3. Write base64 decoding tests
4. Extend ChatMessage content type
5. Implement image preprocessing
6. Integration with vision models

**Files**:
- `src/vision.rs` - NEW
- `src/openai.rs` - Extend content types

---

## Phase 5: Enterprise (Weeks 17-20)

**Goal**: Enterprise-ready deployment features.

### 5.1 Request Queuing [MEDIUM]
**Complexity**: Medium | **Priority**: Medium | **Est**: 4 days

Priority-based request queue with fairness.

```rust
pub struct RequestQueue {
    queues: [SegQueue<QueuedRequest>; 4],  // By priority
    config: QueueConfig,
}

pub struct QueueConfig {
    pub max_queue_depth: usize,
    pub priority_weights: [f32; 4],
    pub starvation_timeout: Duration,
}
```

**TDD Steps**:
1. Write enqueue/dequeue tests
2. Write priority ordering tests
3. Write fairness/starvation tests
4. Write backpressure tests
5. Implement RequestQueue
6. Add queue depth metrics
7. Add queue wait time metrics

**Files**:
- `src/queue.rs` - NEW

---

### 5.2 gRPC API [MEDIUM]
**Complexity**: Medium | **Priority**: Low | **Est**: 5 days

gRPC interface for internal services.

**TDD Steps**:
1. Define protobuf schema
2. Generate Rust code
3. Write service tests
4. Implement gRPC handlers
5. Add reflection support

**Files**:
- `proto/infernum.proto` - NEW
- `src/grpc.rs` - NEW
- `build.rs` - Protobuf compilation

---

### 5.3 Speculative Decoding [HIGH]
**Complexity**: High | **Priority**: Low | **Est**: 8 days

Use draft model for 2-3x speedup.

**TDD Steps**:
1. Write draft generation tests
2. Write verification tests
3. Write token acceptance tests
4. Implement SpeculativeDecoder
5. Benchmark speedup

**Files**:
- `src/speculative.rs` - NEW

---

## Summary Timeline

```
Week 1-3:   Phase 1 - API Parity
Week 4-6:   Phase 2 - Operational Excellence
Week 7-10:  Phase 3 - Performance
Week 11-16: Phase 4 - Advanced Features
Week 17-20: Phase 5 - Enterprise
```

## Quick Reference

| Feature | Phase | Complexity | Days |
|---------|-------|------------|------|
| Token Counting | 1.1 | Low | 2 |
| Logprobs | 1.2 | Low | 1 |
| Function Calling | 1.3 | High | 5 |
| Priority Header | 1.4 | Low | 1 |
| Circuit Breaker | 2.1 | Medium | 3 |
| GPU Metrics | 2.2 | Low | 2 |
| Request Cancellation | 2.3 | Medium | 3 |
| Structured Error Codes | 2.4 | Low | 1 |
| Response Caching | 3.1 | Medium | 4 |
| Admin Model API | 3.2 | Low | 2 |
| Request Deduplication | 3.3 | Medium | 3 |
| Timeout Improvements | 3.4 | Low | 1 |
| Continuous Batching | 4.1 | High | 10 |
| Structured Outputs | 4.2 | Medium | 4 |
| WebSocket Streaming | 4.3 | Medium | 4 |
| Vision/Multimodal | 4.4 | High | 8 |
| Request Queuing | 5.1 | Medium | 4 |
| gRPC API | 5.2 | Medium | 5 |
| Speculative Decoding | 5.3 | High | 8 |

**Total Estimated Effort**: ~71 engineering days

## Metrics for Success

### Performance Targets
- Request overhead: <1ms p99
- Cache hit rate: >30% for production workloads
- Deduplication rate: >10% under load
- Batching efficiency: >80% GPU utilization

### Reliability Targets
- Circuit breaker recovery: <30s
- Request cancellation: <100ms
- Queue overflow handling: Graceful 503

### API Compatibility
- OpenAI SDK compatibility: 100%
- Function calling: Full support
- Streaming: SSE + WebSocket
