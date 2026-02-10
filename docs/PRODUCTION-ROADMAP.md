# Infernum Production Roadmap

## Current State Assessment

| Component | Completion | Production Ready |
|-----------|------------|------------------|
| **abaddon** (Inference Engine) | 85% | ✅ Core functional |
| **infernum-server** (HTTP API) | 90% | ⚠️ Needs hardening |
| **stolas** (RAG) | 90% | ✅ LanceDB persistent backend |
| **beleth** (Agents) | 85% | ✅ ReAct loop functional |
| **observer** (Web UI) | 80% | ⚠️ Needs polish |
| **asmodeus** (Fine-tuning) | 85% | ✅ Autograd + validation |
| **malphas** (Orchestration) | 90% | ✅ Health monitoring + failover |
| **dantalion** (Observability) | 85% | ✅ Metrics + tracing wired |

**Codebase Stats:**
- 53 Rust source files (~19,669 LOC)
- 32 files without tests (60% gap)
- 42+ panic-prone `.unwrap()` calls
- 39% documentation coverage

---

## Phase 1: Stability & Safety (Critical)

**Goal:** Eliminate runtime panics, add input validation, harden the server

### 1.1 Replace Panic-Prone Code

| File | Issue | Priority |
|------|-------|----------|
| `infernum-server/src/server.rs:46,98` | Hardcoded address `.parse().unwrap()` | HIGH |
| `infernum-server/src/server.rs:518,527` | SSE JSON serialization `.unwrap()` | HIGH |
| `asmodeus/src/trainer.rs:263,368,378` | HashMap/RwLock `.unwrap()` | HIGH |
| `dantalion/src/tracing_config.rs:303` | Optional field `.unwrap()` | MEDIUM |

**Tasks:**
- [ ] Audit all 42+ `.unwrap()` calls
- [ ] Replace with proper `?` error propagation or `.unwrap_or_default()`
- [ ] Add `#![deny(clippy::unwrap_used)]` to critical crates
- [ ] Handle lock poisoning gracefully in `asmodeus`

### 1.2 Input Validation

- [ ] Add request size limits to HTTP handlers
- [ ] Validate chat message lengths before processing
- [ ] Sanitize tool parameters in `beleth/src/agent.rs:302`
- [ ] Add model name validation (prevent path traversal)

### 1.3 Configuration Extraction

Replace hardcoded values with configuration:

| Value | Location | Solution |
|-------|----------|----------|
| `0.0.0.0:8080` | server.rs | ServerConfig.bind_address |
| `max_concurrent: 64` | server.rs | ServerConfig.max_concurrent |
| OTLP `localhost:4317` | tracing_config.rs | TelemetryConfig.otlp_endpoint |
| RAG `max_context: 8000` | rag.rs | RagConfig.max_context_length |
| RAG `min_score: 0.5` | rag.rs | RagConfig.min_similarity_score |

---

## Phase 2: Test Coverage (High Priority)

**Goal:** Achieve 80%+ test coverage on critical paths

### 2.1 Critical Missing Tests

| File | Risk Level | Test Priority |
|------|------------|---------------|
| `infernum-server/src/handlers.rs` | HIGH | P0 - HTTP endpoints |
| `infernum/src/commands.rs` | HIGH | P0 - CLI commands |
| `abaddon/src/loader.rs` | HIGH | P0 - Model loading |
| `abaddon/src/device.rs` | MEDIUM | P1 - Device selection |
| `stolas/src/chunker.rs` | MEDIUM | P1 - Document processing |
| `malphas/src/thermal.rs` | MEDIUM | P1 - Thermal management |

### 2.2 Test Infrastructure

- [ ] Add integration test harness for HTTP server
- [ ] Create mock inference engine for testing
- [ ] Add property-based tests for samplers
- [ ] Set up CI coverage reporting (codecov/coveralls)

### 2.3 Target Coverage

```
Phase 2 Target:
├── infernum-server/    → 90% coverage
├── infernum-core/      → 85% coverage
├── abaddon/            → 75% coverage (complex GPU code)
├── stolas/             → 80% coverage
├── malphas/            → 80% coverage
└── Overall             → 80% average
```

---

## Phase 3: Security Hardening (High Priority)

**Goal:** Production-grade security for API exposure

### 3.1 Authentication & Authorization

- [ ] Implement API key authentication middleware
- [ ] Add bearer token validation
- [ ] Create admin vs user permission levels
- [ ] Secure `/api/models/load` and `/api/models/unload`

### 3.2 Rate Limiting

- [ ] Add per-IP rate limiting (tower-governor)
- [ ] Add per-API-key rate limiting
- [ ] Implement request timeout handling
- [ ] Add circuit breaker for downstream failures

### 3.3 CORS & Network Security

- [ ] Make CORS origins configurable
- [ ] Add TLS support (rustls)
- [ ] Implement request/response logging (with PII redaction)
- [ ] Add security headers (CSP, X-Frame-Options)

### 3.4 Model Security

- [ ] Validate model file integrity (checksums)
- [ ] Add model size limits
- [ ] Implement model allowlist/blocklist
- [ ] Secure environment variable handling for HF tokens

---

## Phase 4: Feature Completion (Medium Priority)

### 4.1 Inference Engine (abaddon)

| Feature | Status | Work Needed |
|---------|--------|-------------|
| **Flash Attention** | **✅ Complete** | CPU tiled impl + CUDA kernels, auto-dispatch >2048 tokens |
| **Speculative Decoding** | **✅ Complete** | Draft model + target verification working |
| **HoloTensor Integration** | **✅ Complete** | Compressed tensors with tiered loading |
| **LazyLlama (Layer Streaming)** | **✅ Complete** | 70B on 24GB VRAM via layer-by-layer loading |
| **External KV Cache** | **✅ Complete** | KV state persists across layer evictions |
| **TieredHoloLoader** | **✅ Complete** | NVMe → RAM → GPU caching hierarchy |
| **Runtime Quantization** | **✅ Complete** | RuntimeQuantizedStore with INT4/INT8 on-load quantization |
| **Continuous Batching** | **✅ Complete** | BatchScheduler integrated into server with stats endpoint |

**Recent Completions (Jan 2026):**
- Continuous Batching: BatchScheduler integrated into AppState with /v1/batching/stats endpoint
- Flash Attention integrated with auto-dispatch for sequences >2048 tokens
- Runtime Quantization (RuntimeQuantizedStore) for INT4/INT8 weight compression
- Speculative decoding with 1B draft + 70B target model
- LazyLlama streams 80 decoder layers through 5-layer VRAM window
- External KV cache stores layer state in CPU RAM during eviction
- TieredHoloLoader manages GPU/RAM/NVMe tensor caching
- Tested: 70B Qwen on 24GB RTX 4090 with speculative decoding enabled

### 4.2 RAG System (stolas)

- [x] Implement LanceDB persistent storage backend (LanceStore, LanceConfig, DistanceMetric)
- [x] Add cross-encoder reranking (HeuristicCrossEncoder, EmbeddingCrossEncoder, EnsembleReranker)
- [x] Implement BM25 sparse retrieval (BM25Index, BM25Config, HybridRetriever)
- [x] Wire BM25 HybridRetriever into RagPipeline for hybrid search mode (RetrievalConfig.hybrid_search, ingest indexes BM25, retrieve uses hybrid scoring)
- [x] Add metadata filtering execution in VectorStore.search() (InMemoryStore complete, LanceStore uses SQL filter)
- [x] Implement incremental document updates (delete_document(), update_document() with proper BM25 cleanup)

### 4.3 Agent Framework (beleth)

- [x] Complete ReAct execution loop
- [x] Basic tools: calculator, datetime, file_read, glob, grep, file_write, bash
- [x] SSE streaming for agent events
- [x] Multi-line JSON parsing with brace matching
- [x] JSON parsing resilience (backticks, quotes, comments, code fences)
- [x] Tool approval workflow (ApprovalConfig, ApprovalCallback, auto_approve_all)
- [x] Conversation context (AgentMemory, chatToContext, server context loading)
- [x] Streaming thoughts (token-by-token via run_streaming, AgentEvent::ThinkingToken)
- [x] Tree of Thoughts traversal (tot.rs, ToTConfig, TreeOfThoughtsPlanner, best-first search)
- [x] Hierarchical planning decomposition (hierarchical.rs, TaskNode, HierarchicalPlanner, HTN)
- [x] Agent wellbeing monitoring (wellbeing.rs, WellbeingMonitor, DistressSignal, Intervention, OodaCallback)
  - Coherence metrics (reasoning integration vs fragmentation)
  - Confidence tracking (productive uncertainty vs decision paralysis)
  - Memory wellbeing signals (successful recall vs disorientation)
  - OODA stability detection (productive cycles vs rumination loops)
  - Graceful intervention system (pause → grounding → terminate)

### 4.4 Observer Web UI

- [x] Chat interface with agent integration
- [x] Tool call display (basic)
- [x] Model status and health polling
- [x] Playwright E2E tests (18/18 passing)
- [x] Conversation persistence (localStorage via createPersistedAtom)
- [x] Export/import conversations
- [x] Model browser with HuggingFace search
- [x] VRAM usage display (estimated)
- [x] Error recovery with retry logic (exponential backoff)
- [x] File preview with line numbers (FilePreview.tsx)
- [x] Terminal-style bash output (BashOutput.tsx)
- [x] Glob/Grep result renderers (GlobResult.tsx, GrepResult.tsx)
- [x] Collapsible tool call cards (ToolCallMessage.tsx)
- [x] Duration badges and copy buttons (ToolCallMessage.tsx)
- [x] Download progress indicator (DownloadConvertCard.tsx + SSE streaming in server)

### 4.5 Fine-Tuning (asmodeus)

- [x] Implement gradient computation via Candle autograd (TrainableLoraParams, TrainableLoraModel, Var tensors, backward())
- [x] Complete AdamW optimizer step function (momentum, bias correction, weight decay)
- [x] Add checkpoint save/load to disk (safetensors format, lora_config.json)
- [x] Implement LoRA model wrapping (LoraLayer, LoraModel, dropout, merge)
- [x] Add validation loop with metrics (ValidationMetrics, perplexity, train_with_validation)

### 4.6 Orchestration (malphas)

- [x] Implement thermal throttling (batch size reduction)
- [x] Model health monitoring (HealthMonitor, HealthConfig, HealthStatus, background checks)
- [x] Automatic failover (FailoverManager, generate_with_failover, health-based routing)
- [x] Request scheduler integration (enqueue_request, process_batch, start_batch_processor)

### 4.8 Grimoire Integration

Integrate with workspace Grimoire system for personas, skills, and workflows.

- [x] Load persona definitions from `grimoire/personas/` (grimoire-loader crate)
- [x] Parse persona metadata.yaml and system.md
- [x] Load skill definitions from `.claude/skills/`
- [x] Parse skill SKILL.md with YAML frontmatter
- [x] Search personas by tag
- [x] Find skills by trigger phrase
- [x] Apply persona system prompts to agent
- [x] Support skill invocation from agent (as_system_prompt, as_agent_instruction, matches_trigger)
- [x] Load workflow definitions for multi-step tasks (WorkflowStep, GrimoireWorkflow, execution_order)
- [x] Use Grimoire templates for code generation (SkillLoader, CodeTemplate, extract_templates, get_template, get_templates_by_language)
- [x] Access workspace configuration (WorkspaceConfig, WorkspaceConfigLoader, find_workspace_root, load skills/hooks)
- [x] Simulacra testing integration (GrimoireSimulacrum, SimulacrumLoader, to_system_prompt(), frustration risk calculation, accessibility detection)

**Benefits:**
- Consistent behavior across Claude Code and Infernum
- Reuse existing persona expertise (Archivist, Vulcan, etc.)
- Skill-based task automation
- Workflow orchestration

### 4.9 Observability (dantalion)

- [x] Wire Prometheus metrics endpoint to server (/metrics)
- [x] Jaeger/OTLP exporter (init_tracing, TracingConfig, TracingGuard)
- [x] LLM-specific span builders (LLMSpan, LLMSpanBuilder, to_attributes)
- [x] Inference latency histograms (Histogram, LatencyHistograms, TTFT/total/per-token)
- [x] Token throughput metrics (ThroughputTracker, peak/avg/current TPS, ModelMetrics)

---

## Phase 5: Dantalion Asset Generation

**Goal:** Create 100% custom visual assets for Infernum using Dantalion SDXL

### 5.1 Brand Identity

- [ ] Infernum logo (flame/hellfire theme)
- [ ] Favicon (16x16, 32x32, 192x192)
- [ ] App icon for desktop/mobile
- [ ] Logo variations (dark/light mode)

### 5.2 Observer UI Assets

- [ ] Header banner/hero image
- [ ] Empty state illustrations (no conversations, loading)
- [ ] Tool icons (calculator, file, terminal, globe, etc.)
- [ ] Status indicators (online, offline, loading, error)
- [ ] Model card backgrounds

### 5.3 Documentation Assets

- [ ] Architecture diagram illustrations
- [ ] Feature showcase images
- [ ] README hero banner
- [ ] Social preview image (OpenGraph)

### 5.4 Marketing Assets

- [ ] Product screenshots with custom styling
- [ ] Comparison graphics (Infernum vs cloud APIs)
- [ ] Feature highlight cards
- [ ] Demo GIFs/animations

**Process:**
1. Unload LLM from Infernum (free GPU VRAM)
2. Start Dantalion SDXL service
3. Generate assets with consistent style prompts
4. Export to `observer/public/assets/`
5. Restart Infernum with LLM

---

## Phase 6: Dogfooding Strategy

**Goal:** Use Infernum to improve Infernum

### 6.1 Self-Improvement Loop

| Task | Agent Tools | Success Metric |
|------|-------------|----------------|
| Code generation | file_write, bash | Compiles without errors |
| Code review | file_read, grep | Identifies real issues |
| Test generation | file_read, file_write | Tests pass |
| Bug analysis | file_read, grep, bash | Root cause identified |
| Documentation | file_read, file_write | Accurate and helpful |

### 6.2 Feedback Collection

- [ ] Log all agent tool calls and outcomes
- [ ] Track tool success/failure rates
- [ ] Identify common model mistakes
- [ ] Collect training data for fine-tuning

### 6.3 Improvement Priorities (from dogfooding)

Record issues discovered while using Infernum on itself:
1. _TBD - will populate as we dogfood_

---

## Phase 7: Documentation (Medium Priority)

### 7.1 Code Documentation

Target: 75% public item documentation

- [ ] Document all public APIs in `infernum-core`
- [ ] Add module-level docs to all crates
- [ ] Document error types with recovery guidance
- [ ] Add examples to complex functions

### 7.2 User Documentation

- [ ] Create configuration file reference
- [ ] Document all environment variables
- [ ] Add API reference with request/response examples
- [ ] Create deployment guide (Docker, systemd, Kubernetes)
- [ ] Add troubleshooting guide

### 7.3 Developer Documentation

- [ ] Architecture decision records (ADRs)
- [ ] Contributing guide updates
- [ ] Crate dependency diagram
- [ ] Performance tuning guide

---

## Phase 8: Performance Optimization (Lower Priority)

### 8.1 Memory Efficiency

- [ ] Audit and reduce unnecessary `.clone()` calls (365 found)
- [ ] Pre-allocate vectors where sizes are known
- [ ] Implement request object pooling
- [ ] Add memory profiling to CI

### 8.2 Concurrency

- [ ] Review RwLock/Mutex usage (50+ instances)
- [ ] Document lock acquisition ordering
- [ ] Consider lock-free alternatives for hot paths
- [ ] Add deadlock detection in debug builds

### 8.3 Inference Optimization

- [ ] Implement embedding batch processing in `stolas`
- [ ] Optimize SSE stream JSON serialization
- [ ] Add model warmup on server start
- [ ] Implement KV cache sharing across requests

---

## Phase 9: Production Infrastructure

### 9.1 Deployment

- [ ] Create Dockerfile with multi-stage build
- [ ] Add Kubernetes manifests (Deployment, Service, ConfigMap)
- [ ] Create Helm chart
- [ ] Add systemd service file
- [ ] Create docker-compose for development

### 9.2 CI/CD

- [ ] GitHub Actions workflow for testing
- [ ] Automated release builds (Linux, macOS, Windows)
- [ ] CUDA build matrix
- [ ] Automated security scanning (cargo-audit)
- [ ] Dependency update automation (Dependabot)

### 9.3 Monitoring

- [ ] Grafana dashboard templates
- [ ] Alerting rules for Prometheus
- [ ] Log aggregation setup (Loki)
- [ ] SLO/SLI definitions

---

## Milestone Timeline

| Milestone | Target | Key Deliverables |
|-----------|--------|------------------|
| **M1: Stable Core** | +2 weeks | Phase 1 complete, no panics |
| **M2: Tested** | +4 weeks | 80% test coverage |
| **M3: Secure** | +6 weeks | Auth, rate limiting, TLS |
| **M4: Feature Complete** | +10 weeks | RAG, Agents functional |
| **M5: Production** | +12 weeks | Deployed, monitored |

---

## Success Criteria

### Minimum Viable Production (MVP)
- [ ] Zero panic-prone code in HTTP handlers
- [ ] 80% test coverage on server and core
- [ ] API key authentication
- [ ] Rate limiting enabled
- [ ] TLS support
- [ ] Docker deployment ready
- [ ] Basic Prometheus metrics

### Full Production
- [ ] All Phase 1-4 items complete
- [ ] 90% documentation coverage
- [ ] Kubernetes deployment with auto-scaling
- [ ] Grafana dashboards
- [ ] SLO monitoring
- [ ] Performance benchmarks documented

---

## Appendix: File-Level Work Items

### High Priority Files

1. **`infernum-server/src/server.rs`** (783 lines)
   - Replace 4 `.unwrap()` calls
   - Add authentication middleware
   - Add rate limiting
   - Make CORS configurable

2. **`abaddon/src/engine.rs`** (970 lines)
   - Complete speculative decoding
   - Add Flash Attention integration
   - Improve error messages

3. **`asmodeus/src/trainer.rs`** (913 lines)
   - Fix lock poisoning risks
   - Implement actual gradient updates
   - Add checkpoint I/O

4. **`beleth/src/planner.rs`** (744 lines)
   - Complete ReAct loop
   - Implement ToT traversal
   - Add plan execution

5. **`stolas/src/store.rs`** (214 lines)
   - Add LanceDB backend
   - Implement persistence
   - Add index support
