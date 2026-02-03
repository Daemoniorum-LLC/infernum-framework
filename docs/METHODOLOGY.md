# Infernum Rust/Sigil Comparison Methodology

## Objective

Maintain **feature-parallel implementations** of Infernum's core streaming infrastructure in both Rust and Sigil to:

1. Empirically validate Sigil's morpheme pipes for streaming workloads
2. Test evidentiality markers for LLM output trust boundaries
3. Generate comparative data for streaming pipeline expressiveness
4. Benchmark token streaming latency (target: <5ms first token)

## Directory Structure

```
persona-framework/
├── infernum/                   # Rust implementation (REFERENCE)
├── infernum-sigil/             # Sigil implementation (EXPERIMENTAL)
└── infernum-comparison/        # Testing & analysis harness
    ├── tests/                  # Shared test cases (JSON)
    ├── benchmarks/             # Performance comparisons
    ├── METHODOLOGY.md          # This document
    └── RESULTS.md              # Ongoing findings
```

## Pilot Module: infernum-core

**Rationale:**
- Token streaming is Sigil's sweet spot (morpheme pipes)
- LLM outputs are naturally `~` (reported/untrusted data)
- 1,582 LOC — manageable scope for rigorous comparison
- Defines interfaces for entire Infernum ecosystem
- Pure data types with minimal external dependencies

### Module Breakdown

| File | Rust LOC | Purpose | Sigil Benefit |
|------|----------|---------|---------------|
| types.rs | 222 | Core identifiers, enums | Inline defaults, pattern enums |
| streaming.rs | 215 | Token stream, chunks | Morpheme pipes (`\|tau`, `\|phi`) |
| sampling.rs | 192 | Generation parameters | Builder pattern, validation |
| request.rs | 241 | API request types | Evidentiality for inputs |
| response.rs | 152 | API response types | Evidentiality for outputs (`~`) |
| model.rs | 387 | Model metadata | Sum types, source enum |
| error.rs | 144 | Error handling | Result type, error enums |
| **Total** | **1,582** | | |

## Metrics Tracked

### Size Metrics

| Metric | Description | Tool |
|--------|-------------|------|
| Lines of Code | Non-blank, non-comment lines | `tokei` / `sigil-stats` |
| Character Count | Total characters (morpheme density) | `wc -c` |
| Token Count | LLM tokens (AI tooling cost) | `tiktoken` |
| File Count | Number of source files | `find` |

### Performance Metrics

| Metric | Description | Tool | Target |
|--------|-------------|------|--------|
| Stream First Chunk | Time to first token | `criterion` | <5ms |
| Chunk Throughput | Chunks processed/sec | `criterion` | >100k/s |
| Memory per Stream | RSS during streaming | `/usr/bin/time -v` | <1MB |
| Collect Latency | Time to collect all text | `criterion` | <1ms |

### Quality Metrics

| Metric | Description | Method |
|--------|-------------|--------|
| Test Pass Rate | % of shared tests passing | `cargo test` / `sigil test` |
| Type Safety | Compile-time error catch rate | Manual review |
| Error Clarity | Diagnostic message quality | Subjective 1-5 |
| Evidentiality Coverage | % of untrusted data marked `~` | Static analysis |

## Feature Parity Protocol

### API Equivalence

Both implementations MUST expose identical public interfaces:

```rust
// Rust
pub struct StreamChunk {
    pub request_id: RequestId,
    pub model: ModelId,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}
```

```sigil
// Sigil (equivalent)
struct StreamChunk {
    request_id: RequestId,
    model: ModelId,
    choices: [StreamChoice],
    usage?: Usage,
}
```

### Behavioral Equivalence

For any input `I`:
```
rust_impl(I) == sigil_impl(I)
```

Verified by shared test suite that runs against both implementations.

### Test Sharing

Tests are written in JSON format and executed against both:

```
infernum-comparison/tests/
├── streaming/
│   ├── chunk_creation.json
│   ├── collect_text.json
│   └── empty_stream.json
├── sampling/
│   ├── presets.json
│   ├── validation.json
│   └── builder.json
├── types/
│   ├── model_id.json
│   ├── request_id.json
│   └── usage.json
└── runner/
    ├── rust_runner.rs      # Executes against infernum/
    └── sigil_runner.sigil  # Executes against infernum-sigil/
```

## Porting Protocol

### Phase 1: Types First (Day 1-2)

1. Port all struct/enum definitions
2. Verify JSON serialization compatibility
3. No logic yet — just data shapes

**Files:** types.sigil, error.sigil

### Phase 2: Pure Functions (Day 3-4)

1. Port stateless transformation functions
2. Unit test each function in isolation
3. Compare outputs for identical inputs

**Files:** sampling.sigil (validation, presets)

### Phase 3: Streaming Core (Day 5-7)

1. Port StreamChunk, StreamChoice, StreamDelta
2. Implement morpheme-based collect operations
3. Integration test stream processing

**Files:** streaming.sigil, response.sigil

### Phase 4: Request/Response (Day 8-10)

1. Port GenerateRequest, EmbedRequest
2. Port GenerateResponse, TokenInfo
3. Add evidentiality markers

**Files:** request.sigil, response.sigil

### Phase 5: Model Metadata (Day 11-12)

1. Port ModelMetadata, ModelSource enum
2. Port architecture detection
3. Full integration testing

**Files:** model.sigil

### Phase 6: Integration & Benchmarks (Day 13-15)

1. Wire up full module
2. End-to-end streaming tests
3. Performance benchmarking
4. Documentation

## Evidentiality Mapping

| Rust Pattern | Sigil Equivalent | Meaning |
|--------------|------------------|---------|
| `T` (owned) | `T!` | Known/trusted value |
| `Option<T>` | `T?` | Uncertain/may not exist |
| External API response | `T~` | Reported/untrusted |
| `unsafe { }` | `unsafe { }` | Trust boundary |

### LLM-Specific Mappings

| Data Source | Evidentiality | Rationale |
|-------------|---------------|-----------|
| User prompt | `~` (reported) | External input, untrusted |
| Model output | `~` (reported) | LLM is external oracle |
| Token IDs | `!` (known) | Computed by tokenizer |
| Sampling params | `!` (known) | User-configured, validated |
| Usage stats | `!` (known) | Computed locally |

## Morpheme Pipe Opportunities

### Streaming Collection

```sigil
// Replace nested loops with pipes
fn collect_text(chunks: [StreamChunk]) -> str {
    chunks
    |tau{_.choices}          // Extract choices
    |flatten                  // Flatten nested
    |tau{_.delta.content}    // Get content
    |phi{_.is_some}          // Filter Some
    |tau{_.unwrap}           // Unwrap
    |join("")                 // Concatenate
}
```

### Validation Chains

```sigil
fn validate_sampling(params: SamplingParams) -> Result<SamplingParams!, str> {
    params|validate!{
        temperature: >= 0.0,
        top_p: in_range(0.0, 1.0),
        min_p: in_range(0.0, 1.0),
        max_tokens: > 0,
    }
}
```

## Reporting

Results are logged to `RESULTS.md` with:
- Date of measurement
- Commit hashes (both repos)
- Hardware/OS context
- Raw numbers + analysis

## Success Criteria

The Sigil port is considered **successful** if:

1. **100%** of shared tests pass
2. **LOC reduction ≥25%** vs Rust
3. **Stream latency within 10%** of Rust
4. **Memory usage within 15%** of Rust
5. **All LLM outputs marked `~`** (evidentiality compliance)
6. Code is **subjectively more readable** for streaming logic

## Exit Criteria

Abort Sigil port if:

1. Sigil compiler bugs block progress for >1 week
2. Stream performance degrades >30% vs Rust
3. Async/streaming primitives missing for >2 weeks
4. Cannot express TokenStream trait equivalent

## Timeline

| Phase | Days | Deliverable |
|-------|------|-------------|
| Phase 1: Types | 1-2 | types.sigil, error.sigil |
| Phase 2: Pure Functions | 3-4 | sampling.sigil |
| Phase 3: Streaming | 5-7 | streaming.sigil |
| Phase 4: Request/Response | 8-10 | request.sigil, response.sigil |
| Phase 5: Model | 11-12 | model.sigil |
| Phase 6: Integration | 13-15 | Full benchmark suite |

**Total Estimated Effort:** 15 developer-days
