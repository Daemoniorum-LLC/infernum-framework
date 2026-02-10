---
project: infernum-framework
type: llm-context
stack: Rust 1.91, Candle, Axum, tokio
entry_points:
  - nyx/infernum/infernum-framework/Cargo.toml
  - nyx/infernum/infernum-framework/crates/infernum/src/main.rs
  - nyx/infernum/infernum-framework/crates/infernum-server/
---

# Infernum Framework

## Purpose

Local LLM inference framework written in Rust, providing OpenAI-compatible API and CLI for running large language models locally.

## Architecture

```
nyx/infernum/infernum-framework/
├── Cargo.toml                  # Workspace definition
├── crates/
│   ├── infernum/              # CLI application
│   ├── infernum-core/         # Shared types and traits
│   ├── infernum-server/       # HTTP API server (Axum)
│   ├── abaddon/               # Core inference engine
│   ├── malphas/               # Model orchestration
│   ├── stolas/                # Knowledge/RAG engine
│   ├── beleth/                # Agent framework
│   ├── asmodeus/              # Fine-tuning/adaptation
│   ├── dantalion/             # Observability/metrics
│   ├── grimoire-loader/       # Persona/prompt management
│   └── test-utils/            # Test utilities
└── target/                    # Build output
```

## Key Files

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace dependencies and members |
| `crates/infernum/src/main.rs` | CLI entry point |
| `crates/infernum-server/src/lib.rs` | HTTP server implementation |
| `crates/abaddon/src/lib.rs` | Inference engine core |
| `crates/infernum-core/src/lib.rs` | Shared types |

## Build & Test

```bash
# Build release binary
cargo build --release

# Build with GPU support
cargo build --release --features cuda   # NVIDIA
cargo build --release --features metal  # Apple Silicon

# Run tests
cargo test --workspace

# Run specific crate tests
cargo test -p abaddon

# Format and lint
cargo fmt --all
cargo clippy --workspace

# Generate documentation
cargo doc --workspace --no-deps --open
```

## CLI Commands

```bash
# Interactive chat
infernum chat
infernum chat --model meta-llama/Llama-3.2-3B-Instruct

# Single generation
infernum generate "Explain quantum computing"

# Start API server
infernum serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Model management
infernum model list
infernum model pull meta-llama/Llama-3.2-1B-Instruct
infernum model info <model-name>

# Configuration
infernum config show
infernum config set-model <model-name>

# System check
infernum doctor
```

## HTTP API

**Base URL:** `http://localhost:8080` (or 8081 in Docker)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

**Example Request:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Common Tasks

### Add New Crate

1. Create directory under `crates/`
2. Add `Cargo.toml` with workspace dependencies
3. Add to `[workspace.members]` in root `Cargo.toml`
4. Implement `lib.rs` or `main.rs`

### Add API Endpoint

1. Add handler in `crates/infernum-server/src/routes/`
2. Register route in router
3. Add OpenAPI documentation
4. Write integration test

### Update Dependencies

1. Update version in `[workspace.dependencies]`
2. Run `cargo update`
3. Run full test suite

## Configuration

**Config file:** `~/.config/infernum/config.toml`

```toml
default_model = "meta-llama/Llama-3.2-3B-Instruct"
temperature = 0.7
max_tokens = 256
server_host = "0.0.0.0"
server_port = 8080
```

**Environment variables:** `INFERNUM_*` prefix
```bash
export INFERNUM_DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export INFERNUM_TEMPERATURE="0.8"
```

## Constraints

- **MUST** use Rust 1.91+
- **MUST** run `cargo clippy` before committing
- **MUST** document public APIs with rustdoc
- **MUST NOT** use `unwrap()` or `expect()` in production code
- **MUST NOT** use `unimplemented!()`, `todo!()`, `panic!()`
- **SHOULD** add tests for new functionality
- **SHOULD** use workspace dependencies

## Workspace Lints

```toml
[workspace.lints.rust]
unsafe_code = "warn"
missing_docs = "warn"

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
unwrap_used = "warn"
expect_used = "warn"
```

## Related Documentation

- [README](../../../nyx/infernum/infernum-framework/README.md) - Full Infernum documentation
- [API Documentation](../../reference/API-DOCUMENTATION.md) - Rustdoc generation
- [CLAUDE.md](../../../nyx/CLAUDE.md) - Nyx ecosystem guide
