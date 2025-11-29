# Infernum

> *"From the depths, intelligence rises"*

**Infernum** is Daemoniorum's next-generation AI infrastructure ecosystem, built entirely in Rust for maximum performance, safety, and deployment flexibility.

## Components

| Component | Description | Status |
|-----------|-------------|--------|
| **Abaddon** | Core inference engine with multi-backend support | 🚧 WIP |
| **Malphas** | Model orchestration and intelligent routing | 🚧 WIP |
| **Stolas** | Knowledge engine with RAG capabilities | 🚧 WIP |
| **Beleth** | Autonomous agent framework | 🚧 WIP |
| **Asmodeus** | Fine-tuning and model adaptation | 🚧 WIP |
| **Dantalion** | Observability and telemetry | 🚧 WIP |

## Quick Start

```bash
# Build the project
cargo build --release

# Run the server
./target/release/infernum serve --model meta-llama/Llama-3.2-3B-Instruct

# Generate text
./target/release/infernum generate "Hello, world!" --max-tokens 100

# Interactive chat
./target/release/infernum chat
```

## Features

- **Blazing Performance**: Sub-10ms p99 latency for token generation
- **Multi-Backend**: CUDA, Metal, WebGPU, and CPU support
- **PagedAttention**: Efficient KV-cache memory management
- **FlashAttention**: Fused attention kernels
- **Speculative Decoding**: Draft model acceleration
- **OpenAI-Compatible API**: Drop-in replacement server
- **Grimoire Integration**: Native prompt management

## Building

### Requirements

- Rust 1.83+
- CUDA 12.0+ (optional, for GPU support)
- Metal (macOS, automatic)

### Build Commands

```bash
# CPU-only build
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With Metal support (macOS)
cargo build --release --features metal

# All features
cargo build --release --all-features
```

## Architecture

```
infernum/
├── crates/
│   ├── infernum/           # CLI binary
│   ├── infernum-core/      # Shared types and traits
│   ├── infernum-server/    # HTTP API server
│   ├── abaddon/            # Inference engine
│   ├── malphas/            # Orchestration layer
│   ├── stolas/             # Knowledge/RAG engine
│   ├── beleth/             # Agent framework
│   ├── asmodeus/           # Fine-tuning
│   ├── dantalion/          # Observability
│   └── grimoire-loader/    # Grimoire integration
└── Cargo.toml              # Workspace config
```

## API Usage

### OpenAI-Compatible Endpoint

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Rust SDK

```rust
use abaddon::{Engine, EngineConfig, GenerateRequest, SamplingParams};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = EngineConfig::builder()
        .model("meta-llama/Llama-3.2-3B-Instruct")
        .cuda(0)
        .build()?;

    let engine = Engine::new(config).await?;

    let response = engine.generate(
        GenerateRequest::new("Hello, world!")
            .with_sampling(SamplingParams::balanced().with_max_tokens(100))
    ).await?;

    println!("{}", response.choices[0].text);
    Ok(())
}
```

## Documentation

- [Technical Design Document](../docs/infernum/TECHNICAL_DESIGN.md)

## License

MIT OR Apache-2.0

---

**Daemoniorum, LLC** — Building Tomorrow's Intelligence
