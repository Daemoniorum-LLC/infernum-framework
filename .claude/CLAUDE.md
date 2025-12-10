# Infernum Framework - Development Guide

## Project Overview

Infernum is a blazingly fast local LLM inference framework written in Rust. The codebase is organized as a workspace with 10+ specialized crates.

## Crate Architecture

| Crate | Purpose |
|-------|---------|
| `infernum` | Main CLI application |
| `infernum-core` | Shared types and traits |
| `abaddon` | Inference engine |
| `malphas` | Orchestration and routing |
| `stolas` | Knowledge/RAG engine |
| `beleth` | Agent framework |
| `asmodeus` | Fine-tuning and LoRA |
| `dantalion` | Observability and metrics |
| `infernum-server` | HTTP API server (OpenAI compatible) |
| `grimoire-loader` | Persona/prompt loading |

## Development Workflow

### Building

```bash
cargo build --workspace
cargo build --release --workspace
```

### Testing

```bash
cargo test --workspace
cargo test --workspace -- --nocapture  # with output
```

### Linting

```bash
cargo clippy --workspace --all-features
cargo fmt --all -- --check
```

### Running

```bash
cargo run -p infernum -- --help
cargo run -p infernum -- serve --port 8080
cargo run -p infernum -- chat --model "HuggingFaceTB/SmolLM2-135M"
```

## Git Workflow

**CRITICAL**: Never work directly on main or development branches.

1. Create feature branch: `git checkout -b feature/goal-name`
2. Commit regularly with conventional commits
3. Push and create PR for review

```bash
git checkout -b feature/streaming-chat
git add -A && git commit -m "feat: streaming implementation"
git push origin feature/streaming-chat
```

## Conventional Commits

Use these prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

## Code Standards

- Follow Rust idioms and the project's clippy configuration
- All public items should have documentation
- Use `thiserror` for error types
- Use `tracing` for logging
- Prefer async/await for I/O operations
- Write tests for new functionality

## Benchmarks

Run inference benchmarks:
```bash
cargo bench -p abaddon
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace Hub token for gated models |
| `INFERNUM_LOG` | Log level (trace, debug, info, warn, error) |
| `INFERNUM_CONFIG` | Custom config file path |

## Configuration

Default config location: `~/.config/infernum/config.toml`
