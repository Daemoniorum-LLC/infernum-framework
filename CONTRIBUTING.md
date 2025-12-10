# Contributing to Infernum

Thank you for your interest in contributing to Infernum! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/infernum-framework`
3. Add upstream remote: `git remote add upstream https://github.com/Daemoniorum-LLC/infernum-framework`
4. Create a feature branch: `git checkout -b feature/your-feature`

## Development Setup

### Prerequisites

- Rust 1.91 or later
- Cargo

### Building

```bash
cargo build --workspace
```

### Running Tests

```bash
cargo test --workspace
```

### Linting

```bash
cargo clippy --workspace --all-features
cargo fmt --all -- --check
```

## Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `refactor/` - Code refactoring

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add streaming response support
fix: resolve memory leak in tokenizer cache
docs: update API examples
refactor: simplify model loading logic
test: add integration tests for chat endpoint
chore: update dependencies
perf: optimize token sampling
```

### Code Style

- Follow Rust idioms and best practices
- Run `cargo fmt` before committing
- Ensure `cargo clippy` passes without warnings
- Add documentation for public items
- Write tests for new functionality

### Pull Request Process

1. Update your fork with the latest upstream changes
2. Ensure all tests pass: `cargo test --workspace`
3. Ensure linting passes: `cargo clippy --workspace --all-features`
4. Ensure formatting is correct: `cargo fmt --all -- --check`
5. Push your branch and create a pull request

### PR Guidelines

- Keep PRs focused on a single change
- Write a clear description of what and why
- Link related issues if applicable
- Ensure CI passes before requesting review

## Architecture Overview

Infernum is organized as a Rust workspace with specialized crates:

| Crate | Purpose |
|-------|---------|
| `infernum` | CLI application |
| `infernum-core` | Shared types and traits |
| `abaddon` | Inference engine |
| `malphas` | Orchestration layer |
| `stolas` | RAG/Knowledge engine |
| `beleth` | Agent framework |
| `asmodeus` | Fine-tuning |
| `dantalion` | Observability |
| `infernum-server` | HTTP API server |
| `grimoire-loader` | Prompt loading |

## Reporting Issues

When reporting bugs, please include:

- Rust version (`rustc --version`)
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first
- Describe the use case
- Explain why existing features don't solve it

## License

By contributing, you agree that your contributions will be licensed under the same MIT OR Apache-2.0 dual license as the project.

## Questions?

Open an issue or start a discussion. We're happy to help!
