# Changelog

All notable changes to the Infernum Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-rc.1] - 2026-02-10

### Added

#### Inference Engine (abaddon)
- **llama.cpp backend** for high-performance GGUF model inference
- **Tiered memory system** for models exceeding VRAM capacity
  - Lazy HoloTensor loading for oversized models
  - Async prefetching for tiered memory
  - Eager and progressive weight loaders
  - GPU/CPU memory coordination with eviction policies
- **GPU codec pipeline** (TDD Phases 1-4)
  - CUDA-accelerated tensor compression/decompression
  - Cross-validation tests for codec correctness
- **HoloTensor format** improvements for compressed model storage
- Property-based tests for tiered memory system

#### Server (infernum-server)
- **Tool calling / function calling** with model-aware formatting
  - Streaming tool detection for SSE
  - Multi-model support
  - Validation and enforcement
- **Agent runtime** with Moloch audit chain integration
- **Continuous batching** for high-throughput inference (vLLM-style)
- **Speculative decoding** for 2-3x inference speedup
- **WebSocket streaming** for low-latency inference
- **Vision/multimodal** content support for image inputs
- **Structured outputs** with JSON schema enforcement
- **gRPC API** for high-performance internal services
- **Priority-based request queuing** with fairness
- **Response caching** for deterministic requests
- **Request deduplication** for coalescing identical in-flight requests
- **Circuit breaker** pattern for backend resilience
- **GPU metrics** collection for Prometheus monitoring
- **Request cancellation** support for graceful client disconnection
- OpenAPI 3.1 documentation (`/docs` endpoint)
- Configuration hot reload with file watching
- Model cache management API

#### Agent Framework (beleth)
- Agentic loop implementation
- Code tools integration
- Native tool calling support

#### Build & CI
- Platform-appropriate feature flags (cuda/metal opt-in, not default)
- Workspace lint configuration with pedantic/nursery lints
- MSRV verification (Rust 1.91)

### Changed
- **BREAKING**: CUDA and Metal features now require explicit opt-in
  - Use `--features cuda` or `--features metal` to enable GPU support
  - Default build is CPU-only for broader compatibility
- Migrated to workspace-level lint configuration
- candle-core dependencies now use `default-features = false`

### Fixed
- CI failures on non-GPU runners resolved
- All clippy warnings addressed (3800+ warnings cleaned up)
- Test compilation errors in malphas, infernum-server
- Unused imports and variables cleaned up

## [0.1.0] - 2024-12-20

### Added
- Initial release
- Core inference engine with candle backend
- REST API with standard `/v1/*` endpoints
- Streaming support via Server-Sent Events (SSE)
- HuggingFace Hub integration for model downloads
- Basic CLI with chat and serve commands
- Configuration system with TOML support
- Prometheus metrics and health endpoints
