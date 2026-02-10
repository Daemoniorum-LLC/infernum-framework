# Changelog

All notable changes to infernum-server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Resolved all compiler warnings across the infernum codebase
- Fixed deprecated `TimeoutLayer::new` API usage (now uses `with_status_code`)
- Added accessor methods for previously unused struct fields
- Moved test-only imports to `#[cfg(test)]` modules to eliminate warnings

### Added
- Model cache management API for browsing and managing local models
  - `GET /api/cache/models` - List cached models from HuggingFace and Infernum directories
  - `POST /api/cache/models/delete` - Delete a cached model
  - `POST /api/cache/models/convert` - Convert model to HoloTensor format with SSE streaming
  - `POST /api/models/download` - Download from HuggingFace with SSE progress
  - `CachedModel` type with size, source, architecture, and HoloTensor status
  - Automatic model metadata extraction from config.json
- HoloTensor conversion with real-time SSE streaming progress
  - `ConvertProgress` events for tensor-by-tensor progress
  - Configurable compression options (fragments, max_rank, min_quality, verify)
  - Compression ratio and quality score reporting
- Dynamic shard detection for large models (70B+)
  - Parses `model.safetensors.index.json` for sharded safetensors
  - Parses `pytorch_model.bin.index.json` for sharded PyTorch models
  - Downloads all shards with per-file progress tracking
- Token counting endpoint (`POST /v1/tokenize`) for pre-flight validation
- Logprobs support in chat completions (`logprobs` and `top_logprobs` parameters)
- Function calling / tool use support with standard types
  - `Tool`, `FunctionDefinition`, `ToolCall`, `FunctionCall` types
  - `tools` and `tool_choice` parameters in chat completions
  - Support for tool response messages (`role: "tool"`)
- Request priority header (`X-Priority`) for queue ordering
  - Priority levels: high, normal, low, background
  - Supports both string names and numeric values (1-4)
- Circuit breaker pattern for inference backend resilience
  - States: Closed (normal), Open (rejecting), HalfOpen (testing recovery)
  - Configurable failure threshold, reset timeout, and half-open requests
  - Lock-free atomic state management for high-throughput scenarios
  - Prometheus metrics for circuit breaker monitoring
- GPU metrics collection for Prometheus monitoring
  - GPU utilization, memory used/total, temperature, power usage
  - `GpuMetricsProvider` trait for mock/real implementations
  - `MockGpuMetrics` for testing and CPU-only environments
  - Prometheus-format metrics rendering
- Request cancellation support for graceful client disconnection
  - `CancellationToken` for checking cancellation status
  - `RequestCancellation` controller for triggering cancellation
  - `CancellationMetrics` with Prometheus output
  - Reasons: ClientDisconnected, Timeout, ServerShutdown, Manual
- Structured error subcodes for granular error handling
  - `ErrorSubcode` enum with 27 specific subcodes
  - `limit` and `actual` fields for validation errors
  - Subcodes for context length, validation, rate limiting, auth, and content filter errors
- Response caching for deterministic requests
  - LRU cache with configurable max entries and memory limit
  - TTL-based expiration for cache entries
  - Cacheable requests: temperature <= 0.0 (deterministic)
  - Cache key generation from model, content hash, and max_tokens
  - `X-Cache: HIT/MISS` response header
  - Prometheus metrics for cache hits, misses, evictions
- Admin Model API for runtime model management
  - `POST /admin/models/load` - Load a model with configuration options
  - `POST /admin/models/unload` - Unload a model from memory
  - `GET /admin/models/status` - Get status of all loaded models
  - `POST /admin/models/warmup` - Warmup a model for faster inference
  - `ModelRegistry` for tracking loaded models and request counts
  - Prometheus metrics for model memory, active requests, and status
- Request deduplication for coalescing identical in-flight requests
  - `RequestDeduplicator` for managing in-flight request state
  - `ComputeHandle` for request owners to complete or fail results
  - Only deduplicates deterministic requests (temperature <= 0.0)
  - Streaming requests excluded from deduplication
  - Prometheus metrics for dedup hits, misses, and hit ratio
- Per-request timeout via `X-Request-Timeout` header
  - Accepts timeout in seconds (integer or decimal)
  - Optional suffixes: `30s` or `30sec`
  - Configurable min/max bounds with clamping
  - `TimeoutConfig` for server-side timeout policy
  - Prometheus metrics for custom/clamped timeouts
- Structured outputs for JSON schema enforcement
  - `ResponseFormat` enum: text, json_object, json_schema
  - `JsonSchema` type with name, description, schema, and strict mode
  - `response_format` parameter in chat completions
  - `validate_json()` function for output validation
  - `SchemaRegistry` for reusable schema definitions
  - Support for all JSON Schema validation rules
- WebSocket streaming for low-latency inference
  - `ClientMessage` and `ServerMessage` types for bidirectional protocol
  - `ConnectionState` for tracking connection lifecycle
  - `WsConfig` for configuring message size, ping interval, compression
  - `CloseReason` with standard WebSocket close codes
  - `ConnectionManager` for managing multiple connections
  - `WsMetrics` with Prometheus output for monitoring
  - Support for request cancellation via WebSocket
- Vision/multimodal content support for image inputs
  - `MessageContent` enum: text or array of content parts
  - `ContentPart` types: Text, ImageUrl, ImageBase64
  - `ImageDetail` levels: low (512px), high (2048px), auto
  - `ImageUrl` with detail level configuration
  - `ImageBase64` with media type and decode support
  - `VisionConfig` for validation limits and allowed hosts
  - Support for JPEG, PNG, GIF, and WebP formats
  - Base64 encoding validation and decoding
  - Data URL support for inline images
  - `VisionMetrics` with Prometheus output
- Continuous batching for high-throughput inference (vLLM-style)
  - `BatchScheduler` for managing pending requests and active batches
  - `ActiveBatch` with dynamic sequence addition/removal
  - `TokenIterator` for token-level iteration across batch
  - Scheduling policies: FCFS, Priority, ShortestJobFirst, LongestJobFirst
  - Preemption policies: None, Priority, Recompute, Swap
  - `BatchPriority` levels: Background, Normal, High, Critical
  - `Sequence` and `SequenceGroup` for beam search support
  - `SamplingParams` for temperature, top_p, top_k, repetition penalty
  - Lock-free atomic counters for high-performance metrics
  - Prometheus metrics for scheduler and iteration performance
- Priority-based request queuing with fairness
  - `RequestQueue` with 4 priority levels and weighted fair queuing (WFQ)
  - `QueueConfig` for max depth, per-priority limits, and weights
  - `QueuedRequest` with payload and estimated tokens
  - Starvation prevention with configurable timeout and promotion
  - Batch dequeue for efficient processing
  - `QueueMetrics` with per-priority counters and Prometheus output
  - Request removal and queue draining for graceful shutdown
  - 25 unit tests for comprehensive coverage
- gRPC API for high-performance internal services
  - `InfernumService` trait for implementing gRPC service handlers
  - Request/response types: `ChatCompletionRequest`, `CompletionRequest`, `EmbedRequest`
  - Streaming support: `ChatCompletionChunk`, `CompletionChunk` for server-streaming
  - `GrpcConfig` for server configuration (message size, timeouts, keepalive)
  - `GrpcPriority` enum mapping to `BatchPriority` for queue integration
  - `GrpcError` enum with conversion to `tonic::Status` codes
  - `GrpcMetrics` with Prometheus output for requests, errors, and streams
  - `MockInfernumService` for testing with streaming support
  - Health check support with component-level status
  - 18 unit tests including async streaming tests
- Speculative decoding for 2-3x inference speedup
  - `SpeculativeConfig` with draft model, token count, and acceptance threshold
  - `SpeculativeMode`: Disabled, Enabled, Auto (enable for low temperature)
  - `SpeculativeParams` for per-request speculative control
  - `DraftToken` and `VerificationResult` types for decoding pipeline
  - `SpeculativeScheduler` integrating with `BatchScheduler`
  - Adaptive speculation: adjusts token count based on acceptance rate
  - `SpeculativeStats` with acceptance rate, speedup estimation
  - `SpeculativeMetrics` with comprehensive Prometheus output
  - Request headers: `X-Speculative`, `X-Speculative-Draft-Model`, `X-Speculative-Tokens`
  - 31 unit tests for full coverage
- OpenAPI 3.1 documentation with utoipa (`/docs` endpoint)
- Configuration hot reload with file watching
- OpenTelemetry distributed tracing with OTLP export
- Criterion performance benchmarks
- W3C Trace Context propagation
- InferenceSpan helper for LLM operation tracing

### Changed
- Improved error response structure with retry hints

### Fixed
- None

## [0.1.0] - 2024-12-20

### Added
- Initial release
- REST API with standard `/v1/*` endpoints
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/embeddings`
  - `GET /v1/models`
- Streaming support via Server-Sent Events (SSE)
- API key authentication with scopes (inference, admin, metrics)
- Per-IP and per-key rate limiting
- Request validation with configurable limits
- Health check endpoints (`/health`, `/ready`)
- Prometheus metrics (`/metrics`)
- Security headers (CSP, HSTS, X-Frame-Options)
- CORS configuration
- TLS support
- Audit logging
- Structured error responses with:
  - Machine-readable error codes
  - Request ID correlation
  - Retry information
  - Recovery hints

### Security
- Input validation for all request parameters
- Rate limiting to prevent abuse
- API key scoping for least privilege
- Security headers for browser clients

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned features:

- **Phase 1**: Token counting, logprobs, function calling
- **Phase 2**: Circuit breaker, GPU metrics, request cancellation
- **Phase 3**: Response caching, request deduplication
- **Phase 4**: Continuous batching, structured outputs, vision
- **Phase 5**: Request queuing, gRPC, speculative decoding
