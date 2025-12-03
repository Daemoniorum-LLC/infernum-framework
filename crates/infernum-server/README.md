# infernum-server

HTTP API server with OpenAI-compatible endpoints for the Infernum LLM inference framework.

## Overview

`infernum-server` provides a production-ready HTTP server that exposes Infernum's LLM capabilities through OpenAI-compatible API endpoints. This allows you to use any OpenAI client library or tool with locally-running models.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's API
- **Streaming Responses**: Real-time token-by-token output
- **Health & Metrics**: Built-in health checks and Prometheus metrics
- **CORS Support**: Configurable cross-origin resource sharing
- **Request Validation**: Type-safe request/response handling

## Usage

```rust
use infernum_server::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server = Server::new("0.0.0.0:8080").await?;
    server.run().await
}
```

Or use the CLI:

```bash
infernum server --port 8080
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming/non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Part of Infernum Framework

This crate is part of the [Infernum](https://github.com/Daemoniorum-LLC/infernum-framework) ecosystem:

- **infernum-core**: Shared types and traits
- **abaddon**: Inference engine
- **malphas**: Model orchestration
- **stolas**: Knowledge retrieval (RAG)
- **beleth**: Agent framework
- **dantalion**: Observability

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
