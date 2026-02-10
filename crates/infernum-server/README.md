# infernum-server

HTTP API server for the Infernum LLM inference framework.

## Overview

`infernum-server` provides a production-ready HTTP server that exposes Infernum's LLM capabilities through industry-standard `/v1/*` API endpoints. Works with any client that supports standard chat completion APIs.

## Features

- **Standard API**: Industry-standard `/v1/*` routes compatible with existing clients
- **Streaming Responses**: Real-time token-by-token output via SSE
- **Model Cache Management**: Download, convert, and manage local models
- **HoloTensor Compression**: Convert models to compressed HoloTensor format
- **Agent Framework**: ReAct-style agents with tool execution
- **RAG System**: Knowledge retrieval with vector embeddings
- **Health & Metrics**: Built-in health checks and Prometheus metrics
- **CORS Support**: Configurable cross-origin resource sharing

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

### Chat & Inference Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming/non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Generate embeddings |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/load` | POST | Load a model into memory |
| `/api/models/unload` | POST | Unload current model |
| `/api/status` | GET | Server and model status |

### Model Cache Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cache/models` | GET | List cached models |
| `/api/cache/models/delete` | POST | Delete a cached model |
| `/api/cache/models/convert` | POST | Convert model to HoloTensor (SSE streaming) |
| `/api/models/download` | POST | Download from HuggingFace (SSE streaming) |

### Agent Framework

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agent/tools` | GET | List available tools |
| `/api/agent/run` | POST | Execute agent with objective (SSE streaming) |
| `/api/sessions` | GET | List active agent sessions |
| `/api/sessions/{id}` | GET | Get session details |
| `/api/sessions/{id}/events` | GET | Stream session events (SSE) |

### RAG (Retrieval-Augmented Generation)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/health` | GET | RAG system health |
| `/api/rag/index` | POST | Index documents |
| `/api/rag/search` | POST | Search indexed documents |

### Health & Metrics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/deep` | GET | Deep health check with component status |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

## Streaming Endpoints

Several endpoints use Server-Sent Events (SSE) for real-time progress updates:

### Model Download (`POST /api/models/download`)

```bash
curl -X POST http://localhost:8080/api/models/download \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct"}'
```

SSE events include:
- `progress`: Download progress with `percent`, `file`, `files_done`, `files_total`
- `complete`: Download finished with `bytes_total`
- `error`: Error occurred with `message`

Supports:
- **Sharded models**: Automatically detects and downloads 70B+ models with multiple weight files
- **Single models**: Downloads single safetensors/pytorch files
- **HoloTensor conversion**: Optional post-download conversion via `convert_to_holo: true`

### Model Convert (`POST /api/cache/models/convert`)

```bash
curl -X POST http://localhost:8080/api/cache/models/convert \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "num_fragments": 64,
    "max_rank": 256,
    "min_quality": 0.85,
    "verify": true
  }'
```

SSE events include:
- `progress`: Conversion progress with `operation`, `tensor`, `compression_ratio`
- `complete`: Conversion finished with `metadata` (compression ratio, quality score)
- `error`: Error occurred

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERNUM_PORT` | 8080 | Server port |
| `INFERNUM_HOST` | 0.0.0.0 | Bind address |
| `INFERNUM_MODELS_DIR` | ~/.cache/infernum | Model cache directory |
| `HF_HOME` | ~/.cache/huggingface | HuggingFace cache directory |

## Examples

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Stream Chat

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Run Agent

```bash
curl http://localhost:8080/api/agent/run \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "objective": "List the files in the current directory",
    "tools": ["bash", "file_read"],
    "max_iterations": 10
  }'
```

### List Cached Models

```bash
curl http://localhost:8080/api/cache/models
```

Response:
```json
{
  "models": [
    {
      "id": "meta-llama/Llama-3.2-3B-Instruct",
      "name": "Llama-3.2-3B-Instruct",
      "source": "huggingface",
      "size_bytes": 6438985728,
      "size_str": "6.00 GB",
      "is_holotensor": false,
      "architecture": "llama"
    }
  ],
  "total_size_bytes": 6438985728,
  "total_size_str": "6.00 GB",
  "cache_dir": "/home/user/.cache/huggingface/hub"
}
```

## Part of Infernum Framework

This crate is part of the [Infernum](https://github.com/Daemoniorum-LLC/infernum-framework) ecosystem:

- **infernum-core**: Shared types and traits
- **abaddon**: Inference engine with Flash Attention
- **malphas**: Model orchestration and scheduling
- **stolas**: Knowledge retrieval (RAG) with BM25 and vector search
- **beleth**: Agent framework (ReAct, Tree of Thought)
- **dantalion**: Observability (Prometheus, Jaeger)
- **haagenti**: HoloTensor compression (LRDF holographic encoding)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
