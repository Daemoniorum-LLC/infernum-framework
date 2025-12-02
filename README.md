# Infernum

> *"From the depths, intelligence rises"*

**Blazingly fast local LLM inference for the command line and beyond.**

Infernum is a high-performance LLM inference framework written in Rust, designed to run large language models locally with exceptional speed and minimal setup. Drop-in OpenAI API compatibility means your existing tools just work.

[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

---

## Quick Start

```bash
# Install (from source)
cargo install --path crates/infernum

# Set your default model (do this once)
infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Start chatting!
infernum chat
```

**That's it.** Infernum handles downloading, caching, and optimizing the model automatically.

---

## Features

| Feature | Description |
|---------|-------------|
| **Local Inference** | Run LLMs entirely on your machine - no API keys, no cloud, no data leaving your system |
| **OpenAI Compatible** | Drop-in replacement API at `localhost:8080` - works with any OpenAI client |
| **Streaming** | Real-time token-by-token output for responsive interactions |
| **Multi-Backend** | CPU, CUDA (NVIDIA), and Metal (Apple Silicon) support |
| **Smart Caching** | Models download once and cache locally via HuggingFace Hub |
| **Interactive Chat** | Full-featured chat with history, save/load, and session management |

---

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Daemoniorum-LLC/infernum-framework.git
cd infernum-framework

# Build release binary
cargo build --release

# Binary is at target/release/infernum
# Optionally, install to PATH:
cargo install --path crates/infernum
```

### GPU Acceleration

```bash
# NVIDIA CUDA support
cargo build --release --features cuda

# Apple Metal support
cargo build --release --features metal
```

### Verify Installation

```bash
infernum doctor
```

---

## Usage

### Interactive Chat

```bash
# Start chatting with default model
infernum chat

# Use a specific model
infernum chat --model meta-llama/Llama-3.2-3B-Instruct

# Set a custom system prompt
infernum chat --system "You are a helpful coding assistant"
```

**Chat Commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/history` | View conversation history |
| `/save <file>` | Save conversation to JSON file |
| `/load <file>` | Load conversation from file |
| `exit` or `quit` | End the session |

### Single Generation

```bash
# Quick one-off generation
infernum generate "Explain quantum computing in simple terms"

# With parameters
infernum generate "Write a haiku about Rust" \
  --max-tokens 50 \
  --temperature 0.9 \
  --stream
```

### API Server

```bash
# Start the server
infernum serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Server runs at http://localhost:8080
# Compatible with any OpenAI client!
```

**Use with curl:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Use with Python (OpenAI client):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Model Management

```bash
# List cached models
infernum model list

# Download a model
infernum model pull meta-llama/Llama-3.2-1B-Instruct

# View model details
infernum model info meta-llama/Llama-3.2-1B-Instruct

# Remove a cached model
infernum model remove TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Configuration

```bash
# View current configuration
infernum config show

# Set default model
infernum config set-model meta-llama/Llama-3.2-3B-Instruct

# Clear default model
infernum config clear-model

# Show config file location
infernum config path
```

**Config file location:** `~/.config/infernum/config.toml`

**Example config:**
```toml
default_model = "meta-llama/Llama-3.2-3B-Instruct"
temperature = 0.7
max_tokens = 256
server_host = "0.0.0.0"
server_port = 8080
```

**Environment variables:** All settings can be overridden with `INFERNUM_*` prefix:
```bash
export INFERNUM_DEFAULT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export INFERNUM_TEMPERATURE="0.8"
```

---

## Recommended Models

| Model | Size | Use Case | Command |
|-------|------|----------|---------|
| TinyLlama-1.1B-Chat | ~2GB | Quick testing, low resources | `infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Llama-3.2-1B-Instruct | ~2GB | Fast, capable | `infernum config set-model meta-llama/Llama-3.2-1B-Instruct` |
| Llama-3.2-3B-Instruct | ~6GB | Balanced performance | `infernum config set-model meta-llama/Llama-3.2-3B-Instruct` |
| Qwen2.5-7B-Instruct | ~14GB | High quality, needs GPU | `infernum config set-model Qwen/Qwen2.5-7B-Instruct` |

> **Note:** Larger models require HuggingFace authentication. Run `huggingface-cli login` first.

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/api/status` | GET | Server status with uptime |
| `/api/models/load` | POST | Load a model dynamically |
| `/api/models/unload` | POST | Unload current model |

### Chat Completions Request

```json
{
  "model": "string",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false,
  "stop": ["\\n\\n"]
}
```

### Chat Completions Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}
```

---

## Architecture

Infernum is built as a modular ecosystem of specialized crates:

```
infernum/
├── infernum          # CLI application
├── infernum-core     # Shared types and traits
├── infernum-server   # HTTP API server
├── abaddon           # Core inference engine
├── malphas           # Model orchestration & routing
├── stolas            # Knowledge/RAG engine
├── beleth            # Autonomous agent framework
├── asmodeus          # Fine-tuning & adaptation
├── dantalion         # Observability & metrics
└── grimoire-loader   # Persona/prompt management
```

---

## Troubleshooting

### "Model not found" or download fails

```bash
# Login to HuggingFace for gated models
huggingface-cli login

# Or use an ungated model
infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Slow inference on CPU

GPU acceleration dramatically improves performance:
```bash
# Rebuild with GPU support
cargo build --release --features cuda   # NVIDIA
cargo build --release --features metal  # Apple Silicon
```

### Out of memory

Try a smaller model or enable CPU offloading:
```bash
# Use a smaller model
infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Check system status

```bash
infernum doctor
```

---

## Development

```bash
# Run tests
cargo test --workspace

# Run with debug logging
RUST_LOG=debug infernum chat

# Format code
cargo fmt --all

# Lint
cargo clippy --workspace
```

---

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

---

## Acknowledgments

Part of the [Persona Framework](https://github.com/Daemoniorum-LLC/persona-framework) ecosystem by Daemoniorum LLC.

Built with [Candle](https://github.com/huggingface/candle), [Axum](https://github.com/tokio-rs/axum), and the incredible Rust ecosystem.
