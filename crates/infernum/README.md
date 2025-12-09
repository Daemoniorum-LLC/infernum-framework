# Infernum

> *"From the depths, intelligence rises"*

**Blazingly fast local LLM inference CLI for running large language models on your machine.**

[![Crates.io](https://img.shields.io/crates/v/infernum.svg)](https://crates.io/crates/infernum)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Quick Start

```bash
# Install
cargo install infernum

# Set your model
infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Start chatting
infernum chat
```

## Features

- **Local Inference**: Run LLMs entirely on your machine - no API keys, no cloud
- **OpenAI Compatible API**: Drop-in replacement server at `localhost:8080`
- **Interactive Chat**: Full-featured CLI chat with history and session management
- **Multi-Backend**: CPU, CUDA (NVIDIA), and Metal (Apple Silicon) support
- **Smart Caching**: Models download once via HuggingFace Hub
- **Streaming**: Real-time token-by-token output

## Commands

```bash
# Chat interface
infernum chat [--model MODEL] [--system PROMPT]

# Start API server
infernum server [--port PORT]

# Download a model
infernum pull meta-llama/Llama-3.2-3B-Instruct

# List available models
infernum list

# System diagnostics
infernum doctor

# Configuration
infernum config set-model MODEL
infernum config get-model
```

## GPU Support

```bash
# Install with CUDA support
cargo install infernum --features cuda

# Install with Metal support (macOS)
cargo install infernum --features metal
```

## Architecture

Infernum is built from specialized components:

- **abaddon**: High-performance inference engine
- **malphas**: Model orchestration and lifecycle management
- **stolas**: Knowledge retrieval and RAG capabilities
- **beleth**: Agent framework for autonomous AI
- **asmodeus**: Model fine-tuning and adaptation
- **dantalion**: Observability and telemetry
- **infernum-server**: OpenAI-compatible HTTP API

## Documentation

Full documentation available at [infernum.daemoniorum.com](https://infernum.daemoniorum.com)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
