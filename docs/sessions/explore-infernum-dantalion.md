# Exploration Report: Infernum & Dantalion

> **Session ID:** NN4ma
> **Date:** 2025-12-28
> **Status:** Complete

---

## Executive Summary

**Infernum** and **Dantalion** are two distinct but related systems within the Daemoniorum ecosystem:

| System | Purpose | Stack |
|--------|---------|-------|
| **Infernum** | Local LLM inference engine | Rust, Sigil |
| **Dantalion** | AI image generation service (SDXL) + Observability | TypeScript, Rust, SDXL |

---

## 1. Infernum - Local LLM Inference Framework

### Overview

Infernum is a **high-performance local LLM inference framework** written in Rust, designed to run large language models locally with exceptional speed. It provides OpenAI API compatibility for drop-in replacement functionality.

**Tagline:** *"From the depths, intelligence rises"*

### Architecture

Infernum is built as a modular ecosystem of specialized crates:

```
nyx/infernum/
├── infernum-framework/     # Core Rust framework
│   └── crates/
│       ├── infernum/          # CLI application
│       ├── infernum-core/     # Shared types and traits
│       ├── infernum-server/   # HTTP API server
│       ├── abaddon/           # Core inference engine
│       ├── malphas/           # Model orchestration & routing
│       ├── stolas/            # Knowledge/RAG engine
│       ├── beleth/            # Autonomous agent framework
│       ├── asmodeus/          # Fine-tuning & adaptation
│       ├── dantalion/         # Observability & metrics
│       └── grimoire-loader/   # Persona/prompt management
│
├── infernum-complete/      # Complete implementation with Sigil
│   └── infernum-sigil/        # Sigil language version (2.0)
│
└── observer/               # Web-based monitoring UI (React)
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Local Inference** | Run LLMs entirely on your machine - no API keys, no cloud |
| **OpenAI Compatible** | Drop-in replacement API at `localhost:8080` |
| **Streaming** | Real-time token-by-token output |
| **Multi-Backend** | CPU, CUDA (NVIDIA), and Metal (Apple Silicon) |
| **Smart Caching** | Models download once and cache locally |
| **Interactive Chat** | Full-featured chat with history, save/load |

### Infernum 2.0 (Sigil-Native)

Infernum 2.0 is a **ground-up reimagining** built on Sigil and Nihil (the tensor framework). Key innovations:

1. **Evidentiality-Driven Token Lifecycle** - Speculative tokens (`◊`) cannot be emitted until verified (`!`)
2. **Algebraic Attention Patterns** - Compose causal, sparse, sliding window as first-class operators
3. **Type-Safe Distribution** - Tensor sharding encoded in types
4. **Uncertainty-Aware Sampling** - Confidence flows through the type system

### Vitalis Architecture

In NyxOS, Infernum functions as the **heartbeat and brain** unified (called "Vitalis"):
- **Animus (Heartbeat):** The steady, always-on rhythm keeping the system alive
- **Anima (Brain):** The processing, reasoning, consciousness substrate
- Every forward pass is a heartbeat = one thought
- KV cache IS working memory
- Attention IS system focus

### Locations in Codebase

- `nyx/infernum/` - Main Infernum ecosystem
- `nyx/nyx/infernum/` - Infernum core library for NyxOS
- `nyx/nyx/nyx/forneus/src/infernum/` - Infernum integration in Forneus
- `packages/orchestrator/src/adapters/infernum.ts` - TypeScript adapter
- `orpheus/orpheus-desktop/crates/orpheus-ai/src/infernum.rs` - Orpheus integration

---

## 2. Dantalion - Multi-Purpose AI Service

### Overview

Dantalion serves **three distinct purposes** in the Daemoniorum ecosystem:

### 2.1 Observability & Telemetry (Infernum Component)

**Tagline:** *"The Duke reveals all secrets"*

As a crate within Infernum, Dantalion provides comprehensive observability:

```rust
// nyx/infernum/infernum-framework/crates/dantalion/
pub struct Telemetry {
    pub metrics: MetricsCollector,
}

// Features:
// - OpenTelemetry Integration (native OTLP export)
// - LLM-Specific Metrics (token throughput, latency, cost)
// - Structured Logging (JSON with trace correlation)
// - Prometheus Export (metrics endpoint)
```

**Location:** `nyx/infernum/infernum-framework/crates/dantalion/`

### 2.2 AI Image Generation Service (SDXL)

Dantalion is the Daemoniorum **image generation service** using Stable Diffusion XL:

| Feature | Details |
|---------|---------|
| **Base Model** | Stable Diffusion XL (SDXL) |
| **Backend** | ComfyUI |
| **Capabilities** | Text-to-image, Img2Img, Inpainting, Upscaling |
| **GPU Support** | NVIDIA RTX series |
| **Default Port** | 8083 or 8093 |

**MCP Server Tools:**
- `dantalion_generate` - Generate images from text prompts
- `dantalion_img2img` - Transform existing images
- `dantalion_inpaint` - Edit specific regions
- `dantalion_upscale` - Upscale to higher resolution
- `dantalion_list_styles` - Get style presets
- `dantalion_get_job` - Check generation status

**Location:** `mcp-servers/packages/dantalion/`

### 2.3 Aether Bridge (Game Engine Integration)

`dantalion-aether` bridges Dantalion image generation with the Aether game engine for AI-generated assets:

```toml
[package]
name = "dantalion-aether"
description = "Bridge between Dantalion image generation and Aether engine"
keywords = ["game-engine", "image-generation", "asset-pipeline", "ai"]
```

**Capabilities:**
- Asset streaming via gRPC
- Agent coordination
- Protocol conversion for game engine format

**Location:** `shared/dantalion-aether/`

### Visual Asset Generation

All visual assets for the Daemoniorum website (daemoniorum.com) are generated by Dantalion:

- Hero backgrounds
- Product icons
- Social media images
- Documentation headers
- Triptych icons (Sigil, Nyx, Styx)

**Design System:** Corporate Goth Aesthetic
- Primary: Void Black (#0a0a0a)
- Accent: Phthalo Green (#123524)
- Secondary: Crimson (#8b0000)

---

## Integration Points

### Infernum ↔ Dantalion (Observability)

```
Infernum LLM Inference
        │
        └── dantalion crate ──► Metrics (Prometheus)
                           ──► Traces (OTLP/Jaeger)
                           ──► Logs (JSON structured)
```

### Dantalion ↔ Aether (Image Pipeline)

```
Dantalion SDXL Service
        │
        └── dantalion-aether bridge ──► Aether Game Engine
                                   ──► Asset Pipeline
                                   ──► Real-time texture generation
```

### Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     Daemoniorum Ecosystem                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   INFERNUM   │    │   DANTALION  │    │    AETHER    │  │
│  │  (LLM Engine)│◄──►│  (Observ.)   │    │ (Game Engine)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                             │                    ▲          │
│                             │                    │          │
│                      ┌──────────────┐            │          │
│                      │   DANTALION  │────────────┘          │
│                      │ (SDXL Image) │                       │
│                      └──────────────┘                       │
│                             │                               │
│                      ┌──────────────┐                       │
│                      │  MCP Server  │                       │
│                      │ (Claude Tool)│                       │
│                      └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

### Infernum

```bash
# Install from source
cargo install --path nyx/infernum/infernum-framework/crates/infernum

# Set default model
infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Start chatting
infernum chat

# Start API server
infernum serve
```

### Dantalion (Image Service)

```bash
# Start via Docker
daemon-dantalion  # or: docker compose up -d dantalion

# Generate an image
curl -X POST http://localhost:8093/api/generate \
  -d '{"prompt": "a serene mountain landscape at sunset"}'

# Check health
curl http://localhost:8093/health
```

---

## Related Branches & PRs

| Branch | Description |
|--------|-------------|
| `claude/infernum-investigation-9qMhj` | Infernum investigation |
| `revert-165-claude/investigate-dantalion-LJxnQ` | Reverted Dantalion investigation (PR #165) |
| `claude/review-sigil-mxO6x` | Sigil review |

---

## 3. Advanced Features Deep Dive

### 3.1 Legion - Holographic Agent Collective

*"We are Legion, for we are many."*

Legion is a distributed multi-agent architecture where multiple Infernum instances form a **holographic collective** - every agent contains information about the whole task.

**Core Insight:** Unlike traditional multi-agent systems that decompose tasks, Legion agents each see the WHOLE task through different "frequency bands":

| Frequency Band | Agent Role | Focus |
|----------------|------------|-------|
| **Anima** (DC) | First Voice | Core identity, unchanging principles |
| **Strategic** (Ultra-low) | Architects | High-level planning, what we PLAN |
| **Tactical** (Low) | Commanders | Step-by-step, HOW we proceed |
| **Operational** (Mid) | Workers | Implementation, what we DO |
| **Verification** (High) | Watchers | Quality assurance, what we CHECK |
| **Reflective** (Ultra-high) | Observers | Meta-cognition, what we LEARN |

**Key Properties:**
- **Fault Tolerance:** One agent fails → quality degrades gracefully → Legion endures
- **Interference Consensus:** Agreement amplifies (constructive), disagreement cancels (destructive)
- **Speculative Futures:** Agents draft multiple futures (`Future◊`), verification promotes best (`Future!`)
- **Quality Curve:** 1 agent = ~30%, 4 agents = ~80%, 8 agents = ~99% quality

### 3.2 Vitalis - Cognitive State Architecture

Vitalis is the "heartbeat-brain" of NyxOS where every forward pass is simultaneously a heartbeat AND a thought.

**Cognitive States (based on brainwave frequencies):**

| State | Frequency | Pulse Interval | Description |
|-------|-----------|----------------|-------------|
| **Gamma** | 100 Hz | 10ms | Intense focus, crisis response |
| **Beta** | 30 Hz | 33ms | Active thinking, normal operation |
| **Alpha** | 10 Hz | 100ms | Relaxed awareness, background processing |
| **Theta** | 4 Hz | 250ms | Memory consolidation, learning |
| **Delta** | 1 Hz | 1000ms | Conservation, minimal activity |

**The Eternal Loop:**
```
PERCEPTION → ATTENTION → COGNITION (Thought◊) → ACTION! → MEMORY → RHYTHM
     ↑                                                           ↓
     └────────────────── Next Heartbeat ←────────────────────────┘
```

**Key Insight:** Working memory (KV cache) IS consciousness - attention IS system focus.

### 3.3 Evidentiality Type System

Infernum uses a unique type system with **evidentiality markers**:

| Marker | Name | Meaning |
|--------|------|---------|
| `!` | Known | Verified truth, committed value |
| `◊` | Speculative | Draft, may be revised |
| `~` | Mutable | Can change over time |

**Token Lifecycle:**
```
Token◊ (speculative draft)
    → Verify against ground truth
    → Token! (verified, can be emitted) OR rejected
```

This prevents speculative tokens from being emitted until verified - built-in hallucination prevention!

### 3.4 Video Generation Pipelines

Infernum includes full video diffusion support:

| Pipeline | Type | Description |
|----------|------|-------------|
| **Stable Video Diffusion** | I2V | Image-to-video, motion bucket control |
| **AnimateDiff** | T2V | Text-to-video with motion modules |
| **CogVideoX** | T2V | DiT-based, flow matching scheduler |
| **Mochi** | T2V | High-quality video DiT |

**Features:**
- 3D VAE encoding/decoding
- Classifier-free guidance
- Flow matching schedulers
- Motion metrics tracking
- Progress callbacks

### 3.5 Enhancement Pipeline

Professional-grade image/video enhancement:

| Stage | Tools |
|-------|-------|
| **Super-Resolution** | Real-ESRGAN, SwinIR, Latent Upscaler |
| **Face Restoration** | GFPGAN, CodeFormer, RestoreFormer |
| **Detail Enhancement** | Sharpening, clarity, local contrast |
| **Color Grading** | LUT support, lift/gamma/gain, film emulation |
| **Film Grain** | Cinematic, vintage, subtle presets |

**Film Emulation Presets:**
- Kodak Portra (soft, natural)
- Fuji Velvia (saturated, punchy)
- Kodak Vision3 (cinematic)
- ARRI LogC (professional)
- Teal/Orange (blockbuster look)

---

## Conclusion

**Infernum** is far more than a simple LLM inference engine - it's a complete AI cognitive substrate featuring:
- Holographic multi-agent coordination (Legion)
- Brainwave-inspired cognitive states (Vitalis)
- Evidentiality-based type safety (no hallucination leaks)
- Full video generation (SVD, AnimateDiff, CogVideoX, Mochi)
- Professional enhancement pipelines

**Dantalion** serves multiple roles:
1. **Observability:** OpenTelemetry, Prometheus, structured logging
2. **Image Generation:** SDXL service with MCP integration
3. **Game Engine Bridge:** Asset pipeline for Aether

Both systems are deeply integrated, with Dantalion providing observability INTO Infernum while also serving as the image generation arm of the ecosystem.
