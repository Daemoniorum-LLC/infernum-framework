# Infernum-Sigil

> Complete multimodal AI inference framework in Sigil (~60,000 LOC)

## Overview

Infernum-Sigil is a comprehensive AI inference framework written entirely in [Sigil](https://github.com/Daemoniorum-LLC/sigil-lang), demonstrating the language's expressive power for machine learning workloads. Originally a port of Rust-based LLM inference code, it has grown into a full multimodal platform.

## Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~60,000 |
| **Modules** | 18 |
| **Code Reduction vs Rust** | 43.8% |
| **Image Output** | Up to 8K |
| **Video Output** | Up to 4K @ 60fps |

## Module Architecture

```
src/
├── lib.sigil                    # Main exports
├── core/                        # Core types and infrastructure
│   ├── types.sigil              # ModelId, RequestId, Usage
│   ├── streaming.sigil          # TokenStream, StreamChunk
│   ├── sampling.sigil           # SamplingParams, presets
│   ├── request.sigil            # GenerateRequest, EmbedRequest
│   ├── response.sigil           # GenerateResponse, TokenInfo
│   ├── model.sigil              # ModelMetadata, ModelSource
│   ├── error.sigil              # Error types, Result alias
│   └── multimodal.sigil         # ContentPart, ImageReference
│
├── abaddon/                     # Inference engine
│   ├── sampler.sigil            # Token sampling (top-k, top-p, min-p)
│   ├── kv_cache.sigil           # Paged attention KV cache
│   ├── tokenizer.sigil          # HuggingFace tokenizer wrapper
│   ├── config.sigil             # Engine configuration
│   ├── engine.sigil             # Main inference engine
│   ├── backend.sigil            # Compute backends (CPU, CUDA, Metal)
│   ├── gguf.sigil               # GGUF model loader
│   ├── loader.sigil             # Generic model loading
│   └── models/                  # Model implementations
│       ├── llama.sigil          # Llama 2/3, Mistral, Qwen
│       └── ...
│
├── malphas/                     # Request orchestration
│   ├── router.sigil             # Request routing
│   ├── registry.sigil           # Model registry
│   └── scheduler.sigil          # Batch scheduling
│
├── astaroth/                    # A/B testing framework
│   ├── experiment.sigil         # Experiment definition
│   ├── variant.sigil            # Variant configuration
│   ├── traffic.sigil            # Traffic splitting
│   ├── metrics.sigil            # Metric collection
│   ├── analysis.sigil           # Statistical analysis
│   └── runner.sigil             # Experiment execution
│
├── stolas/                      # RAG and embeddings
│   ├── pipeline.sigil           # RAG pipeline
│   ├── embedder.sigil           # Embedding models
│   ├── store.sigil              # Vector stores
│   ├── chunker.sigil            # Document chunking
│   └── similarity.sigil         # Distance metrics
│
├── beleth/                      # Agent framework
│   ├── agent.sigil              # Agent core
│   ├── tools.sigil              # Tool definitions
│   ├── planner.sigil            # Planning strategies
│   └── memory.sigil             # Agent memory
│
├── asmodeus/                    # Fine-tuning
│   ├── lora.sigil               # LoRA adapters
│   ├── trainer.sigil            # Training loops
│   ├── dataset.sigil            # Dataset handling
│   └── optimizer.sigil          # Optimizers (AdamW)
│
├── dantalion/                   # Observability
│   ├── telemetry.sigil          # Telemetry config
│   ├── metrics.sigil            # Metrics collection
│   └── tracing.sigil            # Distributed tracing
│
├── server/                      # HTTP API server
│   ├── server.sigil             # Server core
│   ├── routes.sigil             # API routes
│   └── openai.sigil             # OpenAI-compatible endpoints
│
├── grimoire_loader/             # Persona management
│   └── loader.sigil             # Prompt loading
│
├── vision/                      # Vision-Language Models
│   ├── encoder.sigil            # Image encoders
│   ├── preprocessor.sigil       # Image preprocessing
│   ├── types.sigil              # Vision types
│   ├── vit.sigil                # Vision Transformer
│   ├── projector.sigil          # Multimodal projector
│   └── vlm.sigil                # VLM integration
│
├── diffusion/                   # Image Generation
│   ├── types.sigil              # Diffusion types
│   ├── vae.sigil                # Variational Autoencoder
│   ├── scheduler.sigil          # Noise schedulers
│   ├── unet.sigil               # U-Net architecture
│   ├── dit.sigil                # Diffusion Transformer
│   ├── text_encoder.sigil       # CLIP/T5 encoders
│   ├── pipeline.sigil           # Generation pipelines
│   ├── lora.sigil               # Diffusion LoRA
│   ├── controlnet.sigil         # ControlNet conditioning
│   ├── editing.sigil            # Inpaint/outpaint/img2img
│   └── enhancement.sigil        # Super-res, face restore, color
│
├── video/                       # Video Generation
│   ├── types.sigil              # Video types
│   ├── vae3d.sigil              # 3D VAE
│   ├── temporal.sigil           # Temporal attention
│   ├── unet3d.sigil             # 3D U-Net
│   ├── transformer.sigil        # Video transformer
│   ├── pipeline.sigil           # Video pipelines
│   └── enhancement.sigil        # Temporal, interpolation, VSR
│
├── optimization/                # Performance
│   ├── config.sigil             # Optimization config
│   ├── attention.sigil          # Flash/memory-efficient attention
│   ├── memory.sigil             # Tiled VAE, offloading
│   ├── quantization.sigil       # FP8/INT8/INT4/NF4
│   ├── cache.sigil              # Prompt/latent caching
│   ├── distilled.sigil          # LCM/Turbo/Lightning
│   └── compile.sigil            # ONNX/TensorRT export
│
├── bench/                       # Benchmarking
│   ├── config.sigil             # Benchmark configuration
│   ├── runner.sigil             # Benchmark runner
│   └── report.sigil             # Report generation
│
├── native/                      # Native FFI bindings
│   ├── ffi.sigil                # Dynamic library loading
│   ├── cuda.sigil               # CUDA runtime API
│   ├── kernels.sigil            # Kernel wrappers
│   ├── memory.sigil             # Memory allocators
│   ├── tensorrt.sigil           # TensorRT integration
│   ├── fused_kernels.sigil      # Fused CUDA kernels
│   ├── cuda_graph.sigil         # CUDA graph capture
│   ├── engine.sigil             # High-perf engine
│   └── distributed.sigil        # Multi-GPU (NCCL)
│
├── compat/                      # Ecosystem compatibility
│   ├── pytorch.sigil            # PyTorch loader
│   ├── diffusers.sigil          # Diffusers compatibility
│   ├── safetensors.sigil        # SafeTensors format
│   ├── huggingface.sigil        # HuggingFace Hub
│   ├── gguf.sigil               # GGUF format
│   └── onnx.sigil               # ONNX format
│
└── presets.sigil                # Quality presets
```

## Module Descriptions

### Core Infrastructure

#### `core` — Foundation Types
Core types with Sigil's evidentiality markers for tracking data provenance.

```sigil
// Untrusted LLM output marked with ~
struct GenerateResponse {
    choices~: [Choice],  // External data
    usage: Usage,
}

// Verified data marked with !
fn validate(resp~: GenerateResponse) -> ValidatedResponse! {
    resp~|validate!{ choices: non_empty }
}
```

#### `abaddon` — Inference Engine
High-performance LLM inference with paged attention and multiple backends.

```sigil
let engine = InferenceEngine::new(config)?
    .load_model("meta-llama/Llama-3-8B")?

let response = engine.generate(request)?
```

#### `malphas` — Request Orchestration
Load balancing, model registry, and batch scheduling.

```sigil
let router = RequestRouter::new()
    .with_strategy(RoutingStrategy::LeastLoaded)
    .register(model_a)
    .register(model_b)
```

### A/B Testing & Experimentation

#### `astaroth` — Experimentation Framework
Full A/B testing with statistical analysis.

```sigil
let experiment = Experiment::new("model-comparison")
    .add_variant(Variant::control("gpt-4"))
    .add_variant(Variant::treatment("llama-70b"))
    .with_primary_metric(MetricType::Latency, lower_is_better: true)

let result = runner.analyze(confidence: 0.95)?
```

### RAG & Agents

#### `stolas` — RAG Pipeline
Retrieval-augmented generation with vector stores.

```sigil
let rag = RagPipeline::new()
    .with_embedder(SentenceEmbedder::new(model)?)
    .with_store(InMemoryStore::new())

let context = rag.retrieve(query, top_k: 5)?
```

#### `beleth` — Agent Framework
Tool-using agents with planning and memory.

```sigil
let agent = Agent::builder()
    .with_tools([Tool::web_search(), Tool::calculator()])
    .with_planner(LLMPlanner::new(engine)?)
    .build()?
```

### Multimodal Generation

#### `vision` — Vision-Language Models
Support for LLaVA, Qwen2-VL, InternVL, and more.

```sigil
let vlm = VisionLanguageModel::new(VLMConfig {
    architecture: VLMArchitecture::LLaVA,
    vision_encoder: VisionConfig::clip_vit_large(),
})?

let response = vlm.generate(
    image: ImageInput::from_path("photo.jpg"),
    prompt: "Describe this image",
)?
```

#### `diffusion` — Image Generation
Full diffusion pipeline for SD 1.5, SDXL, SD3, and Flux.

```sigil
let pipeline = FluxPipeline::new("black-forest-labs/FLUX.1-dev")?

let image = pipeline.generate(
    prompt: "A serene mountain landscape at sunset",
    width: 1024,
    height: 1024,
    num_steps: 28,
)?
```

#### `video` — Video Generation
Video synthesis with SVD, AnimateDiff, CogVideoX, Mochi.

```sigil
let pipeline = MochiPipeline::new("genmo/mochi-1-preview")?

let video = pipeline.generate(
    prompt: "Waves crashing on a rocky shore",
    num_frames: 121,
    fps: 24,
)?
```

### Enhancement & Quality

#### `diffusion/enhancement` — Image Enhancement
Professional post-processing for maximum fidelity.

| Component | Models | Purpose |
|-----------|--------|---------|
| Super-Resolution | RealESRGAN, SwinIR, ESRGAN | Up to 8K output |
| Face Restoration | GFPGAN, CodeFormer, RestoreFormer | Portrait enhancement |
| Color Grading | Lift/Gamma/Gain, 3D LUT | Cinematic color |
| Film Grain | Configurable grain | Photographic texture |

```sigil
let enhancer = EnhancementPipeline::new()
    .with_super_res(SuperResModel::SwinIRLarge, scale: 4)?
    .with_face_restore(FaceRestoreModel::CodeFormer)?
    .with_color_grade(cinematic_lut())?
    .with_film_grain(intensity: 0.06)?
```

#### `video/enhancement` — Video Enhancement
Temporal consistency and quality enhancement.

| Component | Models | Purpose |
|-----------|--------|---------|
| Temporal Processor | Optical Flow, Deflicker | Consistency |
| Frame Interpolation | RIFE, FILM, IFRNet | Up to 8× slow-mo |
| Video SR | BasicVSR++, RealBasicVSR | 4K upscaling |

```sigil
let enhancer = VideoEnhancePipeline::new()
    .with_temporal_consistency(weight: 0.7)?
    .with_interpolation(InterpolationModel::FILM, target_fps: 60)?
    .with_super_res(VideoSRModel::RealBasicVSR, scale: 2)?
```

### Performance & Optimization

#### `optimization` — Performance Tuning
Flash attention, quantization, caching, and compilation.

```sigil
let config = DiffusionOptimizationConfig {
    attention: AttentionBackend::FlashAttentionV2,
    quantization: Some(QuantizationMethod::FP8),
    vae_tiling: true,
    compile: true,
}
```

#### `native` — Zero-Overhead FFI
Direct CUDA kernel access and multi-GPU support.

```sigil
// Multi-GPU with NCCL
let engine = MultiGPUEngine::new(MultiGPUConfig {
    strategy: ParallelismStrategy::Hybrid,
    device_ids: vec![0, 1, 2, 3],
    tensor_parallel_size: 2,
    pipeline_parallel_size: 2,
})?
```

#### `bench` — Benchmarking Suite
Comprehensive benchmarks across all modules.

```sigil
let config = BenchmarkConfig::default()
    .with_categories([
        BenchmarkCategory::Attention,
        BenchmarkCategory::Diffusion,
        BenchmarkCategory::VideoGeneration,
    ])

run_all_benchmarks(config)?
```

### Ecosystem Compatibility

#### `compat` — Format Support
Load models from any popular format.

| Format | Extension | Notes |
|--------|-----------|-------|
| PyTorch | .pt, .pth, .bin | State dict loading |
| SafeTensors | .safetensors | Recommended format |
| GGUF | .gguf | llama.cpp quantized |
| ONNX | .onnx | Cross-platform |
| Diffusers | folder | HuggingFace pipelines |

### Quality Presets

#### `presets` — Easy Configuration
Pre-configured quality levels for optimal results.

| Preset | Base Resolution | Steps | Enhancement | VRAM |
|--------|-----------------|-------|-------------|------|
| Draft | 512×512 | 15 | None | ~2GB |
| Standard | 768×768 | 25 | 2× upscale | ~6GB |
| High | 1024×1024 | 35 | 4× + face | ~12GB |
| Ultra | 1024×1024 | 50 | Full pipeline | ~24GB |
| Cinematic | 1536×1536 | 60 | + Film grain | ~24GB+ |

```sigil
// One-liner for maximum quality
let preset = ImagePreset::cinematic().portrait_mode().output_4k()
```

## Sigil Language Features

### Morpheme Pipes
Composable data transformation chains.

```sigil
// τ = transform, φ = filter, ⋳ = flatten
let text = chunks
    |τ{_.choices~}
    |⋳
    |φ{_.is_some}
    |τ{_.unwrap}
    |·join("")
```

### Evidentiality Types
Track data provenance in the type system.

```sigil
// ~ = untrusted/external (reported)
// ! = verified/computed (known)
// ? = uncertain/optional (inferred)

struct Response {
    content~: str,     // LLM output - untrusted
    validated!: bool,  // Our computation - verified
    cached?: Option,   // Maybe exists - uncertain
}
```

### Inline Defaults
Eliminate boilerplate Default implementations.

```sigil
struct Config {
    temperature: f32 = 0.7,
    max_tokens: u32 = 2048,
    top_p: f32 = 0.9,
}
```

## Running

```bash
# Interpret
sigil run src/lib.sigil

# JIT compile (67x faster than interpretation)
sigil jit src/lib.sigil

# AOT compile (production)
sigil compile src/lib.sigil -o infernum --release

# Run tests
sigil test

# Format code
sigil-glyph .
```

## Hardware Requirements

| Workload | Minimum VRAM | Recommended |
|----------|--------------|-------------|
| LLM (7B) | 8GB | 16GB |
| LLM (70B) | 48GB | 80GB |
| SDXL | 8GB | 12GB |
| SD3/Flux | 16GB | 24GB |
| Video (SVD) | 16GB | 24GB |
| Video (Mochi) | 40GB | 80GB |

## Development History

| Phase | Description | LOC |
|-------|-------------|-----|
| 1 | Core types (vs Rust 1,582 LOC) | ~940 |
| 2 | Abaddon engine | ~2,500 |
| 3 | Malphas, Stolas, Beleth | ~5,000 |
| 4 | Astaroth, Asmodeus, Dantalion | ~6,000 |
| 5 | Vision, Diffusion, Video | ~25,000 |
| 6 | Optimization, Native, Compat | ~15,000 |
| 7 | Enhancement, Presets | ~5,000 |
| **Total** | **18 modules** | **~60,000** |

## Related

- [Sigil Language](https://github.com/Daemoniorum-LLC/sigil-lang)
- [sigil-parser](https://crates.io/crates/sigil-parser)
- [Persona Framework](https://github.com/Daemoniorum-LLC/persona-framework)

## License

MIT — Daemoniorum LLC
