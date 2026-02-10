# Getting Started with Infernum

This guide walks you through setting up and using Infernum for various AI tasks.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Text Generation](#text-generation)
4. [Image Generation](#image-generation)
5. [Video Generation](#video-generation)
6. [Enhancement Pipelines](#enhancement-pipelines)
7. [Using Presets](#using-presets)
8. [Multi-GPU Setup](#multi-gpu-setup)
9. [Production Deployment](#production-deployment)

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.0+ (for GPU inference)
- 8GB+ VRAM (for basic LLM inference)
- 16GB+ VRAM (for image/video generation)
- Linux or macOS (Windows support experimental)

### Install Sigil

```bash
# Install Sigil language
curl -sSf https://sigil-lang.dev/install.sh | sh

# Verify installation
sigil --version
```

### Clone Repository

```bash
git clone https://github.com/Daemoniorum-LLC/infernum-complete.git
cd infernum-complete
```

### Build

```bash
# Development build (faster compilation)
sigil build

# Release build (optimized)
sigil build --release
```

---

## Quick Start

### Hello World: Text Generation

```sigil
use infernum::{Engine, EngineConfig, GenerateRequest}

fn main() -> Result<(), Error> {
    // Create engine with default configuration
    let engine = Engine::new(EngineConfig::default())?
        .load_model("meta-llama/Llama-3-8B-Instruct")?

    // Generate text
    let response = engine.generate(GenerateRequest {
        prompt: "Write a haiku about programming:",
        max_tokens: 50,
        temperature: 0.7,
    })?

    println!("{}", response.choices[0].text)
    Ok(())
}
```

Run it:

```bash
sigil run examples/hello_world.sigil
```

### Hello World: Image Generation

```sigil
use infernum::{FluxPipeline, GenerationRequest}

fn main() -> Result<(), Error> {
    let pipeline = FluxPipeline::new("black-forest-labs/FLUX.1-schnell")?

    let images = pipeline.generate(GenerationRequest {
        prompt: "A cute robot learning to paint",
        width: 1024,
        height: 1024,
        num_steps: 4,  // Schnell is distilled, needs few steps
    })?

    images[0].save("robot_painting.png")?
    println!("Image saved!")
    Ok(())
}
```

---

## Text Generation

### Basic Generation

```sigil
use infernum::{Engine, EngineConfig, GenerateRequest}

let engine = Engine::new(EngineConfig {
    device: "cuda",
    dtype: "bf16",
    max_seq_len: 4096,
})?
.load_model("meta-llama/Llama-3-8B-Instruct")?

let response = engine.generate(GenerateRequest {
    prompt: "Explain quantum computing to a 5 year old",
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.9,
})?

println!("{}", response.choices[0].text)
```

### Chat Format

```sigil
use infernum::{Engine, ChatMessage, PromptInput}

let messages = [
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("What is the capital of France?"),
]

let response = engine.generate(GenerateRequest {
    prompt: PromptInput::Messages(messages),
    max_tokens: 100,
})?
```

### Streaming

```sigil
use infernum::{Engine, GenerateRequest}

let stream = engine.generate_stream(GenerateRequest {
    prompt: "Write a story about a brave knight",
    max_tokens: 500,
    stream: true,
})?

// Process tokens as they arrive
for chunk in stream {
    print!("{}", chunk.text)
    flush()
}
```

### Sampling Parameters

```sigil
// Creative writing (high temperature)
let creative = GenerateRequest {
    prompt: "Once upon a time...",
    temperature: 1.2,
    top_p: 0.95,
    top_k: 50,
}

// Factual/deterministic (low temperature)
let factual = GenerateRequest {
    prompt: "The capital of France is",
    temperature: 0.1,
    top_p: 1.0,
}

// Reproducible generation
let reproducible = GenerateRequest {
    prompt: "Generate a random number:",
    seed: Some(42),
}
```

---

## Image Generation

### Stable Diffusion XL

```sigil
use infernum::{SDXLPipeline, GenerationRequest}

let pipeline = SDXLPipeline::new("stabilityai/stable-diffusion-xl-base-1.0")?

let images = pipeline.generate(GenerationRequest {
    prompt: "A majestic mountain landscape at golden hour, photorealistic",
    negative_prompt: Some("blurry, low quality, distorted"),
    width: 1024,
    height: 1024,
    num_steps: 30,
    guidance_scale: 7.5,
})?

images[0].save("landscape.png")?
```

### Flux (Best Quality)

```sigil
use infernum::{FluxPipeline, GenerationRequest}

// Flux Dev - highest quality, 28-50 steps
let dev = FluxPipeline::new("black-forest-labs/FLUX.1-dev")?

let image = dev.generate(GenerationRequest {
    prompt: "Portrait of an astronaut in a garden, detailed, 8k",
    width: 1024,
    height: 1024,
    num_steps: 28,
    guidance_scale: 3.5,  // Flux uses lower guidance
})?

// Flux Schnell - fast, 1-4 steps (distilled)
let schnell = FluxPipeline::new("black-forest-labs/FLUX.1-schnell")?

let fast_image = schnell.generate(GenerationRequest {
    prompt: "A red sports car",
    num_steps: 4,
    guidance_scale: 0.0,  // Schnell is guidance-free
})?
```

### With LoRA

```sigil
use infernum::{SDXLPipeline, GenerationRequest}

let pipeline = SDXLPipeline::new("stabilityai/stable-diffusion-xl-base-1.0")?
    .load_lora("path/to/style.safetensors", scale: 0.8)?
    .load_lora("path/to/character.safetensors", scale: 0.6)?

let image = pipeline.generate(GenerationRequest {
    prompt: "a portrait in the style of <lora_trigger>",
})?
```

### ControlNet

```sigil
use infernum::{SDXLPipeline, ControlNet, ControlNetCondition}

let pipeline = SDXLPipeline::new("stabilityai/stable-diffusion-xl-base-1.0")?

let controlnet = ControlNet::new(
    "diffusers/controlnet-canny-sdxl-1.0",
    ControlNetCondition::Canny
)?

let image = pipeline.generate_with_control(
    request: GenerationRequest {
        prompt: "A beautiful house",
    },
    control: controlnet,
    control_image: ImageInput::from_path("house_edges.png"),
    conditioning_scale: 0.8,
)?
```

### Inpainting

```sigil
use infernum::{InpaintPipeline, InpaintConfig}

let pipeline = InpaintPipeline::new("stabilityai/stable-diffusion-xl-inpaint")?

let result = pipeline.inpaint(InpaintConfig {
    image: ImageInput::from_path("photo.png"),
    mask: ImageInput::from_path("mask.png"),
    prompt: "A golden retriever",
    num_steps: 30,
})?
```

---

## Video Generation

### Stable Video Diffusion (Image-to-Video)

```sigil
use infernum::{StableVideoDiffusionPipeline, I2VRequest, ImageInput}

let pipeline = StableVideoDiffusionPipeline::new(
    "stabilityai/stable-video-diffusion-img2vid-xt"
)?

let video = pipeline.generate(I2VRequest {
    image: ImageInput::from_path("starting_frame.png"),
    num_frames: 25,
    fps: 6,
    motion_bucket_id: 127,  // Higher = more motion
})?

video.save_mp4("output.mp4")?
```

### CogVideoX (Text-to-Video)

```sigil
use infernum::{CogVideoXPipeline, T2VRequest}

let pipeline = CogVideoXPipeline::new("THUDM/CogVideoX-5b")?

let video = pipeline.generate(T2VRequest {
    prompt: "A timelapse of a flower blooming",
    num_frames: 49,
    guidance_scale: 6.0,
})?

video.save_mp4("flower_bloom.mp4")?
```

### Mochi (Highest Quality)

```sigil
use infernum::{MochiPipeline, T2VRequest}

let pipeline = MochiPipeline::new("genmo/mochi-1-preview")?

let video = pipeline.generate(T2VRequest {
    prompt: "Waves crashing on a rocky coastline at sunset",
    num_frames: 84,
})?

video.save_mp4("waves.mp4")?
```

---

## Enhancement Pipelines

### Image Enhancement

```sigil
use infernum::{
    EnhancementPipeline,
    SuperResModel,
    FaceRestoreModel,
    ColorGradeConfig,
    FilmGrainConfig,
}

// Build enhancement pipeline
let enhancer = EnhancementPipeline::new()
    // 4x upscaling with SwinIR
    .with_super_res(SuperResModel::SwinIRLarge, scale: 4)?
    // Face restoration for portraits
    .with_face_restore(FaceRestoreModel::CodeFormer, fidelity: 0.85)?
    // Cinematic color grading
    .with_color_grade(ColorGradeConfig {
        contrast: 1.08,
        saturation: 0.98,
        lift: [0.0, 0.0, 0.02],     // Cool shadows
        gain: [1.02, 1.0, 0.98],    // Warm highlights
    })?
    // Subtle film grain
    .with_film_grain(FilmGrainConfig {
        intensity: 0.06,
        size: 1.2,
    })?

// Apply to generated image
let enhanced = enhancer.process(image)?
enhanced.save("enhanced_8k.png")?
```

### Video Enhancement

```sigil
use infernum::{
    VideoEnhancePipeline,
    VideoSRModel,
    InterpolationModel,
}

let enhancer = VideoEnhancePipeline::new()
    // Temporal consistency for smoother video
    .with_temporal_consistency(weight: 0.7)?
    // Frame interpolation for 60fps
    .with_interpolation(InterpolationModel::FILM, target_fps: 60)?
    // 2x upscaling to 4K
    .with_super_res(VideoSRModel::RealBasicVSR, scale: 2)?
    // Cinematic color grading
    .with_color_grade(cinematic_preset())?

let enhanced = enhancer.process(video)?
enhanced.save_mp4("enhanced_4k_60fps.mp4")?
```

### Slow Motion

```sigil
use infernum::{VideoEnhancePipeline, InterpolationModel}

// 4x slow motion
let enhancer = VideoEnhancePipeline::new()
    .with_interpolation(
        InterpolationModel::FILM,
        target_fps: video.fps * 4  // 24fps -> 96fps
    )?

let slowmo = enhancer.process(video)?
slowmo.save_mp4("slowmo.mp4", fps: 24)?  // Play at 24fps = 4x slower
```

---

## Using Presets

### Image Presets

```sigil
use infernum::{ImagePreset, QualityPreset, FluxPipeline}

// Quick draft preview
let draft = ImagePreset::draft()
// 512x512, 15 steps, no enhancement, ~2GB VRAM

// Standard quality
let standard = ImagePreset::standard()
// 768x768, 25 steps, 2x upscale, ~6GB VRAM

// High quality
let high = ImagePreset::high()
// 1024x1024, 35 steps, 4x upscale + face restore, ~12GB VRAM

// Maximum quality
let ultra = ImagePreset::ultra()
// 1024x1024, 50 steps, SwinIR + CodeFormer + color, ~24GB VRAM

// Cinematic film-grade
let cinematic = ImagePreset::cinematic()
// 1536x1536, 60 steps, full pipeline + grain, ~24GB+ VRAM

// Apply preset to pipeline
let pipeline = FluxPipeline::new("black-forest-labs/FLUX.1-dev")?
    .with_config(cinematic)?
```

### Specialized Modes

```sigil
// Portrait photography
let portrait = ImagePreset::cinematic()
    .portrait_mode()  // Optimized face restore, soft detail

// Landscape photography
let landscape = ImagePreset::ultra()
    .landscape_mode()  // Enhanced detail, no face restore

// Target 4K output
let hires = ImagePreset::high()
    .output_4k()  // Progressive upscaling to 4K

// Target 8K output
let ultra_hires = ImagePreset::cinematic()
    .output_8k()  // Multi-pass upscaling to 8K
```

### Video Presets

```sigil
use infernum::{VideoPreset, QualityPreset, MochiPipeline}

// Quick preview
let draft = VideoPreset::draft()
// 512x512, 16 frames, no enhancement

// Production quality
let cinematic = VideoPreset::cinematic()
// 1080p, 121 frames, temporal + color + grain

// Apply to pipeline
let pipeline = MochiPipeline::new("genmo/mochi-1-preview")?
    .with_config(VideoPreset::ultra())?
```

---

## Multi-GPU Setup

### Basic Multi-GPU

```sigil
use infernum::{MultiGPUEngine, MultiGPUConfig, ParallelismStrategy}

let engine = MultiGPUEngine::new(MultiGPUConfig {
    strategy: ParallelismStrategy::Data,
    device_ids: vec![0, 1],  // Use GPU 0 and 1
})?

// Requests automatically distributed across GPUs
let response = engine.generate(request)?
```

### Tensor Parallelism (Large Models)

```sigil
// For models that don't fit on a single GPU
let engine = MultiGPUEngine::new(MultiGPUConfig {
    strategy: ParallelismStrategy::Tensor,
    device_ids: vec![0, 1, 2, 3],
    tensor_parallel_size: 4,  // Split across 4 GPUs
})?

// 70B model split across 4 GPUs
engine.load_model("meta-llama/Llama-3-70B")?
```

### Pipeline Parallelism

```sigil
// For maximum throughput
let engine = MultiGPUEngine::new(MultiGPUConfig {
    strategy: ParallelismStrategy::Pipeline,
    device_ids: vec![0, 1, 2, 3],
    pipeline_parallel_size: 4,
})?
```

### Hybrid Parallelism

```sigil
// Best of both worlds
let engine = MultiGPUEngine::new(MultiGPUConfig {
    strategy: ParallelismStrategy::Hybrid,
    device_ids: vec![0, 1, 2, 3, 4, 5, 6, 7],
    tensor_parallel_size: 2,    // 2-way tensor parallel
    pipeline_parallel_size: 4,  // 4-way pipeline parallel
})?
```

---

## Production Deployment

### HTTP API Server

```sigil
use infernum::server::{Server, ServerConfig}

let config = ServerConfig {
    host: "0.0.0.0",
    port: 8080,
    models: ["meta-llama/Llama-3-8B-Instruct"],
    max_concurrent: 32,
    timeout_ms: 30000,
}

Server::new(config)?.run()?
```

Access via standard `/v1/*` endpoints:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Optimizations

```sigil
use infernum::optimization::{
    DiffusionOptimizationConfig,
    AttentionBackend,
    QuantizationMethod,
}

let config = DiffusionOptimizationConfig {
    // Use Flash Attention
    attention: AttentionBackend::FlashAttentionV2,

    // FP8 quantization (H100/RTX 4090)
    quantization: Some(QuantizationMethod::FP8),

    // Tiled VAE for large images
    vae_tiling: true,

    // CPU offloading for low VRAM
    cpu_offload: false,

    // Compile for faster inference
    compile: true,
}
```

### Caching

```sigil
use infernum::optimization::{PromptCache, LatentCache}

// Cache prompt embeddings
let prompt_cache = PromptCache::new(PromptCacheConfig {
    max_entries: 1000,
    ttl_seconds: 3600,
})

// Cache latents for similar prompts
let latent_cache = LatentCache::new(max_entries: 100)
```

### Monitoring

```sigil
use infernum::dantalion::{Telemetry, MetricsCollector}

// Initialize telemetry
let telemetry = Telemetry::new(TelemetryConfig {
    metrics_port: 9090,
    tracing_endpoint: "http://jaeger:14268",
})?

// Access metrics
let metrics = MetricsCollector::global()
println!("Requests/sec: {}", metrics.requests_per_second())
println!("Avg latency: {}ms", metrics.avg_latency_ms())
```

---

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- Explore [example code](../examples/) for more use cases
- Check [benchmarks](../benchmarks/) for performance tuning
- Join the [Discord](https://discord.gg/sigil) for community support

---

## Troubleshooting

### Out of Memory

```sigil
// Enable memory optimizations
let config = EngineConfig {
    // Use FP16 instead of FP32
    dtype: "f16",

    // Enable flash attention
    flash_attention: true,

    // Reduce batch size
    max_batch_size: 1,
}

// For diffusion
let config = DiffusionOptimizationConfig {
    // Enable VAE tiling
    vae_tiling: true,

    // Enable attention slicing
    attention_slicing: Some(1),

    // CPU offload (slower but works)
    cpu_offload: true,
}
```

### CUDA Errors

```bash
# Check CUDA installation
nvidia-smi

# Verify Sigil can see GPU
sigil run -c "print(cuda::device_count())"
```

### Model Loading Failures

```sigil
// Check model exists
let exists = HfApi::model_exists("meta-llama/Llama-3-8B")?

// Download with progress
let model = HfModelLoader::load_with_progress("model-id", |progress| {
    println!("Downloading: {}%", progress * 100.0)
})?
```
