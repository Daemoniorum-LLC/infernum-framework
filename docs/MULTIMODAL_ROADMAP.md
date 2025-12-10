# Multimodal Generation Roadmap

**Status:** Planning
**Target:** Full image/video generation and vision-language model support in Sigil

---

## Executive Summary

This roadmap outlines the path to transform Infernum from a pure LLM inference framework into a comprehensive multimodal AI platform supporting:
- **Vision-Language Models** (VLMs) - Understanding images in context
- **Image Generation** - Diffusion-based image synthesis
- **Video Generation** - Temporal coherent video synthesis

---

## Phase 1: Vision-Language Foundation

**Goal:** Enable models to understand and reason about images

### 1.1 Image Encoder Infrastructure

```sigil
// New module: infernum-sigil/src/abaddon/vision/
pub mod vision

// Core types
pub struct ImageEncoder {
    architecture: VisionArchitecture,
    patch_size: u32,
    hidden_size: u32,
    num_layers: u32,
}

pub enum VisionArchitecture {
    CLIP,           // OpenAI CLIP ViT
    SigLIP,         // Google SigLIP
    EVA,            // EVA-CLIP
    InternViT,      // InternVL
    Qwen2ViT,       // Qwen2-VL vision encoder
}
```

**Tasks:**
- [ ] Implement Vision Transformer (ViT) base architecture
- [ ] Add patch embedding layer with configurable patch sizes (14x14, 16x16)
- [ ] Implement positional embeddings (learned, RoPE-2D, NaViT)
- [ ] Add image preprocessing pipeline (resize, normalize, pad)
- [ ] Support dynamic resolution (any aspect ratio)

### 1.2 Multimodal Input Handling

```sigil
// Extend request types
pub enum PromptInput {
    Text(str~),
    Messages([Message]~),
    Tokens([u32]),
    // NEW: Multimodal inputs
    Multimodal(MultimodalInput~),
}

pub struct MultimodalInput {
    content~: [ContentPart],
}

pub enum ContentPart {
    Text(str~),
    Image(ImageInput),
    Video(VideoInput),
}

pub struct ImageInput {
    source: ImageSource,
    detail: ImageDetail,  // low, high, auto
}

pub enum ImageSource {
    Base64(str~),
    Url(str~),
    Path(str~),
    Tensor(Tensor),
}
```

**Tasks:**
- [ ] Extend `PromptInput` enum with multimodal variant
- [ ] Implement image loading from multiple sources
- [ ] Add image preprocessing with configurable detail levels
- [ ] Support interleaved text-image sequences
- [ ] Implement image tiling for high-resolution inputs

### 1.3 Vision-Language Model Support

| Model | Architecture | Priority |
|-------|--------------|----------|
| LLaVA-1.6 | CLIP + Llama/Mistral | P0 |
| Qwen2-VL | Qwen2-ViT + Qwen2 | P0 |
| Pixtral | Custom ViT + Mistral | P1 |
| InternVL2 | InternViT + InternLM | P1 |
| LLaVA-OneVision | SigLIP + Qwen2 | P2 |
| Molmo | Custom + OLMo | P2 |

**Tasks:**
- [ ] Implement multimodal projector (MLP, C-Abstractor, Perceiver)
- [ ] Add cross-attention fusion for early fusion models
- [ ] Support image token injection strategies
- [ ] Implement anyres (dynamic resolution) support
- [ ] Add video frame sampling for video-capable VLMs

---

## Phase 2: Image Generation (Diffusion)

**Goal:** Generate images from text descriptions

### 2.1 Diffusion Core Infrastructure

```sigil
// New module: infernum-sigil/src/diffusion/
pub mod diffusion

pub struct DiffusionPipeline {
    text_encoder: TextEncoder,
    vae: VAE,
    unet: UNet,           // or DiT
    scheduler: Scheduler,
}

pub struct VAE {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    latent_channels: u32,  // typically 4
    scaling_factor: f32,   // 0.18215 for SD
}

pub enum NoiseScheduler {
    DDPM,
    DDIM,
    EulerDiscrete,
    EulerAncestral,
    DPMSolverMultistep,
    UniPC,
    FlowMatch,  // For newer models
}
```

**Tasks:**
- [ ] Implement Variational Autoencoder (VAE) encoder/decoder
- [ ] Add latent space operations
- [ ] Implement multiple noise schedulers
- [ ] Support classifier-free guidance (CFG)
- [ ] Add negative prompting

### 2.2 U-Net / DiT Architecture

```sigil
pub enum DiffusionBackbone {
    UNet2D(UNet2DConfig),           // Classic SD
    UNetXL(UNetXLConfig),           // SDXL
    DiT(DiTConfig),                 // Diffusion Transformer
    MMDiT(MMDiTConfig),             // SD3/Flux style
}

pub struct UNet2DConfig {
    in_channels: u32,
    out_channels: u32,
    block_out_channels: [u32],
    layers_per_block: u32,
    cross_attention_dim: u32,
    attention_head_dim: u32,
}

pub struct DiTConfig {
    hidden_size: u32,
    num_layers: u32,
    num_heads: u32,
    patch_size: u32,
    in_channels: u32,
}
```

**Tasks:**
- [ ] Implement ResNet blocks with GroupNorm
- [ ] Add cross-attention layers for text conditioning
- [ ] Implement self-attention with memory-efficient attention
- [ ] Add time embedding injection
- [ ] Implement DiT blocks for transformer-based diffusion
- [ ] Add MMDiT joint attention for SD3/Flux architecture

### 2.3 Text Encoders for Diffusion

| Encoder | Used By | Dimension |
|---------|---------|-----------|
| CLIP ViT-L/14 | SD 1.x | 768 |
| OpenCLIP ViT-H/14 | SD 2.x | 1024 |
| CLIP + OpenCLIP | SDXL | 768 + 1280 |
| T5-XXL | SD3, Flux | 4096 |
| CLIP + T5 | SD3 | Multiple |

**Tasks:**
- [ ] Implement CLIP text encoder
- [ ] Add OpenCLIP support
- [ ] Implement T5 encoder for SD3/Flux
- [ ] Support dual/triple text encoder pipelines
- [ ] Add prompt weighting and long prompt handling

### 2.4 Supported Models

| Model | Type | Priority | Notes |
|-------|------|----------|-------|
| Stable Diffusion 1.5 | UNet | P1 | Legacy, good baseline |
| SDXL | UNet | P0 | Current standard |
| Stable Diffusion 3 | MMDiT | P0 | State-of-art quality |
| FLUX.1 | MMDiT | P0 | Best open model |
| Playground v2.5 | UNet | P2 | Aesthetic focus |
| PixArt-Σ | DiT | P1 | Efficient DiT |

---

## Phase 3: Advanced Image Generation

**Goal:** Production-ready image generation with fine-tuning

### 3.1 LoRA & Fine-tuning for Diffusion

```sigil
pub struct DiffusionLoRAConfig {
    target_modules: [str~],  // ["to_q", "to_k", "to_v", "to_out"]
    rank: u32,
    alpha: f32,
    text_encoder_lora: bool,
    unet_lora: bool,
}

pub struct DreamBoothConfig {
    instance_prompt: str~,
    class_prompt: str~,
    prior_preservation: bool,
    prior_loss_weight: f32,
}
```

**Tasks:**
- [ ] Implement LoRA for U-Net attention layers
- [ ] Add LoRA for text encoders
- [ ] Support multiple LoRA merging
- [ ] Implement DreamBooth training
- [ ] Add Textual Inversion support

### 3.2 ControlNet & Conditioning

```sigil
pub enum ControlNetCondition {
    Canny,
    Depth,
    Pose,
    Segmentation,
    Scribble,
    Tile,
    IPAdapter,
    Reference,
}

pub struct ControlNetConfig {
    condition_type: ControlNetCondition,
    conditioning_scale: f32,
    guess_mode: bool,
}
```

**Tasks:**
- [ ] Implement ControlNet architecture
- [ ] Add preprocessing for each condition type
- [ ] Support multi-ControlNet
- [ ] Implement IP-Adapter for image prompting
- [ ] Add reference-only mode

### 3.3 Image Editing

```sigil
pub enum ImageEditOperation {
    Inpaint(InpaintConfig),
    Outpaint(OutpaintConfig),
    Img2Img(Img2ImgConfig),
    Upscale(UpscaleConfig),
}

pub struct InpaintConfig {
    mask: Tensor,
    mask_blur: u32,
    inpaint_full_res: bool,
}
```

**Tasks:**
- [ ] Implement inpainting with mask support
- [ ] Add outpainting (canvas extension)
- [ ] Img2img with strength parameter
- [ ] Upscaling with tile-based processing
- [ ] Implement SDXL Refiner pipeline

---

## Phase 4: Video Generation

**Goal:** Generate temporally coherent video from text/images

### 4.1 Video Diffusion Architecture

```sigil
pub struct VideoDiffusionPipeline {
    text_encoder: TextEncoder,
    vae_3d: VAE3D,
    unet_3d: UNet3D,  // or VideoTransformer
    scheduler: Scheduler,
    temporal_attention: TemporalAttention,
}

pub struct VideoConfig {
    num_frames: u32,
    fps: u32,
    width: u32,
    height: u32,
    temporal_downsample: u32,
}
```

**Tasks:**
- [ ] Implement 3D VAE (temporal + spatial compression)
- [ ] Add temporal attention layers
- [ ] Implement motion modules
- [ ] Add frame interpolation
- [ ] Support video conditioning (first frame, keyframes)

### 4.2 Supported Video Models

| Model | Type | Priority | Notes |
|-------|------|----------|-------|
| AnimateDiff | Motion Module | P1 | SD1.5 base |
| Stable Video Diffusion | Image-to-Video | P0 | Best I2V |
| CogVideoX | DiT | P1 | Long videos |
| Mochi | DiT | P0 | High quality |
| LTX-Video | Custom | P1 | Fast inference |

### 4.3 Video Features

**Tasks:**
- [ ] Image-to-video generation
- [ ] Text-to-video generation
- [ ] Video extension (continue existing video)
- [ ] Motion brush / trajectory control
- [ ] Multi-shot generation with consistency
- [ ] Video upscaling

---

## Phase 5: Optimization & Production

### 5.1 Performance Optimizations

```sigil
pub struct DiffusionOptimizationConfig {
    // Attention
    flash_attention: bool,
    memory_efficient_attention: bool,

    // Quantization
    unet_quantization: Option<QuantizationMethod>,
    vae_tiling: bool,
    vae_slicing: bool,

    // Speed
    compile: bool,  // torch.compile equivalent
    channels_last: bool,

    // Memory
    cpu_offload: bool,
    sequential_offload: bool,
    attention_slicing: Option<u32>,
}
```

**Tasks:**
- [ ] Implement Flash Attention for diffusion
- [ ] Add VAE tiling for large images
- [ ] Support model CPU offloading
- [ ] Implement attention slicing
- [ ] Add FP8 quantization for diffusion
- [ ] Optimize for batch generation

### 5.2 Caching & Acceleration

**Tasks:**
- [ ] Implement prompt embedding caching
- [ ] Add TensorRT/ONNX export
- [ ] Support distilled models (LCM, Turbo, Lightning)
- [ ] Implement consistency models
- [ ] Add speculative decoding for diffusion (parallel denoising)

---

## Implementation Timeline

```
Phase 1: Vision-Language Foundation
├── 1.1 Image Encoder Infrastructure
├── 1.2 Multimodal Input Handling
└── 1.3 VLM Support (LLaVA, Qwen2-VL)

Phase 2: Image Generation Core
├── 2.1 Diffusion Infrastructure
├── 2.2 U-Net / DiT Architecture
├── 2.3 Text Encoders
└── 2.4 SDXL + Flux Support

Phase 3: Advanced Image Features
├── 3.1 LoRA & Fine-tuning
├── 3.2 ControlNet
└── 3.3 Image Editing

Phase 4: Video Generation
├── 4.1 Video Diffusion
├── 4.2 SVD + Mochi Support
└── 4.3 Advanced Video Features

Phase 5: Optimization
├── 5.1 Performance Optimization
└── 5.2 Production Acceleration
```

---

## API Design Preview

### Vision-Language (VLM)

```sigil
let engine = Engine::new()
    .with_model("llava-1.6-34b")
    .with_vision(VisionConfig::default())
    .build()?

let response = engine.generate(
    MultimodalInput::new()
        .add_image(ImageInput::from_path("photo.jpg"))
        .add_text("Describe this image in detail.")
)?
```

### Image Generation

```sigil
let pipeline = DiffusionPipeline::new()
    .with_model("stabilityai/stable-diffusion-3-medium")
    .with_scheduler(NoiseScheduler::FlowMatch)
    .build()?

let images = pipeline.generate(
    prompt: "A serene lake at sunset, mountains in background, photorealistic",
    negative_prompt: "blurry, low quality",
    num_images: 4,
    guidance_scale: 7.5,
    num_inference_steps: 28,
    width: 1024,
    height: 1024,
)?
```

### Video Generation

```sigil
let video_pipeline = VideoDiffusionPipeline::new()
    .with_model("stabilityai/stable-video-diffusion")
    .build()?

let video = video_pipeline.generate(
    image: ImageInput::from_path("start_frame.jpg"),
    num_frames: 25,
    fps: 8,
    motion_bucket_id: 127,
)?
```

---

## Hardware Requirements

| Feature | Minimum VRAM | Recommended |
|---------|--------------|-------------|
| VLM (7B) | 8 GB | 16 GB |
| VLM (34B) | 24 GB | 48 GB |
| SDXL | 8 GB | 12 GB |
| SD3 Medium | 12 GB | 16 GB |
| Flux Dev | 16 GB | 24 GB |
| SVD | 16 GB | 24 GB |
| CogVideoX | 24 GB | 48 GB |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| VLM latency (7B, 512px image) | < 500ms TTFT |
| SDXL generation (1024x1024, 20 steps) | < 3s |
| Flux generation (1024x1024, 28 steps) | < 5s |
| SVD generation (14 frames, 576x1024) | < 30s |
| Memory efficiency vs competitors | ≥ parity |
| Image quality (FID on COCO) | ≤ reference implementations |

---

## Dependencies & Integration

### Required New Dependencies
- Image processing library (Sigil equivalent of Pillow/OpenCV)
- FFmpeg integration for video I/O
- Additional tokenizers (T5, CLIP)

### Integration Points
- Extend existing `Backend` trait for Conv2D, Conv3D operations
- Reuse `FlashAttention` kernels for diffusion attention
- Integrate with existing LoRA infrastructure
- Extend A/B testing (Astaroth) for image quality metrics

---

*This roadmap transforms Infernum into a complete multimodal AI platform while maintaining the performance-focused philosophy of the existing LLM infrastructure.*
