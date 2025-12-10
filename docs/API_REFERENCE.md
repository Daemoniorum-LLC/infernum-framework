# API Reference

Complete API documentation for Infernum-Sigil.

## Table of Contents

- [Core Types](#core-types)
- [Inference Engine](#inference-engine)
- [Image Generation](#image-generation)
- [Video Generation](#video-generation)
- [Enhancement Pipelines](#enhancement-pipelines)
- [Quality Presets](#quality-presets)
- [Multi-GPU](#multi-gpu)
- [RAG & Embeddings](#rag--embeddings)
- [Agents](#agents)
- [A/B Testing](#ab-testing)
- [Optimization](#optimization)

---

## Core Types

### GenerateRequest

```sigil
struct GenerateRequest {
    /// The prompt to generate from
    prompt: PromptInput,

    /// Maximum tokens to generate
    max_tokens: u32 = 2048,

    /// Sampling temperature (0.0 - 2.0)
    temperature: f32 = 0.7,

    /// Top-p nucleus sampling (0.0 - 1.0)
    top_p: f32 = 0.9,

    /// Top-k sampling (0 = disabled)
    top_k: u32 = 0,

    /// Minimum probability threshold
    min_p: f32 = 0.0,

    /// Repetition penalty (1.0 = disabled)
    repetition_penalty: f32 = 1.0,

    /// Stop sequences
    stop: Option<[str]>,

    /// Random seed for reproducibility
    seed: Option<u64>,

    /// Enable streaming output
    stream: bool = false,
}
```

### PromptInput

```sigil
enum PromptInput {
    /// Plain text prompt
    Text(str~),

    /// Chat messages
    Messages([ChatMessage]~),

    /// Pre-tokenized input
    Tokens([u32]),

    /// Multimodal content
    Multimodal(MultimodalInput~),
}
```

### GenerateResponse

```sigil
struct GenerateResponse {
    /// Generated text choices
    choices~: [Choice],

    /// Token usage statistics
    usage: Usage,

    /// Model identifier
    model: str,

    /// Finish reason
    finish_reason: FinishReason,
}

struct Choice {
    /// Generated text
    text~: str,

    /// Log probabilities (if requested)
    logprobs: Option<[f32]>,

    /// Token information
    tokens: Option<[TokenInfo]>,
}
```

---

## Inference Engine

### Engine

```sigil
impl Engine {
    /// Create a new inference engine
    pub fn new(config: EngineConfig) -> Result<Self, Error>

    /// Load a model from HuggingFace Hub or local path
    pub fn load_model(self, model_id: str) -> Result<Self, Error>

    /// Generate text from a request
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, Error>

    /// Generate with streaming output
    pub async fn generate_stream(&self, request: GenerateRequest) -> TokenStream

    /// Get model metadata
    pub fn model_info(&self) -> &ModelMetadata

    /// Unload the current model
    pub fn unload(&mut self)
}
```

### EngineConfig

```sigil
struct EngineConfig {
    /// Device to run on (cuda, cpu, metal)
    device: str = "cuda",

    /// Data type (f32, f16, bf16)
    dtype: str = "f16",

    /// Maximum context length
    max_seq_len: u32 = 4096,

    /// Maximum batch size
    max_batch_size: u32 = 8,

    /// KV cache configuration
    kv_cache: KVCacheConfig,

    /// Memory configuration
    memory: MemoryConfig,

    /// Speculative decoding configuration
    speculative: Option<SpeculativeConfig>,
}
```

---

## Image Generation

### DiffusionPipeline (Base Trait)

```sigil
trait DiffusionPipeline {
    /// Generate images from text prompt
    fn generate(&self, request: GenerationRequest) -> Result<[ImageOutput], Error>

    /// Generate with progress callback
    fn generate_with_callback(
        &self,
        request: GenerationRequest,
        callback: fn(ProgressInfo)
    ) -> Result<[ImageOutput], Error>
}
```

### GenerationRequest

```sigil
struct GenerationRequest {
    /// Text prompt
    prompt: str~,

    /// Negative prompt
    negative_prompt: Option<str~>,

    /// Output width
    width: u32 = 1024,

    /// Output height
    height: u32 = 1024,

    /// Number of images to generate
    num_images: u32 = 1,

    /// Number of inference steps
    num_steps: u32 = 28,

    /// Guidance scale (CFG)
    guidance_scale: f32 = 7.5,

    /// Random seed
    seed: Option<u64>,

    /// Scheduler type
    scheduler: Option<SchedulerType>,
}
```

### StableDiffusionPipeline

```sigil
impl StableDiffusionPipeline {
    /// Create SD 1.5 pipeline
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Load with custom VAE
    pub fn with_vae(self, vae_path: str) -> Result<Self, Error>

    /// Load LoRA weights
    pub fn load_lora(self, lora_path: str, scale: f32) -> Result<Self, Error>
}
```

### SDXLPipeline

```sigil
impl SDXLPipeline {
    /// Create SDXL base pipeline
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Add refiner for two-stage generation
    pub fn with_refiner(self, refiner_path: str) -> Result<Self, Error>

    /// Configure base/refiner split point (0.0-1.0)
    pub fn with_refiner_start(self, start: f32) -> Self
}
```

### SD3Pipeline

```sigil
impl SD3Pipeline {
    /// Create SD3 pipeline
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Configure triple text encoder weights
    pub fn with_encoder_weights(
        self,
        clip_l: f32,
        clip_g: f32,
        t5: f32
    ) -> Self
}
```

### FluxPipeline

```sigil
impl FluxPipeline {
    /// Create Flux pipeline (Dev or Schnell)
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Flux-specific: guidance scale is typically lower (3.5)
    /// Schnell variant uses 0 guidance (distilled)
}
```

### ImageOutput

```sigil
struct ImageOutput {
    /// Raw pixel data (RGB)
    data: Tensor,

    /// Image width
    width: u32,

    /// Image height
    height: u32,

    /// Generation seed used
    seed: u64,

    /// Save to file
    pub fn save(&self, path: str) -> Result<(), Error>

    /// Convert to base64
    pub fn to_base64(&self) -> str
}
```

---

## Video Generation

### VideoDiffusionPipeline (Base Trait)

```sigil
trait VideoDiffusionPipeline {
    /// Generate video from text/image
    fn generate(&self, request: VideoGenerationRequest) -> Result<VideoOutput, Error>
}
```

### VideoGenerationRequest

```sigil
struct VideoGenerationRequest {
    /// Text prompt (for T2V models)
    prompt: Option<str~>,

    /// Conditioning image (for I2V models)
    image: Option<ImageInput>,

    /// Number of frames
    num_frames: u32 = 25,

    /// Output FPS
    fps: u32 = 24,

    /// Output width
    width: u32 = 1024,

    /// Output height
    height: u32 = 576,

    /// Motion amount (model-specific)
    motion_bucket_id: u32 = 127,

    /// Noise augmentation
    noise_aug_strength: f32 = 0.02,

    /// Random seed
    seed: Option<u64>,
}
```

### StableVideoDiffusionPipeline

```sigil
impl StableVideoDiffusionPipeline {
    /// Create SVD pipeline (I2V)
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Generate video from image
    pub fn generate(&self, request: I2VRequest) -> Result<VideoOutput, Error>
}

struct I2VRequest {
    /// Conditioning image (required)
    image: ImageInput,

    /// Number of frames (14 or 25 for SVD)
    num_frames: u32 = 25,

    /// Motion bucket (0-255, higher = more motion)
    motion_bucket_id: u32 = 127,

    /// FPS (affects motion speed)
    fps: u32 = 6,
}
```

### CogVideoXPipeline

```sigil
impl CogVideoXPipeline {
    /// Create CogVideoX pipeline (T2V)
    pub fn new(model_path: str) -> Result<Self, Error>

    /// Generate video from text
    pub fn generate(&self, request: T2VRequest) -> Result<VideoOutput, Error>
}

struct T2VRequest {
    /// Text prompt (required)
    prompt: str~,

    /// Negative prompt
    negative_prompt: Option<str~>,

    /// Number of frames
    num_frames: u32 = 49,

    /// Guidance scale
    guidance_scale: f32 = 6.0,
}
```

### MochiPipeline

```sigil
impl MochiPipeline {
    /// Create Mochi pipeline (highest quality T2V)
    pub fn new(model_path: str) -> Result<Self, Error>
}
```

### VideoOutput

```sigil
struct VideoOutput {
    /// Frame tensors
    frames: [Tensor],

    /// Frame count
    num_frames: u32,

    /// Frames per second
    fps: u32,

    /// Frame width
    width: u32,

    /// Frame height
    height: u32,

    /// Save as MP4
    pub fn save_mp4(&self, path: str, codec: str = "h264") -> Result<(), Error>

    /// Save as GIF
    pub fn save_gif(&self, path: str) -> Result<(), Error>

    /// Extract frame at index
    pub fn get_frame(&self, index: u32) -> ImageOutput
}
```

---

## Enhancement Pipelines

### EnhancementPipeline (Image)

```sigil
impl EnhancementPipeline {
    /// Create new enhancement pipeline
    pub fn new() -> Self

    /// Add super-resolution
    pub fn with_super_res(
        self,
        model: SuperResModel,
        scale: u32 = 4
    ) -> Result<Self, Error>

    /// Add face restoration
    pub fn with_face_restore(
        self,
        model: FaceRestoreModel,
        fidelity: f32 = 0.8
    ) -> Result<Self, Error>

    /// Add detail enhancement
    pub fn with_detail(
        self,
        strength: f32 = 0.3,
        radius: f32 = 2.0
    ) -> Self

    /// Add color grading
    pub fn with_color_grade(self, config: ColorGradeConfig) -> Self

    /// Add film grain
    pub fn with_film_grain(self, config: FilmGrainConfig) -> Self

    /// Process an image
    pub fn process(&self, image: ImageOutput) -> Result<ImageOutput, Error>
}
```

### SuperResModel

```sigil
enum SuperResModel {
    /// RealESRGAN 2x upscaler
    RealESRGAN_x2,

    /// RealESRGAN 4x upscaler (general purpose)
    RealESRGAN_x4plus,

    /// RealESRGAN 4x anime-optimized
    RealESRGAN_x4plus_anime,

    /// SwinIR (high quality, slower)
    SwinIR,

    /// SwinIR Large (best quality)
    SwinIRLarge,

    /// ESRGAN (classic)
    ESRGAN,

    /// Latent upscaler (diffusion-based)
    LatentUpscaler,
}
```

### FaceRestoreModel

```sigil
enum FaceRestoreModel {
    /// GFPGAN v1.4
    GFPGAN,

    /// CodeFormer (best quality)
    CodeFormer,

    /// RestoreFormer
    RestoreFormer,
}
```

### ColorGradeConfig

```sigil
struct ColorGradeConfig {
    /// Contrast adjustment (1.0 = neutral)
    contrast: f32 = 1.0,

    /// Saturation adjustment
    saturation: f32 = 1.0,

    /// Vibrance (smart saturation)
    vibrance: f32 = 0.0,

    /// Shadow adjustment (-1.0 to 1.0)
    shadows: f32 = 0.0,

    /// Highlight adjustment
    highlights: f32 = 0.0,

    /// Lift (shadow color) [R, G, B]
    lift: [f32; 3] = [0.0, 0.0, 0.0],

    /// Gamma (midtone color)
    gamma: [f32; 3] = [1.0, 1.0, 1.0],

    /// Gain (highlight color)
    gain: [f32; 3] = [1.0, 1.0, 1.0],

    /// 3D LUT file path (.cube)
    lut_path: Option<str>,
}
```

### VideoEnhancePipeline

```sigil
impl VideoEnhancePipeline {
    /// Create new video enhancement pipeline
    pub fn new() -> Self

    /// Add temporal consistency processing
    pub fn with_temporal_consistency(
        self,
        weight: f32 = 0.5
    ) -> Result<Self, Error>

    /// Add frame interpolation
    pub fn with_interpolation(
        self,
        model: InterpolationModel,
        target_fps: u32
    ) -> Result<Self, Error>

    /// Add video super-resolution
    pub fn with_super_res(
        self,
        model: VideoSRModel,
        scale: u32 = 2
    ) -> Result<Self, Error>

    /// Add color grading (same as image)
    pub fn with_color_grade(self, config: ColorGradeConfig) -> Self

    /// Add film grain
    pub fn with_film_grain(self, config: FilmGrainConfig) -> Self

    /// Process video
    pub fn process(&self, video: VideoOutput) -> Result<VideoOutput, Error>
}
```

### InterpolationModel

```sigil
enum InterpolationModel {
    /// RIFE (Real-Time Intermediate Flow Estimation)
    RIFE,

    /// FILM (Frame Interpolation for Large Motion)
    FILM,

    /// IFRNet
    IFRNet,
}
```

### VideoSRModel

```sigil
enum VideoSRModel {
    /// BasicVSR
    BasicVSR,

    /// BasicVSR++ (better quality)
    BasicVSRPlusPlus,

    /// RealBasicVSR (real-world video)
    RealBasicVSR,

    /// EDVR
    EDVR,
}
```

---

## Quality Presets

### ImagePreset

```sigil
impl ImagePreset {
    /// Fast preview quality
    pub fn draft() -> Self

    /// Balanced quality/speed
    pub fn standard() -> Self

    /// High quality output
    pub fn high() -> Self

    /// Maximum quality
    pub fn ultra() -> Self

    /// Film-grade cinematic quality
    pub fn cinematic() -> Self

    /// Create from quality level
    pub fn from_level(level: QualityPreset) -> Self

    /// Optimize for portrait photography
    pub fn portrait_mode(self) -> Self

    /// Optimize for landscape photography
    pub fn landscape_mode(self) -> Self

    /// Target 4K output resolution
    pub fn output_4k(self) -> Self

    /// Target 8K output resolution
    pub fn output_8k(self) -> Self
}
```

### VideoPreset

```sigil
impl VideoPreset {
    /// Fast preview quality
    pub fn draft() -> Self

    /// Balanced quality/speed
    pub fn standard() -> Self

    /// High quality output
    pub fn high() -> Self

    /// Maximum quality
    pub fn ultra() -> Self

    /// Film-grade cinematic quality
    pub fn cinematic() -> Self

    /// Create from quality level
    pub fn from_level(level: QualityPreset) -> Self

    /// Target 4K output
    pub fn output_4k(self) -> Self

    /// Enable slow-motion
    pub fn slow_motion(self, factor: f32) -> Self

    /// Configure for long-form video
    pub fn long_form(self, duration_seconds: u32) -> Self
}
```

### QualityPreset

```sigil
enum QualityPreset {
    Draft,      // ~2GB VRAM
    Standard,   // ~6GB VRAM
    High,       // ~12GB VRAM
    Ultra,      // ~24GB VRAM
    Cinematic,  // ~24GB+ VRAM
    Custom,
}
```

---

## Multi-GPU

### MultiGPUEngine

```sigil
impl MultiGPUEngine {
    /// Create multi-GPU engine
    pub fn new(config: MultiGPUConfig) -> Result<Self, Error>

    /// Generate with automatic workload distribution
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, Error>

    /// Get device utilization stats
    pub fn stats(&self) -> MultiGPUStats
}
```

### MultiGPUConfig

```sigil
struct MultiGPUConfig {
    /// Parallelism strategy
    strategy: ParallelismStrategy,

    /// GPU device IDs to use
    device_ids: [i32],

    /// Tensor parallel size (for strategy=Tensor or Hybrid)
    tensor_parallel_size: u32 = 1,

    /// Pipeline parallel size (for strategy=Pipeline or Hybrid)
    pipeline_parallel_size: u32 = 1,

    /// Load balancing strategy
    load_balance: LoadBalanceStrategy = LoadBalanceStrategy::MemoryAware,
}
```

### ParallelismStrategy

```sigil
enum ParallelismStrategy {
    /// Split data/batches across GPUs
    Data,

    /// Split model layers across GPUs
    Pipeline,

    /// Split tensor dimensions across GPUs
    Tensor,

    /// Combine tensor and pipeline parallelism
    Hybrid,

    /// Automatically select best strategy
    Auto,
}
```

---

## RAG & Embeddings

### RagPipeline

```sigil
impl RagPipeline {
    /// Create new RAG pipeline
    pub fn new() -> Self

    /// Set embedding model
    pub fn with_embedder(self, embedder: impl Embedder) -> Self

    /// Set vector store
    pub fn with_store(self, store: impl VectorStore) -> Self

    /// Set document chunker
    pub fn with_chunker(self, chunker: Chunker) -> Self

    /// Index documents
    pub fn index_documents(&mut self, docs: [Document]) -> Result<(), Error>

    /// Retrieve relevant context
    pub fn retrieve(&self, query: str, top_k: u32) -> Result<[ContextItem], Error>
}
```

### Embedder Implementations

```sigil
/// Engine-based embedder (uses loaded LLM)
struct EngineEmbedder {
    engine: &Engine,
}

/// Sentence transformer embedder
struct SentenceEmbedder {
    model_id: str,
}

impl SentenceEmbedder {
    pub fn new(model_id: str) -> Result<Self, Error>
}
```

### VectorStore Implementations

```sigil
/// In-memory vector store
struct InMemoryStore {
    vectors: HashMap<str, [f32]>,
}

impl InMemoryStore {
    pub fn new() -> Self

    /// Search for similar vectors
    pub fn search(&self, query: [f32], top_k: u32) -> [SearchResult]
}
```

---

## Agents

### Agent

```sigil
impl Agent {
    /// Create agent builder
    pub fn builder() -> AgentBuilder

    /// Run agent on a task
    pub fn run(&self, task: str) -> Result<AgentResult, Error>

    /// Run with conversation history
    pub fn run_with_history(
        &self,
        task: str,
        history: [ChatMessage]
    ) -> Result<AgentResult, Error>
}
```

### AgentBuilder

```sigil
impl AgentBuilder {
    /// Set agent persona
    pub fn with_persona(self, persona: Persona) -> Self

    /// Add tools
    pub fn with_tools(self, tools: [Tool]) -> Self

    /// Set planner
    pub fn with_planner(self, planner: impl Planner) -> Self

    /// Set memory strategy
    pub fn with_memory(self, memory: AgentMemory) -> Self

    /// Build the agent
    pub fn build(self) -> Result<Agent, Error>
}
```

### Tool

```sigil
struct Tool {
    name: str,
    description: str,
    parameters: ToolParameters,
    handler: fn(ToolContext) -> ToolResult,
}

impl Tool {
    /// Web search tool
    pub fn web_search() -> Self

    /// Code execution tool
    pub fn code_executor() -> Self

    /// File reader tool
    pub fn file_reader() -> Self

    /// Calculator tool
    pub fn calculator() -> Self

    /// Custom tool
    pub fn custom(
        name: str,
        description: str,
        handler: fn(ToolContext) -> ToolResult
    ) -> Self
}
```

---

## A/B Testing

### Experiment

```sigil
impl Experiment {
    /// Create new experiment
    pub fn new(name: str) -> ExperimentBuilder
}

impl ExperimentBuilder {
    /// Add control variant
    pub fn add_variant(self, variant: Variant) -> Self

    /// Set primary metric
    pub fn with_primary_metric(
        self,
        metric: MetricType,
        lower_is_better: bool
    ) -> Self

    /// Set minimum samples before analysis
    pub fn with_min_samples(self, n: u32) -> Self

    /// Build experiment
    pub fn build(self) -> Experiment
}
```

### Variant

```sigil
impl Variant {
    /// Create control variant
    pub fn control(name: str) -> Self

    /// Create treatment variant
    pub fn treatment(name: str, config: VariantConfig) -> Self
}

struct VariantConfig {
    /// Model to use
    model: Option<str>,

    /// System prompt
    system_prompt: Option<str>,

    /// Sampling parameters
    sampling: Option<SamplingParams>,

    /// Traffic weight
    weight: f32 = 1.0,
}
```

### ExperimentRunner

```sigil
impl ExperimentRunner {
    /// Start experiment
    pub fn start(&self) -> Result<(), Error>

    /// Route a request to a variant
    pub fn route(&self, request: GenerateRequest) -> Result<(SplitDecision, GenerateRequest), Error>

    /// Record response metrics
    pub fn record(
        &self,
        decision: &SplitDecision,
        response: &GenerateResponse,
        latency_ms: u64
    ) -> Result<(), Error>

    /// Analyze results
    pub fn analyze(&self, confidence: f32) -> Result<ExperimentResult, Error>

    /// Stop experiment
    pub fn stop(&self) -> Result<(), Error>
}
```

### ExperimentResult

```sigil
struct ExperimentResult {
    /// Whether a statistically significant winner was found
    pub fn has_winner(&self) -> bool

    /// Get winning variant name
    pub fn winner(&self) -> Option<str>

    /// Get improvement percentage
    pub fn improvement(&self) -> f32

    /// Get confidence interval
    pub fn confidence_interval(&self) -> (f32, f32)

    /// Get p-value
    pub fn p_value(&self) -> f32
}
```

---

## Optimization

### Attention Backends

```sigil
/// Flash Attention v2
struct FlashAttention {
    config: FlashAttentionConfig,
}

struct FlashAttentionConfig {
    head_dim: u32,
    causal: bool = true,
    softmax_scale: Option<f32>,
    dropout: f32 = 0.0,
}

/// Memory-efficient attention (chunked)
struct MemoryEfficientAttention {
    chunk_size: u32,
}
```

### Quantization

```sigil
impl ModelQuantizer {
    /// Quantize model to specified format
    pub fn quantize(
        model: Model,
        method: QuantizationMethod
    ) -> Result<QuantizedModel, Error>
}

enum QuantizationMethod {
    /// 8-bit floating point
    FP8,

    /// 8-bit integer
    INT8,

    /// 4-bit integer
    INT4,

    /// 4-bit NormalFloat
    NF4,
}
```

### Caching

```sigil
/// Prompt embedding cache
struct PromptCache {
    config: PromptCacheConfig,
}

impl PromptCache {
    pub fn new(config: PromptCacheConfig) -> Self

    /// Get cached embedding
    pub fn get(&self, prompt: str) -> Option<CachedEmbedding>

    /// Cache embedding
    pub fn put(&mut self, prompt: str, embedding: Tensor)
}

/// Latent cache for diffusion
struct LatentCache {
    max_entries: u32,
}
```

### Distilled Schedulers

```sigil
/// LCM (Latent Consistency Model) scheduler
/// Enables 4-8 step generation
struct LCMScheduler {
    num_steps: u32,
    guidance_scale: f32,
}

/// Turbo scheduler for SDXL
/// Enables 1-4 step generation
struct TurboScheduler {
    num_steps: u32,
}

/// Lightning scheduler
/// High quality with few steps
struct LightningScheduler {
    num_steps: u32,
}
```

---

## Error Handling

### Error Types

```sigil
enum Error {
    /// Model loading failed
    ModelLoad(str),

    /// Invalid configuration
    Config(str),

    /// Out of memory
    OutOfMemory,

    /// CUDA error
    Cuda(str),

    /// Invalid input
    InvalidInput(str),

    /// Generation failed
    Generation(str),

    /// Network error
    Network(str),

    /// File I/O error
    Io(str),
}

/// Result type alias
type Result<T, E = Error> = std::Result<T, E>
```

---

## Evidentiality Conventions

Throughout the API, Sigil's evidentiality markers indicate data provenance:

| Marker | Meaning | Example |
|--------|---------|---------|
| `~` | Untrusted/external | `prompt~: str` |
| `!` | Verified/computed | `result!: ValidatedOutput` |
| `?` | Uncertain/optional | `cached?: Option<Tensor>` |

```sigil
// LLM output is untrusted
struct Response {
    content~: str,  // Mark as external
}

// Validated data is verified
fn validate(resp~: Response) -> Result! {
    resp~|validate!{ content: safe }
}
```
