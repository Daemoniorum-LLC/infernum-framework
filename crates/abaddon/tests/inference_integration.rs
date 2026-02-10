//! Integration tests for Abaddon inference engine components.
//!
//! Tests the core inference primitives: configuration, device selection,
//! sampling, attention, and KV cache management.

use abaddon::{
    best_device, enumerate_devices, AttentionVariant, DeviceInfo, EngineConfig, FlashAttention,
    FlashAttentionConfig, GgufMetadata, KVCache, MemoryConfig, QuantizedModelConfig, Sampler,
    SpeculativeConfig,
};
use candle_core::{Device, Tensor};
use infernum_core::{DeviceType, ModelSource, QuantizationType, RequestId, SamplingParams};

// ============================================================================
// EngineConfig Builder Pattern Tests
// ============================================================================

#[test]
fn test_engine_config_builder_minimal() {
    let config = EngineConfig::builder()
        .model("meta-llama/Llama-3.2-3B-Instruct")
        .build()
        .expect("build");

    match &config.model {
        ModelSource::HuggingFace { repo_id, revision } => {
            assert_eq!(repo_id, "meta-llama/Llama-3.2-3B-Instruct");
            assert!(revision.is_none());
        },
        _ => panic!("Expected HuggingFace source"),
    }
    assert_eq!(config.device, DeviceType::Cpu);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.max_seq_len, 4096);
}

#[test]
fn test_engine_config_builder_with_cuda() {
    let config = EngineConfig::builder()
        .model("test-model")
        .cuda(0)
        .max_batch_size(64)
        .max_seq_len(8192)
        .build()
        .expect("build");

    assert!(matches!(config.device, DeviceType::Cuda { device_id: 0 }));
    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.max_seq_len, 8192);
}

#[test]
fn test_engine_config_builder_with_metal() {
    let config = EngineConfig::builder()
        .model("test-model")
        .metal()
        .build()
        .expect("build");

    assert!(matches!(config.device, DeviceType::Metal { device_id: 0 }));
}

#[test]
fn test_engine_config_builder_with_quantization() {
    let config = EngineConfig::builder()
        .model("test-model")
        .quantization(QuantizationType::GgufQ4KM)
        .build()
        .expect("build");

    assert_eq!(config.quantization, Some(QuantizationType::GgufQ4KM));
}

#[test]
fn test_engine_config_builder_with_memory_config() {
    let memory = MemoryConfig::low_memory();
    let config = EngineConfig::builder()
        .model("test-model")
        .memory(memory)
        .build()
        .expect("build");

    assert!(config.memory.cpu_offload);
    assert_eq!(config.memory.gpu_layers, Some(20));
}

#[test]
fn test_engine_config_builder_with_speculative() {
    let spec = SpeculativeConfig::new(ModelSource::huggingface("draft-model"));
    let config = EngineConfig::builder()
        .model("main-model")
        .speculative(spec)
        .build()
        .expect("build");

    assert!(config.speculative.is_some());
    let spec_config = config.speculative.unwrap();
    assert_eq!(spec_config.num_speculative_tokens, 5);
}

#[test]
fn test_engine_config_builder_with_cache_dir() {
    let config = EngineConfig::builder()
        .model("test-model")
        .cache_dir("/tmp/infernum-cache")
        .build()
        .expect("build");

    assert_eq!(
        config.cache_dir,
        Some(std::path::PathBuf::from("/tmp/infernum-cache"))
    );
}

#[test]
fn test_engine_config_builder_no_model_fails() {
    let result = EngineConfig::builder().cuda(0).max_batch_size(32).build();

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("model is required"));
}

#[test]
fn test_engine_config_builder_device_overwrite() {
    // Later device setting should overwrite earlier
    let config = EngineConfig::builder()
        .model("test")
        .cuda(0)
        .metal() // Overwrites cuda
        .build()
        .expect("build");

    assert!(matches!(config.device, DeviceType::Metal { .. }));
}

#[test]
fn test_engine_config_builder_with_local_model() {
    let config = EngineConfig::builder()
        .model_source(ModelSource::local("/path/to/model.gguf"))
        .build()
        .expect("build");

    assert!(matches!(config.model, ModelSource::LocalPath { .. }));
}

#[test]
fn test_engine_config_serialization_roundtrip() {
    let config = EngineConfig::builder()
        .model("test-model")
        .cuda(1)
        .max_batch_size(16)
        .max_seq_len(2048)
        .quantization(QuantizationType::GgufQ5KM)
        .build()
        .expect("build");

    let json = serde_json::to_string(&config).expect("serialize");
    let parsed: EngineConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.max_batch_size, 16);
    assert_eq!(parsed.max_seq_len, 2048);
    assert_eq!(parsed.quantization, Some(QuantizationType::GgufQ5KM));
}

// ============================================================================
// MemoryConfig Presets Tests
// ============================================================================

#[test]
fn test_memory_config_default() {
    let config = MemoryConfig::default();

    assert_eq!(config.gpu_memory_limit, 0);
    assert!((config.kv_cache_fraction - 0.9).abs() < 0.01);
    assert!(config.mmap_enabled);
    assert!(!config.cpu_offload);
    assert!(config.gpu_layers.is_none());
}

#[test]
fn test_memory_config_low_memory() {
    let config = MemoryConfig::low_memory();

    assert!((config.kv_cache_fraction - 0.5).abs() < 0.01);
    assert!(config.cpu_offload);
    assert_eq!(config.gpu_layers, Some(20));
}

#[test]
fn test_memory_config_high_throughput() {
    let config = MemoryConfig::high_throughput();

    assert!((config.kv_cache_fraction - 0.95).abs() < 0.01);
    assert!(!config.cpu_offload);
    assert!(config.gpu_layers.is_none());
}

#[test]
fn test_memory_config_rtx_4000_series() {
    let config = MemoryConfig::rtx_4000_series();

    assert_eq!(config.gpu_memory_limit, 22 * 1024 * 1024 * 1024);
    assert!((config.kv_cache_fraction - 0.92).abs() < 0.01);
    assert!(!config.cpu_offload);
}

#[test]
fn test_memory_config_workstation_gpu() {
    let config = MemoryConfig::workstation_gpu();

    assert_eq!(config.gpu_memory_limit, 0); // Auto-detect
    assert!((config.kv_cache_fraction - 0.90).abs() < 0.01);
}

#[test]
fn test_memory_config_large_model() {
    let config = MemoryConfig::large_model();

    assert!((config.kv_cache_fraction - 0.5).abs() < 0.01);
    assert!(config.cpu_offload);
    assert_eq!(config.gpu_layers, Some(60));
}

#[test]
fn test_memory_config_serialization() {
    let config = MemoryConfig {
        gpu_memory_limit: 16 * 1024 * 1024 * 1024,
        kv_cache_fraction: 0.85,
        mmap_enabled: false,
        cpu_offload: true,
        gpu_layers: Some(40),
    };

    let json = serde_json::to_string(&config).expect("serialize");
    let parsed: MemoryConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.gpu_memory_limit, 16 * 1024 * 1024 * 1024);
    assert!((parsed.kv_cache_fraction - 0.85).abs() < 0.01);
    assert!(!parsed.mmap_enabled);
    assert!(parsed.cpu_offload);
    assert_eq!(parsed.gpu_layers, Some(40));
}

// ============================================================================
// SpeculativeConfig Tests
// ============================================================================

#[test]
fn test_speculative_config_new() {
    let draft = ModelSource::huggingface("draft-model");
    let config = SpeculativeConfig::new(draft);

    assert_eq!(config.num_speculative_tokens, 5);
    assert!((config.acceptance_threshold - 0.9).abs() < 0.01);
}

#[test]
fn test_speculative_config_with_local_draft() {
    let draft = ModelSource::local("/path/to/draft.gguf");
    let config = SpeculativeConfig::new(draft);

    match &config.draft_model {
        ModelSource::LocalPath { path } => {
            assert!(path.to_string_lossy().contains("draft.gguf"));
        },
        _ => panic!("Expected LocalPath source"),
    }
}

#[test]
fn test_speculative_config_serialization() {
    let config = SpeculativeConfig {
        draft_model: ModelSource::huggingface("small-draft"),
        num_speculative_tokens: 8,
        acceptance_threshold: 0.85,
    };

    let json = serde_json::to_string(&config).expect("serialize");
    let parsed: SpeculativeConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.num_speculative_tokens, 8);
    assert!((parsed.acceptance_threshold - 0.85).abs() < 0.01);
}

// ============================================================================
// Device Enumeration Tests
// ============================================================================

#[test]
fn test_enumerate_devices_always_has_cpu() {
    let devices = enumerate_devices();

    assert!(!devices.is_empty());
    let has_cpu = devices
        .iter()
        .any(|d| matches!(d.device_type, DeviceType::Cpu));
    assert!(has_cpu, "CPU should always be available");
}

#[test]
fn test_enumerate_devices_marks_one_recommended() {
    let devices = enumerate_devices();

    let recommended_count = devices.iter().filter(|d| d.recommended).count();
    assert_eq!(
        recommended_count, 1,
        "Exactly one device should be recommended"
    );
}

#[test]
fn test_best_device_returns_valid() {
    let device = best_device();

    match device {
        DeviceType::Cpu => (),
        DeviceType::Cuda { device_id } => assert!(device_id < 100),
        DeviceType::Metal { device_id } => assert!(device_id < 100),
        DeviceType::WebGpu => (),
    }
}

#[test]
fn test_device_info_cpu() {
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| matches!(d.device_type, DeviceType::Cpu))
        .expect("CPU should exist");

    assert!(!cpu.name.is_empty());
    assert!(cpu.total_memory > 0);
    assert!(!cpu.has_tensor_cores);
}

#[test]
fn test_device_info_serialization() {
    let info = DeviceInfo {
        device_type: DeviceType::Cuda { device_id: 0 },
        name: "NVIDIA RTX 4090".to_string(),
        total_memory: 24 * 1024 * 1024 * 1024,
        available_memory: 20 * 1024 * 1024 * 1024,
        compute_capability: Some((8, 9)),
        has_tensor_cores: true,
        supports_bf16: true,
        recommended: true,
    };

    let json = serde_json::to_string(&info).expect("serialize");
    let parsed: DeviceInfo = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.name, "NVIDIA RTX 4090");
    assert!(parsed.has_tensor_cores);
    assert!(parsed.supports_bf16);
    assert_eq!(parsed.compute_capability, Some((8, 9)));
}

// ============================================================================
// Sampler Tests
// ============================================================================

#[test]
fn test_sampler_greedy() {
    let params = SamplingParams::greedy();
    let mut sampler = Sampler::new(params);

    let logits = vec![1.0f32, 5.0, 2.0, 0.5];
    let token = sampler.sample(&logits);

    assert_eq!(token, 1, "Greedy should select highest logit");
}

#[test]
fn test_sampler_greedy_consistency() {
    let params = SamplingParams::greedy();
    let mut sampler = Sampler::new(params);

    let logits = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

    // Greedy should always return same result
    let token1 = sampler.sample(&logits);
    let token2 = sampler.sample(&logits);
    let token3 = sampler.sample(&logits);

    assert_eq!(token1, token2);
    assert_eq!(token2, token3);
    assert_eq!(token1, 5, "Should select index of highest value (9.0)");
}

#[test]
fn test_sampler_deterministic_with_seed() {
    let params = SamplingParams::balanced().with_seed(42);
    let mut sampler1 = Sampler::new(params.clone());
    let mut sampler2 = Sampler::new(params);

    let logits = vec![1.0, 1.0, 1.0, 1.0];

    assert_eq!(
        sampler1.sample(&logits),
        sampler2.sample(&logits),
        "Same seed should produce same result"
    );
}

#[test]
fn test_sampler_creative_params() {
    let params = SamplingParams::creative();
    let mut sampler = Sampler::new(params.clone());

    // Creative has higher temperature
    assert!(params.temperature > 0.5);

    let logits = vec![1.0, 1.0, 1.0, 1.0];
    let _ = sampler.sample(&logits); // Should not panic
}

#[test]
fn test_sampler_stop_token_detection() {
    let params = SamplingParams::balanced()
        .with_stop("END")
        .with_stop("STOP");
    let sampler = Sampler::new(params);

    assert!(sampler.is_stop_token("Some text END here"));
    assert!(sampler.is_stop_token("STOP"));
    assert!(!sampler.is_stop_token("Continue"));
}

#[test]
fn test_sampler_params_access() {
    let params = SamplingParams::balanced()
        .with_max_tokens(100)
        .with_temperature(0.7);
    let sampler = Sampler::new(params);

    assert_eq!(sampler.params().max_tokens, 100);
    assert!((sampler.params().temperature - 0.7).abs() < 0.01);
}

// ============================================================================
// FlashAttention Tests
// ============================================================================

#[test]
fn test_flash_attention_config_default() {
    let config = FlashAttentionConfig::default();

    // Default block_size is 512 for high-VRAM GPUs (20GB+), reduces tiling overhead
    assert_eq!(config.block_size, 512);
    assert!(config.causal);
    assert_eq!(config.dropout, 0.0);
    assert!(config.softmax_scale.is_none());
}

#[test]
fn test_flash_attention_config_long_context() {
    let config = FlashAttentionConfig::for_long_context();

    assert_eq!(config.block_size, 128);
    assert_eq!(config.max_seqlen, Some(32768));
    assert!(config.causal);
}

#[test]
fn test_flash_attention_config_non_causal() {
    let config = FlashAttentionConfig::non_causal();

    assert!(!config.causal);
    assert_eq!(config.block_size, 64);
}

#[test]
fn test_flash_attention_config_with_scale() {
    let config = FlashAttentionConfig::default().with_scale(0.125);

    assert_eq!(config.softmax_scale, Some(0.125));
}

#[test]
fn test_flash_attention_config_with_dropout() {
    let config = FlashAttentionConfig::default().with_dropout(0.1);

    assert!((config.dropout - 0.1).abs() < 0.001);
}

#[test]
fn test_flash_attention_creation() {
    let config = FlashAttentionConfig::default();
    let flash_attn = FlashAttention::new(config);

    // Should not panic, config should be accessible
    assert!(flash_attn.config().causal);
}

#[test]
fn test_flash_attention_short_sequence() {
    let config = FlashAttentionConfig::default();
    let flash_attn = FlashAttention::new(config);

    let device = Device::Cpu;
    let batch = 2;
    let heads = 4;
    let seq_len = 16;
    let head_dim = 32;

    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &device).unwrap();

    let output = flash_attn.forward(&q, &k, &v, None, Some(true)).unwrap();

    assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
}

#[test]
fn test_flash_attention_non_causal() {
    let config = FlashAttentionConfig::non_causal();
    let flash_attn = FlashAttention::new(config);

    let device = Device::Cpu;
    let batch = 1;
    let heads = 2;
    let seq_len = 8;
    let head_dim = 16;

    let q = Tensor::randn(0.0f32, 0.5, (batch, heads, seq_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 0.5, (batch, heads, seq_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 0.5, (batch, heads, seq_len, head_dim), &device).unwrap();

    let output = flash_attn.forward(&q, &k, &v, None, Some(false)).unwrap();

    assert_eq!(output.dims(), &[batch, heads, seq_len, head_dim]);
}

#[test]
fn test_flash_attention_different_block_sizes() {
    for block_size in [16, 32, 64, 128] {
        let config = FlashAttentionConfig {
            block_size,
            ..Default::default()
        };
        let flash_attn = FlashAttention::new(config);

        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 0.5, (1, 2, 32, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 0.5, (1, 2, 32, 16), &device).unwrap();
        let v = Tensor::randn(0.0f32, 0.5, (1, 2, 32, 16), &device).unwrap();

        let output = flash_attn.forward(&q, &k, &v, None, None).unwrap();
        assert_eq!(output.dims(), &[1, 2, 32, 16]);
    }
}

// ============================================================================
// AttentionVariant Tests
// ============================================================================

#[test]
fn test_attention_variant_default() {
    let variant = AttentionVariant::default();
    assert_eq!(variant, AttentionVariant::Standard);
}

#[test]
fn test_attention_variant_memory_efficient() {
    assert!(!AttentionVariant::Standard.is_memory_efficient());
    assert!(AttentionVariant::Flash.is_memory_efficient());
    assert!(AttentionVariant::MultiQuery.is_memory_efficient());
    assert!(AttentionVariant::GroupedQuery.is_memory_efficient());
}

#[test]
fn test_attention_variant_recommendation() {
    assert_eq!(
        AttentionVariant::recommended_for_seq_len(512),
        AttentionVariant::Standard
    );
    assert_eq!(
        AttentionVariant::recommended_for_seq_len(2048),
        AttentionVariant::Standard
    );
    assert_eq!(
        AttentionVariant::recommended_for_seq_len(4096),
        AttentionVariant::Flash
    );
    assert_eq!(
        AttentionVariant::recommended_for_seq_len(32768),
        AttentionVariant::Flash
    );
}

// ============================================================================
// KVCache Tests
// ============================================================================

#[test]
fn test_kv_cache_creation() {
    let config = abaddon::kv_cache::KVCacheConfig::default();
    let cache = KVCache::new(config);

    assert!(cache.free_block_count() > 0);
    assert_eq!(cache.utilization(), 0.0);
}

#[test]
fn test_kv_cache_allocation() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let mut cache = KVCache::new(config);

    let request_id = RequestId::new();
    cache.allocate(request_id.clone(), 32).unwrap();

    // 32 tokens / 16 block_size = 2 blocks
    assert_eq!(cache.get_blocks(&request_id).unwrap().len(), 2);
    assert!(cache.utilization() > 0.0);
}

#[test]
fn test_kv_cache_extend() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let mut cache = KVCache::new(config);

    let request_id = RequestId::new();
    cache.allocate(request_id.clone(), 16).unwrap();
    let initial_blocks = cache.get_blocks(&request_id).unwrap().len();

    cache.extend(&request_id, 32).unwrap();
    let final_blocks = cache.get_blocks(&request_id).unwrap().len();

    assert!(final_blocks > initial_blocks);
}

#[test]
fn test_kv_cache_free() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let mut cache = KVCache::new(config);
    let initial_free = cache.free_block_count();

    let request_id = RequestId::new();
    cache.allocate(request_id.clone(), 64).unwrap();
    assert!(cache.free_block_count() < initial_free);

    cache.free(&request_id);
    assert_eq!(cache.free_block_count(), initial_free);
    assert!(cache.get_blocks(&request_id).is_none());
}

#[test]
fn test_kv_cache_multiple_sequences() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 256,
        ..Default::default()
    };
    let mut cache = KVCache::new(config);

    let req1 = RequestId::new();
    let req2 = RequestId::new();
    let req3 = RequestId::new();

    cache.allocate(req1.clone(), 32).unwrap();
    cache.allocate(req2.clone(), 48).unwrap();
    cache.allocate(req3.clone(), 16).unwrap();

    assert!(cache.get_blocks(&req1).is_some());
    assert!(cache.get_blocks(&req2).is_some());
    assert!(cache.get_blocks(&req3).is_some());

    cache.free(&req2);
    assert!(cache.get_blocks(&req1).is_some());
    assert!(cache.get_blocks(&req2).is_none());
    assert!(cache.get_blocks(&req3).is_some());
}

#[test]
fn test_kv_cache_out_of_blocks() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 64, // Only 4 blocks
        ..Default::default()
    };
    let mut cache = KVCache::new(config);

    let req1 = RequestId::new();
    cache.allocate(req1.clone(), 64).unwrap(); // Uses all 4 blocks

    let req2 = RequestId::new();
    let result = cache.allocate(req2, 16); // Should fail

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Not enough free blocks"));
}

#[test]
fn test_kv_cache_utilization() {
    let config = abaddon::kv_cache::KVCacheConfig {
        block_size: 16,
        max_seq_len: 256, // 16 blocks
        ..Default::default()
    };
    let mut cache = KVCache::new(config);

    assert_eq!(cache.utilization(), 0.0);

    let req = RequestId::new();
    cache.allocate(req.clone(), 128).unwrap(); // Uses 8 blocks

    let utilization = cache.utilization();
    assert!((utilization - 0.5).abs() < 0.01);

    cache.free(&req);
    assert_eq!(cache.utilization(), 0.0);
}

// ============================================================================
// GgufMetadata Tests
// ============================================================================

#[test]
fn test_gguf_metadata_to_quantized_config() {
    let metadata = GgufMetadata {
        architecture: "llama".to_string(),
        name: Some("Llama-3.2-3B".to_string()),
        num_attention_heads: 32,
        num_kv_heads: 8,
        num_layers: 28,
        hidden_size: 3072,
        intermediate_size: 8192,
        vocab_size: 128256,
        context_length: 8192,
        rope_theta: 500000.0,
        rms_norm_eps: 1e-5,
        quantization_type: "q4_k_m".to_string(),
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        pad_token_id: None,
    };

    let config = QuantizedModelConfig::from(&metadata);

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.num_layers, 28);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.hidden_size, 3072);
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.context_length, 8192);
    assert!((config.rope_theta - 500000.0).abs() < 0.01);
    assert_eq!(config.bos_token_id, Some(128000));
    assert_eq!(config.eos_token_id, Some(128001));
}

// ============================================================================
// Integration Workflow Tests
// ============================================================================

#[test]
fn test_inference_config_workflow() {
    // Simulate a complete configuration workflow

    // 1. Enumerate devices
    let devices = enumerate_devices();
    assert!(!devices.is_empty());

    // 2. Select best device
    let device = best_device();

    // 3. Build configuration
    let config = EngineConfig::builder()
        .model("meta-llama/Llama-3.2-3B-Instruct")
        .device(device)
        .memory(MemoryConfig::low_memory())
        .max_batch_size(8)
        .max_seq_len(4096)
        .build()
        .expect("build");

    // 4. Create KV cache
    let cache_config = abaddon::kv_cache::KVCacheConfig {
        num_layers: 28,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: config.max_seq_len,
        block_size: 16,
    };
    let mut cache = KVCache::new(cache_config);

    // 5. Allocate for a request
    let request_id = RequestId::new();
    cache.allocate(request_id.clone(), 100).unwrap();

    // 6. Create sampler
    let params = SamplingParams::balanced().with_max_tokens(200);
    let mut sampler = Sampler::new(params);

    // 7. Sample from mock logits
    let logits = vec![1.0, 2.0, 5.0, 1.0];
    let token = sampler.sample(&logits);
    assert!(token < 4);

    // 8. Free cache
    cache.free(&request_id);
    assert_eq!(cache.utilization(), 0.0);
}

#[test]
fn test_attention_with_kv_cache_workflow() {
    // Simulate attention computation with cache management

    let flash_attn = FlashAttention::new(FlashAttentionConfig::default());
    let device = Device::Cpu;

    // Simulate prefill phase
    let batch = 1;
    let heads = 4;
    let prefill_len = 32;
    let head_dim = 64;

    let q = Tensor::randn(0.0f32, 0.5, (batch, heads, prefill_len, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 0.5, (batch, heads, prefill_len, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 0.5, (batch, heads, prefill_len, head_dim), &device).unwrap();

    let output = flash_attn.forward(&q, &k, &v, None, Some(true)).unwrap();
    assert_eq!(output.dims(), &[batch, heads, prefill_len, head_dim]);

    // Simulate decode phase (single token)
    let decode_q = Tensor::randn(0.0f32, 0.5, (batch, heads, 1, head_dim), &device).unwrap();

    // In real scenario, K/V would come from cache
    let decode_k = Tensor::randn(
        0.0f32,
        0.5,
        (batch, heads, prefill_len + 1, head_dim),
        &device,
    )
    .unwrap();
    let decode_v = Tensor::randn(
        0.0f32,
        0.5,
        (batch, heads, prefill_len + 1, head_dim),
        &device,
    )
    .unwrap();

    let decode_output = flash_attn
        .forward(&decode_q, &decode_k, &decode_v, None, Some(true))
        .unwrap();
    assert_eq!(decode_output.dims(), &[batch, heads, 1, head_dim]);
}

#[test]
fn test_batch_sampling_workflow() {
    // Test sampling across multiple requests

    let params = SamplingParams::balanced().with_seed(123);
    let mut sampler = Sampler::new(params);

    let batch_logits = vec![
        vec![1.0, 2.0, 5.0, 1.0],
        vec![3.0, 1.0, 1.0, 0.5],
        vec![0.1, 0.2, 0.1, 9.0],
    ];

    let tokens: Vec<u32> = batch_logits
        .iter()
        .map(|logits| sampler.sample(logits))
        .collect();

    assert_eq!(tokens.len(), 3);
    for token in &tokens {
        assert!(*token < 4);
    }
}

#[test]
fn test_device_selection_for_config() {
    // Test that device selection integrates with config building

    let devices = enumerate_devices();
    let recommended = devices.iter().find(|d| d.recommended).cloned();

    if let Some(device) = recommended {
        let device_type = device.device_type.clone();
        let config = EngineConfig::builder()
            .model("test-model")
            .device(device_type.clone())
            .build()
            .expect("build");

        assert_eq!(config.device, device_type);
    }
}
