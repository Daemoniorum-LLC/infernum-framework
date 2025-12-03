//! Comprehensive benchmarks for inference performance.
//!
//! These benchmarks verify the performance claims made in the documentation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ============================================================================
// SAMPLING BENCHMARKS
// ============================================================================

fn sampling_benchmark(c: &mut Criterion) {
    use abaddon::sampler::Sampler;
    use infernum_core::SamplingParams;

    let logits_32k: Vec<f32> = (0..32000).map(|i| (i as f32).sin()).collect();

    let mut group = c.benchmark_group("sampling");

    // Core sampling strategies
    group.bench_function("greedy_32k_vocab", |b| {
        let mut sampler = Sampler::new(SamplingParams::greedy());
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    group.bench_function("top_p_32k_vocab", |b| {
        let mut sampler = Sampler::new(SamplingParams::balanced());
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    group.bench_function("top_k_50_32k_vocab", |b| {
        let params = SamplingParams::default().with_top_k(50);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    group.bench_function("top_k_10_32k_vocab", |b| {
        let params = SamplingParams::default().with_top_k(10);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    // Combined top-k + top-p (common in practice)
    group.bench_function("top_k_50_top_p_0.9", |b| {
        let params = SamplingParams::default().with_top_k(50).with_top_p(0.9);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    // Creative sampling (high temp + top-k + top-p)
    group.bench_function("creative_sampling", |b| {
        let mut sampler = Sampler::new(SamplingParams::creative());
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    group.finish();
}

/// Benchmark sampling across different vocabulary sizes to verify O(n) scaling claims.
fn vocabulary_scaling_benchmark(c: &mut Criterion) {
    use abaddon::sampler::Sampler;
    use infernum_core::SamplingParams;

    let mut group = c.benchmark_group("vocab_scaling");

    // Test different vocabulary sizes (common in real models)
    let vocab_sizes = [
        (4096, "4k"),     // Small models
        (32000, "32k"),   // Llama-style
        (50257, "50k"),   // GPT-2 style
        (128256, "128k"), // Llama 3 style
    ];

    for (size, label) in vocab_sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("greedy", label), &logits, |b, logits| {
            let mut sampler = Sampler::new(SamplingParams::greedy());
            b.iter(|| sampler.sample(black_box(logits)))
        });

        group.bench_with_input(BenchmarkId::new("top_p", label), &logits, |b, logits| {
            let mut sampler = Sampler::new(SamplingParams::balanced());
            b.iter(|| sampler.sample(black_box(logits)))
        });
    }

    group.finish();
}

// ============================================================================
// KV-CACHE BENCHMARKS
// ============================================================================

fn kv_cache_benchmark(c: &mut Criterion) {
    use abaddon::kv_cache::{KVCache, KVCacheConfig};
    use infernum_core::RequestId;

    let mut group = c.benchmark_group("kv_cache");

    // Benchmark allocation performance
    group.bench_function("allocate_single_sequence_128_tokens", |b| {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 4096,
            ..Default::default()
        };
        b.iter(|| {
            let mut cache = KVCache::new(config.clone());
            let request_id = RequestId::new();
            cache.allocate(request_id, black_box(128)).unwrap();
        })
    });

    group.bench_function("allocate_single_sequence_2048_tokens", |b| {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 8192,
            ..Default::default()
        };
        b.iter(|| {
            let mut cache = KVCache::new(config.clone());
            let request_id = RequestId::new();
            cache.allocate(request_id, black_box(2048)).unwrap();
        })
    });

    // Benchmark extend operations (common during autoregressive generation)
    group.bench_function("extend_sequence_by_1_token", |b| {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 4096,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);
        let request_id = RequestId::new();
        cache.allocate(request_id.clone(), 128).unwrap();

        b.iter(|| {
            // Extending by 1 usually doesn't need new blocks
            let _ = cache.extend(black_box(&request_id), 1);
        })
    });

    // Benchmark free operations
    group.bench_function("free_sequence", |b| {
        b.iter_batched(
            || {
                let config = KVCacheConfig {
                    block_size: 16,
                    max_seq_len: 4096,
                    ..Default::default()
                };
                let mut cache = KVCache::new(config);
                let request_id = RequestId::new();
                cache.allocate(request_id.clone(), 512).unwrap();
                (cache, request_id)
            },
            |(mut cache, request_id)| {
                cache.free(black_box(&request_id));
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark concurrent sequence management (simulates batch inference)
    group.bench_function("manage_32_concurrent_sequences", |b| {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 32768, // Large enough for 32 sequences
            ..Default::default()
        };

        b.iter(|| {
            let mut cache = KVCache::new(config.clone());
            let mut request_ids = Vec::with_capacity(32);

            // Allocate 32 sequences
            for _ in 0..32 {
                let request_id = RequestId::new();
                cache.allocate(request_id.clone(), 128).unwrap();
                request_ids.push(request_id);
            }

            // Free all sequences
            for request_id in &request_ids {
                cache.free(request_id);
            }
        })
    });

    // Benchmark utilization calculation
    group.bench_function("utilization_calculation", |b| {
        let config = KVCacheConfig {
            block_size: 16,
            max_seq_len: 4096,
            ..Default::default()
        };
        let mut cache = KVCache::new(config);

        // Pre-allocate some sequences
        for _ in 0..10 {
            let request_id = RequestId::new();
            cache.allocate(request_id, 128).unwrap();
        }

        b.iter(|| black_box(cache.utilization()))
    });

    group.finish();
}

// ============================================================================
// STREAMING BENCHMARKS
// ============================================================================

fn streaming_benchmark(c: &mut Criterion) {
    use infernum_core::streaming::{StreamChunkBuilder, StreamDelta};
    use infernum_core::{ModelId, RequestId};

    let mut group = c.benchmark_group("streaming");

    // Benchmark chunk creation (happens per token)
    group.bench_function("create_stream_chunk", |b| {
        let request_id = RequestId::new();
        let model_id = ModelId::new("test-model");

        b.iter(|| {
            StreamChunkBuilder::new(request_id.clone(), model_id.clone())
                .text(0, "Hello")
                .build()
        })
    });

    // Benchmark delta creation
    group.bench_function("create_text_delta", |b| {
        b.iter(|| StreamDelta::text(black_box("Hello world")))
    });

    // Benchmark JSON serialization of chunks (important for SSE streaming)
    group.bench_function("serialize_stream_chunk", |b| {
        let request_id = RequestId::new();
        let model_id = ModelId::new("test-model");
        let chunk = StreamChunkBuilder::new(request_id, model_id)
            .text(0, "Hello, how can I help you today?")
            .build();

        b.iter(|| serde_json::to_string(black_box(&chunk)).unwrap())
    });

    // Benchmark typical streaming scenario: 100 tokens
    group.throughput(Throughput::Elements(100));
    group.bench_function("create_100_stream_chunks", |b| {
        let request_id = RequestId::new();
        let model_id = ModelId::new("test-model");

        b.iter(|| {
            let mut chunks = Vec::with_capacity(100);
            for i in 0..100 {
                let chunk = StreamChunkBuilder::new(request_id.clone(), model_id.clone())
                    .text(0, format!("token_{}", i))
                    .build();
                chunks.push(chunk);
            }
            chunks
        })
    });

    group.finish();
}

// ============================================================================
// SOFTMAX / NUMERICAL BENCHMARKS
// ============================================================================

fn numerical_benchmark(c: &mut Criterion) {
    use abaddon::sampler::Sampler;
    use infernum_core::SamplingParams;

    let mut group = c.benchmark_group("numerical");

    // Benchmark softmax computation (called internally by sampler)
    // We test indirectly via top_p which requires softmax
    let logits_32k: Vec<f32> = (0..32000).map(|i| (i as f32).sin()).collect();

    group.bench_function("softmax_32k_via_top_p", |b| {
        let params = SamplingParams::default()
            .with_top_p(0.9)
            .with_temperature(1.0);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&logits_32k)))
    });

    // Test with extreme values (stress test numerical stability)
    let extreme_logits: Vec<f32> = (0..32000)
        .map(|i| if i % 2 == 0 { 100.0 } else { -100.0 })
        .collect();

    group.bench_function("softmax_extreme_values", |b| {
        let params = SamplingParams::default()
            .with_top_p(0.9)
            .with_temperature(1.0);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&extreme_logits)))
    });

    // Benchmark argmax (greedy) with uniform distribution
    let uniform_logits: Vec<f32> = vec![1.0; 32000];

    group.bench_function("argmax_uniform_32k", |b| {
        let mut sampler = Sampler::new(SamplingParams::greedy());
        b.iter(|| sampler.sample(black_box(&uniform_logits)))
    });

    group.finish();
}

// ============================================================================
// MESSAGE/CHAT TEMPLATE BENCHMARKS
// ============================================================================

fn chat_template_benchmark(c: &mut Criterion) {
    use infernum_core::Message;

    let mut group = c.benchmark_group("chat_template");

    // Benchmark message creation
    group.bench_function("create_user_message", |b| {
        b.iter(|| Message::user(black_box("Hello, how are you?")))
    });

    group.bench_function("create_system_message", |b| {
        b.iter(|| Message::system(black_box("You are a helpful assistant.")))
    });

    // Benchmark serialization
    group.bench_function("serialize_message", |b| {
        let msg = Message::user("Hello, how are you?");
        b.iter(|| serde_json::to_string(black_box(&msg)).unwrap())
    });

    // Benchmark typical conversation serialization
    group.bench_function("serialize_conversation_5_messages", |b| {
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is Rust?"),
            Message::assistant("Rust is a systems programming language."),
            Message::user("What makes it special?"),
            Message::assistant("Memory safety without garbage collection."),
        ];
        b.iter(|| serde_json::to_string(black_box(&messages)).unwrap())
    });

    group.finish();
}

// ============================================================================
// REQUEST/RESPONSE SERIALIZATION BENCHMARKS
// ============================================================================

fn serialization_benchmark(c: &mut Criterion) {
    use infernum_core::response::{Choice, GenerateResponse};
    use infernum_core::Usage;

    let mut group = c.benchmark_group("serialization");

    // Benchmark response creation and serialization
    group.bench_function("serialize_generate_response", |b| {
        let response = GenerateResponse {
            request_id: infernum_core::RequestId::new(),
            model: infernum_core::ModelId::new("test-model"),
            choices: vec![Choice {
                index: 0,
                text: "Hello, I'm an AI assistant. How can I help you today?".to_string(),
                finish_reason: Some(infernum_core::FinishReason::Stop),
                logprobs: None,
            }],
            usage: Usage::new(10, 15),
            time_to_first_token_ms: Some(50.0),
            total_time_ms: Some(200.0),
        };

        b.iter(|| serde_json::to_string(black_box(&response)).unwrap())
    });

    // Benchmark response with multiple choices
    group.bench_function("serialize_response_with_4_choices", |b| {
        let choices: Vec<Choice> = (0..4)
            .map(|i| Choice {
                index: i,
                text: format!("Generated response number {}", i),
                finish_reason: Some(infernum_core::FinishReason::Stop),
                logprobs: None,
            })
            .collect();

        let response = GenerateResponse {
            request_id: infernum_core::RequestId::new(),
            model: infernum_core::ModelId::new("test-model"),
            choices,
            usage: Usage::new(50, 100),
            time_to_first_token_ms: Some(50.0),
            total_time_ms: Some(500.0),
        };

        b.iter(|| serde_json::to_string(black_box(&response)).unwrap())
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    benches,
    sampling_benchmark,
    vocabulary_scaling_benchmark,
    kv_cache_benchmark,
    streaming_benchmark,
    numerical_benchmark,
    chat_template_benchmark,
    serialization_benchmark,
);
criterion_main!(benches);
