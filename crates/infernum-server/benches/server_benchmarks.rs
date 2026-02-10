//! Criterion benchmarks for infernum-server.
//!
//! Run with: `cargo bench -p infernum-server`
//!
//! These benchmarks measure:
//! - Request validation performance
//! - Error response generation
//! - Rate limiter throughput
//! - Observability metrics recording

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use infernum_server::{
    error_response::{api_error, ErrorCode},
    observability::ObservabilityState,
    security::{RateLimitConfig, RateLimiter},
    server::ValidationLimits,
};
use std::hint::black_box;
use std::time::Duration;

// ============================================================================
// Request Validation Benchmarks
// ============================================================================

/// Benchmarks chat completion request validation.
fn bench_chat_request_validation(c: &mut Criterion) {
    use infernum_server::api_types::ChatCompletionRequest;

    let _limits = ValidationLimits::default();

    // Small request (1 message)
    let small_request = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    });

    // Medium request (10 messages)
    let medium_messages: Vec<serde_json::Value> = (0..10)
        .map(|i| {
            serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("Message {}", i)
            })
        })
        .collect();
    let medium_request = serde_json::json!({
        "model": "test-model",
        "messages": medium_messages,
        "temperature": 0.7,
        "max_tokens": 1000
    });

    // Large request (100 messages)
    let large_messages: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("This is message number {} with some content.", i)
            })
        })
        .collect();
    let large_request = serde_json::json!({
        "model": "test-model",
        "messages": large_messages,
        "temperature": 0.7,
        "max_tokens": 4096
    });

    let mut group = c.benchmark_group("chat_request_validation");
    group.throughput(Throughput::Elements(1));

    // Benchmark deserialization
    group.bench_function("deserialize_small", |b| {
        b.iter(|| {
            let _req: ChatCompletionRequest =
                serde_json::from_value(black_box(small_request.clone())).unwrap();
        });
    });

    group.bench_function("deserialize_medium", |b| {
        b.iter(|| {
            let _req: ChatCompletionRequest =
                serde_json::from_value(black_box(medium_request.clone())).unwrap();
        });
    });

    group.bench_function("deserialize_large", |b| {
        b.iter(|| {
            let _req: ChatCompletionRequest =
                serde_json::from_value(black_box(large_request.clone())).unwrap();
        });
    });

    group.finish();
}

// ============================================================================
// Error Response Benchmarks
// ============================================================================

/// Benchmarks error response generation.
fn bench_error_responses(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_responses");
    group.throughput(Throughput::Elements(1));

    group.bench_function("bad_request", |b| {
        b.iter(|| {
            let _err = api_error(black_box(ErrorCode::InvalidModel), "req-bench-001");
        });
    });

    group.bench_function("rate_limited", |b| {
        b.iter(|| {
            let _err = api_error(black_box(ErrorCode::RateLimited), "req-bench-002");
        });
    });

    group.bench_function("internal_error", |b| {
        b.iter(|| {
            let _err = api_error(black_box(ErrorCode::InternalError), "req-bench-003");
        });
    });

    group.finish();
}

// ============================================================================
// Rate Limiter Benchmarks
// ============================================================================

/// Benchmarks rate limiter check performance.
fn bench_rate_limiter(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let config = RateLimitConfig::new(1000, Duration::from_secs(60));
    let limiter = RateLimiter::new(config);

    let mut group = c.benchmark_group("rate_limiter");
    group.throughput(Throughput::Elements(1));

    // Single client
    group.bench_function("check_single_client", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = limiter.check(black_box("client-1")).await;
        });
    });

    // Multiple clients (simulates load distribution)
    group.bench_function("check_multiple_clients", |b| {
        let mut client_id = 0u64;
        b.to_async(&rt).iter(|| {
            let id = format!("client-{}", client_id % 100);
            client_id += 1;
            let limiter = limiter.clone();
            async move {
                let _ = limiter.check(black_box(&id)).await;
            }
        });
    });

    group.finish();

    // Disabled limiter (baseline)
    let disabled_config = RateLimitConfig::disabled();
    let disabled_limiter = RateLimiter::new(disabled_config);

    let mut group = c.benchmark_group("rate_limiter_disabled");
    group.throughput(Throughput::Elements(1));

    group.bench_function("check_disabled", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = disabled_limiter.check(black_box("client-1")).await;
        });
    });

    group.finish();
}

// ============================================================================
// Observability Benchmarks
// ============================================================================

/// Benchmarks metrics operations.
fn bench_observability(c: &mut Criterion) {
    let state = ObservabilityState::new();

    let mut group = c.benchmark_group("observability");
    group.throughput(Throughput::Elements(1));

    // Benchmark counter reads
    group.bench_function("read_counters", |b| {
        b.iter(|| {
            let _ = black_box(state.total_requests());
            let _ = black_box(state.server_errors());
            let _ = black_box(state.client_errors());
        });
    });

    group.finish();

    // Metrics rendering
    let mut group = c.benchmark_group("metrics_render");

    group.bench_function("render_prometheus", |b| {
        b.iter(|| {
            let _ = black_box(state.render_http_metrics());
        });
    });

    group.finish();
}

// ============================================================================
// JSON Serialization Benchmarks
// ============================================================================

/// Benchmarks response serialization.
fn bench_json_serialization(c: &mut Criterion) {
    use infernum_server::api_types::{ChatChoice, ChatCompletionResponse, ChatMessage, Usage};

    // Small response
    let small_response = ChatCompletionResponse {
        id: "chatcmpl-test123".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage::new(10, 5),
    };

    // Large response (longer content)
    let large_content = "This is a much longer response that contains ".repeat(100);
    let large_response = ChatCompletionResponse {
        id: "chatcmpl-large123".to_string(),
        object: "chat.completion".to_string(),
        created: 1700000000,
        model: "test-model".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: large_content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage::new(100, 500),
    };

    let mut group = c.benchmark_group("json_serialization");
    group.throughput(Throughput::Elements(1));

    group.bench_function("serialize_small_response", |b| {
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&small_response)).unwrap();
        });
    });

    group.bench_function("serialize_large_response", |b| {
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&large_response)).unwrap();
        });
    });

    group.finish();
}

// ============================================================================
// Throughput Simulation Benchmarks
// ============================================================================

/// Simulates request processing throughput.
fn bench_request_throughput(c: &mut Criterion) {
    use infernum_server::api_types::ChatCompletionRequest;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let limiter = RateLimiter::new(RateLimitConfig::high_throughput());

    // Typical request
    let request_json = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    });

    let mut group = c.benchmark_group("request_throughput");
    group.throughput(Throughput::Elements(1));

    // Simulate request processing (without actual inference)
    group.bench_function("process_request_pipeline", |b| {
        b.to_async(&rt).iter(|| {
            let json = request_json.clone();
            let limiter = limiter.clone();
            async move {
                // 1. Rate limit check
                let _ = limiter.check("client-1").await;

                // 2. Deserialize request
                let _req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_chat_request_validation,
    bench_error_responses,
    bench_rate_limiter,
    bench_observability,
    bench_json_serialization,
    bench_request_throughput,
);

criterion_main!(benches);
