//! Benchmarks for streaming operations.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use infernum_core::{
    streaming::{StreamChunk, StreamChoice, StreamDelta},
    types::{ModelId, RequestId, Usage},
};

/// Creates a test chunk with the given content.
fn create_chunk(content: &str) -> StreamChunk {
    StreamChunk {
        request_id: RequestId::new(),
        model: ModelId::new("benchmark-model"),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                content: Some(content.to_string()),
                token: None,
            },
            finish_reason: None,
        }],
        usage: None,
    }
}

/// Benchmark: StreamChunk creation.
fn bench_chunk_creation(c: &mut Criterion) {
    c.bench_function("StreamChunk::new", |b| {
        b.iter(|| create_chunk("Hello, world!"))
    });
}

/// Benchmark: Collect text from chunks.
fn bench_collect_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("collect_text");

    for num_chunks in [1, 10, 100, 1000] {
        let chunks: Vec<StreamChunk> = (0..num_chunks)
            .map(|i| create_chunk(&format!("Token{} ", i)))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_chunks),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let mut text = String::new();
                    for chunk in chunks {
                        for choice in &chunk.choices {
                            if let Some(content) = &choice.delta.content {
                                text.push_str(content);
                            }
                        }
                    }
                    text
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Usage calculation.
fn bench_usage(c: &mut Criterion) {
    c.bench_function("Usage::new", |b| {
        b.iter(|| Usage::new(1024, 512))
    });
}

criterion_group!(
    benches,
    bench_chunk_creation,
    bench_collect_text,
    bench_usage,
);

criterion_main!(benches);
