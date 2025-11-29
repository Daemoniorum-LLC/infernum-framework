//! Benchmarks for inference performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn sampling_benchmark(c: &mut Criterion) {
    use abaddon::sampler::Sampler;
    use infernum_core::SamplingParams;

    let logits: Vec<f32> = (0..32000).map(|i| (i as f32).sin()).collect();

    c.bench_function("greedy_sampling", |b| {
        let mut sampler = Sampler::new(SamplingParams::greedy());
        b.iter(|| sampler.sample(black_box(&logits)))
    });

    c.bench_function("top_p_sampling", |b| {
        let mut sampler = Sampler::new(SamplingParams::balanced());
        b.iter(|| sampler.sample(black_box(&logits)))
    });

    c.bench_function("top_k_sampling", |b| {
        let params = SamplingParams::default().with_top_k(50);
        let mut sampler = Sampler::new(params);
        b.iter(|| sampler.sample(black_box(&logits)))
    });
}

criterion_group!(benches, sampling_benchmark);
criterion_main!(benches);
