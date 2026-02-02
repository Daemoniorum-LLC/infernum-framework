//! Embedding generation with model integration.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use infernum_core::Result;

use abaddon::{Engine, InferenceEngine};

// ============================================================================
// Embedding Metrics (Sprint 7 - Leviathan Feature Parity)
// ============================================================================

/// Metrics for embedding generation operations.
///
/// Thread-safe counters for monitoring embedding performance.
#[derive(Debug, Default)]
pub struct EmbeddingMetrics {
    /// Total number of embedding requests.
    pub requests_total: AtomicU64,
    /// Total number of texts embedded.
    pub texts_embedded_total: AtomicU64,
    /// Total embedding latency in microseconds.
    pub latency_total_us: AtomicU64,
    /// Number of batch operations.
    pub batch_operations: AtomicU64,
    /// Total batch size (sum of all batch sizes).
    pub batch_size_total: AtomicU64,
    /// Number of failed embedding operations.
    pub failures_total: AtomicU64,
    /// Total dimensions processed (texts * dimension).
    pub dimensions_total: AtomicU64,
}

impl EmbeddingMetrics {
    /// Creates a new metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a successful embedding operation.
    pub fn record_success(&self, text_count: u64, dimension: u64, latency_us: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.texts_embedded_total.fetch_add(text_count, Ordering::Relaxed);
        self.latency_total_us.fetch_add(latency_us, Ordering::Relaxed);
        self.dimensions_total.fetch_add(text_count * dimension, Ordering::Relaxed);
    }

    /// Records a batch operation.
    pub fn record_batch(&self, batch_size: u64) {
        self.batch_operations.fetch_add(1, Ordering::Relaxed);
        self.batch_size_total.fetch_add(batch_size, Ordering::Relaxed);
    }

    /// Records a failed embedding operation.
    pub fn record_failure(&self) {
        self.failures_total.fetch_add(1, Ordering::Relaxed);
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the average latency per request in microseconds.
    #[must_use]
    pub fn average_latency_us(&self) -> f64 {
        let total = self.latency_total_us.load(Ordering::Relaxed);
        let requests = self.requests_total.load(Ordering::Relaxed);
        if requests > 0 {
            total as f64 / requests as f64
        } else {
            0.0
        }
    }

    /// Returns the average batch size.
    #[must_use]
    pub fn average_batch_size(&self) -> f64 {
        let total = self.batch_size_total.load(Ordering::Relaxed);
        let ops = self.batch_operations.load(Ordering::Relaxed);
        if ops > 0 {
            total as f64 / ops as f64
        } else {
            0.0
        }
    }

    /// Returns the success rate (0.0 to 1.0).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_total.load(Ordering::Relaxed);
        let failures = self.failures_total.load(Ordering::Relaxed);
        if total > 0 {
            (total - failures) as f64 / total as f64
        } else {
            1.0
        }
    }

    /// Returns metrics in Prometheus text exposition format.
    #[must_use]
    pub fn to_prometheus(&self) -> String {
        format!(
            "# HELP stolas_embeddings_total Total number of texts embedded\n\
             # TYPE stolas_embeddings_total counter\n\
             stolas_embeddings_total {}\n\
             # HELP stolas_embedding_requests_total Total embedding requests\n\
             # TYPE stolas_embedding_requests_total counter\n\
             stolas_embedding_requests_total {}\n\
             # HELP stolas_embedding_latency_us_total Total embedding latency in microseconds\n\
             # TYPE stolas_embedding_latency_us_total counter\n\
             stolas_embedding_latency_us_total {}\n\
             # HELP stolas_embedding_batch_operations_total Total batch operations\n\
             # TYPE stolas_embedding_batch_operations_total counter\n\
             stolas_embedding_batch_operations_total {}\n\
             # HELP stolas_embedding_failures_total Total failed embedding operations\n\
             # TYPE stolas_embedding_failures_total counter\n\
             stolas_embedding_failures_total {}\n\
             # HELP stolas_embedding_dimensions_total Total dimensions processed\n\
             # TYPE stolas_embedding_dimensions_total counter\n\
             stolas_embedding_dimensions_total {}\n",
            self.texts_embedded_total.load(Ordering::Relaxed),
            self.requests_total.load(Ordering::Relaxed),
            self.latency_total_us.load(Ordering::Relaxed),
            self.batch_operations.load(Ordering::Relaxed),
            self.failures_total.load(Ordering::Relaxed),
            self.dimensions_total.load(Ordering::Relaxed),
        )
    }
}

/// Trait for embedding models.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generates embeddings for the given texts.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Generates a single embedding.
    async fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| infernum_core::Error::internal("No embedding generated"))
    }

    /// Returns the embedding dimension.
    fn dimension(&self) -> usize;

    /// Returns the model name.
    fn model_name(&self) -> &str;
}

/// Engine-based embedder using abaddon inference.
pub struct EngineEmbedder {
    engine: Arc<Engine>,
    dimension: usize,
}

impl EngineEmbedder {
    /// Creates a new engine-based embedder.
    ///
    /// The embedding dimension is automatically determined from the model's
    /// hidden size. For most embedding models, the output dimension equals
    /// the hidden size.
    pub fn new(engine: Arc<Engine>) -> Self {
        // Get dimension from model info - hidden_size is the embedding dimension
        let model_info = engine.model_info();
        let dimension = model_info.hidden_size as usize;

        tracing::debug!(
            model_id = %model_info.id.0,
            dimension = dimension,
            "Created engine embedder"
        );

        Self { engine, dimension }
    }

    /// Creates a new embedder with a specified dimension.
    ///
    /// Use this when you need to override the model's default dimension,
    /// for example when using a projection head or pooler.
    pub fn with_dimension(engine: Arc<Engine>, dimension: usize) -> Self {
        Self { engine, dimension }
    }

    /// Returns the underlying engine.
    #[must_use]
    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }
}

#[async_trait]
impl Embedder for EngineEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let request = infernum_core::EmbedRequest::new(text.to_string());
            let response = self.engine.embed(request).await?;

            // Extract the embedding from the response
            let embedding = response
                .data
                .into_iter()
                .next()
                .ok_or_else(|| infernum_core::Error::internal("No embedding in response"))?;

            let vec = embedding
                .embedding
                .as_floats()
                .map_err(|e| infernum_core::Error::internal(e))?;

            embeddings.push(vec);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.engine.model_info().id.0
    }
}

/// Mock embedder for testing.
pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    /// Creates a new mock embedder.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Generate deterministic pseudo-embeddings based on text hash
        Ok(texts
            .iter()
            .map(|text| {
                let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                (0..self.dimension)
                    .map(|i| {
                        let seed = hash.wrapping_add(i as u64);
                        ((seed % 1000) as f32 / 1000.0) - 0.5
                    })
                    .collect()
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

/// Sentence embedder with pooling strategies.
pub struct SentenceEmbedder {
    embedder: Arc<dyn Embedder>,
    pooling: PoolingStrategy,
    normalize: bool,
}

/// Pooling strategy for multi-token embeddings.
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    /// Use the [CLS] token embedding.
    Cls,
    /// Average all token embeddings.
    #[default]
    Mean,
    /// Use the last token embedding.
    Last,
    /// Max pooling across tokens.
    Max,
}

impl SentenceEmbedder {
    /// Creates a new sentence embedder.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }

    /// Sets the pooling strategy.
    #[must_use]
    pub fn with_pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling = strategy;
        self
    }

    /// Sets whether to normalize embeddings.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Normalizes a vector to unit length.
    fn normalize_vec(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[async_trait]
impl Embedder for SentenceEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = self.embedder.embed(texts).await?;

        if self.normalize {
            for emb in embeddings.iter_mut() {
                Self::normalize_vec(emb);
            }
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    fn model_name(&self) -> &str {
        self.embedder.model_name()
    }
}

/// Batch embedder for efficient processing of large document collections.
pub struct BatchEmbedder {
    embedder: Arc<dyn Embedder>,
    batch_size: usize,
}

impl BatchEmbedder {
    /// Creates a new batch embedder.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            batch_size: 32,
        }
    }

    /// Sets the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Embeds a large collection of texts in batches.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let batch_embeddings = self.embedder.embed(&refs).await?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }
}

/// Computes cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Computes Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Computes dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// Metered Embedder (Sprint 7 - Observability)
// ============================================================================

/// An embedder wrapper that records metrics for all operations.
///
/// Wraps any `Embedder` implementation and tracks:
/// - Request counts
/// - Latency (microseconds)
/// - Batch sizes
/// - Failure rates
pub struct MeteredEmbedder {
    inner: Arc<dyn Embedder>,
    metrics: Arc<EmbeddingMetrics>,
}

impl MeteredEmbedder {
    /// Creates a new metered embedder.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            inner: embedder,
            metrics: Arc::new(EmbeddingMetrics::new()),
        }
    }

    /// Creates with shared metrics.
    pub fn with_metrics(embedder: Arc<dyn Embedder>, metrics: Arc<EmbeddingMetrics>) -> Self {
        Self {
            inner: embedder,
            metrics,
        }
    }

    /// Returns the metrics instance.
    #[must_use]
    pub fn metrics(&self) -> &Arc<EmbeddingMetrics> {
        &self.metrics
    }
}

#[async_trait]
impl Embedder for MeteredEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let text_count = texts.len() as u64;

        match self.inner.embed(texts).await {
            Ok(embeddings) => {
                let latency_us = start.elapsed().as_micros() as u64;
                let dimension = self.inner.dimension() as u64;
                self.metrics.record_success(text_count, dimension, latency_us);
                if text_count > 1 {
                    self.metrics.record_batch(text_count);
                }
                Ok(embeddings)
            }
            Err(e) => {
                self.metrics.record_failure();
                Err(e)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new(384);
        let embeddings = embedder.embed(&["hello", "world"]).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        // Same text should produce same embedding
        let emb1 = embedder.embed(&["hello"]).await.unwrap();
        let emb2 = embedder.embed(&["hello"]).await.unwrap();
        assert_eq!(emb1[0], emb2[0]);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &c) - 5.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_sentence_embedder_normalization() {
        let mock = Arc::new(MockEmbedder::new(3));
        let embedder = SentenceEmbedder::new(mock);

        let embeddings = embedder.embed(&["test"]).await.unwrap();

        // Check normalization
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // Embedding Metrics Tests
    // ========================================================================

    #[test]
    fn test_embedding_metrics_new() {
        let metrics = EmbeddingMetrics::new();
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.texts_embedded_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.failures_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_embedding_metrics_record_success() {
        let metrics = EmbeddingMetrics::new();
        metrics.record_success(5, 384, 1000);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.texts_embedded_total.load(Ordering::Relaxed), 5);
        assert_eq!(metrics.latency_total_us.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.dimensions_total.load(Ordering::Relaxed), 5 * 384);
    }

    #[test]
    fn test_embedding_metrics_record_batch() {
        let metrics = EmbeddingMetrics::new();
        metrics.record_batch(10);
        metrics.record_batch(20);

        assert_eq!(metrics.batch_operations.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.batch_size_total.load(Ordering::Relaxed), 30);
        assert!((metrics.average_batch_size() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_metrics_record_failure() {
        let metrics = EmbeddingMetrics::new();
        metrics.record_success(1, 384, 500);
        metrics.record_failure();

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.failures_total.load(Ordering::Relaxed), 1);
        assert!((metrics.success_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_embedding_metrics_average_latency() {
        let metrics = EmbeddingMetrics::new();
        metrics.record_success(1, 384, 1000);
        metrics.record_success(1, 384, 2000);

        assert!((metrics.average_latency_us() - 1500.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_metrics_prometheus_format() {
        let metrics = EmbeddingMetrics::new();
        metrics.record_success(10, 384, 5000);

        let prometheus = metrics.to_prometheus();
        assert!(prometheus.contains("stolas_embeddings_total 10"));
        assert!(prometheus.contains("stolas_embedding_requests_total 1"));
        assert!(prometheus.contains("stolas_embedding_latency_us_total 5000"));
        assert!(prometheus.contains("# TYPE stolas_embeddings_total counter"));
    }

    #[tokio::test]
    async fn test_metered_embedder_records_metrics() {
        let mock = Arc::new(MockEmbedder::new(384));
        let metered = MeteredEmbedder::new(mock);

        // Single embed
        metered.embed(&["hello"]).await.unwrap();
        assert_eq!(metered.metrics().requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metered.metrics().texts_embedded_total.load(Ordering::Relaxed), 1);

        // Batch embed
        metered.embed(&["a", "b", "c"]).await.unwrap();
        assert_eq!(metered.metrics().requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(metered.metrics().texts_embedded_total.load(Ordering::Relaxed), 4);
        assert_eq!(metered.metrics().batch_operations.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_metered_embedder_shared_metrics() {
        let metrics = Arc::new(EmbeddingMetrics::new());
        let mock1 = Arc::new(MockEmbedder::new(384));
        let mock2 = Arc::new(MockEmbedder::new(384));

        let metered1 = MeteredEmbedder::with_metrics(mock1, metrics.clone());
        let metered2 = MeteredEmbedder::with_metrics(mock2, metrics.clone());

        metered1.embed(&["hello"]).await.unwrap();
        metered2.embed(&["world"]).await.unwrap();

        // Both use the same metrics instance
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.texts_embedded_total.load(Ordering::Relaxed), 2);
    }
}
