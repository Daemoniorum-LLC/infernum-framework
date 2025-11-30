//! Embedding generation with model integration.

use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::Result;

use abaddon::{Engine, InferenceEngine};

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
}
