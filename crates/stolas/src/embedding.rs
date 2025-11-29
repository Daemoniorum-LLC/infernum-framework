//! Embedding generation.

use async_trait::async_trait;
use infernum_core::Result;

/// Trait for embedding models.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generates embeddings for the given texts.
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Returns the embedding dimension.
    fn dimension(&self) -> usize;

    /// Returns the model name.
    fn model_name(&self) -> &str;
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
