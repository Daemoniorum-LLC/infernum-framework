//! Retrieval-Augmented Generation pipeline.

use std::sync::Arc;

use infernum_core::Result;

use crate::embedding::Embedder;
use crate::store::{SearchParams, SearchResult, VectorStore};

/// Configuration for retrieval.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of documents to retrieve.
    pub top_k: usize,
    /// Minimum similarity score.
    pub min_score: f32,
    /// Enable reranking.
    pub rerank: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.5,
            rerank: false,
        }
    }
}

/// RAG pipeline for retrieval-augmented generation.
pub struct RagPipeline {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    config: RetrievalConfig,
}

impl RagPipeline {
    /// Creates a new RAG pipeline.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            embedder,
            store,
            config,
        }
    }

    /// Retrieves relevant documents for a query.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or search fails.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let embeddings = self.embedder.embed(&[query]).await?;
        let query_embedding = &embeddings[0];

        // Search vector store
        let params = SearchParams {
            top_k: self.config.top_k,
            min_score: Some(self.config.min_score),
            ..Default::default()
        };

        let results = self.store.search(query_embedding, params).await?;

        // TODO: Implement reranking if enabled

        Ok(results)
    }

    /// Builds context from retrieved documents.
    #[must_use]
    pub fn build_context(&self, results: &[SearchResult]) -> String {
        results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] {}", i + 1, r.record.content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Augments a prompt with retrieved context.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    pub async fn augment(&self, query: &str, system_prompt: Option<&str>) -> Result<String> {
        let results = self.retrieve(query).await?;
        let context = self.build_context(&results);

        let system = system_prompt.unwrap_or(
            "You are a helpful assistant. Answer questions based on the provided context.",
        );

        Ok(format!(
            "{}\n\nContext:\n{}\n\nQuestion: {}",
            system, context, query
        ))
    }
}
