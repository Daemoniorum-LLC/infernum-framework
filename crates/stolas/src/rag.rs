//! Retrieval-Augmented Generation pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use infernum_core::Result;

use crate::chunker::Chunker;
use crate::embedding::Embedder;
use crate::store::{SearchParams, VectorRecord, VectorStore};

/// Configuration for retrieval.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Number of documents to retrieve.
    pub top_k: usize,
    /// Minimum similarity score threshold.
    pub min_score: f32,
    /// Enable cross-encoder reranking.
    pub rerank: bool,
    /// Number of documents to fetch before reranking.
    pub rerank_top_k: usize,
    /// Include metadata in context.
    pub include_metadata: bool,
    /// Maximum total context length in characters.
    pub max_context_length: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.5,
            rerank: false,
            rerank_top_k: 20,
            include_metadata: true,
            max_context_length: 8000,
        }
    }
}

/// Source document for ingestion.
#[derive(Debug, Clone)]
pub struct Document {
    /// Unique document identifier.
    pub id: String,
    /// Document content.
    pub content: String,
    /// Document metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// Creates a new document.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the document.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A retrieved context item.
#[derive(Debug, Clone)]
pub struct ContextItem {
    /// The content text.
    pub content: String,
    /// Source document ID.
    pub source_id: String,
    /// Chunk index within the document.
    pub chunk_index: usize,
    /// Similarity score.
    pub score: f32,
    /// Associated metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// RAG pipeline for retrieval-augmented generation.
pub struct RagPipeline {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    chunker: Chunker,
    config: RetrievalConfig,
}

impl RagPipeline {
    /// Creates a new RAG pipeline.
    #[must_use]
    pub fn new(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            embedder,
            store,
            chunker: Chunker::default(),
            config,
        }
    }

    /// Creates a new RAG pipeline with custom chunking.
    #[must_use]
    pub fn with_chunker(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        chunker: Chunker,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            embedder,
            store,
            chunker,
            config,
        }
    }

    /// Ingests a document into the vector store.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or storage fails.
    pub async fn ingest(&self, document: Document) -> Result<usize> {
        let chunks = self.chunker.chunk(&document.content);

        if chunks.is_empty() {
            return Ok(0);
        }

        // Get embeddings for all chunks
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self.embedder.embed(&texts).await?;

        // Create vector records
        let records: Vec<VectorRecord> = chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, embedding)| {
                let mut metadata = document.metadata.clone();
                metadata.insert("source_id".to_string(), serde_json::json!(document.id));
                metadata.insert("chunk_index".to_string(), serde_json::json!(chunk.index));
                metadata.insert("start_offset".to_string(), serde_json::json!(chunk.start));
                metadata.insert("end_offset".to_string(), serde_json::json!(chunk.end));

                VectorRecord {
                    id: format!("{}_{}", document.id, chunk.index),
                    vector: embedding.clone(),
                    content: chunk.text.clone(),
                    metadata,
                }
            })
            .collect();

        let count = self.store.upsert(records).await?;
        Ok(count)
    }

    /// Ingests multiple documents.
    ///
    /// # Errors
    ///
    /// Returns an error if any ingestion fails.
    pub async fn ingest_batch(&self, documents: Vec<Document>) -> Result<usize> {
        let mut total = 0;
        for doc in documents {
            total += self.ingest(doc).await?;
        }
        Ok(total)
    }

    /// Retrieves relevant documents for a query.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or search fails.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<ContextItem>> {
        // Generate query embedding
        let embeddings = self.embedder.embed(&[query]).await?;
        let query_embedding = &embeddings[0];

        // Search vector store - fetch more if reranking
        let fetch_k = if self.config.rerank {
            self.config.rerank_top_k
        } else {
            self.config.top_k
        };

        let params = SearchParams {
            top_k: fetch_k,
            min_score: Some(self.config.min_score),
            ..Default::default()
        };

        let results = self.store.search(query_embedding, params).await?;

        // Convert to context items
        let mut items: Vec<ContextItem> = results
            .into_iter()
            .map(|r| {
                let source_id = r
                    .record
                    .metadata
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let chunk_index = r
                    .record
                    .metadata
                    .get("chunk_index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                ContextItem {
                    content: r.record.content,
                    source_id,
                    chunk_index,
                    score: r.score,
                    metadata: r.record.metadata,
                }
            })
            .collect();

        // Apply reranking if enabled
        if self.config.rerank && items.len() > self.config.top_k {
            items = self.rerank(query, items).await?;
            items.truncate(self.config.top_k);
        }

        Ok(items)
    }

    /// Reranks results using cross-encoder scoring.
    async fn rerank(&self, query: &str, items: Vec<ContextItem>) -> Result<Vec<ContextItem>> {
        // For now, use a simple embedding-based reranking
        // In production, this would use a cross-encoder model
        let texts: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
        let doc_embeddings = self.embedder.embed(&texts).await?;
        let query_embedding = &self.embedder.embed(&[query]).await?[0];

        // Compute cosine similarities
        let mut scored: Vec<(f32, ContextItem)> = items
            .into_iter()
            .zip(doc_embeddings.iter())
            .map(|(item, doc_emb)| {
                let score = cosine_similarity(query_embedding, doc_emb);
                (score, item)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Update scores and return
        Ok(scored
            .into_iter()
            .map(|(score, mut item)| {
                item.score = score;
                item
            })
            .collect())
    }

    /// Builds context from retrieved items.
    #[must_use]
    pub fn build_context(&self, items: &[ContextItem]) -> String {
        let mut context = String::new();
        let mut total_len = 0;

        for (i, item) in items.iter().enumerate() {
            let entry = if self.config.include_metadata {
                format!(
                    "[{}] (source: {}, score: {:.2})\n{}\n\n",
                    i + 1,
                    item.source_id,
                    item.score,
                    item.content
                )
            } else {
                format!("[{}] {}\n\n", i + 1, item.content)
            };

            if total_len + entry.len() > self.config.max_context_length {
                break;
            }

            context.push_str(&entry);
            total_len += entry.len();
        }

        context.trim().to_string()
    }

    /// Augments a prompt with retrieved context.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    pub async fn augment(&self, query: &str, system_prompt: Option<&str>) -> Result<String> {
        let items = self.retrieve(query).await?;
        let context = self.build_context(&items);

        let system = system_prompt.unwrap_or(
            "You are a helpful assistant. Answer questions based on the provided context. \
             If the context doesn't contain relevant information, say so.",
        );

        Ok(format!(
            "{}\n\n---\nRelevant Context:\n{}\n---\n\nQuestion: {}",
            system, context, query
        ))
    }

    /// Creates a formatted prompt with context for chat models.
    ///
    /// # Errors
    ///
    /// Returns an error if retrieval fails.
    pub async fn augment_messages(
        &self,
        query: &str,
        system_prompt: Option<&str>,
    ) -> Result<Vec<infernum_core::Message>> {
        let items = self.retrieve(query).await?;
        let context = self.build_context(&items);

        let system = system_prompt.unwrap_or(
            "You are a helpful assistant. Answer questions based on the provided context.",
        );

        let context_msg = format!(
            "Here is relevant context for the question:\n\n{}",
            context
        );

        Ok(vec![
            infernum_core::Message::system(system),
            infernum_core::Message::system(&context_msg),
            infernum_core::Message::user(query),
        ])
    }

    /// Returns the embedder.
    #[must_use]
    pub fn embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }

    /// Returns the vector store.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn VectorStore> {
        &self.store
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedder;
    use crate::store::InMemoryStore;

    #[tokio::test]
    async fn test_document_ingestion() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        let doc = Document::new("doc1", "This is a test document with some content.")
            .with_metadata("author", serde_json::json!("test"));

        let count = pipeline.ingest(doc).await.unwrap();
        assert!(count > 0);
        assert!(store.count().await.unwrap() > 0);
    }

    #[tokio::test]
    async fn test_retrieval() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0, // Accept all for testing
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "The quick brown fox jumps over the lazy dog.");
        pipeline.ingest(doc).await.unwrap();

        let results = pipeline.retrieve("fox").await.unwrap();
        assert!(!results.is_empty());
    }
}
