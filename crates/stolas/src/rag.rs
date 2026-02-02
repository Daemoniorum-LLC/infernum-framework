//! Retrieval-Augmented Generation pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use infernum_core::Result;

use crate::bm25::{BM25Config, HybridRetriever};
use crate::chunker::Chunker;
use crate::cross_encoder::{CrossEncoder, HeuristicCrossEncoder};
use crate::embedding::Embedder;
use crate::store::{SearchParams, VectorRecord, VectorStore};
use parking_lot::RwLock;

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
    /// Enable hybrid search (BM25 + dense retrieval).
    pub hybrid_search: bool,
    /// Weight for BM25 sparse scores (0.0 - 1.0).
    pub bm25_weight: f32,
    /// Weight for dense embedding scores (0.0 - 1.0).
    pub dense_weight: f32,
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
            hybrid_search: false,
            bm25_weight: 0.3,
            dense_weight: 0.7,
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
    /// Optional cross-encoder for improved reranking.
    cross_encoder: Option<Arc<dyn CrossEncoder>>,
    /// Optional hybrid retriever for BM25 + dense search.
    hybrid_retriever: Option<RwLock<HybridRetriever>>,
}

impl RagPipeline {
    /// Creates a new RAG pipeline.
    #[must_use]
    pub fn new(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        config: RetrievalConfig,
    ) -> Self {
        // Use heuristic cross-encoder by default if reranking is enabled
        let cross_encoder: Option<Arc<dyn CrossEncoder>> = if config.rerank {
            Some(Arc::new(HeuristicCrossEncoder::new()))
        } else {
            None
        };

        // Create hybrid retriever if hybrid search is enabled
        let hybrid_retriever = if config.hybrid_search {
            Some(RwLock::new(HybridRetriever::new(
                BM25Config::default(),
                config.bm25_weight,
                config.dense_weight,
            )))
        } else {
            None
        };

        Self {
            embedder,
            store,
            chunker: Chunker::default(),
            config,
            cross_encoder,
            hybrid_retriever,
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
        let cross_encoder: Option<Arc<dyn CrossEncoder>> = if config.rerank {
            Some(Arc::new(HeuristicCrossEncoder::new()))
        } else {
            None
        };

        let hybrid_retriever = if config.hybrid_search {
            Some(RwLock::new(HybridRetriever::new(
                BM25Config::default(),
                config.bm25_weight,
                config.dense_weight,
            )))
        } else {
            None
        };

        Self {
            embedder,
            store,
            chunker,
            config,
            cross_encoder,
            hybrid_retriever,
        }
    }

    /// Creates a RAG pipeline with a custom cross-encoder.
    #[must_use]
    pub fn with_cross_encoder(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        config: RetrievalConfig,
        cross_encoder: Arc<dyn CrossEncoder>,
    ) -> Self {
        let hybrid_retriever = if config.hybrid_search {
            Some(RwLock::new(HybridRetriever::new(
                BM25Config::default(),
                config.bm25_weight,
                config.dense_weight,
            )))
        } else {
            None
        };

        Self {
            embedder,
            store,
            chunker: Chunker::default(),
            config,
            cross_encoder: Some(cross_encoder),
            hybrid_retriever,
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

        // Add chunks to BM25 index for hybrid search
        if let Some(ref hybrid_retriever) = self.hybrid_retriever {
            let mut retriever = hybrid_retriever.write();
            for record in &records {
                retriever.add_document(&record.id, &record.content);
            }
        }

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

    /// Deletes all chunks for a document by source ID.
    ///
    /// This removes all chunks from the vector store and BM25 index
    /// that belong to the specified document.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub async fn delete_document(&self, source_id: &str) -> Result<usize> {
        // Find all chunk IDs that belong to this document
        // Chunks have IDs like "{source_id}_{chunk_index}"
        let prefix = format!("{}_", source_id);

        // Get all records to find matching IDs
        // Note: This is inefficient for large stores but works for the current VectorStore trait
        // A production implementation would add a query-by-prefix method
        let mut ids_to_delete = Vec::new();

        // We need to search with a broad query to find all documents
        // Then filter by source_id in metadata
        let params = crate::store::SearchParams {
            top_k: 10000, // Large limit to get all
            min_score: None,
            ..Default::default()
        };

        // Use a zero vector search to get all records (hacky but works)
        let dim = self.embedder.dimension();
        let zero_query = vec![0.0; dim];
        let all_results = self.store.search(&zero_query, params).await?;

        for result in all_results {
            // Check if this chunk belongs to the document
            if result.record.id.starts_with(&prefix) {
                ids_to_delete.push(result.record.id.clone());
            }
        }

        if ids_to_delete.is_empty() {
            return Ok(0);
        }

        // Remove from BM25 index if hybrid search is enabled
        if let Some(ref hybrid_retriever) = self.hybrid_retriever {
            let mut retriever = hybrid_retriever.write();
            for id in &ids_to_delete {
                retriever.bm25_index_mut().remove_document(id);
            }
        }

        // Delete from vector store
        let deleted = self.store.delete(ids_to_delete).await?;
        Ok(deleted)
    }

    /// Updates a document by deleting old chunks and ingesting new content.
    ///
    /// This is equivalent to calling `delete_document()` followed by `ingest()`,
    /// ensuring that old chunks are properly removed even if the document
    /// gets shorter.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion or ingestion fails.
    pub async fn update_document(&self, document: Document) -> Result<usize> {
        // First delete the old document
        self.delete_document(&document.id).await?;

        // Then ingest the new version
        self.ingest(document).await
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

        // Search vector store - fetch more if reranking or hybrid search
        let fetch_k = if self.config.rerank || self.config.hybrid_search {
            self.config.rerank_top_k.max(self.config.top_k * 2)
        } else {
            self.config.top_k
        };

        let params = SearchParams {
            top_k: fetch_k,
            min_score: Some(self.config.min_score),
            ..Default::default()
        };

        let results = self.store.search(query_embedding, params).await?;

        // Apply hybrid search if enabled
        let results = if self.config.hybrid_search {
            if let Some(ref hybrid_retriever) = self.hybrid_retriever {
                // Prepare dense results for hybrid scoring
                let dense_results: Vec<(String, f32)> = results
                    .iter()
                    .map(|r| (r.record.id.clone(), r.score))
                    .collect();

                // Get hybrid-ranked results
                let retriever = hybrid_retriever.read();
                let hybrid_results = retriever.hybrid_search(query, &dense_results, fetch_k);

                // Build a map of record IDs to records for lookup
                let record_map: HashMap<String, _> = results
                    .into_iter()
                    .map(|r| (r.record.id.clone(), r))
                    .collect();

                // Reorder based on hybrid scores
                hybrid_results
                    .into_iter()
                    .filter_map(|hr| {
                        record_map.get(&hr.id).map(|r| {
                            crate::store::SearchResult {
                                record: r.record.clone(),
                                score: hr.hybrid_score,
                            }
                        })
                    })
                    .collect()
            } else {
                results
            }
        } else {
            results
        };

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

        // Truncate to top_k if not reranking
        if !self.config.rerank {
            items.truncate(self.config.top_k);
        }

        Ok(items)
    }

    /// Reranks results using cross-encoder scoring.
    async fn rerank(&self, query: &str, items: Vec<ContextItem>) -> Result<Vec<ContextItem>> {
        // Use cross-encoder if available
        if let Some(ref cross_encoder) = self.cross_encoder {
            let texts: Vec<&str> = items.iter().map(|i| i.content.as_str()).collect();
            let scores = cross_encoder.score_batch(query, &texts).await?;

            let mut scored: Vec<(f32, ContextItem)> = items
                .into_iter()
                .zip(scores.into_iter())
                .map(|(item, score)| (score, item))
                .collect();

            // Sort by score descending
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            return Ok(scored
                .into_iter()
                .map(|(score, mut item)| {
                    item.score = score;
                    item
                })
                .collect());
        }

        // Fallback: use embedding-based reranking
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

    /// Sets the cross-encoder for reranking.
    pub fn set_cross_encoder(&mut self, cross_encoder: Arc<dyn CrossEncoder>) {
        self.cross_encoder = Some(cross_encoder);
    }

    /// Returns the cross-encoder if configured.
    #[must_use]
    pub fn cross_encoder(&self) -> Option<&Arc<dyn CrossEncoder>> {
        self.cross_encoder.as_ref()
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

        let context_msg = format!("Here is relevant context for the question:\n\n{}", context);

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

    /// Returns the hybrid retriever if configured.
    #[must_use]
    pub fn hybrid_retriever(&self) -> Option<&RwLock<HybridRetriever>> {
        self.hybrid_retriever.as_ref()
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
    use crate::chunker::{Chunker, ChunkingStrategy};
    use crate::cross_encoder::MockCrossEncoder;
    use crate::embedding::MockEmbedder;
    use crate::store::InMemoryStore;

    // ==========================================================================
    // RetrievalConfig Tests
    // ==========================================================================

    #[test]
    fn test_retrieval_config_default() {
        let config = RetrievalConfig::default();
        assert_eq!(config.top_k, 5);
        assert!((config.min_score - 0.5).abs() < 0.001);
        assert!(!config.rerank);
        assert_eq!(config.rerank_top_k, 20);
        assert!(config.include_metadata);
        assert_eq!(config.max_context_length, 8000);
    }

    #[test]
    fn test_retrieval_config_custom() {
        let config = RetrievalConfig {
            top_k: 10,
            min_score: 0.7,
            rerank: true,
            rerank_top_k: 50,
            include_metadata: false,
            max_context_length: 4000,
            ..Default::default()
        };
        assert_eq!(config.top_k, 10);
        assert!((config.min_score - 0.7).abs() < 0.001);
        assert!(config.rerank);
        assert_eq!(config.rerank_top_k, 50);
        assert!(!config.include_metadata);
        assert_eq!(config.max_context_length, 4000);
    }

    #[test]
    fn test_retrieval_config_clone() {
        let config1 = RetrievalConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.top_k, config2.top_k);
        assert_eq!(config1.min_score, config2.min_score);
    }

    // ==========================================================================
    // Document Tests
    // ==========================================================================

    #[test]
    fn test_document_new() {
        let doc = Document::new("doc123", "Hello, world!");
        assert_eq!(doc.id, "doc123");
        assert_eq!(doc.content, "Hello, world!");
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_document_new_with_into() {
        let doc = Document::new(String::from("id"), String::from("content"));
        assert_eq!(doc.id, "id");
        assert_eq!(doc.content, "content");
    }

    #[test]
    fn test_document_with_metadata() {
        let doc = Document::new("doc1", "content")
            .with_metadata("author", serde_json::json!("Alice"))
            .with_metadata("version", serde_json::json!(1));

        assert_eq!(doc.metadata.len(), 2);
        assert_eq!(doc.metadata.get("author").unwrap(), &serde_json::json!("Alice"));
        assert_eq!(doc.metadata.get("version").unwrap(), &serde_json::json!(1));
    }

    #[test]
    fn test_document_with_complex_metadata() {
        let doc = Document::new("doc1", "content")
            .with_metadata("tags", serde_json::json!(["rust", "programming"]))
            .with_metadata("nested", serde_json::json!({"key": "value"}));

        assert_eq!(doc.metadata.len(), 2);
        assert!(doc.metadata.get("tags").unwrap().is_array());
        assert!(doc.metadata.get("nested").unwrap().is_object());
    }

    #[test]
    fn test_document_clone() {
        let doc1 = Document::new("doc1", "content").with_metadata("key", serde_json::json!("value"));
        let doc2 = doc1.clone();
        assert_eq!(doc1.id, doc2.id);
        assert_eq!(doc1.content, doc2.content);
        assert_eq!(doc1.metadata, doc2.metadata);
    }

    // ==========================================================================
    // ContextItem Tests
    // ==========================================================================

    #[test]
    fn test_context_item_structure() {
        let item = ContextItem {
            content: "Test content".to_string(),
            source_id: "source123".to_string(),
            chunk_index: 5,
            score: 0.85,
            metadata: HashMap::new(),
        };

        assert_eq!(item.content, "Test content");
        assert_eq!(item.source_id, "source123");
        assert_eq!(item.chunk_index, 5);
        assert!((item.score - 0.85).abs() < 0.001);
        assert!(item.metadata.is_empty());
    }

    #[test]
    fn test_context_item_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), serde_json::json!("value"));

        let item = ContextItem {
            content: "content".to_string(),
            source_id: "source".to_string(),
            chunk_index: 0,
            score: 0.9,
            metadata,
        };

        assert_eq!(item.metadata.len(), 1);
    }

    #[test]
    fn test_context_item_clone() {
        let item1 = ContextItem {
            content: "content".to_string(),
            source_id: "source".to_string(),
            chunk_index: 1,
            score: 0.75,
            metadata: HashMap::new(),
        };
        let item2 = item1.clone();
        assert_eq!(item1.content, item2.content);
        assert_eq!(item1.source_id, item2.source_id);
        assert_eq!(item1.chunk_index, item2.chunk_index);
        assert_eq!(item1.score, item2.score);
    }

    // ==========================================================================
    // RagPipeline Creation Tests
    // ==========================================================================

    #[test]
    fn test_rag_pipeline_new() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        assert_eq!(pipeline.embedder().dimension(), 384);
        assert!(pipeline.cross_encoder().is_none()); // rerank is false by default
    }

    #[test]
    fn test_rag_pipeline_new_with_rerank() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            rerank: true,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // With rerank enabled, cross_encoder should be set
        assert!(pipeline.cross_encoder().is_some());
    }

    #[test]
    fn test_rag_pipeline_with_chunker() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 50,
            max_size: 200,
        });
        let config = RetrievalConfig::default();

        let pipeline = RagPipeline::with_chunker(embedder, store, chunker, config);
        assert_eq!(pipeline.embedder().dimension(), 384);
    }

    #[test]
    fn test_rag_pipeline_with_cross_encoder() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let cross_encoder = Arc::new(MockCrossEncoder::new());

        let pipeline = RagPipeline::with_cross_encoder(embedder, store, config, cross_encoder);
        assert!(pipeline.cross_encoder().is_some());
    }

    // ==========================================================================
    // RagPipeline Accessors Tests
    // ==========================================================================

    #[test]
    fn test_rag_pipeline_embedder() {
        let embedder = Arc::new(MockEmbedder::new(768));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        assert_eq!(pipeline.embedder().dimension(), 768);
        assert_eq!(pipeline.embedder().model_name(), "mock-embedder");
    }

    #[tokio::test]
    async fn test_rag_pipeline_store() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        // Store should be accessible and functional
        let count = pipeline.store().count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_rag_pipeline_config() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            top_k: 20,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        assert_eq!(pipeline.config().top_k, 20);
    }

    // ==========================================================================
    // Ingestion Tests
    // ==========================================================================

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
    async fn test_ingest_empty_document() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        let doc = Document::new("empty", "");
        let count = pipeline.ingest(doc).await.unwrap();

        // Empty document should produce 0 chunks
        assert_eq!(count, 0);
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_ingest_with_custom_chunker() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 20,
            overlap: 5,
        });
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::with_chunker(embedder, store.clone(), chunker, config);

        let doc = Document::new("doc1", "This is a longer test document that should be split into multiple chunks by the chunker.");
        let count = pipeline.ingest(doc).await.unwrap();

        assert!(count > 1); // Should produce multiple chunks with size 20
    }

    #[tokio::test]
    async fn test_ingest_batch() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        let docs = vec![
            Document::new("doc1", "First document content here."),
            Document::new("doc2", "Second document with different content."),
            Document::new("doc3", "Third document, also with unique content."),
        ];

        let count = pipeline.ingest_batch(docs).await.unwrap();
        assert!(count >= 3); // At least 3 chunks (one per doc minimum)
    }

    #[tokio::test]
    async fn test_ingest_preserves_metadata() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: -1.0, // Accept all scores for testing (MockEmbedder may produce low similarity)
            top_k: 10,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        let doc = Document::new("doc1", "Test document content.")
            .with_metadata("author", serde_json::json!("Alice"))
            .with_metadata("category", serde_json::json!("testing"));

        let count = pipeline.ingest(doc).await.unwrap();
        assert!(count > 0, "Document should be ingested");

        // Verify the store has records
        let store_count = store.count().await.unwrap();
        assert!(store_count > 0, "Store should have records after ingestion");

        // Get records directly from store to verify metadata
        let results = pipeline.retrieve("Test document content").await.unwrap();
        assert!(!results.is_empty(), "Should retrieve at least one result");

        // Metadata should be preserved in the result
        assert!(results[0].metadata.contains_key("author"));
        assert!(results[0].metadata.contains_key("source_id"));
    }

    // ==========================================================================
    // Retrieval Tests
    // ==========================================================================

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

    #[tokio::test]
    async fn test_retrieve_respects_top_k() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            top_k: 2,
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Ingest multiple documents
        for i in 0..5 {
            let doc = Document::new(format!("doc{}", i), format!("Document number {} content.", i));
            pipeline.ingest(doc).await.unwrap();
        }

        let results = pipeline.retrieve("document").await.unwrap();
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_retrieve_respects_min_score() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.99, // Very high threshold
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "Some content here.");
        pipeline.ingest(doc).await.unwrap();

        let results = pipeline.retrieve("random query").await.unwrap();
        // With a very high min_score threshold, may get no results
        for result in &results {
            assert!(result.score >= 0.99);
        }
    }

    #[tokio::test]
    async fn test_retrieve_with_reranking() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            top_k: 2,
            min_score: 0.0,
            rerank: true,
            rerank_top_k: 10,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        for i in 0..5 {
            let doc = Document::new(format!("doc{}", i), format!("Document {} about programming.", i));
            pipeline.ingest(doc).await.unwrap();
        }

        let results = pipeline.retrieve("programming").await.unwrap();
        // Should return at most top_k results after reranking
        assert!(results.len() <= 2);
    }

    // ==========================================================================
    // Build Context Tests
    // ==========================================================================

    #[test]
    fn test_build_context_with_metadata() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            include_metadata: true,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let items = vec![
            ContextItem {
                content: "First item content".to_string(),
                source_id: "source1".to_string(),
                chunk_index: 0,
                score: 0.95,
                metadata: HashMap::new(),
            },
            ContextItem {
                content: "Second item content".to_string(),
                source_id: "source2".to_string(),
                chunk_index: 0,
                score: 0.85,
                metadata: HashMap::new(),
            },
        ];

        let context = pipeline.build_context(&items);
        assert!(context.contains("[1]"));
        assert!(context.contains("source1"));
        assert!(context.contains("0.95"));
        assert!(context.contains("First item content"));
        assert!(context.contains("[2]"));
        assert!(context.contains("source2"));
    }

    #[test]
    fn test_build_context_without_metadata() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            include_metadata: false,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let items = vec![ContextItem {
            content: "Item content here".to_string(),
            source_id: "source1".to_string(),
            chunk_index: 0,
            score: 0.9,
            metadata: HashMap::new(),
        }];

        let context = pipeline.build_context(&items);
        assert!(context.contains("[1]"));
        assert!(context.contains("Item content here"));
        // Should NOT contain source info when metadata is disabled
        assert!(!context.contains("source:"));
    }

    #[test]
    fn test_build_context_respects_max_length() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            max_context_length: 100,
            include_metadata: false,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let items = vec![
            ContextItem {
                content: "A".repeat(50),
                source_id: "s1".to_string(),
                chunk_index: 0,
                score: 0.9,
                metadata: HashMap::new(),
            },
            ContextItem {
                content: "B".repeat(50),
                source_id: "s2".to_string(),
                chunk_index: 0,
                score: 0.8,
                metadata: HashMap::new(),
            },
            ContextItem {
                content: "C".repeat(50),
                source_id: "s3".to_string(),
                chunk_index: 0,
                score: 0.7,
                metadata: HashMap::new(),
            },
        ];

        let context = pipeline.build_context(&items);
        // Context should be truncated to respect max_context_length
        assert!(context.len() <= 150); // Some overhead for formatting
    }

    #[test]
    fn test_build_context_empty_items() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        let context = pipeline.build_context(&[]);
        assert!(context.is_empty());
    }

    // ==========================================================================
    // Augment Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_augment_basic() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "Rust is a systems programming language.");
        pipeline.ingest(doc).await.unwrap();

        let prompt = pipeline.augment("What is Rust?", None).await.unwrap();

        assert!(prompt.contains("Relevant Context:"));
        assert!(prompt.contains("Question: What is Rust?"));
    }

    #[tokio::test]
    async fn test_augment_with_custom_system_prompt() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "Python is a high-level language.");
        pipeline.ingest(doc).await.unwrap();

        let custom_prompt = "You are a programming expert.";
        let prompt = pipeline.augment("Tell me about Python", Some(custom_prompt)).await.unwrap();

        assert!(prompt.contains("You are a programming expert"));
        assert!(prompt.contains("Question: Tell me about Python"));
    }

    #[tokio::test]
    async fn test_augment_messages() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "Machine learning uses data to train models.");
        pipeline.ingest(doc).await.unwrap();

        let messages = pipeline
            .augment_messages("What is machine learning?", None)
            .await
            .unwrap();

        assert_eq!(messages.len(), 3); // System, context, user
    }

    #[tokio::test]
    async fn test_augment_messages_with_custom_prompt() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        let doc = Document::new("doc1", "AI is transforming industries.");
        pipeline.ingest(doc).await.unwrap();

        let custom = "You are an AI expert.";
        let messages = pipeline
            .augment_messages("Explain AI", Some(custom))
            .await
            .unwrap();

        assert_eq!(messages.len(), 3);
    }

    // ==========================================================================
    // Set Cross-Encoder Tests
    // ==========================================================================

    #[test]
    fn test_set_cross_encoder() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let mut pipeline = RagPipeline::new(embedder, store, config);

        assert!(pipeline.cross_encoder().is_none());

        let cross_encoder = Arc::new(MockCrossEncoder::new());
        pipeline.set_cross_encoder(cross_encoder);

        assert!(pipeline.cross_encoder().is_some());
    }

    // ==========================================================================
    // Cosine Similarity Tests
    // ==========================================================================

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_both_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_parallel_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // Same direction, different magnitude
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    // ==========================================================================
    // Hybrid Search Tests (BM25 + Dense)
    // ==========================================================================

    #[test]
    fn test_retrieval_config_hybrid_search_default() {
        let config = RetrievalConfig::default();
        assert!(!config.hybrid_search, "Hybrid search should be disabled by default");
        assert!((config.bm25_weight - 0.3).abs() < 0.001, "Default BM25 weight should be 0.3");
        assert!((config.dense_weight - 0.7).abs() < 0.001, "Default dense weight should be 0.7");
    }

    #[test]
    fn test_retrieval_config_hybrid_search_enabled() {
        let config = RetrievalConfig {
            hybrid_search: true,
            bm25_weight: 0.5,
            dense_weight: 0.5,
            ..Default::default()
        };
        assert!(config.hybrid_search);
        assert!((config.bm25_weight - 0.5).abs() < 0.001);
        assert!((config.dense_weight - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_rag_pipeline_hybrid_retriever() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            hybrid_search: true,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Pipeline should have hybrid retriever when hybrid_search is enabled
        assert!(pipeline.hybrid_retriever().is_some());
    }

    #[test]
    fn test_rag_pipeline_no_hybrid_retriever_by_default() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        // Pipeline should NOT have hybrid retriever by default
        assert!(pipeline.hybrid_retriever().is_none());
    }

    #[tokio::test]
    async fn test_hybrid_search_indexes_documents() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            hybrid_search: true,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Ingest a document
        let doc = Document::new("doc1", "The quick brown fox jumps over the lazy dog");
        pipeline.ingest(doc).await.unwrap();

        // BM25 index should have the document
        let retriever = pipeline.hybrid_retriever().unwrap();
        assert_eq!(retriever.read().document_count(), 1);
    }

    #[tokio::test]
    async fn test_hybrid_search_retrieval() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            top_k: 3,
            min_score: 0.0,
            hybrid_search: true,
            bm25_weight: 0.5,
            dense_weight: 0.5,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Ingest documents with different content
        let docs = vec![
            Document::new("doc1", "The quick brown fox jumps over the lazy dog"),
            Document::new("doc2", "A lazy cat sleeps on the couch all day"),
            Document::new("doc3", "Machine learning models process data efficiently"),
        ];
        for doc in docs {
            pipeline.ingest(doc).await.unwrap();
        }

        // Query that matches "lazy" - should boost docs with that keyword
        let results = pipeline.retrieve("lazy animals").await.unwrap();

        // Results should include documents with "lazy" keyword
        assert!(!results.is_empty());
        // First result should be either doc1 or doc2 (both contain "lazy")
        let first_source = &results[0].source_id;
        assert!(
            first_source.starts_with("doc1") || first_source.starts_with("doc2"),
            "First result should be a doc containing 'lazy', got: {}",
            first_source
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_with_reranking() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            top_k: 2,
            min_score: 0.0,
            hybrid_search: true,
            rerank: true,
            rerank_top_k: 10,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Ingest documents
        let docs = vec![
            Document::new("doc1", "Python programming language for data science"),
            Document::new("doc2", "JavaScript frameworks for web development"),
            Document::new("doc3", "Rust systems programming with memory safety"),
        ];
        for doc in docs {
            pipeline.ingest(doc).await.unwrap();
        }

        // Both hybrid search and reranking should work together
        let results = pipeline.retrieve("programming language").await.unwrap();
        assert!(results.len() <= 2);
    }

    // ==========================================================================
    // Incremental Document Update Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_delete_document() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        // Ingest a document
        let doc = Document::new("doc1", "The quick brown fox jumps over the lazy dog");
        pipeline.ingest(doc).await.unwrap();

        // Verify it's there
        let count_before = store.count().await.unwrap();
        assert!(count_before > 0);

        // Delete the document
        let deleted = pipeline.delete_document("doc1").await.unwrap();
        assert!(deleted > 0);

        // Verify it's gone
        let count_after = store.count().await.unwrap();
        assert_eq!(count_after, 0);
    }

    #[tokio::test]
    async fn test_delete_document_not_found() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();
        let pipeline = RagPipeline::new(embedder, store, config);

        // Try to delete non-existent document
        let deleted = pipeline.delete_document("nonexistent").await.unwrap();
        assert_eq!(deleted, 0);
    }

    #[tokio::test]
    async fn test_update_document_content() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: -1.0, // Accept all scores for testing
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        // Ingest original document
        let doc = Document::new("doc1", "Original content about rust programming.");
        pipeline.ingest(doc).await.unwrap();

        // Verify document is in store
        let count_before = store.count().await.unwrap();
        assert_eq!(count_before, 1);

        // Update document with new content
        let updated_doc = Document::new("doc1", "Updated content about python scripting.");
        pipeline.update_document(updated_doc).await.unwrap();

        // Should still have 1 document (replaced)
        let count_after = store.count().await.unwrap();
        assert_eq!(count_after, 1);

        // Get all results and verify content changed
        let results = pipeline.retrieve("any query").await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Updated"), "Should contain updated content");
        assert!(!results[0].content.contains("Original"), "Should not contain original content");
    }

    #[tokio::test]
    async fn test_update_document_shorter() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens { size: 10, overlap: 0 });
        let config = RetrievalConfig {
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::with_chunker(embedder, store.clone(), chunker, config);

        // Ingest a long document (will create multiple chunks)
        let long_content = "This is chunk one. This is chunk two. This is chunk three. This is chunk four.";
        let doc = Document::new("doc1", long_content);
        let original_chunks = pipeline.ingest(doc).await.unwrap();
        assert!(original_chunks > 1, "Should create multiple chunks");

        let count_before = store.count().await.unwrap();

        // Update with shorter content (fewer chunks)
        let short_content = "Short updated content.";
        let updated_doc = Document::new("doc1", short_content);
        pipeline.update_document(updated_doc).await.unwrap();

        let count_after = store.count().await.unwrap();

        // Should have fewer chunks now
        assert!(count_after < count_before, "Shorter document should have fewer chunks");
    }

    #[tokio::test]
    async fn test_update_document_preserves_others() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            min_score: -1.0, // Accept all scores for testing
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store.clone(), config);

        // Ingest two documents
        let doc1 = Document::new("doc1", "First document about cats.");
        let doc2 = Document::new("doc2", "Second document about dogs.");
        pipeline.ingest(doc1).await.unwrap();
        pipeline.ingest(doc2).await.unwrap();

        let count_before = store.count().await.unwrap();
        assert_eq!(count_before, 2);

        // Update only doc1
        let updated_doc1 = Document::new("doc1", "Updated first document about birds.");
        pipeline.update_document(updated_doc1).await.unwrap();

        // Should still have 2 documents
        let count_after = store.count().await.unwrap();
        assert_eq!(count_after, 2);

        // Get all results and verify both documents exist
        let results = pipeline.retrieve("any").await.unwrap();
        assert_eq!(results.len(), 2);

        // Verify doc1 was updated (contains "birds", not "cats")
        let doc1_result = results.iter().find(|r| r.source_id == "doc1").unwrap();
        assert!(doc1_result.content.contains("birds"));
        assert!(!doc1_result.content.contains("cats"));

        // Verify doc2 is unchanged
        let doc2_result = results.iter().find(|r| r.source_id == "doc2").unwrap();
        assert!(doc2_result.content.contains("dogs"));
    }

    #[tokio::test]
    async fn test_update_removes_from_bm25_index() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let store = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig {
            hybrid_search: true,
            min_score: 0.0,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(embedder, store, config);

        // Ingest document with unique keyword
        let doc = Document::new("doc1", "UniqueKeyword123 in original content");
        pipeline.ingest(doc).await.unwrap();

        // BM25 should have the document
        let retriever = pipeline.hybrid_retriever().unwrap();
        assert_eq!(retriever.read().document_count(), 1);

        // Update document without the unique keyword
        let updated_doc = Document::new("doc1", "Completely different content now");
        pipeline.update_document(updated_doc).await.unwrap();

        // BM25 should still have 1 document (the updated one)
        assert_eq!(retriever.read().document_count(), 1);

        // Old keyword should not match
        let results = pipeline.retrieve("UniqueKeyword123").await.unwrap();
        let has_old = results.iter().any(|r| r.content.contains("UniqueKeyword123"));
        assert!(!has_old, "Old content should be removed from BM25 index");
    }
}
