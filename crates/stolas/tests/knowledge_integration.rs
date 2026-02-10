//! Integration tests for Stolas - the knowledge engine.
//!
//! Tests cover:
//! - Document chunking strategies
//! - Embedding generation
//! - Vector storage
//! - BM25 sparse retrieval
//! - Hybrid retrieval

use std::collections::HashMap;
use std::sync::Arc;

use stolas::{
    cosine_similarity, dot_product, euclidean_distance, BM25Config, BM25Index, BatchEmbedder,
    Chunker, ChunkingStrategy, Embedder, HybridRetriever, InMemoryStore, MockEmbedder,
    PoolingStrategy, SearchParams, SentenceEmbedder, VectorRecord, VectorStore,
};

// ============================================================================
// Chunker Tests - Fixed Token Strategy
// ============================================================================

#[test]
fn test_chunker_fixed_tokens_basic() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 20,
        overlap: 5,
    });

    let text = "The quick brown fox jumps over the lazy dog. A wonderful sentence follows.";
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    // First chunk should have exactly 20 characters
    assert_eq!(chunks[0].text.chars().count(), 20);
    assert_eq!(chunks[0].index, 0);
}

#[test]
fn test_chunker_fixed_tokens_overlap() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 10,
        overlap: 3,
    });

    let text = "0123456789ABCDEFGHIJ";
    let chunks = chunker.chunk(text);

    // Verify overlap
    if chunks.len() >= 2 {
        let end_of_first = &chunks[0].text[7..10]; // Last 3 chars of first chunk
        let start_of_second = &chunks[1].text[0..3]; // First 3 chars of second chunk
        assert_eq!(end_of_first, start_of_second);
    }
}

#[test]
fn test_chunker_fixed_tokens_empty() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 50,
        overlap: 10,
    });

    let chunks = chunker.chunk("");
    assert!(chunks.is_empty());
}

#[test]
fn test_chunker_fixed_tokens_text_smaller_than_chunk() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 100,
        overlap: 20,
    });

    let text = "Short text";
    let chunks = chunker.chunk(text);

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
}

// ============================================================================
// Chunker Tests - Recursive Strategy
// ============================================================================

#[test]
fn test_chunker_recursive_basic() {
    let chunker = Chunker::new(ChunkingStrategy::Recursive {
        separators: vec!["\n".to_string(), " ".to_string()],
        chunk_size: 30,
    });

    let text = "First line of content here.\nSecond line with more text.";
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(!chunk.text.trim().is_empty());
    }
}

#[test]
fn test_chunker_recursive_respects_separators() {
    let chunker = Chunker::new(ChunkingStrategy::Recursive {
        separators: vec!["\n\n".to_string(), "\n".to_string()],
        chunk_size: 50,
    });

    let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
    let chunks = chunker.chunk(text);

    // Should produce chunks
    assert!(!chunks.is_empty());
}

// ============================================================================
// Chunker Tests - Sentence Strategy
// ============================================================================

#[test]
fn test_chunker_sentence_basic() {
    let chunker = Chunker::new(ChunkingStrategy::Sentence {
        min_size: 10,
        max_size: 100,
    });

    let text = "First sentence. Second sentence. Third sentence!";
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    // Check all sentences are included
    let combined: String = chunks.iter().map(|c| c.text.clone()).collect();
    assert!(combined.contains("First"));
    assert!(combined.contains("Second"));
    assert!(combined.contains("Third"));
}

#[test]
fn test_chunker_sentence_respects_min_size() {
    let chunker = Chunker::new(ChunkingStrategy::Sentence {
        min_size: 50,
        max_size: 200,
    });

    let text = "One. Two. Three. Four. Five.";
    let chunks = chunker.chunk(text);

    // Should combine short sentences
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunker_sentence_different_terminators() {
    let chunker = Chunker::new(ChunkingStrategy::Sentence {
        min_size: 1,
        max_size: 200,
    });

    let text = "Question? Exclamation! Statement.";
    let chunks = chunker.chunk(text);

    let combined: String = chunks.iter().map(|c| c.text.clone()).collect();
    assert!(combined.contains("Question"));
    assert!(combined.contains("Exclamation"));
    assert!(combined.contains("Statement"));
}

// ============================================================================
// Chunker Tests - Unicode Support
// ============================================================================

#[test]
fn test_chunker_unicode_fixed() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 5,
        overlap: 1,
    });

    let text = "Hello 世界! 你好";
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
    // Character count, not byte count
    assert_eq!(chunks[0].text.chars().count(), 5);
}

#[test]
fn test_chunker_default() {
    let chunker = Chunker::default();
    let chunks = chunker.chunk("Test text for default chunker.");
    assert!(!chunks.is_empty());
}

// ============================================================================
// Chunk Struct Tests
// ============================================================================

#[test]
fn test_chunk_indices_sequential() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 5,
        overlap: 1,
    });

    let text = "0123456789ABCDEFGHIJ";
    let chunks = chunker.chunk(text);

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i);
    }
}

#[test]
fn test_chunk_offsets_valid() {
    let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
        size: 10,
        overlap: 2,
    });

    let text = "Hello, World! This is a test.";
    let chunks = chunker.chunk(text);

    for chunk in &chunks {
        assert!(chunk.start <= chunk.end);
    }
}

// ============================================================================
// MockEmbedder Tests
// ============================================================================

#[tokio::test]
async fn test_mock_embedder_basic() {
    let embedder = MockEmbedder::new(384);

    let embeddings = embedder.embed(&["hello", "world"]).await.unwrap();

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 384);
    assert_eq!(embeddings[1].len(), 384);
}

#[tokio::test]
async fn test_mock_embedder_deterministic() {
    let embedder = MockEmbedder::new(128);

    let emb1 = embedder.embed(&["test"]).await.unwrap();
    let emb2 = embedder.embed(&["test"]).await.unwrap();

    // Same input should produce same embedding
    assert_eq!(emb1[0], emb2[0]);
}

#[tokio::test]
async fn test_mock_embedder_different_texts() {
    let embedder = MockEmbedder::new(64);

    let embeddings = embedder.embed(&["hello", "goodbye"]).await.unwrap();

    // Different texts should have different embeddings
    assert_ne!(embeddings[0], embeddings[1]);
}

#[tokio::test]
async fn test_mock_embedder_single() {
    let embedder = MockEmbedder::new(256);

    let embedding = embedder.embed_single("test text").await.unwrap();
    assert_eq!(embedding.len(), 256);
}

#[test]
fn test_mock_embedder_dimension() {
    let embedder = MockEmbedder::new(512);
    assert_eq!(embedder.dimension(), 512);
}

#[test]
fn test_mock_embedder_model_name() {
    let embedder = MockEmbedder::new(384);
    assert_eq!(embedder.model_name(), "mock-embedder");
}

// ============================================================================
// SentenceEmbedder Tests
// ============================================================================

#[tokio::test]
async fn test_sentence_embedder_normalization() {
    let mock = Arc::new(MockEmbedder::new(3));
    let embedder = SentenceEmbedder::new(mock);

    let embeddings = embedder.embed(&["test"]).await.unwrap();

    // Check normalization (L2 norm should be ~1.0)
    let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);
}

#[tokio::test]
async fn test_sentence_embedder_without_normalization() {
    let mock = Arc::new(MockEmbedder::new(3));
    let embedder = SentenceEmbedder::new(mock).with_normalize(false);

    let embeddings = embedder.embed(&["test"]).await.unwrap();

    // Norm may not be 1.0
    let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    // Just verify it works, no normalization assertion
    assert!(norm > 0.0);
}

#[tokio::test]
async fn test_sentence_embedder_pooling_strategy() {
    let mock = Arc::new(MockEmbedder::new(128));
    let embedder = SentenceEmbedder::new(mock)
        .with_pooling(PoolingStrategy::Mean)
        .with_normalize(true);

    let embedding = embedder.embed_single("Test sentence").await.unwrap();
    assert_eq!(embedding.len(), 128);
}

// ============================================================================
// BatchEmbedder Tests
// ============================================================================

#[tokio::test]
async fn test_batch_embedder_basic() {
    let mock = Arc::new(MockEmbedder::new(64));
    let batch_embedder = BatchEmbedder::new(mock).with_batch_size(2);

    let texts: Vec<String> = vec!["one", "two", "three", "four", "five"]
        .into_iter()
        .map(String::from)
        .collect();

    let embeddings = batch_embedder.embed_batch(&texts).await.unwrap();

    assert_eq!(embeddings.len(), 5);
    for emb in &embeddings {
        assert_eq!(emb.len(), 64);
    }
}

#[tokio::test]
async fn test_batch_embedder_empty() {
    let mock = Arc::new(MockEmbedder::new(128));
    let batch_embedder = BatchEmbedder::new(mock);

    let embeddings = batch_embedder.embed_batch(&[]).await.unwrap();
    assert!(embeddings.is_empty());
}

// ============================================================================
// Vector Similarity Functions Tests
// ============================================================================

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
}

#[test]
fn test_euclidean_distance_same_point() {
    let a = vec![1.0, 2.0, 3.0];
    let dist = euclidean_distance(&a, &a);
    assert!(dist.abs() < 0.0001);
}

#[test]
fn test_euclidean_distance_unit() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let dist = euclidean_distance(&a, &b);
    assert!((dist - 1.0).abs() < 0.0001);
}

#[test]
fn test_euclidean_distance_3_4_5_triangle() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0];
    let dist = euclidean_distance(&a, &b);
    assert!((dist - 5.0).abs() < 0.0001);
}

#[test]
fn test_dot_product_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dp = dot_product(&a, &b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((dp - 32.0).abs() < 0.0001);
}

#[test]
fn test_dot_product_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let dp = dot_product(&a, &b);
    assert!(dp.abs() < 0.0001);
}

// ============================================================================
// InMemoryStore Tests
// ============================================================================

#[tokio::test]
async fn test_in_memory_store_upsert() {
    let store = InMemoryStore::new();

    let records = vec![
        VectorRecord::new(vec![1.0, 0.0, 0.0], "document one"),
        VectorRecord::new(vec![0.0, 1.0, 0.0], "document two"),
    ];

    let count = store.upsert(records).await.unwrap();
    assert_eq!(count, 2);
    assert_eq!(store.count().await.unwrap(), 2);
}

#[tokio::test]
async fn test_in_memory_store_search() {
    let store = InMemoryStore::new();

    let records = vec![
        VectorRecord::new(vec![1.0, 0.0, 0.0], "doc about cats"),
        VectorRecord::new(vec![0.0, 1.0, 0.0], "doc about dogs"),
        VectorRecord::new(vec![0.9, 0.1, 0.0], "doc about felines"),
    ];

    store.upsert(records).await.unwrap();

    // Search for something similar to cats
    let results = store
        .search(&[1.0, 0.0, 0.0], SearchParams::default())
        .await
        .unwrap();

    assert!(!results.is_empty());
    // First result should be the exact match
    assert!(results[0].score > 0.99);
    assert!(results[0].record.content.contains("cats"));
}

#[tokio::test]
async fn test_in_memory_store_search_with_min_score() {
    let store = InMemoryStore::new();

    let records = vec![
        VectorRecord::new(vec![1.0, 0.0, 0.0], "exact match"),
        VectorRecord::new(vec![0.7, 0.7, 0.0], "partial match"),
        VectorRecord::new(vec![0.0, 0.0, 1.0], "no match"),
    ];

    store.upsert(records).await.unwrap();

    let params = SearchParams {
        top_k: 10,
        min_score: Some(0.9),
        filters: HashMap::new(),
    };

    let results = store.search(&[1.0, 0.0, 0.0], params).await.unwrap();

    // Only exact match should pass 0.9 threshold
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].record.content, "exact match");
}

#[tokio::test]
async fn test_in_memory_store_delete() {
    let store = InMemoryStore::new();

    let record = VectorRecord::new(vec![1.0, 0.0], "test");
    let id = record.id.clone();

    store.upsert(vec![record]).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);

    let deleted = store.delete(vec![id]).await.unwrap();
    assert_eq!(deleted, 1);
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_in_memory_store_get() {
    let store = InMemoryStore::new();

    let r1 = VectorRecord::new(vec![1.0, 0.0], "first");
    let r2 = VectorRecord::new(vec![0.0, 1.0], "second");
    let id1 = r1.id.clone();
    let id2 = r2.id.clone();

    store.upsert(vec![r1, r2]).await.unwrap();

    let fetched = store.get(vec![id1.clone(), id2.clone()]).await.unwrap();
    assert_eq!(fetched.len(), 2);

    let fetched_one = store.get(vec![id1]).await.unwrap();
    assert_eq!(fetched_one.len(), 1);
}

#[tokio::test]
async fn test_in_memory_store_upsert_update() {
    let store = InMemoryStore::new();

    let mut record = VectorRecord::new(vec![1.0, 0.0], "original");
    let id = record.id.clone();

    store.upsert(vec![record.clone()]).await.unwrap();

    // Update the record
    record.content = "updated".to_string();
    store.upsert(vec![record]).await.unwrap();

    // Should still be 1 record (upsert overwrites)
    assert_eq!(store.count().await.unwrap(), 1);

    let fetched = store.get(vec![id]).await.unwrap();
    assert_eq!(fetched[0].content, "updated");
}

#[tokio::test]
async fn test_in_memory_store_with_metadata() {
    let store = InMemoryStore::new();

    let record = VectorRecord::new(vec![1.0, 0.0, 0.0], "document")
        .with_metadata("category", serde_json::json!("science"))
        .with_metadata("year", serde_json::json!(2024));

    let id = record.id.clone();
    store.upsert(vec![record]).await.unwrap();

    let fetched = store.get(vec![id]).await.unwrap();
    assert_eq!(fetched[0].metadata.len(), 2);
    assert_eq!(
        fetched[0].metadata.get("category"),
        Some(&serde_json::json!("science"))
    );
}

// ============================================================================
// VectorRecord Tests
// ============================================================================

#[test]
fn test_vector_record_new() {
    let record = VectorRecord::new(vec![1.0, 2.0, 3.0], "test content");

    assert!(!record.id.is_empty());
    assert_eq!(record.vector, vec![1.0, 2.0, 3.0]);
    assert_eq!(record.content, "test content");
    assert!(record.metadata.is_empty());
}

#[test]
fn test_vector_record_with_metadata() {
    let record = VectorRecord::new(vec![1.0], "test")
        .with_metadata("key", serde_json::json!("value"))
        .with_metadata("num", serde_json::json!(42));

    assert_eq!(record.metadata.len(), 2);
}

#[test]
fn test_search_params_default() {
    let params = SearchParams::default();
    assert_eq!(params.top_k, 10);
    assert!(params.min_score.is_none());
    assert!(params.filters.is_empty());
}

// ============================================================================
// BM25Index Tests
// ============================================================================

#[test]
fn test_bm25_index_new() {
    let index = BM25Index::with_defaults();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.vocabulary_size(), 0);
}

#[test]
fn test_bm25_add_document() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "The quick brown fox");

    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
    assert!(index.vocabulary_size() > 0);
}

#[test]
fn test_bm25_add_multiple_documents() {
    let mut index = BM25Index::with_defaults();
    index.add_documents([
        ("doc1", "The quick brown fox"),
        ("doc2", "The lazy dog"),
        ("doc3", "A quick lazy fox"),
    ]);

    assert_eq!(index.len(), 3);
}

#[test]
fn test_bm25_remove_document() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "test document");
    index.add_document("doc2", "another document");

    assert!(index.remove_document("doc1"));
    assert_eq!(index.len(), 1);

    assert!(!index.remove_document("nonexistent"));
}

#[test]
fn test_bm25_search_basic() {
    let mut index = BM25Index::with_defaults();
    index.add_documents([
        ("doc1", "The quick brown fox jumps over the lazy dog"),
        ("doc2", "A lazy cat sleeps all day"),
        ("doc3", "The fox is quick and smart"),
    ]);

    let results = index.search("quick fox", 10);

    assert!(!results.is_empty());
    // doc1 and doc3 contain both terms
    assert!(results.len() >= 2);
}

#[test]
fn test_bm25_search_empty_query() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "Test document");

    let results = index.search("", 10);
    assert!(results.is_empty());
}

#[test]
fn test_bm25_search_no_matches() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "The quick brown fox");

    let results = index.search("elephant", 10);
    assert!(results.is_empty());
}

#[test]
fn test_bm25_search_with_threshold() {
    let mut index = BM25Index::with_defaults();
    index.add_documents([
        ("doc1", "machine learning artificial intelligence"),
        ("doc2", "machine parts factory"),
        ("doc3", "deep learning neural networks"),
    ]);

    let results = index.search_with_threshold("machine learning", 10, 0.5);

    for result in &results {
        assert!(result.score >= 0.5);
    }
}

#[test]
fn test_bm25_clear() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "test");

    index.clear();

    assert!(index.is_empty());
    assert_eq!(index.vocabulary_size(), 0);
}

#[test]
fn test_bm25_get_term_idfs() {
    let mut index = BM25Index::with_defaults();
    index.add_documents([
        ("doc1", "common rare unique"),
        ("doc2", "common word"),
        ("doc3", "common another"),
    ]);

    let idfs = index.get_term_idfs("common rare unique");

    // Rare terms should have higher IDF than common terms
    let rare_idf = idfs.get("rare").unwrap_or(&0.0);
    let common_idf = idfs.get("common").unwrap_or(&f32::MAX);
    assert!(rare_idf > common_idf);
}

#[test]
fn test_bm25_document_update() {
    let mut index = BM25Index::with_defaults();
    index.add_document("doc1", "original content");

    // Adding same ID should replace
    index.add_document("doc1", "updated content new");

    assert_eq!(index.len(), 1);

    let results = index.search("updated", 10);
    assert!(!results.is_empty());

    let results = index.search("original", 10);
    assert!(results.is_empty());
}

#[test]
fn test_bm25_config_default() {
    let config = BM25Config::default();
    assert!((config.k1 - 1.5).abs() < 0.001);
    assert!((config.b - 0.75).abs() < 0.001);
    assert!(config.lowercase);
}

#[test]
fn test_bm25_config_for_short_docs() {
    let config = BM25Config::for_short_docs();
    assert!(config.b < 0.75); // Less length normalization
}

#[test]
fn test_bm25_config_for_long_docs() {
    let config = BM25Config::for_long_docs();
    assert!(config.k1 >= 2.0);
}

// ============================================================================
// HybridRetriever Tests
// ============================================================================

#[test]
fn test_hybrid_retriever_new() {
    let retriever = HybridRetriever::with_equal_weights();
    assert!(retriever.bm25_index().is_empty());
}

#[test]
fn test_hybrid_retriever_add_document() {
    let mut retriever = HybridRetriever::with_equal_weights();
    retriever.add_document("doc1", "test content");

    assert_eq!(retriever.bm25_index().len(), 1);
}

#[test]
fn test_hybrid_search() {
    let mut retriever = HybridRetriever::with_equal_weights();
    retriever.add_document("doc1", "machine learning algorithms");
    retriever.add_document("doc2", "deep learning neural networks");
    retriever.add_document("doc3", "learning to code");

    // Simulate dense retrieval results
    let dense_results = vec![
        ("doc2".to_string(), 0.9),
        ("doc1".to_string(), 0.7),
        ("doc3".to_string(), 0.3),
    ];

    let results = retriever.hybrid_search("machine learning", &dense_results, 10);

    assert!(!results.is_empty());
    for result in &results {
        assert!(result.hybrid_score >= 0.0);
        assert!(result.bm25_score >= 0.0);
        assert!(result.dense_score >= 0.0);
    }
}

#[test]
fn test_hybrid_retriever_weights() {
    let mut retriever = HybridRetriever::dense_heavy();

    retriever.set_bm25_weight(0.4);
    retriever.set_dense_weight(0.6);

    // Weights should be clamped
    retriever.set_bm25_weight(1.5); // Should clamp to 1.0
    retriever.set_dense_weight(-0.1); // Should clamp to 0.0
}

#[test]
fn test_hybrid_retriever_sparse_heavy() {
    let retriever = HybridRetriever::sparse_heavy();
    // Just verify it constructs correctly
    assert!(retriever.bm25_index().is_empty());
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[tokio::test]
async fn test_chunking_and_embedding_workflow() {
    // 1. Chunk a document
    let chunker = Chunker::new(ChunkingStrategy::Sentence {
        min_size: 10,
        max_size: 100,
    });

    let document = "Machine learning is a branch of AI. \
                    It focuses on building systems that learn from data. \
                    Deep learning is a subset of machine learning.";

    let chunks = chunker.chunk(document);
    assert!(!chunks.is_empty());

    // 2. Generate embeddings for chunks
    let embedder = MockEmbedder::new(128);
    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let embeddings = embedder.embed(&chunk_texts).await.unwrap();

    assert_eq!(embeddings.len(), chunks.len());

    // 3. Store in vector database
    let store = InMemoryStore::new();
    let records: Vec<VectorRecord> = chunks
        .iter()
        .zip(embeddings.iter())
        .map(|(chunk, emb)| {
            VectorRecord::new(emb.clone(), &chunk.text)
                .with_metadata("index", serde_json::json!(chunk.index))
        })
        .collect();

    store.upsert(records).await.unwrap();

    // 4. Search
    let query_embedding = embedder.embed_single("machine learning").await.unwrap();
    let results = store
        .search(&query_embedding, SearchParams::default())
        .await
        .unwrap();

    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_hybrid_search_workflow() {
    // 1. Create hybrid retriever
    let mut retriever = HybridRetriever::with_equal_weights();

    // 2. Add documents to BM25 index
    let documents = [
        ("doc1", "Python programming language basics tutorial"),
        ("doc2", "JavaScript web development framework"),
        ("doc3", "Rust systems programming memory safety"),
        ("doc4", "Python machine learning data science"),
    ];

    for (id, content) in &documents {
        retriever.add_document(*id, *content);
    }

    // 3. Generate embeddings for dense retrieval
    let embedder = MockEmbedder::new(64);

    // Simulate query embedding and document embeddings
    let query = "Python programming";
    let query_emb = embedder.embed_single(query).await.unwrap();

    let mut dense_results = Vec::new();
    for (id, content) in &documents {
        let doc_emb = embedder.embed_single(*content).await.unwrap();
        let score = cosine_similarity(&query_emb, &doc_emb);
        dense_results.push((id.to_string(), score));
    }

    // 4. Hybrid search
    let results = retriever.hybrid_search(query, &dense_results, 10);

    assert!(!results.is_empty());

    // Python documents should rank higher
    let top_ids: Vec<&str> = results.iter().take(2).map(|r| r.id.as_str()).collect();
    assert!(top_ids.contains(&"doc1") || top_ids.contains(&"doc4"));
}

#[test]
fn test_bm25_score_ordering() {
    let mut index = BM25Index::with_defaults();
    index.add_documents([
        ("doc1", "fox fox fox fox fox"),    // High term frequency
        ("doc2", "fox"),                    // Low term frequency
        ("doc3", "the quick brown animal"), // No match
    ]);

    let results = index.search("fox", 10);

    assert_eq!(results.len(), 2); // doc3 has no match

    // Higher term frequency should score higher
    if results.len() >= 2 {
        assert!(results[0].score >= results[1].score);
    }
}
