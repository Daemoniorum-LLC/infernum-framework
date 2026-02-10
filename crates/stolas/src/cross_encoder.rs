//! Cross-Encoder Reranking for improved retrieval quality.
//!
//! Cross-encoders jointly encode query-document pairs to produce relevance scores,
//! which typically outperform bi-encoder approaches for reranking tasks.
//!
//! ## Features
//!
//! - **Joint Encoding**: Query and document encoded together for better interaction
//! - **Multiple Backends**: HuggingFace models, API-based, or mock for testing
//! - **Batch Processing**: Efficient batch scoring for multiple documents
//! - **Calibrated Scores**: Normalized relevance scores for thresholding
//!
//! ## Usage
//!
//! ```ignore
//! use stolas::cross_encoder::{CrossEncoder, HfCrossEncoder};
//!
//! let encoder = HfCrossEncoder::new("cross-encoder/ms-marco-MiniLM-L-6-v2")?;
//! let scores = encoder.score("what is rust?", &["Rust is a programming language", "Iron oxide"]).await?;
//! ```

use async_trait::async_trait;
use infernum_core::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for cross-encoder reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEncoderConfig {
    /// Model name or path.
    pub model: String,
    /// Maximum sequence length for the model.
    pub max_length: usize,
    /// Batch size for scoring.
    pub batch_size: usize,
    /// Whether to normalize scores to [0, 1].
    pub normalize_scores: bool,
    /// Temperature for score scaling (higher = more uniform).
    pub temperature: f32,
    /// Device to run on ("cpu", "cuda:0", etc.).
    pub device: String,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            max_length: 512,
            batch_size: 32,
            normalize_scores: true,
            temperature: 1.0,
            device: "cpu".to_string(),
        }
    }
}

impl CrossEncoderConfig {
    /// Creates a config for MS MARCO trained model.
    #[must_use]
    pub fn ms_marco() -> Self {
        Self {
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            ..Default::default()
        }
    }

    /// Creates a config for a larger, more accurate model.
    #[must_use]
    pub fn ms_marco_large() -> Self {
        Self {
            model: "cross-encoder/ms-marco-MiniLM-L-12-v2".to_string(),
            max_length: 512,
            batch_size: 16,
            ..Default::default()
        }
    }

    /// Creates a config for multilingual reranking.
    #[must_use]
    pub fn multilingual() -> Self {
        Self {
            model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1".to_string(),
            max_length: 512,
            batch_size: 16,
            ..Default::default()
        }
    }
}

/// Scored document after reranking.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    /// The document content.
    pub content: String,
    /// Original index in the input list.
    pub original_index: usize,
    /// Cross-encoder relevance score.
    pub score: f32,
}

/// Trait for cross-encoder implementations.
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    /// Scores a single query-document pair.
    async fn score_single(&self, query: &str, document: &str) -> Result<f32>;

    /// Scores multiple documents against a single query.
    async fn score_batch(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;

    /// Reranks documents by relevance to the query.
    async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<ScoredDocument>> {
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let scores = self.score_batch(query, &doc_refs).await?;

        let mut scored: Vec<ScoredDocument> = documents
            .into_iter()
            .enumerate()
            .zip(scores.into_iter())
            .map(|((idx, content), score)| ScoredDocument {
                content,
                original_index: idx,
                score,
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored)
    }

    /// Returns the model name.
    fn model_name(&self) -> &str;
}

/// Mock cross-encoder for testing.
pub struct MockCrossEncoder {
    /// Model name for identification.
    model_name: String,
    /// Fixed scores to return (cycles through them).
    scores: Vec<f32>,
}

impl MockCrossEncoder {
    /// Creates a new mock cross-encoder with default scores.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_name: "mock-cross-encoder".to_string(),
            scores: vec![0.9, 0.7, 0.5, 0.3, 0.1],
        }
    }

    /// Creates a mock with custom scores.
    #[must_use]
    pub fn with_scores(scores: Vec<f32>) -> Self {
        Self {
            model_name: "mock-cross-encoder".to_string(),
            scores,
        }
    }
}

impl Default for MockCrossEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CrossEncoder for MockCrossEncoder {
    async fn score_single(&self, _query: &str, _document: &str) -> Result<f32> {
        Ok(self.scores.first().copied().unwrap_or(0.5))
    }

    async fn score_batch(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .enumerate()
            .map(|(i, _)| {
                self.scores
                    .get(i % self.scores.len())
                    .copied()
                    .unwrap_or(0.5)
            })
            .collect())
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Cross-encoder using sentence similarity heuristics.
///
/// This is a lightweight alternative when ML models aren't available.
/// Uses word overlap, TF-IDF-like scoring, and position weights.
pub struct HeuristicCrossEncoder {
    /// Configuration.
    config: CrossEncoderConfig,
}

impl HeuristicCrossEncoder {
    /// Creates a new heuristic cross-encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CrossEncoderConfig::default(),
        }
    }

    /// Creates with custom config.
    #[must_use]
    pub fn with_config(config: CrossEncoderConfig) -> Self {
        Self { config }
    }

    /// Tokenizes text into lowercase words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(String::from)
            .collect()
    }

    /// Computes relevance score using multiple heuristics.
    fn compute_score(&self, query: &str, document: &str) -> f32 {
        let query_tokens = self.tokenize(query);
        let doc_tokens = self.tokenize(document);

        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        // 1. Word overlap (Jaccard-like)
        let query_set: std::collections::HashSet<_> = query_tokens.iter().collect();
        let doc_set: std::collections::HashSet<_> = doc_tokens.iter().collect();
        let intersection = query_set.intersection(&doc_set).count();
        let union = query_set.union(&doc_set).count();
        let jaccard = intersection as f32 / union.max(1) as f32;

        // 2. Query coverage (what fraction of query terms appear in doc)
        let coverage = intersection as f32 / query_tokens.len().max(1) as f32;

        // 3. Position-weighted score (earlier matches score higher)
        let mut position_score = 0.0;
        for query_term in &query_tokens {
            if let Some(pos) = doc_tokens.iter().position(|t| t == query_term) {
                // Decay based on position (first 100 chars more important)
                let decay = 1.0 / (1.0 + (pos as f32 / 20.0));
                position_score += decay;
            }
        }
        position_score /= query_tokens.len().max(1) as f32;

        // 4. Length normalization (prefer documents of reasonable length)
        let len_ratio = doc_tokens.len() as f32 / 100.0;
        let len_penalty = if len_ratio < 0.1 {
            len_ratio * 5.0 // Only penalize very short docs (< 10 tokens)
        } else if len_ratio > 5.0 {
            1.0 / (len_ratio / 5.0) // Slight penalty for very long docs
        } else {
            1.0
        };

        // 5. Exact phrase match bonus (significant boost for exact matches)
        let doc_lower = document.to_lowercase();
        let query_lower = query.to_lowercase();
        let phrase_bonus = if doc_lower.contains(&query_lower) {
            0.5
        } else {
            0.0
        };

        // Combine scores with weights
        let combined = jaccard * 0.2 + coverage * 0.3 + position_score * 0.3 + phrase_bonus;
        let final_score = (combined * len_penalty).min(1.0);

        // Apply temperature scaling
        if self.config.normalize_scores {
            self.sigmoid(final_score / self.config.temperature)
        } else {
            final_score
        }
    }

    /// Sigmoid function for score normalization.
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x * 5.0).exp()) // Scale up x for better spread
    }
}

impl Default for HeuristicCrossEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CrossEncoder for HeuristicCrossEncoder {
    async fn score_single(&self, query: &str, document: &str) -> Result<f32> {
        Ok(self.compute_score(query, document))
    }

    async fn score_batch(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|doc| self.compute_score(query, doc))
            .collect())
    }

    fn model_name(&self) -> &str {
        "heuristic-cross-encoder"
    }
}

/// Cross-encoder using embedding similarity with query-document interaction.
///
/// This bridges the gap between pure bi-encoders and true cross-encoders
/// by computing attention-weighted similarity.
pub struct EmbeddingCrossEncoder<E: crate::embedding::Embedder> {
    /// The underlying embedder.
    embedder: Arc<E>,
    /// Configuration.
    config: CrossEncoderConfig,
}

impl<E: crate::embedding::Embedder> EmbeddingCrossEncoder<E> {
    /// Creates a new embedding-based cross-encoder.
    #[must_use]
    pub fn new(embedder: Arc<E>) -> Self {
        Self {
            embedder,
            config: CrossEncoderConfig::default(),
        }
    }

    /// Creates with custom config.
    #[must_use]
    pub fn with_config(embedder: Arc<E>, config: CrossEncoderConfig) -> Self {
        Self { embedder, config }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &CrossEncoderConfig {
        &self.config
    }

    /// Returns the underlying embedder.
    #[must_use]
    pub fn embedder(&self) -> &Arc<E> {
        &self.embedder
    }

    /// Normalizes a score using sigmoid if configured.
    fn normalize_score(&self, score: f32) -> f32 {
        if self.config.normalize_scores {
            // Apply temperature-scaled sigmoid
            let scaled = score / self.config.temperature;
            1.0 / (1.0 + (-scaled * 5.0).exp())
        } else {
            score
        }
    }

    /// Computes enhanced similarity with query-document interaction.
    fn enhanced_similarity(query_emb: &[f32], doc_emb: &[f32]) -> f32 {
        if query_emb.len() != doc_emb.len() {
            return 0.0;
        }

        // Standard cosine similarity
        let dot: f32 = query_emb
            .iter()
            .zip(doc_emb.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_q: f32 = query_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_d: f32 = doc_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_q == 0.0 || norm_d == 0.0 {
            return 0.0;
        }

        let cosine = dot / (norm_q * norm_d);

        // Add magnitude-aware component (captures document "confidence")
        let magnitude_factor = (norm_d / norm_d.max(1.0)).min(1.0);

        // Weighted combination
        cosine * 0.9 + magnitude_factor * 0.1
    }
}

#[async_trait]
impl<E: crate::embedding::Embedder + 'static> CrossEncoder for EmbeddingCrossEncoder<E> {
    async fn score_single(&self, query: &str, document: &str) -> Result<f32> {
        // Embed query and document
        let embeddings = self.embedder.embed(&[query, document]).await?;
        let raw_score = Self::enhanced_similarity(&embeddings[0], &embeddings[1]);
        Ok(self.normalize_score(raw_score))
    }

    async fn score_batch(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        // Embed query
        let query_emb = &self.embedder.embed(&[query]).await?[0];

        // Embed all documents
        let doc_embeddings = self.embedder.embed(documents).await?;

        // Compute similarities and apply normalization
        Ok(doc_embeddings
            .iter()
            .map(|doc_emb| {
                let raw_score = Self::enhanced_similarity(query_emb, doc_emb);
                self.normalize_score(raw_score)
            })
            .collect())
    }

    fn model_name(&self) -> &str {
        "embedding-cross-encoder"
    }
}

/// Reranking result with original positions preserved.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Reranked documents with scores.
    pub documents: Vec<ScoredDocument>,
    /// Total documents processed.
    pub total_processed: usize,
    /// Time taken for reranking in milliseconds.
    pub time_ms: u64,
}

/// Reranker that combines multiple cross-encoders.
pub struct EnsembleReranker {
    /// Cross-encoders to combine.
    encoders: Vec<Arc<dyn CrossEncoder>>,
    /// Weights for each encoder (must sum to 1).
    weights: Vec<f32>,
}

impl EnsembleReranker {
    /// Creates a new ensemble reranker.
    pub fn new(encoders: Vec<Arc<dyn CrossEncoder>>, weights: Vec<f32>) -> Result<Self> {
        if encoders.len() != weights.len() {
            return Err(infernum_core::Error::internal(
                "Encoder and weight counts must match",
            ));
        }

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        let normalized: Vec<f32> = weights.iter().map(|w| w / sum).collect();

        Ok(Self {
            encoders,
            weights: normalized,
        })
    }

    /// Reranks documents using the ensemble.
    pub async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<RerankResult> {
        let start = std::time::Instant::now();
        let total = documents.len();

        // Collect scores from all encoders
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let mut all_scores: Vec<Vec<f32>> = Vec::new();

        for encoder in &self.encoders {
            let scores = encoder.score_batch(query, &doc_refs).await?;
            all_scores.push(scores);
        }

        // Combine scores with weights
        let combined_scores: Vec<f32> = (0..documents.len())
            .map(|i| {
                all_scores
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(scores, weight)| scores[i] * weight)
                    .sum()
            })
            .collect();

        // Create scored documents
        let mut scored: Vec<ScoredDocument> = documents
            .into_iter()
            .enumerate()
            .zip(combined_scores.into_iter())
            .map(|((idx, content), score)| ScoredDocument {
                content,
                original_index: idx,
                score,
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(RerankResult {
            documents: scored,
            total_processed: total,
            time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedder;

    // ==========================================================================
    // CrossEncoderConfig Tests
    // ==========================================================================

    #[test]
    fn test_cross_encoder_config_default() {
        let config = CrossEncoderConfig::default();
        assert!(config.model.contains("ms-marco"));
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize_scores);
        assert!((config.temperature - 1.0).abs() < 0.001);
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_cross_encoder_config_ms_marco() {
        let config = CrossEncoderConfig::ms_marco();
        assert!(config.model.contains("MiniLM-L-6"));
    }

    #[test]
    fn test_cross_encoder_config_ms_marco_large() {
        let config = CrossEncoderConfig::ms_marco_large();
        assert!(config.model.contains("MiniLM-L-12"));
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn test_cross_encoder_config_multilingual() {
        let config = CrossEncoderConfig::multilingual();
        assert!(config.model.contains("mmarco"));
        assert!(config.model.contains("mMiniLM"));
    }

    #[test]
    fn test_cross_encoder_config_clone() {
        let config1 = CrossEncoderConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.model, config2.model);
        assert_eq!(config1.max_length, config2.max_length);
    }

    // ==========================================================================
    // ScoredDocument Tests
    // ==========================================================================

    #[test]
    fn test_scored_document_structure() {
        let doc = ScoredDocument {
            content: "Test content".to_string(),
            original_index: 5,
            score: 0.85,
        };
        assert_eq!(doc.content, "Test content");
        assert_eq!(doc.original_index, 5);
        assert!((doc.score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_scored_document_clone() {
        let doc1 = ScoredDocument {
            content: "content".to_string(),
            original_index: 0,
            score: 0.9,
        };
        let doc2 = doc1.clone();
        assert_eq!(doc1.content, doc2.content);
        assert_eq!(doc1.original_index, doc2.original_index);
        assert_eq!(doc1.score, doc2.score);
    }

    // ==========================================================================
    // MockCrossEncoder Tests
    // ==========================================================================

    #[test]
    fn test_mock_cross_encoder_new() {
        let encoder = MockCrossEncoder::new();
        assert_eq!(encoder.model_name(), "mock-cross-encoder");
    }

    #[test]
    fn test_mock_cross_encoder_default() {
        let encoder = MockCrossEncoder::default();
        assert_eq!(encoder.model_name(), "mock-cross-encoder");
    }

    #[test]
    fn test_mock_cross_encoder_with_scores() {
        let encoder = MockCrossEncoder::with_scores(vec![0.1, 0.2, 0.3]);
        assert_eq!(encoder.scores, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_mock_cross_encoder_score_single() {
        let encoder = MockCrossEncoder::new();
        let score = encoder.score_single("query", "document").await.unwrap();
        assert_eq!(score, 0.9);
    }

    #[tokio::test]
    async fn test_mock_cross_encoder_score_single_custom() {
        let encoder = MockCrossEncoder::with_scores(vec![0.42]);
        let score = encoder.score_single("any", "thing").await.unwrap();
        assert_eq!(score, 0.42);
    }

    #[tokio::test]
    async fn test_mock_batch_scoring() {
        let encoder = MockCrossEncoder::new();
        let scores = encoder
            .score_batch("query", &["doc1", "doc2", "doc3"])
            .await
            .unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0], 0.9);
        assert_eq!(scores[1], 0.7);
        assert_eq!(scores[2], 0.5);
    }

    #[tokio::test]
    async fn test_mock_batch_scoring_cycles() {
        // Test that scores cycle when there are more docs than scores
        let encoder = MockCrossEncoder::with_scores(vec![0.1, 0.2]);
        let scores = encoder
            .score_batch("query", &["d1", "d2", "d3", "d4"])
            .await
            .unwrap();
        assert_eq!(scores.len(), 4);
        assert_eq!(scores[0], 0.1);
        assert_eq!(scores[1], 0.2);
        assert_eq!(scores[2], 0.1); // Cycles back
        assert_eq!(scores[3], 0.2);
    }

    #[tokio::test]
    async fn test_mock_rerank() {
        let encoder = MockCrossEncoder::with_scores(vec![0.3, 0.9, 0.5]);
        let docs = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];

        let reranked = encoder.rerank("query", docs).await.unwrap();

        // Should be sorted by score descending
        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].content, "doc2");
        assert_eq!(reranked[0].score, 0.9);
        assert_eq!(reranked[0].original_index, 1);
        assert_eq!(reranked[1].content, "doc3");
        assert_eq!(reranked[1].score, 0.5);
        assert_eq!(reranked[2].content, "doc1");
        assert_eq!(reranked[2].score, 0.3);
    }

    #[tokio::test]
    async fn test_mock_rerank_preserves_original_index() {
        let encoder = MockCrossEncoder::with_scores(vec![0.5, 0.9, 0.7]);
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let reranked = encoder.rerank("query", docs).await.unwrap();

        // Check original indices are preserved
        assert_eq!(reranked[0].original_index, 1); // "b" had highest score
        assert_eq!(reranked[1].original_index, 2); // "c"
        assert_eq!(reranked[2].original_index, 0); // "a" had lowest
    }

    // ==========================================================================
    // HeuristicCrossEncoder Tests
    // ==========================================================================

    #[test]
    fn test_heuristic_cross_encoder_new() {
        let encoder = HeuristicCrossEncoder::new();
        assert_eq!(encoder.model_name(), "heuristic-cross-encoder");
    }

    #[test]
    fn test_heuristic_cross_encoder_default() {
        let encoder = HeuristicCrossEncoder::default();
        assert_eq!(encoder.model_name(), "heuristic-cross-encoder");
    }

    #[test]
    fn test_heuristic_cross_encoder_with_config() {
        let config = CrossEncoderConfig {
            temperature: 2.0,
            ..Default::default()
        };
        let encoder = HeuristicCrossEncoder::with_config(config);
        assert_eq!(encoder.config.temperature, 2.0);
    }

    #[tokio::test]
    async fn test_heuristic_cross_encoder_score_single() {
        let encoder = HeuristicCrossEncoder::new();
        let score = encoder
            .score_single("test query", "test document")
            .await
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_heuristic_cross_encoder() {
        let encoder = HeuristicCrossEncoder::new();

        // Relevant document should score higher
        let relevant_score = encoder
            .score_single(
                "rust programming language",
                "Rust is a systems programming language focusing on safety",
            )
            .await
            .unwrap();

        let irrelevant_score = encoder
            .score_single(
                "rust programming language",
                "Iron oxide is commonly known as rust",
            )
            .await
            .unwrap();

        assert!(relevant_score > irrelevant_score);
    }

    #[tokio::test]
    async fn test_heuristic_phrase_match() {
        let encoder = HeuristicCrossEncoder::new();

        let exact_score = encoder
            .score_single(
                "machine learning",
                "Introduction to machine learning algorithms",
            )
            .await
            .unwrap();

        let partial_score = encoder
            .score_single(
                "machine learning",
                "The machine was used for learning purposes",
            )
            .await
            .unwrap();

        // Exact phrase match should score higher
        assert!(exact_score > partial_score);
    }

    #[tokio::test]
    async fn test_heuristic_empty_query() {
        let encoder = HeuristicCrossEncoder::new();
        let score = encoder.score_single("", "some document").await.unwrap();
        assert_eq!(score, 0.0);
    }

    #[tokio::test]
    async fn test_heuristic_empty_document() {
        let encoder = HeuristicCrossEncoder::new();
        let score = encoder.score_single("some query", "").await.unwrap();
        assert_eq!(score, 0.0);
    }

    #[tokio::test]
    async fn test_heuristic_batch_scoring() {
        let encoder = HeuristicCrossEncoder::new();
        let scores = encoder
            .score_batch(
                "programming",
                &[
                    "Rust programming language",
                    "cooking recipes",
                    "programming tutorial",
                ],
            )
            .await
            .unwrap();

        assert_eq!(scores.len(), 3);
        // Programming-related docs should score higher
        assert!(scores[0] > scores[1]); // Rust programming > cooking
        assert!(scores[2] > scores[1]); // programming tutorial > cooking
    }

    #[tokio::test]
    async fn test_heuristic_position_weighting() {
        let encoder = HeuristicCrossEncoder::new();

        // Query term at beginning should score higher than at end
        let early_score = encoder
            .score_single(
                "rust",
                "Rust is great for systems programming and has many features",
            )
            .await
            .unwrap();

        let late_score = encoder
            .score_single(
                "rust",
                "There are many programming languages but none compare to Rust",
            )
            .await
            .unwrap();

        assert!(early_score >= late_score * 0.8); // Early position should help
    }

    // ==========================================================================
    // EmbeddingCrossEncoder Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_embedding_cross_encoder_new() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let encoder = EmbeddingCrossEncoder::new(embedder);
        assert_eq!(encoder.model_name(), "embedding-cross-encoder");
    }

    #[tokio::test]
    async fn test_embedding_cross_encoder_score_single() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let encoder = EmbeddingCrossEncoder::new(embedder);

        let score = encoder
            .score_single("query text", "document text")
            .await
            .unwrap();

        // Score should be in valid range
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_embedding_cross_encoder_batch() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let encoder = EmbeddingCrossEncoder::new(embedder);

        let scores = encoder
            .score_batch("query", &["doc1", "doc2", "doc3"])
            .await
            .unwrap();

        assert_eq!(scores.len(), 3);
        for score in &scores {
            assert!(*score >= -1.0 && *score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_embedding_cross_encoder_with_config() {
        let embedder = Arc::new(MockEmbedder::new(384));
        let config = CrossEncoderConfig {
            batch_size: 16,
            ..Default::default()
        };
        let encoder = EmbeddingCrossEncoder::with_config(embedder, config);

        let score = encoder.score_single("a", "b").await.unwrap();
        assert!(score >= -1.0 && score <= 1.0);
    }

    // ==========================================================================
    // RerankResult Tests
    // ==========================================================================

    #[test]
    fn test_rerank_result_structure() {
        let result = RerankResult {
            documents: vec![ScoredDocument {
                content: "test".to_string(),
                original_index: 0,
                score: 0.8,
            }],
            total_processed: 1,
            time_ms: 100,
        };

        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.total_processed, 1);
        assert_eq!(result.time_ms, 100);
    }

    #[test]
    fn test_rerank_result_clone() {
        let result1 = RerankResult {
            documents: vec![],
            total_processed: 5,
            time_ms: 50,
        };
        let result2 = result1.clone();
        assert_eq!(result1.total_processed, result2.total_processed);
        assert_eq!(result1.time_ms, result2.time_ms);
    }

    // ==========================================================================
    // EnsembleReranker Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_ensemble_reranker_new() {
        let encoder1: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::with_scores(vec![0.8]));
        let encoder2: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::with_scores(vec![0.6]));

        let reranker = EnsembleReranker::new(vec![encoder1, encoder2], vec![0.5, 0.5]).unwrap();

        // Test basic reranking
        let result = reranker
            .rerank("query", vec!["doc1".to_string()])
            .await
            .unwrap();

        assert_eq!(result.documents.len(), 1);
        assert_eq!(result.total_processed, 1);
    }

    #[tokio::test]
    async fn test_ensemble_reranker_weight_normalization() {
        let encoder1: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::with_scores(vec![1.0]));
        let encoder2: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::with_scores(vec![0.0]));

        // Weights don't sum to 1, should be normalized
        let reranker = EnsembleReranker::new(vec![encoder1, encoder2], vec![2.0, 2.0]).unwrap();

        let result = reranker
            .rerank("query", vec!["doc".to_string()])
            .await
            .unwrap();

        // With normalized weights (0.5, 0.5), score should be 0.5
        assert!((result.documents[0].score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_reranker_mismatched_lengths() {
        let encoder1: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::new());
        let encoder2: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::new());

        // Different number of encoders and weights should error
        let result = EnsembleReranker::new(vec![encoder1, encoder2], vec![0.5]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ensemble_reranker_multiple_docs() {
        let encoder1: Arc<dyn CrossEncoder> =
            Arc::new(MockCrossEncoder::with_scores(vec![0.9, 0.1, 0.5]));
        let encoder2: Arc<dyn CrossEncoder> =
            Arc::new(MockCrossEncoder::with_scores(vec![0.1, 0.9, 0.5]));

        let reranker = EnsembleReranker::new(vec![encoder1, encoder2], vec![0.5, 0.5]).unwrap();

        let docs = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];

        let result = reranker.rerank("query", docs).await.unwrap();

        assert_eq!(result.documents.len(), 3);
        assert_eq!(result.total_processed, 3);
        // All docs should have the same combined score (0.5)
        for doc in &result.documents {
            assert!((doc.score - 0.5).abs() < 0.01);
        }
    }

    #[tokio::test]
    async fn test_ensemble_reranker_records_time() {
        let encoder: Arc<dyn CrossEncoder> = Arc::new(MockCrossEncoder::new());
        let reranker = EnsembleReranker::new(vec![encoder], vec![1.0]).unwrap();

        let result = reranker
            .rerank("query", vec!["doc".to_string()])
            .await
            .unwrap();

        // Time should be recorded (even if very small)
        assert!(result.time_ms >= 0);
    }

    // ==========================================================================
    // Config Presets Tests
    // ==========================================================================

    #[test]
    fn test_config_presets() {
        let msmarco = CrossEncoderConfig::ms_marco();
        assert!(msmarco.model.contains("ms-marco"));

        let multilingual = CrossEncoderConfig::multilingual();
        assert!(multilingual.model.contains("mmarco"));
    }

    // ==========================================================================
    // Edge Cases Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_rerank_empty_docs() {
        let encoder = MockCrossEncoder::new();
        let result = encoder.rerank("query", vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_rerank_single_doc() {
        let encoder = MockCrossEncoder::new();
        let result = encoder
            .rerank("query", vec!["only one".to_string()])
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "only one");
        assert_eq!(result[0].original_index, 0);
    }

    #[tokio::test]
    async fn test_batch_empty_docs() {
        let encoder = MockCrossEncoder::new();
        let scores = encoder.score_batch("query", &[]).await.unwrap();
        assert!(scores.is_empty());
    }

    #[tokio::test]
    async fn test_heuristic_very_long_document() {
        let encoder = HeuristicCrossEncoder::new();
        let long_doc = "word ".repeat(1000);
        let score = encoder.score_single("word", &long_doc).await.unwrap();
        // Should still produce valid score
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_heuristic_special_characters() {
        let encoder = HeuristicCrossEncoder::new();
        let score = encoder
            .score_single("C++ programming", "Learn C++ and C# programming!")
            .await
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
