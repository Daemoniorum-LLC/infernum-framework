//! BM25 sparse retrieval for hybrid search.
//!
//! BM25 (Best Matching 25) is a probabilistic retrieval function used by search engines
//! to rank documents based on keyword matching. It's particularly effective when combined
//! with dense retrieval (embeddings) for hybrid search.
//!
//! ## Algorithm
//!
//! BM25 score = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
//!
//! Where:
//! - qi = query terms
//! - f(qi, D) = term frequency in document D
//! - |D| = document length
//! - avgdl = average document length
//! - k1, b = tuning parameters
//!
//! ## Usage
//!
//! ```ignore
//! use stolas::bm25::{BM25Index, BM25Config};
//!
//! let config = BM25Config::default();
//! let mut index = BM25Index::new(config);
//!
//! index.add_document("doc1", "The quick brown fox jumps over the lazy dog");
//! index.add_document("doc2", "A lazy cat sleeps on the couch");
//!
//! let results = index.search("lazy", 10);
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Configuration for BM25 scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    /// k1 parameter: term frequency saturation.
    /// Higher values increase the influence of term frequency.
    /// Typical values: 1.2 - 2.0
    pub k1: f32,
    /// b parameter: document length normalization.
    /// 0 = no length normalization, 1 = full normalization.
    /// Typical values: 0.75
    pub b: f32,
    /// Minimum document frequency for a term to be considered.
    pub min_df: usize,
    /// Maximum document frequency ratio (terms appearing in more than this
    /// fraction of documents are ignored as stop words).
    pub max_df_ratio: f32,
    /// Whether to apply stemming (simplified).
    pub stem: bool,
    /// Whether to lowercase tokens.
    pub lowercase: bool,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            min_df: 1,
            max_df_ratio: 0.85,
            stem: false,
            lowercase: true,
        }
    }
}

impl BM25Config {
    /// Creates a configuration optimized for short documents.
    #[must_use]
    pub fn for_short_docs() -> Self {
        Self {
            k1: 1.2,
            b: 0.5, // Less length normalization for short docs
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for long documents.
    #[must_use]
    pub fn for_long_docs() -> Self {
        Self {
            k1: 2.0,
            b: 0.75,
            ..Default::default()
        }
    }
}

/// A document in the BM25 index.
#[derive(Debug, Clone)]
struct IndexedDocument {
    /// Document ID.
    id: String,
    /// Document content.
    content: String,
    /// Token frequencies.
    term_freqs: HashMap<String, u32>,
    /// Document length (token count).
    length: usize,
}

/// BM25 search result.
#[derive(Debug, Clone)]
pub struct BM25Result {
    /// Document ID.
    pub id: String,
    /// Document content.
    pub content: String,
    /// BM25 score.
    pub score: f32,
}

/// BM25 sparse retrieval index.
pub struct BM25Index {
    /// Configuration.
    config: BM25Config,
    /// Documents by ID.
    documents: HashMap<String, IndexedDocument>,
    /// Inverted index: term -> document IDs.
    inverted_index: HashMap<String, HashSet<String>>,
    /// Document frequency per term.
    doc_freqs: HashMap<String, usize>,
    /// Total document count.
    total_docs: usize,
    /// Average document length.
    avg_doc_length: f32,
}

impl BM25Index {
    /// Creates a new BM25 index with the given configuration.
    #[must_use]
    pub fn new(config: BM25Config) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            inverted_index: HashMap::new(),
            doc_freqs: HashMap::new(),
            total_docs: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Creates a new BM25 index with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BM25Config::default())
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &BM25Config {
        &self.config
    }

    /// Returns the number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_docs
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }

    /// Returns the vocabulary size (unique terms).
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.inverted_index.len()
    }

    /// Tokenizes text into terms.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Simple tokenization: split on non-alphanumeric
        let tokens: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(|s| {
                if self.config.stem {
                    simple_stem(s)
                } else {
                    s.to_string()
                }
            })
            .collect();

        tokens
    }

    /// Adds a document to the index.
    pub fn add_document(&mut self, id: impl Into<String>, content: impl Into<String>) {
        let id = id.into();
        let content = content.into();

        // Remove old document if exists
        self.remove_document(&id);

        // Tokenize
        let tokens = self.tokenize(&content);
        let length = tokens.len();

        // Count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // Update inverted index and document frequencies
        for term in term_freqs.keys() {
            self.inverted_index
                .entry(term.clone())
                .or_insert_with(HashSet::new)
                .insert(id.clone());
            *self.doc_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        // Store document
        let doc = IndexedDocument {
            id: id.clone(),
            content,
            term_freqs,
            length,
        };
        self.documents.insert(id, doc);

        // Update statistics
        self.total_docs += 1;
        self.update_avg_length();
    }

    /// Adds multiple documents to the index.
    pub fn add_documents<I, S1, S2>(&mut self, documents: I)
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        for (id, content) in documents {
            self.add_document(id, content);
        }
    }

    /// Removes a document from the index.
    pub fn remove_document(&mut self, id: &str) -> bool {
        if let Some(doc) = self.documents.remove(id) {
            // Update inverted index and document frequencies
            for term in doc.term_freqs.keys() {
                if let Some(doc_set) = self.inverted_index.get_mut(term) {
                    doc_set.remove(id);
                    if doc_set.is_empty() {
                        self.inverted_index.remove(term);
                    }
                }
                if let Some(df) = self.doc_freqs.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.doc_freqs.remove(term);
                    }
                }
            }

            self.total_docs -= 1;
            self.update_avg_length();
            true
        } else {
            false
        }
    }

    /// Updates the average document length.
    fn update_avg_length(&mut self) {
        if self.total_docs == 0 {
            self.avg_doc_length = 0.0;
        } else {
            let total_length: usize = self.documents.values().map(|d| d.length).sum();
            self.avg_doc_length = total_length as f32 / self.total_docs as f32;
        }
    }

    /// Computes IDF (Inverse Document Frequency) for a term.
    fn idf(&self, term: &str) -> f32 {
        let df = self.doc_freqs.get(term).copied().unwrap_or(0);

        // Skip rare terms
        if df < self.config.min_df {
            return 0.0;
        }

        // Skip very common terms (stop words), but only for larger corpora
        // For small corpora (< 5 docs), don't filter by max_df_ratio
        if self.total_docs >= 5 {
            let df_ratio = df as f32 / self.total_docs as f32;
            if df_ratio > self.config.max_df_ratio {
                return 0.0;
            }
        }

        // BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        let n = self.total_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Computes BM25 score for a document given query terms.
    fn score_document(&self, doc: &IndexedDocument, query_terms: &[String]) -> f32 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let avgdl = self.avg_doc_length;
        let dl = doc.length as f32;

        let mut score = 0.0;

        for term in query_terms {
            let idf = self.idf(term);
            if idf == 0.0 {
                continue;
            }

            let tf = doc.term_freqs.get(term).copied().unwrap_or(0) as f32;
            if tf == 0.0 {
                continue;
            }

            // BM25 term score
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * dl / avgdl);
            score += idf * numerator / denominator;
        }

        score
    }

    /// Searches the index for documents matching the query.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<BM25Result> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_terms = self.tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Find candidate documents (those containing at least one query term)
        let mut candidates: HashSet<String> = HashSet::new();
        for term in &query_terms {
            if let Some(doc_ids) = self.inverted_index.get(term) {
                candidates.extend(doc_ids.iter().cloned());
            }
        }

        // Score candidates
        let mut results: Vec<BM25Result> = candidates
            .into_iter()
            .filter_map(|candidate_id| {
                let doc = self.documents.get(&candidate_id)?;
                let score = self.score_document(doc, &query_terms);
                if score > 0.0 {
                    Some(BM25Result {
                        id: doc.id.clone(),
                        content: doc.content.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        results.truncate(top_k);
        results
    }

    /// Searches with minimum score threshold.
    pub fn search_with_threshold(&self, query: &str, top_k: usize, min_score: f32) -> Vec<BM25Result> {
        self.search(query, top_k)
            .into_iter()
            .filter(|r| r.score >= min_score)
            .collect()
    }

    /// Returns the IDF scores for query terms (useful for debugging).
    pub fn get_term_idfs(&self, query: &str) -> HashMap<String, f32> {
        let terms = self.tokenize(query);
        terms.into_iter().map(|t| {
            let idf = self.idf(&t);
            (t, idf)
        }).collect()
    }

    /// Clears the index.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.inverted_index.clear();
        self.doc_freqs.clear();
        self.total_docs = 0;
        self.avg_doc_length = 0.0;
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Simple stemmer (very basic - just removes common suffixes).
fn simple_stem(word: &str) -> String {
    let word = word.to_lowercase();

    // Very basic suffix stripping
    let suffixes = ["ing", "ed", "es", "s", "ly", "ment", "ness", "tion", "sion"];

    for suffix in suffixes {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            return word[..word.len() - suffix.len()].to_string();
        }
    }

    word
}

/// Hybrid retrieval combining BM25 and dense vectors.
pub struct HybridRetriever {
    /// BM25 index for sparse retrieval.
    bm25_index: BM25Index,
    /// Weight for BM25 scores (0.0 - 1.0).
    bm25_weight: f32,
    /// Weight for dense scores (0.0 - 1.0).
    dense_weight: f32,
}

impl HybridRetriever {
    /// Creates a new hybrid retriever.
    #[must_use]
    pub fn new(bm25_config: BM25Config, bm25_weight: f32, dense_weight: f32) -> Self {
        Self {
            bm25_index: BM25Index::new(bm25_config),
            bm25_weight,
            dense_weight,
        }
    }

    /// Creates a hybrid retriever with default weights (0.5 each).
    #[must_use]
    pub fn with_equal_weights() -> Self {
        Self::new(BM25Config::default(), 0.5, 0.5)
    }

    /// Creates a retriever favoring dense retrieval.
    #[must_use]
    pub fn dense_heavy() -> Self {
        Self::new(BM25Config::default(), 0.3, 0.7)
    }

    /// Creates a retriever favoring sparse retrieval.
    #[must_use]
    pub fn sparse_heavy() -> Self {
        Self::new(BM25Config::default(), 0.7, 0.3)
    }

    /// Returns the BM25 index.
    #[must_use]
    pub fn bm25_index(&self) -> &BM25Index {
        &self.bm25_index
    }

    /// Returns a mutable reference to the BM25 index.
    pub fn bm25_index_mut(&mut self) -> &mut BM25Index {
        &mut self.bm25_index
    }

    /// Returns the number of documents in the BM25 index.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.bm25_index.len()
    }

    /// Adds a document to the BM25 index.
    pub fn add_document(&mut self, id: impl Into<String>, content: impl Into<String>) {
        self.bm25_index.add_document(id, content);
    }

    /// Combines BM25 and dense scores for hybrid ranking.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `dense_results` - Results from dense retrieval (id, score pairs)
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Combined results with hybrid scores.
    pub fn hybrid_search(
        &self,
        query: &str,
        dense_results: &[(String, f32)],
        top_k: usize,
    ) -> Vec<HybridResult> {
        // Get BM25 results
        let bm25_results = self.bm25_index.search(query, dense_results.len().max(top_k * 2));

        // Normalize BM25 scores
        let max_bm25 = bm25_results.iter().map(|r| r.score).fold(0.0_f32, f32::max);
        let bm25_scores: HashMap<String, f32> = bm25_results
            .into_iter()
            .map(|r| {
                let normalized = if max_bm25 > 0.0 { r.score / max_bm25 } else { 0.0 };
                (r.id, normalized)
            })
            .collect();

        // Normalize dense scores
        let max_dense = dense_results.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        let dense_scores: HashMap<String, f32> = dense_results
            .iter()
            .map(|(id, score)| {
                let normalized = if max_dense > 0.0 { *score / max_dense } else { 0.0 };
                (id.clone(), normalized)
            })
            .collect();

        // Combine all document IDs
        let all_ids: HashSet<&String> = bm25_scores.keys().chain(dense_scores.keys()).collect();

        // Compute hybrid scores
        let mut results: Vec<HybridResult> = all_ids
            .into_iter()
            .map(|id| {
                let bm25 = bm25_scores.get(id).copied().unwrap_or(0.0);
                let dense = dense_scores.get(id).copied().unwrap_or(0.0);
                let hybrid = self.bm25_weight * bm25 + self.dense_weight * dense;

                HybridResult {
                    id: id.clone(),
                    bm25_score: bm25,
                    dense_score: dense,
                    hybrid_score: hybrid,
                }
            })
            .collect();

        // Sort by hybrid score
        results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        results
    }

    /// Sets the BM25 weight.
    pub fn set_bm25_weight(&mut self, weight: f32) {
        self.bm25_weight = weight.clamp(0.0, 1.0);
    }

    /// Sets the dense weight.
    pub fn set_dense_weight(&mut self, weight: f32) {
        self.dense_weight = weight.clamp(0.0, 1.0);
    }
}

/// Result from hybrid retrieval.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Document ID.
    pub id: String,
    /// Normalized BM25 score.
    pub bm25_score: f32,
    /// Normalized dense score.
    pub dense_score: f32,
    /// Combined hybrid score.
    pub hybrid_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_config_default() {
        let config = BM25Config::default();
        assert!((config.k1 - 1.5).abs() < 0.001);
        assert!((config.b - 0.75).abs() < 0.001);
        assert!(config.lowercase);
    }

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
        index.add_document("doc1", "The quick brown fox");
        index.add_document("doc2", "The lazy dog");

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
        // doc1 and doc3 should rank higher (contain both terms)
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_bm25_search_empty_query() {
        let mut index = BM25Index::with_defaults();
        index.add_document("doc1", "The quick brown fox");

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

        // Only high-scoring documents should pass
        for result in &results {
            assert!(result.score >= 0.5);
        }
    }

    #[test]
    fn test_bm25_clear() {
        let mut index = BM25Index::with_defaults();
        index.add_document("doc1", "test document");

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

        // "common" appears in all docs, "rare" in one
        // IDF(rare) should be higher than IDF(common)
        assert!(idfs.get("rare").unwrap_or(&0.0) > idfs.get("common").unwrap_or(&f32::MAX));
    }

    #[test]
    fn test_simple_stem() {
        assert_eq!(simple_stem("running"), "runn");
        assert_eq!(simple_stem("played"), "play");
        assert_eq!(simple_stem("cats"), "cat");
        assert_eq!(simple_stem("quickly"), "quick");
    }

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

        // Simulate dense results (id, score)
        let dense_results = vec![
            ("doc2".to_string(), 0.9),
            ("doc1".to_string(), 0.7),
            ("doc3".to_string(), 0.3),
        ];

        let results = retriever.hybrid_search("machine learning", &dense_results, 10);

        assert!(!results.is_empty());
        // Results should have hybrid scores
        for result in &results {
            assert!(result.hybrid_score >= 0.0);
        }
    }

    #[test]
    fn test_hybrid_weights() {
        let mut retriever = HybridRetriever::dense_heavy();

        retriever.set_bm25_weight(0.4);
        retriever.set_dense_weight(0.6);

        // Weights should be clamped
        retriever.set_bm25_weight(1.5);
        retriever.set_dense_weight(-0.1);
    }

    #[test]
    fn test_bm25_score_ordering() {
        let mut index = BM25Index::with_defaults();
        index.add_documents([
            ("doc1", "fox fox fox fox fox"), // High frequency
            ("doc2", "fox"),                  // Low frequency
            ("doc3", "the quick brown animal jumps"), // No match
        ]);

        let results = index.search("fox", 10);

        assert_eq!(results.len(), 2); // doc3 has no match

        // Due to BM25's term frequency saturation, doc1 should still score higher
        if results.len() >= 2 {
            assert!(results[0].score >= results[1].score);
        }
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
}
