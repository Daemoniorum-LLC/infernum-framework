//! Vector storage backends.

use std::collections::HashMap;

use async_trait::async_trait;
use infernum_core::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A vector record in the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Unique identifier.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Associated text content.
    pub content: String,
    /// Metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl VectorRecord {
    /// Creates a new vector record.
    #[must_use]
    pub fn new(vector: Vec<f32>, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            vector,
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the record.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A search result from the vector store.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched record.
    pub record: VectorRecord,
    /// Similarity score (higher = more similar).
    pub score: f32,
}

/// Parameters for vector search.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum similarity threshold.
    pub min_score: Option<f32>,
    /// Metadata filters.
    pub filters: HashMap<String, serde_json::Value>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: None,
            filters: HashMap::new(),
        }
    }
}

/// Trait for vector storage backends.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Inserts or updates vectors.
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize>;

    /// Searches for similar vectors.
    async fn search(&self, query: &[f32], params: SearchParams) -> Result<Vec<SearchResult>>;

    /// Deletes vectors by ID.
    async fn delete(&self, ids: Vec<String>) -> Result<usize>;

    /// Gets vectors by ID.
    async fn get(&self, ids: Vec<String>) -> Result<Vec<VectorRecord>>;

    /// Returns the total number of vectors.
    async fn count(&self) -> Result<usize>;
}

/// In-memory vector store (for development/testing).
pub struct InMemoryStore {
    records: parking_lot::RwLock<HashMap<String, VectorRecord>>,
}

impl InMemoryStore {
    /// Creates a new in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize> {
        let count = records.len();
        let mut store = self.records.write();
        for record in records {
            store.insert(record.id.clone(), record);
        }
        Ok(count)
    }

    async fn search(&self, query: &[f32], params: SearchParams) -> Result<Vec<SearchResult>> {
        let store = self.records.read();

        let mut results: Vec<SearchResult> = store
            .values()
            .map(|record| {
                let score = cosine_similarity(query, &record.vector);
                SearchResult {
                    record: record.clone(),
                    score,
                }
            })
            .filter(|r| params.min_score.map_or(true, |min| r.score >= min))
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.top_k);

        Ok(results)
    }

    async fn delete(&self, ids: Vec<String>) -> Result<usize> {
        let mut store = self.records.write();
        let mut count = 0;
        for id in ids {
            if store.remove(&id).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    async fn get(&self, ids: Vec<String>) -> Result<Vec<VectorRecord>> {
        let store = self.records.read();
        Ok(ids
            .into_iter()
            .filter_map(|id| store.get(&id).cloned())
            .collect())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.records.read().len())
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

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0, 0.0], "test 1"),
            VectorRecord::new(vec![0.0, 1.0, 0.0], "test 2"),
        ];

        store.upsert(records).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        let results = store
            .search(&[1.0, 0.0, 0.0], SearchParams::default())
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].score > 0.99);
    }
}
