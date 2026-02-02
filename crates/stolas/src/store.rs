//! Vector storage backends.
//!
//! Provides both in-memory and persistent vector storage options.
//! The Lance-based backend offers disk persistence with efficient ANN search.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use infernum_core::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(feature = "lance")]
use futures::TryStreamExt;
#[cfg(feature = "lance")]
use lance::arrow::array::{Array as ArrowArray, RecordBatch as LanceRecordBatch};
#[cfg(feature = "lance")]
use lancedb::query::{ExecutableQuery, QueryBase};

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
            .filter(|record| {
                // Apply metadata filters (all must match)
                params.filters.iter().all(|(key, filter_value)| {
                    match record.metadata.get(key) {
                        Some(record_value) => record_value == filter_value,
                        None => false, // Missing field doesn't match
                    }
                })
            })
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

// ============================================================================
// LanceDB Persistent Store
// ============================================================================

/// Configuration for the Lance store.
#[derive(Debug, Clone)]
pub struct LanceStoreConfig {
    /// Path to the Lance database directory.
    pub path: PathBuf,
    /// Name of the vectors table.
    pub table_name: String,
    /// Embedding dimension (required for creating the table).
    pub dimension: usize,
    /// Number of partitions for IVF index.
    pub num_partitions: Option<u32>,
    /// Number of sub-vectors for PQ encoding.
    pub num_sub_vectors: Option<u32>,
}

impl Default for LanceStoreConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./lance_data"),
            table_name: "vectors".to_string(),
            dimension: 384, // Default for small embedding models
            num_partitions: None,
            num_sub_vectors: None,
        }
    }
}

impl LanceStoreConfig {
    /// Creates a new config with the specified path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Sets the table name.
    #[must_use]
    pub fn with_table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = name.into();
        self
    }

    /// Sets the embedding dimension.
    #[must_use]
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    /// Sets IVF index parameters.
    #[must_use]
    pub fn with_ivf_index(mut self, num_partitions: u32, num_sub_vectors: u32) -> Self {
        self.num_partitions = Some(num_partitions);
        self.num_sub_vectors = Some(num_sub_vectors);
        self
    }
}

/// Lance-based persistent vector store.
///
/// Provides disk-based vector storage with efficient approximate nearest neighbor search.
/// Uses Apache Arrow columnar format for efficient storage and retrieval.
#[cfg(feature = "lance")]
pub struct LanceStore {
    /// Configuration.
    config: LanceStoreConfig,
    /// LanceDB connection.
    db: lancedb::Connection,
    /// Table handle (lazily initialized).
    table: tokio::sync::RwLock<Option<lancedb::Table>>,
}

#[cfg(feature = "lance")]
impl LanceStore {
    /// Creates a new Lance store with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(config: LanceStoreConfig) -> Result<Self> {
        // Ensure the directory exists
        if let Some(parent) = config.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                infernum_core::Error::internal(format!("Failed to create directory: {}", e))
            })?;
        }

        let db = lancedb::connect(config.path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| {
                infernum_core::Error::internal(format!("Failed to connect to LanceDB: {}", e))
            })?;

        Ok(Self {
            config,
            db,
            table: tokio::sync::RwLock::new(None),
        })
    }

    /// Opens an existing store or creates a new one at the default path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn open_or_create(path: impl Into<PathBuf>, dimension: usize) -> Result<Self> {
        let config = LanceStoreConfig::new(path).with_dimension(dimension);
        Self::new(config).await
    }

    /// Gets or creates the vectors table.
    async fn get_or_create_table(&self) -> Result<lancedb::Table> {
        // Check if we already have the table
        {
            let guard = self.table.read().await;
            if let Some(table) = guard.as_ref() {
                return Ok(table.clone());
            }
        }

        // Try to open existing table
        let table_names = self.db.table_names().execute().await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to list tables: {}", e))
        })?;

        let table = if table_names.contains(&self.config.table_name) {
            self.db
                .open_table(&self.config.table_name)
                .execute()
                .await
                .map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to open table: {}", e))
                })?
        } else {
            // Create new table with initial schema
            use arrow_array::{
                ArrayRef, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
            };
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        self.config.dimension as i32,
                    ),
                    false,
                ),
                Field::new("content", DataType::Utf8, false),
                Field::new("metadata", DataType::Utf8, true), // JSON string
            ]));

            // Create empty batch for schema initialization
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema);

            self.db
                .create_table(&self.config.table_name, Box::new(batches))
                .execute()
                .await
                .map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to create table: {}", e))
                })?
        };

        // Cache the table
        *self.table.write().await = Some(table.clone());
        Ok(table)
    }

    /// Converts vector records to an Arrow RecordBatch.
    fn records_to_batch(
        &self,
        records: &[VectorRecord],
    ) -> Result<arrow_array::RecordBatch> {
        use arrow_array::{
            builder::{FixedSizeListBuilder, Float32Builder, StringBuilder},
            ArrayRef, RecordBatch,
        };
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.config.dimension as i32,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        let mut id_builder = StringBuilder::new();
        let mut vector_builder =
            FixedSizeListBuilder::new(Float32Builder::new(), self.config.dimension as i32);
        let mut content_builder = StringBuilder::new();
        let mut metadata_builder = StringBuilder::new();

        for record in records {
            id_builder.append_value(&record.id);

            // Pad or truncate vector to expected dimension
            let values = vector_builder.values();
            for i in 0..self.config.dimension {
                values.append_value(record.vector.get(i).copied().unwrap_or(0.0));
            }
            vector_builder.append(true);

            content_builder.append_value(&record.content);

            let metadata_json = serde_json::to_string(&record.metadata)
                .unwrap_or_else(|_| "{}".to_string());
            metadata_builder.append_value(&metadata_json);
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(vector_builder.finish()),
            Arc::new(content_builder.finish()),
            Arc::new(metadata_builder.finish()),
        ];

        RecordBatch::try_new(schema, columns).map_err(|e| {
            infernum_core::Error::internal(format!("Failed to create record batch: {}", e))
        })
    }

    /// Returns the database path.
    #[must_use]
    pub fn path(&self) -> &PathBuf {
        &self.config.path
    }

    /// Returns the table name.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.config.table_name
    }

    /// Creates an IVF-PQ index for faster search.
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails.
    pub async fn create_index(&self) -> Result<()> {
        let table = self.get_or_create_table().await?;

        let num_partitions = self.config.num_partitions.unwrap_or(256);
        let num_sub_vectors = self.config.num_sub_vectors.unwrap_or(96);

        table
            .create_index(&["vector"], lancedb::index::Index::Auto)
            .execute()
            .await
            .map_err(|e| {
                infernum_core::Error::internal(format!("Failed to create index: {}", e))
            })?;

        tracing::info!(
            num_partitions = num_partitions,
            num_sub_vectors = num_sub_vectors,
            "Created IVF-PQ index"
        );

        Ok(())
    }
}

#[cfg(feature = "lance")]
#[async_trait]
impl VectorStore for LanceStore {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }

        let table = self.get_or_create_table().await?;
        let count = records.len();

        // Delete existing records with same IDs (for upsert semantics)
        let ids: Vec<_> = records.iter().map(|r| format!("'{}'", r.id)).collect();
        let filter = format!("id IN ({})", ids.join(", "));
        let _ = table.delete(&filter).await; // Ignore errors if records don't exist

        // Insert new records
        let batch = self.records_to_batch(&records)?;
        let batches = arrow_array::RecordBatchIterator::new(
            vec![Ok(batch.clone())],
            batch.schema(),
        );

        table.add(Box::new(batches)).execute().await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to insert records: {}", e))
        })?;

        Ok(count)
    }

    async fn search(&self, query: &[f32], params: SearchParams) -> Result<Vec<SearchResult>> {
        let table = self.get_or_create_table().await?;

        // Pad query vector to expected dimension
        let mut query_vec = query.to_vec();
        query_vec.resize(self.config.dimension, 0.0);

        let mut query_builder = table
            .vector_search(query_vec)
            .map_err(|e| infernum_core::Error::internal(format!("Search error: {}", e)))?
            .limit(params.top_k);

        // Apply distance threshold if min_score specified (convert to distance)
        if let Some(min_score) = params.min_score {
            // Cosine similarity to L2 distance approximation
            let max_distance = 2.0 * (1.0 - min_score);
            query_builder = query_builder.distance_range(Some(0.0), Some(max_distance));
        }

        let results = query_builder.execute().await.map_err(|e| {
            infernum_core::Error::internal(format!("Search execution failed: {}", e))
        })?;

        let mut search_results = Vec::new();
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| infernum_core::Error::internal(format!("Failed to collect results: {}", e)))?;

        for batch in batches {
            use arrow_array::cast::AsArray;

            let id_col: &dyn arrow_array::Array = batch.column_by_name("id").ok_or_else(|| {
                infernum_core::Error::internal("Missing id column".to_string())
            })?;
            let content_col: &dyn arrow_array::Array = batch.column_by_name("content").ok_or_else(|| {
                infernum_core::Error::internal("Missing content column".to_string())
            })?;
            let vector_col: &dyn arrow_array::Array = batch.column_by_name("vector").ok_or_else(|| {
                infernum_core::Error::internal("Missing vector column".to_string())
            })?;
            let metadata_col: Option<&dyn arrow_array::Array> = batch.column_by_name("metadata").map(|c| c.as_ref());
            let distance_col: Option<&dyn arrow_array::Array> = batch.column_by_name("_distance").map(|c| c.as_ref());

            let ids = id_col.as_string::<i32>();
            let contents = content_col.as_string::<i32>();
            let vectors = vector_col.as_fixed_size_list();

            for i in 0..batch.num_rows() {
                let id = ids.value(i).to_string();
                let content = contents.value(i).to_string();

                // Extract vector
                let vec_array = vectors.value(i);
                let vec_values: Vec<f32> = (0..vec_array.len())
                    .filter_map(|j| {
                        vec_array
                            .as_primitive::<arrow_array::types::Float32Type>()
                            .value(j)
                            .into()
                    })
                    .collect();

                // Extract metadata
                let metadata: HashMap<String, serde_json::Value> = metadata_col
                    .and_then(|col: &dyn arrow_array::Array| {
                        let str_array = col.as_string::<i32>();
                        if !str_array.is_null(i) {
                            serde_json::from_str(str_array.value(i)).ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                // Convert distance to similarity score
                let score = distance_col
                    .map(|col: &dyn arrow_array::Array| {
                        let dist_array = col.as_primitive::<arrow_array::types::Float32Type>();
                        let distance = dist_array.value(i);
                        // Convert L2 distance to cosine similarity approximation
                        1.0 - (distance / 2.0).min(1.0)
                    })
                    .unwrap_or(1.0);

                search_results.push(SearchResult {
                    record: VectorRecord {
                        id,
                        vector: vec_values,
                        content,
                        metadata,
                    },
                    score,
                });
            }
        }

        // Sort by score descending
        search_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(search_results)
    }

    async fn delete(&self, ids: Vec<String>) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }

        let table = self.get_or_create_table().await?;
        let count_before = self.count().await?;

        let quoted_ids: Vec<_> = ids.iter().map(|id| format!("'{}'", id)).collect();
        let filter = format!("id IN ({})", quoted_ids.join(", "));

        table.delete(&filter).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to delete records: {}", e))
        })?;

        let count_after = self.count().await?;
        Ok(count_before.saturating_sub(count_after))
    }

    async fn get(&self, ids: Vec<String>) -> Result<Vec<VectorRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let table = self.get_or_create_table().await?;

        let quoted_ids: Vec<_> = ids.iter().map(|id| format!("'{}'", id)).collect();
        let filter = format!("id IN ({})", quoted_ids.join(", "));

        let results = table
            .query()
            .only_if(&filter)
            .execute()
            .await
            .map_err(|e| {
                infernum_core::Error::internal(format!("Failed to query records: {}", e))
            })?;

        let mut records = Vec::new();
        let batches: Vec<arrow_array::RecordBatch> = results
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| infernum_core::Error::internal(format!("Failed to collect results: {}", e)))?;

        for batch in batches {
            use arrow_array::cast::AsArray;

            let id_col: &dyn arrow_array::Array = batch
                .column_by_name("id")
                .ok_or_else(|| infernum_core::Error::internal("Missing 'id' column in result"))?;
            let content_col: &dyn arrow_array::Array = batch
                .column_by_name("content")
                .ok_or_else(|| infernum_core::Error::internal("Missing 'content' column in result"))?;
            let vector_col: &dyn arrow_array::Array = batch
                .column_by_name("vector")
                .ok_or_else(|| infernum_core::Error::internal("Missing 'vector' column in result"))?;
            let metadata_col: Option<&dyn arrow_array::Array> = batch.column_by_name("metadata").map(|c| c.as_ref());

            let id_array = id_col.as_string::<i32>();
            let content_array = content_col.as_string::<i32>();
            let vector_array = vector_col.as_fixed_size_list();

            for i in 0..batch.num_rows() {
                let id = id_array.value(i).to_string();
                let content = content_array.value(i).to_string();

                let vec_arr = vector_array.value(i);
                let vector: Vec<f32> = (0..vec_arr.len())
                    .map(|j| {
                        vec_arr
                            .as_primitive::<arrow_array::types::Float32Type>()
                            .value(j)
                    })
                    .collect();

                let metadata: HashMap<String, serde_json::Value> = metadata_col
                    .and_then(|col: &dyn arrow_array::Array| {
                        let str_array = col.as_string::<i32>();
                        if !str_array.is_null(i) {
                            serde_json::from_str(str_array.value(i)).ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                records.push(VectorRecord {
                    id,
                    vector,
                    content,
                    metadata,
                });
            }
        }

        Ok(records)
    }

    async fn count(&self) -> Result<usize> {
        let table = self.get_or_create_table().await?;
        let count = table.count_rows(None).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to count rows: {}", e))
        })?;
        Ok(count)
    }
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
        assert_eq!(record.metadata.get("key").unwrap(), &serde_json::json!("value"));
        assert_eq!(record.metadata.get("num").unwrap(), &serde_json::json!(42));
    }

    #[test]
    fn test_search_params_default() {
        let params = SearchParams::default();
        assert_eq!(params.top_k, 10);
        assert!(params.min_score.is_none());
        assert!(params.filters.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[tokio::test]
    async fn test_in_memory_store_delete() {
        let store = InMemoryStore::new();

        let record = VectorRecord::new(vec![1.0, 0.0, 0.0], "test");
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

        let record1 = VectorRecord::new(vec![1.0, 0.0], "test 1");
        let record2 = VectorRecord::new(vec![0.0, 1.0], "test 2");
        let id1 = record1.id.clone();
        let id2 = record2.id.clone();

        store.upsert(vec![record1, record2]).await.unwrap();

        let fetched = store.get(vec![id1.clone(), id2.clone()]).await.unwrap();
        assert_eq!(fetched.len(), 2);

        let fetched_single = store.get(vec![id1]).await.unwrap();
        assert_eq!(fetched_single.len(), 1);
    }

    #[tokio::test]
    async fn test_in_memory_store_upsert_update() {
        let store = InMemoryStore::new();

        let mut record = VectorRecord::new(vec![1.0, 0.0], "original");
        let id = record.id.clone();

        store.upsert(vec![record.clone()]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        // Update the record
        record.content = "updated".to_string();
        store.upsert(vec![record]).await.unwrap();

        // Should still be only 1 record (upsert)
        assert_eq!(store.count().await.unwrap(), 1);

        let fetched = store.get(vec![id]).await.unwrap();
        assert_eq!(fetched[0].content, "updated");
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

        // Only "exact match" should pass the 0.9 threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "exact match");
    }

    #[test]
    fn test_lance_store_config_default() {
        let config = LanceStoreConfig::default();
        assert_eq!(config.table_name, "vectors");
        assert_eq!(config.dimension, 384);
        assert!(config.num_partitions.is_none());
    }

    #[test]
    fn test_lance_store_config_builder() {
        let config = LanceStoreConfig::new("/tmp/lance_test")
            .with_table_name("embeddings")
            .with_dimension(768)
            .with_ivf_index(128, 48);

        assert_eq!(config.path, PathBuf::from("/tmp/lance_test"));
        assert_eq!(config.table_name, "embeddings");
        assert_eq!(config.dimension, 768);
        assert_eq!(config.num_partitions, Some(128));
        assert_eq!(config.num_sub_vectors, Some(48));
    }

    // ==========================================================================
    // Metadata Filtering Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_in_memory_store_filter_by_string() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "doc about rust")
                .with_metadata("category", serde_json::json!("programming")),
            VectorRecord::new(vec![0.9, 0.1], "doc about cooking")
                .with_metadata("category", serde_json::json!("food")),
            VectorRecord::new(vec![0.8, 0.2], "doc about python")
                .with_metadata("category", serde_json::json!("programming")),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), serde_json::json!("programming"));

        let params = SearchParams {
            top_k: 10,
            min_score: None,
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        // Should only return programming docs
        assert_eq!(results.len(), 2);
        for result in &results {
            assert_eq!(
                result.record.metadata.get("category").unwrap(),
                &serde_json::json!("programming")
            );
        }
    }

    #[tokio::test]
    async fn test_in_memory_store_filter_by_number() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "chapter 1")
                .with_metadata("chapter", serde_json::json!(1)),
            VectorRecord::new(vec![0.9, 0.1], "chapter 2")
                .with_metadata("chapter", serde_json::json!(2)),
            VectorRecord::new(vec![0.8, 0.2], "chapter 3")
                .with_metadata("chapter", serde_json::json!(3)),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("chapter".to_string(), serde_json::json!(2));

        let params = SearchParams {
            top_k: 10,
            min_score: None,
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "chapter 2");
    }

    #[tokio::test]
    async fn test_in_memory_store_filter_multiple_conditions() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "rust programming book")
                .with_metadata("category", serde_json::json!("programming"))
                .with_metadata("language", serde_json::json!("rust")),
            VectorRecord::new(vec![0.9, 0.1], "python programming book")
                .with_metadata("category", serde_json::json!("programming"))
                .with_metadata("language", serde_json::json!("python")),
            VectorRecord::new(vec![0.8, 0.2], "rust systems book")
                .with_metadata("category", serde_json::json!("systems"))
                .with_metadata("language", serde_json::json!("rust")),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), serde_json::json!("programming"));
        filters.insert("language".to_string(), serde_json::json!("rust"));

        let params = SearchParams {
            top_k: 10,
            min_score: None,
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        // Should only return the rust programming book
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "rust programming book");
    }

    #[tokio::test]
    async fn test_in_memory_store_filter_no_matches() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "doc 1")
                .with_metadata("type", serde_json::json!("article")),
            VectorRecord::new(vec![0.9, 0.1], "doc 2")
                .with_metadata("type", serde_json::json!("article")),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("type".to_string(), serde_json::json!("book"));

        let params = SearchParams {
            top_k: 10,
            min_score: None,
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_store_filter_missing_field() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "doc with category")
                .with_metadata("category", serde_json::json!("programming")),
            VectorRecord::new(vec![0.9, 0.1], "doc without category"),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("category".to_string(), serde_json::json!("programming"));

        let params = SearchParams {
            top_k: 10,
            min_score: None,
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        // Should only return doc with matching category
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "doc with category");
    }

    #[tokio::test]
    async fn test_in_memory_store_filter_with_min_score() {
        let store = InMemoryStore::new();

        let records = vec![
            VectorRecord::new(vec![1.0, 0.0], "high similarity match")
                .with_metadata("type", serde_json::json!("article")),
            VectorRecord::new(vec![0.0, 1.0], "low similarity match")
                .with_metadata("type", serde_json::json!("article")),
        ];

        store.upsert(records).await.unwrap();

        let mut filters = HashMap::new();
        filters.insert("type".to_string(), serde_json::json!("article"));

        let params = SearchParams {
            top_k: 10,
            min_score: Some(0.5),
            filters,
        };

        let results = store.search(&[1.0, 0.0], params).await.unwrap();

        // Both match filter, but only high similarity passes min_score
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.content, "high similarity match");
    }
}
