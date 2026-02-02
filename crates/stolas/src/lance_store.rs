//! LanceDB-backed vector storage.
//!
//! Provides persistent vector storage using LanceDB, a high-performance
//! columnar database optimized for vector similarity search.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use infernum_core::Result;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::Table;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::store::{SearchParams, SearchResult, VectorRecord, VectorStore};

/// Configuration for LanceDB storage.
#[derive(Debug, Clone)]
pub struct LanceConfig {
    /// Database path (directory for local storage).
    pub path: PathBuf,
    /// Table name for vectors.
    pub table_name: String,
    /// Vector dimension (must match embeddings).
    pub dimension: usize,
}

impl Default for LanceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./lance_data"),
            table_name: "vectors".to_string(),
            dimension: 384, // Common embedding dimension
        }
    }
}

impl LanceConfig {
    /// Creates a new configuration with the specified path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Sets the table name.
    #[must_use]
    pub fn with_table(mut self, name: impl Into<String>) -> Self {
        self.table_name = name.into();
        self
    }

    /// Sets the vector dimension.
    #[must_use]
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }
}

/// Internal record format for LanceDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LanceRecord {
    id: String,
    vector: Vec<f32>,
    content: String,
    metadata: String, // JSON-encoded metadata
}

impl From<VectorRecord> for LanceRecord {
    fn from(record: VectorRecord) -> Self {
        Self {
            id: record.id,
            vector: record.vector,
            content: record.content,
            metadata: serde_json::to_string(&record.metadata).unwrap_or_default(),
        }
    }
}

impl TryFrom<LanceRecord> for VectorRecord {
    type Error = infernum_core::Error;

    fn try_from(record: LanceRecord) -> std::result::Result<Self, Self::Error> {
        let metadata: HashMap<String, serde_json::Value> =
            serde_json::from_str(&record.metadata).unwrap_or_default();

        Ok(Self {
            id: record.id,
            vector: record.vector,
            content: record.content,
            metadata,
        })
    }
}

/// LanceDB-backed vector store.
pub struct LanceStore {
    config: LanceConfig,
    db: Arc<lancedb::Connection>,
    table: RwLock<Option<Table>>,
}

impl LanceStore {
    /// Creates a new LanceDB store with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(config: LanceConfig) -> Result<Self> {
        // Ensure directory exists
        if let Some(parent) = config.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to create database directory: {}", e),
            })?;
        }

        let db = connect(config.path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to connect to LanceDB: {}", e),
            })?;

        let store = Self {
            config,
            db: Arc::new(db),
            table: RwLock::new(None),
        };

        // Try to open existing table
        store.ensure_table().await?;

        Ok(store)
    }

    /// Creates a new store with default configuration at the specified path.
    pub async fn at_path(path: impl Into<PathBuf>) -> Result<Self> {
        Self::new(LanceConfig::new(path)).await
    }

    /// Ensures the table exists, creating it if necessary.
    async fn ensure_table(&self) -> Result<()> {
        let table_name = &self.config.table_name;

        // Check if we already have the table cached
        if self.table.read().is_some() {
            return Ok(());
        }

        // Try to open existing table
        let table_result = self.db.open_table(table_name).execute().await;

        match table_result {
            Ok(table) => {
                *self.table.write() = Some(table);
                Ok(())
            }
            Err(_) => {
                // Table doesn't exist, we'll create it on first insert
                Ok(())
            }
        }
    }

    /// Creates the table with initial data.
    async fn create_table_with_data(&self, records: &[LanceRecord]) -> Result<Table> {
        use arrow_array::{
            Array, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
            builder::FixedSizeListBuilder,
        };
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc as StdArc;

        let dim = self.config.dimension as i32;

        // Build schema
        let schema = StdArc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    StdArc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false),
        ]));

        // Build arrays
        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let id_array = StringArray::from(ids);

        let contents: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
        let content_array = StringArray::from(contents);

        let metadatas: Vec<&str> = records.iter().map(|r| r.metadata.as_str()).collect();
        let metadata_array = StringArray::from(metadatas);

        // Build vector array
        let mut vector_builder = FixedSizeListBuilder::new(
            arrow_array::builder::Float32Builder::new(),
            dim,
        );

        for record in records {
            let values = vector_builder.values();
            for &v in &record.vector {
                values.append_value(v);
            }
            vector_builder.append(true);
        }
        let vector_array = vector_builder.finish();

        // Create batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(id_array) as StdArc<dyn Array>,
                StdArc::new(vector_array) as StdArc<dyn Array>,
                StdArc::new(content_array) as StdArc<dyn Array>,
                StdArc::new(metadata_array) as StdArc<dyn Array>,
            ],
        )
        .map_err(|e| infernum_core::Error::Internal {
            message: format!("Failed to create record batch: {}", e),
        })?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let table = self
            .db
            .create_table(&self.config.table_name, Box::new(batches))
            .execute()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to create table: {}", e),
            })?;

        Ok(table)
    }

    /// Adds records to an existing table.
    async fn add_to_table(&self, table: &Table, records: &[LanceRecord]) -> Result<()> {
        use arrow_array::{
            Array, RecordBatch, RecordBatchIterator, StringArray,
            builder::FixedSizeListBuilder,
        };
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc as StdArc;

        let dim = self.config.dimension as i32;

        let schema = StdArc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    StdArc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                ),
                false,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, false),
        ]));

        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let id_array = StringArray::from(ids);

        let contents: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
        let content_array = StringArray::from(contents);

        let metadatas: Vec<&str> = records.iter().map(|r| r.metadata.as_str()).collect();
        let metadata_array = StringArray::from(metadatas);

        let mut vector_builder = FixedSizeListBuilder::new(
            arrow_array::builder::Float32Builder::new(),
            dim,
        );

        for record in records {
            let values = vector_builder.values();
            for &v in &record.vector {
                values.append_value(v);
            }
            vector_builder.append(true);
        }
        let vector_array = vector_builder.finish();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(id_array) as StdArc<dyn Array>,
                StdArc::new(vector_array) as StdArc<dyn Array>,
                StdArc::new(content_array) as StdArc<dyn Array>,
                StdArc::new(metadata_array) as StdArc<dyn Array>,
            ],
        )
        .map_err(|e| infernum_core::Error::Internal {
            message: format!("Failed to create record batch: {}", e),
        })?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        table
            .add(Box::new(batches))
            .execute()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to add records: {}", e),
            })?;

        Ok(())
    }
}

#[async_trait]
impl VectorStore for LanceStore {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }

        let count = records.len();
        let lance_records: Vec<LanceRecord> = records.into_iter().map(Into::into).collect();

        // Check if table exists
        let table_guard = self.table.read();
        if let Some(table) = table_guard.as_ref() {
            // Delete existing records with same IDs first (for upsert semantics)
            let ids: Vec<&str> = lance_records.iter().map(|r| r.id.as_str()).collect();
            let filter = format!(
                "id IN ({})",
                ids.iter()
                    .map(|id| format!("'{}'", id))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // Ignore delete errors (records might not exist)
            let _ = table.delete(&filter).await;

            self.add_to_table(table, &lance_records).await?;
        } else {
            drop(table_guard);

            // Create new table with initial data
            let table = self.create_table_with_data(&lance_records).await?;
            *self.table.write() = Some(table);
        }

        tracing::debug!(count, "Upserted records to LanceDB");
        Ok(count)
    }

    async fn search(&self, query: &[f32], params: SearchParams) -> Result<Vec<SearchResult>> {
        let table_guard = self.table.read();
        let table = table_guard.as_ref().ok_or_else(|| infernum_core::Error::Internal {
            message: "Table not initialized".to_string(),
        })?;

        let results = table
            .vector_search(query.to_vec())
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to create vector search: {}", e),
            })?
            .limit(params.top_k)
            .execute()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Vector search failed: {}", e),
            })?;

        // Convert results to SearchResult
        let mut search_results = Vec::new();

        // Process RecordBatch stream
        use futures::TryStreamExt;
        let batches: Vec<_> = results
            .try_collect()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to collect results: {}", e),
            })?;

        for batch in batches {
            use arrow_array::{Array, Float32Array, StringArray, cast::AsArray};

            let num_rows = batch.num_rows();

            let id_col = batch
                .column_by_name("id")
                .ok_or_else(|| infernum_core::Error::Internal {
                    message: "Missing id column".to_string(),
                })?;
            let id_array = id_col.as_string::<i32>();

            let content_col = batch
                .column_by_name("content")
                .ok_or_else(|| infernum_core::Error::Internal {
                    message: "Missing content column".to_string(),
                })?;
            let content_array = content_col.as_string::<i32>();

            let metadata_col = batch
                .column_by_name("metadata")
                .ok_or_else(|| infernum_core::Error::Internal {
                    message: "Missing metadata column".to_string(),
                })?;
            let metadata_array = metadata_col.as_string::<i32>();

            // Get distance column (LanceDB adds this)
            let distance_col = batch.column_by_name("_distance");
            let distances: Vec<f32> = if let Some(col) = distance_col {
                col.as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|arr| arr.values().to_vec())
                    .unwrap_or_else(|| vec![0.0; num_rows])
            } else {
                vec![0.0; num_rows]
            };

            for i in 0..num_rows {
                let id = id_array.value(i).to_string();
                let content = content_array.value(i).to_string();
                let metadata_str = metadata_array.value(i);
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(metadata_str).unwrap_or_default();

                // Convert L2 distance to similarity score
                let distance = distances.get(i).copied().unwrap_or(0.0);
                let score = 1.0 / (1.0 + distance);

                // Apply minimum score filter
                if let Some(min_score) = params.min_score {
                    if score < min_score {
                        continue;
                    }
                }

                search_results.push(SearchResult {
                    record: VectorRecord {
                        id,
                        vector: vec![], // Don't return full vector for efficiency
                        content,
                        metadata,
                    },
                    score,
                });
            }
        }

        Ok(search_results)
    }

    async fn delete(&self, ids: Vec<String>) -> Result<usize> {
        let table_guard = self.table.read();
        let table = table_guard.as_ref().ok_or_else(|| infernum_core::Error::Internal {
            message: "Table not initialized".to_string(),
        })?;

        if ids.is_empty() {
            return Ok(0);
        }

        let filter = format!(
            "id IN ({})",
            ids.iter()
                .map(|id| format!("'{}'", id))
                .collect::<Vec<_>>()
                .join(", ")
        );

        table
            .delete(&filter)
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to delete records: {}", e),
            })?;

        Ok(ids.len())
    }

    async fn get(&self, ids: Vec<String>) -> Result<Vec<VectorRecord>> {
        let table_guard = self.table.read();
        let table = table_guard.as_ref().ok_or_else(|| infernum_core::Error::Internal {
            message: "Table not initialized".to_string(),
        })?;

        if ids.is_empty() {
            return Ok(vec![]);
        }

        let filter = format!(
            "id IN ({})",
            ids.iter()
                .map(|id| format!("'{}'", id))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let results = table
            .query()
            .only_if(filter)
            .execute()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Query failed: {}", e),
            })?;

        let mut records = Vec::new();

        use futures::TryStreamExt;
        let batches: Vec<_> = results
            .try_collect()
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to collect results: {}", e),
            })?;

        for batch in batches {
            use arrow_array::{Array, cast::AsArray};

            let num_rows = batch.num_rows();

            let id_col = batch.column_by_name("id").ok_or_else(|| {
                infernum_core::Error::Internal {
                    message: "Missing id column".to_string(),
                }
            })?;
            let id_array = id_col.as_string::<i32>();

            let content_col = batch.column_by_name("content").ok_or_else(|| {
                infernum_core::Error::Internal {
                    message: "Missing content column".to_string(),
                }
            })?;
            let content_array = content_col.as_string::<i32>();

            let metadata_col = batch.column_by_name("metadata").ok_or_else(|| {
                infernum_core::Error::Internal {
                    message: "Missing metadata column".to_string(),
                }
            })?;
            let metadata_array = metadata_col.as_string::<i32>();

            for i in 0..num_rows {
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(metadata_array.value(i)).unwrap_or_default();

                records.push(VectorRecord {
                    id: id_array.value(i).to_string(),
                    vector: vec![], // Don't return full vector
                    content: content_array.value(i).to_string(),
                    metadata,
                });
            }
        }

        Ok(records)
    }

    async fn count(&self) -> Result<usize> {
        let table_guard = self.table.read();
        let table = table_guard.as_ref().ok_or_else(|| infernum_core::Error::Internal {
            message: "Table not initialized".to_string(),
        })?;

        let count = table
            .count_rows(None)
            .await
            .map_err(|e| infernum_core::Error::Internal {
                message: format!("Failed to count rows: {}", e),
            })?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_lance_store_basic() {
        let dir = tempdir().unwrap();
        let config = LanceConfig::new(dir.path().join("test_db"))
            .with_dimension(3)
            .with_table("test_vectors");

        let store = LanceStore::new(config).await.unwrap();

        // Insert records
        let records = vec![
            VectorRecord::new(vec![1.0, 0.0, 0.0], "document 1"),
            VectorRecord::new(vec![0.0, 1.0, 0.0], "document 2"),
            VectorRecord::new(vec![0.0, 0.0, 1.0], "document 3"),
        ];

        let count = store.upsert(records).await.unwrap();
        assert_eq!(count, 3);

        // Verify count
        let total = store.count().await.unwrap();
        assert_eq!(total, 3);

        // Search
        let results = store
            .search(&[1.0, 0.0, 0.0], SearchParams { top_k: 2, ..Default::default() })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].record.content.contains("document 1"));
    }
}
