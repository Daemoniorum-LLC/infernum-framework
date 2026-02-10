//! RAG HTTP endpoints for Stolas integration.
//!
//! Provides HTTP API for document indexing and semantic search via the Stolas RAG pipeline.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use stolas::{
    ContextItem, Document, Embedder, InMemoryStore, MockEmbedder, RagPipeline, RetrievalConfig,
    VectorStore,
};

use crate::error_response::{api_error, ErrorCode};

/// RAG state for the server.
pub struct RagState {
    /// The RAG pipeline (lazy-initialized).
    pub pipeline: Option<RagPipeline>,
    /// Document metadata storage (id -> metadata).
    pub documents: HashMap<String, DocumentMeta>,
    /// Whether RAG is initialized.
    pub initialized: bool,
}

impl Default for RagState {
    fn default() -> Self {
        Self::new()
    }
}

impl RagState {
    /// Creates a new RAG state.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            documents: HashMap::new(),
            initialized: false,
        }
    }

    /// Initializes the RAG pipeline with default configuration.
    pub fn initialize(&mut self) {
        if self.initialized {
            return;
        }

        // Use MockEmbedder for now - in production, use EngineEmbedder with the loaded model
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(384));
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryStore::new());
        let config = RetrievalConfig::default();

        self.pipeline = Some(RagPipeline::new(embedder, store, config));
        self.initialized = true;
    }

    /// Returns the pipeline if initialized.
    pub fn pipeline(&self) -> Option<&RagPipeline> {
        self.pipeline.as_ref()
    }
}

/// Document metadata stored in server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMeta {
    /// Document ID.
    pub id: String,
    /// Document name/filename.
    pub name: String,
    /// Number of chunks indexed.
    pub chunk_count: usize,
    /// Timestamp when indexed (Unix ms).
    pub indexed_at: u64,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// RAG health/status response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RagHealthResponse {
    /// Number of indexed documents.
    pub document_count: usize,
    /// Total chunks across all documents.
    pub chunk_count: usize,
    /// Embedding model in use.
    pub embedding_model: Option<String>,
    /// Last update timestamp (ISO 8601).
    pub last_updated: Option<String>,
    /// Whether RAG is ready.
    pub initialized: bool,
}

/// Document index request.
#[derive(Debug, Deserialize)]
pub struct IndexDocumentRequest {
    /// Document name/filename.
    pub name: String,
    /// Document content to index.
    pub content: String,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Document list response.
#[derive(Debug, Serialize)]
pub struct DocumentListResponse {
    /// List of documents.
    pub documents: Vec<DocumentMeta>,
}

/// Document count response.
#[derive(Debug, Serialize)]
pub struct DocumentCountResponse {
    /// Number of documents.
    pub count: usize,
}

/// Search request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    /// Search query.
    pub query: String,
    /// Number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum similarity score.
    #[serde(default)]
    pub min_score: Option<f32>,
    /// Enable cross-encoder reranking.
    #[serde(default)]
    pub rerank: bool,
}

fn default_top_k() -> usize {
    5
}

/// Search result item.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultItem {
    /// Content text.
    pub content: String,
    /// Source document ID.
    pub source_id: String,
    /// Chunk index in source document.
    pub chunk_index: usize,
    /// Similarity score (0-1).
    pub score: f32,
    /// Document metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl From<ContextItem> for SearchResultItem {
    fn from(item: ContextItem) -> Self {
        Self {
            content: item.content,
            source_id: item.source_id,
            chunk_index: item.chunk_index,
            score: item.score,
            metadata: item.metadata,
        }
    }
}

/// Search response.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Search results.
    pub results: Vec<SearchResultItem>,
    /// Total results found.
    pub total: usize,
}

/// Delete response.
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    /// Number of chunks deleted.
    pub deleted: usize,
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /api/rag/health - Get RAG status.
pub async fn rag_health(State(rag): State<Arc<RwLock<RagState>>>) -> impl IntoResponse {
    let state = rag.read().await;

    let (document_count, chunk_count) = if state.initialized {
        (
            state.documents.len(),
            state.documents.values().map(|d| d.chunk_count).sum(),
        )
    } else {
        (0, 0)
    };

    let last_updated = state
        .documents
        .values()
        .map(|d| d.indexed_at)
        .max()
        .map(|ts| {
            chrono::DateTime::from_timestamp_millis(ts as i64)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()
        });

    Json(RagHealthResponse {
        document_count,
        chunk_count,
        embedding_model: if state.initialized {
            Some("mock-embedder-384".to_string())
        } else {
            None
        },
        last_updated,
        initialized: state.initialized,
    })
}

/// GET /api/rag/documents - List indexed documents.
pub async fn list_documents(State(rag): State<Arc<RwLock<RagState>>>) -> impl IntoResponse {
    let state = rag.read().await;

    if !state.initialized {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(api_error(
                ErrorCode::ServiceUnavailable,
                "RAG not initialized",
            )),
        )
            .into_response();
    }

    let documents: Vec<DocumentMeta> = state.documents.values().cloned().collect();
    Json(DocumentListResponse { documents }).into_response()
}

/// GET /api/rag/documents/count - Get document count.
pub async fn document_count(State(rag): State<Arc<RwLock<RagState>>>) -> impl IntoResponse {
    let state = rag.read().await;
    Json(DocumentCountResponse {
        count: state.documents.len(),
    })
}

/// POST /api/rag/documents - Index a new document.
pub async fn index_document(
    State(rag): State<Arc<RwLock<RagState>>>,
    Json(req): Json<IndexDocumentRequest>,
) -> impl IntoResponse {
    // Validate request
    if req.name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(api_error(
                ErrorCode::InvalidRequest,
                "Document name is required",
            )),
        )
            .into_response();
    }

    if req.content.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(api_error(
                ErrorCode::InvalidRequest,
                "Document content is required",
            )),
        )
            .into_response();
    }

    let mut state = rag.write().await;

    // Auto-initialize if not already
    if !state.initialized {
        state.initialize();
    }

    let pipeline = match state.pipeline.as_ref() {
        Some(p) => p,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(api_error(
                    ErrorCode::ServiceUnavailable,
                    "RAG pipeline not available",
                )),
            )
                .into_response();
        },
    };

    // Generate document ID
    let doc_id = format!("doc_{}", uuid::Uuid::new_v4().simple());

    // Create document
    let mut doc = Document::new(&doc_id, &req.content);
    for (key, value) in &req.metadata {
        doc = doc.with_metadata(key, value.clone());
    }
    doc = doc.with_metadata("name", serde_json::json!(req.name));

    // Ingest document
    let chunk_count = match pipeline.ingest(doc).await {
        Ok(count) => count,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(api_error(
                    ErrorCode::InternalError,
                    &format!("Failed to index document: {}", e),
                )),
            )
                .into_response();
        },
    };

    // Store document metadata
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let meta = DocumentMeta {
        id: doc_id.clone(),
        name: req.name,
        chunk_count,
        indexed_at: now,
        metadata: req.metadata,
    };

    state.documents.insert(doc_id, meta.clone());

    (StatusCode::CREATED, Json(meta)).into_response()
}

/// DELETE /api/rag/documents/:id - Delete a document.
pub async fn delete_document(
    State(rag): State<Arc<RwLock<RagState>>>,
    Path(doc_id): Path<String>,
) -> impl IntoResponse {
    let mut state = rag.write().await;

    if !state.initialized {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(api_error(
                ErrorCode::ServiceUnavailable,
                "RAG not initialized",
            )),
        )
            .into_response();
    }

    // Check if document exists
    let meta = match state.documents.remove(&doc_id) {
        Some(m) => m,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(api_error(ErrorCode::NotFound, "Document not found")),
            )
                .into_response();
        },
    };

    // Note: InMemoryStore doesn't support deletion by ID yet, so we just remove metadata
    // In production with LanceDB, we would delete the vectors too

    Json(DeleteResponse {
        deleted: meta.chunk_count,
    })
    .into_response()
}

/// POST /api/rag/search - Search the knowledge base.
pub async fn search(
    State(rag): State<Arc<RwLock<RagState>>>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    if req.query.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(api_error(ErrorCode::InvalidRequest, "Query is required")),
        )
            .into_response();
    }

    let state = rag.read().await;

    if !state.initialized {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(api_error(
                ErrorCode::ServiceUnavailable,
                "RAG not initialized",
            )),
        )
            .into_response();
    }

    let pipeline = match state.pipeline.as_ref() {
        Some(p) => p,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(api_error(
                    ErrorCode::ServiceUnavailable,
                    "RAG pipeline not available",
                )),
            )
                .into_response();
        },
    };

    // Retrieve results
    let results = match pipeline.retrieve(&req.query).await {
        Ok(items) => items,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(api_error(
                    ErrorCode::InternalError,
                    &format!("Search failed: {}", e),
                )),
            )
                .into_response();
        },
    };

    // Convert to response format
    let results: Vec<SearchResultItem> = results
        .into_iter()
        .take(req.top_k)
        .filter(|r| req.min_score.map(|m| r.score >= m).unwrap_or(true))
        .map(SearchResultItem::from)
        .collect();

    let total = results.len();

    Json(SearchResponse { results, total }).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_state_new() {
        let state = RagState::new();
        assert!(!state.initialized);
        assert!(state.pipeline.is_none());
        assert!(state.documents.is_empty());
    }

    #[test]
    fn test_rag_state_initialize() {
        let mut state = RagState::new();
        state.initialize();
        assert!(state.initialized);
        assert!(state.pipeline.is_some());
    }

    #[test]
    fn test_rag_state_double_initialize() {
        let mut state = RagState::new();
        state.initialize();
        state.initialize(); // Should not panic
        assert!(state.initialized);
    }

    #[test]
    fn test_document_meta_serialization() {
        let meta = DocumentMeta {
            id: "doc_123".to_string(),
            name: "test.txt".to_string(),
            chunk_count: 5,
            indexed_at: 1234567890000,
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("doc_123"));
        assert!(json.contains("test.txt"));
    }

    #[test]
    fn test_search_request_defaults() {
        let json = r#"{"query": "test"}"#;
        let req: SearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "test");
        assert_eq!(req.top_k, 5);
        assert!(!req.rerank);
    }

    #[test]
    fn test_context_item_to_search_result() {
        let item = ContextItem {
            content: "test content".to_string(),
            source_id: "doc_1".to_string(),
            chunk_index: 2,
            score: 0.85,
            metadata: HashMap::new(),
        };
        let result: SearchResultItem = item.into();
        assert_eq!(result.content, "test content");
        assert_eq!(result.source_id, "doc_1");
        assert_eq!(result.chunk_index, 2);
        assert!((result.score - 0.85).abs() < 0.001);
    }
}
