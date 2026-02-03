# RAG HTTP Endpoints Implementation Spec

**Status:** Ready for Implementation (2026-01-10 audit: Stolas exists, Observer client ready, server routes needed)
**Author:** Observer Agent
**Date:** 2025-12-27
**Target:** infernum-server

## Overview

The Observer UI has RAG panel support ready, but the infernum-server doesn't expose Stolas RAG functionality via HTTP. This spec defines the endpoints needed to enable RAG in the Observer UI.

## Prerequisites

The Stolas crate already exists at `infernum-framework/crates/stolas/` with:
- `rag.rs` - Main RAG pipeline
- `store.rs` - LanceDB vector store
- `chunker.rs` - Document chunking (semantic, fixed, sentence)
- `embedding.rs` - Embedding generation
- `bm25.rs` - BM25 sparse retrieval
- `rerank.rs` - Cross-encoder reranking

## Endpoints Required

### 1. GET /api/rag/status

Returns the current RAG index status.

**Response:**
```json
{
  "documentCount": 0,
  "chunkCount": 0,
  "embeddingModel": "sentence-transformers/all-MiniLM-L6-v2",
  "lastUpdated": "2025-12-27T10:30:00Z",
  "ready": true
}
```

**Fields:**
- `documentCount` - Number of indexed documents
- `chunkCount` - Total chunks across all documents
- `embeddingModel` - Model used for embeddings (null if not configured)
- `lastUpdated` - ISO timestamp of last index update (null if never)
- `ready` - Whether RAG is ready to accept queries

**Errors:**
- `503 Service Unavailable` - RAG not initialized

---

### 2. GET /api/rag/documents

Lists all indexed documents.

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_abc123",
      "name": "readme.md",
      "chunkCount": 15,
      "indexedAt": "2025-12-27T10:30:00Z",
      "metadata": {
        "type": "text/markdown",
        "size": 4096
      }
    }
  ]
}
```

**Fields per document:**
- `id` - Unique document identifier
- `name` - Original filename
- `chunkCount` - Number of chunks for this document
- `indexedAt` - ISO timestamp when indexed
- `metadata` - Optional key-value metadata

---

### 3. POST /api/rag/documents

Index a new document.

**Request:**
```json
{
  "name": "readme.md",
  "content": "# My Document\n\nThis is the content...",
  "metadata": {
    "type": "text/markdown",
    "size": 4096
  }
}
```

**Fields:**
- `name` (required) - Document name/filename
- `content` (required) - Full text content to index
- `metadata` (optional) - Key-value metadata to store

**Response:**
```json
{
  "id": "doc_abc123",
  "name": "readme.md",
  "chunkCount": 15,
  "indexedAt": "2025-12-27T10:30:00Z",
  "metadata": {
    "type": "text/markdown",
    "size": 4096
  }
}
```

**Processing:**
1. Generate unique document ID
2. Chunk content using configured chunker (semantic recommended)
3. Generate embeddings for each chunk
4. Store in LanceDB vector store
5. Return document metadata

**Errors:**
- `400 Bad Request` - Missing name or content
- `503 Service Unavailable` - RAG not initialized

---

### 4. DELETE /api/rag/documents/:id

Delete a document from the index.

**Response:**
```json
{
  "success": true
}
```

**Errors:**
- `404 Not Found` - Document not found
- `503 Service Unavailable` - RAG not initialized

---

### 5. POST /api/rag/search

Search the knowledge base.

**Request:**
```json
{
  "query": "How do I configure the server?",
  "topK": 5,
  "minScore": 0.5,
  "rerank": false
}
```

**Fields:**
- `query` (required) - Natural language search query
- `topK` (optional, default: 5) - Number of results to return
- `minScore` (optional, default: 0.0) - Minimum similarity threshold (0-1)
- `rerank` (optional, default: false) - Use cross-encoder reranking

**Response:**
```json
{
  "query": "How do I configure the server?",
  "results": [
    {
      "content": "To configure the server, edit the config.toml file...",
      "sourceId": "doc_abc123",
      "chunkIndex": 3,
      "score": 0.89,
      "metadata": {
        "type": "text/markdown"
      }
    }
  ],
  "context": "To configure the server, edit the config.toml file..."
}
```

**Fields per result:**
- `content` - Chunk text content
- `sourceId` - Parent document ID
- `chunkIndex` - Chunk position in document
- `score` - Similarity score (0-1, higher is better)
- `metadata` - Inherited document metadata

**Fields:**
- `query` - Echo of input query
- `results` - Array of matching chunks
- `context` - Concatenated top results for LLM context injection

**Processing:**
1. Generate query embedding
2. Vector similarity search in LanceDB
3. (Optional) BM25 hybrid search
4. (Optional) Cross-encoder reranking
5. Filter by minScore
6. Return top K results

**Errors:**
- `400 Bad Request` - Missing query
- `503 Service Unavailable` - RAG not initialized or no documents indexed

---

## Implementation Notes

### Router Integration

Add to `infernum-server/src/routes/mod.rs`:

```rust
use axum::{Router, routing::{get, post, delete}};

pub fn rag_routes() -> Router<AppState> {
    Router::new()
        .route("/api/rag/status", get(handlers::rag::status))
        .route("/api/rag/documents", get(handlers::rag::list_documents))
        .route("/api/rag/documents", post(handlers::rag::index_document))
        .route("/api/rag/documents/:id", delete(handlers::rag::delete_document))
        .route("/api/rag/search", post(handlers::rag::search))
}
```

### AppState Extension

Add Stolas RAG instance to AppState:

```rust
pub struct AppState {
    // ... existing fields
    pub rag: Option<Arc<stolas::Rag>>,
}
```

Initialize on startup if RAG is enabled:

```rust
let rag = if config.rag.enabled {
    Some(Arc::new(stolas::Rag::new(&config.rag)?))
} else {
    None
};
```

### Configuration

Add to server config:

```toml
[rag]
enabled = true
data_dir = "./data/rag"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_strategy = "semantic"  # semantic | fixed | sentence
chunk_size = 512
chunk_overlap = 50
```

### Error Handling

Use consistent error response format:

```json
{
  "error": {
    "code": "RAG_NOT_INITIALIZED",
    "message": "RAG subsystem is not initialized"
  }
}
```

### Persistence

LanceDB stores data at `{data_dir}/lancedb/`. Document metadata should be stored separately (SQLite recommended) or in a dedicated LanceDB table.

---

## Frontend Integration

The Observer UI (`observer/src/api/client.ts`) already has these endpoints defined:

```typescript
rag: {
  status: () => request<RagIndexStatus>('/api/rag/status'),
  documents: () => request<{ documents: RagDocument[] }>('/api/rag/documents'),
  index: (data) => request<RagDocument>('/api/rag/documents', { method: 'POST', body: data }),
  delete: (id) => request<{ success: boolean }>(`/api/rag/documents/${id}`, { method: 'DELETE' }),
  search: (query, options) => request<RagSearchResponse>('/api/rag/search', { method: 'POST', body: { query, ...options } }),
  isAvailable: async () => { /* checks /api/rag/status */ },
}
```

The `RagPanel` component will automatically detect when endpoints become available and switch from "Not Available" to the full UI.

---

## Testing

### Manual Testing

```bash
# Check status
curl http://localhost:8085/api/rag/status

# Index a document
curl -X POST http://localhost:8085/api/rag/documents \
  -H "Content-Type: application/json" \
  -d '{"name": "test.txt", "content": "Hello world"}'

# List documents
curl http://localhost:8085/api/rag/documents

# Search
curl -X POST http://localhost:8085/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "topK": 5}'

# Delete
curl -X DELETE http://localhost:8085/api/rag/documents/doc_abc123
```

### Integration Tests

Add tests in `infernum-server/tests/rag_test.rs` covering:
- Status endpoint returns correct counts
- Document indexing and retrieval
- Search returns relevant results
- Delete removes document and chunks
- Error handling for missing documents

---

## Open Questions

1. **Embedding model loading** - Should embeddings use the loaded LLM or a separate embedding model?
2. **Async indexing** - Should large documents be indexed asynchronously with status polling?
3. **Batch indexing** - Should there be a bulk upload endpoint?
4. **Document updates** - Should there be a PUT endpoint, or require delete + re-index?

---

## Acceptance Criteria

- [ ] All 5 endpoints implemented and returning correct response formats
- [ ] RAG state persists across server restarts
- [ ] Observer UI shows documents and allows search when backend is running
- [ ] Error responses are consistent and informative
- [ ] Basic integration tests pass
