//! # Stolas
//!
//! *"The Prince reveals hidden knowledge"*
//!
//! Stolas is the knowledge engine for the Infernum ecosystem,
//! providing vector storage, embedding generation, and RAG capabilities.
//!
//! ## Features
//!
//! - **Vector Storage**: Lance-based persistent vector storage
//! - **Embedding Generation**: Integration with embedding models
//! - **Chunking**: Intelligent document chunking strategies
//! - **Hybrid Search**: Combined dense and sparse retrieval

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod chunker;
pub mod embedding;
pub mod rag;
pub mod store;

pub use chunker::{Chunker, ChunkingStrategy};
pub use embedding::Embedder;
pub use rag::{RagPipeline, RetrievalConfig};
pub use store::{VectorStore, VectorRecord, SearchResult};
