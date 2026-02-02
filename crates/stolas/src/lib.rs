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
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod bm25;
pub mod chunker;
pub mod cross_encoder;
pub mod embedding;
pub mod jormungandr;
pub mod rag;
pub mod store;

pub use chunker::{Chunk, Chunker, ChunkingStrategy};
pub use embedding::{
    cosine_similarity, dot_product, euclidean_distance, BatchEmbedder, Embedder, EngineEmbedder,
    MockEmbedder, PoolingStrategy, SentenceEmbedder,
};
pub use jormungandr::{
    CollaborationMode, ConversionPhase, CorpusType, Evidentiality, ExperienceCheckpoint,
    FeatureGap, Frequency, Friction, FrictionCategory, GapPriority, Joy, JoyCategory, Pattern,
    ResearchReport, Severity, SigilKnowledgeBase,
};
pub use rag::{ContextItem, Document, RagPipeline, RetrievalConfig};
pub use store::{InMemoryStore, LanceStoreConfig, SearchParams, SearchResult, VectorRecord, VectorStore};
#[cfg(feature = "lance")]
pub use store::LanceStore;
pub use bm25::{BM25Config, BM25Index, BM25Result, HybridResult, HybridRetriever};
pub use cross_encoder::{
    CrossEncoder, CrossEncoderConfig, EmbeddingCrossEncoder, EnsembleReranker,
    HeuristicCrossEncoder, MockCrossEncoder, RerankResult, ScoredDocument,
};
