//! # Infernum Server
//!
//! HTTP API server with OpenAI-compatible endpoints.
//!
//! ## Features
//!
//! - **OpenAI API Compatibility**: Drop-in replacement for OpenAI's API
//! - **Chat Completions**: `/v1/chat/completions` with streaming support
//! - **Text Completions**: `/v1/completions` for raw text generation
//! - **Embeddings**: `/v1/embeddings` for vector generation
//! - **Model Management**: Load/unload models at runtime
//! - **Health Checks**: `/health` and `/ready` endpoints
//!
//! ## Example
//!
//! ```ignore
//! use infernum_server::{Server, ServerConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ServerConfig::builder()
//!         .addr("0.0.0.0:8080".parse()?)
//!         .model("meta-llama/Llama-3.2-3B-Instruct")
//!         .build();
//!
//!     let server = Server::new(config);
//!     server.run().await?;
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod handlers;
pub mod openai;
pub mod server;

pub use openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
    CompletionRequest, CompletionResponse, EmbeddingData, EmbeddingInput, EmbeddingRequest,
    EmbeddingResponse, ModelObject, ModelsResponse, Usage,
};
pub use server::{AppState, Server, ServerConfig, ServerConfigBuilder};
