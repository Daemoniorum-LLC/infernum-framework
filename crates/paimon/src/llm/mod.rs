//! LLM Client abstraction for Paimon agents.
//!
//! Provides a unified interface for interacting with language models,
//! whether local (Infernum) or cloud-based (Anthropic, OpenAI).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    LlmClientRegistry                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Manages multiple providers, routes requests                │
//! └─────────────────────────────────────────────────────────────┘
//!            │
//!            ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      LlmClient trait                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  - generate(request) -> Response                            │
//! │  - stream(request) -> Stream<Chunk>                         │
//! └─────────────────────────────────────────────────────────────┘
//!            │
//!            ├──────────────┬──────────────┬──────────────┐
//!            ▼              ▼              ▼              ▼
//!    InfernumClient   AnthropicClient  OpenAIClient  MockClient
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use paimon::llm::{LlmClient, GenerateRequest, Message};
//!
//! let client = registry.get("infernum")?;
//!
//! let response = client.generate(GenerateRequest {
//!     messages: vec![Message::user("Hello!")],
//!     ..Default::default()
//! }).await?;
//!
//! println!("Response: {}", response.content);
//! ```

mod client;
mod mock;
mod registry;

pub use client::{
    LlmClient, GenerateRequest, GenerateResponse, StreamChunk,
    Message, MessageRole, GenerateOptions,
};
pub use mock::MockLlmClient;
pub use registry::LlmClientRegistry;
