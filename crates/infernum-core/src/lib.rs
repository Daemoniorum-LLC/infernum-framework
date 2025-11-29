//! # Infernum Core
//!
//! Core types and traits for the Infernum ecosystem.
//!
//! This crate provides the foundational abstractions used across all Infernum components:
//! - Common error types
//! - Request/response structures
//! - Model metadata and configuration
//! - Shared traits for inference, embedding, and streaming

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod error;
pub mod model;
pub mod request;
pub mod response;
pub mod sampling;
pub mod streaming;
pub mod types;

pub use error::{Error, Result};
pub use model::{ModelArchitecture, ModelId, ModelMetadata, ModelSource};
pub use request::{EmbedRequest, GenerateRequest, PromptInput};
pub use response::{EmbedResponse, GenerateResponse, TokenInfo};
pub use sampling::SamplingParams;
pub use streaming::TokenStream;
pub use types::*;
