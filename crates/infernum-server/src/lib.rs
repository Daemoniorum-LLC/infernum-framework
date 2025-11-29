//! # Infernum Server
//!
//! HTTP API server with OpenAI-compatible endpoints.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod handlers;
pub mod openai;
pub mod server;

pub use server::{Server, ServerConfig};
