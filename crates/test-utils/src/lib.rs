//! # Test Utilities for Infernum
//!
//! This crate provides testing utilities shared across Infernum crates:
//!
//! - Mock implementations of core traits
//! - Test fixtures and sample data
//! - Custom assertions for common patterns
//! - Test server helpers for integration tests
//!
//! ## Usage
//!
//! Add to your `Cargo.toml` dev-dependencies:
//!
//! ```toml
//! [dev-dependencies]
//! test-utils = { path = "../test-utils" }
//! ```
//!
//! ## Modules
//!
//! - [`mock`]: Mock implementations (MockInferenceEngine, MockVectorStore)
//! - [`fixtures`]: Sample requests, responses, and test data
//! - [`assertions`]: Custom assertion macros and helpers
//! - [`server`]: Test server setup for integration tests

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(clippy::unwrap_used)]

pub mod assertions;
pub mod fixtures;
pub mod mock;
pub mod server;

pub use assertions::*;
pub use fixtures::*;
pub use mock::*;
pub use server::*;

/// Re-export commonly used test utilities
pub mod prelude {
    pub use super::assertions::*;
    pub use super::fixtures::*;
    pub use super::mock::*;
    pub use super::server::*;
}
