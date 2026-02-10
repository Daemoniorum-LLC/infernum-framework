//! SQLite persistence layer for Paimon LLM Studio.
//!
//! Provides durable storage for datasets, experiments, prompts, and models
//! with support for transactions, migrations, and concurrent access.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              StudioDatabase                      │
//! ├─────────────────────────────────────────────────┤
//! │  - Connection pooling (single writer)           │
//! │  - Transaction support                          │
//! │  - Migration management                         │
//! └─────────────────────────────────────────────────┘
//!            │
//!            ▼
//! ┌─────────────────────────────────────────────────┐
//! │              SQLite (rusqlite)                   │
//! ├─────────────────────────────────────────────────┤
//! │  Tables:                                         │
//! │  - datasets, dataset_examples                   │
//! │  - experiments, runs, run_metrics               │
//! │  - prompts, prompt_versions                     │
//! │  - models, model_versions                       │
//! │  - _migrations (schema versioning)              │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use paimon::persistence::{StudioDatabase, DatabaseConfig};
//!
//! // Create or open database
//! let db = StudioDatabase::new("studio.db").await?;
//!
//! // Use transactions for atomic operations
//! db.transaction(|tx| {
//!     tx.insert_dataset(&dataset)?;
//!     tx.insert_examples(&dataset.id, &examples)?;
//!     Ok(())
//! }).await?;
//! ```

mod database;
mod error;
mod schema;

pub use database::{DatabaseConfig, StudioDatabase, Transaction};
pub use error::{PersistenceError, Result};
pub use schema::{CURRENT_SCHEMA_VERSION, SCHEMA_SQL};
