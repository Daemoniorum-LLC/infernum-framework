//! Database schema definitions for Paimon LLM Studio.
//!
//! This module contains the SQL schema and migration logic.

/// Current schema version.
pub const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Complete schema SQL for version 1.
pub const SCHEMA_SQL: &str = r#"
-- Migration tracking table
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- =============================================================================
-- DATASETS
-- =============================================================================

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    format TEXT NOT NULL DEFAULT 'JsonLines',
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    validation_json TEXT,
    stats_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at);

CREATE TABLE IF NOT EXISTS dataset_examples (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    system TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    quality_score REAL,
    synthetic INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_examples_dataset ON dataset_examples(dataset_id);

-- =============================================================================
-- EXPERIMENTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    config_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON run_metrics(run_id, name);

-- =============================================================================
-- PROMPTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS prompt_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    active_version_id TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_templates_name ON prompt_templates(name);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL REFERENCES prompt_templates(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    system_prompt TEXT,
    message TEXT NOT NULL,
    author TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(template_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_template ON prompt_versions(template_id);

CREATE TABLE IF NOT EXISTS prompt_tests (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    template_id TEXT REFERENCES prompt_templates(id) ON DELETE SET NULL,
    version_a_id TEXT NOT NULL,
    version_b_id TEXT NOT NULL,
    version_a_results_json TEXT NOT NULL DEFAULT '{}',
    version_b_results_json TEXT NOT NULL DEFAULT '{}',
    winner TEXT,
    significance REAL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_prompt_tests_template ON prompt_tests(template_id);
CREATE INDEX IF NOT EXISTS idx_prompt_tests_status ON prompt_tests(status);

-- =============================================================================
-- MODEL REGISTRY
-- =============================================================================

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    base_model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    owner TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_task_type ON models(task_type);

CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'Development',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    artifact_path TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    experiment_run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    stage_history_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(model_id, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_model ON model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_stage ON model_versions(stage);

CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    model_version INTEGER NOT NULL,
    environment TEXT NOT NULL DEFAULT 'Development',
    status TEXT NOT NULL DEFAULT 'Pending',
    endpoint_url TEXT,
    resources_json TEXT NOT NULL DEFAULT '{}',
    history_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
"#;

/// Migrations from version N to N+1.
pub struct Migration {
    /// Target version after this migration.
    pub version: u32,
    /// Description of the migration.
    pub description: &'static str,
    /// SQL to execute.
    pub sql: &'static str,
}

/// All available migrations.
pub const MIGRATIONS: &[Migration] = &[Migration {
    version: 2,
    description: "Expand prompt and model registry tables",
    sql: r#"
-- This migration is for databases created with v1 schema.
-- New databases use the updated CREATE TABLE statements.

-- Prompts: Rename table and add columns
ALTER TABLE prompts RENAME TO prompt_templates;
ALTER TABLE prompt_templates ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE prompt_templates ADD COLUMN active_version_id TEXT;

-- Prompt versions: Add new columns
ALTER TABLE prompt_versions ADD COLUMN template_id TEXT;
UPDATE prompt_versions SET template_id = prompt_id;
ALTER TABLE prompt_versions ADD COLUMN system_prompt TEXT;
ALTER TABLE prompt_versions ADD COLUMN message TEXT NOT NULL DEFAULT '';
ALTER TABLE prompt_versions ADD COLUMN author TEXT;
ALTER TABLE prompt_versions ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE prompt_versions RENAME COLUMN version TO version_number;

-- Prompt tests: Add new columns
ALTER TABLE prompt_tests ADD COLUMN name TEXT NOT NULL DEFAULT 'unnamed';
ALTER TABLE prompt_tests ADD COLUMN template_id TEXT;
ALTER TABLE prompt_tests ADD COLUMN version_a_id TEXT;
ALTER TABLE prompt_tests ADD COLUMN version_b_id TEXT;
ALTER TABLE prompt_tests ADD COLUMN version_a_results_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE prompt_tests ADD COLUMN version_b_results_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE prompt_tests ADD COLUMN significance REAL;
ALTER TABLE prompt_tests RENAME COLUMN created_at TO started_at;

-- Models: Add new columns
ALTER TABLE models ADD COLUMN task_type TEXT NOT NULL DEFAULT 'text-generation';
ALTER TABLE models ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE models ADD COLUMN owner TEXT;

-- Model versions: Add new columns
ALTER TABLE model_versions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE model_versions ADD COLUMN artifact_path TEXT;
ALTER TABLE model_versions ADD COLUMN experiment_run_id TEXT;
ALTER TABLE model_versions ADD COLUMN dataset_id TEXT;
ALTER TABLE model_versions ADD COLUMN stage_history_json TEXT NOT NULL DEFAULT '[]';

-- Deployments table (new)
CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    model_version INTEGER NOT NULL,
    environment TEXT NOT NULL DEFAULT 'Development',
    status TEXT NOT NULL DEFAULT 'Pending',
    endpoint_url TEXT,
    resources_json TEXT NOT NULL DEFAULT '{}',
    history_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_id);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
"#,
}];

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_schema_is_valid_sql() {
        let conn = Connection::open_in_memory().expect("open in-memory db");

        // Execute schema SQL
        conn.execute_batch(SCHEMA_SQL)
            .expect("schema should be valid SQL");

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .expect("prepare")
            .query_map([], |row| row.get(0))
            .expect("query")
            .filter_map(|r| r.ok())
            .collect();

        assert!(tables.contains(&"datasets".to_string()));
        assert!(tables.contains(&"dataset_examples".to_string()));
        assert!(tables.contains(&"experiments".to_string()));
        assert!(tables.contains(&"runs".to_string()));
        assert!(tables.contains(&"run_metrics".to_string()));
        assert!(tables.contains(&"prompt_templates".to_string()));
        assert!(tables.contains(&"prompt_versions".to_string()));
        assert!(tables.contains(&"prompt_tests".to_string()));
        assert!(tables.contains(&"models".to_string()));
        assert!(tables.contains(&"model_versions".to_string()));
        assert!(tables.contains(&"deployments".to_string()));
        assert!(tables.contains(&"_migrations".to_string()));
    }

    #[test]
    fn test_schema_idempotent() {
        let conn = Connection::open_in_memory().expect("open in-memory db");

        // Execute twice should succeed (CREATE IF NOT EXISTS)
        conn.execute_batch(SCHEMA_SQL).expect("first execution");
        conn.execute_batch(SCHEMA_SQL).expect("second execution");
    }

    #[test]
    fn test_foreign_key_cascade() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable FK");
        conn.execute_batch(SCHEMA_SQL).expect("create schema");

        // Insert a dataset
        conn.execute(
            "INSERT INTO datasets (id, name, format, tags_json, created_at, updated_at, stats_json)
             VALUES ('ds1', 'test', 'JsonLines', '[]', '2025-01-01', '2025-01-01', '{}')",
            [],
        )
        .expect("insert dataset");

        // Insert an example
        conn.execute(
            "INSERT INTO dataset_examples (id, dataset_id, input, output, created_at)
             VALUES ('ex1', 'ds1', 'hello', 'world', '2025-01-01')",
            [],
        )
        .expect("insert example");

        // Delete dataset should cascade to examples
        conn.execute("DELETE FROM datasets WHERE id = 'ds1'", [])
            .expect("delete dataset");

        // Example should be gone
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM dataset_examples", [], |r| r.get(0))
            .expect("count examples");
        assert_eq!(count, 0);
    }
}
