//! Core database implementation for Paimon LLM Studio.
//!
//! Provides a thread-safe SQLite database wrapper with transaction support
//! and automatic schema management.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rusqlite::{Connection, params};
use tracing::{info, debug, warn, info_span};

use super::error::{PersistenceError, Result};
use super::schema::{CURRENT_SCHEMA_VERSION, SCHEMA_SQL, MIGRATIONS};
use crate::dataset::{Dataset, Example, DatasetFormat, DatasetStats, ValidationReport};
use crate::experiment::{Experiment, Run, RunStatus};
use crate::prompt::{PromptTemplate, PromptVersion, VersionMetrics, TestResult, TestStatus, TestVersionResults};
use crate::registry::{Model, ModelVersion, ModelMetadata, ModelStage, StageTransition, Deployment, DeploymentEnvironment, DeploymentStatus, DeploymentResources, DeploymentEvent};

/// Configuration for the studio database.
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Path to the database file.
    pub path: PathBuf,

    /// Enable WAL mode for better concurrent read performance.
    pub wal_mode: bool,

    /// Enable foreign key constraints.
    pub foreign_keys: bool,

    /// Busy timeout in milliseconds.
    pub busy_timeout_ms: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("studio.db"),
            wal_mode: true,
            foreign_keys: true,
            busy_timeout_ms: 5000,
        }
    }
}

impl DatabaseConfig {
    /// Creates a config for an in-memory database.
    pub fn in_memory() -> Self {
        Self {
            path: PathBuf::from(":memory:"),
            ..Default::default()
        }
    }

    /// Creates a config with the given path.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }
}

/// A transaction handle for atomic operations.
pub struct Transaction<'a> {
    conn: &'a Connection,
}

impl<'a> Transaction<'a> {
    fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Inserts a dataset into the database.
    pub fn insert_dataset(&self, dataset: &Dataset) -> Result<()> {
        let tags_json = serde_json::to_string(&dataset.tags)?;
        let stats_json = serde_json::to_string(&dataset.stats)?;
        let validation_json = dataset.validation.as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let format_str = format!("{:?}", dataset.format);

        self.conn.execute(
            "INSERT INTO datasets (id, name, description, format, tags_json, created_at, updated_at, validation_json, stats_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &dataset.id,
                &dataset.name,
                &dataset.description,
                &format_str,
                &tags_json,
                &dataset.created_at.to_rfc3339(),
                &dataset.updated_at.to_rfc3339(),
                &validation_json,
                &stats_json,
            ],
        )?;

        Ok(())
    }

    /// Inserts examples for a dataset.
    pub fn insert_examples(&self, dataset_id: &str, examples: &[Example]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO dataset_examples (id, dataset_id, input, output, system, metadata_json, quality_score, synthetic, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )?;

        for example in examples {
            let metadata_json = serde_json::to_string(&example.metadata)?;
            stmt.execute(params![
                &example.id,
                dataset_id,
                &example.input,
                &example.output,
                &example.system,
                &metadata_json,
                &example.quality_score,
                example.synthetic as i32,
                chrono::Utc::now().to_rfc3339(),
            ])?;
        }

        Ok(())
    }

    /// Deletes a dataset and its examples (via cascade).
    pub fn delete_dataset(&self, id: &str) -> Result<bool> {
        let count = self.conn.execute("DELETE FROM datasets WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Inserts an experiment.
    pub fn insert_experiment(&self, experiment: &Experiment) -> Result<()> {
        let config_json = serde_json::to_string(&experiment.config)?;

        self.conn.execute(
            "INSERT INTO experiments (id, name, description, dataset_id, config_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &experiment.id,
                &experiment.config.name,
                &experiment.config.description,
                &experiment.config.dataset_id,
                &config_json,
                &experiment.created_at.to_rfc3339(),
                &experiment.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Inserts a run.
    pub fn insert_run(&self, run: &Run, experiment_id: &str) -> Result<()> {
        let hyperparams_json = serde_json::to_string(&run.hyperparameters)?;
        let status_str = format!("{:?}", run.status);

        self.conn.execute(
            "INSERT INTO runs (id, experiment_id, status, config_json, started_at, completed_at, error_message, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                &run.id,
                experiment_id,
                &status_str,
                &hyperparams_json,
                &run.started_at.to_rfc3339(),
                run.ended_at.map(|t| t.to_rfc3339()),
                &run.error_message,
                &run.started_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Updates run status.
    pub fn update_run_status(&self, run_id: &str, status: RunStatus, error: Option<&str>) -> Result<bool> {
        let status_str = format!("{:?}", status);
        let completed_at = if matches!(status, RunStatus::Completed | RunStatus::Failed) {
            Some(chrono::Utc::now().to_rfc3339())
        } else {
            None
        };

        let count = self.conn.execute(
            "UPDATE runs SET status = ?1, completed_at = ?2, error_message = ?3 WHERE id = ?4",
            params![&status_str, &completed_at, &error, run_id],
        )?;

        Ok(count > 0)
    }

    /// Logs metrics for a run.
    pub fn log_metrics(&self, run_id: &str, metrics: &[(&str, f64)]) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "INSERT INTO run_metrics (run_id, name, value, timestamp)
             VALUES (?1, ?2, ?3, ?4)"
        )?;

        let now = chrono::Utc::now().to_rfc3339();
        for (name, value) in metrics {
            stmt.execute(params![run_id, name, value, &now])?;
        }

        Ok(())
    }

    // =========================================================================
    // Prompt Template Operations
    // =========================================================================

    /// Inserts a prompt template.
    pub fn insert_prompt_template(&self, template: &PromptTemplate) -> Result<()> {
        let tags_json = serde_json::to_string(&template.tags)?;

        self.conn.execute(
            "INSERT INTO prompt_templates (id, name, description, active_version_id, tags_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &template.id,
                &template.name,
                &template.description,
                &template.active_version_id,
                &tags_json,
                &template.created_at.to_rfc3339(),
                &template.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Inserts a prompt version.
    pub fn insert_prompt_version(&self, template_id: &str, version: &PromptVersion) -> Result<()> {
        let metrics_json = serde_json::to_string(&version.metrics)?;

        self.conn.execute(
            "INSERT INTO prompt_versions (id, template_id, version_number, content, system_prompt, message, author, metrics_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &version.id,
                template_id,
                version.version_number,
                &version.content,
                &version.system_prompt,
                &version.message,
                &version.author,
                &metrics_json,
                &version.created_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Updates a prompt template's active version.
    pub fn update_prompt_active_version(&self, template_id: &str, version_id: &str) -> Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let count = self.conn.execute(
            "UPDATE prompt_templates SET active_version_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![version_id, &now, template_id],
        )?;
        Ok(count > 0)
    }

    /// Deletes a prompt template.
    pub fn delete_prompt_template(&self, id: &str) -> Result<bool> {
        let count = self.conn.execute("DELETE FROM prompt_templates WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Inserts a prompt A/B test.
    pub fn insert_prompt_test(&self, template_id: Option<&str>, test: &TestResult) -> Result<()> {
        let version_a_results_json = serde_json::to_string(&test.version_a_results)?;
        let version_b_results_json = serde_json::to_string(&test.version_b_results)?;
        let status_str = match test.status {
            TestStatus::Running => "running",
            TestStatus::Completed => "completed",
            TestStatus::Cancelled => "cancelled",
            TestStatus::Inconclusive => "inconclusive",
        };

        self.conn.execute(
            "INSERT INTO prompt_tests (id, name, template_id, version_a_id, version_b_id, version_a_results_json, version_b_results_json, winner, significance, status, started_at, ended_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                &test.test_id,
                &test.name,
                template_id,
                &test.version_a_id,
                &test.version_b_id,
                &version_a_results_json,
                &version_b_results_json,
                &test.winner,
                test.significance,
                status_str,
                &test.started_at.to_rfc3339(),
                test.ended_at.map(|t| t.to_rfc3339()),
            ],
        )?;

        Ok(())
    }

    /// Updates a prompt A/B test.
    pub fn update_prompt_test(&self, test: &TestResult) -> Result<bool> {
        let version_a_results_json = serde_json::to_string(&test.version_a_results)?;
        let version_b_results_json = serde_json::to_string(&test.version_b_results)?;
        let status_str = match test.status {
            TestStatus::Running => "running",
            TestStatus::Completed => "completed",
            TestStatus::Cancelled => "cancelled",
            TestStatus::Inconclusive => "inconclusive",
        };

        let count = self.conn.execute(
            "UPDATE prompt_tests SET version_a_results_json = ?1, version_b_results_json = ?2, winner = ?3, significance = ?4, status = ?5, ended_at = ?6 WHERE id = ?7",
            params![
                &version_a_results_json,
                &version_b_results_json,
                &test.winner,
                test.significance,
                status_str,
                test.ended_at.map(|t| t.to_rfc3339()),
                &test.test_id,
            ],
        )?;

        Ok(count > 0)
    }

    // =========================================================================
    // Model Registry Operations
    // =========================================================================

    /// Inserts a model.
    pub fn insert_model(&self, model: &Model) -> Result<()> {
        let tags_json = serde_json::to_string(&model.tags)?;

        self.conn.execute(
            "INSERT INTO models (id, name, description, base_model, task_type, tags_json, owner, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &model.id,
                &model.name,
                &model.description,
                &model.base_model,
                &model.task_type,
                &tags_json,
                &model.owner,
                &model.created_at.to_rfc3339(),
                &model.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Inserts a model version.
    pub fn insert_model_version(&self, model_id: &str, version: &ModelVersion) -> Result<()> {
        let metadata_json = serde_json::to_string(&version.metadata)?;
        let metrics_json = serde_json::to_string(&version.metrics)?;
        let stage_history_json = serde_json::to_string(&version.stage_history)?;
        let stage_str = version.stage.as_str();

        self.conn.execute(
            "INSERT INTO model_versions (id, model_id, version, stage, metadata_json, artifact_path, metrics_json, experiment_run_id, dataset_id, stage_history_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &version.id,
                model_id,
                version.version,
                stage_str,
                &metadata_json,
                version.artifact_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                &metrics_json,
                &version.experiment_run_id,
                &version.dataset_id,
                &stage_history_json,
                &version.created_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Updates a model version's stage.
    pub fn update_model_version_stage(&self, version_id: &str, stage: ModelStage, stage_history: &[StageTransition]) -> Result<bool> {
        let stage_str = stage.as_str();
        let stage_history_json = serde_json::to_string(stage_history)?;

        let count = self.conn.execute(
            "UPDATE model_versions SET stage = ?1, stage_history_json = ?2 WHERE id = ?3",
            params![stage_str, &stage_history_json, version_id],
        )?;

        Ok(count > 0)
    }

    /// Updates a model version's metrics.
    pub fn update_model_version_metrics(&self, version_id: &str, metrics: &std::collections::HashMap<String, f64>) -> Result<bool> {
        let metrics_json = serde_json::to_string(metrics)?;

        let count = self.conn.execute(
            "UPDATE model_versions SET metrics_json = ?1 WHERE id = ?2",
            params![&metrics_json, version_id],
        )?;

        Ok(count > 0)
    }

    /// Deletes a model.
    pub fn delete_model(&self, id: &str) -> Result<bool> {
        let count = self.conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Inserts a deployment.
    pub fn insert_deployment(&self, deployment: &Deployment) -> Result<()> {
        let resources_json = serde_json::to_string(&deployment.resources)?;
        let history_json = serde_json::to_string(&deployment.history)?;
        let environment_str = match deployment.environment {
            DeploymentEnvironment::Development => "Development",
            DeploymentEnvironment::Staging => "Staging",
            DeploymentEnvironment::Production => "Production",
        };
        let status_str = match deployment.status {
            DeploymentStatus::Pending => "Pending",
            DeploymentStatus::Deploying => "Deploying",
            DeploymentStatus::Running => "Running",
            DeploymentStatus::Failed => "Failed",
            DeploymentStatus::Stopped => "Stopped",
        };

        self.conn.execute(
            "INSERT INTO deployments (id, name, model_id, model_version, environment, status, endpoint_url, resources_json, history_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &deployment.id,
                &deployment.name,
                &deployment.model_id,
                deployment.model_version,
                environment_str,
                status_str,
                &deployment.endpoint_url,
                &resources_json,
                &history_json,
                &deployment.created_at.to_rfc3339(),
                &deployment.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Updates a deployment.
    pub fn update_deployment(&self, deployment: &Deployment) -> Result<bool> {
        let resources_json = serde_json::to_string(&deployment.resources)?;
        let history_json = serde_json::to_string(&deployment.history)?;
        let status_str = match deployment.status {
            DeploymentStatus::Pending => "Pending",
            DeploymentStatus::Deploying => "Deploying",
            DeploymentStatus::Running => "Running",
            DeploymentStatus::Failed => "Failed",
            DeploymentStatus::Stopped => "Stopped",
        };

        let count = self.conn.execute(
            "UPDATE deployments SET model_version = ?1, status = ?2, endpoint_url = ?3, resources_json = ?4, history_json = ?5, updated_at = ?6 WHERE id = ?7",
            params![
                deployment.model_version,
                status_str,
                &deployment.endpoint_url,
                &resources_json,
                &history_json,
                &deployment.updated_at.to_rfc3339(),
                &deployment.id,
            ],
        )?;

        Ok(count > 0)
    }

    /// Deletes a deployment.
    pub fn delete_deployment(&self, id: &str) -> Result<bool> {
        let count = self.conn.execute("DELETE FROM deployments WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }
}

/// The main database interface for Paimon LLM Studio.
///
/// Thread-safe wrapper around SQLite with automatic schema management.
pub struct StudioDatabase {
    conn: Arc<Mutex<Connection>>,
    config: DatabaseConfig,
}

impl StudioDatabase {
    /// Creates a new database connection.
    ///
    /// If the database doesn't exist, it will be created with the current schema.
    /// If it exists, migrations will be applied if needed.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let config = DatabaseConfig::with_path(path.as_ref());
        Self::with_config(config)
    }

    /// Creates a new database with the given configuration.
    pub fn with_config(config: DatabaseConfig) -> Result<Self> {
        let _span = info_span!("database.init", path = %config.path.display()).entered();

        let conn = if config.path.to_string_lossy() == ":memory:" {
            Connection::open_in_memory()
        } else {
            // Ensure parent directory exists
            if let Some(parent) = config.path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Connection::open(&config.path)
        }.map_err(|e| PersistenceError::Connection(e.to_string()))?;

        // Configure connection
        if config.foreign_keys {
            conn.execute("PRAGMA foreign_keys = ON;", [])?;
        }

        if config.wal_mode && config.path.to_string_lossy() != ":memory:" {
            // journal_mode returns result, use query_row instead of execute
            let _: String = conn.query_row("PRAGMA journal_mode = WAL;", [], |row| row.get(0))
                .unwrap_or_else(|_| "wal".to_string());
        }

        conn.busy_timeout(std::time::Duration::from_millis(config.busy_timeout_ms as u64))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        };

        // Initialize schema
        db.init_schema()?;

        info!("Database initialized successfully");
        Ok(db)
    }

    /// Creates an in-memory database for testing.
    pub fn in_memory() -> Result<Self> {
        Self::with_config(DatabaseConfig::in_memory())
    }

    /// Initializes the database schema.
    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock();

        // Check if database is new
        let has_tables: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='_migrations'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if !has_tables {
            debug!("Creating initial schema");
            conn.execute_batch(SCHEMA_SQL)?;

            // Record initial migration
            conn.execute(
                "INSERT INTO _migrations (version, description) VALUES (?1, ?2)",
                params![CURRENT_SCHEMA_VERSION, "Initial schema"],
            )?;
        } else {
            // Run pending migrations
            self.run_migrations_internal(&conn)?;
        }

        Ok(())
    }

    /// Runs pending migrations.
    fn run_migrations_internal(&self, conn: &Connection) -> Result<()> {
        let current_version: u32 = conn
            .query_row("SELECT MAX(version) FROM _migrations", [], |row| row.get(0))
            .unwrap_or(0);

        if current_version >= CURRENT_SCHEMA_VERSION {
            return Ok(());
        }

        info!(from = current_version, to = CURRENT_SCHEMA_VERSION, "Running migrations");

        for migration in MIGRATIONS {
            if migration.version > current_version {
                debug!(version = migration.version, desc = migration.description, "Applying migration");
                conn.execute_batch(migration.sql)?;
                conn.execute(
                    "INSERT INTO _migrations (version, description) VALUES (?1, ?2)",
                    params![migration.version, migration.description],
                )?;
            }
        }

        Ok(())
    }

    /// Returns the current schema version.
    pub fn schema_version(&self) -> Result<u32> {
        let conn = self.conn.lock();
        let version: u32 = conn
            .query_row("SELECT MAX(version) FROM _migrations", [], |row| row.get(0))
            .unwrap_or(0);
        Ok(version)
    }

    /// Returns the database configuration.
    #[must_use]
    pub fn config(&self) -> &DatabaseConfig {
        &self.config
    }

    /// Returns the database file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.config.path
    }

    /// Returns whether the database is in-memory.
    #[must_use]
    pub fn is_in_memory(&self) -> bool {
        self.config.path.to_string_lossy() == ":memory:"
    }

    /// Checks if a table exists.
    pub fn table_exists(&self, table_name: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name=?1",
                params![table_name],
                |row| row.get(0),
            )
            .unwrap_or(false);
        Ok(exists)
    }

    /// Executes operations within a transaction.
    ///
    /// If the closure returns an error, the transaction is rolled back.
    /// If it succeeds, the transaction is committed.
    pub fn transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> Result<T>,
    {
        let conn = self.conn.lock();
        conn.execute("BEGIN TRANSACTION", [])?;

        let tx = Transaction::new(&conn);
        match f(&tx) {
            Ok(result) => {
                conn.execute("COMMIT", [])?;
                Ok(result)
            }
            Err(e) => {
                if let Err(rollback_err) = conn.execute("ROLLBACK", []) {
                    warn!(error = %rollback_err, "Failed to rollback transaction");
                }
                Err(e)
            }
        }
    }

    // =========================================================================
    // Dataset Operations
    // =========================================================================

    /// Saves a dataset and its examples.
    pub fn save_dataset(&self, dataset: &Dataset) -> Result<()> {
        self.transaction(|tx| {
            // Delete existing if updating
            tx.delete_dataset(&dataset.id)?;
            tx.insert_dataset(dataset)?;
            tx.insert_examples(&dataset.id, &dataset.examples)?;
            Ok(())
        })
    }

    /// Loads a dataset by ID.
    pub fn load_dataset(&self, id: &str) -> Result<Option<Dataset>> {
        let conn = self.conn.lock();

        let result = conn.query_row(
            "SELECT id, name, description, format, tags_json, created_at, updated_at, validation_json, stats_json
             FROM datasets WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, String>(8)?,
                ))
            },
        );

        let (id, name, description, format_str, tags_json, created_at, updated_at, validation_json, stats_json) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // Load examples
        let examples = self.load_examples_internal(&conn, &id)?;

        // Parse format
        let format = match format_str.as_str() {
            "JsonLines" => DatasetFormat::JsonLines,
            "JsonArray" => DatasetFormat::JsonArray,
            "Csv" => DatasetFormat::Csv,
            "Alpaca" => DatasetFormat::Alpaca,
            "ShareGpt" => DatasetFormat::ShareGpt,
            "OpenAI" => DatasetFormat::OpenAI,
            _ => DatasetFormat::JsonLines,
        };

        let tags: Vec<String> = serde_json::from_str(&tags_json)?;
        let stats: DatasetStats = serde_json::from_str(&stats_json)?;
        let validation: Option<ValidationReport> = validation_json
            .map(|j| serde_json::from_str(&j))
            .transpose()?;

        Ok(Some(Dataset {
            id,
            name,
            description,
            format,
            tags,
            examples,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            validation,
            stats,
        }))
    }

    fn load_examples_internal(&self, conn: &Connection, dataset_id: &str) -> Result<Vec<Example>> {
        let mut stmt = conn.prepare(
            "SELECT id, input, output, system, metadata_json, quality_score, synthetic
             FROM dataset_examples WHERE dataset_id = ?1"
        )?;

        let examples = stmt.query_map(params![dataset_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<f32>>(5)?,
                row.get::<_, i32>(6)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in examples {
            let (id, input, output, system, metadata_json, quality_score, synthetic) = row?;
            let metadata = serde_json::from_str(&metadata_json)?;

            result.push(Example {
                id,
                input,
                output,
                system,
                metadata,
                quality_score,
                synthetic: synthetic != 0,
            });
        }

        Ok(result)
    }

    /// Lists all datasets (metadata only, no examples).
    pub fn list_datasets(&self) -> Result<Vec<(String, String, usize)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT d.id, d.name, COUNT(e.id) as example_count
             FROM datasets d
             LEFT JOIN dataset_examples e ON d.id = e.dataset_id
             GROUP BY d.id
             ORDER BY d.created_at DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    }

    /// Deletes a dataset.
    pub fn delete_dataset(&self, id: &str) -> Result<bool> {
        self.transaction(|tx| tx.delete_dataset(id))
    }

    /// Counts datasets.
    pub fn count_datasets(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM datasets", [], |row| row.get(0))?;
        Ok(count)
    }

    // =========================================================================
    // Experiment Operations
    // =========================================================================

    /// Saves an experiment.
    pub fn save_experiment(&self, experiment: &Experiment) -> Result<()> {
        self.transaction(|tx| {
            tx.insert_experiment(experiment)
        })
    }

    /// Saves a run.
    pub fn save_run(&self, experiment_id: &str, run: &Run) -> Result<()> {
        self.transaction(|tx| {
            tx.insert_run(run, experiment_id)
        })
    }

    /// Updates run status.
    pub fn update_run_status(&self, run_id: &str, status: RunStatus, error: Option<&str>) -> Result<bool> {
        self.transaction(|tx| tx.update_run_status(run_id, status, error))
    }

    /// Logs metrics for a run.
    pub fn log_run_metrics(&self, run_id: &str, metrics: &[(&str, f64)]) -> Result<()> {
        self.transaction(|tx| tx.log_metrics(run_id, metrics))
    }

    /// Counts experiments.
    pub fn count_experiments(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM experiments", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Counts runs.
    pub fn count_runs(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM runs", [], |row| row.get(0))?;
        Ok(count)
    }

    // =========================================================================
    // Prompt Operations
    // =========================================================================

    /// Saves a prompt template with all its versions.
    pub fn save_prompt_template(&self, template: &PromptTemplate) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("BEGIN TRANSACTION", [])?;

        // Delete existing template if updating
        conn.execute("DELETE FROM prompt_templates WHERE id = ?1", params![&template.id])?;

        // Insert template
        let tags_json = serde_json::to_string(&template.tags)?;
        conn.execute(
            "INSERT INTO prompt_templates (id, name, description, active_version_id, tags_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &template.id,
                &template.name,
                &template.description,
                &template.active_version_id,
                &tags_json,
                &template.created_at.to_rfc3339(),
                &template.updated_at.to_rfc3339(),
            ],
        )?;

        // Insert all versions
        for version in &template.versions {
            let metrics_json = serde_json::to_string(&version.metrics)?;
            conn.execute(
                "INSERT INTO prompt_versions (id, template_id, version_number, content, system_prompt, message, author, metrics_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    &version.id,
                    &template.id,
                    version.version_number,
                    &version.content,
                    &version.system_prompt,
                    &version.message,
                    &version.author,
                    &metrics_json,
                    &version.created_at.to_rfc3339(),
                ],
            )?;
        }

        conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Loads a prompt template by ID.
    pub fn load_prompt_template(&self, id: &str) -> Result<Option<PromptTemplate>> {
        let conn = self.conn.lock();

        let result = conn.query_row(
            "SELECT id, name, description, active_version_id, tags_json, created_at, updated_at
             FROM prompt_templates WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                ))
            },
        );

        let (id, name, description, active_version_id, tags_json, created_at, updated_at) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // Load versions
        let mut stmt = conn.prepare(
            "SELECT id, version_number, content, system_prompt, message, author, metrics_json, created_at
             FROM prompt_versions WHERE template_id = ?1 ORDER BY version_number"
        )?;

        let versions_result = stmt.query_map(params![&id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u32>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;

        let mut versions = Vec::new();
        for row in versions_result {
            let (vid, version_number, content, system_prompt, message, author, metrics_json, v_created_at) = row?;
            let metrics: VersionMetrics = serde_json::from_str(&metrics_json)?;

            versions.push(PromptVersion {
                id: vid,
                version_number,
                content,
                system_prompt,
                message,
                author,
                created_at: chrono::DateTime::parse_from_rfc3339(&v_created_at)
                    .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                    .with_timezone(&chrono::Utc),
                metrics,
            });
        }

        let tags: Vec<String> = serde_json::from_str(&tags_json)?;

        Ok(Some(PromptTemplate {
            id,
            name,
            description,
            versions,
            active_version_id,
            tags,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
        }))
    }

    /// Lists all prompt templates.
    pub fn list_prompt_templates(&self) -> Result<Vec<(String, String, usize)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT pt.id, pt.name, COUNT(pv.id) as version_count
             FROM prompt_templates pt
             LEFT JOIN prompt_versions pv ON pt.id = pv.template_id
             GROUP BY pt.id
             ORDER BY pt.created_at DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    }

    /// Deletes a prompt template.
    pub fn delete_prompt_template(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let count = conn.execute("DELETE FROM prompt_templates WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Counts prompt templates.
    pub fn count_prompt_templates(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM prompt_templates", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Saves a prompt A/B test.
    pub fn save_prompt_test(&self, template_id: Option<&str>, test: &TestResult) -> Result<()> {
        let conn = self.conn.lock();

        let version_a_results_json = serde_json::to_string(&test.version_a_results)?;
        let version_b_results_json = serde_json::to_string(&test.version_b_results)?;
        let status_str = match test.status {
            TestStatus::Running => "running",
            TestStatus::Completed => "completed",
            TestStatus::Cancelled => "cancelled",
            TestStatus::Inconclusive => "inconclusive",
        };

        conn.execute(
            "INSERT OR REPLACE INTO prompt_tests (id, name, template_id, version_a_id, version_b_id, version_a_results_json, version_b_results_json, winner, significance, status, started_at, ended_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                &test.test_id,
                &test.name,
                template_id,
                &test.version_a_id,
                &test.version_b_id,
                &version_a_results_json,
                &version_b_results_json,
                &test.winner,
                test.significance,
                status_str,
                &test.started_at.to_rfc3339(),
                test.ended_at.map(|t| t.to_rfc3339()),
            ],
        )?;

        Ok(())
    }

    /// Loads a prompt A/B test by ID.
    pub fn load_prompt_test(&self, id: &str) -> Result<Option<TestResult>> {
        let conn = self.conn.lock();

        let result = conn.query_row(
            "SELECT id, name, version_a_id, version_b_id, version_a_results_json, version_b_results_json, winner, significance, status, started_at, ended_at
             FROM prompt_tests WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<f32>>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, Option<String>>(10)?,
                ))
            },
        );

        let (test_id, name, version_a_id, version_b_id, version_a_results_json, version_b_results_json, winner, significance, status_str, started_at, ended_at) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let version_a_results: TestVersionResults = serde_json::from_str(&version_a_results_json)?;
        let version_b_results: TestVersionResults = serde_json::from_str(&version_b_results_json)?;

        let status = match status_str.as_str() {
            "completed" => TestStatus::Completed,
            "cancelled" => TestStatus::Cancelled,
            "inconclusive" => TestStatus::Inconclusive,
            _ => TestStatus::Running,
        };

        Ok(Some(TestResult {
            test_id,
            name,
            version_a_id,
            version_b_id,
            version_a_results,
            version_b_results,
            winner,
            significance,
            status,
            started_at: chrono::DateTime::parse_from_rfc3339(&started_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            ended_at: ended_at.map(|t| {
                chrono::DateTime::parse_from_rfc3339(&t)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .ok()
            }).flatten(),
        }))
    }

    /// Lists all prompt tests for a template.
    pub fn list_prompt_tests(&self, template_id: &str) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name FROM prompt_tests WHERE template_id = ?1 ORDER BY started_at DESC"
        )?;

        let rows = stmt.query_map(params![template_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut tests = Vec::new();
        for row in rows {
            tests.push(row?);
        }
        Ok(tests)
    }

    // =========================================================================
    // Model Registry Operations
    // =========================================================================

    /// Saves a model with all its versions.
    pub fn save_model(&self, model: &Model) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("BEGIN TRANSACTION", [])?;

        // Delete existing model if updating
        conn.execute("DELETE FROM models WHERE id = ?1", params![&model.id])?;

        // Insert model
        let tags_json = serde_json::to_string(&model.tags)?;
        conn.execute(
            "INSERT INTO models (id, name, description, base_model, task_type, tags_json, owner, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                &model.id,
                &model.name,
                &model.description,
                &model.base_model,
                &model.task_type,
                &tags_json,
                &model.owner,
                &model.created_at.to_rfc3339(),
                &model.updated_at.to_rfc3339(),
            ],
        )?;

        // Insert all versions
        for version in &model.versions {
            let metadata_json = serde_json::to_string(&version.metadata)?;
            let metrics_json = serde_json::to_string(&version.metrics)?;
            let stage_history_json = serde_json::to_string(&version.stage_history)?;

            conn.execute(
                "INSERT INTO model_versions (id, model_id, version, stage, metadata_json, artifact_path, metrics_json, experiment_run_id, dataset_id, stage_history_json, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    &version.id,
                    &model.id,
                    version.version,
                    version.stage.as_str(),
                    &metadata_json,
                    version.artifact_path.as_ref().map(|p| p.to_string_lossy().to_string()),
                    &metrics_json,
                    &version.experiment_run_id,
                    &version.dataset_id,
                    &stage_history_json,
                    &version.created_at.to_rfc3339(),
                ],
            )?;
        }

        conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Loads a model by ID.
    pub fn load_model(&self, id: &str) -> Result<Option<Model>> {
        let conn = self.conn.lock();

        let result = conn.query_row(
            "SELECT id, name, description, base_model, task_type, tags_json, owner, created_at, updated_at
             FROM models WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                ))
            },
        );

        let (id, name, description, base_model, task_type, tags_json, owner, created_at, updated_at) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // Load versions
        let mut stmt = conn.prepare(
            "SELECT id, version, stage, metadata_json, artifact_path, metrics_json, experiment_run_id, dataset_id, stage_history_json, created_at
             FROM model_versions WHERE model_id = ?1 ORDER BY version"
        )?;

        let versions_result = stmt.query_map(params![&id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, u32>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
            ))
        })?;

        let mut versions = Vec::new();
        for row in versions_result {
            let (vid, version, stage_str, metadata_json, artifact_path, metrics_json, experiment_run_id, dataset_id, stage_history_json, v_created_at) = row?;

            let stage = match stage_str.as_str() {
                "Staging" => ModelStage::Staging,
                "Production" => ModelStage::Production,
                "Archived" => ModelStage::Archived,
                _ => ModelStage::Development,
            };

            let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;
            let metrics: std::collections::HashMap<String, f64> = serde_json::from_str(&metrics_json)?;
            let stage_history: Vec<StageTransition> = serde_json::from_str(&stage_history_json)?;

            versions.push(ModelVersion {
                id: vid,
                version,
                stage,
                metadata,
                artifact_path: artifact_path.map(std::path::PathBuf::from),
                metrics,
                experiment_run_id,
                dataset_id,
                created_at: chrono::DateTime::parse_from_rfc3339(&v_created_at)
                    .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                    .with_timezone(&chrono::Utc),
                stage_history,
            });
        }

        let tags: Vec<String> = serde_json::from_str(&tags_json)?;

        Ok(Some(Model {
            id,
            name,
            description,
            base_model,
            task_type,
            versions,
            tags,
            owner,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
        }))
    }

    /// Lists all models.
    pub fn list_models(&self) -> Result<Vec<(String, String, usize)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT m.id, m.name, COUNT(mv.id) as version_count
             FROM models m
             LEFT JOIN model_versions mv ON m.id = mv.model_id
             GROUP BY m.id
             ORDER BY m.created_at DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    }

    /// Deletes a model.
    pub fn delete_model(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let count = conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Counts models.
    pub fn count_models(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM models", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Saves a deployment.
    pub fn save_deployment(&self, deployment: &Deployment) -> Result<()> {
        let conn = self.conn.lock();

        let resources_json = serde_json::to_string(&deployment.resources)?;
        let history_json = serde_json::to_string(&deployment.history)?;
        let environment_str = match deployment.environment {
            DeploymentEnvironment::Development => "Development",
            DeploymentEnvironment::Staging => "Staging",
            DeploymentEnvironment::Production => "Production",
        };
        let status_str = match deployment.status {
            DeploymentStatus::Pending => "Pending",
            DeploymentStatus::Deploying => "Deploying",
            DeploymentStatus::Running => "Running",
            DeploymentStatus::Failed => "Failed",
            DeploymentStatus::Stopped => "Stopped",
        };

        conn.execute(
            "INSERT OR REPLACE INTO deployments (id, name, model_id, model_version, environment, status, endpoint_url, resources_json, history_json, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &deployment.id,
                &deployment.name,
                &deployment.model_id,
                deployment.model_version,
                environment_str,
                status_str,
                &deployment.endpoint_url,
                &resources_json,
                &history_json,
                &deployment.created_at.to_rfc3339(),
                &deployment.updated_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Loads a deployment by ID.
    pub fn load_deployment(&self, id: &str) -> Result<Option<Deployment>> {
        let conn = self.conn.lock();

        let result = conn.query_row(
            "SELECT id, name, model_id, model_version, environment, status, endpoint_url, resources_json, history_json, created_at, updated_at
             FROM deployments WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u32>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, String>(10)?,
                ))
            },
        );

        let (id, name, model_id, model_version, environment_str, status_str, endpoint_url, resources_json, history_json, created_at, updated_at) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let environment = match environment_str.as_str() {
            "Staging" => DeploymentEnvironment::Staging,
            "Production" => DeploymentEnvironment::Production,
            _ => DeploymentEnvironment::Development,
        };

        let status = match status_str.as_str() {
            "Deploying" => DeploymentStatus::Deploying,
            "Running" => DeploymentStatus::Running,
            "Failed" => DeploymentStatus::Failed,
            "Stopped" => DeploymentStatus::Stopped,
            _ => DeploymentStatus::Pending,
        };

        let resources: DeploymentResources = serde_json::from_str(&resources_json)?;
        let history: Vec<DeploymentEvent> = serde_json::from_str(&history_json)?;

        Ok(Some(Deployment {
            id,
            name,
            model_id,
            model_version,
            environment,
            status,
            endpoint_url,
            resources,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
                .map_err(|e| PersistenceError::InvalidData(e.to_string()))?
                .with_timezone(&chrono::Utc),
            history,
        }))
    }

    /// Lists all deployments.
    pub fn list_deployments(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, status FROM deployments ORDER BY created_at DESC"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        Ok(result)
    }

    /// Deletes a deployment.
    pub fn delete_deployment(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM deployments WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Counts deployments.
    pub fn count_deployments(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: usize = conn.query_row("SELECT COUNT(*) FROM deployments", [], |row| row.get(0))?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetConfig;
    use tempfile::TempDir;

    fn test_db() -> StudioDatabase {
        StudioDatabase::in_memory().expect("create in-memory db")
    }

    fn create_test_dataset() -> Dataset {
        let examples = vec![
            Example::new("What is 2+2?", "4"),
            Example::new("What is 3+3?", "6"),
        ];
        Dataset::new(DatasetConfig::new("test-dataset"), examples)
    }

    #[test]
    fn test_database_creates_schema_on_init() {
        let db = test_db();

        assert!(db.table_exists("datasets").expect("check datasets"));
        assert!(db.table_exists("dataset_examples").expect("check examples"));
        assert!(db.table_exists("experiments").expect("check experiments"));
        assert!(db.table_exists("runs").expect("check runs"));
        assert!(db.table_exists("run_metrics").expect("check metrics"));
        assert!(db.table_exists("prompt_templates").expect("check prompt_templates"));
        assert!(db.table_exists("prompt_versions").expect("check prompt_versions"));
        assert!(db.table_exists("prompt_tests").expect("check prompt_tests"));
        assert!(db.table_exists("models").expect("check models"));
        assert!(db.table_exists("model_versions").expect("check model_versions"));
        assert!(db.table_exists("deployments").expect("check deployments"));
        assert!(db.table_exists("_migrations").expect("check migrations"));
    }

    #[test]
    fn test_schema_version() {
        let db = test_db();
        let version = db.schema_version().expect("get version");
        assert_eq!(version, CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn test_transaction_commit() {
        let db = test_db();
        let dataset = create_test_dataset();

        db.transaction(|tx| {
            tx.insert_dataset(&dataset)?;
            tx.insert_examples(&dataset.id, &dataset.examples)?;
            Ok(())
        }).expect("transaction should succeed");

        let loaded = db.load_dataset(&dataset.id).expect("load").expect("should exist");
        assert_eq!(loaded.name, "test-dataset");
        assert_eq!(loaded.examples.len(), 2);
    }

    #[test]
    fn test_transaction_rollback_on_error() {
        let db = test_db();
        let dataset = create_test_dataset();

        let result: Result<()> = db.transaction(|tx| {
            tx.insert_dataset(&dataset)?;
            // Simulate error
            Err(PersistenceError::Transaction("simulated error".into()))
        });

        assert!(result.is_err());

        // Dataset should not exist
        let loaded = db.load_dataset(&dataset.id).expect("load");
        assert!(loaded.is_none());
    }

    #[test]
    fn test_save_and_load_dataset() {
        let db = test_db();
        let dataset = create_test_dataset();

        db.save_dataset(&dataset).expect("save");

        let loaded = db.load_dataset(&dataset.id).expect("load").expect("should exist");
        assert_eq!(loaded.id, dataset.id);
        assert_eq!(loaded.name, dataset.name);
        assert_eq!(loaded.examples.len(), dataset.examples.len());
        assert_eq!(loaded.examples[0].input, "What is 2+2?");
    }

    #[test]
    fn test_list_datasets() {
        let db = test_db();

        // Create multiple datasets
        for i in 0..3 {
            let examples = vec![Example::new(format!("input{}", i), "output")];
            let dataset = Dataset::new(DatasetConfig::new(format!("dataset-{}", i)), examples);
            db.save_dataset(&dataset).expect("save");
        }

        let list = db.list_datasets().expect("list");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_delete_dataset() {
        let db = test_db();
        let dataset = create_test_dataset();

        db.save_dataset(&dataset).expect("save");
        assert!(db.load_dataset(&dataset.id).expect("load").is_some());

        let deleted = db.delete_dataset(&dataset.id).expect("delete");
        assert!(deleted);

        assert!(db.load_dataset(&dataset.id).expect("load").is_none());
    }

    #[test]
    fn test_count_datasets() {
        let db = test_db();

        assert_eq!(db.count_datasets().expect("count"), 0);

        for i in 0..5 {
            let dataset = Dataset::new(DatasetConfig::new(format!("ds-{}", i)), vec![]);
            db.save_dataset(&dataset).expect("save");
        }

        assert_eq!(db.count_datasets().expect("count"), 5);
    }

    #[test]
    fn test_file_based_database() {
        let temp = TempDir::new().expect("temp dir");
        let db_path = temp.path().join("test.db");

        // Create and populate
        {
            let db = StudioDatabase::new(&db_path).expect("create");
            let dataset = create_test_dataset();
            db.save_dataset(&dataset).expect("save");
        }

        // Reopen and verify
        {
            let db = StudioDatabase::new(&db_path).expect("reopen");
            let count = db.count_datasets().expect("count");
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_dataset_with_all_fields() {
        let db = test_db();

        let mut example = Example::new("input", "output")
            .with_system("You are helpful")
            .with_metadata("key", serde_json::json!("value"))
            .as_synthetic();
        example.quality_score = Some(0.95);

        let mut dataset = Dataset::new(
            DatasetConfig::new("full-dataset")
                .with_description("A complete dataset"),
            vec![example],
        );

        db.save_dataset(&dataset).expect("save");

        let loaded = db.load_dataset(&dataset.id).expect("load").expect("exists");
        assert_eq!(loaded.description, Some("A complete dataset".to_string()));
        assert_eq!(loaded.examples[0].system, Some("You are helpful".to_string()));
        assert!(loaded.examples[0].synthetic);
        assert_eq!(loaded.examples[0].quality_score, Some(0.95));
    }

    // =========================================================================
    // Prompt Tests
    // =========================================================================

    fn create_test_template() -> PromptTemplate {
        let mut template = PromptTemplate::new("test-template")
            .with_description("A test template")
            .with_tags(vec!["test".to_string()]);

        template.create_version("Hello {{name}}!", "Initial version");
        template.create_version("Hi {{name}}, welcome!", "More friendly");
        template
    }

    #[test]
    fn test_save_and_load_prompt_template() {
        let db = test_db();
        let template = create_test_template();

        db.save_prompt_template(&template).expect("save");

        let loaded = db.load_prompt_template(&template.id).expect("load").expect("exists");
        assert_eq!(loaded.id, template.id);
        assert_eq!(loaded.name, "test-template");
        assert_eq!(loaded.description, Some("A test template".to_string()));
        assert_eq!(loaded.versions.len(), 2);
        assert_eq!(loaded.tags.len(), 1);
    }

    #[test]
    fn test_list_prompt_templates() {
        let db = test_db();

        for i in 0..3 {
            let mut template = PromptTemplate::new(format!("template-{}", i));
            template.create_version("Content", "Initial");
            db.save_prompt_template(&template).expect("save");
        }

        let list = db.list_prompt_templates().expect("list");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_count_prompt_templates() {
        let db = test_db();

        assert_eq!(db.count_prompt_templates().expect("count"), 0);

        let template = create_test_template();
        db.save_prompt_template(&template).expect("save");

        assert_eq!(db.count_prompt_templates().expect("count"), 1);
    }

    #[test]
    fn test_delete_prompt_template() {
        let db = test_db();
        let template = create_test_template();

        db.save_prompt_template(&template).expect("save");
        assert!(db.load_prompt_template(&template.id).expect("load").is_some());

        let deleted = db.delete_prompt_template(&template.id).expect("delete");
        assert!(deleted);

        assert!(db.load_prompt_template(&template.id).expect("load").is_none());
    }

    #[test]
    fn test_save_and_load_prompt_test() {
        let db = test_db();

        let test_result = TestResult::new("A/B Test", "v1".to_string(), "v2".to_string());

        db.save_prompt_test(None, &test_result).expect("save");

        let loaded = db.load_prompt_test(&test_result.test_id).expect("load").expect("exists");
        assert_eq!(loaded.test_id, test_result.test_id);
        assert_eq!(loaded.name, "A/B Test");
        assert_eq!(loaded.version_a_id, "v1");
        assert_eq!(loaded.version_b_id, "v2");
        assert_eq!(loaded.status, TestStatus::Running);
    }

    // =========================================================================
    // Model Registry Tests
    // =========================================================================

    fn create_test_model() -> Model {
        let mut model = Model::new("test-model", "llama-7b", "text-generation")
            .with_description("A test model")
            .with_owner("test-user")
            .with_tags(vec!["test".to_string()]);

        model.create_version(ModelMetadata::new().with_format("safetensors"));
        model.create_version(ModelMetadata::new().with_format("gguf"));
        model
    }

    #[test]
    fn test_save_and_load_model() {
        let db = test_db();
        let model = create_test_model();

        db.save_model(&model).expect("save");

        let loaded = db.load_model(&model.id).expect("load").expect("exists");
        assert_eq!(loaded.id, model.id);
        assert_eq!(loaded.name, "test-model");
        assert_eq!(loaded.base_model, "llama-7b");
        assert_eq!(loaded.task_type, "text-generation");
        assert_eq!(loaded.versions.len(), 2);
        assert_eq!(loaded.tags.len(), 1);
    }

    #[test]
    fn test_list_models() {
        let db = test_db();

        for i in 0..3 {
            let mut model = Model::new(format!("model-{}", i), "base", "task");
            model.create_version(ModelMetadata::new());
            db.save_model(&model).expect("save");
        }

        let list = db.list_models().expect("list");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_count_models() {
        let db = test_db();

        assert_eq!(db.count_models().expect("count"), 0);

        let model = create_test_model();
        db.save_model(&model).expect("save");

        assert_eq!(db.count_models().expect("count"), 1);
    }

    #[test]
    fn test_delete_model() {
        let db = test_db();
        let model = create_test_model();

        db.save_model(&model).expect("save");
        assert!(db.load_model(&model.id).expect("load").is_some());

        let deleted = db.delete_model(&model.id).expect("delete");
        assert!(deleted);

        assert!(db.load_model(&model.id).expect("load").is_none());
    }

    #[test]
    fn test_save_and_load_deployment() {
        let db = test_db();

        // First create a model
        let model = create_test_model();
        db.save_model(&model).expect("save model");

        // Create a deployment
        let deployment = Deployment::new("prod-deploy", &model.id, 1, DeploymentEnvironment::Production);

        db.save_deployment(&deployment).expect("save deployment");

        let loaded = db.load_deployment(&deployment.id).expect("load").expect("exists");
        assert_eq!(loaded.id, deployment.id);
        assert_eq!(loaded.name, "prod-deploy");
        assert_eq!(loaded.model_id, model.id);
        assert_eq!(loaded.model_version, 1);
        assert_eq!(loaded.environment, DeploymentEnvironment::Production);
        assert_eq!(loaded.status, DeploymentStatus::Pending);
    }

    #[test]
    fn test_list_deployments() {
        let db = test_db();

        let model = create_test_model();
        db.save_model(&model).expect("save model");

        for i in 0..3 {
            let deployment = Deployment::new(
                format!("deploy-{}", i),
                &model.id,
                1,
                DeploymentEnvironment::Development,
            );
            db.save_deployment(&deployment).expect("save");
        }

        let list = db.list_deployments().expect("list");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_count_deployments() {
        let db = test_db();

        assert_eq!(db.count_deployments().expect("count"), 0);

        let model = create_test_model();
        db.save_model(&model).expect("save model");

        let deployment = Deployment::new("deploy", &model.id, 1, DeploymentEnvironment::Development);
        db.save_deployment(&deployment).expect("save deployment");

        assert_eq!(db.count_deployments().expect("count"), 1);
    }
}
