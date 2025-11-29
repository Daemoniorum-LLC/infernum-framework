//! Training loop implementation.

use std::path::PathBuf;

use async_trait::async_trait;
use infernum_core::Result;

use crate::config::TrainingConfig;

/// Training run status.
#[derive(Debug, Clone)]
pub enum TrainingStatus {
    /// Training is in progress.
    Running {
        /// Current epoch.
        epoch: u32,
        /// Current step.
        step: u64,
        /// Current loss.
        loss: f64,
    },
    /// Training completed.
    Completed {
        /// Final loss.
        final_loss: f64,
        /// Output path.
        output_path: PathBuf,
    },
    /// Training failed.
    Failed {
        /// Error message.
        error: String,
    },
}

/// Trait for trainers.
#[async_trait]
pub trait TrainerTrait: Send + Sync {
    /// Starts a training run.
    async fn train(&self, config: TrainingConfig) -> Result<String>;

    /// Gets the status of a training run.
    fn status(&self, run_id: &str) -> Option<TrainingStatus>;

    /// Stops a training run.
    async fn stop(&self, run_id: &str) -> Result<()>;
}

/// Default trainer implementation.
pub struct Trainer {
    output_dir: PathBuf,
}

impl Trainer {
    /// Creates a new trainer.
    #[must_use]
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// Returns the output directory.
    #[must_use]
    pub fn output_dir(&self) -> &PathBuf {
        &self.output_dir
    }
}

#[async_trait]
impl TrainerTrait for Trainer {
    async fn train(&self, config: TrainingConfig) -> Result<String> {
        tracing::info!(?config, "Starting training run");

        // TODO: Implement actual training
        let run_id = uuid::Uuid::new_v4().to_string();

        Ok(run_id)
    }

    fn status(&self, _run_id: &str) -> Option<TrainingStatus> {
        // TODO: Track actual training status
        None
    }

    async fn stop(&self, _run_id: &str) -> Result<()> {
        // TODO: Implement training stop
        Ok(())
    }
}
