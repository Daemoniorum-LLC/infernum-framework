//! Training loop implementation with LoRA fine-tuning.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use candle_core::{Device, Result as CandleResult, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use infernum_core::Result;
use tokio::sync::watch;

use crate::config::{LoraConfig, TrainingConfig};
use crate::lora::{LoraLayer, LoraModel};

/// Training run status.
#[derive(Debug, Clone)]
pub enum TrainingStatus {
    /// Training is pending.
    Pending,
    /// Training is in progress.
    Running {
        /// Current epoch.
        epoch: u32,
        /// Current step.
        step: u64,
        /// Total steps.
        total_steps: u64,
        /// Current loss.
        loss: f64,
        /// Learning rate.
        learning_rate: f64,
    },
    /// Training completed.
    Completed {
        /// Final loss.
        final_loss: f64,
        /// Total steps completed.
        total_steps: u64,
        /// Output path.
        output_path: PathBuf,
    },
    /// Training failed.
    Failed {
        /// Error message.
        error: String,
    },
    /// Training stopped by user.
    Stopped {
        /// Step at which training was stopped.
        step: u64,
        /// Last loss value.
        last_loss: f64,
    },
}

/// A training sample.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Input token IDs.
    pub input_ids: Vec<u32>,
    /// Attention mask.
    pub attention_mask: Vec<u32>,
    /// Label token IDs.
    pub labels: Vec<u32>,
}

/// Training dataset abstraction.
pub trait Dataset: Send + Sync {
    /// Returns the number of samples.
    fn len(&self) -> usize;
    /// Returns whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Gets a sample by index.
    fn get(&self, idx: usize) -> Option<TrainingSample>;
}

/// In-memory dataset.
pub struct InMemoryDataset {
    samples: Vec<TrainingSample>,
}

impl InMemoryDataset {
    /// Creates a new in-memory dataset.
    pub fn new(samples: Vec<TrainingSample>) -> Self {
        Self { samples }
    }

    /// Creates a dataset from instruction-response pairs.
    pub fn from_instruction_pairs(
        pairs: Vec<(String, String)>,
        tokenize_fn: impl Fn(&str) -> Vec<u32>,
    ) -> Self {
        let samples = pairs
            .into_iter()
            .map(|(instruction, response)| {
                let full_text = format!(
                    "### Instruction:\n{}\n\n### Response:\n{}",
                    instruction, response
                );
                let input_ids = tokenize_fn(&full_text);
                let attention_mask = vec![1u32; input_ids.len()];

                // For causal LM, labels are shifted input_ids
                let labels = input_ids.clone();

                TrainingSample {
                    input_ids,
                    attention_mask,
                    labels,
                }
            })
            .collect();

        Self { samples }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Option<TrainingSample> {
        self.samples.get(idx).cloned()
    }
}

/// Data loader for batching training samples.
pub struct DataLoader {
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl DataLoader {
    /// Creates a new data loader.
    pub fn new(dataset: Arc<dyn Dataset>, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            // Simple shuffle using deterministic seed
            let seed = 42u64;
            for i in (1..indices.len()).rev() {
                let j = ((seed.wrapping_mul(i as u64 + 1)) % (i as u64 + 1)) as usize;
                indices.swap(i, j);
            }
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Resets the data loader for a new epoch.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            // Re-shuffle with different seed based on some state
            let seed = 42u64.wrapping_add(self.indices.len() as u64);
            for i in (1..self.indices.len()).rev() {
                let j = ((seed.wrapping_mul(i as u64 + 1)) % (i as u64 + 1)) as usize;
                self.indices.swap(i, j);
            }
        }
    }

    /// Returns the number of batches.
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for DataLoader {
    type Item = Vec<TrainingSample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch: Vec<TrainingSample> = self.indices[self.current_idx..end_idx]
            .iter()
            .filter_map(|&idx| self.dataset.get(idx))
            .collect();

        self.current_idx = end_idx;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// AdamW optimizer state for a single parameter.
#[derive(Debug)]
struct AdamWState {
    /// First moment (mean of gradients).
    m: Tensor,
    /// Second moment (variance of gradients).
    v: Tensor,
    /// Step count.
    step: u64,
}

/// AdamW optimizer with weight decay.
pub struct AdamW {
    /// Learning rate.
    lr: f64,
    /// Beta1 (first moment decay).
    beta1: f64,
    /// Beta2 (second moment decay).
    beta2: f64,
    /// Epsilon for numerical stability.
    eps: f64,
    /// Weight decay coefficient.
    weight_decay: f64,
    /// Optimizer state per parameter.
    states: HashMap<String, AdamWState>,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            states: HashMap::new(),
        }
    }

    /// Sets the learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Performs a single optimization step.
    pub fn step(&mut self, name: &str, param: &Tensor, grad: &Tensor) -> CandleResult<Tensor> {
        let device = param.device();
        let dtype = param.dtype();

        // Initialize state if needed
        if !self.states.contains_key(name) {
            let m = Tensor::zeros(param.shape(), dtype, device)?;
            let v = Tensor::zeros(param.shape(), dtype, device)?;
            self.states
                .insert(name.to_string(), AdamWState { m, v, step: 0 });
        }

        let state = self.states.get_mut(name).unwrap();
        state.step += 1;

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        let m_scaled = state.m.affine(self.beta1, 0.0)?;
        let grad_scaled = grad.affine(1.0 - self.beta1, 0.0)?;
        state.m = m_scaled.add(&grad_scaled)?;

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        let v_scaled = state.v.affine(self.beta2, 0.0)?;
        let grad_sq = grad.mul(grad)?;
        let grad_sq_scaled = grad_sq.affine(1.0 - self.beta2, 0.0)?;
        state.v = v_scaled.add(&grad_sq_scaled)?;

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(state.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(state.step as i32);

        // Compute bias-corrected estimates
        let m_hat = state.m.affine(1.0 / bias_correction1, 0.0)?;
        let v_hat = state.v.affine(1.0 / bias_correction2, 0.0)?;

        // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
        let v_sqrt = v_hat.sqrt()?;
        let denom = v_sqrt.affine(1.0, self.eps)?;
        let update = m_hat.div(&denom)?;
        let update = update.affine(self.lr, 0.0)?;

        // Apply weight decay: param = param - lr * weight_decay * param
        let param_decay = param.affine(self.lr * self.weight_decay, 0.0)?;

        // param = param - update - param_decay
        let new_param = param.sub(&update)?;
        new_param.sub(&param_decay)
    }
}

/// Learning rate scheduler with warmup and cosine decay.
pub struct LRScheduler {
    /// Initial learning rate.
    base_lr: f64,
    /// Warmup steps.
    warmup_steps: u64,
    /// Total training steps.
    total_steps: u64,
    /// Minimum learning rate ratio.
    min_lr_ratio: f64,
}

impl LRScheduler {
    /// Creates a new learning rate scheduler.
    pub fn new(base_lr: f64, warmup_steps: u64, total_steps: u64) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            min_lr_ratio: 0.1, // Minimum LR is 10% of base
        }
    }

    /// Gets the learning rate for a given step.
    pub fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress =
                (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
            let progress = progress.min(1.0);

            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            let decayed = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay;

            self.base_lr * decayed
        }
    }
}

/// Training run handle.
pub struct TrainingRun {
    /// Run identifier.
    pub id: String,
    /// Current status.
    status: Arc<RwLock<TrainingStatus>>,
    /// Stop signal sender.
    stop_tx: watch::Sender<bool>,
}

impl TrainingRun {
    /// Creates a new training run.
    fn new(id: String) -> (Self, watch::Receiver<bool>) {
        let (stop_tx, stop_rx) = watch::channel(false);
        (
            Self {
                id,
                status: Arc::new(RwLock::new(TrainingStatus::Pending)),
                stop_tx,
            },
            stop_rx,
        )
    }

    /// Gets the current status.
    pub fn status(&self) -> TrainingStatus {
        self.status.read().unwrap().clone()
    }

    /// Sends stop signal.
    pub fn stop(&self) {
        let _ = self.stop_tx.send(true);
    }

    /// Updates the status.
    fn update_status(&self, status: TrainingStatus) {
        *self.status.write().unwrap() = status;
    }
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

/// LoRA trainer implementation.
pub struct Trainer {
    /// Output directory for checkpoints and models.
    output_dir: PathBuf,
    /// Base model path or ID.
    base_model: String,
    /// Device to use for training.
    device: Device,
    /// Active training runs.
    runs: Arc<RwLock<HashMap<String, Arc<TrainingRun>>>>,
}

impl Trainer {
    /// Creates a new trainer.
    pub fn new(output_dir: impl Into<PathBuf>, base_model: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            base_model: base_model.into(),
            device: Device::Cpu,
            runs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a trainer with a specific device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Returns the output directory.
    #[must_use]
    pub fn output_dir(&self) -> &PathBuf {
        &self.output_dir
    }

    /// Runs a training loop with the given configuration and dataset.
    pub async fn train_with_dataset(
        &self,
        config: TrainingConfig,
        dataset: Arc<dyn Dataset>,
    ) -> Result<TrainingRun> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let (run, mut stop_rx) = TrainingRun::new(run_id.clone());
        let run = Arc::new(run);

        // Store the run
        self.runs
            .write()
            .unwrap()
            .insert(run_id.clone(), run.clone());

        let lora_config = config.lora.clone().unwrap_or_default();
        let output_dir = self.output_dir.clone();
        let base_model = self.base_model.clone();
        let device = self.device.clone();

        let run_clone = run.clone();

        // Spawn training task
        tokio::spawn(async move {
            let result = Self::run_training_loop(
                run_clone.clone(),
                config,
                lora_config,
                dataset,
                output_dir,
                base_model,
                device,
                &mut stop_rx,
            )
            .await;

            match result {
                Ok(_) => {
                    tracing::info!(run_id = %run_clone.id, "Training completed");
                },
                Err(e) => {
                    run_clone.update_status(TrainingStatus::Failed {
                        error: e.to_string(),
                    });
                    tracing::error!(run_id = %run_clone.id, error = %e, "Training failed");
                },
            }
        });

        Ok(Arc::try_unwrap(run).unwrap_or_else(|arc| (*arc).clone()))
    }

    /// Internal training loop.
    async fn run_training_loop(
        run: Arc<TrainingRun>,
        config: TrainingConfig,
        lora_config: LoraConfig,
        dataset: Arc<dyn Dataset>,
        output_dir: PathBuf,
        base_model: String,
        device: Device,
        stop_rx: &mut watch::Receiver<bool>,
    ) -> Result<()> {
        // Create LoRA model
        let mut lora_model = LoraModel::new(&base_model, lora_config.clone());

        // Find target modules and create LoRA layers
        let target_modules = crate::lora::find_target_modules("llama", &lora_config.target_modules);

        // Initialize LoRA layers (in a real implementation, we'd get dimensions from the model)
        // For now, use typical transformer dimensions
        let hidden_size = 4096;
        for module_name in &target_modules {
            let layer = LoraLayer::new(
                module_name,
                lora_config.clone(),
                hidden_size,
                hidden_size,
                &device,
            )
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?;
            lora_model.add_layer(layer);
        }

        tracing::info!(
            parameters = lora_model.total_parameters(),
            layers = lora_model.layers.len(),
            "Initialized LoRA model"
        );

        // Calculate total steps
        let steps_per_epoch =
            (dataset.len() + config.batch_size as usize - 1) / config.batch_size as usize;
        let total_steps = (steps_per_epoch * config.num_epochs as usize) as u64;

        // Initialize optimizer and scheduler
        let mut optimizer = AdamW::new(config.learning_rate, config.weight_decay);
        let scheduler = LRScheduler::new(
            config.learning_rate,
            config.warmup_steps as u64,
            total_steps,
        );

        // Create progress bar
        let progress = ProgressBar::new(total_steps);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) loss: {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut global_step: u64 = 0;
        let mut running_loss = 0.0;
        let mut loss_count = 0;

        // Training loop
        for epoch in 0..config.num_epochs {
            let mut data_loader =
                DataLoader::new(dataset.clone(), config.batch_size as usize, true);

            for batch in data_loader.by_ref() {
                // Check for stop signal
                if *stop_rx.borrow() {
                    progress.finish_with_message("Stopped");
                    run.update_status(TrainingStatus::Stopped {
                        step: global_step,
                        last_loss: running_loss / loss_count.max(1) as f64,
                    });
                    return Ok(());
                }

                // Get current learning rate
                let current_lr = scheduler.get_lr(global_step);
                optimizer.set_lr(current_lr);

                // Process batch
                let batch_loss = Self::process_batch(
                    &batch,
                    &mut lora_model,
                    &mut optimizer,
                    &device,
                    config.max_grad_norm,
                )
                .await?;

                running_loss += batch_loss;
                loss_count += 1;
                global_step += 1;

                // Update progress
                let avg_loss = running_loss / loss_count as f64;
                progress.set_position(global_step);
                progress.set_message(format!("{:.4}", avg_loss));

                // Update status
                run.update_status(TrainingStatus::Running {
                    epoch,
                    step: global_step,
                    total_steps,
                    loss: avg_loss,
                    learning_rate: current_lr,
                });

                // Save checkpoint periodically (every 500 steps)
                if global_step % 500 == 0 {
                    let checkpoint_path = output_dir.join(format!("checkpoint-{}", global_step));
                    lora_model.save(&checkpoint_path)?;
                    tracing::info!(step = global_step, path = ?checkpoint_path, "Saved checkpoint");
                }
            }

            tracing::info!(
                epoch = epoch + 1,
                loss = running_loss / loss_count.max(1) as f64,
                "Epoch completed"
            );
        }

        progress.finish_with_message("Complete");

        // Save final model
        let final_path = output_dir.join("final");
        lora_model.save(&final_path)?;

        let final_loss = running_loss / loss_count.max(1) as f64;
        run.update_status(TrainingStatus::Completed {
            final_loss,
            total_steps: global_step,
            output_path: final_path,
        });

        Ok(())
    }

    /// Processes a single batch and returns the loss.
    async fn process_batch(
        batch: &[TrainingSample],
        lora_model: &mut LoraModel,
        optimizer: &mut AdamW,
        device: &Device,
        max_grad_norm: f64,
    ) -> Result<f64> {
        // In a full implementation, this would:
        // 1. Convert batch to tensors
        // 2. Forward pass through base model + LoRA
        // 3. Compute cross-entropy loss
        // 4. Backward pass to get gradients
        // 5. Clip gradients
        // 6. Update LoRA parameters

        // For now, simulate with a mock loss that decreases over time
        // This demonstrates the training structure without requiring the full model

        let batch_size = batch.len();
        let seq_len = batch.first().map(|s| s.input_ids.len()).unwrap_or(0);

        // Create mock input tensor
        let input_data: Vec<f32> = batch
            .iter()
            .flat_map(|s| s.input_ids.iter().map(|&id| id as f32))
            .collect();

        if input_data.is_empty() {
            return Ok(0.0);
        }

        let input_tensor =
            Tensor::from_vec(input_data, (batch_size, seq_len), device).map_err(|e| {
                infernum_core::Error::Internal {
                    message: e.to_string(),
                }
            })?;

        // Simulate forward pass through LoRA layers
        let mut hidden = input_tensor;
        for (_name, layer) in &lora_model.layers {
            if layer.lora_a.is_some() && layer.lora_b.is_some() {
                // Apply LoRA transformation (simplified)
                let lora_out =
                    layer
                        .forward(&hidden)
                        .map_err(|e| infernum_core::Error::Internal {
                            message: e.to_string(),
                        })?;
                hidden = hidden
                    .add(&lora_out)
                    .map_err(|e| infernum_core::Error::Internal {
                        message: e.to_string(),
                    })?;
            }
        }

        // Compute mock loss (sum of squared values, normalized)
        let loss_tensor = hidden.sqr().map_err(|e| infernum_core::Error::Internal {
            message: e.to_string(),
        })?;
        let loss = loss_tensor
            .sum_all()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?
            .to_scalar::<f32>()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })? as f64;
        let normalized_loss = loss / (batch_size * seq_len) as f64;

        // Simulate gradients (in practice, these come from autograd)
        // For demonstration, create random gradients scaled by the loss
        for (name, layer) in lora_model.layers.iter_mut() {
            if let (Some(lora_a), Some(lora_b)) = (&layer.lora_a, &layer.lora_b) {
                // Create gradient tensors
                let grad_a =
                    Tensor::randn(0.0f32, 0.1f32, lora_a.shape(), device).map_err(|e| {
                        infernum_core::Error::Internal {
                            message: e.to_string(),
                        }
                    })?;
                let grad_b =
                    Tensor::randn(0.0f32, 0.1f32, lora_b.shape(), device).map_err(|e| {
                        infernum_core::Error::Internal {
                            message: e.to_string(),
                        }
                    })?;

                // Clip gradients by norm
                let grad_a = Self::clip_grad_norm(&grad_a, max_grad_norm)?;
                let grad_b = Self::clip_grad_norm(&grad_b, max_grad_norm)?;

                // Apply optimizer step
                let new_a = optimizer
                    .step(&format!("{}.lora_a", name), lora_a, &grad_a)
                    .map_err(|e| infernum_core::Error::Internal {
                        message: e.to_string(),
                    })?;
                let new_b = optimizer
                    .step(&format!("{}.lora_b", name), lora_b, &grad_b)
                    .map_err(|e| infernum_core::Error::Internal {
                        message: e.to_string(),
                    })?;

                layer.lora_a = Some(new_a);
                layer.lora_b = Some(new_b);
            }
        }

        Ok(normalized_loss)
    }

    /// Clips gradient tensor by norm.
    fn clip_grad_norm(grad: &Tensor, max_norm: f64) -> Result<Tensor> {
        let grad_norm = grad
            .sqr()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?
            .sum_all()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?
            .sqrt()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })?
            .to_scalar::<f32>()
            .map_err(|e| infernum_core::Error::Internal {
                message: e.to_string(),
            })? as f64;

        if grad_norm > max_norm {
            let scale = max_norm / grad_norm;
            grad.affine(scale, 0.0)
                .map_err(|e| infernum_core::Error::Internal {
                    message: e.to_string(),
                })
        } else {
            Ok(grad.clone())
        }
    }
}

impl Clone for TrainingRun {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            status: self.status.clone(),
            stop_tx: self.stop_tx.clone(),
        }
    }
}

#[async_trait]
impl TrainerTrait for Trainer {
    async fn train(&self, config: TrainingConfig) -> Result<String> {
        // Create a simple test dataset for demonstration
        let samples = vec![
            TrainingSample {
                input_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
                attention_mask: vec![1, 1, 1, 1, 1, 1, 1, 1],
                labels: vec![2, 3, 4, 5, 6, 7, 8, 9],
            };
            100
        ];
        let dataset = Arc::new(InMemoryDataset::new(samples));

        let run = self.train_with_dataset(config, dataset).await?;
        Ok(run.id)
    }

    fn status(&self, run_id: &str) -> Option<TrainingStatus> {
        self.runs
            .read()
            .unwrap()
            .get(run_id)
            .map(|run| run.status())
    }

    async fn stop(&self, run_id: &str) -> Result<()> {
        if let Some(run) = self.runs.read().unwrap().get(run_id) {
            run.stop();
            Ok(())
        } else {
            Err(infernum_core::Error::Internal {
                message: format!("Training run {} not found", run_id),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_lr_scheduler_warmup() {
        let scheduler = LRScheduler::new(1e-4, 100, 1000);

        // At step 0, LR should be 0
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-10);

        // At step 50, LR should be 0.5 * base
        assert!((scheduler.get_lr(50) - 5e-5).abs() < 1e-10);

        // At step 100, LR should be base
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_lr_scheduler_decay() {
        let scheduler = LRScheduler::new(1e-4, 0, 1000);

        // At step 0, LR should be base
        assert!((scheduler.get_lr(0) - 1e-4).abs() < 1e-10);

        // At step 1000, LR should be min_ratio * base
        let final_lr = scheduler.get_lr(1000);
        assert!((final_lr - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_data_loader() {
        let samples = vec![
            TrainingSample {
                input_ids: vec![1],
                attention_mask: vec![1],
                labels: vec![2],
            },
            TrainingSample {
                input_ids: vec![3],
                attention_mask: vec![1],
                labels: vec![4],
            },
            TrainingSample {
                input_ids: vec![5],
                attention_mask: vec![1],
                labels: vec![6],
            },
        ];
        let dataset = Arc::new(InMemoryDataset::new(samples));
        let loader = DataLoader::new(dataset, 2, false);

        assert_eq!(loader.num_batches(), 2);

        let batches: Vec<_> = loader.collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[tokio::test]
    async fn test_trainer_creation() {
        let dir = tempdir().unwrap();
        let trainer = Trainer::new(dir.path(), "test-model");

        assert_eq!(trainer.output_dir(), dir.path());
    }

    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamW::new(0.001, 0.01);
        let device = Device::Cpu;

        let param = Tensor::randn(0.0f32, 1.0f32, (4, 4), &device).unwrap();
        let grad = Tensor::randn(0.0f32, 0.1f32, (4, 4), &device).unwrap();

        let new_param = optimizer.step("test", &param, &grad).unwrap();

        // New param should be different from original
        let diff = new_param.sub(&param).unwrap();
        let diff_sum = diff
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff_sum > 0.0);
    }
}
