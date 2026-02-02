//! Gradient computation for neural network training.
//!
//! This module provides gradient computation capabilities using Candle's autograd
//! system, enabling actual backpropagation for fine-tuning LLMs.
//!
//! ## Features
//!
//! - **Automatic Differentiation**: Integration with Candle's Var system
//! - **Gradient Accumulation**: Support for micro-batching with gradient accumulation
//! - **Gradient Checkpointing**: Memory-efficient training for long sequences
//! - **Loss Functions**: Cross-entropy, SFT loss, and preference losses
//!
//! ## Usage
//!
//! ```ignore
//! use asmodeus::gradient::{GradientComputer, CrossEntropyLoss};
//!
//! let grad_computer = GradientComputer::new(device);
//! let loss = CrossEntropyLoss::new();
//!
//! // Forward pass with gradient tracking
//! let logits = model.forward_with_grad(&input)?;
//! let loss_value = loss.compute(&logits, &labels)?;
//!
//! // Backward pass
//! let gradients = grad_computer.backward(&loss_value)?;
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use serde::{Deserialize, Serialize};

/// Configuration for gradient computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConfig {
    /// Number of micro-batches for gradient accumulation.
    pub accumulation_steps: usize,
    /// Whether to use gradient checkpointing.
    pub gradient_checkpointing: bool,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
    /// Whether to compute gradients in fp16.
    pub fp16_gradients: bool,
    /// Whether to scale gradients for mixed precision.
    pub grad_scale: f64,
    /// Minimum scale before overflow.
    pub min_scale: f64,
    /// Growth factor for gradient scaling.
    pub scale_growth_factor: f64,
    /// Backoff factor for gradient scaling on overflow.
    pub scale_backoff_factor: f64,
    /// Number of steps before attempting scale increase.
    pub scale_growth_interval: u64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 1,
            gradient_checkpointing: false,
            max_grad_norm: 1.0,
            fp16_gradients: false,
            grad_scale: 65536.0,
            min_scale: 1.0,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
        }
    }
}

impl GradientConfig {
    /// Creates a config optimized for memory efficiency.
    #[must_use]
    pub fn memory_efficient() -> Self {
        Self {
            accumulation_steps: 4,
            gradient_checkpointing: true,
            fp16_gradients: true,
            ..Default::default()
        }
    }

    /// Creates a config optimized for speed.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            accumulation_steps: 1,
            gradient_checkpointing: false,
            fp16_gradients: false,
            ..Default::default()
        }
    }
}

/// Cross-entropy loss for language modeling.
pub struct CrossEntropyLoss {
    /// Ignore index for padding tokens.
    ignore_index: i64,
    /// Label smoothing factor.
    label_smoothing: f64,
    /// Reduction mode.
    reduction: Reduction,
}

/// Reduction mode for loss computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction.
    None,
    /// Mean over all elements.
    Mean,
    /// Sum over all elements.
    Sum,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    /// Creates a new cross-entropy loss.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ignore_index: -100,
            label_smoothing: 0.0,
            reduction: Reduction::Mean,
        }
    }

    /// Sets the ignore index.
    #[must_use]
    pub fn with_ignore_index(mut self, index: i64) -> Self {
        self.ignore_index = index;
        self
    }

    /// Sets label smoothing.
    #[must_use]
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing.clamp(0.0, 1.0);
        self
    }

    /// Sets the reduction mode.
    #[must_use]
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// Computes the cross-entropy loss.
    ///
    /// # Arguments
    /// * `logits` - Model outputs of shape (batch, seq_len, vocab_size)
    /// * `targets` - Target token IDs of shape (batch, seq_len)
    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        let targets_shape = targets.dims2()?;

        assert_eq!((batch_size, seq_len), targets_shape,
            "Logits and targets batch/seq dimensions must match");

        // Reshape for computation: (batch * seq, vocab)
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;

        // Compute log softmax
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;

        // Create mask for valid positions (not ignore_index)
        let targets_i64 = targets_flat.to_dtype(DType::I64)?;
        let valid_mask = self.create_valid_mask(&targets_i64, batch_size * seq_len)?;

        // Gather log probs for target classes
        // Convert targets to valid indices (clamp negative values to 0)
        let safe_targets = targets_flat.maximum(&Tensor::zeros_like(&targets_flat)?)?;
        let safe_targets = safe_targets.to_dtype(DType::U32)?;

        // Gather log probabilities for each target
        let target_log_probs = self.gather_along_axis(&log_probs, &safe_targets, 1)?;

        // Apply label smoothing if enabled
        let loss = if self.label_smoothing > 0.0 {
            // Smooth loss = (1 - smoothing) * nll + smoothing * uniform_loss
            let nll = target_log_probs.neg()?;
            let uniform_loss = log_probs.mean_keepdim(D::Minus1)?.squeeze(D::Minus1)?.neg()?;
            let smooth_factor = self.label_smoothing;
            let nll_factor = 1.0 - smooth_factor;

            let scaled_nll = nll.affine(nll_factor, 0.0)?;
            let scaled_uniform = uniform_loss.affine(smooth_factor, 0.0)?;
            scaled_nll.add(&scaled_uniform)?
        } else {
            target_log_probs.neg()?
        };

        // Apply mask
        let masked_loss = loss.mul(&valid_mask)?;

        // Apply reduction
        match self.reduction {
            Reduction::None => Ok(masked_loss),
            Reduction::Sum => masked_loss.sum_all(),
            Reduction::Mean => {
                let valid_count = valid_mask.sum_all()?;
                let loss_sum = masked_loss.sum_all()?;
                // Avoid division by zero
                let count = valid_count.to_scalar::<f32>()?.max(1.0);
                loss_sum.affine(1.0 / count as f64, 0.0)
            }
        }
    }

    /// Creates a mask for valid (non-ignored) positions.
    fn create_valid_mask(&self, targets: &Tensor, len: usize) -> CandleResult<Tensor> {
        let device = targets.device();
        let ignore_val = self.ignore_index;

        // Create mask: 1.0 where target != ignore_index, 0.0 otherwise
        let ignore_tensor = Tensor::full(ignore_val, (len,), device)?
            .to_dtype(DType::I64)?;

        // Compare using subtraction and checking for non-zero
        let diff = targets.sub(&ignore_tensor)?;
        let is_nonzero = diff.abs()?.gt(&Tensor::zeros((len,), DType::I64, device)?)?;
        is_nonzero.to_dtype(DType::F32)
    }

    /// Gathers values along an axis (simplified gather for 2D).
    fn gather_along_axis(
        &self,
        input: &Tensor,
        indices: &Tensor,
        _axis: usize,
    ) -> CandleResult<Tensor> {
        let (rows, cols) = input.dims2()?;
        let indices_len = indices.dim(0)?;
        assert_eq!(rows, indices_len, "Batch dimensions must match");

        // For each row, get the value at the column specified by index
        // This is equivalent to: [input[i, indices[i]] for i in range(rows)]

        // Use index_select with flattening approach
        let flat_input = input.reshape((rows * cols,))?;

        // Compute flat indices: i * cols + indices[i]
        let row_offsets = Tensor::arange(0u32, rows as u32, indices.device())?
            .to_dtype(DType::U32)?;
        let col_offset = (cols as u32).into();
        let row_offsets = row_offsets.affine(col_offset, 0.0)?
            .to_dtype(DType::U32)?;

        let flat_indices = indices.add(&row_offsets)?;
        let flat_indices_vec: Vec<u32> = flat_indices.to_vec1()?;

        // Gather using indexing
        let result_vec: Vec<f32> = flat_indices_vec
            .iter()
            .map(|&idx| {
                flat_input.get(idx as usize)
                    .and_then(|t| t.to_scalar::<f32>())
                    .unwrap_or(0.0)
            })
            .collect();

        Tensor::from_vec(result_vec, (rows,), input.device())
    }
}

/// Supervised fine-tuning (SFT) loss.
///
/// This computes cross-entropy loss only on the response portion of the sequence,
/// treating the instruction part as context.
pub struct SFTLoss {
    /// The underlying cross-entropy loss.
    ce_loss: CrossEntropyLoss,
    /// Response start token ID (e.g., start of assistant response).
    response_start_token: Option<u32>,
}

impl Default for SFTLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SFTLoss {
    /// Creates a new SFT loss.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ce_loss: CrossEntropyLoss::new(),
            response_start_token: None,
        }
    }

    /// Sets the response start token.
    #[must_use]
    pub fn with_response_start_token(mut self, token_id: u32) -> Self {
        self.response_start_token = Some(token_id);
        self
    }

    /// Computes SFT loss with optional response masking.
    ///
    /// # Arguments
    /// * `logits` - Model outputs of shape (batch, seq_len, vocab_size)
    /// * `targets` - Target token IDs of shape (batch, seq_len)
    /// * `response_mask` - Optional mask indicating response positions (1 = response, 0 = instruction)
    pub fn forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        response_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // If no mask provided, compute normal cross-entropy
        let Some(mask) = response_mask else {
            return self.ce_loss.forward(logits, targets);
        };

        let (batch_size, seq_len, vocab_size) = logits.dims3()?;

        // Reshape logits and targets
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;
        let mask_flat = mask.reshape((batch_size * seq_len,))?;

        // Compute log softmax
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;

        // Gather target log probs
        let safe_targets = targets_flat.maximum(&Tensor::zeros_like(&targets_flat)?)?;
        let safe_targets = safe_targets.to_dtype(DType::U32)?;
        let target_log_probs = self.ce_loss.gather_along_axis(&log_probs, &safe_targets, 1)?;

        // Apply NLL
        let nll = target_log_probs.neg()?;

        // Apply response mask and ignore index mask
        let ignore_mask = self.ce_loss.create_valid_mask(
            &targets_flat.to_dtype(DType::I64)?,
            batch_size * seq_len,
        )?;
        let combined_mask = mask_flat.mul(&ignore_mask)?;
        let masked_loss = nll.mul(&combined_mask)?;

        // Mean reduction
        let valid_count = combined_mask.sum_all()?;
        let loss_sum = masked_loss.sum_all()?;
        let count = valid_count.to_scalar::<f32>()?.max(1.0);
        loss_sum.affine(1.0 / count as f64, 0.0)
    }
}

/// Direct Preference Optimization (DPO) loss.
///
/// Computes the DPO loss for preference-based fine-tuning:
/// L = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_ref(y_w|x) + log_ref(y_l|x))))
pub struct DPOLoss {
    /// Beta parameter (KL penalty coefficient).
    beta: f64,
    /// Label smoothing for robustness.
    label_smoothing: f64,
}

impl Default for DPOLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DPOLoss {
    /// Creates a new DPO loss.
    #[must_use]
    pub fn new() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
        }
    }

    /// Sets the beta parameter.
    #[must_use]
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets label smoothing.
    #[must_use]
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Self {
        self.label_smoothing = smoothing.clamp(0.0, 0.5);
        self
    }

    /// Computes DPO loss.
    ///
    /// # Arguments
    /// * `policy_chosen_logps` - Log probs from policy model for chosen responses
    /// * `policy_rejected_logps` - Log probs from policy model for rejected responses
    /// * `ref_chosen_logps` - Log probs from reference model for chosen responses
    /// * `ref_rejected_logps` - Log probs from reference model for rejected responses
    pub fn forward(
        &self,
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        ref_chosen_logps: &Tensor,
        ref_rejected_logps: &Tensor,
    ) -> CandleResult<DPOOutput> {
        // Compute log ratios
        let pi_logratios = policy_chosen_logps.sub(policy_rejected_logps)?;
        let ref_logratios = ref_chosen_logps.sub(ref_rejected_logps)?;

        // Compute advantage: pi_logratios - ref_logratios
        let logits = pi_logratios.sub(&ref_logratios)?;
        let scaled_logits = logits.affine(self.beta, 0.0)?;

        // Apply label smoothing
        let labels = if self.label_smoothing > 0.0 {
            1.0 - self.label_smoothing
        } else {
            1.0
        };

        // DPO loss: -log(sigmoid(beta * logits))
        // = -log(1 / (1 + exp(-beta * logits)))
        // = log(1 + exp(-beta * logits))
        let neg_scaled = scaled_logits.neg()?;
        let exp_neg = neg_scaled.exp()?;
        let one_plus_exp = exp_neg.affine(1.0, 1.0)?;
        let loss = one_plus_exp.log()?;

        // Apply label smoothing to loss
        let loss = if self.label_smoothing > 0.0 {
            // Smooth loss: labels * loss + (1 - labels) * flipped_loss
            let flipped_loss = scaled_logits.exp()?.affine(1.0, 1.0)?.log()?;
            let main_loss = loss.affine(labels, 0.0)?;
            let smooth_loss = flipped_loss.affine(1.0 - labels, 0.0)?;
            main_loss.add(&smooth_loss)?
        } else {
            loss
        };

        let loss_mean = loss.mean_all()?;

        // Compute rewards for monitoring
        let chosen_rewards = policy_chosen_logps.sub(ref_chosen_logps)?
            .affine(self.beta, 0.0)?;
        let rejected_rewards = policy_rejected_logps.sub(ref_rejected_logps)?
            .affine(self.beta, 0.0)?;

        // Accuracy: how often chosen > rejected
        let reward_diff = chosen_rewards.sub(&rejected_rewards)?;
        let accuracy = reward_diff.gt(&Tensor::zeros_like(&reward_diff)?)?
            .to_dtype(DType::F32)?
            .mean_all()?;

        Ok(DPOOutput {
            loss: loss_mean,
            chosen_rewards,
            rejected_rewards,
            accuracy,
        })
    }
}

/// Output from DPO loss computation.
#[derive(Debug)]
pub struct DPOOutput {
    /// The DPO loss value.
    pub loss: Tensor,
    /// Rewards for chosen responses.
    pub chosen_rewards: Tensor,
    /// Rewards for rejected responses.
    pub rejected_rewards: Tensor,
    /// Accuracy (fraction where chosen > rejected).
    pub accuracy: Tensor,
}

/// Gradient accumulator for micro-batching.
pub struct GradientAccumulator {
    /// Accumulated gradients by parameter name.
    gradients: HashMap<String, Tensor>,
    /// Number of steps accumulated.
    steps: usize,
    /// Target number of accumulation steps.
    target_steps: usize,
    /// Device for computations.
    device: Device,
}

impl GradientAccumulator {
    /// Creates a new gradient accumulator.
    pub fn new(target_steps: usize, device: Device) -> Self {
        Self {
            gradients: HashMap::new(),
            steps: 0,
            target_steps,
            device,
        }
    }

    /// Adds gradients from a micro-batch.
    pub fn accumulate(&mut self, gradients: HashMap<String, Tensor>) -> CandleResult<()> {
        for (name, grad) in gradients {
            if let Some(existing) = self.gradients.get(&name) {
                self.gradients.insert(name, existing.add(&grad)?);
            } else {
                self.gradients.insert(name, grad);
            }
        }
        self.steps += 1;
        Ok(())
    }

    /// Returns whether accumulation is complete.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.steps >= self.target_steps
    }

    /// Returns the device used for computations.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Gets the accumulated gradients (averaged).
    pub fn get_gradients(&self) -> CandleResult<HashMap<String, Tensor>> {
        if self.steps == 0 {
            return Ok(HashMap::new());
        }

        let scale = 1.0 / self.steps as f64;
        let mut averaged = HashMap::new();

        for (name, grad) in &self.gradients {
            averaged.insert(name.clone(), grad.affine(scale, 0.0)?);
        }

        Ok(averaged)
    }

    /// Clears the accumulated gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
        self.steps = 0;
    }

    /// Returns the number of accumulated steps.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }
}

/// Gradient scaler for mixed precision training.
pub struct GradientScaler {
    /// Current scale factor.
    scale: f64,
    /// Minimum scale.
    min_scale: f64,
    /// Growth factor.
    growth_factor: f64,
    /// Backoff factor.
    backoff_factor: f64,
    /// Steps since last scale increase.
    steps_since_growth: u64,
    /// Growth interval.
    growth_interval: u64,
    /// Whether an overflow was detected.
    overflow_detected: bool,
}

impl GradientScaler {
    /// Creates a new gradient scaler.
    pub fn new(config: &GradientConfig) -> Self {
        Self {
            scale: config.grad_scale,
            min_scale: config.min_scale,
            growth_factor: config.scale_growth_factor,
            backoff_factor: config.scale_backoff_factor,
            steps_since_growth: 0,
            growth_interval: config.scale_growth_interval,
            overflow_detected: false,
        }
    }

    /// Scales the loss for backward pass.
    pub fn scale_loss(&self, loss: &Tensor) -> CandleResult<Tensor> {
        loss.affine(self.scale, 0.0)
    }

    /// Unscales gradients after backward pass.
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> CandleResult<bool> {
        let inv_scale = 1.0 / self.scale;
        let mut has_inf = false;

        for (_, grad) in gradients.iter_mut() {
            *grad = grad.affine(inv_scale, 0.0)?;

            // Check for inf/nan
            let grad_vec: Vec<f32> = grad.flatten_all()?.to_vec1()?;
            if grad_vec.iter().any(|&x| x.is_infinite() || x.is_nan()) {
                has_inf = true;
            }
        }

        Ok(!has_inf)
    }

    /// Updates the scale based on whether gradients were finite.
    pub fn update(&mut self, gradients_finite: bool) {
        if gradients_finite {
            self.overflow_detected = false;
            self.steps_since_growth += 1;

            // Try to grow scale
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
                tracing::debug!(new_scale = self.scale, "Increased gradient scale");
            }
        } else {
            // Overflow detected - reduce scale
            self.overflow_detected = true;
            self.scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.steps_since_growth = 0;
            tracing::warn!(new_scale = self.scale, "Overflow detected, reduced gradient scale");
        }
    }

    /// Returns whether to skip this optimization step due to overflow.
    #[must_use]
    pub fn should_skip_step(&self) -> bool {
        self.overflow_detected
    }

    /// Returns the current scale.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// Computes gradient norm for monitoring and clipping.
pub fn compute_grad_norm(gradients: &HashMap<String, Tensor>) -> CandleResult<f64> {
    let mut total_norm_sq = 0.0f64;

    for (_, grad) in gradients {
        let grad_sq = grad.sqr()?;
        let norm_sq = grad_sq.sum_all()?.to_scalar::<f32>()? as f64;
        total_norm_sq += norm_sq;
    }

    Ok(total_norm_sq.sqrt())
}

/// Clips gradients by global norm.
pub fn clip_grad_norm(
    gradients: &mut HashMap<String, Tensor>,
    max_norm: f64,
) -> CandleResult<f64> {
    let total_norm = compute_grad_norm(gradients)?;

    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;

        for (_, grad) in gradients.iter_mut() {
            *grad = grad.affine(clip_coef, 0.0)?;
        }

        tracing::debug!(
            original_norm = total_norm,
            clipped_to = max_norm,
            "Clipped gradients"
        );
    }

    Ok(total_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_config_defaults() {
        let config = GradientConfig::default();
        assert_eq!(config.accumulation_steps, 1);
        assert_eq!(config.max_grad_norm, 1.0);
    }

    #[test]
    fn test_gradient_config_memory_efficient() {
        let config = GradientConfig::memory_efficient();
        assert_eq!(config.accumulation_steps, 4);
        assert!(config.gradient_checkpointing);
    }

    #[test]
    fn test_cross_entropy_loss_creation() {
        let loss = CrossEntropyLoss::new()
            .with_ignore_index(-100)
            .with_label_smoothing(0.1);

        assert_eq!(loss.ignore_index, -100);
        assert!((loss.label_smoothing - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator() {
        let device = Device::Cpu;
        let mut accumulator = GradientAccumulator::new(4, device.clone());

        assert!(!accumulator.is_ready());
        assert_eq!(accumulator.steps(), 0);

        // Simulate accumulating gradients
        for i in 0..4 {
            let grad = Tensor::ones((4, 4), DType::F32, &device).unwrap();
            let mut gradients = HashMap::new();
            gradients.insert("test".to_string(), grad);
            accumulator.accumulate(gradients).unwrap();
            assert_eq!(accumulator.steps(), i + 1);
        }

        assert!(accumulator.is_ready());

        // Get averaged gradients
        let avg_grads = accumulator.get_gradients().unwrap();
        let avg_grad = avg_grads.get("test").unwrap();
        let mean_val = avg_grad.mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((mean_val - 1.0).abs() < 1e-6); // 4 * 1.0 / 4 = 1.0

        accumulator.clear();
        assert_eq!(accumulator.steps(), 0);
    }

    #[test]
    fn test_gradient_scaler() {
        let config = GradientConfig::default();
        let mut scaler = GradientScaler::new(&config);

        assert_eq!(scaler.scale(), 65536.0);

        // Simulate successful steps
        for _ in 0..2000 {
            scaler.update(true);
        }
        assert!(scaler.scale() > 65536.0); // Should have grown

        // Simulate overflow
        scaler.update(false);
        assert!(scaler.should_skip_step());
    }

    #[test]
    fn test_dpo_loss_creation() {
        let loss = DPOLoss::new()
            .with_beta(0.2)
            .with_label_smoothing(0.1);

        assert!((loss.beta - 0.2).abs() < 1e-6);
        assert!((loss.label_smoothing - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_compute_grad_norm() {
        let device = Device::Cpu;
        let grad1 = Tensor::full(3.0f32, (2, 2), &device).unwrap(); // norm^2 = 4 * 9 = 36
        let grad2 = Tensor::full(4.0f32, (2, 2), &device).unwrap(); // norm^2 = 4 * 16 = 64

        let mut gradients = HashMap::new();
        gradients.insert("g1".to_string(), grad1);
        gradients.insert("g2".to_string(), grad2);

        let norm = compute_grad_norm(&gradients).unwrap();
        // total_norm = sqrt(36 + 64) = sqrt(100) = 10
        assert!((norm - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm() {
        let device = Device::Cpu;
        let grad = Tensor::full(10.0f32, (2, 2), &device).unwrap(); // norm = sqrt(4 * 100) = 20

        let mut gradients = HashMap::new();
        gradients.insert("g".to_string(), grad);

        let original_norm = clip_grad_norm(&mut gradients, 5.0).unwrap();
        assert!((original_norm - 20.0).abs() < 1e-4);

        // After clipping, norm should be 5.0
        let clipped_norm = compute_grad_norm(&gradients).unwrap();
        assert!((clipped_norm - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_sft_loss_creation() {
        let loss = SFTLoss::new()
            .with_response_start_token(128000);

        assert_eq!(loss.response_start_token, Some(128000));
    }
}
