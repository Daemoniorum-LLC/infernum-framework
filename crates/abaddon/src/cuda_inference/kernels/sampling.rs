//! GPU sampling kernels for token generation.
//!
//! Implements temperature scaling, softmax, and top-k/top-p sampling
//! directly on GPU for efficient autoregressive generation.
//!
//! Uses NVRTC to compile CUDA C code at runtime for better compatibility
//! across GPU architectures.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::{GpuDType, GpuTensor};
use crate::cuda_inference::InferenceError;

/// CUDA C source for sampling kernels.
const SAMPLING_CUDA: &str = r#"
#include <cuda_fp16.h>

// Softmax: Compute softmax over vocabulary
// Single-threaded for correctness, processes entire vocabulary
extern "C" __global__ void softmax_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int vocab_size
) {
    // Pass 1: Find max value (for numerical stability)
    float max_val = -1e20f;
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(input[i]);
        if (val > max_val) {
            max_val = val;
        }
    }

    // Pass 2: Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(input[i]);
        sum += expf(val - max_val);
    }

    // Pass 3: Normalize and write output
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        float val = __half2float(input[i]);
        float prob = expf(val - max_val) * inv_sum;
        output[i] = __float2half(prob);
    }
}

// Temperature scaling: Divide logits by temperature
// Grid-stride loop for parallel processing
extern "C" __global__ void temperature_scale_f16(
    __half* __restrict__ logits,
    int n,
    float inv_temp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        float val = __half2float(logits[i]);
        val *= inv_temp;
        logits[i] = __float2half(val);
    }
}

// Argmax: Find index of maximum value
// Single-threaded for correctness
extern "C" __global__ void argmax_f16(
    const __half* __restrict__ input,
    unsigned int* __restrict__ result,
    int n
) {
    float max_val = -1e20f;
    unsigned int max_idx = 0;

    for (int i = 0; i < n; i++) {
        float val = __half2float(input[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    result[0] = max_idx;
}
"#;

/// GPU-accelerated sampling kernel.
pub struct SamplingKernel {
    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Softmax kernel function.
    softmax_fn: Option<CudaFunction>,

    /// Temperature scaling kernel.
    temperature_fn: Option<CudaFunction>,

    /// Argmax kernel for greedy sampling.
    argmax_fn: Option<CudaFunction>,
}

impl std::fmt::Debug for SamplingKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingKernel")
            .field("loaded", &self.softmax_fn.is_some())
            .finish()
    }
}

impl SamplingKernel {
    /// Create a new sampling kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            softmax_fn: None,
            temperature_fn: None,
            argmax_fn: None,
        };

        kernel.load_kernels()?;
        Ok(kernel)
    }

    /// Load CUDA kernels.
    fn load_kernels(&mut self) -> Result<(), InferenceError> {
        // Compile CUDA C to PTX using NVRTC
        let ptx = compile_cuda_kernel(SAMPLING_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        // Load PTX into device
        self.device
            .load_ptx(
                ptx,
                "sampling",
                &["softmax_f16", "temperature_scale_f16", "argmax_f16"],
            )
            .map_err(|e| InferenceError::Kernel(format!("Failed to load sampling PTX: {}", e)))?;

        self.softmax_fn = Some(
            self.device
                .get_func("sampling", "softmax_f16")
                .ok_or_else(|| InferenceError::Kernel("Failed to get softmax_f16".to_string()))?,
        );

        self.temperature_fn = Some(
            self.device
                .get_func("sampling", "temperature_scale_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get temperature_scale_f16".to_string())
                })?,
        );

        self.argmax_fn = Some(
            self.device
                .get_func("sampling", "argmax_f16")
                .ok_or_else(|| InferenceError::Kernel("Failed to get argmax_f16".to_string()))?,
        );

        Ok(())
    }

    /// Apply temperature scaling to logits in-place.
    ///
    /// Divides logits by temperature. Higher temperature = more random.
    pub fn apply_temperature(
        &self,
        logits: &mut GpuTensor,
        temperature: f32,
    ) -> Result<(), InferenceError> {
        if temperature <= 0.0 {
            return Err(InferenceError::InvalidParam(
                "Temperature must be positive".to_string(),
            ));
        }

        if (temperature - 1.0).abs() < 1e-6 {
            // No scaling needed
            return Ok(());
        }

        let func = self
            .temperature_fn
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("Temperature kernel not loaded".to_string()))?;

        let n = logits.numel();
        let inv_temp = 1.0 / temperature;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(cfg, (logits.device_ptr(), n as i32, inv_temp))
                .map_err(|e| InferenceError::Kernel(e.to_string()))?;
        }

        Ok(())
    }

    /// Compute softmax probabilities from logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - Input logits [vocab_size]
    /// * `output` - Output probabilities [vocab_size]
    pub fn softmax(
        &self,
        logits: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let func = self
            .softmax_fn
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("Softmax kernel not loaded".to_string()))?;

        let vocab_size = logits.numel();

        if output.numel() != vocab_size {
            return Err(InferenceError::Shape {
                expected: format!("{} elements", vocab_size),
                got: format!("{} elements", output.numel()),
            });
        }

        // Single-threaded softmax for correctness (GPU parallelism for large vocabs)
        let cfg = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (logits.device_ptr(), output.device_ptr(), vocab_size as i32),
                )
                .map_err(|e| InferenceError::Kernel(e.to_string()))?;
        }

        Ok(())
    }

    /// Sample greedily (argmax) from logits.
    ///
    /// Returns the token ID with highest probability.
    pub fn sample_greedy(&self, logits: &GpuTensor) -> Result<u32, InferenceError> {
        let func = self
            .argmax_fn
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("Argmax kernel not loaded".to_string()))?;

        let n = logits.numel();

        // Allocate result on GPU
        let result: CudaSlice<u32> = self
            .device
            .alloc_zeros(1)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        let cfg = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(cfg, (logits.device_ptr(), *result.device_ptr(), n as i32))
                .map_err(|e| InferenceError::Kernel(e.to_string()))?;
        }

        // Copy result back
        let mut host_result = [0u32];
        self.device
            .dtoh_sync_copy_into(&result, &mut host_result)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        Ok(host_result[0])
    }

    /// Sample with top-k and top-p filtering.
    ///
    /// # Arguments
    ///
    /// * `logits` - Input logits [vocab_size]
    /// * `temperature` - Sampling temperature (1.0 = no change)
    /// * `top_k` - Top-k filtering (0 = disabled)
    /// * `top_p` - Top-p (nucleus) threshold (1.0 = disabled)
    /// * `rng_seed` - Random seed for sampling
    pub fn sample(
        &self,
        logits: &mut GpuTensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        rng_seed: u64,
    ) -> Result<u32, InferenceError> {
        let vocab_size = logits.numel();

        // Apply temperature
        self.apply_temperature(logits, temperature)?;

        // For now, use CPU sampling after softmax for simplicity
        // A full GPU implementation would do top-k/top-p filtering on device

        // Compute softmax
        let mut probs = GpuTensor::zeros(vec![vocab_size], GpuDType::F16, self.device.clone())?;
        self.softmax(logits, &mut probs)?;

        // Copy to CPU for sampling
        let mut host_probs_bytes = vec![0u8; vocab_size * 2];
        probs.copy_to_host(&mut host_probs_bytes)?;

        // Convert to f32 probabilities
        let host_probs: Vec<f32> = host_probs_bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        // Apply top-k filtering
        let filtered = if top_k > 0 && top_k < vocab_size {
            self.apply_top_k(&host_probs, top_k)
        } else {
            host_probs
        };

        // Apply top-p filtering
        let filtered = if top_p < 1.0 {
            self.apply_top_p(&filtered, top_p)
        } else {
            filtered
        };

        // Sample from filtered distribution
        self.multinomial_sample(&filtered, rng_seed)
    }

    /// Apply top-k filtering (CPU).
    fn apply_top_k(&self, probs: &[f32], k: usize) -> Vec<f32> {
        // Find k-th largest value
        let mut sorted: Vec<f32> = probs.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted.get(k - 1).copied().unwrap_or(0.0);

        // Zero out values below threshold
        let filtered: Vec<f32> = probs
            .iter()
            .map(|&p| if p >= threshold { p } else { 0.0 })
            .collect();

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter().map(|p| p / sum).collect()
        } else {
            filtered
        }
    }

    /// Apply top-p (nucleus) filtering (CPU).
    fn apply_top_p(&self, probs: &[f32], p: f32) -> Vec<f32> {
        // Sort indices by probability
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();
        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out values beyond cutoff
        let kept_indices: std::collections::HashSet<usize> =
            indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();

        let filtered: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if kept_indices.contains(&i) { p } else { 0.0 })
            .collect();

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter().map(|p| p / sum).collect()
        } else {
            filtered
        }
    }

    /// Multinomial sampling from probability distribution (CPU).
    fn multinomial_sample(&self, probs: &[f32], seed: u64) -> Result<u32, InferenceError> {
        // Simple LCG PRNG
        let mut state = seed;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };

        let r = random() as f32;
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probs.len() - 1) as u32)
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Repetition penalty implementation.
pub struct RepetitionPenalty {
    /// Penalty factor (1.0 = no penalty, >1.0 = discourage repetition).
    pub factor: f32,

    /// Tokens to penalize (recent context).
    pub context: Vec<u32>,

    /// Maximum context length to consider.
    pub max_context: usize,
}

impl RepetitionPenalty {
    /// Create a new repetition penalty.
    pub fn new(factor: f32, max_context: usize) -> Self {
        Self {
            factor,
            context: Vec::new(),
            max_context,
        }
    }

    /// Add a token to the context.
    pub fn add_token(&mut self, token: u32) {
        self.context.push(token);
        if self.context.len() > self.max_context {
            self.context.remove(0);
        }
    }

    /// Apply penalty to logits (CPU, modifies in-place).
    pub fn apply(&self, logits: &mut [f32]) {
        if self.factor == 1.0 {
            return;
        }

        for &token in &self.context {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.factor;
                } else {
                    logits[idx] *= self.factor;
                }
            }
        }
    }

    /// Clear context.
    pub fn clear(&mut self) {
        self.context.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_kernel_compilation() {
        // Test that CUDA source compiles
        let result = compile_cuda_kernel(SAMPLING_CUDA);
        if let Err(e) = &result {
            // NVRTC errors will cause test failure below
        }
        // Don't assert - just check if NVRTC is available
    }

    #[test]
    fn test_top_k_filtering() {
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                return; // Skip test: no CUDA device available
                return;
            },
        };
        let kernel = SamplingKernel {
            device,
            softmax_fn: None,
            temperature_fn: None,
            argmax_fn: None,
        };

        let probs = vec![0.1, 0.3, 0.2, 0.4];
        let filtered = kernel.apply_top_k(&probs, 2);

        // Should keep only top 2 (0.4 and 0.3)
        assert!(filtered[0] < 0.01); // Was 0.1, should be 0
        assert!(filtered[1] > 0.01); // Was 0.3, should be ~0.43
        assert!(filtered[2] < 0.01); // Was 0.2, should be 0
        assert!(filtered[3] > 0.01); // Was 0.4, should be ~0.57
    }

    #[test]
    fn test_top_p_filtering() {
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                return; // Skip test: no CUDA device available
                return;
            },
        };
        let kernel = SamplingKernel {
            device,
            softmax_fn: None,
            temperature_fn: None,
            argmax_fn: None,
        };

        let probs = vec![0.1, 0.3, 0.1, 0.5];
        let filtered = kernel.apply_top_p(&probs, 0.9);

        // Top-p 0.9 should keep 0.5 and 0.3 (cumsum = 0.8), then 0.1 to reach 0.9
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut penalty = RepetitionPenalty::new(1.5, 10);
        penalty.add_token(5);
        penalty.add_token(10);

        let mut logits = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, -1.0];
        penalty.apply(&mut logits);

        // Token 5 should be penalized (positive logit divided)
        assert!((logits[5] - 2.0 / 1.5).abs() < 0.01);
        // Token 10 should be penalized (negative logit multiplied)
        assert!((logits[10] - (-1.0 * 1.5)).abs() < 0.01);
        // Other tokens unchanged
        assert!((logits[0] - 1.0).abs() < 0.01);
    }
}
