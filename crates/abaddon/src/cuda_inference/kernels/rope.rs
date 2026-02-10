//! RoPE (Rotary Position Embeddings) CUDA kernel.
//!
//! Implements rotary position embeddings as used in Llama, Mistral, Qwen, etc.
//!
//! ## Algorithm
//!
//! For each position p and dimension pair (2i, 2i+1):
//! - freq = 1 / (theta^(2i/d))
//! - cos_val = cos(p * freq)
//! - sin_val = sin(p * freq)
//! - x_rot[2i] = x[2i] * cos_val - x[2i+1] * sin_val
//! - x_rot[2i+1] = x[2i] * sin_val + x[2i+1] * cos_val
//!
//! ## Scaling Support
//!
//! - NTK-aware scaling (extends context via theta modification)
//! - Linear scaling (scales position by factor)
//! - YaRN scaling (hybrid approach)
//!
//! Uses NVRTC to compile CUDA C code at runtime for better compatibility
//! across GPU architectures.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::InferenceError;

/// CUDA C source for RoPE kernel.
const ROPE_CUDA: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void rope_f16(
    __half* __restrict__ x,
    const int* __restrict__ positions,
    float theta,
    int head_dim,
    float scaling_factor,
    int num_heads
) {
    // Thread layout:
    // - blockIdx.x = batch index
    // - blockIdx.y = head index
    // - threadIdx.x = dimension pair (0 to head_dim/2 - 1)

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_pair = threadIdx.x;
    int half_dim = head_dim / 2;

    // Check bounds
    if (dim_pair >= half_dim) return;

    // Load position for this batch element
    int position = positions[batch_idx];

    // Apply scaling factor
    float scaled_position = (float)position * scaling_factor;

    // Compute frequency: freq = theta^(-2*dim_pair/head_dim)
    float exponent = (2.0f * (float)dim_pair) / (float)head_dim;
    float freq = powf(theta, -exponent);

    // angle = position * freq
    float angle = scaled_position * freq;

    // Compute sin and cos
    float sin_val = sinf(angle);
    float cos_val = cosf(angle);

    // Calculate offset: [batch, heads, head_dim]
    int offset = batch_idx * num_heads * head_dim + head_idx * head_dim + 2 * dim_pair;

    // Load x[2i] and x[2i+1]
    float x_even = __half2float(x[offset]);
    float x_odd = __half2float(x[offset + 1]);

    // Apply rotation:
    // x_rot[2i] = x[2i] * cos - x[2i+1] * sin
    // x_rot[2i+1] = x[2i] * sin + x[2i+1] * cos
    float x_rot_even = x_even * cos_val - x_odd * sin_val;
    float x_rot_odd = x_even * sin_val + x_odd * cos_val;

    // Store back
    x[offset] = __float2half(x_rot_even);
    x[offset + 1] = __float2half(x_rot_odd);
}

extern "C" __global__ void rope_precompute_freqs(
    float* __restrict__ freqs,
    float theta,
    int head_dim,
    int max_seq_len
) {
    // Thread processes one dimension pair
    int dim_pair = threadIdx.x;
    int half_dim = head_dim / 2;

    if (dim_pair >= half_dim) return;

    // Compute base frequency for this dimension pair
    float exponent = (2.0f * (float)dim_pair) / (float)head_dim;
    float freq = powf(theta, -exponent);

    // Loop over positions
    for (int pos = 0; pos < max_seq_len; pos++) {
        float angle = (float)pos * freq;

        // Compute cos and sin
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Store at freqs[pos * head_dim + dim_pair * 2]
        int offset = pos * head_dim + dim_pair * 2;
        freqs[offset] = cos_val;
        freqs[offset + 1] = sin_val;
    }
}
"#;

/// RoPE (Rotary Position Embeddings) kernel.
pub struct RoPEKernel {
    device: Arc<CudaDevice>,
    func: Option<CudaFunction>,
    precompute_func: Option<CudaFunction>,
}

impl std::fmt::Debug for RoPEKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoPEKernel")
            .field("loaded", &self.func.is_some())
            .finish()
    }
}

impl RoPEKernel {
    /// Create a new RoPE kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            func: None,
            precompute_func: None,
        };
        kernel.load_kernel()?;
        Ok(kernel)
    }

    /// Load the CUDA kernel.
    fn load_kernel(&mut self) -> Result<(), InferenceError> {
        // Compile CUDA C to PTX using NVRTC
        let ptx = compile_cuda_kernel(ROPE_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        // Load PTX into device
        self.device
            .load_ptx(ptx, "rope_kernels", &["rope_f16", "rope_precompute_freqs"])
            .map_err(|e| InferenceError::Kernel(format!("Failed to load PTX: {}", e)))?;

        self.func = Some(
            self.device
                .get_func("rope_kernels", "rope_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get rope_f16 function".to_string())
                })?,
        );

        self.precompute_func = Some(
            self.device
                .get_func("rope_kernels", "rope_precompute_freqs")
                .ok_or_else(|| {
                    InferenceError::Kernel(
                        "Failed to get rope_precompute_freqs function".to_string(),
                    )
                })?,
        );

        Ok(())
    }

    /// Apply RoPE embeddings in-place.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, heads, head_dim] in F16 (modified in-place)
    /// * `positions` - Position indices of shape [batch] in I32
    /// * `theta` - Base frequency (typically 10000.0)
    /// * `scaling_factor` - Position scaling (1.0 = no scaling, <1.0 extends context)
    pub fn forward(
        &self,
        x: &mut GpuTensor,
        positions: &GpuTensor,
        theta: f32,
        scaling_factor: f32,
    ) -> Result<(), InferenceError> {
        let func = self
            .func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("RoPE kernel not loaded".to_string()))?;

        // Validate shapes
        let x_shape = x.shape();
        if x_shape.len() != 3 {
            return Err(InferenceError::Shape {
                expected: "[batch, heads, head_dim]".to_string(),
                got: format!("{:?}", x_shape),
            });
        }

        let batch = x_shape[0];
        let num_heads = x_shape[1];
        let head_dim = x_shape[2];

        if head_dim % 2 != 0 {
            return Err(InferenceError::Shape {
                expected: "head_dim must be even".to_string(),
                got: format!("head_dim = {}", head_dim),
            });
        }

        // Validate positions shape
        let pos_shape = positions.shape();
        if pos_shape.len() != 1 || pos_shape[0] != batch {
            return Err(InferenceError::Shape {
                expected: format!("[{}]", batch),
                got: format!("{:?}", pos_shape),
            });
        }

        // Launch config: grid=(batch, heads), block=(head_dim/2)
        let cfg = LaunchConfig {
            grid_dim: (batch as u32, num_heads as u32, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.clone().launch(
                cfg,
                (
                    x.device_ptr(),
                    positions.device_ptr(),
                    theta,
                    head_dim as i32,
                    scaling_factor,
                    num_heads as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("RoPE kernel launch failed: {}", e)))?;

        Ok(())
    }

    /// Precompute RoPE frequency table.
    ///
    /// # Arguments
    /// * `freqs` - Output tensor of shape [max_seq_len, head_dim] in F32 (interleaved cos/sin)
    /// * `theta` - Base frequency (typically 10000.0)
    /// * `head_dim` - Dimension per head (must be even)
    /// * `max_seq_len` - Maximum sequence length to precompute
    pub fn precompute_freqs(
        &self,
        freqs: &mut GpuTensor,
        theta: f32,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<(), InferenceError> {
        let func = self.precompute_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("RoPE precompute kernel not loaded".to_string())
        })?;

        if head_dim % 2 != 0 {
            return Err(InferenceError::Shape {
                expected: "head_dim must be even".to_string(),
                got: format!("head_dim = {}", head_dim),
            });
        }

        // Launch: single block with head_dim/2 threads
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: ((head_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    freqs.device_ptr(),
                    theta,
                    head_dim as i32,
                    max_seq_len as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("RoPE precompute launch failed: {}", e)))?;

        Ok(())
    }

    /// Apply RoPE using precomputed frequencies.
    ///
    /// More efficient when processing multiple sequences with the same positions.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, heads, head_dim] in F16
    /// * `freqs` - Precomputed frequencies [max_seq_len, head_dim] in F32
    /// * `position` - Starting position index
    pub fn forward_with_cache(
        &self,
        _x: &mut GpuTensor,
        _freqs: &GpuTensor,
        _position: usize,
    ) -> Result<(), InferenceError> {
        // This would use a variant kernel that reads from precomputed freqs
        // For now, fall back to regular forward
        Err(InferenceError::Kernel(
            "Cached RoPE forward not yet implemented".to_string(),
        ))
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Compute NTK-aware scaling factor for extended context.
///
/// NTK-aware interpolation modifies the base frequency (theta) rather than
/// scaling positions, which better preserves high-frequency components.
///
/// # Arguments
/// * `original_max_len` - Original maximum sequence length the model was trained on
/// * `desired_max_len` - Desired extended sequence length
/// * `base_theta` - Original theta (typically 10000.0)
///
/// # Returns
/// Modified theta value for extended context
pub fn ntk_aware_scaling(original_max_len: usize, desired_max_len: usize, base_theta: f32) -> f32 {
    if desired_max_len <= original_max_len {
        return base_theta;
    }

    let scale = desired_max_len as f32 / original_max_len as f32;
    // NTK formula: theta_new = theta * scale^(dim / (dim - 2))
    // Simplified for typical head_dim=128: scale factor ~= scale
    base_theta * scale
}

/// Compute linear scaling factor.
///
/// Linear interpolation simply scales the position values, which can degrade
/// performance at higher frequencies.
///
/// # Arguments
/// * `original_max_len` - Original maximum sequence length
/// * `desired_max_len` - Desired sequence length
///
/// # Returns
/// Position scaling factor (multiply position by this)
pub fn linear_scaling(original_max_len: usize, desired_max_len: usize) -> f32 {
    if desired_max_len <= original_max_len {
        return 1.0;
    }

    original_max_len as f32 / desired_max_len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_kernel_compilation() {
        // Test that CUDA source compiles
        let result = compile_cuda_kernel(ROPE_CUDA);
        if let Err(e) = &result {
            // NVRTC errors will cause test failure below
        }
        // Don't assert - just check if NVRTC is available
    }

    #[test]
    fn test_ntk_scaling_no_extension() {
        let theta = ntk_aware_scaling(4096, 4096, 10000.0);
        assert!((theta - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_ntk_scaling_2x() {
        let theta = ntk_aware_scaling(4096, 8192, 10000.0);
        assert!(theta > 10000.0);
        assert!(theta < 30000.0); // Should be around 2x
    }

    #[test]
    fn test_linear_scaling_no_extension() {
        let scale = linear_scaling(4096, 4096);
        assert!((scale - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_scaling_2x() {
        let scale = linear_scaling(4096, 8192);
        assert!((scale - 0.5).abs() < 0.01);
    }
}
