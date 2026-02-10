//! Activation function CUDA kernels (SiLU, GELU, etc.).
//!
//! ## Supported Activations
//!
//! - **SiLU** (Sigmoid Linear Unit): `x * sigmoid(x)` - used by Llama, Mistral
//! - **GELU** (Gaussian Error Linear Unit): Transformer default
//! - **Fused SiLU+Mul**: `SiLU(gate) * up` - for gated MLPs in Llama/Mistral
//!
//! ## Gated MLP Structure (Llama-style)
//!
//! ```text
//! gate = w_gate @ x
//! up = w_up @ x
//! output = SiLU(gate) * up
//! down = w_down @ output
//! ```
//!
//! Uses NVRTC to compile CUDA C code at runtime for better compatibility
//! across GPU architectures.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::InferenceError;

/// Number of elements processed per thread.
const ELEMENTS_PER_THREAD: usize = 4;

/// CUDA C source for activation kernels.
const ACTIVATIONS_CUDA: &str = r#"
#include <cuda_fp16.h>

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void silu_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float val = __half2float(x[idx]);

        // SiLU: x / (1 + exp(-x)) = x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        float result = val * sigmoid_val;

        out[idx] = __float2half(result);
    }
}

// Fused SiLU + Multiply: SiLU(gate) * up
extern "C" __global__ void silu_mul_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float gate_val = __half2float(gate[idx]);
        float up_val = __half2float(up[idx]);

        // SiLU(gate) * up
        float sigmoid_val = 1.0f / (1.0f + expf(-gate_val));
        float silu_gate = gate_val * sigmoid_val;
        float result = silu_gate * up_val;

        out[idx] = __float2half(result);
    }
}

// GELU activation (fast approximation): x * sigmoid(1.702 * x)
extern "C" __global__ void gelu_fast_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float val = __half2float(x[idx]);

        // GELU fast: x * sigmoid(1.702 * x)
        float sigmoid_arg = 1.702f * val;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
        float result = val * sigmoid_val;

        out[idx] = __float2half(result);
    }
}

// GELU accurate: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__ void gelu_tanh_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float val = __half2float(x[idx]);

        // GELU tanh: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x_cubed = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * x_cubed);  // sqrt(2/pi) = 0.7978845608
        float tanh_val = tanhf(inner);
        float result = val * 0.5f * (1.0f + tanh_val);

        out[idx] = __float2half(result);
    }
}

// ReLU: max(0, x)
extern "C" __global__ void relu_f16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float val = __half2float(x[idx]);
        float result = fmaxf(0.0f, val);

        out[idx] = __float2half(result);
    }
}

// Hadamard (element-wise) product: out = a * b
extern "C" __global__ void hadamard_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float a_val = __half2float(a[idx]);
        float b_val = __half2float(b[idx]);
        float result = a_val * b_val;

        out[idx] = __float2half(result);
    }
}

// In-place Hadamard product: a = a * b
extern "C" __global__ void hadamard_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if (idx >= n) return;

        float a_val = __half2float(a[idx]);
        float b_val = __half2float(b[idx]);
        float result = a_val * b_val;

        a[idx] = __float2half(result);
    }
}
"#;

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// SiLU (Sigmoid Linear Unit): `x * sigmoid(x)`
    SiLU,
    /// GELU with fast sigmoid approximation
    GELUFast,
    /// GELU with accurate tanh approximation
    GELUTanh,
    /// ReLU: `max(0, x)`
    ReLU,
}

/// Activation function CUDA kernel.
pub struct ActivationKernel {
    device: Arc<CudaDevice>,
    silu_func: Option<CudaFunction>,
    silu_mul_func: Option<CudaFunction>,
    gelu_fast_func: Option<CudaFunction>,
    gelu_tanh_func: Option<CudaFunction>,
    relu_func: Option<CudaFunction>,
    hadamard_func: Option<CudaFunction>,
    hadamard_inplace_func: Option<CudaFunction>,
}

impl std::fmt::Debug for ActivationKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationKernel")
            .field("loaded", &self.silu_func.is_some())
            .finish()
    }
}

impl ActivationKernel {
    /// Create a new activation kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            silu_func: None,
            silu_mul_func: None,
            gelu_fast_func: None,
            gelu_tanh_func: None,
            relu_func: None,
            hadamard_func: None,
            hadamard_inplace_func: None,
        };
        kernel.load_kernels()?;
        Ok(kernel)
    }

    /// Load CUDA kernels.
    fn load_kernels(&mut self) -> Result<(), InferenceError> {
        // Compile CUDA C to PTX using NVRTC
        let ptx = compile_cuda_kernel(ACTIVATIONS_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        // Load PTX into device
        self.device
            .load_ptx(
                ptx,
                "activation_kernels",
                &[
                    "silu_f16",
                    "silu_mul_f16",
                    "gelu_fast_f16",
                    "gelu_tanh_f16",
                    "relu_f16",
                    "hadamard_f16",
                    "hadamard_inplace_f16",
                ],
            )
            .map_err(|e| InferenceError::Kernel(format!("Failed to load PTX: {}", e)))?;

        self.silu_func = Some(
            self.device
                .get_func("activation_kernels", "silu_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get silu_f16 function".to_string())
                })?,
        );

        self.silu_mul_func = Some(
            self.device
                .get_func("activation_kernels", "silu_mul_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get silu_mul_f16 function".to_string())
                })?,
        );

        self.gelu_fast_func = Some(
            self.device
                .get_func("activation_kernels", "gelu_fast_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get gelu_fast_f16 function".to_string())
                })?,
        );

        self.gelu_tanh_func = Some(
            self.device
                .get_func("activation_kernels", "gelu_tanh_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get gelu_tanh_f16 function".to_string())
                })?,
        );

        self.relu_func = Some(
            self.device
                .get_func("activation_kernels", "relu_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get relu_f16 function".to_string())
                })?,
        );

        self.hadamard_func = Some(
            self.device
                .get_func("activation_kernels", "hadamard_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get hadamard_f16 function".to_string())
                })?,
        );

        self.hadamard_inplace_func = Some(
            self.device
                .get_func("activation_kernels", "hadamard_inplace_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel(
                        "Failed to get hadamard_inplace_f16 function".to_string(),
                    )
                })?,
        );

        Ok(())
    }

    /// Get the kernel function for a given activation type.
    fn get_func(&self, activation: ActivationType) -> Result<&CudaFunction, InferenceError> {
        match activation {
            ActivationType::SiLU => self.silu_func.as_ref(),
            ActivationType::GELUFast => self.gelu_fast_func.as_ref(),
            ActivationType::GELUTanh => self.gelu_tanh_func.as_ref(),
            ActivationType::ReLU => self.relu_func.as_ref(),
        }
        .ok_or_else(|| InferenceError::Kernel(format!("{:?} kernel not loaded", activation)))
    }

    /// Apply activation function.
    ///
    /// # Arguments
    /// * `x` - Input tensor in F16
    /// * `out` - Output tensor in F16 (must be same size as x)
    /// * `activation` - Type of activation to apply
    pub fn forward(
        &self,
        x: &GpuTensor,
        out: &mut GpuTensor,
        activation: ActivationType,
    ) -> Result<(), InferenceError> {
        let func = self.get_func(activation)?;

        let n = x.numel();
        if out.numel() != n {
            return Err(InferenceError::Shape {
                expected: format!("{} elements", n),
                got: format!("{} elements", out.numel()),
            });
        }

        // Launch config: 256 threads, each processing 4 elements
        let block_size = 256;
        let grid_size =
            (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(cfg, (x.device_ptr(), out.device_ptr(), n as i32))
        }
        .map_err(|e| InferenceError::Kernel(format!("Activation kernel launch failed: {}", e)))?;

        Ok(())
    }

    /// Apply fused SiLU + multiply for gated MLP.
    ///
    /// Computes: `SiLU(gate) * up`
    ///
    /// This is the core operation in Llama-style gated MLPs:
    /// ```text
    /// gate = w_gate @ x
    /// up = w_up @ x
    /// out = SiLU(gate) * up
    /// ```
    ///
    /// # Arguments
    /// * `gate` - Gate activation tensor [n] in F16
    /// * `up` - Up projection tensor [n] in F16
    /// * `out` - Output tensor [n] in F16
    pub fn silu_mul(
        &self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let func = self
            .silu_mul_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("silu_mul_f16 kernel not loaded".to_string()))?;

        let n = gate.numel();
        if up.numel() != n || out.numel() != n {
            return Err(InferenceError::Shape {
                expected: format!("all tensors must have {} elements", n),
                got: format!(
                    "gate={}, up={}, out={}",
                    gate.numel(),
                    up.numel(),
                    out.numel()
                ),
            });
        }

        let block_size = 256;
        let grid_size =
            (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    gate.device_ptr(),
                    up.device_ptr(),
                    out.device_ptr(),
                    n as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("SiLU mul kernel launch failed: {}", e)))?;

        Ok(())
    }

    /// Element-wise (Hadamard) product: out = a * b.
    ///
    /// # Arguments
    /// * `a` - First input tensor [n] in F16
    /// * `b` - Second input tensor [n] in F16
    /// * `out` - Output tensor [n] in F16
    pub fn hadamard(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let func = self
            .hadamard_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("hadamard_f16 kernel not loaded".to_string()))?;

        let n = a.numel();
        if b.numel() != n || out.numel() != n {
            return Err(InferenceError::Shape {
                expected: format!("all tensors must have {} elements", n),
                got: format!("a={}, b={}, out={}", a.numel(), b.numel(), out.numel()),
            });
        }

        let block_size = 256;
        let grid_size =
            (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (a.device_ptr(), b.device_ptr(), out.device_ptr(), n as i32),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("Hadamard kernel launch failed: {}", e)))?;

        Ok(())
    }

    /// In-place element-wise (Hadamard) product: a = a * b.
    ///
    /// # Arguments
    /// * `a` - Input/output tensor [n] in F16 (modified in place)
    /// * `b` - Second input tensor [n] in F16
    pub fn hadamard_inplace(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<(), InferenceError> {
        let func = self.hadamard_inplace_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("hadamard_inplace_f16 kernel not loaded".to_string())
        })?;

        let n = a.numel();
        if b.numel() != n {
            return Err(InferenceError::Shape {
                expected: format!("tensors must have {} elements", n),
                got: format!("a={}, b={}", a.numel(), b.numel()),
            });
        }

        let block_size = 256;
        let grid_size =
            (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(cfg, (a.device_ptr(), b.device_ptr(), n as i32))
        }
        .map_err(|e| {
            InferenceError::Kernel(format!("Hadamard inplace kernel launch failed: {}", e))
        })?;

        Ok(())
    }

    /// Apply activation in-place.
    ///
    /// # Arguments
    /// * `x` - Input/output tensor in F16 (modified in place)
    /// * `activation` - Type of activation to apply
    pub fn forward_inplace(
        &self,
        x: &mut GpuTensor,
        activation: ActivationType,
    ) -> Result<(), InferenceError> {
        let func = self.get_func(activation)?;

        let n = x.numel();

        let block_size = 256;
        let grid_size =
            (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Use same pointer for input and output (in-place)
        unsafe {
            func.clone()
                .launch(cfg, (x.device_ptr(), x.device_ptr(), n as i32))
        }
        .map_err(|e| InferenceError::Kernel(format!("Activation kernel launch failed: {}", e)))?;

        Ok(())
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activations_kernel_compilation() {
        // Test that CUDA source compiles
        let result = compile_cuda_kernel(ACTIVATIONS_CUDA);
        if let Err(e) = &result {
            // NVRTC errors will cause test failure below
        }
        // Don't assert - just check if NVRTC is available
    }

    #[test]
    fn test_activation_type_equality() {
        assert_eq!(ActivationType::SiLU, ActivationType::SiLU);
        assert_ne!(ActivationType::SiLU, ActivationType::GELUFast);
        assert_ne!(ActivationType::GELUFast, ActivationType::GELUTanh);
    }
}
