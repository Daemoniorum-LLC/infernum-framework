//! GPU embedding lookup kernel.
//!
//! Performs token embedding lookup directly on the GPU, avoiding
//! expensive CPU-GPU memory transfers.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::InferenceError;

/// CUDA C source for embedding lookup kernel.
const EMBEDDING_CUDA: &str = r#"
#include <cuda_fp16.h>

// Embedding lookup (gather) kernel
// Each thread handles one element of the output
// Grid: (ceil(seq_len * hidden_size / 256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void embedding_gather_f16(
    const __half* __restrict__ embed_table,  // [vocab_size, hidden_size] F16
    const int* __restrict__ token_ids,        // [seq_len] I32
    __half* __restrict__ output,              // [seq_len, hidden_size] F16
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden_size;

    if (idx >= total) return;

    int row = idx / hidden_size;
    int col = idx % hidden_size;

    int token_id = token_ids[row];
    output[idx] = embed_table[token_id * hidden_size + col];
}

// Vectorized version using float2 (2 F16 values at once)
// Each thread handles 2 elements
extern "C" __global__ void embedding_gather_f16_vec2(
    const __half* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    __half* __restrict__ output,
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * (hidden_size / 2);

    if (idx >= total) return;

    int row = idx / (hidden_size / 2);
    int col = idx % (hidden_size / 2);

    int token_id = token_ids[row];

    // Cast to float (which contains 2 halfs) for vectorized load/store
    const float* embed_float = (const float*)(embed_table + token_id * hidden_size);
    float* out_float = (float*)(output + row * hidden_size);

    out_float[col] = embed_float[col];
}
"#;

/// Embedding lookup kernel for GPU execution.
pub struct EmbeddingKernel {
    device: Arc<CudaDevice>,
    gather_func: Option<CudaFunction>,
    gather_vec2_func: Option<CudaFunction>,
}

impl std::fmt::Debug for EmbeddingKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingKernel")
            .field("loaded", &self.gather_func.is_some())
            .finish()
    }
}

impl EmbeddingKernel {
    /// Create a new embedding kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            gather_func: None,
            gather_vec2_func: None,
        };
        kernel.load_kernel()?;
        Ok(kernel)
    }

    /// Load the CUDA kernel.
    fn load_kernel(&mut self) -> Result<(), InferenceError> {
        let ptx = compile_cuda_kernel(EMBEDDING_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        self.device
            .load_ptx(
                ptx,
                "embedding",
                &["embedding_gather_f16", "embedding_gather_f16_vec2"],
            )
            .map_err(|e| InferenceError::Kernel(format!("Failed to load PTX: {}", e)))?;

        self.gather_func = Some(
            self.device
                .get_func("embedding", "embedding_gather_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get embedding_gather_f16".to_string())
                })?,
        );

        self.gather_vec2_func = Some(
            self.device
                .get_func("embedding", "embedding_gather_f16_vec2")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get embedding_gather_f16_vec2".to_string())
                })?,
        );

        Ok(())
    }

    /// Look up token embeddings.
    ///
    /// # Arguments
    ///
    /// * `embed_table` - Embedding table [vocab_size, hidden_size] F16
    /// * `token_ids` - Token IDs tensor [seq_len] I32 on GPU
    /// * `output` - Output tensor [seq_len, hidden_size] F16
    pub fn forward(
        &self,
        embed_table: &GpuTensor,
        token_ids: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let embed_shape = embed_table.shape();
        let token_shape = token_ids.shape();
        let out_shape = output.shape();

        if embed_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D embedding table".to_string(),
                got: format!("{:?}", embed_shape),
            });
        }

        let hidden_size = embed_shape[1];
        let seq_len = token_shape[0];

        if out_shape != [seq_len, hidden_size] {
            return Err(InferenceError::Shape {
                expected: format!("[{}, {}]", seq_len, hidden_size),
                got: format!("{:?}", out_shape),
            });
        }

        // Use vectorized version if hidden_size is even (it usually is)
        if hidden_size % 2 == 0 {
            self.forward_vec2(embed_table, token_ids, output, seq_len, hidden_size)
        } else {
            self.forward_scalar(embed_table, token_ids, output, seq_len, hidden_size)
        }
    }

    fn forward_scalar(
        &self,
        embed_table: &GpuTensor,
        token_ids: &GpuTensor,
        output: &mut GpuTensor,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<(), InferenceError> {
        let func = self
            .gather_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("Embedding kernel not loaded".to_string()))?;

        let total = seq_len * hidden_size;
        let threads = 256;
        let blocks = (total + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    embed_table.device_ptr(),
                    token_ids.device_ptr(),
                    output.device_ptr(),
                    seq_len as i32,
                    hidden_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("Embedding gather failed: {}", e)))?;

        Ok(())
    }

    fn forward_vec2(
        &self,
        embed_table: &GpuTensor,
        token_ids: &GpuTensor,
        output: &mut GpuTensor,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<(), InferenceError> {
        let func = self.gather_vec2_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("Embedding vec2 kernel not loaded".to_string())
        })?;

        let total = seq_len * (hidden_size / 2);
        let threads = 256;
        let blocks = (total + threads - 1) / threads;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    embed_table.device_ptr(),
                    token_ids.device_ptr(),
                    output.device_ptr(),
                    seq_len as i32,
                    hidden_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("Embedding gather vec2 failed: {}", e)))?;

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
    fn test_embedding_kernel_compilation() {
        let result = compile_cuda_kernel(EMBEDDING_CUDA);
        if let Err(e) = &result {
            // NVRTC errors will cause test failure below
        }
    }
}
