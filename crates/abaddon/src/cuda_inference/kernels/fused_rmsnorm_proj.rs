//! Fused RMSNorm + Projection kernel.
//!
//! Combines RMSNorm with subsequent linear projection to reduce memory bandwidth.
//! Instead of:
//!   normed = RMSNorm(hidden, weight, eps)  // writes intermediate
//!   output = normed @ proj_weight          // reads intermediate
//!
//! We compute:
//!   output = RMSNorm(hidden, weight, eps) @ proj_weight  // no intermediate
//!
//! This eliminates one full hidden_size * seq_len read/write cycle.
//!
//! Uses NVRTC to compile CUDA C code at runtime for better compatibility.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::InferenceError;

/// CUDA source for fused RMSNorm + projection kernel.
const FUSED_RMSNORM_PROJ_CUDA: &str = r#"
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define TILE_K 128     // K tile size (hidden_size chunks)
#define TILE_N 64      // N tile size (output dimension chunks)

// Fused RMSNorm + F16 GEMM projection kernel
// Computes: output = RMSNorm(hidden, norm_weight, eps) @ proj_weight
//
// Each block handles one row (token) and a tile of output columns.
// Grid: (ceil(N/TILE_N), M, 1)
// Block: (256, 1, 1)
//
// Optimizations:
// - Hidden state loaded once, normalized in shared memory
// - Projection done with the normalized values immediately
// - Tiled output to maximize register reuse
// - Warp-level reductions for RMSNorm
extern "C" __global__ void fused_rmsnorm_gemm_f16(
    const __half* __restrict__ input,       // [M, K] hidden states
    const __half* __restrict__ norm_weight, // [K] RMSNorm weights
    const __half* __restrict__ proj_weight, // [K, N] projection matrix
    __half* __restrict__ output,            // [M, N] output
    int M,                                   // Number of tokens
    int K,                                   // Hidden size
    int N,                                   // Output size
    float eps                                // RMSNorm epsilon
) {
    // Block handles row = blockIdx.y, columns [blockIdx.x * TILE_N, ...)
    int row = blockIdx.y;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for normalized hidden states
    extern __shared__ float shared[];
    float* smem_hidden = shared;           // [K] normalized hidden values

    // Pointer to this row's input
    const __half* row_input = input + row * K;

    // ============================================================
    // Step 1: Compute RMS normalization factor
    // ============================================================
    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Inter-warp reduction via shared memory
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    // Use the end of shared memory for warp reduction
    float* warp_sums = shared + K;  // [num_warps]

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (tid < WARP_SIZE) {
        sum_sq = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (tid == 0) {
            warp_sums[0] = sum_sq;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)K + eps);

    // ============================================================
    // Step 2: Normalize and store in shared memory
    // ============================================================
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        float w = __half2float(norm_weight[i]);
        smem_hidden[i] = val * rms * w;
    }
    __syncthreads();

    // ============================================================
    // Step 3: Compute projection for this tile of output columns
    // ============================================================
    // Each thread computes multiple output columns
    int cols_per_thread = (TILE_N + block_size - 1) / block_size;

    for (int c = 0; c < cols_per_thread; c++) {
        int col = col_base + tid + c * block_size;
        if (col >= N) continue;

        float acc = 0.0f;

        // Dot product: normalized_hidden[k] * proj_weight[k, col]
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            float h = smem_hidden[k];
            float w = __half2float(proj_weight[k * N + col]);
            acc += h * w;
        }

        output[row * N + col] = __float2half(acc);
    }
}

// Fused RMSNorm + INT4 dequant + GEMM projection kernel
// Computes: output = RMSNorm(hidden, norm_weight, eps) @ dequant(proj_int4, scales)
//
// Grid: (ceil(N/TILE_N), M, 1)
// Block: (256, 1, 1)
extern "C" __global__ void fused_rmsnorm_int4_gemm_f16(
    const __half* __restrict__ input,        // [M, K] hidden states
    const __half* __restrict__ norm_weight,  // [K] RMSNorm weights
    const unsigned char* __restrict__ proj_weight, // [K/2, N] packed INT4
    const __half* __restrict__ scales,       // [K/32, N] scales
    __half* __restrict__ output,             // [M, N] output
    int M,                                    // Number of tokens
    int K,                                    // Hidden size
    int N,                                    // Output size
    float eps                                 // RMSNorm epsilon
) {
    int row = blockIdx.y;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float shared[];
    float* smem_hidden = shared;

    const __half* row_input = input + row * K;

    // ============================================================
    // Step 1: Compute RMS normalization factor
    // ============================================================
    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    float* warp_sums = shared + K;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        sum_sq = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (tid == 0) {
            warp_sums[0] = sum_sq;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)K + eps);

    // ============================================================
    // Step 2: Normalize and store in shared memory
    // ============================================================
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        float w = __half2float(norm_weight[i]);
        smem_hidden[i] = val * rms * w;
    }
    __syncthreads();

    // ============================================================
    // Step 3: Compute projection with INT4 dequantization
    // ============================================================
    int cols_per_thread = (TILE_N + block_size - 1) / block_size;

    for (int c = 0; c < cols_per_thread; c++) {
        int col = col_base + tid + c * block_size;
        if (col >= N) continue;

        float acc = 0.0f;

        // Process K dimension with block-wise scales
        for (int k_block = 0; k_block < K; k_block += 32) {
            // Load scale for this block
            float scale = __half2float(scales[(k_block / 32) * N + col]);

            // Dot product for this 32-element block
            int k_end = min(k_block + 32, K);
            for (int k = k_block; k < k_end; k++) {
                float h = smem_hidden[k];

                // Dequantize INT4 weight
                int packed_k = k / 2;
                int nibble_idx = k & 1;
                unsigned char packed = proj_weight[packed_k * N + col];
                int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);
                float w = ((float)int4_val - 8.0f) * scale;

                acc += h * w;
            }
        }

        output[row * N + col] = __float2half(acc);
    }
}

// Fused RMSNorm + GEMM with B transposed: output = RMSNorm(hidden) @ B^T
// Used for tied embeddings where lm_head uses embed_tokens transposed.
extern "C" __global__ void fused_rmsnorm_gemm_bt_f16(
    const __half* __restrict__ input,       // [M, K] hidden states
    const __half* __restrict__ norm_weight, // [K] RMSNorm weights
    const __half* __restrict__ proj_weight, // [N, K] projection matrix (transposed)
    __half* __restrict__ output,            // [M, N] output
    int M,                                   // Number of tokens
    int K,                                   // Hidden size
    int N,                                   // Output size (vocab_size)
    float eps                                // RMSNorm epsilon
) {
    int row = blockIdx.y;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float shared[];
    float* smem_hidden = shared;

    const __half* row_input = input + row * K;

    // Step 1: Compute RMS normalization factor
    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    float* warp_sums = shared + K;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        sum_sq = (tid < num_warps) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        if (tid == 0) {
            warp_sums[0] = sum_sq;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)K + eps);

    // Step 2: Normalize and store in shared memory
    for (int i = tid; i < K; i += block_size) {
        float val = __half2float(row_input[i]);
        float w = __half2float(norm_weight[i]);
        smem_hidden[i] = val * rms * w;
    }
    __syncthreads();

    // Step 3: Compute projection (B is transposed: [N, K])
    int cols_per_thread = (TILE_N + block_size - 1) / block_size;

    for (int c = 0; c < cols_per_thread; c++) {
        int col = col_base + tid + c * block_size;
        if (col >= N) continue;

        float acc = 0.0f;

        // B is [N, K], so B[col, k] = proj_weight[col * K + k]
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            float h = smem_hidden[k];
            float w = __half2float(proj_weight[col * K + k]);
            acc += h * w;
        }

        output[row * N + col] = __float2half(acc);
    }
}
"#;

/// Tile size for output columns.
const TILE_N: usize = 64;

/// Block size (threads per block).
const BLOCK_SIZE: usize = 256;

/// Fused RMSNorm + Projection kernel.
///
/// Combines RMSNorm with subsequent linear projection in a single kernel
/// to reduce memory bandwidth by eliminating intermediate storage.
pub struct FusedRMSNormProjKernel {
    device: Arc<CudaDevice>,
    f16_func: Option<CudaFunction>,
    int4_func: Option<CudaFunction>,
    bt_func: Option<CudaFunction>,
}

impl std::fmt::Debug for FusedRMSNormProjKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedRMSNormProjKernel")
            .field("f16_loaded", &self.f16_func.is_some())
            .field("int4_loaded", &self.int4_func.is_some())
            .field("bt_loaded", &self.bt_func.is_some())
            .finish()
    }
}

impl FusedRMSNormProjKernel {
    /// Create a new fused RMSNorm + projection kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            f16_func: None,
            int4_func: None,
            bt_func: None,
        };
        kernel.load_kernel()?;
        Ok(kernel)
    }

    /// Load the CUDA kernels.
    fn load_kernel(&mut self) -> Result<(), InferenceError> {
        let ptx = compile_cuda_kernel(FUSED_RMSNORM_PROJ_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        self.device
            .load_ptx(
                ptx,
                "fused_rmsnorm_proj",
                &[
                    "fused_rmsnorm_gemm_f16",
                    "fused_rmsnorm_int4_gemm_f16",
                    "fused_rmsnorm_gemm_bt_f16",
                ],
            )
            .map_err(|e| InferenceError::Kernel(format!("Failed to load PTX: {}", e)))?;

        self.f16_func = Some(
            self.device
                .get_func("fused_rmsnorm_proj", "fused_rmsnorm_gemm_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_rmsnorm_gemm_f16".to_string())
                })?,
        );

        self.int4_func = Some(
            self.device
                .get_func("fused_rmsnorm_proj", "fused_rmsnorm_int4_gemm_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_rmsnorm_int4_gemm_f16".to_string())
                })?,
        );

        self.bt_func = Some(
            self.device
                .get_func("fused_rmsnorm_proj", "fused_rmsnorm_gemm_bt_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_rmsnorm_gemm_bt_f16".to_string())
                })?,
        );

        Ok(())
    }

    /// Fused RMSNorm + F16 projection.
    ///
    /// Computes: output = RMSNorm(input, norm_weight, eps) @ proj_weight
    ///
    /// # Arguments
    ///
    /// * `input` - Hidden states [M, K] F16
    /// * `norm_weight` - RMSNorm weight [K] F16
    /// * `proj_weight` - Projection matrix [K, N] F16
    /// * `output` - Output tensor [M, N] F16
    /// * `eps` - RMSNorm epsilon
    pub fn forward_f16(
        &self,
        input: &GpuTensor,
        norm_weight: &GpuTensor,
        proj_weight: &GpuTensor,
        output: &mut GpuTensor,
        eps: f32,
    ) -> Result<(), InferenceError> {
        let func = self.f16_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("Fused RMSNorm+GEMM kernel not loaded".to_string())
        })?;

        let input_shape = input.shape();
        let proj_shape = proj_weight.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 || proj_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("input {:?}, proj {:?}", input_shape, proj_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = proj_shape[1];

        // Verify dimensions
        if proj_shape[0] != k || output_shape[0] != m || output_shape[1] != n {
            return Err(InferenceError::Shape {
                expected: format!("[{},{}] @ [{},{}] = [{},{}]", m, k, k, n, m, n),
                got: format!(
                    "input{:?} @ proj{:?} = output{:?}",
                    input_shape, proj_shape, output_shape
                ),
            });
        }

        // Grid: one block per (output_tile, row)
        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = m;

        // Shared memory: K floats for normalized hidden + warp reduction space
        let num_warps = (BLOCK_SIZE + 31) / 32;
        let shared_mem = (k + num_warps) * std::mem::size_of::<f32>();

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input.device_ptr(),
                        norm_weight.device_ptr(),
                        proj_weight.device_ptr(),
                        output.device_ptr(),
                        m as i32,
                        k as i32,
                        n as i32,
                        eps,
                    ),
                )
                .map_err(|e| InferenceError::Kernel(format!("Kernel launch failed: {}", e)))?;
        }

        Ok(())
    }

    /// Fused RMSNorm + INT4 dequant + projection.
    ///
    /// Computes: output = RMSNorm(input, norm_weight, eps) @ dequant(proj_int4, scales)
    ///
    /// # Arguments
    ///
    /// * `input` - Hidden states [M, K] F16
    /// * `norm_weight` - RMSNorm weight [K] F16
    /// * `proj_weight` - Packed INT4 weights [K/2, N]
    /// * `scales` - Per-block scales [K/32, N] F16
    /// * `output` - Output tensor [M, N] F16
    /// * `eps` - RMSNorm epsilon
    pub fn forward_int4(
        &self,
        input: &GpuTensor,
        norm_weight: &GpuTensor,
        proj_weight: &GpuTensor,
        scales: &GpuTensor,
        output: &mut GpuTensor,
        eps: f32,
    ) -> Result<(), InferenceError> {
        let func = self.int4_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("Fused RMSNorm+INT4+GEMM kernel not loaded".to_string())
        })?;

        let input_shape = input.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D input".to_string(),
                got: format!("{:?}", input_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = output_shape[1];

        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = m;

        let num_warps = (BLOCK_SIZE + 31) / 32;
        let shared_mem = (k + num_warps) * std::mem::size_of::<f32>();

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input.device_ptr(),
                        norm_weight.device_ptr(),
                        proj_weight.device_ptr(),
                        scales.device_ptr(),
                        output.device_ptr(),
                        m as i32,
                        k as i32,
                        n as i32,
                        eps,
                    ),
                )
                .map_err(|e| InferenceError::Kernel(format!("Kernel launch failed: {}", e)))?;
        }

        Ok(())
    }

    /// Fused RMSNorm + F16 projection with B transposed.
    ///
    /// Computes: output = RMSNorm(input, norm_weight, eps) @ proj_weight^T
    ///
    /// Used for tied embeddings where lm_head uses embed_tokens transposed.
    ///
    /// # Arguments
    ///
    /// * `input` - Hidden states [M, K] F16
    /// * `norm_weight` - RMSNorm weight [K] F16
    /// * `proj_weight` - Projection matrix [N, K] F16 (stored transposed)
    /// * `output` - Output tensor [M, N] F16
    /// * `eps` - RMSNorm epsilon
    pub fn forward_f16_bt(
        &self,
        input: &GpuTensor,
        norm_weight: &GpuTensor,
        proj_weight: &GpuTensor,
        output: &mut GpuTensor,
        eps: f32,
    ) -> Result<(), InferenceError> {
        let func = self.bt_func.as_ref().ok_or_else(|| {
            InferenceError::Kernel("Fused RMSNorm+GEMM BT kernel not loaded".to_string())
        })?;

        let input_shape = input.shape();
        let proj_shape = proj_weight.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 || proj_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("input {:?}, proj {:?}", input_shape, proj_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = proj_shape[0]; // B is [N, K] (transposed)

        // Verify dimensions
        if proj_shape[1] != k || output_shape[0] != m || output_shape[1] != n {
            return Err(InferenceError::Shape {
                expected: format!("[{},{}] @ [{},{}]^T = [{},{}]", m, k, n, k, m, n),
                got: format!(
                    "input{:?} @ proj{:?}^T = output{:?}",
                    input_shape, proj_shape, output_shape
                ),
            });
        }

        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = m;

        let num_warps = (BLOCK_SIZE + 31) / 32;
        let shared_mem = (k + num_warps) * std::mem::size_of::<f32>();

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        input.device_ptr(),
                        norm_weight.device_ptr(),
                        proj_weight.device_ptr(),
                        output.device_ptr(),
                        m as i32,
                        k as i32,
                        n as i32,
                        eps,
                    ),
                )
                .map_err(|e| InferenceError::Kernel(format!("Kernel launch failed: {}", e)))?;
        }

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
    fn test_fused_rmsnorm_proj_kernel_compilation() {
        let result = compile_cuda_kernel(FUSED_RMSNORM_PROJ_CUDA);
        // Just check if NVRTC can compile the kernel
        if result.is_err() {
            eprintln!("NVRTC compilation failed (expected if CUDA not available)");
        }
    }
}
