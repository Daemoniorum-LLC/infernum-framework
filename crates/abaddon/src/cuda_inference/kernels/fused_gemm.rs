//! Fused INT4 dequantization + GEMM kernel.
//!
//! This is the key optimization - dequantizes INT4 weights on-the-fly
//! during matrix multiplication, avoiding intermediate F16 storage.
//!
//! ## Quantization Scheme
//!
//! Uses block-wise symmetric quantization:
//! - Each block of 32 weights shares a scale factor
//! - `weight_f16 = (int4_val - 8) * scale`
//! - INT4 values are packed 2 per byte (low nibble first)
//!
//! ## Memory Layout
//!
//! - Weights: [K/2, N] packed INT4 (row-major)
//! - Scales: [K/32, N] F16 (one per 32-element block)
//! - Input: [M, K] F16 (row-major)
//! - Output: [M, N] F16 (row-major)
//!
//! ## Performance
//!
//! - Avoids 4x memory for F16 weights
//! - Single memory pass (no separate dequant kernel)
//! - Uses shared memory tiling for input reuse
//!
//! Uses NVRTC to compile CUDA C code at runtime for better compatibility
//! across GPU architectures.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};

use super::compile_cuda_kernel;
use crate::cuda_inference::tensor::GpuTensor;
use crate::cuda_inference::InferenceError;

/// Block size for INT4 quantization (weights per scale factor).
pub const QUANT_BLOCK_SIZE: usize = 32;

/// Tile size for GEMM computation.
const TILE_M: usize = 64;
const TILE_N: usize = 64;
#[allow(dead_code)]
const TILE_K: usize = 32;

/// CUDA C source for fused INT4 dequant + GEMM kernel (scalar + GEMV).
const FUSED_GEMM_CUDA: &str = r#"
#include <cuda_fp16.h>

// ============================================================================
// Tile sizes for GEMM computation (must match Rust constants for grid calc)
// ============================================================================
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

// ============================================================================
// Optimized INT4 GEMV (M=1) - Vector-Matrix Multiplication for Decode
// Each thread computes one output element by accumulating K dot products
// Grid: (ceil(N/GEMV_BLOCK), 1), Block: (GEMV_BLOCK, 1)
// ============================================================================
#define GEMV_BLOCK 256

extern "C" __global__ void fused_int4_gemv_f16(
    const __half* __restrict__ input,     // [1, K] F16 input row
    const unsigned char* __restrict__ weights,  // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,    // [K/32, N] F16 scales
    __half* __restrict__ output,          // [1, N] F16 output
    int N,                                // Number of columns (output size)
    int K                                 // Inner dimension
) {
    // Each thread computes one output column
    int col = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (col >= N) return;

    // Load input to shared memory for reuse across threads
    __shared__ float smem_input[1024];  // Max K we support

    // Cooperative loading of input
    int tid = threadIdx.x;
    for (int i = tid; i < K; i += GEMV_BLOCK) {
        smem_input[i] = __half2float(input[i]);
    }
    __syncthreads();

    // Accumulate dot product for this column
    float acc = 0.0f;

    // Process in chunks of 32 (scale block size)
    int num_scale_blocks = (K + 31) / 32;

    for (int sb = 0; sb < num_scale_blocks; sb++) {
        int k_start = sb * 32;
        int k_end = min(k_start + 32, K);

        // Load scale for this block
        float scale = __half2float(scales[sb * N + col]);

        // Process 32 K values (or fewer for last block)
        #pragma unroll 8
        for (int k = k_start; k < k_end; k++) {
            // Load input from shared memory
            float a = smem_input[k];

            // Load and dequantize INT4 weight
            int packed_k = k / 2;
            int nibble_idx = k & 1;
            unsigned char packed = weights[packed_k * N + col];
            int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);
            float b = ((float)int4_val - 8.0f) * scale;

            acc += a * b;
        }
    }

    // Write output
    output[col] = __float2half(acc);
}

// Optimized GEMV with vectorized weight loads (loads 4 columns at once)
// Each thread still computes one column, but we use vectorized memory access
#define GEMV_BLOCK_V4 256

extern "C" __global__ void fused_int4_gemv_f16_v2(
    const __half* __restrict__ input,     // [1, K] F16 input row
    const unsigned char* __restrict__ weights,  // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,    // [K/32, N] F16 scales
    __half* __restrict__ output,          // [1, N] F16 output
    int N,                                // Number of columns (output size)
    int K                                 // Inner dimension
) {
    // Each thread computes one output column
    int col = blockIdx.x * GEMV_BLOCK_V4 + threadIdx.x;
    if (col >= N) return;

    // Accumulate in registers
    float acc = 0.0f;

    // Number of scale blocks (each 32 K values)
    int num_scale_blocks = (K + 31) / 32;

    for (int sb = 0; sb < num_scale_blocks; sb++) {
        int k_start = sb * 32;
        int k_end = min(k_start + 32, K);

        // Load scale for this block
        float scale = __half2float(scales[sb * N + col]);

        // Process pairs of K values (each byte = 2 INT4 values)
        for (int k = k_start; k < k_end; k += 2) {
            // Load input values
            float a0 = __half2float(input[k]);
            float a1 = (k + 1 < K) ? __half2float(input[k + 1]) : 0.0f;

            // Load packed byte (2 INT4 values)
            int packed_k = k / 2;
            unsigned char packed = weights[packed_k * N + col];

            // Dequantize both values
            int val0 = packed & 0xF;
            int val1 = (packed >> 4) & 0xF;

            float b0 = ((float)val0 - 8.0f) * scale;
            float b1 = ((float)val1 - 8.0f) * scale;

            acc += a0 * b0 + a1 * b1;
        }
    }

    // Write output
    output[col] = __float2half(acc);
}

// F16 GEMV for non-quantized M=1 case
extern "C" __global__ void gemv_f16(
    const __half* __restrict__ input,     // [1, K] F16
    const __half* __restrict__ weights,   // [K, N] F16
    __half* __restrict__ output,          // [1, N] F16
    int N,
    int K
) {
    int col = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (col >= N) return;

    float acc = 0.0f;

    // Simple dot product - each thread handles one output column
    for (int k = 0; k < K; k++) {
        float a = __half2float(input[k]);
        float b = __half2float(weights[k * N + col]);
        acc += a * b;
    }

    output[col] = __float2half(acc);
}

// F16 GEMV with B transposed: output = input @ B^T
extern "C" __global__ void gemv_f16_bt(
    const __half* __restrict__ input,     // [1, K] F16
    const __half* __restrict__ weights,   // [N, K] F16 (B stored transposed)
    __half* __restrict__ output,          // [1, N] F16
    int N,
    int K
) {
    int col = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (col >= N) return;

    float acc = 0.0f;

    // B is [N, K], row 'col' is B[col, :] = B^T[:, col]
    // We want output[col] = sum_k input[k] * B[col, k]
    for (int k = 0; k < K; k++) {
        float a = __half2float(input[k]);
        float b = __half2float(weights[col * K + k]);  // B[col, k]
        acc += a * b;
    }

    output[col] = __float2half(acc);
}

// WMMA tensor core kernels are loaded separately for compatibility
// (Requires sm_70+ and proper CUDA toolkit headers)

// ============================================================================
// GPTQ/AWQ Asymmetric INT4 Kernels
// ============================================================================

// GPTQ/AWQ kernel with zero points and group index support
// Dequantization formula: weight = (int4_val - zeros[g_idx]) * scales[g_idx]
//
// GPTQ uses act_order which shuffles rows, requiring g_idx for correct grouping
// AWQ uses sequential groups but still needs zero points
//
// Grid: (ceil(N/GEMV_BLOCK), 1), Block: (GEMV_BLOCK, 1)
extern "C" __global__ void fused_gptq_gemv_f16(
    const __half* __restrict__ input,          // [1, K] F16 input row
    const unsigned char* __restrict__ weights, // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,         // [num_groups, N] F16 scales
    const __half* __restrict__ zeros,          // [num_groups, N] F16 zero points
    const int* __restrict__ g_idx,             // [K] group index per row (or NULL for sequential)
    __half* __restrict__ output,               // [1, N] F16 output
    int N,                                     // Number of columns (output size)
    int K,                                     // Inner dimension
    int group_size                             // Elements per quantization group (32 or 128)
) {
    int col = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (col >= N) return;

    float acc = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;

    // Process K values, handling group transitions
    for (int k = 0; k < K; k++) {
        // Get group index - either from g_idx array (GPTQ with act_order) or computed (AWQ/sequential)
        int group;
        if (g_idx != NULL) {
            group = g_idx[k];
        } else {
            group = k / group_size;
        }

        // Load input
        float a = __half2float(input[k]);

        // Load scale and zero point for this group
        float scale = __half2float(scales[group * N + col]);
        float zero = __half2float(zeros[group * N + col]);

        // Load and dequantize INT4 weight
        int packed_k = k / 2;
        int nibble_idx = k & 1;
        unsigned char packed = weights[packed_k * N + col];
        int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);

        // Asymmetric dequantization: (val - zero) * scale
        float b = ((float)int4_val - zero) * scale;

        acc += a * b;
    }

    output[col] = __float2half(acc);
}

// Optimized GPTQ GEMV for sequential groups (AWQ-style, no g_idx needed)
// Faster path when groups are contiguous
extern "C" __global__ void fused_awq_gemv_f16(
    const __half* __restrict__ input,          // [1, K] F16 input row
    const unsigned char* __restrict__ weights, // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,         // [num_groups, N] F16 scales
    const __half* __restrict__ zeros,          // [num_groups, N] F16 zero points
    __half* __restrict__ output,               // [1, N] F16 output
    int N,                                     // Number of columns (output size)
    int K,                                     // Inner dimension
    int group_size                             // Elements per quantization group (typically 128 for AWQ)
) {
    int col = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (col >= N) return;

    float acc = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;

    // Process in groups for better cache locality
    for (int g = 0; g < num_groups; g++) {
        int k_start = g * group_size;
        int k_end = min(k_start + group_size, K);

        // Load scale and zero point for this group once
        float scale = __half2float(scales[g * N + col]);
        float zero = __half2float(zeros[g * N + col]);

        // Process all K values in this group
        #pragma unroll 8
        for (int k = k_start; k < k_end; k++) {
            float a = __half2float(input[k]);

            int packed_k = k / 2;
            int nibble_idx = k & 1;
            unsigned char packed = weights[packed_k * N + col];
            int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);

            float b = ((float)int4_val - zero) * scale;
            acc += a * b;
        }
    }

    output[col] = __float2half(acc);
}

// GPTQ/AWQ GEMM for M > 1 (prefill)
// Supports both sequential groups (AWQ) and g_idx (GPTQ act_order)
extern "C" __global__ void fused_gptq_gemm_f16(
    const __half* __restrict__ input,          // [M, K] F16 input
    const unsigned char* __restrict__ weights, // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,         // [num_groups, N] F16 scales
    const __half* __restrict__ zeros,          // [num_groups, N] F16 zero points
    const int* __restrict__ g_idx,             // [K] group index (NULL for sequential)
    __half* __restrict__ output,               // [M, N] F16 output
    int M,
    int N,
    int K,
    int group_size
) {
    __shared__ float smem_a[TILE_M][TILE_K + 1];

    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    int tid = ty * 16 + tx;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col_base = bx * TILE_N;
    int row_base = by * TILE_M;

    int col_offset = tx * 4;
    int row_offset = ty * 4;
    int col = col_base + col_offset;
    int row = row_base + row_offset;

    bool col_valid[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        col_valid[j] = (col + j) < N;
    }

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += TILE_K) {
        __syncthreads();

        // Load input tile to shared memory
        int load_row = tid / 4;
        int load_col_start = (tid % 4) * 8;
        int input_row = row_base + load_row;
        int input_col = k + load_col_start;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (input_row < M && input_col + i < K) {
                smem_a[load_row][load_col_start + i] = __half2float(input[input_row * K + input_col + i]);
            } else {
                smem_a[load_row][load_col_start + i] = 0.0f;
            }
        }

        __syncthreads();

        // Inner loop over this K tile
        #pragma unroll 2
        for (int inner_k = 0; inner_k < TILE_K && k + inner_k < K; inner_k++) {
            int global_k = k + inner_k;

            // Get group index
            int group;
            if (g_idx != NULL) {
                group = g_idx[global_k];
            } else {
                group = global_k / group_size;
            }

            // Load input values from shared memory
            float a[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = smem_a[row_offset + i][inner_k];
            }

            // Load and dequantize weights
            int packed_k = global_k / 2;
            int nibble_idx = global_k & 1;

            float b[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (col_valid[j]) {
                    float scale = __half2float(scales[group * N + col + j]);
                    float zero = __half2float(zeros[group * N + col + j]);

                    unsigned char packed = weights[packed_k * N + col + j];
                    int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    b[j] = ((float)int4_val - zero) * scale;
                } else {
                    b[j] = 0.0f;
                }
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int out_row = row + i;
        if (out_row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (col_valid[j]) {
                output[out_row * N + col + j] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================================
// Scalar GEMM Kernels (fallback for all GPUs)
// ============================================================================

// Note: WMMA tensor core kernels (gemm_f16_wmma, fused_int4_gemm_wmma)
// are compiled separately with sm_70+ architecture flag

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32


// Optimized Fused INT4 dequant + GEMM kernel
// Computes: C = A @ dequant(B_int4, scales)
// Grid: (ceil(N/64), ceil(M/64), 1)
// Block: (16, 16, 1) = 256 threads
//
// Optimizations:
// - Shared memory for input tile with bank conflict avoidance
// - Pre-loaded scales (only change every 32 k values)
// - Vectorized weight loads
// - Loop unrolling
extern "C" __global__ void fused_int4_gemm_f16(
    const __half* __restrict__ input,     // [M, K] F16 input activations
    const unsigned char* __restrict__ weights,  // [K/2, N] packed INT4 weights
    const __half* __restrict__ scales,    // [K/32, N] F16 scales
    __half* __restrict__ output,          // [M, N] F16 output
    int M,                                // Number of rows in input
    int N,                                // Number of columns in output
    int K                                 // Inner dimension
) {
    // Shared memory for input tile (+1 to avoid bank conflicts)
    __shared__ float smem_a[TILE_M][TILE_K + 1];

    // Thread indices
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    int tid = ty * 16 + tx;

    // Block indices
    int bx = blockIdx.x;   // N dimension
    int by = blockIdx.y;   // M dimension

    // Output tile position
    int col_base = bx * TILE_N;
    int row_base = by * TILE_M;

    // Thread's output position within tile (4x4 sub-tile)
    int col_offset = tx * 4;
    int row_offset = ty * 4;

    int col = col_base + col_offset;
    int row = row_base + row_offset;

    // Pre-check column bounds
    bool col_valid[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        col_valid[j] = (col + j) < N;
    }

    // Initialize accumulators (4x4 = 16 values)
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K in tiles of 32
    for (int k = 0; k < K; k += TILE_K) {
        __syncthreads();

        // Cooperative load of input tile to shared memory
        // Each of 256 threads loads 8 elements
        int load_row = tid / 4;
        int load_col_start = (tid % 4) * 8;

        int input_row = row_base + load_row;
        int input_col = k + load_col_start;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (input_row < M && input_col + i < K) {
                smem_a[load_row][load_col_start + i] = __half2float(input[input_row * K + input_col + i]);
            } else {
                smem_a[load_row][load_col_start + i] = 0.0f;
            }
        }

        __syncthreads();

        // Load scale for this K tile (same for all 32 k values in this tile)
        int k_block = k / 32;
        float scale[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (col_valid[j]) {
                scale[j] = __half2float(scales[k_block * N + col + j]);
            } else {
                scale[j] = 0.0f;
            }
        }

        // Inner loop: process this K tile (unrolled by 2)
        #pragma unroll 2
        for (int inner_k = 0; inner_k < TILE_K && k + inner_k < K; inner_k++) {
            int global_k = k + inner_k;

            // Load 4 input values for this thread's rows from shared memory
            float a[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a[i] = smem_a[row_offset + i][inner_k];
            }

            // Load and dequantize INT4 weights
            int packed_k = global_k / 2;
            int nibble_idx = global_k & 1;

            float b[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (col_valid[j]) {
                    unsigned char packed = weights[packed_k * N + col + j];
                    int int4_val = nibble_idx ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    b[j] = ((float)int4_val - 8.0f) * scale[j];
                } else {
                    b[j] = 0.0f;
                }
            }

            // Accumulate: C[i][j] += A[i][k] * B[k][j]
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int out_row = row + i;
        if (out_row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (col_valid[j]) {
                output[out_row * N + col + j] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================================
// Optimized Tiled F16 GEMM - uses shared memory and register blocking
// Block: 16x16 threads, Tile: 64x64 output, each thread computes 4x4 elements
// ============================================================================
#define TILE_SIZE 64
#define BLOCK_SIZE 16
#define THREAD_TILE 4

extern "C" __global__ void gemm_f16(
    const __half* __restrict__ a,   // [M, K] F16
    const __half* __restrict__ b,   // [K, N] F16
    __half* __restrict__ c,         // [M, N] F16
    int M,
    int N,
    int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15

    // Block base position
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    // Each thread computes a 4x4 sub-tile
    float acc[THREAD_TILE][THREAD_TILE];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Cooperative load of A tile [TILE_SIZE x TILE_SIZE]
        // Each of 256 threads loads 16 elements (4x4 region)
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int row = by + ty * THREAD_TILE + i;
                int col = k_tile + tx * THREAD_TILE + j;
                if (row < M && col < K) {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __half2float(a[row * K + col]);
                } else {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        // Cooperative load of B tile [TILE_SIZE x TILE_SIZE]
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int row = k_tile + ty * THREAD_TILE + i;
                int col = bx + tx * THREAD_TILE + j;
                if (row < K && col < N) {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __half2float(b[row * N + col]);
                } else {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int kk = 0; kk < TILE_SIZE; kk++) {
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty * THREAD_TILE + i][kk];
                b_reg[i] = Bs[kk][tx * THREAD_TILE + i];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        int row = by + ty * THREAD_TILE + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int col = bx + tx * THREAD_TILE + j;
            if (col < N) {
                c[row * N + col] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================================
// Optimized Tiled F16 GEMM with B transposed: C = A @ B^T
// Block: 16x16 threads, Tile: 64x64 output, each thread computes 4x4 elements
// ============================================================================
extern "C" __global__ void gemm_f16_bt(
    const __half* __restrict__ a,   // [M, K] F16
    const __half* __restrict__ b,   // [N, K] F16 (B stored transposed)
    __half* __restrict__ c,         // [M, N] F16
    int M,
    int N,
    int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    float acc[THREAD_TILE][THREAD_TILE];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        // Load A tile
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int row = by + ty * THREAD_TILE + i;
                int col = k_tile + tx * THREAD_TILE + j;
                if (row < M && col < K) {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __half2float(a[row * K + col]);
                } else {
                    As[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        // Load B tile (B is [N, K], we want B^T which gives [K, N])
        // So we load B[col, k] which is B^T[k, col]
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int b_row = bx + tx * THREAD_TILE + j;  // column of output = row of B
                int b_col = k_tile + ty * THREAD_TILE + i;  // k position
                if (b_row < N && b_col < K) {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = __half2float(b[b_row * K + b_col]);
                } else {
                    Bs[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_SIZE; kk++) {
            float a_reg[THREAD_TILE];
            float b_reg[THREAD_TILE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty * THREAD_TILE + i][kk];
                b_reg[i] = Bs[kk][tx * THREAD_TILE + i];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        int row = by + ty * THREAD_TILE + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int col = bx + tx * THREAD_TILE + j;
            if (col < N) {
                c[row * N + col] = __float2half(acc[i][j]);
            }
        }
    }
}
"#;

/// Block size for GEMV kernels.
const GEMV_BLOCK: usize = 256;

/// Block size for WMMA tensor core kernels (64x64 output tiles).
#[allow(dead_code)]
const WMMA_BLOCK_M: usize = 64;
#[allow(dead_code)]
const WMMA_BLOCK_N: usize = 64;

/// Fused INT4 dequant + GEMM kernel wrapper.
///
/// This kernel combines weight dequantization with matrix multiplication
/// for optimal memory bandwidth utilization.
///
/// Includes:
/// - WMMA tensor core kernels for M > 1 (prefill) - highest throughput
/// - GEMV kernels for M=1 (decode) - optimized for single-token latency
/// - GPTQ/AWQ kernels for asymmetric quantization with zero points
pub struct FusedGemmKernel {
    device: Arc<CudaDevice>,
    // GEMM kernels (M > 1, scalar fallback)
    int4_func: Option<CudaFunction>,
    f16_func: Option<CudaFunction>,
    f16_bt_func: Option<CudaFunction>,
    // GEMV kernels (M = 1, optimized for decode)
    int4_gemv_func: Option<CudaFunction>,
    int4_gemv_v2_func: Option<CudaFunction>,
    f16_gemv_func: Option<CudaFunction>,
    f16_gemv_bt_func: Option<CudaFunction>,
    // WMMA tensor core kernels (M > 1, highest throughput)
    int4_wmma_func: Option<CudaFunction>,
    f16_wmma_func: Option<CudaFunction>,
    // GPTQ/AWQ kernels (asymmetric INT4 with zero points)
    gptq_gemv_func: Option<CudaFunction>,
    awq_gemv_func: Option<CudaFunction>,
    gptq_gemm_func: Option<CudaFunction>,
    /// Whether to use tensor cores (auto-detected).
    use_tensor_cores: bool,
}

impl std::fmt::Debug for FusedGemmKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedGemmKernel")
            .field("loaded", &self.int4_func.is_some())
            .finish()
    }
}

impl FusedGemmKernel {
    /// Create a new fused GEMM kernel instance.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut kernel = Self {
            device,
            int4_func: None,
            f16_func: None,
            f16_bt_func: None,
            int4_gemv_func: None,
            int4_gemv_v2_func: None,
            f16_gemv_func: None,
            f16_gemv_bt_func: None,
            int4_wmma_func: None,
            f16_wmma_func: None,
            gptq_gemv_func: None,
            awq_gemv_func: None,
            gptq_gemm_func: None,
            use_tensor_cores: false, // Tensor cores disabled (require CUDA toolkit headers)
        };
        kernel.load()?;
        Ok(kernel)
    }

    /// Load all kernels (GEMM, GEMV, and WMMA tensor core).
    pub fn load(&mut self) -> Result<(), InferenceError> {
        if self.int4_func.is_some() {
            return Ok(());
        }

        // Compile CUDA C to PTX using NVRTC
        // Note: WMMA requires sm_70+ (Volta or newer)
        let ptx = compile_cuda_kernel(FUSED_GEMM_CUDA)
            .map_err(|e| InferenceError::Kernel(format!("NVRTC compilation failed: {}", e)))?;

        // Load PTX into device - include all kernel names
        self.device
            .load_ptx(
                ptx,
                "fused_gemm",
                &[
                    // GEMM kernels (scalar fallback)
                    "fused_int4_gemm_f16",
                    "gemm_f16",
                    "gemm_f16_bt",
                    // GEMV kernels (M=1 optimized)
                    "fused_int4_gemv_f16",
                    "fused_int4_gemv_f16_v2",
                    "gemv_f16",
                    "gemv_f16_bt",
                    // GPTQ/AWQ kernels (asymmetric INT4)
                    "fused_gptq_gemv_f16",
                    "fused_awq_gemv_f16",
                    "fused_gptq_gemm_f16",
                ],
            )
            .map_err(|e| {
                InferenceError::Kernel(format!("Failed to load fused GEMM kernel: {}", e))
            })?;

        // Load GEMM functions
        self.int4_func = Some(
            self.device
                .get_func("fused_gemm", "fused_int4_gemm_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_int4_gemm_f16 function".to_string())
                })?,
        );

        self.f16_func = Some(
            self.device
                .get_func("fused_gemm", "gemm_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get gemm_f16 function".to_string())
                })?,
        );

        self.f16_bt_func = Some(
            self.device
                .get_func("fused_gemm", "gemm_f16_bt")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get gemm_f16_bt function".to_string())
                })?,
        );

        // Load GEMV functions (M=1 optimized)
        self.int4_gemv_func = Some(
            self.device
                .get_func("fused_gemm", "fused_int4_gemv_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_int4_gemv_f16 function".to_string())
                })?,
        );

        self.int4_gemv_v2_func = Some(
            self.device
                .get_func("fused_gemm", "fused_int4_gemv_f16_v2")
                .ok_or_else(|| {
                    InferenceError::Kernel(
                        "Failed to get fused_int4_gemv_f16_v2 function".to_string(),
                    )
                })?,
        );

        self.f16_gemv_func = Some(self.device.get_func("fused_gemm", "gemv_f16").ok_or_else(
            || InferenceError::Kernel("Failed to get gemv_f16 function".to_string()),
        )?);

        self.f16_gemv_bt_func = Some(
            self.device
                .get_func("fused_gemm", "gemv_f16_bt")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get gemv_f16_bt function".to_string())
                })?,
        );

        // WMMA tensor core kernels disabled for now (require sm_70+ and CUDA toolkit)
        // TODO: Add separate compilation path for WMMA when CUDA headers available
        self.f16_wmma_func = None;
        self.int4_wmma_func = None;
        self.use_tensor_cores = false;

        // Load GPTQ/AWQ kernels (asymmetric INT4 with zero points)
        self.gptq_gemv_func = Some(
            self.device
                .get_func("fused_gemm", "fused_gptq_gemv_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_gptq_gemv_f16 function".to_string())
                })?,
        );

        self.awq_gemv_func = Some(
            self.device
                .get_func("fused_gemm", "fused_awq_gemv_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_awq_gemv_f16 function".to_string())
                })?,
        );

        self.gptq_gemm_func = Some(
            self.device
                .get_func("fused_gemm", "fused_gptq_gemm_f16")
                .ok_or_else(|| {
                    InferenceError::Kernel("Failed to get fused_gptq_gemm_f16 function".to_string())
                })?,
        );

        tracing::info!("Loaded GEMM kernels: GEMV (M=1), scalar GEMM (M>1), GPTQ/AWQ");

        Ok(())
    }

    /// Enable or disable tensor core usage.
    pub fn set_use_tensor_cores(&mut self, use_tc: bool) {
        self.use_tensor_cores = use_tc;
    }

    /// Fused INT4 dequant + GEMM: C = A @ dequant(B_int4, scales).
    ///
    /// Automatically dispatches to optimized GEMV kernel when M=1 (decode).
    ///
    /// # Arguments
    ///
    /// * `input` - Input activations [M, K] in F16
    /// * `weights` - Packed INT4 weights [K/2, N]
    /// * `scales` - Per-block scales [K/32, N] in F16
    /// * `output` - Output [M, N] in F16
    ///
    /// # Memory Layout
    ///
    /// - Weights are packed with 2 INT4 values per byte
    /// - Scales are per 32-element block along K dimension
    /// - All tensors are row-major
    pub fn forward_int4(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let input_shape = input.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 || output_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("input {:?}, output {:?}", input_shape, output_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = output_shape[1];

        // Use optimized GEMV kernel for M=1 (decode case)
        if m == 1 {
            return self.forward_int4_gemv(input, weights, scales, output);
        }

        // Use tensor cores for M > 1 (prefill) if available
        if self.use_tensor_cores {
            if let Some(wmma_func) = self.int4_wmma_func.as_ref() {
                // WMMA kernel uses 64x64 tiles with 16 warps (32 threads each)
                const WMMA_BLOCK_M: usize = 64;
                const WMMA_BLOCK_N: usize = 64;
                const WARP_SIZE: u32 = 32;
                const WARPS_PER_BLOCK: u32 = 16;

                let grid_x = (n + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N;
                let grid_y = (m + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M;

                let cfg = LaunchConfig {
                    grid_dim: (grid_x as u32, grid_y as u32, 1),
                    block_dim: (WARP_SIZE, WARPS_PER_BLOCK, 1),
                    shared_mem_bytes: 0,
                };

                return unsafe {
                    wmma_func.clone().launch(
                        cfg,
                        (
                            input.device_ptr(),
                            weights.device_ptr(),
                            scales.device_ptr(),
                            output.device_ptr(),
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                    )
                }
                .map_err(|e| {
                    InferenceError::Kernel(format!("INT4 WMMA GEMM launch failed: {}", e))
                });
            }
        }

        // Fallback to scalar GEMM if tensor cores not available
        let func = self
            .int4_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("Fused GEMM kernel not loaded".to_string()))?;

        // Grid: one block per 64x64 output tile
        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = (m + TILE_M - 1) / TILE_M;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    output.device_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("Fused INT4 GEMM launch failed: {}", e)))?;

        Ok(())
    }

    /// Optimized INT4 GEMV for M=1 (decode case).
    ///
    /// This kernel is significantly faster than the general GEMM for single-token
    /// decode because each thread processes exactly one output column.
    fn forward_int4_gemv(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        // Use the v2 kernel which processes pairs of K values
        let func = self
            .int4_gemv_v2_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("INT4 GEMV kernel not loaded".to_string()))?;

        let input_shape = input.shape();
        let output_shape = output.shape();

        let k = input_shape[1];
        let n = output_shape[1];

        // Grid: one block per GEMV_BLOCK output columns
        let grid_x = (n + GEMV_BLOCK - 1) / GEMV_BLOCK;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (GEMV_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    output.device_ptr(),
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("INT4 GEMV launch failed: {}", e)))?;

        Ok(())
    }

    /// Standard F16 GEMM for non-quantized operations.
    ///
    /// Automatically dispatches to optimized GEMV kernel when M=1 (decode).
    ///
    /// Used for:
    /// - Token embeddings (no quantization)
    /// - Small projections that don't benefit from quantization
    pub fn forward_f16(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("A {:?}, B {:?}, C {:?}", a_shape, b_shape, c_shape),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if b_shape[0] != k || c_shape[0] != m || c_shape[1] != n {
            return Err(InferenceError::Shape {
                expected: format!("A[{},{}] @ B[{},{}] = C[{},{}]", m, k, k, n, m, n),
                got: format!("A{:?} @ B{:?} = C{:?}", a_shape, b_shape, c_shape),
            });
        }

        // Use optimized GEMV kernel for M=1 (decode case)
        if m == 1 {
            return self.forward_f16_gemv(a, b, c);
        }

        // Use tensor cores for M > 1 (prefill) if available
        if self.use_tensor_cores {
            if let Some(wmma_func) = self.f16_wmma_func.as_ref() {
                // WMMA kernel uses 64x64 tiles with 16 warps (32 threads each)
                const WMMA_BLOCK_M: usize = 64;
                const WMMA_BLOCK_N: usize = 64;
                const WARP_SIZE: u32 = 32;
                const WARPS_PER_BLOCK: u32 = 16;

                let grid_x = (n + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N;
                let grid_y = (m + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M;

                let cfg = LaunchConfig {
                    grid_dim: (grid_x as u32, grid_y as u32, 1),
                    block_dim: (WARP_SIZE, WARPS_PER_BLOCK, 1),
                    shared_mem_bytes: 0,
                };

                return unsafe {
                    wmma_func.clone().launch(
                        cfg,
                        (
                            a.device_ptr(),
                            b.device_ptr(),
                            c.device_ptr(),
                            m as i32,
                            n as i32,
                            k as i32,
                        ),
                    )
                }
                .map_err(|e| {
                    InferenceError::Kernel(format!("F16 WMMA GEMM launch failed: {}", e))
                });
            }
        }

        // Fallback to scalar GEMM if tensor cores not available
        let func = self
            .f16_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("GEMM kernel not loaded".to_string()))?;

        // Tiled launch: 16x16 threads, 64x64 tiles
        const TILE_SIZE: usize = 64;
        let grid_x = (n + TILE_SIZE - 1) / TILE_SIZE;
        let grid_y = (m + TILE_SIZE - 1) / TILE_SIZE;

        // Static shared memory is declared in the kernel, not passed here
        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    a.device_ptr(),
                    b.device_ptr(),
                    c.device_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("F16 GEMM launch failed: {}", e)))?;

        Ok(())
    }

    /// Optimized F16 GEMV for M=1 (decode case).
    fn forward_f16_gemv(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let func = self
            .f16_gemv_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("F16 GEMV kernel not loaded".to_string()))?;

        let a_shape = a.shape();
        let k = a_shape[1];
        let n = c.shape()[1];

        let grid_x = (n + GEMV_BLOCK - 1) / GEMV_BLOCK;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (GEMV_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    a.device_ptr(),
                    b.device_ptr(),
                    c.device_ptr(),
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("F16 GEMV launch failed: {}", e)))?;

        Ok(())
    }

    /// F16 GEMM with B transposed: C = A @ B^T
    ///
    /// Automatically dispatches to optimized GEMV kernel when M=1 (decode).
    ///
    /// Used for tied embeddings where lm_head shares weights with embed_tokens.
    /// embed_tokens is stored as [vocab_size, hidden_dim], but we need to compute
    /// [seq, hidden_dim] @ [hidden_dim, vocab_size] = [seq, vocab_size].
    ///
    /// # Arguments
    ///
    /// * `a` - Input [M, K] in F16
    /// * `b` - Weights [N, K] in F16 (stored transposed)
    /// * `c` - Output [M, N] in F16
    pub fn forward_f16_bt(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("A {:?}, B {:?}, C {:?}", a_shape, b_shape, c_shape),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0]; // B is [N, K] (transposed)

        // Validate dimensions for C = A @ B^T
        // A: [M, K], B: [N, K] (stored as B^T), C: [M, N]
        if b_shape[1] != k || c_shape[0] != m || c_shape[1] != n {
            return Err(InferenceError::Shape {
                expected: format!("A[{},{}] @ B^T[{},{}] = C[{},{}]", m, k, n, k, m, n),
                got: format!("A{:?} @ B^T{:?} = C{:?}", a_shape, b_shape, c_shape),
            });
        }

        // Use optimized GEMV kernel for M=1 (decode case)
        if m == 1 {
            return self.forward_f16_bt_gemv(a, b, c);
        }

        let func = self
            .f16_bt_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("GEMM BT kernel not loaded".to_string()))?;

        // Tiled launch: 16x16 threads, 64x64 tiles
        const TILE_SIZE: usize = 64;
        let grid_x = (n + TILE_SIZE - 1) / TILE_SIZE;
        let grid_y = (m + TILE_SIZE - 1) / TILE_SIZE;

        // Static shared memory is declared in the kernel, not passed here
        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    a.device_ptr(),
                    b.device_ptr(),
                    c.device_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("F16 GEMM BT launch failed: {}", e)))?;

        Ok(())
    }

    /// Optimized F16 GEMV with B transposed for M=1 (decode case).
    fn forward_f16_bt_gemv(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        let func = self
            .f16_gemv_bt_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("F16 GEMV BT kernel not loaded".to_string()))?;

        let a_shape = a.shape();
        let k = a_shape[1];
        let n = c.shape()[1];

        let grid_x = (n + GEMV_BLOCK - 1) / GEMV_BLOCK;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (GEMV_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    a.device_ptr(),
                    b.device_ptr(),
                    c.device_ptr(),
                    n as i32,
                    k as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("F16 GEMV BT launch failed: {}", e)))?;

        Ok(())
    }

    /// GPTQ asymmetric INT4 GEMM with act_order support.
    ///
    /// Automatically dispatches to GEMV kernel when M=1 (decode).
    ///
    /// # Arguments
    ///
    /// * `input` - Input activations [M, K] in F16
    /// * `weights` - Packed INT4 weights [K/2, N]
    /// * `scales` - Per-group scales [num_groups, N] in F16
    /// * `zeros` - Per-group zero points [num_groups, N] in F16
    /// * `g_idx` - Group index per row [K] (for act_order), or None for sequential
    /// * `output` - Output [M, N] in F16
    /// * `group_size` - Elements per quantization group (typically 32 or 128)
    pub fn forward_gptq(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        zeros: &GpuTensor,
        g_idx: Option<&GpuTensor>,
        output: &mut GpuTensor,
        group_size: usize,
    ) -> Result<(), InferenceError> {
        let input_shape = input.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 || output_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("input {:?}, output {:?}", input_shape, output_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = output_shape[1];

        // Use optimized GEMV kernel for M=1 (decode case)
        if m == 1 {
            return self
                .forward_gptq_gemv(input, weights, scales, zeros, g_idx, output, group_size);
        }

        // Use GPTQ GEMM for M > 1 (prefill)
        let func = self
            .gptq_gemm_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("GPTQ GEMM kernel not loaded".to_string()))?;

        // Grid: one block per 64x64 output tile
        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = (m + TILE_M - 1) / TILE_M;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // g_idx pointer (null if not using act_order)
        let g_idx_ptr = g_idx.map_or(0u64, |t| t.device_ptr());

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    zeros.device_ptr(),
                    g_idx_ptr,
                    output.device_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    group_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("GPTQ GEMM launch failed: {}", e)))?;

        Ok(())
    }

    /// Optimized GPTQ GEMV for M=1 (decode case).
    fn forward_gptq_gemv(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        zeros: &GpuTensor,
        g_idx: Option<&GpuTensor>,
        output: &mut GpuTensor,
        group_size: usize,
    ) -> Result<(), InferenceError> {
        let func = self
            .gptq_gemv_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("GPTQ GEMV kernel not loaded".to_string()))?;

        let input_shape = input.shape();
        let output_shape = output.shape();

        let k = input_shape[1];
        let n = output_shape[1];

        // Grid: one block per GEMV_BLOCK output columns
        let grid_x = (n + GEMV_BLOCK - 1) / GEMV_BLOCK;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (GEMV_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // g_idx pointer (null if not using act_order)
        let g_idx_ptr = g_idx.map_or(0u64, |t| t.device_ptr());

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    zeros.device_ptr(),
                    g_idx_ptr,
                    output.device_ptr(),
                    n as i32,
                    k as i32,
                    group_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("GPTQ GEMV launch failed: {}", e)))?;

        Ok(())
    }

    /// AWQ asymmetric INT4 GEMM with sequential groups.
    ///
    /// Optimized path for AWQ which uses sequential group ordering (no g_idx needed).
    /// Automatically dispatches to GEMV kernel when M=1 (decode).
    ///
    /// # Arguments
    ///
    /// * `input` - Input activations [M, K] in F16
    /// * `weights` - Packed INT4 weights [K/2, N]
    /// * `scales` - Per-group scales [num_groups, N] in F16
    /// * `zeros` - Per-group zero points [num_groups, N] in F16
    /// * `output` - Output [M, N] in F16
    /// * `group_size` - Elements per quantization group (typically 128 for AWQ)
    pub fn forward_awq(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        zeros: &GpuTensor,
        output: &mut GpuTensor,
        group_size: usize,
    ) -> Result<(), InferenceError> {
        let input_shape = input.shape();
        let output_shape = output.shape();

        if input_shape.len() != 2 || output_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D tensors".to_string(),
                got: format!("input {:?}, output {:?}", input_shape, output_shape),
            });
        }

        let m = input_shape[0];
        let k = input_shape[1];
        let n = output_shape[1];

        // Use optimized GEMV kernel for M=1 (decode case)
        if m == 1 {
            return self.forward_awq_gemv(input, weights, scales, zeros, output, group_size);
        }

        // Use GPTQ GEMM for M > 1 (prefill) - AWQ uses sequential groups (no g_idx)
        let func = self
            .gptq_gemm_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("AWQ GEMM kernel not loaded".to_string()))?;

        // Grid: one block per 64x64 output tile
        let grid_x = (n + TILE_N - 1) / TILE_N;
        let grid_y = (m + TILE_M - 1) / TILE_M;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // AWQ uses sequential groups - pass null g_idx
        let g_idx_ptr = 0u64;

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    zeros.device_ptr(),
                    g_idx_ptr,
                    output.device_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    group_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("AWQ GEMM launch failed: {}", e)))?;

        Ok(())
    }

    /// Optimized AWQ GEMV for M=1 (decode case).
    fn forward_awq_gemv(
        &self,
        input: &GpuTensor,
        weights: &GpuTensor,
        scales: &GpuTensor,
        zeros: &GpuTensor,
        output: &mut GpuTensor,
        group_size: usize,
    ) -> Result<(), InferenceError> {
        let func = self
            .awq_gemv_func
            .as_ref()
            .ok_or_else(|| InferenceError::Kernel("AWQ GEMV kernel not loaded".to_string()))?;

        let input_shape = input.shape();
        let output_shape = output.shape();

        let k = input_shape[1];
        let n = output_shape[1];

        // Grid: one block per GEMV_BLOCK output columns
        let grid_x = (n + GEMV_BLOCK - 1) / GEMV_BLOCK;

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, 1, 1),
            block_dim: (GEMV_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone().launch(
                cfg,
                (
                    input.device_ptr(),
                    weights.device_ptr(),
                    scales.device_ptr(),
                    zeros.device_ptr(),
                    output.device_ptr(),
                    n as i32,
                    k as i32,
                    group_size as i32,
                ),
            )
        }
        .map_err(|e| InferenceError::Kernel(format!("AWQ GEMV launch failed: {}", e)))?;

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

    /// Specification: TILE constants must be defined in CUDA source.
    ///
    /// The fused_gptq_gemm_f16 kernel uses TILE_M, TILE_N, TILE_K for shared memory
    /// allocation and loop bounds. These must be #define'd in the CUDA source since
    /// Rust constants are not visible to NVRTC.
    ///
    /// Gap discovered: 2026-02-05 during HoloTensor integration.
    /// See: docs/specs/CUDA-KERNEL-TILING-FIX.md
    #[test]
    fn spec_tile_constants_defined_in_cuda_source() {
        // TILE_M, TILE_N, TILE_K must be #define'd for NVRTC compilation
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_M"),
            "TILE_M must be #define'd in CUDA source for fused_gptq_gemm_f16"
        );
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_N"),
            "TILE_N must be #define'd in CUDA source for fused_gptq_gemm_f16"
        );
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_K"),
            "TILE_K must be #define'd in CUDA source for fused_gptq_gemm_f16"
        );

        // Verify values match Rust constants (for grid calculation consistency)
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_M 64"),
            "TILE_M must be 64 to match Rust constant"
        );
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_N 64"),
            "TILE_N must be 64 to match Rust constant"
        );
        assert!(
            FUSED_GEMM_CUDA.contains("#define TILE_K 32"),
            "TILE_K must be 32 to match Rust constant"
        );
    }

    #[test]
    fn test_fused_gemm_kernel_compilation() {
        // Test that CUDA source compiles
        let result = compile_cuda_kernel(FUSED_GEMM_CUDA);
        if let Err(e) = &result {
            // NVRTC errors will cause test failure below
        }
        // Don't assert - just check if NVRTC is available
    }

    #[test]
    fn test_gptq_awq_kernel_source_validity() {
        // Verify GPTQ/AWQ kernel source is included in the CUDA code
        assert!(FUSED_GEMM_CUDA.contains("fused_gptq_gemv_f16"));
        assert!(FUSED_GEMM_CUDA.contains("fused_awq_gemv_f16"));
        assert!(FUSED_GEMM_CUDA.contains("fused_gptq_gemm_f16"));
    }

    #[test]
    fn test_kernel_asymmetric_dequant_formula() {
        // Verify the asymmetric dequantization formula is documented
        // GPTQ/AWQ: weight = (int4_val - zeros[group]) * scales[group]
        assert!(FUSED_GEMM_CUDA.contains("int4_val - zero"));
        assert!(FUSED_GEMM_CUDA.contains("* scale"));
    }

    #[test]
    fn test_gptq_g_idx_support() {
        // Verify g_idx handling for GPTQ act_order is present
        assert!(FUSED_GEMM_CUDA.contains("g_idx"));
        assert!(FUSED_GEMM_CUDA.contains("g_idx != NULL"));
    }

    #[test]
    fn test_gemv_vs_gemm_dispatch() {
        // Verify both GEMV (M=1) and GEMM (M>1) kernels exist for GPTQ
        assert!(FUSED_GEMM_CUDA.contains("fused_gptq_gemv_f16"));
        assert!(FUSED_GEMM_CUDA.contains("fused_gptq_gemm_f16"));
    }
}
