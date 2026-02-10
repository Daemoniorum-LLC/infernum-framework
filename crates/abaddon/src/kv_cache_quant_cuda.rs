//! CUDA-accelerated INT8 KV cache operations.
//!
//! Provides fused kernels that compute attention directly on INT8 K/V tensors,
//! avoiding the overhead of full dequantization before attention.
//!
//! ## Performance Benefits
//!
//! - **Reduced memory bandwidth**: INT8 K/V is 2x smaller than BF16
//! - **Fused dequantization**: Dequantize on-the-fly per element, not full tensor
//! - **Better cache utilization**: Smaller data fits better in GPU L2 cache
//!
//! ## Algorithm
//!
//! Standard approach (slow):
//! 1. Dequantize K: K_bf16 = (K_u8 - 128) * scale  [full tensor copy]
//! 2. Dequantize V: V_bf16 = (V_u8 - 128) * scale  [full tensor copy]
//! 3. Compute Q @ K^T
//! 4. Softmax
//! 5. Compute attn @ V
//!
//! Fused approach (fast):
//! 1. Compute Q @ K^T with on-the-fly dequantization of K
//! 2. Softmax
//! 3. Compute attn @ V with on-the-fly dequantization of V

/// CUDA-accelerated INT8 KV cache quantization and fused attention.
#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

    /// CUDA kernel source for fused INT8 attention.
    ///
    /// Uses manual BF16<->F32 conversion to avoid dependency on cuda_bf16.h
    /// BF16 is just the top 16 bits of F32, so conversion is simple bit manipulation.
    ///
    /// Optimized kernels use:
    /// - Tiling with shared memory
    /// - Vectorized loads (4 bytes at a time)
    /// - Parallel reduction
    /// - Better thread/block configuration
    const INT8_ATTENTION_KERNEL_SRC: &str = r#"
// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float bf16_to_f32(unsigned short bf16) {
    unsigned int bits = ((unsigned int)bf16) << 16;
    return __int_as_float(bits);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    unsigned int bits = __float_as_int(f);
    bits += 0x7FFF + ((bits >> 16) & 1);
    return (unsigned short)(bits >> 16);
}

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Basic dequantization kernel (vectorized)
// ============================================================================
extern "C" __global__ void int8_dequant_bf16(
    const unsigned char* __restrict__ quant,
    const unsigned short* __restrict__ scales,
    unsigned short* __restrict__ out,
    int num_elements,
    int elements_per_scale
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= num_elements) return;

    int scale_idx = idx / elements_per_scale;
    float scale = bf16_to_f32(scales[scale_idx]);

    // Vectorized load - 4 bytes at a time
    if (idx + 3 < num_elements) {
        uchar4 q4 = *reinterpret_cast<const uchar4*>(&quant[idx]);
        ushort4 o4;
        o4.x = f32_to_bf16(((float)q4.x - 128.0f) * scale);
        o4.y = f32_to_bf16(((float)q4.y - 128.0f) * scale);
        o4.z = f32_to_bf16(((float)q4.z - 128.0f) * scale);
        o4.w = f32_to_bf16(((float)q4.w - 128.0f) * scale);
        *reinterpret_cast<ushort4*>(&out[idx]) = o4;
    } else {
        // Handle tail
        for (int i = 0; i < 4 && idx + i < num_elements; i++) {
            float val = ((float)quant[idx + i] - 128.0f) * scale;
            out[idx + i] = f32_to_bf16(val);
        }
    }
}

// ============================================================================
// Naive Q @ K^T kernel (for small sequences, kept for compatibility)
// ============================================================================
extern "C" __global__ void int8_qk_attention(
    const unsigned short* __restrict__ q,
    const unsigned char* __restrict__ k_quant,
    const unsigned short* __restrict__ k_scale,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float attn_scale
) {
    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int kv_pos = threadIdx.x;

    if (b >= batch_size || h >= num_heads || q_pos >= q_len || kv_pos >= kv_len) return;

    int kv_h = h * num_kv_heads / num_heads;
    int q_offset = ((b * num_heads + h) * q_len + q_pos) * head_dim;
    int k_offset = ((b * num_kv_heads + kv_h) * kv_len + kv_pos) * head_dim;
    int scale_offset = (b * num_kv_heads + kv_h) * kv_len + kv_pos;

    float k_s = bf16_to_f32(k_scale[scale_offset]);

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float q_val = bf16_to_f32(q[q_offset + d]);
        float k_val = ((float)k_quant[k_offset + d] - 128.0f) * k_s;
        sum += q_val * k_val;
    }

    int out_idx = ((b * num_heads + h) * q_len + q_pos) * kv_len + kv_pos;
    out[out_idx] = sum * attn_scale;
}

// ============================================================================
// Optimized Q @ K^T with tiling and shared memory
// Each block: processes one (batch, head, q_pos) and tiles over kv_len
// Block dim: (TILE_KV, 1, 1) where TILE_KV = 128
// Shared memory: Q vector (head_dim) + K scale tile
// ============================================================================
#define TILE_KV 128
#define TILE_HD 32

extern "C" __global__ void int8_qk_attention_tiled(
    const unsigned short* __restrict__ q,
    const unsigned char* __restrict__ k_quant,
    const unsigned short* __restrict__ k_scale,
    float* __restrict__ out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float attn_scale
) {
    // Shared memory for Q vector and K scales
    extern __shared__ float shared[];
    float* s_q = shared;                    // [head_dim]
    float* s_k_scale = &shared[head_dim];   // [TILE_KV]

    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    if (b >= batch_size || h >= num_heads || q_pos >= q_len) return;

    int kv_h = h * num_kv_heads / num_heads;
    int q_offset = ((b * num_heads + h) * q_len + q_pos) * head_dim;
    int k_base = (b * num_kv_heads + kv_h) * kv_len * head_dim;
    int scale_base = (b * num_kv_heads + kv_h) * kv_len;
    int out_base = ((b * num_heads + h) * q_len + q_pos) * kv_len;

    // Load Q vector into shared memory (cooperative load)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = bf16_to_f32(q[q_offset + d]);
    }
    __syncthreads();

    // Process KV in tiles
    for (int kv_tile = 0; kv_tile < kv_len; kv_tile += TILE_KV) {
        int kv_pos = kv_tile + tid;

        // Load scales for this tile
        if (kv_pos < kv_len) {
            s_k_scale[tid] = bf16_to_f32(k_scale[scale_base + kv_pos]);
        }
        __syncthreads();

        if (kv_pos < kv_len) {
            float k_s = s_k_scale[tid];
            int k_offset = k_base + kv_pos * head_dim;

            // Compute dot product with vectorized loads
            float sum = 0.0f;

            // Process 4 elements at a time
            int d = 0;
            for (; d + 3 < head_dim; d += 4) {
                uchar4 k4 = *reinterpret_cast<const uchar4*>(&k_quant[k_offset + d]);

                float k0 = ((float)k4.x - 128.0f) * k_s;
                float k1 = ((float)k4.y - 128.0f) * k_s;
                float k2 = ((float)k4.z - 128.0f) * k_s;
                float k3 = ((float)k4.w - 128.0f) * k_s;

                sum += s_q[d] * k0 + s_q[d+1] * k1 + s_q[d+2] * k2 + s_q[d+3] * k3;
            }

            // Handle remaining elements
            for (; d < head_dim; d++) {
                float k_val = ((float)k_quant[k_offset + d] - 128.0f) * k_s;
                sum += s_q[d] * k_val;
            }

            out[out_base + kv_pos] = sum * attn_scale;
        }
        __syncthreads();
    }
}

// ============================================================================
// Naive attn @ V kernel (for compatibility)
// ============================================================================
extern "C" __global__ void int8_attn_v(
    const float* __restrict__ attn,
    const unsigned char* __restrict__ v_quant,
    const unsigned short* __restrict__ v_scale,
    unsigned short* __restrict__ out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_len,
    int kv_len,
    int head_dim
) {
    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int d = threadIdx.x;

    if (b >= batch_size || h >= num_heads || q_pos >= q_len || d >= head_dim) return;

    int kv_h = h * num_kv_heads / num_heads;
    int attn_offset = ((b * num_heads + h) * q_len + q_pos) * kv_len;
    int v_base = (b * num_kv_heads + kv_h) * kv_len * head_dim;
    int scale_base = (b * num_kv_heads + kv_h) * kv_len;

    float sum = 0.0f;
    for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        float a = attn[attn_offset + kv_pos];
        float v_s = bf16_to_f32(v_scale[scale_base + kv_pos]);
        float v_val = ((float)v_quant[v_base + kv_pos * head_dim + d] - 128.0f) * v_s;
        sum += a * v_val;
    }

    int out_idx = ((b * num_heads + h) * q_len + q_pos) * head_dim + d;
    out[out_idx] = f32_to_bf16(sum);
}

// ============================================================================
// Optimized attn @ V with tiling
// Each block: one (batch, head, q_pos), threads cooperate on head_dim
// Tiles over kv_len to cache attention weights
// ============================================================================
#define TILE_KV_AV 64

extern "C" __global__ void int8_attn_v_tiled(
    const float* __restrict__ attn,
    const unsigned char* __restrict__ v_quant,
    const unsigned short* __restrict__ v_scale,
    unsigned short* __restrict__ out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int q_len,
    int kv_len,
    int head_dim
) {
    // Shared memory for attention weights and scales
    extern __shared__ float shared_av[];
    float* s_attn = shared_av;                  // [TILE_KV_AV]
    float* s_scale = &shared_av[TILE_KV_AV];    // [TILE_KV_AV]

    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int d = threadIdx.x;  // dimension index

    if (b >= batch_size || h >= num_heads || q_pos >= q_len) return;

    int kv_h = h * num_kv_heads / num_heads;
    int attn_offset = ((b * num_heads + h) * q_len + q_pos) * kv_len;
    int v_base = (b * num_kv_heads + kv_h) * kv_len * head_dim;
    int scale_base = (b * num_kv_heads + kv_h) * kv_len;

    float sum = 0.0f;

    // Process kv_len in tiles
    for (int kv_tile = 0; kv_tile < kv_len; kv_tile += TILE_KV_AV) {
        // Cooperative load of attention weights and scales
        if (d < TILE_KV_AV && kv_tile + d < kv_len) {
            s_attn[d] = attn[attn_offset + kv_tile + d];
            s_scale[d] = bf16_to_f32(v_scale[scale_base + kv_tile + d]);
        }
        __syncthreads();

        // Each thread processes its dimension
        if (d < head_dim) {
            int tile_end = min(TILE_KV_AV, kv_len - kv_tile);
            for (int t = 0; t < tile_end; t++) {
                int kv_pos = kv_tile + t;
                float a = s_attn[t];
                float v_s = s_scale[t];
                float v_val = ((float)v_quant[v_base + kv_pos * head_dim + d] - 128.0f) * v_s;
                sum += a * v_val;
            }
        }
        __syncthreads();
    }

    if (d < head_dim) {
        int out_idx = ((b * num_heads + h) * q_len + q_pos) * head_dim + d;
        out[out_idx] = f32_to_bf16(sum);
    }
}

// ============================================================================
// Highly optimized Q @ K^T for decode (single q_pos, many kv)
// Uses warp-level parallelism for dot product reduction
// Each warp computes one attention score
// ============================================================================
#define WARP_SIZE 32

extern "C" __global__ void int8_qk_attention_decode(
    const unsigned short* __restrict__ q,      // [batch, heads, 1, head_dim]
    const unsigned char* __restrict__ k_quant, // [batch, kv_heads, kv_len, head_dim]
    const unsigned short* __restrict__ k_scale,// [batch, kv_heads, kv_len]
    float* __restrict__ out,                   // [batch, heads, 1, kv_len]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int kv_len,
    int head_dim,
    float attn_scale
) {
    // One warp per kv position
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int kv_pos = warp_id % kv_len;
    int h = (warp_id / kv_len) % num_heads;
    int b = warp_id / (kv_len * num_heads);

    if (b >= batch_size) return;

    int kv_h = h * num_kv_heads / num_heads;
    int q_offset = (b * num_heads + h) * head_dim;
    int k_offset = ((b * num_kv_heads + kv_h) * kv_len + kv_pos) * head_dim;
    int scale_offset = (b * num_kv_heads + kv_h) * kv_len + kv_pos;

    float k_s = bf16_to_f32(k_scale[scale_offset]);

    // Each lane processes head_dim/32 elements
    float sum = 0.0f;
    for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
        float q_val = bf16_to_f32(q[q_offset + d]);
        float k_val = ((float)k_quant[k_offset + d] - 128.0f) * k_s;
        sum += q_val * k_val;
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    // Lane 0 writes result
    if (lane_id == 0) {
        int out_idx = (b * num_heads + h) * kv_len + kv_pos;
        out[out_idx] = sum * attn_scale;
    }
}
"#;

    /// Context for CUDA INT8 attention operations.
    pub struct Int8AttentionContext {
        device: Arc<CudaDevice>,
        kernels_loaded: bool,
    }

    impl Int8AttentionContext {
        /// Create a new INT8 attention context.
        pub fn new(device_id: usize) -> Result<Self, Int8AttentionError> {
            let device =
                CudaDevice::new(device_id).map_err(|e| Int8AttentionError::DeviceInit {
                    device_id,
                    message: e.to_string(),
                })?;

            Ok(Self {
                device,
                kernels_loaded: false,
            })
        }

        /// Load CUDA kernels via nvrtc compilation.
        pub fn load_kernels(&mut self) -> Result<(), Int8AttentionError> {
            if self.kernels_loaded {
                return Ok(());
            }

            // Compile CUDA source to PTX using nvrtc
            let ptx = cudarc::nvrtc::compile_ptx(INT8_ATTENTION_KERNEL_SRC).map_err(|e| {
                Int8AttentionError::KernelCompile {
                    message: e.to_string(),
                }
            })?;

            // Load all kernels (naive + optimized)
            self.device
                .load_ptx(
                    ptx,
                    "int8_attention",
                    &[
                        "int8_dequant_bf16",
                        "int8_qk_attention",        // Naive Q @ K^T
                        "int8_qk_attention_tiled",  // Optimized with shared memory
                        "int8_qk_attention_decode", // Warp-optimized decode
                        "int8_attn_v",              // Naive attn @ V
                        "int8_attn_v_tiled",        // Optimized with tiling
                    ],
                )
                .map_err(|e| Int8AttentionError::KernelCompile {
                    message: format!("Failed to load PTX: {}", e),
                })?;

            self.kernels_loaded = true;
            Ok(())
        }

        /// Check if fused kernels are available.
        pub fn has_fused_kernels(&self) -> bool {
            self.kernels_loaded
        }

        /// Get device reference.
        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }

        /// Dequantize INT8 to BF16 on GPU.
        ///
        /// # Arguments
        /// * `quant` - INT8 quantized data
        /// * `scales` - BF16 scales (one per `elements_per_scale` elements)
        /// * `elements_per_scale` - Number of elements sharing each scale (typically head_dim)
        pub fn dequant_int8_to_bf16(
            &self,
            quant: &CudaSlice<u8>,
            scales: &CudaSlice<u16>, // BF16 as u16
            elements_per_scale: usize,
        ) -> Result<CudaSlice<u16>, Int8AttentionError> {
            if !self.kernels_loaded {
                return Err(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_dequant_bf16".to_string(),
                });
            }

            let num_elements = quant.len();
            let output: CudaSlice<u16> = self.device.alloc_zeros(num_elements).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: format!("Failed to allocate output: {}", e),
                }
            })?;

            let kernel = self
                .device
                .get_func("int8_attention", "int8_dequant_bf16")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_dequant_bf16".to_string(),
                })?;

            let threads_per_block = 256;
            let blocks = (num_elements + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            quant,
                            scales,
                            &output,
                            num_elements as i32,
                            elements_per_scale as i32,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }

            Ok(output)
        }

        // Kernel selection thresholds
        //
        // For Q@K: tiled kernel caches Q in shared memory, avoiding redundant global reads.
        // This is beneficial when there are many threads per block (kv_len > TILE_KV).
        //
        // For attn@V: naive kernel is used (tiled V has too much sync overhead).
        const TILED_QK_THRESHOLD: usize = 128; // Use tiled kernel for kv_len > TILE_KV
        const TILE_KV: usize = 128; // Must match TILE_KV in kernel
        #[allow(dead_code)]
        const TILE_KV_AV: usize = 64; // Must match TILE_KV_AV in kernel
        const WARP_SIZE: usize = 32;

        /// Compute Q @ K^T with fused INT8 dequantization.
        ///
        /// Auto-selects the optimal kernel based on input dimensions:
        /// - q_len=1, kv_len>512: decode kernel (warp-level parallelism)
        /// - kv_len>256: tiled kernel (shared memory, vectorized)
        /// - Otherwise: naive kernel
        ///
        /// # Arguments
        /// * `q` - Query tensor BF16 [batch, heads, q_len, head_dim]
        /// * `k_quant` - Key tensor INT8 [batch, kv_heads, kv_len, head_dim]
        /// * `k_scale` - Key scales BF16 [batch, kv_heads, kv_len]
        /// * `batch_size`, `num_heads`, etc. - Tensor dimensions
        /// * `attn_scale` - Attention scale factor (1/sqrt(head_dim))
        ///
        /// # Returns
        /// Attention scores F32 [batch, heads, q_len, kv_len]
        #[allow(clippy::too_many_arguments)]
        pub fn fused_qk_attention(
            &self,
            q: &CudaSlice<u16>,       // BF16
            k_quant: &CudaSlice<u8>,  // INT8
            k_scale: &CudaSlice<u16>, // BF16
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
            attn_scale: f32,
        ) -> Result<CudaSlice<f32>, Int8AttentionError> {
            if !self.kernels_loaded {
                return Err(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_qk_attention".to_string(),
                });
            }

            let output_size = batch_size * num_heads * q_len * kv_len;
            let output: CudaSlice<f32> = self.device.alloc_zeros(output_size).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: format!("Failed to allocate output: {}", e),
                }
            })?;

            // Select optimal kernel based on dimensions
            //
            // Strategy:
            // 1. Decode (q_len=1): Use warp-based decode kernel for parallel dot products
            // 2. Prefill (q_len>1, kv_len large): Use tiled kernel - caches Q in shared memory
            //    to avoid redundant global memory reads across threads
            // 3. Small cases: Use naive kernel - low overhead for small workloads
            //
            if q_len == 1 {
                // Single-token decode: use warp-parallel kernel
                self.launch_qk_decode(
                    q,
                    k_quant,
                    k_scale,
                    &output,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    kv_len,
                    head_dim,
                    attn_scale,
                )?;
            } else if kv_len > Self::TILED_QK_THRESHOLD {
                // Prefill with large context: tiled kernel caches Q in shared memory
                self.launch_qk_tiled(
                    q,
                    k_quant,
                    k_scale,
                    &output,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    attn_scale,
                )?;
            } else {
                // Small sequences: naive kernel has lower overhead
                self.launch_qk_naive(
                    q,
                    k_quant,
                    k_scale,
                    &output,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    attn_scale,
                )?;
            }

            Ok(output)
        }

        /// Launch naive Q @ K^T kernel (small sequences).
        #[allow(clippy::too_many_arguments)]
        fn launch_qk_naive(
            &self,
            q: &CudaSlice<u16>,
            k_quant: &CudaSlice<u8>,
            k_scale: &CudaSlice<u16>,
            output: &CudaSlice<f32>,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
            attn_scale: f32,
        ) -> Result<(), Int8AttentionError> {
            let kernel = self
                .device
                .get_func("int8_attention", "int8_qk_attention")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_qk_attention".to_string(),
                })?;

            let kv_len_block = kv_len.min(1024);
            let config = LaunchConfig {
                block_dim: (kv_len_block as u32, 1, 1),
                grid_dim: (q_len as u32, num_heads as u32, batch_size as u32),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            q,
                            k_quant,
                            k_scale,
                            output,
                            batch_size as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            q_len as i32,
                            kv_len as i32,
                            head_dim as i32,
                            attn_scale,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }
            Ok(())
        }

        /// Launch tiled Q @ K^T kernel (large sequences).
        #[allow(clippy::too_many_arguments)]
        fn launch_qk_tiled(
            &self,
            q: &CudaSlice<u16>,
            k_quant: &CudaSlice<u8>,
            k_scale: &CudaSlice<u16>,
            output: &CudaSlice<f32>,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
            attn_scale: f32,
        ) -> Result<(), Int8AttentionError> {
            let kernel = self
                .device
                .get_func("int8_attention", "int8_qk_attention_tiled")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_qk_attention_tiled".to_string(),
                })?;

            // Shared memory: Q vector (head_dim floats) + K scales (TILE_KV floats)
            let shared_mem_bytes = ((head_dim + Self::TILE_KV) * std::mem::size_of::<f32>()) as u32;

            let config = LaunchConfig {
                block_dim: (Self::TILE_KV as u32, 1, 1),
                grid_dim: (q_len as u32, num_heads as u32, batch_size as u32),
                shared_mem_bytes,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            q,
                            k_quant,
                            k_scale,
                            output,
                            batch_size as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            q_len as i32,
                            kv_len as i32,
                            head_dim as i32,
                            attn_scale,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }
            Ok(())
        }

        /// Launch decode-optimized Q @ K^T kernel (single token generation).
        #[allow(clippy::too_many_arguments)]
        fn launch_qk_decode(
            &self,
            q: &CudaSlice<u16>,
            k_quant: &CudaSlice<u8>,
            k_scale: &CudaSlice<u16>,
            output: &CudaSlice<f32>,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            kv_len: usize,
            head_dim: usize,
            attn_scale: f32,
        ) -> Result<(), Int8AttentionError> {
            let kernel = self
                .device
                .get_func("int8_attention", "int8_qk_attention_decode")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_qk_attention_decode".to_string(),
                })?;

            // One warp per kv position
            let total_warps = batch_size * num_heads * kv_len;
            let threads_per_block = 256; // 8 warps per block
            let blocks =
                (total_warps * Self::WARP_SIZE + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            q,
                            k_quant,
                            k_scale,
                            output,
                            batch_size as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            kv_len as i32,
                            head_dim as i32,
                            attn_scale,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }
            Ok(())
        }

        /// Compute attn @ V with fused INT8 dequantization.
        ///
        /// Auto-selects optimal kernel:
        /// - kv_len > 256: tiled kernel (caches attn weights and scales)
        /// - Otherwise: naive kernel
        ///
        /// # Arguments
        /// * `attn` - Attention weights F32 [batch, heads, q_len, kv_len] (after softmax)
        /// * `v_quant` - Value tensor INT8 [batch, kv_heads, kv_len, head_dim]
        /// * `v_scale` - Value scales BF16 [batch, kv_heads, kv_len]
        ///
        /// # Returns
        /// Output BF16 [batch, heads, q_len, head_dim]
        #[allow(clippy::too_many_arguments)]
        pub fn fused_attn_v(
            &self,
            attn: &CudaSlice<f32>,
            v_quant: &CudaSlice<u8>,  // INT8
            v_scale: &CudaSlice<u16>, // BF16
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
        ) -> Result<CudaSlice<u16>, Int8AttentionError> {
            if !self.kernels_loaded {
                return Err(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_attn_v".to_string(),
                });
            }

            let output_size = batch_size * num_heads * q_len * head_dim;
            let output: CudaSlice<u16> = self.device.alloc_zeros(output_size).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: format!("Failed to allocate output: {}", e),
                }
            })?;

            // The tiled V kernel has excessive sync barrier overhead for large kv_len.
            // Use naive kernel which scales better due to no synchronization.
            // TODO: Implement a parallel-reduction V kernel for better scaling.
            self.launch_attn_v_naive(
                attn,
                v_quant,
                v_scale,
                &output,
                batch_size,
                num_heads,
                num_kv_heads,
                q_len,
                kv_len,
                head_dim,
            )?;

            Ok(output)
        }

        /// Launch naive attn @ V kernel.
        #[allow(clippy::too_many_arguments)]
        fn launch_attn_v_naive(
            &self,
            attn: &CudaSlice<f32>,
            v_quant: &CudaSlice<u8>,
            v_scale: &CudaSlice<u16>,
            output: &CudaSlice<u16>,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
        ) -> Result<(), Int8AttentionError> {
            let kernel = self
                .device
                .get_func("int8_attention", "int8_attn_v")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_attn_v".to_string(),
                })?;

            let head_dim_block = head_dim.min(1024);
            let config = LaunchConfig {
                block_dim: (head_dim_block as u32, 1, 1),
                grid_dim: (q_len as u32, num_heads as u32, batch_size as u32),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            attn,
                            v_quant,
                            v_scale,
                            output,
                            batch_size as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            q_len as i32,
                            kv_len as i32,
                            head_dim as i32,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }
            Ok(())
        }

        /// Launch tiled attn @ V kernel with shared memory.
        #[allow(clippy::too_many_arguments)]
        #[allow(dead_code)]
        fn launch_attn_v_tiled(
            &self,
            attn: &CudaSlice<f32>,
            v_quant: &CudaSlice<u8>,
            v_scale: &CudaSlice<u16>,
            output: &CudaSlice<u16>,
            batch_size: usize,
            num_heads: usize,
            num_kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
        ) -> Result<(), Int8AttentionError> {
            let kernel = self
                .device
                .get_func("int8_attention", "int8_attn_v_tiled")
                .ok_or(Int8AttentionError::KernelNotLoaded {
                    kernel: "int8_attn_v_tiled".to_string(),
                })?;

            // Shared memory: attn weights (TILE_KV_AV) + scales (TILE_KV_AV)
            let shared_mem_bytes = (2 * Self::TILE_KV_AV * std::mem::size_of::<f32>()) as u32;

            // Use head_dim threads per block (each thread computes one output dimension)
            let head_dim_block = head_dim.min(1024);
            let config = LaunchConfig {
                block_dim: (head_dim_block as u32, 1, 1),
                grid_dim: (q_len as u32, num_heads as u32, batch_size as u32),
                shared_mem_bytes,
            };

            unsafe {
                kernel
                    .launch(
                        config,
                        (
                            attn,
                            v_quant,
                            v_scale,
                            output,
                            batch_size as i32,
                            num_heads as i32,
                            num_kv_heads as i32,
                            q_len as i32,
                            kv_len as i32,
                            head_dim as i32,
                        ),
                    )
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;
            }
            Ok(())
        }
    }

    /// Errors for INT8 attention operations.
    #[derive(Debug, thiserror::Error)]
    pub enum Int8AttentionError {
        /// Device initialization error.
        #[error("Failed to initialize CUDA device {device_id}: {message}")]
        DeviceInit {
            /// CUDA device ID.
            device_id: usize,
            /// Error message.
            message: String,
        },
        /// Kernel compilation error.
        #[error("Failed to compile kernel: {message}")]
        KernelCompile {
            /// Error message.
            message: String,
        },
        /// Kernel not loaded error.
        #[error("Kernel not loaded: {kernel}")]
        KernelNotLoaded {
            /// Kernel name.
            kernel: String,
        },
        /// Kernel execution error.
        #[error("Kernel execution failed: {message}")]
        KernelExec {
            /// Error message.
            message: String,
        },
        /// Tensor conversion error.
        #[error("Tensor conversion failed: {message}")]
        TensorConvert {
            /// Error message.
            message: String,
        },
    }

    /// CUDA-accelerated INT8 KV cache with fused attention kernels.
    ///
    /// Stores K/V directly in GPU memory as INT8 with per-token BF16 scales.
    /// Uses fused CUDA kernels to compute attention without full dequantization.
    pub struct CudaQuantizedKvCache {
        /// INT8 quantized keys: [batch, kv_heads, seq_len, head_dim]
        k_quant: Option<CudaSlice<u8>>,
        /// INT8 quantized values: [batch, kv_heads, seq_len, head_dim]
        v_quant: Option<CudaSlice<u8>>,
        /// BF16 key scales: [batch, kv_heads, seq_len]
        k_scales: Option<CudaSlice<u16>>,
        /// BF16 value scales: [batch, kv_heads, seq_len]
        v_scales: Option<CudaSlice<u16>>,
        /// CUDA attention context with compiled kernels.
        attn_ctx: Int8AttentionContext,
        /// Number of KV heads.
        num_kv_heads: usize,
        /// Head dimension.
        #[allow(dead_code)]
        head_dim: usize,
        /// Current sequence length.
        seq_len: usize,
        /// Batch size.
        batch_size: usize,
    }

    impl CudaQuantizedKvCache {
        /// Create a new CUDA quantized KV cache.
        pub fn new(
            num_kv_heads: usize,
            head_dim: usize,
            device_id: usize,
        ) -> Result<Self, Int8AttentionError> {
            let mut attn_ctx = Int8AttentionContext::new(device_id)?;
            attn_ctx.load_kernels()?;

            Ok(Self {
                k_quant: None,
                v_quant: None,
                k_scales: None,
                v_scales: None,
                attn_ctx,
                num_kv_heads,
                head_dim,
                seq_len: 0,
                batch_size: 0,
            })
        }

        /// Append new K/V tensors to the cache.
        ///
        /// Input tensors should be in BF16 format: [batch, kv_heads, new_seq_len, head_dim]
        /// They will be quantized to INT8 and stored in GPU memory.
        pub fn append(
            &mut self,
            k: &candle_core::Tensor,
            v: &candle_core::Tensor,
        ) -> Result<(), Int8AttentionError> {
            let device = self.attn_ctx.device();

            // Get dimensions
            let dims = k.dims();
            if dims.len() != 4 {
                return Err(Int8AttentionError::TensorConvert {
                    message: format!("Expected 4D tensor, got {}D", dims.len()),
                });
            }
            let (batch, kv_heads, new_seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

            if self.seq_len == 0 {
                self.batch_size = batch;
            }

            // Quantize K and V to INT8 with per-token scales
            let (k_quant_new, k_scales_new) = self.quantize_tensor(k)?;
            let (v_quant_new, v_scales_new) = self.quantize_tensor(v)?;

            // Transfer to GPU
            let k_quant_gpu = device.htod_sync_copy(&k_quant_new).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: e.to_string(),
                }
            })?;
            let v_quant_gpu = device.htod_sync_copy(&v_quant_new).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: e.to_string(),
                }
            })?;
            let k_scales_gpu = device.htod_sync_copy(&k_scales_new).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: e.to_string(),
                }
            })?;
            let v_scales_gpu = device.htod_sync_copy(&v_scales_new).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: e.to_string(),
                }
            })?;

            // Concatenate with existing cache
            // Memory layout: [batch, kv_heads, seq_len, head_dim]
            // We need to properly interleave new data into each (batch, head) slice
            if let (Some(prev_k), Some(prev_v), Some(prev_ks), Some(prev_vs)) = (
                self.k_quant.take(),
                self.v_quant.take(),
                self.k_scales.take(),
                self.v_scales.take(),
            ) {
                let prev_seq_len = self.seq_len;
                let new_total_seq = prev_seq_len + new_seq_len;

                // Allocate new buffers
                let k_size = batch * kv_heads * new_total_seq * head_dim;
                let v_size = k_size;
                let scale_size = batch * kv_heads * new_total_seq;

                let mut new_k: CudaSlice<u8> =
                    device
                        .alloc_zeros(k_size)
                        .map_err(|e| Int8AttentionError::KernelExec {
                            message: e.to_string(),
                        })?;
                let mut new_v: CudaSlice<u8> =
                    device
                        .alloc_zeros(v_size)
                        .map_err(|e| Int8AttentionError::KernelExec {
                            message: e.to_string(),
                        })?;
                let mut new_ks: CudaSlice<u16> =
                    device
                        .alloc_zeros(scale_size)
                        .map_err(|e| Int8AttentionError::KernelExec {
                            message: e.to_string(),
                        })?;
                let mut new_vs: CudaSlice<u16> =
                    device
                        .alloc_zeros(scale_size)
                        .map_err(|e| Int8AttentionError::KernelExec {
                            message: e.to_string(),
                        })?;

                // Copy data per (batch, head) slice to maintain correct layout
                // Old layout: [batch, kv_heads, prev_seq_len, head_dim]
                // New layout: [batch, kv_heads, new_total_seq, head_dim]
                for b in 0..batch {
                    for h in 0..kv_heads {
                        // Calculate offsets for this (batch, head) slice
                        let old_slice_start = (b * kv_heads + h) * prev_seq_len * head_dim;
                        let old_slice_len = prev_seq_len * head_dim;
                        let new_slice_start = (b * kv_heads + h) * new_total_seq * head_dim;

                        let append_slice_start = (b * kv_heads + h) * new_seq_len * head_dim;
                        let append_slice_len = new_seq_len * head_dim;
                        let append_dst_start = new_slice_start + prev_seq_len * head_dim;

                        // Copy old data to new position
                        device
                            .dtod_copy(
                                &prev_k.slice(old_slice_start..old_slice_start + old_slice_len),
                                &mut new_k
                                    .slice_mut(new_slice_start..new_slice_start + old_slice_len),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        device
                            .dtod_copy(
                                &prev_v.slice(old_slice_start..old_slice_start + old_slice_len),
                                &mut new_v
                                    .slice_mut(new_slice_start..new_slice_start + old_slice_len),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        // Copy new data after old data
                        device
                            .dtod_copy(
                                &k_quant_gpu.slice(
                                    append_slice_start..append_slice_start + append_slice_len,
                                ),
                                &mut new_k.slice_mut(
                                    append_dst_start..append_dst_start + append_slice_len,
                                ),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        device
                            .dtod_copy(
                                &v_quant_gpu.slice(
                                    append_slice_start..append_slice_start + append_slice_len,
                                ),
                                &mut new_v.slice_mut(
                                    append_dst_start..append_dst_start + append_slice_len,
                                ),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        // Scale offsets (one scale per token per (batch, head))
                        let old_scale_start = (b * kv_heads + h) * prev_seq_len;
                        let old_scale_len = prev_seq_len;
                        let new_scale_start = (b * kv_heads + h) * new_total_seq;

                        let append_scale_start = (b * kv_heads + h) * new_seq_len;
                        let append_scale_len = new_seq_len;
                        let append_scale_dst = new_scale_start + prev_seq_len;

                        device
                            .dtod_copy(
                                &prev_ks.slice(old_scale_start..old_scale_start + old_scale_len),
                                &mut new_ks
                                    .slice_mut(new_scale_start..new_scale_start + old_scale_len),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        device
                            .dtod_copy(
                                &prev_vs.slice(old_scale_start..old_scale_start + old_scale_len),
                                &mut new_vs
                                    .slice_mut(new_scale_start..new_scale_start + old_scale_len),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        device
                            .dtod_copy(
                                &k_scales_gpu.slice(
                                    append_scale_start..append_scale_start + append_scale_len,
                                ),
                                &mut new_ks.slice_mut(
                                    append_scale_dst..append_scale_dst + append_scale_len,
                                ),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;

                        device
                            .dtod_copy(
                                &v_scales_gpu.slice(
                                    append_scale_start..append_scale_start + append_scale_len,
                                ),
                                &mut new_vs.slice_mut(
                                    append_scale_dst..append_scale_dst + append_scale_len,
                                ),
                            )
                            .map_err(|e| Int8AttentionError::KernelExec {
                                message: e.to_string(),
                            })?;
                    }
                }

                self.k_quant = Some(new_k);
                self.v_quant = Some(new_v);
                self.k_scales = Some(new_ks);
                self.v_scales = Some(new_vs);
            } else {
                // First append
                self.k_quant = Some(k_quant_gpu);
                self.v_quant = Some(v_quant_gpu);
                self.k_scales = Some(k_scales_gpu);
                self.v_scales = Some(v_scales_gpu);
            }

            self.seq_len += new_seq_len;
            Ok(())
        }

        /// Quantize a BF16 tensor to INT8 with per-token scales.
        fn quantize_tensor(
            &self,
            tensor: &candle_core::Tensor,
        ) -> Result<(Vec<u8>, Vec<u16>), Int8AttentionError> {
            let dims = tensor.dims();
            let (batch, kv_heads, seq_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

            // Convert to f32 for quantization
            let data = tensor
                .to_dtype(candle_core::DType::F32)
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?
                .flatten_all()
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?
                .to_vec1::<f32>()
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?;

            let mut quant_data = vec![0u8; batch * kv_heads * seq_len * head_dim];
            let mut scales = vec![0u16; batch * kv_heads * seq_len];

            // Quantize per-token (each token position has one scale for all head_dim elements)
            for b in 0..batch {
                for h in 0..kv_heads {
                    for s in 0..seq_len {
                        let offset = ((b * kv_heads + h) * seq_len + s) * head_dim;
                        let token_data = &data[offset..offset + head_dim];

                        // Find max absolute value
                        let max_abs = token_data
                            .iter()
                            .map(|x| x.abs())
                            .fold(0.0f32, |a, b| a.max(b))
                            .max(1e-8);

                        // Scale = max / 127
                        let scale = max_abs / 127.0;

                        // Quantize: round(x / scale) + 128
                        for (i, &val) in token_data.iter().enumerate() {
                            let q = ((val / scale).round() as i32 + 128).clamp(0, 255) as u8;
                            quant_data[offset + i] = q;
                        }

                        // Store scale as BF16
                        let scale_idx = (b * kv_heads + h) * seq_len + s;
                        scales[scale_idx] = f32_to_bf16(scale);
                    }
                }
            }

            Ok((quant_data, scales))
        }

        /// Compute attention using fused CUDA kernels.
        ///
        /// # Arguments
        /// * `q` - Query tensor BF16 [batch, num_heads, q_len, head_dim]
        /// * `num_heads` - Number of attention heads
        /// * `attn_scale` - Attention scale factor (1/sqrt(head_dim))
        ///
        /// # Returns
        /// Attention output BF16 [batch, num_heads, q_len, head_dim]
        pub fn forward_attention(
            &self,
            q: &candle_core::Tensor,
            num_heads: usize,
            attn_scale: f32,
        ) -> Result<candle_core::Tensor, Int8AttentionError> {
            let (k_quant, v_quant, k_scales, v_scales) =
                match (&self.k_quant, &self.v_quant, &self.k_scales, &self.v_scales) {
                    (Some(k), Some(v), Some(ks), Some(vs)) => (k, v, ks, vs),
                    _ => {
                        return Err(Int8AttentionError::KernelExec {
                            message: "KV cache is empty".to_string(),
                        })
                    },
                };

            let device = self.attn_ctx.device();
            let dims = q.dims();
            let (batch, _num_heads, q_len, head_dim) = (dims[0], dims[1], dims[2], dims[3]);

            // Convert Q to GPU BF16
            let q_bf16 = q
                .to_dtype(candle_core::DType::BF16)
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?
                .flatten_all()
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?;

            // Get Q data as u16 (BF16 bit representation)
            let q_data: Vec<u16> = q_bf16
                .to_vec1::<half::bf16>()
                .map_err(|e| Int8AttentionError::TensorConvert {
                    message: e.to_string(),
                })?
                .iter()
                .map(|x| x.to_bits())
                .collect();

            let q_gpu =
                device
                    .htod_sync_copy(&q_data)
                    .map_err(|e| Int8AttentionError::KernelExec {
                        message: e.to_string(),
                    })?;

            // Compute Q @ K^T with fused dequantization
            let attn_scores = self.attn_ctx.fused_qk_attention(
                &q_gpu,
                k_quant,
                k_scales,
                batch,
                num_heads,
                self.num_kv_heads,
                q_len,
                self.seq_len,
                head_dim,
                attn_scale,
            )?;

            // Softmax on CPU (could be optimized with CUDA kernel)
            let mut scores_host = vec![0.0f32; batch * num_heads * q_len * self.seq_len];
            device
                .dtoh_sync_copy_into(&attn_scores, &mut scores_host)
                .map_err(|e| Int8AttentionError::KernelExec {
                    message: e.to_string(),
                })?;

            // Apply causal mask and softmax per row
            // During prefill (q_len > 1), apply causal masking so each query position
            // can only attend to KV positions at or before it.
            // cache_offset = positions already in cache before this forward pass
            let cache_offset = self.seq_len - q_len;

            for b in 0..batch {
                for h in 0..num_heads {
                    for q_pos in 0..q_len {
                        let offset = ((b * num_heads + h) * q_len + q_pos) * self.seq_len;
                        let row = &mut scores_host[offset..offset + self.seq_len];

                        // Apply causal mask: query at position q_pos can only see
                        // KV positions 0..(cache_offset + q_pos) inclusive
                        let max_visible_kv = cache_offset + q_pos;
                        for kv_pos in (max_visible_kv + 1)..self.seq_len {
                            row[kv_pos] = f32::NEG_INFINITY;
                        }

                        // Stable softmax
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        for val in row.iter_mut() {
                            *val = (*val - max_val).exp();
                            sum += *val;
                        }
                        for val in row.iter_mut() {
                            *val /= sum;
                        }
                    }
                }
            }

            // Transfer softmax weights back to GPU
            let attn_weights = device.htod_sync_copy(&scores_host).map_err(|e| {
                Int8AttentionError::KernelExec {
                    message: e.to_string(),
                }
            })?;

            // Compute attn @ V with fused dequantization
            let output_bf16 = self.attn_ctx.fused_attn_v(
                &attn_weights,
                v_quant,
                v_scales,
                batch,
                num_heads,
                self.num_kv_heads,
                q_len,
                self.seq_len,
                head_dim,
            )?;

            // Convert back to Candle tensor
            let mut output_host = vec![0u16; batch * num_heads * q_len * head_dim];
            device
                .dtoh_sync_copy_into(&output_bf16, &mut output_host)
                .map_err(|e| Int8AttentionError::KernelExec {
                    message: e.to_string(),
                })?;

            // Convert u16 to bf16
            let output_bf16_vals: Vec<half::bf16> = output_host
                .iter()
                .map(|&bits| half::bf16::from_bits(bits))
                .collect();

            let candle_device = q.device();
            let output = candle_core::Tensor::from_vec(
                output_bf16_vals,
                (batch, num_heads, q_len, head_dim),
                candle_device,
            )
            .map_err(|e| Int8AttentionError::TensorConvert {
                message: e.to_string(),
            })?
            .to_dtype(q.dtype())
            .map_err(|e| Int8AttentionError::TensorConvert {
                message: e.to_string(),
            })?;

            Ok(output)
        }

        /// Get current sequence length.
        pub fn seq_len(&self) -> usize {
            self.seq_len
        }

        /// Clear the cache.
        pub fn clear(&mut self) {
            self.k_quant = None;
            self.v_quant = None;
            self.k_scales = None;
            self.v_scales = None;
            self.seq_len = 0;
            self.batch_size = 0;
        }

        /// Get memory usage in bytes.
        pub fn memory_bytes(&self) -> usize {
            let k_size = self.k_quant.as_ref().map(|s| s.len()).unwrap_or(0);
            let v_size = self.v_quant.as_ref().map(|s| s.len()).unwrap_or(0);
            let ks_size = self.k_scales.as_ref().map(|s| s.len() * 2).unwrap_or(0);
            let vs_size = self.v_scales.as_ref().map(|s| s.len() * 2).unwrap_or(0);
            k_size + v_size + ks_size + vs_size
        }
    }

    /// Convert f32 to BF16 bit representation.
    fn f32_to_bf16(f: f32) -> u16 {
        let bits = f.to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }
}

/// Optimized INT8 KV cache with multiple quantization strategies.
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};

/// Quantization granularity for KV cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantGranularity {
    /// Per-token scales: highest accuracy, more memory for scales.
    /// Shape: (batch, heads, seq_len, 1)
    PerToken,
    /// Per-head scales: fastest, lowest memory, may lose accuracy.
    /// Shape: (batch, heads, 1, 1)
    PerHead,
    /// Per-channel scales: balance of accuracy and speed.
    /// Shape: (batch, heads, 1, head_dim)
    PerChannel,
}

impl Default for QuantGranularity {
    fn default() -> Self {
        Self::PerToken
    }
}

/// Dynamic quantization configuration.
#[derive(Debug, Clone)]
pub struct DynamicQuantConfig {
    /// Enable dynamic quantization based on memory pressure.
    pub enabled: bool,
    /// Threshold (0.0-1.0) of VRAM usage to trigger quantization.
    pub memory_threshold: f32,
    /// Keep most recent N tokens unquantized for speed.
    pub unquantized_window: usize,
    /// Quantization granularity.
    pub granularity: QuantGranularity,
}

impl Default for DynamicQuantConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_threshold: 0.8,   // Quantize when >80% VRAM used
            unquantized_window: 128, // Keep last 128 tokens in BF16
            granularity: QuantGranularity::PerToken,
        }
    }
}

/// Optimized quantized KV cache with dynamic quantization and configurable granularity.
#[derive(Debug)]
pub struct OptimizedQuantizedKvCache {
    /// Quantized keys (older tokens): U8
    k_quantized: Option<Tensor>,
    /// Quantized values (older tokens): U8
    v_quantized: Option<Tensor>,
    /// Key scales (shape depends on granularity)
    k_scales: Option<Tensor>,
    /// Value scales (shape depends on granularity)
    v_scales: Option<Tensor>,
    /// Unquantized keys (recent tokens): BF16
    k_recent: Option<Tensor>,
    /// Unquantized values (recent tokens): BF16
    v_recent: Option<Tensor>,
    /// Configuration
    config: DynamicQuantConfig,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Device
    #[allow(dead_code)]
    device: Device,
    /// Output dtype
    dtype: DType,
    /// Total sequence length (quantized + recent)
    total_seq_len: usize,
}

impl OptimizedQuantizedKvCache {
    /// Create a new optimized quantized KV cache.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
        dtype: DType,
        config: DynamicQuantConfig,
    ) -> Self {
        Self {
            k_quantized: None,
            v_quantized: None,
            k_scales: None,
            v_scales: None,
            k_recent: None,
            v_recent: None,
            config,
            num_kv_heads,
            head_dim,
            device: device.clone(),
            dtype,
            total_seq_len: 0,
        }
    }

    /// Quantize tensor with specified granularity.
    fn quantize_with_granularity(
        tensor: &Tensor,
        granularity: QuantGranularity,
    ) -> CandleResult<(Tensor, Tensor)> {
        let abs_tensor = tensor.abs()?;

        // Compute scales based on granularity
        let max_vals = match granularity {
            QuantGranularity::PerToken => {
                // Max over head_dim: (batch, heads, seq, head_dim) -> (batch, heads, seq, 1)
                abs_tensor.max_keepdim(D::Minus1)?
            },
            QuantGranularity::PerHead => {
                // Max over seq and head_dim: -> (batch, heads, 1, 1)
                let max_seq = abs_tensor.max_keepdim(D::Minus2)?;
                max_seq.max_keepdim(D::Minus1)?
            },
            QuantGranularity::PerChannel => {
                // Max over seq: (batch, heads, seq, head_dim) -> (batch, heads, 1, head_dim)
                abs_tensor.max_keepdim(D::Minus2)?
            },
        };

        // Avoid division by zero
        let eps = Tensor::new(&[1e-8f32], tensor.device())?
            .broadcast_as(max_vals.shape())?
            .to_dtype(tensor.dtype())?;
        let max_vals = max_vals.maximum(&eps)?;

        // Scale = max / 127
        let scale = (&max_vals / 127.0)?;

        // Quantize: round((x / scale) + 128)
        let scaled = tensor.broadcast_div(&scale)?;
        let offset = (scaled + 128.0)?;
        let clamped = offset.clamp(0.0, 255.0)?;
        let quantized = clamped.round()?.to_dtype(DType::U8)?;

        Ok((quantized, scale))
    }

    /// Dequantize tensor.
    fn dequantize(quantized: &Tensor, scales: &Tensor, dtype: DType) -> CandleResult<Tensor> {
        let float_vals = quantized.to_dtype(dtype)?;
        let unoffset = (float_vals - 128.0)?;
        unoffset.broadcast_mul(scales)
    }

    /// Append new K and V tensors.
    /// Automatically manages quantization based on window size.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<()> {
        let new_seq_len = k.dim(2)?;

        // Concatenate with recent (unquantized) cache
        let (k_recent, v_recent) = match (&self.k_recent, &self.v_recent) {
            (Some(prev_k), Some(prev_v)) => {
                let k_cat = Tensor::cat(&[prev_k, k], 2)?;
                let v_cat = Tensor::cat(&[prev_v, v], 2)?;
                (k_cat, v_cat)
            },
            _ => (k.clone(), v.clone()),
        };

        let recent_len = k_recent.dim(2)?;
        self.total_seq_len += new_seq_len;

        // Check if we need to quantize older tokens
        if self.config.enabled && recent_len > self.config.unquantized_window {
            let tokens_to_quantize = recent_len - self.config.unquantized_window;

            // Split into to-quantize and keep-recent
            let k_to_quant = k_recent.narrow(2, 0, tokens_to_quantize)?;
            let v_to_quant = v_recent.narrow(2, 0, tokens_to_quantize)?;
            let k_keep = k_recent.narrow(2, tokens_to_quantize, self.config.unquantized_window)?;
            let v_keep = v_recent.narrow(2, tokens_to_quantize, self.config.unquantized_window)?;

            // Quantize older tokens
            let (k_quant_new, k_scale_new) =
                Self::quantize_with_granularity(&k_to_quant, self.config.granularity)?;
            let (v_quant_new, v_scale_new) =
                Self::quantize_with_granularity(&v_to_quant, self.config.granularity)?;

            // Merge with existing quantized cache
            let (k_quantized, k_scales) = match (&self.k_quantized, &self.k_scales) {
                (Some(prev_k), Some(prev_s)) => {
                    let k = Tensor::cat(&[prev_k, &k_quant_new], 2)?;
                    let s = Tensor::cat(&[prev_s, &k_scale_new], 2)?;
                    (k, s)
                },
                _ => (k_quant_new, k_scale_new),
            };

            let (v_quantized, v_scales) = match (&self.v_quantized, &self.v_scales) {
                (Some(prev_v), Some(prev_s)) => {
                    let v = Tensor::cat(&[prev_v, &v_quant_new], 2)?;
                    let s = Tensor::cat(&[prev_s, &v_scale_new], 2)?;
                    (v, s)
                },
                _ => (v_quant_new, v_scale_new),
            };

            self.k_quantized = Some(k_quantized);
            self.k_scales = Some(k_scales);
            self.v_quantized = Some(v_quantized);
            self.v_scales = Some(v_scales);
            self.k_recent = Some(k_keep);
            self.v_recent = Some(v_keep);
        } else {
            // Just keep in recent cache
            self.k_recent = Some(k_recent);
            self.v_recent = Some(v_recent);
        }

        Ok(())
    }

    /// Get full K and V tensors for attention.
    /// Dequantizes older tokens and concatenates with recent.
    pub fn get_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        match (&self.k_recent, &self.v_recent) {
            (Some(k_recent), Some(v_recent)) => {
                // Check if we have quantized portion
                match (
                    &self.k_quantized,
                    &self.k_scales,
                    &self.v_quantized,
                    &self.v_scales,
                ) {
                    (Some(k_q), Some(k_s), Some(v_q), Some(v_s)) => {
                        // Dequantize and concatenate
                        let k_dequant = Self::dequantize(k_q, k_s, self.dtype)?;
                        let v_dequant = Self::dequantize(v_q, v_s, self.dtype)?;
                        let k_full = Tensor::cat(&[&k_dequant, k_recent], 2)?;
                        let v_full = Tensor::cat(&[&v_dequant, v_recent], 2)?;
                        Ok(Some((k_full, v_full)))
                    },
                    _ => {
                        // Only recent cache
                        Ok(Some((k_recent.clone(), v_recent.clone())))
                    },
                }
            },
            _ => Ok(None),
        }
    }

    /// Get sequence length.
    pub fn seq_len(&self) -> usize {
        self.total_seq_len
    }

    /// Get quantized sequence length.
    pub fn quantized_len(&self) -> usize {
        self.k_quantized.as_ref().map(|t| t.dims()[2]).unwrap_or(0)
    }

    /// Get recent (unquantized) sequence length.
    pub fn recent_len(&self) -> usize {
        self.k_recent.as_ref().map(|t| t.dims()[2]).unwrap_or(0)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.k_quantized = None;
        self.v_quantized = None;
        self.k_scales = None;
        self.v_scales = None;
        self.k_recent = None;
        self.v_recent = None;
        self.total_seq_len = 0;
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;

        // Quantized: 1 byte per element
        if let Some(k) = &self.k_quantized {
            total += k.elem_count();
        }
        if let Some(v) = &self.v_quantized {
            total += v.elem_count();
        }

        // Scales: 2 bytes (BF16)
        if let Some(s) = &self.k_scales {
            total += s.elem_count() * 2;
        }
        if let Some(s) = &self.v_scales {
            total += s.elem_count() * 2;
        }

        // Recent: 2 bytes (BF16)
        if let Some(k) = &self.k_recent {
            total += k.elem_count() * 2;
        }
        if let Some(v) = &self.v_recent {
            total += v.elem_count() * 2;
        }

        total
    }

    /// Calculate memory savings vs full BF16 cache.
    pub fn memory_savings_ratio(&self) -> f32 {
        let full_bf16 = self.total_seq_len * self.num_kv_heads * self.head_dim * 2 * 2; // K+V, BF16
        if full_bf16 == 0 {
            return 1.0;
        }
        full_bf16 as f32 / self.memory_bytes() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_head_quantization() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let config = DynamicQuantConfig {
            enabled: true,
            memory_threshold: 0.8,
            unquantized_window: 4,
            granularity: QuantGranularity::PerHead,
        };

        let mut cache = OptimizedQuantizedKvCache::new(4, 64, &device, dtype, config);

        // Add 10 tokens - should quantize 6
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 64), &device)?;
        cache.append(&k, &v)?;

        assert_eq!(cache.seq_len(), 10);
        assert_eq!(cache.quantized_len(), 6);
        assert_eq!(cache.recent_len(), 4);

        // Verify we can get KV
        let (k_full, v_full) = cache.get_kv()?.unwrap();
        assert_eq!(k_full.dims(), &[1, 4, 10, 64]);
        assert_eq!(v_full.dims(), &[1, 4, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_dynamic_quantization_window() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let config = DynamicQuantConfig {
            enabled: true,
            memory_threshold: 0.8,
            unquantized_window: 8,
            granularity: QuantGranularity::PerToken,
        };

        let mut cache = OptimizedQuantizedKvCache::new(4, 64, &device, dtype, config);

        // Add 5 tokens - should not quantize (under window)
        let k1 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        let v1 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        cache.append(&k1, &v1)?;
        assert_eq!(cache.quantized_len(), 0);
        assert_eq!(cache.recent_len(), 5);

        // Add 5 more - should quantize 2 (10 - 8 window)
        let k2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        let v2 = Tensor::randn(0.0f32, 1.0, (1, 4, 5, 64), &device)?;
        cache.append(&k2, &v2)?;
        assert_eq!(cache.quantized_len(), 2);
        assert_eq!(cache.recent_len(), 8);

        Ok(())
    }

    #[test]
    fn test_memory_savings() -> CandleResult<()> {
        let device = Device::Cpu;
        let dtype = DType::BF16;

        let config = DynamicQuantConfig {
            enabled: true,
            memory_threshold: 0.8,
            unquantized_window: 16,
            granularity: QuantGranularity::PerToken,
        };

        let mut cache = OptimizedQuantizedKvCache::new(8, 128, &device, dtype, config);

        // Add 1000 tokens
        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 1000, 128), &device)?.to_dtype(dtype)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 1000, 128), &device)?.to_dtype(dtype)?;
        cache.append(&k, &v)?;

        let savings = cache.memory_savings_ratio();
        println!(
            "Memory: {} bytes, savings: {:.2}x",
            cache.memory_bytes(),
            savings
        );

        // Should have good savings (quantized portion >> recent portion)
        assert!(
            savings > 1.5,
            "Expected at least 1.5x savings, got {:.2}x",
            savings
        );

        Ok(())
    }
}
