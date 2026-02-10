//! Fused GPU kernels for combined decompression + dequantization.
//!
//! This module provides CUDA kernels that perform LZ4 decompression and INT4/INT8
//! dequantization in a single pass, avoiding intermediate buffers.
//!
//! ## Why Fused Kernels?
//!
//! Separate decompression and dequantization requires:
//! 1. GPU memory for compressed data
//! 2. GPU memory for decompressed INT4 data (intermediate)
//! 3. GPU memory for final F16 output
//!
//! Fused kernels eliminate step 2, reducing memory bandwidth and VRAM usage:
//! 1. GPU memory for compressed data
//! 2. GPU memory for final F16 output (directly written)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Fused Decompress + Dequant                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Compressed LZ4+INT4          F16 Tensor (GPU)                 │
//! │  ┌──────────────────┐        ┌──────────────────────┐          │
//! │  │ [Block 0]        │        │ [0.5][1.0][1.5]...   │          │
//! │  │ [Block 1]        │ ────>  │                      │          │
//! │  │ ...              │   │    └──────────────────────┘          │
//! │  └──────────────────┘   │                                      │
//! │                         │                                      │
//! │            ┌────────────┴────────────┐                         │
//! │            │  Per-Thread Pipeline:   │                         │
//! │            │  1. Read LZ4 token      │                         │
//! │            │  2. Copy/match literals │                         │
//! │            │  3. Unpack INT4 nibbles │                         │
//! │            │  4. Apply scale + zp    │                         │
//! │            │  5. Write F16 output    │                         │
//! │            └─────────────────────────┘                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

/// CUDA-accelerated fused LZ4 decompression and dequantization.
#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use candle_core::{Device, Tensor};
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;

    /// Fused decompression + dequantization context.
    pub struct GpuFusedContext {
        device: Arc<CudaDevice>,
        device_id: usize,
        lz4_int4_kernel_loaded: bool,
        lz4_int8_kernel_loaded: bool,
    }

    impl GpuFusedContext {
        /// Creates a new fused kernel context.
        pub fn new(device_id: usize) -> Result<Self, GpuFusedError> {
            let device = CudaDevice::new(device_id).map_err(|e| GpuFusedError::DeviceInit {
                device_id,
                message: e.to_string(),
            })?;

            Ok(Self {
                device,
                device_id,
                lz4_int4_kernel_loaded: false,
                lz4_int8_kernel_loaded: false,
            })
        }

        /// Returns the CUDA device ID.
        pub fn device_id(&self) -> usize {
            self.device_id
        }

        /// Loads the fused LZ4+INT4 kernel.
        pub fn load_lz4_int4_kernel(&mut self) -> Result<(), GpuFusedError> {
            if self.lz4_int4_kernel_loaded {
                return Ok(());
            }

            let ptx = Ptx::from_src(FUSED_LZ4_INT4_KERNEL_PTX);
            self.device
                .load_ptx(
                    ptx,
                    "fused_lz4_int4",
                    &["fused_lz4_int4_block", "fused_lz4_int4_blocks_parallel"],
                )
                .map_err(|e| GpuFusedError::KernelLoad {
                    message: e.to_string(),
                })?;

            self.lz4_int4_kernel_loaded = true;
            Ok(())
        }

        /// Loads the fused LZ4+INT8 kernel.
        pub fn load_lz4_int8_kernel(&mut self) -> Result<(), GpuFusedError> {
            if self.lz4_int8_kernel_loaded {
                return Ok(());
            }

            let ptx = Ptx::from_src(FUSED_LZ4_INT8_KERNEL_PTX);
            self.device
                .load_ptx(ptx, "fused_lz4_int8", &["fused_lz4_int8_block"])
                .map_err(|e| GpuFusedError::KernelLoad {
                    message: e.to_string(),
                })?;

            self.lz4_int8_kernel_loaded = true;
            Ok(())
        }

        /// Fused LZ4 decompression + INT4 dequantization for a single block.
        ///
        /// Takes LZ4-compressed INT4 data and outputs dequantized F16 values.
        ///
        /// # Arguments
        ///
        /// * `compressed` - LZ4-compressed INT4 packed data
        /// * `scale` - Scale factor for dequantization
        /// * `zero_point` - Zero point for asymmetric quantization
        /// * `num_values` - Number of output F16 values
        ///
        /// # Returns
        ///
        /// GPU buffer containing dequantized F16 values.
        pub fn fused_lz4_int4_block(
            &self,
            compressed: &[u8],
            scale: half::f16,
            zero_point: i8,
            num_values: usize,
        ) -> Result<CudaSlice<half::f16>, GpuFusedError> {
            if !self.lz4_int4_kernel_loaded {
                return Err(GpuFusedError::KernelNotLoaded {
                    kernel: "fused_lz4_int4".to_string(),
                });
            }

            // Copy compressed data to GPU
            let d_compressed = self.device.htod_copy(compressed.to_vec()).map_err(|e| {
                GpuFusedError::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            // Allocate output buffer
            let d_output: CudaSlice<half::f16> =
                self.device
                    .alloc_zeros(num_values)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Launch kernel
            let func = self
                .device
                .get_func("fused_lz4_int4", "fused_lz4_int4_block")
                .ok_or_else(|| GpuFusedError::KernelLoad {
                    message: "fused_lz4_int4_block not found".to_string(),
                })?;

            let cfg = LaunchConfig::for_num_elems(1);
            let scale_bits = scale.to_bits();

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_compressed,
                        compressed.len() as u32,
                        &d_output,
                        num_values as u32,
                        scale_bits as u32,
                        zero_point as i32,
                    ),
                )
            }
            .map_err(|e| GpuFusedError::KernelExec {
                message: e.to_string(),
            })?;

            self.device
                .synchronize()
                .map_err(|e| GpuFusedError::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Fused LZ4 decompression + INT4 dequantization for multiple blocks.
        ///
        /// Each block is processed in parallel by a separate GPU thread block.
        ///
        /// # Arguments
        ///
        /// * `blocks` - Vector of (compressed_data, num_output_values) tuples
        /// * `scales` - Per-block scale factors
        /// * `zero_points` - Per-block zero points (optional, defaults to 0)
        ///
        /// # Returns
        ///
        /// GPU buffer containing all dequantized F16 values concatenated.
        pub fn fused_lz4_int4_parallel(
            &self,
            blocks: &[(Vec<u8>, usize)],
            scales: &[half::f16],
            zero_points: Option<&[i8]>,
        ) -> Result<CudaSlice<half::f16>, GpuFusedError> {
            if !self.lz4_int4_kernel_loaded {
                return Err(GpuFusedError::KernelNotLoaded {
                    kernel: "fused_lz4_int4".to_string(),
                });
            }

            if blocks.is_empty() {
                return Err(GpuFusedError::InvalidInput {
                    message: "No blocks to process".to_string(),
                });
            }

            let num_blocks = blocks.len();

            // Validate scales
            if scales.len() < num_blocks {
                return Err(GpuFusedError::InvalidInput {
                    message: format!(
                        "Not enough scales: got {}, need {}",
                        scales.len(),
                        num_blocks
                    ),
                });
            }

            // Calculate offsets and total sizes
            let total_compressed: usize = blocks.iter().map(|(b, _)| b.len()).sum();
            let total_output: usize = blocks.iter().map(|(_, n)| *n).sum();

            let mut compressed_offsets: Vec<u32> = Vec::with_capacity(num_blocks);
            let mut output_offsets: Vec<u32> = Vec::with_capacity(num_blocks);
            let mut compressed_sizes: Vec<u32> = Vec::with_capacity(num_blocks);
            let mut output_sizes: Vec<u32> = Vec::with_capacity(num_blocks);

            let mut comp_offset = 0u32;
            let mut out_offset = 0u32;

            for (compressed, num_values) in blocks {
                compressed_offsets.push(comp_offset);
                output_offsets.push(out_offset);
                compressed_sizes.push(compressed.len() as u32);
                output_sizes.push(*num_values as u32);

                comp_offset += compressed.len() as u32;
                out_offset += *num_values as u32;
            }

            // Concatenate all compressed data
            let mut all_compressed = Vec::with_capacity(total_compressed);
            for (compressed, _) in blocks {
                all_compressed.extend_from_slice(compressed);
            }

            // Prepare zero points
            let zp_vec: Vec<i8> = zero_points
                .map(|zp| zp.to_vec())
                .unwrap_or_else(|| vec![0i8; num_blocks]);

            // Copy to GPU
            let d_compressed =
                self.device
                    .htod_copy(all_compressed)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_output: CudaSlice<half::f16> =
                self.device
                    .alloc_zeros(total_output)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_comp_offsets = self.device.htod_copy(compressed_offsets).map_err(|e| {
                GpuFusedError::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            let d_out_offsets =
                self.device
                    .htod_copy(output_offsets)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_comp_sizes = self.device.htod_copy(compressed_sizes).map_err(|e| {
                GpuFusedError::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            let d_out_sizes =
                self.device
                    .htod_copy(output_sizes)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Convert scales to bytes
            let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            let d_scales =
                self.device
                    .htod_copy(scales_bytes)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_zero_points =
                self.device
                    .htod_copy(zp_vec)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Launch parallel kernel
            let func = self
                .device
                .get_func("fused_lz4_int4", "fused_lz4_int4_blocks_parallel")
                .ok_or_else(|| GpuFusedError::KernelLoad {
                    message: "fused_lz4_int4_blocks_parallel not found".to_string(),
                })?;

            let cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (1, 1, 1), // One thread per LZ4 block for sequential decompression
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_compressed,
                        &d_output,
                        &d_comp_offsets,
                        &d_out_offsets,
                        &d_comp_sizes,
                        &d_out_sizes,
                        &d_scales,
                        &d_zero_points,
                        num_blocks as u32,
                    ),
                )
            }
            .map_err(|e| GpuFusedError::KernelExec {
                message: e.to_string(),
            })?;

            self.device
                .synchronize()
                .map_err(|e| GpuFusedError::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Fused LZ4+INT4 decompression directly to a Candle tensor.
        pub fn fused_lz4_int4_to_tensor(
            &self,
            blocks: &[(Vec<u8>, usize)],
            scales: &[half::f16],
            zero_points: Option<&[i8]>,
            shape: &[usize],
        ) -> Result<Tensor, GpuFusedError> {
            let d_output = self.fused_lz4_int4_parallel(blocks, scales, zero_points)?;

            let num_values: usize = shape.iter().product();
            let mut host_data = vec![half::f16::ZERO; num_values];
            self.device
                .dtoh_sync_copy_into(&d_output, &mut host_data)
                .map_err(|e| GpuFusedError::MemoryCopy {
                    message: e.to_string(),
                })?;

            Tensor::from_vec(host_data, shape, &Device::Cpu).map_err(|e| {
                GpuFusedError::TensorCreate {
                    message: e.to_string(),
                }
            })
        }

        /// Fused LZ4 decompression + INT8 dequantization for a single block.
        pub fn fused_lz4_int8_block(
            &self,
            compressed: &[u8],
            scale: half::f16,
            num_values: usize,
        ) -> Result<CudaSlice<half::f16>, GpuFusedError> {
            if !self.lz4_int8_kernel_loaded {
                return Err(GpuFusedError::KernelNotLoaded {
                    kernel: "fused_lz4_int8".to_string(),
                });
            }

            let d_compressed = self.device.htod_copy(compressed.to_vec()).map_err(|e| {
                GpuFusedError::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            let d_output: CudaSlice<half::f16> =
                self.device
                    .alloc_zeros(num_values)
                    .map_err(|e| GpuFusedError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let func = self
                .device
                .get_func("fused_lz4_int8", "fused_lz4_int8_block")
                .ok_or_else(|| GpuFusedError::KernelLoad {
                    message: "fused_lz4_int8_block not found".to_string(),
                })?;

            let cfg = LaunchConfig::for_num_elems(1);
            let scale_bits = scale.to_bits();

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_compressed,
                        compressed.len() as u32,
                        &d_output,
                        num_values as u32,
                        scale_bits as u32,
                    ),
                )
            }
            .map_err(|e| GpuFusedError::KernelExec {
                message: e.to_string(),
            })?;

            self.device
                .synchronize()
                .map_err(|e| GpuFusedError::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }
    }

    /// Errors from fused GPU operations.
    #[derive(Debug, thiserror::Error)]
    pub enum GpuFusedError {
        /// CUDA device initialization failed.
        #[error("Failed to initialize CUDA device {device_id}: {message}")]
        DeviceInit {
            /// CUDA device ID.
            device_id: usize,
            /// Error message.
            message: String,
        },

        /// Kernel loading failed.
        #[error("Failed to load kernel: {message}")]
        KernelLoad {
            /// Error message.
            message: String,
        },

        /// Required kernel not loaded.
        #[error("Kernel not loaded: {kernel}")]
        KernelNotLoaded {
            /// Kernel name.
            kernel: String,
        },

        /// Kernel execution failed.
        #[error("Kernel execution failed: {message}")]
        KernelExec {
            /// Error message.
            message: String,
        },

        /// GPU memory allocation failed.
        #[error("Memory allocation failed: {message}")]
        MemoryAlloc {
            /// Error message.
            message: String,
        },

        /// GPU memory copy failed.
        #[error("Memory copy failed: {message}")]
        MemoryCopy {
            /// Error message.
            message: String,
        },

        /// GPU synchronization failed.
        #[error("Synchronization failed: {message}")]
        Synchronize {
            /// Error message.
            message: String,
        },

        /// Invalid input data.
        #[error("Invalid input: {message}")]
        InvalidInput {
            /// Error message.
            message: String,
        },

        /// Candle tensor creation failed.
        #[error("Tensor creation failed: {message}")]
        TensorCreate {
            /// Error message.
            message: String,
        },
    }

    /// Fused LZ4 decompression + INT4 dequantization kernel.
    ///
    /// This kernel performs both operations in a single pass:
    /// 1. Decompress LZ4 block (sequential per thread)
    /// 2. For each decompressed byte, unpack 2 INT4 values
    /// 3. Apply scale and zero point to get F16 output
    const FUSED_LZ4_INT4_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// Fused LZ4 decompression + INT4 dequantization for single block
// Decompresses LZ4, unpacks INT4 (2 per byte), applies scale/zp, outputs F16
.visible .entry fused_lz4_int4_block(
    .param .u64 input_ptr,       // Compressed LZ4+INT4 data
    .param .u32 input_size,      // Size of compressed data
    .param .u64 output_ptr,      // Output F16 buffer
    .param .u32 num_values,      // Number of output F16 values
    .param .u32 scale_bits,      // F16 scale as u16 bits
    .param .s32 zero_point       // Zero point for dequant
)
{
    .reg .u64 %rd<32>;
    .reg .u32 %r<48>;
    .reg .f32 %f<8>;
    .reg .b16 %h<8>;
    .reg .pred %p<16>;
    .reg .b8 %rb<8>;

    // Load parameters
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u32 %r1, [input_size];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r2, [num_values];
    ld.param.u32 %r3, [scale_bits];
    ld.param.s32 %r4, [zero_point];

    // Convert scale to F32 for computation
    cvt.u16.u32 %h1, %r3;
    cvt.f32.f16 %f1, %h1;  // scale as F32

    // LZ4 decompression state
    mov.u32 %r10, 0;  // in_pos (input position)
    mov.u32 %r11, 0;  // out_pos (output F16 position)
    mov.u32 %r12, 0;  // decomp_byte_pos (decompressed byte position)

    // We'll decompress to a "virtual" byte stream and immediately dequantize
    // For each decompressed byte, we output 2 F16 values (2 INT4 per byte)

FUSED_LOOP:
    // Check if we've produced enough output values
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra FUSED_DONE;

    // Check if we've consumed all input
    setp.ge.u32 %p2, %r10, %r1;
    @%p2 bra FUSED_DONE;

    // Read LZ4 token
    cvt.u64.u32 %rd3, %r10;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u8 %rb0, [%rd4];
    cvt.u32.u8 %r20, %rb0;  // token
    add.u32 %r10, %r10, 1;

    // Literal length = token >> 4
    shr.u32 %r21, %r20, 4;  // lit_len

    // Extended literal length if needed
    setp.ne.u32 %p3, %r21, 15;
    @%p3 bra FUSED_SKIP_LIT_EXT;

FUSED_LIT_EXT:
    cvt.u64.u32 %rd5, %r10;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %rb1, [%rd6];
    cvt.u32.u8 %r22, %rb1;
    add.u32 %r10, %r10, 1;
    add.u32 %r21, %r21, %r22;
    setp.eq.u32 %p4, %r22, 255;
    @%p4 bra FUSED_LIT_EXT;

FUSED_SKIP_LIT_EXT:
    // Process literal bytes - each byte gives 2 F16 outputs
    mov.u32 %r23, 0;  // lit counter

FUSED_LIT_LOOP:
    setp.ge.u32 %p5, %r23, %r21;
    @%p5 bra FUSED_LIT_DONE;

    // Read literal byte (contains 2 INT4 values)
    cvt.u64.u32 %rd7, %r10;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u8 %rb2, [%rd8];
    cvt.u32.u8 %r24, %rb2;  // packed byte
    add.u32 %r10, %r10, 1;
    add.u32 %r23, %r23, 1;

    // Check output bounds before writing
    setp.ge.u32 %p14, %r11, %r2;
    @%p14 bra FUSED_LIT_DONE;

    // Extract low nibble (first INT4)
    and.b32 %r25, %r24, 15;
    // Dequantize: (val - zp) * scale
    sub.s32 %r26, %r25, %r4;
    cvt.rn.f32.s32 %f2, %r26;
    mul.f32 %f3, %f2, %f1;
    cvt.rn.f16.f32 %h2, %f3;
    // Write F16 output
    shl.b32 %r27, %r11, 1;  // byte offset = out_pos * 2
    cvt.u64.u32 %rd9, %r27;
    add.u64 %rd10, %rd2, %rd9;
    st.global.u16 [%rd10], %h2;
    add.u32 %r11, %r11, 1;

    // Check bounds again
    setp.ge.u32 %p15, %r11, %r2;
    @%p15 bra FUSED_LIT_DONE;

    // Extract high nibble (second INT4)
    shr.u32 %r28, %r24, 4;
    sub.s32 %r29, %r28, %r4;
    cvt.rn.f32.s32 %f4, %r29;
    mul.f32 %f5, %f4, %f1;
    cvt.rn.f16.f32 %h3, %f5;
    shl.b32 %r30, %r11, 1;
    cvt.u64.u32 %rd11, %r30;
    add.u64 %rd12, %rd2, %rd11;
    st.global.u16 [%rd12], %h3;
    add.u32 %r11, %r11, 1;

    bra FUSED_LIT_LOOP;

FUSED_LIT_DONE:
    // Check if we're at end of block
    setp.ge.u32 %p6, %r10, %r1;
    @%p6 bra FUSED_DONE;

    // Read 2-byte offset
    cvt.u64.u32 %rd13, %r10;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u8 %rb3, [%rd14];
    cvt.u32.u8 %r31, %rb3;
    add.u32 %r10, %r10, 1;

    cvt.u64.u32 %rd15, %r10;
    add.u64 %rd16, %rd1, %rd15;
    ld.global.u8 %rb4, [%rd16];
    cvt.u32.u8 %r32, %rb4;
    add.u32 %r10, %r10, 1;

    shl.b32 %r33, %r32, 8;
    or.b32 %r34, %r31, %r33;  // offset in bytes

    // Match length = (token & 0x0F) + 4
    and.b32 %r35, %r20, 15;
    add.u32 %r36, %r35, 4;  // match_len in bytes

    // Extended match length
    setp.ne.u32 %p7, %r35, 15;
    @%p7 bra FUSED_SKIP_MATCH_EXT;

FUSED_MATCH_EXT:
    cvt.u64.u32 %rd17, %r10;
    add.u64 %rd18, %rd1, %rd17;
    ld.global.u8 %rb5, [%rd18];
    cvt.u32.u8 %r37, %rb5;
    add.u32 %r10, %r10, 1;
    add.u32 %r36, %r36, %r37;
    setp.eq.u32 %p8, %r37, 255;
    @%p8 bra FUSED_MATCH_EXT;

FUSED_SKIP_MATCH_EXT:
    // For matches, we need to copy from already-output F16 values
    // The offset is in terms of decompressed INT4 bytes
    // Each byte = 2 F16 values, so F16 offset = byte_offset * 2
    shl.b32 %r38, %r34, 1;  // F16 offset = byte_offset * 2
    sub.u32 %r39, %r11, %r38;  // source F16 position

    mov.u32 %r40, 0;  // match counter (in bytes)

FUSED_MATCH_LOOP:
    setp.ge.u32 %p9, %r40, %r36;
    @%p9 bra FUSED_MATCH_DONE;

    // Each match byte = 2 F16 values to copy
    // Copy first F16
    setp.ge.u32 %p10, %r11, %r2;
    @%p10 bra FUSED_MATCH_DONE;

    // Calculate source position for this iteration
    shl.b32 %r41, %r40, 1;  // bytes to F16 offset
    add.u32 %r42, %r39, %r41;  // source F16 idx for first value

    shl.b32 %r43, %r42, 1;  // byte offset
    cvt.u64.u32 %rd19, %r43;
    add.u64 %rd20, %rd2, %rd19;
    ld.global.u16 %h4, [%rd20];  // read source F16

    shl.b32 %r44, %r11, 1;
    cvt.u64.u32 %rd21, %r44;
    add.u64 %rd22, %rd2, %rd21;
    st.global.u16 [%rd22], %h4;  // write dest F16
    add.u32 %r11, %r11, 1;

    // Copy second F16
    setp.ge.u32 %p11, %r11, %r2;
    @%p11 bra FUSED_MATCH_DONE;

    add.u32 %r45, %r42, 1;  // source F16 idx for second value
    shl.b32 %r46, %r45, 1;
    cvt.u64.u32 %rd23, %r46;
    add.u64 %rd24, %rd2, %rd23;
    ld.global.u16 %h5, [%rd24];

    shl.b32 %r47, %r11, 1;
    cvt.u64.u32 %rd25, %r47;
    add.u64 %rd26, %rd2, %rd25;
    st.global.u16 [%rd26], %h5;
    add.u32 %r11, %r11, 1;

    add.u32 %r40, %r40, 1;
    bra FUSED_MATCH_LOOP;

FUSED_MATCH_DONE:
    bra FUSED_LOOP;

FUSED_DONE:
    ret;
}

// Parallel fused kernel for multiple blocks
.visible .entry fused_lz4_int4_blocks_parallel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 comp_offsets_ptr,
    .param .u64 out_offsets_ptr,
    .param .u64 comp_sizes_ptr,
    .param .u64 out_sizes_ptr,
    .param .u64 scales_ptr,
    .param .u64 zp_ptr,
    .param .u32 num_blocks
)
{
    .reg .u64 %rd<48>;
    .reg .u32 %r<64>;
    .reg .f32 %f<8>;
    .reg .b16 %h<8>;
    .reg .pred %p<16>;
    .reg .b8 %rb<8>;

    // Get block index
    mov.u32 %r1, %ctaid.x;

    // Bounds check
    ld.param.u32 %r2, [num_blocks];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra PAR_DONE;

    // Load pointers
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u64 %rd3, [comp_offsets_ptr];
    ld.param.u64 %rd4, [out_offsets_ptr];
    ld.param.u64 %rd5, [comp_sizes_ptr];
    ld.param.u64 %rd6, [out_sizes_ptr];
    ld.param.u64 %rd7, [scales_ptr];
    ld.param.u64 %rd8, [zp_ptr];

    // Load this block's parameters
    mul.lo.u32 %r3, %r1, 4;  // block_idx * 4 (u32 size)
    cvt.u64.u32 %rd9, %r3;

    add.u64 %rd10, %rd3, %rd9;
    ld.global.u32 %r4, [%rd10];  // comp_offset

    add.u64 %rd11, %rd4, %rd9;
    ld.global.u32 %r5, [%rd11];  // out_offset (in F16 units)

    add.u64 %rd12, %rd5, %rd9;
    ld.global.u32 %r6, [%rd12];  // comp_size

    add.u64 %rd13, %rd6, %rd9;
    ld.global.u32 %r7, [%rd13];  // out_size (num F16 values)

    // Load scale (2 bytes per scale)
    shl.b32 %r8, %r1, 1;
    cvt.u64.u32 %rd14, %r8;
    add.u64 %rd15, %rd7, %rd14;
    ld.global.u16 %h1, [%rd15];
    cvt.f32.f16 %f1, %h1;  // scale as F32

    // Load zero point
    cvt.u64.u32 %rd16, %r1;
    add.u64 %rd17, %rd8, %rd16;
    ld.global.s8 %r9, [%rd17];  // zero_point

    // Calculate block-local pointers
    cvt.u64.u32 %rd18, %r4;
    add.u64 %rd19, %rd1, %rd18;  // block input ptr

    shl.b32 %r10, %r5, 1;  // out_offset * 2 (bytes)
    cvt.u64.u32 %rd20, %r10;
    add.u64 %rd21, %rd2, %rd20;  // block output ptr

    // Now run the same decompression logic as single block
    mov.u32 %r20, 0;  // in_pos
    mov.u32 %r21, 0;  // out_pos (relative to block start)

PAR_LOOP:
    setp.ge.u32 %p2, %r21, %r7;
    @%p2 bra PAR_DONE;

    setp.ge.u32 %p3, %r20, %r6;
    @%p3 bra PAR_DONE;

    // Read token
    cvt.u64.u32 %rd22, %r20;
    add.u64 %rd23, %rd19, %rd22;
    ld.global.u8 %rb0, [%rd23];
    cvt.u32.u8 %r30, %rb0;
    add.u32 %r20, %r20, 1;

    shr.u32 %r31, %r30, 4;  // lit_len

    setp.ne.u32 %p4, %r31, 15;
    @%p4 bra PAR_SKIP_LIT_EXT;

PAR_LIT_EXT:
    cvt.u64.u32 %rd24, %r20;
    add.u64 %rd25, %rd19, %rd24;
    ld.global.u8 %rb1, [%rd25];
    cvt.u32.u8 %r32, %rb1;
    add.u32 %r20, %r20, 1;
    add.u32 %r31, %r31, %r32;
    setp.eq.u32 %p5, %r32, 255;
    @%p5 bra PAR_LIT_EXT;

PAR_SKIP_LIT_EXT:
    mov.u32 %r33, 0;

PAR_LIT_LOOP:
    setp.ge.u32 %p6, %r33, %r31;
    @%p6 bra PAR_LIT_DONE;

    cvt.u64.u32 %rd26, %r20;
    add.u64 %rd27, %rd19, %rd26;
    ld.global.u8 %rb2, [%rd27];
    cvt.u32.u8 %r34, %rb2;
    add.u32 %r20, %r20, 1;
    add.u32 %r33, %r33, 1;

    setp.ge.u32 %p7, %r21, %r7;
    @%p7 bra PAR_LIT_DONE;

    // Low nibble
    and.b32 %r35, %r34, 15;
    sub.s32 %r36, %r35, %r9;
    cvt.rn.f32.s32 %f2, %r36;
    mul.f32 %f3, %f2, %f1;
    cvt.rn.f16.f32 %h2, %f3;
    shl.b32 %r37, %r21, 1;
    cvt.u64.u32 %rd28, %r37;
    add.u64 %rd29, %rd21, %rd28;
    st.global.u16 [%rd29], %h2;
    add.u32 %r21, %r21, 1;

    setp.ge.u32 %p8, %r21, %r7;
    @%p8 bra PAR_LIT_DONE;

    // High nibble
    shr.u32 %r38, %r34, 4;
    sub.s32 %r39, %r38, %r9;
    cvt.rn.f32.s32 %f4, %r39;
    mul.f32 %f5, %f4, %f1;
    cvt.rn.f16.f32 %h3, %f5;
    shl.b32 %r40, %r21, 1;
    cvt.u64.u32 %rd30, %r40;
    add.u64 %rd31, %rd21, %rd30;
    st.global.u16 [%rd31], %h3;
    add.u32 %r21, %r21, 1;

    bra PAR_LIT_LOOP;

PAR_LIT_DONE:
    setp.ge.u32 %p9, %r20, %r6;
    @%p9 bra PAR_DONE;

    // Read offset
    cvt.u64.u32 %rd32, %r20;
    add.u64 %rd33, %rd19, %rd32;
    ld.global.u8 %rb3, [%rd33];
    cvt.u32.u8 %r41, %rb3;
    add.u32 %r20, %r20, 1;

    cvt.u64.u32 %rd34, %r20;
    add.u64 %rd35, %rd19, %rd34;
    ld.global.u8 %rb4, [%rd35];
    cvt.u32.u8 %r42, %rb4;
    add.u32 %r20, %r20, 1;

    shl.b32 %r43, %r42, 8;
    or.b32 %r44, %r41, %r43;  // offset in bytes

    and.b32 %r45, %r30, 15;
    add.u32 %r46, %r45, 4;  // match_len

    setp.ne.u32 %p10, %r45, 15;
    @%p10 bra PAR_SKIP_MATCH_EXT;

PAR_MATCH_EXT:
    cvt.u64.u32 %rd36, %r20;
    add.u64 %rd37, %rd19, %rd36;
    ld.global.u8 %rb5, [%rd37];
    cvt.u32.u8 %r47, %rb5;
    add.u32 %r20, %r20, 1;
    add.u32 %r46, %r46, %r47;
    setp.eq.u32 %p11, %r47, 255;
    @%p11 bra PAR_MATCH_EXT;

PAR_SKIP_MATCH_EXT:
    shl.b32 %r48, %r44, 1;  // F16 offset
    sub.u32 %r49, %r21, %r48;  // source F16 position
    mov.u32 %r50, 0;

PAR_MATCH_LOOP:
    setp.ge.u32 %p12, %r50, %r46;
    @%p12 bra PAR_MATCH_DONE;

    setp.ge.u32 %p13, %r21, %r7;
    @%p13 bra PAR_MATCH_DONE;

    shl.b32 %r51, %r50, 1;
    add.u32 %r52, %r49, %r51;

    shl.b32 %r53, %r52, 1;
    cvt.u64.u32 %rd38, %r53;
    add.u64 %rd39, %rd21, %rd38;
    ld.global.u16 %h4, [%rd39];

    shl.b32 %r54, %r21, 1;
    cvt.u64.u32 %rd40, %r54;
    add.u64 %rd41, %rd21, %rd40;
    st.global.u16 [%rd41], %h4;
    add.u32 %r21, %r21, 1;

    setp.ge.u32 %p14, %r21, %r7;
    @%p14 bra PAR_MATCH_DONE;

    add.u32 %r55, %r52, 1;
    shl.b32 %r56, %r55, 1;
    cvt.u64.u32 %rd42, %r56;
    add.u64 %rd43, %rd21, %rd42;
    ld.global.u16 %h5, [%rd43];

    shl.b32 %r57, %r21, 1;
    cvt.u64.u32 %rd44, %r57;
    add.u64 %rd45, %rd21, %rd44;
    st.global.u16 [%rd45], %h5;
    add.u32 %r21, %r21, 1;

    add.u32 %r50, %r50, 1;
    bra PAR_MATCH_LOOP;

PAR_MATCH_DONE:
    bra PAR_LOOP;

PAR_DONE:
    ret;
}
"#;

    /// Fused LZ4+INT8 kernel (simpler than INT4 since no nibble unpacking)
    const FUSED_LZ4_INT8_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// Fused LZ4 decompression + INT8 dequantization
.visible .entry fused_lz4_int8_block(
    .param .u64 input_ptr,
    .param .u32 input_size,
    .param .u64 output_ptr,
    .param .u32 num_values,
    .param .u32 scale_bits
)
{
    .reg .u64 %rd<24>;
    .reg .u32 %r<48>;
    .reg .f32 %f<6>;
    .reg .b16 %h<4>;
    .reg .pred %p<12>;
    .reg .b8 %rb<6>;

    ld.param.u64 %rd1, [input_ptr];
    ld.param.u32 %r1, [input_size];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r2, [num_values];
    ld.param.u32 %r3, [scale_bits];

    cvt.u16.u32 %h1, %r3;
    cvt.f32.f16 %f1, %h1;

    mov.u32 %r10, 0;  // in_pos
    mov.u32 %r11, 0;  // out_pos

INT8_LOOP:
    setp.ge.u32 %p1, %r11, %r2;
    @%p1 bra INT8_DONE;
    setp.ge.u32 %p2, %r10, %r1;
    @%p2 bra INT8_DONE;

    // Read token
    cvt.u64.u32 %rd3, %r10;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u8 %rb0, [%rd4];
    cvt.u32.u8 %r20, %rb0;
    add.u32 %r10, %r10, 1;

    shr.u32 %r21, %r20, 4;

    setp.ne.u32 %p3, %r21, 15;
    @%p3 bra INT8_SKIP_LIT_EXT;

INT8_LIT_EXT:
    cvt.u64.u32 %rd5, %r10;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %rb1, [%rd6];
    cvt.u32.u8 %r22, %rb1;
    add.u32 %r10, %r10, 1;
    add.u32 %r21, %r21, %r22;
    setp.eq.u32 %p4, %r22, 255;
    @%p4 bra INT8_LIT_EXT;

INT8_SKIP_LIT_EXT:
    mov.u32 %r23, 0;

INT8_LIT_LOOP:
    setp.ge.u32 %p5, %r23, %r21;
    @%p5 bra INT8_LIT_DONE;
    setp.ge.u32 %p6, %r11, %r2;
    @%p6 bra INT8_LIT_DONE;

    cvt.u64.u32 %rd7, %r10;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.s8 %r24, [%rd8];  // INT8 value (signed)
    add.u32 %r10, %r10, 1;
    add.u32 %r23, %r23, 1;

    // Dequantize: val * scale
    cvt.rn.f32.s32 %f2, %r24;
    mul.f32 %f3, %f2, %f1;
    cvt.rn.f16.f32 %h2, %f3;

    shl.b32 %r25, %r11, 1;
    cvt.u64.u32 %rd9, %r25;
    add.u64 %rd10, %rd2, %rd9;
    st.global.u16 [%rd10], %h2;
    add.u32 %r11, %r11, 1;

    bra INT8_LIT_LOOP;

INT8_LIT_DONE:
    setp.ge.u32 %p7, %r10, %r1;
    @%p7 bra INT8_DONE;

    // Read offset
    cvt.u64.u32 %rd11, %r10;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u8 %rb2, [%rd12];
    cvt.u32.u8 %r26, %rb2;
    add.u32 %r10, %r10, 1;

    cvt.u64.u32 %rd13, %r10;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u8 %rb3, [%rd14];
    cvt.u32.u8 %r27, %rb3;
    add.u32 %r10, %r10, 1;

    shl.b32 %r28, %r27, 8;
    or.b32 %r29, %r26, %r28;

    and.b32 %r30, %r20, 15;
    add.u32 %r31, %r30, 4;

    setp.ne.u32 %p8, %r30, 15;
    @%p8 bra INT8_SKIP_MATCH_EXT;

INT8_MATCH_EXT:
    cvt.u64.u32 %rd15, %r10;
    add.u64 %rd16, %rd1, %rd15;
    ld.global.u8 %rb4, [%rd16];
    cvt.u32.u8 %r32, %rb4;
    add.u32 %r10, %r10, 1;
    add.u32 %r31, %r31, %r32;
    setp.eq.u32 %p9, %r32, 255;
    @%p9 bra INT8_MATCH_EXT;

INT8_SKIP_MATCH_EXT:
    // For INT8, offset is directly in F16 units (1:1 mapping)
    sub.u32 %r33, %r11, %r29;
    mov.u32 %r34, 0;

INT8_MATCH_LOOP:
    setp.ge.u32 %p10, %r34, %r31;
    @%p10 bra INT8_MATCH_DONE;
    setp.ge.u32 %p11, %r11, %r2;
    @%p11 bra INT8_MATCH_DONE;

    add.u32 %r35, %r33, %r34;
    shl.b32 %r36, %r35, 1;
    cvt.u64.u32 %rd17, %r36;
    add.u64 %rd18, %rd2, %rd17;
    ld.global.u16 %h3, [%rd18];

    shl.b32 %r37, %r11, 1;
    cvt.u64.u32 %rd19, %r37;
    add.u64 %rd20, %rd2, %rd19;
    st.global.u16 [%rd20], %h3;
    add.u32 %r11, %r11, 1;
    add.u32 %r34, %r34, 1;

    bra INT8_MATCH_LOOP;

INT8_MATCH_DONE:
    bra INT8_LOOP;

INT8_DONE:
    ret;
}
"#;

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_core::DType;

        /// Helper to check if CUDA is available.
        fn cuda_available() -> bool {
            GpuFusedContext::new(0).is_ok()
        }

        /// Creates LZ4-compressed INT4 data for testing.
        /// Returns (compressed, num_output_f16_values)
        fn create_lz4_int4_test_data(int4_values: &[u8]) -> (Vec<u8>, usize) {
            // Pack INT4 values into bytes (2 per byte, low nibble first)
            let packed: Vec<u8> = int4_values
                .chunks(2)
                .map(|chunk| {
                    let low = chunk[0] & 0x0F;
                    let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
                    (high << 4) | low
                })
                .collect();

            // Create literals-only LZ4 block
            let mut compressed = Vec::new();
            let len = packed.len();

            if len <= 14 {
                compressed.push((len as u8) << 4);
                compressed.extend_from_slice(&packed);
            } else if len <= 269 {
                compressed.push(0xF0);
                compressed.push((len - 15) as u8);
                compressed.extend_from_slice(&packed);
            } else {
                compressed.push(0xF0);
                let mut remaining = len - 15;
                while remaining >= 255 {
                    compressed.push(255);
                    remaining -= 255;
                }
                compressed.push(remaining as u8);
                compressed.extend_from_slice(&packed);
            }

            (compressed, int4_values.len())
        }

        /// Creates LZ4-compressed INT8 data for testing.
        fn create_lz4_int8_test_data(int8_values: &[i8]) -> (Vec<u8>, usize) {
            // INT8 values are already 1 byte each
            let bytes: Vec<u8> = int8_values.iter().map(|&v| v as u8).collect();

            let mut compressed = Vec::new();
            let len = bytes.len();

            if len <= 14 {
                compressed.push((len as u8) << 4);
                compressed.extend_from_slice(&bytes);
            } else if len <= 269 {
                compressed.push(0xF0);
                compressed.push((len - 15) as u8);
                compressed.extend_from_slice(&bytes);
            } else {
                compressed.push(0xF0);
                let mut remaining = len - 15;
                while remaining >= 255 {
                    compressed.push(255);
                    remaining -= 255;
                }
                compressed.push(remaining as u8);
                compressed.extend_from_slice(&bytes);
            }

            (compressed, int8_values.len())
        }

        #[test]
        fn test_context_creation() {
            match GpuFusedContext::new(0) {
                Ok(ctx) => {
                    assert_eq!(ctx.device_id(), 0);
                    assert!(!ctx.lz4_int4_kernel_loaded);
                    assert!(!ctx.lz4_int8_kernel_loaded);
                },
                Err(GpuFusedError::DeviceInit { .. }) => {
                    eprintln!("Skipping: no CUDA device");
                },
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[test]
        fn test_lz4_int4_kernel_loading() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().expect("kernel load");
            assert!(ctx.lz4_int4_kernel_loaded);

            // Second load should be no-op
            ctx.load_lz4_int4_kernel().expect("second load");
        }

        #[test]
        fn test_fused_lz4_int4_basic() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().unwrap();

            // INT4 values 0-7
            let int4_values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let (compressed, num_values) = create_lz4_int4_test_data(&int4_values);

            // Scale = 0.5, zero_point = 0
            // Expected: 0*0.5, 1*0.5, ... = 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
            let scale = half::f16::from_f32(0.5);

            let result = ctx
                .fused_lz4_int4_block(&compressed, scale, 0, num_values)
                .unwrap();

            let mut host_result = vec![half::f16::ZERO; num_values];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            let expected: Vec<f32> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
            for (i, (got, exp)) in host_result.iter().zip(expected.iter()).enumerate() {
                let got_f32 = got.to_f32();
                assert!(
                    (got_f32 - exp).abs() < 0.01,
                    "Mismatch at {}: got {}, expected {}",
                    i,
                    got_f32,
                    exp
                );
            }
        }

        #[test]
        fn test_fused_lz4_int4_with_zero_point() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().unwrap();

            // INT4 values 0-7, with zero_point = 4
            let int4_values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let (compressed, num_values) = create_lz4_int4_test_data(&int4_values);

            let scale = half::f16::from_f32(1.0);
            let zero_point = 4i8;

            let result = ctx
                .fused_lz4_int4_block(&compressed, scale, zero_point, num_values)
                .unwrap();

            let mut host_result = vec![half::f16::ZERO; num_values];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // Expected: (0-4)*1, (1-4)*1, ... = -4, -3, -2, -1, 0, 1, 2, 3
            let expected: Vec<f32> = vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
            for (i, (got, exp)) in host_result.iter().zip(expected.iter()).enumerate() {
                let got_f32 = got.to_f32();
                assert!(
                    (got_f32 - exp).abs() < 0.01,
                    "Mismatch at {}: got {}, expected {}",
                    i,
                    got_f32,
                    exp
                );
            }
        }

        #[test]
        fn test_fused_lz4_int4_parallel() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().unwrap();

            // Create multiple blocks
            let block1_values: Vec<u8> = vec![0, 1, 2, 3];
            let block2_values: Vec<u8> = vec![4, 5, 6, 7, 8, 9, 10, 11];

            let (comp1, n1) = create_lz4_int4_test_data(&block1_values);
            let (comp2, n2) = create_lz4_int4_test_data(&block2_values);

            let blocks = vec![(comp1, n1), (comp2, n2)];
            let scales = vec![
                half::f16::from_f32(0.1), // Block 1 scale
                half::f16::from_f32(0.2), // Block 2 scale
            ];

            let result = ctx.fused_lz4_int4_parallel(&blocks, &scales, None).unwrap();

            let total = n1 + n2;
            let mut host_result = vec![half::f16::ZERO; total];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // Block 1: 0*0.1, 1*0.1, 2*0.1, 3*0.1 = 0.0, 0.1, 0.2, 0.3
            // Block 2: 4*0.2, 5*0.2, ... = 0.8, 1.0, 1.2, ...
            assert!((host_result[0].to_f32() - 0.0).abs() < 0.01);
            assert!((host_result[1].to_f32() - 0.1).abs() < 0.01);
            assert!((host_result[4].to_f32() - 0.8).abs() < 0.02); // Block 2 starts
        }

        #[test]
        fn test_fused_lz4_int4_to_tensor() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().unwrap();

            let int4_values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let (compressed, num_values) = create_lz4_int4_test_data(&int4_values);

            let blocks = vec![(compressed, num_values)];
            let scales = vec![half::f16::from_f32(1.0)];

            let tensor = ctx
                .fused_lz4_int4_to_tensor(&blocks, &scales, None, &[2, 4])
                .unwrap();

            assert_eq!(tensor.dims(), &[2, 4]);

            let data: Vec<Vec<f32>> = tensor.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
            assert_eq!(data[0], vec![0.0, 1.0, 2.0, 3.0]);
            assert_eq!(data[1], vec![4.0, 5.0, 6.0, 7.0]);
        }

        #[test]
        fn test_fused_lz4_int8_basic() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int8_kernel().unwrap();

            let int8_values: Vec<i8> = vec![-128, -64, 0, 32, 64, 96, 127, 100];
            let (compressed, num_values) = create_lz4_int8_test_data(&int8_values);

            let scale = half::f16::from_f32(0.01);

            let result = ctx
                .fused_lz4_int8_block(&compressed, scale, num_values)
                .unwrap();

            let mut host_result = vec![half::f16::ZERO; num_values];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // Expected: values * 0.01
            let expected: Vec<f32> = vec![-1.28, -0.64, 0.0, 0.32, 0.64, 0.96, 1.27, 1.0];
            for (i, (got, exp)) in host_result.iter().zip(expected.iter()).enumerate() {
                let got_f32 = got.to_f32();
                assert!(
                    (got_f32 - exp).abs() < 0.02,
                    "Mismatch at {}: got {}, expected {}",
                    i,
                    got_f32,
                    exp
                );
            }
        }

        #[test]
        fn test_fused_empty_blocks_error() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int4_kernel().unwrap();

            let empty: Vec<(Vec<u8>, usize)> = vec![];
            let scales: Vec<half::f16> = vec![];

            let result = ctx.fused_lz4_int4_parallel(&empty, &scales, None);
            match result {
                Err(GpuFusedError::InvalidInput { message }) => {
                    assert!(message.contains("No blocks"));
                },
                _ => panic!("Expected InvalidInput error"),
            }
        }

        #[test]
        fn test_fused_kernel_not_loaded_error() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let ctx = GpuFusedContext::new(0).unwrap();
            // Don't load kernel

            let (compressed, num_values) = create_lz4_int4_test_data(&[0, 1, 2, 3]);
            let scale = half::f16::from_f32(1.0);

            let result = ctx.fused_lz4_int4_block(&compressed, scale, 0, num_values);
            match result {
                Err(GpuFusedError::KernelNotLoaded { kernel }) => {
                    assert!(kernel.contains("fused_lz4_int4"));
                },
                _ => panic!("Expected KernelNotLoaded error"),
            }
        }

        #[test]
        fn test_lz4_int4_test_data_helper() {
            // Verify our test data helper creates valid LZ4
            let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let (compressed, num) = create_lz4_int4_test_data(&values);

            assert_eq!(num, 8);
            // 8 INT4 values = 4 packed bytes
            // LZ4: 1 token + 4 literals = 5 bytes
            assert_eq!(compressed.len(), 5);
            assert_eq!(compressed[0], 0x40); // 4 literals, 0 match
        }

        // ==================== Phase 4: Fusion Equivalence Tests ====================
        // Trust boundary §5 (Fusion Equivalence) from GPU-CODEC-PIPELINE-TDD.md.
        //
        // Property: fused_lz4_int4(data) == dequant(lz4_decompress(data))
        // The fused kernel must produce bit-identical results to the sequential pipeline.

        /// Fusion equivalence: fused LZ4+INT4 produces same F16 output as
        /// sequential LZ4 decompress → INT4 dequant.
        #[test]
        fn test_fused_lz4_int4_matches_sequential() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut fused_ctx = GpuFusedContext::new(0).unwrap();
            fused_ctx.load_lz4_int4_kernel().unwrap();

            // Create INT4 test data: 8 known nibble values
            let nibbles: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let (compressed, num_values) = create_lz4_int4_test_data(&nibbles);
            let scale = half::f16::from_f32(0.5);

            // === Fused path ===
            // zero_point=8 because pack_int4_signed adds 8 to convert [-8,7] → [0,15],
            // and the fused kernel does (nibble - zero_point) * scale.
            let fused_result = fused_ctx
                .fused_lz4_int4_block(&compressed, scale, 8, num_values)
                .expect("fused kernel");

            let mut fused_host = vec![half::f16::ZERO; num_values];
            fused_ctx
                .device
                .dtoh_sync_copy_into(&fused_result, &mut fused_host)
                .expect("copy fused");

            // === Sequential path (CPU reference) ===
            // Step 1: LZ4 decompress (extract raw packed bytes)
            // The compressed data IS the packed INT4 bytes wrapped in LZ4.
            // For create_lz4_int4_test_data, the LZ4 payload is the packed nibbles.
            let packed_size = (num_values + 1) / 2;
            // Parse LZ4: token byte says how many literal bytes follow
            let token = compressed[0];
            let lit_len = (token >> 4) as usize;
            let decompressed_packed = &compressed[1..1 + lit_len];

            // Step 2: INT4 dequant (CPU reference)
            let mut sequential_host = Vec::with_capacity(num_values);
            for i in 0..num_values {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    decompressed_packed[byte_idx] & 0x0F
                } else {
                    (decompressed_packed[byte_idx] >> 4) & 0x0F
                };
                // Symmetric dequant: (nibble - 8) * scale
                let val = ((nibble as i32) - 8) as f32 * scale.to_f32();
                sequential_host.push(half::f16::from_f32(val));
            }

            // === Compare ===
            assert_eq!(fused_host.len(), sequential_host.len(), "length mismatch");
            for (i, (f, s)) in fused_host.iter().zip(sequential_host.iter()).enumerate() {
                assert_eq!(
                    f.to_f32(),
                    s.to_f32(),
                    "Element {}: fused={} vs sequential={} (nibble={})",
                    i,
                    f.to_f32(),
                    s.to_f32(),
                    nibbles[i]
                );
            }
        }

        /// Fusion equivalence: fused LZ4+INT8 matches sequential path.
        #[test]
        fn test_fused_lz4_int8_matches_sequential() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuFusedContext::new(0).unwrap();
            ctx.load_lz4_int8_kernel().unwrap();

            // Create INT8 test data: 8 known values
            let int8_values: Vec<i8> = vec![-128, -64, -1, 0, 1, 42, 100, 127];
            let (compressed, num_values) = create_lz4_int8_test_data(&int8_values);
            let scale = half::f16::from_f32(0.1);

            // Fused path
            let fused_result = ctx
                .fused_lz4_int8_block(&compressed, scale, num_values)
                .expect("fused INT8 kernel");

            let mut fused_host = vec![half::f16::ZERO; num_values];
            ctx.device
                .dtoh_sync_copy_into(&fused_result, &mut fused_host)
                .expect("copy fused");

            // Sequential CPU reference
            let sequential: Vec<half::f16> = int8_values
                .iter()
                .map(|&v| {
                    let val = (v as f32) * scale.to_f32();
                    half::f16::from_f32(val)
                })
                .collect();

            for (i, (f, s)) in fused_host.iter().zip(sequential.iter()).enumerate() {
                let diff = (f.to_f32() - s.to_f32()).abs();
                assert!(
                    diff < 0.01,
                    "INT8 element {}: fused={} vs sequential={} (input={})",
                    i,
                    f.to_f32(),
                    s.to_f32(),
                    int8_values[i]
                );
            }
        }

        // ==================== Phase 4: Pipeline Integration Tests ====================
        // GPU-CODEC-PIPELINE-TDD.md §8.1, §8.2

        /// §8.1 + §8.2: Pipeline A (sequential LZ4→INT4) matches Pipeline B (fused LZ4+INT4).
        ///
        /// Both pipelines start from the same LZ4-compressed INT4 data and must produce
        /// identical F16 output, proving the fused optimization is semantically equivalent.
        #[test]
        fn test_pipeline_a_vs_b_sequential_vs_fused_int4() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            use crate::quantize::{Quantizer, DEFAULT_BLOCK_SIZE};
            use candle_core::{Device, Tensor};

            // 1. Create realistic tensor data and quantize
            let original: Vec<f32> = (0..512).map(|i| ((i as f32) / 512.0 - 0.5) * 2.0).collect();
            let tensor = Tensor::from_vec(original.clone(), &[512], &Device::Cpu).unwrap();
            let quantizer = Quantizer::int4_symmetric();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();

            // 2. LZ4-compress the packed INT4 data
            let compressed = lz4_flex::block::compress_prepend_size(&quantized.data);
            // Strip the 4-byte size prefix that lz4_flex adds
            let lz4_data = &compressed[4..];

            // 3. Pipeline A: sequential LZ4 decompress → INT4 dequant
            let mut lz4_ctx = crate::gpu_lz4::GpuLz4Context::new(0).unwrap();
            lz4_ctx.load_kernel().unwrap();
            let d_decompressed = lz4_ctx
                .decompress_block(lz4_data, quantized.data.len())
                .unwrap();
            let mut decompressed_host = vec![0u8; quantized.data.len()];
            lz4_ctx
                .cuda_device()
                .dtoh_sync_copy_into(&d_decompressed, &mut decompressed_host)
                .unwrap();

            // INT4 dequant (sequential path via CPU, since Pipeline A reference is CPU)
            let num_blocks = (quantized.num_values + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
            let mut sequential_f16: Vec<half::f16> = Vec::with_capacity(quantized.num_values);
            for i in 0..quantized.num_values {
                let byte_idx = i / 2;
                let nibble = if i % 2 == 0 {
                    decompressed_host[byte_idx] & 0x0F
                } else {
                    (decompressed_host[byte_idx] >> 4) & 0x0F
                };
                let block_idx = i / DEFAULT_BLOCK_SIZE;
                let scale = quantized.scales[block_idx].to_f32();
                let val = ((nibble as i32) - 8) as f32 * scale;
                sequential_f16.push(half::f16::from_f32(val));
            }

            // 4. Pipeline B: fused LZ4+INT4
            let mut fused_ctx = GpuFusedContext::new(0).unwrap();
            fused_ctx.load_lz4_int4_kernel().unwrap();

            // Fused kernel processes one block_size at a time with a single scale
            // For multi-block data, we process block by block
            let block_packed_size = DEFAULT_BLOCK_SIZE / 2; // 64 bytes per block of 128 nibbles
            let mut fused_f16: Vec<half::f16> = Vec::new();

            for block_idx in 0..num_blocks {
                let start = block_idx * block_packed_size;
                let end = ((block_idx + 1) * block_packed_size).min(quantized.data.len());
                let block_packed = &quantized.data[start..end];
                let block_num_values = if block_idx == num_blocks - 1 {
                    quantized.num_values - block_idx * DEFAULT_BLOCK_SIZE
                } else {
                    DEFAULT_BLOCK_SIZE
                };

                // LZ4-compress each block individually for fused kernel
                let block_compressed = lz4_flex::block::compress_prepend_size(block_packed);
                let block_lz4 = &block_compressed[4..];

                let fused_result = fused_ctx
                    .fused_lz4_int4_block(
                        block_lz4,
                        quantized.scales[block_idx],
                        8, // symmetric offset
                        block_num_values,
                    )
                    .unwrap();

                let mut block_host = vec![half::f16::ZERO; block_num_values];
                fused_ctx
                    .device
                    .dtoh_sync_copy_into(&fused_result, &mut block_host)
                    .unwrap();
                fused_f16.extend_from_slice(&block_host);
            }

            // 5. Compare
            assert_eq!(sequential_f16.len(), fused_f16.len(), "length mismatch");
            let max_diff: f32 = sequential_f16
                .iter()
                .zip(fused_f16.iter())
                .map(|(s, f)| (s.to_f32() - f.to_f32()).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 0.01,
                "Pipeline A vs B max diff: {} (should be < 0.01)",
                max_diff
            );
        }

        /// §8.4: Pipeline F — HoloTensor LRDF end-to-end.
        ///
        /// Full pipeline: encode with GPU LRDF encoder → reconstruct with GPU HoloContext.
        /// Tests that the encode/decode round-trip preserves reasonable quality.
        #[test]
        fn test_pipeline_f_holotensor_lrdf_roundtrip() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            use crate::gpu_holo::GpuHoloContext;
            use crate::gpu_lrdf::cuda::GpuLrdfEncoder;

            // 1. Create a known matrix
            let rows = 16;
            let cols = 16;
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let r = (i / cols) as f32;
                    let c = (i % cols) as f32;
                    (r * 0.3).sin() * (c * 0.2).cos()
                })
                .collect();

            // 2. Encode with GPU LRDF encoder
            let device = cudarc::driver::CudaDevice::new(0).unwrap();
            let encoder = GpuLrdfEncoder::new(device, 4, 42).unwrap().with_max_rank(8);
            let gpu_fragments = encoder.encode_2d(&data, rows, cols).unwrap();
            let holo_fragments: Vec<_> = gpu_fragments.iter().map(|f| f.to_haagenti()).collect();

            // 3. Reconstruct with GpuHoloContext
            let mut ctx = GpuHoloContext::new(0).unwrap();
            ctx.load_lrdf_kernel().unwrap();

            let tensor = ctx.reconstruct_lrdf(&holo_fragments, rows, cols).unwrap();
            let reconstructed = tensor.to_host().unwrap();

            // 4. Verify quality: cosine similarity should be > 0.8 for rank-8 on 16x16
            assert_eq!(reconstructed.len(), data.len());
            let dot: f32 = data
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
            let similarity = if norm_a > 1e-10 && norm_b > 1e-10 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            };

            assert!(
                similarity > 0.8,
                "LRDF round-trip quality too low: cosine_sim={:.4} (expected > 0.8)",
                similarity
            );
        }

        // ==================== Phase 4: §5.3 Fusion Bit-Identical Property Test ====================
        // GPU-CODEC-PIPELINE-TDD.md §5.3: Fused LZ4+INT4 produces bit-identical
        // output to the sequential CPU dequant path for arbitrary inputs.

        mod fusion_proptest {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(20))]
                #[test]
                fn fused_matches_cpu_sequential_arbitrary(
                    nibbles in proptest::collection::vec(0u8..16u8, 2..128),
                    scale_f32 in 0.01f32..5.0f32,
                    zero_point in 0i8..=15i8,
                ) {
                    if !cuda_available() {
                        return Ok(());
                    }

                    let scale = half::f16::from_f32(scale_f32);
                    let num_values = nibbles.len();
                    let (compressed, _) = create_lz4_int4_test_data(&nibbles);

                    // Fused GPU path
                    let mut ctx = GpuFusedContext::new(0).unwrap();
                    ctx.load_lz4_int4_kernel().unwrap();
                    let fused_result = ctx
                        .fused_lz4_int4_block(&compressed, scale, zero_point, num_values)
                        .unwrap();
                    let mut fused_host = vec![half::f16::ZERO; num_values];
                    ctx.device
                        .dtoh_sync_copy_into(&fused_result, &mut fused_host)
                        .unwrap();

                    // Sequential CPU reference: (nibble - zero_point) * scale → F16
                    for (i, &nibble) in nibbles.iter().enumerate() {
                        let expected = half::f16::from_f32(
                            (nibble as i32 - zero_point as i32) as f32 * scale.to_f32(),
                        );
                        prop_assert_eq!(
                            fused_host[i].to_bits(),
                            expected.to_bits(),
                            "Fusion bit mismatch at {}: nibble={}, zp={}, scale={:.4}, \
                             fused={:?}, expected={:?}",
                            i, nibble, zero_point, scale.to_f32(), fused_host[i], expected
                        );
                    }
                }
            }
        }

        // ==================== Phase 4: §8.3 Pipeline E — FP8 Conversion ====================
        // GPU-CODEC-PIPELINE-TDD.md §8.3: End-to-end FP8 E4M3 → F32 pipeline
        // verifying GPU conversion matches hand-computed known values.

        #[test]
        fn test_pipeline_e_fp8_e4m3_known_values() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            use std::sync::Arc;

            let device = match cudarc::driver::CudaDevice::new(0) {
                Ok(d) => Arc::new(d),
                Err(_) => {
                    eprintln!("Skipping: no CUDA device");
                    return;
                },
            };

            let converter = crate::gpu_dtype::cuda::GpuDtypeConverter::new(Arc::clone(&device))
                .expect("FP8 converter creation");

            // Representative FP8 E4M3 values spanning the encoding range:
            // (byte, expected_f32)
            let test_cases: Vec<(u8, f32)> = vec![
                (0x00, 0.0),         // +0
                (0x38, 1.0),         // 2^(7-7) * (1+0/8) = 1.0
                (0xB8, -1.0),        // -1.0
                (0x3C, 1.5),         // 2^(7-7) * (1+4/8) = 1.5
                (0x40, 2.0),         // 2^(8-7) * (1+0/8) = 2.0
                (0x01, 0.001953125), // subnormal: 2^(-6) * (1/8) = 1/512
                (0x77, 240.0),       // max normal: 2^(14-7) * (1+7/8) = 128 * 1.875 = 240
            ];

            let fp8_bytes: Vec<u8> = test_cases.iter().map(|(b, _)| *b).collect();
            let gpu_result = converter
                .fp8_e4m3_to_f32_host(&fp8_bytes)
                .expect("GPU FP8 conversion");

            for (i, ((byte, expected), &gpu_val)) in
                test_cases.iter().zip(gpu_result.iter()).enumerate()
            {
                assert!(
                    (gpu_val - expected).abs() < 1e-7,
                    "Pipeline E: FP8 byte 0x{:02X} at index {}: expected={}, gpu={}",
                    byte,
                    i,
                    expected,
                    gpu_val,
                );
            }

            // Verify batch conversion: all 254 non-NaN bytes match CPU reference
            let all_bytes: Vec<u8> = (0u8..=0xFEu8).collect(); // skip 0x7F (NaN)
            let all_gpu = converter
                .fp8_e4m3_to_f32_host(&all_bytes)
                .expect("batch GPU conversion");

            for (byte, &gpu_val) in all_bytes.iter().zip(all_gpu.iter()) {
                // Inline CPU reference for FP8 E4M3
                let sign = (byte >> 7) & 1;
                let exp = (byte >> 3) & 0xF;
                let man = byte & 0x7;

                let cpu_val = if exp == 0 {
                    if man == 0 {
                        if sign == 1 {
                            -0.0
                        } else {
                            0.0
                        }
                    } else {
                        let v = (man as f32 / 8.0) * 2.0f32.powi(-6);
                        if sign == 1 {
                            -v
                        } else {
                            v
                        }
                    }
                } else if exp == 15 && man == 7 {
                    f32::NAN
                } else {
                    let v = (1.0 + man as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7);
                    if sign == 1 {
                        -v
                    } else {
                        v
                    }
                };

                if cpu_val.is_nan() {
                    assert!(
                        gpu_val.is_nan(),
                        "0x{:02X}: expected NaN, got {}",
                        byte,
                        gpu_val
                    );
                } else {
                    assert_eq!(
                        gpu_val.to_bits(),
                        cpu_val.to_bits(),
                        "Pipeline E batch: byte 0x{:02X}, cpu={}, gpu={}",
                        byte,
                        cpu_val,
                        gpu_val,
                    );
                }
            }
        }
    }
}

/// Stub module when CUDA is not enabled.
#[cfg(not(feature = "cuda"))]
pub mod cuda {

    /// Fused context (stub).
    pub struct GpuFusedContext;

    /// Errors from fused operations (stub).
    #[derive(Debug, thiserror::Error)]
    pub enum GpuFusedError {
        /// CUDA not enabled.
        #[error("CUDA not enabled - compile with --features cuda")]
        CudaNotEnabled,
    }

    impl GpuFusedContext {
        /// Always returns error when CUDA not enabled.
        pub fn new(_device_id: usize) -> Result<Self, GpuFusedError> {
            Err(GpuFusedError::CudaNotEnabled)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_stub_returns_error() {
            match GpuFusedContext::new(0) {
                Err(GpuFusedError::CudaNotEnabled) => {},
                Ok(_) => panic!("Stub should error"),
            }
        }
    }
}

pub use cuda::GpuFusedContext;
#[cfg(feature = "cuda")]
pub use cuda::GpuFusedError;
