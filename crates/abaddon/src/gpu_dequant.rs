//! GPU-accelerated dequantization kernels for quantized model weights.
//!
//! This module provides CUDA-accelerated INT4 and INT8 dequantization for
//! compressed quantized weights. Dequantization converts low-precision integer
//! weights back to floating-point for inference.
//!
//! ## Quantization Formats Supported
//!
//! ### INT4 (4-bit)
//! - Two INT4 values packed per byte (little-endian: low nibble first)
//! - Per-block scaling (typically 32 or 128 values per scale)
//! - Optional zero-point for asymmetric quantization
//! - Compatible with GPTQ, AWQ, and GGML Q4_0/Q4_1 formats
//!
//! ### INT8 (8-bit)
//! - One INT8 value per byte (signed: -128 to 127)
//! - Per-tensor or per-channel scaling
//! - Compatible with standard INT8 quantization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GPU Dequantization                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Packed INT4/INT8 (GPU)        Dequantized F16 (GPU)           │
//! │  ┌──────────────────┐          ┌──────────────────────┐        │
//! │  │ [0x10][0x32]...  │  ──────> │ [0.0][0.5][1.0]...   │        │
//! │  └──────────────────┘    │     └──────────────────────┘        │
//! │                          │                                      │
//! │                     Scale + ZeroPoint                           │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

/// CUDA-accelerated INT4/INT8 dequantization implementation.
#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;

    use candle_core::{Device, Tensor};
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;

    /// Block size for INT4 dequantization (HCT-native).
    /// Re-exports the module-level constant for backward compatibility.
    pub const INT4_BLOCK_SIZE: usize = super::INT4_BLOCK_SIZE;

    /// GPU dequantization context.
    ///
    /// Holds compiled CUDA kernels for INT4/INT8 dequantization.
    pub struct GpuDequantContext {
        device: Arc<CudaDevice>,
        device_id: usize,
        int4_kernel_loaded: bool,
        int8_kernel_loaded: bool,
    }

    impl GpuDequantContext {
        /// Creates a new GPU dequantization context for the specified device.
        pub fn new(device_id: usize) -> Result<Self, GpuDequantError> {
            let device = CudaDevice::new(device_id).map_err(|e| GpuDequantError::DeviceInit {
                device_id,
                message: e.to_string(),
            })?;

            Ok(Self {
                device,
                device_id,
                int4_kernel_loaded: false,
                int8_kernel_loaded: false,
            })
        }

        /// Returns the CUDA device ID.
        pub fn device_id(&self) -> usize {
            self.device_id
        }

        /// Loads the INT4 dequantization kernel.
        pub fn load_int4_kernel(&mut self) -> Result<(), GpuDequantError> {
            if self.int4_kernel_loaded {
                return Ok(());
            }

            let ptx = Ptx::from_src(INT4_DEQUANT_KERNEL_PTX);
            self.device
                .load_ptx(
                    ptx,
                    "int4_dequant",
                    &["int4_dequant_block", "int4_dequant_tensor"],
                )
                .map_err(|e| GpuDequantError::KernelLoad {
                    message: e.to_string(),
                })?;

            self.int4_kernel_loaded = true;
            Ok(())
        }

        /// Loads the INT8 dequantization kernel.
        pub fn load_int8_kernel(&mut self) -> Result<(), GpuDequantError> {
            if self.int8_kernel_loaded {
                return Ok(());
            }

            let ptx = Ptx::from_src(INT8_DEQUANT_KERNEL_PTX);
            self.device
                .load_ptx(ptx, "int8_dequant", &["int8_dequant_tensor"])
                .map_err(|e| GpuDequantError::KernelLoad {
                    message: e.to_string(),
                })?;

            self.int8_kernel_loaded = true;
            Ok(())
        }

        /// Dequantizes INT4 packed data to F16.
        ///
        /// # Arguments
        ///
        /// * `packed` - Packed INT4 data (2 values per byte, low nibble first)
        /// * `scales` - Per-block scale factors (one per INT4_BLOCK_SIZE values)
        /// * `zero_points` - Optional per-block zero points (None for symmetric)
        /// * `num_values` - Total number of INT4 values
        ///
        /// # Returns
        ///
        /// GPU buffer containing dequantized F16 values.
        pub fn dequant_int4(
            &self,
            packed: &[u8],
            scales: &[half::f16],
            zero_points: Option<&[i8]>,
            num_values: usize,
        ) -> Result<CudaSlice<half::f16>, GpuDequantError> {
            if !self.int4_kernel_loaded {
                return Err(GpuDequantError::KernelNotLoaded {
                    kernel: "int4_dequant".to_string(),
                });
            }

            // Validate input sizes
            let expected_packed = (num_values + 1) / 2; // 2 values per byte
            if packed.len() < expected_packed {
                return Err(GpuDequantError::InvalidInput {
                    message: format!(
                        "Packed data too small: got {} bytes, expected {} for {} values",
                        packed.len(),
                        expected_packed,
                        num_values
                    ),
                });
            }

            let num_blocks = (num_values + INT4_BLOCK_SIZE - 1) / INT4_BLOCK_SIZE;
            if scales.len() < num_blocks {
                return Err(GpuDequantError::InvalidInput {
                    message: format!(
                        "Not enough scales: got {}, expected {} for {} values",
                        scales.len(),
                        num_blocks,
                        num_values
                    ),
                });
            }

            // Copy inputs to GPU
            let d_packed = self.device.htod_copy(packed.to_vec()).map_err(|e| {
                GpuDequantError::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            // Convert scales to raw bytes for transfer
            let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            let d_scales =
                self.device
                    .htod_copy(scales_bytes)
                    .map_err(|e| GpuDequantError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Handle zero points (use zeros if not provided)
            let zp_vec: Vec<i8> = zero_points
                .map(|zp| zp.to_vec())
                .unwrap_or_else(|| vec![0i8; num_blocks]);
            let d_zero_points =
                self.device
                    .htod_copy(zp_vec)
                    .map_err(|e| GpuDequantError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Allocate output buffer
            let d_output: CudaSlice<half::f16> =
                self.device
                    .alloc_zeros(num_values)
                    .map_err(|e| GpuDequantError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Launch kernel
            let func = self
                .device
                .get_func("int4_dequant", "int4_dequant_tensor")
                .ok_or_else(|| GpuDequantError::KernelLoad {
                    message: "int4_dequant_tensor not found".to_string(),
                })?;

            // One thread per output value, organized in blocks of 256
            let threads_per_block = 256u32;
            let num_thread_blocks =
                ((num_values as u32) + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_thread_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_packed,
                        &d_scales,
                        &d_zero_points,
                        &d_output,
                        num_values as u32,
                        INT4_BLOCK_SIZE as u32,
                    ),
                )
            }
            .map_err(|e| GpuDequantError::KernelExec {
                message: e.to_string(),
            })?;

            self.device
                .synchronize()
                .map_err(|e| GpuDequantError::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Dequantizes INT4 data directly to a Candle tensor.
        pub fn dequant_int4_to_tensor(
            &self,
            packed: &[u8],
            scales: &[half::f16],
            zero_points: Option<&[i8]>,
            shape: &[usize],
        ) -> Result<Tensor, GpuDequantError> {
            let num_values: usize = shape.iter().product();
            let d_output = self.dequant_int4(packed, scales, zero_points, num_values)?;

            // Copy back to host and create tensor
            let mut host_data = vec![half::f16::ZERO; num_values];
            self.device
                .dtoh_sync_copy_into(&d_output, &mut host_data)
                .map_err(|e| GpuDequantError::MemoryCopy {
                    message: e.to_string(),
                })?;

            Tensor::from_vec(host_data, shape, &Device::Cpu).map_err(|e| {
                GpuDequantError::TensorCreate {
                    message: e.to_string(),
                }
            })
        }

        /// Dequantizes INT8 data to F16.
        ///
        /// # Arguments
        ///
        /// * `data` - INT8 values (signed: -128 to 127)
        /// * `scale` - Scale factor for dequantization
        ///
        /// # Returns
        ///
        /// GPU buffer containing dequantized F16 values.
        pub fn dequant_int8(
            &self,
            data: &[i8],
            scale: half::f16,
        ) -> Result<CudaSlice<half::f16>, GpuDequantError> {
            if !self.int8_kernel_loaded {
                return Err(GpuDequantError::KernelNotLoaded {
                    kernel: "int8_dequant".to_string(),
                });
            }

            let num_values = data.len();

            // Copy input to GPU
            let d_input =
                self.device
                    .htod_copy(data.to_vec())
                    .map_err(|e| GpuDequantError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Allocate output
            let d_output: CudaSlice<half::f16> =
                self.device
                    .alloc_zeros(num_values)
                    .map_err(|e| GpuDequantError::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Launch kernel
            let func = self
                .device
                .get_func("int8_dequant", "int8_dequant_tensor")
                .ok_or_else(|| GpuDequantError::KernelLoad {
                    message: "int8_dequant_tensor not found".to_string(),
                })?;

            let threads_per_block = 256u32;
            let num_blocks = ((num_values as u32) + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };

            // Pass scale as raw bits
            let scale_bits = scale.to_bits();

            unsafe { func.launch(cfg, (&d_input, &d_output, scale_bits, num_values as u32)) }
                .map_err(|e| GpuDequantError::KernelExec {
                    message: e.to_string(),
                })?;

            self.device
                .synchronize()
                .map_err(|e| GpuDequantError::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Dequantizes INT8 data directly to a Candle tensor.
        pub fn dequant_int8_to_tensor(
            &self,
            data: &[i8],
            scale: half::f16,
            shape: &[usize],
        ) -> Result<Tensor, GpuDequantError> {
            let d_output = self.dequant_int8(data, scale)?;

            let num_values = data.len();
            let mut host_data = vec![half::f16::ZERO; num_values];
            self.device
                .dtoh_sync_copy_into(&d_output, &mut host_data)
                .map_err(|e| GpuDequantError::MemoryCopy {
                    message: e.to_string(),
                })?;

            Tensor::from_vec(host_data, shape, &Device::Cpu).map_err(|e| {
                GpuDequantError::TensorCreate {
                    message: e.to_string(),
                }
            })
        }
    }

    /// Errors from GPU dequantization operations.
    #[derive(Debug, thiserror::Error)]
    pub enum GpuDequantError {
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

    /// INT4 dequantization kernel in PTX.
    ///
    /// Unpacks 2 INT4 values per byte and applies: output = (int4_val - zero_point) * scale
    const INT4_DEQUANT_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// INT4 dequantization kernel
// Each thread processes one output value
// Input: packed INT4 (2 per byte), scales (per block), zero points (per block)
// Output: F16 dequantized values
.visible .entry int4_dequant_tensor(
    .param .u64 packed_ptr,      // Packed INT4 data
    .param .u64 scales_ptr,      // Per-block scales (F16)
    .param .u64 zp_ptr,          // Per-block zero points (i8)
    .param .u64 output_ptr,      // Output F16 data
    .param .u32 num_values,      // Total number of output values
    .param .u32 block_size       // Values per quantization block
)
{
    .reg .u64 %rd<16>;
    .reg .u32 %r<24>;
    .reg .f32 %f<8>;
    .reg .b16 %h<4>;
    .reg .pred %p<4>;

    // Calculate global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;  // global_idx

    // Bounds check
    ld.param.u32 %r5, [num_values];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra DONE;

    // Load parameters
    ld.param.u64 %rd1, [packed_ptr];
    ld.param.u64 %rd2, [scales_ptr];
    ld.param.u64 %rd3, [zp_ptr];
    ld.param.u64 %rd4, [output_ptr];
    ld.param.u32 %r6, [block_size];

    // Calculate byte index and nibble position
    // byte_idx = global_idx / 2
    // is_high_nibble = global_idx & 1
    shr.u32 %r7, %r4, 1;           // byte_idx
    and.b32 %r8, %r4, 1;           // is_high_nibble

    // Calculate quantization block index
    // block_idx = global_idx / block_size
    div.u32 %r9, %r4, %r6;         // block_idx

    // Load packed byte
    cvt.u64.u32 %rd5, %r7;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %r10, [%rd6];     // packed byte

    // Extract INT4 value
    setp.eq.u32 %p2, %r8, 0;
    @%p2 bra LOW_NIBBLE;

    // High nibble: shift right by 4
    shr.u32 %r11, %r10, 4;
    bra NIBBLE_DONE;

LOW_NIBBLE:
    // Low nibble: mask with 0x0F
    and.b32 %r11, %r10, 15;

NIBBLE_DONE:
    // Load scale (F16, 2 bytes per scale)
    shl.b32 %r12, %r9, 1;          // block_idx * 2
    cvt.u64.u32 %rd7, %r12;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.u16 %h1, [%rd8];     // scale as F16 bits

    // Load zero point (i8)
    cvt.u64.u32 %rd9, %r9;
    add.u64 %rd10, %rd3, %rd9;
    ld.global.s8 %r13, [%rd10];    // zero_point

    // Dequantize: (int4_val - zero_point) * scale
    // Convert to F32 for computation
    sub.s32 %r14, %r11, %r13;      // int4 - zp
    cvt.rn.f32.s32 %f1, %r14;      // to float

    // Convert scale F16 to F32
    cvt.f32.f16 %f2, %h1;

    // Multiply
    mul.f32 %f3, %f1, %f2;

    // Convert back to F16
    cvt.rn.f16.f32 %h2, %f3;

    // Store output
    shl.b32 %r15, %r4, 1;          // output_idx * 2 (F16 = 2 bytes)
    cvt.u64.u32 %rd11, %r15;
    add.u64 %rd12, %rd4, %rd11;
    st.global.u16 [%rd12], %h2;

DONE:
    ret;
}

// Single block dequantization (for testing)
.visible .entry int4_dequant_block(
    .param .u64 packed_ptr,
    .param .u64 output_ptr,
    .param .u32 scale_bits,
    .param .s32 zero_point,
    .param .u32 num_values
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .f32 %f<4>;
    .reg .b16 %h<4>;
    .reg .pred %p<3>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    ld.param.u32 %r5, [num_values];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra BLOCK_DONE;

    ld.param.u64 %rd1, [packed_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r6, [scale_bits];
    ld.param.s32 %r7, [zero_point];

    // Extract INT4
    shr.u32 %r8, %r4, 1;
    and.b32 %r9, %r4, 1;

    cvt.u64.u32 %rd3, %r8;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u8 %r10, [%rd4];

    setp.eq.u32 %p2, %r9, 0;
    @!%p2 shr.u32 %r10, %r10, 4;
    @%p2 and.b32 %r10, %r10, 15;

    // Dequantize
    sub.s32 %r11, %r10, %r7;
    cvt.rn.f32.s32 %f1, %r11;

    cvt.u16.u32 %h1, %r6;
    cvt.f32.f16 %f2, %h1;
    mul.f32 %f3, %f1, %f2;
    cvt.rn.f16.f32 %h2, %f3;

    shl.b32 %r12, %r4, 1;
    cvt.u64.u32 %rd5, %r12;
    add.u64 %rd6, %rd2, %rd5;
    st.global.u16 [%rd6], %h2;

BLOCK_DONE:
    ret;
}
"#;

    /// INT8 dequantization kernel in PTX.
    const INT8_DEQUANT_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// INT8 dequantization kernel
// output[i] = input[i] * scale
.visible .entry int8_dequant_tensor(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 scale_bits,      // F16 scale passed as u16 bits
    .param .u32 num_values
)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<12>;
    .reg .f32 %f<4>;
    .reg .b16 %h<4>;
    .reg .pred %p<2>;

    // Global thread index
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    // Bounds check
    ld.param.u32 %r5, [num_values];
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra INT8_DONE;

    // Load parameters
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r6, [scale_bits];

    // Load INT8 value
    cvt.u64.u32 %rd3, %r4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.s8 %r7, [%rd4];

    // Convert to F32
    cvt.rn.f32.s32 %f1, %r7;

    // Convert scale bits to F16, then F32
    cvt.u16.u32 %h1, %r6;
    cvt.f32.f16 %f2, %h1;

    // Multiply
    mul.f32 %f3, %f1, %f2;

    // Convert to F16 and store
    cvt.rn.f16.f32 %h2, %f3;

    shl.b32 %r8, %r4, 1;
    cvt.u64.u32 %rd5, %r8;
    add.u64 %rd6, %rd2, %rd5;
    st.global.u16 [%rd6], %h2;

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
            GpuDequantContext::new(0).is_ok()
        }

        // ============== INT4 Tests ==============

        #[test]
        fn test_int4_dequant_context_creation() {
            match GpuDequantContext::new(0) {
                Ok(ctx) => {
                    assert_eq!(ctx.device_id(), 0);
                    assert!(!ctx.int4_kernel_loaded);
                    assert!(!ctx.int8_kernel_loaded);
                },
                Err(GpuDequantError::DeviceInit { .. }) => {
                    eprintln!("Skipping: no CUDA device");
                },
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[test]
        fn test_int4_kernel_loading() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().expect("INT4 kernel load");
            assert!(ctx.int4_kernel_loaded);

            // Second load should be no-op
            ctx.load_int4_kernel().expect("second load");
        }

        #[test]
        fn test_int4_dequant_basic() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // 8 INT4 values packed into 4 bytes
            // Values: 0,1,2,3,4,5,6,7 (low nibble first per byte)
            // Byte 0: 0x10 = (1 << 4) | 0 = values 0, 1
            // Byte 1: 0x32 = (3 << 4) | 2 = values 2, 3
            // Byte 2: 0x54 = (5 << 4) | 4 = values 4, 5
            // Byte 3: 0x76 = (7 << 4) | 6 = values 6, 7
            let packed: Vec<u8> = vec![0x10, 0x32, 0x54, 0x76];

            // Scale = 0.5, zero_point = 0
            // Expected: 0*0.5, 1*0.5, 2*0.5, ... = 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
            let scale = half::f16::from_f32(0.5);
            let scales = vec![scale]; // One block

            let result = ctx.dequant_int4(&packed, &scales, None, 8).unwrap();

            let mut host_result = vec![half::f16::ZERO; 8];
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
        fn test_int4_dequant_with_zero_point() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // Same packed data as above
            let packed: Vec<u8> = vec![0x10, 0x32, 0x54, 0x76];

            // Scale = 1.0, zero_point = 4
            // Expected: (0-4)*1, (1-4)*1, ... = -4, -3, -2, -1, 0, 1, 2, 3
            let scale = half::f16::from_f32(1.0);
            let scales = vec![scale];
            let zero_points = vec![4i8];

            let result = ctx
                .dequant_int4(&packed, &scales, Some(&zero_points), 8)
                .unwrap();

            let mut host_result = vec![half::f16::ZERO; 8];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

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
        fn test_int4_dequant_multiple_blocks() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // Create 256 INT4 values (128 bytes) = 2 blocks of 128
            let packed: Vec<u8> = (0..128).map(|i| ((i * 2 + 1) << 4) | (i * 2)).collect();

            // Different scales for each block
            let scales = vec![
                half::f16::from_f32(0.1), // Block 0
                half::f16::from_f32(0.2), // Block 1
            ];

            let result = ctx.dequant_int4(&packed, &scales, None, 256).unwrap();

            let mut host_result = vec![half::f16::ZERO; 256];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // Verify first value of each block
            // Block 0, value 0: 0 * 0.1 = 0.0
            assert!((host_result[0].to_f32() - 0.0).abs() < 0.01);

            // Block 1, value 0 (index 128): value = 0 (packed[64] low nibble)
            // Actually need to recalculate... packed[64] = ((64*2+1) << 4) | (64*2) = (129 << 4) | 128
            // But we only have 4 bits, so it wraps: 129 % 16 = 1, 128 % 16 = 0
            // So packed[64] = (1 << 4) | 0 = 0x10, values are 0 and 1
            // Value at index 128 = 0, scale = 0.2, result = 0.0
            assert!((host_result[128].to_f32() - 0.0).abs() < 0.01);
        }

        #[test]
        fn test_int4_dequant_to_tensor() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            let packed: Vec<u8> = vec![0x10, 0x32, 0x54, 0x76];
            let scales = vec![half::f16::from_f32(1.0)];

            let tensor = ctx
                .dequant_int4_to_tensor(&packed, &scales, None, &[2, 4])
                .unwrap();

            assert_eq!(tensor.dims(), &[2, 4]);

            let data: Vec<Vec<f32>> = tensor.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
            assert_eq!(data[0], vec![0.0, 1.0, 2.0, 3.0]);
            assert_eq!(data[1], vec![4.0, 5.0, 6.0, 7.0]);
        }

        #[test]
        fn test_int4_dequant_kernel_not_loaded_error() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let ctx = GpuDequantContext::new(0).unwrap();
            // Don't load kernel

            let packed = vec![0x10];
            let scales = vec![half::f16::from_f32(1.0)];

            let result = ctx.dequant_int4(&packed, &scales, None, 2);
            match result {
                Err(GpuDequantError::KernelNotLoaded { kernel }) => {
                    assert!(kernel.contains("int4"));
                },
                _ => panic!("Expected KernelNotLoaded error"),
            }
        }

        #[test]
        fn test_int4_dequant_invalid_input_packed_too_small() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            let packed = vec![0x10]; // Only 1 byte = 2 values
            let scales = vec![half::f16::from_f32(1.0)];

            // Request 8 values but only have 2
            let result = ctx.dequant_int4(&packed, &scales, None, 8);
            match result {
                Err(GpuDequantError::InvalidInput { message }) => {
                    assert!(message.contains("too small"));
                },
                _ => panic!("Expected InvalidInput error"),
            }
        }

        #[test]
        fn test_int4_dequant_invalid_input_not_enough_scales() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // 256 values = 2 blocks, but only 1 scale
            let packed = vec![0u8; 128];
            let scales = vec![half::f16::from_f32(1.0)]; // Only 1 scale

            let result = ctx.dequant_int4(&packed, &scales, None, 256);
            match result {
                Err(GpuDequantError::InvalidInput { message }) => {
                    assert!(message.contains("scales"));
                },
                _ => panic!("Expected InvalidInput error"),
            }
        }

        // ============== INT8 Tests ==============

        #[test]
        fn test_int8_kernel_loading() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int8_kernel().expect("INT8 kernel load");
            assert!(ctx.int8_kernel_loaded);
        }

        #[test]
        fn test_int8_dequant_basic() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int8_kernel().unwrap();

            let data: Vec<i8> = vec![-128, -64, 0, 32, 64, 96, 127, 100];
            let scale = half::f16::from_f32(0.01);

            let result = ctx.dequant_int8(&data, scale).unwrap();

            let mut host_result = vec![half::f16::ZERO; 8];
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
        fn test_int8_dequant_to_tensor() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int8_kernel().unwrap();

            let data: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
            let scale = half::f16::from_f32(0.1);

            let tensor = ctx.dequant_int8_to_tensor(&data, scale, &[2, 4]).unwrap();

            assert_eq!(tensor.dims(), &[2, 4]);

            let result: Vec<Vec<f32>> = tensor.to_dtype(DType::F32).unwrap().to_vec2().unwrap();

            // Expected: [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
            for (i, val) in result[0].iter().enumerate() {
                assert!((val - (i as f32 + 1.0)).abs() < 0.1);
            }
        }

        #[test]
        fn test_int8_dequant_large() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int8_kernel().unwrap();

            // 1M values
            let size = 1024 * 1024;
            let data: Vec<i8> = (0..size)
                .map(|i| (((i % 256) as u8).wrapping_sub(128)) as i8)
                .collect();
            let scale = half::f16::from_f32(0.001);

            let result = ctx.dequant_int8(&data, scale).unwrap();

            let mut host_result = vec![half::f16::ZERO; size];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // Spot check a few values
            assert!((host_result[0].to_f32() - (-0.128)).abs() < 0.01);
            assert!((host_result[128].to_f32() - 0.0).abs() < 0.01);
        }

        #[test]
        fn test_int8_dequant_kernel_not_loaded_error() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let ctx = GpuDequantContext::new(0).unwrap();

            let data = vec![0i8; 8];
            let scale = half::f16::from_f32(1.0);

            let result = ctx.dequant_int8(&data, scale);
            match result {
                Err(GpuDequantError::KernelNotLoaded { kernel }) => {
                    assert!(kernel.contains("int8"));
                },
                _ => panic!("Expected KernelNotLoaded error"),
            }
        }

        // ============== Edge Cases ==============

        #[test]
        fn test_int4_all_zeros() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            let packed = vec![0u8; 64]; // 128 zero values
            let scales = vec![half::f16::from_f32(1.0)];

            let result = ctx.dequant_int4(&packed, &scales, None, 128).unwrap();

            let mut host_result = vec![half::f16::ONE; 128];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            for val in host_result {
                assert_eq!(val.to_f32(), 0.0);
            }
        }

        #[test]
        fn test_int4_max_values() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // All 15s (0xFF = two 15s per byte)
            let packed = vec![0xFFu8; 4]; // 8 values of 15
            let scales = vec![half::f16::from_f32(0.1)];

            let result = ctx.dequant_int4(&packed, &scales, None, 8).unwrap();

            let mut host_result = vec![half::f16::ZERO; 8];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_result)
                .unwrap();

            // All should be 15 * 0.1 = 1.5
            for val in host_result {
                assert!((val.to_f32() - 1.5).abs() < 0.01);
            }
        }

        // ==================== TDD Phase 1: GPU vs CPU Cross-Validation ====================
        // GPU-CODEC-PIPELINE-TDD.md §4.1-4.2
        //
        // Trust boundary: GPU dequantization MUST produce values matching the CPU
        // quantize.rs reference within F16 rounding tolerance.

        /// §4.1: GPU INT4 dequant matches quantize.rs CPU reference.
        ///
        /// Quantize with quantize.rs, dequantize with both CPU (quantize.rs)
        /// and GPU (gpu_dequant), verify they agree.
        #[test]
        fn test_int4_gpu_matches_cpu_reference() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            use crate::quantize::{Quantizer, DEFAULT_BLOCK_SIZE};
            use candle_core::{Device, Tensor};

            // Create known input: 2 blocks (256 elements) with mixed values
            let input: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.05).collect();
            let tensor = Tensor::from_vec(input.clone(), &[256], &Device::Cpu).unwrap();

            // Quantize with quantize.rs
            let quantizer = Quantizer::int4_symmetric();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();
            assert_eq!(quantized.block_size, DEFAULT_BLOCK_SIZE);

            // CPU dequantize (ground truth)
            let cpu_reference = quantizer.dequantize(&quantized).unwrap();
            let cpu_values: Vec<f32> = cpu_reference.to_vec1().unwrap();

            // GPU dequantize
            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            // GPU kernel convention: (nibble - zero_point) * scale
            // quantize.rs packs symmetric INT4 as (q + 8) & 0x0F, so nibbles are [0,15]
            // and zero_points is None. To match CPU dequant (which subtracts 8 internally),
            // we must pass zero_point=8 to the GPU kernel.
            let num_blocks = (quantized.num_values + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
            let zp_for_gpu: Vec<i8> = match &quantized.zero_points {
                Some(zp) => zp.clone(),
                None => vec![8i8; num_blocks], // symmetric packing offset
            };

            let gpu_result = ctx
                .dequant_int4(
                    &quantized.data,
                    &quantized.scales,
                    Some(&zp_for_gpu),
                    quantized.num_values,
                )
                .unwrap();

            let mut gpu_f16 = vec![half::f16::ZERO; quantized.num_values];
            ctx.device
                .dtoh_sync_copy_into(&gpu_result, &mut gpu_f16)
                .unwrap();
            let gpu_values: Vec<f32> = gpu_f16.iter().map(|h| h.to_f32()).collect();

            // CPU dequant returns F32 computed from F16 scale, GPU outputs F16.
            // Both use the same F16 scale, so max error is F16 rounding (~0.01 for these ranges).
            assert_eq!(cpu_values.len(), gpu_values.len());
            for (i, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
                assert!(
                    (cpu - gpu).abs() < 0.01,
                    "INT4 GPU vs CPU mismatch at {}: cpu={}, gpu={} (diff={})",
                    i,
                    cpu,
                    gpu,
                    (cpu - gpu).abs()
                );
            }
        }

        /// §4.1 stress: GPU INT4 dequant matches CPU for 10 blocks (1280 values).
        #[test]
        fn test_int4_gpu_matches_cpu_multi_block() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            use crate::quantize::{Quantizer, DEFAULT_BLOCK_SIZE};
            use candle_core::{Device, Tensor};

            let n = DEFAULT_BLOCK_SIZE * 10;
            let input: Vec<f32> = (0..n)
                .map(|i| {
                    let block = i / DEFAULT_BLOCK_SIZE;
                    let pos = i % DEFAULT_BLOCK_SIZE;
                    ((pos as f32) - 64.0) * 0.001 * ((block + 1) as f32)
                })
                .collect();

            let tensor = Tensor::from_vec(input.clone(), &[n], &Device::Cpu).unwrap();
            let quantizer = Quantizer::int4_symmetric();
            let quantized = quantizer.quantize_tensor(&tensor).unwrap();

            // CPU reference
            let cpu_ref = quantizer.dequantize(&quantized).unwrap();
            let cpu_values: Vec<f32> = cpu_ref.to_vec1().unwrap();

            // GPU dequantize — same zero_point=8 convention as single-block test
            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int4_kernel().unwrap();

            let num_blocks = (quantized.num_values + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
            let zp_for_gpu: Vec<i8> = match &quantized.zero_points {
                Some(zp) => zp.clone(),
                None => vec![8i8; num_blocks],
            };

            let gpu_result = ctx
                .dequant_int4(
                    &quantized.data,
                    &quantized.scales,
                    Some(&zp_for_gpu),
                    quantized.num_values,
                )
                .unwrap();
            let mut gpu_f16 = vec![half::f16::ZERO; n];
            ctx.device
                .dtoh_sync_copy_into(&gpu_result, &mut gpu_f16)
                .unwrap();
            let gpu_values: Vec<f32> = gpu_f16.iter().map(|h| h.to_f32()).collect();

            let max_diff: f32 = cpu_values
                .iter()
                .zip(gpu_values.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 0.01,
                "Max INT4 GPU vs CPU diff across {} values: {} (should be < 0.01)",
                n,
                max_diff
            );
        }

        /// §4.2: GPU INT8 dequant matches CPU reference.
        #[test]
        fn test_int8_gpu_matches_cpu_reference() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            // Create INT8 data (as i8) and known scale
            let data: Vec<i8> = (0..256).map(|i| (i as u8) as i8).collect();
            let scale = half::f16::from_f32(0.05);

            // CPU reference: value * scale
            let cpu_values: Vec<f32> = data.iter().map(|&v| (v as f32) * scale.to_f32()).collect();

            // GPU dequantize: takes &[i8] and a single f16 scale
            let mut ctx = GpuDequantContext::new(0).unwrap();
            ctx.load_int8_kernel().unwrap();
            let gpu_result = ctx.dequant_int8(&data, scale).unwrap();
            let mut gpu_f16 = vec![half::f16::ZERO; data.len()];
            ctx.device
                .dtoh_sync_copy_into(&gpu_result, &mut gpu_f16)
                .unwrap();
            let gpu_values: Vec<f32> = gpu_f16.iter().map(|h| h.to_f32()).collect();

            // GPU outputs F16 then we read back as F32. CPU computes in F32.
            // F16 has ~0.1% relative error, so for values around 4.0 the abs
            // error can be ~0.004. Use 0.01 tolerance (same as INT4 tests).
            for (i, (cpu, gpu)) in cpu_values.iter().zip(gpu_values.iter()).enumerate() {
                assert!(
                    (cpu - gpu).abs() < 0.01,
                    "INT8 GPU vs CPU mismatch at {}: cpu={}, gpu={} (diff={})",
                    i,
                    cpu,
                    gpu,
                    (cpu - gpu).abs()
                );
            }
        }

        // ==================== Phase 4: §4.3 INT4 Dequant Property Test ====================
        // GPU-CODEC-PIPELINE-TDD.md §4.3: Property — GPU INT4 dequant produces
        // bit-exact F16 values matching the formula (nibble - zp) * scale.

        mod dequant_proptest {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(20))]
                #[test]
                fn int4_gpu_matches_cpu_arbitrary(
                    packed in proptest::collection::vec(any::<u8>(), 64..=64),
                    scale_f32 in 0.01f32..5.0f32,
                    zero_point in 0i8..=15i8,
                ) {
                    if !cuda_available() {
                        return Ok(());
                    }

                    let num_values = 128; // 64 bytes = 128 nibbles = 1 block
                    let scale = half::f16::from_f32(scale_f32);
                    let scales = vec![scale];
                    let zps = [zero_point];

                    let mut ctx = GpuDequantContext::new(0).unwrap();
                    ctx.load_int4_kernel().unwrap();
                    let gpu_result = ctx
                        .dequant_int4(&packed, &scales, Some(&zps[..]), num_values)
                        .unwrap();
                    let mut gpu_f16 = vec![half::f16::ZERO; num_values];
                    ctx.device
                        .dtoh_sync_copy_into(&gpu_result, &mut gpu_f16)
                        .unwrap();

                    // CPU reference: (nibble - zero_point) * scale → round to F16
                    for i in 0..num_values {
                        let nibble = if i % 2 == 0 {
                            packed[i / 2] & 0x0F
                        } else {
                            (packed[i / 2] >> 4) & 0x0F
                        };
                        let expected = half::f16::from_f32(
                            (nibble as i32 - zero_point as i32) as f32 * scale.to_f32(),
                        );
                        prop_assert_eq!(
                            gpu_f16[i].to_bits(),
                            expected.to_bits(),
                            "INT4 dequant 0-ULP violation at {}: nibble={}, zp={}, scale={:.4}, \
                             gpu={:?}, expected={:?}",
                            i, nibble, zero_point, scale.to_f32(), gpu_f16[i], expected
                        );
                    }
                }
            }
        }
    }
}

/// Stub module when CUDA is not available.
#[cfg(not(feature = "cuda"))]
pub mod cuda {

    /// GPU dequantization context (stub).
    pub struct GpuDequantContext;

    /// Errors from GPU dequantization (stub).
    #[derive(Debug, thiserror::Error)]
    pub enum GpuDequantError {
        /// CUDA feature not enabled.
        #[error("CUDA not enabled - compile with --features cuda")]
        CudaNotEnabled,
    }

    impl GpuDequantContext {
        /// Always returns error when CUDA is not enabled.
        pub fn new(_device_id: usize) -> Result<Self, GpuDequantError> {
            Err(GpuDequantError::CudaNotEnabled)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_stub_returns_error() {
            match GpuDequantContext::new(0) {
                Err(GpuDequantError::CudaNotEnabled) => {},
                Ok(_) => panic!("Stub should error"),
            }
        }
    }
}

pub use cuda::GpuDequantContext;
#[cfg(feature = "cuda")]
pub use cuda::GpuDequantError;

/// INT4 block size for HCT-native quantization (values per scale factor).
///
/// This is a format constant, not a CUDA-specific value. It must agree with
/// `DEFAULT_BLOCK_SIZE` in `quantize.rs`. Available regardless of CUDA feature.
pub const INT4_BLOCK_SIZE: usize = 128;

// Compile-time assertion: INT4_BLOCK_SIZE must equal quantize::DEFAULT_BLOCK_SIZE.
// If this fails, the HCT reader and quantizer disagree on block layout (see DD-1).
const _: () = assert!(
    INT4_BLOCK_SIZE == crate::quantize::DEFAULT_BLOCK_SIZE,
    "INT4_BLOCK_SIZE != DEFAULT_BLOCK_SIZE: HCT and quantizer block sizes must agree"
);
