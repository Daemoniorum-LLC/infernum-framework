//! GPU-accelerated dtype conversion kernels.
//!
//! Converts FP8 tensors to FP32 on GPU, eliminating:
//! - 4x memory expansion on CPU
//! - Large host→device transfers (1 byte FP8 vs 4 bytes F32)

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::Arc;
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;

    /// GPU dtype converter for FP8 → F32 conversion.
    pub struct GpuDtypeConverter {
        device: Arc<CudaDevice>,
    }

    // CUDA kernel for FP8 E4M3 → F32 conversion
    const FP8_E4M3_KERNEL: &str = r#"
extern "C" __global__ void fp8_e4m3_to_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char byte = input[idx];
    int sign = (byte >> 7) & 1;
    int exponent = (byte >> 3) & 0xF;
    int mantissa = byte & 0x7;

    float value;
    if (exponent == 0) {
        // Subnormal or zero
        if (mantissa == 0) {
            value = sign ? -0.0f : 0.0f;
        } else {
            // Subnormal: (-1)^s * 2^(-6) * (m/8)
            value = (float(mantissa) / 8.0f) * 0.015625f; // 2^-6
            if (sign) value = -value;
        }
    } else if (exponent == 15 && mantissa == 7) {
        // NaN
        value = __int_as_float(0x7FC00000); // quiet NaN
    } else {
        // Normal: (-1)^s * 2^(e-7) * (1 + m/8)
        value = (1.0f + float(mantissa) / 8.0f) * exp2f(float(exponent) - 7.0f);
        if (sign) value = -value;
    }

    output[idx] = value;
}
"#;

    // CUDA kernel for FP8 E5M2 → F32 conversion
    const FP8_E5M2_KERNEL: &str = r#"
extern "C" __global__ void fp8_e5m2_to_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char byte = input[idx];
    int sign = (byte >> 7) & 1;
    int exponent = (byte >> 2) & 0x1F;
    int mantissa = byte & 0x3;

    float value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign ? -0.0f : 0.0f;
        } else {
            // Subnormal
            value = (float(mantissa) / 4.0f) * 6.103515625e-5f; // 2^-14
            if (sign) value = -value;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            value = sign ? __int_as_float(0xFF800000) : __int_as_float(0x7F800000);
        } else {
            value = __int_as_float(0x7FC00000); // NaN
        }
    } else {
        value = (1.0f + float(mantissa) / 4.0f) * exp2f(float(exponent) - 15.0f);
        if (sign) value = -value;
    }

    output[idx] = value;
}
"#;

    impl GpuDtypeConverter {
        /// Create new GPU dtype converter.
        pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
            // Compile both kernels
            let ptx_e4m3 = cudarc::nvrtc::compile_ptx(FP8_E4M3_KERNEL)?;
            let ptx_e5m2 = cudarc::nvrtc::compile_ptx(FP8_E5M2_KERNEL)?;

            device.load_ptx(ptx_e4m3, "fp8_e4m3", &["fp8_e4m3_to_f32"])?;
            device.load_ptx(ptx_e5m2, "fp8_e5m2", &["fp8_e5m2_to_f32"])?;

            Ok(Self { device })
        }

        /// Get device reference.
        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }

        /// Convert FP8 E4M3 data to F32 on GPU.
        ///
        /// Takes raw FP8 bytes, transfers to GPU, converts, returns F32 slice.
        pub fn fp8_e4m3_to_f32(&self, fp8_data: &[u8]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let n = fp8_data.len();

            // Transfer FP8 bytes to GPU (1 byte per element)
            let d_fp8: CudaSlice<u8> = self.device.htod_sync_copy(fp8_data)?;

            // Allocate output F32 buffer on GPU
            let mut d_f32: CudaSlice<f32> = self.device.alloc_zeros(n)?;

            // Launch kernel
            let kernel = self.device.get_func("fp8_e4m3", "fp8_e4m3_to_f32")
                .ok_or("FP8 E4M3 kernel not loaded")?;

            let threads_per_block = 256;
            let blocks = (n + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel.launch(config, (&d_fp8, &mut d_f32, n as i32))?;
            }

            Ok(d_f32)
        }

        /// Convert FP8 E5M2 data to F32 on GPU.
        pub fn fp8_e5m2_to_f32(&self, fp8_data: &[u8]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let n = fp8_data.len();

            let d_fp8: CudaSlice<u8> = self.device.htod_sync_copy(fp8_data)?;
            let mut d_f32: CudaSlice<f32> = self.device.alloc_zeros(n)?;

            let kernel = self.device.get_func("fp8_e5m2", "fp8_e5m2_to_f32")
                .ok_or("FP8 E5M2 kernel not loaded")?;

            let threads_per_block = 256;
            let blocks = (n + threads_per_block - 1) / threads_per_block;

            let config = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel.launch(config, (&d_fp8, &mut d_f32, n as i32))?;
            }

            Ok(d_f32)
        }

        /// Convert FP8 E4M3 and return host F32 vector.
        pub fn fp8_e4m3_to_f32_host(&self, fp8_data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let d_f32 = self.fp8_e4m3_to_f32(fp8_data)?;
            let mut h_f32 = vec![0.0f32; fp8_data.len()];
            self.device.dtoh_sync_copy_into(&d_f32, &mut h_f32)?;
            Ok(h_f32)
        }

        /// Convert FP8 E5M2 and return host F32 vector.
        pub fn fp8_e5m2_to_f32_host(&self, fp8_data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let d_f32 = self.fp8_e5m2_to_f32(fp8_data)?;
            let mut h_f32 = vec![0.0f32; fp8_data.len()];
            self.device.dtoh_sync_copy_into(&d_f32, &mut h_f32)?;
            Ok(h_f32)
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub mod cuda {
    pub struct GpuDtypeConverter;

    impl GpuDtypeConverter {
        pub fn new(_device: std::sync::Arc<()>) -> Result<Self, Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }

        pub fn fp8_e4m3_to_f32_host(&self, _fp8_data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }

        pub fn fp8_e5m2_to_f32_host(&self, _fp8_data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cuda::GpuDtypeConverter;

    #[test]
    fn test_gpu_dtype_stub_without_cuda() {
        #[cfg(not(feature = "cuda"))]
        {
            let result = GpuDtypeConverter::new(std::sync::Arc::new(()));
            assert!(result.is_err());
        }
    }
}
