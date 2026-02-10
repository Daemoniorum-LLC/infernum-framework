//! GPU-accelerated dtype conversion kernels.
//!
//! Converts FP8 tensors to FP32 on GPU, eliminating:
//! - 4x memory expansion on CPU
//! - Large host→device transfers (1 byte FP8 vs 4 bytes F32)

/// CUDA-accelerated FP8 to F32 dtype conversion.
#[cfg(feature = "cuda")]
pub mod cuda {
    use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
    use std::sync::Arc;

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
        pub fn fp8_e4m3_to_f32(
            &self,
            fp8_data: &[u8],
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let n = fp8_data.len();

            // Transfer FP8 bytes to GPU (1 byte per element)
            let d_fp8: CudaSlice<u8> = self.device.htod_sync_copy(fp8_data)?;

            // Allocate output F32 buffer on GPU
            let mut d_f32: CudaSlice<f32> = self.device.alloc_zeros(n)?;

            // Launch kernel
            let kernel = self
                .device
                .get_func("fp8_e4m3", "fp8_e4m3_to_f32")
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
        pub fn fp8_e5m2_to_f32(
            &self,
            fp8_data: &[u8],
        ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let n = fp8_data.len();

            let d_fp8: CudaSlice<u8> = self.device.htod_sync_copy(fp8_data)?;
            let mut d_f32: CudaSlice<f32> = self.device.alloc_zeros(n)?;

            let kernel = self
                .device
                .get_func("fp8_e5m2", "fp8_e5m2_to_f32")
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
        pub fn fp8_e4m3_to_f32_host(
            &self,
            fp8_data: &[u8],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let d_f32 = self.fp8_e4m3_to_f32(fp8_data)?;
            let mut h_f32 = vec![0.0f32; fp8_data.len()];
            self.device.dtoh_sync_copy_into(&d_f32, &mut h_f32)?;
            Ok(h_f32)
        }

        /// Convert FP8 E5M2 and return host F32 vector.
        pub fn fp8_e5m2_to_f32_host(
            &self,
            fp8_data: &[u8],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let d_f32 = self.fp8_e5m2_to_f32(fp8_data)?;
            let mut h_f32 = vec![0.0f32; fp8_data.len()];
            self.device.dtoh_sync_copy_into(&d_f32, &mut h_f32)?;
            Ok(h_f32)
        }
    }
}

/// Stub module when CUDA is not enabled.
#[cfg(not(feature = "cuda"))]
pub mod cuda {
    /// GPU dtype converter stub (requires CUDA feature).
    pub struct GpuDtypeConverter;

    impl GpuDtypeConverter {
        /// Create new converter (returns error without CUDA).
        pub fn new(_device: std::sync::Arc<()>) -> Result<Self, Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }

        /// Convert FP8 E4M3 to F32 on host (requires CUDA).
        pub fn fp8_e4m3_to_f32_host(
            &self,
            _fp8_data: &[u8],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }

        /// Convert FP8 E5M2 to F32 on host (requires CUDA).
        pub fn fp8_e5m2_to_f32_host(
            &self,
            _fp8_data: &[u8],
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

    // ==================== Phase 4: FP8 CPU Reference Tests ====================
    // Trust boundary §4 (Quantization Math) applied to FP8 formats.
    //
    // These CPU reference functions match the CUDA kernel math exactly,
    // enabling cross-validation when a GPU is available.

    /// CPU reference: FP8 E4M3 → F32.
    /// Format: sign(1) + exponent(4) + mantissa(3)
    /// Normal:    (-1)^s * 2^(e-7) * (1 + m/8)
    /// Subnormal: (-1)^s * 2^(-6) * (m/8)
    /// NaN:       e=15 && m=7
    fn fp8_e4m3_to_f32_cpu(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exponent = (byte >> 3) & 0xF;
        let mantissa = byte & 0x7;

        let value = if exponent == 0 {
            if mantissa == 0 {
                0.0
            } else {
                // Subnormal: 2^(-6) * (m/8)
                (mantissa as f32 / 8.0) * 2.0f32.powi(-6)
            }
        } else if exponent == 15 && mantissa == 7 {
            f32::NAN
        } else {
            // Normal: 2^(e-7) * (1 + m/8)
            (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
        };

        // Apply sign uniformly (zero branch returns unsigned 0.0, negated here to -0.0)
        if sign == 1 && !value.is_nan() {
            -value
        } else {
            value
        }
    }

    /// CPU reference: FP8 E5M2 → F32.
    /// Format: sign(1) + exponent(5) + mantissa(2)
    /// Normal:    (-1)^s * 2^(e-15) * (1 + m/4)
    /// Subnormal: (-1)^s * 2^(-14) * (m/4)
    /// Inf:       e=31, m=0
    /// NaN:       e=31, m!=0
    fn fp8_e5m2_to_f32_cpu(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exponent = (byte >> 2) & 0x1F;
        let mantissa = byte & 0x3;

        let value = if exponent == 0 {
            if mantissa == 0 {
                0.0
            } else {
                // Subnormal: 2^(-14) * (m/4)
                (mantissa as f32 / 4.0) * 2.0f32.powi(-14)
            }
        } else if exponent == 31 {
            if mantissa == 0 {
                // Infinity (sign handled by final sign application for zero;
                // infinity needs explicit sign since we skip it below)
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            // Normal: 2^(e-15) * (1 + m/4)
            (1.0 + mantissa as f32 / 4.0) * 2.0f32.powi(exponent as i32 - 15)
        };

        // Apply sign uniformly (zero branch returns unsigned 0.0, negated here to -0.0)
        if sign == 1 && !value.is_nan() && !value.is_infinite() {
            -value
        } else {
            value
        }
    }

    /// E4M3: Zero (0x00 = +0, 0x80 = -0).
    #[test]
    fn test_fp8_e4m3_zero() {
        let pos_zero = fp8_e4m3_to_f32_cpu(0x00);
        let neg_zero = fp8_e4m3_to_f32_cpu(0x80);
        assert_eq!(pos_zero, 0.0);
        assert_eq!(neg_zero, -0.0);
        assert!(pos_zero.is_sign_positive());
        assert!(neg_zero.is_sign_negative());
    }

    /// E4M3: One = 0b0_0111_000 = 0x38 → 2^(7-7) * (1+0) = 1.0.
    #[test]
    fn test_fp8_e4m3_one() {
        assert_eq!(fp8_e4m3_to_f32_cpu(0x38), 1.0);
    }

    /// E4M3: Negative one = 0b1_0111_000 = 0xB8 → -1.0.
    #[test]
    fn test_fp8_e4m3_neg_one() {
        assert_eq!(fp8_e4m3_to_f32_cpu(0xB8), -1.0);
    }

    /// E4M3: Max normal = 0b0_1110_111 = 0x77 → 2^7 * (1+7/8) = 240.0.
    #[test]
    fn test_fp8_e4m3_max_normal() {
        let val = fp8_e4m3_to_f32_cpu(0x77);
        assert!(
            (val - 240.0).abs() < 0.01,
            "Max normal should be 240, got {}",
            val
        );
    }

    /// E4M3: Smallest subnormal = 0b0_0000_001 = 0x01 → 2^(-6) * (1/8).
    #[test]
    fn test_fp8_e4m3_smallest_subnormal() {
        let val = fp8_e4m3_to_f32_cpu(0x01);
        let expected = 2.0f32.powi(-6) / 8.0; // 1/512 ≈ 0.001953
        assert!(
            (val - expected).abs() < 1e-7,
            "Smallest subnormal: expected {}, got {}",
            expected,
            val
        );
    }

    /// E4M3: NaN = 0b0_1111_111 = 0x7F.
    #[test]
    fn test_fp8_e4m3_nan() {
        assert!(fp8_e4m3_to_f32_cpu(0x7F).is_nan());
    }

    /// E5M2: Zero (0x00 = +0, 0x80 = -0).
    #[test]
    fn test_fp8_e5m2_zero() {
        assert_eq!(fp8_e5m2_to_f32_cpu(0x00), 0.0);
        assert!(fp8_e5m2_to_f32_cpu(0x80).is_sign_negative());
    }

    /// E5M2: One = 0b0_01111_00 = 0x3C → 2^(15-15) * (1+0) = 1.0.
    #[test]
    fn test_fp8_e5m2_one() {
        assert_eq!(fp8_e5m2_to_f32_cpu(0x3C), 1.0);
    }

    /// E5M2: Infinity = 0b0_11111_00 = 0x7C.
    #[test]
    fn test_fp8_e5m2_infinity() {
        assert_eq!(fp8_e5m2_to_f32_cpu(0x7C), f32::INFINITY);
        assert_eq!(fp8_e5m2_to_f32_cpu(0xFC), f32::NEG_INFINITY);
    }

    /// E5M2: NaN = 0b0_11111_01 = 0x7D.
    #[test]
    fn test_fp8_e5m2_nan() {
        assert!(fp8_e5m2_to_f32_cpu(0x7D).is_nan());
        assert!(fp8_e5m2_to_f32_cpu(0x7E).is_nan());
        assert!(fp8_e5m2_to_f32_cpu(0x7F).is_nan());
    }

    /// Exhaustive E4M3: verify all 256 byte values produce sane f32 results.
    #[test]
    fn test_fp8_e4m3_exhaustive_sanity() {
        for byte in 0u8..=255 {
            let val = fp8_e4m3_to_f32_cpu(byte);
            if byte == 0x7F || byte == 0xFF {
                assert!(val.is_nan(), "Byte 0x{:02X} should be NaN", byte);
            } else {
                assert!(
                    val.is_finite(),
                    "Byte 0x{:02X} should be finite, got {}",
                    byte,
                    val
                );
            }
        }
    }

    /// Exhaustive E5M2: verify all 256 byte values produce sane f32 results.
    #[test]
    fn test_fp8_e5m2_exhaustive_sanity() {
        for byte in 0u8..=255 {
            let val = fp8_e5m2_to_f32_cpu(byte);
            let exp = (byte >> 2) & 0x1F;
            let man = byte & 0x3;
            if exp == 31 && man != 0 {
                assert!(val.is_nan(), "Byte 0x{:02X} should be NaN", byte);
            } else if exp == 31 && man == 0 {
                assert!(val.is_infinite(), "Byte 0x{:02X} should be Inf", byte);
            } else {
                assert!(
                    val.is_finite(),
                    "Byte 0x{:02X} should be finite, got {}",
                    byte,
                    val
                );
            }
        }
    }

    /// CUDA cross-validation: GPU fp8_e4m3_to_f32 matches CPU reference.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_fp8_e4m3_gpu_matches_cpu() {
        use cudarc::driver::CudaDevice;
        use std::sync::Arc;

        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(_) => {
                eprintln!("Skipping: no CUDA device available");
                return;
            },
        };

        let converter = GpuDtypeConverter::new(Arc::clone(&device)).expect("converter creation");

        // Test all 256 byte values
        let input: Vec<u8> = (0..=255).collect();
        let gpu_result = converter
            .fp8_e4m3_to_f32_host(&input)
            .expect("GPU conversion");

        for (byte, &gpu_val) in input.iter().zip(gpu_result.iter()) {
            let cpu_val = fp8_e4m3_to_f32_cpu(*byte);
            if cpu_val.is_nan() {
                assert!(
                    gpu_val.is_nan(),
                    "Byte 0x{:02X}: CPU=NaN, GPU={}",
                    byte,
                    gpu_val
                );
            } else {
                assert_eq!(
                    gpu_val, cpu_val,
                    "Byte 0x{:02X}: CPU={}, GPU={}",
                    byte, cpu_val, gpu_val
                );
            }
        }
    }

    /// CUDA cross-validation: GPU fp8_e5m2_to_f32 matches CPU reference.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_fp8_e5m2_gpu_matches_cpu() {
        use cudarc::driver::CudaDevice;
        use std::sync::Arc;

        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(_) => {
                eprintln!("Skipping: no CUDA device available");
                return;
            },
        };

        let converter = GpuDtypeConverter::new(Arc::clone(&device)).expect("converter creation");

        let input: Vec<u8> = (0..=255).collect();
        let gpu_result = converter
            .fp8_e5m2_to_f32_host(&input)
            .expect("GPU conversion");

        for (byte, &gpu_val) in input.iter().zip(gpu_result.iter()) {
            let cpu_val = fp8_e5m2_to_f32_cpu(*byte);
            if cpu_val.is_nan() {
                assert!(
                    gpu_val.is_nan(),
                    "Byte 0x{:02X}: CPU=NaN, GPU={}",
                    byte,
                    gpu_val
                );
            } else if cpu_val.is_infinite() {
                assert_eq!(
                    gpu_val, cpu_val,
                    "Byte 0x{:02X}: CPU=Inf, GPU={}",
                    byte, gpu_val
                );
            } else {
                assert!(
                    (gpu_val - cpu_val).abs() < 1e-10,
                    "Byte 0x{:02X}: CPU={}, GPU={}",
                    byte,
                    cpu_val,
                    gpu_val
                );
            }
        }
    }
}
