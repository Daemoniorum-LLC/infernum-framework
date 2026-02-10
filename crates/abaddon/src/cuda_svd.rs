//! GPU-accelerated SVD for HoloTensor LRDF encoding.
//!
//! Uses cuBLAS for matrix operations with hardware-optimized paths:
//! - TF32 mode for RTX 30/40 series (8x faster than FP32)
//! - GPU-resident deflation with `ger` (eliminates CPU round-trips)
//! - GPU normalization with `nrm2` + `scal`

/// CUDA-accelerated SVD using cuBLAS power iteration.
#[cfg(feature = "cuda")]
pub mod cuda {
    use cudarc::cublas::sys::lib as cublas_lib;
    use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
    use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
    use std::sync::Arc;

    /// GPU-accelerated SVD using power iteration with cuBLAS.
    ///
    /// Optimizations:
    /// - All deflation done on GPU (no CPU round-trips)
    /// - Vector normalization on GPU
    /// - TF32 enabled for Ada/Ampere GPUs
    pub struct GpuSvd {
        device: Arc<CudaDevice>,
        blas: CudaBlas,
    }

    impl GpuSvd {
        /// Create new GPU SVD context.
        pub fn new(device: Arc<CudaDevice>) -> Result<Self, cudarc::cublas::result::CublasError> {
            let blas = CudaBlas::new(device.clone())?;

            // Enable TF32 for Ampere+ GPUs (compute 8.0+)
            // TF32 uses 19-bit mantissa, 8x faster than FP32, ~0.1% accuracy loss
            // Safe for SVD where we don't need full FP32 precision
            unsafe {
                let handle = *blas.handle();
                // Try to set TF32 mode - ignore errors on older GPUs
                let _ = cublas_lib().cublasSetMathMode(
                    handle,
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                );
            }

            Ok(Self { device, blas })
        }

        /// Get device reference.
        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }

        /// Compute truncated SVD using GPU-accelerated power iteration.
        ///
        /// **Fully GPU-resident**: Deflation and normalization done on GPU.
        ///
        /// Returns (U, S, V) where:
        /// - U: rows x rank (row-major)
        /// - S: rank singular values
        /// - V: cols x rank (row-major)
        pub fn svd_power_iteration(
            &self,
            matrix: &[f32],
            rows: usize,
            cols: usize,
            rank: usize,
            iterations: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
            let d_matrix: CudaSlice<f32> = self.device.htod_sync_copy(matrix)?;
            self.svd_power_iteration_gpu(d_matrix, rows, cols, rank, iterations)
        }

        /// Compute SVD from GPU-resident matrix (avoids host-to-device copy).
        ///
        /// Use this when data is already on GPU (e.g., after FP8→F32 conversion).
        pub fn svd_power_iteration_gpu(
            &self,
            mut d_matrix: CudaSlice<f32>,
            rows: usize,
            cols: usize,
            rank: usize,
            iterations: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
            let mut d_u: CudaSlice<f32> = self.device.alloc_zeros(rows)?;
            let mut d_v: CudaSlice<f32> = self.device.alloc_zeros(cols)?;

            let mut u_out = vec![0.0f32; rows * rank];
            let mut s_out = vec![0.0f32; rank];
            let mut v_out = vec![0.0f32; cols * rank];

            for r in 0..rank {
                // Initialize v randomly
                let h_v: Vec<f32> = (0..cols)
                    .map(|i| {
                        let x = (((i + r) as u64 * 1103515245 + 12345) % 0x80000000) as f32;
                        (x / 0x80000000u32 as f32) - 0.5
                    })
                    .collect();
                self.device.htod_sync_copy_into(&h_v, &mut d_v)?;
                self.gpu_normalize(&mut d_v, cols)?;

                // Power iteration
                for _ in 0..iterations {
                    self.gemv_row_major(&d_matrix, &d_v, &mut d_u, rows, cols, false)?;
                    self.gpu_normalize(&mut d_u, rows)?;
                    self.gemv_row_major(&d_matrix, &d_u, &mut d_v, rows, cols, true)?;
                    self.gpu_normalize(&mut d_v, cols)?;
                }

                // Final u = A*v, sigma = ||u||, u = u/sigma
                self.gemv_row_major(&d_matrix, &d_v, &mut d_u, rows, cols, false)?;
                let sigma = self.gpu_norm(&d_u, rows)?;
                s_out[r] = sigma;

                if sigma > 1e-10 {
                    self.gpu_scale(&mut d_u, rows, 1.0 / sigma)?;
                }

                // Store u and v
                let mut h_u = vec![0.0f32; rows];
                let mut h_v = vec![0.0f32; cols];
                self.device.dtoh_sync_copy_into(&d_u, &mut h_u)?;
                self.device.dtoh_sync_copy_into(&d_v, &mut h_v)?;

                for i in 0..rows {
                    u_out[i * rank + r] = h_u[i];
                }
                for j in 0..cols {
                    v_out[j * rank + r] = h_v[j];
                }

                // GPU Deflation: A = A - sigma * u * v^T
                if r < rank - 1 {
                    self.gpu_ger(&mut d_matrix, &d_u, &d_v, rows, cols, -sigma)?;
                }
            }

            Ok((u_out, s_out, v_out))
        }

        /// GPU matrix-vector multiply for row-major matrix.
        fn gemv_row_major(
            &self,
            d_matrix: &CudaSlice<f32>,
            d_x: &CudaSlice<f32>,
            d_y: &mut CudaSlice<f32>,
            rows: usize,
            cols: usize,
            transpose: bool,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let (trans, m, n) = if transpose {
                (
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    cols,
                    rows,
                )
            } else {
                (
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                    cols,
                    rows,
                )
            };

            unsafe {
                self.blas.gemv(
                    GemvConfig {
                        trans,
                        m: m as i32,
                        n: n as i32,
                        alpha: 1.0f32,
                        lda: cols as i32,
                        incx: 1,
                        beta: 0.0f32,
                        incy: 1,
                    },
                    d_matrix,
                    d_x,
                    d_y,
                )?;
            }
            Ok(())
        }

        /// GPU rank-1 update: A = alpha * x * y^T + A
        fn gpu_ger(
            &self,
            d_a: &mut CudaSlice<f32>,
            d_u: &CudaSlice<f32>,
            d_v: &CudaSlice<f32>,
            rows: usize,
            cols: usize,
            alpha: f32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                let handle = *self.blas.handle();
                // For row-major A(rows,cols) stored as col-major A^T(cols,rows):
                // To do A[i,j] += alpha * u[i] * v[j] in row-major
                // = A^T[j,i] += alpha * u[i] * v[j] in col-major view
                // Use ger(m=cols, n=rows, x=v, y=u) on the col-major view
                let status = cublas_lib().cublasSger_v2(
                    handle,
                    cols as i32, // m
                    rows as i32, // n
                    &alpha as *const f32,
                    *d_v.device_ptr() as *const f32, // x = v
                    1,
                    *d_u.device_ptr() as *const f32, // y = u
                    1,
                    *d_a.device_ptr_mut() as *mut f32,
                    cols as i32, // lda
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasSger failed: {:?}", status).into());
                }
            }
            Ok(())
        }

        /// GPU vector normalization: x = x / ||x||
        fn gpu_normalize(
            &self,
            d_x: &mut CudaSlice<f32>,
            n: usize,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let norm = self.gpu_norm(d_x, n)?;
            if norm > 1e-10 {
                self.gpu_scale(d_x, n, 1.0 / norm)?;
            }
            Ok(())
        }

        /// GPU vector norm: ||x||
        fn gpu_norm(
            &self,
            d_x: &CudaSlice<f32>,
            n: usize,
        ) -> Result<f32, Box<dyn std::error::Error>> {
            unsafe {
                let handle = *self.blas.handle();
                let mut result: f32 = 0.0;
                let status = cublas_lib().cublasSnrm2_v2(
                    handle,
                    n as i32,
                    *d_x.device_ptr() as *const f32,
                    1,
                    &mut result as *mut f32,
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasSnrm2 failed: {:?}", status).into());
                }
                Ok(result)
            }
        }

        /// GPU vector scale: x = alpha * x
        fn gpu_scale(
            &self,
            d_x: &mut CudaSlice<f32>,
            n: usize,
            alpha: f32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                let handle = *self.blas.handle();
                let status = cublas_lib().cublasSscal_v2(
                    handle,
                    n as i32,
                    &alpha as *const f32,
                    *d_x.device_ptr_mut() as *mut f32,
                    1,
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(format!("cublasSscal failed: {:?}", status).into());
                }
            }
            Ok(())
        }
    }
}

/// Stub module when CUDA is not enabled.
#[cfg(not(feature = "cuda"))]
pub mod cuda {
    /// GPU SVD stub (requires CUDA feature).
    pub struct GpuSvd;

    impl GpuSvd {
        /// Create new GPU SVD context (returns error without CUDA).
        pub fn new(_device: std::sync::Arc<()>) -> Result<Self, std::io::Error> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "CUDA not enabled",
            ))
        }

        /// Compute truncated SVD (returns error without CUDA).
        pub fn svd_power_iteration(
            &self,
            _matrix: &[f32],
            _rows: usize,
            _cols: usize,
            _rank: usize,
            _iterations: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
            Err("CUDA not enabled".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cuda::GpuSvd;

    #[test]
    fn test_gpu_svd_stub_without_cuda() {
        // Without CUDA feature, GpuSvd::new should return an error
        #[cfg(not(feature = "cuda"))]
        {
            let result = GpuSvd::new(std::sync::Arc::new(()));
            assert!(result.is_err());
        }
    }
}
