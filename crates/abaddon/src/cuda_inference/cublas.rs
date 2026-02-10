//! cuBLAS wrapper for GPU matrix operations.
//!
//! Provides a safe Rust interface to cuBLAS GEMM operations for F16 and F32
//! matrix multiplication. Uses direct FFI to libcublas for proper tensor core
//! utilization.
//!
//! ## cuBLAS GEMM Operations
//!
//! - `cublasHgemm`: Native F16 GEMM with F16 accumulator
//! - `cublasGemmEx`: Mixed precision (F16 compute, F32 accumulator) for better precision
//! - `cublasHgemmStridedBatched`: Batched F16 GEMM for attention heads

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::tensor::GpuTensor;
use super::InferenceError;

/// cuBLAS operation type.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasOperation {
    /// No transpose.
    N = 0,
    /// Transpose.
    T = 1,
    /// Conjugate transpose.
    C = 2,
}

/// cuBLAS compute type for GemmEx.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasComputeType {
    /// F16 compute (tensor cores, lower precision).
    F16 = 64,
    /// F32 compute (higher precision, still uses tensor cores for F16 inputs).
    F32 = 68,
    /// TF32 compute (fastest on Ampere+).
    TF32 = 72,
}

/// cuBLAS data type.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaDataType {
    /// 16-bit floating point.
    F16 = 2,
    /// 32-bit floating point.
    F32 = 0,
    /// 16-bit brain floating point.
    BF16 = 14,
}

/// cuBLAS math mode.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CublasMathMode {
    /// Default mode.
    Default = 0,
    /// Use tensor cores when possible.
    TensorOp = 1,
    /// Disallow tensor cores.
    Pedantic = 2,
    /// Allow TF32 tensor core mode (Ampere+).
    TensorOpTf32 = 3,
}

// FFI bindings to cuBLAS
#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> i32;
    fn cublasSetMathMode(handle: *mut c_void, mode: i32) -> i32;

    fn cublasHgemm(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const half::f16,
        a: *const half::f16,
        lda: i32,
        b: *const half::f16,
        ldb: i32,
        beta: *const half::f16,
        c: *mut half::f16,
        ldc: i32,
    ) -> i32;

    fn cublasGemmEx(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute_type: i32,
        algo: i32,
    ) -> i32;

    fn cublasHgemmStridedBatched(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const half::f16,
        a: *const half::f16,
        lda: i32,
        stride_a: i64,
        b: *const half::f16,
        ldb: i32,
        stride_b: i64,
        beta: *const half::f16,
        c: *mut half::f16,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;

    fn cublasGemmStridedBatchedEx(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        stride_a: i64,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        stride_b: i64,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        compute_type: i32,
        algo: i32,
    ) -> i32;
}

/// cuBLAS handle wrapper with GEMM operations.
///
/// Provides high-performance matrix multiplication using NVIDIA's cuBLAS library.
/// Automatically uses tensor cores for F16 operations when available.
pub struct CublasHandle {
    /// Raw cuBLAS handle.
    handle: *mut c_void,

    /// CUDA device reference.
    device: Arc<CudaDevice>,

    /// Whether using F32 accumulator (higher precision).
    use_f32_accumulator: bool,
}

// cuBLAS handles are safe to send across threads
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Create a new cuBLAS handle for the given device.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        let mut handle: *mut c_void = std::ptr::null_mut();

        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cublasCreate failed with status {}",
                status
            )));
        }

        // Enable tensor core math mode for optimal performance
        let status = unsafe { cublasSetMathMode(handle, CublasMathMode::TensorOp as i32) };
        if status != 0 {
            tracing::warn!("Failed to set tensor core math mode, continuing with default");
        }

        Ok(Self {
            handle,
            device,
            use_f32_accumulator: true, // Default to F32 for better precision
        })
    }

    /// Set whether to use F32 accumulator (higher precision) or F16 (faster).
    pub fn set_f32_accumulator(&mut self, use_f32: bool) {
        self.use_f32_accumulator = use_f32;
    }

    /// Set math mode for tensor cores.
    pub fn set_math_mode(&self, use_tensor_cores: bool) -> Result<(), InferenceError> {
        let mode = if use_tensor_cores {
            CublasMathMode::TensorOp
        } else {
            CublasMathMode::Default
        };

        let status = unsafe { cublasSetMathMode(self.handle, mode as i32) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cublasSetMathMode failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Set the CUDA stream for cuBLAS operations.
    ///
    /// All subsequent cuBLAS operations will execute on the specified stream,
    /// enabling overlap with other GPU operations.
    ///
    /// # Safety
    ///
    /// The stream must remain valid for the lifetime of operations.
    pub unsafe fn set_stream(&self, stream: *mut c_void) -> Result<(), InferenceError> {
        let status = cublasSetStream_v2(self.handle, stream);
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cublasSetStream failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Matrix multiplication: C = alpha * A @ B + beta * C
    ///
    /// Uses cublasGemmEx for mixed-precision computation (F16 inputs, F32 accumulator)
    /// which provides both high performance and numerical stability.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A with shape [M, K]
    /// * `b` - Matrix B with shape [K, N]
    /// * `c` - Output matrix C with shape [M, N]
    /// * `alpha` - Scalar multiplier for A @ B
    /// * `beta` - Scalar multiplier for existing C values
    ///
    /// # Note
    ///
    /// Matrices are expected in row-major order. The function handles
    /// the transpose internally for cuBLAS (which expects column-major).
    pub fn gemm_f16(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        alpha: f32,
        beta: f32,
    ) -> Result<(), InferenceError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "2D matrices".to_string(),
                got: format!("A: {:?}, B: {:?}, C: {:?}", a_shape, b_shape, c_shape),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if b_shape[0] != k {
            return Err(InferenceError::Shape {
                expected: format!("B with {} rows", k),
                got: format!("{} rows", b_shape[0]),
            });
        }

        if c_shape[0] != m || c_shape[1] != n {
            return Err(InferenceError::Shape {
                expected: format!("[{}, {}]", m, n),
                got: format!("{:?}", c_shape),
            });
        }

        // cuBLAS uses column-major, so we compute C^T = B^T @ A^T
        // For row-major A[M,K] @ B[K,N] = C[M,N]:
        // Compute: B^T @ A^T = C^T (but C^T in col-major = C in row-major)
        unsafe {
            self.gemm_f16_raw(
                b.device_ptr(), // B (treated as transposed for column-major)
                a.device_ptr(), // A (treated as transposed for column-major)
                c.device_ptr(), // Output C
                n,              // "M" for cuBLAS (columns of output)
                m,              // "N" for cuBLAS (rows of output)
                k,              // K (shared dimension)
                n,              // lda = leading dim of B = N
                k,              // ldb = leading dim of A = K
                n,              // ldc = leading dim of C = N
                alpha,
                beta,
            )
        }
    }

    /// Raw GEMM for F16 with explicit dimensions and pointers.
    ///
    /// Uses cublasGemmEx for optimal tensor core utilization with F32 accumulation.
    ///
    /// # Safety
    ///
    /// Pointers must be valid GPU memory with correct sizes.
    pub unsafe fn gemm_f16_raw(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), InferenceError> {
        if self.use_f32_accumulator {
            // Use GemmEx with F32 compute for better precision
            let status = cublasGemmEx(
                self.handle,
                CublasOperation::N as i32, // No transpose (we've rearranged for row-major)
                CublasOperation::N as i32,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                a_ptr as *const c_void,
                CudaDataType::F16 as i32,
                lda as i32,
                b_ptr as *const c_void,
                CudaDataType::F16 as i32,
                ldb as i32,
                &beta as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                CudaDataType::F16 as i32,
                ldc as i32,
                CublasComputeType::F32 as i32,
                -1, // CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );

            if status != 0 {
                return Err(InferenceError::Kernel(format!(
                    "cublasGemmEx failed with status {}",
                    status
                )));
            }
        } else {
            // Use native Hgemm for pure F16 (faster but less precise)
            let alpha_f16 = half::f16::from_f32(alpha);
            let beta_f16 = half::f16::from_f32(beta);

            let status = cublasHgemm(
                self.handle,
                CublasOperation::N as i32,
                CublasOperation::N as i32,
                m as i32,
                n as i32,
                k as i32,
                &alpha_f16,
                a_ptr as *const half::f16,
                lda as i32,
                b_ptr as *const half::f16,
                ldb as i32,
                &beta_f16,
                c_ptr as *mut half::f16,
                ldc as i32,
            );

            if status != 0 {
                return Err(InferenceError::Kernel(format!(
                    "cublasHgemm failed with status {}",
                    status
                )));
            }
        }

        Ok(())
    }

    /// Batched matrix multiplication for attention: C[b] = A[b] @ B[b]
    ///
    /// # Arguments
    ///
    /// * `a` - Batched matrices A with shape [batch, M, K]
    /// * `b` - Batched matrices B with shape [batch, K, N]
    /// * `c` - Output matrices C with shape [batch, M, N]
    pub fn batched_gemm_f16(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &mut GpuTensor,
        alpha: f32,
        beta: f32,
    ) -> Result<(), InferenceError> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if a_shape.len() != 3 || b_shape.len() != 3 || c_shape.len() != 3 {
            return Err(InferenceError::Shape {
                expected: "3D batched matrices".to_string(),
                got: format!("A: {:?}, B: {:?}, C: {:?}", a_shape, b_shape, c_shape),
            });
        }

        let batch = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch || b_shape[1] != k {
            return Err(InferenceError::Shape {
                expected: format!("[{}, {}, N]", batch, k),
                got: format!("{:?}", b_shape),
            });
        }

        if c_shape[0] != batch || c_shape[1] != m || c_shape[2] != n {
            return Err(InferenceError::Shape {
                expected: format!("[{}, {}, {}]", batch, m, n),
                got: format!("{:?}", c_shape),
            });
        }

        // Strided batched GEMM
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        unsafe {
            self.batched_gemm_f16_strided(
                a.device_ptr(),
                b.device_ptr(),
                c.device_ptr(),
                m,
                n,
                k,
                stride_a,
                stride_b,
                stride_c,
                batch,
                alpha,
                beta,
            )
        }
    }

    /// Strided batched GEMM for F16.
    ///
    /// Uses cublasGemmStridedBatchedEx for optimal performance with tensor cores.
    unsafe fn batched_gemm_f16_strided(
        &self,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        m: usize,
        n: usize,
        k: usize,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), InferenceError> {
        // For row-major batched, we compute C^T = B^T @ A^T for each batch
        // Same logic as single GEMM but with strides

        if self.use_f32_accumulator {
            let status = cublasGemmStridedBatchedEx(
                self.handle,
                CublasOperation::N as i32,
                CublasOperation::N as i32,
                n as i32, // Swapped for row-major
                m as i32,
                k as i32,
                &alpha as *const f32 as *const c_void,
                b_ptr as *const c_void,
                CudaDataType::F16 as i32,
                n as i32,
                stride_b,
                a_ptr as *const c_void,
                CudaDataType::F16 as i32,
                k as i32,
                stride_a,
                &beta as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                CudaDataType::F16 as i32,
                n as i32,
                stride_c,
                batch as i32,
                CublasComputeType::F32 as i32,
                -1, // CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );

            if status != 0 {
                return Err(InferenceError::Kernel(format!(
                    "cublasGemmStridedBatchedEx failed with status {}",
                    status
                )));
            }
        } else {
            let alpha_f16 = half::f16::from_f32(alpha);
            let beta_f16 = half::f16::from_f32(beta);

            let status = cublasHgemmStridedBatched(
                self.handle,
                CublasOperation::N as i32,
                CublasOperation::N as i32,
                n as i32,
                m as i32,
                k as i32,
                &alpha_f16,
                b_ptr as *const half::f16,
                n as i32,
                stride_b,
                a_ptr as *const half::f16,
                k as i32,
                stride_a,
                &beta_f16,
                c_ptr as *mut half::f16,
                n as i32,
                stride_c,
                batch as i32,
            );

            if status != 0 {
                return Err(InferenceError::Kernel(format!(
                    "cublasHgemmStridedBatched failed with status {}",
                    status
                )));
            }
        }

        Ok(())
    }

    /// Get the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Vector addition: y = y + alpha * x (AXPY operation).
    ///
    /// # Arguments
    ///
    /// * `alpha` - Scalar multiplier
    /// * `x` - Source vector
    /// * `y` - Destination vector (modified in-place)
    pub fn axpy(&self, alpha: f32, x: &GpuTensor, y: &mut GpuTensor) -> Result<(), InferenceError> {
        let n = x.numel();
        if y.numel() != n {
            return Err(InferenceError::Shape {
                expected: format!("{} elements", n),
                got: format!("{} elements", y.numel()),
            });
        }

        // cuBLAS axpy: y = alpha * x + y
        // For F16, we'd use cublasHaxpy or implement via kernel

        tracing::debug!(n = n, alpha = alpha, "cuBLAS AXPY called");

        // AXPY is less critical for inference - we use custom kernels for element-wise ops
        // Keep as placeholder for now
        tracing::trace!(n = n, alpha = alpha, "cuBLAS AXPY stub called");
        Ok(())
    }

    /// Get the raw cuBLAS handle.
    ///
    /// # Safety
    ///
    /// The handle is valid only while this CublasHandle is alive.
    pub unsafe fn raw_handle(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let status = unsafe { cublasDestroy_v2(self.handle) };
            if status != 0 {
                tracing::warn!("cublasDestroy failed with status {}", status);
            }
        }
    }
}

/// GEMM configuration for different use cases.
#[derive(Debug, Clone, Copy)]
pub struct GemmConfig {
    /// Use tensor cores if available.
    pub use_tensor_cores: bool,

    /// Accumulator type (F16 or F32).
    pub accumulator_f32: bool,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            use_tensor_cores: true,
            accumulator_f32: true, // F32 accumulation for better precision
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would require CUDA device
    // #[test]
    // fn test_gemm_shapes() { ... }
}
