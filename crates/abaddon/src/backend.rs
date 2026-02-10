//! Compute backend abstractions for different hardware.
//!
//! This module provides a unified interface for tensor operations across different
//! compute backends (CPU, CUDA, Metal, WebGPU). The primary implementation wraps
//! Candle tensors to provide a consistent API while leveraging Candle's optimized
//! kernels.

use async_trait::async_trait;
use candle_core::{Device, Tensor, D};
use infernum_core::{DType, DeviceType, Result};

/// Converts our DType to Candle's DType.
pub fn to_candle_dtype(dtype: DType) -> candle_core::DType {
    match dtype {
        DType::F32 => candle_core::DType::F32,
        DType::F16 => candle_core::DType::F16,
        DType::BF16 => candle_core::DType::BF16,
        DType::I8 => candle_core::DType::I64, // Candle doesn't have I8, use I64
        DType::I4 => candle_core::DType::I64, // Candle doesn't have I4, use I64
    }
}

/// Converts Candle's DType to our DType.
fn from_candle_dtype(dtype: candle_core::DType) -> DType {
    match dtype {
        candle_core::DType::F32 => DType::F32,
        candle_core::DType::F16 => DType::F16,
        candle_core::DType::BF16 => DType::BF16,
        candle_core::DType::F64 => DType::F32, // Map F64 to F32
        candle_core::DType::U8 | candle_core::DType::U32 | candle_core::DType::I64 => DType::I8,
        // Handle new candle_core DType variants (I16, I32, F8E4M3, etc.)
        _ => DType::F32,
    }
}

/// Trait for tensor operations.
pub trait TensorOps: Send + Sync {
    /// Returns the shape of the tensor.
    fn shape(&self) -> &[usize];

    /// Returns the data type of the tensor.
    fn dtype(&self) -> DType;

    /// Returns the total number of elements.
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Trait for device operations.
pub trait DeviceOps: Send + Sync {
    /// Returns the device type.
    fn device_type(&self) -> DeviceType;

    /// Returns the total memory in bytes.
    fn total_memory(&self) -> usize;

    /// Returns the available memory in bytes.
    fn available_memory(&self) -> usize;

    /// Synchronizes all pending operations.
    fn synchronize(&self) -> Result<()>;
}

/// Trait defining a compute backend.
#[async_trait]
pub trait ComputeBackend: Send + Sync {
    /// The tensor type for this backend.
    type Tensor: TensorOps;

    /// The device type for this backend.
    type Device: DeviceOps;

    /// Returns the device.
    fn device(&self) -> &Self::Device;

    /// Allocates a new tensor filled with zeros.
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor>;

    /// Creates a tensor from raw data.
    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor>;

    /// Performs matrix multiplication: C = A @ B.
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Performs batched matrix multiplication.
    fn batch_matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Performs scaled dot-product attention.
    /// Returns softmax(Q @ K^T / sqrt(d_k)) @ V with optional causal masking.
    fn attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: Option<&Self::Tensor>,
        scale: Option<f32>,
    ) -> Result<Self::Tensor>;

    /// Applies RMS normalization: x * weight / sqrt(mean(x^2) + eps).
    fn rms_norm(&self, x: &Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<Self::Tensor>;

    /// Applies Layer normalization.
    fn layer_norm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        eps: f32,
    ) -> Result<Self::Tensor>;

    /// Applies SiLU (Swish) activation: x * sigmoid(x).
    fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;

    /// Applies GELU activation.
    fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;

    /// Applies ReLU activation.
    fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor>;

    /// Applies softmax along the specified dimension.
    fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor>;

    /// Element-wise addition.
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Element-wise multiplication.
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor>;

    /// Transposes the last two dimensions.
    fn transpose(&self, x: &Self::Tensor) -> Result<Self::Tensor>;

    /// Reshapes a tensor.
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor>;

    /// Copies tensor to device.
    fn to_device(&self, tensor: &Self::Tensor) -> Result<Self::Tensor>;

    /// Copies tensor data to CPU as f32 vector.
    fn to_cpu(&self, tensor: &Self::Tensor) -> Result<Vec<f32>>;
}

/// CPU backend implementation using Candle.
pub mod cpu {
    use super::*;
    use std::sync::Arc;
    use sysinfo::System;

    /// CPU tensor wrapping a Candle tensor.
    #[derive(Debug, Clone)]
    pub struct CpuTensor {
        inner: Tensor,
        shape_cache: Vec<usize>,
    }

    impl CpuTensor {
        /// Creates a new CPU tensor from a Candle tensor.
        pub fn new(tensor: Tensor) -> Self {
            let shape_cache = tensor.dims().to_vec();
            Self {
                inner: tensor,
                shape_cache,
            }
        }

        /// Returns a reference to the underlying Candle tensor.
        #[must_use]
        pub fn inner(&self) -> &Tensor {
            &self.inner
        }

        /// Consumes self and returns the underlying Candle tensor.
        #[must_use]
        pub fn into_inner(self) -> Tensor {
            self.inner
        }
    }

    impl TensorOps for CpuTensor {
        fn shape(&self) -> &[usize] {
            &self.shape_cache
        }

        fn dtype(&self) -> DType {
            from_candle_dtype(self.inner.dtype())
        }
    }

    /// CPU device implementation with system memory tracking.
    #[derive(Debug)]
    pub struct CpuDevice {
        system: Arc<parking_lot::Mutex<System>>,
    }

    impl Default for CpuDevice {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CpuDevice {
        /// Creates a new CPU device.
        #[must_use]
        pub fn new() -> Self {
            let mut system = System::new_all();
            system.refresh_memory();
            Self {
                system: Arc::new(parking_lot::Mutex::new(system)),
            }
        }

        /// Refreshes system memory information.
        pub fn refresh(&self) {
            self.system.lock().refresh_memory();
        }
    }

    impl DeviceOps for CpuDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }

        fn total_memory(&self) -> usize {
            let system = self.system.lock();
            system.total_memory() as usize
        }

        fn available_memory(&self) -> usize {
            let mut system = self.system.lock();
            system.refresh_memory();
            system.available_memory() as usize
        }

        fn synchronize(&self) -> Result<()> {
            // CPU operations are synchronous
            Ok(())
        }
    }

    /// CPU compute backend using Candle for tensor operations.
    #[derive(Debug)]
    pub struct CpuBackend {
        device: CpuDevice,
        candle_device: Device,
    }

    impl Default for CpuBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CpuBackend {
        /// Creates a new CPU backend.
        #[must_use]
        pub fn new() -> Self {
            Self {
                device: CpuDevice::new(),
                candle_device: Device::Cpu,
            }
        }

        /// Returns the Candle device.
        #[must_use]
        pub fn candle_device(&self) -> &Device {
            &self.candle_device
        }

        /// Helper to convert Candle errors to our error type.
        fn map_err(e: candle_core::Error) -> infernum_core::Error {
            infernum_core::Error::Backend {
                backend: "cpu".to_string(),
                message: e.to_string(),
            }
        }
    }

    #[async_trait]
    impl ComputeBackend for CpuBackend {
        type Tensor = CpuTensor;
        type Device = CpuDevice;

        fn device(&self) -> &Self::Device {
            &self.device
        }

        fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor> {
            let candle_dtype = to_candle_dtype(dtype);
            let tensor =
                Tensor::zeros(shape, candle_dtype, &self.candle_device).map_err(Self::map_err)?;
            Ok(CpuTensor::new(tensor))
        }

        fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
            let tensor =
                Tensor::from_slice(data, shape, &self.candle_device).map_err(Self::map_err)?;
            Ok(CpuTensor::new(tensor))
        }

        fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn batch_matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            // Candle's matmul handles batched operations
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn attention(
            &self,
            q: &Self::Tensor,
            k: &Self::Tensor,
            v: &Self::Tensor,
            mask: Option<&Self::Tensor>,
            scale: Option<f32>,
        ) -> Result<Self::Tensor> {
            // Get dimensions: q is (batch, heads, seq_len, head_dim)
            let head_dim = q.inner.dim(D::Minus1).map_err(Self::map_err)?;
            let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

            // Compute Q @ K^T
            let k_t = k
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            let scores = q.inner.matmul(&k_t).map_err(Self::map_err)?;

            // Scale
            let scores = (scores * scale as f64).map_err(Self::map_err)?;

            // Apply mask if provided
            let scores = match mask {
                Some(m) => scores.broadcast_add(&m.inner).map_err(Self::map_err)?,
                None => scores,
            };

            // Softmax over last dimension
            let attn_weights = candle_nn::ops::softmax_last_dim(&scores).map_err(Self::map_err)?;

            // Attention output: weights @ V
            let output = attn_weights.matmul(&v.inner).map_err(Self::map_err)?;

            Ok(CpuTensor::new(output))
        }

        fn rms_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            eps: f32,
        ) -> Result<Self::Tensor> {
            // RMS norm: x * weight / sqrt(mean(x^2) + eps)
            let dtype = x.inner.dtype();

            // Convert to f32 for numerical stability
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;

            // Compute variance (mean of squares)
            let variance = x_f32
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;

            // Normalize
            let x_normed = x_f32
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;

            // Convert back to original dtype and apply weight
            let result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;

            Ok(CpuTensor::new(result))
        }

        fn layer_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            bias: Option<&Self::Tensor>,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;

            // Compute mean and variance
            let mean = x_f32.mean_keepdim(D::Minus1).map_err(Self::map_err)?;
            let x_centered = x_f32.broadcast_sub(&mean).map_err(Self::map_err)?;
            let variance = x_centered
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;

            // Normalize
            let x_normed = x_centered
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;

            // Apply weight
            let mut result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;

            // Apply bias if provided
            if let Some(b) = bias {
                result = result.broadcast_add(&b.inner).map_err(Self::map_err)?;
            }

            Ok(CpuTensor::new(result))
        }

        fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = candle_nn::ops::silu(&x.inner).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.gelu_erf().map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.relu().map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor> {
            let result = if dim == -1 {
                candle_nn::ops::softmax_last_dim(&x.inner).map_err(Self::map_err)?
            } else {
                candle_nn::ops::softmax(&x.inner, dim as usize).map_err(Self::map_err)?
            };
            Ok(CpuTensor::new(result))
        }

        fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_add(&b.inner).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_mul(&b.inner).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn transpose(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor> {
            let result = x.inner.reshape(shape).map_err(Self::map_err)?;
            Ok(CpuTensor::new(result))
        }

        fn to_device(&self, tensor: &Self::Tensor) -> Result<Self::Tensor> {
            // Already on CPU, just clone
            Ok(CpuTensor::new(tensor.inner.clone()))
        }

        fn to_cpu(&self, tensor: &Self::Tensor) -> Result<Vec<f32>> {
            let flat = tensor.inner.flatten_all().map_err(Self::map_err)?;
            let data: Vec<f32> = flat
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?
                .to_vec1()
                .map_err(Self::map_err)?;
            Ok(data)
        }
    }
}

/// CUDA backend implementation using Candle.
#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;

    /// CUDA tensor wrapping a Candle tensor.
    #[derive(Debug, Clone)]
    pub struct CudaTensor {
        inner: Tensor,
        shape_cache: Vec<usize>,
    }

    impl CudaTensor {
        /// Creates a new CUDA tensor from a Candle tensor.
        pub fn new(tensor: Tensor) -> Self {
            let shape_cache = tensor.dims().to_vec();
            Self {
                inner: tensor,
                shape_cache,
            }
        }

        /// Returns a reference to the underlying Candle tensor.
        #[must_use]
        pub fn inner(&self) -> &Tensor {
            &self.inner
        }
    }

    impl TensorOps for CudaTensor {
        fn shape(&self) -> &[usize] {
            &self.shape_cache
        }

        fn dtype(&self) -> DType {
            from_candle_dtype(self.inner.dtype())
        }
    }

    /// GPU architecture capability info
    #[derive(Debug, Clone, Copy)]
    pub struct GpuCapabilities {
        /// Compute capability major version (e.g., 8 for Ampere/Ada)
        pub compute_major: u32,
        /// Compute capability minor version (e.g., 9 for Ada Lovelace)
        pub compute_minor: u32,
        /// Total VRAM in bytes
        pub total_memory: usize,
        /// Has tensor cores (compute >= 7.0)
        pub has_tensor_cores: bool,
        /// Has BF16 support (compute >= 8.0)
        pub has_bf16: bool,
        /// Has FP8 support (compute >= 8.9)
        pub has_fp8: bool,
    }

    impl GpuCapabilities {
        /// Compute capability as a float (e.g., 8.9)
        pub fn compute_capability(&self) -> f32 {
            self.compute_major as f32 + self.compute_minor as f32 / 10.0
        }

        /// Whether this GPU supports FP16 tensor core operations efficiently
        pub fn supports_fp16_tensor_cores(&self) -> bool {
            self.has_tensor_cores
        }

        /// Whether this GPU supports BF16 tensor core operations
        pub fn supports_bf16_tensor_cores(&self) -> bool {
            self.has_bf16
        }

        /// Get recommended dtype for this GPU
        pub fn recommended_dtype(&self) -> DType {
            if self.has_bf16 {
                DType::BF16 // Ada/Ampere - use BF16 for best perf
            } else if self.has_tensor_cores {
                DType::F16 // Volta/Turing - use FP16
            } else {
                DType::F32 // Older GPUs - F32 only
            }
        }
    }

    /// CUDA device implementation.
    #[derive(Debug)]
    pub struct CudaDevice {
        device_id: usize,
        candle_device: Device,
        capabilities: GpuCapabilities,
    }

    impl CudaDevice {
        /// Creates a new CUDA device with capability detection.
        pub fn new(device_id: usize) -> Result<Self> {
            let candle_device =
                Device::new_cuda(device_id).map_err(|e| infernum_core::Error::Backend {
                    backend: "cuda".to_string(),
                    message: e.to_string(),
                })?;

            // Query GPU capabilities using cudarc
            let capabilities = Self::query_capabilities(device_id)?;

            tracing::info!(
                device_id = device_id,
                compute = %format!("{}.{}", capabilities.compute_major, capabilities.compute_minor),
                vram_gb = capabilities.total_memory / (1024 * 1024 * 1024),
                tensor_cores = capabilities.has_tensor_cores,
                bf16 = capabilities.has_bf16,
                "CUDA device initialized"
            );

            Ok(Self {
                device_id,
                candle_device,
                capabilities,
            })
        }

        /// Query GPU capabilities from CUDA runtime
        fn query_capabilities(device_id: usize) -> Result<GpuCapabilities> {
            use cudarc::driver::CudaDevice as CudarcDevice;

            let cuda_dev =
                CudarcDevice::new(device_id).map_err(|e| infernum_core::Error::Backend {
                    backend: "cuda".to_string(),
                    message: format!("Failed to query CUDA device: {}", e),
                })?;

            // Get device attributes
            let compute_major = cuda_dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .unwrap_or(7) as u32;

            let compute_minor = cuda_dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .unwrap_or(0) as u32;

            let total_memory = cuda_dev
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)
                .map(|v| v as usize)
                // Fallback: use memory info from device
                .unwrap_or(16 * 1024 * 1024 * 1024);

            // Actually get total memory using a better method if available
            // For now, detect based on compute capability common configurations
            let total_memory = Self::estimate_vram(compute_major, compute_minor, total_memory);

            // Capability thresholds:
            // - 7.0+ (Volta): Tensor Cores, FP16
            // - 8.0+ (Ampere): BF16, improved tensor cores
            // - 8.9+ (Ada Lovelace): FP8, 4th gen tensor cores
            let has_tensor_cores = compute_major >= 7;
            let has_bf16 = compute_major >= 8;
            let has_fp8 = compute_major > 8 || (compute_major == 8 && compute_minor >= 9);

            Ok(GpuCapabilities {
                compute_major,
                compute_minor,
                total_memory,
                has_tensor_cores,
                has_bf16,
                has_fp8,
            })
        }

        /// Estimate VRAM based on compute capability and common GPU configurations
        fn estimate_vram(major: u32, minor: u32, _hint: usize) -> usize {
            // Common VRAM sizes for different compute capabilities
            // This is a fallback when direct memory query fails
            match (major, minor) {
                // Ada Lovelace (RTX 40 series)
                (8, 9) => 24 * 1024 * 1024 * 1024, // RTX 4090/4500: 24GB typical
                // Ampere (RTX 30 series, A100)
                (8, 6) => 12 * 1024 * 1024 * 1024, // RTX 3080: 12GB typical
                (8, 0) => 40 * 1024 * 1024 * 1024, // A100: 40/80GB
                // Turing (RTX 20 series)
                (7, 5) => 8 * 1024 * 1024 * 1024, // RTX 2070: 8GB typical
                // Volta
                (7, 0) => 16 * 1024 * 1024 * 1024, // V100: 16/32GB
                // Older
                _ => 8 * 1024 * 1024 * 1024, // Safe default
            }
        }

        /// Get GPU capabilities
        pub fn capabilities(&self) -> &GpuCapabilities {
            &self.capabilities
        }

        /// Get recommended dtype for this device
        pub fn recommended_dtype(&self) -> DType {
            self.capabilities.recommended_dtype()
        }
    }

    impl DeviceOps for CudaDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::Cuda {
                device_id: self.device_id,
            }
        }

        fn total_memory(&self) -> usize {
            self.capabilities.total_memory
        }

        fn available_memory(&self) -> usize {
            // Estimate 80% available after driver/framework overhead
            // In production, query cudaMemGetInfo for accurate values
            (self.capabilities.total_memory as f64 * 0.8) as usize
        }

        fn synchronize(&self) -> Result<()> {
            // CUDA synchronization via cudarc
            // The candle device handles this internally for us
            Ok(())
        }
    }

    /// CUDA compute backend using Candle.
    #[derive(Debug)]
    pub struct CudaBackend {
        device: CudaDevice,
    }

    impl CudaBackend {
        /// Creates a new CUDA backend for the specified device.
        pub fn new(device_id: usize) -> Result<Self> {
            let device = CudaDevice::new(device_id)?;
            Ok(Self { device })
        }

        fn map_err(e: candle_core::Error) -> infernum_core::Error {
            infernum_core::Error::Backend {
                backend: "cuda".to_string(),
                message: e.to_string(),
            }
        }
    }

    #[async_trait]
    impl ComputeBackend for CudaBackend {
        type Tensor = CudaTensor;
        type Device = CudaDevice;

        fn device(&self) -> &Self::Device {
            &self.device
        }

        fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor> {
            let candle_dtype = to_candle_dtype(dtype);
            let tensor = Tensor::zeros(shape, candle_dtype, &self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(CudaTensor::new(tensor))
        }

        fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
            let cpu_tensor =
                Tensor::from_slice(data, shape, &Device::Cpu).map_err(Self::map_err)?;
            let tensor = cpu_tensor
                .to_device(&self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(CudaTensor::new(tensor))
        }

        fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn batch_matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn attention(
            &self,
            q: &Self::Tensor,
            k: &Self::Tensor,
            v: &Self::Tensor,
            mask: Option<&Self::Tensor>,
            scale: Option<f32>,
        ) -> Result<Self::Tensor> {
            let head_dim = q.inner.dim(D::Minus1).map_err(Self::map_err)?;
            let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

            let k_t = k
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            let scores = q.inner.matmul(&k_t).map_err(Self::map_err)?;
            let scores = (scores * scale as f64).map_err(Self::map_err)?;

            let scores = match mask {
                Some(m) => scores.broadcast_add(&m.inner).map_err(Self::map_err)?,
                None => scores,
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&scores).map_err(Self::map_err)?;
            let output = attn_weights.matmul(&v.inner).map_err(Self::map_err)?;

            Ok(CudaTensor::new(output))
        }

        fn rms_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let variance = x_f32
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_f32
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn layer_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            bias: Option<&Self::Tensor>,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let mean = x_f32.mean_keepdim(D::Minus1).map_err(Self::map_err)?;
            let x_centered = x_f32.broadcast_sub(&mean).map_err(Self::map_err)?;
            let variance = x_centered
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_centered
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let mut result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            if let Some(b) = bias {
                result = result.broadcast_add(&b.inner).map_err(Self::map_err)?;
            }
            Ok(CudaTensor::new(result))
        }

        fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = candle_nn::ops::silu(&x.inner).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.gelu_erf().map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.relu().map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor> {
            let result = if dim == -1 {
                candle_nn::ops::softmax_last_dim(&x.inner).map_err(Self::map_err)?
            } else {
                candle_nn::ops::softmax(&x.inner, dim as usize).map_err(Self::map_err)?
            };
            Ok(CudaTensor::new(result))
        }

        fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_add(&b.inner).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_mul(&b.inner).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn transpose(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor> {
            let result = x.inner.reshape(shape).map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn to_device(&self, tensor: &Self::Tensor) -> Result<Self::Tensor> {
            let result = tensor
                .inner
                .to_device(&self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(CudaTensor::new(result))
        }

        fn to_cpu(&self, tensor: &Self::Tensor) -> Result<Vec<f32>> {
            let cpu_tensor = tensor
                .inner
                .to_device(&Device::Cpu)
                .map_err(Self::map_err)?;
            let flat = cpu_tensor.flatten_all().map_err(Self::map_err)?;
            let data: Vec<f32> = flat
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?
                .to_vec1()
                .map_err(Self::map_err)?;
            Ok(data)
        }
    }
}

/// Metal backend implementation using Candle.
#[cfg(feature = "metal")]
pub mod metal {
    use super::*;

    /// Metal tensor wrapping a Candle tensor.
    #[derive(Debug, Clone)]
    pub struct MetalTensor {
        inner: Tensor,
        shape_cache: Vec<usize>,
    }

    impl MetalTensor {
        /// Creates a new Metal tensor from a Candle tensor.
        pub fn new(tensor: Tensor) -> Self {
            let shape_cache = tensor.dims().to_vec();
            Self {
                inner: tensor,
                shape_cache,
            }
        }

        /// Returns a reference to the underlying Candle tensor.
        #[must_use]
        pub fn inner(&self) -> &Tensor {
            &self.inner
        }
    }

    impl TensorOps for MetalTensor {
        fn shape(&self) -> &[usize] {
            &self.shape_cache
        }

        fn dtype(&self) -> DType {
            from_candle_dtype(self.inner.dtype())
        }
    }

    /// Metal device implementation.
    #[derive(Debug)]
    pub struct MetalDevice {
        device_id: usize,
        candle_device: Device,
    }

    impl MetalDevice {
        /// Creates a new Metal device.
        pub fn new(device_id: usize) -> Result<Self> {
            let candle_device =
                Device::new_metal(device_id).map_err(|e| infernum_core::Error::Backend {
                    backend: "metal".to_string(),
                    message: e.to_string(),
                })?;
            Ok(Self {
                device_id,
                candle_device,
            })
        }
    }

    impl DeviceOps for MetalDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::Metal {
                device_id: self.device_id,
            }
        }

        fn total_memory(&self) -> usize {
            // Metal unified memory - return system memory estimate
            16 * 1024 * 1024 * 1024 // 16 GB
        }

        fn available_memory(&self) -> usize {
            8 * 1024 * 1024 * 1024 // 8 GB
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Metal compute backend using Candle.
    #[derive(Debug)]
    pub struct MetalBackend {
        device: MetalDevice,
    }

    impl MetalBackend {
        /// Creates a new Metal backend.
        pub fn new(device_id: usize) -> Result<Self> {
            let device = MetalDevice::new(device_id)?;
            Ok(Self { device })
        }

        fn map_err(e: candle_core::Error) -> infernum_core::Error {
            infernum_core::Error::Backend {
                backend: "metal".to_string(),
                message: e.to_string(),
            }
        }
    }

    #[async_trait]
    impl ComputeBackend for MetalBackend {
        type Tensor = MetalTensor;
        type Device = MetalDevice;

        fn device(&self) -> &Self::Device {
            &self.device
        }

        fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor> {
            let candle_dtype = to_candle_dtype(dtype);
            let tensor = Tensor::zeros(shape, candle_dtype, &self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(MetalTensor::new(tensor))
        }

        fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
            let cpu_tensor =
                Tensor::from_slice(data, shape, &Device::Cpu).map_err(Self::map_err)?;
            let tensor = cpu_tensor
                .to_device(&self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(MetalTensor::new(tensor))
        }

        fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn batch_matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn attention(
            &self,
            q: &Self::Tensor,
            k: &Self::Tensor,
            v: &Self::Tensor,
            mask: Option<&Self::Tensor>,
            scale: Option<f32>,
        ) -> Result<Self::Tensor> {
            let head_dim = q.inner.dim(D::Minus1).map_err(Self::map_err)?;
            let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

            let k_t = k
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            let scores = q.inner.matmul(&k_t).map_err(Self::map_err)?;
            let scores = (scores * scale as f64).map_err(Self::map_err)?;

            let scores = match mask {
                Some(m) => scores.broadcast_add(&m.inner).map_err(Self::map_err)?,
                None => scores,
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&scores).map_err(Self::map_err)?;
            let output = attn_weights.matmul(&v.inner).map_err(Self::map_err)?;

            Ok(MetalTensor::new(output))
        }

        fn rms_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let variance = x_f32
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_f32
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn layer_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            bias: Option<&Self::Tensor>,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let mean = x_f32.mean_keepdim(D::Minus1).map_err(Self::map_err)?;
            let x_centered = x_f32.broadcast_sub(&mean).map_err(Self::map_err)?;
            let variance = x_centered
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_centered
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let mut result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            if let Some(b) = bias {
                result = result.broadcast_add(&b.inner).map_err(Self::map_err)?;
            }
            Ok(MetalTensor::new(result))
        }

        fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = candle_nn::ops::silu(&x.inner).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.gelu_erf().map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.relu().map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor> {
            let result = if dim == -1 {
                candle_nn::ops::softmax_last_dim(&x.inner).map_err(Self::map_err)?
            } else {
                candle_nn::ops::softmax(&x.inner, dim as usize).map_err(Self::map_err)?
            };
            Ok(MetalTensor::new(result))
        }

        fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_add(&b.inner).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_mul(&b.inner).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn transpose(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor> {
            let result = x.inner.reshape(shape).map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn to_device(&self, tensor: &Self::Tensor) -> Result<Self::Tensor> {
            let result = tensor
                .inner
                .to_device(&self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(MetalTensor::new(result))
        }

        fn to_cpu(&self, tensor: &Self::Tensor) -> Result<Vec<f32>> {
            let cpu_tensor = tensor
                .inner
                .to_device(&Device::Cpu)
                .map_err(Self::map_err)?;
            let flat = cpu_tensor.flatten_all().map_err(Self::map_err)?;
            let data: Vec<f32> = flat
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?
                .to_vec1()
                .map_err(Self::map_err)?;
            Ok(data)
        }
    }
}

/// WebGPU backend implementation.
///
/// This backend provides WebGPU-compatible tensor operations for browser-based
/// inference. Currently uses a CPU fallback implementation while maintaining
/// the WebGPU interface for future wgpu integration.
pub mod webgpu {
    //! WebGPU backend implementation.
    //!
    //! This backend is intended for browser-based inference using WebGPU.
    //! Currently uses CPU fallback until full wgpu integration is complete.

    use super::*;

    /// WebGPU tensor wrapping a Candle tensor (CPU-backed for now).
    #[derive(Debug, Clone)]
    pub struct WebGpuTensor {
        inner: Tensor,
        shape_cache: Vec<usize>,
    }

    impl WebGpuTensor {
        /// Creates a new WebGPU tensor from a Candle tensor.
        pub fn new(tensor: Tensor) -> Self {
            let shape_cache = tensor.dims().to_vec();
            Self {
                inner: tensor,
                shape_cache,
            }
        }

        /// Returns a reference to the underlying Candle tensor.
        #[must_use]
        pub fn inner(&self) -> &Tensor {
            &self.inner
        }

        /// Consumes self and returns the underlying Candle tensor.
        #[must_use]
        pub fn into_inner(self) -> Tensor {
            self.inner
        }
    }

    impl TensorOps for WebGpuTensor {
        fn shape(&self) -> &[usize] {
            &self.shape_cache
        }

        fn dtype(&self) -> DType {
            from_candle_dtype(self.inner.dtype())
        }
    }

    /// WebGPU device implementation.
    ///
    /// Tracks memory limits typical for WebGPU in browsers.
    #[derive(Debug)]
    pub struct WebGpuDevice {
        /// Maximum buffer size (WebGPU typically limits to 128MB-2GB per buffer)
        max_buffer_size: usize,
        /// Total available memory estimate
        total_memory: usize,
        /// Candle device (CPU fallback)
        candle_device: Device,
    }

    impl Default for WebGpuDevice {
        fn default() -> Self {
            Self::new()
        }
    }

    impl WebGpuDevice {
        /// Creates a new WebGPU device with default memory limits.
        #[must_use]
        pub fn new() -> Self {
            Self {
                // WebGPU maxBufferSize is typically 256MB-2GB depending on browser/device
                max_buffer_size: 256 * 1024 * 1024,
                // Estimate 4GB total for most devices
                total_memory: 4 * 1024 * 1024 * 1024,
                candle_device: Device::Cpu,
            }
        }

        /// Creates a WebGPU device with custom memory limits.
        pub fn with_limits(max_buffer_size: usize, total_memory: usize) -> Self {
            Self {
                max_buffer_size,
                total_memory,
                candle_device: Device::Cpu,
            }
        }

        /// Returns the maximum buffer size for this device.
        #[must_use]
        pub fn max_buffer_size(&self) -> usize {
            self.max_buffer_size
        }
    }

    impl DeviceOps for WebGpuDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::WebGpu
        }

        fn total_memory(&self) -> usize {
            self.total_memory
        }

        fn available_memory(&self) -> usize {
            // Estimate 50% available after browser/OS overhead
            self.total_memory / 2
        }

        fn synchronize(&self) -> Result<()> {
            // WebGPU operations are async, but our CPU fallback is synchronous
            Ok(())
        }
    }

    /// WebGPU compute backend using CPU fallback.
    ///
    /// This implementation provides the WebGPU interface while using CPU
    /// operations internally. When wgpu integration is complete, this will
    /// use actual GPU compute shaders.
    #[derive(Debug)]
    pub struct WebGpuBackend {
        device: WebGpuDevice,
    }

    impl Default for WebGpuBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl WebGpuBackend {
        /// Creates a new WebGPU backend.
        #[must_use]
        pub fn new() -> Self {
            Self {
                device: WebGpuDevice::new(),
            }
        }

        /// Creates a WebGPU backend with custom device limits.
        pub fn with_device(device: WebGpuDevice) -> Self {
            Self { device }
        }

        /// Returns the Candle device used for operations.
        #[must_use]
        pub fn candle_device(&self) -> &Device {
            &self.device.candle_device
        }

        /// Helper to convert Candle errors to our error type.
        fn map_err(e: candle_core::Error) -> infernum_core::Error {
            infernum_core::Error::Backend {
                backend: "webgpu".to_string(),
                message: e.to_string(),
            }
        }

        /// Checks if a tensor would exceed WebGPU buffer limits.
        fn check_buffer_size(&self, shape: &[usize], dtype: DType) -> Result<()> {
            let elem_size = match dtype {
                DType::F32 | DType::I8 => 4,
                DType::F16 | DType::BF16 => 2,
                DType::I4 => 1,
            };
            let total_size: usize = shape.iter().product::<usize>() * elem_size;

            if total_size > self.device.max_buffer_size {
                return Err(infernum_core::Error::Backend {
                    backend: "webgpu".to_string(),
                    message: format!(
                        "Tensor size {} exceeds WebGPU max buffer size {}",
                        total_size, self.device.max_buffer_size
                    ),
                });
            }
            Ok(())
        }
    }

    #[async_trait]
    impl ComputeBackend for WebGpuBackend {
        type Tensor = WebGpuTensor;
        type Device = WebGpuDevice;

        fn device(&self) -> &Self::Device {
            &self.device
        }

        fn allocate(&self, shape: &[usize], dtype: DType) -> Result<Self::Tensor> {
            self.check_buffer_size(shape, dtype)?;
            let candle_dtype = to_candle_dtype(dtype);
            let tensor = Tensor::zeros(shape, candle_dtype, &self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(tensor))
        }

        fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor> {
            self.check_buffer_size(shape, DType::F32)?;
            let tensor = Tensor::from_slice(data, shape, &self.device.candle_device)
                .map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(tensor))
        }

        fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn batch_matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.matmul(&b.inner).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn attention(
            &self,
            q: &Self::Tensor,
            k: &Self::Tensor,
            v: &Self::Tensor,
            mask: Option<&Self::Tensor>,
            scale: Option<f32>,
        ) -> Result<Self::Tensor> {
            let head_dim = q.inner.dim(D::Minus1).map_err(Self::map_err)?;
            let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

            let k_t = k
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            let scores = q.inner.matmul(&k_t).map_err(Self::map_err)?;
            let scores = (scores * scale as f64).map_err(Self::map_err)?;

            let scores = match mask {
                Some(m) => scores.broadcast_add(&m.inner).map_err(Self::map_err)?,
                None => scores,
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&scores).map_err(Self::map_err)?;
            let output = attn_weights.matmul(&v.inner).map_err(Self::map_err)?;

            Ok(WebGpuTensor::new(output))
        }

        fn rms_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let variance = x_f32
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_f32
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn layer_norm(
            &self,
            x: &Self::Tensor,
            weight: &Self::Tensor,
            bias: Option<&Self::Tensor>,
            eps: f32,
        ) -> Result<Self::Tensor> {
            let dtype = x.inner.dtype();
            let x_f32 = x
                .inner
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?;
            let mean = x_f32.mean_keepdim(D::Minus1).map_err(Self::map_err)?;
            let x_centered = x_f32.broadcast_sub(&mean).map_err(Self::map_err)?;
            let variance = x_centered
                .sqr()
                .map_err(Self::map_err)?
                .mean_keepdim(D::Minus1)
                .map_err(Self::map_err)?;
            let x_normed = x_centered
                .broadcast_div(
                    &(variance + eps as f64)
                        .map_err(Self::map_err)?
                        .sqrt()
                        .map_err(Self::map_err)?,
                )
                .map_err(Self::map_err)?;
            let mut result = x_normed
                .to_dtype(dtype)
                .map_err(Self::map_err)?
                .broadcast_mul(&weight.inner)
                .map_err(Self::map_err)?;
            if let Some(b) = bias {
                result = result.broadcast_add(&b.inner).map_err(Self::map_err)?;
            }
            Ok(WebGpuTensor::new(result))
        }

        fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = candle_nn::ops::silu(&x.inner).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn gelu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.gelu_erf().map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn relu(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x.inner.relu().map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn softmax(&self, x: &Self::Tensor, dim: i32) -> Result<Self::Tensor> {
            let result = if dim == -1 {
                candle_nn::ops::softmax_last_dim(&x.inner).map_err(Self::map_err)?
            } else {
                candle_nn::ops::softmax(&x.inner, dim as usize).map_err(Self::map_err)?
            };
            Ok(WebGpuTensor::new(result))
        }

        fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_add(&b.inner).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor> {
            let result = a.inner.broadcast_mul(&b.inner).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn transpose(&self, x: &Self::Tensor) -> Result<Self::Tensor> {
            let result = x
                .inner
                .transpose(D::Minus2, D::Minus1)
                .map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor> {
            let result = x.inner.reshape(shape).map_err(Self::map_err)?;
            Ok(WebGpuTensor::new(result))
        }

        fn to_device(&self, tensor: &Self::Tensor) -> Result<Self::Tensor> {
            // Already using CPU fallback, just clone
            Ok(WebGpuTensor::new(tensor.inner.clone()))
        }

        fn to_cpu(&self, tensor: &Self::Tensor) -> Result<Vec<f32>> {
            let flat = tensor.inner.flatten_all().map_err(Self::map_err)?;
            let data: Vec<f32> = flat
                .to_dtype(candle_core::DType::F32)
                .map_err(Self::map_err)?
                .to_vec1()
                .map_err(Self::map_err)?;
            Ok(data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::*;
    use super::*;

    #[test]
    fn test_cpu_backend_allocate() {
        let backend = CpuBackend::new();
        let tensor = backend.allocate(&[2, 3, 4], DType::F32).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_cpu_backend_from_slice() {
        let backend = CpuBackend::new();
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let tensor = backend.from_slice(&data, &[3, 4]).unwrap();
        assert_eq!(tensor.shape(), &[3, 4]);

        let result = backend.to_cpu(&tensor).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_cpu_backend_matmul() {
        let backend = CpuBackend::new();

        // 2x3 @ 3x4 = 2x4
        let a = backend
            .from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
        let b = backend
            .from_slice(
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                &[3, 4],
            )
            .unwrap();

        let c = backend.matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);

        let result = backend.to_cpu(&c).unwrap();
        // [1,2,3] @ [[1,2,3,4],[5,6,7,8],[9,10,11,12]] = [38,44,50,56]
        // [4,5,6] @ ... = [83,98,113,128]
        assert_eq!(
            result,
            vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]
        );
    }

    #[test]
    fn test_cpu_backend_softmax() {
        let backend = CpuBackend::new();
        let x = backend.from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = backend.softmax(&x, -1).unwrap();

        let result = backend.to_cpu(&y).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");
    }

    #[test]
    fn test_cpu_backend_silu() {
        let backend = CpuBackend::new();
        let x = backend.from_slice(&[0.0, 1.0, -1.0, 2.0], &[4]).unwrap();
        let y = backend.silu(&x).unwrap();

        let result = backend.to_cpu(&y).unwrap();
        // SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269, SiLU(2) ≈ 1.762
        assert!((result[0] - 0.0).abs() < 1e-5);
        assert!((result[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_backend_rms_norm() {
        let backend = CpuBackend::new();
        let x = backend.from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let weight = backend.from_slice(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        let y = backend.rms_norm(&x, &weight, 1e-5).unwrap();
        let result = backend.to_cpu(&y).unwrap();

        // RMS = sqrt(mean([1,4,9,16])) = sqrt(7.5) ≈ 2.739
        // Normalized: [0.365, 0.730, 1.095, 1.461]
        assert!(result[0] > 0.0);
        assert!(result[3] > result[0]); // Should preserve relative magnitudes
    }

    #[test]
    fn test_cpu_device_memory() {
        let device = CpuDevice::new();
        assert!(device.total_memory() > 0);
        assert!(device.available_memory() > 0);
        assert!(device.available_memory() <= device.total_memory());
    }
}
