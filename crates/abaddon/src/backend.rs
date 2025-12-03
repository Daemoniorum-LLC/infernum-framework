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
fn to_candle_dtype(dtype: DType) -> candle_core::DType {
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

    /// CUDA device implementation.
    #[derive(Debug)]
    pub struct CudaDevice {
        device_id: usize,
        candle_device: Device,
    }

    impl CudaDevice {
        /// Creates a new CUDA device.
        pub fn new(device_id: usize) -> Result<Self> {
            let candle_device =
                Device::new_cuda(device_id).map_err(|e| infernum_core::Error::Backend {
                    backend: "cuda".to_string(),
                    message: e.to_string(),
                })?;
            Ok(Self {
                device_id,
                candle_device,
            })
        }
    }

    impl DeviceOps for CudaDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::Cuda {
                device_id: self.device_id,
            }
        }

        fn total_memory(&self) -> usize {
            // CUDA memory query would go here
            // For now, return a reasonable default
            16 * 1024 * 1024 * 1024 // 16 GB
        }

        fn available_memory(&self) -> usize {
            // CUDA available memory query would go here
            8 * 1024 * 1024 * 1024 // 8 GB
        }

        fn synchronize(&self) -> Result<()> {
            // CUDA synchronization
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

/// WebGPU backend implementation placeholder.
/// WebGPU support requires the wgpu crate and browser compatibility.
pub mod webgpu {
    //! WebGPU backend implementation.
    //!
    //! This backend is intended for browser-based inference using WebGPU.
    //! Currently a placeholder awaiting wgpu integration.

    use super::*;

    /// WebGPU tensor placeholder.
    #[derive(Debug, Clone)]
    pub struct WebGpuTensor {
        shape: Vec<usize>,
        dtype: DType,
    }

    impl TensorOps for WebGpuTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn dtype(&self) -> DType {
            self.dtype
        }
    }

    /// WebGPU device placeholder.
    #[derive(Debug, Default)]
    pub struct WebGpuDevice;

    impl DeviceOps for WebGpuDevice {
        fn device_type(&self) -> DeviceType {
            DeviceType::WebGpu
        }

        fn total_memory(&self) -> usize {
            // WebGPU memory limits vary by browser
            4 * 1024 * 1024 * 1024 // 4 GB default
        }

        fn available_memory(&self) -> usize {
            2 * 1024 * 1024 * 1024 // 2 GB default
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
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
