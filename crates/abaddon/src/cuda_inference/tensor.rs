//! GPU-resident tensor type.
//!
//! `GpuTensor` wraps a CUDA device buffer with shape and dtype metadata,
//! providing a lightweight tensor abstraction that stays entirely on GPU.
//!
//! ## Optimization: Zero-Copy Views
//!
//! Many tensor operations (reshape, slice) now return views that share
//! the same underlying GPU memory. This eliminates synchronous device-to-device
//! copies that were causing GPU pipeline stalls.
//!
//! - `reshape()` - Returns view with new shape, no copy
//! - `slice_dim0()` - Returns view with offset pointer, no copy
//! - `view_layer()` - Returns view into 4D tensor layer, no copy
//!
//! Use `clone_tensor()` when you need an independent copy.
//!
//! ## Async Operations
//!
//! For stream-based async execution, use the `*_async` variants:
//! - `copy_from_host_async()` - Non-blocking H2D copy
//! - `copy_to_host_async()` - Non-blocking D2H copy
//! - `copy_from_at_dim0_async()` - Non-blocking D2D copy

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};

use super::InferenceError;

// CUDA runtime async memory copy FFI bindings
#[link(name = "cudart")]
extern "C" {
    fn cudaMemcpyAsync(
        dst: u64,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// CUDA memory copy kinds
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// Data types supported by GpuTensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDType {
    /// 16-bit floating point (half precision).
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// 32-bit floating point.
    F32,
    /// 32-bit signed integer.
    I32,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer (for packed INT4).
    U8,
    /// 4-bit integer (packed, 2 per byte).
    I4,
}

impl GpuDType {
    /// Bytes per element (for packed types, bytes per logical element).
    pub fn size_bytes(&self) -> usize {
        match self {
            GpuDType::F16 | GpuDType::BF16 => 2,
            GpuDType::F32 | GpuDType::I32 => 4,
            GpuDType::I8 | GpuDType::U8 => 1,
            GpuDType::I4 => 1, // 2 elements per byte, but we count storage
        }
    }

    /// Whether this is a packed type (multiple values per byte).
    pub fn is_packed(&self) -> bool {
        matches!(self, GpuDType::I4)
    }

    /// Elements per storage byte for packed types.
    pub fn pack_factor(&self) -> usize {
        match self {
            GpuDType::I4 => 2,
            _ => 1,
        }
    }
}

/// Shared GPU buffer for zero-copy views.
#[derive(Debug)]
struct SharedBuffer {
    data: CudaSlice<u8>,
}

/// GPU-resident tensor with shape and dtype metadata.
///
/// This type keeps data entirely on GPU, avoiding the CPU round-trip
/// required by Candle tensors.
///
/// Supports zero-copy views via `offset` field - views share the same
/// underlying buffer but with different shape/offset.
#[derive(Debug)]
pub struct GpuTensor {
    /// Raw data on GPU (Arc for shared ownership in views).
    buffer: Arc<SharedBuffer>,

    /// Tensor shape (e.g., [batch, seq, hidden]).
    shape: Vec<usize>,

    /// Data type.
    dtype: GpuDType,

    /// CUDA device reference.
    device: Arc<CudaDevice>,

    /// Stride for each dimension (in elements, not bytes).
    strides: Vec<usize>,

    /// Byte offset from data pointer (for views).
    offset: usize,

    /// Logical size in bytes (may be less than data.len() for views).
    logical_size: usize,
}

impl GpuTensor {
    /// Create a new tensor from existing GPU data.
    ///
    /// # Safety
    ///
    /// The `data` slice must contain valid data for the given dtype and shape.
    pub fn from_cuda_slice(
        data: CudaSlice<u8>,
        shape: Vec<usize>,
        dtype: GpuDType,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let num_elements: usize = shape.iter().product();
        let expected_bytes = if dtype.is_packed() {
            (num_elements + dtype.pack_factor() - 1) / dtype.pack_factor()
        } else {
            num_elements * dtype.size_bytes()
        };

        if data.len() < expected_bytes {
            return Err(InferenceError::Shape {
                expected: format!("{} bytes for {:?} {:?}", expected_bytes, shape, dtype),
                got: format!("{} bytes", data.len()),
            });
        }

        // Compute strides (row-major order)
        let strides = compute_strides(&shape);
        let logical_size = expected_bytes;

        Ok(Self {
            buffer: Arc::new(SharedBuffer { data }),
            shape,
            dtype,
            device,
            strides,
            offset: 0,
            logical_size,
        })
    }

    /// Allocate a new zeroed tensor on GPU.
    pub fn zeros(
        shape: Vec<usize>,
        dtype: GpuDType,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let num_elements: usize = shape.iter().product();
        let num_bytes = if dtype.is_packed() {
            (num_elements + dtype.pack_factor() - 1) / dtype.pack_factor()
        } else {
            num_elements * dtype.size_bytes()
        };

        let data: CudaSlice<u8> = device
            .alloc_zeros(num_bytes)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        let strides = compute_strides(&shape);

        Ok(Self {
            buffer: Arc::new(SharedBuffer { data }),
            shape,
            dtype,
            device,
            strides,
            offset: 0,
            logical_size: num_bytes,
        })
    }

    /// Allocate uninitialized tensor on GPU.
    ///
    /// # Safety
    ///
    /// Caller must initialize data before reading.
    pub unsafe fn uninit(
        shape: Vec<usize>,
        dtype: GpuDType,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let num_elements: usize = shape.iter().product();
        let num_bytes = if dtype.is_packed() {
            (num_elements + dtype.pack_factor() - 1) / dtype.pack_factor()
        } else {
            num_elements * dtype.size_bytes()
        };

        let data: CudaSlice<u8> = device
            .alloc(num_bytes)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        let strides = compute_strides(&shape);

        Ok(Self {
            buffer: Arc::new(SharedBuffer { data }),
            shape,
            dtype,
            device,
            strides,
            offset: 0,
            logical_size: num_bytes,
        })
    }

    /// Get tensor shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor data type.
    #[inline]
    pub fn dtype(&self) -> GpuDType {
        self.dtype
    }

    /// Get number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes (logical size for views).
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.logical_size
    }

    /// Get strides (in elements).
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get raw device pointer (includes offset for views).
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        *self.buffer.data.device_ptr() + self.offset as u64
    }

    /// Get base device pointer (without offset).
    #[inline]
    pub fn base_device_ptr(&self) -> u64 {
        *self.buffer.data.device_ptr()
    }

    /// Get byte offset from base pointer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Check if this is a view (has non-zero offset or doesn't own full buffer).
    #[inline]
    pub fn is_view(&self) -> bool {
        self.offset > 0 || self.logical_size < self.buffer.data.len()
    }

    /// Get reference to underlying CUDA slice.
    #[inline]
    pub fn as_slice(&self) -> &CudaSlice<u8> {
        &self.buffer.data
    }

    /// Get CUDA device.
    #[inline]
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get underlying buffer length.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.data.len()
    }

    /// Reshape tensor (ZERO-COPY view, just metadata change).
    ///
    /// Returns a view with new shape that shares the same underlying GPU memory.
    /// This is a major optimization - no device-to-device copy needed.
    ///
    /// Returns error if new shape has different total elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<GpuTensor, InferenceError> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(InferenceError::Shape {
                expected: format!("{} elements", old_numel),
                got: format!("{} elements in new shape {:?}", new_numel, new_shape),
            });
        }

        // ZERO-COPY: Return view with same data pointer, just new shape
        // Arc::clone just increments reference count, no data copy
        Ok(GpuTensor {
            buffer: Arc::clone(&self.buffer),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&new_shape),
            offset: self.offset,
            logical_size: self.logical_size,
        })
    }

    /// Reshape tensor with deep copy (use when you need independent data).
    ///
    /// This allocates new memory and copies data synchronously.
    pub fn reshape_copy(&self, new_shape: Vec<usize>) -> Result<GpuTensor, InferenceError> {
        let old_numel: usize = self.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(InferenceError::Shape {
                expected: format!("{} elements", old_numel),
                got: format!("{} elements in new shape {:?}", new_numel, new_shape),
            });
        }

        // Deep copy for when independence is needed
        let new_data: CudaSlice<u8> = self.device
            .alloc_zeros(self.logical_size)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(
                *new_data.device_ptr(),
                self.device_ptr(), // Uses offset-adjusted pointer
                self.logical_size,
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }

        Ok(GpuTensor {
            buffer: Arc::new(SharedBuffer { data: new_data }),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&new_shape),
            offset: 0,
            logical_size: self.logical_size,
        })
    }

    /// Create a ZERO-COPY view into a contiguous slice of the first dimension.
    ///
    /// For tensor [A, B, C], slice(start, end) gives [end-start, B, C].
    /// This shares the same underlying GPU memory - no data copy needed.
    pub fn slice_dim0(&self, start: usize, end: usize) -> Result<GpuTensor, InferenceError> {
        if start >= end || end > self.shape[0] {
            return Err(InferenceError::Shape {
                expected: format!("valid range within 0..{}", self.shape[0]),
                got: format!("{}..{}", start, end),
            });
        }

        let elements_per_row: usize = if self.shape.len() > 1 {
            self.shape[1..].iter().product()
        } else {
            1
        };
        let bytes_per_row = elements_per_row * self.dtype.size_bytes();

        let slice_offset = start * bytes_per_row;
        let slice_length = (end - start) * bytes_per_row;

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        // ZERO-COPY: Just adjust offset, share same buffer
        Ok(GpuTensor {
            buffer: Arc::clone(&self.buffer),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&new_shape),
            offset: self.offset + slice_offset,
            logical_size: slice_length,
        })
    }

    /// Create a slice with deep copy (use when you need independent data).
    pub fn slice_dim0_copy(&self, start: usize, end: usize) -> Result<GpuTensor, InferenceError> {
        if start >= end || end > self.shape[0] {
            return Err(InferenceError::Shape {
                expected: format!("valid range within 0..{}", self.shape[0]),
                got: format!("{}..{}", start, end),
            });
        }

        let elements_per_row: usize = if self.shape.len() > 1 {
            self.shape[1..].iter().product()
        } else {
            1
        };
        let bytes_per_row = elements_per_row * self.dtype.size_bytes();

        let slice_offset = start * bytes_per_row;
        let slice_length = (end - start) * bytes_per_row;

        // Allocate new buffer and copy data
        let new_data: CudaSlice<u8> = self.device
            .alloc_zeros(slice_length)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        // Copy from source offset to new buffer
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(
                *new_data.device_ptr(),
                self.device_ptr() + slice_offset as u64,
                slice_length,
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        Ok(GpuTensor {
            buffer: Arc::new(SharedBuffer { data: new_data }),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&new_shape),
            offset: 0,
            logical_size: slice_length,
        })
    }

    /// Copy data from host to this tensor.
    ///
    /// For views with non-zero offset, copies to the view's region.
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), InferenceError> {
        if data.len() != self.logical_size {
            return Err(InferenceError::Shape {
                expected: format!("{} bytes", self.logical_size),
                got: format!("{} bytes", data.len()),
            });
        }

        // For views with offset, we need to copy at the right position
        if self.offset == 0 && self.logical_size == self.buffer.data.len() {
            // Not a view, can use the fast path
            // SAFETY: Need mutable access, use Arc::get_mut or make_mut
            // Since views share the buffer, we need to handle this carefully
            // For now, just use raw CUDA memcpy
            unsafe {
                cudarc::driver::result::memcpy_htod_sync(
                    self.device_ptr(),
                    data,
                ).map_err(|e| InferenceError::Memory(e.to_string()))?;
            }
        } else {
            // View case: copy to offset position
            unsafe {
                cudarc::driver::result::memcpy_htod_sync(
                    self.device_ptr(),
                    data,
                ).map_err(|e| InferenceError::Memory(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Copy data from this tensor to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), InferenceError> {
        if dst.len() < self.logical_size {
            return Err(InferenceError::Shape {
                expected: format!(">= {} bytes", self.logical_size),
                got: format!("{} bytes", dst.len()),
            });
        }

        // Use offset-adjusted pointer for views
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(
                dst,
                self.device_ptr(),
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }
        Ok(())
    }

    /// Copy data from this tensor to a new host Vec.
    pub fn to_host(&self) -> Result<Vec<u8>, InferenceError> {
        let mut dst = vec![0u8; self.logical_size];
        self.copy_to_host(&mut dst)?;
        Ok(dst)
    }

    /// Copy to a new tensor (deep copy).
    pub fn clone_tensor(&self) -> Result<GpuTensor, InferenceError> {
        // Allocate new buffer for the logical size (not full buffer if we're a view)
        let new_data: CudaSlice<u8> = self.device
            .alloc_zeros(self.logical_size)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        // Copy from our offset-adjusted pointer
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(
                *new_data.device_ptr(),
                self.device_ptr(), // Uses offset
                self.logical_size,
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }

        Ok(GpuTensor {
            buffer: Arc::new(SharedBuffer { data: new_data }),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: self.strides.clone(),
            offset: 0,
            logical_size: self.logical_size,
        })
    }

    /// Copy data from source tensor into this tensor at specified dimension 0 offset.
    ///
    /// For a target [A, B, C] and source [D, B, C], this copies source into
    /// target[offset:offset+D, :, :].
    ///
    /// Used for KV cache updates.
    pub fn copy_from_at_dim0(&mut self, source: &GpuTensor, dim0_offset: usize) -> Result<(), InferenceError> {
        // Validate shapes match (except dim 0)
        if self.shape[1..] != source.shape[1..] {
            return Err(InferenceError::Shape {
                expected: format!("matching trailing dims {:?}", &self.shape[1..]),
                got: format!("{:?}", &source.shape[1..]),
            });
        }

        let source_rows = source.shape[0];
        if dim0_offset + source_rows > self.shape[0] {
            return Err(InferenceError::Shape {
                expected: format!("offset + rows <= {}", self.shape[0]),
                got: format!("{} + {} = {}", dim0_offset, source_rows, dim0_offset + source_rows),
            });
        }

        let elements_per_row: usize = self.shape[1..].iter().product();
        let bytes_per_row = elements_per_row * self.dtype.size_bytes();
        let dest_byte_offset = dim0_offset * bytes_per_row;
        let copy_size = source_rows * bytes_per_row;

        // GPU-to-GPU copy using offset-adjusted pointers
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(
                self.device_ptr() + dest_byte_offset as u64,
                source.device_ptr(),
                copy_size,
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }

        Ok(())
    }

    // ============== ASYNC COPY METHODS ==============

    /// Async copy from host to device (non-blocking).
    ///
    /// The `stream` parameter is the raw CUDA stream pointer.
    /// Caller must ensure the source data remains valid until the stream completes.
    ///
    /// # Safety
    ///
    /// The source `data` slice must remain valid until the stream synchronizes.
    pub unsafe fn copy_from_host_async(
        &mut self,
        data: &[u8],
        stream: *mut c_void,
    ) -> Result<(), InferenceError> {
        if data.len() != self.logical_size {
            return Err(InferenceError::Shape {
                expected: format!("{} bytes", self.logical_size),
                got: format!("{} bytes", data.len()),
            });
        }

        let status = cudaMemcpyAsync(
            self.device_ptr(),
            data.as_ptr() as *const c_void,
            self.logical_size,
            CudaMemcpyKind::HostToDevice as i32,
            stream,
        );

        if status != 0 {
            return Err(InferenceError::Memory(format!(
                "cudaMemcpyAsync H2D failed with status {}",
                status
            )));
        }

        Ok(())
    }

    /// Async copy from device to host (non-blocking).
    ///
    /// # Safety
    ///
    /// The destination `dst` slice must remain valid until the stream synchronizes.
    pub unsafe fn copy_to_host_async(
        &self,
        dst: &mut [u8],
        stream: *mut c_void,
    ) -> Result<(), InferenceError> {
        if dst.len() < self.logical_size {
            return Err(InferenceError::Shape {
                expected: format!(">= {} bytes", self.logical_size),
                got: format!("{} bytes", dst.len()),
            });
        }

        let status = cudaMemcpyAsync(
            dst.as_mut_ptr() as u64,
            self.device_ptr() as *const c_void,
            self.logical_size,
            CudaMemcpyKind::DeviceToHost as i32,
            stream,
        );

        if status != 0 {
            return Err(InferenceError::Memory(format!(
                "cudaMemcpyAsync D2H failed with status {}",
                status
            )));
        }

        Ok(())
    }

    /// Async copy from source tensor at dimension 0 offset (non-blocking).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the destination region.
    pub unsafe fn copy_from_at_dim0_async(
        &mut self,
        source: &GpuTensor,
        dim0_offset: usize,
        stream: *mut c_void,
    ) -> Result<(), InferenceError> {
        // Validate shapes match (except dim 0)
        if self.shape[1..] != source.shape[1..] {
            return Err(InferenceError::Shape {
                expected: format!("matching trailing dims {:?}", &self.shape[1..]),
                got: format!("{:?}", &source.shape[1..]),
            });
        }

        let source_rows = source.shape[0];
        if dim0_offset + source_rows > self.shape[0] {
            return Err(InferenceError::Shape {
                expected: format!("offset + rows <= {}", self.shape[0]),
                got: format!("{} + {} = {}", dim0_offset, source_rows, dim0_offset + source_rows),
            });
        }

        let elements_per_row: usize = self.shape[1..].iter().product();
        let bytes_per_row = elements_per_row * self.dtype.size_bytes();
        let dest_byte_offset = dim0_offset * bytes_per_row;
        let copy_size = source_rows * bytes_per_row;

        // Async GPU-to-GPU copy
        let status = cudaMemcpyAsync(
            self.device_ptr() + dest_byte_offset as u64,
            source.device_ptr() as *const c_void,
            copy_size,
            CudaMemcpyKind::DeviceToDevice as i32,
            stream,
        );

        if status != 0 {
            return Err(InferenceError::Memory(format!(
                "cudaMemcpyAsync D2D failed with status {}",
                status
            )));
        }

        Ok(())
    }

    /// Async write to layer at position (non-blocking).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the destination region.
    pub unsafe fn write_layer_at_async(
        &mut self,
        layer_idx: usize,
        seq_offset: usize,
        source: &GpuTensor,
        stream: *mut c_void,
    ) -> Result<(), InferenceError> {
        if self.shape.len() != 4 {
            return Err(InferenceError::Shape {
                expected: "4D target tensor".to_string(),
                got: format!("{:?}", self.shape),
            });
        }

        if source.shape.len() != 3 {
            return Err(InferenceError::Shape {
                expected: "3D source tensor [seq, kv_heads, head_dim]".to_string(),
                got: format!("{:?}", source.shape),
            });
        }

        let num_layers = self.shape[0];
        let max_seq = self.shape[1];
        let kv_heads = self.shape[2];
        let head_dim = self.shape[3];

        let source_seq = source.shape[0];
        let source_kv_heads = source.shape[1];
        let source_head_dim = source.shape[2];

        if layer_idx >= num_layers {
            return Err(InferenceError::Shape {
                expected: format!("layer_idx < {}", num_layers),
                got: format!("{}", layer_idx),
            });
        }

        if source_kv_heads != kv_heads || source_head_dim != head_dim {
            return Err(InferenceError::Shape {
                expected: format!("[*, {}, {}]", kv_heads, head_dim),
                got: format!("[{}, {}, {}]", source_seq, source_kv_heads, source_head_dim),
            });
        }

        if seq_offset + source_seq > max_seq {
            return Err(InferenceError::Shape {
                expected: format!("seq_offset + seq <= {}", max_seq),
                got: format!("{} + {} = {}", seq_offset, source_seq, seq_offset + source_seq),
            });
        }

        // Calculate byte offsets
        let seq_stride = kv_heads * head_dim * self.dtype.size_bytes();
        let layer_stride = max_seq * seq_stride;
        let dest_byte_offset = layer_idx * layer_stride + seq_offset * seq_stride;
        let copy_size = source_seq * seq_stride;

        // Async GPU-to-GPU copy
        let status = cudaMemcpyAsync(
            self.device_ptr() + dest_byte_offset as u64,
            source.device_ptr() as *const c_void,
            copy_size,
            CudaMemcpyKind::DeviceToDevice as i32,
            stream,
        );

        if status != 0 {
            return Err(InferenceError::Memory(format!(
                "cudaMemcpyAsync D2D (write_layer) failed with status {}",
                status
            )));
        }

        Ok(())
    }

    /// Create a ZERO-COPY view into a layer of a 4D tensor.
    ///
    /// For tensor [num_layers, max_seq, kv_heads, head_dim], returns a view
    /// of shape [max_seq, kv_heads, head_dim] for the specified layer.
    ///
    /// This shares the same underlying GPU memory - no data copy needed.
    ///
    /// Used for getting per-layer KV cache views.
    pub fn slice_layer(&self, layer_idx: usize) -> Result<GpuTensor, InferenceError> {
        if self.shape.len() != 4 {
            return Err(InferenceError::Shape {
                expected: "4D tensor".to_string(),
                got: format!("{:?}", self.shape),
            });
        }

        let num_layers = self.shape[0];
        if layer_idx >= num_layers {
            return Err(InferenceError::Shape {
                expected: format!("layer_idx < {}", num_layers),
                got: format!("{}", layer_idx),
            });
        }

        let max_seq = self.shape[1];
        let kv_heads = self.shape[2];
        let head_dim = self.shape[3];

        // Layer stride in bytes
        let layer_elements = max_seq * kv_heads * head_dim;
        let layer_bytes = layer_elements * self.dtype.size_bytes();
        let layer_offset = layer_idx * layer_bytes;

        // Return with 3D shape [max_seq, kv_heads, head_dim]
        let new_shape = vec![max_seq, kv_heads, head_dim];

        // ZERO-COPY: Just adjust offset, share same buffer
        Ok(GpuTensor {
            buffer: Arc::clone(&self.buffer),
            shape: new_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&new_shape),
            offset: self.offset + layer_offset,
            logical_size: layer_bytes,
        })
    }

    /// Write a slice into a layer of a 4D tensor.
    ///
    /// For target [num_layers, max_seq, kv_heads, head_dim] and source [seq, kv_heads, head_dim],
    /// copies source into target[layer_idx, seq_offset:seq_offset+seq, :, :].
    pub fn write_layer_at(&mut self, layer_idx: usize, seq_offset: usize, source: &GpuTensor) -> Result<(), InferenceError> {
        if self.shape.len() != 4 {
            return Err(InferenceError::Shape {
                expected: "4D target tensor".to_string(),
                got: format!("{:?}", self.shape),
            });
        }

        if source.shape.len() != 3 {
            return Err(InferenceError::Shape {
                expected: "3D source tensor [seq, kv_heads, head_dim]".to_string(),
                got: format!("{:?}", source.shape),
            });
        }

        let num_layers = self.shape[0];
        let max_seq = self.shape[1];
        let kv_heads = self.shape[2];
        let head_dim = self.shape[3];

        let source_seq = source.shape[0];
        let source_kv_heads = source.shape[1];
        let source_head_dim = source.shape[2];

        if layer_idx >= num_layers {
            return Err(InferenceError::Shape {
                expected: format!("layer_idx < {}", num_layers),
                got: format!("{}", layer_idx),
            });
        }

        if source_kv_heads != kv_heads || source_head_dim != head_dim {
            return Err(InferenceError::Shape {
                expected: format!("[*, {}, {}]", kv_heads, head_dim),
                got: format!("[{}, {}, {}]", source_seq, source_kv_heads, source_head_dim),
            });
        }

        if seq_offset + source_seq > max_seq {
            return Err(InferenceError::Shape {
                expected: format!("seq_offset + seq <= {}", max_seq),
                got: format!("{} + {} = {}", seq_offset, source_seq, seq_offset + source_seq),
            });
        }

        // Calculate byte offsets
        let seq_stride = kv_heads * head_dim * self.dtype.size_bytes();
        let layer_stride = max_seq * seq_stride;

        let dest_byte_offset = layer_idx * layer_stride + seq_offset * seq_stride;
        let copy_size = source_seq * seq_stride;

        // GPU-to-GPU copy using offset-adjusted pointers
        unsafe {
            cudarc::driver::result::memcpy_dtod_sync(
                self.device_ptr() + dest_byte_offset as u64,
                source.device_ptr(),
                copy_size,
            ).map_err(|e| InferenceError::Memory(e.to_string()))?;
        }

        Ok(())
    }

    /// Get a ZERO-COPY view of a 4D tensor for a specific layer and sequence range.
    ///
    /// For tensor [num_layers, max_seq, kv_heads, head_dim],
    /// returns a view of [seq_len, kv_heads, head_dim] for the specified layer.
    ///
    /// Note: This returns a 3D view. Caller should reshape for Flash Attention.
    pub fn get_layer_kv_slice(&self, layer_idx: usize, seq_len: usize) -> Result<GpuTensor, InferenceError> {
        if self.shape.len() != 4 {
            return Err(InferenceError::Shape {
                expected: "4D tensor".to_string(),
                got: format!("{:?}", self.shape),
            });
        }

        let num_layers = self.shape[0];
        let max_seq = self.shape[1];
        let kv_heads = self.shape[2];
        let head_dim = self.shape[3];

        if layer_idx >= num_layers {
            return Err(InferenceError::Shape {
                expected: format!("layer_idx < {}", num_layers),
                got: format!("{}", layer_idx),
            });
        }

        let effective_seq = seq_len.min(max_seq);
        if effective_seq == 0 {
            return Err(InferenceError::Shape {
                expected: "seq_len > 0".to_string(),
                got: "0".to_string(),
            });
        }

        // Calculate offsets
        let seq_stride = kv_heads * head_dim * self.dtype.size_bytes();
        let layer_stride = max_seq * seq_stride;
        let layer_offset = layer_idx * layer_stride;
        let slice_bytes = effective_seq * seq_stride;

        // Output shape: [seq_len, kv_heads, head_dim] - caller can reshape to [1, seq, heads, dim]
        let output_shape = vec![effective_seq, kv_heads, head_dim];

        // ZERO-COPY: Just adjust offset, share same buffer
        Ok(GpuTensor {
            buffer: Arc::clone(&self.buffer),
            shape: output_shape.clone(),
            dtype: self.dtype,
            device: Arc::clone(&self.device),
            strides: compute_strides(&output_shape),
            offset: self.offset + layer_offset,
            logical_size: slice_bytes,
        })
    }
}

/// Compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Typed view into a GpuTensor for kernel launches.
///
/// This provides a typed pointer for CUDA kernels while maintaining
/// the underlying tensor's lifetime.
pub struct GpuTensorView<'a, T> {
    ptr: u64,
    numel: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T> GpuTensorView<'a, T> {
    /// Create a typed view from a tensor.
    ///
    /// # Safety
    ///
    /// Caller must ensure T matches the tensor's dtype.
    pub unsafe fn from_tensor(tensor: &'a GpuTensor) -> Self {
        Self {
            ptr: tensor.device_ptr(),
            numel: tensor.numel(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Get device pointer as the typed pointer value.
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.numel
    }

    /// Whether empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.numel == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[10]), vec![1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(GpuDType::F16.size_bytes(), 2);
        assert_eq!(GpuDType::F32.size_bytes(), 4);
        assert_eq!(GpuDType::I4.pack_factor(), 2);
    }
}
