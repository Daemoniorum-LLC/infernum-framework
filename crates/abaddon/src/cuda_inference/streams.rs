//! Multi-stream execution for overlapping compute and memory operations.
//!
//! CUDA streams allow concurrent execution of operations that don't have
//! data dependencies. This module provides infrastructure for:
//!
//! - Overlapping weight loading with previous layer computation
//! - Pipelining attention across multiple heads
//! - Overlapping KV cache updates with computation
//!
//! ## Stream Layout
//!
//! ```text
//! Stream 0 (Compute):  [Layer N attn] [Layer N MLP] [Layer N+1 attn] ...
//! Stream 1 (Memory):   [Load L+1 wts] [KV update]   [Load L+2 wts]   ...
//! Stream 2 (Prefetch): [Prefetch N+2] [Prefetch N+3] ...
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::InferenceError;

// CUDA runtime FFI bindings (cudart library)
#[link(name = "cudart")]
extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamCreateWithPriority(stream: *mut *mut c_void, flags: u32, priority: i32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

/// Stream creation flags.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum StreamFlags {
    /// Default stream behavior.
    Default = 0x00,
    /// Non-blocking: stream doesn't synchronize with default stream.
    NonBlocking = 0x01,
}

/// Event creation flags.
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum EventFlags {
    /// Default event.
    Default = 0x00,
    /// Disable timing to reduce overhead.
    DisableTiming = 0x02,
    /// Interprocess event (not needed for single-process).
    Interprocess = 0x04,
}

/// CUDA event for stream synchronization.
pub struct CudaEvent {
    event: *mut c_void,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Create a new CUDA event.
    pub fn new() -> Result<Self, InferenceError> {
        let mut event: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cudaEventCreate(&mut event) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaEventCreate failed with status {}",
                status
            )));
        }
        Ok(Self { event })
    }

    /// Create a CUDA event with flags.
    pub fn with_flags(flags: EventFlags) -> Result<Self, InferenceError> {
        let mut event: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cudaEventCreateWithFlags(&mut event, flags as u32) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaEventCreateWithFlags failed with status {}",
                status
            )));
        }
        Ok(Self { event })
    }

    /// Record the event on a stream.
    pub fn record(&self, stream: &CudaStream) -> Result<(), InferenceError> {
        let status = unsafe { cudaEventRecord(self.event, stream.stream) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaEventRecord failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Wait for the event to complete (CPU blocking).
    pub fn synchronize(&self) -> Result<(), InferenceError> {
        let status = unsafe { cudaEventSynchronize(self.event) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaEventSynchronize failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Get the raw event pointer.
    pub unsafe fn raw(&self) -> *mut c_void {
        self.event
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe { cudaEventDestroy(self.event) };
        }
    }
}

/// CUDA stream for asynchronous execution.
pub struct CudaStream {
    stream: *mut c_void,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self, InferenceError> {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cudaStreamCreate(&mut stream) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaStreamCreate failed with status {}",
                status
            )));
        }
        Ok(Self { stream })
    }

    /// Create a stream with flags.
    pub fn with_flags(flags: StreamFlags) -> Result<Self, InferenceError> {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cudaStreamCreateWithFlags(&mut stream, flags as u32) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaStreamCreateWithFlags failed with status {}",
                status
            )));
        }
        Ok(Self { stream })
    }

    /// Create a stream with priority (lower = higher priority).
    pub fn with_priority(priority: i32) -> Result<Self, InferenceError> {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let status = unsafe {
            cudaStreamCreateWithPriority(&mut stream, StreamFlags::NonBlocking as u32, priority)
        };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaStreamCreateWithPriority failed with status {}",
                status
            )));
        }
        Ok(Self { stream })
    }

    /// Synchronize the stream (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<(), InferenceError> {
        let status = unsafe { cudaStreamSynchronize(self.stream) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaStreamSynchronize failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Wait for an event from another stream.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), InferenceError> {
        let status = unsafe { cudaStreamWaitEvent(self.stream, event.event, 0) };
        if status != 0 {
            return Err(InferenceError::Kernel(format!(
                "cudaStreamWaitEvent failed with status {}",
                status
            )));
        }
        Ok(())
    }

    /// Get the raw stream pointer.
    pub unsafe fn raw(&self) -> *mut c_void {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { cudaStreamDestroy(self.stream) };
        }
    }
}

/// Stream manager for multi-stream inference.
///
/// Manages a pool of streams for overlapping compute and memory operations.
pub struct StreamManager {
    /// Primary compute stream (high priority).
    compute_stream: CudaStream,

    /// Memory transfer stream.
    memory_stream: CudaStream,

    /// Prefetch stream for weight loading.
    prefetch_stream: CudaStream,

    /// Events for inter-stream synchronization.
    compute_done: CudaEvent,
    memory_done: CudaEvent,

    /// CUDA device reference.
    device: Arc<CudaDevice>,
}

impl StreamManager {
    /// Create a new stream manager.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, InferenceError> {
        // Create streams with priorities
        // Priority 0 = highest, larger numbers = lower priority
        let compute_stream = CudaStream::with_priority(0)?; // Highest priority
        let memory_stream = CudaStream::with_priority(1)?;
        let prefetch_stream = CudaStream::with_priority(2)?;

        // Create events for synchronization (disable timing for lower overhead)
        let compute_done = CudaEvent::with_flags(EventFlags::DisableTiming)?;
        let memory_done = CudaEvent::with_flags(EventFlags::DisableTiming)?;

        Ok(Self {
            compute_stream,
            memory_stream,
            prefetch_stream,
            compute_done,
            memory_done,
            device,
        })
    }

    /// Get the compute stream.
    pub fn compute(&self) -> &CudaStream {
        &self.compute_stream
    }

    /// Get the memory stream.
    pub fn memory(&self) -> &CudaStream {
        &self.memory_stream
    }

    /// Get the prefetch stream.
    pub fn prefetch(&self) -> &CudaStream {
        &self.prefetch_stream
    }

    /// Record that compute is done on compute stream.
    pub fn record_compute_done(&self) -> Result<(), InferenceError> {
        self.compute_done.record(&self.compute_stream)
    }

    /// Record that memory ops are done on memory stream.
    pub fn record_memory_done(&self) -> Result<(), InferenceError> {
        self.memory_done.record(&self.memory_stream)
    }

    /// Make memory stream wait for compute to finish.
    pub fn memory_wait_compute(&self) -> Result<(), InferenceError> {
        self.memory_stream.wait_event(&self.compute_done)
    }

    /// Make compute stream wait for memory to finish.
    pub fn compute_wait_memory(&self) -> Result<(), InferenceError> {
        self.compute_stream.wait_event(&self.memory_done)
    }

    /// Synchronize all streams.
    pub fn synchronize_all(&self) -> Result<(), InferenceError> {
        self.compute_stream.synchronize()?;
        self.memory_stream.synchronize()?;
        self.prefetch_stream.synchronize()?;
        Ok(())
    }

    /// Synchronize just the compute stream.
    pub fn synchronize_compute(&self) -> Result<(), InferenceError> {
        self.compute_stream.synchronize()
    }

    /// Get raw compute stream pointer for CUDA operations.
    pub unsafe fn compute_raw(&self) -> *mut c_void {
        self.compute_stream.raw()
    }

    /// Get raw memory stream pointer for CUDA operations.
    pub unsafe fn memory_raw(&self) -> *mut c_void {
        self.memory_stream.raw()
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Synchronize all CUDA devices (device-wide barrier).
pub fn device_synchronize() -> Result<(), InferenceError> {
    let status = unsafe { cudaDeviceSynchronize() };
    if status != 0 {
        return Err(InferenceError::Kernel(format!(
            "cudaDeviceSynchronize failed with status {}",
            status
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_manager_types() {
        // Test that types are correct (compilation test)
        assert_eq!(StreamFlags::Default as u32, 0);
        assert_eq!(StreamFlags::NonBlocking as u32, 1);
        assert_eq!(EventFlags::DisableTiming as u32, 2);
    }
}
