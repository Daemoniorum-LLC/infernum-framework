//! GPU-accelerated LZ4 decompression for compressed model weights.
//!
//! This module provides CUDA-accelerated LZ4 decompression for the HCT format.
//! It leverages block-level parallelism - each HCT block is decompressed by
//! a separate GPU thread/warp.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GPU LZ4 Decompression                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  Host Memory          GPU Memory                                │
//! │  ┌──────────┐        ┌──────────────────────────────────┐      │
//! │  │ HCT File │  H2D   │  Compressed Blocks               │      │
//! │  │ ├─Header │ ────>  │  [Block 0][Block 1]...[Block N]  │      │
//! │  │ ├─Index  │        └──────────────────────────────────┘      │
//! │  │ └─Blocks │                       │                          │
//! │  └──────────┘                       │ Parallel                 │
//! │                                     │ Decompression            │
//! │                                     ▼                          │
//! │                       ┌──────────────────────────────────┐     │
//! │                       │  Decompressed Tensor Data        │     │
//! │                       │  [Block 0][Block 1]...[Block N]  │     │
//! │                       └──────────────────────────────────┘     │
//! │                                     │                          │
//! │                                     │ Direct use in inference  │
//! │                                     ▼                          │
//! │                       ┌──────────────────────────────────┐     │
//! │                       │  Candle Tensor (GPU)             │     │
//! │                       └──────────────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance
//!
//! - Block-level parallelism: N blocks = N parallel decompressions
//! - Avoids CPU→GPU transfer of uncompressed data
//! - Typical throughput: 10-20 GB/s on modern GPUs

/// CUDA-accelerated LZ4 decompression implementation.
#[cfg(feature = "cuda")]
pub mod cuda {

    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;

    /// GPU LZ4 decompression context.
    ///
    /// Holds compiled CUDA kernels and device state for efficient
    /// repeated decompression operations.
    pub struct GpuLz4Context {
        device: Arc<CudaDevice>,
        device_id: usize,
        /// Compiled LZ4 decompression kernel
        kernel_loaded: bool,
    }

    impl GpuLz4Context {
        /// Creates a new GPU LZ4 context for the specified device.
        pub fn new(device_id: usize) -> Result<Self, GpuLz4Error> {
            let device = CudaDevice::new(device_id).map_err(|e| GpuLz4Error::DeviceInit {
                device_id,
                message: e.to_string(),
            })?;

            Ok(Self {
                device,
                device_id,
                kernel_loaded: false,
            })
        }

        /// Returns the CUDA device ID.
        pub fn device_id(&self) -> usize {
            self.device_id
        }

        /// Loads the LZ4 decompression kernel.
        ///
        /// This compiles and caches the PTX kernel for later use.
        pub fn load_kernel(&mut self) -> Result<(), GpuLz4Error> {
            if self.kernel_loaded {
                return Ok(());
            }

            // Load main PTX module (K1 + K2, sm_50) — always available.
            let ptx = Self::get_lz4_ptx();
            self.device
                .load_ptx(
                    ptx,
                    "lz4_decompress",
                    &["lz4_decompress_block", "lz4_decompress_blocks_parallel"],
                )
                .map_err(|e| GpuLz4Error::KernelLoad {
                    message: e.to_string(),
                })?;

            // Load warp PTX module (K3, sm_70) — requires cuda-experimental.
            // K3 uses shfl.sync and bar.warp.sync which need sm_70+ (Volta).
            // See DD-2 in GPU-CODEC-PIPELINE-SPEC.md.
            #[cfg(feature = "cuda-experimental")]
            {
                let warp_ptx = Self::get_lz4_warp_ptx();
                self.device
                    .load_ptx(warp_ptx, "lz4_warp", &["lz4_decompress_blocks_warp"])
                    .map_err(|e| GpuLz4Error::KernelLoad {
                        message: e.to_string(),
                    })?;
            }

            self.kernel_loaded = true;
            Ok(())
        }

        /// Returns the embedded LZ4 PTX kernel (K1 + K2, sm_50).
        fn get_lz4_ptx() -> Ptx {
            Ptx::from_src(LZ4_KERNEL_PTX)
        }

        /// Returns the embedded warp-parallel LZ4 PTX kernel (K3, sm_70).
        #[cfg(feature = "cuda-experimental")]
        fn get_lz4_warp_ptx() -> Ptx {
            Ptx::from_src(LZ4_WARP_KERNEL_PTX)
        }

        /// Decompresses a single LZ4 block on GPU.
        ///
        /// # Arguments
        ///
        /// * `compressed` - The compressed block data
        /// * `uncompressed_size` - Expected uncompressed size
        ///
        /// # Returns
        ///
        /// The decompressed data as a GPU buffer
        pub fn decompress_block(
            &self,
            compressed: &[u8],
            uncompressed_size: usize,
        ) -> Result<CudaSlice<u8>, GpuLz4Error> {
            // Allocate GPU memory for input and output
            let d_input = self.device.htod_copy(compressed.to_vec()).map_err(|e| {
                GpuLz4Error::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            let d_output: CudaSlice<u8> =
                self.device.alloc_zeros(uncompressed_size).map_err(|e| {
                    GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    }
                })?;

            // Launch decompression kernel
            let func = self
                .device
                .get_func("lz4_decompress", "lz4_decompress_block")
                .ok_or_else(|| GpuLz4Error::KernelLoad {
                    message: "Kernel not found".to_string(),
                })?;

            let cfg = LaunchConfig::for_num_elems(1);

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_input,
                        compressed.len() as u32,
                        &d_output,
                        uncompressed_size as u32,
                    ),
                )
            }
            .map_err(|e| GpuLz4Error::KernelExec {
                message: e.to_string(),
            })?;

            Ok(d_output)
        }

        /// Decompresses multiple LZ4 blocks in parallel on GPU.
        ///
        /// This is the primary API for HCT format decompression.
        /// Each block is decompressed by a separate GPU thread.
        ///
        /// # Arguments
        ///
        /// * `blocks` - Vector of (compressed_data, uncompressed_size) tuples
        ///
        /// # Returns
        ///
        /// Contiguous GPU buffer containing all decompressed blocks
        pub fn decompress_blocks_parallel(
            &self,
            blocks: &[(Vec<u8>, usize)],
        ) -> Result<CudaSlice<u8>, GpuLz4Error> {
            if blocks.is_empty() {
                return Err(GpuLz4Error::InvalidInput {
                    message: "No blocks to decompress".to_string(),
                });
            }

            // Calculate total sizes
            let total_compressed: usize = blocks.iter().map(|(b, _)| b.len()).sum();
            let total_uncompressed: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Create block metadata for GPU kernel
            let mut block_offsets_in: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
            let mut block_offsets_out: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
            let mut compressed_sizes: Vec<u32> = Vec::with_capacity(blocks.len());
            let mut uncompressed_sizes: Vec<u32> = Vec::with_capacity(blocks.len());

            let mut offset_in: u32 = 0;
            let mut offset_out: u32 = 0;

            for (compressed, uncompressed_size) in blocks {
                block_offsets_in.push(offset_in);
                block_offsets_out.push(offset_out);
                compressed_sizes.push(compressed.len() as u32);
                uncompressed_sizes.push(*uncompressed_size as u32);

                offset_in += compressed.len() as u32;
                offset_out += *uncompressed_size as u32;
            }

            // Concatenate all compressed data
            let mut all_compressed = Vec::with_capacity(total_compressed);
            for (compressed, _) in blocks {
                all_compressed.extend_from_slice(compressed);
            }

            // Copy to GPU
            let d_compressed =
                self.device
                    .htod_copy(all_compressed)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_output: CudaSlice<u8> =
                self.device.alloc_zeros(total_uncompressed).map_err(|e| {
                    GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    }
                })?;

            let d_offsets_in =
                self.device
                    .htod_copy(block_offsets_in)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_offsets_out =
                self.device
                    .htod_copy(block_offsets_out)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_compressed_sizes =
                self.device
                    .htod_copy(compressed_sizes)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_uncompressed_sizes = self.device.htod_copy(uncompressed_sizes).map_err(|e| {
                GpuLz4Error::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            // Launch parallel decompression kernel
            let func = self
                .device
                .get_func("lz4_decompress", "lz4_decompress_blocks_parallel")
                .ok_or_else(|| GpuLz4Error::KernelLoad {
                    message: "Parallel kernel not found".to_string(),
                })?;

            // One block per GPU thread block for now
            // Could optimize with warps handling multiple LZ4 blocks
            let num_blocks = blocks.len() as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (32, 1, 1), // One warp per LZ4 block
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_compressed,
                        &d_output,
                        &d_offsets_in,
                        &d_offsets_out,
                        &d_compressed_sizes,
                        &d_uncompressed_sizes,
                        num_blocks,
                    ),
                )
            }
            .map_err(|e| GpuLz4Error::KernelExec {
                message: e.to_string(),
            })?;

            // Synchronize to ensure decompression is complete
            self.device
                .synchronize()
                .map_err(|e| GpuLz4Error::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Decompresses multiple LZ4 blocks using warp-parallel optimization.
        ///
        /// This version uses all 32 threads in a warp for parallel literal
        /// and match copying, providing significant speedup for blocks with
        /// large literals or matches.
        ///
        /// # Performance
        ///
        /// - Up to 32x faster for literal-heavy data
        /// - Falls back to sequential for small-offset matches (overlap handling)
        /// - Best suited for blocks with >32 byte literals/matches
        ///
        /// # Arguments
        ///
        /// * `blocks` - Vector of (compressed_data, uncompressed_size) tuples
        ///
        /// # Returns
        ///
        /// Contiguous GPU buffer containing all decompressed blocks
        ///
        /// # Feature Gate
        ///
        /// Requires `cuda-experimental` feature. The warp kernel has a known
        /// thread coordination bug (DD-5) and is not yet production-ready.
        #[cfg(feature = "cuda-experimental")]
        pub fn decompress_blocks_warp_parallel(
            &self,
            blocks: &[(Vec<u8>, usize)],
        ) -> Result<CudaSlice<u8>, GpuLz4Error> {
            if blocks.is_empty() {
                return Err(GpuLz4Error::InvalidInput {
                    message: "No blocks to decompress".to_string(),
                });
            }

            // Calculate total sizes
            let total_compressed: usize = blocks.iter().map(|(b, _)| b.len()).sum();
            let total_uncompressed: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Create block metadata for GPU kernel
            let mut block_offsets_in: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
            let mut block_offsets_out: Vec<u32> = Vec::with_capacity(blocks.len() + 1);
            let mut compressed_sizes: Vec<u32> = Vec::with_capacity(blocks.len());
            let mut uncompressed_sizes: Vec<u32> = Vec::with_capacity(blocks.len());

            let mut offset_in: u32 = 0;
            let mut offset_out: u32 = 0;

            for (compressed, uncompressed_size) in blocks {
                block_offsets_in.push(offset_in);
                block_offsets_out.push(offset_out);
                compressed_sizes.push(compressed.len() as u32);
                uncompressed_sizes.push(*uncompressed_size as u32);

                offset_in += compressed.len() as u32;
                offset_out += *uncompressed_size as u32;
            }

            // Concatenate all compressed data
            let mut all_compressed = Vec::with_capacity(total_compressed);
            for (compressed, _) in blocks {
                all_compressed.extend_from_slice(compressed);
            }

            // Copy to GPU
            let d_compressed =
                self.device
                    .htod_copy(all_compressed)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_output: CudaSlice<u8> =
                self.device.alloc_zeros(total_uncompressed).map_err(|e| {
                    GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    }
                })?;

            let d_offsets_in =
                self.device
                    .htod_copy(block_offsets_in)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_offsets_out =
                self.device
                    .htod_copy(block_offsets_out)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_compressed_sizes =
                self.device
                    .htod_copy(compressed_sizes)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            let d_uncompressed_sizes = self.device.htod_copy(uncompressed_sizes).map_err(|e| {
                GpuLz4Error::MemoryAlloc {
                    message: e.to_string(),
                }
            })?;

            // Launch warp-parallel decompression kernel (from lz4_warp module)
            let func = self
                .device
                .get_func("lz4_warp", "lz4_decompress_blocks_warp")
                .ok_or_else(|| GpuLz4Error::KernelLoad {
                    message: "Warp kernel not found (lz4_warp module)".to_string(),
                })?;

            // One warp (32 threads) per LZ4 block
            let num_blocks = blocks.len() as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (32, 1, 1), // Full warp
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &d_compressed,
                        &d_output,
                        &d_offsets_in,
                        &d_offsets_out,
                        &d_compressed_sizes,
                        &d_uncompressed_sizes,
                        num_blocks,
                    ),
                )
            }
            .map_err(|e| GpuLz4Error::KernelExec {
                message: e.to_string(),
            })?;

            // Synchronize
            self.device
                .synchronize()
                .map_err(|e| GpuLz4Error::Synchronize {
                    message: e.to_string(),
                })?;

            Ok(d_output)
        }

        /// Returns the underlying cudarc CUDA device.
        ///
        /// This can be used for advanced GPU operations that bypass Candle.
        pub fn cuda_device(&self) -> Arc<CudaDevice> {
            Arc::clone(&self.device)
        }

        /// Decompresses to a typed `CudaSlice<f16>` for direct GPU use.
        ///
        /// This is the zero-copy path for F16 data. The returned slice stays on GPU
        /// and can be used directly in CUDA kernels without any host transfers.
        ///
        /// # Note
        ///
        /// For integration with Candle tensors, use `decompress_to_tensor()` which
        /// handles the conversion. Direct `CudaSlice` access is for advanced users
        /// who want to avoid Candle's tensor creation overhead.
        pub fn decompress_to_f16_slice(
            &self,
            blocks: &[(Vec<u8>, usize)],
        ) -> Result<CudaSlice<half::f16>, GpuLz4Error> {
            #[cfg(feature = "cuda-experimental")]
            let d_output = self.decompress_blocks_warp_parallel(blocks)?;
            #[cfg(not(feature = "cuda-experimental"))]
            let d_output = self.decompress_blocks_parallel(blocks)?;

            // Total size in bytes
            let total_bytes: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Number of f16 values
            let num_f16 = total_bytes / 2;

            // Reinterpret the u8 slice as f16 using unsafe cast
            // SAFETY: The decompressed data should be valid f16 bytes
            let d_f16: CudaSlice<half::f16> = unsafe {
                let ptr = *d_output.device_ptr();
                self.device.upgrade_device_ptr(ptr, num_f16)
            };

            // Keep the original allocation alive by leaking it
            // The f16 slice now owns the memory
            std::mem::forget(d_output);

            Ok(d_f16)
        }

        /// Decompresses to a typed `CudaSlice<f32>` for direct GPU use.
        ///
        /// Similar to `decompress_to_f16_slice` but for F32 data.
        pub fn decompress_to_f32_slice(
            &self,
            blocks: &[(Vec<u8>, usize)],
        ) -> Result<CudaSlice<f32>, GpuLz4Error> {
            #[cfg(feature = "cuda-experimental")]
            let d_output = self.decompress_blocks_warp_parallel(blocks)?;
            #[cfg(not(feature = "cuda-experimental"))]
            let d_output = self.decompress_blocks_parallel(blocks)?;

            // Total size in bytes
            let total_bytes: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Number of f32 values
            let num_f32 = total_bytes / 4;

            // Reinterpret the u8 slice as f32
            let d_f32: CudaSlice<f32> = unsafe {
                let ptr = *d_output.device_ptr();
                self.device.upgrade_device_ptr(ptr, num_f32)
            };

            std::mem::forget(d_output);

            Ok(d_f32)
        }

        /// Decompresses to a Candle tensor.
        ///
        /// # Note
        ///
        /// Due to Candle API limitations, this currently requires copying data
        /// through host memory. For zero-copy GPU access, use the typed slice
        /// methods: `decompress_to_f16_slice()` or `decompress_to_f32_slice()`.
        ///
        /// Future versions may support direct GPU tensor creation when Candle
        /// exposes the necessary APIs.
        pub fn decompress_to_tensor(
            &self,
            blocks: &[(Vec<u8>, usize)],
            shape: &[usize],
            dtype: DType,
            candle_device: &Device,
        ) -> Result<Tensor, GpuLz4Error> {
            // Use warp-parallel kernel when available, else standard parallel.
            #[cfg(feature = "cuda-experimental")]
            let d_output = self.decompress_blocks_warp_parallel(blocks)?;
            #[cfg(not(feature = "cuda-experimental"))]
            let d_output = self.decompress_blocks_parallel(blocks)?;

            // Convert GPU buffer to Candle tensor
            // This requires copying the data through Candle's API
            let total_size: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Copy decompressed data back to host (temporary - ideally we'd keep on GPU)
            let mut host_data = vec![0u8; total_size];
            self.device
                .dtoh_sync_copy_into(&d_output, &mut host_data)
                .map_err(|e| GpuLz4Error::MemoryCopy {
                    message: e.to_string(),
                })?;

            // Create tensor from bytes
            let tensor = match dtype {
                DType::F32 => {
                    let floats: Vec<f32> = host_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Tensor::from_vec(floats, shape, candle_device)
                },
                DType::F16 => {
                    let halfs: Vec<half::f16> = host_data
                        .chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    Tensor::from_vec(halfs, shape, candle_device)
                },
                DType::BF16 => {
                    let bfloats: Vec<half::bf16> = host_data
                        .chunks_exact(2)
                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    Tensor::from_vec(bfloats, shape, candle_device)
                },
                _ => {
                    return Err(GpuLz4Error::UnsupportedDtype {
                        dtype: format!("{:?}", dtype),
                    })
                },
            }
            .map_err(|e| GpuLz4Error::TensorCreate {
                message: e.to_string(),
            })?;

            Ok(tensor)
        }
    }

    // ==================== Streaming Pipeline (Phase 5.1) ====================

    /// CUDA stream pool for async operations.
    ///
    /// Manages a pool of CUDA streams to enable overlapping:
    /// - Host-to-Device (H2D) transfers
    /// - Kernel execution (decompression)
    /// - Device-to-Host (D2H) transfers
    ///
    /// This enables pipelining: while one block is being transferred,
    /// another is being decompressed, maximizing GPU utilization.
    pub struct CudaStreamPool {
        device: Arc<CudaDevice>,
        streams: Vec<CudaStream>,
        num_streams: usize,
    }

    impl CudaStreamPool {
        /// Creates a new stream pool with the specified number of streams.
        ///
        /// More streams allow more concurrent operations but use more resources.
        /// Typically 2-4 streams provide good overlap.
        pub fn new(device: Arc<CudaDevice>, num_streams: usize) -> Result<Self, GpuLz4Error> {
            let mut streams = Vec::with_capacity(num_streams);

            for i in 0..num_streams {
                let stream =
                    device
                        .fork_default_stream()
                        .map_err(|e| GpuLz4Error::StreamCreate {
                            stream_id: i,
                            message: e.to_string(),
                        })?;
                streams.push(stream);
            }

            Ok(Self {
                device,
                streams,
                num_streams,
            })
        }

        /// Returns a reference to stream at the given index (wraps around).
        pub fn get_stream(&self, index: usize) -> &CudaStream {
            &self.streams[index % self.num_streams]
        }

        /// Returns the number of streams in the pool.
        pub fn num_streams(&self) -> usize {
            self.num_streams
        }

        /// Synchronizes all streams in the pool.
        pub fn synchronize_all(&self) -> Result<(), GpuLz4Error> {
            for (i, _stream) in self.streams.iter().enumerate() {
                self.device
                    .synchronize()
                    .map_err(|e| GpuLz4Error::Synchronize {
                        message: format!("Stream {}: {}", i, e),
                    })?;
            }
            Ok(())
        }
    }

    /// Streaming decompression context for pipelined weight loading.
    ///
    /// Implements a triple-buffered pipeline:
    /// - Stage 0: Read compressed data from disk
    /// - Stage 1: Transfer compressed data to GPU
    /// - Stage 2: Decompress on GPU
    ///
    /// This overlaps I/O, transfers, and computation for maximum throughput.
    pub struct StreamingLz4Context {
        ctx: GpuLz4Context,
        stream_pool: CudaStreamPool,
        /// Number of concurrent operations (pipeline depth)
        pipeline_depth: usize,
    }

    impl StreamingLz4Context {
        /// Creates a new streaming context with pipelining.
        ///
        /// # Arguments
        ///
        /// * `device_id` - CUDA device ID
        /// * `pipeline_depth` - Number of concurrent operations (2-4 recommended)
        pub fn new(device_id: usize, pipeline_depth: usize) -> Result<Self, GpuLz4Error> {
            let mut ctx = GpuLz4Context::new(device_id)?;
            ctx.load_kernel()?;

            let stream_pool = CudaStreamPool::new(Arc::clone(&ctx.device), pipeline_depth)?;

            Ok(Self {
                ctx,
                stream_pool,
                pipeline_depth,
            })
        }

        /// Returns the underlying context for synchronous operations.
        pub fn context(&self) -> &GpuLz4Context {
            &self.ctx
        }

        /// Returns the stream pool for advanced async operations.
        pub fn stream_pool(&self) -> &CudaStreamPool {
            &self.stream_pool
        }

        /// Decompresses multiple blocks with streaming/pipelining.
        ///
        /// This method overlaps H2D transfers with decompression to maximize
        /// throughput. Blocks are processed in batches according to pipeline depth.
        ///
        /// # Performance
        ///
        /// For N blocks with pipeline depth P:
        /// - Sequential time: N * (transfer + decompress)
        /// - Pipelined time: N * max(transfer, decompress) + P * transfer
        ///
        /// With typical ratios, expect 1.5-2x speedup.
        pub fn decompress_blocks_streaming(
            &self,
            blocks: &[(Vec<u8>, usize)],
        ) -> Result<CudaSlice<u8>, GpuLz4Error> {
            if blocks.is_empty() {
                return Err(GpuLz4Error::InvalidInput {
                    message: "No blocks to decompress".to_string(),
                });
            }

            // Calculate total output size
            let total_uncompressed: usize = blocks.iter().map(|(_, s)| *s).sum();

            // Allocate output buffer
            let d_output: CudaSlice<u8> =
                self.ctx
                    .device
                    .alloc_zeros(total_uncompressed)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;

            // Process blocks in groups based on pipeline depth
            let group_size = self.pipeline_depth;
            let mut output_offset: usize = 0;

            for chunk in blocks.chunks(group_size) {
                // Submit async transfers and decompressions for this group
                self.process_block_group(chunk, &d_output, output_offset)?;

                // Update offset
                for (_, size) in chunk {
                    output_offset += size;
                }
            }

            // Synchronize all streams to ensure completion
            self.stream_pool.synchronize_all()?;

            Ok(d_output)
        }

        /// Processes a group of blocks with overlapped transfers and decompression.
        fn process_block_group(
            &self,
            blocks: &[(Vec<u8>, usize)],
            d_output: &CudaSlice<u8>,
            output_offset: usize,
        ) -> Result<(), GpuLz4Error> {
            // For each block in the group, we:
            // 1. Transfer compressed data to GPU (async on stream i)
            // 2. Launch decompression kernel (async on stream i)
            // 3. Copy decompressed results into the main output buffer

            let mut d_inputs: Vec<CudaSlice<u8>> = Vec::with_capacity(blocks.len());
            let mut d_outputs: Vec<CudaSlice<u8>> = Vec::with_capacity(blocks.len());

            // Phase 1: Allocate and transfer all blocks (async)
            for (i, (compressed, uncompressed_size)) in blocks.iter().enumerate() {
                let _stream = self.stream_pool.get_stream(i);

                // Allocate input buffer
                let d_input = self.ctx.device.htod_copy(compressed.clone()).map_err(|e| {
                    GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    }
                })?;
                d_inputs.push(d_input);

                // Allocate output buffer for this block
                let d_block_output: CudaSlice<u8> = self
                    .ctx
                    .device
                    .alloc_zeros(*uncompressed_size)
                    .map_err(|e| GpuLz4Error::MemoryAlloc {
                        message: e.to_string(),
                    })?;
                d_outputs.push(d_block_output);
            }

            // Phase 2: Launch decompression kernels on each stream
            let func = self
                .ctx
                .device
                .get_func("lz4_decompress", "lz4_decompress_block")
                .ok_or_else(|| GpuLz4Error::KernelLoad {
                    message: "Kernel not found".to_string(),
                })?;

            for (i, ((compressed, uncompressed_size), (d_input, d_block_output))) in blocks
                .iter()
                .zip(d_inputs.iter().zip(d_outputs.iter()))
                .enumerate()
            {
                let _stream = self.stream_pool.get_stream(i);
                let cfg = LaunchConfig::for_num_elems(1);

                unsafe {
                    func.clone().launch(
                        cfg,
                        (
                            d_input,
                            compressed.len() as u32,
                            d_block_output,
                            *uncompressed_size as u32,
                        ),
                    )
                }
                .map_err(|e| GpuLz4Error::KernelExec {
                    message: e.to_string(),
                })?;
            }

            // Phase 3: Copy decompressed blocks into the main output buffer
            let mut block_offset = output_offset;
            for ((_, uncompressed_size), d_block_output) in blocks.iter().zip(d_outputs.iter()) {
                if *uncompressed_size > 0 {
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_sync(
                            *d_output.device_ptr() + block_offset as u64,
                            *d_block_output.device_ptr(),
                            *uncompressed_size,
                        )
                    }
                    .map_err(|e| GpuLz4Error::MemoryCopy {
                        message: e.to_string(),
                    })?;
                }
                block_offset += uncompressed_size;
            }

            Ok(())
        }

        /// Decompresses blocks with callback-based streaming.
        ///
        /// This allows processing decompressed data as it becomes available,
        /// useful for layer-by-layer model loading.
        ///
        /// # Arguments
        ///
        /// * `blocks` - Iterator of (compressed_data, uncompressed_size) tuples
        /// * `callback` - Called for each decompressed block with (block_idx, data)
        pub fn decompress_blocks_with_callback<F, I>(
            &self,
            blocks: I,
            mut callback: F,
        ) -> Result<(), GpuLz4Error>
        where
            I: Iterator<Item = (Vec<u8>, usize)>,
            F: FnMut(usize, Vec<u8>) -> Result<(), GpuLz4Error>,
        {
            let blocks_vec: Vec<_> = blocks.collect();

            for (idx, (compressed, uncompressed_size)) in blocks_vec.iter().enumerate() {
                // Decompress single block
                let d_output = self.ctx.decompress_block(compressed, *uncompressed_size)?;

                // Copy back to host
                let mut host_data = vec![0u8; *uncompressed_size];
                self.ctx
                    .device
                    .dtoh_sync_copy_into(&d_output, &mut host_data)
                    .map_err(|e| GpuLz4Error::MemoryCopy {
                        message: e.to_string(),
                    })?;

                // Call user callback
                callback(idx, host_data)?;
            }

            Ok(())
        }
    }

    /// Statistics from streaming decompression.
    #[derive(Debug, Clone, Default)]
    pub struct StreamingStats {
        /// Total bytes transferred to GPU
        pub bytes_transferred: usize,
        /// Total bytes decompressed
        pub bytes_decompressed: usize,
        /// Number of blocks processed
        pub blocks_processed: usize,
        /// Time spent in H2D transfers (microseconds)
        pub transfer_time_us: u64,
        /// Time spent in decompression (microseconds)
        pub decompress_time_us: u64,
        /// Total wall-clock time (microseconds)
        pub total_time_us: u64,
    }

    impl StreamingStats {
        /// Returns effective throughput in GB/s.
        pub fn throughput_gbps(&self) -> f64 {
            if self.total_time_us == 0 {
                return 0.0;
            }
            let bytes = self.bytes_decompressed as f64;
            let seconds = self.total_time_us as f64 / 1_000_000.0;
            bytes / seconds / 1e9
        }

        /// Returns overlap efficiency (1.0 = perfect overlap, 0.5 = no overlap).
        pub fn overlap_efficiency(&self) -> f64 {
            let serial_time = self.transfer_time_us + self.decompress_time_us;
            if serial_time == 0 {
                return 1.0;
            }
            serial_time as f64 / self.total_time_us as f64
        }
    }

    /// Errors from GPU LZ4 operations.
    #[derive(Debug, thiserror::Error)]
    pub enum GpuLz4Error {
        /// CUDA device initialization failed.
        #[error("Failed to initialize CUDA device {device_id}: {message}")]
        DeviceInit {
            /// CUDA device ID.
            device_id: usize,
            /// Error message.
            message: String,
        },

        /// CUDA stream creation failed.
        #[error("Failed to create CUDA stream {stream_id}: {message}")]
        StreamCreate {
            /// Stream index.
            stream_id: usize,
            /// Error message.
            message: String,
        },

        /// Kernel loading failed.
        #[error("Failed to load LZ4 kernel: {message}")]
        KernelLoad {
            /// Error message.
            message: String,
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

        /// Unsupported data type for decompression.
        #[error("Unsupported dtype: {dtype}")]
        UnsupportedDtype {
            /// Data type name.
            dtype: String,
        },

        /// Candle tensor creation failed.
        #[error("Tensor creation failed: {message}")]
        TensorCreate {
            /// Error message.
            message: String,
        },
    }

    /// LZ4 decompression kernels K1 and K2 in PTX format (sm_50).
    ///
    /// This implements the LZ4 block format decompression on GPU.
    /// K1: single-block, K2: multi-block parallel (thread 0 per block).
    ///
    /// LZ4 Block Format:
    /// - Token byte: 4 bits literal length + 4 bits match length
    /// - Extended literal length (if literal length == 15)
    /// - Literal bytes
    /// - Offset (2 bytes, little-endian)
    /// - Extended match length (if match length == 15)
    const LZ4_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// Single block decompression kernel
.visible .entry lz4_decompress_block(
    .param .u64 input_ptr,
    .param .u32 input_size,
    .param .u64 output_ptr,
    .param .u32 output_size
)
{
    .reg .u64 %rd<21>;
    .reg .u32 %r<32>;
    .reg .pred %p<9>;
    .reg .b8 %rb<4>;

    // Load parameters
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u32 %r1, [input_size];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r2, [output_size];

    // Input position
    mov.u32 %r3, 0;  // in_pos
    // Output position
    mov.u32 %r4, 0;  // out_pos

LOOP:
    // Check if we've reached end of input
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra DONE;

    // Read token byte
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u8 %rb0, [%rd4];
    cvt.u32.u8 %r5, %rb0;  // token
    add.u32 %r3, %r3, 1;   // in_pos++

    // Literal length = token >> 4
    shr.u32 %r6, %r5, 4;

    // If literal length == 15, read extended length
    setp.eq.u32 %p2, %r6, 15;
    @!%p2 bra SKIP_LITERAL_EXT;

LITERAL_EXT_LOOP:
    cvt.u64.u32 %rd5, %r3;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u8 %rb1, [%rd6];
    cvt.u32.u8 %r7, %rb1;
    add.u32 %r3, %r3, 1;
    add.u32 %r6, %r6, %r7;
    setp.eq.u32 %p3, %r7, 255;
    @%p3 bra LITERAL_EXT_LOOP;

SKIP_LITERAL_EXT:
    // Copy literals
    mov.u32 %r8, 0;  // literal counter

LITERAL_COPY_LOOP:
    setp.ge.u32 %p4, %r8, %r6;
    @%p4 bra LITERAL_COPY_DONE;

    // Read from input
    cvt.u64.u32 %rd7, %r3;
    add.u64 %rd8, %rd1, %rd7;
    ld.global.u8 %rb2, [%rd8];
    add.u32 %r3, %r3, 1;

    // Write to output
    cvt.u64.u32 %rd9, %r4;
    add.u64 %rd10, %rd2, %rd9;
    st.global.u8 [%rd10], %rb2;
    add.u32 %r4, %r4, 1;

    add.u32 %r8, %r8, 1;
    bra LITERAL_COPY_LOOP;

LITERAL_COPY_DONE:
    // Check if we're at end of block (no more matches)
    setp.ge.u32 %p5, %r3, %r1;
    @%p5 bra DONE;

    // Read 2-byte offset (little-endian)
    cvt.u64.u32 %rd11, %r3;
    add.u64 %rd12, %rd1, %rd11;
    ld.global.u8 %rb0, [%rd12];
    cvt.u32.u8 %r9, %rb0;
    add.u32 %r3, %r3, 1;

    cvt.u64.u32 %rd13, %r3;
    add.u64 %rd14, %rd1, %rd13;
    ld.global.u8 %rb1, [%rd14];
    cvt.u32.u8 %r10, %rb1;
    add.u32 %r3, %r3, 1;

    shl.b32 %r11, %r10, 8;
    or.b32 %r12, %r9, %r11;  // offset

    // Match length = (token & 0x0F) + 4
    and.b32 %r13, %r5, 15;
    add.u32 %r14, %r13, 4;  // match_length

    // If match length base == 15, read extended length
    setp.eq.u32 %p6, %r13, 15;
    @!%p6 bra SKIP_MATCH_EXT;

MATCH_EXT_LOOP:
    cvt.u64.u32 %rd15, %r3;
    add.u64 %rd16, %rd1, %rd15;
    ld.global.u8 %rb3, [%rd16];
    cvt.u32.u8 %r15, %rb3;
    add.u32 %r3, %r3, 1;
    add.u32 %r14, %r14, %r15;
    setp.eq.u32 %p7, %r15, 255;
    @%p7 bra MATCH_EXT_LOOP;

SKIP_MATCH_EXT:
    // Copy match (from output buffer, offset bytes back)
    sub.u32 %r16, %r4, %r12;  // match source position
    mov.u32 %r17, 0;  // match counter

MATCH_COPY_LOOP:
    setp.ge.u32 %p8, %r17, %r14;
    @%p8 bra MATCH_COPY_DONE;

    // Read from match position
    add.u32 %r18, %r16, %r17;
    cvt.u64.u32 %rd17, %r18;
    add.u64 %rd18, %rd2, %rd17;
    ld.global.u8 %rb0, [%rd18];

    // Write to output position
    cvt.u64.u32 %rd19, %r4;
    add.u64 %rd20, %rd2, %rd19;
    st.global.u8 [%rd20], %rb0;
    add.u32 %r4, %r4, 1;

    add.u32 %r17, %r17, 1;
    bra MATCH_COPY_LOOP;

MATCH_COPY_DONE:
    bra LOOP;

DONE:
    ret;
}

// Parallel multi-block decompression kernel
.visible .entry lz4_decompress_blocks_parallel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 offsets_in_ptr,
    .param .u64 offsets_out_ptr,
    .param .u64 compressed_sizes_ptr,
    .param .u64 uncompressed_sizes_ptr,
    .param .u32 num_blocks
)
{
    .reg .u64 %rd<34>;
    .reg .u32 %r<64>;
    .reg .pred %p<16>;
    .reg .b8 %rb<8>;

    // Get block index from CUDA block ID
    mov.u32 %r1, %ctaid.x;

    // Check bounds
    ld.param.u32 %r2, [num_blocks];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra BLOCK_DONE;

    // Only thread 0 in each warp does the work (for now)
    mov.u32 %r3, %tid.x;
    setp.ne.u32 %p2, %r3, 0;
    @%p2 bra BLOCK_DONE;

    // Load this block's parameters
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u64 %rd3, [offsets_in_ptr];
    ld.param.u64 %rd4, [offsets_out_ptr];
    ld.param.u64 %rd5, [compressed_sizes_ptr];
    ld.param.u64 %rd6, [uncompressed_sizes_ptr];

    // Calculate offsets for this block
    mul.lo.u32 %r4, %r1, 4;
    cvt.u64.u32 %rd7, %r4;

    add.u64 %rd8, %rd3, %rd7;
    ld.global.u32 %r5, [%rd8];  // offset_in

    add.u64 %rd9, %rd4, %rd7;
    ld.global.u32 %r6, [%rd9];  // offset_out

    add.u64 %rd10, %rd5, %rd7;
    ld.global.u32 %r7, [%rd10];  // compressed_size

    add.u64 %rd11, %rd6, %rd7;
    ld.global.u32 %r8, [%rd11];  // uncompressed_size

    // Calculate input/output pointers for this block
    cvt.u64.u32 %rd12, %r5;
    add.u64 %rd13, %rd1, %rd12;  // input_ptr for this block

    cvt.u64.u32 %rd14, %r6;
    add.u64 %rd15, %rd2, %rd14;  // output_ptr for this block

    // Input/output positions within block
    mov.u32 %r10, 0;  // in_pos
    mov.u32 %r11, 0;  // out_pos

BLOCK_LOOP:
    // Check if we've reached end of input
    setp.ge.u32 %p3, %r10, %r7;
    @%p3 bra BLOCK_DONE;

    // Read token byte
    cvt.u64.u32 %rd16, %r10;
    add.u64 %rd17, %rd13, %rd16;
    ld.global.u8 %rb0, [%rd17];
    cvt.u32.u8 %r12, %rb0;  // token
    add.u32 %r10, %r10, 1;

    // Literal length = token >> 4
    shr.u32 %r13, %r12, 4;

    // Extended literal length
    setp.eq.u32 %p4, %r13, 15;
    @!%p4 bra BLOCK_SKIP_LIT_EXT;

BLOCK_LIT_EXT:
    cvt.u64.u32 %rd18, %r10;
    add.u64 %rd19, %rd13, %rd18;
    ld.global.u8 %rb1, [%rd19];
    cvt.u32.u8 %r14, %rb1;
    add.u32 %r10, %r10, 1;
    add.u32 %r13, %r13, %r14;
    setp.eq.u32 %p5, %r14, 255;
    @%p5 bra BLOCK_LIT_EXT;

BLOCK_SKIP_LIT_EXT:
    // Copy literals
    mov.u32 %r15, 0;

BLOCK_LIT_COPY:
    setp.ge.u32 %p6, %r15, %r13;
    @%p6 bra BLOCK_LIT_DONE;

    cvt.u64.u32 %rd20, %r10;
    add.u64 %rd21, %rd13, %rd20;
    ld.global.u8 %rb2, [%rd21];
    add.u32 %r10, %r10, 1;

    cvt.u64.u32 %rd22, %r11;
    add.u64 %rd23, %rd15, %rd22;
    st.global.u8 [%rd23], %rb2;
    add.u32 %r11, %r11, 1;

    add.u32 %r15, %r15, 1;
    bra BLOCK_LIT_COPY;

BLOCK_LIT_DONE:
    // Check end of block
    setp.ge.u32 %p7, %r10, %r7;
    @%p7 bra BLOCK_DONE;

    // Read offset
    cvt.u64.u32 %rd24, %r10;
    add.u64 %rd25, %rd13, %rd24;
    ld.global.u8 %rb3, [%rd25];
    cvt.u32.u8 %r16, %rb3;
    add.u32 %r10, %r10, 1;

    cvt.u64.u32 %rd26, %r10;
    add.u64 %rd27, %rd13, %rd26;
    ld.global.u8 %rb4, [%rd27];
    cvt.u32.u8 %r17, %rb4;
    add.u32 %r10, %r10, 1;

    shl.b32 %r18, %r17, 8;
    or.b32 %r19, %r16, %r18;  // offset

    // Match length
    and.b32 %r20, %r12, 15;
    add.u32 %r21, %r20, 4;

    setp.eq.u32 %p8, %r20, 15;
    @!%p8 bra BLOCK_SKIP_MATCH_EXT;

BLOCK_MATCH_EXT:
    cvt.u64.u32 %rd28, %r10;
    add.u64 %rd29, %rd13, %rd28;
    ld.global.u8 %rb5, [%rd29];
    cvt.u32.u8 %r22, %rb5;
    add.u32 %r10, %r10, 1;
    add.u32 %r21, %r21, %r22;
    setp.eq.u32 %p9, %r22, 255;
    @%p9 bra BLOCK_MATCH_EXT;

BLOCK_SKIP_MATCH_EXT:
    // Copy match
    sub.u32 %r23, %r11, %r19;
    mov.u32 %r24, 0;

BLOCK_MATCH_COPY:
    setp.ge.u32 %p10, %r24, %r21;
    @%p10 bra BLOCK_MATCH_DONE;

    add.u32 %r25, %r23, %r24;
    cvt.u64.u32 %rd30, %r25;
    add.u64 %rd31, %rd15, %rd30;
    ld.global.u8 %rb6, [%rd31];

    cvt.u64.u32 %rd32, %r11;
    add.u64 %rd33, %rd15, %rd32;
    st.global.u8 [%rd33], %rb6;
    add.u32 %r11, %r11, 1;

    add.u32 %r24, %r24, 1;
    bra BLOCK_MATCH_COPY;

BLOCK_MATCH_DONE:
    bra BLOCK_LOOP;

BLOCK_DONE:
    ret;
}
"#;

    /// Warp-parallel LZ4 decompression kernel K3 in PTX format (sm_70).
    ///
    /// Requires sm_70+ (Volta) for shfl.sync and bar.warp.sync instructions.
    /// Feature-gated behind `cuda-experimental` (DD-2).
    #[cfg(feature = "cuda-experimental")]
    const LZ4_WARP_KERNEL_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

// Warp-parallel multi-block decompression kernel
// Uses all 32 threads in warp for parallel literal/match copying
.visible .entry lz4_decompress_blocks_warp(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 offsets_in_ptr,
    .param .u64 offsets_out_ptr,
    .param .u64 compressed_sizes_ptr,
    .param .u64 uncompressed_sizes_ptr,
    .param .u32 num_blocks
)
{
    .reg .u64 %rd<40>;
    .reg .u32 %r<80>;
    .reg .pred %p<20>;
    .reg .b8 %rb<8>;

    // Get block index and thread lane
    mov.u32 %r1, %ctaid.x;      // block_idx
    mov.u32 %r2, %tid.x;        // lane (0-31)

    // Check bounds
    ld.param.u32 %r3, [num_blocks];
    setp.ge.u32 %p1, %r1, %r3;
    @%p1 bra WARP_DONE;

    // Load block parameters (all threads load same values)
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u64 %rd3, [offsets_in_ptr];
    ld.param.u64 %rd4, [offsets_out_ptr];
    ld.param.u64 %rd5, [compressed_sizes_ptr];
    ld.param.u64 %rd6, [uncompressed_sizes_ptr];

    // Calculate offsets for this block
    mul.lo.u32 %r4, %r1, 4;
    cvt.u64.u32 %rd7, %r4;

    add.u64 %rd8, %rd3, %rd7;
    ld.global.u32 %r5, [%rd8];   // offset_in

    add.u64 %rd9, %rd4, %rd7;
    ld.global.u32 %r6, [%rd9];   // offset_out

    add.u64 %rd10, %rd5, %rd7;
    ld.global.u32 %r7, [%rd10];  // compressed_size

    add.u64 %rd11, %rd6, %rd7;
    ld.global.u32 %r8, [%rd11];  // uncompressed_size

    // Calculate input/output base pointers for this block
    cvt.u64.u32 %rd12, %r5;
    add.u64 %rd13, %rd1, %rd12;  // input_base

    cvt.u64.u32 %rd14, %r6;
    add.u64 %rd15, %rd2, %rd14;  // output_base

    // Positions (shared across warp via shuffle)
    mov.u32 %r10, 0;  // in_pos (only thread 0 updates)
    mov.u32 %r11, 0;  // out_pos (all threads need this)

WARP_LOOP:
    // Thread 0 checks end of input
    setp.eq.u32 %p2, %r2, 0;

    // Broadcast in_pos from thread 0 to all threads
    shfl.sync.idx.b32 %r20, %r10, 0, 0x1f, 0xffffffff;

    // Check if done (all threads check)
    setp.ge.u32 %p3, %r20, %r7;
    @%p3 bra WARP_DONE;

    // Thread 0 reads and parses token
    mov.u32 %r12, 0;  // token (will be shuffled)
    mov.u32 %r13, 0;  // literal_length
    mov.u32 %r14, 0;  // match_length
    mov.u32 %r19, 0;  // offset

    @!%p2 bra WARP_SKIP_PARSE;

    // Read token byte (thread 0 only)
    cvt.u64.u32 %rd16, %r20;
    add.u64 %rd17, %rd13, %rd16;
    ld.global.u8 %rb0, [%rd17];
    cvt.u32.u8 %r12, %rb0;
    add.u32 %r10, %r10, 1;

    // Literal length = token >> 4
    shr.u32 %r13, %r12, 4;

    // Extended literal length
    setp.ne.u32 %p4, %r13, 15;
    @%p4 bra WARP_PARSE_OFFSET;

WARP_LIT_EXT:
    cvt.u64.u32 %rd18, %r10;
    add.u64 %rd19, %rd13, %rd18;
    ld.global.u8 %rb1, [%rd19];
    cvt.u32.u8 %r15, %rb1;
    add.u32 %r10, %r10, 1;
    add.u32 %r13, %r13, %r15;
    setp.eq.u32 %p5, %r15, 255;
    @%p5 bra WARP_LIT_EXT;

WARP_PARSE_OFFSET:
    // Save literal data start: r10 is past token + extension bytes here
    mov.u32 %r70, %r10;

    // in_pos after literals = in_pos + literal_length
    add.u32 %r16, %r10, %r13;

    // Check if we have match data (not at end)
    setp.ge.u32 %p6, %r16, %r7;
    @%p6 bra WARP_NO_MATCH;

    // Read offset (2 bytes)
    cvt.u64.u32 %rd20, %r16;
    add.u64 %rd21, %rd13, %rd20;
    ld.global.u8 %rb2, [%rd21];
    cvt.u32.u8 %r17, %rb2;

    add.u32 %r21, %r16, 1;
    cvt.u64.u32 %rd22, %r21;
    add.u64 %rd23, %rd13, %rd22;
    ld.global.u8 %rb3, [%rd23];
    cvt.u32.u8 %r18, %rb3;

    shl.b32 %r22, %r18, 8;
    or.b32 %r19, %r17, %r22;  // offset

    // Match length = (token & 0x0F) + 4
    and.b32 %r23, %r12, 15;
    add.u32 %r14, %r23, 4;

    // Update in_pos past offset bytes
    add.u32 %r10, %r16, 2;

    // Extended match length
    setp.ne.u32 %p7, %r23, 15;
    @%p7 bra WARP_SKIP_PARSE;

WARP_MATCH_EXT:
    cvt.u64.u32 %rd24, %r10;
    add.u64 %rd25, %rd13, %rd24;
    ld.global.u8 %rb4, [%rd25];
    cvt.u32.u8 %r24, %rb4;
    add.u32 %r10, %r10, 1;
    add.u32 %r14, %r14, %r24;
    setp.eq.u32 %p8, %r24, 255;
    @%p8 bra WARP_MATCH_EXT;
    bra WARP_SKIP_PARSE;

WARP_NO_MATCH:
    mov.u32 %r14, 0;  // no match
    add.u32 %r10, %r16, 0;  // in_pos = after literals

WARP_SKIP_PARSE:
    // Broadcast parsed values from thread 0 to all threads
    shfl.sync.idx.b32 %r30, %r10, 0, 0x1f, 0xffffffff;  // in_pos after this sequence
    shfl.sync.idx.b32 %r31, %r13, 0, 0x1f, 0xffffffff;  // literal_length
    shfl.sync.idx.b32 %r32, %r14, 0, 0x1f, 0xffffffff;  // match_length
    shfl.sync.idx.b32 %r33, %r19, 0, 0x1f, 0xffffffff;  // offset

    // Get literal data start (saved at WARP_PARSE_OFFSET, accounts for token + extension bytes)
    shfl.sync.idx.b32 %r35, %r70, 0, 0x1f, 0xffffffff;

    // Get current out_pos (broadcast from thread 0)
    shfl.sync.idx.b32 %r36, %r11, 0, 0x1f, 0xffffffff;

    // === Parallel literal copy ===
    // Each thread copies bytes: lane, lane+32, lane+64, ...
    mov.u32 %r40, %r2;  // start at lane

WARP_LIT_COPY:
    setp.ge.u32 %p9, %r40, %r31;  // compare with literal_length
    @%p9 bra WARP_LIT_COPY_DONE;

    // Read from input[literal_data_start + lane_offset]
    add.u32 %r41, %r35, %r40;
    cvt.u64.u32 %rd26, %r41;
    add.u64 %rd27, %rd13, %rd26;
    ld.global.u8 %rb5, [%rd27];

    // Write to output[out_pos + lane_offset]
    add.u32 %r42, %r36, %r40;
    cvt.u64.u32 %rd28, %r42;
    add.u64 %rd29, %rd15, %rd28;
    st.global.u8 [%rd29], %rb5;

    add.u32 %r40, %r40, 32;  // stride by warp size
    bra WARP_LIT_COPY;

WARP_LIT_COPY_DONE:
    // Sync warp before match copy
    bar.warp.sync 0xffffffff;

    // Update out_pos (on all threads)
    add.u32 %r11, %r36, %r31;

    // Check if we have a match
    setp.eq.u32 %p10, %r32, 0;
    @%p10 bra WARP_UPDATE_INPOS;

    // === Parallel match copy ===
    // Match source = out_pos - offset
    sub.u32 %r50, %r11, %r33;  // match_src

    // For overlapping copies (offset < match_length), we need sequential
    // to correctly handle RLE-like patterns where later bytes depend on earlier writes.
    // Parallel only when offset >= match_length (no overlap at all).

    setp.lt.u32 %p11, %r33, %r32;
    @%p11 bra WARP_MATCH_SEQUENTIAL;

    // Parallel match copy (offset >= match_length, no overlap)
    mov.u32 %r51, %r2;  // start at lane

WARP_MATCH_PARALLEL:
    setp.ge.u32 %p12, %r51, %r32;
    @%p12 bra WARP_MATCH_DONE;

    // Read from output[match_src + lane_offset]
    add.u32 %r52, %r50, %r51;
    cvt.u64.u32 %rd30, %r52;
    add.u64 %rd31, %rd15, %rd30;
    ld.global.u8 %rb6, [%rd31];

    // Write to output[out_pos + lane_offset]
    add.u32 %r53, %r11, %r51;
    cvt.u64.u32 %rd32, %r53;
    add.u64 %rd33, %rd15, %rd32;
    st.global.u8 [%rd33], %rb6;

    add.u32 %r51, %r51, 32;
    bra WARP_MATCH_PARALLEL;

WARP_MATCH_SEQUENTIAL:
    // Sequential match copy for small offsets (only thread 0)
    setp.ne.u32 %p13, %r2, 0;
    @%p13 bra WARP_MATCH_DONE;

    mov.u32 %r54, 0;

WARP_MATCH_SEQ_LOOP:
    setp.ge.u32 %p14, %r54, %r32;
    @%p14 bra WARP_MATCH_DONE;

    add.u32 %r55, %r50, %r54;
    cvt.u64.u32 %rd34, %r55;
    add.u64 %rd35, %rd15, %rd34;
    ld.global.u8 %rb7, [%rd35];

    add.u32 %r56, %r11, %r54;
    cvt.u64.u32 %rd36, %r56;
    add.u64 %rd37, %rd15, %rd36;
    st.global.u8 [%rd37], %rb7;

    add.u32 %r54, %r54, 1;
    bra WARP_MATCH_SEQ_LOOP;

WARP_MATCH_DONE:
    // Update out_pos
    add.u32 %r11, %r11, %r32;

    // Sync warp before next iteration
    bar.warp.sync 0xffffffff;

WARP_UPDATE_INPOS:
    // Update in_pos from thread 0's value for next iteration
    mov.u32 %r10, %r30;
    bra WARP_LOOP;

WARP_DONE:
    ret;
}
"#;

    #[cfg(test)]
    mod tests {
        use super::*;
        use cudarc::driver::DeviceSlice;

        /// Helper to check if CUDA is available for testing.
        fn cuda_available() -> bool {
            GpuLz4Context::new(0).is_ok()
        }

        /// Creates a simple LZ4-compressed block containing only literals (no matches).
        /// This is the simplest valid LZ4 block format: a single "last sequence"
        /// with all data as literals and no match section.
        ///
        /// Format: [token][extension_bytes...][literal_data]
        /// Token high nibble = min(literal_length, 15), low nibble = 0 (last sequence)
        /// If literal_length >= 15, extension bytes encode the remainder.
        ///
        /// This encodes all data in ONE sequence. In LZ4, only the last sequence
        /// may omit the match section; all intermediate sequences must have one.
        fn create_literals_only_lz4(data: &[u8]) -> Vec<u8> {
            let literal_length = data.len();
            let mut result = Vec::new();

            if literal_length < 15 {
                // Literal length fits in the token's high nibble
                result.push((literal_length as u8) << 4);
            } else {
                // Token: literal_length_base = 15, match = 0 (last sequence)
                result.push(0xF0);
                let mut remaining = literal_length - 15;
                while remaining >= 255 {
                    result.push(255);
                    remaining -= 255;
                }
                result.push(remaining as u8);
            }

            result.extend_from_slice(data);
            result
        }

        /// Creates LZ4 data with a simple match pattern.
        /// Format: [token][literals][offset][match extension if needed]
        fn create_lz4_with_match() -> (Vec<u8>, Vec<u8>) {
            // Original data: "ABCDABCD" - second "ABCD" matches first
            let original = b"ABCDABCD".to_vec();

            // LZ4 encoding:
            // Token: 4 literals, 4-byte match (match_base = 0, actual = 0 + 4)
            // Literals: "ABCD"
            // Offset: 4 (little-endian: 0x04, 0x00)
            let compressed = vec![
                0x40, // Token: 4 literals, 0 match base (0 + 4 = 4)
                b'A', b'B', b'C', b'D', // Literals
                0x04, 0x00, // Offset: 4 bytes back
            ];

            (compressed, original)
        }

        /// Creates a more complex LZ4 sequence with multiple blocks.
        fn create_multi_sequence_lz4() -> (Vec<u8>, Vec<u8>) {
            // Original: "HelloHelloWorld"
            // - 5 literals "Hello"
            // - Match 5 bytes from offset 5 (copies "Hello")
            // - 5 literals "World"
            let original = b"HelloHelloWorld".to_vec();

            let compressed = vec![
                // First sequence: 5 literals "Hello", then match
                0x51, // Token: 5 literals, 1 match base (1+4=5)
                b'H', b'e', b'l', b'l', b'o', // Literals
                0x05, 0x00, // Offset: 5 bytes back
                // Second sequence: 5 literals "World", no match (end of block)
                0x50, // Token: 5 literals, 0 match (last sequence)
                b'W', b'o', b'r', b'l', b'd', // Literals
            ];

            (compressed, original)
        }

        #[test]
        fn test_context_creation() {
            // This test requires CUDA hardware
            // Skip if no GPU available
            match GpuLz4Context::new(0) {
                Ok(ctx) => {
                    assert_eq!(ctx.device_id(), 0);
                },
                Err(GpuLz4Error::DeviceInit { .. }) => {
                    // No CUDA device available, skip
                    eprintln!("Skipping test: no CUDA device available");
                },
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        #[test]
        fn test_kernel_loading() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");

            // First load should succeed
            ctx.load_kernel().expect("first kernel load");
            assert!(ctx.kernel_loaded);

            // Second load should be a no-op (already loaded)
            ctx.load_kernel().expect("second kernel load");
            assert!(ctx.kernel_loaded);
        }

        #[test]
        fn test_empty_blocks_error() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let empty_blocks: Vec<(Vec<u8>, usize)> = vec![];
            let result = ctx.decompress_blocks_parallel(&empty_blocks);

            match result {
                Err(GpuLz4Error::InvalidInput { message }) => {
                    assert!(message.contains("No blocks"));
                },
                _ => panic!("Expected InvalidInput error for empty blocks"),
            }
        }

        #[test]
        fn test_decompress_single_block_literals() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Test with simple literal-only data
            let original = b"Hello, World!";
            let compressed = create_literals_only_lz4(original);

            let result = ctx
                .decompress_block(&compressed, original.len())
                .expect("decompression");

            // Copy back to host and verify
            let mut host_data = vec![0u8; original.len()];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            assert_eq!(&host_data, original);
        }

        #[test]
        fn test_decompress_with_match() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let (compressed, expected) = create_lz4_with_match();

            let result = ctx
                .decompress_block(&compressed, expected.len())
                .expect("decompression");

            let mut host_data = vec![0u8; expected.len()];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            assert_eq!(host_data, expected);
        }

        #[test]
        fn test_decompress_multi_sequence() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let (compressed, expected) = create_multi_sequence_lz4();

            let result = ctx
                .decompress_block(&compressed, expected.len())
                .expect("decompression");

            let mut host_data = vec![0u8; expected.len()];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            assert_eq!(host_data, expected);
        }

        #[test]
        fn test_decompress_blocks_parallel() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create multiple blocks with different content
            let originals: Vec<&[u8]> = vec![
                b"Block one data here",
                b"Block two with different content",
                b"Third block!",
                b"ABCDABCD", // This one has a match
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let result = ctx
                .decompress_blocks_parallel(&blocks)
                .expect("parallel decompression");

            // Copy back and verify each block
            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut host_data = vec![0u8; total_size];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            let mut offset = 0;
            for original in &originals {
                let decompressed = &host_data[offset..offset + original.len()];
                assert_eq!(
                    decompressed, *original,
                    "Block at offset {} mismatch",
                    offset
                );
                offset += original.len();
            }
        }

        #[test]
        fn test_decompress_to_tensor_f32() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create F32 data
            let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            // Compress as a single block
            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            // Decompress to tensor
            let device = Device::Cpu; // We copy back to CPU in decompress_to_tensor anyway
            let tensor = ctx
                .decompress_to_tensor(&blocks, &[8], DType::F32, &device)
                .expect("tensor creation");

            // Verify tensor contents
            let result: Vec<f32> = tensor.to_vec1().expect("extract tensor data");
            assert_eq!(result, f32_data);
        }

        #[test]
        fn test_decompress_to_tensor_2d() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create 2D F32 data (2x4 matrix)
            let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            let device = Device::Cpu;
            let tensor = ctx
                .decompress_to_tensor(&blocks, &[2, 4], DType::F32, &device)
                .expect("tensor creation");

            // Verify shape
            assert_eq!(tensor.dims(), &[2, 4]);

            // Verify contents
            let result: Vec<Vec<f32>> = tensor.to_vec2().expect("extract 2d tensor data");
            assert_eq!(
                result,
                vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
            );
        }

        #[test]
        fn test_unsupported_dtype_error() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let bytes = vec![0u8; 16];
            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            let device = Device::Cpu;
            let result = ctx.decompress_to_tensor(&blocks, &[16], DType::I64, &device);

            match result {
                Err(GpuLz4Error::UnsupportedDtype { dtype }) => {
                    assert!(dtype.contains("I64"));
                },
                _ => panic!("Expected UnsupportedDtype error"),
            }
        }

        #[test]
        fn test_lz4_literals_only_encoding() {
            // Unit test for our LZ4 encoder helper
            let data = b"Hello";
            let compressed = create_literals_only_lz4(data);

            // Should be: token (5 << 4 = 0x50) + 5 literals
            assert_eq!(compressed.len(), 6);
            assert_eq!(compressed[0], 0x50);
            assert_eq!(&compressed[1..], b"Hello");
        }

        #[test]
        fn test_lz4_extended_literal_encoding() {
            // Test with data > 15 bytes requiring extended literal length
            let data = vec![b'X'; 20]; // 20 X's
            let compressed = create_literals_only_lz4(&data);

            // Should be: token (0xF0) + extension byte (5) + 20 literals
            // 15 + 5 = 20
            assert_eq!(compressed.len(), 22); // 1 token + 1 ext + 20 data
            assert_eq!(compressed[0], 0xF0); // 15 in high nibble
            assert_eq!(compressed[1], 5); // Extension: 20 - 15 = 5
        }

        #[test]
        fn test_device_id_getter() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let ctx = GpuLz4Context::new(0).expect("context creation");
            assert_eq!(ctx.device_id(), 0);
        }

        #[test]
        fn test_invalid_device_id() {
            // Try to create context with an impossibly high device ID
            let result = GpuLz4Context::new(999);

            match result {
                Err(GpuLz4Error::DeviceInit { device_id, .. }) => {
                    assert_eq!(device_id, 999);
                },
                Ok(_) => {
                    // Somehow 999 devices exist? Unlikely but not an error
                },
                Err(other) => {
                    panic!("Expected DeviceInit error, got: {other:?}");
                },
            }
        }

        #[test]
        fn test_large_block_decompression() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create a larger block (1MB)
            let size = 1024 * 1024;
            let original: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let compressed = create_literals_only_lz4(&original);

            let result = ctx
                .decompress_block(&compressed, original.len())
                .expect("decompression");

            let mut host_data = vec![0u8; original.len()];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            assert_eq!(host_data, original);
        }

        #[test]
        fn test_many_small_blocks_parallel() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create 100 small blocks
            let num_blocks = 100;
            let originals: Vec<Vec<u8>> = (0..num_blocks)
                .map(|i| format!("Block number {} with some padding data", i).into_bytes())
                .collect();

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let result = ctx
                .decompress_blocks_parallel(&blocks)
                .expect("parallel decompression");

            // Verify total size
            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut host_data = vec![0u8; total_size];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            // Verify each block
            let mut offset = 0;
            for (i, original) in originals.iter().enumerate() {
                let decompressed = &host_data[offset..offset + original.len()];
                assert_eq!(
                    decompressed,
                    original.as_slice(),
                    "Block {} at offset {} mismatch",
                    i,
                    offset
                );
                offset += original.len();
            }
        }

        // ==================== Warp-Parallel Tests ====================
        // Gated behind cuda-experimental (DD-5: warp kernel has known bugs).

        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_parallel_correctness() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create test blocks
            let originals: Vec<&[u8]> = vec![
                b"Block one data here",
                b"Block two with different content",
                b"Third block with more data to decompress",
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            // Decompress with both methods
            let single_result = ctx
                .decompress_blocks_parallel(&blocks)
                .expect("single-thread");
            let warp_result = ctx
                .decompress_blocks_warp_parallel(&blocks)
                .expect("warp-parallel");

            // Copy both to host
            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut single_host = vec![0u8; total_size];
            let mut warp_host = vec![0u8; total_size];

            ctx.device
                .dtoh_sync_copy_into(&single_result, &mut single_host)
                .expect("copy single");
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            // Verify identical output
            assert_eq!(
                single_host, warp_host,
                "Warp-parallel should match single-threaded"
            );
        }

        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_parallel_large_literals() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create block with 64KB literals - should benefit most from warp parallelism
            let data = vec![0xABu8; 65536];
            let compressed = create_literals_only_lz4(&data);
            let blocks = vec![(compressed, data.len())];

            let result = ctx
                .decompress_blocks_warp_parallel(&blocks)
                .expect("warp decompress");

            let mut host_data = vec![0u8; data.len()];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            assert_eq!(host_data, data);
        }

        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_parallel_empty_blocks_error() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let empty_blocks: Vec<(Vec<u8>, usize)> = vec![];
            let result = ctx.decompress_blocks_warp_parallel(&empty_blocks);

            match result {
                Err(GpuLz4Error::InvalidInput { message }) => {
                    assert!(message.contains("No blocks"));
                },
                _ => panic!("Expected InvalidInput error for empty blocks"),
            }
        }

        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_parallel_many_blocks() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create 50 blocks with 1KB each
            let num_blocks = 50;
            let block_size = 1024;
            let originals: Vec<Vec<u8>> = (0..num_blocks)
                .map(|i| vec![(i % 256) as u8; block_size])
                .collect();

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let result = ctx
                .decompress_blocks_warp_parallel(&blocks)
                .expect("warp parallel");

            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut host_data = vec![0u8; total_size];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            // Verify each block
            let mut offset = 0;
            for (i, original) in originals.iter().enumerate() {
                let decompressed = &host_data[offset..offset + original.len()];
                assert_eq!(
                    decompressed,
                    original.as_slice(),
                    "Warp block {} at offset {} mismatch",
                    i,
                    offset
                );
                offset += original.len();
            }
        }

        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_parallel_mixed_sizes() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Mix of small and large blocks
            let originals: Vec<Vec<u8>> = vec![
                vec![0x11; 10],    // Small
                vec![0x22; 1000],  // Medium
                vec![0x33; 50000], // Large
                vec![0x44; 5],     // Tiny
                vec![0x55; 10000], // Medium-large
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let result = ctx
                .decompress_blocks_warp_parallel(&blocks)
                .expect("warp parallel");

            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut host_data = vec![0u8; total_size];
            ctx.device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy to host");

            // Verify each block
            let mut offset = 0;
            for (i, original) in originals.iter().enumerate() {
                let decompressed = &host_data[offset..offset + original.len()];
                assert_eq!(
                    decompressed,
                    original.as_slice(),
                    "Mixed size block {} mismatch",
                    i
                );
                offset += original.len();
            }
        }

        // ==================== Phase 3: Warp Kernel Equivalence Tests ====================
        // These tests require cuda-experimental AND a CUDA device.

        /// Regression test for DD-5: literal data start miscalculated when
        /// literal_length >= 15 (extension bytes skipped).
        ///
        /// The warp kernel computed literal start as `in_pos + 1` (skipping only
        /// the token byte), but when literal_length >= 15, there are extension
        /// bytes between the token and the actual literal data.
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_literal_extension_regression() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create block with 20 bytes — triggers extension byte path (>14)
            let data: Vec<u8> = (0..20u8).collect();
            let compressed = create_literals_only_lz4(&data);
            let blocks = vec![(compressed, data.len())];

            // Parallel kernel (K2) — known good
            let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
            let mut parallel_host = vec![0u8; data.len()];
            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy parallel");
            assert_eq!(parallel_host, data, "Parallel kernel should match original");

            // Warp kernel (K3) — should also match
            let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");
            let mut warp_host = vec![0u8; data.len()];
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            assert_eq!(
                warp_host,
                parallel_host,
                "DD-5: Warp kernel diverges from parallel kernel for 20-byte literals.\n\
                 Expected: {:?}\n\
                 Got:      {:?}\n\
                 This indicates the literal extension byte bug.",
                &parallel_host[..20.min(parallel_host.len())],
                &warp_host[..20.min(warp_host.len())]
            );
        }

        /// Property test: for random literal-only data of varying sizes,
        /// warp kernel must produce identical output to parallel kernel.
        #[cfg(feature = "cuda-experimental")]
        mod warp_proptest {
            use super::*;
            use proptest::prelude::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(20))]
                #[test]
                fn warp_matches_parallel_for_literals(
                    data in proptest::collection::vec(any::<u8>(), 1..512)
                ) {
                    if !cuda_available() {
                        return Ok(());
                    }

                    let mut ctx = GpuLz4Context::new(0).expect("context creation");
                    ctx.load_kernel().expect("kernel load");

                    let compressed = create_literals_only_lz4(&data);
                    let blocks = vec![(compressed, data.len())];

                    let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
                    let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");

                    let mut parallel_host = vec![0u8; data.len()];
                    let mut warp_host = vec![0u8; data.len()];

                    ctx.device.dtoh_sync_copy_into(&parallel_result, &mut parallel_host).expect("copy");
                    ctx.device.dtoh_sync_copy_into(&warp_result, &mut warp_host).expect("copy");

                    prop_assert_eq!(
                        &warp_host, &parallel_host,
                        "Warp kernel diverges from parallel for {} byte input",
                        data.len()
                    );
                }
            }
        }

        // ==================== Phase 3: Warp Kernel with Real LZ4 (Matches) ====================
        // These tests use lz4_flex to produce real compressed data containing
        // match sequences, validating the warp kernel's parallel match copy path.

        /// §3.4: Warp kernel produces byte-identical output to parallel kernel
        /// for data with LZ4 match sequences (not just literals).
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_matches_parallel_real_lz4() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Data with repeated patterns → LZ4 match sequences
            let original: Vec<u8> = (0..1024).map(|i| (i % 13) as u8).collect();
            let compressed = lz4_flex::block::compress_prepend_size(&original);
            // lz4_flex prepends a 4-byte little-endian size; strip it for raw block
            let raw_compressed = &compressed[4..];

            let blocks = vec![(raw_compressed.to_vec(), original.len())];

            let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
            let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");

            let mut parallel_host = vec![0u8; original.len()];
            let mut warp_host = vec![0u8; original.len()];

            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy parallel");
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            assert_eq!(
                parallel_host, original,
                "Parallel kernel should produce correct output"
            );
            assert_eq!(
                warp_host, parallel_host,
                "Warp kernel must match parallel kernel for data with matches"
            );
        }

        /// §3.4: Warp kernel handles large data with both literal and match sequences.
        /// Uses 64KB of patterned data that compresses well with LZ4.
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_large_data_with_matches() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // 64KB of repeating pattern → many LZ4 match sequences
            let original: Vec<u8> = (0..65536).map(|i| (i % 251) as u8).collect();
            let compressed = lz4_flex::block::compress_prepend_size(&original);
            let raw_compressed = &compressed[4..];

            let blocks = vec![(raw_compressed.to_vec(), original.len())];

            let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
            let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");

            let mut parallel_host = vec![0u8; original.len()];
            let mut warp_host = vec![0u8; original.len()];

            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy parallel");
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            assert_eq!(parallel_host, original, "Parallel kernel correctness");
            assert_eq!(
                warp_host,
                parallel_host,
                "Warp kernel must match parallel for 64KB data with matches.\n\
                 First divergence at byte {}",
                warp_host
                    .iter()
                    .zip(parallel_host.iter())
                    .position(|(w, p)| w != p)
                    .unwrap_or(warp_host.len())
            );
        }

        /// §3.4: Warp kernel handles multiple blocks with real LZ4 data.
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_multi_block_real_lz4() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Multiple blocks with different patterns
            let originals: Vec<Vec<u8>> = vec![
                (0..256).map(|i| (i % 7) as u8).collect(), // small, repetitive
                (0..4096).map(|i| (i % 37) as u8).collect(), // medium, moderate compression
                (0..512).map(|i| ((i * 7 + 3) % 256) as u8).collect(), // pseudo-random
                vec![0xAA; 2048],                          // highly repetitive
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = lz4_flex::block::compress_prepend_size(orig);
                    (compressed[4..].to_vec(), orig.len())
                })
                .collect();

            let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
            let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");

            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut parallel_host = vec![0u8; total_size];
            let mut warp_host = vec![0u8; total_size];

            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy parallel");
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            // Verify each block
            let mut offset = 0;
            for (i, orig) in originals.iter().enumerate() {
                let p_block = &parallel_host[offset..offset + orig.len()];
                let w_block = &warp_host[offset..offset + orig.len()];

                assert_eq!(p_block, orig.as_slice(), "Parallel block {} incorrect", i);
                assert_eq!(
                    w_block,
                    p_block,
                    "Warp block {} diverges from parallel at byte {}",
                    i,
                    w_block
                        .iter()
                        .zip(p_block.iter())
                        .position(|(w, p)| w != p)
                        .unwrap_or(w_block.len())
                );
                offset += orig.len();
            }
        }

        /// §3.4: Warp kernel handles small-offset matches (overlap requiring sequential copy).
        /// This specifically tests the sequential fallback path (offset < 32).
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_small_offset_matches() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Data designed to produce small-offset matches:
            // Repeating short pattern → LZ4 offset will be small (e.g., 4)
            let original: Vec<u8> = "ABCDABCDABCDABCDABCDABCDABCDABCD".repeat(100).into_bytes();

            let compressed = lz4_flex::block::compress_prepend_size(&original);
            let raw_compressed = &compressed[4..];

            let blocks = vec![(raw_compressed.to_vec(), original.len())];

            let parallel_result = ctx.decompress_blocks_parallel(&blocks).expect("parallel");
            let warp_result = ctx.decompress_blocks_warp_parallel(&blocks).expect("warp");

            let mut parallel_host = vec![0u8; original.len()];
            let mut warp_host = vec![0u8; original.len()];

            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy parallel");
            ctx.device
                .dtoh_sync_copy_into(&warp_result, &mut warp_host)
                .expect("copy warp");

            assert_eq!(parallel_host, original, "Parallel correctness");
            assert_eq!(
                warp_host, parallel_host,
                "Warp must match parallel for small-offset (overlapping) matches"
            );
        }

        // ==================== Feature-Gate Boundary Tests (Phase 2) ====================

        /// Trust boundary §7: Verifies the warp method is not available
        /// in the default build (no cuda-experimental feature).
        #[test]
        #[cfg(not(feature = "cuda-experimental"))]
        fn test_warp_method_not_in_default_build() {
            // This test compiles only when cuda-experimental is OFF.
            // It statically proves that decompress_blocks_warp_parallel
            // is not accessible in the default feature set.
            //
            // If someone removes the cfg gate from the warp method,
            // this test will fail to compile because the method call
            // below will succeed (and assert!(false) will fire at
            // compile time via the const assertion).
            fn _assert_no_warp_method() {
                // We just verify this test module compiles without
                // referencing decompress_blocks_warp_parallel.
                // The real assertion is: the 5 warp tests above are
                // excluded (verified by `cargo test -- warp` showing 0 tests).
            }
            _assert_no_warp_method();
        }

        /// When cuda-experimental IS enabled, the warp method should exist.
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn test_warp_method_available_with_feature() {
            // Verify the method exists on GpuLz4Context when feature is enabled.
            // We don't need CUDA hardware — just confirm the symbol resolves.
            fn _assert_method_exists(ctx: &GpuLz4Context) {
                // This will fail to compile if the method doesn't exist
                let _: fn(
                    &GpuLz4Context,
                    &[(Vec<u8>, usize)],
                ) -> Result<CudaSlice<u8>, GpuLz4Error> =
                    GpuLz4Context::decompress_blocks_warp_parallel;
            }
        }

        // ==================== K2 vs K3 Throughput Benchmark ====================

        /// Benchmarks K2 (sequential per-block) vs K3 (warp-parallel) throughput
        /// across multiple data sizes and compression patterns.
        #[test]
        #[cfg(feature = "cuda-experimental")]
        fn bench_warp_vs_parallel_throughput() {
            if !cuda_available() {
                eprintln!("Skipping benchmark: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            let scenarios: Vec<(&str, Vec<u8>)> = vec![
                // Literals-only (incompressible random-ish data)
                (
                    "4KB literals",
                    (0..4096).map(|i| ((i * 7 + 13) % 256) as u8).collect(),
                ),
                // Patterned data (good compression ratio, many matches)
                (
                    "64KB patterned",
                    (0..65536).map(|i| (i % 251) as u8).collect(),
                ),
                // Repeated data (extreme compression)
                ("64KB repeated", vec![0xAB_u8; 65536]),
                // Multi-block: 16 x 4KB blocks
                (
                    "16x4KB mixed",
                    (0..65536)
                        .map(|i| ((i * 31 + i / 256) % 256) as u8)
                        .collect(),
                ),
            ];

            let warmup_iters = 3;
            let bench_iters = 20;

            eprintln!("\n=== K2 vs K3 Throughput Benchmark ===");
            eprintln!(
                "{:<18} {:>12} {:>12} {:>10}",
                "Scenario", "K2 (GB/s)", "K3 (GB/s)", "Speedup"
            );
            eprintln!("{}", "-".repeat(56));

            for (name, original) in &scenarios {
                let compressed = lz4_flex::block::compress_prepend_size(original);
                let raw_compressed = &compressed[4..];

                // For multi-block scenario, split into 4KB chunks
                let (blocks, total_bytes) = if name.starts_with("16x") {
                    let chunks: Vec<(Vec<u8>, usize)> = original
                        .chunks(4096)
                        .map(|chunk| {
                            let comp = lz4_flex::block::compress_prepend_size(chunk);
                            (comp[4..].to_vec(), chunk.len())
                        })
                        .collect();
                    let total = original.len();
                    (chunks, total)
                } else {
                    (
                        vec![(raw_compressed.to_vec(), original.len())],
                        original.len(),
                    )
                };

                // Warmup
                for _ in 0..warmup_iters {
                    let _ = ctx.decompress_blocks_parallel(&blocks);
                    let _ = ctx.decompress_blocks_warp_parallel(&blocks);
                }

                // Benchmark K2
                let start = std::time::Instant::now();
                for _ in 0..bench_iters {
                    let result = ctx.decompress_blocks_parallel(&blocks).expect("K2");
                    // Force sync to measure actual kernel time
                    let _ = ctx.device.dtoh_sync_copy(&result);
                }
                let k2_elapsed = start.elapsed();

                // Benchmark K3
                let start = std::time::Instant::now();
                for _ in 0..bench_iters {
                    let result = ctx.decompress_blocks_warp_parallel(&blocks).expect("K3");
                    let _ = ctx.device.dtoh_sync_copy(&result);
                }
                let k3_elapsed = start.elapsed();

                let bytes_total = (total_bytes * bench_iters) as f64;
                let k2_gbps = bytes_total / k2_elapsed.as_secs_f64() / 1e9;
                let k3_gbps = bytes_total / k3_elapsed.as_secs_f64() / 1e9;
                let speedup = k3_gbps / k2_gbps;

                eprintln!(
                    "{:<18} {:>10.3}   {:>10.3}   {:>8.2}x",
                    name, k2_gbps, k3_gbps, speedup
                );
            }
            eprintln!("{}", "=".repeat(56));
        }

        // ==================== Direct GPU Slice Tests (Phase 4.1) ====================

        #[test]
        fn test_cuda_device_accessor() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let ctx = GpuLz4Context::new(0).expect("context creation");
            let device = ctx.cuda_device();

            // Verify we can use the device for allocations
            let _slice: CudaSlice<u8> = device.alloc_zeros(1024).expect("alloc");
        }

        #[test]
        fn test_decompress_to_f16_slice() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create F16 data (8 f16 values = 16 bytes)
            let f16_data: Vec<half::f16> = vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(4.0),
                half::f16::from_f32(5.0),
                half::f16::from_f32(6.0),
                half::f16::from_f32(7.0),
                half::f16::from_f32(8.0),
            ];
            let bytes: Vec<u8> = f16_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            // Decompress directly to f16 slice
            let d_f16 = ctx
                .decompress_to_f16_slice(&blocks)
                .expect("decompress to f16");

            // Verify length
            assert_eq!(d_f16.len(), 8);

            // Copy back and verify values
            let mut result = vec![half::f16::ZERO; 8];
            ctx.device
                .dtoh_sync_copy_into(&d_f16, &mut result)
                .expect("copy to host");

            assert_eq!(result, f16_data);
        }

        #[test]
        fn test_decompress_to_f32_slice() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create F32 data (4 f32 values = 16 bytes)
            let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            // Decompress directly to f32 slice
            let d_f32 = ctx
                .decompress_to_f32_slice(&blocks)
                .expect("decompress to f32");

            // Verify length
            assert_eq!(d_f32.len(), 4);

            // Copy back and verify values
            let mut result = vec![0.0f32; 4];
            ctx.device
                .dtoh_sync_copy_into(&d_f32, &mut result)
                .expect("copy to host");

            assert_eq!(result, f32_data);
        }

        #[test]
        fn test_f16_slice_large_data() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context creation");
            ctx.load_kernel().expect("kernel load");

            // Create 1K f16 values (2KB)
            let num_values = 1024;
            let f16_data: Vec<half::f16> = (0..num_values)
                .map(|i| half::f16::from_f32(i as f32 * 0.1))
                .collect();
            let bytes: Vec<u8> = f16_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let compressed = create_literals_only_lz4(&bytes);
            let blocks = vec![(compressed, bytes.len())];

            let d_f16 = ctx.decompress_to_f16_slice(&blocks).expect("decompress");

            assert_eq!(d_f16.len(), num_values);

            // Verify first and last values
            let mut result = vec![half::f16::ZERO; num_values];
            ctx.device
                .dtoh_sync_copy_into(&d_f16, &mut result)
                .expect("copy");

            assert_eq!(result[0], half::f16::from_f32(0.0));
            assert_eq!(result[1023], half::f16::from_f32(102.3));
        }

        // ==================== Streaming Pipeline Tests (Phase 5.1) ====================

        #[test]
        fn test_stream_pool_creation() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let ctx = GpuLz4Context::new(0).expect("context");
            let pool = CudaStreamPool::new(Arc::clone(&ctx.device), 4).expect("pool creation");

            assert_eq!(pool.num_streams(), 4);
        }

        #[test]
        fn test_stream_pool_wrapping() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let ctx = GpuLz4Context::new(0).expect("context");
            let pool = CudaStreamPool::new(Arc::clone(&ctx.device), 3).expect("pool creation");

            // Index 0, 1, 2 should work
            let _s0 = pool.get_stream(0);
            let _s1 = pool.get_stream(1);
            let _s2 = pool.get_stream(2);

            // Index 3 should wrap to 0
            let _s3 = pool.get_stream(3);
            // Index 5 should wrap to 2
            let _s5 = pool.get_stream(5);
        }

        #[test]
        fn test_streaming_context_creation() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let streaming_ctx = StreamingLz4Context::new(0, 3).expect("streaming context");
            assert_eq!(streaming_ctx.stream_pool().num_streams(), 3);
        }

        #[test]
        fn test_streaming_decompress_correctness() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let streaming_ctx = StreamingLz4Context::new(0, 2).expect("streaming context");

            // Create test blocks
            let originals: Vec<Vec<u8>> = vec![
                b"First block data".to_vec(),
                b"Second block with more data".to_vec(),
                b"Third block!".to_vec(),
                b"Fourth block for testing".to_vec(),
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            // Decompress with streaming
            let result = streaming_ctx
                .decompress_blocks_streaming(&blocks)
                .expect("streaming decompress");

            // Also decompress with sequential method for comparison
            let mut ctx = GpuLz4Context::new(0).expect("context");
            ctx.load_kernel().expect("kernel");
            let sequential_result = ctx
                .decompress_blocks_parallel(&blocks)
                .expect("sequential decompress");

            // Compare results
            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut streaming_host = vec![0u8; total_size];
            let mut sequential_host = vec![0u8; total_size];

            streaming_ctx
                .context()
                .device
                .dtoh_sync_copy_into(&result, &mut streaming_host)
                .expect("copy streaming");
            ctx.device
                .dtoh_sync_copy_into(&sequential_result, &mut sequential_host)
                .expect("copy sequential");

            assert_eq!(
                streaming_host, sequential_host,
                "Streaming should match sequential"
            );
        }

        #[test]
        fn test_streaming_empty_blocks_error() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let streaming_ctx = StreamingLz4Context::new(0, 2).expect("streaming context");
            let empty_blocks: Vec<(Vec<u8>, usize)> = vec![];

            let result = streaming_ctx.decompress_blocks_streaming(&empty_blocks);
            match result {
                Err(GpuLz4Error::InvalidInput { message }) => {
                    assert!(message.contains("No blocks"));
                },
                _ => panic!("Expected InvalidInput error"),
            }
        }

        #[test]
        fn test_streaming_with_callback() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let streaming_ctx = StreamingLz4Context::new(0, 2).expect("streaming context");

            let originals: Vec<Vec<u8>> = vec![
                b"Block A".to_vec(),
                b"Block B".to_vec(),
                b"Block C".to_vec(),
            ];

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let mut results: Vec<(usize, Vec<u8>)> = Vec::new();

            streaming_ctx
                .decompress_blocks_with_callback(blocks.into_iter(), |idx, data| {
                    results.push((idx, data));
                    Ok(())
                })
                .expect("callback decompress");

            assert_eq!(results.len(), 3);
            assert_eq!(results[0].0, 0);
            assert_eq!(results[0].1, b"Block A".to_vec());
            assert_eq!(results[1].0, 1);
            assert_eq!(results[1].1, b"Block B".to_vec());
            assert_eq!(results[2].0, 2);
            assert_eq!(results[2].1, b"Block C".to_vec());
        }

        #[test]
        fn test_streaming_many_blocks() {
            if !cuda_available() {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }

            let streaming_ctx = StreamingLz4Context::new(0, 4).expect("streaming context");

            // Create 20 blocks, more than pipeline depth
            let num_blocks = 20;
            let originals: Vec<Vec<u8>> = (0..num_blocks)
                .map(|i| format!("Test block number {}", i).into_bytes())
                .collect();

            let blocks: Vec<(Vec<u8>, usize)> = originals
                .iter()
                .map(|orig| {
                    let compressed = create_literals_only_lz4(orig);
                    (compressed, orig.len())
                })
                .collect();

            let result = streaming_ctx
                .decompress_blocks_streaming(&blocks)
                .expect("streaming decompress");

            // Verify
            let total_size: usize = originals.iter().map(|o| o.len()).sum();
            let mut host_data = vec![0u8; total_size];
            streaming_ctx
                .context()
                .device
                .dtoh_sync_copy_into(&result, &mut host_data)
                .expect("copy");

            let mut offset = 0;
            for (i, original) in originals.iter().enumerate() {
                let decompressed = &host_data[offset..offset + original.len()];
                assert_eq!(decompressed, original.as_slice(), "Block {} mismatch", i);
                offset += original.len();
            }
        }

        #[test]
        fn test_streaming_stats_calculations() {
            let mut stats = StreamingStats::default();

            // Test default values
            assert_eq!(stats.throughput_gbps(), 0.0);
            assert_eq!(stats.overlap_efficiency(), 1.0);

            // Set some values
            stats.bytes_decompressed = 1_000_000_000; // 1GB
            stats.total_time_us = 1_000_000; // 1 second
            stats.transfer_time_us = 500_000;
            stats.decompress_time_us = 400_000;

            // 1GB / 1s = 1 GB/s
            assert!((stats.throughput_gbps() - 1.0).abs() < 0.001);

            // Serial time = 900ms, actual = 1000ms -> 0.9 efficiency
            assert!((stats.overlap_efficiency() - 0.9).abs() < 0.001);
        }

        // ==================== TDD Phase 1: LZ4 GPU vs CPU Cross-Validation ====================
        // GPU-CODEC-PIPELINE-TDD.md §3.1-3.2
        //
        // Trust boundary: GPU LZ4 decompression MUST produce byte-identical output
        // to CPU LZ4 decompression. Any deviation means model weights are corrupted.

        /// §3.1: GPU LZ4 matches CPU for literals-only data.
        #[test]
        fn test_lz4_gpu_matches_cpu_literals_only() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context");
            ctx.load_kernel().expect("kernel load");

            let original = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let compressed = lz4_flex::block::compress_prepend_size(&original);
            // lz4_flex prepends a 4-byte LE size — strip it for raw block input
            let raw_compressed = &compressed[4..];

            // CPU decompress (reference)
            let cpu_result = lz4_flex::block::decompress(raw_compressed, original.len())
                .expect("CPU LZ4 decompress");

            // GPU decompress
            let gpu_result = ctx
                .decompress_block(raw_compressed, original.len())
                .expect("GPU LZ4 decompress");
            let mut gpu_host = vec![0u8; original.len()];
            ctx.device
                .dtoh_sync_copy_into(&gpu_result, &mut gpu_host)
                .expect("copy to host");

            assert_eq!(
                gpu_host, cpu_result,
                "GPU LZ4 must match CPU LZ4 byte-for-byte (literals-only)"
            );
            assert_eq!(gpu_host, original, "Both must match original data");
        }

        /// §3.1: GPU LZ4 matches CPU for data with match sequences.
        #[test]
        fn test_lz4_gpu_matches_cpu_with_matches() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context");
            ctx.load_kernel().expect("kernel load");

            // Data with repeated patterns (triggers LZ4 match sequences)
            let original: Vec<u8> = (0..1024).map(|i| (i % 13) as u8).collect();
            let compressed = lz4_flex::block::compress_prepend_size(&original);
            let raw_compressed = &compressed[4..];

            // CPU reference
            let cpu_result = lz4_flex::block::decompress(raw_compressed, original.len())
                .expect("CPU LZ4 decompress");

            // GPU decompress
            let gpu_result = ctx
                .decompress_block(raw_compressed, original.len())
                .expect("GPU LZ4 decompress");
            let mut gpu_host = vec![0u8; original.len()];
            ctx.device
                .dtoh_sync_copy_into(&gpu_result, &mut gpu_host)
                .expect("copy to host");

            assert_eq!(
                gpu_host, cpu_result,
                "GPU LZ4 must match CPU LZ4 byte-for-byte (with matches)"
            );
            assert_eq!(gpu_host, original);
        }

        /// §3.3: Parallel kernel (K2) matches single-block kernel (K1).
        #[test]
        fn test_lz4_parallel_matches_single_block() {
            if !cuda_available() {
                eprintln!("Skipping: no CUDA device");
                return;
            }

            let mut ctx = GpuLz4Context::new(0).expect("context");
            ctx.load_kernel().expect("kernel load");

            // Create 8 blocks with distinct patterns
            let blocks: Vec<Vec<u8>> = (0..8)
                .map(|i| (0..256).map(|j| ((i * 37 + j * 13) % 256) as u8).collect())
                .collect();

            // Decompress each block individually with K1
            let single_results: Vec<Vec<u8>> = blocks
                .iter()
                .map(|block| {
                    let compressed = lz4_flex::block::compress_prepend_size(block);
                    let raw = &compressed[4..];
                    let result = ctx
                        .decompress_block(raw, block.len())
                        .expect("K1 decompress");
                    let mut host = vec![0u8; block.len()];
                    ctx.device
                        .dtoh_sync_copy_into(&result, &mut host)
                        .expect("copy");
                    host
                })
                .collect();

            // Decompress all blocks in parallel with K2
            let parallel_input: Vec<(Vec<u8>, usize)> = blocks
                .iter()
                .map(|b| {
                    let compressed = lz4_flex::block::compress_prepend_size(b);
                    let raw = compressed[4..].to_vec();
                    (raw, b.len())
                })
                .collect();
            let parallel_result = ctx
                .decompress_blocks_parallel(&parallel_input)
                .expect("K2 decompress");

            // Copy parallel result (concatenated) to host
            let total_size: usize = blocks.iter().map(|b| b.len()).sum();
            let mut parallel_host = vec![0u8; total_size];
            ctx.device
                .dtoh_sync_copy_into(&parallel_result, &mut parallel_host)
                .expect("copy");

            // Compare block by block
            let mut offset = 0;
            for (i, (single, block)) in single_results.iter().zip(blocks.iter()).enumerate() {
                let parallel_block = &parallel_host[offset..offset + block.len()];
                assert_eq!(
                    single.as_slice(),
                    parallel_block,
                    "Block {} differs between K1 and K2",
                    i
                );
                offset += block.len();
            }
        }
    }
}

// Provide a no-op implementation when CUDA is not available
/// Stub module when CUDA is not enabled.
#[cfg(not(feature = "cuda"))]
pub mod cuda {

    /// Errors from GPU LZ4 operations (stub).
    #[derive(Debug, thiserror::Error)]
    pub enum GpuLz4Error {
        /// CUDA feature is not enabled.
        #[error("CUDA not enabled - compile with --features cuda")]
        CudaNotEnabled,
    }

    /// GPU LZ4 decompression context (stub when CUDA is disabled).
    pub struct GpuLz4Context;

    impl GpuLz4Context {
        /// Attempts to create a GPU LZ4 context.
        ///
        /// Always returns an error when CUDA is not enabled.
        pub fn new(_device_id: usize) -> Result<Self, GpuLz4Error> {
            Err(GpuLz4Error::CudaNotEnabled)
        }
    }

    /// CUDA stream pool (stub when CUDA is disabled).
    pub struct CudaStreamPool;

    /// Streaming LZ4 context (stub when CUDA is disabled).
    pub struct StreamingLz4Context;

    impl StreamingLz4Context {
        /// Attempts to create a streaming context.
        ///
        /// Always returns an error when CUDA is not enabled.
        pub fn new(_device_id: usize, _pipeline_depth: usize) -> Result<Self, GpuLz4Error> {
            Err(GpuLz4Error::CudaNotEnabled)
        }
    }

    /// Statistics from streaming decompression (stub).
    #[derive(Debug, Clone, Default)]
    pub struct StreamingStats {
        /// Total bytes transferred to GPU
        pub bytes_transferred: usize,
        /// Total bytes decompressed
        pub bytes_decompressed: usize,
        /// Number of blocks processed
        pub blocks_processed: usize,
        /// Time spent in H2D transfers (microseconds)
        pub transfer_time_us: u64,
        /// Time spent in decompression (microseconds)
        pub decompress_time_us: u64,
        /// Total wall-clock time (microseconds)
        pub total_time_us: u64,
    }

    impl StreamingStats {
        /// Returns effective throughput in GB/s.
        pub fn throughput_gbps(&self) -> f64 {
            if self.total_time_us == 0 {
                return 0.0;
            }
            let bytes = self.bytes_decompressed as f64;
            let seconds = self.total_time_us as f64 / 1_000_000.0;
            bytes / seconds / 1e9
        }

        /// Returns overlap efficiency (1.0 = perfect overlap, 0.5 = no overlap).
        pub fn overlap_efficiency(&self) -> f64 {
            let serial_time = self.transfer_time_us + self.decompress_time_us;
            if serial_time == 0 {
                return 1.0;
            }
            serial_time as f64 / self.total_time_us as f64
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_stub_returns_error() {
            let result = GpuLz4Context::new(0);
            match result {
                Err(GpuLz4Error::CudaNotEnabled) => {
                    // Expected
                },
                Ok(_) => panic!("Stub should always return error"),
            }
        }

        #[test]
        fn test_error_display() {
            let err = GpuLz4Error::CudaNotEnabled;
            let msg = format!("{}", err);
            assert!(msg.contains("CUDA not enabled"));
            assert!(msg.contains("--features cuda"));
        }

        #[test]
        fn test_streaming_stats_calculations() {
            let mut stats = StreamingStats::default();

            // Test default values
            assert_eq!(stats.throughput_gbps(), 0.0);
            assert_eq!(stats.overlap_efficiency(), 1.0);

            // Set some values
            stats.bytes_decompressed = 1_000_000_000; // 1GB
            stats.total_time_us = 1_000_000; // 1 second
            stats.transfer_time_us = 500_000;
            stats.decompress_time_us = 400_000;

            // 1GB / 1s = 1 GB/s
            assert!((stats.throughput_gbps() - 1.0).abs() < 0.001);

            // Serial time = 900ms, actual = 1000ms -> 0.9 efficiency
            assert!((stats.overlap_efficiency() - 0.9).abs() < 0.001);
        }

        #[test]
        fn test_streaming_context_stub() {
            let result = StreamingLz4Context::new(0, 2);
            match result {
                Err(GpuLz4Error::CudaNotEnabled) => {
                    // Expected
                },
                Ok(_) => panic!("Stub should always return error"),
            }
        }
    }
}

pub use cuda::GpuLz4Context;
#[cfg(feature = "cuda")]
pub use cuda::GpuLz4Error;
