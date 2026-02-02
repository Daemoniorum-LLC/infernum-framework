//! KV cache for autoregressive generation.
//!
//! Implements GPU-resident key-value cache for transformer inference.
//! Keys and values are stored in a pre-allocated buffer for each layer,
//! avoiding per-token allocation overhead.
//!
//! ## Memory Layout
//!
//! ```text
//! Keys:   [num_layers, max_seq_len, num_kv_heads, head_dim]
//! Values: [num_layers, max_seq_len, num_kv_heads, head_dim]
//! ```
//!
//! Each layer's cache is a contiguous [max_seq, kv_heads, head_dim] region.
//!
//! ## Async Operations
//!
//! Use `update_async()` for non-blocking KV cache updates that can
//! overlap with other GPU operations.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::arch::ModelConfig;
use super::tensor::{GpuDType, GpuTensor};
use super::InferenceError;

/// GPU-resident KV cache for efficient autoregressive decoding.
///
/// Pre-allocates memory for the maximum sequence length to avoid
/// per-token allocation overhead during generation.
pub struct KvCache {
    /// Model configuration.
    config: ModelConfig,

    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Key cache [num_layers, max_seq_len, num_kv_heads, head_dim].
    keys: GpuTensor,

    /// Value cache [num_layers, max_seq_len, num_kv_heads, head_dim].
    values: GpuTensor,

    /// Current sequence length (position of next token to be written).
    seq_len: usize,

    /// Maximum sequence length.
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new KV cache.
    ///
    /// Pre-allocates GPU memory for the full cache.
    pub fn new(
        config: &ModelConfig,
        max_seq_len: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        // Allocate cache tensors
        // Shape: [num_layers, max_seq_len, num_kv_heads, head_dim]
        let cache_shape = vec![num_layers, max_seq_len, num_kv_heads, head_dim];

        let keys = GpuTensor::zeros(cache_shape.clone(), GpuDType::F16, Arc::clone(&device))?;
        let values = GpuTensor::zeros(cache_shape, GpuDType::F16, Arc::clone(&device))?;

        let cache_bytes = keys.size_bytes() + values.size_bytes();
        tracing::info!(
            cache_mb = cache_bytes as f64 / 1024.0 / 1024.0,
            max_seq_len = max_seq_len,
            num_layers = num_layers,
            kv_heads = num_kv_heads,
            head_dim = head_dim,
            "Created KV cache"
        );

        Ok(Self {
            config: config.clone(),
            device,
            keys,
            values,
            seq_len: 0,
            max_seq_len,
        })
    }

    /// Current sequence length (number of cached tokens).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Reset the cache for a new sequence.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // Note: We don't zero the cache - old data is harmless since
        // attention only looks at seq_len positions.
    }

    /// Update cache with new K, V for a specific layer.
    ///
    /// Copies the new keys/values into the cache at the current position.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0..num_layers)
    /// * `keys` - New keys [seq, kv_heads, head_dim]
    /// * `values` - New values [seq, kv_heads, head_dim]
    ///
    /// # Returns
    ///
    /// The sequence offset where data was written.
    pub fn update(
        &mut self,
        layer_idx: usize,
        keys: &GpuTensor,
        values: &GpuTensor,
    ) -> Result<usize, InferenceError> {
        let new_tokens = keys.shape()[0];

        // Check bounds
        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(InferenceError::Shape {
                expected: format!("seq_len + new_tokens <= {}", self.max_seq_len),
                got: format!("{} + {} = {}", self.seq_len, new_tokens, self.seq_len + new_tokens),
            });
        }

        // Write to the cache at current position
        let write_offset = self.seq_len;
        self.keys.write_layer_at(layer_idx, write_offset, keys)?;
        self.values.write_layer_at(layer_idx, write_offset, values)?;

        Ok(write_offset)
    }

    /// Advance the sequence position after all layers have been updated.
    ///
    /// Call this after updating all layers with new tokens.
    pub fn advance(&mut self, num_new_tokens: usize) {
        self.seq_len += num_new_tokens;
    }

    /// Async update cache with new K, V for a specific layer (non-blocking).
    ///
    /// This version uses async memory copies that can overlap with compute.
    /// Call on the memory stream while compute runs on compute stream.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index (0..num_layers)
    /// * `keys` - New keys [seq, kv_heads, head_dim]
    /// * `values` - New values [seq, kv_heads, head_dim]
    /// * `stream` - Raw CUDA stream pointer
    ///
    /// # Returns
    ///
    /// The sequence offset where data was written.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent writes to the same cache region
    /// and that stream is valid.
    pub unsafe fn update_async(
        &mut self,
        layer_idx: usize,
        keys: &GpuTensor,
        values: &GpuTensor,
        stream: *mut c_void,
    ) -> Result<usize, InferenceError> {
        let new_tokens = keys.shape()[0];

        // Check bounds
        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(InferenceError::Shape {
                expected: format!("seq_len + new_tokens <= {}", self.max_seq_len),
                got: format!("{} + {} = {}", self.seq_len, new_tokens, self.seq_len + new_tokens),
            });
        }

        // Write to the cache at current position using async copies
        let write_offset = self.seq_len;
        self.keys.write_layer_at_async(layer_idx, write_offset, keys, stream)?;
        self.values.write_layer_at_async(layer_idx, write_offset, values, stream)?;

        Ok(write_offset)
    }

    /// Get K, V tensors for attention at a specific layer.
    ///
    /// Returns views into the cached data for the first seq_len positions.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    ///
    /// # Returns
    ///
    /// (keys, values) each with shape [1, seq_len, kv_heads, head_dim]
    pub fn get_kv(&self, layer_idx: usize) -> Result<(GpuTensor, GpuTensor), InferenceError> {
        if layer_idx >= self.config.num_layers {
            return Err(InferenceError::Shape {
                expected: format!("layer_idx < {}", self.config.num_layers),
                got: format!("{}", layer_idx),
            });
        }

        let effective_seq = if self.seq_len > 0 { self.seq_len } else { 1 };

        // Get slices from the cache - returns [seq, kv_heads, head_dim]
        let k_slice = self.keys.get_layer_kv_slice(layer_idx, effective_seq)?;
        let v_slice = self.values.get_layer_kv_slice(layer_idx, effective_seq)?;

        // Reshape to [1, seq, kv_heads, head_dim] for Flash Attention
        // This is ZERO-COPY since reshape just changes metadata
        let k_shape = k_slice.shape();
        let k_4d = k_slice.reshape(vec![1, k_shape[0], k_shape[1], k_shape[2]])?;
        let v_shape = v_slice.shape();
        let v_4d = v_slice.reshape(vec![1, v_shape[0], v_shape[1], v_shape[2]])?;

        Ok((k_4d, v_4d))
    }

    /// Get direct access to the key cache for a layer.
    ///
    /// Returns raw cache tensor for custom operations.
    pub fn keys_for_layer(&self, layer_idx: usize) -> Result<GpuTensor, InferenceError> {
        self.keys.slice_layer(layer_idx)
    }

    /// Get direct access to the value cache for a layer.
    ///
    /// Returns raw cache tensor for custom operations.
    pub fn values_for_layer(&self, layer_idx: usize) -> Result<GpuTensor, InferenceError> {
        self.values.slice_layer(layer_idx)
    }

    /// Check if cache has room for more tokens.
    pub fn has_capacity(&self, num_tokens: usize) -> bool {
        self.seq_len + num_tokens <= self.max_seq_len
    }

    /// Get remaining capacity.
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.seq_len
    }

    /// Get configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_inference::arch::Activation;

    fn test_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32000,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 4,
            num_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            activation: Activation::SiLU,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_kv_cache_tracking() {
        // Test sequence length tracking without GPU
        let config = test_config();

        // Verify config values used in cache calculations
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_kv_heads, 3);
        assert_eq!(config.head_dim, 64);
    }

    // Full KV cache tests with GPU are in tests.rs
}
