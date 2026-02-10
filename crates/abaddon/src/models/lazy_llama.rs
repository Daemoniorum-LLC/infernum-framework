//! Lazy-loading Llama model for 405B+ inference on limited memory.
//!
//! Unlike the standard `Llama` model which loads all layers at init, `LazyLlama`
//! loads decoder layers on-demand during forward passes. This enables 405B inference
//! on systems with 24GB VRAM + 80GB RAM by keeping only a subset of layers loaded.
//!
//! ## Design
//!
//! ```text
//! Memory Layout During Inference:
//!
//! ┌─────────────────────────────────────────────────────┐
//! │ ALWAYS LOADED (embedding, norm, lm_head)   ~2GB    │
//! ├─────────────────────────────────────────────────────┤
//! │ LAYER WINDOW (N layers in memory)          ~N×6GB  │
//! │   Layer i-2  (will be evicted soon)                │
//! │   Layer i-1  (recently used)                       │
//! │   Layer i    (currently processing)                │
//! │   Layer i+1  (prefetched)                         │
//! │   Layer i+2  (prefetching)                        │
//! ├─────────────────────────────────────────────────────┤
//! │ ON DISK (remaining layers)                 ~700GB  │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use abaddon::models::lazy_llama::LazyLlama;
//! use abaddon::lazy_varbuilder::LazyVarBuilder;
//!
//! // Create lazy loader
//! let provider = TieredHoloLoader::new(hct_dir, config, device, dtype)?;
//! let lazy_vb = LazyVarBuilder::new(Arc::new(provider), device, dtype);
//!
//! // Load model (only embedding/norm/lm_head loaded initially)
//! let mut model = LazyLlama::load(config, lazy_vb, 12)?; // Keep 12 layers max
//!
//! // Forward pass loads layers on-demand
//! let logits = model.forward(&input_ids, 0)?;
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Embedding, Linear};

use crate::attention_cache::{
    attention_with_cache, CacheType, KvCache, KvCacheConfig, StandardCache,
};
use crate::hct::HctError;
use crate::lazy_varbuilder::LazyVarBuilder;

use super::llama::LlamaConfig;

/// External KV cache storage for layer eviction.
///
/// When layers are evicted from GPU to make room for others, their KV caches
/// are saved here on CPU. When layers are reloaded, their caches are restored.
///
/// This enables cross-forward-pass context preservation even with layer streaming.
///
/// # Memory Usage
///
/// For 70B (80 layers, 8 KV heads, 128 head_dim, BF16):
/// - Per layer per token: 2 * 8 * 128 * 2 = 4KB
/// - Per layer for 1024 tokens: 4MB
/// - All 80 layers for 1024 tokens: 320MB (easily fits in CPU RAM)
#[derive(Default)]
struct ExternalKvStore {
    /// Layer KV caches stored on CPU: layer_idx -> (K, V)
    caches: HashMap<usize, (Tensor, Tensor)>,
}

impl ExternalKvStore {
    fn new() -> Self {
        Self {
            caches: HashMap::new(),
        }
    }

    /// Save a layer's KV cache to CPU storage.
    fn save(&mut self, layer_idx: usize, k: Tensor, v: Tensor) -> CandleResult<()> {
        // Move tensors to CPU if they're on GPU
        let k_cpu = if k.device().is_cuda() {
            k.to_device(&Device::Cpu)?
        } else {
            k
        };
        let v_cpu = if v.device().is_cuda() {
            v.to_device(&Device::Cpu)?
        } else {
            v
        };
        self.caches.insert(layer_idx, (k_cpu, v_cpu));
        Ok(())
    }

    /// Restore a layer's KV cache from CPU storage to GPU.
    fn restore(
        &mut self,
        layer_idx: usize,
        device: &Device,
    ) -> CandleResult<Option<(Tensor, Tensor)>> {
        if let Some((k_cpu, v_cpu)) = self.caches.get(&layer_idx) {
            let k = k_cpu.to_device(device)?;
            let v = v_cpu.to_device(device)?;
            Ok(Some((k, v)))
        } else {
            Ok(None)
        }
    }

    /// Clear all stored caches.
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.caches.clear();
    }

    /// Get total memory usage in bytes.
    fn memory_bytes(&self) -> usize {
        self.caches
            .values()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
    }

    /// Get number of cached layers.
    fn len(&self) -> usize {
        self.caches.len()
    }
}

/// Lazy-loading Llama model for 405B+ inference.
pub struct LazyLlama {
    /// Token embedding (always loaded).
    embed_tokens: Embedding,
    /// Final layer norm (always loaded).
    norm: RmsNorm,
    /// LM head projection (always loaded).
    lm_head: Linear,
    /// Rotary embedding cache.
    rotary: RotaryEmbedding,
    /// Model configuration.
    config: LlamaConfig,
    /// Lazy VarBuilder for on-demand loading.
    lazy_vb: LazyVarBuilder,
    /// Currently loaded decoder layers (layer_idx -> layer).
    loaded_layers: HashMap<usize, DecoderLayer>,
    /// LRU order for layer eviction (most recent at end).
    lru_order: Vec<usize>,
    /// Maximum number of layers to keep loaded.
    max_loaded_layers: usize,
    /// KV cache type.
    cache_type: CacheType,
    /// Device.
    device: Device,
    /// Data type.
    dtype: DType,
    /// Layer load count (for stats).
    layer_loads: usize,
    /// Layer eviction count (for stats).
    layer_evictions: usize,
    /// Number of layers to prefetch ahead.
    prefetch_depth: usize,
    /// External KV cache storage for evicted layers.
    /// Caches are saved to CPU when layers are evicted and restored when reloaded.
    kv_store: ExternalKvStore,
}

impl LazyLlama {
    /// Loads a lazy Llama model.
    ///
    /// Only loads embedding, norm, and lm_head initially.
    /// Decoder layers are loaded on-demand during forward.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `lazy_vb` - Lazy VarBuilder for on-demand tensor loading
    /// * `max_loaded_layers` - Maximum number of decoder layers to keep in memory
    pub fn load(
        config: LlamaConfig,
        lazy_vb: LazyVarBuilder,
        max_loaded_layers: usize,
    ) -> Result<Self, LazyLoadError> {
        let device = lazy_vb.device().clone();
        let dtype = lazy_vb.dtype();

        tracing::info!(
            num_layers = config.num_hidden_layers,
            max_loaded_layers = max_loaded_layers,
            "Loading LazyLlama (layers loaded on-demand)"
        );

        // Load embedding
        let embed_tokens = Self::load_embedding(&config, &lazy_vb)?;

        // Load final norm
        let norm = Self::load_norm(
            config.hidden_size,
            config.rms_norm_eps,
            &lazy_vb,
            "model.norm",
        )?;

        // Load lm_head (or tie to embeddings)
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            Self::load_linear(
                &lazy_vb,
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                false,
            )?
        };

        // Create rotary embeddings
        let rotary = RotaryEmbedding::new(&config, dtype, &device)
            .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

        // Default prefetch depth of 2 layers (prefetch while processing current layer)
        let prefetch_depth = 2.min(max_loaded_layers.saturating_sub(1));

        tracing::info!(
            prefetch_depth = prefetch_depth,
            "LazyLlama base loaded (embedding, norm, lm_head). Layers will load on-demand."
        );

        Ok(Self {
            embed_tokens,
            norm,
            lm_head,
            rotary,
            config,
            lazy_vb,
            loaded_layers: HashMap::new(),
            lru_order: Vec::new(),
            max_loaded_layers,
            cache_type: CacheType::Standard,
            device,
            dtype,
            layer_loads: 0,
            layer_evictions: 0,
            prefetch_depth,
            kv_store: ExternalKvStore::new(),
        })
    }

    /// Sets the prefetch depth (number of layers to load ahead).
    ///
    /// Higher values reduce latency but use more memory.
    /// Recommended: 2 for balanced, 4 for low latency.
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth.min(self.max_loaded_layers.saturating_sub(1));
    }

    /// Sets the KV cache type for all layers.
    pub fn set_cache_type(&mut self, cache_type: CacheType) {
        self.cache_type = cache_type;
    }

    /// Forward pass with lazy layer loading.
    ///
    /// Layers are loaded on-demand and evicted via LRU when `max_loaded_layers` is exceeded.
    ///
    /// **Note on KV Cache behavior:**
    /// When layers are evicted and reloaded, their KV caches are destroyed. This means
    /// `start_pos` is used only for rotary position embeddings, not for mask construction.
    /// The mask is created based on the current sequence length only, as the cache is
    /// effectively empty after layer reload.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
    ) -> Result<Tensor, LazyLoadError> {
        let (_batch_size, seq_len) = input_ids
            .dims2()
            .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

        // Embed tokens
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

        // Create causal mask.
        // With external KV cache storage, caches persist across layer evictions.
        // Check the kv_store for the expected cache length to create the correct mask.
        //
        // The mask shape must be [seq_len, kv_len] where kv_len = cache_len + seq_len.
        // If there's no cached data, kv_len = seq_len (standard causal mask).
        //
        // Note: We check kv_store for layer 0's cache as representative of all layers.
        // All layers should have the same cache length after a forward pass.
        let stored_cache_len = self
            .kv_store
            .caches
            .get(&0)
            .map(|(k, _)| k.dims().get(2).copied().unwrap_or(0))
            .unwrap_or(0);
        let kv_len = stored_cache_len + seq_len;

        let mask = if seq_len > 1 {
            Some(
                Self::create_causal_mask_with_kv_len(seq_len, kv_len, &self.device, self.dtype)
                    .map_err(|e| LazyLoadError::Candle(e.to_string()))?,
            )
        } else {
            None
        };

        // Forward through layers (lazy loading)
        for layer_idx in 0..self.config.num_hidden_layers {
            // Ensure layer is loaded
            self.ensure_layer_loaded(layer_idx)?;

            // Get the layer and run forward
            let layer = self
                .loaded_layers
                .get_mut(&layer_idx)
                .expect("Layer should be loaded");

            hidden_states = layer
                .forward(&hidden_states, &self.rotary, mask.as_ref(), start_pos)
                .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

            // Update LRU
            self.touch_layer(layer_idx);

            // Prefetch upcoming layers (hides loading latency during compute)
            for prefetch_offset in 1..=self.prefetch_depth {
                let prefetch_idx = layer_idx + prefetch_offset;
                if prefetch_idx < self.config.num_hidden_layers {
                    // Load layer if not already loaded (ignore errors for prefetch)
                    let _ = self.ensure_layer_loaded(prefetch_idx);
                }
            }
        }

        // Final layer norm
        let hidden_states = self
            .norm
            .forward(&hidden_states)
            .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

        // Project to vocabulary
        self.lm_head
            .forward(&hidden_states)
            .map_err(|e| LazyLoadError::Candle(e.to_string()))
    }

    /// Ensures a layer is loaded, evicting LRU layers if necessary.
    ///
    /// Includes OOM recovery: if loading fails due to out-of-memory,
    /// we aggressively evict more layers and retry.
    fn ensure_layer_loaded(&mut self, layer_idx: usize) -> Result<(), LazyLoadError> {
        if self.loaded_layers.contains_key(&layer_idx) {
            return Ok(());
        }

        // Evict LRU layers if at capacity
        while self.loaded_layers.len() >= self.max_loaded_layers {
            self.evict_lru_layer();
        }

        // Try to load with OOM recovery
        let max_retries = 3;
        let mut total_evictions = 0;

        for attempt in 0..max_retries {
            // Clear CUDA cache before each attempt if not first try
            if attempt > 0 {
                self.force_memory_cleanup();
            }

            tracing::debug!(
                layer = layer_idx,
                attempt = attempt,
                "Loading decoder layer"
            );

            match self.load_decoder_layer(layer_idx) {
                Ok(mut layer) => {
                    // Restore KV cache from external store if available
                    if let Ok(Some((k, v))) = self.kv_store.restore(layer_idx, &self.device) {
                        layer.restore_kv(k, v);
                        tracing::debug!(
                            layer = layer_idx,
                            cache_seq_len = layer.self_attn.cache_len(),
                            "Restored KV cache from CPU store"
                        );
                    }

                    self.loaded_layers.insert(layer_idx, layer);
                    self.lru_order.push(layer_idx);
                    self.layer_loads += 1;
                    return Ok(());
                },
                Err(e) if Self::is_oom_error(&e) => {
                    let (cache_count, cache_bytes) = self.lazy_vb.cache_stats();
                    tracing::warn!(
                        layer = layer_idx,
                        attempt = attempt,
                        loaded_layers = self.loaded_layers.len(),
                        cache_tensors = cache_count,
                        cache_mb = cache_bytes / (1024 * 1024),
                        "OOM during layer load, evicting more layers and clearing tensor cache"
                    );

                    // Evict more layers aggressively (evict_lru_layer now clears tensor cache)
                    let evict_count = (self.loaded_layers.len() / 2).max(1);
                    for _ in 0..evict_count {
                        if self.loaded_layers.is_empty() {
                            break;
                        }
                        self.evict_lru_layer();
                        total_evictions += 1;
                    }

                    // Also clear KV caches from remaining layers
                    for layer in self.loaded_layers.values_mut() {
                        layer.clear_cache();
                    }

                    // If we've evicted all layers and still OOM, clear entire tensor cache
                    if self.loaded_layers.is_empty() {
                        let (cleared_count, cleared_bytes) = self.lazy_vb.clear_all();
                        tracing::warn!(
                            cleared_tensors = cleared_count,
                            cleared_mb = cleared_bytes / (1024 * 1024),
                            "Cleared entire tensor cache due to OOM"
                        );
                    }

                    // If we've evicted everything and still OOM, fail
                    if self.loaded_layers.is_empty() && attempt >= max_retries - 1 {
                        return Err(LazyLoadError::OutOfMemory {
                            layer: layer_idx,
                            attempts: attempt + 1,
                            evictions: total_evictions,
                        });
                    }
                },
                Err(e) => {
                    // Non-OOM error, don't retry
                    return Err(e);
                },
            }
        }

        Err(LazyLoadError::OutOfMemory {
            layer: layer_idx,
            attempts: max_retries,
            evictions: total_evictions,
        })
    }

    /// Checks if an error is an out-of-memory error.
    fn is_oom_error(e: &LazyLoadError) -> bool {
        let msg = e.to_string().to_uppercase();
        msg.contains("OUT_OF_MEMORY")
            || msg.contains("OOM")
            || msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
            || msg.contains("INSUFFICIENT MEMORY")
            || msg.contains("ALLOC")
    }

    /// Forces memory cleanup by dropping caches and waiting for deferred deallocations.
    fn force_memory_cleanup(&mut self) {
        // Clear all KV caches to free VRAM
        for layer in self.loaded_layers.values_mut() {
            layer.clear_cache();
        }

        // Drop scope to trigger tensor deallocations
        // The sleeping allows CUDA's deferred memory operations to complete
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    /// Evicts the least recently used layer from VRAM.
    ///
    /// With the CPU caching architecture (TieredHoloLoader), this ONLY drops the
    /// GPU DecoderLayer struct. The underlying tensor data remains cached on CPU
    /// for fast reload (~100ms CPU→GPU transfer) instead of slow HCT decompression (~30s).
    ///
    /// NOTE: We intentionally do NOT call clear_prefix() here anymore.
    /// The TieredHoloLoader now caches tensors on CPU RAM, not GPU VRAM.
    /// GPU memory is freed when the DecoderLayer struct is dropped.
    fn evict_lru_layer(&mut self) {
        if let Some(layer_idx) = self.lru_order.first().copied() {
            // Save KV cache to CPU before evicting
            if let Some(layer) = self.loaded_layers.get(&layer_idx) {
                if let Ok(Some((k, v))) = layer.extract_kv() {
                    if let Err(e) = self.kv_store.save(layer_idx, k, v) {
                        tracing::warn!(
                            layer = layer_idx,
                            error = %e,
                            "Failed to save KV cache before eviction"
                        );
                    } else {
                        tracing::debug!(
                            layer = layer_idx,
                            kv_store_layers = self.kv_store.len(),
                            kv_store_mb = self.kv_store.memory_bytes() / (1024 * 1024),
                            "Saved KV cache to CPU before eviction"
                        );
                    }
                }
            }

            tracing::debug!(
                layer = layer_idx,
                "Evicting decoder layer (GPU freed, KV cache preserved on CPU)"
            );

            // Remove from loaded_layers - this drops the DecoderLayer struct
            // which frees GPU tensors. KV cache is now in kv_store.
            self.loaded_layers.remove(&layer_idx);
            self.lru_order.remove(0);
            self.layer_evictions += 1;
        }
    }

    /// Updates LRU order (moves layer to end).
    fn touch_layer(&mut self, layer_idx: usize) {
        if let Some(pos) = self.lru_order.iter().position(|&x| x == layer_idx) {
            self.lru_order.remove(pos);
            self.lru_order.push(layer_idx);
        }
    }

    /// Loads a decoder layer from the lazy VarBuilder.
    fn load_decoder_layer(&self, layer_idx: usize) -> Result<DecoderLayer, LazyLoadError> {
        let prefix = format!("model.layers.{}", layer_idx);
        let vb = self.lazy_vb.pp(&prefix);

        DecoderLayer::load(&self.config, vb, self.cache_type.clone())
    }

    /// Clears all KV caches.
    pub fn clear_cache(&mut self) {
        for layer in self.loaded_layers.values_mut() {
            layer.clear_cache();
        }
    }

    /// Prefetches initial layers to hide latency at inference start.
    ///
    /// Call this before the first forward pass to warm up the model.
    /// Loads up to `max_loaded_layers` layers starting from layer 0.
    ///
    /// Returns the number of layers successfully prefetched.
    pub fn warmup(&mut self) -> usize {
        let layers_to_load = self.max_loaded_layers.min(self.config.num_hidden_layers);

        tracing::info!(
            layers = layers_to_load,
            "Warming up LazyLlama by prefetching initial layers"
        );

        let mut loaded = 0;
        for layer_idx in 0..layers_to_load {
            match self.ensure_layer_loaded(layer_idx) {
                Ok(()) => loaded += 1,
                Err(e) => {
                    tracing::warn!(layer = layer_idx, error = %e, "Warmup prefetch failed");
                    break;
                },
            }
        }

        tracing::info!(layers_loaded = loaded, "Warmup complete");
        loaded
    }

    /// Prefetches layers for a specific layer range (useful for multi-pass inference).
    ///
    /// Returns the number of layers successfully prefetched.
    pub fn prefetch_layers(&mut self, start_layer: usize, count: usize) -> usize {
        let end_layer = (start_layer + count).min(self.config.num_hidden_layers);

        let mut loaded = 0;
        for layer_idx in start_layer..end_layer {
            if let Ok(()) = self.ensure_layer_loaded(layer_idx) {
                loaded += 1;
            }
        }
        loaded
    }

    /// Returns statistics about layer loading.
    pub fn stats(&self) -> LazyStats {
        LazyStats {
            total_layers: self.config.num_hidden_layers,
            loaded_layers: self.loaded_layers.len(),
            max_loaded_layers: self.max_loaded_layers,
            layer_loads: self.layer_loads,
            layer_evictions: self.layer_evictions,
            prefetch_depth: self.prefetch_depth,
        }
    }

    /// Returns the current cache sequence length.
    pub fn cache_seq_len(&self) -> usize {
        self.loaded_layers
            .values()
            .next()
            .map_or(0, |l| l.self_attn.cache_len())
    }

    /// Returns total cache memory usage.
    pub fn cache_memory_bytes(&self) -> usize {
        self.loaded_layers
            .values()
            .map(|l| l.self_attn.cache_memory_bytes())
            .sum()
    }

    // ==================== Loading Helpers ====================

    fn load_embedding(
        config: &LlamaConfig,
        vb: &LazyVarBuilder,
    ) -> Result<Embedding, LazyLoadError> {
        let weight = vb.get("model.embed_tokens.weight")?;
        Ok(Embedding::new(weight, config.hidden_size))
    }

    fn load_norm(
        _size: usize,
        eps: f64,
        vb: &LazyVarBuilder,
        prefix: &str,
    ) -> Result<RmsNorm, LazyLoadError> {
        let weight = vb.get(&format!("{}.weight", prefix))?;
        Ok(RmsNorm { weight, eps })
    }

    fn load_linear(
        vb: &LazyVarBuilder,
        name: &str,
        _in_dim: usize,
        _out_dim: usize,
        bias: bool,
    ) -> Result<Linear, LazyLoadError> {
        let weight = vb.get(&format!("{}.weight", name))?;
        let bias_tensor = if bias {
            Some(vb.get(&format!("{}.bias", name))?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias_tensor))
    }

    #[allow(dead_code)]
    fn create_causal_mask(
        seq_len: usize,
        start_pos: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        Self::create_causal_mask_with_kv_len(seq_len, seq_len + start_pos, device, dtype)
    }

    /// Creates a causal mask with explicit KV length.
    ///
    /// This is needed for lazy layer loading where the KV cache may not match `start_pos`
    /// due to layer eviction destroying caches.
    ///
    /// # Arguments
    /// * `seq_len` - Query sequence length (new tokens)
    /// * `kv_len` - Total key/value sequence length (cache + new tokens)
    /// * `device` - Device for the tensor
    /// * `dtype` - Data type for the tensor
    ///
    /// # Shape
    /// Returns tensor of shape `[seq_len, kv_len]` where each query position i
    /// can attend to key positions 0 through (kv_len - seq_len + i), with
    /// positions beyond that masked with -inf.
    fn create_causal_mask_with_kv_len(
        seq_len: usize,
        kv_len: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        // For lazy loading with evicted caches:
        // - seq_len = number of new tokens (e.g., 6)
        // - kv_len = actual cache size + new tokens (may be just seq_len if cache is empty)
        //
        // When kv_len == seq_len (cache was empty/evicted):
        //   The mask is a standard lower triangular matrix.
        //   Each query position i can attend to positions 0..=i.
        //
        // When kv_len > seq_len (cache has previous tokens):
        //   Query position i can attend to:
        //   - All cached positions (0..kv_len-seq_len)
        //   - Current positions 0..=i within the new tokens
        //
        // Unified formula: query i can attend to key j where j <= cache_len + i

        let cache_len = kv_len.saturating_sub(seq_len);
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > cache_len + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (seq_len, kv_len), device)?.to_dtype(dtype)
    }
}

/// Statistics for lazy loading.
#[derive(Debug, Clone)]
pub struct LazyStats {
    /// Total number of layers in the model.
    pub total_layers: usize,
    /// Currently loaded layers.
    pub loaded_layers: usize,
    /// Maximum layers allowed in memory.
    pub max_loaded_layers: usize,
    /// Total number of layer loads.
    pub layer_loads: usize,
    /// Total number of layer evictions.
    pub layer_evictions: usize,
    /// Number of layers prefetched ahead during inference.
    pub prefetch_depth: usize,
}

/// Error type for lazy loading.
#[derive(Debug, thiserror::Error)]
pub enum LazyLoadError {
    /// HCT file loading error.
    #[error("HCT loading error: {0}")]
    Hct(#[from] HctError),
    /// Candle tensor operation error.
    #[error("Candle error: {0}")]
    Candle(String),
    /// Requested layer not found.
    #[error("Layer not found: {0}")]
    LayerNotFound(usize),
    /// Out of memory error after exhausting recovery attempts.
    #[error(
        "Out of memory: layer {layer} failed after {attempts} retries with {evictions} evictions"
    )]
    OutOfMemory {
        /// Layer that failed to load.
        layer: usize,
        /// Number of retry attempts made.
        attempts: usize,
        /// Number of layers evicted during recovery.
        evictions: usize,
    },
}

impl From<candle_core::Error> for LazyLoadError {
    fn from(e: candle_core::Error) -> Self {
        LazyLoadError::Candle(e.to_string())
    }
}

// ==================== Internal Components ====================

/// RMS Layer Normalization.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// Rotary Position Embedding.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &LlamaConfig, dtype: DType, device: &Device) -> CandleResult<Self> {
        let head_dim = config.head_dim();
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        // Compute base inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();

        // Apply Llama3 RoPE scaling if configured
        let inv_freq = if let Some(ref scaling) = config.rope_scaling {
            if scaling.rope_type.as_deref() == Some("llama3") {
                Self::apply_llama3_scaling(&inv_freq, scaling)
            } else if let Some(factor) = scaling.factor {
                // Linear scaling: simply divide frequencies
                inv_freq.iter().map(|f| f / factor as f32).collect()
            } else {
                inv_freq
            }
        } else {
            inv_freq
        };

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply Llama3-style RoPE scaling.
    ///
    /// Llama3 uses a smooth interpolation between scaled and unscaled frequencies
    /// based on wavelength thresholds.
    fn apply_llama3_scaling(
        inv_freq: &[f32],
        scaling: &super::llama::RopeScalingConfig,
    ) -> Vec<f32> {
        let factor = scaling.factor.unwrap_or(1.0) as f32;
        let low_freq_factor = scaling.low_freq_factor.unwrap_or(1.0) as f32;
        let high_freq_factor = scaling.high_freq_factor.unwrap_or(4.0) as f32;
        let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(8192) as f32;

        // Wavelength thresholds
        let low_freq_wavelen = orig_max_pos / low_freq_factor;
        let high_freq_wavelen = orig_max_pos / high_freq_factor;

        inv_freq
            .iter()
            .map(|&freq| {
                // wavelength = 2π / freq (but freq is already 1/theta^(i/dim), so wavelen ∝ 1/freq)
                let wavelen = 2.0 * std::f32::consts::PI / freq;

                if wavelen > low_freq_wavelen {
                    // Low frequency region: scale down by factor
                    freq / factor
                } else if wavelen < high_freq_wavelen {
                    // High frequency region: keep original
                    freq
                } else {
                    // Middle region: smooth interpolation
                    let smooth = (orig_max_pos / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * freq / factor + smooth * freq
                }
            })
            .collect()
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> CandleResult<(Tensor, Tensor)> {
        // q shape: [batch, num_heads, seq_len, head_dim]
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_embed = Self::rotate(q, &cos, &sin)?;
        let k_embed = Self::rotate(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // x shape: [batch, num_heads, seq_len, head_dim]
        // cos/sin shape: [seq_len, head_dim]
        // Need cos/sin to be [1, 1, seq_len, head_dim] for broadcasting
        let x1 = x.narrow(D::Minus1, 0, x.dim(D::Minus1)? / 2)?;
        let x2 = x.narrow(D::Minus1, x.dim(D::Minus1)? / 2, x.dim(D::Minus1)? / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        // [seq_len, head_dim] -> [1, seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?
    }
}

/// MLP layer.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(
        config: &LlamaConfig,
        vb: &LazyVarBuilder,
        prefix: &str,
    ) -> Result<Self, LazyLoadError> {
        let gate_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.gate_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let up_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.up_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let down_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.down_proj", prefix),
            config.intermediate_size,
            config.hidden_size,
            false,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }
}

/// Attention layer.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: Box<dyn KvCache>,
}

impl Attention {
    fn load(
        config: &LlamaConfig,
        vb: &LazyVarBuilder,
        prefix: &str,
        cache_type: CacheType,
    ) -> Result<Self, LazyLoadError> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.q_proj", prefix),
            config.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.k_proj", prefix),
            config.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.v_proj", prefix),
            config.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let o_proj = LazyLlama::load_linear(
            vb,
            &format!("{}.o_proj", prefix),
            num_heads * head_dim,
            config.hidden_size,
            false,
        )?;

        let cache_config = KvCacheConfig {
            num_kv_heads,
            head_dim,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        };
        let cache = cache_type
            .create(&cache_config)
            .map_err(|e| LazyLoadError::Candle(e.to_string()))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            cache,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Attention with cache
        let attn_output = attention_with_cache(
            &q,
            &k,
            &v,
            self.cache.as_mut(),
            self.num_heads,
            self.num_kv_heads,
            mask,
        )?;

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&attn_output)
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn cache_len(&self) -> usize {
        self.cache.seq_len()
    }

    fn cache_memory_bytes(&self) -> usize {
        self.cache.memory_bytes()
    }

    /// Extract KV cache tensors (for saving before eviction).
    fn extract_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        self.cache.get_kv()
    }

    /// Restore KV cache from saved tensors.
    fn restore_kv(&mut self, k: Tensor, v: Tensor) {
        // Replace the cache with a new one containing the restored KV
        self.cache = Box::new(StandardCache::with_kv(k, v));
    }
}

/// Decoder layer.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(
        config: &LlamaConfig,
        vb: LazyVarBuilder,
        cache_type: CacheType,
    ) -> Result<Self, LazyLoadError> {
        let self_attn = Attention::load(config, &vb, "self_attn", cache_type)?;
        let mlp = Mlp::load(config, &vb, "mlp")?;
        let input_layernorm = LazyLlama::load_norm(
            config.hidden_size,
            config.rms_norm_eps,
            &vb,
            "input_layernorm",
        )?;
        let post_attention_layernorm = LazyLlama::load_norm(
            config.hidden_size,
            config.rms_norm_eps,
            &vb,
            "post_attention_layernorm",
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> CandleResult<Tensor> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rotary, mask, start_pos)?;
        let x = (residual + x)?;

        // MLP with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }

    /// Extract KV cache tensors (for saving before eviction).
    fn extract_kv(&self) -> CandleResult<Option<(Tensor, Tensor)>> {
        self.self_attn.extract_kv()
    }

    /// Restore KV cache from saved tensors.
    fn restore_kv(&mut self, k: Tensor, v: Tensor) {
        self.self_attn.restore_kv(k, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockTensorProvider {
        tensors: std::sync::RwLock<HashMap<String, Tensor>>,
    }

    impl MockTensorProvider {
        fn new() -> Self {
            Self {
                tensors: std::sync::RwLock::new(HashMap::new()),
            }
        }

        fn add(&self, name: &str, shape: &[usize], device: &Device) {
            let data: Vec<f32> = (0..shape.iter().product::<usize>())
                .map(|i| (i as f32) * 0.001)
                .collect();
            let tensor = Tensor::from_vec(data, shape, device).unwrap();
            self.tensors
                .write()
                .unwrap()
                .insert(name.to_string(), tensor);
        }
    }

    impl crate::lazy_varbuilder::TensorProvider for MockTensorProvider {
        fn get(&self, name: &str, _device: &Device, _dtype: DType) -> Result<Tensor, HctError> {
            self.tensors
                .read()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(|| HctError::Tensor {
                    message: format!("Tensor not found: {}", name),
                })
        }

        fn contains(&self, name: &str) -> bool {
            self.tensors.read().unwrap().contains_key(name)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.read().unwrap().keys().cloned().collect()
        }
    }

    #[test]
    fn test_lazy_stats() {
        let stats = LazyStats {
            total_layers: 126,
            loaded_layers: 12,
            max_loaded_layers: 12,
            layer_loads: 50,
            layer_evictions: 38,
            prefetch_depth: 2,
        };
        assert_eq!(stats.total_layers, 126);
        assert_eq!(stats.loaded_layers, 12);
    }
}
