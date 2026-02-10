//! Lazy-loading Qwen2 model for 14B+ inference on limited memory.
//!
//! Unlike the standard `Qwen2` model which loads all layers at init, `LazyQwen2`
//! loads decoder layers on-demand during forward passes. This enables 14B-70B inference
//! on systems with 24GB VRAM by keeping only a subset of layers loaded.
//!
//! ## Key Differences from LazyLlama
//!
//! Qwen2 has several architectural differences:
//! - Q/K/V projections have bias (Llama doesn't)
//! - Different rotary embedding implementation
//! - Uses 1e-6 for rms_norm_eps (vs 1e-5 for Llama)
//! - Uses 1M rope_theta (vs 500K for Llama)
//!
//! ## Usage
//!
//! ```ignore
//! use abaddon::models::lazy_qwen2::LazyQwen2;
//! use abaddon::lazy_varbuilder::LazyVarBuilder;
//!
//! // Create lazy loader
//! let provider = TieredHoloLoader::new(hct_dir, config, device, dtype)?;
//! let lazy_vb = LazyVarBuilder::new(Arc::new(provider), device, dtype);
//!
//! // Load model (only embedding/norm/lm_head loaded initially)
//! let mut model = LazyQwen2::load(config, lazy_vb, 8)?; // Keep 8 layers max
//!
//! // Forward pass loads layers on-demand
//! let logits = model.forward(&input_ids, 0)?;
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{Embedding, Linear};

use crate::attention_cache::{attention_with_cache, CacheType, KvCache, KvCacheConfig};
use crate::hct::HctError;
use crate::lazy_varbuilder::LazyVarBuilder;

use super::qwen2::Qwen2Config;

/// Lazy-loading Qwen2 model for 14B+ inference.
pub struct LazyQwen2 {
    /// Token embedding (always loaded).
    embed_tokens: Embedding,
    /// Final layer norm (always loaded).
    norm: RmsNorm,
    /// LM head projection (always loaded).
    lm_head: Linear,
    /// Rotary embedding cache.
    rotary: RotaryEmbedding,
    /// Model configuration.
    config: Qwen2Config,
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
    /// Persisted KV caches for evicted layers (survives layer eviction).
    /// This is critical: when a layer is evicted, its cache is stored here
    /// so it can be restored when the layer is reloaded.
    layer_caches: HashMap<usize, Box<dyn KvCache>>,
}

impl LazyQwen2 {
    /// Loads a lazy Qwen2 model.
    ///
    /// Only loads embedding, norm, and lm_head initially.
    /// Decoder layers are loaded on-demand during forward.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `lazy_vb` - Lazy VarBuilder for on-demand tensor loading
    /// * `max_loaded_layers` - Maximum number of decoder layers to keep in memory
    pub fn load(
        config: Qwen2Config,
        lazy_vb: LazyVarBuilder,
        max_loaded_layers: usize,
    ) -> Result<Self, LazyLoadError> {
        let device = lazy_vb.device().clone();
        let dtype = lazy_vb.dtype();

        tracing::info!(
            num_layers = config.num_hidden_layers,
            max_loaded_layers = max_loaded_layers,
            "Loading LazyQwen2 (layers loaded on-demand)"
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
            "LazyQwen2 base loaded (embedding, norm, lm_head). Layers will load on-demand."
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
            layer_caches: HashMap::new(),
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

        // Create causal mask
        let mask = if seq_len > 1 {
            Some(
                Self::create_causal_mask(seq_len, start_pos, &self.device, self.dtype)
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
                    // Check if there's a persisted cache for this layer and restore it
                    if let Some(cache) = self.layer_caches.remove(&layer_idx) {
                        tracing::debug!(
                            layer = layer_idx,
                            cache_seq_len = cache.seq_len(),
                            "Restoring persisted KV cache for reloaded layer"
                        );
                        layer.set_cache(cache);
                    }

                    self.loaded_layers.insert(layer_idx, layer);
                    self.lru_order.push(layer_idx);
                    self.layer_loads += 1;
                    return Ok(());
                },
                Err(e) if Self::is_oom_error(&e) => {
                    tracing::warn!(
                        layer = layer_idx,
                        attempt = attempt,
                        loaded_layers = self.loaded_layers.len(),
                        "OOM during layer load, evicting more layers"
                    );

                    // Evict more layers aggressively
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

    /// Evicts the least recently used layer.
    ///
    /// IMPORTANT: The KV cache is extracted and stored in `layer_caches` before
    /// the layer is evicted. This ensures cache continuity across layer evictions.
    fn evict_lru_layer(&mut self) {
        if let Some(layer_idx) = self.lru_order.first().copied() {
            tracing::debug!(
                layer = layer_idx,
                "Evicting decoder layer (preserving KV cache)"
            );

            // Extract and preserve the KV cache before evicting the layer
            if let Some(mut layer) = self.loaded_layers.remove(&layer_idx) {
                // Take the cache from the layer and store it
                let cache = layer.self_attn.take_cache();
                if cache.seq_len() > 0 {
                    tracing::debug!(
                        layer = layer_idx,
                        cache_seq_len = cache.seq_len(),
                        "Preserved KV cache for evicted layer"
                    );
                    self.layer_caches.insert(layer_idx, cache);
                }
            }

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

    /// Clears all KV caches (both loaded and persisted).
    pub fn clear_cache(&mut self) {
        // Clear caches in loaded layers
        for layer in self.loaded_layers.values_mut() {
            layer.clear_cache();
        }
        // Clear persisted caches for evicted layers
        self.layer_caches.clear();
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
            "Warming up LazyQwen2 by prefetching initial layers"
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

    /// Returns the model configuration.
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Returns the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // ==================== Loading Helpers ====================

    fn load_embedding(
        config: &Qwen2Config,
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

    fn create_causal_mask(
        seq_len: usize,
        start_pos: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len + start_pos).map(move |j| {
                    if j > i + start_pos {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len + start_pos), device)?.to_dtype(dtype)
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

/// Rotary Position Embedding (Qwen2 style).
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Qwen2Config, dtype: DType, device: &Device) -> CandleResult<Self> {
        let head_dim = config.head_dim();
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> CandleResult<(Tensor, Tensor)> {
        // q shape: [batch, num_heads, seq_len, head_dim]
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        // Double cos/sin to match head_dim (neox-style rotary)
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;

        let q_embed = Self::apply_rotary(q, &cos, &sin)?;
        let k_embed = Self::apply_rotary(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Applies neox-style rotary embedding: x' = x * cos + rotate_half(x) * sin
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // x shape: [batch, num_heads, seq_len, head_dim]
        // cos/sin shape: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x_cos = x.broadcast_mul(&cos)?;
        let x_rot = Self::rotate_half(x)?;
        let x_sin = x_rot.broadcast_mul(&sin)?;

        x_cos + x_sin
    }

    /// Rotates half the hidden dims: splits in half, negates second half, and swaps
    fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        let half = last_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
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
        config: &Qwen2Config,
        vb: &LazyVarBuilder,
        prefix: &str,
    ) -> Result<Self, LazyLoadError> {
        // Qwen2 MLP doesn't use bias
        let gate_proj = LazyQwen2::load_linear(
            vb,
            &format!("{}.gate_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let up_proj = LazyQwen2::load_linear(
            vb,
            &format!("{}.up_proj", prefix),
            config.hidden_size,
            config.intermediate_size,
            false,
        )?;
        let down_proj = LazyQwen2::load_linear(
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

/// Attention layer with bias on Q/K/V (Qwen2 specific).
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: Box<dyn KvCache>,
    /// Device for creating new caches (needed when swapping caches).
    device: Device,
    /// DType for creating new caches.
    dtype: DType,
}

impl Attention {
    fn load(
        config: &Qwen2Config,
        vb: &LazyVarBuilder,
        prefix: &str,
        cache_type: CacheType,
    ) -> Result<Self, LazyLoadError> {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        // Qwen2 uses bias for Q/K/V projections (key difference from Llama)
        let q_proj = LazyQwen2::load_linear(
            vb,
            &format!("{}.q_proj", prefix),
            config.hidden_size,
            num_heads * head_dim,
            true, // Qwen2 has bias
        )?;
        let k_proj = LazyQwen2::load_linear(
            vb,
            &format!("{}.k_proj", prefix),
            config.hidden_size,
            num_kv_heads * head_dim,
            true, // Qwen2 has bias
        )?;
        let v_proj = LazyQwen2::load_linear(
            vb,
            &format!("{}.v_proj", prefix),
            config.hidden_size,
            num_kv_heads * head_dim,
            true, // Qwen2 has bias
        )?;
        // O projection has no bias
        let o_proj = LazyQwen2::load_linear(
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
            device: vb.device().clone(),
            dtype: vb.dtype(),
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

    /// Takes the KV cache, replacing it with an empty cache.
    /// Used when evicting a layer to preserve cache state.
    fn take_cache(&mut self) -> Box<dyn KvCache> {
        // Create a new empty cache of the same type
        let cache_config = KvCacheConfig {
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        let new_cache = CacheType::Standard
            .create(&cache_config)
            .expect("Failed to create replacement cache");

        std::mem::replace(&mut self.cache, new_cache)
    }

    /// Sets the KV cache, replacing the current one.
    /// Used when restoring a cache after reloading a layer.
    fn set_cache(&mut self, cache: Box<dyn KvCache>) {
        self.cache = cache;
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
        config: &Qwen2Config,
        vb: LazyVarBuilder,
        cache_type: CacheType,
    ) -> Result<Self, LazyLoadError> {
        let self_attn = Attention::load(config, &vb, "self_attn", cache_type)?;
        let mlp = Mlp::load(config, &vb, "mlp")?;
        let input_layernorm = LazyQwen2::load_norm(
            config.hidden_size,
            config.rms_norm_eps,
            &vb,
            "input_layernorm",
        )?;
        let post_attention_layernorm = LazyQwen2::load_norm(
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

    /// Sets the KV cache for this layer.
    /// Used when restoring a cache after reloading an evicted layer.
    fn set_cache(&mut self, cache: Box<dyn KvCache>) {
        self.self_attn.set_cache(cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_stats() {
        let stats = LazyStats {
            total_layers: 48,
            loaded_layers: 8,
            max_loaded_layers: 8,
            layer_loads: 20,
            layer_evictions: 12,
            prefetch_depth: 2,
        };
        assert_eq!(stats.total_layers, 48);
        assert_eq!(stats.loaded_layers, 8);
    }
}
