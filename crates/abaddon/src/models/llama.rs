//! Llama model architecture implementation using Candle.
//!
//! Supports Llama 2, Llama 3, Llama 3.1, and Llama 3.2 variants.
//!
//! This implementation supports pluggable KV cache strategies through the
//! [`KvCache`] trait, allowing use of standard caches, quantized caches,
//! or CUDA-accelerated caches.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use serde::Deserialize;

use crate::attention_cache::{attention_with_cache, CacheType, KvCache, KvCacheConfig};

/// RoPE scaling configuration for extended context models.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScalingConfig {
    /// Scaling factor.
    #[serde(default)]
    pub factor: Option<f64>,
    /// High frequency factor (for llama3 rope).
    #[serde(default)]
    pub high_freq_factor: Option<f64>,
    /// Low frequency factor (for llama3 rope).
    #[serde(default)]
    pub low_freq_factor: Option<f64>,
    /// Original max position embeddings before scaling.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// RoPE scaling type: "linear", "dynamic", "llama3", etc.
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Llama model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Intermediate size for MLP.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of hidden layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// BOS token ID.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// EOS token ID.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// RoPE scaling configuration (for Llama 3.2+ extended context).
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

impl LlamaConfig {
    /// Returns the number of key-value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// RMS Layer Normalization.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
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

/// Rotary Position Embedding cache.
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
                // Linear scaling
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
        // Don't concatenate - freqs already has the right dimension (head_dim/2)
        let emb = freqs;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    /// Apply Llama3 RoPE scaling algorithm.
    /// Based on: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
    fn apply_llama3_scaling(inv_freq: &[f32], scaling: &RopeScalingConfig) -> Vec<f32> {
        let factor = scaling.factor.unwrap_or(1.0) as f32;
        let low_freq_factor = scaling.low_freq_factor.unwrap_or(1.0) as f32;
        let high_freq_factor = scaling.high_freq_factor.unwrap_or(4.0) as f32;
        let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(8192) as f32;

        let low_freq_wavelen = orig_max_pos / low_freq_factor;
        let high_freq_wavelen = orig_max_pos / high_freq_factor;

        inv_freq
            .iter()
            .map(|&freq| {
                let wavelen = 2.0 * std::f32::consts::PI / freq;
                if wavelen > low_freq_wavelen {
                    // Low frequency: scale down by factor
                    freq / factor
                } else if wavelen < high_freq_wavelen {
                    // High frequency: keep original
                    freq
                } else {
                    // Middle range: smooth interpolation
                    let smooth = (orig_max_pos / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * freq / factor + smooth * freq
                }
            })
            .collect()
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
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
    /// where rotate_half splits the tensor in half and swaps/negates: [-x2, x1]
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // x shape: (batch, seq, heads, head_dim)
        // cos/sin shape: (seq, head_dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?; // (1, seq, 1, head_dim)
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?; // (1, seq, 1, head_dim)

        let x_cos = x.broadcast_mul(&cos)?;
        let x_rot = Self::rotate_half(x)?;
        let x_sin = x_rot.broadcast_mul(&sin)?;

        x_cos + x_sin
    }

    /// Rotates half the hidden dims: splits in half, negates second half, and swaps
    /// [x0, x1, x2, x3, ..., xn/2, ..., xn-1] -> [-xn/2, ..., -xn-1, x0, x1, ..., xn/2-1]
    fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        let half = last_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
    }
}

/// Self-attention layer with pluggable KV cache.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// KV cache supporting different backends (standard, quantized, CUDA).
    kv_cache: Box<dyn KvCache>,
}

impl Attention {
    #[allow(dead_code)]
    fn load(config: &LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, CacheType::Standard)
    }

    fn load_with_cache_type(
        config: &LlamaConfig,
        vb: VarBuilder,
        cache_type: CacheType,
    ) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let cache_config =
            KvCacheConfig::new(num_kv_heads, head_dim, vb.device().clone(), vb.dtype());
        let kv_cache = cache_type.create(&cache_config)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache,
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
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, start_pos)?;

        // Transpose for attention: (batch, num_heads, seq_len, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Use the generic attention function which handles both standard and fused cache modes
        let attn_output = attention_with_cache(
            &q,
            &k,
            &v,
            &mut *self.kv_cache,
            self.num_heads,
            self.num_kv_heads,
            mask,
        )?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Returns the current cache sequence length.
    fn cache_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    /// Returns the cache memory usage in bytes.
    fn cache_memory_bytes(&self) -> usize {
        self.kv_cache.memory_bytes()
    }
}

/// MLP (Feed-Forward) layer.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(config: &LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

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

/// Transformer decoder layer.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(dead_code)]
    fn load(config: &LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, CacheType::Standard)
    }

    fn load_with_cache_type(
        config: &LlamaConfig,
        vb: VarBuilder,
        cache_type: CacheType,
    ) -> CandleResult<Self> {
        let self_attn = Attention::load_with_cache_type(config, vb.pp("self_attn"), cache_type)?;
        let mlp = Mlp::load(config, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
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
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rotary, mask, start_pos)?;
        let x = (residual + x)?;

        // MLP with residual
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

/// Complete Llama model.
pub struct Llama {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    config: LlamaConfig,
    device: Device,
    dtype: DType,
}

impl Llama {
    /// Loads a Llama model from the given variable builder with default (standard) KV cache.
    pub fn load(config: LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, CacheType::Standard)
    }

    /// Loads a Llama model with a specific KV cache type.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `vb` - Variable builder with model weights
    /// * `cache_type` - Type of KV cache to use (Standard, Quantized, or CudaQuantized)
    ///
    /// # Example
    /// ```ignore
    /// use abaddon::models::llama::Llama;
    /// use abaddon::attention_cache::CacheType;
    ///
    /// // Use CUDA-accelerated INT8 quantized cache
    /// let cache_type = CacheType::CudaQuantized { device_id: 0 };
    /// let model = Llama::load_with_cache_type(config, vb, cache_type)?;
    /// ```
    pub fn load_with_cache_type(
        config: LlamaConfig,
        vb: VarBuilder,
        cache_type: CacheType,
    ) -> CandleResult<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::load_with_cache_type(
                &config,
                vb.pp(format!("model.layers.{}", i)),
                cache_type.clone(),
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        let rotary = RotaryEmbedding::new(&config, dtype, &device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            config,
            device,
            dtype,
        })
    }

    /// Forward pass for the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch_size, seq_len)
    /// * `start_pos` - Starting position for KV cache (0 for prefill, incremental for generation)
    ///
    /// # Returns
    /// Logits tensor of shape (batch_size, seq_len, vocab_size)
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let mask = if seq_len > 1 {
            Some(Self::create_causal_mask(
                seq_len,
                start_pos,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };

        // Forward through layers
        for layer in &mut self.layers {
            hidden_states =
                layer.forward(&hidden_states, &self.rotary, mask.as_ref(), start_pos)?;
        }

        // Final layer norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        self.lm_head.forward(&hidden_states)
    }

    /// Creates a causal attention mask.
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

    /// Clears the KV cache (for starting a new generation).
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &LlamaConfig {
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

    /// Returns the total cache memory usage across all layers in bytes.
    pub fn cache_memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.self_attn.cache_memory_bytes())
            .sum()
    }

    /// Returns the current cache sequence length.
    ///
    /// Returns 0 if cache is empty, otherwise returns the sequence length
    /// from the first layer's cache.
    pub fn cache_seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.self_attn.cache_len())
    }

    /// Forward pass that returns hidden states for embedding extraction.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch_size, seq_len)
    ///
    /// # Returns
    /// Hidden states tensor of shape (batch_size, seq_len, hidden_size)
    pub fn forward_embedding(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        // Clear KV cache to avoid shape mismatches from previous generations
        self.clear_cache();

        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask (using 0 start_pos since we're not caching for embeddings)
        let mask = if seq_len > 1 {
            Some(Self::create_causal_mask(
                seq_len,
                0,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };

        // Forward through layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rotary, mask.as_ref(), 0)?;
        }

        // Final layer norm (return hidden states, not logits)
        self.norm.forward(&hidden_states)
    }

    /// Extracts embeddings by mean pooling over the sequence dimension.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch_size, seq_len)
    ///
    /// # Returns
    /// Embedding tensor of shape (batch_size, hidden_size)
    pub fn extract_embeddings(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.forward_embedding(input_ids)?;
        // Mean pool over the sequence dimension (dim 1)
        hidden_states.mean(1)
    }
}
