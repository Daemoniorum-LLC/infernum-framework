//! Qwen2 model architecture implementation using Candle.
//!
//! Supports Qwen2, Qwen2.5, and Qwen2.5-Coder variants.
//! Key difference from Llama: Q/K/V projections use bias.
//!
//! ## Flash Attention Support
//!
//! Flash attention can be enabled for faster inference on longer sequences.
//! Use `Qwen2::load_with_flash_attention()` to enable it.
//!
//! ## KV Cache Support
//!
//! This model supports multiple cache backends:
//! - **Standard**: Full-precision BF16/FP16 cache
//! - **Quantized**: Basic INT8 quantized cache
//! - **OptimizedQuantized**: Dynamic quantization with recent window in BF16
//! - **CudaQuantized**: CUDA-accelerated INT8 with fused attention kernels
//!
//! Note: Qwen2 uses its own [`CacheType`] enum which provides additional options
//! beyond the generic [`crate::attention_cache::CacheType`]. Both are compatible
//! with the underlying cache implementations.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, VarBuilder};
use serde::Deserialize;

use crate::flash_attention::{FlashAttention, FlashAttentionConfig};
use crate::kv_cache_quant::QuantizedKvCache;
#[cfg(feature = "cuda")]
use crate::kv_cache_quant_cuda::cuda::CudaQuantizedKvCache;
use crate::kv_cache_quant_cuda::{DynamicQuantConfig, OptimizedQuantizedKvCache};

/// Qwen2 model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
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
    #[serde(default, alias = "num_kv_heads")]
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
    /// Use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Sliding window size.
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6 // Qwen2 uses 1e-6 by default
}

fn default_rope_theta() -> f64 {
    1000000.0 // Qwen2.5 uses 1M rope theta
}

fn default_max_position_embeddings() -> usize {
    32768 // Qwen2.5 supports 32K context
}

impl Qwen2Config {
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
        let emb = freqs;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
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

/// KV cache storage - standard, quantized, or optimized quantized.
enum KvCacheStorage {
    /// Standard BF16/FP16 cache.
    Standard(Option<(Tensor, Tensor)>),
    /// Basic quantized INT8 cache.
    Quantized(QuantizedKvCache),
    /// Optimized quantized cache with dynamic quantization and recent window.
    OptimizedQuantized(OptimizedQuantizedKvCache),
    /// CUDA-accelerated INT8 cache with fused attention kernels.
    #[cfg(feature = "cuda")]
    CudaQuantized(CudaQuantizedKvCache),
}

/// Self-attention layer with bias on Q/K/V (Qwen2 specific).
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: KvCacheStorage,
    /// Optional Flash Attention for faster inference.
    flash_attn: Option<FlashAttention>,
}

/// Cache type selection for attention layers.
#[derive(Debug, Clone)]
pub enum CacheType {
    /// Standard BF16/FP16 cache.
    Standard,
    /// Basic INT8 quantized cache.
    Quantized,
    /// Optimized quantized cache with dynamic quantization and configurable granularity.
    OptimizedQuantized(DynamicQuantConfig),
    /// CUDA-accelerated INT8 cache with fused attention kernels.
    /// Uses CUDA device ID.
    #[cfg(feature = "cuda")]
    CudaQuantized(usize),
}

impl Default for CacheType {
    fn default() -> Self {
        Self::Standard
    }
}

impl Attention {
    #[allow(dead_code)]
    fn load(
        config: &Qwen2Config,
        vb: VarBuilder,
        use_flash_attn: bool,
        use_quantized_cache: bool,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let cache_type = if use_quantized_cache {
            CacheType::Quantized
        } else {
            CacheType::Standard
        };
        Self::load_with_cache_type(config, vb, use_flash_attn, cache_type, device, dtype)
    }

    fn load_with_cache_type(
        config: &Qwen2Config,
        vb: VarBuilder,
        use_flash_attn: bool,
        cache_type: CacheType,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        // Qwen2 uses bias for Q/K/V projections (key difference from Llama)
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        // O projection has no bias
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        // Initialize flash attention if requested
        let flash_attn = if use_flash_attn {
            let flash_config =
                FlashAttentionConfig::default().with_scale(1.0 / (head_dim as f32).sqrt());
            Some(FlashAttention::new(flash_config))
        } else {
            None
        };

        // Initialize KV cache based on type
        let kv_cache = match cache_type {
            CacheType::Standard => KvCacheStorage::Standard(None),
            CacheType::Quantized => KvCacheStorage::Quantized(QuantizedKvCache::new(
                num_kv_heads,
                head_dim,
                device,
                dtype,
            )),
            CacheType::OptimizedQuantized(quant_config) => KvCacheStorage::OptimizedQuantized(
                OptimizedQuantizedKvCache::new(num_kv_heads, head_dim, device, dtype, quant_config),
            ),
            #[cfg(feature = "cuda")]
            CacheType::CudaQuantized(device_id) => {
                let cuda_cache = CudaQuantizedKvCache::new(num_kv_heads, head_dim, device_id)
                    .map_err(|e| candle_core::Error::Msg(format!("CUDA cache init failed: {e}")))?;
                KvCacheStorage::CudaQuantized(cuda_cache)
            },
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache,
            flash_attn,
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

        // KV cache handling - supports standard, quantized, optimized quantized, and CUDA quantized storage
        // CUDA quantized uses fused attention kernels, so it returns the attention output directly
        #[cfg(feature = "cuda")]
        let cuda_attn_output: Option<Tensor> =
            if let KvCacheStorage::CudaQuantized(cuda_cache) = &mut self.kv_cache {
                // Append new K/V to CUDA cache (quantizes to INT8 on GPU)
                cuda_cache.append(&k, &v).map_err(|e| {
                    candle_core::Error::Msg(format!("CUDA cache append failed: {e}"))
                })?;

                // Compute attention using fused CUDA kernels that handle GQA internally
                let attn_scale = 1.0 / (self.head_dim as f32).sqrt();
                let output = cuda_cache
                    .forward_attention(&q, self.num_heads, attn_scale)
                    .map_err(|e| candle_core::Error::Msg(format!("CUDA attention failed: {e}")))?;
                Some(output)
            } else {
                None
            };

        #[cfg(not(feature = "cuda"))]
        let cuda_attn_output: Option<Tensor> = None;

        // If CUDA attention was used, skip the standard path
        let attn_output = if let Some(cuda_out) = cuda_attn_output {
            cuda_out
        } else {
            // Standard path: get K/V from cache, repeat for GQA, compute attention
            let (k_full, v_full, cache_is_empty) = match &mut self.kv_cache {
                KvCacheStorage::Standard(cache) => {
                    let (k_out, v_out) = match cache {
                        Some((prev_k, prev_v)) => {
                            let k_cat = Tensor::cat(&[prev_k.as_ref(), &k], 2)?;
                            let v_cat = Tensor::cat(&[prev_v.as_ref(), &v], 2)?;
                            (k_cat, v_cat)
                        },
                        None => (k.clone(), v.clone()),
                    };
                    let is_empty = cache.is_none();
                    *cache = Some((k_out.clone(), v_out.clone()));
                    (k_out, v_out, is_empty)
                },
                KvCacheStorage::Quantized(cache) => {
                    let is_empty = cache.seq_len() == 0;
                    // Append new K/V to quantized cache
                    cache.append(&k, &v)?;
                    // Get dequantized full K/V for attention
                    let (k_full, v_full) = cache.get_dequantized()?.unwrap_or((k, v));
                    (k_full, v_full, is_empty)
                },
                KvCacheStorage::OptimizedQuantized(cache) => {
                    let is_empty = cache.seq_len() == 0;
                    // Append new K/V to optimized quantized cache
                    // This will automatically quantize older tokens beyond the window
                    cache.append(&k, &v)?;
                    // Get full K/V (recent in BF16 + dequantized older)
                    let (k_full, v_full) = cache.get_kv()?.unwrap_or((k, v));
                    (k_full, v_full, is_empty)
                },
                #[cfg(feature = "cuda")]
                KvCacheStorage::CudaQuantized(_) => unreachable!("CUDA path handled above"),
            };

            // Repeat KV heads if using GQA
            let k = Self::repeat_kv(k_full, self.num_heads / self.num_kv_heads)?;
            let v = Self::repeat_kv(v_full, self.num_heads / self.num_kv_heads)?;

            // Compute attention output
            // Use Flash Attention for longer sequences (prompt processing)
            // Fall back to standard attention for single-token generation
            if let Some(flash) = &self.flash_attn {
                if seq_len > 1 || cache_is_empty {
                    // Flash attention for prompt processing
                    // Note: Flash attention handles causal masking internally
                    flash.forward(&q, &k, &v, None, Some(true))?
                } else {
                    // Standard attention for single-token generation with KV cache
                    self.standard_attention(&q, &k, &v, mask)?
                }
            } else {
                // Standard scaled dot-product attention
                self.standard_attention(&q, &k, &v, mask)?
            }
        };

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    /// Standard scaled dot-product attention.
    fn standard_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights / scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights.matmul(v)
    }

    fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
        if n_rep == 1 {
            return Ok(x);
        }
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }

    fn clear_cache(&mut self) {
        match &mut self.kv_cache {
            KvCacheStorage::Standard(cache) => *cache = None,
            KvCacheStorage::Quantized(cache) => cache.clear(),
            KvCacheStorage::OptimizedQuantized(cache) => cache.clear(),
            #[cfg(feature = "cuda")]
            KvCacheStorage::CudaQuantized(cache) => cache.clear(),
        }
    }

    /// Get the current KV cache memory usage in bytes.
    fn cache_memory_bytes(&self) -> usize {
        match &self.kv_cache {
            KvCacheStorage::Standard(cache) => {
                cache.as_ref().map_or(0, |(k, v)| {
                    k.elem_count() * 2 + v.elem_count() * 2 // BF16 = 2 bytes
                })
            },
            KvCacheStorage::Quantized(cache) => cache.memory_bytes(),
            KvCacheStorage::OptimizedQuantized(cache) => cache.memory_bytes(),
            #[cfg(feature = "cuda")]
            KvCacheStorage::CudaQuantized(cache) => cache.memory_bytes(),
        }
    }
}

/// MLP (Feed-Forward) layer with SiLU activation.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(config: &Qwen2Config, vb: VarBuilder) -> CandleResult<Self> {
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
    fn load(
        config: &Qwen2Config,
        vb: VarBuilder,
        use_flash_attn: bool,
        cache_type: CacheType,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let self_attn = Attention::load_with_cache_type(
            config,
            vb.pp("self_attn"),
            use_flash_attn,
            cache_type,
            device,
            dtype,
        )?;
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

/// Complete Qwen2 model.
pub struct Qwen2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    config: Qwen2Config,
    device: Device,
    dtype: DType,
}

impl Qwen2 {
    /// Loads a Qwen2 model from the given variable builder.
    ///
    /// This uses standard attention. For Flash Attention, use `load_with_flash_attention`.
    pub fn load(config: Qwen2Config, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, false, CacheType::Standard)
    }

    /// Loads a Qwen2 model with Flash Attention enabled.
    ///
    /// Flash Attention provides faster inference for longer sequences by using
    /// memory-efficient tiled computation. On CUDA, it uses optimized kernels.
    pub fn load_with_flash_attention(config: Qwen2Config, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, true, CacheType::Standard)
    }

    /// Loads a Qwen2 model with INT8 quantized KV cache.
    ///
    /// This reduces KV cache memory by ~2x, allowing longer context windows.
    /// May have slight quality impact due to quantization.
    pub fn load_with_quantized_cache(config: Qwen2Config, vb: VarBuilder) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, false, CacheType::Quantized)
    }

    /// Loads a Qwen2 model with both Flash Attention and INT8 quantized KV cache.
    ///
    /// Combines memory efficiency benefits:
    /// - Flash Attention: faster long-sequence processing
    /// - Quantized KV cache: ~2x memory reduction
    pub fn load_with_flash_and_quantized_cache(
        config: Qwen2Config,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, true, CacheType::Quantized)
    }

    /// Loads a Qwen2 model with optimized quantized KV cache.
    ///
    /// The optimized cache uses dynamic quantization with:
    /// - Recent tokens kept in BF16 for speed (no dequantization overhead)
    /// - Older tokens quantized to INT8 for memory savings
    /// - Configurable granularity (per-token, per-head, per-channel)
    ///
    /// This provides the best of both worlds: speed for recent context and
    /// memory efficiency for long sequences.
    pub fn load_with_optimized_cache(
        config: Qwen2Config,
        vb: VarBuilder,
        quant_config: DynamicQuantConfig,
    ) -> CandleResult<Self> {
        Self::load_with_cache_type(
            config,
            vb,
            false,
            CacheType::OptimizedQuantized(quant_config),
        )
    }

    /// Loads a Qwen2 model with Flash Attention and optimized quantized KV cache.
    ///
    /// Combines:
    /// - Flash Attention for faster long-sequence processing
    /// - Dynamic INT8 quantization with unquantized window for recent tokens
    /// - Configurable quantization granularity
    pub fn load_with_flash_and_optimized_cache(
        config: Qwen2Config,
        vb: VarBuilder,
        quant_config: DynamicQuantConfig,
    ) -> CandleResult<Self> {
        Self::load_with_cache_type(
            config,
            vb,
            true,
            CacheType::OptimizedQuantized(quant_config),
        )
    }

    /// Loads a Qwen2 model with CUDA-accelerated INT8 quantized KV cache.
    ///
    /// Uses fused CUDA kernels for attention computation:
    /// - K/V stored as INT8 with per-token BF16 scales
    /// - Attention computed with on-the-fly dequantization (no full tensor copy)
    /// - GQA handled internally by kernels
    /// - ~2x memory reduction vs BF16
    ///
    /// Flash Attention is not used since the CUDA kernels implement fused attention.
    #[cfg(feature = "cuda")]
    pub fn load_with_cuda_quantized_cache(
        config: Qwen2Config,
        vb: VarBuilder,
        cuda_device_id: usize,
    ) -> CandleResult<Self> {
        Self::load_with_cache_type(config, vb, false, CacheType::CudaQuantized(cuda_device_id))
    }

    /// Internal load function with flash attention and cache type control.
    fn load_with_cache_type(
        config: Qwen2Config,
        vb: VarBuilder,
        use_flash_attn: bool,
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
            let layer = DecoderLayer::load(
                &config,
                vb.pp(format!("model.layers.{}", i)),
                use_flash_attn,
                cache_type.clone(),
                &device,
                dtype,
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
        self.layers
            .first()
            .map_or(0, |l| match &l.self_attn.kv_cache {
                KvCacheStorage::Standard(cache) => cache.as_ref().map_or(0, |(k, _)| k.dims()[2]),
                KvCacheStorage::Quantized(cache) => cache.seq_len(),
                KvCacheStorage::OptimizedQuantized(cache) => cache.seq_len(),
                #[cfg(feature = "cuda")]
                KvCacheStorage::CudaQuantized(cache) => cache.seq_len(),
            })
    }

    /// Forward pass that returns hidden states for embedding extraction.
    pub fn forward_embedding(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        // Clear KV cache to avoid shape mismatches from previous generations
        self.clear_cache();

        let (_batch_size, seq_len) = input_ids.dims2()?;

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

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

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rotary, mask.as_ref(), 0)?;
        }

        self.norm.forward(&hidden_states)
    }

    /// Extracts embeddings by mean pooling over the sequence dimension.
    pub fn extract_embeddings(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.forward_embedding(input_ids)?;
        hidden_states.mean(1)
    }
}
