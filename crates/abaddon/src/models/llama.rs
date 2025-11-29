//! Llama model architecture implementation using Candle.
//!
//! Supports Llama 2, Llama 3, Llama 3.1, and Llama 3.2 variants.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use serde::Deserialize;

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

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_embed = Self::apply_rotary(q, &cos, &sin)?;
        let k_embed = Self::apply_rotary(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let x_shape = x.dims();
        let x = x.reshape((x_shape[0], x_shape[1], x_shape[2], x_shape[3] / 2, 2))?;

        let x0 = x.i((.., .., .., .., 0))?;
        let x1 = x.i((.., .., .., .., 1))?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rotated_0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let rotated_1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

        let rotated = Tensor::stack(&[rotated_0, rotated_1], D::Minus1)?;
        rotated.reshape(x_shape)
    }
}

/// Self-attention layer.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn load(config: &LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
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

        // KV cache handling
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };

        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV heads if using GQA
        let k = Self::repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights / scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
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
        self.kv_cache = None;
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
    fn load(config: &LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let self_attn = Attention::load(config, vb.pp("self_attn"))?;
        let mlp = Mlp::load(config, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
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
    /// Loads a Llama model from the given variable builder.
    pub fn load(config: LlamaConfig, vb: VarBuilder) -> CandleResult<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::load(&config, vb.pp(format!("model.layers.{}", i)))?;
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
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let mask = if seq_len > 1 {
            Some(Self::create_causal_mask(seq_len, start_pos, &self.device, self.dtype)?)
        } else {
            None
        };

        // Forward through layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rotary, mask.as_ref(), start_pos)?;
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
}
