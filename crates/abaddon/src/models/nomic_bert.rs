//! NomicBERT embedding model.
//!
//! Custom architecture with fused QKV, rotary position embeddings, SwiGLU MLP,
//! and post-norm transformer blocks.  Weight names match the HuggingFace
//! `nomic-ai/nomic-embed-text-v1.5` checkpoint layout.

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{layer_norm, Embedding, LayerNorm, Linear, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a NomicBERT model, deserialized from `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NomicBertConfig {
    /// Vocabulary size.
    #[serde(default = "d_vocab")]
    pub vocab_size: usize,
    /// Hidden layer dimension.
    #[serde(default = "d_hidden")]
    pub hidden_size: usize,
    /// Number of transformer layers.
    #[serde(default = "d_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    /// FFN intermediate dimension.
    #[serde(default = "d_inter")]
    pub intermediate_size: usize,
    /// Token-type vocabulary size.
    #[serde(default = "d_type_vocab")]
    pub type_vocab_size: usize,
    /// Layer-norm epsilon.
    #[serde(default = "d_ln_eps")]
    pub layer_norm_eps: f64,
    /// Padding token ID.
    #[serde(default)]
    pub pad_token_id: usize,
    /// Maximum sequence length.
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: usize,
    /// Rotary embedding base frequency.
    #[serde(default = "d_rotary_base")]
    pub rotary_emb_base: f64,
    /// Fraction of head_dim used for rotary embeddings.
    #[serde(default = "d_rotary_frac")]
    pub rotary_emb_fraction: f64,
    /// Whether QKV projection has bias.
    #[serde(default)]
    pub qkv_proj_bias: bool,
    /// Whether MLP fc1 layers have bias.
    #[serde(default)]
    pub mlp_fc1_bias: bool,
    /// Whether MLP fc2 layer has bias.
    #[serde(default)]
    pub mlp_fc2_bias: bool,
}

fn d_vocab() -> usize { 30528 }
fn d_hidden() -> usize { 768 }
fn d_layers() -> usize { 12 }
fn d_heads() -> usize { 12 }
fn d_inter() -> usize { 3072 }
fn d_type_vocab() -> usize { 2 }
fn d_ln_eps() -> f64 { 1e-12 }
fn d_max_pos() -> usize { 2048 }
fn d_rotary_base() -> f64 { 1000.0 }
fn d_rotary_frac() -> f64 { 1.0 }

impl NomicBertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rotary_emb_fraction) as usize
    }
}

// ---------------------------------------------------------------------------
// Rotary embeddings (non-interleaved / neox-style, matching Llama convention)
// ---------------------------------------------------------------------------

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &NomicBertConfig, dtype: DType, device: &Device) -> CandleResult<Self> {
        let rotary_dim = cfg.rotary_dim();
        let max_seq = cfg.max_position_embeddings;
        let base = cfg.rotary_emb_base;

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (base.powf(i as f64 / rotary_dim as f64) as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let q_rot = rotate_half(q)?;
        let q_embed = (q.broadcast_mul(&cos)? + q_rot.broadcast_mul(&sin)?)?;

        let k_rot = rotate_half(k)?;
        let k_embed = (k.broadcast_mul(&cos)? + k_rot.broadcast_mul(&sin)?)?;

        Ok((q_embed, k_embed))
    }
}

fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

// ---------------------------------------------------------------------------
// Embeddings: word + token_type → LayerNorm
// ---------------------------------------------------------------------------

struct NomicEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    emb_ln: LayerNorm,
}

impl NomicEmbeddings {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> CandleResult<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings").pp("word_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings").pp("token_type_embeddings"),
        )?;
        let emb_ln = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("emb_ln"))?;
        Ok(Self { word_embeddings, token_type_embeddings, emb_ln })
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let tt_ids = Tensor::zeros(seq_len, DType::U32, input_ids.device())?
            .broadcast_left(batch)?;
        let tt_emb = self.token_type_embeddings.forward(&tt_ids)?;
        self.emb_ln.forward(&(word_emb + tt_emb)?)
    }
}

// ---------------------------------------------------------------------------
// Fused-QKV self-attention with rotary
// ---------------------------------------------------------------------------

struct NomicAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

fn linear_maybe_bias(in_d: usize, out_d: usize, bias: bool, vb: VarBuilder) -> CandleResult<Linear> {
    if bias {
        candle_nn::linear(in_d, out_d, vb)
    } else {
        candle_nn::linear_no_bias(in_d, out_d, vb)
    }
}

impl NomicAttention {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> CandleResult<Self> {
        let hidden = cfg.hidden_size;
        let wqkv = linear_maybe_bias(hidden, 3 * hidden, cfg.qkv_proj_bias, vb.pp("Wqkv"))?;
        let out_proj = linear_maybe_bias(hidden, hidden, false, vb.pp("out_proj"))?;
        Ok(Self {
            wqkv,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding) -> CandleResult<Tensor> {
        let (batch, seq_len, hidden) = xs.dims3()?;

        let qkv = self.wqkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, hidden)?;
        let k = qkv.narrow(D::Minus1, hidden, hidden)?;
        let v = qkv.narrow(D::Minus1, hidden * 2, hidden)?;

        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        let (q, k) = rotary.apply(&q, &k)?;

        // Transpose to (batch, heads, seq, head_dim) for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.t()?)?.affine(1.0 / scale, 0.0)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v)?;

        // (batch, heads, seq, head_dim) → (batch, seq, hidden)
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?;
        let attn_out = attn_out.flatten_from(D::Minus2)?;

        self.out_proj.forward(&attn_out)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP: fc11(x) * SiLU(fc12(x)) → fc2
// ---------------------------------------------------------------------------

struct NomicMlp {
    fc11: Linear, // value/up
    fc12: Linear, // gate (receives SiLU)
    fc2: Linear,  // down
}

impl NomicMlp {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let fc11 = linear_maybe_bias(h, i, cfg.mlp_fc1_bias, vb.pp("fc11"))?;
        let fc12 = linear_maybe_bias(h, i, cfg.mlp_fc1_bias, vb.pp("fc12"))?;
        let fc2 = linear_maybe_bias(i, h, cfg.mlp_fc2_bias, vb.pp("fc2"))?;
        Ok(Self { fc11, fc12, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let y = self.fc11.forward(xs)?;
        let gate = candle_nn::Activation::Silu.forward(&self.fc12.forward(xs)?)?;
        self.fc2.forward(&(y * gate)?)
    }
}

// ---------------------------------------------------------------------------
// Post-norm transformer block (prenorm=false in config)
// ---------------------------------------------------------------------------

struct NomicBlock {
    norm1: LayerNorm,
    attn: NomicAttention,
    norm2: LayerNorm,
    mlp: NomicMlp,
}

impl NomicBlock {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> CandleResult<Self> {
        let norm1 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm1"))?;
        let attn = NomicAttention::new(vb.pp("attn"), cfg)?;
        let norm2 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm2"))?;
        let mlp = NomicMlp::new(vb.pp("mlp"), cfg)?;
        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding) -> CandleResult<Tensor> {
        // Post-norm: attn → residual → norm, mlp → residual → norm
        let residual = xs;
        let h = self.attn.forward(xs, rotary)?;
        let xs = self.norm1.forward(&(h + residual)?)?;

        let residual = &xs;
        let h = self.mlp.forward(&xs)?;
        self.norm2.forward(&(h + residual)?)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// NomicBERT embedding model with fused QKV attention and rotary embeddings.
pub struct NomicBert {
    embeddings: NomicEmbeddings,
    layers: Vec<NomicBlock>,
    rotary: RotaryEmbedding,
    device: Device,
}

impl NomicBert {
    /// Loads model weights from a `VarBuilder`.
    pub fn load(cfg: NomicBertConfig, vb: VarBuilder) -> CandleResult<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let embeddings = NomicEmbeddings::new(vb.clone(), &cfg)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| NomicBlock::new(vb.pp("encoder").pp("layers").pp(i), &cfg))
            .collect::<CandleResult<Vec<_>>>()?;
        let rotary = RotaryEmbedding::new(&cfg, dtype, &device)?;
        Ok(Self { embeddings, layers, rotary, device })
    }

    /// Forward pass returning sequence hidden states.
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let mut xs = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary)?;
        }
        Ok(xs)
    }

    /// Forward pass for embedding extraction (same as `forward` for BERT-style models).
    pub fn forward_embedding(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward(input_ids)
    }

    /// Mean-pool over the sequence dimension, then L2-normalize. Always returns F32.
    pub fn extract_embeddings(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let hidden = self.forward(input_ids)?;
        let pooled = hidden.mean(1)?;
        let pooled = pooled.to_dtype(candle_core::DType::F32)?;
        let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        pooled.broadcast_div(&norm.clamp(1e-12, f64::INFINITY)?)
    }

    /// No-op: NomicBERT has no KV cache.
    pub fn clear_cache(&mut self) {}

    /// Returns the device this model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
