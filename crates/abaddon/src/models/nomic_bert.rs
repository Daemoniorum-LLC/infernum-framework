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

fn d_vocab() -> usize {
    30528
}
fn d_hidden() -> usize {
    768
}
fn d_layers() -> usize {
    12
}
fn d_heads() -> usize {
    12
}
fn d_inter() -> usize {
    3072
}
fn d_type_vocab() -> usize {
    2
}
fn d_ln_eps() -> f64 {
    1e-12
}
fn d_max_pos() -> usize {
    2048
}
fn d_rotary_base() -> f64 {
    1000.0
}
fn d_rotary_frac() -> f64 {
    1.0
}

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
    /// Leading slice of each head vector that gets rotated. Equals `head_dim`
    /// when `rotary_emb_fraction` is 1.0 (the published `v1.5` setting); a
    /// smaller value leaves the trailing components untouched.
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &NomicBertConfig, dtype: DType, device: &Device) -> CandleResult<Self> {
        let rotary_dim = cfg.rotary_dim();
        if rotary_dim % 2 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "NomicBERT rotary dimension must be even, got {rotary_dim} from \
                 head_dim {} * rotary_emb_fraction {}",
                cfg.head_dim(),
                cfg.rotary_emb_fraction,
            )));
        }
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

        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    /// Rotates the leading `rotary_dim` components of each head vector and
    /// passes the remainder through unchanged, matching the reference's
    /// `cat([x[..., :ro_dim] * cos + rotate_half(...) * sin, x[..., ro_dim:]])`.
    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let head_dim = x.dim(D::Minus1)?;
        // `rotary_dim == 0` needs no special case: the general path below
        // narrows to an empty prefix and concatenates the untouched tail,
        // which is the identity.
        if self.rotary_dim == head_dim {
            let rot = rotate_half(x)?;
            return x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?;
        }

        let head = x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
        let tail = x.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let rot = rotate_half(&head)?;
        let rotated = (head.broadcast_mul(cos)? + rot.broadcast_mul(sin)?)?;
        Tensor::cat(&[&rotated, &tail], D::Minus1)
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        Ok((self.rotate(q, &cos, &sin)?, self.rotate(k, &cos, &sin)?))
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
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            emb_ln,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let tt_ids =
            Tensor::zeros(seq_len, DType::U32, input_ids.device())?.broadcast_left(batch)?;
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

fn linear_maybe_bias(
    in_d: usize,
    out_d: usize,
    bias: bool,
    vb: VarBuilder,
) -> CandleResult<Linear> {
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
        // The reference wires out_proj's bias to the same `qkv_proj_bias`
        // flag as Wqkv (modeling_hf_nomic_bert.py: `nn.Linear(embed_dim,
        // embed_dim, bias=config.qkv_proj_bias)`). Hardcoding `false` here
        // silently dropped the bias for any checkpoint that has one.
        let out_proj = linear_maybe_bias(hidden, hidden, cfg.qkv_proj_bias, vb.pp("out_proj"))?;
        Ok(Self {
            wqkv,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `additive_mask`, when present, is broadcast onto the attention scores
    /// before the softmax: `0.0` for a real token, a large negative value for
    /// a padding one. Shape `(batch, 1, 1, seq_len)`.
    fn forward(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        additive_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
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
        let attn_weights = match additive_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
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
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        additive_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // Post-norm: attn → residual → norm, mlp → residual → norm
        let residual = xs;
        let h = self.attn.forward(xs, rotary, additive_mask)?;
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
        Ok(Self {
            embeddings,
            layers,
            rotary,
            device,
        })
    }

    /// Forward pass returning sequence hidden states.
    ///
    /// Every position attends to every other. Correct for a single sequence
    /// or an evenly sized batch; for a padded batch use
    /// [`Self::forward_with_mask`].
    pub fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_mask(input_ids, None)
    }

    /// Forward pass honouring an attention mask.
    ///
    /// `attention_mask` is `(batch, seq_len)`, non-zero for a real token and
    /// zero for padding. Masked positions are excluded from every other
    /// token's attention, so a sequence's hidden states do not depend on how
    /// much padding happens to sit beside it in the batch.
    pub fn forward_with_mask(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let additive = attention_mask
            .map(|m| Self::additive_mask(m, input_ids))
            .transpose()?;

        let mut xs = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary, additive.as_ref())?;
        }
        Ok(xs)
    }

    /// Turns a `(batch, seq)` keep/drop mask into the `(batch, 1, 1, seq)`
    /// additive mask the attention softmax wants.
    ///
    /// Padding gets a large finite negative rather than `-inf`: a row that is
    /// entirely padding would otherwise softmax to NaN, and a padded row's
    /// output is still multiplied into the residual stream even though the
    /// pooling discards it.
    fn additive_mask(attention_mask: &Tensor, input_ids: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let mask = attention_mask.to_dtype(DType::F32)?;
        if mask.dims2()? != (batch, seq_len) {
            return Err(candle_core::Error::Msg(format!(
                "attention_mask shape {:?} does not match input_ids shape {:?}",
                mask.dims2()?,
                (batch, seq_len),
            )));
        }
        // keep → 0.0, pad → -1e30
        let keep = mask.ne(0.0)?.to_dtype(DType::F32)?;
        let additive = (keep.affine(1.0, -1.0)? * 1e30)?;
        additive.reshape((batch, 1, 1, seq_len))
    }

    /// Forward pass for embedding extraction (same as `forward` for BERT-style models).
    pub fn forward_embedding(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.forward(input_ids)
    }

    /// Mean-pool over the sequence dimension. Always returns F32.
    ///
    /// Returns raw pooled hidden states, matching `Bert`, `Llama` and `Qwen2`.
    /// L2 normalization is applied once on the way out of the embedding
    /// endpoint by [`crate::models::l2_normalize`], so that every architecture
    /// returns embeddings on the same scale and truncated Matryoshka
    /// embeddings get normalized after slicing rather than before.
    pub fn extract_embeddings(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        self.extract_embeddings_with_mask(input_ids, None)
    }

    /// Mean-pool over the sequence, counting only unmasked positions.
    ///
    /// `attention_mask` is `(batch, seq_len)`, non-zero for a real token. With
    /// a mask, padding is excluded from both the attention and the pool, so
    /// appending padding to a sequence leaves its embedding unchanged — which
    /// is what makes batching variable-length inputs safe.
    ///
    /// Returns raw pooled hidden states; see [`Self::extract_embeddings`] for
    /// where normalization happens.
    pub fn extract_embeddings_with_mask(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let hidden = self
            .forward_with_mask(input_ids, attention_mask)?
            .to_dtype(DType::F32)?;

        let pooled = match attention_mask {
            None => hidden.mean(1)?,
            Some(mask) => {
                let (batch, seq_len) = input_ids.dims2()?;
                let keep = mask
                    .to_dtype(DType::F32)?
                    .ne(0.0)?
                    .to_dtype(DType::F32)?
                    .reshape((batch, seq_len, 1))?;
                let summed = hidden.broadcast_mul(&keep)?.sum(1)?;
                // An all-padding row would divide by zero; clamp so it yields
                // zeros rather than NaNs.
                let counts = keep.sum(1)?.clamp(1.0, f64::INFINITY)?;
                summed.broadcast_div(&counts)?
            },
        };
        Ok(pooled)
    }

    /// No-op: NomicBERT has no KV cache.
    pub fn clear_cache(&mut self) {}

    /// Returns the device this model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests never touch the network. Every fixture is built in-process from
// deterministic weights, so they run identically in CI and on a laptop.
//
// The expected values are *derived*, not recorded. Where a test asserts a
// concrete number it comes from one of two places:
//
//   * a closed-form derivation written out in the test's comment, or
//   * `reference::forward`, a naive scalar re-implementation of the
//     architecture written from the published `nomic-ai/nomic-bert-2048`
//     `modeling_hf_nomic_bert.py` (and the `nomic-embed-text-v1.5`
//     `config.json`) rather than from the tensor code under test.
//
// The reference encodes these facts from that source, each of which is a thing
// this module could plausibly have got wrong:
//
//   * `Wqkv.weight` is `cat([Wq, Wk, Wv], dim=0)`, so the fused projection
//     splits Q, K, V in that order, and within each block the layout is
//     `(head, head_dim)` (`rearrange "... (three h d) -> ... three h d"`).
//   * rotary is non-interleaved (`rotary_emb_interleaved: false`):
//     `x * cos + rotate_half(x) * sin` with `rotate_half = cat(-x2, x1)` over
//     the two halves of the head vector.
//   * `inv_freq = 1 / base ** (arange(0, d, 2) / d)` with `base = 1000`.
//   * attention divides by `sqrt(head_dim)` and is non-causal.
//   * blocks are post-norm (`prenorm: false`):
//     `h = norm1(attn(x) + x)` then `norm2(mlp(h) + h)`.
//   * the MLP is `fc2(fc11(x) * silu(fc12(x)))`.
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const EPS_F32: f32 = 1e-12;

    // -----------------------------------------------------------------------
    // Tensor-backend seam
    // -----------------------------------------------------------------------
    //
    // Everything in this test module that touches the tensor backend lives
    // here. The fixtures below are plain `Vec<f32>`, the reference
    // implementation is plain scalar Rust, and the assertions go through
    // `TensorValues`. Only this module names candle.
    //
    // That matters because `docs/NIHIL-INTEGRATION-SPEC.md` plans to replace
    // candle with the Nihil tensor framework. `VarBuilder` in particular has
    // no direct Nihil equivalent (spec §3.3: it becomes `nihil_io::
    // SafetensorsFile` plus manual loading), so weight loading has to be
    // rewritten whenever that lands. Keeping it in one place makes that a
    // single-module edit rather than a sweep over ~80 call sites.
    //
    // The seam is not total: `NomicBert`'s own signatures take and return
    // `candle_core::Tensor`, so the model API changes with the backend no
    // matter what. What this buys is that the *fixture construction and
    // readback* boilerplate — the bulk of it — is centralised.
    mod backend {
        use super::*;

        /// One weight as flat row-major data plus its shape. Backend-free, so
        /// `TinyWeights` can describe a whole checkpoint without naming a
        /// tensor type.
        pub struct NamedWeight {
            pub name: String,
            pub data: Vec<f32>,
            pub shape: Vec<usize>,
        }

        impl NamedWeight {
            pub fn new(
                name: impl Into<String>,
                data: &[f32],
                shape: impl Into<Vec<usize>>,
            ) -> Self {
                Self {
                    name: name.into(),
                    data: data.to_vec(),
                    shape: shape.into(),
                }
            }
        }

        /// The tensor type `NomicBert`'s own signatures take and return.
        /// Aliased so the rest of the module never names the backend's type
        /// directly, even where the model API forces the dependency.
        pub type Ids = Tensor;

        pub fn device() -> Device {
            Device::Cpu
        }

        /// Builds an f32 tensor of the given shape from flat row-major data.
        pub fn f32_tensor(data: &[f32], shape: impl Into<candle_core::Shape>) -> Tensor {
            Tensor::from_slice(data, shape, &device()).expect("f32 fixture tensor")
        }

        /// Builds a `(batch, seq)` u32 tensor — token ids, or a 1/0 mask.
        pub fn u32_rows(rows: &[&[u32]]) -> Tensor {
            let seq = rows[0].len();
            assert!(rows.iter().all(|r| r.len() == seq), "ragged fixture input");
            let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
            Tensor::from_slice(&flat, (rows.len(), seq), &device()).expect("u32 fixture tensor")
        }

        /// Materialises weights and loads a model. Returns the backend's error
        /// as a string so callers can assert on load failures without naming
        /// the backend's error type.
        pub fn load_model(
            cfg: &NomicBertConfig,
            weights: Vec<NamedWeight>,
        ) -> Result<NomicBert, String> {
            let ts: HashMap<String, Tensor> = weights
                .into_iter()
                .map(|w| (w.name, f32_tensor(&w.data, w.shape)))
                .collect();
            let vb = VarBuilder::from_tensors(ts, DType::F32, &device());
            NomicBert::load(cfg.clone(), vb).map_err(|e| e.to_string())
        }

        /// Builds the rotary table directly, for the isolation tests.
        pub fn rotary(cfg: &NomicBertConfig) -> Result<RotaryEmbedding, String> {
            RotaryEmbedding::new(cfg, DType::F32, &device()).map_err(|e| e.to_string())
        }

        /// Reading values and shapes back out of a tensor.
        pub trait TensorValues {
            /// Every element, flattened to row-major f32.
            fn values(&self) -> Vec<f32>;
            fn shape2(&self) -> (usize, usize);
            fn shape3(&self) -> (usize, usize, usize);
            fn is_f32(&self) -> bool;
        }

        impl TensorValues for Tensor {
            fn values(&self) -> Vec<f32> {
                self.flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("read f32 values")
            }

            fn shape2(&self) -> (usize, usize) {
                self.dims2().expect("expected a rank-2 tensor")
            }

            fn shape3(&self) -> (usize, usize, usize) {
                self.dims3().expect("expected a rank-3 tensor")
            }

            fn is_f32(&self) -> bool {
                self.dtype() == DType::F32
            }
        }
    }

    use backend::{NamedWeight, TensorValues};

    // -----------------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------------

    /// Builds a `NomicBertConfig` through serde, the same path a real
    /// `config.json` takes.
    fn tiny_config(
        hidden: usize,
        heads: usize,
        layers: usize,
        inter: usize,
        vocab: usize,
    ) -> NomicBertConfig {
        serde_json::from_value(serde_json::json!({
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": inter,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 16,
            "rotary_emb_base": 1000.0,
            "rotary_emb_fraction": 1.0,
        }))
        .expect("tiny config deserializes")
    }

    /// Deterministic value source. A plain LCG so fixtures are byte-identical
    /// on every platform and run, without pulling in an RNG dependency.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(2) | 1)
        }

        /// Uniform in `[-0.5, 0.5)`.
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }

        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.next() * scale).collect()
        }
    }

    struct LayerWeights {
        norm1_w: Vec<f32>,
        norm1_b: Vec<f32>,
        wqkv: Vec<f32>,
        out_proj: Vec<f32>,
        norm2_w: Vec<f32>,
        norm2_b: Vec<f32>,
        fc11: Vec<f32>,
        fc12: Vec<f32>,
        fc2: Vec<f32>,
    }

    struct TinyWeights {
        word_emb: Vec<f32>,
        tt_emb: Vec<f32>,
        emb_ln_w: Vec<f32>,
        emb_ln_b: Vec<f32>,
        layers: Vec<LayerWeights>,
    }

    impl TinyWeights {
        /// Non-degenerate weights. Linear layers are scaled by `1/sqrt(fan_in)`
        /// so activations stay O(1) and `silu` is exercised away from its
        /// saturating tails; layer-norm gains sit near 1 as they would after
        /// real training.
        fn deterministic(cfg: &NomicBertConfig, seed: u64) -> Self {
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let mut r = Lcg::new(seed);
            let fan = |n: usize| 1.0 / (n as f32).sqrt();

            let word_emb = r.vec(cfg.vocab_size * h, 1.0);
            let tt_emb = r.vec(cfg.type_vocab_size * h, 0.25);
            let emb_ln_w = r.vec(h, 0.4).iter().map(|v| 1.0 + v).collect();
            let emb_ln_b = r.vec(h, 0.2);

            let layers = (0..cfg.num_hidden_layers)
                .map(|_| LayerWeights {
                    norm1_w: r.vec(h, 0.4).iter().map(|v| 1.0 + v).collect(),
                    norm1_b: r.vec(h, 0.2),
                    wqkv: r.vec(3 * h * h, fan(h)),
                    out_proj: r.vec(h * h, fan(h)),
                    norm2_w: r.vec(h, 0.4).iter().map(|v| 1.0 + v).collect(),
                    norm2_b: r.vec(h, 0.2),
                    fc11: r.vec(inter * h, fan(h)),
                    fc12: r.vec(inter * h, fan(h)),
                    fc2: r.vec(h * inter, fan(inter)),
                })
                .collect();

            Self {
                word_emb,
                tt_emb,
                emb_ln_w,
                emb_ln_b,
                layers,
            }
        }

        /// Every projection zero, every layer-norm at its identity gain. Used
        /// by the degenerate-input test.
        fn zeroed(cfg: &NomicBertConfig) -> Self {
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            Self {
                word_emb: vec![0.0; cfg.vocab_size * h],
                tt_emb: vec![0.0; cfg.type_vocab_size * h],
                emb_ln_w: vec![1.0; h],
                emb_ln_b: vec![0.0; h],
                layers: (0..cfg.num_hidden_layers)
                    .map(|_| LayerWeights {
                        norm1_w: vec![1.0; h],
                        norm1_b: vec![0.0; h],
                        wqkv: vec![0.0; 3 * h * h],
                        out_proj: vec![0.0; h * h],
                        norm2_w: vec![1.0; h],
                        norm2_b: vec![0.0; h],
                        fc11: vec![0.0; inter * h],
                        fc12: vec![0.0; inter * h],
                        fc2: vec![0.0; h * inter],
                    })
                    .collect(),
            }
        }

        /// Materialises the weights under the exact names `NomicBert::load`
        /// asks for. A rename on either side fails the load, which is itself
        /// worth catching.
        /// Describes the whole checkpoint under the exact names
        /// `NomicBert::load` asks for. Backend-free: see `mod backend` for
        /// where these become tensors.
        fn named_weights(&self, cfg: &NomicBertConfig) -> Vec<NamedWeight> {
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let mut ws = vec![
                NamedWeight::new(
                    "embeddings.word_embeddings.weight",
                    &self.word_emb,
                    [cfg.vocab_size, h],
                ),
                NamedWeight::new(
                    "embeddings.token_type_embeddings.weight",
                    &self.tt_emb,
                    [cfg.type_vocab_size, h],
                ),
                NamedWeight::new("emb_ln.weight", &self.emb_ln_w, [h]),
                NamedWeight::new("emb_ln.bias", &self.emb_ln_b, [h]),
            ];

            for (i, l) in self.layers.iter().enumerate() {
                let p = format!("encoder.layers.{i}");
                for (name, data, shape) in [
                    ("norm1.weight", &l.norm1_w, vec![h]),
                    ("norm1.bias", &l.norm1_b, vec![h]),
                    ("norm2.weight", &l.norm2_w, vec![h]),
                    ("norm2.bias", &l.norm2_b, vec![h]),
                    ("attn.Wqkv.weight", &l.wqkv, vec![3 * h, h]),
                    ("attn.out_proj.weight", &l.out_proj, vec![h, h]),
                    ("mlp.fc11.weight", &l.fc11, vec![inter, h]),
                    ("mlp.fc12.weight", &l.fc12, vec![inter, h]),
                    ("mlp.fc2.weight", &l.fc2, vec![h, inter]),
                ] {
                    ws.push(NamedWeight::new(format!("{p}.{name}"), data, shape));
                }
            }

            ws
        }
    }

    fn load(cfg: &NomicBertConfig, w: &TinyWeights) -> NomicBert {
        backend::load_model(cfg, w.named_weights(cfg)).expect("fixture model loads")
    }

    fn ids_tensor(rows: &[&[u32]]) -> backend::Ids {
        backend::u32_rows(rows)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "{what}: element {i} was {a}, expected {e} (tolerance {tol})",
            );
        }
    }

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // -----------------------------------------------------------------------
    // Independent reference implementation
    // -----------------------------------------------------------------------

    mod reference {
        use super::{LayerWeights, TinyWeights};
        use crate::models::nomic_bert::NomicBertConfig;

        pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
            let n = x.len() as f32;
            let mean = x.iter().sum::<f32>() / n;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let denom = (var + eps).sqrt();
            x.iter()
                .zip(w)
                .zip(b)
                .map(|((v, w), b)| (v - mean) / denom * w + b)
                .collect()
        }

        /// `y = x W^T` for a candle `Linear` weight stored row-major as
        /// `[out_dim, in_dim]`. All NomicBERT projections are bias-free.
        pub fn linear(x: &[f32], w: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
            (0..out_d)
                .map(|o| {
                    w[o * in_d..(o + 1) * in_d]
                        .iter()
                        .zip(x)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect()
        }

        pub fn softmax(v: &mut [f32]) {
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in v.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in v.iter_mut() {
                *x /= sum;
            }
        }

        pub fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }

        /// Non-interleaved ("neox") rotary, applied in place to one head
        /// vector at absolute position `pos`.
        ///
        /// `rotate_half(x) = cat(-x2, x1)` over the two halves, so
        /// `x * cos + rotate_half(x) * sin` rotates the pair `(x[i], x[i+half])`
        /// by `pos * inv_freq[i]`. `inv_freq` is cast through f32 to match the
        /// precision the model builds its tables at.
        pub fn apply_rotary(vec: &mut [f32], pos: usize, base: f64) {
            let d = vec.len();
            let half = d / 2;
            let orig = vec.to_vec();
            for i in 0..half {
                let inv_freq = 1.0 / (base.powf((2 * i) as f64 / d as f64) as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = (angle.sin(), angle.cos());
                vec[i] = orig[i] * cos - orig[i + half] * sin;
                vec[i + half] = orig[i + half] * cos + orig[i] * sin;
            }
        }

        fn block(cfg: &NomicBertConfig, l: &LayerWeights, xs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
            let h = cfg.hidden_size;
            let heads = cfg.num_attention_heads;
            let hd = cfg.head_dim();
            let inter = cfg.intermediate_size;
            let eps = cfg.layer_norm_eps as f32;
            let seq = xs.len();

            // Fused QKV, split [Q | K | V] and then (head, head_dim).
            let mut q = vec![vec![0.0f32; hd]; seq * heads];
            let mut k = vec![vec![0.0f32; hd]; seq * heads];
            let mut v = vec![vec![0.0f32; hd]; seq * heads];
            for (t, x) in xs.iter().enumerate() {
                let qkv = linear(x, &l.wqkv, h, 3 * h);
                for head in 0..heads {
                    for d in 0..hd {
                        q[t * heads + head][d] = qkv[head * hd + d];
                        k[t * heads + head][d] = qkv[h + head * hd + d];
                        v[t * heads + head][d] = qkv[2 * h + head * hd + d];
                    }
                }
                for head in 0..heads {
                    apply_rotary(&mut q[t * heads + head], t, cfg.rotary_emb_base);
                    apply_rotary(&mut k[t * heads + head], t, cfg.rotary_emb_base);
                }
            }

            // Non-causal scaled dot-product attention, per head.
            let scale = (hd as f32).sqrt();
            let mut attn = vec![vec![0.0f32; h]; seq];
            for head in 0..heads {
                for t in 0..seq {
                    let mut scores: Vec<f32> = (0..seq)
                        .map(|u| {
                            q[t * heads + head]
                                .iter()
                                .zip(&k[u * heads + head])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / scale
                        })
                        .collect();
                    softmax(&mut scores);
                    for d in 0..hd {
                        attn[t][head * hd + d] =
                            (0..seq).map(|u| scores[u] * v[u * heads + head][d]).sum();
                    }
                }
            }

            // Post-norm: norm1(attn + x), then norm2(mlp + that).
            let mid: Vec<Vec<f32>> = attn
                .iter()
                .zip(&xs)
                .map(|(a, x)| {
                    let o = linear(a, &l.out_proj, h, h);
                    let sum: Vec<f32> = o.iter().zip(x).map(|(a, b)| a + b).collect();
                    layer_norm(&sum, &l.norm1_w, &l.norm1_b, eps)
                })
                .collect();

            mid.iter()
                .map(|m| {
                    let up = linear(m, &l.fc11, h, inter);
                    let gate = linear(m, &l.fc12, h, inter);
                    let act: Vec<f32> = up.iter().zip(&gate).map(|(u, g)| u * silu(*g)).collect();
                    let down = linear(&act, &l.fc2, inter, h);
                    let sum: Vec<f32> = down.iter().zip(m).map(|(a, b)| a + b).collect();
                    layer_norm(&sum, &l.norm2_w, &l.norm2_b, eps)
                })
                .collect()
        }

        /// Full forward pass for a single sequence, returning `seq x hidden`.
        pub fn forward(cfg: &NomicBertConfig, w: &TinyWeights, ids: &[u32]) -> Vec<Vec<f32>> {
            let h = cfg.hidden_size;
            let eps = cfg.layer_norm_eps as f32;

            // word + token_type(all zero ids), then emb_ln.
            let mut xs: Vec<Vec<f32>> = ids
                .iter()
                .map(|&id| {
                    let word = &w.word_emb[id as usize * h..(id as usize + 1) * h];
                    let tt = &w.tt_emb[0..h];
                    let sum: Vec<f32> = word.iter().zip(tt).map(|(a, b)| a + b).collect();
                    layer_norm(&sum, &w.emb_ln_w, &w.emb_ln_b, eps)
                })
                .collect();

            for l in &w.layers {
                xs = block(cfg, l, xs);
            }
            xs
        }

        /// Mean-pool over the sequence. Deliberately does *not* normalise:
        /// `extract_embeddings` returns raw pooled hidden states, and
        /// `models::l2_normalize` is applied later by the embedding endpoint.
        pub fn extract(cfg: &NomicBertConfig, w: &TinyWeights, ids: &[u32]) -> Vec<f32> {
            let hidden = forward(cfg, w, ids);
            let seq = hidden.len() as f32;
            let mut pooled = vec![0.0f32; cfg.hidden_size];
            for row in &hidden {
                for (p, v) in pooled.iter_mut().zip(row) {
                    *p += v;
                }
            }
            for p in pooled.iter_mut() {
                *p /= seq;
            }
            pooled
        }
    }

    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------

    /// Pins the serde defaults against `nomic-ai/nomic-embed-text-v1.5`'s
    /// published `config.json`. `rotary_emb_base` in particular is 1000 for
    /// NomicBERT, not the 10000 almost every other rotary model uses — a
    /// silent "correction" to 10000 would change every embedding this model
    /// produces, and nothing else in the suite would notice.
    #[test]
    fn config_defaults_match_published_checkpoint() {
        let cfg: NomicBertConfig = serde_json::from_str("{}").expect("empty config deserializes");

        assert_eq!(cfg.vocab_size, 30528);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.type_vocab_size, 2);
        assert_eq!(cfg.max_position_embeddings, 2048);
        assert_eq!(cfg.pad_token_id, 0);
        assert_eq!(cfg.layer_norm_eps, 1e-12);
        assert_eq!(cfg.rotary_emb_base, 1000.0);
        assert_eq!(cfg.rotary_emb_fraction, 1.0);
        assert!(!cfg.qkv_proj_bias);
        assert!(!cfg.mlp_fc1_bias);
        assert!(!cfg.mlp_fc2_bias);

        // 768 / 12, fully rotated.
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.rotary_dim(), 64);
    }

    // -----------------------------------------------------------------------
    // Rotary embeddings, in isolation
    // -----------------------------------------------------------------------

    fn rotary_fixture(hidden: usize, heads: usize) -> (NomicBertConfig, RotaryEmbedding) {
        let cfg = tiny_config(hidden, heads, 1, hidden * 2, 8);
        let rotary = backend::rotary(&cfg).expect("rotary table builds");
        (cfg, rotary)
    }

    /// `cos(0) = 1`, `sin(0) = 0`, so the token at position 0 must come back
    /// bit-for-bit unchanged. Catches an off-by-one in the position index.
    #[test]
    fn rotary_at_position_zero_is_identity() {
        let (_, rotary) = rotary_fixture(4, 1);
        let q = backend::f32_tensor(&[1.0f32, -2.0, 3.0, 0.5], (1, 1, 1, 4));
        let (out, _) = rotary.apply(&q, &q).expect("rotary applies");

        assert_close(
            &out.values(),
            &[1.0, -2.0, 3.0, 0.5],
            0.0,
            "position 0 rotation",
        );
    }

    /// Hand-computed against the rotation definition: with `head_dim = 4` and
    /// `base = 1000`, `inv_freq = [1/1000^0, 1/1000^0.5] = [1, 0.0316227766]`.
    /// Non-interleaved rotary pairs `(x[0], x[2])` and `(x[1], x[3])` and turns
    /// each pair by `pos * inv_freq[i]`:
    ///
    /// ```text
    /// out[0] = x0*cos(a0) - x2*sin(a0)     out[2] = x2*cos(a0) + x0*sin(a0)
    /// out[1] = x1*cos(a1) - x3*sin(a1)     out[3] = x3*cos(a1) + x1*sin(a1)
    /// ```
    ///
    /// A `rotate_half` that dropped the minus sign, or that paired adjacent
    /// elements (the interleaved/GPT-J convention, which NomicBERT explicitly
    /// disables via `rotary_emb_interleaved: false`), fails here.
    #[test]
    fn rotary_matches_hand_computed_rotation() {
        let (_, rotary) = rotary_fixture(4, 1);
        let x = [0.3f32, -0.7, 1.1, 0.25];

        // Two positions in one sequence: row 0 is the identity, row 1 is the
        // interesting one.
        let q = backend::f32_tensor(
            &[0.0f32, 0.0, 0.0, 0.0, x[0], x[1], x[2], x[3]],
            (1, 2, 1, 4),
        );
        let (out, _) = rotary.apply(&q, &q).expect("rotary applies");
        let got = out.values();

        let a0 = 1.0f32; // 1 * 1/1000^0
        let a1 = 1.0f32 / 1000f32.powf(0.5); // 1 * 1/1000^(2/4)
        let expected = [
            x[0] * a0.cos() - x[2] * a0.sin(),
            x[1] * a1.cos() - x[3] * a1.sin(),
            x[2] * a0.cos() + x[0] * a0.sin(),
            x[3] * a1.cos() + x[1] * a1.sin(),
        ];

        assert_close(&got[0..4], &[0.0; 4], 0.0, "position 0 stays zero");
        assert_close(&got[4..8], &expected, 1e-6, "position 1 rotation");
    }

    /// A rotation is orthogonal, so it cannot change a head vector's length.
    /// A mismatched `cos`/`sin` concatenation — the easiest way to get rotary
    /// subtly wrong — breaks this at every position but the first.
    #[test]
    fn rotary_preserves_per_head_norm() {
        let (_, rotary) = rotary_fixture(8, 2);
        let mut r = Lcg::new(7);
        let seq = 6;
        let data = r.vec(seq * 2 * 4, 2.0);
        let q = backend::f32_tensor(&data, (1, seq, 2, 4));

        let (out, _) = rotary.apply(&q, &q).expect("rotary applies");
        let got = out.values();

        for head in 0..seq * 2 {
            let before = l2(&data[head * 4..(head + 1) * 4]);
            let after = l2(&got[head * 4..(head + 1) * 4]);
            assert!(
                (before - after).abs() < 1e-5,
                "head vector {head}: norm {before} became {after}",
            );
        }
    }

    /// The defining property of RoPE: `<R_m q, R_n k>` depends only on `m - n`.
    /// So a query at position 2 against a key at position 5 must score exactly
    /// what the same vectors score at positions 0 and 3.
    ///
    /// This is the strongest single check on the rotary code — it fails for a
    /// wrong `rotate_half` sign, a wrong pairing, a transposed `cos`/`sin`, or
    /// a per-position table that is not a pure rotation — and it needs no
    /// reference values at all.
    #[test]
    fn rotary_encodes_only_relative_position() {
        let (cfg, rotary) = rotary_fixture(8, 2);
        let hd = cfg.head_dim();
        let mut r = Lcg::new(11);

        // One sequence long enough to hold both pairs of positions.
        let seq = 6;
        let qd = r.vec(seq * 2 * hd, 1.5);
        let kd = r.vec(seq * 2 * hd, 1.5);

        // Put the same two vectors at (0, 3) and at (2, 5).
        let mut q = vec![0.0f32; seq * 2 * hd];
        let mut k = vec![0.0f32; seq * 2 * hd];
        let (qv, kv) = (&qd[0..hd], &kd[0..hd]);
        for pos in [0usize, 2] {
            q[pos * 2 * hd..pos * 2 * hd + hd].copy_from_slice(qv);
        }
        for pos in [3usize, 5] {
            k[pos * 2 * hd..pos * 2 * hd + hd].copy_from_slice(kv);
        }

        let qt = backend::f32_tensor(&q, (1, seq, 2, hd));
        let kt = backend::f32_tensor(&k, (1, seq, 2, hd));
        let (qr, kr) = rotary.apply(&qt, &kt).expect("rotary applies");
        let qr = qr.values();
        let kr = kr.values();

        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let head0 = |v: &[f32], pos: usize| v[pos * 2 * hd..pos * 2 * hd + hd].to_vec();

        // distance 3, measured at two different absolute offsets
        let near = dot(&head0(&qr, 0), &head0(&kr, 3));
        let far = dot(&head0(&qr, 2), &head0(&kr, 5));

        assert!(
            (near - far).abs() < 1e-4,
            "rotary is not translation invariant: <q0,k3> = {near} but <q2,k5> = {far}",
        );

        // Guard against a vacuous pass: a no-op rotary would also satisfy the
        // check above, so require that position actually changed something.
        let unrotated = dot(qv, kv);
        assert!(
            (near - unrotated).abs() > 1e-3,
            "rotary appears to be a no-op: rotated dot {near} matches unrotated {unrotated}",
        );
    }

    /// `inv_freq` is derived from `rotary_emb_base`, so two models that differ
    /// only in that field must not agree. Guards the 1000-vs-10000 trap from
    /// the far end.
    #[test]
    fn rotary_base_changes_the_rotation() {
        let mut cfg = tiny_config(8, 2, 1, 16, 8);
        let rot_1000 = backend::rotary(&cfg).unwrap();
        cfg.rotary_emb_base = 10_000.0;
        let rot_10000 = backend::rotary(&cfg).unwrap();

        let mut r = Lcg::new(13);
        let data = r.vec(4 * 2 * 4, 1.0);
        let q = backend::f32_tensor(&data, (1, 4, 2, 4));

        let a = rot_1000.apply(&q, &q).unwrap().0;
        let b = rot_10000.apply(&q, &q).unwrap().0;
        let a = a.values();
        let b = b.values();

        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "rotary_emb_base is being ignored: base 1000 and 10000 agree to {max_diff}",
        );
    }

    // -----------------------------------------------------------------------
    // Shapes
    // -----------------------------------------------------------------------

    #[test]
    fn forward_returns_batch_seq_hidden() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 1));

        let ids = ids_tensor(&[&[1, 4, 7, 2, 9], &[3, 3, 0, 11, 5]]);
        let out = model.forward(&ids).expect("forward runs");

        assert_eq!(out.shape3(), (2, 5, cfg.hidden_size));
    }

    #[test]
    fn extract_embeddings_width_matches_hidden_size() {
        for hidden in [8usize, 16] {
            let cfg = tiny_config(hidden, 2, 1, hidden * 2, 12);
            let model = load(&cfg, &TinyWeights::deterministic(&cfg, 2));

            let out = model
                .extract_embeddings(&ids_tensor(&[&[1, 4, 7], &[2, 2, 5], &[9, 0, 3]]))
                .expect("extract runs");

            assert_eq!(
                out.shape2(),
                (3, hidden),
                "pooled embedding should be (batch, hidden_size)",
            );
            assert!(out.is_f32(), "embeddings must be F32");
        }
    }

    // -----------------------------------------------------------------------
    // Invariants of the pooled embedding
    // -----------------------------------------------------------------------

    /// `extract_embeddings` is exactly the mean over the sequence axis of
    /// `forward`. Checking the two against each other pins the pooling without
    /// needing the full reference, and catches a pool that collapses the wrong
    /// axis or weights positions unevenly.
    #[test]
    fn extract_embeddings_mean_pool_the_hidden_states() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 3));
        let rows: [&[u32]; 2] = [&[1, 4, 7, 2, 9], &[3, 3, 0, 11, 5]];
        let ids = ids_tensor(&rows);

        let hidden = model.forward(&ids).expect("forward runs");
        let (batch, seq, h) = hidden.shape3();
        let hidden = hidden.values();

        let pooled = model
            .extract_embeddings(&ids)
            .expect("extract runs")
            .values();

        for b in 0..batch {
            let expected: Vec<f32> = (0..h)
                .map(|d| (0..seq).map(|t| hidden[(b * seq + t) * h + d]).sum::<f32>() / seq as f32)
                .collect();
            assert_close(
                &pooled[b * h..(b + 1) * h],
                &expected,
                1e-6,
                &format!("row {b} pooled vs manual mean of forward()"),
            );
        }
    }

    /// The contract the embedding endpoint actually ships: pooled hidden
    /// states passed through [`crate::models::l2_normalize`] come out at unit
    /// magnitude for any input.
    ///
    /// The normalize lives in the endpoint rather than in this model (see
    /// LARES-464 — it used to be here and nowhere else, so NomicBERT returned
    /// unit vectors while Bert, Llama and Qwen2 returned raw pooled ones
    /// through the same API). This test covers the seam so the guarantee
    /// cannot quietly disappear from both halves at once.
    #[test]
    fn pooled_embeddings_are_unit_norm_once_normalized() {
        for (hidden, heads, layers, seed) in [(8usize, 2usize, 1usize, 3u64), (16, 4, 3, 4)] {
            let cfg = tiny_config(hidden, heads, layers, hidden * 2, 12);
            let model = load(&cfg, &TinyWeights::deterministic(&cfg, seed));

            for rows in [
                vec![&[5u32][..]],                             // single token
                vec![&[1, 1, 1, 1][..]],                       // repeated token
                vec![&[0, 0, 0][..]],                          // all pad
                vec![&[1, 4, 7, 2, 9][..], &[3, 3, 0, 11, 5]], // a real batch
            ] {
                let out = model
                    .extract_embeddings(&ids_tensor(&rows))
                    .expect("extract runs");
                let (batch, _) = out.shape2();
                let flat = out.values();

                for b in 0..batch {
                    let mut row = flat[b * hidden..(b + 1) * hidden].to_vec();
                    assert!(
                        row.iter().all(|v| v.is_finite()),
                        "h{hidden} batch row {b} contains non-finite values: {row:?}",
                    );

                    crate::models::l2_normalize(&mut row);
                    let n = l2(&row);
                    assert!(
                        (n - 1.0).abs() < 1e-5,
                        "h{hidden} batch row {b} has magnitude {n} after normalize, expected 1.0",
                    );
                }
            }
        }
    }

    /// Nothing in this model mixes across the batch dimension, so embedding a
    /// batch must give the same answer as embedding each row alone. Catches
    /// pooling or reshaping that collapses the wrong axis.
    #[test]
    fn extract_embeddings_rows_are_independent_across_batch() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 5));

        let rows: [&[u32]; 3] = [&[1, 4, 7, 2], &[9, 9, 0, 3], &[5, 2, 11, 6]];
        let batched = model
            .extract_embeddings(&ids_tensor(&rows))
            .expect("batched extract runs")
            .values();

        for (b, row) in rows.iter().enumerate() {
            let single = model
                .extract_embeddings(&ids_tensor(&[row]))
                .expect("single extract runs")
                .values();
            assert_close(
                &batched[b * 8..(b + 1) * 8],
                &single,
                1e-5,
                &format!("batch row {b} vs standalone"),
            );
        }
    }

    #[test]
    fn extract_embeddings_are_deterministic() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 6));
        let ids = ids_tensor(&[&[1, 4, 7, 2, 9]]);

        let a = model.extract_embeddings(&ids).unwrap().values();
        let b = model.extract_embeddings(&ids).unwrap().values();

        assert_close(&a, &b, 0.0, "repeated extraction");
    }

    /// Position information reaches the output. Without rotary the model is
    /// permutation-equivariant and mean pooling would erase the reordering, so
    /// a permuted sequence would embed identically. It must not.
    #[test]
    fn extract_embeddings_depend_on_token_order() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 8));

        let forward = model
            .extract_embeddings(&ids_tensor(&[&[1, 4, 7, 2]]))
            .unwrap()
            .values();
        let reversed = model
            .extract_embeddings(&ids_tensor(&[&[2, 7, 4, 1]]))
            .unwrap()
            .values();

        let max_diff = forward
            .iter()
            .zip(&reversed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "reordering the sequence changed nothing (max diff {max_diff}) — \
             positional information is not reaching the output",
        );
    }

    /// With every projection at zero the whole forward pass collapses to zero:
    /// `LayerNorm(0)` is `0 / sqrt(0 + eps) = 0`, attention averages zeros, and
    /// the MLP's `silu(0) = 0` kills the gate. Mean-pooling zeros gives a zero
    /// vector — the degenerate input that a naive L2 divide turns into NaNs.
    ///
    /// The divide now lives in [`crate::models::l2_normalize`], which leaves a
    /// zero-norm vector alone; `l2_normalize_leaves_degenerate_vectors_alone`
    /// covers that half. This test covers the half that produces the zero
    /// vector in the first place, and asserts it stays finite end to end.
    #[test]
    fn zero_weights_produce_zero_embedding_without_nan() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::zeroed(&cfg));

        let hidden = model
            .forward(&ids_tensor(&[&[1, 4, 7, 2]]))
            .expect("forward runs")
            .values();
        assert_close(
            &hidden,
            &vec![0.0; hidden.len()],
            0.0,
            "zero-weight hidden states",
        );

        let emb = model
            .extract_embeddings(&ids_tensor(&[&[1, 4, 7, 2]]))
            .expect("extract runs")
            .values();

        assert!(
            emb.iter().all(|v| v.is_finite()),
            "zero-norm embedding produced non-finite values: {emb:?}",
        );
        assert_close(&emb, &vec![0.0; emb.len()], 0.0, "zero-weight embedding");

        // End to end through the endpoint's normalize: still zeros, no NaNs.
        let mut normalized = emb.clone();
        crate::models::l2_normalize(&mut normalized);
        assert!(
            normalized.iter().all(|v| v.is_finite()),
            "normalizing a zero embedding produced non-finite values: {normalized:?}",
        );
        assert_close(&normalized, &emb, 0.0, "normalized zero-weight embedding");
    }

    // -----------------------------------------------------------------------
    // Fused QKV
    // -----------------------------------------------------------------------

    /// Pins the order of the fused QKV split without needing a golden vector.
    ///
    /// The fixture zeroes the Q and K blocks of `Wqkv` and sets the V block to
    /// the identity, with `out_proj` also the identity and the MLP zeroed.
    /// Then `q = k = 0`, so every attention score is 0, so softmax is uniform
    /// and each position attends equally to all of them:
    ///
    /// ```text
    /// attn(x)_t = mean_u v_u = mean_u x_u = m        (the same m for every t)
    /// out_t     = norm2(0 + norm1(m + x_t)) = LN(LN(m + x_t)) = LN(m + x_t)
    /// ```
    ///
    /// (The outer `LN` is a no-op: its input already has zero mean and unit
    /// variance, and `layer_norm_eps` is 1e-12.)
    ///
    /// If the split ran in any other order the V block would read as zeros,
    /// attention would contribute nothing, and `out_t` would be `LN(x_t)` —
    /// which the final assertion explicitly rules out, so the test cannot pass
    /// by accident.
    #[test]
    fn qkv_splits_as_query_key_value_in_that_order() {
        let h = 4usize;
        let cfg = tiny_config(h, 1, 1, 4, 6);
        let identity: Vec<f32> = (0..h * h).map(|i| f32::from(i / h == i % h)).collect();

        let mut r = Lcg::new(21);
        let mut w = TinyWeights::zeroed(&cfg);
        w.word_emb = r.vec(cfg.vocab_size * h, 2.0);
        // Q block and K block stay zero; V block (rows 2h..3h) is the identity.
        w.layers[0].wqkv[2 * h * h..].copy_from_slice(&identity);
        w.layers[0].out_proj.copy_from_slice(&identity);

        let ids: Vec<u32> = vec![1, 3, 5, 2];
        let got = load(&cfg, &w)
            .forward(&ids_tensor(&[&ids]))
            .expect("forward runs")
            .values();

        // x_t = emb_ln(word_emb[id_t]); the token-type table is all zeros.
        let xs: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| {
                let row = &w.word_emb[id as usize * h..(id as usize + 1) * h];
                reference::layer_norm(row, &w.emb_ln_w, &w.emb_ln_b, EPS_F32)
            })
            .collect();
        let mut m = vec![0.0f32; h];
        for x in &xs {
            for (acc, v) in m.iter_mut().zip(x) {
                *acc += v / xs.len() as f32;
            }
        }

        for (t, x) in xs.iter().enumerate() {
            let sum: Vec<f32> = m.iter().zip(x).map(|(a, b)| a + b).collect();
            let expected = reference::layer_norm(&sum, &vec![1.0; h], &vec![0.0; h], EPS_F32);
            assert_close(
                &got[t * h..(t + 1) * h],
                &expected,
                1e-5,
                &format!("position {t} with V=I, Q=K=0"),
            );

            // Non-vacuity: LN(m + x_t) must be distinguishable from LN(x_t),
            // otherwise a zeroed V block would pass this test too.
            let without = reference::layer_norm(x, &vec![1.0; h], &vec![0.0; h], EPS_F32);
            let diff = expected
                .iter()
                .zip(&without)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff > 1e-2,
                "fixture is vacuous at position {t}: the sequence mean barely \
                 changes the result (max diff {diff})",
            );
        }
    }

    // -----------------------------------------------------------------------
    // Whole forward pass against the independent reference
    // -----------------------------------------------------------------------

    /// The known-good-vector test. `reference::forward` is a naive scalar
    /// transcription of the published NomicBERT architecture; agreeing with it
    /// across a multi-layer, multi-head fixture pins every structural choice
    /// at once — the QKV split order and head layout, the rotary convention
    /// and base, the attention scale, the post-norm residual order, and the
    /// SwiGLU operand order.
    ///
    /// A transposed QKV split, a swapped `fc11`/`fc12`, or a pre-norm block
    /// all produce O(1) disagreement here.
    #[test]
    fn forward_matches_independent_reference() {
        let cfg = tiny_config(8, 2, 3, 16, 12);
        let w = TinyWeights::deterministic(&cfg, 42);
        let model = load(&cfg, &w);

        let ids: Vec<u32> = vec![1, 4, 7, 2, 9];
        let got = model
            .forward(&ids_tensor(&[&ids]))
            .expect("forward runs")
            .values();

        let expected = reference::forward(&cfg, &w, &ids);
        let flat: Vec<f32> = expected.into_iter().flatten().collect();

        assert_close(&got, &flat, 2e-4, "hidden states vs reference");

        // Guard against a reference that has collapsed to something trivial.
        assert!(
            l2(&flat) > 1.0,
            "reference output is degenerate; the fixture proves nothing",
        );
    }

    #[test]
    fn extract_embeddings_match_independent_reference() {
        let cfg = tiny_config(8, 2, 3, 16, 12);
        let w = TinyWeights::deterministic(&cfg, 42);
        let model = load(&cfg, &w);

        let rows: [&[u32]; 2] = [&[1, 4, 7, 2, 9], &[3, 3, 0, 11, 5]];
        let got = model
            .extract_embeddings(&ids_tensor(&rows))
            .expect("extract runs")
            .values();

        for (b, ids) in rows.iter().enumerate() {
            let expected = reference::extract(&cfg, &w, ids);
            assert_close(
                &got[b * cfg.hidden_size..(b + 1) * cfg.hidden_size],
                &expected,
                2e-4,
                &format!("pooled embedding row {b} vs reference"),
            );
            // Guard against a reference that has collapsed to something
            // trivial: a degenerate all-zero expectation would match almost
            // any broken implementation.
            assert!(
                l2(&expected) > 1e-3,
                "reference embedding row {b} is degenerate; the fixture proves nothing",
            );
        }
    }

    // -----------------------------------------------------------------------
    // out_proj bias (LARES-465 item 1)
    // -----------------------------------------------------------------------

    /// The checkpoint for `cfg`, plus the QKV/out_proj bias entries a
    /// `qkv_proj_bias: true` checkpoint would carry.
    fn weights_with_attn_bias(
        cfg: &NomicBertConfig,
        w: &TinyWeights,
        out_proj_bias: Option<&[f32]>,
    ) -> Vec<NamedWeight> {
        let h = cfg.hidden_size;
        let mut ws = w.named_weights(cfg);
        for i in 0..cfg.num_hidden_layers {
            ws.push(NamedWeight::new(
                format!("encoder.layers.{i}.attn.Wqkv.bias"),
                &vec![0.0; 3 * h],
                [3 * h],
            ));
            if let Some(b) = out_proj_bias {
                ws.push(NamedWeight::new(
                    format!("encoder.layers.{i}.attn.out_proj.bias"),
                    b,
                    [h],
                ));
            }
        }
        ws
    }

    /// With `qkv_proj_bias: true` the loader must actually *ask* for
    /// `out_proj.bias`, so a checkpoint missing it fails loudly.
    ///
    /// This is the regression guard for the original defect: `out_proj` was
    /// built with `bias = false` hardcoded, so the tensor was never requested.
    /// A checkpoint carrying one loaded happily and silently ignored it, and
    /// this assertion would have passed vacuously — hence the companion test
    /// below, which proves the bias actually reaches the output.
    #[test]
    fn out_proj_bias_is_required_when_qkv_proj_bias_is_set() {
        let mut cfg = tiny_config(4, 1, 1, 4, 6);
        cfg.qkv_proj_bias = true;
        let w = TinyWeights::deterministic(&cfg, 31);

        let missing = weights_with_attn_bias(&cfg, &w, None);
        assert!(
            backend::load_model(&cfg, missing).is_err(),
            "loading should fail when qkv_proj_bias is set but out_proj.bias is absent",
        );

        // The same fixture with the bias present loads fine, so the failure
        // above is about the bias and not something else in the map.
        let complete = weights_with_attn_bias(&cfg, &w, Some(&[0.0; 4]));
        assert!(
            backend::load_model(&cfg, complete).is_ok(),
            "loading should succeed once out_proj.bias is supplied",
        );
    }

    /// The bias reaches the output, with an analytically derived expectation.
    ///
    /// Every projection is zeroed and both layer norms are at identity gain,
    /// so `q = k = v = 0` and the attention output is exactly `out_proj`'s
    /// bias `b`, the same at every position. The MLP contributes nothing:
    ///
    /// ```text
    /// out_t = norm2(0 + norm1(b + x_t)) = LN(LN(b + x_t)) = LN(b + x_t)
    /// ```
    ///
    /// Dropping the bias would give `LN(x_t)`, which the final assertion rules
    /// out explicitly.
    #[test]
    fn out_proj_bias_is_applied_to_the_attention_output() {
        let h = 4usize;
        let mut cfg = tiny_config(h, 1, 1, 4, 6);
        cfg.qkv_proj_bias = true;

        let mut r = Lcg::new(33);
        let mut w = TinyWeights::zeroed(&cfg);
        w.word_emb = r.vec(cfg.vocab_size * h, 2.0);
        let bias: Vec<f32> = vec![0.5, -1.25, 0.75, -0.5];

        let ws = weights_with_attn_bias(&cfg, &w, Some(&bias));
        let model = backend::load_model(&cfg, ws).expect("model loads");

        let ids: Vec<u32> = vec![1, 3, 5, 2];
        let got = model
            .forward(&ids_tensor(&[&ids]))
            .expect("forward runs")
            .values();

        let ones = vec![1.0f32; h];
        let zeros = vec![0.0f32; h];
        for (t, &id) in ids.iter().enumerate() {
            let row = &w.word_emb[id as usize * h..(id as usize + 1) * h];
            let x = reference::layer_norm(row, &w.emb_ln_w, &w.emb_ln_b, EPS_F32);

            let sum: Vec<f32> = bias.iter().zip(&x).map(|(a, b)| a + b).collect();
            let expected = reference::layer_norm(&sum, &ones, &zeros, EPS_F32);
            assert_close(
                &got[t * h..(t + 1) * h],
                &expected,
                1e-5,
                &format!("position {t} with out_proj bias"),
            );

            // Non-vacuity: the bias must actually move the result.
            let without = reference::layer_norm(&x, &ones, &zeros, EPS_F32);
            let diff = expected
                .iter()
                .zip(&without)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                diff > 1e-2,
                "fixture is vacuous at position {t}: the bias barely changes the \
                 result (max diff {diff})",
            );
        }
    }

    // -----------------------------------------------------------------------
    // Partial rotary (LARES-465 item 3)
    // -----------------------------------------------------------------------

    /// `rotary_emb_fraction < 1.0` rotates only the leading `rotary_dim`
    /// components and passes the rest through untouched, matching the
    /// reference's `cat([rotated_prefix, x[..., ro_dim:]])`.
    ///
    /// This configuration previously failed outright: the cos/sin tables were
    /// built `rotary_dim` wide and broadcast against a `head_dim`-wide vector.
    #[test]
    fn partial_rotary_rotates_only_the_leading_components() {
        let mut cfg = tiny_config(8, 1, 1, 16, 8);
        cfg.rotary_emb_fraction = 0.5; // head_dim 8 → rotary_dim 4
        assert_eq!(cfg.rotary_dim(), 4);

        let rotary = backend::rotary(&cfg).expect("rotary table builds");

        let x = [0.3f32, -0.7, 1.1, 0.25, 9.0, -9.5, 2.5, -2.75];
        let q = backend::f32_tensor(
            &[
                0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // position 0
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], // position 1
            ],
            (1, 2, 1, 8),
        );

        let (out, _) = rotary.apply(&q, &q).expect("rotary applies");
        let got = out.values();

        // rotary_dim 4 pairs (0,2) and (1,3), turned by 1 * inv_freq[i] with
        // inv_freq = [1/1000^0, 1/1000^(2/4)].
        let a0 = 1.0f32;
        let a1 = 1.0f32 / 1000f32.powf(0.5);
        let expected_rotated = [
            x[0] * a0.cos() - x[2] * a0.sin(),
            x[1] * a1.cos() - x[3] * a1.sin(),
            x[2] * a0.cos() + x[0] * a0.sin(),
            x[3] * a1.cos() + x[1] * a1.sin(),
        ];

        assert_close(
            &got[8..12],
            &expected_rotated,
            1e-6,
            "rotated prefix at position 1",
        );
        assert_close(
            &got[12..16],
            &x[4..8],
            0.0,
            "components past rotary_dim must pass through untouched",
        );
    }

    /// `rotary_emb_fraction: 0.0` disables rotary entirely rather than
    /// erroring, matching the reference's `if self.rotary_emb_dim > 0` guard.
    #[test]
    fn zero_rotary_fraction_is_the_identity() {
        let mut cfg = tiny_config(8, 2, 1, 16, 8);
        cfg.rotary_emb_fraction = 0.0;
        assert_eq!(cfg.rotary_dim(), 0);

        let rotary = backend::rotary(&cfg).expect("builds");
        let mut r = Lcg::new(35);
        let data = r.vec(3 * 2 * 4, 1.5);
        let q = backend::f32_tensor(&data, (1, 3, 2, 4));

        let (out, _) = rotary.apply(&q, &q).expect("applies");
        assert_close(
            &out.values(),
            &data,
            0.0,
            "zero rotary fraction should leave the vector alone",
        );
    }

    /// An odd rotary width cannot be split into rotation pairs. Rejecting it
    /// at construction beats a shape-mismatch panic from inside `apply`.
    #[test]
    fn odd_rotary_dimension_is_rejected_with_a_clear_error() {
        let mut cfg = tiny_config(8, 2, 1, 16, 8); // head_dim 4
        cfg.rotary_emb_fraction = 0.75; // → rotary_dim 3

        assert_eq!(cfg.rotary_dim(), 3);
        let msg = match backend::rotary(&cfg) {
            Ok(_) => panic!("odd rotary dim should be rejected"),
            Err(e) => e,
        };
        assert!(
            msg.contains("even"),
            "error should explain the constraint, got: {msg}",
        );
    }

    /// A partial-rotary model runs end to end. Before the fix this panicked
    /// inside the attention on a broadcast shape mismatch.
    #[test]
    fn partial_rotary_model_runs_end_to_end() {
        let mut cfg = tiny_config(8, 2, 2, 16, 12);
        cfg.rotary_emb_fraction = 0.5;
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 37));

        let out = model
            .extract_embeddings(&ids_tensor(&[&[1, 4, 7, 2], &[9, 0, 3, 5]]))
            .expect("partial-rotary forward should not error");

        assert_eq!(out.shape2(), (2, 8));
        let flat = out.values();
        assert!(flat.iter().all(|v| v.is_finite()), "got {flat:?}");
    }

    // -----------------------------------------------------------------------
    // Attention mask (LARES-465 item 2)
    // -----------------------------------------------------------------------

    fn mask_tensor(rows: &[&[u32]]) -> backend::Ids {
        ids_tensor(rows)
    }

    /// The guarantee that makes batching variable-length inputs safe: padding
    /// a sequence, with a mask that marks the padding, must not change its
    /// embedding.
    ///
    /// Without the mask it does change it — padding tokens are attended to and
    /// pooled over — which is why the unmasked comparison is asserted to
    /// differ. That second assertion is what stops this test passing for the
    /// wrong reason.
    #[test]
    fn padding_does_not_change_an_embedding_when_masked() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 41));

        let bare = model
            .extract_embeddings(&ids_tensor(&[&[1, 4, 7]]))
            .expect("bare extract runs")
            .values();

        let padded_ids = ids_tensor(&[&[1, 4, 7, 0, 0]]);
        let mask = mask_tensor(&[&[1, 1, 1, 0, 0]]);

        let masked = model
            .extract_embeddings_with_mask(&padded_ids, Some(&mask))
            .expect("masked extract runs")
            .values();
        assert_close(&masked, &bare, 1e-5, "masked padded vs unpadded");

        // Same input without the mask: padding leaks in.
        let unmasked = model
            .extract_embeddings(&padded_ids)
            .expect("unmasked extract runs")
            .values();
        let diff = unmasked
            .iter()
            .zip(&bare)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-3,
            "padding should change the result when unmasked (max diff {diff}); \
             if it does not, this fixture cannot prove the mask does anything",
        );
    }

    /// What sits in a masked position is irrelevant — attention never sees it
    /// and pooling never counts it.
    #[test]
    fn masked_positions_do_not_affect_the_output() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 43));
        let mask = mask_tensor(&[&[1, 1, 1, 0, 0]]);

        let a = model
            .extract_embeddings_with_mask(&ids_tensor(&[&[1, 4, 7, 0, 0]]), Some(&mask))
            .unwrap()
            .values();
        let b = model
            .extract_embeddings_with_mask(&ids_tensor(&[&[1, 4, 7, 11, 9]]), Some(&mask))
            .unwrap()
            .values();

        assert_close(&a, &b, 1e-5, "differing padding content");
    }

    /// The point of the whole exercise: a padded batch embeds each row exactly
    /// as if that row had been embedded alone.
    #[test]
    fn padded_batch_rows_match_their_standalone_embeddings() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 45));

        let sequences: [&[u32]; 3] = [&[1, 4, 7, 2, 9], &[3, 3], &[5, 2, 11]];
        let width = 5;

        let padded: Vec<Vec<u32>> = sequences
            .iter()
            .map(|s| {
                let mut v = s.to_vec();
                v.resize(width, 0);
                v
            })
            .collect();
        let masks: Vec<Vec<u32>> = sequences
            .iter()
            .map(|s| {
                let mut v = vec![1u32; s.len()];
                v.resize(width, 0);
                v
            })
            .collect();

        let padded_refs: Vec<&[u32]> = padded.iter().map(|v| v.as_slice()).collect();
        let mask_refs: Vec<&[u32]> = masks.iter().map(|v| v.as_slice()).collect();

        let batched = model
            .extract_embeddings_with_mask(&ids_tensor(&padded_refs), Some(&mask_tensor(&mask_refs)))
            .expect("batched masked extract runs")
            .values();

        for (b, seq) in sequences.iter().enumerate() {
            let alone = model
                .extract_embeddings(&ids_tensor(&[seq]))
                .expect("standalone extract runs")
                .values();
            assert_close(
                &batched[b * 8..(b + 1) * 8],
                &alone,
                1e-5,
                &format!("padded batch row {b} (len {}) vs standalone", seq.len()),
            );
        }
    }

    /// A row that is entirely padding divides by a zero count. It must come
    /// back as zeros rather than NaNs, so one empty input cannot poison a
    /// batch.
    #[test]
    fn all_padding_row_pools_to_zero_not_nan() {
        let cfg = tiny_config(8, 2, 1, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 47));

        let out = model
            .extract_embeddings_with_mask(
                &ids_tensor(&[&[1, 4, 7], &[0, 0, 0]]),
                Some(&mask_tensor(&[&[1, 1, 1], &[0, 0, 0]])),
            )
            .expect("extract runs")
            .values();

        assert!(
            out.iter().all(|v| v.is_finite()),
            "all-padding row produced non-finite values: {out:?}",
        );
        assert_close(&out[8..16], &vec![0.0; 8], 0.0, "all-padding row");
        assert!(
            l2(&out[0..8]) > 1e-3,
            "the real row should still carry signal",
        );
    }

    /// A mask that does not line up with the input is a caller bug worth
    /// naming, not a silent broadcast.
    #[test]
    fn mismatched_mask_shape_is_rejected() {
        let cfg = tiny_config(8, 2, 1, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 49));

        let err = model
            .extract_embeddings_with_mask(
                &ids_tensor(&[&[1, 4, 7]]),
                Some(&mask_tensor(&[&[1, 1, 1, 1]])),
            )
            .expect_err("a mask of the wrong width should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("attention_mask"),
            "error should name the offending argument, got: {msg}",
        );
    }

    /// An all-ones mask is the same as no mask at all.
    #[test]
    fn full_mask_matches_the_unmasked_path() {
        let cfg = tiny_config(8, 2, 2, 16, 12);
        let model = load(&cfg, &TinyWeights::deterministic(&cfg, 51));
        let ids = ids_tensor(&[&[1, 4, 7, 2]]);

        let unmasked = model.extract_embeddings(&ids).unwrap().values();
        let masked = model
            .extract_embeddings_with_mask(&ids, Some(&mask_tensor(&[&[1, 1, 1, 1]])))
            .unwrap()
            .values();

        assert_close(&masked, &unmasked, 1e-6, "all-ones mask vs no mask");
    }
}
