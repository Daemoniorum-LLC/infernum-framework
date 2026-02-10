//! Quantized Llama model implementation for GGUF files.
//!
//! This module provides inference support for quantized Llama models
//! loaded from GGUF format files.

use std::path::Path;

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};

use crate::gguf::{GgufLoader, QuantizedModelConfig};

/// Quantized RMS Layer Normalization.
struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new(weight: QTensor, eps: f64) -> CandleResult<Self> {
        let weight = weight.dequantize(&Device::Cpu)?;
        Ok(Self { weight, eps })
    }
}

impl Module for QRmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// Quantized linear layer.
struct QLinear {
    weight: QTensor,
}

impl QLinear {
    fn new(weight: QTensor) -> Self {
        Self { weight }
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_shape = x.dims();
        let x = if x_shape.len() > 2 {
            x.reshape(((), x_shape[x_shape.len() - 1]))?
        } else {
            x.clone()
        };

        // Dequantize weight and perform matrix multiplication
        let weight = self.weight.dequantize(x.device())?;
        let result = x.matmul(&weight.t()?)?;

        if x_shape.len() > 2 {
            let mut new_shape = x_shape[..x_shape.len() - 1].to_vec();
            new_shape.push(result.dim(D::Minus1)?);
            result.reshape(new_shape)
        } else {
            Ok(result)
        }
    }
}

/// Quantized attention layer.
struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QAttention {
    fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // Double cos/sin for neox-style rotary embedding
        let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
        let sin = Tensor::cat(&[sin, sin], D::Minus1)?;

        // Apply rotary embeddings (neox-style)
        let q = Self::apply_rotary(&q, &cos, &sin)?;
        let k = Self::apply_rotary(&k, &cos, &sin)?;

        // Transpose for attention: (batch, heads, seq, dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Handle KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) if start_pos > 0 => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            },
            _ => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let repeat_factor = self.num_heads / self.num_kv_heads;
        let k = if repeat_factor > 1 {
            k.repeat(&[1, repeat_factor, 1, 1])?
        } else {
            k
        };
        let v = if repeat_factor > 1 {
            v.repeat(&[1, repeat_factor, 1, 1])?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    /// Applies neox-style rotary embedding: x' = x * cos + rotate_half(x) * sin
    fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // x shape: (batch, seq, heads, head_dim)
        // cos/sin shape: (seq, head_dim) -> (1, seq, 1, head_dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

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

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Quantized MLP layer.
struct QMlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QMlp {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Quantized decoder layer.
struct QDecoderLayer {
    self_attn: QAttention,
    mlp: QMlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

impl QDecoderLayer {
    fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, cos, sin, mask, start_pos)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

/// Quantized Llama model.
pub struct QuantizedLlama {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QDecoderLayer>,
    norm: QRmsNorm,
    lm_head: QLinear,
    cos_cache: Tensor,
    sin_cache: Tensor,
    config: QuantizedModelConfig,
    device: Device,
}

impl QuantizedLlama {
    /// Loads a quantized Llama model from a GGUF file.
    pub fn from_gguf(path: impl AsRef<Path>, device: &Device) -> CandleResult<Self> {
        let loader = GgufLoader::from_file(&path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load GGUF: {}", e)))?;

        let metadata = loader.metadata();
        let content = loader.content();
        let config = QuantizedModelConfig::from(metadata);

        tracing::info!(
            architecture = %config.architecture,
            layers = config.num_layers,
            hidden_size = config.hidden_size,
            "Loading quantized model"
        );

        // Load embedding layer
        let embed_tokens = Self::load_embedding(content, &config, device)?;

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = Self::load_layer(content, &config, i, device)?;
            layers.push(layer);
        }

        // Load final norm
        let norm_weight = Self::get_tensor(content, "model.norm.weight")?;
        let norm = QRmsNorm::new(norm_weight, config.rms_norm_eps)?;

        // Load lm_head
        let lm_head = if let Ok(weight) = Self::get_tensor(content, "lm_head.weight") {
            QLinear::new(weight)
        } else {
            // Tied embeddings - use embed_tokens weight
            let weight = Self::get_tensor(content, "model.embed_tokens.weight")?;
            QLinear::new(weight)
        };

        // Build rotary embedding cache
        let (cos_cache, sin_cache) = Self::build_rotary_cache(&config, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
            config,
            device: device.clone(),
        })
    }

    fn load_embedding(
        content: &gguf_file::Content,
        config: &QuantizedModelConfig,
        device: &Device,
    ) -> CandleResult<candle_nn::Embedding> {
        let weight = Self::get_tensor(content, "model.embed_tokens.weight")?;
        let weight = weight.dequantize(device)?;
        Ok(candle_nn::Embedding::new(weight, config.hidden_size))
    }

    fn load_layer(
        content: &gguf_file::Content,
        config: &QuantizedModelConfig,
        layer_idx: usize,
        _device: &Device,
    ) -> CandleResult<QDecoderLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        let head_dim = config.hidden_size / config.num_attention_heads;

        // Load attention
        let q_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.self_attn.q_proj.weight", prefix),
        )?);
        let k_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.self_attn.k_proj.weight", prefix),
        )?);
        let v_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.self_attn.v_proj.weight", prefix),
        )?);
        let o_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.self_attn.o_proj.weight", prefix),
        )?);

        let self_attn = QAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            kv_cache: None,
        };

        // Load MLP
        let gate_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.mlp.gate_proj.weight", prefix),
        )?);
        let up_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.mlp.up_proj.weight", prefix),
        )?);
        let down_proj = QLinear::new(Self::get_tensor(
            content,
            &format!("{}.mlp.down_proj.weight", prefix),
        )?);

        let mlp = QMlp {
            gate_proj,
            up_proj,
            down_proj,
        };

        // Load norms
        let input_layernorm = QRmsNorm::new(
            Self::get_tensor(content, &format!("{}.input_layernorm.weight", prefix))?,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = QRmsNorm::new(
            Self::get_tensor(
                content,
                &format!("{}.post_attention_layernorm.weight", prefix),
            )?,
            config.rms_norm_eps,
        )?;

        Ok(QDecoderLayer {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn get_tensor(content: &gguf_file::Content, name: &str) -> CandleResult<QTensor> {
        content.tensor(
            &mut std::io::BufReader::new(std::io::Cursor::new(&[])),
            name,
            &Device::Cpu,
        )
    }

    fn build_rotary_cache(
        config: &QuantizedModelConfig,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.context_length;
        let theta = config.rope_theta;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok((cos, sin))
    }

    /// Forward pass for the quantized model.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Get rotary embeddings for this sequence
        let cos = self.cos_cache.narrow(0, start_pos, seq_len)?;
        let sin = self.sin_cache.narrow(0, start_pos, seq_len)?;

        // Create causal mask
        let mask = if seq_len > 1 {
            Some(Self::create_causal_mask(seq_len, start_pos, &self.device)?)
        } else {
            None
        };

        // Forward through layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, mask.as_ref(), start_pos)?;
        }

        // Final layer norm
        let hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        self.lm_head.forward(&hidden_states)
    }

    fn create_causal_mask(
        seq_len: usize,
        start_pos: usize,
        device: &Device,
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

        Tensor::from_vec(mask, (seq_len, seq_len + start_pos), device)
    }

    /// Clears the KV cache.
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &QuantizedModelConfig {
        &self.config
    }
}
