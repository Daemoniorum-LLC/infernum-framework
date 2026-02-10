//! Compute engine orchestrating CUDA operations for inference.
//!
//! The ComputeEngine implements the complete transformer forward pass:
//!
//! ```text
//! Input IDs
//!     ↓
//! Token Embeddings
//!     ↓
//! For each layer:
//!   ├── RMSNorm (input)
//!   ├── Self-Attention (Q/K/V proj → RoPE → Flash Attention → O proj)
//!   ├── Residual Add
//!   ├── RMSNorm (post-attn)
//!   ├── MLP (gate+up proj → SiLU*gate → down proj)
//!   └── Residual Add
//!     ↓
//! Final RMSNorm
//!     ↓
//! LM Head → Logits
//! ```

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::arch::{Activation, ModelConfig};
use super::cublas::CublasHandle;
use super::kernels::{
    ActivationKernel, ActivationType, EmbeddingKernel, FlashAttentionKernel, FusedGemmKernel,
    FusedRMSNormProjKernel, RMSNormKernel, RoPEKernel,
};
use super::kv_cache::KvCache;
use super::streams::StreamManager;
use super::tensor::{GpuDType, GpuTensor};
use super::weight_store::{LayerWeights, QuantFormat, QuantizedWeight, WeightStore};
use super::InferenceError;

/// Main compute engine for transformer inference.
pub struct ComputeEngine {
    /// CUDA device.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,

    /// Model configuration.
    config: ModelConfig,

    /// Stream manager for async operations.
    streams: StreamManager,

    /// cuBLAS handle for GEMM operations.
    cublas: CublasHandle,

    /// RMSNorm kernel.
    rmsnorm: RMSNormKernel,

    /// RoPE kernel.
    rope: RoPEKernel,

    /// Activation kernel.
    activation: ActivationKernel,

    /// Fused GEMM kernel for quantized weights.
    fused_gemm: FusedGemmKernel,

    /// Attention kernel.
    attention: FlashAttentionKernel,

    /// Embedding lookup kernel.
    embedding: EmbeddingKernel,

    /// Fused RMSNorm + projection kernel.
    fused_rmsnorm_proj: FusedRMSNormProjKernel,

    /// KV cache.
    kv_cache: KvCache,

    // === Reusable buffers ===
    /// Hidden state buffer [seq_len, hidden_size].
    hidden_buffer: GpuTensor,

    /// Attention output buffer [seq_len, hidden_size].
    attn_buffer: GpuTensor,

    /// MLP intermediate buffer [seq_len, intermediate_size].
    mlp_buffer: GpuTensor,

    /// Query tensor [seq_len, num_heads, head_dim].
    q_buffer: GpuTensor,

    /// Key tensor [seq_len, num_kv_heads, head_dim].
    k_buffer: GpuTensor,

    /// Value tensor [seq_len, num_kv_heads, head_dim].
    v_buffer: GpuTensor,

    /// Attention output (pre-projection) [seq_len, num_heads, head_dim].
    attn_out_buffer: GpuTensor,

    /// Gate projection output [seq_len, intermediate_size].
    gate_buffer: GpuTensor,

    /// Up projection output [seq_len, intermediate_size].
    up_buffer: GpuTensor,

    /// Position indices buffer.
    positions: GpuTensor,

    /// Token IDs buffer for GPU embedding lookup.
    token_ids_buffer: GpuTensor,

    /// Logits output buffer [max_seq_len, vocab_size].
    /// Pre-allocated to avoid allocation during forward.
    logits_buffer: GpuTensor,

    /// Current sequence position (for generation).
    current_pos: usize,

    /// Last logits from forward pass (for sampling).
    /// Shape: [seq_len, vocab_size] or None if no forward pass yet.
    last_logits: Option<GpuTensor>,
}

impl ComputeEngine {
    /// Create a new compute engine.
    pub fn new(
        config: ModelConfig,
        max_seq_len: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        // Create stream manager for async operations
        let streams = StreamManager::new(Arc::clone(&device))?;

        let cublas = CublasHandle::new(Arc::clone(&device))?;

        // Set cuBLAS to use compute stream for async execution
        unsafe {
            cublas.set_stream(streams.compute_raw())?;
        }

        let rmsnorm = RMSNormKernel::new(Arc::clone(&device))?;
        let rope = RoPEKernel::new(Arc::clone(&device))?;
        let activation = ActivationKernel::new(Arc::clone(&device))?;
        let fused_gemm = FusedGemmKernel::new(Arc::clone(&device))?;
        let attention = FlashAttentionKernel::new(Arc::clone(&device))?;
        let embedding = EmbeddingKernel::new(Arc::clone(&device))?;
        let fused_rmsnorm_proj = FusedRMSNormProjKernel::new(Arc::clone(&device))?;

        let kv_cache = KvCache::new(&config, max_seq_len, Arc::clone(&device))?;

        tracing::info!(
            "ComputeEngine config: num_heads={}, kv_heads={}, head_dim={}, hidden={}",
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            config.hidden_size
        );

        // Allocate activation buffers
        let hidden_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.hidden_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let attn_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.hidden_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let mlp_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.intermediate_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        // Q/K/V buffers - 2D for GEMM, will be reshaped to 3D for attention
        let q_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.num_attention_heads * config.head_dim],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let k_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.num_kv_heads * config.head_dim],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let v_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.num_kv_heads * config.head_dim],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let attn_out_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.num_attention_heads * config.head_dim],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        // MLP gate/up buffers
        let gate_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.intermediate_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        let up_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.intermediate_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        // Position buffer
        let positions = GpuTensor::zeros(vec![max_seq_len], GpuDType::I32, Arc::clone(&device))?;

        // Token IDs buffer for embedding lookup
        let token_ids_buffer =
            GpuTensor::zeros(vec![max_seq_len], GpuDType::I32, Arc::clone(&device))?;

        // Logits buffer - pre-allocated to avoid allocation during forward
        let logits_buffer = GpuTensor::zeros(
            vec![max_seq_len, config.vocab_size],
            GpuDType::F16,
            Arc::clone(&device),
        )?;

        Ok(Self {
            device,
            config,
            streams,
            cublas,
            rmsnorm,
            rope,
            activation,
            fused_gemm,
            attention,
            embedding,
            fused_rmsnorm_proj,
            kv_cache,
            hidden_buffer,
            attn_buffer,
            mlp_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            attn_out_buffer,
            gate_buffer,
            up_buffer,
            positions,
            token_ids_buffer,
            logits_buffer,
            current_pos: 0,
            last_logits: None,
        })
    }

    /// Perform a linear projection with quantized or F16 weights.
    fn linear(
        &mut self,
        input: &GpuTensor,
        weight: &QuantizedWeight,
        output: &mut GpuTensor,
    ) -> Result<(), InferenceError> {
        match weight.format {
            QuantFormat::Int4 => {
                // Use fused INT4 dequant + GEMM (symmetric quantization)
                let scales = weight
                    .scales
                    .as_ref()
                    .ok_or_else(|| InferenceError::Shape {
                        expected: "INT4 weights require scales".to_string(),
                        got: "no scales".to_string(),
                    })?;

                self.fused_gemm
                    .forward_int4(input, &weight.data, scales, output)
            },
            QuantFormat::Gptq => {
                // GPTQ uses asymmetric INT4 with zero points and optional g_idx for act_order
                let scales = weight
                    .scales
                    .as_ref()
                    .ok_or_else(|| InferenceError::Shape {
                        expected: "GPTQ weights require scales".to_string(),
                        got: "no scales".to_string(),
                    })?;

                let zeros = weight
                    .zero_points
                    .as_ref()
                    .ok_or_else(|| InferenceError::Shape {
                        expected: "GPTQ weights require zero_points".to_string(),
                        got: "no zero_points".to_string(),
                    })?;

                // Use dedicated GPTQ kernel with g_idx support for act_order
                self.fused_gemm.forward_gptq(
                    input,
                    &weight.data,
                    scales,
                    zeros,
                    weight.g_idx.as_ref(),
                    output,
                    weight.block_size,
                )
            },
            QuantFormat::Awq => {
                // AWQ uses asymmetric INT4 with zero points (sequential groups, no g_idx)
                let scales = weight
                    .scales
                    .as_ref()
                    .ok_or_else(|| InferenceError::Shape {
                        expected: "AWQ weights require scales".to_string(),
                        got: "no scales".to_string(),
                    })?;

                let zeros = weight
                    .zero_points
                    .as_ref()
                    .ok_or_else(|| InferenceError::Shape {
                        expected: "AWQ weights require zero_points".to_string(),
                        got: "no zero_points".to_string(),
                    })?;

                // Use optimized AWQ kernel (sequential groups, no g_idx lookup)
                self.fused_gemm.forward_awq(
                    input,
                    &weight.data,
                    scales,
                    zeros,
                    output,
                    weight.block_size,
                )
            },
            QuantFormat::F16 | QuantFormat::BF16 => {
                // Direct F16 GEMM
                if weight.transposed {
                    // Weight is in PyTorch format [out, in], use A @ B^T
                    self.fused_gemm.forward_f16_bt(input, &weight.data, output)
                } else {
                    // Weight is in standard format [in, out], use A @ B
                    self.fused_gemm.forward_f16(input, &weight.data, output)
                }
            },
            QuantFormat::Int8 => {
                // INT8 not yet implemented
                Err(InferenceError::UnsupportedArch(
                    "INT8 quantization not yet supported".to_string(),
                ))
            },
        }
    }

    /// Forward pass through a single transformer layer.
    pub fn layer_forward(
        &mut self,
        hidden: &mut GpuTensor,
        layer: &LayerWeights,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<(), InferenceError> {
        // 1. Input norm
        self.rmsnorm.forward(
            hidden,
            &layer.input_norm.weight,
            &mut self.hidden_buffer,
            self.config.rms_norm_eps,
        )?;

        // 2. Self-attention
        // Project Q, K, V
        let hidden_view = self.hidden_buffer.slice_dim0(0, seq_len)?;

        // Q projection: [seq_len, hidden_size] @ [hidden_size, num_heads * head_dim]
        let mut q_view = self.q_buffer.slice_dim0(0, seq_len)?;
        tracing::debug!(
            "Q projection: hidden_view={:?}, q_proj={:?}, q_view={:?}",
            hidden_view.shape(),
            layer.q_proj.data.shape(),
            q_view.shape()
        );
        self.linear(&hidden_view, &layer.q_proj, &mut q_view)?;

        // K projection: [seq_len, hidden_size] @ [hidden_size, num_kv_heads * head_dim]
        let mut k_view = self.k_buffer.slice_dim0(0, seq_len)?;
        tracing::debug!(
            "K projection: hidden_view={:?}, k_proj={:?}, k_view={:?}",
            hidden_view.shape(),
            layer.k_proj.data.shape(),
            k_view.shape()
        );
        self.linear(&hidden_view, &layer.k_proj, &mut k_view)?;

        // V projection: [seq_len, hidden_size] @ [hidden_size, num_kv_heads * head_dim]
        let mut v_view = self.v_buffer.slice_dim0(0, seq_len)?;
        tracing::debug!(
            "V projection: hidden_view={:?}, v_proj={:?}, v_view={:?}",
            hidden_view.shape(),
            layer.v_proj.data.shape(),
            v_view.shape()
        );
        self.linear(&hidden_view, &layer.v_proj, &mut v_view)?;

        // Reshape Q, K, V to [seq_len, heads, head_dim]
        let q_reshaped = q_view.reshape(vec![
            seq_len,
            self.config.num_attention_heads,
            self.config.head_dim,
        ])?;
        let k_reshaped = k_view.reshape(vec![
            seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
        ])?;
        let v_reshaped = v_view.reshape(vec![
            seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
        ])?;

        // Apply RoPE to Q and K
        // Create position tensor [0, 1, 2, ..., seq_len-1] + start_pos
        // Pad to full buffer size for copy_from_host
        let max_seq = self.positions.shape()[0];
        let mut positions_bytes = vec![0u8; max_seq * 4];
        for i in 0..seq_len {
            let pos = (start_pos + i) as i32;
            positions_bytes[i * 4..(i + 1) * 4].copy_from_slice(&pos.to_le_bytes());
        }
        self.positions.copy_from_host(&positions_bytes)?;
        let pos_view = self.positions.slice_dim0(0, seq_len)?;

        // RoPE scaling based on config
        let scaling_factor = match &self.config.rope_scaling {
            None => 1.0,
            Some(scaling) => {
                use super::arch::RopeScaling;
                match scaling {
                    RopeScaling::Linear(factor) => 1.0 / factor,
                    RopeScaling::Dynamic(factor) => 1.0 / factor,
                    RopeScaling::Yarn { factor, .. } => 1.0 / factor,
                }
            },
        };

        let mut q_rope = q_reshaped;
        let mut k_rope = k_reshaped;

        self.rope.forward(
            &mut q_rope,
            &pos_view,
            self.config.rope_theta,
            scaling_factor,
        )?;
        self.rope.forward(
            &mut k_rope,
            &pos_view,
            self.config.rope_theta,
            scaling_factor,
        )?;

        // Update KV cache for this layer
        // Note: advance() is called after all layers in forward()
        self.kv_cache.update(layer.index, &k_rope, &v_reshaped)?;

        // Get cached K, V for attention (all positions up to current)
        let (k_cached, v_cached) = self.kv_cache.get_kv(layer.index)?;

        // Flash Attention - needs 4D tensors [batch, heads, seq, head_dim]
        // Reshape Q from [seq, heads, head_dim] to [1, heads, seq, head_dim]
        let q_4d = q_rope.reshape(vec![
            1,
            self.config.num_attention_heads,
            seq_len,
            self.config.head_dim,
        ])?;

        // Output buffer
        let attn_out = self.attn_out_buffer.slice_dim0(0, seq_len)?;
        let mut attn_out_4d = attn_out.reshape(vec![
            1,
            self.config.num_attention_heads,
            seq_len,
            self.config.head_dim,
        ])?;

        tracing::debug!(
            "Flash Attention: q_4d={:?}, k_cached={:?}, v_cached={:?}, out={:?}",
            q_4d.shape(),
            k_cached.shape(),
            v_cached.shape(),
            attn_out_4d.shape()
        );
        self.attention.forward(
            &q_4d,
            &k_cached,
            &v_cached,
            &mut attn_out_4d,
            true, // causal
        )?;

        // Reshape attention output back to [seq_len, hidden_size]
        let attn_out_flat = attn_out.reshape(vec![
            seq_len,
            self.config.num_attention_heads * self.config.head_dim,
        ])?;

        // Output projection: [seq_len, num_heads * head_dim] @ [num_heads * head_dim, hidden_size]
        let mut attn_proj = self.attn_buffer.slice_dim0(0, seq_len)?;
        self.linear(&attn_out_flat, &layer.o_proj, &mut attn_proj)?;

        // 3. Residual add: hidden = hidden + attn_proj
        self.add_inplace(hidden, &attn_proj, seq_len)?;

        // 4. Post-attention norm
        self.rmsnorm.forward(
            hidden,
            &layer.post_attn_norm.weight,
            &mut self.hidden_buffer,
            self.config.rms_norm_eps,
        )?;

        // 5. MLP
        let hidden_view = self.hidden_buffer.slice_dim0(0, seq_len)?;

        // Gate projection: [seq_len, hidden_size] @ [hidden_size, intermediate_size]
        let mut gate_out = self.gate_buffer.slice_dim0(0, seq_len)?;
        self.linear(&hidden_view, &layer.gate_proj, &mut gate_out)?;

        // Up projection: [seq_len, hidden_size] @ [hidden_size, intermediate_size]
        let mut up_out = self.up_buffer.slice_dim0(0, seq_len)?;
        self.linear(&hidden_view, &layer.up_proj, &mut up_out)?;

        // Fused SiLU(gate) * up
        let mut mlp_hidden = self.mlp_buffer.slice_dim0(0, seq_len)?;
        match self.config.hidden_act {
            Activation::SiLU => {
                self.activation
                    .silu_mul(&gate_out, &up_out, &mut mlp_hidden)?;
            },
            Activation::GELU => {
                // GELU doesn't have a fused version, do separately
                self.activation
                    .forward(&gate_out, &mut mlp_hidden, ActivationType::GELUFast)?;
                self.mul_inplace(&mut mlp_hidden, &up_out, seq_len)?;
            },
            Activation::GELUApprox => {
                self.activation
                    .forward(&gate_out, &mut mlp_hidden, ActivationType::GELUTanh)?;
                self.mul_inplace(&mut mlp_hidden, &up_out, seq_len)?;
            },
            Activation::ReLU | Activation::ReLU2 => {
                // ReLU and ReLU2 both use ReLU kernel (ReLU2 would need squaring, not yet implemented)
                self.activation
                    .forward(&gate_out, &mut mlp_hidden, ActivationType::ReLU)?;
                self.mul_inplace(&mut mlp_hidden, &up_out, seq_len)?;
            },
        }

        // Down projection: [seq_len, intermediate_size] @ [intermediate_size, hidden_size]
        let mut down_out = self.attn_buffer.slice_dim0(0, seq_len)?;
        self.linear(&mlp_hidden, &layer.down_proj, &mut down_out)?;

        // 6. Residual add: hidden = hidden + down_out
        self.add_inplace(hidden, &down_out, seq_len)?;

        Ok(())
    }

    /// Full forward pass through all layers.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs to process
    /// * `weights` - Model weights
    /// * `start_pos` - Starting position (for incremental generation)
    ///
    /// # Returns
    /// Logits tensor of shape [seq_len, vocab_size]
    pub fn forward(
        &mut self,
        input_ids: &[u32],
        weights: &WeightStore,
        start_pos: usize,
    ) -> Result<GpuTensor, InferenceError> {
        let seq_len = input_ids.len();

        // 1. Token embedding lookup
        let mut hidden = self.embed_tokens(&weights.embed_tokens, input_ids)?;

        // 2. Forward through all layers
        for layer in &weights.layers {
            self.layer_forward(&mut hidden, layer, seq_len, start_pos)?;
        }

        // Advance KV cache position after all layers updated
        self.kv_cache.advance(seq_len);

        // 3. Final norm + 4. LM head projection (fused for reduced memory bandwidth)
        // Use pre-allocated buffer slice to avoid allocation
        let mut logits = self.logits_buffer.slice_dim0(0, seq_len)?;

        if let Some(ref lm_head) = weights.lm_head {
            // Separate LM head: fused RMSNorm + GEMM with B transposed
            // lm_head is stored in PyTorch format [vocab_size, hidden_dim], same as embed_tokens
            // We compute: hidden @ lm_head^T to get [seq, vocab_size]
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &weights.final_norm.weight,
                lm_head,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else if weights.config.tie_word_embeddings {
            // Tied embeddings: fused RMSNorm + GEMM with B transposed
            // embed_tokens is [vocab_size, hidden_dim], we compute hidden @ embed_tokens^T
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &weights.final_norm.weight,
                &weights.embed_tokens,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else {
            return Err(InferenceError::ModelLoad(
                "No LM head and embeddings not tied".to_string(),
            ));
        }

        self.current_pos = start_pos + seq_len;

        // Store logits for get_logits() - ZERO-COPY via Arc sharing
        // We create a view that shares the same underlying buffer
        let logits_for_storage = logits.reshape(logits.shape().to_vec())?;
        self.last_logits = Some(logits_for_storage);

        // Return original logits (also shares same buffer via Arc)
        Ok(logits)
    }

    /// Prefill phase: process the entire prompt.
    pub fn prefill(
        &mut self,
        input_ids: &[u32],
        weights: &WeightStore,
    ) -> Result<GpuTensor, InferenceError> {
        self.kv_cache.reset();
        self.current_pos = 0;
        self.forward(input_ids, weights, 0)
    }

    /// Decode phase: process a single new token.
    pub fn decode(
        &mut self,
        token_id: u32,
        weights: &WeightStore,
    ) -> Result<GpuTensor, InferenceError> {
        self.forward(&[token_id], weights, self.current_pos)
    }

    // ========================================================================
    // Lazy Loading Methods
    //
    // These methods work with LazyWeightStore, loading layers on-demand
    // for memory-efficient inference of large models.
    // ========================================================================

    /// Forward pass with lazy layer loading.
    ///
    /// Loads each layer on-demand, enabling models larger than VRAM.
    pub fn forward_lazy(
        &mut self,
        input_ids: &[u32],
        weights: &mut super::lazy_weight_store::LazyWeightStore,
        start_pos: usize,
    ) -> Result<GpuTensor, InferenceError> {
        let seq_len = input_ids.len();

        // 1. Token embedding lookup
        let mut hidden = self.embed_tokens(&weights.embed_tokens, input_ids)?;

        // 2. Forward through all layers (lazy loading)
        for layer_idx in 0..weights.num_layers() {
            let layer = weights.get_layer(layer_idx)?;
            self.layer_forward(&mut hidden, layer, seq_len, start_pos)?;
        }

        // Advance KV cache position after all layers updated
        self.kv_cache.advance(seq_len);

        // 3. Final norm + 4. LM head projection (fused)
        let mut logits = self.logits_buffer.slice_dim0(0, seq_len)?;

        if let Some(ref lm_head) = weights.lm_head {
            // Separate LM head with B transposed (PyTorch format [vocab_size, hidden_dim])
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &weights.final_norm.weight,
                lm_head,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else if weights.config.tie_word_embeddings {
            // Tied embeddings (same transposed format)
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &weights.final_norm.weight,
                &weights.embed_tokens,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else {
            return Err(InferenceError::ModelLoad(
                "No LM head and embeddings not tied".to_string(),
            ));
        }

        self.current_pos = start_pos + seq_len;

        // Store logits for get_logits()
        let logits_for_storage = logits.reshape(logits.shape().to_vec())?;
        self.last_logits = Some(logits_for_storage);

        Ok(logits)
    }

    /// Prefill phase with lazy layer loading.
    pub fn prefill_lazy(
        &mut self,
        input_ids: &[u32],
        weights: &mut super::lazy_weight_store::LazyWeightStore,
    ) -> Result<GpuTensor, InferenceError> {
        self.kv_cache.reset();
        self.current_pos = 0;
        self.forward_lazy(input_ids, weights, 0)
    }

    /// Decode phase with lazy layer loading.
    pub fn decode_lazy(
        &mut self,
        token_id: u32,
        weights: &mut super::lazy_weight_store::LazyWeightStore,
    ) -> Result<GpuTensor, InferenceError> {
        self.forward_lazy(&[token_id], weights, self.current_pos)
    }

    /// Embed tokens by looking up in embedding table using GPU kernel.
    fn embed_tokens(
        &mut self,
        embed_table: &GpuTensor,
        input_ids: &[u32],
    ) -> Result<GpuTensor, InferenceError> {
        let seq_len = input_ids.len();
        let _hidden_size = self.config.hidden_size;

        // Copy token IDs to GPU
        let token_bytes: Vec<u8> = input_ids
            .iter()
            .flat_map(|&id| (id as i32).to_le_bytes())
            .collect();

        // Create a properly sized token_ids tensor for this batch
        let mut token_ids = self.token_ids_buffer.slice_dim0(0, seq_len)?;
        token_ids.copy_from_host(&token_bytes)?;

        // Use pre-allocated hidden buffer slice to avoid allocation
        let mut hidden = self.hidden_buffer.slice_dim0(0, seq_len)?;

        // Use GPU kernel to gather embeddings
        self.embedding
            .forward(embed_table, &token_ids, &mut hidden)?;

        Ok(hidden)
    }

    /// Element-wise add: a = a + b.
    fn add_inplace(
        &self,
        a: &mut GpuTensor,
        b: &GpuTensor,
        _seq_len: usize,
    ) -> Result<(), InferenceError> {
        // This would ideally be a fused kernel
        // For now, use cuBLAS axpy: a = a + 1.0 * b
        self.cublas.axpy(1.0, b, a)?;
        Ok(())
    }

    /// Element-wise multiply: a = a * b.
    fn mul_inplace(
        &mut self,
        a: &mut GpuTensor,
        b: &GpuTensor,
        _seq_len: usize,
    ) -> Result<(), InferenceError> {
        self.activation.hadamard_inplace(a, b)
    }

    /// Get the KV cache.
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// Get mutable KV cache.
    pub fn kv_cache_mut(&mut self) -> &mut KvCache {
        &mut self.kv_cache
    }

    /// Reset for a new sequence.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.current_pos = 0;
        self.last_logits = None;
    }

    /// Reset KV cache (alias for reset).
    pub fn reset_cache(&mut self) {
        self.reset();
    }

    /// Get current sequence position.
    pub fn current_position(&self) -> usize {
        self.current_pos
    }

    /// Get model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get logits from the last forward pass.
    ///
    /// Returns logits for the last token in the sequence.
    /// Shape: [vocab_size] as F16.
    pub fn get_logits(&self) -> Result<GpuTensor, InferenceError> {
        let logits = self.last_logits.as_ref().ok_or_else(|| {
            InferenceError::Kernel(
                "No logits available - call forward/prefill/decode first".to_string(),
            )
        })?;

        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(InferenceError::Shape {
                expected: "[seq_len, vocab_size]".to_string(),
                got: format!("{:?}", shape),
            });
        }

        let seq_len = shape[0];
        let vocab_size = shape[1];

        // Return logits for the last token only
        // Slice the last row: logits[seq_len-1, :]
        if seq_len == 1 {
            // Already single token, reshape to [vocab_size]
            logits.reshape(vec![vocab_size])
        } else {
            // Get last token's logits
            let last_token_logits = logits.slice_dim0(seq_len - 1, seq_len)?;
            last_token_logits.reshape(vec![vocab_size])
        }
    }

    /// Get the stream manager for async operations.
    pub fn streams(&self) -> &StreamManager {
        &self.streams
    }

    /// Synchronize all streams (wait for pending operations).
    ///
    /// Call this before reading results to ensure all async operations complete.
    pub fn synchronize(&self) -> Result<(), InferenceError> {
        self.streams.synchronize_all()
    }

    /// Synchronize just the compute stream.
    ///
    /// Faster than synchronize_all() when only compute results are needed.
    pub fn synchronize_compute(&self) -> Result<(), InferenceError> {
        self.streams.synchronize_compute()
    }

    // ========================================================================
    // Tiered Storage Methods
    //
    // These methods work with TieredWeightStore, providing efficient
    // inference for models spanning VRAM, RAM, and NVMe tiers.
    // ========================================================================

    /// Forward pass with tiered weight storage.
    ///
    /// Uses the 3-tier memory hierarchy (VRAM ← RAM ← NVMe) for efficient
    /// inference of models larger than VRAM. Includes prefetching for
    /// sequential layer access patterns.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `weights` - TieredWeightStore managing multi-tier weights
    /// * `start_pos` - Starting position in sequence (for KV cache)
    /// * `prefetch_depth` - How many layers ahead to prefetch (0 to disable)
    pub fn forward_tiered(
        &mut self,
        input_ids: &[u32],
        weights: &mut super::tiered::TieredWeightStore,
        start_pos: usize,
        prefetch_depth: usize,
    ) -> Result<GpuTensor, InferenceError> {
        let seq_len = input_ids.len();
        let num_layers = weights.num_layers();

        // 1. Token embedding lookup (shared weights are always in VRAM)
        let shared = weights
            .shared()
            .ok_or_else(|| InferenceError::ModelLoad("Shared weights not loaded".to_string()))?;
        let mut hidden = self.embed_tokens(&shared.embed_tokens, input_ids)?;

        // 2. Forward through all layers with prefetching
        for layer_idx in 0..num_layers {
            // Request prefetch of upcoming layers
            if prefetch_depth > 0 {
                weights.prefetch(layer_idx, prefetch_depth);
            }

            // Get layer (may promote from RAM/NVMe if not in VRAM)
            let layer = weights.get_layer(layer_idx).map_err(|e| {
                InferenceError::ModelLoad(format!("Failed to get layer {}: {}", layer_idx, e))
            })?;

            self.layer_forward(&mut hidden, layer, seq_len, start_pos)?;
        }

        // Advance KV cache position after all layers updated
        self.kv_cache.advance(seq_len);

        // 3. Final norm + 4. LM head projection (fused)
        let mut logits = self.logits_buffer.slice_dim0(0, seq_len)?;

        let shared = weights
            .shared()
            .ok_or_else(|| InferenceError::ModelLoad("Shared weights not loaded".to_string()))?;

        if let Some(ref lm_head) = shared.lm_head {
            // Separate LM head with B transposed
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &shared.final_norm.weight,
                lm_head,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else if self.config.tie_word_embeddings {
            // Tied embeddings
            self.fused_rmsnorm_proj.forward_f16_bt(
                &hidden,
                &shared.final_norm.weight,
                &shared.embed_tokens,
                &mut logits,
                self.config.rms_norm_eps,
            )?;
        } else {
            return Err(InferenceError::ModelLoad(
                "No LM head and embeddings not tied".to_string(),
            ));
        }

        self.current_pos = start_pos + seq_len;

        // Store logits for get_logits()
        let logits_for_storage = logits.reshape(logits.shape().to_vec())?;
        self.last_logits = Some(logits_for_storage);

        Ok(logits)
    }

    /// Prefill phase with tiered storage.
    ///
    /// Processes the entire prompt, resetting the KV cache.
    pub fn prefill_tiered(
        &mut self,
        input_ids: &[u32],
        weights: &mut super::tiered::TieredWeightStore,
        prefetch_depth: usize,
    ) -> Result<GpuTensor, InferenceError> {
        self.kv_cache.reset();
        self.current_pos = 0;
        self.forward_tiered(input_ids, weights, 0, prefetch_depth)
    }

    /// Decode phase with tiered storage.
    ///
    /// Processes a single new token for autoregressive generation.
    pub fn decode_tiered(
        &mut self,
        token_id: u32,
        weights: &mut super::tiered::TieredWeightStore,
        prefetch_depth: usize,
    ) -> Result<GpuTensor, InferenceError> {
        self.forward_tiered(&[token_id], weights, self.current_pos, prefetch_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_inference::arch::ModelArch;

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig {
            arch: ModelArch::Llama,
            vocab_size: 32000,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 30,
            num_attention_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: true,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        };

        assert_eq!(config.hidden_size, 576);
        assert_eq!(config.num_layers, 30);
        assert!(config.tie_word_embeddings);
    }

    // Integration tests for ComputeEngine are in tests.rs
    // They require GPU hardware and a model checkpoint
}
