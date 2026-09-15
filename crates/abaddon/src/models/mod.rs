//! Model architecture implementations.
//!
//! Supports multiple architectures:
//! - Llama (Llama 2, Llama 3, Llama 3.1, Llama 3.2, CodeLlama)
//! - LazyLlama (layer-lazy variant for 405B+ models)
//! - Qwen2 (Qwen2, Qwen2.5, Qwen2.5-Coder)
//! - Bert (BERT, NomicBERT, Jina BERT — embedding models)

pub mod bert;
pub mod lazy_llama;
pub mod lazy_qwen2;
pub mod llama;
pub mod nomic_bert;
#[allow(dead_code)]
mod quantized_llama;
pub mod qwen2;

pub use bert::{Bert, BertConfig};
pub use lazy_llama::{LazyLlama, LazyLoadError, LazyStats};
pub use lazy_qwen2::LazyQwen2;
pub use llama::{Llama, LlamaConfig};
pub use nomic_bert::{NomicBert, NomicBertConfig};
pub use qwen2::{CacheType, Qwen2, Qwen2Config};

use candle_core::{Result as CandleResult, Tensor};

/// Loaded model variant - wraps different model implementations.
///
/// This allows the engine to work with multiple model types through a unified interface.
/// Named `ModelKind` to avoid conflict with `infernum_core::ModelArchitecture`.
pub enum ModelKind {
    /// Llama-family models (Llama, CodeLlama, Mistral, etc.)
    Llama(Llama),
    /// Lazy Llama for 405B+ models (layer-by-layer loading)
    LazyLlama(LazyLlama),
    /// Qwen2-family models (Qwen2, Qwen2.5, Qwen2.5-Coder)
    Qwen2(Qwen2),
    /// Lazy Qwen2 for 14B+ models (layer-by-layer loading)
    LazyQwen2(LazyQwen2),
    /// Standard BERT embedding models (BERT, Jina BERT)
    Bert(Bert),
    /// NomicBERT embedding models (fused QKV, rotary, SwiGLU)
    NomicBert(NomicBert),
}

impl ModelKind {
    /// Forward pass for the model.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.forward(input_ids, start_pos),
            Self::LazyLlama(model) => model
                .forward(input_ids, start_pos)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
            Self::Qwen2(model) => model.forward(input_ids, start_pos),
            Self::LazyQwen2(model) => model
                .forward(input_ids, start_pos)
                .map_err(|e| candle_core::Error::Msg(e.to_string())),
            Self::Bert(_) | Self::NomicBert(_) => Err(candle_core::Error::Msg(
                "BERT is an embedding-only model and does not support causal generation"
                    .to_string(),
            )),
        }
    }

    /// Clears the KV cache.
    pub fn clear_cache(&mut self) {
        match self {
            Self::Llama(model) => model.clear_cache(),
            Self::LazyLlama(model) => model.clear_cache(),
            Self::Qwen2(model) => model.clear_cache(),
            Self::LazyQwen2(model) => model.clear_cache(),
            Self::Bert(model) => model.clear_cache(),
            Self::NomicBert(model) => model.clear_cache(),
        }
    }

    /// Forward pass for embedding extraction.
    pub fn forward_embedding(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.forward_embedding(input_ids),
            Self::LazyLlama(_model) => Err(candle_core::Error::Msg(
                "Embedding extraction not supported for LazyLlama".to_string(),
            )),
            Self::Qwen2(model) => model.forward_embedding(input_ids),
            Self::LazyQwen2(_model) => Err(candle_core::Error::Msg(
                "Embedding extraction not supported for LazyQwen2".to_string(),
            )),
            Self::Bert(model) => model.forward_embedding(input_ids),
            Self::NomicBert(model) => model.forward_embedding(input_ids),
        }
    }

    /// Extract embeddings by mean pooling.
    pub fn extract_embeddings(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Llama(model) => model.extract_embeddings(input_ids),
            Self::LazyLlama(_model) => Err(candle_core::Error::Msg(
                "Embedding extraction not supported for LazyLlama".to_string(),
            )),
            Self::Qwen2(model) => model.extract_embeddings(input_ids),
            Self::LazyQwen2(_model) => Err(candle_core::Error::Msg(
                "Embedding extraction not supported for LazyQwen2".to_string(),
            )),
            Self::Bert(model) => model.extract_embeddings(input_ids),
            Self::NomicBert(model) => model.extract_embeddings(input_ids),
        }
    }
}

/// L2-normalizes an embedding in place, so it ends with unit magnitude.
///
/// `ModelKind::extract_embeddings` returns *mean-pooled hidden states*, whose
/// scale varies by architecture and by input length. Normalization is a
/// serving concern rather than a model one, so it happens once, here, on the
/// way out of the embedding endpoint — not in each model's
/// `extract_embeddings`, where it previously existed for NomicBERT and for
/// none of the other three.
///
/// Callers embedding a Matryoshka model (`nomic-embed-text-v1.5`) must
/// truncate *before* calling this: slicing a unit vector leaves it shorter
/// than unit length, so the order matters.
///
/// A vector whose norm is not a normal float — zero, subnormal, infinite or
/// NaN — has no direction worth preserving and is left untouched rather than
/// divided, so a degenerate embedding stays degenerate instead of becoming
/// NaNs.
pub fn l2_normalize(embedding: &mut [f32]) {
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm.is_normal() {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }
}

/// Applies the embedding endpoint's post-processing to one raw pooled vector:
/// optional Matryoshka truncation to `dimensions`, then L2 normalization.
///
/// The order is the point. `nomic-embed-text-v1.5` is a Matryoshka model, so a
/// leading slice of its embedding is still a usable embedding — but slicing a
/// unit vector leaves it shorter than unit length. Normalizing first and
/// truncating second would hand the caller a vector of arbitrary magnitude;
/// truncating first and normalizing second is what makes the shortened
/// embedding usable.
///
/// `dimensions` larger than the embedding is ignored rather than padded.
pub fn finalize_embedding(mut embedding: Vec<f32>, dimensions: Option<usize>) -> Vec<f32> {
    if let Some(dims) = dimensions {
        if dims < embedding.len() {
            embedding.truncate(dims);
        }
    }
    l2_normalize(&mut embedding);
    embedding
}

/// Supported model architectures for detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureType {
    /// Llama architecture family
    Llama,
    /// Qwen2 architecture family
    Qwen2,
    /// Standard BERT embedding models (BERT, Jina BERT)
    Bert,
    /// NomicBERT embedding models (custom architecture)
    NomicBert,
    /// Unknown/unsupported architecture
    Unknown,
}

impl ArchitectureType {
    /// Detects the architecture type from a model type string or architecture name.
    ///
    /// # Arguments
    /// * `model_type` - The model_type from config.json (e.g., "llama", "qwen2")
    /// * `architectures` - Optional list of architecture names (e.g., ["LlamaForCausalLM"])
    pub fn detect(model_type: Option<&str>, architectures: Option<&[String]>) -> Self {
        // First check model_type (order matters: nomic before generic bert)
        if let Some(mt) = model_type {
            let mt_lower = mt.to_lowercase();
            if mt_lower.contains("llama") || mt_lower.contains("mistral") {
                return Self::Llama;
            }
            if mt_lower.contains("qwen2") || mt_lower == "qwen2" {
                return Self::Qwen2;
            }
            if mt_lower.contains("nomic") {
                return Self::NomicBert;
            }
            if mt_lower.contains("bert") || mt_lower.contains("jina") {
                return Self::Bert;
            }
        }

        // Then check architectures list
        if let Some(archs) = architectures {
            for arch in archs {
                let arch_lower = arch.to_lowercase();
                if arch_lower.contains("llama") || arch_lower.contains("mistral") {
                    return Self::Llama;
                }
                if arch_lower.contains("qwen2") {
                    return Self::Qwen2;
                }
                if arch_lower.contains("nomic") {
                    return Self::NomicBert;
                }
                if arch_lower.contains("bert") {
                    return Self::Bert;
                }
            }
        }

        Self::Unknown
    }

    /// Returns a human-readable name for this architecture.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama",
            Self::Qwen2 => "Qwen2",
            Self::Bert => "Bert",
            Self::NomicBert => "NomicBert",
            Self::Unknown => "Unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{finalize_embedding, l2_normalize, ArchitectureType};

    // -----------------------------------------------------------------------
    // l2_normalize
    // -----------------------------------------------------------------------

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// The guarantee the embedding endpoint makes to its callers.
    #[test]
    fn l2_normalize_produces_unit_norm() {
        for mut v in [
            vec![3.0f32, 4.0],                // norm 5
            vec![1.0; 16],                    // norm 4
            vec![-0.5, 0.25, -0.125, 0.0625], // mixed signs
            vec![1e-6, -2e-6, 3e-6],          // small but normal
            vec![1e6, 2e6, -3e6],             // large
            vec![7.0],                        // single element
        ] {
            let before = v.clone();
            l2_normalize(&mut v);
            let n = norm(&v);
            assert!(
                (n - 1.0).abs() < 1e-5,
                "{before:?} normalized to magnitude {n}, expected 1.0",
            );
        }
    }

    /// Normalization only rescales — it must not rotate the vector. Checked
    /// by requiring each component to keep its proportion of the whole.
    #[test]
    fn l2_normalize_preserves_direction() {
        let original = vec![3.0f32, -4.0, 12.0];
        let mut v = original.clone();
        l2_normalize(&mut v);

        let scale = norm(&original);
        for (i, (got, orig)) in v.iter().zip(&original).enumerate() {
            let want = orig / scale;
            assert!(
                (got - want).abs() < 1e-6,
                "component {i}: {got} vs expected {want}",
            );
        }
    }

    /// An already-normalized vector is a fixed point, so normalizing twice is
    /// the same as normalizing once. This is what makes the endpoint safe to
    /// call on output from a model that happens to normalize internally.
    #[test]
    fn l2_normalize_is_idempotent() {
        let mut once = vec![0.3f32, -1.7, 2.2, 0.0, -0.4];
        l2_normalize(&mut once);
        let mut twice = once.clone();
        l2_normalize(&mut twice);

        for (i, (a, b)) in once.iter().zip(&twice).enumerate() {
            assert!((a - b).abs() < 1e-7, "component {i} moved: {a} -> {b}");
        }
    }

    /// A vector with no direction to preserve is left exactly as it is, rather
    /// than divided by zero. Dividing here is how a degenerate embedding
    /// becomes a vector of NaNs that poisons every downstream similarity.
    #[test]
    fn l2_normalize_leaves_degenerate_vectors_alone() {
        let mut zeros = vec![0.0f32; 8];
        l2_normalize(&mut zeros);
        assert!(
            zeros.iter().all(|v| *v == 0.0),
            "zero vector should stay zero, got {zeros:?}",
        );

        let mut empty: Vec<f32> = Vec::new();
        l2_normalize(&mut empty);
        assert!(empty.is_empty(), "empty input should stay empty");

        // A NaN anywhere makes the norm NaN; leave it visible rather than
        // spreading NaN across every component.
        let mut nan = vec![1.0f32, f32::NAN, 2.0];
        l2_normalize(&mut nan);
        assert!(
            nan[0] == 1.0 && nan[2] == 2.0,
            "finite components were rescaled by a NaN norm"
        );
    }

    /// Truncating a unit vector leaves it shorter than unit length, which is
    /// exactly why the endpoint normalizes *after* applying `dimensions`
    /// rather than before. Matryoshka models like `nomic-embed-text-v1.5` are
    /// trained so a leading slice is still usable — but only once renormalized.
    #[test]
    fn truncating_before_normalizing_is_what_restores_unit_norm() {
        let mut full = vec![0.6f32, 0.8, 0.5, -0.5, 0.25, 0.1];
        l2_normalize(&mut full);
        assert!((norm(&full) - 1.0).abs() < 1e-6);

        // Slicing the already-normalized vector loses magnitude...
        let sliced = full[..3].to_vec();
        assert!(
            norm(&sliced) < 0.999,
            "a truncated unit vector should be short, got {}",
            norm(&sliced),
        );

        // ...which normalizing afterwards restores.
        let mut fixed = sliced.clone();
        l2_normalize(&mut fixed);
        assert!(
            (norm(&fixed) - 1.0).abs() < 1e-6,
            "normalizing after truncation should give unit norm, got {}",
            norm(&fixed),
        );
    }

    // -----------------------------------------------------------------------
    // finalize_embedding — the endpoint's post-processing
    // -----------------------------------------------------------------------

    /// Every embedding leaving the endpoint has unit magnitude, whether or not
    /// the caller asked to truncate. This is the guarantee LARES-464 was filed
    /// about: before the fix, NomicBERT normalized internally and Bert, Llama
    /// and Qwen2 did not, so the scale depended on which model was loaded.
    #[test]
    fn finalize_embedding_always_returns_unit_norm() {
        let raw = vec![3.0f32, -4.0, 12.0, 0.5, -0.25, 8.0, 1.0, -1.0];

        for dims in [None, Some(2), Some(4), Some(7), Some(8), Some(64)] {
            let out = finalize_embedding(raw.clone(), dims);
            let n = norm(&out);
            assert!(
                (n - 1.0).abs() < 1e-5,
                "dimensions={dims:?} gave magnitude {n}, expected 1.0",
            );
        }
    }

    /// The ordering bug, pinned directly: truncating a vector that was already
    /// normalized leaves it short. `finalize_embedding` must truncate first so
    /// the normalize has the last word.
    #[test]
    fn finalize_embedding_normalizes_after_truncating_not_before() {
        let raw = vec![9.0f32, 0.5, 0.25, 0.125];

        let got = finalize_embedding(raw.clone(), Some(2));

        // What the wrong order would produce: normalize the full vector, then
        // slice it. The first component dominates, so the slice keeps most of
        // the magnitude and the error is small but real — which is exactly why
        // this went unnoticed.
        let mut wrong = raw.clone();
        l2_normalize(&mut wrong);
        let wrong = wrong[..2].to_vec();

        assert!(
            (norm(&got) - 1.0).abs() < 1e-6,
            "truncate-then-normalize should be unit norm, got {}",
            norm(&got),
        );
        assert!(
            norm(&wrong) < 0.9999,
            "normalize-then-truncate should be short of unit norm, got {}",
            norm(&wrong),
        );
        assert!(
            (norm(&got) - norm(&wrong)).abs() > 1e-5,
            "the two orderings are indistinguishable on this fixture; it proves nothing",
        );
    }

    /// Output width is exactly what was asked for, across the whole range —
    /// including the boundaries either side of the embedding's own width,
    /// where an off-by-one in the truncation guard would hide.
    #[test]
    fn finalize_embedding_truncates_to_the_requested_width() {
        let raw = vec![1.0f32; 16];

        assert_eq!(finalize_embedding(raw.clone(), None).len(), 16);

        // Every width from 1 up to the full embedding comes back exactly.
        for dims in 1..=16usize {
            assert_eq!(
                finalize_embedding(raw.clone(), Some(dims)).len(),
                dims,
                "dimensions={dims} should give a {dims}-wide embedding",
            );
        }

        // A request wider than the embedding is ignored, not padded.
        for dims in [17usize, 32, 4096] {
            assert_eq!(
                finalize_embedding(raw.clone(), Some(dims)).len(),
                16,
                "dimensions={dims} exceeds the embedding and should be ignored",
            );
        }
    }

    /// Truncation keeps the leading components, which is what makes a
    /// Matryoshka slice meaningful — taking the tail, or reordering, would
    /// produce a vector that no longer matches the model's training.
    #[test]
    fn finalize_embedding_keeps_the_leading_components() {
        let raw = vec![4.0f32, 3.0, 100.0, -100.0];
        let out = finalize_embedding(raw, Some(2));

        // [4, 3] normalized is [0.8, 0.6]; had it kept the tail it would be
        // dominated by the ±100 pair instead.
        assert!((out[0] - 0.8).abs() < 1e-6, "expected 0.8, got {}", out[0]);
        assert!((out[1] - 0.6).abs() < 1e-6, "expected 0.6, got {}", out[1]);
    }

    /// A degenerate embedding stays finite rather than becoming NaNs, at every
    /// width. Zero-length output is the existing behaviour for
    /// `dimensions: 0` — documented here rather than endorsed.
    #[test]
    fn finalize_embedding_handles_degenerate_input() {
        let zeros = vec![0.0f32; 8];
        for dims in [None, Some(4), Some(8)] {
            let out = finalize_embedding(zeros.clone(), dims);
            assert!(
                out.iter().all(|v| *v == 0.0),
                "zero embedding should stay zero at dimensions={dims:?}, got {out:?}",
            );
        }

        assert!(finalize_embedding(zeros, Some(0)).is_empty());
    }

    #[test]
    fn detect_bert_from_model_type() {
        assert_eq!(
            ArchitectureType::detect(Some("bert"), None),
            ArchitectureType::Bert
        );
        assert_eq!(
            ArchitectureType::detect(Some("jina_bert"), None),
            ArchitectureType::Bert
        );
    }

    #[test]
    fn detect_nomic_bert_from_model_type() {
        assert_eq!(
            ArchitectureType::detect(Some("nomic_bert"), None),
            ArchitectureType::NomicBert
        );
    }

    #[test]
    fn detect_bert_from_architectures() {
        let archs = vec!["BertForMaskedLM".to_string()];
        assert_eq!(
            ArchitectureType::detect(None, Some(&archs)),
            ArchitectureType::Bert
        );
    }

    /// `"nomic_bert"` contains both `"nomic"` and `"bert"`, so the order of
    /// the branches in `detect` is load-bearing. If the generic bert check
    /// ever moves ahead of the nomic one, every NomicBERT checkpoint quietly
    /// loads as a Jina/ALiBi BERT against NomicBERT weights — no error, just
    /// wrong embeddings. Same trap for `"NomicBertModel"` in `architectures`.
    #[test]
    fn nomic_wins_over_generic_bert_in_detection() {
        for mt in ["nomic_bert", "NomicBertModel", "nomic-bert-2048"] {
            assert_eq!(
                ArchitectureType::detect(Some(mt), None),
                ArchitectureType::NomicBert,
                "model_type {mt:?} must not fall through to generic Bert",
            );
        }

        let archs = vec!["NomicBertModel".to_string()];
        assert_eq!(
            ArchitectureType::detect(None, Some(&archs)),
            ArchitectureType::NomicBert,
            "architectures entry must not fall through to generic Bert",
        );
    }

    /// `model_type` is consulted before `architectures`, and the real
    /// `nomic-embed-text-v1.5` config sets both.
    #[test]
    fn model_type_takes_precedence_over_architectures() {
        let archs = vec!["BertForMaskedLM".to_string()];
        assert_eq!(
            ArchitectureType::detect(Some("nomic_bert"), Some(&archs)),
            ArchitectureType::NomicBert,
        );

        let archs = vec!["NomicBertModel".to_string()];
        assert_eq!(
            ArchitectureType::detect(Some("nomic_bert"), Some(&archs)),
            ArchitectureType::NomicBert,
            "the shape the published checkpoint actually ships",
        );
    }

    #[test]
    fn unknown_architectures_are_reported_as_unknown() {
        assert_eq!(
            ArchitectureType::detect(None, None),
            ArchitectureType::Unknown
        );
        assert_eq!(
            ArchitectureType::detect(Some("mamba"), None),
            ArchitectureType::Unknown,
        );
        assert_eq!(ArchitectureType::NomicBert.name(), "NomicBert");
        assert_eq!(ArchitectureType::Bert.name(), "Bert");
    }

    #[test]
    fn detect_nomic_bert_from_architectures() {
        let archs = vec!["NomicBertModel".to_string()];
        assert_eq!(
            ArchitectureType::detect(None, Some(&archs)),
            ArchitectureType::NomicBert
        );
    }

    #[test]
    fn detect_llama_still_works() {
        assert_eq!(
            ArchitectureType::detect(Some("llama"), None),
            ArchitectureType::Llama
        );
        assert_eq!(
            ArchitectureType::detect(Some("mistral"), None),
            ArchitectureType::Llama
        );
    }

    #[test]
    fn detect_qwen2_still_works() {
        assert_eq!(
            ArchitectureType::detect(Some("qwen2"), None),
            ArchitectureType::Qwen2
        );
    }

    #[test]
    fn detect_unknown_fallback() {
        assert_eq!(
            ArchitectureType::detect(Some("gpt-j"), None),
            ArchitectureType::Unknown
        );
        assert_eq!(
            ArchitectureType::detect(None, None),
            ArchitectureType::Unknown
        );
    }
}
