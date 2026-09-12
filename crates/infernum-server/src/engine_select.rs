//! Choosing an inference backend for a model source.
//!
//! The server has historically built an [`abaddon::Engine`] (Candle) for every
//! model it was asked to load. Candle cannot read a GGUF: [`abaddon`]'s loader
//! resolves a `.gguf` to `ModelFiles { config: <the .gguf itself>, .. }` on the
//! note that "GGUF has embedded config", and `ModelConfig::from_file` then does
//! a `read_to_string` on it. Serving a GGUF therefore failed with
//! `Failed to read config: stream did not contain valid UTF-8`, which names the
//! symptom and not the cause.
//!
//! llama.cpp is the backend that reads GGUF, and `abaddon::LlamaCppEngine`
//! already wraps it — but until now only the `generate` CLI command could reach
//! it. This module is the seam that lets the HTTP server reach it too.
//!
//! The selection is deliberately narrow: **only a `.gguf` file routes to
//! llama.cpp**, and everything else keeps the Candle path it has always had.
//! `BackendType::detect_from_path` is *not* used here on purpose — it sends a
//! safetensors directory to llama.cpp when the `llama-cpp` feature is on, and a
//! safetensors directory is exactly what the embedding models are. Routing
//! `nomic-embed-text-v1.5` to llama.cpp would break the embedding server that
//! Lares' RAG runs on, silently and at load time.

/// The inference backend a model source needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// Candle, via [`abaddon::Engine`]. Reads safetensors directories,
    /// HoloTensor HCT directories and HuggingFace repo ids.
    Candle,
    /// llama.cpp, via `abaddon::LlamaCppEngine`. Reads GGUF files.
    LlamaCpp,
}

/// Returns the backend required to load `model_source`.
///
/// `model_source` is what reaches the server: a filesystem path, a HuggingFace
/// repo id, or a `holo://` URL.
pub fn engine_kind_for(model_source: &str) -> EngineKind {
    // A `holo://` URL is a HoloTensor model however its path is spelled.
    if model_source.starts_with("holo://") {
        return EngineKind::Candle;
    }

    // Strip any query string before looking at the extension.
    let path = model_source
        .split_once('?')
        .map_or(model_source, |(path, _query)| path);

    if std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return EngineKind::LlamaCpp;
    }

    EngineKind::Candle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_file_needs_llama_cpp() {
        assert_eq!(
            engine_kind_for("/home/lilith/models/qwen2.5-14b-instruct-q8_0.gguf"),
            EngineKind::LlamaCpp
        );
    }

    #[test]
    fn gguf_extension_is_matched_case_insensitively() {
        assert_eq!(engine_kind_for("/models/Model.GGUF"), EngineKind::LlamaCpp);
    }

    /// The embedding model Lares' RAG runs on is a safetensors directory.
    /// It must keep the Candle path; `BackendType::detect_from_path` would
    /// send it to llama.cpp and break it.
    #[test]
    fn safetensors_directory_stays_on_candle() {
        assert_eq!(
            engine_kind_for("/home/lilith/.cache/infernum/models/nomic-embed-text-v1.5"),
            EngineKind::Candle
        );
    }

    #[test]
    fn holotensor_url_stays_on_candle() {
        assert_eq!(
            engine_kind_for("holo:///models/llama-70b.hct?min=0.7&target=0.95"),
            EngineKind::Candle
        );
    }

    /// A `holo://` URL whose path happens to end in `.gguf` is still HoloTensor.
    #[test]
    fn holotensor_url_wins_over_a_gguf_looking_path() {
        assert_eq!(
            engine_kind_for("holo:///models/odd.gguf?min=0.7"),
            EngineKind::Candle
        );
    }

    #[test]
    fn huggingface_repo_id_stays_on_candle() {
        assert_eq!(
            engine_kind_for("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            EngineKind::Candle
        );
    }
}
