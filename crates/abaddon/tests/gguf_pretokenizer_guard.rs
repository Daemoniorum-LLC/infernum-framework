//! Integration tests pinning the pre-tokenizer guard to its **call site**.
//!
//! `gguf_pretokenizer`'s own unit tests prove the check is correct. These
//! prove it is actually *wired into* `LlamaCppEngine::load`, so that deleting
//! the call — or moving it after the model load — fails a test rather than
//! silently restoring the defect.
//!
//! That distinction is the whole point of issue #56: the original bug was not
//! a check that answered wrongly, it was the absence of a check that anyone
//! could see was absent. A guard whose call site is untested can regress the
//! same way.
//!
//! See `crates/abaddon/src/gguf_pretokenizer.rs` and
//! https://github.com/Daemoniorum-LLC/infernum-framework/issues/56

use std::io::Write;

use abaddon::gguf_pretokenizer::{ensure_pre_tokenizer_supported, PRE_TOKENIZER_KEY};

/// Builds a minimal GGUF v3 header carrying the given string metadata pairs.
fn gguf_with(pairs: &[(&str, &str)]) -> Vec<u8> {
    fn gstr(v: &str) -> Vec<u8> {
        let mut out = (v.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(v.as_bytes());
        out
    }

    let mut kvs = Vec::new();
    for (k, v) in pairs {
        kvs.extend_from_slice(&gstr(k));
        kvs.extend_from_slice(&8u32.to_le_bytes()); // STRING
        kvs.extend_from_slice(&gstr(v));
    }

    let mut out = 0x4655_4747u32.to_le_bytes().to_vec(); // "GGUF"
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    out.extend_from_slice(&kvs);
    out
}

fn write_gguf(pairs: &[(&str, &str)]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("tempfile");
    f.write_all(&gguf_with(pairs)).expect("write");
    f.flush().expect("flush");
    f
}

/// The guard is reachable through the crate's public surface.
#[test]
fn guard_is_public_api() {
    let f = write_gguf(&[
        ("general.architecture", "qwen2"),
        (PRE_TOKENIZER_KEY, "qwen2"),
    ]);
    assert!(ensure_pre_tokenizer_supported(f.path()).is_err());
}

/// `LlamaCppEngine::load` refuses a GGUF carrying the key.
///
/// This is the wiring test: it fails if the guard is removed from the load
/// path, even though `gguf_pretokenizer`'s unit tests would still pass.
#[cfg(feature = "llama-cpp")]
#[tokio::test]
async fn llama_cpp_load_refuses_gguf_with_pre_tokenizer_key() {
    let f = write_gguf(&[
        ("general.architecture", "qwen2"),
        (PRE_TOKENIZER_KEY, "qwen2"),
    ]);

    let config = abaddon::LlamaCppConfig::builder()
        .model_path(f.path())
        .n_gpu_layers(0)
        .build()
        .expect("config builds");

    // LlamaCppEngine has no Debug impl, so unwrap the Result by hand.
    let msg = match abaddon::LlamaCppEngine::load(config).await {
        Ok(_) => panic!("load must refuse a GGUF carrying tokenizer.ggml.pre"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("REFUSING TO LOAD"),
        "the guard must be the thing that rejected this, not llama.cpp: {msg}"
    );
    assert!(msg.contains(PRE_TOKENIZER_KEY), "must name the key: {msg}");
    assert!(msg.contains("#6920"), "must cite the upstream PR: {msg}");
}

/// `LlamaCppEngine::load` lets a GGUF *without* the key reach llama.cpp.
///
/// The fixture is not a real model, so llama.cpp rejects it for its own
/// reasons — which is the proof: the failure must come from llama.cpp, not
/// from the guard. Without this direction the guard could regress to
/// refusing everything and the test above would still pass.
#[cfg(feature = "llama-cpp")]
#[tokio::test]
async fn llama_cpp_load_passes_gguf_without_pre_tokenizer_key_to_llama_cpp() {
    let f = write_gguf(&[("general.architecture", "llama")]);

    let config = abaddon::LlamaCppConfig::builder()
        .model_path(f.path())
        .n_gpu_layers(0)
        .build()
        .expect("config builds");

    // Expected to fail — the fixture has no tensors — but *not* via the guard.
    let msg = match abaddon::LlamaCppEngine::load(config).await {
        Ok(_) => panic!("a header-only fixture is not a loadable model"),
        Err(e) => e.to_string(),
    };
    assert!(
        !msg.contains("REFUSING TO LOAD"),
        "the guard must not fire on a GGUF without the key: {msg}"
    );
    assert!(
        msg.contains("llama.cpp"),
        "the rejection should come from llama.cpp itself: {msg}"
    );
}
