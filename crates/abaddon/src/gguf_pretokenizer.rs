//! Load-time guard against silent BPE mis-tokenization.
//!
//! # The defect this guards (issue #56)
//!
//! `crates/abaddon` is pinned to crates.io `llama_cpp` 0.3.2. The llama.cpp
//! that `llama_cpp_sys-0.3.2` vendors predates upstream **PR #6920**
//! (merged 2024-04-29), which introduced BPE pre-tokenizer typing: the
//! `tokenizer.ggml.pre` GGUF metadata key and the `LLAMA_VOCAB_PRE_TYPE_*`
//! constants that interpret it. Inspecting the vendored tree:
//!
//! ```text
//! $ grep -o 'LLAMA_VOCAB_PRE_TYPE_[A-Z0-9_]*' llama.cpp | sort -u | wc -l
//! 0
//! $ grep -c 'tokenizer.ggml.pre' llama.cpp
//! 0
//! ```
//!
//! Every GGUF produced by a current `convert_hf_to_gguf.py` carries that key.
//! A loader that cannot read it falls back to a default pre-tokenizer regex,
//! so the model loads **successfully** and then tokenizes **incorrectly** —
//! for every BPE-tokenized model, which is every current Qwen2/2.5 and
//! Llama-3 derivative.
//!
//! # Why refuse rather than warn
//!
//! The failure is silent and downstream. `general.architecture = qwen2` is in
//! the vendored architecture list, so the model loads without complaint and
//! the corruption surfaces only as degraded output — and the first thing to
//! degrade under bad tokenization is structured output, i.e. tool calls.
//! A warning in a log that nobody reads is indistinguishable from the silent
//! failure it describes. So this refuses the load.
//!
//! # Fail-open by design
//!
//! [`read_pre_tokenizer_key`] answers exactly one narrow question — *does this
//! file carry `tokenizer.ggml.pre`?* — and returns `None` for anything it
//! cannot confidently parse: a short read, an unknown GGUF version, a
//! malformed value type, a file that is not GGUF at all. A guard that
//! produced false refusals would be worse than the defect it catches, so
//! uncertainty always permits the load.
//!
//! Correspondingly, `None` is **not** evidence that a model is safe. It means
//! the question could not be answered here.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// GGUF magic number, little-endian `"GGUF"`.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// The metadata key introduced by llama.cpp PR #6920.
pub const PRE_TOKENIZER_KEY: &str = "tokenizer.ggml.pre";

/// Upper bound on a single metadata string, to bound allocation from a
/// corrupt length field. Real GGUF metadata strings are far smaller; a
/// chat template is the largest and runs to a few tens of KB.
const MAX_STRING_LEN: u64 = 8 * 1024 * 1024;

/// Upper bound on metadata KV pairs, likewise a corruption bound.
const MAX_KV_COUNT: u64 = 100_000;

/// Upper bound on array elements in a metadata value. Token vocabularies
/// live here and run to a few hundred thousand entries.
const MAX_ARRAY_LEN: u64 = 10_000_000;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Refuses the load when `path` carries a pre-tokenizer key this backend
/// cannot honour.
///
/// Call this **before** handing the file to `llama_cpp`, which ignores the
/// key rather than reporting it.
///
/// # Errors
///
/// Returns [`infernum_core::Error::ModelLoad`] when `tokenizer.ggml.pre` is
/// present. The message names the key, explains that the vendored llama.cpp
/// predates PR #6920, and states that continuing would tokenize incorrectly.
///
/// Returns `Ok(())` when the key is absent **or** when the file could not be
/// parsed — see the module docs on fail-open behaviour.
pub fn ensure_pre_tokenizer_supported(path: &Path) -> infernum_core::Result<()> {
    let Some(value) = read_pre_tokenizer_key(path) else {
        return Ok(());
    };

    Err(infernum_core::Error::ModelLoad {
        message: format!(
            "REFUSING TO LOAD: this GGUF requires a pre-tokenizer this backend cannot honour.\n\
             \n\
             File:  {path}\n\
             Key:   {PRE_TOKENIZER_KEY} = \"{value}\"\n\
             \n\
             This build is pinned to llama_cpp 0.3.2, whose vendored llama.cpp predates\n\
             upstream PR #6920 (merged 2024-04-29). That PR introduced the\n\
             `{PRE_TOKENIZER_KEY}` key and the LLAMA_VOCAB_PRE_TYPE_* constants that\n\
             interpret it. This backend contains neither, so it cannot honour the value\n\
             above and would silently fall back to a default pre-tokenizer regex.\n\
             \n\
             Loading anyway would tokenize this model INCORRECTLY -- not sub-optimally,\n\
             not slightly differently, but wrongly, for every BPE-tokenized model\n\
             (all Qwen2/2.5 and Llama-3 derivatives). The model would appear to load and\n\
             run normally while producing corrupted tokenization. Structured output --\n\
             tool calls above all -- degrades first and most.\n\
             \n\
             This is not a warning you can suppress and proceed past. Use one of:\n\
             \n\
               1. Run the model out of process and use --backend openai:\n\
                    llama-server -m {path} -c 32768 --port 8080\n\
                    infernum generate \"...\" --backend openai --api-base http://localhost:8080/v1\n\
                    (tokenization then belongs to a current llama.cpp, not to this binding)\n\
             \n\
               2. Use a GGUF converted before PR #6920, which carries no such key.\n\
                  Correct for that file, but it forgoes every model released since.\n\
             \n\
             Tracking: https://github.com/Daemoniorum-LLC/infernum-framework/issues/56",
            path = path.display(),
        ),
    })
}

/// Returns the value of `tokenizer.ggml.pre` when the file carries it.
///
/// Returns `None` when the key is absent, or whenever the file cannot be
/// confidently parsed. See the module docs: `None` means "not found here",
/// never "verified safe".
#[must_use]
pub fn read_pre_tokenizer_key(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    scan_metadata(&mut reader).unwrap_or(None)
}

// ---------------------------------------------------------------------------
// GGUF header walk
// ---------------------------------------------------------------------------

/// GGUF metadata value type tags, per the GGUF spec.
mod value_type {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// Walks the metadata KV block looking for [`PRE_TOKENIZER_KEY`].
///
/// `Ok(None)` means parsed cleanly and the key was absent; `Err` means the
/// walk could not continue, which callers treat the same way (fail open).
fn scan_metadata<R: Read>(reader: &mut R) -> std::io::Result<Option<String>> {
    if read_u32(reader)? != GGUF_MAGIC {
        return Ok(None);
    }

    let version = read_u32(reader)?;
    // v1 used 32-bit lengths throughout and predates every model this guard
    // is concerned with. Refuse to guess at its layout.
    if !(2..=3).contains(&version) {
        return Ok(None);
    }

    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;
    if kv_count > MAX_KV_COUNT {
        return Ok(None);
    }

    for _ in 0..kv_count {
        let Some(key) = read_string(reader)? else {
            return Ok(None);
        };
        let vtype = read_u32(reader)?;

        if key == PRE_TOKENIZER_KEY {
            // The key is what matters; a non-string value still means the
            // file expects pre-tokenizer typing, so report it either way.
            return Ok(Some(if vtype == value_type::STRING {
                read_string(reader)?.unwrap_or_else(|| "<unreadable>".to_string())
            } else {
                format!("<non-string value, type {vtype}>")
            }));
        }

        if !skip_value(reader, vtype)? {
            return Ok(None);
        }
    }

    Ok(None)
}

/// Skips one metadata value of the given type. Returns `false` on a type tag
/// the walk cannot interpret, which ends the scan.
fn skip_value<R: Read>(reader: &mut R, vtype: u32) -> std::io::Result<bool> {
    match vtype {
        value_type::UINT8 | value_type::INT8 | value_type::BOOL => skip(reader, 1)?,
        value_type::UINT16 | value_type::INT16 => skip(reader, 2)?,
        value_type::UINT32 | value_type::INT32 | value_type::FLOAT32 => skip(reader, 4)?,
        value_type::UINT64 | value_type::INT64 | value_type::FLOAT64 => skip(reader, 8)?,
        value_type::STRING => {
            if read_string(reader)?.is_none() {
                return Ok(false);
            }
        },
        value_type::ARRAY => {
            let elem_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            if len > MAX_ARRAY_LEN {
                return Ok(false);
            }
            // Arrays never nest in GGUF, so this recursion is one level deep.
            for _ in 0..len {
                if !skip_value(reader, elem_type)? {
                    return Ok(false);
                }
            }
        },
        _ => return Ok(false),
    }
    Ok(true)
}

/// Reads a GGUF string: u64 length followed by that many UTF-8 bytes.
///
/// `None` on an implausible length or invalid UTF-8.
fn read_string<R: Read>(reader: &mut R) -> std::io::Result<Option<String>> {
    let len = read_u64(reader)?;
    if len > MAX_STRING_LEN {
        return Ok(None);
    }
    let mut buf = vec![0u8; usize::try_from(len).unwrap_or(0)];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).ok())
}

fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Discards `n` bytes.
fn skip<R: Read>(reader: &mut R, n: u64) -> std::io::Result<()> {
    std::io::copy(&mut reader.take(n), &mut std::io::sink())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF v3 header builder, enough to exercise the metadata walk.
    struct GgufBuilder {
        kvs: Vec<u8>,
        count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                kvs: Vec::new(),
                count: 0,
            }
        }

        fn push_key(&mut self, key: &str) {
            self.kvs
                .extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(key.as_bytes());
        }

        fn string(mut self, key: &str, value: &str) -> Self {
            self.push_key(key);
            self.kvs
                .extend_from_slice(&value_type::STRING.to_le_bytes());
            self.kvs
                .extend_from_slice(&(value.len() as u64).to_le_bytes());
            self.kvs.extend_from_slice(value.as_bytes());
            self.count += 1;
            self
        }

        fn u32(mut self, key: &str, value: u32) -> Self {
            self.push_key(key);
            self.kvs
                .extend_from_slice(&value_type::UINT32.to_le_bytes());
            self.kvs.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
            self
        }

        fn bool(mut self, key: &str, value: bool) -> Self {
            self.push_key(key);
            self.kvs.extend_from_slice(&value_type::BOOL.to_le_bytes());
            self.kvs.push(u8::from(value));
            self.count += 1;
            self
        }

        /// A string array, the shape a real token vocabulary takes.
        fn string_array(mut self, key: &str, values: &[&str]) -> Self {
            self.push_key(key);
            self.kvs.extend_from_slice(&value_type::ARRAY.to_le_bytes());
            self.kvs
                .extend_from_slice(&value_type::STRING.to_le_bytes());
            self.kvs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.kvs.extend_from_slice(&(v.len() as u64).to_le_bytes());
                self.kvs.extend_from_slice(v.as_bytes());
            }
            self.count += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            out.extend_from_slice(&self.count.to_le_bytes());
            out.extend_from_slice(&self.kvs);
            out
        }
    }

    /// Writes bytes to a temp file and returns it (kept alive by the caller).
    fn temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(bytes).expect("write");
        f.flush().expect("flush");
        f
    }

    // -- The two directional guarantees the guard must never invert ---------

    /// A GGUF carrying the key is REFUSED.
    #[test]
    fn gguf_with_pre_tokenizer_key_is_refused() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "qwen2")
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        let f = temp_gguf(&bytes);

        let err = ensure_pre_tokenizer_supported(f.path())
            .expect_err("a GGUF carrying tokenizer.ggml.pre must be refused");

        let msg = err.to_string();
        assert!(msg.contains(PRE_TOKENIZER_KEY), "must name the key: {msg}");
        assert!(msg.contains("#6920"), "must cite the upstream PR: {msg}");
        assert!(
            msg.contains("INCORRECTLY"),
            "must say incorrectly, not sub-optimally: {msg}"
        );
        assert!(msg.contains("issues/56"), "must reference #56: {msg}");
    }

    /// A GGUF without the key still LOADS.
    #[test]
    fn gguf_without_pre_tokenizer_key_is_allowed() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .build();
        let f = temp_gguf(&bytes);

        assert!(
            ensure_pre_tokenizer_supported(f.path()).is_ok(),
            "a GGUF without the key must still load"
        );
        assert_eq!(read_pre_tokenizer_key(f.path()), None);
    }

    // -- The walk must survive every value type that precedes the key -------

    /// The key is found after values of every scalar type and an array.
    ///
    /// Guards the skip logic: a mis-sized skip would desynchronise the walk
    /// and lose the key, silently inverting the guard.
    #[test]
    fn key_is_found_after_every_value_type() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "qwen2")
            .u32("qwen2.block_count", 48)
            .bool("tokenizer.ggml.add_bos_token", true)
            .string_array("tokenizer.ggml.tokens", &["<s>", "hello", "world"])
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        let f = temp_gguf(&bytes);

        assert_eq!(
            read_pre_tokenizer_key(f.path()),
            Some("qwen2".to_string()),
            "the walk must stay in sync across all value types"
        );
    }

    /// The key is reported even when it is the very last entry.
    #[test]
    fn key_is_found_as_final_entry() {
        let bytes = GgufBuilder::new()
            .string_array("tokenizer.ggml.tokens", &["a", "b"])
            .string(PRE_TOKENIZER_KEY, "llama-bpe")
            .build();
        let f = temp_gguf(&bytes);
        assert_eq!(
            read_pre_tokenizer_key(f.path()),
            Some("llama-bpe".to_string())
        );
    }

    // -- Fail-open: uncertainty must never refuse --------------------------

    #[test]
    fn non_gguf_file_is_allowed() {
        let f = temp_gguf(b"this is not a GGUF file at all");
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    #[test]
    fn truncated_header_is_allowed() {
        let full = GgufBuilder::new()
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        // Cut mid-metadata: the key is unreachable, so the guard must not
        // refuse on a guess.
        let f = temp_gguf(&full[..full.len() / 2]);
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    #[test]
    fn empty_file_is_allowed() {
        let f = temp_gguf(b"");
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    #[test]
    fn missing_file_is_allowed() {
        assert!(ensure_pre_tokenizer_supported(Path::new("/nonexistent/model.gguf")).is_ok());
    }

    /// An unknown GGUF version is not guessed at.
    #[test]
    fn unknown_version_is_allowed() {
        let mut bytes = GgufBuilder::new()
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let f = temp_gguf(&bytes);
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    /// A corrupt KV count is bounded rather than trusted.
    #[test]
    fn implausible_kv_count_is_allowed() {
        let mut bytes = GgufBuilder::new()
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        let f = temp_gguf(&bytes);
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    /// An unknown value type ends the walk instead of desynchronising it.
    #[test]
    fn unknown_value_type_is_allowed() {
        let mut builder = GgufBuilder::new();
        builder.push_key("weird.key");
        builder.kvs.extend_from_slice(&4242u32.to_le_bytes());
        builder.count += 1;
        let bytes = builder.string(PRE_TOKENIZER_KEY, "qwen2").build();
        let f = temp_gguf(&bytes);
        assert!(ensure_pre_tokenizer_supported(f.path()).is_ok());
    }

    /// The refusal message must not read as suppressible advice.
    #[test]
    fn refusal_message_offers_a_real_alternative() {
        let bytes = GgufBuilder::new()
            .string(PRE_TOKENIZER_KEY, "qwen2")
            .build();
        let f = temp_gguf(&bytes);
        let msg = ensure_pre_tokenizer_supported(f.path())
            .expect_err("must refuse")
            .to_string();

        assert!(msg.contains("REFUSING TO LOAD"), "{msg}");
        assert!(
            msg.contains("not a warning you can suppress"),
            "must foreclose reading it as ignorable: {msg}"
        );
        assert!(
            msg.contains("--backend openai"),
            "must name a working path forward: {msg}"
        );
    }

    /// A non-string value still trips the guard: the key's presence is the
    /// signal, not its type.
    #[test]
    fn non_string_value_still_refuses() {
        let bytes = GgufBuilder::new().u32(PRE_TOKENIZER_KEY, 7).build();
        let f = temp_gguf(&bytes);
        assert!(ensure_pre_tokenizer_supported(f.path()).is_err());
    }
}
