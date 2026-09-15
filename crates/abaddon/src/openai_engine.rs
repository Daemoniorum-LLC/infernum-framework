//! OpenAI-compatible HTTP inference backend.
//!
//! Implements [`InferenceEngine`] against any server speaking the OpenAI
//! REST dialect — upstream `llama-server`, vLLM, SGLang, TGI, LM Studio, or
//! Infernum's own server.
//!
//! # Why this backend exists
//!
//! The in-process [`llama_cpp_engine`](crate::llama_cpp_engine) is pinned to
//! `llama_cpp` 0.3.2, whose vendored llama.cpp predates upstream PR #6920
//! (BPE pre-tokenizer typing). GGUFs produced by a current
//! `convert_hf_to_gguf.py` carry a `tokenizer.ggml.pre` key that the pinned
//! loader ignores, so modern Qwen2/Llama-3 derivatives load successfully and
//! then tokenize **incorrectly and silently**. See issue #56.
//!
//! Driving an out-of-process server over HTTP sidesteps that entirely:
//! tokenization, chat templating, and architecture support all become the
//! server's problem, and model support stops tracking a Rust binding's
//! release cadence.
//!
//! # Usage
//!
//! ```ignore
//! use abaddon::openai_engine::{OpenAiConfig, OpenAiEngine};
//!
//! let config = OpenAiConfig::builder("http://localhost:8080/v1", "qwen2.5-coder")
//!     .context_length(32_768)
//!     .build();
//! let engine = OpenAiEngine::connect(config).await?;
//! ```
//!
//! # Endpoint mapping
//!
//! | [`PromptInput`] | endpoint |
//! |---|---|
//! | [`Messages`](PromptInput::Messages) | `POST {base_url}/chat/completions` |
//! | [`Text`](PromptInput::Text) | `POST {base_url}/completions` |
//! | [`Tokens`](PromptInput::Tokens) | unsupported — returns [`Error::Backend`] |
//!
//! The agentic loop always builds `GenerateRequest::chat(..)`, so
//! `/chat/completions` is the hot path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use infernum_core::{
    model::{LlamaVersion, ModelArchitecture, ModelMetadata, ModelSource},
    request::EmbedInput,
    response::{Choice, Embedding, EmbeddingData},
    streaming::{StreamChoice, StreamChunk, StreamDelta},
    EmbedRequest, EmbedResponse, Error, FinishReason, GenerateRequest, GenerateResponse, Message,
    ModelId, PromptInput, RequestId, Result, Role, TokenStream, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, warn};

/// Backend name reported in [`Error::Backend`].
const BACKEND: &str = "openai-http";

/// Default request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Default advertised context length when the caller does not specify one.
const DEFAULT_CONTEXT_LENGTH: u32 = 32_768;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the target server accepts a [`GrammarConstraint`].
///
/// This exists because "the server ignored it" and "the server enforced it"
/// are indistinguishable from the response, and the difference is the whole
/// point of asking for a grammar. Rather than guess, the caller declares what
/// it is pointed at, and a constraint that cannot be honoured is refused
/// before the request leaves the process.
///
/// [`GrammarConstraint`]: infernum_core::sampling::GrammarConstraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrammarSupport {
    /// The server accepts a GBNF string in a top-level `grammar` field.
    ///
    /// This is llama.cpp's `llama-server`, which is Infernum's settled
    /// out-of-process runtime, so it is the default.
    #[default]
    Gbnf,

    /// The server has no grammar facility.
    ///
    /// A request carrying a grammar is refused with [`Error::Backend`]
    /// rather than sent unconstrained. Declare this for vLLM, TGI, or the
    /// OpenAI API itself, none of which read a `grammar` field — they would
    /// drop it silently, which is exactly the failure this type prevents.
    None,
}

/// Configuration for [`OpenAiEngine`].
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    /// Base URL including any version prefix, e.g. `http://localhost:8080/v1`.
    ///
    /// A trailing slash is trimmed on construction.
    pub base_url: String,

    /// Model name passed in the request body.
    ///
    /// Single-model servers such as `llama-server` ignore this; vLLM and
    /// Infernum's own server use it to select a loaded model.
    pub model: String,

    /// Optional bearer token. Omitted entirely when `None`.
    pub api_key: Option<String>,

    /// Per-request timeout.
    pub timeout: Duration,

    /// Context length advertised through [`ModelMetadata`].
    ///
    /// Advisory only — the server enforces the real limit. Callers that care
    /// about accurate budgeting should set this to match the server's `-c`.
    pub context_length: u32,

    /// Architecture advertised through [`ModelMetadata`].
    ///
    /// Advisory only. A remote engine needs no architecture knowledge: the
    /// server owns tokenization and templating. It exists because
    /// [`ModelMetadata`] requires a variant and has no `Unknown`.
    pub architecture: ModelArchitecture,

    /// Whether the target server can honour a grammar constraint.
    ///
    /// Defaults to [`GrammarSupport::Gbnf`]. See that type for why this is
    /// declared rather than detected.
    pub grammar_support: GrammarSupport,
}

impl OpenAiConfig {
    /// Starts a builder for the given base URL and model name.
    #[must_use]
    pub fn builder(base_url: impl Into<String>, model: impl Into<String>) -> OpenAiConfigBuilder {
        OpenAiConfigBuilder {
            config: Self {
                base_url: base_url.into(),
                model: model.into(),
                api_key: None,
                timeout: DEFAULT_TIMEOUT,
                context_length: DEFAULT_CONTEXT_LENGTH,
                architecture: ModelArchitecture::Llama {
                    version: LlamaVersion::V3,
                },
                grammar_support: GrammarSupport::default(),
            },
        }
    }

    /// Returns `base_url` with any trailing slash removed.
    fn trimmed_base(&self) -> &str {
        self.base_url.trim_end_matches('/')
    }

    /// Builds a full endpoint URL for `path` (which must start with `/`).
    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.trimmed_base())
    }
}

/// Builder for [`OpenAiConfig`].
#[derive(Debug, Clone)]
pub struct OpenAiConfigBuilder {
    config: OpenAiConfig,
}

impl OpenAiConfigBuilder {
    /// Sets the bearer token.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Sets the per-request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Sets the advertised context length.
    #[must_use]
    pub fn context_length(mut self, length: u32) -> Self {
        self.config.context_length = length;
        self
    }

    /// Sets the advertised architecture.
    #[must_use]
    pub fn architecture(mut self, architecture: ModelArchitecture) -> Self {
        self.config.architecture = architecture;
        self
    }

    /// Declares whether the server can honour a grammar constraint.
    ///
    /// Leave at the default ([`GrammarSupport::Gbnf`]) for `llama-server`.
    /// Set [`GrammarSupport::None`] for a backend with no grammar facility,
    /// which makes a grammar-carrying request fail loudly instead of
    /// generating unconstrained text that looks constrained.
    #[must_use]
    pub fn grammar_support(mut self, support: GrammarSupport) -> Self {
        self.config.grammar_support = support;
        self
    }

    /// Finalizes the configuration.
    #[must_use]
    pub fn build(self) -> OpenAiConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// An [`InferenceEngine`] backed by an OpenAI-compatible HTTP server.
pub struct OpenAiEngine {
    client: reqwest::Client,
    config: OpenAiConfig,
    metadata: ModelMetadata,
    ready: AtomicBool,
}

impl OpenAiEngine {
    /// Creates an engine without contacting the server.
    ///
    /// [`is_ready`](InferenceEngine::is_ready) returns `false` until
    /// [`probe`](Self::probe) succeeds. Use this when the server may start
    /// after the engine is constructed; use [`connect`](Self::connect) to
    /// fail fast instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] if the HTTP client cannot be built.
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("failed to build HTTP client: {e}"),
            })?;

        let metadata = ModelMetadata::builder(config.model.as_str(), config.architecture.clone())
            .source(ModelSource::huggingface(config.model.as_str()))
            .context_length(config.context_length)
            .build();

        Ok(Self {
            client,
            config,
            metadata,
            ready: AtomicBool::new(false),
        })
    }

    /// Creates an engine and probes the server, failing if it is unreachable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] if the client cannot be built or the
    /// server does not respond to `GET {base_url}/models`.
    pub async fn connect(config: OpenAiConfig) -> Result<Self> {
        let engine = Self::new(config)?;
        engine.probe().await?;
        Ok(engine)
    }

    /// Queries `GET {base_url}/models` and marks the engine ready on success.
    ///
    /// Logs a warning when the configured model is not among those listed,
    /// but does not fail: single-model servers commonly report a filesystem
    /// path or an unrelated placeholder as the model id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] if the request fails or returns non-2xx.
    pub async fn probe(&self) -> Result<()> {
        let url = self.config.endpoint("/models");
        let response = self
            .authed(self.client.get(&url))
            .send()
            .await
            .map_err(|e| Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("GET {url} failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("GET {url} returned {status}"),
            });
        }

        if let Ok(body) = response.json::<ModelsResponse>().await {
            let ids: Vec<&str> = body.data.iter().map(|m| m.id.as_str()).collect();
            if !ids.is_empty() && !ids.contains(&self.config.model.as_str()) {
                warn!(
                    configured = %self.config.model,
                    available = ?ids,
                    "configured model not listed by server; sending it anyway"
                );
            }
        }

        self.ready.store(true, Ordering::Release);
        debug!(url = %url, "openai-http backend ready");
        Ok(())
    }

    /// Applies the bearer token to a request builder when configured.
    fn authed(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.config.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }

    /// Builds the JSON body and endpoint path for a generate request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] for [`PromptInput::Tokens`], which has no
    /// portable representation in the OpenAI dialect.
    fn build_body(&self, request: &GenerateRequest, stream: bool) -> Result<(String, Value)> {
        let s = &request.sampling;
        let mut body = json!({
            "model": self.config.model,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "max_tokens": s.max_tokens,
            "stream": stream,
        });

        // Only send optional knobs when they carry a non-default signal, so
        // strict servers that reject unknown-but-present nulls stay happy.
        if s.top_k > 0 {
            body["top_k"] = json!(s.top_k);
        }
        if s.min_p > 0.0 {
            body["min_p"] = json!(s.min_p);
        }
        if s.presence_penalty != 0.0 {
            body["presence_penalty"] = json!(s.presence_penalty);
        }
        if s.frequency_penalty != 0.0 {
            body["frequency_penalty"] = json!(s.frequency_penalty);
        }
        if s.repetition_penalty != 1.0 {
            body["repetition_penalty"] = json!(s.repetition_penalty);
        }
        if !s.stop_sequences.is_empty() {
            body["stop"] = json!(s.stop_sequences);
        }

        // A grammar is not one of the "omit when it carries no signal" knobs
        // above, and must never be treated as one. The others degrade
        // gracefully when dropped — a missing `top_k` yields slightly
        // different text. A dropped grammar yields *unconstrained* text that
        // the caller will parse as though it were constrained, which is the
        // same defect shape as a check that passes because nothing ran.
        // So: carried when it can be honoured, refused when it cannot,
        // silently dropped never. See infernum-framework#17.
        if let Some(grammar) = &s.grammar {
            match self.config.grammar_support {
                GrammarSupport::Gbnf => {
                    body["grammar"] = json!(grammar.to_gbnf());
                },
                GrammarSupport::None => {
                    return Err(Error::Backend {
                        backend: BACKEND.to_string(),
                        message: format!(
                            "request carries a grammar constraint but {} is configured as \
                             GrammarSupport::None, so the constraint cannot be honoured. \
                             Refusing rather than generating unconstrained output that would \
                             look constrained to the caller. Point at a llama.cpp `llama-server` \
                             (GrammarSupport::Gbnf), or drop SamplingParams.grammar.",
                            self.config.trimmed_base()
                        ),
                    });
                },
            }
        }

        let path = match &request.prompt {
            PromptInput::Messages(messages) => {
                body["messages"] = json!(messages.iter().map(wire_message).collect::<Vec<_>>());
                "/chat/completions"
            },
            PromptInput::Text(text) => {
                body["prompt"] = json!(text);
                "/completions"
            },
            PromptInput::Tokens(_) => {
                return Err(Error::Backend {
                    backend: BACKEND.to_string(),
                    message: "pre-tokenized input is not supported by the OpenAI HTTP backend; \
                              send Text or Messages instead"
                        .to_string(),
                });
            },
        };

        Ok((path.to_string(), body))
    }

    /// Sends a request and returns the parsed JSON body, mapping transport
    /// and status failures onto [`Error::Backend`].
    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.config.endpoint(path);
        let response = self
            .authed(self.client.post(&url))
            .json(body)
            .send()
            .await
            .map_err(|e| Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("POST {url} failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("POST {url} returned {status}: {}", truncate(&detail, 512)),
            });
        }

        response.json::<Value>().await.map_err(|e| Error::Backend {
            backend: BACKEND.to_string(),
            message: format!("POST {url} returned unreadable JSON: {e}"),
        })
    }
}

#[async_trait]
impl InferenceEngine for OpenAiEngine {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let started = Instant::now();
        let (path, body) = self.build_body(&request, false)?;
        let value = self.post_json(&path, &body).await?;

        let choices = parse_choices(&value);
        if choices.is_empty() {
            return Err(Error::Backend {
                backend: BACKEND.to_string(),
                message: format!(
                    "response contained no choices: {}",
                    truncate(&value.to_string(), 512)
                ),
            });
        }

        Ok(GenerateResponse {
            request_id: request.request_id,
            created: value
                .get("created")
                .and_then(Value::as_i64)
                .unwrap_or_else(unix_now),
            model: request
                .model
                .unwrap_or_else(|| ModelId::from(self.config.model.as_str())),
            choices,
            usage: parse_usage(&value),
            time_to_first_token_ms: None,
            total_time_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
        })
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<TokenStream> {
        let (path, body) = self.build_body(&request, true)?;
        let url = self.config.endpoint(&path);

        let response = self
            .authed(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("POST {url} failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(Error::Backend {
                backend: BACKEND.to_string(),
                message: format!("POST {url} returned {status}: {}", truncate(&detail, 512)),
            });
        }

        let request_id = request.request_id;
        let model = request
            .model
            .unwrap_or_else(|| ModelId::from(self.config.model.as_str()));

        // Re-frame the byte stream as SSE events, then as StreamChunks.
        // `buffer` carries the partial trailing line across chunk boundaries;
        // `done` latches on `data: [DONE]` so trailing bytes are ignored.
        let byte_stream = response.bytes_stream();
        let stream = futures::stream::unfold(
            SseState {
                inner: Box::pin(byte_stream),
                buffer: String::new(),
                pending: Vec::new(),
                done: false,
                request_id,
                model,
            },
            |mut state| async move {
                loop {
                    if let Some(chunk) = state.pending.pop() {
                        return Some((Ok(chunk), state));
                    }
                    if state.done {
                        return None;
                    }
                    match state.inner.next().await {
                        Some(Ok(bytes)) => {
                            state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                            let mut produced = drain_sse(&mut state.buffer, &mut state.done);
                            // Reverse so `pop()` yields them in arrival order.
                            produced.reverse();
                            let rid = state.request_id.clone();
                            let model = state.model.clone();
                            state.pending = produced
                                .into_iter()
                                .filter_map(|v| sse_to_chunk(&v, rid.clone(), &model))
                                .collect();
                        },
                        Some(Err(e)) => {
                            state.done = true;
                            return Some((
                                Err(Error::Backend {
                                    backend: BACKEND.to_string(),
                                    message: format!("stream read failed: {e}"),
                                }),
                                state,
                            ));
                        },
                        None => return None,
                    }
                }
            },
        );

        Ok(TokenStream::new(stream))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        let inputs: Vec<String> = match &request.input {
            EmbedInput::Single(s) => vec![s.clone()],
            EmbedInput::Multiple(v) => v.clone(),
        };

        let mut body = json!({ "model": self.config.model, "input": inputs });
        if let Some(dims) = request.dimensions {
            body["dimensions"] = json!(dims);
        }

        let value = self.post_json("/embeddings", &body).await?;

        let data = value
            .get("data")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| Embedding {
                        index: item
                            .get("index")
                            .and_then(Value::as_u64)
                            .map_or(i as u32, |n| n as u32),
                        embedding: EmbeddingData::Float(
                            item.get("embedding")
                                .and_then(Value::as_array)
                                .map(|a| {
                                    a.iter()
                                        .filter_map(Value::as_f64)
                                        .map(|f| f as f32)
                                        .collect()
                                })
                                .unwrap_or_default(),
                        ),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(EmbedResponse {
            request_id: request.request_id,
            model: request
                .model
                .unwrap_or_else(|| ModelId::from(self.config.model.as_str())),
            data,
            usage: parse_usage(&value),
        })
    }

    fn model_info(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// SSE plumbing
// ---------------------------------------------------------------------------

/// Carries partial-line state across `bytes_stream()` chunk boundaries.
struct SseState {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
    buffer: String,
    pending: Vec<StreamChunk>,
    done: bool,
    request_id: RequestId,
    model: ModelId,
}

/// Pulls every complete `data:` line out of `buffer`, leaving any partial
/// trailing line in place. Sets `done` on the `[DONE]` sentinel.
fn drain_sse(buffer: &mut String, done: &mut bool) -> Vec<Value> {
    let mut out = Vec::new();

    while let Some(newline) = buffer.find('\n') {
        let line = buffer[..newline].trim_end_matches('\r').trim().to_string();
        buffer.drain(..=newline);

        // Blank lines separate events; comments start with ':'.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            *done = true;
            break;
        }
        match serde_json::from_str::<Value>(payload) {
            Ok(v) => out.push(v),
            Err(e) => warn!(error = %e, "skipping unparseable SSE payload"),
        }
    }

    out
}

/// Converts one SSE JSON payload into a [`StreamChunk`].
///
/// Returns `None` for payloads carrying neither content nor a finish reason,
/// such as the leading role-only delta most servers emit.
fn sse_to_chunk(value: &Value, request_id: RequestId, model: &ModelId) -> Option<StreamChunk> {
    let choices = value.get("choices").and_then(Value::as_array)?;
    let first = choices.first()?;

    let finish_reason = first
        .get("finish_reason")
        .and_then(Value::as_str)
        .and_then(map_finish_reason);

    // Chat servers use `delta.content`; completion servers use `text`.
    let content = first
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(Value::as_str)
        .or_else(|| first.get("text").and_then(Value::as_str))
        .unwrap_or_default();

    if content.is_empty() && finish_reason.is_none() {
        return None;
    }

    Some(StreamChunk {
        request_id,
        model: model.clone(),
        choices: vec![StreamChoice {
            index: first
                .get("index")
                .and_then(Value::as_u64)
                .map_or(0, |n| n as u32),
            delta: StreamDelta::text(content),
            finish_reason,
        }],
        usage: value.get("usage").and_then(|u| {
            serde_json::from_value::<WireUsage>(u.clone())
                .ok()
                .map(Into::into)
        }),
    })
}

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

/// `GET /models` response shape.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// OpenAI usage block.
#[derive(Debug, Default, Deserialize, Serialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl From<WireUsage> for Usage {
    fn from(w: WireUsage) -> Self {
        Self {
            prompt_tokens: w.prompt_tokens,
            completion_tokens: w.completion_tokens,
            // Some servers omit the total; derive rather than report zero.
            total_tokens: if w.total_tokens > 0 {
                w.total_tokens
            } else {
                w.prompt_tokens + w.completion_tokens
            },
        }
    }
}

/// Renders a [`Message`] into the OpenAI wire shape.
fn wire_message(message: &Message) -> Value {
    let mut out = json!({
        "role": role_str(message.role),
        "content": message.content,
    });
    if let Some(name) = &message.name {
        out["name"] = json!(name);
    }
    if let Some(id) = &message.tool_call_id {
        out["tool_call_id"] = json!(id);
    }
    if let Some(calls) = &message.tool_calls {
        if !calls.is_empty() {
            out["tool_calls"] = json!(calls);
        }
    }
    out
}

/// Maps a [`Role`] onto its OpenAI wire name.
fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Extracts choices from a non-streaming response.
///
/// `Choice.text` is populated from `message.content` for chat responses,
/// because [`beleth`'s executor reads `choices[0].text`] and would otherwise
/// see an empty completion.
///
/// [`beleth`'s executor reads `choices[0].text`]: https://github.com/Daemoniorum-LLC/infernum-framework
fn parse_choices(value: &Value) -> Vec<Choice> {
    let Some(items) = value.get("choices").and_then(Value::as_array) else {
        return Vec::new();
    };

    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let message = item.get("message");
            let text = message
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
                .or_else(|| item.get("text").and_then(Value::as_str))
                .unwrap_or_default()
                .to_string();

            Choice {
                index: item
                    .get("index")
                    .and_then(Value::as_u64)
                    .map_or(i as u32, |n| n as u32),
                message: message.map(|_| Message {
                    role: Role::Assistant,
                    content: text.clone(),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }),
                text,
                finish_reason: item
                    .get("finish_reason")
                    .and_then(Value::as_str)
                    .and_then(map_finish_reason),
                logprobs: None,
            }
        })
        .collect()
}

/// Extracts the usage block, defaulting to zeros when absent.
fn parse_usage(value: &Value) -> Usage {
    value
        .get("usage")
        .and_then(|u| serde_json::from_value::<WireUsage>(u.clone()).ok())
        .unwrap_or_default()
        .into()
}

/// Maps an OpenAI `finish_reason` string onto [`FinishReason`].
///
/// Unknown values yield `None` rather than a wrong variant.
fn map_finish_reason(raw: &str) -> Option<FinishReason> {
    match raw {
        "stop" | "eos" => Some(FinishReason::Stop),
        "length" => Some(FinishReason::Length),
        "tool_calls" | "function_call" => Some(FinishReason::ToolCalls),
        "content_filter" => Some(FinishReason::ContentFilter),
        _ => None,
    }
}

/// Seconds since the Unix epoch, saturating at 0 before 1970.
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

/// Truncates `s` to `max` bytes on a char boundary, for error messages.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

use crate::engine::InferenceEngine;

#[cfg(test)]
mod tests {
    use infernum_core::SamplingParams;

    use super::*;

    fn cfg() -> OpenAiConfig {
        OpenAiConfig::builder("http://localhost:9/v1", "test-model").build()
    }

    #[test]
    fn endpoint_trims_trailing_slash() {
        let c = OpenAiConfig::builder("http://x/v1/", "m").build();
        assert_eq!(
            c.endpoint("/chat/completions"),
            "http://x/v1/chat/completions"
        );
    }

    #[test]
    fn messages_route_to_chat_completions() {
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let req = GenerateRequest::chat(vec![Message::user("hi")]);
        let (path, body) = engine.build_body(&req, false).unwrap();
        assert_eq!(path, "/chat/completions");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn text_routes_to_completions() {
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let mut req = GenerateRequest::chat(vec![]);
        req.prompt = PromptInput::Text("once upon".into());
        let (path, body) = engine.build_body(&req, true).unwrap();
        assert_eq!(path, "/completions");
        assert_eq!(body["prompt"], "once upon");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn tokens_are_rejected_not_silently_mangled() {
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let mut req = GenerateRequest::chat(vec![]);
        req.prompt = PromptInput::Tokens(vec![1, 2, 3]);
        assert!(engine.build_body(&req, false).is_err());
    }

    #[test]
    fn chat_choice_populates_text_from_message_content() {
        // Regression guard: beleth's executor reads choices[0].text, so a
        // chat response that only filled `message` would look empty to it.
        let v = json!({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "<answer>42</answer>"},
                "finish_reason": "stop"
            }]
        });
        let choices = parse_choices(&v);
        assert_eq!(choices[0].text, "<answer>42</answer>");
        assert_eq!(choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn completion_choice_reads_text_field() {
        let v = json!({"choices": [{"index": 0, "text": "hello", "finish_reason": "length"}]});
        let choices = parse_choices(&v);
        assert_eq!(choices[0].text, "hello");
        assert_eq!(choices[0].finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn usage_total_is_derived_when_server_omits_it() {
        let v = json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3}});
        assert_eq!(parse_usage(&v).total_tokens, 10);
    }

    #[test]
    fn unknown_finish_reason_is_none_not_a_wrong_variant() {
        assert_eq!(map_finish_reason("something_new"), None);
    }

    #[test]
    fn sse_drains_only_complete_lines() {
        let mut buf = String::from("data: {\"a\":1}\n\ndata: {\"b\"");
        let mut done = false;
        let out = drain_sse(&mut buf, &mut done);
        assert_eq!(out.len(), 1);
        assert!(!done);
        // The partial line survives for the next network chunk.
        assert_eq!(buf, "data: {\"b\"");
    }

    #[test]
    fn sse_done_sentinel_latches() {
        let mut buf = String::from("data: {\"a\":1}\ndata: [DONE]\n");
        let mut done = false;
        let out = drain_sse(&mut buf, &mut done);
        assert_eq!(out.len(), 1);
        assert!(done);
    }

    #[test]
    fn sse_role_only_delta_yields_no_chunk() {
        let v = json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]});
        assert!(sse_to_chunk(&v, RequestId::new(), &ModelId::from("m")).is_none());
    }

    #[test]
    fn sse_content_delta_becomes_chunk() {
        let v = json!({"choices": [{"index": 0, "delta": {"content": "tok"}}]});
        let chunk = sse_to_chunk(&v, RequestId::new(), &ModelId::from("m")).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("tok"));
    }

    #[test]
    fn engine_is_not_ready_before_probe() {
        assert!(!OpenAiEngine::new(cfg()).unwrap().is_ready());
    }

    // === Grammar constraints (infernum-framework#17) ===
    //
    // `build_body` used to assemble eleven sampling knobs and omit
    // `s.grammar` entirely — no warning, no error, no test. The caller
    // believed it was constrained; nothing said otherwise. These four
    // tests pin the three states a grammar can be in on the wire: carried,
    // absent, or refused. None of them can pass by accident, because each
    // asserts on a key that the unfixed `build_body` never writes.

    #[test]
    fn gbnf_grammar_reaches_the_wire() {
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let gbnf = "root ::= \"yes\" | \"no\"";
        let req = GenerateRequest::chat(vec![Message::user("hi")])
            .with_sampling(SamplingParams::default().with_gbnf(gbnf));

        let (_, body) = engine.build_body(&req, false).unwrap();

        assert_eq!(
            body["grammar"],
            json!(gbnf),
            "SamplingParams.grammar must reach llama-server's `grammar` field; \
             a dropped constraint is indistinguishable from an honoured one at \
             the call site, which is the whole defect"
        );
    }

    #[test]
    fn json_mode_is_expanded_to_gbnf_on_the_wire() {
        // llama-server speaks GBNF, not our enum. `Json` and `JsonSchema`
        // must be rendered through `to_gbnf()` rather than serialised as the
        // Rust variant name, which the server would reject.
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let req = GenerateRequest::chat(vec![Message::user("hi")])
            .with_sampling(SamplingParams::default().with_json_mode());

        let (_, body) = engine.build_body(&req, false).unwrap();

        let sent = body["grammar"].as_str().expect("grammar must be a string");
        assert!(
            sent.contains("root ::= value"),
            "expected the expanded JSON GBNF, got {sent:?}"
        );
    }

    #[test]
    fn no_grammar_sends_no_grammar_key() {
        // The other knobs are omitted when they carry no signal so strict
        // servers stay happy; `grammar` follows the same rule.
        let engine = OpenAiEngine::new(cfg()).unwrap();
        let req = GenerateRequest::chat(vec![Message::user("hi")]);

        let (_, body) = engine.build_body(&req, false).unwrap();

        assert!(
            body.get("grammar").is_none(),
            "an unconstrained request must not carry a grammar key"
        );
    }

    #[test]
    fn grammar_against_a_server_without_one_is_refused_not_dropped() {
        // The failure this whole ticket is about: a constraint that cannot be
        // honoured must stop the request, never travel as a no-op.
        let engine = OpenAiEngine::new(
            OpenAiConfig::builder("http://localhost:9/v1", "test-model")
                .grammar_support(GrammarSupport::None)
                .build(),
        )
        .unwrap();
        let req = GenerateRequest::chat(vec![Message::user("hi")])
            .with_sampling(SamplingParams::default().with_gbnf("root ::= \"a\""));

        let err = engine
            .build_body(&req, false)
            .expect_err("a grammar the backend cannot honour must be an error");
        let message = err.to_string();
        assert!(
            message.contains("grammar"),
            "the error must name the constraint it refused: {message}"
        );
    }

    #[test]
    fn unconstrained_request_still_works_without_grammar_support() {
        // Refusal is scoped to requests that actually carry a constraint.
        let engine = OpenAiEngine::new(
            OpenAiConfig::builder("http://localhost:9/v1", "test-model")
                .grammar_support(GrammarSupport::None)
                .build(),
        )
        .unwrap();
        let req = GenerateRequest::chat(vec![Message::user("hi")]);
        assert!(engine.build_body(&req, false).is_ok());
    }
}
