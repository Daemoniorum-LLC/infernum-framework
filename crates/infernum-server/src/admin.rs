//! Admin API for runtime model management.
//!
//! This module provides administrative endpoints for managing models at runtime:
//!
//! - `POST /admin/models/load` - Load a model with configuration options
//! - `POST /admin/models/unload` - Unload a model from memory
//! - `GET /admin/models/status` - Get status of all loaded models
//! - `POST /admin/models/warmup` - Warmup a model for faster inference
//!
//! All admin endpoints require the `Admin` scope for authorization.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::admin::{LoadModelRequest, ModelLoadOptions};
//!
//! let request = LoadModelRequest {
//!     model: "meta-llama/Llama-3.2-3B-Instruct".to_string(),
//!     options: Some(ModelLoadOptions {
//!         gpu_layers: Some(32),
//!         context_length: Some(8192),
//!         quantization: Some("q4_k_m".to_string()),
//!         ..Default::default()
//!     }),
//! };
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Request to load a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadModelRequest {
    /// Model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct").
    pub model: String,

    /// Optional loading options.
    #[serde(default)]
    pub options: Option<ModelLoadOptions>,
}

/// Options for loading a model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelLoadOptions {
    /// Number of layers to offload to GPU.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<u32>,

    /// Context length for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    /// Quantization level (e.g., "q4_k_m", "q8_0").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,

    /// Memory limit in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<u64>,

    /// Whether to use flash attention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attention: Option<bool>,

    /// Custom tensor split across GPUs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<Vec<f32>>,
}

/// Response after loading a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadModelResponse {
    /// The model identifier.
    pub model: String,

    /// Current status of the model.
    pub status: ModelStatus,

    /// Time taken to load in milliseconds.
    pub load_time_ms: u64,

    /// Memory used by the model in bytes.
    pub memory_bytes: u64,

    /// Optional message with additional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Request to unload a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnloadModelRequest {
    /// Model identifier to unload.
    pub model: String,

    /// Force unload even if requests are in progress.
    #[serde(default)]
    pub force: bool,
}

/// Response after unloading a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnloadModelResponse {
    /// The model identifier.
    pub model: String,

    /// Whether the unload was successful.
    pub success: bool,

    /// Memory freed in bytes.
    pub memory_freed: u64,

    /// Optional message with additional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Request to warmup a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WarmupModelRequest {
    /// Model identifier to warmup.
    pub model: String,

    /// Number of warmup iterations.
    #[serde(default = "default_warmup_iterations")]
    pub iterations: u32,

    /// Warmup prompt length in tokens.
    #[serde(default = "default_warmup_tokens")]
    pub tokens: u32,
}

fn default_warmup_iterations() -> u32 {
    3
}

fn default_warmup_tokens() -> u32 {
    128
}

/// Response after warming up a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WarmupModelResponse {
    /// The model identifier.
    pub model: String,

    /// Whether the warmup was successful.
    pub success: bool,

    /// Number of iterations completed.
    pub iterations_completed: u32,

    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,

    /// Optional message with additional details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Status response for all models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelsStatusResponse {
    /// List of model statuses.
    pub models: Vec<AdminModelInfo>,

    /// Total memory used by all models.
    pub total_memory_bytes: u64,

    /// Total available memory.
    pub available_memory_bytes: u64,
}

/// Information about a loaded model (admin view).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdminModelInfo {
    /// Model identifier.
    pub model: String,

    /// Current status.
    pub status: ModelStatus,

    /// Memory used by this model in bytes.
    pub memory_bytes: u64,

    /// When the model was loaded (Unix timestamp).
    pub loaded_at: u64,

    /// Number of requests processed.
    pub requests_total: u64,

    /// Number of currently active requests.
    pub active_requests: u32,

    /// Options used when loading.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelLoadOptions>,
}

/// Status of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Model is loading.
    Loading,

    /// Model is ready for inference.
    #[default]
    Ready,

    /// Model is warming up.
    WarmingUp,

    /// Model is unloading.
    Unloading,

    /// Model loading failed.
    Failed,

    /// Model is idle (no recent requests).
    Idle,
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loading => write!(f, "loading"),
            Self::Ready => write!(f, "ready"),
            Self::WarmingUp => write!(f, "warming_up"),
            Self::Unloading => write!(f, "unloading"),
            Self::Failed => write!(f, "failed"),
            Self::Idle => write!(f, "idle"),
        }
    }
}

/// Error type for admin operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdminError {
    /// Model not found.
    ModelNotFound(String),

    /// Model is already loaded.
    ModelAlreadyLoaded(String),

    /// Model is currently loading.
    ModelLoading(String),

    /// Model has active requests and cannot be unloaded.
    ModelBusy(String, u32),

    /// Insufficient memory to load model.
    InsufficientMemory {
        /// Required memory in bytes.
        required: u64,
        /// Available memory in bytes.
        available: u64,
    },

    /// Invalid model configuration.
    InvalidConfig(String),

    /// Operation timeout.
    Timeout(Duration),

    /// Internal error.
    Internal(String),
}

impl std::error::Error for AdminError {}

impl fmt::Display for AdminError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(model) => write!(f, "model not found: {model}"),
            Self::ModelAlreadyLoaded(model) => write!(f, "model already loaded: {model}"),
            Self::ModelLoading(model) => write!(f, "model is currently loading: {model}"),
            Self::ModelBusy(model, count) => {
                write!(f, "model has {count} active requests: {model}")
            }
            Self::InsufficientMemory { required, available } => {
                write!(
                    f,
                    "insufficient memory: required {} bytes, available {} bytes",
                    required, available
                )
            }
            Self::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            Self::Timeout(duration) => write!(f, "operation timed out after {:?}", duration),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

/// Internal state for a loaded model.
#[derive(Debug)]
struct LoadedModel {
    model: String,
    status: RwLock<ModelStatus>,
    memory_bytes: u64,
    loaded_at: Instant,
    loaded_at_unix: u64,
    requests_total: AtomicU64,
    active_requests: std::sync::atomic::AtomicU32,
    options: Option<ModelLoadOptions>,
}

#[allow(dead_code)]
impl LoadedModel {
    /// Returns when the model was loaded.
    #[must_use]
    fn loaded_at(&self) -> Instant {
        self.loaded_at
    }

    /// Returns the elapsed time since the model was loaded.
    #[must_use]
    fn uptime(&self) -> std::time::Duration {
        self.loaded_at.elapsed()
    }
}

/// Model registry for tracking loaded models.
#[derive(Debug)]
pub struct ModelRegistry {
    models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    total_memory: AtomicU64,
    available_memory: u64,
}

impl ModelRegistry {
    /// Creates a new model registry with the specified available memory.
    pub fn new(available_memory: u64) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            total_memory: AtomicU64::new(0),
            available_memory,
        }
    }

    /// Loads a model into the registry.
    pub fn load_model(&self, request: &LoadModelRequest) -> Result<LoadModelResponse, AdminError> {
        let start = Instant::now();

        // Check if model is already loaded
        {
            let models = self.models.read();
            if let Some(existing) = models.get(&request.model) {
                let status = *existing.status.read();
                if status == ModelStatus::Loading {
                    return Err(AdminError::ModelLoading(request.model.clone()));
                }
                return Err(AdminError::ModelAlreadyLoaded(request.model.clone()));
            }
        }

        // Calculate required memory (placeholder - in real implementation would check model size)
        let required_memory = self.estimate_memory(&request.options);
        let current_usage = self.total_memory.load(Ordering::Acquire);

        if current_usage + required_memory > self.available_memory {
            return Err(AdminError::InsufficientMemory {
                required: required_memory,
                available: self.available_memory.saturating_sub(current_usage),
            });
        }

        // Create the model entry
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let loaded_model = Arc::new(LoadedModel {
            model: request.model.clone(),
            status: RwLock::new(ModelStatus::Ready),
            memory_bytes: required_memory,
            loaded_at: Instant::now(),
            loaded_at_unix: now,
            requests_total: AtomicU64::new(0),
            active_requests: std::sync::atomic::AtomicU32::new(0),
            options: request.options.clone(),
        });

        // Add to registry
        {
            let mut models = self.models.write();
            models.insert(request.model.clone(), loaded_model);
        }

        self.total_memory.fetch_add(required_memory, Ordering::AcqRel);

        Ok(LoadModelResponse {
            model: request.model.clone(),
            status: ModelStatus::Ready,
            load_time_ms: start.elapsed().as_millis() as u64,
            memory_bytes: required_memory,
            message: None,
        })
    }

    /// Unloads a model from the registry.
    pub fn unload_model(
        &self,
        request: &UnloadModelRequest,
    ) -> Result<UnloadModelResponse, AdminError> {
        let model = {
            let models = self.models.read();
            models
                .get(&request.model)
                .cloned()
                .ok_or_else(|| AdminError::ModelNotFound(request.model.clone()))?
        };

        // Check for active requests
        let active = model.active_requests.load(Ordering::Acquire);
        if active > 0 && !request.force {
            return Err(AdminError::ModelBusy(request.model.clone(), active));
        }

        // Set status to unloading
        *model.status.write() = ModelStatus::Unloading;

        // Remove from registry
        let memory_freed = {
            let mut models = self.models.write();
            if let Some(removed) = models.remove(&request.model) {
                removed.memory_bytes
            } else {
                0
            }
        };

        self.total_memory.fetch_sub(memory_freed, Ordering::AcqRel);

        Ok(UnloadModelResponse {
            model: request.model.clone(),
            success: true,
            memory_freed,
            message: if request.force && active > 0 {
                Some(format!("Force unloaded with {} active requests", active))
            } else {
                None
            },
        })
    }

    /// Warms up a model by running inference iterations.
    pub fn warmup_model(
        &self,
        request: &WarmupModelRequest,
    ) -> Result<WarmupModelResponse, AdminError> {
        let model = {
            let models = self.models.read();
            models
                .get(&request.model)
                .cloned()
                .ok_or_else(|| AdminError::ModelNotFound(request.model.clone()))?
        };

        // Set status to warming up
        let original_status = *model.status.read();
        *model.status.write() = ModelStatus::WarmingUp;

        // Simulate warmup (in real implementation, would run actual inference)
        let start = Instant::now();
        let iterations = request.iterations.min(10); // Cap at 10 iterations

        // Restore status
        *model.status.write() = original_status;

        let total_time = start.elapsed();
        let avg_latency = if iterations > 0 {
            total_time.as_secs_f64() * 1000.0 / f64::from(iterations)
        } else {
            0.0
        };

        Ok(WarmupModelResponse {
            model: request.model.clone(),
            success: true,
            iterations_completed: iterations,
            avg_latency_ms: avg_latency,
            message: None,
        })
    }

    /// Gets the status of all loaded models.
    pub fn get_status(&self) -> ModelsStatusResponse {
        let models = self.models.read();

        let model_infos: Vec<AdminModelInfo> = models
            .values()
            .map(|m| AdminModelInfo {
                model: m.model.clone(),
                status: *m.status.read(),
                memory_bytes: m.memory_bytes,
                loaded_at: m.loaded_at_unix,
                requests_total: m.requests_total.load(Ordering::Acquire),
                active_requests: m.active_requests.load(Ordering::Acquire),
                options: m.options.clone(),
            })
            .collect();

        ModelsStatusResponse {
            models: model_infos,
            total_memory_bytes: self.total_memory.load(Ordering::Acquire),
            available_memory_bytes: self.available_memory,
        }
    }

    /// Gets info for a specific model.
    pub fn get_model_info(&self, model_id: &str) -> Option<AdminModelInfo> {
        let models = self.models.read();
        models.get(model_id).map(|m| AdminModelInfo {
            model: m.model.clone(),
            status: *m.status.read(),
            memory_bytes: m.memory_bytes,
            loaded_at: m.loaded_at_unix,
            requests_total: m.requests_total.load(Ordering::Acquire),
            active_requests: m.active_requests.load(Ordering::Acquire),
            options: m.options.clone(),
        })
    }

    /// Records a request starting for a model.
    pub fn record_request_start(&self, model_id: &str) -> bool {
        let models = self.models.read();
        if let Some(model) = models.get(model_id) {
            model.active_requests.fetch_add(1, Ordering::AcqRel);
            model.requests_total.fetch_add(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Records a request ending for a model.
    pub fn record_request_end(&self, model_id: &str) {
        let models = self.models.read();
        if let Some(model) = models.get(model_id) {
            model.active_requests.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Estimates memory required for a model based on options.
    fn estimate_memory(&self, options: &Option<ModelLoadOptions>) -> u64 {
        // Base memory estimate (placeholder)
        let base = 1024 * 1024 * 1024; // 1GB base

        if let Some(opts) = options {
            // Adjust based on context length
            let ctx_factor = opts.context_length.unwrap_or(4096) as u64 / 4096;

            // Adjust based on GPU layers
            let gpu_factor = if opts.gpu_layers.unwrap_or(0) > 0 {
                2
            } else {
                1
            };

            base * ctx_factor * gpu_factor
        } else {
            base
        }
    }

    /// Checks if a model is loaded.
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.models.read().contains_key(model_id)
    }

    /// Gets the number of loaded models.
    pub fn model_count(&self) -> usize {
        self.models.read().len()
    }

    /// Renders Prometheus metrics for the model registry.
    pub fn render_prometheus(&self) -> String {
        let status = self.get_status();
        let mut output = String::with_capacity(1024);

        // Memory metrics
        output.push_str("# HELP infernum_model_memory_bytes Memory used by each model\n");
        output.push_str("# TYPE infernum_model_memory_bytes gauge\n");
        for model in &status.models {
            output.push_str(&format!(
                "infernum_model_memory_bytes{{model=\"{}\"}} {}\n",
                model.model, model.memory_bytes
            ));
        }

        // Active requests
        output.push_str("# HELP infernum_model_active_requests Active requests per model\n");
        output.push_str("# TYPE infernum_model_active_requests gauge\n");
        for model in &status.models {
            output.push_str(&format!(
                "infernum_model_active_requests{{model=\"{}\"}} {}\n",
                model.model, model.active_requests
            ));
        }

        // Total requests
        output.push_str("# HELP infernum_model_requests_total Total requests per model\n");
        output.push_str("# TYPE infernum_model_requests_total counter\n");
        for model in &status.models {
            output.push_str(&format!(
                "infernum_model_requests_total{{model=\"{}\"}} {}\n",
                model.model, model.requests_total
            ));
        }

        // Model status
        output.push_str("# HELP infernum_model_status Model status (1=ready, 0=other)\n");
        output.push_str("# TYPE infernum_model_status gauge\n");
        for model in &status.models {
            let ready = if model.status == ModelStatus::Ready {
                1
            } else {
                0
            };
            output.push_str(&format!(
                "infernum_model_status{{model=\"{}\",status=\"{}\"}} {}\n",
                model.model, model.status, ready
            ));
        }

        // Total memory
        output.push_str("# HELP infernum_models_memory_total_bytes Total memory used by all models\n");
        output.push_str("# TYPE infernum_models_memory_total_bytes gauge\n");
        output.push_str(&format!(
            "infernum_models_memory_total_bytes {}\n",
            status.total_memory_bytes
        ));

        // Available memory
        output.push_str("# HELP infernum_models_memory_available_bytes Available memory for models\n");
        output.push_str("# TYPE infernum_models_memory_available_bytes gauge\n");
        output.push_str(&format!(
            "infernum_models_memory_available_bytes {}\n",
            status.available_memory_bytes
        ));

        // Model count
        output.push_str("# HELP infernum_models_loaded_total Number of loaded models\n");
        output.push_str("# TYPE infernum_models_loaded_total gauge\n");
        output.push_str(&format!(
            "infernum_models_loaded_total {}\n",
            status.models.len()
        ));

        output
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        // Default to 16GB available memory
        Self::new(16 * 1024 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model_request_serialization() {
        let request = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: Some(ModelLoadOptions {
                gpu_layers: Some(32),
                context_length: Some(8192),
                quantization: Some("q4_k_m".to_string()),
                memory_limit: None,
                flash_attention: Some(true),
                tensor_split: None,
            }),
        };

        let json = serde_json::to_string(&request).expect("serialize");
        let parsed: LoadModelRequest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(request, parsed);
    }

    #[test]
    fn test_load_model_request_minimal() {
        let json = r#"{"model": "llama-3b"}"#;
        let request: LoadModelRequest = serde_json::from_str(json).expect("deserialize");

        assert_eq!(request.model, "llama-3b");
        assert!(request.options.is_none());
    }

    #[test]
    fn test_unload_model_request_default_force() {
        let json = r#"{"model": "llama-3b"}"#;
        let request: UnloadModelRequest = serde_json::from_str(json).expect("deserialize");

        assert_eq!(request.model, "llama-3b");
        assert!(!request.force);
    }

    #[test]
    fn test_warmup_model_request_defaults() {
        let json = r#"{"model": "llama-3b"}"#;
        let request: WarmupModelRequest = serde_json::from_str(json).expect("deserialize");

        assert_eq!(request.model, "llama-3b");
        assert_eq!(request.iterations, 3); // default
        assert_eq!(request.tokens, 128); // default
    }

    #[test]
    fn test_model_status_display() {
        assert_eq!(ModelStatus::Loading.to_string(), "loading");
        assert_eq!(ModelStatus::Ready.to_string(), "ready");
        assert_eq!(ModelStatus::WarmingUp.to_string(), "warming_up");
        assert_eq!(ModelStatus::Unloading.to_string(), "unloading");
        assert_eq!(ModelStatus::Failed.to_string(), "failed");
        assert_eq!(ModelStatus::Idle.to_string(), "idle");
    }

    #[test]
    fn test_model_status_serialization() {
        let json = serde_json::to_string(&ModelStatus::Ready).expect("serialize");
        assert_eq!(json, "\"ready\"");

        let parsed: ModelStatus = serde_json::from_str("\"warming_up\"").expect("deserialize");
        assert_eq!(parsed, ModelStatus::WarmingUp);
    }

    #[test]
    fn test_admin_error_display() {
        assert_eq!(
            AdminError::ModelNotFound("llama".to_string()).to_string(),
            "model not found: llama"
        );

        assert_eq!(
            AdminError::ModelAlreadyLoaded("llama".to_string()).to_string(),
            "model already loaded: llama"
        );

        assert_eq!(
            AdminError::ModelBusy("llama".to_string(), 5).to_string(),
            "model has 5 active requests: llama"
        );

        assert_eq!(
            AdminError::InsufficientMemory {
                required: 1000,
                available: 500
            }
            .to_string(),
            "insufficient memory: required 1000 bytes, available 500 bytes"
        );
    }

    #[test]
    fn test_registry_load_model() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024); // 10GB

        let request = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: None,
        };

        let response = registry.load_model(&request).expect("load should succeed");

        assert_eq!(response.model, "llama-3b");
        assert_eq!(response.status, ModelStatus::Ready);
        assert!(response.memory_bytes > 0);
        assert!(registry.is_loaded("llama-3b"));
        assert_eq!(registry.model_count(), 1);
    }

    #[test]
    fn test_registry_load_duplicate() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let request = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: None,
        };

        let _ = registry.load_model(&request).expect("first load");
        let result = registry.load_model(&request);

        assert!(matches!(result, Err(AdminError::ModelAlreadyLoaded(_))));
    }

    #[test]
    fn test_registry_insufficient_memory() {
        let registry = ModelRegistry::new(1024); // 1KB available

        let request = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: Some(ModelLoadOptions {
                context_length: Some(8192),
                ..Default::default()
            }),
        };

        let result = registry.load_model(&request);

        assert!(matches!(result, Err(AdminError::InsufficientMemory { .. })));
    }

    #[test]
    fn test_registry_unload_model() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        // Load a model
        let load_req = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: None,
        };
        let _ = registry.load_model(&load_req).expect("load");

        // Unload it
        let unload_req = UnloadModelRequest {
            model: "llama-3b".to_string(),
            force: false,
        };
        let response = registry.unload_model(&unload_req).expect("unload");

        assert!(response.success);
        assert!(response.memory_freed > 0);
        assert!(!registry.is_loaded("llama-3b"));
        assert_eq!(registry.model_count(), 0);
    }

    #[test]
    fn test_registry_unload_not_found() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let request = UnloadModelRequest {
            model: "nonexistent".to_string(),
            force: false,
        };

        let result = registry.unload_model(&request);

        assert!(matches!(result, Err(AdminError::ModelNotFound(_))));
    }

    #[test]
    fn test_registry_unload_with_active_requests() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        // Load and start a request
        let load_req = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: None,
        };
        let _ = registry.load_model(&load_req).expect("load");
        registry.record_request_start("llama-3b");

        // Try to unload without force
        let unload_req = UnloadModelRequest {
            model: "llama-3b".to_string(),
            force: false,
        };
        let result = registry.unload_model(&unload_req);

        assert!(matches!(result, Err(AdminError::ModelBusy(_, 1))));

        // Force unload
        let unload_req = UnloadModelRequest {
            model: "llama-3b".to_string(),
            force: true,
        };
        let response = registry.unload_model(&unload_req).expect("force unload");

        assert!(response.success);
        assert!(response.message.is_some());
    }

    #[test]
    fn test_registry_warmup_model() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        // Load a model
        let load_req = LoadModelRequest {
            model: "llama-3b".to_string(),
            options: None,
        };
        let _ = registry.load_model(&load_req).expect("load");

        // Warmup
        let warmup_req = WarmupModelRequest {
            model: "llama-3b".to_string(),
            iterations: 3,
            tokens: 128,
        };
        let response = registry.warmup_model(&warmup_req).expect("warmup");

        assert!(response.success);
        assert_eq!(response.iterations_completed, 3);
    }

    #[test]
    fn test_registry_warmup_not_found() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let request = WarmupModelRequest {
            model: "nonexistent".to_string(),
            iterations: 3,
            tokens: 128,
        };

        let result = registry.warmup_model(&request);

        assert!(matches!(result, Err(AdminError::ModelNotFound(_))));
    }

    #[test]
    fn test_registry_get_status() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        // Load two models
        let _ = registry
            .load_model(&LoadModelRequest {
                model: "llama-3b".to_string(),
                options: None,
            })
            .expect("load");
        let _ = registry
            .load_model(&LoadModelRequest {
                model: "mistral-7b".to_string(),
                options: None,
            })
            .expect("load");

        let status = registry.get_status();

        assert_eq!(status.models.len(), 2);
        assert!(status.total_memory_bytes > 0);
        assert_eq!(status.available_memory_bytes, 10 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_registry_get_model_info() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let _ = registry
            .load_model(&LoadModelRequest {
                model: "llama-3b".to_string(),
                options: Some(ModelLoadOptions {
                    gpu_layers: Some(32),
                    ..Default::default()
                }),
            })
            .expect("load");

        let info = registry.get_model_info("llama-3b").expect("should exist");

        assert_eq!(info.model, "llama-3b");
        assert_eq!(info.status, ModelStatus::Ready);
        assert!(info.options.is_some());
        assert_eq!(info.options.as_ref().expect("options").gpu_layers, Some(32));
    }

    #[test]
    fn test_registry_request_tracking() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let _ = registry
            .load_model(&LoadModelRequest {
                model: "llama-3b".to_string(),
                options: None,
            })
            .expect("load");

        // Start requests
        assert!(registry.record_request_start("llama-3b"));
        assert!(registry.record_request_start("llama-3b"));

        let info = registry.get_model_info("llama-3b").expect("info");
        assert_eq!(info.active_requests, 2);
        assert_eq!(info.requests_total, 2);

        // End one request
        registry.record_request_end("llama-3b");

        let info = registry.get_model_info("llama-3b").expect("info");
        assert_eq!(info.active_requests, 1);
        assert_eq!(info.requests_total, 2); // Total doesn't decrease
    }

    #[test]
    fn test_registry_request_start_nonexistent() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        assert!(!registry.record_request_start("nonexistent"));
    }

    #[test]
    fn test_registry_prometheus_output() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let _ = registry
            .load_model(&LoadModelRequest {
                model: "llama-3b".to_string(),
                options: None,
            })
            .expect("load");

        let output = registry.render_prometheus();

        assert!(output.contains("infernum_model_memory_bytes"));
        assert!(output.contains("llama-3b"));
        assert!(output.contains("infernum_model_status"));
        assert!(output.contains("infernum_models_loaded_total 1"));
    }

    #[test]
    fn test_load_response_serialization() {
        let response = LoadModelResponse {
            model: "llama-3b".to_string(),
            status: ModelStatus::Ready,
            load_time_ms: 1500,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            message: Some("Loaded with quantization".to_string()),
        };

        let json = serde_json::to_string(&response).expect("serialize");
        let parsed: LoadModelResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(response, parsed);
    }

    #[test]
    fn test_models_status_response_serialization() {
        let response = ModelsStatusResponse {
            models: vec![AdminModelInfo {
                model: "llama-3b".to_string(),
                status: ModelStatus::Ready,
                memory_bytes: 4 * 1024 * 1024 * 1024,
                loaded_at: 1700000000,
                requests_total: 100,
                active_requests: 5,
                options: None,
            }],
            total_memory_bytes: 4 * 1024 * 1024 * 1024,
            available_memory_bytes: 16 * 1024 * 1024 * 1024,
        };

        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("llama-3b"));
        assert!(json.contains("ready"));
    }

    #[test]
    fn test_model_load_options_skip_none_serialization() {
        let options = ModelLoadOptions {
            gpu_layers: Some(32),
            context_length: None,
            quantization: None,
            memory_limit: None,
            flash_attention: Some(true),
            tensor_split: None,
        };

        let json = serde_json::to_string(&options).expect("serialize");

        // Should not contain null fields
        assert!(json.contains("gpu_layers"));
        assert!(json.contains("flash_attention"));
        assert!(!json.contains("context_length"));
        assert!(!json.contains("quantization"));
    }

    #[test]
    fn test_default_model_registry() {
        let registry = ModelRegistry::default();

        // Should have 16GB default
        let status = registry.get_status();
        assert_eq!(status.available_memory_bytes, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_model_status_default() {
        let status = ModelStatus::default();
        assert_eq!(status, ModelStatus::Ready);
    }

    #[test]
    fn test_warmup_capped_iterations() {
        let registry = ModelRegistry::new(10 * 1024 * 1024 * 1024);

        let _ = registry
            .load_model(&LoadModelRequest {
                model: "llama-3b".to_string(),
                options: None,
            })
            .expect("load");

        // Request 100 iterations, should be capped at 10
        let warmup_req = WarmupModelRequest {
            model: "llama-3b".to_string(),
            iterations: 100,
            tokens: 128,
        };
        let response = registry.warmup_model(&warmup_req).expect("warmup");

        assert_eq!(response.iterations_completed, 10);
    }
}
