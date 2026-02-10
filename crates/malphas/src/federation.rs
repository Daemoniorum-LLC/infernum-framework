//! Multi-model federation and task-based routing.
//!
//! This module provides intelligent routing based on task types, enabling
//! the Jormungandr research initiative to dispatch requests to appropriate
//! model families (reasoning, code, validation, etc.).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use abaddon::Engine;
use infernum_core::{GenerateRequest, Result};
use parking_lot::RwLock;

use crate::registry::{ModelRegistry, RegisteredModel};

/// Task types for intelligent routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Analysis and design tasks requiring reasoning.
    Reasoning,
    /// Code generation and conversion tasks.
    CodeGeneration,
    /// Validation tasks (should use different model family to prevent bias).
    Validation,
    /// Fast summarization and checkpoint generation.
    Summarization,
    /// Embedding generation for RAG.
    Embedding,
    /// General-purpose chat.
    Chat,
    /// Fine-tuning data preparation.
    DataPrep,
}

impl TaskType {
    /// Infers task type from request content.
    pub fn infer(request: &GenerateRequest) -> Self {
        let content = format!("{:?}", request.prompt).to_lowercase();

        // Check for validation-related keywords
        if content.contains("validate")
            || content.contains("verify")
            || content.contains("check correctness")
            || content.contains("review")
        {
            return Self::Validation;
        }

        // Check for code-related keywords
        if content.contains("implement")
            || content.contains("convert")
            || content.contains("code")
            || content.contains("function")
            || content.contains("struct")
            || content.contains("class")
            || content.contains("sigil")
        {
            return Self::CodeGeneration;
        }

        // Check for reasoning keywords
        if content.contains("analyze")
            || content.contains("design")
            || content.contains("explain")
            || content.contains("why")
            || content.contains("how would")
        {
            return Self::Reasoning;
        }

        // Check for summarization
        if content.contains("summarize")
            || content.contains("checkpoint")
            || content.contains("brief")
        {
            return Self::Summarization;
        }

        // Check for embedding requests
        if content.contains("embed") || content.contains("vector") {
            return Self::Embedding;
        }

        Self::Chat
    }
}

/// Model family classification for diversity routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Anthropic Claude models.
    Claude,
    /// OpenAI GPT models.
    Gpt,
    /// Google Gemini models.
    Gemini,
    /// Meta Llama models.
    Llama,
    /// Mistral AI models.
    Mistral,
    /// DeepSeek models (strong at code).
    DeepSeek,
    /// Qwen models.
    Qwen,
    /// CodeLlama and other code-specialized models.
    CodeSpecialist,
    /// Fine-tuned Sigil specialist models.
    SigilSpecialist,
    /// Local quantized models.
    LocalQuantized,
    /// Unknown/other models.
    Other(String),
}

impl ModelFamily {
    /// Detects family from model ID.
    pub fn from_model_id(model_id: &str) -> Self {
        let id = model_id.to_lowercase();

        if id.contains("claude") {
            Self::Claude
        } else if id.contains("gpt") || id.contains("openai") {
            Self::Gpt
        } else if id.contains("gemini") {
            Self::Gemini
        } else if id.contains("llama") && id.contains("code") {
            Self::CodeSpecialist
        } else if id.contains("llama") {
            Self::Llama
        } else if id.contains("mistral") {
            Self::Mistral
        } else if id.contains("deepseek") {
            Self::DeepSeek
        } else if id.contains("qwen") {
            Self::Qwen
        } else if id.contains("sigil") {
            Self::SigilSpecialist
        } else if id.contains("gguf") || id.contains("q4") || id.contains("q8") {
            Self::LocalQuantized
        } else {
            Self::Other(model_id.to_string())
        }
    }

    /// Returns true if this family is well-suited for code tasks.
    pub fn is_code_specialist(&self) -> bool {
        matches!(
            self,
            Self::DeepSeek | Self::CodeSpecialist | Self::SigilSpecialist
        )
    }

    /// Returns true if this family is well-suited for reasoning tasks.
    pub fn is_reasoning_specialist(&self) -> bool {
        matches!(self, Self::Claude | Self::Gpt | Self::Gemini)
    }
}

/// Configuration for a model in the federation.
#[derive(Debug, Clone)]
pub struct FederatedModel {
    /// Model identifier.
    pub model_id: String,
    /// Model family for diversity tracking.
    pub family: ModelFamily,
    /// Task types this model is suitable for.
    pub suitable_tasks: Vec<TaskType>,
    /// Priority for routing (higher = preferred).
    pub priority: u8,
    /// Whether this is an external API (vs local).
    pub is_external: bool,
    /// Cost per 1K input tokens (USD).
    pub cost_per_1k_input: f64,
    /// Cost per 1K output tokens (USD).
    pub cost_per_1k_output: f64,
}

impl FederatedModel {
    /// Creates a new federated model configuration.
    pub fn new(model_id: impl Into<String>) -> Self {
        let id = model_id.into();
        let family = ModelFamily::from_model_id(&id);

        Self {
            model_id: id,
            family,
            suitable_tasks: vec![TaskType::Chat],
            priority: 50,
            is_external: false,
            cost_per_1k_input: 0.0,
            cost_per_1k_output: 0.0,
        }
    }

    /// Sets the model family.
    pub fn with_family(mut self, family: ModelFamily) -> Self {
        self.family = family;
        self
    }

    /// Sets suitable task types.
    pub fn with_tasks(mut self, tasks: Vec<TaskType>) -> Self {
        self.suitable_tasks = tasks;
        self
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Marks as external API.
    pub fn external(mut self) -> Self {
        self.is_external = true;
        self
    }

    /// Sets cost per 1K tokens.
    pub fn with_cost(mut self, input: f64, output: f64) -> Self {
        self.cost_per_1k_input = input;
        self.cost_per_1k_output = output;
        self
    }
}

/// Federation router for multi-model orchestration.
pub struct FederationRouter {
    /// Federated model configurations.
    models: RwLock<HashMap<String, FederatedModel>>,
    /// Task-to-model preferences.
    task_preferences: RwLock<HashMap<TaskType, Vec<String>>>,
    /// Model families used in recent requests (for diversity tracking).
    recent_families: RwLock<Vec<ModelFamily>>,
    /// Maximum recent families to track.
    max_recent: usize,
    /// Whether to enforce model diversity for validation.
    enforce_validation_diversity: bool,
}

impl FederationRouter {
    /// Creates a new federation router.
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            task_preferences: RwLock::new(HashMap::new()),
            recent_families: RwLock::new(Vec::new()),
            max_recent: 10,
            enforce_validation_diversity: true,
        }
    }

    /// Creates a router configured for Jormungandr research.
    pub fn jormungandr() -> Self {
        let router = Self::new();

        // Set up task preferences for Jormungandr
        let mut prefs = HashMap::new();
        prefs.insert(
            TaskType::Reasoning,
            vec!["claude-opus".into(), "gpt-4".into(), "gemini-pro".into()],
        );
        prefs.insert(
            TaskType::CodeGeneration,
            vec![
                "deepseek-coder".into(),
                "codellama".into(),
                "sigil-specialist".into(),
            ],
        );
        prefs.insert(
            TaskType::Validation,
            vec!["gemini-pro".into(), "claude-sonnet".into()],
        );
        prefs.insert(
            TaskType::Summarization,
            vec!["llama-3.2-3b".into(), "phi-3".into()],
        );

        *router.task_preferences.write() = prefs;
        router
    }

    /// Registers a federated model.
    pub fn register(&self, config: FederatedModel) {
        let model_id = config.model_id.clone();
        let tasks = config.suitable_tasks.clone();

        self.models.write().insert(model_id.clone(), config);

        // Update task preferences
        let mut prefs = self.task_preferences.write();
        for task in tasks {
            prefs.entry(task).or_default().push(model_id.clone());
        }
    }

    /// Routes a request based on task type.
    pub fn route_by_task(
        &self,
        request: &GenerateRequest,
        registry: &ModelRegistry,
    ) -> Result<Arc<Engine>> {
        let task = TaskType::infer(request);
        self.route_for_task(task, registry)
    }

    /// Routes for a specific task type.
    pub fn route_for_task(&self, task: TaskType, registry: &ModelRegistry) -> Result<Arc<Engine>> {
        let prefs = self.task_preferences.read();
        let models = self.models.read();

        // Get preferred models for this task
        let preferred = prefs.get(&task).cloned().unwrap_or_default();

        // For validation tasks, ensure we use a different family than recent
        if task == TaskType::Validation && self.enforce_validation_diversity {
            return self.route_diverse_validation(registry, &preferred);
        }

        // Try preferred models in order
        for model_id in &preferred {
            if let Some(registered) = registry.get(model_id) {
                if registered.is_available() {
                    self.record_family(model_id);
                    return Ok(Arc::clone(&registered.engine));
                }
            }
        }

        // Fall back to any suitable model
        let suitable: Vec<_> = models
            .values()
            .filter(|m| m.suitable_tasks.contains(&task))
            .collect();

        for model in suitable {
            if let Some(registered) = registry.get(&model.model_id) {
                if registered.is_available() {
                    self.record_family(&model.model_id);
                    return Ok(Arc::clone(&registered.engine));
                }
            }
        }

        // Final fallback to any available model
        let all = registry.all();
        for model in all {
            if model.is_available() {
                return Ok(Arc::clone(&model.engine));
            }
        }

        Err(infernum_core::Error::ModelNotFound {
            model_id: format!("no model available for task {:?}", task),
        })
    }

    /// Routes validation to a different model family than recent requests.
    fn route_diverse_validation(
        &self,
        registry: &ModelRegistry,
        preferred: &[String],
    ) -> Result<Arc<Engine>> {
        let recent = self.recent_families.read();
        let models = self.models.read();

        // Find a model from a different family
        for model_id in preferred {
            if let Some(config) = models.get(model_id) {
                // Check if this family was used recently
                let family = &config.family;
                if !recent.contains(family) {
                    if let Some(registered) = registry.get(model_id) {
                        if registered.is_available() {
                            drop(recent);
                            self.record_family(model_id);
                            return Ok(Arc::clone(&registered.engine));
                        }
                    }
                }
            }
        }

        // If no diverse model available, fall back to any validation-capable model
        drop(recent);
        for model_id in preferred {
            if let Some(registered) = registry.get(model_id) {
                if registered.is_available() {
                    self.record_family(model_id);
                    tracing::warn!(
                        "Validation using same model family as recent - diversity not achieved"
                    );
                    return Ok(Arc::clone(&registered.engine));
                }
            }
        }

        Err(infernum_core::Error::ModelNotFound {
            model_id: "no validation model available".to_string(),
        })
    }

    /// Records a model family as recently used.
    fn record_family(&self, model_id: &str) {
        let models = self.models.read();
        if let Some(config) = models.get(model_id) {
            let family = config.family.clone();
            drop(models);

            let mut recent = self.recent_families.write();
            recent.push(family);
            if recent.len() > self.max_recent {
                recent.remove(0);
            }
        }
    }

    /// Returns task preferences for inspection.
    pub fn task_preferences(&self) -> HashMap<TaskType, Vec<String>> {
        self.task_preferences.read().clone()
    }

    /// Sets task preferences.
    pub fn set_task_preferences(&self, prefs: HashMap<TaskType, Vec<String>>) {
        *self.task_preferences.write() = prefs;
    }

    /// Returns registered federated models.
    pub fn federated_models(&self) -> Vec<FederatedModel> {
        self.models.read().values().cloned().collect()
    }

    /// Clears the recent family history.
    pub fn clear_recent_history(&self) {
        self.recent_families.write().clear();
    }

    /// Sets whether to enforce validation diversity.
    pub fn set_validation_diversity(&mut self, enforce: bool) {
        self.enforce_validation_diversity = enforce;
    }

    /// Returns detailed status for a specific federated model.
    pub fn get_model_status(
        &self,
        model_id: &str,
        registry: &ModelRegistry,
    ) -> Option<ModelStatus> {
        let models = self.models.read();
        let federated = models.get(model_id)?;
        let registered: Arc<RegisteredModel> = registry.get(model_id)?;

        Some(ModelStatus::from_registered(
            model_id,
            federated.family.clone(),
            &registered,
        ))
    }

    /// Returns status for all federated models.
    pub fn get_all_model_status(&self, registry: &ModelRegistry) -> Vec<ModelStatus> {
        let models = self.models.read();
        models
            .iter()
            .filter_map(|(id, federated)| {
                registry.get(id).map(|registered: Arc<RegisteredModel>| {
                    ModelStatus::from_registered(id, federated.family.clone(), &registered)
                })
            })
            .collect()
    }
}

impl Default for FederationRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// External API provider configuration.
#[derive(Debug, Clone)]
pub struct ExternalProvider {
    /// Provider name.
    pub name: String,
    /// API base URL.
    pub base_url: String,
    /// API key environment variable.
    pub api_key_env: String,
    /// Available models.
    pub models: Vec<String>,
    /// Whether provider is enabled.
    pub enabled: bool,
    /// Rate limit (requests per minute).
    pub rate_limit_rpm: Option<u32>,
}

impl ExternalProvider {
    /// Creates an Anthropic provider configuration.
    pub fn anthropic() -> Self {
        Self {
            name: "anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_key_env: "ANTHROPIC_API_KEY".to_string(),
            models: vec!["claude-opus-4".to_string(), "claude-sonnet-4".to_string()],
            enabled: false,
            rate_limit_rpm: Some(60),
        }
    }

    /// Creates an OpenAI provider configuration.
    pub fn openai() -> Self {
        Self {
            name: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key_env: "OPENAI_API_KEY".to_string(),
            models: vec!["gpt-4o".to_string(), "gpt-4-turbo".to_string()],
            enabled: false,
            rate_limit_rpm: Some(500),
        }
    }

    /// Creates a Google Gemini provider configuration.
    pub fn google() -> Self {
        Self {
            name: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1".to_string(),
            api_key_env: "GOOGLE_API_KEY".to_string(),
            models: vec!["gemini-pro".to_string(), "gemini-2.0-flash".to_string()],
            enabled: false,
            rate_limit_rpm: Some(60),
        }
    }

    /// Checks if the provider is configured (API key exists).
    pub fn is_configured(&self) -> bool {
        std::env::var(&self.api_key_env).is_ok()
    }

    /// Enables the provider.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }
}

/// Status information for a federated model.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    /// Model identifier.
    pub model_id: String,
    /// Model family.
    pub family: ModelFamily,
    /// Whether the model is currently available.
    pub is_available: bool,
    /// Number of active requests.
    pub active_requests: u32,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// P99 latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Total request count.
    pub request_count: u64,
}

impl ModelStatus {
    /// Creates status from a registered model.
    pub fn from_registered(
        model_id: &str,
        family: ModelFamily,
        registered: &RegisteredModel,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            family,
            is_available: registered.is_available(),
            active_requests: registered.active_requests.load(Ordering::Relaxed),
            avg_latency_ms: registered.latency_stats.average_latency_ms(),
            p99_latency_ms: registered.latency_stats.p99_latency_ms(),
            request_count: registered.latency_stats.request_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_inference() {
        // "code" matches CodeGeneration pattern
        let request = GenerateRequest::new("Please analyze this code structure");
        assert_eq!(TaskType::infer(&request), TaskType::CodeGeneration);

        let request = GenerateRequest::new("Implement a function that converts Rust to Sigil");
        assert_eq!(TaskType::infer(&request), TaskType::CodeGeneration);

        let request = GenerateRequest::new("Validate the correctness of this conversion");
        assert_eq!(TaskType::infer(&request), TaskType::Validation);

        let request = GenerateRequest::new("Summarize the changes in this checkpoint");
        assert_eq!(TaskType::infer(&request), TaskType::Summarization);

        // Pure reasoning without code keywords
        let request = GenerateRequest::new("Analyze the design patterns in this architecture");
        assert_eq!(TaskType::infer(&request), TaskType::Reasoning);
    }

    #[test]
    fn test_model_family_detection() {
        assert_eq!(
            ModelFamily::from_model_id("claude-opus-4"),
            ModelFamily::Claude
        );
        assert_eq!(ModelFamily::from_model_id("gpt-4o"), ModelFamily::Gpt);
        assert_eq!(
            ModelFamily::from_model_id("deepseek-coder-33b"),
            ModelFamily::DeepSeek
        );
        assert_eq!(
            ModelFamily::from_model_id("codellama-34b"),
            ModelFamily::CodeSpecialist
        );
        assert_eq!(
            ModelFamily::from_model_id("llama-3.2-70b"),
            ModelFamily::Llama
        );
    }

    #[test]
    fn test_federation_router_creation() {
        let router = FederationRouter::jormungandr();
        let prefs = router.task_preferences();

        assert!(prefs.contains_key(&TaskType::Reasoning));
        assert!(prefs.contains_key(&TaskType::CodeGeneration));
        assert!(prefs.contains_key(&TaskType::Validation));
    }

    #[test]
    fn test_federated_model_builder() {
        let model = FederatedModel::new("sigil-specialist-v1")
            .with_family(ModelFamily::SigilSpecialist)
            .with_tasks(vec![TaskType::CodeGeneration, TaskType::Validation])
            .with_priority(90)
            .with_cost(0.001, 0.003);

        assert_eq!(model.family, ModelFamily::SigilSpecialist);
        assert_eq!(model.priority, 90);
        assert!(model.suitable_tasks.contains(&TaskType::CodeGeneration));
    }
}
