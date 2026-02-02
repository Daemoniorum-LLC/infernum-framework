//! # Malphas
//!
//! *"The Builder constructs order from chaos"*
//!
//! Malphas is the orchestration layer for the Infernum ecosystem,
//! providing intelligent routing, load balancing, and model lifecycle management.
//!
//! ## Features
//!
//! - **Smart Routing**: Route requests based on model capabilities, load, and SLOs
//! - **Model Registry**: Centralized model management and discovery
//! - **Auto-Scaling**: Dynamic scaling based on demand
//! - **Health Monitoring**: Continuous health checks and failover
//! - **Automatic Failover**: Retry failed requests on healthy alternatives

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod batched;
pub mod experiments;
pub mod federation;
pub mod health;
pub mod legion_speculative;
pub mod registry;
pub mod router;
pub mod scheduler;
pub mod spectral_blend;
pub mod tenant;
pub mod thermal;

pub use batched::{BatchedInferenceService, BatchedInferenceServiceBuilder, BatchedServiceConfig, BatchedServiceStats};
pub use experiments::{
    Experiment, ExperimentId, ExperimentManager, ExperimentStatus, ExperimentSummary, Variant,
    VariantMetrics, VariantSummary,
};
pub use federation::{
    ExternalProvider, FederatedModel, FederationRouter, ModelFamily, TaskType,
};
pub use health::{HealthConfig, HealthMonitor, HealthStatus, HealthSummary, ModelHealthState};
pub use registry::{LatencyStats, ModelCost, ModelRegistry, RegisteredModel};
pub use router::{RequestRouter, RoutingStrategy};
pub use scheduler::{BatchScheduler, Priority, SchedulerConfig, SchedulerStats};
pub use tenant::{
    AggregateUsageStats, QuotaCheckResult, QuotaDenialReason, QuotaLimits, Tenant, TenantContext,
    TenantId, TenantManager, TenantUsageStats,
};
pub use thermal::{PowerProfile, ThermalManager, ThermalState, ThermalThresholds};
pub use legion_speculative::{
    LegionSpeculativeDecoder, LegionSpeculativeConfig, LegionSpeculativeStats, BandStats,
};
pub use spectral_blend::{
    SpectralBlendEngine, SpectralBlendEngineBuilder, SpectralBlendConfig, BlendModelConfig,
    SpectralBlendEngineStats, BlendPreset, BlendManager, SpectralBlendEngineError,
};

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use abaddon::{Engine, InferenceEngine};
use infernum_core::{GenerateRequest, GenerateResponse, Result};

/// Configuration for automatic failover behavior.
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// Whether to only failover to healthy models.
    pub require_healthy: bool,
    /// Whether to exclude the failed model from retries.
    pub exclude_failed: bool,
    /// Whether to use exponential backoff for retries.
    pub exponential_backoff: bool,
    /// Maximum backoff delay.
    pub max_backoff: Duration,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_delay: Duration::from_millis(100),
            require_healthy: true,
            exclude_failed: true,
            exponential_backoff: true,
            max_backoff: Duration::from_secs(5),
        }
    }
}

impl FailoverConfig {
    /// Creates a config optimized for production with more retries.
    #[must_use]
    pub fn production() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(50),
            require_healthy: true,
            exclude_failed: true,
            exponential_backoff: true,
            max_backoff: Duration::from_secs(10),
        }
    }

    /// Creates a config with no failover (fail fast).
    #[must_use]
    pub fn fail_fast() -> Self {
        Self {
            max_retries: 0,
            retry_delay: Duration::ZERO,
            require_healthy: false,
            exclude_failed: false,
            exponential_backoff: false,
            max_backoff: Duration::ZERO,
        }
    }

    /// Creates a config with aggressive failover.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            retry_delay: Duration::from_millis(10),
            require_healthy: false,
            exclude_failed: true,
            exponential_backoff: false,
            max_backoff: Duration::from_millis(100),
        }
    }

    /// Sets the maximum retries.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sets the retry delay.
    #[must_use]
    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    /// Calculates delay for a given attempt (with optional exponential backoff).
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if !self.exponential_backoff {
            return self.retry_delay;
        }

        let multiplier = 2_u32.saturating_pow(attempt);
        let delay = self.retry_delay.saturating_mul(multiplier);
        delay.min(self.max_backoff)
    }
}

/// Result of a failover attempt.
#[derive(Debug)]
pub struct FailoverResult {
    /// The final response (if successful).
    pub response: Option<GenerateResponse>,
    /// The final error (if all attempts failed).
    pub error: Option<infernum_core::Error>,
    /// Number of attempts made.
    pub attempts: u32,
    /// Model IDs that were tried.
    pub tried_models: Vec<String>,
    /// Model ID that succeeded (if any).
    pub successful_model: Option<String>,
}

/// The main orchestration service.
pub struct Malphas {
    registry: Arc<ModelRegistry>,
    router: Arc<RequestRouter>,
    scheduler: Arc<BatchScheduler>,
    thermal: Arc<ThermalManager>,
    health: Arc<HealthMonitor>,
    failover: FailoverConfig,
}

impl Malphas {
    /// Creates a new orchestration service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(RoutingStrategy::RoundRobin)),
            scheduler: Arc::new(BatchScheduler::default()),
            thermal: Arc::new(ThermalManager::new()),
            health: Arc::new(HealthMonitor::default_config()),
            failover: FailoverConfig::default(),
        }
    }

    /// Creates with a custom routing strategy.
    #[must_use]
    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(strategy)),
            scheduler: Arc::new(BatchScheduler::default()),
            thermal: Arc::new(ThermalManager::new()),
            health: Arc::new(HealthMonitor::default_config()),
            failover: FailoverConfig::default(),
        }
    }

    /// Creates with full configuration.
    #[must_use]
    pub fn with_config(
        strategy: RoutingStrategy,
        scheduler_config: SchedulerConfig,
        thermal: ThermalManager,
    ) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(strategy)),
            scheduler: Arc::new(BatchScheduler::new(scheduler_config)),
            thermal: Arc::new(thermal),
            health: Arc::new(HealthMonitor::default_config()),
            failover: FailoverConfig::default(),
        }
    }

    /// Creates with full configuration including health monitoring.
    #[must_use]
    pub fn with_full_config(
        strategy: RoutingStrategy,
        scheduler_config: SchedulerConfig,
        thermal: ThermalManager,
        health_config: HealthConfig,
    ) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(strategy)),
            scheduler: Arc::new(BatchScheduler::new(scheduler_config)),
            thermal: Arc::new(thermal),
            health: Arc::new(HealthMonitor::new(health_config)),
            failover: FailoverConfig::default(),
        }
    }

    /// Creates with all configuration including failover.
    #[must_use]
    pub fn with_failover_config(
        strategy: RoutingStrategy,
        scheduler_config: SchedulerConfig,
        thermal: ThermalManager,
        health_config: HealthConfig,
        failover_config: FailoverConfig,
    ) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(strategy)),
            scheduler: Arc::new(BatchScheduler::new(scheduler_config)),
            thermal: Arc::new(thermal),
            health: Arc::new(HealthMonitor::new(health_config)),
            failover: failover_config,
        }
    }

    /// Creates a workstation-optimized orchestrator with thermal management.
    #[must_use]
    pub fn workstation() -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(RoutingStrategy::LeastConnections)),
            scheduler: Arc::new(BatchScheduler::default()),
            thermal: Arc::new(ThermalManager::workstation()),
            health: Arc::new(HealthMonitor::default_config()),
            failover: FailoverConfig::default(),
        }
    }

    /// Creates a production-optimized orchestrator.
    #[must_use]
    pub fn production() -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(RoutingStrategy::LeastConnections)),
            scheduler: Arc::new(BatchScheduler::default()),
            thermal: Arc::new(ThermalManager::workstation()),
            health: Arc::new(HealthMonitor::new(HealthConfig::production())),
            failover: FailoverConfig::production(),
        }
    }

    /// Registers a model with the orchestrator.
    pub fn register(&self, model_id: impl Into<String>, engine: Arc<Engine>) {
        self.registry.register(model_id, engine);
    }

    /// Enqueues a request with thermal-aware batch scheduling.
    ///
    /// Returns `false` if the queue is full.
    pub fn enqueue(&self, request: GenerateRequest, priority: Priority) -> bool {
        self.scheduler.enqueue(request, priority)
    }

    /// Dequeues a batch with thermal-aware sizing.
    ///
    /// The batch size is automatically reduced based on current thermal state.
    pub async fn dequeue_thermal_batch(&self) -> Vec<GenerateRequest> {
        let max_batch = self.scheduler.config().max_batch_size as u32;
        let effective_batch = self.thermal.recommended_batch_size(max_batch).await;
        self.scheduler.dequeue_batch_with_limit(effective_batch as usize)
    }

    /// Routes and executes a generation request with health tracking.
    ///
    /// # Errors
    ///
    /// Returns an error if routing fails or inference fails.
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let engine = self.router.route(&request, &self.registry)?;
        let model_id: String = request
            .model
            .as_ref()
            .map(|m| m.0.clone())
            .unwrap_or_else(|| "unknown".to_string());

        match engine.generate(request).await {
            Ok(response) => {
                self.health.record_success(&model_id);
                Ok(response)
            }
            Err(e) => {
                self.health.record_failure(&model_id, e.to_string());
                Err(e)
            }
        }
    }

    /// Routes and executes a batch of generation requests with thermal awareness.
    ///
    /// This method updates thermal readings before processing.
    ///
    /// # Errors
    ///
    /// Returns errors for any failed requests in the batch.
    pub async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Vec<Result<GenerateResponse>> {
        // Update thermal state before processing
        self.thermal.update_temperatures().await;

        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let result = self.generate(request).await;
            results.push(result);
        }
        results
    }

    /// Starts background thermal monitoring.
    ///
    /// The monitoring interval determines how frequently temperatures are checked.
    pub fn start_thermal_monitoring(&self, interval: Duration) {
        Arc::clone(&self.thermal).start_monitoring(interval);
    }

    /// Stops background thermal monitoring.
    pub fn stop_thermal_monitoring(&self) {
        self.thermal.stop_monitoring();
    }

    /// Sets the power profile for thermal management.
    pub async fn set_power_profile(&self, profile: PowerProfile) {
        self.thermal.set_profile(profile).await;
    }

    /// Returns the current thermal state.
    pub async fn thermal_state(&self) -> ThermalState {
        self.thermal.state().await
    }

    /// Returns the current throttle factor (0.0-1.0).
    pub async fn throttle_factor(&self) -> f32 {
        self.thermal.throttle_factor().await
    }

    /// Returns the model registry.
    #[must_use]
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// Returns the batch scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &BatchScheduler {
        &self.scheduler
    }

    /// Returns the thermal manager.
    #[must_use]
    pub fn thermal(&self) -> &ThermalManager {
        &self.thermal
    }

    /// Returns scheduler statistics.
    #[must_use]
    pub fn scheduler_stats(&self) -> SchedulerStats {
        self.scheduler.stats()
    }

    /// Returns the health monitor.
    #[must_use]
    pub fn health(&self) -> &HealthMonitor {
        &self.health
    }

    /// Starts background health monitoring.
    ///
    /// This will periodically check model health based on latency and error rates.
    pub fn start_health_monitoring(&self) {
        Arc::clone(&self.health).start_monitoring(Arc::clone(&self.registry));
    }

    /// Stops background health monitoring.
    pub fn stop_health_monitoring(&self) {
        self.health.stop_monitoring();
    }

    /// Returns the health status of a specific model.
    #[must_use]
    pub fn model_health_status(&self, model_id: &str) -> HealthStatus {
        self.health.status(model_id)
    }

    /// Returns a summary of all model health states.
    #[must_use]
    pub fn health_summary(&self) -> HealthSummary {
        self.health.summary()
    }

    /// Starts all background monitoring (thermal + health).
    pub fn start_all_monitoring(&self, thermal_interval: Duration) {
        self.start_thermal_monitoring(thermal_interval);
        self.start_health_monitoring();
    }

    /// Stops all background monitoring.
    pub fn stop_all_monitoring(&self) {
        self.stop_thermal_monitoring();
        self.stop_health_monitoring();
    }

    /// Returns the failover configuration.
    #[must_use]
    pub fn failover_config(&self) -> &FailoverConfig {
        &self.failover
    }

    /// Sets a new failover configuration.
    pub fn set_failover_config(&mut self, config: FailoverConfig) {
        self.failover = config;
    }

    /// Routes and executes a generation request with automatic failover.
    ///
    /// If the initial request fails, this will automatically retry with healthy
    /// alternative models according to the failover configuration.
    ///
    /// # Returns
    ///
    /// Returns a `FailoverResult` containing either the successful response
    /// or the final error after all retry attempts.
    pub async fn generate_with_failover(&self, request: GenerateRequest) -> FailoverResult {
        let mut tried_models = Vec::new();

        // Try the initial routing
        let initial_model_id: String = request
            .model
            .as_ref()
            .map(|m| m.0.clone())
            .unwrap_or_else(|| "auto".to_string());

        // First attempt with normal routing
        let mut last_error = match self.try_generate(&request, &initial_model_id, &mut tried_models).await {
            Ok(response) => {
                let successful = tried_models.last().cloned();
                return FailoverResult {
                    response: Some(response),
                    error: None,
                    attempts: 1,
                    tried_models,
                    successful_model: successful,
                };
            }
            Err(e) => {
                tracing::warn!(
                    model_id = %initial_model_id,
                    error = %e,
                    "Initial request failed, attempting failover"
                );
                Some(e)
            }
        };

        // Track attempts starting from the initial failed attempt
        let mut attempts = 1;

        // Retry with failover
        for retry in 0..self.failover.max_retries {
            // Apply delay with optional exponential backoff
            let delay = self.failover.delay_for_attempt(retry);
            if delay > Duration::ZERO {
                tokio::time::sleep(delay).await;
            }

            // Find an alternative model
            let alternative = self.find_failover_model(&tried_models);
            let model_id = match alternative {
                Some(model) => {
                    let id = model.id.0.clone();
                    tracing::info!(
                        model_id = %id,
                        retry = retry + 1,
                        max_retries = self.failover.max_retries,
                        "Attempting failover to alternative model"
                    );
                    id
                }
                None => {
                    tracing::warn!("No healthy alternative models available for failover");
                    break;
                }
            };

            // Try the alternative model
            let mut failover_request = request.clone();
            failover_request.model = Some(infernum_core::ModelId(model_id.clone()));

            match self.try_generate(&failover_request, &model_id, &mut tried_models).await {
                Ok(response) => {
                    tracing::info!(
                        model_id = %model_id,
                        attempts = attempts + 1,
                        "Failover successful"
                    );
                    return FailoverResult {
                        response: Some(response),
                        error: None,
                        attempts: attempts + 1,
                        tried_models,
                        successful_model: Some(model_id),
                    };
                }
                Err(e) => {
                    tracing::warn!(
                        model_id = %model_id,
                        error = %e,
                        retry = retry + 1,
                        "Failover attempt failed"
                    );
                    last_error = Some(e);
                    attempts += 1;
                }
            }
        }

        tracing::error!(
            attempts = attempts,
            tried_models = ?tried_models,
            "All failover attempts exhausted"
        );

        FailoverResult {
            response: None,
            error: last_error,
            attempts,
            tried_models,
            successful_model: None,
        }
    }

    /// Attempts to generate with a specific model, tracking health.
    async fn try_generate(
        &self,
        request: &GenerateRequest,
        model_id: &str,
        tried_models: &mut Vec<String>,
    ) -> Result<GenerateResponse> {
        tried_models.push(model_id.to_string());

        let engine = self.router.route(request, &self.registry)?;

        match engine.generate(request.clone()).await {
            Ok(response) => {
                self.health.record_success(model_id);
                Ok(response)
            }
            Err(e) => {
                self.health.record_failure(model_id, e.to_string());
                Err(e)
            }
        }
    }

    /// Finds an alternative model for failover, excluding already tried models.
    fn find_failover_model(&self, excluded: &[String]) -> Option<Arc<RegisteredModel>> {
        let all_models = self.registry.all();

        // Filter based on failover config
        let candidates: Vec<_> = all_models
            .into_iter()
            .filter(|model| {
                let model_id = &model.id.0;

                // Exclude already tried models if configured
                if self.failover.exclude_failed && excluded.contains(model_id) {
                    return false;
                }

                // Check health status if required
                if self.failover.require_healthy {
                    let status = self.health.status(model_id);
                    if !status.is_routable() {
                        return false;
                    }
                }

                // Must be available (not overloaded)
                model.is_available()
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Select the best candidate based on:
        // 1. Health status (healthy > degraded)
        // 2. Load (fewer active requests)
        // 3. Latency (lower is better)
        candidates
            .into_iter()
            .min_by(|a, b| {
                // Compare health status first
                let health_a = self.health.status(&a.id.0);
                let health_b = self.health.status(&b.id.0);
                let health_cmp = match (health_a, health_b) {
                    (HealthStatus::Healthy, HealthStatus::Degraded) => std::cmp::Ordering::Less,
                    (HealthStatus::Degraded, HealthStatus::Healthy) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                };

                if health_cmp != std::cmp::Ordering::Equal {
                    return health_cmp;
                }

                // Compare load
                let load_a = a.active_requests.load(Ordering::Relaxed);
                let load_b = b.active_requests.load(Ordering::Relaxed);
                let load_cmp = load_a.cmp(&load_b);

                if load_cmp != std::cmp::Ordering::Equal {
                    return load_cmp;
                }

                // Compare latency
                let latency_a = a.latency_stats.average_latency_ms();
                let latency_b = b.latency_stats.average_latency_ms();
                latency_a
                    .partial_cmp(&latency_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Generates with failover using a custom failover configuration.
    ///
    /// This allows overriding the default failover behavior for specific requests.
    pub async fn generate_with_custom_failover(
        &self,
        request: GenerateRequest,
        config: &FailoverConfig,
    ) -> FailoverResult {
        let mut tried_models = Vec::new();

        let initial_model_id: String = request
            .model
            .as_ref()
            .map(|m| m.0.clone())
            .unwrap_or_else(|| "auto".to_string());

        // First attempt
        let mut last_error = match self.try_generate(&request, &initial_model_id, &mut tried_models).await {
            Ok(response) => {
                let successful = tried_models.last().cloned();
                return FailoverResult {
                    response: Some(response),
                    error: None,
                    attempts: 1,
                    tried_models,
                    successful_model: successful,
                };
            }
            Err(e) => Some(e),
        };

        // Track attempts starting from the initial failed attempt
        let mut attempts = 1;

        // Retry with custom config
        for retry in 0..config.max_retries {
            let delay = config.delay_for_attempt(retry);
            if delay > Duration::ZERO {
                tokio::time::sleep(delay).await;
            }

            let alternative = self.find_failover_model_with_config(&tried_models, config);
            let model_id = match alternative {
                Some(model) => model.id.0.clone(),
                None => break,
            };

            let mut failover_request = request.clone();
            failover_request.model = Some(infernum_core::ModelId(model_id.clone()));

            match self.try_generate(&failover_request, &model_id, &mut tried_models).await {
                Ok(response) => {
                    return FailoverResult {
                        response: Some(response),
                        error: None,
                        attempts: attempts + 1,
                        tried_models,
                        successful_model: Some(model_id),
                    };
                }
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                }
            }
        }

        FailoverResult {
            response: None,
            error: last_error,
            attempts,
            tried_models,
            successful_model: None,
        }
    }

    /// Finds a failover model using a custom configuration.
    fn find_failover_model_with_config(
        &self,
        excluded: &[String],
        config: &FailoverConfig,
    ) -> Option<Arc<RegisteredModel>> {
        let all_models = self.registry.all();

        let candidates: Vec<_> = all_models
            .into_iter()
            .filter(|model| {
                let model_id = &model.id.0;

                if config.exclude_failed && excluded.contains(model_id) {
                    return false;
                }

                if config.require_healthy {
                    let status = self.health.status(model_id);
                    if !status.is_routable() {
                        return false;
                    }
                }

                model.is_available()
            })
            .collect();

        candidates
            .into_iter()
            .min_by_key(|m| m.active_requests.load(Ordering::Relaxed))
    }
}

impl Default for Malphas {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === FailoverConfig Default Tests ===

    #[test]
    fn test_failover_config_default() {
        let config = FailoverConfig::default();

        assert_eq!(config.max_retries, 2);
        assert_eq!(config.retry_delay, Duration::from_millis(100));
        assert!(config.require_healthy);
        assert!(config.exclude_failed);
        assert!(config.exponential_backoff);
        assert_eq!(config.max_backoff, Duration::from_secs(5));
    }

    #[test]
    fn test_failover_config_production() {
        let config = FailoverConfig::production();

        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_delay, Duration::from_millis(50));
        assert!(config.require_healthy);
        assert!(config.exclude_failed);
        assert!(config.exponential_backoff);
        assert_eq!(config.max_backoff, Duration::from_secs(10));
    }

    #[test]
    fn test_failover_config_fail_fast() {
        let config = FailoverConfig::fail_fast();

        assert_eq!(config.max_retries, 0);
        assert_eq!(config.retry_delay, Duration::ZERO);
        assert!(!config.require_healthy);
        assert!(!config.exclude_failed);
        assert!(!config.exponential_backoff);
        assert_eq!(config.max_backoff, Duration::ZERO);
    }

    #[test]
    fn test_failover_config_aggressive() {
        let config = FailoverConfig::aggressive();

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.retry_delay, Duration::from_millis(10));
        assert!(!config.require_healthy);
        assert!(config.exclude_failed);
        assert!(!config.exponential_backoff);
        assert_eq!(config.max_backoff, Duration::from_millis(100));
    }

    // === FailoverConfig Builder Tests ===

    #[test]
    fn test_failover_config_with_max_retries() {
        let config = FailoverConfig::default().with_max_retries(5);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_failover_config_with_retry_delay() {
        let config = FailoverConfig::default().with_retry_delay(Duration::from_secs(1));
        assert_eq!(config.retry_delay, Duration::from_secs(1));
    }

    #[test]
    fn test_failover_config_builder_chain() {
        let config = FailoverConfig::default()
            .with_max_retries(10)
            .with_retry_delay(Duration::from_millis(50));

        assert_eq!(config.max_retries, 10);
        assert_eq!(config.retry_delay, Duration::from_millis(50));
    }

    // === delay_for_attempt Tests ===

    #[test]
    fn test_delay_no_exponential_backoff() {
        let config = FailoverConfig {
            retry_delay: Duration::from_millis(100),
            exponential_backoff: false,
            max_backoff: Duration::from_secs(10),
            ..Default::default()
        };

        // Without exponential backoff, all delays are the same
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(5), Duration::from_millis(100));
    }

    #[test]
    fn test_delay_with_exponential_backoff() {
        let config = FailoverConfig {
            retry_delay: Duration::from_millis(100),
            exponential_backoff: true,
            max_backoff: Duration::from_secs(10),
            ..Default::default()
        };

        // With exponential backoff: 100 * 2^attempt
        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100)); // 100 * 1
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200)); // 100 * 2
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400)); // 100 * 4
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800)); // 100 * 8
    }

    #[test]
    fn test_delay_respects_max_backoff() {
        let config = FailoverConfig {
            retry_delay: Duration::from_millis(500),
            exponential_backoff: true,
            max_backoff: Duration::from_secs(2),
            ..Default::default()
        };

        // 500ms * 2^4 = 8000ms, but capped at 2000ms
        assert_eq!(config.delay_for_attempt(4), Duration::from_secs(2));
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(2));
    }

    #[test]
    fn test_delay_first_attempt() {
        let config = FailoverConfig::default();
        // First attempt (0) has base delay * 2^0 = base delay
        let delay = config.delay_for_attempt(0);
        assert_eq!(delay, Duration::from_millis(100));
    }

    // === FailoverResult Tests ===

    #[test]
    fn test_failover_result_success_structure() {
        use infernum_core::types::{FinishReason, ModelId, RequestId, Usage};
        use infernum_core::response::Choice;

        let result = FailoverResult {
            response: Some(GenerateResponse {
                request_id: RequestId::new(),
                model: ModelId::from("test-model"),
                choices: vec![Choice {
                    index: 0,
                    text: "test output".to_string(),
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                },
                time_to_first_token_ms: None,
                total_time_ms: None,
            }),
            error: None,
            attempts: 1,
            tried_models: vec!["model-1".to_string()],
            successful_model: Some("model-1".to_string()),
        };

        assert!(result.response.is_some());
        assert!(result.error.is_none());
        assert_eq!(result.attempts, 1);
        assert_eq!(result.tried_models.len(), 1);
        assert_eq!(result.successful_model, Some("model-1".to_string()));
    }

    #[test]
    fn test_failover_result_failure_structure() {
        let result = FailoverResult {
            response: None,
            error: Some(infernum_core::Error::ModelNotFound {
                model_id: "test-model".to_string(),
            }),
            attempts: 3,
            tried_models: vec![
                "model-1".to_string(),
                "model-2".to_string(),
                "model-3".to_string(),
            ],
            successful_model: None,
        };

        assert!(result.response.is_none());
        assert!(result.error.is_some());
        assert_eq!(result.attempts, 3);
        assert_eq!(result.tried_models.len(), 3);
        assert!(result.successful_model.is_none());
    }

    // === Malphas Constructor Tests ===

    #[test]
    fn test_malphas_new() {
        let malphas = Malphas::new();
        // Verify defaults
        assert_eq!(malphas.failover_config().max_retries, 2);
    }

    #[test]
    fn test_malphas_default() {
        let malphas = Malphas::default();
        // Default is same as new()
        assert_eq!(malphas.failover_config().max_retries, 2);
    }

    #[test]
    fn test_malphas_with_strategy() {
        let malphas = Malphas::with_strategy(RoutingStrategy::LeastConnections);
        // Should create successfully without panic
        assert!(malphas.registry().is_empty());
    }

    #[test]
    fn test_malphas_workstation() {
        let malphas = Malphas::workstation();
        // Workstation config uses default failover
        assert_eq!(malphas.failover_config().max_retries, 2);
    }

    #[test]
    fn test_malphas_production() {
        let malphas = Malphas::production();
        // Production config uses production failover
        assert_eq!(malphas.failover_config().max_retries, 3);
    }

    // === Malphas Component Access Tests ===

    #[test]
    fn test_malphas_registry_access() {
        let malphas = Malphas::new();
        let registry = malphas.registry();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_malphas_scheduler_access() {
        let malphas = Malphas::new();
        let _scheduler = malphas.scheduler();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_malphas_thermal_access() {
        let malphas = Malphas::new();
        let _thermal = malphas.thermal();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_malphas_health_access() {
        let malphas = Malphas::new();
        let _health = malphas.health();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_malphas_scheduler_stats() {
        let malphas = Malphas::new();
        let stats = malphas.scheduler_stats();
        assert_eq!(stats.current_queue_depth, 0);
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_malphas_health_summary() {
        let malphas = Malphas::new();
        let summary = malphas.health_summary();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.healthy, 0);
    }

    // === Malphas Failover Config Tests ===

    #[test]
    fn test_malphas_set_failover_config() {
        let mut malphas = Malphas::new();

        let custom_config = FailoverConfig::fail_fast();
        malphas.set_failover_config(custom_config);

        assert_eq!(malphas.failover_config().max_retries, 0);
    }

    // === FailoverConfig Clone and Debug Tests ===

    #[test]
    fn test_failover_config_clone() {
        let config = FailoverConfig::production();
        let cloned = config.clone();

        assert_eq!(cloned.max_retries, config.max_retries);
        assert_eq!(cloned.retry_delay, config.retry_delay);
    }

    #[test]
    fn test_failover_config_debug() {
        let config = FailoverConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("FailoverConfig"));
        assert!(debug_str.contains("max_retries"));
    }

    // === FailoverResult Debug Test ===

    #[test]
    fn test_failover_result_debug() {
        let result = FailoverResult {
            response: None,
            error: None,
            attempts: 0,
            tried_models: vec![],
            successful_model: None,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("FailoverResult"));
    }

    // === Priority Queue Tests ===

    #[test]
    fn test_malphas_enqueue_priority() {
        let malphas = Malphas::new();
        let request = GenerateRequest::new("test prompt");

        // Should return true when queue is not full
        let enqueued = malphas.enqueue(request, Priority::Normal);
        assert!(enqueued);
    }

    #[test]
    fn test_malphas_enqueue_high_priority() {
        let malphas = Malphas::new();
        let request = GenerateRequest::new("urgent request");

        let enqueued = malphas.enqueue(request, Priority::High);
        assert!(enqueued);
    }
}
