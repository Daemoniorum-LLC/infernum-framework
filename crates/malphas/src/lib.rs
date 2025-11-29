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

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod registry;
pub mod router;
pub mod scheduler;

pub use registry::{ModelRegistry, RegisteredModel};
pub use router::{RequestRouter, RoutingStrategy};
pub use scheduler::BatchScheduler;

use std::sync::Arc;

use abaddon::Engine;
use infernum_core::{GenerateRequest, GenerateResponse, Result};

/// The main orchestration service.
pub struct Malphas {
    registry: Arc<ModelRegistry>,
    router: Arc<RequestRouter>,
}

impl Malphas {
    /// Creates a new orchestration service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(RoutingStrategy::RoundRobin)),
        }
    }

    /// Creates with a custom routing strategy.
    #[must_use]
    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self {
            registry: Arc::new(ModelRegistry::new()),
            router: Arc::new(RequestRouter::new(strategy)),
        }
    }

    /// Registers a model with the orchestrator.
    pub fn register(&self, model_id: impl Into<String>, engine: Arc<Engine>) {
        self.registry.register(model_id, engine);
    }

    /// Routes and executes a generation request.
    ///
    /// # Errors
    ///
    /// Returns an error if routing fails or inference fails.
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let engine = self.router.route(&request, &self.registry)?;
        engine.generate(request).await
    }

    /// Returns the model registry.
    #[must_use]
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }
}

impl Default for Malphas {
    fn default() -> Self {
        Self::new()
    }
}
