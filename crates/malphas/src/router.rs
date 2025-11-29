//! Request routing logic.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use abaddon::Engine;
use infernum_core::{GenerateRequest, Result};
use parking_lot::Mutex;

use crate::registry::ModelRegistry;

/// Strategy for routing requests.
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Round-robin across available backends.
    RoundRobin,
    /// Route to the backend with fewest active connections.
    LeastConnections,
    /// Route based on latency targets.
    LatencyOptimized {
        /// Target p99 latency in milliseconds.
        target_p99_ms: u32,
    },
    /// Route based on cost optimization.
    CostOptimized {
        /// Maximum cost per token in USD.
        max_cost_per_token: f64,
    },
}

/// Routes requests to appropriate backends.
pub struct RequestRouter {
    strategy: RoutingStrategy,
    round_robin_index: Mutex<usize>,
}

impl RequestRouter {
    /// Creates a new router with the given strategy.
    #[must_use]
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            round_robin_index: Mutex::new(0),
        }
    }

    /// Routes a request to an engine.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable backend is found.
    pub fn route(
        &self,
        request: &GenerateRequest,
        registry: &ModelRegistry,
    ) -> Result<Arc<Engine>> {
        // If a specific model is requested, use it
        if let Some(model_id) = &request.model {
            if let Some(registered) = registry.get(&model_id.0) {
                return Ok(Arc::clone(&registered.engine));
            }
            return Err(infernum_core::Error::ModelNotFound {
                model_id: model_id.0.clone(),
            });
        }

        // Otherwise, use routing strategy
        match &self.strategy {
            RoutingStrategy::RoundRobin => self.round_robin(registry),
            RoutingStrategy::LeastConnections => self.least_connections(registry),
            RoutingStrategy::LatencyOptimized { .. } => {
                // TODO: Implement latency-based routing
                self.round_robin(registry)
            }
            RoutingStrategy::CostOptimized { .. } => {
                // TODO: Implement cost-based routing
                self.round_robin(registry)
            }
        }
    }

    /// Round-robin routing.
    fn round_robin(&self, registry: &ModelRegistry) -> Result<Arc<Engine>> {
        let models = registry.list();
        if models.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        let mut index = self.round_robin_index.lock();
        *index = (*index + 1) % models.len();
        let model_id = &models[*index];

        registry
            .get(model_id)
            .map(|r| Arc::clone(&r.engine))
            .ok_or_else(|| infernum_core::Error::ModelNotFound {
                model_id: model_id.clone(),
            })
    }

    /// Least connections routing.
    fn least_connections(&self, registry: &ModelRegistry) -> Result<Arc<Engine>> {
        let models = registry.list();
        if models.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        let mut min_connections = u32::MAX;
        let mut best_model = None;

        for model_id in &models {
            if let Some(registered) = registry.get(model_id) {
                let connections = registered.active_requests.load(Ordering::Relaxed);
                if connections < min_connections {
                    min_connections = connections;
                    best_model = Some(registered);
                }
            }
        }

        best_model
            .map(|r| Arc::clone(&r.engine))
            .ok_or_else(|| infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            })
    }

    /// Returns the routing strategy.
    #[must_use]
    pub fn strategy(&self) -> &RoutingStrategy {
        &self.strategy
    }
}
