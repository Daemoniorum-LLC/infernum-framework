//! Request routing logic.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use abaddon::Engine;
use infernum_core::{GenerateRequest, PromptInput, Result};
use parking_lot::Mutex;

use crate::registry::{ModelRegistry, RegisteredModel};

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
    /// Weighted routing based on multiple factors.
    Weighted {
        /// Weight for latency factor (0.0-1.0).
        latency_weight: f32,
        /// Weight for cost factor (0.0-1.0).
        cost_weight: f32,
        /// Weight for load factor (0.0-1.0).
        load_weight: f32,
    },
    /// Route based on model capabilities.
    CapabilityBased {
        /// Required capability.
        required_capability: String,
        /// Fallback strategy if no capable model found.
        fallback: Box<RoutingStrategy>,
    },
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
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
        self.route_by_strategy(&self.strategy, request, registry)
    }

    /// Routes using a specific strategy.
    fn route_by_strategy(
        &self,
        strategy: &RoutingStrategy,
        request: &GenerateRequest,
        registry: &ModelRegistry,
    ) -> Result<Arc<Engine>> {
        match strategy {
            RoutingStrategy::RoundRobin => self.round_robin(registry),
            RoutingStrategy::LeastConnections => self.least_connections(registry),
            RoutingStrategy::LatencyOptimized { target_p99_ms } => {
                self.latency_optimized(registry, *target_p99_ms)
            }
            RoutingStrategy::CostOptimized { max_cost_per_token } => {
                self.cost_optimized(registry, request, *max_cost_per_token)
            }
            RoutingStrategy::Weighted {
                latency_weight,
                cost_weight,
                load_weight,
            } => self.weighted_routing(
                registry,
                request,
                *latency_weight,
                *cost_weight,
                *load_weight,
            ),
            RoutingStrategy::CapabilityBased {
                required_capability,
                fallback,
            } => {
                let capable = registry.find_by_capability(required_capability);
                if capable.is_empty() {
                    self.route_by_strategy(fallback, request, registry)
                } else {
                    // Among capable models, use least connections
                    self.select_least_loaded(&capable)
                }
            }
        }
    }

    /// Round-robin routing.
    fn round_robin(&self, registry: &ModelRegistry) -> Result<Arc<Engine>> {
        let models = registry.list_available();
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
        let models = registry.all();
        let available: Vec<_> = models.iter().filter(|m| m.is_available()).collect();

        if available.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        let best = available
            .iter()
            .min_by_key(|m| m.active_requests.load(Ordering::Relaxed))
            .unwrap();

        Ok(Arc::clone(&best.engine))
    }

    /// Latency-optimized routing.
    ///
    /// Routes to the model most likely to meet the target P99 latency.
    /// Uses historical latency data to predict performance.
    fn latency_optimized(&self, registry: &ModelRegistry, target_p99_ms: u32) -> Result<Arc<Engine>> {
        let models = registry.all();
        let available: Vec<_> = models.iter().filter(|m| m.is_available()).collect();

        if available.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        // Score models based on how well they meet the latency target
        let target = target_p99_ms as f64;
        let mut best_model: Option<&Arc<RegisteredModel>> = None;
        let mut best_score = f64::MAX;

        for model in &available {
            let p99 = model.latency_stats.p99_latency_ms();
            let load = model.load_factor();

            // If no history, assume moderate latency
            let effective_p99 = if p99 == 0.0 { target * 0.8 } else { p99 };

            // Penalize models under heavy load (latency likely to increase)
            let load_penalty = 1.0 + (load * 0.5);
            let adjusted_p99 = effective_p99 * load_penalty;

            // Score: distance from target (lower is better)
            // Models already below target get bonus
            let score = if adjusted_p99 <= target {
                // Under target: small penalty for being too far under (may be underutilized)
                (target - adjusted_p99) * 0.1
            } else {
                // Over target: penalty for exceeding
                (adjusted_p99 - target) * 2.0
            };

            if score < best_score {
                best_score = score;
                best_model = Some(model);
            }
        }

        best_model
            .map(|m| Arc::clone(&m.engine))
            .ok_or_else(|| infernum_core::Error::ModelNotFound {
                model_id: "no suitable model found".to_string(),
            })
    }

    /// Cost-optimized routing.
    ///
    /// Routes to the cheapest model that can handle the request.
    fn cost_optimized(
        &self,
        registry: &ModelRegistry,
        request: &GenerateRequest,
        max_cost_per_token: f64,
    ) -> Result<Arc<Engine>> {
        let models = registry.all();
        let available: Vec<_> = models.iter().filter(|m| m.is_available()).collect();

        if available.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        // Estimate token counts from request
        let estimated_input_tokens = self.estimate_input_tokens(request);
        let estimated_output_tokens = request.sampling.max_tokens;

        // Find cheapest model within budget
        let mut cheapest_model: Option<&Arc<RegisteredModel>> = None;
        let mut cheapest_cost = f64::MAX;

        for model in &available {
            let cost = model.cost.read();
            let estimated_cost = cost.calculate(estimated_input_tokens, estimated_output_tokens);

            // Calculate cost per token
            let total_tokens = (estimated_input_tokens + estimated_output_tokens) as f64;
            let cost_per_token = if total_tokens > 0.0 {
                estimated_cost / total_tokens
            } else {
                0.0
            };

            // Check if within budget
            if cost_per_token <= max_cost_per_token || max_cost_per_token == 0.0 {
                if estimated_cost < cheapest_cost {
                    cheapest_cost = estimated_cost;
                    cheapest_model = Some(model);
                }
            }
        }

        // If no model within budget, fall back to cheapest overall
        if cheapest_model.is_none() {
            cheapest_model = available
                .iter()
                .min_by(|a, b| {
                    let cost_a = a.cost.read().calculate(estimated_input_tokens, estimated_output_tokens);
                    let cost_b = b.cost.read().calculate(estimated_input_tokens, estimated_output_tokens);
                    cost_a.partial_cmp(&cost_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();
        }

        cheapest_model
            .map(|m| Arc::clone(&m.engine))
            .ok_or_else(|| infernum_core::Error::ModelNotFound {
                model_id: "no suitable model found".to_string(),
            })
    }

    /// Weighted multi-factor routing.
    ///
    /// Combines latency, cost, and load factors with configurable weights.
    fn weighted_routing(
        &self,
        registry: &ModelRegistry,
        request: &GenerateRequest,
        latency_weight: f32,
        cost_weight: f32,
        load_weight: f32,
    ) -> Result<Arc<Engine>> {
        let models = registry.all();
        let available: Vec<_> = models.iter().filter(|m| m.is_available()).collect();

        if available.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no models available".to_string(),
            });
        }

        // Normalize weights
        let total_weight = latency_weight + cost_weight + load_weight;
        let (lat_w, cost_w, load_w) = if total_weight > 0.0 {
            (
                latency_weight / total_weight,
                cost_weight / total_weight,
                load_weight / total_weight,
            )
        } else {
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        };

        // Estimate tokens for cost calculation
        let estimated_input = self.estimate_input_tokens(request);
        let estimated_output = request.sampling.max_tokens;

        // Compute stats for normalization
        let latencies: Vec<f64> = available
            .iter()
            .map(|m| m.latency_stats.average_latency_ms().max(1.0))
            .collect();
        let costs: Vec<f64> = available
            .iter()
            .map(|m| m.cost.read().calculate(estimated_input, estimated_output).max(0.0001))
            .collect();
        let loads: Vec<f64> = available.iter().map(|m| m.load_factor()).collect();

        let max_latency = latencies.iter().cloned().fold(1.0_f64, f64::max);
        let max_cost = costs.iter().cloned().fold(0.0001_f64, f64::max);
        let max_load = loads.iter().cloned().fold(0.1_f64, f64::max);

        // Score each model (lower is better)
        let mut best_model: Option<&Arc<RegisteredModel>> = None;
        let mut best_score = f64::MAX;

        for (i, model) in available.iter().enumerate() {
            let norm_latency = latencies[i] / max_latency;
            let norm_cost = costs[i] / max_cost;
            let norm_load = loads[i] / max_load;

            let score = (norm_latency as f32 * lat_w)
                + (norm_cost as f32 * cost_w)
                + (norm_load as f32 * load_w);

            if (score as f64) < best_score {
                best_score = score as f64;
                best_model = Some(model);
            }
        }

        best_model
            .map(|m| Arc::clone(&m.engine))
            .ok_or_else(|| infernum_core::Error::ModelNotFound {
                model_id: "no suitable model found".to_string(),
            })
    }

    /// Selects the least loaded model from a set.
    fn select_least_loaded(&self, models: &[Arc<RegisteredModel>]) -> Result<Arc<Engine>> {
        let available: Vec<_> = models.iter().filter(|m| m.is_available()).collect();

        if available.is_empty() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: "no available models with required capability".to_string(),
            });
        }

        let best = available
            .iter()
            .min_by_key(|m| m.active_requests.load(Ordering::Relaxed))
            .unwrap();

        Ok(Arc::clone(&best.engine))
    }

    /// Estimates input token count from request.
    fn estimate_input_tokens(&self, request: &GenerateRequest) -> u32 {
        // Rough estimation: ~4 characters per token
        let content_len: usize = match &request.prompt {
            PromptInput::Text(s) => s.len(),
            PromptInput::Messages(messages) => {
                messages.iter().map(|m| m.content.len()).sum()
            }
            PromptInput::Tokens(tokens) => tokens.len() * 4, // Already tokenized
        };

        (content_len / 4).max(1) as u32
    }

    /// Returns the routing strategy.
    #[must_use]
    pub fn strategy(&self) -> &RoutingStrategy {
        &self.strategy
    }

    /// Sets a new routing strategy.
    pub fn set_strategy(&mut self, strategy: RoutingStrategy) {
        self.strategy = strategy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_strategy_default() {
        let strategy = RoutingStrategy::default();
        assert!(matches!(strategy, RoutingStrategy::RoundRobin));
    }

    #[test]
    fn test_estimate_input_tokens() {
        let router = RequestRouter::new(RoutingStrategy::RoundRobin);

        let request = GenerateRequest::new("Hello, world! This is a test.");

        let tokens = router.estimate_input_tokens(&request);
        assert!(tokens > 0);
        assert!(tokens < 100); // Should be reasonable
    }
}
