//! Model registry for tracking available models.

use std::sync::Arc;

use abaddon::Engine;
use dashmap::DashMap;
use infernum_core::ModelId;

/// A registered model in the system.
pub struct RegisteredModel {
    /// Model identifier.
    pub id: ModelId,
    /// The inference engine.
    pub engine: Arc<Engine>,
    /// Number of active requests.
    pub active_requests: std::sync::atomic::AtomicU32,
}

/// Registry of available models.
pub struct ModelRegistry {
    models: DashMap<String, Arc<RegisteredModel>>,
}

impl ModelRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
        }
    }

    /// Registers a model.
    pub fn register(&self, model_id: impl Into<String>, engine: Arc<Engine>) {
        let id = model_id.into();
        let registered = Arc::new(RegisteredModel {
            id: ModelId::new(&id),
            engine,
            active_requests: std::sync::atomic::AtomicU32::new(0),
        });
        self.models.insert(id, registered);
    }

    /// Unregisters a model.
    pub fn unregister(&self, model_id: &str) {
        self.models.remove(model_id);
    }

    /// Gets a model by ID.
    #[must_use]
    pub fn get(&self, model_id: &str) -> Option<Arc<RegisteredModel>> {
        self.models.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Lists all registered models.
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.models.iter().map(|r| r.key().clone()).collect()
    }

    /// Returns the number of registered models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns true if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
