//! LLM Client Registry for managing multiple providers.
//!
//! Provides a central registry for LLM clients with support for:
//! - Multiple providers (Infernum, Anthropic, OpenAI, etc.)
//! - Default provider selection
//! - Health checking
//! - Fallback chains

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{info, warn, debug};

use super::client::{LlmClient, LlmError, Result};

/// Registry for managing LLM clients.
///
/// Allows registering multiple providers and selecting between them.
///
/// # Example
///
/// ```rust,ignore
/// let registry = LlmClientRegistry::new()
///     .register("infernum", infernum_client)
///     .register("anthropic", anthropic_client)
///     .set_default("infernum");
///
/// // Get specific provider
/// let client = registry.get("infernum")?;
///
/// // Get default provider
/// let client = registry.default_client()?;
///
/// // Get first available provider
/// let client = registry.first_available().await?;
/// ```
pub struct LlmClientRegistry {
    /// Registered clients by name.
    clients: RwLock<HashMap<String, Arc<dyn LlmClient>>>,

    /// Default provider name.
    default_provider: RwLock<Option<String>>,

    /// Fallback order.
    fallback_order: RwLock<Vec<String>>,
}

impl LlmClientRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            default_provider: RwLock::new(None),
            fallback_order: RwLock::new(Vec::new()),
        }
    }

    /// Registers a client with the given name.
    pub fn register(self, name: impl Into<String>, client: impl LlmClient + 'static) -> Self {
        let name = name.into();
        debug!(provider = %name, "Registering LLM client");

        {
            let mut clients = self.clients.write();
            clients.insert(name.clone(), Arc::new(client));
        }

        // Add to fallback order if not present
        {
            let mut fallback = self.fallback_order.write();
            if !fallback.contains(&name) {
                fallback.push(name.clone());
            }
        }

        // Set as default if first client
        {
            let mut default = self.default_provider.write();
            if default.is_none() {
                *default = Some(name);
            }
        }

        self
    }

    /// Sets the default provider.
    pub fn set_default(self, name: impl Into<String>) -> Self {
        let name = name.into();
        *self.default_provider.write() = Some(name);
        self
    }

    /// Sets the fallback order for provider selection.
    pub fn set_fallback_order(self, order: Vec<String>) -> Self {
        *self.fallback_order.write() = order;
        self
    }

    /// Gets a client by name.
    pub fn get(&self, name: &str) -> Result<Arc<dyn LlmClient>> {
        self.clients
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| LlmError::ProviderUnavailable(format!("Provider '{}' not registered", name)))
    }

    /// Gets the default client.
    pub fn default_client(&self) -> Result<Arc<dyn LlmClient>> {
        let default = self.default_provider.read().clone();
        match default {
            Some(name) => self.get(&name),
            None => Err(LlmError::ProviderUnavailable("No default provider set".to_string())),
        }
    }

    /// Gets the first available client based on fallback order.
    pub async fn first_available(&self) -> Result<Arc<dyn LlmClient>> {
        let order = self.fallback_order.read().clone();

        for name in &order {
            if let Ok(client) = self.get(name) {
                if client.is_available().await {
                    debug!(provider = %name, "Selected available provider");
                    return Ok(client);
                }
                warn!(provider = %name, "Provider not available, trying next");
            }
        }

        Err(LlmError::ProviderUnavailable("No available providers".to_string()))
    }

    /// Returns the names of all registered providers.
    pub fn providers(&self) -> Vec<String> {
        self.clients.read().keys().cloned().collect()
    }

    /// Returns the number of registered providers.
    pub fn len(&self) -> usize {
        self.clients.read().len()
    }

    /// Returns true if no providers are registered.
    pub fn is_empty(&self) -> bool {
        self.clients.read().is_empty()
    }

    /// Checks health of all providers.
    pub async fn health_check(&self) -> HashMap<String, bool> {
        let clients = self.clients.read().clone();
        let mut results = HashMap::new();

        for (name, client) in clients {
            let available = client.is_available().await;
            results.insert(name, available);
        }

        results
    }

    /// Removes a provider from the registry.
    pub fn unregister(&self, name: &str) -> bool {
        let removed = self.clients.write().remove(name).is_some();

        if removed {
            // Remove from fallback order
            self.fallback_order.write().retain(|n| n != name);

            // Clear default if it was this provider
            let mut default = self.default_provider.write();
            if default.as_ref().map(|d| d == name).unwrap_or(false) {
                *default = self.fallback_order.read().first().cloned();
            }

            info!(provider = %name, "Unregistered LLM client");
        }

        removed
    }
}

impl Default for LlmClientRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mock::MockLlmClient;

    #[test]
    fn test_registry_register() {
        let registry = LlmClientRegistry::new()
            .register("mock1", MockLlmClient::new())
            .register("mock2", MockLlmClient::new());

        assert_eq!(registry.len(), 2);
        assert!(registry.providers().contains(&"mock1".to_string()));
        assert!(registry.providers().contains(&"mock2".to_string()));
    }

    #[test]
    fn test_registry_get() {
        let registry = LlmClientRegistry::new()
            .register("test", MockLlmClient::new().with_name("test-provider"));

        let client = registry.get("test").expect("get client");
        assert_eq!(client.provider_name(), "test-provider");

        let err = registry.get("nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn test_registry_default() {
        let registry = LlmClientRegistry::new()
            .register("first", MockLlmClient::new().with_name("first"))
            .register("second", MockLlmClient::new().with_name("second"));

        // First registered becomes default
        let default = registry.default_client().expect("default");
        assert_eq!(default.provider_name(), "first");

        // Can override default
        let registry = registry.set_default("second");
        let default = registry.default_client().expect("default");
        assert_eq!(default.provider_name(), "second");
    }

    #[tokio::test]
    async fn test_registry_first_available() {
        let registry = LlmClientRegistry::new()
            .register("unavailable", MockLlmClient::new().unavailable())
            .register("available", MockLlmClient::new().with_name("available"))
            .set_fallback_order(vec!["unavailable".to_string(), "available".to_string()]);

        let client = registry.first_available().await.expect("first available");
        assert_eq!(client.provider_name(), "available");
    }

    #[tokio::test]
    async fn test_registry_health_check() {
        let registry = LlmClientRegistry::new()
            .register("healthy", MockLlmClient::new())
            .register("unhealthy", MockLlmClient::new().unavailable());

        let health = registry.health_check().await;

        assert_eq!(health.get("healthy"), Some(&true));
        assert_eq!(health.get("unhealthy"), Some(&false));
    }

    #[test]
    fn test_registry_unregister() {
        let registry = LlmClientRegistry::new()
            .register("test", MockLlmClient::new());

        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("test");
        assert!(removed);
        assert!(registry.is_empty());

        // Should not error on non-existent
        let removed = registry.unregister("nonexistent");
        assert!(!removed);
    }

    #[test]
    fn test_registry_empty() {
        let registry = LlmClientRegistry::new();

        assert!(registry.is_empty());
        assert!(registry.default_client().is_err());
    }
}
