//! Multi-tenant isolation for shared inference infrastructure.
//!
//! This module provides tenant isolation, quota management, and
//! resource allocation for multi-tenant deployments.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Tenant identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantId(pub String);

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Quota limits for a tenant.
#[derive(Debug, Clone)]
pub struct QuotaLimits {
    /// Maximum requests per minute.
    pub requests_per_minute: u32,
    /// Maximum requests per day.
    pub requests_per_day: u32,
    /// Maximum tokens per minute.
    pub tokens_per_minute: u64,
    /// Maximum tokens per day.
    pub tokens_per_day: u64,
    /// Maximum concurrent requests.
    pub max_concurrent: u32,
    /// Maximum context length allowed.
    pub max_context_length: u32,
    /// Allowed model IDs (empty = all allowed).
    pub allowed_models: Vec<String>,
    /// Blocked model IDs.
    pub blocked_models: Vec<String>,
}

impl Default for QuotaLimits {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_day: 10_000,
            tokens_per_minute: 100_000,
            tokens_per_day: 10_000_000,
            max_concurrent: 10,
            max_context_length: 32_768,
            allowed_models: Vec::new(),
            blocked_models: Vec::new(),
        }
    }
}

impl QuotaLimits {
    /// Creates limits for a free tier.
    pub fn free_tier() -> Self {
        Self {
            requests_per_minute: 10,
            requests_per_day: 100,
            tokens_per_minute: 10_000,
            tokens_per_day: 100_000,
            max_concurrent: 2,
            max_context_length: 4_096,
            allowed_models: Vec::new(),
            blocked_models: vec!["gpt-4".into(), "claude-opus".into()],
        }
    }

    /// Creates limits for a standard tier.
    pub fn standard_tier() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_day: 5_000,
            tokens_per_minute: 100_000,
            tokens_per_day: 5_000_000,
            max_concurrent: 10,
            max_context_length: 16_384,
            allowed_models: Vec::new(),
            blocked_models: Vec::new(),
        }
    }

    /// Creates limits for a premium tier.
    pub fn premium_tier() -> Self {
        Self {
            requests_per_minute: 300,
            requests_per_day: 50_000,
            tokens_per_minute: 1_000_000,
            tokens_per_day: 50_000_000,
            max_concurrent: 50,
            max_context_length: 128_000,
            allowed_models: Vec::new(),
            blocked_models: Vec::new(),
        }
    }

    /// Creates unlimited limits (for internal/admin use).
    pub fn unlimited() -> Self {
        Self {
            requests_per_minute: u32::MAX,
            requests_per_day: u32::MAX,
            tokens_per_minute: u64::MAX,
            tokens_per_day: u64::MAX,
            max_concurrent: u32::MAX,
            max_context_length: u32::MAX,
            allowed_models: Vec::new(),
            blocked_models: Vec::new(),
        }
    }

    /// Checks if a model is allowed for this tenant.
    pub fn is_model_allowed(&self, model_id: &str) -> bool {
        // Check blocklist first
        if self.blocked_models.iter().any(|m| model_id.contains(m)) {
            return false;
        }

        // If allowlist is empty, all non-blocked models are allowed
        if self.allowed_models.is_empty() {
            return true;
        }

        // Check allowlist
        self.allowed_models.iter().any(|m| model_id.contains(m))
    }
}

/// Usage tracking for rate limiting.
#[derive(Debug)]
struct UsageWindow {
    /// Requests in current window.
    count: AtomicU64,
    /// Tokens in current window.
    tokens: AtomicU64,
    /// Window start time.
    window_start: RwLock<Instant>,
    /// Window duration.
    window_duration: Duration,
}

impl UsageWindow {
    fn new(duration: Duration) -> Self {
        Self {
            count: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
            window_duration: duration,
        }
    }

    fn reset_if_expired(&self) {
        let mut start = self.window_start.write();
        if start.elapsed() >= self.window_duration {
            self.count.store(0, Ordering::SeqCst);
            self.tokens.store(0, Ordering::SeqCst);
            *start = Instant::now();
        }
    }

    fn increment(&self, tokens: u64) {
        self.reset_if_expired();
        self.count.fetch_add(1, Ordering::SeqCst);
        self.tokens.fetch_add(tokens, Ordering::SeqCst);
    }

    fn current_count(&self) -> u64 {
        self.reset_if_expired();
        self.count.load(Ordering::SeqCst)
    }

    fn current_tokens(&self) -> u64 {
        self.reset_if_expired();
        self.tokens.load(Ordering::SeqCst)
    }
}

/// Tenant configuration and state.
pub struct Tenant {
    /// Tenant identifier.
    pub id: TenantId,
    /// Display name.
    pub name: String,
    /// Quota limits.
    pub limits: QuotaLimits,
    /// Per-minute usage tracking.
    minute_usage: UsageWindow,
    /// Per-day usage tracking.
    day_usage: UsageWindow,
    /// Current concurrent requests.
    concurrent: AtomicU64,
    /// Total requests (lifetime).
    total_requests: AtomicU64,
    /// Total tokens (lifetime).
    total_tokens: AtomicU64,
    /// Whether tenant is active.
    active: RwLock<bool>,
    /// Custom metadata.
    metadata: RwLock<HashMap<String, String>>,
}

impl Tenant {
    /// Creates a new tenant.
    pub fn new(id: impl Into<TenantId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            limits: QuotaLimits::default(),
            minute_usage: UsageWindow::new(Duration::from_secs(60)),
            day_usage: UsageWindow::new(Duration::from_secs(86400)),
            concurrent: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            active: RwLock::new(true),
            metadata: RwLock::new(HashMap::new()),
        }
    }

    /// Sets quota limits.
    pub fn with_limits(mut self, limits: QuotaLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets custom metadata.
    pub fn with_metadata(self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.write().insert(key.into(), value.into());
        self
    }

    /// Checks if tenant is active.
    pub fn is_active(&self) -> bool {
        *self.active.read()
    }

    /// Activates the tenant.
    pub fn activate(&self) {
        *self.active.write() = true;
    }

    /// Deactivates the tenant.
    pub fn deactivate(&self) {
        *self.active.write() = false;
    }

    /// Checks if a request can proceed (quota check).
    pub fn can_request(&self, estimated_tokens: u64) -> QuotaCheckResult {
        if !self.is_active() {
            return QuotaCheckResult::Denied(QuotaDenialReason::TenantInactive);
        }

        // Check concurrent limit
        let concurrent = self.concurrent.load(Ordering::SeqCst);
        if concurrent >= self.limits.max_concurrent as u64 {
            return QuotaCheckResult::Denied(QuotaDenialReason::ConcurrentLimitExceeded {
                current: concurrent as u32,
                limit: self.limits.max_concurrent,
            });
        }

        // Check per-minute request limit
        let minute_requests = self.minute_usage.current_count();
        if minute_requests >= self.limits.requests_per_minute as u64 {
            return QuotaCheckResult::Denied(QuotaDenialReason::RequestRateLimitExceeded {
                window: "minute".to_string(),
                current: minute_requests as u32,
                limit: self.limits.requests_per_minute,
            });
        }

        // Check per-day request limit
        let day_requests = self.day_usage.current_count();
        if day_requests >= self.limits.requests_per_day as u64 {
            return QuotaCheckResult::Denied(QuotaDenialReason::RequestRateLimitExceeded {
                window: "day".to_string(),
                current: day_requests as u32,
                limit: self.limits.requests_per_day,
            });
        }

        // Check per-minute token limit
        let minute_tokens = self.minute_usage.current_tokens();
        if minute_tokens + estimated_tokens > self.limits.tokens_per_minute {
            return QuotaCheckResult::Denied(QuotaDenialReason::TokenRateLimitExceeded {
                window: "minute".to_string(),
                current: minute_tokens,
                limit: self.limits.tokens_per_minute,
            });
        }

        // Check per-day token limit
        let day_tokens = self.day_usage.current_tokens();
        if day_tokens + estimated_tokens > self.limits.tokens_per_day {
            return QuotaCheckResult::Denied(QuotaDenialReason::TokenRateLimitExceeded {
                window: "day".to_string(),
                current: day_tokens,
                limit: self.limits.tokens_per_day,
            });
        }

        QuotaCheckResult::Allowed
    }

    /// Records request start (increments concurrent).
    pub fn request_start(&self) {
        self.concurrent.fetch_add(1, Ordering::SeqCst);
    }

    /// Records request completion.
    pub fn request_complete(&self, tokens_used: u64) {
        self.concurrent.fetch_sub(1, Ordering::SeqCst);
        self.minute_usage.increment(tokens_used);
        self.day_usage.increment(tokens_used);
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        self.total_tokens.fetch_add(tokens_used, Ordering::SeqCst);
    }

    /// Returns current usage statistics.
    pub fn usage_stats(&self) -> TenantUsageStats {
        TenantUsageStats {
            tenant_id: self.id.clone(),
            concurrent_requests: self.concurrent.load(Ordering::SeqCst) as u32,
            minute_requests: self.minute_usage.current_count() as u32,
            minute_tokens: self.minute_usage.current_tokens(),
            day_requests: self.day_usage.current_count() as u32,
            day_tokens: self.day_usage.current_tokens(),
            total_requests: self.total_requests.load(Ordering::SeqCst),
            total_tokens: self.total_tokens.load(Ordering::SeqCst),
        }
    }

    /// Returns metadata.
    pub fn metadata(&self) -> HashMap<String, String> {
        self.metadata.read().clone()
    }
}

/// Result of a quota check.
#[derive(Debug, Clone)]
pub enum QuotaCheckResult {
    /// Request is allowed.
    Allowed,
    /// Request is denied.
    Denied(QuotaDenialReason),
}

impl QuotaCheckResult {
    /// Returns true if allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Reason for quota denial.
#[derive(Debug, Clone)]
pub enum QuotaDenialReason {
    /// Tenant is inactive.
    TenantInactive,
    /// Tenant not found.
    TenantNotFound,
    /// Concurrent request limit exceeded.
    ConcurrentLimitExceeded {
        /// Current number of concurrent requests.
        current: u32,
        /// Maximum allowed concurrent requests.
        limit: u32,
    },
    /// Request rate limit exceeded.
    RequestRateLimitExceeded {
        /// Time window for rate limiting.
        window: String,
        /// Current request count in window.
        current: u32,
        /// Maximum requests per window.
        limit: u32,
    },
    /// Token rate limit exceeded.
    TokenRateLimitExceeded {
        /// Time window for rate limiting.
        window: String,
        /// Current token count in window.
        current: u64,
        /// Maximum tokens per window.
        limit: u64,
    },
    /// Model not allowed for tenant.
    ModelNotAllowed {
        /// The model ID that was requested.
        model_id: String,
    },
    /// Context length exceeded.
    ContextLengthExceeded {
        /// Requested context length.
        requested: u32,
        /// Maximum allowed context length.
        limit: u32,
    },
}

impl std::fmt::Display for QuotaDenialReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TenantInactive => write!(f, "Tenant is inactive"),
            Self::TenantNotFound => write!(f, "Tenant not found"),
            Self::ConcurrentLimitExceeded { current, limit } => {
                write!(
                    f,
                    "Concurrent request limit exceeded: {}/{} requests",
                    current, limit
                )
            }
            Self::RequestRateLimitExceeded {
                window,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Request rate limit exceeded for {}: {}/{} requests",
                    window, current, limit
                )
            }
            Self::TokenRateLimitExceeded {
                window,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Token rate limit exceeded for {}: {}/{} tokens",
                    window, current, limit
                )
            }
            Self::ModelNotAllowed { model_id } => {
                write!(f, "Model '{}' not allowed for this tenant", model_id)
            }
            Self::ContextLengthExceeded { requested, limit } => {
                write!(
                    f,
                    "Context length {} exceeds limit of {} tokens",
                    requested, limit
                )
            }
        }
    }
}

/// Usage statistics for a tenant.
#[derive(Debug, Clone)]
pub struct TenantUsageStats {
    /// Tenant ID.
    pub tenant_id: TenantId,
    /// Current concurrent requests.
    pub concurrent_requests: u32,
    /// Requests in current minute.
    pub minute_requests: u32,
    /// Tokens in current minute.
    pub minute_tokens: u64,
    /// Requests today.
    pub day_requests: u32,
    /// Tokens today.
    pub day_tokens: u64,
    /// Total requests (lifetime).
    pub total_requests: u64,
    /// Total tokens (lifetime).
    pub total_tokens: u64,
}

/// Manager for multi-tenant operations.
pub struct TenantManager {
    /// Registered tenants.
    tenants: RwLock<HashMap<TenantId, Arc<Tenant>>>,
    /// Default limits for new tenants.
    default_limits: QuotaLimits,
}

impl TenantManager {
    /// Creates a new tenant manager.
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            default_limits: QuotaLimits::default(),
        }
    }

    /// Creates with custom default limits.
    pub fn with_default_limits(limits: QuotaLimits) -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            default_limits: limits,
        }
    }

    /// Registers a tenant.
    pub fn register(&self, tenant: Tenant) -> Arc<Tenant> {
        let t = Arc::new(tenant);
        self.tenants.write().insert(t.id.clone(), Arc::clone(&t));
        t
    }

    /// Creates and registers a tenant with default limits.
    pub fn create_tenant(
        &self,
        id: impl Into<TenantId>,
        name: impl Into<String>,
    ) -> Arc<Tenant> {
        let tenant = Tenant::new(id, name).with_limits(self.default_limits.clone());
        self.register(tenant)
    }

    /// Gets a tenant by ID.
    pub fn get(&self, id: &TenantId) -> Option<Arc<Tenant>> {
        self.tenants.read().get(id).cloned()
    }

    /// Gets or creates a tenant.
    pub fn get_or_create(&self, id: impl Into<TenantId>, name: impl Into<String>) -> Arc<Tenant> {
        let id = id.into();
        if let Some(tenant) = self.get(&id) {
            return tenant;
        }
        self.create_tenant(id, name)
    }

    /// Removes a tenant.
    pub fn remove(&self, id: &TenantId) -> Option<Arc<Tenant>> {
        self.tenants.write().remove(id)
    }

    /// Lists all tenants.
    pub fn list(&self) -> Vec<Arc<Tenant>> {
        self.tenants.read().values().cloned().collect()
    }

    /// Lists active tenants.
    pub fn list_active(&self) -> Vec<Arc<Tenant>> {
        self.tenants
            .read()
            .values()
            .filter(|t| t.is_active())
            .cloned()
            .collect()
    }

    /// Checks quota for a tenant.
    pub fn check_quota(
        &self,
        tenant_id: &TenantId,
        model_id: &str,
        estimated_tokens: u64,
        context_length: u32,
    ) -> QuotaCheckResult {
        let tenant = match self.get(tenant_id) {
            Some(t) => t,
            None => return QuotaCheckResult::Denied(QuotaDenialReason::TenantNotFound),
        };

        // Check model allowance
        if !tenant.limits.is_model_allowed(model_id) {
            return QuotaCheckResult::Denied(QuotaDenialReason::ModelNotAllowed {
                model_id: model_id.to_string(),
            });
        }

        // Check context length
        if context_length > tenant.limits.max_context_length {
            return QuotaCheckResult::Denied(QuotaDenialReason::ContextLengthExceeded {
                requested: context_length,
                limit: tenant.limits.max_context_length,
            });
        }

        // Check rate limits
        tenant.can_request(estimated_tokens)
    }

    /// Returns usage stats for all tenants.
    pub fn all_usage_stats(&self) -> Vec<TenantUsageStats> {
        self.tenants
            .read()
            .values()
            .map(|t| t.usage_stats())
            .collect()
    }

    /// Returns aggregate usage stats.
    pub fn aggregate_stats(&self) -> AggregateUsageStats {
        let tenants = self.tenants.read();
        let mut stats = AggregateUsageStats::default();

        for tenant in tenants.values() {
            let usage = tenant.usage_stats();
            stats.total_tenants += 1;
            if tenant.is_active() {
                stats.active_tenants += 1;
            }
            stats.total_concurrent += usage.concurrent_requests as u64;
            stats.total_minute_requests += usage.minute_requests as u64;
            stats.total_minute_tokens += usage.minute_tokens;
            stats.total_requests += usage.total_requests;
            stats.total_tokens += usage.total_tokens;
        }

        stats
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate usage statistics across all tenants.
#[derive(Debug, Clone, Default)]
pub struct AggregateUsageStats {
    /// Total number of tenants.
    pub total_tenants: u64,
    /// Number of active tenants.
    pub active_tenants: u64,
    /// Total concurrent requests.
    pub total_concurrent: u64,
    /// Total requests in current minute.
    pub total_minute_requests: u64,
    /// Total tokens in current minute.
    pub total_minute_tokens: u64,
    /// Total requests (lifetime).
    pub total_requests: u64,
    /// Total tokens (lifetime).
    pub total_tokens: u64,
}

/// Request context with tenant information.
#[derive(Debug, Clone)]
pub struct TenantContext {
    /// Tenant ID.
    pub tenant_id: TenantId,
    /// Request ID.
    pub request_id: String,
    /// Estimated tokens.
    pub estimated_tokens: u64,
}

impl TenantContext {
    /// Creates a new tenant context.
    pub fn new(tenant_id: impl Into<TenantId>, request_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            request_id: request_id.into(),
            estimated_tokens: 0,
        }
    }

    /// Sets estimated tokens.
    pub fn with_estimated_tokens(mut self, tokens: u64) -> Self {
        self.estimated_tokens = tokens;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quota_limits_tiers() {
        let free = QuotaLimits::free_tier();
        assert_eq!(free.requests_per_minute, 10);
        assert!(free.blocked_models.contains(&"gpt-4".to_string()));

        let standard = QuotaLimits::standard_tier();
        assert_eq!(standard.requests_per_minute, 60);

        let premium = QuotaLimits::premium_tier();
        assert_eq!(premium.requests_per_minute, 300);
    }

    #[test]
    fn test_model_allowance() {
        let mut limits = QuotaLimits::default();
        limits.blocked_models = vec!["expensive".to_string()];

        assert!(limits.is_model_allowed("cheap-model"));
        assert!(!limits.is_model_allowed("expensive-model"));

        limits.allowed_models = vec!["allowed".to_string()];
        assert!(limits.is_model_allowed("allowed-model"));
        assert!(!limits.is_model_allowed("other-model"));
    }

    #[test]
    fn test_tenant_creation() {
        let tenant = Tenant::new("tenant-001", "Test Tenant")
            .with_limits(QuotaLimits::standard_tier())
            .with_metadata("plan", "standard");

        assert!(tenant.is_active());
        assert_eq!(tenant.limits.requests_per_minute, 60);
        assert_eq!(tenant.metadata().get("plan"), Some(&"standard".to_string()));
    }

    #[test]
    fn test_quota_check() {
        let tenant = Tenant::new("test", "Test").with_limits(QuotaLimits {
            requests_per_minute: 2,
            max_concurrent: 1,
            ..Default::default()
        });

        // First request should be allowed
        assert!(tenant.can_request(100).is_allowed());

        // Start a request
        tenant.request_start();

        // Second concurrent should be denied
        let result = tenant.can_request(100);
        assert!(matches!(
            result,
            QuotaCheckResult::Denied(QuotaDenialReason::ConcurrentLimitExceeded { .. })
        ));

        // Complete the request
        tenant.request_complete(100);

        // Now should be allowed again
        assert!(tenant.can_request(100).is_allowed());
    }

    #[test]
    fn test_tenant_manager() {
        let manager = TenantManager::new();

        let tenant = manager.create_tenant("t1", "Tenant 1");
        assert!(tenant.is_active());

        let fetched = manager.get(&"t1".into());
        assert!(fetched.is_some());

        let list = manager.list();
        assert_eq!(list.len(), 1);

        manager.remove(&"t1".into());
        let list = manager.list();
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_usage_stats() {
        let tenant = Tenant::new("test", "Test");

        tenant.request_start();
        tenant.request_complete(500);
        tenant.request_start();
        tenant.request_complete(300);

        let stats = tenant.usage_stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.total_tokens, 800);
    }
}
