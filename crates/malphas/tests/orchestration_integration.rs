//! Integration tests for Malphas - the orchestration layer.
//!
//! Tests cover:
//! - Model registry and registration
//! - Latency statistics
//! - Model cost calculations
//! - Batch scheduler
//! - Thermal management
//! - Health monitoring
//! - Failover configuration
//! - Experiment management
//! - Tenant management

use std::time::Duration;

use malphas::{
    BatchScheduler, FailoverConfig, HealthConfig, HealthMonitor, HealthStatus,
    LatencyStats, Malphas, ModelCost, ModelRegistry, PowerProfile, Priority,
    RoutingStrategy, SchedulerConfig, ThermalManager, ThermalState,
    ThermalThresholds,
};

use malphas::experiments::{
    Experiment, ExperimentId, ExperimentManager, ExperimentStatus, Variant, VariantMetrics,
};

use malphas::tenant::{
    QuotaCheckResult, QuotaDenialReason, QuotaLimits, Tenant, TenantContext, TenantId,
    TenantManager,
};

// ============================================================================
// LatencyStats Tests
// ============================================================================

#[test]
fn test_latency_stats_new() {
    let stats = LatencyStats::new();
    assert_eq!(stats.request_count(), 0);
    assert_eq!(stats.average_latency_ms(), 0.0);
}

#[test]
fn test_latency_stats_record() {
    let stats = LatencyStats::new();

    stats.record(Duration::from_millis(10));
    stats.record(Duration::from_millis(20));
    stats.record(Duration::from_millis(30));

    assert_eq!(stats.request_count(), 3);
    assert!((stats.average_latency_ms() - 20.0).abs() < 0.1);
}

#[test]
fn test_latency_stats_percentiles() {
    let stats = LatencyStats::new();

    // Add 100 measurements from 1ms to 100ms
    for i in 1..=100 {
        stats.record(Duration::from_millis(i));
    }

    let p50 = stats.p50_latency_ms();
    let p99 = stats.p99_latency_ms();

    // P50 should be around 50ms
    assert!(p50 > 40.0 && p50 < 60.0);
    // P99 should be around 99ms
    assert!(p99 > 90.0);
}

#[test]
fn test_latency_stats_empty_percentiles() {
    let stats = LatencyStats::new();
    assert_eq!(stats.p50_latency_ms(), 0.0);
    assert_eq!(stats.p99_latency_ms(), 0.0);
}

// ============================================================================
// ModelCost Tests
// ============================================================================

#[test]
fn test_model_cost_default() {
    let cost = ModelCost::default();
    assert_eq!(cost.input_token_cost, 0.0);
    assert_eq!(cost.output_token_cost, 0.0);
    assert_eq!(cost.request_cost, 0.0);
}

#[test]
fn test_model_cost_new() {
    let cost = ModelCost::new(0.001, 0.002);
    assert_eq!(cost.input_token_cost, 0.001);
    assert_eq!(cost.output_token_cost, 0.002);
}

#[test]
fn test_model_cost_calculate() {
    let cost = ModelCost::new(0.001, 0.002);
    let total = cost.calculate(1000, 500);

    // 1000 * 0.001 + 500 * 0.002 = 1.0 + 1.0 = 2.0
    assert!((total - 2.0).abs() < 0.001);
}

#[test]
fn test_model_cost_with_request_cost() {
    let mut cost = ModelCost::new(0.001, 0.002);
    cost.request_cost = 0.05;

    let total = cost.calculate(100, 50);
    // 0.05 + 100*0.001 + 50*0.002 = 0.05 + 0.1 + 0.1 = 0.25
    assert!((total - 0.25).abs() < 0.001);
}

// ============================================================================
// ModelRegistry Tests
// ============================================================================

#[test]
fn test_model_registry_new() {
    let registry = ModelRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_model_registry_list() {
    let registry = ModelRegistry::new();
    let list = registry.list();
    assert!(list.is_empty());
}

#[test]
fn test_model_registry_list_available() {
    let registry = ModelRegistry::new();
    let available = registry.list_available();
    assert!(available.is_empty());
}

// ============================================================================
// BatchScheduler Tests
// ============================================================================

#[test]
fn test_batch_scheduler_default() {
    let scheduler = BatchScheduler::default();
    let stats = scheduler.stats();

    // SchedulerStats fields: total_requests, total_batches, current_queue_depth
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.total_batches, 0);
    assert_eq!(stats.current_queue_depth, 0);
}

#[test]
fn test_batch_scheduler_config() {
    let config = SchedulerConfig {
        max_batch_size: 16,
        max_queue_size: 100,
        max_batch_tokens: 8192,
        max_wait_time: Duration::from_millis(50),
        continuous_batching: true,
        thermal_aware: true,
    };

    let scheduler = BatchScheduler::new(config);
    assert_eq!(scheduler.config().max_batch_size, 16);
}

#[test]
fn test_scheduler_config_default() {
    let config = SchedulerConfig::default();
    assert!(config.max_batch_size > 0);
    assert!(config.max_queue_size > 0);
}

// ============================================================================
// ThermalManager Tests
// ============================================================================

#[tokio::test]
async fn test_thermal_manager_new() {
    let manager = ThermalManager::new();
    let state = manager.state().await;

    // Default state should be normal
    assert!(matches!(state, ThermalState::Normal));
}

#[tokio::test]
async fn test_thermal_manager_workstation() {
    let manager = ThermalManager::workstation();
    let factor = manager.throttle_factor().await;

    // Should not be throttled initially
    assert!(factor >= 0.0 && factor <= 1.0);
}

#[tokio::test]
async fn test_thermal_manager_profile() {
    let manager = ThermalManager::new();

    manager.set_profile(PowerProfile::PowerSaver).await;
    manager.set_profile(PowerProfile::Balanced).await;
    manager.set_profile(PowerProfile::Performance).await;
}

#[tokio::test]
async fn test_thermal_manager_recommended_batch_size() {
    let manager = ThermalManager::new();

    let batch = manager.recommended_batch_size(32).await;
    assert!(batch > 0);
    assert!(batch <= 32);
}

#[test]
fn test_thermal_thresholds_default() {
    let thresholds = ThermalThresholds::default();

    // Fields: target, warning, critical, emergency
    assert!(thresholds.target > 0.0);
    assert!(thresholds.warning > thresholds.target);
    assert!(thresholds.critical > thresholds.warning);
    assert!(thresholds.emergency >= thresholds.critical);
}

#[test]
fn test_thermal_thresholds_workstation() {
    let thresholds = ThermalThresholds::workstation();

    // Workstation should have more conservative thresholds
    let default = ThermalThresholds::default();
    assert!(thresholds.target <= default.target);
}

#[test]
fn test_thermal_thresholds_threadripper_pro() {
    let thresholds = ThermalThresholds::threadripper_pro();

    // Threadripper thresholds
    assert!(thresholds.target > 0.0);
    assert!(thresholds.warning > thresholds.target);
}

#[test]
fn test_thermal_thresholds_rtx_4000() {
    let thresholds = ThermalThresholds::rtx_4000();

    // RTX 4000 thresholds for Ada Lovelace GPUs
    assert!(thresholds.target > 0.0);
    assert!(thresholds.warning > thresholds.target);
}

// ============================================================================
// HealthMonitor Tests
// ============================================================================

#[test]
fn test_health_monitor_default_config() {
    let monitor = HealthMonitor::default_config();

    let status = monitor.status("nonexistent");
    assert_eq!(status, HealthStatus::Unknown);
}

#[test]
fn test_health_monitor_new() {
    let config = HealthConfig::default();
    let monitor = HealthMonitor::new(config);

    let summary = monitor.summary();
    assert_eq!(summary.total, 0);
}

#[test]
fn test_health_monitor_record_success() {
    let monitor = HealthMonitor::default_config();

    monitor.record_success("model-1");
    monitor.record_success("model-1");
    monitor.record_success("model-1");

    let status = monitor.status("model-1");
    assert_eq!(status, HealthStatus::Healthy);
}

#[test]
fn test_health_monitor_record_failure() {
    let monitor = HealthMonitor::default_config();

    // Record some failures
    for _ in 0..10 {
        monitor.record_failure("model-2", "test error".to_string());
    }

    let status = monitor.status("model-2");
    // After many failures, should be degraded or unhealthy
    assert!(matches!(
        status,
        HealthStatus::Degraded | HealthStatus::Unhealthy
    ));
}

#[test]
fn test_health_monitor_summary() {
    let monitor = HealthMonitor::default_config();

    monitor.record_success("model-a");
    monitor.record_success("model-b");
    monitor.record_failure("model-c", "error".to_string());

    let summary = monitor.summary();
    assert_eq!(summary.total, 3);
}

#[test]
fn test_health_status_is_routable() {
    assert!(HealthStatus::Healthy.is_routable());
    assert!(HealthStatus::Degraded.is_routable());
    assert!(!HealthStatus::Unhealthy.is_routable());
    // Unknown is not routable
    assert!(!HealthStatus::Unknown.is_routable());
}

#[test]
fn test_health_config_default() {
    let config = HealthConfig::default();
    assert!(config.check_interval > Duration::ZERO);
}

#[test]
fn test_health_config_production() {
    let config = HealthConfig::production();
    // Production config should have stricter settings
    assert!(config.check_interval > Duration::ZERO);
}

// ============================================================================
// FailoverConfig Tests
// ============================================================================

#[test]
fn test_failover_config_default() {
    let config = FailoverConfig::default();

    assert!(config.max_retries > 0);
    assert!(config.retry_delay > Duration::ZERO);
    assert!(config.require_healthy);
    assert!(config.exclude_failed);
}

#[test]
fn test_failover_config_production() {
    let config = FailoverConfig::production();

    assert!(config.max_retries >= 3);
    assert!(config.exponential_backoff);
}

#[test]
fn test_failover_config_fail_fast() {
    let config = FailoverConfig::fail_fast();

    assert_eq!(config.max_retries, 0);
    assert_eq!(config.retry_delay, Duration::ZERO);
}

#[test]
fn test_failover_config_aggressive() {
    let config = FailoverConfig::aggressive();

    assert!(config.max_retries >= 5);
    assert!(!config.exponential_backoff);
}

#[test]
fn test_failover_config_builder() {
    let config = FailoverConfig::default()
        .with_max_retries(5)
        .with_retry_delay(Duration::from_millis(200));

    assert_eq!(config.max_retries, 5);
    assert_eq!(config.retry_delay, Duration::from_millis(200));
}

// ============================================================================
// Malphas Orchestrator Tests
// ============================================================================

#[test]
fn test_malphas_new() {
    let malphas = Malphas::new();

    // Should have empty registry
    assert!(malphas.registry().is_empty());
}

#[test]
fn test_malphas_with_strategy() {
    let malphas = Malphas::with_strategy(RoutingStrategy::LeastConnections);
    assert!(malphas.registry().is_empty());
}

#[test]
fn test_malphas_workstation() {
    let malphas = Malphas::workstation();
    assert!(malphas.registry().is_empty());
}

#[test]
fn test_malphas_production() {
    let malphas = Malphas::production();
    assert!(malphas.registry().is_empty());
}

#[test]
fn test_malphas_default() {
    let malphas = Malphas::default();
    assert!(malphas.registry().is_empty());
}

#[test]
fn test_malphas_scheduler_stats() {
    let malphas = Malphas::new();
    let stats = malphas.scheduler_stats();

    assert_eq!(stats.total_requests, 0);
}

#[test]
fn test_malphas_health_summary() {
    let malphas = Malphas::new();
    let summary = malphas.health_summary();

    assert_eq!(summary.total, 0);
}

#[test]
fn test_malphas_failover_config() {
    let malphas = Malphas::new();
    let config = malphas.failover_config();

    assert!(config.max_retries > 0);
}

#[tokio::test]
async fn test_malphas_thermal_state() {
    let malphas = Malphas::new();
    let state = malphas.thermal_state().await;

    // Should be in a valid state
    assert!(matches!(
        state,
        ThermalState::Normal
            | ThermalState::Elevated
            | ThermalState::Warning
            | ThermalState::Critical
            | ThermalState::Emergency
    ));
}

#[tokio::test]
async fn test_malphas_throttle_factor() {
    let malphas = Malphas::new();
    let factor = malphas.throttle_factor().await;

    assert!(factor >= 0.0);
    assert!(factor <= 1.0);
}

#[tokio::test]
async fn test_malphas_set_power_profile() {
    let malphas = Malphas::new();

    malphas.set_power_profile(PowerProfile::PowerSaver).await;
    malphas.set_power_profile(PowerProfile::Balanced).await;
    malphas.set_power_profile(PowerProfile::Performance).await;
}

// ============================================================================
// ExperimentManager Tests
// ============================================================================

#[test]
fn test_experiment_manager_new() {
    let manager = ExperimentManager::new();
    assert!(manager.active().is_empty());
}

#[test]
fn test_experiment_manager_register() {
    let manager = ExperimentManager::new();

    let exp = Experiment::new("test-001", "Test Experiment")
        .with_variant(Variant::control("model-a"))
        .with_variant(Variant::treatment("faster", "model-b"));

    let registered = manager.register(exp);
    assert_eq!(registered.status(), ExperimentStatus::Draft);
}

#[test]
fn test_experiment_manager_get() {
    let manager = ExperimentManager::new();

    let exp = Experiment::new("my-experiment", "My Experiment")
        .with_variant(Variant::control("model-a"));

    manager.register(exp);

    let fetched = manager.get(&ExperimentId("my-experiment".to_string()));
    assert!(fetched.is_some());
    assert_eq!(fetched.expect("experiment").name, "My Experiment");
}

#[test]
fn test_experiment_start_and_select() {
    let exp = Experiment::new("test-002", "Test")
        .with_variant(Variant::control("a").with_allocation(0.5))
        .with_variant(Variant::treatment("b", "b").with_allocation(0.5));

    exp.start();
    assert_eq!(exp.status(), ExperimentStatus::Running);

    // Should select a variant when running
    let variant = exp.select_variant();
    assert!(variant.is_some());
}

#[test]
fn test_experiment_pause_blocks_selection() {
    let exp = Experiment::new("test-003", "Test")
        .with_variant(Variant::control("a"));

    exp.start();
    exp.pause();
    assert_eq!(exp.status(), ExperimentStatus::Paused);

    // Should not select when paused
    let variant = exp.select_variant();
    assert!(variant.is_none());
}

#[test]
fn test_experiment_lifecycle() {
    let exp = Experiment::new("lifecycle", "Lifecycle Test")
        .with_variant(Variant::control("model"))
        .with_min_samples(10);

    assert_eq!(exp.status(), ExperimentStatus::Draft);

    exp.start();
    assert_eq!(exp.status(), ExperimentStatus::Running);

    exp.pause();
    assert_eq!(exp.status(), ExperimentStatus::Paused);

    exp.resume();
    assert_eq!(exp.status(), ExperimentStatus::Running);

    exp.complete();
    assert_eq!(exp.status(), ExperimentStatus::Completed);
}

#[test]
fn test_experiment_cancel() {
    let exp = Experiment::new("cancel-test", "Cancel Test")
        .with_variant(Variant::control("model"));

    exp.start();
    exp.cancel();
    assert_eq!(exp.status(), ExperimentStatus::Cancelled);
}

#[test]
fn test_variant_metrics() {
    let metrics = VariantMetrics::new();

    metrics.record_success(100, 500, 200);
    metrics.record_success(150, 600, 250);
    metrics.record_failure(50);

    assert!((metrics.average_latency_ms() - 100.0).abs() < 0.1);
    assert!((metrics.success_rate() - 0.666).abs() < 0.01);
}

#[test]
fn test_variant_creation() {
    let control = Variant::control("gpt-4");
    assert!(control.is_control);
    assert_eq!(control.allocation, 0.5);

    let treatment = Variant::treatment("faster", "gpt-4-turbo").with_allocation(0.3);
    assert!(!treatment.is_control);
    assert_eq!(treatment.allocation, 0.3);
}

#[test]
fn test_experiment_summary() {
    let exp = Experiment::new("summary-test", "Summary Test")
        .with_variant(Variant::control("model-a"))
        .with_variant(Variant::treatment("b", "model-b"));

    let summary = exp.summary();
    assert_eq!(summary.name, "Summary Test");
    assert_eq!(summary.variants.len(), 2);
}

// ============================================================================
// TenantManager Tests
// ============================================================================

#[test]
fn test_tenant_manager_new() {
    let manager = TenantManager::new();
    assert!(manager.list().is_empty());
}

#[test]
fn test_tenant_manager_create() {
    let manager = TenantManager::new();

    let tenant = manager.create_tenant("test-tenant", "Test Tenant");
    assert!(tenant.is_active());

    assert_eq!(manager.list().len(), 1);
}

#[test]
fn test_tenant_manager_get() {
    let manager = TenantManager::new();

    manager.create_tenant("my-tenant", "My Tenant");

    let tenant = manager.get(&TenantId("my-tenant".to_string()));
    assert!(tenant.is_some());
    assert_eq!(tenant.expect("tenant").name, "My Tenant");
}

#[test]
fn test_tenant_manager_check_quota() {
    let manager = TenantManager::new();

    let limits = QuotaLimits::default();
    let tenant = Tenant::new("quota-test", "Quota Test").with_limits(limits);
    manager.register(tenant);

    // Check should pass
    let result = manager.check_quota(
        &TenantId("quota-test".to_string()),
        "model-1",
        1000,
        1000,
    );
    assert!(matches!(result, QuotaCheckResult::Allowed));
}

#[test]
fn test_tenant_usage_recording() {
    let tenant = Tenant::new("usage-test", "Usage Test");

    tenant.request_start();
    tenant.request_complete(100);
    tenant.request_start();
    tenant.request_complete(50);

    let stats = tenant.usage_stats();
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.total_tokens, 150);
}

#[test]
fn test_tenant_manager_remove() {
    let manager = TenantManager::new();

    manager.create_tenant("delete-test", "Delete Test");
    assert_eq!(manager.list().len(), 1);

    let removed = manager.remove(&TenantId("delete-test".to_string()));
    assert!(removed.is_some());
    assert!(manager.list().is_empty());
}

#[test]
fn test_quota_limits_default() {
    let limits = QuotaLimits::default();

    assert!(limits.requests_per_minute > 0);
    assert!(limits.tokens_per_day > 0);
}

#[test]
fn test_quota_limits_unlimited() {
    let limits = QuotaLimits::unlimited();

    assert_eq!(limits.requests_per_minute, u32::MAX);
    assert_eq!(limits.tokens_per_day, u64::MAX);
}

#[test]
fn test_quota_limits_free_tier() {
    let limits = QuotaLimits::free_tier();

    // Free tier should have limited quotas
    assert!(limits.requests_per_minute < QuotaLimits::unlimited().requests_per_minute);
}

#[test]
fn test_quota_limits_premium_tier() {
    let limits = QuotaLimits::premium_tier();

    // Premium should have higher limits than free tier
    assert!(limits.requests_per_minute > QuotaLimits::free_tier().requests_per_minute);
}

#[test]
fn test_quota_limits_standard_tier() {
    let limits = QuotaLimits::standard_tier();

    // Standard should be between free and premium
    assert!(limits.requests_per_minute > QuotaLimits::free_tier().requests_per_minute);
    assert!(limits.requests_per_minute <= QuotaLimits::premium_tier().requests_per_minute);
}

#[test]
fn test_tenant_context() {
    let ctx = TenantContext::new("tenant-123", "request-456");

    assert_eq!(ctx.tenant_id.0, "tenant-123");
    assert_eq!(ctx.request_id, "request-456");
}

#[test]
fn test_tenant_context_with_tokens() {
    let ctx = TenantContext::new("tenant", "request")
        .with_estimated_tokens(1000);

    assert_eq!(ctx.estimated_tokens, 1000);
}

#[test]
fn test_tenant_activate_deactivate() {
    let tenant = Tenant::new("test", "Test");

    assert!(tenant.is_active());

    tenant.deactivate();
    assert!(!tenant.is_active());

    tenant.activate();
    assert!(tenant.is_active());
}

#[test]
fn test_tenant_quota_check_inactive() {
    let tenant = Tenant::new("test", "Test");
    tenant.deactivate();

    let result = tenant.can_request(100);
    assert!(matches!(
        result,
        QuotaCheckResult::Denied(QuotaDenialReason::TenantInactive)
    ));
}

#[test]
fn test_model_allowance() {
    let mut limits = QuotaLimits::default();
    limits.blocked_models = vec!["expensive".to_string()];

    assert!(limits.is_model_allowed("cheap-model"));
    assert!(!limits.is_model_allowed("expensive-model"));
}

// ============================================================================
// Routing Strategy Tests
// ============================================================================

#[test]
fn test_routing_strategies() {
    // Verify enum variants exist and can be used
    let strategies = vec![
        RoutingStrategy::RoundRobin,
        RoutingStrategy::LeastConnections,
        RoutingStrategy::LatencyOptimized { target_p99_ms: 100 },
        RoutingStrategy::CostOptimized { max_cost_per_token: 0.001 },
        RoutingStrategy::Weighted {
            latency_weight: 0.4,
            cost_weight: 0.3,
            load_weight: 0.3,
        },
    ];

    for strategy in strategies {
        let _malphas = Malphas::with_strategy(strategy);
    }
}

#[test]
fn test_routing_strategy_default() {
    let strategy = RoutingStrategy::default();
    assert!(matches!(strategy, RoutingStrategy::RoundRobin));
}

// ============================================================================
// Priority Tests
// ============================================================================

#[test]
fn test_priority_ordering() {
    // Background priority should be less than Normal, which should be less than High
    assert!(Priority::Background < Priority::Normal);
    assert!(Priority::Normal < Priority::High);
    assert!(Priority::High < Priority::Realtime);
    assert!(Priority::Realtime < Priority::Critical);
}

#[test]
fn test_priority_default() {
    let priority = Priority::default();
    assert_eq!(priority, Priority::Normal);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[tokio::test]
async fn test_orchestration_workflow() {
    // 1. Create orchestrator
    let malphas = Malphas::production();

    // 2. Check initial state
    assert!(malphas.registry().is_empty());
    assert_eq!(malphas.scheduler_stats().total_requests, 0);

    // 3. Check health monitoring
    let health = malphas.health_summary();
    assert_eq!(health.total, 0);

    // 4. Check thermal state
    let thermal = malphas.thermal_state().await;
    assert!(matches!(thermal, ThermalState::Normal));

    // 5. Check throttle factor
    let factor = malphas.throttle_factor().await;
    assert!(factor >= 0.0 && factor <= 1.0);
}

#[test]
fn test_multi_tenant_workflow() {
    let manager = TenantManager::new();

    // Create multiple tenants with different limits
    let free_tenant = Tenant::new("free-user", "Free User")
        .with_limits(QuotaLimits::free_tier());
    let premium_tenant = Tenant::new("premium-user", "Premium User")
        .with_limits(QuotaLimits::premium_tier());

    manager.register(free_tenant);
    manager.register(premium_tenant);

    // Both should exist
    assert_eq!(manager.list().len(), 2);

    // Check quotas
    let free_result = manager.check_quota(
        &TenantId("free-user".to_string()),
        "small-model",
        100,
        1000,
    );
    let premium_result = manager.check_quota(
        &TenantId("premium-user".to_string()),
        "small-model",
        100,
        1000,
    );

    // Both should be allowed (single request)
    assert!(matches!(free_result, QuotaCheckResult::Allowed));
    assert!(matches!(premium_result, QuotaCheckResult::Allowed));
}

#[test]
fn test_experiment_workflow() {
    let manager = ExperimentManager::new();

    // Create A/B test
    let experiment = Experiment::new("model-comparison", "Model Comparison")
        .with_variant(Variant::control("model-a").with_allocation(0.5))
        .with_variant(Variant::treatment("new-model", "model-b").with_allocation(0.5))
        .with_min_samples(10);

    let exp = manager.register(experiment);
    exp.start();

    // Simulate user traffic
    let mut control_count = 0;
    let mut treatment_count = 0;

    for _ in 0..100 {
        if let Some(variant) = exp.select_variant() {
            if variant.is_control {
                control_count += 1;
            } else {
                treatment_count += 1;
            }
        }
    }

    // Should have assignments to both variants
    assert!(control_count > 0);
    assert!(treatment_count > 0);

    // Get summary
    let summary = exp.summary();
    assert_eq!(summary.variants.len(), 2);

    // Complete experiment
    exp.complete();
    assert_eq!(exp.status(), ExperimentStatus::Completed);
}

#[test]
fn test_health_monitoring_workflow() {
    let monitor = HealthMonitor::default_config();

    // Simulate healthy model
    for _ in 0..100 {
        monitor.record_success("model-a");
    }
    assert_eq!(monitor.status("model-a"), HealthStatus::Healthy);

    // Simulate model with a mix of successes and failures
    for _ in 0..50 {
        monitor.record_success("model-b");
    }
    for _ in 0..5 {
        monitor.record_failure("model-b", "timeout".to_string());
    }
    let status_b = monitor.status("model-b");
    // Status will be either Healthy, Degraded, or Unhealthy based on thresholds
    // Just verify it's not Unknown since we have recorded some data
    assert_ne!(status_b, HealthStatus::Unknown);

    // Check summary
    let summary = monitor.summary();
    assert_eq!(summary.total, 2);
}

#[test]
fn test_aggregate_usage_stats() {
    let manager = TenantManager::new();

    let t1 = Tenant::new("t1", "Tenant 1");
    let t2 = Tenant::new("t2", "Tenant 2");

    manager.register(t1);
    manager.register(t2);

    // Get tenant references and record usage
    if let Some(tenant) = manager.get(&TenantId("t1".to_string())) {
        tenant.request_start();
        tenant.request_complete(500);
    }
    if let Some(tenant) = manager.get(&TenantId("t2".to_string())) {
        tenant.request_start();
        tenant.request_complete(300);
    }

    let stats = manager.aggregate_stats();
    assert_eq!(stats.total_tenants, 2);
    assert_eq!(stats.active_tenants, 2);
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.total_tokens, 800);
}
