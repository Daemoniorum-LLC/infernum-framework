//! Integration tests for Dantalion - the observability layer.
//!
//! Tests cover:
//! - TelemetryConfig creation and builder pattern
//! - Telemetry initialization and global access
//! - MetricsCollector request recording
//! - PrometheusRegistry metrics rendering
//! - InferenceMetrics atomic counters
//! - Timer elapsed time tracking
//! - ActiveRequestGuard RAII pattern
//! - ResearchTracker event recording
//! - SessionStats calculations
//! - EventListener pattern

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tempfile::tempdir;

use dantalion::{
    ActiveRequestGuard, ConsoleListener, EventType, GlobalStats,
    InferenceMetrics, JsonFileListener, MetricsCollector, ModelFamily,
    PrometheusRegistry, ResearchEvent, ResearchTracker, SessionId,
    SessionStats, Telemetry, TelemetryConfig, Timer,
};

// ============================================================================
// TelemetryConfig Tests
// ============================================================================

#[test]
fn test_telemetry_config_default() {
    let config = TelemetryConfig::default();

    assert!(config.service_name.is_empty());
    assert!(config.otlp_endpoint.is_none());
    assert!(!config.prometheus_enabled);
    assert!(config.prometheus_addr.is_none());
}

#[test]
fn test_telemetry_config_new() {
    let config = TelemetryConfig::new("infernum");

    assert_eq!(config.service_name, "infernum");
    assert_eq!(config.log_level, "info");
    assert!(!config.json_logs);
}

#[test]
fn test_telemetry_config_with_otlp() {
    let config = TelemetryConfig::new("test")
        .with_otlp("http://localhost:4317");

    assert_eq!(
        config.otlp_endpoint,
        Some("http://localhost:4317".to_string())
    );
}

#[test]
fn test_telemetry_config_with_prometheus() {
    let config = TelemetryConfig::new("test")
        .with_prometheus("0.0.0.0:9090");

    assert!(config.prometheus_enabled);
    assert_eq!(
        config.prometheus_addr,
        Some("0.0.0.0:9090".to_string())
    );
}

#[test]
fn test_telemetry_config_with_log_level() {
    let config = TelemetryConfig::new("test")
        .with_log_level("debug");

    assert_eq!(config.log_level, "debug");
}

#[test]
fn test_telemetry_config_with_json_logs() {
    let config = TelemetryConfig::new("test")
        .with_json_logs();

    assert!(config.json_logs);
}

#[test]
fn test_telemetry_config_builder_chain() {
    let config = TelemetryConfig::new("my-service")
        .with_otlp("http://jaeger:4317")
        .with_prometheus("0.0.0.0:8080")
        .with_log_level("trace")
        .with_json_logs();

    assert_eq!(config.service_name, "my-service");
    assert_eq!(
        config.otlp_endpoint,
        Some("http://jaeger:4317".to_string())
    );
    assert!(config.prometheus_enabled);
    assert_eq!(config.log_level, "trace");
    assert!(config.json_logs);
}

#[test]
fn test_telemetry_config_clone() {
    let config = TelemetryConfig::new("clone-test")
        .with_otlp("http://localhost:4317")
        .with_json_logs();

    let cloned = config.clone();
    assert_eq!(cloned.service_name, config.service_name);
    assert_eq!(cloned.otlp_endpoint, config.otlp_endpoint);
    assert_eq!(cloned.json_logs, config.json_logs);
}

#[test]
fn test_telemetry_config_debug() {
    let config = TelemetryConfig::new("debug-test");
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("TelemetryConfig"));
    assert!(debug_str.contains("debug-test"));
}

// ============================================================================
// Telemetry Tests
// ============================================================================

#[test]
fn test_telemetry_init() {
    let config = TelemetryConfig::new("test-service");
    let telemetry = Telemetry::init(config);

    assert_eq!(telemetry.metrics.inference().requests(), 0);
}

#[test]
fn test_telemetry_global() {
    let config = TelemetryConfig::new("global-test");
    let _telemetry = Telemetry::init(config);

    let global = Telemetry::global();
    assert!(global.is_some());
}

#[test]
fn test_telemetry_metrics_access() {
    let config = TelemetryConfig::new("metrics-test");
    let telemetry = Telemetry::init(config);

    telemetry.metrics.inference().record_request(100, 50);
    telemetry.metrics.inference().record_request(200, 100);
    telemetry.metrics.record_error("chat", "test-model", "timeout");

    assert_eq!(telemetry.metrics.inference().requests(), 2);
    assert_eq!(telemetry.metrics.inference().prompt_tokens(), 300);
    assert_eq!(telemetry.metrics.inference().tokens_generated(), 150);
    assert_eq!(telemetry.metrics.inference().errors(), 1);
}

// ============================================================================
// MetricsCollector Tests
// ============================================================================

#[test]
fn test_metrics_collector_new() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    assert_eq!(collector.inference().requests(), 0);
}

#[test]
fn test_metrics_collector_chat_request() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    collector.record_chat_request(100, 50, 0.5, "test-model");

    assert_eq!(collector.inference().requests(), 1);
    assert_eq!(collector.inference().prompt_tokens(), 100);
    assert_eq!(collector.inference().tokens_generated(), 50);
}

#[test]
fn test_metrics_collector_completion_request() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    collector.record_completion_request(200, 100, 0.3, "llama-3");

    assert_eq!(collector.inference().requests(), 1);
    assert_eq!(collector.inference().prompt_tokens(), 200);
    assert_eq!(collector.inference().tokens_generated(), 100);
}

#[test]
fn test_metrics_collector_embedding_request() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    collector.record_embedding_request(500, 0.1, "e5-large", 10);

    assert_eq!(collector.inference().requests(), 1);
    assert_eq!(collector.inference().prompt_tokens(), 500);
}

#[test]
fn test_metrics_collector_error() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    collector.record_error("chat", "gpt-4", "rate_limit");
    collector.record_error("chat", "gpt-4", "rate_limit");

    assert_eq!(collector.inference().errors(), 2);
}

#[test]
fn test_metrics_collector_multiple_requests() {
    let config = TelemetryConfig::default();
    let collector = MetricsCollector::new(&config);

    for i in 0..10 {
        collector.record_chat_request(100 + i, 50 + i, 0.1, "model");
    }

    assert_eq!(collector.inference().requests(), 10);
}

#[test]
fn test_metrics_collector_prometheus_render() {
    let config = TelemetryConfig::new("prometheus-test")
        .with_prometheus("0.0.0.0:9090");
    let collector = MetricsCollector::new(&config);

    collector.record_chat_request(50, 25, 0.1, "test-model");

    let output = collector.render_prometheus();

    assert!(output.contains("infernum_requests_total"));
    assert!(output.contains("infernum_active_requests"));
    assert!(output.contains("infernum_model_loaded"));
}

// ============================================================================
// PrometheusRegistry Tests
// ============================================================================

#[test]
fn test_prometheus_registry_new() {
    let registry = PrometheusRegistry::new();
    let output = registry.render();
    assert!(output.contains("infernum_active_requests 0"));
}

#[test]
fn test_prometheus_registry_active_requests() {
    let registry = PrometheusRegistry::new();

    registry.inc_active_requests();
    registry.inc_active_requests();
    let output = registry.render();
    assert!(output.contains("infernum_active_requests 2"));

    registry.dec_active_requests();
    let output = registry.render();
    assert!(output.contains("infernum_active_requests 1"));
}

#[test]
fn test_prometheus_registry_model_loaded() {
    let registry = PrometheusRegistry::new();

    let output = registry.render();
    assert!(output.contains("infernum_model_loaded 0"));

    registry.set_model_loaded(true);
    let output = registry.render();
    assert!(output.contains("infernum_model_loaded 1"));

    registry.set_model_loaded(false);
    let output = registry.render();
    assert!(output.contains("infernum_model_loaded 0"));
}

#[test]
fn test_prometheus_registry_render() {
    let registry = PrometheusRegistry::new();

    registry.set_model_loaded(true);

    let output = registry.render();

    assert!(output.contains("# HELP infernum_requests_total"));
    assert!(output.contains("# TYPE infernum_requests_total counter"));
    assert!(output.contains("# HELP infernum_active_requests"));
    assert!(output.contains("# HELP infernum_model_loaded"));
    assert!(output.contains("infernum_model_loaded 1"));
}

#[test]
fn test_prometheus_registry_default() {
    let registry = PrometheusRegistry::default();
    let output = registry.render();
    assert!(output.contains("infernum_active_requests 0"));
}

// ============================================================================
// InferenceMetrics Tests
// ============================================================================

#[test]
fn test_inference_metrics_default() {
    let metrics = InferenceMetrics::default();

    assert_eq!(metrics.requests(), 0);
    assert_eq!(metrics.tokens_generated(), 0);
    assert_eq!(metrics.prompt_tokens(), 0);
    assert_eq!(metrics.errors(), 0);
}

#[test]
fn test_inference_metrics_record_request() {
    let metrics = InferenceMetrics::default();

    metrics.record_request(100, 50);
    metrics.record_request(200, 75);

    assert_eq!(metrics.requests(), 2);
    assert_eq!(metrics.prompt_tokens(), 300);
    assert_eq!(metrics.tokens_generated(), 125);
}

#[test]
fn test_inference_metrics_record_error() {
    let metrics = InferenceMetrics::default();

    metrics.record_error();
    metrics.record_error();
    metrics.record_error();

    assert_eq!(metrics.errors(), 3);
}

#[test]
fn test_inference_metrics_concurrent() {
    let metrics = Arc::new(InferenceMetrics::default());

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                for _ in 0..100 {
                    m.record_request(10, 5);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(metrics.requests(), 1000);
    assert_eq!(metrics.prompt_tokens(), 10000);
    assert_eq!(metrics.tokens_generated(), 5000);
}

// ============================================================================
// Timer Tests
// ============================================================================

#[test]
fn test_timer_start() {
    let timer = Timer::start("test-operation");
    assert!(timer.elapsed_ms() >= 0.0);
}

#[test]
fn test_timer_elapsed() {
    let timer = Timer::start("sleep-test");
    thread::sleep(Duration::from_millis(50));

    let elapsed_ms = timer.elapsed_ms();
    let elapsed_secs = timer.elapsed_secs();

    assert!(elapsed_ms >= 50.0);
    assert!(elapsed_secs >= 0.05);
}

#[test]
fn test_timer_multiple() {
    let timer1 = Timer::start("timer1");
    thread::sleep(Duration::from_millis(10));
    let timer2 = Timer::start("timer2");
    thread::sleep(Duration::from_millis(10));

    // Timer1 should have elapsed more time than timer2
    assert!(timer1.elapsed_ms() > timer2.elapsed_ms());
}

// ============================================================================
// ActiveRequestGuard Tests
// ============================================================================

#[test]
fn test_active_request_guard() {
    let registry = PrometheusRegistry::new();

    {
        let _guard = ActiveRequestGuard::new(&registry);
        let output = registry.render();
        assert!(output.contains("infernum_active_requests 1"));
    }

    let output = registry.render();
    assert!(output.contains("infernum_active_requests 0"));
}

#[test]
fn test_active_request_guard_nested() {
    let registry = PrometheusRegistry::new();

    {
        let _guard1 = ActiveRequestGuard::new(&registry);
        let output = registry.render();
        assert!(output.contains("infernum_active_requests 1"));

        {
            let _guard2 = ActiveRequestGuard::new(&registry);
            let output = registry.render();
            assert!(output.contains("infernum_active_requests 2"));
        }

        let output = registry.render();
        assert!(output.contains("infernum_active_requests 1"));
    }

    let output = registry.render();
    assert!(output.contains("infernum_active_requests 0"));
}

// ============================================================================
// SessionId Tests
// ============================================================================

#[test]
fn test_session_id_from_str() {
    let id: SessionId = "my-session".into();
    assert_eq!(id.0, "my-session");
}

#[test]
fn test_session_id_default() {
    let id = SessionId::default();
    assert!(id.0.is_empty());
}

#[test]
fn test_session_id_clone() {
    let id: SessionId = "clone-test".into();
    let cloned = id.clone();
    assert_eq!(id.0, cloned.0);
}

#[test]
fn test_session_id_equality() {
    let id1: SessionId = "same".into();
    let id2: SessionId = "same".into();
    let id3: SessionId = "different".into();

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

// ============================================================================
// ModelFamily Tests
// ============================================================================

#[test]
fn test_model_family_variants() {
    let families = vec![
        ModelFamily::Claude,
        ModelFamily::Gpt,
        ModelFamily::Gemini,
        ModelFamily::Llama,
        ModelFamily::Mistral,
        ModelFamily::Other("custom".to_string()),
    ];

    assert_eq!(families.len(), 6);
}

#[test]
fn test_model_family_clone() {
    let family = ModelFamily::Claude;
    let cloned = family.clone();
    assert_eq!(family, cloned);
}

#[test]
fn test_model_family_serialization() {
    let family = ModelFamily::Claude;
    let json = serde_json::to_string(&family).expect("serialize");
    assert!(json.contains("Claude"));
}

// ============================================================================
// SessionStats Tests
// ============================================================================

#[test]
fn test_session_stats_new() {
    let stats = SessionStats::new("test-session".into());

    assert_eq!(stats.session_id.0, "test-session");
    assert_eq!(stats.total_requests, 0);
}

#[test]
fn test_session_stats_average_latency() {
    let mut stats = SessionStats::new("test".into());

    assert_eq!(stats.average_latency_ms(), 0.0);

    stats.total_requests = 10;
    stats.total_latency_ms = 1000;

    assert_eq!(stats.average_latency_ms(), 100.0);
}

#[test]
fn test_session_stats_success_rate() {
    let mut stats = SessionStats::new("test".into());

    assert_eq!(stats.success_rate(), 0.0);

    stats.total_requests = 10;
    stats.successful_requests = 9;

    assert!((stats.success_rate() - 0.9).abs() < 0.001);
}

#[test]
fn test_session_stats_avg_tokens() {
    let mut stats = SessionStats::new("test".into());

    let (input, output) = stats.avg_tokens_per_request();
    assert_eq!(input, 0.0);
    assert_eq!(output, 0.0);

    stats.total_requests = 10;
    stats.total_input_tokens = 1000;
    stats.total_output_tokens = 500;

    let (input, output) = stats.avg_tokens_per_request();
    assert_eq!(input, 100.0);
    assert_eq!(output, 50.0);
}

#[test]
fn test_session_stats_joy_friction_ratio() {
    let mut stats = SessionStats::new("test".into());

    // No frictions = infinity
    assert!(stats.joy_friction_ratio().is_infinite());

    stats.total_joys = 10;
    stats.total_frictions = 4;

    assert_eq!(stats.joy_friction_ratio(), 2.5);
}

#[test]
fn test_session_stats_default() {
    let stats = SessionStats::default();

    assert!(stats.session_id.0.is_empty());
    assert!(stats.start_time.is_none());
    assert!(stats.end_time.is_none());
}

// ============================================================================
// ResearchTracker Tests
// ============================================================================

#[test]
fn test_research_tracker_new() {
    let tracker = ResearchTracker::new();

    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 0);
    assert_eq!(global.total_tokens, 0);
    assert_eq!(global.total_checkpoints, 0);
}

#[test]
fn test_research_tracker_start_session() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session-001".to_string()),
        "infernum",
        "agent-001",
        "claude-opus-4",
    );

    assert_eq!(session_id.0, "session-001");

    let stats = tracker.get_stats(&session_id);
    assert!(stats.is_some());
    assert!(stats.unwrap().start_time.is_some());
}

#[test]
fn test_research_tracker_end_session() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session-002".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.end_session(&session_id);

    let stats = tracker.get_stats(&session_id);
    assert!(stats.is_some());
    assert!(stats.unwrap().end_time.is_some());
}

#[test]
fn test_research_tracker_record_inference() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_inference(
        &session_id,
        100,
        50,
        150,
        true,
        "project",
        "claude-opus-4",
    );

    let stats = tracker.get_stats(&session_id).unwrap();
    assert_eq!(stats.total_requests, 1);
    assert_eq!(stats.successful_requests, 1);
    assert_eq!(stats.total_input_tokens, 100);
    assert_eq!(stats.total_output_tokens, 50);
    assert_eq!(stats.total_latency_ms, 150);

    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 1);
    assert_eq!(global.total_tokens, 150);
}

#[test]
fn test_research_tracker_record_failed_inference() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_inference(&session_id, 100, 0, 50, false, "project", "model");

    let stats = tracker.get_stats(&session_id).unwrap();
    assert_eq!(stats.total_requests, 1);
    assert_eq!(stats.successful_requests, 0);
    assert_eq!(stats.failed_requests, 1);
}

#[test]
fn test_research_tracker_record_checkpoint() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_checkpoint(&session_id, "checkpoint-001", 5, 2);
    tracker.record_checkpoint(&session_id, "checkpoint-002", 3, 1);

    let stats = tracker.get_stats(&session_id).unwrap();
    assert_eq!(stats.checkpoints_created, 2);
    assert_eq!(stats.total_joys, 8);
    assert_eq!(stats.total_frictions, 3);

    let global = tracker.global_stats();
    assert_eq!(global.total_checkpoints, 2);
}

#[test]
fn test_research_tracker_record_phase_transition() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_phase_transition(&session_id, "analysis", "implementation");

    let events = tracker.get_events(&session_id);
    assert!(events.len() >= 2); // SessionStart + PhaseTransition
}

#[test]
fn test_research_tracker_record_pattern() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_pattern(&session_id, "Builder Pattern");
    tracker.record_pattern(&session_id, "Strategy Pattern");

    let stats = tracker.get_stats(&session_id).unwrap();
    assert_eq!(stats.patterns_discovered, 2);
}

#[test]
fn test_research_tracker_record_friction() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );

    tracker.record_friction(&session_id, "api-complexity", "medium", false);
    tracker.record_friction(&session_id, "documentation-gap", "high", true);

    let events = tracker.get_events(&session_id);
    let friction_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e.event_type, EventType::FrictionEncountered { .. }))
        .collect();

    assert_eq!(friction_events.len(), 2);
}

#[test]
fn test_research_tracker_get_events() {
    let tracker = ResearchTracker::new();

    let session_id = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );
    tracker.record_inference(&session_id, 100, 50, 150, true, "project", "model");
    tracker.end_session(&session_id);

    let events = tracker.get_events(&session_id);
    assert_eq!(events.len(), 3); // Start + Inference + End
}

#[test]
fn test_research_tracker_all_stats() {
    let tracker = ResearchTracker::new();

    tracker.start_session(SessionId("session-1".to_string()), "project", "agent", "model");
    tracker.start_session(SessionId("session-2".to_string()), "project", "agent", "model");
    tracker.start_session(SessionId("session-3".to_string()), "project", "agent", "model");

    let all_stats = tracker.all_stats();
    assert_eq!(all_stats.len(), 3);
}

#[test]
fn test_research_tracker_global_stats() {
    let tracker = ResearchTracker::new();

    let session1 = tracker.start_session(
        SessionId("s1".to_string()),
        "p",
        "a",
        "m",
    );
    let session2 = tracker.start_session(
        SessionId("s2".to_string()),
        "p",
        "a",
        "m",
    );

    tracker.record_inference(&session1, 100, 50, 100, true, "p", "m");
    tracker.record_inference(&session2, 200, 100, 200, true, "p", "m");
    tracker.record_checkpoint(&session1, "c1", 3, 1);

    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 2);
    assert_eq!(global.total_tokens, 450);
    assert_eq!(global.total_checkpoints, 1);
    assert_eq!(global.active_sessions, 2);
}

#[test]
fn test_research_tracker_export_json() {
    let tracker = ResearchTracker::new();

    let session = tracker.start_session(
        SessionId("session".to_string()),
        "project",
        "agent",
        "model",
    );
    tracker.record_inference(&session, 100, 50, 100, true, "project", "model");

    let json = tracker.export_json();
    assert!(!json.is_empty());
    assert!(json.contains("session"));
}

#[test]
fn test_research_tracker_default() {
    let tracker = ResearchTracker::default();

    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 0);
}

// ============================================================================
// EventListener Tests
// ============================================================================

#[test]
fn test_console_listener() {
    let listener = ConsoleListener;

    let event = ResearchEvent {
        id: "test".to_string(),
        session_id: SessionId("session".to_string()),
        event_type: EventType::SessionStart,
        timestamp: chrono::Utc::now(),
        project: "test".to_string(),
        phase: "init".to_string(),
        agent_id: "agent".to_string(),
        model_id: "model".to_string(),
        model_family: ModelFamily::Claude,
        metadata: HashMap::new(),
    };

    // Just verify it doesn't panic
    use dantalion::EventListener;
    listener.on_event(&event);
}

#[test]
fn test_json_file_listener() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("events.jsonl");

    let listener = JsonFileListener::new(&path);

    let event = ResearchEvent {
        id: "test".to_string(),
        session_id: SessionId("session".to_string()),
        event_type: EventType::SessionStart,
        timestamp: chrono::Utc::now(),
        project: "test".to_string(),
        phase: "init".to_string(),
        agent_id: "agent".to_string(),
        model_id: "model".to_string(),
        model_family: ModelFamily::Claude,
        metadata: HashMap::new(),
    };

    use dantalion::EventListener;
    listener.on_event(&event);

    // Verify file was created and has content
    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("session"));
}

// ============================================================================
// EventType Tests
// ============================================================================

#[test]
fn test_event_type_session_start() {
    let event_type = EventType::SessionStart;
    let json = serde_json::to_string(&event_type).unwrap();
    assert!(json.contains("SessionStart"));
}

#[test]
fn test_event_type_inference_request() {
    let event_type = EventType::InferenceRequest {
        input_tokens: 100,
        output_tokens: 50,
        latency_ms: 150,
        success: true,
    };

    let json = serde_json::to_string(&event_type).unwrap();
    assert!(json.contains("InferenceRequest"));
    assert!(json.contains("100"));
}

#[test]
fn test_event_type_checkpoint_created() {
    let event_type = EventType::CheckpointCreated {
        checkpoint_id: "ckpt-001".to_string(),
        joy_count: 5,
        friction_count: 2,
    };

    let json = serde_json::to_string(&event_type).unwrap();
    assert!(json.contains("CheckpointCreated"));
    assert!(json.contains("ckpt-001"));
}

#[test]
fn test_event_type_custom() {
    let mut data = HashMap::new();
    data.insert("key".to_string(), "value".to_string());

    let event_type = EventType::Custom {
        name: "CustomEvent".to_string(),
        data,
    };

    let json = serde_json::to_string(&event_type).unwrap();
    assert!(json.contains("Custom"));
    assert!(json.contains("CustomEvent"));
}

// ============================================================================
// GlobalStats Tests
// ============================================================================

#[test]
fn test_global_stats_clone() {
    let stats = GlobalStats {
        total_requests: 100,
        total_tokens: 5000,
        total_checkpoints: 10,
        active_sessions: 5,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_requests, 100);
    assert_eq!(cloned.total_tokens, 5000);
}

#[test]
fn test_global_stats_debug() {
    let stats = GlobalStats {
        total_requests: 100,
        total_tokens: 5000,
        total_checkpoints: 10,
        active_sessions: 5,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("GlobalStats"));
    assert!(debug_str.contains("100"));
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_telemetry_workflow() {
    // 1. Create telemetry config
    let config = TelemetryConfig::new("infernum-test")
        .with_prometheus("0.0.0.0:9090")
        .with_log_level("info");

    // 2. Initialize telemetry
    let telemetry = Telemetry::init(config);

    // 3. Record some metrics
    telemetry.metrics.record_chat_request(100, 50, 0.1, "claude");
    telemetry.metrics.record_chat_request(200, 100, 0.2, "claude");
    telemetry.metrics.record_error("chat", "claude", "timeout");

    // 4. Verify metrics
    assert_eq!(telemetry.metrics.inference().requests(), 2);
    assert_eq!(telemetry.metrics.inference().errors(), 1);

    // 5. Render prometheus
    let output = telemetry.metrics.render_prometheus();
    assert!(output.contains("infernum_requests_total"));
}

#[test]
fn test_research_session_workflow() {
    // 1. Create tracker
    let tracker = ResearchTracker::new();

    // 2. Start session
    let session_id = tracker.start_session(
        SessionId("jormungandr-session-001".to_string()),
        "infernum",
        "claude-agent",
        "claude-opus-4",
    );

    // 3. Record multiple inference requests
    for i in 0u32..10 {
        let success = i % 3 != 0;
        tracker.record_inference(
            &session_id,
            100 + i * 10,      // input_tokens: u32
            50 + i * 5,        // output_tokens: u32
            100 + (i as u64) * 20, // latency_ms: u64
            success,
            "infernum",
            "claude-opus-4",
        );
    }

    // 4. Record checkpoints
    tracker.record_checkpoint(&session_id, "checkpoint-001", 5, 2);
    tracker.record_checkpoint(&session_id, "checkpoint-002", 3, 1);

    // 5. Record patterns and frictions
    tracker.record_pattern(&session_id, "Builder Pattern");
    tracker.record_friction(&session_id, "api-complexity", "medium", false);

    // 6. Record phase transition
    tracker.record_phase_transition(&session_id, "analysis", "implementation");

    // 7. End session
    tracker.end_session(&session_id);

    // 8. Verify stats
    let stats = tracker.get_stats(&session_id).unwrap();
    assert_eq!(stats.total_requests, 10);
    assert_eq!(stats.checkpoints_created, 2);
    assert_eq!(stats.patterns_discovered, 1);
    assert!(stats.end_time.is_some());

    // 9. Verify global stats
    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 10);
    assert_eq!(global.total_checkpoints, 2);

    // 10. Export JSON
    let json = tracker.export_json();
    assert!(!json.is_empty());
}

#[test]
fn test_concurrent_tracking() {
    let tracker = Arc::new(ResearchTracker::new());

    let handles: Vec<_> = (0..5)
        .map(|i| {
            let t = Arc::clone(&tracker);
            thread::spawn(move || {
                let session_id = t.start_session(
                    SessionId(format!("session-{}", i)),
                    "project",
                    "agent",
                    "model",
                );

                for j in 0..10 {
                    t.record_inference(
                        &session_id,
                        100,
                        50,
                        100,
                        j % 2 == 0,
                        "project",
                        "model",
                    );
                }

                t.end_session(&session_id);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let global = tracker.global_stats();
    assert_eq!(global.total_requests, 50);
    assert_eq!(global.active_sessions, 5);
}

#[test]
fn test_metrics_with_request_guard() {
    let config = TelemetryConfig::new("guard-test");
    let collector = MetricsCollector::new(&config);
    let registry = collector.prometheus();

    // Simulate concurrent requests with guards
    {
        let _guard1 = ActiveRequestGuard::new(registry);
        let _guard2 = ActiveRequestGuard::new(registry);
        let _guard3 = ActiveRequestGuard::new(registry);

        let output = registry.render();
        assert!(output.contains("infernum_active_requests 3"));
    }

    // All guards dropped
    let output = registry.render();
    assert!(output.contains("infernum_active_requests 0"));
}
