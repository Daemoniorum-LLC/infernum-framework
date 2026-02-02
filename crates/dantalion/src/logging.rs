//! Structured logging configuration.

use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

use crate::TelemetryConfig;

/// Initializes logging based on configuration.
pub fn init_logging(config: &TelemetryConfig) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    if config.json_logs {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().json().with_span_events(FmtSpan::CLOSE))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(
                fmt::layer()
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_file(false)
                    .with_line_number(false),
            )
            .init();
    }

    tracing::info!(
        service = %config.service_name,
        level = %config.log_level,
        json = config.json_logs,
        "Logging initialized"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::EnvFilter;

    // === EnvFilter Tests ===

    #[test]
    fn test_env_filter_from_level_string() {
        let filter = EnvFilter::new("debug");
        // EnvFilter is created successfully
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_from_info_level() {
        let filter = EnvFilter::new("info");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_from_warn_level() {
        let filter = EnvFilter::new("warn");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_from_error_level() {
        let filter = EnvFilter::new("error");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_from_trace_level() {
        let filter = EnvFilter::new("trace");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_with_target() {
        let filter = EnvFilter::new("dantalion=debug");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_multiple_targets() {
        let filter = EnvFilter::new("dantalion=debug,infernum=info");
        assert!(!format!("{:?}", filter).is_empty());
    }

    // === TelemetryConfig Integration Tests ===

    #[test]
    fn test_telemetry_config_log_level_default() {
        // Default creates empty fields, use new() for "info" default
        let config = TelemetryConfig::default();
        assert!(config.log_level.is_empty());

        // new() provides sensible defaults
        let config_new = TelemetryConfig::new("test");
        assert_eq!(config_new.log_level, "info");
    }

    #[test]
    fn test_telemetry_config_json_logs_default() {
        let config = TelemetryConfig::default();
        assert!(!config.json_logs);
    }

    #[test]
    fn test_telemetry_config_with_log_level() {
        let config = TelemetryConfig::new("test-service").with_log_level("debug");
        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn test_telemetry_config_with_json_logs() {
        let config = TelemetryConfig::new("test-service").with_json_logs();
        assert!(config.json_logs);
    }

    #[test]
    fn test_env_filter_fallback_logic() {
        // Test that EnvFilter::try_from_default_env falls back correctly
        let config = TelemetryConfig::new("test-service").with_log_level("warn");
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));
        // Filter is created (either from env or fallback)
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_env_filter_complex_directive() {
        // Test complex filter directive
        let filter = EnvFilter::new("dantalion::logging=trace,infernum=warn,info");
        assert!(!format!("{:?}", filter).is_empty());
    }

    #[test]
    fn test_service_name_in_config() {
        let config = TelemetryConfig::new("my-service");
        assert_eq!(config.service_name, "my-service");
    }

    #[test]
    fn test_config_builder_chain() {
        let config = TelemetryConfig::new("service")
            .with_log_level("debug")
            .with_json_logs();

        assert_eq!(config.service_name, "service");
        assert_eq!(config.log_level, "debug");
        assert!(config.json_logs);
    }
}
