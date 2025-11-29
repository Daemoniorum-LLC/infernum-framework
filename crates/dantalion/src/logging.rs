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
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));

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
