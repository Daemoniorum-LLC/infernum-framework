//! Claude Code usage metrics API.
//!
//! Provides endpoints to access Claude Code usage statistics, token consumption,
//! and cost estimates.
//!
//! ## Endpoints
//!
//! - `GET /api/claude/metrics` - Full usage dashboard data
//! - `GET /api/claude/metrics/daily` - Daily activity breakdown
//! - `GET /api/claude/metrics/costs` - Cost breakdown by model

use axum::{routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;
use tracing::{debug, info, warn};

use crate::claude_discovery;

// =============================================================================
// API Response Types
// =============================================================================

/// Full Claude Code metrics response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMetrics {
    /// Daily activity breakdown (messages, sessions, tool calls)
    pub daily_activity: Vec<DailyActivity>,
    /// Daily token usage by model
    pub daily_model_tokens: Vec<DailyModelTokens>,
    /// Per-model cumulative usage
    pub model_usage: HashMap<String, ModelUsage>,
    /// Aggregated totals
    pub totals: MetricsTotals,
    /// Subscription info (safe fields only)
    pub subscription: Option<SubscriptionInfo>,
    /// Longest session details
    pub longest_session: Option<LongestSession>,
    /// Activity by hour of day (0-23)
    pub hour_counts: HashMap<String, u64>,
}

/// Daily activity summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyActivity {
    pub date: String,
    pub message_count: u64,
    pub session_count: u64,
    pub tool_call_count: u64,
}

/// Daily token usage by model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyModelTokens {
    pub date: String,
    pub tokens_by_model: HashMap<String, u64>,
}

/// Per-model cumulative usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    /// Total input tokens
    pub input_tokens: u64,
    /// Total output tokens
    pub output_tokens: u64,
    /// Tokens read from prompt cache (huge savings!)
    pub cache_read_input_tokens: u64,
    /// Tokens written to prompt cache
    pub cache_creation_input_tokens: u64,
    /// Web search requests
    pub web_search_requests: u64,
    /// Calculated cost estimate
    pub estimated_cost_usd: f64,
}

/// Aggregated totals across all models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsTotals {
    pub total_messages: u64,
    pub total_sessions: u64,
    pub first_session_date: Option<String>,
    pub last_computed_date: Option<String>,
    /// Sum of all input tokens
    pub total_input_tokens: u64,
    /// Sum of all output tokens
    pub total_output_tokens: u64,
    /// Sum of all cache read tokens
    pub total_cache_read_tokens: u64,
    /// Sum of all cache creation tokens
    pub total_cache_creation_tokens: u64,
    /// Calculated total cost
    pub estimated_total_cost_usd: f64,
}

/// Subscription information (safe fields only - no tokens!).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInfo {
    /// Plan tier: "free", "pro", "max", etc.
    pub subscription_type: String,
    /// Rate limit tier identifier
    pub rate_limit_tier: String,
    /// Granted scopes
    pub scopes: Vec<String>,
}

/// Longest session details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongestSession {
    pub session_id: String,
    pub duration_ms: u64,
    pub message_count: u64,
    pub timestamp: String,
}

/// Cost breakdown response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Total estimated cost across all models
    pub total_cost_usd: f64,
    /// Cost per model
    pub by_model: HashMap<String, ModelCost>,
    /// Estimated savings from prompt caching
    pub cache_savings_usd: f64,
}

/// Cost details for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub model: String,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: f64,
    pub cache_creation_cost: f64,
    pub total_cost: f64,
    /// What it would have cost without caching
    pub cost_without_cache: f64,
}

// =============================================================================
// Raw JSON Types (for parsing stats-cache.json and .credentials.json)
// =============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawStatsCache {
    version: Option<u32>,
    last_computed_date: Option<String>,
    daily_activity: Option<Vec<RawDailyActivity>>,
    daily_model_tokens: Option<Vec<RawDailyModelTokens>>,
    model_usage: Option<HashMap<String, RawModelUsage>>,
    total_sessions: Option<u64>,
    total_messages: Option<u64>,
    first_session_date: Option<String>,
    longest_session: Option<RawLongestSession>,
    hour_counts: Option<HashMap<String, u64>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawDailyActivity {
    date: String,
    message_count: Option<u64>,
    session_count: Option<u64>,
    tool_call_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawDailyModelTokens {
    date: String,
    tokens_by_model: Option<HashMap<String, u64>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModelUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    web_search_requests: Option<u64>,
    cost_usd: Option<f64>,
    context_window: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLongestSession {
    session_id: Option<String>,
    duration: Option<u64>,
    message_count: Option<u64>,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawCredentials {
    claude_ai_oauth: Option<RawClaudeOauth>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawClaudeOauth {
    subscription_type: Option<String>,
    rate_limit_tier: Option<String>,
    scopes: Option<Vec<String>>,
    // Note: We intentionally do NOT expose access_token, refresh_token, etc.
}

// =============================================================================
// Router
// =============================================================================

/// Create the Claude metrics router.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/", get(get_metrics))
        .route("/daily", get(get_daily_metrics))
        .route("/costs", get(get_cost_breakdown))
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /api/claude/metrics
/// Returns full usage dashboard data.
pub async fn get_metrics() -> Json<ClaudeMetrics> {
    let metrics = load_metrics().await;
    Json(metrics)
}

/// GET /api/claude/metrics/daily
/// Returns daily activity breakdown.
pub async fn get_daily_metrics() -> Json<Vec<DailyActivity>> {
    let metrics = load_metrics().await;
    Json(metrics.daily_activity)
}

/// GET /api/claude/metrics/costs
/// Returns cost breakdown by model.
pub async fn get_cost_breakdown() -> Json<CostBreakdown> {
    let metrics = load_metrics().await;

    let mut by_model = HashMap::new();
    let mut total_cost = 0.0;
    let mut total_cache_savings = 0.0;

    for (model, usage) in &metrics.model_usage {
        let cost = calculate_model_cost(model, usage);
        total_cost += cost.total_cost;
        total_cache_savings += cost.cost_without_cache - cost.total_cost;
        by_model.insert(model.clone(), cost);
    }

    Json(CostBreakdown {
        total_cost_usd: total_cost,
        by_model,
        cache_savings_usd: total_cache_savings,
    })
}

// =============================================================================
// Data Loading
// =============================================================================

/// Load and aggregate metrics from all Claude Code data sources.
async fn load_metrics() -> ClaudeMetrics {
    // Discover all data sources
    let stats_paths = claude_discovery::get_all_stats_paths().await;

    if stats_paths.is_empty() {
        warn!("No Claude stats-cache.json files found");
        return empty_metrics();
    }

    info!("Aggregating metrics from {} sources", stats_paths.len());

    // Aggregated data structures
    let mut all_daily_activity: HashMap<String, DailyActivity> = HashMap::new();
    let mut all_daily_model_tokens: HashMap<String, HashMap<String, u64>> = HashMap::new();
    let mut all_model_usage: HashMap<String, ModelUsage> = HashMap::new();
    let mut total_messages = 0u64;
    let mut total_sessions = 0u64;
    let mut first_session_date: Option<String> = None;
    let mut last_computed_date: Option<String> = None;
    let mut longest_session: Option<LongestSession> = None;
    let mut all_hour_counts: HashMap<String, u64> = HashMap::new();

    // Load and aggregate from each source
    for (stats_path, label) in &stats_paths {
        let content = match fs::read_to_string(stats_path).await {
            Ok(c) => c,
            Err(e) => {
                debug!("Could not read {}: {}", stats_path.display(), e);
                continue;
            }
        };

        let stats: RawStatsCache = match serde_json::from_str(&content) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to parse {}: {}", stats_path.display(), e);
                continue;
            }
        };

        debug!("Loaded stats from source: {}", label);

        // Aggregate daily activity
        if let Some(daily) = stats.daily_activity {
            for d in daily {
                let entry = all_daily_activity
                    .entry(d.date.clone())
                    .or_insert_with(|| DailyActivity {
                        date: d.date.clone(),
                        message_count: 0,
                        session_count: 0,
                        tool_call_count: 0,
                    });
                entry.message_count += d.message_count.unwrap_or(0);
                entry.session_count += d.session_count.unwrap_or(0);
                entry.tool_call_count += d.tool_call_count.unwrap_or(0);
            }
        }

        // Aggregate daily model tokens
        if let Some(daily_tokens) = stats.daily_model_tokens {
            for d in daily_tokens {
                let entry = all_daily_model_tokens
                    .entry(d.date.clone())
                    .or_insert_with(HashMap::new);
                if let Some(tokens) = d.tokens_by_model {
                    for (model, count) in tokens {
                        *entry.entry(model).or_insert(0) += count;
                    }
                }
            }
        }

        // Aggregate model usage
        if let Some(usage) = stats.model_usage {
            for (model, u) in usage {
                let entry = all_model_usage
                    .entry(model.clone())
                    .or_insert_with(|| ModelUsage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        web_search_requests: 0,
                        estimated_cost_usd: 0.0,
                    });
                entry.input_tokens += u.input_tokens.unwrap_or(0);
                entry.output_tokens += u.output_tokens.unwrap_or(0);
                entry.cache_read_input_tokens += u.cache_read_input_tokens.unwrap_or(0);
                entry.cache_creation_input_tokens += u.cache_creation_input_tokens.unwrap_or(0);
                entry.web_search_requests += u.web_search_requests.unwrap_or(0);
            }
        }

        // Aggregate totals
        total_messages += stats.total_messages.unwrap_or(0);
        total_sessions += stats.total_sessions.unwrap_or(0);

        // Track earliest first session
        if let Some(fsd) = stats.first_session_date {
            if first_session_date.is_none() || first_session_date.as_ref().map(|d| &fsd < d).unwrap_or(false) {
                first_session_date = Some(fsd);
            }
        }

        // Track latest computed date
        if let Some(lcd) = stats.last_computed_date {
            if last_computed_date.is_none() || last_computed_date.as_ref().map(|d| &lcd > d).unwrap_or(false) {
                last_computed_date = Some(lcd);
            }
        }

        // Track longest session
        if let Some(ls) = stats.longest_session {
            let duration = ls.duration.unwrap_or(0);
            if longest_session.as_ref().map(|l| duration > l.duration_ms).unwrap_or(true) {
                longest_session = Some(LongestSession {
                    session_id: ls.session_id.unwrap_or_default(),
                    duration_ms: duration,
                    message_count: ls.message_count.unwrap_or(0),
                    timestamp: ls.timestamp.unwrap_or_default(),
                });
            }
        }

        // Aggregate hour counts
        if let Some(hours) = stats.hour_counts {
            for (hour, count) in hours {
                *all_hour_counts.entry(hour).or_insert(0) += count;
            }
        }
    }

    // Load subscription from main source
    let home_dir = dirs::home_dir();
    let subscription = if let Some(home) = home_dir {
        let credentials_path = home.join(".claude/.credentials.json");
        match fs::read_to_string(&credentials_path).await {
            Ok(content) => match serde_json::from_str::<RawCredentials>(&content) {
                Ok(creds) => creds.claude_ai_oauth.map(|oauth| SubscriptionInfo {
                    subscription_type: oauth.subscription_type.unwrap_or_else(|| "unknown".to_string()),
                    rate_limit_tier: oauth.rate_limit_tier.unwrap_or_else(|| "unknown".to_string()),
                    scopes: oauth.scopes.unwrap_or_default(),
                }),
                Err(_) => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    // Convert aggregated daily activity to sorted vec
    let mut daily_activity: Vec<DailyActivity> = all_daily_activity.into_values().collect();
    daily_activity.sort_by(|a, b| a.date.cmp(&b.date));

    // Convert aggregated daily model tokens to sorted vec
    let mut daily_model_tokens: Vec<DailyModelTokens> = all_daily_model_tokens
        .into_iter()
        .map(|(date, tokens_by_model)| DailyModelTokens {
            date,
            tokens_by_model,
        })
        .collect();
    daily_model_tokens.sort_by(|a, b| a.date.cmp(&b.date));

    // Calculate costs for aggregated model usage
    let mut total_input = 0u64;
    let mut total_output = 0u64;
    let mut total_cache_read = 0u64;
    let mut total_cache_creation = 0u64;
    let mut total_cost = 0.0;

    for (model, usage) in all_model_usage.iter_mut() {
        usage.estimated_cost_usd = calculate_cost(
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        );
        total_input += usage.input_tokens;
        total_output += usage.output_tokens;
        total_cache_read += usage.cache_read_input_tokens;
        total_cache_creation += usage.cache_creation_input_tokens;
        total_cost += usage.estimated_cost_usd;
    }

    // Build totals
    let totals = MetricsTotals {
        total_messages,
        total_sessions,
        first_session_date,
        last_computed_date,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        total_cache_read_tokens: total_cache_read,
        total_cache_creation_tokens: total_cache_creation,
        estimated_total_cost_usd: total_cost,
    };

    ClaudeMetrics {
        daily_activity,
        daily_model_tokens,
        model_usage: all_model_usage,
        totals,
        subscription,
        longest_session,
        hour_counts: all_hour_counts,
    }
}

/// Return empty metrics when no data is available.
fn empty_metrics() -> ClaudeMetrics {
    ClaudeMetrics {
        daily_activity: Vec::new(),
        daily_model_tokens: Vec::new(),
        model_usage: HashMap::new(),
        totals: MetricsTotals {
            total_messages: 0,
            total_sessions: 0,
            first_session_date: None,
            last_computed_date: None,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read_tokens: 0,
            total_cache_creation_tokens: 0,
            estimated_total_cost_usd: 0.0,
        },
        subscription: None,
        longest_session: None,
        hour_counts: HashMap::new(),
    }
}

// =============================================================================
// Cost Calculation
// =============================================================================

/// Calculate cost for a model based on token usage.
///
/// Pricing per million tokens (as of 2026-02):
/// - Opus: $15 input, $75 output, $1.50 cache read, $18.75 cache write
/// - Sonnet: $3 input, $15 output, $0.30 cache read, $3.75 cache write
/// - Haiku: $0.25 input, $1.25 output, $0.025 cache read, $0.3125 cache write
fn calculate_cost(
    model: &str,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_creation: u64,
) -> f64 {
    let (input_price, output_price, cache_read_price, cache_write_price) =
        get_model_pricing(model);

    let input_cost = (input as f64 / 1_000_000.0) * input_price;
    let output_cost = (output as f64 / 1_000_000.0) * output_price;
    let cache_read_cost = (cache_read as f64 / 1_000_000.0) * cache_read_price;
    let cache_write_cost = (cache_creation as f64 / 1_000_000.0) * cache_write_price;

    input_cost + output_cost + cache_read_cost + cache_write_cost
}

/// Calculate detailed cost breakdown for a model.
fn calculate_model_cost(model: &str, usage: &ModelUsage) -> ModelCost {
    let (input_price, output_price, cache_read_price, cache_write_price) =
        get_model_pricing(model);

    let input_cost = (usage.input_tokens as f64 / 1_000_000.0) * input_price;
    let output_cost = (usage.output_tokens as f64 / 1_000_000.0) * output_price;
    let cache_read_cost = (usage.cache_read_input_tokens as f64 / 1_000_000.0) * cache_read_price;
    let cache_creation_cost =
        (usage.cache_creation_input_tokens as f64 / 1_000_000.0) * cache_write_price;

    let total_cost = input_cost + output_cost + cache_read_cost + cache_creation_cost;

    // Cost without cache = all cache reads would have been full-price input
    let cost_without_cache = input_cost
        + output_cost
        + (usage.cache_read_input_tokens as f64 / 1_000_000.0) * input_price
        + cache_creation_cost;

    ModelCost {
        model: model.to_string(),
        input_cost,
        output_cost,
        cache_read_cost,
        cache_creation_cost,
        total_cost,
        cost_without_cache,
    }
}

/// Get pricing for a model (input, output, cache_read, cache_write per million tokens).
fn get_model_pricing(model: &str) -> (f64, f64, f64, f64) {
    let model_lower = model.to_lowercase();

    if model_lower.contains("opus") {
        (15.0, 75.0, 1.5, 18.75)
    } else if model_lower.contains("haiku") {
        (0.25, 1.25, 0.025, 0.3125)
    } else {
        // Default to Sonnet pricing
        (3.0, 15.0, 0.3, 3.75)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_cost_opus() {
        // 1M input, 1M output, 10M cache read, 500K cache write
        let cost = calculate_cost(
            "claude-opus-4-5-20251101",
            1_000_000,
            1_000_000,
            10_000_000,
            500_000,
        );

        // Expected: $15 + $75 + $15 + $9.375 = $114.375
        assert!((cost - 114.375).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cost_sonnet() {
        let cost = calculate_cost("claude-sonnet-4", 1_000_000, 1_000_000, 10_000_000, 500_000);

        // Expected: $3 + $15 + $3 + $1.875 = $22.875
        assert!((cost - 22.875).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cost_haiku() {
        let cost = calculate_cost("claude-haiku-4", 1_000_000, 1_000_000, 10_000_000, 500_000);

        // Expected: $0.25 + $1.25 + $0.25 + $0.15625 = $1.90625
        assert!((cost - 1.90625).abs() < 0.01);
    }

    #[test]
    fn test_get_model_pricing_defaults_to_sonnet() {
        let pricing = get_model_pricing("some-unknown-model");
        assert_eq!(pricing, (3.0, 15.0, 0.3, 3.75));
    }

    #[test]
    fn test_model_cost_cache_savings() {
        let usage = ModelUsage {
            input_tokens: 100_000,
            output_tokens: 100_000,
            cache_read_input_tokens: 10_000_000, // 10M cache reads
            cache_creation_input_tokens: 100_000,
            web_search_requests: 0,
            estimated_cost_usd: 0.0,
        };

        let cost = calculate_model_cost("claude-sonnet-4", &usage);

        // Cache savings = cache_read * (input_price - cache_read_price)
        // = 10M * ($3 - $0.30) / 1M = $27
        let expected_savings = (10_000_000.0 / 1_000_000.0) * (3.0 - 0.3);
        let actual_savings = cost.cost_without_cache - cost.total_cost;

        assert!((actual_savings - expected_savings).abs() < 0.01);
    }
}
