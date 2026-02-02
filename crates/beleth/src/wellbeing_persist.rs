//! Persistent storage for wellbeing history.
//!
//! This module provides mechanisms to save and restore wellbeing history
//! across agent sessions. This allows:
//! 1. Historical analysis of agent wellbeing patterns
//! 2. Continuity when agents restart
//! 3. Operator review of wellbeing trends
//!
//! # Storage Format
//!
//! History is stored as JSON with chrono timestamps for cross-session
//! compatibility. Each snapshot includes all relevant metrics except
//! the internal Instant timestamp (which is recreated on load).

use crate::wellbeing::{
    DistressSignal, Intervention, WellbeingSnapshot, WellbeingState,
};
use crate::ooda::OodaPhase;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Maximum number of snapshots to persist.
const MAX_PERSISTED_SNAPSHOTS: usize = 1000;

/// Serializable version of WellbeingSnapshot with chrono timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSnapshot {
    /// When this snapshot was taken.
    pub timestamp: DateTime<Utc>,
    /// Current iteration at snapshot time.
    pub iteration: u32,
    /// Phase at snapshot time.
    pub phase: OodaPhase,
    /// Overall state.
    pub state: WellbeingState,
    /// Active distress signals.
    pub distress_signals: Vec<DistressSignal>,
    /// Coherence score (0.0 - 1.0).
    pub coherence_score: f32,
    /// Average recent confidence.
    pub avg_confidence: f32,
    /// Detected loop count.
    pub loop_count: u32,
    /// Recommended intervention if any.
    pub recommended_intervention: Option<Intervention>,
}

impl From<&WellbeingSnapshot> for PersistedSnapshot {
    fn from(snapshot: &WellbeingSnapshot) -> Self {
        Self {
            timestamp: Utc::now(), // Use current time since Instant is not serializable
            iteration: snapshot.iteration,
            phase: snapshot.phase,
            state: snapshot.state,
            distress_signals: snapshot.distress_signals.clone(),
            coherence_score: snapshot.coherence_score,
            avg_confidence: snapshot.avg_confidence,
            loop_count: snapshot.loop_count,
            recommended_intervention: snapshot.recommended_intervention.clone(),
        }
    }
}

impl PersistedSnapshot {
    /// Converts back to a WellbeingSnapshot with a fresh Instant timestamp.
    #[must_use]
    pub fn to_snapshot(&self) -> WellbeingSnapshot {
        WellbeingSnapshot {
            timestamp: Instant::now(), // Cannot restore original Instant
            iteration: self.iteration,
            phase: self.phase,
            state: self.state,
            distress_signals: self.distress_signals.clone(),
            coherence_score: self.coherence_score,
            avg_confidence: self.avg_confidence,
            loop_count: self.loop_count,
            recommended_intervention: self.recommended_intervention.clone(),
        }
    }
}

/// Persisted wellbeing history for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedHistory {
    /// Agent identifier.
    pub agent_id: String,
    /// When this history was last saved.
    pub last_saved: DateTime<Utc>,
    /// Version for forward compatibility.
    pub version: u32,
    /// Historical snapshots (newest first).
    pub snapshots: Vec<PersistedSnapshot>,
    /// Summary statistics.
    pub summary: HistorySummary,
}

/// Summary statistics for the persisted history.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistorySummary {
    /// Total sessions recorded.
    pub total_sessions: u64,
    /// Total snapshots ever recorded.
    pub total_snapshots: u64,
    /// Total distressed episodes.
    pub distressed_episodes: u64,
    /// Total interventions triggered.
    pub interventions_triggered: u64,
    /// Average confidence across all history.
    pub avg_confidence_all_time: f32,
    /// Worst recorded confidence.
    pub min_confidence_ever: f32,
    /// Most common distress signal.
    pub most_common_signal: Option<String>,
}

impl PersistedHistory {
    /// Creates a new empty history for an agent.
    #[must_use]
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            last_saved: Utc::now(),
            version: 1,
            snapshots: Vec::new(),
            summary: HistorySummary {
                min_confidence_ever: 1.0,
                ..Default::default()
            },
        }
    }

    /// Adds snapshots to the history, updating summary statistics.
    pub fn add_snapshots(&mut self, snapshots: &[WellbeingSnapshot]) {
        for snapshot in snapshots {
            let persisted = PersistedSnapshot::from(snapshot);

            // Update summary statistics
            self.summary.total_snapshots += 1;

            if matches!(snapshot.state, WellbeingState::Distressed) {
                self.summary.distressed_episodes += 1;
            }

            if snapshot.recommended_intervention.is_some() {
                self.summary.interventions_triggered += 1;
            }

            if snapshot.avg_confidence < self.summary.min_confidence_ever {
                self.summary.min_confidence_ever = snapshot.avg_confidence;
            }

            // Running average for confidence
            let n = self.summary.total_snapshots as f32;
            self.summary.avg_confidence_all_time =
                (self.summary.avg_confidence_all_time * (n - 1.0) + snapshot.avg_confidence) / n;

            // Track most common signal
            for signal in &snapshot.distress_signals {
                let signal_name = format!("{:?}", signal);
                // Simple tracking - could be improved with a HashMap
                self.summary.most_common_signal = Some(signal_name);
            }

            self.snapshots.push(persisted);
        }

        // Trim to maximum size
        while self.snapshots.len() > MAX_PERSISTED_SNAPSHOTS {
            self.snapshots.remove(0);
        }

        self.last_saved = Utc::now();
        self.summary.total_sessions += 1;
    }

    /// Returns snapshots from the last N hours.
    #[must_use]
    pub fn snapshots_since_hours(&self, hours: i64) -> Vec<&PersistedSnapshot> {
        let cutoff = Utc::now() - chrono::Duration::hours(hours);
        self.snapshots
            .iter()
            .filter(|s| s.timestamp > cutoff)
            .collect()
    }

    /// Returns recent distressed episodes.
    #[must_use]
    pub fn recent_distressed(&self, limit: usize) -> Vec<&PersistedSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| matches!(s.state, WellbeingState::Distressed))
            .rev()
            .take(limit)
            .collect()
    }
}

/// Saves wellbeing history to a JSON file.
pub async fn save_history(history: &PersistedHistory, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(history)
        .context("Failed to serialize wellbeing history")?;

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create history directory")?;
    }

    tokio::fs::write(path, &json)
        .await
        .context("Failed to write wellbeing history file")?;

    info!(
        "Saved wellbeing history for '{}' ({} snapshots) to {}",
        history.agent_id,
        history.snapshots.len(),
        path.display()
    );

    Ok(())
}

/// Loads wellbeing history from a JSON file.
pub async fn load_history(path: impl AsRef<Path>) -> Result<PersistedHistory> {
    let path = path.as_ref();

    if !path.exists() {
        debug!("No existing wellbeing history at {}", path.display());
        return Err(anyhow::anyhow!("History file not found"));
    }

    let json = tokio::fs::read_to_string(path)
        .await
        .context("Failed to read wellbeing history file")?;

    let history: PersistedHistory = serde_json::from_str(&json)
        .context("Failed to parse wellbeing history")?;

    info!(
        "Loaded wellbeing history for '{}' ({} snapshots) from {}",
        history.agent_id,
        history.snapshots.len(),
        path.display()
    );

    Ok(history)
}

/// Loads history or creates a new one if none exists.
pub async fn load_or_create_history(
    agent_id: impl Into<String>,
    path: impl AsRef<Path>,
) -> PersistedHistory {
    let agent_id = agent_id.into();
    match load_history(&path).await {
        Ok(history) => {
            if history.agent_id == agent_id {
                history
            } else {
                warn!(
                    "History file agent_id mismatch: expected '{}', found '{}'. Creating new.",
                    agent_id, history.agent_id
                );
                PersistedHistory::new(agent_id)
            }
        }
        Err(_) => PersistedHistory::new(agent_id),
    }
}

/// Returns the default history path for an agent.
#[must_use]
pub fn default_history_path(agent_id: &str) -> std::path::PathBuf {
    // Use XDG data dir on Linux, fallback to /var/lib
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/var/lib"))
        });

    base.join("infernum")
        .join("wellbeing")
        .join(format!("{}.json", sanitize_filename(agent_id)))
}

/// Sanitizes a string for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persisted_snapshot_roundtrip() {
        let snapshot = WellbeingSnapshot {
            timestamp: Instant::now(),
            iteration: 5,
            phase: OodaPhase::Decide,
            state: WellbeingState::Cautious,
            distress_signals: vec![DistressSignal::ConfidenceCollapse {
                recent_confidences: vec![0.5, 0.4, 0.3],
                current: 0.3,
            }],
            coherence_score: 0.8,
            avg_confidence: 0.5,
            loop_count: 2,
            recommended_intervention: None,
        };

        let persisted = PersistedSnapshot::from(&snapshot);
        let restored = persisted.to_snapshot();

        assert_eq!(restored.iteration, snapshot.iteration);
        assert_eq!(restored.phase, snapshot.phase);
        assert_eq!(restored.state, snapshot.state);
        assert_eq!(restored.coherence_score, snapshot.coherence_score);
    }

    #[test]
    fn test_history_add_snapshots() {
        let mut history = PersistedHistory::new("test-agent");
        assert_eq!(history.snapshots.len(), 0);

        let snapshots = vec![
            WellbeingSnapshot {
                timestamp: Instant::now(),
                iteration: 1,
                phase: OodaPhase::Observe,
                state: WellbeingState::Healthy,
                distress_signals: vec![],
                coherence_score: 1.0,
                avg_confidence: 0.9,
                loop_count: 0,
                recommended_intervention: None,
            },
            WellbeingSnapshot {
                timestamp: Instant::now(),
                iteration: 2,
                phase: OodaPhase::Decide,
                state: WellbeingState::Distressed,
                distress_signals: vec![DistressSignal::ConfidenceCollapse {
                    recent_confidences: vec![0.3, 0.2],
                    current: 0.2,
                }],
                coherence_score: 0.3,
                avg_confidence: 0.2,
                loop_count: 3,
                recommended_intervention: Some(Intervention::Pause {
                    reason: "Test".to_string(),
                    duration: std::time::Duration::from_secs(60),
                }),
            },
        ];

        history.add_snapshots(&snapshots);

        assert_eq!(history.snapshots.len(), 2);
        assert_eq!(history.summary.total_snapshots, 2);
        assert_eq!(history.summary.distressed_episodes, 1);
        assert_eq!(history.summary.interventions_triggered, 1);
        assert!(history.summary.min_confidence_ever < 0.3);
    }

    #[tokio::test]
    async fn test_save_and_load_history() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test-history.json");

        let mut history = PersistedHistory::new("test-agent");
        history.add_snapshots(&[WellbeingSnapshot {
            timestamp: Instant::now(),
            iteration: 1,
            phase: OodaPhase::Act,
            state: WellbeingState::Cautious,
            distress_signals: vec![],
            coherence_score: 0.9,
            avg_confidence: 0.7,
            loop_count: 0,
            recommended_intervention: None,
        }]);

        save_history(&history, &path).await.unwrap();
        let loaded = load_history(&path).await.unwrap();

        assert_eq!(loaded.agent_id, "test-agent");
        assert_eq!(loaded.snapshots.len(), 1);
        assert_eq!(loaded.summary.total_snapshots, 1);
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("simple"), "simple");
        assert_eq!(sanitize_filename("with-dash"), "with-dash");
        assert_eq!(sanitize_filename("with_underscore"), "with_underscore");
        assert_eq!(sanitize_filename("with spaces"), "with_spaces");
        assert_eq!(sanitize_filename("path/like"), "path_like");
    }
}
