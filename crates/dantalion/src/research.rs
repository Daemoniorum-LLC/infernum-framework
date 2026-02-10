//! Jormungandr Research Activity Capture
//!
//! This module captures and logs all research activities from the Jormungandr
//! initiative, including agent conversations, token usage, latency metrics,
//! and checkpoint completion rates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Research session identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Model family for tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    /// Anthropic Claude.
    Claude,
    /// OpenAI GPT.
    Gpt,
    /// Google Gemini.
    Gemini,
    /// Meta Llama.
    Llama,
    /// Mistral AI.
    Mistral,
    /// Other/local models.
    Other(String),
}

/// A research activity event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchEvent {
    /// Event ID.
    pub id: String,
    /// Session ID.
    pub session_id: SessionId,
    /// Event type.
    pub event_type: EventType,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Project being worked on.
    pub project: String,
    /// Conversion phase.
    pub phase: String,
    /// Agent ID.
    pub agent_id: String,
    /// Model used.
    pub model_id: String,
    /// Model family.
    pub model_family: ModelFamily,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Types of research events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Session started.
    SessionStart,
    /// Session ended.
    SessionEnd,
    /// Inference request made.
    InferenceRequest {
        /// Input tokens.
        input_tokens: u32,
        /// Output tokens.
        output_tokens: u32,
        /// Latency in milliseconds.
        latency_ms: u64,
        /// Whether request succeeded.
        success: bool,
    },
    /// Checkpoint created.
    CheckpointCreated {
        /// Checkpoint ID.
        checkpoint_id: String,
        /// Joy count.
        joy_count: u32,
        /// Friction count.
        friction_count: u32,
    },
    /// Phase transition.
    PhaseTransition {
        /// Previous phase.
        from_phase: String,
        /// New phase.
        to_phase: String,
    },
    /// Pattern discovered.
    PatternDiscovered {
        /// Pattern name.
        pattern_name: String,
    },
    /// Friction encountered.
    FrictionEncountered {
        /// Friction category.
        category: String,
        /// Severity.
        severity: String,
        /// Whether it was blocking.
        blocking: bool,
    },
    /// Model switched.
    ModelSwitch {
        /// Previous model.
        from_model: String,
        /// New model.
        to_model: String,
        /// Reason for switch.
        reason: String,
    },
    /// Custom event.
    Custom {
        /// Event name.
        name: String,
        /// Event data.
        data: HashMap<String, String>,
    },
}

/// Statistics for a research session.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Session ID.
    pub session_id: SessionId,
    /// Start time.
    pub start_time: Option<DateTime<Utc>>,
    /// End time.
    pub end_time: Option<DateTime<Utc>>,
    /// Total inference requests.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Total input tokens.
    pub total_input_tokens: u64,
    /// Total output tokens.
    pub total_output_tokens: u64,
    /// Total latency (ms).
    pub total_latency_ms: u64,
    /// Checkpoints created.
    pub checkpoints_created: u32,
    /// Total joys.
    pub total_joys: u32,
    /// Total frictions.
    pub total_frictions: u32,
    /// Patterns discovered.
    pub patterns_discovered: u32,
    /// Models used.
    pub models_used: Vec<String>,
}

impl SessionStats {
    /// Creates new session stats.
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            ..Default::default()
        }
    }

    /// Returns average latency in milliseconds.
    pub fn average_latency_ms(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_latency_ms as f64 / self.total_requests as f64
    }

    /// Returns success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.successful_requests as f64 / self.total_requests as f64
    }

    /// Returns average tokens per request.
    pub fn avg_tokens_per_request(&self) -> (f64, f64) {
        if self.total_requests == 0 {
            return (0.0, 0.0);
        }
        (
            self.total_input_tokens as f64 / self.total_requests as f64,
            self.total_output_tokens as f64 / self.total_requests as f64,
        )
    }

    /// Returns joy/friction ratio.
    pub fn joy_friction_ratio(&self) -> f64 {
        if self.total_frictions == 0 {
            return f64::INFINITY;
        }
        self.total_joys as f64 / self.total_frictions as f64
    }
}

/// Research activity tracker.
pub struct ResearchTracker {
    /// Events by session.
    events: RwLock<HashMap<SessionId, Vec<ResearchEvent>>>,
    /// Session stats.
    stats: RwLock<HashMap<SessionId, SessionStats>>,
    /// Global counters.
    global_requests: AtomicU64,
    global_tokens: AtomicU64,
    global_checkpoints: AtomicU64,
    /// Event listeners.
    listeners: RwLock<Vec<Arc<dyn EventListener>>>,
}

impl ResearchTracker {
    /// Creates a new research tracker.
    pub fn new() -> Self {
        Self {
            events: RwLock::new(HashMap::new()),
            stats: RwLock::new(HashMap::new()),
            global_requests: AtomicU64::new(0),
            global_tokens: AtomicU64::new(0),
            global_checkpoints: AtomicU64::new(0),
            listeners: RwLock::new(Vec::new()),
        }
    }

    /// Starts a new research session.
    pub fn start_session(
        &self,
        session_id: impl Into<SessionId>,
        project: impl Into<String>,
        agent_id: impl Into<String>,
        model_id: impl Into<String>,
    ) -> SessionId {
        let id = session_id.into();

        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: id.clone(),
            event_type: EventType::SessionStart,
            timestamp: Utc::now(),
            project: project.into(),
            phase: "init".to_string(),
            agent_id: agent_id.into(),
            model_id: model_id.into(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);

        let mut stats = self.stats.write();
        let session_stats = stats
            .entry(id.clone())
            .or_insert_with(|| SessionStats::new(id.clone()));
        session_stats.start_time = Some(Utc::now());

        id
    }

    /// Ends a research session.
    pub fn end_session(&self, session_id: &SessionId) {
        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::SessionEnd,
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);

        let mut stats = self.stats.write();
        if let Some(session_stats) = stats.get_mut(session_id) {
            session_stats.end_time = Some(Utc::now());
        }
    }

    /// Records an inference request.
    pub fn record_inference(
        &self,
        session_id: &SessionId,
        input_tokens: u32,
        output_tokens: u32,
        latency_ms: u64,
        success: bool,
        project: impl Into<String>,
        model_id: impl Into<String>,
    ) {
        let model = model_id.into();
        let family = Self::detect_family(&model);

        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::InferenceRequest {
                input_tokens,
                output_tokens,
                latency_ms,
                success,
            },
            timestamp: Utc::now(),
            project: project.into(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: model.clone(),
            model_family: family,
            metadata: HashMap::new(),
        };

        self.record_event(event);

        // Update stats
        let mut stats = self.stats.write();
        if let Some(session_stats) = stats.get_mut(session_id) {
            session_stats.total_requests += 1;
            if success {
                session_stats.successful_requests += 1;
            } else {
                session_stats.failed_requests += 1;
            }
            session_stats.total_input_tokens += input_tokens as u64;
            session_stats.total_output_tokens += output_tokens as u64;
            session_stats.total_latency_ms += latency_ms;

            if !session_stats.models_used.contains(&model) {
                session_stats.models_used.push(model);
            }
        }

        // Update global counters
        self.global_requests.fetch_add(1, Ordering::Relaxed);
        self.global_tokens
            .fetch_add((input_tokens + output_tokens) as u64, Ordering::Relaxed);
    }

    /// Records a checkpoint creation.
    pub fn record_checkpoint(
        &self,
        session_id: &SessionId,
        checkpoint_id: impl Into<String>,
        joy_count: u32,
        friction_count: u32,
    ) {
        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::CheckpointCreated {
                checkpoint_id: checkpoint_id.into(),
                joy_count,
                friction_count,
            },
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);

        // Update stats
        let mut stats = self.stats.write();
        if let Some(session_stats) = stats.get_mut(session_id) {
            session_stats.checkpoints_created += 1;
            session_stats.total_joys += joy_count;
            session_stats.total_frictions += friction_count;
        }

        self.global_checkpoints.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a phase transition.
    pub fn record_phase_transition(
        &self,
        session_id: &SessionId,
        from_phase: impl Into<String>,
        to_phase: impl Into<String>,
    ) {
        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::PhaseTransition {
                from_phase: from_phase.into(),
                to_phase: to_phase.into(),
            },
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Records a pattern discovery.
    pub fn record_pattern(&self, session_id: &SessionId, pattern_name: impl Into<String>) {
        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::PatternDiscovered {
                pattern_name: pattern_name.into(),
            },
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);

        let mut stats = self.stats.write();
        if let Some(session_stats) = stats.get_mut(session_id) {
            session_stats.patterns_discovered += 1;
        }
    }

    /// Records a friction.
    pub fn record_friction(
        &self,
        session_id: &SessionId,
        category: impl Into<String>,
        severity: impl Into<String>,
        blocking: bool,
    ) {
        let event = ResearchEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.clone(),
            event_type: EventType::FrictionEncountered {
                category: category.into(),
                severity: severity.into(),
                blocking,
            },
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Other("unknown".to_string()),
            metadata: HashMap::new(),
        };

        self.record_event(event);
    }

    /// Records an event.
    fn record_event(&self, event: ResearchEvent) {
        // Add to events store
        let mut events = self.events.write();
        events
            .entry(event.session_id.clone())
            .or_default()
            .push(event.clone());

        // Notify listeners
        let listeners = self.listeners.read();
        for listener in listeners.iter() {
            listener.on_event(&event);
        }
    }

    /// Adds an event listener.
    pub fn add_listener(&self, listener: Arc<dyn EventListener>) {
        self.listeners.write().push(listener);
    }

    /// Gets events for a session.
    pub fn get_events(&self, session_id: &SessionId) -> Vec<ResearchEvent> {
        self.events
            .read()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Gets stats for a session.
    pub fn get_stats(&self, session_id: &SessionId) -> Option<SessionStats> {
        self.stats.read().get(session_id).cloned()
    }

    /// Gets all session stats.
    pub fn all_stats(&self) -> Vec<SessionStats> {
        self.stats.read().values().cloned().collect()
    }

    /// Gets global statistics.
    pub fn global_stats(&self) -> GlobalStats {
        GlobalStats {
            total_requests: self.global_requests.load(Ordering::Relaxed),
            total_tokens: self.global_tokens.load(Ordering::Relaxed),
            total_checkpoints: self.global_checkpoints.load(Ordering::Relaxed),
            active_sessions: self.stats.read().len(),
        }
    }

    /// Detects model family from model ID.
    fn detect_family(model_id: &str) -> ModelFamily {
        let id = model_id.to_lowercase();
        if id.contains("claude") {
            ModelFamily::Claude
        } else if id.contains("gpt") {
            ModelFamily::Gpt
        } else if id.contains("gemini") {
            ModelFamily::Gemini
        } else if id.contains("llama") {
            ModelFamily::Llama
        } else if id.contains("mistral") {
            ModelFamily::Mistral
        } else {
            ModelFamily::Other(model_id.to_string())
        }
    }

    /// Exports events to JSON.
    pub fn export_json(&self) -> String {
        let events = self.events.read();
        serde_json::to_string_pretty(&*events).unwrap_or_default()
    }
}

impl Default for ResearchTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global statistics.
#[derive(Debug, Clone)]
pub struct GlobalStats {
    /// Total inference requests.
    pub total_requests: u64,
    /// Total tokens processed.
    pub total_tokens: u64,
    /// Total checkpoints created.
    pub total_checkpoints: u64,
    /// Number of active sessions.
    pub active_sessions: usize,
}

/// Trait for event listeners.
pub trait EventListener: Send + Sync {
    /// Called when an event is recorded.
    fn on_event(&self, event: &ResearchEvent);
}

/// Console logging listener.
pub struct ConsoleListener;

impl EventListener for ConsoleListener {
    fn on_event(&self, event: &ResearchEvent) {
        tracing::info!(
            session_id = %event.session_id.0,
            event_type = ?event.event_type,
            "Research event recorded"
        );
    }
}

/// JSON file logging listener.
pub struct JsonFileListener {
    /// Output file path.
    path: std::path::PathBuf,
}

impl JsonFileListener {
    /// Creates a new JSON file listener.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl EventListener for JsonFileListener {
    fn on_event(&self, event: &ResearchEvent) {
        if let Ok(json) = serde_json::to_string(event) {
            // Append to file
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let _ = writeln!(file, "{}", json);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // ==========================================================================
    // SessionId Tests
    // ==========================================================================

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
        let id1 = SessionId("session-123".to_string());
        let id2 = id1.clone();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_session_id_eq() {
        let id1 = SessionId("test".to_string());
        let id2 = SessionId("test".to_string());
        let id3 = SessionId("other".to_string());
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_session_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SessionId("a".to_string()));
        set.insert(SessionId("b".to_string()));
        set.insert(SessionId("a".to_string())); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_session_id_serialize() {
        let id = SessionId("serialize-test".to_string());
        let json = serde_json::to_string(&id).unwrap();
        assert!(json.contains("serialize-test"));
    }

    #[test]
    fn test_session_id_deserialize() {
        let json = r#""deserialize-test""#;
        let id: SessionId = serde_json::from_str(json).unwrap();
        assert_eq!(id.0, "deserialize-test");
    }

    // ==========================================================================
    // ModelFamily Tests
    // ==========================================================================

    #[test]
    fn test_model_family_all_variants() {
        let claude = ModelFamily::Claude;
        let gpt = ModelFamily::Gpt;
        let gemini = ModelFamily::Gemini;
        let llama = ModelFamily::Llama;
        let mistral = ModelFamily::Mistral;
        let other = ModelFamily::Other("custom".to_string());

        assert!(matches!(claude, ModelFamily::Claude));
        assert!(matches!(gpt, ModelFamily::Gpt));
        assert!(matches!(gemini, ModelFamily::Gemini));
        assert!(matches!(llama, ModelFamily::Llama));
        assert!(matches!(mistral, ModelFamily::Mistral));
        assert!(matches!(other, ModelFamily::Other(_)));
    }

    #[test]
    fn test_model_family_clone() {
        let family = ModelFamily::Claude;
        let cloned = family.clone();
        assert_eq!(family, cloned);
    }

    #[test]
    fn test_model_family_eq() {
        assert_eq!(ModelFamily::Claude, ModelFamily::Claude);
        assert_ne!(ModelFamily::Claude, ModelFamily::Gpt);
        assert_eq!(
            ModelFamily::Other("a".to_string()),
            ModelFamily::Other("a".to_string())
        );
    }

    #[test]
    fn test_model_family_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelFamily::Claude);
        set.insert(ModelFamily::Gpt);
        set.insert(ModelFamily::Claude); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_model_family_serialize() {
        let family = ModelFamily::Claude;
        let json = serde_json::to_string(&family).unwrap();
        assert!(json.contains("Claude"));
    }

    #[test]
    fn test_model_family_serialize_other() {
        let family = ModelFamily::Other("custom-model".to_string());
        let json = serde_json::to_string(&family).unwrap();
        assert!(json.contains("custom-model"));
    }

    // ==========================================================================
    // EventType Tests
    // ==========================================================================

    #[test]
    fn test_event_type_session_start() {
        let event = EventType::SessionStart;
        assert!(matches!(event, EventType::SessionStart));
    }

    #[test]
    fn test_event_type_session_end() {
        let event = EventType::SessionEnd;
        assert!(matches!(event, EventType::SessionEnd));
    }

    #[test]
    fn test_event_type_inference_request() {
        let event = EventType::InferenceRequest {
            input_tokens: 100,
            output_tokens: 50,
            latency_ms: 200,
            success: true,
        };
        if let EventType::InferenceRequest {
            input_tokens,
            output_tokens,
            latency_ms,
            success,
        } = event
        {
            assert_eq!(input_tokens, 100);
            assert_eq!(output_tokens, 50);
            assert_eq!(latency_ms, 200);
            assert!(success);
        } else {
            panic!("Expected InferenceRequest");
        }
    }

    #[test]
    fn test_event_type_checkpoint_created() {
        let event = EventType::CheckpointCreated {
            checkpoint_id: "cp-001".to_string(),
            joy_count: 5,
            friction_count: 2,
        };
        if let EventType::CheckpointCreated {
            checkpoint_id,
            joy_count,
            friction_count,
        } = event
        {
            assert_eq!(checkpoint_id, "cp-001");
            assert_eq!(joy_count, 5);
            assert_eq!(friction_count, 2);
        } else {
            panic!("Expected CheckpointCreated");
        }
    }

    #[test]
    fn test_event_type_phase_transition() {
        let event = EventType::PhaseTransition {
            from_phase: "init".to_string(),
            to_phase: "running".to_string(),
        };
        if let EventType::PhaseTransition {
            from_phase,
            to_phase,
        } = event
        {
            assert_eq!(from_phase, "init");
            assert_eq!(to_phase, "running");
        } else {
            panic!("Expected PhaseTransition");
        }
    }

    #[test]
    fn test_event_type_pattern_discovered() {
        let event = EventType::PatternDiscovered {
            pattern_name: "test-pattern".to_string(),
        };
        if let EventType::PatternDiscovered { pattern_name } = event {
            assert_eq!(pattern_name, "test-pattern");
        } else {
            panic!("Expected PatternDiscovered");
        }
    }

    #[test]
    fn test_event_type_friction_encountered() {
        let event = EventType::FrictionEncountered {
            category: "ui".to_string(),
            severity: "high".to_string(),
            blocking: true,
        };
        if let EventType::FrictionEncountered {
            category,
            severity,
            blocking,
        } = event
        {
            assert_eq!(category, "ui");
            assert_eq!(severity, "high");
            assert!(blocking);
        } else {
            panic!("Expected FrictionEncountered");
        }
    }

    #[test]
    fn test_event_type_model_switch() {
        let event = EventType::ModelSwitch {
            from_model: "gpt-4".to_string(),
            to_model: "claude-3".to_string(),
            reason: "cost optimization".to_string(),
        };
        if let EventType::ModelSwitch {
            from_model,
            to_model,
            reason,
        } = event
        {
            assert_eq!(from_model, "gpt-4");
            assert_eq!(to_model, "claude-3");
            assert_eq!(reason, "cost optimization");
        } else {
            panic!("Expected ModelSwitch");
        }
    }

    #[test]
    fn test_event_type_custom() {
        let mut data = HashMap::new();
        data.insert("key".to_string(), "value".to_string());
        let event = EventType::Custom {
            name: "custom-event".to_string(),
            data,
        };
        if let EventType::Custom { name, data } = event {
            assert_eq!(name, "custom-event");
            assert_eq!(data.get("key"), Some(&"value".to_string()));
        } else {
            panic!("Expected Custom");
        }
    }

    #[test]
    fn test_event_type_serialize() {
        let event = EventType::InferenceRequest {
            input_tokens: 10,
            output_tokens: 5,
            latency_ms: 100,
            success: true,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("InferenceRequest"));
        assert!(json.contains("10"));
    }

    // ==========================================================================
    // ResearchEvent Tests
    // ==========================================================================

    #[test]
    fn test_research_event_creation() {
        let event = ResearchEvent {
            id: "event-001".to_string(),
            session_id: SessionId("session-001".to_string()),
            event_type: EventType::SessionStart,
            timestamp: Utc::now(),
            project: "test-project".to_string(),
            phase: "init".to_string(),
            agent_id: "agent-001".to_string(),
            model_id: "claude-3".to_string(),
            model_family: ModelFamily::Claude,
            metadata: HashMap::new(),
        };

        assert_eq!(event.id, "event-001");
        assert_eq!(event.project, "test-project");
        assert_eq!(event.agent_id, "agent-001");
    }

    #[test]
    fn test_research_event_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());

        let event = ResearchEvent {
            id: "event-002".to_string(),
            session_id: SessionId::default(),
            event_type: EventType::SessionStart,
            timestamp: Utc::now(),
            project: String::new(),
            phase: String::new(),
            agent_id: String::new(),
            model_id: String::new(),
            model_family: ModelFamily::Gpt,
            metadata,
        };

        assert_eq!(event.metadata.len(), 2);
    }

    #[test]
    fn test_research_event_clone() {
        let event = ResearchEvent {
            id: "clone-test".to_string(),
            session_id: SessionId("session".to_string()),
            event_type: EventType::SessionEnd,
            timestamp: Utc::now(),
            project: "project".to_string(),
            phase: "end".to_string(),
            agent_id: "agent".to_string(),
            model_id: "model".to_string(),
            model_family: ModelFamily::Llama,
            metadata: HashMap::new(),
        };

        let cloned = event.clone();
        assert_eq!(event.id, cloned.id);
        assert_eq!(event.project, cloned.project);
    }

    #[test]
    fn test_research_event_serialize() {
        let event = ResearchEvent {
            id: "serialize-test".to_string(),
            session_id: SessionId("s".to_string()),
            event_type: EventType::SessionStart,
            timestamp: Utc::now(),
            project: "p".to_string(),
            phase: "init".to_string(),
            agent_id: "a".to_string(),
            model_id: "m".to_string(),
            model_family: ModelFamily::Claude,
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("serialize-test"));
        assert!(json.contains("SessionStart"));
    }

    // ==========================================================================
    // SessionStats Tests
    // ==========================================================================

    #[test]
    fn test_session_stats_new() {
        let stats = SessionStats::new("test-session".into());
        assert_eq!(stats.session_id.0, "test-session");
        assert_eq!(stats.total_requests, 0);
        assert!(stats.start_time.is_none());
    }

    #[test]
    fn test_session_stats_default() {
        let stats = SessionStats::default();
        assert!(stats.session_id.0.is_empty());
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_session_stats_calculations() {
        let mut stats = SessionStats::new("test".into());
        stats.total_requests = 10;
        stats.successful_requests = 9;
        stats.total_latency_ms = 1000;
        stats.total_joys = 5;
        stats.total_frictions = 2;

        assert_eq!(stats.success_rate(), 0.9);
        assert_eq!(stats.average_latency_ms(), 100.0);
        assert_eq!(stats.joy_friction_ratio(), 2.5);
    }

    #[test]
    fn test_session_stats_average_latency_zero_requests() {
        let stats = SessionStats::new("test".into());
        assert_eq!(stats.average_latency_ms(), 0.0);
    }

    #[test]
    fn test_session_stats_success_rate_zero_requests() {
        let stats = SessionStats::new("test".into());
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_session_stats_avg_tokens_per_request() {
        let mut stats = SessionStats::new("test".into());
        stats.total_requests = 5;
        stats.total_input_tokens = 500;
        stats.total_output_tokens = 250;

        let (avg_input, avg_output) = stats.avg_tokens_per_request();
        assert_eq!(avg_input, 100.0);
        assert_eq!(avg_output, 50.0);
    }

    #[test]
    fn test_session_stats_avg_tokens_zero_requests() {
        let stats = SessionStats::new("test".into());
        let (avg_input, avg_output) = stats.avg_tokens_per_request();
        assert_eq!(avg_input, 0.0);
        assert_eq!(avg_output, 0.0);
    }

    #[test]
    fn test_session_stats_joy_friction_ratio_zero_frictions() {
        let mut stats = SessionStats::new("test".into());
        stats.total_joys = 10;
        stats.total_frictions = 0;

        assert!(stats.joy_friction_ratio().is_infinite());
    }

    #[test]
    fn test_session_stats_clone() {
        let mut stats = SessionStats::new("clone-test".into());
        stats.total_requests = 5;
        stats.models_used.push("model-1".to_string());

        let cloned = stats.clone();
        assert_eq!(cloned.session_id.0, "clone-test");
        assert_eq!(cloned.total_requests, 5);
        assert_eq!(cloned.models_used.len(), 1);
    }

    // ==========================================================================
    // ResearchTracker Tests
    // ==========================================================================

    #[test]
    fn test_research_tracker_new() {
        let tracker = ResearchTracker::new();
        let global = tracker.global_stats();
        assert_eq!(global.total_requests, 0);
        assert_eq!(global.active_sessions, 0);
    }

    #[test]
    fn test_research_tracker_default() {
        let tracker = ResearchTracker::default();
        let global = tracker.global_stats();
        assert_eq!(global.total_requests, 0);
    }

    #[test]
    fn test_research_tracker() {
        let tracker = ResearchTracker::new();

        let session_id =
            tracker.start_session("session-001", "infernum", "agent-001", "claude-opus-4");

        tracker.record_inference(
            &session_id,
            500,
            200,
            150,
            true,
            "infernum",
            "claude-opus-4",
        );

        tracker.record_checkpoint(&session_id, "checkpoint-001", 3, 1);

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.checkpoints_created, 1);
        assert_eq!(stats.total_joys, 3);
        assert_eq!(stats.total_frictions, 1);
    }

    #[test]
    fn test_research_tracker_start_session() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("my-session", "project", "agent", "model");

        assert_eq!(session_id.0, "my-session");

        let stats = tracker.get_stats(&session_id).unwrap();
        assert!(stats.start_time.is_some());
    }

    #[test]
    fn test_research_tracker_end_session() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("end-test", "project", "agent", "model");
        tracker.end_session(&session_id);

        let stats = tracker.get_stats(&session_id).unwrap();
        assert!(stats.end_time.is_some());
    }

    #[test]
    fn test_research_tracker_record_inference_success() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("inference-test", "p", "a", "m");

        tracker.record_inference(&session_id, 100, 50, 200, true, "project", "claude-3");

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.total_input_tokens, 100);
        assert_eq!(stats.total_output_tokens, 50);
        assert_eq!(stats.total_latency_ms, 200);
    }

    #[test]
    fn test_research_tracker_record_inference_failure() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("failure-test", "p", "a", "m");

        tracker.record_inference(&session_id, 100, 0, 50, false, "project", "gpt-4");

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 1);
    }

    #[test]
    fn test_research_tracker_record_multiple_inferences() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("multi", "p", "a", "m");

        for i in 0..5 {
            tracker.record_inference(&session_id, 100, 50, 100, i % 2 == 0, "project", "model");
        }

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.failed_requests, 2);
    }

    #[test]
    fn test_research_tracker_record_checkpoint() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("checkpoint-test", "p", "a", "m");

        tracker.record_checkpoint(&session_id, "cp-1", 10, 3);
        tracker.record_checkpoint(&session_id, "cp-2", 5, 1);

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.checkpoints_created, 2);
        assert_eq!(stats.total_joys, 15);
        assert_eq!(stats.total_frictions, 4);
    }

    #[test]
    fn test_research_tracker_record_phase_transition() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("phase-test", "p", "a", "m");

        tracker.record_phase_transition(&session_id, "init", "running");

        let events = tracker.get_events(&session_id);
        assert!(events.len() >= 2); // Start + transition
        assert!(events
            .iter()
            .any(|e| matches!(e.event_type, EventType::PhaseTransition { .. })));
    }

    #[test]
    fn test_research_tracker_record_pattern() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("pattern-test", "p", "a", "m");

        tracker.record_pattern(&session_id, "pattern-a");
        tracker.record_pattern(&session_id, "pattern-b");

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.patterns_discovered, 2);
    }

    #[test]
    fn test_research_tracker_record_friction() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("friction-test", "p", "a", "m");

        tracker.record_friction(&session_id, "ui", "high", true);

        let events = tracker.get_events(&session_id);
        assert!(events.iter().any(|e| matches!(
            &e.event_type,
            EventType::FrictionEncountered { blocking, .. } if *blocking
        )));
    }

    #[test]
    fn test_research_tracker_get_events_empty() {
        let tracker = ResearchTracker::new();
        let events = tracker.get_events(&SessionId("nonexistent".to_string()));
        assert!(events.is_empty());
    }

    #[test]
    fn test_research_tracker_get_stats_none() {
        let tracker = ResearchTracker::new();
        let stats = tracker.get_stats(&SessionId("nonexistent".to_string()));
        assert!(stats.is_none());
    }

    #[test]
    fn test_research_tracker_all_stats() {
        let tracker = ResearchTracker::new();
        tracker.start_session("s1", "p1", "a1", "m1");
        tracker.start_session("s2", "p2", "a2", "m2");

        let all = tracker.all_stats();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_research_tracker_global_stats() {
        let tracker = ResearchTracker::new();
        let s1 = tracker.start_session("s1", "p", "a", "m");
        let s2 = tracker.start_session("s2", "p", "a", "m");

        tracker.record_inference(&s1, 100, 50, 100, true, "p", "m");
        tracker.record_inference(&s2, 200, 100, 200, true, "p", "m");
        tracker.record_checkpoint(&s1, "cp", 1, 0);

        let global = tracker.global_stats();
        assert_eq!(global.total_requests, 2);
        assert_eq!(global.total_tokens, 450); // 100+50 + 200+100
        assert_eq!(global.total_checkpoints, 1);
        assert_eq!(global.active_sessions, 2);
    }

    #[test]
    fn test_research_tracker_models_used() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("models-test", "p", "a", "m");

        tracker.record_inference(&session_id, 10, 5, 10, true, "p", "claude-3");
        tracker.record_inference(&session_id, 10, 5, 10, true, "p", "gpt-4");
        tracker.record_inference(&session_id, 10, 5, 10, true, "p", "claude-3"); // Duplicate

        let stats = tracker.get_stats(&session_id).unwrap();
        assert_eq!(stats.models_used.len(), 2);
        assert!(stats.models_used.contains(&"claude-3".to_string()));
        assert!(stats.models_used.contains(&"gpt-4".to_string()));
    }

    #[test]
    fn test_research_tracker_export_json() {
        let tracker = ResearchTracker::new();
        let session_id = tracker.start_session("export-test", "project", "agent", "model");
        tracker.record_inference(&session_id, 100, 50, 100, true, "project", "model");

        let json = tracker.export_json();
        assert!(!json.is_empty());
        assert!(json.contains("export-test"));
    }

    // ==========================================================================
    // Model Family Detection Tests
    // ==========================================================================

    #[test]
    fn test_model_family_detection() {
        assert!(matches!(
            ResearchTracker::detect_family("claude-opus-4"),
            ModelFamily::Claude
        ));
        assert!(matches!(
            ResearchTracker::detect_family("gpt-4o"),
            ModelFamily::Gpt
        ));
        assert!(matches!(
            ResearchTracker::detect_family("llama-3.2-70b"),
            ModelFamily::Llama
        ));
    }

    #[test]
    fn test_model_family_detection_gemini() {
        assert!(matches!(
            ResearchTracker::detect_family("gemini-pro"),
            ModelFamily::Gemini
        ));
        assert!(matches!(
            ResearchTracker::detect_family("GEMINI-1.5"),
            ModelFamily::Gemini
        ));
    }

    #[test]
    fn test_model_family_detection_mistral() {
        assert!(matches!(
            ResearchTracker::detect_family("mistral-7b"),
            ModelFamily::Mistral
        ));
        assert!(matches!(
            ResearchTracker::detect_family("Mistral-Large"),
            ModelFamily::Mistral
        ));
    }

    #[test]
    fn test_model_family_detection_other() {
        let family = ResearchTracker::detect_family("custom-model-v1");
        if let ModelFamily::Other(name) = family {
            assert_eq!(name, "custom-model-v1");
        } else {
            panic!("Expected Other variant");
        }
    }

    #[test]
    fn test_model_family_detection_case_insensitive() {
        assert!(matches!(
            ResearchTracker::detect_family("CLAUDE-OPUS-4"),
            ModelFamily::Claude
        ));
        assert!(matches!(
            ResearchTracker::detect_family("GPT-4-TURBO"),
            ModelFamily::Gpt
        ));
    }

    // ==========================================================================
    // GlobalStats Tests
    // ==========================================================================

    #[test]
    fn test_global_stats_structure() {
        let stats = GlobalStats {
            total_requests: 100,
            total_tokens: 5000,
            total_checkpoints: 10,
            active_sessions: 5,
        };

        assert_eq!(stats.total_requests, 100);
        assert_eq!(stats.total_tokens, 5000);
        assert_eq!(stats.total_checkpoints, 10);
        assert_eq!(stats.active_sessions, 5);
    }

    #[test]
    fn test_global_stats_clone() {
        let stats = GlobalStats {
            total_requests: 50,
            total_tokens: 2500,
            total_checkpoints: 5,
            active_sessions: 2,
        };
        let cloned = stats.clone();
        assert_eq!(stats.total_requests, cloned.total_requests);
    }

    // ==========================================================================
    // EventListener Tests
    // ==========================================================================

    struct TestListener {
        call_count: AtomicUsize,
    }

    impl TestListener {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }

        fn count(&self) -> usize {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    impl EventListener for TestListener {
        fn on_event(&self, _event: &ResearchEvent) {
            self.call_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn test_add_listener() {
        let tracker = ResearchTracker::new();
        let listener = Arc::new(TestListener::new());
        tracker.add_listener(listener.clone());

        tracker.start_session("listener-test", "p", "a", "m");

        assert_eq!(listener.count(), 1);
    }

    #[test]
    fn test_listener_receives_multiple_events() {
        let tracker = ResearchTracker::new();
        let listener = Arc::new(TestListener::new());
        tracker.add_listener(listener.clone());

        let session_id = tracker.start_session("multi-event", "p", "a", "m");
        tracker.record_inference(&session_id, 100, 50, 100, true, "p", "m");
        tracker.record_checkpoint(&session_id, "cp", 1, 0);
        tracker.end_session(&session_id);

        assert_eq!(listener.count(), 4);
    }

    // ==========================================================================
    // ConsoleListener Tests
    // ==========================================================================

    #[test]
    fn test_console_listener_creation() {
        let _listener = ConsoleListener;
        // ConsoleListener just logs, so we just test it can be created
    }

    // ==========================================================================
    // JsonFileListener Tests
    // ==========================================================================

    #[test]
    fn test_json_file_listener_new() {
        let listener = JsonFileListener::new("/tmp/test.json");
        assert_eq!(listener.path, std::path::PathBuf::from("/tmp/test.json"));
    }

    #[test]
    fn test_json_file_listener_path_from_string() {
        let listener = JsonFileListener::new(String::from("/tmp/events.jsonl"));
        assert_eq!(listener.path, std::path::PathBuf::from("/tmp/events.jsonl"));
    }
}
