//! Jormungandr Research Initiative - Sigil Knowledge Base
//!
//! This module provides the RAG corpus setup for the Jormungandr research
//! initiative, storing Sigil documentation, conversion checkpoints,
//! and patterns for agent retrieval.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{Document, InMemoryStore};

/// Corpus types for Jormungandr research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorpusType {
    /// Sigil language documentation and specification.
    SigilDocs,
    /// Conversion experience checkpoints.
    Checkpoints,
    /// Resolved frictions (what worked before).
    ResolvedFrictions,
    /// Pattern library (emergent idioms).
    Patterns,
    /// Source code being converted.
    SourceCode,
    /// Generated Sigil code.
    GeneratedSigil,
}

impl CorpusType {
    /// Returns the namespace prefix for this corpus.
    pub fn namespace(&self) -> &'static str {
        match self {
            Self::SigilDocs => "sigil_docs",
            Self::Checkpoints => "checkpoints",
            Self::ResolvedFrictions => "frictions",
            Self::Patterns => "patterns",
            Self::SourceCode => "source",
            Self::GeneratedSigil => "generated",
        }
    }
}

/// Experience checkpoint from Jormungandr conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceCheckpoint {
    /// Unique checkpoint ID.
    pub id: String,
    /// Project being converted.
    pub project: String,
    /// Conversion phase.
    pub phase: ConversionPhase,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
    /// Agent that created the checkpoint.
    pub agent_id: String,
    /// Model used.
    pub model_id: String,
    /// Duration of this phase.
    pub duration_secs: u64,
    /// Lines of code converted.
    pub lines_converted: u32,
    /// Sigil lines written.
    pub sigil_lines_written: u32,
    /// Compression/expansion ratio.
    pub ratio: f32,
    /// Joys experienced.
    pub joys: Vec<Joy>,
    /// Frictions encountered.
    pub frictions: Vec<Friction>,
    /// Patterns discovered.
    pub patterns_discovered: Vec<Pattern>,
    /// Missing features identified.
    pub missing_features: Vec<FeatureGap>,
    /// Confidence level.
    pub confidence: Evidentiality,
    /// Whether agent would use Sigil again.
    pub would_use_again: bool,
    /// Freeform notes.
    pub notes: Option<String>,
}

impl ExperienceCheckpoint {
    /// Creates a new checkpoint.
    pub fn new(project: impl Into<String>, phase: ConversionPhase) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            project: project.into(),
            phase,
            timestamp: Utc::now(),
            agent_id: String::new(),
            model_id: String::new(),
            duration_secs: 0,
            lines_converted: 0,
            sigil_lines_written: 0,
            ratio: 1.0,
            joys: Vec::new(),
            frictions: Vec::new(),
            patterns_discovered: Vec::new(),
            missing_features: Vec::new(),
            confidence: Evidentiality::Reported,
            would_use_again: true,
            notes: None,
        }
    }

    /// Sets agent info.
    pub fn with_agent(mut self, agent_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        self.agent_id = agent_id.into();
        self.model_id = model_id.into();
        self
    }

    /// Adds a joy.
    pub fn add_joy(&mut self, joy: Joy) {
        self.joys.push(joy);
    }

    /// Adds a friction.
    pub fn add_friction(&mut self, friction: Friction) {
        self.frictions.push(friction);
    }

    /// Adds a discovered pattern.
    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.patterns_discovered.push(pattern);
    }

    /// Calculates joy/friction ratio.
    pub fn joy_friction_ratio(&self) -> f32 {
        if self.frictions.is_empty() {
            return f32::INFINITY;
        }
        self.joys.len() as f32 / self.frictions.len() as f32
    }

    /// Converts to a document for RAG indexing.
    pub fn to_document(&self) -> Document {
        let content = format!(
            "Project: {}\nPhase: {:?}\nAgent: {}\nModel: {}\n\
             Lines: {} → {} (ratio: {:.2})\n\
             Joys: {:?}\nFrictions: {:?}\nPatterns: {:?}\n\
             Notes: {}",
            self.project,
            self.phase,
            self.agent_id,
            self.model_id,
            self.lines_converted,
            self.sigil_lines_written,
            self.ratio,
            self.joys.iter().map(|j| &j.description).collect::<Vec<_>>(),
            self.frictions.iter().map(|f| &f.description).collect::<Vec<_>>(),
            self.patterns_discovered.iter().map(|p| &p.name).collect::<Vec<_>>(),
            self.notes.as_deref().unwrap_or("None")
        );

        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        metadata.insert("project".to_string(), serde_json::Value::String(self.project.clone()));
        metadata.insert("phase".to_string(), serde_json::Value::String(format!("{:?}", self.phase)));
        metadata.insert("agent_id".to_string(), serde_json::Value::String(self.agent_id.clone()));
        metadata.insert("model_id".to_string(), serde_json::Value::String(self.model_id.clone()));
        metadata.insert("joy_count".to_string(), serde_json::json!(self.joys.len()));
        metadata.insert("friction_count".to_string(), serde_json::json!(self.frictions.len()));

        Document {
            id: self.id.clone(),
            content,
            metadata,
        }
    }
}

/// Conversion phases for Jormungandr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConversionPhase {
    /// Understand the existing codebase.
    Analysis,
    /// Plan the Sigil architecture.
    Design,
    /// Convert core data structures and types.
    Core,
    /// Convert business logic and algorithms.
    Logic,
    /// Wire up dependencies, I/O, external APIs.
    Integration,
    /// Error handling, edge cases, optimization.
    Polish,
    /// Testing, verification, comparison.
    Validation,
}

impl ConversionPhase {
    /// Returns recommended collaboration mode.
    pub fn collaboration_mode(&self) -> CollaborationMode {
        match self {
            Self::Analysis => CollaborationMode::Solo,
            Self::Design => CollaborationMode::Solo,
            Self::Core => CollaborationMode::Pair,
            Self::Logic => CollaborationMode::Pair,
            Self::Integration => CollaborationMode::Solo,
            Self::Polish => CollaborationMode::Solo,
            Self::Validation => CollaborationMode::Independent,
        }
    }
}

/// Collaboration modes for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollaborationMode {
    /// Individual work, fresh perspective.
    Solo,
    /// Two agents collaborating.
    Pair,
    /// Must be done by different agent than converter.
    Independent,
}

/// A joy experienced during conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joy {
    /// Description of the joy.
    pub description: String,
    /// Category of joy.
    pub category: JoyCategory,
    /// Intensity (0.0-1.0).
    pub intensity: f32,
    /// Example code snippet.
    pub example: Option<String>,
    /// Whether this is consistently reproducible.
    pub reproducible: bool,
}

impl Joy {
    /// Creates a new joy.
    pub fn new(description: impl Into<String>, category: JoyCategory) -> Self {
        Self {
            description: description.into(),
            category,
            intensity: 0.5,
            example: None,
            reproducible: true,
        }
    }

    /// Sets intensity.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Sets example.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }
}

/// Categories of joy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JoyCategory {
    /// "I could say this concisely"
    Expressiveness,
    /// "The type system caught my mistake"
    Safety,
    /// "The code reads naturally"
    Clarity,
    /// "I did something hard easily"
    Power,
    /// "This is beautiful"
    Elegance,
    /// "I found a new way to think"
    Discovery,
    /// "I was in the zone"
    Flow,
}

/// A friction encountered during conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Friction {
    /// Description of the friction.
    pub description: String,
    /// Category of friction.
    pub category: FrictionCategory,
    /// Severity level.
    pub severity: Severity,
    /// Workaround if found.
    pub workaround: Option<String>,
    /// Whether this blocked progress.
    pub blocking: bool,
    /// Example code snippet.
    pub example: Option<String>,
}

impl Friction {
    /// Creates a new friction.
    pub fn new(description: impl Into<String>, category: FrictionCategory) -> Self {
        Self {
            description: description.into(),
            category,
            severity: Severity::Moderate,
            workaround: None,
            blocking: false,
            example: None,
        }
    }

    /// Sets severity.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Sets workaround.
    pub fn with_workaround(mut self, workaround: impl Into<String>) -> Self {
        self.workaround = Some(workaround.into());
        self
    }

    /// Marks as blocking.
    pub fn blocking(mut self) -> Self {
        self.blocking = true;
        self.severity = Severity::Blocking;
        self
    }
}

/// Categories of friction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrictionCategory {
    /// "The grammar is awkward here"
    Syntax,
    /// "This doesn't mean what I expected"
    Semantics,
    /// "The compiler/LSP failed me"
    Tooling,
    /// "I couldn't find how to do X"
    Documentation,
    /// "I needed X but it doesn't exist"
    MissingFeature,
    /// "This was too slow"
    Performance,
    /// "I couldn't understand the error"
    ErrorMessages,
    /// "Connecting to external systems was hard"
    Interop,
}

/// Severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Minor inconvenience.
    Minor,
    /// Noticeable but manageable.
    Moderate,
    /// Significant impact on productivity.
    Major,
    /// Completely blocked progress.
    Blocking,
}

/// A pattern discovered during conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Example code.
    pub example: String,
    /// How often used.
    pub frequency: Frequency,
    /// Should this be a builtin?
    pub should_be_builtin: bool,
}

impl Pattern {
    /// Creates a new pattern.
    pub fn new(name: impl Into<String>, description: impl Into<String>, example: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            example: example.into(),
            frequency: Frequency::Sometimes,
            should_be_builtin: false,
        }
    }
}

/// Frequency of pattern usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    /// Used once.
    Once,
    /// Used a few times.
    Sometimes,
    /// Used frequently.
    Often,
    /// Used constantly.
    Always,
}

/// A missing feature identified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGap {
    /// Description of missing feature.
    pub description: String,
    /// Use case that needed it.
    pub use_case: String,
    /// How it was worked around.
    pub workaround: Option<String>,
    /// Priority level.
    pub priority: GapPriority,
    /// Similar features in other languages.
    pub similar_in_other_langs: Vec<String>,
}

/// Priority for feature gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GapPriority {
    /// Nice to have.
    Low,
    /// Would help significantly.
    Medium,
    /// Critically needed.
    High,
    /// Blocking adoption.
    Critical,
}

/// Evidentiality markers (from Sigil).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Evidentiality {
    /// Known with certainty.
    Known,
    /// Uncertain/speculative.
    Uncertain,
    /// Reported/external source.
    Reported,
}

/// Sigil knowledge base for Jormungandr research.
pub struct SigilKnowledgeBase {
    /// Corpus stores by type.
    stores: HashMap<CorpusType, Arc<InMemoryStore>>,
    /// Checkpoints.
    checkpoints: RwLock<Vec<ExperienceCheckpoint>>,
    /// Patterns catalog.
    patterns: RwLock<Vec<Pattern>>,
    /// Root directory for persistence.
    root_dir: Option<PathBuf>,
}

impl SigilKnowledgeBase {
    /// Creates a new in-memory knowledge base.
    pub fn new() -> Self {
        let mut stores = HashMap::new();
        for corpus_type in &[
            CorpusType::SigilDocs,
            CorpusType::Checkpoints,
            CorpusType::ResolvedFrictions,
            CorpusType::Patterns,
            CorpusType::SourceCode,
            CorpusType::GeneratedSigil,
        ] {
            stores.insert(*corpus_type, Arc::new(InMemoryStore::new()));
        }

        Self {
            stores,
            checkpoints: RwLock::new(Vec::new()),
            patterns: RwLock::new(Vec::new()),
            root_dir: None,
        }
    }

    /// Creates with a persistence directory.
    pub fn with_persistence(root_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&root_dir)?;
        let mut kb = Self::new();
        kb.root_dir = Some(root_dir);
        Ok(kb)
    }

    /// Gets a store by corpus type.
    pub fn store(&self, corpus_type: CorpusType) -> Option<Arc<InMemoryStore>> {
        self.stores.get(&corpus_type).cloned()
    }

    /// Adds a checkpoint.
    pub fn add_checkpoint(&self, checkpoint: ExperienceCheckpoint) {
        // Add to checkpoints list
        self.checkpoints.write().push(checkpoint.clone());

        // Index patterns
        for pattern in &checkpoint.patterns_discovered {
            self.patterns.write().push(pattern.clone());
        }
    }

    /// Gets all checkpoints.
    pub fn checkpoints(&self) -> Vec<ExperienceCheckpoint> {
        self.checkpoints.read().clone()
    }

    /// Gets checkpoints by project.
    pub fn checkpoints_for_project(&self, project: &str) -> Vec<ExperienceCheckpoint> {
        self.checkpoints
            .read()
            .iter()
            .filter(|c| c.project == project)
            .cloned()
            .collect()
    }

    /// Gets all patterns.
    pub fn patterns(&self) -> Vec<Pattern> {
        self.patterns.read().clone()
    }

    /// Generates an aggregated research report.
    pub fn generate_report(&self) -> ResearchReport {
        let checkpoints = self.checkpoints.read();

        let total_joys: usize = checkpoints.iter().map(|c| c.joys.len()).sum();
        let total_frictions: usize = checkpoints.iter().map(|c| c.frictions.len()).sum();

        // Aggregate joys by category
        let mut joy_by_category: HashMap<JoyCategory, Vec<&Joy>> = HashMap::new();
        for cp in checkpoints.iter() {
            for joy in &cp.joys {
                joy_by_category.entry(joy.category).or_default().push(joy);
            }
        }

        // Aggregate frictions by category
        let mut friction_by_category: HashMap<FrictionCategory, Vec<&Friction>> = HashMap::new();
        for cp in checkpoints.iter() {
            for friction in &cp.frictions {
                friction_by_category
                    .entry(friction.category)
                    .or_default()
                    .push(friction);
            }
        }

        // Find most common frictions
        let mut friction_counts: Vec<_> = friction_by_category
            .iter()
            .map(|(cat, frictions)| (*cat, frictions.len()))
            .collect();
        friction_counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        ResearchReport {
            checkpoint_count: checkpoints.len(),
            total_joys,
            total_frictions,
            joy_friction_ratio: if total_frictions > 0 {
                total_joys as f32 / total_frictions as f32
            } else {
                f32::INFINITY
            },
            top_friction_categories: friction_counts
                .into_iter()
                .take(5)
                .map(|(cat, count)| (cat, count))
                .collect(),
            pattern_count: self.patterns.read().len(),
            projects_analyzed: checkpoints
                .iter()
                .map(|c| c.project.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }

    /// Saves checkpoints to disk (if persistence enabled).
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(root) = &self.root_dir {
            let checkpoints_file = root.join("checkpoints.json");
            let checkpoints = self.checkpoints.read();
            let content = serde_json::to_string_pretty(&*checkpoints)?;
            std::fs::write(checkpoints_file, content)?;

            let patterns_file = root.join("patterns.json");
            let patterns = self.patterns.read();
            let content = serde_json::to_string_pretty(&*patterns)?;
            std::fs::write(patterns_file, content)?;
        }
        Ok(())
    }

    /// Loads checkpoints from disk (if persistence enabled).
    pub fn load(&self) -> std::io::Result<()> {
        if let Some(root) = &self.root_dir {
            let checkpoints_file = root.join("checkpoints.json");
            if checkpoints_file.exists() {
                let content = std::fs::read_to_string(checkpoints_file)?;
                let loaded: Vec<ExperienceCheckpoint> = serde_json::from_str(&content)?;
                *self.checkpoints.write() = loaded;
            }

            let patterns_file = root.join("patterns.json");
            if patterns_file.exists() {
                let content = std::fs::read_to_string(patterns_file)?;
                let loaded: Vec<Pattern> = serde_json::from_str(&content)?;
                *self.patterns.write() = loaded;
            }
        }
        Ok(())
    }
}

impl Default for SigilKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated research report.
#[derive(Debug, Clone)]
pub struct ResearchReport {
    /// Number of checkpoints analyzed.
    pub checkpoint_count: usize,
    /// Total joys recorded.
    pub total_joys: usize,
    /// Total frictions recorded.
    pub total_frictions: usize,
    /// Joy/friction ratio.
    pub joy_friction_ratio: f32,
    /// Top friction categories.
    pub top_friction_categories: Vec<(FrictionCategory, usize)>,
    /// Number of patterns discovered.
    pub pattern_count: usize,
    /// Number of projects analyzed.
    pub projects_analyzed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let mut checkpoint = ExperienceCheckpoint::new("infernum", ConversionPhase::Core)
            .with_agent("agent-001", "claude-opus-4");

        checkpoint.add_joy(
            Joy::new("Type inference is excellent", JoyCategory::Safety)
                .with_intensity(0.9)
        );
        checkpoint.add_friction(
            Friction::new("Async syntax is verbose", FrictionCategory::Syntax)
                .with_severity(Severity::Minor)
        );

        assert_eq!(checkpoint.project, "infernum");
        assert_eq!(checkpoint.joys.len(), 1);
        assert_eq!(checkpoint.frictions.len(), 1);
        assert_eq!(checkpoint.joy_friction_ratio(), 1.0);
    }

    #[test]
    fn test_knowledge_base() {
        let kb = SigilKnowledgeBase::new();

        let checkpoint = ExperienceCheckpoint::new("test-project", ConversionPhase::Analysis);
        kb.add_checkpoint(checkpoint);

        let checkpoints = kb.checkpoints();
        assert_eq!(checkpoints.len(), 1);

        let report = kb.generate_report();
        assert_eq!(report.checkpoint_count, 1);
    }

    #[test]
    fn test_conversion_phase_collaboration() {
        assert_eq!(
            ConversionPhase::Validation.collaboration_mode(),
            CollaborationMode::Independent
        );
        assert_eq!(
            ConversionPhase::Core.collaboration_mode(),
            CollaborationMode::Pair
        );
    }
}
