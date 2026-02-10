//! Long-term memory system with persistent file storage.
//!
//! Inspired by Leviathan/Persona's `LongTermMemory` system that provides:
//! - File-based persistent storage
//! - Memory types (session summary, project learning, decisions, etc.)
//! - Full-text search capability
//! - Importance levels
//! - Automatic tagging and metadata

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use infernum_core::Result;

// ============================================================================
// Memory Types and Importance
// ============================================================================

/// Types of long-term memory entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Summary of a multi-turn conversation session.
    SessionSummary,
    /// Knowledge learned across sessions.
    ProjectLearning,
    /// Important decisions made.
    Decision,
    /// Project context and background.
    Context,
    /// Error patterns and their solutions.
    ErrorPattern,
    /// Performance optimizations discovered.
    Optimization,
    /// User preferences and behaviors.
    UserPreference,
    /// Code patterns and conventions.
    CodePattern,
}

impl MemoryType {
    /// Returns the directory name for this memory type.
    #[must_use]
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::SessionSummary => "sessions",
            Self::ProjectLearning => "learnings",
            Self::Decision => "decisions",
            Self::Context => "context",
            Self::ErrorPattern => "errors",
            Self::Optimization => "optimizations",
            Self::UserPreference => "preferences",
            Self::CodePattern => "patterns",
        }
    }
}

/// Importance level of a memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImportanceLevel {
    /// Low importance, may be pruned first.
    Low,
    /// Medium importance, normal retention.
    Medium,
    /// High importance, retained longer.
    High,
    /// Critical importance, never pruned automatically.
    Critical,
}

impl Default for ImportanceLevel {
    fn default() -> Self {
        Self::Medium
    }
}

// ============================================================================
// Memory Entry
// ============================================================================

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: String,
    /// Type of memory.
    pub memory_type: MemoryType,
    /// The memory content.
    pub content: String,
    /// Summary of the content.
    pub summary: Option<String>,
    /// Importance level.
    pub importance: ImportanceLevel,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Last access timestamp.
    pub last_accessed: u64,
    /// Number of times this memory was accessed.
    pub access_count: u64,
    /// Related memory IDs.
    pub related_ids: Vec<String>,
}

impl MemoryEntry {
    /// Creates a new memory entry.
    pub fn new(memory_type: MemoryType, content: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id: generate_id(),
            memory_type,
            content: content.into(),
            summary: None,
            importance: ImportanceLevel::default(),
            tags: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            related_ids: Vec::new(),
        }
    }

    /// Sets the summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Sets the importance level.
    #[must_use]
    pub fn with_importance(mut self, importance: ImportanceLevel) -> Self {
        self.importance = importance;
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Adds tags.
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Adds metadata.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Records an access to this memory.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Checks if content matches the search query.
    pub fn matches(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.content.to_lowercase().contains(&query_lower)
            || self
                .summary
                .as_ref()
                .map_or(false, |s| s.to_lowercase().contains(&query_lower))
            || self
                .tags
                .iter()
                .any(|t| t.to_lowercase().contains(&query_lower))
    }
}

// ============================================================================
// Memory Statistics
// ============================================================================

/// Statistics about the memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Entries by type.
    pub by_type: HashMap<String, usize>,
    /// Entries by importance.
    pub by_importance: HashMap<String, usize>,
    /// Total size in bytes.
    pub total_size_bytes: u64,
    /// Oldest entry timestamp.
    pub oldest_entry: Option<u64>,
    /// Newest entry timestamp.
    pub newest_entry: Option<u64>,
}

// ============================================================================
// Long-Term Memory Store
// ============================================================================

/// Long-term memory store with file-based persistence.
pub struct LongTermMemory {
    /// Base directory for storage.
    base_dir: PathBuf,
    /// In-memory cache of entries.
    cache: HashMap<String, MemoryEntry>,
    /// Index by memory type.
    type_index: HashMap<MemoryType, Vec<String>>,
    /// Index by tag.
    tag_index: HashMap<String, Vec<String>>,
    /// ID to memory type index for O(1) lookups.
    id_type_index: HashMap<String, MemoryType>,
    /// Maximum cache size.
    max_cache_size: usize,
}

impl LongTermMemory {
    /// Creates a new long-term memory store.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();

        // Create directory structure
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir).map_err(|e| {
                infernum_core::Error::internal(format!("Failed to create memory dir: {}", e))
            })?;
        }

        // Create subdirectories for each memory type
        for memory_type in [
            MemoryType::SessionSummary,
            MemoryType::ProjectLearning,
            MemoryType::Decision,
            MemoryType::Context,
            MemoryType::ErrorPattern,
            MemoryType::Optimization,
            MemoryType::UserPreference,
            MemoryType::CodePattern,
        ] {
            let type_dir = base_dir.join(memory_type.dir_name());
            if !type_dir.exists() {
                fs::create_dir_all(&type_dir).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to create type dir: {}", e))
                })?;
            }
        }

        let mut memory = Self {
            base_dir,
            cache: HashMap::new(),
            type_index: HashMap::new(),
            tag_index: HashMap::new(),
            id_type_index: HashMap::new(),
            max_cache_size: 1000,
        };

        // Load existing entries
        memory.load_all()?;

        Ok(memory)
    }

    /// Creates with a custom cache size.
    #[must_use]
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Stores a memory entry.
    pub fn store(&mut self, entry: MemoryEntry) -> Result<String> {
        let id = entry.id.clone();
        let memory_type = entry.memory_type;
        let tags = entry.tags.clone();

        // Write to disk
        self.write_entry(&entry)?;

        // Update indices
        self.type_index
            .entry(memory_type)
            .or_default()
            .push(id.clone());

        // Update ID-to-type index for O(1) lookups
        self.id_type_index.insert(id.clone(), memory_type);

        for tag in &tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .push(id.clone());
        }

        // Add to cache
        self.cache.insert(id.clone(), entry);

        // Prune cache if needed
        self.prune_cache();

        Ok(id)
    }

    /// Retrieves a memory entry by ID.
    pub fn get(&mut self, id: &str) -> Option<&MemoryEntry> {
        // Try cache first
        if let Some(entry) = self.cache.get_mut(id) {
            entry.record_access();
            return self.cache.get(id);
        }

        // Load from disk if not in cache
        if let Ok(entry) = self.load_entry(id) {
            self.cache.insert(id.to_string(), entry);
            return self.cache.get(id);
        }

        None
    }

    /// Searches for memories matching a query.
    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        self.cache.values().filter(|e| e.matches(query)).collect()
    }

    /// Gets memories by type.
    pub fn get_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryEntry> {
        self.type_index
            .get(&memory_type)
            .map(|ids| ids.iter().filter_map(|id| self.cache.get(id)).collect())
            .unwrap_or_default()
    }

    /// Gets memories by tag.
    pub fn get_by_tag(&self, tag: &str) -> Vec<&MemoryEntry> {
        self.tag_index
            .get(tag)
            .map(|ids| ids.iter().filter_map(|id| self.cache.get(id)).collect())
            .unwrap_or_default()
    }

    /// Gets the most important memories.
    pub fn get_important(
        &self,
        min_importance: ImportanceLevel,
        limit: usize,
    ) -> Vec<&MemoryEntry> {
        let mut entries: Vec<_> = self
            .cache
            .values()
            .filter(|e| e.importance >= min_importance)
            .collect();

        entries.sort_by(|a, b| b.importance.cmp(&a.importance));
        entries.truncate(limit);
        entries
    }

    /// Gets recently accessed memories.
    pub fn get_recent(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut entries: Vec<_> = self.cache.values().collect();
        entries.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
        entries.truncate(limit);
        entries
    }

    /// Deletes a memory entry.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(entry) = self.cache.remove(id) {
            // Remove from type index
            if let Some(ids) = self.type_index.get_mut(&entry.memory_type) {
                ids.retain(|i| i != id);
            }

            // Remove from ID-to-type index
            self.id_type_index.remove(id);

            // Remove from tag indices
            for tag in &entry.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|i| i != id);
                }
            }

            // Delete from disk
            let path = self.entry_path(&entry);
            if path.exists() {
                fs::remove_file(&path).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to delete: {}", e))
                })?;
            }

            return Ok(true);
        }
        Ok(false)
    }

    /// Returns statistics about the memory store.
    pub fn stats(&self) -> MemoryStats {
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut by_importance: HashMap<String, usize> = HashMap::new();
        let mut oldest: Option<u64> = None;
        let mut newest: Option<u64> = None;
        let mut total_size = 0u64;

        for entry in self.cache.values() {
            *by_type
                .entry(format!("{:?}", entry.memory_type))
                .or_default() += 1;
            *by_importance
                .entry(format!("{:?}", entry.importance))
                .or_default() += 1;

            oldest = Some(oldest.map_or(entry.created_at, |o| o.min(entry.created_at)));
            newest = Some(newest.map_or(entry.created_at, |o| o.max(entry.created_at)));
            total_size += entry.content.len() as u64;
        }

        MemoryStats {
            total_entries: self.cache.len(),
            by_type,
            by_importance,
            total_size_bytes: total_size,
            oldest_entry: oldest,
            newest_entry: newest,
        }
    }

    /// Prunes low-importance entries if cache is too large.
    fn prune_cache(&mut self) {
        if self.cache.len() <= self.max_cache_size {
            return;
        }

        // Sort by importance (ascending) and last access (ascending)
        let mut entries: Vec<_> = self.cache.keys().cloned().collect();
        entries.sort_by(|a, b| {
            let entry_a = self.cache.get(a);
            let entry_b = self.cache.get(b);
            match (entry_a, entry_b) {
                (Some(a), Some(b)) => a
                    .importance
                    .cmp(&b.importance)
                    .then(a.last_accessed.cmp(&b.last_accessed)),
                _ => std::cmp::Ordering::Equal,
            }
        });

        // Remove lowest priority entries
        let to_remove = self.cache.len() - self.max_cache_size;
        for id in entries.into_iter().take(to_remove) {
            // Don't remove critical entries
            if let Some(entry) = self.cache.get(&id) {
                if entry.importance != ImportanceLevel::Critical {
                    self.cache.remove(&id);
                }
            }
        }
    }

    fn entry_path(&self, entry: &MemoryEntry) -> PathBuf {
        self.base_dir
            .join(entry.memory_type.dir_name())
            .join(format!("{}.json", entry.id))
    }

    fn write_entry(&self, entry: &MemoryEntry) -> Result<()> {
        let path = self.entry_path(entry);
        let json = serde_json::to_string_pretty(entry)
            .map_err(|e| infernum_core::Error::internal(format!("Failed to serialize: {}", e)))?;
        fs::write(&path, json)
            .map_err(|e| infernum_core::Error::internal(format!("Failed to write: {}", e)))?;
        Ok(())
    }

    fn load_entry(&self, id: &str) -> Result<MemoryEntry> {
        // Use ID-to-type index for O(1) lookup if available
        if let Some(memory_type) = self.id_type_index.get(id) {
            let path = self
                .base_dir
                .join(memory_type.dir_name())
                .join(format!("{}.json", id));
            if path.exists() {
                let content = fs::read_to_string(&path).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to read: {}", e))
                })?;
                let entry: MemoryEntry = serde_json::from_str(&content).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to parse: {}", e))
                })?;
                return Ok(entry);
            }
        }

        // Fallback: search in all type directories (for entries not yet indexed)
        for memory_type in [
            MemoryType::SessionSummary,
            MemoryType::ProjectLearning,
            MemoryType::Decision,
            MemoryType::Context,
            MemoryType::ErrorPattern,
            MemoryType::Optimization,
            MemoryType::UserPreference,
            MemoryType::CodePattern,
        ] {
            let path = self
                .base_dir
                .join(memory_type.dir_name())
                .join(format!("{}.json", id));
            if path.exists() {
                let content = fs::read_to_string(&path).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to read: {}", e))
                })?;
                let entry: MemoryEntry = serde_json::from_str(&content).map_err(|e| {
                    infernum_core::Error::internal(format!("Failed to parse: {}", e))
                })?;
                return Ok(entry);
            }
        }
        Err(infernum_core::Error::internal(format!(
            "Entry not found: {}",
            id
        )))
    }

    fn load_all(&mut self) -> Result<()> {
        for memory_type in [
            MemoryType::SessionSummary,
            MemoryType::ProjectLearning,
            MemoryType::Decision,
            MemoryType::Context,
            MemoryType::ErrorPattern,
            MemoryType::Optimization,
            MemoryType::UserPreference,
            MemoryType::CodePattern,
        ] {
            let type_dir = self.base_dir.join(memory_type.dir_name());
            if let Ok(entries) = fs::read_dir(&type_dir) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |e| e == "json") {
                        if let Ok(content) = fs::read_to_string(entry.path()) {
                            if let Ok(mem_entry) = serde_json::from_str::<MemoryEntry>(&content) {
                                let id = mem_entry.id.clone();
                                let tags = mem_entry.tags.clone();

                                self.type_index
                                    .entry(memory_type)
                                    .or_default()
                                    .push(id.clone());

                                // Populate ID-to-type index for O(1) lookups
                                self.id_type_index.insert(id.clone(), memory_type);

                                for tag in &tags {
                                    self.tag_index
                                        .entry(tag.clone())
                                        .or_default()
                                        .push(id.clone());
                                }

                                self.cache.insert(id, mem_entry);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Generates a unique ID.
fn generate_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{:x}-{:04x}", timestamp, counter)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_memory() -> (LongTermMemory, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let memory = LongTermMemory::new(temp_dir.path()).expect("Failed to create memory");
        (memory, temp_dir)
    }

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new(MemoryType::SessionSummary, "Test content");
        assert_eq!(entry.memory_type, MemoryType::SessionSummary);
        assert_eq!(entry.content, "Test content");
        assert_eq!(entry.importance, ImportanceLevel::Medium);
        assert!(entry.tags.is_empty());
    }

    #[test]
    fn test_memory_entry_builder() {
        let entry = MemoryEntry::new(MemoryType::Decision, "Important decision")
            .with_importance(ImportanceLevel::High)
            .with_tag("architecture")
            .with_tag("database")
            .with_summary("Summary of decision")
            .with_metadata("author", "test");

        assert_eq!(entry.importance, ImportanceLevel::High);
        assert_eq!(entry.tags.len(), 2);
        assert!(entry.tags.contains(&"architecture".to_string()));
        assert_eq!(entry.summary, Some("Summary of decision".to_string()));
        assert_eq!(entry.metadata.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_memory_entry_matches() {
        let entry = MemoryEntry::new(MemoryType::Context, "Rust programming language")
            .with_tag("rust")
            .with_summary("About Rust");

        assert!(entry.matches("rust"));
        assert!(entry.matches("RUST")); // case insensitive
        assert!(entry.matches("programming"));
        assert!(!entry.matches("python"));
    }

    #[test]
    fn test_long_term_memory_store_and_retrieve() {
        let (mut memory, _temp) = create_test_memory();

        let entry = MemoryEntry::new(MemoryType::ProjectLearning, "Learning about Rust")
            .with_importance(ImportanceLevel::High)
            .with_tag("rust");

        let id = memory.store(entry).expect("Failed to store");
        let retrieved = memory.get(&id).expect("Failed to retrieve");

        assert_eq!(retrieved.content, "Learning about Rust");
        assert_eq!(retrieved.importance, ImportanceLevel::High);
    }

    #[test]
    fn test_long_term_memory_search() {
        let (mut memory, _temp) = create_test_memory();

        memory
            .store(MemoryEntry::new(
                MemoryType::Context,
                "Rust is a systems language",
            ))
            .ok();
        memory
            .store(MemoryEntry::new(
                MemoryType::Context,
                "Python is interpreted",
            ))
            .ok();
        memory
            .store(MemoryEntry::new(
                MemoryType::Context,
                "Rust has zero-cost abstractions",
            ))
            .ok();

        let results = memory.search("rust");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_long_term_memory_get_by_type() {
        let (mut memory, _temp) = create_test_memory();

        memory
            .store(MemoryEntry::new(MemoryType::Decision, "Decision 1"))
            .ok();
        memory
            .store(MemoryEntry::new(MemoryType::Decision, "Decision 2"))
            .ok();
        memory
            .store(MemoryEntry::new(MemoryType::Context, "Context 1"))
            .ok();

        let decisions = memory.get_by_type(MemoryType::Decision);
        assert_eq!(decisions.len(), 2);
    }

    #[test]
    fn test_long_term_memory_get_by_tag() {
        let (mut memory, _temp) = create_test_memory();

        memory
            .store(MemoryEntry::new(MemoryType::Context, "Content 1").with_tag("important"))
            .ok();
        memory
            .store(MemoryEntry::new(MemoryType::Context, "Content 2").with_tag("important"))
            .ok();
        memory
            .store(MemoryEntry::new(MemoryType::Context, "Content 3").with_tag("other"))
            .ok();

        let important = memory.get_by_tag("important");
        assert_eq!(important.len(), 2);
    }

    #[test]
    fn test_long_term_memory_get_important() {
        let (mut memory, _temp) = create_test_memory();

        memory
            .store(
                MemoryEntry::new(MemoryType::Context, "Low").with_importance(ImportanceLevel::Low),
            )
            .ok();
        memory
            .store(
                MemoryEntry::new(MemoryType::Context, "High")
                    .with_importance(ImportanceLevel::High),
            )
            .ok();
        memory
            .store(
                MemoryEntry::new(MemoryType::Context, "Critical")
                    .with_importance(ImportanceLevel::Critical),
            )
            .ok();

        let important = memory.get_important(ImportanceLevel::High, 10);
        assert_eq!(important.len(), 2);
    }

    #[test]
    fn test_long_term_memory_delete() {
        let (mut memory, _temp) = create_test_memory();

        let id = memory
            .store(MemoryEntry::new(MemoryType::Context, "To delete"))
            .expect("Failed to store");

        assert!(memory.get(&id).is_some());
        assert!(memory.delete(&id).expect("Failed to delete"));
        assert!(memory.get(&id).is_none());
    }

    #[test]
    fn test_long_term_memory_stats() {
        let (mut memory, _temp) = create_test_memory();

        memory
            .store(
                MemoryEntry::new(MemoryType::Decision, "D1").with_importance(ImportanceLevel::High),
            )
            .ok();
        memory
            .store(
                MemoryEntry::new(MemoryType::Decision, "D2").with_importance(ImportanceLevel::Low),
            )
            .ok();
        memory
            .store(MemoryEntry::new(MemoryType::Context, "C1"))
            .ok();

        let stats = memory.stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.by_type.get("Decision"), Some(&2));
        assert_eq!(stats.by_type.get("Context"), Some(&1));
    }

    #[test]
    fn test_memory_type_dir_name() {
        assert_eq!(MemoryType::SessionSummary.dir_name(), "sessions");
        assert_eq!(MemoryType::ProjectLearning.dir_name(), "learnings");
        assert_eq!(MemoryType::Decision.dir_name(), "decisions");
    }

    #[test]
    fn test_importance_ordering() {
        assert!(ImportanceLevel::Critical > ImportanceLevel::High);
        assert!(ImportanceLevel::High > ImportanceLevel::Medium);
        assert!(ImportanceLevel::Medium > ImportanceLevel::Low);
    }
}
