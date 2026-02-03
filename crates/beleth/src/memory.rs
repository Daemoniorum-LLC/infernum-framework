//! Agent memory systems with conversation persistence.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use abaddon::{Engine, InferenceEngine};
use async_trait::async_trait;
use infernum_core::{GenerateRequest, Message, Result, Role, SamplingParams};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Memory summarization strategy.
#[derive(Debug, Clone, Copy, Default)]
pub enum SummarizationStrategy {
    /// Drop oldest messages (no summarization).
    #[default]
    DropOldest,
    /// Summarize oldest messages into a single message.
    Summarize,
    /// Use a sliding window with overlap.
    SlidingWindow {
        /// Number of messages to keep from the window.
        keep_recent: usize,
    },
}

/// A summary of previous conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    /// The summary text.
    pub text: String,
    /// Number of messages summarized.
    pub message_count: usize,
    /// Timestamp of summarization.
    pub created_at: u64,
}

/// Agent memory containing conversation history.
pub struct AgentMemory {
    /// Working memory (current conversation).
    messages: Vec<Message>,
    /// Maximum messages before summarization.
    max_messages: usize,
    /// Summarization strategy.
    strategy: SummarizationStrategy,
    /// Previous summaries.
    summaries: Vec<ConversationSummary>,
    /// Optional engine for LLM-based summarization.
    engine: Option<Arc<Engine>>,
    /// Number of messages to summarize at once.
    summarize_batch_size: usize,
}

impl AgentMemory {
    /// Creates a new agent memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: 50,
            strategy: SummarizationStrategy::default(),
            summaries: Vec::new(),
            engine: None,
            summarize_batch_size: 10,
        }
    }

    /// Creates with a custom max messages limit.
    #[must_use]
    pub fn with_max_messages(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
            strategy: SummarizationStrategy::default(),
            summaries: Vec::new(),
            engine: None,
            summarize_batch_size: 10,
        }
    }

    /// Sets the summarization strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SummarizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the engine for LLM-based summarization.
    #[must_use]
    pub fn with_engine(mut self, engine: Arc<Engine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Sets the batch size for summarization.
    #[must_use]
    pub fn with_summarize_batch_size(mut self, size: usize) -> Self {
        self.summarize_batch_size = size;
        self
    }

    /// Adds a message to memory.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);

        // Check if we need to manage memory
        if self.messages.len() > self.max_messages {
            self.manage_memory_sync();
        }
    }

    /// Adds a message and performs async summarization if needed.
    pub async fn add_message_async(&mut self, message: Message) -> Result<()> {
        self.messages.push(message);

        if self.messages.len() > self.max_messages {
            self.manage_memory().await?;
        }

        Ok(())
    }

    /// Synchronous memory management (fallback).
    fn manage_memory_sync(&mut self) {
        match self.strategy {
            SummarizationStrategy::DropOldest => {
                self.drop_oldest_messages();
            },
            SummarizationStrategy::Summarize => {
                // Fall back to drop oldest if no engine available
                if self.engine.is_none() {
                    self.drop_oldest_messages();
                } else {
                    // Will be handled by async version
                    self.drop_oldest_messages();
                }
            },
            SummarizationStrategy::SlidingWindow { keep_recent } => {
                self.apply_sliding_window(keep_recent);
            },
        }
    }

    /// Async memory management with LLM summarization.
    async fn manage_memory(&mut self) -> Result<()> {
        match self.strategy {
            SummarizationStrategy::DropOldest => {
                self.drop_oldest_messages();
            },
            SummarizationStrategy::Summarize => {
                if let Some(engine) = &self.engine {
                    self.summarize_messages(engine.clone()).await?;
                } else {
                    self.drop_oldest_messages();
                }
            },
            SummarizationStrategy::SlidingWindow { keep_recent } => {
                self.apply_sliding_window(keep_recent);
            },
        }
        Ok(())
    }

    /// Drops oldest non-system messages.
    fn drop_oldest_messages(&mut self) {
        // Keep system messages and recent messages
        let to_remove = self.messages.len().saturating_sub(self.max_messages);

        if to_remove > 0 {
            // Find and remove oldest non-system messages
            let mut removed = 0;
            self.messages.retain(|m| {
                if removed >= to_remove {
                    return true;
                }
                if matches!(m.role, Role::System) {
                    return true;
                }
                removed += 1;
                false
            });
        }
    }

    /// Applies a sliding window strategy.
    fn apply_sliding_window(&mut self, keep_recent: usize) {
        if self.messages.len() <= keep_recent {
            return;
        }

        // Separate system messages
        let system_messages: Vec<_> = self
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .cloned()
            .collect();

        // Get recent non-system messages
        let non_system: Vec<_> = self
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .cloned()
            .collect();

        let keep_count = keep_recent.min(non_system.len());
        let recent: Vec<_> = non_system
            .into_iter()
            .rev()
            .take(keep_count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Rebuild messages
        self.messages = system_messages;
        self.messages.extend(recent);
    }

    /// Summarizes older messages using LLM.
    async fn summarize_messages(&mut self, engine: Arc<Engine>) -> Result<()> {
        // Find non-system messages to summarize
        let non_system_indices: Vec<_> = self
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| !matches!(m.role, Role::System))
            .map(|(i, _)| i)
            .collect();

        if non_system_indices.len() <= self.summarize_batch_size {
            return Ok(());
        }

        // Take oldest batch for summarization
        let summarize_count = self.summarize_batch_size.min(
            non_system_indices
                .len()
                .saturating_sub(self.max_messages / 2),
        );

        if summarize_count == 0 {
            return Ok(());
        }

        let indices_to_summarize: Vec<_> = non_system_indices
            .iter()
            .take(summarize_count)
            .copied()
            .collect();

        // Extract messages to summarize
        let messages_to_summarize: Vec<_> = indices_to_summarize
            .iter()
            .filter_map(|&i| self.messages.get(i).cloned())
            .collect();

        // Generate summary using LLM
        let summary_text = self
            .generate_summary(&engine, &messages_to_summarize)
            .await?;

        // Store the summary
        let summary = ConversationSummary {
            text: summary_text.clone(),
            message_count: messages_to_summarize.len(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.summaries.push(summary);

        // Remove summarized messages (in reverse order to preserve indices)
        for &idx in indices_to_summarize.iter().rev() {
            if idx < self.messages.len() {
                self.messages.remove(idx);
            }
        }

        // Insert summary as a system message after the original system prompt
        let insert_pos = self
            .messages
            .iter()
            .position(|m| !matches!(m.role, Role::System))
            .unwrap_or(self.messages.len());

        self.messages.insert(
            insert_pos,
            Message {
                role: Role::System,
                content: format!("[Previous conversation summary: {}]", summary_text),
                name: Some("memory_summary".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        );

        tracing::debug!(
            summarized_count = messages_to_summarize.len(),
            remaining_messages = self.messages.len(),
            "Summarized conversation history"
        );

        Ok(())
    }

    /// Generates a summary of messages using the LLM.
    async fn generate_summary(&self, engine: &Engine, messages: &[Message]) -> Result<String> {
        // Format messages for summarization
        let conversation = messages
            .iter()
            .map(|m| format!("{}: {}", role_to_str(&m.role), m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Summarize the following conversation in a concise paragraph.
Focus on:
- Key information exchanged
- Decisions made
- Important context for continuing the conversation

Conversation:
{}

Summary:"#,
            conversation
        );

        let request = GenerateRequest::new(prompt).with_sampling(
            SamplingParams::default()
                .with_max_tokens(256)
                .with_temperature(0.3),
        );

        let response = engine.generate(request).await?;
        let summary = response
            .choices
            .first()
            .map(|c| c.text.trim().to_string())
            .unwrap_or_else(|| "Previous conversation context.".to_string());

        Ok(summary)
    }

    /// Returns all messages including summary context.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns all messages without summary messages.
    #[must_use]
    pub fn messages_without_summaries(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.name.as_deref() != Some("memory_summary"))
            .collect()
    }

    /// Returns the conversation summaries.
    #[must_use]
    pub fn summaries(&self) -> &[ConversationSummary] {
        &self.summaries
    }

    /// Clears non-system messages.
    pub fn clear(&mut self) {
        self.messages.retain(|m| matches!(m.role, Role::System));
        self.summaries.clear();
    }

    /// Clears all messages including system messages.
    pub fn clear_all(&mut self) {
        self.messages.clear();
        self.summaries.clear();
    }

    /// Returns the number of messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns true if memory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns the total number of messages ever processed (including summarized).
    #[must_use]
    pub fn total_processed(&self) -> usize {
        let summarized: usize = self.summaries.iter().map(|s| s.message_count).sum();
        self.messages.len() + summarized
    }

    /// Gets the context window size estimate (for token budgeting).
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        // Rough estimate: ~4 chars per token
        self.messages.iter().map(|m| m.content.len() / 4).sum()
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a role to string for display.
fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
    }
}

// ============================================================================
// Conversation Persistence
// ============================================================================

/// Serializable version of a Message for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMessage {
    /// Role of the message sender.
    pub role: String,
    /// Content of the message.
    pub content: String,
    /// Optional name.
    pub name: Option<String>,
    /// Optional tool call ID.
    pub tool_call_id: Option<String>,
}

impl From<&Message> for SerializableMessage {
    fn from(msg: &Message) -> Self {
        Self {
            role: role_to_str(&msg.role).to_lowercase(),
            content: msg.content.clone(),
            name: msg.name.clone(),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

impl SerializableMessage {
    /// Converts back to a Message.
    pub fn to_message(&self) -> Message {
        let role = match self.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };

        Message {
            role,
            content: self.content.clone(),
            name: self.name.clone(),
            tool_calls: None,
            tool_call_id: self.tool_call_id.clone(),
        }
    }
}

/// A serializable conversation for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConversation {
    /// Unique conversation ID.
    pub id: String,
    /// Conversation title/name.
    pub title: Option<String>,
    /// Messages in the conversation.
    pub messages: Vec<SerializableMessage>,
    /// Conversation summaries.
    pub summaries: Vec<ConversationSummary>,
    /// Creation timestamp (Unix epoch seconds).
    pub created_at: u64,
    /// Last updated timestamp.
    pub updated_at: u64,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

impl SerializableConversation {
    /// Creates a new conversation with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: id.into(),
            title: None,
            messages: Vec::new(),
            summaries: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Creates from AgentMemory.
    pub fn from_memory(id: impl Into<String>, memory: &AgentMemory) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: id.into(),
            title: None,
            messages: memory.messages().iter().map(SerializableMessage::from).collect(),
            summaries: memory.summaries().to_vec(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Restores messages to an AgentMemory.
    pub fn restore_to_memory(&self, memory: &mut AgentMemory) {
        memory.clear_all();
        for msg in &self.messages {
            memory.add_message(msg.to_message());
        }
    }

    /// Sets the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Trait for conversation storage backends.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Saves a conversation.
    async fn save(&self, conversation: &SerializableConversation) -> Result<()>;

    /// Loads a conversation by ID.
    async fn load(&self, id: &str) -> Result<Option<SerializableConversation>>;

    /// Lists all conversation IDs.
    async fn list(&self) -> Result<Vec<String>>;

    /// Deletes a conversation.
    async fn delete(&self, id: &str) -> Result<bool>;

    /// Checks if a conversation exists.
    async fn exists(&self, id: &str) -> Result<bool>;
}

/// In-memory conversation store (for testing/temporary storage).
pub struct MemoryConversationStore {
    conversations: RwLock<HashMap<String, SerializableConversation>>,
}

impl MemoryConversationStore {
    /// Creates a new in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conversations: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConversationStore for MemoryConversationStore {
    async fn save(&self, conversation: &SerializableConversation) -> Result<()> {
        let mut conversations = self.conversations.write();
        conversations.insert(conversation.id.clone(), conversation.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<SerializableConversation>> {
        let conversations = self.conversations.read();
        Ok(conversations.get(id).cloned())
    }

    async fn list(&self) -> Result<Vec<String>> {
        let conversations = self.conversations.read();
        Ok(conversations.keys().cloned().collect())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let mut conversations = self.conversations.write();
        Ok(conversations.remove(id).is_some())
    }

    async fn exists(&self, id: &str) -> Result<bool> {
        let conversations = self.conversations.read();
        Ok(conversations.contains_key(id))
    }
}

/// File-based conversation store.
pub struct FileConversationStore {
    /// Base directory for conversation files.
    base_dir: PathBuf,
}

impl FileConversationStore {
    /// Creates a new file store in the given directory.
    pub fn new(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !base_dir.exists() {
            std::fs::create_dir_all(&base_dir).map_err(|e| {
                infernum_core::Error::internal(format!("Failed to create conversation directory: {}", e))
            })?;
        }

        Ok(Self { base_dir })
    }

    /// Returns the file path for a conversation ID.
    fn conversation_path(&self, id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.json", id))
    }
}

#[async_trait]
impl ConversationStore for FileConversationStore {
    async fn save(&self, conversation: &SerializableConversation) -> Result<()> {
        let path = self.conversation_path(&conversation.id);

        let json = serde_json::to_string_pretty(conversation).map_err(|e| {
            infernum_core::Error::internal(format!("Failed to serialize conversation: {}", e))
        })?;

        tokio::fs::write(&path, json).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to write conversation file: {}", e))
        })?;

        tracing::debug!(
            conversation_id = %conversation.id,
            path = %path.display(),
            "Saved conversation"
        );

        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<SerializableConversation>> {
        let path = self.conversation_path(id);

        if !path.exists() {
            return Ok(None);
        }

        let contents = tokio::fs::read_to_string(&path).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to read conversation file: {}", e))
        })?;

        let conversation: SerializableConversation = serde_json::from_str(&contents).map_err(|e| {
            infernum_core::Error::internal(format!("Failed to parse conversation: {}", e))
        })?;

        tracing::debug!(
            conversation_id = %id,
            messages = conversation.messages.len(),
            "Loaded conversation"
        );

        Ok(Some(conversation))
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();

        let mut entries = tokio::fs::read_dir(&self.base_dir).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to read conversation directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }

        Ok(ids)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let path = self.conversation_path(id);

        if !path.exists() {
            return Ok(false);
        }

        tokio::fs::remove_file(&path).await.map_err(|e| {
            infernum_core::Error::internal(format!("Failed to delete conversation: {}", e))
        })?;

        tracing::debug!(conversation_id = %id, "Deleted conversation");

        Ok(true)
    }

    async fn exists(&self, id: &str) -> Result<bool> {
        let path = self.conversation_path(id);
        Ok(path.exists())
    }
}

/// Persistent conversation that wraps AgentMemory with auto-save.
pub struct PersistentConversation {
    /// Unique conversation ID.
    id: String,
    /// The underlying memory.
    memory: AgentMemory,
    /// The storage backend.
    store: Arc<dyn ConversationStore>,
    /// Whether to auto-save on each message.
    auto_save: bool,
    /// Title of the conversation.
    title: Option<String>,
    /// Metadata.
    metadata: HashMap<String, String>,
}

impl PersistentConversation {
    /// Creates a new persistent conversation.
    pub fn new(id: impl Into<String>, store: Arc<dyn ConversationStore>) -> Self {
        Self {
            id: id.into(),
            memory: AgentMemory::new(),
            store,
            auto_save: true,
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates with a custom memory configuration.
    pub fn with_memory(id: impl Into<String>, memory: AgentMemory, store: Arc<dyn ConversationStore>) -> Self {
        Self {
            id: id.into(),
            memory,
            store,
            auto_save: true,
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Loads an existing conversation or creates a new one.
    pub async fn load_or_create(id: impl Into<String>, store: Arc<dyn ConversationStore>) -> Result<Self> {
        let id = id.into();

        if let Some(saved) = store.load(&id).await? {
            let mut memory = AgentMemory::new();
            saved.restore_to_memory(&mut memory);

            Ok(Self {
                id,
                memory,
                store,
                auto_save: true,
                title: saved.title,
                metadata: saved.metadata,
            })
        } else {
            Ok(Self::new(id, store))
        }
    }

    /// Sets whether to auto-save on each message.
    #[must_use]
    pub fn with_auto_save(mut self, auto_save: bool) -> Self {
        self.auto_save = auto_save;
        self
    }

    /// Sets the conversation title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Returns the conversation ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the underlying memory.
    #[must_use]
    pub fn memory(&self) -> &AgentMemory {
        &self.memory
    }

    /// Returns a mutable reference to the underlying memory.
    pub fn memory_mut(&mut self) -> &mut AgentMemory {
        &mut self.memory
    }

    /// Adds a message and optionally saves.
    pub async fn add_message(&mut self, message: Message) -> Result<()> {
        self.memory.add_message(message);

        if self.auto_save {
            self.save().await?;
        }

        Ok(())
    }

    /// Saves the conversation to storage.
    pub async fn save(&self) -> Result<()> {
        let mut conversation = SerializableConversation::from_memory(&self.id, &self.memory);
        conversation.title = self.title.clone();
        conversation.metadata = self.metadata.clone();
        conversation.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.store.save(&conversation).await
    }

    /// Reloads the conversation from storage.
    pub async fn reload(&mut self) -> Result<bool> {
        if let Some(saved) = self.store.load(&self.id).await? {
            saved.restore_to_memory(&mut self.memory);
            self.title = saved.title;
            self.metadata = saved.metadata;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clears the conversation and optionally deletes from storage.
    pub async fn clear(&mut self, delete_from_storage: bool) -> Result<()> {
        self.memory.clear();
        self.metadata.clear();

        if delete_from_storage {
            self.store.delete(&self.id).await?;
        } else if self.auto_save {
            self.save().await?;
        }

        Ok(())
    }

    /// Returns the number of messages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns true if the conversation is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Returns the conversation title.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Returns the messages.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        self.memory.messages()
    }
}

/// Manager for multiple persistent conversations.
pub struct ConversationManager {
    store: Arc<dyn ConversationStore>,
    active: RwLock<HashMap<String, Arc<RwLock<PersistentConversation>>>>,
}

impl ConversationManager {
    /// Creates a new conversation manager with the given store.
    pub fn new(store: Arc<dyn ConversationStore>) -> Self {
        Self {
            store,
            active: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a manager with file-based storage.
    pub fn with_file_store(base_dir: impl AsRef<Path>) -> Result<Self> {
        let store = FileConversationStore::new(base_dir)?;
        Ok(Self::new(Arc::new(store)))
    }

    /// Creates a manager with in-memory storage.
    #[must_use]
    pub fn with_memory_store() -> Self {
        Self::new(Arc::new(MemoryConversationStore::new()))
    }

    /// Creates or loads a conversation.
    pub async fn get_or_create(&self, id: impl Into<String>) -> Result<Arc<RwLock<PersistentConversation>>> {
        let id = id.into();

        // Check if already active
        {
            let active = self.active.read();
            if let Some(conv) = active.get(&id) {
                return Ok(Arc::clone(conv));
            }
        }

        // Load or create
        let conversation = PersistentConversation::load_or_create(&id, Arc::clone(&self.store)).await?;
        let conversation = Arc::new(RwLock::new(conversation));

        // Store in active conversations
        {
            let mut active = self.active.write();
            active.insert(id.clone(), Arc::clone(&conversation));
        }

        Ok(conversation)
    }

    /// Lists all conversation IDs (both active and stored).
    pub async fn list_all(&self) -> Result<Vec<String>> {
        self.store.list().await
    }

    /// Deletes a conversation.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        // Remove from active
        {
            let mut active = self.active.write();
            active.remove(id);
        }

        // Delete from storage
        self.store.delete(id).await
    }

    /// Returns the number of active (loaded) conversations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }

    /// Saves all active conversations.
    pub async fn save_all(&self) -> Result<()> {
        let active = self.active.read();
        for conversation in active.values() {
            conversation.read().save().await?;
        }
        Ok(())
    }

    /// Unloads inactive conversations to free memory.
    pub fn unload_conversation(&self, id: &str) {
        let mut active = self.active.write();
        active.remove(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==========================================================================
    // SummarizationStrategy Tests
    // ==========================================================================

    #[test]
    fn test_summarization_strategy_default() {
        let strategy = SummarizationStrategy::default();
        assert!(matches!(strategy, SummarizationStrategy::DropOldest));
    }

    #[test]
    fn test_summarization_strategy_drop_oldest() {
        let strategy = SummarizationStrategy::DropOldest;
        assert!(matches!(strategy, SummarizationStrategy::DropOldest));
    }

    #[test]
    fn test_summarization_strategy_summarize() {
        let strategy = SummarizationStrategy::Summarize;
        assert!(matches!(strategy, SummarizationStrategy::Summarize));
    }

    #[test]
    fn test_summarization_strategy_sliding_window() {
        let strategy = SummarizationStrategy::SlidingWindow { keep_recent: 10 };
        if let SummarizationStrategy::SlidingWindow { keep_recent } = strategy {
            assert_eq!(keep_recent, 10);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_summarization_strategy_clone() {
        let strategy = SummarizationStrategy::SlidingWindow { keep_recent: 5 };
        let cloned = strategy;
        assert!(matches!(cloned, SummarizationStrategy::SlidingWindow { keep_recent: 5 }));
    }

    // ==========================================================================
    // ConversationSummary Tests
    // ==========================================================================

    #[test]
    fn test_conversation_summary_creation() {
        let summary = ConversationSummary {
            text: "Summary of conversation".to_string(),
            message_count: 10,
            created_at: 1234567890,
        };

        assert_eq!(summary.text, "Summary of conversation");
        assert_eq!(summary.message_count, 10);
        assert_eq!(summary.created_at, 1234567890);
    }

    #[test]
    fn test_conversation_summary_serialization() {
        let summary = ConversationSummary {
            text: "Test summary".to_string(),
            message_count: 5,
            created_at: 1000,
        };

        let json = serde_json::to_string(&summary).expect("serialize");
        assert!(json.contains("Test summary"));

        let parsed: ConversationSummary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.text, summary.text);
        assert_eq!(parsed.message_count, summary.message_count);
    }

    #[test]
    fn test_conversation_summary_clone() {
        let summary = ConversationSummary {
            text: "Clone test".to_string(),
            message_count: 3,
            created_at: 500,
        };

        let cloned = summary.clone();
        assert_eq!(cloned.text, summary.text);
        assert_eq!(cloned.message_count, summary.message_count);
    }

    // ==========================================================================
    // AgentMemory Basic Tests
    // ==========================================================================

    #[test]
    fn test_memory_new() {
        let memory = AgentMemory::new();
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_memory_default() {
        let memory = AgentMemory::default();
        assert!(memory.is_empty());
    }

    #[test]
    fn test_memory_with_max_messages() {
        let memory = AgentMemory::with_max_messages(100);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_memory_basic() {
        let mut memory = AgentMemory::new();
        assert!(memory.is_empty());

        memory.add_message(Message::user("Hello"));
        assert_eq!(memory.len(), 1);

        memory.add_message(Message::assistant("Hi there!"));
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_memory_messages() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("Test"));

        let messages = memory.messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Test");
    }

    #[test]
    fn test_memory_with_strategy() {
        let memory = AgentMemory::new()
            .with_strategy(SummarizationStrategy::Summarize);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_memory_with_summarize_batch_size() {
        let memory = AgentMemory::new()
            .with_summarize_batch_size(20);
        assert!(memory.is_empty());
    }

    // ==========================================================================
    // AgentMemory Drop Oldest Tests
    // ==========================================================================

    #[test]
    fn test_memory_drop_oldest() {
        let mut memory = AgentMemory::with_max_messages(3);

        memory.add_message(Message::user("Message 1"));
        memory.add_message(Message::assistant("Response 1"));
        memory.add_message(Message::user("Message 2"));
        assert_eq!(memory.len(), 3);

        // Adding a 4th message should trigger drop
        memory.add_message(Message::assistant("Response 2"));
        assert_eq!(memory.len(), 3);

        // The oldest message should be dropped
        assert_eq!(memory.messages()[0].content, "Response 1");
    }

    #[test]
    fn test_memory_preserves_system() {
        let mut memory = AgentMemory::with_max_messages(2);

        memory.add_message(Message {
            role: Role::System,
            content: "System prompt".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
        memory.add_message(Message::user("User 1"));
        memory.add_message(Message::user("User 2"));
        memory.add_message(Message::user("User 3"));

        // System message should be preserved
        assert!(memory
            .messages()
            .iter()
            .any(|m| m.content == "System prompt"));
    }

    // ==========================================================================
    // AgentMemory Sliding Window Tests
    // ==========================================================================

    #[test]
    fn test_sliding_window() {
        let mut memory = AgentMemory::with_max_messages(5)
            .with_strategy(SummarizationStrategy::SlidingWindow { keep_recent: 2 });

        for i in 0..10 {
            memory.add_message(Message::user(format!("Message {}", i)));
        }

        // Should keep only the most recent 2 non-system messages
        let non_system: Vec<_> = memory
            .messages()
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .collect();

        assert!(non_system.len() <= 2);
    }

    #[test]
    fn test_sliding_window_with_system() {
        let mut memory = AgentMemory::with_max_messages(3)
            .with_strategy(SummarizationStrategy::SlidingWindow { keep_recent: 2 });

        memory.add_message(Message {
            role: Role::System,
            content: "System".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        for i in 0..5 {
            memory.add_message(Message::user(format!("Msg {}", i)));
        }

        // System should be preserved
        assert!(memory.messages().iter().any(|m| matches!(m.role, Role::System)));
    }

    // ==========================================================================
    // AgentMemory Clear Tests
    // ==========================================================================

    #[test]
    fn test_clear() {
        let mut memory = AgentMemory::new();

        memory.add_message(Message {
            role: Role::System,
            content: "System".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
        memory.add_message(Message::user("User"));
        memory.add_message(Message::assistant("Assistant"));

        memory.clear();

        // Only system message should remain
        assert_eq!(memory.len(), 1);
        assert!(matches!(memory.messages()[0].role, Role::System));
    }

    #[test]
    fn test_clear_all() {
        let mut memory = AgentMemory::new();

        memory.add_message(Message {
            role: Role::System,
            content: "System".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
        memory.add_message(Message::user("User"));

        memory.clear_all();

        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }

    // ==========================================================================
    // AgentMemory Utility Methods Tests
    // ==========================================================================

    #[test]
    fn test_estimated_tokens() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("Hello world")); // ~3 tokens
        memory.add_message(Message::assistant("Hi there friend")); // ~4 tokens

        let estimate = memory.estimated_tokens();
        assert!(estimate > 0);
        assert!(estimate < 20); // Should be a reasonable small number
    }

    #[test]
    fn test_total_processed_no_summaries() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("Test 1"));
        memory.add_message(Message::user("Test 2"));

        assert_eq!(memory.total_processed(), 2);
    }

    #[test]
    fn test_summaries_empty() {
        let memory = AgentMemory::new();
        assert!(memory.summaries().is_empty());
    }

    #[test]
    fn test_messages_without_summaries() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("User message"));
        memory.add_message(Message {
            role: Role::System,
            content: "Summary".to_string(),
            name: Some("memory_summary".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        let without_summaries = memory.messages_without_summaries();
        assert_eq!(without_summaries.len(), 1);
        assert_eq!(without_summaries[0].content, "User message");
    }

    // ==========================================================================
    // SerializableMessage Tests
    // ==========================================================================

    #[test]
    fn test_serializable_message_from_user() {
        let msg = Message::user("Hello");
        let serializable = SerializableMessage::from(&msg);

        assert_eq!(serializable.role, "user");
        assert_eq!(serializable.content, "Hello");
        assert!(serializable.name.is_none());
    }

    #[test]
    fn test_serializable_message_from_assistant() {
        let msg = Message::assistant("Hi there");
        let serializable = SerializableMessage::from(&msg);

        assert_eq!(serializable.role, "assistant");
        assert_eq!(serializable.content, "Hi there");
    }

    #[test]
    fn test_serializable_message_from_system() {
        let msg = Message {
            role: Role::System,
            content: "System prompt".to_string(),
            name: Some("test".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let serializable = SerializableMessage::from(&msg);

        assert_eq!(serializable.role, "system");
        assert_eq!(serializable.name, Some("test".to_string()));
    }

    #[test]
    fn test_serializable_message_to_message() {
        let serializable = SerializableMessage {
            role: "user".to_string(),
            content: "Test content".to_string(),
            name: None,
            tool_call_id: None,
        };

        let msg = serializable.to_message();
        assert!(matches!(msg.role, Role::User));
        assert_eq!(msg.content, "Test content");
    }

    #[test]
    fn test_serializable_message_to_message_all_roles() {
        for (role_str, expected_role) in [
            ("system", Role::System),
            ("user", Role::User),
            ("assistant", Role::Assistant),
            ("tool", Role::Tool),
            ("unknown", Role::User), // Defaults to User
        ] {
            let serializable = SerializableMessage {
                role: role_str.to_string(),
                content: "Test".to_string(),
                name: None,
                tool_call_id: None,
            };
            let msg = serializable.to_message();
            assert!(std::mem::discriminant(&msg.role) == std::mem::discriminant(&expected_role));
        }
    }

    #[test]
    fn test_serializable_message_roundtrip() {
        let original = Message {
            role: Role::Assistant,
            content: "Response".to_string(),
            name: Some("test_name".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let serializable = SerializableMessage::from(&original);
        let restored = serializable.to_message();

        assert_eq!(restored.content, original.content);
        assert_eq!(restored.name, original.name);
        assert_eq!(restored.tool_call_id, original.tool_call_id);
    }

    #[test]
    fn test_serializable_message_json_serialization() {
        let serializable = SerializableMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&serializable).expect("serialize");
        let parsed: SerializableMessage = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.role, serializable.role);
        assert_eq!(parsed.content, serializable.content);
    }

    // ==========================================================================
    // SerializableConversation Tests
    // ==========================================================================

    #[test]
    fn test_serializable_conversation_new() {
        let conv = SerializableConversation::new("test-id");

        assert_eq!(conv.id, "test-id");
        assert!(conv.title.is_none());
        assert!(conv.messages.is_empty());
        assert!(conv.summaries.is_empty());
        assert!(conv.created_at > 0);
    }

    #[test]
    fn test_serializable_conversation_with_title() {
        let conv = SerializableConversation::new("id")
            .with_title("My Conversation");

        assert_eq!(conv.title, Some("My Conversation".to_string()));
    }

    #[test]
    fn test_serializable_conversation_with_metadata() {
        let conv = SerializableConversation::new("id")
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        assert_eq!(conv.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(conv.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_serializable_conversation_from_memory() {
        let mut memory = AgentMemory::new();
        memory.add_message(Message::user("Hello"));
        memory.add_message(Message::assistant("Hi"));

        let conv = SerializableConversation::from_memory("conv-1", &memory);

        assert_eq!(conv.id, "conv-1");
        assert_eq!(conv.messages.len(), 2);
    }

    #[test]
    fn test_serializable_conversation_restore_to_memory() {
        let conv = SerializableConversation {
            id: "test".to_string(),
            title: None,
            messages: vec![
                SerializableMessage {
                    role: "user".to_string(),
                    content: "Restored".to_string(),
                    name: None,
                    tool_call_id: None,
                },
            ],
            summaries: vec![],
            created_at: 0,
            updated_at: 0,
            metadata: HashMap::new(),
        };

        let mut memory = AgentMemory::new();
        conv.restore_to_memory(&mut memory);

        assert_eq!(memory.len(), 1);
        assert_eq!(memory.messages()[0].content, "Restored");
    }

    // ==========================================================================
    // MemoryConversationStore Tests
    // ==========================================================================

    #[test]
    fn test_memory_store_new() {
        let store = MemoryConversationStore::new();
        let _ = store; // Just verify construction
    }

    #[test]
    fn test_memory_store_default() {
        let store = MemoryConversationStore::default();
        let _ = store; // Just verify construction
    }

    #[tokio::test]
    async fn test_memory_store_save_and_load() {
        let store = MemoryConversationStore::new();
        let conv = SerializableConversation::new("test-1")
            .with_title("Test Conversation");

        store.save(&conv).await.expect("save");

        let loaded = store.load("test-1").await.expect("load");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().title, Some("Test Conversation".to_string()));
    }

    #[tokio::test]
    async fn test_memory_store_load_nonexistent() {
        let store = MemoryConversationStore::new();
        let loaded = store.load("nonexistent").await.expect("load");
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_memory_store_list() {
        let store = MemoryConversationStore::new();

        store.save(&SerializableConversation::new("a")).await.expect("save");
        store.save(&SerializableConversation::new("b")).await.expect("save");
        store.save(&SerializableConversation::new("c")).await.expect("save");

        let list = store.list().await.expect("list");
        assert_eq!(list.len(), 3);
    }

    #[tokio::test]
    async fn test_memory_store_delete() {
        let store = MemoryConversationStore::new();

        store.save(&SerializableConversation::new("to-delete")).await.expect("save");
        assert!(store.exists("to-delete").await.expect("exists"));

        let deleted = store.delete("to-delete").await.expect("delete");
        assert!(deleted);

        assert!(!store.exists("to-delete").await.expect("exists"));
    }

    #[tokio::test]
    async fn test_memory_store_delete_nonexistent() {
        let store = MemoryConversationStore::new();
        let deleted = store.delete("nonexistent").await.expect("delete");
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_memory_store_exists() {
        let store = MemoryConversationStore::new();

        assert!(!store.exists("test").await.expect("exists"));

        store.save(&SerializableConversation::new("test")).await.expect("save");

        assert!(store.exists("test").await.expect("exists"));
    }

    // ==========================================================================
    // FileConversationStore Tests
    // ==========================================================================

    #[tokio::test]
    async fn test_file_store_save_and_load() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileConversationStore::new(temp.path()).expect("create store");

        let conv = SerializableConversation::new("file-test")
            .with_title("File Test");

        store.save(&conv).await.expect("save");

        let loaded = store.load("file-test").await.expect("load");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, "file-test");
    }

    #[tokio::test]
    async fn test_file_store_load_nonexistent() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileConversationStore::new(temp.path()).expect("create store");

        let loaded = store.load("nonexistent").await.expect("load");
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_file_store_list() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileConversationStore::new(temp.path()).expect("create store");

        store.save(&SerializableConversation::new("conv-1")).await.expect("save");
        store.save(&SerializableConversation::new("conv-2")).await.expect("save");

        let list = store.list().await.expect("list");
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_file_store_delete() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileConversationStore::new(temp.path()).expect("create store");

        store.save(&SerializableConversation::new("delete-me")).await.expect("save");

        let deleted = store.delete("delete-me").await.expect("delete");
        assert!(deleted);

        assert!(!store.exists("delete-me").await.expect("exists"));
    }

    #[tokio::test]
    async fn test_file_store_exists() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileConversationStore::new(temp.path()).expect("create store");

        assert!(!store.exists("test").await.expect("exists"));

        store.save(&SerializableConversation::new("test")).await.expect("save");

        assert!(store.exists("test").await.expect("exists"));
    }

    // ==========================================================================
    // PersistentConversation Tests
    // ==========================================================================

    #[test]
    fn test_persistent_conversation_new() {
        let store = Arc::new(MemoryConversationStore::new());
        let conv = PersistentConversation::new("test-id", store);

        assert_eq!(conv.id(), "test-id");
        assert!(conv.is_empty());
    }

    #[test]
    fn test_persistent_conversation_with_memory() {
        let store = Arc::new(MemoryConversationStore::new());
        let memory = AgentMemory::with_max_messages(100);
        let conv = PersistentConversation::with_memory("id", memory, store);

        assert_eq!(conv.id(), "id");
    }

    #[test]
    fn test_persistent_conversation_with_auto_save() {
        let store = Arc::new(MemoryConversationStore::new());
        let conv = PersistentConversation::new("id", store)
            .with_auto_save(false);

        assert_eq!(conv.id(), "id");
    }

    #[test]
    fn test_persistent_conversation_with_title() {
        let store = Arc::new(MemoryConversationStore::new());
        let conv = PersistentConversation::new("id", store)
            .with_title("My Title");

        assert_eq!(conv.title(), Some("My Title"));
    }

    #[test]
    fn test_persistent_conversation_set_metadata() {
        let store = Arc::new(MemoryConversationStore::new());
        let mut conv = PersistentConversation::new("id", store);
        conv.set_metadata("key", "value");

        // Metadata is set (can't easily verify without save/load)
        assert_eq!(conv.id(), "id");
    }

    #[tokio::test]
    async fn test_persistent_conversation_add_message() {
        let store = Arc::new(MemoryConversationStore::new());
        let mut conv = PersistentConversation::new("id", store);

        conv.add_message(Message::user("Hello")).await.expect("add");

        assert_eq!(conv.len(), 1);
        assert_eq!(conv.messages()[0].content, "Hello");
    }

    #[tokio::test]
    async fn test_persistent_conversation_load_or_create_new() {
        let store = Arc::new(MemoryConversationStore::new());
        let conv = PersistentConversation::load_or_create("new-id", store)
            .await
            .expect("load or create");

        assert_eq!(conv.id(), "new-id");
        assert!(conv.is_empty());
    }

    #[tokio::test]
    async fn test_persistent_conversation_load_or_create_existing() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());

        // Save a conversation first
        {
            let mut conv = PersistentConversation::new("existing", Arc::clone(&store));
            conv.add_message(Message::user("Saved message")).await.expect("add");
            conv.save().await.expect("save");
        }

        // Load it
        let conv = PersistentConversation::load_or_create("existing", store)
            .await
            .expect("load");

        assert_eq!(conv.len(), 1);
        assert_eq!(conv.messages()[0].content, "Saved message");
    }

    #[tokio::test]
    async fn test_persistent_conversation_reload() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());
        let mut conv = PersistentConversation::new("reload-test", Arc::clone(&store));

        conv.add_message(Message::user("Original")).await.expect("add");
        conv.save().await.expect("save");

        // Modify and reload
        conv.memory_mut().add_message(Message::user("Not saved"));
        assert_eq!(conv.len(), 2);

        let reloaded = conv.reload().await.expect("reload");
        assert!(reloaded);
        assert_eq!(conv.len(), 1);
    }

    #[tokio::test]
    async fn test_persistent_conversation_clear() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());
        let mut conv = PersistentConversation::new("clear-test", Arc::clone(&store))
            .with_auto_save(false);

        conv.add_message(Message::user("Message")).await.expect("add");
        assert!(!conv.is_empty());

        conv.clear(false).await.expect("clear");
        assert!(conv.is_empty());
    }

    // ==========================================================================
    // ConversationManager Tests
    // ==========================================================================

    #[test]
    fn test_conversation_manager_with_memory_store() {
        let manager = ConversationManager::with_memory_store();
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_conversation_manager_with_file_store() {
        let temp = TempDir::new().expect("temp dir");
        let manager = ConversationManager::with_file_store(temp.path())
            .expect("create manager");
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_conversation_manager_get_or_create() {
        let manager = ConversationManager::with_memory_store();

        let conv = manager.get_or_create("test-conv").await.expect("get or create");
        assert_eq!(manager.active_count(), 1);

        let conv_read = conv.read();
        assert_eq!(conv_read.id(), "test-conv");
    }

    #[tokio::test]
    async fn test_conversation_manager_get_same_twice() {
        let manager = ConversationManager::with_memory_store();

        let conv1 = manager.get_or_create("same-id").await.expect("get 1");
        let conv2 = manager.get_or_create("same-id").await.expect("get 2");

        // Should be the same Arc
        assert_eq!(manager.active_count(), 1);

        // Verify they point to same data
        {
            let mut w1 = conv1.write();
            w1.memory_mut().add_message(Message::user("Test"));
        }
        {
            let r2 = conv2.read();
            assert_eq!(r2.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_conversation_manager_list_all() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());
        let manager = ConversationManager::new(store);

        manager.get_or_create("conv-a").await.expect("create a");
        manager.get_or_create("conv-b").await.expect("create b");

        // Save them
        manager.save_all().await.expect("save all");

        let list = manager.list_all().await.expect("list");
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_conversation_manager_delete() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());
        let manager = ConversationManager::new(Arc::clone(&store));

        manager.get_or_create("to-delete").await.expect("create");
        manager.save_all().await.expect("save");

        let deleted = manager.delete("to-delete").await.expect("delete");
        assert!(deleted);
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_conversation_manager_unload() {
        let manager = ConversationManager::with_memory_store();

        manager.get_or_create("unload-me").await.expect("create");
        assert_eq!(manager.active_count(), 1);

        manager.unload_conversation("unload-me");
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_conversation_manager_save_all() {
        let store: Arc<dyn ConversationStore> = Arc::new(MemoryConversationStore::new());
        let manager = ConversationManager::new(Arc::clone(&store));

        {
            let conv = manager.get_or_create("save-test").await.expect("create");
            let mut w = conv.write();
            w.add_message(Message::user("To be saved")).await.expect("add");
        }

        manager.save_all().await.expect("save all");

        // Verify saved
        let loaded = store.load("save-test").await.expect("load");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().messages.len(), 1);
    }

    // ==========================================================================
    // role_to_str Tests
    // ==========================================================================

    #[test]
    fn test_role_to_str_all_roles() {
        assert_eq!(role_to_str(&Role::System), "System");
        assert_eq!(role_to_str(&Role::User), "User");
        assert_eq!(role_to_str(&Role::Assistant), "Assistant");
        assert_eq!(role_to_str(&Role::Tool), "Tool");
    }
}
