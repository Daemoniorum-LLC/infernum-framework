//! Room persistence for durable storage.
//!
//! This module handles saving and loading room state to disk:
//!
//! - **Room snapshots**: Full room state including participants
//! - **Message history**: All messages across channels
//! - **Coordinator state**: Turn queue and speaker info
//!
//! # Storage Layout
//!
//! ```text
//! ~/.local/share/infernum/conclave/
//! ├── rooms/
//! │   ├── <room-uuid>.json          # Room snapshot
//! │   ├── <room-uuid>_messages.jsonl  # Message log (append-only)
//! │   └── <room-uuid>_coordinator.json # Coordinator state
//! └── registry.json                 # Room registry index
//! ```
//!
//! # Recovery Process
//!
//! On startup:
//! 1. Load registry index
//! 2. For each non-archived room, load snapshot and messages
//! 3. Agent sessions are NOT restored (ephemeral)
//! 4. Agents can be re-spawned by the user

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info, warn};

use crate::coordinator::CoordinatorState;
use crate::error::Result;
use crate::types::{
    CoordinatorConfig, InvitePolicy, Message, Participant, ProjectRef, Room, RoomId,
};

// =============================================================================
// Persistence Configuration
// =============================================================================

/// Configuration for room persistence.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Base directory for persistence data.
    pub data_dir: PathBuf,

    /// Whether to auto-save on changes.
    pub auto_save: bool,

    /// How often to flush writes (if buffering).
    pub flush_interval_secs: u64,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
            .join("conclave");

        Self {
            data_dir,
            auto_save: true,
            flush_interval_secs: 30,
        }
    }
}

impl PersistenceConfig {
    /// Creates config with a custom data directory.
    pub fn with_data_dir(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            ..Default::default()
        }
    }

    /// Returns the rooms directory.
    pub fn rooms_dir(&self) -> PathBuf {
        self.data_dir.join("rooms")
    }

    /// Returns the path for a room snapshot.
    pub fn room_path(&self, room_id: RoomId) -> PathBuf {
        self.rooms_dir().join(format!("{}.json", room_id.0))
    }

    /// Returns the path for a room's message log.
    pub fn messages_path(&self, room_id: RoomId) -> PathBuf {
        self.rooms_dir().join(format!("{}_messages.jsonl", room_id.0))
    }

    /// Returns the path for a room's coordinator state.
    pub fn coordinator_path(&self, room_id: RoomId) -> PathBuf {
        self.rooms_dir().join(format!("{}_coordinator.json", room_id.0))
    }

    /// Returns the path for the registry index.
    pub fn registry_path(&self) -> PathBuf {
        self.data_dir.join("registry.json")
    }
}

// =============================================================================
// Persistence Data Types
// =============================================================================

/// Snapshot of a room for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSnapshot {
    /// Room metadata.
    pub id: RoomId,
    pub name: String,
    pub working_dir: PathBuf,
    pub project: Option<ProjectRef>,

    /// Policies.
    pub invite_policy: InvitePolicy,
    pub coordinator_config: CoordinatorConfig,

    /// Participants (agents won't have active sessions).
    pub participants: Vec<Participant>,
    pub alumni: Vec<Participant>,

    /// Archive status.
    pub archived: bool,
    pub fork_of: Option<RoomId>,

    /// Timestamps.
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    /// Version for future migration.
    pub version: u32,
}

impl RoomSnapshot {
    /// Current snapshot version.
    pub const VERSION: u32 = 1;

    /// Creates a snapshot from a room.
    pub fn from_room(room: &Room) -> Self {
        Self {
            id: room.id,
            name: room.name.clone(),
            working_dir: room.working_dir.clone(),
            project: room.project.clone(),
            invite_policy: room.invite_policy,
            coordinator_config: room.coordinator_config.clone(),
            participants: room.participants.clone(),
            alumni: room.alumni.clone(),
            archived: room.archived,
            fork_of: room.fork_of,
            created_at: room.created_at,
            updated_at: room.updated_at,
            version: Self::VERSION,
        }
    }

    /// Converts snapshot back to a room.
    ///
    /// Note: Agent sessions will need to be re-established separately.
    pub fn into_room(self) -> Room {
        Room {
            id: self.id,
            name: self.name,
            working_dir: self.working_dir,
            project: self.project,
            invite_policy: self.invite_policy,
            coordinator_config: self.coordinator_config,
            participants: self.participants,
            alumni: self.alumni,
            archived: self.archived,
            fork_of: self.fork_of,
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

/// Registry index for quick room lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryIndex {
    /// List of room IDs.
    pub rooms: Vec<RoomId>,
    /// Last modified timestamp.
    pub updated_at: DateTime<Utc>,
    /// Version for future migration.
    pub version: u32,
}

impl RegistryIndex {
    /// Current index version.
    pub const VERSION: u32 = 1;

    /// Creates a new empty index.
    pub fn new() -> Self {
        Self {
            rooms: Vec::new(),
            updated_at: Utc::now(),
            version: Self::VERSION,
        }
    }
}

// =============================================================================
// Persistence Store
// =============================================================================

/// Handles reading and writing room data to disk.
pub struct PersistenceStore {
    config: PersistenceConfig,
}

impl PersistenceStore {
    /// Creates a new persistence store.
    pub fn new(config: PersistenceConfig) -> Self {
        Self { config }
    }

    /// Creates a persistence store with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PersistenceConfig::default())
    }

    /// Ensures the data directories exist.
    pub async fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&self.config.rooms_dir()).await?;
        info!("Initialized persistence at {:?}", self.config.data_dir);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Room Operations
    // -------------------------------------------------------------------------

    /// Saves a room snapshot to disk.
    pub async fn save_room(&self, room: &Room) -> Result<()> {
        let snapshot = RoomSnapshot::from_room(room);
        let path = self.config.room_path(room.id);

        let json = serde_json::to_string_pretty(&snapshot)?;
        fs::write(&path, json).await?;

        debug!("Saved room {} to {:?}", room.id, path);
        Ok(())
    }

    /// Loads a room from disk.
    pub async fn load_room(&self, room_id: RoomId) -> Result<Room> {
        let path = self.config.room_path(room_id);
        let json = fs::read_to_string(&path).await?;
        let snapshot: RoomSnapshot = serde_json::from_str(&json)?;

        debug!("Loaded room {} from {:?}", room_id, path);
        Ok(snapshot.into_room())
    }

    /// Checks if a room exists on disk.
    pub async fn room_exists(&self, room_id: RoomId) -> bool {
        let path = self.config.room_path(room_id);
        fs::metadata(&path).await.is_ok()
    }

    /// Deletes a room and its associated data.
    pub async fn delete_room(&self, room_id: RoomId) -> Result<()> {
        // Delete room snapshot
        let room_path = self.config.room_path(room_id);
        if fs::metadata(&room_path).await.is_ok() {
            fs::remove_file(&room_path).await?;
        }

        // Delete messages
        let messages_path = self.config.messages_path(room_id);
        if fs::metadata(&messages_path).await.is_ok() {
            fs::remove_file(&messages_path).await?;
        }

        // Delete coordinator state
        let coordinator_path = self.config.coordinator_path(room_id);
        if fs::metadata(&coordinator_path).await.is_ok() {
            fs::remove_file(&coordinator_path).await?;
        }

        info!("Deleted room {} from disk", room_id);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Message Operations
    // -------------------------------------------------------------------------

    /// Appends a message to the room's message log.
    pub async fn append_message(&self, room_id: RoomId, message: &Message) -> Result<()> {
        let path = self.config.messages_path(room_id);

        let mut json = serde_json::to_string(message)?;
        json.push('\n');

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        file.write_all(json.as_bytes()).await?;
        Ok(())
    }

    /// Saves all messages for a room, replacing any existing message log.
    ///
    /// This is used for full persistence to avoid message duplication that
    /// would occur from repeated append operations.
    pub async fn save_all_messages(&self, room_id: RoomId, messages: &[Message]) -> Result<()> {
        let path = self.config.messages_path(room_id);

        // Build full content first
        let mut content = String::new();
        for message in messages {
            let json = serde_json::to_string(message)?;
            content.push_str(&json);
            content.push('\n');
        }

        // Truncate and write (not append)
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .await?;

        file.write_all(content.as_bytes()).await?;
        Ok(())
    }

    /// Loads all messages for a room.
    pub async fn load_messages(&self, room_id: RoomId) -> Result<Vec<Message>> {
        let path = self.config.messages_path(room_id);

        if fs::metadata(&path).await.is_err() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut messages = Vec::new();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Message>(&line) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    warn!("Failed to parse message line: {}", e);
                }
            }
        }

        debug!("Loaded {} messages for room {}", messages.len(), room_id);
        Ok(messages)
    }

    // -------------------------------------------------------------------------
    // Coordinator Operations
    // -------------------------------------------------------------------------

    /// Saves coordinator state for a room.
    pub async fn save_coordinator(&self, room_id: RoomId, state: &CoordinatorState) -> Result<()> {
        let path = self.config.coordinator_path(room_id);
        let json = serde_json::to_string_pretty(state)?;
        fs::write(&path, json).await?;
        Ok(())
    }

    /// Loads coordinator state for a room.
    pub async fn load_coordinator(&self, room_id: RoomId) -> Result<Option<CoordinatorState>> {
        let path = self.config.coordinator_path(room_id);

        if fs::metadata(&path).await.is_err() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path).await?;
        let state: CoordinatorState = serde_json::from_str(&json)?;
        Ok(Some(state))
    }

    // -------------------------------------------------------------------------
    // Registry Operations
    // -------------------------------------------------------------------------

    /// Saves the registry index.
    pub async fn save_registry(&self, rooms: &[RoomId]) -> Result<()> {
        let index = RegistryIndex {
            rooms: rooms.to_vec(),
            updated_at: Utc::now(),
            version: RegistryIndex::VERSION,
        };

        let path = self.config.registry_path();
        let json = serde_json::to_string_pretty(&index)?;
        fs::write(&path, json).await?;

        debug!("Saved registry with {} rooms", rooms.len());
        Ok(())
    }

    /// Loads the registry index.
    pub async fn load_registry(&self) -> Result<RegistryIndex> {
        let path = self.config.registry_path();

        if fs::metadata(&path).await.is_err() {
            return Ok(RegistryIndex::new());
        }

        let json = fs::read_to_string(&path).await?;
        let index: RegistryIndex = serde_json::from_str(&json)?;

        info!("Loaded registry with {} rooms", index.rooms.len());
        Ok(index)
    }

    // -------------------------------------------------------------------------
    // Bulk Operations
    // -------------------------------------------------------------------------

    /// Loads all rooms from disk.
    pub async fn load_all_rooms(&self) -> Result<HashMap<RoomId, Room>> {
        let index = self.load_registry().await?;
        let mut rooms = HashMap::new();

        for room_id in index.rooms {
            match self.load_room(room_id).await {
                Ok(room) => {
                    rooms.insert(room_id, room);
                }
                Err(e) => {
                    error!("Failed to load room {}: {}", room_id, e);
                }
            }
        }

        Ok(rooms)
    }

    /// Saves all rooms to disk.
    pub async fn save_all_rooms(&self, rooms: &HashMap<RoomId, Room>) -> Result<()> {
        for room in rooms.values() {
            self.save_room(room).await?;
        }

        let room_ids: Vec<_> = rooms.keys().copied().collect();
        self.save_registry(&room_ids).await?;

        info!("Saved {} rooms to disk", rooms.len());
        Ok(())
    }
}

// =============================================================================
// RoomRegistry Integration
// =============================================================================

impl crate::room::RoomRegistry {
    /// Creates a room registry with persistence.
    pub async fn with_persistence(store: PersistenceStore) -> Result<Self> {
        store.initialize().await?;

        let rooms = store.load_all_rooms().await?;
        let mut all_messages: HashMap<RoomId, Vec<Message>> = HashMap::new();

        // Load messages for each room
        for room_id in rooms.keys() {
            let messages = store.load_messages(*room_id).await?;
            if !messages.is_empty() {
                all_messages.insert(*room_id, messages);
            }
        }

        // Create registry with loaded data
        let registry = Self::with_defaults();

        // Populate rooms
        {
            let mut registry_rooms = registry.rooms.write().await;
            *registry_rooms = rooms;
        }

        // Populate messages
        {
            let mut registry_messages = registry.messages.write().await;
            *registry_messages = all_messages;
        }

        info!("Restored room registry from persistence");
        Ok(registry)
    }

    /// Persists the current registry state.
    pub async fn persist(&self, store: &PersistenceStore) -> Result<()> {
        let rooms = self.rooms.read().await;
        store.save_all_rooms(&rooms).await?;

        // Save messages for each room (full replacement to avoid duplicates)
        for (room_id, messages) in self.messages.read().await.iter() {
            store.save_all_messages(*room_id, messages).await?;
        }

        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CreateRoomRequest, UserId};
    use tempfile::TempDir;

    fn test_user() -> UserId {
        UserId("test_user".to_string())
    }

    fn test_working_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    async fn setup_store() -> (PersistenceStore, TempDir) {
        let temp = TempDir::new().unwrap();
        let config = PersistenceConfig::with_data_dir(temp.path());
        let store = PersistenceStore::new(config);
        store.initialize().await.unwrap();
        (store, temp)
    }

    #[tokio::test]
    async fn test_persistence_config_defaults() {
        let config = PersistenceConfig::default();
        assert!(config.auto_save);
        assert!(config.data_dir.to_string_lossy().contains("conclave"));
    }

    #[tokio::test]
    async fn test_save_and_load_room() {
        let (store, _temp) = setup_store().await;

        // Create a room
        let registry = crate::room::RoomRegistry::with_defaults();
        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();

        // Save
        store.save_room(&room).await.unwrap();

        // Verify exists
        assert!(store.room_exists(room_id).await);

        // Load
        let loaded = store.load_room(room_id).await.unwrap();
        assert_eq!(loaded.id, room.id);
        assert_eq!(loaded.name, room.name);
        assert_eq!(loaded.participants.len(), room.participants.len());
    }

    #[tokio::test]
    async fn test_save_and_load_messages() {
        let (store, _temp) = setup_store().await;

        use crate::types::{ChannelType, MessageContent, MessageId, ParticipantId};
        use std::collections::HashMap as StdHashMap;

        let room_id = RoomId::new();
        let sender = ParticipantId::new();

        // Create messages
        let msg1 = Message {
            id: MessageId::new(),
            channel: ChannelType::Main,
            sender,
            content: MessageContent::Text {
                content: "Hello".to_string(),
            },
            timestamp: Utc::now(),
            metadata: StdHashMap::new(),
        };

        let msg2 = Message {
            id: MessageId::new(),
            channel: ChannelType::Main,
            sender,
            content: MessageContent::Text {
                content: "World".to_string(),
            },
            timestamp: Utc::now(),
            metadata: StdHashMap::new(),
        };

        // Append
        store.append_message(room_id, &msg1).await.unwrap();
        store.append_message(room_id, &msg2).await.unwrap();

        // Load
        let loaded = store.load_messages(room_id).await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, msg1.id);
        assert_eq!(loaded[1].id, msg2.id);
    }

    #[tokio::test]
    async fn test_save_and_load_registry() {
        let (store, _temp) = setup_store().await;

        let rooms = vec![RoomId::new(), RoomId::new(), RoomId::new()];
        store.save_registry(&rooms).await.unwrap();

        let loaded = store.load_registry().await.unwrap();
        assert_eq!(loaded.rooms.len(), 3);
        assert_eq!(loaded.rooms, rooms);
    }

    #[tokio::test]
    async fn test_delete_room() {
        let (store, _temp) = setup_store().await;

        // Create a room with messages
        let registry = crate::room::RoomRegistry::with_defaults();
        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();

        store.save_room(&room).await.unwrap();

        use crate::types::{ChannelType, MessageContent, MessageId, ParticipantId};
        use std::collections::HashMap as StdHashMap;

        let msg = Message {
            id: MessageId::new(),
            channel: ChannelType::Main,
            sender: ParticipantId::new(),
            content: MessageContent::Text {
                content: "Test".to_string(),
            },
            timestamp: Utc::now(),
            metadata: StdHashMap::new(),
        };
        store.append_message(room_id, &msg).await.unwrap();

        // Verify exists
        assert!(store.room_exists(room_id).await);

        // Delete
        store.delete_room(room_id).await.unwrap();

        // Verify gone
        assert!(!store.room_exists(room_id).await);
        let messages = store.load_messages(room_id).await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_room_snapshot_version() {
        let registry = crate::room::RoomRegistry::with_defaults();
        let request = CreateRoomRequest::new(
            "Test Room",
            test_working_dir(),
            test_user(),
        );
        let room_id = registry.create_room(request).await.unwrap();
        let room = registry.get_room(room_id).await.unwrap();

        let snapshot = RoomSnapshot::from_room(&room);
        assert_eq!(snapshot.version, RoomSnapshot::VERSION);
    }

    #[tokio::test]
    async fn test_load_empty_registry() {
        let (store, _temp) = setup_store().await;

        let index = store.load_registry().await.unwrap();
        assert!(index.rooms.is_empty());
    }

    #[tokio::test]
    async fn test_load_all_rooms() {
        let (store, _temp) = setup_store().await;

        // Create and save rooms
        let registry = crate::room::RoomRegistry::with_defaults();

        let request1 = CreateRoomRequest::new("Room 1", test_working_dir(), test_user());
        let request2 = CreateRoomRequest::new("Room 2", test_working_dir(), test_user());

        let id1 = registry.create_room(request1).await.unwrap();
        let id2 = registry.create_room(request2).await.unwrap();

        let room1 = registry.get_room(id1).await.unwrap();
        let room2 = registry.get_room(id2).await.unwrap();

        store.save_room(&room1).await.unwrap();
        store.save_room(&room2).await.unwrap();
        store.save_registry(&[id1, id2]).await.unwrap();

        // Load all
        let loaded = store.load_all_rooms().await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key(&id1));
        assert!(loaded.contains_key(&id2));
    }
}
