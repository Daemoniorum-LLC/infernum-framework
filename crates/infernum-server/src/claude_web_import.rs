//! Claude.ai web export import functionality.
//!
//! Parses and stores data from Claude.ai data exports (Settings > Privacy > Export).
//!
//! ## Export Format
//!
//! The export is a ZIP file containing:
//! - `users.json` - Account information
//! - `memories.json` - Claude's memory about the user
//! - `conversations.json` - All conversations with messages
//! - `projects.json` - Projects with attached documents
//!
//! ## Endpoints
//!
//! - `POST /api/claude/import` - Import a web export ZIP file
//! - `GET /api/claude/web-conversations` - List imported web conversations
//! - `GET /api/claude/web-conversations/:id` - Get a specific conversation

use axum::{
    extract::{Multipart, Path, Query},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, warn};
use zip::ZipArchive;

// =============================================================================
// API Response Types
// =============================================================================

/// Import result response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// Whether the import was successful
    pub success: bool,
    /// Number of conversations imported
    pub conversations_imported: usize,
    /// Number of messages imported
    pub messages_imported: usize,
    /// Number of projects imported
    pub projects_imported: usize,
    /// User info if found
    pub user_name: Option<String>,
    /// Any warnings during import
    pub warnings: Vec<String>,
}

/// Imported conversation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConversation {
    /// Conversation UUID
    pub uuid: String,
    /// Conversation name/title
    pub name: Option<String>,
    /// Summary if available
    pub summary: Option<String>,
    /// When created
    pub created_at: String,
    /// When last updated
    pub updated_at: String,
    /// Number of messages
    pub message_count: usize,
    /// Source: "web" for claude.ai imports
    pub source: String,
}

/// Full conversation with messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConversationFull {
    /// Conversation metadata
    #[serde(flatten)]
    pub info: WebConversation,
    /// All messages
    pub messages: Vec<WebMessage>,
}

/// A single message from a web conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebMessage {
    /// Message UUID
    pub uuid: String,
    /// Sender: "human" or "assistant"
    pub sender: String,
    /// Message text content
    pub text: String,
    /// When created
    pub created_at: String,
    /// Whether message has attachments
    pub has_attachments: bool,
}

/// Query params for listing conversations.
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    /// Maximum results
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Search in name/content
    pub search: Option<String>,
}

fn default_limit() -> usize {
    100
}

// =============================================================================
// Raw Import Types (for parsing export JSON)
// =============================================================================

#[derive(Debug, Deserialize)]
struct RawUser {
    uuid: String,
    full_name: Option<String>,
    email_address: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawMemory {
    conversations_memory: Option<String>,
    account_uuid: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawConversation {
    uuid: String,
    name: Option<String>,
    summary: Option<String>,
    created_at: String,
    updated_at: String,
    chat_messages: Vec<RawChatMessage>,
}

#[derive(Debug, Deserialize)]
struct RawChatMessage {
    uuid: String,
    text: Option<String>,
    sender: String,
    created_at: String,
    content: Option<Vec<RawContentBlock>>,
    attachments: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct RawContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawProject {
    uuid: String,
    name: String,
    description: Option<String>,
    docs: Option<Vec<RawDoc>>,
}

#[derive(Debug, Deserialize)]
struct RawDoc {
    uuid: String,
    filename: String,
    content: Option<String>,
}

// =============================================================================
// Storage
// =============================================================================

/// Get the storage path for imported web data.
fn get_storage_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude")
        .join("web-imports")
}

/// Stored import data - supports multiple accounts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct StoredImport {
    /// All imported accounts
    accounts: Vec<StoredAccount>,
}

/// A single account's imported data.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredAccount {
    /// User info
    user: Option<StoredUser>,
    /// Memory content
    memory: Option<String>,
    /// All conversations
    conversations: Vec<StoredConversation>,
    /// All projects
    projects: Vec<StoredProject>,
    /// Import timestamp
    imported_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredUser {
    uuid: String,
    name: Option<String>,
    email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredConversation {
    uuid: String,
    name: Option<String>,
    summary: Option<String>,
    created_at: String,
    updated_at: String,
    messages: Vec<StoredMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMessage {
    uuid: String,
    sender: String,
    text: String,
    created_at: String,
    has_attachments: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredProject {
    uuid: String,
    name: String,
    description: Option<String>,
    docs: Vec<StoredDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredDoc {
    uuid: String,
    filename: String,
    content: String,
}

// =============================================================================
// Router
// =============================================================================

/// Create the Claude web import router.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/import", post(import_export))
        .route("/web-conversations", get(list_web_conversations))
        .route("/web-conversations/{id}", get(get_web_conversation))
        .route("/web-stats", get(get_web_stats))
}

// =============================================================================
// Handlers
// =============================================================================

/// POST /api/claude/import
/// Import a Claude.ai web export ZIP file.
pub async fn import_export(mut multipart: Multipart) -> Json<ImportResult> {
    let mut warnings = Vec::new();

    // Load existing imports to append to
    let mut stored = load_stored_import().await;

    // Read the uploaded file
    let mut zip_data: Vec<u8> = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().map(|s| s.to_string());
        if name.as_deref() == Some("file") {
            match field.bytes().await {
                Ok(bytes) => zip_data = bytes.to_vec(),
                Err(e) => {
                    return Json(ImportResult {
                        success: false,
                        conversations_imported: 0,
                        messages_imported: 0,
                        projects_imported: 0,
                        user_name: None,
                        warnings: vec![format!("Failed to read upload: {}", e)],
                    });
                }
            }
        }
    }

    if zip_data.is_empty() {
        return Json(ImportResult {
            success: false,
            conversations_imported: 0,
            messages_imported: 0,
            projects_imported: 0,
            user_name: None,
            warnings: vec!["No file uploaded".to_string()],
        });
    }

    // Parse the ZIP
    let cursor = std::io::Cursor::new(zip_data);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            return Json(ImportResult {
                success: false,
                conversations_imported: 0,
                messages_imported: 0,
                projects_imported: 0,
                user_name: None,
                warnings: vec![format!("Invalid ZIP file: {}", e)],
            });
        }
    };

    // Create a new account for this import
    let mut account = StoredAccount {
        user: None,
        memory: None,
        conversations: Vec::new(),
        projects: Vec::new(),
        imported_at: chrono::Utc::now().to_rfc3339(),
    };

    // Parse users.json
    if let Ok(mut file) = archive.by_name("users.json") {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            if let Ok(users) = serde_json::from_str::<Vec<RawUser>>(&content) {
                if let Some(user) = users.into_iter().next() {
                    account.user = Some(StoredUser {
                        uuid: user.uuid,
                        name: user.full_name.clone(),
                        email: user.email_address,
                    });
                }
            }
        }
    }

    // Parse memories.json
    if let Ok(mut file) = archive.by_name("memories.json") {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            if let Ok(memories) = serde_json::from_str::<Vec<RawMemory>>(&content) {
                if let Some(memory) = memories.into_iter().next() {
                    account.memory = memory.conversations_memory;
                }
            }
        }
    }

    // Parse conversations.json
    let mut total_messages = 0;
    if let Ok(mut file) = archive.by_name("conversations.json") {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            if let Ok(convs) = serde_json::from_str::<Vec<RawConversation>>(&content) {
                for conv in convs {
                    let messages: Vec<StoredMessage> = conv
                        .chat_messages
                        .into_iter()
                        .map(|msg| {
                            // Extract text from content blocks if text field is empty
                            let text = msg.text.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| {
                                msg.content
                                    .as_ref()
                                    .map(|blocks| {
                                        blocks
                                            .iter()
                                            .filter_map(|b| b.text.clone())
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    })
                                    .unwrap_or_default()
                            });

                            let has_attachments = msg
                                .attachments
                                .as_ref()
                                .map(|a| !a.is_empty())
                                .unwrap_or(false);

                            StoredMessage {
                                uuid: msg.uuid,
                                sender: msg.sender,
                                text,
                                created_at: msg.created_at,
                                has_attachments,
                            }
                        })
                        .collect();

                    total_messages += messages.len();

                    account.conversations.push(StoredConversation {
                        uuid: conv.uuid,
                        name: conv.name,
                        summary: conv.summary,
                        created_at: conv.created_at,
                        updated_at: conv.updated_at,
                        messages,
                    });
                }
            }
        }
    }

    // Parse projects.json
    if let Ok(mut file) = archive.by_name("projects.json") {
        let mut content = String::new();
        if file.read_to_string(&mut content).is_ok() {
            if let Ok(projects) = serde_json::from_str::<Vec<RawProject>>(&content) {
                for proj in projects {
                    let docs: Vec<StoredDoc> = proj
                        .docs
                        .unwrap_or_default()
                        .into_iter()
                        .map(|d| StoredDoc {
                            uuid: d.uuid,
                            filename: d.filename,
                            content: d.content.unwrap_or_default(),
                        })
                        .collect();

                    account.projects.push(StoredProject {
                        uuid: proj.uuid,
                        name: proj.name,
                        description: proj.description,
                        docs,
                    });
                }
            }
        }
    }

    // Check if this account already exists (by user UUID) and update or add
    let user_name = account.user.as_ref().and_then(|u| u.name.clone());
    let conv_count = account.conversations.len();
    let proj_count = account.projects.len();

    if let Some(user) = &account.user {
        // Remove existing account with same UUID to replace
        stored.accounts.retain(|a| {
            a.user.as_ref().map(|u| &u.uuid) != Some(&user.uuid)
        });
    }
    stored.accounts.push(account);

    // Save to disk
    let storage_path = get_storage_path();
    if let Err(e) = fs::create_dir_all(&storage_path).await {
        warnings.push(format!("Failed to create storage dir: {}", e));
    }

    let data_file = storage_path.join("import.json");
    match serde_json::to_string_pretty(&stored) {
        Ok(json) => {
            if let Err(e) = fs::write(&data_file, json).await {
                warnings.push(format!("Failed to save import: {}", e));
            } else {
                info!(
                    "Imported {} conversations, {} messages, {} projects (total accounts: {})",
                    conv_count,
                    total_messages,
                    proj_count,
                    stored.accounts.len()
                );
            }
        }
        Err(e) => warnings.push(format!("Failed to serialize: {}", e)),
    }

    Json(ImportResult {
        success: warnings.is_empty(),
        conversations_imported: conv_count,
        messages_imported: total_messages,
        projects_imported: proj_count,
        user_name,
        warnings,
    })
}

/// GET /api/claude/web-conversations
/// List imported web conversations from all accounts.
pub async fn list_web_conversations(Query(params): Query<ListQuery>) -> Json<Vec<WebConversation>> {
    let stored = load_stored_import().await;

    // Aggregate conversations from all accounts
    let mut conversations: Vec<WebConversation> = stored
        .accounts
        .into_iter()
        .flat_map(|account| {
            let account_name = account.user.as_ref().and_then(|u| u.name.clone());
            account.conversations.into_iter().map(move |c| {
                (c, account_name.clone())
            })
        })
        .filter(|(c, _)| {
            if let Some(ref search) = params.search {
                let search_lower = search.to_lowercase();
                c.name
                    .as_ref()
                    .map(|n| n.to_lowercase().contains(&search_lower))
                    .unwrap_or(false)
                    || c.summary
                        .as_ref()
                        .map(|s| s.to_lowercase().contains(&search_lower))
                        .unwrap_or(false)
                    || c.messages
                        .iter()
                        .any(|m| m.text.to_lowercase().contains(&search_lower))
            } else {
                true
            }
        })
        .map(|(c, account_name)| WebConversation {
            uuid: c.uuid,
            name: c.name,
            summary: c.summary,
            created_at: c.created_at,
            updated_at: c.updated_at,
            message_count: c.messages.len(),
            source: account_name.unwrap_or_else(|| "web".to_string()),
        })
        .collect();

    // Sort by updated_at descending
    conversations.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    // Apply limit
    conversations.truncate(params.limit);

    Json(conversations)
}

/// GET /api/claude/web-conversations/:id
/// Get a specific web conversation with all messages.
pub async fn get_web_conversation(Path(id): Path<String>) -> Json<Option<WebConversationFull>> {
    let stored = load_stored_import().await;

    // Search across all accounts
    for account in stored.accounts {
        let account_name = account.user.as_ref().and_then(|u| u.name.clone());
        if let Some(conv) = account.conversations.into_iter().find(|c| c.uuid == id) {
            return Json(Some(WebConversationFull {
                info: WebConversation {
                    uuid: conv.uuid.clone(),
                    name: conv.name,
                    summary: conv.summary,
                    created_at: conv.created_at,
                    updated_at: conv.updated_at,
                    message_count: conv.messages.len(),
                    source: account_name.unwrap_or_else(|| "web".to_string()),
                },
                messages: conv
                    .messages
                    .into_iter()
                    .map(|m| WebMessage {
                        uuid: m.uuid,
                        sender: m.sender,
                        text: m.text,
                        created_at: m.created_at,
                        has_attachments: m.has_attachments,
                    })
                    .collect(),
            }));
        }
    }

    Json(None)
}

/// GET /api/claude/web-stats
/// Get stats about imported web data across all accounts.
pub async fn get_web_stats() -> Json<WebImportStats> {
    let stored = load_stored_import().await;

    let account_count = stored.accounts.len();
    let account_names: Vec<String> = stored.accounts.iter()
        .filter_map(|a| a.user.as_ref().and_then(|u| u.name.clone()))
        .collect();

    let total_conversations: usize = stored.accounts.iter()
        .map(|a| a.conversations.len())
        .sum();
    let total_messages: usize = stored.accounts.iter()
        .flat_map(|a| a.conversations.iter())
        .map(|c| c.messages.len())
        .sum();
    let total_projects: usize = stored.accounts.iter()
        .map(|a| a.projects.len())
        .sum();
    let total_docs: usize = stored.accounts.iter()
        .flat_map(|a| a.projects.iter())
        .map(|p| p.docs.len())
        .sum();
    let has_memory = stored.accounts.iter().any(|a| a.memory.is_some());
    let latest_import = stored.accounts.iter()
        .map(|a| &a.imported_at)
        .max()
        .cloned();

    Json(WebImportStats {
        has_import: !stored.accounts.is_empty(),
        account_count,
        account_names,
        conversation_count: total_conversations,
        message_count: total_messages,
        project_count: total_projects,
        document_count: total_docs,
        has_memory,
        imported_at: latest_import,
    })
}

/// Stats about imported web data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebImportStats {
    /// Whether any data has been imported
    pub has_import: bool,
    /// Number of accounts imported
    pub account_count: usize,
    /// Names of imported accounts
    pub account_names: Vec<String>,
    /// Number of conversations across all accounts
    pub conversation_count: usize,
    /// Total messages across all conversations
    pub message_count: usize,
    /// Number of projects across all accounts
    pub project_count: usize,
    /// Total documents across all projects
    pub document_count: usize,
    /// Whether any account has memory data
    pub has_memory: bool,
    /// When the latest import was done
    pub imported_at: Option<String>,
}

/// Load stored import data from disk.
async fn load_stored_import() -> StoredImport {
    let data_file = get_storage_path().join("import.json");

    if !data_file.exists() {
        return StoredImport::default();
    }

    match fs::read_to_string(&data_file).await {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(e) => {
            warn!("Failed to load stored import: {}", e);
            StoredImport::default()
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_path() {
        let path = get_storage_path();
        assert!(path.ends_with("web-imports"));
    }

    #[test]
    fn test_empty_stored_import() {
        let stored = StoredImport::default();
        assert!(stored.conversations.is_empty());
        assert!(stored.user.is_none());
    }
}
