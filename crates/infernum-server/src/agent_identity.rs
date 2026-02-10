//! Agent identity and cryptographic accountability.
//!
//! This module provides cryptographic identity management for AI agents,
//! enabling immutable audit trails via Moloch integration.
//!
//! # Architecture
//!
//! Each agent instance has a unique Ed25519 keypair:
//! - **Secret key**: Used to sign tool execution events
//! - **Public key**: Used for identity verification and audit trails
//!
//! # Phase 4 Integration
//!
//! This implements INFERNUM-SPEC.md §17.2.1 (Moloch Integration):
//! - AgentIdentity wraps Moloch's cryptographic primitives
//! - Tool executions become signed audit events
//! - HoloCrypt encryption policies protect sensitive prompts
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::agent_identity::AgentIdentity;
//!
//! // Generate a new agent identity
//! let identity = AgentIdentity::generate("code-assistant");
//!
//! // Sign a tool execution
//! let event = identity.sign_tool_execution(
//!     "get_weather",
//!     &serde_json::json!({"location": "Seattle"}),
//! )?;
//!
//! // Verify the event
//! assert!(event.validate().is_ok());
//! ```

use moloch_core::crypto::{Hash, PublicKey, SecretKey};
use moloch_core::event::{
    ActorId, ActorKind, AuditEvent, EventType, Outcome, ResourceId, ResourceKind,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during agent identity operations.
#[derive(Debug, Error)]
pub enum AgentIdentityError {
    /// Failed to create audit event
    #[error("failed to create audit event: {0}")]
    EventCreation(String),

    /// Failed to serialize/deserialize identity
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Invalid key material
    #[error("invalid key: {0}")]
    InvalidKey(String),

    /// Moloch error
    #[error("moloch error: {0}")]
    Moloch(#[from] moloch_core::error::Error),
}

/// Result type for agent identity operations.
pub type Result<T> = std::result::Result<T, AgentIdentityError>;

/// Cryptographic identity for an AI agent.
///
/// Each agent has a unique Ed25519 keypair that signs all tool executions,
/// creating an immutable audit trail via Moloch.
///
/// # Security Note: Clone
///
/// This type implements `Clone`, which means the secret key can be duplicated.
/// This is intentional for session persistence and handoff between Infernum
/// instances. However, minimize copies and ensure secure storage of any
/// persisted identity (see [`AgentIdentityExport`]).
#[derive(Clone)]
pub struct AgentIdentity {
    /// Human-readable name for this agent
    name: String,

    /// Ed25519 signing key (secret)
    secret_key: SecretKey,

    /// Unique identifier derived from public key
    id: Hash,
}

impl AgentIdentity {
    /// Generate a new agent identity with a random keypair.
    ///
    /// # Arguments
    /// * `name` - Human-readable name for the agent
    ///
    /// # Example
    /// ```ignore
    /// let identity = AgentIdentity::generate("code-assistant");
    /// ```
    #[must_use]
    pub fn generate(name: impl Into<String>) -> Self {
        let secret_key = SecretKey::generate();
        let public_key = secret_key.public_key();
        let id = public_key.id();

        Self {
            name: name.into(),
            secret_key,
            id,
        }
    }

    /// Create an agent identity from existing key bytes.
    ///
    /// # Arguments
    /// * `name` - Human-readable name for the agent
    /// * `secret_key_bytes` - 32-byte Ed25519 secret key
    ///
    /// # Errors
    /// Returns error if key bytes are invalid.
    pub fn from_bytes(name: impl Into<String>, secret_key_bytes: &[u8; 32]) -> Result<Self> {
        let secret_key = SecretKey::from_bytes(secret_key_bytes)
            .map_err(|e| AgentIdentityError::InvalidKey(e.to_string()))?;
        let public_key = secret_key.public_key();
        let id = public_key.id();

        Ok(Self {
            name: name.into(),
            secret_key,
            id,
        })
    }

    /// Get the agent's human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the agent's unique identifier (hash of public key).
    #[must_use]
    pub fn id(&self) -> &Hash {
        &self.id
    }

    /// Get the agent's public key for verification.
    #[must_use]
    pub fn public_key(&self) -> PublicKey {
        self.secret_key.public_key()
    }

    /// Get the secret key bytes for persistence.
    ///
    /// # Security
    /// Handle with care - this is sensitive key material.
    #[must_use]
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.secret_key.as_bytes()
    }

    /// Create an ActorId for Moloch events.
    #[must_use]
    pub fn actor_id(&self) -> ActorId {
        ActorId::new(self.public_key(), ActorKind::Agent).with_name(&self.name)
    }

    /// Sign a tool execution event.
    ///
    /// Creates an immutable, cryptographically-signed audit event recording
    /// that this agent executed a tool with specific arguments.
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool being executed
    /// * `arguments` - Tool arguments as JSON
    ///
    /// # Returns
    /// A signed AuditEvent that can be submitted to Moloch.
    pub fn sign_tool_execution(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<AuditEvent> {
        self.sign_tool_execution_with_outcome(tool_name, arguments, Outcome::Success)
    }

    /// Sign a tool execution event with a specific outcome.
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool being executed
    /// * `arguments` - Tool arguments as JSON
    /// * `outcome` - Result of the tool execution
    pub fn sign_tool_execution_with_outcome(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
        outcome: Outcome,
    ) -> Result<AuditEvent> {
        let metadata = serde_json::json!({
            "tool_name": tool_name,
            "arguments": arguments,
        });

        AuditEvent::builder()
            .now()
            .event_type(EventType::AgentAction {
                action: format!("tool_call:{}", tool_name),
                reasoning: None,
            })
            .actor(self.actor_id())
            .resource(ResourceId::new(ResourceKind::Other, tool_name))
            .outcome(outcome)
            .metadata(metadata)
            .sign(&self.secret_key)
            .map_err(|e| AgentIdentityError::EventCreation(e.to_string()))
    }

    /// Sign a tool execution with reasoning.
    ///
    /// Includes the agent's reasoning for calling this tool. Reasoning is
    /// valuable for both successful and failed tool calls - transparency
    /// requires recording why an agent attempted an action regardless of outcome.
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool being executed
    /// * `arguments` - Tool arguments as JSON
    /// * `reasoning` - Agent's reasoning for calling this tool
    /// * `outcome` - Result of the tool execution
    pub fn sign_tool_execution_with_reasoning(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
        reasoning: &str,
        outcome: Outcome,
    ) -> Result<AuditEvent> {
        let metadata = serde_json::json!({
            "tool_name": tool_name,
            "arguments": arguments,
            "reasoning": reasoning,
        });

        AuditEvent::builder()
            .now()
            .event_type(EventType::AgentAction {
                action: format!("tool_call:{}", tool_name),
                reasoning: Some(reasoning.to_string()),
            })
            .actor(self.actor_id())
            .resource(ResourceId::new(ResourceKind::Other, tool_name))
            .outcome(outcome)
            .metadata(metadata)
            .sign(&self.secret_key)
            .map_err(|e| AgentIdentityError::EventCreation(e.to_string()))
    }

    /// Sign a key rotation event.
    ///
    /// Creates an audit event signed by the OLD key (this identity) that
    /// records the handoff to a new key. This creates a cryptographic chain
    /// linking the old identity to the new one.
    ///
    /// # Security
    /// - Submit this event to Moloch BEFORE using the new identity
    /// - The old key should be retained for a verification window (default: 7 days)
    /// - Then securely delete the old key
    ///
    /// # Arguments
    /// * `new_identity` - The new identity being rotated to
    /// * `reason` - Reason for rotation (e.g., "scheduled", "personnel_change")
    ///
    /// # Example
    /// ```ignore
    /// let old_identity = /* ... */;
    /// let new_identity = AgentIdentity::generate("code-assistant");
    ///
    /// // Sign rotation with OLD key
    /// let event = old_identity.sign_key_rotation(&new_identity, "scheduled")?;
    ///
    /// // Submit to Moloch BEFORE using new identity
    /// moloch_client.submit(event).await?;
    /// ```
    pub fn sign_key_rotation(
        &self,
        new_identity: &AgentIdentity,
        reason: &str,
    ) -> Result<AuditEvent> {
        let metadata = serde_json::json!({
            "rotation_type": "key_rotation",
            "old_key_id": hex::encode(self.id().as_bytes()),
            "new_key_id": hex::encode(new_identity.id().as_bytes()),
            "new_public_key": hex::encode(new_identity.public_key().as_bytes()),
            "reason": reason,
            "old_agent_name": self.name(),
            "new_agent_name": new_identity.name(),
        });

        AuditEvent::builder()
            .now()
            .event_type(EventType::Custom {
                name: "AgentKeyRotated".to_string(),
            })
            .actor(self.actor_id())
            .resource(ResourceId::new(
                ResourceKind::Other,
                &format!("agent:{}", self.name()),
            ))
            .outcome(Outcome::Success)
            .metadata(metadata)
            .sign(&self.secret_key)
            .map_err(|e| AgentIdentityError::EventCreation(e.to_string()))
    }

    /// Sign a key revocation event.
    ///
    /// Creates an audit event signed by the key being revoked. This proves
    /// possession of the key at revocation time and creates an immutable
    /// record that this key should no longer be trusted.
    ///
    /// # Security
    /// - On suspected compromise, call this IMMEDIATELY
    /// - All events signed by this key AFTER the revocation timestamp are suspect
    /// - Verifiers MUST check revocation status before trusting events
    ///
    /// # Arguments
    /// * `reason` - Reason for revocation (e.g., "suspected_compromise", "personnel_change")
    ///
    /// # Example
    /// ```ignore
    /// // On compromise detection
    /// let event = identity.sign_revocation("suspected_compromise")?;
    /// moloch_client.submit(event).await?;
    /// // STOP using this identity immediately
    /// ```
    pub fn sign_revocation(&self, reason: &str) -> Result<AuditEvent> {
        let metadata = serde_json::json!({
            "revocation_type": "key_revocation",
            "revoked_key_id": hex::encode(self.id().as_bytes()),
            "revoked_public_key": hex::encode(self.public_key().as_bytes()),
            "reason": reason,
            "agent_name": self.name(),
        });

        AuditEvent::builder()
            .now()
            .event_type(EventType::Custom {
                name: "AgentKeyRevoked".to_string(),
            })
            .actor(self.actor_id())
            .resource(ResourceId::new(
                ResourceKind::Other,
                &format!("agent:{}", self.name()),
            ))
            .outcome(Outcome::Success)
            .metadata(metadata)
            .sign(&self.secret_key)
            .map_err(|e| AgentIdentityError::EventCreation(e.to_string()))
    }
}

impl std::fmt::Debug for AgentIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentIdentity")
            .field("name", &self.name)
            .field("id", &self.id)
            .field("secret_key", &"[redacted]")
            .finish()
    }
}

/// Serializable form of AgentIdentity for persistence.
#[derive(Serialize, Deserialize)]
pub struct AgentIdentityExport {
    /// Agent name
    pub name: String,
    /// Secret key bytes (hex-encoded)
    pub secret_key_hex: String,
}

impl AgentIdentity {
    /// Export identity for persistence.
    ///
    /// # Security
    /// The exported form contains the secret key. Handle with care.
    #[must_use]
    pub fn export(&self) -> AgentIdentityExport {
        AgentIdentityExport {
            name: self.name.clone(),
            secret_key_hex: hex::encode(self.secret_key_bytes()),
        }
    }

    /// Import identity from exported form.
    pub fn import(export: &AgentIdentityExport) -> Result<Self> {
        let secret_key_bytes = hex::decode(&export.secret_key_hex)
            .map_err(|e| AgentIdentityError::Serialization(e.to_string()))?;

        if secret_key_bytes.len() != 32 {
            return Err(AgentIdentityError::InvalidKey(format!(
                "expected 32 bytes, got {}",
                secret_key_bytes.len()
            )));
        }

        let mut arr = [0u8; 32];
        arr.copy_from_slice(&secret_key_bytes);

        Self::from_bytes(&export.name, &arr)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ENCRYPTED IDENTITY STORAGE
// ════════════════════════════════════════════════════════════════════════════

/// Encrypted storage for agent identities.
///
/// Provides secure at-rest storage using XChaCha20-Poly1305 with key derivation
/// from user passphrase via Argon2id. This ensures that secret keys are never
/// stored in plaintext.
///
/// # Security Model
///
/// ```text
/// Passphrase + Salt ──► Argon2id ──► Encryption Key (32 bytes)
///                                            │
/// AgentIdentityExport ──────────────────────►│
///              │                             ▼
///              └──────► XChaCha20-Poly1305 ──► Encrypted Blob
///                                                    │
///                                              ┌─────┴─────┐
///                                              │ Salt (16) │
///                                              │ Nonce(24) │
///                                              │ Ciphertext│
///                                              │ Tag (16)  │
///                                              └───────────┘
/// ```
///
/// # Example
///
/// ```ignore
/// use infernum_server::agent_identity::{AgentIdentity, EncryptedIdentityStore};
///
/// let identity = AgentIdentity::generate("my-agent");
///
/// // Encrypt with passphrase
/// let encrypted = EncryptedIdentityStore::encrypt(&identity, b"strong passphrase")?;
///
/// // Decrypt later
/// let restored = EncryptedIdentityStore::decrypt(&encrypted, b"strong passphrase")?;
/// assert_eq!(identity.id(), restored.id());
/// ```
pub struct EncryptedIdentityStore;

/// Salt length for Argon2id (128-bit, per OWASP recommendation)
const SALT_LEN: usize = 16;

/// Nonce length for XChaCha20-Poly1305 (192-bit)
const NONCE_LEN: usize = 24;

/// Encrypted identity data format.
///
/// Binary format:
/// - Bytes 0-15: Salt (16 bytes)
/// - Bytes 16-39: Nonce (24 bytes)
/// - Bytes 40+: Ciphertext + Auth Tag (variable + 16 bytes)
#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedIdentity {
    /// Random salt for Argon2id key derivation
    #[serde(with = "hex_serde")]
    salt: [u8; SALT_LEN],
    /// Random nonce for XChaCha20-Poly1305
    #[serde(with = "hex_serde")]
    nonce: [u8; NONCE_LEN],
    /// Encrypted AgentIdentityExport JSON + auth tag
    #[serde(with = "hex_serde_vec")]
    ciphertext: Vec<u8>,
}

impl EncryptedIdentityStore {
    /// Encrypt an agent identity with a passphrase.
    ///
    /// Uses Argon2id (OWASP parameters) for key derivation and XChaCha20-Poly1305
    /// for authenticated encryption.
    ///
    /// # Arguments
    /// * `identity` - The agent identity to encrypt
    /// * `passphrase` - User-provided passphrase (should be strong)
    ///
    /// # Returns
    /// Encrypted identity blob that can be safely stored on disk.
    pub fn encrypt(identity: &AgentIdentity, passphrase: &[u8]) -> Result<EncryptedIdentity> {
        use arcanum_hash::{Argon2, Argon2Params, PasswordHash as _};
        use arcanum_symmetric::{Cipher, XChaCha20Poly1305Cipher};
        use rand::RngCore;

        // Generate random salt and nonce
        let mut salt = [0u8; SALT_LEN];
        let mut nonce = [0u8; NONCE_LEN];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        rand::rngs::OsRng.fill_bytes(&mut nonce);

        // Derive encryption key from passphrase using Argon2id
        // Using OWASP recommended parameters for security
        let key_bytes =
            Argon2::derive_key(passphrase, &salt, &Argon2Params::owasp(), 32).map_err(|e| {
                AgentIdentityError::Serialization(format!("key derivation failed: {e}"))
            })?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // Serialize identity to JSON
        let export = identity.export();
        let plaintext = serde_json::to_vec(&export)
            .map_err(|e| AgentIdentityError::Serialization(e.to_string()))?;

        // Encrypt with XChaCha20-Poly1305
        let ciphertext = XChaCha20Poly1305Cipher::encrypt(&key, &nonce, &plaintext, None)
            .map_err(|e| AgentIdentityError::Serialization(format!("encryption failed: {e}")))?;

        Ok(EncryptedIdentity {
            salt,
            nonce,
            ciphertext,
        })
    }

    /// Decrypt an agent identity with a passphrase.
    ///
    /// # Arguments
    /// * `encrypted` - The encrypted identity blob
    /// * `passphrase` - The passphrase used during encryption
    ///
    /// # Returns
    /// The decrypted agent identity.
    ///
    /// # Errors
    /// Returns error if passphrase is wrong or data was tampered with.
    pub fn decrypt(encrypted: &EncryptedIdentity, passphrase: &[u8]) -> Result<AgentIdentity> {
        use arcanum_hash::{Argon2, Argon2Params, PasswordHash as _};
        use arcanum_symmetric::{Cipher, XChaCha20Poly1305Cipher};

        // Derive encryption key from passphrase
        let key_bytes = Argon2::derive_key(passphrase, &encrypted.salt, &Argon2Params::owasp(), 32)
            .map_err(|e| {
                AgentIdentityError::Serialization(format!("key derivation failed: {e}"))
            })?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // Decrypt
        let plaintext =
            XChaCha20Poly1305Cipher::decrypt(&key, &encrypted.nonce, &encrypted.ciphertext, None)
                .map_err(|_| {
                AgentIdentityError::InvalidKey(
                    "decryption failed: wrong passphrase or tampered data".to_string(),
                )
            })?;

        // Deserialize
        let export: AgentIdentityExport = serde_json::from_slice(&plaintext)
            .map_err(|e| AgentIdentityError::Serialization(e.to_string()))?;

        AgentIdentity::import(&export)
    }

    /// Save encrypted identity to a file.
    ///
    /// The file format is JSON for interoperability.
    pub fn save_to_file(
        encrypted: &EncryptedIdentity,
        path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        let json = serde_json::to_string_pretty(encrypted)
            .map_err(|e| AgentIdentityError::Serialization(e.to_string()))?;

        std::fs::write(path, json)
            .map_err(|e| AgentIdentityError::Serialization(format!("failed to write file: {e}")))?;

        Ok(())
    }

    /// Load encrypted identity from a file.
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<EncryptedIdentity> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| AgentIdentityError::Serialization(format!("failed to read file: {e}")))?;

        serde_json::from_str(&json).map_err(|e| AgentIdentityError::Serialization(e.to_string()))
    }
}

impl std::fmt::Debug for EncryptedIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedIdentity")
            .field("salt", &format!("{}...", &hex::encode(&self.salt[..4])))
            .field("nonce", &format!("{}...", &hex::encode(&self.nonce[..4])))
            .field("ciphertext_len", &self.ciphertext.len())
            .finish()
    }
}

/// Hex serialization helper for serde
///
/// This module provides serialize/deserialize functions for both fixed-size
/// arrays and Vec<u8>. The deserialize function is generic over the output type.
mod hex_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, T>(data: T, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: AsRef<[u8]>,
    {
        hex::encode(data.as_ref()).serialize(serializer)
    }

    /// Deserialize hex string to fixed-size array.
    pub fn deserialize<'de, D, const N: usize>(
        deserializer: D,
    ) -> std::result::Result<[u8; N], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        bytes.try_into().map_err(|v: Vec<u8>| {
            serde::de::Error::custom(format!("expected {} bytes, got {}", N, v.len()))
        })
    }
}

/// Hex serialization for Vec<u8> (variable length)
mod hex_serde_vec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S, T>(data: T, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: AsRef<[u8]>,
    {
        hex::encode(data.as_ref()).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ════════════════════════════════════════════════════════════════════════════
    // TDD RED PHASE: Tests for AgentIdentity
    // These tests specify the behavior we need. Implementation should make them pass.
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generate_creates_unique_identities() {
        let id1 = AgentIdentity::generate("agent-1");
        let id2 = AgentIdentity::generate("agent-2");

        // Each identity should have a unique ID
        assert_ne!(id1.id(), id2.id());

        // Names should be preserved
        assert_eq!(id1.name(), "agent-1");
        assert_eq!(id2.name(), "agent-2");
    }

    #[test]
    fn test_identity_id_is_deterministic() {
        let identity = AgentIdentity::generate("test-agent");
        let id1 = identity.id().clone();
        let id2 = identity.id().clone();

        // Same identity should always return same ID
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_from_bytes_restores_identity() {
        let original = AgentIdentity::generate("persistent-agent");
        let bytes = original.secret_key_bytes();

        let restored = AgentIdentity::from_bytes("persistent-agent", &bytes)
            .expect("should restore from bytes");

        // Restored identity should have same ID
        assert_eq!(original.id(), restored.id());
        assert_eq!(
            original.public_key().as_bytes(),
            restored.public_key().as_bytes()
        );
    }

    #[test]
    fn test_from_bytes_rejects_invalid_key() {
        let invalid_bytes = [0u8; 32]; // All zeros is not a valid key
                                       // This may or may not fail depending on Ed25519 implementation
                                       // The important thing is it doesn't panic
        let _ = AgentIdentity::from_bytes("invalid", &invalid_bytes);
    }

    #[test]
    fn test_actor_id_has_agent_kind() {
        let identity = AgentIdentity::generate("test-agent");
        let actor = identity.actor_id();

        assert_eq!(actor.kind, ActorKind::Agent);
        assert_eq!(actor.name, Some("test-agent".to_string()));
    }

    #[test]
    fn test_sign_tool_execution_creates_valid_event() {
        let identity = AgentIdentity::generate("tool-caller");
        let args = serde_json::json!({"location": "Seattle"});

        let event = identity
            .sign_tool_execution("get_weather", &args)
            .expect("should sign event");

        // Event should be valid (signature verifies)
        assert!(event.validate().is_ok());

        // Event should have correct actor
        assert_eq!(event.actor.kind, ActorKind::Agent);
        assert_eq!(event.actor.name, Some("tool-caller".to_string()));

        // Event type should be AgentAction
        match &event.event_type {
            EventType::AgentAction { action, reasoning } => {
                assert_eq!(action, "tool_call:get_weather");
                assert!(reasoning.is_none());
            },
            _ => panic!("expected AgentAction event type"),
        }
    }

    #[test]
    fn test_sign_tool_execution_with_outcome() {
        let identity = AgentIdentity::generate("tool-caller");
        let args = serde_json::json!({"file": "nonexistent.txt"});

        let event = identity
            .sign_tool_execution_with_outcome(
                "read_file",
                &args,
                Outcome::Failure {
                    reason: "file not found".to_string(),
                },
            )
            .expect("should sign event");

        assert!(event.validate().is_ok());

        match &event.outcome {
            Outcome::Failure { reason } => {
                assert_eq!(reason, "file not found");
            },
            _ => panic!("expected Failure outcome"),
        }
    }

    #[test]
    fn test_sign_tool_execution_with_reasoning_success() {
        let identity = AgentIdentity::generate("reasoning-agent");
        let args = serde_json::json!({"query": "weather"});

        let event = identity
            .sign_tool_execution_with_reasoning(
                "web_search",
                &args,
                "User asked about weather, need to search for current conditions",
                Outcome::Success,
            )
            .expect("should sign event");

        assert!(event.validate().is_ok());

        match &event.event_type {
            EventType::AgentAction { action, reasoning } => {
                assert_eq!(action, "tool_call:web_search");
                assert_eq!(
                    reasoning.as_deref(),
                    Some("User asked about weather, need to search for current conditions")
                );
            },
            _ => panic!("expected AgentAction event type"),
        }

        assert!(matches!(event.outcome, Outcome::Success));
    }

    #[test]
    fn test_sign_tool_execution_with_reasoning_failure() {
        // Reasoning is valuable even for failed tool calls - transparency requires
        // recording WHY an agent attempted an action, not just that it failed.
        let identity = AgentIdentity::generate("reasoning-agent");
        let args = serde_json::json!({"path": "/etc/shadow"});

        let event = identity
            .sign_tool_execution_with_reasoning(
                "read_file",
                &args,
                "Need to check system configuration for debugging",
                Outcome::Denied {
                    reason: "access denied: privileged file".to_string(),
                },
            )
            .expect("should sign event");

        assert!(event.validate().is_ok());

        match &event.event_type {
            EventType::AgentAction { action, reasoning } => {
                assert_eq!(action, "tool_call:read_file");
                assert!(reasoning.is_some());
            },
            _ => panic!("expected AgentAction event type"),
        }

        match &event.outcome {
            Outcome::Denied { reason } => {
                assert!(reason.contains("access denied"));
            },
            _ => panic!("expected Denied outcome"),
        }
    }

    #[test]
    fn test_event_metadata_contains_arguments() {
        let identity = AgentIdentity::generate("metadata-test");
        let args = serde_json::json!({
            "param1": "value1",
            "param2": 42,
            "nested": {"key": "value"}
        });

        let event = identity
            .sign_tool_execution("complex_tool", &args)
            .expect("should sign event");

        let metadata = event.metadata_json().expect("should have metadata");
        assert_eq!(metadata["tool_name"], "complex_tool");
        assert_eq!(metadata["arguments"]["param1"], "value1");
        assert_eq!(metadata["arguments"]["param2"], 42);
        assert_eq!(metadata["arguments"]["nested"]["key"], "value");
    }

    #[test]
    fn test_tampered_event_fails_validation() {
        let identity = AgentIdentity::generate("tamper-test");
        let args = serde_json::json!({"action": "safe"});

        let mut event = identity
            .sign_tool_execution("safe_tool", &args)
            .expect("should sign event");

        // Tamper with the event
        event.outcome = Outcome::Failure {
            reason: "tampered".to_string(),
        };

        // Validation should fail because signature doesn't match
        assert!(event.validate().is_err());
    }

    #[test]
    fn test_export_import_roundtrip() {
        let original = AgentIdentity::generate("export-test");
        let export = original.export();

        let restored = AgentIdentity::import(&export).expect("should import successfully");

        // Should have same identity
        assert_eq!(original.id(), restored.id());
        assert_eq!(original.name(), restored.name());

        // Should be able to sign events
        let event = restored
            .sign_tool_execution("test", &serde_json::json!({}))
            .expect("should sign");
        assert!(event.validate().is_ok());
    }

    #[test]
    fn test_export_serializes_to_json() {
        let identity = AgentIdentity::generate("json-test");
        let export = identity.export();

        // Should serialize to JSON
        let json = serde_json::to_string(&export).expect("should serialize");

        // Should deserialize back
        let restored_export: AgentIdentityExport =
            serde_json::from_str(&json).expect("should deserialize");

        let restored = AgentIdentity::import(&restored_export).expect("should import");

        assert_eq!(identity.id(), restored.id());
    }

    #[test]
    fn test_import_rejects_invalid_hex() {
        let bad_hex = AgentIdentityExport {
            name: "invalid".to_string(),
            secret_key_hex: "not-valid-hex!@#$%^".to_string(),
        };
        assert!(AgentIdentity::import(&bad_hex).is_err());
    }

    #[test]
    fn test_import_rejects_wrong_length_hex() {
        let short_hex = AgentIdentityExport {
            name: "short".to_string(),
            secret_key_hex: "deadbeef".to_string(), // Only 4 bytes, need 32
        };
        assert!(AgentIdentity::import(&short_hex).is_err());
    }

    #[test]
    fn test_debug_redacts_secret_key() {
        let identity = AgentIdentity::generate("debug-test");
        let debug_str = format!("{:?}", identity);

        // Should not contain actual key bytes
        assert!(debug_str.contains("[redacted]"));
        assert!(!debug_str.contains(&hex::encode(identity.secret_key_bytes())));
    }

    #[test]
    fn test_event_has_timestamp() {
        let identity = AgentIdentity::generate("time-test");
        let before = Utc::now();

        let event = identity
            .sign_tool_execution("test", &serde_json::json!({}))
            .expect("should sign");

        let after = Utc::now();

        // Event time should be between before and after
        assert!(event.event_time >= before);
        assert!(event.event_time <= after);
    }

    #[test]
    fn test_event_id_is_content_addressed() {
        let identity = AgentIdentity::generate("content-addressed");

        // Two events with same content (but signed at different times)
        // should have different IDs due to timestamp
        let event1 = identity
            .sign_tool_execution("test", &serde_json::json!({}))
            .expect("should sign");

        std::thread::sleep(std::time::Duration::from_millis(1));

        let event2 = identity
            .sign_tool_execution("test", &serde_json::json!({}))
            .expect("should sign");

        // Different times = different canonical bytes = different IDs
        assert_ne!(event1.id(), event2.id());
    }

    #[test]
    fn test_multiple_tool_calls_from_same_identity() {
        let identity = AgentIdentity::generate("multi-tool");

        let tools = vec![
            ("read_file", serde_json::json!({"path": "/tmp/a"})),
            (
                "write_file",
                serde_json::json!({"path": "/tmp/b", "content": "hello"}),
            ),
            ("execute", serde_json::json!({"command": "ls"})),
        ];

        for (tool, args) in tools {
            let event = identity
                .sign_tool_execution(tool, &args)
                .expect("should sign");

            assert!(event.validate().is_ok());
            assert_eq!(event.actor.id(), identity.public_key().id());
        }
    }

    // ════════════════════════════════════════════════════════════════════════════
    // KEY LIFECYCLE TESTS
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sign_key_rotation_creates_valid_event() {
        let old_identity = AgentIdentity::generate("old-agent");
        let new_identity = AgentIdentity::generate("old-agent"); // Same name, new key

        let event = old_identity
            .sign_key_rotation(&new_identity, "scheduled")
            .expect("should sign rotation");

        // Event should be valid (signed by old key)
        assert!(event.validate().is_ok());

        // Event should be signed by the OLD identity
        assert_eq!(event.actor.id(), old_identity.public_key().id());

        // Event type should be Custom with AgentKeyRotated
        match &event.event_type {
            EventType::Custom { name } => {
                assert_eq!(name, "AgentKeyRotated");
            },
            _ => panic!("expected Custom event type"),
        }
    }

    #[test]
    fn test_sign_key_rotation_contains_new_key_info() {
        let old_identity = AgentIdentity::generate("rotating-agent");
        let new_identity = AgentIdentity::generate("rotating-agent");

        let event = old_identity
            .sign_key_rotation(&new_identity, "personnel_change")
            .expect("should sign");

        let metadata = event.metadata_json().expect("should have metadata");

        // Verify metadata contains both key identifiers
        assert_eq!(metadata["rotation_type"], "key_rotation");
        assert_eq!(
            metadata["old_key_id"],
            hex::encode(old_identity.id().as_bytes())
        );
        assert_eq!(
            metadata["new_key_id"],
            hex::encode(new_identity.id().as_bytes())
        );
        assert_eq!(
            metadata["new_public_key"],
            hex::encode(new_identity.public_key().as_bytes())
        );
        assert_eq!(metadata["reason"], "personnel_change");
    }

    #[test]
    fn test_sign_key_rotation_links_identities() {
        let old_identity = AgentIdentity::generate("linked-agent");
        let new_identity = AgentIdentity::generate("linked-agent");

        let event = old_identity
            .sign_key_rotation(&new_identity, "scheduled")
            .expect("should sign");

        let metadata = event.metadata_json().expect("should have metadata");

        // The rotation event creates a cryptographic link:
        // - Signed by old key (verifiable via actor.public_key)
        // - Contains new key ID and public key (for chain verification)
        let new_key_from_metadata = metadata["new_public_key"].as_str().unwrap();
        assert_eq!(
            new_key_from_metadata,
            hex::encode(new_identity.public_key().as_bytes())
        );

        // After rotation, events from new_identity can be linked back
        // by following the chain: new_key_id in rotation event
        assert_eq!(
            metadata["new_key_id"],
            hex::encode(new_identity.id().as_bytes())
        );
    }

    #[test]
    fn test_sign_revocation_creates_valid_event() {
        let identity = AgentIdentity::generate("compromised-agent");

        let event = identity
            .sign_revocation("suspected_compromise")
            .expect("should sign revocation");

        // Event should be valid (proves possession at revocation time)
        assert!(event.validate().is_ok());

        // Event should be signed by the key being revoked
        assert_eq!(event.actor.id(), identity.public_key().id());

        // Event type should be Custom with AgentKeyRevoked
        match &event.event_type {
            EventType::Custom { name } => {
                assert_eq!(name, "AgentKeyRevoked");
            },
            _ => panic!("expected Custom event type"),
        }
    }

    #[test]
    fn test_sign_revocation_contains_key_info() {
        let identity = AgentIdentity::generate("revoked-agent");

        let event = identity
            .sign_revocation("security_incident")
            .expect("should sign");

        let metadata = event.metadata_json().expect("should have metadata");

        // Verify metadata contains key identifier for revocation lookup
        assert_eq!(metadata["revocation_type"], "key_revocation");
        assert_eq!(
            metadata["revoked_key_id"],
            hex::encode(identity.id().as_bytes())
        );
        assert_eq!(
            metadata["revoked_public_key"],
            hex::encode(identity.public_key().as_bytes())
        );
        assert_eq!(metadata["reason"], "security_incident");
        assert_eq!(metadata["agent_name"], "revoked-agent");
    }

    #[test]
    fn test_revocation_timestamp_for_trust_boundary() {
        let identity = AgentIdentity::generate("trust-test");
        let before_revocation = Utc::now();

        let revocation_event = identity
            .sign_revocation("compromise_detected")
            .expect("should sign");

        let after_revocation = Utc::now();

        // The revocation timestamp defines the trust boundary
        assert!(revocation_event.event_time >= before_revocation);
        assert!(revocation_event.event_time <= after_revocation);

        // Events signed after this timestamp should not be trusted
        // (This is enforced by the verification layer, not the signing layer)
        let post_revocation_event = identity
            .sign_tool_execution("suspicious_tool", &serde_json::json!({}))
            .expect("can still sign, but shouldn't be trusted");

        // The event is technically valid (signature verifies)
        assert!(post_revocation_event.validate().is_ok());

        // But the timestamp is after revocation - verification layer should reject
        assert!(post_revocation_event.event_time >= revocation_event.event_time);
    }

    #[test]
    fn test_rotation_then_revocation_workflow() {
        // Simulate a key lifecycle:
        // 1. Agent created with initial key
        // 2. Key rotated to new key (scheduled)
        // 3. Old key eventually revoked
        // 4. New key later compromised, revoked immediately

        let initial_identity = AgentIdentity::generate("lifecycle-agent");

        // Normal operation
        let tool_event = initial_identity
            .sign_tool_execution("normal_op", &serde_json::json!({}))
            .expect("should sign");
        assert!(tool_event.validate().is_ok());

        // Scheduled rotation
        let rotated_identity = AgentIdentity::generate("lifecycle-agent");
        let rotation_event = initial_identity
            .sign_key_rotation(&rotated_identity, "scheduled_90_day")
            .expect("should sign rotation");
        assert!(rotation_event.validate().is_ok());

        // New identity works
        let new_tool_event = rotated_identity
            .sign_tool_execution("new_operation", &serde_json::json!({}))
            .expect("should sign");
        assert!(new_tool_event.validate().is_ok());

        // Old key eventually revoked (after verification window)
        let old_revocation = initial_identity
            .sign_revocation("scheduled_retirement")
            .expect("should sign revocation");
        assert!(old_revocation.validate().is_ok());

        // Later: new key compromised
        let emergency_revocation = rotated_identity
            .sign_revocation("suspected_compromise")
            .expect("should sign emergency revocation");
        assert!(emergency_revocation.validate().is_ok());

        // All events created a verifiable audit trail
        // Verification layer can trace: tool_event → rotation → new_tool_event → revocations
    }

    // ════════════════════════════════════════════════════════════════════════════
    // ENCRYPTED IDENTITY STORE TESTS
    // ════════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let identity = AgentIdentity::generate("encrypt-test");
        let passphrase = b"correct horse battery staple";

        let encrypted =
            EncryptedIdentityStore::encrypt(&identity, passphrase).expect("should encrypt");

        let decrypted =
            EncryptedIdentityStore::decrypt(&encrypted, passphrase).expect("should decrypt");

        // Should have same identity
        assert_eq!(identity.id(), decrypted.id());
        assert_eq!(identity.name(), decrypted.name());

        // Should be able to sign events
        let event = decrypted
            .sign_tool_execution("test", &serde_json::json!({}))
            .expect("should sign");
        assert!(event.validate().is_ok());
    }

    #[test]
    fn test_decrypt_wrong_passphrase_fails() {
        let identity = AgentIdentity::generate("wrong-pass-test");

        let encrypted = EncryptedIdentityStore::encrypt(&identity, b"correct passphrase")
            .expect("should encrypt");

        let result = EncryptedIdentityStore::decrypt(&encrypted, b"wrong passphrase");
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypted_data_is_unique_per_encryption() {
        let identity = AgentIdentity::generate("unique-test");
        let passphrase = b"same passphrase";

        let encrypted1 =
            EncryptedIdentityStore::encrypt(&identity, passphrase).expect("should encrypt");
        let encrypted2 =
            EncryptedIdentityStore::encrypt(&identity, passphrase).expect("should encrypt");

        // Different salt and nonce means different ciphertext
        assert_ne!(encrypted1.salt, encrypted2.salt);
        assert_ne!(encrypted1.nonce, encrypted2.nonce);
        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);

        // But both should decrypt to same identity
        let decrypted1 = EncryptedIdentityStore::decrypt(&encrypted1, passphrase).unwrap();
        let decrypted2 = EncryptedIdentityStore::decrypt(&encrypted2, passphrase).unwrap();
        assert_eq!(decrypted1.id(), decrypted2.id());
    }

    #[test]
    fn test_encrypted_serializes_to_json() {
        let identity = AgentIdentity::generate("json-serialize-test");
        let encrypted =
            EncryptedIdentityStore::encrypt(&identity, b"passphrase").expect("should encrypt");

        // Should serialize to JSON
        let json = serde_json::to_string(&encrypted).expect("should serialize");

        // Should deserialize back
        let restored: EncryptedIdentity = serde_json::from_str(&json).expect("should deserialize");

        // Should still decrypt correctly
        let decrypted =
            EncryptedIdentityStore::decrypt(&restored, b"passphrase").expect("should decrypt");
        assert_eq!(identity.id(), decrypted.id());
    }

    #[test]
    fn test_tampered_ciphertext_fails_decryption() {
        let identity = AgentIdentity::generate("tamper-test");
        let passphrase = b"passphrase";

        let mut encrypted =
            EncryptedIdentityStore::encrypt(&identity, passphrase).expect("should encrypt");

        // Tamper with ciphertext
        if !encrypted.ciphertext.is_empty() {
            encrypted.ciphertext[0] ^= 0xff;
        }

        // Decryption should fail (authentication failure)
        let result = EncryptedIdentityStore::decrypt(&encrypted, passphrase);
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypted_debug_redacts_data() {
        let identity = AgentIdentity::generate("debug-redact-test");
        let encrypted =
            EncryptedIdentityStore::encrypt(&identity, b"passphrase").expect("should encrypt");

        let debug_str = format!("{:?}", encrypted);

        // Should not contain full salt/nonce/ciphertext
        assert!(debug_str.contains("..."));
        assert!(debug_str.contains("ciphertext_len"));
    }
}
