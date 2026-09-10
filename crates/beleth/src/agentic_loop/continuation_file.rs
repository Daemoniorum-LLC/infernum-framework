//! Filesystem-backed [`ContinuationStore`].
//!
//! [`InMemoryContinuationStore`](super::continuation::InMemoryContinuationStore)
//! dies with the process, which makes it useless for a CLI: the token printed
//! at the end of a run would refer to state that no longer exists by the time
//! anyone typed it. Advertising `--resume <token>` on top of an in-memory store
//! would be a fresh instance of exactly the defect the resume driver exists to
//! fix — a field that is true in the narrow sense (a token was minted) and
//! false in the sense the reader cares about (it can be used).
//!
//! One JSON file per continuation, named by token. TTL is enforced on read as
//! well as by [`cleanup_expired`](FileContinuationStore::cleanup_expired), so
//! an expired state is never returned even if nothing has swept yet.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;

use super::continuation::{ContinuationState, ContinuationStore, StoreError};

/// Default lifetime of a stored continuation.
const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Stores continuations as JSON files under a directory.
#[derive(Debug, Clone)]
pub struct FileContinuationStore {
    dir: PathBuf,
    ttl: Duration,
}

impl FileContinuationStore {
    /// Creates a store rooted at `dir`, creating it if absent.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Backend`] if the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>, ttl: Duration) -> Result<Self, StoreError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| StoreError::Backend {
            message: format!("creating {}: {e}", dir.display()),
        })?;
        Ok(Self { dir, ttl })
    }

    /// Creates a store under the user's data directory with a 24-hour TTL.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::Backend`] if the directory cannot be created.
    pub fn with_defaults() -> Result<Self, StoreError> {
        let base = dirs::data_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("infernum")
            .join("continuations");
        Self::new(base, DEFAULT_TTL)
    }

    /// Path for a token, rejecting anything that could escape the directory.
    ///
    /// Tokens are `cont_{session}_{uuid}` and arrive from a command line, so
    /// they are treated as untrusted input rather than assumed well-formed.
    fn path_for(&self, token: &str) -> Result<PathBuf, StoreError> {
        let safe = token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !safe || token.is_empty() {
            return Err(StoreError::NotFound {
                token: token.to_string(),
            });
        }
        Ok(self.dir.join(format!("{token}.json")))
    }

    /// True when `stored_at` is older than the TTL.
    fn is_expired(&self, stored_at: SystemTime) -> bool {
        SystemTime::now()
            .duration_since(stored_at)
            .map(|age| age > self.ttl)
            .unwrap_or(false)
    }

    /// Reads and parses one file, returning `None` when absent or expired.
    fn read_state(&self, path: &Path) -> Option<ContinuationState> {
        let bytes = std::fs::read(path).ok()?;
        let state: ContinuationState = serde_json::from_slice(&bytes).ok()?;
        if self.is_expired(state.stored_at) {
            // Expired states are removed on sight so a later sweep is an
            // optimisation rather than a correctness requirement.
            let _ = std::fs::remove_file(path);
            return None;
        }
        Some(state)
    }
}

#[async_trait]
impl ContinuationStore for FileContinuationStore {
    async fn store(&self, state: ContinuationState) -> Result<String, StoreError> {
        let token = state.token.clone();
        let path = self.path_for(&token)?;
        let json = serde_json::to_vec_pretty(&state).map_err(|e| StoreError::Backend {
            message: format!("serializing continuation: {e}"),
        })?;

        // Write-then-rename so a crash mid-write cannot leave a truncated file
        // that would later parse as a valid-but-wrong continuation.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &json).map_err(|e| StoreError::Backend {
            message: format!("writing {}: {e}", tmp.display()),
        })?;
        std::fs::rename(&tmp, &path).map_err(|e| StoreError::Backend {
            message: format!("renaming into {}: {e}", path.display()),
        })?;

        Ok(token)
    }

    async fn load(&self, token: &str) -> Result<Option<ContinuationState>, StoreError> {
        Ok(self.read_state(&self.path_for(token)?))
    }

    async fn remove(&self, token: &str) -> Result<(), StoreError> {
        let _ = std::fs::remove_file(self.path_for(token)?);
        Ok(())
    }

    async fn cleanup_expired(&self) -> Result<u32, StoreError> {
        let mut removed = 0;
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Ok(0);
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                // read_state deletes expired files as a side effect.
                if self.read_state(&path).is_none() {
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic_loop::continuation::create_continuation_state;
    use crate::agentic_loop::types::{
        AutonomyGrant, LoopConfig, NaturalTermination, StuckRequest, TerminationReason,
    };

    fn sample(session: &str) -> ContinuationState {
        create_continuation_state(
            session,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            3,
            2,
            100,
            LoopConfig::default(),
            AutonomyGrant::default(),
            Some("sys".to_string()),
            None,
            TerminationReason::Natural(NaturalTermination::AgentStuck {
                attempts: 1,
                request: StuckRequest::Clarification(vec!["which file?".to_string()]),
            }),
        )
    }

    #[tokio::test]
    async fn round_trips_through_the_filesystem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileContinuationStore::new(dir.path(), Duration::from_secs(3600)).unwrap();

        let state = sample("sess-a");
        let token = store.store(state.clone()).await.expect("store");

        // A *separate* store instance, which is the case that matters: the
        // CLI process that resumes is not the one that stored.
        let reader = FileContinuationStore::new(dir.path(), Duration::from_secs(3600)).unwrap();
        let loaded = reader.load(&token).await.expect("load").expect("present");

        assert_eq!(loaded.session_id, "sess-a");
        assert_eq!(loaded.iterations_completed, 3);
        assert_eq!(loaded.tool_calls_made, 2);
    }

    #[tokio::test]
    async fn unknown_token_is_none_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileContinuationStore::new(dir.path(), Duration::from_secs(3600)).unwrap();
        assert!(store.load("cont_nope_1234").await.expect("load").is_none());
    }

    #[tokio::test]
    async fn expired_state_is_not_returned() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Zero TTL: anything stored is already expired.
        let store = FileContinuationStore::new(dir.path(), Duration::from_secs(0)).unwrap();
        let token = store.store(sample("sess-b")).await.expect("store");
        assert!(
            store.load(&token).await.expect("load").is_none(),
            "an expired continuation must not be resumable"
        );
    }

    #[tokio::test]
    async fn removed_state_is_gone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileContinuationStore::new(dir.path(), Duration::from_secs(3600)).unwrap();
        let token = store.store(sample("sess-c")).await.expect("store");
        store.remove(&token).await.expect("remove");
        assert!(store.load(&token).await.expect("load").is_none());
    }

    /// A token containing path separators must not escape the store directory.
    #[tokio::test]
    async fn traversal_tokens_are_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FileContinuationStore::new(dir.path(), Duration::from_secs(3600)).unwrap();
        for bad in ["../escape", "a/b", "/etc/passwd", "", "x.y"] {
            assert!(
                store.load(bad).await.is_err(),
                "token {bad:?} should be refused"
            );
        }
    }
}
