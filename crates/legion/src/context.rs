//! Shared context management for Legion.
//!
//! Implements holographic context distribution where every agent
//! holds a fragment of the shared context.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// Unique identifier for a context fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FragmentId(pub u64);

impl FragmentId {
    /// Creates a new fragment ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// A fragment of shared context.
#[derive(Debug, Clone)]
pub struct ContextFragment {
    /// Fragment identifier.
    pub id: FragmentId,
    /// Content of this fragment.
    pub content: String,
    /// Importance weight (0.0 - 1.0).
    pub importance: f32,
    /// Which agents have this fragment.
    pub holders: Vec<String>,
}

impl ContextFragment {
    /// Creates a new context fragment.
    pub fn new(id: FragmentId, content: impl Into<String>) -> Self {
        Self {
            id,
            content: content.into(),
            importance: 1.0,
            holders: Vec::new(),
        }
    }

    /// Sets the importance weight.
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Adds a holder agent.
    pub fn add_holder(&mut self, agent_id: impl Into<String>) {
        self.holders.push(agent_id.into());
    }
}

/// Shared context distributed across agents.
///
/// Implements holographic distribution where:
/// - Essential fragments are replicated to all agents
/// - Importance-weighted fragments are distributed proportionally
/// - Any subset of agents can reconstruct approximate context
pub struct SharedContext {
    /// All context fragments.
    fragments: RwLock<HashMap<FragmentId, ContextFragment>>,
    /// Fragment counter for ID generation.
    next_id: AtomicU64,
    /// Number of agents sharing context.
    agent_count: usize,
    /// Total context size in characters.
    total_size: AtomicU64,
}

impl SharedContext {
    /// Creates a new shared context.
    pub fn new(agent_count: usize) -> Self {
        Self {
            fragments: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            agent_count,
            total_size: AtomicU64::new(0),
        }
    }

    /// Adds a fragment to the shared context.
    pub fn add_fragment(&self, content: impl Into<String>, importance: f32) -> FragmentId {
        let content = content.into();
        let size = content.len() as u64;
        let id = FragmentId::new(self.next_id.fetch_add(1, Ordering::Relaxed));

        let fragment = ContextFragment::new(id, content).with_importance(importance);

        self.fragments.write().insert(id, fragment);
        self.total_size.fetch_add(size, Ordering::Relaxed);

        id
    }

    /// Gets a fragment by ID.
    pub fn get_fragment(&self, id: FragmentId) -> Option<ContextFragment> {
        self.fragments.read().get(&id).cloned()
    }

    /// Returns all fragment IDs.
    pub fn fragment_ids(&self) -> Vec<FragmentId> {
        self.fragments.read().keys().copied().collect()
    }

    /// Returns fragments for an agent at a given context fraction.
    ///
    /// Higher fractions get more fragments, weighted by importance.
    pub fn fragments_for_fraction(&self, fraction: f32) -> Vec<ContextFragment> {
        let fragments = self.fragments.read();
        let mut sorted: Vec<_> = fragments.values().cloned().collect();

        // Sort by importance (highest first)
        sorted.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

        // Take proportion based on fraction
        let count = ((sorted.len() as f32 * fraction).ceil() as usize).min(sorted.len());
        sorted.truncate(count);

        sorted
    }

    /// Returns total context size in characters.
    pub fn total_size(&self) -> u64 {
        self.total_size.load(Ordering::Relaxed)
    }

    /// Returns number of fragments.
    pub fn fragment_count(&self) -> usize {
        self.fragments.read().len()
    }

    /// Returns the configured agent count.
    pub fn agent_count(&self) -> usize {
        self.agent_count
    }

    /// Clears all context.
    pub fn clear(&self) {
        self.fragments.write().clear();
        self.total_size.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = SharedContext::new(4);
        assert_eq!(ctx.agent_count(), 4);
        assert_eq!(ctx.fragment_count(), 0);
    }

    #[test]
    fn test_add_fragment() {
        let ctx = SharedContext::new(4);

        let id = ctx.add_fragment("Test content", 1.0);
        assert_eq!(ctx.fragment_count(), 1);

        let fragment = ctx.get_fragment(id);
        assert!(fragment.is_some());
        assert_eq!(fragment.as_ref().map(|f| f.content.as_str()), Some("Test content"));
    }

    #[test]
    fn test_fragments_for_fraction() {
        let ctx = SharedContext::new(4);

        // Add fragments with varying importance
        ctx.add_fragment("Critical", 1.0);
        ctx.add_fragment("Important", 0.8);
        ctx.add_fragment("Normal", 0.5);
        ctx.add_fragment("Low priority", 0.2);

        // 25% should get 1 fragment (the most important)
        let gamma_frags = ctx.fragments_for_fraction(0.25);
        assert_eq!(gamma_frags.len(), 1);
        assert_eq!(gamma_frags[0].content, "Critical");

        // 50% should get 2 fragments
        let beta_frags = ctx.fragments_for_fraction(0.50);
        assert_eq!(beta_frags.len(), 2);

        // 100% should get all fragments
        let delta_frags = ctx.fragments_for_fraction(1.0);
        assert_eq!(delta_frags.len(), 4);
    }

    #[test]
    fn test_context_size_tracking() {
        let ctx = SharedContext::new(2);

        ctx.add_fragment("Hello", 1.0); // 5 chars
        ctx.add_fragment("World", 0.5); // 5 chars

        assert_eq!(ctx.total_size(), 10);
    }

    #[test]
    fn test_clear_context() {
        let ctx = SharedContext::new(2);

        ctx.add_fragment("Test", 1.0);
        assert_eq!(ctx.fragment_count(), 1);

        ctx.clear();
        assert_eq!(ctx.fragment_count(), 0);
        assert_eq!(ctx.total_size(), 0);
    }
}
