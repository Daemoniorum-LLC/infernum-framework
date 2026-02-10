//! LRU (Least Recently Used) tracking for cache eviction.

use std::collections::HashMap;
use std::time::Instant;

/// LRU tracker for cache eviction ordering.
///
/// Tracks access times for items and provides eviction ordering
/// based on least recent use within priority tiers.
#[derive(Debug)]
pub struct LruTracker<K: std::hash::Hash + Eq + Clone> {
    /// Last access time for each item.
    access_times: HashMap<K, Instant>,

    /// Priority for each item (higher = less likely to evict).
    priorities: HashMap<K, f32>,
}

impl<K: std::hash::Hash + Eq + Clone> LruTracker<K> {
    /// Create a new LRU tracker.
    pub fn new() -> Self {
        Self {
            access_times: HashMap::new(),
            priorities: HashMap::new(),
        }
    }

    /// Record an access to an item.
    pub fn touch(&mut self, key: K) {
        self.access_times.insert(key, Instant::now());
    }

    /// Record an access with priority.
    pub fn touch_with_priority(&mut self, key: K, priority: f32) {
        self.access_times.insert(key.clone(), Instant::now());
        self.priorities.insert(key, priority);
    }

    /// Remove an item from tracking.
    pub fn remove(&mut self, key: &K) {
        self.access_times.remove(key);
        self.priorities.remove(key);
    }

    /// Get last access time for an item.
    pub fn last_access(&self, key: &K) -> Option<Instant> {
        self.access_times.get(key).copied()
    }

    /// Get priority for an item.
    pub fn priority(&self, key: &K) -> f32 {
        self.priorities.get(key).copied().unwrap_or(0.5)
    }

    /// Check if an item is being tracked.
    pub fn contains(&self, key: &K) -> bool {
        self.access_times.contains_key(key)
    }

    /// Get the number of tracked items.
    pub fn len(&self) -> usize {
        self.access_times.len()
    }

    /// Check if tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.access_times.is_empty()
    }

    /// Get eviction candidates sorted by priority (lowest first), then by age (oldest first).
    ///
    /// Returns items that should be evicted first at the front of the list.
    pub fn eviction_order(&self) -> Vec<K> {
        let mut items: Vec<_> = self
            .access_times
            .iter()
            .map(|(k, &time)| {
                let priority = self.priorities.get(k).copied().unwrap_or(0.5);
                (k.clone(), priority, time)
            })
            .collect();

        // Sort by priority (ascending), then by time (ascending = oldest first)
        items.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });

        items.into_iter().map(|(k, _, _)| k).collect()
    }

    /// Get the single best eviction candidate (lowest priority, oldest access).
    pub fn pop_lru(&mut self) -> Option<K> {
        let candidates = self.eviction_order();
        if let Some(key) = candidates.into_iter().next() {
            self.remove(&key);
            Some(key)
        } else {
            None
        }
    }

    /// Get eviction candidates up to a target size.
    ///
    /// Returns items to evict to free at least `target_bytes`, using the
    /// provided size function to determine each item's size.
    pub fn eviction_candidates_for_size<F>(&self, target_bytes: u64, size_fn: F) -> Vec<K>
    where
        F: Fn(&K) -> u64,
    {
        let mut candidates = Vec::new();
        let mut freed = 0u64;

        for key in self.eviction_order() {
            if freed >= target_bytes {
                break;
            }
            freed += size_fn(&key);
            candidates.push(key);
        }

        candidates
    }

    /// Clear all tracking.
    pub fn clear(&mut self) {
        self.access_times.clear();
        self.priorities.clear();
    }
}

impl<K: std::hash::Hash + Eq + Clone> Default for LruTracker<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_lru_basic() {
        let mut tracker = LruTracker::new();

        tracker.touch(1);
        tracker.touch(2);
        tracker.touch(3);

        assert_eq!(tracker.len(), 3);
        assert!(tracker.contains(&1));
        assert!(tracker.contains(&2));
        assert!(tracker.contains(&3));
    }

    #[test]
    fn test_lru_eviction_order_by_time() {
        let mut tracker = LruTracker::new();

        // All same priority, order by time
        tracker.touch_with_priority(1, 0.5);
        sleep(Duration::from_millis(10));
        tracker.touch_with_priority(2, 0.5);
        sleep(Duration::from_millis(10));
        tracker.touch_with_priority(3, 0.5);

        let order = tracker.eviction_order();
        // Oldest first
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn test_lru_eviction_order_by_priority() {
        let mut tracker = LruTracker::new();

        // Different priorities - priority trumps time
        tracker.touch_with_priority(1, 0.9); // High priority, should be last
        tracker.touch_with_priority(2, 0.1); // Low priority, should be first
        tracker.touch_with_priority(3, 0.5); // Medium priority

        let order = tracker.eviction_order();
        // Lowest priority first
        assert_eq!(order[0], 2);
        assert_eq!(order[1], 3);
        assert_eq!(order[2], 1);
    }

    #[test]
    fn test_lru_pop() {
        let mut tracker = LruTracker::new();

        tracker.touch_with_priority(1, 0.9);
        tracker.touch_with_priority(2, 0.1);

        let evicted = tracker.pop_lru();
        assert_eq!(evicted, Some(2)); // Lowest priority
        assert!(!tracker.contains(&2));
        assert!(tracker.contains(&1));
    }

    #[test]
    fn test_eviction_candidates_for_size() {
        let mut tracker = LruTracker::new();

        tracker.touch_with_priority(1, 0.1);
        tracker.touch_with_priority(2, 0.2);
        tracker.touch_with_priority(3, 0.3);

        // Each item is 100 bytes, need 250 bytes -> 3 items
        let candidates = tracker.eviction_candidates_for_size(250, |_| 100);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates, vec![1, 2, 3]); // Priority order
    }
}
