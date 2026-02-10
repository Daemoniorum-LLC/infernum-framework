//! Autonomy enforcement for the agentic loop.
//!
//! Implements the permission system from AGENTIC-LOOP-SPEC.md §1.1 Principle 5.
//! Agents know their constraints upfront — they don't discover them by hitting walls.

use super::types::{AutonomyGrant, Permission, ToolPattern};

impl AutonomyGrant {
    /// Creates a builder for constructing an autonomy grant.
    pub fn builder() -> AutonomyGrantBuilder {
        AutonomyGrantBuilder::default()
    }

    /// Checks what permission level a tool call has.
    ///
    /// Priority: forbidden > require_approval > auto_approve > default (require_approval).
    ///
    /// `require_approval` beats `auto_approve` so that grant narrowing works:
    /// a client can add require_approval patterns that override server auto_approve.
    pub fn check(&self, tool_name: &str, argument: Option<&str>) -> Permission {
        // Forbidden takes highest priority
        if self.matches_any(&self.forbidden, tool_name, argument) {
            return Permission::Forbidden;
        }

        // Require-approval beats auto-approve (enables narrowing)
        if self.matches_any(&self.require_approval, tool_name, argument) {
            return Permission::RequiresApproval;
        }

        // Then check auto-approve
        if self.matches_any(&self.auto_approve, tool_name, argument) {
            return Permission::Allowed;
        }

        // Default: require approval for anything not explicitly covered
        Permission::RequiresApproval
    }

    /// Convenience: is this tool call allowed without approval?
    pub fn is_allowed(&self, tool_name: &str, argument: Option<&str>) -> bool {
        self.check(tool_name, argument) == Permission::Allowed
    }

    /// Convenience: is this tool call forbidden?
    pub fn is_forbidden(&self, tool_name: &str, argument: Option<&str>) -> bool {
        self.check(tool_name, argument) == Permission::Forbidden
    }

    /// Convenience: does this tool call require approval?
    pub fn requires_approval(&self, tool_name: &str, argument: Option<&str>) -> bool {
        self.check(tool_name, argument) == Permission::RequiresApproval
    }

    /// Narrows this grant with a client-supplied grant.
    ///
    /// The client can only restrict, never widen:
    /// - Server's forbidden patterns are always preserved
    /// - Client can add more forbidden patterns
    /// - Client can move auto_approve to require_approval
    /// - Client cannot move forbidden to allowed
    pub fn narrow_with(&self, client: &AutonomyGrant) -> AutonomyGrant {
        let mut result = self.clone();

        // Client's forbidden always adds
        for pattern in &client.forbidden {
            if !result.forbidden.iter().any(|p| p.same_as(pattern)) {
                result.forbidden.push(pattern.clone());
            }
        }

        // Client's require_approval narrows auto_approve
        // (items in client's require_approval that were in our auto_approve move to require_approval)
        for pattern in &client.require_approval {
            if !result.require_approval.iter().any(|p| p.same_as(pattern)) {
                result.require_approval.push(pattern.clone());
            }
        }

        result
    }

    /// Returns all forbidden patterns.
    pub fn forbidden_patterns(&self) -> &[ToolPattern] {
        &self.forbidden
    }

    /// Returns all auto-approved patterns.
    pub fn allowed_patterns(&self) -> &[ToolPattern] {
        &self.auto_approve
    }

    fn matches_any(
        &self,
        patterns: &[ToolPattern],
        tool_name: &str,
        argument: Option<&str>,
    ) -> bool {
        patterns.iter().any(|p| p.matches(tool_name, argument))
    }
}

impl ToolPattern {
    /// Checks if this pattern matches a tool call.
    pub fn matches(&self, tool_name: &str, argument: Option<&str>) -> bool {
        match self {
            Self::Read(glob) => {
                matches!(tool_name, "read_file" | "read")
                    && argument.map_or(false, |a| glob_matches(glob, a))
            },
            Self::Write(glob) => {
                matches!(tool_name, "write_file" | "edit_file" | "write" | "edit")
                    && argument.map_or(false, |a| glob_matches(glob, a))
            },
            Self::Bash(glob) => {
                matches!(tool_name, "bash" | "shell")
                    && argument.map_or(false, |a| glob_matches(glob, a))
            },
            Self::Tool(name) => glob_matches(name, tool_name),
        }
    }

    /// Check if two patterns represent the same rule (structural equality).
    pub fn same_as(&self, other: &ToolPattern) -> bool {
        match (self, other) {
            (Self::Read(a), Self::Read(b)) => a == b,
            (Self::Write(a), Self::Write(b)) => a == b,
            (Self::Bash(a), Self::Bash(b)) => a == b,
            (Self::Tool(a), Self::Tool(b)) => a == b,
            _ => false,
        }
    }
}

/// Simple glob-style matching.
///
/// Supports:
/// - `*` matches any sequence of non-separator characters
/// - `**` matches any sequence of characters including separators
/// - Literal matching otherwise
fn glob_matches(pattern: &str, text: &str) -> bool {
    if pattern == "**/*" || pattern == "*" {
        return true;
    }

    // Exact match
    if !pattern.contains('*') {
        return pattern == text;
    }

    // Split on `**` first for recursive matching
    if pattern.contains("**") {
        return glob_matches_recursive(pattern, text);
    }

    // Simple `*` glob: split on `*` and check segments match in order
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }

    let mut pos = 0;

    // First part must match at start (unless pattern starts with *)
    if !pattern.starts_with('*') {
        if !text.starts_with(parts[0]) {
            return false;
        }
        pos = parts[0].len();
    }

    // Middle and last parts must appear in order
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 && !pattern.starts_with('*') {
            continue; // already handled
        }
        if let Some(idx) = text[pos..].find(part) {
            pos += idx + part.len();
        } else {
            return false;
        }
    }

    // If pattern ends without *, text must be fully consumed
    if !pattern.ends_with('*') {
        if let Some(last) = parts.last() {
            if !last.is_empty() {
                return text.ends_with(last);
            }
        }
    }

    true
}

/// Recursive glob matching for `**` patterns.
fn glob_matches_recursive(pattern: &str, text: &str) -> bool {
    // Simple approach: `**` matches everything
    let parts: Vec<&str> = pattern.split("**").collect();

    if parts.len() == 1 {
        // No ** found, fall back to simple glob
        return glob_matches(parts[0], text);
    }

    // Check prefix (before first **)
    let prefix = parts[0].trim_end_matches('/');
    if !prefix.is_empty() && !text.starts_with(prefix) {
        return false;
    }

    // Check suffix (after last **)
    let suffix = parts.last().unwrap_or(&"").trim_start_matches('/');
    if !suffix.is_empty() {
        // suffix may contain simple globs
        if suffix.contains('*') {
            // Recursive: check the suffix as a glob against the tail of text
            // For simplicity, just check if any tail matches
            for i in 0..=text.len() {
                if glob_matches(suffix, &text[i..]) {
                    return true;
                }
            }
            return false;
        }
        return text.ends_with(suffix);
    }

    true
}

/// Builder for `AutonomyGrant`.
#[derive(Debug, Default)]
pub struct AutonomyGrantBuilder {
    auto_approve: Vec<ToolPattern>,
    require_approval: Vec<ToolPattern>,
    forbidden: Vec<ToolPattern>,
}

impl AutonomyGrantBuilder {
    /// Add an auto-approved pattern.
    pub fn allow(mut self, pattern: ToolPattern) -> Self {
        self.auto_approve.push(pattern);
        self
    }

    /// Add a require-approval pattern.
    pub fn require_approval(mut self, pattern: ToolPattern) -> Self {
        self.require_approval.push(pattern);
        self
    }

    /// Add a forbidden pattern.
    pub fn forbid(mut self, pattern: ToolPattern) -> Self {
        self.forbidden.push(pattern);
        self
    }

    /// Build the autonomy grant.
    pub fn build(self) -> AutonomyGrant {
        AutonomyGrant {
            auto_approve: self.auto_approve,
            require_approval: self.require_approval,
            forbidden: self.forbidden,
        }
    }
}
