# Agent Collaboration Specification

**Version:** 0.2.0
**Status:** Draft
**Date:** 2026-02-14
**Crate:** `conclave`

---

## 1. Purpose

This specification defines the **Conclave** — the room-based multi-agent
collaboration system. It provides the infrastructure for humans and AI agents
to communicate within shared workspaces.

### 1.1 Scope

**In scope:**
- Room lifecycle (create, archive, fork)
- Participant management (humans, agents)
- Channel architecture (Main, DM, AgentReasoning, Thread)
- Backend session management (Claude Code, Infernum, HTTP)
- Event processing and routing
- Turn coordination
- Attention state management
- Persistence and recovery

**Out of scope:**
- Individual agent behavior (see AGENTIC-LOOP-SPEC.md)
- Supervisor orchestration (see MULTI-AGENT-SUPERVISOR-SPEC.md)
- Tool definitions and execution (see TOOL-CALLING-SPEC.md)

---

## 2. Design Principles

**Principle 1: Rooms are Workspaces**

A room represents a bounded collaboration context with a working directory,
participants, and shared message history. Rooms are the unit of coordination.

**Principle 2: Channels Separate Concerns**

Different communication needs use different channels:
- Main channel for coordinated discussion
- DMs for private agent-human conversations
- AgentReasoning for internal thought processes
- Threads for focused sub-discussions

**Principle 3: Turn-Taking Enables Coordination**

The turn system prevents agents from talking over each other while allowing
natural conversation flow. Priority-based queuing respects urgency and
human preference.

**Principle 4: Backends are Swappable**

Agent backends (Claude Code, Infernum, custom HTTP) implement a common trait.
The collaboration system is backend-agnostic.

---

## 3. Architecture

### 3.1 Component Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                           RoomRegistry                                  │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │    Rooms    │  │  Messages   │  │  Sessions   │  │    Events    │ │
│  │ HashMap<Id> │  │ HashMap<Id> │  │ HashMap<Id> │  │   Channel    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
└─────────┼────────────────┼───────────────┼─────────────────┼─────────┘
          │                │               │                 │
          ▼                ▼               ▼                 ▼
     ┌────────┐      ┌──────────┐   ┌───────────┐     ┌───────────┐
     │  Room  │      │ Messages │   │  Backend  │     │ Subscribers│
     │ struct │      │ Vec<Msg> │   │  Session  │     │            │
     └────────┘      └──────────┘   └───────────┘     └───────────┘
```

### 3.2 Core Types

```rust
/// Unique room identifier.
pub struct RoomId(Uuid);

/// Unique participant identifier.
pub struct ParticipantId(Uuid);

/// A collaboration room.
pub struct Room {
    pub id: RoomId,
    pub name: String,
    pub working_dir: PathBuf,
    pub participants: Vec<Participant>,
    pub creator: UserId,
    pub archived: bool,
    pub turn_coordination: TurnCoordinator,
    pub attention: AttentionManager,
    // ...
}

/// A participant in a room.
pub struct Participant {
    pub id: ParticipantId,
    pub display_name: String,
    pub kind: ParticipantKind,
    pub attention: AttentionState,
    // ...
}

/// Participant type (human or agent).
pub enum ParticipantKind {
    Human { user_id: UserId },
    Agent { backend: AgentBackend, session_id: String, spawned_by: ParticipantId },
}
```

---

## 4. Room Lifecycle

### 4.1 States

```
┌──────────┐    create()    ┌────────┐    archive()    ┌──────────┐
│          │ ─────────────► │ Active │ ────────────────► │ Archived │
│  (none)  │                │        │                   │          │
│          │                │        │    fork()         │          │
└──────────┘                └────────┘ ──────────┐       └──────────┘
                                 ▲                │
                                 │                │
                                 └────────────────┘
```

### 4.2 Create Room

**Preconditions:**
- Working directory exists
- Max room limit not exceeded

**Postconditions:**
- Room created with unique ID
- Creator added as human participant
- RoomCreated event emitted

### 4.3 Archive Room

**Preconditions:**
- Room exists and is active

**Postconditions:**
- Room marked as archived
- All agent sessions terminated
- RoomArchived event emitted
- No new messages or participants allowed

### 4.4 Fork Room

**Preconditions:**
- Source room exists

**Postconditions:**
- New room created with copied configuration
- Message history NOT copied (fresh start)
- Participants NOT copied (fresh start)

---

## 5. Participant Lifecycle

### 5.1 Human Participants

```rust
// Join a room
registry.join_room(room_id, user_id, display_name).await?;

// Leave a room
registry.leave_room(room_id, participant_id, reason).await?;
```

### 5.2 Agent Participants

```rust
// Spawn an agent (no event processing)
let agent_id = registry.spawn_agent(room_id, config, spawned_by).await?;

// Spawn with event processing (preferred)
let (agent_id, handle) = registry
    .spawn_agent_with_events(room_id, config, spawned_by)
    .await?;

// Terminate an agent
registry.terminate_agent(room_id, agent_id, reason).await?;
```

### 5.3 Agent Configuration

```rust
pub struct AgentConfig {
    pub backend: AgentBackend,
    pub display_name: Option<String>,
    pub persona: Option<String>,
}

pub enum AgentBackend {
    ClaudeCode { tier: Option<String> },
    Infernum { model: String },
    Codex,
    Cursor,
    Custom { endpoint: String, protocol: AgentProtocol },
}
```

---

## 6. Channel Architecture

### 6.1 Channel Types

| Channel | Purpose | Visibility | Turn-Coordinated |
|---------|---------|------------|------------------|
| Main | Primary discussion | All participants | Yes |
| DirectMessage | Private 1:1 or group | Specified participants | No |
| AgentReasoning | Internal thought process | Single agent | No |
| Thread | Sub-discussion | Thread participants | No |

### 6.2 Message Content

```rust
pub enum MessageContent {
    /// Plain text message.
    Text { content: String },

    /// Tool invocation.
    ToolCall { tool: String, input: Value, call_id: String },

    /// Tool result.
    ToolResult { tool: String, output: Value, call_id: String, success: bool },

    /// System event.
    System { event: SystemEvent },
}
```

### 6.3 Channel Identity

DirectMessage channels use sorted participant lists for consistent identity:
- `DM([alice, bob])` equals `DM([bob, alice])`

---

## 7. Backend Sessions

### 7.1 Session Trait

```rust
#[async_trait]
pub trait AgentBackendSession: Send + Sync {
    /// Returns the unique session ID.
    fn session_id(&self) -> &str;

    /// Returns the backend configuration.
    fn backend(&self) -> &AgentBackend;

    /// Sends a message to the agent.
    async fn send_message(&self, message: &Message) -> Result<()>;

    /// Interrupts the current operation.
    async fn interrupt(&self) -> Result<()>;

    /// Terminates the session.
    async fn terminate(&self) -> Result<()>;

    /// Checks if the session is still running.
    fn is_running(&self) -> bool;

    /// Takes the event receiver (can only be called once).
    fn take_event_receiver(&self) -> Option<mpsc::Receiver<AgentEvent>>;
}
```

### 7.2 Backend Implementations

| Backend | Process Type | Communication | Streaming |
|---------|--------------|---------------|-----------|
| ClaudeCode | External CLI | stdin/stdout JSON | Yes |
| Infernum | External CLI | stdin/stdout JSON | Yes |
| HTTP | HTTP client | REST/SSE | Yes |

### 7.3 Agent Events

```rust
pub enum AgentEvent {
    /// Agent sent a message.
    Message { content: String, mentions: Vec<ParticipantId> },

    /// Agent invoked a tool.
    ToolCall { tool: String, input: Value, call_id: String },

    /// Tool execution completed.
    ToolResult { tool: String, call_id: String, output: String, success: bool, duration_ms: u32 },

    /// Agent is thinking.
    Thinking { content: String },

    /// Attention state changed.
    AttentionChanged { new_state: AttentionState },

    /// Turn requested.
    TurnRequested { reason: Option<String>, priority: TurnPriority },

    /// Turn yielded.
    TurnYielded,

    /// Session terminated.
    Terminated { reason: TerminationReason },

    /// Error occurred.
    Error { message: String, recoverable: bool },
}
```

---

## 8. Event Processing

### 8.1 Per-Session Processing

Each agent session has a dedicated `SessionEventProcessor` that:
1. Receives events from the backend session's channel
2. Routes events through `handle_agent_event`
3. Converts events to messages in appropriate channels
4. Handles termination cleanup

```rust
// Recommended: spawns agent and starts event processing
let (agent_id, handle) = registry
    .spawn_agent_with_events(room_id, config, spawned_by)
    .await?;

// Wait for processor to finish (optional)
handle.await?;
```

### 8.2 Global Event Loop

The global event loop handles system-wide concerns:
- Session health monitoring
- Dead session cleanup
- System-wide events

```rust
let registry = Arc::new(RoomRegistry::with_defaults());
let loop_handle = start_event_loop(Arc::clone(&registry));

// ... use registry ...

loop_handle.shutdown().await;
```

---

## 9. Turn Coordination

### 9.1 Turn States

| State | Description |
|-------|-------------|
| NoSpeaker | No one has the floor |
| Speaking(id) | Participant `id` has the floor |

### 9.2 Priority Levels

| Priority | Description | Human Boost |
|----------|-------------|-------------|
| Low | Background task | +0 |
| Normal | Standard request | +1 |
| High | Important update | +2 |
| Urgent | Critical issue | +3 |

### 9.3 Turn Operations

```rust
// Request the turn
let position = registry
    .request_turn(room_id, participant_id, priority, reason)
    .await?;

// Yield the turn
registry.yield_turn(room_id, participant_id).await?;

// Check current speaker
let speaker = registry.current_speaker(room_id).await?;
```

---

## 10. Attention Management

### 10.1 Attention States

| State | Can Interrupt | Auto-Focus on Mention |
|-------|---------------|----------------------|
| Available | Yes | Yes |
| Focused { interruptible } | Depends | No |
| Away | No | No |
| DoNotDisturb | No | No |

### 10.2 Focus Decay

When an agent becomes focused, a decay timer starts. After the decay period
(default 5 minutes), attention returns to Available unless activity occurs.

```rust
// Set attention state
registry.set_attention(room_id, participant_id, state).await?;

// Human can override agent attention
registry.set_attention_override(room_id, agent_id, state, overrider_id).await?;
```

---

## 11. Message Routing

### 11.1 Routing Rules

| Message Type | Route To |
|--------------|----------|
| Human message in Main | All agent backends |
| Agent message in Main | Event subscribers only |
| DM to agent | That agent's backend only |
| Tool call | Tool executor |
| Tool result | Originating agent |

### 11.2 Routing Implementation

```rust
// Human sends to main channel
registry.send_main_channel_message(room_id, human_id, content).await?;

// Send DM
registry.send_dm(room_id, sender_id, recipients, content).await?;

// Handle agent event (routes automatically)
registry.handle_agent_event(room_id, agent_id, event).await?;
```

---

## 12. Persistence

### 12.1 Storage Structure

```
~/.local/share/conclave/
├── rooms/
│   ├── {room_id}.json          # Room snapshot
│   └── ...
└── messages/
    ├── {room_id}.jsonl         # Message log (append-only)
    └── ...
```

### 12.2 Operations

```rust
// Initialize store
let store = PersistenceStore::new(config);
store.initialize().await?;

// Persist registry state
registry.persist(&store).await?;

// Restore registry from disk
let registry = RoomRegistry::restore(&store).await?;
```

---

## 13. Recovery

### 13.1 Agent Reconnection

When an agent session is lost but the participant record remains:

```rust
// Check for disconnected agents
let disconnected = registry.get_disconnected_agents(room_id).await?;

// Recover a specific agent
registry.recover_agent(room_id, agent_id, spawned_by).await?;

// Recover all disconnected agents
let count = registry.recover_all_agents(room_id, spawned_by).await?;
```

### 13.2 Recovery Status

```rust
pub struct AgentRecoveryStatus {
    pub agent_id: ParticipantId,
    pub display_name: String,
    pub backend: AgentBackend,
    pub connected: bool,
    pub spawned_by: ParticipantId,
}
```

---

## 14. Error Handling

### 14.1 Error Categories

| Category | Recoverable | Action |
|----------|-------------|--------|
| RoomNotFound | No | Return error |
| NotInRoom | No | Return error |
| RoomArchived | No | Return error |
| SpawnFailed | Yes | Retry with backoff |
| BackendTerminated | Yes | Reconnect |
| CommunicationFailed | Yes | Retry |

### 14.2 Error Type

```rust
#[derive(Debug, Error)]
pub enum ConclaveError {
    #[error("room {0} not found")]
    RoomNotFound(RoomId),

    #[error("participant {0} not in room {1}")]
    NotInRoom(ParticipantId, RoomId),

    #[error("room {0} is archived")]
    RoomArchived(RoomId),

    #[error("failed to spawn agent: {0}")]
    SpawnFailed(String),

    #[error("backend session {session_id} terminated")]
    BackendTerminated { session_id: String },

    // ...
}
```

---

## 15. Configuration

### 15.1 Room Policies

```rust
pub struct RoomPolicies {
    pub max_participants: usize,        // Default: 32
    pub max_agents: usize,              // Default: 8
    pub max_message_length: usize,      // Default: 100_000
    pub turn_timeout_secs: u64,         // Default: 300
}
```

### 15.2 Attention Configuration

```rust
pub struct AttentionConfig {
    pub focus_decay_minutes: u64,       // Default: 5
    pub mention_escalates_to_focused: bool,  // Default: true
}
```

---

## 16. Future Work

The following features are planned but not yet implemented:

- **Tool Lock Manager**: Exclusive tool access for coordinated multi-agent work
- **Resource Quota Manager**: Shared budgets for API calls, tokens, etc.
- **Cross-Room Messaging**: Agent communication between rooms
- **Supervisor Integration**: Connection to MULTI-AGENT-SUPERVISOR-SPEC
- **Streaming Responses**: Progressive message rendering during generation

---

## Appendix A: Events Reference

### Room Events

| Event | Payload | When Emitted |
|-------|---------|--------------|
| RoomCreated | room_id, creator | Room created |
| RoomArchived | room_id | Room archived |
| ParticipantJoined | room_id, participant | Join/spawn |
| ParticipantLeft | room_id, participant_id, reason | Leave/terminate |
| MessageSent | room_id, message | Message sent |
| AttentionChanged | room_id, participant_id, old, new | Attention update |
| TurnChanged | room_id, old_speaker, new_speaker | Turn change |

### System Events (in messages)

| Event | Payload | When Emitted |
|-------|---------|--------------|
| ParticipantJoined | participant_id, display_name | Join/spawn |
| ParticipantLeft | participant_id, reason | Leave/terminate |
| TurnGranted | participant_id | Turn acquired |
| TurnYielded | participant_id | Turn released |

---

## Appendix B: Related Specifications

- **AGENTIC-LOOP-SPEC.md**: Individual agent behavior and coordination primitives
- **MULTI-AGENT-SUPERVISOR-SPEC.md**: Supervisor orchestration of multiple agents
- **TOOL-CALLING-SPEC.md**: Tool definition and execution protocols
