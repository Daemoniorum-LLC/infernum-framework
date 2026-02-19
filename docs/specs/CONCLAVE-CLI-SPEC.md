# Conclave CLI Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-15
**Crate:** `infernum`
**Dependencies:** `conclave`, `ratatui`, `crossterm`

---

## 1. Overview

This specification defines the CLI interface for Conclave multi-agent collaboration rooms,
enabling users to create rooms, spawn agents, send messages, and observe collaboration in real-time.

### 1.1 Problem Statement

The `conclave` crate provides a comprehensive library for multi-agent collaboration:
- RoomRegistry for room management
- Agent spawning (Claude Code, Infernum backends)
- Turn coordination and message routing
- Event subscription system

However, there is **no user-facing interface**. Users cannot:
- Create rooms from the command line
- Spawn agents interactively
- Observe agent collaboration in real-time
- Send messages to guide agent work

### 1.2 Goal

Create `infernum room` subcommands that:

1. Manage collaboration rooms via CLI
2. Spawn Claude Code or Infernum agents
3. Send messages as a human participant
4. Observe collaboration in a terminal UI (TUI)
5. Enable dogfooding: use agents to build web observer

### 1.3 Design Principles

> **CLI-first, GUI later.** Build the foundation that web/desktop can call.

> **Daemon for persistence.** Rooms and agents must outlive CLI invocations.

> **Unix socket IPC.** Simple, fast, secure local communication.

> **TUI for observation.** Real-time visibility into agent work.

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Conclave CLI System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│   │  infernum    │         │  infernum    │         │  infernum    │   │
│   │  room create │         │  room spawn  │         │  room observe│   │
│   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘   │
│          │                        │                        │           │
│          └────────────────────────┼────────────────────────┘           │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌───────────────────────────┐                       │
│                    │      Unix Socket IPC      │                       │
│                    │  ~/.config/infernum/      │                       │
│                    │       room.sock           │                       │
│                    └─────────────┬─────────────┘                       │
│                                  │                                      │
│                                  ▼                                      │
│                    ┌───────────────────────────┐                       │
│                    │       Room Daemon         │                       │
│                    │                           │                       │
│                    │  ┌─────────────────────┐  │                       │
│                    │  │   RoomRegistry      │  │                       │
│                    │  │   (from conclave)   │  │                       │
│                    │  └─────────┬───────────┘  │                       │
│                    │            │              │                       │
│                    │  ┌─────────┴───────────┐  │                       │
│                    │  │                     │  │                       │
│                    │  │   Room 1   Room 2   │  │                       │
│                    │  │    ▼         ▼      │  │                       │
│                    │  │  Agents   Agents    │  │                       │
│                    │  │                     │  │                       │
│                    │  └─────────────────────┘  │                       │
│                    │                           │                       │
│                    │  Event broadcast ─────────┼──▶ TUI Observers      │
│                    │                           │                       │
│                    └───────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `main.rs` | CLI parsing, command dispatch |
| `room_daemon.rs` | Daemon server, hosts RoomRegistry |
| `room_client.rs` | Unix socket client for CLI |
| `commands.rs` | Command implementations |
| `tui.rs` | Terminal UI observer |

---

## 3. Type Architecture

### 3.1 Daemon Protocol

```
DaemonRequest:
    | CreateRoom { name: String, working_dir: PathBuf }
    | ListRooms
    | GetRoom { room_id: RoomId }
    | SpawnAgent { room_id: RoomId, config: AgentConfig }
    | SendMessage { room_id: RoomId, sender: ParticipantId, content: String }
    | Subscribe { room_id: RoomId }
    | ArchiveRoom { room_id: RoomId }

DaemonResponse:
    | RoomCreated { room_id: RoomId }
    | Rooms { rooms: [RoomSummary] }
    | Room { room: RoomSnapshot }
    | AgentSpawned { participant_id: ParticipantId }
    | MessageSent { message_id: MessageId }
    | Event { event: RoomEvent }
    | Error { message: String }
```

### 3.2 Agent Configuration

```
parse_agent_spec(spec: String) → AgentConfig:
    match spec:
        "claude-opus"   → AgentConfig.claude_code(ClaudeTier.Opus)
        "claude-sonnet" → AgentConfig.claude_code(ClaudeTier.Sonnet)
        "claude-haiku"  → AgentConfig.claude_code(ClaudeTier.Haiku)
        "infernum:<model>" → AgentConfig.infernum(model)
        _ → Error("Unknown agent type")
```

### 3.3 Room ID Resolution

```
resolve_room_id(prefix: String, rooms: [Room]) → Result<RoomId>:
    matches ← rooms.filter(r → r.id.starts_with(prefix))
    match len(matches):
        0 → Error("No room matches prefix")
        1 → Ok(matches[0].id)
        _ → Error("Ambiguous: multiple rooms match prefix")
```

---

## 4. Behavioral Contracts

### 4.1 Daemon Lifecycle

**Startup:**
```
room_daemon():
    socket_path ← ~/.config/infernum/room.sock
    ensure_parent_dir(socket_path)

    if exists(socket_path):
        if is_daemon_alive():
            return Error("Daemon already running")
        else:
            remove(socket_path)

    registry ← RoomRegistry.with_defaults()
    listener ← UnixListener.bind(socket_path)

    loop:
        accept_and_handle(listener, registry)
```

**Shutdown:**
```
room_daemon_stop():
    client ← connect_daemon()
    client.send(Shutdown)
    wait_for_socket_removal()
```

### 4.2 Room Creation

```
room_create(name: String, working_dir: String):
    client ← connect_daemon()
    request ← CreateRoom { name, working_dir }
    response ← client.send(request)

    match response:
        RoomCreated { room_id } → print("Room created: {}", room_id)
        Error { message } → return Error(message)
```

### 4.3 Agent Spawning

```
room_spawn(room_id: String, agent: String, name: Option<String>):
    client ← connect_daemon()
    resolved_id ← resolve_room_id(room_id, client.list_rooms())
    config ← parse_agent_spec(agent)

    if name.is_some():
        config.display_name ← name

    response ← client.spawn_agent(resolved_id, config)

    match response:
        AgentSpawned { participant_id } → print("Agent spawned: {}", participant_id)
        Error { message } → return Error(message)
```

### 4.4 TUI Observer

```
room_observe(room_id: String):
    client ← connect_daemon()
    resolved_id ← resolve_room_id(room_id)

    events_rx ← client.subscribe(resolved_id)
    room ← client.get_room(resolved_id)

    terminal ← setup_terminal()
    state ← TuiState { room, messages: [], participants: [] }

    loop:
        draw(terminal, state)

        select:
            key ← poll_keyboard():
                match key:
                    Ctrl-C | 'q' → break
                    Enter → send_message_from_input()
                    char → append_to_input()

            event ← events_rx.recv():
                update_state(state, event)
```

---

## 5. Constraints & Invariants

### 5.1 Socket Invariants

```
P1: socket_path = ~/.config/infernum/room.sock
    // Fixed location for all CLI invocations

P2: ∀ t: daemon_running ⟹ socket_exists
    // Socket exists iff daemon is alive

P3: socket permissions = 0o600
    // Only owner can access
```

### 5.2 Room Invariants

```
P4: ∀ room ∈ registry:
    room.state ∈ {Active, Archived}

P5: ∀ agent ∈ room:
    agent.backend ∈ {ClaudeCode, Infernum}

P6: room_id.is_unique()
    // Generated with UUID
```

### 5.3 Event Ordering

```
P7: ∀ subscriber s, events e1 < e2:
    s.received(e1) before s.received(e2)
    // Events delivered in order
```

---

## 6. CLI Commands

### 6.1 Command Reference

| Command | Description | Status |
|---------|-------------|--------|
| `infernum room daemon` | Start daemon (foreground) | 🔮 |
| `infernum room daemon-start` | Start daemon (background) | 🔮 |
| `infernum room daemon-stop` | Stop daemon | 🔮 |
| `infernum room create <name>` | Create room | 🔮 |
| `infernum room list` | List active rooms | 🔮 |
| `infernum room info <id>` | Show room details | 🔮 |
| `infernum room spawn <id> -a <agent>` | Spawn agent | 🔮 |
| `infernum room send <id> <msg>` | Send message | 🔮 |
| `infernum room observe <id>` | Watch in TUI | 🔮 |
| `infernum room archive <id>` | Archive room | 🔮 |

### 6.2 Usage Examples

```bash
# Start the daemon
infernum room daemon

# In another terminal:
infernum room create "Fix auth bug" --working-dir ~/project
# Output: Room created: room_abc123

infernum room spawn room_abc --agent claude-opus --name "Claude"
# Output: Agent spawned: part_def456

infernum room send room_abc "Please investigate the login failure"

infernum room observe room_abc
# Opens TUI

infernum room archive room_abc
```

---

## 7. TUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Room: Fix auth bug                                   [room_abc] │
├─────────────────────────────────────────┬───────────────────────┤
│ Messages                                │ Participants          │
│                                         │                       │
│ [10:32] Alice: Can you fix the login?  │ Alice (human)         │
│ [10:32] Claude: I'll investigate...    │ Claude (opus)         │
│                                         │   └─ Reading...       │
│ [Tool] Read: src/auth.rs               │                       │
│ [Tool] Edit: src/auth.rs               │                       │
│                                         │                       │
├─────────────────────────────────────────┴───────────────────────┤
│ > Type message (Ctrl+C to exit)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Error Conditions

| Condition | Error Message |
|-----------|---------------|
| Daemon not running | "Room daemon not running. Start with: infernum room daemon" |
| Room not found | "No room matches ID prefix: {prefix}" |
| Ambiguous room ID | "Multiple rooms match prefix. Be more specific." |
| Agent spawn failed | "Failed to spawn agent: {reason}" |
| Socket permission denied | "Cannot connect to daemon: permission denied" |

---

## 9. Integration Points

### 9.1 With Conclave Crate

- Uses `RoomRegistry` for room management
- Uses `AgentConfig::claude_code()` and `AgentConfig::infernum()`
- Subscribes to `RoomEvent` via `registry.subscribe()`
- Calls `spawn_agent()` to start agent sessions

### 9.2 With TUI Libraries

- `ratatui` for terminal rendering
- `crossterm` for terminal input/control
- Event loop with `tokio::select!` for async handling

---

## 10. Open Questions

1. **Daemon supervision:** Should we use systemd/launchd for production?
   - Pro: Proper service management
   - Con: More complex setup

2. **Multi-user access:** Should socket support multiple users?
   - Current: Single-user with 0o600 permissions
   - Future: Could add auth for shared daemon

3. **Room persistence:** Should rooms survive daemon restart?
   - Current: In-memory only
   - Future: Could serialize to disk

4. **Web observer:** How to expose events over WebSocket?
   - Planned: `infernum-server` endpoint for web clients

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-15 | Initial draft |
