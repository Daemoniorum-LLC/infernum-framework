//! TUI observer for collaboration rooms.
//!
//! Implements CONCLAVE-CLI-SPEC.md §7 TUI Layout.

use std::io::{self, Stdout};
use std::time::Duration;

use color_eyre::eyre::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::room_client::DaemonClient;
use crate::room_daemon::{ParticipantInfo, RoomSnapshot};

// =============================================================================
// TUI State
// =============================================================================

/// State for the TUI observer.
pub struct TuiState {
    /// Room snapshot.
    pub room: RoomSnapshot,
    /// Message history.
    pub messages: Vec<TuiMessage>,
    /// Current input.
    pub input: String,
    /// Scroll offset for messages.
    pub scroll: usize,
    /// Whether we should quit.
    pub should_quit: bool,
}

/// A message in the TUI.
#[derive(Debug, Clone)]
pub struct TuiMessage {
    pub timestamp: String,
    pub sender: String,
    pub content: String,
    pub is_tool: bool,
    pub is_system: bool,
}

impl TuiState {
    /// Creates a new TUI state from a room snapshot.
    pub fn new(room: RoomSnapshot) -> Self {
        Self {
            room,
            messages: Vec::new(),
            input: String::new(),
            scroll: 0,
            should_quit: false,
        }
    }

    /// Adds a message to the history.
    pub fn add_message(&mut self, msg: TuiMessage) {
        self.messages.push(msg);
        // Auto-scroll to bottom
        if self.messages.len() > 20 {
            self.scroll = self.messages.len() - 20;
        }
    }

    /// Adds an event string (from daemon) as a message.
    pub fn add_event(&mut self, event: &str) {
        let now = chrono::Local::now().format("%H:%M").to_string();

        // Parse event to extract meaningful info
        let (sender, content, is_tool, is_system) = if event.contains("MessageSent") {
            ("Agent".to_string(), event.to_string(), false, false)
        } else if event.contains("Tool") {
            ("System".to_string(), event.to_string(), true, false)
        } else if event.contains("Participant") {
            ("System".to_string(), event.to_string(), false, true)
        } else {
            ("Event".to_string(), event.to_string(), false, true)
        };

        self.add_message(TuiMessage {
            timestamp: now,
            sender,
            content,
            is_tool,
            is_system,
        });
    }
}

// =============================================================================
// TUI Runner
// =============================================================================

/// Runs the TUI observer.
pub async fn run_tui(mut client: DaemonClient, room: RoomSnapshot) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create state
    let mut state = TuiState::new(room);

    // Add initial system message
    state.add_message(TuiMessage {
        timestamp: chrono::Local::now().format("%H:%M").to_string(),
        sender: "System".to_string(),
        content: format!("Connected to room: {}", state.room.name),
        is_tool: false,
        is_system: true,
    });

    // Main loop
    let result = run_event_loop(&mut terminal, &mut state, &mut client).await;

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Main event loop.
async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut TuiState,
    _client: &mut DaemonClient,
) -> Result<()> {
    loop {
        // Draw UI
        terminal.draw(|f| draw_ui(f, state))?;

        // Poll for events with timeout
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            state.should_quit = true;
                        }
                        KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            state.should_quit = true;
                        }
                        KeyCode::Esc => {
                            state.should_quit = true;
                        }
                        KeyCode::Enter => {
                            if !state.input.is_empty() {
                                let content = state.input.clone();
                                state.input.clear();

                                // Add message locally
                                state.add_message(TuiMessage {
                                    timestamp: chrono::Local::now().format("%H:%M").to_string(),
                                    sender: "You".to_string(),
                                    content: content.clone(),
                                    is_tool: false,
                                    is_system: false,
                                });

                                // TODO: Send to daemon when message sending is wired up
                                // client.send_message(state.room.id.clone(), content).await?;
                            }
                        }
                        KeyCode::Backspace => {
                            state.input.pop();
                        }
                        KeyCode::Char(c) => {
                            state.input.push(c);
                        }
                        KeyCode::Up => {
                            if state.scroll > 0 {
                                state.scroll -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if state.scroll < state.messages.len().saturating_sub(1) {
                                state.scroll += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Check for daemon events (non-blocking)
        // Note: This is a simplified version - real implementation would use
        // async select! with the event stream
        // For now, we just check occasionally

        if state.should_quit {
            break;
        }
    }

    Ok(())
}

// =============================================================================
// UI Drawing
// =============================================================================

/// Draws the UI.
fn draw_ui(f: &mut Frame, state: &TuiState) {
    let size = f.area();

    // Main layout: header, body, footer
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Body
            Constraint::Length(3), // Input
        ])
        .split(size);

    // Draw header
    draw_header(f, main_chunks[0], state);

    // Body layout: messages (left), participants (right)
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70), // Messages
            Constraint::Percentage(30), // Participants
        ])
        .split(main_chunks[1]);

    // Draw messages
    draw_messages(f, body_chunks[0], state);

    // Draw participants
    draw_participants(f, body_chunks[1], state);

    // Draw input
    draw_input(f, main_chunks[2], state);
}

/// Draws the header.
fn draw_header(f: &mut Frame, area: Rect, state: &TuiState) {
    let short_id = if state.room.id.len() > 8 {
        &state.room.id[..8]
    } else {
        &state.room.id
    };

    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled("Room: ", Style::default().fg(Color::Gray)),
        Span::styled(&state.room.name, Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(
            format!("[{}]", short_id),
            Style::default().fg(Color::DarkGray),
        ),
    ])])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Conclave Observer "),
    );

    f.render_widget(header, area);
}

/// Draws the messages pane.
fn draw_messages(f: &mut Frame, area: Rect, state: &TuiState) {
    let messages: Vec<ListItem> = state
        .messages
        .iter()
        .skip(state.scroll)
        .take(area.height as usize - 2) // Account for borders
        .map(|msg| {
            let style = if msg.is_system {
                Style::default().fg(Color::DarkGray)
            } else if msg.is_tool {
                Style::default().fg(Color::Cyan)
            } else if msg.sender == "You" {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::White)
            };

            let prefix = if msg.is_tool {
                format!("[{}] [Tool] ", msg.timestamp)
            } else {
                format!("[{}] {}: ", msg.timestamp, msg.sender)
            };

            ListItem::new(Line::from(vec![
                Span::styled(prefix, style.add_modifier(Modifier::DIM)),
                Span::styled(&msg.content, style),
            ]))
        })
        .collect();

    let messages_widget = List::new(messages).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Messages "),
    );

    f.render_widget(messages_widget, area);
}

/// Draws the participants pane.
fn draw_participants(f: &mut Frame, area: Rect, state: &TuiState) {
    let participants: Vec<ListItem> = state
        .room
        .participants
        .iter()
        .map(|p| {
            let icon = if p.is_agent { "🤖" } else { "👤" };
            let kind = if p.is_agent {
                p.agent_type.as_deref().unwrap_or("agent")
            } else {
                "human"
            };

            ListItem::new(Line::from(vec![
                Span::raw(format!("{} ", icon)),
                Span::styled(&p.display_name, Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(format!(" ({})", kind), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let participants_widget = List::new(participants).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Participants "),
    );

    f.render_widget(participants_widget, area);
}

/// Draws the input area.
fn draw_input(f: &mut Frame, area: Rect, state: &TuiState) {
    let input = Paragraph::new(vec![Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Green)),
        Span::raw(&state.input),
        Span::styled("█", Style::default().fg(Color::Gray)), // Cursor
    ])])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Type message (Esc to exit) "),
    );

    f.render_widget(input, area);
}
