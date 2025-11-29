//! # Beleth
//!
//! *"The King commands legions"*
//!
//! Beleth is the agent framework for the Infernum ecosystem,
//! enabling autonomous task execution with tool use, planning, and memory.
//!
//! ## Features
//!
//! - **Tool System**: Extensible tool interface for agent actions
//! - **Planning**: Multiple planning strategies (ReAct, ToT, Hierarchical)
//! - **Memory**: Working, episodic, and semantic memory systems
//! - **Grimoire Integration**: Native support for Grimoire personas

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod agent;
pub mod memory;
pub mod planner;
pub mod tool;

pub use agent::{Agent, AgentBuilder, Persona};
pub use memory::AgentMemory;
pub use planner::{Planner, PlanningStrategy};
pub use tool::{Tool, ToolRegistry, ToolResult};
