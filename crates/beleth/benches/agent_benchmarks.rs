//! Benchmarks for agent framework components.
//!
//! Run with: `cargo bench --package beleth`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use beleth::dynamic_context::{
    score_message_relevance, semantic_chunk, ContextComplexity, DynamicContextManager,
};
use beleth::long_term_memory::{ImportanceLevel, MemoryEntry, MemoryType};
use beleth::tool::{TaskComplexity, ToolTimeoutConfig};
use infernum_core::{Message, Role};
use std::time::Duration;

// ============================================================================
// Dynamic Context Benchmarks
// ============================================================================

fn bench_semantic_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_chunking");

    // Test different content sizes
    let sizes = [100, 1000, 10000, 50000];

    for size in sizes {
        let content = generate_mixed_content(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("chunk", size), &content, |b, content| {
            b.iter(|| semantic_chunk(black_box(content), 2000))
        });
    }

    group.finish();
}

fn bench_complexity_classification(c: &mut Criterion) {
    let tasks = [
        "fix typo in readme",
        "add a new function to handle user input",
        "refactor the entire authentication system with OAuth2 support",
        "implement end-to-end encryption for all user data",
    ];

    c.bench_function("complexity_classification", |b| {
        b.iter(|| {
            for task in &tasks {
                black_box(ContextComplexity::classify(task));
            }
        })
    });
}

fn bench_message_relevance_scoring(c: &mut Criterion) {
    let messages: Vec<Message> = (0..100)
        .map(|i| Message {
            role: if i % 3 == 0 {
                Role::System
            } else if i % 2 == 0 {
                Role::Assistant
            } else {
                Role::User
            },
            content: format!("Message {} with some content about Rust and programming", i),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        })
        .collect();

    c.bench_function("relevance_scoring_100_messages", |b| {
        b.iter(|| {
            for (i, msg) in messages.iter().enumerate() {
                black_box(score_message_relevance(
                    msg,
                    i,
                    messages.len(),
                    Some("implement Rust feature"),
                ));
            }
        })
    });
}

fn bench_context_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_optimization");

    for msg_count in [10, 50, 100, 500] {
        let messages: Vec<Message> = (0..msg_count)
            .map(|i| Message {
                role: Role::User,
                content: format!("Message {} with content", i),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        let manager = DynamicContextManager::new().with_task("implement feature");

        group.bench_with_input(
            BenchmarkId::new("optimize", msg_count),
            &messages,
            |b, messages| b.iter(|| manager.optimize(black_box(messages))),
        );
    }

    group.finish();
}

// ============================================================================
// Long-Term Memory Benchmarks
// ============================================================================

fn bench_memory_entry_creation(c: &mut Criterion) {
    c.bench_function("memory_entry_creation", |b| {
        b.iter(|| {
            MemoryEntry::new(MemoryType::Decision, "Important architectural decision")
                .with_importance(ImportanceLevel::High)
                .with_tag("architecture")
                .with_tag("database")
                .with_summary("We chose PostgreSQL for ACID compliance")
                .with_metadata("author", "team-lead")
        })
    });
}

fn bench_memory_entry_matching(c: &mut Criterion) {
    let entries: Vec<MemoryEntry> = (0..1000)
        .map(|i| {
            MemoryEntry::new(
                MemoryType::Context,
                format!(
                    "Entry {} about Rust programming and system design patterns",
                    i
                ),
            )
            .with_tag("rust")
            .with_tag("patterns")
        })
        .collect();

    let queries = ["rust", "python", "programming", "design", "unknown"];

    c.bench_function("memory_matching_1000_entries", |b| {
        b.iter(|| {
            for query in &queries {
                let matches: Vec<_> = entries.iter().filter(|e| e.matches(query)).collect();
                black_box(matches);
            }
        })
    });
}

// ============================================================================
// Tool Timeout Benchmarks
// ============================================================================

fn bench_timeout_calculation(c: &mut Criterion) {
    let config = ToolTimeoutConfig::new(Duration::from_secs(30))
        .with_tool_timeout("http", Duration::from_secs(60))
        .with_tool_timeout("database", Duration::from_secs(45))
        .with_tool_timeout("file_read", Duration::from_secs(10))
        .with_complexity_multiplier(1.5);

    let tools = ["http", "database", "file_read", "unknown", "search"];

    c.bench_function("timeout_calculation", |b| {
        b.iter(|| {
            for tool in &tools {
                black_box(config.get_timeout(tool));
            }
        })
    });
}

fn bench_task_complexity_multiplier(c: &mut Criterion) {
    let complexities = [
        TaskComplexity::Simple,
        TaskComplexity::Moderate,
        TaskComplexity::Complex,
    ];

    c.bench_function("complexity_multiplier", |b| {
        b.iter(|| {
            for complexity in &complexities {
                black_box(complexity.multiplier());
            }
        })
    });
}

// ============================================================================
// Helper Functions
// ============================================================================

fn generate_mixed_content(approx_chars: usize) -> String {
    let mut content = String::with_capacity(approx_chars);

    // Add headers
    content.push_str("# Main Title\n\n");
    content.push_str("Some introductory text about the topic.\n\n");

    // Add code block
    content.push_str("```rust\n");
    content.push_str("fn main() {\n");
    content.push_str("    println!(\"Hello, world!\");\n");
    content.push_str("}\n");
    content.push_str("```\n\n");

    // Add list
    content.push_str("- Item one\n");
    content.push_str("- Item two\n");
    content.push_str("- Item three\n\n");

    // Fill remaining with prose
    while content.len() < approx_chars {
        content.push_str("This is additional prose content to fill the buffer. ");
        content.push_str("It contains various words and sentences for testing. ");
    }

    content.truncate(approx_chars);
    content
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    dynamic_context,
    bench_semantic_chunking,
    bench_complexity_classification,
    bench_message_relevance_scoring,
    bench_context_optimization,
);

criterion_group!(
    memory,
    bench_memory_entry_creation,
    bench_memory_entry_matching,
);

criterion_group!(
    tool_timeout,
    bench_timeout_calculation,
    bench_task_complexity_multiplier,
);

criterion_main!(dynamic_context, memory, tool_timeout);
