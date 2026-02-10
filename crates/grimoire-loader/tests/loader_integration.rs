//! Integration tests for Grimoire persona loader.
//!
//! Tests the persona loading, caching, and listing workflows.

use grimoire_loader::{default_grimoire_path, GrimoireLoader, GrimoirePersona, GRIMOIRE_PATH_ENV};
use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// GrimoirePersona Structure Tests
// ============================================================================

#[test]
fn test_persona_creation() {
    let persona = GrimoirePersona {
        id: "test-agent".to_string(),
        name: "Test Agent".to_string(),
        system_prompt: "You are a helpful test agent.".to_string(),
        variants: HashMap::new(),
        metadata: HashMap::new(),
    };

    assert_eq!(persona.id, "test-agent");
    assert_eq!(persona.name, "Test Agent");
    assert!(persona.system_prompt.contains("helpful"));
}

#[test]
fn test_persona_with_variants() {
    let mut variants = HashMap::new();
    variants.insert(
        "formal".to_string(),
        "You are a formal assistant.".to_string(),
    );
    variants.insert("casual".to_string(), "Hey! I'm here to help!".to_string());
    variants.insert(
        "technical".to_string(),
        "I provide technical assistance.".to_string(),
    );

    let persona = GrimoirePersona {
        id: "multi-variant".to_string(),
        name: "Multi Variant Agent".to_string(),
        system_prompt: "Default prompt".to_string(),
        variants,
        metadata: HashMap::new(),
    };

    assert_eq!(persona.variants.len(), 3);
    assert!(persona.variants.get("formal").unwrap().contains("formal"));
    assert!(persona.variants.get("casual").unwrap().contains("Hey"));
    assert!(persona
        .variants
        .get("technical")
        .unwrap()
        .contains("technical"));
}

#[test]
fn test_persona_with_metadata() {
    let mut metadata = HashMap::new();
    metadata.insert("version".to_string(), "2.0".to_string());
    metadata.insert("author".to_string(), "Test Author".to_string());
    metadata.insert("category".to_string(), "development".to_string());
    metadata.insert("model_preference".to_string(), "claude-opus-4".to_string());

    let persona = GrimoirePersona {
        id: "metadata-test".to_string(),
        name: "Metadata Test".to_string(),
        system_prompt: "Test prompt".to_string(),
        variants: HashMap::new(),
        metadata,
    };

    assert_eq!(persona.metadata.get("version"), Some(&"2.0".to_string()));
    assert_eq!(
        persona.metadata.get("author"),
        Some(&"Test Author".to_string())
    );
    assert_eq!(
        persona.metadata.get("category"),
        Some(&"development".to_string())
    );
    assert_eq!(
        persona.metadata.get("model_preference"),
        Some(&"claude-opus-4".to_string())
    );
}

#[test]
fn test_persona_serialization_roundtrip() {
    let mut variants = HashMap::new();
    variants.insert("v1".to_string(), "Variant 1".to_string());

    let mut metadata = HashMap::new();
    metadata.insert("key".to_string(), "value".to_string());

    let persona = GrimoirePersona {
        id: "serialize-test".to_string(),
        name: "Serialization Test".to_string(),
        system_prompt: "You are a test assistant.".to_string(),
        variants,
        metadata,
    };

    // Serialize to JSON
    let json = serde_json::to_string(&persona).expect("serialize");
    assert!(json.contains("serialize-test"));
    assert!(json.contains("Serialization Test"));
    assert!(json.contains("test assistant"));

    // Deserialize back
    let parsed: GrimoirePersona = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.id, persona.id);
    assert_eq!(parsed.name, persona.name);
    assert_eq!(parsed.system_prompt, persona.system_prompt);
    assert_eq!(parsed.variants.len(), 1);
    assert_eq!(parsed.metadata.len(), 1);
}

#[test]
fn test_persona_clone() {
    let persona = GrimoirePersona {
        id: "clone-test".to_string(),
        name: "Clone Test".to_string(),
        system_prompt: "Original prompt".to_string(),
        variants: HashMap::new(),
        metadata: HashMap::new(),
    };

    let cloned = persona.clone();

    assert_eq!(cloned.id, persona.id);
    assert_eq!(cloned.name, persona.name);
    assert_eq!(cloned.system_prompt, persona.system_prompt);
}

#[test]
fn test_persona_debug_format() {
    let persona = GrimoirePersona {
        id: "debug-test".to_string(),
        name: "Debug Test".to_string(),
        system_prompt: "Debug prompt".to_string(),
        variants: HashMap::new(),
        metadata: HashMap::new(),
    };

    let debug = format!("{:?}", persona);
    assert!(debug.contains("debug-test"));
    assert!(debug.contains("Debug Test"));
}

// ============================================================================
// GrimoireLoader Creation Tests
// ============================================================================

#[test]
fn test_loader_with_custom_path() {
    let loader = GrimoireLoader::with_path("/custom/grimoire/path");
    assert_eq!(
        loader.base_path().to_string_lossy(),
        "/custom/grimoire/path"
    );
}

#[test]
fn test_loader_with_pathbuf() {
    let path = std::path::PathBuf::from("/another/path");
    let loader = GrimoireLoader::with_path(path);
    assert_eq!(loader.base_path().to_string_lossy(), "/another/path");
}

#[test]
fn test_loader_default_implementation() {
    // Clear env var to test default behavior
    std::env::remove_var(GRIMOIRE_PATH_ENV);

    let loader = GrimoireLoader::default();
    assert!(loader.base_path().to_string_lossy().contains("personas"));
}

#[test]
fn test_loader_from_env_var() {
    let custom_path = "/env/grimoire/path";
    std::env::set_var(GRIMOIRE_PATH_ENV, custom_path);

    let path = default_grimoire_path();
    assert_eq!(path.to_string_lossy(), custom_path);

    // Clean up
    std::env::remove_var(GRIMOIRE_PATH_ENV);
}

#[test]
fn test_default_grimoire_path_without_env() {
    std::env::remove_var(GRIMOIRE_PATH_ENV);

    let path = default_grimoire_path();
    assert!(path.to_string_lossy().contains("infernum"));
    assert!(path.to_string_lossy().contains("personas"));
}

// ============================================================================
// Cache Management Tests
// ============================================================================

#[tokio::test]
async fn test_cache_clear() {
    let temp = TempDir::new().expect("temp dir");

    // Create persona files
    for i in 0..3 {
        fs::write(
            temp.path().join(format!("persona-{}.md", i)),
            format!("Prompt {}", i),
        )
        .expect("write");
    }

    let loader = GrimoireLoader::with_path(temp.path());

    // Load personas to populate cache
    for i in 0..3 {
        loader.load(&format!("persona-{}", i)).await.expect("load");
    }

    // Clear cache - just verify it doesn't panic
    loader.clear_cache();

    // Load again after clear - should work
    let persona = loader.load("persona-0").await.expect("reload");
    assert_eq!(persona.id, "persona-0");
}

// ============================================================================
// Async Loading Tests
// ============================================================================

#[tokio::test]
async fn test_load_from_markdown_file() {
    let temp = TempDir::new().expect("temp dir");
    let prompt = "You are a helpful coding assistant specialized in Rust.";
    fs::write(temp.path().join("rust-helper.md"), prompt).expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let persona = loader.load("rust-helper").await.expect("load");

    assert_eq!(persona.id, "rust-helper");
    assert_eq!(persona.system_prompt, prompt);
}

#[tokio::test]
async fn test_load_from_directory_with_prompt_md() {
    let temp = TempDir::new().expect("temp dir");
    let persona_dir = temp.path().join("complex-agent");
    fs::create_dir(&persona_dir).expect("mkdir");

    let prompt = "You are a complex agent with multiple capabilities and tools.";
    fs::write(persona_dir.join("prompt.md"), prompt).expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let persona = loader.load("complex-agent").await.expect("load");

    assert_eq!(persona.id, "complex-agent");
    assert_eq!(persona.system_prompt, prompt);
}

#[tokio::test]
async fn test_load_multiple_personas() {
    let temp = TempDir::new().expect("temp dir");

    // Create multiple personas
    let personas_data = [
        ("coder", "You are a coding assistant."),
        ("reviewer", "You are a code reviewer."),
        ("writer", "You are a technical writer."),
        ("debugger", "You are a debugging expert."),
    ];

    for (name, prompt) in &personas_data {
        fs::write(temp.path().join(format!("{}.md", name)), prompt).expect("write");
    }

    let loader = GrimoireLoader::with_path(temp.path());

    // Load all personas
    for (name, expected_prompt) in &personas_data {
        let persona = loader.load(name).await.expect("load");
        assert_eq!(persona.id, *name);
        assert_eq!(persona.system_prompt, *expected_prompt);
    }
}

#[tokio::test]
async fn test_load_caching_behavior() {
    let temp = TempDir::new().expect("temp dir");
    let initial_prompt = "Initial prompt content.";
    fs::write(temp.path().join("cached.md"), initial_prompt).expect("write");

    let loader = GrimoireLoader::with_path(temp.path());

    // First load
    let persona1 = loader.load("cached").await.expect("load");
    assert_eq!(persona1.system_prompt, initial_prompt);

    // Modify file on disk
    let modified_prompt = "Modified prompt content.";
    fs::write(temp.path().join("cached.md"), modified_prompt).expect("write");

    // Second load should return cached version
    let persona2 = loader.load("cached").await.expect("load");
    assert_eq!(persona2.system_prompt, initial_prompt);

    // Clear cache
    loader.clear_cache();

    // Third load should get new content
    let persona3 = loader.load("cached").await.expect("load");
    assert_eq!(persona3.system_prompt, modified_prompt);
}

#[tokio::test]
async fn test_load_nonexistent_returns_error() {
    let temp = TempDir::new().expect("temp dir");
    let loader = GrimoireLoader::with_path(temp.path());

    let result = loader.load("nonexistent-persona").await;
    assert!(result.is_err());
}

// ============================================================================
// Listing Tests
// ============================================================================

#[tokio::test]
async fn test_list_empty_directory() {
    let temp = TempDir::new().expect("temp dir");
    let loader = GrimoireLoader::with_path(temp.path());

    let personas = loader.list().await.expect("list");
    assert!(personas.is_empty());
}

#[tokio::test]
async fn test_list_nonexistent_directory() {
    let loader = GrimoireLoader::with_path("/nonexistent/path/12345");

    let personas = loader.list().await.expect("list");
    assert!(personas.is_empty());
}

#[tokio::test]
async fn test_list_markdown_files() {
    let temp = TempDir::new().expect("temp dir");

    // Create markdown files
    fs::write(temp.path().join("agent-one.md"), "Prompt 1").expect("write");
    fs::write(temp.path().join("agent-two.md"), "Prompt 2").expect("write");
    fs::write(temp.path().join("agent-three.md"), "Prompt 3").expect("write");

    // Create non-markdown files (should be ignored)
    fs::write(temp.path().join("readme.txt"), "Not a persona").expect("write");
    fs::write(temp.path().join("config.json"), "{}").expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let personas = loader.list().await.expect("list");

    assert_eq!(personas.len(), 3);
    assert!(personas.contains(&"agent-one".to_string()));
    assert!(personas.contains(&"agent-two".to_string()));
    assert!(personas.contains(&"agent-three".to_string()));
}

#[tokio::test]
async fn test_list_directories() {
    let temp = TempDir::new().expect("temp dir");

    // Create persona directories
    fs::create_dir(temp.path().join("dir-agent-1")).expect("mkdir");
    fs::create_dir(temp.path().join("dir-agent-2")).expect("mkdir");

    let loader = GrimoireLoader::with_path(temp.path());
    let personas = loader.list().await.expect("list");

    assert!(personas.contains(&"dir-agent-1".to_string()));
    assert!(personas.contains(&"dir-agent-2".to_string()));
}

#[tokio::test]
async fn test_list_mixed_files_and_directories() {
    let temp = TempDir::new().expect("temp dir");

    // Create markdown files
    fs::write(temp.path().join("file-agent.md"), "File prompt").expect("write");

    // Create directories
    fs::create_dir(temp.path().join("dir-agent")).expect("mkdir");
    fs::write(
        temp.path().join("dir-agent").join("prompt.md"),
        "Directory prompt",
    )
    .expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let personas = loader.list().await.expect("list");

    assert_eq!(personas.len(), 2);
    assert!(personas.contains(&"file-agent".to_string()));
    assert!(personas.contains(&"dir-agent".to_string()));
}

// ============================================================================
// Workflow Integration Tests
// ============================================================================

#[tokio::test]
async fn test_complete_persona_workflow() {
    let temp = TempDir::new().expect("temp dir");

    // 1. Create a set of personas
    let personas = [
        ("analyst", "You are a data analyst."),
        ("architect", "You are a software architect."),
        ("tester", "You are a QA engineer."),
    ];

    for (name, prompt) in &personas {
        fs::write(temp.path().join(format!("{}.md", name)), prompt).expect("write");
    }

    let loader = GrimoireLoader::with_path(temp.path());

    // 2. List all personas
    let available = loader.list().await.expect("list");
    assert_eq!(available.len(), 3);

    // 3. Load each persona and verify
    for (name, expected_prompt) in &personas {
        let persona = loader.load(name).await.expect("load");
        assert_eq!(persona.id, *name);
        assert_eq!(persona.system_prompt, *expected_prompt);
    }

    // 4. Verify caching
    let cached = loader.load("analyst").await.expect("cached load");
    assert_eq!(cached.system_prompt, "You are a data analyst.");

    // 5. Clear cache
    loader.clear_cache();

    // 6. Reload
    let reloaded = loader.load("analyst").await.expect("reload");
    assert_eq!(reloaded.system_prompt, "You are a data analyst.");
}

#[tokio::test]
async fn test_persona_update_workflow() {
    let temp = TempDir::new().expect("temp dir");

    // Create initial persona
    fs::write(
        temp.path().join("evolving.md"),
        "Version 1: Basic assistant.",
    )
    .expect("write");

    let loader = GrimoireLoader::with_path(temp.path());

    // Load v1
    let v1 = loader.load("evolving").await.expect("load v1");
    assert!(v1.system_prompt.contains("Version 1"));

    // Update persona file
    fs::write(
        temp.path().join("evolving.md"),
        "Version 2: Enhanced assistant with new capabilities.",
    )
    .expect("write");

    // Still returns cached v1
    let still_v1 = loader.load("evolving").await.expect("load still v1");
    assert!(still_v1.system_prompt.contains("Version 1"));

    // Clear cache to get updated version
    loader.clear_cache();

    // Now get v2
    let v2 = loader.load("evolving").await.expect("load v2");
    assert!(v2.system_prompt.contains("Version 2"));
}

#[tokio::test]
async fn test_hierarchical_persona_structure() {
    let temp = TempDir::new().expect("temp dir");

    // Create hierarchical structure
    // personas/
    //   simple.md
    //   complex/
    //     prompt.md

    fs::write(temp.path().join("simple.md"), "Simple persona prompt").expect("write");

    let complex_dir = temp.path().join("complex");
    fs::create_dir(&complex_dir).expect("mkdir");
    fs::write(complex_dir.join("prompt.md"), "Complex persona prompt").expect("write");

    let loader = GrimoireLoader::with_path(temp.path());

    // List should show both
    let list = loader.list().await.expect("list");
    assert!(list.contains(&"simple".to_string()));
    assert!(list.contains(&"complex".to_string()));

    // Load both
    let simple = loader.load("simple").await.expect("load simple");
    assert_eq!(simple.system_prompt, "Simple persona prompt");

    let complex = loader.load("complex").await.expect("load complex");
    assert_eq!(complex.system_prompt, "Complex persona prompt");
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[tokio::test]
async fn test_persona_with_special_characters_in_prompt() {
    let temp = TempDir::new().expect("temp dir");

    let prompt = r#"You are an assistant that handles special characters:
- Quotes: "double" and 'single'
- Unicode: 日本語, 中文, emoji 🎉
- Code blocks: `code`
- Markdown: **bold** and *italic*
- Math: E = mc²
"#;

    fs::write(temp.path().join("special.md"), prompt).expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let persona = loader.load("special").await.expect("load");

    assert!(persona.system_prompt.contains("日本語"));
    assert!(persona.system_prompt.contains("🎉"));
    assert!(persona.system_prompt.contains("mc²"));
}

#[tokio::test]
async fn test_empty_prompt_file() {
    let temp = TempDir::new().expect("temp dir");
    fs::write(temp.path().join("empty.md"), "").expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let persona = loader.load("empty").await.expect("load");

    assert_eq!(persona.id, "empty");
    assert_eq!(persona.system_prompt, "");
}

#[tokio::test]
async fn test_large_prompt_file() {
    let temp = TempDir::new().expect("temp dir");

    // Create a large prompt (100KB)
    let large_prompt: String = (0..10000)
        .map(|i| {
            format!(
                "Line {}: This is part of a very detailed system prompt.\n",
                i
            )
        })
        .collect();

    fs::write(temp.path().join("large.md"), &large_prompt).expect("write");

    let loader = GrimoireLoader::with_path(temp.path());
    let persona = loader.load("large").await.expect("load");

    assert_eq!(persona.system_prompt.len(), large_prompt.len());
}

#[test]
fn test_env_constant_value() {
    assert_eq!(GRIMOIRE_PATH_ENV, "INFERNUM_GRIMOIRE_PATH");
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_loads() {
    let temp = TempDir::new().expect("temp dir");

    // Create test persona
    fs::write(temp.path().join("concurrent.md"), "Concurrent test prompt").expect("write");

    let loader = std::sync::Arc::new(GrimoireLoader::with_path(temp.path()));

    // Spawn multiple concurrent loads
    let mut handles = Vec::new();
    for _ in 0..10 {
        let loader_clone = loader.clone();
        let handle = tokio::spawn(async move { loader_clone.load("concurrent").await });
        handles.push(handle);
    }

    // All should succeed
    for handle in handles {
        let result = handle.await.expect("join");
        let persona = result.expect("load");
        assert_eq!(persona.id, "concurrent");
    }
}

#[tokio::test]
async fn test_concurrent_list_and_load() {
    let temp = TempDir::new().expect("temp dir");

    // Create multiple personas
    for i in 0..5 {
        fs::write(
            temp.path().join(format!("agent-{}.md", i)),
            format!("Agent {} prompt", i),
        )
        .expect("write");
    }

    let loader = std::sync::Arc::new(GrimoireLoader::with_path(temp.path()));

    // Concurrent list operations
    let loader_list = loader.clone();
    let list_handle = tokio::spawn(async move { loader_list.list().await });

    // Concurrent load operations
    let mut load_handles = Vec::new();
    for i in 0..5 {
        let loader_clone = loader.clone();
        let handle = tokio::spawn(async move { loader_clone.load(&format!("agent-{}", i)).await });
        load_handles.push(handle);
    }

    // Verify results
    let list_result = list_handle.await.expect("join list");
    let personas = list_result.expect("list");
    assert_eq!(personas.len(), 5);

    for handle in load_handles {
        let result = handle.await.expect("join");
        assert!(result.is_ok());
    }
}
