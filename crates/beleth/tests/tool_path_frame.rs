//! The path-frame contract between the file tools (INFERNUM-25).
//!
//! `search_files` and `list_files` report paths; `read_file`, `edit_file` and
//! `write_file` accept them. The contract is that these are the *same frame*:
//! a path emitted by one tool is a valid input to any other, unchanged.
//!
//! Before this was pinned, `search_files` reported paths relative to the
//! directory it was told to search. Searching `src/` for a symbol returned
//! `lib.rs`, and `edit_file("lib.rs")` then failed with
//! "No such file or directory" — the file is at `src/lib.rs`. A model that
//! trusted the output of one tool could not use it in the next.
//!
//! These tests are deliberately written as the agent uses the tools: take the
//! path out of the search output *as text* and hand it straight to the next
//! tool. Asserting on the string shape alone would not catch a frame that is
//! self-consistently wrong.

use std::collections::HashMap;

use beleth::tool::{Tool, ToolContext};
use beleth::tools::{EditFileTool, ListFilesTool, ReadFileTool, SearchFilesTool};
use serde_json::{json, Value};

/// Builds the fixture the `rename-symbol` eval task uses: a symbol defined in
/// one file under `src/` and referenced from two more.
fn make_crate() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub mod parse;\npub mod render;\n\n\
         pub fn legacy_slug(s: &str) -> String {\n    \
         s.to_lowercase().replace(' ', \"-\")\n}\n",
    )
    .expect("write");
    std::fs::write(
        dir.path().join("src/parse.rs"),
        "pub fn parse_title(s: &str) -> String {\n    crate::legacy_slug(s)\n}\n",
    )
    .expect("write");
    std::fs::write(
        dir.path().join("src/render.rs"),
        "pub fn render_link(s: &str) -> String {\n    \
         format!(\"/{}\", crate::legacy_slug(s))\n}\n",
    )
    .expect("write");
    dir
}

fn ctx_for(dir: &std::path::Path) -> ToolContext {
    let mut state = HashMap::new();
    state.insert(
        "working_dir".to_string(),
        Value::String(dir.to_string_lossy().to_string()),
    );
    let mut ctx = ToolContext::new("path-frame-test");
    ctx.state = state;
    ctx
}

/// Parses `path:line:content` the way a caller reading the tool's output must.
fn paths_from_search(output: &str) -> Vec<String> {
    let mut paths: Vec<String> = output
        .lines()
        .filter_map(|line| line.split(':').next())
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

/// The invariant: grep for a symbol, then feed every path the search reported
/// straight into `read_file`. Every one must resolve.
#[tokio::test]
async fn search_files_paths_are_valid_read_file_input() {
    let dir = make_crate();
    let ctx = ctx_for(dir.path());

    let search = SearchFilesTool
        .execute(json!({"pattern": "legacy_slug", "path": "src"}), &ctx)
        .await
        .expect("search executes");
    assert!(search.success, "search failed: {:?}", search.error);

    let paths = paths_from_search(&search.output);
    assert_eq!(
        paths,
        vec!["src/lib.rs", "src/parse.rs", "src/render.rs"],
        "search output was:\n{}",
        search.output
    );

    for path in &paths {
        let read = ReadFileTool
            .execute(json!({"path": path}), &ctx)
            .await
            .expect("read executes");
        assert!(
            read.success,
            "search_files reported '{}' but read_file rejected it: {:?}\n\
             search output was:\n{}",
            path, read.error, search.output
        );
        assert!(
            read.output.contains("legacy_slug"),
            "read_file('{}') returned a file without the searched symbol",
            path
        );
    }
}

/// The same path, handed to the tool that actually mutates the file. This is
/// the step that failed three times in the traced `rename-symbol` run.
#[tokio::test]
async fn search_files_paths_are_valid_edit_file_input() {
    let dir = make_crate();
    let ctx = ctx_for(dir.path());

    let search = SearchFilesTool
        .execute(json!({"pattern": "legacy_slug", "path": "src"}), &ctx)
        .await
        .expect("search executes");
    assert!(search.success, "search failed: {:?}", search.error);

    for path in paths_from_search(&search.output) {
        let edit = EditFileTool
            .execute(
                json!({
                    "path": path,
                    "old_string": "legacy_slug",
                    "new_string": "make_slug",
                    "replace_all": true,
                }),
                &ctx,
            )
            .await
            .expect("edit executes");
        assert!(
            edit.success,
            "search_files reported '{}' but edit_file rejected it: {:?}",
            path, edit.error
        );
    }

    // And the rename actually landed, in the frame the search reported.
    for name in ["src/lib.rs", "src/parse.rs", "src/render.rs"] {
        let content = std::fs::read_to_string(dir.path().join(name)).expect("read back");
        assert!(
            !content.contains("legacy_slug"),
            "{} still contains the old symbol",
            name
        );
        assert!(content.contains("make_slug"), "{} was not renamed", name);
    }
}

/// `search_files` with no `path` argument defaults to the working directory,
/// where the two frames coincide. That case was never broken; this pins it so
/// the fix cannot regress it.
#[tokio::test]
async fn search_files_default_path_is_unchanged() {
    let dir = make_crate();
    let ctx = ctx_for(dir.path());

    let search = SearchFilesTool
        .execute(json!({"pattern": "legacy_slug"}), &ctx)
        .await
        .expect("search executes");
    assert!(search.success);
    assert_eq!(
        paths_from_search(&search.output),
        vec!["src/lib.rs", "src/parse.rs", "src/render.rs"]
    );
}

/// `list_files` output is consumed the same way, so it carries the same
/// contract — in both its directory-listing and its glob mode.
#[tokio::test]
async fn list_files_paths_are_valid_read_file_input() {
    let dir = make_crate();
    let ctx = ctx_for(dir.path());

    for params in [
        json!({"path": "src"}),
        json!({"path": "src", "pattern": "**/*.rs"}),
    ] {
        let listed = ListFilesTool
            .execute(params.clone(), &ctx)
            .await
            .expect("list executes");
        assert!(listed.success, "list failed: {:?}", listed.error);

        let entries: Vec<&str> = listed
            .output
            .lines()
            .filter(|line| !line.ends_with('/'))
            .collect();
        assert!(
            !entries.is_empty(),
            "expected files for {}, got:\n{}",
            params,
            listed.output
        );

        for entry in entries {
            let read = ReadFileTool
                .execute(json!({"path": entry}), &ctx)
                .await
                .expect("read executes");
            assert!(
                read.success,
                "list_files({}) reported '{}' but read_file rejected it: {:?}",
                params, entry, read.error
            );
        }
    }
}
