//! Regression test against the actual completions captured for
//! infernum-framework#70/#71, not a hand-written stand-in for them.
//!
//! `fixtures/issue_70_14b_completions.json` is the real data: all 126 raw
//! completions from a real Qwen2.5-Coder-14B-Instruct model, served through
//! upstream `llama-server`, across three runs (see the file's own
//! `provenance` field for exact model/server/commit details). Every number
//! quoted in #70's discriminator comment and #71's PR description is
//! recomputed here from that committed file, not asserted from memory —
//! if the fixture and a claim ever disagree, the fixture is the one to
//! trust, and this test is what would catch that disagreement.
//!
//! This exists because the first version of #71 embedded a handful of
//! hand-typed strings as "verbatim" evidence in unit tests, with the full
//! 126-completion dataset sitting outside the repository. That meant the
//! claim couldn't actually be re-run by anyone — including a mistranscribed
//! path in one of those strings and a synthetic (not captured) string that
//! wasn't flagged as such. This test reads the real file instead.

use std::collections::HashMap;

use beleth::{QwenToolCallDetector, ToolCallDetector};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    completions: Vec<Completion>,
}

#[derive(Debug, Deserialize)]
struct Completion {
    id: String,
    phase: String,
    expect_tool: Option<String>,
    required_args: Option<Vec<String>>,
    text: String,
    category: String,
}

fn load_fixture() -> Fixture {
    let raw = include_str!("fixtures/issue_70_14b_completions.json");
    serde_json::from_str(raw).expect("fixture must be valid JSON matching the Fixture schema")
}

/// The corpus is exactly the 126 completions captured for #70. If this
/// fails, the fixture was edited — which should never happen casually,
/// since every other assertion here depends on it being the real data.
#[test]
fn fixture_has_the_full_captured_corpus() {
    let fixture = load_fixture();
    assert_eq!(
        fixture.completions.len(),
        126,
        "expected all 126 completions captured for #70; the fixture appears to have been edited"
    );

    let mut by_phase: HashMap<&str, usize> = HashMap::new();
    for c in &fixture.completions {
        *by_phase.entry(c.phase.as_str()).or_default() += 1;
    }
    assert_eq!(by_phase.get("A").copied().unwrap_or(0), 90);
    assert_eq!(by_phase.get("B").copied().unwrap_or(0), 20);
    assert_eq!(by_phase.get("C").copied().unwrap_or(0), 16);
}

/// The headline recovery number from #71: how many of the 126 real
/// completions now produce a detected call, using the actual production
/// `QwenToolCallDetector` — not a reimplementation of it.
#[test]
fn detector_recovers_the_claimed_fraction_of_the_real_corpus() {
    let fixture = load_fixture();
    let detector = QwenToolCallDetector::new();

    let recovered = fixture
        .completions
        .iter()
        .filter(|c| !detector.detect(&c.text).is_empty())
        .count();

    assert_eq!(
        recovered, 89,
        "89/126 (70.6%) is the number quoted in PR #71 — recompute and update \
         the PR description together if this ever legitimately changes"
    );
}

/// Phase A specifically has a hard ground truth per probe (`expect_tool` /
/// `required_args`), so right-tool and schema-valid can be recomputed
/// exactly rather than approximated by "any call at all".
#[test]
fn phase_a_right_tool_and_schema_valid_match_the_claimed_numbers() {
    let fixture = load_fixture();
    let detector = QwenToolCallDetector::new();

    let phase_a: Vec<&Completion> = fixture
        .completions
        .iter()
        .filter(|c| c.phase == "A")
        .collect();
    assert_eq!(phase_a.len(), 90);

    let mut right_tool = 0;
    let mut schema_valid = 0;
    for c in &phase_a {
        let expect_tool = c
            .expect_tool
            .as_deref()
            .expect("every Phase A completion carries its probe's expected tool");
        let required_args = c
            .required_args
            .as_deref()
            .expect("every Phase A completion carries its probe's required args");

        if let Some(call) = detector
            .detect(&c.text)
            .into_iter()
            .find(|call| call.name == expect_tool)
        {
            right_tool += 1;
            let all_present = required_args.iter().all(|a| {
                call.arguments
                    .get(a)
                    .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
            });
            if all_present {
                schema_valid += 1;
            }
        }
    }

    assert_eq!(
        right_tool, 72,
        "72/90 (80.0%) right-tool, as quoted in PR #71"
    );
    assert_eq!(
        schema_valid, 72,
        "72/90 (80.0%) schema-valid, as quoted in PR #71"
    );
}

/// The negative control, run against the real corpus rather than a
/// hand-picked subset of it: every completion NOT categorized "recovered"
/// must still produce zero detected calls. This is what "do not relax the
/// detector until it passes" actually means — checked against all 37 real
/// non-recovered completions, not just the ones I remembered to name.
#[test]
fn every_non_recovered_completion_still_produces_zero_calls() {
    let fixture = load_fixture();
    let detector = QwenToolCallDetector::new();

    let mut checked = 0;
    for c in fixture
        .completions
        .iter()
        .filter(|c| c.category != "recovered")
    {
        checked += 1;
        let calls = detector.detect(&c.text);
        assert!(
            calls.is_empty(),
            "{} is categorized '{}' (expected to reject) but the detector found {calls:?}: {:?}",
            c.id,
            c.category,
            c.text
        );
    }

    assert_eq!(
        checked, 37,
        "expected 37 non-recovered completions (126 total - 89 recovered)"
    );
}

/// Per-category breakdown, so a category drifting silently (e.g. a future
/// change accidentally starting to recognize `function_name`) is a failing
/// test naming the exact category, not a vague count mismatch.
///
/// Correction folded in here: #70's discriminator comment said "17 genuine
/// non-attempts" and #71's tests hardcoded 17. Recomputing mechanically
/// from the full corpus turns up 18 — one (`A/tempA/read-with-awkward-
/// name#4`) was missed when that count was hand-assembled. This test uses
/// the true, recomputed number.
#[test]
fn category_counts_match_the_corrected_breakdown() {
    let fixture = load_fixture();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for c in &fixture.completions {
        *counts.entry(c.category.as_str()).or_default() += 1;
    }

    assert_eq!(counts.get("recovered").copied().unwrap_or(0), 89);
    assert_eq!(
        counts.get("no_attempt").copied().unwrap_or(0),
        18,
        "corrected count — #70's discriminator comment undercounted this as 17"
    );
    assert_eq!(counts.get("template_echo").copied().unwrap_or(0), 16);
    assert_eq!(counts.get("function_name_gap").copied().unwrap_or(0), 3);
}

/// The two literal strings the template-ambiguity issue (#72) is about:
/// the model completing the system prompt's own fill-in-the-blank example
/// verbatim, ellipsis included. Every `template_echo`-categorized
/// completion must be exactly one of these two strings — if a completion
/// is tagged `template_echo` but isn't a literal match, the fixture's
/// categorization (not the detector) has a bug.
#[test]
fn template_echo_completions_are_the_exact_literal_examples() {
    let fixture = load_fixture();
    let exact_strings = ["<yield>...</yield>", "<stuck>...</stuck>"];

    for c in fixture
        .completions
        .iter()
        .filter(|c| c.category == "template_echo")
    {
        assert!(
            exact_strings.contains(&c.text.as_str()),
            "{} is categorized 'template_echo' but its text is not one of the exact \
             example strings: {:?}",
            c.id,
            c.text
        );
    }
}
