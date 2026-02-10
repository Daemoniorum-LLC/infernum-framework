//! Meta-signal detection for the agentic loop.
//!
//! Detects structured (explicit) and natural-language (implicit) signals
//! from agent output: answers, uncertainty, stuck states, and yield signals.
//!
//! Reference: AGENTIC-LOOP-SPEC.md §4.

use super::types::{AttemptSummary, DetectionConfig, MetaSignal, StuckRequest};

/// Patterns that suggest uncertainty (not failure).
const UNCERTAINTY_PATTERNS: &[&str] = &[
    "I'm not certain",
    "I couldn't find definitive",
    "This might be",
    "I would need",
    "Without access to",
    "I'm not sure",
    "It's unclear",
    "I don't have enough information",
];

/// Patterns that suggest being stuck (need help, not broken).
const STUCK_PATTERNS: &[&str] = &[
    "I've tried several approaches",
    "I'm not making progress",
    "I'm going in circles",
    "I need clarification",
    "I'm stuck",
    "I can't figure out",
    "I've exhausted",
];

/// Detect a meta-signal in model output.
///
/// Checks for explicit XML-structured signals first, then falls back to
/// implicit natural-language pattern detection if enabled.
pub fn detect_meta_signal(output: &str, config: &DetectionConfig) -> Option<MetaSignal> {
    // Explicit signals take priority
    if let Some(signal) = parse_explicit_answer(output) {
        return Some(signal);
    }
    if let Some(signal) = parse_explicit_uncertain(output) {
        return Some(signal);
    }
    if let Some(signal) = parse_explicit_stuck(output) {
        return Some(signal);
    }
    if let Some(signal) = parse_explicit_yield(output) {
        return Some(signal);
    }
    if let Some(signal) = parse_explicit_thinking(output) {
        return Some(signal);
    }

    // Implicit detection (pattern matching on natural language)
    if config.detect_implicit {
        if let Some(signal) = detect_stuck_patterns(output) {
            return Some(signal);
        }
        if let Some(signal) = detect_uncertainty_patterns(output) {
            return Some(signal);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Explicit signal parsing
// ---------------------------------------------------------------------------

/// Parse `<answer confidence="0.92">content<caveat>...</caveat></answer>`
fn parse_explicit_answer(output: &str) -> Option<MetaSignal> {
    let start_idx = output.find("<answer")?;
    let end_tag = "</answer>";
    let end_idx = output.find(end_tag)?;

    let tag_content = &output[start_idx..end_idx + end_tag.len()];

    // Extract confidence attribute
    let confidence = extract_attribute(tag_content, "confidence")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.8);

    // Extract inner content (between > and </answer>)
    let inner_start = tag_content.find('>')? + 1;
    let inner = &tag_content[inner_start..tag_content.len() - end_tag.len()];

    // Extract caveats
    let caveats = extract_all_tags(inner, "caveat");

    // Remove caveat tags from content to get the answer
    let mut content = inner.to_string();
    for caveat in &caveats {
        let caveat_tag = format!("<caveat>{caveat}</caveat>");
        content = content.replace(&caveat_tag, "");
    }
    let content = content.trim().to_string();

    Some(MetaSignal::Answer {
        content,
        confidence: confidence.clamp(0.0, 1.0),
        caveats,
    })
}

/// Parse `<uncertain><partial>...</partial><missing>...</missing></uncertain>`
fn parse_explicit_uncertain(output: &str) -> Option<MetaSignal> {
    let start_tag = "<uncertain>";
    let end_tag = "</uncertain>";
    let start_idx = output.find(start_tag)?;
    let end_idx = output.find(end_tag)?;

    let inner = &output[start_idx + start_tag.len()..end_idx];

    let partial_answer = extract_tag_content(inner, "partial");
    let missing_information = extract_all_tags(inner, "missing");
    let would_help = extract_all_tags(inner, "would_help");

    Some(MetaSignal::Uncertain {
        partial_answer,
        missing_information,
        would_help,
    })
}

/// Parse `<stuck><hypothesis>...</hypothesis><request>...</request></stuck>`
fn parse_explicit_stuck(output: &str) -> Option<MetaSignal> {
    let start_tag = "<stuck>";
    let end_tag = "</stuck>";
    let start_idx = output.find(start_tag)?;
    let end_idx = output.find(end_tag)?;

    let inner = &output[start_idx + start_tag.len()..end_idx];

    let hypothesis = extract_tag_content(inner, "hypothesis");
    let attempts = extract_all_tags(inner, "attempt")
        .into_iter()
        .map(|desc| AttemptSummary {
            description: desc,
            outcome: String::new(),
        })
        .collect();

    let request_text = extract_tag_content(inner, "request");
    let request = match request_text.as_deref() {
        Some(text) if text.contains("clarif") => {
            StuckRequest::Clarification(vec![text.to_string()])
        },
        Some(text) if text.contains("context") => StuckRequest::MoreContext {
            about: text.to_string(),
        },
        Some(text) if text.contains("tool") => StuckRequest::DifferentTools {
            need: vec![text.to_string()],
        },
        Some(text) => StuckRequest::HumanIntervention {
            reason: text.to_string(),
        },
        None => StuckRequest::HumanIntervention {
            reason: "Agent is stuck".to_string(),
        },
    };

    Some(MetaSignal::Stuck {
        attempts,
        hypothesis,
        request,
    })
}

/// Parse `<yield><partial>...</partial><expertise>...</expertise></yield>`
fn parse_explicit_yield(output: &str) -> Option<MetaSignal> {
    let start_tag = "<yield>";
    let end_tag = "</yield>";
    let start_idx = output.find(start_tag)?;
    let end_idx = output.find(end_tag)?;

    let inner = &output[start_idx + start_tag.len()..end_idx];

    let partial_progress = extract_tag_content(inner, "partial");
    let suggested_expertise = extract_all_tags(inner, "expertise");

    Some(MetaSignal::Yield {
        partial_progress,
        suggested_expertise,
    })
}

/// Parse `<thinking direction="...">...</thinking>`
fn parse_explicit_thinking(output: &str) -> Option<MetaSignal> {
    let start_idx = output.find("<thinking")?;
    let end_tag = "</thinking>";
    let end_idx = output.find(end_tag)?;

    let tag_content = &output[start_idx..end_idx + end_tag.len()];

    let direction = extract_attribute(tag_content, "direction").unwrap_or_default();

    let estimated_steps =
        extract_attribute(tag_content, "steps").and_then(|s| s.parse::<u32>().ok());

    // If no direction attribute, use inner text
    let direction = if direction.is_empty() {
        let inner_start = tag_content.find('>')? + 1;
        let inner = &tag_content[inner_start..tag_content.len() - end_tag.len()];
        inner.trim().to_string()
    } else {
        direction
    };

    if direction.is_empty() {
        return None;
    }

    Some(MetaSignal::Thinking {
        direction,
        estimated_steps,
    })
}

// ---------------------------------------------------------------------------
// Implicit signal detection
// ---------------------------------------------------------------------------

/// Detect uncertainty from natural language patterns.
fn detect_uncertainty_patterns(output: &str) -> Option<MetaSignal> {
    let lower = output.to_lowercase();
    let matched: Vec<&&str> = UNCERTAINTY_PATTERNS
        .iter()
        .filter(|p| lower.contains(&p.to_lowercase()))
        .collect();

    if matched.is_empty() {
        return None;
    }

    Some(MetaSignal::Uncertain {
        partial_answer: Some(output.to_string()),
        missing_information: vec![],
        would_help: vec![],
    })
}

/// Detect stuck signals from natural language patterns.
fn detect_stuck_patterns(output: &str) -> Option<MetaSignal> {
    let lower = output.to_lowercase();
    let matched: Vec<&&str> = STUCK_PATTERNS
        .iter()
        .filter(|p| lower.contains(&p.to_lowercase()))
        .collect();

    if matched.is_empty() {
        return None;
    }

    Some(MetaSignal::Stuck {
        attempts: vec![],
        hypothesis: None,
        request: StuckRequest::HumanIntervention {
            reason: "Implicit stuck signal detected".to_string(),
        },
    })
}

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

/// Extract an attribute value from an opening tag: `<tag attr="value">`.
fn extract_attribute(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract the text content of a single XML tag.
fn extract_tag_content(text: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = text.find(&start_tag)? + start_tag.len();
    let end = text.find(&end_tag)?;
    Some(text[start..end].trim().to_string())
}

/// Extract all occurrences of a tag's text content.
fn extract_all_tags(text: &str, tag: &str) -> Vec<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start) = text[search_from..].find(&start_tag) {
        let abs_start = search_from + start + start_tag.len();
        if let Some(end) = text[abs_start..].find(&end_tag) {
            let content = text[abs_start..abs_start + end].trim().to_string();
            results.push(content);
            search_from = abs_start + end + end_tag.len();
        } else {
            break;
        }
    }

    results
}
