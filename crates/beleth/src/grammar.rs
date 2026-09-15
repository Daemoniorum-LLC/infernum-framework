//! GBNF grammars that constrain the agentic loop's output envelope.
//!
//! # The problem this solves
//!
//! Recomputed from the 126 completions captured from Qwen2.5-Coder-14B-Instruct
//! in `tests/fixtures/issue_70_14b_completions.json`:
//!
//! | | |
//! |---|---|
//! | completions containing a `<tool_call>` tag | **0 of 126** |
//! | opening with `<answer` | 74 |
//! | opening with `<yield>` | 33 |
//! | opening with `<stuck>` | 9 |
//! | opening with prose or a fence | 10 |
//!
//! The model almost always produced *correct* `{"name":..,"arguments":..}` JSON.
//! It never once put it in the envelope the system prompt asked for. The 80.0%
//! (72/90) figure quoted for Phase A in #70/#71 is the **production detector's
//! recovery rate** over that corpus — [`QwenToolCallDetector`] falls back to a
//! wrapper-agnostic scan — not envelope fidelity, which was 0%.
//!
//! That distinction decides the shape of this grammar. A grammar whose
//! `<answer>` envelope admitted a JSON object would be satisfied by the exact
//! bytes the model already emits, and would change nothing.
//!
//! [`QwenToolCallDetector`]: crate::agentic_loop::executor::QwenToolCallDetector
//!
//! # One envelope, measured into existence
//!
//! ```text
//! root      ::= tool-call+
//! tool-call ::= "<tool_call>" call "</tool_call>"
//! call      ::= call-read-file | … | call-final-answer
//! ```
//!
//! **There is no `<answer>` alternative, and that is the whole design.** Two
//! earlier drafts kept one, and both were measured on Qwen2.5-Coder-14B through
//! `llama-server` before this one was written:
//!
//! | grammar | emission (`<tool_call>` emitted) | right tool |
//! |---|---|---|
//! | none (baseline) | 0/30 | 27/30 *(via the lenient detector)* |
//! | `root ::= prose? (tool-call+ \| answer)` | 0/30 | 27/30 |
//! | `root ::= tool-call+ \| answer`, `answer-body ::= [^<{]+` | 0/30 | **0/30** |
//!
//! The second draft excluded `{` from the answer body so call JSON could not
//! hide there. The model opened `<answer` anyway — it picks the envelope at the
//! *first token*, and no constraint downstream of that choice can undo it — and
//! then, unable to emit an object, narrated its intent in prose instead:
//! `<answer confidence="0.9">I will read the contents of the Cargo.toml
//! file.</answer>`. That is strictly worse than the baseline, because the
//! detector could at least recover the JSON the unconstrained model emitted.
//!
//! A grammar can only remove an alternative, never re-rank one. So the
//! `<answer>` alternative is gone: finishing is a call to the synthetic
//! [`FINAL_ANSWER_TOOL`], which the loop intercepts. That leaves exactly one
//! envelope, and:
//!
//! - **envelope fidelity is 100% by construction** — there is nowhere else to go;
//! - `<yield>...</yield>` is unreachable: no tag but `<tool_call>` exists;
//! - a hallucinated tool name is unreachable — the alternatives come from the
//!   registry;
//! - a wrong key is unreachable: `"function_name"` appears in 3 captures and the
//!   call rule admits only the literal `"name"`;
//! - a call missing a required argument is unreachable, because required
//!   properties are non-optional in the corresponding `args-*` rule;
//! - an answer may contain braces again, because it lives inside a JSON string.
//!
//! # What this costs
//!
//! The model cannot emit free prose at all while the grammar is on: every reply
//! is a call, and thinking-aloud has to happen inside `final_answer`'s `answer`
//! argument or not at all. For a loop whose every turn is "act or conclude"
//! that is the contract anyway, but it is a real narrowing and it is why
//! [`with_tool_call_grammar(false)`](crate::agentic_loop::ExecutorConfig::with_tool_call_grammar)
//! exists.
//!
//! # Why `<uncertain>`, `<stuck>` and `<yield>` are not in it
//!
//! The loop treats `<stuck>` and `<yield>` as terminal (see
//! [`executor`](crate::agentic_loop::executor)), so a turn-one echo of the
//! prompt's own example ends a run with `completed = false` before it has
//! begun. 42 of the 126 captures open with one of those two tags. Offering four
//! similarly-styled tags in one paragraph is what #70 diagnosed as the cause,
//! so the constrained prompt built by [`compose_system_prompt`] offers none of
//! them and the grammar admits none of them.
//!
//! A model that cannot make progress now says so inside `final_answer`, where
//! the loop records it as an answer rather than discarding it. The deliberate
//! trade: explicit yield-to-a-specialist signalling is unavailable while the
//! grammar is on. Callers that need it set
//! [`with_tool_call_grammar(false)`](crate::agentic_loop::ExecutorConfig::with_tool_call_grammar).
//!
//! # Dialect
//!
//! llama.cpp GBNF. Rule names may contain only `[a-zA-Z0-9-]` — notably **not**
//! underscore — so tool names are sanitised and index-suffixed by
//! [`rule_ident`], which keeps them both readable and collision-free.

use std::fmt::Write as _;

use infernum_core::sampling::GrammarConstraint;
use serde_json::Value;

use crate::tool::ToolRegistry;

/// Shared JSON-value rules, appended once per grammar.
///
/// Deliberately a *subset* of the JSON grammar in
/// [`infernum_core::sampling`]: tool arguments are flat-ish objects, and a
/// tighter grammar is a tighter constraint. `value` still admits nested
/// objects and arrays so a tool taking structured arguments is not blocked.
const JSON_RULES: &str = r#"
value   ::= object | array | string | number | boolean | null
object  ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* ws )? "}"
array   ::= "[" ws ( value ( ws "," ws value )* ws )? "]"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" escape
escape  ::= ["\\/bfnrt] | "u" hex hex hex hex
hex     ::= [0-9a-fA-F]
number  ::= integer frac? exp?
integer ::= "-"? ( "0" | [1-9] [0-9]* )
frac    ::= "." [0-9]+
exp     ::= [eE] [-+]? [0-9]+
boolean ::= "true" | "false"
null    ::= "null"
ws      ::= [ \t\n\r]*
"#;

/// Builds a GBNF constraining output to this registry's tool-call envelope.
///
/// Returns `None` for an empty registry: with no tools the only reachable call
/// would be `final_answer`, which would silently convert the loop into a
/// single-shot completion. `None` means "send the request unconstrained",
/// which is the honest behaviour for that case.
///
/// The grammar is derived from [`Tool::parameters_schema`](crate::tool::Tool::parameters_schema),
/// so registering a tool is all it takes for the grammar to admit calls to it.
#[must_use]
pub fn tool_call_grammar(tools: &ToolRegistry) -> Option<GrammarConstraint> {
    Some(GrammarConstraint::Gbnf(tool_call_gbnf(tools)?))
}

/// The GBNF source behind [`tool_call_grammar`], exposed for tests and for
/// operators who want to read what their model is actually being held to.
///
/// Returns `None` for an empty registry — see [`tool_call_grammar`].
#[must_use]
pub fn tool_call_gbnf(tools: &ToolRegistry) -> Option<String> {
    // Sorted so the grammar is byte-identical across runs: the registry is a
    // HashMap, and a grammar that changed between two runs of the same eval
    // would make those runs incomparable for no reason.
    let mut names: Vec<&str> = tools.list();
    if names.is_empty() {
        return None;
    }
    names.sort_unstable();

    let mut out = String::new();
    out.push_str(
        "# Generated by beleth::grammar::tool_call_gbnf — do not hand-edit.\n\
         # Output must be one or more <tool_call> envelopes and nothing else.\n\
         # Finishing is a call to `final_answer`, which the loop intercepts.\n",
    );
    out.push_str("root ::= tool-call ( ws tool-call )*\n");
    out.push_str("tool-call ::= \"<tool_call>\" ws call ws \"</tool_call>\"\n");

    // One `call-*` alternative per registered tool. Generating the name
    // alternatives from the registry is what makes a hallucinated tool name
    // unreachable rather than merely unlikely.
    let idents: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, n)| rule_ident(n, i))
        .collect();
    let mut alternatives: Vec<String> = idents.iter().map(|id| format!("call-{id}")).collect();
    let synthesise_final_answer = !names.contains(&FINAL_ANSWER_TOOL);
    if synthesise_final_answer {
        alternatives.push("call-final-answer".to_string());
    }
    let _ = writeln!(out, "call ::= {}", alternatives.join(" | "));

    for (name, ident) in names.iter().zip(&idents) {
        let schema = tools
            .get(name)
            .map(|t| t.parameters_schema())
            .unwrap_or(Value::Null);

        let _ = writeln!(out, "\n# tool: {name}");
        let _ = writeln!(
            out,
            "call-{ident} ::= \"{{\" ws \"\\\"name\\\"\" ws \":\" ws \"{}\" ws \",\" ws \
             \"\\\"arguments\\\"\" ws \":\" ws args-{ident} ws \"}}\"",
            gbnf_quoted_json_string(name)
        );
        let _ = writeln!(out, "args-{ident} ::= {}", args_rule(&schema));
    }

    if synthesise_final_answer {
        out.push_str(FINAL_ANSWER_RULES);
    }

    out.push_str(JSON_RULES);
    Some(out)
}

/// Name of the synthetic tool a model calls to finish.
///
/// Not a registered [`Tool`](crate::tool::Tool): nothing executes it. The
/// loop intercepts a call by this name and records it as the final answer.
/// A registry that defines a real tool with this name wins — the synthetic
/// rule is then not generated, and the call dispatches normally.
pub const FINAL_ANSWER_TOOL: &str = "final_answer";

/// The `final_answer` call rule.
///
/// `confidence` is optional and is a plain JSON number here, not the
/// `"0.9"`-style attribute the retired `<answer>` tag used.
const FINAL_ANSWER_RULES: &str = r#"
# tool: final_answer (synthetic — the loop intercepts it, nothing executes it)
call-final-answer ::= "{" ws "\"name\"" ws ":" ws "\"final_answer\"" ws "," ws "\"arguments\"" ws ":" ws args-final-answer ws "}"
args-final-answer ::= "{" ws "\"answer\"" ws ":" ws string ( ws "," ws "\"confidence\"" ws ":" ws number )? ws "}"
"#;

/// Right-hand side of one tool's `args-*` rule, derived from its JSON Schema.
///
/// Required properties come first, in the order `required` lists them, and are
/// unconditional — which is what makes "called the right tool but omitted a
/// required argument" unreachable rather than merely unlikely. Optional
/// properties follow in the schema's own (deterministic) order, each able to
/// be present or absent.
///
/// A schema this cannot read falls back to the generic `object` rule: a weaker
/// constraint, never a wrong one.
///
/// # Commas
///
/// The fiddly part is that a comma belongs *between* two present pairs, and
/// GBNF has no way to say "join these". Grouping each optional pair with a
/// leading comma emits `{,"a":1}` when it is the first one present; with a
/// trailing comma it emits `{"a":1,}`. Neither is JSON, and either would make
/// the grammar itself the source of malformed calls this ticket exists to
/// remove. So when at least one required pair anchors the list, every optional
/// pair carries a leading comma; and when nothing is required,
/// [`optional_tail_alternation`] enumerates which pair comes first.
fn args_rule(schema: &Value) -> String {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        // No usable property list: constrain to "some JSON object" and let
        // the tool's own parameter validation do the rest.
        return "object".to_string();
    };

    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let pair = |key: &str| {
        format!(
            "\"{}\" ws \":\" ws {}",
            gbnf_quoted_json_string(key),
            value_rule_for(properties.get(key))
        )
    };

    let required_pairs: Vec<String> = required
        .iter()
        .filter(|k| properties.contains_key(**k))
        .map(|k| pair(k))
        .collect();
    let optional_pairs: Vec<String> = properties
        .keys()
        .filter(|k| !required.contains(&k.as_str()))
        .map(|k| pair(k))
        .collect();

    if required_pairs.is_empty() && optional_pairs.is_empty() {
        return "\"{\" ws \"}\"".to_string();
    }

    let body = if required_pairs.is_empty() {
        match optional_tail_alternation(&optional_pairs) {
            Some(alternation) => format!("( {alternation} )?"),
            // Too many optional properties to enumerate without the rule text
            // growing quadratically. Widen rather than emit something huge.
            None => return "object".to_string(),
        }
    } else {
        let mut parts = vec![required_pairs.join(" ws \",\" ws ")];
        for p in &optional_pairs {
            parts.push(format!("( ws \",\" ws {p} )?"));
        }
        parts.join(" ")
    };

    format!("\"{{\" ws {body} ws \"}}\"")
}

/// Enumerates every way a list of all-optional pairs can start.
///
/// With nothing required there is no anchor for comma placement, so each
/// alternative names the first pair present and makes the rest optional after
/// it: `a (,b)? (,c)? | b (,c)? | c`. Key *order* stays fixed, which is what
/// keeps this linear in alternatives rather than factorial.
///
/// Returns `None` past [`MAX_OPTIONAL_ALTERNATION`] properties, where the
/// generated text stops being worth reading; the caller widens to `object`.
fn optional_tail_alternation(pairs: &[String]) -> Option<String> {
    if pairs.is_empty() || pairs.len() > MAX_OPTIONAL_ALTERNATION {
        return None;
    }
    let alternatives: Vec<String> = (0..pairs.len())
        .map(|i| {
            let mut alt = vec![pairs[i].clone()];
            for p in &pairs[i + 1..] {
                alt.push(format!("( ws \",\" ws {p} )?"));
            }
            alt.join(" ")
        })
        .collect();
    Some(alternatives.join(" | "))
}

/// Above this many all-optional properties, widen to `object` instead of
/// enumerating. The alternation is O(n²) in text size; no built-in tool comes
/// close, and the ceiling keeps a future one from generating a wall of GBNF.
const MAX_OPTIONAL_ALTERNATION: usize = 8;

/// Maps a JSON Schema property type onto a rule from [`JSON_RULES`].
///
/// Unknown or absent types widen to `value`, which admits anything JSON can
/// express. Narrowing a type we did not understand would block a legitimate
/// call.
fn value_rule_for(property: Option<&Value>) -> &'static str {
    match property.and_then(|p| p.get("type")).and_then(Value::as_str) {
        Some("string") => "string",
        Some("integer") => "integer",
        Some("number") => "number",
        Some("boolean") => "boolean",
        Some("array") => "array",
        Some("object") => "object",
        _ => "value",
    }
}

/// A GBNF rule identifier for `name`, unique by construction.
///
/// llama.cpp's grammar parser accepts only `[a-zA-Z0-9-]` in a rule name, so
/// `read_file` cannot be used directly. Every other character becomes `-`, and
/// the tool's index is appended so two tools that sanitise to the same string
/// still get distinct rules.
fn rule_ident(name: &str, index: usize) -> String {
    let sanitised: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{sanitised}-{index}")
}

/// Renders `s` as a GBNF literal containing a quoted JSON string.
///
/// Two escaping layers stack here and it is easy to get wrong: the JSON string
/// needs its own `"` quotes, and the whole thing sits inside a GBNF double
/// quoted literal, so each of those quotes must be written `\"`. The result is
/// spliced between `"` characters by the caller.
fn gbnf_quoted_json_string(s: &str) -> String {
    let mut out = String::from("\\\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\\\\\""),
            '\\' => out.push_str("\\\\\\\\"),
            _ => out.push(c),
        }
    }
    out.push_str("\\\"");
    out
}

/// Builds the loop's system prompt.
///
/// Shared by [`LoopExecutor`](crate::agentic_loop::LoopExecutor) and by the
/// `toolcall-eval` harness so the two cannot drift: the harness previously
/// kept its own copy of this string, which made "what the eval measures" and
/// "what the loop elicits" two separately-maintained facts.
///
/// When `constrained` is true the prompt advertises exactly the two envelopes
/// [`tool_call_grammar`] permits. That alignment is the point — a prompt
/// offering tags the grammar forbids sets the model fighting the sampler, and
/// #72 is what that looks like from the outside.
#[must_use]
pub fn compose_system_prompt(base: &str, tools: &ToolRegistry, constrained: bool) -> String {
    let tool_desc = tools.to_qwen_native_description();
    if constrained {
        // The `final_answer` instruction goes BEFORE the tool block, because
        // `to_qwen_native_description` ends with the `<tool_call>` template and
        // whatever sits last is what the model copies — that recency is the
        // mechanism behind #72. Every reply is a call, so there is nothing else
        // to describe.
        format!(
            "{base}\n\n\
             Every reply must be exactly one or more tool calls and nothing else. \
             When you have finished, call the `final_answer` function, passing your \
             answer as its `answer` argument and optionally a `confidence` between \
             0.0 and 1.0. If you cannot determine something, say so in that answer. \
             Do not narrate what you are about to do — call the function instead.\n\n\
             {tool_desc}"
        )
    } else {
        format!(
            "{base}\n\n{tool_desc}\n\n\
             You may express uncertainty with <uncertain>...</uncertain>, \
             signal you're stuck with <stuck>...</stuck>, \
             yield with <yield>...</yield>, \
             or provide a final answer with <answer confidence=\"0.9\">...</answer>."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gbnf() -> String {
        tool_call_gbnf(&ToolRegistry::with_code_tools()).expect("code registry is not empty")
    }

    #[test]
    fn empty_registry_yields_no_grammar() {
        // Not an empty grammar: a grammar admitting only <answer> would turn
        // the loop into a single-shot completion without saying so.
        assert!(tool_call_gbnf(&ToolRegistry::new()).is_none());
        assert!(tool_call_grammar(&ToolRegistry::new()).is_none());
    }

    #[test]
    fn every_registered_tool_gets_a_call_alternative() {
        let tools = ToolRegistry::with_code_tools();
        let g = tool_call_gbnf(&tools).unwrap();
        for name in tools.list() {
            assert!(
                g.contains(&format!("# tool: {name}")),
                "{name} is registered but has no rule in the grammar"
            );
            assert!(
                g.contains(&format!("\\\"{name}\\\"")),
                "{name} is registered but its literal name is not in the grammar, \
                 so the model could not call it"
            );
        }
    }

    #[test]
    fn rule_names_use_only_characters_llama_cpp_accepts() {
        // llama.cpp's is_word_char is [a-zA-Z0-9-]: an underscore in a rule
        // name is a parse error at the server, which surfaces as an opaque
        // 400 rather than as anything naming the grammar.
        for line in gbnf().lines() {
            let Some((lhs, _)) = line.split_once("::=") else {
                continue;
            };
            let lhs = lhs.trim();
            assert!(
                lhs.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "rule name {lhs:?} contains a character llama.cpp will reject"
            );
        }
    }

    #[test]
    fn required_arguments_are_not_optional_in_the_grammar() {
        let g = gbnf();
        let line = g
            .lines()
            .find(|l| l.starts_with("args-read-file-"))
            .expect("read_file must have an args rule");

        // `path` is required, so it appears with no `?` group; `offset` and
        // `limit` are optional, so they do. This is the mechanism behind
        // Phase A's schema-valid number, so it is worth pinning exactly.
        assert!(
            line.contains(r#""\"path\"" ws ":" ws string"#),
            "required `path` must be unconditional: {line}"
        );
        assert!(
            line.contains(r#"( ws "," ws "\"offset\"" ws ":" ws integer )?"#),
            "optional `offset` must be an optional group: {line}"
        );
        assert!(
            !line.contains(r#"( ws "," ws "\"path\"""#),
            "`path` must never appear inside an optional group: {line}"
        );
    }

    #[test]
    fn all_optional_properties_are_still_constrained_by_key() {
        // `list_files` declares no `required` at all. The keys are still
        // pinned — a hallucinated argument name stays unreachable — and every
        // subset the grammar admits is valid JSON, with no leading or
        // trailing comma.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}}
        });
        let rule = args_rule(&schema);
        assert_eq!(
            rule,
            r#""{" ws ( "\"a\"" ws ":" ws string ( ws "," ws "\"b\"" ws ":" ws string )? | "\"b\"" ws ":" ws string )? ws "}""#
        );
    }

    #[test]
    fn too_many_optional_properties_widen_to_object() {
        let props: serde_json::Map<String, Value> = (0..=MAX_OPTIONAL_ALTERNATION)
            .map(|i| (format!("k{i}"), serde_json::json!({"type": "string"})))
            .collect();
        let schema = serde_json::json!({"type": "object", "properties": props});
        assert_eq!(args_rule(&schema), "object");
    }

    #[test]
    fn no_generated_args_rule_can_emit_a_dangling_comma() {
        // The failure mode this grammar exists to prevent must not be
        // reintroduced by the grammar itself.
        let g = gbnf();
        for line in g.lines().filter(|l| l.starts_with("args-")) {
            assert!(
                !line.contains(r#"ws "{" ws "," "#) && !line.contains(r#"" ws "," ws "}""#),
                "rule can emit a dangling comma: {line}"
            );
        }
    }

    #[test]
    fn a_schema_without_properties_falls_back_to_object() {
        assert_eq!(args_rule(&Value::Null), "object");
        assert_eq!(args_rule(&serde_json::json!({"type": "string"})), "object");
    }

    #[test]
    fn no_meta_signal_tag_is_reachable_under_the_grammar() {
        // The #72 tags must appear nowhere in the language. Their absence is
        // what makes `<yield>...</yield>` unreachable: `<y` is not a prefix
        // of any literal the grammar defines.
        let g = gbnf();
        for tag in ["<yield>", "<stuck>", "<uncertain>", "<thinking>", "<answer"] {
            assert!(
                !g.contains(tag),
                "{tag} must not be in the grammar — the loop treats it as \
                 terminal, and an echo of the prompt's own example would end \
                 the run on turn one (infernum-framework#72)"
            );
        }
    }

    #[test]
    fn the_tool_call_envelope_is_the_only_one() {
        // The property the first two drafts lacked, and the reason they
        // measured worse than no grammar at all: as long as `<answer>` was a
        // peer alternative at position 0, Qwen2.5-Coder-14B took it 30/30 and
        // never emitted a `<tool_call>` tag. A grammar can remove an
        // alternative; it cannot re-rank one. See the module docs for the
        // measured table.
        let g = gbnf();
        assert_eq!(
            g.lines().find(|l| l.starts_with("root ::=")),
            Some("root ::= tool-call ( ws tool-call )*"),
            "root must admit tool calls and nothing else"
        );
        assert!(
            !g.contains("<answer"),
            "no <answer> envelope may survive in the grammar"
        );
    }

    #[test]
    fn finishing_is_a_call_not_a_tag() {
        let g = gbnf();
        assert!(g.contains("call-final-answer"));
        assert!(
            g.contains(r#"args-final-answer ::= "{" ws "\"answer\"" ws ":" ws string"#),
            "final_answer must take a required `answer` string"
        );
        // Because the answer now lives inside a JSON string, braces are legal
        // again — the restriction the second draft had to impose is gone.
        assert!(g.contains("char    ::= [^\"\\\\] | \"\\\\\" escape"));
    }

    #[test]
    fn a_real_final_answer_tool_wins_over_the_synthetic_one() {
        // A registry defining its own `final_answer` must not get two
        // conflicting rules for the same name.
        use crate::tool::Tool;
        use async_trait::async_trait;
        use infernum_core::Result;

        struct RealFinalAnswer;
        #[async_trait]
        impl Tool for RealFinalAnswer {
            fn name(&self) -> &str {
                super::FINAL_ANSWER_TOOL
            }
            fn description(&self) -> &str {
                "a real one"
            }
            fn parameters_schema(&self) -> Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                })
            }
            fn risk_level(&self) -> crate::tool::RiskLevel {
                crate::tool::RiskLevel::Safe
            }
            async fn execute(
                &self,
                _params: Value,
                _ctx: &crate::tool::ToolContext,
            ) -> Result<crate::tool::ToolResult> {
                unreachable!("never executed in this test")
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(std::sync::Arc::new(RealFinalAnswer));
        let g = tool_call_gbnf(&tools).unwrap();
        assert!(
            !g.contains("(synthetic"),
            "the synthetic rule must not be emitted alongside a real tool of \
             the same name: {g}"
        );
        // Exactly one `call ::=` alternative, pointing at the real tool's
        // index-suffixed rule rather than the synthetic `call-final-answer`.
        assert_eq!(
            g.lines().find(|l| l.starts_with("call ::=")),
            Some("call ::= call-final-answer-0")
        );
        // …and it uses the REAL tool's schema, not the synthetic `answer` arg.
        assert!(g.contains(r#""\"text\"" ws ":" ws string"#));
        assert!(!g.contains(r#""\"answer\"" ws ":" ws string"#));
    }

    #[test]
    fn the_grammar_is_byte_stable_across_builds() {
        // The registry is a HashMap. Two eval runs whose grammars differed
        // would not be comparable, and the difference would be invisible.
        let a = tool_call_gbnf(&ToolRegistry::with_code_tools()).unwrap();
        let b = tool_call_gbnf(&ToolRegistry::with_code_tools()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn constrained_prompt_drops_the_tags_the_grammar_forbids() {
        let tools = ToolRegistry::with_code_tools();
        let constrained = compose_system_prompt("base", &tools, true);
        for tag in ["<yield>", "<stuck>", "<uncertain>"] {
            assert!(
                !constrained.contains(tag),
                "the constrained prompt must not offer {tag}: the grammar \
                 forbids it, and offering it sets the model against the sampler"
            );
        }
        assert!(
            !constrained.contains("<answer"),
            "the constrained prompt must not offer <answer>: the grammar has no \
             such envelope, and offering it is what makes the model open one"
        );
        assert!(constrained.contains("final_answer"));
        // The <tool_call> template must be the LAST thing the model sees.
        let tool_template = constrained.rfind("<tool_call>").unwrap();
        let answer_mention = constrained.rfind("final_answer").unwrap();
        assert!(
            tool_template > answer_mention,
            "the <tool_call> template must come last; whatever sits last is what \
             the model copies, which is the mechanism behind issue #72"
        );
    }

    #[test]
    fn unconstrained_prompt_is_unchanged() {
        // The pre-#17 wording, kept verbatim so an unconstrained run stays
        // comparable with the numbers already recorded for #70/#71.
        let tools = ToolRegistry::with_code_tools();
        let p = compose_system_prompt("base", &tools, false);
        assert!(p.contains("<uncertain>...</uncertain>"));
        assert!(p.contains("<stuck>...</stuck>"));
        assert!(p.contains("<yield>...</yield>"));
    }

    #[test]
    fn rule_idents_are_unique_even_when_names_sanitise_alike() {
        // `a_b` and `a-b` both sanitise to `a-b`; the index keeps them apart.
        assert_ne!(rule_ident("a_b", 0), rule_ident("a-b", 1));
    }

    #[test]
    fn json_string_literals_escape_both_layers() {
        assert_eq!(gbnf_quoted_json_string("read_file"), "\\\"read_file\\\"");
        // A quote inside the name needs escaping for JSON *and* for GBNF.
        assert_eq!(gbnf_quoted_json_string("a\"b"), "\\\"a\\\\\\\"b\\\"");
    }
}
