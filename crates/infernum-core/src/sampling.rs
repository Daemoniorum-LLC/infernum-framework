//! Sampling parameters for text generation.

use serde::{Deserialize, Serialize};

/// Grammar constraint for structured generation.
///
/// Grammars constrain the model's output to follow specific syntactic rules,
/// enabling reliable structured output like JSON, XML, or custom formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GrammarConstraint {
    /// GBNF (GGML BNF) grammar string for llama.cpp.
    ///
    /// Example:
    /// ```text
    /// root ::= object
    /// object ::= "{" pair ("," pair)* "}"
    /// pair ::= string ":" value
    /// ```
    Gbnf(String),

    /// JSON mode - constrains output to valid JSON.
    /// Equivalent to the standard JSON GBNF grammar.
    Json,

    /// JSON Schema constraint - output must match the provided schema.
    /// The schema is converted to GBNF at runtime.
    JsonSchema(String),
}

impl GrammarConstraint {
    /// Create a GBNF grammar constraint.
    pub fn gbnf(grammar: impl Into<String>) -> Self {
        Self::Gbnf(grammar.into())
    }

    /// Create a JSON mode constraint.
    pub fn json() -> Self {
        Self::Json
    }

    /// Create a JSON Schema constraint.
    pub fn json_schema(schema: impl Into<String>) -> Self {
        Self::JsonSchema(schema.into())
    }

    /// Get the GBNF grammar string for this constraint.
    ///
    /// For `Json` mode, returns a standard JSON grammar.
    /// For `JsonSchema`, converts the schema to GBNF (simplified).
    pub fn to_gbnf(&self) -> String {
        match self {
            Self::Gbnf(grammar) => grammar.clone(),
            Self::Json => Self::json_gbnf().to_string(),
            Self::JsonSchema(schema) => Self::schema_to_gbnf(schema),
        }
    }

    /// Standard JSON GBNF grammar.
    fn json_gbnf() -> &'static str {
        r#"root ::= value
value ::= object | array | string | number | boolean | null
object ::= "{" ws (pair ("," ws pair)*)? "}" ws
pair ::= string ":" ws value
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" char* "\"" ws
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= "-"? integer fraction? exponent? ws
integer ::= "0" | [1-9] [0-9]*
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
boolean ::= ("true" | "false") ws
null ::= "null" ws
ws ::= [ \t\n\r]*"#
    }

    /// Convert JSON schema to GBNF (simplified implementation).
    ///
    /// This handles basic object schemas with required string/number/boolean properties.
    /// For complex schemas, users should provide GBNF directly.
    fn schema_to_gbnf(schema: &str) -> String {
        // Parse the JSON schema
        let Ok(value) = serde_json::from_str::<serde_json::Value>(schema) else {
            // Fall back to generic JSON if schema is invalid
            return Self::json_gbnf().to_string();
        };

        // Check if it's an object type with properties
        if value.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Self::json_gbnf().to_string();
        }

        let Some(properties) = value.get("properties").and_then(|p| p.as_object()) else {
            return Self::json_gbnf().to_string();
        };

        // Build a grammar for this specific object schema
        let mut rules = Vec::new();
        let mut pair_rules = Vec::new();

        for (key, prop) in properties {
            let prop_type = prop.get("type").and_then(|t| t.as_str()).unwrap_or("string");
            let value_rule = match prop_type {
                "string" => "string",
                "number" | "integer" => "number",
                "boolean" => "boolean",
                "null" => "null",
                "array" => "array",
                "object" => "object",
                _ => "value",
            };
            pair_rules.push(format!("\"\\\"{}\\\":\" ws {}", key, value_rule));
        }

        // Build the root rule for this object
        let pairs = pair_rules.join(" \",\" ws ");
        rules.push(format!("root ::= \"{{\" ws {} \"}}\" ws", pairs));

        // Add standard JSON rules
        rules.push(r#"value ::= object | array | string | number | boolean | null"#.to_string());
        rules.push(r#"object ::= "{" ws (pair ("," ws pair)*)? "}" ws"#.to_string());
        rules.push(r#"pair ::= string ":" ws value"#.to_string());
        rules.push(r#"array ::= "[" ws (value ("," ws value)*)? "]" ws"#.to_string());
        rules.push(r#"string ::= "\"" char* "\"" ws"#.to_string());
        rules.push(r#"char ::= [^"\\] | "\\" escape"#.to_string());
        rules.push(r#"escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]"#.to_string());
        rules.push(r#"number ::= "-"? integer fraction? exponent? ws"#.to_string());
        rules.push(r#"integer ::= "0" | [1-9] [0-9]*"#.to_string());
        rules.push(r#"fraction ::= "." [0-9]+"#.to_string());
        rules.push(r#"exponent ::= [eE] [+-]? [0-9]+"#.to_string());
        rules.push(r#"boolean ::= ("true" | "false") ws"#.to_string());
        rules.push(r#"null ::= "null" ws"#.to_string());
        rules.push(r#"ws ::= [ \t\n\r]*"#.to_string());

        rules.join("\n")
    }
}

/// Parameters controlling text generation sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = greedy, higher = more random).
    /// Default: 1.0
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    /// Default: 1.0
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled).
    /// Default: 0
    #[serde(default)]
    pub top_k: u32,

    /// Minimum probability for min-p sampling (0.0 = disabled).
    /// Default: 0.0
    #[serde(default)]
    pub min_p: f32,

    /// Repetition penalty (1.0 = no penalty).
    /// Default: 1.0
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Presence penalty (-2.0 to 2.0).
    /// Default: 0.0
    #[serde(default)]
    pub presence_penalty: f32,

    /// Frequency penalty (-2.0 to 2.0).
    /// Default: 0.0
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Stop sequences that halt generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Maximum number of tokens to generate.
    /// Default: 256
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Random seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Grammar constraint for structured output (JSON mode, GBNF, etc.).
    ///
    /// When set, the model's output will be constrained to follow the grammar,
    /// guaranteeing syntactically valid structured output.
    #[serde(default)]
    pub grammar: Option<GrammarConstraint>,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_max_tokens() -> u32 {
    256
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: Vec::new(),
            max_tokens: 256,
            seed: None,
            grammar: None,
        }
    }
}

impl SamplingParams {
    /// Creates greedy sampling parameters (temperature = 0).
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Creates balanced sampling parameters.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            ..Default::default()
        }
    }

    /// Creates creative sampling parameters.
    #[must_use]
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }

    /// Sets the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the top-p value.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets the top-k value.
    #[must_use]
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the maximum tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Adds a stop sequence.
    #[must_use]
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    /// Sets the random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the repetition penalty (1.0 = no penalty, >1.0 = discourage repetition).
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Sets the presence penalty (-2.0 to 2.0).
    #[must_use]
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Sets the frequency penalty (-2.0 to 2.0).
    #[must_use]
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Sets the min-p threshold (0.0 to 1.0).
    #[must_use]
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Sets a grammar constraint for structured output.
    #[must_use]
    pub fn with_grammar(mut self, grammar: GrammarConstraint) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Enables JSON mode - constrains output to valid JSON.
    #[must_use]
    pub fn with_json_mode(mut self) -> Self {
        self.grammar = Some(GrammarConstraint::Json);
        self
    }

    /// Sets a GBNF grammar constraint.
    #[must_use]
    pub fn with_gbnf(mut self, grammar: impl Into<String>) -> Self {
        self.grammar = Some(GrammarConstraint::Gbnf(grammar.into()));
        self
    }

    /// Sets a JSON Schema constraint.
    #[must_use]
    pub fn with_json_schema(mut self, schema: impl Into<String>) -> Self {
        self.grammar = Some(GrammarConstraint::JsonSchema(schema.into()));
        self
    }

    /// Validates the sampling parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err("top_p must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err("min_p must be between 0.0 and 1.0".to_string());
        }
        if self.repetition_penalty < 0.0 {
            return Err("repetition_penalty must be non-negative".to_string());
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err("presence_penalty must be between -2.0 and 2.0".to_string());
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err("frequency_penalty must be between -2.0 and 2.0".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.min_p, 0.0);
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert_eq!(params.frequency_penalty, 0.0);
        assert!(params.stop_sequences.is_empty());
        assert_eq!(params.max_tokens, 256);
        assert!(params.seed.is_none());
        assert!(params.grammar.is_none());
    }

    #[test]
    fn test_greedy_params() {
        let params = SamplingParams::greedy();
        assert_eq!(params.temperature, 0.0);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_balanced_params() {
        let params = SamplingParams::balanced();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_creative_params() {
        let params = SamplingParams::creative();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 50);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_builder_methods() {
        let params = SamplingParams::default()
            .with_temperature(0.8)
            .with_top_p(0.9)
            .with_top_k(40)
            .with_max_tokens(512)
            .with_stop("END")
            .with_seed(42);

        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.stop_sequences, vec!["END".to_string()]);
        assert_eq!(params.seed, Some(42));
    }

    #[test]
    fn test_validate_valid_params() {
        let params = SamplingParams::default();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_validate_negative_temperature() {
        let params = SamplingParams::default().with_temperature(-0.1);
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("temperature"));
    }

    #[test]
    fn test_validate_top_p_out_of_range() {
        let params = SamplingParams {
            top_p: 1.5,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("top_p"));

        let params = SamplingParams {
            top_p: -0.1,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_min_p_out_of_range() {
        let params = SamplingParams {
            min_p: 1.5,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("min_p"));
    }

    #[test]
    fn test_validate_presence_penalty_out_of_range() {
        let params = SamplingParams {
            presence_penalty: 3.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("presence_penalty"));

        let params = SamplingParams {
            presence_penalty: -3.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_frequency_penalty_out_of_range() {
        let params = SamplingParams {
            frequency_penalty: 2.5,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("frequency_penalty"));
    }

    #[test]
    fn test_validate_zero_max_tokens() {
        let params = SamplingParams {
            max_tokens: 0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("max_tokens"));
    }

    #[test]
    fn test_validate_negative_repetition_penalty() {
        let params = SamplingParams {
            repetition_penalty: -0.5,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("repetition_penalty"));
    }

    #[test]
    fn test_serialization() {
        let params = SamplingParams::default()
            .with_temperature(0.7)
            .with_max_tokens(100);

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: SamplingParams = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.temperature, 0.7);
        assert_eq!(deserialized.max_tokens, 100);
    }

    #[test]
    fn test_deserialization_with_defaults() {
        let json = r#"{"temperature": 0.5}"#;
        let params: SamplingParams = serde_json::from_str(json).unwrap();

        assert_eq!(params.temperature, 0.5);
        // Check defaults are applied
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.max_tokens, 256);
    }

    #[test]
    fn test_multiple_stop_sequences() {
        let params = SamplingParams::default()
            .with_stop("END")
            .with_stop("STOP")
            .with_stop("\n\n");

        assert_eq!(params.stop_sequences.len(), 3);
        assert!(params.stop_sequences.contains(&"END".to_string()));
        assert!(params.stop_sequences.contains(&"STOP".to_string()));
        assert!(params.stop_sequences.contains(&"\n\n".to_string()));
    }

    // === Grammar Constraint Tests ===

    #[test]
    fn test_grammar_constraint_json_mode() {
        let params = SamplingParams::default().with_json_mode();
        assert!(params.grammar.is_some());
        assert_eq!(params.grammar, Some(GrammarConstraint::Json));
    }

    #[test]
    fn test_grammar_constraint_gbnf() {
        let grammar = r#"root ::= "hello" | "world""#;
        let params = SamplingParams::default().with_gbnf(grammar);
        assert!(params.grammar.is_some());
        assert_eq!(params.grammar, Some(GrammarConstraint::Gbnf(grammar.to_string())));
    }

    #[test]
    fn test_grammar_constraint_json_schema() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let params = SamplingParams::default().with_json_schema(schema);
        assert!(params.grammar.is_some());
        assert_eq!(params.grammar, Some(GrammarConstraint::JsonSchema(schema.to_string())));
    }

    #[test]
    fn test_grammar_constraint_with_grammar() {
        let constraint = GrammarConstraint::json();
        let params = SamplingParams::default().with_grammar(constraint);
        assert_eq!(params.grammar, Some(GrammarConstraint::Json));
    }

    #[test]
    fn test_grammar_json_to_gbnf() {
        let gbnf = GrammarConstraint::Json.to_gbnf();
        // Should contain key JSON grammar rules
        assert!(gbnf.contains("root ::= value"));
        assert!(gbnf.contains("object"));
        assert!(gbnf.contains("array"));
        assert!(gbnf.contains("string"));
        assert!(gbnf.contains("number"));
        assert!(gbnf.contains("boolean"));
        assert!(gbnf.contains("null"));
    }

    #[test]
    fn test_grammar_gbnf_passthrough() {
        let custom = r#"root ::= "yes" | "no""#;
        let constraint = GrammarConstraint::gbnf(custom);
        assert_eq!(constraint.to_gbnf(), custom);
    }

    #[test]
    fn test_grammar_json_schema_to_gbnf() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}"#;
        let constraint = GrammarConstraint::json_schema(schema);
        let gbnf = constraint.to_gbnf();

        // Should generate a root rule for this specific object
        assert!(gbnf.contains("root ::="));
        // Should contain property names
        assert!(gbnf.contains("name"));
        assert!(gbnf.contains("age"));
    }

    #[test]
    fn test_grammar_invalid_json_schema_fallback() {
        // Invalid JSON should fall back to generic JSON grammar
        let constraint = GrammarConstraint::json_schema("not valid json");
        let gbnf = constraint.to_gbnf();
        assert!(gbnf.contains("root ::= value"));
    }

    #[test]
    fn test_grammar_serialization() {
        let params = SamplingParams::default().with_json_mode();
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.grammar, Some(GrammarConstraint::Json));
    }

    #[test]
    fn test_grammar_deserialization_gbnf() {
        let json = r#"{"grammar": {"Gbnf": "root ::= value"}}"#;
        let params: SamplingParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.grammar, Some(GrammarConstraint::Gbnf("root ::= value".to_string())));
    }

    #[test]
    fn test_default_no_grammar() {
        let params = SamplingParams::default();
        assert!(params.grammar.is_none());
    }
}
