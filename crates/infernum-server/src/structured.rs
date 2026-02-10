//! Structured output support for JSON schema enforcement.
//!
//! This module provides support for constraining LLM outputs to follow
//! specific JSON schemas, enabling reliable structured data extraction.
//!
//! # Response Formats
//!
//! Two response format types are supported:
//!
//! - `json_object`: Ensures the output is valid JSON
//! - `json_schema`: Ensures the output matches a specific JSON schema
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::structured::{ResponseFormat, JsonSchema};
//!
//! let format = ResponseFormat::JsonSchema {
//!     json_schema: JsonSchema {
//!         name: "person".to_string(),
//!         description: Some("A person object".to_string()),
//!         schema: serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "name": { "type": "string" },
//!                 "age": { "type": "integer" }
//!             },
//!             "required": ["name", "age"]
//!         }),
//!         strict: Some(true),
//!     },
//! };
//! ```
//!
//! # Validation
//!
//! The module can validate that generated outputs conform to the schema:
//!
//! ```ignore
//! use infernum_server::structured::validate_output;
//!
//! let result = validate_output(&output, &schema);
//! if let Err(errors) = result {
//!     println!("Validation failed: {:?}", errors);
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

/// Response format specification for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text output (default).
    Text,

    /// JSON object output (must be valid JSON).
    JsonObject,

    /// JSON schema output (must match the provided schema).
    JsonSchema {
        /// The JSON schema specification.
        json_schema: JsonSchema,
    },
}

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Text
    }
}

impl ResponseFormat {
    /// Creates a text response format.
    pub fn text() -> Self {
        Self::Text
    }

    /// Creates a JSON object response format.
    pub fn json_object() -> Self {
        Self::JsonObject
    }

    /// Creates a JSON schema response format.
    pub fn json_schema(schema: JsonSchema) -> Self {
        Self::JsonSchema {
            json_schema: schema,
        }
    }

    /// Returns true if this format requires JSON output.
    pub fn requires_json(&self) -> bool {
        matches!(self, Self::JsonObject | Self::JsonSchema { .. })
    }

    /// Returns the schema if this is a JSON schema format.
    pub fn schema(&self) -> Option<&JsonSchema> {
        match self {
            Self::JsonSchema { json_schema } => Some(json_schema),
            _ => None,
        }
    }

    /// Returns true if strict mode is enabled.
    pub fn is_strict(&self) -> bool {
        match self {
            Self::JsonSchema { json_schema } => json_schema.strict.unwrap_or(false),
            _ => false,
        }
    }
}

impl fmt::Display for ResponseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::JsonObject => write!(f, "json_object"),
            Self::JsonSchema { json_schema } => write!(f, "json_schema({})", json_schema.name),
        }
    }
}

/// JSON schema specification for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct JsonSchema {
    /// Name of the schema (for identification).
    pub name: String,

    /// Optional description of the schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The JSON Schema definition.
    pub schema: Value,

    /// Whether to enforce strict schema matching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl JsonSchema {
    /// Creates a new JSON schema with the given name and schema definition.
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema,
            strict: None,
        }
    }

    /// Builder method to add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder method to set strict mode.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Returns true if strict mode is enabled.
    pub fn is_strict(&self) -> bool {
        self.strict.unwrap_or(false)
    }
}

/// Error type for schema validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// JSON path to the error location.
    pub path: String,

    /// Description of the error.
    pub message: String,

    /// The expected type or value.
    pub expected: Option<String>,

    /// The actual type or value found.
    pub actual: Option<String>,
}

impl ValidationError {
    /// Creates a new validation error.
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            expected: None,
            actual: None,
        }
    }

    /// Builder method to set expected value.
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    /// Builder method to set actual value.
    pub fn with_actual(mut self, actual: impl Into<String>) -> Self {
        self.actual = Some(actual.into());
        self
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)?;
        if let Some(expected) = &self.expected {
            write!(f, " (expected: {})", expected)?;
        }
        if let Some(actual) = &self.actual {
            write!(f, " (got: {})", actual)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

/// Result of schema validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub valid: bool,

    /// List of validation errors (empty if valid).
    pub errors: Vec<ValidationError>,

    /// Warnings that don't fail validation.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Creates a successful validation result.
    pub fn valid() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Creates a failed validation result with the given errors.
    pub fn invalid(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Creates a validation result with a single error.
    pub fn error(error: ValidationError) -> Self {
        Self::invalid(vec![error])
    }

    /// Adds a warning to the result.
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Returns true if validation passed.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Returns true if there are warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::valid()
    }
}

/// Validates that a JSON value conforms to a JSON schema.
///
/// This is a simplified validator that handles common schema patterns.
/// For production use, consider using a full JSON Schema validator.
pub fn validate_json(value: &Value, schema: &Value) -> ValidationResult {
    let mut errors = Vec::new();
    validate_value(value, schema, "$", &mut errors);

    if errors.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(errors)
    }
}

/// Validates a JSON string against a schema.
pub fn validate_json_string(json_str: &str, schema: &Value) -> Result<ValidationResult, String> {
    let value: Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {}", e))?;
    Ok(validate_json(&value, schema))
}

/// Validates a value against a schema at the given path.
fn validate_value(value: &Value, schema: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    // Handle boolean schemas
    if let Some(b) = schema.as_bool() {
        if !b {
            errors.push(ValidationError::new(
                path,
                "Schema is false, nothing is valid",
            ));
        }
        return;
    }

    let schema_obj = match schema.as_object() {
        Some(obj) => obj,
        None => return, // Invalid schema, skip
    };

    // Check type constraint
    if let Some(type_val) = schema_obj.get("type") {
        if !validate_type(value, type_val, path, errors) {
            return; // Type mismatch, skip further validation
        }
    }

    // Check enum constraint
    if let Some(enum_val) = schema_obj.get("enum") {
        validate_enum(value, enum_val, path, errors);
    }

    // Check const constraint
    if let Some(const_val) = schema_obj.get("const") {
        if value != const_val {
            errors.push(
                ValidationError::new(path, "Value does not match const")
                    .with_expected(const_val.to_string())
                    .with_actual(value.to_string()),
            );
        }
    }

    // Validate based on value type
    match value {
        Value::Object(obj) => validate_object(obj, schema_obj, path, errors),
        Value::Array(arr) => validate_array(arr, schema_obj, path, errors),
        Value::String(s) => validate_string(s, schema_obj, path, errors),
        Value::Number(n) => validate_number(n, schema_obj, path, errors),
        _ => {},
    }
}

/// Validates the type of a value.
fn validate_type(
    value: &Value,
    type_val: &Value,
    path: &str,
    errors: &mut Vec<ValidationError>,
) -> bool {
    let types: Vec<&str> = if let Some(s) = type_val.as_str() {
        vec![s]
    } else if let Some(arr) = type_val.as_array() {
        arr.iter().filter_map(|v| v.as_str()).collect()
    } else {
        return true; // Invalid type spec
    };

    let value_type = get_json_type(value);
    let matches = types.iter().any(|t| {
        *t == value_type || (*t == "integer" && value_type == "number" && is_integer(value))
    });

    if !matches {
        errors.push(
            ValidationError::new(path, "Type mismatch")
                .with_expected(types.join(" | "))
                .with_actual(value_type.to_string()),
        );
        return false;
    }

    true
}

/// Gets the JSON type name of a value.
fn get_json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Checks if a number value is an integer.
fn is_integer(value: &Value) -> bool {
    if let Some(n) = value.as_f64() {
        n.fract() == 0.0
    } else {
        false
    }
}

/// Validates enum constraint.
fn validate_enum(value: &Value, enum_val: &Value, path: &str, errors: &mut Vec<ValidationError>) {
    if let Some(enum_arr) = enum_val.as_array() {
        if !enum_arr.contains(value) {
            let allowed: Vec<String> = enum_arr.iter().map(|v| v.to_string()).collect();
            errors.push(
                ValidationError::new(path, "Value not in enum")
                    .with_expected(allowed.join(", "))
                    .with_actual(value.to_string()),
            );
        }
    }
}

/// Validates an object against its schema.
fn validate_object(
    obj: &serde_json::Map<String, Value>,
    schema: &serde_json::Map<String, Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    // Check required properties
    if let Some(required) = schema.get("required") {
        if let Some(required_arr) = required.as_array() {
            for req in required_arr {
                if let Some(prop_name) = req.as_str() {
                    if !obj.contains_key(prop_name) {
                        let prop_path = format!("{}.{}", path, prop_name);
                        errors.push(ValidationError::new(prop_path, "Required property missing"));
                    }
                }
            }
        }
    }

    // Validate properties
    if let Some(properties) = schema.get("properties") {
        if let Some(props) = properties.as_object() {
            for (key, prop_schema) in props {
                if let Some(prop_value) = obj.get(key) {
                    let prop_path = format!("{}.{}", path, key);
                    validate_value(prop_value, prop_schema, &prop_path, errors);
                }
            }
        }
    }

    // Check additionalProperties
    if let Some(additional) = schema.get("additionalProperties") {
        if additional.as_bool() == Some(false) {
            let properties = schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|p| p.keys().cloned().collect::<std::collections::HashSet<_>>())
                .unwrap_or_default();

            for key in obj.keys() {
                if !properties.contains(key) {
                    let prop_path = format!("{}.{}", path, key);
                    errors.push(ValidationError::new(
                        prop_path,
                        "Additional property not allowed",
                    ));
                }
            }
        }
    }

    // Check minProperties/maxProperties
    if let Some(min) = schema.get("minProperties").and_then(|v| v.as_u64()) {
        if (obj.len() as u64) < min {
            errors.push(
                ValidationError::new(path, "Too few properties")
                    .with_expected(format!("at least {}", min))
                    .with_actual(obj.len().to_string()),
            );
        }
    }

    if let Some(max) = schema.get("maxProperties").and_then(|v| v.as_u64()) {
        if (obj.len() as u64) > max {
            errors.push(
                ValidationError::new(path, "Too many properties")
                    .with_expected(format!("at most {}", max))
                    .with_actual(obj.len().to_string()),
            );
        }
    }
}

/// Validates an array against its schema.
fn validate_array(
    arr: &[Value],
    schema: &serde_json::Map<String, Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    // Validate items
    if let Some(items_schema) = schema.get("items") {
        for (i, item) in arr.iter().enumerate() {
            let item_path = format!("{}[{}]", path, i);
            validate_value(item, items_schema, &item_path, errors);
        }
    }

    // Check minItems/maxItems
    if let Some(min) = schema.get("minItems").and_then(|v| v.as_u64()) {
        if (arr.len() as u64) < min {
            errors.push(
                ValidationError::new(path, "Array too short")
                    .with_expected(format!("at least {}", min))
                    .with_actual(arr.len().to_string()),
            );
        }
    }

    if let Some(max) = schema.get("maxItems").and_then(|v| v.as_u64()) {
        if (arr.len() as u64) > max {
            errors.push(
                ValidationError::new(path, "Array too long")
                    .with_expected(format!("at most {}", max))
                    .with_actual(arr.len().to_string()),
            );
        }
    }

    // Check uniqueItems
    if schema.get("uniqueItems").and_then(|v| v.as_bool()) == Some(true) {
        let mut seen: Vec<&Value> = Vec::new();
        for (i, item) in arr.iter().enumerate() {
            if seen.contains(&item) {
                let item_path = format!("{}[{}]", path, i);
                errors.push(ValidationError::new(item_path, "Duplicate item in array"));
            }
            seen.push(item);
        }
    }
}

/// Validates a string against its schema.
fn validate_string(
    s: &str,
    schema: &serde_json::Map<String, Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    // Check minLength/maxLength
    if let Some(min) = schema.get("minLength").and_then(|v| v.as_u64()) {
        if (s.len() as u64) < min {
            errors.push(
                ValidationError::new(path, "String too short")
                    .with_expected(format!("at least {} characters", min))
                    .with_actual(s.len().to_string()),
            );
        }
    }

    if let Some(max) = schema.get("maxLength").and_then(|v| v.as_u64()) {
        if (s.len() as u64) > max {
            errors.push(
                ValidationError::new(path, "String too long")
                    .with_expected(format!("at most {} characters", max))
                    .with_actual(s.len().to_string()),
            );
        }
    }

    // Check pattern
    if let Some(pattern) = schema.get("pattern").and_then(|v| v.as_str()) {
        if let Ok(re) = regex::Regex::new(pattern) {
            if !re.is_match(s) {
                errors.push(
                    ValidationError::new(path, "String does not match pattern")
                        .with_expected(pattern.to_string()),
                );
            }
        }
    }

    // Check format (basic support)
    if let Some(format) = schema.get("format").and_then(|v| v.as_str()) {
        if !validate_format(s, format) {
            errors.push(
                ValidationError::new(path, "Invalid format").with_expected(format.to_string()),
            );
        }
    }
}

/// Validates a number against its schema.
fn validate_number(
    n: &serde_json::Number,
    schema: &serde_json::Map<String, Value>,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let value = n.as_f64().unwrap_or(0.0);

    // Check minimum/maximum
    if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
        if value < min {
            errors.push(
                ValidationError::new(path, "Number too small")
                    .with_expected(format!(">= {}", min))
                    .with_actual(value.to_string()),
            );
        }
    }

    if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64()) {
        if value > max {
            errors.push(
                ValidationError::new(path, "Number too large")
                    .with_expected(format!("<= {}", max))
                    .with_actual(value.to_string()),
            );
        }
    }

    // Check exclusiveMinimum/exclusiveMaximum
    if let Some(min) = schema.get("exclusiveMinimum").and_then(|v| v.as_f64()) {
        if value <= min {
            errors.push(
                ValidationError::new(path, "Number must be greater than minimum")
                    .with_expected(format!("> {}", min))
                    .with_actual(value.to_string()),
            );
        }
    }

    if let Some(max) = schema.get("exclusiveMaximum").and_then(|v| v.as_f64()) {
        if value >= max {
            errors.push(
                ValidationError::new(path, "Number must be less than maximum")
                    .with_expected(format!("< {}", max))
                    .with_actual(value.to_string()),
            );
        }
    }

    // Check multipleOf
    if let Some(mult) = schema.get("multipleOf").and_then(|v| v.as_f64()) {
        if mult != 0.0 && (value / mult).fract().abs() > f64::EPSILON {
            errors.push(
                ValidationError::new(path, "Number must be multiple of")
                    .with_expected(mult.to_string())
                    .with_actual(value.to_string()),
            );
        }
    }
}

/// Validates common string formats.
fn validate_format(s: &str, format: &str) -> bool {
    match format {
        "email" => s.contains('@') && s.contains('.'),
        "uri" | "url" => s.starts_with("http://") || s.starts_with("https://"),
        "date" => {
            // Basic YYYY-MM-DD validation
            s.len() == 10
                && s.chars().nth(4) == Some('-')
                && s.chars().nth(7) == Some('-')
                && s[0..4].parse::<u16>().is_ok()
                && s[5..7].parse::<u8>().is_ok()
                && s[8..10].parse::<u8>().is_ok()
        },
        "date-time" => {
            // Basic ISO 8601 validation
            s.contains('T') && (s.ends_with('Z') || s.contains('+') || s.contains('-'))
        },
        "uuid" => {
            // Basic UUID validation
            s.len() == 36
                && s.chars().nth(8) == Some('-')
                && s.chars().nth(13) == Some('-')
                && s.chars().nth(18) == Some('-')
                && s.chars().nth(23) == Some('-')
        },
        "ipv4" => {
            let parts: Vec<&str> = s.split('.').collect();
            parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok())
        },
        "ipv6" => s.contains(':') && s.split(':').count() >= 3,
        _ => true, // Unknown format, accept
    }
}

/// Schema registry for reusable schemas.
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    schemas: HashMap<String, JsonSchema>,
}

impl SchemaRegistry {
    /// Creates a new empty schema registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a schema by name.
    pub fn register(&mut self, name: impl Into<String>, schema: JsonSchema) {
        self.schemas.insert(name.into(), schema);
    }

    /// Gets a schema by name.
    pub fn get(&self, name: &str) -> Option<&JsonSchema> {
        self.schemas.get(name)
    }

    /// Removes a schema by name.
    pub fn remove(&mut self, name: &str) -> Option<JsonSchema> {
        self.schemas.remove(name)
    }

    /// Returns the number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Returns true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    /// Returns an iterator over all schema names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.schemas.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_response_format_default() {
        let format = ResponseFormat::default();
        assert!(matches!(format, ResponseFormat::Text));
        assert!(!format.requires_json());
    }

    #[test]
    fn test_response_format_text() {
        let format = ResponseFormat::text();
        assert!(!format.requires_json());
        assert!(format.schema().is_none());
        assert_eq!(format.to_string(), "text");
    }

    #[test]
    fn test_response_format_json_object() {
        let format = ResponseFormat::json_object();
        assert!(format.requires_json());
        assert!(format.schema().is_none());
        assert_eq!(format.to_string(), "json_object");
    }

    #[test]
    fn test_response_format_json_schema() {
        let schema = JsonSchema::new("test", json!({"type": "object"}));
        let format = ResponseFormat::json_schema(schema);

        assert!(format.requires_json());
        assert!(format.schema().is_some());
        assert_eq!(format.to_string(), "json_schema(test)");
    }

    #[test]
    fn test_response_format_serialization() {
        let format = ResponseFormat::json_object();
        let json = serde_json::to_string(&format).expect("serialize");
        assert!(json.contains("json_object"));

        let parsed: ResponseFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(format, parsed);
    }

    #[test]
    fn test_response_format_json_schema_serialization() {
        let schema =
            JsonSchema::new("person", json!({"type": "object"})).with_description("A person");
        let format = ResponseFormat::json_schema(schema);

        let json = serde_json::to_string(&format).expect("serialize");
        assert!(json.contains("json_schema"));
        assert!(json.contains("person"));

        let parsed: ResponseFormat = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(format, parsed);
    }

    #[test]
    fn test_json_schema_new() {
        let schema = JsonSchema::new("test", json!({"type": "string"}));

        assert_eq!(schema.name, "test");
        assert!(schema.description.is_none());
        assert!(!schema.is_strict());
    }

    #[test]
    fn test_json_schema_builder() {
        let schema = JsonSchema::new("person", json!({"type": "object"}))
            .with_description("A person object")
            .with_strict(true);

        assert_eq!(schema.name, "person");
        assert_eq!(schema.description, Some("A person object".to_string()));
        assert!(schema.is_strict());
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::new("$.name", "Required property missing");
        assert_eq!(error.to_string(), "$.name: Required property missing");

        let error = ValidationError::new("$.age", "Type mismatch")
            .with_expected("number")
            .with_actual("string");
        assert!(error.to_string().contains("expected: number"));
        assert!(error.to_string().contains("got: string"));
    }

    #[test]
    fn test_validation_result_valid() {
        let result = ValidationResult::valid();
        assert!(result.is_valid());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_result_invalid() {
        let result = ValidationResult::error(ValidationError::new("$", "test"));
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_validation_result_with_warning() {
        let result = ValidationResult::valid().with_warning("Deprecated field");
        assert!(result.is_valid());
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validate_type_string() {
        let schema = json!({"type": "string"});
        let valid = json!("hello");
        let invalid = json!(123);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_number() {
        let schema = json!({"type": "number"});
        let valid = json!(42.5);
        let invalid = json!("hello");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_integer() {
        let schema = json!({"type": "integer"});
        let valid = json!(42);
        let invalid = json!(42.5);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_boolean() {
        let schema = json!({"type": "boolean"});
        let valid = json!(true);
        let invalid = json!("true");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_null() {
        let schema = json!({"type": "null"});
        let valid = json!(null);
        let invalid = json!("");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_array() {
        let schema = json!({"type": "array"});
        let valid = json!([1, 2, 3]);
        let invalid = json!({"a": 1});

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_type_object() {
        let schema = json!({"type": "object"});
        let valid = json!({"a": 1});
        let invalid = json!([1, 2]);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_enum() {
        let schema = json!({"enum": ["a", "b", "c"]});
        let valid = json!("b");
        let invalid = json!("d");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_const() {
        let schema = json!({"const": "fixed"});
        let valid = json!("fixed");
        let invalid = json!("other");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_required_properties() {
        let schema = json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        });

        let valid = json!({"name": "John", "age": 30});
        let missing_age = json!({"name": "John"});
        let missing_name = json!({"age": 30});

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&missing_age, &schema).is_valid());
        assert!(!validate_json(&missing_name, &schema).is_valid());
    }

    #[test]
    fn test_validate_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false
        });

        let valid = json!({"name": "John"});
        let invalid = json!({"name": "John", "extra": "field"});

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_min_max_properties() {
        let schema = json!({
            "type": "object",
            "minProperties": 1,
            "maxProperties": 2
        });

        let valid = json!({"a": 1, "b": 2});
        let too_few = json!({});
        let too_many = json!({"a": 1, "b": 2, "c": 3});

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&too_few, &schema).is_valid());
        assert!(!validate_json(&too_many, &schema).is_valid());
    }

    #[test]
    fn test_validate_array_items() {
        let schema = json!({
            "type": "array",
            "items": {"type": "number"}
        });

        let valid = json!([1, 2, 3]);
        let invalid = json!([1, "two", 3]);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_array_min_max_items() {
        let schema = json!({
            "type": "array",
            "minItems": 2,
            "maxItems": 4
        });

        let valid = json!([1, 2, 3]);
        let too_few = json!([1]);
        let too_many = json!([1, 2, 3, 4, 5]);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&too_few, &schema).is_valid());
        assert!(!validate_json(&too_many, &schema).is_valid());
    }

    #[test]
    fn test_validate_unique_items() {
        let schema = json!({
            "type": "array",
            "uniqueItems": true
        });

        let valid = json!([1, 2, 3]);
        let invalid = json!([1, 2, 2]);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_string_length() {
        let schema = json!({
            "type": "string",
            "minLength": 2,
            "maxLength": 5
        });

        let valid = json!("abc");
        let too_short = json!("a");
        let too_long = json!("abcdef");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&too_short, &schema).is_valid());
        assert!(!validate_json(&too_long, &schema).is_valid());
    }

    #[test]
    fn test_validate_string_pattern() {
        let schema = json!({
            "type": "string",
            "pattern": "^[a-z]+$"
        });

        let valid = json!("abc");
        let invalid = json!("ABC");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_number_range() {
        let schema = json!({
            "type": "number",
            "minimum": 0,
            "maximum": 100
        });

        let valid = json!(50);
        let too_small = json!(-1);
        let too_large = json!(101);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&too_small, &schema).is_valid());
        assert!(!validate_json(&too_large, &schema).is_valid());
    }

    #[test]
    fn test_validate_number_exclusive_range() {
        let schema = json!({
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 100
        });

        let valid = json!(50);
        let at_min = json!(0);
        let at_max = json!(100);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&at_min, &schema).is_valid());
        assert!(!validate_json(&at_max, &schema).is_valid());
    }

    #[test]
    fn test_validate_multiple_of() {
        let schema = json!({
            "type": "number",
            "multipleOf": 5
        });

        let valid = json!(15);
        let invalid = json!(17);

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_format_email() {
        let schema = json!({
            "type": "string",
            "format": "email"
        });

        let valid = json!("test@example.com");
        let invalid = json!("not-an-email");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_format_uri() {
        let schema = json!({
            "type": "string",
            "format": "uri"
        });

        let valid = json!("https://example.com");
        let invalid = json!("not-a-uri");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_format_date() {
        let schema = json!({
            "type": "string",
            "format": "date"
        });

        let valid = json!("2024-01-15");
        let invalid = json!("01/15/2024");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_format_ipv4() {
        let schema = json!({
            "type": "string",
            "format": "ipv4"
        });

        let valid = json!("192.168.1.1");
        let invalid = json!("999.999.999.999");

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"}
                    }
                }
            }
        });

        let valid = json!({"person": {"name": "John", "age": 30}});
        let invalid = json!({"person": {"age": 30}});

        assert!(validate_json(&valid, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }

    #[test]
    fn test_validate_json_string() {
        let schema = json!({"type": "string"});

        let result = validate_json_string("\"hello\"", &schema);
        assert!(result.is_ok());
        assert!(result.expect("valid").is_valid());

        let result = validate_json_string("invalid json", &schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_boolean_schema_false() {
        let schema = json!(false);
        let value = json!("anything");

        let result = validate_json(&value, &schema);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_boolean_schema_true() {
        let schema = json!(true);
        let value = json!("anything");

        let result = validate_json(&value, &schema);
        assert!(result.is_valid());
    }

    #[test]
    fn test_schema_registry() {
        let mut registry = SchemaRegistry::new();
        assert!(registry.is_empty());

        let schema = JsonSchema::new("person", json!({"type": "object"}));
        registry.register("person", schema);

        assert_eq!(registry.len(), 1);
        assert!(registry.get("person").is_some());
        assert!(registry.get("unknown").is_none());

        let names: Vec<_> = registry.names().collect();
        assert!(names.contains(&&"person".to_string()));

        registry.remove("person");
        assert!(registry.is_empty());
    }

    #[test]
    fn test_response_format_is_strict() {
        let text = ResponseFormat::text();
        assert!(!text.is_strict());

        let json_obj = ResponseFormat::json_object();
        assert!(!json_obj.is_strict());

        let non_strict = ResponseFormat::json_schema(JsonSchema::new("test", json!({})));
        assert!(!non_strict.is_strict());

        let strict =
            ResponseFormat::json_schema(JsonSchema::new("test", json!({})).with_strict(true));
        assert!(strict.is_strict());
    }

    #[test]
    fn test_union_types() {
        let schema = json!({
            "type": ["string", "number"]
        });

        let string = json!("hello");
        let number = json!(42);
        let invalid = json!(true);

        assert!(validate_json(&string, &schema).is_valid());
        assert!(validate_json(&number, &schema).is_valid());
        assert!(!validate_json(&invalid, &schema).is_valid());
    }
}
