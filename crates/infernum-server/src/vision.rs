//! Vision and multimodal content support.
//!
//! This module provides support for vision models that can process both text
//! and image inputs. It extends the standard chat message format to support
//! multimodal content.
//!
//! # Content Types
//!
//! Messages can contain multiple content parts:
//!
//! - **Text**: Plain text content
//! - **ImageUrl**: Image from a URL (with optional detail level)
//! - **ImageBase64**: Base64-encoded image data
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::vision::{MessageContent, ContentPart, ImageDetail};
//!
//! let content = MessageContent::Parts(vec![
//!     ContentPart::Text { text: "What's in this image?".to_string() },
//!     ContentPart::ImageUrl {
//!         image_url: ImageUrl {
//!             url: "https://example.com/image.png".to_string(),
//!             detail: Some(ImageDetail::High),
//!         },
//!     },
//! ]);
//! ```
//!
//! # Image Validation
//!
//! The module provides validation for:
//! - Supported media types (JPEG, PNG, GIF, WebP)
//! - Base64 data format
//! - URL format
//! - Maximum image size

use std::fmt;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Message content that can be text or multimodal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content (backward compatible).
    Text(String),

    /// Array of content parts for multimodal messages.
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl MessageContent {
    /// Creates text content.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Creates content with a single image URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl::new(url),
        }])
    }

    /// Creates content with text and an image URL.
    pub fn text_and_image(text: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Parts(vec![
            ContentPart::Text { text: text.into() },
            ContentPart::ImageUrl {
                image_url: ImageUrl::new(url),
            },
        ])
    }

    /// Creates content with a base64 image.
    pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Parts(vec![ContentPart::ImageBase64 {
            image_base64: ImageBase64 {
                data: data.into(),
                media_type: media_type.into(),
            },
        }])
    }

    /// Returns true if this is text-only content.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Returns true if this content contains images.
    pub fn has_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|p| p.is_image()),
        }
    }

    /// Gets the text content if this is text-only.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            Self::Parts(_) => None,
        }
    }

    /// Extracts all text from the content.
    pub fn extract_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// Returns the number of content parts.
    pub fn part_count(&self) -> usize {
        match self {
            Self::Text(_) => 1,
            Self::Parts(parts) => parts.len(),
        }
    }

    /// Returns the number of images in the content.
    pub fn image_count(&self) -> usize {
        match self {
            Self::Text(_) => 0,
            Self::Parts(parts) => parts.iter().filter(|p| p.is_image()).count(),
        }
    }

    /// Validates the content.
    pub fn validate(&self, config: &VisionConfig) -> Result<(), VisionError> {
        match self {
            Self::Text(s) => {
                if s.len() > config.max_text_length {
                    return Err(VisionError::TextTooLong {
                        actual: s.len(),
                        max: config.max_text_length,
                    });
                }
            }
            Self::Parts(parts) => {
                if parts.is_empty() {
                    return Err(VisionError::EmptyContent);
                }

                let image_count = parts.iter().filter(|p| p.is_image()).count();
                if image_count > config.max_images_per_message {
                    return Err(VisionError::TooManyImages {
                        actual: image_count,
                        max: config.max_images_per_message,
                    });
                }

                for part in parts {
                    part.validate(config)?;
                }
            }
        }
        Ok(())
    }
}

/// A part of multimodal content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content.
    Text {
        /// The text content.
        text: String,
    },

    /// Image from URL.
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL specification.
        image_url: ImageUrl,
    },

    /// Base64-encoded image.
    #[serde(rename = "image_base64")]
    ImageBase64 {
        /// The base64 image specification.
        image_base64: ImageBase64,
    },
}

impl ContentPart {
    /// Creates a text part.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Creates an image URL part.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl::new(url),
        }
    }

    /// Creates a base64 image part.
    pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::ImageBase64 {
            image_base64: ImageBase64 {
                data: data.into(),
                media_type: media_type.into(),
            },
        }
    }

    /// Returns true if this is an image part.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::ImageUrl { .. } | Self::ImageBase64 { .. })
    }

    /// Returns true if this is a text part.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Validates the content part.
    pub fn validate(&self, config: &VisionConfig) -> Result<(), VisionError> {
        match self {
            Self::Text { text } => {
                if text.len() > config.max_text_length {
                    return Err(VisionError::TextTooLong {
                        actual: text.len(),
                        max: config.max_text_length,
                    });
                }
            }
            Self::ImageUrl { image_url } => {
                image_url.validate(config)?;
            }
            Self::ImageBase64 { image_base64 } => {
                image_base64.validate(config)?;
            }
        }
        Ok(())
    }
}

/// Image URL specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ImageUrl {
    /// The URL of the image.
    pub url: String,

    /// Detail level for image processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl ImageUrl {
    /// Creates a new image URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
        }
    }

    /// Builder method to set detail level.
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Gets the effective detail level.
    pub fn effective_detail(&self) -> ImageDetail {
        self.detail.unwrap_or_default()
    }

    /// Validates the image URL.
    pub fn validate(&self, _config: &VisionConfig) -> Result<(), VisionError> {
        if self.url.is_empty() {
            return Err(VisionError::InvalidUrl("URL is empty".to_string()));
        }

        // Check for valid URL format
        if !self.url.starts_with("http://")
            && !self.url.starts_with("https://")
            && !self.url.starts_with("data:")
        {
            return Err(VisionError::InvalidUrl(
                "URL must start with http://, https://, or data:".to_string(),
            ));
        }

        // If it's a data URL, validate the format
        if self.url.starts_with("data:") {
            validate_data_url(&self.url)?;
        }

        Ok(())
    }
}

/// Base64-encoded image specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ImageBase64 {
    /// The base64-encoded image data.
    pub data: String,

    /// The MIME type of the image.
    pub media_type: String,
}

impl ImageBase64 {
    /// Creates a new base64 image.
    pub fn new(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            media_type: media_type.into(),
        }
    }

    /// Validates the base64 image.
    pub fn validate(&self, config: &VisionConfig) -> Result<(), VisionError> {
        // Validate media type
        if !is_supported_media_type(&self.media_type) {
            return Err(VisionError::UnsupportedMediaType(self.media_type.clone()));
        }

        // Validate base64 data is not empty
        if self.data.is_empty() {
            return Err(VisionError::InvalidBase64("Empty data".to_string()));
        }

        // Validate base64 format (rough check)
        if !is_valid_base64(&self.data) {
            return Err(VisionError::InvalidBase64("Invalid base64 format".to_string()));
        }

        // Check approximate decoded size
        let estimated_size = estimate_base64_size(&self.data);
        if estimated_size > config.max_image_size {
            return Err(VisionError::ImageTooLarge {
                actual: estimated_size,
                max: config.max_image_size,
            });
        }

        Ok(())
    }

    /// Decodes the base64 data.
    pub fn decode(&self) -> Result<Vec<u8>, VisionError> {
        // Remove whitespace and padding issues
        let clean_data: String = self.data.chars().filter(|c| !c.is_whitespace()).collect();

        // Simple base64 decode
        decode_base64(&clean_data)
    }

    /// Converts to a data URL.
    pub fn to_data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data)
    }
}

/// Image processing detail level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Low resolution (512x512, faster, cheaper).
    Low,

    /// High resolution (up to 2048x2048, slower, more expensive).
    High,

    /// Automatic selection based on image size.
    #[default]
    Auto,
}

impl ImageDetail {
    /// Gets the maximum dimension for this detail level.
    pub fn max_dimension(&self) -> u32 {
        match self {
            Self::Low => 512,
            Self::High | Self::Auto => 2048,
        }
    }

    /// Gets the tile size for high-detail processing.
    pub fn tile_size(&self) -> u32 {
        match self {
            Self::Low => 512,
            Self::High | Self::Auto => 512, // High uses 512x512 tiles
        }
    }
}

impl fmt::Display for ImageDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::High => write!(f, "high"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Supported image media types.
pub const SUPPORTED_MEDIA_TYPES: &[&str] = &[
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
];

/// Checks if a media type is supported.
pub fn is_supported_media_type(media_type: &str) -> bool {
    let normalized = media_type.to_lowercase();
    SUPPORTED_MEDIA_TYPES.contains(&normalized.as_str())
}

/// Vision configuration.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Maximum text length per part.
    pub max_text_length: usize,

    /// Maximum image size in bytes.
    pub max_image_size: usize,

    /// Maximum images per message.
    pub max_images_per_message: usize,

    /// Maximum total images per request.
    pub max_images_per_request: usize,

    /// Allow external URLs.
    pub allow_external_urls: bool,

    /// Allowed URL hosts (empty = allow all).
    pub allowed_hosts: Vec<String>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            max_text_length: 100_000,
            max_image_size: 20 * 1024 * 1024, // 20MB
            max_images_per_message: 10,
            max_images_per_request: 50,
            allow_external_urls: true,
            allowed_hosts: Vec::new(),
        }
    }
}

impl VisionConfig {
    /// Creates a new vision config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set max image size.
    pub fn with_max_image_size(mut self, size: usize) -> Self {
        self.max_image_size = size;
        self
    }

    /// Builder method to set max images per message.
    pub fn with_max_images_per_message(mut self, max: usize) -> Self {
        self.max_images_per_message = max;
        self
    }

    /// Builder method to disable external URLs.
    pub fn with_external_urls_disabled(mut self) -> Self {
        self.allow_external_urls = false;
        self
    }

    /// Builder method to set allowed hosts.
    pub fn with_allowed_hosts(mut self, hosts: Vec<String>) -> Self {
        self.allowed_hosts = hosts;
        self
    }
}

/// Vision error types.
#[derive(Debug, Clone, PartialEq)]
pub enum VisionError {
    /// Text content is too long.
    TextTooLong {
        /// Actual length.
        actual: usize,
        /// Maximum allowed.
        max: usize,
    },

    /// Image is too large.
    ImageTooLarge {
        /// Actual size in bytes.
        actual: usize,
        /// Maximum allowed.
        max: usize,
    },

    /// Too many images in a message.
    TooManyImages {
        /// Actual count.
        actual: usize,
        /// Maximum allowed.
        max: usize,
    },

    /// Invalid image URL.
    InvalidUrl(String),

    /// Invalid base64 data.
    InvalidBase64(String),

    /// Unsupported media type.
    UnsupportedMediaType(String),

    /// Empty content.
    EmptyContent,

    /// External URLs not allowed.
    ExternalUrlNotAllowed,

    /// Host not in allowed list.
    HostNotAllowed(String),
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextTooLong { actual, max } => {
                write!(f, "Text too long: {} characters (max {})", actual, max)
            }
            Self::ImageTooLarge { actual, max } => {
                write!(f, "Image too large: {} bytes (max {})", actual, max)
            }
            Self::TooManyImages { actual, max } => {
                write!(f, "Too many images: {} (max {})", actual, max)
            }
            Self::InvalidUrl(msg) => write!(f, "Invalid URL: {}", msg),
            Self::InvalidBase64(msg) => write!(f, "Invalid base64: {}", msg),
            Self::UnsupportedMediaType(mt) => write!(f, "Unsupported media type: {}", mt),
            Self::EmptyContent => write!(f, "Content is empty"),
            Self::ExternalUrlNotAllowed => write!(f, "External URLs are not allowed"),
            Self::HostNotAllowed(host) => write!(f, "Host not allowed: {}", host),
        }
    }
}

impl std::error::Error for VisionError {}

/// Validates a data URL format.
fn validate_data_url(url: &str) -> Result<(), VisionError> {
    // data:[<mediatype>][;base64],<data>
    if !url.starts_with("data:") {
        return Err(VisionError::InvalidUrl("Not a data URL".to_string()));
    }

    let rest = &url[5..];

    // Find the comma separating metadata from data
    let comma_pos = rest
        .find(',')
        .ok_or_else(|| VisionError::InvalidUrl("Missing comma in data URL".to_string()))?;

    let metadata = &rest[..comma_pos];

    // Check for base64 encoding
    if !metadata.contains(";base64") {
        return Err(VisionError::InvalidUrl(
            "Data URL must use base64 encoding".to_string(),
        ));
    }

    // Extract media type
    let media_type = metadata.split(';').next().unwrap_or("");
    if !media_type.is_empty() && !is_supported_media_type(media_type) {
        return Err(VisionError::UnsupportedMediaType(media_type.to_string()));
    }

    Ok(())
}

/// Checks if a string is valid base64.
fn is_valid_base64(data: &str) -> bool {
    // Check for valid base64 characters
    data.chars().all(|c| {
        c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' || c.is_whitespace()
    })
}

/// Estimates the decoded size of base64 data.
fn estimate_base64_size(data: &str) -> usize {
    // Base64 encoding uses 4 characters for every 3 bytes
    // Remove whitespace and padding for estimate
    let clean_len = data.chars().filter(|c| !c.is_whitespace() && *c != '=').count();
    (clean_len * 3) / 4
}

/// Decodes base64 data.
fn decode_base64(data: &str) -> Result<Vec<u8>, VisionError> {
    // Simple base64 decoding
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = Vec::with_capacity((data.len() * 3) / 4);
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;

    for c in data.chars() {
        if c == '=' {
            break;
        }
        if c.is_whitespace() {
            continue;
        }

        let value = alphabet
            .iter()
            .position(|&x| x == c as u8)
            .ok_or_else(|| VisionError::InvalidBase64(format!("Invalid character: {}", c)))?
            as u32;

        buffer = (buffer << 6) | value;
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            result.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(result)
}

/// Vision metrics for monitoring.
#[derive(Debug, Default)]
pub struct VisionMetrics {
    /// Total images processed.
    images_processed: std::sync::atomic::AtomicU64,

    /// Total images rejected.
    images_rejected: std::sync::atomic::AtomicU64,

    /// Total bytes processed.
    bytes_processed: std::sync::atomic::AtomicU64,
}

impl VisionMetrics {
    /// Creates new vision metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an image processed.
    pub fn record_image_processed(&self, bytes: usize) {
        self.images_processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_processed
            .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records an image rejected.
    pub fn record_image_rejected(&self) {
        self.images_rejected
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Gets images processed count.
    pub fn images_processed(&self) -> u64 {
        self.images_processed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Gets images rejected count.
    pub fn images_rejected(&self) -> u64 {
        self.images_rejected
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Gets bytes processed.
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_vision_images_processed_total Total images processed\n");
        output.push_str("# TYPE infernum_vision_images_processed_total counter\n");
        output.push_str(&format!(
            "infernum_vision_images_processed_total {}\n",
            self.images_processed()
        ));

        output.push_str("# HELP infernum_vision_images_rejected_total Total images rejected\n");
        output.push_str("# TYPE infernum_vision_images_rejected_total counter\n");
        output.push_str(&format!(
            "infernum_vision_images_rejected_total {}\n",
            self.images_rejected()
        ));

        output.push_str("# HELP infernum_vision_bytes_processed_total Total bytes processed\n");
        output.push_str("# TYPE infernum_vision_bytes_processed_total counter\n");
        output.push_str(&format!(
            "infernum_vision_bytes_processed_total {}\n",
            self.bytes_processed()
        ));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_content_text() {
        let content = MessageContent::text("Hello");
        assert!(content.is_text());
        assert!(!content.has_images());
        assert_eq!(content.as_text(), Some("Hello"));
        assert_eq!(content.extract_text(), "Hello");
        assert_eq!(content.part_count(), 1);
        assert_eq!(content.image_count(), 0);
    }

    #[test]
    fn test_message_content_from_string() {
        let content: MessageContent = "Test".into();
        assert!(content.is_text());
        assert_eq!(content.as_text(), Some("Test"));
    }

    #[test]
    fn test_message_content_image_url() {
        let content = MessageContent::image_url("https://example.com/image.png");
        assert!(!content.is_text());
        assert!(content.has_images());
        assert_eq!(content.image_count(), 1);
    }

    #[test]
    fn test_message_content_text_and_image() {
        let content = MessageContent::text_and_image("What's this?", "https://example.com/img.jpg");
        assert!(!content.is_text());
        assert!(content.has_images());
        assert_eq!(content.part_count(), 2);
        assert_eq!(content.image_count(), 1);
        assert!(content.extract_text().contains("What's this?"));
    }

    #[test]
    fn test_message_content_base64() {
        let content = MessageContent::image_base64("SGVsbG8=", "image/png");
        assert!(content.has_images());
        assert_eq!(content.image_count(), 1);
    }

    #[test]
    fn test_message_content_serialization() {
        let content = MessageContent::text("Hello");
        let json = serde_json::to_string(&content).expect("serialize");
        assert_eq!(json, "\"Hello\"");

        let parsed: MessageContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(content, parsed);
    }

    #[test]
    fn test_message_content_parts_serialization() {
        let content = MessageContent::Parts(vec![
            ContentPart::text("Hello"),
            ContentPart::image_url("https://example.com/img.png"),
        ]);

        let json = serde_json::to_string(&content).expect("serialize");
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"type\":\"image_url\""));
    }

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::text("Hello");
        assert!(part.is_text());
        assert!(!part.is_image());
    }

    #[test]
    fn test_content_part_image_url() {
        let part = ContentPart::image_url("https://example.com/img.png");
        assert!(!part.is_text());
        assert!(part.is_image());
    }

    #[test]
    fn test_content_part_image_base64() {
        let part = ContentPart::image_base64("SGVsbG8=", "image/png");
        assert!(part.is_image());
    }

    #[test]
    fn test_image_url_new() {
        let url = ImageUrl::new("https://example.com/img.png");
        assert_eq!(url.url, "https://example.com/img.png");
        assert!(url.detail.is_none());
        assert_eq!(url.effective_detail(), ImageDetail::Auto);
    }

    #[test]
    fn test_image_url_with_detail() {
        let url = ImageUrl::new("https://example.com/img.png").with_detail(ImageDetail::High);
        assert_eq!(url.detail, Some(ImageDetail::High));
        assert_eq!(url.effective_detail(), ImageDetail::High);
    }

    #[test]
    fn test_image_url_validation() {
        let config = VisionConfig::default();

        let valid = ImageUrl::new("https://example.com/img.png");
        assert!(valid.validate(&config).is_ok());

        let empty = ImageUrl::new("");
        assert!(empty.validate(&config).is_err());

        let invalid = ImageUrl::new("ftp://example.com/img.png");
        assert!(invalid.validate(&config).is_err());
    }

    #[test]
    fn test_image_url_data_url() {
        let config = VisionConfig::default();

        let valid = ImageUrl::new("data:image/png;base64,SGVsbG8=");
        assert!(valid.validate(&config).is_ok());

        let no_base64 = ImageUrl::new("data:image/png,SGVsbG8=");
        assert!(no_base64.validate(&config).is_err());
    }

    #[test]
    fn test_image_base64_new() {
        let img = ImageBase64::new("SGVsbG8=", "image/png");
        assert_eq!(img.data, "SGVsbG8=");
        assert_eq!(img.media_type, "image/png");
    }

    #[test]
    fn test_image_base64_validation() {
        let config = VisionConfig::default();

        let valid = ImageBase64::new("SGVsbG8=", "image/png");
        assert!(valid.validate(&config).is_ok());

        let empty = ImageBase64::new("", "image/png");
        assert!(empty.validate(&config).is_err());

        let bad_media = ImageBase64::new("SGVsbG8=", "image/bmp");
        assert!(bad_media.validate(&config).is_err());
    }

    #[test]
    fn test_image_base64_decode() {
        let img = ImageBase64::new("SGVsbG8=", "image/png");
        let decoded = img.decode().expect("decode");
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_image_base64_to_data_url() {
        let img = ImageBase64::new("SGVsbG8=", "image/png");
        assert_eq!(img.to_data_url(), "data:image/png;base64,SGVsbG8=");
    }

    #[test]
    fn test_image_detail_defaults() {
        assert_eq!(ImageDetail::default(), ImageDetail::Auto);
    }

    #[test]
    fn test_image_detail_dimensions() {
        assert_eq!(ImageDetail::Low.max_dimension(), 512);
        assert_eq!(ImageDetail::High.max_dimension(), 2048);
        assert_eq!(ImageDetail::Auto.max_dimension(), 2048);
    }

    #[test]
    fn test_image_detail_display() {
        assert_eq!(ImageDetail::Low.to_string(), "low");
        assert_eq!(ImageDetail::High.to_string(), "high");
        assert_eq!(ImageDetail::Auto.to_string(), "auto");
    }

    #[test]
    fn test_supported_media_types() {
        assert!(is_supported_media_type("image/png"));
        assert!(is_supported_media_type("image/jpeg"));
        assert!(is_supported_media_type("image/JPEG")); // Case insensitive
        assert!(is_supported_media_type("image/gif"));
        assert!(is_supported_media_type("image/webp"));

        assert!(!is_supported_media_type("image/bmp"));
        assert!(!is_supported_media_type("image/tiff"));
        assert!(!is_supported_media_type("text/plain"));
    }

    #[test]
    fn test_vision_config_default() {
        let config = VisionConfig::default();

        assert_eq!(config.max_text_length, 100_000);
        assert_eq!(config.max_image_size, 20 * 1024 * 1024);
        assert_eq!(config.max_images_per_message, 10);
        assert!(config.allow_external_urls);
    }

    #[test]
    fn test_vision_config_builder() {
        let config = VisionConfig::new()
            .with_max_image_size(1024)
            .with_max_images_per_message(5)
            .with_external_urls_disabled();

        assert_eq!(config.max_image_size, 1024);
        assert_eq!(config.max_images_per_message, 5);
        assert!(!config.allow_external_urls);
    }

    #[test]
    fn test_vision_error_display() {
        let err = VisionError::TextTooLong {
            actual: 1000,
            max: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));

        let err = VisionError::UnsupportedMediaType("image/bmp".to_string());
        assert!(err.to_string().contains("image/bmp"));
    }

    #[test]
    fn test_message_content_validation() {
        let config = VisionConfig::default();

        let valid = MessageContent::text("Hello");
        assert!(valid.validate(&config).is_ok());

        let empty = MessageContent::Parts(vec![]);
        assert!(empty.validate(&config).is_err());
    }

    #[test]
    fn test_message_content_validation_too_many_images() {
        let config = VisionConfig::new().with_max_images_per_message(2);

        let parts = vec![
            ContentPart::image_url("https://example.com/1.png"),
            ContentPart::image_url("https://example.com/2.png"),
            ContentPart::image_url("https://example.com/3.png"),
        ];

        let content = MessageContent::Parts(parts);
        let result = content.validate(&config);

        assert!(result.is_err());
        match result {
            Err(VisionError::TooManyImages { actual, max }) => {
                assert_eq!(actual, 3);
                assert_eq!(max, 2);
            }
            _ => panic!("Expected TooManyImages error"),
        }
    }

    #[test]
    fn test_is_valid_base64() {
        assert!(is_valid_base64("SGVsbG8="));
        assert!(is_valid_base64("SGVs bG8=")); // With whitespace
        assert!(!is_valid_base64("SGVs!bG8="));
    }

    #[test]
    fn test_estimate_base64_size() {
        // "Hello" encoded is "SGVsbG8=" (8 chars for 5 bytes)
        assert_eq!(estimate_base64_size("SGVsbG8="), 5);

        // Longer string
        let data = "SGVsbG8gV29ybGQh"; // "Hello World!" (12 bytes)
        assert_eq!(estimate_base64_size(data), 12);
    }

    #[test]
    fn test_decode_base64() {
        let result = decode_base64("SGVsbG8=").expect("decode");
        assert_eq!(result, b"Hello");

        let result = decode_base64("SGVsbG8gV29ybGQh").expect("decode");
        assert_eq!(result, b"Hello World!");
    }

    #[test]
    fn test_decode_base64_invalid() {
        let result = decode_base64("SGVs!bG8=");
        assert!(result.is_err());
    }

    #[test]
    fn test_vision_metrics_new() {
        let metrics = VisionMetrics::new();

        assert_eq!(metrics.images_processed(), 0);
        assert_eq!(metrics.images_rejected(), 0);
        assert_eq!(metrics.bytes_processed(), 0);
    }

    #[test]
    fn test_vision_metrics_record() {
        let metrics = VisionMetrics::new();

        metrics.record_image_processed(1024);
        metrics.record_image_processed(2048);
        metrics.record_image_rejected();

        assert_eq!(metrics.images_processed(), 2);
        assert_eq!(metrics.images_rejected(), 1);
        assert_eq!(metrics.bytes_processed(), 3072);
    }

    #[test]
    fn test_vision_metrics_prometheus() {
        let metrics = VisionMetrics::new();
        metrics.record_image_processed(1024);
        metrics.record_image_rejected();

        let output = metrics.prometheus();

        assert!(output.contains("infernum_vision_images_processed_total 1"));
        assert!(output.contains("infernum_vision_images_rejected_total 1"));
        assert!(output.contains("infernum_vision_bytes_processed_total 1024"));
    }

    #[test]
    fn test_content_part_serialization() {
        let part = ContentPart::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&part).expect("serialize");
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_content_part_image_url_serialization() {
        let part = ContentPart::ImageUrl {
            image_url: ImageUrl::new("https://example.com/img.png"),
        };
        let json = serde_json::to_string(&part).expect("serialize");
        assert!(json.contains("\"type\":\"image_url\""));
        assert!(json.contains("\"url\":\"https://example.com/img.png\""));
    }

    #[test]
    fn test_content_part_deserialization() {
        let json = r#"{"type":"text","text":"Hello world"}"#;
        let part: ContentPart = serde_json::from_str(json).expect("deserialize");

        match part {
            ContentPart::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected Text part"),
        }
    }

    #[test]
    fn test_image_url_deserialization() {
        let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/img.png","detail":"high"}}"#;
        let part: ContentPart = serde_json::from_str(json).expect("deserialize");

        match part {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl part"),
        }
    }

    #[test]
    fn test_image_detail_serialization() {
        let detail = ImageDetail::High;
        let json = serde_json::to_string(&detail).expect("serialize");
        assert_eq!(json, "\"high\"");

        let parsed: ImageDetail = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, ImageDetail::High);
    }
}
