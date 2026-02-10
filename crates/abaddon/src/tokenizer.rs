//! Tokenizer wrapper for text encoding/decoding.

use std::path::Path;

use infernum_core::Result;

/// Wrapper around tokenizers for encoding and decoding text.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Beginning of sequence token ID.
    pub bos_token_id: Option<u32>,
    /// End of sequence token ID.
    pub eos_token_id: Option<u32>,
    /// Padding token ID.
    pub pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Loads a tokenizer from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot be loaded.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            infernum_core::Error::Tokenization {
                message: e.to_string(),
            }
        })?;

        Ok(Self::from_tokenizer(inner))
    }

    /// Creates a tokenizer from a pre-trained model name.
    ///
    /// Note: This requires the tokenizer.json to be downloaded from HuggingFace Hub first.
    /// Use `ModelLoader::download` to fetch model files including the tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer file cannot be found or loaded.
    pub fn from_pretrained(_name: &str) -> Result<Self> {
        // The tokenizers crate's from_pretrained feature requires additional dependencies.
        // For now, users should download tokenizer.json via ModelLoader and use from_file.
        Err(infernum_core::Error::Tokenization {
            message: "from_pretrained not available; use from_file with downloaded tokenizer.json"
                .to_string(),
        })
    }

    /// Creates a wrapper from an existing tokenizer.
    fn from_tokenizer(inner: tokenizers::Tokenizer) -> Self {
        // Try to extract special token IDs from added vocabulary
        let added_vocab = inner.get_added_vocabulary().get_vocab();

        let bos_token_id = added_vocab
            .get("<s>")
            .or_else(|| added_vocab.get("<|begin_of_text|>"))
            .copied();

        let eos_token_id = added_vocab
            .get("</s>")
            .or_else(|| added_vocab.get("<|end_of_text|>"))
            .or_else(|| added_vocab.get("<|eot_id|>"))
            .copied();

        let pad_token_id = added_vocab
            .get("<pad>")
            .or_else(|| added_vocab.get("[PAD]"))
            .copied();

        Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        }
    }

    /// Encodes text to token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special_tokens).map_err(|e| {
            infernum_core::Error::Tokenization {
                message: e.to_string(),
            }
        })?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes token IDs to text.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| {
            infernum_core::Error::Tokenization {
                message: e.to_string(),
            }
        })
    }

    /// Decodes a single token ID to text.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id], false)
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Returns the token ID for a given token string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Returns the token string for a given token ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Applies the chat template to format messages.
    ///
    /// Automatically detects the appropriate template format based on
    /// the tokenizer's special tokens:
    /// - Llama 3 format (uses `<|start_header_id|>`)
    /// - ChatML format (uses `<|im_start|>`)
    /// - Mistral format (uses `[INST]`)
    /// - Alpaca/Vicuna format (fallback)
    ///
    /// # Errors
    ///
    /// Returns an error if the template cannot be applied.
    pub fn apply_chat_template(
        &self,
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // Detect template format from special tokens
        let template = self.detect_template_format();
        template.apply(messages, add_generation_prompt)
    }

    /// Detects the chat template format based on tokenizer vocabulary.
    fn detect_template_format(&self) -> ChatTemplate {
        // Check for Llama 3 format
        if self.inner.token_to_id("<|start_header_id|>").is_some() {
            return ChatTemplate::Llama3;
        }

        // Check for ChatML format (Qwen, Phi, etc.)
        if self.inner.token_to_id("<|im_start|>").is_some() {
            return ChatTemplate::ChatML;
        }

        // Check for Mistral format
        if self.inner.token_to_id("[INST]").is_some() {
            return ChatTemplate::Mistral;
        }

        // Check for Gemma format
        if self.inner.token_to_id("<start_of_turn>").is_some() {
            return ChatTemplate::Gemma;
        }

        // Default to Alpaca/simple format
        ChatTemplate::Alpaca
    }
}

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3 / Llama 3.1 / Llama 3.2 format
    Llama3,
    /// ChatML format (Qwen, Phi, Yi, etc.)
    ChatML,
    /// Mistral / Mixtral format
    Mistral,
    /// Gemma format
    Gemma,
    /// Alpaca / Vicuna format (fallback)
    Alpaca,
}

impl ChatTemplate {
    /// Applies this template to the given messages.
    pub fn apply(
        self,
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        match self {
            Self::Llama3 => Self::apply_llama3(messages, add_generation_prompt),
            Self::ChatML => Self::apply_chatml(messages, add_generation_prompt),
            Self::Mistral => Self::apply_mistral(messages, add_generation_prompt),
            Self::Gemma => Self::apply_gemma(messages, add_generation_prompt),
            Self::Alpaca => Self::apply_alpaca(messages, add_generation_prompt),
        }
    }

    fn apply_llama3(
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut result = String::from("<|begin_of_text|>");

        for message in messages {
            let role = match message.role {
                infernum_core::Role::System => "system",
                infernum_core::Role::User => "user",
                infernum_core::Role::Assistant => "assistant",
                infernum_core::Role::Tool => "tool",
            };

            result.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n"));
            result.push_str(&message.content);
            result.push_str("<|eot_id|>");
        }

        if add_generation_prompt {
            result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        Ok(result)
    }

    fn apply_chatml(
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut result = String::new();

        for message in messages {
            let role = match message.role {
                infernum_core::Role::System => "system",
                infernum_core::Role::User => "user",
                infernum_core::Role::Assistant => "assistant",
                infernum_core::Role::Tool => "tool",
            };

            result.push_str(&format!("<|im_start|>{role}\n"));
            result.push_str(&message.content);
            result.push_str("<|im_end|>\n");
        }

        if add_generation_prompt {
            result.push_str("<|im_start|>assistant\n");
        }

        Ok(result)
    }

    fn apply_mistral(
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut result = String::from("<s>");
        let mut has_system = false;

        for message in messages {
            match message.role {
                infernum_core::Role::System => {
                    // Mistral prepends system message to first user message
                    has_system = true;
                    result.push_str(&format!("[INST] {}", message.content));
                },
                infernum_core::Role::User => {
                    if has_system {
                        result.push_str(&format!("\n\n{} [/INST]", message.content));
                        has_system = false;
                    } else {
                        result.push_str(&format!("[INST] {} [/INST]", message.content));
                    }
                },
                infernum_core::Role::Assistant => {
                    result.push_str(&format!(" {}</s>", message.content));
                },
                infernum_core::Role::Tool => {
                    result.push_str(&format!("[TOOL_RESULT] {} [/TOOL_RESULT]", message.content));
                },
            }
        }

        if add_generation_prompt && !has_system {
            // No additional prompt needed for Mistral
        }

        Ok(result)
    }

    fn apply_gemma(
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut result = String::new();

        for message in messages {
            let role = match message.role {
                infernum_core::Role::System | infernum_core::Role::User => "user",
                infernum_core::Role::Assistant => "model",
                infernum_core::Role::Tool => "user",
            };

            result.push_str(&format!("<start_of_turn>{role}\n"));
            result.push_str(&message.content);
            result.push_str("<end_of_turn>\n");
        }

        if add_generation_prompt {
            result.push_str("<start_of_turn>model\n");
        }

        Ok(result)
    }

    fn apply_alpaca(
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut result = String::new();

        for message in messages {
            match message.role {
                infernum_core::Role::System => {
                    result.push_str(&format!("### System:\n{}\n\n", message.content));
                },
                infernum_core::Role::User => {
                    result.push_str(&format!("### User:\n{}\n\n", message.content));
                },
                infernum_core::Role::Assistant => {
                    result.push_str(&format!("### Assistant:\n{}\n\n", message.content));
                },
                infernum_core::Role::Tool => {
                    result.push_str(&format!("### Tool:\n{}\n\n", message.content));
                },
            }
        }

        if add_generation_prompt {
            result.push_str("### Assistant:\n");
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    // Tests would require actual tokenizer files
}
