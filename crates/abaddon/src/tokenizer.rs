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
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| infernum_core::Error::Tokenization {
                message: e.to_string(),
            })?;

        Ok(Self::from_tokenizer(inner))
    }

    /// Creates a tokenizer from a pre-trained model name.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot be loaded.
    pub fn from_pretrained(name: &str) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_pretrained(name, None)
            .map_err(|e| infernum_core::Error::Tokenization {
                message: e.to_string(),
            })?;

        Ok(Self::from_tokenizer(inner))
    }

    /// Creates a wrapper from an existing tokenizer.
    fn from_tokenizer(inner: tokenizers::Tokenizer) -> Self {
        // Try to extract special token IDs
        let bos_token_id = inner
            .get_added_vocabulary()
            .get_vocab(true)
            .get("<s>")
            .or_else(|| inner.get_added_vocabulary().get_vocab(true).get("<|begin_of_text|>"))
            .copied();

        let eos_token_id = inner
            .get_added_vocabulary()
            .get_vocab(true)
            .get("</s>")
            .or_else(|| inner.get_added_vocabulary().get_vocab(true).get("<|end_of_text|>"))
            .or_else(|| inner.get_added_vocabulary().get_vocab(true).get("<|eot_id|>"))
            .copied();

        let pad_token_id = inner
            .get_added_vocabulary()
            .get_vocab(true)
            .get("<pad>")
            .or_else(|| inner.get_added_vocabulary().get_vocab(true).get("[PAD]"))
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
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| infernum_core::Error::Tokenization {
                message: e.to_string(),
            })?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes token IDs to text.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| infernum_core::Error::Tokenization {
                message: e.to_string(),
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
    /// # Errors
    ///
    /// Returns an error if the template cannot be applied.
    pub fn apply_chat_template(
        &self,
        messages: &[infernum_core::Message],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // TODO: Implement proper chat template handling
        // For now, use a simple format
        let mut result = String::new();

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
}

#[cfg(test)]
mod tests {
    // Tests would require actual tokenizer files
}
