//! Document chunking strategies.

/// Strategy for splitting documents into chunks.
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Fixed token count with overlap.
    FixedTokens {
        /// Chunk size in tokens.
        size: usize,
        /// Overlap between chunks.
        overlap: usize,
    },
    /// Recursive character splitting.
    Recursive {
        /// Separators to split on.
        separators: Vec<String>,
        /// Maximum chunk size.
        chunk_size: usize,
    },
    /// Sentence-based chunking.
    Sentence {
        /// Minimum chunk size.
        min_size: usize,
        /// Maximum chunk size.
        max_size: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::FixedTokens {
            size: 512,
            overlap: 50,
        }
    }
}

/// A chunk of text.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The chunk text.
    pub text: String,
    /// Start offset in original document.
    pub start: usize,
    /// End offset in original document.
    pub end: usize,
    /// Chunk index.
    pub index: usize,
}

/// Document chunker.
pub struct Chunker {
    strategy: ChunkingStrategy,
}

impl Chunker {
    /// Creates a new chunker with the given strategy.
    #[must_use]
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Chunks a document.
    #[must_use]
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        match &self.strategy {
            ChunkingStrategy::FixedTokens { size, overlap } => {
                self.chunk_fixed(text, *size, *overlap)
            }
            ChunkingStrategy::Recursive {
                separators,
                chunk_size,
            } => self.chunk_recursive(text, separators, *chunk_size),
            ChunkingStrategy::Sentence { min_size, max_size } => {
                self.chunk_sentence(text, *min_size, *max_size)
            }
        }
    }

    /// Fixed-size chunking.
    fn chunk_fixed(&self, text: &str, size: usize, overlap: usize) -> Vec<Chunk> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < chars.len() {
            let end = (start + size).min(chars.len());
            let chunk_text: String = chars[start..end].iter().collect();

            chunks.push(Chunk {
                text: chunk_text,
                start,
                end,
                index,
            });

            if end >= chars.len() {
                break;
            }

            start = end.saturating_sub(overlap);
            index += 1;
        }

        chunks
    }

    /// Recursive splitting.
    fn chunk_recursive(&self, text: &str, separators: &[String], max_size: usize) -> Vec<Chunk> {
        // Simple implementation - just split and respect max size
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut start = 0;
        let mut index = 0;

        for (i, c) in text.char_indices() {
            current.push(c);

            if current.len() >= max_size {
                // Find last separator
                let split_at = separators
                    .iter()
                    .filter_map(|sep| current.rfind(sep.as_str()))
                    .max()
                    .unwrap_or(current.len());

                let chunk_text = current[..split_at].to_string();
                if !chunk_text.trim().is_empty() {
                    chunks.push(Chunk {
                        text: chunk_text,
                        start,
                        end: start + split_at,
                        index,
                    });
                    index += 1;
                }

                current = current[split_at..].to_string();
                start = i - current.len() + 1;
            }
        }

        if !current.trim().is_empty() {
            chunks.push(Chunk {
                text: current.clone(),
                start,
                end: start + current.len(),
                index,
            });
        }

        chunks
    }

    /// Sentence-based chunking.
    fn chunk_sentence(&self, text: &str, min_size: usize, max_size: usize) -> Vec<Chunk> {
        let sentences: Vec<&str> = text.split_terminator(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut start = 0;
        let mut index = 0;

        for sentence in sentences {
            let with_period = format!("{}. ", sentence);

            if current.len() + with_period.len() > max_size && current.len() >= min_size {
                chunks.push(Chunk {
                    text: current.trim().to_string(),
                    start,
                    end: start + current.len(),
                    index,
                });
                index += 1;
                start += current.len();
                current = String::new();
            }

            current.push_str(&with_period);
        }

        if current.len() >= min_size || chunks.is_empty() {
            chunks.push(Chunk {
                text: current.trim().to_string(),
                start,
                end: start + current.len(),
                index,
            });
        }

        chunks
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(ChunkingStrategy::default())
    }
}
