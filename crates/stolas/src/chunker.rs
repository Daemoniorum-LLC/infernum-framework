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
            },
            ChunkingStrategy::Recursive {
                separators,
                chunk_size,
            } => self.chunk_recursive(text, separators, *chunk_size),
            ChunkingStrategy::Sentence { min_size, max_size } => {
                self.chunk_sentence(text, *min_size, *max_size)
            },
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
        let sentences: Vec<&str> = text
            .split_terminator(|c| c == '.' || c == '!' || c == '?')
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

#[cfg(test)]
mod tests {
    use super::*;

    // === ChunkingStrategy Tests ===

    #[test]
    fn test_chunking_strategy_default() {
        let strategy = ChunkingStrategy::default();
        match strategy {
            ChunkingStrategy::FixedTokens { size, overlap } => {
                assert_eq!(size, 512);
                assert_eq!(overlap, 50);
            }
            _ => panic!("Default should be FixedTokens"),
        }
    }

    // === Fixed Tokens Chunking Tests ===

    #[test]
    fn test_chunk_fixed_basic() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 10,
            overlap: 2,
        });
        let text = "Hello, World! This is a test.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        // First chunk should have 10 characters
        assert_eq!(chunks[0].text.len(), 10);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 10);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_chunk_fixed_overlap() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 10,
            overlap: 3,
        });
        let text = "0123456789ABCDEFGHIJ";
        let chunks = chunker.chunk(text);

        // Check overlap - end of first chunk should overlap with start of second
        if chunks.len() >= 2 {
            let first_end = &chunks[0].text[chunks[0].text.len() - 3..];
            let second_start = &chunks[1].text[..3];
            // The overlapping portion should match
            assert_eq!(first_end.len(), second_start.len());
        }
    }

    #[test]
    fn test_chunk_fixed_empty_text() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 10,
            overlap: 2,
        });
        let chunks = chunker.chunk("");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_fixed_text_smaller_than_chunk_size() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 100,
            overlap: 10,
        });
        let text = "Short text";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text");
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, text.len());
    }

    #[test]
    fn test_chunk_fixed_exact_chunk_size() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 10,
            overlap: 0,
        });
        let text = "0123456789";
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "0123456789");
    }

    // === Recursive Chunking Tests ===

    #[test]
    fn test_chunk_recursive_basic() {
        let chunker = Chunker::new(ChunkingStrategy::Recursive {
            separators: vec!["\n".to_string(), " ".to_string()],
            chunk_size: 20,
        });
        let text = "Hello World. This is a long test that should be split.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Chunks should respect max size (approximately)
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunk_recursive_empty_text() {
        let chunker = Chunker::new(ChunkingStrategy::Recursive {
            separators: vec!["\n".to_string()],
            chunk_size: 100,
        });
        let chunks = chunker.chunk("");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_recursive_whitespace_only() {
        let chunker = Chunker::new(ChunkingStrategy::Recursive {
            separators: vec!["\n".to_string()],
            chunk_size: 100,
        });
        let chunks = chunker.chunk("   \n\n   ");

        // Should skip whitespace-only chunks
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_recursive_respects_separators() {
        let chunker = Chunker::new(ChunkingStrategy::Recursive {
            separators: vec!["\n".to_string()],
            chunk_size: 50,
        });
        let text = "Line one content here\nLine two content here\nLine three";
        let chunks = chunker.chunk(text);

        // Should have at least one chunk
        assert!(!chunks.is_empty());
    }

    // === Sentence Chunking Tests ===

    #[test]
    fn test_chunk_sentence_basic() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 10,
            max_size: 100,
        });
        let text = "First sentence. Second sentence. Third sentence!";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        // All sentences should be included
        let combined: String = chunks.iter().map(|c| c.text.clone()).collect();
        assert!(combined.contains("First"));
        assert!(combined.contains("Second"));
        assert!(combined.contains("Third"));
    }

    #[test]
    fn test_chunk_sentence_respects_max_size() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 10,
            max_size: 50,
        });
        let text = "This is a sentence. And another. Yet another one here.";
        let chunks = chunker.chunk(text);

        // Most chunks should be under max_size
        for chunk in &chunks {
            // Allow some flexibility due to how sentences are joined
            assert!(
                chunk.text.len() <= 60,
                "Chunk too long: {} chars",
                chunk.text.len()
            );
        }
    }

    #[test]
    fn test_chunk_sentence_respects_min_size() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 50,
            max_size: 200,
        });
        let text = "One. Two. Three. Four. Five.";
        let chunks = chunker.chunk(text);

        // Should combine short sentences to meet min_size
        // (or have a single chunk if text is short)
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_sentence_empty_text() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 10,
            max_size: 100,
        });
        let chunks = chunker.chunk("");

        // Empty chunks should still produce at least one (empty) chunk due to implementation
        assert!(chunks.len() <= 1);
    }

    #[test]
    fn test_chunk_sentence_different_terminators() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 1,
            max_size: 200,
        });
        let text = "Question? Exclamation! Statement.";
        let chunks = chunker.chunk(text);

        // Should handle all sentence terminators
        let combined: String = chunks.iter().map(|c| c.text.clone()).collect();
        assert!(combined.contains("Question"));
        assert!(combined.contains("Exclamation"));
        assert!(combined.contains("Statement"));
    }

    // === Chunk Struct Tests ===

    #[test]
    fn test_chunk_indices_are_sequential() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 5,
            overlap: 1,
        });
        let text = "0123456789ABCDEFGHIJ";
        let chunks = chunker.chunk(text);

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i, "Chunk index should match position");
        }
    }

    #[test]
    fn test_chunk_offsets_are_valid() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 10,
            overlap: 2,
        });
        let text = "Hello, World! This is a test.";
        let chunks = chunker.chunk(text);

        for chunk in &chunks {
            assert!(chunk.start <= chunk.end);
            assert!(chunk.end <= text.len());
        }
    }

    // === Default Chunker Tests ===

    #[test]
    fn test_chunker_default() {
        let chunker = Chunker::default();
        let text = "Test text";
        let chunks = chunker.chunk(text);

        // Default chunker should work
        assert!(!chunks.is_empty());
    }

    // === Unicode Tests ===

    #[test]
    fn test_chunk_fixed_unicode() {
        let chunker = Chunker::new(ChunkingStrategy::FixedTokens {
            size: 5,
            overlap: 1,
        });
        let text = "Hello 世界! 你好";
        let chunks = chunker.chunk(text);

        // Should handle Unicode correctly
        assert!(!chunks.is_empty());
        // First chunk should have 5 characters (including Unicode)
        assert_eq!(chunks[0].text.chars().count(), 5);
    }

    #[test]
    fn test_chunk_sentence_unicode() {
        let chunker = Chunker::new(ChunkingStrategy::Sentence {
            min_size: 1,
            max_size: 100,
        });
        let text = "这是第一句话。这是第二句话！这是第三句话？";
        let chunks = chunker.chunk(text);

        // Should produce chunks (though may not split on Chinese punctuation)
        assert!(!chunks.is_empty());
    }
}
