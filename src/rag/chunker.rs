/// Split text into chunks with overlap. Tries to break at whitespace boundaries.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    // Handle edge cases
    if text.is_empty() || chunk_size == 0 {
        return vec![];
    }
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let step = if chunk_size > overlap {
        chunk_size - overlap
    } else {
        1
    };
    let mut chunks = Vec::with_capacity((text.len() / step) + 1);
    let mut start = 0;
    let bytes = text.as_bytes();

    while start < text.len() {
        let end = (start + chunk_size).min(text.len());

        // If we're not at the end, try to break at a whitespace boundary
        let actual_end = if end < text.len() {
            // Look backward for whitespace
            let mut break_point = end;
            while break_point > start && !bytes[break_point].is_ascii_whitespace() {
                break_point -= 1;
            }
            if break_point == start {
                end // No whitespace found, hard break
            } else {
                break_point
            }
        } else {
            end
        };

        chunks.push(text[start..actual_end].to_string());

        // Advance by (chunk_size - overlap), but at least 1
        let advance = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            1
        };
        let new_start = start + advance;
        if new_start <= start {
            break;
        }
        start = new_start;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_chunking() {
        let text = "Hello world this is a test of the chunking system that splits text";
        let chunks = chunk_text(text, 20, 0);
        assert!(chunks.len() > 1);
        // All chunks should be non-empty
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn overlap_produces_overlapping_content() {
        let text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let chunks = chunk_text(text, 20, 10);
        assert!(chunks.len() >= 2);
        // With overlap, later chunks should start before the previous chunk ended
        // Check that there's shared content between consecutive chunks
        for i in 0..chunks.len() - 1 {
            let chunk_a = &chunks[i];
            let chunk_b = &chunks[i + 1];
            // At least some words from chunk_a should appear in chunk_b
            let words_a: Vec<&str> = chunk_a.split_whitespace().collect();
            let words_b: Vec<&str> = chunk_b.split_whitespace().collect();
            let has_overlap = words_a.iter().any(|w| words_b.contains(w));
            assert!(
                has_overlap,
                "Expected overlap between chunks {} and {}",
                i,
                i + 1
            );
        }
    }

    #[test]
    fn empty_text_returns_empty() {
        let chunks = chunk_text("", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn zero_chunk_size_returns_empty() {
        let chunks = chunk_text("some text", 0, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn text_shorter_than_chunk_size_returns_single_chunk() {
        let text = "short text";
        let chunks = chunk_text(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn no_whitespace_forces_hard_break() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk_text(text, 10, 0);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].len(), 10);
    }

    #[test]
    fn large_overlap() {
        // overlap >= chunk_size should still terminate
        let text = "hello world foo bar baz qux";
        let chunks = chunk_text(text, 5, 100);
        assert!(!chunks.is_empty());
    }
}
