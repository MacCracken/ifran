/// Split text into chunks with overlap. Tries to break at whitespace boundaries.
///
/// All slice points are guaranteed to land on valid UTF-8 character boundaries,
/// so this is safe for multi-byte text (CJK, emoji, accented characters, etc.).
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

    while start < text.len() {
        let raw_end = (start + chunk_size).min(text.len());
        // Clamp to a valid char boundary (walk backward)
        let end = floor_char_boundary(text, raw_end);

        // If we're not at the end, try to break at a whitespace boundary
        let actual_end = if end < text.len() {
            // Look backward for whitespace on char boundaries
            let mut break_point = end;
            while break_point > start {
                if text.as_bytes()[break_point].is_ascii_whitespace()
                    || text.is_char_boundary(break_point)
                        && text[break_point..].starts_with(|c: char| c.is_whitespace())
                {
                    break;
                }
                break_point = floor_char_boundary(text, break_point.saturating_sub(1));
            }
            if break_point == start {
                end // No whitespace found, hard break at char boundary
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
        // Clamp new_start to a valid char boundary (walk forward)
        let new_start = ceil_char_boundary(text, new_start);
        if new_start <= start {
            break;
        }
        start = new_start;
    }

    chunks
}

/// Find the largest byte index <= `idx` that is a valid char boundary.
#[inline]
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let idx = idx.min(s.len());
    let mut i = idx;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the smallest byte index >= `idx` that is a valid char boundary.
#[inline]
fn ceil_char_boundary(s: &str, idx: usize) -> usize {
    let idx = idx.min(s.len());
    let mut i = idx;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
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

    #[test]
    fn multibyte_utf8_does_not_panic() {
        // CJK characters are 3 bytes each; chunk_size in bytes may land mid-char
        let text = "你好世界测试文本数据处理";
        let chunks = chunk_text(text, 7, 2);
        assert!(!chunks.is_empty());
        // Every chunk must be valid UTF-8 (implicit — they're Strings)
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn emoji_does_not_panic() {
        let text = "Hello 🌍🌎🌏 world 🚀🎉 testing 🔥";
        let chunks = chunk_text(text, 10, 3);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn mixed_ascii_and_multibyte() {
        let text = "café résumé naïve über straße";
        let chunks = chunk_text(text, 8, 2);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }
}
