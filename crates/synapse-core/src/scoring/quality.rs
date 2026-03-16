//! Inference quality scoring for training data filtering.
//!
//! Scores inference sessions to identify high-quality responses suitable
//! for use as training data. Supports multiple heuristic scoring criteria.

use serde::{Deserialize, Serialize};

/// Quality score for an inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall score (0.0 to 1.0).
    pub overall: f64,
    /// Individual criterion scores.
    pub criteria: Vec<CriterionScore>,
}

/// A single scoring criterion result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    pub name: String,
    pub score: f64,
    pub weight: f64,
}

/// Score an inference response using heuristic criteria.
///
/// Criteria:
/// - Length adequacy: penalizes very short or very long responses
/// - Completeness: checks for sentence-ending punctuation
/// - Repetition: penalizes repeated n-grams
/// - Coherence: checks vocabulary diversity (type-token ratio)
pub fn score_response(prompt: &str, response: &str) -> QualityScore {
    let criteria = vec![
        CriterionScore {
            name: "length_adequacy".into(),
            score: score_length(prompt, response),
            weight: 0.2,
        },
        CriterionScore {
            name: "completeness".into(),
            score: score_completeness(response),
            weight: 0.3,
        },
        CriterionScore {
            name: "repetition".into(),
            score: score_repetition(response),
            weight: 0.25,
        },
        CriterionScore {
            name: "coherence".into(),
            score: score_coherence(response),
            weight: 0.25,
        },
    ];

    let overall = criteria.iter().map(|c| c.score * c.weight).sum::<f64>()
        / criteria.iter().map(|c| c.weight).sum::<f64>();

    QualityScore { overall, criteria }
}

/// Filter responses by quality threshold.
pub fn filter_high_quality(
    pairs: &[(String, String)],
    threshold: f64,
) -> Vec<(usize, QualityScore)> {
    pairs
        .iter()
        .enumerate()
        .map(|(i, (prompt, response))| (i, score_response(prompt, response)))
        .filter(|(_, score)| score.overall >= threshold)
        .collect()
}

fn score_length(prompt: &str, response: &str) -> f64 {
    let prompt_words = prompt.split_whitespace().count();
    let response_words = response.split_whitespace().count();

    if response_words == 0 {
        return 0.0;
    }

    // Ideal response is 1-5x prompt length
    let ratio = response_words as f64 / (prompt_words.max(1) as f64);
    if ratio < 0.5 {
        ratio * 2.0 // too short
    } else if ratio <= 5.0 {
        1.0 // good range
    } else {
        (10.0 - ratio).max(0.0) / 5.0 // too long
    }
}

fn score_completeness(response: &str) -> f64 {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return 0.0;
    }

    // Check for sentence-ending punctuation
    let ends_properly = trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || trimmed.ends_with('?')
        || trimmed.ends_with(':');

    if ends_properly { 1.0 } else { 0.5 }
}

fn score_repetition(response: &str) -> f64 {
    let words: Vec<&str> = response.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    if words.len() < 4 {
        return 1.0;
    }

    // Count repeated trigrams
    let mut trigrams = std::collections::HashSet::new();
    let mut repeated = 0u64;
    for window in words.windows(3) {
        let trigram = format!(
            "{} {} {}",
            window[0].to_lowercase(),
            window[1].to_lowercase(),
            window[2].to_lowercase()
        );
        if !trigrams.insert(trigram) {
            repeated += 1;
        }
    }

    let total_trigrams = (words.len() - 2) as f64;
    let repetition_rate = repeated as f64 / total_trigrams;

    (1.0 - repetition_rate * 3.0).max(0.0) // 33% repetition -> 0 score
}

fn score_coherence(response: &str) -> f64 {
    let words: Vec<String> = response
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if words.is_empty() {
        return 0.0;
    }

    // Type-token ratio (vocabulary diversity)
    let unique: std::collections::HashSet<&str> = words.iter().map(|w| w.as_str()).collect();
    let ttr = unique.len() as f64 / words.len() as f64;

    // TTR naturally decreases with longer texts, so be generous
    (ttr * 1.5).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_good_response() {
        let score = score_response(
            "What is machine learning?",
            "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.",
        );
        assert!(score.overall > 0.6);
        assert_eq!(score.criteria.len(), 4);
    }

    #[test]
    fn score_empty_response() {
        let score = score_response("Hello?", "");
        assert_eq!(score.overall, 0.0);
    }

    #[test]
    fn score_repetitive_response() {
        let score = score_response(
            "Tell me about AI.",
            "AI is great AI is great AI is great AI is great AI is great.",
        );
        let rep = score
            .criteria
            .iter()
            .find(|c| c.name == "repetition")
            .unwrap();
        assert!(rep.score < 0.5);
    }

    #[test]
    fn score_incomplete_response() {
        let score = score_response("What is Rust?", "Rust is a programming language that");
        let completeness = score
            .criteria
            .iter()
            .find(|c| c.name == "completeness")
            .unwrap();
        assert_eq!(completeness.score, 0.5); // no ending punctuation
    }

    #[test]
    fn filter_by_threshold() {
        let pairs = vec![
            ("Q?".into(), "Good complete answer.".into()),
            ("Q?".into(), "".into()),
            ("Q?".into(), "Another solid response with detail.".into()),
        ];
        let high = filter_high_quality(&pairs, 0.5);
        assert!(high.len() >= 1); // empty response filtered out
    }

    #[test]
    fn length_score_ranges() {
        assert_eq!(score_length("hello", ""), 0.0);
        assert!(
            score_length(
                "short prompt",
                "a reasonable length response that is proportional"
            ) > 0.8
        );
    }

    #[test]
    fn coherence_diverse_text() {
        let score =
            score_coherence("The quick brown fox jumps over the lazy dog near a river bank.");
        assert!(score > 0.7);
    }

    #[test]
    fn coherence_repetitive_text() {
        let score = score_coherence("the the the the the the the");
        assert!(score < 0.5);
    }

    #[test]
    fn very_long_response_penalized() {
        let prompt = "What is AI?";
        // Response about 10x the prompt word count (well beyond 5x threshold)
        let long_response = "word ".repeat(100);
        let score = score_response(prompt, &long_response);
        let length_criterion = score
            .criteria
            .iter()
            .find(|c| c.name == "length_adequacy")
            .unwrap();
        assert!(length_criterion.score < 1.0);
    }

    #[test]
    fn perfect_response() {
        let score = score_response(
            "What is Rust?",
            "Rust is a systems programming language that emphasizes safety, performance, and concurrency without a garbage collector.",
        );
        // Well-formed, good length, no repetition, ends with punctuation.
        assert!(score.overall > 0.7);
    }

    #[test]
    fn single_word_response() {
        let score = score_response("Explain quantum computing in detail.", "Yes.");
        // Very short relative to prompt — length should be penalized.
        let length = score
            .criteria
            .iter()
            .find(|c| c.name == "length_adequacy")
            .unwrap();
        assert!(length.score < 1.0);
        // But completeness should be fine (ends with period).
        let completeness = score
            .criteria
            .iter()
            .find(|c| c.name == "completeness")
            .unwrap();
        assert_eq!(completeness.score, 1.0);
    }
}
