//! Benchmark implementations.
//!
//! Each benchmark takes inference results and computes a score.
//! Supports: perplexity, MMLU, HellaSwag, HumanEval, and custom benchmarks.

use crate::types::eval::BenchmarkKind;
use serde::Deserialize;

/// A single evaluation sample.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalSample {
    pub prompt: String,
    pub expected: String,
    /// Multiple-choice options (for MMLU-style benchmarks).
    pub choices: Option<Vec<String>>,
    /// Correct choice index.
    pub answer_index: Option<usize>,
}

/// Computed score from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkScore {
    pub kind: BenchmarkKind,
    pub score: f64,
    pub samples_evaluated: u64,
}

/// Score a custom benchmark: exact-match accuracy.
///
/// Compares model outputs to expected outputs, returning the fraction
/// of exact matches (case-insensitive, trimmed).
#[inline]
#[must_use]
pub fn score_exact_match(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, expected)| pred.trim().eq_ignore_ascii_case(expected.trim()))
        .count();
    correct as f64 / predictions.len() as f64
}

#[inline]
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Score a custom benchmark: contains-match accuracy.
///
/// Checks if the model output contains the expected answer.
#[inline]
#[must_use]
pub fn score_contains_match(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, expected)| contains_ignore_ascii_case(pred, expected))
        .count();
    correct as f64 / predictions.len() as f64
}

/// Score MMLU-style multiple-choice: extract first letter (A/B/C/D) from
/// model output and compare to expected answer letter.
#[inline]
#[must_use]
pub fn score_mmlu(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, expected)| {
            let pred_letter = extract_answer_letter(pred);
            let expected_letter = extract_answer_letter(expected);
            pred_letter.is_some() && pred_letter == expected_letter
        })
        .count();
    correct as f64 / predictions.len() as f64
}

/// Extract the first A/B/C/D answer letter from a response.
///
/// Looks for patterns like "A", "A)", "(A)", "A.", or standalone letter
/// at the beginning. Falls back to scanning for the first isolated
/// A/B/C/D in the text.
#[inline]
fn extract_answer_letter(text: &str) -> Option<char> {
    let trimmed = text.trim();

    // If the entire trimmed text is a single letter A-D
    if trimmed.len() == 1 {
        let ch = trimmed.chars().next()?.to_ascii_uppercase();
        if matches!(ch, 'A' | 'B' | 'C' | 'D') {
            return Some(ch);
        }
    }

    // Look for patterns: "(A)", "A)", "A.", or standalone A-D at start
    let upper = trimmed.to_uppercase();
    for pattern in [
        "(A)", "(B)", "(C)", "(D)", "A)", "B)", "C)", "D)", "A.", "B.", "C.", "D.",
    ] {
        if upper.starts_with(pattern) {
            return pattern.chars().find(|c| matches!(c, 'A' | 'B' | 'C' | 'D'));
        }
    }

    // Scan for first isolated A/B/C/D (preceded by space/start, followed by non-alpha)
    let chars: Vec<char> = upper.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        if !matches!(ch, 'A' | 'B' | 'C' | 'D') {
            continue;
        }
        let prev_ok = i == 0 || !chars[i - 1].is_alphabetic();
        let next_ok = i + 1 >= chars.len() || !chars[i + 1].is_alphabetic();
        if prev_ok && next_ok {
            return Some(ch);
        }
    }

    None
}

/// Compute approximate perplexity from sliding-window predictions.
///
/// Since we don't have token-level log-probs, we approximate by checking
/// whether the model can reproduce segments of text. Lower score = better.
/// Returns the inverse of the average contains-match rate, bounded to [1.0, 1000.0].
#[must_use]
pub fn score_perplexity(predictions: &[(String, String)]) -> f64 {
    if predictions.is_empty() {
        return 1000.0;
    }
    let match_rate = score_contains_match(predictions);
    if match_rate <= 0.001 {
        1000.0
    } else {
        (1.0 / match_rate).min(1000.0)
    }
}

/// Format an MMLU-style multiple-choice prompt.
#[must_use]
pub fn format_mmlu_prompt(sample: &EvalSample) -> String {
    let mut prompt = format!("Question: {}\n", sample.prompt);
    if let Some(ref choices) = sample.choices {
        let labels = ['A', 'B', 'C', 'D', 'E', 'F'];
        for (i, choice) in choices.iter().enumerate() {
            let label = labels.get(i).copied().unwrap_or('?');
            prompt.push_str(&format!("{label}) {choice}\n"));
        }
    }
    prompt.push_str("Answer:");
    prompt
}

/// Format a HellaSwag-style completion prompt.
#[must_use]
pub fn format_hellaswag_prompt(sample: &EvalSample) -> String {
    format!("{}\nComplete the following:", sample.prompt)
}

/// Format a HumanEval-style code generation prompt.
#[must_use]
pub fn format_humaneval_prompt(sample: &EvalSample) -> String {
    format!(
        "Complete the following Python function:\n\n{}\n",
        sample.prompt
    )
}

/// Format a perplexity measurement prompt — ask model to continue text.
#[must_use]
pub fn format_perplexity_prompt(sample: &EvalSample) -> String {
    // Use first half of expected as context, ask model to generate continuation
    let words: Vec<&str> = sample.expected.split_whitespace().collect();
    let split = words.len() / 2;
    if split > 0 {
        let context = words[..split].join(" ");
        format!("Continue the following text:\n\n{context}")
    } else {
        format!("Continue the following text:\n\n{}", sample.prompt)
    }
}

/// Get the expected answer letter for an MMLU sample.
#[must_use]
pub fn mmlu_expected_letter(sample: &EvalSample) -> String {
    if let Some(idx) = sample.answer_index {
        let labels = ['A', 'B', 'C', 'D', 'E', 'F'];
        labels
            .get(idx)
            .map(|c| c.to_string())
            .unwrap_or_else(|| sample.expected.clone())
    } else {
        sample.expected.clone()
    }
}

/// Load eval samples from a JSONL file.
///
/// Uses buffered line-by-line reading to avoid loading the entire file into memory.
pub fn load_samples(
    path: &str,
    limit: Option<usize>,
) -> crate::types::error::Result<Vec<EvalSample>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut samples = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let sample: EvalSample = serde_json::from_str(&line).map_err(|e| {
            crate::types::IfranError::EvalError(format!("Invalid eval sample: {e}"))
        })?;
        samples.push(sample);
        if let Some(max) = limit {
            if samples.len() >= max {
                break;
            }
        }
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_scoring() {
        let preds = vec![
            ("Paris".into(), "Paris".into()),
            ("london".into(), "London".into()),
            ("wrong".into(), "Berlin".into()),
        ];
        let score = score_exact_match(&preds);
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn contains_match_scoring() {
        let preds = vec![
            ("The answer is Paris, the capital".into(), "Paris".into()),
            ("I think it's Berlin".into(), "Berlin".into()),
            ("No idea".into(), "Tokyo".into()),
        ];
        let score = score_contains_match(&preds);
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn empty_predictions() {
        assert_eq!(score_exact_match(&[]), 0.0);
        assert_eq!(score_contains_match(&[]), 0.0);
    }

    #[test]
    fn mmlu_scoring() {
        let preds = vec![
            ("A".into(), "A".into()),
            ("B) Paris".into(), "B".into()),
            ("The answer is C".into(), "C".into()),
            ("D".into(), "A".into()), // wrong
        ];
        let score = score_mmlu(&preds);
        assert!((score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn extract_answer_letter_variants() {
        assert_eq!(extract_answer_letter("A"), Some('A'));
        assert_eq!(extract_answer_letter("B) Paris"), Some('B'));
        assert_eq!(extract_answer_letter("  C"), Some('C'));
        assert_eq!(extract_answer_letter("(D)"), Some('D'));
        assert_eq!(extract_answer_letter(""), None);
    }

    #[test]
    fn perplexity_scoring() {
        // Perfect prediction → perplexity ~1.0
        let preds = vec![("hello world".into(), "hello".into())];
        let score = score_perplexity(&preds);
        assert!((score - 1.0).abs() < 1e-6);

        // No matches → perplexity 1000
        let preds = vec![("foo".into(), "bar".into())];
        let score = score_perplexity(&preds);
        assert!((score - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn format_mmlu() {
        let sample = EvalSample {
            prompt: "What is the capital of France?".into(),
            expected: "A".into(),
            choices: Some(vec![
                "Paris".into(),
                "London".into(),
                "Berlin".into(),
                "Madrid".into(),
            ]),
            answer_index: Some(0),
        };
        let prompt = format_mmlu_prompt(&sample);
        assert!(prompt.contains("A) Paris"));
        assert!(prompt.contains("D) Madrid"));
        assert!(prompt.ends_with("Answer:"));
    }

    #[test]
    fn format_hellaswag() {
        let sample = EvalSample {
            prompt: "The cat sat on".into(),
            expected: "the mat".into(),
            choices: None,
            answer_index: None,
        };
        let prompt = format_hellaswag_prompt(&sample);
        assert!(prompt.contains("The cat sat on"));
        assert!(prompt.contains("Complete the following"));
    }

    #[test]
    fn format_humaneval() {
        let sample = EvalSample {
            prompt: "def add(a, b):".into(),
            expected: "return a + b".into(),
            choices: None,
            answer_index: None,
        };
        let prompt = format_humaneval_prompt(&sample);
        assert!(prompt.contains("def add(a, b):"));
        assert!(prompt.contains("Python function"));
    }

    #[test]
    fn mmlu_expected_from_index() {
        let sample = EvalSample {
            prompt: "test".into(),
            expected: "A".into(),
            choices: Some(vec!["x".into(), "y".into()]),
            answer_index: Some(1),
        };
        assert_eq!(mmlu_expected_letter(&sample), "B");
    }

    #[test]
    fn perplexity_prompt_splits_text() {
        let sample = EvalSample {
            prompt: "test".into(),
            expected: "one two three four".into(),
            choices: None,
            answer_index: None,
        };
        let prompt = format_perplexity_prompt(&sample);
        assert!(prompt.contains("one two"));
        assert!(prompt.contains("Continue"));
    }

    #[test]
    fn perplexity_prompt_single_word() {
        let sample = EvalSample {
            prompt: "hello".into(),
            expected: "world".into(),
            choices: None,
            answer_index: None,
        };
        let prompt = format_perplexity_prompt(&sample);
        // Single word splits at 0, so falls through to prompt-based
        assert!(prompt.contains("hello"));
    }

    #[test]
    fn extract_answer_letter_lowercase() {
        assert_eq!(extract_answer_letter("a"), Some('A'));
        assert_eq!(extract_answer_letter("b)"), Some('B'));
        assert_eq!(extract_answer_letter("(c)"), Some('C'));
    }

    #[test]
    fn extract_answer_letter_no_match() {
        assert_eq!(extract_answer_letter("xyz"), None);
        assert_eq!(extract_answer_letter("123"), None);
        assert_eq!(extract_answer_letter("E"), None);
    }

    #[test]
    fn extract_answer_letter_embedded() {
        // "The answer is A" → should find A
        assert_eq!(extract_answer_letter("The answer is A"), Some('A'));
    }

    #[test]
    fn mmlu_expected_letter_no_index() {
        let sample = EvalSample {
            prompt: "test".into(),
            expected: "C".into(),
            choices: None,
            answer_index: None,
        };
        assert_eq!(mmlu_expected_letter(&sample), "C");
    }

    #[test]
    fn mmlu_expected_letter_out_of_range() {
        let sample = EvalSample {
            prompt: "test".into(),
            expected: "fallback".into(),
            choices: Some(vec!["x".into()]),
            answer_index: Some(99),
        };
        assert_eq!(mmlu_expected_letter(&sample), "fallback");
    }

    #[test]
    fn score_exact_match_all_correct() {
        let preds = vec![("A".into(), "a".into()), ("B".into(), "b".into())];
        assert!((score_exact_match(&preds) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_exact_match_whitespace() {
        let preds = vec![("  hello  ".into(), "hello".into())];
        assert!((score_exact_match(&preds) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_contains_match_empty_expected() {
        let preds = vec![("anything".into(), "".into())];
        // Empty string is always contained
        assert!((score_contains_match(&preds) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_mmlu_empty() {
        assert_eq!(score_mmlu(&[]), 0.0);
    }

    #[test]
    fn perplexity_empty_predictions() {
        assert!((score_perplexity(&[]) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn format_mmlu_no_choices() {
        let sample = EvalSample {
            prompt: "What?".into(),
            expected: "A".into(),
            choices: None,
            answer_index: None,
        };
        let prompt = format_mmlu_prompt(&sample);
        assert!(prompt.contains("What?"));
        assert!(prompt.ends_with("Answer:"));
    }

    #[test]
    fn load_samples_from_file() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        writeln!(tmp, r#"{{"prompt": "hello", "expected": "world"}}"#).unwrap();
        writeln!(tmp, r#"{{"prompt": "foo", "expected": "bar"}}"#).unwrap();
        tmp.flush().unwrap();

        let samples = load_samples(tmp.path().to_str().unwrap(), None).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].prompt, "hello");
        assert_eq!(samples[1].expected, "bar");
    }

    #[test]
    fn load_samples_with_limit() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        for i in 0..10 {
            writeln!(tmp, r#"{{"prompt": "p{i}", "expected": "e{i}"}}"#).unwrap();
        }
        tmp.flush().unwrap();

        let samples = load_samples(tmp.path().to_str().unwrap(), Some(3)).unwrap();
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn load_samples_skips_blank_lines() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        writeln!(tmp, r#"{{"prompt": "a", "expected": "b"}}"#).unwrap();
        writeln!(tmp).unwrap();
        writeln!(tmp, r#"{{"prompt": "c", "expected": "d"}}"#).unwrap();
        tmp.flush().unwrap();

        let samples = load_samples(tmp.path().to_str().unwrap(), None).unwrap();
        assert_eq!(samples.len(), 2);
    }
}
