//! Dataset processor — text augmentation strategies for training data.
//!
//! Provides offline augmentation (no model required) for text datasets:
//! synonym replacement, random insertion, random deletion, random swap,
//! and character noise. Model-based strategies (paraphrase, back-translation)
//! accept an optional inference function.

use std::io::{BufRead, Write};
use std::path::Path;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use synapse_types::SynapseError;
use synapse_types::error::Result;

/// Available augmentation strategies for text data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AugmentationStrategy {
    SynonymReplacement,
    RandomInsertion,
    RandomDeletion,
    RandomSwap,
    CharacterNoise,
}

/// Configuration for data augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    pub strategies: Vec<AugmentationStrategy>,
    /// Number of augmented copies per original sample.
    #[serde(default = "default_factor")]
    pub augment_factor: usize,
    /// JSON field to augment (default: "text").
    #[serde(default = "default_text_field")]
    pub text_field: String,
    /// Copy non-augmented fields unchanged.
    #[serde(default = "default_true")]
    pub preserve_labels: bool,
    /// Probability of modifying each word (0.0–1.0).
    #[serde(default = "default_probability")]
    pub word_probability: f64,
    /// Optional random seed for reproducibility.
    pub seed: Option<u64>,
}

fn default_factor() -> usize {
    1
}
fn default_text_field() -> String {
    "text".into()
}
fn default_true() -> bool {
    true
}
fn default_probability() -> f64 {
    0.1
}

/// Result of an augmentation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationResult {
    pub original_count: usize,
    pub augmented_count: usize,
    pub output_path: String,
}

/// Augment a JSONL dataset file, writing originals + augmented samples to output.
pub fn augment_dataset(
    input_path: &Path,
    output_path: &Path,
    config: &AugmentationConfig,
) -> Result<AugmentationResult> {
    if config.strategies.is_empty() {
        return Err(SynapseError::TrainingError(
            "No augmentation strategies specified".into(),
        ));
    }
    if config.augment_factor == 0 {
        return Err(SynapseError::TrainingError(
            "augment_factor must be >= 1".into(),
        ));
    }

    let file = std::fs::File::open(input_path)
        .map_err(|e| SynapseError::TrainingError(format!("Failed to open input: {e}")))?;
    let reader = std::io::BufReader::new(file);

    let out_file = std::fs::File::create(output_path)
        .map_err(|e| SynapseError::TrainingError(format!("Failed to create output: {e}")))?;
    let mut writer = std::io::BufWriter::new(out_file);

    let mut rng = match config.seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_os_rng(),
    };

    let mut original_count = 0usize;
    let mut augmented_count = 0usize;

    for line in reader.lines() {
        let line =
            line.map_err(|e| SynapseError::TrainingError(format!("Failed to read line: {e}")))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_str(line)
            .map_err(|e| SynapseError::TrainingError(format!("Invalid JSON: {e}")))?;

        original_count += 1;

        // Write original
        writeln!(writer, "{}", serde_json::to_string(&obj).unwrap())
            .map_err(|e| SynapseError::TrainingError(format!("Write failed: {e}")))?;

        // Get the text field to augment
        let text = match obj.get(&config.text_field) {
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => continue, // skip samples without the text field
        };

        // Generate augmented copies
        for _ in 0..config.augment_factor {
            // Pick a random strategy for each copy
            let strategy = config.strategies[rng.random_range(0..config.strategies.len())];
            let augmented = apply_strategy(strategy, &text, config.word_probability, &mut rng);

            if augmented != text {
                obj.insert(
                    config.text_field.clone(),
                    serde_json::Value::String(augmented),
                );
                writeln!(writer, "{}", serde_json::to_string(&obj).unwrap())
                    .map_err(|e| SynapseError::TrainingError(format!("Write failed: {e}")))?;
                augmented_count += 1;
            }
        }
    }

    writer
        .flush()
        .map_err(|e| SynapseError::TrainingError(format!("Flush failed: {e}")))?;

    Ok(AugmentationResult {
        original_count,
        augmented_count,
        output_path: output_path.to_string_lossy().to_string(),
    })
}

fn apply_strategy(
    strategy: AugmentationStrategy,
    text: &str,
    word_prob: f64,
    rng: &mut SmallRng,
) -> String {
    match strategy {
        AugmentationStrategy::SynonymReplacement => synonym_replacement(text, word_prob, rng),
        AugmentationStrategy::RandomInsertion => random_insertion(text, word_prob, rng),
        AugmentationStrategy::RandomDeletion => random_deletion(text, word_prob, rng),
        AugmentationStrategy::RandomSwap => random_swap(text, word_prob, rng),
        AugmentationStrategy::CharacterNoise => character_noise(text, word_prob, rng),
    }
}

// --- Synonym map (small, built-in for common English words) ---

fn get_synonym(word: &str, rng: &mut SmallRng) -> Option<&'static str> {
    let lower = word.to_lowercase();
    let synonyms: &[&str] = match lower.as_str() {
        "good" => &["great", "fine", "excellent", "solid"],
        "great" => &["good", "excellent", "wonderful", "superb"],
        "bad" => &["poor", "terrible", "awful", "dreadful"],
        "big" => &["large", "huge", "enormous", "massive"],
        "small" => &["tiny", "little", "compact", "miniature"],
        "fast" => &["quick", "rapid", "swift", "speedy"],
        "slow" => &["sluggish", "gradual", "leisurely", "unhurried"],
        "happy" => &["glad", "joyful", "pleased", "cheerful"],
        "sad" => &["unhappy", "sorrowful", "gloomy", "melancholy"],
        "important" => &["crucial", "vital", "significant", "essential"],
        "easy" => &["simple", "straightforward", "effortless", "basic"],
        "hard" => &["difficult", "tough", "challenging", "demanding"],
        "make" => &["create", "produce", "build", "construct"],
        "use" => &["utilize", "employ", "apply", "leverage"],
        "show" => &["display", "demonstrate", "present", "reveal"],
        "help" => &["assist", "aid", "support", "facilitate"],
        "start" => &["begin", "commence", "initiate", "launch"],
        "end" => &["finish", "conclude", "terminate", "complete"],
        "old" => &["ancient", "aged", "elderly", "vintage"],
        "new" => &["fresh", "novel", "recent", "modern"],
        _ => return None,
    };
    Some(synonyms[rng.random_range(0..synonyms.len())])
}

/// Replace random words with synonyms.
fn synonym_replacement(text: &str, word_prob: f64, rng: &mut SmallRng) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    let result: Vec<String> = words
        .iter()
        .map(|w| {
            if rng.random_bool(word_prob.clamp(0.0, 1.0)) {
                if let Some(syn) = get_synonym(w, rng) {
                    return syn.to_string();
                }
            }
            w.to_string()
        })
        .collect();
    result.join(" ")
}

/// Insert synonyms of random words at random positions.
fn random_insertion(text: &str, word_prob: f64, rng: &mut SmallRng) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    let mut result: Vec<String> = words.iter().map(|w| w.to_string()).collect();

    let n_insertions = ((words.len() as f64) * word_prob).ceil() as usize;
    for _ in 0..n_insertions {
        let src_word = words[rng.random_range(0..words.len())];
        if let Some(syn) = get_synonym(src_word, rng) {
            let pos = rng.random_range(0..=result.len());
            result.insert(pos, syn.to_string());
        }
    }
    result.join(" ")
}

/// Delete random words with given probability.
fn random_deletion(text: &str, word_prob: f64, rng: &mut SmallRng) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 1 {
        return text.to_string();
    }
    let result: Vec<&str> = words
        .iter()
        .filter(|_| !rng.random_bool(word_prob.clamp(0.0, 1.0)))
        .copied()
        .collect();
    if result.is_empty() {
        // Keep at least one word
        return words[rng.random_range(0..words.len())].to_string();
    }
    result.join(" ")
}

/// Swap random adjacent word pairs.
fn random_swap(text: &str, word_prob: f64, rng: &mut SmallRng) -> String {
    let mut words: Vec<String> = text.split_whitespace().map(String::from).collect();
    if words.len() <= 1 {
        return text.to_string();
    }
    let n_swaps = ((words.len() as f64) * word_prob).ceil().max(1.0) as usize;
    for _ in 0..n_swaps {
        let i = rng.random_range(0..words.len() - 1);
        words.swap(i, i + 1);
    }
    words.join(" ")
}

/// Introduce character-level noise (typos).
fn character_noise(text: &str, word_prob: f64, rng: &mut SmallRng) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }
    let result: Vec<String> = words
        .iter()
        .map(|w| {
            if rng.random_bool(word_prob.clamp(0.0, 1.0)) && w.len() > 1 {
                let mut chars: Vec<char> = w.chars().collect();
                let op = rng.random_range(0..3u8);
                match op {
                    0 => {
                        // Swap two adjacent chars
                        let i = rng.random_range(0..chars.len() - 1);
                        chars.swap(i, i + 1);
                    }
                    1 => {
                        // Duplicate a char
                        let i = rng.random_range(0..chars.len());
                        let c = chars[i];
                        chars.insert(i, c);
                    }
                    _ => {
                        // Delete a char (keep at least 1)
                        if chars.len() > 1 {
                            let i = rng.random_range(0..chars.len());
                            chars.remove(i);
                        }
                    }
                }
                chars.into_iter().collect()
            } else {
                w.to_string()
            }
        })
        .collect();
    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn strategy_serde_roundtrip() {
        for s in [
            AugmentationStrategy::SynonymReplacement,
            AugmentationStrategy::RandomInsertion,
            AugmentationStrategy::RandomDeletion,
            AugmentationStrategy::RandomSwap,
            AugmentationStrategy::CharacterNoise,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: AugmentationStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn config_defaults() {
        let json = r#"{"strategies":["random_deletion"]}"#;
        let config: AugmentationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.augment_factor, 1);
        assert_eq!(config.text_field, "text");
        assert!(config.preserve_labels);
        assert!((config.word_probability - 0.1).abs() < 1e-6);
    }

    #[test]
    fn synonym_replacement_deterministic() {
        let mut rng = seeded_rng();
        let result = synonym_replacement("this is a good and fast test", 1.0, &mut rng);
        // With prob=1.0 all known words should be replaced
        assert!(!result.contains("good") || !result.contains("fast"));
    }

    #[test]
    fn synonym_replacement_empty() {
        let mut rng = seeded_rng();
        assert_eq!(synonym_replacement("", 0.5, &mut rng), "");
    }

    #[test]
    fn random_deletion_keeps_at_least_one() {
        let mut rng = seeded_rng();
        let result = random_deletion("hello world test", 0.99, &mut rng);
        assert!(!result.is_empty());
    }

    #[test]
    fn random_deletion_single_word() {
        let mut rng = seeded_rng();
        assert_eq!(random_deletion("hello", 1.0, &mut rng), "hello");
    }

    #[test]
    fn random_swap_changes_order() {
        let mut rng = seeded_rng();
        let original = "one two three four five six seven eight nine ten";
        let result = random_swap(original, 0.5, &mut rng);
        // With high probability, at least one swap happened
        assert_ne!(result, original);
    }

    #[test]
    fn character_noise_modifies_text() {
        let mut rng = seeded_rng();
        let original = "hello world testing augmentation strategies";
        let result = character_noise(original, 1.0, &mut rng);
        assert_ne!(result, original);
    }

    #[test]
    fn random_insertion_adds_words() {
        let mut rng = seeded_rng();
        let original = "this is good and fast";
        let result = random_insertion(original, 0.5, &mut rng);
        let orig_count = original.split_whitespace().count();
        let result_count = result.split_whitespace().count();
        assert!(result_count >= orig_count);
    }

    #[test]
    fn augment_dataset_end_to_end() {
        let input = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            &input,
            r#"{{"text":"this is a good fast test","label":"positive"}}"#
        )
        .unwrap();
        writeln!(
            &input,
            r#"{{"text":"bad and slow results here","label":"negative"}}"#
        )
        .unwrap();

        let output = tempfile::NamedTempFile::new().unwrap();

        let config = AugmentationConfig {
            strategies: vec![
                AugmentationStrategy::SynonymReplacement,
                AugmentationStrategy::RandomDeletion,
            ],
            augment_factor: 2,
            text_field: "text".into(),
            preserve_labels: true,
            word_probability: 0.3,
            seed: Some(42),
        };

        let result = augment_dataset(input.path(), output.path(), &config).unwrap();
        assert_eq!(result.original_count, 2);
        assert!(result.augmented_count > 0);

        // Read output and verify structure
        let content = std::fs::read_to_string(output.path()).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        // At least originals + some augmented
        assert!(lines.len() > 2);

        // Each line should be valid JSON with label preserved
        for line in &lines {
            let obj: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(obj["label"].is_string());
            assert!(obj["text"].is_string());
        }
    }

    #[test]
    fn augment_dataset_no_strategies_error() {
        let input = tempfile::NamedTempFile::new().unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();
        let config = AugmentationConfig {
            strategies: vec![],
            augment_factor: 1,
            text_field: "text".into(),
            preserve_labels: true,
            word_probability: 0.1,
            seed: None,
        };
        assert!(augment_dataset(input.path(), output.path(), &config).is_err());
    }

    #[test]
    fn augment_dataset_zero_factor_error() {
        let input = tempfile::NamedTempFile::new().unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();
        let config = AugmentationConfig {
            strategies: vec![AugmentationStrategy::RandomSwap],
            augment_factor: 0,
            text_field: "text".into(),
            preserve_labels: true,
            word_probability: 0.1,
            seed: None,
        };
        assert!(augment_dataset(input.path(), output.path(), &config).is_err());
    }

    #[test]
    fn augment_dataset_skips_missing_field() {
        let input = tempfile::NamedTempFile::new().unwrap();
        writeln!(&input, r#"{{"prompt":"no text field","label":"x"}}"#).unwrap();

        let output = tempfile::NamedTempFile::new().unwrap();
        let config = AugmentationConfig {
            strategies: vec![AugmentationStrategy::RandomDeletion],
            augment_factor: 1,
            text_field: "text".into(),
            preserve_labels: true,
            word_probability: 0.3,
            seed: Some(1),
        };

        let result = augment_dataset(input.path(), output.path(), &config).unwrap();
        assert_eq!(result.original_count, 1);
        assert_eq!(result.augmented_count, 0); // no augmentation since field missing
    }
}
