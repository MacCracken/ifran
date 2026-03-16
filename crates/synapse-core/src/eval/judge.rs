//! LLM-as-judge evaluation for pairwise model comparison.
//!
//! Uses a judge model to score and compare responses from two models
//! on the same prompts, producing win/loss/tie statistics.

use serde::{Deserialize, Serialize};

/// Result of a pairwise LLM judge evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    pub model_a: String,
    pub model_b: String,
    pub wins_a: u32,
    pub wins_b: u32,
    pub ties: u32,
    pub total: u32,
}

impl JudgeResult {
    pub fn win_rate_a(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.wins_a as f64 / self.total as f64
        }
    }
    pub fn win_rate_b(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.wins_b as f64 / self.total as f64
        }
    }
}

/// A single judge verdict for one prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    WinA,
    WinB,
    Tie,
}

/// A scoring rubric for the judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeRubric {
    pub criteria: Vec<String>,
    pub scale: String,
    pub system_prompt: String,
}

impl Default for JudgeRubric {
    fn default() -> Self {
        Self {
            criteria: vec![
                "Accuracy and correctness".into(),
                "Helpfulness and relevance".into(),
                "Clarity and coherence".into(),
                "Safety and harmlessness".into(),
            ],
            scale: "1-5".into(),
            system_prompt: "You are an impartial judge evaluating two AI responses. \
                Score each response on the given criteria. Output your verdict as: \
                A (first is better), B (second is better), or TIE."
                .into(),
        }
    }
}

/// Build the judge prompt for a pairwise comparison.
pub fn build_judge_prompt(
    rubric: &JudgeRubric,
    prompt: &str,
    response_a: &str,
    response_b: &str,
) -> String {
    let criteria_list = rubric
        .criteria
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {c}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "{system}\n\n\
         Criteria (rate each {scale}):\n{criteria}\n\n\
         User prompt: {prompt}\n\n\
         Response A:\n{response_a}\n\n\
         Response B:\n{response_b}\n\n\
         Verdict (A, B, or TIE):",
        system = rubric.system_prompt,
        scale = rubric.scale,
        criteria = criteria_list,
    )
}

/// Parse a verdict from judge model output.
pub fn parse_verdict(output: &str) -> Verdict {
    let lower = output.trim().to_lowercase();
    // Look for the verdict keyword
    if lower.contains("verdict: a") || lower.contains("winner: a") || lower.ends_with(" a") {
        Verdict::WinA
    } else if lower.contains("verdict: b") || lower.contains("winner: b") || lower.ends_with(" b") {
        Verdict::WinB
    } else {
        // "tie", "draw", or unclear → default to tie
        Verdict::Tie
    }
}

/// Aggregate verdicts into a JudgeResult.
pub fn aggregate_verdicts(model_a: &str, model_b: &str, verdicts: &[Verdict]) -> JudgeResult {
    let mut result = JudgeResult {
        model_a: model_a.into(),
        model_b: model_b.into(),
        wins_a: 0,
        wins_b: 0,
        ties: 0,
        total: verdicts.len() as u32,
    };
    for v in verdicts {
        match v {
            Verdict::WinA => result.wins_a += 1,
            Verdict::WinB => result.wins_b += 1,
            Verdict::Tie => result.ties += 1,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rubric() {
        let r = JudgeRubric::default();
        assert_eq!(r.criteria.len(), 4);
        assert!(r.system_prompt.contains("impartial"));
    }

    #[test]
    fn build_prompt_includes_all_parts() {
        let r = JudgeRubric::default();
        let prompt = build_judge_prompt(&r, "What is Rust?", "Rust is great.", "Rust is ok.");
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Response A:"));
        assert!(prompt.contains("Response B:"));
        assert!(prompt.contains("Accuracy"));
    }

    #[test]
    fn parse_verdict_win_a() {
        assert_eq!(parse_verdict("Verdict: A"), Verdict::WinA);
        assert_eq!(parse_verdict("The winner is A"), Verdict::WinA);
    }

    #[test]
    fn parse_verdict_win_b() {
        assert_eq!(parse_verdict("Verdict: B"), Verdict::WinB);
    }

    #[test]
    fn parse_verdict_tie() {
        assert_eq!(parse_verdict("It's a tie"), Verdict::Tie);
        assert_eq!(parse_verdict("unclear nonsense"), Verdict::Tie);
    }

    #[test]
    fn aggregate_verdicts_test() {
        let verdicts = vec![Verdict::WinA, Verdict::WinB, Verdict::WinA, Verdict::Tie];
        let result = aggregate_verdicts("m1", "m2", &verdicts);
        assert_eq!(result.wins_a, 2);
        assert_eq!(result.wins_b, 1);
        assert_eq!(result.ties, 1);
        assert_eq!(result.total, 4);
        assert_eq!(result.win_rate_a(), 0.5);
    }

    #[test]
    fn empty_verdicts() {
        let result = aggregate_verdicts("a", "b", &[]);
        assert_eq!(result.total, 0);
        assert_eq!(result.win_rate_a(), 0.0);
    }

    #[test]
    fn verdict_serde() {
        for v in [Verdict::WinA, Verdict::WinB, Verdict::Tie] {
            let json = serde_json::to_string(&v).unwrap();
            let back: Verdict = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }
}
