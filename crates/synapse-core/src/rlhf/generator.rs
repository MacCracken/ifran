use synapse_types::rlhf::AnnotationPair;
use uuid::Uuid;

/// Create a single annotation pair from a prompt and two responses.
pub fn generate_pair(
    session_id: Uuid,
    prompt: String,
    response_a: String,
    response_b: String,
) -> AnnotationPair {
    AnnotationPair {
        id: Uuid::new_v4(),
        session_id,
        prompt,
        response_a,
        response_b,
        preference: None,
        annotated_at: None,
    }
}

/// Generate pairs from prompts using an inference function.
/// The `infer_fn` is called twice per prompt (for response A and B).
pub fn generate_pairs_from_prompts(
    session_id: Uuid,
    prompts: &[String],
    infer_fn: impl Fn(&str) -> String,
) -> Vec<AnnotationPair> {
    prompts
        .iter()
        .map(|prompt| {
            let response_a = infer_fn(prompt);
            let response_b = infer_fn(prompt);
            generate_pair(session_id, prompt.clone(), response_a, response_b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_pair_creates_unannotated() {
        let session_id = Uuid::new_v4();
        let pair = generate_pair(session_id, "hello".into(), "a".into(), "b".into());
        assert_eq!(pair.session_id, session_id);
        assert_eq!(pair.prompt, "hello");
        assert_eq!(pair.response_a, "a");
        assert_eq!(pair.response_b, "b");
        assert!(pair.preference.is_none());
        assert!(pair.annotated_at.is_none());
    }

    #[test]
    fn generate_pairs_from_prompts_uses_infer_fn() {
        let session_id = Uuid::new_v4();
        let prompts = vec!["q1".to_string(), "q2".to_string()];
        let call_count = std::cell::Cell::new(0u32);
        let pairs = generate_pairs_from_prompts(session_id, &prompts, |prompt| {
            let c = call_count.get();
            call_count.set(c + 1);
            format!("response-{c}-{prompt}")
        });
        assert_eq!(pairs.len(), 2);
        // 2 prompts * 2 calls each = 4 calls
        assert_eq!(call_count.get(), 4);
        assert_eq!(pairs[0].prompt, "q1");
        assert_eq!(pairs[1].prompt, "q2");
    }

    #[test]
    fn generate_pair_unique_ids() {
        let session_id = Uuid::new_v4();
        let p1 = generate_pair(session_id, "a".into(), "b".into(), "c".into());
        let p2 = generate_pair(session_id, "a".into(), "b".into(), "c".into());
        assert_ne!(p1.id, p2.id);
    }
}
