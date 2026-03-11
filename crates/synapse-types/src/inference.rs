use serde::{Deserialize, Serialize};

/// Parameters for an inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub system_prompt: Option<String>,
}

/// Result of an inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    MaxTokens,
    StopSequence,
}

/// A chunk of streaming inference output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub text: String,
    pub done: bool,
    pub usage: Option<TokenUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_request_serde_roundtrip() {
        let req = InferenceRequest {
            prompt: "Hello, world!".into(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            stop_sequences: Some(vec!["</s>".into()]),
            system_prompt: Some("You are helpful.".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, "Hello, world!");
        assert_eq!(back.max_tokens, Some(100));
        assert_eq!(back.temperature, Some(0.7));
        assert_eq!(back.stop_sequences.unwrap().len(), 1);
    }

    #[test]
    fn inference_request_minimal() {
        let json = r#"{"prompt":"hi"}"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "hi");
        assert!(req.max_tokens.is_none());
        assert!(req.system_prompt.is_none());
    }

    #[test]
    fn inference_response_serde() {
        let resp = InferenceResponse {
            text: "Hello!".into(),
            usage: TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 10,
                total_tokens: 15,
            },
            finish_reason: FinishReason::Stop,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "Hello!");
        assert_eq!(back.usage.total_tokens, 15);
    }

    #[test]
    fn token_usage_serde() {
        let usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let back: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt_tokens, 0);
    }

    #[test]
    fn finish_reason_serde_values() {
        assert_eq!(serde_json::to_string(&FinishReason::Stop).unwrap(), "\"stop\"");
        assert_eq!(
            serde_json::to_string(&FinishReason::MaxTokens).unwrap(),
            "\"max_tokens\""
        );
        assert_eq!(
            serde_json::to_string(&FinishReason::StopSequence).unwrap(),
            "\"stop_sequence\""
        );
    }

    #[test]
    fn stream_chunk_serde() {
        let chunk = StreamChunk {
            text: "tok".into(),
            done: false,
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "tok");
        assert!(!back.done);
        assert!(back.usage.is_none());
    }

    #[test]
    fn stream_chunk_with_usage() {
        let chunk = StreamChunk {
            text: "".into(),
            done: true,
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: StreamChunk = serde_json::from_str(&json).unwrap();
        assert!(back.done);
        assert_eq!(back.usage.unwrap().total_tokens, 30);
    }
}
