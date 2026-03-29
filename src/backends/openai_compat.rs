//! Shared helpers for OpenAI-compatible backend APIs.
//!
//! Many inference backends (vLLM, TensorRT-LLM, TPU, Metal, Vulkan, Gaudi,
//! Inferentia, OneAPI, Qualcomm, XDNA, llama.cpp) expose an OpenAI-compatible
//! HTTP API. This module extracts the common message building, response
//! parsing, and SSE stream handling to avoid duplication across backends.

use crate::types::error::Result;
use crate::types::inference::{
    FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tracing::warn;

/// Build an OpenAI-compatible messages array from an [`InferenceRequest`].
///
/// Produces a `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`
/// array. The system message is only included when `req.system_prompt` is `Some`.
#[inline]
#[must_use]
pub(crate) fn build_openai_messages(req: &InferenceRequest) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    if let Some(ref system) = req.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": &req.prompt,
    }));
    messages
}

/// Parse an OpenAI-compatible chat completion JSON response into an
/// [`InferenceResponse`].
///
/// Handles the standard `choices[0].message.content`, `choices[0].finish_reason`,
/// and `usage` fields. Missing fields default to empty/zero values.
#[inline]
pub(crate) fn parse_openai_response(json: &serde_json::Value) -> Result<InferenceResponse> {
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let finish_reason = match json["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::MaxTokens,
        _ => FinishReason::Stop,
    };

    let usage = TokenUsage {
        prompt_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(InferenceResponse {
        text,
        usage,
        finish_reason,
    })
}

/// Stream an OpenAI-compatible SSE response, sending parsed chunks to `tx`.
///
/// Parses `data: {...}` lines from the SSE stream, extracting
/// `choices[0].delta.content` text. Sends a final `done: true` chunk when
/// `data: [DONE]` is received.
///
/// The buffer is capped at 1 MB to prevent unbounded memory growth from
/// misbehaving servers.
pub(crate) async fn stream_openai_sse(
    response: reqwest::Response,
    tx: mpsc::Sender<StreamChunk>,
) -> Result<()> {
    const MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        if tx.is_closed() {
            break;
        }
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                warn!("SSE stream error: {e}");
                break;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        if buffer.len() > MAX_BUFFER_SIZE {
            warn!("SSE stream buffer exceeded {MAX_BUFFER_SIZE} bytes, aborting");
            break;
        }

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    let _ = tx
                        .send(StreamChunk {
                            text: String::new(),
                            done: true,
                            usage: None,
                        })
                        .await;
                    return Ok(());
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let text = json["choices"][0]["delta"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();

                    if !text.is_empty() {
                        let _ = tx
                            .send(StreamChunk {
                                text,
                                done: false,
                                usage: None,
                            })
                            .await;
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_messages_user_only() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
    }

    #[test]
    fn build_messages_with_system() {
        let req = InferenceRequest {
            prompt: "Hi".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: Some("Be helpful.".into()),
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be helpful.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hi");
    }

    #[test]
    fn parse_response_stop() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 8);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_response_length() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 512, "total_tokens": 522}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::MaxTokens));
    }

    #[test]
    fn parse_response_missing_fields() {
        let json = serde_json::json!({});
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[test]
    fn parse_response_empty_choices_array() {
        let json = serde_json::json!({
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });
        let resp = parse_openai_response(&json).unwrap();
        // choices[0] is null -> falls back to defaults
        assert_eq!(resp.text, "");
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
        assert_eq!(resp.usage.total_tokens, 3);
    }

    #[test]
    fn parse_response_null_content() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": null}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
    }

    #[test]
    fn parse_response_null_finish_reason() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": null}],
            "usage": {}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "hi");
        // null finish_reason falls to _ arm -> Stop
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_response_unknown_finish_reason() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}],
            "usage": {}
        });
        let resp = parse_openai_response(&json).unwrap();
        // Unknown finish_reason maps to Stop (default)
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_response_missing_usage_entirely() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "hello");
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert_eq!(resp.usage.completion_tokens, 0);
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[test]
    fn parse_response_missing_message_key() {
        let json = serde_json::json!({
            "choices": [{"finish_reason": "stop"}],
            "usage": {}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
    }

    #[test]
    fn build_messages_with_stop_sequences() {
        // stop_sequences don't affect message building, but verify the
        // request structure is handled correctly
        let req = InferenceRequest {
            prompt: "Count to 3".into(),
            max_tokens: Some(100),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: None,
            stop_sequences: Some(vec!["\n".into(), "END".into()]),
            system_prompt: Some("You are a counter.".into()),
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"], "Count to 3");
    }

    #[test]
    fn build_messages_empty_prompt() {
        let req = InferenceRequest {
            prompt: String::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "");
    }

    #[test]
    fn build_messages_empty_system_prompt() {
        let req = InferenceRequest {
            prompt: "Hi".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: Some(String::new()),
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        // Even empty system prompt is included when Some
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "");
    }

    #[test]
    fn parse_response_multiple_choices_uses_first() {
        let json = serde_json::json!({
            "choices": [
                {"message": {"content": "first"}, "finish_reason": "stop"},
                {"message": {"content": "second"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "first");
    }

    #[test]
    fn parse_response_integer_content_returns_empty() {
        // content is a number, not a string — as_str() returns None
        let json = serde_json::json!({
            "choices": [{"message": {"content": 42}, "finish_reason": "stop"}],
            "usage": {}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
    }

    #[test]
    fn parse_response_partial_usage() {
        // Only prompt_tokens present
        let json = serde_json::json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.completion_tokens, 0);
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[test]
    fn sse_delta_content_parsing() {
        // Test the core parsing logic used inside stream_openai_sse
        let json_str = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let text = json["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap_or("");
        assert_eq!(text, "Hello");
    }

    #[test]
    fn sse_line_parsing_data_prefix() {
        let line = "data: {\"choices\":[{\"delta\":{\"content\":\"test\"}}]}";
        let data = line.strip_prefix("data: ");
        assert!(data.is_some());
        let json: serde_json::Value = serde_json::from_str(data.unwrap()).unwrap();
        let text = json["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap_or("");
        assert_eq!(text, "test");
    }

    #[test]
    fn sse_line_parsing_done_signal() {
        let line = "data: [DONE]";
        let data = line.strip_prefix("data: ").unwrap();
        assert_eq!(data, "[DONE]");
    }

    #[test]
    fn sse_line_parsing_non_data_line_ignored() {
        let line = "event: ping";
        let data = line.strip_prefix("data: ");
        assert!(data.is_none());
    }

    #[test]
    fn sse_line_parsing_empty_content() {
        let line = "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}";
        let data = line.strip_prefix("data: ").unwrap();
        let json: serde_json::Value = serde_json::from_str(data).unwrap();
        let text = json["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap_or("");
        assert_eq!(text, "");
    }

    #[test]
    fn sse_line_parsing_missing_delta() {
        let line = "data: {\"choices\":[{}]}";
        let data = line.strip_prefix("data: ").unwrap();
        let json: serde_json::Value = serde_json::from_str(data).unwrap();
        let text = json["choices"][0]["delta"]["content"]
            .as_str()
            .unwrap_or("");
        assert_eq!(text, "");
    }

    #[test]
    fn sse_line_parsing_invalid_json_skipped() {
        let line = "data: not-valid-json";
        let data = line.strip_prefix("data: ").unwrap();
        let result = serde_json::from_str::<serde_json::Value>(data);
        assert!(result.is_err());
    }

    #[test]
    fn build_messages_long_prompt() {
        let long_prompt = "x".repeat(10_000);
        let req = InferenceRequest {
            prompt: long_prompt.clone(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"].as_str().unwrap().len(), 10_000);
    }

    #[test]
    fn parse_response_large_token_counts() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100000, "completion_tokens": 50000, "total_tokens": 150000}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.usage.prompt_tokens, 100_000);
        assert_eq!(resp.usage.completion_tokens, 50_000);
        assert_eq!(resp.usage.total_tokens, 150_000);
    }
}
