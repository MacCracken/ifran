# ADR-009: API Security Hardening

**Status**: Accepted

## Context

Ifran exposes a REST API for model inference, training, evaluation, RLHF annotation, RAG pipelines, and marketplace operations. Before this work, the API had:

- Optional Bearer token auth (disabled by default)
- Permissive CORS (`Access-Control-Allow-Origin: *`)
- No rate limiting
- No request body size limits
- No prompt length validation
- No input validation on model names or filenames

This left the API vulnerable to prompt injection, denial-of-service via resource exhaustion, path traversal, and cross-origin abuse.

## Decision

Implement six defense-in-depth security layers, all configurable via a new `[security]` TOML section with backward-compatible serde defaults:

1. **Rate limiting** — Global per-server rate limiter using the `governor` crate. Configurable requests-per-second and burst size. Returns HTTP 429 when exceeded. Applied as the outermost middleware layer so it runs before any request processing.

2. **Request body size limits** — `tower-http::limit::RequestBodyLimitLayer` caps request bodies at `max_body_size_bytes` (default 10 MB). Returns HTTP 413 for oversized payloads.

3. **Prompt length validation** — Handler-level validation rejects prompts exceeding `max_prompt_length` characters (default 100,000). Applied to `/inference`, `/inference/stream`, and `/v1/chat/completions`. Each message in OpenAI-compatible requests is validated individually.

4. **CORS lockdown** — `cors_allowed_origins` config replaces `CorsLayer::permissive()`. Empty list preserves backward-compatible permissive behavior. Specific origins restrict to listed domains.

5. **Input validation** — `validate_model_name()` rejects path traversal (`..`), control characters, and overly long names. `validate_filename()` rejects path separators, hidden files, and traversal. Applied at handler entry points for inference, RAG ingest, and marketplace operations.

6. **Auth-required mode** — When `auth_required = true`, the server refuses to start without `IFRAN_API_KEY` set. Fail-fast at startup rather than silently running unauthenticated.

### Middleware stack order (outermost to innermost)

```
Request → Rate Limit → Telemetry → Body Limit → CORS → Auth → Handler
```

## Consequences

- Existing deployments without a `[security]` section continue working unchanged (serde defaults)
- Production deployments should set `auth_required = true` and configure `cors_allowed_origins`
- Rate limiting is global (not per-IP) for simplicity; per-IP keying can be added later
- Prompt length validation is character-based, not token-based — a rough but effective guard
- The `governor` crate adds one new dependency to the workspace
