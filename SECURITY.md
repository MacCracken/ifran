# Security Policy

## Supported Versions

| Version       | Supported |
| ------------- | --------- |
| Latest CalVer | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Synapse, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email **security@maccracken.dev** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You should receive an acknowledgment within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Security Considerations

### API Authentication

Synapse supports Bearer token authentication via the `SYNAPSE_API_KEY` environment variable. When deployed in production:

- Always set `SYNAPSE_API_KEY` to a strong, random token
- Set `auth_required = true` in `[security]` config — the server will refuse to start without an API key
- Use TLS termination (reverse proxy) in front of the API server
- Bind to `127.0.0.1` instead of `0.0.0.0` if only local access is needed

### Rate Limiting

The API enforces global rate limiting via the `governor` crate:

- Default: 60 requests/second with burst of 120
- Returns HTTP 429 Too Many Requests when exceeded
- Configure via `[security]` in `synapse.toml`:
  ```toml
  [security]
  rate_limit_per_second = 30
  rate_limit_burst = 60
  ```

### Request Size Limits

- Request bodies are capped at `max_body_size_bytes` (default 10 MB)
- Oversized payloads return HTTP 413 Payload Too Large
- Prompts are capped at `max_prompt_length` characters (default 100,000)

### CORS

CORS is configurable via `cors_allowed_origins`:

- Empty list (default) = permissive (backward compatible for development)
- `["*"]` = permissive (explicit wildcard)
- `["https://your-domain.com"]` = restrictive (recommended for production)

```toml
[security]
cors_allowed_origins = ["https://app.example.com"]
```

### Input Validation

- Model names are validated: alphanumeric, hyphens, underscores, slashes, dots only. Path traversal (`..`) is rejected.
- Filenames for RAG ingestion reject path separators, hidden files, and traversal sequences.
- All SQL queries use parameterized statements (no SQL injection).
- Subprocess commands use `Command::new().args()` — no shell expansion (no command injection).

### Systemd Hardening

The provided `synapse.service` unit includes security hardening:

- `ProtectSystem=strict` — read-only filesystem except allowed paths
- `ProtectHome=true` — no access to home directories
- `PrivateTmp=true` — isolated /tmp
- `NoNewPrivileges=true` — no privilege escalation
- Dedicated `synapse` system user with minimal permissions

### Model Storage

- Models are stored in `~/.synapse/models/` (user) or `/var/lib/synapse/models/` (system)
- Downloaded files are verified via SHA-256 checksums when available
- The model catalog uses SQLite with parameterized queries (no SQL injection)

### Training

- Docker-based training runs in isolated containers
- GPU passthrough is limited to specific device nodes
- Training configurations are validated before execution
