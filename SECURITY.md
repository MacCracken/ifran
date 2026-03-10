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

Synapse supports optional Bearer token authentication via the `SYNAPSE_API_KEY` environment variable. When deployed in production:

- Always set `SYNAPSE_API_KEY` to a strong, random token
- Use TLS termination (reverse proxy) in front of the API server
- Bind to `127.0.0.1` instead of `0.0.0.0` if only local access is needed

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
