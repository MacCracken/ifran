# Multi-Tenancy

Synapse supports two authentication modes: **single-tenant** (the default) and **multi-tenant**. This guide covers how to enable multi-tenancy, manage tenants, and enforce resource budgets.

## Single-Tenant vs Multi-Tenant

| | Single-tenant (default) | Multi-tenant |
|---|---|---|
| Auth mechanism | `SYNAPSE_API_KEY` env var | Per-tenant API keys in SQLite |
| Tenant identity | Implicit default tenant | Unique `TenantId` per key |
| Admin API | Disabled | Enabled (`/admin/tenants`) |
| Resource isolation | N/A | Jobs, models, lineage scoped per tenant |
| No key configured | Open access (no auth) | Not possible -- every request needs a key |

In single-tenant mode, all requests share a single implicit tenant. If `SYNAPSE_API_KEY` is not set, auth is disabled entirely (open access). Multi-tenant mode requires every request to carry a valid per-tenant Bearer token.

## Enabling Multi-Tenancy

Add the following to `synapse.toml`:

```toml
[security]
multi_tenant = true
```

Then set the admin key as an environment variable:

```bash
export SYNAPSE_ADMIN_KEY="your-secret-admin-key"
```

The admin key protects the `/admin/tenants` endpoints. Without it, all admin requests return `403 Forbidden`.

> **Note:** In multi-tenant mode the `auth_required` flag is ignored -- auth is always enforced through tenant API keys.

## Tenant Management API

All admin endpoints require a `Bearer` token matching `SYNAPSE_ADMIN_KEY`.

### Create a Tenant

```
POST /admin/tenants
Authorization: Bearer <SYNAPSE_ADMIN_KEY>
Content-Type: application/json

{ "name": "Acme Corp" }
```

Response (`201 Created`):

```json
{
  "tenant": {
    "id": "a1b2c3d4-...",
    "name": "Acme Corp",
    "enabled": true,
    "created_at": "2026-03-19T12:00:00Z"
  },
  "api_key": "syn_8f3a..."
}
```

### List Tenants

```
GET /admin/tenants
Authorization: Bearer <SYNAPSE_ADMIN_KEY>
```

Returns an array of tenant objects (without API keys). Results are ordered by creation time, newest first.

### Disable a Tenant

```
DELETE /admin/tenants/:id
Authorization: Bearer <SYNAPSE_ADMIN_KEY>
```

Returns `204 No Content` on success, `404 Not Found` if the tenant ID does not exist.

## API Key Lifecycle

When a tenant is created, Synapse generates a random API key with the format `syn_<32 hex chars>` (36 characters total). The raw key is returned **exactly once** in the creation response. Store it immediately -- it cannot be retrieved later.

For storage and lookup, the key is hashed with **BLAKE3** and stored in SQLite. On each request the auth middleware hashes the incoming Bearer token and matches it against the stored hashes. This means Synapse never holds plaintext keys at rest.

Tenants authenticate by passing their key as a Bearer token:

```
Authorization: Bearer syn_8f3a...
```

## Resource Isolation

In multi-tenant mode, every authenticated request carries a `TenantId` injected by the auth middleware. Downstream subsystems -- jobs, models, lineage tracking -- use this ID to scope all operations. A tenant can only see and act on its own resources.

## Tenant Disabling Behavior

When a tenant is disabled via `DELETE /admin/tenants/:id`:

1. The tenant record is soft-deleted (marked `enabled = false` in SQLite).
2. All **in-flight training jobs** for that tenant are cancelled asynchronously.
3. Any subsequent API request using the tenant's key receives `403 Forbidden`.

The tenant record is not deleted -- it can be re-enabled programmatically through the store. Disabling is idempotent; disabling an already-disabled tenant is a no-op.

## GPU Budget Enforcement

Synapse can enforce per-tenant GPU time budgets through integration with a Hoosh accounting service.

```toml
[budget]
enabled = true
hoosh_endpoint = "http://10.0.0.1:9401"
max_gpu_hours_per_day = 48.0
```

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Toggle budget enforcement on/off |
| `hoosh_endpoint` | `http://127.0.0.1:9401` | Hoosh accounting service URL |
| `max_gpu_hours_per_day` | `0.0` (unlimited) | Per-tenant daily GPU-hour cap |

When enabled, Synapse queries the Hoosh endpoint before scheduling GPU workloads. If a tenant has exhausted its daily allocation, new jobs are rejected. Setting `max_gpu_hours_per_day` to `0` disables the cap (unlimited).

The value must be a non-negative finite number; negative, `NaN`, or infinite values are rejected at config validation time.
