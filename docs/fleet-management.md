# Fleet Management

Fleet management lets you run multiple Synapse nodes as a coordinated cluster. Each node registers itself, sends periodic heartbeats, and is monitored for liveness. An aggregated view of GPU resources across the fleet is available through the REST API.

## Configuration

Add a `[fleet]` section to your Synapse config file:

```toml
[fleet]
enabled = true
suspect_timeout_secs = 30
offline_timeout_secs = 90
health_check_interval_secs = 15
```

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable fleet management and the health check loop. |
| `suspect_timeout_secs` | `30` | Seconds without a heartbeat before a node is marked **Suspect**. |
| `offline_timeout_secs` | `90` | Seconds without a heartbeat before a node is marked **Offline**. |
| `health_check_interval_secs` | `15` | How often (in seconds) the health check loop evaluates node states. |

## Node Health States

Every registered node is in one of three states:

```
Online ──(no heartbeat for suspect_timeout)──> Suspect ──(no heartbeat for offline_timeout)──> Offline
  ^                                                                                               |
  └─────────────────────── heartbeat received ────────────────────────────────────────────────────┘
```

- **Online** -- The node is sending heartbeats on time.
- **Suspect** -- No heartbeat received for `suspect_timeout_secs`. The node may be experiencing issues.
- **Offline** -- No heartbeat received for `offline_timeout_secs`. The node is considered unreachable.

A heartbeat from any state resets the node back to **Online**.

## Self-Registration on Startup

When `fleet.enabled` is `true`, each Synapse instance registers itself at startup. The instance ID is determined by:

1. The `SYNAPSE_INSTANCE_ID` environment variable, if set.
2. Otherwise, the server bind address with non-alphanumeric characters replaced by hyphens.

During self-registration the node reports its detected GPU count and total GPU memory. After registering, the instance starts the background health check loop at the configured interval.

## Health Check Loop

A background Tokio task runs every `health_check_interval_secs` seconds. On each tick it iterates all registered nodes and transitions any node whose `last_heartbeat` exceeds the configured thresholds. The loop is cancelled automatically when the `FleetManager` is dropped or explicitly stopped.

## REST API

All endpoints are under the `/fleet` prefix.

### Register a Node

```
POST /fleet/nodes
```

**Request body:**

```json
{
  "id": "gpu-node-1",
  "endpoint": "http://gpu-node-1:8420",
  "gpu_count": 2,
  "total_gpu_memory_mb": 48000
}
```

Registers a new node or re-registers an existing one (idempotent). Returns the full node object on success.

**Validation rules:**
- `id` -- 1-128 characters, alphanumeric and hyphens only.
- `endpoint` -- must start with `http://` or `https://`.
- `gpu_count` -- at most 64.
- `total_gpu_memory_mb` -- at most 10,000,000 (10 TB).

Returns `400 Bad Request` on validation failure.

### List Nodes

```
GET /fleet/nodes
```

Returns all registered nodes with their current health state and GPU info:

```json
{
  "nodes": [
    {
      "id": "gpu-node-1",
      "endpoint": "http://gpu-node-1:8420",
      "health": "online",
      "gpu_info": {
        "gpu_count": 2,
        "total_gpu_memory_mb": 48000,
        "gpu_utilization_pct": 75.0,
        "gpu_memory_used_mb": 20000,
        "gpu_temperature_c": 68.0
      },
      "last_heartbeat": "2026-03-19T12:00:00Z",
      "registered_at": "2026-03-19T11:00:00Z"
    }
  ]
}
```

### Send Heartbeat

```
POST /fleet/nodes/{id}/heartbeat
```

**Request body (all fields optional):**

```json
{
  "gpu_utilization_pct": 75.0,
  "gpu_memory_used_mb": 20000,
  "gpu_temperature_c": 68.0
}
```

Resets the node to **Online** and updates its GPU telemetry. An empty body `{}` is valid -- it refreshes the heartbeat without telemetry.

**Responses:**
- `204 No Content` on success.
- `404 Not Found` if the node ID is not registered.
- `400 Bad Request` if telemetry values are out of range (`gpu_utilization_pct` must be 0-100, `gpu_temperature_c` must be -50 to 250).

### Remove a Node

```
DELETE /fleet/nodes/{id}
```

Removes the node from the fleet.

**Responses:**
- `204 No Content` on success.
- `404 Not Found` if the node ID does not exist.

### Fleet Statistics

```
GET /fleet/stats
```

Returns aggregate statistics across all registered nodes:

```json
{
  "total_nodes": 3,
  "online": 2,
  "suspect": 1,
  "offline": 0,
  "total_gpus": 6,
  "total_gpu_memory_mb": 144000
}
```
