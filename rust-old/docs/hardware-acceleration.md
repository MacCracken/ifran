# Hardware Acceleration

Ifran auto-detects compute accelerators at startup and routes inference to the best available backend. This guide covers supported hardware, detection mechanics, configuration, and verification.

## Supported Accelerators

| Accelerator | `AcceleratorKind` | Detection Method | Detection Tool / Path |
|---|---|---|---|
| NVIDIA CUDA | `Cuda` | CLI query | `nvidia-smi --query-gpu=...` |
| AMD ROCm | `Rocm` | sysfs scan | `/sys/class/drm/card*/device/driver` (looks for `amdgpu`) |
| Apple Metal | `Metal` | CLI query | `system_profiler SPDisplaysDataType -json` |
| Vulkan | `Vulkan` | CLI query | `vulkaninfo --summary` |
| Google TPU | `Tpu` | Device nodes | `/dev/accel*` + sysfs vendor check (`0x1ae0`) |
| Intel Gaudi | `Gaudi` | CLI query | `hl-smi --query-aip=...` |
| AWS Inferentia/Trainium | `Inferentia` | CLI query | `neuron-ls --json-output` |
| Intel oneAPI (Arc/Max) | `OneApi` | CLI query | `xpu-smi discovery --dump 1,2,18,19` |
| Qualcomm Cloud AI 100 | `QualcommAi` | Device nodes | `/dev/qaic*` + sysfs |
| AMD XDNA (Ryzen AI) | `AmdXdna` | sysfs scan | `/sys/class/accel/*/device/driver` (looks for `amdxdna`) |

When multiple accelerators are present, `best_accelerator()` picks the highest-priority one:

```
Cuda > Tpu > Gaudi > Rocm > Inferentia > OneApi > Metal > Vulkan > QualcommAi > AmdXdna
```

If no accelerator is found, Ifran runs in CPU-only mode.

## How Detection Works

At startup, `ifran_core::hardware::detect::detect()` probes the system and returns a `SystemHardware` snapshot containing CPU info and a list of `GpuDevice` entries. Each device includes name, memory (total/free), accelerator kind, and optional compute capability.

There are two detection paths:

1. **Built-in detection** (default) -- each accelerator family has its own function (`detect_nvidia()`, `detect_rocm()`, etc.) that shells out to vendor tools or reads sysfs. If a tool is missing or fails, that backend is silently skipped.

2. **`ai-hwaccel` delegation** (opt-in) -- when the `ai-hwaccel` feature is enabled, detection delegates to `ai_hwaccel::AcceleratorRegistry::detect()` and converts results back to Ifran types. The built-in detection functions are compiled out.

## The `ai-hwaccel` Feature Flag

The `ai-hwaccel` feature on `ifran-core` replaces built-in detection with the external `ai-hwaccel` crate. It provides:

- **Broader discovery** -- 13 backend families including Apple ANE and Intel NPU (mapped to `Vulkan` in Ifran since dedicated kinds don't exist yet).
- **Richer metadata** -- driver versions, generation info, ranked device selection.
- **Advanced APIs** -- quantization suggestions, sharding plans, and accelerator profiles via `detect_registry()`.
- **Re-export** -- `ai_hwaccel` is re-exported from the hardware module for callers that want the full API.

Enable it in your workspace build:

```toml
ifran-core = { path = "crates/ifran-core", features = ["ai-hwaccel"] }
```

When disabled, zero dead code is emitted -- the two paths are gated with `cfg(feature = "ai-hwaccel")` / `cfg(not(feature = "ai-hwaccel"))`.

## Configuration

### `[hardware]` section in `ifran.toml`

```toml
[hardware]
# VRAM (MB) reserved for OS/driver use when calculating model fit.
# Ifran subtracts this from reported free memory before loading models.
gpu_memory_reserve_mb = 512

# How often (seconds) to poll hardware telemetry. 0 = disabled.
telemetry_interval_secs = 10
```

| Key | Type | Default | Description |
|---|---|---|---|
| `gpu_memory_reserve_mb` | `u64` | `512` | VRAM headroom in MB kept free for system use |
| `telemetry_interval_secs` | `u64` | `10` | Polling interval for hardware metrics; `0` disables polling |

## Backend Feature Flags

`ifran-backends` uses Cargo feature flags to compile in only the backends you need. All accelerator-specific backends are included in the default feature set.

```toml
[features]
default = [
  "llamacpp", "candle-backend", "gguf", "ollama", "vllm",
  "onnx", "tensorrt", "tpu", "gaudi", "inferentia",
  "oneapi", "qualcomm", "xdna"
]
```

Additional opt-in features (not in default): `wasm`, `metal`, `vulkan`.

To build a minimal binary with only CUDA-oriented backends:

```sh
cargo build -p ifran-backends --no-default-features --features llamacpp,candle-backend,tensorrt
```

## Verifying Detection

### CLI

```sh
ifran status
```

The status command prints a `SystemHardware` summary:

```
CPU: AMD EPYC 7763 (64 cores, 128 threads, 524288 MB RAM)
GPU 0: NVIDIA A100-SXM4-80GB [Cuda] — 81920 MB total, 79462 MB free, compute 8.0
```

### REST API

```
GET /system/status
```

Returns system hardware info including detected accelerators, memory, and the active backend configuration.
