#!/usr/bin/env bash
set -euo pipefail
cargo build --release --workspace
echo "Build complete."
