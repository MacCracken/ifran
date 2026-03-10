#!/usr/bin/env bash
set -euo pipefail
cargo test --workspace
echo "All tests passed."
