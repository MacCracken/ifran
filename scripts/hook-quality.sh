#!/usr/bin/env bash
# PostToolUse hook: run code quality checks after Write|Edit on .rs files
set -euo pipefail

FILE=$(jq -r '.tool_input.file_path // .tool_response.filePath' < /dev/stdin)

# Only run on Rust files
if [[ "$FILE" != *.rs ]]; then
  exit 0
fi

cd /home/macro/Repos/ifran
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo check --workspace
