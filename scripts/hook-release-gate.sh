#!/usr/bin/env bash
# PreToolUse hook: block git commit unless quality + tests pass
set -euo pipefail

CMD=$(jq -r '.tool_input.command' < /dev/stdin)

# Only gate on git commit commands
if ! echo "$CMD" | grep -qE '^git\s+commit\b'; then
  exit 0
fi

cd /home/macro/Repos/ifran

cargo fmt --all -- --check || { echo '{"decision":"block","reason":"cargo fmt check failed"}'; exit 0; }
cargo clippy --workspace -- -D warnings || { echo '{"decision":"block","reason":"cargo clippy failed"}'; exit 0; }
cargo check --workspace || { echo '{"decision":"block","reason":"cargo check failed"}'; exit 0; }
cargo test --workspace || { echo '{"decision":"block","reason":"cargo test failed"}'; exit 0; }
