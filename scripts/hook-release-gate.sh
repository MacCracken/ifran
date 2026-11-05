#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher "Bash"): block `git commit` unless the
# first-party sources are cyrfmt-clean and lint-clean, lib/ + cyrius.lock match
# the pinned tags, ifran builds, and the suite passes. Not wired by default —
# add it under hooks.PreToolUse in .claude/settings.json to use it.
set -euo pipefail

# The Bash command from the hook's JSON on stdin (jq if installed, else python3)
INPUT=$(cat)
if command -v jq &>/dev/null; then
    CMD=$(jq -r '.tool_input.command // empty' <<< "$INPUT")
else
    CMD=$(python3 -I -c 'import json, sys
d = json.load(sys.stdin)
print((d.get("tool_input") or {}).get("command") or "")' <<< "$INPUT")
fi

# Only gate on git commit commands
if ! echo "$CMD" | grep -qE '^git\s+commit\b'; then
    exit 0
fi

cd "$(dirname "$0")/.."

block() {
    printf '{"decision":"block","reason":"%s"}\n' "$1"
    exit 0
}

for f in src/*.cyr tests/*.tcyr tests/*.bcyr tests/*.fcyr; do
    cyrius fmt --check "$f" >/dev/null 2>&1 || block "cyrius fmt --check failed: $f"
    cyrius lint --exit-with-count "$f" >/dev/null 2>&1 || block "cyrius lint failed: $f"
done
cyrius deps --locked >/dev/null 2>&1 || block "cyrius deps --locked failed: lib/ or cyrius.lock differ from the pinned tags"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cyrius build src/main.cyr "$OUT/ifran" >/dev/null 2>&1 || block "cyrius build failed"
cyrius test tests/ifran.tcyr >/dev/null 2>&1 || block "tests/ifran.tcyr failed"
