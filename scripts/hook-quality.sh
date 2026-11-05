#!/usr/bin/env bash
# Claude Code PostToolUse hook (matcher "Write|Edit"): format-check a
# first-party Cyrius file after it changes, and lint it when it is under src/.
# Nothing is rewritten; findings go
# back to Claude on stderr with exit 2. Not wired by default — add it under
# hooks.PostToolUse in .claude/settings.json to use it.
set -euo pipefail

# The edited path from the hook's JSON on stdin (jq if installed, else python3)
INPUT=$(cat)
if command -v jq &>/dev/null; then
    FILE=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' <<< "$INPUT")
else
    FILE=$(python3 -I -c 'import json, sys
d = json.load(sys.stdin)
print((d.get("tool_input") or {}).get("file_path") or (d.get("tool_response") or {}).get("filePath") or "")' <<< "$INPUT")
fi
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Only Cyrius sources inside this repo, and never the vendored lib/
case "$FILE" in
    *.cyr|*.tcyr|*.bcyr|*.fcyr) ;;
    *) exit 0 ;;
esac
case "$FILE" in
    "$REPO_ROOT"/lib/*) exit 0 ;;
    "$REPO_ROOT"/*) ;;
    *) exit 0 ;;
esac

cd "$REPO_ROOT"
if ! out=$(cyrius fmt --check "$FILE" 2>&1); then
    echo "$out" >&2
    exit 2
fi
# 0 lint warnings is the src/ rule (docs/development/README.md)
case "$FILE" in
    "$REPO_ROOT"/src/*)
        if ! out=$(cyrius lint --exit-with-count "$FILE" 2>&1); then
            echo "$out" >&2
            exit 2
        fi
        ;;
esac
