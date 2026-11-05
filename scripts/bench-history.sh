#!/usr/bin/env bash
set -euo pipefail

# bench-history.sh — Run the cyrius bench harness and append results to CSV history.
#
# Usage:
#   ./scripts/bench-history.sh [label]
#
# The optional label tags the run (e.g. "baseline", "post-audit", "2.4.0").
# Results are appended to benchmarks/history.csv, converted to nanoseconds.
# Rows before 2.3.3 are the Rust line's criterion benchmarks, kept as history.

cd "$(dirname "$0")/.."

LABEL="${1:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CSV="benchmarks/history.csv"

mkdir -p benchmarks

# Write header if the CSV doesn't exist yet
if [ ! -f "$CSV" ]; then
    echo "timestamp,label,benchmark,time_ns,time_unit" > "$CSV"
fi

echo "=== Running benchmarks (label: $LABEL) ==="

OUTPUT=$(cyrius bench tests/ifran.bcyr 2>&1)

echo "$OUTPUT"

# cyrius bench prints one line per benchmark:
#   <name>: <value><ns|us|ms|s> avg (min=... max=...) [<n> iters]
ROWS=0
while IFS= read -r line; do
    NAME=$(echo "$line" | sed -E 's/^[[:space:]]*([A-Za-z0-9_]+): .*/\1/')
    VAL=$(echo "$line" | sed -E 's/^[^:]*: ([0-9.]+)(ns|us|ms|s) avg.*/\1/')
    UNIT=$(echo "$line" | sed -E 's/^[^:]*: [0-9.]+(ns|us|ms|s) avg.*/\1/')
    NS=$(awk -v v="$VAL" -v u="$UNIT" 'BEGIN {
        m = 1; if (u == "us") m = 1000; else if (u == "ms") m = 1000000; else if (u == "s") m = 1000000000;
        printf "%.0f", v * m }')
    echo "${TIMESTAMP},${LABEL},${NAME},${NS},ns" >> "$CSV"
    ROWS=$((ROWS + 1))
done < <(echo "$OUTPUT" | grep -E '^[[:space:]]*[A-Za-z0-9_]+: [0-9.]+(ns|us|ms|s) avg')

if [ "$ROWS" -eq 0 ]; then
    echo "ERROR: no benchmark lines found in the cyrius bench output"
    exit 1
fi

echo ""
echo "=== $ROWS result(s) appended to $CSV ==="
tail -5 "$CSV"
