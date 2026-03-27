#!/usr/bin/env bash
set -euo pipefail

# bench-history.sh — Run criterion benchmarks and append results to CSV history.
#
# Usage:
#   ./scripts/bench-history.sh [label]
#
# The optional label tags the run (e.g. "baseline", "post-audit", "v2026.3.26").
# Results are appended to benchmarks/history.csv.

LABEL="${1:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CSV="benchmarks/history.csv"

mkdir -p benchmarks

# Write header if the CSV doesn't exist yet
if [ ! -f "$CSV" ]; then
    echo "timestamp,label,benchmark,time_ns,time_unit" > "$CSV"
fi

echo "=== Running benchmarks (label: $LABEL) ==="

# Run criterion benchmarks, capture output
OUTPUT=$(cargo bench --bench core_benchmarks -- --output-format=bencher 2>&1)

echo "$OUTPUT"

# Parse criterion's bencher output format:
#   test <name> ... bench:   <time> ns/iter (+/- <variance>)
echo "$OUTPUT" | grep 'bench:' | while IFS= read -r line; do
    NAME=$(echo "$line" | sed 's/^test \(.*\) \.\.\. bench:.*/\1/' | xargs)
    TIME_NS=$(echo "$line" | sed 's/.*bench: *\([0-9,]*\) ns.*/\1/' | tr -d ',')
    echo "${TIMESTAMP},${LABEL},${NAME},${TIME_NS},ns" >> "$CSV"
done

echo ""
echo "=== Results appended to $CSV ==="
tail -20 "$CSV"
