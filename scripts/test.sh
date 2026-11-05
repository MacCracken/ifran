#!/usr/bin/env bash
# The full local gate: lib/ + cyrius.lock match the pinned tags, then the
# suite, the bench harness and the fuzz harness. Everything runs against /tmp.
set -euo pipefail
cd "$(dirname "$0")/.."
cyrius deps --locked
cyrius test tests/ifran.tcyr
cyrius bench tests/ifran.bcyr
cyrius run tests/ifran.fcyr
echo "All checks passed."
