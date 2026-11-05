#!/usr/bin/env bash
# Resolve deps (git first: the pinned tags, as CI does) and build ifran into
# build/ifran — the same two steps as .github/workflows/ci.yml.
set -euo pipefail
cd "$(dirname "$0")/.."
cyrius deps
mkdir -p build
cyrius build src/main.cyr build/ifran
echo "Build complete: build/ifran"
