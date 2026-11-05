#!/usr/bin/env bash
# Set the project version everywhere it is recorded.
#
# Usage:
#   ./scripts/version-set.sh 2.4.0
#   ./scripts/version-set.sh 2.4.0-rc1
#
# Updates:
#   VERSION       the source of truth (cyrius.cyml reads it via ${file:VERSION})
#   src/main.cyr  the IFRAN_VERSION() literal that `ifran --version` prints
#
# The CHANGELOG entry and docs/development/state.md are written by hand.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>"
    echo "  e.g., $0 2.4.0"
    echo "  e.g., $0 2.4.0-rc1"
    exit 1
fi

VERSION="$1"

# Validate semver format (MAJOR.MINOR.PATCH with optional pre-release)
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "ERROR: Invalid version format: '${VERSION}'"
    echo "Expected: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-pre (e.g., 2.4.0 or 2.4.0-rc1)"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$REPO_ROOT/src/main.cyr"
LITERAL_RE='^fn IFRAN_VERSION\(\): i64 \{ return "[^"]*"; \}$'

# Refuse before writing anything if the literal is not where we expect it.
if [[ $(grep -cE "$LITERAL_RE" "$MAIN") -ne 1 ]]; then
    echo "ERROR: expected exactly one IFRAN_VERSION() literal line in src/main.cyr"
    exit 1
fi

echo "$VERSION" > "$REPO_ROOT/VERSION"
sed -i -E "s/^(fn IFRAN_VERSION\(\): i64 \{ return \")[^\"]*(\"; \})$/\1${VERSION}\2/" "$MAIN"

# Verify consistency
FILE_VERSION=$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")
SRC_VERSION=$(grep -E "$LITERAL_RE" "$MAIN" | sed -E 's/.*return "([^"]*)".*/\1/')

if [[ "$FILE_VERSION" != "$VERSION" || "$SRC_VERSION" != "$VERSION" ]]; then
    echo "ERROR: Version mismatch after update"
    echo "  VERSION file: $FILE_VERSION"
    echo "  src/main.cyr: $SRC_VERSION"
    exit 1
fi

echo "Version set to $VERSION"
echo "  VERSION:      $FILE_VERSION"
echo "  src/main.cyr: $SRC_VERSION"
