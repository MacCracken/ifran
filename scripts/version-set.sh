#!/usr/bin/env bash
# Update the project version across all version sources.
#
# Usage:
#   ./scripts/version-set.sh 2026.3.18
#   ./scripts/version-set.sh        # defaults to CalVer today: YYYY.M.D
#
# Updates: VERSION, Cargo.toml (workspace), Cargo.lock

set -euo pipefail

# Default to today's date in CalVer format
if [[ $# -ge 1 ]]; then
    VERSION="$1"
else
    VERSION="$(date +%Y.%-m.%-d)"
fi

# Validate CalVer format
if [[ ! "$VERSION" =~ ^[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}(-[0-9]+)?$ ]]; then
    echo "ERROR: Invalid CalVer format: '${VERSION}'"
    echo "Expected: YYYY.M.D or YYYY.M.D-N (e.g., 2026.3.18 or 2026.3.18-1)"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Update VERSION file
echo "$VERSION" > "$REPO_ROOT/VERSION"

# Update workspace Cargo.toml
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" "$REPO_ROOT/Cargo.toml"

# Regenerate Cargo.lock
(cd "$REPO_ROOT" && cargo generate-lockfile 2>/dev/null)

# Verify consistency
CARGO_VERSION=$(grep '^version = ' "$REPO_ROOT/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')
FILE_VERSION=$(cat "$REPO_ROOT/VERSION" | tr -d '[:space:]')

if [[ "$CARGO_VERSION" != "$FILE_VERSION" ]]; then
    echo "ERROR: Version mismatch after update"
    echo "  VERSION file: $FILE_VERSION"
    echo "  Cargo.toml:   $CARGO_VERSION"
    exit 1
fi

echo "Version set to $VERSION"
echo "  VERSION:    $FILE_VERSION"
echo "  Cargo.toml: $CARGO_VERSION"
