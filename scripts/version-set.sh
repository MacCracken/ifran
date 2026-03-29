#!/usr/bin/env bash
# Update the project version across all version sources.
#
# Usage:
#   ./scripts/version-set.sh 1.2.0
#   ./scripts/version-set.sh 1.2.0-rc1
#
# Updates: VERSION, Cargo.toml (workspace), Cargo.lock

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <version>"
    echo "  e.g., $0 1.2.0"
    echo "  e.g., $0 1.2.0-rc1"
    exit 1
fi

VERSION="$1"

# Validate semver format (MAJOR.MINOR.PATCH with optional pre-release)
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "ERROR: Invalid version format: '${VERSION}'"
    echo "Expected: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-pre (e.g., 1.2.0 or 1.2.0-rc1)"
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
