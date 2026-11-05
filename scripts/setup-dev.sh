#!/usr/bin/env bash
# Check the cyrius toolchain against the cyrius.cyml pin, then confirm the
# vendored lib/ + cyrius.lock match the pinned dependency tags.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Setting up the ifran development environment..."

PIN=$(grep '^cyrius = ' cyrius.cyml | head -1 | sed 's/cyrius = "\(.*\)"/\1/')
INSTALL="curl -sSf https://raw.githubusercontent.com/MacCracken/cyrius/main/scripts/install.sh | CYRIUS_VERSION=$PIN sh"

if ! command -v cyrius &>/dev/null; then
    echo "cyrius is not on PATH. Install the pinned toolchain ($PIN):"
    echo "  $INSTALL"
    exit 1
fi

HAVE=$(cyrius version | head -1 | awk '{print $2}')
if [[ "$HAVE" != "$PIN" ]]; then
    echo "cyrius $HAVE is active, but cyrius.cyml pins $PIN. Install the pin:"
    echo "  $INSTALL"
    exit 1
fi
echo "cyrius $HAVE matches the pin."

cyrius deps --locked

echo "Setup complete. Next: ./scripts/build.sh and ./scripts/test.sh"
