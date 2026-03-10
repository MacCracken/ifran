#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Synapse development environment..."

# Install protobuf compiler
if ! command -v protoc &>/dev/null; then
    echo "Installing protobuf compiler..."
    if command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm protobuf
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y protobuf-compiler
    fi
fi

# Install cargo-watch for development
if ! command -v cargo-watch &>/dev/null; then
    cargo install cargo-watch
fi

echo "Setup complete."
