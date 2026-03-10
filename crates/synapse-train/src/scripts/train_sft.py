#!/usr/bin/env python3
"""Supervised fine-tuning script for Synapse training jobs."""

import json
import sys
from pathlib import Path


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/config.json"
    config = json.loads(Path(config_path).read_text())
    print(f"Starting SFT training with config: {config}")
    # TODO: Implementation


if __name__ == "__main__":
    main()
