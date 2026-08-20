#!/usr/bin/env bash
# Compatibility entry point for the current two-round benchmark harness.
#
# Examples:
#   benchmark/combos.sh --round coder
#   benchmark/combos.sh --round features
#   benchmark/combos.sh --round all --output .build/benchmark-rounds/my-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/coder_cli_benchmark.py" "$@"
