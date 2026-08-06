#!/usr/bin/env bash
# Launch NVMAIServer for 8-bit model with prompt cache (multi-prefix) and MTP ON.
#
# NOTE: When MTP is active, the server forces prompt cache OFF. This is a
# hard limit — a target-only cache snapshot cannot restore the draft stream.
#
# The config is referred to as "ON/ON" for consistency with the benchmark
# matrix, but the cache will run in OFF mode in practice.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."

BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODEL="$BASE_DIR/models/qwen36-8bit.gturbo"
MTP_MODEL="$BASE_DIR/models/qwen36-mtp.gturbo"
PORT=8083

if [[ ! -f "$BINARY" ]]; then
  echo "ERROR: NVMAIServer binary not found at $BINARY" >&2
  echo "Build it first: swift build -c release" >&2
  exit 1
fi

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: 8-bit model not found at $MODEL" >&2
  exit 1
fi

if [[ ! -d "$MTP_MODEL" ]]; then
  echo "ERROR: MTP sidecar not found at $MTP_MODEL" >&2
  exit 1
fi

echo "============================================================"
echo " NVMAIServer — 8-bit + cache ON + MTP ON"
echo "============================================================"
echo " Port:      $PORT"
echo " Model:     $MODEL"
echo " MTP Model: $MTP_MODEL"
echo "============================================================"
echo ""

# Kill any stale server on this port
if lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "Stopping existing process on port $PORT..."
  lsof -ti :"$PORT" | xargs kill
  sleep 2
fi

exec "$BINARY" \
  --model "$MODEL" \
  --mtp-model "$MTP_MODEL" \
  --port "$PORT" \
  --prompt-cache-mode multi-prefix