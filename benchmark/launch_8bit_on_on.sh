#!/usr/bin/env bash
# Launch NVMAIServer for 8-bit model with prompt cache ON, MTP OFF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."

BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODEL="$BASE_DIR/models/qwen36-8bit.gturbo"
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

echo "============================================================"
echo " NVMAIServer — 8-bit + cache ON + MTP OFF"
echo "============================================================"
echo " Port:      $PORT"
echo " Model:     $MODEL"
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
  --port "$PORT" \
  --prompt-cache-mode multi-prefix