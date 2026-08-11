#!/usr/bin/env bash
# Launch NVMAIServer for 6-bit model with concise mode ON, prompt cache ON, MTP OFF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."

BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODEL="$BASE_DIR/models/qwen3.6_35B_A3B_6Bit"
PORT=8082
export NVMAI_CONCISE_MODE=1

if [[ ! -f "$BINARY" ]]; then
  echo "ERROR: NVMAIServer binary not found at $BINARY" >&2
  echo "Build it first: swift build -c release" >&2
  exit 1
fi

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: 6-bit model not found at $MODEL" >&2
  exit 1
fi

echo "============================================================"
echo " NVMAIServer — 6-bit + concise + cache ON + MTP OFF"
echo "============================================================"
echo " Port:      $PORT"
echo " Model:     $MODEL"
echo " Concise:   on (standard 6-bit prompt)"
echo "============================================================"
echo ""

# Kill only a stale NVMAIServer on this port — never an unrelated process
# that happens to hold it — then wait until the port actually frees (no
# fixed sleep, so a slow teardown can't race the new server's bind).
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Stopping existing NVMAIServer on port $PORT..."
  for pid in $(lsof -ti :"$PORT" -sTCP:LISTEN); do
    if ps -p "$pid" -o command= | grep -q NVMAIServer; then
      kill "$pid"
    else
      echo "  skipping non-NVMAIServer pid $pid on port $PORT" >&2
    fi
  done
  for _ in $(seq 1 100); do
    if ! lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port $PORT still in use after waiting for it to free" >&2
    exit 1
  fi
fi

exec "$BINARY" \
  --model "$MODEL" \
  --port "$PORT" \
  --prompt-cache-mode multi-prefix
