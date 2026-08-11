#!/usr/bin/env bash
# One-command Codex over NVMAI: starts the server if it isn't already
# running, then hands the rest of the arguments to the Codex CLI.
#
#   benchmark/codex.sh exec "Reply with exactly the number 17, nothing else."
#   benchmark/codex.sh                      # interactive Codex session
#
# Overrides: NVMAI_PORT (default 8081), CODEX (default ~/.local/bin/codex).
# The Codex config lives in its own CODEX_HOME (~/.codex-nvmai) so your real
# ~/.codex setup is never touched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
PORT="${NVMAI_PORT:-8081}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
CODEX="${CODEX:-$HOME/.local/bin/codex}"
MODEL="qwen3.6-35b-a3b"
CONFIG_DIR="${CODEX_HOME_NVMAI:-$HOME/.codex-nvmai}"

# 1. Ensure the NVMAI provider config for Codex.
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.toml" <<EOF
model = "$MODEL"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "$BASE_URL"
wire_api = "responses"
EOF

# 2. Start the server if nothing is listening yet.
if ! curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "Starting NVMAIServer on port $PORT..."
  nohup "$BASE_DIR/benchmark/launch_4bit.sh" >/tmp/nvmai-codex-server.log 2>&1 &
  for _ in $(seq 1 120); do
    curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 && break
    sleep 5
  done
fi
if ! curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1; then
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  exit 1
fi

# 3. Run Codex against NVMAI.
export CODEX_HOME="$CONFIG_DIR"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # NVMAI ignores it; codex needs the var set
exec "$CODEX" "$@"
