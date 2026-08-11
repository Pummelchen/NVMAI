#!/usr/bin/env bash
# Launch a coding CLI against NVMAI in the macOS terminal (interactive TUI).
#
#   tools/nvmai-cli.sh codex [default|concise]
#   tools/nvmai-cli.sh qwen [default|concise]
#   tools/nvmai-cli.sh opencode [default|concise]
#
# Without arguments, prompts 1-2-3 for the CLI and then default/concise mode.
# Stops any running NVMAIServer, starts a fresh one on 8081 in the chosen
# mode, wires the CLI's provider config to it, then hands the terminal over
# to the CLI. Overrides: NVMAI_PORT, CODEX, QWEN, OPENCODE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
PORT="${NVMAI_PORT:-8081}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
MODEL="qwen3.6-35b-a3b"

cli="${1:-}"
if [[ -z "$cli" ]]; then
  echo "Which coding CLI do you want to launch?"
  echo "  1) Codex"
  echo "  2) Qwen Code"
  echo "  3) OpenCode"
  printf "Choice [1-3]: "
  read -r choice || exit 1
  case "$choice" in
    1) cli=codex ;;
    2) cli=qwen ;;
    3) cli=opencode ;;
    *) echo "invalid choice: $choice" >&2; exit 2 ;;
  esac
fi
case "$cli" in
  codex|qwen|opencode) : ;;
  *) echo "unknown CLI: $cli (codex|qwen|opencode)" >&2; exit 2 ;;
esac

# --- NVMAI mode: default or concise (terse answers) ---
if [[ -n "${2:-}" ]]; then
  case "$2" in
    default|1) launch_script="launch_4bit.sh" ;;
    concise|2) launch_script="launch_4bit_concise.sh" ;;
    *) echo "unknown mode: $2 (default|concise)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Which NVMAI mode?"
  echo "  1) Default"
  echo "  2) Concise (terse answers)"
  printf "Choice [1-2] (default 1): "
  read -r mode_choice || exit 1
  case "${mode_choice:-1}" in
    1) launch_script="launch_4bit.sh" ;;
    2) launch_script="launch_4bit_concise.sh" ;;
    *) echo "invalid choice: $mode_choice" >&2; exit 2 ;;
  esac
fi

# --- clean slate: stop any running NVMAIServer, then start fresh ---
if pgrep -x NVMAIServer >/dev/null 2>&1; then
  echo "Stopping existing NVMAIServer instance(s)..."
  pkill -x NVMAIServer
  for _ in $(seq 1 50); do
    pgrep -x NVMAIServer >/dev/null 2>&1 || break
    sleep 0.2
  done
  if pgrep -x NVMAIServer >/dev/null 2>&1; then
    echo "ERROR: NVMAIServer did not stop" >&2
    exit 1
  fi
fi
echo "Starting NVMAIServer ($launch_script)..."
nohup "$BASE_DIR/tools/$launch_script" >/tmp/nvmai-cli-server.log 2>&1 &
for _ in $(seq 1 120); do
  curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 || {
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  exit 1
}
echo "NVMAIServer ready at $BASE_URL (model $MODEL)"

case "$cli" in
  codex)
    CONFIG_DIR="${CODEX_HOME_NVMAI:-$HOME/.codex-nvmai}"
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.toml" <<EOF
model = "$MODEL"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "$BASE_URL"
wire_api = "responses"
EOF
    export CODEX_HOME="$CONFIG_DIR"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
    echo "Launching Codex..."
    exec "${CODEX:-$HOME/.local/bin/codex}"
    ;;
  qwen)
    mkdir -p "$HOME/.config/qwen-code"
    cat > "$HOME/.config/qwen-code/config.toml" <<EOF
model = "$MODEL"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "$BASE_URL"
wire_api = "responses"
EOF
    export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
    echo "Launching Qwen Code..."
    exec "${QWEN:-$HOME/.qwen-code/bin/qwen-code}" -i
    ;;
  opencode)
    # OpenCode reads the built-in openai provider override in its global
    # config (baseURL -> NVMAI), so no per-run config is written here.
    if ! grep -q "$BASE_URL" "$HOME/.config/opencode/opencode.jsonc" 2>/dev/null; then
      echo "WARNING: opencode global config does not point the openai provider at $BASE_URL" >&2
    fi
    echo "Launching OpenCode..."
    exec "${OPENCODE:-/opt/homebrew/bin/opencode}"
    ;;
  *)
    echo "unknown CLI: $cli (codex|qwen|opencode)" >&2
    exit 2
    ;;
esac
