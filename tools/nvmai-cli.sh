#!/usr/bin/env bash
# Launch a coding CLI against NVMAI in the macOS terminal (interactive TUI).
#
#   tools/nvmai-cli.sh [codex|qwen|opencode] [fast|full] [4|6|8] [default|concise] [nothink|think]
#
# With no arguments, prompts 1-2-3-4-5 for the CLI, then the model
# (fast/full), then the quantization, then default/concise mode, then
# reasoning off/on. Every choice has a default (codex / full / 4-bit /
# default / thinking on), so pressing Enter through the prompts launches
# that configuration. Stops any running NVMAIServer, starts a fresh one on
# 8081 in the chosen quant/mode, wires the CLI's provider config to the
# chosen model (the "<model>-fast" alias strips CLI boilerplate before
# prefill for seconds-per-answer chat speed; the base model keeps the CLI's
# agentic tool loop), then hands the terminal over to the CLI.
# Overrides: NVMAI_PORT, CODEX_HOME_NVMAI, QWEN_HOME_NVMAI, CODEX, QWEN,
# OPENCODE, NVMAI_STRIP_TAGS (comma-separated scaffolding tag list).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
PORT="${NVMAI_PORT:-8081}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
MODEL="qwen3.6-35b-a3b"
# The "<model>-fast" alias serves the same weights with the CLI-strip
# heuristic enabled per request (chat-only speed): system prompts, tool
# definitions, and <system-reminder> scaffolding are dropped before prefill.
FAST_MODEL="${MODEL}-fast"

# --- 1) coding CLI: codex (default) / qwen / opencode ---
cli="${1:-}"
if [[ -z "$cli" ]]; then
  echo "Which coding CLI do you want to launch?"
  echo "  1) Codex"
  echo "  2) Qwen Code"
  echo "  3) OpenCode"
  printf "Choice [1-3] (default 1): "
  read -r cli_choice || exit 1
  case "${cli_choice:-1}" in
    1) cli=codex ;;
    2) cli=qwen ;;
    3) cli=opencode ;;
    *) echo "invalid choice: $cli_choice" >&2; exit 2 ;;
  esac
fi
case "$cli" in
  codex|qwen|opencode) : ;;
  *) echo "unknown CLI: $cli (codex|qwen|opencode)" >&2; exit 2 ;;
esac

# --- 2) model: full (default, agentic tool loop) or fast (strip boilerplate) ---
if [[ -n "${2:-}" ]]; then
  case "$2" in
    fast|1) launch_model="$FAST_MODEL" ;;
    full|0) launch_model="$MODEL" ;;
    *) echo "unknown model: $2 (fast|full)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Which model?"
  echo "  1) Full (keep agent tools; multi-thousand-token prefill, slower)"
  echo "  2) Fast (strip CLI boilerplate, seconds-per-answer chat)"
  printf "Choice [1-2] (default 1): "
  read -r model_choice || exit 1
  case "${model_choice:-1}" in
    1) launch_model="$MODEL" ;;
    2) launch_model="$FAST_MODEL" ;;
    *) echo "invalid choice: $model_choice" >&2; exit 2 ;;
  esac
fi

# --- 3) quantization: 4-bit (default) / 6-bit / 8-bit ---
quant="${3:-}"
if [[ -z "$quant" ]]; then
  echo ""
  echo "Which quantization?"
  echo "  1) 4-bit"
  echo "  2) 6-bit"
  echo "  3) 8-bit"
  printf "Choice [1-3] (default 1): "
  read -r quant_choice || exit 1
  case "${quant_choice:-1}" in
    1) quant=4bit ;;
    2) quant=6bit ;;
    3) quant=8bit ;;
    *) echo "invalid choice: $quant_choice" >&2; exit 2 ;;
  esac
fi
case "$quant" in
  4|4bit) quant=4bit ;;
  6|6bit) quant=6bit ;;
  8|8bit) quant=8bit ;;
  *) echo "unknown quantization: $quant (4|6|8)" >&2; exit 2 ;;
esac

# --- 4) NVMAI mode: default (default) or concise (terse answers) ---
if [[ -n "${4:-}" ]]; then
  case "$4" in
    default|1) mode_suffix="" ;;
    concise|2) mode_suffix="_concise" ;;
    *) echo "unknown mode: $4 (default|concise)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Which NVMAI mode?"
  echo "  1) Default"
  echo "  2) Concise (terse answers)"
  printf "Choice [1-2] (default 1): "
  read -r mode_choice || exit 1
  case "${mode_choice:-1}" in
    1) mode_suffix="" ;;
    2) mode_suffix="_concise" ;;
    *) echo "invalid choice: $mode_choice" >&2; exit 2 ;;
  esac
fi
launch_script="launch_${quant}${mode_suffix}.sh"

# --- 5) reasoning: on (default) or off (direct answers) ---
if [[ -n "${5:-}" ]]; then
  case "$5" in
    nothink|0|off) thinking="" ;;
    think|1|on) thinking="1" ;;
    *) echo "unknown reasoning: $5 (nothink|think)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Reasoning (thinking)?"
  echo "  1) On (model reasons before answering, default)"
  echo "  2) Off (direct answers)"
  printf "Choice [1-2] (default 1): "
  read -r think_choice || exit 1
  case "${think_choice:-1}" in
    1) thinking="1" ;;
    2) thinking="" ;;
    *) echo "invalid choice: $think_choice" >&2; exit 2 ;;
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
# Reasoning mode: on (default) or off. The server reads this once at load;
# on opens a <think> block in the generation prompt so the model reasons
# before answering (costs wall time and tokens); off gives direct answers.
export NVMAI_THINKING_MODE="${thinking:-1}"
nohup "$BASE_DIR/tools/$launch_script" >/tmp/nvmai-cli-server.log 2>&1 &
for _ in $(seq 1 120); do
  curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 || {
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  exit 1
}
echo "NVMAIServer ready at $BASE_URL (model $launch_model; base $MODEL)"

case "$cli" in
  codex)
    CONFIG_DIR="${CODEX_HOME_NVMAI:-$HOME/.codex-nvmai}"
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.toml" <<EOF
model = "$launch_model"
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
    # Qwen Code reads JSON settings from $QWEN_HOME (default ~/.qwen/), not a
    # Codex-style TOML. Use a dedicated home so the user's real qwen-code
    # config (providers, keys, memories) is left untouched.
    CONFIG_DIR="${QWEN_HOME_NVMAI:-$HOME/.qwen-nvmai}"
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/settings.json" <<EOF
{
  "modelProviders": {
    "openai": [
      {
        "id": "$launch_model",
        "name": "[NVMAI] $launch_model",
        "baseUrl": "$BASE_URL",
        "description": "NVMAI local server",
        "envKey": "OPENAI_API_KEY"
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "model": {
    "name": "$launch_model"
  },
  "memory": {
    "enableManagedAutoMemory": false,
    "enableManagedAutoDream": false,
    "enableAutoSkill": false
  }
}
EOF
    export QWEN_HOME="$CONFIG_DIR"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
    # NVMAI's cold 4-bit prefill of qwen-code's large system prompt can
    # exceed qwen-code's default 240s stream-idle timeout; disable it so the
    # request is not aborted mid-generation.
    export QWEN_STREAM_IDLE_TIMEOUT_MS=0
    echo "Launching Qwen Code..."
    exec "${QWEN:-$HOME/.qwen-code/bin/qwen-code}" -i
    ;;
  opencode)
    # OpenCode reads the built-in openai provider override in its global
    # config (baseURL -> NVMAI), so no per-run config is written here. The
    # global config lists both the base model and the "-fast" alias; pick
    # the matching one in the TUI ("Qwen 3.6 35B-A3B (fast)" for the
    # chat-only speed mode, "Qwen 3.6 35B-A3B" for the full agent loop).
    if ! grep -q "$BASE_URL" "$HOME/.config/opencode/opencode.jsonc" 2>/dev/null; then
      echo "WARNING: opencode global config does not point the openai provider at $BASE_URL" >&2
    fi
    echo "Launching OpenCode ($launch_model)..."
    exec "${OPENCODE:-/opt/homebrew/bin/opencode}"
    ;;
  *)
    echo "unknown CLI: $cli (codex|qwen|opencode)" >&2
    exit 2
    ;;
esac
