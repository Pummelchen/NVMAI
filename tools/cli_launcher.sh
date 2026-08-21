#!/usr/bin/env bash
# Launch a coding CLI against NVMAI in the macOS terminal (interactive TUI).
# Asks the same five question blocks as tools/server_launcher.sh — CLI,
# model, quantization, mode, thinking — and uses all of them: it stops any
# stale NVMAIServer, starts a fresh one via tools/server_launcher.sh for the
# chosen quantization/mode/thinking, wires the CLI's provider config to the
# chosen model (the "<model>-fast" alias strips CLI boilerplate before
# prefill for seconds-per-answer chat speed; the base model keeps the CLI's
# agentic tool loop), then hands the terminal over to the CLI.
#
#   tools/cli_launcher.sh [codex|qwen|opencode] [fast|full] [4|8] [default|concise] [off|on]
#
# With no arguments, prompts 1-2-3-4-5 for the CLI, then the model
# (fast/full), then the quantization, then default/concise mode, then
# thinking off/on. Every choice has a default (codex / full / 8-bit /
# standard / thinking off), so pressing Enter through the prompts launches
# that configuration. The server uses native 262,144-token context,
# a 256 MiB multi-prefix prompt cache, an 8 GiB routed-expert cache, 8-bit KV,
# and MTP off. The server keeps running after
# the CLI exits; the next
# launcher run stops it and starts fresh.
# Overrides: NVMAI_PORT, CODEX_HOME_NVMAI, QWEN_HOME_NVMAI, CODEX, QWEN,
# OPENCODE, NVMAI_STRIP_TAGS (comma-separated scaffolding tag list),
# NVMAI_THINKING_MODE (off|on).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
MODEL="ornith-1.5-35b-a3b"
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
    fast|1) launch_model="$FAST_MODEL" ; model_word=fast ;;
    full|0) launch_model="$MODEL" ; model_word=full ;;
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
    1) launch_model="$MODEL" ; model_word=full ;;
    2) launch_model="$FAST_MODEL" ; model_word=fast ;;
    *) echo "invalid choice: $model_choice" >&2; exit 2 ;;
  esac
fi

# --- 3) quantization: 8-bit (default) / 4-bit ---  (6-bit withdrawn)
quant="${3:-}"
if [[ -z "$quant" ]]; then
  echo ""
  echo "Which quantization?"
  echo "  1) 8-bit (default)"
  echo "  2) 4-bit"
  printf "Choice [1-2] (default 1): "
  read -r quant_choice || exit 1
  case "${quant_choice:-1}" in
    1) quant=8bit ;;
    2) quant=4bit ;;
    *) echo "invalid choice: $quant_choice" >&2; exit 2 ;;
  esac
fi
case "$quant" in
  4|4bit) quant=4bit ;;
  8|8bit) quant=8bit ;;
  *) echo "unknown quantization: $quant (4|8)" >&2; exit 2 ;;
esac

# --- 4) NVMAI mode: standard (default) or concise ---
if [[ -n "${4:-}" ]]; then
  case "$4" in
    default|1) mode_suffix="" ; mode_word=default ;;
    concise|2) mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "unknown mode: $4 (default|concise)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Which NVMAI mode?"
  echo "  1) Standard (default)"
  echo "  2) Concise (terse answers)"
  printf "Choice [1-2] (default 1): "
  read -r mode_choice || exit 1
  case "${mode_choice:-1}" in
    1) mode_suffix="" ; mode_word=default ;;
    2) mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "invalid choice: $mode_choice" >&2; exit 2 ;;
  esac
fi

# --- 5) thinking: off (default) or on ---
thinking_default="${NVMAI_THINKING_MODE:-off}"
case "$thinking_default" in
  0|off|false|no) thinking_default=off ; default_think_choice=1 ;;
  1|on|true|yes) thinking_default=on ; default_think_choice=2 ;;
  *) echo "invalid NVMAI_THINKING_MODE: $thinking_default (off|on)" >&2; exit 2 ;;
esac
if [[ -n "${5:-}" ]]; then
  case "$5" in
    nothink|0|off) thinking_mode=off ; think_word=off ;;
    think|1|on) thinking_mode=on ; think_word=on ;;
    *) echo "unknown thinking mode: $5 (off|on)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Reasoning (thinking)?"
  echo "  1) Off (direct answers, default)"
  echo "  2) On (model reasons before answering)"
  printf "Choice [1-2] (default %s): " "$default_think_choice"
  read -r think_choice || exit 1
  case "${think_choice:-$default_think_choice}" in
    1) thinking_mode=off ; think_word=off ;;
    2) thinking_mode=on ; think_word=on ;;
    *) echo "invalid choice: $think_choice" >&2; exit 2 ;;
  esac
fi

# --- resolve quantization -> port (server_launcher.sh picks the same) ---
case "$quant" in
  4bit) PORT="${NVMAI_PORT:-8081}" ;;
  8bit) PORT="${NVMAI_PORT:-8083}" ;;
esac
BASE_URL="http://127.0.0.1:${PORT}/v1"

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
echo "Starting NVMAIServer ($quant, $mode_word, $think_word)..."
LAUNCH_LOG_DIR="$BASE_DIR/.build/launcher-logs"
mkdir -p "$LAUNCH_LOG_DIR"
LAUNCH_LOG="$LAUNCH_LOG_DIR/nvmai-server.log"
nohup "$BASE_DIR/tools/server_launcher.sh" "$cli" "$model_word" "$quant" "$mode_word" "$thinking_mode" >"$LAUNCH_LOG" 2>&1 &
for _ in $(seq 1 120); do
  curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -s --max-time 2 "$BASE_URL/models" >/dev/null 2>&1 || {
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  echo "Check $LAUNCH_LOG" >&2
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
    # request is not aborted mid-generation. Also disable the 900s
    # stream-lifetime cap, which would otherwise abort long full-model or
    # reasoning generations mid-answer.
    export QWEN_STREAM_IDLE_TIMEOUT_MS=0
    export QWEN_STREAM_MAX_LIFETIME_MS=0
    echo "Launching Qwen Code..."
    exec "${QWEN:-$HOME/.qwen-code/bin/qwen-code}" -i
    ;;
  opencode)
    # OpenCode reads the built-in openai provider override in its global
    # config (baseURL -> NVMAI), so no per-run config is written here. The
    # global config lists both the base model and the "-fast" alias; pick
    # the matching one in the TUI ("Ornith 1.5 35B-A3B (fast)" for the
    # chat-only speed mode, "Ornith 1.5 35B-A3B" for the full agent loop).
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
