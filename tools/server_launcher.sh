#!/usr/bin/env bash
# Start the NVMAIServer for the chosen quantization and mode (interactive
# TUI or positional). Asks the same five question blocks as
# tools/cli_launcher.sh — CLI, model, quantization, mode, thinking — so the
# two launchers behave identically. Quantization/mode/thinking drive the
# server; the CLI/model answers pick the cli_launcher.sh command to connect
# a coding CLI afterwards.
#
#   tools/server_launcher.sh [codex|qwen|opencode] [fast|full] [4|8] [default|concise] [off|on]
#
# With no arguments, prompts 1-2-3-4-5 for the CLI, then the model
# (fast/full), then the quantization, then default/concise mode, then
# thinking off/on. Every choice has a default (codex / full / 8-bit /
# standard / thinking off), so pressing Enter through the prompts launches
# that configuration. The server runtime is pinned to native 262,144-token
# context, a 256 MiB multi-prefix prompt cache, an 8 GiB routed-expert
# cache, 8-bit KV, and MTP off. Stops any stale
# NVMAIServer on the port and starts a
# fresh one in the foreground (Ctrl-C to stop). Once the server is up it
# prints the OpenAI API setup (base URL, API key, the chosen model ID) so
# any OpenAI-compatible client can be pointed at it. The server binds to
# 127.0.0.1.
# Overrides: NVMAI_PORT, NVMAI_CONCISE_MODE, NVMAI_THINKING_MODE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODEL="ornith-1.5-35b-a3b"
# The "<model>-fast" alias is served alongside the base model name by the
# same server; the fast alias applies the CLI-strip heuristic per request
# (chat-only speed) instead of the base model's agentic tool loop.
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
    fast|1) model_word=fast ;;
    full|0) model_word=full ;;
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
    1) model_word=full ;;
    2) model_word=fast ;;
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

# --- resolve quantization -> model directory / port ---
case "$quant" in
  4bit) MODEL_DIR="$BASE_DIR/models/ornith-1.5_35B_A3B_4Bit"; PORT="${NVMAI_PORT:-8081}" ;;
  8bit) MODEL_DIR="$BASE_DIR/models/ornith-1.5_35B_A3B_8Bit"; PORT="${NVMAI_PORT:-8083}" ;;
esac

if [[ "$mode_suffix" == "_concise" ]]; then
  export NVMAI_CONCISE_MODE=1
  concise_label="concise + "
else
  unset NVMAI_CONCISE_MODE
  concise_label=""
fi
# Thinking mode: off (default) or on. The server reads this once at load;
# on opens a <think> block in the generation prompt so the model reasons
# before answering (costs wall time and tokens); off gives direct answers.
export NVMAI_THINKING_MODE="$thinking_mode"

if [[ ! -f "$BINARY" ]]; then
  echo "ERROR: NVMAIServer binary not found at $BINARY" >&2
  echo "Build it first: swift build -c release" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: $quant model not found at $MODEL_DIR" >&2
  exit 1
fi

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

echo "Starting NVMAIServer ($quant, $mode_word, $think_word)..."
"$BINARY" \
  --model "$MODEL_DIR" \
  --port "$PORT" \
  --max-context 262144 \
  --rope-scaling none \
  --prompt-cache-mode multi-prefix \
  --prompt-cache-memory-mib 256 \
  --ram-budget 8G \
  --kv-bits 8 \
  --thinking "$thinking_mode" &
server_pid=$!

for _ in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi

# The API setup advertises the model chosen in the question flow — the fast
# alias for chat-only speed, or the base model for the agentic tool loop.
if [[ "$model_word" == fast ]]; then
  api_model="$FAST_MODEL"
  api_model_note="(fast alias, seconds-per-answer chat)"
else
  api_model="$MODEL"
  api_model_note="(full agent loop)"
fi

echo ""
echo "============================================================"
echo " NVMAIServer ready — $quant + ${concise_label}cache ON + MTP OFF"
echo "============================================================"
echo ""
echo "OpenAI API setup — point any OpenAI-compatible client at this:"
echo "  Base URL:   http://127.0.0.1:${PORT}/v1"
echo "  API key:    any value (the server does not authenticate)"
echo "  Model:      $api_model $api_model_note"
echo "  Endpoints:  POST /v1/chat/completions, POST /v1/responses"
echo "  Runtime:    context 262144 | KV 8-bit | cache on | MTP off"
echo ""
echo "Or let the CLI launcher wire Codex / Qwen Code / OpenCode for you:"
echo "  tools/cli_launcher.sh $cli $model_word $quant $mode_word $think_word"
echo ""
echo "Model: $MODEL_DIR | Thinking: $think_word | Ctrl-C to stop"
echo "============================================================"
echo ""

wait "$server_pid"
