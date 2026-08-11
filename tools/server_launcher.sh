#!/usr/bin/env bash
# Start the NVMAIServer for the chosen quantization and mode (interactive
# TUI or positional). Asks the same five question blocks as
# tools/cli_launcher.sh — CLI, model, quantization, mode, reasoning — so the
# two launchers behave identically. Quantization/mode/reasoning drive the
# server; the CLI/model answers pick the cli_launcher.sh command to connect
# a coding CLI afterwards.
#
#   tools/server_launcher.sh [codex|qwen|opencode] [fast|full] [4|6|8] [default|concise] [nothink|think]
#
# With no arguments, prompts 1-2-3-4-5 for the CLI, then the model
# (fast/full), then the quantization, then default/concise mode, then
# reasoning off/on. Every choice has a default (codex / full / 4-bit /
# default / thinking on), so pressing Enter through the prompts launches
# that configuration. Stops any stale NVMAIServer on the port and starts a
# fresh one in the foreground (Ctrl-C to stop). Once the server is up it
# prints the OpenAI API setup (base URL, API key, model IDs) so any
# OpenAI-compatible client can be pointed at it. The server binds to
# 127.0.0.1.
# Overrides: NVMAI_PORT, NVMAI_CONCISE_MODE, NVMAI_THINKING_MODE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODEL="qwen3.6-35b-a3b"
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
    default|1) mode_suffix="" ; mode_word=default ;;
    concise|2) mode_suffix="_concise" ; mode_word=concise ;;
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
    1) mode_suffix="" ; mode_word=default ;;
    2) mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "invalid choice: $mode_choice" >&2; exit 2 ;;
  esac
fi

# --- 5) reasoning: on (default) or off (direct answers) ---
if [[ -n "${5:-}" ]]; then
  case "$5" in
    nothink|0|off) thinking="" ; think_word=nothink ;;
    think|1|on) thinking="1" ; think_word=think ;;
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
    1) thinking="1" ; think_word=think ;;
    2) thinking="" ; think_word=nothink ;;
    *) echo "invalid choice: $think_choice" >&2; exit 2 ;;
  esac
fi

# --- resolve quantization -> model directory / port ---
case "$quant" in
  4bit) MODEL_DIR="$BASE_DIR/models/qwen3.6_35B_A3B_4Bit"; PORT="${NVMAI_PORT:-8081}" ;;
  6bit) MODEL_DIR="$BASE_DIR/models/qwen3.6_35B_A3B_6Bit"; PORT="${NVMAI_PORT:-8082}" ;;
  8bit) MODEL_DIR="$BASE_DIR/models/qwen3.6_35B_A3B_8Bit"; PORT="${NVMAI_PORT:-8083}" ;;
esac

if [[ "$mode_suffix" == "_concise" ]]; then
  export NVMAI_CONCISE_MODE=1
  concise_label="concise + "
else
  concise_label=""
fi
# Reasoning mode: on (default) or off. The server reads this once at load;
# on opens a <think> block in the generation prompt so the model reasons
# before answering (costs wall time and tokens); off gives direct answers.
export NVMAI_THINKING_MODE="${thinking:-1}"

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
  --prompt-cache-mode multi-prefix &
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

echo ""
echo "============================================================"
echo " NVMAIServer ready — $quant + ${concise_label}cache ON + MTP OFF"
echo "============================================================"
echo ""
echo "OpenAI API setup — point any OpenAI-compatible client at this:"
echo "  Base URL:   http://127.0.0.1:${PORT}/v1"
echo "  API key:    any value (the server does not authenticate)"
echo "  Models:     $MODEL       (full agent loop)"
echo "              $FAST_MODEL  (fast alias, seconds-per-answer chat)"
echo "  Endpoints:  POST /v1/chat/completions, POST /v1/responses"
echo ""
echo "Or let the CLI launcher wire Codex / Qwen Code / OpenCode for you:"
echo "  tools/cli_launcher.sh $cli $model_word $quant $mode_word $think_word"
echo ""
echo "Model: $MODEL_DIR | Reasoning: $think_word | Ctrl-C to stop"
echo "============================================================"
echo ""

wait "$server_pid"
