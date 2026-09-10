#!/usr/bin/env bash
# Start NVMAIServer on one port with every installed model reachable by
# name (interactive TUI or positional). Asks which API the client speaks,
# full agent loop or fast chat, which model to load first (one list of
# every installed model and quantization, GPU and CPU, read from the
# server's own catalog), standard or concise, and the thinking level the
# chosen model should use. The server speaks OpenAI and Anthropic at once,
# so the API answer changes nothing it runs: it only picks which client
# setup is printed once the server is up.
#
#   tools/server_launcher.sh [openai|anthropic] [fast|full] [<model>] [4|8] [default|concise] [<thinking>]
#
# <model> is a catalog id (e.g. ornith-1.5-35b-a3b_8-Bit, which names its
# own width, so 4|8 may be left out after it) or an install key --
# ornith|qwen36|agentworld|qwen38, or qwen35-2b|qwen35-4b for the CPU
# models -- followed by the width. <thinking> is
# off, on, or any level the chosen model lists (minimal, low, medium, high,
# xhigh, max). The first position still takes codex|qwen|opencode and reads
# them as openai, because the start scripts and the memory harness pass
# codex. The old five-argument form without the model still means Ornith.
#
# With no arguments, prompts for each in turn. Every choice has a default
# (openai / full / Ornith 8-bit, or the first model listed / standard /
# thinking off), so pressing Enter through the prompts launches that.
#
# The server runs in dynamic mode: --models-dir plus the model to load
# first, so a client that names another installed model gets it, the server
# swapping models on demand with one resident at a time. GPU models run
# pinned to native 262,144-token context, a 256 MiB multi-prefix prompt
# cache, 8-bit KV, and MTP off. Everything tuned per model and quantization
# -- the routed-expert cache budget, expert prefetch and its disk I/O tier,
# the prefill chunk and the sampling defaults -- is deliberately NOT set
# here: the runtime resolves it from the install's tuning profile
# (ModelProfile, one row per model and width, clamped to this machine's
# RAM), so the launcher never overrides a measured optimum. CPU models get
# --cpu and none of the pinned flags: that backend has no prompt cache and
# no quantized KV, and clamps the context to what four cores can walk.
#
# If the catalog cannot be read -- a binary that predates --catalog, no
# python3 -- the launcher says so, offers the built-in list of GPU installs,
# and starts the single-model command line such a binary accepts.
#
# Stops any stale NVMAIServer on the port and starts a fresh one in the
# foreground (Ctrl-C to stop). One port for every model, 8080; the server
# binds to 127.0.0.1 only.
# Overrides: NVMAI_PORT, NVMAI_THINKING_MODE (the default thinking answer),
# NVMAI_CATALOG_JSON (read the catalog from a file, not the binary),
# NVMAI_LAUNCHER_DRY_RUN=1 (print the server command and client setup;
# start nothing, stop nothing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODELS_DIR="$BASE_DIR/models"
# One catalogue for the model list, the install paths and the port.
# shellcheck source=tools/nvmai_models.sh
source "$SCRIPT_DIR/nvmai_models.sh"

# A dry run asks and checks everything that decides the command, then
# prints it instead of running it. It never stops a server either, so it is
# safe to run beside one somebody is using.
DRY_RUN=0
if [[ "${NVMAI_LAUNCHER_DRY_RUN:-0}" == 1 ]]; then DRY_RUN=1; fi

# Before the questions, not after them: a missing build is the one answer
# that makes every other answer moot.
if [[ ! -x "$BINARY" ]]; then
  if (( DRY_RUN )); then
    echo "(dry run: no NVMAIServer binary at $BINARY yet)" >&2
  else
    echo "ERROR: NVMAIServer binary not found at $BINARY" >&2
    echo "Build it first: swift build -c release" >&2
    exit 1
  fi
fi

# Old five-argument form (no AI model): insert the default so the positions
# below line up.
case "${3:-}" in
  4|8|4bit|8bit) set -- "${1:-}" "${2:-}" ornith "$3" "${4:-}" "${5:-}" ;;
esac

backend_label() {
  case "$1" in cpu) echo CPU ;; *) echo GPU ;; esac
}

# --- 1) API: openai (default) / anthropic ---
api="${1:-}"
if [[ -z "$api" ]]; then
  echo "Which API will your client use?"
  echo "  1) OpenAI (default)"
  echo "  2) Anthropic"
  printf "Choice [1-2] (default 1): "
  read -r api_choice || exit 1
  case "${api_choice:-1}" in
    1) api=openai ;;
    2) api=anthropic ;;
    *) echo "invalid choice: $api_choice" >&2; exit 2 ;;
  esac
fi
case "$api" in
  openai|anthropic) : ;;
  # The coding-CLI answers from before the server spoke two APIs. All three
  # CLIs talk OpenAI.
  codex|qwen|opencode) api=openai ;;
  *) echo "unknown API: $api (openai|anthropic)" >&2; exit 2 ;;
esac

# --- 2) full (default, agentic tool loop) or fast (strip boilerplate) ---
if [[ -n "${2:-}" ]]; then
  case "$2" in
    fast|1) model_word=fast ;;
    full|0) model_word=full ;;
    *) echo "unknown model: $2 (fast|full)" >&2; exit 2 ;;
  esac
else
  echo ""
  echo "Full agent loop or fast chat?"
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

# --- the installed models: the server's catalog, or the built-in list ---
if nvmai_load_catalog "$BINARY" "$MODELS_DIR"; then
  dynamic=1
else
  dynamic=0
  echo "" >&2
  echo "NOTE: the model catalog is unavailable ($NVMAI_CATALOG_ERROR)." >&2
  echo "      Offering the built-in list of GPU installs instead; the server will" >&2
  echo "      serve only the model chosen here, and switching needs a restart." >&2
  nvmai_static_catalog "$MODELS_DIR"
fi

# --- 3) model: one list, every installed model and quantization ---
model_arg="${3:-}"
if [[ -n "$model_arg" ]]; then
  if idx="$(nvmai_catalog_find_id "$model_arg")"; then
    case "${4:-}" in
      4|8|4bit|8bit)
        if [[ "${4%bit}" != "${NVMAI_CAT_QUANT[$idx]}" ]]; then
          echo "$model_arg is ${NVMAI_CAT_QUANT[$idx]}-bit, not ${4%bit}-bit" >&2
          exit 2
        fi ;;
      # An id names its own width; with none after it, shift the rest
      # into the positions they have in the key form.
      *) set -- "$1" "$2" "$3" "" "${4:-}" "${5:-}" ;;
    esac
  else
    if ! nvmai_resolve_model "$model_arg" 2>/dev/null; then
      echo "unknown model: $model_arg (a model id, or ornith|qwen36|agentworld|qwen38|qwen35-2b|qwen35-4b)" >&2
      if (( dynamic )); then
        echo "installed: ${NVMAI_CAT_ID[*]}" >&2
      fi
      exit 2
    fi
    # No width after a key means 8-bit, the old quantization default.
    nvmai_resolve_quant "${4:-8}" || exit 2
    if ! idx="$(nvmai_catalog_find_dir "${NVMAI_MODEL_STEM}_${NVMAI_QUANT_DIR}")"; then
      echo "ERROR: $NVMAI_MODEL_LABEL $NVMAI_QUANT is not installed (the catalog has no ${NVMAI_MODEL_STEM}_${NVMAI_QUANT_DIR})" >&2
      echo "Install it first: tools/install_models.sh (see --help for the target names)" >&2
      exit 1
    fi
  fi
else
  count=${#NVMAI_CAT_ID[@]}
  default_idx="$(nvmai_catalog_find_dir ornith-1.5_35B_A3B_8Bit)" || default_idx=0
  echo ""
  if (( dynamic )); then
    echo "Which model? (loaded first; every other one stays available by name)"
  else
    echo "Which model?"
  fi
  for (( i = 0; i < count; i++ )); do
    size=""
    if [[ "${NVMAI_CAT_SIZE[$i]}" != "-" ]]; then size="${NVMAI_CAT_SIZE[$i]} GB"; fi
    id_column=""
    if (( dynamic )); then id_column="${NVMAI_CAT_ID[$i]}"; fi
    note=""
    if (( i == default_idx )); then note="  (default)"; fi
    if (( ! dynamic )) && [[ ! -e "${NVMAI_CAT_PATH[$i]}" ]]; then note="$note  (not installed)"; fi
    printf "  %2d) %-28s %s-bit  %s  %8s  %s%s\n" "$((i + 1))" "${NVMAI_CAT_NAME[$i]}" \
      "${NVMAI_CAT_QUANT[$i]}" "$(backend_label "${NVMAI_CAT_BACKEND[$i]}")" "$size" "$id_column" "$note"
  done
  printf "Choice [1-%d] (default %d): " "$count" "$((default_idx + 1))"
  read -r pick || exit 1
  pick="${pick:-$((default_idx + 1))}"
  if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= count )); then
    idx=$((pick - 1))
  else
    echo "invalid choice: $pick" >&2
    exit 2
  fi
fi
MODEL_ID="${NVMAI_CAT_ID[$idx]}"
MODEL_NAME="${NVMAI_CAT_NAME[$idx]}"
MODEL_QUANT="${NVMAI_CAT_QUANT[$idx]}"
MODEL_BACKEND="${NVMAI_CAT_BACKEND[$idx]}"
MODEL_DIR="${NVMAI_CAT_PATH[$idx]}"
IFS=',' read -r -a levels <<< "${NVMAI_CAT_THINKING[$idx]}"

# --- 5) NVMAI mode: standard (default) or concise ---
if [[ -n "${5:-}" ]]; then
  case "$5" in
    default|1) mode_suffix="" ; mode_word=default ;;
    concise|2) mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "unknown mode: $5 (default|concise)" >&2; exit 2 ;;
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

# --- 6) thinking: only the levels this model lists, off first and default ---
thinking_label() {
  case "$1" in xhigh) echo "extra high" ;; *) echo "$1" ;; esac
}
has_level() {
  local level
  for level in "${levels[@]}"; do
    if [[ "$level" == "$1" ]]; then return 0; fi
  done
  return 1
}
# Any spelling of the old on/off switch, or a level by name.
normalize_level() {
  local word
  word="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$word" in
    0|off|nothink|false|no) echo off ;;
    1|on|think|true|yes) echo on ;;
    minimal|low|medium|high|max) echo "$word" ;;
    xhigh|extra-high|"extra high") echo xhigh ;;
    *) return 1 ;;
  esac
}
# A normalized level -> the level this model uses for it. "on" has no
# exact match on a model with effort levels (the old --thinking on left the
# effort to the chat template), so it takes medium, the middle of every
# effort scale here, or failing that the first level above off.
level_for_model() {
  if has_level "$1"; then echo "$1"; return 0; fi
  if [[ "$1" == on ]] && (( ${#levels[@]} > 1 )); then
    if has_level medium; then echo medium; else echo "${levels[1]}"; fi
    return 0
  fi
  return 1
}
level_names() {
  local level out=""
  for level in "${levels[@]}"; do out="${out:+$out, }$(thinking_label "$level")"; done
  echo "$out"
}

thinking_default="${NVMAI_THINKING_MODE:-off}"
if ! default_word="$(normalize_level "$thinking_default")"; then
  echo "invalid NVMAI_THINKING_MODE: $thinking_default (off|on|minimal|low|medium|high|xhigh|max)" >&2
  exit 2
fi
# The variable applies to every model, so a level this one lacks falls back
# to off here rather than refusing to start.
default_level="$(level_for_model "$default_word")" || default_level="${levels[0]}"

if [[ -n "${6:-}" ]]; then
  if ! word="$(normalize_level "$6")"; then
    echo "unknown thinking level: $6 (off|on|minimal|low|medium|high|xhigh|max)" >&2
    exit 2
  fi
  if ! thinking_level="$(level_for_model "$word")"; then
    echo "$MODEL_NAME ${MODEL_QUANT}-bit has no thinking level $6; it has: $(level_names)" >&2
    exit 2
  fi
  if [[ "$word" != "$thinking_level" ]]; then
    echo "Thinking on -> $(thinking_label "$thinking_level"): $MODEL_NAME takes an effort level, not on/off." >&2
  fi
elif (( ${#levels[@]} == 1 )); then
  thinking_level="${levels[0]}"
else
  echo ""
  if [[ "${levels[*]}" == "off on" ]]; then
    echo "Reasoning (thinking)?"
  else
    echo "Reasoning effort?"
  fi
  default_choice=1
  for (( i = 0; i < ${#levels[@]}; i++ )); do
    level="${levels[$i]}"
    note=""
    case "$level" in
      off) note="direct answers" ;;
      on) note="model reasons before answering" ;;
    esac
    if [[ "$level" == "$default_level" ]]; then
      default_choice=$((i + 1))
      note="${note:+$note, }default"
    fi
    printf "  %d) %s%s\n" "$((i + 1))" "$(thinking_label "$level")" "${note:+ ($note)}"
  done
  printf "Choice [1-%d] (default %d): " "${#levels[@]}" "$default_choice"
  read -r think_choice || exit 1
  think_choice="${think_choice:-$default_choice}"
  if [[ "$think_choice" =~ ^[0-9]+$ ]] && (( think_choice >= 1 && think_choice <= ${#levels[@]} )); then
    thinking_level="${levels[$((think_choice - 1))]}"
  else
    echo "invalid choice: $think_choice" >&2
    exit 2
  fi
fi
think_word="$(thinking_label "$thinking_level")"
# The on/off projection of the level: what NVMAI_THINKING_MODE and the old
# --thinking flag understand.
if [[ "$thinking_level" == off ]]; then thinking_mode=off; else thinking_mode=on; fi

# --- one port for every model ---
PORT="${NVMAI_PORT:-$NVMAI_DEFAULT_PORT}"

if [[ "$mode_suffix" == "_concise" ]]; then
  export NVMAI_CONCISE_MODE=1
  concise_label="concise, "
else
  unset NVMAI_CONCISE_MODE
  concise_label=""
fi
# The server reads this once at load. A level beyond on travels in
# --reasoning; this keeps anything that still reads the variable agreeing
# with it.
export NVMAI_THINKING_MODE="$thinking_mode"

# --- the server command ---
# Pinned for GPU models; the per-install tuning comes from the profile.
gpu_runtime=(--max-context 262144 --rope-scaling none --prompt-cache-mode multi-prefix --prompt-cache-memory-mib 256 --kv-bits 8)
if (( dynamic )); then
  # No --cpu: the catalog knows which engine each model uses, and the server
  # refuses the flag here. The GPU flags go in even when the first model is a
  # CPU one, because a client can switch to a GPU model later and that load
  # takes them from this command line; a CPU load ignores them.
  server_cmd=("$BINARY" --models-dir "$MODELS_DIR" --model "$MODEL_ID" --reasoning "$thinking_level" --port "$PORT" "${gpu_runtime[@]}")
else
  # A binary without --catalog has no --models-dir or --reasoning either;
  # this is the single-model command line it has always accepted.
  server_cmd=("$BINARY" --model "$MODEL_DIR" --port "$PORT" "${gpu_runtime[@]}" --thinking "$thinking_mode")
fi

if [[ "$MODEL_BACKEND" == cpu ]]; then
  runtime_note="CPU backend | no prompt cache | context clamped by the backend | sampling from the model"
else
  runtime_note="context 262144 | KV 8-bit | cache on | MTP off | expert cache, prefetch, sampling from the model profile"
fi

# The API id is what the catalog calls the install (it ends in the
# routed-expert width, e.g. ornith-1.5-35b-a3b_8-Bit; the bare name is not
# accepted). The "<id>-fast" alias serves the same weights with the
# CLI-strip heuristic (chat-only speed) instead of the agentic tool loop.
set_api_model() {
  if [[ "$model_word" == fast ]]; then
    api_model="${1}-fast"
    api_model_note="(fast alias, seconds-per-answer chat)"
  else
    api_model="$1"
    api_model_note="(full agent loop)"
  fi
}

print_setup() {
  echo ""
  echo "============================================================"
  echo " NVMAIServer ready — $MODEL_NAME ${MODEL_QUANT}-bit ($(backend_label "$MODEL_BACKEND")), ${concise_label}thinking $think_word"
  echo "============================================================"
  echo ""
  if [[ "$api" == anthropic ]]; then
    echo "Anthropic API setup — point any Anthropic Messages client at this:"
    echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
    echo "  ANTHROPIC_API_KEY=nvmai   (any value; the server does not authenticate)"
    echo "  Model:      $api_model $api_model_note"
    echo "  Endpoint:   POST /v1/messages"
    echo "  Claude Code names a model for its background tasks too; without these it"
    echo "  asks for a claude-* id this server does not have and gets a 404:"
    echo "  ANTHROPIC_MODEL=$api_model"
    echo "  ANTHROPIC_DEFAULT_HAIKU_MODEL=$api_model"
    echo "  ANTHROPIC_SMALL_FAST_MODEL=$api_model   (older Claude Code releases)"
  else
    echo "OpenAI API setup — point any OpenAI-compatible client at this:"
    echo "  Base URL:   http://127.0.0.1:${PORT}/v1"
    echo "  API key:    any value (the server does not authenticate)"
    echo "  Model:      $api_model $api_model_note"
    echo "  Endpoints:  POST /v1/chat/completions, POST /v1/responses"
  fi
  echo ""
  if (( dynamic )); then
    echo "Every installed model is available by name through the API: send any"
    echo "id from GET /v1/models and the server switches to it on demand, keeping"
    echo "one model resident at a time (a switch reloads weights)."
  else
    echo "Single-model mode (no catalog): this server serves only the model above;"
    echo "switching models needs a restart."
  fi
  echo "  Runtime:    $runtime_note"
  echo ""
  echo "Model: $MODEL_DIR | Thinking: $think_word | Ctrl-C to stop"
  echo "============================================================"
  echo ""
}

# Persistent memory, when NVMAI_MEMORY=1. The workspace is the directory
# this was launched from, so each repository keeps its own memory.
nvmai_export_memory_environment "$PWD"

if (( DRY_RUN )); then
  if (( dynamic )); then
    set_api_model "$MODEL_ID"
  else
    set_api_model "<the id GET /v1/models reports>"
  fi
  echo ""
  echo "DRY RUN (NVMAI_LAUNCHER_DRY_RUN=1): nothing started, nothing stopped."
  echo "Environment: NVMAI_THINKING_MODE=$NVMAI_THINKING_MODE${NVMAI_CONCISE_MODE:+ NVMAI_CONCISE_MODE=$NVMAI_CONCISE_MODE} NVMAI_MEMORY=$NVMAI_MEMORY"
  echo "Server command:"
  printf '  '
  printf '%q ' "${server_cmd[@]}"
  echo ""
  echo ""
  echo "Once the server is up, the launcher prints:"
  print_setup
  exit 0
fi

if [[ ! -e "$MODEL_DIR" ]]; then
  echo "ERROR: $MODEL_NAME ${MODEL_QUANT}-bit not found at $MODEL_DIR" >&2
  echo "Install it first: tools/install_models.sh (see --help for the target names)" >&2
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

if [[ "${NVMAI_MEMORY:-0}" == "1" ]]; then
  echo "Memory: on (in-process, ${NVMAI_MEMORY_DIR}${NVMAI_MEMORY_CACHE_MIB:+, cap ${NVMAI_MEMORY_CACHE_MIB} MiB}, workspace $(basename "$PWD"))"
fi

echo "Starting NVMAIServer ($MODEL_NAME ${MODEL_QUANT}-bit $(backend_label "$MODEL_BACKEND"), $model_word, $mode_word, thinking $think_word)..."
# No RAM-budget, expert-slot, prefill-chunk or sampling flags here:
# the runtime takes them from the install's tuning profile (logged at load
# under NVMAI_RUNNER_STATS=1 as "NVMAI profile ...").
"${server_cmd[@]}" &
server_pid=$!

for _ in $(seq 1 120); do
  curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
if ! models_json="$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" || [[ -z "$models_json" ]]; then
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi

if (( dynamic )); then
  # A dynamic server lists every installed model, so the first id is not
  # necessarily the one just loaded; the catalog already named it.
  MODEL="$MODEL_ID"
  if ! grep -qF "\"$MODEL\"" <<< "$models_json"; then
    echo "WARNING: /v1/models does not list $MODEL" >&2
  fi
else
  MODEL="$(grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' <<< "$models_json" \
    | sed -E 's/.*"([^"]+)"$/\1/' | grep -v -- '-fast$' | head -1)"
  if [[ -z "$MODEL" ]]; then
    echo "ERROR: could not read the model id from /v1/models" >&2
    kill "$server_pid" 2>/dev/null || true
    exit 1
  fi
fi
set_api_model "$MODEL"
print_setup

wait "$server_pid"
