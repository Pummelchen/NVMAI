#!/usr/bin/env bash
# The one NVMAI launcher: start the server, or start it and open a coding
# client against it. Interactive by default; every question has a default, so
# pressing Enter through them launches the recommended setup.
#
#   tools/server_launcher.sh
#
# Positional form (backwards compatible with the old launcher):
#
#   tools/server_launcher.sh [<client>] [fast|full] [<model> [4|8]] \
#                            [default|concise] [<thinking>] [<ram>]
#
#   <client>   server (default), or one of the coding clients:
#              codex, claude, qwen, opencode, zed
#   <model>    an install key (ornith|qwen36|agentworld|katcoder|qwen38, or
#              qwen35-2b|qwen35-4b|qwen35-9b) optionally followed by 4|8, or a
#              catalog id such as ornith-1.5-35b-a3b_8-Bit, which names its own
#              width. The three qwen35-* keys are the dense Qwen 3.5 models
#              (2B, 4B, 9B): both engines implement their family
#              (`qwen3_5_dense`), so they are the one install shape whose engine
#              is a real choice -- GPU by default, CPU on request.
#   <thinking> off, on, or any level the chosen model lists
#              (minimal, low, medium, high, xhigh, max). The dense Qwen 3.5
#              models define the binary thinking switch, so their levels are
#              exactly off|on -- off for a direct answer, on to reason first.
#   <ram>      the routed-expert cache budget: 1, 2, 4, 8, 16 or 32 (GB).
#              Omit it to use the install's own measured profile. The CPU
#              engine has no expert cache, so it does not ask and the flag
#              does not apply there.
#
# Flags (override the positional form, and work in any order):
#
#   --client <c>    server|codex|claude|qwen|opencode|zed
#   --model <m>     model key or catalog id
#   --bits <4|8>    quantization for a model key
#   --engine <cpu|gpu>  which engine serves the model. Almost every install
#              declares exactly one -- a MoE family is GPU-only, a snapshot is
#              CPU-only -- and asking for the other is refused with the reason.
#              The dense Qwen 3.5 models (2B/4B/9B) are the exception: both
#              engines implement their family, so the engine is a real choice,
#              asked interactively and named by an `@cpu`/`@gpu` suffix on the
#              model id a request uses.
#   --mode <fast|full>       fast strips CLI boilerplate; full keeps tools
#   --answers <default|concise>
#   --thinking <level>
#   --ram <1|2|4|8|16|32>   expert-cache budget in GB (GPU models only)
#   --context <n|native|max> native 262144, or 524288/1048576 with --yarn
#   --kv <4|8|16>   KV-cache precision (default 8)
#   --yarn          enable YaRN context scaling
#   --port <n>      default 8080 (NVMAI_PORT overrides)
#   --memory        enable persistent agent memory for this project
#   --dry-run       print the server command and client setup; start nothing
#   --help, -h
#
# What the server runs: every installed model is reachable by name through
# the API (--models-dir), one resident at a time; GPU models are pinned to
# native 262,144-token context, a 256 MiB multi-prefix prompt cache, 8-bit KV
# and MTP off. Everything tuned per model and quantization -- the expert-cache
# budget, prefetch and its I/O tier, the prefill chunk, sampling -- comes from
# the install's own ModelProfile row, clamped to this machine's RAM, so the
# launcher never overrides a measured optimum unless you pass --ram. CPU
# models take none of the pinned flags: that backend has no prompt cache, no
# quantized KV and no expert cache, so --kv, --context, --yarn and --ram are
# reported as not applying rather than passed or dropped in silence.
#
# Overrides: NVMAI_PORT, NVMAI_THINKING_MODE, NVMAI_CATALOG_JSON,
# NVMAI_LAUNCHER_DRY_RUN=1, and the per-client ones below.
# NVMAI_LAUNCHER_ASSUME_TTY=1 answers the interactive questions from a pipe
# while still starting nothing (it is the test seam for this script's
# questions, not something a person needs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$BASE_DIR/.build/arm64-apple-macosx/release/NVMAIServer"
MODELS_DIR="$BASE_DIR/models"
# One catalogue for the model list, the install paths and the port.
# shellcheck source=tools/nvmai_models.sh
source "$SCRIPT_DIR/nvmai_models.sh"

say()  { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------"; }

# ============================================================
# Arguments
# ============================================================

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

DRY_RUN=0
if [[ "${NVMAI_LAUNCHER_DRY_RUN:-0}" == 1 ]]; then DRY_RUN=1; fi

# A dry run and a piped run must never block on a question: they take every
# default instead. Interactive only when a person is actually there -- except
# under the test seam, which answers the questions from a pipe while the dry
# run still decides that nothing is started.
INTERACTIVE=1
if [[ "$DRY_RUN" == "1" || ! -t 0 ]]; then INTERACTIVE=0; fi
if [[ "${NVMAI_LAUNCHER_ASSUME_TTY:-0}" == "1" ]]; then INTERACTIVE=1; fi

CLIENT=""; MODE=""; MODEL_ARG=""; BITS=""; ANSWERS=""; THINKING_ARG=""
RAM_ARG=""; CONTEXT_ARG=""; KV_ARG=""; YARN=0; PORT_ARG=""; MEMORY=0; ENGINE_ARG=""
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --client)   CLIENT="${2:?--client needs a value}"; shift 2 ;;
    --model)    MODEL_ARG="${2:?--model needs a value}"; shift 2 ;;
    --bits)     BITS="${2:?--bits needs 4 or 8}"; shift 2 ;;
    --engine)   ENGINE_ARG="${2:?--engine needs cpu or gpu}"; shift 2 ;;
    --mode)     MODE="${2:?--mode needs fast or full}"; shift 2 ;;
    --answers)  ANSWERS="${2:?--answers needs default or concise}"; shift 2 ;;
    --thinking) THINKING_ARG="${2:?--thinking needs a level}"; shift 2 ;;
    --ram)      RAM_ARG="${2:?--ram needs a GB size}"; shift 2 ;;
    --context)  CONTEXT_ARG="${2:?--context needs a value}"; shift 2 ;;
    --kv)       KV_ARG="${2:?--kv needs 4, 8 or 16}"; shift 2 ;;
    --yarn)     YARN=1; shift ;;
    --port)     PORT_ARG="${2:?--port needs a number}"; shift 2 ;;
    --memory)   MEMORY=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  usage; exit 0 ;;
    --)         shift; while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done ;;
    -*)         echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *)          positional+=("$1"); shift ;;
  esac
done

# Old five-argument form without a model: `... <api> <mode> <bits> ...`
# inserts the default model so the positions below line up. Kept because the
# benchmark harness and older notes call it that way.
if (( ${#positional[@]} >= 3 )); then
  case "${positional[1]}:${positional[2]}" in
    full:4|full:8|full:4bit|full:8bit|fast:4|fast:8|fast:4bit|fast:8bit)
      positional=( "${positional[0]:-}" "${positional[1]:-}" ornith "${positional[2]}" \
                   "${positional[3]:-}" "${positional[4]:-}" "${positional[5]:-}" )
      ;;
  esac
fi

# Positional defaults for anything the flags did not set. Values are consumed
# in order and disambiguated by shape, so a flag may supply the model and
# still leave the width or the later fields positional.
pos=0
# The old launcher's first position named the API (openai|anthropic) rather
# than a client. The server now speaks both at once, so such a value selects
# no client and is simply consumed.
if [[ -n "$CLIENT" && ${#positional[@]} -gt 0 ]]; then
  case "${positional[0]}" in
    openai|anthropic) pos=1 ;;
  esac
fi
# Only a value that names a client is taken as one. `ornith 4` and
# `--model ornith 4` must leave the model where the model pass looks for it.
if [[ -z "$CLIENT" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    server|codex|claude|qwen|opencode|zed|openai|anthropic|none|api)
      CLIENT="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
if [[ -z "$MODE" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    fast|full|1|0) MODE="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
if [[ -z "$MODEL_ARG" && ${#positional[@]} -gt $pos ]]; then
  MODEL_ARG="${positional[$pos]}"; pos=$((pos + 1))
fi
if [[ -z "$BITS" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    4|8|4bit|8bit) BITS="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
# Answers, thinking and RAM are optional and unlabelled, so take them in
# whatever order they appear. A value that can be none of them is ignored
# rather than silently treated as a mode.
for value in "${positional[@]:$pos}"; do
  [[ -z "$value" ]] && continue
  if [[ -z "$ANSWERS" ]]; then
    case "$value" in
      default|standard|concise) ANSWERS="$value"; continue ;;
    esac
  fi
  if [[ -z "$THINKING_ARG" ]]; then
    if normalize_level "$value" >/dev/null 2>&1; then THINKING_ARG="$value"; continue; fi
  fi
  if [[ -z "$RAM_ARG" ]]; then
    if ram_tier "$value" >/dev/null 2>&1; then RAM_ARG="$value"; continue; fi
  fi
done

# ============================================================
# 1) Client
# ============================================================

client_label() {
  case "$1" in
    server)   echo "Server only (no client)" ;;
    codex)    echo "Codex" ;;
    claude)   echo "Claude Code" ;;
    qwen)     echo "Qwen Code" ;;
    opencode) echo "OpenCode" ;;
    zed)      echo "Zed editor" ;;
    *)        echo "$1" ;;
  esac
}

normalize_client() {
  case "$1" in
    ""|server|none|api) echo server ;;
    # The coding-CLI spellings the old launcher accepted.
    codex)              echo codex ;;
    claude|claude-code) echo claude ;;
    qwen|qwen-code)     echo qwen ;;
    opencode)           echo opencode ;;
    zed)                echo zed ;;
    # The old API names meant "start the server for that API".
    openai|anthropic)   echo server ;;
    *) return 1 ;;
  esac
}

if [[ -z "$CLIENT" ]] && (( INTERACTIVE )); then
  echo "What do you want to launch?"
  echo "  1) Server only — start the API, no client (default)"
  echo "  2) Codex"
  echo "  3) Claude Code"
  echo "  4) Qwen Code"
  echo "  5) OpenCode"
  echo "  6) Zed editor"
  printf "Choice [1-6] (default 1): "
  read -r client_choice || exit 1
  case "${client_choice:-1}" in
    1) CLIENT=server ;;
    2) CLIENT=codex ;;
    3) CLIENT=claude ;;
    4) CLIENT=qwen ;;
    5) CLIENT=opencode ;;
    6) CLIENT=zed ;;
    *) echo "invalid choice: $client_choice" >&2; exit 2 ;;
  esac
fi
[[ -z "$CLIENT" ]] && CLIENT=server
if ! CLIENT="$(normalize_client "$CLIENT")"; then
  echo "unknown client: $CLIENT (server|codex|claude|qwen|opencode|zed)" >&2
  exit 2
fi

# Which API surface the client needs. Only used to print the right setup and
# to pick the right config shape; the server speaks all of them at once.
case "$CLIENT" in
  claude) API=anthropic ;;
  *)      API=openai ;;
esac

# ============================================================
# 2) Mode: full agent loop, or fast chat
# ============================================================

if [[ -n "$MODE" ]]; then
  case "$MODE" in
    fast|1) model_word=fast ;;
    full|0) model_word=full ;;
    *) echo "unknown mode: $MODE (fast|full)" >&2; exit 2 ;;
  esac
else
  # A plain API server has no agent loop to preserve, so it does not ask.
  if [[ "$CLIENT" == "server" || ! -t 0 ]]; then
    model_word=full
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
fi

# ============================================================
# 3) Model and quantization
# ============================================================

# The installed models: the server's own catalog, or the built-in list.
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

if [[ -z "$MODEL_ARG" ]]; then
  count=${#NVMAI_CAT_ID[@]}
  default_idx="$(nvmai_catalog_find_dir ornith-1.5_35B_A3B_8Bit)" || default_idx=0
  echo ""
  if (( dynamic )); then
    echo "Which model? (loaded first; every other one stays available by name)"
  else
    echo "Which model?"
  fi
  # An install both engines can serve says so: `GPU+CPU`. One is the default
  # and the other is a choice the next question offers.
  engine_column() {
    local list="$1"
    if [[ "$list" == *,* ]]; then
      printf '%s' "${list^^}" | tr ',' '+'
    else
      [[ "$list" == cpu ]] && echo CPU || echo GPU
    fi
  }
  # Engine and thinking levels are the two things a person is choosing between
  # here, so both are in the list rather than discovered after the fact.
  printf "  %-3s %-28s %-6s %-4s %8s  %-22s %-24s %s\n" \
    "#" "model" "bits" "engine" "size" "api id" "thinking" ""
  for (( i = 0; i < count; i++ )); do
    size=""
    if [[ "${NVMAI_CAT_SIZE[$i]}" != "-" ]]; then size="${NVMAI_CAT_SIZE[$i]} GB"; fi
    note=""
    if (( i == default_idx )); then note="  (default)"; fi
    if (( ! dynamic )) && [[ ! -e "${NVMAI_CAT_PATH[$i]}" ]]; then note="$note  (not installed)"; fi
    printf "  %2d) %-28s %s-bit  %-4s %8s  %-22s %-24s%s\n" "$((i + 1))" \
      "${NVMAI_CAT_NAME[$i]}" "${NVMAI_CAT_QUANT[$i]}" \
      "$( engine_column "${NVMAI_CAT_ENGINES[$i]:-${NVMAI_CAT_BACKEND[$i]}}" )" \
      "$size" "${NVMAI_CAT_ID[$i]}" "${NVMAI_CAT_THINKING[$i]//,/, }" "$note"
  done
  printf "Choice [1-%d] (default %d): " "$count" "$((default_idx + 1))"
  read -r pick || exit 1
  pick="${pick:-$((default_idx + 1))}"
  if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= count )); then
    idx=$((pick - 1))
  else
    echo "invalid choice: $pick" >&2; exit 2
  fi
else
  if idx="$(nvmai_catalog_find_id "$MODEL_ARG")"; then
    if [[ -n "$BITS" ]]; then
      case "$BITS" in
        4|8|4bit|8bit)
          if [[ "${BITS%bit}" != "${NVMAI_CAT_QUANT[$idx]}" ]]; then
            echo "$MODEL_ARG is ${NVMAI_CAT_QUANT[$idx]}-bit, not ${BITS%bit}-bit" >&2
            exit 2
          fi ;;
        *) echo "unknown bits: $BITS (4|8)" >&2; exit 2 ;;
      esac
    fi
  else
    if ! nvmai_resolve_model "$MODEL_ARG" 2>/dev/null; then
      echo "unknown model: $MODEL_ARG (a model id, or ornith|qwen36|agentworld|katcoder|qwen38|qwen35-2b|qwen35-4b|qwen35-9b)" >&2
      if (( dynamic )); then echo "installed: ${NVMAI_CAT_ID[*]}" >&2; fi
      exit 2
    fi
    # No width given means 8-bit, the historical default.
    nvmai_resolve_quant "${BITS:-8}" || exit 2
    if ! idx="$(nvmai_catalog_find_dir "${NVMAI_MODEL_STEM}_${NVMAI_QUANT_DIR}")"; then
      # The fallback list is the GPU installs, so a CPU model is absent from
      # it even when it is installed. Saying "not installed" here would send
      # someone to re-install a model that is already there.
      if (( ! dynamic )) && [[ "${NVMAI_MODEL_KEY}" == qwen35-* ]]; then
        echo "ERROR: $NVMAI_MODEL_LABEL is a CPU-engine model, and the built-in" >&2
        echo "       fallback list covers the GPU installs only, so it is not offered" >&2
        echo "       here. Two ways out:" >&2
        echo "         - let this launcher read the catalog by building the server:" >&2
        echo "             swift build -c release --product NVMAIServer" >&2
        echo "           (or point NVMAI_CATALOG_JSON at that binary's --catalog output)" >&2
        echo "         - or start it yourself, which the single-model form supports:" >&2
        echo "             .build/release/NVMAIServer --model <install dir> --cpu" >&2
        exit 1
      fi
      echo "ERROR: $NVMAI_MODEL_LABEL $NVMAI_QUANT is not installed (the catalog has no ${NVMAI_MODEL_STEM}_${NVMAI_QUANT_DIR})" >&2
      echo "Install it first: tools/install_models.sh (see --help for the target names)" >&2
      exit 1
    fi
  fi
fi

MODEL_ID="${NVMAI_CAT_ID[$idx]}"
MODEL_NAME="${NVMAI_CAT_NAME[$idx]}"
MODEL_QUANT="${NVMAI_CAT_QUANT[$idx]}"
MODEL_BACKEND="${NVMAI_CAT_BACKEND[$idx]}"
MODEL_DIR="${NVMAI_CAT_PATH[$idx]}"
IFS=',' read -r -a levels <<< "${NVMAI_CAT_THINKING[$idx]}"

# ============================================================
# 3b) Engine: which engine serves this install, and what was asked for
# ============================================================

# The catalog says which engines can serve an install. Almost every install has
# exactly one -- a MoE family is GPU-only because the CPU engine does not
# implement those shapes, and a converted snapshot is CPU-only -- but the dense
# Qwen 3.5 models (2B/4B/9B) are implemented by *both*, from the same `.gturbo`
# payload. That is the one case where the engine is a real choice, so it is the
# one case that asks.
IFS=',' read -r -a engines <<< "${NVMAI_CAT_ENGINES[$idx]:-$MODEL_BACKEND}"
default_engine="${NVMAI_CAT_BACKEND[$idx]}"
engine_family="${NVMAI_CAT_FAMILY[$idx]:--}"
engine_name() { if [[ "$1" == cpu ]]; then echo CPU; else echo GPU; fi; }
engine_available() {
  local candidate
  for candidate in "${engines[@]}"; do [[ "$candidate" == "$1" ]] && return 0; done
  return 1
}
engine_reason() {
  if [[ "$1" == cpu ]]; then
    echo "$MODEL_NAME ${MODEL_QUANT}-bit declares family $engine_family, which the GPU engine does not implement; it runs on the CPU engine"
  else
    echo "$MODEL_NAME ${MODEL_QUANT}-bit declares family $engine_family, which the CPU engine does not implement; it runs on the GPU engine"
  fi
}

if [[ -n "$ENGINE_ARG" ]]; then
  case "$ENGINE_ARG" in
    cpu|gpu) ;;
    *) echo "unknown --engine: $ENGINE_ARG (cpu|gpu)" >&2; exit 2 ;;
  esac
  if ! engine_available "$ENGINE_ARG"; then
    echo "$(engine_reason "$default_engine")." >&2
    echo "--engine $ENGINE_ARG is not available for it." >&2
    exit 2
  fi
  ENGINE="$ENGINE_ARG"
elif (( ${#engines[@]} > 1 )) && (( INTERACTIVE )); then
  echo ""
  echo "Which engine? $MODEL_NAME ${MODEL_QUANT}-bit runs on both."
  default_choice=1
  for (( i = 0; i < ${#engines[@]}; i++ )); do
    note=""
    if [[ "${engines[$i]}" == "$default_engine" ]]; then
      default_choice=$((i + 1))
      note="  (default)"
    fi
    printf "  %d) %s%s\n" "$((i + 1))" "$(engine_name "${engines[$i]}")" "$note"
  done
  printf "Choice [1-%d] (default %d): " "${#engines[@]}" "$default_choice"
  read -r engine_choice || exit 1
  engine_choice="${engine_choice:-$default_choice}"
  if [[ "$engine_choice" =~ ^[0-9]+$ ]] \
     && (( engine_choice >= 1 && engine_choice <= ${#engines[@]} )); then
    ENGINE="${engines[$((engine_choice - 1))]}"
  else
    echo "invalid choice: $engine_choice" >&2; exit 2
  fi
else
  ENGINE="$default_engine"
fi

# The engine is stated, and whether it was a choice. The list's engine column
# shows the default; this says which one this run will use and why there is (or
# is not) an alternative.
if (( ${#engines[@]} > 1 )); then
  engine_line="$(engine_name "$ENGINE") -- your choice of $(printf '%s' "${engines[*]}" | tr ' ' '/')"
else
  engine_line="$(engine_name "$ENGINE") only -- $(engine_reason "$ENGINE")"
fi
if (( INTERACTIVE )); then echo "Engine: $engine_line"; fi
# The request names the engine with an `@cpu`/`@gpu` suffix when it is not the
# install's default; the server registers those aliases from the same catalog
# field, so the two cannot disagree about what is available.
MODEL_ID_LAUNCH="$MODEL_ID"
if [[ "$ENGINE" != "$default_engine" ]]; then
  MODEL_ID_LAUNCH="${MODEL_ID}@${ENGINE}"
fi
MODEL_BACKEND="$ENGINE"

# ============================================================
# 4) Answers: standard or concise
# ============================================================

if [[ -n "$ANSWERS" ]]; then
  case "$ANSWERS" in
    default|standard|1) mode_suffix="" ; mode_word=default ;;
    concise|2)          mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "unknown answers mode: $ANSWERS (default|concise)" >&2; exit 2 ;;
  esac
elif (( ! INTERACTIVE )); then
  mode_suffix="" ; mode_word=default
else
  echo ""
  echo "Which answer style?"
  echo "  1) Standard (default)"
  echo "  2) Concise (terse answers)"
  printf "Choice [1-2] (default 1): "
  read -r answers_choice || exit 1
  case "${answers_choice:-1}" in
    1) mode_suffix="" ; mode_word=default ;;
    2) mode_suffix="_concise" ; mode_word=concise ;;
    *) echo "invalid choice: $answers_choice" >&2; exit 2 ;;
  esac
fi

# ============================================================
# 5) Thinking: only the levels this model lists, off first and default
# ============================================================

thinking_label() { case "$1" in xhigh) echo "extra high" ;; *) echo "$1" ;; esac; }
has_level() {
  local level
  for level in "${levels[@]}"; do [[ "$level" == "$1" ]] && return 0; done
  return 1
}
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
# "on" has no exact match on a model with effort levels, so it takes medium,
# the middle of every effort scale here, or failing that the first level above
# off. A level the model lacks is refused rather than mapped to a neighbour.
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
if ! default_level="$(level_for_model "$default_word")"; then
  # A default the model cannot render is a person's mistake worth naming, not
  # a level to substitute in silence.
  default_level="${levels[0]}"
  echo "NOTE: NVMAI_THINKING_MODE=$thinking_default is not a level $MODEL_NAME ${MODEL_QUANT}-bit renders" >&2
  echo "      ($(level_names)); using $default_level." >&2
fi

if [[ -n "$THINKING_ARG" ]]; then
  if ! word="$(normalize_level "$THINKING_ARG")"; then
    echo "unknown thinking level: $THINKING_ARG (off|on|minimal|low|medium|high|xhigh|max)" >&2
    exit 2
  fi
  if ! thinking_level="$(level_for_model "$word")"; then
    echo "$MODEL_NAME ${MODEL_QUANT}-bit has no thinking level $THINKING_ARG; it has: $(level_names)" >&2
    exit 2
  fi
  if [[ "$word" != "$thinking_level" ]]; then
    echo "Thinking on -> $(thinking_label "$thinking_level"): $MODEL_NAME takes an effort level, not on/off." >&2
  fi
elif (( ${#levels[@]} == 1 )); then
  thinking_level="${levels[0]}"
elif (( ! INTERACTIVE )); then
  thinking_level="$default_level"
else
  echo ""
  if [[ "${levels[*]}" == "off on" ]]; then
    echo "Reasoning (thinking)? $MODEL_NAME ${MODEL_QUANT}-bit defines the binary switch"
    echo "off|on; a client can switch it per request with reasoning_effort."
  else
    echo "Reasoning effort? $MODEL_NAME ${MODEL_QUANT}-bit renders $(level_names)."
  fi
  default_choice=1
  for (( i = 0; i < ${#levels[@]}; i++ )); do
    level="${levels[$i]}"
    note=""
    case "$level" in
      off) note="direct answers" ;;
      on)  note="model reasons before answering" ;;
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
    echo "invalid choice: $think_choice" >&2; exit 2
  fi
fi
think_word="$(thinking_label "$thinking_level")"

# ============================================================
# 6) RAM limit: the expert-cache budget
# ============================================================

# The tiers are the sizes a person can reason about, not the runtime's own
# slot rungs. The runtime derives slots from the budget and the model's own
# expert stride, so the same tier means fewer slots on a wider model.
ram_tier() {
  case "$1" in
    1|1G|1g)   echo 1 ;;
    2|2G|2g)   echo 2 ;;
    4|4G|4g)   echo 4 ;;
    8|8G|8g)   echo 8 ;;
    16|16G|16g) echo 16 ;;
    32|32G|32g) echo 32 ;;
    *) return 1 ;;
  esac
}

if [[ -n "$RAM_ARG" ]]; then
  if ! ram_gb="$(ram_tier "$RAM_ARG")"; then
    echo "unknown RAM limit: $RAM_ARG (1, 2, 4, 8, 16 or 32 GB)" >&2
    exit 2
  fi
elif [[ "$ENGINE" == "cpu" ]]; then
  # Nothing to ask: the CPU engine holds the whole model resident and has no
  # routed-expert cache, so a budget would be a number that changes nothing.
  ram_gb=""
elif (( ! INTERACTIVE )); then
  # Unattended or dry run: keep the install's own measured profile.
  ram_gb=""
else
  echo ""
  echo "RAM limit for the expert cache?"
  echo "  More memory means fewer SSD reads and faster answers, and less"
  echo "  left for everything else on the Mac."
  echo "  1) 1 GB    2) 2 GB    3) 4 GB"
  echo "  4) 8 GB    5) 16 GB   6) 32 GB"
  echo "  7) Model default (measured per install; recommended)"
  printf "Choice [1-7] (default 7): "
  read -r ram_choice || exit 1
  case "${ram_choice:-7}" in
    1) ram_gb=1 ;;  2) ram_gb=2 ;;  3) ram_gb=4 ;;
    4) ram_gb=8 ;;  5) ram_gb=16 ;; 6) ram_gb=32 ;;
    7) ram_gb="" ;;
    *) echo "invalid choice: $ram_choice" >&2; exit 2 ;;
  esac
fi
ram_note="model default (measured)"
if [[ -n "$ram_gb" ]]; then ram_note="${ram_gb} GB (your choice)"; fi

# A CPU model ignores the routed-expert cache entirely, so say what the number
# does rather than leaving a budget that never takes effect.
if [[ "$ENGINE" == "cpu" ]]; then
  ram_note="not applicable (no routed-expert cache on the CPU engine)"
  if [[ -n "$RAM_ARG" ]]; then
    echo "NOTE: $MODEL_NAME runs on the CPU and has no routed-expert cache;" >&2
    echo "      --ram $RAM_ARG does not apply to it." >&2
  fi
fi

# ============================================================
# 7) Advanced server features
# ============================================================

# Context and KV are only offered interactively when the caller asked for the
# detailed choices; the defaults are what almost everyone wants.
kv_bits="${KV_ARG:-8}"
case "$kv_bits" in 4|8|16) : ;; *) echo "unknown --kv: $kv_bits (4|8|16)" >&2; exit 2 ;; esac

max_context=""
rope_scaling="none"
if (( YARN )); then
  rope_scaling="yarn"
  max_context="${CONTEXT_ARG:-1048576}"
elif [[ -n "$CONTEXT_ARG" ]]; then
  case "$CONTEXT_ARG" in
    native|max) max_context=262144 ;;
    *[!0-9]*) echo "unknown --context: $CONTEXT_ARG (a token count, native, or max)" >&2; exit 2 ;;
    *) max_context="$CONTEXT_ARG" ;;
  esac
fi

PORT="${PORT_ARG:-${NVMAI_PORT:-$NVMAI_DEFAULT_PORT}}"

# The context, KV and YaRN flags reach the GPU runtime only. Asking for one of
# them against a CPU model is a misunderstanding worth naming: the CPU engine
# takes its context from the model's own config (clamped by the backend) and
# has no quantized KV cache to pick a width for.
if [[ "$ENGINE" == "cpu" ]]; then
  inert_flags=()
  [[ -n "$KV_ARG" ]] && inert_flags+=(--kv)
  [[ -n "$CONTEXT_ARG" ]] && inert_flags+=(--context)
  (( YARN )) && inert_flags+=(--yarn)
  if (( ${#inert_flags[@]} > 0 )); then
    echo "NOTE: ${inert_flags[*]} do not apply to the CPU engine; ignoring" >&2
    echo "      ${inert_flags[*]} for $MODEL_NAME." >&2
  fi
  kv_bits=""; max_context=""; rope_scaling="none"
fi

if [[ "$MEMORY" == "1" ]]; then
  export NVMAI_MEMORY=1
fi

# ============================================================
# 8) The server command
# ============================================================

# Pinned for GPU models; per-install tuning comes from the profile.
gpu_runtime=(--prompt-cache-mode multi-prefix --prompt-cache-memory-mib 256 --kv-bits "$kv_bits")
if [[ -n "$max_context" ]]; then
  gpu_runtime+=(--max-context "$max_context" --rope-scaling "$rope_scaling")
elif [[ "$rope_scaling" == "none" ]]; then
  gpu_runtime+=(--max-context 262144 --rope-scaling none)
fi
if [[ -n "$ram_gb" && "$MODEL_BACKEND" != "cpu" ]]; then
  gpu_runtime+=(--ram-budget "${ram_gb}G")
fi

if (( dynamic )); then
  server_cmd=("$BINARY" --models-dir "$MODELS_DIR" --model "$MODEL_ID_LAUNCH" --reasoning "$thinking_level" --port "$PORT")
else
  server_cmd=("$BINARY" --model "$MODEL_DIR" --port "$PORT" --thinking "$( [[ "$thinking_level" == off ]] && echo off || echo on )")
fi

if [[ "$MODEL_BACKEND" == "cpu" ]]; then
  if (( dynamic )); then
    # The catalog knows which engine each model uses; --cpu is refused here.
    runtime_note="CPU backend | no prompt cache | context clamped by the backend | sampling from the model"
  else
    server_cmd+=(--cpu)
    runtime_note="CPU backend | no prompt cache | context clamped by the backend | sampling from the model"
  fi
else
  server_cmd+=("${gpu_runtime[@]}")
  runtime_note="context ${max_context:-262144} | KV ${kv_bits}-bit | cache on | MTP off"
  if [[ -n "$ram_gb" ]]; then
    runtime_note="$runtime_note | expert cache ${ram_gb} GB (yours)"
  else
    runtime_note="$runtime_note | expert cache, prefetch, sampling from the model profile"
  fi
fi

# The base API id is what the catalog calls the install (it ends in the
# routed-expert width, e.g. ornith-1.5-35b-a3b_8-Bit). The "<id>-fast" alias
# serves the same weights with the CLI-strip heuristic instead of the agentic
# tool loop.
# The API id is whatever the server advertises (it ends in the routed-expert
# width, e.g. ornith-1.5-35b-a3b_8-Bit; the bare name is not accepted). It is
# read back from /v1/models once the server is up, so the client config is
# written with the id that really exists rather than a guess.
resolve_launch_model() {
  local base="$1"
  if [[ "$model_word" == "fast" ]]; then
    launch_model="${base}-fast"
    api_model_note="(fast alias, seconds-per-answer chat)"
  else
    launch_model="$base"
    api_model_note="(full agent loop)"
  fi
}
launch_model="$MODEL_ID"
api_model_note=""

# ============================================================
# 9) Printing
# ============================================================

print_setup() {
  echo ""
  echo "============================================================"
  echo " NVMAIServer ready — $MODEL_NAME ${MODEL_QUANT}-bit ($( [[ "$MODEL_BACKEND" == cpu ]] && echo CPU || echo GPU ))"
  echo "                                            answers ${mode_word}, thinking ${think_word}"
  echo "============================================================"
  echo ""
  if [[ "$API" == "anthropic" ]]; then
    echo "Anthropic API setup — point any Anthropic Messages client at this:"
    echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
    echo "  ANTHROPIC_API_KEY=nvmai   (any value; the server does not authenticate)"
    echo "  Model:      $launch_model $api_model_note"
    echo "  Endpoint:   POST /v1/messages"
  else
    echo "OpenAI API setup — point any OpenAI-compatible client at this:"
    echo "  Base URL:   http://127.0.0.1:${PORT}/v1"
    echo "  API key:    any value (the server does not authenticate)"
    echo "  Model:      $launch_model $api_model_note"
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
  echo "  Engine:     $engine_line"
  if [[ "$MEMORY" == "1" ]]; then
    echo "  Memory:     on (persistent, repo-scoped; guard on)"
  fi
  echo ""
  echo "Model: $MODEL_DIR"
  echo "Thinking: $think_word | RAM: $ram_note | Port: $PORT | Ctrl-C to stop"
  echo "============================================================"
  echo ""
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  rule
  echo "DRY RUN — nothing started, nothing stopped."
  rule
  echo "Client:     $(client_label "$CLIENT")"
  echo "Server cmd:"
  printf '  '; printf '%q ' "${server_cmd[@]}"; echo ""
  echo ""
  echo "Client setup the launcher would write/use:"
  case "$CLIENT" in
    server)   echo "  Server only; nothing to configure." ;;
    codex)    echo "  ~/.codex-nvmai/config.toml -> $launch_model via $API at port $PORT" ;;
    claude)   echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT, model $launch_model" ;;
    qwen)     echo "  ~/.qwen-nvmai/settings.json -> $launch_model at port $PORT" ;;
    opencode) echo "  ~/.config/opencode/opencode.jsonc provider 'nvmai' (model $launch_model)" ;;
    zed)      echo "  ~/.config/zed/settings.json openai_compatible 'nvmai' (model $launch_model)" ;;
  esac
  echo ""
  print_setup
  exit 0
fi

# Persistent memory, when NVMAI_MEMORY=1. The workspace is the directory this
# was launched from, so each repository keeps its own memory.
nvmai_export_memory_environment "$PWD"

if [[ "$MEMORY" == "1" ]]; then
  echo "Memory: on (in-process, ${NVMAI_MEMORY_DIR}${NVMAI_MEMORY_CACHE_MIB:+, cap ${NVMAI_MEMORY_CACHE_MIB} MiB}, workspace $(basename "$PWD"))"
fi

# ============================================================
# 10) Start the server
# ============================================================

# Stop only a stale NVMAIServer on this port -- never an unrelated process --
# then wait until the port actually frees, so a slow teardown cannot race the
# new server's bind.
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Stopping the existing NVMAIServer on port $PORT..."
  for pid in $(lsof -ti :"$PORT" -sTCP:LISTEN); do
    if ps -p "$pid" -o command= | grep -q NVMAIServer; then
      kill "$pid"
    else
      echo "  skipping non-NVMAIServer pid $pid on port $PORT" >&2
    fi
  done
  for _ in $(seq 1 100); do
    lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
fi

if [[ ! -e "$MODEL_DIR" ]]; then
  echo "ERROR: $MODEL_NAME ${MODEL_QUANT}-bit not found at $MODEL_DIR" >&2
  echo "Install it first: tools/install_models.sh (see --help for the target names)" >&2
  exit 1
fi

echo "Starting NVMAIServer ($MODEL_NAME ${MODEL_QUANT}-bit, $model_word, $mode_word, thinking $think_word) on port $PORT..."

"${server_cmd[@]}" &
server_pid=$!

# Wait for the API to answer, not for a fixed number of seconds.
for _ in $(seq 1 240); do
  curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 2
done
if ! models_json="$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" || [[ -z "$models_json" ]]; then
  echo "ERROR: NVMAIServer did not come up on port $PORT" >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi
# The id the server actually advertises for this install. The catalog knows
# it when it is available; otherwise ask the running server.
if (( dynamic )); then
  MODEL="$MODEL_ID_LAUNCH"
else
  MODEL="$(printf '%s' "$models_json" \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
    | sed -E 's/.*"([^"]+)"$//' | grep -v -- '-fast$' | head -1)"
fi
if [[ -z "$MODEL" ]]; then
  echo "WARNING: could not read the model id from /v1/models" >&2
  MODEL="$MODEL_ID"
fi
resolve_launch_model "$MODEL"

# ============================================================
# 11) Client configuration
# ============================================================

BASE_URL="http://127.0.0.1:${PORT}/v1"

# Timestamped backup, once per file per run, before anything is rewritten.
backup_once() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local backup="${file}.nvmai-backup"
  [[ -f "$backup" ]] || cp "$file" "$backup"
}

configure_codex() {
  local dir="${CODEX_HOME_NVMAI:-$HOME/.codex-nvmai}"
  mkdir -p "$dir"
  backup_once "$dir/config.toml"
  cat > "$dir/config.toml" <<EOF
# Written by NVMAI tools/server_launcher.sh
model = "$launch_model"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "$BASE_URL"
wire_api = "responses"
EOF
  export CODEX_HOME="$dir"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  echo "  Codex configured: $dir/config.toml"
}

configure_claude() {
  # Claude Code reads the Anthropic surface from the environment, so there is
  # no file to write. The two extra model variables matter: without them it
  # asks for a claude-* id for its background tasks and gets a 404.
  export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-nvmai}"
  export ANTHROPIC_MODEL="$launch_model"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$launch_model"
  export ANTHROPIC_SMALL_FAST_MODEL="$launch_model"
  echo "  Claude Code configured: ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
}

configure_qwen() {
  # Qwen Code reads JSON settings from $QWEN_HOME (default ~/.qwen/), not a
  # Codex-style TOML. A dedicated home leaves the user's real qwen-code
  # config (providers, keys, memories) untouched.
  local dir="${QWEN_HOME_NVMAI:-$HOME/.qwen-nvmai}"
  mkdir -p "$dir"
  backup_once "$dir/settings.json"
  cat > "$dir/settings.json" <<EOF
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
  export QWEN_HOME="$dir"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  # NVMAI's cold prefill of qwen-code's large system prompt can exceed
  # qwen-code's default 240s stream-idle timeout; disable it so the request is
  # not aborted mid-generation. Also disable the 900s stream-lifetime cap,
  # which would otherwise abort long reasoning generations mid-answer.
  export QWEN_STREAM_IDLE_TIMEOUT_MS=0
  export QWEN_STREAM_MAX_LIFETIME_MS=0
  echo "  Qwen Code configured: $dir/settings.json"
}

# Merge the NVMAI provider into a JSONC settings file, keeping everything
# else. Comments and trailing commas are tolerated on read; the file is
# rewritten as plain JSON (Zed and OpenCode both accept that) with a backup
# kept beside it. Each client has its own schema, so the shape is passed in:
#   zed       language_models.openai_compatible.<id> = {api_url, available_models}
#   opencode  provider.<id> = {npm, name, options.baseURL, models}
merge_client_config() {
  local file="$1" schema="$2" provider="$3" model="$4" context="$5"
  mkdir -p "$(dirname "$file")"
  backup_once "$file"
  NVMAI_MERGE_FILE="$file" NVMAI_MERGE_SCHEMA="$schema" NVMAI_MERGE_PROVIDER="$provider" \
  NVMAI_MERGE_MODEL="$model" NVMAI_MERGE_URL="$BASE_URL" NVMAI_MERGE_CONTEXT="$context" \
  python3 - <<'PY'
import json, os, re, sys

path = os.environ["NVMAI_MERGE_FILE"]
schema = os.environ["NVMAI_MERGE_SCHEMA"]
provider = os.environ["NVMAI_MERGE_PROVIDER"]
model = os.environ["NVMAI_MERGE_MODEL"]
base_url = os.environ["NVMAI_MERGE_URL"]
context = int(os.environ["NVMAI_MERGE_CONTEXT"])


def strip_comments(text):
    """JSONC -> JSON: drop // and /* */ comments outside strings."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


try:
    raw = open(path, encoding="utf-8").read()
except FileNotFoundError:
    raw = "{}"
text = re.sub(r",(\s*[}\]])", r"\1", strip_comments(raw))
try:
    data = json.loads(text) if text.strip() else {}
except json.JSONDecodeError as error:
    print(f"  WARNING: {path} is not parseable ({error}); left untouched.", file=sys.stderr)
    sys.exit(0)
if not isinstance(data, dict):
    print(f"  WARNING: {path} is not a JSON object; left untouched.", file=sys.stderr)
    sys.exit(0)


def table(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


if schema == "opencode":
    block = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "NVMAI (local)",
        "options": {"baseURL": base_url, "apiKey": "nvmai"},
        "models": {
            model: {
                "name": f"NVMAI — {model}",
                "limit": {"context": context, "output": 65536},
            }
        },
    }
    table(table(data, "provider"), provider).update(block)
else:  # zed
    block = {
        "api_url": base_url,
        "available_models": [
            {
                "name": model,
                "display_name": f"NVMAI — {model}",
                "max_tokens": context,
                "capabilities": {
                    "tools": True,
                    "images": False,
                    "parallel_tool_calls": False,
                    "prompt_cache_key": False,
                    "chat_completions": True,
                    "interleaved_reasoning": False,
                    "max_tokens_parameter": False,
                },
            }
        ],
    }
    table(table(data, "language_models"), "openai_compatible")[provider] = block

with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
}

configure_opencode() {
  local file="$HOME/.config/opencode/opencode.jsonc"
  # OpenCode has no environment override for a provider, so the provider block
  # is written into its config. The provider is named "nvmai" so nothing of
  # the user's own "openai" setup is touched.
  merge_client_config "$file" opencode nvmai "$launch_model" "${max_context:-262144}"
  echo "  Pick \"$launch_model\" under the nvmai provider in OpenCode's model list."
}

configure_zed() {
  local file="$HOME/.config/zed/settings.json"
  merge_client_config "$file" zed nvmai "$launch_model" "${max_context:-262144}"
  echo "  In Zed: Settings -> AI -> LLM Providers; choose the nvmai provider's model."
}

case "$CLIENT" in
  server)   : ;;
  codex)    configure_codex ;;
  claude)   configure_claude ;;
  qwen)     configure_qwen ;;
  opencode) configure_opencode ;;
  zed)      configure_zed ;;
esac

# ============================================================
# 12) Hand over to the client
# ============================================================

# A server-only run stays in the foreground so Ctrl-C stops the model.
if [[ "$CLIENT" == "server" ]]; then
  wait "$server_pid"
  exit $?
fi

client_bin() {
  case "$1" in
    codex)    echo "${CODEX:-$(command -v codex 2>/dev/null || echo "$HOME/.local/bin/codex")}" ;;
    claude)   echo "${CLAUDE:-$(command -v claude 2>/dev/null || echo "$HOME/.local/bin/claude")}" ;;
    qwen)     echo "${QWEN:-$(command -v qwen-code 2>/dev/null || echo "$HOME/.qwen-code/bin/qwen-code")}" ;;
    opencode) echo "${OPENCODE:-$(command -v opencode 2>/dev/null || echo /opt/homebrew/bin/opencode)}" ;;
    zed)      echo "${ZED:-$(command -v zed 2>/dev/null || echo /usr/local/bin/zed)}" ;;
  esac
}

BIN="$(client_bin "$CLIENT")"
if [[ ! -x "$BIN" ]]; then
  echo "" >&2
  echo "WARNING: $(client_label "$CLIENT") was not found at $BIN." >&2
  echo "         The server is running on http://127.0.0.1:${PORT} — point your" >&2
  echo "         client at that address, or install it and re-run." >&2
  wait "$server_pid"
  exit $?
fi

echo "Launching $(client_label "$CLIENT")..."
echo ""

# The model keeps running after the client exits; the next launcher run stops
# it on this port and starts fresh. The trap keeps the two lifetimes matched
# when the person Ctrl-Cs the client instead.
cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

case "$CLIENT" in
  codex)    "$BIN" ;;
  claude)   "$BIN" ;;
  qwen)     "$BIN" -i ;;
  opencode) "$BIN" ;;
  zed)      "$BIN" "${ZED_PROJECT:-$PWD}" ;;
esac
