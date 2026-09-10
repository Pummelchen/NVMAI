#!/usr/bin/env bash
# Shared model catalogue for every launcher and start script.
#
# One place decides which installs exist, what they are called and where
# they live, so the server launcher, the CLI launcher and the eight
# per-model start scripts cannot drift apart. A launcher that resolved the
# model itself and a start script that hardcoded a port is exactly how the
# "-fast" alias and the width-suffixed model id got out of step before.
#
# Every model serves on one port. The server switches models on demand by
# name, one resident at a time, so a port per install only left client
# configs to keep in step. NVMAI_PORT still overrides it: the memory
# harness runs its own server on 8096 that way.
NVMAI_DEFAULT_PORT=8080

# nvmai_resolve_model <key> -> NVMAI_MODEL_{KEY,STEM,LABEL}
nvmai_resolve_model() {
  case "${1:-}" in
    ornith|ornith15|ornith1.5)
      NVMAI_MODEL_KEY=ornith
      NVMAI_MODEL_STEM="ornith-1.5_35B_A3B"
      NVMAI_MODEL_LABEL="Ornith 1.5 35B-A3B" ;;
    qwen36|qwen3.6)
      NVMAI_MODEL_KEY=qwen36
      NVMAI_MODEL_STEM="qwen3.6_35B_A3B"
      NVMAI_MODEL_LABEL="Qwen 3.6 35B-A3B" ;;
    agentworld|aw)
      NVMAI_MODEL_KEY=agentworld
      NVMAI_MODEL_STEM="qwen-agentworld_35B_A3B"
      NVMAI_MODEL_LABEL="Qwen-AgentWorld 35B-A3B" ;;
    qwen38|qwen3.8)
      NVMAI_MODEL_KEY=qwen38
      NVMAI_MODEL_STEM="qwen3.8-flash-next_125B_A6B"
      NVMAI_MODEL_LABEL="Qwen3.8-Flash-Next 125B-A6B" ;;
    # The CPU models. They exist only in the server's catalog, so these
    # resolve there and never in the built-in GPU list below.
    qwen35-2b|qwen3.5-2b)
      NVMAI_MODEL_KEY=qwen35-2b
      NVMAI_MODEL_STEM="qwen3.5_2B"
      NVMAI_MODEL_LABEL="Qwen 3.5 2B" ;;
    qwen35-4b|qwen3.5-4b)
      NVMAI_MODEL_KEY=qwen35-4b
      NVMAI_MODEL_STEM="qwen3.5_4B"
      NVMAI_MODEL_LABEL="Qwen 3.5 4B" ;;
    *)
      echo "unknown AI model: ${1:-} (ornith|qwen36|agentworld|qwen38|qwen35-2b|qwen35-4b)" >&2
      return 2 ;;
  esac
}

# nvmai_resolve_quant <4|8|4bit|8bit> -> NVMAI_QUANT ("4bit"/"8bit"), NVMAI_QUANT_DIR ("4Bit"/"8Bit")
nvmai_resolve_quant() {
  case "${1:-}" in
    4|4bit) NVMAI_QUANT=4bit; NVMAI_QUANT_DIR=4Bit ;;
    8|8bit) NVMAI_QUANT=8bit; NVMAI_QUANT_DIR=8Bit ;;
    *) echo "unknown quantization: ${1:-} (4|8)" >&2; return 2 ;;
  esac
}

# nvmai_model_port -> the one port. Kept under this name because
# tools/cli_launcher.sh still asks for it.
nvmai_model_port() {
  echo "$NVMAI_DEFAULT_PORT"
}

# Every (model, quantization) this checkout knows about, for help text and
# for the fallback list when the server cannot report its catalog.
NVMAI_ALL_MODELS=(ornith qwen36 agentworld qwen38)

# --- Installed models, from the server's catalog -----------------------
#
# The server knows what is installed -- GPU installs and CPU snapshots,
# their ids, widths and the thinking levels each one supports -- so the
# launcher asks it instead of keeping a second list that goes stale:
# `NVMAIServer --catalog --models-dir <dir>` prints JSON. NVMAI_CATALOG_JSON
# reads the same JSON from a file instead, so the launcher can be exercised
# against tools/testdata/catalog-example.json without a build.
#
# nvmai_load_catalog <binary> <models-dir> fills parallel arrays, one entry
# per model and quantization, GPU entries first:
#   NVMAI_CAT_ID  NVMAI_CAT_NAME  NVMAI_CAT_QUANT (4|8)  NVMAI_CAT_BACKEND (gpu|cpu)
#   NVMAI_CAT_PATH  NVMAI_CAT_THINKING (comma-separated, off first)  NVMAI_CAT_SIZE (GB, or -)
# and returns 1 with the reason in NVMAI_CATALOG_ERROR when there is no
# usable catalog.

# Levels in the order they are offered; anything newer the server adds is
# kept, after these.
NVMAI_CATALOG_PARSER='
import json, sys
LEVELS = ["off", "on", "minimal", "low", "medium", "high", "xhigh", "max"]
def field(value):
    text = str(value)
    if not text or any(c in text for c in "\t\r\n"):
        raise ValueError("empty or multi-line field")
    return text
try:
    models = json.load(sys.stdin)["models"]
except (ValueError, KeyError, TypeError) as error:
    sys.exit("catalog: not the expected JSON (%s)" % error)
if not isinstance(models, list):
    sys.exit("catalog: \"models\" is not a list")
gpu, cpu = [], []
for model in models:
    try:
        backend = field(model["backend"])
        if backend not in ("gpu", "cpu"):
            raise ValueError("backend %r" % backend)
        listed = [str(level) for level in (model.get("thinking") or ["off"])]
        levels = [l for l in LEVELS if l in listed] + [l for l in listed if l not in LEVELS]
        size = model.get("size_gb")
        row = [field(model["id"]), field(model["name"]), field(int(model["quant"])),
               backend, field(model["path"]), field(",".join(levels)),
               "%.1f" % float(size) if size is not None else "-"]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        name = model.get("id") if isinstance(model, dict) else model
        print("catalog: skipping %r (%s)" % (name, error), file=sys.stderr)
        continue
    (gpu if backend == "gpu" else cpu).append(row)
for row in gpu + cpu:
    print("\t".join(row))
'

nvmai_reset_catalog() {
  NVMAI_CAT_ID=(); NVMAI_CAT_NAME=(); NVMAI_CAT_QUANT=(); NVMAI_CAT_BACKEND=()
  NVMAI_CAT_PATH=(); NVMAI_CAT_THINKING=(); NVMAI_CAT_SIZE=()
}

nvmai_load_catalog() {
  local binary="$1" models_dir="$2" json rows
  local id name quant backend path thinking size
  NVMAI_CATALOG_ERROR=""
  nvmai_reset_catalog
  if [[ -n "${NVMAI_CATALOG_JSON:-}" ]]; then
    if ! json="$(cat "$NVMAI_CATALOG_JSON" 2>/dev/null)"; then
      NVMAI_CATALOG_ERROR="cannot read NVMAI_CATALOG_JSON=$NVMAI_CATALOG_JSON"
      return 1
    fi
  elif [[ ! -x "$binary" ]]; then
    NVMAI_CATALOG_ERROR="no server binary at $binary"
    return 1
  elif ! json="$("$binary" --catalog --models-dir "$models_dir" 2>/dev/null)"; then
    NVMAI_CATALOG_ERROR="$(basename "$binary") --catalog failed; this build may predate it"
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    NVMAI_CATALOG_ERROR="python3 is needed to read it"
    return 1
  fi
  if ! rows="$(printf '%s' "$json" | python3 -c "$NVMAI_CATALOG_PARSER")"; then
    NVMAI_CATALOG_ERROR="its output could not be read"
    return 1
  fi
  # Tab-separated with no empty fields (the parser writes "-"), because
  # read collapses runs of tabs and an empty column would shift the rest.
  while IFS=$'\t' read -r id name quant backend path thinking size; do
    [[ -n "$id" ]] || continue
    NVMAI_CAT_ID+=("$id"); NVMAI_CAT_NAME+=("$name"); NVMAI_CAT_QUANT+=("$quant")
    NVMAI_CAT_BACKEND+=("$backend"); NVMAI_CAT_PATH+=("$path")
    NVMAI_CAT_THINKING+=("$thinking"); NVMAI_CAT_SIZE+=("$size")
  done <<< "$rows"
  if (( ${#NVMAI_CAT_ID[@]} == 0 )); then
    NVMAI_CATALOG_ERROR="it lists no installed models"
    return 1
  fi
}

# nvmai_static_catalog <models-dir>: the same arrays from the built-in
# list, for a server that cannot report its catalog. GPU installs only,
# thinking off/on, and no ids: such a server reports its one id once up.
nvmai_static_catalog() {
  local models_dir="$1" key bits
  nvmai_reset_catalog
  for key in "${NVMAI_ALL_MODELS[@]}"; do
    nvmai_resolve_model "$key"
    for bits in 8 4; do
      nvmai_resolve_quant "$bits"
      NVMAI_CAT_ID+=("-"); NVMAI_CAT_NAME+=("$NVMAI_MODEL_LABEL"); NVMAI_CAT_QUANT+=("$bits")
      NVMAI_CAT_BACKEND+=(gpu); NVMAI_CAT_PATH+=("$models_dir/${NVMAI_MODEL_STEM}_${NVMAI_QUANT_DIR}")
      NVMAI_CAT_THINKING+=("off,on"); NVMAI_CAT_SIZE+=("-")
    done
  done
}

# nvmai_catalog_find_id <id> -> echoes the index of the entry with that id.
nvmai_catalog_find_id() {
  local i
  for (( i = 0; i < ${#NVMAI_CAT_ID[@]}; i++ )); do
    if [[ "${NVMAI_CAT_ID[$i]}" != "-" && "${NVMAI_CAT_ID[$i]}" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# nvmai_catalog_find_dir <install directory name> -> echoes the index of the
# entry installed there. This is how an install key and a width (ornith 8)
# find their entry: the directory name is fixed by the installer, while
# the id is the server's to choose.
nvmai_catalog_find_dir() {
  local i
  for (( i = 0; i < ${#NVMAI_CAT_PATH[@]}; i++ )); do
    if [[ "$(basename "${NVMAI_CAT_PATH[$i]}")" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# --- Persistent memory -------------------------------------------------
#
# Off unless NVMAI_MEMORY=1. Memory runs inside the server process, so there
# is no database to install or start, and it takes what it needs: measured,
# a hundred-chapter novel over ten sessions was about 100 KB. There is no
# RAM ceiling by default; NVMAI_MEMORY_CACHE_MIB sets one for anyone who
# wants it. The files live in a dedicated folder under the checkout,
# <NVMAI>/memory, beside models/ (override with NVMAI_MEMORY_DIR). One file
# per project; a project untouched for 30 days is deleted, and at most 100
# are kept (NVMAI_MEMORY_RETENTION_DAYS, NVMAI_MEMORY_MAX_WORKSPACES).
#
# The workspace defaults to the directory the launcher was run from, which
# is the repository being worked on, so two checkouts never share memory.

nvmai_export_memory_environment() {
  local workspace_dir="${1:-$PWD}"
  export NVMAI_MEMORY="${NVMAI_MEMORY:-0}"
  [[ "$NVMAI_MEMORY" == "1" ]] || return 0
  # The home directory, its parent and the root are not projects. A server
  # launched from one and used for everything would put a novel and a
  # codebase in one fact store, so this refuses to start rather than mix.
  # The server applies the same rule on its own; this just says it earlier
  # and in the terminal the person is looking at.
  if [[ -z "${NVMAI_MEMORY_WORKSPACE:-}" ]]; then
    local resolved home_dir
    resolved="$(cd "$workspace_dir" 2>/dev/null && pwd -P)"
    home_dir="$(cd "$HOME" 2>/dev/null && pwd -P)"
    if [[ "$resolved" == "$home_dir" || "$resolved" == "$(dirname "$home_dir")" || "$resolved" == "/" ]]; then
      echo "ERROR: NVMAI_MEMORY=1 but this is launched from $resolved, which is not a project directory." >&2
      echo "       Memory would collect every project into one store. Either:" >&2
      echo "         cd <your project>   and run the start script again" >&2
      echo "       or name the workspace explicitly:" >&2
      echo "         NVMAI_MEMORY_WORKSPACE=my-project <start script>" >&2
      exit 2
    fi
  fi
  export NVMAI_WORKSPACE_DIR="${NVMAI_WORKSPACE_DIR:-$workspace_dir}"
  export NVMAI_MEMORY_NAMESPACE="${NVMAI_MEMORY_NAMESPACE:-nvmai}"
  local nvmai_root
  nvmai_root="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  export NVMAI_MEMORY_DIR="${NVMAI_MEMORY_DIR:-$nvmai_root/memory}"
}
