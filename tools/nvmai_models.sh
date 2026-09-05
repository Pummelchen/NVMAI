#!/usr/bin/env bash
# Shared model catalogue for every launcher and start script.
#
# One place decides which installs exist, what they are called, where they
# live and which port each serves on, so the server launcher, the CLI
# launcher and the eight per-model start scripts cannot drift apart. A
# launcher that resolved the model itself and a start script that hardcoded
# a port is exactly how the "-fast" alias and the width-suffixed model id
# got out of step before.
#
# Ports are distinct per (model, quantization) so two configurations can be
# up at once without colliding. Ornith keeps 8081 / 8083, the ports every
# existing client config points at.

# nvmai_resolve_model <key> -> NVMAI_MODEL_{KEY,STEM,LABEL}, NVMAI_PORT_{4,8}
nvmai_resolve_model() {
  case "${1:-}" in
    ornith|ornith15|ornith1.5)
      NVMAI_MODEL_KEY=ornith
      NVMAI_MODEL_STEM="ornith-1.5_35B_A3B"
      NVMAI_MODEL_LABEL="Ornith 1.5 35B-A3B"
      NVMAI_PORT_4=8081; NVMAI_PORT_8=8083 ;;
    qwen36|qwen3.6)
      NVMAI_MODEL_KEY=qwen36
      NVMAI_MODEL_STEM="qwen3.6_35B_A3B"
      NVMAI_MODEL_LABEL="Qwen 3.6 35B-A3B"
      NVMAI_PORT_4=8085; NVMAI_PORT_8=8086 ;;
    agentworld|aw)
      NVMAI_MODEL_KEY=agentworld
      NVMAI_MODEL_STEM="qwen-agentworld_35B_A3B"
      NVMAI_MODEL_LABEL="Qwen-AgentWorld 35B-A3B"
      NVMAI_PORT_4=8087; NVMAI_PORT_8=8088 ;;
    qwen38|qwen3.8)
      NVMAI_MODEL_KEY=qwen38
      NVMAI_MODEL_STEM="qwen3.8-flash-next_125B_A6B"
      NVMAI_MODEL_LABEL="Qwen3.8-Flash-Next 125B-A6B"
      NVMAI_PORT_4=8089; NVMAI_PORT_8=8090 ;;
    *)
      echo "unknown AI model: ${1:-} (ornith|qwen36|agentworld|qwen38)" >&2
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

# nvmai_model_port -> echoes the port for the resolved model + quantization.
nvmai_model_port() {
  if [[ "$NVMAI_QUANT" == 4bit ]]; then echo "$NVMAI_PORT_4"; else echo "$NVMAI_PORT_8"; fi
}

# Every (model, quantization) this checkout knows about, for help text.
NVMAI_ALL_MODELS=(ornith qwen36 agentworld qwen38)

# --- Persistent memory -------------------------------------------------
#
# Off unless NVMAI_MEMORY=1. Memory runs inside the server process, so there
# is no database to install or start. The store ceiling defaults by machine
# memory, since the working set is a few thousand short facts and does not
# grow with the host: 256 MiB at 8 GB, 512 MiB at 16 GB, 1 GiB above that.
# Override with NVMAI_MEMORY_CACHE_MIB, and the location with
# NVMAI_MEMORY_DIR (default ~/.nvmai/memory).
#
# This RAM is ADDITIONAL. It is not taken out of --ram-budget: an 8 GB
# machine runs the expert cache at its 4 GiB and memory brings the total to
# 4 GiB + 256 MiB. The ceiling covers every open workspace together, not
# each one, so turning memory on costs the same whether a session touches
# one repository or five.
#
# The workspace defaults to the directory the launcher was run from, which
# is the repository being worked on, so two checkouts never share memory.

nvmai_default_cache_mib() {
  local bytes
  bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
  if [[ -z "$bytes" ]]; then
    # Linux: MemTotal is in kB.
    bytes="$(awk '/MemTotal/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || true)"
  fi
  if [[ -z "$bytes" ]]; then echo 512; return; fi
  local gib=$(( bytes / 1073741824 ))
  if   (( gib <= 8 ));  then echo 256
  elif (( gib <= 16 )); then echo 512
  else                       echo 1024
  fi
}

# Exports the memory environment a server inherits. Everything already set
# by the caller wins, so a start script never overrides an explicit choice.
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
  export NVMAI_MEMORY_CACHE_MIB="${NVMAI_MEMORY_CACHE_MIB:-$(nvmai_default_cache_mib)}"
  export NVMAI_WORKSPACE_DIR="${NVMAI_WORKSPACE_DIR:-$workspace_dir}"
  export NVMAI_MEMORY_NAMESPACE="${NVMAI_MEMORY_NAMESPACE:-nvmai}"
  export NVMAI_MEMORY_DIR="${NVMAI_MEMORY_DIR:-$HOME/.nvmai/memory}"
}
