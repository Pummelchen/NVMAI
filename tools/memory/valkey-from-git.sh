#!/usr/bin/env bash
# Build and run Valkey from source for NVMAI's agent memory.
#
#   tools/memory/valkey-from-git.sh build     clone and compile
#   tools/memory/valkey-from-git.sh start     run it on loopback
#   tools/memory/valkey-from-git.sh stop
#   tools/memory/valkey-from-git.sh status
#
# From git rather than a package manager or a container: the memory store is
# a dependency of the engine's behaviour, so it is pinned to a known tag and
# built the same way on macOS and Linux, with no daemon or brew tap in
# between. Everything lives under .build/valkey, so removing that directory
# removes it entirely.
#
# Data is RAM-first with persistence: an appendonly log plus periodic
# snapshots, `noeviction`, bound to loopback. NVMAI applies its own memory
# ceiling at connect (NVMAI_MEMORY_CACHE_MIB), so the one here only has to be
# at least as large.
set -euo pipefail

VERSION="${VALKEY_VERSION:-8.1}"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$BASE_DIR/.build/valkey"
SRC="$ROOT/src-$VERSION"
DATA="$ROOT/data"
PIDFILE="$ROOT/valkey.pid"
LOGFILE="$ROOT/valkey.log"
PORT="${VALKEY_PORT:-6379}"
MAXMEMORY="${VALKEY_MAXMEMORY:-1gb}"

server_binary() { echo "$SRC/src/valkey-server"; }
cli_binary() { echo "$SRC/src/valkey-cli"; }

build() {
  if [[ -x "$(server_binary)" ]]; then
    echo "Valkey $VERSION already built at $(server_binary)"
    return 0
  fi
  mkdir -p "$ROOT"
  if [[ ! -d "$SRC/.git" ]]; then
    echo "Cloning Valkey $VERSION..."
    rm -rf "$SRC"
    git clone --depth 1 --branch "$VERSION" https://github.com/valkey-io/valkey.git "$SRC"
  fi
  echo "Building (this takes a couple of minutes)..."
  make -C "$SRC" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
  echo "Built $(server_binary)"
}

start() {
  build
  if status >/dev/null 2>&1; then
    echo "Already running on port $PORT"
    return 0
  fi
  mkdir -p "$DATA"
  "$(server_binary)" \
    --port "$PORT" \
    --bind 127.0.0.1 \
    --appendonly yes \
    --appendfsync everysec \
    --save 300 10 \
    --maxmemory "$MAXMEMORY" \
    --maxmemory-policy noeviction \
    --dir "$DATA" \
    --pidfile "$PIDFILE" \
    --logfile "$LOGFILE" \
    --daemonize yes
  for _ in $(seq 1 50); do
    if "$(cli_binary)" -h 127.0.0.1 -p "$PORT" ping >/dev/null 2>&1; then
      echo "Valkey $VERSION listening on 127.0.0.1:$PORT (data in $DATA)"
      return 0
    fi
    sleep 0.2
  done
  echo "ERROR: Valkey did not come up; see $LOGFILE" >&2
  return 1
}

stop() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "Stopped"
  else
    echo "Not running"
  fi
}

status() {
  "$(cli_binary)" -h 127.0.0.1 -p "$PORT" ping >/dev/null 2>&1 || {
    echo "not running on port $PORT"
    return 1
  }
  local version keys
  version="$("$(cli_binary)" -p "$PORT" info server | awk -F: '/valkey_version/ {print $2}' | tr -d '\r')"
  keys="$("$(cli_binary)" -p "$PORT" dbsize | tr -d '\r')"
  echo "running: Valkey $version on 127.0.0.1:$PORT, $keys keys"
}

case "${1:-status}" in
  build) build ;;
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 [build|start|stop|status]" >&2; exit 2 ;;
esac
