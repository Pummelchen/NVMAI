#!/usr/bin/env bash
# Runs the memory-value benchmarks arm by arm against Qwen 3.6 35B 4-bit.
#
#   benchmark/memval_run.sh smoke           # placement + one fact, ~3 minutes
#   benchmark/memval_run.sh pong            # control, minimal, full
#   benchmark/memval_run.sh book            # summary, minimal, full
#   benchmark/memval_run.sh pong full       # one arm
#
# Each arm gets a freshly started server with its own configuration and its
# own empty memory directory under the scratch root, so nothing an arm writes
# can reach another arm or the user's real ~/.nvmai/memory. The server is
# stopped between arms. Never build while this is running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${1:?pong|book}"
ONLY="${2:-}"
PORT="${NVMAI_PORT:-8096}"
SCRATCH="${NVMAI_MEMVAL_SCRATCH:-$ROOT/.build/benchmark-logs/memval-scratch}"
LOGS="$ROOT/.build/benchmark-logs/memory-$( [[ "$BENCH" == pong ]] && echo value || echo book )"
mkdir -p "$LOGS" "$SCRATCH"

case "$BENCH" in
  smoke) SCRIPT="$ROOT/benchmark/memory_smoke.py"; ARMS=(full) ;;
  pong)  SCRIPT="$ROOT/benchmark/memory_value.py"; ARMS=(control minimal full) ;;
  book)  SCRIPT="$ROOT/benchmark/memory_book.py";  ARMS=(summary minimal full) ;;
  *) echo "usage: $0 smoke|pong|book [arm]" >&2; exit 2 ;;
esac
[[ -n "$ONLY" ]] && ARMS=("$ONLY")

BINARY="$ROOT/.build/arm64-apple-macosx/release/NVMAIServer"
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: no release binary at $BINARY; run: swift build -c release --product NVMAIServer" >&2
  exit 1
fi
# The release binary must be newer than every source file, or the arms
# measure whatever was last built. This is the check that was missing when
# three arms of numbers turned out to be the same arm.
newest_source="$(find "$ROOT/sources" -name '*.swift' -newer "$BINARY" | head -1)"
if [[ -n "$newest_source" ]]; then
  echo "ERROR: $newest_source is newer than the release binary; rebuild first." >&2
  exit 1
fi

# The PIDs actually listening on the port -- the server binary, never the
# launcher shell around it. Killing the launcher leaves the model running:
# bash does not forward a signal to the child it is waiting on, and an
# orphaned server then answers the next run's readiness poll with the
# previous run's configuration. That is exactly what the first smoke test
# did, and its "empty replies in 0 s" were the orphan shutting down under
# the requests.
listening_pids() { lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null || true; }

stop_server() {
  local pid
  for pid in $(listening_pids); do
    ps -p "$pid" -o command= | grep -q NVMAIServer && kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 90); do
    [[ -z "$(listening_pids)" ]] && break
    sleep 1
  done
  for pid in $(listening_pids); do kill -KILL "$pid" 2>/dev/null || true; done
  [[ -n "${LAUNCHER_PID:-}" ]] && kill "$LAUNCHER_PID" 2>/dev/null || true
  LAUNCHER_PID=""
}

# Ready means a completion answers, not that the port is open. The port is
# bound before the first token can be produced, and a poll on /v1/models
# has no way to tell a loaded server from one that is still reading
# weights.
wait_ready() {
  local deadline=$(( $(date +%s) + 900 )) model
  until [[ -n "$(listening_pids)" ]]; do
    if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
      echo "ERROR: launcher exited before the server bound; see $SERVER_LOG" >&2; return 1
    fi
    (( $(date +%s) > deadline )) && { echo "ERROR: no server after 15 min; see $SERVER_LOG" >&2; return 1; }
    sleep 3
  done
  until model="$(curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" \
                 | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)" \
        && [[ -n "$model" ]]; do
    (( $(date +%s) > deadline )) && { echo "ERROR: /v1/models never answered" >&2; return 1; }
    sleep 3
  done
  local body status
  body="$(printf '{"model":"%s","messages":[{"role":"user","content":"Say OK."}],"max_completion_tokens":4,"temperature":0}' "$model")"
  until status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 600 \
                  -H 'Content-Type: application/json' -d "$body" \
                  "http://127.0.0.1:$PORT/v1/chat/completions")" && [[ "$status" == 200 ]]; do
    (( $(date +%s) > deadline )) && { echo "ERROR: server never answered a completion (last status $status)" >&2; return 1; }
    sleep 5
  done
  echo "ready: $model answers completions"
}

trap stop_server EXIT

for ARM in "${ARMS[@]}"; do
  case "$ARM" in
    control|summary) MEMORY=0; TOOLS=off ;;
    minimal)         MEMORY=1; TOOLS=minimal ;;
    full)            MEMORY=1; TOOLS=full ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
  esac
  MEMDIR="$SCRATCH/$BENCH-$ARM"
  rm -rf "$MEMDIR"; mkdir -p "$MEMDIR"
  SERVER_LOG="$LOGS/server-$ARM.log"

  echo "=== $BENCH / $ARM  (memory=$MEMORY tools=$TOOLS dir=$MEMDIR port=$PORT)"
  # Never let the launcher find a server to "stop": that path races the
  # readiness poll. The port is free before every arm, or the arm does not
  # start.
  stop_server
  if [[ -n "$(listening_pids)" ]]; then
    echo "ERROR: port $PORT is still held by $(listening_pids); refusing to start" >&2; exit 1
  fi
  (
    cd "$ROOT"
    NVMAI_PORT="$PORT" NVMAI_MEMORY="$MEMORY" NVMAI_MEMORY_TOOLS="$TOOLS" \
    NVMAI_MEMORY_DIR="$MEMDIR" NVMAI_MEMORY_JOURNAL=1 \
      exec tools/start-qwen3.6-4bit.sh codex full default off
  ) >"$SERVER_LOG" 2>&1 &
  LAUNCHER_PID=$!
  wait_ready
  grep -m1 "memory enabled" "$SERVER_LOG" || echo "(memory line: none, as expected for $ARM)"

  NVMAI_PORT="$PORT" NVMAI_MEMVAL_MEMDIR="$MEMDIR" python3 "$SCRIPT" "$ARM" 2>&1 | tee "$LOGS/run-$ARM.log"
  stop_server
  echo "--- journal files for $ARM:"; find "$MEMDIR" -name '*.ndjson' -exec ls -la {} \; 2>/dev/null || true
done

[[ "$BENCH" == smoke ]] || { echo; echo "=== report"; python3 "$SCRIPT" report; }
