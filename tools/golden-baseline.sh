#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass". Phase 2 of the production audit needs this before RealForwardRunner
# is decomposed.
#
#   tools/golden-baseline.sh [target ...]     # default: ornith-8
#   tools/golden-baseline.sh --check [...]    # compare against the stored file
#   tools/golden-baseline.sh --server [...]   # CLI and server must agree
#
# Targets: ornith-4, ornith-8, qwen36-4, qwen36-8, qwen38-4, qwen38-8,
# agentworld-4, agentworld-8.
# Bare 4 and 8 still mean
# Ornith, because the stored baselines are named for it.
#
# Qwen3.8-Flash-Next is the broader gate: it is the only family exercising
# hyper-connections, the PLE block and the QSA indexer, and at 8-bit it is the
# only one exercising the affine GEMV path through them. Ornith reaches none
# of that, so an Ornith-only baseline can pass while those kernels are wrong.
#
# Determinism comes from greedy decoding: --temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels — exactly what a runtime refactor must not change.
#
# --server exists because the stored baselines only ever drove NVMAICLI, and a
# server binary 15 hours older than the runtime it links went unnoticed until
# it refused a promoted 8-bit model by hand. The two modes catch different
# things and neither substitutes for the other:
#
#   the stored file   the runtime changed        (CLI vs its own past output)
#   --server          the two front ends diverged (CLI vs server, same build)
#
# So --server stores nothing. It renders one message list through both paths at
# the same settings and requires the text to match, which needs no re-capture
# after a deliberate numerics change and cannot go stale. Comparing raw
# --prompt output against the server would be meaningless: the server always
# applies the chat template, so both sides are driven from --messages-file.
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M3 is not a reference for an M4. Re-capture after a deliberate numerics
# change; a diff at any other time is a regression.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$ROOT/.build/arm64-apple-macosx/release/NVMAICLI"
SERVER="$ROOT/.build/arm64-apple-macosx/release/NVMAIServer"
OUT_DIR="${OUT_DIR:-$ROOT/benchmark/golden}"

PROMPT="${PROMPT:-Explain what a mutex is and when you would use one.}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"
# Loading a 220 GB install off SSD is minutes, not seconds, and the readiness
# line is block-buffered when stdout is a file — so poll the port, not the log.
PORT="${PORT:-8757}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

mode=capture
server_mode=0
targets=()
for arg in "$@"; do
  case "$arg" in
    --check)  mode=check ;;
    --server) server_mode=1 ;;
    *)        targets+=("$arg") ;;
  esac
done
[ ${#targets[@]} -eq 0 ] && targets=(ornith-8)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI — run: swift build -c release" >&2
  exit 2
fi
if [ "$server_mode" = 1 ] && [ ! -x "$SERVER" ]; then
  echo "missing $SERVER — run: swift build -c release --product NVMAIServer" >&2
  exit 2
fi

# AGENTS.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
if pgrep -f 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null 2>&1; then
  echo "a model process is already running; stop it yourself, then re-run" >&2
  exit 3
fi

mkdir -p "$OUT_DIR" "$ROOT/.build"
status=0

# Only ever holds a PID this script started, which is what makes killing it in
# the trap compatible with the rule above.
srv_pid=""
cleanup() {
  if [ -n "$srv_pid" ] && kill -0 "$srv_pid" 2>/dev/null; then
    kill "$srv_pid" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$srv_pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$srv_pid" 2>/dev/null
  fi
  srv_pid=""
}
trap cleanup EXIT INT TERM

# Render the same message list through the server and through the CLI, and
# require the generated text to match.
server_check() {
  local model="$1" t="$2"
  local work msg cli_out srv_out srv_log body
  work="$(mktemp -d "$ROOT/.build/golden-server.XXXXXX")"
  msg="$work/messages.json"
  cli_out="$work/cli.txt"; srv_out="$work/server.txt"; srv_log="$work/server.log"

  python3 - "$msg" "$PROMPT" <<'PY'
import json, sys
json.dump([{"role": "user", "content": sys.argv[2]}], open(sys.argv[1], "w"))
PY

  "$CLI" --model "$model" --messages-file "$msg" --max-new "$MAX_NEW" \
         --temperature 0 --seed "$SEED" --quiet > "$cli_out" 2>"$work/cli.err"
  if [ $? -ne 0 ]; then
    echo "  server: FAILED — CLI leg exited nonzero"
    sed 's/^/    /' "$work/cli.err" | head -10
    rm -rf "$work"; return 1
  fi

  # Prompt caching is off so a rerun cannot answer from a cached prefix, which
  # would prove the cache works rather than that the paths agree.
  "$SERVER" --model "$model" --port "$PORT" --prompt-cache-mode off \
      > "$srv_log" 2>&1 &
  srv_pid=$!

  local ready=0 waited=0
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "  server: FAILED — exited during load"
      sed 's/^/    /' "$srv_log" | tail -5
      rm -rf "$work"; srv_pid=""; return 1
    fi
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
            "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = "200" ]; then
      ready=1; break
    fi
    sleep 5; waited=$(( waited + 5 ))
  done
  if [ "$ready" != 1 ]; then
    echo "  server: FAILED — not ready after ${READY_TIMEOUT}s"
    cleanup; rm -rf "$work"; return 1
  fi

  local model_id
  model_id="$(curl -s --max-time 30 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"

  body="$(python3 - "$model_id" "$PROMPT" "$MAX_NEW" "$SEED" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1],
                  "messages": [{"role": "user", "content": sys.argv[2]}],
                  "max_tokens": int(sys.argv[3]),
                  "seed": int(sys.argv[4]),
                  "temperature": 0}))
PY
)"
  curl -s --max-time 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$body" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"], end="")' \
    > "$srv_out" 2>"$work/srv.err"
  if [ ! -s "$srv_out" ]; then
    echo "  server: FAILED — no completion returned"
    sed 's/^/    /' "$work/srv.err" | head -10
    cleanup; rm -rf "$work"; return 1
  fi
  cleanup

  # The CLI writes the generated text to stdout; the server returns it inside a
  # JSON string. Whether a trailing newline survives that is a property of the
  # transport, not of the tokens, so normalise trailing whitespace per line and
  # trailing blank lines on both sides. Everything else still has to match
  # exactly -- this must fail when the two front ends really do disagree.
  norm() {
    python3 -c 'import sys; print("\n".join(l.rstrip() for l in sys.stdin.read().split("\n")).rstrip())' \
      < "$1"
  }
  if diff -q <(norm "$cli_out") <(norm "$srv_out") >/dev/null; then
    echo "  server: ok — CLI and server agree ($(wc -c < "$srv_out" | tr -d ' ') bytes)"
    rm -rf "$work"; return 0
  fi
  echo "  server: MISMATCH — CLI and server disagree on the same messages:"
  diff <(norm "$cli_out") <(norm "$srv_out") | head -30 | sed 's/^/    /'
  rm -rf "$work"; return 1
}

for t in "${targets[@]}"; do
  case "$t" in
    4|ornith-4) dir="ornith-1.5_35B_A3B_4Bit"; file="ornith-1.5-35b-a3b-4bit.txt"; q=4 ;;
    8|ornith-8) dir="ornith-1.5_35B_A3B_8Bit"; file="ornith-1.5-35b-a3b-8bit.txt"; q=8 ;;
    qwen38-4)   dir="qwen3.8-flash-next_125B_A6B_4Bit"
                file="qwen3.8-flash-next-125b-a6b-4bit.txt"; q=4 ;;
    qwen38-8)   dir="qwen3.8-flash-next_125B_A6B_8Bit"
                file="qwen3.8-flash-next-125b-a6b-8bit.txt"; q=8 ;;
    qwen36-4) dir="qwen3.6_35B_A3B_4Bit"; file="qwen3.6-35b-a3b-4bit.txt"; q=4 ;;
    qwen36-8) dir="qwen3.6_35B_A3B_8Bit"; file="qwen3.6-35b-a3b-8bit.txt"; q=8 ;;
    agentworld-4) dir="qwen-agentworld_35B_A3B_4Bit"; file="qwen-agentworld-35b-a3b-4bit.txt"; q=4 ;;
    agentworld-8) dir="qwen-agentworld_35B_A3B_8Bit"; file="qwen-agentworld-35b-a3b-8bit.txt"; q=8 ;;
    *) echo "unknown target: $t (ornith-4, ornith-8, qwen36-4, qwen36-8, qwen38-4, qwen38-8, agentworld-4, agentworld-8)" >&2
       status=1; continue ;;
  esac
  model="$ROOT/models/$dir"
  if [ ! -f "$model/verified-install.json" ]; then
    echo "missing baseline model $t: no verified install at ${model#$ROOT/}" >&2
    status=1
    continue
  fi
  file="$OUT_DIR/$file"
  work="$(mktemp "$ROOT/.build/golden-baseline.XXXXXX")"

  echo "== $t =="

  if [ "$server_mode" = 1 ]; then
    server_check "$model" "$t" || status=1
    rm -f "$work" "$work.err"
    continue
  fi

  # --quiet keeps the timing footer out of the compared text; only the
  # generated tokens are the contract. Timings vary run to run by design.
  "$CLI" --model "$model" --prompt "$PROMPT" --max-new "$MAX_NEW" \
         --temperature 0 --seed "$SEED" --quiet > "$work" 2>"$work.err"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED (exit $rc)"; sed 's/^/  /' "$work.err" | head -20
    rm -f "$work" "$work.err"; status=1; continue
  fi

  if [ "$mode" = capture ]; then
    {
      echo "# prompt:      $PROMPT"
      echo "# max-new:     $MAX_NEW"
      echo "# temperature: 0 (greedy)"
      echo "# seed:        $SEED"
      echo "# target:      $t (${q}-bit)"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD)"
      echo "---"
      cat "$work"
    } > "$file"
    echo "  captured -> ${file#$ROOT/} ($(wc -c < "$work" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at ${file#$ROOT/}; run without --check first"; status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok — output identical to baseline"
    else
      echo "  MISMATCH against ${file#$ROOT/}:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
