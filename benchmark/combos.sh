#!/usr/bin/env bash
# Run all 72 launcher combinations, ordered by expected wall time from
# fastest to slowest (fastest server configs first; combos sorted within each
# config). The server is restarted only when (quant, mode, reasoning)
# changes - a per-combo reorder would pay a model-load restart per run.
# Resumable: skips combos with an existing answer file.
# Results go to benchmark/benchmark-results/ (git-ignored).
set -uo pipefail

NVMAI="${NVMAI:-/Users/andreborchert/Downloads/NVMAI}"
PROMPT="difference of swift and c++ in detail"
OUT="${OUT:-$NVMAI/benchmark/benchmark-results}"
RESULTS=$OUT/results.tsv
CODEX_BIN="${CODEX_BIN:-$HOME/.local/bin/codex}"
QWEN_BIN="${QWEN_BIN:-$HOME/.qwen-code/bin/qwen-code}"
OPENCODE_BIN="${OPENCODE_BIN:-/opt/homebrew/bin/opencode}"
# LIMIT=N runs only the first N combos (fastest-first). Unset/empty = all 72.
LIMIT="${LIMIT:-}"
mkdir -p "$OUT/answers"
if [[ ! -s "$RESULTS" ]]; then
  echo -e "quant\tmode\treasoning\tcli\tmodel\twall_s\tprompt_tok\tcompletion_tok\tdecode_tok_s\tanswer_file" > "$RESULTS"
fi

stop_server() {
  # Kill only NVMAIServer processes holding the benchmark ports — never an
  # unrelated NVMAIServer elsewhere (AGENTS.md: never terminate an existing
  # server you did not launch).
  for port in 8081 8082 8083; do
    for pid in $(lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null); do
      if ps -p "$pid" -o command= | grep -q NVMAIServer; then
        kill "$pid" 2>/dev/null
      fi
    done
  done
  # Wait until the NVMAI ports actually free so the next readiness poll can
  # never be satisfied by a stale server mid-teardown (bounded 10s).
  for _ in $(seq 1 100); do
    if ! lsof -i :8081 -sTCP:LISTEN >/dev/null 2>&1 \
       && ! lsof -i :8082 -sTCP:LISTEN >/dev/null 2>&1 \
       && ! lsof -i :8083 -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
}
trap stop_server EXIT

start_server() {
  local quant="$1" mode="$2" reasoning="$3" tag="$4"
  local port
  case "$quant" in 4) port=8081 ;; 6) port=8082 ;; 8) port=8083 ;; esac
  local mode_word=default
  [[ "$mode" == "_concise" ]] && mode_word=concise
  local think_word=nothink
  [[ "$reasoning" == "1" ]] && think_word=think
  stop_server
  nohup "$NVMAI/tools/server_launcher.sh" codex full "$quant" "$mode_word" "$think_word" > "$OUT/server_${tag}.log" 2>&1 &
  local up=0
  for _ in $(seq 1 90); do
    curl -s --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { up=1; break; }
    sleep 3
  done
  [[ "$up" == "1" ]]
}

run_one() {
  local quant="$1" mode="$2" reasoning="$3" cli="$4" model="$5"
  local tag="q${quant}_m${mode}_r${reasoning}_${cli}_${model}"
  local outfile="$OUT/answers/${tag}.txt"
  local mode_word=default
  [[ "$mode" == "_concise" ]] && mode_word=concise
  if [[ -s "$outfile" ]]; then echo "SKIP $tag"; return; fi
  local port
  case "$quant" in 4) port=8081 ;; 6) port=8082 ;; 8) port=8083 ;; esac
  local base="http://127.0.0.1:${port}/v1"
  local api_model
  if [[ "$model" == "full" ]]; then api_model="qwen3.6-35b-a3b"; else api_model="qwen3.6-35b-a3b-fast"; fi

  echo "RUN $tag"
  local log_start
  log_start=$(wc -l < "$OUT/server_${srvtag}.log" 2>/dev/null || echo 0)
  mkdir -p "$HOME/.codex-nvmai"
  cat > "$HOME/.codex-nvmai/config.toml" <<EOF2
model = "$api_model"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "$base"
wire_api = "responses"
EOF2
  mkdir -p "$HOME/.qwen-nvmai"
  cat > "$HOME/.qwen-nvmai/settings.json" <<EOF2
{
  "modelProviders": {
    "openai": [
      {
        "id": "$api_model",
        "name": "[NVMAI] $api_model",
        "baseUrl": "$base",
        "description": "NVMAI local server",
        "envKey": "OPENAI_API_KEY"
      }
    ]
  },
  "security": { "auth": { "selectedType": "openai" } },
  "model": { "name": "$api_model" },
  "memory": {
    "enableManagedAutoMemory": false,
    "enableManagedAutoDream": false,
    "enableAutoSkill": false
  }
}
EOF2
  local oc_home="$OUT/opencode-config-$tag"
  if [[ -f "$HOME/.config/opencode/opencode.jsonc" ]]; then
    mkdir -p "$oc_home/opencode"
    cp "$HOME/.config/opencode/opencode.jsonc" "$oc_home/opencode/opencode.jsonc"
    sed -i.bak "s|http://127.0.0.1:[0-9]*/v1|http://127.0.0.1:${port}/v1|" "$oc_home/opencode/opencode.jsonc"
  fi

  local wall
  case "$cli" in
    codex)
      /usr/bin/time -p env CODEX_HOME="$HOME/.codex-nvmai" OPENAI_API_KEY=dummy \
        "$CODEX_BIN" exec --skip-git-repo-check "$PROMPT" \
        > "$outfile" 2>"$OUT/time_${tag}.txt"
      ;;
    qwen)
      /usr/bin/time -p env QWEN_HOME="$HOME/.qwen-nvmai" OPENAI_API_KEY=dummy \
        QWEN_STREAM_IDLE_TIMEOUT_MS=0 QWEN_STREAM_MAX_LIFETIME_MS=0 \
        "$QWEN_BIN" -p "$PROMPT" \
        > "$outfile" 2>"$OUT/time_${tag}.txt"
      ;;
    opencode)
      /usr/bin/time -p env XDG_CONFIG_HOME="$oc_home" OPENAI_API_KEY=dummy \
        "$OPENCODE_BIN" run -m "openai/$api_model" "$PROMPT" \
        > "$outfile" 2>"$OUT/time_${tag}.txt"
      ;;
  esac
  local cli_status=$?
  wall=$(grep '^real' "$OUT/time_${tag}.txt" 2>/dev/null | awk '{print $2}')
  wall="${wall:-0}"
  if [[ "$cli_status" -ne 0 || "$wall" == "0" ]]; then
    echo "  FAILED ($tag): CLI exited $cli_status, no timing captured"
    echo -e "$quant\t${mode_word}\t$reasoning\t$cli\t$model\tFAILED\t0\t0\t0\t$tag" >> "$RESULTS"
    return
  fi

  local stats ptok ctok dtoks
  # take the first completed line logged after this combo started (log_start+1)
  stats=$(tail -n +$((log_start + 1)) "$OUT/server_${srvtag}.log" 2>/dev/null \
    | grep "completed" | head -1)
  ptok="${stats#*prompt=}"; ptok="${ptok%% *}"; ptok="${ptok:-0}"
  ctok="${stats#*completion=}"; ctok="${ctok%% *}"; ctok="${ctok:-0}"
  dtoks=$(python3 -c "print(f'{$ctok/$wall:.1f}')" 2>/dev/null || echo 0)
  echo -e "$quant\t${mode_word}\t$reasoning\t$cli\t$model\t$wall\t$ptok\t$ctok\t$dtoks\t$tag" >> "$RESULTS"
  echo "  done: ${wall}s, ${ctok} completion tok (${dtoks} tok/s)"
}

# 72 combos ordered fastest-first (expected wall time, calibrated on the
# observed runs); grouped by server config so the server is only restarted
# when (quant, mode, reasoning) changes. Mode uses words (default/concise)
# so no field can be empty when parsed with read.
ORDER=(
  # q4_m_concise_r0 total   46.9 min
  "4 concise 0 opencode full"
  "4 concise 0 opencode fast"
  "4 concise 0 qwen fast"
  "4 concise 0 codex fast"
  "4 concise 0 codex full"
  "4 concise 0 qwen full"
  # q4_m_concise_r1 total   51.0 min
  "4 concise 1 opencode full"
  "4 concise 1 opencode fast"
  "4 concise 1 qwen fast"
  "4 concise 1 codex full"
  "4 concise 1 codex fast"
  "4 concise 1 qwen full"
  # q4_m_r0         total   69.4 min
  "4 default 0 opencode full"
  "4 default 0 codex full"
  "4 default 0 opencode fast"
  "4 default 0 qwen fast"
  "4 default 0 codex fast"
  "4 default 0 qwen full"
  # q6_m_concise_r0 total   70.9 min
  "6 concise 0 opencode full"
  "6 concise 0 opencode fast"
  "6 concise 0 qwen fast"
  "6 concise 0 codex fast"
  "6 concise 0 codex full"
  "6 concise 0 qwen full"
  # q6_m_concise_r1 total   78.0 min
  "6 concise 1 opencode full"
  "6 concise 1 opencode fast"
  "6 concise 1 codex full"
  "6 concise 1 qwen fast"
  "6 concise 1 codex fast"
  "6 concise 1 qwen full"
  # q4_m_r1         total   79.1 min
  "4 default 1 opencode full"
  "4 default 1 codex full"
  "4 default 1 opencode fast"
  "4 default 1 codex fast"
  "4 default 1 qwen fast"
  "4 default 1 qwen full"
  # q8_m_concise_r0 total   88.3 min
  "8 concise 0 opencode full"
  "8 concise 0 opencode fast"
  "8 concise 0 qwen fast"
  "8 concise 0 codex full"
  "8 concise 0 codex fast"
  "8 concise 0 qwen full"
  # q8_m_concise_r1 total   97.8 min
  "8 concise 1 opencode full"
  "8 concise 1 opencode fast"
  "8 concise 1 codex full"
  "8 concise 1 qwen fast"
  "8 concise 1 codex fast"
  "8 concise 1 qwen full"
  # q6_m_r0         total  110.3 min
  "6 default 0 opencode full"
  "6 default 0 codex full"
  "6 default 0 opencode fast"
  "6 default 0 codex fast"
  "6 default 0 qwen fast"
  "6 default 0 qwen full"
  # q6_m_r1         total  127.3 min
  "6 default 1 opencode full"
  "6 default 1 codex full"
  "6 default 1 opencode fast"
  "6 default 1 codex fast"
  "6 default 1 qwen fast"
  "6 default 1 qwen full"
  # q8_m_r0         total  140.8 min
  "8 default 0 opencode full"
  "8 default 0 codex full"
  "8 default 0 opencode fast"
  "8 default 0 codex fast"
  "8 default 0 qwen fast"
  "8 default 0 qwen full"
  # q8_m_r1         total  163.4 min
  "8 default 1 opencode full"
  "8 default 1 codex full"
  "8 default 1 opencode fast"
  "8 default 1 codex fast"
  "8 default 1 qwen fast"
  "8 default 1 qwen full"
)

prev_cfg=""
failed_cfg=""
runs=0
for combo in "${ORDER[@]}"; do
  read -r quant mode_word reasoning cli model <<< "$combo"
  mode=""
  [[ "$mode_word" == "concise" ]] && mode="_concise"
  cfg="q${quant}_m${mode}_r${reasoning}"
  if [[ "$cfg" == "$failed_cfg" ]]; then continue; fi
  if [[ "$cfg" != "$prev_cfg" ]]; then
    srvtag="$cfg"
    if start_server "$quant" "$mode" "$reasoning" "$srvtag"; then
      prev_cfg="$cfg"
    else
      echo "ERROR: server failed for $cfg" >&2
      failed_cfg="$cfg"
      continue
    fi
  fi
  run_one "$quant" "$mode" "$reasoning" "$cli" "$model"
  runs=$((runs + 1))
  if [[ -n "$LIMIT" && "$runs" -ge "$LIMIT" ]]; then
    echo "Reached LIMIT=$LIMIT runs; stopping."
    break
  fi
done
stop_server
echo "ALL DONE"
