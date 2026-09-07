#!/usr/bin/env bash
# Runs the memory benchmarks across every 35B install: three models, 4-bit
# and 8-bit, sequentially on one port. Qwen 3.6 4-bit gets all four arms,
# because it has a frozen v2 to compare against; every other install runs
# the baseline and the shipped configuration.
#
#   benchmark/memval_matrix.sh                  # everything, ~a day
#   NVMAI_MEMVAL_RUNS=1 benchmark/memval_matrix.sh   # a quick pass
#
# Never edit anything under benchmark/ or tools/ while this runs: bash reads
# scripts by offset, and a shifted file mid-run is how a report step once
# executed the letter "e".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/.build/benchmark-logs/memval-matrix.log"
mkdir -p "$(dirname "$LOG")"
echo "##### MATRIX START $(date)" | tee -a "$LOG"
for install in qwen36:4 qwen36:8 ornith:4 ornith:8 agentworld:4 agentworld:8; do
  model="${install%%:*}"; quant="${install##*:}"
  if [[ "$install" == "qwen36:4" ]]; then arms_book="summary auto minimal full"; arms_pong="control auto minimal full"
  else arms_book="summary auto"; arms_pong="control auto"; fi
  for bench in book pong; do
    arms="$arms_book"; [[ "$bench" == pong ]] && arms="$arms_pong"
    echo "##### $install $bench arms=[$arms] $(date)" | tee -a "$LOG"
    NVMAI_MEMVAL_MODEL="$model" NVMAI_MEMVAL_QUANT="$quant" NVMAI_MEMVAL_ARMS="$arms" \
      benchmark/memval_run.sh "$bench" 2>&1 | tee -a "$LOG" | tail -3
    echo "##### $install $bench DONE $(date) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
  done
done
echo "##### MATRIX DONE $(date)" | tee -a "$LOG"
