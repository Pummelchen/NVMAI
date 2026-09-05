#!/usr/bin/env bash
# Start NVMAI serving Qwen 3.6 35B-A3B at 4-bit, no questions asked.
#
# The runtime's own tuning for this install (expert-cache budget and slot
# count, prefetch depth, prefill chunk, sampling defaults) comes from its
# ModelProfile row and is deliberately not set here.
#
#   tools/start-qwen3.6-4bit.sh [codex|qwen|opencode] [full|fast] [default|concise] [off|on]
#
# Defaults: codex, full, standard, thinking off. Override the port with
# NVMAI_PORT; the interactive chooser is tools/server_launcher.sh.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/server_launcher.sh" \
  "${1:-codex}" "${2:-full}" qwen36 4 "${3:-default}" "${4:-${NVMAI_THINKING_MODE:-off}}"
