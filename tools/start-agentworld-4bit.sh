#!/usr/bin/env bash
# Start NVMAI serving Qwen-AgentWorld 35B-A3B at 4-bit, no questions asked.
#
# The runtime's own tuning for this install (expert-cache budget and slot
# count, prefetch depth, prefill chunk, sampling defaults) comes from its
# ModelProfile row and is deliberately not set here.
#
#   tools/start-agentworld-4bit.sh [openai|anthropic] [full|fast] [default|concise] [<thinking>]
#
# Defaults: openai, full, standard, thinking off. <thinking> is off, on or
# any level the model lists; codex|qwen|opencode still work and mean
# openai. Serves on 127.0.0.1:8080 (NVMAI_PORT overrides it), and the same
# server switches to any other installed model a client names; the
# interactive chooser is tools/server_launcher.sh.
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/server_launcher.sh" \
  "${1:-openai}" "${2:-full}" agentworld 4 "${3:-default}" "${4:-${NVMAI_THINKING_MODE:-off}}"
