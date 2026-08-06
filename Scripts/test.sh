#!/usr/bin/env bash
# Run the NVMAI package test suite serially.
#
# AGENTS.md and CONTRIBUTING.md mandate `swift test --no-parallel`: the
# package shares one Metal device and global tokenizer state, so parallel
# suites interleave model-using tests. Extra arguments (e.g. --filter) are
# passed through.
set -euo pipefail

swift test --no-parallel "$@"
