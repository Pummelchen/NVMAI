#!/usr/bin/env bash
# Pick a Python interpreter for the tools that need the analysis stack
# (numpy, ml_dtypes, safetensors).
#
# Sourced, never executed: `source tools/lib/python.sh` then use
# `$NVMAI_PYTHON` (call `nvmai_resolve_python` first).
#
# Why detection rather than a `python3` name or a pinned `python3.13`:
#
#   * A bare `python3` is not a version. On a stock macOS it is 3.9 from
#     /usr/bin, older than the 3.10 syntax these tools use and without any of
#     the dependencies -- so substituting the name breaks every conversion with
#     a confusing SyntaxError or ModuleNotFoundError.
#   * A pinned `python3.13` fails on a machine whose analysis stack is installed
#     under 3.12 or 3.14, which is the complaint this resolver exists to fix.
#
# So the candidates are tried newest-first and each one is *tested*: it must be
# at least the required version and must import the dependencies. The first that
# passes wins. `NVMAI_PYTHON` overrides the search entirely, for a virtualenv or
# a build that pins its own interpreter.

# Minimum interpreter version. 3.10 is what the syntax actually needs (`X | Y`
# in annotations under `from __future__ import annotations` is fine, but `match`
# statements and the newer typing forms are not universally avoidable).
NVMAI_PYTHON_MIN_MAJOR=3
NVMAI_PYTHON_MIN_MINOR=10

# Dependency check the tools share. Kept as one string so the shell resolver and
# the Python files cannot drift.
NVMAI_PYTHON_DEPS="import numpy, ml_dtypes, safetensors"

nvmai_python_note() {
  # One line, usable in any message, naming this shell's resolved interpreter.
  # Call it after `nvmai_resolve_python`.
  echo "this checkout uses ${NVMAI_PYTHON:-python3}; install them for it with \"${NVMAI_PYTHON:-python3} -m pip install safetensors numpy ml_dtypes\", or point NVMAI_PYTHON at another Python 3"
}

nvmai_resolve_python() {
  # Already resolved in this shell.
  if [[ -n "${NVMAI_PYTHON:-}" ]]; then
    printf '%s' "$NVMAI_PYTHON"
    return 0
  fi

  local candidate
  # Newest first. `python3` is tested last on purpose: it is the name most
  # likely to be an old system interpreter.
  for candidate in \
    "$(command -v python3.14 2>/dev/null)" \
    "$(command -v python3.13 2>/dev/null)" \
    "$(command -v python3.12 2>/dev/null)" \
    "$(command -v python3.11 2>/dev/null)" \
    "$(command -v python3.10 2>/dev/null)" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
  do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    "$candidate" -c "import sys
assert sys.version_info >= ($NVMAI_PYTHON_MIN_MAJOR, $NVMAI_PYTHON_MIN_MINOR), sys.version.split()[0]
$NVMAI_PYTHON_DEPS" >/dev/null 2>&1 || continue
    printf '%s' "$candidate"
    return 0
  done

  {
    echo "no usable Python interpreter found for the NVMAI tools." >&2
    echo "  need: Python >= $NVMAI_PYTHON_MIN_MAJOR.$NVMAI_PYTHON_MIN_MINOR with $NVMAI_PYTHON_DEPS" >&2
    echo "  tried (newest first): python3.14 python3.13 python3.12 python3.11 python3.10 python3" >&2
    echo "  fix:  install the packages for a Python 3, e.g." >&2
    echo "          python3 -m pip install safetensors numpy ml_dtypes" >&2
    echo "        or set NVMAI_PYTHON=/path/to/python3 to choose one explicitly" >&2
  } >&2
  return 1
}
