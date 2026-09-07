#!/usr/bin/env python3
"""Run the book and the coder benchmark against a small local model.

The memory work so far asked whether a small model can keep the store for
the 35B. This asks the opposite question: how much continuity does a small
model have on its own? It runs the same two benchmarks, scored by the same
code, against llama.cpp on the CPU -- so the numbers sit directly beside
every 35B row already measured.

Only the memory-off arms make sense here: llama-server has no memory
feature, so `summary` (the harness carries a 200-word note) and `control`
(nothing is carried) are what a small model can be asked for. That is the
right comparison anyway -- it is the baseline column of every install table.

The harness is driven rather than edited, because the matrix is invoking the
same modules and a change to them would reach a run in flight.

    benchmark/memory_small_model.py book     # summary arm, one run
    benchmark/memory_small_model.py pong     # control arm, one run
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

PORT = os.environ.get("NVMAI_PORT", "8098")
TEMPERATURE = float(os.environ.get("NVMAI_SMALL_TEMP", "0.7"))
TOP_P = float(os.environ.get("NVMAI_SMALL_TOP_P", "0.95"))
THINK = os.environ.get("NVMAI_SMALL_THINK") == "1"

os.environ.setdefault("NVMAI_PORT", PORT)
os.environ.setdefault(
    "NVMAI_MEMVAL_RESULTS",
    str(ROOT / f".build/benchmark-logs/memory-small-{os.environ.get('NVMAI_SMALL_LABEL', 'qwen2b')}"))


def patch(module):
    """Sampling the harness cannot express: top_p, and the thinking switch.

    Thinking is off for the same reason it is off everywhere else in this
    work -- a 2B spends its whole budget deliberating and never reaches the
    answer. Here that would cost the chapters themselves, not just a verdict.
    """
    def sampling():
        return {"temperature": TEMPERATURE, "top_p": TOP_P,
                "chat_template_kwargs": {"enable_thinking": THINK}}
    module.sampling = sampling
    return module


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "book"
    if which == "book":
        import memory_book as bench
        arm = "summary"
    else:
        import memory_value as bench
        arm = "control"
    patch(bench)
    Path(os.environ["NVMAI_MEMVAL_RESULTS"]).mkdir(parents=True, exist_ok=True)
    print(f"{which} / {arm} against port {PORT}, temp={TEMPERATURE} top_p={TOP_P} "
          f"thinking={'on' if THINK else 'off'}")
    bench.run_arm(arm)
