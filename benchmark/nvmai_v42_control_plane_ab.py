#!/usr/bin/env python3
"""Qualification matrix for v4.2 expert control-plane modes.

Every case uses a fresh server, identical requests, bounded pread unless the
case explicitly selects Metal I/O, and the response-equality gate inherited
from the v4.1 hit/fixup benchmark. Experimental cases are not promoted merely
because this script can run them.
"""

from __future__ import annotations

import argparse
import json

import nvmai_hit_fixup_ab as benchmark


CASES = (
    ("production-deferred", "hit-fixup", "pread", "host", "per-slot", "lfu", "deferred"),
    ("immediate-host", "hit-fixup", "pread", "host", "per-slot", "lfu", "immediate"),
    ("event-pread", "hit-fixup", "pread", "event", "per-slot", "lfu", "immediate"),
    ("event-pool", "hit-fixup", "pread", "event", "pool", "lfu", "immediate"),
    ("gpu-residency", "gpu-residency", "pread", "event", "pool", "lfu", "immediate"),
    ("gpu-residency-aging", "gpu-residency", "pread", "event", "pool", "aging-lfu", "immediate"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare v4.2 expert scheduling, event, pool, and residency modes")
    parser.add_argument("--prompt", choices=tuple(benchmark.PROMPTS) + ("all",),
                        default="all")
    parser.add_argument("--case", action="append", dest="selected_cases",
                        choices=tuple(case[0] for case in CASES),
                        help="run only this case (repeatable)")
    args = parser.parse_args()
    metadata = benchmark.preflight()
    prompts = benchmark.PROMPTS if args.prompt == "all" else {
        args.prompt: benchmark.PROMPTS[args.prompt]
    }
    results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    cases = tuple(case for case in CASES
                  if not args.selected_cases or case[0] in args.selected_cases)
    for name, mode, backend, sync, layout, policy, submission in cases:
        for prompt_name, prompt in prompts.items():
            rows, log = benchmark.run_case(
                mode, prompt_name, prompt, io_backend=backend,
                io_sync=sync, cache_layout=layout, cache_policy=policy,
                io_submission=submission)
            for row in rows:
                row["case"] = name
            results.extend(rows)
            logs[f"{name}/{prompt_name}"] = str(log)

    reference = cases[0][0]
    mismatches: list[str] = []
    grouped = {(row["case"], row["prompt"], row["warmth"]): row for row in results}
    for name, *_ in cases[1:]:
        for prompt_name in prompts:
            for warmth in ("cold", "warm"):
                if grouped[(name, prompt_name, warmth)]["content"] != grouped[
                        (reference, prompt_name, warmth)]["content"]:
                    mismatches.append(f"{name}/{prompt_name}/{warmth}")

    output = benchmark.ROOT / ".build/benchmark-results/v4.2-control-plane-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "metadata": metadata,
        "configuration": {
            "model": str(benchmark.DEFAULT_MODEL_PATH),
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "seed": 41,
            "max_completion_tokens": 128,
        },
        "cases": [case[0] for case in cases],
        "logs": logs,
        "results": results,
        "response_mismatches": mismatches,
        "passed": not mismatches,
    }, indent=2) + "\n")
    print(f"results: {output}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
