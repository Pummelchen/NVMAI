#!/usr/bin/env python3
"""Compare bounded F_NOCACHE pread with experimental Metal I/O.

Uses the same fresh-server cold/repeat protocol and response-equality gate as
the hit/fixup A/B. Metal I/O is not eligible as a production default unless
these timing results are paired with whole-machine memory-pressure evidence.
"""

from __future__ import annotations

import argparse
import json

import nvmai_hit_fixup_ab as benchmark


def main() -> int:
    argparse.ArgumentParser(
        description="Compare bounded pread and experimental Metal I/O"
    ).parse_args()
    metadata = benchmark.preflight()
    results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    for backend in ("pread", "metal"):
        for prompt_name, prompt in benchmark.PROMPTS.items():
            rows, log = benchmark.run_case(
                "hit-fixup", prompt_name, prompt, io_backend=backend)
            results.extend(rows)
            logs[f"{backend}/{prompt_name}"] = str(log)

    by_case = {
        (row["prompt"], row["warmth"], row["io_backend"]): row
        for row in results
    }
    mismatches = []
    for prompt in benchmark.PROMPTS:
        for warmth in ("cold", "warm"):
            pread = by_case[(prompt, warmth, "pread")]["content"]
            metal = by_case[(prompt, warmth, "metal")]["content"]
            if pread != metal:
                mismatches.append(f"{prompt}/{warmth}")

    output = benchmark.ROOT / ".build/benchmark-results/io-backend-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "metadata": metadata,
        "warning": "Metal I/O requires separate page-cache/memory-pressure validation",
        "logs": logs,
        "results": results,
        "response_mismatches": mismatches,
        "passed": not mismatches,
    }, indent=2) + "\n")
    print(f"results: {output}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
