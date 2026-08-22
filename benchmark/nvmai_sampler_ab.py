#!/usr/bin/env python3
"""Interleaved A/B for the Top-K sampler path, at the published v4.1 profile.

`generic` is the pre-change behavior: for any k != 64 the sampler fell through
to a single-threadgroup kernel that extracts Top-K in k full passes over a
262,144-entry vocabulary. `tiled` routes every k in 1...64 to the existing
three-stage reduction. The two are required to emit the same token — this
script checks that on the real model and measures what it costs.

Both arms run from **one binary in one machine state**, alternating
A/B/B/A with a fresh server process per run and a discarded warmup per arm,
because throughput on this Mac carries run-to-run spread wider than many
effects being tested and a sequential sweep warms the page cache as it goes.

  python3 benchmark/nvmai_sampler_ab.py --quant 4bit --pairs 2
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys

import nvmai_gate0_profile as g0
from nvmai_profile import ROOT


def one_run(quant: str, arm: str, tag: str) -> dict:
    port = g0.PORTS[quant]
    log_name = f"samplerab_{quant}_{arm}_{tag}.log"
    proc = g0.launch(quant, port, log_name, sampler_path=arm)
    if not g0.wait_ready(port):
        g0._terminate_all()
        raise SystemExit(f"[{quant}/{arm}] server did not become healthy")
    result = g0.generate(port)
    # stdout is block-buffered into the log; footers land only on exit.
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{quant}/{arm}] request failed")
    records = g0.parse_log(g0.benchmark_log_path(log_name))
    if not records:
        raise SystemExit(f"[{quant}/{arm}] no footer lines")
    record = records[-1]
    record.update(result)
    record["arm"] = arm
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(g0.QUANTS), default="4bit")
    parser.add_argument("--pairs", type=int, default=2,
                        help="A/B/B/A blocks; each block is 4 measured runs")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    rows: list[dict] = []
    try:
        # One discarded warmup per arm before any measurement, so neither arm
        # pays the first-touch cost the other does not.
        for arm in ("generic", "tiled"):
            print(f"[{args.quant}] warmup {arm}", flush=True)
            one_run(args.quant, arm, "warmup")
        for block in range(args.pairs):
            for arm in ("generic", "tiled", "tiled", "generic"):
                row = one_run(args.quant, arm, f"b{block}_{len(rows)}")
                rows.append(row)
                print(f"[{args.quant}] {arm:<7} "
                      f"{row['decode_tok_s']:.3f} tok/s  "
                      f"busy/token {row.get('busy_per_token_ms', 0):.2f} ms  "
                      f"sample_gap {row['gaps'].get('head_logits->embed', {}).get('per_token_ms', 0):.2f} ms  "
                      f"sha {row['completion_sha256']}", flush=True)
    finally:
        g0._terminate_all()

    out: dict = {"quant": args.quant, "arms": {}}
    for arm in ("generic", "tiled"):
        sel = [r for r in rows if r["arm"] == arm]
        rates = [r["decode_tok_s"] for r in sel]
        busy = [r.get("busy_per_token_ms", 0) for r in sel]
        gaps = [r["gaps"].get("head_logits->embed", {}).get("per_token_ms", 0)
                for r in sel]
        out["arms"][arm] = {
            "runs": len(sel),
            "median_tok_s": round(statistics.median(rates), 4),
            "rates": [round(x, 4) for x in rates],
            "median_busy_per_token_ms": round(statistics.median(busy), 4),
            "median_sample_gap_ms": round(statistics.median(gaps), 4),
            "digests": sorted({r["completion_sha256"] for r in sel}),
        }

    a, b = out["arms"]["generic"], out["arms"]["tiled"]
    delta = (b["median_tok_s"] / a["median_tok_s"] - 1) * 100
    out["delta_percent"] = round(delta, 2)
    out["output_identical"] = a["digests"] == b["digests"] and len(a["digests"]) == 1

    print("\n" + "=" * 68)
    print(f"SAMPLER A/B — {args.quant}, published v4.1 profile, one binary")
    print("=" * 68)
    for arm in ("generic", "tiled"):
        d = out["arms"][arm]
        print(f"  {arm:<8} median {d['median_tok_s']:7.3f} tok/s   "
              f"runs {d['rates']}")
        print(f"           busy/token {d['median_busy_per_token_ms']:.2f} ms   "
              f"sampler gap {d['median_sample_gap_ms']:.2f} ms")
    print(f"\n  DELTA: {delta:+.2f}%   "
          f"(gate is +10%: {'PASS' if delta >= 10 else 'FAIL'})")
    print(f"  Output identical across every run of both arms: "
          f"{'YES' if out['output_identical'] else 'NO'}")
    if not out["output_identical"]:
        print(f"    generic digests {a['digests']}")
        print(f"    tiled   digests {b['digests']}")
        print("    A sampling change that moves the token stream is a numerics "
              "change and must be re-baselined deliberately, not accepted here.")

    path = ROOT / f".build/benchmark-results/sampler-ab-{args.quant}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {path}")
    return 0 if out["output_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
