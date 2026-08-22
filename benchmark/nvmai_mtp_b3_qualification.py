#!/usr/bin/env python3
"""Track B3: qualify MTP with the pair verify schedule against scalar decode.

Gate (docs/v4.4-decode-width-plan.md Track B3): on a scenario with acceptance
>= 0.65, MTP-on must beat the MTP-off scalar control by >= 10% median, with
byte-identical greedy output. Run per quantization; arms are interleaved
off/on/on/off with a fresh server per run and a discarded warmup per arm.

Two scenarios:
  table    — rote multiplication table; moderate acceptance (stress case)
  function — predictable docstring'd function; the published high-acceptance
             scenario class (92.6% in the wiki qualification)

  python3 benchmark/nvmai_mtp_b3_qualification.py --quant 8bit --scenario function
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys

import nvmai_gate0_profile as g0
import nvmai_mtp_phases as ph
from nvmai_profile import ROOT

SCENARIOS = {
    "table": ph.PROMPT,
    "function": (
        "Complete this Python file exactly. Output only code, no prose.\n\n"
        "def add(a, b):\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        '    """Return the difference of a and b."""\n'
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        '    """Return the product of a and b."""\n'
        "    return a * b\n\n"
        "Now write divide, modulo, and power in the identical style."),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(ph.MTP_MODELS), default="8bit")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="function")
    parser.add_argument("--blocks", type=int, default=2,
                        help="off/on/on/off blocks after the two warmups")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    ph.PROMPT = SCENARIOS[args.scenario]
    ph.VERIFY_ARM = "pair"

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    rows: list[dict] = []
    tag_prefix = f"b3_{args.scenario}"
    try:
        for mtp in (False, True):
            print(f"[{args.quant}/{args.scenario}] warmup "
                  f"mtp={'on' if mtp else 'off'}", flush=True)
            ph.one_run(args.quant, mtp, f"{tag_prefix}_warmup")
        for block in range(args.blocks):
            for mtp in (False, True, True, False):
                row = ph.one_run(args.quant, mtp,
                                 f"{tag_prefix}_b{block}_{len(rows)}")
                rows.append(row)
                extra = (f"acc {row.get('acceptance', 0):.1f}%"
                         if row["arm"] == "on" else "")
                print(f"[{args.quant}/{args.scenario}] mtp={row['arm']:<3} "
                      f"{row.get('decode_tok_s', 0):7.3f} tok/s  "
                      f"sha {row['sha256']}  {extra}", flush=True)
    finally:
        g0._terminate_all()

    off = [r for r in rows if r["arm"] == "off"]
    on = [r for r in rows if r["arm"] == "on"]
    med = lambda sel, k: statistics.median([r[k] for r in sel if k in r])
    off_rate = med(off, "decode_tok_s")
    on_rate = med(on, "decode_tok_s")
    delta = (on_rate / off_rate - 1) * 100
    accept = med(on, "acceptance")
    off_digests = sorted({r["sha256"] for r in off})
    on_digests = sorted({r["sha256"] for r in on})
    identical = off_digests == on_digests and len(off_digests) == 1

    print("\n" + "=" * 70)
    print(f"B3 QUALIFICATION — {args.quant}, scenario={args.scenario}, "
          f"verify=pair, greedy, cache off")
    print("=" * 70)
    print(f"  scalar   median {off_rate:7.3f} tok/s  "
          f"runs {[r['decode_tok_s'] for r in off]}")
    print(f"  MTP      median {on_rate:7.3f} tok/s  "
          f"runs {[r['decode_tok_s'] for r in on]}")
    print(f"  acceptance {accept:.1f}%   emitted/pass "
          f"{med(on, 'emitted_per_pass'):.3f}")
    in_domain = accept >= 65
    print(f"  DELTA: {delta:+.2f}%   gate +10% at p>=0.65: "
          + ("PASS" if delta >= 10 and in_domain else
             "FAIL" if in_domain else
             f"OUT OF DOMAIN (acceptance {accept:.1f}% < 65%)"))
    print(f"  output identical: {'YES' if identical else 'NO'} "
          f"(off {off_digests}, on {on_digests})")

    out = ROOT / (f".build/benchmark-results/mtp-b3-{args.quant}"
                  f"-{args.scenario}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"rows": rows, "delta_percent": round(delta, 2),
                   "acceptance": accept,
                   "output_identical": identical}, handle, indent=2)
    print(f"\nwrote {out}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
