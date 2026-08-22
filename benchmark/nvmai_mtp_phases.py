#!/usr/bin/env python3
"""Track B1: attribute the MTP verify-pass cost, phase by phase.

The strongest historical MTP case (8-bit, 92.6% acceptance, 1.926 tokens per
pass) still lost 2%, which back-solves to a verify pass costing ~1.965x a
single decode token — against a union model predicting well under 1.3x once
`useTwoRowProjection` amortizes attention. This script measures where the
difference actually goes, using the per-pass phase counters recorded by
`StreamingMTPDecoder.advance` and printed by the server's `mtp-phases` footer.

Arms (interleaved off/on/on/off, fresh server per run, greedy, cache off in
both so the only variable is MTP):

  off — plain scalar decode; per-token cost is the denominator
  on  — draft proposal + checkpoint + width-2 verify + commit/rollback

Output: per-pass phase table, the implied verify multiple, and byte-identity
of the emitted text between arms.

  python3 benchmark/nvmai_mtp_phases.py --quant 8bit --pairs 1
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time

import nvmai_gate0_profile as g0
from nvmai_profile import (
    DEFAULT_API_MODEL, DEFAULT_CONTEXT_TOKENS, DEFAULT_KV_BITS, ROOT,
    benchmark_log_path, server_environment,
)

# Rote continuation: maximally predictable so acceptance is high, mirroring
# the published "predictable function" scenario. 256 tokens is enough passes
# (~100+) to average the phase clocks.
PROMPT = ("Write out the multiplication table for 7, from 7 x 1 to 7 x 30, "
          "one line each, in the exact format '7 x N = M'. Output only the "
          "table, nothing else.")
MAX_TOKENS = 256

MTP_MODELS = {
    "4bit": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    "8bit": ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}
SIDECAR = ROOT / "models/ornith-1.5_35B_A3B_MTP_4Bit"
PORT = 8095

MTP_RE = re.compile(
    r"NVMAI mtp drafted=(\d+) accepted=(\d+) acceptance=([0-9.]+)% "
    r"target_passes=(\d+) emitted_per_pass=([0-9.]+) "
    r"prefill_s=([0-9.]+) decode_s=([0-9.]+) decode_tok_s=([0-9.]+)")
PHASES_RE = re.compile(
    r"NVMAI mtp-phases per_pass_ms proposal=([0-9.]+) checkpoint=([0-9.]+) "
    r"verify=([0-9.]+) verify_backbone=([0-9.]+) verify_head=([0-9.]+) "
    r"verify_argmax=([0-9.]+) commit=([0-9.]+) rollback=([0-9.]+) passes=(\d+)")


VERIFY_ARM = "pair"  # NVMAI_MTP_VERIFY for the mtp-on arm


def launch(quant: str, mtp: bool, log_name: str) -> subprocess.Popen:
    binary = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
    cmd = [str(binary),
           "--port", str(PORT),
           "--model", str(MTP_MODELS[quant]),
           "--max-context", str(DEFAULT_CONTEXT_TOKENS),
           "--rope-scaling", "none",
           "--prompt-cache-mode", "off",
           "--prompt-cache-memory-mib", "0",
           "--ram-budget", "8G",
           "--kv-bits", str(DEFAULT_KV_BITS),
           "--thinking", "off"]
    if mtp:
        cmd += ["--mtp-model", str(SIDECAR), "--mtp-memory-mib", "384"]
    env = server_environment()
    env["NVMAI_RUNNER_STATS"] = "1"
    env["NVMAI_KERNEL_STATS"] = "1"
    if mtp:
        env["NVMAI_MTP_VERIFY"] = VERIFY_ARM
    log = open(benchmark_log_path(log_name), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    g0._servers.append(proc)
    return proc


def generate() -> dict | None:
    payload = json.dumps({
        "model": DEFAULT_API_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "seed": 41,
        "max_completion_tokens": MAX_TOKENS,
    }, separators=(",", ":")).encode()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        body = json.loads(conn.getresponse().read().decode())
        conn.close()
    except (OSError, ValueError) as exc:
        print(f"  ERROR: request failed: {exc}", flush=True)
        return None
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def one_run(quant: str, mtp: bool, tag: str) -> dict:
    arm = "on" if mtp else "off"
    log_name = f"mtpphases_{quant}_{arm}_{tag}.log"
    launch(quant, mtp, log_name)
    if not g0.wait_ready(PORT):
        g0._terminate_all()
        raise SystemExit(f"[{quant}/{arm}] server not healthy")
    result = generate()
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{quant}/{arm}] request failed")
    row: dict = {"arm": arm, **result}
    with open(benchmark_log_path(log_name)) as handle:
        text = handle.read()
    m = MTP_RE.search(text)
    if m:
        row.update(drafted=int(m.group(1)), accepted=int(m.group(2)),
                   acceptance=float(m.group(3)), passes=int(m.group(4)),
                   emitted_per_pass=float(m.group(5)),
                   prefill_s=float(m.group(6)), decode_s=float(m.group(7)),
                   decode_tok_s=float(m.group(8)))
    else:
        gen = g0.GENERATION_RE.search(text)
        if gen:
            row.update(prefill_s=float(gen.group(1)),
                       decode_s=float(gen.group(2)),
                       decode_tok_s=float(gen.group(3)))
    p = PHASES_RE.search(text)
    if p:
        row["phases"] = {k: float(p.group(i + 1)) for i, k in enumerate(
            ("proposal", "checkpoint", "verify", "verify_backbone",
             "verify_head", "verify_argmax", "commit", "rollback"))}
    # Per-role GPU time from the target runner, for the verify decomposition.
    roles: dict[str, float] = {}
    for rm in g0.ROLE_RE.finditer(text):
        roles[rm.group(1)] = roles.get(rm.group(1), 0) + float(rm.group(2))
    row["roles_total_ms"] = roles
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(MTP_MODELS), default="8bit")
    parser.add_argument("--verify-arm", choices=("pair", "tile"), default="pair",
                        help="NVMAI_MTP_VERIFY for the mtp-on arm")
    parser.add_argument("--pairs", type=int, default=1,
                        help="off/on/on/off blocks after the two warmups")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()
    global VERIFY_ARM
    VERIFY_ARM = args.verify_arm

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    rows: list[dict] = []
    try:
        for mtp in (False, True):
            print(f"[{args.quant}] warmup mtp={'on' if mtp else 'off'}",
                  flush=True)
            one_run(args.quant, mtp, "warmup")
        for block in range(args.pairs):
            for mtp in (False, True, True, False):
                row = one_run(args.quant, mtp, f"b{block}_{len(rows)}")
                rows.append(row)
                extra = (f"acc {row.get('acceptance', 0):.1f}% "
                         f"passes {row.get('passes', 0)}"
                         if row["arm"] == "on" else "")
                print(f"[{args.quant}] mtp={row['arm']:<3} "
                      f"{row.get('decode_tok_s', 0):7.3f} tok/s  "
                      f"sha {row['sha256']}  {extra}", flush=True)
    finally:
        g0._terminate_all()

    off = [r for r in rows if r["arm"] == "off"]
    on = [r for r in rows if r["arm"] == "on"]
    med = lambda sel, k: statistics.median([r[k] for r in sel if k in r])

    print("\n" + "=" * 70)
    print(f"MTP PHASE ATTRIBUTION — {args.quant}, greedy, cache off, "
          f"verify={args.verify_arm}")
    print("=" * 70)
    off_rate = med(off, "decode_tok_s")
    on_rate = med(on, "decode_tok_s")
    token_ms = 1000 / off_rate
    print(f"  scalar decode      {off_rate:.3f} tok/s = {token_ms:.2f} ms/token")
    print(f"  MTP decode         {on_rate:.3f} tok/s  "
          f"({(on_rate / off_rate - 1) * 100:+.1f}%)")
    accept = med(on, "acceptance")
    emitted = med(on, "emitted_per_pass")
    print(f"  acceptance         {accept:.1f}%   emitted/pass {emitted:.3f}")

    phases = [r["phases"] for r in on if "phases" in r]
    if phases:
        keys = phases[0].keys()
        pass_ms = {k: statistics.median([p[k] for p in phases]) for k in keys}
        # `verify` already contains backbone+head+argmax; the pass total is
        # the top-level phases only.
        total = sum(pass_ms[k] for k in
                    ("proposal", "checkpoint", "verify", "commit", "rollback"))
        print(f"\n  per-pass wall attribution (median across runs; "
              f"one pass emits {emitted:.3f} tokens):")
        for k in ("proposal", "checkpoint", "verify_backbone", "verify_head",
                  "verify_argmax", "commit", "rollback"):
            v = pass_ms[k]
            print(f"    {k:<18} {v:8.3f} ms   {v / token_ms:6.3f}x tokens")
        other = pass_ms["verify"] - pass_ms["verify_backbone"] \
            - pass_ms["verify_head"] - pass_ms["verify_argmax"]
        print(f"    {'verify_other':<18} {other:8.3f} ms   "
              f"{other / token_ms:6.3f}x tokens")
        print(f"    {'TOTAL':<18} {total:8.3f} ms   "
              f"{total / token_ms:6.3f}x tokens")
        print(f"\n  break-even: pass must cost < {emitted:.3f}x a token; "
              f"it costs {total / token_ms:.3f}x")

    off_digests = sorted({r["sha256"] for r in off})
    on_digests = sorted({r["sha256"] for r in on})
    identical = off_digests == on_digests and len(off_digests) == 1
    print(f"\n  output identical mtp-on vs mtp-off: "
          f"{'YES' if identical else 'NO'} "
          f"(off {off_digests}, on {on_digests})")

    out = ROOT / (f".build/benchmark-results/mtp-phases-{args.quant}"
                  f"-{args.verify_arm}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
