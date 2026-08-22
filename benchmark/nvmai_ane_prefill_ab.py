#!/usr/bin/env python3
"""Track A qualification: ANE prefill against the GPU path, interleaved.

Arms differ in exactly one variable: NVMAI_PREFILL_ANE. Fresh server per run,
one discarded warmup per arm, off/on/on/off blocks. The prompt is
prefill-dominated (~6.1K tokens of project documentation); a short greedy
continuation confirms decode runs correctly off the KV cache the ANE path
wrote.

Interpretation notes:
- prefill_s is the qualification metric.
- Output digests are expected to be STABLE WITHIN an arm (greedy) but to
  DIFFER BETWEEN arms: the ANE computes attention in fp16 with a different
  reduction order (~1% per-layer deviation). The script prints both digests
  and the first line of each arm's continuation so plausibility is visible.

  python3 benchmark/nvmai_ane_prefill_ab.py --quant 4bit --pairs 1
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import statistics
import subprocess
import sys

import nvmai_gate0_profile as g0
from nvmai_profile import (
    DEFAULT_API_MODEL, ROOT, benchmark_log_path, server_command,
    server_environment,
)

PORT = 8098
MODELS = {
    "4bit": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    "8bit": ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}


# A pinned, self-contained prompt body. It was previously built from README.md
# and AGENTS.md, which made the benchmark silently non-reproducible: editing
# those files changed the prompt length (6,103 -> 6,593 tokens between two
# runs) and therefore the quadratic attention cost, so the arms of different
# sweeps were not comparable. A benchmark prompt must never be derived from
# files under active edit.
_PARAGRAPH = (
    "The runtime keeps routed mixture-of-experts weights on solid-state "
    "storage and loads only the experts that the router selects for each "
    "token, so a large model runs inside a small declared memory budget. "
    "Attention state is held in a compressed key-value cache whose precision "
    "is chosen independently of the weight precision. Prefill processes the "
    "prompt in fixed chunks, while decode emits one token at a time and is "
    "bounded by memory bandwidth rather than arithmetic. "
)


def build_prompt() -> str:
    """~56,000 characters of stable English prose (about 6K tokens), matching
    the scale of the original workload and independent of repository files."""
    return ("Summarize the following technical description in 40 words.\n\n"
            + _PARAGRAPH * 119)


def launch(quant: str, ane: bool, log_name: str) -> None:
    binary = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
    cmd = server_command(binary, PORT, model=MODELS[quant], cache_mode="off")
    env = server_environment()
    env["NVMAI_PREFILL_ANE"] = "on" if ane else "off"
    log = open(benchmark_log_path(log_name), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    g0._servers.append(proc)


def generate(prompt: str, max_tokens: int) -> dict | None:
    payload = json.dumps({
        "model": DEFAULT_API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 41,
        "max_completion_tokens": max_tokens,
    }, separators=(",", ":")).encode()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        body = json.loads(conn.getresponse().read().decode())
        conn.close()
    except (OSError, ValueError) as exc:
        print(f"  ERROR: request failed: {exc}", flush=True)
        return None
    text = ((body.get("choices") or [{}])[0].get("message") or {}
            ).get("content") or ""
    usage = body.get("usage") or {}
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "first_line": text.strip().splitlines()[0][:90] if text.strip()
            else "(empty)"}


def one_run(quant: str, ane: bool, prompt: str, tag: str) -> dict:
    arm = "ane" if ane else "gpu"
    log_name = f"aneab_{quant}_{arm}_{tag}.log"
    launch(quant, ane, log_name)
    if not g0.wait_ready(PORT):
        g0._terminate_all()
        raise SystemExit(f"[{quant}/{arm}] server not healthy")
    result = generate(prompt, 48)
    g0._terminate_all()
    if result is None:
        raise SystemExit(f"[{quant}/{arm}] request failed")
    row: dict = {"arm": arm, **result}
    with open(benchmark_log_path(log_name)) as handle:
        text = handle.read()
    gen = g0.GENERATION_RE.search(text)
    if gen:
        row.update(prefill_s=float(gen.group(1)),
                   decode_s=float(gen.group(2)),
                   decode_tok_s=float(gen.group(3)))
    row["fallback"] = "ane-prefill fallback" in text
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(MODELS), default="4bit")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, g0._on_signal)
    signal.signal(signal.SIGTERM, g0._on_signal)
    g0.preflight(100 if args.allow_busy_gpu else 15)

    prompt = build_prompt()
    rows: list[dict] = []
    try:
        for ane in (False, True):
            print(f"[{args.quant}] warmup {'ane' if ane else 'gpu'}",
                  flush=True)
            one_run(args.quant, ane, prompt, "warmup")
        for block in range(args.pairs):
            for ane in (False, True, True, False):
                row = one_run(args.quant, ane, prompt,
                              f"b{block}_{len(rows)}")
                rows.append(row)
                print(f"[{args.quant}] {row['arm']:<3} "
                      f"prefill {row.get('prefill_s', 0):7.2f} s  "
                      f"decode {row.get('decode_tok_s', 0):6.3f} tok/s  "
                      f"sha {row['sha256']}"
                      + ("  [FALLBACK]" if row.get("fallback") else ""),
                      flush=True)
    finally:
        g0._terminate_all()

    gpu = [r for r in rows if r["arm"] == "gpu"]
    ane = [r for r in rows if r["arm"] == "ane"]
    med = lambda sel, k: statistics.median([r[k] for r in sel if k in r])
    gpu_prefill = med(gpu, "prefill_s")
    ane_prefill = med(ane, "prefill_s")

    print("\n" + "=" * 70)
    print(f"ANE PREFILL A/B — {args.quant}, "
          f"{gpu[0].get('prompt_tokens')} prompt tokens, greedy, cache off")
    print("=" * 70)
    print(f"  GPU prefill median {gpu_prefill:8.2f} s   "
          f"runs {[r.get('prefill_s') for r in gpu]}")
    print(f"  ANE prefill median {ane_prefill:8.2f} s   "
          f"runs {[r.get('prefill_s') for r in ane]}")
    print(f"  SPEEDUP: {gpu_prefill / ane_prefill:.2f}x   "
          f"(gate is >=1.5x: "
          f"{'PASS' if gpu_prefill / ane_prefill >= 1.5 else 'FAIL'})")
    print(f"  decode after prefill: GPU {med(gpu, 'decode_tok_s'):.3f} "
          f"vs ANE {med(ane, 'decode_tok_s'):.3f} tok/s")
    for arm_rows, label in ((gpu, "gpu"), (ane, "ane")):
        digests = sorted({r["sha256"] for r in arm_rows})
        stable = len(digests) == 1
        print(f"  {label} digests {digests} "
              f"({'stable' if stable else 'UNSTABLE'})")
        print(f"    continuation: {arm_rows[0]['first_line']}")
    if any(r.get("fallback") for r in ane):
        print("  WARNING: an ANE run logged a GPU fallback — "
              "the arms did not measure what they claim")

    out = ROOT / f".build/benchmark-results/ane-prefill-ab-{args.quant}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
