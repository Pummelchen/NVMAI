#!/usr/bin/env python3
"""Sweep decode knobs one at a time against a fixed baseline, on one model.

Each arm is a fresh server, because every knob here is read once at startup.
One prompt, one warm measured request, the server's own decode_tok_s footer
plus the runner counters that explain *why* an arm moved -- a rate without
io_hidden_pct or the hit rate is a number you cannot act on.

    python3 benchmark/nvmai_knob_sweep.py                    # all arms
    python3 benchmark/nvmai_knob_sweep.py --arms base,sync_event

The baseline is re-run last. This machine has +/-15% run-to-run spread and a
sweep takes hours, so a drifting baseline is the difference between a real 8%
win and a machine that got quieter. If the two baselines disagree by more than
the smallest win claimed, the sweep is inconclusive and says so.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvmai_profile import benchmark_log_path, server_command, server_environment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
MODEL = os.environ.get("NVMAI_BENCH_MODEL",
                       str(ROOT / "models/qwen3.8-flash-next_125B_A6B_4Bit"))
PORT = 8131
MAX_TOKENS = int(os.environ.get("SWEEP_TOKENS", "256"))
PROMPT = ("Write a detailed essay about the history of computing.")

# Every arm is one env delta from the shipped defaults, so a win is
# attributable. Combinations come later, built from what wins here.
ARMS: list[tuple[str, dict[str, str], str]] = [
    ("base", {}, "shipped defaults"),
    ("sync_event", {"NVMAI_EXPERT_IO_SYNC": "event"},
     "GPU event instead of a host wait per layer (25.7 host waits/token)"),
    ("submit_now", {"NVMAI_EXPERT_IO_SUBMISSION": "immediate"},
     "submit expert reads as planned, not deferred (io_hidden_pct 18.9%)"),
    ("gpu_resident", {"NVMAI_DECODE_EXPERT_EXECUTION": "gpu-residency"},
     "GPU-side hit/miss classification (gpu_classified_* all zero today)"),
    ("barrier", {"NVMAI_DECODE_EXPERT_EXECUTION": "barrier"},
     "control: the simple schedule, expected slower"),
    ("cache_lru", {"NVMAI_EXPERT_CACHE_POLICY": "lru"},
     "evictions(78/tok) > misses(60/tok) suggests thrash"),
    ("cache_aging", {"NVMAI_EXPERT_CACHE_POLICY": "aging-lfu"}, "ditto"),
    ("layout_pool", {"NVMAI_EXPERT_CACHE_LAYOUT": "pool"},
     "one pool instead of per-slot buffers"),
    ("prefetch_2", {"NVMAI_PREFETCH_TOP_M": "2"},
     "re-check depth 2; measured -6% worse than 1 previously"),
    ("prefetch_off", {"NVMAI_PREDICTIVE_PREFETCH": "0"},
     "control: confirm prefetch still earns its place"),
    ("slots_128", {"NVMAI_EXPERT_CACHE_SLOTS": "128"},
     "hit rate 87.4% at 96; does the curve still climb?"),
    ("slots_64", {"NVMAI_EXPERT_CACHE_SLOTS": "64"},
     "control: fewer slots must be worse if the cache matters"),
    ("no_parallel_io", {"NVMAI_PARALLEL_IO": "0"},
     "control: confirm parallel io earns its place"),
    ("keep_wired", {"NVMAI_KEEP_WIRED": "1"},
     "skip per-decode pinning"),
    # --- second pass: knobs the first sweep listed but never ran -------------
    ("rdadvise_off", {"NVMAI_RDADVISE_POLICY": "off"},
     "rdadvise costs 5.1 ms/token, 3% of the budget"),
    ("rdadvise_adaptive", {"NVMAI_RDADVISE_POLICY": "adaptive"}, "ditto"),
    ("rdadvise_bounded", {"NVMAI_RDADVISE_POLICY": "bounded"}, "ditto"),
    ("io_metal", {"NVMAI_EXPERT_IO_BACKEND": "metal"},
     "the other I/O backend, never measured on this model"),
    ("bounded_io_off", {"NVMAI_BOUNDED_IO": "0"},
     "unbounded reader footprint"),
    ("sampler_generic", {"NVMAI_SAMPLER_PATH": "generic"},
     "control: the tiled sampler should win"),
    ("slots_112", {"NVMAI_EXPERT_CACHE_SLOTS": "112"},
     "the untested middle: 96 fits at 85.4%, 128 swaps at 89.8%"),
    ("prefetch_4", {"NVMAI_PREFETCH_TOP_M": "4"},
     "control: recorded -9.8%, confirm on this build"),
    # The three arms that each landed inside drift. If they are real they
    # stack; if they are noise they will not.
    ("combo_marginal", {"NVMAI_KEEP_WIRED": "1",
                        "NVMAI_PARALLEL_IO": "0",
                        "NVMAI_EXPERT_CACHE_LAYOUT": "pool"},
     "keep_wired +4.6%, no_parallel_io +3.7%, layout_pool +1.1% together"),
    # --- 8-bit: the cache is sized in bytes, so a 1.89x expert stride buys
    # fewer slots. 64 slots at 8-bit is 16.1 GB of cache -- essentially the
    # footprint that cost 4-bit 68% at 128 slots. Sweep downward.
    ("slots_32", {"NVMAI_EXPERT_CACHE_SLOTS": "32"}, "8-bit: ~8 GB of cache"),
    ("slots_24", {"NVMAI_EXPERT_CACHE_SLOTS": "24"}, "8-bit: ~6 GB of cache"),
    ("slots_16", {"NVMAI_EXPERT_CACHE_SLOTS": "16"}, "8-bit: ~4 GB of cache"),
    # --- GDN in_proj kernel variants: gdn.metal has carried these since the
    # kernel was written and nothing ever selected them. attn_norm_qkv is
    # 29.8 ms/token at an effective 26.2 GB/s against ~100 GB/s peak, and every
    # row re-reads the 5 KiB x vector from device memory.
    ("gdn_xsh8", {"NVMAI_GDN_INPROJ": "xsh8"}, "stage x in threadgroup memory"),
    ("gdn_r16", {"NVMAI_GDN_INPROJ": "r16"}, "16 rows/threadgroup, x unstaged"),
    ("gdn_xsh16", {"NVMAI_GDN_INPROJ": "xsh16"}, "both"),
    # Each async piece was measured alone, where it pays its own overhead and
    # still cannot remove the host wait because the other two force one.
    # Together they are the only configuration that actually removes it.
    ("async_all", {"NVMAI_DECODE_EXPERT_EXECUTION": "gpu-residency",
                   "NVMAI_EXPERT_IO_SYNC": "event",
                   "NVMAI_EXPERT_IO_BACKEND": "metal"},
     "GPU classification + event sync + MTLIO together"),
    ("async_event_metal", {"NVMAI_EXPERT_IO_SYNC": "event",
                           "NVMAI_EXPERT_IO_BACKEND": "metal"},
     "event sync on the backend that signals the event natively"),
    ("async_pread", {"NVMAI_DECODE_EXPERT_EXECUTION": "gpu-residency",
                     "NVMAI_EXPERT_IO_SYNC": "event"},
     "GPU classification + event sync on pread -- avoids the MTLIO crash"),
    ("base_again", {}, "drift check -- must match base"),
]

COUNTERS = ("expert_hit_rate", "io_hidden_pct", "io_ms", "wait_ms", "body_ms",
            "expert_evictions", "io_host_waits", "cache_plan_ms", "rdadvise_ms")


def wait_ready(proc, timeout=2400) -> str | None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            conn.request("GET", "/v1/models")
            mid = json.loads(conn.getresponse().read())["data"][0]["id"]
            conn.close()
            return mid
        except OSError:
            time.sleep(5)
    return None


def request(model_id: str, tokens: int) -> None:
    payload = json.dumps({
        "model": model_id, "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "top_p": 0.95, "top_k": 20,
        "max_completion_tokens": tokens, "stream": True,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    while resp.read(8192):
        pass
    conn.close()


def run_arm(name: str, env_delta: dict[str, str]) -> dict:
    log_path = benchmark_log_path(f"sweep_{name}.log")
    env = server_environment()
    # NVMAI_RUNNER_STATS is what makes a result explainable rather than just
    # faster or slower.
    env["NVMAI_RUNNER_STATS"] = "1"
    env.update(env_delta)
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(server_command(BIN, PORT, model=MODEL),
                                env=env, stdout=fh, stderr=subprocess.STDOUT)
    try:
        model_id = wait_ready(proc)
        if model_id is None:
            tail = Path(log_path).read_text().strip().splitlines()[-3:]
            return {"arm": name, "ok": False, "note": "; ".join(tail)[:160]}
        # An arm that crashes the server drops the stream mid-read. Record it
        # and continue: losing the rest of a multi-hour sweep to one bad
        # configuration is worse than losing the arm.
        try:
            request(model_id, 32)            # warm shaders and the cache
            request(model_id, MAX_TOKENS)    # measured
        except Exception as exc:
            alive = proc.poll() is None
            return {"arm": name, "ok": False,
                    "note": f"{type(exc).__name__}: {exc}"[:110]
                            + ("" if alive else " (server died)")}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)

    text = Path(log_path).read_text()
    rates = [float(m) for m in
             re.findall(r"NVMAI generation .*?decode_tok_s=([\d.]+)", text)]
    if not rates:
        return {"arm": name, "ok": False, "note": "no decode footer"}
    out = {"arm": name, "ok": True, "tok_s": rates[-1]}
    runner = re.findall(r"NVMAI runner (.+)", text)
    if runner:
        fields = dict(kv.split("=", 1) for kv in runner[-1].split()
                      if "=" in kv)
        for c in COUNTERS:
            if c in fields:
                out[c] = float(fields[c])
    return out


def table(results: list[dict], baseline: float | None) -> str:
    head = (f"{'arm':<15}{'tok/s':>8}{'vs base':>9}{'hit%':>7}"
            f"{'io_hid%':>9}{'io_ms':>8}{'evict':>8}{'waits':>8}")
    lines = [head, "-" * len(head)]
    for r in results:
        if not r.get("ok"):
            lines.append(f"{r['arm']:<15}{'FAILED':>8}   {r.get('note','')[:50]}")
            continue
        delta = (f"{(r['tok_s'] / baseline - 1) * 100:+.1f}%"
                 if baseline else "--")
        lines.append(
            f"{r['arm']:<15}{r['tok_s']:>8.2f}{delta:>9}"
            f"{r.get('expert_hit_rate', 0) * 100:>7.1f}"
            f"{r.get('io_hidden_pct', 0):>9.1f}"
            f"{r.get('io_ms', 0):>8.1f}"
            f"{r.get('expert_evictions', 0):>8.0f}"
            f"{r.get('io_host_waits', 0):>8.0f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in ARMS if not wanted or a[0] in wanted]

    busy = subprocess.run(["pgrep", "-f", "NVMAIServer|NVMAICLI"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        print("another model process is running; stop it first", file=sys.stderr)
        return 3

    results: list[dict] = []
    baseline: float | None = None
    for name, delta, why in arms:
        print(f"\n== {name} == {why}", flush=True)
        r = run_arm(name, delta)
        results.append(r)
        if r.get("ok"):
            if name == "base":
                baseline = r["tok_s"]
            print(f"   {r['tok_s']:.2f} tok/s  hit={r.get('expert_hit_rate',0)*100:.1f}%"
                  f"  io_hidden={r.get('io_hidden_pct',0):.1f}%"
                  f"  io_ms={r.get('io_ms',0):.1f}", flush=True)
        else:
            print(f"   FAILED: {r.get('note')}", flush=True)
        print("\n" + table(results, baseline), flush=True)

    first = next((r for r in results if r["arm"] == "base" and r.get("ok")), None)
    last = next((r for r in results if r["arm"] == "base_again" and r.get("ok")), None)
    if first and last:
        drift = abs(last["tok_s"] / first["tok_s"] - 1) * 100
        print(f"\nbaseline drift over the sweep: {drift:.1f}% "
              f"({first['tok_s']:.2f} -> {last['tok_s']:.2f})")
        print("Any win smaller than this is inside the noise and is not a win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
