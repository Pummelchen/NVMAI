#!/usr/bin/env python3
"""Gate 0: decompose the decode token at the published v4.1 benchmark profile.

Every post-4.1 experiment targeted exposed expert I/O, which the v4.2
qualification measures at 18.8% of decode. The other 78.7% has never been
decomposed, and never at the configuration the published benchmark actually
uses (512 tokens, native 262K context, 8 GiB budget, 8-bit KV, prompt cache
on, temperature 0.6). This script produces that decomposition.

It changes no runtime behavior. It launches the ordinary production server
profile from `nvmai_profile` with `NVMAI_RUNNER_STATS=1` and
`NVMAI_KERNEL_STATS=1`, replays the published continuous-prose workload, and
parses the server's own footer lines.

Protocol (matches the published v4.1 benchmark):
  - one discarded warmup run, then N measured runs
  - a fresh server process per run, so every measured run starts cold
  - identical prompt, seed, sampling, context, KV precision, and budget
  - medians decide; the spread is reported alongside

Decision rule for Gate 0, per docs/v4.4-decode-width-plan.md:
  busy_per_token >= 45 ms  -> decode is bandwidth-bound; ceiling ~1.35x
  busy_per_token ~ 28-33 ms -> the remainder is dependency stall; ceiling ~1.9-2.1x

Usage:
  python3 benchmark/nvmai_gate0_profile.py                 # both quantizations
  python3 benchmark/nvmai_gate0_profile.py --quant 4bit    # one
  python3 benchmark/nvmai_gate0_profile.py --runs 3
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import time

from nvmai_profile import (
    DEFAULT_API_MODEL,
    ROOT,
    benchmark_log_path,
    server_command,
    server_environment,
)

# The published v4.1 continuous-generation workload: 512 tokens of plain
# English prose about an ordinary day in a small town.
PROMPT = (
    "Write continuous plain English prose about an ordinary day in a small "
    "town. Do not use lists, headings, or bullet points. Keep writing until "
    "you are told to stop."
)
MAX_TOKENS = 512
TEMPERATURE = 0.6
SEED = 42
TOP_K = 20  # GenerationDefaults.topK; overridable with --top-k for A/B probes

QUANTS = {
    "4bit": ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    "8bit": ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}
PORTS = {"4bit": 8091, "8bit": 8092}

GENERATION_RE = re.compile(
    r"NVMAI generation prefill_s=([0-9.]+) decode_s=([0-9.]+) "
    r"decode_tok_s=([0-9.]+)")
RUNNER_RE = re.compile(r"NVMAI runner (.+)")
ROLE_RE = re.compile(
    r"NVMAI kernel role=(\S+) gpu_ms=([0-9.]+) per_token_ms=([0-9.]+) "
    r"count=(\d+)")
GAP_RE = re.compile(
    r"NVMAI gap (\S+) total_ms=([0-9.]+) per_token_ms=([0-9.]+) count=(\d+)")
OCCUPANCY_RE = re.compile(
    r"NVMAI kernel busy_ms=([0-9.]+) span_ms=([0-9.]+) occupancy=([0-9.]+)% "
    r"busy_share_of_decode=([0-9.]+)% busy_per_token_ms=([0-9.]+)")
TOTAL_GPU_RE = re.compile(
    r"NVMAI kernel total_gpu_ms=([0-9.]+) gpu_share_of_decode=([0-9.]+)%")

_servers: list[subprocess.Popen] = []


def _terminate_all() -> None:
    """Terminate by PID only; never pkill by pattern."""
    for proc in list(_servers):
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        _servers.remove(proc)


def _on_signal(_sig, _frame):
    _terminate_all()
    sys.exit(130)


def idle_gpu_utilization() -> int | None:
    """Percent GPU utilization from another process, or None if unreadable."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = [int(m) for m in
              re.findall(r'"Device Utilization %"=(\d+)', out)]
    return max(values) if values else None


def machine_load() -> dict:
    """Whole-machine contention signals a model run is sensitive to."""
    out: dict = {}
    try:
        ps = subprocess.run(["ps", "-Ao", "%cpu,comm", "-r"],
                            capture_output=True, text=True, timeout=15).stdout
        rows = []
        for line in ps.splitlines()[1:]:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                try:
                    rows.append((float(parts[0]), parts[1]))
                except ValueError:
                    pass
        mine = ("NVMAIServer", "NVMAICLI", "python")
        out["busy_processes"] = [
            (c, n) for c, n in rows[:8]
            if c >= 25 and not any(m in n for m in mine)]
    except (OSError, subprocess.SubprocessError):
        out["busy_processes"] = []
    try:
        mp = subprocess.run(["memory_pressure", "-Q"],
                            capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"free percentage:\s*(\d+)", mp)
        out["free_percent"] = int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        out["free_percent"] = None
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                            capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"used\s*=\s*([0-9.]+)M", sw)
        out["swap_used_mb"] = float(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        out["swap_used_mb"] = None
    return out


def preflight(max_gpu_percent: int) -> None:
    """Refuse to run if another model process is live (AGENTS.md rule), or if
    another application is already consuming the GPU.

    The second check exists because it has burned a full run: a background
    game held the GPU at 62-78% utilization, which more than doubled
    `busy_per_token` and cut decode to 40% of the published rate. Nothing in
    the tok/s number explains that on its own, so the guard is here rather
    than in a reviewer's head.
    """
    pattern = ("NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|"
               "NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm")
    found = subprocess.run(["pgrep", "-fl", pattern],
                           capture_output=True, text=True)
    mine = str(pathlib.Path(__file__).name)
    lines = [ln for ln in found.stdout.splitlines() if mine not in ln]
    if lines:
        raise SystemExit(
            "refusing to start: model processes already running:\n  "
            + "\n  ".join(lines))

    used = idle_gpu_utilization()
    if used is None:
        print("WARNING: could not read GPU utilization; "
              "confirm the machine is idle before trusting these numbers",
              flush=True)
    elif used > max_gpu_percent:
        raise SystemExit(
            f"refusing to start: GPU is already {used}% busy before NVMAI "
            f"launches (threshold {max_gpu_percent}%).\n"
            "Quit whatever is using the GPU, or pass --allow-busy-gpu to "
            "measure anyway. Results from a contended GPU are not comparable "
            "with the published benchmark.")

    # CPU and memory contention are as invalidating as GPU contention and far
    # less visible. A busy machine does not merely add noise here: it pushes
    # the expert slot cache out of RAM, and a run whose hit rate collapses
    # (0.277 against a normal 0.877 was observed) reports timings that look
    # like a code regression and are not one. Checked, not assumed.
    if max_gpu_percent < 100:
        load = machine_load()
        problems = []
        if load["busy_processes"]:
            listing = ", ".join(f"{n.split('/')[-1]} {c:.0f}%"
                                for c, n in load["busy_processes"])
            problems.append(f"other processes are busy: {listing}")
        if load["free_percent"] is not None and load["free_percent"] < 55:
            problems.append(
                f"only {load['free_percent']}% of memory is free; the expert "
                "cache will not stay resident")
        if load["swap_used_mb"] is not None and load["swap_used_mb"] > 3072:
            problems.append(
                f"{load['swap_used_mb']:.0f} MB of swap is in use; let the "
                "machine settle before measuring")
        if problems:
            raise SystemExit(
                "refusing to start: the machine is not idle.\n  - "
                + "\n  - ".join(problems)
                + "\nQuit the busy applications, or pass --allow-busy-gpu to "
                "measure anyway (results will not be comparable).")


def launch(quant: str, port: int, log_name: str,
           sampler_path: str | None = None) -> subprocess.Popen:
    binary = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
    if not binary.exists():
        raise SystemExit(f"missing release binary: {binary}")
    cmd = server_command(binary, port, model=QUANTS[quant])
    env = server_environment()
    env["NVMAI_RUNNER_STATS"] = "1"
    env["NVMAI_KERNEL_STATS"] = "1"
    if sampler_path:
        env["NVMAI_SAMPLER_PATH"] = sampler_path
    log = open(benchmark_log_path(log_name), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    _servers.append(proc)
    return proc


def wait_ready(port: int, attempts: int = 40) -> bool:
    for _ in range(attempts):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            ok = "ok" in conn.getresponse().read().decode()
            conn.close()
            if ok:
                return True
        except OSError:
            pass
        time.sleep(5)
    return False


def generate(port: int) -> dict | None:
    payload = json.dumps({
        "model": DEFAULT_API_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "top_k": TOP_K,
        "presence_penalty": 0.0,
        "seed": SEED,
        "max_completion_tokens": MAX_TOKENS,
    }, separators=(",", ":")).encode()
    start = time.time()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        body = json.loads(conn.getresponse().read().decode())
        conn.close()
    except (OSError, ValueError) as exc:
        print(f"  ERROR: request failed: {exc}", flush=True)
        return None
    usage = body.get("usage") or {}
    choices = body.get("choices") or [{}]
    text = ((choices[0].get("message") or {}).get("content")) or ""
    return {
        "wall_s": round(time.time() - start, 3),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        # Sampling changes are only safe if the token stream is unchanged at a
        # fixed seed, so every run carries a digest of what it actually emitted.
        "completion_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def parse_log(path: str) -> list[dict]:
    """Group footer lines into one record per completed request."""
    records: list[dict] = []
    current: dict | None = None
    with open(path) as handle:
        for line in handle:
            gen = GENERATION_RE.search(line)
            if gen:
                if current:
                    records.append(current)
                current = {
                    "prefill_s": float(gen.group(1)),
                    "decode_s": float(gen.group(2)),
                    "decode_tok_s": float(gen.group(3)),
                    "roles": {}, "gaps": {},
                }
                continue
            if current is None:
                continue
            runner = RUNNER_RE.search(line)
            if runner:
                for field in runner.group(1).split():
                    if "=" in field:
                        key, value = field.split("=", 1)
                        try:
                            current[key] = float(value)
                        except ValueError:
                            pass
                continue
            role = ROLE_RE.search(line)
            if role:
                current["roles"][role.group(1)] = {
                    "gpu_ms": float(role.group(2)),
                    "per_token_ms": float(role.group(3)),
                    "count": int(role.group(4)),
                }
                continue
            gap = GAP_RE.search(line)
            if gap:
                current["gaps"][gap.group(1)] = {
                    "total_ms": float(gap.group(2)),
                    "per_token_ms": float(gap.group(3)),
                    "count": int(gap.group(4)),
                }
                continue
            total = TOTAL_GPU_RE.search(line)
            if total:
                current["total_gpu_ms"] = float(total.group(1))
                current["gpu_share_of_decode"] = float(total.group(2))
                continue
            occ = OCCUPANCY_RE.search(line)
            if occ:
                current["busy_ms"] = float(occ.group(1))
                current["span_ms"] = float(occ.group(2))
                current["occupancy_pct"] = float(occ.group(3))
                current["busy_share_of_decode_pct"] = float(occ.group(4))
                current["busy_per_token_ms"] = float(occ.group(5))
    if current:
        records.append(current)
    return records


def run_quant(quant: str, runs: int) -> dict:
    port = PORTS[quant]
    measured: list[dict] = []
    for index in range(runs + 1):
        label = "warmup" if index == 0 else f"run {index}"
        log_name = f"gate0_{quant}_{index}.log"
        print(f"[{quant}] {label}: starting fresh server", flush=True)
        proc = launch(quant, port, log_name)
        if not wait_ready(port):
            _terminate_all()
            raise SystemExit(f"[{quant}] server did not become healthy")
        result = generate(port)
        # stdout is block-buffered into the log; the footer only lands on disk
        # when the process exits, so always terminate before parsing.
        _terminate_all()
        time.sleep(3)
        if result is None:
            raise SystemExit(f"[{quant}] {label} failed")
        records = parse_log(benchmark_log_path(log_name))
        if not records:
            raise SystemExit(f"[{quant}] {label}: no footer lines in log")
        record = records[-1]
        record.update(result)
        print(f"[{quant}] {label}: {record['decode_tok_s']:.3f} tok/s, "
              f"busy_per_token={record.get('busy_per_token_ms', 0):.3f} ms, "
              f"occupancy={record.get('occupancy_pct', 0):.1f}%", flush=True)
        if index > 0:
            measured.append(record)
    return summarize(quant, measured)


def _median(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(statistics.median(values), 4) if values else None


def _spread(rows: list[dict], key: str) -> list[float]:
    return [round(r[key], 4) for r in rows
            if isinstance(r.get(key), (int, float))]


def summarize(quant: str, rows: list[dict]) -> dict:
    scalar_keys = [
        "decode_tok_s", "decode_s", "prefill_s",
        "busy_per_token_ms", "occupancy_pct", "busy_share_of_decode_pct",
        "gpu_share_of_decode",
        "cb1_ms", "io_ms", "cb2_ms", "head_ms", "head_fused_ms",
        "rdadvise_ms", "wait_ms", "body_ms",
        "router_readback_ms", "cache_plan_ms", "io_queue_ms",
        "io_completion_to_fixup_ms",
        "expert_hit_rate", "expert_read_mib", "io_hidden_pct",
        "expert_load_p50_ms", "expert_load_p95_ms", "expert_load_p99_ms",
        "io_host_waits", "io_host_waits_avoided", "hit_fixup_layers",
        "completion_tokens",
    ]
    digests = sorted({r.get("completion_sha256") for r in rows if r.get("completion_sha256")})
    summary = {
        "quant": quant,
        "runs": len(rows),
        "median": {k: _median(rows, k) for k in scalar_keys},
        "spread": {k: _spread(rows, k) for k in
                   ("decode_tok_s", "busy_per_token_ms", "occupancy_pct")},
    }
    roles: dict[str, list[float]] = {}
    for row in rows:
        for name, data in row.get("roles", {}).items():
            roles.setdefault(name, []).append(data["per_token_ms"])
    summary["roles_per_token_ms"] = {
        name: round(statistics.median(values), 4)
        for name, values in sorted(roles.items(),
                                   key=lambda kv: -statistics.median(kv[1]))
    }
    gaps: dict[str, list[float]] = {}
    for row in rows:
        for name, data in row.get("gaps", {}).items():
            gaps.setdefault(name, []).append(data["per_token_ms"])
    summary["gaps_per_token_ms"] = {
        name: round(statistics.median(values), 4)
        for name, values in sorted(gaps.items(),
                                   key=lambda kv: -statistics.median(kv[1]))
    }
    summary["completion_sha256"] = digests
    summary["raw"] = rows
    return summary


def report(summaries: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("GATE 0 — decode token decomposition, published v4.1 profile")
    print("=" * 72)
    for s in summaries:
        m = s["median"]
        tok_s = m["decode_tok_s"] or 0
        token_ms = 1000 / tok_s if tok_s else 0
        busy = m["busy_per_token_ms"] or 0
        print(f"\n## {s['quant']}  ({s['runs']} measured runs)")
        print(f"  decode            {tok_s:.3f} tok/s  "
              f"= {token_ms:.2f} ms/token")
        print(f"  spread            {s['spread']['decode_tok_s']}")
        print(f"  GPU busy/token    {busy:.3f} ms "
              f"({(busy / token_ms * 100) if token_ms else 0:.1f}% of token)")
        print(f"  NOT GPU busy      {token_ms - busy:.3f} ms "
              f"({((token_ms - busy) / token_ms * 100) if token_ms else 0:.1f}%)")
        print(f"  queue occupancy   {m['occupancy_pct']:.1f}%")
        print(f"  expert hit rate   {(m['expert_hit_rate'] or 0) * 100:.2f}%"
              f"   I/O hidden {m['io_hidden_pct']:.2f}%")
        print(f"  host wait/token   {m['wait_ms']:.3f} ms"
              f"   expert io {m['io_ms']:.3f} ms")
        print(f"  router readback   {m['router_readback_ms']:.4f} ms"
              f"   cache plan {m['cache_plan_ms']:.4f} ms")
        print("  top GPU roles (ms/token):")
        for name, value in list(s["roles_per_token_ms"].items())[:8]:
            print(f"      {name:<28} {value:.4f}")
        print("  top inter-command gaps (ms/token):")
        for name, value in list(s["gaps_per_token_ms"].items())[:6]:
            print(f"      {name:<28} {value:.4f}")
        verdict = ("BANDWIDTH-BOUND (ceiling ~1.35x; only Track C moves it)"
                   if busy >= 45 else
                   "DEPENDENCY-STALLED (ceiling ~1.9-2.1x; Track B is the game)"
                   if busy <= 35 else
                   "MIXED — neither branch of the Gate 0 rule fires cleanly")
        print(f"  VERDICT: {verdict}")
        digests = s.get("completion_sha256") or []
        print(f"  output digest    {digests}"
              f"{'  (RUNS DISAGREE)' if len(digests) > 1 else ''}")


def main() -> int:
    global TOP_K
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=sorted(QUANTS), action="append")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="probe override; production default is 20")
    parser.add_argument("--allow-busy-gpu", action="store_true",
                        help="measure even if another process holds the GPU; "
                             "results are not comparable with the benchmark")
    args = parser.parse_args()
    TOP_K = args.top_k

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    preflight(100 if args.allow_busy_gpu else 15)

    quants = args.quant or ["4bit", "8bit"]
    summaries = []
    try:
        for quant in quants:
            summaries.append(run_quant(quant, args.runs))
    finally:
        _terminate_all()

    report(summaries)
    out = args.out or str(ROOT / ".build/benchmark-results/gate0-profile.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"summaries": summaries}, handle, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
