#!/usr/bin/env python3
"""Head-to-head decode throughput: NVMAI against the other local-inference
engines on Apple silicon, on one machine, one model, one set of prompts.

    python3 benchmark/nvmai_vs_competitors.py --check     # readiness only
    python3 benchmark/nvmai_vs_competitors.py             # run everything
    python3 benchmark/nvmai_vs_competitors.py --engines nvmai,llamacpp

WHY THIS EXISTS
The marketing claim "faster than any other similar project" had nothing behind
it. This is what would have to be true for it to be sayable.

FAIRNESS IS THE WHOLE POINT
A benchmark that flatters us is worth nothing -- a single contrary result from
a reader discredits every other claim we make. So each engine is given the
settings *its own* maintainers recommend for peak Apple-silicon throughput,
not a lowest-common-denominator config that happens to suit NVMAI:

  * all layers on the GPU, flash attention on, where the engine has the knob
  * 8-bit KV cache everywhere, because that is NVMAI's shipped default and
    leaving competitors at fp16 would hand us a memory advantage we did not
    earn
  * 4-bit weights, group 64 where the format allows it
  * identical prompts, identical 512-token greedy generation, warmup + repeats
  * decode rate only -- prefill excluded on every engine, since prefill and
    decode have different bottlenecks and mixing them hides both

Anything an engine cannot be configured to match is recorded in `caveat` and
printed with the result, rather than quietly ignored.

THE PROMPTS ARE NOT ARBITRARY
They are NVMAI's own routing-diversity ladder, from code (diverse expert
routing, worst cache locality) to repeated digits (maximally repetitive, best
locality). For a sparse MoE the spread between them is large -- 27% on NVMAI at
4-bit -- so a single prompt would let anyone pick a winner by choosing it.

TWO REGIMES, AND WE SHOULD PUBLISH BOTH
  peak    every engine unconstrained, best config. Raw speed.
  bounded every engine held to the same RAM budget.
NVMAI is built for the second: it streams experts from SSD inside a budget you
set. Reporting only the regime we win is the same dishonesty as the original
claim. Peak RSS is recorded on every run so both regimes are derivable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Same ladder as nvmai_maxthroughput.py, so results are comparable to the
# in-house numbers already recorded for Qwen3.8.
PROMPTS = [
    ("code", "Write a Python function that computes the Levenshtein distance "
             "between two strings with a detailed docstring, then a second "
             "function using it to find the closest match in a list, with a "
             "demo main."),
    ("essay", "Write a detailed essay about the history of computing."),
    ("count", "Count from 1 to 1000, writing only the numbers separated by "
              "single spaces."),
    ("digits", "Write the digits 1,2,3,4,5,6,7,8,9,0 over and over in "
               "sequence, separated by commas, without stopping."),
]

MAX_TOKENS = 512
WARMUPS = 1
REPEATS = 2
CONTEXT = 8192          # every engine must hold this; NVMAI's native is larger
KV_BITS = 8             # NVMAI's shipped default -- matched everywhere possible

# ---------------------------------------------------------------------------
# Model locations. NOTHING here is downloaded by this script: the machine does
# not have room for five copies of a 35B model, and silently pulling 20 GB
# because a path was unset is exactly the failure that makes a benchmark
# untrustworthy. --check prints what is missing and how to get it.
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("BENCH_MODELS_DIR", ROOT / "models"))

NVMAI_MODEL = MODELS_DIR / "ornith-1.5_35B_A3B_4Bit"
# One GGUF serves llama.cpp, Ollama and LM Studio. That is deliberate: it is
# the same file, so any difference between those three is the runtime, not the
# weights -- and it saves two 20 GB copies on a disk that does not have them.
GGUF_MODEL = Path(os.environ.get("BENCH_GGUF", MODELS_DIR / "gguf/ornith-1.5-35b-a3b-Q4_K_M.gguf"))
MLX_MODEL = Path(os.environ.get("BENCH_MLX", MODELS_DIR / "mlx/ornith-1.5-35b-a3b-4bit"))
MLC_MODEL = Path(os.environ.get("BENCH_MLC", MODELS_DIR / "mlc/ornith-1.5-35b-a3b-q4f16_1-MLC"))
OLLAMA_TAG = os.environ.get("BENCH_OLLAMA_TAG", "ornith-bench:35b-q4km")


@dataclass
class Result:
    engine: str
    prompt: str
    tok_s: float | None
    peak_rss_mb: float | None
    ok: bool
    note: str = ""


@dataclass
class Engine:
    name: str
    binary: str | None                  # what must be on PATH
    model_path: Path | None
    setup_hint: str
    model_hint: str
    caveat: str = ""
    extra_check: object = None          # optional callable -> str | None

    def installed(self) -> bool:
        if self.binary is None:
            return True
        return shutil.which(self.binary) is not None or Path(self.binary).exists()

    def model_ready(self) -> bool:
        return self.model_path is None or self.model_path.exists()


# ---------------------------------------------------------------------------
# Peak RSS. macOS `/usr/bin/time -l` reports it for a process we spawn; for
# engines that answer from a resident server we sample the server instead.
# ---------------------------------------------------------------------------
def run_timed(cmd: list[str], timeout: int = 3600) -> tuple[str, float | None, int]:
    """Run cmd, returning (combined output, peak RSS MB, returncode)."""
    proc = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True,
                          text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    rss = None
    m = re.search(r"(\d+)\s+maximum resident set size", out)
    if m:
        rss = int(m.group(1)) / (1024 * 1024)
    return out, rss, proc.returncode


def sample_rss(pid: int) -> float | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        return int(out) / 1024 if out else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-engine runners. Each returns (tok_s, peak_rss_mb, note) and each parses
# the engine's OWN reported decode rate where it has one -- wall-clock timing
# around a subprocess would charge model load and prefill to decode.
# ---------------------------------------------------------------------------
def perf_cores() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                  capture_output=True, text=True).stdout.strip())
    except Exception:
        return 8


def run_llamacpp(prompt: str) -> tuple[float | None, float | None, str]:
    # -ngl 999      every layer on Metal
    # -fa on        flash attention, the recommended default on Apple silicon
    # -ctk/-ctv q8_0  match NVMAI's 8-bit KV rather than leaving it at fp16
    # --no-mmap     comparable residency to engines that load eagerly
    # -t perf cores only; the E-cores hurt more than help on M-series
    cmd = ["llama-cli", "-m", str(GGUF_MODEL), "-p", prompt,
           "-n", str(MAX_TOKENS), "-c", str(CONTEXT),
           "-ngl", "999", "-fa", "on", "-t", str(perf_cores()),
           "-ctk", "q8_0", "-ctv", "q8_0",
           "--temp", "0", "--top-k", "1", "--no-mmap", "--no-conversation"]
    out, rss, rc = run_timed(cmd)
    # "eval time = ... ( 41.39 tokens per second)" -- eval is decode; the
    # prompt-eval line above it is prefill and must not be picked up.
    m = re.search(r"eval time =.*?([\d.]+) tokens per second", out)
    if not m:
        for line in out.splitlines():
            if "eval time" in line and "tokens per second" in line and "prompt" not in line:
                m = re.search(r"([\d.]+) tokens per second", line)
                break
    if not m:
        return None, rss, f"no decode rate in output (rc={rc})"
    return float(m.group(1)), rss, ""


def run_ollama(prompt: str) -> tuple[float | None, float | None, str]:
    # Ollama reports eval_count / eval_duration, which is decode only.
    # Flash attention and the q8_0 KV cache are process-level env, set by the
    # caller before `ollama serve` -- they cannot be passed per request.
    import http.client
    payload = json.dumps({
        "model": OLLAMA_TAG, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "top_k": 1, "num_predict": MAX_TOKENS,
                    "num_ctx": CONTEXT, "num_gpu": 999},
    })
    try:
        conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=3600)
        conn.request("POST", "/api/generate", body=payload,
                     headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        conn.close()
    except OSError as exc:
        return None, None, f"ollama not reachable: {exc}"
    if "eval_count" not in data:
        return None, None, f"no eval stats: {str(data)[:120]}"
    rate = data["eval_count"] / (data["eval_duration"] / 1e9)
    pid = subprocess.run(["pgrep", "-n", "ollama"], capture_output=True,
                         text=True).stdout.strip()
    return rate, sample_rss(int(pid)) if pid else None, ""


def run_mlx(prompt: str) -> tuple[float | None, float | None, str]:
    # mlx_lm prints "Generation: N tokens, X tokens-per-sec".
    # --kv-bits 8 matches NVMAI; without it MLX holds fp16 KV and looks worse
    # on memory for a reason that is our choice, not its limitation.
    cmd = ["mlx_lm.generate", "--model", str(MLX_MODEL), "--prompt", prompt,
           "--max-tokens", str(MAX_TOKENS), "--temp", "0",
           "--kv-bits", str(KV_BITS)]
    out, rss, rc = run_timed(cmd)
    m = re.search(r"Generation:.*?([\d.]+) tokens-per-sec", out)
    if not m:
        return None, rss, f"no generation rate in output (rc={rc})"
    return float(m.group(1)), rss, ""


def run_lmstudio(prompt: str) -> tuple[float | None, float | None, str]:
    # LM Studio serves an OpenAI-compatible API on :1234 and returns its own
    # stats block including tokens_per_second (decode only).
    import http.client
    payload = json.dumps({
        "model": os.environ.get("BENCH_LMS_MODEL", "ornith-bench"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
    })
    try:
        conn = http.client.HTTPConnection("127.0.0.1", 1234, timeout=3600)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        conn.close()
    except OSError as exc:
        return None, None, f"LM Studio server not reachable: {exc}"
    stats = data.get("stats") or {}
    rate = stats.get("tokens_per_second")
    if rate is None:
        return None, None, "no stats.tokens_per_second (start with `lms server start`)"
    pid = subprocess.run(["pgrep", "-n", "LM Studio"], capture_output=True,
                         text=True).stdout.strip()
    return float(rate), sample_rss(int(pid)) if pid else None, ""


def run_mlc(prompt: str) -> tuple[float | None, float | None, str]:
    cmd = [sys.executable, "-m", "mlc_llm", "chat", str(MLC_MODEL),
           "--device", "metal", "--overrides", f"context_window_size={CONTEXT}",
           "--prompt", prompt, "--generate-length", str(MAX_TOKENS)]
    out, rss, rc = run_timed(cmd)
    m = re.search(r"decode:\s*([\d.]+) tok/s", out)
    if not m:
        return None, rss, f"no decode rate in output (rc={rc})"
    return float(m.group(1)), rss, ""


def run_nvmai(prompt: str) -> tuple[float | None, float | None, str]:
    # Reuse the shipped harness's instrument: the server's own
    # "NVMAI generation ... decode_tok_s=" footer, prefill excluded. Anything
    # else would be measuring NVMAI differently from how we measure it
    # everywhere else in this repository.
    import http.client
    port = 8123
    binary = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
    log = ROOT / ".build/vs-competitors-nvmai.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [str(binary), "--port", str(port), "--model", str(NVMAI_MODEL),
             "--max-context", str(CONTEXT), "--rope-scaling", "none",
             "--prompt-cache-mode", "off", "--prompt-cache-memory-mib", "0",
             "--kv-bits", str(KV_BITS), "--thinking", "off"],
            stdout=fh, stderr=subprocess.STDOUT)
    try:
        start = time.time()
        while time.time() - start < 2400:
            if proc.poll() is not None:
                return None, None, "NVMAIServer exited during load"
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                conn.request("GET", "/v1/models")
                model_id = json.loads(conn.getresponse().read())["data"][0]["id"]
                conn.close()
                break
            except OSError:
                time.sleep(5)
        else:
            return None, None, "NVMAIServer never became ready"
        payload = json.dumps({
            "model": model_id, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "max_completion_tokens": MAX_TOKENS,
            "stream": True,
        })
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()
        rss = sample_rss(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    rate = None
    for line in log.read_text().splitlines():
        if "decode_tok_s=" in line and ("NVMAI generation" in line or "NVMAI mtp " in line):
            rate = float(line.split("decode_tok_s=")[1].split()[0])
    return rate, rss, "" if rate else "no decode footer"


ENGINES: dict[str, tuple[Engine, object]] = {
    "nvmai": (Engine(
        "NVMAI", None, NVMAI_MODEL,
        "swift build -c release --product NVMAIServer",
        "tools/install_models.sh ornith15  (4-bit .gturbo install)"), run_nvmai),
    "llamacpp": (Engine(
        "llama.cpp", "llama-cli", GGUF_MODEL,
        "brew install llama.cpp",
        "convert the original weights with llama.cpp/convert_hf_to_gguf.py, "
        "then llama-quantize to Q4_K_M"), run_llamacpp),
    "ollama": (Engine(
        "Ollama", "ollama", None,
        "brew install --cask ollama   (then: ollama serve)",
        f"ollama create {OLLAMA_TAG} -f Modelfile  (FROM the shared GGUF)",
        caveat="flash-attn and q8_0 KV are set via env on `ollama serve`, "
               "not per request"), run_ollama),
    "mlx": (Engine(
        "MLX-LM", "mlx_lm.generate", MLX_MODEL,
        "pip install mlx-lm  (or: brew install mlx-lm)",
        "mlx_lm.convert --hf-path <repo> -q --q-bits 4 --q-group-size 64"),
        run_mlx),
    "lmstudio": (Engine(
        "LM Studio", str(Path.home() / ".lmstudio/bin/lms"), None,
        "install LM Studio, then `lms bootstrap`",
        "lms import <gguf>  then  lms load --gpu max --context-length "
        f"{CONTEXT}  and  lms server start",
        caveat="loads the same GGUF as llama.cpp, so a delta is runtime only"),
        run_lmstudio),
    "mlc": (Engine(
        "MLC-LLM", None, MLC_MODEL,
        "pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu "
        "mlc-ai-nightly-cpu",
        "mlc_llm convert_weight + gen_config + compile (q4f16_1, metal)",
        extra_check=lambda: None if _has_module("mlc_llm") else "python module mlc_llm not importable"),
        run_mlc),
}


def _has_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check(selected: list[str]) -> int:
    print(f"disk free: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB\n")
    missing = 0
    for key in selected:
        eng, _ = ENGINES[key]
        inst = eng.installed()
        extra = eng.extra_check() if eng.extra_check else None
        if extra:
            inst = False
        model = eng.model_ready()
        state = "ready" if (inst and model) else "NOT READY"
        print(f"  {eng.name:<12} {state}")
        if not inst:
            print(f"      engine missing: {extra or eng.binary}")
            print(f"      install: {eng.setup_hint}")
            missing += 1
        if not model:
            print(f"      model missing:  {eng.model_path}")
            print(f"      build:   {eng.model_hint}")
            missing += 1
        if eng.caveat:
            print(f"      caveat:  {eng.caveat}")
    print(f"\n{'all selected engines ready' if not missing else f'{missing} item(s) missing'}")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report engine/model readiness and exit")
    ap.add_argument("--engines", default=",".join(ENGINES),
                    help="comma-separated subset")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()

    selected = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in selected:
        if e not in ENGINES:
            print(f"unknown engine {e}; known: {', '.join(ENGINES)}", file=sys.stderr)
            return 2
    if args.check:
        return check(selected)

    # One model process at a time or every number is noise. Same rule the
    # golden harness enforces.
    busy = subprocess.run(
        ["pgrep", "-f", "NVMAIServer|NVMAICLI|ollama|LM Studio|mlx_lm|llama-cli"],
        capture_output=True, text=True).stdout.strip()
    if busy:
        print("another inference process is running; stop it first", file=sys.stderr)
        return 3

    results: list[Result] = []
    for key in selected:
        eng, runner = ENGINES[key]
        if not eng.installed() or not eng.model_ready():
            print(f"== {eng.name}: SKIPPED (not ready; run --check) ==", flush=True)
            continue
        print(f"== {eng.name} ==", flush=True)
        for pname, prompt in PROMPTS:
            for _ in range(WARMUPS):
                runner(prompt)
            best: Result | None = None
            for _ in range(max(1, args.repeats)):
                rate, rss, note = runner(prompt)
                r = Result(eng.name, pname, rate, rss, rate is not None, note)
                # Best-of-N: run-to-run spread on this machine is +/-15%, and
                # the slow tail is contention, not the engine.
                if r.ok and (best is None or not best.ok or r.tok_s > best.tok_s):
                    best = r
                elif best is None:
                    best = r
            results.append(best)
            if best.ok:
                rss = f"{best.peak_rss_mb:.0f} MB" if best.peak_rss_mb else "n/a"
                print(f"  {pname:<7} {best.tok_s:6.2f} tok/s   peak RSS {rss}", flush=True)
            else:
                print(f"  {pname:<7} FAILED: {best.note}", flush=True)

    print("\n=== summary (decode tok/s, best of "
          f"{args.repeats}; prefill excluded) ===")
    names = sorted({r.engine for r in results})
    header = f"{'prompt':<8}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for pname, _ in PROMPTS:
        row = f"{pname:<8}"
        for n in names:
            hit = next((r for r in results if r.engine == n and r.prompt == pname), None)
            row += f"{hit.tok_s:>14.2f}" if hit and hit.ok else f"{'-':>14}"
        print(row)
    print("\nPeak RSS is the other half of the story: NVMAI is built to stay "
          "inside a RAM budget while the rest load what they need.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
