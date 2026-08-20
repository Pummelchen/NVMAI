#!/usr/bin/env python3
"""NVMAI tuning sweep for M5-class hardware.

Measures prefill seconds and decode tok/s across expert-cache slot counts and
prefill chunk sizes, using the CLI's timing footer. The two sweeps answer the
questions raised in the M5 feedback:

  slots  — does decode still climb past 32/64 on higher-bandwidth hardware?
  chunk  — is smaller or larger prefill better at your prompt length?

Each run loads the model fresh (cold page cache), so runs are minutes long:
the default sweeps are 6 slots + 5 chunks = 11 runs. Pick one sweep to halve
that, or trim the lists with --slots / --chunks.

Usage:
  python3 benchmark/nvmai_m5_sweep.py
  python3 benchmark/nvmai_m5_sweep.py --model ... --sweep slots
  python3 benchmark/nvmai_m5_sweep.py --model ... --sweep chunk --prompt-tokens 22800
  python3 benchmark/nvmai_m5_sweep.py --model ... --slots 32,64,128 --chunks 512,4096

Requires a release build: swift build -c release
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

from nvmai_profile import DEFAULT_CONTEXT_TOKENS, DEFAULT_MODEL_PATH

DEFAULT_SLOTS = [16, 24, 32, 64, 96, 128]
DEFAULT_CHUNKS = [256, 512, 1024, 2048, 4096]

# Mirrors RuntimeConfiguration.supportedContextTokens. The CLI rejects any
# other value, so the sweep picks from this list rather than passing an
# arbitrary size.
SUPPORTED_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Headroom over prompt + generated tokens, covering the ChatML wrapper and
# special tokens that make_prompt cannot account for.
CONTEXT_HEADROOM_TOKENS = 512

# A run that produces no output in this long is wedged; fail it and continue
# rather than hanging the whole sweep. Generous: a cold 128k-token prefill on
# slow hardware is legitimately many minutes.
RUN_TIMEOUT_S = 3600

# A repeatable ~50-word paragraph; tokens/word is roughly 1.3 for English, so
# 40 words ≈ 50 tokens. The script repeats it to reach the target token count.
PARAGRAPH = (
    "The server streams routed expert weights from the model store on demand, "
    "keeping only a resident working window, which is what lets it run within a "
    "bounded memory budget while retaining the full model quality at decode time. "
    "The chunked prefill path amortizes per-chunk route readbacks across long "
    "prompts, and the expert cache raises the hit rate for repeated generations. "
)


def make_prompt(target_tokens: int) -> str:
    words_per_block = len(PARAGRAPH.split())
    approx_tokens_per_block = int(words_per_block * 1.3)
    blocks = max(1, (target_tokens + approx_tokens_per_block - 1) // approx_tokens_per_block)
    return (PARAGRAPH * blocks).strip()


def validate_context(prompt_tokens: int, max_new: int, max_context: int) -> int:
    """Reject a selected context that cannot hold the estimated request."""
    need = prompt_tokens + max_new + CONTEXT_HEADROOM_TOKENS
    if max_context < need:
        raise SystemExit(
            "--prompt-tokens %d + --max-new %d needs ~%d tokens of context, "
            "above --max-context %d"
            % (prompt_tokens, max_new, need, max_context))
    return max_context


def run_once(cli: str, model: str, slots: int, chunk: int, prompt_file: str,
             max_new: int, max_context: int) -> dict:
    cmd = [cli, "--model", model,
           "--expert-cache-slots", str(slots),
           "--prefill-chunk", str(chunk),
           "--messages-file", prompt_file,
           "--max-context", str(max_context),
           "--rope-scaling", "none",
           "--kv-bits", "8",
           "--concise",
           "--max-new", str(max_new)]
    # Every return below carries slots/chunk, so a failed run is still
    # attributable to its configuration in the CSV.
    tags = {"slots": slots, "chunk": chunk}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {**tags, "error": "timed out after %ds" % RUN_TIMEOUT_S}
    elapsed = time.time() - t0
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr else "exit %d" % proc.returncode
        return {**tags, "error": detail}
    footer = proc.stderr
    m = re.search(r"prefill=(\d+)tok/([\d.]+)s new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)",
                  footer)
    if not m:
        return {**tags, "error": "no timing footer in output"}
    return {
        "slots": slots, "chunk": chunk,
        "prefill_tokens": int(m.group(1)), "prefill_s": float(m.group(2)),
        "new_tokens": int(m.group(3)), "decode_s": float(m.group(4)),
        "tok_per_s": float(m.group(5)),
        "wall_s": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--cli", default=None,
                    help="path to NVMAICLI (default: .build/arm64-apple-macosx/release/NVMAICLI)")
    ap.add_argument("--sweep", choices=["slots", "chunk", "both"], default="both")
    ap.add_argument("--slots", default=",".join(map(str, DEFAULT_SLOTS)))
    ap.add_argument("--chunks", default=",".join(map(str, DEFAULT_CHUNKS)))
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-context", type=int, choices=SUPPORTED_CONTEXTS,
                    default=DEFAULT_CONTEXT_TOKENS)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli = args.cli or os.path.join(base, ".build", "arm64-apple-macosx", "release", "NVMAICLI")
    if not os.path.isfile(cli):
        sys.exit("NVMAICLI not found at %s — run `swift build -c release` first" % cli)

    slots = [int(v) for v in args.slots.split(",")]
    chunks = [int(v) for v in args.chunks.split(",")]
    max_context = validate_context(args.prompt_tokens, args.max_new, args.max_context)
    prompt_directory = os.path.join(base, ".build", "benchmark-prompts")
    os.makedirs(prompt_directory, exist_ok=True)
    prompt_file = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=prompt_directory
    ).name
    with open(prompt_file, "w") as f:
        f.write(json.dumps([{"role": "user", "content": make_prompt(args.prompt_tokens)}]))

    print("NVMAI M5 sweep")
    print("  model: %s" % args.model)
    print("  prompt: ~%d tokens | max-new: %d | context: %d | first run is cold" % (
        args.prompt_tokens, args.max_new, max_context))
    print()

    results = []
    try:
        if args.sweep in ("slots", "both"):
            print("== slot sweep (chunk 4096) ==")
            for s in slots:
                r = run_once(cli, args.model, s, 4096, prompt_file, args.max_new,
                             max_context)
                results.append(r)
                if "error" in r:
                    print("  slots=%3d  ERROR: %s" % (s, r["error"]))
                else:
                    print("  slots=%3d  prefill %6.1fs (%4d tok)  decode %6.2f tok/s (%3d tok)  wall %5.1fs"
                          % (s, r["prefill_s"], r["prefill_tokens"], r["tok_per_s"], r["new_tokens"], r["wall_s"]))
            print()
        if args.sweep in ("chunk", "both"):
            print("== chunk sweep (slots 64) ==")
            for c in chunks:
                r = run_once(cli, args.model, 64, c, prompt_file, args.max_new,
                             max_context)
                results.append(r)
                if "error" in r:
                    print("  chunk=%4d  ERROR: %s" % (c, r["error"]))
                else:
                    print("  chunk=%4d  prefill %6.1fs (%4d tok)  decode %6.2f tok/s (%3d tok)  wall %5.1fs"
                          % (c, r["prefill_s"], r["prefill_tokens"], r["tok_per_s"], r["new_tokens"], r["wall_s"]))
            print()
    except KeyboardInterrupt:
        # A full sweep is 11 cold runs. Keep whatever completed rather than
        # discarding an hour of measurements on Ctrl-C.
        print("\ninterrupted — writing partial results")
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass

    # make_prompt sizes the prompt by a words-to-tokens estimate; the footer
    # reports what the tokenizer actually produced. Surface the gap so a sweep
    # is never silently run at a different length than requested.
    measured = [r["prefill_tokens"] for r in results if "prefill_tokens" in r]
    if measured:
        actual = measured[0]
        drift = abs(actual - args.prompt_tokens) / max(args.prompt_tokens, 1)
        note = "  (estimate off by %.0f%%)" % (drift * 100) if drift > 0.05 else ""
        print("prompt: requested ~%d tokens, tokenizer produced %d%s"
              % (args.prompt_tokens, actual, note))

    csv = "slots,chunk,prefill_s,prefill_tokens,tok_per_s,new_tokens,decode_s,wall_s,error\n"
    for r in results:
        if "error" in r:
            csv += "%s\n" % ",".join([str(r.get("slots", "")), str(r.get("chunk", "")),
                                      "", "", "", "", "", "", r["error"]])
        else:
            csv += "%d,%d,%.2f,%d,%.2f,%d,%.2f,%.1f,\n" % (
                r["slots"], r["chunk"], r["prefill_s"], r["prefill_tokens"],
                r["tok_per_s"], r["new_tokens"], r["decode_s"], r["wall_s"])
    # Alongside the other harnesses, in the git-ignored results directory —
    # the repo root is tracked, so writing there leaves the tree dirty.
    out_dir = os.path.join(base, "benchmark", "benchmark-results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "m5_sweep_results.csv")
    with open(out, "w") as f:
        f.write(csv)
    print("results written to %s" % out)


if __name__ == "__main__":
    main()
