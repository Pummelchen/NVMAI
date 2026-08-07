#!/usr/bin/env python3
"""Long-context decode measurement: KV pressure at ~10k tokens + long generation.

A: ~10k-token prompt (built from /tmp/prompt_10k.txt if present, else generated
   inline), 128 max new -> KV pressure (prefill_s, decode, RSS)
B: short prompt, 1024 max new     -> long-generation sustained decode
C: ~10k-token prompt, 1024 max new -> combined

Decode rates come from the server footer (authoritative); per-chunk timestamps
give the decode-rate drift over the generation. Server RSS sampled around the
10k-token prefill. Usage: python3 benchmark/nvmai_longctx.py
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import http.client

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.path.join(BASE, "models", "qwen36.gturbo")
PORT = 8091
def build_prompt():
    try:
        return open("/tmp/prompt_10k.txt").read()
    except OSError:
        para = ("The NVMAI inference engine executes Qwen 3.6 35B-A3B on Apple "
                "Silicon via Metal with a 40-layer mixture-of-experts network "
                "and routed-expert pread streaming. This paragraph repeats to "
                "build a long context for decode-pressure measurement. ")
        return "Continue the technical discussion: " + para * 40

PROMPT_10K = build_prompt()
SHORT = "Write a detailed essay about the history of computing."


def launch_server():
    log = open("/tmp/longctx_server.log", "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                return proc
            conn.close()
        except OSError:
            pass
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server not ready")


def rss_mb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip()) / 1024
    except Exception:
        return None


def request(messages, max_new, label):
    """Stream one request; returns dict incl. chunk_times (t_since_start,
    cum_chars) per recv."""
    payload = json.dumps({
        "model": "qwen3.6-35b-a3b",
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": max_new,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
    start = time.time()
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    buf = ""
    usage = None
    content = []
    ttft = None
    chunk_times = []
    cum = 0
    first = False
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        now = time.time() - start
        buf += chunk.decode("utf-8", errors="ignore")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            js = line[5:].strip()
            if js in ("[DONE]", ""):
                continue
            try:
                d = json.loads(js)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                t = (ch.get("delta") or {}).get("content")
                if t:
                    if not first:
                        ttft = now
                        first = True
                    content.append(t)
                    cum += len(t)
        if cum and (not chunk_times or chunk_times[-1][1] != cum):
            chunk_times.append((now, cum))
    conn.close()
    wall = time.time() - start
    ct = int(usage["completion_tokens"]) if usage else 0
    pt = int(usage["prompt_tokens"]) if usage else 0
    print(f"  [{label}] wall={wall:.1f}s ttft={ttft:.1f}s pt={pt} ct={ct} "
          f"client_decode={ct/(wall-ttft):.2f} tok/s", flush=True)
    return {"wall": wall, "ttft": ttft, "pt": pt, "ct": ct,
            "chunk_times": chunk_times}


def rate_at(times, frac):
    """decode tok/s over the last `frac` of the generation (by chars)."""
    if not times:
        return None
    total = times[-1][1]
    threshold = total * (1 - frac)
    start_t = next((t for t, c in times if c >= threshold), times[0][0])
    seg = [(t, c) for t, c in times if t >= start_t]
    if len(seg) < 2:
        return None
    chars = seg[-1][1] - seg[0][1]
    dt = seg[-1][0] - seg[0][0]
    return (chars / dt) / 4.0 if dt > 0 else None  # ~4 chars/token


def main():
    proc = launch_server()
    print(f"server pid {proc.pid}", flush=True)
    time.sleep(1)
    base_rss = rss_mb(proc.pid)

    # A: KV pressure - 10k prompt, short generation
    r = request([{"role": "user", "content": PROMPT_10K}], 128, "A:10k+128")
    post_a_rss = rss_mb(proc.pid)

    # B: long generation - short prompt, 1024 tokens
    r = request([{"role": "user", "content": SHORT}], 1024, "B:short+1024")
    b_times = r["chunk_times"]

    # C: combined - 10k prompt, 1024 tokens
    r = request([{"role": "user", "content": PROMPT_10K}], 1024, "C:10k+1024")
    c_times = r["chunk_times"]

    post_c_rss = rss_mb(proc.pid)

    print(f"\nRSS: base={base_rss:.0f}MB afterA={post_a_rss:.0f}MB "
          f"afterC={post_c_rss:.0f}MB", flush=True)
    for name, times in (("B", b_times), ("C", c_times)):
        if len(times) >= 4:
            first_half = rate_at(times[:max(2, len(times)//2)], 0.5)
            last_half = rate_at(times, 0.5)
            print(f"{name} decode drift: first-half={first_half:.2f} tok/s "
                  f"last-half={last_half:.2f} tok/s", flush=True)

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    print("\n--- server footers ---", flush=True)
    for line in open("/tmp/longctx_server.log"):
        if "NVMAI generation" in line or "completed in" in line:
            print(line.strip(), flush=True)


if __name__ == "__main__":
    main()
