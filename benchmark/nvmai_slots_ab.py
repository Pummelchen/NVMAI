#!/usr/bin/env python3
"""A/B: expert-cache slot count (the 8 GB RAM lever). 32 vs 128 slots/layer
(128 x 40 x ~1.55 MiB ~= 7.9 GB), 512-token greedy essay, server footers.
io_ms is the decisive counter: a higher hit rate must shrink the pread wall
on the critical path (the pread window is GPU-idle except shared+phase-1).
Usage: python3 benchmark/nvmai_slots_ab.py [slots...]
"""
import http.client
import json
import os
import subprocess
import sys
import time

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.environ.get("NVMAI_BENCH_MODEL",
                       os.path.join(BASE, "models", "ornith-1.5_35B_A3B_4Bit"))
PORT = 8115
PROMPT = "Write a detailed essay about the history of computing."
MAX_TOKENS = int(os.environ.get("NVMAI_AB_TOKENS", "512"))


def run(slots, pin=None):
    env = dict(os.environ)
    env["NVMAI_RUNNER_STATS"] = "1"
    env["NVMAI_KERNEL_STATS"] = "1"
    env["NVMAI_EXPERT_CACHE_SLOTS"] = str(slots)
    if pin is not None:
        if pin:
            env.pop("NVMAI_NO_PIN", None)
        else:
            env["NVMAI_NO_PIN"] = "1"
    log_path = f"/tmp/nvmai_slots_{slots}{'_pin' if pin else ''}.log"
    log = open(log_path, "w")
    boot = time.time()
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            print(f"slots={slots}: server exited early", file=sys.stderr)
            sys.exit(1)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                break
            conn.close()
        except OSError:
            pass
        time.sleep(0.05)
    boot_s = time.time() - boot
    print(f"--- slots={slots} pin={pin} boot_s={boot_s:.1f} ---")
    payload = json.dumps({
        "model": "ornith-1.5-35b-a3b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 0.0, "max_completion_tokens": MAX_TOKENS, "stream": True,
    }).encode()
    for i in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(0.3)
    gen, runner, gpu = [], [], []
    with open(log_path) as f:
        for line in f:
            if "NVMAI generation" in line and "decode_tok_s=" in line:
                gen.append(line.strip())
            if "NVMAI runner" in line and "cb1_ms=" in line:
                runner.append(line.strip())
            if "NVMAI kernel total_gpu_ms=" in line:
                gpu.append(line.strip())
    for l in gen: print(l)
    for l in runner: print(l)
    for l in gpu: print(l)


def main():
    args = sys.argv[1:]
    if args and args[0] in ("pin", "nopin"):
        mode = args[0]
        slot_list = [int(x) for x in args[1:]] or [32, 64]
        for slots in slot_list:
            run(slots, pin=(mode == "pin"))
        return
    slots_list = [int(x) for x in args] or [32, 128]
    for slots in slots_list:
        run(slots)


if __name__ == "__main__":
    main()
