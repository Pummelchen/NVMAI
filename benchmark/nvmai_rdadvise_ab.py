#!/usr/bin/env python3
"""A/B: rdadvise default vs off on the overlap counters. Interleaved fresh
servers, 512-token greedy, warm-cache second request, server footers.
"""
import http.client
import json
import os
import subprocess
import time
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit")
PORT = 8113
PROMPT = "Write a detailed essay about the history of computing."


def run(mode):
    env = dict(os.environ)
    env["NVMAI_RUNNER_STATS"] = "1"
    env["NVMAI_KERNEL_STATS"] = "1"
    if mode == "off":
        env["NVMAI_RDADVISE_POLICY"] = "off"
    log_path = f"/tmp/nvmai_rd_{mode}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            print(f"{mode}: server exited early", file=sys.stderr)
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
    payload = json.dumps({
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "max_completion_tokens": 512, "stream": True,
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
    print(f"--- {mode} ---")
    for l in gen: print(l)
    for l in runner: print(l)
    for l in gpu: print(l)


for mode in ("default", "off"):
    run(mode)
