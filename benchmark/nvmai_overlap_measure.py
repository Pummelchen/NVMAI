#!/usr/bin/env python3
"""GPU/IO overlap analysis: per-token stage splits (NVMAI_RUNNER_STATS) and
per-role GPU spans (NVMAI_KERNEL_STATS) on a warm cache. Two 512-token greedy
requests; the second is the representative warm measurement.
"""
import http.client
import json
import os
import subprocess
import time
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.environ.get("NVMAI_BENCH_MODEL",
                       os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit"))
PORT = 8111
PROMPT = "Write a detailed essay about the history of computing."
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512

env = dict(os.environ)
env["NVMAI_RUNNER_STATS"] = "1"
env["NVMAI_KERNEL_STATS"] = "1"
log_path = "/tmp/nvmai_overlap.log"
log = open(log_path, "w")
proc = subprocess.Popen(
    [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
    env=env, stdout=log, stderr=subprocess.STDOUT)
start = time.time()
while time.time() - start < 120:
    if proc.poll() is not None:
        print("server exited early", file=sys.stderr)
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
    print(f"request {i+1} done")
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()

with open(log_path) as f:
    lines = f.readlines()
gen, runner, kernels = [], [], []
for line in lines:
    if "NVMAI generation" in line and "decode_tok_s=" in line:
        gen.append(line.strip())
    if "NVMAI runner" in line and "cb1_ms=" in line:
        runner.append(line.strip())
    if "NVMAI kernel role=" in line:
        kernels.append(line.strip())
    if "NVMAI kernel total_gpu_ms=" in line:
        kernels.append(line.strip())
print("=== generation footers ===")
for l in gen: print(l)
print("=== runner stage splits ===")
for l in runner: print(l)
print("=== kernel GPU roles (per request) ===")
for l in kernels: print(l)
