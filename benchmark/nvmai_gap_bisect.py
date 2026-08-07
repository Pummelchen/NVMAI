#!/usr/bin/env python3
"""Bisect the unmeasured wall: length sweep (fixed vs per-token overhead) and
an rdadvise-off comparison, using authoritative server footers.
"""
import http.client
import json
import os
import subprocess
import time
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.path.join(BASE, "models", "qwen36.gturbo")
PORT = 8112
PROMPT = "Write a detailed essay about the history of computing."


def run(max_tokens, tag, extra_env=None):
    env = dict(os.environ)
    env["NVMAI_RUNNER_STATS"] = "1"
    env["NVMAI_KERNEL_STATS"] = "1"
    if extra_env:
        env.update(extra_env)
    log_path = f"/tmp/nvmai_gap_{tag}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            print(f"{tag}: server exited early", file=sys.stderr)
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
        "temperature": 0, "max_completion_tokens": max_tokens, "stream": True,
    }).encode()
    # one warm request then the measured one
    for i in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
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
    out = {"gen": [], "runner": [], "gpu": []}
    with open(log_path) as f:
        for line in f:
            if "NVMAI generation" in line and "decode_tok_s=" in line:
                out["gen"].append(line.strip())
            if "NVMAI runner" in line and "cb1_ms=" in line:
                out["runner"].append(line.strip())
            if "NVMAI kernel role=" in line:
                out["gpu"].append(line.strip())
            if "NVMAI kernel total_gpu_ms=" in line:
                out["gpu"].append(line.strip())
    print(f"--- {tag} (max_tokens={max_tokens}) ---")
    for l in out["gen"]: print(l)
    for l in out["runner"]: print(l)
    for l in out["gpu"]: print(l)


run(128, "len128")
run(1024, "len1024")
