#!/usr/bin/env python3
"""Long-generation decode benchmark (code-generation style): 512-token greedy
coding prompt, server-footer decode rates, 1 warmup + 3 measured runs per
quant. Usage: python3 benchmark/nvmai_longgen.py [model1] [model2] ...
"""
import http.client
import json
import os
import subprocess
import sys
import time

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
PORT = 8114
PROMPT = (
    "Write a Python function that computes the Levenshtein distance between "
    "two strings using dynamic programming with O(min(m,n)) space, including "
    "a detailed docstring and comments. Then add a main block that tests it "
    "on several pairs of strings and prints the results. Then write a "
    "second function that uses it to find the closest match to a target "
    "string in a list of candidates, and demonstrate it."
)
MAX_TOKENS = 512


def run_quant(model_path, label):
    env = dict(os.environ)
    env["NVMAI_RUNNER_STATS"] = "1"
    log_path = f"/tmp/longgen_{label}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", model_path, "--prompt-cache-mode", "off"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            return None
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

    def request():
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()

    request()  # warmup
    time.sleep(0.5)
    for _ in range(3):
        request()
        time.sleep(0.5)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(0.5)

    rates, cts = [], []
    with open(log_path) as f:
        for line in f:
            if "NVMAI generation" in line and "decode_tok_s=" in line:
                rates.append(float(line.split("decode_tok_s=")[1].split()[0]))
            if "completed in" in line and "completion=" in line:
                ct = int(line.split("completion=")[1].split()[0])
                cts.append(ct)
    return rates, cts


def main():
    models = sys.argv[1:] or [
        os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit"),
        os.path.join(BASE, "models", "qwen3.6_35B_A3B_6Bit"),
        os.path.join(BASE, "models", "qwen3.6_35B_A3B_8Bit"),
    ]
    for model in models:
        label = "4bit" if "6bit" not in model and "8bit" not in model else (
            "6bit" if "6bit" in model else "8bit")
        result = run_quant(model, label)
        if not result:
            print(f"{label}: FAILED", flush=True)
            continue
        rates, cts = result
        mean = sum(rates) / len(rates) if rates else 0
        print(f"{label}: rates={[f'{r:.2f}' for r in rates]} "
              f"mean={mean:.2f} tok/s ct={cts[1:] if len(cts) > 1 else cts}",
              flush=True)


if __name__ == "__main__":
    main()
