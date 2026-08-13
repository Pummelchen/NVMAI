#!/usr/bin/env python3
"""Peak-throughput probe: decode rate as a function of expert-cache locality.
Prompts rank from diverse routing (code) to maximally repetitive (digit
cycles). 512-token greedy, server-footer rates, warmup + 2 measured runs per
prompt. Usage: python3 benchmark/nvmai_maxthroughput.py
"""
import http.client
import json
import os
import subprocess
import sys
import time

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit")
PORT = 8117
PROMPTS = [
    ("code", "Write a Python function that computes the Levenshtein distance between two strings with a detailed docstring, then a second function using it to find the closest match in a list, with a demo main."),
    ("essay", "Write a detailed essay about the history of computing."),
    ("count", "Count from 1 to 1000, writing only the numbers separated by single spaces."),
    ("digits", "Write the digits 1,2,3,4,5,6,7,8,9,0 over and over in sequence, separated by commas, without stopping."),
]


def main():
    models = sys.argv[1:] or [os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit")]
    for model in models:
        label = "4bit" if "6bit" not in model and "8bit" not in model else (
            "6bit" if "6bit" in model else "8bit")
        run_quant(model, label)


def run_quant(model, label):
    log_path = f"/tmp/maxtput_{label}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", model, "--prompt-cache-mode", "off"],
        stdout=log, stderr=subprocess.STDOUT)
    start = time.time()
    while time.time() - start < 120:
        if proc.poll() is not None:
            raise SystemExit("server failed to start")
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

    def request(prompt):
        payload = json.dumps({
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_completion_tokens": 512, "stream": True,
        }).encode()
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        while resp.read(8192):
            pass
        conn.close()

    for _, prompt in PROMPTS:
        request(prompt)  # warmup
        time.sleep(0.3)
        request(prompt)  # measured
        time.sleep(0.3)

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
                cts.append(int(line.split("completion=")[1].split()[0]))

    idx = 0
    for pname, _ in PROMPTS:
        r = rates[idx:idx + 2]
        c = cts[idx:idx + 2]
        if len(r) == 2:
            print(f"{label} {pname}: measured={r[1]:.2f} (warmup {r[0]:.2f}) "
                  f"ct={c}", flush=True)
        else:
            print(f"{label} {pname}: missing rate footer (request failed?)", flush=True)
        idx += 2


if __name__ == "__main__":
    main()
