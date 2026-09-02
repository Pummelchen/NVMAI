#!/usr/bin/env python3
"""A/B: serial vs NVMAI_PARALLEL_IO=1 expert pread fills. Interleaved fresh
servers; IO wall + decode rate from the request footers (512-token greedy).
"""
import http.client
import json
import os
import subprocess
import time

from nvmai_profile import (
    DEFAULT_API_MODEL, DEFAULT_MODEL_PATH, benchmark_log_path,
    server_command, server_environment, resolve_api_model,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = str(DEFAULT_MODEL_PATH)
PORT = 8110
PROMPT = "Write a detailed essay about the history of computing."


def run(mode):
    env = server_environment()
    if mode == "serial":
        env["NVMAI_PARALLEL_IO"] = "0"
    env["NVMAI_RUNNER_STATS"] = "1"
    log_path = benchmark_log_path(f"ioab_{mode}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        server_command(BIN, PORT, model=MODEL),
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
        "model": resolve_api_model(PORT),
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 0.0, "max_completion_tokens": 512, "stream": True,
    }).encode()
    for i in range(3):
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
    time.sleep(0.5)
    rates, ios = [], []
    with open(log_path) as f:
        for line in f:
            if "NVMAI generation" in line and "decode_tok_s=" in line:
                rates.append(float(line.split("decode_tok_s=")[1].split()[0]))
            if "NVMAI runner" in line and "io_ms=" in line:
                ios.append(float(line.split("io_ms=")[1].split()[0]))
    return rates, ios


def main():
    for mode in ("serial", "parallel"):
        print(f"{mode}:", run(mode))
    # interleaved second round
    for mode in ("serial", "parallel"):
        print(f"{mode} r2:", run(mode))


if __name__ == "__main__":
    main()
