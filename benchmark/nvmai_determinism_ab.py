#!/usr/bin/env python3
"""Cross-process greedy determinism check: two fresh servers, 512-token greedy
for the essay and digit-cycle prompts, comparing the CONTENT deltas token by
token (never the raw SSE bytes). A raw-byte comparison is invalid: each
response embeds a per-request chatcmpl id and created timestamp, so identical
content hashes differently across processes. first_diff_index=None means the
streams are identical token-for-token.

Usage: python3 benchmark/nvmai_determinism_ab.py [max_tokens]
"""
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(BASE, ".build", "arm64-apple-macosx", "release", "NVMAIServer")
MODEL = os.path.join(BASE, "models", "qwen3.6_35B_A3B_4Bit")
PORT = 8114
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
PROMPTS = [
    ("essay", "Write a detailed essay about the history of computing."),
    ("digits", "Write the digits 1,2,3,4,5,6,7,8,9,0 over and over in sequence, separated by commas, without stopping."),
]


def extract_deltas(resp):
    """Content deltas only — the raw bytes contain the per-request chatcmpl id
    and created timestamp, so hashing/compare the stream itself is invalid."""
    deltas = []
    for raw in resp:
        line = raw.decode()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content is not None:
                deltas.append(content)
    return deltas


def first_diff_index(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def run_server():
    log = open("/tmp/nvmai_determinism_server.log", "w")
    proc = subprocess.Popen(
        [BIN, "--port", str(PORT), "--model", MODEL, "--prompt-cache-mode", "off"],
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

    result = {}
    for name, prompt in PROMPTS:
        payload = json.dumps({
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0, "max_completion_tokens": MAX_TOKENS, "stream": True,
        }).encode()
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        deltas = extract_deltas(resp)
        conn.close()
        result[name] = deltas
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(0.5)
    return result


def main():
    print(f"max_tokens={MAX_TOKENS} prompts={[n for n, _ in PROMPTS]}")
    print("server A...", flush=True)
    a = run_server()
    print("server B...", flush=True)
    b = run_server()
    for name, _ in PROMPTS:
        da, db = a[name], b[name]
        diff = first_diff_index(da, db)
        joined_a = "".join(da)
        joined_b = "".join(db)
        sha_a = hashlib.sha256(joined_a.encode()).hexdigest()[:16]
        sha_b = hashlib.sha256(joined_b.encode()).hexdigest()[:16]
        print(f"{name}: tokens A={len(da)} B={len(db)} "
              + f"first_diff_index={diff} "
              + f"content_sha256={sha_a}=={sha_b} equal={sha_a == sha_b}")
    print("None for both => 512-token greedy is deterministic across fresh processes")


if __name__ == "__main__":
    main()
