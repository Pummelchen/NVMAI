#!/usr/bin/env python3
"""MTP 4-bit Matrix Benchmark - cache {off,on} x MTP {off,on}

Run: COND=0 python3 mtp_bench.py (or COND=1,2,3)
Produces benchmark-results/mtp-bench-{cond-label}/
"""
import json, signal, subprocess, sys, time, re, os, statistics, urllib.request
from pathlib import Path

ROOT = Path("/Users/andreborchert/Downloads/NVMAI")
COND = int(os.environ.get("COND", "0"))
CONFIG = os.environ.get("CONFIG", "8bit")
MODEL = str(ROOT / f"scratch/qwen36-{CONFIG}.gturbo")
MODEL_ID = f"qwen3.6-35b-a3b-{CONFIG}"
CACHE_MODE = "multi-prefix" if COND in (1, 3) else "off"
MTP_ON = COND in (2, 3)
TEMP = 0.0 if COND in (2, 3) else 0.6
LABELS = {0: "cacheoff_mtpoff", 1: "cacheon_mtpoff",
          2: "cacheoff_mtp_on", 3: "cacheon_mtp_on"}
LABEL = f"{CONFIG}_{LABELS[COND]}"
OUTDIR = ROOT / f"benchmark-results/mtp-bench-{LABEL}-{time.strftime('%Y%m%dT%H%M%S')}"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"=== MTP Benchmark: Condition {COND} ({LABEL}) ===")
print(f"Cache={'on' if CACHE_MODE == 'multi-prefix' else 'off'}, MTP={'on' if MTP_ON else 'off'}, Temp={TEMP}")

# ---- Start Server ----
print("Starting server...")
args = [
    str(ROOT / ".build/release/TurboFieldfareServer"),
    "--model", MODEL,
    "--model-id", MODEL_ID,
    "--port", "8080",
    "--max-context", "4096",
    "--queue-limit", "4",
    "--prefill-chunk", "1024",
    "--prompt-cache-mode", CACHE_MODE,
]
if MTP_ON:
    args += ["--mtp-model", str(ROOT / "scratch/qwen36-mtp.gturbo"), 
             "--mtp-memory-mib", "384"]
if CACHE_MODE == "multi-prefix":
    args += ["--prompt-cache-entries", "64", 
             "--prompt-cache-memory-mib", "512",
             "--prompt-cache-disk", str(OUTDIR / "cache"),
             "--prompt-cache-disk-mib", "4096"]

proc = subprocess.Popen(
    args, 
    stdout=open(OUTDIR / "server.log", "w"), 
    stderr=subprocess.STDOUT,
    start_new_session=True
)

# Wait for health
deadline = time.time() + 120
server_ready = False
while time.time() < deadline:
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if json.loads(resp.read())["status"] == "ok":
                server_ready = True
                break
    except:
        pass
    time.sleep(0.5)

if not server_ready:
    proc.kill()
    print("ERROR: Server failed to start")
    sys.exit(1)
print("Server ready")

# ---- Workload ----
user_prompts = [
    "Design a Swift actor-based bounded work queue. Show enqueue/dequeue. Under 80 tokens.",
    "Design a Swift async filesystem indexer for 1M files with bounded memory. Under 80 tokens.",
    "Implement a generic O(1) LRU cache in Swift. Under 80 tokens.",
    "C++23 RAII wrapper for POSIX socket. Under 80 tokens.",
    "Safe Swift-to-C++ FFI boundary for byte buffers. Under 80 tokens.",
    "Metal compute kernel for reducing Float array on M3. Under 80 tokens.",
    "Recursive-descent parser for arithmetic in Swift. Under 80 tokens.",
    "Swift CLI analyzing huge git diff. Under 80 tokens.",
    "Versioned SQLite migration runner in Swift. Under 80 tokens.",
    "Reconnecting SSE client in Swift with URLSession. Under 80 tokens.",
]

followup1 = [
    "Add cancellation without leaking continuations. Under 80 tokens.",
    "Add backpressure between dir walking and hashing. Under 80 tokens.",
    "Make concurrent-async-safe with an actor. Under 80 tokens.",
    "Add move construction and move assignment. Under 80 tokens.",
    "Clarify lifetime when C++ retains buffer async. Under 80 tokens.",
    "Remove threadgroup bank conflicts. Under 80 tokens.",
    "Add source-range diagnostics. Under 80 tokens.",
    "Handle quoted paths, renames, binary files. Under 80 tokens.",
    "Add rollback for failed migration. Under 80 tokens.",
    "Add Last-Event-ID, exponential backoff. Under 80 tokens.",
]

followup2 = [
    "Three Swift Testing cases. Under 80 tokens.",
    "Failure and retry policy for I/O. Under 80 tokens.",
    "Time/space complexity + invariant. Under 80 tokens.",
    "Three tests for double-close. Under 80 tokens.",
    "Minimal error ABI to Swift throws. Under 80 tokens.",
    "Fair benchmark against MPS. Under 80 tokens.",
    "Property-based fuzzing strategy. Under 80 tokens.",
    "Test streaming without invoking git. Under 80 tokens.",
    "Prevent two processes migrating. Under 80 tokens.",
    "Backpressure and cancellation propagation. Under 80 tokens.",
]

# ---- Send Request ----
def send_request(messages, seed):
    """Send a request and return (wall_s, prompt_tokens, completion_tokens, cached_tokens)."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMP,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "max_completion_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    start = time.time()
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    content = []
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                json_str = line[5:].strip()
                if json_str == "[DONE]":
                    break
                chunk = json.loads(json_str)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        content.append(text)
    except Exception as e:
        print(f"  Request failed: {e}")
        return None
    
    elapsed = time.time() - start
    if not usage:
        return None
    
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    
    return elapsed, pt, ct, cached

# ---- Run Benchmarks ----
print("Running warmup...")
try:
    send_request([{"role": "user", "content": "Test"}], 999)
except:
    pass

print("Running 30 requests...")
histories = [[{"role": "user", "content": p}] for p in user_prompts]
orders = [list(range(10)), list(reversed(range(10))), list(range(10))]
completed = 0
total_wall = 0.0
total_completion = 0
total_prompt = 0
total_cached = 0

for turn, order in enumerate(orders):
    for cid in order:
        if turn > 0:
            followup = followup1[cid] if turn == 1 else followup2[cid]
            histories[cid].append({"role": "user", "content": followup})
        
        seed = 20260803 + cid * 10 + turn
        result = send_request(histories[cid], seed)
        
        if not result:
            print(f"  req={turn*10+order.index(cid)+1:2d}/30 FAILED")
            continue
        
        wall, pt, ct, cached = result
        total_wall += wall
        total_completion += ct
        total_prompt += pt
        total_cached += cached
        
        decode_rate = max(0, ct - 1) / max(0.001, wall)
        req_num = turn * 10 + order.index(cid) + 1
        mtp_tag = "[MTP]" if MTP_ON else "    "
        print(f"  [{mtp_tag}] req={req_num:2d}/30 conv={cid+1} turn={turn} "
              f"pt={pt} cached={cached} ct={ct} wall={wall:.2f}s "
              f"decode={decode_rate:.2f} tok/s")
        
        histories[cid].append({"role": "assistant", "content": "x"})
        completed += 1

# ---- Stop Server ----
print("\nStopping server...")
if proc.poll() is None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
        proc.wait()

# ---- Aggregate Results ----
num_requests = completed
print(f"\n{'='*60}")
print(f"MTP BENCHMARK RESULTS: Condition {COND} ({LABEL})")
print(f"{'='*60}")
print(f"Requests completed: {num_requests}/30")
print(f"Total wall time: {total_wall:.1f}s")
print(f"Completion tokens: {total_completion}")
print(f"Prompt tokens: {total_prompt}")
print(f"Cached tokens: {total_cached}")

if total_wall > 0:
    avg_wall = total_wall / max(1, num_requests)
    decoded_tokens = max(0, total_completion - num_requests)
    decode_tok_s = decoded_tokens / total_wall
    e2e_tok_s = total_completion / total_wall
    cached_ratio = total_cached / max(1, total_prompt) * 100
else:
    avg_wall = 0
    decode_tok_s = 0
    e2e_tok_s = 0
    cached_ratio = 0

print(f"\nAggregated Metrics:")
print(f"  Average wall time: {avg_wall:.2f}s")
print(f"  Decode throughput: {decode_tok_s:.2f} tok/s")
print(f"  End-to-end throughput: {e2e_tok_s:.2f} tok/s")
print(f"  Cached token ratio: {cached_ratio:.1f}%")

# Save results
results = {
    "condition": COND,
    "label": LABEL,
    "cache_mode": CACHE_MODE,
    "mtp_enabled": MTP_ON,
    "temperature": TEMP,
    "requests_completed": num_requests,
    "total_wall_s": round(total_wall, 2),
    "avg_wall_s": round(avg_wall, 2),
    "total_completion_tokens": total_completion,
    "total_prompt_tokens": total_prompt,
    "total_cached_tokens": total_cached,
    "decoded_tokens": decoded_tokens if total_wall > 0 else 0,
    "decode_tok_s": round(decode_tok_s, 2),
    "e2e_tok_s": round(e2e_tok_s, 2),
    "cached_ratio_pct": round(cached_ratio, 1)
}

with open(OUTDIR / "aggregate.json", "w") as f:
    json.dump(results, f, indent=2)
    f.write("\n")

print(f"\nResults saved to: {OUTDIR}/")
print(f"Server log: {OUTDIR}/server.log")