#!/usr/bin/env python3
"""3-prompt sequential session benchmark - 8-bit quantized model.

One session, 3 prompts:
  1. "difference swift and c++ in detail"
  2. "tell me all about swift" (follow-up)
  3. "tell me all about c++" (follow-up)

Run: COND=0 python3 seq3_bench.py (or COND=1,2,3)

Produces benchmark-results/seq3-{cond-label}/aggregate.json
"""
import json, signal, subprocess, sys, time, os, urllib.request

ROOT = "/Users/andreborchert/Downloads/NVMAI"
COND = int(os.environ.get("COND", "0"))
MODEL = f"{ROOT}/scratch/qwen36-8bit.gturbo"
MODEL_ID = "qwen3.6-35b-a3b-8bit"
CACHE_MODE = "multi-prefix" if COND in (1, 3) else "off"
MTP_ON = COND in (2, 3)
TEMP = 0.0 if COND in (2, 3) else 0.2
LABELS = {0: "q8_cacheoff_mtpoff", 1: "q8_cacheon_mtpoff",
          2: "q8_cacheoff_mtp_on", 3: "q8_cacheon_mtp_on"}
LABEL = LABELS[COND]
OUTDIR = f"benchmark-results/seq3-{LABEL}-{time.strftime('%Y%m%dT%H%M%S')}"
os.makedirs(OUTDIR, exist_ok=True)

print(f"=== Sequential 3-Prompt Benchmark: Cond {COND} ({LABEL}) ===")
print(f"Cache={'on' if CACHE_MODE=='multi-prefix' else 'off'}, MTP={'on' if MTP_ON else 'off'}, Temp={TEMP}")
print()

# ---- Start Server ----
print("Starting server...")
args = [
    f"{ROOT}/.build/release/TurboFieldfareServer",
    "--model", MODEL,
    "--model-id", MODEL_ID,
    "--port", "8080",
    "--max-context", "4096",
    "--queue-limit", "4",
    "--prefill-chunk", "1024",
    "--prompt-cache-mode", CACHE_MODE,
]
if MTP_ON:
    args += ["--mtp-model", f"{ROOT}/scratch/qwen36-mtp.gturbo",
             "--mtp-memory-mib", "384"]
if CACHE_MODE == "multi-prefix":
    args += ["--prompt-cache-entries", "64",
             "--prompt-cache-memory-mib", "512",
             "--prompt-cache-disk", f"{OUTDIR}/cache",
             "--prompt-cache-disk-mib", "4096"]

proc = subprocess.Popen(
    args,
    stdout=open(f"{OUTDIR}/server.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)

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
print()

# ---- Send Request ----
def send_request(messages, seed):
    """Send a request and return (wall_s, prompt_tokens, completion_tokens, cached_tokens)."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMP,
        "top_k": 64,
        "top_p": 0.95,
        "seed": seed,
        "max_completion_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    if TEMP == 0.0:
        payload["repetition_penalty"] = 1.0

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

    return elapsed, pt, ct, cached, "".join(content)

# ---- Warmup ----
print("Running warmup...")
try:
    send_request([{"role": "user", "content": "Test"}], 999)
except:
    pass

# ---- Sequential 3-Prompt Session ----
print("\nRunning 3-prompt sequential session...")
print()

prompts = [
    "difference swift and c++ in detail",
    "tell me all about swift",
    "tell me all about c++",
]

histories = []
results = []
total_wall = 0.0

for i, prompt in enumerate(prompts):
    seed = 20260805 + i
    if not histories:
        # First prompt - no history
        messages = [{"role": "user", "content": prompt}]
    else:
        # Follow-up - include full history
        messages = histories[-1].copy()

    print(f"Prompt {i+1}/3: \"{prompt[:50]}...\"")
    wall, pt, ct, cached, response = send_request(messages, seed)

    if not wall:
        print(f"  FAILED")
        continue

    decode_rate = max(0, ct - 1) / max(0.001, wall)
    cached_pct = (cached / max(1, pt) * 100) if pt > 0 else 0

    mtp_tag = "[MTP]" if MTP_ON else "    "
    print(f"  [{mtp_tag}] wall={wall:.1f}s pt={pt} cached={cached}({cached_pct:.0f}%) "
          f"ct={ct} decode={decode_rate:.2f} tok/s")

    results.append({
        "prompt_idx": i + 1,
        "prompt": prompt,
        "wall_s": round(wall, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cached_tokens": cached,
        "cached_pct": round(cached_pct, 1),
        "decode_tok_s": round(decode_rate, 2)
    })

    total_wall += wall
    histories.append(messages.copy())
    # Add assistant placeholder
    histories[-1].append({"role": "assistant", "content": "..."})
    print()

# ---- Stop Server ----
print("Stopping server...")
if proc.poll() is None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
        proc.wait()

# ---- Aggregate Results ----
n = len(results)
print(f"\n{'='*60}")
print(f"SEQUENTIAL 3-PROMPT RESULTS: Condition {COND} ({LABEL})")
print(f"{'='*60}")
print(f"Requests completed: {n}/3")
print(f"Total wall time: {total_wall:.1f}s")

if total_wall > 0:
    avg_wall = total_wall / n
    total_ct = sum(r["completion_tokens"] for r in results)
    total_pt = sum(r["prompt_tokens"] for r in results)
    total_cached = sum(r["cached_tokens"] for r in results)
    decode_tok_s = max(0, total_ct - n) / total_wall
    e2e_tok_s = total_ct / total_wall
    cached_ratio = total_cached / max(1, total_pt) * 100
else:
    avg_wall = 0
    total_ct = 0
    total_pt = 0
    total_cached = 0
    decode_tok_s = 0
    e2e_tok_s = 0
    cached_ratio = 0

print(f"Total completion tokens: {total_ct}")
print(f"Total prompt tokens: {total_pt}")
print(f"Total cached tokens: {total_cached}")
print()
print(f"Aggregated Metrics:")
print(f"  Average wall time: {avg_wall:.1f}s")
print(f"  Decode throughput: {decode_tok_s:.2f} tok/s")
print(f"  End-to-end throughput: {e2e_tok_s:.2f} tok/s")
print(f"  Cached token ratio: {cached_ratio:.1f}%")
print()

# Save results
aggregate = {
    "condition": COND,
    "label": LABEL,
    "cache_mode": "on" if CACHE_MODE == "multi-prefix" else "off",
    "mtp_enabled": MTP_ON,
    "temperature": TEMP,
    "requests_completed": n,
    "total_wall_s": round(total_wall, 2),
    "avg_wall_s": round(avg_wall, 2),
    "total_completion_tokens": total_ct,
    "total_prompt_tokens": total_pt,
    "total_cached_tokens": total_cached,
    "decode_tok_s": round(decode_tok_s, 2),
    "e2e_tok_s": round(e2e_tok_s, 2),
    "cached_ratio_pct": round(cached_ratio, 1)
}

per_request = []
for r in results:
    per_request.append({
        "prompt": r["prompt"],
        "wall_s": r["wall_s"],
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "cached_tokens": r["cached_tokens"],
        "cached_pct": r["cached_pct"],
        "decode_tok_s": r["decode_tok_s"]
    })

with open(f"{OUTDIR}/aggregate.json", "w") as f:
    json.dump(aggregate, f, indent=2)
    f.write("\n")
with open(f"{OUTDIR}/per-request.json", "w") as f:
    json.dump(per_request, f, indent=2)
    f.write("\n")

print(f"\nResults saved to: {OUTDIR}/")
print(f"  aggregate.json    - overall metrics")
print(f"  per-request.json  - per-prompt breakdown")
print(f"  server.log        - server output")