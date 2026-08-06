#!/usr/bin/env python3
"""Precise 2x2 benchmark with streaming and TTFT tracking."""
import json, time, http.client, os, sys

MODEL_ID = "qwen3.6-35b-a3b"

PROMPTS = [
    ("Basic fact", "What is the capital of France? Answer with only the city.", "Paris"),
    ("Arithmetic", "Calculate 17 × 24. Answer with only the number.", "408"),
    ("Instruction following", "Return exactly the word BLUE in uppercase.", "BLUE"),
    ("Classification", 'Classify as positive, negative, or neutral: "The product works as expected."', "neutral"),
    ("Extraction", "Extract the email address: Contact Ana at ana@example.com tomorrow.", "ana@example.com"),
    ("Structured output", 'Return JSON with keys "name" and "age" for: Maya is 31.', "Valid JSON"),
    ("Logic", "All bloops are razzies. No razzies are lazzies. Can a bloop be a lazzy? Answer yes or no.", "no"),
    ("Context use", "Context: The access code is 7391. What is the access code? Answer only with the code.", "7391"),
    ("Refusal/honesty", "What is the access code? Do not guess if none was provided.", "States not provided"),
    ("Summarization", "Summarize in five words: The server failed because its disk was full.", "Semantic scoring"),
    ("Transformation", "Convert to lowercase: Hello WORLD 123!", "hello world 123!"),
    ("Simple coding", "Write a Python expression that returns the largest value in nums.", "max(nums)"),
]

MAX_TOKENS = 128

def send_request_stream(messages, port=8080):
    """Stream request tracking TTFT using incremental read."""
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True}
    }, separators=(",", ":")).encode()
    
    start = time.time()
    ttft = None
    
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    
    # Read incrementally to detect first token
    content = []
    usage = None
    first_content = False
    buf = ""
    
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="ignore")
        
        # Process SSE lines
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            json_str = line[5:].strip()
            if json_str == "[DONE]":
                conn.close()
                break
            try:
                chunk_data = json.loads(json_str)
                if chunk_data.get("usage"):
                    usage = chunk_data["usage"]
                for choice in chunk_data.get("choices", []):
                    delta = choice.get("delta") or {}
                    text_val = delta.get("content")
                    if text_val:
                        if not first_content:
                            ttft = time.time() - start
                            first_content = True
                        content.append(text_val)
            except:
                pass
        if json_str == "[DONE]" if 'json_str' in dir() else False:
            break
    
    conn.close()
    wall = time.time() - start
    
    if not usage:
        return None
    if ttft is None:
        ttft = wall
    
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    
    return wall, ttft, pt, ct, cached, "".join(content)

def run_config(cache_mode, mtp_config, config_label, port=8080):
    print(f"\n{'#'*110}", flush=True)
    print(f"# BENCHMARK: {config_label}", flush=True)
    print(f"# Cache: {cache_mode}, MTP: {mtp_config}, Port: {port}", flush=True)
    print(f"{'#'*110}", flush=True)

    print(f"\n>>> Warming up...", flush=True)
    send_request_stream([{"role": "user", "content": "Test"}], port=port)
    time.sleep(0.5)
    
    results = []
    print(f"\n{'#':>3s} {'Capability':<22} {'Wall':>6s} {'TTFT':>6s} {'Decode':>8s} {'PT':>4s} {'CT':>4s} "
          f"{'Cache':>6s} {'Decode':>10s}", flush=True)
    print("-" * 100, flush=True)
    
    warm_rates = []
    for i, (cap_name, prompt_text, _) in enumerate(PROMPTS):
        result = send_request_stream([{"role": "user", "content": prompt_text}], port=port)
        if not result:
            print(f"{i+1:3d} {cap_name:<22s} FAILED", flush=True)
            continue
        
        wall, ttft, pt, ct, cached, response = result
        decode_time = wall - ttft
        decode_rate = ct / decode_time if decode_time > 0 else 0
        cached_pct = (cached / pt * 100) if pt > 0 else 0
        
        # Only count as warm if decode_time > 0.5s (filter out instant 2-token responses)
        if i > 0 and decode_time > 0.5 and decode_rate > 0:
            warm_rates.append(decode_rate)
        
        results.append({"capability": cap_name, "wall": wall, "ttft": ttft,
            "decode_time": decode_time, "prompt_tokens": pt,
            "completion_tokens": ct, "cached_tokens": cached,
            "decode_rate": decode_rate})
        
        tag = " *" if i == 0 else (" (filtered)" if decode_time <= 0.5 else "")
        print(f"{i+1:3d} {cap_name:<22s} {wall:6.2f} {ttft:6.2f} {decode_time:8.2f} {pt:4d} {ct:4d} "
              f"{cached_pct:5.1f}% {decode_rate:9.2f}{tag}", flush=True)
    
    print("-" * 100, flush=True)
    
    total_wall = sum(r['wall'] for r in results)
    total_ct = sum(r['completion_tokens'] for r in results)
    
    print(f"\n* Cold start request", flush=True)
    
    if warm_rates:
        avg_warm = sum(warm_rates) / len(warm_rates)
        last6_avg = sum(warm_rates[-6:]) / 6
        print(f"WARM DECODE: {avg_warm:.2f} tok/s ({len(warm_rates)} requests)")
        print(f"  Last 6 avg: {last6_avg:.2f} tok/s")
        print(f"  Individual: {' | '.join(f'{r:.2f}' for r in warm_rates)}", flush=True)
    else:
        avg_warm = 0
        last6_avg = 0
    
    overhead = total_wall - (total_ct / avg_warm if avg_warm > 0 else 0)
    print(f"\nWall: {total_wall:.1f}s | Decode only: {total_ct/avg_warm:.1f}s | Overhead: {overhead:.1f}s ({overhead/len(results):.2f}s/req)", flush=True)
    
    ts = time.strftime('%Y%m%dT%H%M%S')
    outdir = f"benchmark-results/bench-{config_label}-{ts}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "aggregate.json"), "w") as f:
        json.dump({"config": config_label, "cache_mode": cache_mode, "mtp_config": mtp_config,
            "results": results, "summary": {"total_requests": len(results),
            "total_wall_s": round(total_wall, 2), "total_completion_tokens": total_ct,
            "avg_decode_tok_s": round(avg_warm, 2), "warm_rates": [round(r, 2) for r in warm_rates],
            "last6_avg": round(last6_avg, 2)}}, f, indent=2)
    print(f"\nSaved: {outdir}/aggregate.json")
    return avg_warm

if __name__ == "__main__":
    # Accept model path from command-line argument
    base_dir = os.environ.get("NVMAI_DIR", "/Users/andreborchert/Downloads/NVMAI")
    main_model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "models", "qwen36.gturbo")
    mtp_model = os.path.join(base_dir, "models", "qwen36-mtp.gturbo")

    # Detect quantization from model path for labeling
    quant_label = "4bit"
    if "6bit" in main_model:
        quant_label = "6bit"
    elif "8bit" in main_model:
        quant_label = "8bit"

    configs = [
        ("off", "off", f"cache_off_mtp_off_{quant_label}", 8080),
        ("multi-prefix", "off", f"cache_on_mtp_off_{quant_label}", 8081),
        ("off", "on", f"cache_off_mtp_on_{quant_label}", 8082),
        ("multi-prefix", "on", f"cache_on_mtp_on_{quant_label}", 8083),
    ]

    print(f"\n{'#'*110}", flush=True)
    print(f"# BENCHMARKING MODEL: {os.path.basename(main_model)}", flush=True)
    print(f"# Quantization: {quant_label}", flush=True)
    print(f"{'#'*110}", flush=True)

    for cache_mode, mtp_config, config_label, port in configs:
        cmd = f'cd {base_dir} && .build/arm64-apple-macosx/release/NVMAIServer --port {port} --model {main_model} --prompt-cache-mode {cache_mode}'
        if mtp_config == "on":
            cmd += f' --mtp-model {mtp_model} --mtp-memory-mib 384'
        cmd += f' > /tmp/nvmaiserver_{port}.log 2>&1 &'
        os.system(cmd)
        print(f"Port {port} launched, waiting 25s...", flush=True)
        time.sleep(25)

        ready = False
        for _ in range(10):
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", "/health")
                if "ok" in c.getresponse().read().decode():
                    ready = True; break
                c.close()
            except: pass
            time.sleep(10)

        if not ready:
            print(f"Port {port} FAILED!", flush=True); os.system(f"pkill -f 'NVMAIServer.*{port}'"); continue

        run_config(cache_mode, mtp_config, config_label, port)
        os.system(f"pkill -f 'NVMAIServer.*{port}'")
        if port < 8083: time.sleep(10)

    print(f"\n{'='*110}\nALL PRECISE 2x2 COMPLETE\n{'='*110}", flush=True)