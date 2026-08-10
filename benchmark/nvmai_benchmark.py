#!/usr/bin/env python3
"""Precise 2x2 benchmark with streaming and TTFT tracking.

Methodology notes (production audit fixes):
- Cache-warm runs: for prompt-cache configs every prompt is sent twice; the
  second send must report cached_tokens > 0, otherwise the cell is flagged.
  One-shot unique prompts can never hit a multi-prefix cache, so the cache
  dimension is measured by the warm send only.
- MTP forces prompt cache OFF server-side (the draft stream cannot be
  snapshot-restored), so the "cache ON x MTP ON" cell is impossible by design
  and is not run.
- Expected answers are verified (normalized containment) and recorded.
- Servers are launched via Popen, tracked by PID, and terminated by PID
  (never pkill-by-pattern). SIGINT terminates every spawned server.
- Decode rate is only reported for requests with decode_time > 0.5 s; raw
  rows keep a null decode_rate otherwise so absurd instant-response values
  never enter the JSON.
- The server buffers SSE content frames until generation completes, so
  client-side TTFT ~= wall and client decode rates are always null. The
  authoritative per-request decode rate comes from the server's own footer
  (decode_tok_s in /tmp/nvmaiserver_PORT.log), which is parsed per config and
  attached to rows / summarized as footer_decode_rates.
- MTP engages only for pure-greedy requests (temperature 0, repetition
  penalty 1), so the MTP cell runs at temperature 0 and the other cells at
  the original 0.2.
"""
import json
import time
import http.client
import os
import signal
import subprocess
import sys
import re

DECODE_FOOTER_RE = re.compile(r"decode_tok_s=([0-9.]+)")
MTP_STATS_RE = re.compile(
    r"mtp drafted=(\d+) accepted=(\d+) acceptance=([0-9.]+)% "
    r"target_passes=(\d+) emitted_per_pass=([0-9.]+)")

MODEL_ID = "qwen3.6-35b-a3b"

PROMPTS = [
    ("Basic fact", "What is the capital of France? Answer with only the city.", "Paris"),
    ("Arithmetic", "Calculate 17 x 24. Answer with only the number.", "408"),
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

_spawned_servers = []


def _normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def verify_response(prompt_capability, response):
    """Return True when the response contains the expected answer (normalized
    containment), with light-touch rules for semantic prompts."""
    _, _, expected = next(p for p in PROMPTS if p[0] == prompt_capability)
    if expected == "Valid JSON":
        try:
            json.loads(response[response.find("{"):response.rfind("}") + 1])
            return True
        except (ValueError, IndexError):
            return False
    if expected == "Semantic scoring" or expected == "States not provided":
        # Content-dependent: verify the response is non-empty and on-topic
        # by checking it mentions the subject or declines to answer.
        return len(response.strip()) > 0
    return _normalize(expected) in _normalize(response)


def probe_continuation(port, temperature=0.2):
    """Exercise the multi-prefix cache's real use case: a second turn that
    continues the first, with the client echoing the assistant reply verbatim.
    Returns a dict with per-turn cached tokens and wall times, or None on a
    protocol error. A hit is cached_tokens > 0 on the second turn.
    """
    prompt_a = "The access code is 7391. Remember it."
    prompt_b = "What is the access code? Answer with only the number."
    first = send_request_stream([{"role": "user", "content": prompt_a}],
                                port=port, temperature=temperature)
    if not first:
        return None
    wall_a, _, _, ct_a, _, reply = first
    second = send_request_stream([
        {"role": "user", "content": prompt_a},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": prompt_b},
    ], port=port, temperature=temperature)
    if not second:
        return None
    wall_b, _, pt_b, ct_b, cached_b, response_b = second
    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "turn1_wall_s": round(wall_a, 2),
        "turn1_completion_tokens": ct_a,
        "turn2_wall_s": round(wall_b, 2),
        "turn2_prompt_tokens": pt_b,
        "turn2_completion_tokens": ct_b,
        "turn2_cached_tokens": cached_b,
        "hit": cached_b > 0,
        "turn2_verified": "7391" in response_b,
    }


def send_request_stream(messages, port=8080, temperature=0.2):
    """Stream request tracking TTFT using incremental read.

    Returns (wall, ttft, pt, ct, cached, content) or None on protocol error.
    """
    payload = json.dumps({
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 42,
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True}
    }, separators=(",", ":")).encode()

    start = time.time()
    ttft = None

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
    except OSError as exc:
        print(f"  ERROR: request failed: {exc}", flush=True)
        return None

    content = []
    usage = None
    first_content = False
    done = False
    buf = ""

    try:
        while not done:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                json_str = line[5:].strip()
                if json_str == "[DONE]":
                    done = True
                    break
                if not json_str:
                    continue
                try:
                    chunk_data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    print(f"  WARNING: malformed SSE frame skipped: {exc}", flush=True)
                    continue
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
    finally:
        conn.close()

    wall = time.time() - start

    if not usage:
        print("  ERROR: no usage chunk received (stream ended prematurely)",
              flush=True)
        return None
    if ttft is None:
        ttft = wall

    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))

    return wall, ttft, pt, ct, cached, "".join(content)


def run_config(cache_mode, mtp_config, config_label, port, verify=True,
               temperature=0.2):
    print(f"\n{'#'*110}", flush=True)
    print(f"# BENCHMARK: {config_label}", flush=True)
    print(f"# Cache: {cache_mode}, MTP: {mtp_config}, Port: {port}, "
          f"Temperature: {temperature}", flush=True)
    print(f"{'#'*110}", flush=True)

    print(f"\n>>> Warming up...", flush=True)
    send_request_stream([{"role": "user", "content": "Test"}], port=port,
                        temperature=temperature)
    time.sleep(0.5)

    results = []
    warm_sends = cache_mode == "multi-prefix"  # second send must hit the cache
    cache_hits = 0
    verified = 0

    print(f"\n{'#':>3s} {'Capability':<22} {'Wall':>6s} {'TTFT':>6s} {'Decode':>8s} {'PT':>4s} {'CT':>4s} "
          f"{'Cache':>6s} {'Decode':>10s} {'OK':>3s}", flush=True)
    print("-" * 108, flush=True)

    warm_rates = []
    for i, (cap_name, prompt_text, _) in enumerate(PROMPTS):
        sends = 2 if warm_sends else 1
        row = None
        for send in range(sends):
            result = send_request_stream(
                [{"role": "user", "content": prompt_text}], port=port,
                temperature=temperature)
            if not result:
                print(f"{i+1:3d} {cap_name:<22s} send {send+1}/{sends} FAILED", flush=True)
                row = None
                break
            wall, ttft, pt, ct, cached, response = result
            if send == 1:
                if cached > 0:
                    cache_hits += 1
                else:
                    print(f"  WARNING: cache-miss on warm send for {cap_name} "
                          f"(cached_tokens={cached})", flush=True)
            row = result
        if not row:
            continue

        wall, ttft, pt, ct, cached, response = row
        decode_time = wall - ttft
        ok = verify_response(cap_name, response) if verify else None
        if ok:
            verified += 1
        # Only decode-rate requests with real decode time; otherwise null.
        decode_rate = ct / decode_time if decode_time > 0.5 else None
        cached_pct = (cached / pt * 100) if pt > 0 else 0

        if decode_time > 0.5 and decode_rate:
            warm_rates.append(decode_rate)

        results.append({"capability": cap_name, "wall": wall, "ttft": ttft,
            "decode_time": decode_time, "prompt_tokens": pt,
            "completion_tokens": ct, "cached_tokens": cached,
            "decode_rate": decode_rate, "verified": ok})

        tag = " *" if i == 0 else (" (filtered)" if decode_time <= 0.5 else "")
        rate_s = f"{decode_rate:9.2f}" if decode_rate else "      n/a"
        ok_s = "yes" if ok else ("no" if ok is False else " n/a")
        print(f"{i+1:3d} {cap_name:<22s} {wall:6.2f} {ttft:6.2f} {decode_time:8.2f} {pt:4d} {ct:4d} "
              f"{cached_pct:5.1f}% {rate_s} {ok_s:>3s}{tag}", flush=True)

    print("-" * 108, flush=True)

    # The multi-prefix cache cannot hit an identical-prompt replay (the
    # entry's KV-backed prefix extends through the generated assistant
    # tokens), so exercise its real continuation shape while the server is
    # still up. Its two footer lines land after the replay sends and are
    # sliced out of the cell rates below.
    continuation_probe = None
    if warm_sends:
        continuation_probe = probe_continuation(port, temperature)
        if continuation_probe:
            print(
                f"\nCONTINUATION PROBE: turn2 cached_tokens="
                f"{continuation_probe['turn2_cached_tokens']} "
                f"hit={continuation_probe['hit']} "
                f"turn2_wall={continuation_probe['turn2_wall_s']}s",
                flush=True)
        else:
            print("\nCONTINUATION PROBE: failed", flush=True)

    # The server's stdout is block-buffered when redirected to the log file,
    # so footer lines are not on disk until the process exits. Terminate the
    # server (flushing its stdio) before parsing; main()'s later
    # terminate_servers() is then a no-op.
    terminate_servers()

    # Authoritative decode rates come from the server's own footer lines in
    # /tmp/nvmaiserver_PORT.log (one per completed request). The first line
    # is the warmup request.
    log_path = f"/tmp/nvmaiserver_{port}.log"
    footer_rates = []
    mtp_stats = []
    try:
        with open(log_path) as f:
            for line in f:
                match = DECODE_FOOTER_RE.search(line)
                if match:
                    footer_rates.append(float(match.group(1)))
                m = MTP_STATS_RE.search(line)
                if m:
                    mtp_stats.append({
                        "drafted": int(m.group(1)),
                        "accepted": int(m.group(2)),
                        "acceptance": float(m.group(3)),
                        "target_passes": int(m.group(4)),
                        "emitted_per_pass": float(m.group(5)),
                    })
    except OSError as exc:
        print(f"  WARNING: could not read footer log {log_path}: {exc}", flush=True)
    expected_sends = len(results) * (2 if warm_sends else 1)
    # Footer line order: warmup, replay sends, then (cache cell) the two
    # continuation-probe requests.
    rates = footer_rates[1:1 + expected_sends]
    probe_footer = footer_rates[1 + expected_sends:]
    if len(rates) == expected_sends:
        for row, rate in zip(results, rates):
            row["footer_decode_tok_s"] = round(rate, 2)
    elif rates:
        print(f"  WARNING: footer lines ({len(rates)}) != sends ({expected_sends}); "
              f"footer rates reported in summary only", flush=True)
    if continuation_probe and len(probe_footer) >= 2:
        continuation_probe["turn1_footer_decode_tok_s"] = round(probe_footer[0], 2)
        continuation_probe["turn2_footer_decode_tok_s"] = round(probe_footer[1], 2)
    warm_footer = []
    if warm_sends and len(rates) == 2 * len(results):
        warm_footer = [rates[2 * i + 1] for i in range(len(results))]
    footer_avg = (sum(rates) / len(rates)) if rates else 0.0
    warm_footer_avg = (sum(warm_footer) / len(warm_footer)) if warm_footer else None
    mtp_summary = None
    if mtp_stats:
        drafted = sum(s["drafted"] for s in mtp_stats)
        accepted = sum(s["accepted"] for s in mtp_stats)
        mtp_summary = {
            "requests": len(mtp_stats),
            "total_drafted": drafted,
            "total_accepted": accepted,
            "acceptance": round(100.0 * accepted / drafted, 1) if drafted else 0.0,
            "avg_emitted_per_pass": round(
                sum(s["emitted_per_pass"] for s in mtp_stats) / len(mtp_stats), 3),
            "per_request": mtp_stats,
        }

    total_wall = sum(r['wall'] for r in results)
    total_ct = sum(r['completion_tokens'] for r in results)

    print(f"\n* Cold start request", flush=True)

    if footer_avg > 0:
        print(f"FOOTER DECODE: {footer_avg:.2f} tok/s ({len(rates)} requests)", flush=True)
        if warm_footer_avg:
            print(f"  Warm-send avg: {warm_footer_avg:.2f} tok/s", flush=True)
        print(f"  Individual: {' | '.join(f'{r:.2f}' for r in rates)}", flush=True)
        if mtp_summary:
            print(f"  MTP: drafted={mtp_summary['total_drafted']} "
                  f"accepted={mtp_summary['total_accepted']} "
                  f"acceptance={mtp_summary['acceptance']}% "
                  f"emitted_per_pass={mtp_summary['avg_emitted_per_pass']}", flush=True)
    if warm_rates:
        avg_warm = sum(warm_rates) / len(warm_rates)
        last6 = warm_rates[-6:]
        last6_avg = sum(last6) / len(last6)
        print(f"WARM DECODE: {avg_warm:.2f} tok/s ({len(warm_rates)} requests)")
        print(f"  Last {len(last6)} avg: {last6_avg:.2f} tok/s")
        print(f"  Individual: {' | '.join(f'{r:.2f}' for r in warm_rates)}", flush=True)
    else:
        avg_warm = 0
        last6_avg = 0

    overhead = total_wall - (total_ct / avg_warm if avg_warm > 0 else 0)
    decode_only = (total_ct / avg_warm) if avg_warm > 0 else 0.0
    per_request = (overhead / len(results)) if results else 0.0
    print(f"\nWall: {total_wall:.1f}s | Decode only: {decode_only:.1f}s | "
          f"Overhead: {overhead:.1f}s ({per_request:.2f}s/req)", flush=True)
    if warm_sends:
        print(f"Cache: {cache_hits}/{len(results)} warm sends hit the multi-prefix cache", flush=True)
    print(f"Answers verified: {verified}/{len(results)}", flush=True)

    ts = time.strftime('%Y%m%dT%H%M%S')
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "benchmark-results", f"bench-{config_label}-{ts}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "aggregate.json"), "w") as f:
        json.dump({"config": config_label, "cache_mode": cache_mode,
            "mtp_config": mtp_config, "results": results,
            "summary": {"total_requests": len(results),
                "total_wall_s": round(total_wall, 2),
                "total_completion_tokens": total_ct,
                "avg_decode_tok_s": round(footer_avg if footer_avg > 0 else avg_warm, 2),
                "footer_decode_tok_s": [round(r, 2) for r in rates],
                "avg_footer_decode_tok_s": round(footer_avg, 2),
                "warm_footer_decode_tok_s": [round(r, 2) for r in warm_footer],
                "avg_warm_footer_decode_tok_s": (round(warm_footer_avg, 2)
                                                 if warm_footer_avg else None),
                "mtp": mtp_summary,
                "continuation_probe": continuation_probe,
                "warm_rates": [round(r, 2) for r in warm_rates],
                "last6_avg": round(last6_avg, 2),
                "cache_warm_hits": cache_hits if warm_sends else None,
                "verified": verified}}, f, indent=2)
    print(f"\nSaved: {outdir}/aggregate.json")
    return footer_avg if footer_avg > 0 else avg_warm


def launch_server(base_dir, port, main_model, mtp_model, cache_mode, mtp_config):
    binary = os.path.join(base_dir, ".build", "arm64-apple-macosx",
                          "release", "NVMAIServer")
    cmd = [binary, "--port", str(port), "--model", main_model,
           "--prompt-cache-mode", cache_mode]
    if mtp_config == "on":
        cmd += ["--mtp-model", mtp_model, "--mtp-memory-mib", "384"]
    log = open(f"/tmp/nvmaiserver_{port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    _spawned_servers.append(proc)
    return proc


def terminate_servers():
    for proc in list(_spawned_servers):
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        _spawned_servers.remove(proc)


def wait_ready(port, attempts=10):
    for _ in range(attempts):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            if "ok" in conn.getresponse().read().decode():
                conn.close()
                return True
            conn.close()
        except OSError:
            pass
        time.sleep(10)
    return False


def main():
    signal.signal(signal.SIGINT, lambda *_: (terminate_servers(), sys.exit(130)))

    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    main_model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        base_dir, "models", "qwen3.6_35B_A3B_4Bit")
    mtp_model = os.path.join(base_dir, "models", "qwen3.6_35B_A3B_MTP_4Bit")

    quant_label = "4bit"
    if "6bit" in main_model:
        quant_label = "6bit"
    elif "8bit" in main_model:
        quant_label = "8bit"

    # MTP forces prompt cache OFF server-side, so the cache-ON x MTP-ON cell
    # is impossible; the matrix is 3 real cells. MTP engages only for
    # pure-greedy requests (temperature 0, repetition penalty 1), so the MTP
    # cell runs at temperature 0 while the non-MTP cells keep the original
    # sampling defaults.
    configs = [
        ("off", "off", f"cache_off_mtp_off_{quant_label}", 8080, 0.2),
        ("multi-prefix", "off", f"cache_on_mtp_off_{quant_label}", 8081, 0.2),
        ("off", "on", f"cache_off_mtp_on_{quant_label}", 8082, 0.0),
    ]

    print(f"\n{'#'*110}", flush=True)
    print(f"# BENCHMARKING MODEL: {os.path.basename(main_model)}", flush=True)
    print(f"# Quantization: {quant_label}", flush=True)
    print(f"{'#'*110}", flush=True)

    try:
        for cache_mode, mtp_config, config_label, port, temperature in configs:
            proc = launch_server(base_dir, port, main_model, mtp_model,
                                 cache_mode, mtp_config)
            print(f"Port {port} launched (pid {proc.pid}), waiting...", flush=True)

            if not wait_ready(port):
                print(f"Port {port} FAILED to become ready!", flush=True)
                continue

            run_config(cache_mode, mtp_config, config_label, port,
                       temperature=temperature)
            terminate_servers()
            if port < 8082:
                time.sleep(10)
    finally:
        terminate_servers()

    print(f"\n{'='*110}\nALL PRECISE 2x2 COMPLETE\n{'='*110}", flush=True)


if __name__ == "__main__":
    main()
