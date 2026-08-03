#!/usr/bin/env python3
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/Users/andreborchert/Downloads/NVMAI")
OUT = Path("/Users/andreborchert/Downloads/NVMAI-benchmark-results/cache-stress-20260803T110702Z")
SERVER = ROOT / ".build/release/TurboFieldfareServer"
PORT = 8080
MODEL_ID = "qwen3.6-35b-a3b"
MAX_CONTEXT = 4096
MAX_COMPLETION = 128
RUNS = [("6", "off"), ("6", "on"), ("8", "on"), ("8", "off")]

WORKLOAD = [
    {
        "initial": "Design a Swift 6 actor-based bounded work queue. Show the critical enqueue/dequeue code and invariants. Keep the entire answer under 80 tokens.",
        "followups": [
            "Add cancellation without leaking continuations. Only show the essential change, under 80 tokens.",
            "Give three high-value Swift Testing cases for the revised queue, under 80 tokens.",
        ],
    },
    {
        "initial": "Design a Swift async filesystem indexer for one million files with bounded memory. Include the core concurrency structure, under 80 tokens.",
        "followups": [
            "Add backpressure between directory walking and hashing. Be concrete and stay under 80 tokens.",
            "Explain the failure and retry policy for transient I/O errors, under 80 tokens.",
        ],
    },
    {
        "initial": "Implement the core of a generic O(1) LRU cache in Swift using a dictionary and linked list. Answer under 80 tokens.",
        "followups": [
            "Make it safe for concurrent async callers using an actor. Show only changed API shape, under 80 tokens.",
            "State its time and space complexity plus one subtle correctness invariant, under 80 tokens.",
        ],
    },
    {
        "initial": "Write a concise C++23 RAII wrapper design for a POSIX socket, including ownership and close behavior. Stay under 80 tokens.",
        "followups": [
            "Add correct move construction and move assignment semantics, under 80 tokens.",
            "List three tests that catch double-close and descriptor-reuse bugs, under 80 tokens.",
        ],
    },
    {
        "initial": "Design a safe Swift-to-C++ FFI boundary for passing byte buffers without unnecessary copies. Be concrete, under 80 tokens.",
        "followups": [
            "Clarify lifetime and ownership rules when C++ retains the buffer asynchronously, under 80 tokens.",
            "Propose a minimal error ABI that maps cleanly into Swift throws, under 80 tokens.",
        ],
    },
    {
        "initial": "Sketch a Metal compute kernel strategy for reducing a large Float array on Apple M3. Mention threadgroup memory and dispatch, under 80 tokens.",
        "followups": [
            "Remove likely threadgroup bank conflicts and explain the indexing change, under 80 tokens.",
            "Define a fair benchmark against MPS without including compilation time, under 80 tokens.",
        ],
    },
    {
        "initial": "Outline a recursive-descent parser for arithmetic expressions in Swift with precedence and associativity. Include key functions, under 80 tokens.",
        "followups": [
            "Add source-range diagnostics and recovery after a missing closing parenthesis, under 80 tokens.",
            "Give a focused property-based fuzzing strategy for this parser, under 80 tokens.",
        ],
    },
    {
        "initial": "Design a Swift CLI that analyzes a huge git diff incrementally and reports per-language churn. Keep memory bounded; answer under 80 tokens.",
        "followups": [
            "Handle quoted paths, renames, binary files, and malformed input, under 80 tokens.",
            "Show how you would test streaming behavior without invoking git, under 80 tokens.",
        ],
    },
    {
        "initial": "Design a versioned SQLite migration runner in Swift that is crash-safe and idempotent. Include transaction boundaries, under 80 tokens.",
        "followups": [
            "Add rollback behavior for a failed multi-step migration, under 80 tokens.",
            "Prevent two application processes from migrating concurrently, under 80 tokens.",
        ],
    },
    {
        "initial": "Implement the architecture of a reconnecting SSE client in Swift using URLSession.AsyncBytes. Cover event framing, under 80 tokens.",
        "followups": [
            "Add Last-Event-ID, exponential backoff, and server retry fields, under 80 tokens.",
            "Explain how downstream backpressure and cancellation should propagate, under 80 tokens.",
        ],
    },
]


def model_path(bits):
    return ROOT / f"scratch/qwen36-{bits}bit.gturbo"


def server_args(bits, cache_mode, cache_dir):
    args = [
        str(SERVER),
        "--model", str(model_path(bits)),
        "--model-id", MODEL_ID,
        "--port", str(PORT),
        "--max-context", str(MAX_CONTEXT),
        "--queue-limit", "4",
        "--prefill-chunk", "1024",
        "--prompt-cache-mode", "off" if cache_mode == "off" else "multi-prefix",
    ]
    if cache_mode == "on":
        args += [
            "--prompt-cache-entries", "64",
            "--prompt-cache-memory-mib", "512",
            "--prompt-cache-disk", str(cache_dir),
            "--prompt-cache-disk-mib", "4096",
        ]
    return args


def wait_healthy(proc, timeout=240):
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{PORT}/health"
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = json.load(response)
                if response.status == 200 and body.get("status") == "ok":
                    return
        except Exception as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(f"server health timeout: {last_error}")


def start_server(bits, cache_mode, run_dir, label):
    cache_dir = run_dir / f"{label}-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{label}-server.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        server_args(bits, cache_mode, cache_dir),
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_healthy(proc)
    except Exception:
        stop_server(proc, log)
        raise
    return proc, log


def stop_server(proc, log):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    log.flush()
    log.close()


def request_chat(messages, seed, maximum_completion=MAX_COMPLETION):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "top_k": 64,
        "top_p": 0.95,
        "seed": seed,
        "max_completion_tokens": maximum_completion,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
        headers={
            "Content-Type": "application/json",
            "X-NVMAI-Client": "opencode",
            "X-NVMAI-Profile": "coding-lean",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_content = None
    last_content = None
    content = []
    finish_reason = None
    usage = None
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    now = time.perf_counter()
                    if first_content is None:
                        first_content = now
                    last_content = now
                    content.append(text)
    ended = time.perf_counter()
    if usage is None:
        raise RuntimeError("stream ended without usage")
    if first_content is None:
        first_content = ended
    if last_content is None:
        last_content = ended
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    cached_tokens = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    wall = ended - started
    ttft = first_content - started
    decode_window = last_content - first_content
    finalize_window = ended - last_content
    decoded_intervals = max(0, completion_tokens - 1)
    return {
        "content": "".join(content),
        "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_prompt_tokens": prompt_tokens - cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage["total_tokens"]),
        "wall_seconds": wall,
        "ttft_seconds": ttft,
        "decode_window_seconds": decode_window,
        "server_finalize_seconds": finalize_window,
        "decode_tokens_per_second": (
            decoded_intervals / decode_window if decode_window > 0 else None
        ),
        "end_to_end_output_tokens_per_second": (
            completion_tokens / wall if wall > 0 else None
        ),
        "approx_computed_prefill_tokens_per_second": (
            (prompt_tokens - cached_tokens) / ttft if ttft > 0 else None
        ),
    }


def warmup(bits, cache_mode, run_dir, system_prompt):
    proc, log = start_server(bits, cache_mode, run_dir, "warmup")
    try:
        result = request_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Name one Swift concurrency primitive in five words."},
            ],
            seed=20260803,
            maximum_completion=32,
        )
        (run_dir / "warmup.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
    finally:
        stop_server(proc, log)


def measured_run(bits, cache_mode, run_dir, system_prompt):
    proc, log = start_server(bits, cache_mode, run_dir, "measured")
    records = []
    histories = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case["initial"]},
        ]
        for case in WORKLOAD
    ]
    orders = [list(range(10)), list(reversed(range(10))), list(range(10))]
    output_path = run_dir / "requests.jsonl"
    try:
        with open(output_path, "w", encoding="utf-8") as output:
            request_number = 0
            for turn, order in enumerate(orders):
                for conversation in order:
                    if turn > 0:
                        followup = WORKLOAD[conversation]["followups"][turn - 1]
                        histories[conversation].append({"role": "user", "content": followup})
                    request_number += 1
                    result = request_chat(
                        histories[conversation],
                        seed=20260803 + conversation * 10 + turn,
                    )
                    result.update(
                        {
                            "quant_bits": int(bits),
                            "cache": cache_mode,
                            "request_number": request_number,
                            "conversation": conversation + 1,
                            "turn": turn,
                        }
                    )
                    records.append(result)
                    histories[conversation].append(
                        {"role": "assistant", "content": result["content"]}
                    )
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        f"{bits}bit cache={cache_mode} request={request_number}/30 "
                        f"conversation={conversation + 1} turn={turn} "
                        f"prompt={result['prompt_tokens']} cached={result['cached_tokens']} "
                        f"completion={result['completion_tokens']} wall={result['wall_seconds']:.2f}s "
                        f"ttft={result['ttft_seconds']:.2f}s "
                        f"decode={result['decode_tokens_per_second']:.2f} tok/s",
                        flush=True,
                    )
    finally:
        stop_server(proc, log)
    return records


def aggregate(records):
    decode_tokens = sum(max(0, row["completion_tokens"] - 1) for row in records)
    decode_seconds = sum(row["decode_window_seconds"] for row in records)
    completion_tokens = sum(row["completion_tokens"] for row in records)
    wall = sum(row["wall_seconds"] for row in records)
    ttfts = [row["ttft_seconds"] for row in records]
    return {
        "requests": len(records),
        "prompt_tokens": sum(row["prompt_tokens"] for row in records),
        "cached_tokens": sum(row["cached_tokens"] for row in records),
        "computed_prompt_tokens": sum(row["computed_prompt_tokens"] for row in records),
        "completion_tokens": completion_tokens,
        "total_wall_seconds": wall,
        "total_server_finalize_seconds": sum(
            row["server_finalize_seconds"] for row in records
        ),
        "aggregate_decode_tokens_per_second": decode_tokens / decode_seconds,
        "aggregate_end_to_end_output_tokens_per_second": completion_tokens / wall,
        "mean_ttft_seconds": statistics.mean(ttfts),
        "median_ttft_seconds": statistics.median(ttfts),
        "max_ttft_seconds": max(ttfts),
        "mean_server_finalize_seconds": statistics.mean(
            row["server_finalize_seconds"] for row in records
        ),
        "finish_reasons": {
            reason: sum(1 for row in records if row["finish_reason"] == reason)
            for reason in sorted({row["finish_reason"] for row in records})
        },
    }


def build_summary(all_records):
    summary = {"configuration": {}, "by_turn": {}, "comparisons": {}}
    for key, records in all_records.items():
        summary["configuration"][key] = aggregate(records)
        summary["by_turn"][key] = {
            str(turn): aggregate([row for row in records if row["turn"] == turn])
            for turn in range(3)
        }
    for bits in ("6", "8"):
        off = summary["configuration"][f"{bits}-off"]
        on = summary["configuration"][f"{bits}-on"]
        summary["comparisons"][f"{bits}-cache-on-vs-off"] = {
            "decode_tokens_per_second_change_pct":
                (on["aggregate_decode_tokens_per_second"] / off["aggregate_decode_tokens_per_second"] - 1) * 100,
            "end_to_end_output_tokens_per_second_change_pct":
                (on["aggregate_end_to_end_output_tokens_per_second"] / off["aggregate_end_to_end_output_tokens_per_second"] - 1) * 100,
            "total_time_reduction_pct":
                (1 - on["total_wall_seconds"] / off["total_wall_seconds"]) * 100,
            "mean_ttft_reduction_pct":
                (1 - on["mean_ttft_seconds"] / off["mean_ttft_seconds"]) * 100,
        }
    for cache_mode in ("off", "on"):
        six = summary["configuration"][f"6-{cache_mode}"]
        eight = summary["configuration"][f"8-{cache_mode}"]
        summary["comparisons"][f"8-vs-6-cache-{cache_mode}"] = {
            "decode_tokens_per_second_change_pct":
                (eight["aggregate_decode_tokens_per_second"] / six["aggregate_decode_tokens_per_second"] - 1) * 100,
            "end_to_end_output_tokens_per_second_change_pct":
                (eight["aggregate_end_to_end_output_tokens_per_second"] / six["aggregate_end_to_end_output_tokens_per_second"] - 1) * 100,
            "total_time_change_pct":
                (eight["total_wall_seconds"] / six["total_wall_seconds"] - 1) * 100,
        }
    return summary


def main():
    fixture = json.loads(
        (ROOT / "Tests/TurboFieldfareServer/Fixtures/opencode-1.15.11-initial.json").read_text()
    )
    system_prompt = fixture["messages"][0]["content"]
    all_records = {}
    for bits, cache_mode in RUNS:
        key = f"{bits}-{cache_mode}"
        run_dir = OUT / key
        run_dir.mkdir(parents=True, exist_ok=True)
        request_log = run_dir / "requests.jsonl"
        if request_log.exists():
            existing = [
                json.loads(line)
                for line in request_log.read_text().splitlines()
                if line.strip()
            ]
            if len(existing) == 30:
                all_records[key] = existing
                print(f"SKIP {key}; 30 completed requests already recorded", flush=True)
                continue
        print(f"START {key} warmup", flush=True)
        warmup(bits, cache_mode, run_dir, system_prompt)
        print(f"START {key} measured", flush=True)
        records = measured_run(bits, cache_mode, run_dir, system_prompt)
        all_records[key] = records
        (run_dir / "aggregate.json").write_text(
            json.dumps(aggregate(records), indent=2) + "\n"
        )
        print(f"DONE {key}", flush=True)
    summary = build_summary(all_records)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        raise
