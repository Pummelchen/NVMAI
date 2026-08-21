#!/usr/bin/env python3
"""Reproducible barrier vs hit/fixup decode A/B for the production profile.

Each mode gets a fresh server. Every fixed prompt is sent twice with the same
seed: the first request observes colder routed-expert slots and the second a
warmer working set. The server's NVMAI_RUNNER_STATS footer supplies cache and
I/O measurements; response text must match across modes before results pass.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

from nvmai_profile import (
    DEFAULT_API_MODEL,
    DEFAULT_MODEL_PATH,
    benchmark_log_path,
    server_command,
    server_environment,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
PORT = 8112
PROCESS_PATTERN = (
    "NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|"
    "swiftpm-testing-helper|mlx_lm|mlx-lm|llama-server|unsloth-studio"
)
PROMPTS = {
    "short": "Explain why a mutex protects shared state in one short paragraph.",
    "medium": "Explain how an SSD-backed sparse mixture-of-experts cache can bound memory "
              "while preserving inference correctness. Include hits, misses, and eviction.",
    "long": (ROOT / "AGENTS.md").read_text(),
}


def command_output(command: list[str]) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()


def preflight() -> dict[str, object]:
    if not SERVER.is_file():
        raise RuntimeError(f"release server missing: {SERVER}; run swift build -c release")
    for required in ("manifest.json", "verified-install.json"):
        if not (DEFAULT_MODEL_PATH / required).is_file():
            raise RuntimeError(f"incomplete model installation: missing {required}")
    process_listing = command_output(["ps", "-axo", "pid=,command="])
    process_pattern = re.compile(PROCESS_PATTERN)
    processes = [line.strip() for line in process_listing.splitlines()
                 if process_pattern.search(line)]
    if processes:
        raise RuntimeError("model process already running; refusing to benchmark:\n"
                           + "\n".join(processes))
    pressure = command_output(["memory_pressure", "-Q"])
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    if not free or int(free.group(1)) < 10:
        raise RuntimeError("memory pressure is not acceptable: " + pressure)
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "hardware": command_output(["system_profiler", "SPHardwareDataType"]),
        "macos": platform.mac_ver()[0],
        "swift": command_output(["swift", "--version"]),
        "model": str(DEFAULT_MODEL_PATH),
    }


def wait_until_healthy(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
            connection.request("GET", "/health")
            healthy = connection.getresponse().read().decode()
            connection.close()
            if "ok" in healthy:
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError("server health check timed out")


def memory_snapshot(process_id: int) -> dict[str, object]:
    rss = command_output(["ps", "-o", "rss=", "-p", str(process_id)])
    pressure = command_output(["memory_pressure", "-Q"])
    free = re.search(r"free percentage:\s*(\d+)%", pressure)
    return {
        "server_rss_kib": int(rss),
        "machine_free_percent": int(free.group(1)) if free else None,
    }


def request(prompt: str) -> dict[str, object]:
    payload = json.dumps({
        "model": DEFAULT_API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "seed": 41,
        "max_completion_tokens": 128,
        "stream": False,
    }).encode()
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
    connection.request("POST", "/v1/chat/completions", body=payload,
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"request failed HTTP {response.status}: {body.decode()}")
    decoded = json.loads(body)
    return {
        "wall_seconds": time.monotonic() - started,
        "content": decoded["choices"][0]["message"]["content"],
        "usage": decoded.get("usage", {}),
    }


def parse_footers(log_path: pathlib.Path) -> tuple[list[str], list[str]]:
    generation: list[str] = []
    runner: list[str] = []
    for line in log_path.read_text().splitlines():
        if "NVMAI generation" in line and "decode_tok_s=" in line:
            generation.append(line.strip())
        if "NVMAI runner" in line and "expert_hit_rate=" in line:
            runner.append(line.strip())
    return generation, runner


def run_case(mode: str, prompt_name: str, prompt: str,
             io_backend: str = "pread", io_sync: str = "host",
             cache_layout: str = "per-slot",
             cache_policy: str = "lfu",
             io_submission: str = "deferred") -> tuple[list[dict[str, object]], pathlib.Path]:
    environment = server_environment()
    environment["NVMAI_DECODE_EXPERT_EXECUTION"] = mode
    environment["NVMAI_EXPERT_IO_BACKEND"] = io_backend
    environment["NVMAI_EXPERT_IO_SYNC"] = io_sync
    environment["NVMAI_EXPERT_CACHE_LAYOUT"] = cache_layout
    environment["NVMAI_EXPERT_CACHE_POLICY"] = cache_policy
    environment["NVMAI_EXPERT_IO_SUBMISSION"] = io_submission
    environment["NVMAI_RUNNER_STATS"] = "1"
    environment["NVMAI_KERNEL_STATS"] = "1"
    log_path = pathlib.Path(benchmark_log_path(
        f"expert-ab-{mode}-{io_backend}-{io_sync}-{cache_layout}-"
        f"{cache_policy}-{io_submission}-{prompt_name}.log"))
    with log_path.open("w") as log:
        process = subprocess.Popen(
            server_command(SERVER, PORT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_healthy(process)
            baseline_memory = memory_snapshot(process.pid)
            results = []
            for warmth in ("cold", "warm"):
                print(f"{mode:9s} {prompt_name:6s} {warmth}", flush=True)
                results.append({
                    "mode": mode,
                    "io_backend": io_backend,
                    "io_sync": io_sync,
                    "cache_layout": cache_layout,
                    "cache_policy": cache_policy,
                    "io_submission": io_submission,
                    "prompt": prompt_name,
                    "warmth": warmth,
                    "baseline_memory": baseline_memory,
                    **request(prompt),
                    "memory_after": memory_snapshot(process.pid),
                })
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    generation, runner = parse_footers(log_path)
    if len(generation) != len(results) or len(runner) != len(results):
        raise RuntimeError(
            f"missing benchmark footers in {log_path}: "
            f"generation={len(generation)} runner={len(runner)} expected={len(results)}")
    for index, result in enumerate(results):
        result["generation_footer"] = generation[index]
        result["runner_footer"] = runner[index]
    return results, log_path


def main() -> int:
    argparse.ArgumentParser(
        description="Compare barrier and hit/fixup decode with fixed production settings"
    ).parse_args()
    metadata = preflight()
    all_results: list[dict[str, object]] = []
    logs: dict[str, str] = {}
    for mode in ("barrier", "hit-fixup"):
        for prompt_name, prompt in PROMPTS.items():
            results, log = run_case(mode, prompt_name, prompt)
            all_results.extend(results)
            logs[f"{mode}/{prompt_name}"] = str(log)

    by_case = {(row["prompt"], row["warmth"], row["mode"]): row for row in all_results}
    mismatches = []
    for prompt in PROMPTS:
        for warmth in ("cold", "warm"):
            old = by_case[(prompt, warmth, "barrier")]["content"]
            new = by_case[(prompt, warmth, "hit-fixup")]["content"]
            if old != new:
                mismatches.append(f"{prompt}/{warmth}")

    output = ROOT / ".build/benchmark-results/hit-fixup-ab.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "metadata": metadata,
        "configuration": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "seed": 41,
            "max_completion_tokens": 128,
        },
        "logs": logs,
        "results": all_results,
        "response_mismatches": mismatches,
        "passed": not mismatches,
    }, indent=2) + "\n")
    print(f"results: {output}")
    if mismatches:
        print("ERROR: response mismatches: " + ", ".join(mismatches), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
