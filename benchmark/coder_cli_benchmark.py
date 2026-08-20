#!/usr/bin/env python3
"""Reproducible Ornith coding-client and NVMAI feature benchmark rounds.

Round ``coder`` runs Codex, Qwen Code, OpenCode, and (through the adjacent
loopback adapter) Claude Code against Ornith 4-bit and 8-bit with fixed short,
medium, and long prompts.  Round ``features`` uses direct OpenAI requests to
isolate prompt-cache, fast-alias, and concise-mode behavior on Ornith, then
checks MTP off/on against the supported Qwen target.  Results are resumable and
kept below .build by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
ADAPTER = ROOT / "benchmark/claude_openai_adapter.py"
MODEL_PATHS = {
    "ornith": {
        4: ROOT / "models/ornith-1.5_35B_A3B_4Bit",
        8: ROOT / "models/ornith-1.5_35B_A3B_8Bit",
    },
    "qwen": {
        4: ROOT / "models/qwen3.6_35B_A3B_4Bit",
        8: ROOT / "models/qwen3.6_35B_A3B_8Bit",
    },
}
MTP_PATH = ROOT / "models/qwen3.6_35B_A3B_MTP_4Bit"
PROCESS_PATTERN = (
    "NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|"
    "swiftpm-testing-helper|mlx_lm|mlx-lm"
)


@dataclass
class RunningProcess:
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: pathlib.Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log_handle.close()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def process_descendants(root_pid: int) -> list[int]:
    """Return descendants of one process, deepest children last."""
    try:
        rows = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid="], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except subprocess.SubprocessError:
        return []
    children: dict[int, list[int]] = {}
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            continue
        pid, parent = (int(field) for field in fields)
        children.setdefault(parent, []).append(pid)
    result: list[int] = []
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        result.append(pid)
        pending.extend(children.get(pid, []))
    return result


def signal_client_tree(process: subprocess.Popen[str], sent_signal: signal.Signals) -> None:
    """Signal only the benchmark client group and its snapshotted descendants."""
    descendants = process_descendants(process.pid)
    try:
        os.killpg(process.pid, sent_signal)
    except ProcessLookupError:
        pass
    for pid in reversed(descendants):
        try:
            os.kill(pid, sent_signal)
        except ProcessLookupError:
            pass


def terminate_client(process: subprocess.Popen[str]) -> tuple[str, str]:
    signal_client_tree(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        signal_client_tree(process, signal.SIGKILL)
        return process.communicate(timeout=5)


def manifest_api_model(model_path: pathlib.Path) -> str:
    manifest = json.loads((model_path / "manifest.json").read_text())
    return re.sub(r"-(?:4|8)bit$", "", manifest["modelID"])


def fixed_prompts() -> dict[str, str]:
    readme = (ROOT / "README.md").read_text()
    agents = (ROOT / "AGENTS.md").read_text()
    instruction = "Do not change files or run commands. Answer only from the supplied context."
    return {
        "short": (
            f"{instruction}\n\nExplain what a mutex is and when you would use one. "
            "Keep the answer within 60 words."
        ),
        "medium": (
            f"{instruction}\n\nUsing the README below, summarize NVMAI's purpose, "
            "core features, and basic usage in at most 80 words.\n\n"
            f"--- README.md ---\n{readme}\n--- end README.md ---"
        ),
        "long": (
            f"{instruction}\n\nUsing the repository documents below, explain NVMAI's "
            "bounded expert-streaming architecture, its safety constraints, and the correct "
            "model install/run workflow in at most 80 words.\n\n"
            f"--- README.md ---\n{readme}\n--- end README.md ---\n\n"
            f"--- AGENTS.md ---\n{agents}\n--- end AGENTS.md ---"
        ),
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def preflight(required_models: Iterable[pathlib.Path]) -> dict[str, Any]:
    if not SERVER.is_file():
        raise RuntimeError(f"release server missing: {SERVER}; run swift build -c release")
    process_check = subprocess.run(
        ["pgrep", "-fl", PROCESS_PATTERN], text=True, capture_output=True, check=False
    )
    if process_check.returncode == 0 and process_check.stdout.strip():
        raise RuntimeError("model process already running; refusing to benchmark:\n" + process_check.stdout)
    for model in required_models:
        for name in ("manifest.json", "verified-install.json"):
            if not (model / name).is_file():
                raise RuntimeError(f"incomplete model installation: {model} lacks {name}")
    memory = run_text(["memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*(\d+)%", memory)
    if not match:
        raise RuntimeError("could not establish acceptable memory pressure: " + memory)
    if int(match.group(1)) < 10:
        raise RuntimeError("memory pressure is too high for a model benchmark: " + memory)
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "status": run_text(["git", "status", "--short"]),
        "macos": run_text(["sw_vers"]),
        "swift": run_text(["xcrun", "swift", "--version"]),
        "chip": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(run_text(["sysctl", "-n", "hw.memsize"])),
        "memory_pressure": memory,
        "started_at": utc_now(),
    }


def wait_http(url: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"endpoint did not become ready: {url}: {last_error!r}")


def start_server(*, output: pathlib.Path, family: str, quant: int, port: int,
                 cache: bool, concise: bool, mtp: bool, label: str) -> RunningProcess:
    model = MODEL_PATHS[family][quant]
    command = [
        str(SERVER), "--model", str(model), "--port", str(port),
        "--max-context", "32768", "--prompt-cache-mode", "multi-prefix" if cache else "off",
        "--prompt-cache-memory-mib", "256" if cache else "0",
    ]
    if mtp:
        command += ["--mtp-model", str(MTP_PATH), "--mtp-memory-mib", "384"]
    env = os.environ.copy()
    # Coding-client answers are the measured artifact. Keep reasoning tokens
    # out of the visible response so every client can be judged against the
    # same explicit word limit; MTP qualification also requires direct greedy
    # generation rather than a separately rendered reasoning block.
    env["NVMAI_THINKING_MODE"] = "0"
    if concise:
        env["NVMAI_CONCISE_MODE"] = "1"
    else:
        env.pop("NVMAI_CONCISE_MODE", None)
    log_path = output / "server" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", buffering=1)
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    running = RunningProcess(process, log_handle, log_path)
    try:
        wait_http(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            ids = {
                item.get("id") for item in json.loads(response.read()).get("data", [])
                if isinstance(item.get("id"), str)
            }
        expected_model = manifest_api_model(model)
        if expected_model not in ids:
            raise RuntimeError(
                f"server model mismatch: expected {expected_model!r}, advertised {sorted(ids)!r}"
            )
    except Exception:
        running.stop()
        raise
    return running


def start_adapter(output: pathlib.Path, openai_url: str, model: str, port: int,
                  label: str) -> RunningProcess:
    log_path = output / "adapter" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", buffering=1)
    process = subprocess.Popen(
        [sys.executable, str(ADAPTER), "--port", str(port), "--openai-url", openai_url,
         "--model", model, "--max-tokens", "2048"],
        cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    running = RunningProcess(process, log_handle, log_path)
    try:
        wait_http(f"http://127.0.0.1:{port}/health", timeout=30)
    except Exception:
        running.stop()
        raise
    return running


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def completed_keys(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value.get("key"), str):
            keys.add(value["key"])
    return keys


def command_version(binary: str) -> str:
    try:
        return subprocess.check_output([binary, "--version"], text=True, stderr=subprocess.STDOUT,
                                       timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def client_binaries() -> dict[str, str | None]:
    return {
        "codex": shutil.which("codex"),
        "qwen": shutil.which("qwen") or shutil.which("qwen-code"),
        "opencode": shutil.which("opencode"),
        "claude": shutil.which("claude"),
    }


def prepare_client_config(output: pathlib.Path, client: str, base_url: str,
                          model: str) -> pathlib.Path:
    home = output / "client-config" / client
    home.mkdir(parents=True, exist_ok=True)
    if client == "codex":
        (home / "config.toml").write_text(
            f'model = "{model}"\nmodel_provider = "nvmai"\n\n'
            "[model_providers.nvmai]\nname = \"NVMAI\"\n"
            f'base_url = "{base_url}"\nwire_api = "responses"\n'
        )
    elif client == "qwen":
        write_json(home / "settings.json", {
            "modelProviders": {"openai": [{
                "id": model, "name": f"[NVMAI] {model}", "baseUrl": base_url,
                "description": "NVMAI local benchmark", "envKey": "OPENAI_API_KEY",
            }]},
            "security": {"auth": {"selectedType": "openai"}},
            "model": {"name": model},
            "memory": {
                "enableManagedAutoMemory": False,
                "enableManagedAutoDream": False,
                "enableAutoSkill": False,
            },
        })
    elif client == "opencode":
        write_json(home / "opencode" / "opencode.json", {
            "$schema": "https://opencode.ai/config.json",
            "model": f"nvmai/{model}",
            "small_model": f"nvmai/{model}",
            "provider": {"nvmai": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "NVMAI",
                "options": {
                    "baseURL": base_url,
                    "headers": {"Authorization": "Bearer local"},
                },
                "models": {model: {
                    "name": model,
                    "limit": {"context": 32768, "output": 2048},
                }},
            }},
        })
    return home


def parse_opencode_answer(stdout: str) -> str:
    texts: list[str] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = value.get("part") if isinstance(value, dict) else None
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif isinstance(value, dict) and value.get("type") == "text" and isinstance(value.get("text"), str):
            texts.append(value["text"])
    return "".join(texts) if texts else stdout


def parse_claude_answer(stdout: str) -> str:
    try:
        value = json.loads(stdout)
        return value.get("result") or stdout
    except json.JSONDecodeError:
        return stdout


def run_client(*, client: str, binary: str, prompt: str, base_url: str, model: str,
               output: pathlib.Path, key: str, adapter_url: str | None,
               timeout: int) -> dict[str, Any]:
    config = prepare_client_config(output, client, base_url, model)
    env = os.environ.copy()
    env.update({"OPENAI_API_KEY": "local", "DISABLE_AUTOUPDATER": "1"})
    raw_path = output / "raw" / f"{key}.stdout.txt"
    error_path = output / "raw" / f"{key}.stderr.txt"
    answer_path = output / "answers" / f"{key}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    if client == "codex":
        env["CODEX_HOME"] = str(config)
        command = [
            binary, "exec", "--sandbox", "read-only", "--ephemeral", "--color", "never",
            "--skip-git-repo-check", "-C", str(ROOT), "-o", str(answer_path), prompt,
        ]
    elif client == "qwen":
        env.update({
            "QWEN_HOME": str(config),
            "QWEN_STREAM_IDLE_TIMEOUT_MS": "0",
            "QWEN_STREAM_MAX_LIFETIME_MS": "0",
        })
        command = [binary, "--safe-mode", "-m", model, "-p", prompt, "-o", "text"]
    elif client == "opencode":
        env["XDG_CONFIG_HOME"] = str(config)
        command = [
            binary, "run", "--pure", "--format", "json", "--dir", str(ROOT),
            "-m", f"nvmai/{model}", prompt,
        ]
    elif client == "claude":
        if not adapter_url:
            raise RuntimeError("Claude Code requires the benchmark Anthropic adapter")
        env.update({
            "ANTHROPIC_BASE_URL": adapter_url,
            "ANTHROPIC_API_KEY": "local",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        })
        command = [
            binary, "-p", "--model", "sonnet", "--tools", "Read,Glob,Grep",
            "--permission-mode", "plan", "--max-turns", "3", "--no-session-persistence",
            "--output-format", "json", prompt,
        ]
    else:
        raise ValueError(client)
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        status = process.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr = terminate_client(process)
        stderr += f"\nbenchmark timeout after {timeout}s\n"
        status = 124
    except KeyboardInterrupt:
        terminate_client(process)
        raise
    wall = time.monotonic() - started
    raw_path.write_text(stdout)
    error_path.write_text(stderr)
    if client == "codex" and answer_path.exists():
        answer = answer_path.read_text()
    elif client == "opencode":
        answer = parse_opencode_answer(stdout)
        answer_path.write_text(answer)
    elif client == "claude":
        answer = parse_claude_answer(stdout)
        answer_path.write_text(answer)
    else:
        answer = stdout
        answer_path.write_text(answer)
    return {
        "exit_code": status,
        "wall_seconds": round(wall, 3),
        "answer": answer,
        "answer_file": str(answer_path.relative_to(output)),
        "stdout_file": str(raw_path.relative_to(output)),
        "stderr_file": str(error_path.relative_to(output)),
        "command": command,
    }


def visible_answer(answer: str) -> str:
    without_thinking = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    return without_thinking.strip()


def quality_check(prompt_name: str, answer: str, exit_code: int) -> dict[str, Any]:
    visible = visible_answer(answer)
    lower = visible.lower()
    failures: list[str] = []
    if exit_code != 0:
        failures.append(f"exit code {exit_code}")
    if len(visible) < 20:
        failures.append("empty or implausibly short answer")
    if any(marker in lower for marker in ("api error", "internal server error", "traceback", "context_length_exceeded")):
        failures.append("answer contains an error marker")
    if prompt_name == "short":
        if "mutex" not in lower or not any(word in lower for word in ("lock", "exclusive", "one thread")):
            failures.append("mutex/exclusion explanation missing")
        if not any(word in lower for word in ("race", "shared", "thread")):
            failures.append("concurrency use case missing")
    elif prompt_name == "medium":
        for label, terms in {
            "project": ("nvmai",),
            "model": ("ornith",),
            "streaming": ("ssd", "expert"),
            "quantization": ("4-bit", "8-bit"),
        }.items():
            if not any(term in lower for term in terms):
                failures.append(f"{label} context missing")
    elif prompt_name == "long":
        for label, terms in {
            "expert streaming": ("ssd", "expert"),
            "loopback safety": ("127.0.0.1", "loopback"),
            "receipt safety": ("receipt", "verified-install", "verify-install"),
            "install/run workflow": ("repack", "install", "server", "cli"),
        }.items():
            if not any(term in lower for term in terms):
                failures.append(f"{label} context missing")
    words = re.findall(r"\b[\w.-]+\b", visible)
    limit = 60 if prompt_name == "short" else 80
    if len(words) > limit + 15:
        failures.append(f"answer exceeds requested length materially ({len(words)} words)")
    return {"automatic_pass": not failures, "failures": failures, "visible_words": len(words)}


def server_log_delta(path: pathlib.Path, offset: int) -> tuple[str, int]:
    with path.open() as handle:
        handle.seek(offset)
        data = handle.read()
        return data, handle.tell()


def run_coder_round(args: argparse.Namespace, output: pathlib.Path, prompts: dict[str, str],
                    results: pathlib.Path) -> None:
    binaries = client_binaries()
    clients = args.clients or ["codex", "qwen", "opencode", "claude"]
    for client in clients:
        if not binaries.get(client):
            raise RuntimeError(f"requested coding client is not installed: {client}")
    done = completed_keys(results)
    for quant in args.quantizations:
        port = 18080 + quant
        model_path = MODEL_PATHS["ornith"][quant]
        model = manifest_api_model(model_path)
        server = start_server(
            output=output, family="ornith", quant=quant, port=port,
            cache=False, concise=False, mtp=False, label=f"coder-ornith-q{quant}",
        )
        adapter: RunningProcess | None = None
        try:
            base_url = f"http://127.0.0.1:{port}/v1"
            if "claude" in clients:
                adapter = start_adapter(
                    output, base_url, model, 19080 + quant, f"coder-ornith-q{quant}"
                )
            log_offset = server.log_path.stat().st_size
            cases = itertools.product(clients, args.prompts)
            for client, prompt_name in cases:
                runs = [("warmup", 0)] if args.warmups else []
                runs += [("measured", index) for index in range(1, args.repetitions + 1)]
                for phase, repetition in runs:
                    key = f"coder-q{quant}-{client}-{prompt_name}-{phase}-{repetition}"
                    if key in done:
                        print(f"SKIP {key}", flush=True)
                        continue
                    if args.limit is not None and args.executed >= args.limit:
                        return
                    print(f"RUN {key}", flush=True)
                    result = run_client(
                        client=client, binary=binaries[client] or "", prompt=prompts[prompt_name],
                        base_url=base_url, model=model, output=output, key=key,
                        adapter_url=f"http://127.0.0.1:{19080 + quant}" if client == "claude" else None,
                        timeout=args.timeout,
                    )
                    time.sleep(0.2)
                    log_delta, log_offset = server_log_delta(server.log_path, log_offset)
                    quality = quality_check(prompt_name, result["answer"], result["exit_code"])
                    record = {
                        "key": key, "round": "coder", "status": "ok" if result["exit_code"] == 0 else "failed",
                        "phase": phase, "repetition": repetition, "family": "ornith",
                        "quantization": quant, "client": client, "prompt": prompt_name,
                        "prompt_sha256": sha256_text(prompts[prompt_name]), "model": model,
                        "server_log": log_delta, "quality": quality, "finished_at": utc_now(),
                        **{name: value for name, value in result.items() if name != "answer"},
                    }
                    append_jsonl(results, record)
                    args.executed += 1
                    print(
                        f"DONE {key} exit={result['exit_code']} wall={result['wall_seconds']}s "
                        f"quality={'pass' if quality['automatic_pass'] else 'REVIEW'}",
                        flush=True,
                    )
        finally:
            if adapter:
                adapter.stop()
            server.stop()


def post_chat(base_url: str, model: str, messages: list[dict[str, Any]],
              tools: list[dict[str, Any]], timeout: int) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "max_tokens": 512,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer local"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return value, time.monotonic() - started


def synthetic_tools(count: int = 12) -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": f"read_context_{index}",
            "description": "Read a named repository context item without modifying it. " * 3,
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Context name"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    } for index in range(count)]


def feature_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a repository coding agent. Follow the supplied safety policy, inspect context, "
                "and provide a correct text answer. Do not modify files or call tools during this benchmark. " * 12
            ),
        },
        {
            "role": "user",
            "content": (
                "<system-reminder>Benchmark scaffolding and workspace metadata. Do not quote this block."
                "</system-reminder>\n" + prompt
            ),
        },
    ]


def response_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def feature_key(*, family: str, quant: int, cache: bool, mtp: bool,
                fast: bool, concise: bool, prompt_name: str) -> str:
    return (
        f"feature-{family}-q{quant}-cache{int(cache)}-mtp{int(mtp)}-"
        f"fast{int(fast)}-concise{int(concise)}-{prompt_name}"
    )


def run_feature_case(*, args: argparse.Namespace, output: pathlib.Path,
                     results: pathlib.Path, prompts: dict[str, str], family: str,
                     quant: int, cache: bool, fast: bool, concise: bool, mtp: bool,
                     prompt_name: str, base_url: str, model: str,
                     server: RunningProcess, log_offset: int) -> int:
    key = feature_key(
        family=family, quant=quant, cache=cache, mtp=mtp, fast=fast,
        concise=concise, prompt_name=prompt_name,
    )
    if key in completed_keys(results):
        print(f"SKIP {key}", flush=True)
        return log_offset
    if args.limit is not None and args.executed >= args.limit:
        return log_offset
    print(f"RUN {key}", flush=True)
    requested_model = model + "-fast" if fast else model
    first_messages = feature_messages(prompts[prompt_name])
    tools = synthetic_tools()
    try:
        first, first_wall = post_chat(
            base_url, requested_model, first_messages, tools, args.timeout
        )
        first_answer = response_answer(first)
        followup_prompt = (
            "State one mutex safety rule and the corresponding lock/unlock action. Keep it under 35 words."
            if prompt_name == "short" else
            "Based only on that same context, state one important project safety constraint and one "
            "relevant command or action. Keep it under 45 words."
        )
        second_messages = first_messages + [
            {"role": "assistant", "content": first_answer},
            {"role": "user", "content": followup_prompt},
        ]
        second, second_wall = post_chat(
            base_url, requested_model, second_messages, tools, args.timeout
        )
        second_answer = response_answer(second)
        exit_code = 0
        error = None
    except Exception as exc:
        first = {}
        second = {}
        first_wall = second_wall = 0.0
        first_answer = second_answer = ""
        exit_code = 1
        error = repr(exc)
    answer_path = output / "answers" / f"{key}.txt"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        "FIRST RESPONSE\n" + first_answer + "\n\nFOLLOW-UP RESPONSE\n" + second_answer + "\n"
    )
    time.sleep(0.2)
    log_delta, next_offset = server_log_delta(server.log_path, log_offset)
    first_quality = quality_check(prompt_name, first_answer, exit_code)
    follow_lower = visible_answer(second_answer).lower()
    follow_terms = (("mutex", "lock", "unlock", "race") if prompt_name == "short" else
                    ("127.0.0.1", "receipt", "verify", "server", "install"))
    follow_quality = {
        "automatic_pass": exit_code == 0 and len(follow_lower) >= 20
            and any(word in follow_lower for word in follow_terms),
        "visible_words": len(re.findall(r"\b[\w.-]+\b", visible_answer(second_answer))),
    }
    usage_first = first.get("usage", {})
    usage_second = second.get("usage", {})
    record = {
        "key": key, "round": "features", "status": "ok" if exit_code == 0 else "failed",
        "family": family, "quantization": quant, "cache": cache, "mtp": mtp,
        "fast": fast, "concise": concise, "prompt": prompt_name, "model": requested_model,
        "prompt_sha256": sha256_text(prompts[prompt_name]),
        "first_wall_seconds": round(first_wall, 3),
        "followup_wall_seconds": round(second_wall, 3),
        "first_usage": usage_first, "followup_usage": usage_second,
        "first_quality": first_quality, "followup_quality": follow_quality,
        "answer_file": str(answer_path.relative_to(output)),
        "server_log": log_delta, "error": error, "finished_at": utc_now(),
    }
    append_jsonl(results, record)
    args.executed += 1
    passed = first_quality["automatic_pass"] and follow_quality["automatic_pass"]
    print(
        f"DONE {key} exit={exit_code} first={first_wall:.1f}s follow={second_wall:.1f}s "
        f"quality={'pass' if passed else 'REVIEW'}",
        flush=True,
    )
    return next_offset


def run_feature_round(args: argparse.Namespace, output: pathlib.Path,
                      prompts: dict[str, str], results: pathlib.Path) -> None:
    # Ornith: all supported cache/fast/concise combinations. MTP is not mixed
    # into these rows because the installed Ornith sources omit its draft layer.
    # Each case gets a fresh server so a previous prompt cannot warm the first
    # request of a later cache-enabled cell.
    ornith_cases = itertools.product(
        args.quantizations, (False, True), (False, True), (False, True), args.prompts
    )
    done = completed_keys(results)
    for quant, cache, concise, fast, prompt_name in ornith_cases:
        if args.limit is not None and args.executed >= args.limit:
            return
        key = feature_key(
            family="ornith", quant=quant, cache=cache, mtp=False, fast=fast,
            concise=concise, prompt_name=prompt_name,
        )
        if key in done:
            print(f"SKIP {key}", flush=True)
            continue
        port = 18180 + quant
        label = (
            f"features-ornith-q{quant}-cache{int(cache)}-fast{int(fast)}-"
            f"concise{int(concise)}-{prompt_name}"
        )
        server = start_server(
            output=output, family="ornith", quant=quant, port=port,
            cache=cache, concise=concise, mtp=False, label=label,
        )
        try:
            model = manifest_api_model(MODEL_PATHS["ornith"][quant])
            base_url = f"http://127.0.0.1:{port}/v1"
            offset = server.log_path.stat().st_size
            run_feature_case(
                args=args, output=output, results=results, prompts=prompts,
                family="ornith", quant=quant, cache=cache, fast=fast,
                concise=concise, mtp=False, prompt_name=prompt_name,
                base_url=base_url, model=model, server=server, log_offset=offset,
            )
        finally:
            server.stop()

    # MTP: off/on must use the matching Qwen target and draft sidecar. Run both
    # target quantizations; target verification preserves correctness.
    for quant, mtp, prompt_name in itertools.product(
        args.quantizations, (False, True), args.prompts
    ):
        if args.limit is not None and args.executed >= args.limit:
            return
        key = feature_key(
            family="qwen", quant=quant, cache=False, mtp=mtp, fast=False,
            concise=False, prompt_name=prompt_name,
        )
        if key in done:
            print(f"SKIP {key}", flush=True)
            continue
        port = 18280 + quant
        label = f"features-qwen-q{quant}-mtp{int(mtp)}-{prompt_name}"
        server = start_server(
            output=output, family="qwen", quant=quant, port=port,
            cache=False, concise=False, mtp=mtp, label=label,
        )
        try:
            model = manifest_api_model(MODEL_PATHS["qwen"][quant])
            base_url = f"http://127.0.0.1:{port}/v1"
            offset = server.log_path.stat().st_size
            run_feature_case(
                args=args, output=output, results=results, prompts=prompts,
                family="qwen", quant=quant, cache=False, fast=False,
                concise=False, mtp=mtp, prompt_name=prompt_name,
                base_url=base_url, model=model, server=server, log_offset=offset,
            )
        finally:
            server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=("coder", "features", "all"), default="all")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--quantizations", nargs="+", type=int, choices=(4, 8), default=[4, 8])
    parser.add_argument("--clients", nargs="+", choices=("codex", "qwen", "opencode", "claude"))
    parser.add_argument("--prompts", nargs="+", choices=("short", "medium", "long"),
                        default=["short", "medium", "long"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--no-warmup", dest="warmups", action="store_false")
    parser.set_defaults(warmups=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    args.executed = 0
    return args


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / ".build/benchmark-rounds" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = fixed_prompts()
    prompt_dir = output / "prompts"
    for name, prompt in prompts.items():
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / f"{name}.txt").write_text(prompt)
    required = [MODEL_PATHS["ornith"][quant] for quant in args.quantizations]
    if args.round in ("features", "all"):
        required += [MODEL_PATHS["qwen"][quant] for quant in args.quantizations] + [MTP_PATH]
    environment = preflight(required)
    binaries = client_binaries()
    environment["client_versions"] = {
        name: command_version(binary) if binary else "not installed"
        for name, binary in binaries.items()
    }
    environment["prompt_sha256"] = {name: sha256_text(value) for name, value in prompts.items()}
    environment["arguments"] = vars(args) | {"output": str(output)}
    write_json(output / f"environment-{args.round}.json", environment)
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", environment)
    print(f"OUTPUT {output}", flush=True)
    try:
        if args.round in ("coder", "all"):
            run_coder_round(args, output, prompts, output / "coder-results.jsonl")
        if args.round in ("features", "all"):
            run_feature_round(args, output, prompts, output / "feature-results.jsonl")
    except KeyboardInterrupt:
        print(f"PAUSED {output}", flush=True)
        return 130
    print(f"COMPLETE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
