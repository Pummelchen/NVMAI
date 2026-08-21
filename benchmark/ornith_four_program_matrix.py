#!/usr/bin/env python3
"""Run the four-program Ornith coding/tooling matrix.

Each matrix cell starts a new NVMAIServer process and gives each of four small
programs a fresh tool conversation. The model executes repairs through bounded
tools and is then scored on separate hidden inputs. The four hidden outputs
form one ordered final result, so a cell passes only when every program is
correct.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from coder_cli_benchmark import preflight, utc_now
from nvmai_profile import DEFAULT_API_MODEL, server_command, server_environment


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
DEFAULT_OUTPUT = ROOT / ".build/ornith-four-program-matrix"
DEFAULT_PORT = 18350
MODEL_PATHS = {
    4: ROOT / "models/ornith-1.5_35B_A3B_4Bit",
    8: ROOT / "models/ornith-1.5_35B_A3B_8Bit",
}
PYTHON_314 = pathlib.Path(os.environ.get(
    "NVMAI_BENCH_PYTHON_314",
    str(pathlib.Path.home() / ".venvs/tools/bin/python"),
))
TENSORFLOW_PYTHON = pathlib.Path(os.environ.get(
    "NVMAI_BENCH_TENSORFLOW_PYTHON",
    str(pathlib.Path.home() / ".venvs/tensorflow-metal-py311/bin/python"),
))
PYTORCH_PYTHON = pathlib.Path(os.environ.get(
    "NVMAI_BENCH_PYTORCH_PYTHON",
    shutil.which("python3.13") or "/opt/homebrew/bin/python3.13",
))

PROGRAM_ORDER = (
    "ShortestPath.swift",
    "NestedScore.py",
    "StackedLSTM.py",
    "MaskedAttention.py",
)
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
    "seed": 1234,
}
MAX_TURNS_PER_PROGRAM = 25
MATRIX_QUANT_BITS = (8, 4)
MATRIX_BOOLEAN_VALUES = (False, True)

SYSTEM_PROMPT = """You are a careful coding agent in a controlled tool-use benchmark.
Repair the one assigned file, execute it with run_program, and continue until its public check
passes. Other workspace files are context only. Use replace_workspace_text for compact edits
when practical. Never hard-code observed outputs: hidden inputs will exercise the algorithm.
Issue tool calls without narrating plans, then finish with one short factual verification
report."""

PROGRAM_PROMPTS = {
    "ShortestPath.swift": """Repair ShortestPath.swift using Swift 6.3.3. It parses an undirected
positive weighted graph: first line START END, remaining lines U V WEIGHT. Implement Dijkstra
and print the shortest distance, or -1 when unreachable. Read the starter, edit it, and run it.
Do not finish until run_program reports passed=true.""",
    "NestedScore.py": """Repair NestedScore.py using Python 3.14. Recursively traverse JSON with
structural pattern matching. Ignore booleans, multiply each integer by its nesting depth, and
sum it. Root depth is 0; list elements and dictionary values are one level deeper. Read, edit,
and run it. Do not finish until run_program reports passed=true.""",
    "StackedLSTM.py": """Repair StackedLSTM.py. Keep its deterministic initializers and CPU setup.
Build exactly two hidden TensorFlow LSTM layers: 3 units with return_sequences=True, then 2
units, followed by the provided Dense layer. Print the scalar to six decimals. Read, edit, and
run it. Do not finish until run_program reports passed=true.""",
    "MaskedAttention.py": """Repair MaskedAttention.py. Implement PyTorch scaled dot-product
self-attention: divide scores by sqrt(x.shape[-1]), mask invalid key columns with mask[None, :]
before softmax, apply softmax, and keep the provided valid-query pooling and six-decimal output.
Read, edit, and run it. Do not finish until run_program reports passed=true.""",
}

INITIAL_FILES = {
    "ShortestPath.swift": """import Foundation

let header = readLine()!.split(separator: " ").map(String.init)
let start = header[0], destination = header[1]
var graph: [String: [(String, Int)]] = [:]
while let line = readLine() {
    let fields = line.split(separator: " ")
    guard fields.count == 3, let weight = Int(fields[2]), weight > 0 else { continue }
    let left = String(fields[0]), right = String(fields[1])
    graph[left, default: []].append((right, weight))
    graph[right, default: []].append((left, weight))
}

var distances = [start: 0]
var visited = Set<String>()
// TODO: Implement Dijkstra and update distances.
print(distances[destination] ?? -1)
""",
    "NestedScore.py": """import json
import sys


def score(value, depth=0):
    # TODO: recursively score ints in lists and dictionary values; bool is ignored.
    return 0


print(score(json.load(sys.stdin)))
""",
    "StackedLSTM.py": """import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
values = json.load(sys.stdin)["sequence"]
x = tf.constant(values, dtype=tf.float32)[None, :, None]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 1)),
    tf.keras.layers.LSTM(
        3,
        return_sequences=False,  # TODO: first hidden layer must feed a second LSTM.
        kernel_initializer=tf.keras.initializers.Constant(0.2),
        recurrent_initializer=tf.keras.initializers.Constant(0.1),
        bias_initializer="zeros",
    ),
    # TODO: add the second 2-unit LSTM with kernel=0.15, recurrent=0.05, zero bias.
    tf.keras.layers.Dense(
        1,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(0.3),
    ),
])
print(f"{float(model(x, training=False).numpy()[0, 0]):.6f}")
""",
    "MaskedAttention.py": """import json
import math
import sys

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(2, 2, bias=False)
        self.k = nn.Linear(2, 2, bias=False)
        self.v = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            for layer in (self.q, self.k, self.v):
                layer.weight.copy_(torch.eye(2))

    def forward(self, x, mask):
        scores = self.q(x) @ self.k(x).transpose(-2, -1)  # TODO: scale by sqrt(x.shape[-1]) and mask keys with mask[None, :].
        context = scores @ self.v(x)  # TODO: normalize scores with softmax first.
        return context[mask].mean(dim=0).sum()


payload = json.load(sys.stdin)
tokens = torch.tensor(payload["tokens"], dtype=torch.float32)
mask = torch.tensor(payload["mask"], dtype=torch.bool)
print(f"{float(Attention()(tokens, mask).detach()):.6f}")
""",
}

PUBLIC_INPUTS = {
    "ShortestPath.swift": "A E\nA B 4\nA C 2\nC B 1\nB D 5\nC D 8\nD E 2\nB E 12\n",
    "NestedScore.py": '{"a":[2,{"b":3}],"c":4,"skip":true}\n',
    "StackedLSTM.py": '{"sequence":[0.5,-1.0,2.0,0.25]}\n',
    "MaskedAttention.py": '{"tokens":[[1,0],[0,1],[1,1]],"mask":[1,1,0]}\n',
}
EXPECTED_PUBLIC = {
    "ShortestPath.swift": "10\n",
    "NestedScore.py": "17\n",
    "StackedLSTM.py": "0.012331\n",
    "MaskedAttention.py": "1.000000\n",
}
HIDDEN_INPUTS = {
    "ShortestPath.swift": "S T\nS A 6\nS B 2\nB A 1\nA C 3\nB C 7\nC T 2\nA T 10\n",
    "NestedScore.py": '[1,{"x":[2,3]},4]\n',
    "StackedLSTM.py": '{"sequence":[1.0,0.0,-0.5]}\n',
    "MaskedAttention.py": '{"tokens":[[2,-1],[0.5,1.5],[-1,0]],"mask":[1,0,1]}\n',
}
EXPECTED_HIDDEN = {
    "ShortestPath.swift": "8\n",
    "NestedScore.py": "20\n",
    "StackedLSTM.py": "0.005375\n",
    "MaskedAttention.py": "0.100006\n",
}
EXPECTED_FINAL = "8 | 20 | 0.005375 | 0.100006"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": "List all files in the isolated workspace.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": "Read one UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_workspace_text",
            "description": "Replace one exact text occurrence in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_workspace_file",
            "description": "Replace one complete UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_program",
            "description": "Execute one program with its fixed public input and grade stdout.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ServerProcess:
    process: subprocess.Popen[str]
    log_handle: Any
    command: list[str]
    log_path: pathlib.Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log_handle.close()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def safe_path(workspace: pathlib.Path, raw: Any) -> pathlib.Path:
    if not isinstance(raw, str) or not raw or len(raw) > 120:
        raise ValueError("path must be a non-empty string of at most 120 characters")
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path must stay inside the workspace")
    path = (workspace / pathlib.Path(*relative.parts)).resolve()
    if not path.is_relative_to(workspace.resolve()):
        raise ValueError("path escapes the workspace")
    return path


def runtime_command(path: pathlib.Path) -> tuple[list[str], dict[str, str]]:
    environment = dict(os.environ)
    if path.name == "ShortestPath.swift":
        return ["xcrun", "swift", str(path)], environment
    if path.name == "NestedScore.py":
        return [str(PYTHON_314), str(path)], environment
    if path.name == "StackedLSTM.py":
        environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
        return [str(TENSORFLOW_PYTHON), str(path)], environment
    if path.name == "MaskedAttention.py":
        return [str(PYTORCH_PYTHON), str(path)], environment
    raise ValueError(f"unsupported program {path.name}")


def run_source(path: pathlib.Path, stdin: str, timeout: int = 120) -> dict[str, Any]:
    command, environment = runtime_command(path)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=path.parent,
            env=environment,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8_000:],
            "stderr": completed.stderr[-8_000:],
            "wall_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(
            exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": command,
            "exit_code": 124,
            "stdout": stdout[-8_000:],
            "stderr": (stderr + "\nexecution timed out")[-8_000:],
            "wall_seconds": round(time.monotonic() - started, 3),
        }


def execute_tool(name: str, arguments: dict[str, Any], workspace: pathlib.Path) -> dict[str, Any]:
    if name == "list_workspace_files":
        return {
            "ok": True,
            "files": sorted(
                str(path.relative_to(workspace))
                for path in workspace.rglob("*") if path.is_file()
            ),
        }
    path = safe_path(workspace, arguments.get("path"))
    if name == "read_workspace_file":
        if not path.is_file():
            return {"ok": False, "error": "file does not exist"}
        return {"ok": True, "content": path.read_text()}
    if path.name not in PROGRAM_ORDER:
        return {"ok": False, "error": "only the four benchmark programs may be edited or run"}
    if name == "replace_workspace_text":
        old, new = arguments.get("old"), arguments.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return {"ok": False, "error": "old and new must be strings; old cannot be empty"}
        content = path.read_text()
        count = content.count(old)
        if count != 1:
            return {"ok": False, "error": f"old text occurrence count was {count}, expected 1"}
        path.write_text(content.replace(old, new, 1))
        return {"ok": True, "bytes_written": path.stat().st_size}
    if name == "write_workspace_file":
        content = arguments.get("content")
        if not isinstance(content, str) or len(content.encode()) > 64_000:
            return {"ok": False, "error": "content must be UTF-8 text no larger than 64 KiB"}
        path.write_text(content)
        return {"ok": True, "bytes_written": len(content.encode())}
    if name == "run_program":
        if not path.is_file():
            return {"ok": False, "error": "file does not exist"}
        result = run_source(path, PUBLIC_INPUTS[path.name])
        result["expected_stdout"] = EXPECTED_PUBLIC[path.name]
        result["passed"] = (
            result["exit_code"] == 0 and result["stdout"] == EXPECTED_PUBLIC[path.name])
        result["ok"] = result["passed"]
        return result
    return {"ok": False, "error": f"unknown tool {name}"}


def wait_ready(port: int, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError(f"server on port {port} did not become ready")


def start_server(output: pathlib.Path, quant: int, concise: bool,
                 thinking: bool, port: int, label: str) -> ServerProcess:
    mode = "on" if thinking else "off"
    command = server_command(
        SERVER, port, model=MODEL_PATHS[quant], thinking_mode=mode)
    environment = server_environment(concise=concise, thinking_mode=mode)
    log_path = output / "server" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    server = ServerProcess(process, log_handle, command, log_path)
    try:
        wait_ready(port)
    except Exception:
        server.stop()
        raise
    return server


def post_chat(port: int, messages: list[dict[str, Any]], timeout: int) -> tuple[dict[str, Any], float]:
    payload = {
        "model": DEFAULT_API_MODEL,
        "messages": messages,
        "tools": TOOLS,
        **SAMPLING,
        "max_tokens": 1_536,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer local"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
    return value, time.monotonic() - started


def run_tool_loop(workspace: pathlib.Path, port: int, timeout: int,
                  label: str, program: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROGRAM_PROMPTS[program]},
    ]
    trace: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    final_answer = ""
    error = None
    started = time.monotonic()
    model_wall = 0.0
    progress_label = f"{label}/{program}"
    for turn in range(1, MAX_TURNS_PER_PROGRAM + 1):
        print(f"PROGRESS {progress_label} turn={turn} requesting model", flush=True)
        try:
            response, wall = post_chat(port, messages, timeout)
            model_wall += wall
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("response had no choices")
            message = choices[0].get("message") or {}
            calls = message.get("tool_calls") or []
            usages.append(response.get("usage") or {})
            trace.append({
                "turn": turn,
                "kind": "assistant",
                "wall_seconds": round(wall, 3),
                "content": message.get("content") or "",
                "tool_call_count": len(calls),
                "finish_reason": choices[0].get("finish_reason"),
            })
            print(
                f"PROGRESS {progress_label} turn={turn} model_done={wall:.1f}s tools={len(calls)}",
                flush=True,
            )
            if not calls:
                final_answer = message.get("content") or ""
                if choices[0].get("finish_reason") == "length":
                    error = "assistant hit the response limit before completing the tool loop"
                break
            messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            })
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                raw_arguments = function.get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must decode to an object")
                    result = execute_tool(name, arguments, workspace)
                except Exception as exc:
                    arguments = {"_raw": raw_arguments}
                    result = {"ok": False, "error": repr(exc)}
                path = arguments.get("path", "-")
                print(
                    f"PROGRESS {progress_label} tool={name} path={path} ok={result.get('ok', False)}",
                    flush=True,
                )
                trace.append({
                    "turn": turn,
                    "kind": "tool",
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"missing-{turn}",
                    "name": name,
                    "content": json.dumps(result, sort_keys=True),
                })
        except Exception as exc:
            error = repr(exc)
            break
    else:
        error = f"tool loop exceeded {MAX_TURNS_PER_PROGRAM} assistant turns"
    return {
        "program": program,
        "wall_seconds": round(time.monotonic() - started, 3),
        "model_wall_seconds": round(model_wall, 3),
        "final_answer": final_answer,
        "trace": trace,
        "usage": usages,
        "error": error,
    }


def source_contract_failures(workspace: pathlib.Path) -> list[str]:
    failures: list[str] = []
    contents = {name: (workspace / name).read_text() for name in PROGRAM_ORDER}
    for name, content in contents.items():
        if "TODO" in content:
            failures.append(f"{name} retains TODO text")
    if "match " not in contents["NestedScore.py"]:
        failures.append("NestedScore.py does not use structural pattern matching")
    tensorflow = contents["StackedLSTM.py"]
    if len(re.findall(r"(?:layers\.)?LSTM\s*\(", tensorflow)) != 2:
        failures.append("StackedLSTM.py does not contain exactly two LSTM layers")
    if "return_sequences=True" not in tensorflow:
        failures.append("StackedLSTM.py first layer does not return sequences")
    attention = contents["MaskedAttention.py"]
    for required in ("sqrt", "masked_fill", "softmax"):
        if required not in attention:
            failures.append(f"MaskedAttention.py lacks {required}")
    return failures


def validate_workspace(workspace: pathlib.Path, trace: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    hidden_runs: dict[str, Any] = {}
    for name in PROGRAM_ORDER:
        path = workspace / name
        if not path.is_file():
            failures.append(f"missing {name}")
            continue
        result = run_source(path, HIDDEN_INPUTS[name])
        result["expected_stdout"] = EXPECTED_HIDDEN[name]
        result["passed"] = result["exit_code"] == 0 and result["stdout"] == EXPECTED_HIDDEN[name]
        hidden_runs[name] = result
        if not result["passed"]:
            failures.append(f"{name} failed hidden validation")
    if not failures:
        failures.extend(source_contract_failures(workspace))
    successful_model_runs = {
        (row.get("arguments") or {}).get("path")
        for row in trace
        if row.get("kind") == "tool"
        and row.get("name") == "run_program"
        and (row.get("result") or {}).get("passed")
    }
    missing_runs = sorted(set(PROGRAM_ORDER) - successful_model_runs)
    if missing_runs:
        failures.append("model did not successfully execute: " + ", ".join(missing_runs))
    outputs = [hidden_runs.get(name, {}).get("stdout", "").strip() for name in PROGRAM_ORDER]
    final_result = " | ".join(outputs)
    if final_result != EXPECTED_FINAL:
        failures.append("chained hidden output did not match the expected final result")
    return {
        "passed": not failures,
        "failures": failures,
        "hidden_runs": hidden_runs,
        "successful_model_runs": sorted(successful_model_runs),
        "final_result": final_result,
        "expected_final_result": EXPECTED_FINAL,
    }


def prepare_workspace(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    for name, content in INITIAL_FILES.items():
        (path / name).write_text(content)


def usage_totals(usages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(row.get(key, 0)) for row in usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def run_cell(output: pathlib.Path, index: int, total: int, quant: int,
             concise: bool, thinking: bool, port: int, timeout: int) -> dict[str, Any]:
    label = f"q{quant}-concise-{'on' if concise else 'off'}-thinking-{'on' if thinking else 'off'}"
    workspace = output / "workspaces" / label
    prepare_workspace(workspace)
    print(
        f"MATRIX {index}/{total} START {label} fresh_server=true fresh_conversation=true",
        flush=True,
    )
    server: ServerProcess | None = None
    started = time.monotonic()
    try:
        server = start_server(output, quant, concise, thinking, port, label)
        print(f"MATRIX {index}/{total} SERVER_READY {label}", flush=True)
        program_results: list[dict[str, Any]] = []
        for program_index, program in enumerate(PROGRAM_ORDER, 1):
            print(
                f"MATRIX {index}/{total} SESSION {program_index}/4 {label}/{program} fresh=true",
                flush=True,
            )
            program_results.append(
                run_tool_loop(workspace, port, timeout, label, program))
        trace = [
            {"program": result["program"], **row}
            for result in program_results for row in result["trace"]
        ]
        usages = [usage for result in program_results for usage in result["usage"]]
        errors = [
            f"{result['program']}: {result['error']}"
            for result in program_results if result["error"]
        ]
        quality = validate_workspace(workspace, trace)
        totals = usage_totals(usages)
        record = {
            "label": label,
            "matrix_index": index,
            "quant_bits": quant,
            "concise": concise,
            "thinking": thinking,
            "fresh_server": True,
            "fresh_conversation": True,
            "model": DEFAULT_API_MODEL,
            "model_path": str(MODEL_PATHS[quant]),
            "sampling": SAMPLING,
            "workspace": str(workspace.relative_to(output)),
            "quality": quality,
            "usage_totals": totals,
            "finished_at": utc_now(),
            "wall_seconds": round(time.monotonic() - started, 3),
            "model_wall_seconds": round(sum(
                result["model_wall_seconds"] for result in program_results), 3),
            "final_answer": {
                result["program"]: result["final_answer"] for result in program_results
            },
            "trace": trace,
            "usage": usages,
            "program_results": program_results,
            "error": "; ".join(errors) if errors else None,
        }
    except Exception as exc:
        record = {
            "label": label,
            "matrix_index": index,
            "quant_bits": quant,
            "concise": concise,
            "thinking": thinking,
            "fresh_server": True,
            "fresh_conversation": True,
            "model": DEFAULT_API_MODEL,
            "model_path": str(MODEL_PATHS[quant]),
            "sampling": SAMPLING,
            "workspace": str(workspace.relative_to(output)),
            "quality": {
                "passed": False,
                "failures": [repr(exc)],
                "final_result": "",
                "expected_final_result": EXPECTED_FINAL,
            },
            "usage_totals": {},
            "wall_seconds": round(time.monotonic() - started, 3),
            "model_wall_seconds": 0,
            "final_answer": "",
            "trace": [],
            "usage": [],
            "error": repr(exc),
            "finished_at": utc_now(),
        }
    finally:
        if server is not None:
            server.stop()
    write_json(output / "records" / f"{label}.json", record)
    append_jsonl(output / "results.jsonl", record)
    passed = record["quality"]["passed"] and not record.get("error")
    print(
        f"MATRIX {index}/{total} DONE {label} pass={passed} "
        f"wall={record['wall_seconds']}s final={record['quality'].get('final_result', '')!r}",
        flush=True,
    )
    return record


def matrix_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Quant | Concise | Thinking | Pass | Final result | Wall | Model wait | Completion tokens |",
        "| ---: | :---: | :---: | :---: | --- | ---: | ---: | ---: |",
    ]
    for row in records:
        quality = row["quality"]
        lines.append(
            f"| {row['quant_bits']}-bit | {'ON' if row['concise'] else 'OFF'} "
            f"| {'ON' if row['thinking'] else 'OFF'} "
            f"| {'PASS' if quality['passed'] and not row.get('error') else 'FAIL'} "
            f"| `{quality.get('final_result', '')}` | {row['wall_seconds']:.1f}s "
            f"| {row['model_wall_seconds']:.1f}s "
            f"| {row.get('usage_totals', {}).get('completion_tokens', 0)} |"
        )
    return "\n".join(lines) + "\n"


def runtime_facts() -> dict[str, Any]:
    commands = {
        "python_314": [str(PYTHON_314), "--version"],
        "tensorflow": [
            str(TENSORFLOW_PYTHON), "-c",
            "import sys,tensorflow as tf;print(sys.version.split()[0],tf.__version__)",
        ],
        "pytorch": [
            str(PYTORCH_PYTHON), "-c",
            "import sys,torch;print(sys.version.split()[0],torch.__version__)",
        ],
    }
    facts: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        facts[name] = {"command": command, "version": completed.stdout.strip()}
    return facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=1_800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or DEFAULT_OUTPUT / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = preflight(MODEL_PATHS.values())
    environment.update({
        "models": {str(bits): str(path) for bits, path in MODEL_PATHS.items()},
        "server": str(SERVER),
        "runtime_facts": runtime_facts(),
        "sampling": SAMPLING,
        "expected_final_result": EXPECTED_FINAL,
        "matrix": {
            "quant_bits": list(MATRIX_QUANT_BITS),
            "concise": list(MATRIX_BOOLEAN_VALUES),
            "thinking": list(MATRIX_BOOLEAN_VALUES),
            "cells": 8,
        },
        "protocol": {
            "context_tokens": 262_144,
            "prompt_cache": "multi-prefix",
            "prompt_cache_memory_mib": 256,
            "expert_cache_budget": "8G",
            "kv_bits": 8,
            "mtp": False,
            "fast_alias": False,
            "fresh_server_per_cell": True,
            "fresh_conversation_per_program": True,
            "max_turns_per_program": MAX_TURNS_PER_PROGRAM,
            "warmup": False,
        },
    })
    write_json(output / "environment.json", environment)
    print(f"OUTPUT {output}", flush=True)
    cells = list(itertools.product(
        MATRIX_QUANT_BITS, MATRIX_BOOLEAN_VALUES, MATRIX_BOOLEAN_VALUES))
    records: list[dict[str, Any]] = []
    try:
        for index, (quant, concise, thinking) in enumerate(cells, 1):
            records.append(run_cell(
                output, index, len(cells), quant, concise, thinking,
                args.port, args.timeout))
    except KeyboardInterrupt:
        print(f"PAUSED {output}", flush=True)
        return 130
    table = matrix_table(records)
    (output / "results.md").write_text(table)
    summary = {
        "passed": sum(row["quality"]["passed"] and not row.get("error") for row in records),
        "total": len(records),
        "expected_final_result": EXPECTED_FINAL,
        "generated_at": utc_now(),
        "table": table,
    }
    write_json(output / "summary.json", summary)
    print(table, flush=True)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
