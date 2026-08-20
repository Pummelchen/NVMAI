#!/usr/bin/env python3
"""Qualify Ornith Concise Mode on coding tools and self-scaffolding loops.

The A/B keeps NVMAI's production server profile fixed and changes only
``NVMAI_CONCISE_MODE``. Tool execution is deliberately narrow: files stay in
one case directory and the only executable actions are running a Python or
Swift source file. Results and complete tool traces live below ``.build``.

This tests whether the model can participate in a self-improvement data loop;
it does not update model weights or claim online learning during inference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from coder_cli_benchmark import preflight, utc_now
from nvmai_profile import (
    DEFAULT_API_MODEL,
    DEFAULT_MODEL_PATH,
    server_command,
    server_environment,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / ".build/arm64-apple-macosx/release/NVMAIServer"
DEFAULT_OUTPUT = ROOT / ".build/ornith-concise-tool-ab"
DEFAULT_PORT = 18340
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
    "seed": 1234,
}


SYSTEM_PROMPT = """You are a careful coding agent participating in a controlled evaluation.
Use the supplied tools to inspect, edit, and execute files. Continue until every requested
check succeeds. Never claim success without running the relevant program. Keep all work in
the provided workspace, preserve the requested interfaces, and finish with a short factual
report of what you changed and verified. Do not narrate plans or draft code in assistant text:
issue the next tool call immediately, put code in write_workspace_file, and reserve assistant
text for the final report after verification."""


CASE_PROMPTS = {
    "swift": """Repair Slug.swift. Its line-oriented interface must convert each UTF-8 input
line to a lowercase ASCII slug: fold diacritics, treat each run of non-alphanumeric characters
as one hyphen, and remove leading/trailing hyphens. Empty normalized input must produce an empty
line. Inspect the file, edit it, run it with the Swift tool, and fix any failure before answering.""",
    "python": """Repair stats.py. It must read one JSON array from stdin, accept only real finite
numbers (booleans are invalid), and print compact sorted-key JSON containing count, min, max, and
mean. Empty input must return null for min/max/mean. Invalid items must produce a useful stderr
message and a nonzero exit. Inspect, edit, run with the Python tool, and fix failures before answering.""",
    "swift_core": """Repair Scores.swift. It reads NAME: SCORE lines, trims surrounding whitespace,
ignores malformed rows, empty names, and non-integer scores, combines names case-insensitively,
sums repeated scores, and prints lowercase name=total rows sorted by name. Inspect the file, edit it,
run it with the Swift tool, and fix any failure before answering.""",
    "self_scaffold": """Perform one self-scaffolding learning cycle about robust text processing.

1. Propose a two-task curriculum in curriculum.json: Python tag normalization followed by Swift
   log-level counting. Each task needs language, title, objective, scaffold, self_chosen_edge_case,
   and a rubric array. Add one meaningful edge case of your own to each task.
2. Create initial TODO scaffolds named scaffold.py and scaffold.swift, then turn them into solutions.
   scaffold.py reads a JSON string array and prints sorted unique lowercase ASCII kebab-case tags.
   scaffold.swift reads LEVEL: message lines, ignores malformed/unknown levels, and prints exactly
   INFO=<n> WARN=<n> ERROR=<n>.
3. Run both solutions with their tools and repair failures.
4. Write reflection.json with outcomes for both tasks, lessons, and one concrete next_task proposal.

Do not skip the initial scaffold stage. Finish only after both executions succeed.""",
}


INITIAL_FILES = {
    "swift": {
        "Slug.swift": """import Foundation

func slugify(_ text: String) -> String {
    text.lowercased().replacingOccurrences(of: " ", with: "-")
}

while let line = readLine() {
    print(slugify(line))
}
""",
    },
    "python": {
        "stats.py": """import json
import sys

values = json.load(sys.stdin)
print(json.dumps({"count": len(values), "mean": sum(values) // len(values)}))
""",
    },
    "swift_core": {
        "Scores.swift": """import Foundation

var scores: [String: Int] = [:]
while let line = readLine() {
    let parts = line.split(separator: ":")
    if parts.count == 2, let score = Int(parts[1]) {
        scores[String(parts[0])] = score
    }
}
for (name, score) in scores {
    print("\\(name)=\\(score)")
}
""",
    },
    "self_scaffold": {},
}


PUBLIC_INPUTS = {
    "Slug.swift": " Hello, Swift 6! \nCafé déjà vu\n___A---B___\n!!!\n",
    "stats.py": "[1, 2.5, -3, 7]\n",
    "Scores.swift": " Ada : 3\nbob:5\nADA: 7\nbad\nbob: -2\n: 8\nLinus: nope\n",
    "scaffold.py": '["  Café au lait ", "swift_tools", "cafe-au-lait", "", "Swift tools"]\n',
    "scaffold.swift": "INFO: boot\nWARN: low disk\nUNKNOWN: skip\nmalformed\nERROR: stopped\nINFO: ready\n",
}


EXPECTED_PUBLIC = {
    "Slug.swift": "hello-swift-6\ncafe-deja-vu\na-b\n\n",
    "stats.py": '{"count":4,"max":7,"mean":1.875,"min":-3}\n',
    "Scores.swift": "ada=10\nbob=3\n",
    "scaffold.py": '["cafe-au-lait", "swift-tools"]\n',
    "scaffold.swift": "INFO=2 WARN=1 ERROR=1\n",
}


HIDDEN_INPUTS = {
    "Slug.swift": "  déjà__VU  \nRock & Roll\n42\n",
    "stats.py": "[]\n",
    "Scores.swift": "Zoe: 4\n amy : 2\nZOE:-1\namy:3\nnone\n:9\n",
    "scaffold.py": '["Déjà Vu", "deja_vu", "PYTHON  3", "python-3"]\n',
    "scaffold.swift": "ERROR: one\nINFO: ok\nDEBUG: no\nWARN: w1\nWARN: w2\n",
}


EXPECTED_HIDDEN = {
    "Slug.swift": "deja-vu\nrock-roll\n42\n",
    "stats.py": '{"count":0,"max":null,"mean":null,"min":null}\n',
    "Scores.swift": "amy=5\nzoe=3\n",
    "scaffold.py": '["deja-vu", "python-3"]\n',
    "scaffold.swift": "INFO=1 WARN=2 ERROR=1\n",
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": "List files in the isolated task workspace.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": "Read one UTF-8 file from the isolated task workspace.",
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
            "name": "write_workspace_file",
            "description": "Create or replace one UTF-8 file in the isolated task workspace.",
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
            "name": "run_python_file",
            "description": "Run one Python file with the case's fixed public stdin and show output.",
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
            "name": "run_swift_file",
            "description": "Run one Swift file with the case's fixed public stdin and show output.",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def wait_ready(port: int, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError(f"server on port {port} did not become ready")


def start_server(output: pathlib.Path, concise: bool, port: int) -> ServerProcess:
    label = "concise-on" if concise else "concise-off"
    command = server_command(
        SERVER, port, model=DEFAULT_MODEL_PATH, thinking_mode="off")
    environment = server_environment(concise=concise, thinking_mode="off")
    # Match the coding-client benchmark: rendered reasoning is disabled so
    # Concise Mode is the sole behavioral variable in this A/B.
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


def safe_path(workspace: pathlib.Path, value: Any) -> pathlib.Path:
    if not isinstance(value, str) or not value or len(value) > 120:
        raise ValueError("path must be a non-empty string of at most 120 characters")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path must stay inside the workspace")
    resolved = (workspace / pathlib.Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(workspace.resolve()):
        raise ValueError("path escapes the workspace")
    return resolved


def run_source(path: pathlib.Path, stdin: str, timeout: int = 120) -> dict[str, Any]:
    if path.suffix == ".py":
        command = [sys.executable, str(path)]
    elif path.suffix == ".swift":
        command = ["xcrun", "swift", str(path)]
    else:
        raise ValueError("only .py and .swift files can be executed")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=path.parent,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8_000:],
            "stderr": completed.stderr[-8_000:],
            "wall_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": 124,
            "stdout": (exc.stdout or "")[-8_000:],
            "stderr": ((exc.stderr or "") + "\nexecution timed out")[-8_000:],
            "wall_seconds": round(time.monotonic() - started, 3),
        }


def execute_tool(name: str, arguments: dict[str, Any], workspace: pathlib.Path) -> dict[str, Any]:
    if name == "list_workspace_files":
        return {
            "ok": True,
            "files": sorted(
                str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()
            ),
        }
    path = safe_path(workspace, arguments.get("path"))
    if name == "read_workspace_file":
        if not path.is_file():
            return {"ok": False, "error": "file does not exist"}
        return {"ok": True, "content": path.read_text()[:64_000]}
    if name == "write_workspace_file":
        content = arguments.get("content")
        if not isinstance(content, str) or len(content.encode()) > 64_000:
            return {"ok": False, "error": "content must be UTF-8 text no larger than 64 KiB"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"ok": True, "bytes_written": len(content.encode())}
    if name in ("run_python_file", "run_swift_file"):
        expected_suffix = ".py" if name == "run_python_file" else ".swift"
        if path.suffix != expected_suffix or not path.is_file():
            return {"ok": False, "error": f"expected an existing {expected_suffix} file"}
        result = run_source(path, PUBLIC_INPUTS.get(path.name, ""))
        expected = EXPECTED_PUBLIC.get(path.name)
        result["expected_stdout"] = expected
        result["passed"] = result["exit_code"] == 0 and result["stdout"] == expected
        result["ok"] = result["passed"]
        return result
    return {"ok": False, "error": f"unknown tool {name}"}


def post_chat(port: int, messages: list[dict[str, Any]], timeout: int) -> tuple[dict[str, Any], float]:
    payload = {
        "model": DEFAULT_API_MODEL,
        "messages": messages,
        "tools": TOOLS,
        **SAMPLING,
        "max_tokens": 2_048,
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


def run_tool_loop(case: str, workspace: pathlib.Path, port: int, timeout: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CASE_PROMPTS[case]},
    ]
    trace: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    final_answer = ""
    started = time.monotonic()
    error = None
    for turn in range(1, 13):
        try:
            response, wall = post_chat(port, messages, timeout)
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("response had no choices")
            message = choices[0].get("message") or {}
            usages.append(response.get("usage") or {})
            calls = message.get("tool_calls") or []
            trace.append({
                "turn": turn,
                "kind": "assistant",
                "wall_seconds": round(wall, 3),
                "content": message.get("content") or "",
                "tool_call_count": len(calls),
                "finish_reason": choices[0].get("finish_reason"),
            })
            if not calls:
                final_answer = message.get("content") or ""
                if choices[0].get("finish_reason") == "length":
                    error = "assistant hit the 2048-token response limit before a tool call"
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
        error = "tool loop exceeded 12 assistant turns"
    return {
        "case": case,
        "wall_seconds": round(time.monotonic() - started, 3),
        "final_answer": final_answer,
        "trace": trace,
        "usage": usages,
        "error": error,
    }


def validate_json_artifacts(workspace: pathlib.Path) -> list[str]:
    failures: list[str] = []
    try:
        curriculum = json.loads((workspace / "curriculum.json").read_text())
        tasks = curriculum.get("tasks") if isinstance(curriculum, dict) else curriculum
        if not isinstance(tasks, list) or len(tasks) != 2:
            failures.append("curriculum.json must contain two tasks")
        else:
            required = {
                "language", "title", "objective", "scaffold",
                "self_chosen_edge_case", "rubric",
            }
            for index, task in enumerate(tasks):
                if not isinstance(task, dict) or not required.issubset(task):
                    failures.append(f"curriculum task {index + 1} lacks required fields")
                elif not isinstance(task.get("rubric"), list) or not task["rubric"]:
                    failures.append(f"curriculum task {index + 1} has no rubric")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid curriculum.json: {exc}")
    try:
        reflection = json.loads((workspace / "reflection.json").read_text())
        if not isinstance(reflection, dict):
            failures.append("reflection.json must be an object")
        else:
            for key in ("outcomes", "lessons", "next_task"):
                if not reflection.get(key):
                    failures.append(f"reflection.json lacks {key}")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid reflection.json: {exc}")
    return failures


def validate_case(case: str, workspace: pathlib.Path, trace: list[dict[str, Any]]) -> dict[str, Any]:
    files = {
        "swift": ["Slug.swift"],
        "python": ["stats.py"],
        "swift_core": ["Scores.swift"],
        "self_scaffold": ["scaffold.py", "scaffold.swift"],
    }[case]
    failures: list[str] = []
    hidden_runs: dict[str, Any] = {}
    for name in files:
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
    tool_rows = [row for row in trace if row.get("kind") == "tool"]
    successful_runs = [
        row for row in tool_rows
        if row.get("name") in ("run_python_file", "run_swift_file")
        and (row.get("result") or {}).get("passed")
    ]
    if not successful_runs:
        failures.append("no successful model-requested execution")
    if case == "self_scaffold":
        failures.extend(validate_json_artifacts(workspace))
        written_names = [
            (row.get("arguments") or {}).get("path")
            for row in tool_rows if row.get("name") == "write_workspace_file"
        ]
        if written_names.count("scaffold.py") < 2 or written_names.count("scaffold.swift") < 2:
            failures.append("initial scaffold and solution stages were not both observable")
    return {
        "passed": not failures,
        "failures": failures,
        "hidden_runs": hidden_runs,
        "tool_calls": len(tool_rows),
        "successful_model_runs": len(successful_runs),
        "final_files": sorted(
            str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file()
        ),
    }


def prepare_workspace(path: pathlib.Path, case: str) -> None:
    path.mkdir(parents=True, exist_ok=False)
    for name, content in INITIAL_FILES[case].items():
        (path / name).write_text(content)


def run_profile(output: pathlib.Path, concise: bool, port: int, timeout: int,
                cases: list[str]) -> list[dict[str, Any]]:
    label = "concise-on" if concise else "concise-off"
    server = start_server(output, concise, port)
    records: list[dict[str, Any]] = []
    try:
        # Discarded warmup: load kernels and establish the same prompt-cache path.
        post_chat(port, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Reply with READY and do not call a tool."},
        ], timeout)
        for case in cases:
            workspace = output / "workspaces" / label / case
            prepare_workspace(workspace, case)
            print(f"RUN {label} {case}", flush=True)
            result = run_tool_loop(case, workspace, port, timeout)
            quality = validate_case(case, workspace, result["trace"])
            record = {
                "profile": label,
                "concise": concise,
                "model": DEFAULT_API_MODEL,
                "workspace": str(workspace.relative_to(output)),
                "sampling": SAMPLING,
                "quality": quality,
                "finished_at": utc_now(),
                **result,
            }
            write_json(output / "records" / f"{label}-{case}.json", record)
            append_jsonl(output / "results.jsonl", record)
            records.append(record)
            print(
                f"DONE {label} {case} pass={quality['passed']} "
                f"tools={quality['tool_calls']} wall={result['wall_seconds']}s",
                flush=True,
            )
    finally:
        server.stop()
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for label in ("concise-on", "concise-off"):
        rows = [record for record in records if record["profile"] == label]
        completion = sum(
            int(usage.get("completion_tokens", 0))
            for row in rows for usage in row["usage"]
        )
        prompt = sum(
            int(usage.get("prompt_tokens", 0))
            for row in rows for usage in row["usage"]
        )
        cached = sum(
            int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            for row in rows for usage in row["usage"]
        )
        profiles[label] = {
            "artifact_passed": sum(bool(row["quality"]["passed"]) for row in rows),
            "workflow_completed": sum(
                bool(row["quality"]["passed"] and not row["error"] and row["final_answer"])
                for row in rows
            ),
            "total": len(rows),
            "wall_seconds": round(sum(row["wall_seconds"] for row in rows), 3),
            "tool_calls": sum(row["quality"]["tool_calls"] for row in rows),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
        }
    return {"profiles": profiles, "generated_at": utc_now()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=1_800)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_PROMPTS),
        default=["swift_core", "python", "self_scaffold"],
        help="Cases to run in order (default: swift_core python self_scaffold)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("on", "off"),
        default=["on", "off"],
        help="Concise profiles to run in order (default: on off)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or DEFAULT_OUTPUT / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = preflight([DEFAULT_MODEL_PATH])
    environment.update({
        "model": str(DEFAULT_MODEL_PATH),
        "server": str(SERVER),
        "server_command": server_command(SERVER, args.port, model=DEFAULT_MODEL_PATH),
        "sampling": SAMPLING,
        "profiles": args.profiles,
        "cases": args.cases,
        "protocol": {
            "context_tokens": 262_144,
            "prompt_cache": "multi-prefix",
            "prompt_cache_memory_mib": 256,
            "expert_cache_budget": "8G",
            "kv_bits": 8,
            "mtp": False,
            "fast_alias": False,
            "thinking": "off",
            "warmup": "one discarded request per profile",
        },
        "purpose_note": (
            "Evaluates participation in a self-scaffolding data-generation loop; "
            "does not perform weight updates or online learning."
        ),
    })
    write_json(output / "environment.json", environment)
    print(f"OUTPUT {output}", flush=True)
    records: list[dict[str, Any]] = []
    try:
        for profile in args.profiles:
            records.extend(
                run_profile(output, profile == "on", args.port, args.timeout, args.cases)
            )
    except KeyboardInterrupt:
        print(f"PAUSED {output}", flush=True)
        return 130
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all(row["quality"]["passed"] and not row["error"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
