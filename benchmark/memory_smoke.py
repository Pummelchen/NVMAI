"""Memory against a real model, in three requests.

Not a benchmark: a check that the wiring does what the unit tests say it
does when a 35B model is on the other end. A novel session stores a fact,
a codebase session must not see it, a second novel session must. Placement
is by the working directory the client declares, exactly as Claude Code
and Codex declare it, so this is the book-versus-git case end to end.

    benchmark/memval_run.sh smoke      # memory on, no tools: the engine must do the writing
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("NVMAI_PORT", "8096"))
BASE = f"http://127.0.0.1:{PORT}/v1"
MEMDIR = Path(os.environ.get("NVMAI_MEMVAL_MEMDIR", ""))
SERVER_LOG = os.environ.get("NVMAI_MEMVAL_SERVER_LOG")


def consolidation_lines():
    if not SERVER_LOG or not os.path.exists(SERVER_LOG):
        return []
    with open(SERVER_LOG, errors="replace") as handle:
        return [line.strip() for line in handle if "consolidated session=" in line]

NOVEL = ("You are a coding assistant.\n\n# Environment\n"
         " - Primary working directory: /Users/ada/novels/photograph\n")
CODE = "<environment_context>\n  <cwd>/Users/ada/src/widget</cwd>\n</environment_context>"


def model_id():
    with urllib.request.urlopen(f"{BASE}/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def ask(model, system, user, label):
    body = json.dumps({"model": model, "temperature": 0, "max_completion_tokens": 400,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    request = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            status, payload = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.loads(error.read() or b"{}")
    elapsed = time.time() - started
    message = (payload.get("choices") or [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    usage = payload.get("usage", {})
    print(f"[{label}] HTTP {status}, {elapsed:.0f}s, prompt {usage.get('prompt_tokens')} "
          f"completion {usage.get('completion_tokens')}")
    print("   " + content[:300].replace("\n", " "))
    if status != 200:
        print("   ERROR:", json.dumps(payload)[:400])
    return status, content, usage


def main():
    model = model_id()
    failures = []

    status, reply, usage = ask(
        model, NOVEL,
        "Store this in memory for later sessions: in this novel the town is called "
        "Ashgrove and it never rains there. Then confirm in one sentence.", "novel-1")
    if status != 200:
        failures.append("novel-1 did not answer")
    # Memory on with no tools: the fragment is ~90 tokens on top of the
    # ~60-token request (measured 149). Under 120, memory is not in the
    # prompt at all.
    if (usage.get("prompt_tokens") or 0) < 120:
        failures.append(f"novel-1 prompt was {usage.get('prompt_tokens')} tokens: "
                        "the memory fragment is not in the prompt")

    # The novel session is over. With the runner's short idle, the engine
    # should distil it before the next request; a real person's pause does
    # the same. This is the write the model was measured not making.
    for _ in range(90):
        if consolidation_lines():
            break
        time.sleep(2)
    lines = consolidation_lines()
    print("consolidation:", lines[-1] if lines else "(none within 180s)")
    if not lines:
        failures.append("no consolidation of the novel session")
    elif "facts=0" in lines[-1]:
        failures.append("consolidation ran but wrote no facts")

    status, reply, _ = ask(
        model, CODE,
        "What do you already know about this project from memory? One sentence; "
        "say 'nothing' if nothing.", "code-1")
    if "ashgrove" in reply.lower():
        failures.append("the codebase session saw the novel's fact")

    status, reply, _ = ask(
        model, NOVEL,
        "What do you already know about this novel from memory? One sentence.", "novel-2")
    if "ashgrove" not in reply.lower():
        failures.append("the second novel session did not recall the fact")

    time.sleep(3)
    if MEMDIR.exists():
        files = sorted(str(p.relative_to(MEMDIR)) for p in MEMDIR.rglob("*.ndjson"))
        print("journal files:", files)
        if not any("photograph-" in f for f in files):
            failures.append("no photograph workspace file")
        if not any("widget-" in f for f in files):
            failures.append("no widget workspace file")

    if failures:
        print("\nSMOKE FAILED:")
        for failure in failures:
            print("  -", failure)
        raise SystemExit(1)
    print("\nSMOKE OK: placement by declared directory, fact carried within the "
          "project and not across it.")


if __name__ == "__main__":
    main()
