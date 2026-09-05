"""Memory against a real model, in three requests.

Not a benchmark: a check that the wiring does what the unit tests say it
does when a 35B model is on the other end. A novel session stores a fact,
a codebase session must not see it, a second novel session must. Placement
is by the working directory the client declares, exactly as Claude Code
and Codex declare it, so this is the book-versus-git case end to end.

    python3 benchmark/memory_smoke.py full     # via benchmark/memval_run.sh smoke
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
    if (usage.get("prompt_tokens") or 0) < 800:
        failures.append(f"novel-1 prompt was {usage.get('prompt_tokens')} tokens: "
                        "memory tools are not in the prompt")

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
