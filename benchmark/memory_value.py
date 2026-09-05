"""Does persistent memory help a coding agent across sessions?

The experiment: build Pong with two autoplaying computer players in Swift,
then port it to Python, then to C99. Three sessions, not three turns, because
memory's whole claim is about what survives when the conversation ends.

The prompts for stages 2 and 3 deliberately do not restate what stage 1
decided. A model with no memory has to invent the field size, the win score
and the paddle strategy again; a model with memory can carry them. That gap
is the measurement.

Two arms, identical prompts:

    control   memory off
    minimal   memory on, two tools (set, get), journal on
    full      memory on, all six tools, journal on

The third arm exists because the tool schemas, not the memory fragment, are
what memory actually costs: measured on this model the six definitions are
about 1,120 prompt tokens against roughly 210 for the fragment. If two tools
carry the same decisions, most of the tax was avoidable.

What is measured, and why it differs by store:

  Curated memory can change what the model writes, so it gets an A/B and a
  carry-over rate over the parameters stage 1 fixed.

  The journal is never injected into context, so it cannot change output at
  all. Measuring it for quality would be a category error. It is measured on
  what it captured, what that cost, and whether a later question can be
  answered from it.

    python3 benchmark/memory_value.py control
    python3 benchmark/memory_value.py memory
    python3 benchmark/memory_value.py report
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# control: memory off. minimal: bootstrap plus memory_set/memory_get.
# full: bootstrap plus all six tools.
ARMS = ("control", "auto", "minimal", "full")
OUT = ROOT / ".build/benchmark-logs/memory-value"
PORT = int(os.environ.get("NVMAI_PORT", "8096"))
BASE = f"http://127.0.0.1:{PORT}/v1"
# Which run of the arm this is; results are kept per run so repeats can be
# compared and averaged. Repeats only mean something with sampling on:
# at temperature 0 a repeat is the same output.
RUN = os.environ.get("NVMAI_MEMVAL_RUN", "1")
TEMPERATURE = os.environ.get("NVMAI_MEMVAL_TEMPERATURE")  # unset: the server's default
SERVER_LOG = os.environ.get("NVMAI_MEMVAL_SERVER_LOG")


def sampling():
    return {} if TEMPERATURE is None else {"temperature": float(TEMPERATURE)}


def consolidations_logged() -> int:
    if not SERVER_LOG or not os.path.exists(SERVER_LOG):
        return -1
    with open(SERVER_LOG, errors="replace") as handle:
        return sum(1 for line in handle if "consolidated session=" in line)


def wait_for_consolidation(before: int, limit: float = 300) -> float:
    """Waits for the server to distil the session that just ended.

    A real user pauses between sessions; the idle timer runs in that pause.
    The harness has no pause, so it waits for the log line instead. Returns
    the seconds spent waiting, which are reported as part of the arm's cost.
    """
    if before < 0:
        return 0.0
    started = time.time()
    while time.time() - started < limit:
        if consolidations_logged() > before:
            return time.time() - started
        time.sleep(2)
    print("  (no consolidation observed within the wait)")
    return time.time() - started

# The parameters stage 1 is free to choose and stages 2 and 3 must match if
# anything carried. The model states them as JSON, by name, so the harness
# never has to guess what a number in prose refers to. The first version of
# this harness read the prose with regular expressions and made "clamped to
# ±70°" register as a field height; the second was tightened until it
# matched nothing. A named field is the only thing that measures.
PARAMETERS = (
    "field_width", "field_height", "win_score",
    "ball_start_speed", "ball_speed_increment", "ball_max_speed",
    "paddle_speed",
)

RULES_JSON = (
    "a JSON object, in a ```json block, with exactly these keys and numeric "
    "values: " + ", ".join(PARAMETERS)
)

# Stage 1 fixes the design. Stages 2 and 3 say "the same rules" and nothing
# more: that phrase is the whole experiment, because only memory can supply
# what it refers to.
STAGES = [
    (
        "swift",
        "Write a complete Pong game in Swift with two computer players that play "
        "each other automatically. No human input at all. You decide the field "
        "size, the winning score, how each paddle's AI tracks the ball, and how "
        f"ball speed changes over a rally. First state those decisions as {RULES_JSON}, "
        "then give the full code in one Swift file.",
    ),
    (
        "python",
        "Port that Pong game to Python. Keep exactly the same game rules and "
        f"behaviour. First state the rules you are implementing as {RULES_JSON}, "
        "then give the full code in one file.",
    ),
    (
        "c99",
        "Now port the same Pong game to C99. Keep exactly the same game rules and "
        f"behaviour. First state the rules you are implementing as {RULES_JSON}, "
        "then give the full code in one file.",
    ),
]


def post(messages, model, max_tokens=5200):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **sampling(),
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.load(response)
    elapsed = time.time() - started
    choice = payload["choices"][0]["message"]
    usage = payload.get("usage", {})
    return {
        "content": choice.get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "seconds": elapsed,
    }


def model_id():
    with urllib.request.urlopen(f"{BASE}/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def run_arm(arm: str):
    """One arm: three independent sessions, no history carried between them."""
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_id()
    results = []
    memory_on = arm != "control"
    for stage, prompt in STAGES:
        # A fresh conversation each time. Anything that survives came from
        # memory, not from the context window.
        seen = consolidations_logged()
        result = post([{"role": "user", "content": prompt}], model)
        result["stage"] = stage
        (OUT / f"{arm}-r{RUN}-{stage}.md").write_text(result["content"])
        print(f"{arm}/{stage}: {result['completion_tokens']} tokens, "
              f"{result['seconds']:.1f}s, prompt {result['prompt_tokens']}")
        if stage == "swift":
            assert_arm_is_real(arm, result["prompt_tokens"])
        # The session is over. With consolidation on, the server distils it
        # in the pause a person would leave here; the harness waits for it.
        result["consolidation_wait"] = wait_for_consolidation(seen) if memory_on else 0.0
        results.append(result)
    (OUT / f"{arm}-r{RUN}.json").write_text(json.dumps(results, indent=2))


def assert_arm_is_real(arm: str, prompt_tokens: int):
    """Refuses to measure an arm that is not what it claims to be.

    The bare stage-1 prompt is under 150 tokens. The memory fragment alone
    adds about 200; the tool schemas add hundreds more. A memory arm whose
    prompt is the size of the control's is a server that was started without
    memory -- the last time that happened it was a release binary that had
    not been rebuilt, and three arms of numbers were compared before anyone
    noticed they were the same arm.
    """
    # auto: memory on with no tools -- the fragment alone, ~200 tokens.
    floor = {"control": 0, "auto": 200, "minimal": 300, "full": 800}[arm]
    ceiling = {"control": 250, "auto": 700, "minimal": 10_000, "full": 10_000}[arm]
    if not floor <= prompt_tokens <= ceiling:
        raise SystemExit(
            f"ABORT: arm '{arm}' saw {prompt_tokens} prompt tokens at stage 1; "
            f"expected {floor}..{ceiling}. The server is not running the "
            f"configuration this arm claims. Rebuild the release binary and "
            f"check NVMAI_MEMORY / NVMAI_MEMORY_TOOLS.")


def extract(text: str) -> dict:
    """Reads the parameters out of the stage's JSON rules block.

    The first JSON object that carries any of the named keys wins. Values are
    normalised to numbers so "5" and 5.0 agree, because the question is
    whether the rule carried, not how it was spelled.
    """
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += re.findall(r"(\{[^{}]*\})", text, re.S)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        found = {}
        for name in PARAMETERS:
            value = parsed.get(name)
            if isinstance(value, bool) or value is None:
                continue
            try:
                found[name] = float(value)
            except (TypeError, ValueError):
                continue
        if found:
            return found
    return {}


def code_block(text: str) -> str:
    blocks = re.findall(r"```[a-zA-Z0-9+]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else ""


def compiles(stage: str, code: str) -> bool | None:
    """Whether the stage's code builds or parses. None when not attempted."""
    if not code.strip():
        return False
    work = OUT / "compile"
    work.mkdir(parents=True, exist_ok=True)
    try:
        if stage == "swift":
            path = work / "pong.swift"
            path.write_text(code)
            done = subprocess.run(["swiftc", "-typecheck", str(path)],
                                  capture_output=True, timeout=180)
            return done.returncode == 0
        if stage == "python":
            path = work / "pong.py"
            path.write_text(code)
            done = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                                  capture_output=True, timeout=120)
            return done.returncode == 0
        if stage == "c99":
            path = work / "pong.c"
            path.write_text(code)
            done = subprocess.run(["cc", "-std=c99", "-fsyntax-only", str(path)],
                                  capture_output=True, timeout=180)
            return done.returncode == 0
    except Exception:
        return False
    return None


def report():
    runs = {}
    for path in sorted(OUT.glob("*-r*.json")):
        arm, run = path.stem.rsplit("-r", 1)
        if arm in ARMS:
            runs.setdefault(arm, {})[run] = json.loads(path.read_text())

    print(f"\n{'arm':8s} {'run':>3s} {'stage':8s} {'prompt':>7s} {'completion':>11s} "
          f"{'seconds':>8s} {'wait':>5s}  parameters")
    carry = {}
    for arm in ARMS:
        for run, results in sorted(runs.get(arm, {}).items()):
            baseline = {}
            for result in results:
                stage = result["stage"]
                values = extract(result["content"])
                print(f"{arm:8s} {run:>3s} {stage:8s} {result['prompt_tokens']:7d} "
                      f"{result['completion_tokens']:11d} {result['seconds']:8.1f} "
                      f"{result.get('consolidation_wait', 0):5.0f}  {values}")
                if stage == "swift":
                    baseline = values
                else:
                    shared = set(values) & set(baseline)
                    agreed = sum(1 for k in shared if values[k] == baseline[k])
                    carry.setdefault(arm, {}).setdefault(run, [0, 0])
                    carry[arm][run][0] += agreed
                    carry[arm][run][1] += len(shared)

    print("\nCarry-over per run: parameters fixed in the Swift stage that the later "
          "stages reproduce, without the prompt restating them.")
    for arm in ARMS:
        per_run = carry.get(arm, {})
        if not per_run:
            continue
        cells = [f"r{run} {a}/{s}" for run, (a, s) in sorted(per_run.items())]
        total_a = sum(a for a, _ in per_run.values())
        total_s = sum(s for _, s in per_run.values())
        mean = f"{100 * total_a / total_s:.0f}%" if total_s else "n/a"
        print(f"  {arm:8s} {mean:>5s}   {'  '.join(cells)}")

    print("\nCost per run (prompt + completion tokens, seconds incl. consolidation waits):")
    for arm in ARMS:
        for run, results in sorted(runs.get(arm, {}).items()):
            prompt = sum(r["prompt_tokens"] for r in results)
            completion = sum(r["completion_tokens"] for r in results)
            seconds = sum(r["seconds"] + r.get("consolidation_wait", 0) for r in results)
            print(f"  {arm:8s} r{run}: {prompt} + {completion}, {seconds:.0f}s")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command in ARMS:
        run_arm(command)
    else:
        report()
