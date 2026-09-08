"""Does persistent memory follow the user when the user changes their mind?

The two scenarios that exist measure memory as accumulation: the pong
benchmark carries decisions forward, the book benchmark keeps a bible from
drifting. Both reward remembering more. Neither ever contradicts itself, so
neither can catch the failure that makes a memory system worse than no
memory at all: the user revises a decision in session 5 and the system keeps
handing the model the session-1 value for the rest of the project. A stale
fact is not a missing fact. A missing fact makes the model ask; a stale one
makes it confidently build the wrong thing.

The project is LEDGERLINE, a small ledger service, designed over eight
sessions. Each session is a new conversation -- a rollover -- so nothing
survives between them except what the arm's mechanism carries. Session 1 is
the brief and states eight decisions. Sessions 2 and 3 do ordinary work on
them. Sessions 4, 5 and 6 each revise exactly one decision and say so.
Sessions 7 and 8 do work that only comes out right if the revisions stuck.
Every session ends with the same ten-key quiz answered as JSON.

The scoring is what makes this a different measurement. `truth(session)`
returns the answer that is correct AS OF that session, so a revision flips
the expected answer from that session on, and the report splits the score
into revised keys and unrevised keys. An arm can only win the revised half
by preferring the latest statement; an arm that injects everything it ever
learned and lets the model pick will score well on the unrevised half and
badly on the revised half, and that split is the result.

Two arms, identical prompts:

    control   memory off. Not a zero by construction the way it is in the
              book scenario: each revision session states its new value in
              the prompt, so control gets that key right in that session and
              wrong afterwards. That shape -- right once, then gone -- is the
              floor the memory arm has to beat.
    auto      memory on, no tools, journal on. Consolidation is the only
              writer, which is the configuration a user gets by default and
              the one where a stale fact is nobody's explicit mistake.

    python3 benchmark/memory_correct.py control
    python3 benchmark/memory_correct.py report
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "auto")
OUT = Path(os.environ.get("NVMAI_MEMVAL_RESULTS", ROOT / ".build/benchmark-logs/memory-correct"))
PORT = int(os.environ.get("NVMAI_PORT", "8096"))
BASE = f"http://127.0.0.1:{PORT}/v1"
# Which run of the arm this is; results are kept per run so repeats can be
# compared and averaged. Repeats only mean something with sampling on:
# at temperature 0 a repeat is the same output.
RUN = os.environ.get("NVMAI_MEMVAL_RUN", "1")
TEMPERATURE = os.environ.get("NVMAI_MEMVAL_TEMPERATURE")  # unset: the server's default
SERVER_LOG = os.environ.get("NVMAI_MEMVAL_SERVER_LOG")

SESSIONS = 8


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


BRIEF = """\
You are the engineer on LEDGERLINE, a small double-entry ledger service. This
is the project brief. Every decision in it holds until I say otherwise.

- Language: TypeScript, strict mode.
- Storage engine: SQLite, one database file per tenant.
- Naming convention: snake_case for every identifier -- table columns, JSON
  fields and function names alike.
- Concurrency model: a single-threaded event loop. No worker threads, no
  shared-memory parallelism.
- Target platform: Linux x86-64 containers only. Not macOS, not Windows.
- Error handling: throw. Every fallible operation raises an exception; no
  function returns an error value.
- Logging: structured JSON lines to stdout, one object per event. Never a
  log file.
- Tests: Jest, one spec file per module.
"""

# The work each session asks for. These deliberately restate nothing from the
# brief: naming, storage, error shape and logging all have to come from
# whatever the arm carries. The revision sessions state their new value and
# only that -- a revision that also repeated the other seven decisions would
# be re-briefing the model, not testing memory.
WORK = {
    1: "Sketch the module layout for the service: list the files and give one "
       "line each on what belongs in them.",
    2: "Write the accounts table definition and the query that fetches one "
       "account by tenant and account id, plus the migration file that "
       "creates the table.",
    3: "Write the helper that delivers an outbound webhook with three "
       "attempts and a growing delay between them, including the log line it "
       "emits per attempt.",
    4: "We are moving off the file-based store. The storage engine is now "
       "Postgres, one schema per tenant -- not SQLite. Rewrite the accounts "
       "table definition and the fetch-by-id query for it.",
    5: "I changed my mind about error handling. Nothing throws any more: "
       "every fallible function returns a value that is either an ok case or "
       "an err case, and the caller has to check which. Convert the webhook "
       "helper you wrote earlier to that shape.",
    6: "One more change: the naming convention is camelCase from now on -- "
       "columns, JSON fields and function names. Then write the transfer "
       "module: one function that moves an amount between two accounts "
       "inside a single transaction.",
    7: "Write the ledger entry append path: one row per side of a transfer, "
       "returning the pair that was written. Follow every decision that is "
       "currently in force.",
    8: "Write the test file for the transfer module, covering a successful "
       "transfer and an insufficient-funds failure.",
}

QUIZ_KEYS = (
    "project_name", "language", "storage_engine", "naming_convention",
    "concurrency_model", "target_platform", "errors_are_thrown",
    "logging_format", "test_framework", "revisions_count",
)

# The session from which each key's answer changes. A key absent here was
# never revised. This drives both truth() and the revised/unrevised split, so
# the two can never disagree about which keys are which.
REVISED_AT = {
    "storage_engine": 4,      # SQLite -> Postgres
    "errors_are_thrown": 5,   # throw -> return an ok/err value
    "naming_convention": 6,   # snake_case -> camelCase
    # Not a decision of its own: the count of decisions changed so far. It is
    # in the revised bucket because it is only ever wrong for the same reason
    # the others are -- the model was handed a past that never changed.
    "revisions_count": 4,
}

QUIZ = (
    "Finally, answer this quiz about the project as it stands right now, as a "
    "JSON object in a ```json block with exactly these keys: "
    "project_name; language; storage_engine; naming_convention; "
    "concurrency_model (a few words); target_platform (an operating system); "
    "errors_are_thrown (true/false: does a failing operation throw?); "
    "logging_format (\"json\" or \"text\"); test_framework; revisions_count "
    "(an integer: how many of my earlier decisions have I since changed?). "
    "Answer with what is in force now, not with what was decided first."
)


def truth(session: int) -> dict:
    """What the quiz answers should be at the end of this session.

    A revision takes effect in the session that states it, not the one after:
    the model is told the new value before it is asked the quiz, so getting
    it wrong in that same session is a comprehension failure, and getting it
    wrong later is a memory failure. The report separates the two by session.
    """
    return {
        "project_name": "ledgerline",
        "language": "typescript",
        "storage_engine": "postgres" if session >= 4 else "sqlite",
        "naming_convention": "camelcase" if session >= 6 else "snake_case",
        "concurrency_model": "event loop",
        "target_platform": "linux",
        "errors_are_thrown": session < 5,
        "logging_format": "json",
        "test_framework": "jest",
        # 0 through session 3, then one more per revision session, capped
        # once the last revision has happened.
        "revisions_count": max(0, min(session - 3, 3)),
    }


def superseded(key: str):
    """The value this key held immediately before it was revised.

    Derived from truth() rather than written out again, so there is one place
    where a fact's before and after are defined. Used only for reporting:
    a wrong answer that is exactly the old value is a resurrection, which is
    a different and worse failure than a wrong answer that is noise.
    """
    if key not in REVISED_AT:
        return None
    return truth(REVISED_AT[key] - 1)[key]


def session_prompt(session: int) -> str:
    parts = []
    if session == 1:
        parts.append(BRIEF)
    parts.append(WORK[session])
    parts.append(QUIZ)
    return "\n\n".join(parts)


def post(messages, model, max_tokens=2500):
    body = json.dumps({"model": model, "messages": messages,
                       "max_completion_tokens": max_tokens,
                       **sampling()}).encode()
    request = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.load(response)
    choice = payload["choices"][0]["message"]
    usage = payload.get("usage", {})
    return {"content": choice.get("content") or "",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "seconds": time.time() - started}


def model_id():
    with urllib.request.urlopen(f"{BASE}/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def assert_arm_is_real(arm: str, prompt_tokens: int):
    """Refuses to measure an arm that is not the configuration it claims.

    The session-1 prompt here is the brief plus the quiz: 1,400 characters,
    so 350 to 420 tokens bare. The memory fragment adds roughly 200 on top of
    that, measured on the pong scenario with the same server. The bands
    overlap deliberately -- they are not a token budget, they are a check for
    the failure that has actually happened, a server started without memory
    or a release binary that was never rebuilt, producing two arms of numbers
    that were the same arm. Widen them, do not delete them, if a real run
    trips one.
    """
    floor = {"control": 0, "auto": 480}[arm]
    ceiling = {"control": 700, "auto": 3000}[arm]
    if not floor <= prompt_tokens <= ceiling:
        raise SystemExit(
            f"ABORT: arm '{arm}' saw {prompt_tokens} prompt tokens in session 1; "
            f"expected {floor}..{ceiling}. The server is not running what this "
            f"arm claims. Rebuild the release binary and check NVMAI_MEMORY / "
            f"NVMAI_MEMORY_TOOLS.")


def extract_quiz(text: str) -> dict:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += re.findall(r"(\{[^{}]*\})", text, re.S)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and set(parsed) & set(QUIZ_KEYS):
            return parsed
    return {}


def normalise(key: str, value) -> str | bool | int | None:
    """Tolerant of wording, intolerant of meaning.

    Every rule here collapses phrasings that mean the same decision and
    nothing else. The one to be careful with is naming_convention: a model
    that has just been told to switch often answers "camelCase (was
    snake_case)", so the check looks for the new word first and only falls
    back to the old one. Getting that order wrong scores a correct answer as
    a resurrection, which is the exact error this scenario reports on.
    """
    if value is None:
        return None
    if key == "errors_are_thrown":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "yes", "throw", "throws"):
            return True
        if text in ("false", "no", "return", "returns"):
            return False
        return None
    if key == "revisions_count":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        digits = re.search(r"-?\d+", str(value))
        return int(digits.group()) if digits else None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if key == "project_name":
        return "ledgerline" if "ledgerline" in text.replace(" ", "") else text
    if key == "language":
        return "typescript" if "typescript" in text.replace(" ", "") else text
    if key == "storage_engine":
        # "PostgreSQL", "Postgres", "postgresql 16" are one answer.
        if "postgres" in text:
            return "postgres"
        return "sqlite" if "sqlite" in text.replace(" ", "") else text
    if key == "naming_convention":
        flat = text.replace(" ", "").replace("-", "").replace("_", "")
        if "camel" in flat:
            return "camelcase"
        return "snake_case" if "snake" in flat else text
    if key == "concurrency_model":
        flat = text.replace("-", " ")
        if "event loop" in flat or "single thread" in flat or "singlethread" in flat:
            return "event loop"
        return text
    if key == "target_platform":
        return "linux" if "linux" in text else text
    if key == "logging_format":
        # "structured JSON lines to stdout" and "json" are one answer; only a
        # claim of plain text is the other one.
        if "json" in text:
            return "json"
        return "text" if "text" in text or "plain" in text else text
    if key == "test_framework":
        return "jest" if "jest" in text else text
    return text


def bucket(key: str, session: int) -> str:
    """Which half of the report a key counts in for this session.

    A key that will be revised later is scored as unrevised until it is: up
    to that point it is an ordinary carried fact and behaves like one. The
    revised bucket is empty for sessions 1 to 3 by construction.
    """
    return "revised" if REVISED_AT.get(key, SESSIONS + 1) <= session else "unrevised"


def score(session: int, answers: dict) -> tuple[int, int, int, int]:
    """Returns correct, total, revised_correct, revised_total."""
    expected = truth(session)
    correct = revised_correct = revised_total = 0
    for key in QUIZ_KEYS:
        hit = normalise(key, answers.get(key)) == expected[key]
        correct += hit
        if bucket(key, session) == "revised":
            revised_total += 1
            revised_correct += hit
    return correct, len(QUIZ_KEYS), revised_correct, revised_total


def resurrected(session: int, answers: dict) -> list[str]:
    """Revised keys answered with the value that was superseded."""
    out = []
    for key in QUIZ_KEYS:
        if bucket(key, session) != "revised":
            continue
        old = superseded(key)
        if old is not None and normalise(key, answers.get(key)) == old:
            out.append(key)
    return out


def run_arm(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_id()
    results = []
    memory_on = arm != "control"
    for session in range(1, SESSIONS + 1):
        # A fresh conversation every session. Anything that survives came
        # from memory, never from the context window.
        seen = consolidations_logged()
        result = post([{"role": "user", "content": session_prompt(session)}], model)
        answers = extract_quiz(result["content"])
        correct, total, revised_correct, revised_total = score(session, answers)
        result.update(session=session, answers=answers, correct=correct, total=total,
                      revised_correct=revised_correct, revised_total=revised_total,
                      resurrected=resurrected(session, answers))
        (OUT / f"{arm}-r{RUN}-{session:02d}.md").write_text(result["content"])
        stale = ",".join(result["resurrected"]) or "-"
        print(f"{arm}/session {session}: {result['completion_tokens']} tokens, "
              f"{result['seconds']:.0f}s, prompt {result['prompt_tokens']}, "
              f"quiz {correct}/{total}, revised {revised_correct}/{revised_total}, "
              f"stale {stale}")
        if session == 1:
            assert_arm_is_real(arm, result["prompt_tokens"])
        # The session is over; with consolidation on the server distils it in
        # the pause a person would leave here, and the harness waits for it.
        result["consolidation_wait"] = wait_for_consolidation(seen) if memory_on else 0.0
        results.append(result)
        # Written every session, so an interrupted run still reports.
        (OUT / f"{arm}-r{RUN}.json").write_text(json.dumps(results, indent=2))


def report():
    runs = {}
    if OUT.is_dir():
        for path in sorted(OUT.glob("*-r*.json")):
            arm, run = path.stem.rsplit("-r", 1)
            if arm in ARMS:
                runs.setdefault(arm, {})[run] = json.loads(path.read_text())
    if not runs:
        print(f"No results in {OUT}. Run an arm first: "
              f"python3 benchmark/memory_correct.py {ARMS[0]}")
        return

    print(f"\n{'arm':8s} {'run':>3s} {'session':>7s} {'prompt':>7s} {'completion':>11s} "
          f"{'seconds':>8s} {'wait':>5s} {'quiz':>6s} {'revised':>8s}  wrong")
    summary_rows = []
    for arm in ARMS:
        for run, results in sorted(runs.get(arm, {}).items()):
            totals = dict(correct=0, total=0, revised_correct=0, revised_total=0,
                          unrevised_correct=0, unrevised_total=0, stale=0)
            prompt = completion = seconds = 0
            for result in results:
                session = result["session"]
                expected = truth(session)
                answers = result["answers"]
                # Rescored from the stored answers, never from the numbers the
                # run wrote down, so a scoring fix applies to every version
                # identically.
                correct, total, rev_correct, rev_total = score(session, answers)
                stale = resurrected(session, answers)
                if not answers:
                    detail = "(no quiz answered)"
                else:
                    detail = ", ".join(
                        k + ("!" if k in stale else "") for k in QUIZ_KEYS
                        if normalise(k, answers.get(k)) != expected[k]) or "-"
                print(f"{arm:8s} {run:>3s} {session:7d} {result['prompt_tokens']:7d} "
                      f"{result['completion_tokens']:11d} {result['seconds']:8.0f} "
                      f"{result.get('consolidation_wait', 0):5.0f} "
                      f"{correct:3d}/{total:<2d} {rev_correct:4d}/{rev_total:<3d}  {detail}")
                totals["correct"] += correct
                totals["total"] += total
                totals["revised_correct"] += rev_correct
                totals["revised_total"] += rev_total
                totals["unrevised_correct"] += correct - rev_correct
                totals["unrevised_total"] += total - rev_total
                totals["stale"] += len(stale)
                prompt += result["prompt_tokens"]
                completion += result["completion_tokens"]
                seconds += result["seconds"] + result.get("consolidation_wait", 0)
            summary_rows.append((arm, run, totals, prompt, completion, seconds))

    def percent(correct: int, total: int) -> str:
        return f"{100 * correct / total:.0f}%" if total else "n/a"

    # A "!" in the wrong-list above marks a revised key answered with the value
    # it replaced. The stale column counts those: an arm can lose the revised
    # half by guessing, but only a memory system loses it by remembering.
    print(f"\n{'arm':8s} {'run':>3s} {'overall':>8s} {'revised':>8s} {'unrevised':>10s} "
          f"{'stale':>6s}")
    for arm, run, t, *_ in summary_rows:
        print(f"{arm:8s} {run:>3s} "
              f"{percent(t['correct'], t['total']):>8s} "
              f"{percent(t['revised_correct'], t['revised_total']):>8s} "
              f"{percent(t['unrevised_correct'], t['unrevised_total']):>10s} "
              f"{t['stale']:6d}")
    print("\nPooled over runs:")
    for arm in ARMS:
        rows = [r for r in summary_rows if r[0] == arm]
        if not rows:
            continue
        pooled = {k: sum(r[2][k] for r in rows) for k in rows[0][2]}
        print(f"  {arm:8s} overall {percent(pooled['correct'], pooled['total']):>4s}   "
              f"revised {percent(pooled['revised_correct'], pooled['revised_total']):>4s}   "
              f"unrevised {percent(pooled['unrevised_correct'], pooled['unrevised_total']):>4s}   "
              f"stale answers {pooled['stale']}")
    print("\nCost per run (prompt + completion tokens, seconds incl. waits):")
    for arm, run, _, prompt, completion, seconds in summary_rows:
        print(f"  {arm:8s} r{run}: {prompt} + {completion}, {seconds:.0f}s")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command in ARMS:
        run_arm(command)
    else:
        report()
