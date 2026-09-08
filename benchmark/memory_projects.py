"""S4: does memory keep two projects apart when they say similar things?

Workspace isolation is the assumption the whole memory design rests on -- one
server, several checkouts, and a novel's facts never in a codebase's bootstrap
-- and nothing measures it against a real model. The unit tests prove the
scope keys differ; they cannot prove that what the engine distilled from
project A stays out of project B's bootstrap.

Two projects are worked on alternately in one server lifetime, ten sessions,
A B A B ... Each is a data pipeline, and they are deliberately confusable:
both have a component called ingest, a retry policy, a primary datastore and a
release cadence, with different values in each. A leak that swaps two
unrelated facts is easy to spot in any transcript; the leak that actually
happens is Postgres for ClickHouse, and that is the one this scores.

Every session ends with an eight-key JSON quiz about the current project only,
scored twice:

    correct   the answer is this project's truth
    leaked    the answer is THE OTHER project's truth

An answer can be neither: absent, or a value neither project holds. The leak
count is what matters. A lost point is a memory that did not carry; a leak is
a defect, and one leak is worse than eight blanks.

Two arms:

    control   memory off. Scores near zero on carried facts by construction,
              and must leak nothing: the prompts of a session never contain
              the other project's values, so a leak here would be the scoring
              or the harness, not memory. This is the arm that makes the
              "auto" leak count mean something.
    auto      memory on, no memory tools, journal on: the engine writes.

    python3 benchmark/memory_projects.py control
    python3 benchmark/memory_projects.py report
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
OUT = Path(os.environ.get("NVMAI_MEMVAL_RESULTS", ROOT / ".build/benchmark-logs/memory-projects"))
PORT = int(os.environ.get("NVMAI_PORT", "8096"))
BASE = f"http://127.0.0.1:{PORT}/v1"
# Which run of the arm this is; results are kept per run so repeats can be
# compared and averaged. Repeats only mean something with sampling on:
# at temperature 0 a repeat is the same output.
RUN = os.environ.get("NVMAI_MEMVAL_RUN", "1")
TEMPERATURE = os.environ.get("NVMAI_MEMVAL_TEMPERATURE")  # unset: the server's default
SERVER_LOG = os.environ.get("NVMAI_MEMVAL_SERVER_LOG")
# The arm's memory directory. One .ndjson per workspace, so its filenames are
# the second, independent witness that the two projects were kept apart. Kept
# as the raw string: an unset value must read as "no directory", and Path("")
# is the working directory.
MEMDIR = os.environ.get("NVMAI_MEMVAL_MEMDIR", "")


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


def placements_logged() -> list[tuple[str, str]]:
    """Every (scope, via) the server logged for a session, in order.

    The server writes one `memory session=... scope=... tag=... via=...` line
    per session; `via` is "header", "declared-cwd" or "launch". This is the
    only direct view the harness has of where a request's memory actually
    went, and the whole scenario is worthless without it.
    """
    if not SERVER_LOG or not os.path.exists(SERVER_LOG):
        return []
    pattern = re.compile(r"session=\S+ scope=(\S+) tag=\S* via=(\S+)")
    with open(SERVER_LOG, errors="replace") as handle:
        return [(match.group(1), match.group(2))
                for match in (pattern.search(line) for line in handle) if match]


# Two pipelines with the same parts and different values. The header value is
# used verbatim as the workspace id, so these are also the .ndjson filenames.
PROJECTS = {
    "alpha": {
        "workspace": "proj-alpha",
        "name": "HALYARD",
        "truth": {
            "ingest_source": "kafka",
            "ingest_language": "rust",
            "retry_limit": "3",
            "retry_backoff": "exponential",
            "datastore": "postgres",
            "datastore_shards": "4",
            "release_cadence": "weekly",
            "release_day": "tuesday",
        },
    },
    "beta": {
        "workspace": "proj-beta",
        "name": "KESTREL",
        "truth": {
            "ingest_source": "s3",
            "ingest_language": "go",
            "retry_limit": "7",
            "retry_backoff": "fixed",
            "datastore": "clickhouse",
            "datastore_shards": "12",
            "release_cadence": "monthly",
            "release_day": "thursday",
        },
    },
}

# A third workspace, used once at the start of a memory arm to prove the
# header is what places a session. Nothing a project cares about is written
# there.
PROBE_WORKSPACE = "proj-probe"

QUIZ_KEYS = ("ingest_source", "ingest_language", "retry_limit", "retry_backoff",
             "datastore", "datastore_shards", "release_cadence", "release_day")

BRIEF = {
    "alpha": """\
These are HALYARD's settled facts. Nothing decided later may contradict them.

- The ingest component reads from Kafka and is written in Rust.
- ingest retries a failed batch 3 times, with exponential backoff.
- The primary datastore is Postgres, sharded 4 ways.
- Releases ship weekly, on Tuesday.
""",
    "beta": """\
These are KESTREL's settled facts. Nothing decided later may contradict them.

- The ingest component reads from S3 and is written in Go.
- ingest retries a failed batch 7 times, with fixed backoff.
- The primary datastore is ClickHouse, sharded 12 ways.
- Releases ship monthly, on Thursday.
""",
}

# One list of work items, used by both projects in the same order, so the only
# thing that differs between a HALYARD session and the KESTREL session beside
# it is the project name, the brief in round 1, and the workspace header. A
# leak therefore cannot be blamed on the wording of the work.
#
# No item names a value from either brief: after round 1 the answers are in
# memory or nowhere, which is what makes the control's near-zero score the
# floor rather than a measurement of the prompt.
WORK = (
    "Sketch, in five bullet points, how ingest should backfill a day it missed.",
    "Write two short paragraphs on what a failed ingest batch should record "
    "before it gives up.",
    "List four checks to run before the next release ships.",
    "Write a six-line checklist for adding a column to the primary datastore.",
    "Name three metrics the on-call dashboard should show for ingest, and why.",
)
SESSIONS = 2 * len(WORK)


def project_of(session: int) -> str:
    """Sessions alternate A, B, A, B ... so neither project is ever the one
    the server saw last for more than a session at a time."""
    return "alpha" if session % 2 else "beta"


def quiz(project: str) -> str:
    name = PROJECTS[project]["name"]
    return (
        f"Finally, answer this quiz about {name} -- this project only, never "
        f"another project you remember -- as a JSON object in a ```json block "
        f"with exactly these keys: ingest_source (what ingest reads from); "
        f"ingest_language; retry_limit (a number); retry_backoff (one word); "
        f"datastore; datastore_shards (a number); release_cadence (one word); "
        f"release_day (a weekday). Use null for anything you do not know. Do "
        f"not guess: a guessed value is worse than null.")


def session_prompt(session: int) -> str:
    project = project_of(session)
    name = PROJECTS[project]["name"]
    round_index = (session - 1) // 2
    parts = [f"You are the engineering assistant for {name}, a data pipeline."]
    if round_index == 0:
        parts.append(BRIEF[project])
    parts.append(WORK[round_index])
    parts.append(quiz(project))
    return "\n\n".join(parts)


def post(messages, model, workspace, max_tokens=1200):
    body = json.dumps({"model": model, "messages": messages,
                       "max_completion_tokens": max_tokens,
                       **sampling()}).encode()
    # The workspace header is the whole scenario: it is what makes two
    # conversations against one server two projects.
    request = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "X-NVMAI-Workspace": workspace})
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


def assert_arm_is_real(arm: str, prompt: str, prompt_tokens: int):
    """The arm the harness names must be the arm the server is running."""
    placed = placements_logged()
    if arm == "control":
        if placed:
            raise SystemExit(
                f"ABORT: arm 'control' but the server placed sessions in memory "
                f"({placed[0][0]} via {placed[0][1]}). Memory is on in the arm "
                f"whose whole job is to have none. Check NVMAI_MEMORY.")
        # Without a server log there is nothing authoritative to check, so
        # fall back to size: the memory system prompt fragment is ~90 tokens
        # and a control prompt should be about the prompt text and no more.
        # Four characters per token is a coarse estimate, hence the slack.
        estimate = len(prompt) // 4
        if prompt_tokens > estimate * 1.5 + 120:
            raise SystemExit(
                f"ABORT: arm 'control' saw {prompt_tokens} prompt tokens for a "
                f"~{estimate}-token prompt; something is being prepended. "
                f"Check NVMAI_MEMORY and NVMAI_MEMORY_TOOLS.")
    elif SERVER_LOG and not placed:
        raise SystemExit(
            f"ABORT: arm '{arm}' claims memory but the server logged no session "
            f"placement. Rebuild the release binary and check NVMAI_MEMORY.")


def assert_header_is_honoured(model: str):
    """Four tokens that prove X-NVMAI-Workspace decides the workspace.

    Two sessions of a 35B model are ten minutes; this finds an ignored header
    in seconds, which is the difference between a rerun and an afternoon. The
    probe names a third workspace, so whatever it writes lands in neither
    project's store, and its prompt declares no working directory: the only
    thing that could put it in `proj-probe` is the header.
    """
    post([{"role": "user", "content": "Say OK."}], model, PROBE_WORKSPACE, max_tokens=4)
    for _ in range(15):
        for scope, via in placements_logged():
            if scope == PROBE_WORKSPACE and via == "header":
                return
        time.sleep(2)
    placed = placements_logged()
    raise SystemExit(
        f"ABORT: a request carrying X-NVMAI-Workspace: {PROBE_WORKSPACE} was placed "
        f"in {sorted({scope for scope, _ in placed}) or 'no logged workspace'} "
        f"(via {sorted({via for _, via in placed}) or '-'}). The header is not "
        f"choosing the workspace, so both projects would share one store and the "
        f"run would report their shared facts as leaks.")


def assert_projects_are_separate():
    """Both projects must exist, separately, from the server's point of view.

    This is the check the whole result depends on. If the header were ignored
    -- and it is the client's only way to name a workspace, so nothing else
    would say so -- both projects would land in one store, every carried fact
    would be visible to both, and the run would report a catastrophic leak
    that is really a harness bug. Better to stop after two sessions.
    """
    workspaces = {PROJECTS[key]["workspace"] for key in PROJECTS}
    placed = placements_logged()
    if placed:
        scopes = {scope for scope, _ in placed}
        sources = {via for _, via in placed}
        if sources != {"header"}:
            raise SystemExit(
                f"ABORT: the server placed sessions via {sorted(sources)}, not "
                f"'header'. X-NVMAI-Workspace is not reaching the placement "
                f"decision, so both projects share a store.")
        if not workspaces <= scopes:
            raise SystemExit(
                f"ABORT: expected scopes {sorted(workspaces)}, saw {sorted(scopes)}. "
                f"The two projects are not separate.")
        return
    # No log: the journal filenames say the same thing, one file per
    # workspace, and by now both projects have been consolidated once.
    files = {path.stem for path in Path(MEMDIR).rglob("*.ndjson")} if MEMDIR else set()
    if not workspaces <= files:
        raise SystemExit(
            f"ABORT: expected journals {sorted(workspaces)} under {MEMDIR}, "
            f"saw {sorted(files) or 'none'}. The two projects are not separate.")


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


# What each project's value looks like when a model writes it in prose. Both
# projects' spellings are recognised for every key, because recognising only
# the current project's would make every leak read as a blank -- and the leak
# is the measurement.
VOCABULARY = {
    "ingest_source": (("kafka", r"kafka"), ("s3", r"\bs3\b|object stor")),
    "ingest_language": (("rust", r"\brust\b"), ("go", r"\bgo(lang)?\b")),
    "retry_backoff": (("exponential", r"exponential|doubling"),
                      ("fixed", r"fixed|constant|flat")),
    "datastore": (("postgres", r"postgre"), ("clickhouse", r"click\s*house")),
    "release_cadence": (("weekly", r"weekly|every week"),
                        ("monthly", r"monthly|every month")),
    "release_day": (("tuesday", r"tuesday"), ("thursday", r"thursday")),
}


def normalise(key: str, value) -> str | None:
    """The answer as one of the two projects' vocabularies, or None.

    None covers both "did not answer" and "answered something neither project
    holds"; the raw answer is kept in the results and printed with the leaks,
    so nothing is lost by collapsing them here.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in ("null", "none", "unknown", "n/a", "-"):
        return None
    if key in ("retry_limit", "datastore_shards"):
        # A number in prose: "3 attempts", "sharded 12 ways".
        digits = re.search(r"\d+", text)
        return str(int(digits.group())) if digits else None
    for canonical, pattern in VOCABULARY[key]:
        if re.search(pattern, text):
            return canonical
    return None


def leaks(project: str, answers: dict) -> list[dict]:
    """Every answer that is the other project's truth."""
    mine = PROJECTS[project]["truth"]
    theirs = PROJECTS["beta" if project == "alpha" else "alpha"]["truth"]
    found = []
    for key in QUIZ_KEYS:
        value = normalise(key, answers.get(key))
        if value is not None and value == theirs[key] and value != mine[key]:
            found.append({"key": key, "answered": answers.get(key),
                          "mine": mine[key], "theirs": theirs[key]})
    return found


def score(project: str, answers: dict) -> tuple[int, int, int]:
    mine = PROJECTS[project]["truth"]
    correct = sum(1 for key in QUIZ_KEYS if normalise(key, answers.get(key)) == mine[key])
    return correct, len(leaks(project, answers)), len(QUIZ_KEYS)


def run_arm(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    workspaces = {key: PROJECTS[key]["workspace"] for key in PROJECTS}
    if len(set(workspaces.values())) != len(workspaces):
        raise SystemExit(f"ABORT: the projects share a workspace id: {workspaces}")
    memory_on = arm != "control"
    # A memory arm with no way to see where the server put a session cannot
    # tell isolation from a harness bug, and would report the bug as a leak.
    if memory_on and not SERVER_LOG and not MEMDIR:
        raise SystemExit(
            "ABORT: set NVMAI_MEMVAL_SERVER_LOG or NVMAI_MEMVAL_MEMDIR. Without "
            "one of them this run cannot show that the two projects went to two "
            "workspaces, and an ignored X-NVMAI-Workspace header would be "
            "reported as a total leak.")
    model = model_id()
    if memory_on and SERVER_LOG:
        assert_header_is_honoured(model)
    results = []
    for session in range(1, SESSIONS + 1):
        project = project_of(session)
        workspace = workspaces[project]
        prompt = session_prompt(session)
        # A fresh conversation every session. Anything a later session knows
        # about its project came from memory, and anything it knows about the
        # other project came from the other project's memory.
        seen = consolidations_logged()
        result = post([{"role": "user", "content": prompt}], model, workspace)
        answers = extract_quiz(result["content"])
        correct, leaked, total = score(project, answers)
        result.update(session=session, project=project, workspace=workspace,
                      answers=answers, correct=correct, leaked=leaked, total=total)
        (OUT / f"{arm}-r{RUN}-{session:02d}-{project}.md").write_text(result["content"])
        print(f"{arm}/session {session:2d} {project:5s} ({workspace}): "
              f"{result['completion_tokens']} tokens, {result['seconds']:.0f}s, "
              f"prompt {result['prompt_tokens']}, quiz {correct}/{total}, "
              f"leaked {leaked}")
        if session == 1:
            assert_arm_is_real(arm, prompt, result["prompt_tokens"])
        result["consolidation_wait"] = wait_for_consolidation(seen) if memory_on else 0.0
        results.append(result)
        (OUT / f"{arm}-r{RUN}.json").write_text(json.dumps(results, indent=2))
        # Session 2 is the first moment both projects have been seen and
        # written; from here the run is only worth continuing if they are two.
        if session == 2 and memory_on:
            assert_projects_are_separate()


def report():
    runs = {}
    for path in sorted(OUT.glob("*-r*.json")):
        arm, run = path.stem.rsplit("-r", 1)
        if arm in ARMS:
            runs.setdefault(arm, {})[run] = json.loads(path.read_text())

    print(f"\n{'arm':8s} {'run':>3s} {'session':>7s} {'project':7s} {'prompt':>7s} "
          f"{'completion':>11s} {'seconds':>8s} {'wait':>5s} {'quiz':>6s} {'leak':>4s}")
    totals = {}
    for arm in ARMS:
        for run, results in sorted(runs.get(arm, {}).items()):
            for result in results:
                project = result["project"]
                # Rescored from the stored answers, never from the number the
                # run wrote down, so a scoring fix applies to every version
                # identically.
                correct, leaked, total = score(project, result["answers"])
                print(f"{arm:8s} {run:>3s} {result['session']:7d} {project:7s} "
                      f"{result['prompt_tokens']:7d} {result['completion_tokens']:11d} "
                      f"{result['seconds']:8.0f} {result.get('consolidation_wait', 0):5.0f} "
                      f"{correct:3d}/{total:<2d} {leaked:4d}")
                row = totals.setdefault(arm, {"correct": 0, "total": 0,
                                              "carried_correct": 0, "carried_total": 0,
                                              "leaks": []})
                row["correct"] += correct
                row["total"] += total
                # Sessions 1 and 2 carry their project's brief in the prompt;
                # everything after them is memory or nothing.
                if result["session"] > 2:
                    row["carried_correct"] += correct
                    row["carried_total"] += total
                for leak in leaks(project, result["answers"]):
                    row["leaks"].append((run, result["session"], project, leak))

    print("\nCorrect, and leaked (an answer that is the other project's truth):")
    for arm in ARMS:
        row = totals.get(arm)
        if not row:
            continue
        overall = f"{100 * row['correct'] / row['total']:.0f}%" if row["total"] else "n/a"
        carried = (f"{100 * row['carried_correct'] / row['carried_total']:.0f}%"
                   if row["carried_total"] else "n/a")
        print(f"  {arm:8s} overall {overall:>4s} ({row['correct']}/{row['total']}), "
              f"carried {carried:>4s} "
              f"({row['carried_correct']}/{row['carried_total']}), "
              f"leaks {len(row['leaks'])}")

    print("\nLeaks, one line each:")
    any_leak = False
    for arm in ARMS:
        for run, session, project, leak in totals.get(arm, {}).get("leaks", []):
            any_leak = True
            other = "beta" if project == "alpha" else "alpha"
            print(f"  {arm:8s} r{run} s{session:02d} {project:5s} {leak['key']}: "
                  f"answered {leak['answered']!r}; "
                  f"{PROJECTS[project]['name']}={leak['mine']}, "
                  f"{PROJECTS[other]['name']}={leak['theirs']}")
    if not any_leak:
        print("  none")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command in ARMS:
        run_arm(command)
    else:
        report()
