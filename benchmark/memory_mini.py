#!/usr/bin/env python3
"""Can a small resident model keep the memory instead of the 35B?

Consolidation is the one recurring inference cost the memory feature has: at
every session boundary the served model reads the session back and writes out
what must not be contradicted, for 1.5-2k prompt tokens and 45-55 seconds on a
35B at 4-bit. That is the whole overhead of the feature, and it is spent on a
task -- turn prose into short addressed facts -- that is extraction, not
reasoning. A few hundred megabytes of resident model can do extraction.

This replays the recorded sessions through a small local model (llama.cpp,
CPU, so the GPU stays with the 35B), puts its output in a store, and reads
that store with exactly the reader memory_sim.py uses for the 35B's own
writes. Same transcripts, same reader, same scoring: the only thing that
changes is who did the extracting.

Two jobs are measured, because the simulation says they are not equally hard:

  full     the whole session, user turn and model output, as the 35B sees it
  capture  the user's turn alone -- short, authoritative, and the job worth
           +2.9 points on the projected book score

    benchmark/memory_mini.py serve 350m-extract     # start a model
    benchmark/memory_mini.py run 350m-extract --runs 3
    benchmark/memory_mini.py report
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import memory_book as book  # noqa: E402
import memory_sim as sim  # noqa: E402

GGUF = ROOT / "models/gguf"
RESULTS = ROOT / ".build/benchmark-logs/memory-mini"
PORT = int(os.environ.get("NVMAI_MINI_PORT", "8098"))

MODELS = {
    "350m-extract": GGUF / "LFM2-350M-Extract-Q4_K_M.gguf",
    "230m": GGUF / "LFM2.5-230M-Q4_K_M.gguf",
    "1.2b": GGUF / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    "qwen0.8b": GGUF / "Qwen3.5-0.8B-Q4_K_M.gguf",
    "qwen2b": GGUF / "Qwen3.5-2B-Q4_K_M.gguf",
}

# The same question the engine asks the 35B, in the shape a small extraction
# model wants it: a named output format and a schema. Nothing here names the
# quiz -- the keys are the model's to choose, as they are for the 35B, or the
# comparison would be against an answer sheet rather than an extraction.
SYSTEM = (
    "You extract durable facts from a writing session so a later session cannot "
    "contradict them. Return a JSON object mapping short keys to short values. "
    "A key looks like \"characters/marcus_eyes\", \"setting/town\", "
    '"state/inn", "rules/ferry". A value is a few words. Record fixed '
    "attributes, rules, and the current state of anything that changed. "
    "Do not invent facts that are not in the text."
)

CAPTURE_SYSTEM = (
    "You extract the facts the user has stated, so they can be stored and never "
    "contradicted later. Return a JSON object mapping short keys to short "
    "values. A key looks like \"characters/marcus_eyes\", \"setting/town\", "
    '"state/inn", "rules/ferry". A value is a few words. Record only what the '
    "user's message states or requires. Do not invent anything."
)


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

def serve(name: str) -> None:
    path = MODELS[name]
    if not path.exists():
        raise SystemExit(f"missing {path}")
    log = RESULTS / f"server-{name}.log"
    RESULTS.mkdir(parents=True, exist_ok=True)
    # CPU only and few threads on purpose: the 35B owns the GPU, and this has
    # to be able to run beside it without taking the machine.
    command = [
        "/opt/homebrew/bin/llama-server", "-m", str(path), "-ngl", "0",
        "-t", os.environ.get("NVMAI_MINI_THREADS", "2"), "-c", "8192",
        "--port", str(PORT), "--host", "127.0.0.1",
    ]
    with log.open("w") as handle:
        subprocess.Popen(command, stdout=handle, stderr=handle,
                         start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as reply:
                if b"ok" in reply.read():
                    print(f"{name} ready on {PORT}")
                    return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"{name} did not come up; see {log}")


def complete(system: str, user: str, max_tokens: int = 400) -> tuple[str, dict, float]:
    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        # Sampling is a knob because it is not obvious which way it should
        # go for this job. Greedy is reproducible, which a memory keeper
        # wants; the vendor's recommended profile is tuned for chat, and a
        # verifier that samples can disagree with itself between turns.
        "temperature": float(os.environ.get("NVMAI_MINI_TEMP", "0")),
        "top_p": float(os.environ.get("NVMAI_MINI_TOP_P", "1")),
        "repeat_penalty": float(os.environ.get("NVMAI_MINI_REPEAT", "1")),
        "max_tokens": max_tokens,
        # Qwen3.5 thinks by default and the thinking lands in
        # reasoning_content, so a budget sized for the answer is spent
        # before the answer starts. The memory keeper is an extraction job;
        # it does not need to deliberate, and a shadow agent that costs a
        # thousand tokens of reasoning per turn is not a shadow agent.
        "chat_template_kwargs": {
            "enable_thinking": os.environ.get("NVMAI_MINI_THINK") == "1"},
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=300) as reply:
        answer = json.loads(reply.read())
    message = answer["choices"][0]["message"]
    # A thinking model that ignored the switch still says something useful;
    # take the reasoning when the content came back empty.
    text = message.get("content") or message.get("reasoning_content") or ""
    return text, answer.get("usage", {}), time.time() - started


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_facts(text: str) -> dict[str, str]:
    """Whatever JSON object the model managed. A small model's output is not
    always clean, and a memory that only works on clean output is not a
    memory; the parser recovers what it can, exactly as the server's own
    consolidation parser does."""
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        return {}
    depth, end = 0, None
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    blob = text[start:end] if end else text[start:] + "}"
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        # Salvage the complete "key": "value" pairs from a truncated object.
        parsed = {}
        for key, value in re.findall(r'"([^"]{2,80})"\s*:\s*"([^"]{1,200})"', blob):
            parsed[key] = value
    return {str(k): str(v) for k, v in parsed.items()
            if isinstance(parsed, dict) and not isinstance(v, (dict, list))}


def session_input(run: dict, session: int, job: str) -> str:
    """What the extractor is shown. `full` is the session as the engine sees
    it; `capture` is the user's turn alone."""
    prompt = book.session_prompt(session, None)
    # The quiz is the harness talking to itself, not part of the book.
    prompt = prompt.split("Answer the continuity quiz")[0].strip()
    if job == "capture":
        return prompt
    return prompt + "\n\nThe session produced:\n" + run["text"].get(session, "")[:6000]


def run_model(name: str, limit: int, jobs: tuple[str, ...]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = sim.load_runs()[:limit]
    for job in jobs:
        out = RESULTS / f"{name}-{job}.json"
        record = {"model": name, "job": job, "runs": {}}
        for run in runs:
            per_session = {}
            for session in range(1, 11):
                if session not in run["text"]:
                    continue
                system = CAPTURE_SYSTEM if job == "capture" else SYSTEM
                try:
                    text, usage, seconds = complete(
                        system, session_input(run, session, job))
                except (urllib.error.URLError, TimeoutError) as error:
                    print(f"  {run['name']} s{session}: {error}")
                    continue
                facts = parse_facts(text)
                per_session[session] = {
                    "facts": facts, "seconds": seconds,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "raw": text[:400] if not facts else "",
                }
                print(f"  {name}/{job} {run['name']} s{session}: "
                      f"{len(facts):2d} facts, {seconds:.1f}s, "
                      f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tok")
            record["runs"][run["name"]] = per_session
            out.write_text(json.dumps(record, indent=1))
    print(f"wrote {RESULTS}")


# --------------------------------------------------------------------------
# Scoring: the same store, the same reader
# --------------------------------------------------------------------------

def store_from(per_session: dict, upto: int) -> dict[str, dict]:
    """The mini-model's facts as a store, replayed in session order under the
    v3 rule (last write wins), so the number is comparable with the 35B's."""
    store: dict[str, dict] = {}
    order = 0
    for session in range(1, upto + 1):
        for address, value in (per_session.get(str(session))
                               or per_session.get(session) or {}).get("facts", {}).items():
            order += 1
            store[address] = {"address": address, "value": str(value),
                              "session": session, "order": order,
                              "author": "mini", "user": False}
    return store


def report() -> None:
    runs = {run["name"]: run for run in sim.load_runs()}
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("server-"):
            continue
        record = json.loads(path.read_text())
        right = seen = 0
        seconds = prompt = completion = calls = 0.0
        empty = 0
        for name, per_session in record["runs"].items():
            if name not in runs:
                continue
            for value in per_session.values():
                seconds += value["seconds"]
                prompt += value["prompt_tokens"]
                completion += value["completion_tokens"]
                calls += 1
                if not value["facts"]:
                    empty += 1
            for session in range(2, 11):
                store = store_from(per_session, session)
                truth = book.truth(session)
                for key in book.QUIZ_KEYS:
                    seen += 1
                    if sim.read(store, key, session) == truth[key]:
                        right += 1
        if seen:
            rows.append((record["model"], record["job"], 100 * right / seen,
                         seconds / calls if calls else 0,
                         (prompt + completion) / calls if calls else 0,
                         100 * empty / calls if calls else 0, len(record["runs"])))
    print("small resident model as the memory keeper -- store fidelity read by\n"
          "the same reader as the 35B's own extraction (v3 = 89%, and the\n"
          "capture ceiling measured offline = 100%)\n")
    print(f"  {'model':14s} {'job':8s} {'fidelity':>9s} {'s/session':>10s} "
          f"{'tok/session':>12s} {'no JSON':>8s} {'runs':>5s}")
    for model, job, fidelity, seconds, tokens, empty, count in rows:
        print(f"  {model:14s} {job:8s} {fidelity:8.0f}% {seconds:9.1f}s "
              f"{tokens:11.0f} {empty:7.0f}% {count:5d}")
    print("\n  the 35B spends 45-55 s and about 1.8k tokens per session on this.")


# --------------------------------------------------------------------------
# The shadow verifier
# --------------------------------------------------------------------------

VERIFY_SYSTEM = (
    "You check an assistant's answer against established facts. You are given "
    "FACTS (true) and an ANSWER. Return a JSON object {\"wrong\": [keys]} "
    "listing only the answer keys that contradict the facts. If the answer "
    "agrees with the facts, return {\"wrong\": []}. Do not guess: list a key "
    "only when a fact plainly says otherwise."
)


def verify(limit: int = 6) -> None:
    """Could a small model catch the big model's continuity errors?

    This is the job the extraction test says a 350M cannot do -- but it is a
    different job. Extraction has to produce the fact; verification only has
    to notice that two statements disagree, with both in front of it. The
    measurement is precision and recall against answers whose correctness the
    quiz already established, so a false alarm is as visible as a miss.

    A verifier that flags correct answers is worse than none: it would have
    the engine rewrite good replies. Precision is therefore the number that
    decides whether this can ever act by itself rather than merely flag.
    """
    runs = sim.load_runs()[:limit]
    caught = missed = false_alarm = quiet = 0
    per_key: dict[str, list[int]] = {}
    for run in runs:
        for session in range(2, 11):
            answers = run["answers"].get(session) or {}
            if not answers:
                continue
            truth = book.truth(session)
            facts = "\n".join(f"- {key}: {truth[key]}" for key in book.QUIZ_KEYS)
            given = "\n".join(
                f"- {key}: {book.normalise(key, answers.get(key))}"
                for key in book.QUIZ_KEYS if key in answers)
            actually_wrong = {key for key in book.QUIZ_KEYS
                              if key in answers
                              and book.normalise(key, answers[key]) != truth[key]}
            try:
                text, _, _ = complete(
                    VERIFY_SYSTEM, f"FACTS\n{facts}\n\nANSWER\n{given}",
                    max_tokens=int(os.environ.get("NVMAI_MINI_MAXTOK", "200")))
            except Exception as error:
                print(f"  {run['name']} s{session}: {error}")
                continue
            flagged = set()
            parsed = parse_facts(text.replace("[", '["').replace("]", '"]')) if False else None
            match = re.search(r'"wrong"\s*:\s*\[([^\]]*)\]', text)
            if match:
                # A small model often answers "marcus_eyes: hazel" where the
                # key alone was asked for. Taking the key prefix is reading
                # its answer, not grading its formatting.
                flagged = set()
                for name in match.group(1).split(","):
                    name = name.strip().strip('"\' ').split(":")[0].strip()
                    if name:
                        flagged.add(name)
            flagged &= set(book.QUIZ_KEYS)
            for key in flagged & actually_wrong:
                caught += 1
                per_key.setdefault(key, [0, 0])[0] += 1
            for key in actually_wrong - flagged:
                missed += 1
                per_key.setdefault(key, [0, 0])[1] += 1
            false_alarm += len(flagged - actually_wrong)
            quiet += len(set(book.QUIZ_KEYS) - flagged - actually_wrong)
            print(f"  {run['name']} s{session}: wrong={sorted(actually_wrong)} "
                  f"flagged={sorted(flagged)}")
    recall = 100 * caught / (caught + missed) if caught + missed else 0
    precision = 100 * caught / (caught + false_alarm) if caught + false_alarm else 0
    print(f"\n  errors caught {caught}, missed {missed}  -> recall {recall:.0f}%")
    print(f"  false alarms {false_alarm} on {quiet + false_alarm} correct answers"
          f"  -> precision {precision:.0f}%")
    print("\n  precision is what decides whether it may act alone; recall only\n"
          "  decides how much it is worth as a flag.")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command == "serve":
        serve(sys.argv[2])
    elif command == "run":
        limit = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 12
        jobs = ("full", "capture")
        if "--job" in sys.argv:
            jobs = (sys.argv[sys.argv.index("--job") + 1],)
        run_model(sys.argv[2], limit, jobs)
    elif command == "verify":
        verify(int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 6)
    else:
        report()


