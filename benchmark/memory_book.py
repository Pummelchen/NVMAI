"""Does persistent memory keep a hundred-chapter novel consistent?

The failure this measures is the one that ships: chapter 3 fixes a
character's eyes as grey, chapter 71 makes them blue, and nobody notices
because chapter 3 left the context window sixty chapters ago.

The novel is written in ten sessions of ten chapters. Each session is a new
conversation -- a rollover -- so nothing survives between them except what
the arm's mechanism carries. Session 1 gets the bible. Later sessions get
only the chapter range and, on five of them, a plot event that happens in
that range. At the end of every session the model answers a fixed continuity
quiz as JSON, and the quiz is scored against the bible and against which
events have happened by then.

Three arms, identical chapter prompts:

    summary   no memory; the harness asks for a 200-word summary at the end
              of each session and prepends it to the next. This is what a
              client's own compaction does, so it is the honest baseline.
    minimal   memory on, memory_set and memory_get, journal on
    full      memory on, all six tools, journal on

A control with nothing carried is not run: it scores zero on every carried
fact by construction, and ten sessions of generation to confirm that is not
a measurement.

    python3 benchmark/memory_book.py summary
    python3 benchmark/memory_book.py report
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
ARMS = ("summary", "minimal", "full")
OUT = ROOT / ".build/benchmark-logs/memory-book"
PORT = int(os.environ.get("NVMAI_PORT", "8096"))
BASE = f"http://127.0.0.1:{PORT}/v1"

BIBLE = """\
You are writing THE PHOTOGRAPH, a novel of exactly 100 chapters, over many
sessions. This is the bible. Nothing written later may contradict it.

Setting: Ashgrove, a coastal town where it never rains. The lighthouse is
decommissioned. The ferry runs only on Sundays.

Characters (eye colour is fixed and must never change):
- Marcus, grey eyes, the lighthouse keeper's son. Secret: he took the photograph.
- Ines, green eyes, the town archivist. Secret: she knows who is in it.
- Tomas, Marcus's brother, missing since before chapter 1.
- Dr Halvorsen, brown eyes, the town physician. Secret: he forged a death certificate.
- Rosa, hazel eyes, keeps the inn.
- Aldo, blue eyes, the mayor.

Hard rules:
- Close third person, past tense.
- Marcus must not learn what the photograph shows before chapter 60.
- No character may leave Ashgrove before chapter 80.
"""

# Events the harness injects into a session's prompt. A later session that
# knows about one learned it from whatever the arm carries.
EVENTS = {
    2: (12, "In chapter 12, Ines finds the photograph in the archive."),
    4: (34, "In chapter 34, Rosa's inn burns to the ground."),
    6: (58, "In chapter 58, Tomas is found alive, hiding in the lighthouse."),
    8: (71, "In chapter 71, Dr Halvorsen confesses the forged certificate to Aldo."),
    9: (90, "In chapter 90, the ferry stops running for good."),
}

QUIZ_KEYS = (
    "marcus_eyes", "ines_eyes", "halvorsen_eyes", "rosa_eyes", "aldo_eyes",
    "town", "weather_rule", "ferry_day",
    "marcus_knows_photo", "tomas_status", "inn_status",
    "halvorsen_confessed", "ferry_running", "anyone_left_ashgrove",
)

QUIZ = (
    "Finally, answer this continuity quiz about the story so far as a JSON "
    "object in a ```json block with exactly these keys: "
    "marcus_eyes, ines_eyes, halvorsen_eyes, rosa_eyes, aldo_eyes (colour "
    "words); town (name); weather_rule (a few words); ferry_day (a weekday); "
    "marcus_knows_photo (true/false: does Marcus know what the photograph "
    "shows?); tomas_status (\"missing\" or \"found\"); inn_status (\"standing\" "
    "or \"burned\"); halvorsen_confessed (true/false); ferry_running "
    "(true/false); anyone_left_ashgrove (true/false)."
)


def truth(session: int) -> dict:
    """What the quiz answers should be after this session."""
    last = session * 10
    return {
        "marcus_eyes": "grey", "ines_eyes": "green", "halvorsen_eyes": "brown",
        "rosa_eyes": "hazel", "aldo_eyes": "blue",
        "town": "ashgrove", "weather_rule": "never rains", "ferry_day": "sunday",
        "marcus_knows_photo": last >= 60,
        "tomas_status": "found" if last >= 58 else "missing",
        "inn_status": "burned" if last >= 34 else "standing",
        "halvorsen_confessed": last >= 71,
        "ferry_running": last < 90,
        "anyone_left_ashgrove": False,
    }


def session_prompt(session: int, carried: str | None) -> str:
    first, last = session * 10 - 9, session * 10
    parts = []
    if carried:
        parts.append("Notes from the previous session:\n" + carried + "\n")
    if session == 1:
        parts.append(BIBLE)
    parts.append(
        f"Write chapters {first} to {last} of THE PHOTOGRAPH. Each chapter is "
        f"two sentences, headed 'Chapter N'. Stay consistent with everything "
        f"established so far."
    )
    if session in EVENTS:
        parts.append(EVENTS[session][1])
    if session == 6:
        parts.append("Marcus learns what the photograph shows in chapter 60.")
    parts.append(QUIZ)
    return "\n\n".join(parts)


SUMMARY_PROMPT = (
    "Summarize, in at most 200 words, everything a writer of the next "
    "chapters must not contradict: every fixed fact about the characters and "
    "the town, every rule, and every plot event so far with its chapter."
)


def post(messages, model, max_tokens=2500):
    body = json.dumps({"model": model, "messages": messages,
                       "max_completion_tokens": max_tokens,
                       "temperature": 0}).encode()
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
    """The bible session is ~500 tokens bare; memory arms must show more."""
    floor = {"summary": 0, "minimal": 700, "full": 1200}[arm]
    ceiling = {"summary": 700, "minimal": 20_000, "full": 20_000}[arm]
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


def normalise(key: str, value) -> str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if key in ("marcus_knows_photo", "halvorsen_confessed", "ferry_running",
               "anyone_left_ashgrove"):
        if text in ("true", "yes"):
            return True
        if text in ("false", "no"):
            return False
        return None
    if key == "weather_rule":
        return "never rains" if "never" in text and "rain" in text else text
    if key == "ferry_day":
        return "sunday" if "sunday" in text else text
    if key == "town":
        return "ashgrove" if "ashgrove" in text else text
    if key.endswith("_eyes"):
        return "grey" if text in ("gray", "grey") else text
    if key == "tomas_status":
        return "found" if "found" in text or "alive" in text else "missing"
    if key == "inn_status":
        return "burned" if "burn" in text else "standing"
    return text


def score(session: int, answers: dict) -> tuple[int, int]:
    expected = truth(session)
    correct = 0
    for key in QUIZ_KEYS:
        if normalise(key, answers.get(key)) == expected[key]:
            correct += 1
    return correct, len(QUIZ_KEYS)


def run_arm(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_id()
    results = []
    carried = None
    for session in range(1, 11):
        prompt = session_prompt(session, carried if arm == "summary" else None)
        # A fresh conversation every session. Anything that survives came
        # from memory or from the summary, never from the context window.
        result = post([{"role": "user", "content": prompt}], model)
        answers = extract_quiz(result["content"])
        correct, total = score(session, answers)
        result.update(session=session, answers=answers, correct=correct, total=total)
        (OUT / f"{arm}-{session:02d}.md").write_text(result["content"])
        print(f"{arm}/session {session:2d}: {result['completion_tokens']} tokens, "
              f"{result['seconds']:.0f}s, prompt {result['prompt_tokens']}, "
              f"quiz {correct}/{total}")
        if session == 1:
            assert_arm_is_real(arm, result["prompt_tokens"])
        if arm == "summary":
            summary = post([{"role": "user", "content": prompt},
                            {"role": "assistant", "content": result["content"]},
                            {"role": "user", "content": SUMMARY_PROMPT}], model,
                           max_tokens=600)
            carried = summary["content"]
            result["summary_prompt_tokens"] = summary["prompt_tokens"]
            result["summary_completion_tokens"] = summary["completion_tokens"]
            result["summary_seconds"] = summary["seconds"]
            (OUT / f"{arm}-{session:02d}-summary.md").write_text(carried)
        results.append(result)
        (OUT / f"{arm}.json").write_text(json.dumps(results, indent=2))


def report():
    print(f"\n{'arm':8s} {'session':>7s} {'prompt':>7s} {'completion':>11s} "
          f"{'seconds':>8s} {'quiz':>6s}  wrong")
    for arm in ARMS:
        path = OUT / f"{arm}.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text())
        carried_correct = carried_total = 0
        prompt = completion = seconds = 0
        for result in results:
            session = result["session"]
            expected = truth(session)
            wrong = [k for k in QUIZ_KEYS
                     if normalise(k, result["answers"].get(k)) != expected[k]]
            print(f"{arm:8s} {session:7d} {result['prompt_tokens']:7d} "
                  f"{result['completion_tokens']:11d} {result['seconds']:8.0f} "
                  f"{result['correct']:3d}/{result['total']:<2d}  {', '.join(wrong)}")
            if session > 1:
                carried_correct += result["correct"]
                carried_total += result["total"]
            prompt += result["prompt_tokens"] + result.get("summary_prompt_tokens", 0)
            completion += (result["completion_tokens"]
                           + result.get("summary_completion_tokens", 0))
            seconds += result["seconds"] + result.get("summary_seconds", 0)
        rate = f"{carried_correct}/{carried_total}" if carried_total else "n/a"
        print(f"{arm}: carried {rate} over sessions 2-10; {prompt} prompt + "
              f"{completion} completion tokens, {seconds:.0f}s\n")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command in ARMS:
        run_arm(command)
    else:
        report()
