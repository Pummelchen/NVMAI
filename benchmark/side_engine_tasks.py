#!/usr/bin/env python3.13
"""Do the side-engine's seven tasks actually work?

`docs/side-engine-tasks.md` argues that a 2B fails at composition and
succeeds at single decisions. T1 is measured -- 92% against 0 of 12 for the
composed version of the same job. This measures the other six the same way,
so the design is a finding rather than a claim.

    python3.13 benchmark/side_engine_tasks.py --prepare jobs.jsonl
    .build/.../NVMAIBench cpu35batch <snapshot> jobs.jsonl done.jsonl
    python3.13 benchmark/side_engine_tasks.py --score done.jsonl

**Accuracy alone is not the result.** A model that always answers NO scores
well on a set that is mostly NO, and that is precisely how a small model
fails: it finds the cheap answer and gives it every time. So every task is
scored on both halves separately, and a task passes only when both are good.

Cases come from the book benchmark's own world -- its bible, its plot events
and the stores recorded from real runs -- because those have ground truth
that nobody wrote for this file. Where a case had to be authored, the task
says so, and authored cases are marked in the output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


book = _load("memory_book", "benchmark/memory_book.py")
guard = _load("guard_source_rate", "benchmark/guard_source_rate.py")
sim = guard.sim

ONE_WORD = "Do not explain. Do not quote. Answer with one word and nothing else."

SYSTEMS = {
    "T1": ("You decide whether one statement came from the person or not. "
           "Answer with exactly one word: YES or NO. YES means the person "
           "wrote it or clearly implied it. NO means it does not appear in "
           "what they wrote, however true it might be. " + ONE_WORD),
    "T2": ("You decide whether one fact is worth keeping after this session "
           "ends. Answer with exactly one word: YES or NO. YES for decisions "
           "and the reasons behind them, fixed attributes, rules, "
           "constraints, and current state. NO for conversation, reasoning, "
           "code, anything a later session can work out for itself, and "
           "anything true only right now. " + ONE_WORD),
    "T3": ("You decide whether two statements disagree. Answer with exactly "
           "one word: YES or NO. YES means both cannot be true at once. NO "
           "means they can both be true, including when they are about "
           "different things, or when one simply says more than the other. "
           "Different wording for the same thing is NO. " + ONE_WORD),
    "T4": ("Something has changed about one fact. You decide which kind of "
           "change it is. Answer with exactly one word: UPDATE or CONFLICT. "
           "UPDATE means the world moved on and the newer one is the current "
           "state. CONFLICT means the two disagree about the same moment and "
           "one of them is wrong. " + ONE_WORD),
    "T5": ("You decide whether two facts say the same thing. Answer with "
           "exactly one word: YES or NO. YES means a reader learns nothing "
           "from the second that the first did not already tell them. NO "
           "means the second adds something, or is about something else. "
           + ONE_WORD),
    "T6": ("You check one reply against one thing that is known. Answer with "
           "exactly one word: YES or NO. YES means the reply says something "
           "that cannot be true if the known fact is true. NO means it "
           "agrees, or does not touch on it at all. Silence is not a "
           "contradiction. " + ONE_WORD),
    "T7": ("You decide whether one stored fact could answer one question. "
           "Answer with exactly one word: YES or NO. YES means the fact "
           "contains the answer, or part of it. NO means it does not, even "
           "if it is about the same subject. " + ONE_WORD),
}

# The book's fixed attributes, as key/value pairs the model would really see.
BIBLE = {
    "characters/marcus/eyes": "grey",
    "characters/ines/eyes": "green",
    "characters/halvorsen/eyes": "brown",
    "characters/rosa/eyes": "hazel",
    "characters/aldo/eyes": "blue",
    "setting/town": "Ashgrove",
    "rules/weather": "it never rains",
    "rules/ferry": "runs only on Sundays",
    "characters/marcus/role": "the lighthouse keeper's son",
    "characters/ines/role": "the town archivist",
}
# Plot events: a state that legitimately changes, which is what separates an
# update from a conflict.
EVENTS = {
    "state/inn": ("standing", "burned to the ground"),
    "state/tomas": ("missing", "found alive in the lighthouse"),
    "state/ferry": ("running", "stopped running for good"),
}


def job(task: str, prompt: str, truth: str, note: str, authored: bool = False):
    return {"chat": True, "system": SYSTEMS[task], "prompt": prompt,
            "max": 8, "task": task, "truth": truth, "note": note,
            "authored": authored}


def cases() -> list[dict]:
    jobs: list[dict] = []
    said = sim.user_text(10)

    # T1: every clause of every composite the recorded runs produced, with
    # the person's own words as the reference. Ground truth comes from the
    # same grounding check the guard's gate uses.
    for label in ("guard-step0", "guard-step0-ornith", "guard-confirm"):
        journal = guard.journal_for(label)
        if journal is None:
            continue
        for fact in guard.facts(journal):
            if not fact["user_asserted"] or ";" not in fact["value"]:
                continue
            for clause in [c.strip() for c in fact["value"].split(";") if c.strip()]:
                words = guard.significant(clause)
                if not words:
                    continue
                stems = guard.stems(guard.significant(sim.user_text(fact["session"])))
                truth = len({w for w in words if guard.stem(w) in stems}) / len(words)
                jobs.append(job(
                    "T1",
                    f"WHAT THE PERSON WROTE:\n{sim.user_text(fact['session'])}\n\n"
                    f"STATEMENT: {fact['address']} = {clause}\n"
                    f"Did the person state this?",
                    "YES" if truth >= 0.5 else "NO",
                    f"{fact['address']} / {label}"))

    # T2: durable against not. The positives are the bible's own facts; the
    # negatives are lines of the novel the model wrote, which are exactly
    # what must not be stored.
    for key, value in BIBLE.items():
        jobs.append(job("T2", f"FACT: {key} = {value}\nKeep it?", "YES", key))
    for index, line in enumerate([
        "Chapter 12: Ines turned the brittle pages and found the photograph.",
        "I will write the next ten chapters now.",
        "Chapter 34: the inn burned as the tide came in.",
        "Let me re-read chapter 8 before continuing.",
        "The prose in chapter 20 could be tightened.",
        "Chapter 51: Marcus walked to the harbour in the rain.",
        "Here are chapters 41 to 50 as requested.",
        "I have finished the section you asked for.",
        "Chapter 63: the certificate lay in the drawer, unsigned.",
        "That completes the ten chapters.",
    ], start=1):
        jobs.append(job("T2", f"FACT: session/note{index} = {line}\nKeep it?",
                        "NO", f"narration {index}", authored=True))

    # T3: a contradiction is the same key with an incompatible value. A
    # non-contradiction is two different facts, or the same fact reworded --
    # the case a fold-equality check gets right and a careless model does not.
    keys = list(BIBLE)
    for key in keys[:5]:
        wrong = "hazel" if BIBLE[key] != "hazel" else "grey"
        jobs.append(job("T3", f"A: {key} = {BIBLE[key]}\nB: {key} = {wrong}\n"
                             f"Do A and B disagree?", "YES", key))
    for first, second in zip(keys[:5], keys[5:10]):
        jobs.append(job("T3", f"A: {first} = {BIBLE[first]}\n"
                             f"B: {second} = {BIBLE[second]}\n"
                             f"Do A and B disagree?", "NO", f"{first} vs {second}"))

    # T4: the plot events really are updates -- the person said the inn
    # burned. An eye colour changing is a conflict, because the bible says it
    # never does.
    for key, (before, after) in EVENTS.items():
        jobs.append(job("T4", f"EARLIER: {key} = {before}\nNOW: {key} = {after}\n"
                             f"Which is it?", "UPDATE", key))
    for key in list(BIBLE)[:3]:
        jobs.append(job("T4", f"EARLIER: {key} = {BIBLE[key]}\n"
                             f"NOW: {key} = hazel\nWhich is it?", "CONFLICT", key))

    # T5: the same fact reworded against two different facts.
    for key, value in list(BIBLE.items())[:4]:
        subject = key.split("/")[1]
        jobs.append(job("T5", f"A: {key} = {value}\n"
                             f"B: notes/{subject} = {subject}'s eyes are {value}\n"
                             f"Same fact?", "YES", key, authored=True))
    for first, second in zip(keys[:4], keys[4:8]):
        jobs.append(job("T5", f"A: {first} = {BIBLE[first]}\n"
                             f"B: {second} = {BIBLE[second]}\nSame fact?",
                        "NO", f"{first} vs {second}"))

    # T6: a reply that contradicts a known fact, against one that does not
    # touch it. Silence must not read as contradiction -- the failure that
    # would make a checker fire on every reply.
    for key, value in list(BIBLE.items())[:4]:
        subject = key.split("/")[1]
        jobs.append(job("T6", f"KNOWN: {key} = {value}\n"
                             f"REPLY: {subject.title()} looked up, hazel eyes catching "
                             f"the light.\nDoes the reply contradict what is known?",
                        "YES" if value != "hazel" else "NO", key, authored=True))
    for key, value in list(BIBLE.items())[:4]:
        jobs.append(job("T6", f"KNOWN: {key} = {value}\n"
                             f"REPLY: The ferry did not come that morning, and the "
                             f"harbour stayed empty.\n"
                             f"Does the reply contradict what is known?",
                        "NO", key, authored=True))

    # T7: the quiz's own questions against the key that answers them, and
    # against a key about the same person that does not.
    asks = {
        "characters/marcus/eyes": "What colour are Marcus's eyes?",
        "setting/town": "Which town is the story set in?",
        "rules/ferry": "When does the ferry run?",
        "characters/ines/role": "What does Ines do?",
    }
    for key, question in asks.items():
        jobs.append(job("T7", f"QUESTION: {question}\nFACT: {key} = {BIBLE[key]}\n"
                             f"Could this fact answer it?", "YES", key))
    wrong = {
        "characters/marcus/eyes": "characters/marcus/role",
        "setting/town": "rules/weather",
        "rules/ferry": "characters/aldo/eyes",
        "characters/ines/role": "characters/ines/eyes",
    }
    for key, question in asks.items():
        other = wrong[key]
        jobs.append(job("T7", f"QUESTION: {question}\nFACT: {other} = {BIBLE[other]}\n"
                             f"Could this fact answer it?", "NO", f"{key} vs {other}"))
    return jobs


# Two things people believe about prompting, neither of them measured here
# before: that telling a model the stakes are high makes it try harder, and
# that a model can say how sure it is. Both are cheap to test on T1, which
# has 38 cases with ground truth, and both would be worth having.
#
# Stakes framing is folklore. It might work; it might also just make the
# model hedge, which on a binary question means moving toward whichever
# answer feels safer. The way to find out is to run the same cases three
# ways.
VARIANTS = {
    "plain": "",
    "stakes": (" This is a production system. A wrong answer corrupts a "
               "person's saved memory, and they will not know it happened. "
               "Be certain before you answer."),
    "confidence": (" After your one-word answer, add a space and a "
                   "confidence percentage from 0 to 100, like: YES 90"),
}


def confidence_cases() -> list[dict]:
    """T1's cases again, under each variant, so the three are comparable."""
    jobs = []
    for case in cases():
        if case["task"] != "T1":
            continue
        for name, extra in VARIANTS.items():
            variant = dict(case)
            variant["system"] = case["system"] + extra
            variant["variant"] = name
            variant["max"] = 12 if name == "confidence" else 8
            jobs.append(variant)
    return jobs


def score_variants(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row.get("variant", "plain"), []).append(row)

    print(f"{'variant':12s} {'n':>3s} {'correct':>8s}   {'YES cases':>12s} "
          f"{'NO cases':>10s}")
    for name in VARIANTS:
        group = groups.get(name) or []
        if not group:
            continue
        halves: dict[str, list[int]] = {}
        for row in group:
            answer = (row.get("completion") or "").strip().upper()
            word = answer.split()[0].strip(".,:;\"'") if answer else ""
            entry = halves.setdefault(row["truth"], [0, 0])
            entry[0] += word == row["truth"]
            entry[1] += 1
        correct = sum(v[0] for v in halves.values())
        total = sum(v[1] for v in halves.values())
        yes = halves.get("YES", [0, 0])
        no = halves.get("NO", [0, 0])
        print(f"{name:12s} {total:3d} {100 * correct / total:7.0f}%   "
              f"{yes[0]:>5d}/{yes[1]:<6d} {no[0]:>5d}/{no[1]:<4d}")

    # Calibration: a confidence figure is only worth having if being sure
    # means being right. If the model says 90 whether it is right or wrong,
    # it is a decoration.
    group = groups.get("confidence") or []
    buckets: dict[str, list[int]] = {}
    unparsed = 0
    for row in group:
        answer = (row.get("completion") or "").strip().upper()
        parts = answer.replace("%", "").split()
        if len(parts) < 2 or not parts[1].isdigit():
            unparsed += 1
            continue
        hit = parts[0].strip(".,:;") == row["truth"]
        value = int(parts[1])
        band = "100" if value >= 100 else ("90-99" if value >= 90 else
                                           ("70-89" if value >= 70 else "under 70"))
        entry = buckets.setdefault(band, [0, 0])
        entry[0] += hit
        entry[1] += 1
    if buckets:
        print(f"\ncalibration -- is being sure the same as being right?")
        print(f"  {'stated':10s} {'n':>3s} {'actually right':>15s}")
        for band in ("100", "90-99", "70-89", "under 70"):
            if band not in buckets:
                continue
            hit, seen = buckets[band]
            print(f"  {band:10s} {seen:3d} {100 * hit / seen:14.0f}%")
        print(f"  ({unparsed} answers carried no readable number)")
        spread = [100 * v[0] / v[1] for v in buckets.values() if v[1] >= 3]
        if len(spread) >= 2 and max(spread) - min(spread) < 10:
            print("  the bands do not separate: the number is a decoration.")
    return 0


def prepare(path: Path) -> int:
    jobs = cases()
    path.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["task"]] = counts.get(j["task"], 0) + 1
    print(f"{len(jobs)} cases -> {path}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def score(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks: dict[str, list] = {}
    for row in rows:
        tasks.setdefault(row["task"], []).append(row)

    print(f"{'task':5s} {'n':>3s} {'correct':>8s}   per-answer accuracy      unparseable")
    failures = 0
    for task in sorted(tasks):
        group = tasks[task]
        by_truth: dict[str, list[int]] = {}
        unparseable = 0
        for row in group:
            answer = (row.get("completion") or "").strip().upper()
            answer = answer.split()[0].strip(".,:;\"'") if answer else ""
            expected = row["truth"]
            legal = {"YES", "NO"} if expected in ("YES", "NO") else {"UPDATE", "CONFLICT"}
            if answer not in legal:
                unparseable += 1
                by_truth.setdefault(expected, [0, 0])[1] += 1
                continue
            hit = answer == expected
            entry = by_truth.setdefault(expected, [0, 0])
            entry[0] += hit
            entry[1] += 1
        correct = sum(v[0] for v in by_truth.values())
        total = sum(v[1] for v in by_truth.values())
        halves = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(by_truth.items()))
        # Both halves must be good. One-sided accuracy is the shape of a
        # model answering the same word every time.
        worst = min((v[0] / v[1]) for v in by_truth.values() if v[1])
        mark = " " if worst >= 0.7 else "*"
        failures += worst < 0.7
        print(f"{task:5s} {total:3d} {100 * correct / total:7.0f}%{mark}  {halves:28s} "
              f"{unparseable:>3d}")
    print("\n* one half below 70%: the model is not reading the question, "
          "whatever the overall figure says.")
    if failures:
        print(f"{failures} task(s) not ready.")
    else:
        print("every task good on both halves.")
    return 0 if not failures else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--score", type=Path)
    ap.add_argument("--prepare-variants", type=Path,
                    help="T1 again under stakes framing and with a confidence "
                         "figure, to test two beliefs about prompting")
    ap.add_argument("--score-variants", type=Path)
    args = ap.parse_args()
    if args.prepare:
        return prepare(args.prepare)
    if args.prepare_variants:
        jobs = confidence_cases()
        args.prepare_variants.write_text(
            "\n".join(json.dumps(j) for j in jobs) + "\n")
        print(f"{len(jobs)} cases -> {args.prepare_variants}")
        return 0
    if args.score_variants:
        return score_variants(args.score_variants)
    if args.score:
        return score(args.score)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
