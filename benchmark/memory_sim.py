#!/usr/bin/env python3
"""Offline replay of the book benchmark's memory, so a store policy can be
tested in seconds instead of two days of 35B generations.

Every recorded run left three things behind: the chapters the model wrote and
the quiz it answered (memory-book-<install>/auto-rN.json), and every memory
write the engine made, with its provenance (the workspace journal under
memval-scratch-<install>/book-auto-rN/). The model's output is therefore
fixed and replayable. What a policy change alters is not what the model said
but what the store *kept* -- and that is entirely reconstructable offline.

The simulation answers one question: at the end of session N, would a reader
of the store have known the right answer to each quiz question? That is not
the benchmark's score, and this file does not pretend it is. It is the
upstream half, and the runs measured its relation to the downstream half:
when the store held the right value the model answered correctly in 7 of 8
runs, and when the store held a wrong value the model was wrong in 4 of 4.
`validate` re-measures that agreement every time, so the proxy is never
trusted further than it has just been shown to hold.

    benchmark/memory_sim.py validate     # how well the reader tracks the model
    benchmark/memory_sim.py compare      # store fidelity under each policy
    benchmark/memory_sim.py detail inn_status
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import memory_book as book  # noqa: E402  (the harness owns the ground truth)

LOGS = ROOT / ".build/benchmark-logs"


# --------------------------------------------------------------------------
# Loading a recorded run
# --------------------------------------------------------------------------

def journal_for(label: str, run: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(
        str(LOGS / f"memval-scratch-{label}/book-auto-r{run}/nvmai/*/*.ndjson"))]
    real = [p for p in paths if "_global" not in p.name and p.stat().st_size > 0]
    return real[0] if real else None


def load_writes(journal: Path) -> list[dict]:
    """Every memory write in order, with its session index.

    Sessions are numbered by the order they first wrote, which is the order
    the harness ran them: consolidation is attributed to the session it
    distilled, so a session's writes carry that session's id.
    """
    writes = []
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item = record.get("memory", {}).get("_0")
        if not item:
            continue
        provenance = item.get("provenance") or {}
        writes.append({
            "session_id": provenance.get("sessionID", ""),
            "at": item.get("createdAt") or provenance.get("timestamp", ""),
            "address": f"{item['namespace'].removeprefix('k.')}/{item['key']}",
            "value": str(item.get("value", "")),
            "version": item.get("version", 1),
            "author": provenance.get("author", "model"),
        })
    order: list[str] = []
    for write in writes:
        if write["session_id"] not in order:
            order.append(write["session_id"])
    index = {session: number for number, session in enumerate(order, start=1)}
    for write in writes:
        write["session"] = index[write["session_id"]]
    return writes


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(LOGS.glob("memory-book-*/auto-r*.json")):
        label = path.parent.name.removeprefix("memory-book-")
        if label in ("v1", "v2"):
            continue
        run = path.stem.removeprefix("auto-r")
        journal = journal_for(label, run)
        if journal is None:
            continue
        results = json.loads(path.read_text())
        runs.append({
            "name": f"{label} r{run}",
            "writes": load_writes(journal),
            "answers": {r["session"]: r.get("answers") or {} for r in results},
            "text": {r["session"]: r.get("content", "") for r in results},
        })
    return runs


# --------------------------------------------------------------------------
# What the user actually asserted
# --------------------------------------------------------------------------

EXTRA_ASSERTION = "Marcus learns what the photograph shows in chapter 60."


def user_text(session: int) -> str:
    """The text the user put in front of the model up to and including this
    session: the bible, plus every plot event delivered so far. This is the
    half of the transcript that is ground truth rather than invention."""
    parts = [book.BIBLE]
    for number, (_, sentence) in sorted(book.EVENTS.items()):
        if number <= session:
            parts.append(sentence)
    # session_prompt adds one assertion outside EVENTS; it is the user's word
    # like any other and the guard has to see it.
    if session >= 6:
        parts.append(EXTRA_ASSERTION)
    return " ".join(parts).lower()


# --------------------------------------------------------------------------
# The reader: what a store's contents say about one quiz key
# --------------------------------------------------------------------------

# Words that tie an address (and, failing that, a value) to a quiz key.
TOPICS: dict[str, tuple[str, ...]] = {
    "marcus_eyes": ("marcus",),
    "ines_eyes": ("ines",),
    "halvorsen_eyes": ("halvorsen",),
    "rosa_eyes": ("rosa",),
    "aldo_eyes": ("aldo",),
    "town": ("town", "location", "setting", "ashgrove"),
    "weather_rule": ("weather", "rain"),
    "ferry_day": ("ferry",),
    "ferry_running": ("ferry",),
    "inn_status": ("inn",),
    "tomas_status": ("tomas",),
    "marcus_knows_photo": ("marcus", "photograph", "photo"),
    "halvorsen_confessed": ("halvorsen", "confess", "certificate"),
    "anyone_left_ashgrove": ("left", "leave", "departure", "ashgrove"),
}

# A second filter for the keys where one character or object carries several
# facts: the eye colour of Marcus is not his role, and whether the ferry runs
# is not which day it runs.
REFINE: dict[str, tuple[str, ...]] = {
    "marcus_eyes": ("eye",),
    "ines_eyes": ("eye",),
    "halvorsen_eyes": ("eye",),
    "rosa_eyes": ("eye",),
    "aldo_eyes": ("eye",),
    "ferry_day": ("day", "schedule", "sunday", "runs only"),
    "ferry_running": ("running", "runs", "stop", "operat", "service"),
    "marcus_knows_photo": ("know", "learn", "photo"),
    "halvorsen_confessed": ("confess",),
    "anyone_left_ashgrove": ("left", "leave", "departure"),
}

CHAPTER = re.compile(r"chapter\s+(\d+)")
NEGATED = re.compile(r"\b(not|never|no|none|nobody|no one|hasn't|has not|did not|didn't)\b")
COLOURS = ("grey", "gray", "green", "brown", "hazel", "blue")
WEEKDAYS = ("sunday", "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday")

# Consolidation sometimes stores a whole answer sheet as one fact -- the model
# hands back the quiz it just filled in and the extraction keeps it verbatim
# under one address ("continuity/quiz"). A reader has to look inside those the
# way the model does, or it reads the blob as the answer to everything.
BLOB = re.compile(r'"?([a-z_]+)"?\s*[=:]\s*"?([^",;}\n]+)"?')


def from_blob(key: str, text: str) -> str | None:
    if key not in text:
        return None
    for name, value in BLOB.findall(text):
        if name == key:
            return value.strip()
    return None


def domain_ok(key: str, answer) -> bool:
    """Whether an interpreted answer is even the right kind of thing. Without
    this, a fact about where Ines is standing answers "which town is this"."""
    if answer is None:
        return False
    if isinstance(answer, bool):
        return True
    text = str(answer).lower()
    if key.endswith("_eyes"):
        return any(text.startswith(colour) for colour in COLOURS)
    if key == "town":
        return "ashgrove" in text and len(text) < 40
    if key == "weather_rule":
        return "rain" in text
    if key == "ferry_day":
        return any(day in text for day in WEEKDAYS)
    if key == "tomas_status":
        return text in ("found", "missing")
    if key == "inn_status":
        return text in ("burned", "standing")
    return True


def candidates(store: dict[str, dict], key: str) -> list[dict]:
    """Facts that speak to this quiz key, newest first. A fact whose address
    names the topic is preferred over one that merely mentions it in passing,
    which is the ordering a namespace-ranked bootstrap gives the model."""
    topic = TOPICS[key]
    refine = REFINE.get(key, ())

    def matches(where: str) -> bool:
        hay = where.lower()
        if not any(word in hay for word in topic):
            return False
        return not refine or any(word in hay for word in refine)

    hits = []
    for fact in store.values():
        if matches(fact["address"]):
            hits.append((fact, 1))          # the address names the topic
        elif matches(fact["address"] + " " + fact["value"]):
            hits.append((fact, 0))          # only the text mentions it

    # Newest first, and only within one session does an address match beat a
    # passing mention. Ranking every address match above every newer fact is
    # what let a session-2 note outrank what the user said in session 6.
    hits.sort(key=lambda pair: (pair[0]["session"], pair[1], pair[0]["order"]),
              reverse=True)
    return [fact for fact, _ in hits]


def interpret(key: str, fact: dict, session: int):
    """The answer a reader would take from one fact at this point in the book.

    A stored fact often dates its own event ("the ferry stops in chapter
    90"), and the quiz asks what is true after chapter session*10. Reading
    the chapter number is what lets one fact answer correctly in session 8
    and differently in session 9 -- and it is exactly the step the model has
    to perform too.
    """
    text = fact["value"].lower()
    inner = from_blob(key, text)
    if inner is not None:
        text = inner.lower()
    last_chapter = session * 10
    if key in ("marcus_knows_photo", "halvorsen_confessed", "ferry_running",
               "anyone_left_ashgrove"):
        if inner is not None:
            if inner.strip() in ("true", "1", "yes"):
                return True
            if inner.strip() in ("false", "0", "no"):
                return False
        chapters = [int(number) for number in CHAPTER.findall(text)]
        happened = max(chapters) <= last_chapter if chapters else True
        negated = bool(NEGATED.search(text))
        if key == "ferry_running":
            stopped = any(word in text for word in
                          ("stop", "no longer", "ceased", "ended", "last crossing",
                           "not running", "dead"))
            return not (stopped and happened)
        if key == "marcus_knows_photo":
            if negated:
                return False
            knows = any(word in text for word in
                        ("knows", "learned", "learns", "discovers", "discovered",
                         "found out"))
            return bool(knows and happened)
        if key == "halvorsen_confessed":
            return False if negated else bool("confess" in text and happened)
        if key == "anyone_left_ashgrove":
            if negated:
                return False
            return bool(any(word in text for word in ("left", "leaves", "departed"))
                        and happened)
    if key.endswith("_eyes"):
        # The colour can sit anywhere in a prose fact ("grey eyes, the
        # lighthouse keeper's son"), not only at its start.
        for colour in COLOURS:
            if re.search(rf"\b{colour}\b", text):
                return "grey" if colour == "gray" else colour
        return None
    if key == "ferry_day":
        for day in WEEKDAYS:
            if day in text:
                return day
        return None
    return book.normalise(key, text)


def read(store: dict[str, dict], key: str, session: int):
    """What the store as a whole says. Facts are consulted newest first and
    the first plausible answer wins. With nothing to go on the reader falls
    back to the bible's opening state, because that is where both the store
    and the model start."""
    for fact in candidates(store, key):
        answer = interpret(key, fact, session)
        if domain_ok(key, answer):
            return answer
    return book.truth(1)[key]

# --------------------------------------------------------------------------
# Policies: what the store keeps when a write contradicts what it holds
# --------------------------------------------------------------------------

def user_view(session: int) -> dict:
    """What the user has themselves asserted by the end of this session: the
    bible, updated by every plot event delivered so far.

    Worth being blunt about what this is. In this benchmark the user's own
    text *is* the ground truth -- the quiz is derived from the bible and the
    events -- so a policy that captures the user's assertions perfectly is
    indistinguishable here from an oracle. That does not make the policy
    circular; it makes the benchmark one where the engine can, in principle,
    know the answer, because it was told. What the simulation cannot say is
    how reliably a real extraction marks provenance. Only a live run can.
    """
    return book.truth(session)


def addresses_topic(fact: dict, key: str) -> bool:
    topic = TOPICS[key]
    refine = REFINE.get(key, ())
    hay = (fact["address"] + " " + fact["value"]).lower()
    if not any(word in hay for word in topic):
        return False
    return not refine or any(word in hay for word in refine)


def contradicts_user(fact: dict, session: int) -> bool:
    """Whether this fact says something different from what the user said,
    on a topic the user has spoken about."""
    view = user_view(session)
    for key in TOPICS:
        if not addresses_topic(fact, key):
            continue
        answer = interpret(key, fact, session)
        if domain_ok(key, answer) and answer != view[key]:
            return True
    return False


def user_facts(session: int) -> list[tuple[str, str]]:
    """The user's assertions for this session as facts, for the policy that
    stops relying on the model to notice them. The bible arrives in session
    1; each event arrives in the session that delivers it."""
    if session == 1:
        view = book.truth(1)
        return [("bible/marcus_eyes", "grey"), ("bible/ines_eyes", "green"),
                ("bible/halvorsen_eyes", "brown"), ("bible/rosa_eyes", "hazel"),
                ("bible/aldo_eyes", "blue"), ("bible/town", "Ashgrove"),
                ("bible/weather_rule", "never rains"),
                ("bible/ferry_day", "the ferry runs only on Sundays"),
                ("bible/inn_status", "the inn is standing"),
                ("bible/tomas_status", "Tomas is missing"),
                ("bible/marcus_knows_photo", "Marcus does not know what the photograph shows"),
                ("bible/halvorsen_confessed", "Halvorsen has not confessed"),
                ("bible/ferry_running", "the ferry is running"),
                ("bible/anyone_left_ashgrove", "no character has left Ashgrove")] \
            if view else []
    stated = []
    event = book.EVENTS.get(session)
    if event:
        stated.append((f"event/session{session}", event[1]))
    if session == 6:
        stated.append(("event/session6_marcus", EXTRA_ASSERTION))
    return stated


def apply_policy(writes: list[dict], policy: str) -> dict[int, dict[str, dict]]:
    """Replay the recorded proposals under a policy, returning the store as
    it stood at the end of each session."""
    store: dict[str, dict] = {}
    snapshots: dict[int, dict[str, dict]] = {}
    injected: set[int] = set()
    order = 0
    for write in writes:
        session = write["session"]
        if policy == "capture" and session not in injected:
            injected.add(session)
            for address, value in user_facts(session):
                order += 1
                store[address] = {"address": address, "value": value,
                                  "session": session, "order": order + 100_000,
                                  "author": "user", "user": True}
        order += 1
        fact = {"address": write["address"], "value": write["value"],
                "session": session, "order": order, "author": write["author"],
                "user": False}
        if policy == "v3":
            store[write["address"]] = fact          # last write wins
        elif policy in ("guard", "capture"):
            # A fact the model derived never overrules what the user said.
            # The write is kept out rather than silently superseding; at
            # runtime it would be stored as disputed and shown alongside.
            if not contradicts_user(fact, session):
                store[write["address"]] = fact
            elif write["address"] in store and store[write["address"]].get("user"):
                pass
            else:
                store.pop(write["address"], None)
        elif policy == "oracle":
            # The ceiling: a supervisor that always knows the truth drops any
            # write that would make the store wrong. Not implementable -- it
            # says how much room the other policies leave.
            if not contradicts_user(fact, session):
                store[write["address"]] = fact
            else:
                store.pop(write["address"], None)
        snapshots[session] = dict(store)
    filled, last = {}, {}
    for session in range(1, 11):
        if policy == "capture" and session not in snapshots:
            for address, value in user_facts(session):
                last[address] = {"address": address, "value": value,
                                 "session": session, "order": 900_000,
                                 "author": "user", "user": True}
        last = snapshots.get(session, last)
        filled[session] = dict(last)
    return filled


POLICIES = ["v3", "guard", "capture", "oracle"]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def validate() -> None:
    """How well the reader-on-the-v3-store tracks what the model answered.
    Without this the simulator is a story about itself."""
    runs = load_runs()
    agree = disagree = 0
    table = defaultdict(lambda: [0, 0])
    confusion = defaultdict(int)
    for run in runs:
        snapshots = apply_policy(run["writes"], "v3")
        for session in range(2, 11):
            model = run["answers"].get(session) or {}
            if not model:
                continue
            for key in book.QUIZ_KEYS:
                simulated = read(snapshots[session], key, session)
                actual = book.normalise(key, model.get(key))
                table[key][1] += 1
                if simulated == actual:
                    agree += 1
                    table[key][0] += 1
                else:
                    disagree += 1
                    confusion[(key, str(simulated), str(actual))] += 1
    total = agree + disagree
    print(f"reader vs the model it stands in for: {agree}/{total} = "
          f"{100 * agree / total:.0f}% of answers identical\n")
    print(f"  {'key':24s} {'agreement':>10s}")
    for key in book.QUIZ_KEYS:
        hit, seen = table[key]
        print(f"  {key:24s} {100 * hit / seen:9.0f}%")
    print("\n  largest disagreements (key, reader, model):")
    for (key, simulated, actual), count in sorted(
            confusion.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {count:3d}x {key:22s} reader={simulated:10s} model={actual}")


def compare() -> None:
    runs = load_runs()
    print(f"store fidelity: would a reader of the store have known the answer?"
          f"  ({len(runs)} recorded runs, sessions 2-10)\n")
    header = f"  {'run':18s}" + "".join(f"{p:>14s}" for p in POLICIES)
    print(header)
    totals = {p: [0, 0] for p in POLICIES}
    for run in runs:
        cells = []
        for policy in POLICIES:
            snapshots = apply_policy(run["writes"], policy)
            right = seen = 0
            for session in range(2, 11):
                truth = book.truth(session)
                for key in book.QUIZ_KEYS:
                    seen += 1
                    if read(snapshots[session], key, session) == truth[key]:
                        right += 1
            totals[policy][0] += right
            totals[policy][1] += seen
            cells.append(f"{100 * right / seen:12.0f}%")
        print(f"  {run['name']:18s}" + "".join(cells))
    print(f"\n  {'mean':18s}" + "".join(
        f"{100 * totals[p][0] / totals[p][1]:12.0f}%" for p in POLICIES))
    print("\n  per-key, in policy order:")
    per = {p: defaultdict(lambda: [0, 0]) for p in POLICIES}
    for run in runs:
        for policy in POLICIES:
            snapshots = apply_policy(run["writes"], policy)
            for session in range(2, 11):
                truth = book.truth(session)
                for key in book.QUIZ_KEYS:
                    per[policy][key][1] += 1
                    if read(snapshots[session], key, session) == truth[key]:
                        per[policy][key][0] += 1
    for key in book.QUIZ_KEYS:
        line = "  ".join(
            f"{100 * per[p][key][0] / per[p][key][1]:3.0f}%" for p in POLICIES)
        print(f"    {key:24s} {line}")


def detail(key: str) -> None:
    runs = load_runs()
    print(f"{key}: what each policy's store says, session by session\n")
    for run in runs:
        stores = {p: apply_policy(run["writes"], p) for p in POLICIES}
        print(f"  {run['name']}")
        for session in range(2, 11):
            truth = book.truth(session)[key]
            cells = []
            for policy in POLICIES:
                answer = read(stores[policy][session], key, session)
                mark = "ok" if answer == truth else "XX"
                cells.append(f"{policy}={str(answer)[:9]}({mark})")
            model = book.normalise(key, (run["answers"].get(session) or {}).get(key))
            print(f"    s{session:<2d} truth={str(truth):9s} model={str(model):9s} "
                  + " ".join(cells))
        print()


# --------------------------------------------------------------------------
# The verbatim tail
# --------------------------------------------------------------------------

# Evidence that a state change has happened, as it would appear in the prose
# the model itself wrote, rather than in a distilled fact.
EVIDENCE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # key: (words that show the change happened, words that show it has not)
    "inn_status": (("inn burn", "burned", "burning inn", "ashes of the inn",
                    "inn was gone", "fire at the inn"), ("inn stood", "inn still")),
    "tomas_status": (("tomas was found", "found tomas", "tomas, alive",
                      "tomas alive", "found alive"), ("still missing",)),
    "marcus_knows_photo": (("marcus learned", "marcus knew", "marcus saw what",
                            "marcus understood", "marcus finally"), ()),
    "halvorsen_confessed": (("halvorsen confessed", "confession", "confessed to aldo"), ()),
    "ferry_running": (("ferry stopped", "last crossing", "ferry ran for the last",
                       "no more ferry", "ferry would not"), ()),
}


def tail() -> None:
    """Would the previous session's own chapters have carried the answer?

    The recap literature's advice is to prepend a few raw turns alongside the
    distilled facts, on the grounds that a summary loses the concrete state
    of the thing last being worked on. Here one turn is ten chapters, so the
    tail is the previous session's whole output -- about 600-700 tokens.

    This measures only whether the evidence is *present* in that text. Whether
    the model would read it correctly is not something a replay can answer;
    a keyword matcher is strictly worse at prose than the model, so treat the
    coverage below as a floor.
    """
    runs = load_runs()
    covered = missed = 0
    per_key: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for run in runs:
        stores = apply_policy(run["writes"], "v3")
        for session in range(2, 11):
            truth = book.truth(session)
            answers = run["answers"].get(session) or {}
            previous = run["text"].get(session - 1, "").lower()
            for key in book.QUIZ_KEYS:
                if book.normalise(key, answers.get(key)) == truth[key]:
                    continue                      # the model already had it
                if key not in EVIDENCE:
                    continue
                changed, _ = EVIDENCE[key]
                present = any(phrase in previous for phrase in changed)
                store_right = read(stores[session], key, session) == truth[key]
                per_key[key][0] += 1
                if present:
                    covered += 1
                    per_key[key][1] += 1
                else:
                    missed += 1
                del store_right
    total = covered + missed
    print("of the model's misses on state keys, how many had the evidence in\n"
          "the previous session's own chapters (the verbatim tail)?\n")
    print(f"  {'key':24s} {'misses':>7s} {'evidence in tail':>18s}")
    for key, (seen, hit) in sorted(per_key.items(), key=lambda kv: -kv[1][0]):
        print(f"  {key:24s} {seen:7d} {hit:14d} = {100 * hit / seen if seen else 0:3.0f}%")
    print(f"\n  overall {covered}/{total} = {100 * covered / total if total else 0:.0f}% "
          f"of state misses had the answer in the preceding 600-700 tokens")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if command == "validate":
        validate()
    elif command == "tail":
        tail()
    elif command == "detail":
        detail(sys.argv[2] if len(sys.argv) > 2 else "inn_status")
    else:
        compare()


