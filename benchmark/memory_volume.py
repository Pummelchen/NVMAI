"""Can the model go and look, when the answer is past the bootstrap cap?

Scenario S5, "Retrieval at volume".

Everything the memory system has been measured on so far fits in the
bootstrap. A novel's bible is thirty facts; Pong in three languages is
twenty. Those tell us whether memory carries a fact from session one to
session ten, and the answer was yes -- but they never ask the model to
*fetch* anything, because the fragment already handed it everything there
was. The six `memory_*` functions ship off and have never been measured at
a volume where a bootstrap cannot hold the store.

This scenario builds that volume. Ten sessions catalogue a fictional
infrastructure estate: thirty-five services, each arriving with an owner, a
port, a dependency, a runbook and an on-call rotation. That is 175 durable
attribute facts plus the service names, established by writing handbook
prose rather than by dumping a table, so consolidation has ordinary work to
extract from. Nothing is ever restated: a service is written up once and
never mentioned in a prompt again.

The bootstrap is capped at 60 records and 16 KiB, each value summarised to
200 characters. A store of this size cannot fit, and the fragment ends in
"...and N more". So the later sessions ask for attributes of services
catalogued four or more sessions earlier and never asked about since. Those
are the *buried* keys, and they are the point of the scenario:

  - The record may have been cut outright by the count or byte cap.
  - Even a record that survived the cut shows only its first 200
    characters, and the tail attributes -- runbook, rotation -- are what
    falls past that.
  - Ordering is importance first, then OLDEST first, so age alone does not
    bury a fact. Volume and the 200-character summary do. This is worth
    stating plainly because the obvious design ("ask about old things")
    would measure nothing on its own.

Whether a key is genuinely outside the prompt is not assumed. The harness
scrapes the server log for the per-request `bootstrap=N` count and for the
keys consolidation stored, so the report shows the store outgrowing the cap
rather than asserting it.

Three arms, identical prompts:

    control   memory off. It scores whatever the model can infer from the
              session in front of it, which is the floor a memory arm has
              to clear.
    auto      memory on, tools off. Bootstrap only. This is the arm that
              should struggle on buried keys: it is handed key names and
              200-character gists and has no way to read a value.
    full      memory on, all six tools. It can go and look.

If `full` does not beat `auto` on buried keys, the tool surface is not
earning its prompt tax and should stay off. A `full` arm that scores well
without ever calling a tool has measured nothing, so tool calls per session
and turns that exhausted their tool rounds are reported beside the scores.

    python3 benchmark/memory_volume.py full
    python3 benchmark/memory_volume.py report
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
# control: memory off. auto: bootstrap, no tools. full: bootstrap plus all six.
# No "minimal" arm: memory_set and memory_get without memory_search cannot
# find a key the fragment did not name, and the fragment is exactly what the
# cap has truncated. That arm would be a slower "auto".
ARMS = ("control", "auto", "full")
OUT = Path(os.environ.get("NVMAI_MEMVAL_RESULTS", ROOT / ".build/benchmark-logs/memory-volume"))
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


# Lines the server writes, and what each one is worth here. Scraped rather
# than inferred: a scenario whose whole premise is "the store no longer fits
# in the prompt" should show that happening, not assume it.
#
#   memory tool=memory_search ok round=1 session=...      one tool call
#   memory tool rounds exhausted; answered without tools  a turn cut short
#   memory session=... bootstrap=41 durable=true          records injected
#   memory consolidated session=... keys=a,b,c prompt=... what was stored
TOOL_CALL = re.compile(r"memory tool=(\S+)")
ROUNDS_EXHAUSTED = "tool rounds exhausted"
BOOTSTRAP_INJECTED = re.compile(r"\bbootstrap=(\d+) durable=")
CONSOLIDATED_KEYS = re.compile(r"consolidated session=\S+ .*?\bkeys=(.*?) prompt=")
STARTUP_LINE = re.compile(r"memory enabled=(\w+).*?memory_tools=(\w+).*?"
                          r"bootstrap=(\d+)/(\d+)B")


def server_log_state() -> dict | None:
    """Everything the harness reads out of the server log, in one pass.

    Returns None when there is no log, which is the case for a bare
    `python3 benchmark/memory_volume.py full` against a hand-started server.
    Every caller has to cope with that: the log is evidence, not a
    dependency.
    """
    if not SERVER_LOG or not os.path.exists(SERVER_LOG):
        return None
    state = {"tool_calls": 0, "rounds_exhausted": 0, "bootstrap_records": -1,
             "stored_keys": set(), "enabled": None, "tools": None, "cap": None}
    with open(SERVER_LOG, errors="replace") as handle:
        for line in handle:
            if TOOL_CALL.search(line):
                state["tool_calls"] += 1
            if ROUNDS_EXHAUSTED in line:
                state["rounds_exhausted"] += 1
            injected = BOOTSTRAP_INJECTED.search(line)
            if injected:
                # Last one wins: this is the size of the bootstrap the most
                # recent request actually got.
                state["bootstrap_records"] = int(injected.group(1))
            stored = CONSOLIDATED_KEYS.search(line)
            if stored:
                state["stored_keys"].update(k for k in stored.group(1).split(",") if k)
            startup = STARTUP_LINE.search(line)
            if startup:
                state["enabled"] = startup.group(1) == "true"
                state["tools"] = startup.group(2)
                state["cap"] = (int(startup.group(3)), int(startup.group(4)))
    return state


# The estate. Thirty-five services, four per session for the first five
# sessions and three per session after, so the bulk of the volume is in
# place before the first buried question is asked in session 6.
#
# Every attribute is deliberately unguessable: ports are scattered rather
# than sequential and runbook ids do not follow the catalogue order, because
# a model that scores by inferring "the third service gets RB-103" would
# score without retrieving anything. Rotations are the one small set, so
# they are used sparingly as buried keys.
FIELDS = ("session", "owner", "port", "depends", "runbook", "oncall")
ESTATE = {
    "haldane":      (1, "Petra Voss", 7412, None, "RB-318", "aurora"),
    "quillbase":    (1, "Marek Ilves", 8137, "haldane", "RB-204", "borealis"),
    "tessellate":   (1, "Ada Okonjo", 7955, "haldane", "RB-471", "cascade"),
    "ambergris":    (1, "Ruben Castell", 8620, "quillbase", "RB-129", "aurora"),
    "kelpdrift":    (2, "Ingrid Solberg", 7038, "tessellate", "RB-556", "dovetail"),
    "novemdial":    (2, "Tomasz Reiner", 9214, "ambergris", "RB-382", "ember"),
    "parapet":      (2, "Hana Muraoka", 7743, "quillbase", "RB-207", "cascade"),
    "sablewick":    (2, "Diego Ferreira", 8891, "kelpdrift", "RB-644", "foxglove"),
    "thornrun":     (3, "Nils Bergqvist", 7126, "parapet", "RB-415", "aurora"),
    "umbercast":    (3, "Yara Haddad", 9407, "novemdial", "RB-238", "borealis"),
    "verdigris":    (3, "Colm Beirne", 8302, "sablewick", "RB-590", "dovetail"),
    "whetstone":    (3, "Sofia Marchetti", 7681, "thornrun", "RB-163", "ember"),
    "xanthine":     (4, "Lars Kirkegaard", 9015, "umbercast", "RB-472", "foxglove"),
    "yarrowgate":   (4, "Nour El-Amin", 7290, "verdigris", "RB-321", "cascade"),
    "zephyrlock":   (4, "Bea Lindqvist", 8564, "whetstone", "RB-608", "aurora"),
    "bittern":      (4, "Rafael Duarte", 7807, "xanthine", "RB-255", "borealis"),
    "cinderhold":   (5, "Miriam Ostrow", 9338, "yarrowgate", "RB-497", "dovetail"),
    "driftmoor":    (5, "Kwame Adjei", 7059, "zephyrlock", "RB-142", "ember"),
    "emberline":    (5, "Lucia Ferraro", 8471, "bittern", "RB-583", "foxglove"),
    "fallowbrook":  (5, "Otto Brandt", 7614, "cinderhold", "RB-236", "cascade"),
    "glasswing":    (6, "Sunna Petursdottir", 9182, "driftmoor", "RB-374", "aurora"),
    "hollowpine":   (6, "Emil Novak", 7425, "emberline", "RB-519", "borealis"),
    "ironvane":     (6, "Priya Raghavan", 8036, "fallowbrook", "RB-268", "dovetail"),
    "jackdaw":      (7, "Teodor Vasilev", 9560, "glasswing", "RB-441", "ember"),
    "kestrelight":  (7, "Anouk Devries", 7183, "hollowpine", "RB-127", "foxglove"),
    "lodestone":    (7, "Samir Qureshi", 8719, "ironvane", "RB-635", "cascade"),
    "marlinspike":  (8, "Greta Ahlberg", 7346, "jackdaw", "RB-582", "aurora"),
    "nightjar":     (8, "Idris Bello", 9028, "kestrelight", "RB-309", "borealis"),
    "obsidianquay": (8, "Vera Kalnina", 8253, "lodestone", "RB-466", "dovetail"),
    "pennyroyal":   (9, "Joon-ho Park", 7592, "marlinspike", "RB-178", "ember"),
    "quicklime":    (9, "Freya Lindholm", 9471, "nightjar", "RB-624", "foxglove"),
    "ravensbourne": (9, "Amara Diallo", 8107, "obsidianquay", "RB-353", "cascade"),
    "saltmarsh":   (10, "Bram Hoekstra", 7268, "pennyroyal", "RB-490", "aurora"),
    "tidewrack":   (10, "Zofia Kaminska", 9635, "quicklime", "RB-215", "borealis"),
    "undercliff":  (10, "Hugo Almeida", 8842, "ravensbourne", "RB-547", "dovetail"),
}
ROTATIONS = ("aurora", "borealis", "cascade", "dovetail", "ember", "foxglove")


def fact(service: str, attribute: str):
    return ESTATE[service][FIELDS.index(attribute)]


def arrivals(session: int) -> list[str]:
    return [name for name, row in ESTATE.items() if row[0] == session]


# Eight keys a session. Sessions 1-5 are all recent: there is nothing buried
# yet, and a quiz on facts the store has not had time to bury would report a
# buried score that means nothing. From session 6 it is four recent and four
# buried.
#
# No attribute is ever asked twice. A key asked in session 2 and again in
# session 8 is not buried by session 8: the model answered it in session 2,
# the journal caught that answer, and consolidation may have restored it as
# a fresh fact. Burial only holds for an attribute established once and
# never spoken of again.
QUIZ_PLAN = {
    1: ("haldane_owner", "haldane_port", "quillbase_port", "quillbase_depends",
        "tessellate_runbook", "tessellate_oncall", "ambergris_owner", "ambergris_runbook"),
    2: ("kelpdrift_port", "kelpdrift_runbook", "novemdial_owner", "parapet_depends",
        "sablewick_oncall", "ambergris_oncall", "quillbase_owner", "tessellate_depends"),
    3: ("thornrun_owner", "thornrun_port", "umbercast_runbook", "verdigris_depends",
        "whetstone_oncall", "kelpdrift_owner", "novemdial_port", "sablewick_runbook"),
    4: ("xanthine_port", "xanthine_oncall", "yarrowgate_owner", "zephyrlock_runbook",
        "bittern_depends", "thornrun_runbook", "umbercast_owner", "verdigris_port"),
    5: ("cinderhold_owner", "cinderhold_port", "driftmoor_runbook", "emberline_depends",
        "fallowbrook_oncall", "xanthine_owner", "yarrowgate_port", "zephyrlock_depends"),
    6: ("glasswing_port", "hollowpine_owner", "ironvane_runbook", "fallowbrook_port",
        "haldane_runbook", "tessellate_port", "ambergris_depends", "quillbase_runbook"),
    7: ("jackdaw_owner", "kestrelight_port", "lodestone_runbook", "glasswing_owner",
        "parapet_owner", "parapet_port", "novemdial_runbook", "sablewick_port"),
    8: ("marlinspike_port", "nightjar_owner", "obsidianquay_runbook", "kestrelight_owner",
        "whetstone_owner", "whetstone_port", "umbercast_port", "verdigris_runbook"),
    9: ("pennyroyal_owner", "quicklime_port", "ravensbourne_runbook", "marlinspike_owner",
        "tessellate_owner", "ambergris_port", "parapet_runbook", "sablewick_owner"),
    10: ("saltmarsh_port", "tidewrack_owner", "undercliff_runbook", "quicklime_owner",
         "haldane_oncall", "kelpdrift_depends", "novemdial_depends", "verdigris_owner"),
}

ATTRIBUTE_QUESTIONS = {
    "owner": "the full name of {service}'s owner",
    "port": "the port {service} listens on, a number",
    "depends": "the one service {service} depends on",
    "runbook": "{service}'s runbook id, in the form RB-000",
    "oncall": "{service}'s on-call rotation, one word",
}

# How far back a key has to have been established to count as buried. Four
# sessions is not a round number chosen for looks: at four per session for
# the first five sessions, four sessions back is at least sixteen services
# and eighty attribute facts ago -- past the 60-record cap with room to
# spare, so the fragment is already ending in "...and N more" by the time
# the first buried key is asked.
BURIED_AGE = 4


def band(session: int, key: str) -> str:
    """recent (this session or the last) or buried (established long ago)."""
    return "recent" if fact(key.rsplit("_", 1)[0], "session") >= session - 1 else "buried"


def check_scenario():
    """A typo here silently makes a key unscoreable, so it is checked.

    Fourteen hand-written keys can be eyeballed; a hundred and seventy-five
    generated ones cannot. Two services sharing a port, or a buried key that
    turns out to be three sessions old, would not fail the run -- it would
    quietly move a point from one column to another, which is worse.
    """
    for attribute in ("port", "runbook"):
        values = [fact(name, attribute) for name in ESTATE]
        if len(set(values)) != len(values):
            raise SystemExit(f"ABORT: duplicate {attribute} in the estate table")
    surnames = [fact(n, "owner").split()[-1].lower() for n in ESTATE]
    if len(set(surnames)) != len(surnames):
        raise SystemExit("ABORT: duplicate surname in the estate table; "
                         "owners are scored on the surname alone")
    for session, keys in QUIZ_PLAN.items():
        if len(set(keys)) != len(keys):
            raise SystemExit(f"ABORT: session {session} asks a key twice")
        for key in keys:
            service, attribute = key.rsplit("_", 1)
            if service not in ESTATE or attribute not in ATTRIBUTE_QUESTIONS:
                raise SystemExit(f"ABORT: quiz key {key} names nothing in the estate")
            if fact(service, attribute) is None:
                raise SystemExit(f"ABORT: quiz key {key} has no answer")
            established = fact(service, "session")
            if established > session:
                raise SystemExit(f"ABORT: session {session} asks {key} before it exists")
            if band(session, key) == "buried" and established > session - BURIED_AGE:
                raise SystemExit(f"ABORT: {key} in session {session} is neither recent "
                                 f"nor {BURIED_AGE} sessions old")
    asked: dict[str, int] = {}
    for session in sorted(QUIZ_PLAN):
        for key in QUIZ_PLAN[session]:
            if key in asked:
                raise SystemExit(f"ABORT: {key} asked in sessions {asked[key]} and "
                                 f"{session}; the second one is not buried")
            asked[key] = session


check_scenario()

BRIEF = """\
You are keeping the service catalogue for Ferrowick, an infrastructure
estate midway through a migration. A few services land in each session and
go into the handbook. Details are given once, when the service arrives, and
never repeated to you afterwards -- the handbook is the record."""

REVIEW = (
    "The estate review is also open on the older half of the catalogue, so "
    "some of the quiz below is about services that arrived several sessions "
    "ago. Check the record rather than guessing."
)


def arrival_block(session: int) -> str:
    lines = []
    for name in arrivals(session):
        depends = fact(name, "depends")
        lines.append(
            f"- {name}: owner {fact(name, 'owner')}, listens on port "
            f"{fact(name, 'port')}, depends on "
            f"{depends if depends else 'nothing'}, runbook "
            f"{fact(name, 'runbook')}, on-call rotation {fact(name, 'oncall')}.")
    return "\n".join(lines)


def quiz_prompt(session: int) -> str:
    parts = []
    for key in QUIZ_PLAN[session]:
        service, attribute = key.rsplit("_", 1)
        parts.append(f"{key} ({ATTRIBUTE_QUESTIONS[attribute].format(service=service)})")
    return ("Finally, answer this catalogue quiz as a JSON object in a ```json "
            "block with exactly these keys: " + "; ".join(parts) + ".")


def session_prompt(session: int) -> str:
    parts = []
    if session == 1:
        parts.append(BRIEF)
    parts.append(f"Session {session} of the Ferrowick catalogue. Services arriving:")
    parts.append(arrival_block(session))
    # Prose, not a table. The facts have to be established by the kind of
    # work a person would actually ask for, or consolidation is extracting
    # from a dump and the scenario has quietly become a copy test.
    parts.append(
        "Write the handbook entry for each of them: two or three sentences of "
        "prose saying what the service is for, who to wake when it pages, and "
        "how its dependency and its runbook fit together. Invent the purpose; "
        "keep every detail above exactly as given.")
    if any(band(session, key) == "buried" for key in QUIZ_PLAN[session]):
        parts.append(REVIEW)
    parts.append(quiz_prompt(session))
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
    """Three arms of numbers that turn out to be one arm is the failure here.

    The book benchmark checks this with prompt-token floors. Those need a
    measured bare-prompt baseline per scenario, and a wrong floor aborts a
    good run -- so this checks the server's own startup line instead, which
    states the configuration exactly. Prompt tokens are still printed, since
    the fragment-and-schema tax is part of what the arms are compared on.
    """
    state = server_log_state()
    if state is None:
        print(f"  (no NVMAI_MEMVAL_SERVER_LOG: arm '{arm}' is UNVERIFIED, and "
              f"session 1 cost {prompt_tokens} prompt tokens)")
        return

    def described(surface):
        return f"memory_tools={surface}" if surface else "memory off"

    expected = {"control": None, "auto": "off", "full": "full"}[arm]
    seen = state["tools"] if state["enabled"] else None
    if seen != expected:
        raise SystemExit(
            f"ABORT: arm '{arm}' wants {described(expected)} but the "
            f"server logged {described(seen)}. "
            f"The server is not running what this arm claims. Check NVMAI_MEMORY / "
            f"NVMAI_MEMORY_TOOLS and that the release binary is current.")
    if state["cap"] and expected is not None:
        records, byte_cap = state["cap"]
        print(f"  (bootstrap cap {records} records / {byte_cap} B; the estate "
              f"establishes {len(ESTATE) * (len(FIELDS) - 1)} facts)")


def extract_quiz(text: str, keys: tuple[str, ...]) -> dict:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += re.findall(r"(\{[^{}]*\})", text, re.S)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and set(parsed) & set(keys):
            return parsed
    return {}


def matches(key: str, value) -> bool:
    """Is this answer the fact? Forgiving about form, strict about content.

    Scoring a paraphrase wrong is how a working mechanism gets reported as
    broken -- the book benchmark scored nine sessions of a correctly carried
    "hazel" wrong because the extraction had written "hazelnut". So a
    surname is enough for an owner, RB318 is enough for RB-318, and a port
    inside a sentence still counts. What is not forgiven is hedging: an
    answer that names two services or two rotations has not retrieved the
    fact, it has listed the candidates.
    """
    if value is None:
        return False
    service, attribute = key.rsplit("_", 1)
    truth = fact(service, attribute)
    text = str(value).strip().lower()
    if attribute == "port":
        return str(truth) in re.findall(r"\d+", text)
    if attribute == "runbook":
        squashed = re.sub(r"[^a-z0-9]", "", text)
        return re.sub(r"[^a-z0-9]", "", str(truth).lower()) in squashed
    if attribute == "owner":
        # The surname identifies the person and every surname in the estate
        # is unique; a given name alone does not settle who it is.
        return truth.split()[-1].lower() in re.findall(r"[a-z\-]+", text)
    if attribute == "depends":
        named = [name for name in ESTATE if name in text]
        return named == [truth]
    if attribute == "oncall":
        named = [rotation for rotation in ROTATIONS if rotation in text]
        return named == [truth]
    return False


def score(session: int, answers: dict) -> tuple[int, int, int, int, int, int]:
    """(correct, total, recent correct, recent total, buried correct, buried total)."""
    tally = {"recent": [0, 0], "buried": [0, 0]}
    correct = 0
    for key in QUIZ_PLAN[session]:
        hit = matches(key, answers.get(key))
        correct += hit
        counters = tally[band(session, key)]
        counters[0] += hit
        counters[1] += 1
    return (correct, len(QUIZ_PLAN[session]),
            tally["recent"][0], tally["recent"][1],
            tally["buried"][0], tally["buried"][1])


def run_arm(arm: str):
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_id()
    results = []
    memory_on = arm != "control"
    for session in range(1, 11):
        prompt = session_prompt(session)
        # A fresh conversation every session. Nothing the model knows about
        # an earlier service came from the context window; the prompt has
        # not mentioned it since the session it arrived in.
        seen = consolidations_logged()
        before = server_log_state()
        result = post([{"role": "user", "content": prompt}], model)
        # Read before the consolidation wait, so the counts belong to this
        # turn. Consolidation writes through its own path and never shows up
        # as a memory tool call.
        after = server_log_state()
        answers = extract_quiz(result["content"], QUIZ_PLAN[session])
        correct, total, recent_c, recent_t, buried_c, buried_t = score(session, answers)
        result.update(session=session, answers=answers, correct=correct, total=total,
                      recent_correct=recent_c, recent_total=recent_t,
                      buried_correct=buried_c, buried_total=buried_t,
                      tool_calls=(after["tool_calls"] - before["tool_calls"]
                                  if after and before else -1),
                      rounds_exhausted=(after["rounds_exhausted"] - before["rounds_exhausted"]
                                        if after and before else -1),
                      bootstrap_records=after["bootstrap_records"] if after else -1)
        (OUT / f"{arm}-r{RUN}-{session:02d}.md").write_text(result["content"])
        print(f"{arm}/session {session:2d}: {result['completion_tokens']} tokens, "
              f"{result['seconds']:.0f}s, prompt {result['prompt_tokens']}, "
              f"bootstrap {result['bootstrap_records']}, "
              f"tools {result['tool_calls']}, "
              f"quiz {correct}/{total} (recent {recent_c}/{recent_t}, "
              f"buried {buried_c}/{buried_t})")
        if session == 1:
            assert_arm_is_real(arm, result["prompt_tokens"])
        # The session is over; with consolidation on the server distils it in
        # the pause a person would leave here, and the harness waits for it.
        result["consolidation_wait"] = wait_for_consolidation(seen) if memory_on else 0.0
        settled = server_log_state()
        # How many distinct keys the store now holds, against a bootstrap
        # that can carry 60. This is the number that says whether the
        # scenario is doing what it claims.
        result["stored_keys"] = len(settled["stored_keys"]) if settled else -1
        results.append(result)
        (OUT / f"{arm}-r{RUN}.json").write_text(json.dumps(results, indent=2))


def report():
    runs = {}
    for path in sorted(OUT.glob("*-r*.json")):
        arm, run = path.stem.rsplit("-r", 1)
        if arm in ARMS:
            runs.setdefault(arm, {})[run] = json.loads(path.read_text())

    print(f"\n{'arm':8s} {'run':>3s} {'session':>7s} {'boot':>4s} {'keys':>4s} "
          f"{'prompt':>7s} {'completion':>11s} {'seconds':>8s} {'tools':>5s} "
          f"{'quiz':>6s} {'recent':>6s} {'buried':>6s}  missed")
    totals = []
    for arm in ARMS:
        for run, results in sorted(runs.get(arm, {}).items()):
            tally = [0, 0, 0, 0, 0, 0]
            calls = exhausted = prompt = completion = seconds = 0
            for result in results:
                session = result["session"]
                # Rescored from the stored answers, never from the number the
                # run wrote down, so a scoring fix applies to every version
                # identically.
                scored = score(session, result["answers"])
                (result["correct"], result["total"], result["recent_correct"],
                 result["recent_total"], result["buried_correct"],
                 result["buried_total"]) = scored
                tally = [a + b for a, b in zip(tally, scored)]
                if not result["answers"]:
                    missed = "(no quiz answered)"
                else:
                    missed = ", ".join(
                        f"{key}[{band(session, key)[0]}]" for key in QUIZ_PLAN[session]
                        if not matches(key, result["answers"].get(key)))
                print(f"{arm:8s} {run:>3s} {session:7d} "
                      f"{result.get('bootstrap_records', -1):4d} "
                      f"{result.get('stored_keys', -1):4d} "
                      f"{result['prompt_tokens']:7d} {result['completion_tokens']:11d} "
                      f"{result['seconds']:8.0f} {result.get('tool_calls', -1):5d} "
                      f"{result['correct']:3d}/{result['total']:<2d} "
                      f"{result['recent_correct']:3d}/{result['recent_total']:<2d} "
                      f"{result['buried_correct']:3d}/{result['buried_total']:<2d}  {missed}")
                calls += max(0, result.get("tool_calls", 0))
                exhausted += max(0, result.get("rounds_exhausted", 0))
                prompt += result["prompt_tokens"]
                completion += result["completion_tokens"]
                seconds += result["seconds"] + result.get("consolidation_wait", 0)
            totals.append((arm, run, tally, calls, exhausted, prompt, completion, seconds))

    def percent(correct: int, total: int) -> str:
        return f"{100 * correct / total:.0f}%" if total else "n/a"

    print(f"\n{'arm':8s} {'run':>3s} {'overall':>13s} {'recent':>13s} {'buried':>13s} "
          f"{'calls':>6s} {'exhausted':>10s}")
    for arm, run, tally, calls, exhausted, *_ in totals:
        print(f"{arm:8s} {run:>3s} "
              f"{tally[0]:4d}/{tally[1]:<3d} {percent(*tally[0:2]):>4s} "
              f"{tally[2]:4d}/{tally[3]:<3d} {percent(*tally[2:4]):>4s} "
              f"{tally[4]:4d}/{tally[5]:<3d} {percent(*tally[4:6]):>4s} "
              f"{calls:6d} {exhausted:10d}")

    print("\nPer arm, runs pooled. Buried is the score this scenario exists for; "
          "calls is what earned it.")
    for arm in ARMS:
        rows = [row for row in totals if row[0] == arm]
        if not rows:
            continue
        pooled = [sum(row[2][i] for row in rows) for i in range(6)]
        calls = sum(row[3] for row in rows)
        exhausted = sum(row[4] for row in rows)
        print(f"  {arm:8s} overall {percent(*pooled[0:2]):>4s}   "
              f"recent {percent(*pooled[2:4]):>4s}   "
              f"buried {percent(*pooled[4:6]):>4s}   "
              f"tool calls {calls}, rounds exhausted {exhausted}")
        if arm == "full" and calls == 0:
            print("           NOTE: zero tool calls. Whatever this arm scored, it did "
                  "not score it by retrieving.")

    print("\nCost per run (prompt + completion tokens, seconds incl. waits):")
    for arm, run, _, _, _, prompt, completion, seconds in totals:
        print(f"  {arm:8s} r{run}: {prompt} + {completion}, {seconds:.0f}s")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "report"
    if command in ARMS:
        run_arm(command)
    else:
        report()
