#!/usr/bin/env python3.13
"""How often does the extraction mislabel who said something?

The memory guard rests on one bit per fact: did the *person* assert this, or
did the model derive it? With the guard on, a fact labelled `user` can no
longer be silently superseded. That is protection when the label is right
and damage when it is wrong -- a fact the model invented and then had
protected is worse than today's behaviour, not merely different. So the
label's error rate has to be measured before the guard is switched on
anywhere, and this is that measurement.

It reads a recorded book run's journal, which holds every fact the engine
wrote together with the provenance author the label became, and scores each
fact claiming the person's authority
against what the person actually put in front of the model in that session:
the story bible plus the plot events delivered up to that point
(`memory_sim.user_text`). A fact claiming to come from the user whose
substance is nowhere in the user's own words is a mislabel.

    python3.13 benchmark/guard_source_rate.py                    # newest run
    python3.13 benchmark/guard_source_rate.py --label guard-step0
    python3.13 benchmark/guard_source_rate.py --show             # each fact

**This finds candidates; it does not decide.** Word overlap cannot judge a
fact whose claim lives in its key and whose value is a boolean --
`continuity/marcus_knows_photo = false` is the bible's hard rule and neither
word appears in what the person wrote. Scored on the value alone it reported
nine mislabels on a run that had none; scored on key and value together it
reported six across three runs, and hand review found all six correct. The
flags are worth reading and the count is not worth trusting.

So the gate is: run this, read every flag, and record the verdict. The list
is small -- two to nine per run -- which is what makes that practical.

The gate itself, from `docs/plan-memory-guard-and-shadow.md`: under 5%
mislabelled, and no invented fact labelled `user` at all.

The scoring is deliberately generous to the model. A fact counts as
grounded if its *distinctive* words -- the ones carrying the claim, not the
scaffolding -- appear in the user's text. Paraphrase passes; only a value
the user never said anywhere fails. That direction is the safe one: it
under-reports mislabels, so a rate this measures as low could be lower still
but never higher.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / ".build/benchmark-logs"

_spec = importlib.util.spec_from_file_location("memory_sim", ROOT / "benchmark/memory_sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

# Words that carry no claim. A fact grounded only in these is grounded in
# nothing, so they are removed before the overlap is taken.
SCAFFOLD = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had", "in",
    "on", "at", "of", "to", "and", "or", "but", "that", "this", "it", "its",
    "as", "by", "for", "with", "from", "not", "no", "be", "been", "being",
    "he", "she", "they", "his", "her", "their", "who", "which", "what",
    "when", "where", "chapter", "character", "story", "book", "novel",
    "user", "assistant", "model", "fact", "facts", "memory", "session",
}
WORD = re.compile(r"[a-z0-9]+")


def significant(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower())
            if w not in SCAFFOLD and len(w) > 2}


# Inflection, not paraphrase. The first two candidates this script produced
# were both of this shape and both wrong: the bible says "Rosa, hazel eyes"
# and the extraction wrote "hazels"; the user's session-4 event says "Rosa's
# inn burns to the ground" and the extraction wrote "burned". Both facts are
# the user's, and an exact-token comparison called them inventions.
#
# A blunt prefix would fix those and blur real differences with them, so this
# strips the English suffixes that carry no claim instead. It is deliberately
# generous in one direction only: a value that survives it has no
# morphological relative anywhere in what the person wrote, which is what an
# invention looks like. Everything it flags is printed, because "no invented
# fact labelled user" is a claim a person checks, not a number.
SUFFIXES = ("ing", "ed", "es", "s", "d")


def atomic(value: str) -> bool:
    """Whether a value looks like one fact rather than several.

    A mirror of `MemoryRecord.isAtomic`, because this measures what ships:
    the store refuses to let a composite value carry the person's authority,
    so a composite is not a candidate for mislabelling. If the two ever
    disagree, this number stops describing the guard.
    """
    if ";" in value:
        return False
    if len(value) > 120:
        return False
    return re.search(r"[.!?]\s+\S", value) is None


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def stems(words: set[str]) -> set[str]:
    return {stem(w) for w in words}


def newest_label() -> str | None:
    runs = sorted(LOGS.glob("memval-scratch-*/book-auto-r*"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    return runs[0].parent.name.removeprefix("memval-scratch-")


def journal_for(label: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(
        str(LOGS / f"memval-scratch-{label}/book-auto-r*/nvmai/*/*.ndjson"))]
    real = [p for p in paths if "_global" not in p.name and p.stat().st_size > 0]
    return max(real, key=lambda p: p.stat().st_size) if real else None


def facts(journal: Path) -> list[dict]:
    """Every fact the engine wrote, in order, with its provenance flag.

    Sessions are numbered by the order they first wrote, the order the
    harness ran them, so a fact can be scored against what the person had
    said by that point and not against the whole book.
    """
    written = []
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item = record.get("memory", {}).get("_0")
        if not item:
            continue
        provenance = item.get("provenance") or {}
        written.append({
            "session_id": provenance.get("sessionID", ""),
            "address": f"{item['namespace'].removeprefix('k.')}/{item['key']}",
            "value": str(item.get("value", "")),
            # The record's `isUserAsserted` is not stored as a field: the
            # store translates it into the provenance author, which is what
            # the guard reads back on the next write. So that is what has to
            # be scored -- reading the record field here would silently find
            # nothing and report a perfect run.
            "user_asserted": (provenance.get("author") == "user"),
        })
    order: list[str] = []
    for fact in written:
        if fact["session_id"] not in order:
            order.append(fact["session_id"])
    index = {session: number for number, session in enumerate(order, start=1)}
    for fact in written:
        fact["session"] = index[fact["session_id"]]
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fraction of a value's distinctive words that must "
                         "appear in the user's own text")
    args = ap.parse_args()

    label = args.label or newest_label()
    if label is None:
        print(f"no recorded book runs under {LOGS}")
        return 1
    journal = journal_for(label)
    if journal is None:
        print(f"no journal for label {label}")
        return 1

    written = facts(journal)
    if not written:
        print(f"{journal} holds no facts")
        return 1

    def score(fact: dict) -> float | None:
        """How much of a fact is traceable to the person's own words.

        The **address and the value together**, because a fact is both. The
        first version of this scored the value alone and reported nine
        mislabels on a run that had none: every one was a boolean whose key
        carries the claim -- `rules/marcus_must_not_learn_photo_before_
        chapter_60 = true` is the person's rule from the story bible, and
        "true" appears nowhere in what they wrote. Scoring half a fact
        measures nothing.
        """
        said = stems(significant(sim.user_text(fact["session"])))
        words = significant(fact["address"].replace("/", " ").replace("_", " "))
        words |= significant(fact["value"])
        if not words:
            return None
        return len({w for w in words if stem(w) in said}) / len(words)

    # Only the facts that would actually carry authority. A composite value
    # cannot have one source, so the store demotes it before the guard ever
    # sees it -- and scoring demoted facts would report a risk that is not
    # taken.
    labelled = [f for f in written if f["user_asserted"]]
    claimed = [f for f in labelled if atomic(f["value"])]
    demoted = len(labelled) - len(claimed)
    mislabelled, grounded = [], []
    for fact in claimed:
        overlap = score(fact)
        if overlap is None:
            continue
        fact["overlap"] = overlap
        (grounded if overlap >= args.threshold else mislabelled).append(fact)

    # The control. A scorer generous enough to ground anything would report a
    # perfect run whatever the model did, so the facts labelled *model* are
    # put through the same test: they are the model's own output, and most of
    # them should not be traceable to the person's words. If the two rates
    # are close, this measurement is not measuring anything.
    derived = [f for f in written if not f["user_asserted"]]
    derived_scores = [s for s in (score(f) for f in derived) if s is not None]
    derived_grounded = sum(1 for s in derived_scores if s >= args.threshold)

    print(f"{label}: {journal.name}\n")
    print(f"  facts written                {len(written):5d}")
    print(f"  labelled user                {len(labelled):5d}  "
          f"({len(labelled) / len(written):.0%})")
    print(f"  labelled model               {len(written) - len(labelled):5d}")
    print(f"  demoted, value not atomic    {demoted:5d}  "
          f"(a composite cannot have one source)")
    print(f"  carrying authority           {len(claimed):5d}")
    if claimed:
        rate = len(mislabelled) / len(claimed)
        print(f"  of those, mislabelled        {len(mislabelled):5d}  ({rate:.1%})")
        if derived_scores:
            print(f"\n  control: model-labelled facts that would also score as "
                  f"the user's: {derived_grounded}/{len(derived_scores)} "
                  f"({derived_grounded / len(derived_scores):.0%})")
            print("  (a rate near the user rate would mean the test does not "
                  "discriminate)")
        print(f"\n  {len(mislabelled)} candidate(s) below the grounding "
              f"threshold -- read them; the count is not the verdict")
    else:
        print("\n  nothing was labelled user; the guard would never fire, "
              "and the gate cannot be judged from this run")

    if mislabelled:
        print(f"\n{len(mislabelled)} facts claiming the user's authority "
              f"that the user never said:")
        for fact in mislabelled:
            print(f"  s{fact['session']:<2d} {fact['address']:38s} "
                  f"overlap={fact['overlap']:.0%}  {fact['value'][:70]}")
    if args.show:
        print(f"\n{len(grounded)} grounded:")
        for fact in grounded:
            print(f"  s{fact['session']:<2d} {fact['address']:38s} "
                  f"overlap={fact['overlap']:.0%}  {fact['value'][:70]}")
    return 0 if claimed and not mislabelled else 2


if __name__ == "__main__":
    raise SystemExit(main())
