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

The gate, from `docs/plan-memory-guard-and-shadow.md`: under 5% mislabelled,
and no invented fact labelled `user` at all.

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

    claimed = [f for f in written if f["user_asserted"]]
    mislabelled, grounded = [], []
    for fact in claimed:
        said = sim.user_text(fact["session"])
        words = significant(fact["value"])
        if not words:
            continue
        overlap = len({w for w in words if w in said}) / len(words)
        fact["overlap"] = overlap
        (grounded if overlap >= args.threshold else mislabelled).append(fact)

    print(f"{label}: {journal.name}\n")
    print(f"  facts written                {len(written):5d}")
    print(f"  labelled user                {len(claimed):5d}  "
          f"({len(claimed) / len(written):.0%})")
    print(f"  labelled model               {len(written) - len(claimed):5d}")
    if claimed:
        rate = len(mislabelled) / len(claimed)
        print(f"  of those, mislabelled        {len(mislabelled):5d}  ({rate:.1%})")
        print(f"\n  gate: under 5% mislabelled -- "
              f"{'MET' if rate < 0.05 else 'NOT MET'}")
        print(f"  gate: no invented fact labelled user -- "
              f"{'MET' if not mislabelled else 'NOT MET'}")
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
