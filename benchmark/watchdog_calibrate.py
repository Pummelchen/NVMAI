#!/usr/bin/env python3.13
"""What do the watchdogs say about replies that were fine?

A threshold picked by eye is a guess, and a guess that stops a good
generation costs the user the whole answer. This runs the detectors over
every reply this project has recorded -- the book and coder benchmark runs
across five models -- and counts how often each one would have fired. All of
those replies completed; every trip here is therefore a false positive, and
the target is zero.

    python3.13 benchmark/watchdog_calibrate.py
    python3.13 benchmark/watchdog_calibrate.py --repeats 5 --window 48
    python3.13 benchmark/watchdog_calibrate.py --show 3   # what tripped

The loop and stub detectors are ports of the Swift ones in
`sources/NVMAIServer/Core/Watchdogs/`. `--selftest` checks both against the
shared fixture the Swift unit tests use, so the port cannot drift silently.

Two watchdogs are not calibrated here, for reasons rather than by omission:

  * **stall** has no text to run on. Its threshold is set against measured
    decode rates -- 6.7 to 20 tokens a second on this machine, so a 90-second
    gap after the first token is four hundred times a normal inter-token
    interval -- and against the prefill it deliberately does not watch (B1).
  * **ping-pong** needs recorded tool-arm conversations with their message
    histories. Those are replayed from the journal files when present, and
    reported as unavailable when not.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / ".build/benchmark-logs"
FIXTURE = ROOT / "tests/NVMAIServer/Fixtures/watchdog-cases.json"

# Replies that are genuinely broken, hand-reviewed once and recorded here so
# a true catch is never counted as a false alarm. The corpus is otherwise
# assumed good: these runs were all scored, so anything else the detectors
# fire on is a mistake.
#
# `memory-value/auto-r2` is a C99 pong program in which the model emitted
# `SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);` back to back,
# dozens of times, in the middle of its render function. The v2 directory is
# the same run copied. This is the failure the loop watchdog exists for.
KNOWN_LOOPS = {
    "memory-value/auto-r2.json",
    "memory-value-v2/auto-r2.json",
}

BASE = 1_000_003
MASK = (1 << 64) - 1
MINIMUM_PERIOD = 8
MINIMUM_DISTINCT = 12


def loop_trip(text: str, window: int = 64, history: int = 1200,
              repeats: int = 6) -> dict | None:
    """A port of `LoopWatchdog`: a rolling hash over a fixed window, counting
    non-overlapping repeats inside a recent history, ignoring windows too
    uniform to be a phrase."""
    data = text.encode("utf-8")
    factor = pow(BASE, window, 1 << 64)
    ring = bytearray(window)
    counts = [0] * 256
    distinct = 0
    digest = 0
    seen: dict[int, list[int]] = {}
    for index, byte in enumerate(data):
        slot = index % window
        # A byte only leaves once the ring has been round once.
        displaces = index >= window
        leaving = ring[slot]
        ring[slot] = byte
        if counts[byte] == 0:
            distinct += 1
        counts[byte] += 1
        digest = (digest * BASE + byte) & MASK
        if displaces:
            digest = (digest - leaving * factor) & MASK
            counts[leaving] -= 1
            if counts[leaving] == 0:
                distinct -= 1
        position = index + 1
        if position < window:
            continue
        if distinct < MINIMUM_DISTINCT:
            continue
        entry = seen.get(digest)
        if entry is None or position - entry[0] > history:
            seen[digest] = [position, position, 1]
            continue
        if position - entry[1] < MINIMUM_PERIOD:
            continue
        entry[1] = position
        entry[2] += 1
        if entry[2] >= repeats:
            period = (entry[1] - entry[0]) // max(1, entry[2] - 1)
            return {"count": entry[2], "period": period, "at": position}
    return None


def stub_trip(text: str, finish_reason: str, request_bytes: int,
              threshold: int = 96, asked: int = 200) -> dict | None:
    return stub_trip_bytes(len(text.encode("utf-8")), finish_reason,
                           request_bytes, threshold, asked)


def stub_trip_bytes(visible: int, finish_reason: str, request_bytes: int,
                    threshold: int = 96, asked: int = 200) -> dict | None:
    """A port of `StubWatchdog`: something substantial was asked for, and the
    reply finished normally with nothing in it.

    The request side is not decoration. Without it the rule fires on "Say
    OK." answered with "OK." -- which it did, on the very first request of
    the first observation run, because the harness's readiness probe is
    exactly that shape. A short answer to a short question is an answer.
    """
    if finish_reason == "stop" and visible < threshold and request_bytes >= asked:
        return {"visible": visible, "asked": request_bytes}
    return None


def replies() -> list[dict]:
    """Every recorded reply: the book runs, the coder runs, and the
    per-session markdown each of them wrote alongside its results."""
    found: list[dict] = []
    for path in sorted(LOGS.glob("memory-*/*.json")):
        try:
            records = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            content = record.get("content")
            if not isinstance(content, str) or not content:
                continue
            found.append({
                "source": f"{path.parent.name}/{path.name}",
                "label": str(record.get("session") or record.get("task") or ""),
                "text": content,
                # These runs all completed and were scored; a truncated one
                # would have been thrown out of the benchmark, so a normal
                # stop is the right assumption for the stub rule.
                "finish": "stop",
            })
    return found


# The journal does not store a long reply verbatim; it stores a placeholder
# that states what it left out, e.g. "[29 lines, 2228 bytes of output
# omitted]". Measuring the placeholder instead of the reply makes every long
# answer look like a 79-byte stub, which is what the first attempt at this
# corpus did.
ELISION = re.compile(r"\[(\d+) lines, (\d+) bytes of \w+ omitted\]")


def reply_bytes(text: str) -> int:
    """The reply's real size, with the journal's elisions added back."""
    omitted = sum(int(m.group(2)) for m in ELISION.finditer(text))
    kept = len(ELISION.sub("", text).encode("utf-8"))
    return kept + omitted


def exchanges() -> list[dict]:
    """Real (request, reply) pairs, from the session journals.

    The reply corpus above has no prompts, so it cannot say anything about a
    rule that depends on what was asked. The journals record both, across
    every recorded run, which is what the stub rule has to be judged on.
    """
    found: list[dict] = []
    for path in sorted(LOGS.glob("memval-scratch-*/**/*.ndjson")):
        if "_global" in path.name:
            continue
        pending: str | None = None
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line).get("event", {}).get("_0")
            except json.JSONDecodeError:
                continue
            if not event:
                continue
            if event.get("kind") == "userPrompt":
                pending = ((event.get("payload") or {}).get("text") or {}).get("_0")
            elif event.get("kind") == "assistantResponse" and pending is not None:
                reply = ((event.get("payload") or {}).get("response") or {}).get("_0") or {}
                found.append({
                    "source": path.parent.parent.parent.name,
                    "request_bytes": len(pending.encode("utf-8")),
                    "text": reply.get("text") or reply.get("content") or "",
                    "elided": True,
                    "finish": reply.get("finishReason") or "stop",
                })
                pending = None
    return found


def tool_conversations() -> list[list[tuple[str, str]]]:
    """Recorded tool-arm histories, as (name, canonical arguments) pairs.

    The tool arms wrote their transcripts as journals rather than as request
    bodies, so this returns what is actually on disk and the report says so
    when that is nothing.
    """
    conversations = []
    for path in sorted(LOGS.glob("memory-*/**/*.ndjson")):
        calls: list[tuple[str, str]] = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for call in entry.get("toolCalls") or []:
                name = call.get("name") or call.get("function", {}).get("name")
                arguments = call.get("arguments")
                if name:
                    calls.append((name, json.dumps(arguments, sort_keys=True)))
        if calls:
            conversations.append(calls)
    return conversations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--history", type=int, default=1200)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--stub-bytes", type=int, default=96)
    ap.add_argument("--stub-asked", type=int, default=200,
                    help="bytes the last user message must reach before a "
                         "short reply counts as a stub")
    ap.add_argument("--pingpong-repeats", type=int, default=3)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many tripped replies in full")
    ap.add_argument("--selftest", action="store_true",
                    help="check this port against the Swift unit-test fixture")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    corpus = replies()
    if not corpus:
        print(f"no recorded replies under {LOGS}")
        return 1

    loop_hits, stub_hits, caught = [], [], []
    total_bytes = 0
    for reply in corpus:
        total_bytes += len(reply["text"].encode("utf-8"))
        trip = loop_trip(reply["text"], args.window, args.history, args.repeats)
        if trip:
            known = reply["source"] in KNOWN_LOOPS
            (caught if known else loop_hits).append((reply, trip))
        # The reply corpus carries no prompt, so the request side of the
        # stub rule cannot be applied to it. It is judged on `exchanges()`
        # below instead, and counted here only for the loop.

    print(f"{len(corpus)} recorded replies, {total_bytes / 1e6:.1f} MB, "
          f"from {len({r['source'].split('/')[0] for r in corpus})} runs\n")
    print(f"  window={args.window} history={args.history} repeats={args.repeats} "
          f"stub_bytes={args.stub_bytes}\n")
    print(f"  {'watchdog':10s} {'false positives':>16s} {'rate':>8s}")
    rate = len(loop_hits) / len(corpus)
    print(f"  {'loop':10s} {len(loop_hits):16d} {rate:7.1%}")

    pairs = exchanges()
    for pair in pairs:
        trip = stub_trip_bytes(reply_bytes(pair["text"]), pair["finish"],
                               pair["request_bytes"], args.stub_bytes,
                               args.stub_asked)
        if trip:
            stub_hits.append((pair, trip))
    if pairs:
        print(f"  {'stub':10s} {len(stub_hits):16d} "
              f"{len(stub_hits) / len(pairs):7.1%}   "
              f"over {len(pairs)} recorded exchanges")
    else:
        print(f"  {'stub':10s} {'no corpus':>16s}")

    conversations = tool_conversations()
    if conversations:
        pingpong = 0
        for calls in conversations:
            counts: dict[tuple[str, str], int] = {}
            for call in calls:
                counts[call] = counts.get(call, 0) + 1
            if counts and max(counts.values()) >= args.pingpong_repeats:
                pingpong += 1
        print(f"  {'pingpong':10s} {pingpong:16d} "
              f"{pingpong / len(conversations):7.1%}")
    else:
        print(f"  {'pingpong':10s} {'no corpus':>16s}")
    print(f"  {'stall':10s} {'not applicable':>16s}   "
          f"(no timings recorded; see the module docstring)")
    print(f"\n  {len(caught)} of {len(KNOWN_LOOPS)} known-bad replies caught")

    if loop_hits:
        print(f"\n{len(loop_hits)} loop trips:")
        for reply, trip in loop_hits[:20]:
            print(f"  {reply['source']} {reply['label']:>10s} "
                  f"count={trip['count']} period={trip['period']} at={trip['at']} "
                  f"of {len(reply['text'])}")
    if stub_hits:
        print(f"\n{len(stub_hits)} stub trips:")
        for reply, trip in stub_hits[:20]:
            print(f"  {reply['source']:34s} asked={trip['asked']}B "
                  f"visible={trip['visible']}B")

    for reply, trip in loop_hits[: args.show]:
        start = max(0, trip["at"] - 4 * trip["period"])
        print(f"\n--- {reply['source']} {reply['label']} ---")
        print(reply["text"][start : trip["at"] + 80])

    clean = not loop_hits and not stub_hits
    if clean:
        print("\nzero false positives on "
              f"{len(corpus)} replies; {len(caught)} true catches")
    else:
        print("\nNOT clean: raise a threshold, or keep the watchdog observing only")
    return 0 if clean else 2


def selftest() -> int:
    """The Swift detectors and this port must agree, or the calibration says
    nothing about what will actually ship."""
    if not FIXTURE.exists():
        print(f"missing fixture: {FIXTURE}")
        return 1
    cases = json.loads(FIXTURE.read_text())["cases"]
    failures = 0
    for case in cases:
        text = case["text"] * case.get("repeat", 1)
        tripped = loop_trip(text, repeats=case.get("repeats", 6)) is not None
        if tripped != case["loopTrips"]:
            failures += 1
            print(f"  MISMATCH {case['name']}: python={tripped} "
                  f"swift={case['loopTrips']}")
    print(f"{len(cases)} fixture cases, {failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
