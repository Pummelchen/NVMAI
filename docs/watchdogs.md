# Watchdogs

A generation that has gone wrong costs the user the whole generation — up to
forty-five seconds on a 35B — and they find out at the end. The watchdogs
are four cheap, deterministic checks that notice the four ways this project
has actually seen a generation fail. None of them needs a model, a second
process, or a byte of extra memory.

They are **off by default**, and when switched on they **only log**. Stopping
a generation is a second, separate opt-in, per watchdog.

## The four

| Watchdog | What it looks for | Where it was seen |
| --- | --- | --- |
| `loop` | a 64-byte window of the output repeated six times inside the last 1,200 bytes | Qwen 0.8B repeating "Wait, I need to check the facts again" until its 600-token budget died; a C99 reply emitting the same `SDL_SetRenderDrawColor` line dozens of times |
| `stall` | no visible token for 90 s, counted from the **first** token | orphaned servers during harness bring-up: a bound port that never produced a token |
| `stub` | finished normally, no tool call, under 96 visible bytes | Ornith's C99 stage returning forty tokens of empty `<tool_call>` markup where a program belonged |
| `pingpong` | the same tool called with identical arguments three times in a row in the incoming message history | a tool loop that exhausted its rounds and returned a 31-token preamble as the answer |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_WATCHDOGS` | `0` | `1` enables observation and logging |
| `NVMAI_WATCHDOG_ACT` | empty | comma list of watchdogs allowed to intervene: `loop,stall,stub,pingpong` |
| `NVMAI_WATCHDOG_STALL_SECONDS` | `90` | stall threshold |
| `NVMAI_WATCHDOG_LOOP_REPEATS` | `6` | repeats before a loop is called |

The startup banner reports `watchdogs=off`, `watchdogs=observe` or
`watchdogs=act(...)`, so what is running is visible without reading the
environment.

## What the user sees

Nothing, unless a watchdog is allowed to act and something trips.

When one does, the generation returns what it had produced, with a plain
sentence appended saying the server stopped it and why, and `finish_reason`
mapped to `length`. Neither the OpenAI nor the Responses protocol has an
honest reason for "the server stopped this", and inventing one breaks
clients — so the reason is the nearest existing value and the truth is told
in the one place that cannot break anything, the content itself.

`pingpong` never intervenes, whatever the operator asks for. There is
nothing to stop — the loop is in the request that just arrived — and the
obvious intervention was tried and is unsafe: withholding the tools leaves
the prompt still rendered with the tool template, because the trip requires
three tool calls in the history, so the model goes on emitting tool calls
into a decoder that now allows none of them and the request fails outright.
That is worse than the loop. Naming `pingpong` in `NVMAI_WATCHDOG_ACT` is
dropped at parse time rather than honoured into a worse failure. It reports,
and the client, which owns the loop, decides.

A run is consecutive: a user turn or an assistant reply between two
identical calls resets it, because the conversation moved on. Tool results
do not, because they are the answer to the call being repeated.

## What they never do

- **Fail a request.** There is no throwing path. The worst a watchdog can do
  is end a generation early and say so.
- **Watch the engine's own generations.** Memory consolidation is a
  server-internal call with a deliberately repetitive prompt and a
  deliberately terse answer — exactly the shape `loop` and `stub` hunt — and
  no person is waiting on it. It runs unwatched.
- **Put generated text in the log.** A trip reports what repeated, how often
  and how far apart, never the text. What was repeated is in the reply the
  user already has.

## Calibration

Thresholds are measured, not guessed. `benchmark/watchdog_calibrate.py` runs
the detectors over every reply this project has recorded — 999 replies, 4.4
MB, sixteen benchmark runs across five models. All of those runs were scored
and accepted, so any trip is a false positive except the ones hand-reviewed
and listed in the script.

The plan's proposed 40-byte window at four repeats fired on **8.1%** of the
corpus, every one of them real code: repeated SDL calls, repeated struct
initialisers, tables of similar cases. At 64 bytes and six repeats the
corpus is clean, and the one genuinely broken reply in it is still caught:

```
999 recorded replies, 4.4 MB, from 16 runs
  window=64 history=1200 repeats=6 stub_bytes=96
  loop       0 false positives   stub  0 false positives
  2 of 2 known-bad replies caught
```

The clean region is wide rather than a knife-edge: every combination from a
56-byte window at five repeats upward gives the same result, so the defaults
sit in the middle of a plateau.

Three rules do that work, and each rejects a shape that repeats perfectly
without being a loop:

- repeats must be at least 8 bytes apart, so one run of something is not two
  occurrences of it;
- a window must hold at least 12 distinct bytes, which rejects table rules,
  horizontal lines and blocks of indentation while admitting prose and code;
- repeats age out after 1,200 bytes, so a phrase that recurs naturally
  across a long document never accumulates.

The calibration script carries a Python port of the loop detector. The port
and the Swift original are checked against a shared fixture
(`tests/NVMAIServer/Fixtures/watchdog-cases.json`) from both sides —
`watchdog_calibrate.py --selftest` and a unit test — so the corpus numbers
cannot quietly stop describing what ships.

Two watchdogs are not calibrated against text, for reasons rather than by
omission. `stall` has no text to run on: its threshold is set against
measured decode rates of 6.7 to 20 tokens a second, which makes 90 seconds
about four hundred times a normal gap. `pingpong` needs recorded tool-arm
message histories, and the recorded runs kept journals rather than request
bodies; the script reports that corpus as unavailable instead of inventing
a number.

## Known limitation

The loop detector cannot tell heavily templated output from a loop. A
numbered list where only the number changes repeats its invariant span
exactly, and no local rule can distinguish that from a model stuck in a
cycle. The detector says "something repeated", which is true; whether that
is a failure needs to know what was asked, and it does not.

That is why `loop` is not a candidate for the default acting list. It is
also why the calibration corpus matters more here than anywhere else, and
why the corpus is still thin: it is mostly prose plus a handful of short
programs, and real coding sessions are where the risk lives.

## Before any watchdog acts by default

1. Zero false positives on the recorded corpus. **Met**, for `loop` and
   `stub`.
2. Observation mode run across one full book and one full coder install,
   with every trip reviewed by hand. **Not done.**
3. Only then may a watchdog join the default acting list. The first
   candidate is `stall`, which has no plausible false positive; `loop` is
   the last, for the reason above.

## Testing

`tests/NVMAIServer/WatchdogTests.swift` — 39 tests over synthetic streams,
no model, fully deterministic. Roughly half are false-positive cases,
because that is the half that costs a user an answer.

The stall watchdog reports a hang but cannot stop one. The runner polls for
a stop between tokens, so a generation that has genuinely wedged — no token,
no poll — is reported here and ended nowhere. Killing a stuck GPU command
buffer is a process-level decision, not a watchdog's.
