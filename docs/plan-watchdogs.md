# Plan: watchdogs

A generation that has gone wrong today costs the user the whole generation —
45 seconds on a 35B — and they find out at the end. Every failure below has
been produced by this project's own runs, and none of them needs a model to
detect. That is the point of keeping this separate from the memory work: it
is cheap, deterministic, and has no false-positive tail worse than a missed
catch, provided the thresholds are calibrated rather than guessed.

Companion plan: `plan-memory-guard-and-shadow.md`. Independent; shares only
the release.

## The failures, and where each was seen

| Watchdog | Observed in this project | Signal |
| --- | --- | --- |
| **Loop** | Qwen 0.8B: "Wait, I need to check the facts again" repeated until the 600-token budget died | an n-gram of the recent output repeats |
| **Stall** | orphaned servers during harness bring-up; a bound port that never produced a token | no token for N seconds while generating |
| **Stub** | Qwen2B book session 9 returned 151 tokens where ten chapters belonged (quiz 4/14); Ornith's C99 stage returned 40 tokens of empty `<tool_call>` | finished normally with almost no visible content |
| **Tool ping-pong** | previously fixed by hand: rounds exhausted, a 31-token preamble returned as the answer | the same tool called with the same arguments N times in the *incoming* messages (request validation, not the output stream — see B2) |

## Audit revisions (2026-09-08)

**B1 — the stall watchdog as first drafted would fire on every long
prompt.** Prefill emits no tokens, and this project has measured a 10k-token
prompt taking 652 s of prefill. A 90 s no-token threshold would stop it. The
clock must start at the **first emitted token**, and a generation that has
not yet produced one is covered by a separate, much longer prefill budget
derived from the prompt length, not by this watchdog. Without this fix the
watchdog is not merely useless, it is destructive.

**B2 — the ping-pong watchdog is in the wrong place.** It inspects the
*incoming* request's message history, because NVMAI returns tool calls to
the client and repeats appear in the next request, not in the output stream.
It belongs in request validation, beside the existing unresolved-tool-call
check, not in `publish`. Corrected below.

**B3 — the calibration corpus is thin where the risk is highest.** The 180
recorded replies are mostly prose plus three short programs. Loop detection
on real code — repeated boilerplate, similar switch cases, generated tables
— is the largest false-positive risk and the corpus barely covers it. Until
a corpus of real coding sessions exists, `loop` stays observation-only
regardless of what the calibration says.

**B4 — the finish-reason mapping tells the client something false.** A
watchdog stop is not `length`. Since neither protocol has an honest reason
and inventing one breaks clients, the mapping stays, but the *content* must
carry a plain sentence saying the server stopped the generation and why. The
content is where the truth can be told without breaking a client.

**B5 — the detectors must be O(1) per chunk.** A naive substring search over
growing output is quadratic and would tax exactly the long generations most
likely to need watching. Fixed rolling window, bounded work per chunk, and a
unit test that asserts the cost does not grow with output length.

**B6 — watchdogs must not police the engine's own generations.** Memory
consolidation is a server-internal call with deliberately repetitive
structure. It runs with watchdogs disabled; only client-facing generations
are watched.

## Principles

- **Observe before acting.** Every watchdog defaults to logging. Stopping a
  generation is opt-in per watchdog, because a false stop is worse than a
  slow failure.
- **Calibrate, do not guess.** We hold roughly 180 recorded replies from the
  book and coder runs across five models. Every threshold is chosen so the
  false-positive rate on that corpus is zero, and the rate is recorded in
  the test.
- **Never fail a completion.** A watchdog that throws is a watchdog that
  took the machine down. All of them degrade to logging.

## Design

One protocol, four implementations, one place they are driven.

```
protocol Watchdog {
    mutating func observe(_ chunk: String, at: ContinuousClock.Instant) -> WatchdogVerdict
    mutating func finish(visibleTokens: Int, reason: String) -> WatchdogVerdict
}
enum WatchdogVerdict { case fine, concern(String), stop(String) }
```

Driven from `publish(_:)` in `sources/NVMAIServer/Core/ServerInference.swift`,
which is already the single point every content chunk passes through and
already owns a `shouldStop` flag for the stop-string matcher. A `stop`
verdict sets the same flag and records a finish reason of `watchdog`.

`Stall` is the exception: it is a time check, not a content check, so it
runs on the existing heartbeat timer in the HTTP layer rather than on chunk
arrival, and reports through the same protocol.

### Thresholds, to be calibrated

| Watchdog | Proposed rule | Calibrate against |
| --- | --- | --- |
| Loop | a 40-character window repeats ≥ 4 times within the last 1,200 characters | 180 recorded replies, including code with legitimate boilerplate |
| Stall | no visible token for 90 s **after the first token** (B1) | measured decode rates: 6.7–20 tok/s, so 90 s is far outside normal |
| Stub | `finish_reason == stop`, no tool call, and < 24 visible tokens | recorded replies; session-9-style stubs are 151 tokens, so this rule deliberately does **not** catch them (see below) |
| Ping-pong | same tool name and identical arguments ≥ 3 times in the request's messages | recorded tool-arm conversations |

**On the stub watchdog, a deliberate limit.** The 151-token book reply was
wrong but not malformed — it was a short answer to a long request. Detecting
that needs to know what was asked, which is a judgment call and belongs to
the shadow, not here. This watchdog catches only the unambiguous case: a
reply that finished normally with essentially nothing in it. Trying to catch
more from here would be the first false-positive source.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_WATCHDOGS` | `0` | `1` enables observation and logging |
| `NVMAI_WATCHDOG_ACT` | `` | comma list of watchdogs allowed to stop a generation: `loop,stall,stub,pingpong` |
| `NVMAI_WATCHDOG_STALL_SECONDS` | `90` | stall threshold |
| `NVMAI_WATCHDOG_LOOP_REPEATS` | `4` | repeats before a loop is called |

Off by default. With `NVMAI_WATCHDOGS=1` they only log; a watchdog stops a
generation only when named in `NVMAI_WATCHDOG_ACT`. The startup line gains
`watchdogs=off|observe|act(...)`.

## What the user sees

A stopped generation returns what was produced, with `finish_reason:
"watchdog"` and a one-line reason in the server log. It is never silent: a
truncated answer with no explanation is worse than a loop the user can see.

For the OpenAI and Anthropic surfaces this needs a finish-reason mapping —
`length` on the OpenAI paths and `max_tokens` on the Anthropic one are the
honest approximations, since neither protocol has a "the server stopped
this" reason and inventing one breaks clients.

## Tests

**Unit — `tests/NVMAIServer/WatchdogTests.swift` (new)**

Synthetic streams, no model, fully deterministic:

- a stream that repeats a phrase four times trips `loop`; three times does not
- a stream of ordinary prose of the same length does not trip it
- a stream of legitimately repetitive code (a switch with 30 similar cases)
  does not trip it
- `stall` fires after the threshold and not before; a slow-but-progressing
  stream never fires it
- `stub` fires on a 5-token reply that finished normally, and does not fire
  on a 5-token reply that ended in a tool call
- `pingpong` fires on the same call three times, not on the same tool with
  different arguments
- every watchdog with `NVMAI_WATCHDOGS=0` returns `.fine` without inspecting
  anything

**Corpus calibration — `benchmark/watchdog_calibrate.py` (new)**

Runs the four detectors over every recorded reply in
`.build/benchmark-logs/memory-{book,value}-*` and reports the false-positive
count per watchdog. The test asserts zero. This is what turns the thresholds
from guesses into measurements, and it is re-runnable when a new model is
added.

**Integration — extend `tests/NVMAIServer/HTTPServerTests.swift`**

- a scripted backend that emits a looping stream, with `loop` in
  `NVMAI_WATCHDOG_ACT`, ends the response with `finish_reason` mapped
  correctly on all three API surfaces
- with `NVMAI_WATCHDOGS=0` the same stream completes untouched

## Gate before any watchdog may act by default

- zero false positives on the recorded corpus for every watchdog
- the observation-only mode run across one full book and coder install, with
  every trip reviewed by hand
- only then may a watchdog be added to the default `NVMAI_WATCHDOG_ACT`, and
  the first candidate is `stall`, which has no plausible false positive

## Order of work

0. Fix B1 and B2 in the design before writing code: the stall clock starts
   at the first token, and ping-pong lives in request validation.
1. The protocol, the four detectors, unit tests.
2. `watchdog_calibrate.py`; tune thresholds until the corpus is clean.
3. Wire into `publish` and the heartbeat, observation only.
4. Finish-reason mapping across the three API surfaces, with tests.
5. Acting mode, `stall` first.
