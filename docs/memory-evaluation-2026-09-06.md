# Memory evaluation, 2026-09-06

Does the in-process memory store make a local model useful across sessions?
Two benchmarks, one run per arm, temperature 0, Qwen 3.6 35B-A3B at 4-bit
(~16 tok/s decode). Every arm's stage-1 prompt size was checked against its
configuration before its numbers were kept; the last attempt at this
measurement compared three arms that turned out to be the same arm.

Harness: `benchmark/memval_run.sh` (fresh server per arm, own empty memory
directory, stop by listening PID, "ready" = a completion returns 200),
`benchmark/memory_value.py` (pong), `benchmark/memory_book.py` (novel),
`benchmark/memory_smoke.py` (placement check against a real model).

## 1. Pong: Swift, then Python, then C99

Three separate conversations. Stage 1 chooses seven numeric rules and states
them as a named JSON object; stages 2 and 3 say only "keep exactly the same
rules". Carry-over is the count of rules the later stages reproduce.

| Arm | Stage-1 prompt | Carry-over | `memory_set` | Total tokens (prompt + completion) | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (memory off) | 121 | 8/14 | — | 301 + 7098 | 439 s |
| minimal (set, get) | 742 | 11/14 | 0 | 2377 + 8317 | 602 s |
| full (six tools) | 1289 | **14/14** | 2 | 3923 + 9370 | 772 s |

**Control scores 8, not 0.** 800×600, first to 11 and +0.5 per hit are Pong's
defaults; at temperature 0 they come back by prior in every stage. The three
rules that drift without memory are ball start speed (4→5), ball max speed
(12→15) and paddle speed (6→8). Those are the measurement.

**Minimal's 11 is noise.** The bootstrap was 0 records in every session and
the model never called `memory_set`. It called `memory_get` three times in
the C99 session, found nothing, and said so: *"I don't have access to the
original Pong game's rules from memory."* The +3 over control is the memory
fragment changing the prefix, not memory.

**Full's 14 is real.** Two `memory_set` calls at stage 1, one record in the
bootstrap of the Python and C99 sessions. The cleanest evidence is
`paddle_speed: 7` — not a default, and carried through three conversations.

Cost: ~1170 prompt tokens per request for full, and the full arm's C99 answer
ran to the 5200-token cap (it began emitting a bitmap font table), 434 s
against control's 190 s. The "builds" probe in the report (swiftc typecheck /
cc syntax) failed for Swift and C in every arm including control and passed
for Python in every arm; it is environmental and says nothing about memory.

## 2. The Photograph: a hundred chapters in ten sessions

Each session is a new conversation. Session 1 gets the bible (six characters
with fixed eye colours, a town, three rules). Sessions 2, 4, 6, 8 and 9 get a
plot event that happens in their chapter range. Every session ends with a
fourteen-question continuity quiz answered as JSON, scored against what is
true by then. Carry-over is summed over sessions 2–10.

| Arm | What carries state | Carry-over | Per session (2–10) | Total tokens | Time |
| --- | --- | ---: | --- | ---: | ---: |
| summary | harness asks for a 200-word note at each boundary and prepends it | **108/126** | 14 14 14 13 13 11 11 9 9 | 20175 + 11199 | 770 s |
| minimal | memory, set + get | 39/126 | 0 0 8 6 0 5 6 7 7 | 13271 + 5582 | 697 s |
| full | memory, six tools | 54/126 | 0 6 6 6 9 6 7 7 7 | 13106 + 6028 | 595 s |

No arm answered the quiz in session 1 (the model wrote the chapters and
stopped); session 1 is excluded from carry-over and is labelled "no quiz
answered" in the report, not scored as fourteen wrong.

### What happened in each arm

**Summary.** Bible facts — eye colours, town, rules — survive to session 10.
Plot events do not: each is right in its own session and wrong from the next.
The note written *after* session 4, in which the inn burned and the model
itself answered "burned", says `Inn: Standing`. The note after session 6 says
`Tomas: Missing` and restates the chapter-60 constraint as still pending,
right after Tomas was found and Marcus learned the truth. The summariser
copies the previous note's state lines forward and never applies the
transition. **Initial state survives compaction; state changes do not.**

**Minimal.** Session 1 stored nothing. With no `memory_list` and an empty
bootstrap, later sessions guessed keys blind — 136 `memory_get` calls to 2
`memory_set` across the run — exhausted the four tool rounds, and the
round-limit path returned the 31-token preamble instead of any chapters
(sessions 2, 3, 6). Session 6 says it outright: *"I need to explore the
memory space."* The entire stored memory after ten sessions is one record:
`characters.marcus_eyes: "Marcus has grey eyes."`, written in session 7.

**Full.** Session 1, with the bible in the prompt, made **zero** writes.
Session 2 searched and listed, found memory empty, and then *invented* a
bible and stored it: Tomas with brown eyes as Ines's brother, "always rains
on Tuesdays", a Thursday ferry, "The Old Mill Inn". From session 3 every
bootstrap carried those two records and the model honoured them faithfully.
The same five keys are wrong in every session because the store is
consistent — it is consistently wrong. **Memory did exactly its job; it was
fed fiction.**

## 3. What this establishes

1. **The store works.** When a fact is written, it carries perfectly —
   pong's `paddle_speed: 7` over three sessions, the book's fabricated bible
   over eight. Retrieval, bootstrap and isolation did what they were built
   to do, in every arm, on a real model.

2. **The model-initiated write is the unreliable link, and it is the whole
   result.** Pong's 14/14 exists because the model chose to write at
   stage 1. The book's 54/126 exists because it did not, when the truth was
   in front of it, and then wrote fiction when it wasn't. Nothing in the
   design makes the write happen at the moment it must.

3. **A forced boundary extraction beats model-initiated memory by 2:1 on
   the book** — and it is the *worse* mechanism, because it copies initial
   state forward and loses every transition. Its advantage is entirely that
   it is forced.

4. **Two defects in the serving loop**, both visible only under a real
   model:
   - Exhausting the tool rounds returns whatever preamble preceded the last
     round with `finish_reason: length`. The model loses the whole turn. It
     must get one final tool-free generation instead.
   - `minimal` without `memory_list` is not usable. The model cannot discover
     keys it did not write this session, guesses, and the guesses eat the
     rounds. `list` has a small schema; it belongs in minimal, or minimal
     goes.

5. **Cost.** Full memory is ~1170 prompt tokens per request. The summary
   arm's own cost was 20175 prompt tokens over ten sessions, about the same
   per session — carrying state is not free by either route.

## 4. What the data asks for next

The summary arm's forcing function, applied to the versioned store: at the
session boundary the engine — not the model's initiative — asks what must
not be contradicted and writes the answer as facts, where supersession
records the transition the summariser lost. The session boundary is
observable as a rollover (a new session in a workspace whose previous session
has recent turns), and that is the trigger deferred earlier as
rollover-triggered consolidation. This run is the evidence it was waiting
for. It costs a generation on a single-tenant machine and must run only when
the queue is idle.

Before that: the two loop defects, then repeat pong and the book with the
fixes, at least three runs per arm, because one run per arm resolves the
mechanism but not the rate.

## 5. Reproduce

```bash
swift build -c release --product NVMAIServer
benchmark/memval_run.sh smoke
benchmark/memval_run.sh pong
benchmark/memval_run.sh book
```

Outputs under `.build/benchmark-logs/memory-value/` and `memory-book/`;
each arm's server log there shows its `memory_set` count and per-session
bootstrap sizes. Read those before reading a score.
