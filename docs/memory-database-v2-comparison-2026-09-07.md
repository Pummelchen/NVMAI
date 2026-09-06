# Memory database v1 vs v2, 2026-09-07

Same benchmarks, same model (Qwen 3.6 35B-A3B 4-bit), same harness, three
runs per arm with the server's own sampling, both versions scored by the same
rescoring pass. v1 is tag `memory-v1` (`0175e5b`); v2 is tag `memory-v2`
(`7ad163c`). What v2 changed is listed in `docs/agent-memory.md` under
"Versions". The question this answers is the one that was asked: did anything
get worse, and did quality or time improve.

Arms: **summary** is memory off with a forced 200-word note carried by the
harness at each boundary (what a client's own compaction does); **auto** is
memory on with memory tools off, the engine writing by consolidation;
**minimal** and **full** add memory tools. Carry-over on the book is the
fourteen-question continuity quiz over sessions 2–10; on pong it is the seven
rules the later ports must reproduce.

## Carry-over

| Book, sessions 2–10 | v1 (3 runs) | v2 (3 runs) |
| --- | ---: | ---: |
| summary (unchanged code path) | 96% — 124, 120, 119 | 90% — 117, 108, 115 |
| **auto** | 96% — 118, 124, 121 | **95% — 123, 122, 115** |
| minimal | 83% — 95, 113, 107 | 89% — 101, 119, 118 |
| full | 83% — 100, 102, 110 | 92% — 119, 110, 119 |

| Pong | v1 | v2 |
| --- | ---: | ---: |
| control (no memory) | 24% | 52% |
| auto / minimal / full | 100% / 100% / 100% | 100% / 100% / 100% |

Read the book table with its noise floor in view: the summary arm runs
identical code in both versions and moved six points between them. That is
the run-to-run spread of this benchmark under sampling. Auto's one-point
change is inside it. Read *within* each run, v2's auto beat its concurrent
baseline by five points where v1's tied it — consistent with the ranked
bootstrap and the disputed marker doing something, and not strong enough to
claim on its own. Both tool arms improved by six to nine points, which is at
the edge of the noise but consistent across both. Nothing got worse.

Pong's control moving from 24% to 52% is the same noise on a smaller sample:
Pong's defaults come back by prior more or less often per run. Every memory
arm carried all seven rules in all six runs on both versions.

The scoring change: eye colours are matched by prefix. In v2 auto's second
run the extraction stored `characters/rosa/eyes = hazelnut`, the model
answered "hazelnut" in nine sessions, and the strict scorer marked every one
wrong against "hazel". Memory carried the fact perfectly; the quiz was strict
on a synonym. Both versions were rescored from their stored answers with the
same rule, so the fix cannot favour either.

## Cost, consolidation included

The harness only sees request tokens. Consolidation is a server-internal
generation, counted here from the server logs. Per run of the book, means of
three; model-busy time is request time plus consolidation time.

| Book | Request prompt | Consolidation prompt | Total prompt | Total completion | Model-busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary v1 → v2 | 19.9k → 20.7k | — | 19.9k → 20.7k | 11.2k → 11.7k | 747 → 803 s |
| **auto v1 → v2** | 8.8k → 9.4k | 16.7k → 19.2k | 25.6k → **28.6k** | 11.1k → **13.0k** | 1101 → **1336 s** |
| minimal v1 → v2 | 15.2k → 13.7k | 17.1k → 16.5k | 32.3k → 30.2k | 10.9k → 9.8k | 1251 → 1131 s |
| full v1 → v2 | 20.4k → 17.5k | 19.9k → 18.5k | 40.4k → 36.0k | 13.2k → 11.5k | 1791 → 1390 s |

**v2's auto arm costs about 15% more than v1's** for carry-over that is
equal within noise. The change meant to cut the extraction prompt — values
only for keys the session mentions — did cut the values, but the list of
every key by name grows with the store, and the extraction now writes more
facts per session (29–41 in session one against 23), so consolidation prompt
and completion both rose. The request path also grew slightly: the ranked,
dependency-aware bootstrap is a little larger than the flat one. The tool
arms went the other way, cheaper on every column, because the smaller
extraction prompt and the ranked bootstrap reduced their read-spam.

| Pong | Total prompt | Total completion | Model-busy |
| --- | ---: | ---: | ---: |
| control v1 → v2 | 301 → 301 | 7.2k → 8.3k | 459 → 479 s |
| auto v1 → v2 | 2.7k → 2.9k | 8.1k → 9.2k | 573 → 635 s |
| minimal v1 → v2 | 5.7k → 5.4k | 9.5k → 9.5k | 817 → 710 s |
| full v1 → v2 | 6.9k → 7.0k | 11.1k → 9.3k | 1004 → 889 s |

Pong is a wash. Its v2 timing excludes run 3: Tailscale's network extension
was at 94% CPU for that run and decode fell from ~17 to ~9–13 tokens per
second on every arm; the carry-over of that run is unaffected and is counted.

Two context notes on time. The benchmark's sessions are one turn long, so
consolidation is paid once per turn — the worst possible ratio; a real
session is ten to fifty turns for the same one consolidation. And a person
never waits for it: it runs in the pause after a reply, and a request that
arrives during one waits behind that one generation at most.

## What v2 established

1. **Nothing regressed in outcome.** Auto equal within noise on the book,
   perfect on pong in six of six runs; the tool arms up.
2. **The quality mechanisms worked as designed on the real model**: the
   bootstrap kept every fact in the window it needed (the one residual
   "miss" was a synonym), reversions were caught and flagged as disputed
   (`project/chapter_count`, `state/…`), and routing folded renamed
   namespaces without the false merges v1's first rule had.
3. **The cost target was missed.** The extraction prompt is still the whole
   overhead of the feature, and v2 made it slightly larger on the long task.
   The lever is now specific: the every-key-by-name list. Listing only keys
   under namespaces the transcript touches, or capping the list, is the v3
   change; it is a prompt edit, not a design change.
4. **Memory tools remain a net loss on this model** — zero model-initiated
   writes in the six v2 tool-arm runs on the book, all writes the engine's —
   but they are no longer worse than the baseline, and no session was lost
   to round exhaustion.

Verdict: keep v2 as the default (memory on, memory tools off, consolidation
on). Its quality is as good as v1 and its mechanisms are the right ones; its
cost is a prompt-size problem with a known fix.

## Reproduce

```bash
git checkout memory-v2 && swift build -c release --product NVMAIServer
benchmark/memval_run.sh smoke
benchmark/memval_run.sh book      # 3 runs x 4 arms, ~4 h
benchmark/memval_run.sh pong      # 3 runs x 4 arms, ~2 h
NVMAI_MEMVAL_RESULTS=.build/benchmark-logs/memory-book python3 benchmark/memory_book.py report
```

Results for both versions are kept under `.build/benchmark-logs/memory-*-v1`
and `-v2`; point `NVMAI_MEMVAL_RESULTS` at either.
