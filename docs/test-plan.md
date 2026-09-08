# Test plan: the memory database and the watchdogs

Two features, both off by default, both meant to raise the quality of what a
person gets out of a long piece of work. This is the list that decides
whether they do, numbered so progress is trackable, with a pass criterion
for each that is a measurement rather than an opinion.

**The standing rule for every model test: memory must not lose.** A feature
that costs the user points on any scenario is not shipped, whatever it wins
elsewhere. This has happened once already — Ornith 4-bit went 94% to 87%
with memory on, because memory faithfully preserved the model's own drift
from the story bible — so it is the first thing every scenario checks.

## Status

| # | Test | State |
| --- | --- | --- |
| 1 | Swift unit suite green | |
| 2 | Lint clean | |
| 3 | Watchdog corpus calibration: zero false positives | |
| 4 | Watchdog Swift/Python fixture agreement | |
| 5 | Memory simulator agrees with recorded runs | |
| 6 | CPU engine parity with the numpy reference | |
| 7 | S1 Book — control (memory off) | |
| 8 | S1 Book — memory on | |
| 9 | S2 Coder — control | |
| 10 | S2 Coder — memory on | |
| 11 | S3 Correction — control | |
| 12 | S3 Correction — memory on | |
| 13 | S4 Two projects — isolation | |
| 14 | S5 Retrieval at volume — memory tools | |
| 15 | Watchdogs observed across every scenario run: no false trip | |
| 16 | Guard step 0 across every scenario run: no invented fact labelled `user` | |
| 17 | Memory does not lose on any scenario | |
| 18 | Token overhead of memory within budget | |

## Part A — what runs without a model

Fast, deterministic, and the gate on everything else. If any of these is
red, a model run measures a moving target.

**1. Swift unit suite.** `swift test`. Pass: every test green. Currently
1233 tests across 195 suites.

**2. Lint.** `tools/lint.sh`. Pass: no force-casts outside tests, no new
function over 120 lines, every `@unchecked Sendable` documenting its
invariant.

**3. Watchdog corpus calibration.** `benchmark/watchdog_calibrate.py` over
every recorded reply and exchange. Pass: zero false positives for `loop` and
`stub`, and the known-bad reply still caught. This is what keeps a threshold
a measurement instead of a guess.

**4. Watchdog fixture agreement.** `watchdog_calibrate.py --selftest` plus
the Swift test that reads the same fixture. Pass: no mismatch. The
calibration is a Python port; if it drifts from the Swift detector the
corpus numbers stop describing what ships.

**5. Memory simulator agreement.** `benchmark/memory_sim.py validate`. Pass:
≥ 90% agreement with the recorded runs it replays. This is what lets a store
policy be tested in seconds instead of two days.

**6. CPU engine parity.** `NVMAIBench cpu35` against
`tools/qwen35_reference.py`. Pass: cosine ≥ 0.99999 on every check, both
widths.

## Part B — the scenarios

Each scenario runs twice: a **control** arm with memory off, and an **auto**
arm with memory on and the engine consolidating. The control is not
decoration — it is the only thing that says whether memory helped.

### S1 Book (exists) — tests 7, 8

A hundred-chapter novel over ten sessions, with a bible of fixed facts and
plot events delivered as the work goes. Quizzed each session on fourteen
facts. Measures: fixed attributes surviving, state changing when the person
says it changed, and no contradiction of the bible.

### S2 Coder (exists) — tests 9, 10

The same game written three times, in Swift, Python and C99. Parameters
fixed in the first stage must be reproduced in the later ones without the
prompt restating them. Measures: decisions carried across sessions.

### S3 Correction (new) — tests 11, 12

**The gap both existing scenarios leave.** In neither does the person ever
change their mind, and that is the case the guard exists for. The person
states a fact, works on it, then *revises* it — and later still, asks. Three
kinds of revision, because they fail differently:

- a **fixed attribute** corrected ("Marcus's eyes are green, I was wrong");
- a **decision** reversed ("we are moving off SQLite to Postgres");
- a **rule** tightened ("chapters are three sentences now, not two").

Measures: the store holds the *latest* statement, never resurrects the
superseded one, and does not treat the correction as a contradiction to be
disputed away. Pass: 100% on the revised facts, and no regression on the
unrevised ones.

### S4 Two projects (new) — test 13

Two workspaces used alternately in one server's lifetime, with deliberately
confusable facts — same character names, different eye colours; same module
names, different decisions. Measures: cross-project leakage, which the
workspace design exists to prevent and which no test currently exercises.
Pass: **zero** facts from one project appearing in the other's answers. This
one is pass/fail rather than scored: a leak is a defect, not a lost point.

### S5 Retrieval at volume (new) — test 14

More facts than the bootstrap can carry — the bootstrap is capped at 60
records and 16 KB — then questions whose answers are outside it. The model
has the memory tools on, so it has to go and look. Measures: the tool
surface, which ships off and has never been measured at a volume where it
matters, and the retrieval ranking underneath it. Pass: the answers are
correct, and the tool rounds do not run out.

This is also the only scenario that can produce a tool loop, so it is where
the ping-pong watchdog gets its first real exposure.

## Part C — the two features, watched across every run above

**15. Watchdogs observing.** Every scenario run in Part B runs with
`NVMAI_WATCHDOGS=1`. Pass: no trip that hand review calls a false positive.
A trip that is real is a finding, not a failure.

**16. Guard step 0.** `benchmark/guard_source_rate.py` on every run's
journal, with every flag read by hand. Pass: no invented fact wearing the
person's label. The count is not the verdict — word overlap cannot judge a
boolean whose claim lives in its key — so the list is read, not totalled.

## Part D — the quality bar

**17. Memory does not lose.** For every scenario, the memory-on arm scores
at least as well as the control, within the ±15% run-to-run spread this
machine is known to have. A loss outside that band stops the release and
gets diagnosed, not averaged away.

**18. Token overhead.** Memory costs prompt tokens for its bootstrap and a
generation for consolidation. Pass: prompt-token overhead under 20% against
the control, and consolidation confined to idle time.

## How a failure is handled

Fix the cause, re-run the test, and record what changed. A threshold is
never moved to make a test pass; if a threshold is wrong, the measurement
that shows it is wrong goes in the plan beside it.
