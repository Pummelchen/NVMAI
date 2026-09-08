# Plan: the memory guard, and the shadow that feeds it

Two features, one subsystem. The **guard** is a store rule and needs no
model. The **shadow** is a small resident model that proposes facts and
checks replies. They ship separately, both off by default, and the guard
does not depend on the shadow — which matters, because the guard is the part
the measurements already justify.

Companion plan: `plan-watchdogs.md`. That work is independent and shares
nothing but the release.

## Why, in one paragraph of measurement

Replaying twelve recorded book runs through different store policies
(`benchmark/memory_sim.py`): today's rule — last write wins — leaves the
store right about 89% of the questions asked of it. Stopping a
model-derived fact from overruling one the user asserted takes that to 95%,
which is exactly what an oracle that always knows the truth reaches, so no
amount of model judgment beats the rule. Guaranteeing the user's assertions
are captured takes it to 100%. Converted through the measured store→model
transfer (a right store gives a right answer in 7 of 8 cases), that is a
projected book score of 93.1% → 93.9% (guard) → 96.0% (guard + capture).
The worst recorded run, where one extraction shuffled five characters' eye
colours, goes from 67% store fidelity to 97% under the guard alone.

## Audit revisions (2026-09-08)

Six findings from auditing this plan against the evidence it cites. Each
changed the plan.

**A1 — the simulation measured a different mechanism than this plan
specifies.** `memory_sim.py`'s `guard` matches facts by *topic* (fuzzy
keywords); this plan specifies matching by *address* (exact), which is what
Swift can actually do at write time. Those are not the same rule. Measured
both: topic guard 95% store / 93.9% projected / 27 repairs / 13 breaks;
**address guard 93% store / 94.4% projected / 31 repairs / 9 breaks**. The
planned mechanism is the better one on the metric that matters, and its
blocked-transition regressions are fewer. The plan stands; the number to
quote is 94.4%, not 93.9%.

**A2 — provenance labelling is the largest unvalidated assumption.** The
simulation derived "the user asserted this" from the harness's own event
text, which is ground truth. The implementation derives it from the
*extraction model* answering `"source": "user"`. If the model mislabels an
invention as user-asserted, the guard *protects* it, and a wrong fact
becomes harder to dislodge than it is today — a regression, not a
no-op. **Labelling accuracy must be measured before the guard is enabled**
(see the new step 0 below).

**A3 — the shadow must not be trusted to declare user authority.** Job 1
requires the shadow to mark facts as user-asserted, which is a 2B deciding
what the user said, with protection attached. Mitigation, now required: a
proposal claiming `.user` authority must carry a verbatim quote, and the
engine grants that authority only if the quote is a substring of the user's
turn. Cheap, total, and it makes the quote-only discipline load-bearing in
both jobs rather than only in the notes.

**A4 — an address guard is escapable by renaming.** One recorded run wrote
`state/inn`, `state/inn_status`, `state/rosas_inn` and `narrative/inn_burning`
for the same fact. A model that renames escapes the guard entirely. The
measurement above already includes that leakage, so the +1.3 points is net
of it; the existing `reconcile` routing for renamed namespaces mitigates
part. Stated as a known limit, not solved.

**A5 — legacy stores degrade gracefully.** Facts written before this change
carry `.model`, `.engine` or `.extractor` authority, so nothing is protected
and the guard is a no-op until new user facts arrive. No migration needed.

**A6 — the guard's regressions are blocked transitions.** All nine are a
state change refused because the store held an older protected value
(`marcus_knows_photo`, `halvorsen_confessed`). They are the price of the
rule, they are outnumbered 3:1 by the repairs, and they are visible: every
one is a `.disputed` address the model is shown. That is the intended
behaviour — ask rather than guess — but it should be reported to the user in
`nvmai-memory list`, not left silent.

## Non-goals

- Not raising the book benchmark to 99–100%. Roughly half the residual
  misses are the model failing to answer from a store that was already
  right, and three quiz questions are underdetermined. The realistic target
  is ~96%.
- Not replacing the served model's extraction with a small one. Measured:
  aggregate store fidelity looks equal (83% vs 84%) but a small model breaks
  28–45 answers to repair 2–16, for −6 points end to end.
- Not changing the coder case. Memory already carries 100% of the rules on
  every install tested, against 9–52% without. There is no headroom.

---

# Part 1 — The guard

## What it is

A fact carries an authority already: `Provenance.author` is one of `.user`,
`.model`, `.engine`, `.extractor`. Today nothing reads it when writing. The
guard reads it:

> A write whose author is the model never silently supersedes an active fact
> whose author is the user. The two are kept, the address is marked
> `.disputed`, and the bootstrap shows both with their provenance so the
> model settles it with the conversation in front of it.

A newer **user** fact supersedes an older user fact normally — that is how
state changes ("the inn burned in chapter 34") reach the store. The guard
only constrains model-derived writes.

## The two halves

**1. Provenance marking** — `sources/NVMAIServer/Core/ServerMemory.swift`

The consolidation prompt already asks for one flag per fact (`"global"`).
Add a second: `"source": "user" | "assistant"`, meaning *where in the
transcript this fact came from*, and parse it into `ProvenanceAuthor` in
`consolidationRecords`. Absent or unparseable ⇒ `.model`, so an old or
confused extraction degrades to today's behaviour rather than to a
protected fact.

The prompt must ask for the distinction plainly — the user's turns are
marked `USER:` in the transcript the extraction already receives, so this is
a labelling task, not an inference.

**2. The precedence rule** — `sources/NVMAIMemory/ContinuityStore.swift`

`set(_:in:flaggingReversions:)` grows a sibling that takes the incoming
author and the guard flag. Order of checks:

1. Address has no active fact ⇒ write.
2. Incoming author is `.user` ⇒ write (a user may always update themselves).
3. Active fact's author is not `.user` ⇒ write (today's behaviour).
4. Values fold-equal ⇒ no write (existing "unchanged" skip).
5. Otherwise ⇒ **do not supersede**; record the incoming as a disputed
   version and leave the active fact active. Log `guardHeld(key:)`.

The reversion flag and the guard are independent and compose: a reversion
still marks disputed, the guard additionally refuses the supersession.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_MEMORY_GUARD` | `0` | `1` enables the precedence rule |

Off by default. With it off, `ServerMemory` still asks for and stores the
`source` flag — provenance is free to record and worth having in the journal
before the rule that uses it is trusted. The startup line gains `guard=on|off`.

## Tests

**Unit — `tests/NVMAIMemory/MemoryGuardTests.swift` (new)**

Each case mirrors one the offline replay already measures, so the Swift
implementation and the Python oracle cannot drift apart silently:

- a model fact does not overwrite a user fact; both readable; address disputed
- a user fact does overwrite an older user fact (the burned-inn transition)
- a model fact overwrites a model fact (unchanged behaviour)
- fold-equal values write nothing regardless of authority
- with the guard off, every case behaves as today
- an absent or garbage `source` degrades to `.model`

**Unit — `tests/NVMAIServer/Core/ConsolidationSourceTests.swift` (new)**

- `"source": "user"` parses to `.user`; `"assistant"` to `.model`
- a record with no `source` parses to `.model`
- the flag survives a fenced block, a bare object, and a truncated array
  (the three shapes the parser already recovers)

**Integration — extend `tests/NVMAIServer/MemoryConsolidationTests.swift`**

- a scripted backend emits a user fact then a contradicting model fact; the
  bootstrap of the next session contains the user's value and shows the
  dispute

**Offline regression — `benchmark/memory_sim.py`**

Already implements the policy as `guard`. Add `sim` to the release gate as a
non-blocking report so a change in the Swift rule that diverges from the
oracle is visible.

## Gate before it goes on by default

Guard on, one install, book and coder, three runs each, against the frozen
v3 results. Required: book auto ≥ v3 within noise **and** no coder
regression (it is at 100%; anything below is a fail). Expected effect
+0.8 points mean, and the worst-case run repaired.

---

# Part 2 — The shadow (Qwen3.5-2B, CPU)

Later, and gated. Everything below is contingent on Part 1 shipping first,
because the shadow checks replies *against the store*, and a store the guard
has not protected is not a reference worth checking against.

## Design decision that needs a call first

NVMAI deliberately removed its last external process when Valkey went. A
small model means either:

- **(a) subprocess** — spawn `llama-server`, talk HTTP on a loopback port.
  Simple, matches what was measured, reintroduces a process to supervise and
  a port to own.
- **(b) in-process** — link llama.cpp through a Swift package. No process,
  no port; a C++ dependency and a build cost.
- **(c) NVMAI's own runtime** — needs a conversion path for a 2B into the
  engine's format, and a second resident model in an engine written around
  one.

I would build (a) first behind a protocol narrow enough that (b) replaces it
without touching callers, and say plainly in the docs that the shadow is the
one feature that runs a second process. **This is a decision to take before
implementation, not during.**

## Jobs, in the order they earn their place

**Job 1 — ledger keeper.** Proposes facts from each turn with provenance,
so a session that ends abruptly is not lost and the store does not wait for
a boundary. It *proposes*; the guard disposes. It must never be the sole
writer: measured, a small model given precedence took the store from 84% to
70%.

Per A3, a proposal claiming `.user` authority must carry a verbatim quote
from the user's turn, and the engine verifies the quote by substring before
granting that authority. A proposal whose quote does not appear is demoted
to `.model` and logged. This is the mechanical check that lets a 2B
contribute protected facts without being trusted.

**Job 2 — reply checker.** After a reply, compare its claims against the
store. **Quote only**: the note says "at 14:02 the user wrote: '…'", never
"in fact X". A quotation is checkable against the transcript, so the shadow
can be verified mechanically before it is believed; an assertion is a
hallucination waiting to happen (the 350M invented `tomas_origin: "Tomas of
Venice"`).

**Job 3 — speak to the model.** A note into the next turn's context, framed
as an observation. Last, and only on a measured precision.

## Settings, fixed by measurement

Temperature 0, `top_p` 1, thinking **off**. Thinking drops recall 65% → 12%:
the model reasons correctly, then loops re-verifying and never emits an
answer. Sampling at 0.7 is not reproducible — the same configuration gave
44% and 65% recall on two runs of the same input.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_SHADOW` | `0` | `1` starts the shadow |
| `NVMAI_SHADOW_MODEL` | none | path to the GGUF; absent ⇒ shadow stays off with one log line |
| `NVMAI_SHADOW_JOBS` | `ledger` | `ledger`, `check`, or both |
| `NVMAI_SHADOW_THREADS` | `2` | CPU threads |

Never on without an explicit model path. A missing or unloadable model
disables the shadow and logs it; it never fails a completion, exactly as
memory never fails a completion today.

## Tests

**Unit, no model — `tests/NVMAIServer/ShadowAgentTests.swift` (new)**

A `ShadowModel` protocol with a scripted double, so the plumbing is tested
without a GGUF in CI:

- a proposal reaches the store as `.model` authority and is subject to the guard
- a check that finds nothing produces no note
- a note quoting text absent from the transcript is **dropped**, and the drop
  is logged — the mechanical verification that makes quote-only safe
- the shadow being slow never delays or fails the reply
- shadow off ⇒ no calls at all

**Offline — `benchmark/memory_mini.py`**

Already runs the recorded sessions through a real small model and scores its
store with the same reader as the 35B's. Extend with a `check` mode that
measures precision on **free-text replies** rather than a clean claim list.

## The gate that decides whether the shadow ever speaks

Real precision on free-text replies, measured on the recorded runs. The
65%/100% we have was on a canonicalised fact list against a canonicalised
claim list; deployment requires extracting the claim from prose first, and
that is the step where every small model tested did badly (71–83%, with
hallucinations).

- **precision < 95%** ⇒ the shadow may write to the user only, never to the model
- **precision ≥ 95% and recall ≥ 50%** ⇒ Job 3 may proceed
- any run where a note quotes text not in the transcript ⇒ stop, this is the
  failure that would destroy trust in the feature

## Order of work

0. **Measure labelling accuracy first (A2).** Re-run consolidation over the
   recorded transcripts with the `source` flag in the prompt, and score the
   labels against the known user text. A fact the model calls user-asserted
   that the user never said is the failure that makes this feature worse
   than nothing. Required before the rule is enabled anywhere:
   **mislabel rate < 5%**, and no mislabel on a fact the model invented.
1. Guard: provenance marking, precedence rule, unit tests, integration test.
2. Guard gate: one install, book + coder, three runs.
3. Shadow runtime decision (a/b/c), then the protocol and the scripted double.
4. Job 1 behind `NVMAI_SHADOW=1`, with the store still guarded.
5. Free-text precision measurement offline.
6. Jobs 2 and 3 only if the gate passes.


## Post-implementation audit (2026-09-08)

An independent read of the implemented guard found four ways a
model-derived write still reached a fact the person had asserted. The plan
described the rule correctly; the implementation applied it in one place.

**A7 — every model tool write walked past the guard.** The guarded write
existed only on `ContinuityStore` and had exactly one caller, consolidation.
`memory_set` and `memory_delete` used the plain protocol methods, so with
`NVMAI_MEMORY_GUARD=1` the model could overwrite or retire the person's fact
by tool call in the very next session. The guarded write and delete are now
on the `MemoryStore` protocol, which is what makes "every writer" true
rather than aspirational; a store with no provenance says plainly that it
only wrote, instead of reporting protection it cannot give.

**A8 — authority did not survive a read.** `record(from:)` restored
`isDisputed` and not `isUserAsserted`, so every read-modify-write relabelled
the person's fact as the model's. A single `memory_append` disarmed the
guard on that address permanently.

**A9 — the disputed marking was deterministically invisible.** Disputing an
address updates it; the bootstrap's "changed in the most recent session"
list is sorted by exactly that, so a freshly disputed key is *always* on
that list — and the marker was rendered only on the other list. The guard
held the write, recorded the conflict, and then handed the next session the
surviving value with nothing to say anything was disputed. Since being seen
is the entire reason for disputing rather than refusing silently, this made
the feature's visible half a no-op.

**A10 — the person's own facts were the one category never covered.** The
shared-workspace path took the unguarded write, and the degraded local store
skipped the guard through a failed cast. Conventions and preferences are the
most user-asserted things in the store.

**A11 — `NVMAI_MEMORY_GUARD=off` switched the guard on**, because the parse
was `!= "0"`.

All five are fixed, with a test each. The step-0 gate is unchanged and still
stands in front of enabling any of it.


## Step 0 measured (2026-09-08)

The gate in front of enabling the guard has been run on two models, with
`benchmark/guard_source_rate.py`.

| Run | Facts | Labelled `user` | Mislabelled | Gate |
| --- | --- | --- | --- | --- |
| Qwen 3.6 35B 8-bit | 59 | 28 | 0 | met |
| Ornith 1.5 35B 4-bit | 45 | 19 | 2 (10.5%) | not met |

**The guard stays off, and the reason is now measured rather than
suspected.** One model labels reliably and the other does not, which is the
same split every other memory measurement on this project has shown.

Both Ornith failures are composite facts: one key holding a clause from the
person and a clause the model invented, labelled `user` because half of it
is. That is the obstacle, and it is upstream of the guard. `source` cannot
be a property of a value that has two sources, and a guard that protects a
composite protects the invented half. Splitting composites is the next piece
of work if the guard is to be enabled anywhere -- not a better prompt for
the label, which is being asked an unanswerable question.

The book scores for these two runs were 94% and 92%, both inside the
established noise band, so the labelling work costs nothing measurable.


## The composite fact, and what the side-engine could not do (2026-09-08)

Step 0 failed on Ornith because of composite facts: one key holding a clause
the person wrote and a clause the model invented, labelled as the person's
because half of it is.

The obvious repair is to ask the small resident model which half is whose --
a verification task, which is what the earlier measurements said a 2B is
good for, and the CPU side-engine now exists to run one. It was tried, on
all 47 facts that claimed the person's authority across both runs.

| prompt | repaired | damaged |
| --- | --- | --- |
| the value alone | 0 | 6 |
| the key too, and told that wording may differ | 1 | 3 |

Net negative both times, and the damage falls exactly where the repairs were
meant to. Shown "Aldo: blue eyes, the mayor; unlocks the pre-decommission
lighthouse records in chapter 64" -- half the person's, half the model's --
it answers NONE and throws away the half that is theirs. The first prompt
was unfair, withholding the key so the model had to attribute "brown" with
no idea it was an eye colour; fixing that halved the damage. A third attempt
would be tuning toward a desired answer on a set containing two positives.

**So the composite has to be prevented, not repaired.** The consolidation
prompt already says "one fact per key" and the 35B violated it anyway. That
is the next thing to work on if the guard is to be enabled: an extraction
whose values have one source each, checked, rather than a repair pass over
values that have two.

`benchmark/composite_split.py` stays, because a measured negative is worth
keeping and worth re-running when either the model or the extraction
changes.


## Authority requires one fact (2026-09-08)

The composite was the obstacle, the side-engine could not repair one, so the
guard stops relying on the extraction getting it right: **a value that holds
more than one fact cannot carry the person's authority.** It is demoted to
the model's before the guard sees it, in `ContinuityStore.author(of:)`, via
`MemoryRecord.carriesUserAuthority`.

Three signs of more than one fact, each measured on the 47 facts the two
recorded runs labelled as the person's:

| sign | catches, of 2 known bad | flags, of 45 that scored grounded |
| --- | --- | --- |
| a semicolon | 2 | 10 |
| more than one sentence | 1 | 0 |
| over 120 characters | 2 | 4 |

The semicolon does the work, and four of the ten it flags turn out to be
mislabels the word-overlap scorer missed — Aldo "unlocks the
pre-decommission lighthouse records in chapter 64", Halvorsen "wrote a
second, truthful certificate in chapter 63" — where the invented clause
reused enough of the person's vocabulary to pass. So the true mislabel rate
on Ornith was worse than the 10.5% first measured, and the rule removes all
of it.

With the rule, both gate conditions are met on both models:

| Run | Labelled `user` | Demoted | Carrying authority | Mislabelled |
| --- | --- | --- | --- | --- |
| Qwen 3.6 8-bit | 28 | 0 | 28 | 0 |
| Ornith 4-bit | 19 | 12 | 7 | 0 |

**Two things this does not mean.** It does not mean the guard should be on
by default: step 0 is one of the gates, and the rest — observation on a real
workload — has not been run. And the rule was drawn from the same corpus it
is scored on, so what is established is that it removes every composite
*here*, not that the threshold generalises; it wants a run these numbers did
not shape.

Its failure mode is the safe one. A composite the person really did assert
loses protection, which is today's behaviour; nothing gains protection it
should not have.
