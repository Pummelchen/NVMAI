# Memory that remembers, and the guard

Everything else in this series makes NVMAI *work*. This is the article about
the feature that makes it feel like it *knows your project*.

It is also the one with a genuine safety question attached, and the project
has a real answer to it. So this article is in two halves: what memory does,
and the guard that stops it going wrong.

> **Off by default.** Nothing about memory is constructed until you switch it
> on. If you never want it, you never get it.

## The problem

A coding assistant rediscovers the same things every session.

Why an odd class still exists. Which refactor was already tried and
abandoned, and why. What the build actually does *on this machine*. Which
decisions you made two weeks ago and have since forgotten you made.

None of that fits in a prompt. It is unbounded, and most of it is irrelevant
to any single question. It belongs in a **store the model can query** — which
is what NVMAI's memory is.

**It is not the KV cache.** The KV cache is the current conversation's
working state and vanishes when you stop. Memory is durable, cross-session,
and survives restarts. If you read
[Long context and the KV cache](07-long-context.md), this is the other thing.

## Turning it on

One environment variable:

```bash
NVMAI_MEMORY=1 tools/server_launcher.sh
```

That is the whole configuration for a first try. Three details worth knowing:

- **There is no database to install and no service to start.** Memory runs
  inside the server process. No port, no connection to lose, nothing else to
  keep alive.
- **It is scoped to your project.** Memory is filed under
  `<namespace> / <user> / <workspace>`, where the workspace is the repository
  you launched from — its directory name plus a digest of its full path, so
  two checkouts of the same repo never share memory.
- **Launch from your project folder.** If you start NVMAI from your home
  directory, it refuses and tells you why: one store would otherwise collect
  every project you own into a single fact pile. (`NVMAI_MEMORY_WORKSPACE=name`
  overrides it when you have a reason.)

You can point a different workspace per request with an `X-NVMAI-Workspace`
header, which lets one running server serve several checkouts.

## What it looks like in practice

A short session, start to finish, because the value is easier to see than to
describe.

```bash
cd ~/code/my-ledger
NVMAI_MEMORY=1 ~/NVMAI/tools/server_launcher.sh
```

Now connect your client as usual and work normally. Say, for a first session:

> The ledger service uses PostgreSQL 16 and migrations live in `db/migrate`.
> Please add an index on `transactions.created_at`.

Ask it to remember the important parts if you like — with memory tools on,
that is `memory_set`. With the defaults, the engine works it out for itself
once you stop for a moment.

**Then quit everything, come back tomorrow, and start the same way:**

```bash
cd ~/code/my-ledger
NVMAI_MEMORY=1 ~/NVMAI/tools/server_launcher.sh
```

The next session opens already knowing the database, the migration folder, and
the decisions you made — not because the conversation was replayed, but
because the model was handed a short list of what is known, and can fetch the
detail for whatever it is about to do.

That is the whole idea: **look up what the project already established, before
starting the task.** It is the difference between an assistant that rediscovers
your codebase every morning and one that remembers it.

Two practical notes:

- **Keep working in the same folder.** That is what makes it *your* project's
  memory rather than a pile of everything.
- **It is plain files.** One journal per project, under the `memory/` folder in
  the NVMAI checkout by default. You can read them, and deleting one forgets
  that project. Nothing goes anywhere else.

## What the model gets

A short system note — about 200 words — explains that memory exists, when to
read it, when to write it, and what *not* to store. At the start of a session
the model receives **a list of what exists with one-line summaries, never full
values**. The detail is one function call away. That bootstrap is capped twice
— by record count and by bytes — so starting a session can never dump the
entire store into the prompt.

**By default, the model can *read* memory but not write it directly.** Writing
is done by the engine instead: after about 30 seconds of quiet — while you are
reading the reply, or have walked away — it distils the session and records
what is worth keeping. That costs no latency on the critical path, because it
never runs while you are waiting. If you want the model to also write during a
session, you turn on its memory functions:

```bash
NVMAI_MEMORY=1 NVMAI_MEMORY_TOOLS=full tools/server_launcher.sh
```

| Function | What it is for |
| --- | --- |
| `memory_search` | Find memories by text, prefix, tags, or importance |
| `memory_get` | Read one memory by its exact key |
| `memory_list` | List keys, optionally under a prefix |
| `memory_set` | Write or replace a memory |
| `memory_append` | Add a line to an existing memory |
| `memory_delete` | Remove something wrong or obsolete |

`NVMAI_MEMORY_TOOLS=minimal` gives you the first three essentials
(`memory_set`, `memory_get`, `memory_list`).

The model never gets a raw database command — just these functions. They are
answered by the engine itself; your client's own tools still pass through
untouched (see [Connecting your apps](06-connecting-your-apps.md)).

**Why tools are off by default is a real trade, not caution.** All six
definitions cost about **1,120 prompt tokens**, against roughly 210 for the
instruction note alone — five sixths of memory's prompt cost is tool schemas,
paid on every session. On an engine that prefills at about twice its decode
rate, that is wall-clock time you would feel. Meanwhile the `memory_list`
function is not optional even in the minimal set: without it, a model whose
bootstrap is empty guesses keys, and measured on a 35B model that guessing ate
every tool round — 136 reads to 2 writes across ten sessions, and three
sessions returned no answer at all.

Cost, for scale: a hundred-chapter novel across ten sessions came to about
**100 KB**. Retention defaults to 30 days since last touch, with at most 100
projects kept.

## The guard: the part worth understanding

Now the risk, stated plainly.

If a model can write facts, it can write a *wrong* fact. Worse, it can
**overwrite something you told it** — and silently win. You say the project
uses PostgreSQL; a later session hallucinates MySQL; the store now disagrees
with the person who actually knows. A memory system that does that is worse
than no memory at all, because a missing fact makes the model ask, while a
stale one makes it confidently build the wrong thing.

The **guard** is the rule that stops it:

> A fact attributed to the *model* does not overwrite a live fact that the
> *person* asserted. The person's value stays, the address is marked
> **disputed**, and the next session sees both.

The details matter, so here they are:

- **The person always supersedes their own facts.** If *you* change your mind,
  that is an update, not a dispute.
- **Model-over-model is untouched.** The guard is about protecting your
  statements, not freezing the store.
- **A model write that merely agrees is stored normally.**
- **Disputed is visible, not hidden.** Both values surface, marked, rather
  than one quietly winning.

It is on by default whenever memory is on, because it was measured rather
than hoped for. Across twelve recorded runs, the old "last write wins" rule
left the store right about 89% of the time. Protecting user-asserted facts
lifts that to **93% store fidelity** with the shipped address-matching rule
(94.4% projected onto answers) — and, importantly, that is essentially the
ceiling: a simulator given an *oracle* that always knows the truth reaches
95%. There is almost nothing left for a cleverer model to win. In the
project's book scenario, the guard scored **97% against a 98% control** — one
point off, inside the noise — and fired exactly once, holding a rule the
model had tried to overwrite. One hold, three points.

## The 2B on the CPU: a second, smaller worker

Here is the part of memory that is still being built, and it is worth
understanding because the idea is genuinely good.

Extracting facts from a session means asking a model to read a conversation
and judge what is durable. You do not want the 35B — the one you are waiting
on — spending time on bookkeeping. So the design calls for a **second,
much smaller model on the CPU**: **Qwen 3.5 2B**, running on the processor
cores while the big model owns the graphics chip.

Why it is interesting, and why it is not free:

- The CPU side is *core*-idle during generation — NVMAIServer uses about
  **0.20 of one core out of eight**. So the idea looks free.
- **It is not free, because decode is bound by memory, not cores.** Both
  engines compete for the same memory system. Measured, with a 35B
  generating:

| CPU threads | 2B speed | Cost to the 35B |
| --- | ---: | ---: |
| 1 | ~7 tokens/s | **3%** (inside the noise) |
| 2 | ~13 tokens/s | 13% |
| 3 | ~16 tokens/s | 30% |
| 4 | ~22 tokens/s | 31% |

**One thread is effectively invisible and still buys a usable side model.**
Past two threads the two engines are simply taking turns at the memory
controller. So the width is a *scheduling* decision: one thread while you are
waiting for an answer, four in the gaps between requests. That is the design
the measurement produced — and the opposite of what "the CPU is idle, so it is
free" would have suggested.

There is a measured limit on what a 2B can be trusted with, and it is
instructive. Asked to decompose a long statement and return the surviving
clauses, it failed badly — it echoed its input back in 38 of 47 answers.
Asked **one yes/no question at a time** — "did the person state this
clause?" — it scored **92%**, kept 28 of 30 of the person's clauses, and
rejected 7 of 8 model inventions.

The lesson: a small model is not a small version of a big one. It is a
capable *worker* that fails at composition. Asked one question at a time, it
does exactly the job memory needs.

**Status, honestly:** the memory store and the guard are implemented and on
by default when memory is on. The Qwen 3.5 2B side-engine is measured and
designed but still being built — today the CPU models are a selectable
engine in their own right (see [Choosing a model](04-choosing-a-model.md)),
and using one as memory's resident helper is the next step. The plan and its
measurements live in the repository's design notes.

## Should you turn it on?

**Yes, if** you work on a long-running project across many sessions and want
the assistant to remember decisions. This is its whole purpose, and repo
scoping keeps it tidy.

**Try it first if** you are unsure. It is one environment variable, it is off
by default, it lives in one folder you can delete, and the guard means a
model cannot quietly overrule you.

**Leave it off if** you only ever have one-off conversations. There is
nothing to gain.

## Where to go next

That is the whole feature tour. Now the question behind all of it — *how does
this run a model that does not fit?* →
**[Why NVMAI can run models that "don't fit"](09-why-nvmai-runs-big-models.md)**

The memory design document and the guard's measurements:
[`docs/agent-memory.md`](https://github.com/Pummelchen/NVMAI/blob/main/docs/agent-memory.md)
and
[`docs/plan-memory-guard-and-shadow.md`](https://github.com/Pummelchen/NVMAI/blob/main/docs/plan-memory-guard-and-shadow.md).
Runtime settings:
[Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls).

*Memory defaults and the guard/CPU figures are from NVMAI 5.1's own
measurements on a base 8-core M3 with 24 GB. The side-engine measurements are
recorded in the repository's CPU side-engine plan.*
