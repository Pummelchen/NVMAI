# Agent memory

NVMAI can give a model memory that outlives a conversation: durable facts it
writes in one session and reads in another, scoped to the repository being
worked on. It is off by default, it is not the KV cache, and the serving path
does not depend on it.

## Why it exists

A coding agent rediscovers the same things every session. Why an odd class
survives, which refactor was tried and abandoned, what the build does on this
machine. None of that belongs in a prompt, because it is unbounded and mostly
irrelevant to any one question. It belongs in a store the model can query.

## Architecture

```
NVMAIServer ── MemoryBackend (decorator) ── inner backend (inference)
                     │
                     ├─ installs the instruction fragment + memory tools
                     ├─ executes memory_* calls the model makes
                     └─ MemoryService
                            ├─ ContinuityStore ─┐
                            ├─ ContinuityJournalStore ─┴─ ContinuityEngine
                            │                              └─ FileJournal
                            └─ InMemoryStore (fallback, and the test double)
```

Memory runs inside the server process. There is no database to install, no
port to open and no connection to lose: `ContinuityEngine` is a Swift actor in
the same binary as the model, and the only thing it touches outside memory is
one journal file per workspace. The engine itself is documented in
[`sources/ContinuityCore/README.md`](../sources/ContinuityCore/README.md).

The engine's request lifecycle is unchanged. `MemoryBackend` wraps any
`ServerInferenceBackend`: on the way in it installs a short system fragment
and the memory tool definitions, and on the way out it services the memory
tool calls the model made and asks the inner backend to continue. With memory
disabled the decorator is never constructed.

**Why the engine runs these tools when it runs no others.** NVMAI returns tool
calls to the client, which executes them. That is right for the client's own
tools and useless for memory: no coding CLI knows about NVMAI memory, so a
memory tool the client would have to run is a memory tool nothing runs. Memory
tools are therefore the one kind the engine answers itself. Client tools still
pass through untouched, and a turn that calls one ends the memory loop rather
than stranding its result.

### Scoping

```
<namespace> / <user> / <workspace>
```

- **namespace** separates deployments sharing one machine (`nvmai` by default).
- **user** separates people sharing one server (the OS user by default).
- **workspace** is the repository. The start scripts pass the directory they
  were launched from, and the identifier is the directory name plus a digest
  of its full path, so two checkouts of one repository never share memory.

Every backend key carries the scope as a hash tag:

```
nvmai:mem:{nvmai/ada/nvmai-4f2a91c3}:r:decisions/sync    the record, JSON
nvmai:mem:{nvmai/ada/nvmai-4f2a91c3}:idx                 sorted set of keys
nvmai:mem:{nvmai/ada/nvmai-4f2a91c3}:sessions            last 50 sessions
```

The index is what keeps this bounded. Listing, searching and bootstrap read
the index and then fetch a capped batch; nothing issues `KEYS` or `SCAN`, so
one scope's cost never depends on what other scopes hold. A test asserts those
commands are never sent.

A single server can serve several checkouts: send `X-NVMAI-Workspace` with a
request, or pin the server to one workspace by setting
`allowsPerRequestWorkspace` false.

### Records

Arbitrary UTF-8, including JSON, stored verbatim. Metadata is optional:
importance (ranks the bootstrap), confidence, tags, the session that wrote it,
and created/updated timestamps. A rewrite keeps the original creation time,
because the model is correcting a fact rather than making a new one.

## Memory tools (not the client's tools)

| Tool | Purpose |
| --- | --- |
| `memory_search` | Find memories by text, prefix, tags or importance |
| `memory_get` | Read one memory by exact key |
| `memory_list` | List keys, optionally under a prefix |
| `memory_set` | Write or replace a memory |
| `memory_append` | Add a line to an existing memory |
| `memory_delete` | Remove a memory that is wrong or obsolete |

The model never receives a database command, and the scope comes from the
session, not from the call, so naming another workspace in the arguments
cannot redirect a write.

## The system prompt fragment

About 200 words, merged into the session's system message. It says memory
exists, when to read, when to write, what not to store, and that retrieved
memory may be stale and worth verifying. It lists bootstrap keys with
one-line summaries, never their full values: the bootstrap says what exists,
and the text is a tool call away.

The bootstrap is bounded twice, by record count and by bytes — forty records
and 16 KiB by default, each value summarised to 200 characters, ties going to
the older fact so a foundation outranks last session's state. Twenty was too
few: a novel's bible plus its running state passed thirty keys by the fourth
session and the eye colours were crowded out by `state/*`. A test fills a
store with 500 records and asserts session start can never return more than
the limits allow.

## Configuration

Environment variables, which is how the start scripts pass them:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_MEMORY` | `0` | `1` enables memory |
| `NVMAI_MEMORY_DIR` | `<NVMAI>/memory` | Directory holding the project files (the binary alone falls back to `~/.nvmai/memory`) |
| `NVMAI_MEMORY_RETENTION_DAYS` | `30` | Delete a project file untouched this long; `0` keeps all |
| `NVMAI_MEMORY_MAX_WORKSPACES` | `100` | Keep at most this many project files, oldest first out; `0` is no cap |
| `NVMAI_MEMORY_FSYNC` | `0` | `1` forces every append to disk |
| `NVMAI_MEMORY_CACHE_MIB` | none | Optional ceiling for the whole store, in MiB |
| `NVMAI_MEMORY_NAMESPACE` | `nvmai` | Deployment namespace |
| `NVMAI_MEMORY_USER` | OS user | User component of the scope |
| `NVMAI_MEMORY_WORKSPACE` | from `NVMAI_WORKSPACE_DIR` | Explicit workspace id |
| `NVMAI_WORKSPACE_DIR` | launch directory | Directory the workspace id derives from |
| `NVMAI_MEMORY_MAX_VALUE_BYTES` | `65536` | Largest single memory |
| `NVMAI_MEMORY_BOOTSTRAP_LIMIT` | `60` | Bootstrap record cap |
| `NVMAI_MEMORY_BOOTSTRAP_BYTES` | `16384` | Bootstrap byte cap |
| `NVMAI_MEMORY_TOOL_ROUNDS` | `4` | Memory-tool rounds serviced per request |
| `NVMAI_MEMORY_TOOLS` | `off` | The six `memory_*` functions: `off`, `minimal` (set, get, list) or `full`. Never affects the client's own tools. |
| `NVMAI_MEMORY_CONSOLIDATION` | `1` | `0` disables the engine writing memory at session boundaries |
| `NVMAI_MEMORY_CONSOLIDATION_IDLE_SECONDS` | `30` | Quiet time after a turn before a session is distilled |
| `NVMAI_MEMORY_LOCAL_FALLBACK` | `1` | `0` disables memory instead of degrading |
| `NVMAI_MEMORY_GUARD` | `1` | Stops a model-derived fact from silently superseding one the person asserted. `0` turns it off; see below for what it is worth. |

### The guard

With `NVMAI_MEMORY_GUARD=1`, a write the extraction attributed to the model
does not overwrite a live fact the person asserted. The person's value
stays, the address is marked disputed, and both values are visible to the
next session. The person always supersedes their own facts, model-over-model
is untouched, and a model write that agrees is stored rather than held --
holding it would put a conflict in front of the next session over nothing.

It rests entirely on the `source` field the consolidation prompt asks for,
which is why it stayed off until that field was measured. The gate was under
5% mislabelled and no mislabel at all on an invented fact: a mislabelled
invention would be *protected*, which is worse than the old behaviour rather
than merely different.

**It is on now, and it earns the default.** On Ornith 1.5 at 4-bit the book
scenario is the one place memory has ever lost — 98% without it and 94% with
— because memory faithfully preserves that model's drift from the story
bible. With the guard on it scores 97%, one point off the control and inside
the noise. It fired exactly once in that run:

```
memory guard kept the user's fact, marked disputed: rules/marcus_knowledge
```

That is the bible's hard rule about what Marcus may not learn before chapter
60, which the model tried to overwrite. One hold, three points. The offline
simulator predicted it independently and specifically, putting an unguarded
store at 61% of the answers on this install and a guarded one at 98%.

`benchmark/guard_source_rate.py` is that measurement. It replays a recorded
book run's facts against exactly what the person put in front of the model
in each session -- the story bible plus the plot events delivered so far,
which the offline simulator already knows precisely -- and flags any fact
claiming the person's authority whose substance is nowhere in the person's
own words.

| Run | Facts | Labelled `user` | Demoted | Carrying authority | Mislabelled |
| --- | --- | --- | --- | --- | --- |
| Qwen 3.6 35B 8-bit, book, auto | 59 | 28 | 0 | 28 | 0 |
| Ornith 1.5 35B 4-bit, book, auto | 45 | 19 | 12 | 7 | 0 |

**Both gate conditions are met, and the demotion column is why.** Measured
first without it, Ornith mislabelled 2 of 19 — 10.5%, a clear failure — and
Qwen mislabelled none of 28. The split was the finding: the label's quality
is a property of the model, and it lined up with everything else measured
here, where memory was a clean win on Qwen 8-bit and a loss on Ornith 4-bit.

Both of Ornith's failures were the same shape. At session 7 it wrote
`characters/rosa` as "Rosa: hazel eyes, keeps the inn; raised a new inn from
charred walls in chapter 65 while keeping the old hearth as a monument" and
labelled it `user`. The first clause is the person's, from the bible; the
second is the model's own chapter 65. One key, two sources, one label — and
no label is right.

So authority now requires an **atomic** value, and a composite is demoted to
the model's before the guard ever sees it. That removes every composite in
the corpus, including four whose invented clause reused enough of the
person's vocabulary to slip past a word-overlap check. It costs nothing
where the extraction already behaves — Qwen loses none of its 28 — and
Ornith, which writes composites, keeps 7 of 19.

**A third run, which these numbers did not shape, changed the conclusion
about the *measurement* rather than about the model.** Ornith 4-bit again:
41 facts, 22 labelled as the person's, and the scorer flagged nine. All nine
were correct — every one a boolean whose claim lives in its key, like
`rules/marcus_must_not_learn_photo_before_chapter_60 = true`, where the
value carries no words at all. Scoring the key as well as the value cut the
flags to four, and those four were correct too.

So across three runs and 104 facts claiming the person's authority, hand
review finds **no invented fact wearing the person's label**. The labelling
is good on both models. What is not good is the automatic scorer: word
overlap cannot tell a boolean rule from an invention, and tuning what it
reads flips its verdict. It finds candidates worth reading; it does not
decide.

The two genuine composites it did find on the first Ornith run are still
genuine — chapter 65 and the ten-chapter summary are the model's own output
under the person's label — so the atomicity rule keeps its justification.
Its failure mode is the safe one: a composite the person really did assert
loses protection, and nothing gains protection it should not have.

The real fix is still upstream — an extraction whose values have one source
each, which its own prompt already asks for. This is the guard refusing to
depend on that until it is true.

The control matters as much as the number. A scorer generous enough to
ground anything would report a perfect run whatever the model did, so the
facts labelled `model` go through the same test: they ground at 36% against
the user-labelled facts' 100%. The gap is what says the test discriminates.

Two things it got wrong first, both worth knowing. Its opening candidates
were "hazels" against a bible that says "Rosa, hazel eyes", and "burned"
against a user event that says the inn "burns" -- inflection, not invention
-- so it strips English suffixes before comparing. And it prints every fact
it flags, because "no invented fact labelled `user`" is a claim a person
checks, not a number a script reports.

### Store size

There is no RAM ceiling by default, and there is nothing to defend against.
Measured by replaying the benchmark journals with the store's own accounting:

| Task | Facts held | Resident (facts, history, log) | Journal on disk |
| --- | ---: | ---: | ---: |
| A hundred-chapter novel, ten sessions | 93–134 | **96–121 KB** | 194–249 KB |
| Pong in three languages, three sessions | 10–22 | **10–20 KB** | 25–45 KB |

That is a rounding error next to one KV-cache block, so the machine-tier
ceilings that used to sit here (256 MiB at 8 GB and so on) could never bind
and only confused. The bounds that remain are hygiene, not budgets: a fact is
at most 64 KiB, an address keeps 32 versions, a session keeps 200 turns and a
workspace 100 sessions in memory, and the file keeps everything.

`NVMAI_MEMORY_CACHE_MIB` sets a ceiling for anyone who wants one. It covers
every open workspace together, split three quarters to facts and one to the
journal; at the cap facts refuse and the journal evicts its oldest sessions
from memory, and the least recently used workspace is closed.

### One writer per workspace

A workspace's journal is held under an exclusive lock for as long as a server
has it open. Start a second server on the same workspace and it runs *without*
persistence rather than writing into the first one's file: two processes each
holding their own copy of the state would see none of each other's writes and
interleave their appends into something that replays as two braided histories.

The second server logs the refusal, and the session prompt tells the model its
writes will not outlive the session. The lock is released when the process
exits, however it exits, so a crash never strands a workspace.

## Setup

There is none. Memory is off until you ask for it, and turning it on needs no
service:

```bash
NVMAI_MEMORY=1 tools/server_launcher.sh --client server --model qwen36 8
```

The launcher exports the memory environment itself: the workspace is
the directory you launched from, so two checkouts never share memory, and the
ceiling follows the table above. To place the store elsewhere or name the
workspace explicitly:

```bash
NVMAI_MEMORY=1 NVMAI_MEMORY_DIR=/var/lib/nvmai \
  NVMAI_MEMORY_WORKSPACE=my-project tools/server_launcher.sh --client server --model ornith 8
```

State lives in one file per project,
`<NVMAI>/memory/<namespace>/<user>/<workspace>.ndjson`, created owner-readable
only inside an owner-only directory, with a `.lock` sidecar beside it. Sessions
are records inside that file, not files of their own. Deleting a project's
memory is deleting its file, and backing it up is copying it.

The files do not pile up, and retention never removes a fact. A project file
untouched for 30 days (`NVMAI_MEMORY_RETENTION_DAYS`) has its session log
expired — the transcript, which is the bulk of it — and keeps its facts, their
history and its title, so a novel paused for six weeks comes back with its
bible. At most 100 project files are kept (`NVMAI_MEMORY_MAX_WORKSPACES`),
oldest by last write going first; that cap is the only rule that deletes
facts. The sweep runs at start and whenever a new project file is created; a
project this server has open, and the person's shared file, are never touched.
Each action is logged: `memory expired the session log of 2 project file(s),
facts kept: …` and `memory swept 1 project file(s) (more than 100 projects): …`.

### The person's own facts

A consolidation can mark a fact `"global": true` — a convention wanted
everywhere, a language, a tone, a tool always used. Those go to the shared
workspace `_global` under the same user, and every project's bootstrap opens
with them under "About this person, in every project". Project facts never go
there, and a request cannot name `_global`: a derived workspace id can never
spell it and the header is refused.

### Seeing and correcting memory

```bash
swift run nvmai-memory projects                 # every project, newest first
swift run nvmai-memory list photograph          # a project's facts (prefix is enough)
swift run nvmai-memory show photograph state/inn
swift run nvmai-memory delete photograph state/inn   # retired, kept in history
swift run nvmai-memory forget photograph --yes       # the whole project
swift run nvmai-memory list global              # the person's shared facts
```

Reads take no lock and work while a server is running. `delete` and `forget`
need the workspace and refuse it while a server holds it — stop the server, or
let the next consolidation supersede the fact. `--dir` or `NVMAI_MEMORY_DIR`
selects the directory.

A workspace named per request gets its own file too, so one project's memory
can never be written into another's.

### Which project a session belongs to

Every session is placed in a workspace, and tagged with it, in this order:

1. The `X-NVMAI-Workspace` header, when the client sent one.
2. **The working directory the client declared in its system prompt.** Claude
   Code writes a `Working directory:` line and Codex a `<cwd>` element on every
   request, and where they are running is the project. So one server serves a
   novel in `~/novels/photograph` and a codebase in `~/src/nvmai` with two
   separate fact stores and no configuration at all: each conversation's memory
   lands in the project the client is standing in. Only absolute paths in
   *system* messages count, so a user pasting a transcript cannot move their
   memory, and a declared home directory or root falls back to the launch
   workspace rather than becoming one.
3. The launch directory.

The startup log shows the placement per session: `scope=photograph-3f2a9c1e
tag=photograph via=declared-cwd`. `swift run ContinuityDemo inspect` lists
each project's sessions with their tags.

The home directory, its parent and the filesystem root are refused as
workspaces. A server launched from `~` and used for everything would collect a
novel and a codebase into one fact store, and the bootstrap for the codebase
would open with the plot of the novel. With `NVMAI_MEMORY=1` the start script
stops and says so; the server applies the same rule on its own and logs it.
Launch from the project, or name the workspace with `NVMAI_MEMORY_WORKSPACE`.

To look inside one, including while a server is running:

```bash
swift run ContinuityDemo inspect ~/.nvmai/memory/nvmai/$USER/<workspace>.ndjson
```

That read takes no lock. `jq` works on it as well; it is JSON lines.

## Consolidation: the engine writes

The store is only as good as what gets written into it, and measured on a real
model the model-initiated write is the unreliable link. Given a novel's bible
in its prompt, Qwen 3.6 35B made zero writes in that session; in the next it
found memory empty, invented a bible, and stored that — which memory then
carried faithfully for eight sessions. A harness that simply forced a 200-word
summary at every boundary carried twice as much. The forcing is what works.

So the engine forces it. When a session goes quiet for
`NVMAI_MEMORY_CONSOLIDATION_IDLE_SECONDS` (thirty seconds by default), or when a
new conversation starts in the same workspace before that — a rollover, the
end-of-conversation signal the API never sends — the engine asks the model,
in a separate tool-free request, what from that session must not be
contradicted later: decisions, fixed attributes, rules, current state and what
changed. The answer comes back as addressed facts (`characters/marcus`,
`state/inn`, `decisions/storage`) and is written to memory, reusing an existing
key when a fact updates it. That last part is why facts and not a note: a
later state supersedes the earlier one, where the summary baseline copied
`Inn: Standing` forward one session after the inn burned.

It costs one generation per session — a few thousand prompt tokens and a few
hundred out, a minute or so on a 35B model — and it runs only in the pauses:
never inline with a request, and on a rollover only after the new session's
first reply has been returned. With memory on it is on; `NVMAI_MEMORY_CONSOLIDATION=0`
turns it off. It needs no tools at all, which is the point: with
`NVMAI_MEMORY_TOOLS=off` the model pays ~200 prompt tokens for the fragment
and bootstrap, reads what the engine wrote, and never has to decide to write.

The extraction is shown what memory already holds — every key by name in the
namespaces the session touches, the other namespaces as one line with a
count, and current values only for keys the session mentions — and asked for
only what the session added or changed, one fact per key, never a placeholder.
A returned fact whose value memory already holds is not written again. A long
session is distilled incrementally: each consolidation reads only the turns
after the last one it distilled, plus one for context. That wording is not cosmetic. Shown keys alone, the model
re-derived every one of them from sessions that said nothing about them,
wrote `not specified` over a character's eye colour, and confirmed `standing`
for an inn that had burned three sessions earlier. The parser drops
placeholder values, recovers every complete object from an array the output
cap truncated, and a consolidation that yields nothing logs the head of its
raw output so it can be diagnosed.

The server log shows each one: `consolidated session=… turns=… facts=… keys=…`.

### When the tool rounds run out

A model that keeps calling memory tools past `NVMAI_MEMORY_TOOL_ROUNDS` used to
get its preamble returned as the answer — measured, a 31-token "I need to check
the existing memories" where ten chapters should have been. Now its last calls
are answered, it is told the rounds are used up, and it gets one more
generation to answer with what it has. The tools stay in that request so the
prompt prefix does not move; any tool call it makes anyway is dropped.

## Failure behaviour

Memory never fails a completion.

- A journal file that cannot be opened: the session runs in memory only, and
  the prompt tells the model its writes will not persist. Set
  `NVMAI_MEMORY_LOCAL_FALLBACK=0` to run with no memory instead.
- An operation that fails mid-session degrades the same way, once, and logs it.
- A failed write is reported to the model as a tool error. It is never
  reported as success: a model that believes it saved a fact it did not is
  worse than one with no memory.
- A torn final line in a journal, the normal result of a crash, is dropped on
  replay rather than stranding every good record behind it.
- Writes go to the page cache in microseconds; the durability barrier is taken
  a couple of seconds after the last write, once the drive is idle, and forced
  within thirty seconds if writes never stop. Never inline with a request, and
  never on the same moment the expert streamer needs the disk. A process crash
  loses nothing; a power cut loses at most what arrived since the last idle
  moment. `NVMAI_MEMORY_FSYNC=1` makes every write durable inline instead, at
  about 5 ms each.
- The workspace journal is replayed at boot, not on the first request.
- A workspace already held by another server means this one runs without
  persistence and says so, rather than writing into a file someone else owns.

## Security

- Keys and scope components are parsed, not trusted: traversal, separators,
  globs, control characters, empty segments and overlong keys are rejected
  before any backend sees them.
- The model gets logical memory operations, never raw commands, and cannot
  name another workspace.
- Log lines carry operational detail only, never memory contents, which a
  test asserts.
- Nothing in the memory path opens a socket. `NVMAIMemory` and
  `ContinuityCore` have no networking dependency at all, so memory cannot
  reach off the machine and cannot be reached from it.
- Values are capped, results are capped, and index scans are capped.
- Deletion is per key within the current scope. There is no bulk delete.

## Testing

```bash
swift test --filter NVMAIMemoryTests     # store, config, service, durability
swift test --filter ContinuityCoreTests  # the engine underneath it
swift test --filter MemoryBackendTests   # the decorator in the request path
```

The durable backend is tested the same way the reference store is, with the
same contract, plus what the reference store never had to satisfy: an address
mapping that round-trips, and state that survives a restart. Nothing needs a
server, in CI or on a machine, because there is no server.

## Two sessions, worked through

Session 1:

> **User:** We're keeping the weird FooManager because it prevents a race in
> background sync.

The model calls:

```
memory_set(key="decisions/sync/foo-manager",
           value="FooManager is kept deliberately: it prevents a race in
                  background sync. Removing it reintroduces the race.",
           importance=0.9, tags=["sync","concurrency"])
```

Session 2, days later:

> **User:** Can we simplify the sync architecture?

The session starts with a bootstrap naming `decisions/sync/foo-manager`. The
model calls `memory_search("sync architecture")`, reads the decision, and
then checks the repository before proposing anything, because the prompt tells
it retrieved memory is evidence rather than truth.

## Versions

**v1** (tag `memory-v1`, commit `0175e5b`): the in-process store with
engine-driven consolidation, routed keys, the idle durability barrier,
per-project placement and the single-writer lock. Measured at three runs per
arm on pong and the hundred-chapter novel; see the evaluation document.

**v2** (tag `memory-v2`): v1 plus — the bootstrap ranked by the request with
priority namespaces and dependencies; a "changed in the most recent session"
section; reversions flagged as disputed instead of overwritten; the
extraction shown every key by name and values only for keys the session
touches; consolidation after thirty seconds of quiet instead of two minutes;
no RAM ceiling by default; thirty-day retention and a cap on project files;
the store under `<NVMAI>/memory`; and "memory tools" named as such
everywhere. Measured against v1 on the same benchmarks: see
`docs/memory-database-v2-comparison-2026-09-07.md`.

**v3** (tag `memory-v3`): v2 plus — the extraction lists keys by name only
for namespaces the session touches and summarises the rest (the every-key
list was v2's whole extra cost); a fact returned with its current value is
not rewritten; a long session is distilled incrementally, new turns only;
retention expires a project's session log and keeps its facts; the person's
own facts live in a shared workspace shown to every project; and
`nvmai-memory` lists, shows, retires and forgets. Not yet measured against
v2 — benchmarks held.

## Say what is absent, not only what is present

With the tool surface off, the memory fragment used to tell the model it
"has memory that outlives this conversation", name its workspace, and list
what was known — and stop. Measured on the correction scenario, two sessions
of eight were then lost entirely: one model announced it would "retrieve
what I actually know" and emitted a shell command to list a directory, the
other tried to write its decision to a file. Both produced no answer at all.

Told it has something and given no way to reach it, a model goes looking.
The fragment now says plainly that there are no memory tools in this
request, that everything memory has is already above, and that it is kept
between sessions automatically. That one paragraph took the scenario from
60% to 94%, took every lost session back, and made the run a third faster.

## Known limitations

- **Search is lexical.** Filtering by prefix, tags and importance, then
  ranking by where query terms appear, with a key match weighted above a body
  match. There is no embedding index; adding one would be a dependency and an
  index to maintain for a store holding a few hundred short facts.
- **Session identity is derived, not given.** The API is stateless, so a
  session is identified by the first user message plus the workspace. Two
  conversations opening with exactly the same message in one workspace share
  a session.
- **Consolidation is a hook, not a behaviour.** The service can store a
  consolidation, but nothing triggers it: the API has no end-of-conversation
  signal. It is off by default.
- **Streaming shows memory rounds as text.** Content the model produces before
  a memory call is streamed as it happens. The tool calls are hidden; the
  words around them are not.
- **The store is RAM-primary.** A workspace's facts are held in memory and
  journalled to a file; there is no partial load. That is right for a few
  thousand short facts and would be wrong for a million. The session journal
  is different: memory holds a bounded window and the file holds everything,
  so reading further back means reading the file.
- **Search is a bounded scan, not an index.** A query filters by namespace in
  the engine and then ranks at most 2,000 records. That keeps one workspace's
  cost from growing with how much it has stored, which is what the sorted-set
  index did before, but it is still a scan.
- **One process per workspace.** Enforced rather than assumed, but it is a
  real constraint: several servers cannot share one memory the way they could
  share a database.

## A future semantic layer

The interface was shaped so this can be added without changing anything the
model sees. `MemoryQuery` already carries free text; `MemoryRanking` is the
only thing that interprets it. A semantic layer would:

1. Embed each record on write, storing the vector beside it in the same scope.
2. Add a `rank` implementation that scores by cosine similarity and keeps the
   existing prefix, tag and importance filters as pre-filters.
3. Fall back to the lexical ranking when an embedding is unavailable, so the
   store keeps working before the index is built.

The model-facing tools, the key layout and the scope rules would not change.
Doing it well needs an embedding model resident alongside the LLM, which is a
real memory cost on a machine already streaming experts from SSD, so it is
worth measuring against the lexical ranking before adopting.
