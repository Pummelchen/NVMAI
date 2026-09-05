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

## Model-facing tools

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

The bootstrap is bounded twice, by record count and by bytes. A test fills a
store with 500 records and asserts session start can never return more than
the limits allow.

## Configuration

Environment variables, which is how the start scripts pass them:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_MEMORY` | `0` | `1` enables memory |
| `NVMAI_MEMORY_DIR` | `~/.nvmai/memory` | Directory holding the journals |
| `NVMAI_MEMORY_FSYNC` | `0` | `1` forces every append to disk |
| `NVMAI_MEMORY_CACHE_MIB` | by machine memory | Store ceiling, as a worst-case item bound |
| `NVMAI_MEMORY_NAMESPACE` | `nvmai` | Deployment namespace |
| `NVMAI_MEMORY_USER` | OS user | User component of the scope |
| `NVMAI_MEMORY_WORKSPACE` | from `NVMAI_WORKSPACE_DIR` | Explicit workspace id |
| `NVMAI_WORKSPACE_DIR` | launch directory | Directory the workspace id derives from |
| `NVMAI_MEMORY_MAX_VALUE_BYTES` | `65536` | Largest single memory |
| `NVMAI_MEMORY_BOOTSTRAP_LIMIT` | `20` | Bootstrap record cap |
| `NVMAI_MEMORY_BOOTSTRAP_BYTES` | `8192` | Bootstrap byte cap |
| `NVMAI_MEMORY_TOOL_ROUNDS` | `4` | Memory rounds serviced per request |
| `NVMAI_MEMORY_TOOLS` | `1` | `0` advertises no tools |
| `NVMAI_MEMORY_LOCAL_FALLBACK` | `1` | `0` disables memory instead of degrading |
| `NVMAI_MEMORY_CONSOLIDATION` | `0` | Session-end consolidation hook |

### Store sizing

The ceiling defaults by machine memory, because the working set is a few
thousand short facts and does not grow with the host:

| Machine memory | Default ceiling |
| --- | ---: |
| Up to 8 GB | 256 MiB |
| Up to 16 GB | 512 MiB |
| More than 16 GB | 1 GiB |

**This RAM is additional.** It is not taken out of `--ram-budget`, which is the
expert cache's own ceiling (default 8 GiB, capped at half of physical memory).
On an 8 GB machine the expert cache gets its 4 GiB and memory brings the total
to 4 GiB + 256 MiB. Sizing the machine means adding the two.

The ceiling covers every open workspace **together**, not each one, so turning
memory on costs the same whether a session touches one repository or five. Over
the ceiling, the least recently used workspace is closed; nothing is lost,
because everything it held is in its journal and touching it again replays it.

Override with `NVMAI_MEMORY_CACHE_MIB`. The ceiling is enforced by counting
actual bytes, and within a workspace it is split between the two stores:

| Store | Share | At the limit |
| --- | ---: | --- |
| Curated facts | 3/4 | Refuses the write |
| Session journal | 1/4 | Drops the oldest sessions from memory |

Facts get the larger share because they are the half that has to be resident:
they are what a session searches and what goes into a prompt. The journal never
enters a prompt and every byte of it is already in the file, so on an 8 GB Mac
resident transcript buys nothing but faster reads of history nobody reads; its
quarter is a window over recent sessions, not a home for them. The behaviours
at the limit differ too: silently dropping a fact the model relies on is the
worse failure, so facts refuse and the caller archives to make room, while the
journal evicts from memory only — the dropped sessions remain in the file.

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
NVMAI_MEMORY=1 tools/start-qwen3.6-8bit.sh
```

The start scripts export the memory environment themselves: the workspace is
the directory you launched from, so two checkouts never share memory, and the
ceiling follows the table above. To place the store elsewhere or name the
workspace explicitly:

```bash
NVMAI_MEMORY=1 NVMAI_MEMORY_DIR=/var/lib/nvmai \
  NVMAI_MEMORY_WORKSPACE=my-project tools/start-ornith-8bit.sh
```

State lives in one file per workspace,
`<dir>/<namespace>/<user>/<workspace>.ndjson`, created owner-readable only
inside an owner-only directory, with a `.lock` sidecar beside it. Deleting a
project's memory is deleting its file, and backing it up is copying it.

A workspace named per request gets its own file too, so one project's memory
can never be written into another's.

To look inside one, including while a server is running:

```bash
swift run ContinuityDemo inspect ~/.nvmai/memory/nvmai/$USER/<workspace>.ndjson
```

That read takes no lock. `jq` works on it as well; it is JSON lines.

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
