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
                            ├─ ValkeyMemoryStore ── RESP client ── Valkey
                            └─ InMemoryStore (fallback, and the test double)
```

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

- **namespace** separates deployments sharing one Valkey (`nvmai` by default).
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
| `VALKEY_URL` | `redis://127.0.0.1:6379` | Server, optionally `user:pass@host:port/db` |
| `VALKEY_HOST`, `VALKEY_PORT` | | Override the URL's host and port |
| `VALKEY_USERNAME`, `VALKEY_PASSWORD` | | Credentials, never logged |
| `VALKEY_DB` | `0` | Database index |
| `NVMAI_MEMORY_CACHE_MIB` | by machine memory | Valkey `maxmemory` applied at connect |
| `NVMAI_MEMORY_TIMEOUT_MS` | `250` | Per-operation deadline |
| `NVMAI_MEMORY_CONNECT_TIMEOUT_MS` | `1000` | Connect deadline |
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

### Cache sizing

The Valkey ceiling defaults by machine memory, because the working set is a
few thousand short facts and does not grow with the host:

| Machine memory | Default cache |
| --- | ---: |
| Up to 8 GB | 256 MiB |
| Up to 16 GB | 512 MiB |
| More than 16 GB | 1 GiB |

Override with `NVMAI_MEMORY_CACHE_MIB`. NVMAI applies the ceiling and
`noeviction` at connect: durable memory should refuse writes when full rather
than quietly drop the facts the model relies on.

## Setup

### macOS

```bash
brew install valkey
brew services start valkey
```

### Linux

```bash
sudo apt install valkey-server   # or: dnf install valkey
sudo systemctl enable --now valkey
```

### Either, with Docker

```bash
docker compose -f tools/memory/docker-compose.yml up -d
```

That compose file is RAM-first with persistence: appendonly on, periodic
snapshots, `noeviction`, bound to loopback.

### Running the server with memory

```bash
NVMAI_MEMORY=1 tools/start-qwen3.6-8bit.sh
```

The start scripts export the memory environment themselves: the workspace is
the directory you launched from, and the cache ceiling follows the table
above. To point at a different server or workspace:

```bash
NVMAI_MEMORY=1 VALKEY_URL=redis://127.0.0.1:6379 \
  NVMAI_MEMORY_WORKSPACE=my-project tools/start-ornith-8bit.sh
```

## Failure behaviour

Memory never fails a completion.

- Valkey unreachable at session start: the session runs on a process-local
  store, and the prompt tells the model its writes will not persist. Set
  `NVMAI_MEMORY_LOCAL_FALLBACK=0` to run with no memory instead.
- An operation that fails mid-session degrades the same way, once, and logs it.
- A failed write is reported to the model as a tool error. It is never
  reported as success: a model that believes it saved a fact it did not is
  worse than one with no memory.
- A timeout closes the connection rather than risking a reply being matched to
  the next command.

## Security

- Keys and scope components are parsed, not trusted: traversal, separators,
  globs, control characters, empty segments and overlong keys are rejected
  before any backend sees them.
- The model gets logical memory operations, never raw commands, and cannot
  name another workspace.
- Credentials are never logged and never reach the model. Log lines carry
  operational detail only, never memory contents, which a test asserts.
- Values are capped, results are capped, and index scans are capped.
- Deletion is per key within the current scope. There is no bulk delete.

## Testing

```bash
swift test --filter NVMAIMemoryTests     # store, Valkey wire, config, service
swift test --filter MemoryBackendTests   # the decorator in the request path
```

The Valkey suite runs against an in-process fake that speaks real RESP over a
real socket, so framing, pipelining, error replies, timeouts and the config
handshake are covered with no server on the machine or in CI. The same
conformance rules are asserted against both backends.

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
- **One connection.** Pipelined and shared, which is ample for memory-sized
  traffic, but a very large store searched constantly would want a pool.

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
