> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** agent-memory.md (the design doc), docs/plan-memory-guard-and-shadow.md

# Agent memory: memory that outlives the conversation

Off by default, no database to install, and it runs *inside* the server process. When you turn it on, a model writes durable facts in one session and reads them in another, scoped to the repository it's working in. This is what it is, and — because it's an opt-in that touches what the model believes — what it protects against.

## The problem

A coding agent rediscovers the same things every session. Why an odd class survives. Which refactor was tried and abandoned. What the build actually does on *this* machine. None of it belongs in a prompt — it's unbounded and mostly irrelevant to any one question. It belongs in a **store the model can query**.

> **It is not the KV cache.** The KV cache is per-conversation attention state. Agent memory is durable, cross-session, and survives the process. The serving path does not depend on it: with memory off, none of the code below is even constructed.

## How it's built (the short version)

```
NVMAIServer ── MemoryBackend (decorator) ── inner backend (inference)
                     │
                     ├─ installs a short system fragment + the memory tools
                     ├─ services the memory_* calls the model makes
                     └─ MemoryService → ContinuityEngine → one journal file per workspace
```

- `MemoryBackend` **wraps** any inference backend. On the way in it installs a ~200-word system fragment and the memory tool definitions; on the way out it services the memory calls the model made and asks the inner backend to continue. No database, no port, no connection to lose — the engine is a Swift actor in the same binary as the model, and the only thing it touches outside memory is one journal file per workspace.
- **Memory is repo-scoped** as `<namespace> / <user> / <workspace>` — namespace separates deployments on one machine, user separates people on one server, and the workspace is the repository (directory name plus a digest of its full path, so two checkouts of one repo never share memory). The scope comes from the *session*, not from the tool call, so a model cannot name another workspace and redirect a write.
- **Bounded by design.** An index keeps listing/search/bootstrap from ever scanning the whole store; nothing issues `KEYS` or `SCAN`. A single server can serve several checkouts (send `X-NVMAI-Workspace`, or pin it to one).

## The six tools

| Tool | Purpose |
| --- | --- |
| `memory_search` | Find memories by text, prefix, tags, or importance. |
| `memory_get` | Read one memory by exact key. |
| `memory_list` | List keys, optionally under a prefix. |
| `memory_set` | Write or replace a memory. |
| `memory_append` | Add a line to an existing memory. |
| `memory_delete` | Remove a memory that's wrong or obsolete. |

The model never receives a database command — it gets these six functions and nothing lower. (These are *memory* tools, not the client's tools: the engine services them itself, because no coding CLI would know how to run a memory tool. Client tools still pass through untouched.)

## What the model sees

A ~200-word system fragment says memory exists, when to read, when to write, what *not* to store, and that retrieved memory may be stale and worth verifying. It lists **bootstrap keys with one-line summaries, never full values** — the bootstrap says what exists; the text is a tool call away. The bootstrap is bounded twice (by record count and by bytes; 40 records / 16 KiB by default), so session start can't return more than the limits allow.

## The guard: the part worth understanding

The risk with any model-written memory is that the model *corrects* a fact the **person** actually asserted — and silently wins. `NVMAI_MEMORY_GUARD` (on by default when memory is on) stops exactly that: a write the consolidation attributed to the *model* does **not** overwrite a live fact the *person* asserted. The person's value stays, the address is marked **disputed**, and both values are visible to the next session. The person always supersedes their own facts; model-over-model is untouched; a model write that merely *agrees* is stored, not held.

It's a measured default, not a hopeful one. On Ornith 1.5 4-bit, the book scenario is the one place memory has ever lost (98% without the guard, 94% with, because memory faithfully preserves that model's drift from the story bible). **With the guard on it scores 97%** — one point off the control and inside the noise — and it fired exactly once in the whole run, holding the bible's hard rule the model tried to overwrite. One hold, three points.

## Turning it on

Environment variables, which is how the start scripts pass them. The load-bearing ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVMAI_MEMORY` | `0` | `1` enables memory. |
| `NVMAI_MEMORY_DIR` | `<NVMAI>/memory` | Where the project files live. |
| `NVMAI_MEMORY_TOOLS` | `off` | The six functions: `off`, `minimal` (set/get/list), or `full`. |
| `NVMAI_MEMORY_RETENTION_DAYS` | `30` | Delete a project file untouched this long; `0` keeps all. |
| `NVMAI_MEMORY_GUARD` | `1` | Stop a model fact from silently superseding a person-asserted one. |
| `NVMAI_MEMORY_CONSOLIDATION` | `1` | `0` disables the engine writing memory at session boundaries. |
| `NVMAI_WORKSPACE_DIR` | launch directory | Derives the workspace id. |

The rest (cache ceiling, fsync, max workspaces, bootstrap caps, tool-round budget, local fallback) are in the design doc. The default shape to reason about: memory on, guard on, full tools, consolidation on, 30-day retention.

## Where to go next

- The server that hosts memory: [The OpenAI-compatible server](#06)
- The side-engine it was built to run: the in-repo side-engine design (Qwen3.5-2B on CPU beside the 35B)

*Version at time of writing: NVMAI 5.1.*
