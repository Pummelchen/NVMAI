# ContinuityCore

State that outlives a context window, for work that outlives a session.

A model with a 200k window and a project that runs for six months has a
problem no window size fixes. The transcript grows without bound, so something
must be dropped on every call; a chat history drops whatever sorts oldest.
That is how a novel loses a character's eye colour by chapter 40 and a
codebase re-litigates a decision that was settled in week two.

ContinuityCore separates the two things a conversation conflates:

- **What happened.** Append-only, engine-authored, complete. Nobody spends a
  token recording it.
- **What is true.** Small, addressed, versioned, and deliberately written.
  This is what goes into a prompt.

It is a Swift package with no dependencies. Not Valkey, not Redis, not SQLite,
not a service. It runs in your process, and the only thing it touches outside
memory is a journal file you hand it.

## The pieces

| Type | What it is |
| --- | --- |
| `SessionLog` | Every event, append-only. Tasks, sessions, prompts, replies, memory writes, assembled contexts. |
| `TaskMemory` | Addressed facts, versioned, with provenance. Writes supersede, never destroy. |
| `ContextAssembler` | Turns accumulated state into the block of text a call actually receives, inside a budget. |
| `Journal` | Optional append-only persistence. RAM stays the source of truth. |
| `ContinuityEngine` | The façade that owns all four and wires them together. |

Everything underneath the engine is public, so an application that outgrows
the façade drives the pieces directly instead of forking the package.

## Five minutes

```swift
import ContinuityCore

let engine = ContinuityEngine(journal: try FileJournal(url: journalURL))
try await engine.start()                    // replays whatever is on disk

let task = try await engine.createTask(title: "Pong",
                                       objective: "Two autoplayers, no human input")
let session = try await engine.beginSession(taskID: task.id, model: "qwen35b")

try await engine.recordUserPrompt(sessionID: session.id, text: prompt)

// What the model should see next time, chosen and budgeted.
let context = try await engine.assembleContext(taskID: task.id,
                                               sessionID: session.id,
                                               focus: prompt)
let reply = try await model.complete(system: context.renderedContext, user: prompt)
try await engine.recordAssistantResponse(sessionID: session.id, text: reply)

// A fact worth keeping, attributed to the session that produced it.
try await engine.remember(sessionID: session.id,
                          namespace: "decision", key: "field_size",
                          value: "800 by 600, first to 11",
                          importance: 0.9)
_ = try await engine.endSession(session.id)
```

## Addressing

A fact lives at `namespace.key`. The namespace is dotted and hierarchical, the
key is not. The engine never interprets either one; it isolates by task and
indexes by address, which is what keeps it generic.

```
decision.storage            constraint.no_network
plot.act2.turn              character.marcus.knows_about_photo
source.hausmann_1998        chapter.041.status
```

Lowercase letters, digits, underscore and hyphen. Segments are separated by
dots, and keys carry none, so an address always splits at the last dot.

## Versions, not overwrites

```swift
try await engine.remember(sessionID: s, namespace: "plot", key: "brother",
                          value: "missing")
try await engine.remember(sessionID: s, namespace: "plot", key: "brother",
                          value: "found in act three")

await engine.history(taskID: task.id, namespace: "plot", key: "brother")
// [v1 "missing" (superseded), v2 "found in act three" (active)]
```

Chapter 100 can still answer what chapter 40 believed, and every version knows
which session and which author wrote it. `.disputed` marks a contradiction
nothing has resolved yet; disputed facts stay visible and are flagged, because
the model is the thing best placed to settle them and cannot if the conflict
is hidden. `.archived` retires a fact without destroying it.

Concurrent editors pass `expectedVersion:` and get a `versionConflict` instead
of silently losing a write.

## The budget

```swift
let budget = ContextBudget(maxTokens: 4000,
                           priorityNamespaces: ["objective", "constraint", "decision"],
                           recentTurnCount: 3,
                           turnShare: 0.3)
```

Priority order first, then importance, then recency. Recent turns get a capped
share, so one long exchange cannot crowd out the durable state. An included
item drags its `dependencies` in with it, so a decision never arrives without
the constraint that produced it.

What comes back names what went in:

```swift
snapshot.memoryItemIDs      // included, in render order
snapshot.memoryVersions     // address -> version, so it can be explained later
snapshot.droppedItemIDs     // matched but did not fit
snapshot.estimatedTokenCount
```

A persistent entry in `droppedItemIDs` means the budget or the priorities are
wrong. That is a signal worth logging.

## Streaming

```swift
let response = try await engine.beginAssistantResponse(sessionID: session.id)
for try await chunk in stream {
    try await engine.appendAssistantChunk(responseID: response, text: chunk)
}
try await engine.completeAssistantResponse(responseID: response,
                                           outputTokens: 812, finishReason: "stop")
```

Chunks are buffered and the reply lands as one event, so nothing has to
deduplicate it downstream. Set `SessionLogOptions.persistsChunks` if a partial
reply must survive a crash mid-stream; readers fold chunks away once the
completion exists.

## Persistence

`FileJournal` writes JSON lines, owner-readable only, and replays on `start()`.
A torn final line is dropped rather than stranding every good record behind it.
`compactJournal()` collapses the file to a single checkpoint, written to a
temporary file and renamed into place, so a crash during compaction leaves the
old journal intact.

**One writer per workspace.** A journal takes an exclusive advisory lock for as
long as it is open. Without it, two processes on one file would each hold their
own copy of the state, see none of the other's writes, and interleave their
appends into something that replays as a braid of two histories. Opening a
locked journal throws `JournalError.locked`; the right response is to run
without persistence and say so, never to write anyway.

The lock lives on a `.lock` sidecar so compaction can replace the journal
without giving it up, and the kernel releases it however the process exits, so
a crash never leaves a workspace that cannot be reopened. To hand a workspace
over deliberately, call `await engine.shutDown()` — waiting for deallocation is
not a contract anyone can rely on.

Durability is a barrier at each session boundary plus every 64 records, using
`F_FULLFSYNC` rather than `fsync`, which on Darwin only promises the write
reached the drive's cache. `synchronizesEveryWrite` makes every append durable
for callers who want that, and `flush()` takes the barrier on demand. A session
boundary costs about 5 ms.

To read a journal without disturbing a running server, `FileJournal.read(contentsOf:)`
takes no lock. `swift run ContinuityDemo inspect <file>` prints what is in one.

## Storage budgets

Both stores are bounded by counted bytes, not by an item count multiplied by a
worst case:

```swift
MemoryLimits(maxValueBytes: 16 << 10, maxBytesPerTask: 192 << 20)
SessionLogOptions(maxBytesPerTask: 64 << 20)
```

They behave differently at the limit, on purpose. Facts **refuse** a write with
`ContinuityError.storeFull`: silently dropping something the model deliberately
wrote is worse than declining to add one, and the caller can archive to make
room. The log **evicts**, dropping the oldest whole sessions from memory —
they stay in the journal file, so this bounds what the process holds rather
than what was recorded. The session currently in progress is never evicted.

The defaults lean toward facts because facts are the half that has to be
resident: they are what gets searched and put in a prompt. The log never enters
a prompt and is on disk regardless, so its share is a window over recent
sessions, not a home for them.

`memory.utilization(taskID:)` and `statistics()` report where a task stands.

The session log holds complete prompts and replies. Treat it as sensitive
application data:

- Nothing in this package opens a socket. There is no telemetry and no upload.
- The journal and its lock are created `0600`, inside a `0700` directory.
- `ContinuityConfiguration.journalsSessionContent = false` keeps prompts and
  replies out of the file while still persisting memory.
- `forget(taskID:)` deletes a task, its sessions, its log and its memory, then
  compacts, because a delete that leaves the content in an append-only file is
  not a delete.

## Two shapes of work

**A long coding project.** Namespaces of `objective`, `decision`,
`constraint`, `gotcha`, `state`. Priority order puts constraints above
decisions above notes. When the model changes its mind, the old decision is
superseded and still explains the code written under it.

**A hundred-chapter novel.** Namespaces of `character`, `plot`, `setting`,
`style`, `chapter`. Facts carry dependencies, so pulling in a scene pulls in
the character constraints that scene has to respect. `.disputed` catches the
moment two chapters disagree about someone's age, which is the error that
otherwise ships.

Both use the same engine. The difference is the namespaces and the budget, not
the code.

Both are worked end to end in `sources/ContinuityDemo`. Run them:

```bash
swift run ContinuityDemo         # both, plus a scale check
swift run ContinuityDemo novel   # or coding, or diagnose
```
