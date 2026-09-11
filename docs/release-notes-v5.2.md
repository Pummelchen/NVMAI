## NVMAI 5.2 — one server for every model, the dense family on both engines, and memory that outlives a conversation

This is the largest NVMAI release so far. Three changes alter what you can do
with it, and a fourth alters what it will do quietly:

- **One server serves every installed model.** The launcher starts it with the
  client of your choice, and any other installed model stays reachable by name
  on the same port — one resident at a time, switched on demand.
- **The dense Qwen 3.5 models run on the GPU as well as the CPU**, and the
  engine is a per-request choice for them.
- **Optional agent memory**, inside the server process: durable facts scoped to
  the repository you are working in, with no database to install.
- **Thinking is not the answer.** A model's reasoning reaches clients as
  `reasoning_content`, apart from `content`, including the case where a model
  thinks although the request turned thinking off.

Everything below is in the tagged commit; the checks that back it are named as
they come up.

### One server, every model, one launcher

`NVMAIServer --models-dir models` serves the whole catalogue: `/v1/models`
lists every installed model and quantization, and a request naming another one
unloads the resident model and loads that one. The launcher
(`tools/server_launcher.sh`) asks what to launch (the API alone, or the API
plus Codex, Claude Code, Qwen Code, OpenCode or Zed), whether to keep the
coding-CLI boilerplate (`full` or `fast`), which model and quantization, the
answer style, the thinking level, and — only for a model that has one — the
RAM limit for the expert cache. Its model list comes from the server's own
catalogue, so it cannot disagree with what the server serves.

Retired with it: the eight per-model start scripts and the separate
client-wiring launcher. Both are replaced by this one entry point, and the
launcher reads the served model id from the running server rather than
assuming it.

### The dense Qwen 3.5 family runs on both engines

The 2B, 4B and 9B dense models (4- and 8-bit) were CPU-only in 5.1 because the
GPU runtime refused their family (`qwen3_5_dense`) by name: the decode and
prefill pipelines encoded the router, the prefetch probes, the residency
classification and the streamed routed FFN unconditionally, which is the
silent-fluent-nonsense class of failure this project has shipped once before.
That stage is now conditional, and a dense model skips it entirely: its
`gate/up/down` FFN runs through the block that already implements a shared
expert, the tensor schema is the family's own, and the layer conventions come
from the manifest's `arch` block rather than from a guess.

The engine is now selectable **per request** for these installs: the bare
catalogue id is the GPU spelling, and `<id>@cpu` / `<id>@gpu` name an engine
explicitly, so a client can choose without restarting anything. Residency is
still one model at a time, so switching engines reloads the install.

The port's acceptance bar was numerical, not "it loads": logits equivalent to
the CPU engine's on the real install, layer by layer, before the refusal was
lifted. That gate paid for itself — it found that the GPU path read
`k_proj`/`v_proj` through the attention slot's 4-bit kernel while the install
stores them at 8 bits, which is a plausible-looking wrong attention output
rather than an error. Widths now resolve per tensor stem, and the runtime
builds one affine dispatcher per width a role needs. Three further
assumptions that had never run without a routed mixture (a manifest that
demanded packed expert files for every layer, a layout validator that required
at least one expert, and range guards that trapped on top-0) were fixed with
it.

### Memory that outlives a conversation (optional)

`NVMAI_MEMORY=1` gives a model durable facts it writes in one session and reads
in another, scoped to the repository the client is standing in. It runs inside
the server process — no database, no port, no connection to lose — on a
continuity engine that journals one file per workspace. Six memory tools are
answered by the engine itself, and the extraction that writes facts marks
whether the *person* asserted something or the model inferred it.

That distinction is enforced: the guard (`NVMAI_MEMORY_GUARD=1`, on by default
wherever memory is on) stops a model-derived fact from silently superseding one
the person asserted. Measured by replaying a recorded long session, an
unguarded store answered 61% of the questions the recording supports and a
guarded one 98%; on a control session the guard scored 97%, one point off the
unguarded control, so it is not buying safety with silence. It is off by
default overall and the serving path does not depend on it; see
`docs/agent-memory.md`.

The Qwen 3.5 2B was measured as a resident helper that would propose and check
facts. It is **not** wired in: one decision per call with a closed answer set
scored 92%, but asked to decompose a long fact in one shot it echoed its input
in 38 of 47 answers, and one CPU thread while a 35B generates costs that 35B 3%.
The measurement is recorded so the design stays honest about what a 2B can be
trusted with.

### Thinking is not the answer

A model's reasoning now reaches clients as `reasoning_content`, apart from
`content`, on every surface: a client that knows the field shows the thought
apart from the answer, and one that does not sees the answer alone. The
reasoning levels a model offers are the levels its template actually renders —
the binary switch for Ornith, Qwen 3.6, Qwen-AgentWorld and the dense Qwen 3.5
models; `off`, `low`, `medium`, `xhigh` for Qwen3.8-Flash-Next — and a level a
model does not define is refused rather than mapped to a neighbour.

Two defects were found here and fixed. A model's `reasoning_effort` change was
an HTTP 500 on every installed CPU model, because both re-render paths handed
the tokenizer loader a model directory where a `.gturbo` install keeps a
`tokenizer/` sidecar. And a thought the *model* opens while thinking is off was
streamed as the answer: Qwen AgentWorld 35B-A3B 8-bit, asked a false-premise
question with the switch off, reopens a `<think>` block and spends the whole
token budget inside it, so a client capping `max_tokens` received a thinking
transcript where it expected an answer. The thought is now split into
`reasoning_content` on every engine, and the server logs
`thinking off, but the model wrote N characters of reasoning` on the request's
line. The measurements, with every reply verbatim: the wiki's
[One Prompt, Every Model](https://github.com/Pummelchen/NVMAI/wiki/Capital-of-Paris-Smartness)
page.

### The protocols coding agents actually speak

One server speaks OpenAI Chat Completions, the **OpenAI Responses API** in full
(stored responses, `previous_response_id`, the complete event grammar) and the
**Anthropic Messages API** (`/v1/messages`, `count_tokens`, streaming). All
three were exercised against the real Codex and Claude Code CLIs, not only
against test doubles, and the launcher wires a chosen client to the model the
server advertises. Reasoning levels a client requests are mapped to what the
served model renders, and the mapping is logged rather than left to be
inferred.

### The deep audit: 89 code findings, 8 documentation defects, none open

Seven read-only passes over every module, then verification of each finding
against the source before any fix — a subagent's word is not evidence. The
register is `docs/audit-2026-09-11-findings.md`: **89 code findings and 8
documentation defects resolved, 0 open, 3 disproved** (and recorded as
disproved rather than deleted).

The worst were the ones that were silent rather than loud: an out-of-bounds
router write on Qwen3.8-Flash-Next that did not fault only because driver
allocations are page-granular; a prompt-cache restore that left the sparse
indexer holding the *previous* conversation's pooled keys, which is silently
wrong output on a pinned model; a fused hyper-connection read kernel that could
never be built, so every measurement of that path had measured the unfused one;
and unbounded request headers on a loopback server, with an oversized-body 413
that arrived only after the whole body had been read.

### Structure, naming, and the Swift 6.3 baseline

`HTTPServer.swift` (2,599 lines: an actor, a 2,111-line handler and five
support types) is eight files now, the largest 604 lines; `Model.swift` split
into the model and its loading path; three benchmark families that shared one
1,153-line `main.swift` are separate files; and two files named `main.swift`
while holding `@main` are named after their type. The test tree mirrors the
source tree. The compiler warning count is zero, and a `defer` that deleted a
demo's scratch directory before the demo wrote to it is one of the warnings
that had been learned past. The conventions are written down in
`docs/repository-layout.md`.

The tree is Swift 6.3.3, tools format 6.3, Swift 6 language mode, and now also
enforces the three upcoming features it was already clean under:
`InferIsolatedConformances`, `ImmutableWeakCaptures` and
`MemberImportVisibility` (which needed direct imports in twelve files). The
three features with a real migration cost — `ExistentialAny`,
`InternalImportsByDefault`, `NonisolatedNonsendingByDefault` — are recorded as
deliberate, with their measured cost, in `docs/swift-language-standard.md`.

### Also in this release

- **Watchdogs, opt-in**: `NVMAI_WATCHDOGS=1` watches four failure modes — a
  repetition loop, a stall, a stub answer and a ping-pong turn — and
  `NVMAI_WATCHDOG_ACT=<kinds>` names the ones that may stop a generation
  instead of only reporting it. Off by default.
- **A whole-model CPU verifier**: `tools/verify_cpu_models.sh` runs the dense
  installs end to end through the CPU engine (`all continuations correct`), and
  the bench commands that did this had been pinned to a conversion
  intermediate that no longer exists.
- **The dense installs are `.gturbo`** like everything else, with manifest and
  path-bound receipt, and the repacker's output is verified against the
  snapshot it came from rather than assumed equivalent.
- **`--engine cpu|gpu`** in the launcher states which engine serves an install;
  the catalogue shows each install's engine and thinking levels, and a model
  that cannot render a value is reported rather than silently replaced.
- **A manifest now has to describe its payload**, not only parse: the verifier
  re-hashes what it finds and refuses an install whose manifest and files
  disagree.

### Performance

This release is not a decode-rate release: it changes which engine can serve a
family, adds a memory subsystem that is off by default, and fixes a
reasoning-split defect. The numbers measured on the release commit are on the
wiki's
[One Prompt, Every Model](https://github.com/Pummelchen/NVMAI/wiki/Capital-of-Paris-Smartness)
page — every model and both engines, three repeats each, with time to first
token and decode rate. For scale: on this base M3 with 24 GB, Ornith 1.5
35B-A3B decoded at 19.4 tok/s (12.6–22.8) at 4-bit and 10.8 (8.6–12.1) at
8-bit, Qwen 3.6 35B-A3B at 19.5 / 11.4, Qwen-AgentWorld 35B-A3B at 23.1 /
10.9, and Qwen3.8-Flash-Next 125B-A6B at 3.6 / 2.4.

The 512-token story benchmark on the wiki's
[Benchmarks](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks) page was
measured at 5.1 and has **not** been re-run for 5.2; the ranges above are
short-generation measurements on a different protocol and are not comparable
with that table row for row.

### Checksum

`nvmai-5.2-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
