# The dials: what each setting actually does

This is the article to come back to. It explains the settings you will
actually meet, in plain terms, and says which ones you can safely ignore.

**The first rule: start with the defaults.** They were chosen from
measurements, not guesses. **The second rule: change one thing at a time.**
If you change three dials and the answers get worse, you have learned nothing.

That is not a style preference. The project itself measured that run-to-run
variation can be around ±15%, so a single before/after comparison can easily
fool you. One change at a time is the only way to know what did it.

## Answers: the sampling dials

These decide *how the model picks the next word*. They are the ones people
fiddle with most, and mostly should not.

| Setting | Default | What it really does |
| --- | --- | --- |
| **Temperature** | 0.6 (1.0 on Qwen 3.8 Flash Next) | Lower = more literal and predictable. Higher = more varied and creative. **0 = always pick the single most likely word** — fully repeatable. |
| **Top-K** | 20 | Only consider the 20 best candidate words at each step. `0` turns the limit off. |
| **Top-P** | 0.95 | Consider candidates until they add up to 95% of the probability. A gentler cousin of Top-K. |
| **Max new tokens** | 1,024 in the CLI; the rest of the context in the app and server | The longest answer allowed. It stops early when finished. |
| **Stop text** | none | Stop as soon as this text appears. Useful for automation. |
| **Seed** | off | Fixes the randomness so the same input gives the same output. |
| **Repetition penalty** | 1.0 (off) | Discourages repeating itself. `1.0` means no penalty. |

Each model family carries its own sampling defaults — Qwen 3.8 Flash Next, for
instance, starts at temperature 1.0 rather than 0.6 — and smaller models use
their own. The values above are what the Ornith and Qwen 3.6 families use.

**Practical advice.** For a factual question or code, use a low temperature or
`0`. For brainstorming and writing, 0.6 is already a good middle. If answers
feel wild, turn the temperature down before touching anything else — it is
almost always the culprit. If the model repeats itself, a small repetition
penalty helps.

One honest note, because you may see it in an API client: the OpenAI-style
`presence_penalty` is only accepted as `0.0`. Send any other value and the
server rejects the request with a `400` and a clear message, rather than
silently ignoring it. Do not build a workflow around it.

## Thinking: a real switch, per model

"Thinking" means the model reasons at length before it gives its answer. It
helps on genuinely hard problems and wastes time on simple ones.

NVMAI exposes only what each model's own template implements:

- **Ornith 1.5 and Qwen 3.6** have a plain **off/on** switch. They do not
  define effort levels, and NVMAI will not pretend otherwise.
- **Qwen 3.8 Flash Next** also supports **low**, **medium**, and
  **extra high** effort. Its template's own default is extra high.

That restraint is a design choice worth respecting: a "high effort" setting
that changes nothing about the prompt changes nothing about the answer
either. If your client offers you effort levels on a model that has none,
you are getting a relabelled on/off switch.

**Practical advice.** Leave thinking off for chat, and on for the hard 5% of
questions. On Qwen 3.8, start at its default effort rather than jumping to
maximum.

## Concise mode

By default the model answers as a chat assistant: a little preamble, the
answer, then often a closing summary. **Concise mode** replaces that with a
system prompt asking for the answer without the padding.

Standard responses remain the default because, in the project's coding and
tooling qualification, they generalised more reliably. Concise is for when
you want the answer and nothing around it — scripting, quick lookups, or when
you are reading many replies in a row.

## Memory and context: the two "how much can it hold" dials

People mix these up constantly. They are unrelated.

| | **Context length** | **KV cache precision** |
| --- | --- | --- |
| What it is | How many tokens of conversation the model can consider at once | How precisely that conversation is stored while it is being considered |
| Setting | `--max-context` | `--kv-bits` (`4`, `8`, or `16`) |
| Default | Native up to 262,144; 1M with YaRN | 8-bit |
| Costs | Much more memory as it grows | Memory, and a little accuracy at 4-bit |

The KV cache is the model's *working* memory of the current conversation: it
grows as you talk and is discarded when you stop. It is **not** the same as
the persistent memory in [Memory that remembers](08-memory-that-remembers.md),
which survives between sessions.

**Practical advice.** Leave both alone at first. If you routinely feed in very
long documents and run out of memory, that is when you look at 4-bit KV or
YaRN — [Long context and the KV cache](07-long-context.md) covers it properly.

## Speed and memory: the expert-cache dials

These are the dials specific to what makes NVMAI unusual.

| Setting | Default | What it does |
| --- | --- | --- |
| **RAM limit** (`--ram` / `--ram-budget`) | 10–12 GiB, per model and width | The ceiling for the expert cache — the slice of the model kept in memory. |
| **Expert cache slots** (`--expert-cache-slots`) | derived from the budget | Same idea expressed as a count. Overrides the budget. |
| **Prefill chunk** (`--prefill-chunk`) | 4096 for the supported text models | How much of your prompt is processed per step. Larger uses more temporary memory but can reduce repeated disk reads. |

### The RAM limit, which the launcher asks you about

This is the one dial the launcher offers directly, because it is the one that
decides whether your Mac stays comfortable. It asks for a limit in
**1, 2, 4, 8, 16 or 32 GB**, and you can also pass it:

```bash
tools/server_launcher.sh --client server --model ornith 4 --ram 8
```

The default — the seventh choice, "model default" — is the install's own
measured profile: 10–12 GiB, tuned per model and weight width and clamped to
half your Mac's physical memory. That is the fastest setting, and the one most
people should keep.

Pick a tier deliberately when:

- **Your Mac feels sluggish while it generates.** Go down a tier or two. Fewer
  experts held in memory means more SSD reads and slower answers, and a machine
  that still responds to you — usually the right trade.
- **You have RAM to spare and want the fastest answers.** Go up a tier.

A CPU model has no expert cache at all, so the limit does not apply to it; the
launcher says so rather than pretending the choice did something.

**Practical advice.** Leave it on the model default. Change it only if you are
short on memory, and change one tier at a time.

## The performance features that are already on

You do not need to enable these; they are on by default, and knowing what
they are explains the speed you see.

- **Prompt reuse** remembers the state of a conversation so a follow-up does
  not re-read the whole history. This is why your second question in a chat
  feels much faster than the first.
- **Predictive expert prefetch** reads the experts the next layer will
  probably want *while* the current layer is still working. It is on or off
  per model and weight width — 35B models have it on, the 125B model does
  not — and that choice comes from measurement, not intuition.
- **Tiled Top-K sampling** is just the fast way NVMAI picks the next word.
  It cut that step from 15.5 ms to 1.4 ms per token, with an identical
  output stream. You never see it; you benefit from it.
- **Neural Engine prefill** routes the full-attention part of prompt
  processing through the Apple Neural Engine — measured around **2.3× faster**
  than the GPU cores on a 6,000-token prompt. It needs an exported Core ML
  support file next to the model, and **no shipped install includes one yet**,
  so in practice you are on the GPU path until you export it yourself. When
  the file is missing NVMAI falls back silently, so there is nothing to
  configure and nothing to break. The project still calls this experimental
  because its output is not byte-identical to the GPU path.

## Two load-time settings you may want later

Both are off by default and exist for specific situations:

- **`--lazy-load`** opens the server port immediately and defers loading the
  model until the first request. Handy when you want the API up but are not
  ready to spend the memory yet.
- **`--idle-unload-seconds <n>`** releases the model after `n` seconds with no
  requests, and reloads it transparently on the next one. This is the polite
  option if you keep NVMAI running all day on a busy Mac. Pair it with a disk
  prompt cache, because unloading discards the in-memory one.

## What you should probably not touch

- **The per-model tuning** — expert cache, prefetch depth, sampling defaults.
  The runtime reads these from the install's own measured profile. The
  launcher deliberately does not override them.
- **The experimental switches** (`NVMAI_EXPERT_IO_BACKEND=metal`,
  `NVMAI_SAMPLER_PATH=generic`, and friends). These exist as comparison paths
  for benchmarking — the sampler one is the slower control arm, and the Metal
  I/O backend is incomplete. None of them is a speed-up you are missing out
  on, and there is no measured case for leaving them on.

The complete list of every environment variable and flag, with the
measurement behind each default, is
[Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
on the wiki. This article is the map; that page is the territory.

## Where to go next

Now let's point your own apps at it → **[Connecting your apps](06-connecting-your-apps.md)**

*Defaults are from NVMAI 5.1. The ±15% measurement note and the prefill figure
come from the project's own benchmark notes on a base 8-core M3 with 24 GB.*
