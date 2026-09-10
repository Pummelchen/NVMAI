# Long context and the KV cache

Some questions need the model to hold a lot in mind at once: a long document
to summarise, a whole file to review, an hour of notes to reason over.

"Context" is how much the model can hold. This article explains the three
settings that control it, what they cost, and the one limitation worth
knowing before you plan around it.

**If you are just chatting, you can skip this.** The defaults handle ordinary
conversation comfortably.

## Context, in one paragraph

Context is measured in **tokens** (roughly word-pieces). Your prompt and the
model's reply both count against it. When the conversation grows past the
limit, something has to give — older parts fall out of view.

NVMAI's native context goes up to **262,144 tokens** (262K). That is a large
amount: a substantial document, or a long working session, fits comfortably.
Optional **YaRN** scaling extends that to **512K or 1M** tokens.

## The three settings

| Setting | Values | Default | What it changes |
| --- | --- | --- | --- |
| **Context length** | up to 262K native; 512K/1M with YaRN | 262K | How many tokens the model can consider |
| **RoPE scaling** | `none` or `yarn` | `none` | Whether the model extrapolates beyond its native window |
| **KV cache precision** | `4`, `8`, or `16` bits | `8` | How precisely the conversation is stored while in use |

These are **load-time** settings: change them, then reload the model.

## What the KV cache is, and why precision matters

To answer your next question, the model must attend to the conversation so
far. It keeps that as a **KV cache** — the running memory of the current
session.

That cache takes real memory, and it grows as the conversation does. Its
precision is independent of the model's own quantization: an 8-bit model can
use a 4-bit KV cache, and vice versa. That is deliberate — it gives you a
memory dial that does not require reinstalling anything.

For the production Qwen/Ornith setup, full-length attention KV costs
approximately:

| Context | 16-bit KV | 8-bit KV | 4-bit KV |
| ---: | ---: | ---: | ---: |
| 512K | 10.0 GiB | 5.31 GiB | 2.81 GiB |
| 1M | 20.0 GiB | 10.63 GiB | 5.63 GiB |

The cache starts small (8,192 tokens) and grows on demand, so you do not pay
for 1M tokens on a three-line question.

**The practical rule:** if you want the 1M context to fit on a 24 GB Mac,
use 4-bit KV. That is exactly the kind of trade it exists for.

## The honest warnings

Two things to know rather than discover.

**1. YaRN is an extrapolation, and it can affect quality.** Native context is
what the model was trained for. Stretching past it is a mathematical
extension, and answers beyond the native window can be less reliable. Reach
for YaRN when you have a document that genuinely does not fit, not just
because the number is bigger. It is also **not compatible with MTP**
speculative decoding.

**2. Long generation used to be capped, and no longer is.** Earlier versions
refused contexts past the window where their attention path was exact, rather
than quietly approximating. That limit has since been removed: the indexer
that selects which parts of a long context to attend to now runs on both
prompt processing and generation, so long prompts and long answers are both
supported. If you are planning a workflow that leans hard on the far end of
the context range, it is still worth asking on the forum first — but it is no
longer a hard refusal.

## Prompt reuse: why follow-up questions feel fast

There is a feature here you never have to switch on, but you will notice it.

When you ask a follow-up, NVMAI **reuses the conversation state** it already
computed instead of re-processing the whole history. That is why your second
question is much faster than your first — and the effect is dramatic on long
conversations, where re-reading everything would otherwise dominate the wait.

There are two levels:

- **In memory** (the default): fast, and gone when the server stops.
- **On disk** (optional): survives restarts, so a long session can resume
  without re-reading its history. Unloading an idle model discards the
  in-memory cache, so pair `--idle-unload-seconds` with the disk cache if you
  use both.

## When should you actually change any of this?

| Your situation | What to do |
| --- | --- |
| Normal chat and questions | Nothing. Leave it alone. |
| Reviewing a whole file or a long document | Try it as-is first; 262K may well be enough |
| A document that clearly does not fit | Enable YaRN, and use 4-bit KV if memory is tight |
| Long sessions, want instant resumption | Turn on the disk prompt cache |
| Running out of memory | Lower KV precision before touching context length |

## What this is not

**This is not the persistent memory feature.** The KV cache lives and dies
with the conversation. The memory in
[Memory that remembers](08-memory-that-remembers.md) is a different thing
entirely: durable facts that survive between sessions, scoped to your project.
They are unrelated, and confusing them is the most common misunderstanding
about NVMAI.

## Where to go next

Now the feature worth getting excited about →
**[Memory that remembers](08-memory-that-remembers.md)**

Exact YaRN behaviour and every KV option:
[Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
on the wiki.

*Memory figures are computed from the production Qwen/Ornith model topology
and its cache layout at NVMAI 5.1, and match the project's published tables.*
