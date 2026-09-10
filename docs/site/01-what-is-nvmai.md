# What NVMAI is (in plain words)

Welcome to the forum. Let's start with the short version, no jargon.

**NVMAI runs large AI models on your own Mac.** Not a cut-down model, not a
cloud service — the real thing, on the laptop or desktop you already own,
with nothing sent anywhere.

That is a harder trick than it sounds, and how NVMAI pulls it off is the
whole idea.

## The wall that usually stops you

AI models are big files. A capable one is tens of gigabytes. Your Mac has a
fixed amount of memory (RAM) — maybe 16, maybe 24, maybe 64 GB — and macOS
itself, your browser, and your photos all want a share of it.

So the usual advice is: get a smaller model, or rent a bigger machine.

NVMAI takes a third path. It keeps the model on your **SSD** (your internal
drive) and streams the parts it needs into memory while it is answering. The
model's size is then limited by your *free disk space*, not your memory.

The practical result: a 125-billion-parameter model runs on a 24 GB Mac. On
this project's own reference machine — a base 8-core M3 MacBook Pro with
24 GB — the 35B models answer at around **21.7 tokens per second** in 4-bit
and **about 12** in 8-bit, and the 125B model at **5.46**. (A token is roughly
a word-piece; 21 tokens a second is comfortably faster than you read.)

That is the headline. It is genuinely unusual, and it is why NVMAI exists.

## Where your memory goes instead

Streaming does not mean ignoring your RAM — it means *sizing* it.

You give NVMAI a budget for the part of the model it keeps in memory at any
moment, and it stays inside that budget. Give it 4 GB or 8 GB and it holds
the line, so your Mac stays responsive while the model works. This is
covered properly in [Why NVMAI can run models that "don't fit"](09-why-nvmai-runs-big-models.md).

## What it feels like to use

Three ways in, depending on who you are:

- **A Mac app** with a normal window — type, read, adjust a few settings with
  actual controls. Nothing to code. This is the right door for most people.
- **A chat in the Terminal** for a quick single question, if you like that
  sort of thing.
- **A local server** that your existing apps can talk to. This is the one
  programmers get excited about: point a coding assistant at your Mac and it
  uses *your* model, locally. See
  [Connecting your apps](06-connecting-your-apps.md).

All three use the same engine and the same installed model.

## What makes it different

Most projects that run models on a Mac use an existing engine (MLX or
GGUF/llama.cpp). NVMAI built its own:

- **Its own streaming runtime.** The part that reads experts off the SSD and
  keeps memory bounded is the core of the project, not a wrapper around
  someone else's work.
- **Its own Metal kernels.** Metal is Apple's graphics layer; NVMAI uses the
  GPU's full width rather than going through a translation layer.
- **Its own model format.** A converter builds it straight from the original
  published weights, and `.gturbo` installs are verified end to end with a
  receipt — so a model that loads is the model that was promised, byte for
  byte.
- **The Apple Neural Engine for prompt processing.** When an exported support
  file is present, long-document prefill runs on the Neural Engine instead of
  the GPU cores — measured around **2.3× faster** on a 6,000-token prompt. No
  shipped install carries that file yet, so today this is a capability you
  would enable yourself; without it, NVMAI quietly uses the GPU and nothing
  breaks.

There is one more thing, and it might be the most interesting feature here:
NVMAI can give the model **memory that survives the conversation** — so it
remembers your project from one session to the next, with a safeguard so it
cannot quietly overwrite something *you* told it. It is off by default; you
turn it on with one setting. That is
[Memory that remembers](08-memory-that-remembers.md).

## What it is not

Worth saying on the first page, not the tenth:

- **It is text in, text out.** No image or audio understanding.
- **It runs one model at a time.** One model process per Mac is the rule.
- **Your disk is the new budget.** The 4-bit 35B models need about 20 GB
  installed; 8-bit about 37 GB; the 125B model needs roughly 174 GB.
- **The server is local by design.** It only accepts connections from your
  own machine, and it must not be exposed to a network. That is a safety
  feature, not an oversight.
- **The models are ones this project supports**, not anything on the
  internet. See [Choosing a model](04-choosing-a-model.md).

The full list of limits, with no small print, is
[What NVMAI will not do](10-what-nvmai-will-not-do.md). Please read it before
you plan around NVMAI — it will save us both some time.

## Is my Mac good enough?

You need Apple Silicon (M1 or newer), macOS 26 or later, and free SSD
space. That is the honest list; [Getting NVMAI running](02-getting-nvmai-running.md)
walks through it.

Most of the published measurements come from one base M3 with 24 GB, so
that is the configuration this project knows best. Other chips are expected
to work well but have not been measured here to the same depth — the project
says so plainly rather than guessing.

## Where to go next

Ready to try it? → **[Getting NVMAI running on your Mac](02-getting-nvmai-running.md)**

Want the precise engineering version first? → the
[wiki](https://github.com/Pummelchen/NVMAI/wiki) is the professional
reference. This series is the friendly one; the wiki is the exact one.

*NVMAI 5.1 at the time of writing.*
