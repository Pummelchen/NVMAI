# Choosing a model: the one real decision

NVMAI itself has few choices that can ruin your day. This is the one: which
model you install. Everything else is a dial you can turn back.

Here is the honest version, up front.

**The one-line answer: start with Ornith 1.5 35B-A3B at 4-bit.** It is about
20 GB, it is roughly twice as fast as the 8-bit build, and it is the default
baseline the project's own getting-started guide tells you to use. Choose
8-bit instead when the extra weight fidelity is worth a 37 GB download and
about half the speed. Both are supported, both are good, and you can have both
installed.

The 8-bit build is what NVMAI's installer and library default to, because it
is the higher-quality option. The 4-bit build is what you should *start* with,
unless you already know you want maximum fidelity over speed.

## First, the two words you need

**Parameters** ("35B", "125B") are the rough size of the model's brain.
Bigger usually means more capable and definitely means slower and larger.

**Quantization** ("4-bit", "8-bit") is how much the model's numbers are
compressed.

- **8-bit** keeps more precision. Better fidelity, larger, slower.
- **4-bit** throws away more. Smaller, faster, slightly less accurate.

If you take one thing from this article: **8-bit is for when you care about
the answer, 4-bit is for when you care about the speed and the disk space.**
Both are real, supported choices. Neither is a trap.

## The models NVMAI runs

Only specific models are supported. This is not a general-purpose runner, and
a random model from the internet will not load. What works:

| Model | Sizes | Runs on | Best for |
| --- | --- | --- | --- |
| **Ornith 1.5 35B-A3B** | 4-bit (~20 GB), 8-bit (~37 GB) | GPU | The recommended default. Best all-round, strong at coding and tools. |
| **Qwen 3.6 35B-A3B** | 4-bit (~20 GB), 8-bit (~37 GB) | GPU | A capable general alternative to Ornith. |
| **Qwen-AgentWorld 35B-A3B** | 4-bit (~20 GB), 8-bit (~37 GB) | GPU | Tuned for agent-style work (multi-step tool use). |
| **Qwen 3.8 Flash Next 125B-A6B** | 4-bit (~174 GB), 8-bit (~236 GB) | GPU | The "biggest brain locally" option. Slow, enormous, impressive. |
| **Qwen 3.5 2B** | 4-bit (~1.3 GB), 8-bit (~2 GB) | **CPU** | Fast, light, runs *alongside* a big model. See below. |
| **Qwen 3.5 4B** | 4-bit (~2.7 GB), 8-bit (~4.5 GB) | **CPU** | Same idea, a little more capable. |
| **Qwen 3.5 9B** | 4-bit (~6.4 GB), 8-bit (~9.9 GB) | **CPU** | The largest CPU model. Capable on its own, no graphics chip needed. |

The 125B model's numbers are not a typo, and a large part of them is its
hashed n-gram table — about **95 GiB** on its own, which is well over half of
the 4-bit install. That table is part of how the model works, not waste. Check
your free space twice before starting that download. It is also the slowest:
about 5.5 tokens a second at 4-bit on the project's reference Mac.

## The two smaller CPU models, and why they are interesting

The Qwen 3.5 2B and 4B are a different kind of thing, and worth understanding
because they are the pleasant surprise of this project.

They run on the **processor cores, not the graphics chip**. That means they
do not compete with a big GPU model for the graphics work — and they are
fast: roughly 23 tokens a second at 8-bit alongside a 35B model doing its own
work. One thread costs the big model almost nothing (about 3%, which is
inside the measurement noise), and that single thread is still good for
about 7 tokens a second.

So they are not "a tiny model because you have no RAM". They are **a second,
small worker beside the big one** — the shape a memory feature wants. That is
[Memory that remembers](08-memory-that-remembers.md), and it is the most
interesting idea in the project.

They install with the same command as everything else. The project's own
converter builds them from Qwen's published 16-bit weights and quantizes them
here, then writes the result straight into `models/`:

```bash
tools/install_models.sh qwen35-2b     # or qwen35-4b, qwen35-9b
```

Add `-8bit` for the 8-bit build (`qwen35-2b-8bit`), and note that each one
runs on the processor rather than the graphics chip.

These install exactly like the 35B models do: a `.gturbo` directory with a
manifest and a verification receipt, so `--verify-install` applies to them too.
The only difference left is where they run — on the processor rather than the
graphics chip. Everything you run is quantized from Qwen's own release, never a
third-party repack.

## Which should you pick?

**You want one answer:** Ornith 1.5 35B-A3B, 4-bit. It is the project's
baseline, the quickest useful download at about 20 GB, and the fastest of the
35B builds.

**You care more about fidelity than speed:** Ornith 1.5 35B-A3B, 8-bit. It is
what the installer defaults to, and it is the configuration behind most of the
project's coding and tooling results — at about half the speed and nearly
double the disk.

**Disk space is very tight:** the Qwen 3.5 2B on the CPU, at about 1.3 GB.
Small, quick, and it runs on the processor rather than the graphics chip.

**You want to connect a coding assistant:** a 35B model — 8-bit if you have
the space and want the best results, 4-bit for speed. Both work; see
[Connecting your apps](06-connecting-your-apps.md).

**You have 200+ GB free and want the biggest:** Qwen 3.8 Flash Next 125B-A6B,
4-bit. Expect to wait, and expect ~5 tokens a second.

**You want a companion for a big model, or to experiment with memory:**
Qwen 3.5 2B on the CPU.

## A short word on four things that are off by default

- **MTP (speculative decoding)** exists and is experimental. It is off because
  measured Ornith runs showed no speed benefit, and it requires greedy
  decoding, native context, and the prompt cache switched off — it cannot be
  combined with YaRN. Leave it alone unless you are deliberately
  experimenting.
- **Thinking** is a per-model switch, not a magic quality dial. A model only
  gets the levels its own template defines: Ornith 1.5 and Qwen 3.6 have
  plain off/on, while Qwen 3.8 Flash Next additionally offers low, medium,
  and extra-high effort. See [The dials](05-the-dials.md).
- **KV-cache precision** defaults to 8-bit regardless of the model's own
  quantization. That is intentional and does not need changing.
- **The per-model tuning** — expert cache, prefetch, sampling defaults — is
  read from the install's own profile and clamped to your Mac's memory. It is
  deliberately not something you set in the launcher, because it was measured.

## What you can stop worrying about

You do not need to pick the "right" model to avoid breaking anything. Models
are removable and re-installable; nothing about one install affects another.
Download the 4-bit model, use it, and get the 8-bit one later if you want
the extra fidelity. Your Mac will not be harmed by either.

## Where to go next

Model chosen and installed. Now the settings that actually change your
results → **[The dials](05-the-dials.md)**

Precise sizes and the full capability matrix live in
[Features](https://github.com/Pummelchen/NVMAI/wiki/Features) and
[Getting Started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
on the wiki.

*Sizes and speeds are the project's published figures from a base 8-core M3
with 24 GB, at NVMAI 5.1. Installed sizes vary slightly by model.*
