> Working plan for the NVMAI community forum at https://nvmai.discourse.group/
> Source of truth for the articles in this folder. The wiki stays the
> professional reference; these articles are the friendly front door.
> Status: drafted, ready to publish article by article.

# NVMAI forum — the article series

This folder holds the ten articles that introduce NVMAI on the community
forum, plus the welcome text below. Each file becomes one forum topic.

## Why this series exists

The [GitHub wiki](https://github.com/Pummelchen/NVMAI/wiki) is the precise,
engineering-facing documentation: exact flags, measured numbers, and the
limits behind them. It is the right place to look something up.

It is not the right place to *land*. So the forum gets a series written for
someone who has never opened a terminal, never heard of quantization, and
just wants a large AI model running on the Mac they already own. The wiki is
the reference; these articles are the explanation. Every article links back
into the wiki for the precise version of whatever it simplified.

## Two things to keep in mind about the audience

Roughly half of NVMAI's users are not programmers. They should be able to
read the whole series without compiling anything, and the one unavoidable
setup step is stated plainly rather than hidden (article 02). Once NVMAI is
running, the Mac app needs no code at all.

The other half *are* programmers, and they will notice if a number or a
limit is glossed over. "Not a coder" is not "not paying attention" — the
articles keep the real numbers, and every limit gets the same space as the
feature it belongs to. A limit that surprises someone later is a support
ticket, and worse, it is a broken promise.

## The articles

Part 1 and part 2 are the how-to: getting started and the choices you make
while using it. Part 3 is about NVMAI itself — what it is, why it is built
the way it is, and what it will not do.

| # | File | Title | Part | Wiki reference |
| --- | --- | --- | --- | --- |
| — | (welcome, below) | Welcome to the NVMAI forum | Start here | [Home](https://github.com/Pummelchen/NVMAI/wiki) |
| 01 | `01-what-is-nvmai.md` | What NVMAI is (in plain words) | About | [Home](https://github.com/Pummelchen/NVMAI/wiki/Home) · [Features](https://github.com/Pummelchen/NVMAI/wiki/Features) |
| 02 | `02-getting-nvmai-running.md` | Getting NVMAI running on your Mac | Getting started | [Getting Started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started) |
| 03 | `03-your-first-conversation.md` | Your first conversation | Getting started | [Getting Started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started) |
| 04 | `04-choosing-a-model.md` | Choosing a model: the one real decision | Features | [Features](https://github.com/Pummelchen/NVMAI/wiki/Features) |
| 05 | `05-the-dials.md` | The dials: what each setting actually does | Features | [Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls) |
| 06 | `06-connecting-your-apps.md` | Connecting your apps (the local server) | Features | [Local Server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server) |
| 07 | `07-long-context.md` | Long context and the KV cache | Features | [Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls) |
| 08 | `08-memory-that-remembers.md` | Memory that remembers, and the guard | Features | [Runtime Controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls) · `docs/agent-memory.md` |
| 09 | `09-why-nvmai-runs-big-models.md` | Why NVMAI can run models that "don't fit" | About | [System Design](https://github.com/Pummelchen/NVMAI/wiki/System-Design) |
| 10 | `10-what-nvmai-will-not-do.md` | What NVMAI will not do (and how to get help) | About | [FAQ](https://github.com/Pummelchen/NVMAI/wiki/FAQ) |

## The welcome text (for the pinned "Welcome" topic)

> **Welcome to the NVMAI forum 👋**
>
> NVMAI runs large Qwen AI models locally on an Apple Silicon Mac — a 125B
> model on a 24 GB laptop, by streaming what it needs from your SSD instead
> of holding it all in memory. No cloud, no account, no data leaving your
> machine.
>
> This forum is the friendly place to start. **You do not need to be a
> programmer.** The [ten-part series](01-what-is-nvmai.md) walks from "what
> is this" to "it is answering me", in plain words, and explains every
> choice along the way. Getting started is one command:
>
> ```bash
> curl -fsSL https://raw.githubusercontent.com/Pummelchen/NVMAI/main/tools/install_nvmai.sh | bash
> ```
>
> The installer checks your Mac, builds NVMAI, optionally downloads a model,
> and puts a proper double-clickable **NVMAI app** in your Applications
> folder. [Article 02](02-getting-nvmai-running.md) shows the same steps by
> hand if you would rather see what is happening.
>
> If you want the precise version — exact settings, measured numbers, and
> the honest limits — the [wiki](https://github.com/Pummelchen/NVMAI/wiki)
> is the professional reference.
>
> **Where to go**
>
> - 📗 **New here?** Start with [What NVMAI is](01-what-is-nvmai.md), then
>   [Getting NVMAI running](02-getting-nvmai-running.md).
> - 🖥️ **Already running?** [Choosing a model](04-choosing-a-model.md) and
>   [The dials](05-the-dials.md) explain the choices in the app and the
>   launcher.
> - 🔌 **Connecting an app?** [Connecting your apps](06-connecting-your-apps.md).
> - 🧠 **The interesting part?** [Memory that remembers](08-memory-that-remembers.md).
> - 🐞 **Something broke?** [What NVMAI will not do](10-what-nvmai-will-not-do.md).
>
> Ask anything. "This may be a silly question" is the most useful kind of
> post here — if it was unclear to you, it is unclear to someone else, and
> we would rather fix the article than have you stay stuck.

## Publishing workflow

1. One article is one forum topic — paste the file body, keep the `#` title as the topic title.
2. Create the categories first: **Guides** 📗 (articles 02–08), **About NVMAI** 📘 (01, 09, 10), **Show & Tell** 🖼️, **Site Feedback** 💬.
3. Pin the welcome text as the first topic in **About NVMAI**.
4. Post in reading order and replace the relative `NN-*.md` links with topic URLs as each one goes up.
5. Update the status column below after each post, and commit.

## Keeping the articles true to the launcher

Articles 02, 03, 05 and 06 describe what a person actually types. Those
commands come from one place now: `tools/server_launcher.sh`, which starts the
server alone or with Codex, Claude Code, Qwen Code, OpenCode or Zed, and asks
about the model, the thinking level and the RAM limit. When that script's
questions or flags change, these four articles need the same edit — a
screenshot-fresh command in a forum post is the fastest way to lose a new
user.

| Article | Posted | Topic URL |
| --- | --- | --- |
| Welcome | ⬜ | |
| 01 What NVMAI is | ⬜ | |
| 02 Getting NVMAI running | ⬜ | |
| 03 Your first conversation | ⬜ | |
| 04 Choosing a model | ⬜ | |
| 05 The dials | ⬜ | |
| 06 Connecting your apps | ⬜ | |
| 07 Long context | ⬜ | |
| 08 Memory that remembers | ⬜ | |
| 09 Why NVMAI runs big models | ⬜ | |
| 10 What NVMAI will not do | ⬜ | |

## Style rules

- Warm and plain. Short paragraphs. A forum reader is skimming, not studying.
- No unexplained jargon. When a term is needed, define it where it first appears.
- Every number carries its source: the benchmark, the release note, or the machine it ran on.
- Limits get the same space as features. Never oversell.
- One article, one subject. Cross-link instead of repeating.
- End every article with a "where to go next" that continues the series.
