# Getting NVMAI running on your Mac

This article covers the one-time setup. You will paste a single command into
the Terminal — the article then explains what it did, and how to do the same
steps by hand if you prefer or need to fix something.

Paste this into the Terminal and press Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/Pummelchen/NVMAI/main/tools/install_nvmai.sh | bash
```

That script checks your Mac, downloads and builds NVMAI, offers to fetch a
model, and installs a proper double-clickable **NVMAI app** into your
Applications folder. Budget an evening rather than a coffee break — building
and downloading a model is mostly waiting, and how long depends on your
connection and the model you pick. It is safe to re-run: if something
interrupts it, running it again picks up where it can instead of starting
over.

After it finishes, **you never need the Terminal again** to *use* NVMAI — the
app is a normal Mac app.

If you have never opened the Terminal: you can still do this. The rest of this
article is the same installation **by hand**, one step at a time, for anyone
who wants to see what is happening or fix something that went wrong. If a step
looks different on your Mac, stop and ask on the forum rather than guessing.

> **The one thing to know up front.** NVMAI is not a notarized download, because
> that needs an Apple Developer account the project does not have. This is why
> it is *built on your machine* from source: your Mac trusts what it compiled
> itself. That single command is the price of that, and you pay it once.

## The manual route (optional)

Open the Terminal (`Cmd + Space`, type `Terminal`), then follow the steps
below. They are exactly what the installer does.

## What you need before you start

| Check | Why | How to check |
| --- | --- | --- |
| **Apple Silicon Mac** (M1 or newer) | NVMAI is built for these chips only |  menu → **About This Mac** → look for "Chip: Apple M…" |
| **macOS 26 or later** | The project targets the current system | Same window — "macOS 26.x" |
| **Xcode** (the full app, not just the command line tools) | Provides the Swift 6.3+ compiler | See step 1 |
| **Free disk space** | Models are big; see the table below |  menu → **System Settings** → **General** → **Storage** |
| **A stable internet connection** | The model is a large download | — |

### How much disk space?

| Model | Installed size |
| --- | ---: |
| Ornith 1.5 35B-A3B, 4-bit | about 20 GB |
| Qwen 3.6 35B-A3B, 4-bit | about 20 GB |
| Qwen-AgentWorld 35B-A3B, 4-bit | about 20 GB |
| Ornith 1.5 35B-A3B, 8-bit | about 37 GB |
| Qwen 3.6 35B-A3B, 8-bit | about 37 GB |
| Qwen 3.8 Flash Next 125B-A6B, 4-bit | about 174 GB |
| Qwen 3.5 2B / 4B / 9B (run on the CPU) | about 1.3 GB – 9.9 GB |

**Add room to spare** — a few gigabytes for the build itself and for macOS
to work comfortably. The 125B model is a serious commitment: check your free
space twice before starting it.

Unsure which to pick? [Choosing a model](04-choosing-a-model.md) is the next
article. The short version: the installer's default and the library's default
are both **Ornith 1.5 35B-A3B, 8-bit**, the highest-fidelity option. The
lighter **4-bit** build is about half the download and roughly twice as fast,
at a small cost in accuracy. Either is a fine first choice.

## Step 0 — Open the Terminal

Press `Cmd + Space`, type `Terminal`, press Enter. A plain window opens with
a line ending in `%` or `$`. That is the prompt; you type here and press
Enter.

You will be **pasting** most of what follows, so copy each block exactly.
Nothing here edits your system settings or removes anything.

## Step 1 — Install Xcode

Open the **App Store**, search for **Xcode**, and install it. It is a large
download (several gigabytes) and this is the slowest part of the whole
process.

Then install its command-line components. Paste this and press Enter:

```bash
xcode-select --install
```

If a dialog says the tools are already installed, that is fine — you are
done with this step.

Now check that Swift is available and new enough:

```bash
swift --version
```

You want to see **Swift 6.3** or higher. If you see an older version or
"command not found", finish the Xcode install, open Xcode once so it can
finish setting itself up, and try again.

## Step 2 — Download NVMAI

In the Terminal, paste these three lines. The first moves you to your home
folder, the second downloads NVMAI, the third opens its folder:

```bash
cd ~
git clone https://github.com/Pummelchen/NVMAI.git
cd NVMAI
```

You now have a folder called `NVMAI` in your home folder — the same one you
see in the Finder sidebar.

**If `git` is not found**, install the Xcode command-line tools from step 1
and try again.

## Step 3 — Build it

Paste this and press Enter:

```bash
swift build -c release
```

This compiles NVMAI for your Mac. It takes a while on the first run — expect
several minutes, and it is normal for the screen to sit there printing
progress. When it finishes you should see:

```
Build complete!
```

That is the whole build. Nothing was installed system-wide; everything
lives inside the `NVMAI` folder.

## Step 4 — Download a model

Models are downloaded and verified in one pass. This is the big wait — the
8-bit model is about 37 GB.

```bash
swift run -c release NVMAIRepack --output models/ornith-1.5_35B_A3B_8Bit
```

A progress line reports how much has arrived and how fast. **Get the casing
right** — that path is `models/ornith-1.5_35B_A3B_8Bit`. (The *name* is only a
convention, though: NVMAI identifies a model by the metadata inside it, not by
the folder name, so a rename does not confuse the server.)

Good to know while it runs:

- **If it is interrupted** (you close the lid, the connection drops), just
  run the same command again with `--resume` added. It continues from what
  it already verified rather than starting over:
  ```bash
  swift run -c release NVMAIRepack --model ornith15-8bit --output models/ornith-1.5_35B_A3B_8Bit --resume
  ```
- **If you want to abandon it**, this removes the partial download:
  ```bash
  swift run -c release NVMAIRepack --discard-partial --output models/ornith-1.5_35B_A3B_8Bit
  ```
- Hugging Face may ask for a token for some models. Only then, set `HF_TOKEN`
  as the message tells you to. You do not need an account for the default
  model.

A note on the small CPU models: the Qwen 3.5 2B, 4B and 9B are much smaller
(about 1.3–9 GB) and run on the processor rather than the graphics chip, which
makes them a lovely way to get a feel for NVMAI without a long download. They
install through the same script:

```bash
tools/install_models.sh qwen35-2b        # also qwen35-4b, qwen35-9b
```

[Choosing a model](04-choosing-a-model.md) explains what each one is good for.

## Step 5 — Check the install

This verifies the model on disk without loading it into memory:

```bash
swift run -c release NVMAIRepack --verify-install --input-gturbo models/ornith-1.5_35B_A3B_8Bit
```

A pass here means every byte matches what was published. If you see a
"model directory mismatch" message instead, you have moved or renamed the
folder — that is a known, fixable thing, explained in
[What NVMAI will not do](10-what-nvmai-will-not-do.md).

## Step 6 — Start it

The friendliest way: run the launcher and answer its questions. You have to be
**inside the NVMAI folder** for the relative path to work (step 2 left you
there):

```bash
tools/server_launcher.sh
```

If you are somewhere else, use the full path — substitute wherever you put the
folder:

```bash
~/NVMAI/tools/server_launcher.sh
```

It asks a few things in plain language — which API your client uses, which
model to load, whether you want the model to think before answering. Press
Enter to accept each default. Then leave that window open: **the launcher is
the program.** Closing it stops the model.

Once it says **"NVMAIServer ready"**, it prints the address your apps should
use:

```
Base URL:   http://127.0.0.1:8080/v1
API key:    any value (the server does not authenticate)
```

That is the handoff point. Keep the Terminal window visible while you use
NVMAI, and press `Ctrl-C` in it when you are done.

## One at a time, please

NVMAI runs **one model process per Mac**. Before starting it, check nothing
else is already using a model:

```bash
memory_pressure -Q
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI'
```

The first line should report a healthy amount of free memory. The second
should print **nothing**. If it prints something, a model is already
running — use that one, or stop the one you started. Never kill a process
you did not start yourself.

## If something went wrong

| What you see | What it usually means |
| --- | --- |
| `command not found: swift` | Xcode's tools are not installed — redo step 1 |
| `error: 'nvmai': Invalid manifest` | The Swift toolchain is too old; you need 6.3+ |
| Build stops with an error | Copy the **whole** message to the forum; the last 20 lines matter most |
| Download restarts from zero | Use `--resume` as shown in step 4 |
| Launcher says a model is not installed | The download has not finished, or it landed outside the `models/` folder |
| Launcher says "catalog is unavailable" | The built binary is older than the source; rebuild with step 3 |

## Where to go next

It built, it downloaded, it started. Now let's actually talk to it →
**[Your first conversation](03-your-first-conversation.md)**

The precise, engineering version of this page — every flag and every
requirement, stated exactly — is
[Getting Started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
on the wiki.

*NVMAI 5.1 at the time of writing. Requirements reflect the project's
documented minimums: Apple Silicon, macOS 26+, Swift 6.3+.*
