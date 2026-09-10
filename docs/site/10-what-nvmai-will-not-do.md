# What NVMAI will not do (and how to get help)

Every project has limits. The good ones tell you before you spend an evening
discovering them.

This is that list, with no small print. If NVMAI is wrong for what you need,
better to find out here than after a 174 GB download.

## The hard limits

**It is text and function calls. Nothing else.**
No images, no audio, no video. If your task is "describe this photo," NVMAI is
the wrong tool.

**One model at a time.**
One model process per Mac is the operating rule, not a bug. The server can
*switch* between installed models on demand, but only one is loaded at once.
Two NVMAI processes on one Mac will fight over memory. Before starting,
check that nothing else is running:

```bash
memory_pressure -Q
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI'
```

The second command should print nothing. If it prints something, a model is
already running — use it, or stop the one you started. **Never kill a process
you did not start yourself.**

**The server is local, permanently.**
It binds to `127.0.0.1`, with no authentication and no TLS. It must never be
proxied, tunneled, or exposed to another machine. There is no password to
leak because there is no password — the protection is that only your Mac can
reach it. Want NVMAI on another device? Run it on that device.

**Tool calls belong to your client.**
NVMAI proposes tool calls; it never executes or authorizes them itself. Your
client's permission prompts still apply, exactly as with a cloud model. The
one exception is memory tools, which the engine answers itself.

**Only supported models run.**
This is not a general-purpose model runner. You cannot point it at an
arbitrary download and expect it to work. See
[Choosing a model](04-choosing-a-model.md) for what is supported.

**Disk is a real requirement.**
About 20 GB for a 4-bit 35B model, 37 GB at 8-bit, and roughly 174 GB for the
125B model at 4-bit. A nearly full SSD is a problem, and a slow or
spin-down-prone external drive makes loading a large model painful. Storage
speed matters — it is just not the whole story, as the next point explains.

**Speed is bounded by the hardware, not by effort.**
Streaming from SSD costs speed, and the 125B model currently runs at about 5.5
tokens/s. It is tempting to blame the disk, and the project did at first — but
its own profiling showed the real limit on that model is **GPU compute**, not
storage: at the measured cache hit rate the disk work would allow roughly
19–21 tokens/s, while compute caps it nearer 12.6. Nothing you can tune changes
a hardware ceiling.
[Why NVMAI can run models that "don't fit"](09-why-nvmai-runs-big-models.md)
walks through that measurement and the optimisations it ruled out.

**Some things are experimental, and labelled so.**
MTP speculative decoding is off by default because measured Ornith runs showed
no benefit; it also requires greedy decoding and cannot be combined with YaRN.
Six-bit support was withdrawn and will not load. The Metal expert-I/O backend
is experimental and incomplete. The project labels these rather than quietly
shipping them.

**Synced folders are not the place to put models.**
If a cloud service backs up your home folder, a 174 GB model directory is
going to have a bad time. Keep models somewhere sensible.

## The one confusing message, explained

If you move or rename an installed model folder, NVMAI refuses to load it:

```
trusted receipt invalid: model directory mismatch
```

**This does not mean your model is corrupt.** It means the install's
verification receipt is bound to the absolute path it was installed at, and
the path changed. That binding is the feature — it is how NVMAI detects a
moved or swapped directory instead of running something unverified.

The fix takes seconds and needs no re-download:

```bash
swift run -c release NVMAIRepack \
  --verify-install \
  --input-gturbo /new/path/to/the/model
```

**Do not hand-edit `verified-install.json`.** Editing it to match a new path
forges the attestation rather than re-establishing it. The whole point of the
path binding is that it cannot be talked around.

## Before you report a problem

Three checks find most issues:

```bash
# 1. Is something already using a model?
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI'

# 2. Is memory tight?
memory_pressure -Q

# 3. Is the model itself intact?
swift run -c release NVMAIRepack --verify-install --input-gturbo models/<your-model>
```

Also worth checking: that your macOS is 26 or later, your Swift is 6.3 or
later, and that the model folder name matches what the installer created
(including the `.` in `ornith-1.5` and the exact bit width).

## Common questions

**It is very slow.**
Check for another model process, check free memory, and consider a smaller
quantization. First-token time on a huge prompt is genuinely large — that is
prefill, not a fault.

**It repeats itself or rambles.**
Lower the temperature. If that is already low, try a small repetition
penalty. See [The dials](05-the-dials.md).

**The answer is wrong.**
Try an 8-bit model over 4-bit, and turn thinking on for hard questions.
There is no benchmark harness here for answer *quality* — the project measures
weight fidelity, not task success — so treat quality claims with the same
caution the project does.

**Can I run it on an Intel Mac, or on Linux?**
No. Apple Silicon only.

**Can I use it from my phone or another computer?**
Not through the server — it must stay local. Run NVMAI on the device you want
to use it from.

**The app will not open.**
It is ad-hoc signed and built locally, not notarized. If macOS complains,
right-click the app and choose Open once. If it still fails, re-run the
installer — [Getting NVMAI running](02-getting-nvmai-running.md).

**Something is unclear in these articles.**
That is a documentation bug, and telling us is genuinely useful. Say which
article and which step.

## Where to get help

- **This forum.** Include your Mac model, RAM, macOS version, which model you
  are running, and the exact command or screen. Paste the *whole* error — the
  last few lines are usually the answer.
- **The wiki** — the precise reference, for when an article here is too
  friendly: [github.com/Pummelchen/NVMAI/wiki](https://github.com/Pummelchen/NVMAI/wiki)
- **The issue tracker** — for reproducible bugs:
  [github.com/Pummelchen/NVMAI/issues](https://github.com/Pummelchen/NVMAI/issues)
- **The project tracker** — what is planned, what is blocked, and what was
  measured and rejected. Honest reading:
  [Project Tracker](https://github.com/Pummelchen/NVMAI/wiki/Project-Tracker)

## When NVMAI is the wrong choice

Being straight with you:

- **You want the fastest possible local model that fits in RAM.** Use a
  smaller model and let it be resident. Streaming is a compromise for size,
  and you would be paying its cost for no benefit.
- **You need image or audio understanding.** Not supported.
- **You need a hosted API for a team.** The server is loopback-only by
  design.
- **You want a signed, notarized, double-click installer.** Today it is built
  from source on your machine. The installer script makes that one step, but
  it is still a build.
- **You are on a deadline and cannot debug.** This is a young, fast-moving
  project built on measurement. It is honest about its limits, but it is not
  a product with a support contract.

For everyone else — someone with an Apple Silicon Mac who wants a genuinely
capable model running privately, on hardware they already own — that is
exactly the gap NVMAI fills.

## Thanks for reading

That is the series: ten articles from "what is this" to "it is answering me,
and here is why it works." The
[forum](https://nvmai.discourse.group/) is where the interesting parts
continue — post your benchmarks, your setups, and your questions.

*NVMAI 5.1 at the time of writing. Every limit above is drawn from the
project's own documentation; where the project is uncertain, this article
says so rather than inventing confidence.*
