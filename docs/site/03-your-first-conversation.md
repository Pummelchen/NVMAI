# Your first conversation

The build is done and a model is on disk. Let's make it say something.

There are two doors into NVMAI, and they use the same engine. Pick whichever
sounds like you.

- **The Mac app** — a normal window, nothing to type but your question. Start
  here if you are not a programmer.
- **A chat in the Terminal** — one command, one prompt, answer. Start here if
  you like the command line or want to script it.

## Door one: the Mac app

Open the **NVMAI** app from your Applications folder. The installer in
[Getting NVMAI running](02-getting-nvmai-running.md) puts it there. (If you
built NVMAI by hand instead, the app is the file `.build/release/NVMAIMac`
inside the project folder — double-click it in the Finder, or run it from the
Terminal.) A window opens with an input box.

The first thing that happens is the slow thing: the model has to load from
disk into memory. On the 35B models that is a wait of tens of seconds, and
the app tells you it is working. That delay is the price of a model this
size running locally, and it happens once per launch, not once per question.

Then type a question and send it. Try something ordinary to start:

> Explain what a mutex is, as if I have never programmed.

You will see the answer appear a few words at a time. That streaming is
normal — it is the model generating, not a download.

**Two things worth knowing immediately:**

- **It is only as fast as it is.** On a 24 GB M-series Mac, expect the 35B
  models at roughly 20 tokens a second in 4-bit or about 12 in 8-bit, and the
  125B model at about 5. If you are used to cloud chatbots, this will feel
  slower, and that is the honest trade: nothing leaves your machine, and
  nothing is metered.
- **One model at a time.** If a second NVMAI process is already running, they
  will fight over memory. Quit one first.

### The controls you will actually use

Default settings are sane, so you can ignore most of the app. The ones worth
knowing on day one:

| Control | What it does |
| --- | --- |
| **Context length** | How much conversation the model can "hold" at once. Bigger costs more memory. |
| **Temperature** | Lower is more literal and repeatable; higher is more varied. |
| **Thinking** | Whether the model reasons before it answers. See below. |
| **Concise** | Strips the polite preamble and closing summary. |

There is no wrong setting to begin with. When you want to tune, read
[The dials](05-the-dials.md) — it explains each one, including the ones you
should probably leave alone.

### A word about "thinking"

Some models can reason at length before answering. It genuinely helps with
hard questions — a tricky bug, a piece of analysis — and genuinely wastes
your time on easy ones.

NVMAI only offers the thinking controls each model's own template actually
implements. That is deliberate: with thinking off it answers directly, and it
will not invent "low/medium/high effort" modes for a model that does not have
them. If you want the details of which model offers what, that is
[Choosing a model](04-choosing-a-model.md).

For a first session, leave thinking **off**. You get the answer faster, and
you will see exactly what the model does with no help.

## Door two: the Terminal

Open Terminal and start the server:

```bash
~/NVMAI/tools/server_launcher.sh
```

(The installer also made a shortcut, so `nvmai` works if
`~/.local/bin` is on your `PATH`.)

Answer its questions — pressing Enter through them all gives you the
recommended setup. When it prints **"NVMAIServer ready"**, the model is up.
Leave this window open: the launcher *is* the server. `Ctrl-C` in that window
stops it.

**For a quick one-off question**, without a server at all, use the CLI
directly. This is the simplest possible thing that works:

```bash
.build/release/NVMAICLI \
  --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "The capital of France is" \
  --max-new 32 \
  --temperature 0
```

Text goes to standard output; the timing and token count go to standard
error. `--temperature 0` means "always pick the most likely next token", so
the same prompt gives the same answer every time — useful when you are
testing whether something changed.

A few flags worth meeting early:

- `--max-new 64` — stop after 64 new tokens instead of rambling on.
- `--quiet` — hide the timing footer.
- `--messages-file chat.json` — send a proper chat conversation (a JSON list
  of `role` and `content`) instead of a raw prompt.

**If you want to talk to it like a chat assistant**, use the server and a
client rather than the raw CLI. [Connecting your apps](06-connecting-your-apps.md)
sets that up, and it is genuinely the best way to use NVMAI day to day.

## Which ever you chose, the model is the same

App, CLI, and server all run the same weights with the same engine. The app
and the server just add convenience — a window, or an API that other programs
can call. Nothing is second-class.

## When it does not answer

| What you see | What to check |
| --- | --- |
| It sits "loading" for minutes | Normal for the first load of a large model; watch free memory |
| It starts, then dies | Another model process may be running — quit it and retry |
| The app opens but cannot find a model | The model folder name must match exactly; see [Getting NVMAI running](02-getting-nvmai-running.md) |
| Very slow, machine sluggish | Free up memory, or pick a smaller model — [Choosing a model](04-choosing-a-model.md) |
| Nonsense or repeated text | Lower the temperature; see [The dials](05-the-dials.md) |

And genuinely: ask on the forum. Include which model and which app you used —
that is usually enough for someone to spot it.

## Where to go next

You have it answering. The next real decision is which model you are running
and why → **[Choosing a model](04-choosing-a-model.md)**

*NVMAI 5.1 at the time of writing. Speed figures come from the project's
published benchmarks on a base 8-core M3 with 24 GB; your Mac will differ.*
