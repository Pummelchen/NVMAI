> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** Getting-Started (paraphrased, commands verified against the repo)

# Getting Started: install and your first run

This gets you from a bare Mac to a model answering questions. About twenty minutes, most of it the download.

## Which model, first

Start with **Ornith 1.5 8-bit**. It completed every 8-bit coding/tooling qualification cell in the current matrix, and the fastest correct configuration was standard responses with Thinking off. Choose **4-bit** when smaller storage and higher raw decode speed matter more than that result.

| Model | Size on disk | When to pick it |
| --- | --- | --- |
| Ornith 1.5 8-bit | ~36.9 GB | Default. The one that qualified cleanest. |
| Ornith 1.5 4-bit | ~19.5 GB | Smaller, faster raw decode. |
| Qwen 3.6 4-bit / 8-bit | ~19.5 / 37.8 GB | The other supported 35B family. |

(Add Qwen-AgentWorld and Qwen3.8-Flash-Next 125B once you have the basics working — see [What NVMAI is](#01).)

## 1. Check the Mac

Apple Silicon, **macOS 26+**, **Swift 6.3+**, and enough free internal SSD. Before loading any model, confirm the machine is actually free to run one — this matters more than the hardware line, because a second model process is how most "it's broken" reports start:

```bash
memory_pressure -Q
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Continue only when the process check prints nothing. **Do not kill a process you did not start.**

## 2. Build

```bash
git clone https://github.com/Pummelchen/NVMAI.git
cd NVMAI
swift build -c release
```

The release build is the one every command below points at.

## 3. Install a model

The installer (`NVMAIRepack`) streams verified ranges straight into the final `.gturbo` directory — it does not stage a second full checkpoint. An interrupted download resumes with `--resume`; `HF_TOKEN` is only needed if Hugging Face actually asks.

```bash
# Default: Ornith 1.5 8-bit
swift run -c release NVMAIRepack \
  --output models/ornith-1.5_35B_A3B_8Bit

# Smaller/faster: Ornith 1.5 4-bit
swift run -c release NVMAIRepack \
  --model ornith15 \
  --output models/ornith-1.5_35B_A3B_4Bit
```

## 4. Verify, then run a smoke test

Verify without loading the weights (it re-hashes the payload against the manifest and checks the receipt):

```bash
swift run -c release NVMAIRepack \
  --verify-install \
  --input-gturbo models/ornith-1.5_35B_A3B_8Bit
```

Then a deterministic smoke test. `temperature 0` makes it greedy, so you should get the same 32 tokens every time:

```bash
.build/release/NVMAICLI \
  --model models/ornith-1.5_35B_A3B_8Bit \
  --prompt "The capital of France is" \
  --max-new 32 \
  --temperature 0
```

Text goes to stdout; timing and token counts go to stderr. The live KV cache defaults to 8-bit whether the weights are 4- or 8-bit — that's a runtime control, not an install choice.

## 5. Pick your interface

- **Mac app:** `.build/release/NVMAIMac` (set the model with `defaults write NVMAI model ornith15-8bit`).
- **CLI / scripts:** `.build/release/NVMAICLI --help`.
- **API + coding CLIs:** the [OpenAI-compatible server](#06) — this is where Codex, Qwen Code, and OpenCode connect.

## The one thing that trips people up: moving the model

The install's `verified-install.json` receipt is **bound to the absolute path** it was installed to. Move or rename the directory and the model refuses to load with `trusted receipt invalid: model directory mismatch`. That is not corruption and does not need a re-download — re-issue the receipt in place:

```bash
swift run -c release NVMAIRepack --verify-install --input-gturbo /new/path/to/model
```

Never hand-edit the receipt to match a new path. The path binding is what detects a moved or swapped directory; editing it forges the attestation instead of re-establishing it. (Full treatment in [Installs and verified receipts](#05).)

## Where to go next

- The engine's signature trick, explained: [SSD expert streaming](#03)
- Why the RAM budget behaves the way it does: [The RAM budget](#04)
- Connect a coding CLI: [The OpenAI-compatible server](#06)

*Version at time of writing: NVMAI 5.1.*
