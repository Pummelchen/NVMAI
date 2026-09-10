> **Category:** General (or a future "Guides" category)
> **Status:** draft v1 — to be reviewed before posting

# What NVMAI is, and what it is made for

NVMAI is a native Swift 6 + Metal inference runtime for Qwen-based MoE text models on Apple Silicon, from M1 to M6. It exists to break one specific wall: **the RAM wall.**

A 35B mixture-of-experts model is 15–40 GB of weights. A Mac is 16–64 GB of RAM. That arithmetic usually means "downsize the model." NVMAI instead keeps the routed experts on the NVMe drive and streams them into a small, bounded slice of memory as the model generates. The model size is then bounded by your **disk**, not your memory. A 125B model runs on a 24 GB Mac — and even on 8 GB — because only the experts the next token needs are ever resident at once.

You set the RAM budget, and NVMAI holds the line. Give it 4 GB or 8 GB for the expert cache and it stays inside it, so your Mac stays responsive while the model runs.

## What it is good at

- **Huge models on small Macs.** Qwen3.8-Flash-Next 125B-A6B in 4-bit streams at 5.46 tok/s on an 8-core M3. The 35B-A3B models run at about **21.7 tok/s** (4-bit) — close to what people expect from models a third their size.
- **A real API, not a toy.** The loopback server speaks OpenAI Chat Completions and the Responses API, with streaming and client-side tool calls. Launch scripts wire up Codex, Qwen Code, and OpenCode directly.
- **One command per model.** `tools/start-<model>-<bits>.sh` installs nothing to configure: pinned model, tuned settings, own port, done. Installs are verified end to end with a path-bound receipt, so a model that loads is the model that was promised.
- **It uses the Mac at the physical limit.** Own Metal kernels, own model format (no MLX, no GGUF), and a converter that builds the format straight from the original weights. The Apple Neural Engine runs long-prompt prefill about 2.3× faster than the GPU; tiled GPU Top-K sampling cut per-token sampling cost from 15.5 ms to 1.4 ms.
- **Long context.** Native RoPE to 262K tokens; optional YaRN to 512K or 1M. The KV cache can be stored in 16-, 8-, or 4-bit, independent of the model's own quantization (8-bit is the default).
- **It is a measurement project, not a marketing project.** Every performance claim in this forum comes from a script in `benchmark/` with a fixed model, seed, and hardware, and the numbers change only when a release notes them — v5.1, for example, reports 10–13% faster 8-bit decode on the 35B family and byte-identical output on all eight golden runs.

## What it is not, and what it cannot do

Limits are part of the design, so here they are without fine print:

- **Text only.** No vision, no audio. (Ornith's vision and audio are not included.)
- **One model at a time.** One loaded model, one generated choice, per server. One model process per machine is the operating rule, not a bug.
- **Disk is the new budget.** 19.5 GB for a 35B in 4-bit, roughly 37 GB in 8-bit, and about 161 GB for the 125B in 4-bit — its hashed n-gram table alone is 95 GB.
- **The server is local by design.** It binds to `127.0.0.1`, has no authentication and no TLS, and must never be proxied, tunneled, or exposed. That is a feature (nothing can reach your model) and a limit (no remote access).
- **Tool calls go back to the client.** NVMAI proposes tool calls; it never executes or authorizes them. Your client's permission policy is still the one that decides.
- **Thinking is binary.** Ornith and Qwen expose reasoning as an on/off chat-template switch. NVMAI does not dress prompt tricks up as Low/Medium/High effort levels.
- **Speculative decoding (MTP) is off by default.** It is experimental, requires greedy decoding and native RoPE, and measured Ornith runs showed no speed benefit — so it stays off until it does.

## What it runs today

| Model | Quantizations | Notes |
| --- | --- | --- |
| Qwen-AgentWorld 35B-A3B | 4-bit, 8-bit | fastest 4-bit decode of the 35B family (21.74 tok/s, M3) |
| Qwen 3.6 35B-A3B | 4-bit, 8-bit | |
| Ornith 1.5 35B-A3B | 4-bit, 8-bit | 8-bit is the historical default; 4-bit is the baseline |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit, 8-bit | the "125B on a laptop" model |

Hardware: Apple M1–M6, macOS 26 or later, Swift 6.3 or later, and SSD space as above. Six-bit support was withdrawn; it will not load.

## How to get the first feel for it

Clone the repo and run one start script:

```bash
git clone https://github.com/Pummelchen/NVMAI
cd NVMAI
tools/start-ornith-8bit.sh   # or any other start-<model>-<bits>.sh
```

Point Codex, Qwen Code, OpenCode — or any OpenAI-compatible client — at `http://127.0.0.1:<port>` and ask it something. Then read the [Getting Started guide] and come back here with your first question.

This forum will grow one article at a time, each explaining one main feature: SSD expert streaming, the RAM budget, installs and verified receipts, the OpenAI-compatible server, runtime controls, ANE prefill, long context and KV compression, agent memory, and benchmarking. The plan for the series is in the [wiki → forum migration topic].

*Version at time of writing: NVMAI 5.1.*
