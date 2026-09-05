<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />




# NVMAI

NVMAI is the fastest SSD streamer for AI models on Mac - M1 to M6

### Core Benefits

- NVMAI streams LLM's faster than any other similar project.
- Run large MOE AI models on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. NVMAI streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. NVMAI stays inside it. Give it 4 GB or 8 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing - 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered to use your Mac at the physical limit.
- No MLX. No GGUF. NVMAI ships its own high-speed model format and a converter that builds it straight from the original weights.
  

### Supported LLMs

- **Qwen3.8-Flash-Next 125B-A6B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**


### Special Features

- **Bounded expert RAM:** The resident expert cache is sized per family from
  the model's own expert stride and clamped to half of physical memory, so a
  smaller Mac is not handed a budget tuned on a larger one. `--ram-budget`
  overrides it with any size. Model state, KV cache, and runtime scratch use
  additional memory.
- **Long context:** Native RoPE supports up to 262K tokens, while optional YaRN
  extends the context to 512K or 1M tokens.
- **Compressed KV cache:** Live attention state can use 16-bit, 8-bit, or 4-bit
  storage independently of the installed model quantization.
- **Thinking mode:** Ornith and Qwen support truthful Off/On reasoning control;
  their chat templates do not define Low, Medium, or High effort levels.
- **MTP off by default:** Native speculative decoding remains experimental and
  disabled because measured Ornith runs showed no speed benefit and it
  currently requires greedy decoding, native RoPE, and prompt-cache reuse off.

### Performance Improvements

- **Tiled Top-K sampling:** Production sampling (Top-K 1–64) runs a
  three-stage tiled GPU reduction, cutting per-token sampling cost from
  15.5 ms to 1.4 ms with a token-for-token identical stream — the main
  source of the v4.6 decode gain.
- **ANE prefill:** `NVMAI_PREFILL_ANE=on` runs
  full-attention prefill blocks on the Neural Engine from a one-time
  exported Core ML sidecar, roughly halving long-prompt time to first
  token; short prompts and decode are untouched.
- **Follow-up cache:** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Concise mode:** An optional terse system prompt reduces generated text for
  workloads that benefit from it; standard responses are the default because
  they generalized more reliably in the coding/tooling qualification.
- **Fast alias:** The chat-only `-fast` model alias strips coding-agent
  boilerplate before prefill for quicker direct answers, while the base alias
  preserves tools and agent loops.

### Usage

- **OpenAI-compatible server:** A loopback Chat Completions and Responses API
  includes launch scripts for starting NVMAI and connecting supported coding
  clients.
- **One start script per model and quantization:** `tools/start-<model>-<bits>.sh`
  starts the server for that install with no questions asked, each on its own
  port, with the model's own tuning applied. `tools/server_launcher.sh` asks
  instead, and `tools/cli_launcher.sh` also wires up a coding CLI.

```bash
tools/start-ornith-4bit.sh        tools/start-ornith-8bit.sh
tools/start-qwen3.6-4bit.sh       tools/start-qwen3.6-8bit.sh
tools/start-agentworld-4bit.sh    tools/start-agentworld-8bit.sh
tools/start-qwen3.8-4bit.sh       tools/start-qwen3.8-8bit.sh
```

- **Tested coding CLIs:** The launch workflow supports Codex, Qwen Code, and
  OpenCode against the local server.
- **Mac app and tools:** NVMAI also provides a native Mac app, direct CLI
  generation, streaming responses, and client-authorized function-tool calls.

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB.


| Model | Quantization | Peak decode |
| --- | --- | ---: |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** |

## Core Links

- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [Features](https://github.com/Pummelchen/NVMAI/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [Benchmarks](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)

## Credits

NVMAI is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), which
provides the bounded-memory runtime, installer, CLI, Mac app, and local server.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).
