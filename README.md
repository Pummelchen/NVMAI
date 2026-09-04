<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/slogan-dark.svg">
  <img alt="NVMAI is the fastest SSD/NVMe streamer for LLMs on Mac - M1 through M6."
       src="assets/slogan-light.svg" width="660">
</picture>

# NVMAI

### Core Benefits

- NVMAI streams LLM's faster than any other similar project.
- Run large MOE LLM's on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. NVMAI streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. NVMAI stays inside it. Give it 4 GB or 16 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing, 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered against the hardware's real bandwidth limit, not a portability layer's.
- No MLX. No GGUF. NVMAI ships its own high-speed model format and a converter that builds it straight from the original weights.
  

### Supported LLMs

- **Qwen3.8-Flash-Next 125B-A6B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**


Every model id ends in the routed-expert width, read from the manifest rather
than parsed from the name, so two quantizations of the same weights stay
distinguishable. Ask `/v1/models` rather than assuming an id.

### Special Features

- **Bounded expert RAM:** The resident expert cache is sized per family from
  the model's own expert stride and clamped to half of physical memory, so a
  smaller Mac is not handed a budget tuned on a larger one. `--ram-budget`
  overrides it with any size. Model state, KV cache, and runtime scratch use
  additional memory.
- **Long context:** Native RoPE supports up to 262K tokens, while optional YaRN
  extends the context to 512K or 1M tokens.
- **Compressed KV cache:** Live attention state can use 16-bit, 8-bit, or 4-bit
  storage independently of the installed model quantization, with 8-bit as the
  default.
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
- **ANE prefill (experimental, opt-in):** `NVMAI_PREFILL_ANE=on` runs
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
- **Tested coding CLIs:** The launch workflow supports Codex, Qwen Code, and
  OpenCode against the local server.
- **Mac app and tools:** NVMAI also provides a native Mac app, direct CLI
  generation, streaming responses, and client-authorized function-tool calls.

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB.


| Model | Quantization | Peak decode |
| --- | --- | ---: |
| Ornith 1.5 35B-A3B | 4-bit | **19.24 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | **19.21 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | **18.57 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.25 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **9.73 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | **9.72 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | **9.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.03 tok/s** |

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
