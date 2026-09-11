<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/dc91bc31-0cd4-42e6-bc7a-67ffb277efe0" />




# NVMAI

NVMAI is the fastest SSD streamer for AI models on Mac - M1 to M6

## New in 5.2

- **One server serves every installed model.** `--models-dir` puts the whole
  catalogue on one port with one model resident at a time, and a request naming
  another model switches to it. One launcher replaces the per-model start
  scripts: pick the client (Codex, Claude Code, Qwen Code, OpenCode, Zed), the
  model, the thinking level and the RAM limit.
- **The dense Qwen 3.5 2B / 4B / 9B run on the GPU as well as the CPU**, with
  the engine selectable per request — `<id>@cpu` or `<id>@gpu`. The GPU path was
  accepted only after its logits matched the CPU engine's on the real install.
- **Optional agent memory:** `NVMAI_MEMORY=1` gives a model facts that outlive a
  conversation, scoped per repository, with nothing to install.
- **Thinking arrives as `reasoning_content`**, apart from the answer, including
  a thought a model opens when thinking is off — which is now split out and
  logged instead of being streamed as the answer.
- **A deep audit of the whole tree:** 89 code findings and 8 documentation
  defects, 0 open (`docs/audit-2026-09-11-findings.md`).

Fixed releases are tagged; the full history is in the
[Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog).

## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB.


| Model | Quantization | Peak decode |
| --- | --- | ---: |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** |


### Supported LLMs

- **Qwen3.8-Flash-Next 125B-A6B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**
- **Qwen 3.5 2B / 4B / 9B** — dense models at 4-bit and 8-bit, on either engine:
  the GPU by default, the CPU on request (`--engine cpu`, or the `@cpu` model id
  for one request). Converted from Qwen's own bf16 release by this project's
  converter (`tools/install_models.sh qwen35-2b|qwen35-4b|qwen35-9b`). The 9B is
  the vision-language build and is converted text-only, like every model here.
  These install as `.gturbo` directories with the same manifest and
  path-bound verification receipt as every other model here. They were affine
  snapshots until the repacker learned the dense shape; the two formats are
  verified equivalent rather than assumed to be, by a byte comparison of every
  resident tensor and by an identical-logits check
  (`tools/repack_dense.sh`, `docs/gturbo-format.md`).


### Usage

- **Easiest install:** one command checks the Mac, builds NVMAI, optionally
  downloads a model, and installs a double-clickable Mac app in
  `~/Applications`. Safe to re-run; it updates instead of cloning twice.
  ```bash
  curl -fsSL https://raw.githubusercontent.com/Pummelchen/NVMAI/main/tools/install_nvmai.sh | bash
  ```
  From a clone, `tools/install_nvmai.sh` does the same. See
  [docs/site](docs/site/) for the plain-language article series, or
  `tools/install_nvmai.sh --help` for its flags.
- **OpenAI-compatible server:** A loopback Chat Completions and Responses API
  for starting NVMAI and connecting supported coding clients.
- **One server, one port, one launcher:** `tools/server_launcher.sh` starts the
  API on its own, or starts it and opens one of the supported clients — Codex,
  Claude Code, Qwen Code, OpenCode or the Zed editor — wiring that client's
  provider config to the model the server advertises. It asks what to launch
  from one list of every installed model and quantization (GPU and CPU), the
  thinking level that model supports, and an optional RAM limit for the expert
  cache (1/2/4/8/16/32 GB; the default is the install's own measured profile).
  It serves on `127.0.0.1:8080` (`NVMAI_PORT` overrides it), and every other
  installed model stays available by name through the API; the server switches
  on demand, keeping one model resident at a time.

```bash
tools/server_launcher.sh                                    # interactive
tools/server_launcher.sh --client codex --model ornith 4     # server + Codex
tools/server_launcher.sh --client zed --model qwen38 4 --ram 16
```

- **Persistent agent memory (optional):** With `NVMAI_MEMORY=1` the model gets
  memory that outlives a conversation, scoped per repository, with six memory
  tools the engine answers itself. It runs inside the server process, so there
  is no database to install and nothing to start. Off by default; see
  [docs/agent-memory.md](docs/agent-memory.md).
- **Three client protocols on one server:** OpenAI Chat Completions, the
  OpenAI Responses API (stored responses, `previous_response_id`, the full
  event grammar) and the Anthropic Messages API (`/v1/messages`,
  `count_tokens`, streaming), so Codex, Claude Code and the OpenAI and
  Anthropic SDKs all talk to the same model; see
  [docs/server-api.md](docs/server-api.md).
- **Tested coding CLIs:** The launch workflow supports Codex, Qwen Code, and
  OpenCode against the local server.
- **Mac app and tools:** NVMAI also provides a native Mac app, direct CLI
  generation, streaming responses, and client-authorized function-tool calls.


### Core Benefits

- NVMAI streams LLM's faster than any other similar project.
- Run large MOE AI models on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. NVMAI streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. NVMAI stays inside it. Give it 4 GB or 8 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing - 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered to use your Mac at the physical limit.
- No MLX. No GGUF. NVMAI ships its own high-speed model format and a converter that builds it straight from the original weights.
  

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


## Core Links

- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [Features](https://github.com/Pummelchen/NVMAI/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [Benchmarks](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)
- [Repository layout](docs/repository-layout.md) — where everything lives, and
  the naming and file-size conventions

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
