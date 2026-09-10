> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** OpenAI-Compatible-Server

# The OpenAI-compatible server

This is how you point Codex, Qwen Code, OpenCode, Claude Code, or any OpenAI- or Anthropic-compatible client at models running on your own Mac. One `NVMAIServer` serves **every** installed model and quantization, GPU and CPU alike, over a **loopback** API that speaks the OpenAI and Anthropic protocols at once. One model is in memory at a time. A request naming another model switches to it.

> **Local by design, and non-negotiable.** The server binds to `127.0.0.1`, has **no authentication and no TLS**, and must never be proxied, tunneled, or exposed to another host. That is a security decision: nothing can reach your model, because there is nothing to authenticate and nothing to encrypt over the wire. If you need remote access, that is a different product.

## Start it

First confirm the machine is free to run a model (you already know this check):

```bash
memory_pressure -Q
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Continue only when the process check prints nothing. Then:

```bash
swift build -c release --product NVMAIServer
.build/release/NVMAIServer \
  --models-dir models \
  --model ornith-1.5-35b-a3b_8-Bit \
  --port 8080
```

`--models-dir` makes every install under `models/` available by name. `--model` picks the one loaded at startup. To see what the server found, with ids, engines and thinking levels, without loading anything:

```bash
.build/release/NVMAIServer --catalog --models-dir models
```

The single-model form, `--model models/ornith-1.5_35B_A3B_8Bit` without `--models-dir`, still works and serves just that install.

Keep that terminal open; stop it with Control-C.

## Talk to it

From another terminal:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

Send a request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-1.5-35b-a3b_8-Bit",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

The KV cache defaults to 8-bit regardless of the weights. For extended context, pass `--rope-scaling yarn` and, when memory is tight, `--kv-bits 4` (see [Runtime controls](#07)).

### Model IDs end in the weight width

The model id **always ends in the weight width**, `_4-Bit` or `_8-Bit`. For a GPU install that is the width of the routed experts, read from the manifest rather than parsed from the name. That keeps a 4-bit and an 8-bit install of the *same* weights apart instead of both answering to one id. This is a **breaking change**: the bare `ornith-1.5-35b-a3b` is no longer accepted. Don't guess: `curl -s http://127.0.0.1:8080/v1/models` lists exactly what it serves.

## Switching models

Name a different id in `model` and the server switches:

1. It waits for the generation in progress to finish.
2. It unloads the resident model.
3. It loads the one you asked for.

It never holds two models at once: on a Mac with 26 GB, two 35B models would swap. The first request to a new model therefore pays that model's full load, and so does the next request back to the old one. Keep a conversation on one model.

Up to five requests can be admitted at once: one generating and four waiting their turn (`--queue-limit`, default 4). A request beyond that gets HTTP 429 rather than an unbounded wait.

`POST /v1/messages/count_tokens` never switches. It counts with the named model's tokenizer and leaves the resident model in place.

## What the API supports

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health; never loads a model. |
| `GET /v1/models` | Every installed model and quantization, GPU and CPU. The response is in Anthropic's shape when the request carries `anthropic-version`, OpenAI's otherwise. |
| `POST /v1/chat/completions` | OpenAI Chat Completions, JSON or SSE streaming. |
| `POST /v1/responses` | OpenAI Responses API, JSON or SSE streaming. |
| `POST /v1/messages` | Anthropic Messages, JSON or SSE streaming. |
| `POST /v1/messages/count_tokens` | Anthropic token count for a request. |
| `POST /v1/models/unload` | Release the resident model; the next request loads whichever model it names. |

It supports system/developer/user/assistant/tool messages, sampling controls, stop strings, seeds, usage reporting, and function tools. It does **not** provide embeddings, multimodal input, legacy Completions, batching, log probabilities, or structured-output enforcement.

Sampling defaults come from the model: temperature 0.6 and top-p 0.95 for the Qwen 3.5 and 3.6 models, temperature 1.0 and top-p 0.95 for Qwen3.8. Values in a request override them.

## Thinking

`--reasoning <level>` sets thinking for the whole server: `off`, `on`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. Each model supports only the levels its chat template renders differently:

| Models | Levels |
| --- | --- |
| Qwen 3.6, AgentWorld, Ornith 1.5, Qwen 3.5 (CPU) | off, on |
| Qwen3.8-Flash-Next | off, low, medium, extra high |

A model without the requested level gets the closest one it has:
- Any effort level on an on/off model means `on`.
- `on` for Qwen3.8 means its template's default, extra high, which is also what `--thinking on` gives a single-model server.
- `off` is always off.

`GET /v1/models` and `--catalog` list each model's levels.

## CPU models

Qwen 3.5 2B and 4B, each at 4-bit and 8-bit, run on the CPU with their weights held in RAM: `qwen3.5-2b_4-Bit`, `qwen3.5-2b_8-Bit`, `qwen3.5-4b_4-Bit`, `qwen3.5-4b_8-Bit`. They are listed and switched to like every other model. Differences from the GPU models:
- The context is capped at 32,768 tokens.
- There is no prompt cache.
- They answer chat, but they are not offered the request's tool definitions, so they never propose a tool call.

## Tool calls: proposed, never executed

NVMAI **returns** function-call requests but never runs them. The client applies its normal permission policy, executes what's approved, appends each result with the matching `tool_call_id`, and resends the complete history. A model on your Mac proposing a tool call is not a model that gets to run one — your client's permission system is still the gate. Only function tools, with automatic or disabled tool choice, are supported.

## Base model vs. the `-fast` alias

- **Base model** (e.g. `ornith-1.5-35b-a3b_8-Bit`): use for coding agents and tool loops.
- **`-fast` alias** (`..._8-Bit-fast`): same weights, but it strips coding-CLI system prompts, tool definitions, tool history, and known reminder blocks *before prefill*. That can cut first-response wall time a lot — at the cost of the information a tool loop needs. It's **chat-only** for that request. Direct question? Use `-fast`. Tool loop? Use the base model.

Every model accepts its `-fast` alias, but `GET /v1/models` does not list the aliases. Listing them would double the menu without giving clients a new model to choose.

## Prompt reuse (why long conversations get faster)

Multi-prefix reuse is **on by default**. Send the complete conversation each turn; a compatible continuation reports how much it reused in `usage.prompt_tokens_details.cached_tokens`. The default keeps up to four prefixes in RAM with a 256 MiB snapshot budget.

To make compatible prefixes **survive restarts**:

```bash
.build/release/NVMAIServer \
  --models-dir models \
  --model ornith-1.5-35b-a3b_8-Bit \
  --port 8080 \
  --prompt-cache-disk "$HOME/Library/Caches/NVMAI/prompt-cache" \
  --prompt-cache-disk-mib 8192
```

> ⚠️ The disk cache can contain conversation text and source code. Keep it **private, local, and unsynchronized**. Prompt reuse improves prefill and time-to-first-token; it does **not** increase decode tokens-per-second.

## Launcher scripts

The repo ships helpers that pick the settings for you:

- `tools/server_launcher.sh`: starts the server in the foreground on `127.0.0.1:8080` and prints the client settings.
- `tools/start-<model>-<bits>.sh`: the same for one install, no questions asked.
- `tools/cli_launcher.sh`: starts a fresh server **and** opens Codex, Qwen Code, or OpenCode, writing isolated configs (it does not clobber your normal config homes).

`tools/server_launcher.sh` asks, in order:

1. **Which API will your client use?** OpenAI (default) or Anthropic. The server speaks both at once; the answer only picks which settings it prints — a base URL ending in `/v1` for OpenAI, `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` for Anthropic. The Anthropic answer also prints `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`, all set to the chosen model. Claude Code runs its background tasks on a second model, and without them it asks for a `claude-*` id the server does not have.
2. **Full agent loop or fast chat?** Full (default) keeps the tool loop; fast serves the `-fast` alias.
3. **Which model?** One numbered list of every installed model and quantization, GPU and CPU, read from `NVMAIServer --catalog`. Ornith 1.5 8-bit is the default.
4. **Which NVMAI mode?** Standard (default) or concise.
5. **Thinking.** Only the levels the chosen model supports, with `off` first and the default: `on` for a model that has only an on/off switch, effort levels such as low, medium and extra high for one that has them.

Every model and quantization shares port 8080 (`NVMAI_PORT` overrides it). The model you choose is only the one loaded first: every installed model is available by name through the API, and the server switches when a request names a different one. The same answers work as arguments, and a dry run prints the server command without starting or stopping anything:

```bash
tools/server_launcher.sh openai full ornith 8 default off
tools/server_launcher.sh anthropic full qwen38 4 default medium
NVMAI_LAUNCHER_DRY_RUN=1 tools/server_launcher.sh
```

If the server binary cannot report its catalog, the launcher says so, offers the built-in list of GPU installs, and starts that one model only.

> ⚠️ The server helper **replaces** an NVMAI server already on port 8080 (or `NVMAI_PORT`); the CLI helper **stops existing `NVMAIServer` processes** before starting. If another session must stay up, start the server manually on another port.

## Model residency (for a shared Mac)

- `--lazy-load` — load on the first inference request, not at startup.
- `--idle-unload-seconds N` — unload after idle; implies lazy load. Single-model servers only; with `--models-dir`, use the endpoint below.
- `POST /v1/models/unload` — unload on demand.

Pair unloading with `--prompt-cache-disk`; unloading discards the in-memory prompt cache.

## Where to go next

- Every flag the server takes: [Runtime controls](#07)
- Why the first token is slow: it's prefill — [Long context and KV cache](#08)
- The engine doing the work: [SSD expert streaming](#03)

*Version at time of writing: NVMAI 5.1.*
