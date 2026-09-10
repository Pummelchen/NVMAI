> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** OpenAI-Compatible-Server

# The OpenAI-compatible server

This is how you point Codex, Qwen Code, OpenCode — or any OpenAI-compatible client — at a model running on your own Mac. `NVMAIServer` serves **one** installed model over a **loopback** OpenAI-compatible API.

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
  --model models/ornith-1.5_35B_A3B_8Bit \
  --port 8083
```

Keep that terminal open; stop it with Control-C.

## Talk to it

From another terminal:

```bash
curl -s http://127.0.0.1:8083/health
curl -s http://127.0.0.1:8083/v1/models
```

Send a request:

```bash
curl -s http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-1.5-35b-a3b_8-Bit",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

The KV cache defaults to 8-bit regardless of the weights. For extended context, pass `--rope-scaling yarn` and, when memory is tight, `--kv-bits 4` (see [Runtime controls](#07)).

### Model IDs end in the expert width

The model id **always ends in the routed-expert width** — `_4-Bit` or `_8-Bit`. The width is read from the manifest's routed-expert slot, not parsed from the name, so a 4-bit and an 8-bit install of the *same* weights are distinguishable instead of both answering to one id. This is a **breaking change**: the bare `ornith-1.5-35b-a3b` is no longer accepted. Don't guess — `curl -s http://127.0.0.1:8083/v1/models` lists exactly what it serves.

## What the API supports

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health; never loads a lazy model. |
| `GET /v1/models` | The base model and its `-fast` alias. |
| `POST /v1/chat/completions` | Chat Completions, JSON or SSE streaming. |
| `POST /v1/responses` | Responses API, JSON or SSE streaming. |
| `POST /v1/models/unload` | Release a dynamically managed model. |

It supports system/developer/user/assistant/tool messages, sampling controls, stop strings, seeds, usage reporting, and function tools. It does **not** provide embeddings, multimodal input, legacy Completions, model switching, batching, log probabilities, or structured-output enforcement.

## Tool calls: proposed, never executed

NVMAI **returns** function-call requests but never runs them. The client applies its normal permission policy, executes what's approved, appends each result with the matching `tool_call_id`, and resends the complete history. A model on your Mac proposing a tool call is not a model that gets to run one — your client's permission system is still the gate. Only function tools, with automatic or disabled tool choice, are supported.

## Base model vs. the `-fast` alias

- **Base model** (e.g. `ornith-1.5-35b-a3b_8-Bit`): use for coding agents and tool loops.
- **`-fast` alias** (`..._8-Bit-fast`): same weights, but it strips coding-CLI system prompts, tool definitions, tool history, and known reminder blocks *before prefill*. That can cut first-response wall time a lot — at the cost of the information a tool loop needs. It's **chat-only** for that request. Direct question? Use `-fast`. Tool loop? Use the base model.

## Prompt reuse (why long conversations get faster)

Multi-prefix reuse is **on by default**. Send the complete conversation each turn; a compatible continuation reports how much it reused in `usage.prompt_tokens_details.cached_tokens`. The default keeps up to four prefixes in RAM with a 256 MiB snapshot budget.

To make compatible prefixes **survive restarts**:

```bash
.build/release/NVMAIServer \
  --model models/ornith-1.5_35B_A3B_8Bit \
  --port 8083 \
  --prompt-cache-disk "$HOME/Library/Caches/NVMAI/prompt-cache" \
  --prompt-cache-disk-mib 8192
```

> ⚠️ The disk cache can contain conversation text and source code. Keep it **private, local, and unsynchronized**. Prompt reuse improves prefill and time-to-first-token; it does **not** increase decode tokens-per-second.

## Launcher scripts (the no-questions path)

The repo ships helpers that pick the settings for you:

- `tools/server_launcher.sh` — starts the server in the foreground, prints the client settings.
- `tools/cli_launcher.sh` — starts a fresh server **and** opens Codex, Qwen Code, or OpenCode, writing isolated configs (it does not clobber your normal config homes).

Run with no arguments (Enter selects `codex / full / 8 / default / off` on Ornith 1.5) or pass all five choices:

```bash
tools/cli_launcher.sh codex full 4 default on
tools/server_launcher.sh opencode full 8 default on
```

> ⚠️ The server helper **replaces** an NVMAI server already on its selected port; the CLI helper **stops existing `NVMAIServer` processes** before starting. If another session must stay up, start the server manually.

## Model residency (for a shared Mac)

- `--lazy-load` — load on the first inference request, not at startup.
- `--idle-unload-seconds N` — unload after idle; implies lazy load.
- `POST /v1/models/unload` — unload on demand.

Pair idle unloading with `--prompt-cache-disk`; unloading discards the in-memory prompt cache.

## Where to go next

- Every flag the server takes: [Runtime controls](#07)
- Why the first token is slow: it's prefill — [Long context and KV cache](#08)
- The engine doing the work: [SSD expert streaming](#03)

*Version at time of writing: NVMAI 5.1.*
