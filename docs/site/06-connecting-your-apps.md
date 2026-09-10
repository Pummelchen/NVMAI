# Connecting your apps (the local server)

This is where NVMAI stops being a chatbot window and becomes *infrastructure*:
your model, answering your existing tools, on your machine.

The idea is small. NVMAI serves a **local** web address that speaks the same
language as OpenAI's and Anthropic's APIs. Any app that can talk to those can
talk to your Mac instead — and then it is your model doing the work, with no
account, no bill, and nothing leaving the machine.

If you are not a programmer, you can skip this article and stay happy with the
Mac app. But if you have ever thought "I wish this assistant ran on my own
computer," this is the article.

## What the server actually speaks

Three compatible surfaces from the same process:

| Surface | Endpoint | Who uses it |
| --- | --- | --- |
| **OpenAI Chat Completions** | `POST /v1/chat/completions` | Almost every local-client tool |
| **OpenAI Responses API** | `POST /v1/responses` | Codex, and the newer OpenAI tooling |
| **Anthropic Messages** | `POST /v1/messages` | Claude Code, Anthropic SDKs |

Plus the small housekeeping routes: `/health` and `/v1/models`.

Because it serves all three, the launcher asking "which API will your client
use?" does **not** change how the server runs. It only decides which setup
instructions it prints for you.

## Starting it

```bash
~/NVMAI/tools/server_launcher.sh
```

The launcher asks a few plain questions and then prints the exact settings
your client needs:

```
Base URL:   http://127.0.0.1:8080/v1
API key:    any value (the server does not authenticate)
Model:      ornith-1.5-35b-a3b_8-Bit (full agent loop)
Endpoints:  POST /v1/chat/completions, POST /v1/responses
```

Two things about that output:

- **The API key is decorative.** The server does not authenticate anything,
  because it only accepts connections from your own machine. Any placeholder
  value works.
- **The model ID ends in the width** — `_4-Bit` or `_8-Bit`. That is
  deliberate: two installs of the same model at different precisions are
  distinguishable instead of both answering to one name. Always ask the
  server rather than assuming:
  ```bash
  curl -s http://127.0.0.1:8080/v1/models
  ```

Leave the launcher window open. It *is* the server; `Ctrl-C` stops it.

## One server, every installed model

This is the pleasant part. You do not pick one model and restart to change it.

With the launcher running, **every installed model is available by name**. Send
a request naming a different model and the server waits for any generation in
flight, unloads the current one, and loads the one you asked for. One model
resident at a time, but no manual switching.

So you can have a fast 4-bit model for chat and a heavy 8-bit model for real
work, both reachable from the same address, and let the client choose.

## The `-fast` alias, and when to use it

Alongside each model ID there is a `-fast` variant:

| Model you request | What you get |
| --- | --- |
| `ornith-1.5-35b-a3b_8-Bit` | The full agent loop. Keeps the tools and the agentic behaviour. |
| `ornith-1.5-35b-a3b_8-Bit-fast` | Chat-only. Strips coding-agent boilerplate before the model sees it, so answers arrive in seconds. |

Use the plain ID for a coding assistant that needs to call tools. Use `-fast`
for asking questions. The launcher asks "full agent loop or fast chat?" for
exactly this reason.

## Client setups

There are two ways to get a client connected:

- **Let the launcher do it.** Run `tools/server_launcher.sh` (no arguments)
  and pick the client when it asks what to launch. It writes that client's
  configuration for you and then opens it.
- **Do it yourself**, using the shapes below. This is also what to do for a
  client the launcher does not know about.

**OpenAI-compatible clients** need only the base URL, any API key, and the
model ID. That is the whole story for most tools.

**Codex** wants a provider entry in its own config (`~/.codex-nvmai/config.toml`):

```toml
model = "ornith-1.5-35b-a3b_8-Bit"
model_provider = "nvmai"

[model_providers.nvmai]
name = "NVMAI"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

Note `wire_api = "responses"` — that is why NVMAI implements the Responses
API and not just Chat Completions.

**Claude Code** uses the Anthropic surface:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=nvmai
export ANTHROPIC_MODEL=ornith-1.5-35b-a3b_8-Bit
export ANTHROPIC_DEFAULT_HAIKU_MODEL=ornith-1.5-35b-a3b_8-Bit
```

Those last two exist because Claude Code names a model for its own background
tasks. Without them it asks for a `claude-*` model this server does not have
and gets a 404 — a confusing failure that is entirely fixable.

**Zed** is an editor rather than a terminal client, and it takes an
OpenAI-compatible provider in its settings file
(`~/.config/zed/settings.json`). The shape, if you would rather add it
yourself:

```json
{
  "language_models": {
    "openai_compatible": {
      "nvmai": {
        "api_url": "http://127.0.0.1:8080/v1",
        "available_models": [
          { "name": "ornith-1.5-35b-a3b_8-Bit",
            "display_name": "NVMAI — Ornith 1.5 8-bit",
            "max_tokens": 262144 }
        ]
      }
    }
  }
}
```

Then choose that model in Zed's agent panel. Set `max_tokens` to the context
window you actually run — 262144 for the native setting, or 524288 / 1048576
with YaRN — because Zed uses it to decide how much context it has left.

**Codex, Claude Code, Qwen Code, OpenCode and Zed** are all supported directly.
For Codex and Qwen Code the launcher writes their configuration into dedicated
directories (`~/.codex-nvmai`, `~/.qwen-nvmai`) so your real configuration is
left untouched, and it disables Qwen Code's stream timeouts, which would
otherwise cut off a long local generation mid-answer. Claude Code is set up
through environment variables. OpenCode and Zed read their own settings files,
so the launcher merges an `nvmai` provider into them — leaving the rest of
your configuration, and your original file as a `.nvmai-backup` — and then
opens the editor for you.

## A two-minute smoke test

Before wiring up a client, prove the server works. With it running:

```bash
curl --silent http://127.0.0.1:8080/health
curl --silent http://127.0.0.1:8080/v1/models
```

Then ask it something:

```bash
curl --silent http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-1.5-35b-a3b_8-Bit",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

If you get `READY` back, the server is fine and any remaining problem is in
the client's configuration.

## Tool calls: who is in charge

Worth being precise about, because it is a safety property and not a
limitation.

When a coding assistant wants to run a command or edit a file, NVMAI **proposes
that tool call and hands it back to your client**. NVMAI never executes it,
never authorizes it, and never bypasses the client's permission rules. Your
client's normal "may I run this?" prompt still happens, exactly as it would
with a cloud model.

The single exception is **memory tools**, which the engine answers itself —
because no coding client would know what to do with a request to remember
something. That is [Memory that remembers](08-memory-that-remembers.md).

## The security rule, and why it is not optional

**The server binds to `127.0.0.1` only.** It has no authentication and no
encryption, by design: everything happens on your machine, so there is nothing
to authenticate and nothing to intercept.

That means one rule, and it is firm:

> **Never proxy, tunnel, or expose the NVMAI server to another machine.**

There is no password to guess because there is no password. The protection is
that only your Mac can reach it. Opening it up would hand anyone on that
network the ability to run a model as you, with your files in reach of whatever
client is connected.

If you want NVMAI on another device, run it on that device.

## If it will not connect

| Symptom | Likely cause |
| --- | --- |
| Connection refused | The server is not running, or you closed the launcher window |
| Model not found (404) | Ask `/v1/models` — the ID must include `_4-Bit` or `_8-Bit` |
| Claude Code 404s on a `claude-*` model | Set the two `ANTHROPIC_*_MODEL` variables above |
| Answer cut off mid-generation | The client's stream timeout is too short; the launcher's configs disable it |
| Very slow first reply | The model is loading, or prefilling a very large prompt |
| The whole Mac is sluggish | A model is already running elsewhere — one at a time |

## Where to go next

Your apps are talking to it. Now let's handle long documents properly →
**[Long context and the KV cache](07-long-context.md)**

The complete API reference — every route, every field, the streaming event
grammar — is
[Local Server](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
on the wiki.

*NVMAI 5.1 at the time of writing. Client names are their owners' trademarks;
NVMAI is not affiliated with them.*
