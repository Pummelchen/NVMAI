# Local OpenAI-compatible server

`TurboFieldfareServer` exposes a local Chat Completions API for one loaded
model. It binds to `127.0.0.1` without authentication or TLS. Do not expose it
through a proxy or tunnel.

## Start the server

First, install the model with the Mac app or `TurboFieldfareRepack`. Then check
that no other TurboFieldfare model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/qwen36-8bit.gturbo \
  --model-id qwen3.6-35b-a3b \
  --port 8080 \
  --max-context 131072
```

The server loads the model before opening the port. Wait for
`TurboFieldfareServer ready`, then keep the process running while clients use
it.

Check the server from another terminal:

```bash
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/v1/models
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

By default, the server runs one generation and queues up to four requests. Use
`--queue-limit` to change the queue size. Press Control-C to stop the server.
The 128K setting allocates about 2.5 GB more Qwen FP16 KV capacity than 16K;
check `memory_pressure -Q` before using it on a memory-constrained Mac.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "turbofieldfare": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboFieldfare",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local",
        "headers": {
          "X-NVMAI-Client": "opencode",
          "X-NVMAI-Profile": "coding-lean"
        }
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 35B-A3B"
        }
      }
    }
  }
}
```

Select `turbofieldfare/qwen3.6-35b-a3b` in OpenCode. Qwen 3.6 server requests
use a 1,024-token chunked-prefill ceiling by default. `--prefill-chunk` accepts
32 through 4,096 in powers of two; larger chunks reduce repeated expert-file
sweeps for long prompts while using more temporary GPU memory. The advertised
model ID is unchanged, so existing client configuration continues to work.

### OpenCode lean profiles

Lean filtering is opt-in and never uses `User-Agent` detection. The server
requires `X-NVMAI-Client: opencode` together with one of these profile headers:

- `X-NVMAI-Profile: coding-lean` replaces OpenCode's client guidance with a
  compact coding instruction, keeps the complete conversation and tool-result
  lineage, and exposes only `read`, `grep`, `glob`, `list`, `bash`, `edit`,
  `write`, and `apply_patch`. Tool descriptions and JSON Schema annotations are
  compacted without changing argument structure.
- `X-NVMAI-Profile: prompt-only` sends only the latest user text to the model.
  It removes all system/developer guidance, earlier conversation, tools, calls,
  and results. Use it for ordinary questions, not agentic coding work.

Create a second OpenCode provider with the same base URL and model ID but the
`prompt-only` header when both modes are wanted in the model picker. Requests
without `X-NVMAI-Profile` retain the standard OpenAI-compatible behavior.
Supplying a profile without the explicit OpenCode client header, or supplying
an unknown profile, returns `400 invalid_lean_profile`.

For each filtered request, the server logs only non-sensitive measurements:
the profile, request-body bytes, original and filtered prompt-token counts,
message counts, and tool counts. It never logs prompt or tool-result content.

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Use `--prompt-cache-mode off` to disable reuse.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

## Supported API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

Requests may contain system, developer, user, assistant, and tool messages.
Supported options include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, and function-tool fields.

The server supports one model and one choice. It does not support the Responses
API, legacy Completions, embeddings, multimodal input, structured output,
batching, log probabilities, or remote model switching.

Context length can be 4K, 8K, 16K, 32K, 64K, or 128K. The default is 16K.
Larger FP16 KV contexts use more memory. On an 8 GB Mac, run one model process
at a time and watch memory pressure.
