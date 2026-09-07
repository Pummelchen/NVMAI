# The server's APIs

NVMAIServer speaks three client protocols over one generation path. A request
on any of them becomes the same validated chat request, runs through the same
queue, prompt cache, memory decorator and decoder, and is answered in the
protocol's own object and event shapes. What the model can do is identical on
all three; what differs is the wire format.

| Protocol | Routes | Spoken by |
| --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | most clients, Aider, Continue, OpenCode |
| OpenAI Responses | `POST /v1/responses`, `GET/DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, `GET /v1/responses/{id}/input_items` | Codex CLI, the OpenAI SDKs' `responses` namespace |
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Claude Code, the Anthropic SDKs |

Shared: `GET /health`, `GET /v1/models`, `GET /v1/models/{id}` (in the
Anthropic list shape when the request carries `anthropic-version`),
`POST /v1/models/unload`.

There is no authentication: `Authorization` and `x-api-key` are accepted and
ignored, because the server binds to 127.0.0.1 and the machine is the trust
boundary. Every response carries `openai-version: 2020-10-01`; Anthropic
responses also carry `request-id`.

## The model name

A request must name the served model or its `-fast` alias (`GET /v1/models`
lists both). Any other name is refused: 404 `model_not_found` on the OpenAI
paths, 404 `not_found_error` on the Anthropic path. That includes the
`claude-*` names Claude Code sends by default, so point it at the served id:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8096 ANTHROPIC_MODEL=$(curl -s 127.0.0.1:8096/v1/models | jq -r '.data[0].id') claude
```

The alternative — answering any name with whatever is loaded — would make a
misconfigured client believe it was talking to a model it was not.

## What every protocol shares

- **Text only.** Image, file, audio and document inputs are refused with the
  protocol's error envelope naming the offending part
  (`unsupported_content` / `invalid_request_error`), never silently dropped.
- **Function tools.** Tools are the client's to run; the server returns the
  call. Hosted and built-in tool types (`web_search`, `file_search`,
  `code_interpreter`, `mcp`, `computer_use`, Anthropic's `bash_*`,
  `text_editor_*`, `web_search_*`) are refused by type. The server runs
  exactly one kind of tool itself: its own `memory_*` functions, when memory
  tools are on (see `agent-memory.md`).
- **Tool choice.** `auto` and `none` are honoured. Forcing a call —
  `required`, a named function, Anthropic's `any` / `tool` — is refused,
  because the decoder cannot guarantee one. `parallel_tool_calls=false` and
  `disable_parallel_tool_use` are refused for the same reason.
- **Reasoning is a load-time setting.** `--thinking on` and
  `--reasoning-effort` decide what the template renders. A request may
  confirm the active level (`reasoning.effort`, `reasoning_effort`,
  `thinking.type`) and is refused if it asks for another, with the restart
  flag named in the message. The model's thoughts are never returned.
- **Structured output** (`response_format`, `text.format: json_schema`,
  `output_config.format`) is refused; the decoder has no grammar constraint.
- **Logprobs**, `n > 1`, `background: true`, prompt templates, hosted
  conversations, containers and MCP servers are refused by name.
- **Output cap.** Omitting `max_tokens` / `max_output_tokens` lets the model
  run to the context window. The Messages API requires `max_tokens`, as the
  real one does.
- **Usage** is exact: prompt and generated tokens, with the prompt-cache hit
  reported as `cached_tokens` (OpenAI) or `cache_read_input_tokens`
  (Anthropic, where `input_tokens` excludes it).

## Chat Completions

Unchanged from before: `messages` with `system`/`developer`/`user`/
`assistant`/`tool` roles, `tools`, `stream` with `stream_options.include_usage`,
`stop` (up to four strings), `seed`, `temperature`, `top_p`, `top_k`,
`repetition_penalty`. Streams end with `data: [DONE]`; a failure mid-stream
sends an error object first.

## Responses

`input` is a string or a list of items. Item kinds: `message` (roles
`user`, `system`, `developer`, `assistant`; content a string or parts of type
`input_text`, `output_text`, `refusal`), `function_call`, `function_call_output`
(output a string or `input_text` parts), `reasoning` (accepted and skipped: a
client replaying an earlier turn returns what it was given) and
`item_reference` (resolved against stored responses). `instructions` and any
`system`/`developer` items merge into the single leading system message the
chat template accepts.

**Storage.** `store` defaults to true, as in the API. A finished response is
kept in memory (the newest 256) so that `previous_response_id` continues it —
the stored conversation and its output are prepended to the new input — and
`GET /v1/responses/{id}`, `GET .../input_items` and `DELETE` work on it.
`POST .../cancel` answers 400: nothing this server produces is a background
response. `store: false` keeps nothing, and a `previous_response_id` that
names nothing is 404 `previous_response_not_found`-style (`not_found`).

**The object** has every field the API defines (`background`,
`completed_at`, `incomplete_details`, `max_tool_calls`, `prompt_cache_key`,
`safety_identifier`, `service_tier`, `text.verbosity`, `top_logprobs`,
`truncation`, `usage.input_tokens_details.cache_write_tokens`, ...). Fields the
server cannot act on are echoed, not invented. A generation the output cap
ended is `status: "incomplete"` with `incomplete_details.reason:
"max_output_tokens"`.

**Streaming** follows the API's event grammar exactly, every event carrying
`sequence_number`:

```
response.created → response.in_progress
→ response.output_item.added (message) → response.content_part.added
→ response.output_text.delta … → response.output_text.done
→ response.content_part.done → response.output_item.done
→ [per call] response.output_item.added (function_call)
             → response.function_call_arguments.delta …
             → response.function_call_arguments.done → response.output_item.done
→ response.completed | response.incomplete
```

A failure after the stream opened ends it with `response.failed` carrying the
error inside the response object. There is no `[DONE]`; the terminal event is
the end.

## Messages (Anthropic)

`messages` with `user` and `assistant` roles, content a string or blocks of
type `text`, `tool_use` (assistant), `tool_result` (user; `is_error` prefixes
the result with `Error:`), `thinking` and `redacted_thinking` (accepted and
skipped). `system` is a string or text blocks. Consecutive user turns combine
into one, as the API documents. A trailing assistant message (prefill) is
refused. `tools` are `{name, description, input_schema}`; `cache_control`,
`strict`, `input_examples` and the other tool decorations are accepted and
ignored. `stop_sequences`, `temperature` (0–1), `top_p`, `top_k`, `metadata`
and `service_tier` are honoured or accepted as the real API does.

**The object**: `{id: "msg_…", type: "message", role: "assistant", model,
content: [text | tool_use …], stop_reason, stop_sequence, usage}` with
`stop_reason` one of `end_turn`, `max_tokens`, `stop_sequence` (and the
string in `stop_sequence`), `tool_use`.

**Streaming**:

```
message_start → content_block_start (text) → content_block_delta (text_delta) …
→ content_block_stop → [per call] content_block_start (tool_use)
→ content_block_delta (input_json_delta) … → content_block_stop
→ message_delta (stop_reason, stop_sequence, usage) → message_stop
```

`ping` events keep a slow first token alive. A failure sends
`event: error` with `{"type":"error","error":{...}}`. Errors on every
Anthropic route use that envelope with the API's types and codes:
`invalid_request_error` 400, `not_found_error` 404, `overloaded_error` 529
when the queue is full, `api_error` 5xx.

`POST /v1/messages/count_tokens` returns `{"input_tokens": N}` for the prompt
as the chat template renders it, from the model's own tokenizer. A backend
without a tokenizer (a test double) answers 501.

## Testing

```bash
swift test --filter "ResponsesAPI|AnthropicMapper|AnthropicMessages|HTTPServerTests|OpenAIValidation"
```

The HTTP suites drive real sockets against scripted backends and check the
event grammars event by event; the mapper suites cover the request grammars
and the refusals. None of them needs a model.
