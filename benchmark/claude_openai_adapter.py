#!/usr/bin/env python3
"""Minimal Anthropic Messages -> OpenAI Chat adapter for NVMAI benchmarks.

Claude Code speaks the Anthropic Messages API, while NVMAI intentionally
exposes OpenAI-compatible endpoints.  This loopback-only adapter translates
the subset Claude Code uses in non-interactive benchmark runs, including text
and function-tool blocks.  It is not installed as a service and is not a
general Anthropic API compatibility promise.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _system_message(system: Any) -> dict[str, Any] | None:
    text = _text_content(system)
    return {"role": "system", "content": text} if text else None


def _openai_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system = _system_message(body.get("system"))
    system_parts = [system["content"]] if system else []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            guidance = _text_content(content)
            if guidance:
                system_parts.append(guidance)
            continue
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", "call_" + uuid.uuid4().hex),
                    "type": "function",
                    "function": {
                        "name": block.get("name", "tool"),
                        "arguments": json.dumps(block.get("input", {}), separators=(",", ":")),
                    },
                })
            elif block_type == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _text_content(block.get("content", "")),
                })
        if role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                item["tool_calls"] = tool_calls
            result.append(item)
        elif role == "user":
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
            result.extend(tool_results)
    if system_parts:
        result.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return result


def _openai_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in body.get("tools", []):
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return tools


class AdapterHandler(BaseHTTPRequestHandler):
    server_version = "NVMAIClaudeBenchmarkAdapter/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("adapter: " + (fmt % args), file=sys.stderr, flush=True)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] != "/v1/messages":
            self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": "not found"}})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length))
            self._serve_messages(body)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            self._json(exc.code, {"type": "error", "error": {"type": "api_error", "message": detail}})
        except Exception as exc:  # benchmark boundary: preserve exact failure for the report
            self._json(500, {"type": "error", "error": {"type": "api_error", "message": repr(exc)}})

    def _serve_messages(self, body: dict[str, Any]) -> None:
        server = self.server
        assert isinstance(server, AdapterServer)
        model = body.get("model") or server.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": _openai_messages(body),
            "max_tokens": min(int(body.get("max_tokens", 2048)), server.max_tokens),
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "stream": True,
        }
        tools = _openai_tools(body)
        if tools:
            payload["tools"] = tools
        print(
            "adapter request roles="
            + ",".join(str(message.get("role")) for message in payload["messages"])
            + f" tools={len(tools)}",
            file=sys.stderr,
            flush=True,
        )
        request = urllib.request.Request(
            server.openai_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": "Bearer local"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=server.timeout)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        message_id = "msg_" + uuid.uuid4().hex
        self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        text_index: int | None = None
        tool_indexes: dict[int, int] = {}
        open_blocks: list[int] = []
        next_index = 0
        stop_reason = "end_turn"
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if text_index is None:
                    text_index = next_index
                    next_index += 1
                    open_blocks.append(text_index)
                    self._event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                self._event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": text},
                })
            for call in delta.get("tool_calls") or []:
                call_slot = int(call.get("index", 0))
                if call_slot not in tool_indexes:
                    block_index = next_index
                    next_index += 1
                    tool_indexes[call_slot] = block_index
                    open_blocks.append(block_index)
                    function = call.get("function") or {}
                    self._event("content_block_start", {
                        "type": "content_block_start", "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.get("id") or "call_" + uuid.uuid4().hex,
                            "name": function.get("name") or "tool",
                            "input": {},
                        },
                    })
                arguments = (call.get("function") or {}).get("arguments")
                if arguments:
                    self._event("content_block_delta", {
                        "type": "content_block_delta", "index": tool_indexes[call_slot],
                        "delta": {"type": "input_json_delta", "partial_json": arguments},
                    })
            finish = choice.get("finish_reason")
            if finish == "tool_calls":
                stop_reason = "tool_use"
            elif finish == "length":
                stop_reason = "max_tokens"
        for index in open_blocks:
            self._event("content_block_stop", {"type": "content_block_stop", "index": index})
        self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        self._event("message_stop", {"type": "message_stop"})

    def _event(self, name: str, value: dict[str, Any]) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(value, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def _json(self, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AdapterServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], openai_url: str, model: str,
                 timeout: int, max_tokens: int) -> None:
        super().__init__(address, AdapterHandler)
        self.openai_url = openai_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--openai-url", required=True, help="NVMAI base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    server = AdapterServer(("127.0.0.1", args.port), args.openai_url, args.model,
                           args.timeout, args.max_tokens)
    print(f"Claude adapter ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
