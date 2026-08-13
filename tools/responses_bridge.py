#!/usr/bin/env python3
"""Minimal OpenAI Responses-API -> Chat-Completions bridge for NVMAI.

Codex CLI (>= ~0.101) only speaks the Responses API (/v1/responses); NVMAI's
server implements chat completions. This proxy translates one to the other:

    codex  ->  responses-bridge (SSE responses events)  ->  NVMAI /v1/chat/completions

Usage:
    python3 responses_bridge.py --port 8130 --upstream http://127.0.0.1:8081/v1/chat/completions

Then point codex at it:
    [model_providers.nvmai]
    base_url = "http://127.0.0.1:8130/v1"
    wire_api = "responses"

The bridge maps Responses input items (message / function_call /
function_call_output) and tools onto chat messages, forwards NVMAI's SSE
stream, and re-emits it as Responses-API stream events (response.created,
content deltas, function_call argument deltas, response.completed).
"""
import json
import sys
import time
import urllib.request
import urllib.error
import uuid
import http.server

DEFAULT_PORT = 8130


def chat_tools(resp_tools):
    out = []
    for t in resp_tools or []:
        if t.get("type") != "function":
            continue
        fn = t.get("name", "")
        params = t.get("parameters", {"type": "object", "properties": {}})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": fn,
                "description": t.get("description", ""),
                "parameters": params,
            },
        })
    return out


def chat_messages(instructions, input_items):
    """Responses input -> chat messages.

    NVMAI's chat template requires exactly one leading system message and
    rejects the developer role, so instructions and developer guidance are
    merged into a single opening system message.
    """
    system_parts = []
    if instructions:
        system_parts.append(instructions)
    messages = []
    for item in input_items or []:
        t = item.get("type")
        if t == "message":
            role = item.get("role", "user")
            content = item.get("content", [])
            if isinstance(content, str):
                text = content
            else:
                text = "".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "input_text")
            if role in ("system", "developer"):
                if text:
                    system_parts.append(text)
            else:
                messages.append({"role": role, "content": text})
        elif t == "function_call":
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": item.get("call_id") or "call_" + uuid.uuid4().hex[:12],
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "{}",
                    },
                }],
            })
        elif t == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": item.get("output") or "",
            })
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return messages


def build_chat_request(body):
    max_out = body.get("max_output_tokens")
    req = {
        "model": body.get("model"),
        "messages": chat_messages(body.get("instructions"), body.get("input")),
        "stream": True,
        "temperature": 0.2,
        # Bound generation so a runaway reply cannot hog the server for many
        # minutes on a slow local model; codex's max_output_tokens wins. A
        # caller's 0 means "no cap" — forward null, not the 2048 default.
        "max_tokens": None if max_out == 0 else (max_out or 2_048),
    }
    tools = chat_tools(body.get("tools"))
    if tools:
        req["tools"] = tools
        req["tool_choice"] = "auto"
    return req


class BridgeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/responses":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self.send_json_error(400, "invalid_json", "malformed JSON request: %s" % exc)
            return

        chat_req = build_chat_request(body)
        if not chat_req.get("model"):
            self.send_json_error(400, "invalid_request_error", "missing model")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.stream_responses(body, chat_req)
        except Exception as exc:  # upstream failure mid-stream
            self.emit_error(str(exc))
        self.wfile.flush()

    def send_json_error(self, status, code, message):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(
            {"error": {"code": code, "message": message,
                       "type": "invalid_request_error"}}).encode())
        self.wfile.flush()

    def emit(self, event_type, data):
        self.wfile.write(("event: %s\ndata: %s\n\n" % (event_type, json.dumps(data))).encode())
        self.wfile.flush()

    def emit_error(self, message):
        self.emit("error", {"code": "upstream_error", "message": message, "type": "error"})

    def stream_responses(self, resp_body, chat_req):
        rid = "resp_" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = chat_req.get("model")
        response = {
            "id": rid, "object": "response", "created_at": created,
            "status": "in_progress", "error": None, "incomplete_details": None,
            "instructions": None, "max_output_tokens": None, "model": model,
            "output": [], "parallel_tool_calls": resp_body.get("parallel_tool_calls", False),
            "previous_response_id": None, "reasoning": {"effort": None, "summary": None},
            "store": False, "temperature": 0.2, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 0.95, "truncation": None,
            "usage": None, "user": None, "metadata": {},
        }
        self.emit("response.created", {"type": "response.created", "response": response})
        self.emit("response.in_progress", {"type": "response.in_progress", "response": response})

        request = urllib.request.Request(
            self.server.upstream_url, method="POST",
            data=json.dumps(chat_req).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream"})
        try:
            with urllib.request.urlopen(request, timeout=1200) as upstream:
                self.forward_chat_stream(upstream, rid, created, model, response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            self.emit_error("upstream HTTP %s: %s" % (exc.code, detail))
            return
        except urllib.error.URLError as exc:
            self.emit_error("upstream unreachable: %s" % exc.reason)
            return

    def forward_chat_stream(self, upstream, rid, created, model, response):
        output_index = 0
        usage = None
        text_parts = []
        self.emit("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": rid + "_msg0", "type": "message", "role": "assistant",
                     "status": "in_progress", "content": []}})
        for raw in upstream:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
                self.emit_output_text_delta(rid, output_index, delta["content"])
                self.emit_content_part_delta(rid, output_index, delta["content"])
            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0) + output_index
                if tc.get("id"):
                    self.emit_function_call_added(rid, index, tc["id"],
                                                  tc.get("function", {}).get("name", ""))
                if tc.get("function", {}).get("arguments"):
                    self.emit("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": rid + "_fc%d" % index,
                        "output_index": index,
                        "delta": tc["function"]["arguments"],
                    })
        full_text = "".join(text_parts)
        self.emit("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 0,
            "item": {"id": rid + "_msg0", "type": "message", "role": "assistant",
                     "status": "completed",
                     "content": [{"type": "output_text", "text": full_text, "annotations": []}]}})
        response["status"] = "completed"
        response["output"] = [{
            "id": rid + "_msg0", "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": full_text, "annotations": []}]}]
        if usage:
            response["usage"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": usage.get("completion_tokens", 0),
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.get("total_tokens", 0),
            }
        self.emit("response.completed", {"type": "response.completed", "response": response})

    def emit_output_text_delta(self, rid, output_index, text):
        self.emit("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": rid + "_msg0", "output_index": output_index,
            "content_index": 0, "delta": text})

    def emit_content_part_delta(self, rid, output_index, text):
        self.emit("response.content_part.delta", {
            "type": "response.content_part.delta",
            "item_id": rid + "_msg0", "output_index": output_index,
            "content_index": 0,
            "delta": {"type": "output_text", "text": text, "annotations": []}})

    def emit_function_call_added(self, rid, output_index, call_id, name):
        self.emit("response.output_item.added", {
            "type": "response.output_item.added", "output_index": output_index,
            "item": {"id": rid + "_fc%d" % output_index, "type": "function_call",
                     "status": "in_progress", "name": name,
                     "arguments": "", "call_id": call_id, "output_index": output_index}})


def main():
    port = DEFAULT_PORT
    upstream = None
    args = sys.argv[1:]
    while args:
        flag = args.pop(0)
        if flag == "--port":
            port = int(args.pop(0))
        elif flag == "--upstream":
            upstream = args.pop(0)
        elif flag in ("-h", "--help"):
            print(__doc__)
            return
        else:
            print("unknown flag %s" % flag, file=sys.stderr)
            return
    if not upstream:
        print("--upstream is required (NVMAI chat-completions URL)", file=sys.stderr)
        return
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), BridgeHandler)
    server.upstream_url = upstream
    print("responses bridge listening on 127.0.0.1:%d -> %s" % (port, upstream), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
