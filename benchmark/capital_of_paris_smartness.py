#!/usr/bin/env python3
"""Every served model, one or more prompts, TTFT and tok/s from the SSE stream.

Warm-up request per *model change* (loads or switches it, discarded), then one
measured streaming request per (model, prompt, repeat). A repeat of the model
already resident is measured back to back, and the row says which it was, so a
cold load can be told apart from the steady state. One JSON object per run is
appended to $RESULTS as it goes, so progress is visible while it works.

The run behind `capital_of_paris_smartness.md`:

    .build/release/NVMAIServer --models-dir models \
        --model qwen3.5-2b_4-Bit --port 8091 --reasoning off

    PORT=8091 MAXTOK=128 REPEATS=3 \
        PROMPTS='["Capital of Paris","What is the capital of France? Answer in one short sentence."]' \
        RESULTS=/tmp/smartness_results.jsonl RUNS="$(cat runs.json)" \
        python3 benchmark/capital_of_paris_smartness.py

`RUNS` is `[[id, engine, label, quant], ...]`; the dense installs are named
`<id>@cpu` for the CPU engine and by their bare id for the GPU. `PROMPTS` is a
JSON array of prompts; a bare `PROMPT` string is still accepted and is one.
"""
import json, os, time, urllib.request, urllib.error

PORT = int(os.environ.get("PORT", "8091"))
if "PROMPTS" in os.environ:
    PROMPTS = json.loads(os.environ["PROMPTS"])
else:
    PROMPTS = [os.environ.get("PROMPT", "Capital of Paris")]
MAXTOK = int(os.environ.get("MAXTOK", "128"))
REPEATS = int(os.environ.get("REPEATS", "1"))
RESULTS = os.environ.get("RESULTS", "/tmp/smartness_results.jsonl")
RUNS = json.loads(os.environ["RUNS"])   # [[id, engine, model_label, quant], ...]

BASE = f"http://127.0.0.1:{PORT}"


def post(payload, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def warm(model, prompt):
    """Load or switch to the model, with the request that is then measured.

    The prompt is the one under test, not a placeholder: prefill for a long
    prompt is part of what the request costs, and warming with a different
    prompt would leave the KV cache holding the wrong prefix.
    """
    t0 = time.monotonic()
    with post({"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1, "temperature": 0}) as r:
        r.read()
    return time.monotonic() - t0


def measure(model, prompt):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAXTOK, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t_send = time.monotonic()
    ttft = None
    t_last = t_send
    content, reasoning, usage, finish = "", "", None, None
    with post(payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece_c = delta.get("content") or ""
                piece_r = delta.get("reasoning_content") or ""
                if (piece_c or piece_r) and ttft is None:
                    ttft = time.monotonic() - t_send
                if piece_c or piece_r:
                    t_last = time.monotonic()
                content += piece_c
                reasoning += piece_r
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.monotonic() - t_send
    tokens = (usage or {}).get("completion_tokens")
    decode = None
    if tokens and ttft is not None and t_last > t_send + ttft:
        decode = (tokens - 1) / (t_last - (t_send + ttft))
    return {"ttft_s": ttft, "total_s": total, "completion_tokens": tokens,
            "decode_tok_s": decode, "e2e_tok_s": (tokens / total) if tokens else None,
            "finish": finish, "content": content, "reasoning": reasoning,
            "content_chars": len(content), "reasoning_chars": len(reasoning)}


def main():
    resident = None
    for model, engine, label, quant in RUNS:
        for prompt in PROMPTS:
            for repeat in range(1, REPEATS + 1):
                row = {"model": model, "engine": engine, "label": label,
                       "quant": quant, "prompt": prompt, "repeat": repeat}
                switched = model != resident
                try:
                    # Warm only on a switch, so a repeat measures the resident
                    # model rather than another load.
                    row["load_s"] = round(warm(model, prompt), 2) if switched else 0.0
                    row["cold"] = switched
                    if switched:
                        resident = model
                    row.update(measure(model, prompt))
                    row["status"] = "ok"
                except urllib.error.HTTPError as e:
                    row["status"] = "http_error"
                    row["error"] = f"{e.code} {e.read()[:300].decode('utf-8','replace')}"
                except Exception as e:                                    # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(e).__name__}: {e}"
                for k in ("ttft_s", "total_s", "decode_tok_s", "e2e_tok_s"):
                    if isinstance(row.get(k), float):
                        row[k] = round(row[k], 3)
                with open(RESULTS, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{row['status']:10s} {label:26s} {quant}-bit {engine:3s} "
                      f"repeat={repeat} cold={str(row.get('cold')):5s} "
                      f"load={row.get('load_s')}s ttft={row.get('ttft_s')}s "
                      f"tok/s={row.get('decode_tok_s')} tokens={row.get('completion_tokens')} "
                      f"content={row.get('content_chars')}ch reasoning={row.get('reasoning_chars')}ch "
                      f"prompt={prompt[:28]!r}",
                      flush=True)
                if row["status"] != "ok":
                    print("           " + str(row.get("error"))[:300], flush=True)


main()
