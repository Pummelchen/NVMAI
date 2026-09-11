#!/usr/bin/env python3
"""One prompt, every served model, TTFT and tok/s measured from the SSE stream.

Warm-up request per model (loads or switches it, discarded), then one measured
streaming request. Appends one JSON object per run to $RESULTS so progress is
visible while it works.

The run behind `capital_of_paris_smartness.md`:

    .build/release/NVMAIServer --models-dir models \
        --model qwen3.5-2b_4-Bit --port 8091 --reasoning off

    PORT=8091 PROMPT="Capital of Paris" MAXTOK=128 \
        RESULTS=/tmp/smartness_results.jsonl RUNS="$(cat runs.json)" \
        python3 benchmark/capital_of_paris_smartness.py

`RUNS` is `[[id, engine, label, quant], ...]`; the dense installs are named
`<id>@cpu` for the CPU engine and by their bare id for the GPU.
"""
import json, os, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("PORT", "8091"))
PROMPT = os.environ.get("PROMPT", "Capital of Paris")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
RESULTS = os.environ.get("RESULTS", "/tmp/smartness_results.jsonl")
RUNS = json.loads(os.environ["RUNS"])   # [[id, engine, model_label, quant], ...]

BASE = f"http://127.0.0.1:{PORT}"


def post(payload, stream, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def warm(model):
    t0 = time.monotonic()
    with post({"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": 1, "temperature": 0}, stream=False) as r:
        r.read()
    return time.monotonic() - t0


def measure(model):
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": MAXTOK, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t_send = time.monotonic()
    ttft = None
    t_last = t_send
    content, reasoning, usage, finish = "", "", None, None
    with post(payload, stream=True) as r:
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
            "finish": finish, "content": content, "reasoning": reasoning}


def main():
    for model, engine, label, quant in RUNS:
        row = {"model": model, "engine": engine, "label": label, "quant": quant,
               "prompt": PROMPT}
        try:
            row["load_s"] = round(warm(model), 2)
            row.update(measure(model))
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
        print(f"{row['status']:10s} {label:12s} {quant}-bit {engine:3s} "
              f"load={row.get('load_s')}s ttft={row.get('ttft_s')}s "
              f"tok/s={row.get('decode_tok_s')} tokens={row.get('completion_tokens')}",
              flush=True)
        if row["status"] != "ok":
            print("           " + str(row.get("error"))[:300], flush=True)


main()
