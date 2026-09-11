#!/usr/bin/env python3
"""Turn the smartness matrix's raw rows into its markdown report.

    python3 benchmark/capital_of_paris_report.py results.jsonl > \
        benchmark/capital_of_paris_smartness.md

Reads the JSONL that `capital_of_paris_smartness.py` writes: one object per
(model, prompt, repeat). Groups by (label, quant, engine, prompt), reports the
mean and range of TTFT, decode rate and load across repeats, and says how many
repeats answered in `content` -- the column that matters for a model that
thinks with the switch off.

Every number in the tables comes from the rows. The prose around them is fixed
and says which session's rows it is describing, because the honest comparison
across sessions (the 167 s outlier, the cold load) is not derivable from one
file.

Rows that predate `repeat`/`content_chars` still render: the fields are derived
when missing, so an archived file can be regenerated rather than lost.
"""
import json
import os
import sys
from collections import OrderedDict

DEVICE = os.environ.get(
    "DEVICE", "macOS 26.6.2, Swift 6.3.3, Apple M3 24 GB")
SERVER = os.environ.get(
    "SERVER", ".build/release/NVMAIServer --models-dir models "
              "--model qwen3.5-2b_4-Bit --port 8091 --reasoning off")
COMMIT = os.environ.get("COMMIT", "the commit this report is committed with")


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row.setdefault("repeat", 1)
                row.setdefault("content_chars", len(row.get("content") or ""))
                row.setdefault("reasoning_chars", len(row.get("reasoning") or ""))
                row.setdefault("cold", True)
                rows.append(row)
    return rows


def stat(values):
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return (sum(clean) / len(clean), min(clean), max(clean))


def cell(stats, unit="", decimals=2):
    if stats is None:
        return "-"
    mean, low, high = stats
    fmt = f"{{:.{decimals}f}}"
    if abs(high - low) < 1e-9:
        return fmt.format(mean) + unit
    return f"{fmt.format(mean)}{unit} ({fmt.format(low)}-{fmt.format(high)})"


def by_key(rows):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(
            (row["label"], row["quant"], row["engine"], row["prompt"]), []).append(row)
    return groups


def answered(rows):
    return sum(1 for r in rows if r.get("content_chars", 0) > 0)


def table(groups, prompt):
    out = ["| Model | Quant | Engine | Repeats | Answered | Cold load s | TTFT s | Decode tok/s | End-to-end tok/s | Tokens | Finish |",
           "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for (label, quant, engine, row_prompt), rows in groups.items():
        if row_prompt != prompt:
            continue
        ok = [r for r in rows if r["status"] == "ok"]
        finishes = sorted({r.get("finish") for r in ok})
        out.append("| {} | {}-bit | {} | {} | {}/{} | {} | {} | {} | {} | {} | `{}` |".format(
            label, quant, engine, len(rows), answered(ok), len(rows),
            cell(stat([r.get("load_s") for r in ok if r.get("cold")])),
            cell(stat([r.get("ttft_s") for r in ok]), decimals=3),
            cell(stat([r.get("decode_tok_s") for r in ok])),
            cell(stat([r.get("e2e_tok_s") for r in ok])),
            cell(stat([r.get("completion_tokens") for r in ok]), decimals=0),
            "/".join(f or "-" for f in finishes)))
    return "\n".join(out)


def thinkers_table(rows):
    out = ["| Model | Quant | Engine | Prompt | Repeats | Reasoning chars | Answered | Finish |",
           "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |"]
    groups = by_key([r for r in rows if r["status"] == "ok"
                     and r.get("reasoning_chars", 0) > 0])
    for (label, quant, engine, prompt), group_rows in groups.items():
        out.append("| {} | {}-bit | {} | `{}` | {} | {} | {}/{} | `{}` |".format(
            label, quant, engine, prompt, len(group_rows),
            cell(stat([r.get("reasoning_chars") for r in group_rows]), decimals=0),
            answered(group_rows), len(group_rows),
            "/".join(sorted({r.get("finish") for r in group_rows}))))
    return "\n".join(out)


def verbatim(rows, prompt, repeat=1):
    out = []
    groups = by_key([r for r in rows
                     if r["status"] == "ok" and r["prompt"] == prompt
                     and r.get("repeat", 1) == repeat])
    for (label, quant, engine, _), group_rows in groups.items():
        row = group_rows[0]
        repeats = len([r for r in rows if r["label"] == label and r["quant"] == quant
                       and r["engine"] == engine and r["prompt"] == prompt])
        same = len({" ".join(r["content"].split())
                    for r in rows if r["label"] == label and r["quant"] == quant
                    and r["engine"] == engine and r["prompt"] == prompt}) == 1
        note = f"identical in all {repeats} repeats" if same and repeats > 1 else \
               "varies between repeats" if repeats > 1 else ""
        out.append(f"#### {label} {quant}-bit, {engine}"
                   + (f" — {note}" if note else ""))
        out.append(f"`finish: {row.get('finish')}`, {row.get('completion_tokens')} tokens, "
                   f"TTFT {row.get('ttft_s')} s, decode {row.get('decode_tok_s')} tok/s\n")
        out.append("```text\n" + (row.get("content") or "") + "\n```")
        if row.get("reasoning"):
            out.append("Reasoning (`reasoning_content`):\n")
            out.append("```text\n" + row["reasoning"] + "\n```")
        out.append("")
    return "\n".join(out)


def main():
    rows = load(sys.argv[1:] or ["/tmp/smartness_v2.jsonl"])
    prompts = list(OrderedDict((r["prompt"], None) for r in rows))
    groups = by_key(rows)
    repeats = max((r.get("repeat", 1) for r in rows), default=1)
    models = len({(r["label"], r["quant"], r["engine"]) for r in rows})
    ok = sum(1 for r in rows if r["status"] == "ok")

    print("# \"Capital of Paris\" on every served model and engine\n")
    print("A fixed, deliberately ambiguous prompt -- *\"Capital of Paris\"* -- sent to")
    print(f"every model the local server serves, on every engine it serves it on, plus a")
    print("plain control question, with **thinking off**. The tables are generated from")
    print("the raw rows by `benchmark/capital_of_paris_report.py`; the harness that")
    print("produced them is `benchmark/capital_of_paris_smartness.py`, and the rows")
    print("themselves are in the gitignored")
    print("`benchmark/benchmark-results/capital-of-paris-20260911T1935/` "
          "(`results-v2-3x2.jsonl`")
    print("for this report, `results.jsonl` for the single-pass first one).\n")

    print("## Protocol\n")
    print(f"- Commit `{COMMIT}`; `{DEVICE}`.")
    print(f"- Server: `{SERVER}`")
    print("- Request: `POST /v1/chat/completions`, `temperature: 0`, `max_tokens: 128`, "
          "`stream: true` with usage, `thinking off`.")
    print(f"- {models} model/engine combinations x {len(prompts)} prompts x "
          f"{repeats} repeats = **{len(rows)} measured runs**, {ok} of them ok.")
    print("- One model resident at a time. A warm-up request runs only when the model "
          "*changes*, with the prompt that is then measured, so a repeat is measured "
          "on the resident model; `Cold load` is that warm-up, blank for repeats.")
    print("- The three MTP draft heads are not standalone models -- the catalog "
          "refuses them (`served only beside its target`) -- so they are not rows.")
    print("- Times come from the SSE stream: TTFT is the first `content` or "
          "`reasoning_content` delta, decode rate is `(completion_tokens - 1)` over "
          "the time between that delta and the last one.\n")

    for index, prompt in enumerate(prompts):
        label = "the ambiguous prompt" if index == 0 else "the plain control question"
        print(f"## Prompt {index + 1} ({label}): `{prompt}`\n")
        print(f"{repeats} repeats per model. `Answered` counts runs whose `content` was")
        print("non-empty: a model that thinks with the switch off puts its thought in")
        print("`reasoning_content`, so it does not count here even though it did reply.\n")
        print(table(groups, prompt))
        print()

    print("## The model thought although the request said not to\n")
    print("The whole matrix ran at `--reasoning off`. Every run that produced reasoning")
    print("anyway is below; a model absent from this table produced none.\n")
    print(thinkers_table(rows))
    print()
    print("One install, on one prompt, every time: the **8-bit AgentWorld**, on the")
    print("ambiguous phrasing only -- its own control-prompt runs are clean (31")
    print("characters of answer, no reasoning, 3/3). Its 4-bit sibling never does it.")
    print("Because the prompt closes the block and the template is byte-identical")
    print("between the two installs, this is the weights answering a false-premise")
    print("question with a long deliberation; *why* the two widths differ is not")
    print("established.\n")
    print("**What the runtime does with it (C90).** The thought is split into")
    print("`reasoning_content`, the answer channel stays empty rather than carrying the")
    print("scaffold, and the server logs `thinking off, but the model wrote N characters")
    print("of reasoning` on the request's completion line. Before that fix the scaffold")
    print("was `content`: a client capping `max_tokens` received a thinking transcript")
    print("where it expected an answer. What the fix does not do is make the answer")
    print("arrive -- 128 tokens is not enough for this install on this prompt, and 512")
    print("is (it stops by itself at 482).\n")

    print("## What the repeats changed about the earlier, single-run numbers\n")
    print("- **The 9B 8-bit CPU row's 167 s TTFT was a one-off, and the load is where")
    print("  the variance lives.** The first matrix measured 53.8 s load and 167.4 s")
    print("  TTFT for it; this session's cold load was 46.4 s with **2.6-2.9 s** TTFT")
    print("  and 3.8-5.0 tok/s across the three repeats, and its GPU row is stable at")
    print("  1.46-1.48 s TTFT and 8.8-9.0 tok/s. The first fault-in of the 9.5 GB")
    print("  snapshot is the expensive moment, not the request.")
    print("- **The 2B 4-bit \"degenerate\" reply is deterministic, not noise.** All three")
    print("  repeats produced the same text on the GPU path, the same install on the CPU")
    print("  answered cleanly (51 tokens), and the plain control prompt answered cleanly")
    print("  on both (8-9 tokens). So it is one install, one prompt, one engine -- and")
    print("  the two engines only agree on the fact, not the wording, at 4-bit.")
    print("- **GPU decode rates are tight; the CPU engine's are not.** The 9B at 4-bit")
    print("  spans 14.81-15.02 tok/s on the GPU and 3.62-4.33 on the CPU; the 2B at")
    print("  4-bit spans 52.0-54.6 and 11.9-18.9. A page-faulting CPU engine on a shared")
    print("  machine is the wider instrument.")
    print("- **Cold loads are the least reproducible number in the report** -- 16.2 s,")
    print("  46.4 s and 53.8 s for the same 9B 8-bit CPU install across sessions -- which")
    print("  is why the repeats do not include one: a switch is warmed once and then")
    print("  measured.\n")

    print("## Rows worth looking at\n")
    empty = [r for r in rows if r["status"] == "ok" and r.get("content_chars", 0) == 0]
    if empty:
        print(f"- **{len(empty)} run(s) produced no answer in `content`** -- the thought "
              "used the whole budget before the answer began:")
        for r in empty:
            print(f"  - {r['label']} {r['quant']}-bit {r['engine']}, "
                  f"{r['reasoning_chars']} characters of reasoning, "
                  f"finish `{r.get('finish')}`, prompt `{r['prompt']}`")
    truncated = [r for r in rows if r.get("finish") == "length"]
    if truncated:
        print(f"- **{len(truncated)} run(s) hit the token cap** (`finish_reason: length`).")
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"- **{len(failed)} run(s) failed.**" if failed else
          "- **No run failed**: every request was served, including the ones whose "
          "answer was empty.")
    for r in failed:
        print(f"  - {r['label']} {r['quant']}-bit {r['engine']}: {r.get('error')}")

    for index, prompt in enumerate(prompts):
        print(f"\n## Every reply, verbatim: prompt {index + 1}\n")
        print(verbatim(rows, prompt))


main()
