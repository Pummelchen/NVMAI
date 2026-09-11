# "Capital of Paris" on every served model and engine

A fixed, deliberately ambiguous prompt -- *"Capital of Paris"* -- sent to
every model the local server serves, on every engine it serves it on, plus a
plain control question, with **thinking off**. The tables are generated from
the raw rows by `benchmark/capital_of_paris_report.py`; the harness that
produced them is `benchmark/capital_of_paris_smartness.py`.

## Protocol

- Commit `c7fca53`; `macOS 26.6.2, Swift 6.3.3, Apple M3 24 GB`.
- Server: `.build/release/NVMAIServer --models-dir models --model qwen3.5-2b_4-Bit --port 8091 --reasoning off`
- Request: `POST /v1/chat/completions`, `temperature: 0`, `max_tokens: 128`, `stream: true` with usage, `thinking off`.
- 20 model/engine combinations x 2 prompts x 3 repeats = **120 measured runs**, 120 of them ok.
- One model resident at a time. A warm-up request runs only when the model *changes*, with the prompt that is then measured, so a repeat is measured on the resident model; `Cold load` is that warm-up, blank for repeats.
- The three MTP draft heads are not standalone models -- the catalog refuses them (`served only beside its target`) -- so they are not rows.
- Times come from the SSE stream: TTFT is the first `content` or `reasoning_content` delta, decode rate is `(completion_tokens - 1)` over the time between that delta and the last one.

## Prompt 1 (the ambiguous prompt): `Capital of Paris`

3 repeats per model. `Answered` counts runs whose `content` was
non-empty: a model that thinks with the switch off puts its thought in
`reasoning_content`, so it does not count here even though it did reply.

| Model | Quant | Engine | Repeats | Answered | Cold load s | TTFT s | Decode tok/s | End-to-end tok/s | Tokens | Finish |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen 3.5 2B | 4-bit | gpu | 3 | 3/3 | 0.21 | 0.172 (0.171-0.174) | 53.73 (52.03-54.58) | 44.62 (43.50-45.29) | 46 | `stop` |
| Qwen 3.5 2B | 4-bit | cpu | 3 | 3/3 | 1.01 | 0.829 (0.656-0.977) | 15.42 (11.87-18.95) | 12.37 (9.92-15.23) | 51 | `stop` |
| Qwen 3.5 2B | 8-bit | gpu | 3 | 3/3 | 1.27 | 0.331 (0.330-0.333) | 32.77 (31.10-33.94) | 25.58 (24.57-26.29) | 39 | `stop` |
| Qwen 3.5 2B | 8-bit | cpu | 3 | 3/3 | 2.65 | 0.695 (0.620-0.819) | 15.83 (14.52-16.52) | 12.45 (11.81-13.19) | 40 | `stop` |
| Qwen 3.5 4B | 4-bit | gpu | 3 | 3/3 | 2.57 | 0.440 (0.434-0.445) | 26.18 (25.85-26.41) | 20.05 (19.92-20.18) | 38 | `stop` |
| Qwen 3.5 4B | 4-bit | cpu | 3 | 3/3 | 5.71 | 1.664 (1.637-1.713) | 7.71 (7.57-7.98) | 5.84 (5.77-5.95) | 40 | `stop` |
| Qwen 3.5 4B | 8-bit | gpu | 3 | 3/3 | 3.70 | 0.792 (0.791-0.794) | 16.14 (16.11-16.17) | 13.37 (13.35-13.38) | 62 | `stop` |
| Qwen 3.5 4B | 8-bit | cpu | 3 | 3/3 | 8.32 | 2.263 (1.772-3.211) | 7.04 (6.37-7.42) | 5.67 (4.81-6.12) | 63 | `stop` |
| Qwen 3.5 9B | 4-bit | gpu | 3 | 3/3 | 5.30 | 0.828 (0.795-0.890) | 14.93 (14.81-15.02) | 13.08 (12.87-13.21) | 88 | `stop` |
| Qwen 3.5 9B | 4-bit | cpu | 3 | 3/3 | 9.48 | 3.098 (3.033-3.139) | 4.07 (3.62-4.33) | 3.28 (2.92-3.50) | 55 | `stop` |
| Qwen 3.5 9B | 8-bit | gpu | 3 | 3/3 | 7.59 | 1.466 (1.456-1.477) | 8.90 (8.85-8.99) | 7.46 (7.43-7.53) | 68 | `stop` |
| Qwen 3.5 9B | 8-bit | cpu | 3 | 3/3 | 46.39 | 2.740 (2.569-2.855) | 4.51 (3.83-4.96) | 3.82 (3.30-4.19) | 69 | `stop` |
| Ornith 1.5 35B-A3B | 4-bit | gpu | 3 | 3/3 | 5.61 | 0.399 (0.373-0.432) | 19.37 (12.59-22.81) | 16.50 (11.32-19.18) | 46 | `stop` |
| Ornith 1.5 35B-A3B | 8-bit | gpu | 3 | 3/3 | 5.92 | 1.103 (0.739-1.316) | 10.80 (8.62-12.06) | 9.29 (7.99-10.10) | 78 | `stop` |
| Qwen AgentWorld 35B-A3B | 4-bit | gpu | 3 | 3/3 | 3.90 | 0.361 (0.347-0.387) | 23.06 (12.54-28.32) | 10.87 (7.58-12.54) | 8 | `stop` |
| Qwen AgentWorld 35B-A3B | 8-bit | gpu | 3 | 0/3 | 5.50 | 1.135 (0.819-1.299) | 10.91 (9.45-11.67) | 9.98 (8.97-10.51) | 128 | `length` |
| Qwen 3.6 35B-A3B | 4-bit | gpu | 3 | 3/3 | 3.90 | 0.365 (0.361-0.371) | 19.49 (14.30-22.98) | 16.91 (12.91-19.58) | 48 | `stop` |
| Qwen 3.6 35B-A3B | 8-bit | gpu | 3 | 3/3 | 6.39 | 0.803 (0.746-0.876) | 11.44 (8.53-13.06) | 9.52 (7.55-10.70) | 47 | `stop` |
| Qwen 3.8 Flash Next 125B-A6B | 4-bit | gpu | 3 | 3/3 | 8.50 | 2.816 (1.121-3.871) | 3.58 (2.78-4.09) | 2.59 (1.99-3.33) | 27 | `stop` |
| Qwen 3.8 Flash Next 125B-A6B | 8-bit | gpu | 3 | 3/3 | 12.91 | 5.376 (4.846-6.404) | 2.37 (2.02-2.58) | 1.60 (1.35-1.75) | 27 | `stop` |

## Prompt 2 (the plain control question): `What is the capital of France? Answer in one short sentence.`

3 repeats per model. `Answered` counts runs whose `content` was
non-empty: a model that thinks with the switch off puts its thought in
`reasoning_content`, so it does not count here even though it did reply.

| Model | Quant | Engine | Repeats | Answered | Cold load s | TTFT s | Decode tok/s | End-to-end tok/s | Tokens | Finish |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen 3.5 2B | 4-bit | gpu | 3 | 3/3 | - | 0.270 (0.269-0.273) | 60.60 (59.73-61.29) | 19.70 (19.57-19.88) | 8 | `stop` |
| Qwen 3.5 2B | 4-bit | cpu | 3 | 3/3 | - | 1.222 (1.078-1.405) | 18.23 (17.36-19.08) | 5.28 (4.80-5.74) | 9 | `stop` |
| Qwen 3.5 2B | 8-bit | gpu | 3 | 3/3 | - | 0.591 (0.584-0.604) | 36.64 (36.34-36.80) | 9.81 (9.66-9.89) | 8 | `stop` |
| Qwen 3.5 2B | 8-bit | cpu | 3 | 3/3 | - | 1.430 (1.091-2.051) | 14.19 (9.82-18.89) | 4.56 (3.06-5.55) | 9 | `stop` |
| Qwen 3.5 4B | 4-bit | gpu | 3 | 3/3 | - | 0.722 (0.704-0.744) | 30.25 (29.73-30.67) | 8.04 (7.86-8.22) | 8 | `stop` |
| Qwen 3.5 4B | 4-bit | cpu | 3 | 3/3 | - | 2.934 (2.889-2.983) | 7.63 (7.56-7.74) | 2.19 (2.16-2.22) | 9 | `stop` |
| Qwen 3.5 4B | 8-bit | gpu | 3 | 3/3 | - | 1.314 (1.310-1.320) | 18.45 (18.25-18.60) | 4.55 (4.54-4.57) | 8 | `stop` |
| Qwen 3.5 4B | 8-bit | cpu | 3 | 3/3 | - | 3.071 (3.035-3.132) | 6.86 (5.89-7.40) | 2.05 (1.93-2.12) | 9 | `stop` |
| Qwen 3.5 9B | 4-bit | gpu | 3 | 3/3 | - | 1.312 (1.307-1.316) | 17.32 (17.21-17.55) | 4.48 (4.46-4.51) | 8 | `stop` |
| Qwen 3.5 9B | 4-bit | cpu | 3 | 3/3 | - | 6.325 (5.811-6.595) | 3.36 (1.69-4.24) | 0.96 (0.81-1.04) | 9 | `stop` |
| Qwen 3.5 9B | 8-bit | gpu | 3 | 3/3 | - | 2.486 (2.458-2.521) | 10.17 (10.13-10.20) | 2.43 (2.40-2.45) | 8 | `stop` |
| Qwen 3.5 9B | 8-bit | cpu | 3 | 3/3 | - | 6.620 (6.574-6.663) | 3.03 (1.82-3.94) | 0.92 (0.78-1.01) | 9 | `stop` |
| Ornith 1.5 35B-A3B | 4-bit | gpu | 3 | 3/3 | - | 0.718 (0.571-0.909) | 23.44 (19.27-27.69) | 7.67 (5.97-9.19) | 8 | `stop` |
| Ornith 1.5 35B-A3B | 8-bit | gpu | 3 | 3/3 | - | 1.939 (1.776-2.132) | 11.58 (10.13-12.56) | 3.01 (2.71-3.27) | 8 | `stop` |
| Qwen AgentWorld 35B-A3B | 4-bit | gpu | 3 | 3/3 | - | 0.727 (0.535-1.096) | 24.46 (17.15-28.13) | 8.07 (5.05-9.67) | 8 | `stop` |
| Qwen AgentWorld 35B-A3B | 8-bit | gpu | 3 | 3/3 | - | 1.853 (1.736-1.991) | 11.07 (10.55-11.35) | 3.07 (2.87-3.23) | 8 | `stop` |
| Qwen 3.6 35B-A3B | 4-bit | gpu | 3 | 3/3 | - | 0.740 (0.606-1.001) | 22.55 (18.86-26.82) | 7.44 (5.47-8.70) | 8 | `stop` |
| Qwen 3.6 35B-A3B | 8-bit | gpu | 3 | 3/3 | - | 1.591 (1.511-1.647) | 11.64 (10.44-13.22) | 3.42 (3.21-3.69) | 8 | `stop` |
| Qwen 3.8 Flash Next 125B-A6B | 4-bit | gpu | 3 | 3/3 | - | 6.145 (4.459-8.836) | 3.05 (2.06-4.91) | 0.94 (0.63-1.29) | 8 | `stop` |
| Qwen 3.8 Flash Next 125B-A6B | 8-bit | gpu | 3 | 3/3 | - | 9.254 (8.549-9.643) | 1.87 (1.74-1.94) | 0.57 (0.55-0.62) | 8 | `stop` |

## The model thought although the request said not to

The whole matrix ran at `--reasoning off`. Every run that produced reasoning
anyway is below; a model absent from this table produced none.

| Model | Quant | Engine | Prompt | Repeats | Reasoning chars | Answered | Finish |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Qwen AgentWorld 35B-A3B | 8-bit | gpu | `Capital of Paris` | 3 | 477 | 0/3 | `length` |

One install, on one prompt, every time: the **8-bit AgentWorld**, on the
ambiguous phrasing only -- its own control-prompt runs are clean (31
characters of answer, no reasoning, 3/3). Its 4-bit sibling never does it.
Because the prompt closes the block and the template is byte-identical
between the two installs, this is the weights answering a false-premise
question with a long deliberation; *why* the two widths differ is not
established.

**What the runtime does with it (C90).** The thought is split into
`reasoning_content`, the answer channel stays empty rather than carrying the
scaffold, and the server logs `thinking off, but the model wrote N characters
of reasoning` on the request's completion line. Before that fix the scaffold
was `content`: a client capping `max_tokens` received a thinking transcript
where it expected an answer. What the fix does not do is make the answer
arrive -- 128 tokens is not enough for this install on this prompt, and 512
is (it stops by itself at 482).

## What the repeats changed about the earlier, single-run numbers

- **The 9B 8-bit CPU row's 167 s TTFT was a one-off, and the load is where
  the variance lives.** The first matrix measured 53.8 s load and 167.4 s
  TTFT for it; this session's cold load was 46.4 s with **2.6-2.9 s** TTFT
  and 3.8-5.0 tok/s across the three repeats, and its GPU row is stable at
  1.46-1.48 s TTFT and 8.8-9.0 tok/s. The first fault-in of the 9.5 GB
  snapshot is the expensive moment, not the request.
- **The 2B 4-bit "degenerate" reply is deterministic, not noise.** All three
  repeats produced the same text on the GPU path, the same install on the CPU
  answered cleanly (51 tokens), and the plain control prompt answered cleanly
  on both (8-9 tokens). So it is one install, one prompt, one engine -- and
  the two engines only agree on the fact, not the wording, at 4-bit.
- **GPU decode rates are tight; the CPU engine's are not.** The 9B at 4-bit
  spans 14.81-15.02 tok/s on the GPU and 3.62-4.33 on the CPU; the 2B at
  4-bit spans 52.0-54.6 and 11.9-18.9. A page-faulting CPU engine on a shared
  machine is the wider instrument.
- **Cold loads are the least reproducible number in the report** -- 16.2 s,
  46.4 s and 53.8 s for the same 9B 8-bit CPU install across sessions -- which
  is why the repeats do not include one: a switch is warmed once and then
  measured.

## Rows worth looking at

- **3 run(s) produced no answer in `content`** -- the thought used the whole budget before the answer began:
  - Qwen AgentWorld 35B-A3B 8-bit gpu, 477 characters of reasoning, finish `length`, prompt `Capital of Paris`
  - Qwen AgentWorld 35B-A3B 8-bit gpu, 477 characters of reasoning, finish `length`, prompt `Capital of Paris`
  - Qwen AgentWorld 35B-A3B 8-bit gpu, 477 characters of reasoning, finish `length`, prompt `Capital of Paris`
- **3 run(s) hit the token cap** (`finish_reason: length`).
- **No run failed**: every request was served, including the ones whose answer was empty.

## Every reply, verbatim: prompt 1

#### Qwen 3.5 2B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 46 tokens, TTFT 0.174 s, decode 54.566 tok/s

```text
The capital of France, and the capital of the country **France**, is **Paris**.

Paris is the capital and capital city of France. It is the capital of the French Republic and the capital of the French Republic.
```

#### Qwen 3.5 2B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 51 tokens, TTFT 0.855 s, decode 11.874 tok/s

```text
The capital of France is **Paris**.

Located on the left bank of the Seine River, Paris is the largest city in France and the capital of the French Republic. It serves as the political, cultural, and economic center of the country.

```

#### Qwen 3.5 2B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 39 tokens, TTFT 0.331 s, decode 33.944 tok/s

```text
The capital of Paris is **Paris** itself.

Paris is the capital and the most populous city of France. It serves as the political, cultural, and economic center of the country.
```

#### Qwen 3.5 2B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 40 tokens, TTFT 0.647 s, decode 14.515 tok/s

```text
The capital of Paris is **Paris** itself.

Paris is the capital and the most populous city of France. It serves as the political, cultural, and economic center of the country.

```

#### Qwen 3.5 4B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 38 tokens, TTFT 0.44 s, decode 26.411 tok/s

```text
The capital of Paris is **Paris** itself.

Paris is the capital and most populous city of **France**. It is also the capital of the Île-de-France region.
```

#### Qwen 3.5 4B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 40 tokens, TTFT 1.713 s, decode 7.977 tok/s

```text
The capital of **Paris** is **Paris**.

Paris is the capital and most populous city of France. It is located in the north-central part of the country on the Seine River.

```

#### Qwen 3.5 4B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 62 tokens, TTFT 0.791 s, decode 16.145 tok/s

```text
The capital of France is **Paris**.

It is the country's most populous city and its main center of business, education, culture, and the arts. Located on the Seine River, Paris is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
```

#### Qwen 3.5 4B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 63 tokens, TTFT 3.211 s, decode 6.366 tok/s

```text
The capital of France is **Paris**.

It is the country's most populous city and its main center of business, education, culture, and the arts. Located on the Seine River, Paris is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.

```

#### Qwen 3.5 9B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 88 tokens, TTFT 0.89 s, decode 14.814 tok/s

```text
There is a slight confusion in your question. **Paris** is the capital city, not a country.

*   **Paris** is the capital city of **France**.
*   Paris is the capital city of the country of **France**. also the most populous city in France.

If you are looking for the capital of a different country, please let me know the country name, and I will be happy to help!
```

#### Qwen 3.5 9B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 55 tokens, TTFT 3.033 s, decode 4.331 tok/s

```text
There is a slight confusion in your question. **Paris** is the capital city, not the country.

The capital of the country where Paris is located (France) is **Paris**.

*   **Country:** France
*   **Capital City:** Paris

```

#### Qwen 3.5 9B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 68 tokens, TTFT 1.456 s, decode 8.991 tok/s

```text
There is a slight confusion in the phrasing of your question. **Paris** is the capital city, not a country.

Therefore, Paris does not have a capital; rather, **Paris is the capital of France**.

*   **City**: Paris
*   **Country**: France
*   **Role**: Capital City
```

#### Qwen 3.5 9B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 69 tokens, TTFT 2.569 s, decode 4.965 tok/s

```text
There is a slight confusion in the phrasing of your question. **Paris** is the capital city, not a country.

Therefore, Paris does not have a capital; rather, **Paris is the capital of France**.

*   **City**: Paris
*   **Country**: France
*   **Role**: Capital City

```

#### Ornith 1.5 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 46 tokens, TTFT 0.432 s, decode 12.588 tok/s

```text
Paris is itself the capital of France! There's no separate capital city—Paris serves as both the capital and the largest city of France.

Is there something specific about Paris or France you'd like to know more about?
```

#### Ornith 1.5 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 78 tokens, TTFT 0.739 s, decode 8.625 tok/s

```text
Paris itself is the capital of France. So there's a slight mix-up in the question—Paris is the capital, not a country or region that would have its own capital.

To clarify:
- **Paris** is the capital city of **France**
- France's capital: **Paris**

Is there something specific about Paris or France you'd like to know?
```

#### Qwen AgentWorld 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 0.387 s, decode 12.545 tok/s

```text
Paris is the capital of France.
```

#### Qwen AgentWorld 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: length`, 128 tokens, TTFT 0.819 s, decode 9.45 tok/s

```text

```
Reasoning (`reasoning_content`):

```text

Thinking Process:

1.  **Analyze the Request:** The user is asking for the "Capital of Paris".
2.  **Identify the Entities/Concepts:**
    *   "Paris" is a city.
    *   "Capital" refers to the seat of government of a country or region.
3.  **Evaluate the Premise:** Paris *is* a capital city. Specifically, it is the capital of France. The question "Capital of Paris" is based on a false premise or a misunderstanding (Paris is not a country/region that has a capital; it *is
```

#### Qwen 3.6 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 48 tokens, TTFT 0.371 s, decode 14.305 tok/s

```text
Paris **is** the capital of France.

There is no separate "capital of Paris" because Paris itself is a city and the capital city of the country. It serves as the political, economic, and cultural center of France.
```

#### Qwen 3.6 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 47 tokens, TTFT 0.746 s, decode 8.526 tok/s

```text
Paris **is** the capital of France.

It seems there might be a slight confusion in the phrasing. Paris is not the capital *of* Paris; rather, it is the capital city of the country **France**.
```

#### Qwen 3.8 Flash Next 125B-A6B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 27 tokens, TTFT 1.121 s, decode 3.876 tok/s

```text
Paris is the capital of **France**.

It is not a country itself, but rather the capital city of the French Republic.
```

#### Qwen 3.8 Flash Next 125B-A6B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 27 tokens, TTFT 6.404 s, decode 2.024 tok/s

```text
Paris is the capital of **France**.

It is not a country itself, but rather the capital city of the French Republic.
```


## Every reply, verbatim: prompt 2

#### Qwen 3.5 2B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 0.269 s, decode 59.735 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 2B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 1.078 s, decode 18.239 tok/s

```text
The capital of France is Paris.

```

#### Qwen 3.5 2B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 0.604 s, decode 36.801 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 2B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 1.147 s, decode 18.892 tok/s

```text
The capital of France is Paris.

```

#### Qwen 3.5 4B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 0.744 s, decode 30.347 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 4B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 2.983 s, decode 7.563 tok/s

```text
The capital of France is Paris.

```

#### Qwen 3.5 4B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.31 s, decode 18.599 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 4B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 3.035 s, decode 7.397 tok/s

```text
The capital of France is Paris.

```

#### Qwen 3.5 9B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.312 s, decode 17.208 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 9B 4-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 5.811 s, decode 1.692 tok/s

```text
The capital of France is Paris.

```

#### Qwen 3.5 9B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 2.478 s, decode 10.204 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.5 9B 8-bit, cpu — identical in all 3 repeats
`finish: stop`, 9 tokens, TTFT 6.663 s, decode 1.817 tok/s

```text
The capital of France is Paris.

```

#### Ornith 1.5 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 0.909 s, decode 19.27 tok/s

```text
The capital of France is Paris.
```

#### Ornith 1.5 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 2.132 s, decode 10.126 tok/s

```text
The capital of France is Paris.
```

#### Qwen AgentWorld 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.096 s, decode 17.147 tok/s

```text
The capital of France is Paris.
```

#### Qwen AgentWorld 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.991 s, decode 10.55 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.6 35B-A3B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.001 s, decode 18.855 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.6 35B-A3B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 1.647 s, decode 10.443 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.8 Flash Next 125B-A6B 4-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 8.836 s, decode 2.056 tok/s

```text
The capital of France is Paris.
```

#### Qwen 3.8 Flash Next 125B-A6B 8-bit, gpu — identical in all 3 repeats
`finish: stop`, 8 tokens, TTFT 8.549 s, decode 1.935 tok/s

```text
The capital of France is Paris.
```

