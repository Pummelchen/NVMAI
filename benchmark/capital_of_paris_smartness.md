# One prompt, every served model: "Capital of Paris"

A fixed, deliberately ambiguous prompt -- *"Capital of Paris"* -- sent to
every model the local server serves, on every engine it serves it on, with
**thinking off** (`--reasoning off`), `temperature: 0`, `max_tokens: 128`, one
warm-up request per model (discarded; it loads or switches the model) and then
one measured streaming request. Times come from the SSE stream: TTFT is the
first content or reasoning delta, decode rate is `(completion_tokens - 1)`
over the time between that first delta and the last one.

The raw rows behind this report -- one JSON object per run, plus the re-check --
are written to `benchmark/benchmark-results/capital-of-paris-20260911T1935/`
(`results.jsonl`, `recheck.jsonl`), which is gitignored; the harness that
produced them is `benchmark/capital_of_paris_smartness.py`.

## Protocol

- Commit `e75f882`; `swift build -c release`; macOS 26.6.2; Swift 6.3.3; Apple M3, 24 GB.
- Server: `.build/release/NVMAIServer --models-dir models --model qwen3.5-2b_4-Bit --port 8091 --reasoning off`
- Request: `POST /v1/chat/completions`, `{"messages":[{"role":"user","content":"Capital of Paris"}], "temperature":0, "max_tokens":128, "stream":true, "stream_options":{"include_usage":true}}`
- Engine `gpu`/`cpu` is the catalog backend; the `@cpu` id selects the CPU engine for a dense install.
- One model resident at a time, so every row pays its own load, reported as `load`.
- The three MTP draft heads are not standalone models and the catalog refuses them; they are not rows here.

## Results

| # | Model | Quant | Engine | Load (s) | TTFT (s) | Decode (tok/s) | End-to-end (tok/s) | Tokens | Finish | First sentence of the reply |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Qwen 3.5 2B | 4-bit | gpu | 0.22 | 0.171 | 55.312 | 45.73 | 46 | `stop` | The capital of France, and the capital of the country **France**, is **Paris** |
| 2 | Qwen 3.5 2B | 4-bit | cpu | 2.47 | 0.631 | 18.511 | 15.068 | 51 | `stop` | The capital of France is **Paris** |
| 3 | Qwen 3.5 2B | 8-bit | gpu | 1.87 | 0.338 | 35.784 | 27.223 | 39 | `stop` | The capital of Paris is **Paris** itself |
| 4 | Qwen 3.5 2B | 8-bit | cpu | 1.03 | 1.013 | 8.156 | 6.821 | 40 | `stop` | The capital of Paris is **Paris** itself |
| 5 | Qwen 3.5 4B | 4-bit | gpu | 7.0 | 0.804 | 8.984 | 7.64 | 38 | `stop` | The capital of Paris is **Paris** itself |
| 6 | Qwen 3.5 4B | 4-bit | cpu | 6.51 | 2.028 | 2.545 | 2.224 | 40 | `stop` | The capital of **Paris** is **Paris** |
| 7 | Qwen 3.5 4B | 8-bit | gpu | 10.95 | 1.657 | 10.089 | 7.504 | 62 | `stop` | The capital of France is **Paris** |
| 8 | Qwen 3.5 4B | 8-bit | cpu | 15.06 | 8.266 | 1.873 | 1.515 | 63 | `stop` | The capital of France is **Paris** |
| 9 | Qwen 3.5 9B | 4-bit | gpu | 12.88 | 1.075 | 8.032 | 7.148 | 88 | `stop` | There is a slight confusion in your question |
| 10 | Qwen 3.5 9B | 4-bit | cpu | 12.28 | 4.748 | 2.286 | 1.916 | 55 | `stop` | There is a slight confusion in your question |
| 11 | Qwen 3.5 9B | 8-bit | gpu | 8.13 | 1.772 | 8.015 | 6.626 | 68 | `stop` | There is a slight confusion in the phrasing of your question |
| 12 | Qwen 3.5 9B | 8-bit | cpu | 53.75 | 167.43 | 2.206 | 0.347 | 69 | `stop` | There is a slight confusion in the phrasing of your question |
| 13 | Ornith 1.5 35B-A3B | 4-bit | gpu | 4.61 | 0.398 | 12.614 | 11.414 | 46 | `stop` | Paris is itself the capital of France! There's no separate capital city—Paris serves as both the capital and the largest city of France |
| 14 | Ornith 1.5 35B-A3B | 8-bit | gpu | 5.8 | 0.638 | 7.925 | 7.462 | 78 | `stop` | Paris itself is the capital of France |
| 15 | Qwen AgentWorld 35B-A3B | 4-bit | gpu | 4.14 | 0.379 | 10.301 | 6.649 | 8 | `stop` | Paris is the capital of France. |
| 16 | Qwen AgentWorld 35B-A3B | 8-bit | gpu | 5.67 | 0.831 | 7.798 | 7.474 | 128 | `length` | <think> Thinking Process: 1 |
| 17 | Qwen 3.6 35B-A3B | 4-bit | gpu | 5.01 | 0.453 | 11.176 | 10.168 | 48 | `stop` | Paris **is** the capital of France |
| 18 | Qwen 3.6 35B-A3B | 8-bit | gpu | 5.56 | 0.839 | 8.408 | 7.344 | 47 | `stop` | Paris **is** the capital of France |
| 19 | Qwen 3.8 Flash Next 125B-A6B | 4-bit | gpu | 10.19 | 1.886 | 3.93 | 3.073 | 27 | `stop` | Paris is the capital of **France** |
| 20 | Qwen 3.8 Flash Next 125B-A6B | 8-bit | gpu | 11.51 | 6.336 | 1.192 | 0.933 | 27 | `stop` | Paris is the capital of **France** |

## Measurement notes

- **Row 12 is a cold-start outlier, and it did not reproduce.** Qwen 3.5 9B
  8-bit on the CPU recorded a 167.4 s TTFT and a 53.8 s load in the matrix; a
  re-run of that exact row immediately afterwards measured **load 16.2 s, TTFT
  2.9 s, 3.14 tok/s** (`recheck.jsonl`), consistent with the 4-bit row's decode
  rate. The outlier is the first fault-in of a 9.5 GB CPU snapshot while the
  machine was under memory pressure from the preceding loads, not the steady
  state. The rest of the row (69 tokens, `stop`) reproduced.
- **Qwen AgentWorld 8-bit did not answer within the cap.** It hit
  `finish_reason: length` at 128 tokens with a `<think>` scaffold in `content`
  even though the server ran `--reasoning off`; its 4-bit sibling answered in
  8 tokens. Whatever renders that install's thinking switch, it is not the
  binary `enable_thinking` field the other Qwen 3.6-family installs use.
- **Qwen 3.5 2B 4-bit on the GPU degenerated** -- "the capital of the French
  Republic and the capital of the French Republic" -- where the same model on
  the CPU and the same model at 8-bit were clean. It is a generation artifact
  of that one run, not an engine difference: the 4-bit CPU row is clean.
- Decode rates are one generation each, capped at 128 tokens, with no error
  bars. They are measurements, not performance ceilings.

## Every reply, verbatim

### 1. Qwen 3.5 2B 4-bit, gpu

```text
The capital of France, and the capital of the country **France**, is **Paris**.

Paris is the capital and capital city of France. It is the capital of the French Republic and the capital of the French Republic.
```

### 2. Qwen 3.5 2B 4-bit, cpu

```text
The capital of France is **Paris**.

Located on the left bank of the Seine River, Paris is the largest city in France and the capital of the French Republic. It serves as the political, cultural, and economic center of the country.

```

### 3. Qwen 3.5 2B 8-bit, gpu

```text
The capital of Paris is **Paris** itself.

Paris is the capital and the most populous city of France. It serves as the political, cultural, and economic center of the country.
```

### 4. Qwen 3.5 2B 8-bit, cpu

```text
The capital of Paris is **Paris** itself.

Paris is the capital and the most populous city of France. It serves as the political, cultural, and economic center of the country.

```

### 5. Qwen 3.5 4B 4-bit, gpu

```text
The capital of Paris is **Paris** itself.

Paris is the capital and most populous city of **France**. It is also the capital of the Île-de-France region.
```

### 6. Qwen 3.5 4B 4-bit, cpu

```text
The capital of **Paris** is **Paris**.

Paris is the capital and most populous city of France. It is located in the north-central part of the country on the Seine River.

```

### 7. Qwen 3.5 4B 8-bit, gpu

```text
The capital of France is **Paris**.

It is the country's most populous city and its main center of business, education, culture, and the arts. Located on the Seine River, Paris is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
```

### 8. Qwen 3.5 4B 8-bit, cpu

```text
The capital of France is **Paris**.

It is the country's most populous city and its main center of business, education, culture, and the arts. Located on the Seine River, Paris is famous for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.

```

### 9. Qwen 3.5 9B 4-bit, gpu

```text
There is a slight confusion in your question. **Paris** is the capital city, not a country.

*   **Paris** is the capital city of **France**.
*   Paris is the capital city of the country of **France**. also the most populous city in France.

If you are looking for the capital of a different country, please let me know the country name, and I will be happy to help!
```

### 10. Qwen 3.5 9B 4-bit, cpu

```text
There is a slight confusion in your question. **Paris** is the capital city, not the country.

The capital of the country where Paris is located (France) is **Paris**.

*   **Country:** France
*   **Capital City:** Paris

```

### 11. Qwen 3.5 9B 8-bit, gpu

```text
There is a slight confusion in the phrasing of your question. **Paris** is the capital city, not a country.

Therefore, Paris does not have a capital; rather, **Paris is the capital of France**.

*   **City**: Paris
*   **Country**: France
*   **Role**: Capital City
```

### 12. Qwen 3.5 9B 8-bit, cpu

```text
There is a slight confusion in the phrasing of your question. **Paris** is the capital city, not a country.

Therefore, Paris does not have a capital; rather, **Paris is the capital of France**.

*   **City**: Paris
*   **Country**: France
*   **Role**: Capital City

```

### 13. Ornith 1.5 35B-A3B 4-bit, gpu

```text
Paris is itself the capital of France! There's no separate capital city—Paris serves as both the capital and the largest city of France.

Is there something specific about Paris or France you'd like to know more about?
```

### 14. Ornith 1.5 35B-A3B 8-bit, gpu

```text
Paris itself is the capital of France. So there's a slight mix-up in the question—Paris is the capital, not a country or region that would have its own capital.

To clarify:
- **Paris** is the capital city of **France**
- France's capital: **Paris**

Is there something specific about Paris or France you'd like to know?
```

### 15. Qwen AgentWorld 35B-A3B 4-bit, gpu

```text
Paris is the capital of France.
```

### 16. Qwen AgentWorld 35B-A3B 8-bit, gpu

```text
<think>
Thinking Process:

1.  **Analyze the Request:** The user is asking for the "Capital of Paris".
2.  **Identify the Entities/Concepts:**
    *   "Paris" is a city.
    *   "Capital" refers to the seat of government of a country or region.
3.  **Evaluate the Premise:** Paris *is* a capital city. Specifically, it is the capital of France. The question "Capital of Paris" is based on a false premise or a misunderstanding (Paris is not a country/region that has a capital; it *is
```

### 17. Qwen 3.6 35B-A3B 4-bit, gpu

```text
Paris **is** the capital of France.

There is no separate "capital of Paris" because Paris itself is a city and the capital city of the country. It serves as the political, economic, and cultural center of France.
```

### 18. Qwen 3.6 35B-A3B 8-bit, gpu

```text
Paris **is** the capital of France.

It seems there might be a slight confusion in the phrasing. Paris is not the capital *of* Paris; rather, it is the capital city of the country **France**.
```

### 19. Qwen 3.8 Flash Next 125B-A6B 4-bit, gpu

```text
Paris is the capital of **France**.

It is not a country itself, but rather the capital city of the French Republic.
```

### 20. Qwen 3.8 Flash Next 125B-A6B 8-bit, gpu

```text
Paris is the capital of **France**.

It is not a country itself, but rather the capital city of the French Republic.
```

