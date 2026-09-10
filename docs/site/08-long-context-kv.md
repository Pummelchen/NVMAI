> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** Runtime-Controls (context, KV), System-Design (prefill/decode, prompt-state reuse)

# Long context, YaRN, and the KV cache

Two questions live here: *how far* can the model look back, and *how much memory* does remembering cost. The answer to the second one is why the first one has a default and an extension, not just a number.

## The window

- **Native RoPE** (the default): up to **262,144** prompt-plus-response tokens. This is what the model was trained for, and what everything is tuned against.
- **YaRN** (`--rope-scaling yarn`): extends to **524,288** or **1,048,576** tokens. Select 1M by default; add `--max-context 524288` for 512K. YaRN changes RoPE *extrapolation*, so it **can affect quality beyond the native window** — you're asking the model to do something it didn't train for, and it's not currently compatible with MTP.

The CLI/app default for `--max-context` is 4K; the **server** default is 262K. So a bare CLI run is short by design and a bare server run is long by design.

## Why the KV cache is the memory story

Every token the model remembers is stored as a key/value row. That's the **KV cache**, and it grows with the context you actually use. NVMAI stores it in selectable **16-, 8-, or 4-bit** and **grows it from 8,192 tokens on demand** — it does not reserve max context at load. Two important things:

- `--kv-bits` is **independent of the model's weight precision**. A 4-bit or 8-bit *model* can run an 8-bit KV cache (the default), or a 4-bit one, without touching the installed weights. The weights keep their format; only the live attention state changes.
- At long context the KV is what dominates memory, so the precision choice is a memory choice.

Full-length attention KV for the production Qwen/Ornith topology:

| Context | 16-bit | 8-bit | 4-bit |
| --- | --- | --- | --- |
| 512K | 10.0 GiB | 5.31 GiB | 2.81 GiB |
| 1M | 20.0 GiB | 10.63 GiB | 5.63 GiB |

**The practical rule:** if you need 1M to fit on a 24 GB Mac, that's `--kv-bits 4`. 16-bit is for when you have headroom and want the cache to be as lossless as the weights.

## Why the first token is slow

The prompt must be **prefilled** before decode starts — the model has to read and attend over every prompt token to build its state. A large coding client can send thousands of tokens of instructions and tools, so that prefill is the visible "time to first token." Two mitigations, each in its place:

- **Prompt-state reuse** — send the complete conversation each turn and a compatible continuation reuses the already-built prefix. The server reports how much in `usage.prompt_tokens_details.cached_tokens`. This improves prefill and time-to-first-token (it does *not* raise decode tokens-per-second).
- **The `-fast` alias** — for direct questions, it strips the coding-CLI scaffolding before prefill, so there's less to prefill. Chat-only; see [The server](#06).

## Prompt-state reuse, precisely

The cache stores **inference state**, not completed responses. A snapshot includes both the full-attention KV rows *and* the gated-DeltaNet recurrent state — restoring only the KV would be incorrect. An entry is rejected when the model, runtime profile, context, template, tools, or prompt lineage differs. RAM reuse is on by default; an optional private SSD tier survives restarts (`--prompt-cache-disk`). The first request is always a miss; a hit needs an *exact compatible prefix*.

## Where to go next

- The engine that prefill feeds into: [SSD expert streaming](#03)
- Prefill acceleration on the Neural Engine: [ANE prefill](#09)
- All the context/KV flags in one table: [Runtime controls](#07)

*Version at time of writing: NVMAI 5.1.*
