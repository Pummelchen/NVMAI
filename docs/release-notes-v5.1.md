## NVMAI 5.1 — the 35B models decode 10–13% faster at 8-bit

NVMAI 5.1 repairs the expert prefetch ring, which had been effectively off
since 5.0.2, and turns it back on for the 35B models. Qwen 3.6, Ornith 1.5
and Qwen-AgentWorld gain 10–13% at 8-bit and 2–4% at 4-bit. Output is
byte-identical to 5.0.2 on all eight goldens.

### The prefetch ring was clogged

Per-token counters added for this release showed the ring issuing about
five speculative reads per token where it should issue tens, with 1.78 of
its 2 slots permanently held by completed reads whose layer had already
passed. The reclaim rule frees a slot only when its layer index is at or
below the current one, so a prediction made for layer 46 or 47 at the end
of one token carried that index into the next, where the layer counter
restarts at zero. A prediction for the last layer was never reclaimed at
all. The rule arrived in 5.0.2 with the two-layer-ahead experiment.

Every prefetch measurement taken since then compared variants of a
mechanism that was not running, which is why the 35B rows had asked for a
speculative read and received nothing.

The ring now reclaims every finished read at the token boundary.

### What that is worth, per install

Measured on the repaired ring, arms interleaved within each round so the
machine's own drift cancels, 512-token generations, five rounds on
Qwen 3.6 and three on the others. Every 95% interval excludes zero:

| Model | 4-bit | 8-bit |
| --- | ---: | ---: |
| Qwen 3.6 35B-A3B | +1.8% | +11.3% |
| Ornith 1.5 35B-A3B | +1.8% | +12.6% |
| Qwen-AgentWorld 35B-A3B | +1.4% | +11.4% |

One read in flight remains the right depth: two returns +2.7% where one
returns +9.8%. On Qwen3.8-Flash-Next the repaired ring **loses** at every
setting, because its experts are larger and land after the layer that
would have used them, so that model ships with prefetch off. The two
families now differ on purpose.

### The expert cache stays wired through prefill

Prefill released the 10–12 GiB slot cache, memory pressure swapped it out,
and the first decode token faulted all of it back in: 1.6–4.7 seconds per
request, which a short answer pays in full. The cache now stays wired for
every model. Worth +7.1% and +5.1% on 48-token generations (Qwen 3.6
4-bit and 8-bit) and about +1% on long ones; prefill is not slower.

Two smaller fixes on the same path: the pin check no longer rebuilds a
dictionary from the process environment once per layer per token, and it
returns early once every layer is wired instead of walking all of them.

### A start script per model and quantization

Eight scripts that start the server for one install with no questions and
each on its own port, so two configurations can run at once:

```
tools/start-ornith-4bit.sh        tools/start-ornith-8bit.sh
tools/start-qwen3.6-4bit.sh       tools/start-qwen3.6-8bit.sh
tools/start-agentworld-4bit.sh    tools/start-agentworld-8bit.sh
tools/start-qwen3.8-4bit.sh       tools/start-qwen3.8-8bit.sh
```

Ornith keeps ports 8081 and 8083. The model list, install paths and ports
live in one catalogue that both the start scripts and the two interactive
launchers read, so they cannot disagree about where a model is or which
port it serves. `tools/server_launcher.sh` still asks, and
`tools/cli_launcher.sh` also wires up Codex, Qwen Code or OpenCode.

### The Mac app recognizes the models you have installed

The app's checkpoint catalogue listed only MLX repacks. After 5.0.1 moved
every install to a build quantized from the model's own bf16 release, the
app matched none of them: it reported each as a foreign checkpoint and
offered to download an MLX build over it. It now carries a fingerprint for
each of the eight installs, and separates what it can download itself from
what needs the command-line converter — for those it prints the
`tools/install_models.sh` target instead of a Download button. Older MLX
installs stay recognized and downloadable under `-mlx` selectors.

### Also in this release

- Every launch path takes its tuning from the model's profile row: the Mac
  app sizes the expert cache from it instead of a flat 64 slots, and its
  sampling follows the model until you change a value.
- Both launchers read the served model id from the running server. Since
  5.0 those ids carry the quantization suffix, and the bare name the
  launchers wrote into Codex and Qwen Code configs was rejected.
- New decode diagnostics under `NVMAI_RUNNER_STATS`: per-segment token
  timings, prefetch reads issued and adopted per token, and a prefetch-ring
  slot-state line.
- Opt-in and measured, off by default: early expert hits
  (`NVMAI_EARLY_HITS=1`, a wash within ±0.16 tok/s over five paired runs),
  a probe-weight prefetch gate, and a decayed-frequency cache policy (+4%
  on a 3000-word prompt, −4% on a short one).
- Three test races fixed that only a parallel `swift test` exposed.

### Performance

512-token story-generation benchmark at the shipped defaults, base M3,
24 GB, measured on this release:

| Model | Quantization | 5.0.2 | 5.1 |
| --- | --- | ---: | ---: |
| Qwen-AgentWorld 35B-A3B | 4-bit | 21.28 tok/s | **21.74 tok/s** |
| Ornith 1.5 35B-A3B | 4-bit | 20.99 tok/s | **21.65 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | 20.95 tok/s | **21.41 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | 11.23 tok/s | **12.37 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | 11.16 tok/s | **12.28 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | 10.89 tok/s | **11.93 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | 5.40 tok/s | **5.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | 2.06 tok/s | **2.10 tok/s** |

A decode profile of Qwen3.8-Flash-Next, including every lever measured and
closed, is in `docs/qwen38-decode-profile-2026-09-05.md`.

### Checksum

`nvmai-5.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
