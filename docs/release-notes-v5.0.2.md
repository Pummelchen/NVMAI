## NVMAI 5.0.2 — per-model tuning profiles, bigger expert caches for the 35B family

NVMAI 5.0.2 gives every installed model and width its own tuning row and
uses it to lift the three 35B-A3B models by 9% at 4-bit and 15% at 8-bit,
with no change to any weight, head or KV width. Output is byte-identical
to 5.0.1 on all eight goldens.

### Per-model profiles

`ModelProfile` is a table with one row per (model, routed-expert width):
expert-cache budget, prefetch depth and disk I/O tier, prefill chunk,
sampling defaults and the kernel switches. Resolution is family default,
then the row, then the environment, so a row can be edited without moving
any other install and an experiment can still override a shipped value
with `NVMAI_*` variables. The resolved profile is logged once at load under
`NVMAI_RUNNER_STATS`.

### Where the 35B token went

A per-role split of a Qwen 3.6 token at the 5.0.1 defaults: 49.8 ms at
4-bit, of which 13.2 ms was expert I/O with only a quarter hidden behind
the GPU, at an 87.7% cache hit rate with 128 slots per layer; 121.9 ms at
8-bit, of which 51.7 ms was expert I/O, 12% hidden, at 79.3% with 64
slots. The 8-bit vocabulary head, which a head-width test had pointed at,
runs at 89.9 GB/s, the practical ceiling on this GPU.

The 128-slot cap was only a list entry. The allowed slot counts gain 160,
192 and 256, and each install was measured on its own with the story
prompt, arms interleaved, swap sampled around every run:

| Install | 4-bit, 128 → 160 slots | 8-bit, 64 → 96 slots |
| --- | ---: | ---: |
| Qwen 3.6 35B-A3B | 19.50 / 20.45 → 20.55 / 21.04 | 9.85 / 9.72 → 11.13 / 11.19 |
| Ornith 1.5 35B-A3B | 19.91 / 20.41 → 20.84 / 21.02 | 8.69 / 9.12 → 10.83 / 10.86 |
| Qwen-AgentWorld 35B-A3B | 20.52 / 20.50 → 21.11 / 20.92 | 9.31 / 9.25 → 11.15 / 11.21 |

All six rows now ship 10 GiB at 4-bit (160 slots) and 12 GiB at 8-bit (96
slots); swap stayed flat on every arm. 192 slots at 4-bit measured another
4% on Qwen 3.6 (21.61 / 21.60) but pushed 1.5 GB to swap on first contact
on a 24 GB machine, so it stays one `--ram-budget 12G` away for machines
with more memory.

### Qwen3.8-Flash-Next

- 4-bit: predictive prefetch runs two deep on the utility disk I/O tier,
  +3% over one deep on the default tier (5.46 / 5.40 vs 5.21 / 5.33). The
  throttle tier costs 22%; two-layer-ahead prefetch (`NVMAI_PREFETCH_AHEAD=2`,
  opt-in) measured a wash.
- 8-bit: 40 slots (9.5 GiB) instead of 32, +8% (2.18 / 2.27 vs 2.05 / 2.06)
  with no paging; 48 slots grew swap by about 1 GB per run.
- The prefetch levers that helped Qwen 3.8 measured a wash on every 35B
  install at both widths, which is why the rows differ.

### Also in this release

- Ornith 1.5 rebuilt from ornith-ai's bf16 release through the shared
  converter, like Qwen 3.6 and AgentWorld; no third-party quantization is
  left in `install_models.sh`. Goldens `ornith-4` / `ornith-8` pin it.
- The prefetch disk I/O tier is a profile field (`NVMAI_PREFETCH_IO_TIER`
  still overrides it).
- `NVMAIBench head_affine8 | head_affine4 | head_int4` benchmarks the
  vocabulary head GEMV at the 35B shape.

### Performance

512-token story-generation benchmark at the shipped defaults, base M3,
24 GB, one run each on the release binary:

| Model | Quantization | 5.0.1 | 5.0.2 |
| --- | --- | ---: | ---: |
| Qwen-AgentWorld 35B-A3B | 4-bit | 18.57 tok/s | **21.28 tok/s** |
| Ornith 1.5 35B-A3B | 4-bit | 19.24 tok/s | **20.99 tok/s** |
| Qwen 3.6 35B-A3B | 4-bit | 19.21 tok/s | **20.95 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | 5.25 tok/s | **5.40 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | 9.73 tok/s | **11.23 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | 9.46 tok/s | **11.16 tok/s** |
| Ornith 1.5 35B-A3B | 8-bit | 9.72 tok/s | **10.89 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | 2.03 tok/s | **2.06 tok/s** |

### Checksum

`nvmai-5.0.2-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
