## NVMAI 5.0.1 — 35B decode regression fix, Qwen-AgentWorld

NVMAI 5.0.1 fixes a decode regression that shipped in 5.0 for the 35B
models (Qwen 3.6, Ornith 1.5) and adds Qwen-AgentWorld 35B-A3B as a fourth
model.

### The regression

Instrumentation added shortly before 5.0 read
`ProcessInfo.processInfo.environment` on every decode kernel, which rebuilds
a dictionary from the process environment each time: about 800 rebuilds
per token. Inside Qwen3.8-Flash-Next's 160 ms token it was invisible; on the
35B family's ~45 ms token it was most of the token. Measured on the 5.0
binary, Qwen 3.6 35B-A3B 4-bit decoded at 6.67 tok/s against the 23.28
published for 4.6. Every such flag is now read once per process. Output is
unchanged: all four goldens are byte-identical.

### Qwen-AgentWorld 35B-A3B

Qwen's agentic fine-tune of the 35B-A3B geometry, installed straight from
Qwen's bf16 release by `tools/install_models.sh agentworld` (or
`agentworld-8bit`): one 70 GB download quantized one shard at a time into
both widths, with the router, the scalar shared-expert gate, the DeltaNet
gating projections and every norm kept at bf16. Goldens `agentworld-4` and
`agentworld-8` pin both installs and gate releases.

### Every install from its own bf16 release

`tools/install_models.sh` no longer repacks any third-party quantization.
Ornith 1.5, Qwen 3.6 and AgentWorld go through one converter
(`tools/prepare_agentworld.py --model {ornith15,qwen36,agentworld}`, both
widths from one ~70 GB download); Qwen 3.8 through its own; the three draft
heads from the same originals (Qwen 3.6's through the converter's new
`--draft-head` mode). In every build the router, the scalar shared-expert
gate, the DeltaNet gating projections and every norm stay at bf16. The
Qwen 3.6 rows below are from that build, not from mlx-community's. The
Qwen 3.6 draft head built the same way pairs with its target (75%
acceptance, 1.75 tokens per verify pass) and, like every draft head here,
stays off by default because the verify pass costs more than it returns.

The one quantity these builds give up against the mlx-community ones is
speed on the 35B 4-bit installs, and it is entirely the head: mlx-community
quantized the embedding and lm_head at 4-bit, NVMAI keeps them at 8-bit.
Measured on Qwen 3.6 4-bit, the same build with a 4-bit head decodes at
23.0 tok/s against 19.2. 8-bit stays the default for quality; the converter's
`--head-bits 4` is there for anyone who wants the 20% back.

### Also in this release

- Top-8 routing now uses the one-simdgroup selector too (byte-identical,
  a few percent on the 35B family).
- `install_models.sh` no longer fails every first install by passing
  `--resume` without resume state.

### Performance

512-token story-generation benchmark at the shipped defaults, base M3, 24 GB
(Qwen 3.6 one run, AgentWorld medians of three and two, Qwen3.8 one run):

| Model | Quantization | 5.0 | 5.0.1 |
| --- | --- | ---: | ---: |
| Qwen 3.6 35B-A3B | 4-bit | 6.67 tok/s | **19.21 tok/s** (bf16-sourced build; 22.30 on the mlx-community build) |
| Qwen 3.6 35B-A3B | 8-bit | n/a | **9.73 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | 11.94 tok/s | **18.57 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | 8.41 tok/s | **9.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | 5.25 tok/s | **5.05 tok/s** (unchanged within noise) |

### Checksum

`nvmai-5.0.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
