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
| Qwen 3.6 35B-A3B | 4-bit | 6.67 tok/s | **22.30 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | 11.94 tok/s | **18.57 tok/s** |
| Qwen-AgentWorld 35B-A3B | 8-bit | 8.41 tok/s | **9.46 tok/s** |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | 5.25 tok/s | **5.05 tok/s** (unchanged within noise) |

### Checksum

`nvmai-5.0.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
