# Qwen3.8 decode rewrite: what was measured, 2026-09-04

The brief was "rewrite the engine for Qwen3.8 for 10-20 decode tok/s on
the M3 base / 24 GB". This records what each candidate measured so the
next pass starts from data. Machine: M3 base, 10 GPU cores, 24 GB, ~3.6
GB/s SSD. Model: qwen3.8-flash-next_125B_A6B, 4-bit unless stated.

## Shipped

| change | commit | evidence |
|---|---|---|
| simd top-k router select (k != 8), default on | 1fc180f | router pair 11.0 -> 3.8 ms/token; story 4.97/5.03 -> 5.25/5.26 tok/s same day; both goldens byte-identical |
| simdgroup-per-key split-KV attention past the dense window | 4879f6c | qsa.attention 31.6 -> 3.45 ms/token at 3.7k context; 661 vs 2278 us isolated; goldens untouched by construction |
| split-timing roles through the QSA decode path | a293385 | located the kernel above |

## Measured and closed

| candidate | result |
|---|---|
| fused hyper-connection read (5 -> 2 dispatches) and write (2 -> 1), bit-exact | GPU time unchanged (entry_attn 3.83 -> 3.90 ms/token); decode 4.97/5.03 vs 5.04/4.83. Launch cost was never the cost: each read moves 3.3 MB of int4 gate weights. Kept behind NVMAI_HC_FUSED=1. |
| GDN in_proj GEMV layouts (unroll 4, split-K 4, pure 16-byte read) | 64 / 59 / 65 GB/s against the production 59; the old qkv bandwidth kernel tops at 74, MoE phase 2 at 88. The practical ceiling is ~75-88 GB/s, not 100, and the GEMVs are within 10-15% of it. |
| MoE phase 1 layouts (two rows per simdgroup; 4x unroll) | 33.6 vs 32.6 GB/s in the unspecialized bench; 0.447 vs 0.445 ms/layer in situ. The specialized routed path already runs the whole 27.7 MB blob set at 62 GB/s. |
| GPU-side QSA key selection (removes a runSync per full-attention layer) | verified 0 mask mismatches at 3.7k context; throughput a wash (the sync was not the cost). Opt-in: NVMAI_QSA_GPU_SELECT=1. |

## What bounds decode now

Short context (story prompt): ~5.25 tok/s. The token is ~92 ms GPU at or
near bandwidth plus ~86 ms of expert I/O with ~15% overlapped; the
overlap window per layer fits about one speculative expert read, and the
predictor is already the next layer's router on the current residual.
Deeper prefetch loses to SSD contention (measured earlier). 10-20 tok/s is
not reachable on this machine; the I/O floor alone puts the ceiling near
19 and the dependency structure well below that.

Long context (3.7k tokens): the GPU cliff is fixed. What remains is that a
long prefill leaves the expert cache cold for decode -- 52% hit rate over
the first 64 decode tokens, io_ms 86 of a ~210 ms token. Measure long
context decode with 200+ tokens, or give the prefill->decode transition a
warm-up policy; that is the next lever for coding-length sessions.

## Measurement notes

- The long-context server harness is bimodal (same-arm repeats of 1.73
  and 2.77 tok/s). Kernel decisions came from the unit test's
  gpuStartTime/gpuEndTime timing, which resolves them in seconds.
- Bench kernels with the runtime's function constants; the unspecialized
  phase-1 kernel reads 38 GB/s where the specialized one reads 62.
- A bit-exact fusion that moves no bytes is a wash here; look for serial
  single-thread work (the top-k) before suspecting bandwidth.
