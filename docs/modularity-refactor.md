# Modularity refactor — audit, plan, and status

Motivation: several model families with materially different requirements are
coming (Qwen3.8-Flash-Next first: hyper-connection residuals, sparse-indexed
attention, hashed n-gram embeddings, top-10-of-512 experts, group-32
quantization). The core must absorb new families without interleaving their
logic into shared code. This document records what the audit found, the
refactor plan, and what has landed.

Method constraint, from this repository's own history: incremental and
behavior-preserving, never a clean-sheet rewrite (v4.0 describes one that was
never built). Every step passes the full gates — warning-free build,
`tools/lint.sh`, the complete serial test suite, and a byte-identical
`tools/golden-baseline.sh` run — before the next step starts.

## Audit (2026-08-28)

Healthy and kept as-is:

- Kernels are one concern per file with Swift wrappers over Metal, reusable
  across families; specialization already falls back to generic paths.
- The expert streaming stack is fully parameterized (stride, slot count,
  budget) and knows nothing about families.
- `ArchConfig` is the single geometry vehicle; optional sub-structs
  (`HyperConnectionConfig`, `SparseIndexerConfig`, `PLEConfig`) default to
  `.none` so families only pay for what they use.
- Repacker family logic is already localized in `RepackPlanner`.

Bloat vectors:

| # | Finding | Risk |
| --- | --- | --- |
| 1 | `RealForwardRunner.swift` was 4,313 lines: decode loop, prefill loop, two attention encoders in two phases each, three MoE stages, the MTP verify and adapter, the ANE bridge, RDAdvise policy, and diagnostics in one class | Every new family interleaves branches into every concern; the file compounds toward unmaintainable |
| 2 | Tensor names are string literals inside `Model.swift` accessors | A family with different names forks the accessors; new roles scatter |
| 3 | `MoE.maxStreamedExperts = 8`, `Quantization.groupSize = 64`, kernel shape lists as inline constants | Top-10 / group-32 families hit walls that look architectural but are just hardcoding |
| 4 | `family` switches in five files (Model, runner, MTP, server naming, app descriptors) | Acceptable individually; the missing convention is that one file per family owns its knowledge |

## Plan and status

- **Step 0 — one codebase.** Merge the inert Qwen3.8 P0 foundation into
  main. DONE.
- **Step 1 — split the runner by concern (pure code motion).** DONE:
  `RealForwardRunner.swift` keeps state, init, and shared helpers; decode,
  prefill, MTP, ANE, and diagnostics move to extension files. No signature
  or behavior changes; `private` members used across the split become
  `internal` with the class treated as module-internal state. Creates the
  seams that later let a family plug in per-layer blocks instead of editing
  shared loops.
- **Step 2 — family schemas.** DONE: New `Runtime/Family/` directory: a
  `TensorSchema` maps logical roles to tensor names per family, and the
  runtime-schema validation bodies move out of `Model.swift` into one file
  per family. Adding a family means adding a file, not editing shared code.
- **Step 3 — data-driven limits.** DONE: The MoE Swift plumbing takes its expert
  count from `ArchConfig.topKExperts` instead of a constant (still 8 for
  every shipped family — behavior identical); group size is recorded per
  architecture (`quantGroupSize`) for the loaders. Metal-side function
  constants for top-10 and group-32 stay on the port branch: they are new
  behavior, not housekeeping.

All four steps landed 2026-08-28, each verified by warning-free build,
lint, the full serial suite (791), and byte-identical 4-bit and 8-bit golden
baselines. `RealForwardRunner.swift` is 922 lines of state and lifecycle;
Decode/Prefill/MTP/ANE/Diagnostics live in their own files;
`Runtime/Family/` owns tensor naming; the MoE wrapper takes its expert count
from `ArchConfig`.

Deliberately out of scope: protocol-witness dispatch inside the per-token hot
loop (the control plane measured ~3% of a token; coarse per-layer seams are
enough and measurable), server/app cosmetics, and any kernel change.
