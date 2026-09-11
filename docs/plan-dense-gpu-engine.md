# Plan: the dense Qwen 3.5 models on the GPU engine

**Status: not implemented. The three dense models are CPU-only today, by
refusal rather than by design.** This document is the design and the step
plan; it is updated as steps land.

Motivation: the three dense models — Qwen 3.5 **2B / 4B / 9B** at 4 and 8 bits
— should be runnable on **either** engine, chosen per launch and per request,
the way every other model is. Today the GPU runtime refuses their family by
name, so the launcher can only tell a person that CPU is the only engine.

## What is already there (measured, not assumed)

The distance is much shorter than "port an architecture", because the GPU path
already implements every *structural* feature the dense family needs:

| Needed | Where it already exists | Evidence |
| --- | --- | --- |
| Gated-DeltaNet linear attention | `Metal/GDN/gdn.metal` + `Kernels/GDN/` | Ornith/Qwen 3.6 use it |
| Full attention every 4th layer | `fullAttentionLayerMask` handling | Ornith declares `[2,2,2,1,…]` too |
| **Partial** RoPE + NeoX subdim | `RealForwardRunner+Decode.swift:755`, `+Prefill.swift:927` | both families declare `partialRotaryFactor 0.25`, `ropeNeoxSubdim true` |
| Attention output gate | `attnOutputGate` (C44's audit) | dense manifest declares `true` |
| A dense `gate/up/down` FFN with affine/int8 weights | `SharedExpertRuntime`, `PrefillSharedExpert` | used as the shared expert of every MoE family |
| Tied or separate LM head | `Model.head()` returns `embedding()` when tied | `Model.swift:194` |
| Per-layer / per-tensor quantization | resident index + `manifest.quant` | the dense install declares every tensor |

The dense install itself is complete and carries all the geometry
(`models/qwen3.5_2B_4Bit/manifest.json`): `family: qwen3_5_dense`,
`hiddenSize 2048`, `numLayers 24`, `ffnIntermediate 6144`, `numExperts 0`,
`topKExperts 0`, `numFullKVHeads 2`, `headDim 256`, GDN widths, per-layer
`self_attn` 4/8-bit and `mlp` 4-bit, `tieWordEmbeddings true` (2B/4B) or a
separate head (9B).

## What is missing (the actual gap)

1. **The routed-MoE stage is unconditional.** `RealForwardRunner+Decode.swift`
   and `+Prefill.swift` always encode the router, the prefetch probes, the
   residency classification and the streamed routed FFN, clamping with
   `cfg.numExperts - 1` — which is `-1` for a dense model. That is why
   `validateRuntimeSchema` refuses the family
   (`Model+Loading.swift`: `"qwen35Dense is served by the CPU engine, not
   Model.load"`): running the dense tensors through the MoE stage would produce
   fluent nonsense, the failure mode this project has shipped once before and
   the one the audit treats as the worst kind.
2. **No schema validation for the dense tensor set.** The `qwen36` branch
   requires routed-expert and router tensors the dense install does not have.
3. **No `ModelProfile` row** for the dense family (sampling defaults exist on
   the CPU side as `CPUModelFamily.samplingDefaults`; the GPU side resolves
   profiles per identity).
4. **The catalog routes the family to the CPU unconditionally**
   (`ModelCatalog.probeInstall`: `identity.family == .qwen35Dense ? .cpu(.qwen35Dense) : .gpu(…)`),
   and the reason it does so is (1) — the routing is honest, not arbitrary.
5. **No engine selection.** Nothing accepts "this request on the CPU, that one
   on the GPU", and the router's one-model-resident invariant has no notion of
   the same install on two engines.

## The design

**The dense FFN is the shared-expert stage.** The layer pipeline already
computes, before the router:

```
h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
h1 = SharedExpert(h1)          // gate/up/down, silu, optional scalar gate
```

For a dense model that *is* the FFN, with the dense `mlp.gate_proj/up_proj/
down_proj` as its weights and no routed sum. So the change is not a new
kernel: it is to point the existing stage at the dense tensor names and to make
the routed half conditional.

1. **Validation.** Replace the refusal in `validateRuntimeSchema` with a
   `qwen35Dense` branch that requires the dense tensor set (embedding, final
   norm, per-layer `input_layernorm`, `post_attention_layernorm`,
   `pre_feedforward_layernorm`, `mlp.{gate,up,down}_proj`, the GDN projections
   on linear layers, `self_attn.{q,k,v,o}_proj` on full-attention layers, q/k
   head norms, and the head when the embedding is not tied), each at the width
   its `manifest.quant` entry declares — the same `RuntimeSchemaChecks`
   mechanism C44 used.
2. **The FFN stage becomes the whole FFN when `numExperts == 0`.** In decode
   and in prefill: skip the router, the prefetch probes, the residency
   classification, the expert fetch and the routed FFN; `h = h + denseFFN(h1)`
   then the layer scalar. Both engines then agree by construction on *what* is
   computed, which is what makes the equivalence test below meaningful.
3. **Weights.** `SharedExpertRuntime` already reads gate/up/down views with a
   weight-bits parameter; the dense path feeds it the `mlp.*` views and the
   manifest's per-tensor width instead of the `shared_expert.*` names.
4. **Profile.** A `ModelProfile` row for the dense family: sampling from
   `CPUModelFamily.samplingDefaults` (0.6 / 0.95, the Qwen 3.5 card), and the
   decode knobs the GPU path needs. No expert cache, no MTP.
5. **Engine selection.**
   - The catalog lists an install once per engine that can serve it, so the
     dense installs appear twice (CPU and GPU) and every other install once.
     The entry carries the engine, so nothing else has to change shape.
   - The launcher asks which engine when an install has more than one — the
     question that was deliberately *not* added while the answer would have
     been ignored. `--engine cpu|gpu` selects it non-interactively.
   - Per request: the model id names the engine (`<id>@cpu` / `<id>@gpu`),
     which reuses the existing per-request routing rather than inventing a new
     field, and keeps a client that names no engine on the install's default.
   - Residency: the router's one-resident-at-a-time rule becomes "one resident
     per engine", so a request may hold the CPU copy in host RAM and the GPU
     copy in unified memory at once. On this machine that is 1.3–9.5 GB for
     the dense models at 4/8 bits, so it fits; the rule stays a rule rather
     than a free-for-all.

## The acceptance bar

A new family is trusted here by **numbers against a reference**, not by a
smoke test. The reference already exists: the CPU engine, which the opt-in
dense equivalence gate and `NVMAIBench cpu35` both check against the numpy
oracle.

1. **Logit equivalence, GPU vs CPU, on the real install**: the same prompt
   through `Model` (GPU) and `CPUQwen35` (CPU), compared element-wise at the
   `worst == 0` bar the dense equivalence test already uses for
   snapshot-vs-install. This is the step that decides whether the port is
   correct, and it is written before the refusal is lifted.
2. **The three oracle continuations** through the GPU path
   (`NVMAIBench cpu35`'s checks, or their GPU equivalent): "Once upon a time",
   "The capital of France is Paris", "… lazy dog" — greedy, fixed seed.
3. If the numbers match, and only then: the refusal is removed, the catalog
   lists both engines, and the launcher's engine question becomes real.
4. Both golden baselines still byte-identical (the change touches shared
   decode/prefill code, so the existing families are the regression gate).

## Steps

Each step passes the full gates before the next: warning-free build,
`tools/lint.sh`, the serial suite, and a golden baseline.

### What the code requires, read before writing S1

Four details decide whether S1/S2 are correct, and each was checked against the
install and the loader:

1. **The FFN is per-tensor quantized, not slot quantized.** The dense install's
   global slots are `attention` 4-bit, `embedding` 8-bit, `sharedExpert` 8-bit,
   `router` 8-bit -- but the tensors that matter carry **per-tensor** entries:
   `layers.N.mlp.{gate,up,down}_proj` are 4-bit, and `self_attn.k_proj` /
   `v_proj` are 8-bit on full-attention layers while `q_proj`/`o_proj` are
   4-bit. So the dense schema must resolve each tensor's width from its own
   manifest entry, not from `quant.sharedExpert` (8-bit) or `quant.attention`
   (4-bit). The `qwen38flash` branch already does this for its embedding and
   head ("validated against the embedding slot the manifest declares rather
   than an assumed width"), so the mechanism exists; the dense branch has to
   use it for every MLP and attention tensor.
2. **`TensorSchema.qwen35Dense` should map the shared-expert roles onto the
   dense MLP**: `sharedExpertGate` -> `layers.N.mlp.gate_proj.weight`,
   `sharedExpertUp` -> `…up_proj.weight`, `sharedExpertDown` ->
   `…down_proj.weight`. That is what makes S2 small: the existing shared-expert
   stage *is* a dense SwiGLU FFN (`silu(gate(x)) * up(x)` through `down`), the
   dense family has `sharedExpertGated: false` so the scalar-gate branch is
   already inert, and no new kernel is needed. `router` and
   `sharedExpertScalarGate` must be names that cannot exist (the family has
   neither) so that any accidental read fails at the lookup instead of reading
   the FFN gate as router logits.
3. **The FFN width is `ffnIntermediate` (6144 for the 2B, 12288 for the 9B),
   not `moeIntermediateSize`** -- which is 0 for this family. The runner sizes
   the shared-expert stage from `cfg.moeIntermediateSize` today
   (`RealForwardRunner.swift:531`), so S2 has to select `ffnIntermediate` when
   `numExperts == 0`; otherwise the stage is allocated at width 0.
4. **The head differs by model within the family**: the 2B and 4B tie the
   embedding (`tieWordEmbeddings: true`, no head tensor) and are served by
   `Model.head()`'s tied path, while the 9B is untied and its tensor is named
   `language_model.lm_head` -- *without* the `.weight` suffix qwen36 uses. One
   dense schema cannot spell both, so either the schema's `lmHead` follows the
   tie flag or the dense branch validates the head conditionally. The 9B's
   tensor name is the reason this is recorded here rather than discovered in
   S2.

- [x] **S1a — dense schema and validation.** Landed: `TensorSchema.qwen35Dense`
      (the FFN spelled through the shared-expert roles), `ManifestQuant.slot(forTensorNamed:overrides:fallback:)`
      so per-tensor widths are honoured rather than assumed, the dense branch in
      `validateRuntimeSchema`, a dense-aware `validateLayerTensors` (no router,
      no scalar gate, FFN width per tensor), and the routed-layout cross-check
      skipped when there are no experts. Build, lint and the suite are green
      (1458 tests). Execution still refuses, so nothing can produce wrong
      output.
- [~] **S1b — architecture resolution.** *Partly landed.* The resolver exists
      (`ArchConfig.resolved(forFamily:directoryURL:)` + `ArchConfig.from(manifest:family:)`)
      and the CLI and server use it, so the family refusal is gone and the dense
      install resolves its geometry -- and the layer conventions, which the
      format struct now decodes instead of discarding -- from the manifest. Three
      MoE-shaped assumptions on the way out were found by following the load path
      and fixed: the manifest demanded `packed_experts/layer_NN.bin` for every
      layer, the layout validator rejected a zero-expert document, and its
      cross-check demanded a size for files that were never written. The MoE
      path is unaffected (Ornith still loads and generates; both golden baselines
      byte-identical). **Still open:** the trusted-install receipt wants the same
      packed layer files, which is the next assumption in the chain; then
      `Model.load` reaches S1a's validation and the S2 boundary. The original
      note follows.  The runtime resolves a family's
      architecture from `ArchConfig.knownArchitectures`, which has no dense
      entry *and could not hold one*: the family is a single enum case and the
      three models have different geometry (2B/4B hidden 2048 / FFN 6144, 9B
      FFN 12288). The dense family must take its `ArchConfig` from the
      manifest's `arch` block (which carries every field, GDN widths included),
      at the three call sites that currently refuse it
      (`NVMAICLI/Run.swift:131`, `ServerInference.swift:647`, and the two app
      probes, which do not offer dense models and can keep their behaviour).
      Until this lands, `Model.load` is not reached for a dense install and the
      S1a validation is not yet exercised by a load — which is why S1a is
      committed as a checkpoint rather than as a finished step.
- [ ] **S2 — decode FFN.** Dense stage in `+Decode` (skip the routed half when
      `numExperts == 0`), logits compared against the CPU engine on the real
      2B install.
- [ ] **S3 — prefill FFN.** The same in `+Prefill`, so prompts are processed on
      the GPU too.
- [ ] **S4 — enable.** Remove the refusal and the forced CPU routing; catalog
      lists per engine; `ModelProfile` row; the launcher's engine question and
      `--engine`.
- [ ] **S5 — per request.** `<id>@cpu` / `<id>@gpu` routing and per-engine
      residency.
- [ ] **S6 — verify and record.** Oracle continuations on the GPU, both golden
      baselines, the register and the tracker.

S1–S3 are the feature; S4–S5 are the interface the request asks for. Nothing
user-visible changes until S4, and S4 does not happen until S2's numbers match.
