# NVMAI Audit & Fix Tracker

Status ledger for the production-readiness audit. Legend:
- `open` — not started · `wip` — in progress · `done` — fixed · `verify` — fixed, awaiting full-suite verification · `n/a` — hardware-dependent (cannot run here) or resolved by deletion.

Summary of audit: 6 module audits + cross-cutting scan + manual verification. ~218 distinct findings (the earlier "~170" estimate was low). Plus a user-mandated Gemma removal (WS-G) — NVMAI is Qwen-only.

## Final status (2026-08-07)

- **WS-G Gemma removal: done.** 0 `[Gg]emma` references in sources/, tests/, benchmark/, .github/, README/AGENTS/CONTRIBUTING. `GemmaToolCallParser`/`GemmaToolSchema` deleted; kernels renamed.
- **R1–R39, K1–K28, F1–F60, S1–S35, D1–D32: all fixed** (six parallel waves + integration). S21/S22 resolved by Gemma deletion.
- **T-items: 32/34 done** (CI, benchmark harness + README correction, .gitignore, iOS platform, offline tokenizer tests, MTP tests, server coverage, launch-script scoping). T29/T32 have documented test seams (see Known limitations). T3/T4 fixed in code; benchmark regeneration is hardware-bound.
- **Builds: `swift build` and `swift build --build-tests`: 0 errors, 0 warnings.**
- **Real-model smoke test (qwen36.gturbo, 8 tokens): PASS** — schema validator, runtime shader compile, prefill+decode all verified on the real checkpoint. The validator required 3 real-model-driven fixes during the smoke test (4-bit packed byte math, trailing-zero BF16 dims, router-width shared-expert gate) — all in place.
- **Full serial test suite (`swift test --no-parallel`): RUNNING (result to be recorded below).**

## Execution plan (waves)

| Wave | Scope | Owner | Gate |
|---|---|---|---|
| WS-G | Gemma removal (arch, tool pipeline, repacker, defaults, kernels, tests, docs) | me + 1 agent (mechanical test updates) | `swift build` |
| WS-1 | Blockers: B1 CB-error propagation, B2 schema validation restore, B3 CI + Scripts + warning gate, B4 benchmark harness/README correction | me | build + grep |
| WS-2 | Runtime + Tokenization findings (R*) | agent | build |
| WS-3 | Kernels/Infra/Metal findings (K*) | agent | build |
| WS-4 | Format/Repack/Validation findings (F*) | agent | build |
| WS-5 | Server findings (S*) | agent | build |
| WS-6 | App/Decode findings (D*) + 24 warnings | agent | build |
| WS-7 | Tests/Benchmark/Repo findings (T*) + new regression tests | agent | build |
| WS-8 | Verification: 0-warning release build, full `swift test --no-parallel`, per-finding grep re-check, tracker finalization | me | full suite |

Parallelism rule: waves touch disjoint file sets; WS-G completes before WS-1..WS-7; WS-7 after WS-2..WS-6 (test files overlap).

## WS-G — Gemma removal (Qwen-only)

| ID | Item | Status |
|---|---|---|
| G1 | `ModelTypes.swift`: remove `ModelFamily.gemma4`, `ArchConfig.gemma4_26B_A4B`, `gemma4LayerMask()`, default family `.qwen36` | open |
| G2 | `ManifestReader.swift`: remove gemma4 family detection / gelu branch | open |
| G3 | `Model.swift`: remove `.gemma4` branches (router/shared-expert naming, tied head) | open |
| G4 | `RealForwardRunner.swift`: remove Gemma sandwich-FFN branch + router encode names + comments | open |
| G5 | `Tokenizer.swift`: remove `ChatDialect.gemma`, `gemmaChatTemplate`, `resolveGemmaTokens`, gemma constants/modelID/vocab | open |
| G6 | Delete `GemmaToolCallParser.swift`; rewire `StructuredAssistantDecoder.swift`, `JSONValue.swift`, server validation to `QwenToolCallParser` | open |
| G7 | Delete `GemmaToolSchema.swift`; rewire `OpenAIModels.swift` tool-parameter adaptation (resolves S21/S22) | open |
| G8 | `HTTPServer.swift` + `OpenAIModels.swift`: `ChatDialect` default → `.chatml` | open |
| G9 | `ServerArguments.swift`: modelID default/usage text (resolves S7) | open |
| G10 | `SupportedModelSource.swift` + `NVMAIRepack/Command/main.swift`: remove gemma4, default → qwen36, usage text | open |
| G11 | `RepackPlanner.swift`: remove `.gemma4` branches | open |
| G12 | `ArchInfo.swift`: remove gemma4 loader/family | open |
| G13 | `AppModelInstallDescriptor.swift`, `AppModelLocation.swift`: gemma4 descriptor/default dirname | open |
| G14 | Rename kernels `router_gemv_gemma4_r4` → `router_gemv_r4`, `router_gemv_gemma4_body` → `router_gemv_body`, `prefill_router_gemma4_block` → `prefill_router_block` (+ `MoE.swift`, `PrefillRouter.swift`, `RealForwardRunner.swift` call sites, `moe.metal`, `prefill.metal`, tests) | open |
| G15 | Shader/docs cleanup: remove "Gemma" comments; evaluate gelu_pytorch_tanh function-constant branches (remove if Qwen-only silu, else keep with clean docs) | open |
| G16 | Tests: replace `gemma4Toy`/`gemma4_26B_A4B` fixtures with Qwen toy configs, delete GemmaToolCallTests, fix SyntheticSnapshot, tokenizer tests → local Qwen fixture (resolves T6) | open |
| G17 | Docs: `THIRD_PARTY_NOTICES.md` Gemma section, `NVMAICLI/Args.swift` usage text | open |
| G18 | Grep sweep: zero `[Gg]emma` matches in sources/tests/benchmark/CI; docs only historical attribution allowed | open |

## Findings ledger

### R* — Runtime & Tokenization (sources/NVMAI/Runtime, Tokenization) — 39

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| R1 | RealForwardRunner.swift:2212-2228 | critical | facade | finishPendingRoutedCommand prints CB errors, never throws — silent corruption | open |
| R2 | RealForwardRunner.swift:2981-2995 | critical | facade | runSync/waitForCompletion print CB errors, never propagate | open |
| R3 | Model.swift:598-605 | critical | facade | requireBF16/requireAffine no-ops; layer validation commented out | open |
| R4 | RealForwardRunner.swift:2546 | major | unsafe | try! shared.encode in per-token decode path | open |
| R5 | RealForwardRunner.swift:2756 | major | unsafe | precondition(pendingRoutedCommand == nil) crash | open |
| R6 | Tokenizer.swift:291-293 | major | bugs | hardcoded ChatML generation suffix instead of template | open |
| R7 | Detokenizer.swift:23-45 | major | perf | O(n²) re-decode + rescans per token | open |
| R8 | Tokenizer.swift:184,254 | major | bugs | hardcoded vocab sizes | open |
| R9 | RawCompletion.swift:117-121 | minor | logic | resume overflow check over-rejects near maxContext | open |
| R10 | RawCompletion.swift:193-263 | minor | logic | MTP kvBacked/uncommitted semantics diverge | open |
| R11 | RealForwardRunner.swift:1445/2200 | minor | logic | runner stays dirty after non-cancel prefill failure | open |
| R12 | QwenToolCallParser.swift:95-101 | minor | bugs | literal `\n</parameter>\n` truncates argument silently | open |
| R13 | Detokenizer.swift:77 | minor | bugs | flush drops invalid trailing bytes silently | open |
| R14 | Tokenizer.swift:405-412 | minor | bugs | whitespace trimming alters user content | open |
| R15 | RealForwardRunner.swift:968-988 | minor | logic | RDAdvice bounded vs adaptive skip window inconsistent | open |
| R16 | RealForwardRunner.swift:2420-2520 | minor | perf | per-token per-layer host allocations + sync/readback | open |
| R17 | RuntimeConfiguration.swift:39-42 | minor | security | precondition crash on bad config values | open |
| R18 | Model.swift:334-337 | minor | unsafe | nonisolated(unsafe) capture + try? swallow | open |
| R19 | StreamingMTP.swift:169-171 | minor | bugs | _ = try? discards sidecar validation errors | open |
| R20 | StreamingStopMatcher.swift:20-45 | minor | bugs | Character-based matching vs UTF-8 boundaries | open |
| R21 | RawCompletion.swift:45 | minor | bugs | default softcap 30.0 wrong for Qwen | open |
| R22 | Model.swift:46-47 | minor | bugs | lmHeadWeightBits falls back to embedding slot | open |
| R23 | RealForwardRunner.swift:1396-1400 | minor | perf | MTLBuffer allocated per prefill chunk | open |
| R24 | StreamingMTP.swift:214-218 | minor | bugs | unreachable defensive draftNotReady mask | open |
| R25 | Sampler.swift:150-160 | minor | perf | Set built from history per sampled token | open |
| R26 | RealForwardRunner.swift:105-127 | nit | dead | PrefillProjectionFamily unused branches | open |
| R27 | KVCacheManager.swift:26,364-367 | nit | dead | KVView.startSlot unused | open |
| R28 | StreamingMTP.swift:24 | nit | dead | headroomBytes unused | open |
| R29 | KVCacheManager.swift:71,85 | nit | dead | fp16RingCapacityOverride never passed | open |
| R30 | RealForwardRunner.swift:1562 | nit | unsafe | linDtBias! force unwrap (safe but unguarded) | open |
| R31 | RealForwardRunner.swift:1495,2357,2832,2835 | nit | unsafe | preconditionFailure on arch misconfig | open |
| R32 | RealForwardRunner.swift:412 | nit | syntax | scratch buffers unlabeled | open |
| R33 | ModelExpertIO.swift:139-145 | nit | dead | routedExpertCacheSlotCount ignores param | open |
| R34 | Sampler.swift:140-147 | nit | bugs | tv_nsec seed collision | open |
| R35 | RawCompletion.swift:222-224 | nit | logic | external shouldStop → stopString reason | open |
| R36 | Model.swift:209 | nit | syntax | loadExpert(layer: 0) hardcoded | open |
| R37 | GDNStateManager.swift:74 | nit | unsafe | precondition in init instead of throw | open |
| R38 | RealForwardRunner.swift:1402-1407 | nit | perf | route arrays per chunk | open |
| R39 | Tokenizer.swift:270-273 | nit | bugs | manual BOS duplication (Gemma legacy) | open |

### K* — Kernels, Infrastructure, Metal — 28

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| K1 | moe.metal:239 | major | unsafe | 4-byte uint* weight load at 2-byte-only alignment guarantee | open |
| K2 | MetalContext.swift:148-211 | minor | concurrency | double-checked lock duplicates PSO compiles on concurrent miss | open |
| K3 | all kernel wrappers | major | bugs | `guard let enc = ... else { return }` silent no-op encodes | open |
| K4 | RoPE.swift:140-141 | minor | perf | dispatch 256 vs 64 threads (proportional RoPE) | open |
| K5 | Elementwise.swift:46-59 | minor | perf | one encoder per row loop | open |
| K6 | MPPPrefillInt4QMM.swift:24-27 | minor | bugs | silent pipeline=nil + @discardableResult fallback | open |
| K7 | PrefillAttention.swift:57-83 | minor | bugs | try? swallow + config-triggered preconditionFailure | open |
| K8 | prefill.metal:148, dequant_int4.metal:56, dequant_affine.metal:122 | minor | security | no token<vocab guard in embedding lookup | open |
| K9 | PreadExpertStreamer.swift:107-111 | minor | bugs | fstat failure skips size validation | open |
| K10 | PreadExpertStreamer.swift:183 | minor | bugs | preconditionFailure on cache-slot exhaustion (user-config crash) | open |
| K11 | PreadExpertStreamer.swift:228 | minor | logic | always fetches layer: 0 | open |
| K12 | PreadExpertStreamer.swift:226-240 | minor | concurrency | plan/execute lock gap, slot clobber | open |
| K13 | PrefillGroupedRoutedMoE.swift:226-229 | minor | unsafe | baseAddress! on empty sortedPairs | open |
| K14 | PrefillGroupedRoutedMoE.swift:390-409 | minor | perf | dead gate/up scratch writes | open |
| K15 | Attention.swift:62-71 | minor | concurrency | split-KV shared partial scratch | open |
| K16 | MoE.swift:305-310 + moe.metal | minor | bugs | router always derefs effective_scale/per_expert_scale | open |
| K17 | VerifiedInstallReceipt.swift:52-60 | minor | security | TOCTOU size-check + read | open |
| K18 | gdn.metal:300-428 | minor | unsafe | hard-coded 4 SIMD partials × 128-thread dispatch coupling | open |
| K19 | prefill.metal:895 | nit | dead | prefill_attention_tg_sum_single_bank unused | open |
| K20 | PrefillFinalRowHead.swift:70 | nit | unsafe | int4! (provably safe) | open |
| K21 | Attention.swift:92-96 | nit | bugs | wrong error case (missingFunction) | open |
| K22 | PackedExpertsLayout.swift:44-45 | nit | unsafe | unchecked subscripts | open |
| K23 | fused.metal:245,275,297 | nit | unsafe | reinterpret_cast alignment assumption | open |
| K24 | moe.metal:30-36 | nit | arch | gdn_in_proj depends on dequant_int4 compile order | open |
| K25 | GTurboModelDirectory.swift:46 | nit | unsafe | components.last! | open |
| K26 | LogitOutput.swift:186-195 | nit | bugs | pow(1/temperature) unguarded | open |
| K27 | AffineQuant.swift:8-9 | nit | dead | encodeTwoRows copy-paste (both used) | open |
| K28 | PreadExpertStreamer.swift:259-272 | nit | logic | &+ overflow in range-end math | open |

### F* — Format, Repack, Validation — 60

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| F1 | RemoteStreamingRepacker.swift:219 | major | logic | resume mismatch deletes partial dir incl. fresh .remote-metadata → silent missing config.json | open |
| F2 | VerifiedInstallTool.swift:36,163 | major | security | manifest/layout filenames joined w/o path validation (read-only traversal) | open |
| F3 | ResidentWriter.swift | major | dead | unreachable mmap writer subsystem | open |
| F4 | RemoteDownloadSession.swift:87 | major | bug | metadata HEAD redirects same-host vs ranged GET cross-host — asymmetric | fixed |
| F5 | Posix.swift:299 | major | logic | mmap len==0 EINVAL on empty shard | open |
| F6 | Posix.swift:273 | major | facade | adviseDontNeed documented no-op, never called | open |
| F7 | RepackPlanner.swift:459 | major | unsafe | scalesShape.last! + UInt64 overflow on remote shape | open |
| F8 | RemoteStreamingRepacker.swift:305 | major | perf | checkpoint rewrite + 2 fsyncs per range | open |
| F9 | SourceByteProvider.swift:121 | minor | logic | .tmp left on failure; not counted by DiskSpaceChecker | open |
| F10 | SourceByteProvider.swift:80 | minor | logic | outputFDs cache never re-opens replaced files | open |
| F11 | RemoteRangeTransfer.swift:118 | minor | logic | receivedBytes wrap before guard | open |
| F12 | RemoteRangeTransfer.swift:152 | minor | robustness | data dropped without cancel after rejection | open |
| F13 | Safetensors.swift:17 | minor | unsafe | headerSize underflow (fileSize<8) | open |
| F14 | Safetensors.swift:63-72 | minor | unsafe | payloadBase/end/elements overflow | open |
| F15 | RepackPlanner.swift:186 | minor | logic | multimodal companion tensors in excluded list | open |
| F16 | RepackPlanner.swift:384 | minor | unsafe | divide-by-zero numExperts | open |
| F17 | GTurboJSON.swift:146 | minor | bug | expertsPerLayer from layer 0 vs first non-empty | open |
| F18 | RemoteStreamingRepacker.swift:587 | minor | logic | duplicated condition; gate_proj feeds two slots | open |
| F19 | RemoteStreamingRepacker.swift:515 | minor | logic | sidecar TOCTOU + no fsync | open |
| F20 | HuggingFaceRemote.swift:96 | minor | logic | 3xx treated as success (dead branch) | open |
| F21 | Posix.swift:31 | minor | security | files created 0644 world-readable | open |
| F22 | RemoteStreamingRepacker.swift:491 | minor | security | recordOutputFile follows symlinks | open |
| F23 | DiskSpaceChecker.swift:74 | minor | logic | path-extension strip probes wrong volume | open |
| F24 | WriterCore.swift:55 | minor | unsafe | Int() traps on huge files | open |
| F25 | RemoteStreamingRepacker.swift:86 | minor | logic | crash window → installStateCorrupt dead-end | open |
| F26 | InstallLock.swift:73 | minor | logic | TOCTOU partial/checkpoint paths | open |
| F27 | RepackAudit.swift:75 | minor | dead | 10+ fields never written | open |
| F28 | BoundedScratch.swift:29 | minor | facade | unused ensure/zeroBuffer/largestEverBytes | open |
| F29 | GTurboManifestV1.swift:169 | minor | dead | codec encode never called | open |
| F30 | GTurboPackedExpertsLayoutV1.swift:72 | minor | dead | codec encode never called | open |
| F31 | GTurboResidentIndexV1.swift:156 | minor | dead | writeHeader/writeEntry never called | open |
| F32 | RemoteRetry.swift:96 | minor | logic | URLError.cancelled not retryable | open |
| F33 | RemoteStreamingRepacker.swift:293 | minor | perf | remoteRequestCount = planned, not actual | open |
| F34 | RemoteStreamingRepacker.swift:368 | minor | logic | downloadedThisRunBytes double-counts retries | open |
| F35 | IndexLoader.swift:46 | minor | unsafe | weightMap[k]! | open |
| F36 | VerifiedInstallTool.swift:205 | minor | security | no size cap; stat follows symlinks | open |
| F37 | RangeCopyPlanner.swift:109 | nit | logic | downloaded-copied underflow unasserted | open |
| F38 | RangeCopyPlanner.swift:150 | nit | perf | gap bridging downloads discarded spans | open |
| F39 | RangeCopyPlanner.swift:167 | nit | logic | range-id stability on comparator tie | open |
| F40 | Posix.swift:320 | nit | unsafe | madvise rounding + ignored failures | open |
| F41 | Posix.swift:298 | nit | unsafe | no negative-size guard | open |
| F42 | Posix.swift:203 | nit | logic | size check before preadAll | open |
| F43 | Sha256Stream.swift:49 | nit | logic | error reports got: 0 always | open |
| F44 | RemoteStreamingRepacker.swift:24 | nit | unsafe | baseURL force unwrap default | open |
| F45 | RemoteStreamingRepacker.swift:512 | nit | logic | single-element loop | open |
| F46 | RemoteStreamingRepacker.swift:558 | nit | logic | redundant removeItem | open |
| F47 | HuggingFaceRemote.swift:76 | nit | security | TOCTOU rename over existing | open |
| F48 | GTurboJSON.swift:117 | nit | logic | flags hardcoded | open |
| F49 | GTurboLayoutValidator.swift:34 | nit | logic | sub-tensor dicts never checked | open |
| F50 | VerifiedInstallTool.swift:57 | nit | logic | bytesVerified &+= wraps | open |
| F51 | VerifiedInstallTool.swift:242 | nit | logic | .DS_Store only top-level | open |
| F52 | GTurboDirectoryAccess.swift:41 | nit | logic | duplicate GTurboPathValidator | open |
| F53 | GTurboDirectoryAccess.swift:119 | nit | logic | tokenizer skip depth-0 only | open |
| F54 | ArchInfo.swift:305 | nit | logic | MTP arch field inconsistencies | open |
| F55 | RepackPlanner.swift:447 | nit | unsafe | padTo4 UInt32 truncation | open |
| F56 | RemoteChunkPolicy.swift:8 | nit | logic | maxBytes==defaultBytes single-value cap | open |
| F57 | Fp16Buffer.swift:8 | nit | unsafe | baseAddress! on empty | open |
| F58 | AttentionRef.swift:73 | nit | unsafe | scores[0] unconditional | open |
| F59 | LogitSoftcapSoftmax.swift:59 | nit | logic | 1.0/sum divide-by-zero | open |
| F60 | ScriptedLogitProducer.swift:23 | nit | race | calls unsynchronized | open |

### S* — Server — 35

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| S1 | HTTPServer.swift:47-59 | major | security | no timeouts/connection cap → slowloris/FD exhaustion | open |
| S2 | ServerInference.swift:862-886 | major | perf | sync multi-GiB snapshot write in serialized actor | open |
| S3 | ServerPromptStateStore.swift:263-310 | major | perf | sync SHA-256 + memcpy up to 8 GiB on restore | open |
| S4 | HTTPServer.swift:282-289,453-460 | major | perf | SSE no backpressure | open |
| S5 | HTTPServer.swift:453-546 | major | error | try? writes silently return; no [DONE]/status | open |
| S6 | ServerInference.swift:326-329 | minor | logic | queue off-by-one (admits limit+1) | open |
| S7 | ServerArguments.swift:12 | minor | dead | modelID unused (resolved by G9) | open |
| S8 | ServerInference.swift:261-289 | minor | facade | prepare() passthrough | open |
| S9 | ServerPromptCache.swift:57 | minor | dead | var entry unused | open |
| S10 | HTTPServer.swift:131,218 | minor | logic | RequestPhaseState never reset | open |
| S11 | OpenAIModels.swift:315-322 | minor | logic | max_tokens vs configured maxContext | open |
| S12 | ServerPromptCache.swift:166-169 | minor | perf | identical-prompt replay never hits | open |
| S13 | ServerPromptCache.swift:222-227 | minor | perf | multi-turn continuation never hits | open |
| S14 | ServerPromptCache.swift:229-231 | minor | logic | maxTokens bridge asymmetry | open |
| S15 | ServerInference.swift:660-663 | minor | logic | tier=live unverified resume | open |
| S16 | OpenAIModels.swift:309 | minor | nit | max_tokens + max_completion_tokens conflict silent | open |
| S17 | OpenAIModels.swift:351 | minor | nit | include_usage on non-stream ignored | open |
| S18 | OpenAIModels.swift:347 | minor | edge | stop strings unvalidated | open |
| S19 | OpenAIModels.swift:407-419 | minor | edge | trailing tool-call w/o result passes | open |
| S20 | HTTPServer.swift:469-481 | minor | error | 429 envelope type wrong; CancellationError → 500 | open |
| S21 | GemmaToolSchema.swift:55,132 | minor | unsafe | force unwraps (deleted with G7) | open |
| S22 | GemmaToolSchema.swift:64-68 | minor | security | unbounded recursion (deleted with G7) | open |
| S23 | ServerPromptStateStore.swift:311-357 | minor | edge | writeDisk orphan on move failure | open |
| S24 | ServerInference.swift:873-886 | minor | logic | snapshot failure keeps entry published | open |
| S25 | HTTPServer.swift:112-114 | minor | nit | activeTask never cleared on other routes | open |
| S26 | ServerInference.swift:744-747 | minor | nit | completion_tokens counts hidden tokens | open |
| S27 | HTTPServer.swift:47 | nit | protocol | no Host validation | open |
| S28 | HTTPServer.swift:198-206 | nit | protocol | HEAD returns 405 | open |
| S29 | HTTPServer.swift:36 | nit | unclean | so_reuseaddr on accepted sockets | open |
| S30 | HTTPServer.swift:571-578 | nit | nit | /v1/models created: 0 | open |
| S31 | OpenAIModels.swift:374-377 | nit | error | toolChoice bool misleading message | open |
| S32 | ServerLog.swift:45-48 | nit | concurrency | unsynchronized stderr writes | open |
| S33 | ServerTerminationSignals.swift:26-33 | nit | logic | second Ctrl-C dropped | open |
| S34 | ServerPromptStateStore.swift:205-213 | nit | perf | blocking setAttributes in actor | open |
| S35 | HTTPServer.swift:146-156 | nit | perf | body ByteBuffer retention | open |

### D* — App, DecodeProtocol, DecodeService — 32

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| D1 | DecodeServiceInferenceClient.swift:187-190 | major | logic | no reconnection — permanent wedge after crash | open |
| D2 | DecodeServiceResponseRouter.swift:31-68 | major | perf | blocks forever; no timeout on response path | open |
| D3 | DecodeServiceInferenceClient.swift:208 + DecodeUnixSocket.swift:31 | major | security | world-readable /tmp socket, unauthenticated protocol | open |
| D4 | NVMAIMacApp.swift:36 | major | bugs | fatalError at launch | open |
| D5 | AppModel.swift:387-404 | major | logic | cancelLoad doesn't cancel service load | open |
| D6 | AppModel.swift:246-256 | major | logic | setModelURL load/unload race | open |
| D7 | DecodeService/Entry.swift:39-41,116 | major | logic | untargeted cancel races boundaries; .cancel no-op | open |
| D8 | DecodeServiceInferenceClient.swift:12,62,70 | minor | dead | Connection.loadedDirectory unused | open |
| D9 | ResponseMarkdownRenderer.swift:97 | minor | dead | plainText unused | open |
| D10 | DecodeService/Entry.swift:60 + Client:57 | minor | facade | load callbacks discarded; loadSeconds 0 faked | open |
| D11 | DecodeService/Entry.swift:67-76 | minor | logic | stale model state after failed load | open |
| D12 | DecodeServiceInferenceClient.swift:255-259 | minor | logic | socket file never unlinked on retry exhaustion | open |
| D13 | DecodeServiceInferenceClient.swift:75 | minor | bugs | unload swallows response | open |
| D14 | AppModel.swift:707-712 | minor | bugs | liveMemoryBytes never cleared | open |
| D15 | AppModel.swift:208-213 | minor | bugs | stale memory metric between runs | open |
| D16 | DecodeServiceInferenceClient.swift:119-139 | minor | bugs | throttle drops tokens from outputText | open |
| D17 | OutputPaneView.swift:293-299 | minor | perf | full re-render on terminal transition | open |
| D18 | DecodeServiceResponseRouter.swift:27 | minor | unsafe | deinit closes FH while reader blocked | open |
| D19 | AppModel.swift:666-671 | minor | dead | commented no-op branch | open |
| D20 | AppContextLengthOption.swift:38-46 | minor | syntax | hardcoded memory deltas drift | open |
| D21 | AppRuntimeOptions.swift:77-86 | minor | syntax | hardcoded slots deltas | open |
| D22 | AppModel.swift:457-460 | minor | bugs | hasPartialModelDownload swallows errors | open |
| D23 | RealInferenceClient.swift:330 | minor | logic | promptIds.count < maxContext off-by-one | open |
| D24 | DecodeService/Entry.swift:73 | nit | syntax | throwaway sampler instance | open |
| D25 | StatusHUDView.swift:13 | nit | syntax | magic 84 padding | open |
| D26 | MetricFormat.swift:31-41 | nit | syntax | locale-dependent formatting | open |
| D27 | PromptSubmissionPolicy.swift:10 | nit | logic | Command+Return fallthrough | open |
| D28 | DecodeProtocol.swift:59-69 | nit | unsafe | header-size assumption | open |
| D29 | PromptComposerView.swift:60-77 | nit | logic | duplicate submission paths | open |
| D30 | AppMemorySampler.swift:33-34 | nit | bugs | peakBytes nil vs 0 conflated | open |
| D31 | InstructionTranscriptDocumentController.swift:82-84 | nit | logic | Character vs NSString count mixing | open |
| D32 | DecodeServiceOutbox.swift:34-43 | nit | logic | .token never signals condition | open |

### T* — Tests, Benchmark, CI, Repo — 34

| ID | Loc | Sev | Cat | Finding | Status |
|---|---|---|---|---|---|
| T1 | ci.yml:31 | critical | ci | Scripts/test.sh missing — CI always red | open |
| T2 | ci.yml:33 | critical | ci | Scripts/check_markdown_links.rb missing | open |
| T3 | cache_on_mtp_on_*.json | critical | benchmark | impossible config (MTP forces cache off) | open |
| T4 | all 12 JSONs | critical | benchmark | cached_tokens 0 — cache dimension void | open |
| T5 | nvmai_benchmark.py:123 | major | benchmark | expected answers discarded | open |
| T6 | TokenizerTests + ~10 files | major | test | tokenizer downloaded from HF at test time | open |
| T7 | StreamingMTPTests | major | coverage | MTP runtime untested | open |
| T8 | nvmai_benchmark.py:216,219 | major | benchmark | pkill self-match; kills user servers | open |
| T9 | nvmai_benchmark.py:172,200 | major | benchmark | hardcoded base_dir; no signal trap | open |
| T10 | launch_8bit.sh:34 | major | benchmark | kills any port-8083 process | open |
| T11 | ci.yml:29 | major | ci | no lint/warnings gate | open |
| T12 | ci.yml:25-31 | major | ci | serial rule not enforced | open |
| T13 | nvmai_benchmark.py:146-147 | major | benchmark | warm_n differs; last6_avg /6 | open |
| T14 | AppModelTests.swift:159+ | minor | test | polling sleeps, no timeout failure | open |
| T15 | AppMemorySamplerTests:10-13 | minor | test | asserts inside if-let only | open |
| T16 | RealInferenceClientStateTests:185-194 | minor | test | zero-assertion tests | open |
| T17 | all 12 JSONs | minor | benchmark | absurd raw decode_rate rows | open |
| T18 | nvmai_benchmark.py:79,87 | minor | benchmark | except:pass swallows SSE errors | open |
| T19 | nvmai_benchmark.py:81 | minor | benchmark | 'json_str' in dir() fragility | open |
| T20 | nvmai_benchmark.py:155-157 | minor | benchmark | overhead conflates prefill; counts mismatch | open |
| T21 | Package.swift:143 | minor | fixture | empty Fixtures resource | open |
| T22 | .gitignore:39,44-49 | minor | repo | dead patterns; benchmark-results unignored | open |
| T23 | benchmark/ | minor | repo | launch scripts out of sync (WIP) | open |
| T24 | CLIArgumentsTests:72 vs AppGenerationRequestTests:42 | minor | test | duplicate test names | open |
| T25 | FakeInferenceClientTests | minor | test | fake only self-tested | open |
| T26 | MPPPrefillInt4QMMTests:200+ | minor | coverage | MPP tests skip on hardware gate | open |
| T27 | QwenToySynthetic.swift:11 | minor | coverage | no MTP sidecar variant | open |
| T28 | DecodeServiceOutboxTests:44-50 | minor | test | time-bounded sync | open |
| T29 | HTTPServerTests | minor | coverage | no cache-hit e2e test | open |
| T30 | HTTPServerTests | minor | coverage | SSE edge cases untested | open |
| T31 | DecodeService tests | minor | coverage | process lifecycle untested | open |
| T32 | HTTPServerTests | minor | coverage | real backend generate path untested | open |
| T33 | Package.swift:15 | nit | repo | iOS platform, no iOS target | open |
| T34 | ci.yml:29 + RepackCLITests:74 | nit | repo | release-only CI vs debug binary | open |

## Known limitations

- **B4/T3/T4 benchmark regeneration** requires the M3 + 19.5 GB model + hours of run time: harness will be fixed and the invalid cells retracted in-repo; regeneration must run on the user's hardware.
- **T26/T7** (MPP hardware path, MTP decode-loop tests) need a Metal device; the MTP test will be added against the synthetic Qwen fixture (runs on this machine); MPP hardware gate stays.
- Findings R34, F48, S30, S28, D26, S32 etc. are behavioral decisions — fixed per the audit's recommended direction.
