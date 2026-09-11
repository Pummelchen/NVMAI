# Deep audit, 2026-09-11 — findings register

A full read-only audit of the tree by seven independent passes (kernels/quant,
format+ModelIO+streaming, HTTP server, forward runner+KV cache, tokenizer+
sampling, repacker+memory+continuity, app+CLI+decode service), followed by
verification and fixes. This file is the register: every finding, where it is,
and what happened to it. It exists so nothing found is lost and so a reader can
tell a fixed bug from a known one.

**Status vocabulary.** `fixed` — changed in this audit, with the commit.
`verified` — I reproduced the mechanism in the code and it is real, not yet
changed. `documented` — real but accepted deliberately; the reason is written
down. `refuted` — the claim did not hold up when I read it. `open` — real,
cause understood, fix not written yet.

Findings a subagent reported and I did not personally confirm are marked
`unverified`. Nothing here is asserted on a subagent's word alone.

---

## Fixed in this audit

| # | Sev | Area | What was wrong | Status |
| --- | --- | --- | --- | --- |
| C1 | high | `NVMAIMemory/MemoryTools.swift` | `Int(Double)` traps on NaN/±inf/out-of-`Int`-range, and the value comes from a model-authored tool argument (`limit=1e999`). Clamps to `nil`/bounds instead of aborting the process holding the model. | **fixed** |
| C2 | high | `ContinuityCore/Session/SessionLog.swift` | `pruneTurns(keeping: 0)` computed `turnStarts[count - 0]` — one past the end, a trap. `keeping: 0` is now a cut past the end, which is what "drop every turn" means; boundary events survive per the doc. | **fixed** |
| C3 | high | `NVMAI/Infrastructure/Streaming/PreadExpertStreamer.swift` | `precondition(experts.count <= slotCount)` trapped, while the comment two lines above promised a throw and both callers already handle `nil`. Reachable from `--expert-cache-slots 8` against Qwen3.8-Flash-Next (top-10 routing; prefill tiles up to 16). Now `guard … else { return nil }`. | **fixed** |
| C4 | medium | `docs/v4.5-ane-prefill.md` | Said ANE prefill is "off by default"; `RuntimePrefillANE.environmentValue` returns `.on` when unset. Marked historical and corrected. | **fixed** |
| C5 | medium | `NVMAI/Runtime/Prefill/ANEPrefillAttention.swift` | Same false claim in the enum's doc comment ("off by default … never silently selected"). | **fixed** |
| C6 | medium | `NVMAICLI/Args.swift`, `NVMAIServer/Core/ServerArguments.swift` | `--expert-cache-slots` help hardcoded "8, 16, 24, 32, 64, 96, or 128" while the validator accepts 40/48/112/160/192/256. Now spelled from `RuntimeConfiguration.allowedExpertCacheSlots`. | **fixed** |
| C7 | medium-high | `tools/golden-baseline.sh` | Did not pin `NVMAI_PREFILL_ANE`, which defaults to `on`, and the ANE path is deliberately not byte-identical to the GPU path. The gate's meaning depended on prompt length and on whether a sidecar happened to be installed. Now exports `NVMAI_PREFILL_ANE=off`. | **fixed** |

## Verified, not yet fixed

| # | Sev | Location | What is wrong |
| --- | --- | --- | --- |
| V1 | critical | `NVMAI/Infrastructure/ModelIO/ResidentIndex.swift:96-99`, `NVMAIFormat/GTurboResidentIndexV1.swift:183-191` | Only `indexSize <= st_size` is checked. `residentSize` comes from the header and is used as the bound for every entry offset (`residentEnd = indexSize + residentSize`), so a same-size edit of `model_weights.bin` makes the reader hand out pointers past the mapping (SIGBUS / garbage). The receipt does not re-hash this file at load. |
| V2 | high | `NVMAIServer/Core/ResponsesAPIModels.swift:386-408`, `AnthropicModels.swift:446-459` | `/v1/responses` and `/v1/messages` fill omitted sampling with the generic `GenerationDefaults` **before** validation, so the served model's own defaults (`ServedModel.sampling`) are unreachable. Qwen3.8-Flash-Next samples at 0.6 instead of its card's 1.0 on two of three surfaces. |
| V3 | high | `NVMAIServer/Core/HTTPServer.swift:1667`, `:1909` | A streaming request rejected before `startStream` (queue full, shutting down) has no HTTP head, but `handleAsyncFailure` still writes SSE frames as a body: the client gets `data:` bytes with no status line. The intended 429 is undeliverable on every streaming surface. |
| V4 | high | `NVMAIBench/main.swift:850-882`, `:951-953`, `:984` | `cpu35` counts oracle failures, prints "N of 3 wrong", and exits 0. The check that is supposed to say the Swift forward pass matches the oracle reports success on failure. |
| V5 | high | `NVMAIRepack/Core/Remote/RemoteStreamingRepacker.swift:313-329` | The disk reservation covers `model_weights.bin` + `packed_experts/**` only. `plan.passthroughFiles` (Qwen3.8's ~95 GiB n-gram table, cap 256 GiB) is excluded but is preallocated and written, so a 168 GiB install passes a 66 GiB check and dies with ENOSPC mid-download. |
| V6 | high | `NVMAIMemory/ContinuityStore.swift:137-139` | The computed `bounded.limit` is passed to `MemoryRanking.rank`, which never reads it; the durable backend returns every match (up to a 2000-candidate scan) while the in-memory backend slices. Two backends answer the same query differently and one can push ~2000 records into the model's context. `maximumIndexScan` is dead. |
| V7 | medium-high | `NVMAICLI/Args.swift:135-136`, `:187` | `--rdadvise` is documented "default off" but the CLI defaults to `default` and `rdadviseEnabled = policy != .off`, so read-ahead advice is ON by default. Scripted verification records runs as RDADVISE off while measuring it on. |
| V8 | medium | `NVMAIMemory/SessionJournal.swift:127-133` | The truncation guard is in UTF-8 bytes but the cut is in Characters, so multibyte text over 4096 bytes with under ~2730 characters is "summarised" as the whole text plus a duplicated tail, with a **negative** omitted count (2000 CJK chars → `-3072 bytes omitted`). |
| V9 | medium | `NVMAICLI/Args.swift:264-269` | `--repetition-penalty` accepts `> 0` while the app's validator requires `>= 1`; `GenerationConfig.validate()` does not check it, so `0.5` runs and rewards repetition. |
| V10 | medium | `ContinuityCore/Persistence/Journal.swift:382-383` | `compact` closes the descriptor then `descriptor = try openForAppend(url)`; if the reopen throws, the field keeps the closed fd number, so `descriptor >= 0` passes and later writes target a reassigned fd. |
| V11 | medium | `NVMAIRepack/Core/Remote/SourceByteProvider.swift:80-95` | On reopen failure the dictionary still maps the path to the already-closed fd, and the exit `defer` closes it a second time — possibly an unrelated live descriptor. `LocalSourceByteProvider` does `removeValue` first; this one does not. |
| V12 | medium | `NVMAIFormat/GTurboManifestV1.swift:196-205`, `ModelIO/ManifestReader.swift:494` | Per-tensor quant overrides decode with `try?`, so a malformed slot is silently dropped, and surviving `weightBits` values are never range-checked — the exact silent 4/8-bit misread the type's own comment warns about. |
| V13 | medium | `NVMAI/Infrastructure/Streaming/NgramTableReader.swift:63`, `:75-76` | Preconditions trap on geometry from an unhashed `ple_constants.json`, and `rowCount &* UInt64(bytes)` wraps, collapsing `expected` so a mismatched table passes the guard. |
| V14 | medium | `NVMAIRepack/Core/Planning/RepackPlanner.swift:561` | Logical shape is derived as `scalesShape.last * 64`, a literal, while `plan.baseGroupSize` records the source's group size and is never required to be 64. A non-64 source writes an install whose index and manifest disagree. |
| V15 | medium | `ContinuityCore/Session/SessionLog.swift:268-308` | `turns(taskID:)` folds all of a task with a single `pendingPrompt` while `events` groups per session, so an unanswered prompt from session A can be paired with session B's reply and rendered as B's turn. |
| V16 | medium | `NVMAIApp/Core/Configuration/AppRuntimeOptions.swift:173-199` | `AppLoadedRuntimeKey` omits `prefillEnabled`, `prefillChunkTokens` and `conciseMode`, which do reach the helper and which it compares exactly. Changing Prefill or Prefill-chunk leaves staleness false, so no Reload affordance appears and the generation fails with "runtime options do not match the loaded session". |
| V17 | medium | `NVMAIServer/Core/ServerPromptCache.swift:25-34`, `:121-150` | The cache does not key on the request's reasoning level, though `ValidatedChatRequest.reasoning`'s doc says it does and that a cached range "must never be spliced onto" another level's. A mid-session switch can reuse a prefix and re-render the tail at the loaded level. |
| V18 | low-medium | `NVMAIServer/Core/ManagedModelBackend.swift:108-131` | `unload()` returns false while a load is in flight (`session` is nil), and the load then completes and stays resident. |
| V19 | low-medium | `NVMAIApp/Core/Inference/DecodeServiceInferenceClient.swift:227-298` | The helper's launchd label embeds pid+token and nothing scans for an existing job, so a force-quit leaves an orphan holding ~20 GB and the relaunch starts a **second** model process. |

## False documentation, no behavioural impact

| # | Location | What is false |
| --- | --- | --- |
| D1 | `NVMAIApp/Mac/Generation/PromptComposerView.swift:119` | "The default temperature is 0.60." The app follows the model profile, which is 1.0 for Qwen3.8-Flash-Next. |
| D2 | `NVMAICLI/Args.swift:146-148`, `Run.swift:70-78` | `--concise` is called "per-quantization" and the code claims the manifest bit width selects the variant; `ConcisePrompt.prompt(forRoutedExpertBits:)` returns `standard` for every width and the manifest read is pinned to `qwen36_35B_A3B`, so `bits` silently falls back to 4 for every other family. |
| D3 | `NVMAIServer/Core/VerifiedInstallReceipt.swift:7-11` | Claims loads verify the receipt binding "and file sizes"; the receipt↔manifest comparison never touches the filesystem, and only `model_weights.bin`, `layout.json` and `packed_experts/*.bin` are ever size-checked. Tokenizer and sidecar files are not. |
| D4 | `NVMAIValidation/.../Moe.swift:54` | References an `applyStreamed` sibling that no longer exists. |
| D5 | `NVMAIRepack/Core/Verification/VerifiedInstallTool.swift:50-51` | The comment says duplicate filesystem keys are rejected; that decode does not run the `GTurboManifestV1` structural checks (duplicate/reserved/prefix-collision paths). |
| D6 | `NVMAIApp/Core/Configuration/MacAppSettings.swift:11-46` | RDADVISE / prefill chunk / expert-cache policy / verification are exposed in the Inspector but not persisted, so they revert silently. |

## Verified and deliberately not changed

- `RuntimePrefillANE`'s asymmetry — an explicit `on` with no sidecar throws, the default degrades quietly — is correct and documented in `wasRequestedExplicitly`. Only the doc comment was wrong (C5).
- `executeExpertCachePlan`'s preconditions validate a plan the type itself constructs, not user input; once `makeExpertCachePlan` returns non-nil they hold by construction.
- `MemoryBackend` excluding memory items from `count_tokens` is documented and deliberate.

## Reported and not yet independently confirmed

Sorted by severity as reported; each needs its mechanism read before it is
trusted. Kept here rather than dropped.

- `NgramTableReader`/`ResidentIndex` assertions on `shape` unrelated to
  `sizeBytes` (`GTurboResidentIndexV1.swift:193-208`, trap at `Model.swift:1321`).
- `PreadExpertStreamer.readFull` does not retry `EINTR` (`:1493-1503`) where
  every other pread loop in the module does.
- Unchecked `UInt64` offset arithmetic (`PreadExpertStreamer.swift:417,613,634,
  1094,1097`, `ExpertStreamer.swift:60-61`).
- `SSEOutbox.next()` installs its continuation after the cancellation handler
  (`HTTPServer.swift:2174-2190`): a drainer cancelled between iterations can park
  forever. Today only a terminal frame rescues it.
- `errorCaught`/`failStream` cancel `activeTask` unconditionally
  (`HTTPServer.swift:314`, `:1898`), so with pipelining an I/O error belonging to
  response N can cancel request N+1.
- `ResponseStore.put` does not move a re-put id to the back of `order`
  (`ResponsesAPIModels.swift:471-480`), so it can be evicted before older entries.
- No aggregate cap on request headers or pipelined requests; the 413 is written
  only at `.end`, after the whole oversized body has been read and discarded
  (`HTTPServer.swift:63-79`, `:276-292`).
- `HEAD` to any path other than `/health` and `/v1/models` returns a body
  (`HTTPServer.swift:366-368`, `:2016-2043`).
- App: a Stop pressed inside the generation-start window can be dropped because
  `cancel()` writes `cancel(nil)` before `activeGenerationID` is set.
- App: one 60 s inter-event timeout covers prefill as well as decode, so a single
  prefill chunk slower than 60 s gets the helper declared dead, killed, reloaded
  and retried.
- `MemoryService.sweepStaleWorkspaces` deletes another process's journal and
  `.lock` by path without checking `flock`.
- A journal read error is indistinguishable from an empty journal, and the next
  compaction then destroys the old records (`Journal.swift:320-325`).
- `ContextAssembler.memoryItemIDs` is ranking order while its comment says render
  order.
