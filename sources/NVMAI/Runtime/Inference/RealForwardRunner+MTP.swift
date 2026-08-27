import Foundation
import Metal

/// MTP: the speculative checkpoint, the width-2 verify (pair schedule), and the one-layer draft adapter.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    func captureSpeculativeCheckpoint(maximumBytes: Int) throws
        -> SpeculativeInferenceCheckpoint {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        let required = gdnState?.speculativePayloadBytes ?? 0
        guard required <= maximumBytes else {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: required,
                limit: maximumBytes)
        }
        return SpeculativeInferenceCheckpoint(position: kv.position)
    }

    func rollbackSpeculativeCheckpoint(_ checkpoint: SpeculativeInferenceCheckpoint) throws {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        if let gdnState {
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            try gdnState.encodeSpeculativeRestore(commandBuffer: cb)
            cb.commit()
            try waitForCompletion(cb)
        }
        // Row zero was confirmed and is present in the on-GPU checkpoint.
        try kv.rewind(to: checkpoint.position + 1)
        resetTransientState()
    }

    /// Discard an unaccepted native-MTP cache row. The draft contains only
    /// trimmable full-attention KV, so its logical cursor can move back without
    /// copying payload bytes; the next draft pass overwrites the stale row.
    func rewindMTP(to position: Int) throws {
        guard cfg.family == .qwen36MTP, let kv else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        try kv.rewind(to: position)
        resetTransientState()
    }

    var speculativeRollbackBytes: Int {
        gdnState?.speculativePayloadBytes ?? 0
    }

    /// Verify `[confirmed, draft]` in the existing batched prefill path. The
    /// two target logits and target hidden rows are produced from one 40-layer
    /// backbone traversal, which is where MTP's decode speedup would come from.
    ///
    /// It does not currently come out ahead, and the reason is structural
    /// rather than a tuning problem. On a sparse MoE the cost of a verify pass
    /// tracks the *union* of the experts its rows route to, not the row count:
    /// rows sharing an expert ride along on one weight read (the grouping in
    /// `PrefillMoEGrouping` sorts by expert so this already happens), rows that
    /// do not each pay in full. Measured on Qwen3.6-35B-A3B, 40 layers,
    /// topK=8 of 256:
    ///
    ///     width 1   8.00 experts/layer   cost 1.000x
    ///     width 2  12.68 experts/layer   cost 1.585x   <- verifyGreedyPair
    ///
    /// Against that, acceptance of 57.4% emits 1.574 tokens per pass. Cost
    /// 1.585 versus benefit 1.574: the two cancel, and every other per-pass
    /// overhead turns it into a net loss (~0.85x end to end).
    ///
    /// Widening the block does not rescue it. Benefit is a geometric series
    /// capped at 1/(1-p) = 2.35, while the union keeps growing -- measured
    /// 5.18x at width 13 and 11.25x at width 42. Width 2 is the closest this
    /// model ever gets to break-even, and it still misses.
    ///
    /// So the lever is acceptance, not the verify path: p must exceed ~0.585
    /// merely to break even. Faster projections cannot help -- the attention
    /// side already amortizes across both rows via `useTwoRowProjection`, and
    /// the expert side is bounded by the union above, not by matmul shape.
    /// Parsed once: the schedule cannot change mid-process, and
    /// ProcessInfo.environment is a dictionary copy per call.
    static let mtpVerifyScheduleResult =
        Result { try RuntimeMTPVerifySchedule.environmentValue() }

    func verifyGreedyPair(_ tokens: [Int32],
                          startPosition: Int) async throws -> TargetPairVerification {
        guard tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported("MTP verification requires exactly two tokens")
        }
        let schedule = try Self.mtpVerifyScheduleResult.get()
        // The pair schedule plans the union of both rows' experts as one
        // cache plan, which needs the slot cache to hold at least 2*topK.
        // Below that (a sub-1 GiB budget) the tile path remains correct.
        let slotCount = model.routedExpertCacheSlotCount() ?? 0
        let pairMoE = schedule == .pair && slotCount >= 2 * cfg.topKExperts
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: config)
        let tBackbone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try await executePrefillChunk(tokens: tokens[...],
                                      startPosition: startPosition,
                                      outputMode: .logits,
                                      logits: verificationLogits,
                                      scratch: scratch,
                                      config: config,
                                      writeFinalHead: false,
                                      snapshotGDNAfterFirstToken: true,
                                      useTwoRowProjection: true,
                                      pairRoutedMoE: pairMoE)
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let cb = ctx.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: scratch.hidden,
                  sourceOffset: 0,
                  to: verificationHidden,
                  destinationOffset: 0,
                  size: 2 * cfg.hiddenSize * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        // One lm_head weight read for both rows. The former per-row loop
        // read the model's largest tensor twice per verify pass.
        try prefillFinalRowHead.encodeLogitsPair(
            commandBuffer: cb,
            hiddenBlock: scratch.hidden,
            rowStrideElements: cfg.hiddenSize,
            normWeight: finalNorm.buffer,
            normWeightOffset: Int(finalNorm.offset),
            weights: lm.buffer,
            weightsOffset: Int(lm.offset),
            scales: lm.buffer,
            scalesOffset: Int(lm.scaleOffset),
            biases: lm.buffer,
            biasesOffset: Int(lm.biasOffset),
            logits: verificationLogits,
            d: UInt32(cfg.hiddenSize),
            vocab: UInt32(cfg.vocabSize),
            rmsEps: 1e-6)
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: "verify_head", cb)
        let tArgmax = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        let logits = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        func argmax(row: Int) -> Int32 {
            let base = row * cfg.vocabSize
            var best = 0
            var bestValue = Float(logits[base])
            for index in 1..<cfg.vocabSize {
                let value = Float(logits[base + index])
                if value > bestValue {
                    bestValue = value
                    best = index
                }
            }
            return Int32(best)
        }
        let first = argmax(row: 0)
        let second = argmax(row: 1)
        let tDone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        return TargetPairVerification(
            predictionAfterFirst: first,
            predictionAfterSecond: second,
            hiddenRows: Data(bytes: verificationHidden.contents(),
                             count: 2 * cfg.hiddenSize * MemoryLayout<Float16>.stride),
            backboneNanos: tHead &- tBackbone,
            headNanos: tArgmax &- tHead,
            argmaxNanos: tDone &- tArgmax)
    }

    /// Advance the one-layer MTP sidecar with aligned `(target hidden,
    /// next-token)` pairs. At most 32 rows are admitted so adapter scratch is
    /// fixed and the routed expert cache remains exactly top-k sized.
    func advanceMTP(tokens: ArraySlice<Int32>,
                    targetHiddenRows: Data,
                    startPosition: Int,
                    predictNext: Bool) async throws -> Int32? {
        guard cfg.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= Self.mtpChunkCapacity else {
            throw PrefillError.chunkedUnsupported(
                "MTP adapter accepts 1...\(Self.mtpChunkCapacity) aligned rows")
        }
        let D = cfg.hiddenSize
        let expectedBytes = tokens.count * D * MemoryLayout<Float16>.stride
        guard targetHiddenRows.count == expectedBytes else {
            throw PrefillError.chunkedUnsupported(
                "MTP target hidden payload has \(targetHiddenRows.count) bytes; expected \(expectedBytes)")
        }
        guard let tokenBuffer = mtpTokenBlock,
              let embeddingBlock = mtpEmbeddingBlock,
              let normalizedEmbedding = mtpNormalizedEmbeddingBlock,
              let normalizedHidden = mtpNormalizedHiddenBlock,
              let concat = mtpConcatBlock,
              let projected = mtpProjectedBlock,
              let targetHidden = mtpTargetHiddenBlock,
              let elementwise else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        targetHiddenRows.copyBytes(to: targetHidden.contents()
            .assumingMemoryBound(to: UInt8.self), count: expectedBytes)
        let ids = tokens.map { UInt32(bitPattern: $0) }
        ids.withUnsafeBytes { bytes in
            tokenBuffer.contents().copyMemory(from: bytes.baseAddress!,
                                              byteCount: bytes.count)
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let emb = try model.embedding()
        try prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: embeddingBlock,
                            t: UInt32(tokens.count),
                            d: UInt32(D),
                            outScale: 1,
                            vocab: UInt32(cfg.vocabSize))
        let embeddingNorm = try model.mtpEmbeddingNorm()
        let hiddenNorm = try model.mtpHiddenNorm()
        try prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: embeddingBlock,
                               weight: embeddingNorm.buffer,
                               weightOffset: Int(embeddingNorm.offset),
                               out: normalizedEmbedding,
                               t: UInt32(tokens.count),
                               d: UInt32(D), eps: 1e-6)
        try prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: targetHidden,
                               weight: hiddenNorm.buffer,
                               weightOffset: Int(hiddenNorm.offset),
                               out: normalizedHidden,
                               t: UInt32(tokens.count),
                               d: UInt32(D), eps: 1e-6)
        try elementwise.encodeConcatRows(commandBuffer: cb,
                                     lhs: normalizedEmbedding,
                                     rhs: normalizedHidden,
                                     out: concat,
                                     rows: tokens.count,
                                     dim: D)
        let projection = try model.mtpProjection()
        try prefillQMM.encode(commandBuffer: cb,
                          weights: projection.buffer,
                          weightsOffset: Int(projection.offset),
                          scales: projection.buffer,
                          scalesOffset: Int(projection.scaleOffset),
                          biases: projection.buffer,
                          biasesOffset: Int(projection.biasOffset),
                          x: concat,
                          y: projected,
                          t: tokens.count,
                          n: D,
                          k: 2 * D)
        cb.commit()
        try waitForCompletion(cb)

        let runtime = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: runtime)
        let mode: PrefillOutputMode = useFusedGreedyHead ? .greedyIfAvailable : .logits
        try await executePrefillChunk(tokens: tokens,
                                      startPosition: startPosition,
                                      outputMode: mode,
                                      logits: verificationLogits,
                                      scratch: scratch,
                                      config: runtime,
                                      writeFinalHead: predictNext,
                                      preparedHidden: projected)
        guard predictNext else { return nil }
        if useFusedGreedyHead {
            return Int32(bitPattern: lastGreedyToken)
        }
        let values = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<cfg.vocabSize {
            let value = Float(values[index])
            if value > bestValue {
                best = index
                bestValue = value
            }
        }
        return Int32(best)
    }

    func ensureMTPPrefillReadback(rows: Int) throws -> MTLBuffer {
        let bytes = rows * cfg.hiddenSize * MemoryLayout<Float16>.stride
        if let existing = mtpPrefillReadback, existing.length >= bytes {
            return existing
        }
        guard let buffer = ctx.device.makeBuffer(length: bytes,
                                                 options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = "mtp.target-hidden-readback"
        mtpPrefillReadback = buffer
        return buffer
    }

    /// Target prefill with a bounded hidden-state tap that simultaneously
    /// aligns the streaming MTP sidecar. Only one target chunk is exposed at a
    /// time; no prompt-sized hidden-state tensor is retained.
    func prefillChunkedWithMTP(tokens: ArraySlice<Int32>,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               mtp: RealForwardRunner,
                               onProgress: (Int) -> Void) async throws -> MTPPrefillResult {
        guard cfg.family == .qwen36 else {
            throw StreamingMTPError.targetMustBeQwen36
        }
        guard mtp.cfg.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= mtp.maxContext else {
            throw PrefillError.chunkedUnsupported(
                "MTP prompt must fit its bounded \(mtp.maxContext)-token draft context")
        }
        reset()
        mtp.reset()
        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: 0,
                                              config: config)
        var carry: Data?
        do {
            for (spanIndex, span) in spans.enumerated() {
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                let chunk = tokens[lower..<upper]
                try await executePrefillChunk(tokens: chunk,
                                              startPosition: span.startPosition,
                                              outputMode: useFusedGreedyHead
                                                ? .greedyIfAvailable : .logits,
                                              logits: logits,
                                              scratch: scratch,
                                              config: config,
                                              writeFinalHead: spanIndex == spans.count - 1)

                let readback = try ensureMTPPrefillReadback(rows: span.tokenCount)
                guard let cb = ctx.queue.makeCommandBuffer(),
                      let blit = cb.makeBlitCommandEncoder() else {
                    throw ModelError.residentBufferWrapFailed
                }
                let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
                blit.copy(from: scratch.hidden, sourceOffset: 0,
                          to: readback, destinationOffset: 0,
                          size: span.tokenCount * rowBytes)
                blit.endEncoding()
                cb.commit()
                try waitForCompletion(cb)
                let chunkHidden = Data(bytes: readback.contents(),
                                       count: span.tokenCount * rowBytes)

                var pairTokens: [Int32] = []
                var pairHidden = Data()
                if let carry {
                    pairTokens.reserveCapacity(span.tokenCount)
                    pairTokens.append(contentsOf: chunk)
                    pairHidden.reserveCapacity(span.tokenCount * rowBytes)
                    pairHidden.append(carry)
                    if span.tokenCount > 1 {
                        pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                    }
                } else if span.tokenCount > 1 {
                    pairTokens.append(contentsOf: chunk.dropFirst())
                    pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                }
                var pairOffset = 0
                while pairOffset < pairTokens.count {
                    let count = min(Self.mtpChunkCapacity, pairTokens.count - pairOffset)
                    let hiddenStart = pairOffset * rowBytes
                    let hiddenEnd = hiddenStart + count * rowBytes
                    _ = try await mtp.advanceMTP(
                        tokens: pairTokens[pairOffset..<(pairOffset + count)],
                        targetHiddenRows: pairHidden.subdata(in: hiddenStart..<hiddenEnd),
                        startPosition: mtp.continuationPosition,
                        predictNext: false)
                    pairOffset += count
                }
                carry = Data(chunkHidden.suffix(rowBytes))
                onProgress(span.completedCount)
            }
        } catch {
            // A failed chunk (cancellation, GPU error, expert-fetch I/O) may
            // have left partial KV rows in both runners; clear both so the
            // next request starts clean.
            reset()
            mtp.reset()
            throw error
        }
        guard let lastTargetHidden = carry else {
            throw StreamingMTPError.draftNotReady
        }
        let seed: PrefillSeed = useFusedGreedyHead
            ? .greedyToken(lastGreedyToken) : .logitsWritten
        return MTPPrefillResult(
            target: PrefillResult(newPosition: tokens.count, seed: seed),
            lastTargetHidden: lastTargetHidden)
    }

    /// Routed-MoE stage for the width-2 MTP verify pass (B2 pair schedule).
    ///
    /// Replaces the prefill tile scheduler for exactly this shape. One union
    /// cache plan covers both rows' experts, so a shared expert is read from
    /// SSD once; the miss fetch runs as one parallel batch overlapped with the
    /// shared-expert GPU work instead of per-tile awaits behind a synchronous
    /// shared-expert wait; and the routed math uses the decode phase-1/phase-2
    /// kernels per row, which B1 measured at roughly a third of the grouped
    /// tile kernels' GPU cost at width 2. Numerics are unchanged: phase 2
    /// reduces each row's experts in router order with the shared branch as
    /// its residual, exactly as decode does.
    ///
    /// lint:allow-long one layer's verify-MoE stage is a single ordered
    /// pipeline in the same shape as its decode and tile siblings: route
    /// readback, union plan, overlapped fetch, per-row encode, commit.
    func encodeRoutedMoEVerifyPair(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        hiddenSize D: Int
    ) async throws {
        let t = 2
        let topK = UInt32(cfg.topKExperts)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let halfBytes = MemoryLayout<Float16>.stride
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (onesPerExpertScale!, 0)
        try prefillRouter.encodeBlock(
                    commandBuffer: cb,
                    weights: views.router.buffer,
                    weightsOffset: Int(views.router.offset),
                    scales: views.router.buffer,
                    scalesOffset: Int(views.router.scaleOffset),
                    biases: views.router.buffer,
                    biasesOffset: Int(views.router.biasOffset),
                    hidden: scratch.routedX,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: perExpertScale.offset,
                    outIndices: scratch.routeIDs,
                    outWeights: scratch.routeWeights,
                    queryCount: UInt32(t),
                    numExperts: UInt32(cfg.numExperts),
                    d: UInt32(D),
                    topK: topK,
                    hiddenStrideElements: UInt32(D))
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: cfg.layerIsLinear(L) ? "prefill_gdn_router"
                            : "prefill_attn_router", cb)

        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: t * cfg.topKExperts)
        var rowExperts = [[Int]](repeating: [], count: t)
        var union: [Int] = []
        var unionIndex: [Int: Int] = [:]
        for row in 0..<t {
            for k in 0..<cfg.topKExperts {
                let expert = min(Int(idPtr[row * cfg.topKExperts + k]),
                                 cfg.numExperts - 1)
                rowExperts[row].append(expert)
                if unionIndex[expert] == nil {
                    unionIndex[expert] = union.count
                    union.append(expert)
                }
            }
        }

        let plan = try model.planRoutedExperts(layer: L, experts: union)
        let lease = try plan.map { try model.pinRoutedExperts(for: $0) }
        var leaseTransferred = false
        defer { if !leaseTransferred { lease?.release() } }

        // Shared expert for both rows, committed WITHOUT a host wait so its
        // GPU work overlaps the union miss fetch below. The tile path's
        // synchronous wait here was one of B1's three structural findings.
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                            x: scratch.routedX,
                                            y: scratch.h1,
                                            gate: sharedProj.gate,
                                            up: sharedProj.up,
                                            down: sharedProj.down,
                                            scratchGate: scratch.sharedGateScratch,
                                            scratchUp: scratch.sharedUpScratch,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t,
                                            d: D,
                                            intermediate: cfg.intermediateSize,
                                            xStrideElements: D,
                                            yStrideElements: D)
        if cfg.sharedExpertGated {
            let gateView = sharedProj.scalarGate!
            for row in 0..<t {
                try int8ScalarGate!.encode(
                    commandBuffer: sharedCB,
                    weights: gateView.buffer,
                    weightsOffset: Int(gateView.offset),
                    scales: gateView.buffer,
                    scalesOffset: Int(gateView.scaleOffset),
                    biases: gateView.buffer,
                    biasesOffset: Int(gateView.biasOffset),
                    x: scratch.routedX,
                    xOffset: row * D * halfBytes,
                    y: scratch.sharedScalarGate,
                    yOffset: row * halfBytes,
                    m: 1, n: UInt32(D))
            }
            for row in 0..<t {
                try elementwise!.encodeSigmoidScalarMul(
                    commandBuffer: sharedCB,
                    y: scratch.h1,
                    yOffset: row * D * halfBytes,
                    gate: scratch.sharedScalarGate,
                    gateOffset: row * halfBytes,
                    count: D)
            }
        }
        sharedCB.commit()

        let blobs: [TensorView]
        if let plan {
            if plan.misses.isEmpty {
                blobs = try model.routedExpertBuffers(for: plan)
            } else {
                let load = try model.beginFetchRoutedExperts(plan: plan)
                blobs = try await load.completion()
            }
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: union)
        }

        while verifyPairActs.count < t {
            guard let made = ctx.device.makeBuffer(
                length: cfg.topKExperts * cfg.moeIntermediateSize * halfBytes,
                options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.acts.\(verifyPairActs.count)"
            verifyPairActs.append(made)
        }
        while verifyPairY.count < t {
            guard let made = ctx.device.makeBuffer(
                length: D * halfBytes,
                options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.y.\(verifyPairY.count)"
            verifyPairY.append(made)
        }
        while verifyPairArgBuffers.count < t {
            guard let made = moe.makeEmptyRoutedArgumentBuffer(device: ctx.device) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "verify.pair.args.\(verifyPairArgBuffers.count)"
            verifyPairArgBuffers.append(made)
        }

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        var rowBlobBuffers: [[MTLBuffer]] = []
        for row in 0..<t {
            var rowBufs: [MTLBuffer] = []
            var rowOffsets: [Int] = []
            rowBufs.reserveCapacity(cfg.topKExperts)
            rowOffsets.reserveCapacity(cfg.topKExperts)
            for expert in rowExperts[row] {
                let view = blobs[unionIndex[expert]!]
                rowBufs.append(view.buffer)
                rowOffsets.append(Int(view.offset))
            }
            rowBlobBuffers.append(rowBufs)
            let argBuf = verifyPairArgBuffers[row]
            moe.writeRoutedArgumentBuffer(argBuf,
                                          routedBlobs: rowBufs,
                                          topK: topK,
                                          routedBufferOffsets: rowOffsets)
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: rowBufs,
                routedOffsets: routedOffsets,
                x: scratch.routedX,
                xOffset: row * D * halfBytes,
                acts: verifyPairActs[row],
                d: UInt32(D),
                f: FmoE,
                topK: topK)
        }
        for row in 0..<t {
            try moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: verifyPairArgBuffers[row],
                routedBlobs: rowBlobBuffers[row],
                routedOffsets: routedOffsets,
                acts: verifyPairActs[row],
                routingWeights: scratch.routeWeights,
                routingWeightsOffset: row * cfg.topKExperts * halfBytes,
                residual: scratch.h1,
                residualOffset: row * D * halfBytes,
                y: verifyPairY[row],
                d: UInt32(D),
                f: FmoE,
                topK: topK)
        }
        // Phase 2 already folded the shared branch (h1 rows as residual), so
        // the tail is one residual add per row — the writes into `hidden`
        // serialize on each other, but they are elementwise and tiny.
        for row in 0..<t {
            try elementwise!.encodeResidualAdd(commandBuffer: routedCB,
                                           hidden: scratch.hidden,
                                           hiddenOffset: row * D * halfBytes,
                                           delta: verifyPairY[row],
                                           count: D)
        }
        routedCB.commit()
        try waitForCompletion(routedCB)
        recordKernelGPU(role: "prefill_shared_expert", sharedCB)
        recordKernelGPU(role: "verify_routed_pair", routedCB)
        lease?.release()
        leaseTransferred = true

        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = nextCB
        }
    }
}
