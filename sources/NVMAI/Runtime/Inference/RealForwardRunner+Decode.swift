import Foundation
import Metal

/// Single-token decode: the layer loop, both attention encoders, and the hit/fixup routed-MoE stage.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    /// lint:allow-long the orchestrator for one decode step, in the same
    /// shape as executePrefillChunk: embed, the per-layer dispatch, the head.
    func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        // Decode must not share RAM with an idle ANE context (Track A):
        // prompts that end exactly on a chunk boundary reach here with the
        // last model still resident. No-op when ANE prefill is off or empty.
        let handoverStart = PreadExpertStreamer.wireTraceEnabled
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
        anePrefill?.releaseModels()
        if PreadExpertStreamer.wireTraceEnabled, handoverStart != 0 {
            let ms = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                            - handoverStart) / 1e6
            if ms > 1 {
                FileHandle.standardError.write(Data(
                    "[wire] releaseModels \(String(format: "%.1f", ms)) ms\n".utf8))
            }
        }
        // Snapshot expert I/O at the handover so decode's share can be
        // separated from prefill's. The two phases stream through the same
        // cache, so a whole-request total cannot answer whether ANE prefill
        // leaves decode re-reading experts -- which is the standing claim
        // ("94% more expert I/O after ANE prefill") that has never been
        // tested directly.
        if Self.decodeIOTraceEnabled, decodeIOBaseline == nil {
            decodeIOBaseline = model.routedExpertStatistics()
        }
        // Wire the slot cache for decode. Unwired, unrelated memory churn can
        // reclaim the budget and decode then re-reads routed experts from SSD
        // for the rest of the request -- measured at 94% more expert I/O after
        // ANE prefill.
        //
        // This is the only place the cache is ever wired: the release at
        // prefill start is a no-op (see there). Wiring costs ~136 ms for
        // 4.2 GiB across 40 layers, so it is not itself a decode cost --
        // wiring at allocation instead measured identically (-28.5% against
        // -27.9% for 4-bit ANE decode), which is why no wiring policy ships.
        model.setExpertCachePinned(true)
        try kv?.reserve(tokens: position + 1)
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        // A decode step at `position` makes position + 1 keys visible.
        try requireQSAExact(visibleKeys: position + 1)
        let D    = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        var pendingRoutedCommand: PendingRoutedCommand?

        /// Drain a routed layer's command buffers, surfacing any `.error`
        /// (R1/R2): the routed-CB failure must fail the generation rather than
        /// print-and-continue into silently corrupt output. The per-layer call
        /// (waitIfNeeded: false) runs right after the next layer's tailCB
        /// wait, so the routed CBs have completed on the GPU and their spans
        /// are valid — recording them here (not only in the waitIfNeeded
        /// drain) makes NVMAI_KERNEL_STATS cover every layer instead of just
        /// the final layer of each token.

        // Embed lookup + sqrt(H) fused.
        let emb = try model.embedding()
        let embedCB = try runSync { cb in
            if let affineEmbed {
                try affineEmbed.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: UInt32(bitPattern: token),
                             d: D, outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            } else {
                try embedInt4.encode(commandBuffer: cb,
                             table:  emb.buffer, tableOffset:  Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden,
                             tokenId: UInt32(bitPattern: token),
                             d: D,
                             outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            }
        }
        guard embedCB != nil else {
            throw ModelError.residentBufferWrapFailed
        }
        // Entry to a hyper-connection stack: every stream starts from the
        // token embedding. The embed kernel wrote stream 0; replicate it.
        if cfg.hyperConnections.enabled {
            _ = try runSync { cb in
                try elementwise!.encodeHCBroadcast(
                    commandBuffer: cb, streams: hidden,
                    dim: cfg.hiddenSize,
                    streamCount: cfg.hyperConnections.count)
            }
        }
        if let embedCB { recordKernelGPU(role: "embed", embedCB) }
        // The n-gram rows depend only on this token and its predecessors, so
        // the gather can run here, before any layer needs it.
        try gatherPLERows(token: token)
        if activationDumpActive(position: position) {
            dumpActivationToken(token, position: position)
            dumpActivation("embed", hidden, count: residualWidth, position: position)
            if let ple = pleBlock {
                dumpActivation("ple_embedding", ple.embedding,
                               count: cfg.ple.embedDim, position: position)
            }
        }

        for L in 0..<cfg.numLayers {
            // Dumping drains the previous layer's routed command first. The
            // residual is only settled once that has landed, and a dump taken
            // at encode time would read whatever the buffer held before the
            // GPU ran -- which reads exactly like a wrong answer.
            if activationDumpActive(position: position) {
                if let pending = pendingRoutedCommand {
                    try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                    pendingRoutedCommand = nil
                    if L <= dumpLayerLimit {
                        dumpActivation("L\(L - 1)_mlp_out", h2Buf,
                                       count: cfg.hiddenSize, position: position)
                        dumpActivation("L\(L - 1)_moe_acts", moeActs,
                                       count: cfg.topKExperts
                                           * cfg.moeIntermediateSize,
                                       position: position)
                    }
                }
                dumpActivation("L\(L)_entry", hidden, count: residualWidth, position: position)
            }
            let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let isLinear = cfg.layerIsLinear(L)

            let inNorm   = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let routerW  = try model.router(layer: L)
            let nextRouterW: TensorView?
            if nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                nextRouterW = try model.router(layer: L + 1)
            } else {
                nextRouterW = nil
            }
            let residencyResources = decodeExpertExecution == .gpuResidency
                ? try model.routedExpertResidency(layer: L) : nil
            let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
                (onesPerExpertScale!, 0)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Attention+router split into measured sub-command-buffers
            // (NVMAI_KERNEL_STATS): attnCB = input norm + QKV + epilogue
            // (or the linear/gated attention), softmaxCB = the softmax
            // attention pass on full layers, tailCB = O-proj + residual +
            // post-norm + router. Same queue, same order, one wait on the
            // last CB; only the router readback forces the barrier.
            guard var attnCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            // The PLE block rewrites the wide residual before the attention
            // read gate sees it, so it is encoded ahead of the entry.
            try encodePLEDecode(commandBuffer: attnCB, layer: L,
                                position: position, eps: eps)
            // Full-attention layers of a sparse-attention family run their
            // entry ahead of the rest, because the indexer's key cache and
            // its selection both hang off the block input.
            //
            // That entry can run on its own command buffer, committed before
            // this one, so a PLE block encoded here would rewrite the
            // residual *after* the entry had already read it. The two never
            // coincide in this architecture -- the n-gram layer is a linear
            // one -- and the guard is here so that stays a fact rather than
            // an assumption.
            precondition(!(qsaIndexer != nil
                           && cfg.fullAttentionLayerMask[L] == 1
                           && cfg.ple.layerIndices.contains(L)),
                         "layer \(L) is both a PLE layer and a sparse-attention "
                             + "layer; the residual entry would be reordered "
                             + "around the n-gram block")
            var keepMask: MTLBuffer?
            if qsaIndexer != nil, cfg.fullAttentionLayerMask[L] == 1 {
                keepMask = try encodeQSAEntryAndSelect(
                    passthrough: attnCB,
                    hidden: hidden, norm: inNorm, out: normed,
                    layer: L, position: position, eps: eps)
            } else {
                try encodeResidualEntryDecode(commandBuffer: attnCB,
                                              hidden: hidden, norm: inNorm,
                                              out: normed, sublayer: .attention,
                                              layer: L, eps: eps)
            }
            var softmaxCB: MTLCommandBuffer?
            guard let tailCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }

            try encodeDecodeAttention(attnCB: &attnCB, tailCB: tailCB,
                                      softmaxCB: &softmaxCB,
                                      layer: L, position: position,
                                      isLinear: isLinear, rmsEps: eps,
                                      keepMask: keepMask)
            try encodeResidualExitDecode(commandBuffer: tailCB,
                                         hidden: hidden, delta: oOut,
                                         sublayer: .attention, layer: L)
            try encodeResidualEntryDecode(commandBuffer: tailCB,
                                          hidden: hidden, norm: postAttn,
                                          out: routedX, sublayer: .mlp,
                                          layer: L, eps: eps)

            try moe.encodeRouter(commandBuffer: tailCB,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            if let nextRouterW {
                // Probe only: score the next router against the current
                // post-attention normalized residual. The exact router above
                // remains authoritative; this result never selects experts for
                // this layer. It drives NVMAI_PREFETCH_TRACE and, when
                // NVMAI_PREDICTIVE_PREFETCH is set, the speculative ring.
                try moe.encodeRouter(commandBuffer: tailCB,
                    weights: nextRouterW.buffer, weightsOffset: Int(nextRouterW.offset),
                    scales: nextRouterW.buffer, scalesOffset: Int(nextRouterW.scaleOffset),
                    biases: nextRouterW.buffer, biasesOffset: Int(nextRouterW.biasOffset),
                    hidden: routedX,
                    effectiveScale: effectiveScaleBuffers[L + 1],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: perExpertScale.offset,
                    outIndices: prefetchPredictionIndices,
                    outWeights: prefetchPredictionWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
            }
            if let residencyResources {
                try moe.encodeResidencyClassification(
                    commandBuffer: tailCB,
                    topKIndices: outIndices,
                    residencyTable: residencyResources.table,
                    hitCount: residencyHitCount,
                    hitPositions: residencyHitPositions,
                    missCount: residencyMissCount,
                    missPositions: residencyMissPositions,
                    missExperts: residencyMissExperts,
                    resolvedSlots: residencyResolvedSlots,
                    resolvedGenerations: residencyResolvedGenerations,
                    topK: UInt32(cfg.topKExperts),
                    numExperts: UInt32(cfg.numExperts))
            }
            attnCB.commit()
            if let attentionCB = softmaxCB {
                attentionCB.commit()
            }
            tailCB.commit()
            // Queued before the wait below, not after: the GPU runs the shared
            // MLP while the CPU blocks on tailCB for the routing.
            let overlapCompletionClock = runnerStatsEnabled ? CommandCompletionClock() : nil
            let sharedCB = try encodeAndCommitSharedExpert(
                layer: L,
                completionClock: overlapCompletionClock)
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try waitForCompletion(tailCB)
            // After the tail wait every attention-side buffer is settled and
            // nothing overwrites them until the next layer, so the dumps cost
            // no extra synchronization.
            if activationDumpActive(position: position) && L <= dumpLayerLimit {
                if let ple = pleBlock, cfg.ple.layerIndices.contains(L) {
                    let wide = residualWidth
                    dumpActivation("ple_key", ple.keyBuf, count: wide, position: position)
                    dumpActivation("ple_value", ple.valueBuf, count: cfg.hiddenSize, position: position)
                    dumpActivation("ple_key_normed", ple.keyNormed, count: wide, position: position)
                    dumpActivation("ple_query", ple.queryNormed, count: wide, position: position)
                    dumpActivation("ple_score", ple.scoreBuf,
                                   count: cfg.hyperConnections.count, position: position)
                    dumpActivation("ple_gate", ple.gateBuf,
                                   count: cfg.hyperConnections.count, position: position)
                    dumpActivation("ple_gated", ple.gatedBuf, count: wide, position: position)
                    dumpActivation("ple_conv", ple.convOut, count: wide, position: position)
                }
                dumpActivation("L\(L)_attn_in", normed, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_attn_out", oOut, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_mlp_in", routedX, count: cfg.hiddenSize, position: position)
                dumpActivation("L\(L)_hidden_post_attn", hidden,
                               count: residualWidth, position: position)
            }
            recordKernelGPU(role: "attn_norm_qkv", attnCB)
            // Split-mode kernels were committed ahead of attnCB on the same
            // queue, so they completed before the tail wait above.
            for (role, cb) in splitTimedBuffers { recordKernelGPU(role: role, cb) }
            splitTimedBuffers.removeAll(keepingCapacity: true)
            if let attentionCB = softmaxCB {
                recordKernelGPU(role: "attn_softmax", attentionCB)
            }
            recordKernelGPU(role: "attn_tail_router", tailCB)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            totalWaitNanos &+= waitNanos
            var prevRoutedUs: Double = 0
            if let pending = pendingRoutedCommand {
                prevRoutedUs = (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1_000_000
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos
            let predictedNextLayer: [Int]
            if nextLayerPredictionEnabled, L + 1 < cfg.numLayers {
                let ptr = prefetchPredictionIndices.contents().bindMemory(
                    to: UInt32.self, capacity: cfg.topKExperts)
                predictedNextLayer = (0..<cfg.topKExperts).map {
                    min(Int(ptr[$0]), cfg.numExperts - 1)
                }
            } else {
                predictedNextLayer = []
            }

            // CPU readback to fetch routed-expert blobs from disk. The expert
            // id list is reused host scratch (R16); the runner is single-flight
            // per generation, so it never aliases concurrent decode work.
            try await encodeDecodeRoutedMoE(
                layer: L, position: position, sharedProj: sharedProj,
                attnCB: attnCB, tailCB: tailCB,
                sharedCB: sharedCB,
                overlapCompletionClock: overlapCompletionClock,
                pending: &pendingRoutedCommand,
                bodyStart: tBodyStart, cb1Start: tCb1Start,
                waitMark: tWait, waitNanos: waitNanos,
                previousRoutedMicros: prevRoutedUs,
                predictedNextLayer: predictedNextLayer)
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = try model.finalNorm()
        let lm    = try model.lmHead()
        let gFinalNorm: (MTLCommandBuffer) throws -> Void = { cb in
            if let hc = self.hyperConnection {
                // The stack ends by collapsing the streams through the
                // model-level mixer: the same gated read a sublayer uses, with
                // no inject, and its hc_norm serving as the final norm.
                try hc.encodeRead(commandBuffer: cb,
                                  streamsBuffer: self.hidden,
                                  hcNorm: fNorm.buffer,
                                  hcNormOffset: Int(fNorm.offset),
                                  down: self.gateWeightsPublic(try self.model.hcMixerDown()),
                                  up: self.gateWeightsPublic(try self.model.hcMixerUp()),
                                  blockInput: self.normed, eps: eps)
            } else {
                try self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                     weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                     out: self.normed, d: D, eps: eps)
            }
        }
        let gLmHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodeHeadGEMV(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if activationDumpActive(position: position) {
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                pendingRoutedCommand = nil
            }
            dumpActivation("stack_out", hidden, count: residualWidth, position: position)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                if let headCB = try runSync(gFusionHead) {
                    recordKernelGPU(role: "head_fused", headCB)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                guard let headCB = try runSync({ cb in
                    try gFinalNorm(cb)
                    try gLmHead(cb)
                }) else {
                    throw ModelError.residentBufferWrapFailed
                }
                recordKernelGPU(role: "head_logits", headCB)
                if activationDumpActive(position: position) {
                    dumpActivation("mixer_out", normed, count: cfg.hiddenSize,
                                   position: position)
                    dumpActivation("logits", logits, count: cfg.vocabSize,
                                   position: position)
                }
                // The last prompt position's logits, however prefill produced
                // them: the one place a batched path and the sequential
                // oracle can be compared as numbers rather than as text.
                if activationDumpDirectory != nil, emitHead {
                    dumpActivation("prefill_logits", logits,
                                   count: cfg.vocabSize, position: 0)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    /// Attribution only. NVMAI_ABLATE=<name> skips one kernel of the GDN
    /// decode chain so the attention command buffer's GPU time can be
    /// differenced per kernel. Output is wrong while it is set; the timing is
    /// not, because none of these kernels' cost depends on the data.
    func ablated(_ name: String) -> Bool {
        ProcessInfo.processInfo.environment["NVMAI_ABLATE"] == name
    }

    /// Per-kernel GPU attribution for the GDN decode chain.
    ///
    /// NVMAI_KERNEL_SPLIT=1 gives every kernel of this chain its own command
    /// buffer, committed in order on the same queue, so `recordKernelGPU` can
    /// time each one rather than the chain as a whole. Data stays valid --
    /// unlike ablation, which feeds garbage downstream and measures NaN
    /// slow-paths and shortened generations instead of the kernel. The split
    /// buffers are collected here and recorded after the layer's tail wait,
    /// where their GPU timestamps exist.
    var splitKernelTiming: Bool {
        ProcessInfo.processInfo.environment["NVMAI_KERNEL_SPLIT"] == "1"
    }

    private func rotate(_ cb: inout MTLCommandBuffer, role: String) throws {
        guard splitKernelTiming else { return }
        cb.commit()
        splitTimedBuffers.append((role, cb))
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
    }

    func encodeLinearAttentionDecode(_ cb: inout MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        if model.attentionWeightBits == 4 {
            if !ablated("inproj") {
        try gdn.encodeInputProjections(commandBuffer: cb,
                                       x: normed,
                                       qkv: qkvW, qkvOut: gdnQKVRaw,
                                       z: zW, zOut: gdnZ,
                                       a: aW, aOut: gdnA,
                                       b: bW, bOut: gdnB,
                                       hiddenSize: cfg.hiddenSize)
        }
        } else {
            try encodePrimaryGEMV(commandBuffer: cb, projection: qkvW,
                              x: normed, y: gdnQKVRaw,
                              m: UInt32(la.qkvDim), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: zW,
                              x: normed, y: gdnZ,
                              m: UInt32(la.valueDim), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: aW,
                              x: normed, y: gdnA,
                              m: UInt32(la.numVHeads), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: bW,
                              x: normed, y: gdnB,
                              m: UInt32(la.numVHeads), n: D)
        }
        try rotate(&cb, role: "gdn.inproj")

        if !ablated("conv") {
        try gdn.encodeConvDecode(commandBuffer: cb,
                                 tail: gdnState.convTailBuffer(layer: L),
                                 qkv: gdnQKVRaw,
                                 convWeight: convW.buffer,
                                 convWeightOffset: Int(convW.offset),
                                 out: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.conv")
        if !ablated("qknorm") {
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        }
        try rotate(&cb, role: "gdn.qknorm")
        if !ablated("delta") {
        try gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                      convOut: gdnConvOut,
                                      aProj: gdnA,
                                      bProj: gdnB,
                                      aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                      dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                      state: gdnState.stateBuffer(layer: L),
                                      y: gdnY)
        }
        try rotate(&cb, role: "gdn.delta")
        if !ablated("gatednorm") {
        try gdn.encodeGatedNorm(commandBuffer: cb,
                                y: gdnY,
                                z: gdnZ,
                                weight: gatedNormW.buffer,
                                weightOffset: Int(gatedNormW.offset),
                                out: gdnOut)
        }
        try rotate(&cb, role: "gdn.gatednorm")
        if !ablated("outproj") {
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
        }
        try rotate(&cb, role: "gdn.outproj")
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    func encodeGatedFullQKVProjection(
        _ cb: MTLCommandBuffer,
        layer: Int,
        qOutput: MTLBuffer,
        kOutput: (buffer: MTLBuffer, offset: Int),
        vOutput: (buffer: MTLBuffer, offset: Int),
        qDimension: UInt32,
        kvDimension: UInt32
    ) throws {
        let q = try model.qProj(layer: layer)
        let k = try model.kProj(layer: layer)
        let v = try model.vProj(layer: layer)
        let hiddenDimension = UInt32(cfg.hiddenSize)
        if model.attentionWeightBits == 4 {
            try fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qOutput,
                            kOut: kOutput.buffer, kOutOffset: kOutput.offset,
                            vOut: vOutput.buffer, vOutOffset: vOutput.offset,
                            qRows: 2 * qDimension,
                            kvRows: kvDimension,
                            n: hiddenDimension)
        } else {
            try encodePrimaryGEMV(commandBuffer: cb, projection: q,
                              x: normed, y: qOutput,
                              m: 2 * qDimension, n: hiddenDimension)
            try encodePrimaryGEMV(commandBuffer: cb, projection: k,
                              x: normed, y: kOutput.buffer,
                              yOffset: kOutput.offset,
                              m: kvDimension, n: hiddenDimension)
            try encodePrimaryGEMV(commandBuffer: cb, projection: v,
                              x: normed, y: vOutput.buffer,
                              yOffset: vOutput.offset,
                              m: kvDimension, n: hiddenDimension)
        }
    }

    func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32,
                                                keepMask: MTLBuffer? = nil) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail: "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "full attention requires a KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let quantizedKV = kv.precision.isQuantized
        let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
        let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        try encodeGatedFullQKVProjection(
            cb, layer: L, qOutput: qPackedScratch,
            kOutput: kWrite, vOutput: vWrite,
            qDimension: qDim, kvDimension: kvDim)
        try elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        try rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: eps)
        try rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kWrite.buffer, xOffset: kWrite.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kWrite.buffer, outOffset: kWrite.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kWrite.buffer,
                              dataOffset: kWrite.offset,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        if quantizedKV {
            try encodeQuantizedKV(commandBuffer: cb, kv: kv, layer: L,
                                  position: position, keySource: kStage,
                                  valueSource: vStage, elementCount: Int(kvDim))
        }
        let keyView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
        let valueView = kv.valueView(layer: L, validTokenCount: Int(seqLen))
        try attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: keyView.buffer, kOffset: keyView.offset,
                             v: valueView.buffer, vOffset: valueView.offset,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale),
                             kvFormat: keyView,
                             keepMask: keepMask)
        try elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   projection p: TensorView,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        // A promoted tensor carries no scales or biases, so it cannot go
        // through the quantized GEMV -- that would read the companions from
        // offset zero, which is the file header, and produce NaN. Decided from
        // the tensor's dtype because promotion is per tensor, not per slot.
        if p.dtype == 1 {
            try bf16Projection.encode(commandBuffer: cb,
                                      weights: p.buffer,
                                      weightsOffset: Int(p.offset),
                                      x: x, xOffset: xOffset,
                                      y: y, yOffset: yOffset,
                                      m: m, n: n)
            return
        }
        try encodePrimaryGEMV(commandBuffer: cb,
                          weights: p.buffer, weightsOffset: Int(p.offset),
                          scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                          biases: p.buffer, biasesOffset: Int(p.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
    }

    func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int,
                                   scales: MTLBuffer, scalesOffset: Int,
                                   biases: MTLBuffer, biasesOffset: Int,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        if let affine {
            try affine.encode(commandBuffer: cb,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
        } else {
            try int4.encode(commandBuffer: cb,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        }
    }

    /// Vocabulary head GEMV. Same shape as `encodePrimaryGEMV` but keyed off
    /// the head's own quantization, which a model may set independently of
    /// the attention slot.
    func encodeHeadGEMV(commandBuffer cb: MTLCommandBuffer,
                        weights: MTLBuffer, weightsOffset: Int,
                        scales: MTLBuffer, scalesOffset: Int,
                        biases: MTLBuffer, biasesOffset: Int,
                        x: MTLBuffer, xOffset: Int = 0,
                        y: MTLBuffer, yOffset: Int = 0,
                        m: UInt32, n: UInt32) throws {
        if let affineHead {
            try affineHead.encode(commandBuffer: cb,
                                  weights: weights, weightsOffset: weightsOffset,
                                  scales: scales, scalesOffset: scalesOffset,
                                  biases: biases, biasesOffset: biasesOffset,
                                  x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                                  m: m, n: n)
        } else {
            try int4.encode(commandBuffer: cb,
                            weights: weights, weightsOffset: weightsOffset,
                            scales: scales, scalesOffset: scalesOffset,
                            biases: biases, biasesOffset: biasesOffset,
                            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                            m: m, n: n)
        }
    }

    func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> MTLCommandBuffer? {
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
        return cb
    }

    /// Attention stage of one decode layer: the gated-DeltaNet branch or the
    /// softmax branch, both writing into `oOut` for the residual add.
    ///
    /// lint:allow-long the two branches are alternatives over the same set of
    /// scratch buffers; splitting them apart again would only re-create the
    /// dispatch this method exists to hold.
    func encodeDecodeAttention(
        attnCB: inout MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        softmaxCB: inout MTLCommandBuffer?,
        layer L: Int,
        position: Int,
        isLinear: Bool,
        rmsEps eps: Float,
        keepMask: MTLBuffer? = nil
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let isFull = cfg.fullAttentionLayerMask[L] == 1
        let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
        let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim     = UInt32(cfg.numHeads * headDimL)
        let kvDim    = UInt32(numKVL * headDimL)
        let seqLen   = UInt32(position + 1)
        if isLinear {
            // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
            // fixed-size recurrent state updated in place.
            try encodeLinearAttentionDecode(&attnCB, layer: L)
        } else if cfg.attnOutputGate {
            // Qwen full attention: packed [query ; gate] q_proj, real
            // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
            try encodeGatedFullAttentionDecode(attnCB, layer: L,
                                               position: position,
                                               seqLen: seqLen,
                                               keepMask: keepMask)
        } else {
            let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let quantizedKV = kv?.precision.isQuantized == true
            let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
            let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
            let q     = try model.qProj(layer: L)
            let k     = try model.kProj(layer: L)
            // Under the K=V quirk full layers reuse k_proj; otherwise
            // v_proj is a real tensor.
            let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
            let o     = try model.oProj(layer: L)
            let qNorm = try model.qNorm(layer: L)
            let kNorm = try model.kNorm(layer: L)

            try fusedQKVGEMV.encode(commandBuffer: attnCB,
                                qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                x: normed,
                                qOut: qScratch,
                                kOut: kWrite.buffer, kOutOffset: kWrite.offset,
                                vOut: vWrite.buffer, vOutOffset: vWrite.offset,
                                qRows: qDim,
                                kvRows: kvDim,
                                n: D)

            let rotated = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDimL / 2)
            try fusedQKVEpilogue.encode(commandBuffer: attnCB,
                                    q: qScratch,
                                    k: kWrite.buffer,
                                    kOffset: kWrite.offset,
                                    v: vWrite.buffer,
                                    vOffset: vWrite.offset,
                                    qWeight: qNorm.buffer,
                                    qWeightOffset: Int(qNorm.offset),
                                    kWeight: kNorm.buffer,
                                    kWeightOffset: Int(kNorm.offset),
                                    headDim: UInt32(headDimL),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKVL),
                                    position: UInt32(position),
                                    theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                    rotatedPairs: rotated,
                                    eps: eps)

            guard let kv else {
                throw ModelError.internalInconsistency(
                    detail: "attention requires a KV cache")
            }
            if quantizedKV {
                try encodeQuantizedKV(commandBuffer: attnCB, kv: kv, layer: L,
                                      position: position, keySource: kStage,
                                      valueSource: vStage, elementCount: Int(kvDim))
            }
            let keyView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
            let valueView = kv.valueView(layer: L, validTokenCount: Int(seqLen))
            guard let attentionCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            softmaxCB = attentionCB
            if isFull {
                try attention.encodeFull(commandBuffer: attentionCB,
                                     q: qScratch,
                                     k: keyView.buffer, kOffset: keyView.offset,
                                     v: valueView.buffer, vOffset: valueView.offset,
                                     out: attnOut,
                                     headDim: UInt32(headDimL),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(numKVL),
                                     seqLen: seqLen,
                                     scale: Float(cfg.attentionScale),
                                     kvFormat: keyView)
            } else {
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                try attention.encodeSWA(commandBuffer: attentionCB,
                                    q: qScratch,
                                    k: kSlot.buffer, kOffset: 0,
                                    v: vSlot.buffer, vOffset: 0,
                                    out: attnOut,
                                    headDim: UInt32(headDimL),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKVL),
                                    seqLen: seqLen,
                                    window: UInt32(cfg.slidingWindow),
                                    scale: Float(cfg.attentionScale),
                                    ringCapacity: activeRingCapacity,
                                    kvFormat: keyView)
            }
            try int4.encode(commandBuffer: tailCB,
                        weights: o.buffer, weightsOffset: Int(o.offset),
                        scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                        biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)
        }

        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
    }

    func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                    waitIfNeeded: Bool) throws {
        defer { pending.expertLease?.release() }
        // A staged Metal-I/O batch owns its source buffers until the compute
        // command has completed. If any command/error path exits early, leave
        // the cache entries empty rather than retaining a LOADING slot.
        var finalizedStagingTransfer = false
        defer {
            if let operation = pending.storageOperation {
                if operation.storage.requiresGPUFinalization,
                   !finalizedStagingTransfer {
                    model.failRoutedExpertStagingTransfer(plan: operation.plan)
                }
                operation.storage.releaseStagingTransfer()
            }
        }
        if waitIfNeeded {
            if let sharedCB = pending.sharedCB {
                try waitForCompletion(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try waitForCompletion(phase1HitCB)
            }
            try waitForCompletion(pending.cb)
        } else if let err = pending.cb.error {
            throw ModelError.commandBufferFailed(
                detail: "routed layer command buffer: \(err)")
        }
        if let operation = pending.storageOperation {
            // Event-gated commands cannot complete before this operation is
            // terminal, so this is an error check, not a successful-I/O host
            // wait. A failed read is surfaced after safe no-op kernels have
            // prevented incomplete slot bytes from being dereferenced.
            try operation.storage.wait()
            if operation.storage.requiresGPUFinalization {
                try model.finalizeRoutedExpertStagingTransfer(plan: operation.plan)
                finalizedStagingTransfer = true
            }
            totalIOQueueNanos &+= operation.storage.submissionToStartNanos
            totalIoNanos &+= operation.storage.loadNanos
            totalMissIoNanos &+= operation.storage.loadNanos
            if let latest = pending.overlapCompletionClock?.latest(
                expected: pending.expectedOverlapCompletions) {
                let completed = operation.storage.completedNanos
                if completed > latest {
                    totalExposedIoNanos &+= completed - latest
                }
            }
        }
        if let sharedCB = pending.sharedCB, let err = sharedCB.error {
            throw ModelError.commandBufferFailed(
                detail: "shared-expert command buffer: \(err)")
        }
        if let phase1HitCB = pending.phase1HitCB, let err = phase1HitCB.error {
            throw ModelError.commandBufferFailed(
                detail: "routed phase-1 hit command buffer: \(err)")
        }
        if let sharedCB = pending.sharedCB {
            recordKernelGPU(role: "shared_expert", sharedCB)
        }
        if let phase1HitCB = pending.phase1HitCB {
            recordKernelGPU(role: "moe_phase1_hit", phase1HitCB)
        }
        recordKernelGPU(role: pending.kernelRole, pending.cb)
        totalCb2Nanos &+= pending.encodeAndCommitNanos
    }

    func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<slots.count { ptr[i] = slots[i] }
    }

    /// Encodes the shared dense MLP and commits it immediately.
    ///
    /// It depends only on `routedX`, which `tailCB` produces, so it can be
    /// queued the moment `tailCB` is committed -- before the router readback,
    /// not after it. Both sit on the same queue, so the GPU runs this while the
    /// CPU is blocked waiting for `tailCB` to report the routing.
    ///
    /// That ordering is the whole point. Encoding it after the readback left a
    /// measured 7.88 ms/token of GPU idle in the
    /// `attn_tail_router -> shared_expert` transition -- 0.197 ms per layer of
    /// command-buffer round trip during which the GPU had nothing queued, and
    /// the largest single component of decode's idle time.
    func encodeAndCommitSharedExpert(
        layer L: Int,
        completionClock: CommandCompletionClock?
    ) throws -> MTLCommandBuffer {
        let sharedProj = sharedExpertProjections[L]
        let D = UInt32(cfg.hiddenSize)
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try shared.encode(commandBuffer: sharedCB,
                          x: routedX,
                          gate: sharedProj.gate,
                          up: sharedProj.up,
                          down: sharedProj.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate,
                          scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
            let gateView = sharedProj.scalarGate!
            try encodeScalarGate(commandBuffer: sharedCB,
                                 view: gateView,
                                 x: routedX,
                                 y: sharedScalarGateBuf!,
                                 n: D)
            try elementwise!.encodeSigmoidScalarMul(commandBuffer: sharedCB,
                                                y: h1Buf,
                                                gate: sharedScalarGateBuf!,
                                                count: cfg.hiddenSize)
        }
        completionClock?.track(sharedCB)
        sharedCB.commit()
        return sharedCB
    }

    /// Routed-expert stage of one decode layer: top-k readback, expert fetch,
    /// phase-1/phase-2 encode, and the deferred completion hand-off.
    ///
    /// lint:allow-long one pipeline whose phases share the fetch plan, the
    /// argument buffer and the slot scratch; the layer trace at the end reports
    /// timings from every phase, so splitting it would mean threading those
    /// back out purely to shorten a function.
    func encodeDecodeRoutedMoE(
        layer L: Int,
        position: Int,
        sharedProj: LayerSharedExpertProjections,
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        sharedCB: MTLCommandBuffer,
        overlapCompletionClock: CommandCompletionClock?,
        pending pendingRoutedCommand: inout PendingRoutedCommand?,
        bodyStart tBodyStart: UInt64,
        cb1Start tCb1Start: UInt64,
        waitMark tWait: UInt64,
        waitNanos: UInt64,
        previousRoutedMicros prevRoutedUs: Double,
        predictedNextLayer: [Int]
    ) async throws {
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let readbackStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                      capacity: cfg.topKExperts)
        decodeExpertsScratch.removeAll(keepingCapacity: true)
        decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
        for i in 0..<cfg.topKExperts {
            decodeExpertsScratch.append(min(Int(idxPtr[i]), cfg.numExperts - 1))
        }
        totalRouterReadbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readbackStarted
        let experts = decodeExpertsScratch
        recordRouteTrace(layer: L, position: position, experts: experts)

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let canUsePlannedFetch = cfg.topKExperts <= moe.maxStreamedExperts
        let residentBeforePlan = prefetchTraceFD >= 0
            ? try model.routedExpertResidentIDs(layer: L) : []
        let readyPrefetches = predictivePrefetch?.readyBuffers(layer: L, experts: experts) ?? [:]
        let cachePlanStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedFetch = canUsePlannedFetch
            ? try model.planRoutedExperts(
                layer: L, experts: experts, prefetched: readyPrefetches)
            : nil
        totalCachePlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cachePlanStarted
        if !readyPrefetches.isEmpty {
            predictivePrefetch?.consume(layer: L, experts: Set(readyPrefetches.keys))
        }
        let missesForTrace = plannedFetch.map { plan in
            plan.misses.map { experts[$0] }
        } ?? experts
        recordPrefetchTrace(layer: L, position: position, experts: experts,
                            misses: missesForTrace, resident: residentBeforePlan,
                            nextLayerPrediction: predictedNextLayer)
        let expertLease = try plannedFetch.map { try model.pinRoutedExperts(for: $0) }
        // v4.2 Phase B: once slots and generations are reserved and pinned,
        // submit real storage immediately. Hit partitioning, argument binding,
        // and command encoding below now overlap the reader queue.
        let shouldSubmitImmediately = expertIOSubmission == .immediate
            || expertIOSynchronization == .event
        let plannedLoad = shouldSubmitImmediately
            ? try plannedFetch.map {
                try model.beginFetchRoutedExperts(
                    plan: $0,
                    eventDriven: expertIOSynchronization == .event && !$0.misses.isEmpty)
            }
            : nil
        var transferredExpertLease = false
        var phase1HitCB: MTLCommandBuffer?
        defer {
            if !transferredExpertLease {
                // A thrown fetch/encode must not make a hit slot evictable
                // while its already-committed phase-1 command is still reading.
                if let phase1HitCB, let expertLease {
                    try? waitForCompletion(phase1HitCB)
                    expertLease.release()
                } else {
                    expertLease?.release()
                }
            }
        }
        var phase1HitSplitArgBuf: MTLBuffer?
        decodeHitSplitRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeHitSplitRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        decodeHitSlotsScratch.removeAll(keepingCapacity: true)
        decodeMissSlotsScratch.removeAll(keepingCapacity: true)

        if let plan = plannedFetch,
           (decodeExpertExecution == .hitFixup
                || decodeExpertExecution == .gpuResidency) {
            if decodeExpertExecution == .gpuResidency {
                let hitCount = min(
                    Int(residencyHitCount.contents().load(as: UInt32.self)),
                    cfg.topKExperts)
                let missCount = min(
                    Int(residencyMissCount.contents().load(as: UInt32.self)),
                    cfg.topKExperts)
                let hitPointer = residencyHitPositions.contents()
                    .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
                let missPointer = residencyMissPositions.contents()
                    .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
                for index in 0..<hitCount {
                    decodeHitSlotsScratch.append(hitPointer[index])
                }
                for index in 0..<missCount {
                    decodeMissSlotsScratch.append(missPointer[index])
                }
                // The GPU appends through an atomic counter, so its ordering
                // is the order threads finished -- not the ascending order
                // `DecodeExpertPartition.populate` produces on the CPU. Sort
                // before both the comparison and the use: the encoders below
                // consume these in the CPU path's order, so an unsorted GPU
                // partition would execute a different assignment even when it
                // classified every expert correctly.
                decodeHitSlotsScratch.sort()
                decodeMissSlotsScratch.sort()
                // The GPU classifies against the residency table as it stood
                // when the kernel ran, which is *before* the cache plan adopts
                // prefetched experts. Adoption memcpys a prefetched expert into
                // its slot and drops it from `plan.misses`, so the GPU's list is
                // legitimately the CPU's plus the adoptions -- one per layer at
                // the shipped prefetch depth of 1. Requiring equality made the
                // mode unusable on any build with prefetch enabled.
                //
                // A miss the CPU sees and the GPU does not is the real error:
                // that direction means the table claimed residency for
                // something the eviction authority had already reclaimed.
                let gpuMisses = decodeMissSlotsScratch.map(Int.init)
                let planMisses = plan.misses.sorted()
                guard Set(gpuMisses).isSuperset(of: planMisses) else {
                    throw ModelError.internalInconsistency(
                        detail: "GPU residency classification disagrees with cache "
                            + "plan: gpu=\(gpuMisses) plan=\(planMisses)")
                }
                totalGPUClassifiedHits &+= UInt64(hitCount)
                totalGPUClassifiedMisses &+= UInt64(missCount)
                if missCount == 0 { totalGPUResidencyAllHitLayers &+= 1 }
                // Execute the CPU partition regardless. It is the authority,
                // and it is the only one that reflects adoption; running the
                // GPU's partition would re-fetch an expert already resident.
                DecodeExpertPartition.populate(
                    topK: cfg.topKExperts,
                    missIndices: plan.misses,
                    hits: &decodeHitSlotsScratch,
                    misses: &decodeMissSlotsScratch)
            } else {
                DecodeExpertPartition.populate(
                    topK: cfg.topKExperts,
                    missIndices: plan.misses,
                    hits: &decodeHitSlotsScratch,
                    misses: &decodeMissSlotsScratch)
            }
        }
        // Capture the populated arrays. Capturing them before `populate` made
        // empty value-semantic snapshots and silently disabled hit/fixup.
        let phase1HitSlots = decodeHitSlotsScratch
        let phase1MissSlots = decodeMissSlotsScratch
        func encodeRoutedPhase1Full(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            ioStatus: MTLBuffer? = nil,
            ioStatusOffset: Int = 0
        ) throws {
            try moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                    routedArgBuffer: argBuf,
                                                    routedBlobs: routedBufs,
                                                    routedOffsets: routedOffsets,
                                                    x: routedX,
                                                    acts: moeActs,
                                                    d: D,
                                                    f: FmoE,
                                                    topK: topK,
                                                    ioStatus: ioStatus,
                                                    ioStatusOffset: ioStatusOffset)
        }

        func encodeRoutedPhase1Subset(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            activeSlots: MTLBuffer,
            activeSlotIndices: [UInt32],
            activeCount: UInt32,
            ioStatus: MTLBuffer? = nil,
            ioStatusOffset: Int = 0
        ) throws {
            try moe.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: cb,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                activeSlots: activeSlots,
                activeSlotIndices: activeSlotIndices,
                activeCount: activeCount,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus,
                ioStatusOffset: ioStatusOffset)
        }

        if let plan = plannedFetch,
           plan.hits > 0,
           !plan.misses.isEmpty {
            let plannedBlobs = try model.routedExpertBuffers(for: plan)
            for blob in plannedBlobs {
                decodeHitSplitRoutedBufsScratch.append(blob.buffer)
                decodeHitSplitRoutedOffsetsScratch.append(Int(blob.offset))
            }
            phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: decodeHitSplitRoutedBufsScratch,
                topK: topK,
                routedBufferOffsets: decodeHitSplitRoutedOffsetsScratch)
            if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                guard let cb = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeRoutedPhase1Subset(
                    cb,
                    argBuf: argBuf,
                    routedBufs: decodeHitSplitRoutedBufsScratch,
                    activeSlots: moeHitActiveSlots,
                    activeSlotIndices: phase1HitSlots,
                    activeCount: UInt32(phase1HitSlots.count))
                phase1HitCB = cb
            }
        }

        if let cb = phase1HitCB {
            overlapCompletionClock?.track(cb)
            cb.commit()
        }
        let missCount = plannedFetch?.misses.count ?? experts.count
        let completionClock = missCount > 0 ? overlapCompletionClock : nil
        let expectedOverlapCompletions = phase1HitCB == nil ? 1 : 2
        if plannedLoad == nil && rdadviseEnabled && rdadvisePolicyMode != .off {
            let requestedMisses = missCount
            let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                layer: L,
                missCount: requestedMisses)
            if let skipped = shouldSkipRDAdvice(position: position,
                                                requestedMisses: requestedMisses,
                                                estimatedBytes: estimatedAdviceBytes,
                                                canOverlapUsefulGPUWork: true) {
                recordRDAdvice(skipped, wallNanos: 0)
            } else {
                let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let result: ExpertIOAdviceResult
                if let plannedFetch {
                    result = try model.adviseRoutedExperts(plan: plannedFetch)
                } else {
                    result = try model.adviseRoutedExperts(layer: L, experts: experts)
                }
                let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                recordRDAdvice(result, wallNanos: wallNanos)
                updateRDAdvicePolicy(after: result, position: position)
            }
        }

        // Routed-expert pread — overlaps the shared MLP GPU work above.
        let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        var completedStorageNanos: UInt64?
        let eventLoad = plannedLoad.flatMap { operation -> RoutedExpertLoadOperation? in
            operation.storage.completionToken == nil ? nil : operation
        }
        if let eventLoad {
            // Slot resources and offsets are known from the reservation. Their
            // bytes are consumed only after the shared-event wait encoded
            // below, so no successful completion has to resume this task.
            blobs = try model.routedExpertBuffers(for: eventLoad.plan)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedFetch, plannedFetch.misses.isEmpty {
            // An all-hit layer has already pinned its current slot generations.
            // Do not manufacture a completed storage operation and an async
            // continuation only to retrieve the same cache views.
            blobs = try model.routedExpertBuffers(for: plannedFetch)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedLoad {
            totalExpertIOHostWaits &+= plannedLoad.plan.misses.isEmpty ? 0 : 1
            blobs = try await plannedLoad.completion()
            totalIOQueueNanos &+= plannedLoad.storage.submissionToStartNanos
            completedStorageNanos = plannedLoad.storage.completedNanos
        } else if let plannedFetch {
            // The production deferred schedule still uses the split operation
            // so queueing and completion remain observable. It deliberately
            // begins here, after the independent hit work is committed.
            let deferredLoad = try model.beginFetchRoutedExperts(plan: plannedFetch)
            totalExpertIOHostWaits &+= plannedFetch.misses.isEmpty ? 0 : 1
            blobs = try await deferredLoad.completion()
            totalIOQueueNanos &+= deferredLoad.storage.submissionToStartNanos
            completedStorageNanos = deferredLoad.storage.completedNanos
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        let layerIo = eventLoad == nil
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart : 0
        if eventLoad == nil { totalIoNanos &+= layerIo }
        if missCount > 0 && eventLoad == nil {
            totalMissIoNanos &+= layerIo
            if let latest = completionClock?.latest(expected: expectedOverlapCompletions) {
                let overlapEnd = max(tIoStart, latest)
                if overlapEnd < tIoStart + layerIo {
                    totalExposedIoNanos &+= tIoStart + layerIo - overlapEnd
                }
            }
        }
        if let predictivePrefetch, L + 1 < cfg.numLayers {
            let resident = Set(try model.routedExpertResidentIDs(layer: L + 1))
            // The whole ranked prediction goes in; `begin` drops the experts
            // that are already resident or already in flight and then fills
            // whatever ring slots are free, in rank order.
            //
            // Truncating to top-M here first (v4.3) spent the budget before
            // the residency filter ran. About 85% of the top-M predictions are
            // already cached, so the filter starved the ring instead of aiming
            // it: on a 191-position qwen38 trace, top-4 issued 0.94 reads per
            // layer and covered 23% of the demand misses, where filtering
            // first issues 2.59 and covers 46%. The ring size stays the cap on
            // reads in flight; only the order of cap and filter changed.
            try predictivePrefetch.begin(
                model: model, layer: L + 1,
                experts: predictedNextLayer,
                resident: resident)
        }
        decodeRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        for blob in blobs {
            decodeRoutedBufsScratch.append(blob.buffer)
            decodeRoutedOffsetsScratch.append(Int(blob.offset))
        }
        let routedBufs = decodeRoutedBufsScratch
        let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // The phase-2 reduce already folded the shared branch (h1Buf
        // as its residual); the tail is a plain residual add.
        let gTail: (MTLCommandBuffer) throws -> Void = { [self] cb in
            try encodeResidualExitDecode(commandBuffer: cb,
                                         hidden: hidden, delta: h2Buf,
                                         sublayer: .mlp, layer: L)
        }
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let ioToken = eventLoad?.storage.completionToken
        let ioStatus = ioToken.map { ($0.status, $0.statusOffset) }
        if let token = ioToken {
            routedCB.encodeWaitForEvent(token.event, value: token.value)
        }
        if let stagingTransfer = eventLoad?.storage.metalStagingTransfer {
            // The compute command references cache slots only after it has
            // waited for the MTLIO staging event. This is deliberately a GPU
            // blit, not a CPU memcpy or a completion-handler submission.
            try stagingTransfer.encodeCopy(commandBuffer: routedCB)
        }
        let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
            ? phase1HitSplitArgBuf
            : nil
        let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
            routedBlobs: routedBufs,
            topK: topK,
            routedBufferOffsets: decodeRoutedOffsetsScratch)
        if splitArgBuf != nil {
            totalHitFixupLayers &+= 1
            writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
            try encodeRoutedPhase1Subset(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count),
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        } else {
            try encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs,
                                       ioStatus: ioStatus?.0,
                                       ioStatusOffset: ioStatus?.1 ?? 0)
        }
        try moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                               routedArgBuffer: argBuf,
                                               routedBlobs: routedBufs,
                                               routedOffsets: routedOffsets,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: h1Buf,
                                               y: h2Buf,
                                               d: D,
                                               f: FmoE,
                                               topK: topK,
                                               ioStatus: ioStatus?.0,
                                               ioStatusOffset: ioStatus?.1 ?? 0)
        try gTail(routedCB)
        routedCB.commit()
        if missCount > 0, let completed = completedStorageNanos, completed > 0 {
            let submitted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if submitted >= completed {
                totalIOCompletionToFixupSubmitNanos &+= submitted - completed
            }
        }
        guard pendingRoutedCommand == nil else {
            // The pipeline drains the previous layer's routed CB before
            // queuing the next, so this is a logic error, not a user
            // condition — but it must fail the generation, not trap.
            throw ModelError.internalInconsistency(
                detail: "routed command-buffer pipeline not drained before queuing the next layer")
        }
        pendingRoutedCommand = PendingRoutedCommand(
            cb: routedCB,
            sharedCB: sharedCB,
            phase1HitCB: phase1HitCB,
            expertLease: expertLease,
            storageOperation: eventLoad,
            overlapCompletionClock: eventLoad == nil ? nil : overlapCompletionClock,
            expectedOverlapCompletions: expectedOverlapCompletions,
            kernelRole: splitArgBuf == nil
                ? "moe_phase1_2_routed"
                : "moe_phase1_miss_fixup_phase2",
            encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
        transferredExpertLease = true
        totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
        if ProcessInfo.processInfo.environment["NVMAI_LAYER_TRACE"] != nil,
           position < 3 || position % 16 == 0 {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let attnUs = (attnCB.gpuEndTime - attnCB.gpuStartTime) * 1_000_000
            let tailUs = (tailCB.gpuEndTime - tailCB.gpuStartTime) * 1_000_000
            print("NVMAI layer pos=\(position) L=\(L) "
                + "body_us=\((now - tBodyStart) / 1000) "
                + "wait_us=\(waitNanos / 1000) io_us=\(layerIo / 1000) "
                + "cb1_us=\((tWait - tCb1Start) / 1000) "
                + "cb2_us=\((now - tCb2Start) / 1000) "
                + "gpu_attn_us=\(Int(attnUs)) gpu_tail_us=\(Int(tailUs)) "
                + "gpu_routed_us=\(Int(prevRoutedUs))")
        }
    }
}
