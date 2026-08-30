import Foundation
import Metal

/// Chunked prefill: the chunk executor, both attention encoders, the KV cache writes, the routed-MoE tile stage, and the final head.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
extension RealForwardRunner {
    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        // The chunked path does not go through `produceToken`, so it needs
        // the sparse-attention gate of its own. The chunk's last query sees
        // the most keys and decides the whole chunk.
        try requireQSADensePrefill(visibleKeys: startPosition + tokens.count)
        // The one-token-at-a-time prefill a hyper-connection family started
        // on, kept as the oracle the batched path is checked against.
        //
        // It runs the verified decode path per token, so it produces the KV
        // state and logits the batched path must reproduce. It is also
        // unusably slow -- every token pays a full pass over the routed
        // experts, where a chunk amortizes them -- so it is not the default.
        if cfg.hyperConnections.enabled && Self.sequentialHyperConnectionPrefill {
            var position = startPosition
            for (offset, token) in tokens.enumerated() {
                try Task.checkCancellation()
                try await produceToken(token: token,
                                       position: position,
                                       into: logits,
                                       emitHead: offset == tokens.count - 1,
                                       outputMode: outputMode)
                position += 1
                onProgress(offset + 1)
            }
            return PrefillResult(newPosition: position, seed: .logitsWritten)
        }
        // Intended to release the slot-cache wiring for prefill, which streams
        // experts in bulk and, on the ANE path, has to leave Core ML room for
        // its arenas.
        //
        // In practice this is a no-op and has always been: the cache is not
        // wired until the first decode token, so `slotsPinned` is already
        // false when prefill asks. Measured with `NVMAI_WIRE_TRACE=1` over a
        // full ANE-prefill request: 40 `mlock` calls at the handover, zero
        // `munlock` calls anywhere. The shipped behaviour is "wire once, at
        // the handover", not the release/re-apply cycle `703f35a`'s message
        // describes.
        //
        // Kept because it is correct for any future path that does wire
        // earlier, and because removing it would silently change that path's
        // behaviour. It is not load-bearing today.
        //
        // Skipped entirely under NVMAI_KEEP_WIRED: pinning at allocation and
        // then releasing here is self-defeating, and cost me one wrong
        // conclusion already.
        if !Self.keepExpertCacheWired {
            model.setExpertCachePinned(false)
        }
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        do {
            for (spanIndex, span) in spans.enumerated() {
                try Task.checkCancellation()
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                try await executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: config,
                    writeFinalHead: spanIndex == spans.count - 1)
                try Task.checkCancellation()
                onProgress(span.completedCount)
            }
        } catch {
            // Any failure — cancellation, a GPU command-buffer error, an I/O
            // error mid-routed-fetch — may have written partial KV rows and
            // left the chunk state dirty. Reset so the next request does not
            // trip `chunkedRunnerDirty` on a stale in-flight chunk.
            reset()
            throw error
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    /// lint:allow-long the orchestrator for one prefill chunk: scratch setup,
    /// the per-layer dispatch, and the head. Each stage it calls is its own
    /// method; what remains is the sequence, and inlining less of it would
    /// only hide the order the stages must run in.
    func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     preparedHidden: MTLBuffer? = nil,
                                     snapshotGDNAfterFirstToken: Bool = false,
                                     useTwoRowProjection: Bool = false,
                                     pairRoutedMoE: Bool = false) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires a KV cache")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        // KV grows on demand rather than reserving maxContext, so make room for
        // this chunk before anything writes into it.
        try kv?.reserve(tokens: startPosition + tokens.count)
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        guard !snapshotGDNAfterFirstToken || tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported(
                "Gated-DeltaNet speculative checkpoint requires two rows")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }


        let layerViews = try makeLayerPrefillViews()

        // Reused UInt32 token-ID buffer, sized to the largest chunk seen so
        // far and grown on demand (R23); the prefill hot path never allocates
        // a Metal buffer per chunk.
        let tokenBytes = tokens.count * MemoryLayout<UInt32>.stride
        let tokenBuffer: MTLBuffer
        if let existing = prefillTokenBuffer, existing.length >= tokenBytes {
            tokenBuffer = existing
        } else {
            guard let made = ctx.device.makeBuffer(length: tokenBytes,
                                                   options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "prefill.tokenIDs"
            prefillTokenBuffer = made
            tokenBuffer = made
        }
        let tokenPtr = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, token) in tokens.enumerated() {
            tokenPtr[i] = UInt32(bitPattern: token)
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = try model.embedding()


        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)
        // The n-gram rows depend only on token ids, so the whole chunk's
        // gather runs before any layer needs it.
        try gatherPLERowsPrefill(tokens: tokens)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let preparedHidden {
            // The caller hands over `[t, D]` rows -- an MTP draft's fused
            // hidden. A hyper-connection stack starts every stream from that
            // same vector, so it lands in the narrow staging buffer and is
            // widened exactly the way an embedding would be.
            let target = hyperConnection == nil ? scratch.hidden : scratch.normed
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: preparedHidden,
                      sourceOffset: 0,
                      to: target,
                      destinationOffset: 0,
                      size: t * D * MemoryLayout<Float16>.stride)
            blit.endEncoding()
            if hyperConnection != nil {
                try elementwise!.encodeHCExpand(
                    commandBuffer: cb,
                    source: scratch.normed, destination: scratch.hidden,
                    dim: D, streamCount: residualStreamCount, tokens: t)
            }
        } else {
            // A hyper-connection stack starts every stream from the token
            // embedding, so the lookup lands in a one-stream staging buffer
            // and is widened from there. `normed` is free until the first
            // layer's read gate writes it.
            let embedTarget = hyperConnection == nil ? scratch.hidden : scratch.normed
            try prefillEmbed.encode(commandBuffer: cb,
                                table: emb.buffer,
                                tableOffset: Int(emb.offset),
                                scales: emb.buffer,
                                scalesOffset: Int(emb.scaleOffset),
                                biases: emb.buffer,
                                biasesOffset: Int(emb.biasOffset),
                                tokens: tokenBuffer,
                                out: embedTarget,
                                t: UInt32(t),
                                d: UInt32(D),
                                outScale: embedOutScale,
                                vocab: UInt32(cfg.vocabSize))
            if hyperConnection != nil {
                try elementwise!.encodeHCExpand(
                    commandBuffer: cb,
                    source: scratch.normed, destination: scratch.hidden,
                    dim: D, streamCount: residualStreamCount, tokens: t)
            }
        }

        // Track A: whether this chunk's full-attention layers run on the ANE.
        // The MTP verify (two-row projection / GDN snapshot), MTP adapter
        // chunks (preparedHidden), non-4096 chunk configs, and prompts beyond
        // the sidecar's history variants all stay on the GPU; continuity is
        // enforced inside eligibleChunk so a fallback mid-prompt sticks for
        // the rest of the request.
        let aneChunk: ANEPrefillAttention? = {
            guard let ane = anePrefill,
                  !snapshotGDNAfterFirstToken,
                  !useTwoRowProjection,
                  !pairRoutedMoE,
                  preparedHidden == nil,
                  ane.eligibleChunk(startPosition: startPosition,
                                    tokenCount: tokens.count,
                                    configChunkTokens: config.chunkTokens)
            else { return nil }
            return ane
        }()

        let prefillProfile = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASES"] != nil
        var prefillRouteNanos: UInt64 = 0
        var prefillTileNanos: UInt64 = 0
        var prefillTailNanos: UInt64 = 0
        var prefillActiveExperts: UInt64 = 0

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            let prefillLayerStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            if cfg.ple.layerIndices.contains(L) {
                try encodePLEPrefill(commandBuffer: cb,
                                     hidden: scratch.hidden,
                                     layer: L, tokens: t, eps: eps)
            }
            try encodeResidualEntryPrefill(commandBuffer: cb,
                                           hidden: scratch.hidden,
                                           norm: views.inputNorm,
                                           out: scratch.normed,
                                           sublayer: .attention, layer: L,
                                           tokens: t, eps: eps)
            // The indexer caches a key for every prefilled token, in or out
            // of the dense-exact window: decode crossing the boundary later
            // must not find holes behind it.
            let qsaSelection = try encodeQSAPrefill(
                cb: &cb, blockInput: scratch.normed,
                layer: L, startPosition: startPosition,
                tokens: t, eps: eps)
            if isLinear {
                try encodeLinearAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                    useTwoRowProjection: useTwoRowProjection)
            } else if let ane = aneChunk, ane.coveredLayers.contains(L) {
                try await runANEFullAttentionPrefill(
                    ane: ane, cb: &cb, layer: L, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    startPosition: startPosition, kvDim: kvDim)
            } else {
                try encodeFullAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D, startPosition: startPosition,
                    isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                    qDim: qDim, kvDim: kvDim, rmsEps: eps,
                    useTwoRowProjection: useTwoRowProjection,
                    keepMask: qsaSelection)
            }
            // Plain pre-norm residual block: hidden += attention branch,
            // then one post-attention norm feeds router, shared expert,
            // and routed phase 1 (routedX doubles as moeX).
            try encodeResidualExitPrefill(commandBuffer: cb,
                                          hidden: scratch.hidden,
                                          delta: scratch.h1,
                                          sublayer: .attention, layer: L,
                                          tokens: t)
            try encodeResidualEntryPrefill(commandBuffer: cb,
                                           hidden: scratch.hidden,
                                           norm: views.postAttention,
                                           out: scratch.routedX,
                                           sublayer: .mlp, layer: L,
                                           tokens: t, eps: eps)
            if pairRoutedMoE, t == 2 {
                try await encodeRoutedMoEVerifyPair(
                    cb: &cb, layer: L, views: views, scratch: scratch,
                    hiddenSize: D)
            } else {
                try await encodeRoutedMoEPrefill(
                    cb: &cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    layerStart: prefillLayerStart,
                    routeNanos: &prefillRouteNanos,
                    tileNanos: &prefillTileNanos,
                    tailNanos: &prefillTailNanos,
                    activeExperts: &prefillActiveExperts)
            }
        }

        if prefillProfile {
            let prefillTotal = prefillRouteNanos + prefillTileNanos + prefillTailNanos
            print("[prefill phases over \(t) tokens, \(prefillTotal / 1_000_000) ms total]")
            print("  route readback + GPU: \(String(format: "%.1f", Double(prefillRouteNanos) / 1e6)) ms")
            print("  expert fetch + tiles: \(String(format: "%.1f", Double(prefillTileNanos) / 1e6)) ms")
            print("  tail + residual:      \(String(format: "%.1f", Double(prefillTailNanos) / 1e6)) ms")
            let perLayer = Double(prefillActiveExperts) / Double(max(1, cfg.numLayers))
            print("  active experts/layer: \(String(format: "%.2f", perLayer))"
                + " (topK=\(cfg.topKExperts), max possible \(t * cfg.topKExperts))")
        }

        if writeFinalHead {
            try encodeFinalHead(logits: logits, scratch: scratch,
                                tokenCount: t, hiddenSize: D, rmsEps: eps,
                                outputMode: outputMode)
        }

        aneChunk?.finishChunk(startPosition: startPosition,
                              tokenCount: tokens.count)
        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    /// Per-layer tensor views resolved once before the chunk loop.
    struct LayerPrefillQKVViews {
        let inputNorm: TensorView
        let postAttention: TensorView
        let router: TensorView
        // Softmax-attention layers only (nil on linear-attention layers).
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet linear-attention layers only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?
    }

    func encodeAffineProjection(commandBuffer: MTLCommandBuffer,
                              family: PrefillProjectionFamily,
                              weights: TensorView,
                              x: MTLBuffer,
                              y: MTLBuffer,
                              rows: Int,
                              columns: Int,
                              tokenCount: Int,
                              xStrideElements: Int,
                              yStrideElements: Int,
                              useTwoRowProjection: Bool) throws {
        if tokenCount >= 32,
           family == .q || family == .kv || family == .o,
           let candidate = prefillMPPAffineInt4 {
            let path = try candidate.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                m: tokenCount,
                n: rows,
                k: columns)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        if useTwoRowProjection && tokenCount == 2
            && xStrideElements == columns && yStrideElements == rows {
            if model.attentionWeightBits == 4 {
                try int4.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            } else {
                try affine!.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            }
            return
        }
        if PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount) == .qmm {
            try prefillQMM.encode(commandBuffer: commandBuffer,
                              weights: weights.buffer,
                              weightsOffset: Int(weights.offset),
                              scales: weights.buffer,
                              scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer,
                              biasesOffset: Int(weights.biasOffset),
                              x: x,
                              y: y,
                              t: tokenCount,
                              n: rows,
                              k: columns)
            return
        }
        for row in 0..<tokenCount {
            try encodePrimaryGEMV(
                commandBuffer: commandBuffer,
                projection: weights,
                x: x,
                xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                y: y,
                yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                m: UInt32(rows),
                n: UInt32(columns))
        }
    }

    func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                       source: MTLBuffer,
                       destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                       sourceTokenOffset: Int,
                       tokenCount: Int,
                       bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: source,
                  sourceOffset: sourceTokenOffset * bytesPerToken,
                  to: destination.buffer,
                  destinationOffset: destination.offset,
                  size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                              kv: KVCacheManager,
                              layer: Int,
                              startPosition: Int,
                              tokenCount: Int,
                              keySource: MTLBuffer,
                              valueSource: MTLBuffer,
                              bytesPerToken: Int) throws {
        if kv.precision.isQuantized {
            guard let kvQuantizer else {
                throw ModelError.internalInconsistency(
                    detail: "quantized KV cache has no quantizer")
            }
            let elements = bytesPerToken / MemoryLayout<Float16>.stride
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(layer: layer, start: startPosition,
                                             count: firstSpan),
                tokenCount: firstSpan,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(layer: layer, start: startPosition,
                                               count: firstSpan),
                tokenCount: firstSpan,
                elementCount: elements)
            guard firstSpan < tokenCount else { return }
            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let sourceOffset = firstSpan * bytesPerToken
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(layer: layer, start: secondStart,
                                             count: secondCount),
                tokenCount: secondCount,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(layer: layer, start: secondStart,
                                               count: secondCount),
                tokenCount: secondCount,
                elementCount: elements)
            return
        }
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
        let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keyFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
        let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keySecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueSecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
    }

    func encodeQuantizedKV(commandBuffer: MTLCommandBuffer,
                                   kv: KVCacheManager,
                                   layer: Int,
                                   position: Int,
                                   keySource: MTLBuffer,
                                   valueSource: MTLBuffer,
                                   elementCount: Int) throws {
        guard let kvQuantizer else {
            throw ModelError.internalInconsistency(
                detail: "quantized KV cache has no quantizer")
        }
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: keySource,
            sourceTokenStrideElements: elementCount,
            destination: kv.keyRangeView(layer: layer, start: position, count: 1),
            tokenCount: 1,
            elementCount: elementCount)
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: valueSource,
            sourceTokenStrideElements: elementCount,
            destination: kv.valueRangeView(layer: layer, start: position, count: 1),
            tokenCount: 1,
            elementCount: elementCount)
    }

    /// Gated-DeltaNet (linear attention) branch of one chunked-prefill layer.
    ///
    /// lint:allow-long one layer's linear-attention pipeline is a single
    /// ordered sequence -- in-projection, causal conv, QK norm, delta step,
    /// gated norm, out-projection -- sharing scratch buffers at every step.
    /// Splitting it would thread a dozen buffers through sub-functions to make
    /// a line count smaller while making the data flow harder to follow.
    func encodeLinearAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int,
        snapshotGDNAfterFirstToken: Bool, useTwoRowProjection: Bool
    ) throws {
        // Gated-DeltaNet linear attention over the chunk: batched
        // projections, causal conv (+ tail carry), delta-rule
        // recurrence, gated norm, out_proj. No KV writes, no
        // attention, no blit.
        guard let gdn, let gdnState else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        // `LayerPrefillQKVViews` fills the linear_attn slots exactly
        // when `layerIsLinear(L)`, so these are provably non-nil; the
        // guard turns a future arch/view regression into a thrown
        // error instead of a force-unwrap trap.
        guard let linQKV = views.linQKV,
              let linZ = views.linZ,
              let linA = views.linA,
              let linB = views.linB,
              let linConv = views.linConv,
              let linALog = views.linALog,
              let linDtBias = views.linDtBias,
              let linNorm = views.linNorm,
              let linOut = views.linOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) is missing a required linear_attn tensor view")
        }
        let la = cfg.linearAttention
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: linQKV,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: la.qkvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.qkvDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linZ,
                             x: scratch.normed,
                             y: scratch.gdnZ,
                             rows: la.valueDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.valueDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linA,
                             x: scratch.normed,
                             y: scratch.gdnA,
                             rows: la.numVHeads,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.numVHeads,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linB,
                             x: scratch.normed,
                             y: scratch.gdnB,
                             rows: la.numVHeads,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.numVHeads,
                             useTwoRowProjection: useTwoRowProjection)
        let convW = linConv
        let tail = gdnState.convTailBuffer(layer: L)
        try gdn.encodeConvPrefill(commandBuffer: cb,
                              tail: tail,
                              qkvRows: scratch.q,
                              convWeight: convW.buffer,
                              convWeightOffset: Int(convW.offset),
                              out: scratch.gdnConvOut,
                              rows: t)
        if snapshotGDNAfterFirstToken {
            try gdn.encodeConvTailCheckpoint(
                commandBuffer: cb,
                tail: tail,
                qkvRows: scratch.q,
                checkpoint: gdnState.speculativeConvTailBuffer(layer: L))
        }
        try gdn.encodeConvTailUpdate(commandBuffer: cb,
                                 tail: tail,
                                 qkvRows: scratch.q,
                                 rows: t)
        try gdn.encodeQKNorm(commandBuffer: cb,
                         convOut: scratch.gdnConvOut,
                         rows: t)
        let aLog = linALog
        let dtBias = linDtBias
        try gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                   convOut: scratch.gdnConvOut,
                                   aProj: scratch.gdnA,
                                   bProj: scratch.gdnB,
                                   aLog: aLog.buffer,
                                   aLogOffset: Int(aLog.offset),
                                   dtBias: dtBias.buffer,
                                   dtBiasOffset: Int(dtBias.offset),
                                   state: gdnState.stateBuffer(layer: L),
                                   checkpointState: snapshotGDNAfterFirstToken
                                    ? gdnState.speculativeStateBuffer(layer: L) : nil,
                                   y: scratch.gdnY,
                                   rows: t)
        let gatedNormW = linNorm
        try gdn.encodeGatedNorm(commandBuffer: cb,
                            y: scratch.gdnY,
                            z: scratch.gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: scratch.attentionOutput,
                            rows: t)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .o,
                             weights: linOut,
                             x: scratch.attentionOutput,
                             y: scratch.h1,
                             rows: D,
                             columns: la.valueDim,
                             tokenCount: t,
                             xStrideElements: la.valueDim,
                             yStrideElements: D,
                             useTwoRowProjection: useTwoRowProjection)
    }

    /// Softmax-attention branch of one chunked-prefill layer.
    ///
    /// lint:allow-long same shape as the linear branch: QKV projection, RoPE,
    /// KV-cache write and attention are one ordered pipeline over shared
    /// scratch, and the intermediate buffers have no meaning outside it.
    func encodeFullAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int, rmsEps eps: Float,
        useTwoRowProjection: Bool,
        keepMask: (buffer: MTLBuffer, stride: Int)? = nil
    ) throws {
        let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: views.q!,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: qProjRows,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: qProjRows,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.k!,
                             x: scratch.normed,
                             y: scratch.kStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.v!,
                             x: scratch.normed,
                             y: scratch.vStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim,
                             useTwoRowProjection: useTwoRowProjection)

        // The attention input Q: the packed q_proj output is split
        // into per-head query/gate halves for gated architectures.
        let attnQ: MTLBuffer
        if cfg.attnOutputGate {
            try elementwise!.encodeSplitQGate(commandBuffer: cb,
                                          packed: scratch.q,
                                          q: scratch.attnQ,
                                          gate: scratch.attnGate,
                                          heads: cfg.numHeads,
                                          dim: headDim,
                                          rows: t)
            attnQ = scratch.attnQ
        } else {
            attnQ = scratch.q
        }

        if cfg.ropeNeoxSubdim {
            let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
            try prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                qWeight: views.qNorm!.buffer,
                qWeightOffset: Int(views.qNorm!.offset),
                kWeight: views.kNorm!.buffer,
                kWeightOffset: Int(views.kNorm!.offset),
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: Float(cfg.fullRopeTheta),
                rotaryDim: rotaryDim,
                eps: eps)
        } else {
            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            try prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: attnQ,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm!.buffer,
                                       qWeightOffset: Int(views.qNorm!.offset),
                                       kWeight: views.kNorm!.buffer,
                                       kWeightOffset: Int(views.kNorm!.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)
        }

        if let kv {
            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try copyPrefillKVToCache(commandBuffer: cb,
                                     kv: kv,
                                     layer: L,
                                     startPosition: startPosition,
                                     tokenCount: t,
                                     keySource: scratch.kStage,
                                     valueSource: scratch.vStage,
                                     bytesPerToken: bytes / t)
        }
        let kvView = kv?.keyView(layer: L, validTokenCount: startPosition + t)
        let params = PrefillAttentionParams(
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                kvValidCount: UInt32(startPosition + t),
                slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                kvTokenStrideElements: UInt32(kvDim),
                qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim),
                scale: Float(cfg.attentionScale),
                kvBits: UInt32(kvView?.precision.rawValue ?? 16),
                kvTokenStrideBytes: UInt32(kvView?.stride ?? (kvDim * 2)),
                kvValueBytes: UInt32(kvView?.valueBytes ?? (kvDim * 2)),
                kvGroupSize: UInt32(kvView?.groupSize
                    ?? KVCacheManager.quantizationGroupSize))
        if let kv {
                let keyView = kv.keyView(layer: L, validTokenCount: startPosition + t)
                let valueView = kv.valueView(layer: L, validTokenCount: startPosition + t)
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                try prefillAttention.encodeCausal(commandBuffer: cb,
                                              q: attnQ,
                                              k: keyView.buffer,
                                              v: valueView.buffer,
                                              out: scratch.attentionOutput,
                                              params: params,
                                              kvRingCapacity: activeRingCapacity,
                                              keepMask: keepMask?.buffer,
                                              keepStride: keepMask?.stride ?? 0,
                                              path: keepMask == nil
                                                  ? prefillAttentionPath
                                                  // Only the tiled kernel
                                                  // honours a selection.
                                                  : .causalTiled)
        } else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill attention requires a KV cache")
        }
        if cfg.attnOutputGate {
            try elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                              out: scratch.attentionOutput,
                                              gate: scratch.attnGate,
                                              count: t * qDim)
        }
        try encodeAffineProjection(commandBuffer: cb,
                                 family: .o,
                                 weights: views.o!,
                                 x: scratch.attentionOutput,
                                 y: scratch.h1,
                                 rows: D,
                                 columns: qDim,
                                 tokenCount: t,
                                 xStrideElements: qDim,
                                 yStrideElements: D,
                                 useTwoRowProjection: useTwoRowProjection)
    }

    /// Resolve every layer's tensor views once, before the chunk loop.
    func makeLayerPrefillViews() throws -> [LayerPrefillQKVViews] {
        try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }
    }

    /// Final norm and lm_head for the last chunk, writing logits or a fused
    /// greedy token depending on the output mode.
    func encodeFinalHead(
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        rmsEps eps: Float,
        outputMode: PrefillOutputMode
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let finalCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let hc = hyperConnection {
            // The stack ends by collapsing the streams through the
            // model-level mixer, and only the last row feeds the head, so the
            // one-row decode gate serves here. `normed` is free once the last
            // layer has run.
            let rowBytes = D * residualStreamCount * MemoryLayout<Float16>.stride
            try hc.encodeRead(commandBuffer: finalCB,
                              streamsBuffer: scratch.hidden,
                              streamsOffset: (t - 1) * rowBytes,
                              hcNorm: finalNorm.buffer,
                              hcNormOffset: Int(finalNorm.offset),
                              down: gateWeightsPublic(try model.hcMixerDown()),
                              up: gateWeightsPublic(try model.hcMixerUp()),
                              blockInput: scratch.normed, eps: eps)
            try encodeHeadGEMV(commandBuffer: finalCB,
                               weights: lm.buffer, weightsOffset: Int(lm.offset),
                               scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                               biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                               x: scratch.normed, y: logits,
                               m: UInt32(cfg.vocabSize), n: UInt32(D))
            finalCB.commit()
            try waitForCompletion(finalCB)
            recordKernelGPU(role: "prefill_head", finalCB)
            if activationDumpDirectory != nil {
                dumpActivation("prefill_logits", logits, count: cfg.vocabSize,
                               position: 0)
            }
            return
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: finalCB,
                hidden: scratch.hidden,
                hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                normWeight: finalNorm.buffer,
                normOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            try prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                             hiddenBlock: scratch.hidden,
                                             row: t - 1,
                                             rowStrideElements: D,
                                             normWeight: finalNorm.buffer,
                                             normWeightOffset: Int(finalNorm.offset),
                                             weights: lm.buffer,
                                             weightsOffset: Int(lm.offset),
                                             scales: lm.buffer,
                                             scalesOffset: Int(lm.scaleOffset),
                                             biases: lm.buffer,
                                             biasesOffset: Int(lm.biasOffset),
                                             logits: logits,
                                             d: UInt32(D),
                                             vocab: UInt32(cfg.vocabSize),
                                             rmsEps: eps)
        }
        finalCB.commit()
        try waitForCompletion(finalCB)
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
    }

    /// Router, routed-expert fetch and the MoE tail for one prefill layer.
    ///
    /// lint:allow-long one layer's MoE stage is a single ordered pipeline:
    /// route readback, expert streaming, tiled phase-1/phase-2, then the
    /// residual tail. It rebinds the command buffer partway through (the
    /// resident buffer wraps between layers), so the stages share mutable
    /// encoding state and cannot be separated without threading it back out.
    func encodeRoutedMoEPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        layerStart prefillLayerStart: UInt64,
        routeNanos prefillRouteNanos: inout UInt64,
        tileNanos prefillTileNanos: inout UInt64,
        tailNanos prefillTailNanos: inout UInt64,
        activeExperts prefillActiveExperts: inout UInt64
    ) async throws {
        var prefillRouteEnd = prefillLayerStart
        var prefillTileEnd = prefillLayerStart
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
                    topK: UInt32(cfg.topKExperts),
                    hiddenStrideElements: UInt32(D))

                cb.commit()
                try waitForCompletion(cb)
                // Prefill had no occupancy instrumentation at all: these buffers
                // never reached recordKernelGPU, so NVMAI_KERNEL_STATS reported
                // only the decode tokens of a request and prefill looked idle.
                // Split by layer kind: the Track A go/no-go needs to know how
                // the attention-block time divides between full-attention
                // layers (whole block is ANE-expressible) and Gated-DeltaNet
                // layers (only the dense projections are; the recurrent scan
                // is not representable in a static Core ML graph).
                recordKernelGPU(role: cfg.layerIsLinear(L) ? "prefill_gdn_router"
                                    : "prefill_attn_router", cb)

                let routeCount = t * cfg.topKExperts
                let idPtr = scratch.routeIDs.contents()
                    .bindMemory(to: UInt32.self, capacity: routeCount)
                let weightPtr = scratch.routeWeights.contents()
                    .bindMemory(to: Float16.self, capacity: routeCount)
                // Reused per-chunk host scratch (R38): cleared in place so
                // the routed-tile planner never allocates per chunk.
                routeIDScratch.removeAll(keepingCapacity: true)
                routeWeightScratch.removeAll(keepingCapacity: true)
                routeIDScratch.reserveCapacity(routeCount)
                routeWeightScratch.reserveCapacity(routeCount)
                for i in 0..<routeCount {
                    routeIDScratch.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                    routeWeightScratch.append(weightPtr[i])
                }
                let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDScratch,
                                                               weights: routeWeightScratch,
                                                               queryCount: t,
                                                               topK: cfg.topKExperts)
                let schedulerConfig: PrefillRoutedTileSchedulerConfig
                let routeTileExpertCount: Int
                if let slotCount = model.routedExpertCacheSlotCount() {
                    guard let fitted = Self.prefillRoutedTileSchedulerConfig.fitting(
                        slotCount: slotCount) else {
                        throw PrefillError.chunkedUnsupported(
                            "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
                    }
                    schedulerConfig = fitted
                    routeTileExpertCount = fitted.tileExperts
                } else {
                    schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    routeTileExpertCount = schedulerConfig.tileExperts
                }
                let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                    pairs,
                    queryCount: t,
                    topK: cfg.topKExperts,
                    numExperts: cfg.numExperts,
                    tileExpertCount: routeTileExpertCount,
                    expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))
                prefillRouteEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                prefillRouteNanos &+= prefillRouteEnd - prefillLayerStart
                // One group per *distinct* expert this chunk touches. For a
                // 1-token chunk this is topK; for a speculative 2-token verify
                // it is the union of the two tokens' routes, which is what
                // decides whether the extra row rides along on weights the
                // first row already pulled in or pays for its own.
                prefillActiveExperts &+= UInt64(routes.groups.count)

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
                    // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
                    // per chunk row.
                    let gateView = sharedProj.scalarGate!
                    let halfBytes = MemoryLayout<Float16>.stride
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
                try waitForCompletion(sharedCB)
                recordKernelGPU(role: "prefill_shared_expert", sharedCB)

                let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                    device: ctx.device,
                    routes: routes)
                let routedOffsets = try model.routedExpertOffsets(layer: L)
                struct PendingPrefillTile {
                    let tileIndex: Int
                    let commandBuffer: MTLCommandBuffer
                    let fetch: PrefillStreamedTileFetchResult
                    let argumentBuffer: PrefillStreamedTileArgumentBuffer
                }
                var pendingTiles: [PendingPrefillTile] = []
                var tileLifetime = PrefillStreamedTileSlotLifetime()
                // `withExtendedLifetime` below takes a non-throwing closure,
                // so the wait error is captured here and rethrown after the
                // fetched blobs are released.
                var pendingTileError: Error?
                var tailError: Error?
                func drainOldestPendingTile() throws {
                    guard !pendingTiles.isEmpty else { return }
                    let pending = pendingTiles.removeFirst()
                    withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                        do {
                            try waitForCompletion(pending.commandBuffer)
                            recordKernelGPU(role: "prefill_routed_tile",
                                            pending.commandBuffer)
                        } catch {
                            // Rethrown after the fetched blobs are released.
                            pendingTileError = error
                        }
                    }
                    if let error = pendingTileError {
                        pendingTileError = nil
                        throw error
                    }
                    if !pending.fetch.plannedMissSlots.isEmpty {
                        try tileLifetime.complete(tileIndex: pending.tileIndex)
                    }
                }

                let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                for (tileIndex, tile) in routes.tiles.enumerated() {
                    let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                        forTile: tileIndex,
                        routes: routes)
                    var plannedFetch: RoutedExpertFetchPlan?
                    if !pendingTiles.isEmpty {
                        let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                        if !pendingAssignedSlots.isEmpty {
                            let pendingSlots = Set(pendingAssignedSlots)
                            let plan = try model.planRoutedExpertsIfPossible(
                                layer: L,
                                experts: expertIDs,
                                avoidingSlots: pendingSlots)
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: true,
                                    pendingDepth: pendingTiles.count,
                                    pendingAssignedSlots: pendingAssignedSlots,
                                    avoidingSlotPlanAvailable: plan != nil))
                            switch decision {
                            case .prefetchNext:
                                guard let plan else {
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler requested missing plan")
                                }
                                plannedFetch = plan
                            case .drainBeforeIssue:
                                try drainOldestPendingTile()
                            case .issueWithoutPending:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler ignored pending tile")
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: true,
                                    pendingDepth: pendingTiles.count,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .drainBeforeIssue:
                                try drainOldestPendingTile()
                            case .issueWithoutPending, .prefetchNext:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler failed to drain empty-slot pending tile")
                            }
                        }
                    } else {
                        let decision = routedTileScheduler.decide(
                            PrefillRoutedTileSchedulerInput(
                                hasPendingTile: false,
                                pendingAssignedSlots: [],
                                avoidingSlotPlanAvailable: false))
                        switch decision {
                        case .issueWithoutPending:
                            break
                        case .prefetchNext, .drainBeforeIssue:
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested pending action without pending tile")
                        }
                    }
                    let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                        model: model,
                        layer: L,
                        tileIndex: tileIndex,
                        routes: routes,
                        plannedFetch: plannedFetch,
                        avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                    try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                          pairStart: Int(tile.pairStart),
                                                          pairCount: Int(tile.pairCount))
                    if !fetch.plannedMissSlots.isEmpty {
                        try tileLifetime.begin(tileIndex: tileIndex,
                                               plannedSlots: fetch.plannedMissSlots)
                    }
                    let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                        device: ctx.device,
                        binding: fetch.binding)
                    let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                        pairStart: tile.pairStart,
                        pairCount: tile.pairCount,
                        d: UInt32(D),
                        routedIntermediate: UInt32(cfg.moeIntermediateSize),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D),
                        binding: fetch.binding,
                        offsets: routedOffsets)
                    guard let tileCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    _ = try prefillGroupedMoE.encodeStreamedBatched(
                        commandBuffer: tileCB,
                        hidden: scratch.routedX,
                        sortedPairs: metadata.sortedPairs,
                        routePartials: scratch.routePartials,
                        gateUpActScratch: scratch.routedGateUpActScratch,
                        downScratch: scratch.routedDownScratch,
                        argumentBuffer: argumentBuffer,
                        binding: fetch.binding,
                        params: streamedParams,
                        pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                    tileCB.commit()
                    pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                           commandBuffer: tileCB,
                                                           fetch: fetch,
                                                           argumentBuffer: argumentBuffer))
                    while pendingTiles.count > schedulerConfig.maxPendingDepth {
                        try drainOldestPendingTile()
                    }
                }
                while !pendingTiles.isEmpty {
                    try drainOldestPendingTile()
                }
                prefillTileEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                prefillTileNanos &+= prefillTileEnd - prefillRouteEnd
                guard let tailCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                  routePartials: scratch.routePartials,
                                                  routeWeights: scratch.routeWeights,
                                                  h2: scratch.h2,
                                                  queryCount: UInt32(t),
                                                  topK: UInt32(cfg.topKExperts),
                                                  d: UInt32(D))
                if hyperConnection != nil {
                    // The gated write injects one block output per stream, so
                    // the two MLP branches are summed first and written once.
                    // Adding them separately would apply the inject gate
                    // twice.
                    try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                   hidden: scratch.h2,
                                                   delta: scratch.h1,
                                                   count: t * D)
                    try encodeResidualExitPrefill(commandBuffer: tailCB,
                                                  hidden: scratch.hidden,
                                                  delta: scratch.h2,
                                                  sublayer: .mlp, layer: L,
                                                  tokens: t)
                } else {
                    // Plain pre-norm tail: hidden += gated shared branch
                    // + routed branch.
                    try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                   hidden: scratch.hidden,
                                                   delta: scratch.h1,
                                                   count: t * D)
                    try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                   hidden: scratch.hidden,
                                                   delta: scratch.h2,
                                                   count: t * D)
                }
                tailCB.commit()
                withExtendedLifetime(metadata) {
                    do {
                        try waitForCompletion(tailCB)
                        recordKernelGPU(role: "prefill_moe_reduce", tailCB)
                    } catch {
                        // Rethrown after `metadata` is released.
                        tailError = error
                    }
                }
                if let error = tailError {
                    tailError = nil
                    throw error
                }
                if L + 1 < cfg.numLayers {
                    guard let nextCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    cb = nextCB
                }
                prefillTailNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillTileEnd
    }
}
