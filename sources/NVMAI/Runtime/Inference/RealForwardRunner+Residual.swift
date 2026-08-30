import Foundation
import Metal

/// How a family moves data between the residual stream and a sublayer.
///
/// The Qwen3.5-MoE families use plain pre-norm: normalize the residual into a
/// block's input, add the block's output back. Every layer does this twice,
/// once around attention and once around the MLP.
///
/// Qwen3.8-Flash-Next replaces both halves. Its residual is four parallel
/// 2560-wide streams, read through a learned gate that collapses them to one
/// vector and written through another that injects the block's output into
/// every stream with its own weight (`HyperConnection`). The shape of the
/// layer loop is unchanged — still normalize-in, combine-out — so the
/// difference belongs at this seam rather than branching the loop.
///
/// These four call sites are the entire coupling. Keeping them in one file
/// means adding the new family's behaviour touches this file and not the
/// decode or prefill bodies, matching how `TensorSchema` isolates naming.
/// Which sublayer a residual entry/exit pair brackets. A hyper-connection
/// family has separate gate weights for each; a pre-norm family ignores it.
enum ResidualSublayer {
    case attention
    case mlp
}

extension RealForwardRunner {
    /// Residual streams this model carries. One for pre-norm families; the
    /// hyper-connection families carry `hc_count`.
    var residualStreamCount: Int {
        cfg.hyperConnections.enabled ? cfg.hyperConnections.count : 1
    }

    /// Width of the residual buffer in elements.
    var residualWidth: Int { cfg.hiddenSize * residualStreamCount }

    func gateWeightsPublic(_ view: TensorView) -> HyperConnection.Weights {
        gateWeights(view)
    }

    private func gateWeights(_ view: TensorView) -> HyperConnection.Weights {
        HyperConnection.Weights(weights: view.buffer,
                                weightsOffset: Int(view.offset),
                                scales: view.buffer,
                                scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer,
                                biasesOffset: Int(view.biasOffset))
    }

    /// Residual -> block input, for one decode token.
    func encodeResidualEntryDecode(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   norm: TensorView,
                                   out: MTLBuffer,
                                   sublayer: ResidualSublayer,
                                   layer: Int,
                                   eps: Float) throws {
        if let hc = hyperConnection {
            let down = sublayer == .attention
                ? try model.hcAttnMixDown(layer: layer)
                : try model.hcMlpMixDown(layer: layer)
            let up = sublayer == .attention
                ? try model.hcAttnMixUp(layer: layer)
                : try model.hcMlpMixUp(layer: layer)
            try hc.encodeRead(commandBuffer: commandBuffer,
                              streamsBuffer: hidden,
                              hcNorm: norm.buffer,
                              hcNormOffset: Int(norm.offset),
                              down: gateWeights(down), up: gateWeights(up),
                              blockInput: out, eps: eps)
            return
        }
        try rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: norm.buffer, weightOffset: Int(norm.offset),
                            out: out,
                            d: UInt32(cfg.hiddenSize), eps: eps)
    }

    /// Block output -> residual, for one decode token.
    ///
    /// Consumes the `normed` the matching entry left inside the
    /// `HyperConnection`, so the two must bracket exactly one block on the
    /// same command buffer.
    func encodeResidualExitDecode(commandBuffer: MTLCommandBuffer,
                                  hidden: MTLBuffer,
                                  delta: MTLBuffer,
                                  sublayer: ResidualSublayer,
                                  layer: Int) throws {
        if let hc = hyperConnection {
            let inject = sublayer == .attention
                ? try model.hcAttnInject(layer: layer)
                : try model.hcMlpInject(layer: layer)
            try hc.encodeWrite(commandBuffer: commandBuffer,
                               streamsBuffer: hidden,
                               inject: gateWeights(inject),
                               blockOut: delta)
            return
        }
        try elementwise!.encodeResidualAdd(commandBuffer: commandBuffer,
                                           hidden: hidden,
                                           delta: delta,
                                           count: cfg.hiddenSize)
    }

    /// The batched projection the hyper-connection gates hand their GEMMs to.
    /// Routing them through the runner's own prefill dispatch keeps one
    /// policy for which kernel serves which shape.
    var prefillGateProjection: HyperConnection.BatchedProjection {
        { [self] commandBuffer, weights, x, y, rows, columns, tokens in
            try prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.weights,
                                  weightsOffset: weights.weightsOffset,
                                  scales: weights.scales,
                                  scalesOffset: weights.scalesOffset,
                                  biases: weights.biases,
                                  biasesOffset: weights.biasesOffset,
                                  x: x, y: y,
                                  t: tokens, n: rows, k: columns)
        }
    }

    /// Residual -> block input, for a prefill chunk of `tokens` rows.
    func encodeResidualEntryPrefill(commandBuffer: MTLCommandBuffer,
                                    hidden: MTLBuffer,
                                    norm: TensorView,
                                    out: MTLBuffer,
                                    sublayer: ResidualSublayer,
                                    layer: Int,
                                    tokens: Int,
                                    eps: Float) throws {
        if let hc = hyperConnection {
            let down = sublayer == .attention
                ? try model.hcAttnMixDown(layer: layer)
                : try model.hcMlpMixDown(layer: layer)
            let up = sublayer == .attention
                ? try model.hcAttnMixUp(layer: layer)
                : try model.hcMlpMixUp(layer: layer)
            try hc.encodeReadRows(commandBuffer: commandBuffer,
                                  streamsBuffer: hidden,
                                  hcNorm: norm.buffer,
                                  hcNormOffset: Int(norm.offset),
                                  down: gateWeightsPublic(down),
                                  up: gateWeightsPublic(up),
                                  blockInput: out,
                                  tokens: tokens, eps: eps,
                                  project: prefillGateProjection)
            return
        }
        try prefillRMS.encodeBF16W(commandBuffer: commandBuffer,
                                   x: hidden,
                                   weight: norm.buffer,
                                   weightOffset: Int(norm.offset),
                                   out: out,
                                   t: UInt32(tokens),
                                   d: UInt32(cfg.hiddenSize),
                                   eps: eps)
    }

    /// Block output -> residual, for a prefill chunk of `tokens` rows.
    func encodeResidualExitPrefill(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   delta: MTLBuffer,
                                   sublayer: ResidualSublayer,
                                   layer: Int,
                                   tokens: Int) throws {
        if let hc = hyperConnection {
            let inject = sublayer == .attention
                ? try model.hcAttnInject(layer: layer)
                : try model.hcMlpInject(layer: layer)
            try hc.encodeWriteRows(commandBuffer: commandBuffer,
                                   streamsBuffer: hidden,
                                   inject: gateWeightsPublic(inject),
                                   blockOut: delta,
                                   tokens: tokens,
                                   project: prefillGateProjection)
            return
        }
        try elementwise!.encodeResidualAdd(commandBuffer: commandBuffer,
                                           hidden: hidden,
                                           delta: delta,
                                           count: tokens * cfg.hiddenSize)
    }
}

// MARK: - QSA sparse attention

extension RealForwardRunner {
    /// The indexer's weights for one full-attention layer.
    func indexerWeights(layer: Int) throws -> QSAIndexer.Weights {
        QSAIndexer.Weights(
            queryProjection: try model.indexerQProj(layer: layer),
            keyProjection: try model.indexerKProj(layer: layer),
            queryNorm: try model.indexerQNorm(layer: layer),
            keyNorm: try model.indexerKNorm(layer: layer))
    }

    /// Whether this layer needs the indexer to choose keys for this query.
    /// Below the dense-exact window every visible key is kept anyway, so the
    /// selection is skipped rather than computed and thrown away.
    func qsaSelectionNeeded(layer: Int, position: Int) -> Bool {
        guard qsaIndexer != nil, cfg.fullAttentionLayerMask[layer] == 1,
              let exactness = qsaExactness else { return false }
        return !exactness.isDenseExact(visibleKeys: position + 1)
    }

    /// Runs the residual entry and the indexer for one full-attention layer,
    /// returning the selection the attention should honour.
    ///
    /// This is synchronous, and deliberately so: turning block scores into a
    /// per-key selection reproduces an ordering (score descending, index
    /// ascending, with the cell budget able to cut a block in half) that is
    /// clear on the host and fiddly on the GPU. The sync costs one barrier on
    /// the twelve full-attention layers, against a decode step that is bound
    /// by expert I/O; doing the selection on the GPU is the optimization, and
    /// it should be made against this as the reference.
    func encodeQSAEntryAndSelect(passthrough: MTLCommandBuffer,
                                 hidden: MTLBuffer,
                                 norm: TensorView,
                                 out: MTLBuffer,
                                 layer: Int,
                                 position: Int,
                                 eps: Float) throws -> MTLBuffer? {
        guard let indexer = qsaIndexer else { return nil }
        let weights = try indexerWeights(layer: layer)
        let selecting = qsaSelectionNeeded(layer: layer, position: position)
        guard selecting else {
            // Inside the window nothing is selected, so there is nothing to
            // read back and no reason to break the pipeline: the entry and
            // the key append ride the layer's own command buffer.
            try encodeResidualEntryDecode(commandBuffer: passthrough,
                                          hidden: hidden, norm: norm, out: out,
                                          sublayer: .attention, layer: layer,
                                          eps: eps)
            try indexer.encodeAppendKey(commandBuffer: passthrough, hidden: out,
                                        weights: weights, layer: layer,
                                        position: position, eps: eps)
            return nil
        }
        guard try runSync({ cb in
            try encodeResidualEntryDecode(commandBuffer: cb, hidden: hidden,
                                          norm: norm, out: out,
                                          sublayer: .attention, layer: layer,
                                          eps: eps)
            // The key is cached at every position, in or out of the window:
            // crossing the boundary later must not find holes behind it.
            try indexer.encodeAppendKey(commandBuffer: cb, hidden: out,
                                        weights: weights, layer: layer,
                                        position: position, eps: eps)
            try indexer.encodeScores(commandBuffer: cb, hidden: out,
                                     weights: weights, layer: layer,
                                     position: position, eps: eps)
        }) != nil else {
            throw ModelError.residentBufferWrapFailed
        }
        let mask = indexer.selectKeys(visibleKeys: position + 1)
        if layer == Self.qsaSnapshotLayer {
            dumpQSASnapshot(layer: layer, visibleKeys: position + 1)
        }
        return mask
    }

    /// The same snapshot from the chunked path, taken from its last row.
    func dumpQSAChunkSnapshot(selection: (buffer: MTLBuffer, stride: Int),
                              lastVisible: Int, rows: Int) {
        guard let directory = activationDumpDirectory,
              let indexer = qsaIndexer else { return }
        let base = (rows - 1) * selection.stride
        let keep = selection.buffer.contents()
            .bindMemory(to: UInt8.self, capacity: base + lastVisible)
        let bytes = (0..<lastVisible).map { keep[base + $0] }
        try? Data(bytes).write(
            to: directory.appendingPathComponent("qsa_keep.bin"))
        let blocks = (lastVisible - 1) / indexer.compressRatio + 1
        let scores = indexer.debugSnapshot(
            cells: 1, blocks: indexer.scoredBlocksPerRow * rows).scores
        let row = Array(scores[(rows - 1) * indexer.scoredBlocksPerRow ..<
                               (rows - 1) * indexer.scoredBlocksPerRow + blocks])
        row.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(
                to: directory.appendingPathComponent("qsa_scores.f32"))
        }
        writeQSACaches(directory: directory, indexer: indexer,
                       visibleKeys: lastVisible, blocks: blocks)
    }

    /// Writes the indexer's scores and selection for the newest query, so the
    /// decode and prefill paths can be compared as numbers.
    ///
    /// Only the first full-attention layer is snapshotted, and that is the
    /// point: later layers see inputs that have already drifted, so a
    /// disagreement there says nothing about whether the two paths select the
    /// same way. At the first one they have the same input, and any
    /// difference is the selection's own.
    func dumpQSASnapshot(layer: Int, visibleKeys: Int) {
        guard let directory = activationDumpDirectory,
              let indexer = qsaIndexer else { return }
        let blocks = (visibleKeys - 1) / indexer.compressRatio + 1
        let snapshot = indexer.debugSnapshot(cells: visibleKeys, blocks: blocks)
        try? Data(snapshot.keep).write(
            to: directory.appendingPathComponent("qsa_keep.bin"))
        snapshot.scores.withUnsafeBufferPointer {
            try? Data(buffer: $0).write(
                to: directory.appendingPathComponent("qsa_scores.f32"))
        }
        writeQSACaches(directory: directory, indexer: indexer,
                       visibleKeys: visibleKeys, blocks: blocks)
    }

    private func writeQSACaches(directory: URL, indexer: QSAIndexer,
                                visibleKeys: Int, blocks: Int) {
        if let pooled = indexer.debugPooled(layer: Self.qsaSnapshotLayer,
                                            blocks: blocks) {
            pooled.withUnsafeBufferPointer {
                try? Data(buffer: $0).write(
                    to: directory.appendingPathComponent("qsa_pooled.f32"))
            }
        }
        if let raw = indexer.debugRawKeys(layer: Self.qsaSnapshotLayer,
                                          count: visibleKeys) {
            raw.withUnsafeBufferPointer {
                try? Data(buffer: $0).write(
                    to: directory.appendingPathComponent("qsa_raw.f32"))
            }
        }
    }
}

/// Fills the indexer's caches for a prefill chunk, so decode can cross the
/// dense-exact boundary later without finding holes behind it.
extension RealForwardRunner {
    func encodeQSAPrefill(cb: inout MTLCommandBuffer,
                          blockInput: MTLBuffer,
                          layer: Int, startPosition: Int, tokens: Int,
                          eps: Float) throws -> (buffer: MTLBuffer, stride: Int)? {
        guard let indexer = qsaIndexer,
              cfg.fullAttentionLayerMask[layer] == 1 else { return nil }
        let commandBuffer = cb
        let weights = try indexerWeights(layer: layer)
        let destination = try indexer.rawKeyDestination(
            layer: layer, startPosition: startPosition)
        let key = weights.keyProjection
        try prefillQMM.encode(commandBuffer: commandBuffer,
                              weights: key.buffer,
                              weightsOffset: Int(key.offset),
                              scales: key.buffer,
                              scalesOffset: Int(key.scaleOffset),
                              biases: key.buffer,
                              biasesOffset: Int(key.biasOffset),
                              x: blockInput,
                              y: destination.buffer,
                              yOffset: destination.offset,
                              t: tokens,
                              n: indexer.headDim,
                              k: Int(key.shape.1))
        try indexer.encodePoolPrefill(commandBuffer: commandBuffer,
                                      weights: weights, layer: layer,
                                      startPosition: startPosition,
                                      tokens: tokens, eps: eps)

        // Nothing to choose while the chunk's longest query still sees
        // everything, so no scoring pass and no barrier.
        guard let exactness = qsaExactness,
              !exactness.isDenseExact(visibleKeys: startPosition + tokens)
        else { return nil }

        try indexer.encodeScoresPrefill(
            commandBuffer: commandBuffer, blockInput: blockInput,
            weights: weights, layer: layer,
            startPosition: startPosition, tokens: tokens, eps: eps,
            project: { cb, view, x, y, rows, columns, count in
                try self.prefillQMM.encode(commandBuffer: cb,
                                           weights: view.buffer,
                                           weightsOffset: Int(view.offset),
                                           scales: view.buffer,
                                           scalesOffset: Int(view.scaleOffset),
                                           biases: view.buffer,
                                           biasesOffset: Int(view.biasOffset),
                                           x: x, y: y,
                                           t: count, n: rows, k: columns)
            })
        // The selection is a host computation over the scores, so the chunk's
        // command buffer has to land first. The same barrier the routed MoE
        // already takes for its route readback, one layer earlier.
        commandBuffer.commit()
        try waitForCompletion(commandBuffer)
        recordKernelGPU(role: "prefill_qsa_index", commandBuffer)
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
        let selection = indexer.selectKeysPrefill(startPosition: startPosition,
                                                  tokens: tokens)
        // The chunk's last row is the one a sequential run's final decode
        // step also produces, so it is the comparable one.
        if activationDumpDirectory != nil, layer == Self.qsaSnapshotLayer,
           let selection {
            dumpQSAChunkSnapshot(selection: selection,
                                 lastVisible: startPosition + tokens,
                                 rows: tokens)
        }
        return selection
    }
}

// MARK: - PLE n-gram block

extension RealForwardRunner {
    /// Reads this token's n-gram rows off storage into the block's input.
    ///
    /// A no-op for families without PLE. The gather is 16 rows of 320 bytes —
    /// 5 KiB — which is three orders of magnitude under one token's routed
    /// expert traffic, so it is done inline rather than scheduled.
    func gatherPLERows(token: Int32) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        pleContext.insert(token, at: 0)
        if pleContext.count > hash.ngramSize {
            pleContext.removeLast(pleContext.count - hash.ngramSize)
        }
        let rows = hash.rows(context: pleContext)
        try table.gather(rows: rows, into: ple.embedding.contents())
    }

    /// Rewinds the n-gram block's carried state after a speculative pass whose
    /// rows were not all accepted.
    func rewindPLE(acceptedRows: Int, passRows: Int) {
        guard pleBlock != nil, acceptedRows < passRows else { return }
        pleBlock?.rewindWindow(acceptedRows: acceptedRows, passRows: passRows)
        // The hashed context must drop the same rows, or every later token's
        // n-gram ids are computed one position out.
        let discard = min(passRows - acceptedRows, pleContext.count)
        pleContext.removeFirst(discard)
    }

    /// Clears the state that spans a completion: the convolution history and
    /// the token context it hashes. Leaving either in place would let one
    /// prompt's trailing n-grams open the next one.
    func resetPLEState() {
        pleContext.removeAll(keepingCapacity: true)
        pleBlock?.resetState()
    }

    /// Encodes the n-gram block on the layers that carry one.
    func encodePLEDecode(commandBuffer: MTLCommandBuffer,
                         layer: Int, position: Int, eps: Float) throws {
        guard let ple = pleBlock,
              cfg.ple.layerIndices.contains(layer) else { return }
        if activationDumpActive(position: position) {
            dumpActivation("L\(layer)_ple_pre", hidden, count: residualWidth, position: position)
        }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        try ple.encodeDecode(commandBuffer: commandBuffer,
                             streamsBuffer: hidden,
                             weights: weights, eps: eps)
    }

    /// Gathers a whole prefill chunk's n-gram rows.
    ///
    /// Unlike decode, the context for row `i` is the chunk's own tokens plus
    /// whatever preceded the chunk, so this walks the chunk in order and
    /// leaves `pleContext` positioned for the next one.
    func gatherPLERowsPrefill(tokens: ArraySlice<Int32>) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        let rowBytes = hash.headCount * table.rowBytes
        for (index, token) in tokens.enumerated() {
            pleContext.insert(token, at: 0)
            if pleContext.count > hash.ngramSize {
                pleContext.removeLast(pleContext.count - hash.ngramSize)
            }
            try table.gather(rows: hash.rows(context: pleContext),
                             into: ple.embedding.contents()
                                .advanced(by: index * rowBytes))
        }
    }

    /// Encodes the n-gram block over a whole prefill chunk.
    func encodePLEPrefill(commandBuffer: MTLCommandBuffer,
                          hidden: MTLBuffer,
                          layer: Int, tokens: Int, eps: Float) throws {
        guard let ple = pleBlock,
              cfg.ple.layerIndices.contains(layer) else { return }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        try ple.encodeRows(commandBuffer: commandBuffer,
                           streamsBuffer: hidden, weights: weights,
                           tokens: tokens, eps: eps) {
            [self] cb, proj, x, y, rows, columns, count in
            try prefillQMM.encode(commandBuffer: cb,
                                  weights: proj.weights,
                                  weightsOffset: proj.weightsOffset,
                                  scales: proj.scales,
                                  scalesOffset: proj.scalesOffset,
                                  biases: proj.biases,
                                  biasesOffset: proj.biasesOffset,
                                  x: x, y: y,
                                  t: count, n: rows, k: columns)
        }
    }

    private func projection(_ view: TensorView) -> PLEBlock.Projection {
        PLEBlock.Projection(weights: view.buffer,
                            weightsOffset: Int(view.offset),
                            scales: view.buffer,
                            scalesOffset: Int(view.scaleOffset),
                            biases: view.buffer,
                            biasesOffset: Int(view.biasOffset))
    }

    private func vector(_ view: TensorView) -> PLEBlock.Vector {
        PLEBlock.Vector(buffer: view.buffer, offset: Int(view.offset))
    }
}
