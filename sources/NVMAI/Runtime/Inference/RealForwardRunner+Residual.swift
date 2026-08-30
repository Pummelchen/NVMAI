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
