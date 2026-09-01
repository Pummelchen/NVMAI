import Foundation
import Metal

/// Qwen3.8-Flash-Next's Gated Residual ("hyper-connections").
///
/// The residual is `streams` parallel copies of `dim` (4 x 2560 in
/// production), carried that wide for the whole stack. Every sublayer reads a
/// single `dim`-wide vector out of it through a learned gate and writes its
/// output back into every stream through another:
///
///     normed = groupedRMSNorm(streams, hcNorm)         // per-stream slices
///     mix    = sigmoid(Wup @ silu(Wdown @ normed / S)) // S = stream count
///     input  = mean over streams of mix * normed       // -> dim
///     ... block runs on `input`, producing `blockOut` ...
///     inject = 2 * sigmoid(Winj @ normed / S)          // -> S
///     streams += blockOut (x) inject                   // outer product
///
/// Composition rather than new kernels: the three projections are ordinary
/// INT4 affine GEMVs, and only the two stream-axis steps needed writing. The
/// model-level mixer that collapses the streams before `lm_head` is the same
/// read path with no inject, so `encodeRead` serves both.
///
/// Scratch is owned per instance and sized once; `encodeRead` must complete
/// before `encodeWrite` reuses `normed`, which the caller gets for free by
/// encoding the block between them on the same command buffer.
final class HyperConnection {
    let dim: Int
    let streams: Int
    let lowRank: Int

    private let rms: RMSNorm
    private let gemv: SlotGEMV
    private let elementwise: Elementwise

    /// [streams * dim] normalized residual, shared by the read gate, the
    /// stream reduction, and the write gate.
    let normed: MTLBuffer
    /// [lowRank] down-projection, then its silu in place.
    private let lowRankScratch: MTLBuffer
    /// [streams * dim] read gate.
    private let mixScratch: MTLBuffer
    /// [streams] write gate.
    private let injectScratch: MTLBuffer

    /// The 1/S that appears inside both nonlinearities. It cannot be folded
    /// out afterwards: silu(x/S) != silu(x)/S.
    private var gateInputScale: Float { 1.0 / Float(streams) }

    /// Rows the scratch is sized for. One for decode; a prefill chunk needs
    /// the whole chunk resident, because the write gate consumes the
    /// `normed` its matching read produced and the block runs in between.
    let maxRows: Int

    init(context: MetalContext, dim: Int, streams: Int, lowRank: Int,
         maxRows: Int = 1, weightBits: Int = 4) throws {
        precondition(dim > 0 && streams > 0 && lowRank > 0 && maxRows > 0)
        self.maxRows = maxRows
        self.dim = dim
        self.streams = streams
        self.lowRank = lowRank
        self.rms = try RMSNorm(context: context)
        self.gemv = try SlotGEMV(context: context, weightBits: weightBits)
        self.elementwise = try Elementwise(context: context)
        let wide = dim * streams * maxRows * MemoryLayout<Float16>.stride
        guard let normed = context.device.makeBuffer(
                  length: wide, options: .storageModeShared),
              let mix = context.device.makeBuffer(
                  length: wide, options: .storageModeShared),
              let low = context.device.makeBuffer(
                  length: lowRank * maxRows * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let inject = context.device.makeBuffer(
                  length: max(streams * maxRows, 8) * MemoryLayout<Float16>.stride,
                  options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("HyperConnection scratch")
        }
        self.normed = normed
        self.mixScratch = mix
        self.lowRankScratch = low
        self.injectScratch = inject
    }

    /// Weight set for one gate. All three are INT4 affine with the usual
    /// scales/biases companions.
    struct Weights {
        let weights: MTLBuffer
        let weightsOffset: Int
        let scales: MTLBuffer
        let scalesOffset: Int
        let biases: MTLBuffer
        let biasesOffset: Int
        /// The tensor's own dtype, not the slot's. A promoted family is kept
        /// at the checkpoint's bf16 inside an otherwise-quantized slot, and
        /// scales/biases are then absent.
        let isBF16: Bool

        init(weights: MTLBuffer, weightsOffset: Int = 0,
             scales: MTLBuffer, scalesOffset: Int = 0,
             biases: MTLBuffer, biasesOffset: Int = 0,
             isBF16: Bool = false) {
            self.weights = weights
            self.weightsOffset = weightsOffset
            self.scales = scales
            self.scalesOffset = scalesOffset
            self.biases = biases
            self.biasesOffset = biasesOffset
            self.isBF16 = isBF16
        }
    }

    /// Normalize the residual and gate it down to the block's input vector.
    /// Leaves `normed` populated for the matching `encodeWrite`.
    /// A batched GEMM the caller supplies, so this type stays free of the
    /// runner's projection-dispatch policy: `(commandBuffer, weights, x, y,
    /// rows, columns, tokens)`.
    typealias BatchedProjection = (MTLCommandBuffer, Weights, MTLBuffer,
                                   MTLBuffer, Int, Int, Int) throws -> Void

    /// Residual -> block input for a whole prefill chunk.
    ///
    /// Identical arithmetic to `encodeRead`, with every stage taking `tokens`
    /// rows and the three projections going through `project` rather than a
    /// per-row GEMV. The residual is `[tokens, streams, dim]`, so a token's
    /// streams stay contiguous and the grouped norm indexes the same way it
    /// does for one token.
    func encodeReadRows(commandBuffer: MTLCommandBuffer,
                        streamsBuffer: MTLBuffer, streamsOffset: Int = 0,
                        hcNorm: MTLBuffer, hcNormOffset: Int = 0,
                        down: Weights, up: Weights,
                        blockInput: MTLBuffer, blockInputOffset: Int = 0,
                        tokens: Int, eps: Float,
                        project: BatchedProjection) throws {
        precondition(tokens <= maxRows,
                     "HyperConnection scratch holds \(maxRows) rows, asked for \(tokens)")
        let wide = dim * streams
        try rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                                   x: streamsBuffer, xOffset: streamsOffset,
                                   weight: hcNorm, weightOffset: hcNormOffset,
                                   out: normed,
                                   groupDim: UInt32(dim), numGroups: streams,
                                   eps: eps, tokens: tokens)
        try project(commandBuffer, down, normed, lowRankScratch,
                    lowRank, wide, tokens)
        try elementwise.encodeSilu(commandBuffer: commandBuffer,
                                   x: lowRankScratch, out: lowRankScratch,
                                   count: lowRank * tokens,
                                   inScale: gateInputScale)
        try project(commandBuffer, up, lowRankScratch, mixScratch,
                    wide, lowRank, tokens)
        try elementwise.encodeHCMixReduce(commandBuffer: commandBuffer,
                                          mix: mixScratch, normed: normed,
                                          out: blockInput,
                                          outOffset: blockInputOffset,
                                          dim: dim, streams: streams,
                                          tokens: tokens, inScale: 1)
    }

    /// Block output -> residual for a whole prefill chunk. Consumes the
    /// `normed` the matching `encodeReadRows` left behind, so the two must
    /// bracket exactly one block over the same rows.
    func encodeWriteRows(commandBuffer: MTLCommandBuffer,
                         streamsBuffer: MTLBuffer, streamsOffset: Int = 0,
                         inject: Weights,
                         blockOut: MTLBuffer, blockOutOffset: Int = 0,
                         tokens: Int,
                         project: BatchedProjection) throws {
        precondition(tokens <= maxRows,
                     "HyperConnection scratch holds \(maxRows) rows, asked for \(tokens)")
        try project(commandBuffer, inject, normed, injectScratch,
                    streams, dim * streams, tokens)
        try elementwise.encodeHCInject(commandBuffer: commandBuffer,
                                       streams: streamsBuffer,
                                       streamsOffset: streamsOffset,
                                       blockOut: blockOut,
                                       blockOutOffset: blockOutOffset,
                                       inject: injectScratch,
                                       dim: dim, streamCount: streams,
                                       tokens: tokens, inScale: gateInputScale)
    }

    func encodeRead(commandBuffer: MTLCommandBuffer,
                    streamsBuffer: MTLBuffer, streamsOffset: Int = 0,
                    hcNorm: MTLBuffer, hcNormOffset: Int = 0,
                    down: Weights, up: Weights,
                    blockInput: MTLBuffer, blockInputOffset: Int = 0,
                    eps: Float) throws {
        let wide = dim * streams
        try rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                                   x: streamsBuffer, xOffset: streamsOffset,
                                   weight: hcNorm, weightOffset: hcNormOffset,
                                   out: normed,
                                   groupDim: UInt32(dim), numGroups: streams,
                                   eps: eps)
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: down.weights, weightsOffset: down.weightsOffset,
                        scales: down.scales, scalesOffset: down.scalesOffset,
                        biases: down.biases, biasesOffset: down.biasesOffset,
                        x: normed, y: lowRankScratch,
                        m: UInt32(lowRank), n: UInt32(wide),
                        isBF16: down.isBF16)
        try elementwise.encodeSilu(commandBuffer: commandBuffer,
                                   x: lowRankScratch, out: lowRankScratch,
                                   count: lowRank, inScale: gateInputScale)
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: up.weights, weightsOffset: up.weightsOffset,
                        scales: up.scales, scalesOffset: up.scalesOffset,
                        biases: up.biases, biasesOffset: up.biasesOffset,
                        x: lowRankScratch, y: mixScratch,
                        m: UInt32(wide), n: UInt32(lowRank),
                        isBF16: up.isBF16)
        // The read gate is applied inside the reduce, which already reads
        // every element of the mix exactly once.
        try elementwise.encodeHCMixReduce(commandBuffer: commandBuffer,
                                          mix: mixScratch, normed: normed,
                                          out: blockInput,
                                          outOffset: blockInputOffset,
                                          dim: dim, streams: streams,
                                          inScale: 1)
    }

    /// Inject the block's output back into every stream. Consumes the `normed`
    /// left by the matching `encodeRead`, so the two must bracket one block.
    func encodeWrite(commandBuffer: MTLCommandBuffer,
                     streamsBuffer: MTLBuffer, streamsOffset: Int = 0,
                     inject: Weights,
                     blockOut: MTLBuffer, blockOutOffset: Int = 0) throws {
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: inject.weights, weightsOffset: inject.weightsOffset,
                        scales: inject.scales, scalesOffset: inject.scalesOffset,
                        biases: inject.biases, biasesOffset: inject.biasesOffset,
                        x: normed, y: injectScratch,
                        m: UInt32(streams), n: UInt32(dim * streams),
                        isBF16: inject.isBF16)
        // The write gate, 2 * sigmoid(...), opens to twice the read gate's
        // range so a stream can amplify a block rather than only attenuate it.
        // It is applied inside the inject rather than in its own dispatch.
        try elementwise.encodeHCInject(commandBuffer: commandBuffer,
                                       streams: streamsBuffer,
                                       streamsOffset: streamsOffset,
                                       blockOut: blockOut,
                                       blockOutOffset: blockOutOffset,
                                       inject: injectScratch,
                                       dim: dim, streamCount: streams,
                                       inScale: gateInputScale)
    }
}
