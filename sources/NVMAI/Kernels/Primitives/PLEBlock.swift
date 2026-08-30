import Foundation
import Metal

/// Qwen3.8-Flash-Next's per-layer n-gram embedding block (layer 1, 0-based).
///
/// The block runs *before* the attention read gate and rewrites the wide
/// residual in place. Given the rows this token's n-gram context hashes to
/// (see `PLEHash`) and their gathered embeddings (`NgramTableReader`), it
/// produces two terms and adds both:
///
///     key        = keyProj   @ emb                    // -> streams * dim
///     value      = valueProj @ emb                    // -> dim
///     key_w      = groupedRMSNorm(key,     normKey)
///     query      = groupedRMSNorm(streams, normQuery)
///     s[t]       = <key_w[t], query[t]> / sqrt(dim)    // one per stream
///     gate       = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
///     gated      = value (x) gate                      // -> streams * dim
///     normalized = groupedRMSNorm(gated, normConv)
///     conv       = dilated depthwise causal conv over time, K taps, dil = N
///     streams   += gated + silu(conv)
///
/// Two details here are silent if wrong, so they are stated rather than
/// implied: `gated` is added *unnormalized* while the convolution consumes
/// the normalized copy, and the gate's square root has a 1e-6 magnitude floor
/// that matters exactly where the derivative would blow up.
///
/// The convolution carries `(K - 1) * dilation` rows of history across
/// tokens. Decode advances one row at a time, so the state is double-buffered
/// and rotated with a blit rather than shifted in place -- an overlapping
/// same-buffer copy is not something Metal will do for us.
final class PLEBlock {
    /// One INT4 affine projection, laid out the way the repacker writes a
    /// quantized tensor: weights, scales and biases in one allocation.
    struct Projection {
        let weights: MTLBuffer
        let weightsOffset: Int
        let scales: MTLBuffer
        let scalesOffset: Int
        let biases: MTLBuffer
        let biasesOffset: Int
    }

    /// A bf16 vector parameter (the three norms and the conv taps).
    struct Vector {
        let buffer: MTLBuffer
        let offset: Int
    }

    struct Weights {
        let keyProj: Projection
        let valueProj: Projection
        let normKey: Vector
        let normQuery: Vector
        let normConv: Vector
        /// `[hcDim, kernelSize]`, channel-major: tap `k` of channel `c` is at
        /// `c * kernelSize + k`.
        let conv1d: Vector
    }

    let dim: Int
    let streams: Int
    let embedDim: Int
    let kernelSize: Int
    let dilation: Int
    /// Rows of convolution state carried between tokens.
    var history: Int { (kernelSize - 1) * dilation }
    private var hcDim: Int { dim * streams }

    private let rms: RMSNorm
    private let gemv: DequantInt4GEMV
    private let elementwise: Elementwise

    /// Host-written gather destination: `embedDim` fp16 values.
    let embedding: MTLBuffer
    // Not private: the parity harness reads these back to compare the block
    // stage by stage against the reference implementation.
    let keyBuf: MTLBuffer      // [hcDim]
    let valueBuf: MTLBuffer    // [dim]
    let keyNormed: MTLBuffer   // [hcDim]
    let queryNormed: MTLBuffer // [hcDim]
    let scoreBuf: MTLBuffer    // [streams]
    let gateBuf: MTLBuffer     // [streams]
    let gatedBuf: MTLBuffer    // [hcDim]
    let convOut: MTLBuffer     // [hcDim]
    /// `[history + 1, hcDim]`, ping-ponged so the row shift is a
    /// non-overlapping blit.
    private var xpad: [MTLBuffer]
    private var xpadIndex = 0

    init(context: MetalContext, dim: Int, streams: Int, embedDim: Int,
         kernelSize: Int, dilation: Int) throws {
        precondition(dim > 0 && streams > 0 && embedDim > 0)
        precondition(kernelSize > 1 && dilation > 0)
        self.dim = dim
        self.streams = streams
        self.embedDim = embedDim
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.rms = try RMSNorm(context: context)
        self.gemv = try DequantInt4GEMV(context: context)
        self.elementwise = try Elementwise(context: context)
        let wide = dim * streams
        let f16 = MemoryLayout<Float16>.stride
        func make(_ count: Int) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(
                      length: max(count, 1) * f16, options: .storageModeShared) else {
                throw MetalError.bufferAllocationFailed("PLE scratch")
            }
            return b
        }
        self.embedding = try make(embedDim)
        self.keyBuf = try make(wide)
        self.valueBuf = try make(dim)
        self.keyNormed = try make(wide)
        self.queryNormed = try make(wide)
        self.scoreBuf = try make(streams)
        self.gateBuf = try make(streams)
        self.gatedBuf = try make(wide)
        self.convOut = try make(wide)
        let padRows = (kernelSize - 1) * dilation + 1
        self.xpad = [try make(padRows * wide), try make(padRows * wide)]
        resetState()
    }

    /// Clears the carried convolution history. Call between completions: a
    /// state left over from a previous prompt would leak that prompt's
    /// n-grams into the first tokens of the next one.
    func resetState() {
        for buffer in xpad {
            memset(buffer.contents(), 0, buffer.length)
        }
        xpadIndex = 0
    }

    /// Encodes the block for a single token, rewriting `streams` in place.
    ///
    /// `embedding` must already hold this token's gathered rows. The state
    /// rotation is encoded on the same command buffer, after the read, so the
    /// caller only has to keep tokens in order.
    func encodeDecode(commandBuffer: MTLCommandBuffer,
                      streamsBuffer: MTLBuffer,
                      weights: Weights,
                      eps: Float) throws {
        let wide = hcDim
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: weights.keyProj.weights,
                        weightsOffset: weights.keyProj.weightsOffset,
                        scales: weights.keyProj.scales,
                        scalesOffset: weights.keyProj.scalesOffset,
                        biases: weights.keyProj.biases,
                        biasesOffset: weights.keyProj.biasesOffset,
                        x: embedding, y: keyBuf,
                        m: UInt32(wide), n: UInt32(embedDim))
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: weights.valueProj.weights,
                        weightsOffset: weights.valueProj.weightsOffset,
                        scales: weights.valueProj.scales,
                        scalesOffset: weights.valueProj.scalesOffset,
                        biases: weights.valueProj.biases,
                        biasesOffset: weights.valueProj.biasesOffset,
                        x: embedding, y: valueBuf,
                        m: UInt32(dim), n: UInt32(embedDim))
        try rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                                   x: keyBuf,
                                   weight: weights.normKey.buffer,
                                   weightOffset: weights.normKey.offset,
                                   out: keyNormed,
                                   groupDim: UInt32(dim), numGroups: streams,
                                   eps: eps)
        try rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                                   x: streamsBuffer,
                                   weight: weights.normQuery.buffer,
                                   weightOffset: weights.normQuery.offset,
                                   out: queryNormed,
                                   groupDim: UInt32(dim), numGroups: streams,
                                   eps: eps)
        try elementwise.encodePLEStreamScore(commandBuffer: commandBuffer,
                                             key: keyNormed, query: queryNormed,
                                             out: scoreBuf,
                                             dim: dim, streams: streams)
        try elementwise.encodePLESignedSqrtGate(commandBuffer: commandBuffer,
                                                x: scoreBuf, out: gateBuf,
                                                count: streams)
        try elementwise.encodePLEBroadcastScale(commandBuffer: commandBuffer,
                                                value: valueBuf, gate: gateBuf,
                                                out: gatedBuf,
                                                dim: dim, streams: streams)
        // The convolution's newest row is the normalized copy of `gated`,
        // written straight into the last slot of the padded window.
        let rowBytes = wide * MemoryLayout<Float16>.stride
        let current = xpad[xpadIndex]
        try rms.encodeBF16WGrouped(commandBuffer: commandBuffer,
                                   x: gatedBuf,
                                   weight: weights.normConv.buffer,
                                   weightOffset: weights.normConv.offset,
                                   out: current, outOffset: history * rowBytes,
                                   groupDim: UInt32(dim), numGroups: streams,
                                   eps: eps)
        try elementwise.encodePLEDilatedConv(commandBuffer: commandBuffer,
                                             xpad: current,
                                             weight: weights.conv1d.buffer,
                                             weightOffset: weights.conv1d.offset,
                                             out: convOut,
                                             channels: wide, tokens: 1,
                                             kernelSize: kernelSize,
                                             dilation: dilation)
        // Both terms land on the residual: the gated value unnormalized, and
        // the convolution, whose kernel has already applied the silu.
        try elementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                          hidden: streamsBuffer,
                                          delta: gatedBuf, count: wide)
        try elementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                          hidden: streamsBuffer,
                                          delta: convOut, count: wide)
        // Advance the window by one row: rows 1...history of the buffer we
        // just used become rows 0..<history of the next one.
        let next = xpad[1 - xpadIndex]
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(from: current, sourceOffset: rowBytes,
                  to: next, destinationOffset: 0,
                  size: history * rowBytes)
        blit.endEncoding()
        xpadIndex = 1 - xpadIndex
    }
}
