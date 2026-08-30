import Foundation
import Metal

/// Small elementwise kernels used by the Qwen 3.6 layer graph: the
/// full-attention output gate, the shared-expert scalar gate, and the plain
/// pre-norm residual add (architectures without a fused sandwich tail).
final class Elementwise {
    private let sigmoidGateMulPSO: MTLComputePipelineState
    private let sigmoidScalarMulPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState
    private let splitQGatePSO: MTLComputePipelineState
    private let concatRowsPSO: MTLComputePipelineState
    // Hyper-connection (Gated Residual) stream plumbing.
    private let hcMixReducePSO: MTLComputePipelineState
    private let hcInjectPSO: MTLComputePipelineState
    private let sigmoidPSO: MTLComputePipelineState
    private let siluPSO: MTLComputePipelineState
    // PLE (n-gram) block.
    private let pleScorePSO: MTLComputePipelineState
    private let pleGatePSO: MTLComputePipelineState
    private let pleBroadcastPSO: MTLComputePipelineState
    private let pleConvPSO: MTLComputePipelineState
    private let hcBroadcastPSO: MTLComputePipelineState
    private let hcExpandPSO: MTLComputePipelineState
    private let hcMeanPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.sigmoidGateMulPSO = try context.pipeline("sigmoid_gate_mul_fp16")
        self.sigmoidScalarMulPSO = try context.pipeline("sigmoid_scalar_mul_fp16")
        self.residualAddPSO = try context.pipeline("residual_add_fp16")
        self.splitQGatePSO = try context.pipeline("split_q_gate_fp16")
        self.concatRowsPSO = try context.pipeline("concat_rows_fp16")
        self.hcMixReducePSO = try context.pipeline("hc_stream_mix_reduce_fp16")
        self.hcInjectPSO = try context.pipeline("hc_stream_inject_fp16")
        self.sigmoidPSO = try context.pipeline("sigmoid_fp16")
        self.siluPSO = try context.pipeline("silu_fp16")
        self.pleScorePSO = try context.pipeline("ple_stream_score_fp16")
        self.pleGatePSO = try context.pipeline("ple_signed_sqrt_gate_fp16")
        self.pleBroadcastPSO = try context.pipeline("ple_broadcast_scale_fp16")
        self.pleConvPSO = try context.pipeline("ple_dilated_depthwise_conv_fp16")
        self.hcBroadcastPSO = try context.pipeline("hc_stream_broadcast_fp16")
        self.hcExpandPSO = try context.pipeline("hc_stream_expand_fp16")
        hcMeanPSO = try context.pipeline("hc_stream_mean_fp16")
    }

    private func encodeUnary(_ pso: MTLComputePipelineState,
                             commandBuffer: MTLCommandBuffer,
                             x: MTLBuffer, xOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             count: Int,
                             inScale: Float,
                             outScale: Float?) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var n = UInt32(count)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)
        var inS = inScale
        enc.setBytes(&inS, length: MemoryLayout<Float>.size, index: 3)
        if var outS = outScale {
            enc.setBytes(&outS, length: MemoryLayout<Float>.size, index: 4)
        }
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// out = sigmoid(x). Standalone, unlike `sigmoid_gate_mul` which folds the
    /// gate into an existing value.
    func encodeSigmoid(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       count: Int,
                       inScale: Float = 1, outScale: Float = 1) throws {
        try encodeUnary(sigmoidPSO, commandBuffer: commandBuffer,
                        x: x, xOffset: xOffset, out: out, outOffset: outOffset,
                        count: count, inScale: inScale, outScale: outScale)
    }

    /// out = silu(x) = x * sigmoid(x), where `silu_mul` fuses a second operand
    /// the hyper-connection low-rank path does not have.
    func encodeSilu(commandBuffer: MTLCommandBuffer,
                    x: MTLBuffer, xOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    count: Int, inScale: Float = 1) throws {
        try encodeUnary(siluPSO, commandBuffer: commandBuffer,
                        x: x, xOffset: xOffset, out: out, outOffset: outOffset,
                        count: count, inScale: inScale, outScale: nil)
    }

    /// The hyper-connection gated read: out[d] = mean over streams of
    /// mix[s,d] * normed[s,d]. Collapses the S-wide residual to the single
    /// D-wide vector the block consumes.
    func encodeHCMixReduce(commandBuffer: MTLCommandBuffer,
                           mix: MTLBuffer, mixOffset: Int = 0,
                           normed: MTLBuffer, normedOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           dim: Int, streams: Int, tokens: Int = 1,
                           inScale: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcMixReducePSO)
        enc.setBuffer(mix, offset: mixOffset, index: 0)
        enc.setBuffer(normed, offset: normedOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim), s = UInt32(streams), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        var gs = inScale
        enc.setBytes(&gs, length: MemoryLayout<Float>.size, index: 6)
        let w = min(hcMixReducePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: dim * tokens, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The hyper-connection gated write: streams[s,d] += blockOut[d] *
    /// inject[s], in place.
    func encodeHCInject(commandBuffer: MTLCommandBuffer,
                        streams: MTLBuffer, streamsOffset: Int = 0,
                        blockOut: MTLBuffer, blockOutOffset: Int = 0,
                        inject: MTLBuffer, injectOffset: Int = 0,
                        dim: Int, streamCount: Int, tokens: Int = 1,
                        inScale: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcInjectPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(blockOut, offset: blockOutOffset, index: 1)
        enc.setBuffer(inject, offset: injectOffset, index: 2)
        var d = UInt32(dim), s = UInt32(streamCount), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        var gs = inScale
        enc.setBytes(&gs, length: MemoryLayout<Float>.size, index: 6)
        let total = dim * streamCount * tokens
        let w = min(hcInjectPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-stream score: key . query / sqrt(D), one scalar per stream.
    func encodePLEStreamScore(commandBuffer: MTLCommandBuffer,
                              key: MTLBuffer, keyOffset: Int = 0,
                              query: MTLBuffer, queryOffset: Int = 0,
                              out: MTLBuffer, outOffset: Int = 0,
                              dim: Int, streams: Int, tokens: Int = 1) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleScorePSO)
        enc.setBuffer(key, offset: keyOffset, index: 0)
        enc.setBuffer(query, offset: queryOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim), s = UInt32(streams), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: streams * tokens, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: min(pleScorePSO.maxTotalThreadsPerThreadgroup, 32),
                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// gate = sigmoid(signed sqrt of the score), with a magnitude floor.
    func encodePLESignedSqrtGate(commandBuffer: MTLCommandBuffer,
                                 x: MTLBuffer, xOffset: Int = 0,
                                 out: MTLBuffer, outOffset: Int = 0,
                                 count: Int, floor: Float = 1e-6) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleGatePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var n = UInt32(count), f = floor
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&f, length: MemoryLayout<Float>.size, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: min(pleGatePSO.maxTotalThreadsPerThreadgroup, 32),
                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// out[s,d] = value[d] * gate[s].
    func encodePLEBroadcastScale(commandBuffer: MTLCommandBuffer,
                                 value: MTLBuffer, valueOffset: Int = 0,
                                 gate: MTLBuffer, gateOffset: Int = 0,
                                 out: MTLBuffer, outOffset: Int = 0,
                                 dim: Int, streams: Int, tokens: Int = 1) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleBroadcastPSO)
        enc.setBuffer(value, offset: valueOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim), s = UInt32(streams), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        let total = dim * streams * tokens
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: min(pleBroadcastPSO.maxTotalThreadsPerThreadgroup, 256),
                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Dilated depthwise causal convolution with silu, over `history + T` rows
    /// of carried state and chunk.
    func encodePLEDilatedConv(commandBuffer: MTLCommandBuffer,
                              xpad: MTLBuffer, xpadOffset: Int = 0,
                              weight: MTLBuffer, weightOffset: Int = 0,
                              out: MTLBuffer, outOffset: Int = 0,
                              channels: Int, tokens: Int,
                              kernelSize: Int, dilation: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleConvPSO)
        enc.setBuffer(xpad, offset: xpadOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var c = UInt32(channels), t = UInt32(tokens)
        var k = UInt32(kernelSize), dil = UInt32(dilation)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dil, length: MemoryLayout<UInt32>.size, index: 6)
        let total = channels * tokens
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: min(pleConvPSO.maxTotalThreadsPerThreadgroup, 256),
                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Collapse `[tokens, streams, dim]` to `[tokens, dim]` by a plain mean.
    func encodeHCMean(commandBuffer: MTLCommandBuffer,
                      streams: MTLBuffer, streamsOffset: Int = 0,
                      out: MTLBuffer, outOffset: Int = 0,
                      dim: Int, streamCount: Int, tokens: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcMeanPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var d = UInt32(dim), s = UInt32(streamCount), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        let w = min(hcMeanPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: dim * tokens, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Widen a `[tokens, dim]` block into the `[tokens, streams, dim]`
    /// residual. Distinct from the in-place broadcast: a chunk's embeddings
    /// are contiguous rows, and expanding them in place would overwrite rows
    /// that have not been read yet.
    func encodeHCExpand(commandBuffer: MTLCommandBuffer,
                        source: MTLBuffer, sourceOffset: Int = 0,
                        destination: MTLBuffer, destinationOffset: Int = 0,
                        dim: Int, streamCount: Int, tokens: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcExpandPSO)
        enc.setBuffer(source, offset: sourceOffset, index: 0)
        enc.setBuffer(destination, offset: destinationOffset, index: 1)
        var d = UInt32(dim), s = UInt32(streamCount), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        let total = dim * streamCount * tokens
        let w = min(hcExpandPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Replicate stream 0 across the remaining residual streams, in place.
    func encodeHCBroadcast(commandBuffer: MTLCommandBuffer,
                           streams: MTLBuffer, streamsOffset: Int = 0,
                           dim: Int, streamCount: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcBroadcastPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        var d = UInt32(dim), s = UInt32(streamCount)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 2)
        enc.dispatchThreads(MTLSize(width: dim * streamCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(
                                width: min(hcBroadcastPSO.maxTotalThreadsPerThreadgroup, 256),
                                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// packed [H, 2D] per-head [query ; gate] → q [H, D], gate [H, D].
    /// `rows` > 1 processes consecutive token rows (packed stride 2*H*D,
    /// output strides H*D).
    ///
    /// K5: all rows are dispatched from ONE encoder (a per-row dispatch loop
    /// with per-row buffer offsets) instead of creating one encoder per row.
    func encodeSplitQGate(commandBuffer: MTLCommandBuffer,
                          packed: MTLBuffer, packedOffset: Int = 0,
                          q: MTLBuffer, qOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          heads: Int, dim: Int, rows: Int = 1) throws {
        let rowElems = heads * dim
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(splitQGatePSO)
        var headCount = UInt32(heads)
        var headDim = UInt32(dim)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 4)
        for row in 0..<rows {
            encoder.setBuffer(packed, offset: packedOffset + row * 2 * rowElems * 2, index: 0)
            encoder.setBuffer(q, offset: qOffset + row * rowElems * 2, index: 1)
            encoder.setBuffer(gate, offset: gateOffset + row * rowElems * 2, index: 2)
            dispatch(encoder, pipeline: splitQGatePSO, threads: rowElems)
        }
        encoder.endEncoding()
    }

    /// out[i] *= sigmoid(gate[i])
    func encodeSigmoidGateMul(commandBuffer: MTLCommandBuffer,
                              out: MTLBuffer, outOffset: Int = 0,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(sigmoidGateMulPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidGateMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// y[i] *= sigmoid(gate[0])
    func encodeSigmoidScalarMul(commandBuffer: MTLCommandBuffer,
                                y: MTLBuffer, yOffset: Int = 0,
                                gate: MTLBuffer, gateOffset: Int = 0,
                                count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(sigmoidScalarMulPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidScalarMulPSO, threads: count)
        encoder.endEncoding()
    }

    /// hidden[i] += delta[i]
    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           delta: MTLBuffer, deltaOffset: Int = 0,
                           count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: residualAddPSO, threads: count)
        encoder.endEncoding()
    }

    func encodeConcatRows(commandBuffer: MTLCommandBuffer,
                          lhs: MTLBuffer,
                          rhs: MTLBuffer,
                          out: MTLBuffer,
                          rows: Int,
                          dim: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(concatRowsPSO)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var rowCount = UInt32(rows)
        var dimension = UInt32(dim)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(encoder, pipeline: concatRowsPSO, threads: rows * dim)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
