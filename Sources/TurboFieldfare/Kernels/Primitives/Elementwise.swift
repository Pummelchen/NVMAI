import Foundation
import Metal

/// Small elementwise kernels used by the Qwen 3.6 layer graph: the
/// full-attention output gate, the shared-expert scalar gate, and the plain
/// pre-norm residual add (architectures without Gemma's fused sandwich tail).
final class Elementwise {
    private let sigmoidGateMulPSO: MTLComputePipelineState
    private let sigmoidScalarMulPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.sigmoidGateMulPSO = try context.pipeline("sigmoid_gate_mul_fp16")
        self.sigmoidScalarMulPSO = try context.pipeline("sigmoid_scalar_mul_fp16")
        self.residualAddPSO = try context.pipeline("residual_add_fp16")
    }

    /// out[i] *= sigmoid(gate[i])
    func encodeSigmoidGateMul(commandBuffer: MTLCommandBuffer,
                              out: MTLBuffer, outOffset: Int = 0,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
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
                                count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
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
                           count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: residualAddPSO, threads: count)
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
