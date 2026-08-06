import Metal

/// Correctness-first shared-expert implementation for affine formats that do
/// not have a dedicated fused kernel (currently Qwen's 6-bit checkpoint).
final class SharedExpertAffineQuant {
    private let gemv: AffineQuantGEMV
    private let activationPSO: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int,
         siluActivation: Bool) throws {
        gemv = try AffineQuantGEMV(context: context, weightBits: weightBits)
        activationPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                x: MTLBuffer, xOffset: Int,
                gate: SharedExpertProjection,
                up: SharedExpertProjection,
                down: SharedExpertProjection,
                y: MTLBuffer, yOffset: Int,
                scratchGate: MTLBuffer, scratchGateOffset: Int,
                scratchUp: MTLBuffer, scratchUpOffset: Int,
                scratchAct: MTLBuffer, scratchActOffset: Int) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch("incompatible projection shapes")
        }
        let bytes = Int(gate.rows) * MemoryLayout<Float16>.stride
        guard scratchGateOffset + bytes <= scratchGate.length,
              scratchUpOffset + bytes <= scratchUp.length,
              scratchActOffset + bytes <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall("need \(bytes) bytes per intermediate buffer")
        }
        func project(_ p: SharedExpertProjection, _ input: MTLBuffer,
                     _ inputOffset: Int, _ output: MTLBuffer,
                     _ outputOffset: Int) {
            gemv.encode(commandBuffer: cb,
                        weights: p.weights, weightsOffset: p.weightsOffset,
                        scales: p.scales, scalesOffset: p.scalesOffset,
                        biases: p.biases, biasesOffset: p.biasesOffset,
                        x: input, xOffset: inputOffset,
                        y: output, yOffset: outputOffset,
                        m: p.rows, n: p.cols)
        }
        project(gate, x, xOffset, scratchGate, scratchGateOffset)
        project(up, x, xOffset, scratchUp, scratchUpOffset)
        if let encoder = cb.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(activationPSO)
            encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
            encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
            encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
            var count = gate.rows
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, activationPSO.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            encoder.endEncoding()
        }
        project(down, scratchAct, scratchActOffset, y, yOffset)
    }
}
