import Foundation
import Metal

final class PrefillFinalRowHeadInt4 {
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV?
    private let affine: AffineQuantGEMV?
    private let normed: MTLBuffer
    private let maxD: Int

    init(context: MetalContext, maxD: Int = 2816, weightBits: Int = 4) throws {
        precondition(maxD > 0, "maxD must be positive")
        precondition([4, 6, 8].contains(weightBits))
        self.rms = try RMSNorm(context: context)
        self.int4 = weightBits == 4 ? try DequantInt4GEMV(context: context) : nil
        self.affine = weightBits == 4
            ? nil
            : try AffineQuantGEMV(context: context, weightBits: weightBits)
        self.maxD = maxD
        guard let normed = context.device.makeBuffer(length: maxD * MemoryLayout<Float16>.size,
                                                     options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.normed = normed
    }

    func encodeLogits(commandBuffer: MTLCommandBuffer,
                             hiddenBlock: MTLBuffer,
                             row: Int,
                             rowStrideElements: Int,
                             normWeight: MTLBuffer,
                             normWeightOffset: Int = 0,
                             weights: MTLBuffer,
                             weightsOffset: Int = 0,
                             scales: MTLBuffer,
                             scalesOffset: Int = 0,
                             biases: MTLBuffer,
                             biasesOffset: Int = 0,
                             logits: MTLBuffer,
                             logitsOffset: Int = 0,
                             d: UInt32,
                             vocab: UInt32,
                             rmsEps: Float) {
        precondition(row >= 0, "row must be non-negative")
        precondition(rowStrideElements >= Int(d), "row stride must cover d")
        precondition(Int(d) <= maxD, "d=\(d) exceeds maxD=\(maxD)")
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        let hiddenOffset = (row * rowStrideElements) * MemoryLayout<Float16>.size
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hiddenBlock,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normWeightOffset,
                        out: normed,
                        d: d,
                        eps: rmsEps)
        if let affine {
            affine.encode(commandBuffer: commandBuffer,
                          weights: weights,
                          weightsOffset: weightsOffset,
                          scales: scales,
                          scalesOffset: scalesOffset,
                          biases: biases,
                          biasesOffset: biasesOffset,
                          x: normed,
                          y: logits,
                          yOffset: logitsOffset,
                          m: vocab,
                          n: d)
        } else {
            int4!.encode(commandBuffer: commandBuffer,
                         weights: weights,
                         weightsOffset: weightsOffset,
                         scales: scales,
                         scalesOffset: scalesOffset,
                         biases: biases,
                         biasesOffset: biasesOffset,
                         x: normed,
                         y: logits,
                         yOffset: logitsOffset,
                         m: vocab,
                         n: d)
        }
    }
}
