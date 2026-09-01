import Metal

/// The GEMV for one quantization slot, chosen by that slot's width.
///
/// `DequantInt4GEMV` and `AffineQuantGEMV` present the same `encode`
/// signature, so a caller needs nothing but to be handed the right one. The
/// reason this type exists is that three kernels did not: `HyperConnection`,
/// `PLEBlock` and `QSAIndexer` each constructed `DequantInt4GEMV`
/// unconditionally, and an 8-bit install then handed them 8-bit weights that
/// they read as nibbles -- same buffers, half the bytes, no error. The model
/// loaded, answered " Paris", and degenerated.
///
/// Anything reading weights whose width comes from the manifest should hold
/// one of these rather than a concrete class.
struct SlotGEMV {
    private enum Quantized {
        case int4(DequantInt4GEMV)
        case affine(AffineQuantGEMV)
    }

    private let quantized: Quantized
    /// Built alongside the quantized kernel because promotion is per *tensor*,
    /// not per slot: a family kept at the checkpoint's bf16 sits inside an
    /// otherwise-8-bit slot, so both kernels have to be available at once and
    /// the choice made when the weights are known.
    private let bf16: BF16GEMV
    let weightBits: Int

    init(context: MetalContext, weightBits: Int) throws {
        precondition([4, 8].contains(weightBits),
                     "unsupported slot width \(weightBits)")
        self.quantized = weightBits == 4
            ? .int4(try DequantInt4GEMV(context: context))
            : .affine(try AffineQuantGEMV(context: context, weightBits: weightBits))
        self.bf16 = try BF16GEMV(context: context)
        self.weightBits = weightBits
    }

    /// `isBF16` comes from the tensor's own dtype in the resident index, not
    /// from the slot. Passing it wrongly is silent in the same way the
    /// INT4-only kernels were: the buffers are the right size for whichever
    /// reading you pick, and only the numbers come out wrong.
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32,
                isBF16: Bool = false) throws {
        if isBF16 {
            try bf16.encode(commandBuffer: commandBuffer,
                            weights: weights, weightsOffset: weightsOffset,
                            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                            m: m, n: n)
            return
        }
        switch quantized {
        case .int4(let gemv):
            try gemv.encode(commandBuffer: commandBuffer,
                            weights: weights, weightsOffset: weightsOffset,
                            scales: scales, scalesOffset: scalesOffset,
                            biases: biases, biasesOffset: biasesOffset,
                            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                            m: m, n: n)
        case .affine(let gemv):
            try gemv.encode(commandBuffer: commandBuffer,
                            weights: weights, weightsOffset: weightsOffset,
                            scales: scales, scalesOffset: scalesOffset,
                            biases: biases, biasesOffset: biasesOffset,
                            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                            m: m, n: n)
        }
    }
}
