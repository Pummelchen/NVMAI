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
enum SlotGEMV {
    case int4(DequantInt4GEMV)
    case affine(AffineQuantGEMV)

    init(context: MetalContext, weightBits: Int) throws {
        precondition([4, 8].contains(weightBits),
                     "unsupported slot width \(weightBits)")
        self = weightBits == 4
            ? .int4(try DequantInt4GEMV(context: context))
            : .affine(try AffineQuantGEMV(context: context, weightBits: weightBits))
    }

    var weightBits: Int {
        switch self {
        case .int4: return 4
        case .affine(let gemv): return gemv.weightBits
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32) throws {
        switch self {
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
