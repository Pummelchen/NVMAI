import Foundation
import Metal

enum MPPPrefillInt4QMMError: Error, CustomStringConvertible {
    case pipelineUnavailable(reason: String)
    case invalidArguments(String)

    public var description: String {
        switch self {
        case .pipelineUnavailable(let reason):
            return "MPP prefill QMM path unavailable: \(reason)"
        case .invalidArguments(let detail):
            return "MPP prefill QMM invalid arguments: \(detail)"
        }
    }
}

final class MPPPrefillInt4QMM {
    enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case unavailable
    }

    static let tileM = 64
    static let tileN = 32
    static let tileK = Quantization.groupSize

    private var pipeline: MTLComputePipelineState?
    /// K6: the compile failure reason, recorded once at init so an explicit
    /// MPP request can throw the real cause instead of silently degrading.
    private let unavailableReason: String

    init(context: MetalContext, weightBits: Int = 4) {
        precondition([4, 6, 8].contains(weightBits))
        do {
            let library = try Self.compileTensorOpsLibrary(device: context.device)
            let constants = MTLFunctionConstantValues()
            var bits = UInt32(weightBits)
            constants.setConstantValue(&bits, type: .uint, index: 78)
            let function = try library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16",
                constantValues: constants)
            self.pipeline = try context.device.makeComputePipelineState(function: function)
            self.unavailableReason = ""
        } catch {
            // Capability probe: this path is optional on non-Apple10 hardware,
            // so init stays non-throwing. Record the reason so a later
            // explicit request can throw it (K6).
            self.pipeline = nil
            self.unavailableReason = "\(error)"
        }
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    /// `required: true` makes an unavailable path a thrown error instead of a
    /// silent `.unavailable` fallback — use it when the caller explicitly
    /// requests the MPP path. Auto-selected callers keep `required: false`
    /// and check the returned `Path`.
    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int,
                       required: Bool = false) throws -> Path {
        guard m > 0,
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride) else {
            if required {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "m=\(m) n=\(n) k=\(k) offsets \(weightsOffset)/\(scalesOffset)/\(biasesOffset)/\(xOffset)/\(yOffset)")
            }
            return .unavailable
        }
        guard let pipeline else {
            if required {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(
                    reason: unavailableReason.isEmpty
                        ? "MPP pipeline failed to compile"
                        : unavailableReason)
            }
            return .unavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            if required { throw MetalError.commandEncoderFailed }
            return .unavailable
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return .affineThreadgroupF16
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }
}
