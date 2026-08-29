import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    /// Experts one routed dispatch serves — the architecture's `topKExperts`,
    /// supplied at init. It sizes the routed argument buffer and every
    /// per-dispatch validation; nothing may assume the literal 8 (the
    /// Qwen3.5-MoE value) because Qwen3.8-Flash-Next routes top-10. The Metal
    /// side is currently compiled for k = 8 (`moe_phase2_reduce_k8` and the
    /// argument-buffer encoder length); init refuses anything else so a new
    /// k arrives as explicit kernel work, never as silent misexecution.
    let maxStreamedExperts: Int

    private let realDecodeD: UInt32
    private let realDecodeF: UInt32
    private static let realDecodeTopK: UInt32 = 8
    private let realDecodeNumExperts: UInt32

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    /// Used when top-k is not 8. The k8 kernel stays the golden path.
    private let routerSelectKNPSO: MTLComputePipelineState
    private let residencyClassifyPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    /// Used when top-k is not 8; the k8 kernels stay the golden path.
    private let phase2ReduceKNPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer
    private let alwaysReadyIOStatus: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (the specialized
    /// defaults 2816/704/128 predate Qwen-only support; Qwen 3.6 passes
    /// 2048/512/256). `siluActivation` selects the expert FFN activation
    /// (false = gelu_pytorch_tanh, true = silu).
    init(context: MetalContext,
         siluActivation: Bool = false,
         routedWeightBits: Int = 4,
         routerWeightBits: Int = 8,
         eventGatedIO: Bool = false,
         specializedD: UInt32 = 2816,
         specializedF: UInt32 = 704,
         specializedNumExperts: UInt32 = 128,
         topKExperts: Int = 8) throws {
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeNumExperts = specializedNumExperts
        // 16 is the argument buffer's blob-array extent (kMaxStreamedExperts
        // in moe.metal). Router select and phase-2 reduce both have k != 8
        // variants; see docs/qwen38-flash-next-port.md.
        precondition((1...16).contains(topKExperts),
                     "top-\(topKExperts) exceeds the routed argument buffer's "
                         + "expert slots (16)")
        self.maxStreamedExperts = topKExperts
        precondition([4, 8].contains(routedWeightBits))
        precondition([4, 8].contains(routerWeightBits))
        let activationConstants: [MetalFunctionConstant] = siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        let weightConstants = routedWeightBits == 4 ? [] : [
            MetalFunctionConstant(index: 5, value: .uint32(UInt32(routedWeightBits)))
        ]
        let ioConstants = [MetalFunctionConstant(index: 6, value: .bool(eventGatedIO))]
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(Self.realDecodeTopK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ] + activationConstants + weightConstants + ioConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(Self.realDecodeTopK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
            MetalFunctionConstant(index: 44, value: .uint32(UInt32(routerWeightBits))),
        ]
        let routerName = "router_gemv_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [MetalFunctionConstant(index: 44,
                                              value: .uint32(UInt32(routerWeightBits)))],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: routerConstants)
        self.routerSelectKNPSO = try context.pipeline("router_topk_select_kn")
        self.residencyClassifyPSO = try context.pipeline("moe_classify_expert_residency")
        let phase1Name = routedWeightBits == 4
            ? "moe_phase1_gate_up_act_u16load" : "moe_affine_phase1_gate_up_act"
        let phase1SubsetName = routedWeightBits == 4
            ? "moe_phase1_gate_up_act_subset_u16load" : "moe_affine_phase1_gate_up_act_subset"
        let phase2Name = routedWeightBits == 4
            ? "moe_phase2_down_reduce_k8" : "moe_affine_phase2_down_reduce_k8"
        self.phase1U16PSO = try context.pipeline(
            phase1Name, constants: activationConstants + weightConstants + ioConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            phase1Name,
            constants: moeConstants)
        self.phase1SubsetU16PSO = try context.pipeline(
            phase1SubsetName, constants: activationConstants + weightConstants + ioConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            phase1SubsetName,
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline(
            phase2Name, constants: weightConstants + ioConstants)
        let phase2KNName = routedWeightBits == 4
            ? "moe_phase2_down_reduce_kn" : "moe_affine_phase2_down_reduce_kn"
        self.phase2ReduceKNPSO = try context.pipeline(
            phase2KNName, constants: weightConstants + ioConstants)
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            phase2Name,
            constants: moeConstants)

        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let readyStatus = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(name: phase1Name) else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        readyStatus.contents().storeBytes(of: UInt32(1), as: UInt32.self)
        self.alwaysReadyIOStatus = readyStatus
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) throws {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        // 512 for Qwen3.8-Flash-Next. Expert ids are UInt32 end to end
        // (ExpertResidencyTable) and the kernels read num_experts
        // dynamically, so the old 256 was a conservative guard rather than a
        // width limit -- confirmed by reading both, not assumed.
        precondition(numExperts <= 512)
        precondition(topK == UInt32(maxStreamedExperts))
        // K16: `router_gemv_r4` multiplies every hidden element by
        // `effective_scale[idx]` and `router_topk_select_k8` multiplies every
        // weight by `per_expert_scale[expert]`. Qwen 3.6 has no router scale
        // tensors, so the runner synthesizes 1.0-filled buffers for both —
        // they must always be supplied with at least the addressed element
        // count, never nil/undersized, or the kernels read out of bounds.
        precondition(effectiveScale.length >= Int(d) * MemoryLayout<UInt16>.stride,
                     "encodeRouter: effectiveScale must cover [d] BF16 (runner synthesizes a 1.0 buffer for Qwen; the kernel always reads it)")
        precondition(perExpertScale.length >= Int(numExperts) * MemoryLayout<UInt16>.stride,
                     "encodeRouter: perExpertScale must cover [numExperts] BF16 (runner synthesizes a 1.0 buffer for Qwen; router_topk_select_k8 always dereferences it)")

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        encoder.setBuffer(routerLogits, offset: 0, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()

        guard let selector = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        selector.setComputePipelineState(
            maxStreamedExperts == 8
                ? (useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
                : routerSelectKNPSO)
        selector.setBuffer(routerLogits, offset: 0, index: 0)
        selector.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        selector.setBuffer(outIndices, offset: 0, index: 2)
        selector.setBuffer(outWeights, offset: 0, index: 3)
        selector.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        if maxStreamedExperts != 8 {
            var k = UInt32(maxStreamedExperts)
            selector.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 5)
        }
        selector.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        selector.endEncoding()
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                         topK: UInt32,
                                         routedBufferOffsets: [Int]? = nil) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs,
                                   routedBufferOffsets: routedBufferOffsets)
        return buffer
    }

    func encodeResidencyClassification(
        commandBuffer: MTLCommandBuffer,
        topKIndices: MTLBuffer,
        residencyTable: MTLBuffer,
        hitCount: MTLBuffer,
        hitPositions: MTLBuffer,
        missCount: MTLBuffer,
        missPositions: MTLBuffer,
        missExperts: MTLBuffer,
        resolvedSlots: MTLBuffer,
        resolvedGenerations: MTLBuffer,
        topK: UInt32,
        numExperts: UInt32
    ) throws {
        precondition(topK <= UInt32(maxStreamedExperts))
        var topKValue = topK
        var expertCount = numExperts
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residencyClassifyPSO)
        encoder.setBuffer(topKIndices, offset: 0, index: 0)
        encoder.setBuffer(residencyTable, offset: 0, index: 1)
        encoder.setBuffer(hitCount, offset: 0, index: 2)
        encoder.setBuffer(hitPositions, offset: 0, index: 3)
        encoder.setBuffer(missCount, offset: 0, index: 4)
        encoder.setBuffer(missPositions, offset: 0, index: 5)
        encoder.setBuffer(missExperts, offset: 0, index: 6)
        encoder.setBuffer(resolvedSlots, offset: 0, index: 7)
        encoder.setBuffer(resolvedGenerations, offset: 0, index: 8)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// An argument buffer with no views encoded yet, for callers that
    /// re-encode per use via `writeRoutedArgumentBuffer`.
    func makeEmptyRoutedArgumentBuffer(device: MTLDevice) -> MTLBuffer? {
        device.makeBuffer(length: routedArgEncoder.encodedLength,
                          options: .storageModeShared)
    }

    /// Re-encode the views of an argument buffer created by
    /// `makeRoutedArgumentBuffer`. The caller owns the hazard: the buffer must
    /// not be rewritten while a committed command still reads it.
    func writeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                   routedBlobs: [MTLBuffer],
                                   topK: UInt32,
                                   routedBufferOffsets: [Int]? = nil) {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs,
                                   routedBufferOffsets: routedBufferOffsets)
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                               topK: UInt32,
                                               routedBufferOffsets: [Int]? = nil) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs,
                                   routedBufferOffsets: routedBufferOffsets)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        xOffset: Int = 0,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(acts, offset: actsOffset, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 7)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x), so the
        // dispatch is (topK*f)/16 groups of 512 threads.
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 9)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x).
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        routingWeights: MTLBuffer,
        routingWeightsOffset: Int = 0,
        residual: MTLBuffer,
        residualOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            maxStreamedExperts == 8
                ? (useRealDecodeConstants(d: d, f: f)
                    ? phase2ReduceK8SpecializedPSO
                    : phase2ReduceK8PSO)
                : phase2ReduceKNPSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: actsOffset, index: 2)
        encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        encoder.setBuffer(residual, offset: residualOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 8)
        // One simdgroup per expert slot. The kn kernel has no sg >= k guard
        // precisely because the launch width says k, so this must stay in
        // step with it: 32 lanes x k.
        if maxStreamedExperts != 8 {
            var k = UInt32(maxStreamedExperts)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 9)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * maxStreamedExperts,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == UInt32(maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer],
                                            routedBufferOffsets: [Int]?) {
        precondition(routedBufferOffsets == nil
                     || routedBufferOffsets?.count == routedBlobs.count)
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(
                blob,
                offset: routedBufferOffsets?[index] ?? 0,
                index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF
    }
}
