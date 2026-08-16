import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Qwen 3.6 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<40:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = v_proj(a)
///     per-head q/k_norm (bf16w)
///     NeoX RoPE on Q + K (full-attention layers; linear layers use
///     Gated-DeltaNet recurrent state instead of K/V slots)
///     write K and V into the cache slots
///     attn = attention(scale, full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)  // sigmoid-gated by shared_expert_gate
///     xr = rmsnorm_no_scale(h)
///     idx, w = router_topk(xr, effective_scale[L], per_expert_scale[L])
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=h1, routedBlobs=fetch(idx), w)
///     h = h + h2
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), lm_head^T)
///   // final softmax happens in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Qwen-only [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let affineEmbed: AffineQuantEmbeddingLookup?
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let affine: AffineQuantGEMV?
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let int8ScalarGate: DequantInt8GEMV?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    private let verificationHidden: MTLBuffer // [2, D] FP16 shared readback
    private let verificationLogits: MTLBuffer // [2, vocab] FP16 shared readback
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    private let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    private let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer?             // [valueDim]
    private let gdnA: MTLBuffer?             // [numVHeads]
    private let gdnB: MTLBuffer?             // [numVHeads]
    private let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    private let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    private static let mtpChunkCapacity = 32
    private let mtpTokenBlock: MTLBuffer?
    private let mtpEmbeddingBlock: MTLBuffer?
    private let mtpNormalizedEmbeddingBlock: MTLBuffer?
    private let mtpNormalizedHiddenBlock: MTLBuffer?
    private let mtpConcatBlock: MTLBuffer?
    private let mtpProjectedBlock: MTLBuffer?
    private let mtpTargetHiddenBlock: MTLBuffer?
    private var mtpPrefillReadback: MTLBuffer?
    /// Reusable UInt32 token-ID buffer for chunked prefill (R23): sized to the
    /// largest chunk seen so far and grown on demand, so the prefill hot path
    /// never allocates an MTLBuffer per chunk.
    private var prefillTokenBuffer: MTLBuffer?

    /// Host scratch reused across prefill chunks (R38) and decode layers (R16).
    /// The runner is single-flight per generation (guarded by
    /// `prefillChunkState` and the callers' serial decode loop), so these
    /// never alias concurrent work.
    private var routeIDScratch: [UInt32] = []
    private var routeWeightScratch: [Float16] = []
    private var decodeExpertsScratch: [Int] = []
    private var decodeHitSlotsScratch: [UInt32] = []
    private var decodeMissSlotsScratch: [UInt32] = []
    private var decodeHitSplitRoutedBufsScratch: [MTLBuffer] = []
    private var decodeRoutedBufsScratch: [MTLBuffer] = []

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production,
                enableSpeculativeGDN: Bool = false) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
            && model.lmHeadWeightBits == 4
            && model.attentionWeightBits == 4
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillChunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.affineEmbed = model.embeddingWeightBits == 4 ? nil
            : try AffineQuantEmbeddingLookup(context: context,
                                             weightBits: model.embeddingWeightBits)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes)
        self.affine = model.attentionWeightBits == 4 ? nil
            : try AffineQuantGEMV(context: context,
                                  weightBits: model.attentionWeightBits)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 routedWeightBits: model.routedExpertWeightBits,
                                 routerWeightBits: model.routerWeightBits,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts))
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(
            context: context,
            weightBits: model.embeddingWeightBits)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(
            context: context,
            weightBits: model.attentionWeightBits)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillRouter = try PrefillRouter(context: context,
                                               weightBits: model.routerWeightBits)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            siluActivation: silu)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(
            context: context,
            siluActivation: silu,
            weightBits: model.routedExpertWeightBits)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(
            context: context,
            maxD: cfg.hiddenSize,
            weightBits: model.lmHeadWeightBits)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise = cfg.attnOutputGate
            || cfg.sharedExpertGated
            || cfg.hasLinearAttentionLayers
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(context: context, config: cfg.linearAttention,
                               specializedHiddenSize: cfg.hiddenSize)
            self.gdnState = try GDNStateManager(
                device: context.device,
                config: cfg,
                enableSpeculativeCheckpoint: enableSpeculativeGDN)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = cfg.ropeNeoxSubdim ? try RoPE(context: context) : nil
        self.int8ScalarGate = cfg.sharedExpertGated
            ? try DequantInt8GEMV(context: context,
                                  additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int,
                 _ stride: Int = MemoryLayout<Float16>.size,
                 label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        self.hidden        = try buf(D, label: "decode.hidden")
        self.normed        = try buf(D, label: "decode.normed")
        self.attnOut       = try buf(maxQ, label: "decode.attnOut")
        self.qScratch      = try buf(maxQ, label: "decode.qScratch")
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.kStage")
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim), label: "decode.vStage")
        self.oOut          = try buf(D, label: "decode.oOut")
        self.h1Buf         = try buf(D, label: "decode.h1")
        self.h2Buf         = try buf(D, label: "decode.h2")
        self.routedX       = try buf(D, label: "decode.routedX")
        self.denseX        = try buf(D, label: "decode.denseX")
        self.denseScratchGate = try buf(F, label: "decode.denseScratchGate")
        self.denseScratchUp   = try buf(F, label: "decode.denseScratchUp")
        self.denseScratchAct  = try buf(F, label: "decode.denseScratchAct")
        self.routerInput   = try buf(D, label: "decode.routerInput")
        self.zeroResidual  = try buf(D, label: "decode.zeroResidual")
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.outIndices")
        self.outWeights    = try buf(cfg.topKExperts, label: "decode.outWeights")
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize, label: "decode.moeActs")
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeHitActiveSlots")
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeMissActiveSlots")
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        tok.label = "decode.greedyToken"
        self.greedyTokenBuf = tok
        self.verificationHidden = try buf(2 * D, label: "decode.verificationHidden")
        self.verificationLogits = try buf(2 * cfg.vocabSize, label: "decode.verificationLogits")

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ, label: "decode.qPackedScratch")
            self.attnGateScratch = try buf(maxQ, label: "decode.attnGateScratch")
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim, label: "decode.gdnQKVRaw")
            self.gdnConvOut = try buf(la.qkvDim, label: "decode.gdnConvOut")
            self.gdnZ = try buf(la.valueDim, label: "decode.gdnZ")
            self.gdnA = try buf(la.numVHeads, label: "decode.gdnA")
            self.gdnB = try buf(la.numVHeads, label: "decode.gdnB")
            self.gdnY = try buf(la.valueDim, label: "decode.gdnY")
            self.gdnOut = try buf(la.valueDim, label: "decode.gdnOut")
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf = cfg.sharedExpertGated ? try buf(1, label: "decode.sharedScalarGate") : nil
        if cfg.family == .qwen36MTP {
            guard let tokenBlock = ctx.device.makeBuffer(
                length: Self.mtpChunkCapacity * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            self.mtpTokenBlock = tokenBlock
            self.mtpEmbeddingBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.embedding")
            self.mtpNormalizedEmbeddingBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.normalizedEmbedding")
            self.mtpNormalizedHiddenBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.normalizedHidden")
            self.mtpConcatBlock = try buf(Self.mtpChunkCapacity * 2 * D, label: "mtp.concat")
            self.mtpProjectedBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.projected")
            self.mtpTargetHiddenBlock = try buf(Self.mtpChunkCapacity * D, label: "mtp.targetHidden")
        } else {
            self.mtpTokenBlock = nil
            self.mtpEmbeddingBlock = nil
            self.mtpNormalizedEmbeddingBlock = nil
            self.mtpNormalizedHiddenBlock = nil
            self.mtpConcatBlock = nil
            self.mtpProjectedBlock = nil
            self.mtpTargetHiddenBlock = nil
        }
        self.mtpPrefillReadback = nil

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                scalarGate: cfg.sharedExpertGated
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        // Plain linear router (Qwen): one shared BF16 ones buffer keeps
        // the router kernel's effective_scale multiply neutral, and a ones
        // per_expert_scale keeps the top-k weights untouched. (Softmax
        // over top-k then renormalize equals Qwen's softmax over all
        // experts then renormalize the selected top-k.)
        let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
        self.effectiveScaleBuffers = [MTLBuffer](repeating: ones,
                                                 count: cfg.numLayers)
        self.onesPerExpertScale = try bf16OnesBuffer(count: cfg.numExperts,
                                                     label: "per_expert_scale.ones")
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    public func captureInferenceState(
        maximumBytes: Int? = nil
    ) throws -> InferenceStateSnapshot {
        guard let kv, kv.position > 0 else {
            throw InferenceStateSnapshotError.invalidPosition(kv?.position ?? 0)
        }
        let kvLengths = try kv.snapshotSegmentLengths(at: kv.position)
        let gdnLengths = gdnState?.snapshotSegmentLengths() ?? []
        var payloadBytes = 0
        for length in kvLengths + gdnLengths {
            let (next, overflow) = payloadBytes.addingReportingOverflow(length)
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            payloadBytes = next
        }
        if let maximumBytes, payloadBytes > maximumBytes {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: payloadBytes,
                limit: maximumBytes)
        }
        var payload = Data()
        payload.reserveCapacity(payloadBytes)
        try kv.appendSnapshotPayload(to: &payload, segmentLengths: kvLengths)
        try gdnState?.appendSnapshotPayload(to: &payload, segmentLengths: gdnLengths)
        guard payload.count == payloadBytes else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: payloadBytes,
                actual: payload.count)
        }
        return InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: kv.position,
                kvSegmentLengths: kvLengths,
                gdnSegmentLengths: gdnLengths,
                payloadBytes: payloadBytes),
            payload: payload)
    }

    func captureSpeculativeCheckpoint(maximumBytes: Int) throws
        -> SpeculativeInferenceCheckpoint {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        let required = gdnState?.speculativePayloadBytes ?? 0
        guard required <= maximumBytes else {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: required,
                limit: maximumBytes)
        }
        return SpeculativeInferenceCheckpoint(position: kv.position)
    }

    func rollbackSpeculativeCheckpoint(_ checkpoint: SpeculativeInferenceCheckpoint) throws {
        guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
        if let gdnState {
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            try gdnState.encodeSpeculativeRestore(commandBuffer: cb)
            cb.commit()
            try waitForCompletion(cb)
        }
        // Row zero was confirmed and is present in the on-GPU checkpoint.
        try kv.rewind(to: checkpoint.position + 1)
        resetTransientState()
    }

    /// Discard an unaccepted native-MTP cache row. The draft contains only
    /// trimmable full-attention KV, so its logical cursor can move back without
    /// copying payload bytes; the next draft pass overwrites the stale row.
    func rewindMTP(to position: Int) throws {
        guard cfg.family == .qwen36MTP, let kv else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        try kv.rewind(to: position)
        resetTransientState()
    }

    var speculativeRollbackBytes: Int {
        gdnState?.speculativePayloadBytes ?? 0
    }

    /// Verify `[confirmed, draft]` in the existing batched prefill path. The
    /// two target logits and target hidden rows are produced from one 40-layer
    /// backbone traversal, which is the source of MTP's decode speedup.
    func verifyGreedyPair(_ tokens: [Int32],
                          startPosition: Int) async throws -> TargetPairVerification {
        guard tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported("MTP verification requires exactly two tokens")
        }
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: config)
        try await executePrefillChunk(tokens: tokens[...],
                                      startPosition: startPosition,
                                      outputMode: .logits,
                                      logits: verificationLogits,
                                      scratch: scratch,
                                      config: config,
                                      writeFinalHead: false,
                                      snapshotGDNAfterFirstToken: true,
                                      useTwoRowProjection: true)

        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let cb = ctx.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: scratch.hidden,
                  sourceOffset: 0,
                  to: verificationHidden,
                  destinationOffset: 0,
                  size: 2 * cfg.hiddenSize * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        let logitsStride = cfg.vocabSize * MemoryLayout<Float16>.stride
        for row in 0..<2 {
            try prefillFinalRowHead.encodeLogits(
                commandBuffer: cb,
                hiddenBlock: scratch.hidden,
                row: row,
                rowStrideElements: cfg.hiddenSize,
                normWeight: finalNorm.buffer,
                normWeightOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                logits: verificationLogits,
                logitsOffset: row * logitsStride,
                d: UInt32(cfg.hiddenSize),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: 1e-6)
        }
        cb.commit()
        try waitForCompletion(cb)

        let logits = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        func argmax(row: Int) -> Int32 {
            let base = row * cfg.vocabSize
            var best = 0
            var bestValue = Float(logits[base])
            for index in 1..<cfg.vocabSize {
                let value = Float(logits[base + index])
                if value > bestValue {
                    bestValue = value
                    best = index
                }
            }
            return Int32(best)
        }
        return TargetPairVerification(
            predictionAfterFirst: argmax(row: 0),
            predictionAfterSecond: argmax(row: 1),
            hiddenRows: Data(bytes: verificationHidden.contents(),
                             count: 2 * cfg.hiddenSize * MemoryLayout<Float16>.stride))
    }

    /// Advance the one-layer MTP sidecar with aligned `(target hidden,
    /// next-token)` pairs. At most 32 rows are admitted so adapter scratch is
    /// fixed and the routed expert cache remains exactly top-k sized.
    func advanceMTP(tokens: ArraySlice<Int32>,
                    targetHiddenRows: Data,
                    startPosition: Int,
                    predictNext: Bool) async throws -> Int32? {
        guard cfg.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= Self.mtpChunkCapacity else {
            throw PrefillError.chunkedUnsupported(
                "MTP adapter accepts 1...\(Self.mtpChunkCapacity) aligned rows")
        }
        let D = cfg.hiddenSize
        let expectedBytes = tokens.count * D * MemoryLayout<Float16>.stride
        guard targetHiddenRows.count == expectedBytes else {
            throw PrefillError.chunkedUnsupported(
                "MTP target hidden payload has \(targetHiddenRows.count) bytes; expected \(expectedBytes)")
        }
        guard let tokenBuffer = mtpTokenBlock,
              let embeddingBlock = mtpEmbeddingBlock,
              let normalizedEmbedding = mtpNormalizedEmbeddingBlock,
              let normalizedHidden = mtpNormalizedHiddenBlock,
              let concat = mtpConcatBlock,
              let projected = mtpProjectedBlock,
              let targetHidden = mtpTargetHiddenBlock,
              let elementwise else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        targetHiddenRows.copyBytes(to: targetHidden.contents()
            .assumingMemoryBound(to: UInt8.self), count: expectedBytes)
        let ids = tokens.map { UInt32(bitPattern: $0) }
        ids.withUnsafeBytes { bytes in
            tokenBuffer.contents().copyMemory(from: bytes.baseAddress!,
                                              byteCount: bytes.count)
        }
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let emb = try model.embedding()
        try prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: embeddingBlock,
                            t: UInt32(tokens.count),
                            d: UInt32(D),
                            outScale: 1,
                            vocab: UInt32(cfg.vocabSize))
        let embeddingNorm = try model.mtpEmbeddingNorm()
        let hiddenNorm = try model.mtpHiddenNorm()
        try prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: embeddingBlock,
                               weight: embeddingNorm.buffer,
                               weightOffset: Int(embeddingNorm.offset),
                               out: normalizedEmbedding,
                               t: UInt32(tokens.count),
                               d: UInt32(D), eps: 1e-6)
        try prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: targetHidden,
                               weight: hiddenNorm.buffer,
                               weightOffset: Int(hiddenNorm.offset),
                               out: normalizedHidden,
                               t: UInt32(tokens.count),
                               d: UInt32(D), eps: 1e-6)
        try elementwise.encodeConcatRows(commandBuffer: cb,
                                     lhs: normalizedEmbedding,
                                     rhs: normalizedHidden,
                                     out: concat,
                                     rows: tokens.count,
                                     dim: D)
        let projection = try model.mtpProjection()
        try prefillQMM.encode(commandBuffer: cb,
                          weights: projection.buffer,
                          weightsOffset: Int(projection.offset),
                          scales: projection.buffer,
                          scalesOffset: Int(projection.scaleOffset),
                          biases: projection.buffer,
                          biasesOffset: Int(projection.biasOffset),
                          x: concat,
                          y: projected,
                          t: tokens.count,
                          n: D,
                          k: 2 * D)
        cb.commit()
        try waitForCompletion(cb)

        let runtime = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: runtime)
        let mode: PrefillOutputMode = useFusedGreedyHead ? .greedyIfAvailable : .logits
        try await executePrefillChunk(tokens: tokens,
                                      startPosition: startPosition,
                                      outputMode: mode,
                                      logits: verificationLogits,
                                      scratch: scratch,
                                      config: runtime,
                                      writeFinalHead: predictNext,
                                      preparedHidden: projected)
        guard predictNext else { return nil }
        if useFusedGreedyHead {
            return Int32(bitPattern: lastGreedyToken)
        }
        let values = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<cfg.vocabSize {
            let value = Float(values[index])
            if value > bestValue {
                best = index
                bestValue = value
            }
        }
        return Int32(best)
    }

    private func ensureMTPPrefillReadback(rows: Int) throws -> MTLBuffer {
        let bytes = rows * cfg.hiddenSize * MemoryLayout<Float16>.stride
        if let existing = mtpPrefillReadback, existing.length >= bytes {
            return existing
        }
        guard let buffer = ctx.device.makeBuffer(length: bytes,
                                                 options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = "mtp.target-hidden-readback"
        mtpPrefillReadback = buffer
        return buffer
    }

    public func restoreInferenceState(_ snapshot: InferenceStateSnapshot) throws {
        do {
            let descriptor = snapshot.descriptor
            guard descriptor.version == InferenceStateSnapshotDescriptor.currentVersion else {
                throw InferenceStateSnapshotError.unsupportedVersion(descriptor.version)
            }
            guard descriptor.position > 0, descriptor.position <= maxContext else {
                throw InferenceStateSnapshotError.invalidPosition(descriptor.position)
            }
            let expectedBytes = try descriptor.validatedPayloadBytes()
            guard snapshot.payload.count == expectedBytes else {
                throw InferenceStateSnapshotError.invalidPayloadSize(
                    expected: expectedBytes,
                    actual: snapshot.payload.count)
            }
            guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
            try snapshot.payload.withUnsafeBytes { bytes in
                var offset = 0
                try kv.restoreSnapshot(
                    position: descriptor.position,
                    segmentLengths: descriptor.kvSegmentLengths,
                    bytes: bytes,
                    offset: &offset)
                if let gdnState {
                    try gdnState.restoreSnapshot(
                        segmentLengths: descriptor.gdnSegmentLengths,
                        bytes: bytes,
                        offset: &offset)
                } else if !descriptor.gdnSegmentLengths.isEmpty {
                    throw InferenceStateSnapshotError.invalidLayout
                }
                guard offset == bytes.count else {
                    throw InferenceStateSnapshotError.invalidLayout
                }
            }
            resetTransientState()
        } catch {
            reset()
            throw error
        }
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (NVMAI_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + rdadvise + io + cb2.
    public private(set) var totalWaitNanos: UInt64 = 0
    public private(set) var totalBodyNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    // MARK: - Per-command-buffer GPU timing (NVMAI_KERNEL_STATS)

    /// One command buffer's GPU span for a named kernel role. The decode path
    /// is synchronous (commit + wait), so `gpuStartTime`/`gpuEndTime` are
    /// valid right after completion and cost nothing to read.
    private struct KernelGPUTiming {
        let role: String
        let start: TimeInterval
        let end: TimeInterval
    }
    private var kernelGPUTimings: [KernelGPUTiming] = []
    private let kernelGPUTimingsEnabled =
        ProcessInfo.processInfo.environment["NVMAI_KERNEL_STATS"] != nil

    public func resetKernelGPUTimings() {
        kernelGPUTimings.removeAll(keepingCapacity: true)
    }

    func recordKernelGPU(role: String, _ cb: MTLCommandBuffer) {
        guard kernelGPUTimingsEnabled, cb.gpuEndTime > 0 else { return }
        kernelGPUTimings.append(
            KernelGPUTiming(role: role, start: cb.gpuStartTime, end: cb.gpuEndTime))
    }

    /// Aggregated per-role GPU milliseconds for the current generation,
    /// largest first.
    public func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] {
        var acc: [String: (millis: Double, count: Int)] = [:]
        for t in kernelGPUTimings {
            let millis = (t.end - t.start) * 1000
            acc[t.role, default: (0, 0)].millis += millis
            acc[t.role]!.count += 1
        }
        return acc.map { (role: $0.key, millis: $0.value.millis, count: $0.value.count) }
            .sorted { $0.millis > $1.millis }
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            // Skip window is inclusive of `position`, matching the adaptive
            // policy (`position <= skipUntilPosition`), so both policies
            // suppress advice for the same token window after a slow call.
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        do {
            for (spanIndex, span) in spans.enumerated() {
                try Task.checkCancellation()
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                try await executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: config,
                    writeFinalHead: spanIndex == spans.count - 1)
                try Task.checkCancellation()
                onProgress(span.completedCount)
            }
        } catch {
            // Any failure — cancellation, a GPU command-buffer error, an I/O
            // error mid-routed-fetch — may have written partial KV rows and
            // left the chunk state dirty. Reset so the next request does not
            // trip `chunkedRunnerDirty` on a stale in-flight chunk.
            reset()
            throw error
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    /// Target prefill with a bounded hidden-state tap that simultaneously
    /// aligns the streaming MTP sidecar. Only one target chunk is exposed at a
    /// time; no prompt-sized hidden-state tensor is retained.
    func prefillChunkedWithMTP(tokens: ArraySlice<Int32>,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               mtp: RealForwardRunner,
                               onProgress: (Int) -> Void) async throws -> MTPPrefillResult {
        guard cfg.family == .qwen36 else {
            throw StreamingMTPError.targetMustBeQwen36
        }
        guard mtp.cfg.family == .qwen36MTP else {
            throw StreamingMTPError.sidecarMustBeQwen36MTP
        }
        guard !tokens.isEmpty, tokens.count <= mtp.maxContext else {
            throw PrefillError.chunkedUnsupported(
                "MTP prompt must fit its bounded \(mtp.maxContext)-token draft context")
        }
        reset()
        mtp.reset()
        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: 0,
                                              config: config)
        var carry: Data?
        do {
            for (spanIndex, span) in spans.enumerated() {
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                let chunk = tokens[lower..<upper]
                try await executePrefillChunk(tokens: chunk,
                                              startPosition: span.startPosition,
                                              outputMode: useFusedGreedyHead
                                                ? .greedyIfAvailable : .logits,
                                              logits: logits,
                                              scratch: scratch,
                                              config: config,
                                              writeFinalHead: spanIndex == spans.count - 1)

                let readback = try ensureMTPPrefillReadback(rows: span.tokenCount)
                guard let cb = ctx.queue.makeCommandBuffer(),
                      let blit = cb.makeBlitCommandEncoder() else {
                    throw ModelError.residentBufferWrapFailed
                }
                let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
                blit.copy(from: scratch.hidden, sourceOffset: 0,
                          to: readback, destinationOffset: 0,
                          size: span.tokenCount * rowBytes)
                blit.endEncoding()
                cb.commit()
                try waitForCompletion(cb)
                let chunkHidden = Data(bytes: readback.contents(),
                                       count: span.tokenCount * rowBytes)

                var pairTokens: [Int32] = []
                var pairHidden = Data()
                if let carry {
                    pairTokens.reserveCapacity(span.tokenCount)
                    pairTokens.append(contentsOf: chunk)
                    pairHidden.reserveCapacity(span.tokenCount * rowBytes)
                    pairHidden.append(carry)
                    if span.tokenCount > 1 {
                        pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                    }
                } else if span.tokenCount > 1 {
                    pairTokens.append(contentsOf: chunk.dropFirst())
                    pairHidden.append(chunkHidden.prefix((span.tokenCount - 1) * rowBytes))
                }
                var pairOffset = 0
                while pairOffset < pairTokens.count {
                    let count = min(Self.mtpChunkCapacity, pairTokens.count - pairOffset)
                    let hiddenStart = pairOffset * rowBytes
                    let hiddenEnd = hiddenStart + count * rowBytes
                    _ = try await mtp.advanceMTP(
                        tokens: pairTokens[pairOffset..<(pairOffset + count)],
                        targetHiddenRows: pairHidden.subdata(in: hiddenStart..<hiddenEnd),
                        startPosition: mtp.continuationPosition,
                        predictNext: false)
                    pairOffset += count
                }
                carry = Data(chunkHidden.suffix(rowBytes))
                onProgress(span.completedCount)
            }
        } catch {
            // A failed chunk (cancellation, GPU error, expert-fetch I/O) may
            // have left partial KV rows in both runners; clear both so the
            // next request starts clean.
            reset()
            mtp.reset()
            throw error
        }
        guard let lastTargetHidden = carry else {
            throw StreamingMTPError.draftNotReady
        }
        let seed: PrefillSeed = useFusedGreedyHead
            ? .greedyToken(lastGreedyToken) : .logitsWritten
        return MTPPrefillResult(
            target: PrefillResult(newPosition: tokens.count, seed: seed),
            lastTargetHidden: lastTargetHidden)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     preparedHidden: MTLBuffer? = nil,
                                     snapshotGDNAfterFirstToken: Bool = false,
                                     useTwoRowProjection: Bool = false) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        guard !snapshotGDNAfterFirstToken || tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported(
                "Gated-DeltaNet speculative checkpoint requires two rows")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }


        let layerViews = try makeLayerPrefillViews()

        // Reused UInt32 token-ID buffer, sized to the largest chunk seen so
        // far and grown on demand (R23); the prefill hot path never allocates
        // a Metal buffer per chunk.
        let tokenBytes = tokens.count * MemoryLayout<UInt32>.stride
        let tokenBuffer: MTLBuffer
        if let existing = prefillTokenBuffer, existing.length >= tokenBytes {
            tokenBuffer = existing
        } else {
            guard let made = ctx.device.makeBuffer(length: tokenBytes,
                                                   options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "prefill.tokenIDs"
            prefillTokenBuffer = made
            tokenBuffer = made
        }
        let tokenPtr = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, token) in tokens.enumerated() {
            tokenPtr[i] = UInt32(bitPattern: token)
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = try model.embedding()


        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let preparedHidden {
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: preparedHidden,
                      sourceOffset: 0,
                      to: scratch.hidden,
                      destinationOffset: 0,
                      size: t * D * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        } else {
            try prefillEmbed.encode(commandBuffer: cb,
                                table: emb.buffer,
                                tableOffset: Int(emb.offset),
                                scales: emb.buffer,
                                scalesOffset: Int(emb.scaleOffset),
                                biases: emb.buffer,
                                biasesOffset: Int(emb.biasOffset),
                                tokens: tokenBuffer,
                                out: scratch.hidden,
                                t: UInt32(t),
                                d: UInt32(D),
                                outScale: embedOutScale,
                                vocab: UInt32(cfg.vocabSize))
        }

        let prefillProfile = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASES"] != nil
        var prefillRouteNanos: UInt64 = 0
        var prefillTileNanos: UInt64 = 0
        var prefillTailNanos: UInt64 = 0

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            let prefillLayerStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var prefillRouteEnd = prefillLayerStart
            var prefillTileEnd = prefillLayerStart
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            try prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if isLinear {
                try encodeLinearAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    snapshotGDNAfterFirstToken: snapshotGDNAfterFirstToken,
                    useTwoRowProjection: useTwoRowProjection)
            } else {
                try encodeFullAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D, startPosition: startPosition,
                    isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                    qDim: qDim, kvDim: kvDim, rmsEps: eps,
                    useTwoRowProjection: useTwoRowProjection)
            }
            // Plain pre-norm residual block: hidden += attention branch,
            // then one post-attention norm feeds router, shared expert,
            // and routed phase 1 (routedX doubles as moeX).
            try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                           hidden: scratch.hidden,
                                           delta: scratch.h1,
                                           count: t * D)
            try prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.postAttention.buffer,
                                   weightOffset: Int(views.postAttention.offset),
                                   out: scratch.routedX,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
                (onesPerExpertScale!, 0)
            try prefillRouter.encodeBlock(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routedX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                    cb.commit()
                    try waitForCompletion(cb)

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    // Reused per-chunk host scratch (R38): cleared in place so
                    // the routed-tile planner never allocates per chunk.
                    routeIDScratch.removeAll(keepingCapacity: true)
                    routeWeightScratch.removeAll(keepingCapacity: true)
                    routeIDScratch.reserveCapacity(routeCount)
                    routeWeightScratch.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDScratch.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeightScratch.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDScratch,
                                                                   weights: routeWeightScratch,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig: PrefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount() {
                        guard let fitted = Self.prefillRoutedTileSchedulerConfig.fitting(
                            slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
                        }
                        schedulerConfig = fitted
                        routeTileExpertCount = fitted.tileExperts
                    } else {
                        schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))
                    prefillRouteEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    prefillRouteNanos &+= prefillRouteEnd - prefillLayerStart

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.routedX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    if cfg.sharedExpertGated {
                        // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
                        // per chunk row.
                        let gateView = sharedProj.scalarGate!
                        let halfBytes = MemoryLayout<Float16>.stride
                        for row in 0..<t {
                            try int8ScalarGate!.encode(
                                commandBuffer: sharedCB,
                                weights: gateView.buffer,
                                weightsOffset: Int(gateView.offset),
                                scales: gateView.buffer,
                                scalesOffset: Int(gateView.scaleOffset),
                                biases: gateView.buffer,
                                biasesOffset: Int(gateView.biasOffset),
                                x: scratch.routedX,
                                xOffset: row * D * halfBytes,
                                y: scratch.sharedScalarGate,
                                yOffset: row * halfBytes,
                                m: 1, n: UInt32(D))
                        }
                        for row in 0..<t {
                            try elementwise!.encodeSigmoidScalarMul(
                                commandBuffer: sharedCB,
                                y: scratch.h1,
                                yOffset: row * D * halfBytes,
                                gate: scratch.sharedScalarGate,
                                gateOffset: row * halfBytes,
                                count: D)
                        }
                    }
                    sharedCB.commit()
                    try waitForCompletion(sharedCB)

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = try model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    // `withExtendedLifetime` below takes a non-throwing closure,
                    // so the wait error is captured here and rethrown after the
                    // fetched blobs are released.
                    var pendingTileError: Error?
                    var tailError: Error?
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            do {
                                try waitForCompletion(pending.commandBuffer)
                            } catch {
                                // Rethrown after the fetched blobs are released.
                                pendingTileError = error
                            }
                        }
                        if let error = pendingTileError {
                            pendingTileError = nil
                            throw error
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = try prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    prefillTileEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    prefillTileNanos &+= prefillTileEnd - prefillRouteEnd
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    try prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    // Plain pre-norm tail: hidden += gated shared branch
                    // + routed branch.
                    try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                   hidden: scratch.hidden,
                                                   delta: scratch.h1,
                                                   count: t * D)
                    try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                   hidden: scratch.hidden,
                                                   delta: scratch.h2,
                                                   count: t * D)
                    tailCB.commit()
                    withExtendedLifetime(metadata) {
                        do {
                            try waitForCompletion(tailCB)
                        } catch {
                            // Rethrown after `metadata` is released.
                            tailError = error
                        }
                    }
                    if let error = tailError {
                        tailError = nil
                        throw error
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    prefillTailNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillTileEnd
                    continue
        }

        if prefillProfile {
            let prefillTotal = prefillRouteNanos + prefillTileNanos + prefillTailNanos
            print("[prefill phases over \(t) tokens, \(prefillTotal / 1_000_000) ms total]")
            print("  route readback + GPU: \(String(format: "%.1f", Double(prefillRouteNanos) / 1e6)) ms")
            print("  expert fetch + tiles: \(String(format: "%.1f", Double(prefillTileNanos) / 1e6)) ms")
            print("  tail + residual:      \(String(format: "%.1f", Double(prefillTailNanos) / 1e6)) ms")
        }

        if writeFinalHead {
            try encodeFinalHead(logits: logits, scratch: scratch,
                                tokenCount: t, hiddenSize: D, rmsEps: eps,
                                outputMode: outputMode)
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        /// Drain a routed layer's command buffers, surfacing any `.error`
        /// (R1/R2): the routed-CB failure must fail the generation rather than
        /// print-and-continue into silently corrupt output. The per-layer call
        /// (waitIfNeeded: false) runs right after the next layer's tailCB
        /// wait, so the routed CBs have completed on the GPU and their spans
        /// are valid — recording them here (not only in the waitIfNeeded
        /// drain) makes NVMAI_KERNEL_STATS cover every layer instead of just
        /// the final layer of each token.
        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                if let sharedCB = pending.sharedCB {
                    try waitForCompletion(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    try waitForCompletion(phase1HitCB)
                }
                try waitForCompletion(pending.cb)
            } else if let err = pending.cb.error {
                throw ModelError.commandBufferFailed(
                    detail: "routed layer command buffer: \(err)")
            }
            if let sharedCB = pending.sharedCB, let err = sharedCB.error {
                throw ModelError.commandBufferFailed(
                    detail: "shared-expert command buffer: \(err)")
            }
            if let phase1HitCB = pending.phase1HitCB, let err = phase1HitCB.error {
                throw ModelError.commandBufferFailed(
                    detail: "routed phase-1 hit command buffer: \(err)")
            }
            if let sharedCB = pending.sharedCB {
                recordKernelGPU(role: "shared_expert", sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                recordKernelGPU(role: "moe_phase1_hit", phase1HitCB)
            }
            recordKernelGPU(role: "moe_phase1_2_routed", pending.cb)
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = try model.embedding()
        let embedCB = try runSync { cb in
            if let affineEmbed {
                try affineEmbed.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: UInt32(bitPattern: token),
                             d: D, outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            } else {
                try embedInt4.encode(commandBuffer: cb,
                             table:  emb.buffer, tableOffset:  Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden,
                             tokenId: UInt32(bitPattern: token),
                             d: D,
                             outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            }
        }
        guard embedCB != nil else {
            throw ModelError.residentBufferWrapFailed
        }
        if let embedCB { recordKernelGPU(role: "embed", embedCB) }

        for L in 0..<cfg.numLayers {
            let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let routerW  = try model.router(layer: L)
            let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
                (onesPerExpertScale!, 0)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Attention+router split into measured sub-command-buffers
            // (NVMAI_KERNEL_STATS): attnCB = input norm + QKV + epilogue
            // (or the linear/gated attention), softmaxCB = the softmax
            // attention pass on full layers, tailCB = O-proj + residual +
            // post-norm + router. Same queue, same order, one wait on the
            // last CB; only the router readback forces the barrier.
            guard let attnCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            try rms.encodeBF16W(commandBuffer: attnCB,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
            var softmaxCB: MTLCommandBuffer?
            guard let tailCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }

            if isLinear {
                // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
                // fixed-size recurrent state updated in place.
                try encodeLinearAttentionDecode(attnCB, layer: L)
            } else if cfg.attnOutputGate {
                // Qwen full attention: packed [query ; gate] q_proj, real
                // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
                try encodeGatedFullAttentionDecode(attnCB, layer: L,
                                                   position: position,
                                                   seqLen: seqLen)
            } else {
                let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
                let q     = try model.qProj(layer: L)
                let k     = try model.kProj(layer: L)
                // Under the K=V quirk full layers reuse k_proj; otherwise
                // v_proj is a real tensor.
                let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
                let o     = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)

                try fusedQKVGEMV.encode(commandBuffer: attnCB,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)

                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                try fusedQKVEpilogue.encode(commandBuffer: attnCB,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)

                guard kv != nil else {
                    throw ModelError.internalInconsistency(
                        detail: "FP16 attention requires an FP16 KV cache")
                }
                guard let attentionCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                softmaxCB = attentionCB
                if isFull {
                    try attention.encodeFull(commandBuffer: attentionCB,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    try attention.encodeSWA(commandBuffer: attentionCB,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                try int4.encode(commandBuffer: tailCB,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            // Plain pre-norm residual block: hidden += attention branch,
            // then one post-attention norm feeds router, shared expert,
            // and routed phase 1 (routedX doubles as moeX).
            try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                           hidden: hidden,
                                           delta: oOut,
                                           count: cfg.hiddenSize)
            try rms.encodeBF16W(commandBuffer: tailCB,
                            x: hidden,
                            weight: postAttn.buffer,
                            weightOffset: Int(postAttn.offset),
                            out: routedX,
                            d: D, eps: eps)

            try moe.encodeRouter(commandBuffer: tailCB,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            attnCB.commit()
            if let attentionCB = softmaxCB {
                attentionCB.commit()
            }
            tailCB.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try waitForCompletion(tailCB)
            recordKernelGPU(role: "attn_norm_qkv", attnCB)
            if let attentionCB = softmaxCB {
                recordKernelGPU(role: "attn_softmax", attentionCB)
            }
            recordKernelGPU(role: "attn_tail_router", tailCB)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            totalWaitNanos &+= waitNanos
            var prevRoutedUs: Double = 0
            if let pending = pendingRoutedCommand {
                prevRoutedUs = (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1_000_000
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk. The expert
            // id list is reused host scratch (R16); the runner is single-flight
            // per generation, so it never aliases concurrent decode work.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            decodeExpertsScratch.removeAll(keepingCapacity: true)
            decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                decodeExpertsScratch.append(min(Int(idxPtr[i]), cfg.numExperts - 1))
            }
            let experts = decodeExpertsScratch

            let routedOffsets = try model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            decodeHitSplitRoutedBufsScratch.removeAll(keepingCapacity: true)
            decodeHitSlotsScratch.removeAll(keepingCapacity: true)
            decodeMissSlotsScratch.removeAll(keepingCapacity: true)
            let phase1HitSlots = decodeHitSlotsScratch
            let phase1MissSlots = decodeMissSlotsScratch

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                for index in 0..<cfg.topKExperts where !missSet.contains(index) {
                    decodeHitSlotsScratch.append(UInt32(index))
                }
                for miss in plan.misses {
                    decodeMissSlotsScratch.append(UInt32(miss))
                }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) throws {
                try moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) throws {
                try moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                for blob in plannedBlobs {
                    decodeHitSplitRoutedBufsScratch.append(blob.buffer)
                }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: decodeHitSplitRoutedBufsScratch,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    guard let cb = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    try encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: decodeHitSplitRoutedBufsScratch,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on its normed input, not on
            // the routed experts. Commit it without waiting so its GPU work
            // overlaps the routed-expert pread. The routed CB follows it on
            // the same queue, so the combine sees h1Buf.
            guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            try shared.encode(commandBuffer: sharedCB,
                              x: routedX,
                              gate: sharedProj.gate,
                              up: sharedProj.up,
                              down: sharedProj.down,
                              y: h1Buf,
                              scratchGate: denseScratchGate,
                              scratchUp: denseScratchUp,
                              scratchAct: denseScratchAct)
            if cfg.sharedExpertGated {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                try int8ScalarGate!.encode(commandBuffer: sharedCB,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: routedX,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
                try elementwise!.encodeSigmoidScalarMul(commandBuffer: sharedCB,
                                                    y: h1Buf,
                                                    gate: sharedScalarGateBuf!,
                                                    count: cfg.hiddenSize)
            }
            sharedCB.commit()
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            decodeRoutedBufsScratch.removeAll(keepingCapacity: true)
            for blob in blobs {
                decodeRoutedBufsScratch.append(blob.buffer)
            }
            let routedBufs = decodeRoutedBufsScratch
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // The phase-2 reduce already folded the shared branch (h1Buf
            // as its residual); the tail is a plain residual add.
            let gTail: (MTLCommandBuffer) throws -> Void = { [self] cb in
                try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: hidden,
                                               delta: h2Buf,
                                               count: cfg.hiddenSize)
            }
            guard let routedCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                try encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                try encodeRoutedPhase1Full(routedCB,
                                           argBuf: argBuf,
                                           routedBufs: routedBufs)
            }
            try moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: h1Buf,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            try gTail(routedCB)
            routedCB.commit()
            guard pendingRoutedCommand == nil else {
                // The pipeline drains the previous layer's routed CB before
                // queuing the next, so this is a logic error, not a user
                // condition — but it must fail the generation, not trap.
                throw ModelError.internalInconsistency(
                    detail: "routed command-buffer pipeline not drained before queuing the next layer")
            }
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
            if ProcessInfo.processInfo.environment["NVMAI_LAYER_TRACE"] != nil,
               position < 3 || position % 16 == 0 {
                let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let attnUs = (attnCB.gpuEndTime - attnCB.gpuStartTime) * 1_000_000
                let tailUs = (tailCB.gpuEndTime - tailCB.gpuStartTime) * 1_000_000
                print("NVMAI layer pos=\(position) L=\(L) "
                    + "body_us=\((now - tBodyStart) / 1000) "
                    + "wait_us=\(waitNanos / 1000) io_us=\(layerIo / 1000) "
                    + "cb1_us=\((tWait - tCb1Start) / 1000) "
                    + "cb2_us=\((now - tCb2Start) / 1000) "
                    + "gpu_attn_us=\(Int(attnUs)) gpu_tail_us=\(Int(tailUs)) "
                    + "gpu_routed_us=\(Int(prevRoutedUs))")
            }
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = try model.finalNorm()
        let lm    = try model.lmHead()
        let gFinalNorm: (MTLCommandBuffer) throws -> Void = { cb in
            try self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodePrimaryGEMV(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                if let headCB = try runSync(gFusionHead) {
                    recordKernelGPU(role: "head_fused", headCB)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                guard let headCB = try runSync({ cb in
                    try gFinalNorm(cb)
                    try gLmHead(cb)
                }) else {
                    throw ModelError.residentBufferWrapFailed
                }
                recordKernelGPU(role: "head_logits", headCB)
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        if model.attentionWeightBits == 4 {
            try gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)
        } else {
            try encodePrimaryGEMV(commandBuffer: cb, projection: qkvW,
                              x: normed, y: gdnQKVRaw,
                              m: UInt32(la.qkvDim), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: zW,
                              x: normed, y: gdnZ,
                              m: UInt32(la.valueDim), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: aW,
                              x: normed, y: gdnA,
                              m: UInt32(la.numVHeads), n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: bW,
                              x: normed, y: gdnB,
                              m: UInt32(la.numVHeads), n: D)
        }

        try gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        try gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        try gdn.encodeGatedNorm(commandBuffer: cb,
                            y: gdnY,
                            z: gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: gdnOut)
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail: "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "FP16 attention requires an FP16 KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        if model.attentionWeightBits == 4 {
            try fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        } else {
            try encodePrimaryGEMV(commandBuffer: cb, projection: q,
                              x: normed, y: qPackedScratch,
                              m: 2 * qDim, n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: k,
                              x: normed, y: kSlot.buffer,
                              yOffset: kSlot.offset, m: kvDim, n: D)
            try encodePrimaryGEMV(commandBuffer: cb, projection: v,
                              x: normed, y: vSlot.buffer,
                              yOffset: vSlot.offset, m: kvDim, n: D)
        }
        try elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        try rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: eps)
        try rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(position),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        try attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale))
        try elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    private func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   projection p: TensorView,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        try encodePrimaryGEMV(commandBuffer: cb,
                          weights: p.buffer, weightsOffset: Int(p.offset),
                          scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                          biases: p.buffer, biasesOffset: Int(p.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
    }

    private func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int,
                                   scales: MTLBuffer, scalesOffset: Int,
                                   biases: MTLBuffer, biasesOffset: Int,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        if let affine {
            try affine.encode(commandBuffer: cb,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
        } else {
            try int4.encode(commandBuffer: cb,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        }
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> MTLCommandBuffer? {
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
        return cb
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
    }


    // MARK: - Chunked prefill helpers

    /// Per-layer tensor views resolved once before the chunk loop.
    private struct LayerPrefillQKVViews {
        let inputNorm: TensorView
        let postAttention: TensorView
        let router: TensorView
        // Softmax-attention layers only (nil on linear-attention layers).
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet linear-attention layers only.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?
    }

    private func encodeAffineProjection(commandBuffer: MTLCommandBuffer,
                              family: PrefillProjectionFamily,
                              weights: TensorView,
                              x: MTLBuffer,
                              y: MTLBuffer,
                              rows: Int,
                              columns: Int,
                              tokenCount: Int,
                              xStrideElements: Int,
                              yStrideElements: Int,
                              useTwoRowProjection: Bool) throws {
        if tokenCount >= 32,
           family == .q || family == .kv || family == .o,
           let candidate = prefillMPPAffineInt4 {
            let path = try candidate.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                m: tokenCount,
                n: rows,
                k: columns)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        if useTwoRowProjection && tokenCount == 2
            && xStrideElements == columns && yStrideElements == rows {
            if model.attentionWeightBits == 4 {
                try int4.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            } else {
                try affine!.encodeTwoRows(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: UInt32(rows),
                    n: UInt32(columns))
            }
            return
        }
        if PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount) == .qmm {
            try prefillQMM.encode(commandBuffer: commandBuffer,
                              weights: weights.buffer,
                              weightsOffset: Int(weights.offset),
                              scales: weights.buffer,
                              scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer,
                              biasesOffset: Int(weights.biasOffset),
                              x: x,
                              y: y,
                              t: tokenCount,
                              n: rows,
                              k: columns)
            return
        }
        for row in 0..<tokenCount {
            try encodePrimaryGEMV(
                commandBuffer: commandBuffer,
                projection: weights,
                x: x,
                xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                y: y,
                yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                m: UInt32(rows),
                n: UInt32(columns))
        }
    }

    private func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                       source: MTLBuffer,
                       destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                       sourceTokenOffset: Int,
                       tokenCount: Int,
                       bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: source,
                  sourceOffset: sourceTokenOffset * bytesPerToken,
                  to: destination.buffer,
                  destinationOffset: destination.offset,
                  size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    private func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                              kv: KVCacheManager,
                              layer: Int,
                              startPosition: Int,
                              tokenCount: Int,
                              keySource: MTLBuffer,
                              valueSource: MTLBuffer,
                              bytesPerToken: Int) throws {
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
        let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keyFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
        let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keySecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueSecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
    }

    /// Gated-DeltaNet (linear attention) branch of one chunked-prefill layer.
    ///
    /// lint:allow-long one layer's linear-attention pipeline is a single
    /// ordered sequence -- in-projection, causal conv, QK norm, delta step,
    /// gated norm, out-projection -- sharing scratch buffers at every step.
    /// Splitting it would thread a dozen buffers through sub-functions to make
    /// a line count smaller while making the data flow harder to follow.
    private func encodeLinearAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int,
        snapshotGDNAfterFirstToken: Bool, useTwoRowProjection: Bool
    ) throws {
        // Gated-DeltaNet linear attention over the chunk: batched
        // projections, causal conv (+ tail carry), delta-rule
        // recurrence, gated norm, out_proj. No KV writes, no
        // attention, no blit.
        guard let gdn, let gdnState else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        // `LayerPrefillQKVViews` fills the linear_attn slots exactly
        // when `layerIsLinear(L)`, so these are provably non-nil; the
        // guard turns a future arch/view regression into a thrown
        // error instead of a force-unwrap trap.
        guard let linQKV = views.linQKV,
              let linZ = views.linZ,
              let linA = views.linA,
              let linB = views.linB,
              let linConv = views.linConv,
              let linALog = views.linALog,
              let linDtBias = views.linDtBias,
              let linNorm = views.linNorm,
              let linOut = views.linOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) is missing a required linear_attn tensor view")
        }
        let la = cfg.linearAttention
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: linQKV,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: la.qkvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.qkvDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linZ,
                             x: scratch.normed,
                             y: scratch.gdnZ,
                             rows: la.valueDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.valueDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linA,
                             x: scratch.normed,
                             y: scratch.gdnA,
                             rows: la.numVHeads,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.numVHeads,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linB,
                             x: scratch.normed,
                             y: scratch.gdnB,
                             rows: la.numVHeads,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.numVHeads,
                             useTwoRowProjection: useTwoRowProjection)
        let convW = linConv
        let tail = gdnState.convTailBuffer(layer: L)
        try gdn.encodeConvPrefill(commandBuffer: cb,
                              tail: tail,
                              qkvRows: scratch.q,
                              convWeight: convW.buffer,
                              convWeightOffset: Int(convW.offset),
                              out: scratch.gdnConvOut,
                              rows: t)
        if snapshotGDNAfterFirstToken {
            try gdn.encodeConvTailCheckpoint(
                commandBuffer: cb,
                tail: tail,
                qkvRows: scratch.q,
                checkpoint: gdnState.speculativeConvTailBuffer(layer: L))
        }
        try gdn.encodeConvTailUpdate(commandBuffer: cb,
                                 tail: tail,
                                 qkvRows: scratch.q,
                                 rows: t)
        try gdn.encodeQKNorm(commandBuffer: cb,
                         convOut: scratch.gdnConvOut,
                         rows: t)
        let aLog = linALog
        let dtBias = linDtBias
        try gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                   convOut: scratch.gdnConvOut,
                                   aProj: scratch.gdnA,
                                   bProj: scratch.gdnB,
                                   aLog: aLog.buffer,
                                   aLogOffset: Int(aLog.offset),
                                   dtBias: dtBias.buffer,
                                   dtBiasOffset: Int(dtBias.offset),
                                   state: gdnState.stateBuffer(layer: L),
                                   checkpointState: snapshotGDNAfterFirstToken
                                    ? gdnState.speculativeStateBuffer(layer: L) : nil,
                                   y: scratch.gdnY,
                                   rows: t)
        let gatedNormW = linNorm
        try gdn.encodeGatedNorm(commandBuffer: cb,
                            y: scratch.gdnY,
                            z: scratch.gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: scratch.attentionOutput,
                            rows: t)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .o,
                             weights: linOut,
                             x: scratch.attentionOutput,
                             y: scratch.h1,
                             rows: D,
                             columns: la.valueDim,
                             tokenCount: t,
                             xStrideElements: la.valueDim,
                             yStrideElements: D,
                             useTwoRowProjection: useTwoRowProjection)
    }

    /// Softmax-attention branch of one chunked-prefill layer.
    ///
    /// lint:allow-long same shape as the linear branch: QKV projection, RoPE,
    /// KV-cache write and attention are one ordered pipeline over shared
    /// scratch, and the intermediate buffers have no meaning outside it.
    private func encodeFullAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int, rmsEps eps: Float,
        useTwoRowProjection: Bool
    ) throws {
        let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: views.q!,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: qProjRows,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: qProjRows,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.k!,
                             x: scratch.normed,
                             y: scratch.kStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim,
                             useTwoRowProjection: useTwoRowProjection)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.v!,
                             x: scratch.normed,
                             y: scratch.vStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim,
                             useTwoRowProjection: useTwoRowProjection)

        // The attention input Q: the packed q_proj output is split
        // into per-head query/gate halves for gated architectures.
        let attnQ: MTLBuffer
        if cfg.attnOutputGate {
            try elementwise!.encodeSplitQGate(commandBuffer: cb,
                                          packed: scratch.q,
                                          q: scratch.attnQ,
                                          gate: scratch.attnGate,
                                          heads: cfg.numHeads,
                                          dim: headDim,
                                          rows: t)
            attnQ = scratch.attnQ
        } else {
            attnQ = scratch.q
        }

        if cfg.ropeNeoxSubdim {
            let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
            try prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                qWeight: views.qNorm!.buffer,
                qWeightOffset: Int(views.qNorm!.offset),
                kWeight: views.kNorm!.buffer,
                kWeightOffset: Int(views.kNorm!.offset),
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: Float(cfg.fullRopeTheta),
                rotaryDim: rotaryDim,
                eps: eps)
        } else {
            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            try prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: attnQ,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm!.buffer,
                                       qWeightOffset: Int(views.qNorm!.offset),
                                       kWeight: views.kNorm!.buffer,
                                       kWeightOffset: Int(views.kNorm!.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)
        }

        if let kv {
            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try copyPrefillKVToCache(commandBuffer: cb,
                                     kv: kv,
                                     layer: L,
                                     startPosition: startPosition,
                                     tokenCount: t,
                                     keySource: scratch.kStage,
                                     valueSource: scratch.vStage,
                                     bytesPerToken: bytes / t)
        }
        let params = PrefillAttentionParams(
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                kvValidCount: UInt32(startPosition + t),
                slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                kvTokenStrideElements: UInt32(kvDim),
                qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim),
                scale: Float(cfg.attentionScale))
        if let kv {
                let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                try prefillAttention.encodeCausal(commandBuffer: cb,
                                              q: attnQ,
                                              k: keyBuffer,
                                              v: valueBuffer,
                                              out: scratch.attentionOutput,
                                              params: params,
                                              kvRingCapacity: activeRingCapacity,
                                              path: prefillAttentionPath)
        } else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill attention requires FP16 KV")
        }
        if cfg.attnOutputGate {
            try elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                              out: scratch.attentionOutput,
                                              gate: scratch.attnGate,
                                              count: t * qDim)
        }
        try encodeAffineProjection(commandBuffer: cb,
                                 family: .o,
                                 weights: views.o!,
                                 x: scratch.attentionOutput,
                                 y: scratch.h1,
                                 rows: D,
                                 columns: qDim,
                                 tokenCount: t,
                                 xStrideElements: qDim,
                                 yStrideElements: D,
                                 useTwoRowProjection: useTwoRowProjection)
    }

    /// Resolve every layer's tensor views once, before the chunk loop.
    private func makeLayerPrefillViews() throws -> [LayerPrefillQKVViews] {
        try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: isLinear ? nil : try model.kProj(layer: L),
                v: isLinear ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: isLinear ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }
    }

    /// Final norm and lm_head for the last chunk, writing logits or a fused
    /// greedy token depending on the output mode.
    private func encodeFinalHead(
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        rmsEps eps: Float,
        outputMode: PrefillOutputMode
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let finalCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: finalCB,
                hidden: scratch.hidden,
                hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                normWeight: finalNorm.buffer,
                normOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            try prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                             hiddenBlock: scratch.hidden,
                                             row: t - 1,
                                             rowStrideElements: D,
                                             normWeight: finalNorm.buffer,
                                             normWeightOffset: Int(finalNorm.offset),
                                             weights: lm.buffer,
                                             weightsOffset: Int(lm.offset),
                                             scales: lm.buffer,
                                             scalesOffset: Int(lm.scaleOffset),
                                             biases: lm.buffer,
                                             biasesOffset: Int(lm.biasOffset),
                                             logits: logits,
                                             d: UInt32(D),
                                             vocab: UInt32(cfg.vocabSize),
                                             rmsEps: eps)
        }
        finalCB.commit()
        try waitForCompletion(finalCB)
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
    }
}
