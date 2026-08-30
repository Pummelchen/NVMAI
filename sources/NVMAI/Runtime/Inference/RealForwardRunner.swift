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
    var skipUntilPosition: Int = -1
    internal(set) var recentSlowCallNanos: UInt64 = 0

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

/// Compatible Qwen3.5-MoE real-forward decode pass.
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

/// unchecked-invariant: exclusively owned by one caller for its lifetime and
/// never shared. In the server it is a `private let` on the `ServerModelSession`
/// actor, so every entry point is already actor-isolated; the CLI and the
/// decode service each drive one runner from a single task. Its ~19 mutable
/// properties are decode cursors and scratch handles with no internal locking,
/// so two concurrent callers would corrupt them -- the ownership is the whole
/// safety argument, not an implementation detail.
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Qwen3.5-MoE [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    let model: Model
    let ctx: MetalContext
    let kv: KVCacheManager?
    let cfg: ArchConfig

    // Kernels
    let embedInt4: EmbedLookupInt4
    let affineEmbed: AffineQuantEmbeddingLookup?
    let rms: RMSNorm
    let int4: DequantInt4GEMV
    let affine: AffineQuantGEMV?
    /// Vocabulary head GEMV, keyed off `lmHeadWeightBits` rather than the
    /// attention slot. Nil when the head is 4-bit.
    let affineHead: AffineQuantGEMV?
    let attention: Attention
    let kvQuantizer: KVCacheQuantizer?
    let shared: SharedExpertRuntime
    let moe: MoE
    let fusionHead: LMHeadChainInt4
    let fusedQKVGEMV: FusedQKVGEMV
    let fusedQKVEpilogue: FusedQKVEpilogue

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    let elementwise: Elementwise?
    /// The Gated Residual, for families that carry one. Owns its own scratch,
    /// so a family without hyper-connections allocates nothing.
    /// Set by `NVMAI_ACT_DUMP`; nil disables every dump call site.
    let activationDumpDirectory: URL?
    let hyperConnection: HyperConnection?
    /// The n-gram (PLE) block, its row addressing, and the table it reads.
    /// All three or none: a family without PLE layers leaves them nil.
    /// The sparse-attention indexer, for families that have one. Nil leaves
    /// the runtime on dense attention, which is exact only inside
    /// `QSAExactness`'s window.
    let qsaIndexer: QSAIndexer?
    let pleBlock: PLEBlock?
    let pleHash: PLEHash?
    let ngramTable: NgramTableReader?
    /// The current token and its predecessors, nearest first, as `PLEHash`
    /// wants them. Only `ngramSize` entries are ever consulted.
    var pleContext: [Int32] = []
    let gdn: GDN?
    let gdnState: GDNStateManager?
    let rope: RoPE?
    let int8ScalarGate: DequantInt8GEMV?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    let prefillEmbed: PrefillEmbedLookupInt4
    let prefillRMS: PrefillRMSNorm
    let prefillQMM: PrefillInt4QMM
    let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    let prefillQKVEpilogue: PrefillQKVEpilogue
    let prefillAttention: PrefillAttention
    let prefillRouter: PrefillRouter
    let prefillSharedExpert: PrefillSharedExpert
    let prefillGroupedMoE: PrefillGroupedRoutedMoE
    let prefillMoE: PrefillMoE
    let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    let hidden: MTLBuffer        // [D] FP16
    let normed: MTLBuffer        // [D] FP16
    let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    let oOut: MTLBuffer          // [D] FP16
    let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    let denseScratchGate: MTLBuffer // [F=2112] FP16
    let denseScratchUp: MTLBuffer   // [F=2112] FP16
    let denseScratchAct: MTLBuffer  // [F=2112] FP16
    let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    let outIndices: MTLBuffer    // [topK] UInt32
    let outWeights: MTLBuffer    // [topK] FP16
    /// Trace-only next-layer router result. It is never read by inference.
    let prefetchPredictionIndices: MTLBuffer
    let prefetchPredictionWeights: MTLBuffer
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    let moeActs: MTLBuffer       // [topK * FmoE] FP16
    /// Width-2 MTP verify scratch (B2 pair schedule): per-row activation and
    /// output buffers plus two persistent routed argument buffers, created on
    /// first verify. Per-row buffers are deliberately *separate allocations*,
    /// not offsets into one: Metal hazard tracking is whole-buffer, so a
    /// shared acts buffer would falsely serialize row 1's phase 1 behind
    /// row 0's phase 2 and cost real GPU concurrency. The rewrite-per-layer
    /// hazard on the argument buffers is safe because the pair schedule waits
    /// on each layer's routed command before the next layer re-encodes them.
    var verifyPairActs: [MTLBuffer] = []
    var verifyPairY: [MTLBuffer] = []
    var verifyPairArgBuffers: [MTLBuffer] = []
    let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    let residencyHitCount: MTLBuffer
    let residencyHitPositions: MTLBuffer
    let residencyMissCount: MTLBuffer
    let residencyMissPositions: MTLBuffer
    let residencyMissExperts: MTLBuffer
    let residencyResolvedSlots: MTLBuffer
    let residencyResolvedGenerations: MTLBuffer
    let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    let verificationHidden: MTLBuffer // [2, D] FP16 shared readback
    let verificationLogits: MTLBuffer // [2, vocab] FP16 shared readback
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    let gdnZ: MTLBuffer?             // [valueDim]
    let gdnA: MTLBuffer?             // [numVHeads]
    let gdnB: MTLBuffer?             // [numVHeads]
    let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    let onesPerExpertScale: MTLBuffer?
    var prefillChunkState = PrefillChunkCommitState()
    var prefillScratch: PrefillChunkScratchBuffers?
    static let mtpChunkCapacity = 32
    let mtpTokenBlock: MTLBuffer?
    let mtpEmbeddingBlock: MTLBuffer?
    let mtpNormalizedEmbeddingBlock: MTLBuffer?
    let mtpNormalizedHiddenBlock: MTLBuffer?
    let mtpConcatBlock: MTLBuffer?
    let mtpProjectedBlock: MTLBuffer?
    let mtpTargetHiddenBlock: MTLBuffer?
    var mtpPrefillReadback: MTLBuffer?
    /// Reusable UInt32 token-ID buffer for chunked prefill (R23): sized to the
    /// largest chunk seen so far and grown on demand, so the prefill hot path
    /// never allocates an MTLBuffer per chunk.
    var prefillTokenBuffer: MTLBuffer?

    /// Host scratch reused across prefill chunks (R38) and decode layers (R16).
    /// The runner is single-flight per generation (guarded by
    /// `prefillChunkState` and the callers' serial decode loop), so these
    /// never alias concurrent work.
    var routeIDScratch: [UInt32] = []
    var routeWeightScratch: [Float16] = []
    var decodeExpertsScratch: [Int] = []
    var decodeHitSlotsScratch: [UInt32] = []
    var decodeMissSlotsScratch: [UInt32] = []
    var decodeHitSplitRoutedBufsScratch: [MTLBuffer] = []
    var decodeHitSplitRoutedOffsetsScratch: [Int] = []
    var decodeRoutedBufsScratch: [MTLBuffer] = []
    var decodeRoutedOffsetsScratch: [Int] = []

    static let rdadviseBoundedMissCap = 12
    static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    static let rdadviseAdaptiveMissCap = 12
    static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    let effectiveScaleBuffers: [MTLBuffer]
    let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    let useFusedGreedyHead: Bool
    let prefillAttentionPath: RuntimePrefillAttentionPath
    let decodeExpertExecution: RuntimeDecodeExpertExecution
    let expertIOSynchronization: RuntimeExpertIOSynchronization
    let expertIOSubmission: RuntimeExpertIOSubmission
    let expertIOBackend: ExpertIOBackend
    let predictivePrefetch: ExpertPrefetchRing?
    let anePrefill: ANEPrefillAttention?
    let predictivePrefetchTopM: Int
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    var rdadviseSkipUntilPosition: Int = -1
    var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    var rdadviseAdaptivePosition: Int = -1
    var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production,
                enableSpeculativeGDN: Bool = false) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        try runtimeConfiguration.validate(maxContext: maxContext)
        let yarnParameters: YaRNRoPEParameters?
        if runtimeConfiguration.ropeScalingMode == .yarn {
            guard model.config.ropeNeoxSubdim else {
                throw RuntimeConfigurationError.yaRNUnsupportedArchitecture
            }
            yarnParameters = YaRNRoPEParameters(
                headDim: model.config.fullHeadDim,
                partialRotaryFactor: model.config.partialRotaryFactor,
                theta: model.config.fullRopeTheta,
                targetContextTokens: runtimeConfiguration.yarnContextTokens)
        } else {
            yarnParameters = nil
        }
        // The fused greedy head folds a plain RMSNorm into the vocabulary
        // GEMV. A hyper-connection model does not end in an RMSNorm: it ends
        // in the gated mixer that collapses the residual streams, so the
        // fused path would normalize the wide residual and read stream 0 as
        // if it were the whole hidden state. Correct output beats one fused
        // dispatch; the family takes the two-step head.
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
            && !cfg.hyperConnections.enabled
            && model.lmHeadWeightBits == 4
            && model.attentionWeightBits == 4
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        self.decodeExpertExecution = runtimeConfiguration.decodeExpertExecution
        self.expertIOSynchronization = runtimeConfiguration.expertIOSynchronization
        self.expertIOSubmission = runtimeConfiguration.expertIOSubmission
        self.expertIOBackend = try ExpertIOBackend.environmentValue()
        let rawPrefetchEnabled = ProcessInfo.processInfo.environment[
            "NVMAI_PREDICTIVE_PREFETCH"] == "1"
        let rawPrefetchTopM = Int(ProcessInfo.processInfo.environment[
            "NVMAI_PREFETCH_TOP_M"] ?? "4") ?? 4
        guard (1...cfg.topKExperts).contains(rawPrefetchTopM) else {
            throw ModelError.internalInconsistency(
                detail: "NVMAI_PREFETCH_TOP_M must be 1...\(cfg.topKExperts)")
        }
        self.predictivePrefetchTopM = rawPrefetchTopM
        self.predictivePrefetch = rawPrefetchEnabled
            ? try ExpertPrefetchRing(
                device: context.device,
                expertStride: model.routedExpertByteStride(layer: 0),
                slotCount: rawPrefetchTopM)
            : nil
        // Track A: the ANE prefill sidecar, opt-in. Only the qwen36 target
        // family qualifies (the one-layer MTP draft has no exported sidecar
        // and must stay silently on the GPU); with the switch on and the
        // sidecar missing, construction fails closed with the export command.
        self.anePrefill = try RuntimePrefillANE.environmentValue() == .on
                && model.config.family == .qwen36
            ? try ANEPrefillAttention(
                modelDirectory: model.directoryURL,
                device: context.device,
                hiddenSize: model.config.hiddenSize,
                kvDim: model.config.numFullKVHeads * model.config.fullHeadDim,
                weightsSha256: model.weightsDigestFromManifest)
            : nil
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
                                     precision: runtimeConfiguration.kvCachePrecision,
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
        // The vocabulary head carries its own bit width. It matched the
        // attention slot in every earlier family, so the head simply reused
        // the attention GEMV -- which reads an 8-bit head as packed 4-bit the
        // moment a model quantizes the two differently, as this one does
        // (4-bit attention, 8-bit embedding and head). The result is a full
        // logit vector of confident nonsense, so the width is selected here
        // rather than inherited.
        self.affineHead = model.lmHeadWeightBits == 4 ? nil
            : try AffineQuantGEMV(context: context,
                                  weightBits: model.lmHeadWeightBits)
        self.attention = try Attention(context: context)
        self.kvQuantizer = runtimeConfiguration.kvCachePrecision.isQuantized
            ? try KVCacheQuantizer(context: context) : nil
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu)
        self.moe       = try MoE(context: context,
                                 siluActivation: silu,
                                 routedWeightBits: model.routedExpertWeightBits,
                                 routerWeightBits: model.routerWeightBits,
                                 eventGatedIO: expertIOSynchronization == .event,
                                 specializedD: UInt32(cfg.hiddenSize),
                                 specializedF: UInt32(cfg.moeIntermediateSize),
                                 specializedNumExperts: UInt32(cfg.numExperts),
                                 topKExperts: cfg.topKExperts)
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
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context,
                                                         yarn: yarnParameters)
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
        self.activationDumpDirectory = ProcessInfo.processInfo
            .environment["NVMAI_ACT_DUMP"].map { URL(fileURLWithPath: $0) }
        // Both gated blocks keep a whole prefill chunk resident: the write
        // gate consumes what its matching read produced, and the block runs
        // in between, so the rows cannot be streamed one at a time.
        let gateRows = max(1, runtimeConfiguration.prefillChunkTokens)
        self.hyperConnection = cfg.hyperConnections.enabled
            ? try HyperConnection(context: context,
                                  dim: cfg.hiddenSize,
                                  streams: cfg.hyperConnections.count,
                                  lowRank: cfg.hyperConnections.lowRank,
                                  maxRows: gateRows)
            : nil
        self.qsaIndexer = cfg.sparseIndexer.enabled
            ? try QSAIndexer(context: context,
                             config: cfg.sparseIndexer,
                             budget: Self.qsaBudget(cfg.sparseIndexer),
                             ropeTheta: Float(cfg.fullRopeTheta),
                             capacity: maxContext)
            : nil
        if cfg.ple.enabled {
            let constants = try PLEConstants.load(
                directoryURL: model.directoryURL)
            self.pleHash = constants.makeHash()
            self.ngramTable = try NgramTableReader(
                path: model.directoryURL.appendingPathComponent(
                    Qwen38FlashTensors.ngramTableFile).path,
                rowDim: constants.pleHeadDim,
                rowCount: constants.tableRowCount)
            self.pleBlock = try PLEBlock(
                context: context,
                dim: cfg.hiddenSize,
                streams: cfg.hyperConnections.count,
                embedDim: cfg.ple.embedDim,
                kernelSize: cfg.ple.convKernelSize,
                // The dilation is the n-gram size, not a constant of its own.
                dilation: cfg.ple.ngramSize,
                maxRows: gateRows)
        } else {
            self.pleHash = nil
            self.ngramTable = nil
            self.pleBlock = nil
        }
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
        self.rope = cfg.ropeNeoxSubdim
            ? try RoPE(context: context, yarn: yarnParameters) : nil
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
        // The residual is the one scratch buffer whose width is not D. A
        // hyper-connection family carries `hc_count` parallel streams of D and
        // reads a single D-wide vector out of them per sublayer, so only this
        // allocation widens -- every downstream buffer stays D. Families
        // without them get exactly D, as before.
        let residualElements = cfg.hyperConnections.enabled
            ? D * cfg.hyperConnections.count
            : D
        self.hidden        = try buf(residualElements, label: "decode.hidden")
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
        self.prefetchPredictionIndices = try buf(
            cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.prefetchPredictionIndices")
        self.prefetchPredictionWeights = try buf(
            cfg.topKExperts, label: "decode.prefetchPredictionWeights")
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize, label: "decode.moeActs")
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeHitActiveSlots")
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size, label: "decode.moeMissActiveSlots")
        self.residencyHitCount = try buf(1, MemoryLayout<UInt32>.size,
                                         label: "decode.residencyHitCount")
        self.residencyHitPositions = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size,
                                             label: "decode.residencyHitPositions")
        self.residencyMissCount = try buf(1, MemoryLayout<UInt32>.size,
                                          label: "decode.residencyMissCount")
        self.residencyMissPositions = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size,
                                              label: "decode.residencyMissPositions")
        self.residencyMissExperts = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size,
                                            label: "decode.residencyMissExperts")
        self.residencyResolvedSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size,
                                              label: "decode.residencyResolvedSlots")
        self.residencyResolvedGenerations = try buf(cfg.topKExperts,
                                                    MemoryLayout<UInt64>.size,
                                                    label: "decode.residencyResolvedGenerations")
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

    /// The window in which dense attention is exact for this model, or nil
    /// for a family without sparse attention.
    var qsaExactness: QSAExactness? {
        guard cfg.sparseIndexer.enabled else { return nil }
        return QSAExactness(budget: Self.qsaBudget(cfg.sparseIndexer),
                            compressRatio: cfg.sparseIndexer.compressRatio)
    }

    /// The indexer's key budget, with `NVMAI_QSA_BUDGET` able to lower it.
    ///
    /// Verification knob, not a tuning one. At the shipped budget the sparse
    /// path only engages past 2,051 tokens, which makes every check of it a
    /// multi-thousand-token run; lowering the budget moves the boundary down
    /// so the same code runs against a reference at a length that can be
    /// diffed in seconds. It only ever lowers.
    static func qsaBudget(_ config: SparseIndexerConfig) -> Int {
        guard let raw = ProcessInfo.processInfo.environment["NVMAI_QSA_BUDGET"],
              let value = Int(raw), value > 0 else { return config.budget }
        return min(value, config.budget)
    }

    /// Refuses a position the sparse-attention path cannot serve faithfully.
    /// See `QSAIndexerRequired` for why this is an error rather than a note.
    func requireQSAExact(visibleKeys: Int) throws {
        guard qsaIndexer == nil, let exactness = qsaExactness,
              !exactness.isDenseExact(visibleKeys: visibleKeys) else { return }
        throw QSAIndexerRequired(visibleKeys: visibleKeys,
                                 exactWindow: exactness.maximumExactVisibleKeys)
    }

    /// The same gate for prefill, which the indexer does not lift.
    ///
    /// Decode selects keys per query and can run past the window; a prefill
    /// chunk still attends densely, because masking the chunked attention is
    /// a separate kernel from the decode one. So a long *prompt* is still
    /// refused while a long *generation* from a short prompt is not.
    func requireQSADensePrefill(visibleKeys: Int) throws {
        guard let exactness = qsaExactness,
              !exactness.isDenseExact(visibleKeys: visibleKeys) else { return }
        throw QSAIndexerRequired(visibleKeys: visibleKeys,
                                 exactWindow: exactness.maximumExactVisibleKeys)
    }

    /// Forces the one-token-at-a-time prefill for hyper-connection families
    /// (`NVMAI_SEQUENTIAL_HC_PREFILL=1`). It is the reference the batched
    /// path is checked against, and far too slow to ship.
    static let sequentialHyperConnectionPrefill =
        ProcessInfo.processInfo.environment["NVMAI_SEQUENTIAL_HC_PREFILL"] == "1"

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetPLEState()
        qsaIndexer?.reset()
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

    func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public internal(set) var totalIoNanos: UInt64 = 0
    public internal(set) var totalCb1Nanos: UInt64 = 0
    public internal(set) var totalCb2Nanos: UInt64 = 0
    public internal(set) var totalHeadNanos: UInt64 = 0
    public internal(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (NVMAI_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + rdadvise + io + cb2.
    public internal(set) var totalWaitNanos: UInt64 = 0
    public internal(set) var totalBodyNanos: UInt64 = 0
    public internal(set) var totalMissIoNanos: UInt64 = 0
    public internal(set) var totalExposedIoNanos: UInt64 = 0
    public internal(set) var totalHitFixupLayers: UInt64 = 0
    public internal(set) var totalRouterReadbackNanos: UInt64 = 0
    public internal(set) var totalCachePlanNanos: UInt64 = 0
    public internal(set) var totalIOQueueNanos: UInt64 = 0
    public internal(set) var totalIOCompletionToFixupSubmitNanos: UInt64 = 0
    public internal(set) var totalExpertIOHostWaits: UInt64 = 0
    public internal(set) var totalExpertIOHostWaitsAvoided: UInt64 = 0
    public internal(set) var totalGPUClassifiedHits: UInt64 = 0
    public internal(set) var totalGPUClassifiedMisses: UInt64 = 0
    public internal(set) var totalGPUResidencyAllHitLayers: UInt64 = 0
    public internal(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public internal(set) var totalRDAdviseNanos: UInt64 = 0
    public internal(set) var totalRDAdviseCalls: UInt64 = 0
    public internal(set) var totalRDAdviseBytes: UInt64 = 0
    public internal(set) var totalRDAdviseFailures: UInt64 = 0
    public internal(set) var totalRDAdviseSkipped: UInt64 = 0

    public func expertStreamingStatistics() -> ExpertStreamingStatistics {
        model.routedExpertStatistics()
    }


    // MARK: - Per-command-buffer GPU timing (NVMAI_KERNEL_STATS)

    /// One command buffer's GPU span for a named kernel role. The decode path
    /// is synchronous (commit + wait), so `gpuStartTime`/`gpuEndTime` are
    /// valid right after completion and cost nothing to read.
    struct KernelGPUTiming {
        let role: String
        let start: TimeInterval
        let end: TimeInterval
    }
    var kernelGPUTimings: [KernelGPUTiming] = []
    let kernelGPUTimingsEnabled =
        ProcessInfo.processInfo.environment["NVMAI_KERNEL_STATS"] != nil
    let runnerStatsEnabled =
        ProcessInfo.processInfo.environment["NVMAI_RUNNER_STATS"] != nil

    /// Open file descriptor for NVMAI_ROUTE_TRACE, or -1. Opened once and
    /// never closed: the runner lives as long as the process, and a decode
    /// loop is the wrong place to manage a diagnostic file's lifetime.
    let routeTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["NVMAI_ROUTE_TRACE"],
              !path.isEmpty else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    }()

    /// JSONL trace for the v4.3 predictive-prefetch qualification probe.
    /// It deliberately records only exact routing and authoritative cache
    /// residency before planning; enabling it cannot submit I/O or alter cache
    /// decisions. Kept separate from NVMAI_ROUTE_TRACE for compatibility.
    let prefetchTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["NVMAI_PREFETCH_TRACE"],
              !path.isEmpty else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    }()

    var nextLayerPredictionEnabled: Bool {
        prefetchTraceFD >= 0 || predictivePrefetch != nil
    }






    // MARK: - Routing trace (NVMAI_ROUTE_TRACE)








    @discardableResult










    nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
    }


    // MARK: - Chunked prefill helpers













    // MARK: - Decode routed-expert helpers

    /// A routed-expert command whose completion is deferred to the next layer.
    struct PendingRoutedCommand {
        let cb: MTLCommandBuffer
        let sharedCB: MTLCommandBuffer?
        let phase1HitCB: MTLCommandBuffer?
        let expertLease: RoutedExpertLease?
        let storageOperation: RoutedExpertLoadOperation?
        let overlapCompletionClock: CommandCompletionClock?
        let expectedOverlapCompletions: Int
        let kernelRole: String
        let encodeAndCommitNanos: UInt64
    }

    /// Diagnostic-only completion clock used to measure the I/O tail left
    /// after already-runnable GPU work. It is allocated only with
    /// NVMAI_RUNNER_STATS, never in the production hot path.
    /// unchecked-invariant: completion timestamps are mutated and read only
    /// while holding `lock`.
    final class CommandCompletionClock: @unchecked Sendable {
        let lock = NSLock()
        var completionCount = 0
        var latestCompletion: UInt64 = 0

        func track(_ commandBuffer: MTLCommandBuffer) {
            commandBuffer.addCompletedHandler { [self] _ in
                lock.lock()
                completionCount += 1
                latestCompletion = max(
                    latestCompletion,
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
                lock.unlock()
            }
        }

        func latest(expected: Int) -> UInt64? {
            lock.lock()
            defer { lock.unlock() }
            guard completionCount == expected else { return nil }
            return latestCompletion
        }
    }





}
