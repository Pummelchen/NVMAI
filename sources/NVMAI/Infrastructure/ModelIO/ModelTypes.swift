import Foundation
import Metal

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape, and family-specific kernel behavior. Stored in
/// `manifest.json -> arch.family`; absent means the compatible Qwen3.5-MoE
/// 35B-A3B baseline used by Qwen 3.6 and Ornith 1.5.
public enum ModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    /// Qwen3.8-Flash-Next (`qwen4_exp` text config), 125B-A6B: 48 layers of
    /// (3x gated-DeltaNet -> 1x sparse-indexed attention), 512 experts top-10,
    /// hyper-connection residual streams, and hashed n-gram embeddings at
    /// layer 1. Text-only; the checkpoint's vision tower is never repacked.
    /// See docs/qwen38-flash-next-port.md for the verified design record.
    case qwen38flash = "qwen38flash"
    /// The Qwen3.8-Flash-Next draft head: one full-attention layer with its
    /// own 512 experts, hyper-connection gates and indexer, plus two fusion
    /// projections that combine the target's wide residual with the next
    /// token's embedding. Shares the target's embedding and head.
    case qwen38flashMTP = "qwen38flash_mtp"
}

/// Hyper-connection residual configuration (Qwen3.8-Flash-Next). The residual
/// stream is `count` parallel 2560-wide streams; each sublayer mixes them to
/// one block input through a low-rank gate and injects its output back into
/// every stream with learned per-stream weights. Zeroed for plain-residual
/// architectures.
public struct HyperConnectionConfig: Sendable, Equatable {
    public let count: Int
    public let lowRank: Int

    public init(count: Int, lowRank: Int) {
        self.count = count
        self.lowRank = lowRank
    }

    public static let none = HyperConnectionConfig(count: 0, lowRank: 0)
    public var enabled: Bool { count > 0 }
}

/// Qwen Sparse Attention indexer configuration. Each full-attention layer
/// scores mean-pooled key blocks with a small MQA head set and keeps the
/// `budget` highest-scoring visible tokens (plus the incomplete tail block).
/// Dense attention is exact whenever a query sees at most `budget` keys plus
/// the tail — the runtime's dense path is gated on that window. Zeroed for
/// dense-attention architectures.
public struct SparseIndexerConfig: Sendable, Equatable {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let budget: Int
    public let compressRatio: Int

    public init(numHeads: Int, numKVHeads: Int, headDim: Int,
                budget: Int, compressRatio: Int) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.budget = budget
        self.compressRatio = compressRatio
    }

    public static let none = SparseIndexerConfig(
        numHeads: 0, numKVHeads: 0, headDim: 0, budget: 0, compressRatio: 0)
    public var enabled: Bool { budget > 0 }
}

/// Hashed n-gram "per-layer embedding" configuration (Qwen3.8-Flash-Next).
/// At each layer in `layerIndices` (0-based), every token gathers
/// `(ngramSize - 1) * headsPerNgram` rows from a prime-partitioned hashed
/// table and injects them into the hyper streams through key/value
/// projections and a dilated depthwise conv. The multipliers, per-head vocab
/// sizes, and offsets ship as checkpoint tensors and are read, not
/// re-derived; `seed` is recorded for validation only. Zeroed when absent.
public struct PLEConfig: Sendable, Equatable {
    public let layerIndices: [Int]
    public let embedDim: Int
    public let convKernelSize: Int
    public let ngramSize: Int
    public let vocabSizeBase: Int
    public let headsPerNgram: Int
    public let vocabDivisor: Int
    public let seed: Int

    public init(layerIndices: [Int], embedDim: Int, convKernelSize: Int,
                ngramSize: Int, vocabSizeBase: Int, headsPerNgram: Int,
                vocabDivisor: Int, seed: Int) {
        self.layerIndices = layerIndices
        self.embedDim = embedDim
        self.convKernelSize = convKernelSize
        self.ngramSize = ngramSize
        self.vocabSizeBase = vocabSizeBase
        self.headsPerNgram = headsPerNgram
        self.vocabDivisor = vocabDivisor
        self.seed = seed
    }

    public static let none = PLEConfig(
        layerIndices: [], embedDim: 0, convKernelSize: 0, ngramSize: 0,
        vocabSizeBase: 0, headsPerNgram: 0, vocabDivisor: 0, seed: 0)
    public var enabled: Bool { !layerIndices.isEmpty }
    /// Lookups per token: (ngramSize - 1) n-gram orders x headsPerNgram.
    public var ngramHeads: Int { (ngramSize - 1) * headsPerNgram }
    /// Row width of the hashed table.
    public var headDim: Int { ngramHeads > 0 ? embedDim / ngramHeads : 0 }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    /// Nonlinearity applied to `z` before it scales the normalized delta
    /// readout. Qwen 3.6 (like Qwen3-Next) uses silu; Qwen3.8-Flash-Next uses
    /// sigmoid, which its config states as `output_gate_type`. Both are
    /// smooth and positive-ish, so picking the wrong one produces confident
    /// nonsense rather than an error -- it is declared per family, never
    /// defaulted from the other one.
    public enum OutputGate: String, Sendable, Equatable {
        case silu
        case sigmoid
    }

    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int
    public let outputGate: OutputGate

    public init(numKHeads: Int, numVHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convKernelSize: Int,
                outputGate: OutputGate = .silu) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
        self.outputGate = outputGate
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE baseline so earlier manifests validate unchanged.
    public let family: ModelFamily
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention (Qwen 3.6 uses 0.0625 = 256^-0.5).
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize). False for Qwen 3.6.
    public let embeddingScaledBySqrtHidden: Bool
    /// Router carries `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors. False (Qwen 3.6): plain quantized linear router with
    /// renormalized top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Dual-branch FFN sandwich: pre/post feedforward norms plus a per-layer
    /// residual scalar. False (Qwen 3.6) = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. True (Qwen/NeoX sub-dim): rotation confined to
    /// the first `rotaryDim` elements, pairing (i, rotaryDim/2 + i), frequency
    /// divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig
    public let hyperConnections: HyperConnectionConfig
    public let sparseIndexer: SparseIndexerConfig
    public let ple: PLEConfig
    /// Softmax-then-top-k with renormalization of the kept probabilities
    /// (`norm_topk_prob`). False for Qwen3.5-MoE; true for Qwen3.8-Flash-Next.
    public let routerNormTopK: Bool
    /// Affine quantization group size this architecture's checkpoints use.
    /// 64 for every Qwen3.5-MoE-era install; 32 for the available
    /// Qwen3.8-Flash-Next community quantizations. The manifest records the
    /// per-slot value; this is the architecture-level default for planning.
    public let quantGroupSize: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        family: ModelFamily = .qwen36,
        attnOutputGate: Bool = true,
        attentionScale: Double = 0.0625,
        embeddingScaledBySqrtHidden: Bool = false,
        routerScaled: Bool = false,
        ffnSandwichNorms: Bool = false,
        sharedExpertGated: Bool = true,
        ropeNeoxSubdim: Bool = true,
        linearAttention: LinearAttentionConfig = .none,
        hyperConnections: HyperConnectionConfig = .none,
        sparseIndexer: SparseIndexerConfig = .none,
        ple: PLEConfig = .none,
        routerNormTopK: Bool = false,
        quantGroupSize: Int = 64
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.family = family
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.linearAttention = linearAttention
        self.hyperConnections = hyperConnections
        self.sparseIndexer = sparseIndexer
        self.ple = ple
        self.routerNormTopK = routerNormTopK
        self.quantGroupSize = quantGroupSize
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
    public static let qwen36_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: .qwen36,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    /// Native one-layer Qwen3.5-MoE MTP draft for compatible Qwen/Ornith
    /// targets. Its 65,536-token KV cache is at most
    /// 128 MiB in FP16 and smaller with compressed storage; truncating draft
    /// context can only lower acceptance because every emitted token is still
    /// verified by the full target.
    public static let qwen36MTP = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 65_536,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 1,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [1],
        hiddenActivation: "silu",
        family: .qwen36MTP,
        attnOutputGate: true,
        attentionScale: 0.0625,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true)

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Qwen3.8-Flash-Next 125B-A6B (`qwen4_exp` text config), text-only.
    /// Geometry read from the official checkpoint's config.json and the
    /// transformers modeling source; see docs/qwen38-flash-next-port.md.
    /// The full-attention slots describe the QSA layers (every 4th layer);
    /// the sliding-window mirrors mask value 1 exactly as qwen36 does.
    public static let qwen38FlashNext = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 48,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen38LayerMask(),
        hiddenActivation: "silu",
        family: .qwen38flash,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 48,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4,
            outputGate: .sigmoid),
        hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
        sparseIndexer: SparseIndexerConfig(
            numHeads: 4, numKVHeads: 1, headDim: 128,
            budget: 2048, compressRatio: 4),
        ple: PLEConfig(
            layerIndices: [1],          // config ple_layer_ids [2] is 1-based
            embedDim: 2560,
            convKernelSize: 4,
            ngramSize: 3,
            vocabSizeBase: 20_000_000,
            headsPerNgram: 8,
            vocabDivisor: 128,
            seed: 1234),
        routerNormTopK: true,
        // Verified against the pinned artifact's `quantization.group_size`
        // (benchmark/nvmai_quant_fidelity.py decodes it correctly at 64).
        // The design record had assumed 32 from the Vontra checkpoints.
        quantGroupSize: 64)

    /// The Qwen3.8-Flash-Next MTP draft. One full-attention layer, no
    /// DeltaNet, no n-gram block; everything else is the target's geometry
    /// because the draft has to speak the same residual.
    public static let qwen38FlashNextMTP = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 1,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [1],
        hiddenActivation: "silu",
        family: .qwen38flashMTP,
        attnOutputGate: true,
        attentionScale: 0.0625,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: .none,
        hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
        sparseIndexer: SparseIndexerConfig(
            numHeads: 4, numKVHeads: 1, headDim: 128,
            budget: 2048, compressRatio: 4),
        ple: .none,
        routerNormTopK: true,
        quantGroupSize: 64)

    private static func qwen38LayerMask() -> [UInt8] {
        // Same kinds as qwen36: 2 = gated-DeltaNet linear, 1 = full (QSA)
        // attention on every 4th layer ((i + 1) % 4 == 0), 48 layers.
        var mask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Registry keyed by `manifest.arch.family` for auto-detection at load.
    public static let knownArchitectures: [ModelFamily: ArchConfig] = [
        .qwen36: .qwen36_35B_A3B,
        .qwen36MTP: .qwen36MTP,
        .qwen38flash: .qwen38FlashNext,
        .qwen38flashMTP: .qwen38FlashNextMTP,
    ]

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
        shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        shapes.append((m: intermediateSize, n: hiddenSize))
        shapes.append((m: hiddenSize, n: intermediateSize))
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = [(m: numExperts, n: hiddenSize)]
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case unsupportedArchitecture(detail: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case expertCacheUnplaceable(detail: String)
    /// A Metal command buffer reported `.error`; the GPU work it carried
    /// (decode layer, head, or routed-expert pass) did not complete.
    case commandBufferFailed(detail: String)
    /// A runtime invariant the code believes is impossible was violated
    /// (arch/kernel mismatch, pipeline state corruption). Thrown instead of
    /// trapping so generation fails loudly without crashing the process.
    case internalInconsistency(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .unsupportedArchitecture(let detail):
            return "unsupported architecture: \(detail)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .expertCacheUnplaceable(let detail):
            return "expert cache cannot place requested experts: \(detail)"
        case .commandBufferFailed(let detail):
            return "Metal command buffer failed: \(detail)"
        case .internalInconsistency(let detail):
            return "internal inconsistency: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
/// unchecked-invariant: every stored property is a `let`; only MTLBuffer's
/// lack of Sendable forces @unchecked. The view describes where a tensor sits
/// in the resident buffer and never mutates it.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
