import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`.
/// Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    case qwen38flash = "qwen38flash"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE text architecture used by Qwen 3.6 and Ornith 1.5.
    let family: RepackModelFamily
    let attnOutputGate: Bool
    let attentionScale: Double
    let embeddingScaledBySqrtHidden: Bool
    let routerScaled: Bool
    let ffnSandwichNorms: Bool
    let sharedExpertGated: Bool
    let ropeNeoxSubdim: Bool
    let linearNumKHeads: Int
    let linearNumVHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelSize: Int

    // Qwen3.8-Flash-Next extensions. Zeroed for architectures that do not
    // have them, which keeps the qwen36 contract and its cross-check exact.
    let hcCount: Int
    let hcLowRank: Int
    let indexerNumHeads: Int
    let indexerNumKVHeads: Int
    let indexerHeadDim: Int
    let indexerBudget: Int
    let indexerCompressRatio: Int
    /// 0-based layer indices carrying the PLE block (config is 1-based).
    let pleLayerIndices: [Int]
    let pleEmbedDim: Int
    let pleConvKernelSize: Int
    let pleNgramSize: Int
    let pleVocabSizeBase: Int
    let pleHeadsPerNgram: Int
    let pleVocabDivisor: Int
    /// Router renormalizes the top-k probabilities (`norm_topk_prob`).
    let routerNormTopK: Bool
    /// Affine quantization group size of the source checkpoint.
    let quantGroupSize: Int

    init(hiddenSize: Int, intermediateSize: Int, moeIntermediateSize: Int,
         numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
         headDim: Int, fullHeadDim: Int, vocabSize: Int, slidingWindow: Int,
         finalLogitSoftcap: Double, ropeTheta: Double, fullRopeTheta: Double,
         partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
         topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
         fullAttentionLayerMask: [UInt8], hiddenActivation: String,
         family: RepackModelFamily, attnOutputGate: Bool,
         attentionScale: Double, embeddingScaledBySqrtHidden: Bool,
         routerScaled: Bool, ffnSandwichNorms: Bool, sharedExpertGated: Bool,
         ropeNeoxSubdim: Bool, linearNumKHeads: Int, linearNumVHeads: Int,
         linearKeyHeadDim: Int, linearValueHeadDim: Int,
         linearConvKernelSize: Int,
         hcCount: Int = 0, hcLowRank: Int = 0,
         indexerNumHeads: Int = 0, indexerNumKVHeads: Int = 0,
         indexerHeadDim: Int = 0, indexerBudget: Int = 0,
         indexerCompressRatio: Int = 0,
         pleLayerIndices: [Int] = [], pleEmbedDim: Int = 0,
         pleConvKernelSize: Int = 0, pleNgramSize: Int = 0,
         pleVocabSizeBase: Int = 0, pleHeadsPerNgram: Int = 0,
         pleVocabDivisor: Int = 0,
         routerNormTopK: Bool = false, quantGroupSize: Int = 64) {
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
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
        self.hcCount = hcCount
        self.hcLowRank = hcLowRank
        self.indexerNumHeads = indexerNumHeads
        self.indexerNumKVHeads = indexerNumKVHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.pleLayerIndices = pleLayerIndices
        self.pleEmbedDim = pleEmbedDim
        self.pleConvKernelSize = pleConvKernelSize
        self.pleNgramSize = pleNgramSize
        self.pleVocabSizeBase = pleVocabSizeBase
        self.pleHeadsPerNgram = pleHeadsPerNgram
        self.pleVocabDivisor = pleVocabDivisor
        self.routerNormTopK = routerNormTopK
        self.quantGroupSize = quantGroupSize
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if (root["model_type"] as? String) == "qwen3_5_mtp" {
            return try loadQwen36MTP(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen35MoE(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen4_exp" {
            return try loadQwen4Exp(configPath: configPath, tc: tc, root: root)
        }
        throw RepackError.configJsonInvalid(
            path: configPath,
            detail: "unsupported model_type (expected qwen3_5_moe, "
                + "qwen3_5_mtp or qwen4_exp)")
    }

    // MARK: - Qwen3.5-MoE text (`model_type == "qwen3_5_moe"`)

    private static func loadQwen35MoE(configPath: String,
                                     tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"))
        try crossCheckProductionQwen35MoE(arch, configPath: configPath)
        return arch
    }

    /// Qwen3.6 MTP is a single full-attention decoder layer. It intentionally
    /// carries neither an embedding table nor an LM head: both are shared from
    /// the verified target model at runtime. Treating it as a distinct family
    /// keeps a draft sidecar from ever being accepted as a standalone target.
    private static func loadQwen36MTP(configPath: String,
                                      tc: [String: Any]) throws -> ArchInfo {
        var base = try loadQwen35MoE(configPath: configPath, tc: tc)
        guard let count = (tc["mtp_num_hidden_layers"] as? Int)
            ?? (tc["mtp_num_hidden_layers"] as? NSNumber)?.intValue,
              count == 1 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "Qwen3.6 MTP requires mtp_num_hidden_layers == 1")
        }
        guard (tc["mtp_use_dedicated_embeddings"] as? Bool) == false else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "Qwen3.6 MTP must reuse the target embedding and head")
        }
        // MTP contract (mirrors the runtime's `qwen36MTP` arch config):
        // `numExperts` and `ropeNeoxSubdim` are deliberately kept from the
        // target baseline. The draft layer shares the target's router shape
        // (numExperts 256 drives the sidecar's per-expert layout), and the
        // MTP layer applies the same rotary embedding variant as the target,
        // so ropeNeoxSubdim stays true. The linear-attention parameters are
        // zeroed because the MTP layer is pure full-attention and carries no
        // DeltaNet bundle. numLayers collapses to 1 and the MTP arch reports
        // no embedding/head of its own (tieWordEmbeddings false).
        base = ArchInfo(
            hiddenSize: base.hiddenSize,
            intermediateSize: base.intermediateSize,
            moeIntermediateSize: base.moeIntermediateSize,
            numHeads: base.numHeads,
            numKVHeads: base.numKVHeads,
            numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim,
            fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize,
            slidingWindow: 65_536,
            finalLogitSoftcap: base.finalLogitSoftcap,
            ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: 1,
            numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [1],
            hiddenActivation: base.hiddenActivation,
            family: .qwen36MTP,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
        return base
    }

    // MARK: - Qwen3.8-Flash-Next text (`model_type == "qwen4_exp"`)

    /// The multimodal `Qwen4ExpForConditionalGeneration` checkpoint, read as
    /// text-only: the vision tower is never repacked, and with equal text
    /// positions the interleaved mrope collapses exactly onto the existing
    /// NeoX-subdim rotary. Every field below is read from the config; nothing
    /// is inferred. See docs/qwen38-flash-next-port.md.
    private static func loadQwen4Exp(configPath: String,
                                     tc: [String: Any],
                                     root: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func iOpt(_ k: String, _ fallback: Int) -> Int {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue ?? fallback
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue
            ?? (tc["partial_rotary_factor"] as? Double) else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing partial_rotary_factor")
        }
        // `ple_layer_ids` is 1-based in the config; the runtime indexes layers
        // from 0. Converting here keeps the off-by-one in one place.
        let pleIDs = ((tc["ple_layer_ids"] as? [Int]) ?? []).map { $0 - 1 }
        for id in pleIDs where id < 0 || id >= mask.count {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "ple_layer_ids entry out of range after 1-based "
                    + "conversion: \(id + 1)")
        }
        // The quantized checkpoint declares its own affine group size; NVMAI
        // must repack at whatever the source used, never at a default.
        let quant = (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any]) ?? [:]
        let group = (quant["group_size"] as? Int)
            ?? (quant["group_size"] as? NSNumber)?.intValue ?? 64
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .qwen38flash,
            // `output_gate_type: sigmoid` is this family's spelling of the
            // attention output gate qwen36 declares as `attn_output_gate`.
            attnOutputGate: (tc["output_gate_type"] as? String) == "sigmoid",
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"),
            hcCount: try i("hc_count"),
            hcLowRank: try i("hc_lowrank"),
            indexerNumHeads: try i("indexer_n_heads"),
            indexerNumKVHeads: try i("indexer_kv_heads"),
            indexerHeadDim: try i("indexer_head_dim"),
            indexerBudget: try i("indexer_budget"),
            indexerCompressRatio: try i("indexer_compress_ratio"),
            pleLayerIndices: pleIDs,
            pleEmbedDim: iOpt("ple_embed_dim", try i("hidden_size")),
            pleConvKernelSize: try i("ple_conv_kernel_size"),
            pleNgramSize: try i("ngram_size"),
            pleVocabSizeBase: try i("ngram_vocab_size_base"),
            pleHeadsPerNgram: try i("heads_per_ngram"),
            pleVocabDivisor: iOpt("make_ngram_vocab_size_divisible_by", 128),
            routerNormTopK: (tc["norm_topk_prob"] as? Bool) ?? true,
            quantGroupSize: group)
        try crossCheckQwen38FlashNext(arch, configPath: configPath)
        return arch
    }

    /// Qwen3.8-Flash-Next 125B-A6B contract (mirrors the runtime's
    /// `ArchConfig.qwen38FlashNext`). A config claiming the production shape
    /// must agree on the load-bearing geometry; anything else is a different
    /// model wearing the same `model_type`.
    private static func crossCheckQwen38FlashNext(_ a: ArchInfo,
                                                  configPath: String) throws {
        guard a.hiddenSize == 2560, a.numLayers == 48 else { return }
        var problems: [String] = []
        func want(_ ok: Bool, _ what: String) { if !ok { problems.append(what) } }
        want(a.numExperts == 512, "num_experts 512")
        want(a.topKExperts == 10, "num_experts_per_tok 10")
        want(a.moeIntermediateSize == 640, "moe_intermediate_size 640")
        want(a.vocabSize == 248_320, "vocab_size 248320")
        want(a.numHeads == 24 && a.numKVHeads == 2, "24 Q / 2 KV heads")
        want(a.headDim == 256, "head_dim 256")
        want(a.linearNumKHeads == 16 && a.linearNumVHeads == 48,
             "GDN 16 K / 48 V heads")
        want(a.hcCount == 4 && a.hcLowRank == 320,
             "hyper-connections 4 x 320")
        want(a.indexerBudget == 2048 && a.indexerCompressRatio == 4,
             "indexer budget 2048 / compress 4")
        want(a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12,
             "12 full-attention layers")
        want(a.pleLayerIndices == [1], "ple_layer_ids [2] (1-based)")
        guard problems.isEmpty else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen4_exp config does not match the supported "
                    + "Qwen3.8-Flash-Next contract: expected "
                    + problems.joined(separator: ", "))
        }
    }

    /// Production Qwen3.5-MoE 35B-A3B contract (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen35MoE(_ a: ArchInfo,
                                                      configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
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
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the supported "
                    + "35B-A3B architecture contract")
        }
    }
}
