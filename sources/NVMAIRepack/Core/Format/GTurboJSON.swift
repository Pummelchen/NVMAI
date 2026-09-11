import Foundation

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = "GTURBO"
    static let versionMajor = 1
    static let versionMinor = 0

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        var archDict: [String: Any] = [
            "hiddenSize": arch.hiddenSize,
            "ffnIntermediate": arch.intermediateSize,
            "moeIntermediateSize": arch.moeIntermediateSize,
            "numHeads": arch.numHeads,
            "numKVHeads": arch.numKVHeads,
            "numFullKVHeads": arch.numFullKVHeads,
            "headDim": arch.headDim,
            "fullHeadDim": arch.fullHeadDim,
            "vocabSize": arch.vocabSize,
            "slidingWindow": arch.slidingWindow,
            "finalLogitSoftcap": arch.finalLogitSoftcap,
            "ropeTheta": arch.ropeTheta,
            "fullRopeTheta": arch.fullRopeTheta,
            "partialRotaryFactor": arch.partialRotaryFactor,
            "numLayers": arch.numLayers,
            "numExperts": arch.numExperts,
            "topKExperts": arch.topKExperts,
            "tieWordEmbeddings": arch.tieWordEmbeddings,
            "attentionKEqV": arch.attentionKEqV,
            "hiddenActivation": arch.hiddenActivation,
            "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) }
        ]
        // Family extension fields. Always written for the Qwen families.
        archDict["family"] = arch.family.rawValue
        archDict["attnOutputGate"] = arch.attnOutputGate
        archDict["attentionScale"] = arch.attentionScale
        archDict["embeddingScaledBySqrtHidden"] = arch.embeddingScaledBySqrtHidden
        archDict["routerScaled"] = arch.routerScaled
        archDict["ffnSandwichNorms"] = arch.ffnSandwichNorms
        archDict["sharedExpertGated"] = arch.sharedExpertGated
        archDict["ropeNeoxSubdim"] = arch.ropeNeoxSubdim
        archDict["linearNumKHeads"] = arch.linearNumKHeads
        archDict["linearNumVHeads"] = arch.linearNumVHeads
        archDict["linearKeyHeadDim"] = arch.linearKeyHeadDim
        archDict["linearValueHeadDim"] = arch.linearValueHeadDim
        archDict["linearConvKernelSize"] = arch.linearConvKernelSize
        // Extension geometry is written only for the families that have it, so
        // manifests for the existing families stay byte-identical. The reader
        // validates these whenever present, which is what stops a checkpoint
        // with different hyper-connection / indexer / PLE geometry from being
        // run silently against the runtime's hardcoded constants.
        if arch.family == .qwen38flash {
            archDict["hcCount"] = arch.hcCount
            archDict["hcLowRank"] = arch.hcLowRank
            archDict["indexerNumHeads"] = arch.indexerNumHeads
            archDict["indexerNumKVHeads"] = arch.indexerNumKVHeads
            archDict["indexerHeadDim"] = arch.indexerHeadDim
            archDict["indexerBudget"] = arch.indexerBudget
            archDict["indexerCompressRatio"] = arch.indexerCompressRatio
            archDict["pleLayerIndices"] = arch.pleLayerIndices
            archDict["pleEmbedDim"] = arch.pleEmbedDim
            archDict["pleConvKernelSize"] = arch.pleConvKernelSize
            archDict["pleNgramSize"] = arch.pleNgramSize
            archDict["pleVocabSizeBase"] = arch.pleVocabSizeBase
            archDict["pleHeadsPerNgram"] = arch.pleHeadsPerNgram
            archDict["pleVocabDivisor"] = arch.pleVocabDivisor
            archDict["routerNormTopK"] = arch.routerNormTopK
            archDict["quantGroupSize"] = arch.quantGroupSize
        }
        let quantDict = quantObject(plan: plan, bitWidths: bitWidths)

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        let manifest: [String: Any] = [
            "magic": GTurboJSON.magic,
            "versionMajor": GTurboJSON.versionMajor,
            "versionMinor": GTurboJSON.versionMinor,
            "flags": [
                "streamingPresent": plan.streamingPresent,
                "turboQuantKV": plan.turboQuantKV,
                "aneSharedExpert": plan.aneSharedExpert
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
            "bitWidthOverridesHonored": plan.bitsOverrideCount
        ]
        return try JSONSerialization.data(withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// The `quant` object: the five width *slots*, plus one entry per
    /// quantified resident tensor.
    ///
    /// The slots alone cannot describe a build. They are what the installer
    /// *derives* from the tensor data, not what the source checkpoint
    /// declared, and a "4-bit" model is not uniformly 4-bit: the Qwen 3.5
    /// 2B/4B/9B keep their embedding and several attention K/V pairs at 8
    /// bits while the slots say `attention: 4`. Both statements are true of
    /// different tensors, and only the per-tensor entries say which is which.
    ///
    /// This is the bug that was shipped and then found by comparing `.gturbo`
    /// logits against the snapshot's: a reader that trusts the slots unpacks
    /// those 8-bit tensors as 4-bit. The word count changes, the strides
    /// still divide evenly, every shape check passes, and the model answers
    /// fluently and wrongly. Writing the real width for every tensor makes
    /// the manifest say what was actually packed, so no reader has to
    /// re-derive it from the slots and none can get it wrong.
    ///
    /// Unquantized tensors (norms, scalars) carry no `quantSpec` and are
    /// deliberately absent: they are read as BF16 by `dtype` and never
    /// dequantized. A stem that collides with a slot name would overwrite a
    /// slot, so it is skipped.
    private static func quantObject(plan: RepackPlan,
                                    bitWidths: QuantBitWidths) -> [String: Any] {
        let slots = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        func entry(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": plan.baseMode,
             "scaleType": "BF16", "biasType": "BF16",
             "groupSize": plan.baseGroupSize]
        }
        var dict: [String: Any] = [:]
        for (slot, bits) in slots { dict[slot] = entry(bits) }
        for resident in plan.resident.entries {
            guard let spec = resident.quantSpec else { continue }
            let stem = resident.name.hasSuffix(".weight")
                ? String(resident.name.dropLast(".weight".count)) : resident.name
            guard dict[stem] == nil else { continue }
            dict[stem] = entry(spec.bits)
        }
        return dict
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        // Every layer that carries experts must share one stride. The manifest
        // records a single value and `GTurboLayoutValidator` refuses a layout
        // whose layers disagree, so a plan like that cannot be written correctly
        // -- and the validator runs after the caller has already written the
        // packed payload, which for the 35B families is hundreds of gigabytes.
        // The two callers take the stride from the first non-empty layer, so a
        // disagreement between layers is exactly what would go unnoticed here.
        let layerStrides = Set(plan.layers.filter { $0.expertsPerLayer > 0 }
            .map(\.expertStride))
        guard layerStrides.count <= 1 else {
            throw RepackError.configurationInvalid(
                detail: "packed expert stride differs between layers: "
                    + "\(layerStrides.sorted())")
        }
        if let only = layerStrides.first, only != expertStride {
            throw RepackError.configurationInvalid(
                detail: "packed expert stride \(expertStride) does not match the "
                    + "layers' \(only); the layout is written from the first "
                    + "non-empty layer, so these have to agree")
        }
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size":   slice.sizeInExpertBlob,
                        "dtype":  slice.dtype == 0 ? "U32" : "BF16",
                        "shape":  slice.logicalShape.map { Int($0) }
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size":   lp.expertStride,
                    "tensors": tensors
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file":  layerFile,
                "experts": experts
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            // Same source as `expertStride`: the first layer that actually
            // packs experts. GTurboPackedExpertsLayoutCodec.decode requires
            // expertsPerLayer > 0 and consistent across all layers.
            "expertsPerLayer": plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}

private extension RepackPlan {
    /// The .gturbo layout always streams routed experts from per-layer files;
    /// the remaining flags are fixed for the Qwen 3.6 baseline (no quantized
    /// KV, no ANE shared-expert fusion) and are computed here so the manifest
    /// mirrors the plan rather than a hardcoded dictionary.
    var streamingPresent: Bool { layers.contains { $0.expertsPerLayer > 0 } }
    var turboQuantKV: Bool { false }
    var aneSharedExpert: Bool { false }
}
