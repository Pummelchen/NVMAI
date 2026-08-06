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
        let quantBits = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        var quantDict: [String: Any] = [:]
        for (slot, bits) in quantBits {
            quantDict[slot] = [
                "weightBits": bits,
                "scheme": plan.baseMode,
                "scaleType": "BF16",
                "biasType": "BF16",
                "groupSize": plan.baseGroupSize
            ]
        }

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

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
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
