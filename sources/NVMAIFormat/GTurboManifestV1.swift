import Foundation

package struct GTurboManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct GTurboManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayerMask: [Int]

    /// The family the payload was repacked for. Absent in the earliest
    /// manifests, which predate more than one family; the reader falls back to
    /// inferring from layer shape when it is missing.
    package let family: String?

    // Family extension geometry; absent in manifests written before these
    // families existed, validated by the reader whenever present.
    package let hcCount: Int?
    package let hcLowRank: Int?
    package let indexerNumHeads: Int?
    package let indexerNumKVHeads: Int?
    package let indexerHeadDim: Int?
    package let indexerBudget: Int?
    package let indexerCompressRatio: Int?
    package let pleLayerIndices: [Int]?
    package let pleEmbedDim: Int?
    package let pleConvKernelSize: Int?
    package let pleNgramSize: Int?
    package let pleVocabSizeBase: Int?
    package let pleHeadsPerNgram: Int?
    package let pleVocabDivisor: Int?
    package let routerNormTopK: Bool?
    package let quantGroupSize: Int?

    /// Gated-DeltaNet geometry. Optional for the same reason as the block
    /// above: manifests written before a reader needed them do not carry them.
    /// The writer has always emitted these keys, so they were present on disk
    /// and simply not decoded.
    package let linearNumKHeads: Int?
    package let linearNumVHeads: Int?
    package let linearKeyHeadDim: Int?
    package let linearValueHeadDim: Int?
    package let linearConvKernelSize: Int?

    /// The layer conventions, which the writer has always emitted and no reader
    /// decoded until a family without an architecture preset needed them (the
    /// dense Qwen 3.5 models). Optional so every earlier manifest still
    /// decodes.
    package let attnOutputGate: Bool?
    package let attentionScale: Double?
    package let embeddingScaledBySqrtHidden: Bool?
    package let routerScaled: Bool?
    package let ffnSandwichNorms: Bool?
    package let sharedExpertGated: Bool?
    package let ropeNeoxSubdim: Bool?

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int],
                 family: String? = nil,
                 hcCount: Int? = nil,
                 hcLowRank: Int? = nil,
                 indexerNumHeads: Int? = nil,
                 indexerNumKVHeads: Int? = nil,
                 indexerHeadDim: Int? = nil,
                 indexerBudget: Int? = nil,
                 indexerCompressRatio: Int? = nil,
                 pleLayerIndices: [Int]? = nil,
                 pleEmbedDim: Int? = nil,
                 pleConvKernelSize: Int? = nil,
                 pleNgramSize: Int? = nil,
                 pleVocabSizeBase: Int? = nil,
                 pleHeadsPerNgram: Int? = nil,
                 pleVocabDivisor: Int? = nil,
                 routerNormTopK: Bool? = nil,
                 quantGroupSize: Int? = nil,
                 linearNumKHeads: Int? = nil,
                 linearNumVHeads: Int? = nil,
                 linearKeyHeadDim: Int? = nil,
                 linearValueHeadDim: Int? = nil,
                 linearConvKernelSize: Int? = nil,
                 attnOutputGate: Bool? = nil,
                 attentionScale: Double? = nil,
                 embeddingScaledBySqrtHidden: Bool? = nil,
                 routerScaled: Bool? = nil,
                 ffnSandwichNorms: Bool? = nil,
                 sharedExpertGated: Bool? = nil,
                 ropeNeoxSubdim: Bool? = nil) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
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
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.family = family
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
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
    }
}

package struct GTurboManifestQuantSlotV1: Codable, Equatable, Sendable {
    /// Weight widths a reader in this project implements.
    ///
    /// The kernel arithmetic is `32 / bits` lanes per u32 word and `columns / 8`
    /// packed words per row at the 4-bit end, so a value outside this set is not
    /// a wider or narrower packing — it is a number nothing can decode. 6-bit
    /// was withdrawn (non-power-of-two packing measured 46.8 GB/s against 60 for
    /// 4-bit and 8-bit).
    package static let supportedWeightBits: Set<Int> = [4, 8]

    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct GTurboManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: GTurboManifestQuantSlotV1
    package let attention: GTurboManifestQuantSlotV1
    package let router: GTurboManifestQuantSlotV1
    package let sharedExpert: GTurboManifestQuantSlotV1
    package let routedExpert: GTurboManifestQuantSlotV1
    /// Per-tensor width overrides, keyed by tensor stem.
    ///
    /// The writer emits these beside the five slots and the decoder used to
    /// drop them, which mattered as soon as a reader needed them: a 4-bit
    /// build keeps its embedding and every attention K/V at 8 bits, and
    /// dequantizing those as 4-bit unpacks the same bytes wrongly. Absent in
    /// manifests written before this field existed, and in builds that have no
    /// overrides at all.
    package let overrides: [String: GTurboManifestQuantSlotV1]?

    private enum CodingKeys: String, CodingKey {
        case embedding, attention, router, sharedExpert, routedExpert
    }

    /// Hand-written because the object mixes five fixed slots with an open set
    /// of per-tensor overrides keyed by tensor stem. A synthesised `Codable`
    /// silently dropped the open set, which is how a 4-bit install's 8-bit
    /// K/V got read back as 4-bit.
    package init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        embedding = try container.decode(GTurboManifestQuantSlotV1.self, forKey: .embedding)
        attention = try container.decode(GTurboManifestQuantSlotV1.self, forKey: .attention)
        router = try container.decode(GTurboManifestQuantSlotV1.self, forKey: .router)
        sharedExpert = try container.decode(GTurboManifestQuantSlotV1.self,
                                            forKey: .sharedExpert)
        routedExpert = try container.decode(GTurboManifestQuantSlotV1.self,
                                            forKey: .routedExpert)
        let dynamic = try decoder.container(keyedBy: AnyKey.self)
        var overrides: [String: GTurboManifestQuantSlotV1] = [:]
        for key in dynamic.allKeys {
            guard CodingKeys(stringValue: key.stringValue) == nil else { continue }
            // `try`, not `try?`. A malformed override used to be dropped in
            // silence, which is precisely how a width goes missing: the reader
            // then falls back to a slot, and a slot that disagrees with the
            // payload unpacks the same bytes wrongly. Every key in this object
            // is one the writer emitted, so a key that does not decode is
            // corruption rather than an extension this build should tolerate.
            let slot = try dynamic.decode(GTurboManifestQuantSlotV1.self, forKey: key)
            // A width no kernel implements would be read by arithmetic that
            // assumes 4 or 8 bits per value (`32 / bits` lanes, `columns / 8`
            // packed words). Refusing here is the difference between a load
            // error and fluent nonsense.
            guard GTurboManifestQuantSlotV1.supportedWeightBits.contains(slot.weightBits) else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: dynamic,
                    debugDescription: "quant override for \(key.stringValue) declares "
                        + "\(slot.weightBits) bits; supported: "
                        + "\(GTurboManifestQuantSlotV1.supportedWeightBits.sorted())")
            }
            overrides[key.stringValue] = slot
        }
        self.overrides = overrides.isEmpty ? nil : overrides
    }

    package func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(embedding, forKey: .embedding)
        try container.encode(attention, forKey: .attention)
        try container.encode(router, forKey: .router)
        try container.encode(sharedExpert, forKey: .sharedExpert)
        try container.encode(routedExpert, forKey: .routedExpert)
        if let overrides {
            var dynamic = encoder.container(keyedBy: AnyKey.self)
            for (stem, slot) in overrides {
                try dynamic.encode(slot, forKey: AnyKey(stringValue: stem)!)
            }
        }
    }

    /// A decoder/encoder key for the arbitrary tensor stems in the object.
    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    package init(embedding: GTurboManifestQuantSlotV1,
                 attention: GTurboManifestQuantSlotV1,
                 router: GTurboManifestQuantSlotV1,
                 sharedExpert: GTurboManifestQuantSlotV1,
                 routedExpert: GTurboManifestQuantSlotV1,
                 overrides: [String: GTurboManifestQuantSlotV1]? = nil) {
        self.overrides = overrides
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}

package struct GTurboManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestArchV1
    package let quant: GTurboManifestQuantV1?
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64
    package let bitWidthOverridesHonored: Int?

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV1.versionMajor,
                 versionMinor: Int = GTurboFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestArchV1,
                 quant: GTurboManifestQuantV1?,
                 files: [String: GTurboManifestFileV1],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64,
                 bitWidthOverridesHonored: Int?) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
        self.bitWidthOverridesHonored = bitWidthOverridesHonored
    }
}

package enum GTurboManifestCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> GTurboManifestV1 {
        let manifest: GTurboManifestV1
        do { manifest = try JSONDecoder().decode(GTurboManifestV1.self, from: data) }
        catch { throw NVMAIFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func validate(_ manifest: GTurboManifestV1) throws {
        guard manifest.magic == GTurboFormatV1.magic else {
            throw NVMAIFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV1.versionMajor,
              manifest.versionMinor >= 0 else {
            throw NVMAIFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !GTurboFormatV1.knownFlags.contains(flag) {
            throw NVMAIFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        // A dense payload has no routed experts at all -- the planner writes
        // expertsPerLayer 0 and expertStride 0 for one -- so "greater than
        // zero" is the wrong test for it. Every other family still needs both.
        let isDense = manifest.arch.family == "qwen3_5_dense"
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0,
              isDense
                ? (manifest.expertsPerLayer == 0 && manifest.expertStride == 0)
                : (manifest.expertsPerLayer > 0 && manifest.expertStride > 0),
              manifest.expertStride % GTurboFormatV1.alignmentBytes == 0 else {
            throw NVMAIFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.numExperts == manifest.expertsPerLayer else {
            throw NVMAIFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        // `moeIntermediateSize` is 0 for a dense model, which has no
        // per-expert FFN; its FFN width is `ffnIntermediate`.
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
              isDense || arch.moeIntermediateSize > 0, arch.numHeads > 0,
              arch.numKVHeads > 0, arch.numFullKVHeads > 0,
              arch.headDim > 0, arch.fullHeadDim > 0,
              arch.vocabSize > 0, arch.slidingWindow >= 0,
              isDense || (arch.topKExperts > 0 && arch.topKExperts <= arch.numExperts),
              arch.finalLogitSoftcap.isFinite,
              arch.ropeTheta.isFinite, arch.ropeTheta > 0,
              arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
              arch.partialRotaryFactor.isFinite,
              arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
              !arch.hiddenActivation.isEmpty,
              arch.fullAttentionLayerMask.count == arch.numLayers,
              arch.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 || $0 == 2 }) else {
            throw NVMAIFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        if let quant = manifest.quant {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ] {
                guard slot.weightBits > 0, slot.weightBits <= 32,
                      slot.groupSize > 0,
                      !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                      !slot.biasType.isEmpty else {
                    throw NVMAIFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
        }
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw NVMAIFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw NVMAIFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw NVMAIFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw NVMAIFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

}