import Foundation
import NVMAIFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
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
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]

    // Family extension geometry. Optional because manifests written before
    // these families existed do not carry them; when a manifest DOES declare
    // them they are validated, so a checkpoint whose hyper-connection, QSA
    // indexer or PLE geometry differs from the runtime's cannot be run
    // silently against the wrong constants.
    public let hcCount: Int?
    public let hcLowRank: Int?
    public let indexerNumHeads: Int?
    public let indexerNumKVHeads: Int?
    public let indexerHeadDim: Int?
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?
    public let pleLayerIndices: [Int]?
    public let pleEmbedDim: Int?
    public let pleConvKernelSize: Int?
    public let pleNgramSize: Int?
    public let pleVocabSizeBase: Int?
    public let pleHeadsPerNgram: Int?
    public let pleVocabDivisor: Int?
    public let routerNormTopK: Bool?
    public let quantGroupSize: Int?
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public struct ManifestIdentity: Equatable, Sendable {
    public let modelID: String
    public let family: ModelFamily
    /// Routed-expert width, which is what "a 4-bit model" names: the experts
    /// are almost all of the payload.
    public let weightBits: Int
}

public enum ManifestReader {
    /// Weight widths this build accepts. 6-bit was withdrawn: non-power-of-two
    /// packing measured 46.8 GB/s against 60 for 4-bit and 8-bit, and a 26 GB
    /// model does not fit a 24 GB machine.
    public static let supportedWeightBits: Set<Int> = [4, 8]

    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = GTurboFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    package static func decode(data: Data,
                               expecting: ArchConfig) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try GTurboManifestCodec.decodeUnchecked(data)
            guard wire.magic == GTurboFormatV1.magic else {
                throw ModelError.notAGTurboDirectory
            }
            guard wire.versionMajor == GTurboFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !GTurboFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % GTurboFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(GTurboFormatV1.alignmentBytes))
            }
            try GTurboManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    /// Extract the architecture dimensions from the manifest without full
    /// cross-validation so it can be used to auto-select the expected
    /// configuration (e.g. by the installation probe).
    public static func peekFamily(directoryURL: URL) throws -> ModelFamily {
        try peekIdentity(directoryURL: directoryURL).family
    }

    /// Read the manifest's model identity and compatible runtime family without
    /// mapping weights or creating a Metal device.
    public static func peekIdentity(directoryURL: URL) throws -> ManifestIdentity {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata("manifest.json", maxBytes: 4 * 1024 * 1024)
        let wire = try JSONDecoder().decode(GTurboManifestV1.self, from: data)
        guard !wire.modelID.isEmpty else {
            throw ModelError.indexCorrupt(detail: "manifest modelID is empty")
        }
        let bits = wire.quant?.routedExpert.weightBits ?? 4
        switch wire.arch.hiddenActivation {
        case "silu":
            break
        default:
            throw ModelError.unsupportedArchitecture(
                detail: "hiddenActivation=\(wire.arch.hiddenActivation)")
        }
        // Prefer what the manifest declares. Inferring family from layer shape
        // only worked while the families differed in shape; it silently
        // reports qwen36 for anything it does not recognise, which is how a
        // Qwen3.8-Flash-Next payload gets loaded against the wrong
        // architecture and fails on a hidden-size mismatch rather than being
        // identified.
        if let declared = wire.arch.family,
           let family = ModelFamily(rawValue: declared) {
            return ManifestIdentity(modelID: wire.modelID, family: family,
                                    weightBits: bits)
        }
        let mtp = ArchConfig.qwen36MTP
        if wire.arch.numLayers == mtp.numLayers,
           wire.arch.slidingWindow == mtp.slidingWindow,
           wire.arch.fullAttentionLayerMask == mtp.fullAttentionLayerMask.map(Int.init) {
            return ManifestIdentity(modelID: wire.modelID, family: .qwen36MTP,
                                    weightBits: bits)
        }
        return ManifestIdentity(modelID: wire.modelID, family: .qwen36,
                                weightBits: bits)
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant, family: expected.family)
        } else if expected.numLayers == ArchConfig.qwen36_35B_A3B.numLayers,
                  expected.hiddenSize == ArchConfig.qwen36_35B_A3B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        // Validate that all expected layer files are listed in the manifest.
        // Accept both `layer_0.bin` and `layer_00.bin` naming conventions.
        for L in 0..<m.numLayers {
            let layerFileShort = String(format: "packed_experts/layer_%d.bin", L)
            let layerFilePadded = String(format: "packed_experts/layer_%02d.bin", L)
            if m.files[layerFileShort] == nil && m.files[layerFilePadded] == nil {
                throw ModelError.missingFile(name: layerFileShort)
            }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant,
                                      family: ModelFamily) throws {
        let allowedRouterBits: Set<Int>
        switch family {
        case .qwen36MTP:
            allowedRouterBits = [4, 8]
        case .qwen36:
            allowedRouterBits = [8]
        case .qwen38flash, .qwen38flashMTP:
            // The community MLX checkpoints quantize the router at the model's
            // uniform width (4 or 8 bits, group 32). The draft head is
            // quantized with the target, so it inherits the same rule.
            allowedRouterBits = [4, 8]
        }
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, Self.supportedWeightBits),
            ("attention", quant.attention, Self.supportedWeightBits),
            ("router", quant.router, allowedRouterBits),
            ("sharedExpert", quant.sharedExpert, Self.supportedWeightBits),
            ("routedExpert", quant.routedExpert, Self.supportedWeightBits),
        ]
        for (name, slot, allowedBits) in slots {
            // 6-bit was withdrawn rather than deprecated, so say so instead of
            // letting a previously working model fail as "unsupported
            // quantization" with no route forward.
            if slot.weightBits == 6 {
                // Not `indexCorrupt`: the payload is intact and the user would
                // otherwise be told to re-download a file that is fine.
                throw ModelError.unsupportedArchitecture(detail: """
                    6-bit models are no longer supported (\(name) is 6-bit). \
                    Its packing is not a power of two, which measured 46.8 GB/s \
                    against 60 for both 4-bit and 8-bit, and it does not fit \
                    24 GB. Install the 4-bit or 8-bit build instead.
                    """)
            }
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)

        // Extension geometry is checked only when the manifest declares it.
        // Absent means an older manifest, which the core fields above already
        // pin; present and disagreeing means the payload is not the
        // architecture this runtime would execute, which must not be silent.
        func checkOptional<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T?, _ expected: T) throws {
            guard let actual else { return }
            try check(field, actual, expected)
        }
        try checkOptional("hcCount", a.hcCount, e.hyperConnections.count)
        try checkOptional("hcLowRank", a.hcLowRank, e.hyperConnections.lowRank)
        try checkOptional("indexerNumHeads", a.indexerNumHeads,
                          e.sparseIndexer.numHeads)
        try checkOptional("indexerNumKVHeads", a.indexerNumKVHeads,
                          e.sparseIndexer.numKVHeads)
        try checkOptional("indexerHeadDim", a.indexerHeadDim,
                          e.sparseIndexer.headDim)
        try checkOptional("indexerBudget", a.indexerBudget,
                          e.sparseIndexer.budget)
        try checkOptional("indexerCompressRatio", a.indexerCompressRatio,
                          e.sparseIndexer.compressRatio)
        try checkOptional("pleLayerIndices", a.pleLayerIndices?.description,
                          e.ple.layerIndices.description)
        try checkOptional("pleEmbedDim", a.pleEmbedDim, e.ple.embedDim)
        try checkOptional("pleConvKernelSize", a.pleConvKernelSize,
                          e.ple.convKernelSize)
        try checkOptional("pleNgramSize", a.pleNgramSize, e.ple.ngramSize)
        try checkOptional("pleVocabSizeBase", a.pleVocabSizeBase,
                          e.ple.vocabSizeBase)
        try checkOptional("pleHeadsPerNgram", a.pleHeadsPerNgram,
                          e.ple.headsPerNgram)
        try checkOptional("pleVocabDivisor", a.pleVocabDivisor,
                          e.ple.vocabDivisor)
        try checkOptional("routerNormTopK", a.routerNormTopK, e.routerNormTopK)
        try checkOptional("quantGroupSize", a.quantGroupSize, e.quantGroupSize)
    }
}

private extension ManifestFileEntry {
    init(wire: GTurboManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

private extension ManifestArch {
    init(wire: GTurboManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask,
                  hcCount: wire.hcCount,
                  hcLowRank: wire.hcLowRank,
                  indexerNumHeads: wire.indexerNumHeads,
                  indexerNumKVHeads: wire.indexerNumKVHeads,
                  indexerHeadDim: wire.indexerHeadDim,
                  indexerBudget: wire.indexerBudget,
                  indexerCompressRatio: wire.indexerCompressRatio,
                  pleLayerIndices: wire.pleLayerIndices,
                  pleEmbedDim: wire.pleEmbedDim,
                  pleConvKernelSize: wire.pleConvKernelSize,
                  pleNgramSize: wire.pleNgramSize,
                  pleVocabSizeBase: wire.pleVocabSizeBase,
                  pleHeadsPerNgram: wire.pleHeadsPerNgram,
                  pleVocabDivisor: wire.pleVocabDivisor,
                  routerNormTopK: wire.routerNormTopK,
                  quantGroupSize: wire.quantGroupSize)
    }
}

private extension ManifestQuantSlot {
    init(wire: GTurboManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: GTurboManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  router: ManifestQuantSlot(wire: wire.router),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert))
    }
}

private extension Manifest {
    init(wire: GTurboManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride)
    }
}
