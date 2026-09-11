import Testing
import Foundation
@testable import NVMAI

@Suite struct ManifestReaderTests {

    /// Build a manifest dictionary for a 2-layer toy ArchConfig and write it
    /// into a temp directory. Returns the directory URL and the toy config.
    static func writeToyManifest(_ overrides: [String: Any] = [:],
                                 flags: [String: Bool] = ["streamingPresent": true,
                                                          "turboQuantKV": false,
                                                          "aneSharedExpert": false],
                                 archOverrides: [String: Any] = [:],
                                 filesOverride: [String: [String: Any]]? = nil,
                                 config: ArchConfig = .qwenToy()) throws
                                 -> (URL, ArchConfig) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)

        let toy = config
        var archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        for (k, v) in archOverrides { archDict[k] = v }

        var files: [String: [String: Any]]
        if let f = filesOverride {
            files = f
        } else {
            files = [
                "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            ]
            for L in 0..<toy.numLayers {
                files["packed_experts/layer_\(L).bin"] = ["size": 16384, "sha256": String(repeating: "0", count: 64)]
            }
        }

        var root: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": flags,
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": 16384,
        ]
        for (k, v) in overrides { root[k] = v }

        let data = try JSONSerialization.data(withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: dir.appendingPathComponent("manifest.json"))
        return (dir, toy)
    }

    static func quant(sharedExpertBits: Int = 4,
                      routerBits: Int = 8) -> [String: Any] {
        func slot(_ bits: Int) -> [String: Any] {
            [
                "weightBits": bits,
                "scheme": "affine",
                "scaleType": "bf16",
                "biasType": "bf16",
                "groupSize": Quantization.groupSize,
            ]
        }
        return [
            "embedding": slot(4),
            "attention": slot(4),
            "router": slot(routerBits),
            "sharedExpert": slot(sharedExpertBits),
            "routedExpert": slot(4),
        ]
    }

    @Test func loadsValidManifest() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.magic == "GTURBO")
        #expect(m.numLayers == toy.numLayers)
        #expect(m.expertStride == 16384)
    }

    @Test func peekFamilyKeepsFullQwenWhenBitWidthOverridesPresent() throws {
        let arch = ArchConfig.qwen36_35B_A3B
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
        ]
        for layer in 0..<arch.numLayers {
            files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
                "size": 1,
                "sha256": String(repeating: "0", count: 64),
            ]
        }
        let (dir, _) = try Self.writeToyManifest(
            ["bitWidthOverridesHonored": 80],
            archOverrides: [
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
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            filesOverride: files,
            config: arch)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .qwen36)
    }

    @Test func peekIdentityPreservesCompatibleModelID() throws {
        let modelID = "ornith-1.5-35b-a3b-4bit"
        let (dir, _) = try Self.writeToyManifest(["modelID": modelID])
        defer { try? FileManager.default.removeItem(at: dir) }

        let identity = try ManifestReader.peekIdentity(directoryURL: dir)
        #expect(identity.modelID == modelID)
        #expect(identity.family == .qwen36)
    }

    @Test func peekIdentityRejectsEmptyModelID() throws {
        let (dir, _) = try Self.writeToyManifest(["modelID": ""])
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect {
            _ = try ManifestReader.peekIdentity(directoryURL: dir)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("modelID is empty")
        }
    }

    @Test func peekFamilyDetectsMTPByArchitectureShape() throws {
        let arch = ArchConfig.qwen36MTP
        let (dir, _) = try Self.writeToyManifest(
            ["bitWidthOverridesHonored": 12],
            archOverrides: [
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
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
            ],
            filesOverride: [
                "model_weights.bin": ["size": 1, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
                "packed_experts/layer_0.bin": ["size": 1, "sha256": String(repeating: "0", count: 64)],
            ],
            config: arch)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(try ManifestReader.peekFamily(directoryURL: dir) == .qwen36MTP)
    }

    @Test func missingManifestThrowsPartialInstall() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: .qwenToy())
        } throws: { error in
            if case ModelError.partialInstall = error { return true }
            return false
        }
    }

    @Test func oversizedManifestRejectsBeforeDecode() throws {
        let (dir, toy) = try Self.writeToyManifest()
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try Data(repeating: 0x20, count: 64).write(to: manifestURL)

        #expect {
            _ = try ManifestReader.load(directoryURL: dir,
                                        expecting: toy,
                                        maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }

    @Test func wrongMagicThrowsNotAGTurboDirectory() throws {
        let (dir, toy) = try Self.writeToyManifest(["magic": "NOT_GTURBO"])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: ModelError.notAGTurboDirectory) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        }
    }

    @Test func versionTwoThrowsUnsupportedVersion() throws {
        let (dir, toy) = try Self.writeToyManifest(["versionMajor": 2])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unsupportedVersion(let maj, _) = error { return maj == 2 }
            return false
        }
    }

    @Test func unknownFlagThrowsUnknownFlag() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "newFangledOption": true])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.unknownFlag(let n) = error { return n == "newFangledOption" }
            return false
        }
    }

    @Test func removedTurboQuantFlagIsRejected() throws {
        let (dir, toy) = try Self.writeToyManifest(flags: ["streamingPresent": true,
                                                           "turboQuantKV": true,
                                                           "aneSharedExpert": false])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("removed TurboQuant KV")
        }
    }

    @Test func productionManifestRequiresQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("manifest.quant is required")
        }
    }

    @Test func productionManifestAcceptsInt4SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 4)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 4)
    }

    @Test func productionManifestAcceptsHistoricalInt8SharedExpert() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 8)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        let manifest = try ManifestReader.load(directoryURL: dir, expecting: config)
        #expect(manifest.quant?.sharedExpert.weightBits == 8)
    }

    @Test func productionManifestRejectsUnsupportedQuantMetadata() throws {
        let (dir, config) = try Self.writeToyManifest(
            ["quant": Self.quant(sharedExpertBits: 3, routerBits: 4)],
            config: .qwen36_35B_A3B)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: config)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("unsupported quantization")
        }
    }

    @Test func archMismatchThrowsArchMismatch() throws {
        let (dir, toy) = try Self.writeToyManifest(archOverrides: ["hiddenSize": 4096])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            guard case let ModelError.archMismatch(field, _, _) = error else { return false }
            return field == "hiddenSize"
        }
    }

    @Test func nonPageAlignedExpertStrideThrows() throws {
        let (dir, toy) = try Self.writeToyManifest(["expertStride": 1024])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.expertStrideNotPageAligned = error { return true }
            return false
        }
    }

    @Test func missingLayerFileThrowsMissingFile() throws {
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            // intentionally do not list layer_0.bin or layer_1.bin
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try ManifestReader.load(directoryURL: dir, expecting: toy)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    @Test func acceptsZeroPaddedLayerFilenames() throws {
        // Writer emits packed_experts/layer_%02d.bin; loader should accept either form.
        let files: [String: [String: Any]] = [
            "model_weights.bin": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layout.json": ["size": 1024, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_00.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_01.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_02.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
            "packed_experts/layer_03.bin": ["size": 16384, "sha256": String(repeating: "0", count: 64)],
        ]
        let (dir, toy) = try Self.writeToyManifest(filesOverride: files)
        defer { try? FileManager.default.removeItem(at: dir) }
        let m = try ManifestReader.load(directoryURL: dir, expecting: toy)
        #expect(m.numLayers == toy.numLayers)
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests. 4 layers alternating
    /// gated-DeltaNet linear (mask 2) and full attention (mask 1), 8 experts
    /// (top-8 for the fixed-top-k decode kernels), gated shared expert,
    /// untied lm_head. Numbers are intentionally toy but respect every kernel
    /// divisibility constraint (D % 64, keyHeadDim % 32, even rotaryDim).
    static func qwenToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 1024,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 8,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [2, 1, 2, 1],
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.125,   // 32^-0.5
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4)
        )
    }
}

/// Validation of the family-extension geometry (hyper-connections, the QSA
/// indexer, PLE). These fields are optional so older manifests still load, but
/// when a manifest declares them they must match the architecture the runtime
/// would actually execute -- otherwise a checkpoint with, say, a different
/// hyper-connection rank runs silently against the wrong constants and
/// produces plausible nonsense.
@Suite("Manifest extension geometry")
struct ManifestExtensionGeometryTests {
    private static func toyWithExtensions() -> ArchConfig {
        var base = ArchConfig.qwenToy()
        base = ArchConfig(
            hiddenSize: base.hiddenSize, intermediateSize: base.intermediateSize,
            moeIntermediateSize: base.moeIntermediateSize, numHeads: base.numHeads,
            numKVHeads: base.numKVHeads, numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim, fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize, slidingWindow: base.slidingWindow,
            finalLogitSoftcap: base.finalLogitSoftcap, ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: base.numLayers, numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: base.tieWordEmbeddings,
            attentionKEqV: base.attentionKEqV,
            fullAttentionLayerMask: base.fullAttentionLayerMask,
            hiddenActivation: base.hiddenActivation, family: base.family,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: base.embeddingScaledBySqrtHidden,
            routerScaled: base.routerScaled,
            ffnSandwichNorms: base.ffnSandwichNorms,
            sharedExpertGated: base.sharedExpertGated,
            ropeNeoxSubdim: base.ropeNeoxSubdim,
            linearAttention: base.linearAttention,
            hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
            sparseIndexer: SparseIndexerConfig(numHeads: 4, numKVHeads: 1,
                                               headDim: 128, budget: 2048,
                                               compressRatio: 4),
            ple: PLEConfig(layerIndices: [1], embedDim: 2560,
                           convKernelSize: 4, ngramSize: 3,
                           vocabSizeBase: 20_000_000, headsPerNgram: 8,
                           vocabDivisor: 128, seed: 1234),
            routerNormTopK: true, quantGroupSize: 64)
        return base
    }

    @Test("A manifest that omits the extension fields still validates")
    func absentFieldsAreAccepted() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
    }

    @Test("Matching extension fields validate")
    func matchingFieldsAccepted() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["hcCount": 4, "hcLowRank": 320,
                            "indexerBudget": 2048, "quantGroupSize": 64,
                            "pleLayerIndices": [1]],
            config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
    }

    @Test("A disagreeing hyper-connection rank is rejected, not ignored")
    func mismatchedRankRejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["hcLowRank": 256], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }

    @Test("A disagreeing quantization group size is rejected")
    func mismatchedGroupSizeRejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["quantGroupSize": 32], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        // Repacking at the wrong group size would corrupt every weight, so
        // this is the single most important field in this set.
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }

    @Test("A disagreeing PLE layer set is rejected")
    func mismatchedPLERejected() throws {
        let cfg = Self.toyWithExtensions()
        let (dir, _) = try ManifestReaderTests.writeToyManifest(
            archOverrides: ["pleLayerIndices": [2]], config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: (any Error).self) {
            _ = try ManifestReader.load(directoryURL: dir, expecting: cfg)
        }
    }
}
