//
//  Model+Loading.swift
//  NVMAI
//
//  Opening a `.gturbo` install: the hashing/verification pipeline that turns a
//  directory into a typed `Model`, split out of `Model.swift` so the model's
//  shape and the code that loads it are separate reads.
//

import Foundation
import Metal
import Darwin
import NVMAIFormat

extension Model {

    /// Open a `.gturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    /// lint:allow-long a sequential load pipeline -- open, hash, verify the
    /// receipt, decode the layout, map the resident buffer -- whose stages
    /// share a descriptor, sizes and timing stats. Extracting any of them
    /// needs six or seven parameters, trading one readable sequence for
    /// several functions with unwieldy signatures.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig = .qwen36_35B_A3B,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 32),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256

        // -- create the directory handle and open manifest
        let modelDirectory = try GTurboModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do {
            manifestFD = try modelDirectory.openFile("manifest.json")
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        defer { close(manifestFD) }

        // -- read manifest data and compute hash from the in-memory buffer
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD,
            relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart

        // -- optional trusted-receipt validation
        let receipt: VerifiedInstallReceipt?
        var trustedReceiptUsable = false
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            do {
                let receiptFD = try modelDirectory.openFile(
                    VerifiedInstallReceiptReader.fileName)
                defer { close(receiptFD) }
                let receiptData = try modelDirectory.readMetadata(
                    fileDescriptor: receiptFD,
                    relativePath: VerifiedInstallReceiptReader.fileName,
                    maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
                let loadedReceipt = try JSONDecoder().decode(
                    VerifiedInstallReceipt.self, from: receiptData)
                try VerifiedInstallReceiptReader.validateManifestBinding(
                    loadedReceipt,
                    directoryURL: directoryURL,
                    manifestSha256: manifestSha)
                stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
                receipt = loadedReceipt
                trustedReceiptUsable = true
            } catch {
                // The trusted-receipt policy is strict: a missing or invalid
                // receipt is a hard error, because silently falling back to a
                // full re-hash would mask tampering or a moved directory and
                // defeat the policy's purpose.
                if let receiptError = error as? ModelError,
                   case .trustedReceiptInvalid = receiptError {
                    throw receiptError
                }
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName): \(error)")
            }
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // -- verify the small, always-touched files before mapping model data
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }

        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
        defer { close(layoutFD) }

        // Read layout.json and validate size via modelDirectory
        let layoutData = try modelDirectory.readMetadata(
            fileDescriptor: layoutFD,
            relativePath: "packed_experts/layout.json",
            maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
        guard UInt64(layoutData.count) == layoutEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "packed_experts/layout.json",
                expected: layoutEntry.size,
                actual: UInt64(layoutData.count))
        }

        // Validate weights file size via modelDirectory
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }

        // SHA-256: weights via FD, layout via in-memory data. Under a usable
        // trusted-receipt policy the installer already pinned these hashes at
        // install time, so re-hashing the full weights file is skipped; the
        // payload is instead warmed with F_RDADVISE so GPU first-touch does
        // not fault on cold pages. A receipt that failed to validate falls
        // back to the full hash here.
        if resolvedIntegrityPolicy == .fullSha256 || !trustedReceiptUsable {
            let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try Sha256Verifier.verifyFile(fileDescriptor: weightsFD,
                                          named: "model_weights.bin",
                                          expectedHex: weightsEntry.sha256)
            guard Sha256Verifier.hashData(layoutData).lowercased()
                    == layoutEntry.sha256.lowercased() else {
                throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
            }
            stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart
        } else {
            _ = RDAdvice.call(fd: weightsFD, offset: 0, byteCount: weightsSize)
        }

        // -- decode layout from NVMAIFormat wire codec
        let layout = try PackedExpertsLayoutReader.decode(data: layoutData,
                                                          manifest: manifest)
        if trustedReceiptUsable {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(modelDirectory: modelDirectory,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // -- load resident index using the FD passed from openFile()
        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  manifest: manifest,
                                  config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(fileSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        // -- create resident buffer, reusing the opened FD
        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: weightsFD)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    private static func validateTrustedReceiptLayerLayout(
        modelDirectory: GTurboModelDirectory,
        manifest: Manifest,
        layout: PackedExpertsLayout
    ) throws {
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    /// Width of the fixed threadgroup tiles the MoE and GDN kernels stage
    /// activations into: `kMoEXMaxD` in `moe.metal` and `xt[2816]` in
    /// `gdn.metal`. A hidden size above this writes past the tile, so
    /// `validateRuntimeSchema` refuses it rather than letting the kernel do it.
    static let maximumThreadgroupTileWidth = 2816

    /// Refuses model geometry the compiled kernels cannot serve.
    ///
    /// Split out of `validateRuntimeSchema` to keep it inside the project's
    /// function-length gate. Both checks are about the *shape* of the model
    /// rather than its tensors, and both exist because the failure they prevent
    /// is silent: the kernels index threadgroup memory and pick attention
    /// geometry from these values, so a shape they were not compiled for does not
    /// fail, it produces wrong numbers.
    static func validateExecutableGeometry(_ config: ArchConfig) throws {
        // A sliding-window layer (mask 0) is a valid `ArchConfig` value and the
        // CPU engine implements it, but the GPU path does not: the gated
        // attention branch runs every non-linear layer as *full* attention with
        // `fullHeadDim`, `numFullKVHeads` and `fullRopeTheta`, and prefill's neox
        // branch uses `fullRopeTheta` unconditionally too. A mask-0 layer would
        // attend to the whole context with the wrong row geometry, silently. No
        // shipped preset declares one (they use 1 and 2), so this refuses rather
        // than guessing — implementing windowed handling in that branch is the
        // alternative, and this guard is what says which one is missing.
        guard !config.fullAttentionLayerMask.contains(0) else {
            throw ModelError.unsupportedArchitecture(
                detail: "a sliding-window layer (fullAttentionLayerMask == 0) is not "
                    + "implemented on the GPU path: the gated attention branch runs "
                    + "every non-linear layer as full attention")
        }
        guard config.hiddenSize <= Self.maximumThreadgroupTileWidth else {
            throw ModelError.unsupportedArchitecture(
                detail: "hiddenSize \(config.hiddenSize) exceeds the "
                    + "\(Self.maximumThreadgroupTileWidth)-element threadgroup tiles "
                    + "the MoE and GDN kernels are compiled with")
        }
        // The attention kernels size their scratch from two compile-time
        // ceilings: `Attention.maxQHeads` (the host-side split-KV reduction
        // buffer) and `kAttnMaxHeadDim` in `attention.metal`, which declares
        // `q_smem` and the per-thread row from it. Nothing bounded the config
        // against either. A manifest with more query heads than the ceiling
        // reaches `Attention.encode`'s own precondition and **traps** -- an abort
        // on install-derived data -- and a head dimension above the kernel's
        // constant overruns threadgroup memory, silently. Refused at load, for
        // the same reason as the hidden-size bound above.
        guard config.numHeads <= Attention.maxQHeads else {
            throw ModelError.unsupportedArchitecture(
                detail: "numHeads \(config.numHeads) exceeds the \(Attention.maxQHeads)-head "
                    + "split-KV scratch the attention kernels are built with")
        }
        guard config.fullHeadDim <= Attention.maxHeadDim,
              config.headDim <= Attention.maxHeadDim else {
            throw ModelError.unsupportedArchitecture(
                detail: "head dimension \(max(config.fullHeadDim, config.headDim)) exceeds "
                    + "the \(Attention.maxHeadDim)-element attention threadgroup tile")
        }

    }

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }
        // The MoE and GDN kernels stage activations into fixed threadgroup tiles
        // of 2816 elements (`kMoEXMaxD` in moe.metal, `xt[2816]` in gdn.metal),
        // and their staging loops are bounded by the configured hidden size. A
        // model wider than that writes past the tile into whatever shares the
        // threadgroup's memory -- undefined behaviour rather than a caught
        // error, and reachable only by a config that has never shipped (every
        // preset here is 2048, 2560 or 2816). This is the guard for the next
        // family, and it is what lets the tile stay a compile-time constant.
        try Self.validateExecutableGeometry(config)

        let checks = RuntimeSchemaChecks(residentIndex: residentIndex, quant: quant)

        switch config.family {
        case .qwen35Dense:
            // Unreachable by construction: a dense family is served by the CPU
            // engine, which validates its own snapshot schema and never goes
            // through `Model.load`. Throwing is the honest outcome if that
            // routing is ever broken -- the alternative is validating a dense
            // payload against the MoE checks, which would fail confusingly on
            // a missing routed-expert tensor.
            throw ModelError.unsupportedArchitecture(
                detail: "qwen35Dense is served by the CPU engine, not Model.load")
        case .qwen38flash:
            // Embedding and head are 8-bit in this checkpoint while the body
            // is 4-bit, so both are validated against the embedding slot the
            // manifest declares rather than an assumed width.
            try checks.requireAffine("model.language_model.embed_tokens.weight",
                                     rows: config.vocabSize,
                                     columns: config.hiddenSize,
                                     slot: quant.embedding)
            // `lm_head` sits at the archive root in this family, not under the
            // language-model prefix.
            try checks.requireAffine("lm_head.weight",
                                     rows: config.vocabSize,
                                     columns: config.hiddenSize,
                                     slot: quant.embedding)
            // The hyper-connection residual is the family's defining feature
            // and the one thing whose absence would let a mis-repacked payload
            // load and then compute a plain-residual model. Check the
            // model-level mixer and one layer's worth of both sublayer gates.
            let hcDim = config.hiddenSize * config.hyperConnections.count
            try checks.requireBF16(
                "model.language_model.hyper_connection_mixer.hc_norm",
                count: hcDim)
            for layer in 0..<config.numLayers {
                try checks.requireBF16(
                    "model.language_model.layers.\(layer)."
                        + "attn_hyper_connection.hc_norm", count: hcDim)
                try checks.requireBF16(
                    "model.language_model.layers.\(layer)."
                        + "mlp_hyper_connection.hc_norm", count: hcDim)
            }
            // The PLE block exists on exactly the configured layers, and its
            // constants and table are passthrough files rather than tensors --
            // their presence is the manifest's business, checked below.
            for layer in config.ple.layerIndices {
                try checks.requireBF16(
                    "model.language_model.layers.\(layer).ple.conv1d",
                    count: hcDim * config.ple.convKernelSize)
            }
            guard manifest.files[Qwen38FlashTensors.pleConstantsFile] != nil else {
                throw ModelError.missingFile(
                    name: Qwen38FlashTensors.pleConstantsFile)
            }
        case .qwen38flashMTP:
            try Self.validateQwen38DraftSchema(checks: checks, quant: quant,
                                               config: config)
        case .qwen36:
            try checks.requireAffine(
                                     "language_model.model.embed_tokens.weight",
                                     rows: config.vocabSize,
                                     columns: config.hiddenSize,
                                     slot: quant.embedding)
            // The untied lm_head is quantized with the embedding slot layout
            // (padded to the same vocab rows). `Model.lmHeadWeightBits` falls
            // back to that slot, so the coupling is validated here — the
            // fallback is only reachable when this check already passed.
            try checks.requireAffine("language_model.lm_head.weight",
                                     rows: config.vocabSize,
                                     columns: config.hiddenSize,
                                     slot: quant.embedding)
        case .qwen36MTP:
            // The MTP sidecar shares the target's embedding and lm_head; it
            // carries only the 2D->D projection and its two input norms.
            try checks.requireAffine("fc.weight",
                                     rows: config.hiddenSize,
                                     columns: 2 * config.hiddenSize,
                                     slot: quant.attention)
            try checks.requireBF16("pre_fc_norm_embedding.weight", count: config.hiddenSize)
            try checks.requireBF16("pre_fc_norm_hidden.weight", count: config.hiddenSize)
        }
        // Resolved through the family's schema: this norm is not always
        // `model.norm`, and not always `hiddenSize` wide. A hyper-connection
        // family collapses its streams through a mixer whose norm spans the
        // full residual.
        try checks.requireBF16(
            TensorSchema.schema(for: config.family).finalNorm,
            count: config.hyperConnections.enabled
                ? config.hiddenSize * config.hyperConnections.count
                : config.hiddenSize)

        try validateLayerSchema(checks: checks, layout: layout,
                                config: config, quant: quant)
    }

    /// Per-layer tensor schema: shapes, dtypes and quant layouts for every
    /// transformer layer, plus the packed-expert layout cross-check.
    private static func validateLayerSchema(
        checks: RuntimeSchemaChecks,
        layout: PackedExpertsLayout,
        config: ArchConfig,
        quant: ManifestQuant
    ) throws {
        // Qwen 3.6 schema, verified against the installed checkpoints:
        // every layer carries the layer norms, the router and the gated
        // shared expert; full-attention layers carry the gate-packed
        // [query; gate] q_proj, and gated-DeltaNet layers carry the
        // linear_attn bundle. The Qwen checkpoints keep no auxiliary
        // sandwich/scale tensors.
        try validateFamilyQuantSupport(config: config, quant: quant)
        try validateLayerTensors(checks: checks, config: config, quant: quant)
        try validateRoutedExpertLayout(checks: checks, layout: layout,
                                       config: config, quant: quant)
    }

    /// Refuse a width no kernel on the path can execute.
    ///
    /// `HyperConnection`, `PLEBlock` and `QSAIndexer` read weights whose width
    /// comes from the attention slot. They took a `DequantInt4GEMV`
    /// unconditionally until `SlotGEMV` gave them both paths, and an 8-bit
    /// install then read half the bytes of every gate as nibbles -- no error,
    /// no noise, just a model that answered " Paris" and degenerated.
    ///
    /// They now dispatch on the slot, so 4 and 8 are both executable and only
    /// a width neither GEMV implements is refused. Kept as a guard rather than
    /// deleted: the failure it catches is silent, and the next width added to
    /// the format will reach these kernels before anyone remembers they exist.
    static func validateFamilyQuantSupport(
        config: ArchConfig,
        quant: ManifestQuant
    ) throws {
        guard config.hyperConnections.enabled else { return }
        guard [4, 8].contains(quant.attention.weightBits) else {
            throw ModelError.unsupportedArchitecture(
                detail: "\(config.family) runs its hyper-connection, PLE and "
                    + "QSA-indexer projections through SlotGEMV, which "
                    + "implements 4- and 8-bit; this install declares "
                    + "\(quant.attention.weightBits)-bit.")
        }
    }

    /// Per-layer norms, router, shared expert, attention and GDN tensors.
    private static func validateLayerTensors(
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant
    ) throws {
        // Names resolve through the family's schema; only shapes are spelled
        // here. A family whose per-sublayer norm is the hyper-connection's
        // spans the whole residual rather than one stream.
        let schema = TensorSchema.schema(for: config.family)
        let blockNormWidth = config.hyperConnections.enabled
            ? config.hiddenSize * config.hyperConnections.count
            : config.hiddenSize
        for layer in 0..<config.numLayers {
            try checks.requireBF16(schema.inputNorm(layer), count: blockNormWidth)
            try checks.requireBF16(schema.postAttnNorm(layer), count: blockNormWidth)
            try checks.requireAffineOrBF16(schema.router(layer),
                                     rows: config.numExperts, columns: config.hiddenSize,
                                     slot: quant.router)
            // The shared-expert scalar gate is quantized at the ROUTER's bit
            // width (8-bit on the target checkpoint, 4-bit on the MTP
            // sidecar), independent of the sharedExpert slot.
            try checks.requireAffineOrBF16(schema.sharedExpertScalarGate(layer),
                                     rows: 1, columns: config.hiddenSize,
                                     slot: quant.router)
            try checks.requireAffine(schema.sharedExpertGate(layer),
                                     rows: config.intermediateSize, columns: config.hiddenSize,
                                     slot: quant.sharedExpert)
            try checks.requireAffine(schema.sharedExpertUp(layer),
                                     rows: config.intermediateSize, columns: config.hiddenSize,
                                     slot: quant.sharedExpert)
            try checks.requireAffine(schema.sharedExpertDown(layer),
                                     rows: config.hiddenSize, columns: config.intermediateSize,
                                     slot: quant.sharedExpert)

            if config.layerIsFull(layer) {
                // Gate-packed [query ; gate] q_proj: 2 * heads * headDim rows.
                let queryDimension = try checks.checkedIntMultiply(
                    2 * config.numHeads, config.fullHeadDim,
                    field: "layer \(layer) query")
                let kvDimension = try checks.checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim,
                    field: "layer \(layer) key/value")
                try checks.requireBF16(schema.qNorm(layer),
                                       count: config.fullHeadDim)
                try checks.requireBF16(schema.kNorm(layer),
                                       count: config.fullHeadDim)
                try checks.requireAffine(schema.qProj(layer),
                                         rows: queryDimension, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.kProj(layer),
                                         rows: kvDimension, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.vProj(layer),
                                         rows: kvDimension, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.oProj(layer),
                                         rows: config.hiddenSize,
                                         columns: config.numHeads * config.fullHeadDim,
                                         slot: quant.attention)
            } else if config.layerIsLinear(layer) {
                let la = config.linearAttention
                try checks.requireAffine(schema.gdnQKV(layer),
                                         rows: la.qkvDim, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.gdnZ(layer),
                                         rows: la.valueDim, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffineOrBF16(schema.gdnA(layer),
                                         rows: la.numVHeads, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffineOrBF16(schema.gdnB(layer),
                                         rows: la.numVHeads, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.gdnOut(layer),
                                         rows: config.hiddenSize, columns: la.valueDim,
                                         slot: quant.attention)
                try checks.requireBF16(schema.gdnConv(layer),
                                       count: la.qkvDim * la.convKernelSize)
                try checks.requireBF16(schema.gdnALog(layer), count: la.numVHeads)
                try checks.requireBF16(schema.gdnDtBias(layer), count: la.numVHeads)
                try checks.requireBF16(schema.gdnNorm(layer),
                                       count: la.valueHeadDim)
            }
        }

    }

    /// Routed-expert tensor shapes cross-checked against the packed layout.
    private static func validateRoutedExpertLayout(
        checks: RuntimeSchemaChecks,
        layout: PackedExpertsLayout,
        config: ArchConfig,
        quant: ManifestQuant
    ) throws {
        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try checks.affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }

}
