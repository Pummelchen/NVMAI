import Foundation
import Metal
import Darwin
import NVMAIFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.gturbo/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    /// unchecked-invariant: all `let`, holding two read-only TensorViews and
    /// their bit widths. @unchecked only because TensorView is.
    struct SharedTargetWeights: @unchecked Sendable {
        let embedding: TensorView
        let lmHead: TensorView
        let embeddingBits: Int
        let lmHeadBits: Int
    }
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var embeddingWeightBits: Int {
        sharedTargetWeights?.embeddingBits ?? manifest.quant?.embedding.weightBits ?? 4
    }
    public var lmHeadWeightBits: Int {
        // Fallback to the embedding slot: qwen36 keeps a separate lm_head, but
        // the repacker quantizes it with the same layout as the embedding
        // (padded to the same vocab rows). `validateRuntimeSchema` checks the
        // lm_head tensor against the embedding slot for qwen36, so the
        // fallback is only reachable when the validator already accepted the
        // coupling.
        sharedTargetWeights?.lmHeadBits ?? manifest.quant?.embedding.weightBits ?? 4
    }
    public var attentionWeightBits: Int { manifest.quant?.attention.weightBits ?? 4 }
    public var routerWeightBits: Int { manifest.quant?.router.weightBits ?? 8 }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }
    public var routedExpertWeightBits: Int { manifest.quant?.routedExpert.weightBits ?? 4 }
    /// The manifest's recorded digest of `model_weights.bin`. The manifest is
    /// itself bound by the install receipt, so this is a trustworthy identity
    /// for anything derived from these weights — the ANE prefill sidecar uses
    /// it to refuse a sidecar exported from a different model.
    public var weightsDigestFromManifest: String? {
        manifest.files["model_weights.bin"]?.sha256
    }
    var mtpResidentTensorBytes: Int { residentBuffer.buffer.length }
    var mtpExpertStrideBytes: Int { Int(packedExpertsLayout.expertStride) }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL
    let modelDirectory: GTurboModelDirectory
    let sharedTargetWeights: SharedTargetWeights?

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue
    let expertIOEventCoordinator: ExpertIOEventCoordinator?

    /// unchecked-invariant: every access goes through `streamersQueue`, the
    /// serial queue on the owning Model. The box exists so Model can stay a
    /// struct while still mutating per-layer streamer state.
    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        /// One staging ring is shared by every lazy layer streamer. Allocating
        /// one per layer would turn a small event bridge into hundreds of MiB
        /// of undeclared working set.
        var metalStagingPool: MetalExpertStagingPool?
        /// Layer files need separate handles, but not separate MTLIO queues.
        /// One queue prevents prefill from exhausting Metal-I/O worker threads.
        var metalIOService: MetalExpertIOService?
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory,
         sharedTargetWeights: SharedTargetWeights? = nil) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.sharedTargetWeights = sharedTargetWeights
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "NVMAI.expert-streamers")
        self.expertIOEventCoordinator = ExpertIOEventCoordinator(device: device)
    }

    // MARK: - Resident accessors
    //
    // Names resolve through the family's TensorSchema (Runtime/Family/): a
    // family with different naming supplies a schema file; these accessors
    // never change.

    var schema: TensorSchema { TensorSchema.schema(for: config.family) }

    public func embedding() throws -> TensorView {
        if let sharedTargetWeights { return sharedTargetWeights.embedding }
        return try resident(name: schema.embedding)
    }

    /// Qwen 3.6 carries a separate `lm_head` tensor. The transpose for the
    /// lm_head GEMV path is the kernel's job, not the loader's.
    public func lmHead() throws -> TensorView {
        if let sharedTargetWeights { return sharedTargetWeights.lmHead }
        if config.tieWordEmbeddings { return try embedding() }
        return try resident(name: schema.lmHead)
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.qProj(L))
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.kProj(L))
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.vProj(L))
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.oProj(L))
    }
    public func router(layer L: Int) throws -> TensorView {
        try resident(name: schema.router(L))
    }
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertGate(L))
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertUp(L))
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertDown(L))
    }
    /// Qwen3.5-MoE scalar gate on the shared-expert branch: a `[1, hidden]`
    /// 8-bit projection whose sigmoid multiplies the shared FFN output.
    public func sharedExpertScalarGate(layer L: Int) throws -> TensorView {
        try resident(name: schema.sharedExpertScalarGate(L))
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.inputNorm(L))
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.postAttnNorm(L))
    }
    public func finalNorm() throws -> TensorView {
        try resident(name: schema.finalNorm)
    }

    /// MTP projection over the normalized next-token embedding followed by the
    /// normalized target hidden state: `[embedding, hidden]`, `[2D] -> [D]`.
    public func mtpProjection() throws -> TensorView {
        return try resident(name: "fc.weight")
    }
    public func mtpEmbeddingNorm() throws -> TensorView {
        return try resident(name: "pre_fc_norm_embedding.weight")
    }
    public func mtpHiddenNorm() throws -> TensorView {
        return try resident(name: "pre_fc_norm_hidden.weight")
    }

    /// Attach a native MTP sidecar to a target without copying either large
    /// tensor. The returned model retains the target's Metal buffers and uses
    /// its actual 4/6/8-bit head kernels.
    public func sharingTargetWeights(from target: Model) throws -> Model {
        guard config.family == .qwen36MTP,
              target.config.family == .qwen36,
              config.hiddenSize == target.config.hiddenSize,
              config.vocabSize == target.config.vocabSize,
              Self.mtpLineagesAreCompatible(sidecarID: modelID,
                                             targetID: target.modelID) else {
            throw ModelError.indexCorrupt(
                detail: "MTP sidecar is incompatible with the target model")
        }
        return Model(device: device,
                     config: config,
                     streamingMode: streamingMode,
                     expertCachePolicy: expertCachePolicy,
                     integrityPolicy: integrityPolicy,
                     residentBuffer: residentBuffer,
                     residentIndex: residentIndex,
                     packedExpertsLayout: packedExpertsLayout,
                     manifest: manifest,
                     directoryURL: directoryURL,
                     modelDirectory: modelDirectory,
                     sharedTargetWeights: SharedTargetWeights(
                        embedding: try target.embedding(),
                        lmHead: try target.lmHead(),
                        embeddingBits: target.embeddingWeightBits,
                        lmHeadBits: target.lmHeadWeightBits))
    }

    /// The Qwen3.5-MoE tensor contract is shared by Qwen 3.6 and Ornith 1.5,
    /// but their trained embeddings and heads are not interchangeable. Keep
    /// synthetic and privately named compatible checkpoints usable while
    /// rejecting a known cross-model pairing before any generation begins.
    static func mtpLineagesAreCompatible(sidecarID: String,
                                         targetID: String) -> Bool {
        func lineage(_ modelID: String) -> String? {
            let normalized = modelID.lowercased()
            if normalized.contains("ornith-1.5") { return "ornith-1.5" }
            if normalized.contains("qwen3.6") || normalized.hasPrefix("qwen-") {
                return "qwen3.6"
            }
            return nil
        }
        guard let sidecar = lineage(sidecarID),
              let target = lineage(targetID) else {
            return true
        }
        return sidecar == target
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.qNorm(L))
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.kNorm(L))
    }

    // MARK: - Gated-DeltaNet linear attention (Qwen 3.6)
    //
    // Layers whose mask value is 2 replace full/sliding attention with the
    // gated delta rule. Projections are 4/6/8-bit affine; the depthwise conv
    // weight, A_log, dt_bias, and the gated output norm are BF16.

    public func linearInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnQKV(L))
    }
    public func linearInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnZ(L))
    }
    public func linearInProjA(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnA(L))
    }
    public func linearInProjB(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnB(L))
    }
    public func linearOutProj(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnOut(L))
    }
    /// Depthwise causal conv weight, source shape `[convDim, kernel, 1]`, BF16.
    public func linearConv1d(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnConv(L))
    }
    /// Per-value-head decay base, shape `[numVHeads]`, BF16.
    public func linearALog(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnALog(L))
    }
    /// Per-value-head dt bias, shape `[numVHeads]`, BF16.
    public func linearDtBias(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnDtBias(L))
    }
    /// Gated RMSNorm weight over the value head dim, shape `[valueHeadDim]`.
    public func linearNorm(layer L: Int) throws -> TensorView {
        try resident(name: schema.gdnNorm(L))
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        let relativeOffset = entry.fileOffset - residentFileOffset
        let scaleRel: UInt64 = entry.scaleSize > 0
            ? entry.scaleOffset - residentFileOffset : 0
        let biasRel: UInt64 = entry.biasSize > 0
            ? entry.biasOffset - residentFileOffset : 0
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        // The streamer is per-layer: `openLayerLocked(L)` bound it to layer
        // L's file with `expertOffsets = layers[L].experts.map(\.offset)`, and
        // `StreamLayout.expertOffset(layer: 0, ...)` is the branch that
        // consults that per-layer offset table. Passing the actual layer here
        // would select the dense cross-layer formula and mis-offset every
        // expert on layers above 0 — layer 0 is intentional.
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: 0)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    /// The open is retried synchronously by `ensureLayerOpened(_:)` before any
    /// expert fetch on the layer, which rethrows the identical error — so a
    /// failure here is never dropped end-to-end.
    ///
    /// `nonisolated(unsafe)` is required because `Model` is not formally
    /// `Sendable`; the capture is safe because every mutable member
    /// (`streamersBox`) is confined behind the serial `streamersQueue` and the
    /// remaining members are immutable values.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            do {
                try model.openLayerLocked(L)
            } catch {
                // Deferred: the synchronous `ensureLayerOpened(L)` that
                // precedes every expert fetch on this layer performs the same
                // idempotent open and rethrows this error to the prefill loop.
                // Nothing is silently lost; the async path only overlaps the
                // SHA-256 verification with the chunk's GPU work.
            }
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let manifestRel = "packed_experts/\(basename)"
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: layerFD,
                                              named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
            streamersBox.layerVerified[L] = true
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        let metalStagingPool: MetalExpertStagingPool?
        let metalIOService: MetalExpertIOService?
        if try ExpertIOBackend.environmentValue() == .metal {
            if streamersBox.metalStagingPool == nil {
                streamersBox.metalStagingPool = try MetalExpertStagingPool(
                    device: device,
                    byteCount: Int(packedExpertsLayout.expertStride),
                    // Decode routes at most top-8 experts. A single exclusive
                    // lease keeps native MTLIO shared-event values ordered.
                    slotCapacity: 8)
            }
            if streamersBox.metalIOService == nil {
                streamersBox.metalIOService = try MetalExpertIOService(
                    device: device, maximumCommandsInFlight: 4)
            }
            metalStagingPool = streamersBox.metalStagingPool
            metalIOService = streamersBox.metalIOService
        } else {
            metalStagingPool = nil
            metalIOService = nil
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy,
            eventCoordinator: expertIOEventCoordinator,
            metalStagingPool: metalStagingPool,
            metalIOService: metalIOService)
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

    /// Wire the routed-expert slot cache for decode, or release it for
    /// prefill.
    ///
    /// Decode is the phase where a reclaimed slot page costs an SSD read on
    /// the critical path, so that is the phase worth wiring. Prefill streams
    /// experts in bulk and instead needs the headroom -- holding the cache
    /// wired throughout measurably slowed ANE prefill, which has to place
    /// Core ML arenas alongside it. Called at the phase boundaries; cheap and
    /// idempotent, since each streamer skips a state it is already in.
    public func setExpertCachePinned(_ pinned: Bool) {
        streamersQueue.sync {
            for streamer in streamersBox.streamers {
                streamer?.setSlotsPinned(pinned)
            }
        }
    }

}

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

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }

        let checks = RuntimeSchemaChecks(residentIndex: residentIndex, quant: quant)

        switch config.family {
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
        try validateLayerTensors(checks: checks, config: config, quant: quant)
        try validateRoutedExpertLayout(checks: checks, layout: layout,
                                       config: config, quant: quant)
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
            try checks.requireAffine(schema.router(layer),
                                     rows: config.numExperts, columns: config.hiddenSize,
                                     slot: quant.router)
            // The shared-expert scalar gate is quantized at the ROUTER's bit
            // width (8-bit on the target checkpoint, 4-bit on the MTP
            // sidecar), independent of the sharedExpert slot.
            try checks.requireAffine(schema.sharedExpertScalarGate(layer),
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
                try checks.requireAffine(schema.gdnA(layer),
                                         rows: la.numVHeads, columns: config.hiddenSize,
                                         slot: quant.attention)
                try checks.requireAffine(schema.gdnB(layer),
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

/// The schema checks `validateRuntimeSchema` runs, bound to the index and
/// quant slots they read. Extracted from that function so the per-family and
/// per-layer rules below read as rules rather than as one 250-line body.
private struct RuntimeSchemaChecks {
    let residentIndex: ResidentIndex
    let quant: ManifestQuant

    func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
        }
        return value
    }

    func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
        }
        return value
    }

    func entry(_ name: String) throws -> ResidentIndexEntry {
        guard let e = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return e
    }

    func requireBF16(_ name: String, count: Int) throws {
        let e = try entry(name)
        guard e.dtype == 1 else {
            throw ModelError.indexCorrupt(detail: "\(name) is not BF16")
        }
        // Trailing zero dims encode a lower-rank tensor (e.g. a [2048]
        // vector is stored as shape (2048, 0, 0, 0)); treat them as 1.
        let dims = [e.shape.0, e.shape.1, e.shape.2, e.shape.3]
        let elements = dims.reduce(1) { $0 * ($1 == 0 ? 1 : Int($1)) }
        guard elements == count else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: UInt64(count), actual: UInt64(elements))
        }
    }

    func requireAffine(_ name: String, rows: Int, columns: Int,
                       slot: ManifestQuantSlot) throws {
        let e = try entry(name)
        guard columns % slot.groupSize == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) columns \(columns) not divisible by group size \(slot.groupSize)")
        }
        // Bit-packed affine weights: rows*cols*bits must pack into whole
        // bytes (4-bit packs 2/byte, 6-bit packs across 32-bit words).
        let elementBits = try checkedMultiply(
            UInt64(rows) * UInt64(columns), UInt64(slot.weightBits),
            field: name)
        guard elementBits % 8 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) \(slot.weightBits)-bit layout does not pack into whole bytes")
        }
        let weightBytes = elementBits / 8
        let auxBytes = try checkedMultiply(
            UInt64(rows) * UInt64(columns / slot.groupSize), 2, field: name)
        guard e.dtype == 0,                       // U32-packed weights
              e.sizeBytes == weightBytes,
              e.scaleOffset > 0, e.scaleSize == auxBytes,
              e.biasOffset > 0, e.biasSize == auxBytes else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: weightBytes, actual: e.sizeBytes)
        }
    }

    func affineSizes(rows: Int, columns: Int, slot: ManifestQuantSlot,
                     field: String) throws -> (weight: UInt64, aux: UInt64, shape: (UInt32, UInt32)) {
        guard columns % slot.groupSize == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(field) has an invalid quant layout (group \(slot.groupSize), \(slot.weightBits)-bit)")
        }
        let elementBits = try checkedMultiply(
            UInt64(rows) * UInt64(columns), UInt64(slot.weightBits),
            field: field)
        guard elementBits % 8 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(field) \(slot.weightBits)-bit layout does not pack into whole bytes")
        }
        let weightBytes = elementBits / 8
        let auxBytes = try checkedMultiply(
            UInt64(rows) * UInt64(columns / slot.groupSize), 2, field: field)
        return (weightBytes, auxBytes, (UInt32(rows), UInt32(columns)))
    }
}
