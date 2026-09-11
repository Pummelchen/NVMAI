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

    /// True when the GDN `in_proj_a` / `in_proj_b` pair was promoted to the
    /// checkpoint's bf16. They travel together -- both are `numVHeads` rows of
    /// the same projection -- so one probe decides the pair.
    public var gdnABIsBF16: Bool {
        guard let view = try? linearInProjA(layer: 0) else { return false }
        return view.dtype == 1
    }

    /// The width the router GEMV must actually be built for.
    ///
    /// The manifest slot says how the slot is stored; a family can be promoted
    /// to the checkpoint's own bf16 inside it, and then the tensor's dtype is
    /// what the kernel has to match. 16 means unquantized -- the shader reads
    /// bfloat directly and ignores the scale and bias companions, which a
    /// promoted tensor does not have.
    ///
    /// Read from the tensor rather than the slot because that is the thing
    /// that can differ: getting it from the slot is exactly the mistake the
    /// INT4-only kernels made.
    public var effectiveRouterWeightBits: Int {
        guard let view = try? router(layer: 0) else { return routerWeightBits }
        return view.dtype == 1 ? 16 : routerWeightBits
    }
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
    /// Mirrors `RealForwardRunner.keepExpertCacheWired` (NVMAI_KEEP_WIRED=1).
    static let keepExpertCacheWired = ProcessInfo.processInfo.environment["NVMAI_KEEP_WIRED"] == "1"

    /// unchecked-invariant: every member, including the wiring flags and
    /// the pin diagnostics added in 5.0.3, is read and written only inside
    /// `streamersQueue.sync` / `.async` blocks; the queue is the lock.
    final class StreamersBox: @unchecked Sendable {
        /// Overrides the configured slot count for layers opened while set.
        var concentratedSlotCount: Int?
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        /// True once every opened streamer's slots are wired (see
        /// `setExpertCachePinned`); cleared by any unpin or partial wire.
        var pinnedComplete = false
        /// Wire each layer as it opens (profile `keepExpertCacheWired` or
        /// NVMAI_KEEP_WIRED=1).
        var keepWired = false
        /// Cache layout for layers opened from now on; nil takes the
        /// environment's value (see `ModelProfile.earlyExpertHits`).
        var cacheLayoutOverride: ExpertCacheLayout?
        /// Diagnostic: time spent waiting to enter the serial queue in
        /// `setExpertCachePinned`.
        var pinQueueWaitNanos: UInt64 = 0
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
    // MARK: Hyper-connection (Gated Residual) weights
    //
    // Present only on families whose `hyperConnections` is enabled; asking for
    // them elsewhere fails at the resident-index lookup with the missing name,
    // which is the right error for a misconfigured install.

    public func hcAttnMixDown(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.attnMixDown(L))
    }
    public func hcAttnMixUp(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.attnMixUp(L))
    }
    public func hcAttnInject(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.attnInject(L))
    }
    public func hcMlpMixDown(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.mlpMixDown(L))
    }
    // MARK: QSA indexer

    public func indexerQProj(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.indexerQProj(L))
    }
    public func indexerKProj(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.indexerKProj(L))
    }
    public func indexerQNorm(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.indexerQNorm(L))
    }
    public func indexerKNorm(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.indexerKNorm(L))
    }

    // MARK: PLE n-gram block

    public func pleKeyProj(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleKeyProj(L))
    }
    public func pleValueProj(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleValueProj(L))
    }
    public func pleNormKey(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleNormKey(L))
    }
    public func pleNormQuery(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleNormQuery(L))
    }
    public func pleNormConv(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleNormConv(L))
    }
    public func pleConv(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.pleConv(L))
    }

    public func hcMlpMixUp(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.mlpMixUp(L))
    }
    public func hcMlpInject(layer L: Int) throws -> TensorView {
        try resident(name: Qwen38FlashTensors.mlpInject(L))
    }
    /// The model-level mixer that collapses the streams before `lm_head`.
    /// Same read gate as a sublayer's, with no inject.
    public func hcMixerDown() throws -> TensorView {
        try resident(name: Qwen38FlashTensors.mixerDown)
    }
    public func hcMixerUp() throws -> TensorView {
        try resident(name: Qwen38FlashTensors.mixerUp)
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

    // MARK: Qwen3.8-Flash-Next draft head
    //
    // This family fuses with two projections rather than one over a
    // concatenation: `fc_hidden` takes the target's wide residual and
    // `fc_embedding` takes the next token's embedding, and their outputs are
    // summed. Two [2560, 2560] matrices, not one [2560, 5120] -- the shapes
    // are what say so, and getting it wrong would show up only as a draft
    // that is never accepted.

    /// `[hidden, hc_dim]`: the target's wide residual down to one stream.
    public func mtpHiddenProjection() throws -> TensorView {
        try resident(name: "fc_hidden.weight")
    }
    /// `[hidden, hidden]`: the next token's embedding.
    public func mtpEmbeddingProjection() throws -> TensorView {
        try resident(name: "fc_embedding.weight")
    }
    /// Grouped over the hyper-connection streams, unlike the embedding norm.
    /// Stored already folded (+1) by the MLX conversion, like every other
    /// gamma in this checkpoint; used as-is.
    public func mtpWideNorm() throws -> TensorView {
        try resident(name: "pre_fc_norm_hidden")
    }
    public func mtpTokenNorm() throws -> TensorView {
        try resident(name: "pre_fc_norm_embedding")
    }

    /// The Qwen3.8-Flash-Next draft head's own tensors.
    ///
    /// It has no embedding, no head and no n-gram block: it borrows the first
    /// two from the target it drafts for and does not have the third. What is
    /// its own is the single decoder layer and the fusion pair, so that is
    /// what gets checked.
    static func validateQwen38DraftSchema(
        checks: RuntimeSchemaChecks,
        quant: ManifestQuant,
        config: ArchConfig
    ) throws {
        let hcDim = config.hiddenSize * config.hyperConnections.count
        try checks.requireBF16(
            "model.language_model.hyper_connection_mixer.hc_norm", count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.attn_hyper_connection.hc_norm",
            count: hcDim)
        try checks.requireBF16(
            "model.language_model.layers.0.mlp_hyper_connection.hc_norm",
            count: hcDim)
        // Without both projections this is not a draft head, it is a
        // detached layer.
        try checks.requireBF16("pre_fc_norm_hidden", count: hcDim)
        try checks.requireBF16("pre_fc_norm_embedding", count: config.hiddenSize)
        // [D, D], not [D, hc_dim]: the wide residual is mean-collapsed to one
        // stream before this projection sees it. The reference's prose says
        // otherwise and its own code works out that the prose cannot be right;
        // the tensor shape settles it.
        try checks.requireAffine("fc_hidden.weight",
                                 rows: config.hiddenSize,
                                 columns: config.hiddenSize,
                                 slot: quant.attention)
        try checks.requireAffine("fc_embedding.weight",
                                 rows: config.hiddenSize,
                                 columns: config.hiddenSize,
                                 slot: quant.attention)
    }

    /// Attach a native MTP sidecar to a target without copying either large
    /// tensor. The returned model retains the target's Metal buffers and uses
    /// its actual 4/6/8-bit head kernels.
    public func sharingTargetWeights(from target: Model) throws -> Model {
        let pairing = (config.family, target.config.family)
        let familiesMatch = pairing == (.qwen36MTP, .qwen36)
            || pairing == (.qwen38flashMTP, .qwen38flash)
        guard familiesMatch,
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
            // Checked before the generic qwen3.6 prefix so the flash lineage
            // is not swallowed by it.
            if normalized.contains("qwen3.8-flash-next") { return "qwen3.8-flash-next" }
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

    /// Slots to give a layer opened from here, overriding the configured
    /// count. Set only by layer-major prefill, which can afford a large cache
    /// because it keeps one layer live at a time. Stored on the streamers box
    /// because `Model` is a value type and the runner holds it by `let`.
    public var concentratedSlotCount: Int? {
        get { streamersQueue.sync { streamersBox.concentratedSlotCount } }
        nonmutating set { streamersQueue.sync { streamersBox.concentratedSlotCount = newValue } }
    }

    /// Drop a layer's expert cache. Safe once that layer's work is finished:
    /// the next use reopens it lazily, which is how it was created.
    public func releaseLayerStreamer(_ L: Int) {
        streamersQueue.sync {
            guard L >= 0, L < streamersBox.streamers.count else { return }
            streamersBox.streamers[L] = nil
            streamersBox.layerVerified[L] = false
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
        // Checked at the one place this product is formed. `StreamLayout.
        // expertOffset` multiplies `perLayer` again on every expert read
        // (`@inline(always)`, recomputed per call, on the decode path), so the
        // product is validated here rather than there -- a wrapped one would make
        // `streamSize` small, pass the streamer's own file-size check, and leave
        // every derived offset pointing outside the layer file.
        let (streamSize, streamSizeOverflow) = UInt64(packedExpertsLayout.expertsPerLayer)
            .multipliedReportingOverflow(by: packedExpertsLayout.expertStride)
        guard !streamSizeOverflow else {
            throw ModelError.internalInconsistency(
                detail: "packed expert layer \(L) declares \(packedExpertsLayout.expertsPerLayer) "
                    + "experts of \(packedExpertsLayout.expertStride) bytes, which overflows "
                    + "the stream size")
        }
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
        // Layer-major prefill finishes a layer entirely before the next, so
        // only one layer's cache has to exist at a time. That inverts the
        // budget: 512 slots for one layer is 1.3 GiB, against 96 slots x 48
        // layers at 11.9 GiB today -- less memory, and enough to hold a
        // layer's whole ~512-expert working set instead of thrashing it. The
        // measured 47,423 reloads against 65,234 misses is that thrashing.
        let concentratedSlots = streamersBox.concentratedSlotCount
        let effectiveSlotCount = concentratedSlots ?? slotCount
        let metalStagingPool: MetalExpertStagingPool?
        let metalIOService: MetalExpertIOService?
        if try ExpertIOBackend.environmentValue() == .metal {
            if streamersBox.metalStagingPool == nil {
                streamersBox.metalStagingPool = try MetalExpertStagingPool(
                    device: device,
                    byteCount: Int(packedExpertsLayout.expertStride),
                    // One staging slot per routed expert: a layer can miss all
                    // of them, and tryAcquire fails the whole request if the
                    // ring is short. This was hardcoded to 8 for the top-8
                    // families, which made the MTLIO + event path unreachable
                    // on Qwen3.8-Flash-Next -- it routes top-10, so any layer
                    // missing nine or more failed with "staging ring is
                    // unavailable" and the combination could never be measured.
                    slotCapacity: config.topKExperts)
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
        let layoutOverride = streamersBox.cacheLayoutOverride
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: effectiveSlotCount,
            cachePolicy: expertCachePolicy,
            cacheLayout: layoutOverride,
            eventCoordinator: expertIOEventCoordinator,
            metalStagingPool: metalStagingPool,
            metalIOService: metalIOService)
        // A newly opened layer is not wired yet; the next pin walks again.
        // With NVMAI_KEEP_WIRED=1 it is wired here, so a cache that is never
        // unpinned is never swapped out and the first decode token does not
        // pay to fault it back (measured 1.6-4.7 s per request on Qwen3.8).
        streamersBox.pinnedComplete = false
        if Self.keepExpertCacheWired || streamersBox.keepWired {
            streamersBox.streamers[L]?.setSlotsPinned(true)
        }
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
    /// Open later layers with this cache layout (see
    /// `ModelProfile.earlyExpertHits`, which needs the pooled one).
    public func setExpertCacheLayout(_ layout: ExpertCacheLayout) {
        streamersQueue.sync { streamersBox.cacheLayoutOverride = layout }
    }

    /// Wire layers as they open from now on (see `ModelProfile.keepExpertCacheWired`).
    public func setKeepExpertCacheWired(_ keep: Bool) {
        streamersQueue.sync { streamersBox.keepWired = keep }
    }

    public var expertCachePinQueueWaitNanos: UInt64 {
        streamersQueue.sync { streamersBox.pinQueueWaitNanos }
    }

    /// Returns true when every opened streamer is in the requested state.
    @discardableResult
    public func setExpertCachePinned(_ pinned: Bool) -> Bool {
        // Every decode step asks for the cache to be wired. Once every
        // streamer reports it is, there is nothing to do, and walking all of
        // them under the serial queue per token measured 73-91 ms on a
        // 48-layer model (NVMAI_RUNNER_STATS pre_pin_ms). The walk only runs
        // again after an unpin or a partial wire.
        let tEnter = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var walked = 0
        var unpinnedAfter = 0
        var earlyReturn = false
        var walkNanos: UInt64 = 0
        streamersQueue.sync {
            streamersBox.pinQueueWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
            if pinned, streamersBox.pinnedComplete { earlyReturn = true; return }
            let tWalk = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var complete = pinned
            for streamer in streamersBox.streamers {
                guard let streamer else { continue }
                walked += 1
                streamer.setSlotsPinned(pinned)
                if pinned, !streamer.isPinned { complete = false; unpinnedAfter += 1 }
            }
            walkNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWalk
            streamersBox.pinnedComplete = complete
        }
        let total = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEnter
        if PreadExpertStreamer.wireTraceEnabled, total > 2_000_000 {
            FileHandle.standardError.write(Data(
                "[wire] setExpertCachePinned(\(pinned)) \(Double(total) / 1e6) ms early=\(earlyReturn) walked=\(walked) walk_ms=\(Double(walkNanos) / 1e6) unpinned_after=\(unpinnedAfter)\n".utf8))
        }
        return streamersQueue.sync { streamersBox.pinnedComplete } == pinned
    }

}


/// The schema checks `validateRuntimeSchema` runs, bound to the index and
/// quant slots they read. Extracted from that function so the per-family and
/// per-layer rules below read as rules rather than as one 250-line body.
struct RuntimeSchemaChecks {
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

    /// Accepts a tensor that is either affine-quantized at the slot's width or
    /// kept at the checkpoint's own bf16.
    ///
    /// Promotion is per tensor, so a family can be unquantized inside a slot
    /// that is not. Checking only the slot would refuse a correct install; not
    /// checking at all would let a wrong one through, which for these tensors
    /// is silent -- the kernels pick their reading from the same dtype.
    func requireAffineOrBF16(_ name: String, rows: Int, columns: Int,
                             slot: ManifestQuantSlot) throws {
        let e = try entry(name)
        if e.dtype == 1 {
            try requireBF16(name, count: rows * columns)
            return
        }
        try requireAffine(name, rows: rows, columns: columns, slot: slot)
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
