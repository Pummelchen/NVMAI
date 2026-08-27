import CoreML
import Foundation
import Metal

/// Opt-in switch for the ANE prefill attention path (Track A).
///
/// `on` routes every full-attention layer's prefill attention block through a
/// Core ML sidecar exported by `tools/export_ane_prefill.py`, leaving GDN
/// layers, the MoE, the KV cache format, and all of decode untouched. Output
/// is NOT byte-identical to the GPU path — the sidecar computes in fp16 with
/// a different reduction order (measured ~1% per-layer mean deviation against
/// an fp32 reference) — so this ships like every other experiment here: off
/// by default, qualified on its own evidence, never silently selected.
public enum RuntimePrefillANE: String, Codable, Sendable {
    case off
    case on

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimePrefillANE {
        guard let raw = environment["NVMAI_PREFILL_ANE"] else { return .off }
        guard let value = RuntimePrefillANE(rawValue: raw) else {
            throw PrefillError.chunkedUnsupported(
                "unsupported NVMAI_PREFILL_ANE '\(raw)'; allowed: off, on")
        }
        return value
    }
}

/// Transfers a loaded `MLModel` out of a background load task.
///
/// unchecked-invariant: the box is created inside the loading task, handed to
/// exactly one awaiting consumer, and never mutated; the prefill loop that
/// consumes it is single-flight, so no two threads ever hold the same model.
private struct LoadedModelBox: @unchecked Sendable {
    let model: MLModel
}

/// Runs the full-attention prefill block on the Neural Engine.
///
/// One multifunction `.mlpackage` per full-attention layer holds functions
/// `h0, h4096, ...` sharing a single fp16 weight set; the function name is
/// the KV-history length, which in this runtime is always the chunk-aligned
/// `startPosition`. The block consumes the post-input-norm hidden chunk and a
/// token-major fp16 K/V history, and produces the attention branch output
/// plus the chunk's cache-layout K/V — the same bytes the GPU path stages
/// before quantizing into the cache, so `copyPrefillKVToCache` is reused
/// verbatim and decode sees an ordinary cache.
///
/// Design constants mirror the exporter and are validated against its
/// manifest at load; a mismatch fails closed rather than computing nonsense.
///
/// unchecked-invariant: driven exclusively by the single-flight prefill loop
/// of one runner; buffers and lazy caches are never touched concurrently.
final class ANEPrefillAttention: @unchecked Sendable {
    struct SidecarMetadata: Decodable {
        let version: Int
        let family: String
        let chunkTokens: Int
        let histories: [Int]
        let layers: [Int]
        /// SHA-256 of the `model_weights.bin` the sidecar was exported from,
        /// copied out of that model's install receipt at export time.
        let weightsSha256: String?
    }

    static let expectedVersion = 1
    /// -30000 underflows fp16 exp() exactly like -inf without putting
    /// infinity arithmetic into the ANE graph (whose fused SDPA op NaNs).
    static let maskNegative = Float16(-30000)

    let chunkTokens: Int
    let histories: Set<Int>
    let coveredLayers: Set<Int>
    let maxPromptTokens: Int

    /// Shared-mode staging: the GPU blits `normed` in, Core ML writes the
    /// three outputs back via output backings, the GPU quantizes K/V into the
    /// cache from the same memory, and the shadow append memcpys from it.
    let stagingNormed: MTLBuffer
    let stagingOut: MTLBuffer
    let stagingK: MTLBuffer
    let stagingV: MTLBuffer

    private let hiddenSize: Int
    private let kvDim: Int
    private let packageDir: URL
    private let compiledDir: URL
    /// At most one loaded model at a time. Each loaded function pins an
    /// E5RT/ANE inference arena (the h4096 variant's score tensors alone are
    /// ~1 GB); keeping 20 of them resident during a long prefill pressured
    /// the 8 GiB expert slot cache out of RAM and collapsed the decode that
    /// followed to ~2 tok/s. One-at-a-time bounds the ANE footprint to a
    /// single context at ~0.5 s reload cost per layer-chunk.
    private var residentModel: (layer: Int, history: Int, model: MLModel)?
    /// In-flight load of the *next* layer's model, started as soon as this
    /// layer's prediction returns so the ~0.5 s load overlaps the GPU's MoE
    /// stage instead of serializing in front of the next prediction. At most
    /// one is outstanding, which keeps the one-resident-arena rule intact:
    /// the preloaded model only becomes resident when `model(layer:history:)`
    /// adopts it, and that is the same moment the previous one is dropped.
    private var preloaded: (layer: Int, history: Int, task: Task<LoadedModelBox, Error>)?
    private var masks: [Int: MLMultiArray] = [:]
    private var maskStorage: [Int: UnsafeMutableRawPointer] = [:]
    /// Token-major fp16 K/V rows per layer, at absolute prompt positions, so
    /// later chunks can attend to exact-precision history without
    /// re-dequantizing the cache. Allocated on the first append (single-chunk
    /// prompts never pay for it) and reused across requests.
    private var shadowK: [Int: UnsafeMutableRawPointer] = [:]
    private var shadowV: [Int: UnsafeMutableRawPointer] = [:]
    private(set) var shadowTokens = 0
    private var loggedFallback = false

    /// - Parameter weightsSha256: the model's own recorded `model_weights.bin`
    ///   digest, taken from its install receipt. A sidecar exported from
    ///   different weights computes plausible-looking but wrong attention, and
    ///   nothing downstream would catch it — so the binding is checked here
    ///   and fails closed. Nil skips the check (no receipt available) and says
    ///   so, rather than silently trusting.
    init(modelDirectory: URL, device: MTLDevice,
         hiddenSize: Int, kvDim: Int, weightsSha256: String?) throws {
        let dir = modelDirectory.appendingPathComponent("ane_prefill")
        let metaURL = dir.appendingPathComponent("ane_prefill.json")
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw PrefillError.chunkedUnsupported(
                "NVMAI_PREFILL_ANE=on but \(metaURL.path) is missing; run "
                + "tools/export_ane_prefill.py for this model first")
        }
        let meta = try JSONDecoder().decode(
            SidecarMetadata.self, from: Data(contentsOf: metaURL))
        guard meta.version == Self.expectedVersion else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar version \(meta.version) != supported \(Self.expectedVersion); re-export")
        }
        guard meta.family == "qwen36" else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar family '\(meta.family)' is not qwen36")
        }
        if let weightsSha256, let exported = meta.weightsSha256 {
            guard exported.lowercased() == weightsSha256.lowercased() else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar was exported from different weights "
                    + "(sidecar \(exported.prefix(12))..., model "
                    + "\(weightsSha256.prefix(12))...); re-export it for this model")
            }
        } else {
            print("NVMAI ane-prefill: sidecar/weights binding unverified "
                  + "(no receipt digest available); a stale sidecar would not "
                  + "be detected")
        }
        self.chunkTokens = meta.chunkTokens
        self.histories = Set(meta.histories)
        self.coveredLayers = Set(meta.layers)
        self.maxPromptTokens = (meta.histories.max() ?? 0) + meta.chunkTokens
        self.hiddenSize = hiddenSize
        self.kvDim = kvDim
        self.packageDir = dir
        self.compiledDir = dir.appendingPathComponent("compiled-v\(meta.version)")
        try FileManager.default.createDirectory(
            at: compiledDir, withIntermediateDirectories: true)

        let halfBytes = MemoryLayout<Float16>.stride
        func staging(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: elements * halfBytes,
                                               options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = label
            return made
        }
        self.stagingNormed = try staging(meta.chunkTokens * hiddenSize,
                                         "ane.staging.normed")
        self.stagingOut = try staging(meta.chunkTokens * hiddenSize,
                                      "ane.staging.out")
        self.stagingK = try staging(meta.chunkTokens * kvDim, "ane.staging.k")
        self.stagingV = try staging(meta.chunkTokens * kvDim, "ane.staging.v")
    }

    deinit {
        for pointer in maskStorage.values { pointer.deallocate() }
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
    }

    /// Whether this chunk can run on the ANE. Continuity matters: a chunk at
    /// a nonzero start needs the shadow rows of every earlier chunk, so a
    /// resumed or partially GPU-processed prefill falls back for the rest of
    /// the request instead of attending to a hole.
    func eligibleChunk(startPosition: Int, tokenCount: Int,
                       configChunkTokens: Int) -> Bool {
        // A short prompt is one partial chunk; padding it to 4,096 costs
        // ~2 s of ANE work against under a second on the GPU, so the ANE
        // serves only full chunks and the continuation chunks of long
        // prompts — the workload it wins by 26x.
        let fullOrContinuation = tokenCount == chunkTokens || startPosition > 0
        guard configChunkTokens == chunkTokens,
              fullOrContinuation,
              tokenCount <= chunkTokens,
              startPosition % chunkTokens == 0,
              histories.contains(startPosition) else {
            if !loggedFallback {
                loggedFallback = true
                print("NVMAI ane-prefill fallback: chunk at \(startPosition) "
                      + "(+\(tokenCount)) outside sidecar coverage "
                      + "(chunk \(chunkTokens), max prompt \(maxPromptTokens)); "
                      + "using the GPU path")
            }
            return false
        }
        if startPosition == 0 {
            shadowTokens = 0
            return true
        }
        return shadowTokens == startPosition
    }

    private func model(layer: Int, history: Int) async throws -> MLModel {
        if let cached = residentModel,
           cached.layer == layer, cached.history == history {
            return cached.model
        }
        if let pending = preloaded, pending.layer == layer,
           pending.history == history {
            preloaded = nil
            // Drop the old arena only once the new model is in hand, then
            // adopt it — never two resident at once for longer than the
            // handover itself.
            let loaded = try await pending.task.value.model
            residentModel = (layer, history, loaded)
            return loaded
        }
        // A preload for a different layer is now useless; await and discard it
        // rather than leaking an arena behind the resident one.
        if let stale = preloaded {
            preloaded = nil
            _ = try? await stale.task.value
        }
        residentModel = nil
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        let loaded = try MLModel(contentsOf: compiled,
                                 configuration: configuration)
        residentModel = (layer, history, loaded)
        return loaded
    }

    /// The on-disk compiled model for `layer`, compiling it from the package
    /// on first use and whenever the package is newer.
    private func compiledModelURL(layer: Int) async throws -> URL {
        let package = packageDir.appendingPathComponent("layer_\(layer).mlpackage")
        let compiled = compiledDir.appendingPathComponent("layer_\(layer).mlmodelc")
        let fm = FileManager.default
        func modifiedDate(_ url: URL) -> Date {
            (try? fm.attributesOfItem(atPath: url.path)[.modificationDate]
             as? Date) ?? .distantPast
        }
        if !fm.fileExists(atPath: compiled.path)
            || modifiedDate(compiled) < modifiedDate(package) {
            guard fm.fileExists(atPath: package.path) else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar is missing \(package.lastPathComponent)")
            }
            let temporary = try await MLModel.compileModel(at: package)
            _ = try? fm.removeItem(at: compiled)
            try fm.moveItem(at: temporary, to: compiled)
        }
        return compiled
    }

    /// The compile-and-load half of `model(layer:history:)`, without touching
    /// `residentModel` — safe to run detached for a preload.
    private func loadModel(layer: Int, history: Int) async throws -> MLModel {
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    /// Drops everything prefill allocated. Called when prefill hands over to
    /// decode so nothing ANE-side competes with the expert cache for
    /// residency; cheap no-op when nothing is loaded.
    ///
    /// The Core ML model is not the only thing that has to go. Decode runs
    /// entirely on the GPU and never reads the shadow rows or the masks, but
    /// they stay allocated for the life of this object unless dropped here:
    /// ~33 MB of shadow per covered layer plus one mask per history window
    /// (33 MB at h0, 100 MB at h8192) is roughly half a gigabyte taken from
    /// the expert slot cache for the whole of decode. Releasing only the
    /// model left that behind and cost measured decode throughput -- the
    /// bounded cache is the entire reason a model larger than RAM runs at
    /// all, so prefill scratch must not outlive prefill.
    ///
    /// Masks are rebuilt on the next prefill (a few tens of ms of fills
    /// against a prefill measured in minutes); shadow rows are per-request by
    /// construction, so `shadowTokens` resets with them and a later chunk
    /// correctly falls back rather than attending to a freed history.
    func releaseModels() {
        residentModel = nil
        preloaded?.task.cancel()
        preloaded = nil
        // Release the borrowing MLMultiArrays before the storage they point
        // at: they are built with `deallocator: nil`, so this dictionary owns
        // the memory.
        masks.removeAll()
        for pointer in maskStorage.values { pointer.deallocate() }
        maskStorage.removeAll()
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
        shadowK.removeAll()
        shadowV.removeAll()
        shadowTokens = 0
    }

    /// Starts loading `layer`'s model for `history` in the background, if it
    /// is not already resident or in flight. Called right after a prediction
    /// returns, so the load runs while the caller encodes and executes the
    /// layer's MoE stage on the GPU.
    func preload(layer: Int, history: Int) {
        if let cached = residentModel,
           cached.layer == layer, cached.history == history { return }
        if let pending = preloaded,
           pending.layer == layer, pending.history == history { return }
        preloaded?.task.cancel()
        preloaded = (layer, history, Task { [self] in
            LoadedModelBox(model: try await loadModel(layer: layer,
                                                      history: history))
        })
    }

    private func mask(history: Int) throws -> MLMultiArray {
        if let cached = masks[history] { return cached }
        let total = history + chunkTokens
        let count = chunkTokens * total
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: count * MemoryLayout<Float16>.stride,
            alignment: 16_384)
        let values = storage.bindMemory(to: Float16.self, capacity: count)
        for row in 0..<chunkTokens {
            let base = row * total
            let allowed = history + row + 1
            for column in 0..<allowed { values[base + column] = 0 }
            for column in allowed..<total {
                values[base + column] = Self.maskNegative
            }
        }
        let array = try MLMultiArray(
            dataPointer: storage,
            shape: [1, 1, NSNumber(value: chunkTokens), NSNumber(value: total)],
            dataType: .float16,
            strides: [NSNumber(value: count), NSNumber(value: count),
                      NSNumber(value: total), 1],
            deallocator: nil)
        maskStorage[history] = storage
        masks[history] = array
        return array
    }

    private func wrap(_ buffer: MTLBuffer, rows: Int,
                      columns: Int) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: buffer.contents(),
                         shape: [NSNumber(value: rows), NSNumber(value: columns)],
                         dataType: .float16,
                         strides: [NSNumber(value: columns), 1],
                         deallocator: nil)
    }

    private func wrapShadow(_ storage: UnsafeMutableRawPointer,
                            rows: Int) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: storage,
                         shape: [NSNumber(value: rows), NSNumber(value: kvDim)],
                         dataType: .float16,
                         strides: [NSNumber(value: kvDim), 1],
                         deallocator: nil)
    }

    /// Runs one layer's attention block. `stagingNormed` must already hold
    /// the chunk's post-norm hidden rows; results land in `stagingOut` /
    /// `stagingK` / `stagingV` (real `tokenCount` rows; padding discarded).
    func predict(layer: Int, history: Int, tokenCount: Int) async throws {
        let halfBytes = MemoryLayout<Float16>.stride
        if tokenCount < chunkTokens {
            // Padded rows must be zeros: zero queries attend uniformly and
            // produce finite garbage that is discarded, whereas stale staging
            // bytes could push fp16 out of range.
            let start = tokenCount * hiddenSize * halfBytes
            let length = (chunkTokens - tokenCount) * hiddenSize * halfBytes
            memset(stagingNormed.contents().advanced(by: start), 0, length)
        }
        var features: [String: MLMultiArray] = [
            "normed": try wrap(stagingNormed, rows: chunkTokens,
                               columns: hiddenSize),
            "mask": try mask(history: history),
        ]
        if history > 0 {
            guard let kShadow = shadowK[layer], let vShadow = shadowV[layer] else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill shadow missing for layer \(layer) at history \(history)")
            }
            features["k_hist"] = try wrapShadow(kShadow, rows: history)
            features["v_hist"] = try wrapShadow(vShadow, rows: history)
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: features.mapValues { MLFeatureValue(multiArray: $0) })
        let options = MLPredictionOptions()
        options.outputBackings = [
            "out": try wrap(stagingOut, rows: chunkTokens, columns: hiddenSize),
            "k_new": try wrap(stagingK, rows: chunkTokens, columns: kvDim),
            "v_new": try wrap(stagingV, rows: chunkTokens, columns: kvDim),
        ]
        let model = try await model(layer: layer, history: history)
        let result = try await model.prediction(from: provider, options: options)
        // Output backings are best-effort; copy back any output Core ML chose
        // to allocate elsewhere.
        try copyIfNotBacked(result, name: "out", buffer: stagingOut,
                            elements: chunkTokens * hiddenSize)
        try copyIfNotBacked(result, name: "k_new", buffer: stagingK,
                            elements: chunkTokens * kvDim)
        try copyIfNotBacked(result, name: "v_new", buffer: stagingV,
                            elements: chunkTokens * kvDim)
    }

    private func copyIfNotBacked(_ result: MLFeatureProvider, name: String,
                                 buffer: MTLBuffer, elements: Int) throws {
        guard let array = result.featureValue(for: name)?.multiArrayValue else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill output '\(name)' missing from prediction")
        }
        array.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  base != buffer.contents() else { return }
            memcpy(buffer.contents(), base,
                   elements * MemoryLayout<Float16>.stride)
        }
    }

    /// Saves the chunk's K/V rows for later chunks. Partial chunks are always
    /// the last chunk of a prompt, so their rows can never be history and the
    /// shadow (33 MB per layer) is never allocated for single-chunk prompts.
    func appendShadow(layer: Int, startPosition: Int, tokenCount: Int) {
        guard tokenCount == chunkTokens else { return }
        let rowBytes = kvDim * MemoryLayout<Float16>.stride
        let capacityBytes = maxPromptTokens * rowBytes
        if shadowK[layer] == nil {
            shadowK[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
            shadowV[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
        }
        let offset = startPosition * rowBytes
        let length = tokenCount * rowBytes
        memcpy(shadowK[layer]!.advanced(by: offset),
               stagingK.contents(), length)
        memcpy(shadowV[layer]!.advanced(by: offset),
               stagingV.contents(), length)
    }

    /// Marks the chunk's shadow rows visible to the next chunk. Called once
    /// after every covered layer appended, so a thrown mid-chunk error leaves
    /// `shadowTokens` behind `startPosition` and the next attempt falls back
    /// to the GPU instead of attending to partial history.
    func finishChunk(startPosition: Int, tokenCount: Int) {
        shadowTokens = tokenCount == chunkTokens
            ? startPosition + tokenCount : 0
        if tokenCount < chunkTokens {
            // A partial chunk is the prompt's last: decode is next.
            releaseModels()
        }
    }
}
