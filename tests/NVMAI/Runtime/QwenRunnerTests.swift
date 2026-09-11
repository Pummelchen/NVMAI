import Testing
import Foundation
import Metal
@testable import NVMAI
import NVMAIValidationSupport

/// Qwen 3.6 runtime integration: runner construction against the qwen toy
/// fixture (no auxiliary sandwich/scale tensors present — init must not touch
/// them), decode and chunked-prefill smoke over the hybrid linear/full layer
/// graph, and the KV-cache + GDN-state reset interplay.
@Suite struct QwenRunnerTests {

    private func makeRunner(weightBits: Int = 4) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: weightBits)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwenToy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        return (dir, ctx, runner)
    }

    @Test(arguments: [8])
    func decodeAndChunkedPrefillSupportHigherBitCheckpoints(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(!runner.usesFusedGreedyHead)
        let logits = try makeLogits(ctx, vocab: 1024)
        let tokens: [Int32] = [1, 2]
        var progress: [Int] = []
        let result = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { progress.append($0) })
        #expect(result == PrefillResult(newPosition: 2, seed: .logitsWritten))
        #expect(progress == [2])
        try await runner.produce(token: 3, position: 2, into: logits)
        #expect(runner.continuationPosition == 3)
        let values = Fp16Buffer.read(logits, count: 1024)
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test(arguments: [8])
    func higherBitPrefillThenDecodeMatchesPureDecode(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        let actual = Fp16Buffer.read(logits, count: 1024)

        for (lhs, rhs) in zip(actual, reference) {
            #expect(abs(Float(lhs) - Float(rhs)) < 0.05)
        }
    }

    @Test(arguments: [8])
    func higherBitMultiTokenPrefillMatchesPureDecode(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        let tokens: [Int32] = [11, 7, 5, 3, 9]

        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(logits, count: 1024)

        for (lhs, rhs) in zip(actual, reference) {
            #expect(abs(Float(lhs) - Float(rhs)) < 0.05)
        }
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// (a) Runner init with the qwen arch: GDN kernels, state manager, and
    /// the int8 router/gate buffers must construct without touching any
    /// auxiliary sandwich/scale tensors (the fixture does not contain them —
    /// any access would throw `tensorNotFound`).
    @Test func runnerInit_qwenToy_doesNotTouchAuxTensors() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 64)
        #expect(runner.usesFusedGreedyHead)
    }

    /// Decode smoke over the hybrid layer graph: two linear (GDN) layers and
    /// two gated full-attention layers, 8-of-8 routing, gated shared expert,
    /// untied greedy head. Verifies tokens are produced, the cursor advances,
    /// and reset() rewinds both the KV cache and the GDN state.
    @Test func decodeSmoke_hybridLayerGraph() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        #expect(first < 1024)
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        // Same input from the empty state must reproduce the same argmax —
        // this fails if reset() leaves stale GDN state or conv tail behind.
        #expect(runner.lastGreedyToken == first)
    }

    /// Chunked prefill smoke: one chunk through the qwen prefill path
    /// (batched GDN projections + conv tail carry, packed q_proj split,
    /// sub-dim RoPE, no V norm, affine router scales, residual-add tail),
    /// then a decode continuation on top of the prefilled state.
    @Test func prefillChunkedSmoke_thenDecodeContinuation() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        let tokens: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let result = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == 8)
        if case .greedyToken(let token) = result.seed {
            #expect(token < 1024)
        } else {
            Issue.record("expected a greedy seed token from the fused head")
        }

        try await runner.produce(token: 9, position: 8, into: logits)
        #expect(runner.lastGreedyToken < 1024)
        #expect(runner.continuationPosition == 9)
    }

    /// Prefill/decode consistency: prefilling [t0] then decoding t1 must give
    /// the same argmax as decoding t0, t1 step by step from a fresh state —
    /// the two paths share state (KV + GDN recurrent state + conv tail).
    @Test func prefillThenDecode_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        // Pure decode reference.
        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        // Prefill the first token, then decode the second.
        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    @Test(arguments: [4, 8])
    func inferenceStateSnapshotRestoresKVAndGDNExactly(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let snapshot = try runner.captureInferenceState()
        #expect(snapshot.descriptor.position == 2)
        #expect(!snapshot.descriptor.kvSegmentLengths.isEmpty)
        #expect(!snapshot.descriptor.gdnSegmentLengths.isEmpty)
        #expect(snapshot.payload.count == snapshot.descriptor.payloadBytes)

        try await runner.produce(token: 5, position: 2, into: logits)
        let reference = Fp16Buffer.read(logits, count: 1024)

        runner.reset()
        try runner.restoreInferenceState(snapshot)
        #expect(runner.continuationPosition == 2)
        try await runner.produce(token: 5, position: 2, into: logits)
        let restored = Fp16Buffer.read(logits, count: 1024)

        #expect(restored == reference)
    }

    @Test func inferenceStateSnapshotHonorsSizeLimitWithoutMutatingRunner() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        try await runner.produce(token: 1, position: 0, into: logits)

        #expect(throws: InferenceStateSnapshotError.self) {
            try runner.captureInferenceState(maximumBytes: 1)
        }
        #expect(runner.continuationPosition == 1)
    }

    @Test func corruptInferenceStateSnapshotFailsClosedAndResets() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 1024)
        try await runner.produce(token: 1, position: 0, into: logits)
        let snapshot = try runner.captureInferenceState()
        let corrupt = InferenceStateSnapshot(
            descriptor: snapshot.descriptor,
            payload: snapshot.payload.dropLast())

        #expect(throws: InferenceStateSnapshotError.self) {
            try runner.restoreInferenceState(corrupt)
        }
        #expect(runner.continuationPosition == 0)
    }

    /// (b) KV manager + GDN state manager interplay under the qwen mask:
    /// linear layers carry no per-token KV storage, full layers are linear
    /// append-only, and reset() returns both to the empty-context state.
    @Test func kvManagerAndGdnState_resetInterplay() throws {
        let cfg = ArchConfig.qwenToy()
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: cfg,
                                    maxContext: 32,
                                    fp16RingEnabled: true,
                                    slidingWindow: cfg.slidingWindow,
                                    maxPrefillChunkTokens: 32)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.layerKind(1) == .full)
        #expect(kv.layerKind(2) == .linear)
        #expect(kv.layerKind(3) == .full)
        // Linear layers: no KV rows; full layers: 2 heads * 32 dim * 2 B.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 1) == 32)
        #expect(kv.stride(layer: 1) == 2 * 32 * 2)
        // No SWA layers => the FP16 ring never engages anywhere.
        for L in 0..<cfg.numLayers {
            #expect(kv.ringCapacity(layer: L) == 0)
        }

        let gdnState = try GDNStateManager(device: ctx.device, config: cfg)
        #expect(gdnState.isLinear(layer: 0))
        #expect(!gdnState.isLinear(layer: 1))
        let la = cfg.linearAttention
        #expect(gdnState.stateBytesPerLayer
                == la.numVHeads * la.valueHeadDim * la.keyHeadDim * 4)
        #expect(gdnState.convTailBytesPerLayer
                == (la.convKernelSize - 1) * la.qkvDim * 2)

        // Dirty the recurrent state and the KV cursor, then reset both.
        let state = gdnState.stateBuffer(layer: 0)
        state.contents().assumingMemoryBound(to: Float.self)[0] = 42
        let tail = gdnState.convTailBuffer(layer: 2)
        tail.contents().assumingMemoryBound(to: UInt16.self)[0] = 0x3C00
        kv.advance(by: 4)
        #expect(kv.position == 4)

        kv.reset()
        gdnState.reset()
        #expect(kv.position == 0)
        #expect(state.contents().assumingMemoryBound(to: Float.self)[0] == 0)
        #expect(tail.contents().assumingMemoryBound(to: UInt16.self)[0] == 0)
    }

    /// Prefill scratch layout: the qwen shape must size the packed q_proj /
    /// split-gate / GDN buffers; the MTP sidecar shape differs (no linear
    /// layers, no GDN buffers, but still gated attention output).
    @Test func prefillScratchLayout_qwenAndMtpSizes() {
        let qwen = PrefillChunkScratchLayout(config: .qwenToy(), chunkTokens: 32)
        // max(packed q_proj 2*4*32, gdn qkvDim 256) = 256.
        #expect(qwen.qProjElementsPerToken == 256)
        #expect(qwen.attnGateElementsPerToken == 4 * 32)
        #expect(qwen.gdnQKVDim == 256)
        #expect(qwen.gdnValueDim == 128)
        #expect(qwen.gdnVHeads == 4)
        #expect(qwen.sharedScalarGateElements == 1)
        #expect(qwen.qElements == 32 * 256)
        #expect(qwen.attentionOutputElements == 32 * 4 * 32)

        let mtp = PrefillChunkScratchLayout(config: .qwen36MTP, chunkTokens: 128)
        // Gate-packed q_proj with no linear layers: 2x the per-head rows.
        #expect(mtp.qProjElementsPerToken == 2 * mtp.maxQElementsPerToken)
        #expect(mtp.attnGateElementsPerToken == mtp.maxQElementsPerToken)
        #expect(mtp.gdnQKVDim == 0)
        #expect(mtp.gdnValueDim == 0)
        #expect(mtp.sharedScalarGateElements == 1)
        #expect(mtp.qElements == 128 * 2 * mtp.maxQElementsPerToken)
    }
}
