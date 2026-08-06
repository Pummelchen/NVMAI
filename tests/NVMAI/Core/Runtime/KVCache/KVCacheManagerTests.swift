import Testing
import Foundation
import Metal
@testable import NVMAI

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Qwen 3.6 config. Qwen has no
/// sliding-window layers (mask values are 1 = full and 2 = linear), so the
/// FP16 ring never engages and linear layers carry no per-token K/V rows.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.qwen36_35B_A3B

    private func makeManager(maxContext: Int,
                             fp16RingEnabled: Bool = false) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: 128)
        return (ctx, kv)
    }

    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // Full: numFullKVHeads(2) * fullHeadDim(256) * 2 = 1024 B/token.
        // Linear layers carry no per-token K/V storage at all.
        #expect(kv.kRange(layer: 3, start: 0, count: 1).stride == 2 * 256 * 2)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 128 * 1024)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 0) == 0)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers run k_norm + RoPE on K while V runs the no-scale norm
    /// without RoPE, so they require separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 3, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 3, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 3, position: 3)
        let vs = kv.vSlot(layer: 3, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 3).offset == 3 * 1024)
        #expect(kv.vSlot(layer: 3, position: 7).offset == 7 * 1024)
    }

    @Test func fp16Ring_neverEngagesWithoutSWALayers() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        // Full layers stay linear; no SWA layer exists to cap.
        #expect(kv.capacity(layer: 3) == 4096)
        #expect(kv.ringCapacity(layer: 3) == 0)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 4096 * 1024)
        // Linear layers keep no KV rows even with the ring enabled.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.ringCapacity(layer: 0) == 0)
    }

    @Test func fp16Ring_slotOffsetsNeverWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true)

        // No SWA layer wraps; full-layer slots stay linear within maxContext.
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 127).offset == 127 * 1024)
        #expect(kv.vSlot(layer: 3, position: 35).offset == 35 * 1024)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let fullStride = kv.kRange(layer: 3, start: 0, count: 1).stride

        let k = kv.kRange(layer: 3, start: 7, count: 3)
        let v = kv.vRange(layer: 7, start: 11, count: 5)

        #expect(k.offset == 7 * fullStride)
        #expect(k.stride == fullStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 3, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 7, validTokenCount: 0))
    }

    @Test func advanceByCountTracksCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        #expect(kv.position == 31)
        kv.advance(by: 0)
        #expect(kv.position == 31)
        kv.advance()
        #expect(kv.position == 32)
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }

    @Test func speculativeRewindMovesOnlyTheLogicalCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 17)
        let slotBefore = kv.kSlot(layer: 3, position: 12)
        try kv.rewind(to: 12)
        #expect(kv.position == 12)
        let slotAfter = kv.kSlot(layer: 3, position: 12)
        #expect(slotBefore.buffer === slotAfter.buffer)
        #expect(slotBefore.offset == slotAfter.offset)
        #expect(throws: InferenceStateSnapshotError.self) {
            try kv.rewind(to: 13)
        }
    }

}
