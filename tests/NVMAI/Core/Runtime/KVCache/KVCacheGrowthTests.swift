import Testing
import Foundation
import Metal
@testable import NVMAI

/// KV storage grows on demand instead of reserving `maxContext`.
///
/// Reserving the maximum cost throughput even when unused: at 262144 tokens a
/// full reservation is 512 MiB per layer and 20 GiB across 40, and on a 24 GB
/// machine the same 25-token prompt measured 13.80 s of decode at maxContext 8192
/// against 22.44 s at 262144. What these tests protect is the part that could go
/// wrong silently -- growth must carry the existing KV forward, or a long
/// conversation would keep generating from subtly corrupted history.
@Suite struct KVCacheGrowthTests {
    private static func make(maxContext: Int) throws -> KVCacheManager? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return try KVCacheManager(device: device,
                                  config: .qwen36_35B_A3B,
                                  maxContext: maxContext)
    }

    /// Qwen 3.6 interleaves linear-attention layers, which hold a page-sized
    /// placeholder and a capacity of zero. Growth only concerns the full layers,
    /// so every test has to find one rather than assume layer 0.
    private static func fullLayer(_ kv: KVCacheManager) -> Int? {
        (0..<ArchConfig.qwen36_35B_A3B.numLayers)
            .first { kv.layerKind($0) == .full }
    }

    /// The whole point: a large advertised limit must not be allocated up front.
    @Test func initialCapacityIgnoresALargeMaxContext() throws {
        guard let kv = try Self.make(maxContext: 262_144) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        #expect(kv.capacity(layer: L) == KVCacheManager.initialCapacityTokens)
        #expect(kv.capacity(layer: L) < 262_144)
    }

    @Test func smallMaxContextIsNotRoundedUp() throws {
        guard let kv = try Self.make(maxContext: 1_024) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        #expect(kv.capacity(layer: L) == 1_024)
    }

    @Test func reserveGrowsByDoublingAndStopsAtMaxContext() throws {
        guard let kv = try Self.make(maxContext: 262_144) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        let start = kv.capacity(layer: L)
        try kv.reserve(tokens: start + 1)
        #expect(kv.capacity(layer: L) == start * 2)
        try kv.reserve(tokens: 262_144)
        #expect(kv.capacity(layer: L) == 262_144)
        // Asking past the ceiling must clamp, not overshoot or throw.
        try kv.reserve(tokens: 999_999)
        #expect(kv.capacity(layer: L) == 262_144)
    }

    @Test func reserveBelowCurrentCapacityIsANoOp() throws {
        guard let kv = try Self.make(maxContext: 262_144) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        let before = kv.capacity(layer: L)
        let buffer = kv.kSlot(layer: L, position: 0).buffer
        try kv.reserve(tokens: 8)
        #expect(kv.capacity(layer: L) == before)
        // Same buffer object: a needless reallocation would discard live KV.
        #expect(kv.kSlot(layer: L, position: 0).buffer === buffer)
    }

    /// The failure this exists to catch: growth that loses history.
    @Test func growthPreservesWrittenTokens() throws {
        guard let kv = try Self.make(maxContext: 262_144) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        let stride = kv.stride(layer: L)
        let start = kv.capacity(layer: L)
        // Write a recognisable byte per token across the first 64 positions.
        for position in 0..<64 {
            let slot = kv.kSlot(layer: L, position: position)
            slot.buffer.contents().advanced(by: slot.offset)
                .assumingMemoryBound(to: UInt8.self)
                .pointee = UInt8(position % 251)
            kv.advance(by: 1)
        }
        try kv.reserve(tokens: start + 1)
        #expect(kv.capacity(layer: L) == start * 2)
        for position in 0..<64 {
            let slot = kv.kSlot(layer: L, position: position)
            let byte = slot.buffer.contents().advanced(by: slot.offset)
                .assumingMemoryBound(to: UInt8.self).pointee
            #expect(byte == UInt8(position % 251),
                    "token \(position) lost across growth")
        }
        #expect(kv.stride(layer: L) == stride, "stride must not change")
    }

    @Test func repeatedReserveIsIdempotent() throws {
        guard let kv = try Self.make(maxContext: 262_144) else { return }
        guard let L = Self.fullLayer(kv) else { return }
        try kv.reserve(tokens: 20_000)
        let capacity = kv.capacity(layer: L)
        let buffer = kv.kSlot(layer: L, position: 0).buffer
        try kv.reserve(tokens: 20_000)
        #expect(kv.capacity(layer: L) == capacity)
        #expect(kv.kSlot(layer: L, position: 0).buffer === buffer)
    }
}
