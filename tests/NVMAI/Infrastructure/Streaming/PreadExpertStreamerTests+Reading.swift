import Darwin
import Foundation
import Metal
import Testing

@testable import NVMAI

extension PreadExpertStreamerTests {
  /// A stride that cannot be represented as an `Int` used to trap in the slot
  /// allocation (`Int(layout.expertStride)`), which is a crash on a value that
  /// arrived in a layout file. The stream range is small and fits the synthetic
  /// file, so the open-time check passes and the allocation is what is asked
  /// about.
  @Test func aStrideTooLargeToAllocateIsRefused() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let huge = StreamLayout(path: url.path,
                            streamOffset: 0,
                            streamSize: Self.streamOffset + Self.streamSize,
                            expertsPerLayer: 1,
                            expertStride: UInt64.max)

    #expect(throws: StreamerError.self) {
      _ = try PreadExpertStreamer(layout: huge, device: device, slotCount: 1)
    }
  }

  /// A layout whose stream range wraps must be refused, not wrapped. `required`
  /// came out of an unchecked `streamOffset + streamSize`, so an install layout
  /// with a corrupt offset could produce a small `required`, pass the file-size
  /// check, and leave every later offset -- all of them `streamOffset +
  /// regionOffset` -- pointing outside the file.
  @Test func aWrappingStreamRangeIsRefused() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let wrapping = StreamLayout(path: url.path,
                                streamOffset: UInt64.max - 8,
                                streamSize: 64,
                                expertsPerLayer: Self.numExperts,
                                expertStride: UInt64(Self.expertStride))

    #expect(throws: StreamerError.self) {
      _ = try PreadExpertStreamer(layout: wrapping, device: device, slotCount: 2)
    }

    // The valid layout still opens, so this is not a guard that refuses
    // everything.
    _ = try PreadExpertStreamer(layout: Self.makeLayout(path: url.path),
                                device: device, slotCount: 2)
  }

  @Test func preadRoundTrip_matchesTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    for e in 0..<Self.numExperts {
      let r = try streamer.loadExpert(layer: 0, expert: e)
      #expect(r.offset == 0)
      #expect(r.size == UInt64(Self.expertStride))
      let got = Self.bytes(of: r.buffer, offset: 0, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(e) },
        "expert \(e) slot not uniformly tagged")
    }
  }

  @Test func shortRead_throwsSizeMismatch() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 1)

    // Truncate the file on disk to just past expert 0; the already-open fd
    // now hits EOF mid-read for any later expert.
    let truncatedLen = off_t(Self.streamOffset) + off_t(Self.expertStride)
    #expect(truncate(url.path, truncatedLen) == 0)

    #expect(throws: StreamerError.self) {
      _ = try streamer.loadExpert(layer: 0, expert: Self.numExperts - 1)
    }
  }

  @Test func slotReuse_roundRobinOverwrites() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    // With slotCount=2, experts 0,1,2,3 land in slots 0,1,0,1. Capture the
    // first buffer (slot 0, expert 0), then load expert 2 which reuses
    // slot 0 — the same MTLBuffer must now hold expert 2's tag.
    let r0 = try streamer.loadExpert(layer: 0, expert: 0)
    let r1 = try streamer.loadExpert(layer: 0, expert: 1)
    let r2 = try streamer.loadExpert(layer: 0, expert: 2)
    let r3 = try streamer.loadExpert(layer: 0, expert: 3)

    #expect(r0.buffer === r2.buffer, "expert 0 and 2 should share slot 0's buffer")
    #expect(r1.buffer === r3.buffer, "expert 1 and 3 should share slot 1's buffer")
    #expect(r0.buffer !== r1.buffer, "slots 0 and 1 must be distinct buffers")

    // r0 was overwritten by r2; reading slot 0 now yields expert 2's tag.
    let slot0 = Self.bytes(of: r2.buffer, offset: 0, count: Self.expertStride)
    #expect(slot0.allSatisfy { $0 == Self.tagByte(2) })
    let slot1 = Self.bytes(of: r3.buffer, offset: 0, count: Self.expertStride)
    #expect(slot1.allSatisfy { $0 == Self.tagByte(3) })
  }

}
