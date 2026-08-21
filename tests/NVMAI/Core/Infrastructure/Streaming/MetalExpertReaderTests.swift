import Foundation
import Testing

@testable import NVMAI

@Suite struct MetalExpertReaderTests {
    @Test func loadsDisjointFileRangesDirectlyIntoMetalBuffers() throws {
        let page = 16_384
        let bytes = [UInt8](repeating: 0x31, count: page)
            + [UInt8](repeating: 0x72, count: page)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metal-io-reader-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(bytes).write(to: url)

        let context = try MetalContext()
        let first = context.device.makeBuffer(length: page, options: .storageModeShared)!
        let second = context.device.makeBuffer(length: page, options: .storageModeShared)!
        let reader = try MetalExpertReader(path: url.path, device: context.device)

        try reader.fetch(
            offsets: [0, UInt64(page)],
            into: [first, second],
            byteCount: page)

        let firstBytes = [UInt8](UnsafeRawBufferPointer(
            start: first.contents(), count: page))
        let secondBytes = [UInt8](UnsafeRawBufferPointer(
            start: second.contents(), count: page))
        #expect(firstBytes.allSatisfy { $0 == 0x31 })
        #expect(secondBytes.allSatisfy { $0 == 0x72 })
    }
}
