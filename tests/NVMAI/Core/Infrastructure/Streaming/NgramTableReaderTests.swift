import Foundation
import Testing
@testable import NVMAI

/// Row gather over the n-gram table, against a synthetic file whose contents
/// encode their own row index so a misaddressed read is detectable rather than
/// merely wrong-looking.
@Suite("N-gram table reader")
struct NgramTableReaderTests {
    private static let rowDim = 8
    private static let rowCount: UInt64 = 64

    /// Row r is filled with the value r, so reading row r must yield r in
    /// every lane. An off-by-one or a wrong stride lands on a different row and
    /// fails loudly instead of returning plausible embedding data.
    private static func makeTable() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-\(UUID().uuidString).bin")
        var bytes = [Float16]()
        bytes.reserveCapacity(Int(rowCount) * rowDim)
        for r in 0..<Int(rowCount) {
            bytes += [Float16](repeating: Float16(r), count: rowDim)
        }
        try bytes.withUnsafeBufferPointer {
            try Data(buffer: $0).write(to: url)
        }
        return url
    }

    private static func reader(_ url: URL) throws -> NgramTableReader {
        // Page cache left on for the test: the synthetic file is tiny and
        // F_NOCACHE on a just-written temp file measures nothing useful.
        try NgramTableReader(path: url.path, rowDim: rowDim,
                             rowCount: rowCount, bypassCache: false)
    }

    @Test("Gathers the requested rows in the requested order")
    func gathersInOrder() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        let want: [UInt32] = [7, 0, 63, 31, 7]
        var out = [Float16](repeating: 0, count: want.count * Self.rowDim)
        try out.withUnsafeMutableBytes { try r.gather(rows: want, into: $0.baseAddress!) }
        for (i, row) in want.enumerated() {
            for lane in 0..<Self.rowDim {
                #expect(out[i * Self.rowDim + lane] == Float16(row),
                        "slot \(i) lane \(lane) should hold row \(row)")
            }
        }
    }

    @Test("A row past the end is refused, not read out of bounds")
    func rejectsOutOfRange() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        var out = [Float16](repeating: 0, count: Self.rowDim)
        #expect(throws: NgramTableReader.Failure.self) {
            try out.withUnsafeMutableBytes {
                try r.gather(rows: [UInt32(Self.rowCount)], into: $0.baseAddress!)
            }
        }
    }

    @Test("A table whose size disagrees with the geometry is refused at open")
    func rejectsSizeMismatch() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        // Claiming more rows than the file holds would otherwise read
        // plausible values from wrong offsets rather than failing.
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(path: url.path, rowDim: Self.rowDim,
                                     rowCount: Self.rowCount + 1,
                                     bypassCache: false)
        }
        // A wrong row width is the same class of error.
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(path: url.path, rowDim: Self.rowDim * 2,
                                     rowCount: Self.rowCount,
                                     bypassCache: false)
        }
    }

    @Test("A missing table fails at open rather than at first gather")
    func rejectsMissingFile() {
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(path: "/nonexistent/ngram_table.bin",
                                     rowDim: 160, rowCount: 1, bypassCache: false)
        }
    }

    @Test("Production geometry moves ~5 KiB per token")
    func productionGatherBudget() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        // 160 fp16 x 16 heads at the real geometry; the fixture's rowDim is
        // smaller, so compute against the real numbers directly.
        let realRowBytes = 160 * MemoryLayout<Float16>.stride
        #expect(realRowBytes == 320)
        #expect(16 * realRowBytes == 5120)
        // And the reader reports its own geometry consistently.
        #expect(r.gatherBytes(headCount: 16) == 16 * r.rowBytes)
    }
}
