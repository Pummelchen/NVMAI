import Foundation

/// Row gather over Qwen3.8-Flash-Next's n-gram embedding table.
///
/// The table is ~320M rows of `embedDim / heads` fp16 values — 102 GB at the
/// production geometry, far past anything that could be resident. It is read
/// the way routed experts are: bypassing the page cache, so the declared
/// memory budget stays the real footprint.
///
/// The gather is much friendlier than an expert miss, though, and the
/// difference is worth stating because it shapes how this should be scheduled:
/// a row id depends only on token ids (see `PLEHash`), never on router output
/// or any activation. The rows for token N+1 are therefore knowable the moment
/// token N is sampled, so this read can be issued entirely off the decode
/// critical path — where an expert miss cannot be, because routing is not
/// known until the layer runs.
///
/// unchecked-invariant: immutable after `init`. The descriptor and geometry
/// are never mutated afterwards, and `pread` does not move a shared file
/// offset, so concurrent gathers only read.
public final class NgramTableReader: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case openFailed(path: String, errno: Int32)
        case sizeMismatch(path: String, expected: UInt64, actual: UInt64)
        case rowOutOfRange(row: UInt32, rowCount: UInt64)
        case readFailed(row: UInt32, errno: Int32)
        case shortRead(row: UInt32, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .openFailed(let p, let e):
                return "n-gram table open(\(p)) failed: \(String(cString: strerror(e)))"
            case .sizeMismatch(let p, let expected, let actual):
                return "n-gram table \(p) is \(actual) bytes; expected at least "
                    + "\(expected) and at most 1 MiB of alignment padding past "
                    + "it. The table does not match the manifest's geometry"
            case .rowOutOfRange(let row, let count):
                return "n-gram row \(row) is outside the table's \(count) rows"
            case .readFailed(let row, let e):
                return "n-gram row \(row) read failed: \(String(cString: strerror(e)))"
            case .shortRead(let row, let expected, let got):
                return "n-gram row \(row) short read: expected \(expected), got \(got)"
            }
        }
    }

    private let fd: Int32
    public let path: String
    /// Values per row: `embedDim / totalHeads` (160 in production).
    public let rowDim: Int
    public let rowBytes: Int
    public let rowCount: UInt64

    /// - Parameters:
    ///   - rowCount: expected rows, from the manifest's PLE geometry. The file
    ///     size is checked against it: a table that disagrees would otherwise
    ///     be read at plausible but wrong offsets, returning real embedding
    ///     data from the wrong rows.
    ///   - bypassCache: keep the gather out of the page cache. Default on,
    ///     because a bounded footprint is the point of streaming.
    public init(path: String, rowDim: Int, rowCount: UInt64,
                bypassCache: Bool = true) throws {
        precondition(rowDim > 0 && rowCount > 0)
        let opened = open(path, O_RDONLY)
        guard opened >= 0 else {
            throw Failure.openFailed(path: path, errno: errno)
        }
        if bypassCache { _ = fcntl(opened, F_NOCACHE, 1) }
        var st = stat()
        guard fstat(opened, &st) == 0 else {
            let code = errno
            close(opened)
            throw Failure.openFailed(path: path, errno: code)
        }
        let bytes = rowDim * MemoryLayout<Float16>.stride
        let expected = rowCount &* UInt64(bytes)
        // The checkpoint pads the table past its last addressable row (the
        // shipped one rounds up to a 512-row boundary), so an exact size is
        // the wrong check. What this must still catch is a table built for a
        // different geometry, and that differs by a whole head -- 20M rows,
        // 6.4 GB -- not by alignment. Allowing at most 1 MiB of surplus
        // separates the two by three orders of magnitude.
        let size = UInt64(st.st_size)
        let surplusLimit: UInt64 = 1 << 20
        guard size >= expected, size - expected <= surplusLimit,
              size % UInt64(bytes) == 0 else {
            close(opened)
            throw Failure.sizeMismatch(path: path, expected: expected,
                                       actual: size)
        }
        self.fd = opened
        self.path = path
        self.rowDim = rowDim
        self.rowBytes = bytes
        self.rowCount = rowCount
    }

    deinit { close(fd) }

    /// Gathers `rows` into `destination`, which must hold
    /// `rows.count * rowDim` fp16 values. Rows land in the order given, which
    /// is head order — the caller concatenates them into the PLE input.
    public func gather(rows: [UInt32],
                       into destination: UnsafeMutableRawPointer) throws {
        for (i, row) in rows.enumerated() {
            guard UInt64(row) < rowCount else {
                throw Failure.rowOutOfRange(row: row, rowCount: rowCount)
            }
            let offset = UInt64(row) &* UInt64(rowBytes)
            let target = destination.advanced(by: i * rowBytes)
            var moved = 0
            while moved < rowBytes {
                let n = pread(fd, target.advanced(by: moved),
                              rowBytes - moved, off_t(offset) + off_t(moved))
                if n < 0 {
                    if errno == EINTR { continue }
                    throw Failure.readFailed(row: row, errno: errno)
                }
                if n == 0 {
                    throw Failure.shortRead(row: row, expected: rowBytes,
                                            got: moved)
                }
                moved += n
            }
        }
    }

    /// Bytes one token's gather moves, for budgeting. At the production
    /// geometry this is 16 x 320 B = 5 KiB per token — three orders of
    /// magnitude under a routed-expert token's traffic.
    public func gatherBytes(headCount: Int) -> Int { headCount * rowBytes }
}
