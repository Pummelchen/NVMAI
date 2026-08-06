import Foundation
import Darwin

/// Shared building blocks for the resident LM and routed-expert layer writers.
public enum WriterCore {

    /// Tile size for pwrite (and the subsequent SHA-256 hashing pass). Chosen
    /// so per-worker scratch and per-syscall payload both stay well under
    /// the 1 MB BoundedScratch budget.
    public static let tileBytes: Int = 512 * 1024

    /// Compute SHA-256 of an entire (presumed-written) file by streaming it
    /// through `tileBytes` pread chunks. Drops pages with `F_NOCACHE` style
    /// behaviour via fcntl. Allocates one bounded scratch buffer.
    public static func hashEntireFile(path: String, size: UInt64,
                                      audit: RepackAudit,
                                      cancellationCheck: () throws -> Void = {}) throws -> String {
        guard size <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "file \(path) size \(size) exceeds the hashing range")
        }
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        // Hint the kernel that we will read this file sequentially and then
        // drop it from cache — keeps the post-write working set from blowing
        // up the dev box.
        _ = fcntl(fd, F_NOCACHE, 1)

        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: WriterCore.tileBytes,
                                                         alignment: 16_384)
        defer { buf.deallocate() }
        if buf.count > audit.largestScratchBytes {
            audit.largestScratchBytes = buf.count
        }

        var hasher = Sha256Stream()
        var off: UInt64 = 0
        var remaining = size
        while remaining > 0 {
            try cancellationCheck()
            let want = Int(min(remaining, UInt64(WriterCore.tileBytes)))
            errno = 0
            let got = pread(fd, buf.baseAddress, want, off_t(off))
            if got < 0, errno == EINTR { continue }
            guard got > 0 else {
                // Report the actual short-read count (0 at EOF) with an
                // accurate errno instead of a stale value.
                throw RepackError.preadShort(
                    path: path, expected: want, got: max(got, 0), errno: errno)
            }
            hasher.update(UnsafeRawBufferPointer(start: buf.baseAddress, count: got))
            audit.byteCopyTiles &+= 1
            off += UInt64(got)
            remaining -= UInt64(got)
        }
        return hasher.finalizeHexString()
    }
}
