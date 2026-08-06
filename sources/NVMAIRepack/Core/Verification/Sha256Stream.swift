import Foundation
import CryptoKit

/// Streaming SHA-256 hasher. Wraps CryptoKit's incremental API so we can hash
/// a file as we walk it (mmap'd source pages + zero-filled gaps) without
/// allocating a Swift heap buffer for the whole file.
struct Sha256Stream {
    private var hasher: SHA256

    init() { self.hasher = SHA256() }

    mutating func update(_ ptr: UnsafeRawBufferPointer) {
        hasher.update(bufferPointer: ptr)
    }

    func finalizeHexString() -> String {
        let digest = self.hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// One-shot helper to hash a file from disk in tile-bounded chunks. Used
    /// for fingerprinting `model.safetensors.index.json`.
    static func hashFile(path: String,
                                tileBytes: Int = 65_536,
                                noCache: Bool = false,
                                noFollow: Bool = false) throws -> String {
        let flags = O_RDONLY | (noFollow ? O_NOFOLLOW : 0)
        let fd = open(path, flags)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        defer { close(fd) }
        return try hashFileDescriptor(fd, displayPath: path,
                                      tileBytes: tileBytes, noCache: noCache)
    }

    package static func hashFileDescriptor(_ fd: Int32,
                                           displayPath: String,
                                           tileBytes: Int = 65_536,
                                           noCache: Bool = false) throws -> String {
        guard tileBytes > 0 else {
            throw RepackError.configurationInvalid(detail: "SHA-256 tile size must be positive")
        }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 0 else {
            throw RepackError.fileStatFailed(path: displayPath, errno: errno)
        }
        let expectedBytes = UInt64(st.st_size)
        if noCache { _ = fcntl(fd, F_NOCACHE, 1) }
        var hasher = Sha256Stream()
        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: tileBytes, alignment: 16_384)
        defer { buf.deallocate() }
        var total: UInt64 = 0
        while total < expectedBytes {
            let want = Int(min(UInt64(tileBytes), expectedBytes - total))
            errno = 0
            let got = pread(fd, buf.baseAddress, want, off_t(total))
            if got < 0, errno == EINTR { continue }
            if got < 0 {
                throw RepackError.preadShort(path: displayPath,
                                             expected: want, got: 0, errno: errno)
            }
            guard got > 0 else {
                // EOF before the fstat size: the file shrank mid-hash. Report
                // the actual short-read count (0) instead of silently hashing
                // a truncated prefix.
                throw RepackError.preadShort(path: displayPath,
                                             expected: want, got: 0, errno: errno)
            }
            hasher.update(UnsafeRawBufferPointer(start: buf.baseAddress, count: got))
            total += UInt64(got)
        }
        return hasher.finalizeHexString()
    }
}
