import Foundation
import Darwin
import NVMAIFormat

/// Writes the resident LM `.bin` file (`model_weights.bin`) for the streaming
/// installer. The remote copy path (HTTPRangeSourceByteProvider) fills the
/// tensor payload via ranged requests; this type only creates the file, sizes
/// it, and lays down the binary index (header + entries + string table).
enum ResidentWriter {

    static func createAndWriteIndex(plan: ResidentFilePlan,
                                    audit: RepackAudit) throws -> Int32 {
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        do {
            try Posix.ftruncate(fd, path: plan.path, size: plan.totalSize)
            try writeIndex(plan: plan, fd: fd, audit: audit)
            return fd
        } catch {
            close(fd)
            throw error
        }
    }

    static func encodeIndex(plan: ResidentFilePlan) throws -> Data {
        let idxBytes = Int(plan.indexSize)
        // Bounded by the format's own v1 ceiling, which is the same constant
        // `GTurboResidentIndexCodec` and `VerifiedInstallTool` enforce when they
        // read the file back. It was `BoundedScratch.defaultLimitBytes` (under
        // 1 MB), which is the per-worker *staging* budget and has nothing to do
        // with this allocation: the index is the finished output, and it scales
        // with `tensors x name length`. KAT-Coder-V2.5-Dev's is ~28 MB, so the
        // staging budget refused an install the format itself can hold -- and
        // would have reported it as a scratch overrun rather than a size the
        // reader would also reject.
        let limitBytes = Int(GTurboFormatV1.residentIndexMaxBytes)
        guard idxBytes <= limitBytes else {
            throw RepackError.scratchExceeded(requested: idxBytes, limit: limitBytes)
        }
        // A tensor name comes from the source manifest, so its length is input,
        // not an invariant of this build: the index stores it in a UInt16 and
        // `GTurboBinary.writeIndexEntry` traps on anything longer. Checked here
        // so an over-long name is a report naming the tensor -- and so the trap
        // inside the binary writer stays an invariant a caller cannot reach.
        for entry in plan.entries where entry.name.utf8.count > Int(UInt16.max) {
            throw RepackError.configurationInvalid(
                detail: "resident tensor name is \(entry.name.utf8.count) bytes, over the "
                    + "\(UInt16.max)-byte limit the index stores: \(entry.name.prefix(120))")
        }
        let idxBuf = UnsafeMutableRawBufferPointer.allocate(byteCount: idxBytes,
                                                            alignment: 16_384)
        defer { idxBuf.deallocate() }
        idxBuf.initializeMemory(as: UInt8.self, repeating: 0)
        GTurboBinary.writeIndexHeader(into: idxBuf.baseAddress!,
                                      indexSize: plan.indexSize,
                                      residentSize: plan.residentSize,
                                      entryCount: UInt64(plan.entries.count))
        let entriesBase = 24
        let stringTableBase = entriesBase + plan.entries.count * GTurboBinary.indexEntryBytes
        for i in 0..<plan.entries.count {
            let dst = idxBuf.baseAddress!.advanced(by: entriesBase + i * GTurboBinary.indexEntryBytes)
            let nameOff = UInt32(stringTableBase) + plan.stringTableOffsets[i]
            GTurboBinary.writeIndexEntry(into: dst, entry: plan.entries[i], nameOffset: nameOff)
        }
        plan.stringTable.withUnsafeBufferPointer { src in
            let dst = idxBuf.baseAddress!.advanced(by: stringTableBase)
            memcpy(dst, src.baseAddress!, src.count)
        }
        return Data(bytes: idxBuf.baseAddress!, count: idxBytes)
    }

    private static func writeIndex(plan: ResidentFilePlan,
                                   fd: Int32,
                                   audit: RepackAudit) throws {
        let data = try encodeIndex(plan: plan)
        let idxBytes = data.count
        if idxBytes > audit.largestScratchBytes {
            audit.largestScratchBytes = idxBytes
        }
        try data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.pwriteAll(fd: fd, path: plan.path,
                                buf: base, count: idxBytes, offset: 0)
        }
        audit.recordWrite(bytes: idxBytes)
    }
}
