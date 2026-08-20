import Darwin
import Foundation

/// Copies planned safetensors ranges from a completed local snapshot. This is
/// used for derived sidecars whose source tensor conversion must happen
/// locally, while preserving the same bounded-copy and destination-digest
/// contracts as the remote installer.
final class LocalSourceByteProvider: SourceByteProvider {
    private let snapshotDirectory: String
    private let writeTileBytes: Int

    init(snapshotDirectory: String,
         writeTileBytes: Int = WriterCore.tileBytes) {
        self.snapshotDirectory = snapshotDirectory
        self.writeTileBytes = writeTileBytes
    }

    func copyBatch(
        _ copies: [CoalescedRangeCopy],
        completedRangeIDs: Set<String>,
        partialDirectory: String,
        temporaryPath: String,
        audit: RepackAudit,
        progress: @escaping @Sendable (UInt64) -> Void,
        commit: (RemoteCompletedRange) throws -> Void
    ) async throws {
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: writeTileBytes,
            alignment: 16_384)
        defer { scratch.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, scratch.count)

        var outputDescriptors: [String: Int32] = [:]
        defer { outputDescriptors.values.forEach { close($0) } }
        var copiedBytes: UInt64 = 0

        for copy in copies where !completedRangeIDs.contains(copy.id) {
            try Task.checkCancellation()
            let sourcePath = try resolvedSourcePath(copy.shardID)
            let sourceDescriptor = try Posix.openReadNoFollow(sourcePath)
            var touched = Set<String>()
            do {
                let sourceSize = try Posix.fileSize(fd: sourceDescriptor, path: sourcePath)
                guard copy.sourceOffset <= sourceSize,
                      copy.size <= sourceSize - copy.sourceOffset else {
                    throw RepackError.safetensorsTensorOutOfRange(
                        path: sourcePath,
                        name: copy.id,
                        end: copy.sourceOffset + copy.size,
                        fileSize: sourceSize)
                }
                for destination in copy.destinations {
                    let descriptor = try outputDescriptor(
                        path: destination.destinationPath,
                        cache: &outputDescriptors)
                    touched.insert(destination.destinationPath)
                    try copyBytes(
                        sourceDescriptor: sourceDescriptor,
                        sourcePath: sourcePath,
                        destinationDescriptor: descriptor,
                        destinationPath: destination.destinationPath,
                        sourceOffset: destination.sourceOffset,
                        destinationOffset: destination.destinationOffset,
                        size: destination.size,
                        scratch: scratch,
                        audit: audit)
                }
                close(sourceDescriptor)
            } catch {
                close(sourceDescriptor)
                throw error
            }
            for path in touched {
                if let descriptor = outputDescriptors[path] {
                    try Posix.fsync(descriptor, path: path)
                }
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory,
                scratch: scratch)
            try commit(RemoteCompletedRange(
                id: copy.id,
                destinationDigest: digest,
                sourceBytes: copy.size,
                destinationBytes: copy.destinations.reduce(0) { $0 + $1.size }))
            copiedBytes += copy.size
            progress(copiedBytes)
        }
    }

    private func resolvedSourcePath(_ shard: String) throws -> String {
        guard !shard.isEmpty,
              !shard.hasPrefix("/"),
              !shard.contains(".."),
              !shard.contains("/"),
              !shard.contains("\\") else {
            throw RepackError.configurationInvalid(
                detail: "unsafe local snapshot shard path \(shard)")
        }
        let root = URL(fileURLWithPath: snapshotDirectory).standardizedFileURL.path
        let path = URL(fileURLWithPath: shard, relativeTo:
            URL(fileURLWithPath: root, isDirectory: true)).standardizedFileURL.path
        guard path.hasPrefix(root + "/") else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot shard escapes its directory: \(shard)")
        }
        return path
    }

    private func outputDescriptor(path: String,
                                  cache: inout [String: Int32]) throws -> Int32 {
        if let existing = cache[path],
           try Posix.descriptorMatchesPath(existing, path: path) {
            return existing
        }
        if let existing = cache.removeValue(forKey: path) {
            close(existing)
        }
        let descriptor = try Posix.openExistingRW(path)
        cache[path] = descriptor
        return descriptor
    }

    private func copyBytes(sourceDescriptor: Int32,
                           sourcePath: String,
                           destinationDescriptor: Int32,
                           destinationPath: String,
                           sourceOffset: UInt64,
                           destinationOffset: UInt64,
                           size: UInt64,
                           scratch: UnsafeMutableRawBufferPointer,
                           audit: RepackAudit) throws {
        var remaining = size
        var source = sourceOffset
        var destination = destinationOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let count = min(Int(remaining), scratch.count)
            try Posix.preadAll(fd: sourceDescriptor,
                               path: sourcePath,
                               buf: scratch.baseAddress!,
                               count: count,
                               offset: source)
            try Posix.pwriteAll(fd: destinationDescriptor,
                                path: destinationPath,
                                buf: scratch.baseAddress!,
                                count: count,
                                offset: destination)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            remaining -= UInt64(count)
            source += UInt64(count)
            destination += UInt64(count)
        }
    }
}
