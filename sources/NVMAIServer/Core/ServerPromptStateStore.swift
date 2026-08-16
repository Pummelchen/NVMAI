import CryptoKit
import Foundation
import NVMAI
import Synchronization

struct ServerPromptCacheStorageConfiguration: Sendable, Equatable {
    let memoryLimitBytes: Int
    let diskDirectory: URL?
    let diskLimitBytes: Int

    init(memoryLimitBytes: Int,
         diskDirectory: URL?,
         diskLimitBytes: Int) {
        precondition(memoryLimitBytes >= 0)
        precondition(diskLimitBytes >= 0)
        self.memoryLimitBytes = memoryLimitBytes
        self.diskDirectory = diskDirectory
        self.diskLimitBytes = diskLimitBytes
    }
}

struct ServerPromptStateSaveResult: Sendable, Equatable {
    let unbackedEntryIDs: [UUID]
    let diskError: String?
    let memoryBytes: Int
    let diskBytes: Int
}

enum ServerPromptStateStoreError: Error, CustomStringConvertible {
    case missing(UUID)
    case corrupt(UUID, String)

    var description: String {
        switch self {
        case .missing(let id):
            "prompt-cache state \(id.uuidString.lowercased()) is unavailable"
        case .corrupt(let id, let reason):
            "prompt-cache state \(id.uuidString.lowercased()) is corrupt: \(reason)"
        }
    }
}

/// Persistent + in-memory backing for published prompt-cache entries.
///
/// Concurrency model: the store is shared between the session actor (restore,
/// remove, contains) and the detached save tasks. All in-memory bookkeeping is
/// guarded by `state`; all disk writes are serialized on `diskQueue` so
/// concurrent saves (a later generation's snapshot while an earlier write is
/// still in flight) never interleave file operations or clobber each other.
final class ServerPromptStateStore: @unchecked Sendable {
    private struct DiskMetadata: Codable {
        static let currentVersion = 1

        let version: Int
        let entry: ServerPromptCacheEntry
        let descriptor: InferenceStateSnapshotDescriptor
        let payloadSHA256: String
    }

    private struct DiskRecord {
        let metadata: DiskMetadata
        let directory: URL
        let payload: URL
        var modificationDate: Date
    }

    private static let metadataName = "metadata.json"
    private static let payloadName = "state.bin"
    private static let maximumMetadataBytes = 16 * 1_048_576

    /// Hard ceiling for one snapshot capture (S2). A snapshot above this is
    /// never allocated, bounding the multi-GiB capture + write cost regardless
    /// of a large configured disk budget (default 8 GiB).
    static let maximumCaptureBytes = 4 * 1_024 * 1_048_576

    private struct State {
        var memory: [UUID: InferenceStateSnapshot] = [:]
        var memoryLRU: [UUID] = []
        var memoryBytes = 0
        var disk: [UUID: DiskRecord] = [:]
        var diskLRU: [UUID] = []
        var diskBytes = 0
    }

    private let configuration: ServerPromptCacheStorageConfiguration
    private let fileManager: FileManager
    private let state = Mutex(State())
    private let diskQueue = DispatchQueue(
        label: "nvmai.prompt-state-store.disk",
        qos: .utility)

    var maximumSnapshotBytes: Int {
        min(max(configuration.memoryLimitBytes,
                configuration.diskDirectory == nil ? 0 : configuration.diskLimitBytes),
            Self.maximumCaptureBytes)
    }

    init(configuration: ServerPromptCacheStorageConfiguration,
         fileManager: FileManager = .default) throws {
        self.configuration = configuration
        self.fileManager = fileManager
        if let root = configuration.diskDirectory {
            try fileManager.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: root.path)
        }
    }

    func loadEntries(domain: ServerPromptCacheDomain) -> [ServerPromptCacheEntry] {
        state.withLock { state in
            state.disk.removeAll(keepingCapacity: true)
            state.diskLRU.removeAll(keepingCapacity: true)
            state.diskBytes = 0
        }
        guard let root = configuration.diskDirectory,
              configuration.diskLimitBytes > 0 else { return [] }
        let keys: Set<URLResourceKey> = [
            .isDirectoryKey,
            .isSymbolicLinkKey,
            .contentModificationDateKey,
            .fileSizeKey,
        ]
        guard let children = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]) else { return [] }

        var loaded: [DiskRecord] = []
        for directory in children {
            guard let directoryID = UUID(uuidString: directory.lastPathComponent),
                  let values = try? directory.resourceValues(forKeys: keys),
                  values.isDirectory == true,
                  values.isSymbolicLink != true else { continue }
            let metadataURL = directory.appendingPathComponent(Self.metadataName)
            let payloadURL = directory.appendingPathComponent(Self.payloadName)
            guard let metadataValues = try? metadataURL.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]),
                  metadataValues.isRegularFile == true,
                  metadataValues.isSymbolicLink != true,
                  let metadataSize = metadataValues.fileSize,
                  metadataSize <= Self.maximumMetadataBytes,
                  let metadataData = try? Data(contentsOf: metadataURL),
                  let metadata = try? JSONDecoder().decode(
                    DiskMetadata.self,
                    from: metadataData),
                  metadata.version == DiskMetadata.currentVersion,
                  metadata.entry.id == directoryID,
                  metadata.descriptor.version
                    == InferenceStateSnapshotDescriptor.currentVersion,
                  metadata.entry.kvPosition == metadata.descriptor.position,
                  (try? metadata.descriptor.validatedPayloadBytes()) != nil,
                  let payloadValues = try? payloadURL.resourceValues(
                    forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey]),
                  payloadValues.isRegularFile == true,
                  payloadValues.isSymbolicLink != true,
                  payloadValues.fileSize == metadata.descriptor.payloadBytes else {
                continue
            }
            loaded.append(DiskRecord(
                metadata: metadata,
                directory: directory,
                payload: payloadURL,
                modificationDate: values.contentModificationDate ?? .distantPast))
        }

        loaded.sort { $0.modificationDate < $1.modificationDate }
        state.withLock { state in
            for record in loaded {
                let id = record.metadata.entry.id
                state.disk[id] = record
                state.diskLRU.append(id)
                state.diskBytes += record.metadata.descriptor.payloadBytes
            }
        }
        _ = evictDiskIfNeeded()
        return state.withLock { state in
            state.diskLRU.compactMap {
                guard let entry = state.disk[$0]?.metadata.entry,
                      entry.domain == domain else { return nil }
                return entry
            }
        }
    }

    /// Persist a snapshot. Async: the write (SHA-256 + state.bin +
    /// metadata.json) runs on the store's serial disk queue, never on the
    /// caller's actor, so a multi-GiB write does not stall the session.
    func save(entry: ServerPromptCacheEntry,
              snapshot: InferenceStateSnapshot) async -> ServerPromptStateSaveResult {
        await withCheckedContinuation { continuation in
            diskQueue.async { [self] in
                continuation.resume(returning: saveSync(entry: entry, snapshot: snapshot))
            }
        }
    }

    private func saveSync(entry: ServerPromptCacheEntry,
                          snapshot: InferenceStateSnapshot) -> ServerPromptStateSaveResult {
        precondition(entry.kvPosition == snapshot.descriptor.position)
        var unbacked: [UUID] = []
        var diskError: String?

        if snapshot.payload.count <= configuration.memoryLimitBytes,
           configuration.memoryLimitBytes > 0 {
            insertMemory(snapshot, id: entry.id)
        }

        if configuration.diskDirectory != nil,
           configuration.diskLimitBytes > 0,
           snapshot.payload.count <= configuration.diskLimitBytes {
            do {
                try writeDisk(entry: entry, snapshot: snapshot)
            } catch {
                diskError = String(describing: error)
            }
        }

        let memoryEvicted = evictMemoryIfNeeded()
        let diskEvicted = evictDiskIfNeeded()
        for id in memoryEvicted + diskEvicted where !contains(id) {
            if !unbacked.contains(id) { unbacked.append(id) }
        }
        if !contains(entry.id), !unbacked.contains(entry.id) {
            unbacked.append(entry.id)
        }

        let bytes = state.withLock { state in (state.memoryBytes, state.diskBytes) }
        return ServerPromptStateSaveResult(
            unbackedEntryIDs: unbacked,
            diskError: diskError,
            memoryBytes: bytes.0,
            diskBytes: bytes.1)
    }

    func restore(entryID: UUID,
                 into runner: RealForwardRunner) async throws -> String {
        // S3: the disk read + SHA-256 verification run off the caller's actor
        // (detached task, awaited here); only the Metal-buffer restore stays on
        // the actor.
        let loaded = try await Task.detached(priority: .userInitiated) { [self] in
            try loadSnapshot(entryID: entryID)
        }.value
        do {
            try runner.restoreInferenceState(loaded.snapshot)
            return loaded.tier
        } catch {
            remove(entryIDs: [entryID])
            throw error
        }
    }

    func loadSnapshot(
        entryID: UUID
    ) throws -> (snapshot: InferenceStateSnapshot, tier: String) {
        if let snapshot = state.withLock({ $0.memory[entryID] }) {
            touchMemory(entryID)
            touchDisk(entryID)
            return (snapshot, "ram")
        }
        guard let record = state.withLock({ $0.disk[entryID] }) else {
            throw ServerPromptStateStoreError.missing(entryID)
        }
        do {
            // S3: stream the payload in bounded chunks, verifying the SHA-256
            // while copying (single pass over the file) — never materialize a
            // second full copy just to hash it.
            let payload = try readPayloadVerifying(record: record, entryID: entryID)
            let snapshot = InferenceStateSnapshot(
                descriptor: record.metadata.descriptor,
                payload: payload)
            if payload.count <= configuration.memoryLimitBytes,
               configuration.memoryLimitBytes > 0 {
                insertMemory(snapshot, id: entryID)
                _ = evictMemoryIfNeeded()
            }
            touchDisk(entryID)
            return (snapshot, "ssd")
        } catch {
            remove(entryIDs: [entryID])
            throw error
        }
    }

    private func readPayloadVerifying(
        record: DiskRecord,
        entryID: UUID
    ) throws -> Data {
        let expected = record.metadata.descriptor.payloadBytes
        let handle = try FileHandle(forReadingFrom: record.payload)
        defer { try? handle.close() }
        var hasher = SHA256()
        var payload = Data()
        payload.reserveCapacity(expected)
        while true {
            guard let chunk = try handle.read(upToCount: 1_048_576),
                  !chunk.isEmpty else { break }
            hasher.update(data: chunk)
            payload.append(chunk)
        }
        guard payload.count == expected else {
            throw ServerPromptStateStoreError.corrupt(entryID, "payload size mismatch")
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == record.metadata.payloadSHA256 else {
            throw ServerPromptStateStoreError.corrupt(entryID, "SHA-256 mismatch")
        }
        return payload
    }

    func contains(_ entryID: UUID) -> Bool {
        state.withLock { $0.memory[entryID] != nil || $0.disk[entryID] != nil }
    }

    func remove(entryIDs: some Sequence<UUID>) {
        var directories: [URL] = []
        state.withLock { state in
            for id in entryIDs {
                if let snapshot = state.memory.removeValue(forKey: id) {
                    state.memoryBytes -= snapshot.payload.count
                }
                state.memoryLRU.removeAll { $0 == id }
                if let record = state.disk.removeValue(forKey: id) {
                    state.diskBytes -= record.metadata.descriptor.payloadBytes
                    directories.append(record.directory)
                }
                state.diskLRU.removeAll { $0 == id }
            }
        }
        for directory in directories {
            try? fileManager.removeItem(at: directory)
        }
    }

    private func insertMemory(_ snapshot: InferenceStateSnapshot, id: UUID) {
        state.withLock { state in
            if let previous = state.memory.updateValue(snapshot, forKey: id) {
                state.memoryBytes -= previous.payload.count
            }
            state.memoryBytes += snapshot.payload.count
        }
        touchMemory(id)
    }

    private func touchMemory(_ id: UUID) {
        state.withLock { state in
            state.memoryLRU.removeAll { $0 == id }
            state.memoryLRU.append(id)
        }
    }

    private func touchDisk(_ id: UUID) {
        let record = state.withLock { state -> DiskRecord? in
            guard var record = state.disk[id] else { return nil }
            record.modificationDate = Date()
            state.disk[id] = record
            state.diskLRU.removeAll { $0 == id }
            state.diskLRU.append(id)
            return record
        }
        guard let record else { return }
        // S34: the mtime only feeds cross-restart LRU ordering; update it on
        // the background disk queue so the caller's actor never blocks on
        // setAttributes.
        let date = record.modificationDate
        let path = record.directory.path
        diskQueue.async { [self] in
            try? self.fileManager.setAttributes([.modificationDate: date], ofItemAtPath: path)
        }
    }

    private func evictMemoryIfNeeded() -> [UUID] {
        state.withLock { state -> [UUID] in
            var evicted: [UUID] = []
            while state.memoryBytes > configuration.memoryLimitBytes,
                  let id = state.memoryLRU.first {
                state.memoryLRU.removeFirst()
                if let snapshot = state.memory.removeValue(forKey: id) {
                    state.memoryBytes -= snapshot.payload.count
                    evicted.append(id)
                }
            }
            return evicted
        }
    }

    private func evictDiskIfNeeded() -> [UUID] {
        var directories: [URL] = []
        let evicted = state.withLock { state -> [UUID] in
            var evicted: [UUID] = []
            while state.diskBytes > configuration.diskLimitBytes,
                  let id = state.diskLRU.first {
                state.diskLRU.removeFirst()
                if let record = state.disk.removeValue(forKey: id) {
                    state.diskBytes -= record.metadata.descriptor.payloadBytes
                    directories.append(record.directory)
                    evicted.append(id)
                }
            }
            return evicted
        }
        for directory in directories {
            try? fileManager.removeItem(at: directory)
        }
        return evicted
    }

    private func writeDisk(entry: ServerPromptCacheEntry,
                           snapshot: InferenceStateSnapshot) throws {
        guard let root = configuration.diskDirectory else { return }
        let id = entry.id
        let finalDirectory = root.appendingPathComponent(id.uuidString.lowercased())
        let temporaryDirectory = root.appendingPathComponent(
            ".tmp-\(id.uuidString.lowercased())-\(UUID().uuidString.lowercased())")
        try fileManager.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        do {
            let payloadURL = temporaryDirectory.appendingPathComponent(Self.payloadName)
            try snapshot.payload.write(to: payloadURL, options: [.atomic])
            try fileManager.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: payloadURL.path)
            let metadata = DiskMetadata(
                version: DiskMetadata.currentVersion,
                entry: entry,
                descriptor: snapshot.descriptor,
                payloadSHA256: Self.sha256(snapshot.payload))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let metadataURL = temporaryDirectory.appendingPathComponent(Self.metadataName)
            try encoder.encode(metadata).write(to: metadataURL, options: [.atomic])
            try fileManager.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: metadataURL.path)
            // S23: never remove the live directory before the replacement is
            // in place. Swap via renames: move the old directory aside, move
            // the new one into place, then drop the backup. If the swap fails,
            // restore the old directory so the previously persisted record
            // stays loadable instead of being orphaned.
            if fileManager.fileExists(atPath: finalDirectory.path) {
                let backupDirectory = root.appendingPathComponent(
                    ".bak-\(id.uuidString.lowercased())-\(UUID().uuidString.lowercased())")
                try fileManager.moveItem(at: finalDirectory, to: backupDirectory)
                do {
                    try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
                    try? fileManager.removeItem(at: backupDirectory)
                } catch {
                    try? fileManager.moveItem(at: backupDirectory, to: finalDirectory)
                    throw error
                }
            } else {
                try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
            }
            state.withLock { state in
                if let previous = state.disk[id] {
                    state.diskBytes -= previous.metadata.descriptor.payloadBytes
                }
                let record = DiskRecord(
                    metadata: metadata,
                    directory: finalDirectory,
                    payload: finalDirectory.appendingPathComponent(Self.payloadName),
                    modificationDate: Date())
                state.disk[id] = record
                state.diskBytes += snapshot.payload.count
            }
            touchDisk(id)
        } catch {
            try? fileManager.removeItem(at: temporaryDirectory)
            throw error
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
