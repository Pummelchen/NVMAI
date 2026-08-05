import CryptoKit
import Foundation
import NVMAI

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

final class ServerPromptStateStore {
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

    private let configuration: ServerPromptCacheStorageConfiguration
    private let fileManager: FileManager
    private var memory: [UUID: InferenceStateSnapshot] = [:]
    private var memoryLRU: [UUID] = []
    private var memoryBytes = 0
    private var disk: [UUID: DiskRecord] = [:]
    private var diskLRU: [UUID] = []
    private var diskBytes = 0

    var maximumSnapshotBytes: Int {
        max(configuration.memoryLimitBytes,
            configuration.diskDirectory == nil ? 0 : configuration.diskLimitBytes)
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
        disk.removeAll(keepingCapacity: true)
        diskLRU.removeAll(keepingCapacity: true)
        diskBytes = 0
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
        for record in loaded {
            let id = record.metadata.entry.id
            disk[id] = record
            diskLRU.append(id)
            diskBytes += record.metadata.descriptor.payloadBytes
        }
        _ = evictDiskIfNeeded()
        return diskLRU.compactMap {
            guard let entry = disk[$0]?.metadata.entry,
                  entry.domain == domain else { return nil }
            return entry
        }
    }

    func save(entry: ServerPromptCacheEntry,
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

        return ServerPromptStateSaveResult(
            unbackedEntryIDs: unbacked,
            diskError: diskError,
            memoryBytes: memoryBytes,
            diskBytes: diskBytes)
    }

    func restore(entryID: UUID,
                 into runner: RealForwardRunner) throws -> String {
        let loaded = try loadSnapshot(entryID: entryID)
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
        if let snapshot = memory[entryID] {
            touchMemory(entryID)
            touchDisk(entryID)
            return (snapshot, "ram")
        }
        guard let record = disk[entryID] else {
            throw ServerPromptStateStoreError.missing(entryID)
        }
        do {
            let payload = try Data(contentsOf: record.payload, options: [.mappedIfSafe])
            guard payload.count == record.metadata.descriptor.payloadBytes else {
                throw ServerPromptStateStoreError.corrupt(entryID, "payload size mismatch")
            }
            let digest = Self.sha256(payload)
            guard digest == record.metadata.payloadSHA256 else {
                throw ServerPromptStateStoreError.corrupt(entryID, "SHA-256 mismatch")
            }
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

    func contains(_ entryID: UUID) -> Bool {
        memory[entryID] != nil || disk[entryID] != nil
    }

    func remove(entryIDs: some Sequence<UUID>) {
        for id in entryIDs {
            if let snapshot = memory.removeValue(forKey: id) {
                memoryBytes -= snapshot.payload.count
            }
            memoryLRU.removeAll { $0 == id }
            if let record = disk.removeValue(forKey: id) {
                diskBytes -= record.metadata.descriptor.payloadBytes
                try? fileManager.removeItem(at: record.directory)
            }
            diskLRU.removeAll { $0 == id }
        }
    }

    private func insertMemory(_ snapshot: InferenceStateSnapshot, id: UUID) {
        if let previous = memory.updateValue(snapshot, forKey: id) {
            memoryBytes -= previous.payload.count
        }
        memoryBytes += snapshot.payload.count
        touchMemory(id)
    }

    private func touchMemory(_ id: UUID) {
        memoryLRU.removeAll { $0 == id }
        memoryLRU.append(id)
    }

    private func touchDisk(_ id: UUID) {
        guard var record = disk[id] else { return }
        record.modificationDate = Date()
        disk[id] = record
        diskLRU.removeAll { $0 == id }
        diskLRU.append(id)
        try? fileManager.setAttributes(
            [.modificationDate: record.modificationDate],
            ofItemAtPath: record.directory.path)
    }

    private func evictMemoryIfNeeded() -> [UUID] {
        var evicted: [UUID] = []
        while memoryBytes > configuration.memoryLimitBytes,
              let id = memoryLRU.first {
            memoryLRU.removeFirst()
            if let snapshot = memory.removeValue(forKey: id) {
                memoryBytes -= snapshot.payload.count
                evicted.append(id)
            }
        }
        return evicted
    }

    private func evictDiskIfNeeded() -> [UUID] {
        var evicted: [UUID] = []
        while diskBytes > configuration.diskLimitBytes,
              let id = diskLRU.first {
            diskLRU.removeFirst()
            if let record = disk.removeValue(forKey: id) {
                diskBytes -= record.metadata.descriptor.payloadBytes
                try? fileManager.removeItem(at: record.directory)
                evicted.append(id)
            }
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
            if fileManager.fileExists(atPath: finalDirectory.path) {
                try fileManager.removeItem(at: finalDirectory)
            }
            try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
            if let previous = disk[id] {
                diskBytes -= previous.metadata.descriptor.payloadBytes
            }
            let record = DiskRecord(
                metadata: metadata,
                directory: finalDirectory,
                payload: finalDirectory.appendingPathComponent(Self.payloadName),
                modificationDate: Date())
            disk[id] = record
            diskBytes += snapshot.payload.count
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
