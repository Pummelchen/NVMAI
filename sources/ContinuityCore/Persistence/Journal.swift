import Foundation

/// One durable line of the record.
///
/// The journal is a log of facts, not of commands. Replaying it applies the
/// same values again in the same order, so a replay of a replay is identical
/// and a partially written tail can be dropped without corrupting what came
/// before.
public enum JournalRecord: Codable, Sendable, Equatable {
    case task(ContinuityTask)
    case session(Session)
    case event(SessionEvent)
    case memory(MemoryItem)
    case memoryVersion(MemoryVersion)
    /// A compaction point. Everything before it in the file is redundant.
    case checkpoint(SessionLogSnapshot, MemorySnapshot)
}

/// Durable storage for the engine's state.
///
/// RAM stays the source of truth during a run. The journal exists so that
/// state survives a restart, which is the whole point of a system built for
/// work that spans months.
public protocol ContinuityJournal: Sendable {
    func append(_ record: JournalRecord) async throws
    func append(_ records: [JournalRecord]) async throws
    /// Every record in write order. A truncated final line is discarded.
    func replay() async throws -> [JournalRecord]
    /// Replace the file with a single checkpoint.
    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws
    /// Discard everything.
    func truncate() async throws
}

public extension ContinuityJournal {
    func append(_ records: [JournalRecord]) async throws {
        for record in records { try await append(record) }
    }
}

/// Keeps nothing. The default, so an engine that was never given a file does
/// not quietly write one.
public struct NullJournal: ContinuityJournal {
    public init() {}
    public func append(_ record: JournalRecord) async throws {}
    public func replay() async throws -> [JournalRecord] { [] }
    public func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {}
    public func truncate() async throws {}
}

public enum JournalError: Error, CustomStringConvertible {
    case cannotOpen(URL, underlying: String)
    case notAFile(URL)

    public var description: String {
        switch self {
        case .cannotOpen(let url, let underlying):
            return "cannot open journal at \(url.path): \(underlying)"
        case .notAFile(let url):
            return "journal path \(url.path) is not a regular file"
        }
    }
}

/// An append-only file of JSON lines.
///
/// The file holds complete user prompts and model replies. It is created
/// with owner-only permissions and never leaves the machine: nothing in this
/// package opens a socket, and no caller should hand this file to one without
/// the user deciding to.
public actor FileJournal: ContinuityJournal {
    public let url: URL
    private var handle: FileHandle?
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let synchronizesEveryWrite: Bool
    private var pendingSinceSync = 0
    private let syncInterval: Int

    /// - Parameters:
    ///   - synchronizesEveryWrite: force each append to disk. Correct across
    ///     a power cut, and slow enough that it is off by default; a crash of
    ///     the process alone loses nothing either way, because the write has
    ///     already reached the kernel.
    ///   - syncInterval: when not synchronizing every write, flush after this
    ///     many records.
    public init(url: URL,
                synchronizesEveryWrite: Bool = false,
                syncInterval: Int = 64) throws {
        self.url = url
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.syncInterval = max(1, syncInterval)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.withoutEscapingSlashes]
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
        try Self.prepare(url: url)
    }

    private static func prepare(url: URL) throws {
        let manager = FileManager.default
        let directory = url.deletingLastPathComponent()
        if !manager.fileExists(atPath: directory.path) {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        }
        var isDirectory: ObjCBool = false
        if manager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            if isDirectory.boolValue { throw JournalError.notAFile(url) }
        } else {
            guard manager.createFile(atPath: url.path, contents: nil,
                                     attributes: [.posixPermissions: 0o600]) else {
                throw JournalError.cannotOpen(url, underlying: "could not create the file")
            }
        }
    }

    private func openHandle() throws -> FileHandle {
        if let handle { return handle }
        do {
            let opened = try FileHandle(forWritingTo: url)
            try opened.seekToEnd()
            handle = opened
            return opened
        } catch {
            throw JournalError.cannotOpen(url, underlying: String(describing: error))
        }
    }

    public func append(_ record: JournalRecord) async throws {
        let handle = try openHandle()
        var data = try encoder.encode(record)
        data.append(0x0A)
        try handle.write(contentsOf: data)
        pendingSinceSync += 1
        if synchronizesEveryWrite || pendingSinceSync >= syncInterval {
            try handle.synchronize()
            pendingSinceSync = 0
        }
    }

    public func replay() async throws -> [JournalRecord] {
        guard let contents = FileManager.default.contents(atPath: url.path) else { return [] }
        var records: [JournalRecord] = []
        var start = contents.startIndex
        while let newline = contents[start...].firstIndex(of: 0x0A) {
            let line = contents[start..<newline]
            start = contents.index(after: newline)
            guard !line.isEmpty else { continue }
            // A line that will not decode is a torn tail or a record from an
            // incompatible build. Skipping is right: refusing to start
            // because of one bad line would strand every good one behind it.
            if let record = try? decoder.decode(JournalRecord.self, from: Data(line)) {
                records.append(record)
            }
        }
        return records
    }

    public func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        try handle?.close()
        handle = nil
        let temporary = url.appendingPathExtension("compacting")
        FileManager.default.createFile(atPath: temporary.path, contents: nil,
                                       attributes: [.posixPermissions: 0o600])
        let target = try FileHandle(forWritingTo: temporary)
        var data = try encoder.encode(JournalRecord.checkpoint(sessionLog, memory))
        data.append(0x0A)
        try target.write(contentsOf: data)
        try target.synchronize()
        try target.close()
        // Replace only once the new file is complete on disk, so a crash
        // during compaction leaves the old journal intact.
        _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        pendingSinceSync = 0
    }

    public func truncate() async throws {
        try handle?.close()
        handle = nil
        try Data().write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: url.path)
        pendingSinceSync = 0
    }

    public func close() throws {
        try handle?.synchronize()
        try handle?.close()
        handle = nil
    }
}
