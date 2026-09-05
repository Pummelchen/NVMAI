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
    /// Another process already has this journal open for writing.
    case locked(URL)
    case writeFailed(URL, errno: Int32)

    public var description: String {
        switch self {
        case .cannotOpen(let url, let underlying):
            return "cannot open journal at \(url.path): \(underlying)"
        case .notAFile(let url):
            return "journal path \(url.path) is not a regular file"
        case .locked(let url):
            return "another process is already writing the journal at \(url.path)"
        case .writeFailed(let url, let code):
            return "writing \(url.path) failed: \(String(cString: strerror(code)))"
        }
    }
}

/// An append-only file of JSON lines.
///
/// The file holds complete user prompts and model replies. It is created
/// with owner-only permissions and never leaves the machine: nothing in this
/// package opens a socket, and no caller should hand this file to one without
/// the user deciding to.
///
/// Exactly one process may write a given journal. That is enforced with an
/// exclusive advisory lock on a sidecar `.lock` file, taken for the life of
/// this object. Without it two servers launched from the same directory would
/// each hold their own copy of the state in memory, see none of the other's
/// writes, and interleave their appends into one file that replays as a
/// braid of two divergent histories. The lock is on a sidecar rather than on
/// the journal itself so that compaction can replace the journal without ever
/// letting go of it.
///
/// The lock is released by the kernel when the process exits, however it
/// exits, so a crash never leaves a journal that cannot be reopened.
public actor FileJournal: ContinuityJournal {
    public let url: URL
    private let lockURL: URL
    private var lockDescriptor: Int32 = -1
    private var descriptor: Int32 = -1
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let synchronizesEveryWrite: Bool
    private var pendingSinceSync = 0
    private let syncInterval: Int

    /// - Parameters:
    ///   - synchronizesEveryWrite: force each append to disk. Correct across
    ///     a power cut, and slow enough that it is off by default; a crash of
    ///     the process alone loses nothing either way, because the write has
    ///     already reached the kernel. Callers that want a middle ground call
    ///     `sync()` at a natural boundary such as the end of a session.
    ///   - syncInterval: when not synchronizing every write, flush after this
    ///     many records.
    /// - Throws: `JournalError.locked` when another process holds this
    ///   journal. Callers should treat that as "run without persistence and
    ///   say so", never as a reason to write anyway.
    public init(url: URL,
                synchronizesEveryWrite: Bool = false,
                syncInterval: Int = 64) throws {
        self.url = url
        self.lockURL = url.appendingPathExtension("lock")
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.syncInterval = max(1, syncInterval)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.withoutEscapingSlashes]
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder

        try Self.prepareDirectory(for: url)
        self.lockDescriptor = try Self.acquireLock(at: lockURL, journal: url)
        do {
            self.descriptor = try Self.openForAppend(url)
        } catch {
            close(lockDescriptor)
            lockDescriptor = -1
            throw error
        }
    }

    deinit {
        if descriptor >= 0 { close(descriptor) }
        // Closing releases the flock. Doing it explicitly rather than relying
        // on process exit means a journal dropped mid-run frees its workspace
        // for another server immediately.
        if lockDescriptor >= 0 { close(lockDescriptor) }
    }

    // MARK: - Opening

    private static func prepareDirectory(for url: URL) throws {
        let manager = FileManager.default
        let directory = url.deletingLastPathComponent()
        if !manager.fileExists(atPath: directory.path) {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        }
        var isDirectory: ObjCBool = false
        if manager.fileExists(atPath: url.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            throw JournalError.notAFile(url)
        }
    }

    private static func acquireLock(at lockURL: URL, journal: URL) throws -> Int32 {
        let descriptor = open(lockURL.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw JournalError.cannotOpen(lockURL,
                                          underlying: String(cString: strerror(errno)))
        }
        // flock is per open-file-description, so a second FileJournal on the
        // same path inside this process conflicts too. fcntl locks would not,
        // which is exactly why they are the wrong tool here.
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            close(descriptor)
            if code == EWOULDBLOCK { throw JournalError.locked(journal) }
            throw JournalError.cannotOpen(lockURL, underlying: String(cString: strerror(code)))
        }
        return descriptor
    }

    /// `O_APPEND` so every write lands at the end without a seek, which is
    /// what keeps a record from being written into the middle of another.
    private static func openForAppend(_ url: URL) throws -> Int32 {
        let descriptor = open(url.path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else {
            throw JournalError.cannotOpen(url, underlying: String(cString: strerror(errno)))
        }
        return descriptor
    }

    // MARK: - Writing

    public func append(_ record: JournalRecord) async throws {
        var data = try encoder.encode(record)
        data.append(0x0A)
        try writeFully(data)
        pendingSinceSync += 1
        if synchronizesEveryWrite || pendingSinceSync >= syncInterval {
            try flush()
        }
    }

    public func append(_ records: [JournalRecord]) async throws {
        guard !records.isEmpty else { return }
        // One write for the batch: a partial batch on disk is a torn tail the
        // replay would drop anyway, and one syscall is cheaper than many.
        var data = Data()
        for record in records {
            data.append(try encoder.encode(record))
            data.append(0x0A)
        }
        try writeFully(data)
        pendingSinceSync += records.count
        if synchronizesEveryWrite || pendingSinceSync >= syncInterval {
            try flush()
        }
    }

    /// Writes every byte or throws. A short write is normal for `write(2)` on
    /// a large buffer and silently dropping the remainder would corrupt the
    /// record that followed it.
    private func writeFully(_ data: Data) throws {
        guard descriptor >= 0 else { throw JournalError.writeFailed(url, errno: EBADF) }
        try data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let written = write(descriptor, buffer.baseAddress!.advanced(by: offset),
                                    buffer.count - offset)
                if written < 0 {
                    if errno == EINTR { continue }
                    throw JournalError.writeFailed(url, errno: errno)
                }
                offset += written
            }
        }
    }

    /// Force everything written so far to the platter.
    ///
    /// `F_FULLFSYNC` rather than `fsync` because on Darwin `fsync` only
    /// promises the write reached the drive's cache, which a power cut can
    /// still lose. The stronger barrier is the point of calling this at all.
    public func sync() throws { try flush() }

    private func flush() throws {
        guard descriptor >= 0 else { return }
        if fcntl(descriptor, F_FULLFSYNC) == -1 {
            // Not every filesystem implements it; fall back rather than fail.
            guard fsync(descriptor) == 0 else {
                throw JournalError.writeFailed(url, errno: errno)
            }
        }
        pendingSinceSync = 0
    }

    // MARK: - Reading

    public func replay() async throws -> [JournalRecord] {
        guard let contents = FileManager.default.contents(atPath: url.path) else { return [] }
        return Self.decodeRecords(contents, decoder: decoder)
    }

    /// Read a journal without opening it for writing.
    ///
    /// Takes no lock, so it is safe to point at the file of a running server.
    /// A store only its own process can look at is a store nobody can debug.
    public static func read(contentsOf url: URL) throws -> [JournalRecord] {
        guard let contents = FileManager.default.contents(atPath: url.path) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decodeRecords(contents, decoder: decoder)
    }

    /// Splits a journal into records, dropping anything that will not decode.
    ///
    /// A line that will not decode is a torn tail or a record from an
    /// incompatible build. Skipping is right: refusing to start because of one
    /// bad line would strand every good one behind it.
    static func decodeRecords(_ contents: Data, decoder: JSONDecoder) -> [JournalRecord] {
        var records: [JournalRecord] = []
        var start = contents.startIndex
        while let newline = contents[start...].firstIndex(of: 0x0A) {
            let line = contents[start..<newline]
            start = contents.index(after: newline)
            guard !line.isEmpty else { continue }
            if let record = try? decoder.decode(JournalRecord.self, from: Data(line)) {
                records.append(record)
            }
        }
        return records
    }

    // MARK: - Rewriting

    public func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        let temporary = url.appendingPathExtension("compacting")
        try? FileManager.default.removeItem(at: temporary)
        let target = open(temporary.path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0o600)
        guard target >= 0 else {
            throw JournalError.cannotOpen(temporary, underlying: String(cString: strerror(errno)))
        }
        var data = try encoder.encode(JournalRecord.checkpoint(sessionLog, memory))
        data.append(0x0A)
        do {
            try Self.writeFully(data, to: target, url: temporary)
            if fcntl(target, F_FULLFSYNC) == -1 { _ = fsync(target) }
        } catch {
            close(target)
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        close(target)

        // Replace only once the new file is complete on disk, so a crash
        // during compaction leaves the old journal intact. The lock lives on
        // a sidecar, so swapping this file never gives it up.
        guard rename(temporary.path, url.path) == 0 else {
            let code = errno
            try? FileManager.default.removeItem(at: temporary)
            throw JournalError.writeFailed(url, errno: code)
        }
        syncDirectory()

        if descriptor >= 0 { close(descriptor) }
        descriptor = try Self.openForAppend(url)
        pendingSinceSync = 0
    }

    public func truncate() async throws {
        if descriptor >= 0 { close(descriptor) }
        descriptor = -1
        let emptied = open(url.path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0o600)
        guard emptied >= 0 else {
            throw JournalError.cannotOpen(url, underlying: String(cString: strerror(errno)))
        }
        close(emptied)
        descriptor = try Self.openForAppend(url)
        pendingSinceSync = 0
    }

    /// Makes the rename itself durable. Without it the new file's contents
    /// are on disk but the directory entry pointing at them may not be.
    private func syncDirectory() {
        let directory = url.deletingLastPathComponent()
        let handle = open(directory.path, O_RDONLY | O_CLOEXEC)
        guard handle >= 0 else { return }
        _ = fsync(handle)
        close(handle)
    }

    /// Flush, close the journal and release the workspace lock.
    ///
    /// Named `shutDown` rather than `close` so it cannot be confused with
    /// `close(2)`, which this type calls throughout.
    public func shutDown() throws {
        try? flush()
        if descriptor >= 0 { close(descriptor) }
        descriptor = -1
        if lockDescriptor >= 0 { close(lockDescriptor) }
        lockDescriptor = -1
    }

    private static func writeFully(_ data: Data, to descriptor: Int32, url: URL) throws {
        try data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let written = write(descriptor, buffer.baseAddress!.advanced(by: offset),
                                    buffer.count - offset)
                if written < 0 {
                    if errno == EINTR { continue }
                    throw JournalError.writeFailed(url, errno: errno)
                }
                offset += written
            }
        }
    }
}
