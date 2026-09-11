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
    /// Delay after the last append before the durability barrier is taken.
    private let idleDelay: Duration
    /// Longest a record may wait for a barrier while appends keep arriving.
    private let maximumLatency: Duration
    private var barrier: Task<Void, Never>?
    private var lastBarrier = ContinuousClock.now
    /// Every blocking syscall this type makes runs here, not on the
    /// cooperative pool. A barrier is tens of milliseconds of the drive
    /// doing nothing else; on a Mac with a two-thread pool that would be
    /// half of every actor in the process stalled behind a memory write.
    private static let blockingQueue = DispatchQueue(label: "ContinuityCore.journal",
                                                     qos: .utility)

    /// - Parameters:
    ///   - synchronizesEveryWrite: take the barrier on every append, inline.
    ///     Correct across a power cut, and slow enough that it is off by
    ///     default; a crash of the process alone loses nothing either way,
    ///     because the write has already reached the kernel.
    ///   - idleDelay: how long after the last append the barrier is taken.
    ///     The point of waiting is that an append happens while a model is
    ///     answering, and the moment after the answer is the one moment the
    ///     drive is not being asked for expert weights.
    ///   - maximumLatency: the longest a record waits for a barrier while
    ///     appends keep arriving, so a busy tool loop cannot postpone
    ///     durability indefinitely.
    /// - Throws: `JournalError.locked` when another process holds this
    ///   journal. Callers should treat that as "run without persistence and
    ///   say so", never as a reason to write anyway.
    public init(url: URL,
                synchronizesEveryWrite: Bool = false,
                idleDelay: Duration = .seconds(2),
                maximumLatency: Duration = .seconds(30)) throws {
        self.url = url
        self.lockURL = url.appendingPathExtension("lock")
        self.synchronizesEveryWrite = synchronizesEveryWrite
        self.idleDelay = idleDelay
        self.maximumLatency = maximumLatency
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
        try await afterAppend()
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
        try await afterAppend()
    }

    /// Decides when the barrier happens. Never inline unless asked for.
    ///
    /// `write(2)` to the page cache is microseconds and is all the request
    /// path ever pays. The barrier is scheduled for when appends stop, and
    /// forced only when a record has been waiting longer than the maximum.
    private func afterAppend() async throws {
        if synchronizesEveryWrite {
            try await performBarrier()
            return
        }
        let overdue = ContinuousClock.now - lastBarrier > maximumLatency
        barrier?.cancel()
        let delay = overdue ? Duration.zero : idleDelay
        barrier = Task { [weak self] in
            if delay > .zero {
                try? await Task.sleep(for: delay)
            }
            guard !Task.isCancelled, let self else { return }
            try? await self.performBarrier()
        }
    }

    /// Records written but not yet behind a barrier. For diagnostics.
    public var pendingRecords: Int { pendingSinceSync }

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

    /// Force everything written so far to the platter, now.
    ///
    /// `F_FULLFSYNC` rather than `fsync` because on Darwin `fsync` only
    /// promises the write reached the drive's cache, which a power cut can
    /// still lose. The stronger barrier is the point of calling this at all.
    public func sync() async throws {
        barrier?.cancel()
        barrier = nil
        try await performBarrier()
    }

    /// The barrier itself, on the blocking queue. The actor suspends until
    /// it is done, so appends queue behind it in order, but no pool thread is
    /// held while the drive works.
    private func performBarrier() async throws {
        guard descriptor >= 0, pendingSinceSync > 0 else { return }
        let target = descriptor
        let failure: Int32 = await withCheckedContinuation { continuation in
            Self.blockingQueue.async {
                var code: Int32 = 0
                if fcntl(target, F_FULLFSYNC) == -1 {
                    // Not every filesystem implements it; fall back rather
                    // than fail.
                    if fsync(target) != 0 { code = errno }
                }
                continuation.resume(returning: code)
            }
        }
        // The descriptor may have been swapped by a compaction while the
        // barrier was in flight; only settle the count if it was not.
        guard target == descriptor else { return }
        if failure != 0 { throw JournalError.writeFailed(url, errno: failure) }
        pendingSinceSync = 0
        lastBarrier = ContinuousClock.now
    }

    // MARK: - Reading

    public func replay() async throws -> [JournalRecord] {
        // Reading and decoding a journal is the one bulk operation this type
        // does, and it is done once, at start. It still goes to the blocking
        // queue: a large file on a slow disk must not pin a pool thread.
        let path = url.path
        let decoder = self.decoder
        return await withCheckedContinuation { continuation in
            Self.blockingQueue.async {
                guard let contents = FileManager.default.contents(atPath: path) else {
                    continuation.resume(returning: [])
                    return
                }
                continuation.resume(returning: Self.decodeRecords(contents, decoder: decoder))
            }
        }
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
        try await settleBarrier()
        var data = try encoder.encode(JournalRecord.checkpoint(sessionLog, memory))
        data.append(0x0A)
        let temporary = url.appendingPathExtension("compacting")
        let target = url
        let payload = data
        // Write, barrier, rename and directory barrier all happen on the
        // blocking queue: compaction is the largest write this type makes and
        // the one most likely to be big enough to notice.
        let outcome: Result<Void, JournalError> = await withCheckedContinuation { continuation in
            Self.blockingQueue.async {
                continuation.resume(returning: Self.writeCheckpoint(payload, to: temporary,
                                                                    replacing: target))
            }
        }
        if case .failure(let error) = outcome { throw error }

        // The lock lives on a sidecar, so swapping this file never gives it
        // up. Reopen the append descriptor on the new inode.
        //
        // Forgetting the number before reopening is the point: if
        // `openForAppend` throws, the field must not still hold the descriptor
        // just closed, or the `descriptor >= 0` guard in `writeFully` and
        // `barrier` passes and a later append writes into whatever object the
        // kernel has since handed that number to.
        if descriptor >= 0 {
            close(descriptor)
            descriptor = -1
        }
        descriptor = try Self.openForAppend(url)
        pendingSinceSync = 0
        lastBarrier = ContinuousClock.now
    }

    /// The checkpoint write, in full, for the blocking queue.
    ///
    /// Replace only once the new file is complete on disk, so a crash during
    /// compaction leaves the old journal intact, and sync the directory so
    /// the rename itself is durable rather than only the bytes it points at.
    private static func writeCheckpoint(_ data: Data, to temporary: URL,
                                        replacing target: URL) -> Result<Void, JournalError> {
        try? FileManager.default.removeItem(at: temporary)
        let handle = open(temporary.path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0o600)
        guard handle >= 0 else {
            return .failure(.cannotOpen(temporary, underlying: String(cString: strerror(errno))))
        }
        do {
            try writeFully(data, to: handle, url: temporary)
        } catch let error as JournalError {
            close(handle)
            try? FileManager.default.removeItem(at: temporary)
            return .failure(error)
        } catch {
            close(handle)
            return .failure(.writeFailed(temporary, errno: EIO))
        }
        if fcntl(handle, F_FULLFSYNC) == -1 { _ = fsync(handle) }
        close(handle)

        guard rename(temporary.path, target.path) == 0 else {
            let code = errno
            try? FileManager.default.removeItem(at: temporary)
            return .failure(.writeFailed(target, errno: code))
        }
        let directory = open(target.deletingLastPathComponent().path, O_RDONLY | O_CLOEXEC)
        if directory >= 0 {
            _ = fsync(directory)
            close(directory)
        }
        return .success(())
    }

    public func truncate() async throws {
        try await settleBarrier()
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

    /// Flush, close the journal and release the workspace lock.
    ///
    /// Named `shutDown` rather than `close` so it cannot be confused with
    /// `close(2)`, which this type calls throughout.
    public func shutDown() async throws {
        try? await settleBarrier()
        try? await performBarrier()
        if descriptor >= 0 { close(descriptor) }
        descriptor = -1
        if lockDescriptor >= 0 { close(lockDescriptor) }
        lockDescriptor = -1
    }

    /// Cancels a scheduled barrier and waits for one already running, so a
    /// descriptor is never closed or replaced underneath the drive.
    private func settleBarrier() async throws {
        barrier?.cancel()
        if let running = barrier { await running.value }
        barrier = nil
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
