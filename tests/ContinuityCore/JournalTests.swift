import Foundation
import Testing
@testable import ContinuityCore

/// The persistence layer on its own: the workspace lock, durability, and what
/// a damaged file does on the way back in.
@Suite struct JournalTests {
    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("journal-\(UUID().uuidString)")
    }

    // MARK: - The workspace lock

    /// The failure this prevents: two servers launched from the same
    /// directory, each holding its own copy of the state, seeing none of the
    /// other's writes, interleaving appends into one file that replays as a
    /// braid of two histories.
    @Test func aSecondWriterIsRefusedWhileTheFirstHoldsTheJournal() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")

        let first = try FileJournal(url: url)
        #expect(throws: JournalError.self) {
            _ = try FileJournal(url: url)
        }
        do {
            _ = try FileJournal(url: url)
            Issue.record("a second writer was allowed")
        } catch let error as JournalError {
            guard case .locked = error else {
                Issue.record("expected a lock refusal, got \(error)")
                return
            }
            // The message has to name the path, because the person reading it
            // is trying to work out which other process has the workspace.
            #expect(error.description.contains(url.path))
        }

        // Once the holder lets go, the workspace is available again.
        try await first.shutDown()
        let second = try FileJournal(url: url)
        try await second.append(.task(ContinuityTask(title: "after")))
        try await second.shutDown()
    }

    @Test func aDifferentWorkspaceIsNotBlocked() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try FileJournal(url: directory.appendingPathComponent("a.ndjson"))
        let second = try FileJournal(url: directory.appendingPathComponent("b.ndjson"))
        try await first.append(.task(ContinuityTask(title: "a")))
        try await second.append(.task(ContinuityTask(title: "b")))
        #expect(try await first.replay().count == 1)
        #expect(try await second.replay().count == 1)
        try await first.shutDown()
        try await second.shutDown()
    }

    /// The lock lives on a sidecar precisely so this works: compaction
    /// replaces the journal file, and the workspace stays held throughout.
    @Test func compactionKeepsTheLock() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        for index in 0..<10 {
            try await journal.append(.task(ContinuityTask(title: "t\(index)")))
        }
        try await journal.compact(sessionLog: SessionLogSnapshot(), memory: MemorySnapshot())
        #expect(try await journal.replay().count == 1)

        // Still the holder, and still writable afterwards.
        #expect(throws: JournalError.self) { _ = try FileJournal(url: url) }
        try await journal.append(.task(ContinuityTask(title: "after compaction")))
        #expect(try await journal.replay().count == 2)
        try await journal.shutDown()
    }

    // MARK: - Durability and damage

    @Test func recordsSurviveAndReplayInOrder() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url, synchronizesEveryWrite: true)
        let task = ContinuityTask(title: "ordered")
        try await journal.append([.task(task),
                                  .memory(MemoryItem(taskID: task.id, namespace: "n",
                                                     key: "k", value: "one"))])
        try await journal.sync()
        try await journal.shutDown()

        let reopened = try FileJournal(url: url)
        let records = try await reopened.replay()
        #expect(records.count == 2)
        guard case .task(let first) = records[0] else {
            Issue.record("expected the task first")
            return
        }
        #expect(first.title == "ordered")
        try await reopened.shutDown()
    }

    @Test func aTornTailIsDroppedAndTheRestSurvives() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(.task(ContinuityTask(title: "kept")))
        try await journal.sync()
        try await journal.shutDown()

        var raw = try Data(contentsOf: url)
        raw.append(contentsOf: Array(#"{"task":{"_0":{"tit"#.utf8))
        raw.append(0x0A)
        // Also a line that is valid JSON but not a record, which is what a
        // future build's extra case would look like to an older one.
        raw.append(contentsOf: Array(#"{"somethingElse":{}}"#.utf8))
        raw.append(0x0A)
        try raw.write(to: url)

        let reopened = try FileJournal(url: url)
        let records = try await reopened.replay()
        #expect(records.count == 1)
        try await reopened.shutDown()
    }

    @Test func anUnterminatedFinalLineIsDropped() async throws {
        // A record and its newline go out in one write, so a line without a
        // terminator is by definition a partial write. Dropping it is the
        // safe reading; keeping it would mean decoding half a record.
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(.task(ContinuityTask(title: "first")))
        try await journal.shutDown()

        var raw = try Data(contentsOf: url)
        let complete = raw.count
        raw.append(contentsOf: Array(#"{"task":{"_0":{"broken"#.utf8))
        try raw.write(to: url)
        #expect(raw.count > complete)

        let reopened = try FileJournal(url: url)
        #expect(try await reopened.replay().count == 1)
        try await reopened.shutDown()
    }

    @Test func theJournalAndItsLockAreOwnerOnly() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("nested").appendingPathComponent("j.ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(.task(ContinuityTask(title: "t")))

        let manager = FileManager.default
        for path in [url.path, url.appendingPathExtension("lock").path] {
            let attributes = try manager.attributesOfItem(atPath: path)
            #expect((attributes[.posixPermissions] as? NSNumber)?.int16Value == 0o600,
                    "\(path) should be owner-only")
        }
        // The directory the engine created is owner-only too, or the
        // permissions on the file inside it are decorative.
        let directoryAttributes = try manager
            .attributesOfItem(atPath: url.deletingLastPathComponent().path)
        #expect((directoryAttributes[.posixPermissions] as? NSNumber)?.int16Value == 0o700)
        try await journal.shutDown()
    }

    @Test func truncateEmptiesWithoutLosingTheLock() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        try await journal.append(.task(ContinuityTask(title: "gone")))
        try await journal.truncate()
        #expect(try await journal.replay().isEmpty)
        #expect(throws: JournalError.self) { _ = try FileJournal(url: url) }
        try await journal.append(.task(ContinuityTask(title: "new")))
        #expect(try await journal.replay().count == 1)
        try await journal.shutDown()
    }

    /// A record larger than a pipe buffer exercises the short-write loop,
    /// which is the difference between a durable log and a subtly corrupt one.
    @Test func aLargeRecordIsWrittenWhole() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("journal.ndjson")
        let journal = try FileJournal(url: url)
        let big = String(repeating: "x", count: 2 << 20)
        let task = UUID()
        try await journal.append(.memory(MemoryItem(taskID: task, namespace: "n",
                                                    key: "k", value: big)))
        try await journal.sync()
        try await journal.shutDown()

        let reopened = try FileJournal(url: url)
        let records = try await reopened.replay()
        #expect(records.count == 1)
        guard case .memory(let item)? = records.first else {
            Issue.record("expected the record back")
            return
        }
        #expect(item.value.count == big.count)
        try await reopened.shutDown()
    }
}
