import Foundation
import Testing
@testable import ContinuityCore

/// A journal that refuses writes on demand, standing in for a full disk, an
/// I/O error or a descriptor closed underneath it.
private actor RefusingJournal: ContinuityJournal {
    private(set) var records: [JournalRecord] = []
    private var refusing = false

    func refuse(_ value: Bool) { refusing = value }

    func append(_ record: JournalRecord) async throws {
        try check()
        records.append(record)
    }

    func replay() async throws -> [JournalRecord] { records }

    func compact(sessionLog: SessionLogSnapshot, memory: MemorySnapshot) async throws {
        try check()
        records = [.checkpoint(sessionLog, memory)]
    }

    func truncate() async throws { records = [] }

    private func check() throws {
        guard !refusing else {
            throw JournalError.writeFailed(URL(fileURLWithPath: "/journal.ndjson"), errno: ENOSPC)
        }
    }
}

/// RAM is the source of truth during a run, so a refused journal write
/// never loses the change in-process. What these pin down is who gets told:
/// the writer of a fact, always; the author of a session event, never.
@Suite struct JournalFailureTests {
    private func started(_ journal: RefusingJournal) async throws
        -> (engine: ContinuityEngine, task: ContinuityTask, session: Session) {
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "Disk")
        let session = try await engine.beginSession(taskID: task.id)
        return (engine, task, session)
    }

    @Test func aMemoryWriteTheJournalRefusesFailsItsCaller() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        await journal.refuse(true)

        do {
            try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
            Issue.record("a write the journal refused was reported as saved")
        } catch ContinuityError.notPersisted {}

        // Kept for the session, and the engine no longer claims the file
        // matches it.
        #expect(await engine.recall(taskID: task.id, namespace: "n", key: "k")?.value == "v")
        #expect(await engine.journalFailure != nil)

        // A later write landing does not bring back the one that did not.
        await journal.refuse(false)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k2", value: "v2")
        #expect(await engine.journalFailure != nil)
    }

    @Test func anArchiveTheJournalRefusesFailsItsCaller() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
        await journal.refuse(true)

        // An archive that does not reach the file is a delete that undoes
        // itself on restart.
        await #expect(throws: ContinuityError.self) {
            try await engine.archive(taskID: task.id, namespace: "n", key: "k")
        }
        #expect(await engine.journalFailure != nil)
    }

    @Test func aSessionEventTheJournalRefusesDoesNotFailTheTurn() async throws {
        let journal = RefusingJournal()
        let (engine, task, session) = try await started(journal)
        await journal.refuse(true)

        try await engine.recordUserPrompt(sessionID: session.id, text: "hello")
        try await engine.recordAssistantResponse(sessionID: session.id, text: "hi")

        #expect(await engine.turns(taskID: task.id).count == 1)
        #expect(await engine.journalFailure != nil)
    }

    @Test func writesThatLandLeaveNoFailure() async throws {
        let journal = RefusingJournal()
        let (engine, _, session) = try await started(journal)
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k", value: "v")
        try await engine.recordUserPrompt(sessionID: session.id, text: "hello")

        #expect(await engine.journalFailure == nil)
        let records = await journal.records
        #expect(records.contains { record in
            if case .memory(let item) = record { return item.value == "v" }
            return false
        })
    }
}
