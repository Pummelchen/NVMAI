import Foundation
import Testing
@testable import ContinuityCore

@Suite struct ContinuityEngineTests {
    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("continuity-\(UUID().uuidString)")
            .appendingPathComponent("journal.ndjson")
    }

    @Test func aSessionCanBeRecordedEndToEnd() async throws {
        let engine = ContinuityEngine()
        try await engine.start()
        let task = try await engine.createTask(title: "Pong", objective: "Two autoplayers")
        let session = try await engine.beginSession(taskID: task.id, model: "qwen35b")

        try await engine.recordUserPrompt(sessionID: session.id, text: "write it in swift")
        try await engine.remember(sessionID: session.id, namespace: "decision",
                                  key: "language", value: "Swift first, then Python, then C99",
                                  importance: 0.9)
        try await engine.recordAssistantResponse(sessionID: session.id, text: "done",
                                                 outputTokens: 1)
        _ = try await engine.endSession(session.id)

        let stats = await engine.statistics()
        #expect(stats.taskCount == 1)
        #expect(stats.sessionCount == 1)
        #expect(stats.memoryItemCount == 1)

        let item = await engine.recall(taskID: task.id, namespace: "decision", key: "language")
        #expect(item?.provenance?.sessionID == session.id)
        #expect(item?.provenance?.author == .model)

        // The write is visible in the log too, so the transcript explains the
        // change in belief without consulting the memory store.
        let kinds = await engine.events(sessionID: session.id).map(\.kind)
        #expect(kinds.contains(.memoryWritten))
    }

    @Test func rememberingRequiresAKnownSessionOrTask() async throws {
        let engine = ContinuityEngine()
        try await engine.start()
        await #expect(throws: ContinuityError.self) {
            try await engine.remember(sessionID: UUID(), namespace: "n", key: "k", value: "v")
        }
        await #expect(throws: ContinuityError.self) {
            try await engine.remember(taskID: UUID(), namespace: "n", key: "k", value: "v")
        }
    }

    @Test func contextComesBackWithWhatWentIntoIt() async throws {
        let engine = ContinuityEngine(
            configuration: ContinuityConfiguration(
                defaultBudget: ContextBudget(maxTokens: 2000,
                                             priorityNamespaces: ["decision"],
                                             recentTurnCount: 2)))
        try await engine.start()
        let task = try await engine.createTask(title: "Pong", objective: "Two autoplayers")
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.remember(sessionID: session.id, namespace: "decision", key: "size",
                                  value: "800 by 600")
        try await engine.recordUserPrompt(sessionID: session.id, text: "convert to python")
        try await engine.recordAssistantResponse(sessionID: session.id, text: "here it is")

        let snapshot = try await engine.assembleContext(taskID: task.id,
                                                        sessionID: session.id,
                                                        focus: "python port")
        #expect(snapshot.renderedContext.contains("800 by 600"))
        #expect(snapshot.renderedContext.contains("convert to python"))
        #expect(snapshot.memoryVersions["decision.size"] == 1)
        #expect(snapshot.sessionID == session.id)

        let events = await engine.events(sessionID: session.id)
        let assembled = events.last { $0.kind == .contextAssembled }
        guard case .context(let id, let count, _)? = assembled?.payload else {
            Issue.record("expected a context event")
            return
        }
        #expect(id == snapshot.id)
        #expect(count == 1)
    }

    // MARK: - Persistence

    @Test func stateSurvivesARestart() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let taskID: UUID
        let sessionID: UUID
        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let task = try await engine.createTask(title: "Novel", objective: "100 chapters")
            let session = try await engine.beginSession(taskID: task.id, model: "qwen35b")
            try await engine.recordUserPrompt(sessionID: session.id, text: "chapter one")
            try await engine.recordAssistantResponse(sessionID: session.id, text: "a storm")
            try await engine.remember(sessionID: session.id, namespace: "plot",
                                      key: "brother", value: "missing")
            try await engine.remember(sessionID: session.id, namespace: "plot",
                                      key: "brother", value: "found in act three")
            _ = try await engine.endSession(session.id)
            taskID = task.id
            sessionID = session.id
        }

        let reopened = ContinuityEngine(journal: try FileJournal(url: url))
        try await reopened.start()

        let task = await reopened.task(taskID)
        #expect(task?.title == "Novel")
        #expect(task?.objective == "100 chapters")

        let item = await reopened.recall(taskID: taskID, namespace: "plot", key: "brother")
        #expect(item?.value == "found in act three")
        #expect(item?.version == 2)

        let history = await reopened.history(taskID: taskID, namespace: "plot", key: "brother")
        #expect(history.map(\.version) == [1, 2])
        #expect(history.first?.value == "missing")

        let turns = await reopened.turns(taskID: taskID)
        #expect(turns.count == 1)
        #expect(turns.first?.prompt == "chapter one")
        #expect(turns.first?.response == "a storm")

        let sessions = await reopened.sessions(taskID: taskID)
        #expect(sessions.count == 1)
        #expect(sessions.first?.id == sessionID)
        #expect(sessions.first?.isOpen == false)
    }

    /// The failure this guards against is a crash between the last write and
    /// a clean shutdown, which is the normal way a long task ends.
    @Test func aTornFinalLineDoesNotStrandEarlierRecords() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let taskID: UUID
        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let task = try await engine.createTask(title: "Interrupted")
            let session = try await engine.beginSession(taskID: task.id)
            try await engine.remember(sessionID: session.id, namespace: "n", key: "k",
                                      value: "survived")
            taskID = task.id
        }

        var raw = try Data(contentsOf: url)
        raw.append(contentsOf: Array(#"{"memory":{"_0":{"taskI"#.utf8))
        raw.append(0x0A)
        try raw.write(to: url)

        let reopened = ContinuityEngine(journal: try FileJournal(url: url))
        try await reopened.start()
        #expect(await reopened.task(taskID)?.title == "Interrupted")
        #expect(await reopened.recall(taskID: taskID, namespace: "n", key: "k")?.value
                    == "survived")
    }

    @Test func compactionPreservesStateAndShrinksTheJournal() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let journal = try FileJournal(url: url)
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let task = try await engine.createTask(title: "Long")
        let session = try await engine.beginSession(taskID: task.id)
        for index in 0..<50 {
            try await engine.remember(sessionID: session.id, namespace: "n",
                                      key: "k\(index)", value: "v\(index)")
        }
        let before = try await journal.replay().count
        try await engine.compactJournal()
        let after = try await journal.replay()
        #expect(before > after.count)
        #expect(after.count == 1)

        let reopened = ContinuityEngine(journal: try FileJournal(url: url))
        try await reopened.start()
        #expect(await reopened.recall(taskID: task.id).count == 50)
        #expect(await reopened.task(task.id)?.title == "Long")
    }

    @Test func forgettingATaskRemovesItFromDiskToo() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let task = try await engine.createTask(title: "Private")
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.recordUserPrompt(sessionID: session.id,
                                          text: "a sentence that must not survive")
        try await engine.forget(taskID: task.id)

        let contents = String(data: try Data(contentsOf: url), encoding: .utf8) ?? ""
        #expect(contents.contains("must not survive") == false)

        let reopened = ContinuityEngine(journal: try FileJournal(url: url))
        try await reopened.start()
        #expect(await reopened.task(task.id) == nil)
    }

    /// Turning off content journalling is the difference between durable
    /// continuity and a transcript on disk, so it is checked at the byte
    /// level rather than through the API.
    @Test func contentCanBeKeptOutOfTheJournal() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let configuration = ContinuityConfiguration(journalsSessionContent: false)
        let engine = ContinuityEngine(configuration: configuration,
                                      journal: try FileJournal(url: url))
        try await engine.start()
        let task = try await engine.createTask(title: "Quiet")
        let session = try await engine.beginSession(taskID: task.id)
        try await engine.recordUserPrompt(sessionID: session.id, text: "a private sentence")
        try await engine.recordAssistantResponse(sessionID: session.id, text: "a private reply")
        try await engine.remember(sessionID: session.id, namespace: "n", key: "k",
                                  value: "a durable fact")

        let contents = String(data: try Data(contentsOf: url), encoding: .utf8) ?? ""
        #expect(contents.contains("a private sentence") == false)
        #expect(contents.contains("a private reply") == false)
        #expect(contents.contains("a durable fact"))

        let reopened = ContinuityEngine(journal: try FileJournal(url: url))
        try await reopened.start()
        #expect(await reopened.recall(taskID: task.id, namespace: "n", key: "k")?.value
                    == "a durable fact")
        #expect(await reopened.turns(taskID: task.id).isEmpty)
    }

    @Test func theJournalFileIsOwnerReadableOnly() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        _ = try FileJournal(url: url)
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let permissions = attributes[.posixPermissions] as? NSNumber
        #expect(permissions?.int16Value == 0o600)
    }

    // MARK: - Concurrency

    @Test func concurrentSessionsOnSeparateTasksDoNotInterfere() async throws {
        let engine = ContinuityEngine()
        try await engine.start()
        var tasks: [UUID] = []
        for index in 0..<8 {
            tasks.append(try await engine.createTask(title: "task \(index)").id)
        }

        await withTaskGroup(of: Void.self) { group in
            for (index, taskID) in tasks.enumerated() {
                group.addTask {
                    guard let session = try? await engine.beginSession(taskID: taskID)
                    else { return }
                    for step in 0..<25 {
                        _ = try? await engine.remember(sessionID: session.id, namespace: "n",
                                                       key: "k\(step)", value: "task\(index)")
                    }
                    _ = try? await engine.endSession(session.id)
                }
            }
        }

        for (index, taskID) in tasks.enumerated() {
            let items = await engine.recall(taskID: taskID)
            #expect(items.count == 25)
            #expect(items.allSatisfy { $0.value == "task\(index)" })
        }
    }

    @Test func concurrentWritesToOneAddressLeaveACompleteVersionChain() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxVersionsPerAddress: 64))
        let taskID = UUID()
        await withTaskGroup(of: Void.self) { group in
            for index in 0..<32 {
                group.addTask {
                    try? await memory.write(taskID: taskID, namespace: "n", key: "k",
                                            value: "v\(index)")
                }
            }
        }
        let history = await memory.history(taskID: taskID, namespace: "n", key: "k")
        #expect(history.count == 32)
        #expect(history.map(\.version) == Array(1...32))
        #expect(await memory.item(taskID: taskID, namespace: "n", key: "k")?.version == 32)
    }

    @Test func startingTwiceIsHarmless() async throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let task = try await engine.createTask(title: "Once")
        try await engine.start()
        #expect(await engine.tasks().count == 1)
        #expect(await engine.task(task.id) != nil)
    }
}
