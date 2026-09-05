import Foundation
import Testing
@testable import ContinuityCore

@Suite struct SessionLogTests {
    @Test func sessionLifecycleRecordsBoundaries() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "Pong", objective: "Two autoplayers")
        let session = try await log.beginSession(taskID: task.id, model: "qwen35b")
        _ = try await log.recordUserPrompt(sessionID: session.id, text: "start")
        _ = try await log.endSession(session.id)

        let kinds = await log.events(sessionID: session.id).map(\.kind)
        #expect(kinds == [.sessionStarted, .userPrompt, .sessionEnded])
        let stored = await log.session(session.id)
        #expect(stored?.isOpen == false)
    }

    @Test func beginningASessionOnAnUnknownTaskFails() async throws {
        let log = SessionLog()
        await #expect(throws: ContinuityError.self) {
            try await log.beginSession(taskID: UUID())
        }
    }

    @Test func recordingAfterTheSessionEndedFails() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "T")
        let session = try await log.beginSession(taskID: task.id)
        _ = try await log.endSession(session.id)
        await #expect(throws: ContinuityError.self) {
            try await log.recordUserPrompt(sessionID: session.id, text: "late")
        }
        await #expect(throws: ContinuityError.self) {
            try await log.endSession(session.id)
        }
    }

    /// The reason chunks are buffered rather than appended: a streamed reply
    /// must appear in the log exactly once, or every consumer has to learn to
    /// deduplicate it.
    @Test func streamedReplyIsRecordedOnce() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "T")
        let session = try await log.beginSession(taskID: task.id)
        let response = try await log.beginAssistantResponse(sessionID: session.id,
                                                            model: "qwen35b")
        for piece in ["Hello", ", ", "world"] {
            try await log.appendAssistantChunk(responseID: response, text: piece)
        }
        _ = try await log.completeAssistantResponse(responseID: response,
                                                    outputTokens: 3,
                                                    finishReason: "stop")

        let transcript = await log.transcript(sessionID: session.id)
        let replies = transcript.filter { $0.kind == .assistantResponseCompleted }
        #expect(replies.count == 1)
        #expect(replies.first?.payload.text == "Hello, world")
        #expect(transcript.allSatisfy { $0.kind != .assistantResponseChunk })

        guard case .response(let record)? = replies.first?.payload else {
            Issue.record("expected a response payload")
            return
        }
        #expect(record.model == "qwen35b")
        #expect(record.outputTokens == 3)
        #expect(record.finishReason == "stop")
        #expect((record.latencyMilliseconds ?? -1) >= 0)
    }

    @Test func persistedChunksAreFoldedAwayOnceTheReplyCompletes() async throws {
        let log = SessionLog(options: SessionLogOptions(persistsChunks: true))
        let task = await log.createTask(title: "T")
        let session = try await log.beginSession(taskID: task.id)
        let response = try await log.beginAssistantResponse(sessionID: session.id)
        try await log.appendAssistantChunk(responseID: response, text: "a")
        try await log.appendAssistantChunk(responseID: response, text: "b")

        // Before completion the chunks are the only record of the reply, so
        // they survive folding. That is the case this option exists for.
        var folded = await log.transcript(sessionID: session.id)
        #expect(folded.filter { $0.kind == .assistantResponseChunk }.count == 2)

        _ = try await log.completeAssistantResponse(responseID: response)
        folded = await log.transcript(sessionID: session.id)
        #expect(folded.filter { $0.kind == .assistantResponseChunk }.isEmpty)
        #expect(folded.filter { $0.kind == .assistantResponseCompleted }.count == 1)

        // The raw log still holds them; folding is a read-time concern.
        let raw = await log.events(sessionID: session.id)
        #expect(raw.filter { $0.kind == .assistantResponseChunk }.count == 2)
    }

    @Test func endingASessionClosesAReplyStillStreaming() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "T")
        let session = try await log.beginSession(taskID: task.id)
        let response = try await log.beginAssistantResponse(sessionID: session.id)
        try await log.appendAssistantChunk(responseID: response, text: "partial")
        _ = try await log.endSession(session.id)

        let transcript = await log.transcript(sessionID: session.id)
        let reply = transcript.first { $0.kind == .assistantResponseCompleted }
        #expect(reply?.payload.text == "partial")
    }

    @Test func turnsPairPromptsWithRepliesAcrossSessions() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "Novel")

        let first = try await log.beginSession(taskID: task.id)
        _ = try await log.recordUserPrompt(sessionID: first.id, text: "chapter one")
        _ = try await log.recordAssistantResponse(sessionID: first.id,
                                                  ResponseRecord(text: "written"))
        _ = try await log.endSession(first.id)

        let second = try await log.beginSession(taskID: task.id)
        _ = try await log.recordUserPrompt(sessionID: second.id, text: "chapter two")

        let turns = await log.turns(taskID: task.id)
        #expect(turns.count == 2)
        #expect(turns[0].prompt == "chapter one")
        #expect(turns[0].response == "written")
        #expect(turns[1].prompt == "chapter two")
        // A prompt with no reply is kept, not dropped: an unanswered request
        // is exactly the thing a resumed session needs to see.
        #expect(turns[1].response == nil)

        let limited = await log.turns(taskID: task.id, limit: 1)
        #expect(limited.count == 1)
        #expect(limited[0].prompt == "chapter two")
    }

    @Test func forgettingATaskRemovesItsSessionsAndEvents() async throws {
        let log = SessionLog()
        let kept = await log.createTask(title: "Keep")
        let dropped = await log.createTask(title: "Drop")
        let keptSession = try await log.beginSession(taskID: kept.id)
        let droppedSession = try await log.beginSession(taskID: dropped.id)
        _ = try await log.recordUserPrompt(sessionID: droppedSession.id, text: "secret")

        await log.forget(taskID: dropped.id)

        #expect(await log.task(dropped.id) == nil)
        #expect(await log.session(droppedSession.id) == nil)
        #expect(await log.events(sessionID: droppedSession.id).isEmpty)
        #expect(await log.task(kept.id) != nil)
        #expect(await log.events(sessionID: keptSession.id).isEmpty == false)
    }

    @Test func snapshotRoundTripsThroughRestore() async throws {
        let log = SessionLog()
        let task = await log.createTask(title: "T", objective: "O")
        let session = try await log.beginSession(taskID: task.id)
        _ = try await log.recordUserPrompt(sessionID: session.id, text: "hello")
        let snapshot = await log.snapshot()

        let restored = SessionLog()
        await restored.restore(snapshot)

        #expect(await restored.task(task.id)?.objective == "O")
        #expect(await restored.sessions(taskID: task.id).count == 1)
        #expect(await restored.events(sessionID: session.id).count == 2)
        #expect(await restored.turns(taskID: task.id).first?.prompt == "hello")
    }

    @Test func observerSeesEveryAppendedEvent() async throws {
        let collector = EventCollector()
        let log = SessionLog()
        await log.setObserver { event in await collector.add(event) }
        let task = await log.createTask(title: "T")
        let session = try await log.beginSession(taskID: task.id)
        _ = try await log.recordUserPrompt(sessionID: session.id, text: "x")
        _ = try await log.endSession(session.id)

        let seen = await collector.kinds
        #expect(seen == [.sessionStarted, .userPrompt, .sessionEnded])
    }
}

actor EventCollector {
    private(set) var events: [SessionEvent] = []
    var kinds: [SessionEventKind] { events.map(\.kind) }
    func add(_ event: SessionEvent) { events.append(event) }
}
