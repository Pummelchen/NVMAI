import Foundation
import Testing
import ContinuityCore
@testable import NVMAIMemory

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

/// `memory_set` answering "stored" is a promise that the fact survives a
/// restart. These drive the real engine and store over a journal that
/// refuses on demand, which is the only way to see what the model is told
/// when the disk fills in the middle of a session.
@Suite struct MemoryJournalFailureTests {
    private struct Harness {
        let journal: RefusingJournal
        let service: MemoryService
        let events: LogCollector
        let context: MemorySessionContext
    }

    private func harness(journalingTurns: Bool = false) async throws -> Harness {
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let journal = RefusingJournal()
        let engine = ContinuityEngine(journal: journal)
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let events = LogCollector()
        let service = MemoryService(
            configuration: configuration,
            durableStore: store,
            journal: journalingTurns ? ContinuityJournalStore(engine: engine, store: store) : nil,
            log: { events.append($0) })
        let context = try #require(await service.beginSession(id: "s1"))
        return Harness(journal: journal, service: service, events: events, context: context)
    }

    private func set(_ key: String, _ value: String, in harness: Harness) async
        -> MemoryToolResult {
        await harness.service.execute(name: "memory_set",
                                      arguments: ["key": .string(key), "value": .string(value)],
                                      in: harness.context)
    }

    private func journalFailureLines(_ harness: Harness) -> Int {
        harness.events.messages().filter { $0.contains("degraded during journal") }.count
    }

    @Test func aWriteThatReachesTheJournalIsStillReportedAsStored() async throws {
        let harness = try await harness()
        let result = await set("decisions/db", "postgres", in: harness)

        #expect(result.jsonString().contains("\"stored\":true"))
        #expect(harness.context.isDurable)
        #expect(await harness.service.isDurable)
        #expect(!harness.events.messages().contains { $0.contains("degraded") })
        let records = await harness.journal.records
        #expect(records.contains { record in
            if case .memory(let item) = record { return item.value == "postgres" }
            return false
        })
    }

    @Test func aSetTheJournalRefusesIsAFailureAndNotDurable() async throws {
        let harness = try await harness()
        _ = await set("decisions/db", "postgres", in: harness)
        await harness.journal.refuse(true)

        let refused = await set("decisions/cache", "redis", in: harness)
        #expect(refused.isFailure)
        #expect(!refused.jsonString().contains("\"stored\":true"))
        #expect(await harness.service.isDurable == false)

        // Not sent to the empty local store: the engine still holds every
        // fact, and a local retry would have answered "stored".
        let again = await set("decisions/queue", "sqs", in: harness)
        #expect(again.isFailure)
        let read = await harness.service.execute(name: "memory_get",
                                                 arguments: ["key": .string("decisions/db")],
                                                 in: harness.context)
        #expect(read.jsonString().contains("postgres"))
        #expect(journalFailureLines(harness) == 1)

        // The disk recovering does not make the lost write durable, so the
        // next session is not told its memory persists.
        await harness.journal.refuse(false)
        let next = try #require(await harness.service.beginSession(id: "s2"))
        #expect(!next.isDurable)
        #expect(journalFailureLines(harness) == 1)
    }

    @Test func aDeleteTheJournalRefusesIsAFailure() async throws {
        let harness = try await harness()
        _ = await set("decisions/db", "postgres", in: harness)
        await harness.journal.refuse(true)

        let result = await harness.service.execute(name: "memory_delete",
                                                   arguments: ["key": .string("decisions/db")],
                                                   in: harness.context)
        #expect(result.isFailure)
        #expect(!result.jsonString().contains("\"deleted\":true"))
        #expect(await harness.service.isDurable == false)
    }

    @Test func aTurnTheJournalRefusesFlipsDurabilityWithoutFailingIt() async throws {
        let harness = try await harness(journalingTurns: true)
        await harness.journal.refuse(true)

        for index in 0..<2 {
            await harness.service.recordTurn(session: harness.context, index: index,
                                             prompt: "prompt \(index)", reply: "reply \(index)",
                                             model: nil, promptTokens: 1, completionTokens: 1,
                                             latencyMilliseconds: 1, stopReason: "stop")
        }

        #expect(await harness.service.isDurable == false)
        #expect(journalFailureLines(harness) == 1)
    }
}

/// Collects log events from the service's `@Sendable` callback.
///
/// unchecked-invariant: every access is under `lock`.
private final class LogCollector: @unchecked Sendable {
    private var events: [MemoryLogEvent] = []
    private let lock = NSLock()

    func append(_ event: MemoryLogEvent) {
        lock.withLock { events.append(event) }
    }

    func messages() -> [String] {
        lock.withLock { events.map(\.message) }
    }
}
