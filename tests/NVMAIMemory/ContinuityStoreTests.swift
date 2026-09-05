import Foundation
import Testing
import ContinuityCore
@testable import NVMAIMemory

/// The durable backend, which used to be Valkey and is now the in-process
/// continuity engine.
///
/// It has to satisfy the same contract `InMemoryStore` does, and two things
/// the reference store never had to: an address mapping that round-trips, and
/// state that survives a restart.
@Suite struct ContinuityStoreTests {
    private func scope(_ workspace: String = "repo-a",
                       user: String = "local",
                       namespace: String = "nvmai") throws -> MemoryScope {
        try MemoryScope(namespace: namespace, user: user, workspace: workspace)
    }

    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    private func makeStore() -> ContinuityStore {
        ContinuityStore(engine: ContinuityEngine())
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("continuity-store-\(UUID().uuidString)")
    }

    // MARK: - The contract

    @Test func setThenGetPreservesEveryField() async throws {
        let store = makeStore()
        let scope = try scope()
        let record = MemoryRecord(key: try key("decisions/sync"),
                                  value: "FooManager stays; it prevents a sync race.",
                                  importance: 0.9,
                                  confidence: 0.6,
                                  tags: ["sync", "concurrency"],
                                  sourceSession: "session-1")
        try await store.set(record, in: scope)

        let loaded = try #require(try await store.get(try key("decisions/sync"), in: scope))
        #expect(loaded.value == record.value)
        #expect(loaded.importance == 0.9)
        // Confidence and the writing session have no home in the engine's own
        // model, so the record is stored whole. If that ever regresses the
        // fields come back nil rather than wrong.
        #expect(loaded.confidence == 0.6)
        #expect(loaded.sourceSession == "session-1")
        #expect(loaded.tags == ["sync", "concurrency"])
    }

    @Test func scopesDoNotSeeEachOther() async throws {
        let store = makeStore()
        let first = try scope("repo-a")
        let second = try scope("repo-b")
        try await store.set(MemoryRecord(key: try key("k"), value: "a"), in: first)
        try await store.set(MemoryRecord(key: try key("k"), value: "b"), in: second)
        #expect(try await store.get(try key("k"), in: first)?.value == "a")
        #expect(try await store.get(try key("k"), in: second)?.value == "b")

        let byUser = try scope("repo-a", user: "other")
        #expect(try await store.get(try key("k"), in: byUser) == nil)
        let byNamespace = try scope("repo-a", namespace: "other")
        #expect(try await store.get(try key("k"), in: byNamespace) == nil)
    }

    @Test func missingKeysReadAsNil() async throws {
        let store = makeStore()
        #expect(try await store.get(try key("nothing/here"), in: try scope()) == nil)
        #expect(try await store.exists(try key("nothing/here"), in: try scope()) == false)
    }

    @Test func deleteReportsWhetherSomethingWasThere() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("gotchas/build"), value: "x"), in: scope)
        #expect(try await store.delete(try key("gotchas/build"), in: scope))
        #expect(try await store.get(try key("gotchas/build"), in: scope) == nil)
        // Deleting twice is not an error, and the second call says nothing
        // was removed.
        #expect(try await store.delete(try key("gotchas/build"), in: scope) == false)
    }

    @Test func listFiltersByPrefixNewestFirst() async throws {
        let store = makeStore()
        let scope = try scope()
        for name in ["decisions/a", "decisions/b", "gotchas/c"] {
            try await store.set(MemoryRecord(key: try key(name), value: name), in: scope)
        }
        let decisions = try await store.list(prefix: "decisions/", limit: 10, in: scope)
        #expect(Set(decisions.map(\.rawValue)) == ["decisions/a", "decisions/b"])
        let all = try await store.list(prefix: "", limit: 10, in: scope)
        #expect(all.count == 3)
        let limited = try await store.list(prefix: "", limit: 1, in: scope)
        #expect(limited.count == 1)
    }

    @Test func searchRanksByRelevance() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("decisions/sync"),
                                         value: "the sync race is prevented by FooManager",
                                         importance: 0.9, tags: ["sync"]), in: scope)
        try await store.set(MemoryRecord(key: try key("notes/colour"),
                                         value: "the palette is warm"), in: scope)

        let hits = try await store.search(MemoryQuery(text: "sync", limit: 5), in: scope)
        #expect(hits.first?.key.rawValue == "decisions/sync")
        // A query the store cannot match returns nothing rather than
        // everything, which is the failure that quietly fills a context window.
        let empty = try await store.search(MemoryQuery(text: "kubernetes", limit: 5), in: scope)
        #expect(empty.isEmpty)
    }

    @Test func appendExtendsAnExistingRecordAndCreatesAMissingOne() async throws {
        let store = makeStore()
        let scope = try scope()
        let created = try await store.append("first", to: try key("log/notes"), in: scope)
        #expect(created.value == "first")
        let extended = try await store.append("second", to: try key("log/notes"), in: scope)
        #expect(extended.value == "first\nsecond")
        #expect(try await store.get(try key("log/notes"), in: scope)?.value == "first\nsecond")
    }

    @Test func oversizedValuesAreRefusedWithTheCallersLimit() async throws {
        let store = ContinuityStore(engine: ContinuityEngine(),
                                    limits: NVMAIMemory.MemoryLimits(maximumValueBytes: 64))
        let scope = try scope()
        await #expect(throws: MemoryError.self) {
            try await store.set(MemoryRecord(key: try self.key("big"),
                                             value: String(repeating: "x", count: 65)),
                                in: scope)
        }
    }

    @Test func bootstrapIsBounded() async throws {
        let limits = NVMAIMemory.MemoryLimits(bootstrapRecords: 2, bootstrapBytes: 4096)
        let store = ContinuityStore(engine: ContinuityEngine(), limits: limits)
        let scope = try scope()
        for index in 0..<6 {
            try await store.set(MemoryRecord(key: try key("k\(index)"),
                                             value: "value \(index)",
                                             importance: Double(index) / 10),
                                in: scope)
        }
        let bootstrap = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        #expect(bootstrap.records.count == 2)
        #expect(bootstrap.omittedCount == 4)
        // The most important records are the ones that survive the bound.
        #expect(bootstrap.records.first?.key.rawValue == "k5")
    }

    // MARK: - Address mapping

    @Test func keysMapToAddressesAndBack() throws {
        let mapped = ContinuityStore.address(for: try key("decisions/sync"))
        #expect(mapped == ContinuityStore.Address(namespace: "k.decisions", key: "sync"))

        let deep = ContinuityStore.address(for: try key("a/b/c/d"))
        #expect(deep == ContinuityStore.Address(namespace: "k.a.b.c", key: "d"))

        // A one-segment key does not collide with a two-segment one, which is
        // what the leading marker segment is for.
        let flat = ContinuityStore.address(for: try key("sync"))
        #expect(flat == ContinuityStore.Address(namespace: "k", key: "sync"))
        #expect(flat != mapped)
    }

    @Test func foldingIsIdempotentSoAReturnedKeyStillResolves() async throws {
        let store = makeStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("Decisions/Sync.v2"), value: "held"),
                            in: scope)

        // The stored key is the folded one, and using it verbatim works.
        let listed = try await store.list(prefix: "", limit: 5, in: scope)
        #expect(listed.map(\.rawValue) == ["decisions/sync-v2"])
        let round = try await store.get(listed[0], in: scope)
        #expect(round?.value == "held")
        // And the original spelling still finds it.
        #expect(try await store.get(try key("Decisions/Sync.v2"), in: scope)?.value == "held")
    }

    @Test func aScopeAlwaysResolvesToTheSameTask() throws {
        let first = ContinuityStore.taskIdentifier(for: try scope("repo-a"))
        let again = ContinuityStore.taskIdentifier(for: try scope("repo-a"))
        let other = ContinuityStore.taskIdentifier(for: try scope("repo-b"))
        #expect(first == again)
        #expect(first != other)
    }

    // MARK: - Durability

    @Test func factsSurviveARestart() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.ndjson")
        let scope = try scope()

        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let store = ContinuityStore(engine: engine)
            try await store.set(MemoryRecord(key: try key("decisions/storage"),
                                             value: "native swift, same process",
                                             importance: 0.95,
                                             tags: ["architecture"]),
                                in: scope)
        }

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        let loaded = try #require(try await store.get(try key("decisions/storage"), in: scope))
        #expect(loaded.value == "native swift, same process")
        #expect(loaded.importance == 0.95)
        #expect(loaded.tags == ["architecture"])
    }

    /// A delete has to be a delete from the model's point of view even though
    /// the engine keeps the chain, or a "forget that" leaves the fact in every
    /// later prompt.
    @Test func aDeletedKeyStaysGoneAcrossARestart() async throws {
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.ndjson")
        let scope = try scope()

        do {
            let engine = ContinuityEngine(journal: try FileJournal(url: url))
            try await engine.start()
            let store = ContinuityStore(engine: engine)
            try await store.set(MemoryRecord(key: try key("temp/thing"), value: "x"), in: scope)
            #expect(try await store.delete(try key("temp/thing"), in: scope))
        }

        let engine = ContinuityEngine(journal: try FileJournal(url: url))
        try await engine.start()
        let store = ContinuityStore(engine: engine)
        #expect(try await store.get(try key("temp/thing"), in: scope) == nil)
        #expect(try await store.list(prefix: "", limit: 10, in: scope).isEmpty)
    }
}

@Suite struct ContinuityJournalStoreTests {
    private func scope() throws -> MemoryScope {
        try MemoryScope(namespace: "nvmai", user: "local", workspace: "repo-a")
    }

    private func pair() -> (ContinuityStore, ContinuityJournalStore) {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        return (store, ContinuityJournalStore(engine: engine, store: store))
    }

    private func turn(_ index: Int,
                      session: String = "s1",
                      prompt: String,
                      reply: String) -> JournalTurn {
        JournalTurn(session: session, workspace: "repo-a", index: index,
                    prompt: prompt, reply: reply, model: "qwen35b",
                    promptTokens: 10, completionTokens: 20,
                    latencyMilliseconds: 30, stopReason: "stop")
    }

    @Test func turnsComeBackNewestFirstWithTheirMeasurements() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(turn(0, prompt: "first", reply: "one"), in: scope)
        await journal.record(turn(1, prompt: "second", reply: "two"), in: scope)

        let stored = await journal.turns(session: "s1", limit: 10, in: scope)
        #expect(stored.map(\.prompt) == ["second", "first"])
        #expect(stored.first?.reply == "two")
        #expect(stored.first?.model == "qwen35b")
        #expect(stored.first?.completionTokens == 20)
        #expect(stored.first?.stopReason == "stop")
        #expect(stored.first?.workspace == "repo-a")
    }

    @Test func sessionsAreSummarisedNewestFirst() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(turn(0, session: "s1", prompt: "a", reply: "1"), in: scope)
        await journal.record(turn(0, session: "s2", prompt: "b", reply: "2"), in: scope)
        await journal.record(turn(1, session: "s2", prompt: "c", reply: "3"), in: scope)

        let summaries = await journal.sessions(limit: 10, in: scope)
        #expect(Set(summaries.map(\.session)) == ["s1", "s2"])
        let second = try #require(summaries.first { $0.session == "s2" })
        #expect(second.turnCount == 2)
        #expect(second.model == "qwen35b")
    }

    @Test func searchFindsTurnsAcrossSessions() async throws {
        let (_, journal) = pair()
        let scope = try scope()
        await journal.record(turn(0, session: "s1", prompt: "about sync races",
                                  reply: "kept FooManager"), in: scope)
        await journal.record(turn(0, session: "s2", prompt: "about colour",
                                  reply: "warm palette"), in: scope)

        let hits = await journal.search("SYNC", limit: 10, in: scope)
        #expect(hits.count == 1)
        #expect(hits.first?.session == "s1")
        #expect(await journal.search("kubernetes", limit: 10, in: scope).isEmpty)
    }

    /// The two stores describe one session, not two that share a name. A
    /// fact written during a conversation and the turns of that conversation
    /// have to be attributable to each other.
    @Test func theJournalReusesTheSessionTheStoreOpened() async throws {
        let (store, journal) = pair()
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "shared", modelID: "qwen35b"),
                                        in: scope)
        await journal.record(turn(0, session: "shared", prompt: "hello", reply: "hi"),
                             in: scope)

        let summaries = await journal.sessions(limit: 10, in: scope)
        #expect(summaries.count == 1)
        #expect(summaries.first?.session == "shared")
        #expect(summaries.first?.turnCount == 1)
    }

    @Test func retentionBoundsTurnsPerSession() async throws {
        let engine = ContinuityEngine()
        let store = ContinuityStore(engine: engine)
        let journal = ContinuityJournalStore(engine: engine, store: store,
                                             limits: JournalLimits(turnsPerSession: 3))
        let scope = try scope()
        for index in 0..<10 {
            await journal.record(turn(index, prompt: "ask \(index)", reply: "reply \(index)"),
                                 in: scope)
        }
        let stored = await journal.turns(session: "s1", limit: 50, in: scope)
        #expect(stored.count == 3)
        // The newest survive, which is the only useful direction to trim.
        #expect(stored.first?.prompt == "ask 9")
        #expect(stored.last?.prompt == "ask 7")
    }
}
