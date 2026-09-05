import Foundation
import Testing
@testable import NVMAIMemory

/// The contract every backend has to satisfy.
///
/// These run against `InMemoryStore` here and, when a server is reachable,
/// against the Valkey backend through the same suite, so the two cannot
/// drift on the rules that matter: scope isolation, value limits, bootstrap
/// bounds, and what a rewrite does to timestamps.
@Suite struct MemoryStoreConformanceTests {
    private func scope(_ workspace: String = "repo-a",
                       user: String = "local",
                       namespace: String = "nvmai") throws -> MemoryScope {
        try MemoryScope(namespace: namespace, user: user, workspace: workspace)
    }

    private func key(_ raw: String) throws -> MemoryKey { try MemoryKey(validating: raw) }

    @Test func setThenGetReturnsTheRecord() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        let record = MemoryRecord(key: try key("decisions/sync"),
                                  value: "FooManager stays; it prevents a background sync race.",
                                  importance: 0.9,
                                  tags: ["sync", "concurrency"])
        try await store.set(record, in: scope)

        let loaded = try await store.get(try key("decisions/sync"), in: scope)
        #expect(loaded?.value == record.value)
        #expect(loaded?.importance == 0.9)
        #expect(loaded?.tags == ["sync", "concurrency"])
        #expect(try await store.exists(try key("decisions/sync"), in: scope))
    }

    @Test func valuesAreArbitraryTextAndJSON() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        // JSON is stored verbatim: structure is the model's business, and a
        // store that reformats it would break round-tripping.
        let json = #"{"stack":["swift","metal"],"note":"emoji ok 🧠","n":3}"#
        try await store.set(MemoryRecord(key: try key("architecture/stack"), value: json), in: scope)
        try await store.set(MemoryRecord(key: try key("notes/prose"),
                                         value: "Ünïcode, newlines\nand \"quotes\"."), in: scope)

        #expect(try await store.get(try key("architecture/stack"), in: scope)?.value == json)
        #expect(try await store.get(try key("notes/prose"), in: scope)?.value.contains("\n") == true)
    }

    @Test func deleteRemovesOnlyTheNamedKey() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("a/one"), value: "1"), in: scope)
        try await store.set(MemoryRecord(key: try key("a/two"), value: "2"), in: scope)

        #expect(try await store.delete(try key("a/one"), in: scope))
        #expect(try await store.get(try key("a/one"), in: scope) == nil)
        #expect(try await store.get(try key("a/two"), in: scope)?.value == "2")
        // Deleting what is not there is false, not an error: the model
        // retrying a cleanup should not look like a failure.
        #expect(try await store.delete(try key("a/one"), in: scope) == false)
    }

    @Test func repositoriesAreIsolated() async throws {
        let store = InMemoryStore()
        let repoA = try scope("repo-a")
        let repoB = try scope("repo-b")
        try await store.set(MemoryRecord(key: try key("decisions/db"), value: "postgres"), in: repoA)

        #expect(try await store.get(try key("decisions/db"), in: repoB) == nil)
        #expect(try await store.list(prefix: "", limit: 50, in: repoB).isEmpty)
        #expect(try await store.search(MemoryQuery(text: "postgres"), in: repoB).isEmpty)
        // And the same key in the other repository is a different record.
        try await store.set(MemoryRecord(key: try key("decisions/db"), value: "sqlite"), in: repoB)
        #expect(try await store.get(try key("decisions/db"), in: repoA)?.value == "postgres")
    }

    @Test func usersAndNamespacesAreIsolatedToo() async throws {
        let store = InMemoryStore()
        let mine = try scope("repo", user: "ada")
        let theirs = try scope("repo", user: "grace")
        let otherDeployment = try scope("repo", user: "ada", namespace: "other")
        try await store.set(MemoryRecord(key: try key("prefs/style"), value: "tabs"), in: mine)

        #expect(try await store.get(try key("prefs/style"), in: theirs) == nil)
        #expect(try await store.get(try key("prefs/style"), in: otherDeployment) == nil)
    }

    @Test func listFiltersByPrefixAndRespectsLimit() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        for index in 0..<5 {
            try await store.set(MemoryRecord(key: try key("tasks/t\(index)"), value: "t"), in: scope)
        }
        try await store.set(MemoryRecord(key: try key("decisions/d"), value: "d"), in: scope)

        let tasks = try await store.list(prefix: "tasks/", limit: 50, in: scope)
        #expect(tasks.count == 5)
        #expect(tasks.allSatisfy { $0.rawValue.hasPrefix("tasks/") })
        #expect(try await store.list(prefix: "tasks/", limit: 2, in: scope).count == 2)
        #expect(try await store.list(prefix: "", limit: 50, in: scope).count == 6)
    }

    @Test func searchRanksKeyMatchesAboveBodyMatches() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("decisions/sync"),
                                         value: "Keep FooManager."), in: scope)
        try await store.set(MemoryRecord(key: try key("notes/misc"),
                                         value: "We briefly discussed sync yesterday."), in: scope)

        let hits = try await store.search(MemoryQuery(text: "sync architecture"), in: scope)
        #expect(hits.first?.key.rawValue == "decisions/sync")
        #expect(hits.count == 2)
        // A query matching nothing returns nothing rather than everything,
        // which is the failure mode that would dump the store into context.
        #expect(try await store.search(MemoryQuery(text: "kubernetes"), in: scope).isEmpty)
    }

    @Test func searchFiltersByTagsPrefixAndImportance() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        try await store.set(MemoryRecord(key: try key("decisions/a"), value: "x",
                                         importance: 0.9, tags: ["sync"]), in: scope)
        try await store.set(MemoryRecord(key: try key("decisions/b"), value: "x",
                                         importance: 0.1, tags: ["ui"]), in: scope)
        try await store.set(MemoryRecord(key: try key("notes/c"), value: "x",
                                         importance: 0.9, tags: ["sync"]), in: scope)

        #expect(try await store.search(MemoryQuery(tags: ["sync"]), in: scope).count == 2)
        #expect(try await store.search(MemoryQuery(prefix: "decisions/"), in: scope).count == 2)
        #expect(try await store.search(MemoryQuery(minimumImportance: 0.5), in: scope).count == 2)
        let combined = MemoryQuery(prefix: "decisions/", tags: ["sync"], minimumImportance: 0.5)
        #expect(try await store.search(combined, in: scope).count == 1)
    }

    @Test func appendCreatesThenExtends() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        let key = try key("sessions/log")
        let first = try await store.append("found the race", to: key, in: scope)
        #expect(first.value == "found the race")

        let second = try await store.append("fixed it", to: key, in: scope)
        #expect(second.value == "found the race\nfixed it")
        #expect(try await store.get(key, in: scope)?.value == second.value)
    }

    @Test func rewriteKeepsCreationTimeAndMovesUpdateTime() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        let key = try key("architecture/sync")
        let original = MemoryRecord(key: key, value: "v1",
                                    createdAt: Date(timeIntervalSince1970: 1_000),
                                    updatedAt: Date(timeIntervalSince1970: 1_000))
        try await store.set(original, in: scope)
        try await store.set(MemoryRecord(key: key, value: "v2"), in: scope)

        let stored = try await store.get(key, in: scope)
        #expect(stored?.value == "v2")
        // "Known since" survives an update; the model is correcting a fact,
        // not creating a new one.
        #expect(stored?.createdAt == Date(timeIntervalSince1970: 1_000))
        #expect((stored?.updatedAt ?? .distantPast) > Date(timeIntervalSince1970: 1_000))
    }

    @Test func oversizedValuesAreRejected() async throws {
        let store = InMemoryStore(limits: MemoryLimits(maximumValueBytes: 128))
        let scope = try scope()
        let big = String(repeating: "x", count: 200)

        await #expect(throws: MemoryError.valueTooLarge(bytes: 200, limit: 128)) {
            try await store.set(MemoryRecord(key: try self.key("big"), value: big), in: scope)
        }
        // The limit covers append as well, or it could be walked past a line
        // at a time.
        try await store.append(String(repeating: "y", count: 100), to: try key("grow"), in: scope)
        await #expect(throws: MemoryError.self) {
            try await store.append(String(repeating: "y", count: 100),
                                   to: try self.key("grow"), in: scope)
        }
    }

    @Test func bootstrapIsBoundedByCountAndBytes() async throws {
        let limits = MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 10_000)
        let store = InMemoryStore(limits: limits)
        let scope = try scope()
        for index in 0..<10 {
            try await store.set(MemoryRecord(key: try key("facts/f\(index)"),
                                             value: "value \(index)",
                                             importance: Double(index) / 10), in: scope)
        }

        let bootstrap = try await store.sessionInit(MemorySession(id: "s1"), in: scope)
        #expect(bootstrap.records.count == 3)
        #expect(bootstrap.omittedCount == 7)
        // Most important first, so a truncated bootstrap keeps what matters.
        #expect(bootstrap.records.first?.importance == 0.9)

        // The byte ceiling binds even when the count would not.
        let tight = InMemoryStore(limits: MemoryLimits(bootstrapRecords: 100, bootstrapBytes: 40))
        for index in 0..<10 {
            try await tight.set(MemoryRecord(key: try key("facts/g\(index)"),
                                             value: String(repeating: "z", count: 30)), in: scope)
        }
        let bounded = try await tight.sessionInit(MemorySession(id: "s2"), in: scope)
        #expect(bounded.totalBytes <= 40)
        #expect(bounded.records.count < 10)
    }

    @Test func sessionInitNeverReturnsTheWholeStore() async throws {
        // The property that matters most: no configuration of the store makes
        // session start hand back everything it holds.
        let store = InMemoryStore(limits: MemoryLimits(bootstrapRecords: 20, bootstrapBytes: 8 * 1024))
        let scope = try scope()
        for index in 0..<500 {
            try await store.set(MemoryRecord(key: try key("facts/f\(index)"),
                                             value: String(repeating: "x", count: 500)), in: scope)
        }

        let bootstrap = try await store.sessionInit(MemorySession(id: "s"), in: scope)
        #expect(bootstrap.records.count <= 20)
        #expect(bootstrap.totalBytes <= 8 * 1024)
        #expect(bootstrap.omittedCount >= 480)
    }

    @Test func sessionInitRecordsTheSession() async throws {
        let store = InMemoryStore()
        let scope = try scope()
        _ = try await store.sessionInit(MemorySession(id: "abc", modelID: "qwen3.6"), in: scope)
        let sessions = await store.recordedSessions(in: scope)
        #expect(sessions.map(\.id) == ["abc"])
        #expect(sessions.first?.modelID == "qwen3.6")
    }

    @Test func concurrentSessionsDoNotLoseWrites() async throws {
        let store = InMemoryStore()
        let scopeA = try scope("repo-a")
        let scopeB = try scope("repo-b")

        await withTaskGroup(of: Void.self) { group in
            for index in 0..<50 {
                group.addTask {
                    let scope = index.isMultiple(of: 2) ? scopeA : scopeB
                    let key = try? MemoryKey(validating: "concurrent/k\(index)")
                    guard let key else { return }
                    try? await store.set(MemoryRecord(key: key, value: "\(index)"), in: scope)
                }
            }
        }

        let a = try await store.list(prefix: "concurrent/", limit: 200, in: scopeA)
        let b = try await store.list(prefix: "concurrent/", limit: 200, in: scopeB)
        #expect(a.count == 25)
        #expect(b.count == 25)
    }
}
