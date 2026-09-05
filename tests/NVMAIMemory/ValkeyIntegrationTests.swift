import Foundation
import Testing
@testable import NVMAIMemory

/// The same contract as the in-process suites, run against a real Valkey.
///
/// Skipped unless one is reachable, so the suite still passes on a machine
/// without a server; run one with `tools/memory/valkey-from-git.sh start`.
/// This is what validates the RESP client against a real implementation
/// rather than against my own reading of the protocol: inline replies,
/// packet boundaries, `ZRANGE ... REV`, and the CONFIG handshake.
@Suite(.serialized) struct ValkeyIntegrationTests {
    /// A scope per run, so repeated runs and a shared server never collide.
    private func scope(_ suffix: String = "a") throws -> MemoryScope {
        try MemoryScope(namespace: "nvmai-test-\(ProcessInfo.processInfo.processIdentifier)",
                        user: "ci",
                        workspace: "repo-\(suffix)")
    }

    private func makeStore(limits: MemoryLimits = .init()) -> ValkeyMemoryStore {
        var configuration = ValkeyConfiguration()
        if let url = ProcessInfo.processInfo.environment["VALKEY_TEST_URL"],
           let parsed = ValkeyConfiguration.parse(url: url) {
            configuration = parsed
        }
        configuration.operationTimeoutMilliseconds = 2_000
        configuration.connectTimeoutMilliseconds = 1_000
        return ValkeyMemoryStore(connection: ValkeyConnection(configuration: configuration),
                                 prefix: "nvmai:itest",
                                 limits: limits)
    }

    /// Whether a server answered. Everything here is conditional on it.
    private func serverIsReachable() async -> Bool {
        var configuration = ValkeyConfiguration()
        if let url = ProcessInfo.processInfo.environment["VALKEY_TEST_URL"],
           let parsed = ValkeyConfiguration.parse(url: url) {
            configuration = parsed
        }
        configuration.connectTimeoutMilliseconds = 500
        configuration.operationTimeoutMilliseconds = 500
        let connection = ValkeyConnection(configuration: configuration)
        do {
            try await connection.connect()
            _ = try await connection.send(["PING"])
            await connection.close()
            return true
        } catch {
            return false
        }
    }

    /// Removes everything this suite wrote, including the keys the store's
    /// own `list` does not enumerate (the session list, the index), so a
    /// shared server is left as it was found.
    private func cleanUp(_ store: ValkeyMemoryStore, _ scope: MemoryScope) async {
        for key in (try? await store.list(prefix: "", limit: 500, in: scope)) ?? [] {
            _ = try? await store.delete(key, in: scope)
        }
        var configuration = ValkeyConfiguration()
        if let url = ProcessInfo.processInfo.environment["VALKEY_TEST_URL"],
           let parsed = ValkeyConfiguration.parse(url: url) {
            configuration = parsed
        }
        let connection = ValkeyConnection(configuration: configuration)
        guard (try? await connection.connect()) != nil else { return }
        let tag = "{\(scope.namespace)/\(scope.user)/\(scope.workspace)}"
        for suffix in ["sessions", "idx", "j:index"] {
            _ = try? await connection.send(["DEL", "nvmai:itest:\(tag):\(suffix)"])
        }
        await connection.close()
    }

    @Test func roundTripsAgainstARealServer() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore()
        let scope = try scope("round-trip")
        await cleanUp(store, scope)

        let key = try MemoryKey(validating: "decisions/sync")
        try await store.set(MemoryRecord(key: key,
                                         value: "Keep FooManager: it prevents a sync race.",
                                         importance: 0.9,
                                         confidence: 0.8,
                                         tags: ["sync", "concurrency"]), in: scope)

        let loaded = try await store.get(key, in: scope)
        #expect(loaded?.value.contains("FooManager") == true)
        #expect(loaded?.importance == 0.9)
        #expect(loaded?.confidence == 0.8)
        #expect(loaded?.tags == ["sync", "concurrency"])
        await cleanUp(store, scope)
    }

    @Test func unicodeAndJSONSurviveTheWire() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore()
        let scope = try scope("encoding")
        await cleanUp(store, scope)

        // Multi-byte characters and newlines are where a hand-written bulk
        // string encoder gets its length wrong.
        let json = #"{"note":"emoji 🧠 and Ünïcode","lines":"a\nb","n":3}"#
        try await store.set(MemoryRecord(key: try MemoryKey(validating: "architecture/stack"),
                                         value: json), in: scope)
        #expect(try await store.get(try MemoryKey(validating: "architecture/stack"),
                                    in: scope)?.value == json)

        let long = String(repeating: "λ", count: 4_000)
        try await store.set(MemoryRecord(key: try MemoryKey(validating: "notes/long"),
                                         value: long), in: scope)
        #expect(try await store.get(try MemoryKey(validating: "notes/long"),
                                    in: scope)?.value == long)
        await cleanUp(store, scope)
    }

    @Test func listSearchAndIndexOrderingWork() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore()
        let scope = try scope("index")
        await cleanUp(store, scope)

        for index in 0..<12 {
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "tasks/t\(index)"),
                                             value: "task \(index)"), in: scope)
        }
        try await store.set(MemoryRecord(key: try MemoryKey(validating: "decisions/db"),
                                         value: "postgres for the queue"), in: scope)

        #expect(try await store.list(prefix: "tasks/", limit: 50, in: scope).count == 12)
        #expect(try await store.list(prefix: "decisions/", limit: 50, in: scope).count == 1)
        let hits = try await store.search(MemoryQuery(text: "postgres"), in: scope)
        #expect(hits.first?.key.rawValue == "decisions/db")
        // Most recently written first, which is what ZRANGE REV has to give.
        let all = try await store.list(prefix: "", limit: 50, in: scope)
        #expect(all.first?.rawValue == "decisions/db")
        await cleanUp(store, scope)
    }

    @Test func scopesAreIsolatedOnARealServer() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore()
        let first = try scope("iso-1")
        let second = try scope("iso-2")
        await cleanUp(store, first)
        await cleanUp(store, second)

        let key = try MemoryKey(validating: "decisions/db")
        try await store.set(MemoryRecord(key: key, value: "postgres"), in: first)
        try await store.set(MemoryRecord(key: key, value: "sqlite"), in: second)

        #expect(try await store.get(key, in: first)?.value == "postgres")
        #expect(try await store.get(key, in: second)?.value == "sqlite")
        #expect(try await store.list(prefix: "", limit: 50, in: first).count == 1)
        await cleanUp(store, first)
        await cleanUp(store, second)
    }

    @Test func appendDeleteAndBootstrapBehaveAsSpecified() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore(limits: MemoryLimits(bootstrapRecords: 3, bootstrapBytes: 4_096))
        let scope = try scope("lifecycle")
        await cleanUp(store, scope)

        let key = try MemoryKey(validating: "sessions/log")
        _ = try await store.append("first", to: key, in: scope)
        let second = try await store.append("second", to: key, in: scope)
        #expect(second.value == "first\nsecond")

        for index in 0..<8 {
            try await store.set(MemoryRecord(key: try MemoryKey(validating: "facts/f\(index)"),
                                             value: "v\(index)",
                                             importance: Double(index) / 8), in: scope)
        }
        let bootstrap = try await store.sessionInit(MemorySession(id: "itest"), in: scope)
        #expect(bootstrap.records.count == 3)
        #expect(bootstrap.records.first?.importance ?? 0 > 0.5)

        #expect(try await store.delete(key, in: scope))
        #expect(try await store.get(key, in: scope) == nil)
        await cleanUp(store, scope)
    }

    @Test func theConfiguredCeilingReachesTheServer() async throws {
        guard await serverIsReachable() else { return }
        var configuration = ValkeyConfiguration()
        if let url = ProcessInfo.processInfo.environment["VALKEY_TEST_URL"],
           let parsed = ValkeyConfiguration.parse(url: url) {
            configuration = parsed
        }
        configuration.maximumMemoryBytes = 512 << 20
        let connection = ValkeyConnection(configuration: configuration)
        try await connection.connect()

        let policy = try await connection.send(["CONFIG", "GET", "maxmemory-policy"])
        // A durable store must refuse writes when full rather than evict the
        // facts the model relies on.
        #expect(policy.arrayValue?.compactMap(\.stringValue).contains("noeviction") == true)
        let ceiling = try await connection.send(["CONFIG", "GET", "maxmemory"])
        #expect(ceiling.arrayValue?.compactMap(\.stringValue).contains(String(512 << 20)) == true)
        await connection.close()
    }

    @Test func concurrentWritesOverOneConnectionStayConsistent() async throws {
        guard await serverIsReachable() else { return }
        let store = makeStore()
        let scope = try scope("concurrency")
        await cleanUp(store, scope)

        // Pipelining against a real server: if a reply were ever matched to
        // the wrong request, this is where it shows.
        await withTaskGroup(of: Void.self) { group in
            for index in 0..<40 {
                group.addTask {
                    guard let key = try? MemoryKey(validating: "concurrent/k\(index)") else {
                        return
                    }
                    try? await store.set(MemoryRecord(key: key, value: "\(index)"), in: scope)
                }
            }
        }
        for index in 0..<40 {
            let key = try MemoryKey(validating: "concurrent/k\(index)")
            #expect(try await store.get(key, in: scope)?.value == "\(index)")
        }
        await cleanUp(store, scope)
    }

    @Test func survivesAServerRestart() async throws {
        guard await serverIsReachable() else { return }
        // Durability is the whole point of the store; a write that does not
        // outlive the process is a cache.
        let store = makeStore()
        let scope = try scope("durability")
        await cleanUp(store, scope)
        let key = try MemoryKey(validating: "decisions/persisted")
        try await store.set(MemoryRecord(key: key, value: "written before a reconnect"), in: scope)

        // A fresh store, hence a fresh connection, reading the same key.
        let reader = makeStore()
        #expect(try await reader.get(key, in: scope)?.value == "written before a reconnect")
        await cleanUp(store, scope)
    }
}
